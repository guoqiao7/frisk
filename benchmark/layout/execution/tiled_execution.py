"""Independent-CTA lift of a checked, fixed-shape Frisk tile program.

The tile coordinate is a separate affine address dimension: CTA b uses
base_i + b * stride_i. No logical intra-tile layout, arithmetic or barrier
changes. This is a batch of independent tiles, not long-sequence Attention.
Importing this module neither initializes CUDA nor starts compilation.
"""
import copy
import hashlib
import re

_VERSION = 'independent-tile-cta-v1'
_MAX_U64 = (1 << 64) - 1
_MAX_U32 = (1 << 32) - 1


def _integer(value, lower, upper, name):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(name + ' outside checked integer domain')
    return value


def _contract(info):
    contract = info.get('tiled_execution')
    fields = {'version', 'tile_count', 'grid', 'threads', 'root_bytes',
              'root_stride_bytes', 'descriptor_stride_bytes', 'single_tile_tt_sha256',
              'semantics'}
    if type(contract) is not dict or set(contract) != fields or contract['version'] != _VERSION:
        raise ValueError('unknown independent-tile execution contract')
    count = _integer(contract['tile_count'], 1, (1 << 31) - 1, 'tile_count')
    if type(contract['grid']) is not list or any(type(x) is not int for x in contract['grid']) or contract['grid'] != [count, 1, 1] or type(contract['threads']) is not int or contract['threads'] != 128:
        raise ValueError('independent-tile grid/thread mismatch')
    if contract['descriptor_stride_bytes'] != 128 or contract['semantics'] != 'independent fixed-shape tiles; no cross-CTA communication':
        raise ValueError('unknown independent-tile addressing semantics')
    if not re.fullmatch('[0-9a-f]{64}', contract['single_tile_tt_sha256']):
        raise ValueError('missing original emission identity')
    spans, strides = contract['root_bytes'], contract['root_stride_bytes']
    aligns = info.get('global_alignments')
    if any(type(x) is not list or len(x) != 4 for x in (spans, strides, aligns)):
        raise ValueError('four Global span/stride/alignment entries required')
    for span, stride, align in zip(spans, strides, aligns):
        _integer(span, 1, _MAX_U32, 'root_bytes')
        _integer(stride, span, _MAX_U32, 'root_stride_bytes')
        _integer(align, 1, _MAX_U32, 'alignment')
        if align & (align - 1) or stride % align:
            raise ValueError('each CTA must preserve declared root alignment')
        if (count - 1) * stride + span > _MAX_U64:
            raise ValueError('independent-tile allocation size overflow')
    return contract


def _lift_text(text, strides, descriptor_count):
    """Fail closed on any change to the backend's single-inline-asm envelope."""
    if text.count('tt.elementwise_inline_asm ') != 1 or text.count('tt.func public @frisk_execution(') != 1:
        raise ValueError('expected the checked single-kernel inline-asm envelope')
    match = re.search(r'tt\.elementwise_inline_asm "([^\"]*)" \{constraints = "(=r(?:,l)+)"', text)
    if not match or match.group(2) != '=r' + ',l' * (4 + descriptor_count):
        raise ValueError('unknown checked inline-asm pointer ABI')
    body = match.group(1)
    if not body.startswith('{\\0A') or not body.endswith('\\0A}') or '%ctaid.' in body or 'frisk_tile_' in body:
        raise ValueError('unknown or already tiled inline-asm body')
    maximum = 4 + descriptor_count
    if any(int(x) > maximum for x in re.findall(r'\$(\d+)', body)):
        raise ValueError('inline-asm operand outside checked pointer ABI')
    # Replace operands before adding the prologue, whose sources must stay $N.
    transformed = re.sub(r'\$(\d+)\b', lambda m: m.group(0) if m.group(1) == '0' else 'frisk_tile_base' + m.group(1), body[4:])
    prologue = ['.reg .u32 frisk_tile_cta;', 'mov.u32 frisk_tile_cta, %ctaid.x;']
    declarations = []
    instructions = []
    for operand, stride in enumerate(list(strides) + [128] * descriptor_count, 1):
        name = 'frisk_tile_base' + str(operand)
        declarations.append('.reg .u64 ' + name + ';')
        instructions += [f'mul.wide.u32 {name}, frisk_tile_cta, {stride};',
                         f'add.u64 {name}, ${operand}, {name};']
    prologue = declarations + prologue + instructions
    lifted = '{\\0A' + '\\0A'.join(prologue) + '\\0A' + transformed
    return text[:match.start(1)] + lifted + text[match.end(1):]


def lift_emission(data, text, info, *, tile_count, root_stride_bytes=None):
    """Lift an emission already validated by backend.emit; independently check roots.

    This API is intended for callers that retain the exact output of backend.emit.
    A caller with an untrusted text/config pair should use emit_tiled instead.
    """
    _integer(tile_count, 1, (1 << 31) - 1, 'tile_count')
    if 'tiled_execution' in info:
        raise ValueError('emission is already tiled')
    from storage_addresses import global_root_spans
    roots = info.get('root_order')
    if type(roots) is not list or len(roots) != 4 or len(set(roots)) != 4:
        raise ValueError('four distinct declared Global roots required')
    spans = global_root_spans(data, roots)
    aligns = info.get('global_alignments')
    if type(aligns) is not list or len(aligns) != 4 or any(type(a) is not int or a < 1 or a & (a-1) for a in aligns):
        raise ValueError('invalid root alignment declaration')
    strides = [((span + align - 1) // align) * align for span, align in zip(spans, aligns)] if root_stride_bytes is None else copy.deepcopy(root_stride_bytes)
    result = copy.deepcopy(info)
    result['tiled_execution'] = dict(version=_VERSION, tile_count=tile_count,
        grid=[tile_count, 1, 1], threads=128, root_bytes=spans, root_stride_bytes=strides,
        descriptor_stride_bytes=128, single_tile_tt_sha256=hashlib.sha256(text.encode()).hexdigest(),
        semantics='independent fixed-shape tiles; no cross-CTA communication')
    _contract(result)
    return _lift_text(text, strides, len(info['tma_plans'])), result


def emit_tiled(data, *, tile_count, root_stride_bytes=None, **emit_options):
    from backend import emit
    text, info = emit(data, **emit_options)
    return lift_emission(data, text, info, tile_count=tile_count, root_stride_bytes=root_stride_bytes)


def validate_tiled_launch(info, pointers, buffer_bytes, *, grid):
    """Validate complete allocated envelopes before passing pointers to CUDA.

    buffer_bytes are available allocation bytes beginning at each pointer. Read-only
    inputs may overlap each other; the complete output envelope must be disjoint.
    """
    from async_host import validate_launch
    contract = _contract(info)
    if type(grid) is not list or any(type(x) is not int for x in grid) or grid != contract['grid']:
        raise ValueError('launch grid differs from the compiled tile contract')
    if type(pointers) is not list or type(buffer_bytes) is not list or len(pointers) != 4 or len(buffer_bytes) != 4:
        raise ValueError('four Global pointer/available-size entries required')
    validate_launch(info, pointers)
    ranges = []
    count = contract['tile_count']
    for pointer, available, span, stride in zip(pointers, buffer_bytes, contract['root_bytes'], contract['root_stride_bytes']):
        _integer(pointer, 1, _MAX_U64, 'pointer')
        _integer(available, 1, _MAX_U64, 'buffer_bytes')
        required = (count - 1) * stride + span
        if required > available or available > _MAX_U64 - pointer:
            raise ValueError('insufficient allocation or uint64 pointer-range overflow')
        ranges.append((pointer, pointer + required))
    begin, end = ranges[3]
    if any(begin < hi and lo < end for lo, hi in ranges[:3]):
        raise ValueError('output tile envelope overlaps an input envelope')
    return dict(grid=list(grid), threads=128, required_bytes=[hi-lo for lo,hi in ranges],
                output_disjoint=True, independent_tiles=count)


class TiledTensorMaps:
    """Own every per-tile encoding and contiguous per-plan 128-byte device array."""
    def __init__(self, info, pointers, buffer_bytes, *, grid):
        validate_tiled_launch(info, pointers, buffer_bytes, grid=grid)
        from async_host import TensorMaps
        import torch
        contract = _contract(info)
        count, strides = contract['tile_count'], contract['root_stride_bytes']
        self.owners = [TensorMaps(info, [p + tile*s for p, s in zip(pointers, strides)]) for tile in range(count)]
        self.device = [torch.cat([owner.device[plan] for owner in self.owners]) for plan in range(len(info['tma_plans']))]
        if any(x.data_ptr() % 64 or x.numel() != 128 * count or not x.is_contiguous() for x in self.device):
            raise ValueError('invalid per-CTA descriptor array alignment or extent')
        self.setup_ns = [value for owner in self.owners for value in owner.setup_ns]
