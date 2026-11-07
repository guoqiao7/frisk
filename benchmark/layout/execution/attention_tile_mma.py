"""Native SM90 WGMMA witnesses for 64-query D64 Attention tiles."""

from dataclasses import dataclass
from functools import lru_cache

from access_plan import linear
from native_static_support import STATIC_SHARED_LIMIT_BYTES


TILE = 64
THREADS = 128
FRAGMENT_SLOTS = 32
SHARED_ALIGNMENT = 1024
ROOT_BYTES = TILE * TILE * 2
_ATOM_K = 16
_SLICES = TILE // _ATOM_K


def fragment_coordinate(thread, register, *, kv_tile=TILE):
    """Native m64nNk16 accumulator coordinate; N=64 is also the RS P carrier."""
    if (type(kv_tile) is not int or kv_tile not in (8, 16, 32, TILE, 2 * TILE, 4 * TILE)
            or type(thread) is not int or not 0 <= thread < THREADS
            or type(register) is not int or not 0 <= register < kv_tile // 2):
        raise ValueError("coordinate outside the D64 native fragment")
    warp, lane = divmod(thread, 32)
    row = warp * 16 + lane // 4 + (register % 4 // 2) * 8
    column = (lane % 4) * 2 + register % 2 + (register // 4) * 8
    return row, column


@lru_cache(maxsize=1)
def _fragment_columns():
    table = []
    for thread in range(THREADS):
        warp, lane = divmod(thread, 32)
        for slot in range(FRAGMENT_SLOTS):
            row, column = fragment_coordinate(thread, slot)
            atom, within_atom = divmod(slot, 8)
            pair, half = divmod(within_atom, 2)
            rs_row = warp * 16 + lane // 4 + (pair % 2) * 8
            rs_column = lane % 4 * 2 + half + (pair // 2) * 8 + atom * _ATOM_K
            if (row, column) != (rs_row, rs_column):
                raise ValueError("QK and packed PV native fragment coordinates disagree")
            table.append(row * TILE + column)
    origin, columns = linear(table)
    if origin or set(table) != set(range(TILE * TILE)):
        raise ValueError("D64 native fragment is not a GF(2) bijection")
    # The input bits are [register (5), lane (5), warp (2)] and the output
    # bits are [column (6), row (6)], both ordered least significant first.
    return tuple(columns)


@lru_cache(maxsize=2)
def _wide_fragment_columns(kv_tile=2 * TILE):
    # PTX ISA m64nNk16 accumulator fragments repeat the same eight-column
    # group. Lift row bits by the selected key-width increase and use the new
    # high register bits to select another 64-column carrier.
    if kv_tile not in (2 * TILE, 4 * TILE):
        raise ValueError("D64 wide fragment requires KV128 or KV256")
    width = kv_tile
    table = []
    for thread in range(THREADS):
        for slot in range(width // 2):
            row, column = fragment_coordinate(thread, slot, kv_tile=width)
            half, narrow_slot = divmod(slot, FRAGMENT_SLOTS)
            narrow_row, narrow_column = fragment_coordinate(thread, narrow_slot)
            if (row, column) != (narrow_row, narrow_column + half * TILE):
                raise ValueError("D64 wide fragment does not project onto its native halves")
            table.append(row * width + column)
    origin, columns = linear(table)
    extra_bits = (width // TILE).bit_length() - 1
    lifted = tuple((basis & (TILE - 1)) | ((basis & ~(TILE - 1)) << extra_bits)
                   for basis in _fragment_columns())
    register_bits = FRAGMENT_SLOTS.bit_length() - 1
    expected = (lifted[:register_bits] + tuple(TILE << i for i in range(extra_bits))
                + lifted[register_bits:])
    if origin or tuple(columns) != expected or set(table) != set(range(TILE * width)):
        raise ValueError("D64 wide fragment is not the checked GF(2) direct sum")
    return tuple(columns)


@lru_cache(maxsize=3)
def _narrow_fragment_columns(kv_tile):
    if kv_tile not in (8, 16, 32):
        raise ValueError("D64 prefix fragment requires KV8, KV16 or KV32")
    slots = kv_tile // 2
    table = []
    for thread in range(THREADS):
        for register in range(slots):
            row, column = fragment_coordinate(thread, register, kv_tile=kv_tile)
            base_row, base_column = fragment_coordinate(thread, register)
            if (row, column) != (base_row, base_column) or column >= kv_tile:
                raise ValueError("D64 prefix does not preserve native register owners")
            table.append(row * kv_tile + column)
    origin, columns = linear(table)
    removed_bits = (TILE // kv_tile).bit_length() - 1
    register_bits = slots.bit_length() - 1
    original = _fragment_columns()
    selected = original[:register_bits] + original[FRAGMENT_SLOTS.bit_length() - 1:]
    expected = tuple((basis & (kv_tile - 1)) | ((basis >> 6) << (6 - removed_bits))
                     for basis in selected)
    if origin or tuple(columns) != expected or set(table) != set(range(TILE * kv_tile)):
        raise ValueError("D64 prefix is not the checked GF(2) restriction")
    return tuple(columns)


def score_width_for_prefix(valid_columns):
    if type(valid_columns) is not int or not 1 <= valid_columns <= TILE:
        raise ValueError("D64 native score prefix must contain one to 64 keys")
    return max(8, 1 << (valid_columns - 1).bit_length())


def _swizzle128(offset):
    return offset ^ (((offset >> 7) & 7) << 4)


@lru_cache(maxsize=1)
def shared_offsets_64x64():
    """Root-relative halfword offsets of the common Q/K/V SW128 carrier."""
    table = tuple(_swizzle128(index * 2) for index in range(TILE * TILE))
    linear(table)
    if set(table) != set(range(0, ROOT_BYTES, 2)):
        raise ValueError("D64 Shared layout must own every halfword exactly once")
    for first in range(0, len(table), 8):
        begin = table[first]
        if begin % 16 or table[first:first + 8] != tuple(range(begin, begin + 16, 2)):
            raise ValueError("D64 Shared layout splits a 16-byte payload")
    return table


@dataclass(frozen=True)
class _Descriptor:
    # The lower 14 address bits of the encoded descriptor are in 16B units.
    offset: int
    fields: int

    def emit(self, emit, tmp, smem):
        address = tmp("b32")
        descriptor = tmp("b64")
        emit(f"add.u32 {address}, {smem}, {self.offset};")
        emit(f"shr.u32 {address}, {address}, 4;")
        emit(f"cvt.u64.u32 {descriptor}, {address};")
        emit(f"or.b64 {descriptor}, {descriptor}, {self.fields};")
        return descriptor


def _checked_descriptors(root, *, k_major, mn_extent=TILE, k_extent=TILE):
    layout = shared_offsets_64x64()
    if (type(mn_extent) is not int or type(k_extent) is not int
            or k_extent not in (TILE // 2, TILE)):
        raise ValueError("D64 descriptor extent is not a supported carrier")
    if k_major:
        if mn_extent not in (8, 16, 32, TILE, 2 * TILE, 4 * TILE) or k_extent != TILE:
            raise ValueError("D64 K-major descriptor requires native key width and D64")
        if mn_extent > TILE:
            layout = tuple(block * ROOT_BYTES + offset
                           for block in range(mn_extent // TILE) for offset in layout)
        else:
            layout = layout[:mn_extent * TILE]
    elif mn_extent == TILE:
        layout = layout[:k_extent * TILE]
    else:
        raise ValueError("D64 V descriptor requires a 64-column output")
    extent_bytes = len(layout) * 2
    if not 0 <= root <= (1 << 18) - extent_bytes:
        raise ValueError("D64 descriptor payload exceeds its address field")
    leading, stride = (16, 1024) if k_major else (1024, 1024)
    descriptors = []
    for start_k in range(0, k_extent, _ATOM_K):
        start = layout[start_k if k_major else start_k * TILE]
        phase = (start >> 7) & 7
        if start % 16 or root + start >= 1 << 18:
            raise ValueError("D64 descriptor start exceeds its aligned address field")
        for mn in range(mn_extent):
            for k in range(_ATOM_K):
                if k_major:
                    relative = mn % 8 * 128 + mn // 8 * stride + k % 8 * 2 + k // 8 * 16
                    logical = mn * TILE + start_k + k
                else:
                    relative = mn * 2 + k % 8 * 128 + k // 8 * stride
                    logical = (start_k + k) * TILE + mn
                unswizzled = (start ^ (phase << 4)) + relative
                expected = _swizzle128(unswizzled)
                if expected != layout[logical] or not 0 <= expected <= extent_bytes - 2:
                    raise ValueError("D64 descriptor differs from its checked Shared halfword")
        fields = (leading // 16 << 16) | (stride // 16 << 32) | (phase << 49) | (1 << 62)
        descriptors.append(_Descriptor(root + start, fields))
    return tuple(descriptors)


def _register_group(registers, count):
    group = tuple(registers)
    if (len(group) != count or any(type(name) is not str or not name for name in group)
            or len(set(group)) != count):
        raise ValueError("wrong number of distinct native registers for D64 WGMMA")
    return group


@dataclass(frozen=True)
class D64TileMma:
    q_offset: int
    k_offset: int
    v_offset: int
    shared_bytes: int
    fragment_columns: tuple[int, ...]
    _q: tuple[_Descriptor, ...]
    _k: tuple[_Descriptor, ...]
    _v: tuple[_Descriptor, ...]
    _wide_k: tuple[_Descriptor, ...] | None = None
    kv_tile: int = TILE

    @property
    def score_fragment_columns(self):
        """GF(2) basis of the configured QK carrier; PV stays on fragment_columns."""
        if self.kv_tile < TILE:
            return _narrow_fragment_columns(self.kv_tile)
        return _wide_fragment_columns(self.kv_tile) if self._wide_k is not None else self.fragment_columns

    def _emit_group(self, emit, tmp, outputs, args):
        # The caller publishes Shared after the copy and supplies ready
        # accumulator/RS registers. This fence orders those generic accesses.
        # Wait zero before a generic result use or overwrite of Shared.
        acc = tmp("pred")
        emit(f"setp.eq.u32 {acc}, 0, 0;")
        emit("wgmma.fence.sync.aligned;")
        fragment = "{" + ", ".join(outputs) + "}"
        width = len(outputs) * 2
        for operand_a, operand_b, majors in args:
            emit(f"wgmma.mma_async.sync.aligned.m64n{width}k16.f32.f16.f16 {fragment}, "
                 f"{operand_a}, {operand_b}, {acc}, 1, 1, {majors};")
        emit("wgmma.commit_group.sync.aligned;")
        emit("wgmma.wait_group.sync.aligned 0;")

    def emit_qk(self, emit, tmp, smem, score_regs):
        """Accumulate ascending depth slices into caller-zeroed FP32 scores."""
        scores = tuple(score_regs)
        if self._wide_k is not None and len(scores) == self.kv_tile // 2:
            self._emit_qk(emit, tmp, smem, _register_group(scores, self.kv_tile // 2), self._wide_k)
        elif self.kv_tile < TILE:
            self._emit_qk(emit, tmp, smem, _register_group(scores, self.kv_tile // 2), self._k)
        else:
            self.emit_qk_half(emit, tmp, smem, scores)

    def emit_qk_half(self, emit, tmp, smem, score_regs):
        """Use K0 alone for a single full or masked 64-key tile."""
        if self.kv_tile < TILE:
            raise ValueError("compact KV32 has no complete K64 Shared root")
        self._emit_qk(emit, tmp, smem, _register_group(score_regs, FRAGMENT_SLOTS), self._k)

    def emit_qk_prefix(self, emit, tmp, smem, score_regs, valid_columns):
        """Read the smallest native K0 prefix, with caller-proved zero fill."""
        width = score_width_for_prefix(valid_columns)
        if width > min(self.kv_tile, TILE):
            raise ValueError("score prefix exceeds the configured K0 carrier")
        if width < TILE:
            _narrow_fragment_columns(width)
        descriptors = _checked_descriptors(self.k_offset, k_major=True, mn_extent=width)
        self._emit_qk(emit, tmp, smem, _register_group(score_regs, width // 2), descriptors)

    def _emit_qk(self, emit, tmp, smem, outputs, key):
        args = []
        for a, b in zip(self._q, key, strict=True):
            q = a.emit(emit, tmp, smem)
            k = b.emit(emit, tmp, smem)
            args.append((q, k, "0, 0"))
        self._emit_group(emit, tmp, outputs, args)

    def emit_pv(self, emit, tmp, smem, packed_probs, output_regs):
        """Consume native low/high FP16 packs in ascending KV slice order."""
        outputs = _register_group(output_regs, FRAGMENT_SLOTS)
        packed = _register_group(packed_probs, min(self.kv_tile, TILE) // 4)
        if set(outputs).intersection(packed):
            raise ValueError("D64 packed input may not alias its FP32 accumulator")
        args = []
        for atom, b in enumerate(self._v):
            v = b.emit(emit, tmp, smem)
            source = ", ".join(packed[atom * 4:(atom + 1) * 4])
            args.append(("{" + source + "}", v, "1"))
        self._emit_group(emit, tmp, outputs, args)


def build_d64_wgmma_tiles(shared_offsets=None, *, kv_tile=64):
    """Check a four-warp QK carrier and its Shared operand lifetimes.

    Wide QK reads adjacent K64 roots. The caller must wait for all QK
    operations before overwriting the last K64 with V and publishing it.
    The 64-key fallback reads only K0 and may publish V concurrently.
    Compact KV32 owns disjoint 4 KiB K/V roots and has no K64 fallback.
    """
    if type(kv_tile) is not int or kv_tile not in (TILE // 2, TILE, 2 * TILE, 4 * TILE):
        raise ValueError("D64 WGMMA requires a 32-, 64-, 128- or 256-key carrier")
    wide = kv_tile > TILE
    key_bytes = kv_tile * TILE * 2
    value_bytes = min(kv_tile, TILE) * TILE * 2
    if shared_offsets is None:
        v = key_bytes if wide else ROOT_BYTES + key_bytes
        shared_offsets = dict(q=0, k=ROOT_BYTES, v=v)
    if type(shared_offsets) is not dict or set(shared_offsets) != {"q", "k", "v"}:
        raise ValueError("expected named Q/K/V Shared roots")
    q, k, v = (shared_offsets[name] for name in ("q", "k", "v"))
    if any(type(offset) is not int or offset < 0 or offset % SHARED_ALIGNMENT
           for offset in (q, k, v)):
        raise ValueError("D64 SW128 roots require 1024-byte alignment")
    if wide and v != k + key_bytes - ROOT_BYTES:
        raise ValueError("D64 wide schedule requires V to alias its last K64 root")
    spans = [(q, q + ROOT_BYTES), (k, k + key_bytes)]
    if not wide:
        spans.append((v, v + value_bytes))
    spans.sort()
    shared_bytes = spans[-1][1]
    if (any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
            or shared_bytes > STATIC_SHARED_LIMIT_BYTES):
        raise ValueError("D64 tile Shared roots overlap or exceed the static allocation domain")
    wide_key = _checked_descriptors(k, k_major=True, mn_extent=kv_tile) if wide else None
    if wide:
        _wide_fragment_columns(kv_tile)
    elif kv_tile < TILE:
        _narrow_fragment_columns(kv_tile)
    return D64TileMma(q, k, v, shared_bytes, _fragment_columns(),
                      _checked_descriptors(q, k_major=True),
                      _checked_descriptors(k, k_major=True, mn_extent=min(kv_tile, TILE)),
                      _checked_descriptors(v, k_major=False, k_extent=min(kv_tile, TILE)),
                      wide_key, kv_tile)
