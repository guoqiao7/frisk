"""Native SM90 WGMMA witnesses for a 64-query, 64-key, D64 Attention tile."""

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


def fragment_coordinate(thread, register):
    """Native m64n64k16 accumulator coordinate, also the RS P carrier."""
    if (type(thread) is not int or not 0 <= thread < THREADS
            or type(register) is not int or not 0 <= register < FRAGMENT_SLOTS):
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


def _checked_descriptors(root, *, k_major):
    layout = shared_offsets_64x64()
    leading, stride = (16, 1024) if k_major else (1024, 1024)
    descriptors = []
    for start_k in range(0, TILE, _ATOM_K):
        start = layout[start_k if k_major else start_k * TILE]
        phase = (start >> 7) & 7
        if start % 16 or root + start >= 1 << 18:
            raise ValueError("D64 descriptor start exceeds its aligned address field")
        for mn in range(TILE):
            for k in range(_ATOM_K):
                if k_major:
                    relative = mn % 8 * 128 + mn // 8 * stride + k % 8 * 2 + k // 8 * 16
                    logical = mn * TILE + start_k + k
                else:
                    relative = mn * 2 + k % 8 * 128 + k // 8 * stride
                    logical = (start_k + k) * TILE + mn
                unswizzled = (start ^ (phase << 4)) + relative
                expected = _swizzle128(unswizzled)
                if expected != layout[logical] or not 0 <= expected <= ROOT_BYTES - 2:
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

    def _emit_group(self, emit, tmp, outputs, args):
        # The caller publishes Shared after the copy and supplies ready
        # accumulator/RS registers. This fence orders those generic accesses.
        # Wait zero before a generic result use or overwrite of Shared.
        acc = tmp("pred")
        emit(f"setp.eq.u32 {acc}, 0, 0;")
        emit("wgmma.fence.sync.aligned;")
        fragment = "{" + ", ".join(outputs) + "}"
        for operand_a, operand_b, majors in args:
            emit(f"wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {fragment}, "
                 f"{operand_a}, {operand_b}, {acc}, 1, 1, {majors};")
        emit("wgmma.commit_group.sync.aligned;")
        emit("wgmma.wait_group.sync.aligned 0;")

    def emit_qk(self, emit, tmp, smem, score_regs):
        """Accumulate ascending depth slices into caller-zeroed FP32 scores."""
        outputs = _register_group(score_regs, FRAGMENT_SLOTS)
        args = []
        for a, b in zip(self._q, self._k, strict=True):
            q = a.emit(emit, tmp, smem)
            k = b.emit(emit, tmp, smem)
            args.append((q, k, "0, 0"))
        self._emit_group(emit, tmp, outputs, args)

    def emit_pv(self, emit, tmp, smem, packed_probs, output_regs):
        """Consume sixteen low/high FP16 packs in ascending KV slice order."""
        outputs = _register_group(output_regs, FRAGMENT_SLOTS)
        packed = _register_group(packed_probs, FRAGMENT_SLOTS // 2)
        if set(outputs).intersection(packed):
            raise ValueError("D64 packed input may not alias its FP32 accumulator")
        args = []
        for atom, b in enumerate(self._v):
            v = b.emit(emit, tmp, smem)
            source = ", ".join(packed[atom * 4:(atom + 1) * 4])
            args.append(("{" + source + "}", v, "1"))
        self._emit_group(emit, tmp, outputs, args)


def build_d64_wgmma_tiles(shared_offsets=None, *, kv_tile=64):
    """Check one fixed four-warp native carrier and three disjoint Shared roots."""
    if type(kv_tile) is not int or kv_tile != TILE:
        raise ValueError("D64 WGMMA requires one 64-key carrier")
    if shared_offsets is None:
        shared_offsets = dict(q=0, k=ROOT_BYTES, v=ROOT_BYTES * 2)
    if type(shared_offsets) is not dict or set(shared_offsets) != {"q", "k", "v"}:
        raise ValueError("expected named Q/K/V Shared roots")
    q, k, v = (shared_offsets[name] for name in ("q", "k", "v"))
    if any(type(offset) is not int or offset < 0 or offset % SHARED_ALIGNMENT
           for offset in (q, k, v)):
        raise ValueError("D64 SW128 roots require 1024-byte alignment")
    starts = sorted((q, k, v))
    if (any(a + ROOT_BYTES > b for a, b in zip(starts, starts[1:]))
            or starts[-1] + ROOT_BYTES > STATIC_SHARED_LIMIT_BYTES):
        raise ValueError("D64 tile Shared roots overlap or exceed the static allocation domain")
    return D64TileMma(q, k, v, starts[-1] + ROOT_BYTES, _fragment_columns(),
                      _checked_descriptors(q, k_major=True),
                      _checked_descriptors(k, k_major=True),
                      _checked_descriptors(v, k_major=False))
