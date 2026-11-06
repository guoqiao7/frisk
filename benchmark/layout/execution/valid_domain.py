"""Checked affine activity and memory boundaries for padded SM90 matrix tiles.

The GF(2) carrier stays rectangular and complete. Validity selects a prefix of
logical coordinates and does not alter the carrier's payload or first owners.
The copy plan is for full 16-byte groups of f16 elements and a padded Shared
destination. Global side effects require the activity predicate. Inactive
groups write zero only to their checked Shared intervals.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
import math

from access_plan import linear, plan_copy_owners


_THREADS = 128
_MAX_SLOTS = 65536
_MAX_BYTES = 1 << 32


def _shape(value):
    if type(value) not in (tuple, list) or len(value) != 2:
        raise ValueError('expected a rank-two matrix shape')
    if any(type(extent) is not int or not 0 <= extent <= _MAX_SLOTS for extent in value):
        raise ValueError('matrix extent outside checked domain')
    return tuple(value)


def _prefix_blocks(extent):
    """Yield disjoint (origin, free low bits) blocks for [0, extent)."""
    origin = 0
    while extent:
        width = extent.bit_length() - 1
        yield origin, width
        size = 1 << width
        origin += size
        extent -= size


@dataclass(frozen=True)
class AffineCoset:
    origin: int
    basis: tuple[int, ...]

    @property
    def size(self):
        return 1 << len(self.basis)


def _solve_affine(rows, rhs, width):
    pivots = {}
    for row, bit in zip(rows, rhs):
        while row:
            pivot = row & -row
            if pivot not in pivots:
                pivots[pivot] = row, bit
                break
            previous, previous_bit = pivots[pivot]
            row ^= previous
            bit ^= previous_bit
        else:
            if bit:
                return None

    def extend(free, translate):
        for pivot in sorted(pivots, reverse=True):
            row, bit = pivots[pivot]
            if ((row & free).bit_count() & 1) != (bit if translate else 0):
                free |= pivot
        return free

    basis = tuple(extend(1 << bit, False) for bit in range(width) if 1 << bit not in pivots)
    return AffineCoset(extend(0, True), basis)


@dataclass(frozen=True)
class PrefixDomain:
    carrier_shape: tuple[int, int]
    valid_shape: tuple[int, int]

    def __post_init__(self):
        carrier, valid = _shape(self.carrier_shape), _shape(self.valid_shape)
        if (any(not extent or extent & (extent - 1) for extent in carrier)
                or math.prod(carrier) > _MAX_SLOTS
                or any(live > full for live, full in zip(valid, carrier))):
            raise ValueError('invalid carrier or logical prefix')
        object.__setattr__(self, 'carrier_shape', carrier)
        object.__setattr__(self, 'valid_shape', valid)

    def contains(self, slot):
        rows, cols = self.carrier_shape
        if type(slot) is not int or not 0 <= slot < rows * cols:
            return False
        row, col = divmod(slot, cols)
        return row < self.valid_shape[0] and col < self.valid_shape[1]

    def affine_pullback(self, base, columns):
        """Return disjoint cosets of hardware indices mapped into this prefix."""
        volume = math.prod(self.carrier_shape)
        if (type(base) is not int or not 0 <= base < volume
                or type(columns) not in (tuple, list) or len(columns) > 16
                or any(type(col) is not int or not 0 <= col < volume for col in columns)):
            raise ValueError('invalid bounded affine map for valid prefix')
        col_bits = self.carrier_shape[1].bit_length() - 1
        logical_bits = volume.bit_length() - 1
        rows = tuple(sum(((column >> bit) & 1) << i for i, column in enumerate(columns))
                     for bit in range(logical_bits))
        row_blocks = _prefix_blocks(self.valid_shape[0])
        col_blocks = _prefix_blocks(self.valid_shape[1])
        result = []
        for (row, row_free), (col, col_free) in product(row_blocks, col_blocks):
            free_mask = (((1 << row_free) - 1) << col_bits) | ((1 << col_free) - 1)
            fixed = tuple(bit for bit in range(logical_bits) if not (free_mask >> bit & 1))
            target = (row << col_bits | col) ^ base
            coset = _solve_affine((rows[bit] for bit in fixed),
                                  ((target >> bit) & 1 for bit in fixed), len(columns))
            if coset is not None:
                result.append(coset)
        return tuple(result)

    def activity_mask(self, slots):
        """Prove the exact active set of a finite affine hardware lane table."""
        if (type(slots) not in (tuple, list) or len(slots) > _THREADS
                or any(type(slot) is not int or not 0 <= slot < math.prod(self.carrier_shape) for slot in slots)):
            raise ValueError('invalid bounded lane table for valid prefix')
        base, columns = linear(slots)
        if self.carrier_shape == self.valid_shape:
            return (1 << len(slots)) - 1
        mask = 0
        for coset in self.affine_pullback(base, columns):
            points = [coset.origin]
            for column in coset.basis:
                points += [point ^ column for point in points]
            for point in points:
                if mask >> point & 1:
                    raise ValueError('valid-domain pullback cosets overlap')
                mask |= 1 << point
        expected = sum(1 << i for i, slot in enumerate(slots) if self.contains(slot))
        if mask != expected:
            raise ValueError('valid-domain pullback differs from checked lane table')
        return mask


def dense_copy_execution(shape):
    """A complete 128-thread execution with consecutive slots per thread."""
    domain = PrefixDomain(shape, shape)
    volume = math.prod(domain.carrier_shape)
    if volume % _THREADS:
        raise ValueError('dense Copy carrier requires full hardware threads')
    return dict(shape=list(domain.carrier_shape), topology=[volume // _THREADS, 32, 4, 1, 1],
                logical=list(range(volume)), first=list(range(volume)), replication=1)


def _carrier(execution, valid_shape):
    if type(execution) is not dict:
        raise ValueError('expected a checked execution distribution')
    if 'symbolic' in execution:
        from execution_support import normalize_distribution
        execution = normalize_distribution(execution)
    domain = PrefixDomain(execution.get('shape'), valid_shape)
    topology = execution.get('topology')
    if (type(topology) not in (list, tuple) or len(topology) != 5
            or any(type(n) is not int for n in topology)
            or tuple(topology[1:]) != (32, 4, 1, 1) or not 1 <= topology[0] <= _MAX_SLOTS // _THREADS):
        raise ValueError('unsupported matrix carrier topology')
    nr = topology[0]
    logical = execution.get('logical')
    if not isinstance(logical, Sequence) or len(logical) != _THREADS * nr:
        raise ValueError('incomplete matrix carrier table')
    slots = tuple(logical)
    volume = math.prod(domain.carrier_shape)
    if any(type(slot) is not int or not 0 <= slot < volume for slot in slots):
        raise ValueError('logical slot outside padded matrix')
    linear(slots)
    first = [-1] * volume
    for holder, slot in enumerate(slots):
        if first[slot] == -1:
            first[slot] = holder
    supplied = execution.get('first')
    if (type(supplied) not in (list, tuple) or len(supplied) != volume
            or any(type(owner) is not int for owner in supplied) or -1 in first
            or tuple(first) != tuple(supplied)):
        raise ValueError('matrix first-owner table differs from the carrier')
    if ('replication' in execution and (type(execution['replication']) is not int
            or execution['replication'] != len(slots) // volume)):
        raise ValueError('matrix replication differs from the carrier')
    return domain, nr, slots, tuple(first)


def _prove_global_extent(domain, stride, span, element_bytes, alignment):
    rows, cols = domain.valid_shape
    carrier_rows, carrier_cols = domain.carrier_shape
    if (type(stride) is not int or not max(1, cols * element_bytes) <= stride < _MAX_BYTES
            or stride % alignment or type(span) is not int or not 0 <= span <= _MAX_BYTES):
        raise ValueError('invalid Global row stride, alignment or byte span')
    end = (rows - 1) * stride + cols * element_bytes if rows and cols else 0
    if end > span:
        raise ValueError('active prefix escapes the Global byte span')
    conceptual_end = (carrier_rows - 1) * stride + carrier_cols * element_bytes
    if conceptual_end > _MAX_BYTES:
        raise ValueError('matrix carrier offset exceeds unsigned 32-bit addressing')


@dataclass(frozen=True)
class Copy16Group:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    source_offsets: tuple[int, ...]
    destination_offsets: tuple[int, ...]
    active_mask: int


@dataclass(frozen=True)
class GuardedCopyPlan:
    domain: PrefixDomain
    groups: tuple[Copy16Group, ...]
    lane_bit_order: tuple[int, ...] | None


def plan_guarded_copy_owners(execution, *, valid_shape, source_row_stride_bytes,
                            source_span_bytes, destination_offsets,
                            destination_span_bytes, owner_policy='gf2'):
    """Prove copy/zero ownership from compact Global f16 to a padded Shared tile.

    Both root bases require 16-byte alignment. Relative Global addresses of
    inactive groups are conceptual only and must never be dereferenced. A
    single shape-independent group assignment proves any runtime row prefix
    with the same column extent, after the caller binds the correct root span.
    """
    domain, nr, slots, first = _carrier(execution, valid_shape)
    rows, cols = domain.carrier_shape
    group_size = 8
    if (owner_policy not in ('execution', 'gf2') or cols % group_size
            or domain.valid_shape[1] % group_size or nr % group_size
            or nr * _THREADS != rows * cols):
        raise ValueError('Copy16 requires complete unique vector groups')
    _prove_global_extent(domain, source_row_stride_bytes, source_span_bytes, 2, 16)
    if type(destination_span_bytes) is not int or not 16 <= destination_span_bytes <= _MAX_BYTES:
        raise ValueError('invalid Shared byte span')
    try:
        destination = tuple(destination_offsets[slot] for slot in range(rows * cols))
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError('incomplete Shared destination address table') from error
    if any(type(offset) is not int or not 0 <= offset < destination_span_bytes for offset in destination):
        raise ValueError('Shared destination address outside its root')
    virtual_source, shared, covered = [], [], set()
    for reg in range(0, nr, group_size):
        source_row, shared_row = [], []
        for thread in range(_THREADS):
            holder = thread * nr + reg
            group = slots[holder:holder + group_size]
            start, target = group[0], destination[group[0]]
            if (start % group_size or group != tuple(range(start, start + group_size))
                    or any(first[slot] != holder + i for i, slot in enumerate(group))
                    or target % 16 or target + 16 > destination_span_bytes
                    or any(destination[slot] != target + 2 * i for i, slot in enumerate(group))
                    or target in covered):
                raise ValueError('invalid or overlapping Shared Copy16 byte group')
            covered.add(target)
            source_row.append(start * 2)
            shared_row.append(target)
        virtual_source.append(tuple(source_row))
        shared.append(tuple(shared_row))
    lane_order = None
    if owner_policy == 'gf2':
        reowned = plan_copy_owners(virtual_source, shared)
        if reowned is not None:
            (virtual_source, shared), evidence = reowned
            lane_order = tuple(evidence['lane_bit_order'])

    groups = []
    copied = set()
    for source_row, shared_row in zip(virtual_source, shared):
        starts = tuple(offset // 2 for offset in source_row)
        row_table, col_table = map(tuple, zip(*(divmod(slot, cols) for slot in starts)))
        effective = tuple(row * source_row_stride_bytes + col * 2 for row, col in zip(row_table, col_table))
        mask = domain.activity_mask(starts)
        for lane, (slot, src, dst) in enumerate(zip(starts, effective, shared_row)):
            if (slot in copied or dst != destination[slot]
                    or (mask >> lane & 1 and (src % 16 or src + 16 > source_span_bytes))):
                raise ValueError('Copy16 ownership changed its byte pairing or bounds')
            copied.add(slot)
        for table in (row_table, col_table, effective, shared_row):
            linear(table)
        groups.append(Copy16Group(row_table, col_table, effective, tuple(shared_row), mask))
    if len(copied) != rows * cols // group_size:
        raise ValueError('Copy16 does not initialize the complete Shared carrier')
    return GuardedCopyPlan(domain, tuple(groups), lane_order)


@dataclass(frozen=True)
class ScalarStoreGroup:
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    destination_offsets: tuple[int, ...]
    owner_mask: int
    active_mask: int


@dataclass(frozen=True)
class GuardedStorePlan:
    domain: PrefixDomain
    groups: tuple[ScalarStoreGroup, ...]


def plan_guarded_store(execution, *, valid_shape, destination_row_stride_bytes,
                       destination_span_bytes):
    """Prove unique scalar f32 stores, predicated by ownership and validity."""
    domain, nr, slots, first = _carrier(execution, valid_shape)
    _prove_global_extent(domain, destination_row_stride_bytes, destination_span_bytes, 4, 4)
    cols = domain.carrier_shape[1]
    groups = []
    written = set()
    for reg in range(nr):
        lane_slots = slots[reg::nr]
        rows, columns = map(tuple, zip(*(divmod(slot, cols) for slot in lane_slots)))
        offsets = tuple(row * destination_row_stride_bytes + col * 4 for row, col in zip(rows, columns))
        owners = sum(1 << t for t, slot in enumerate(lane_slots) if first[slot] == t * nr + reg)
        active = owners & domain.activity_mask(lane_slots)
        for lane, offset in enumerate(offsets):
            if active >> lane & 1:
                if offset in written or offset + 4 > destination_span_bytes:
                    raise ValueError('active scalar store overlaps or escapes its Global root')
                written.add(offset)
        for table in (rows, columns, offsets):
            linear(table)
        groups.append(ScalarStoreGroup(rows, columns, offsets, owners, active))
    if len(written) != math.prod(domain.valid_shape):
        raise ValueError('scalar stores do not cover the valid prefix')
    return GuardedStorePlan(domain, tuple(groups))
