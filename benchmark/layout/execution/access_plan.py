"""Checked address reuse and vector grouping for one Global memory operation."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations


_MAX_BYTE_OFFSET = (1 << 32) - 1
_MAX_DISPLACEMENT = (1 << 31) - 1


@dataclass(frozen=True)
class AccessGroup:
    # Slots are in ascending byte-address order, which is the PTX operand order.
    slots: tuple[int, ...]
    base_index: int
    byte_offset: int
    width_bytes: int
    # None means all lanes perform an unconditional load.
    owner_mask: int | None


@dataclass(frozen=True)
class AccessPlan:
    # Each base is an original offset table, accepted by the same GF(2)
    # verifier as an unfactored access. byte_offset is an integer displacement.
    bases: tuple[tuple[int, ...], ...]
    groups: tuple[AccessGroup, ...]


def linear(table):
    """Return affine GF(2) bit bases after full-domain equivalence."""
    size = len(table)
    if size < 1 or size & (size - 1):
        raise ValueError("non power-of-two map domain")
    base = table[0]
    bits = [table[1 << i] ^ base for i in range(size.bit_length() - 1)]
    for index, expected in enumerate(table):
        actual = base
        for i, bit in enumerate(bits):
            if index >> i & 1:
                actual ^= bit
        if actual != expected:
            raise ValueError("map is outside checked GF(2) subset")
    return base, bits


def _copy_cost(source, target):
    sectors = tuple(
        sum(len({(offset + phase) // 32 for offset in row[warp:warp + 32]})
            for row in source for warp in range(0, 128, 32))
        for phase in (0, 16)
    )
    # Each 128-byte portion of STS.128 has eight lanes. Each 16-byte
    # vector occupies one aligned group of four consecutive Shared banks.
    shared_excess = sum(
        max(Counter((offset >> 4) & 7 for offset in row[lane:lane + 8]).values()) - 1
        for row in target for lane in range(0, 128, 8)
    )
    return (*sectors, shared_excess)


def _independent_bank_axes(columns, axes):
    first, second, third = (columns[axis] for axis in axes)
    return (first != 0 and second not in (0, first)
            and third not in (0, first, second, first ^ second))


def plan_copy_owners(source, target):
    """Reassign whole Copy16 segments, keeping each source/destination pair intact.

    The caller proves each segment's byte pairing and checks its resulting
    addresses with plan_access. Return None when the original execution
    ownership is preferable or the checked affine subset does not apply.
    """
    if not source or len(source) != len(target) or any(len(row) != 128 for row in (*source, *target)):
        return None
    pairs = sorted(pair for src, dst in zip(source, target) for pair in zip(src, dst))
    base = pairs[0][0]
    if base % 16 or any(src != base + 16 * i or dst % 16 for i, (src, dst) in enumerate(pairs)):
        return None

    rows = [tuple(zip(*pairs[start:start + 128])) for start in range(0, len(pairs), 128)]
    try:
        bank_columns = []
        for src, dst in rows:
            linear(src)
            _, bits = linear(dst)
            bank_columns.append(tuple((bit >> 4) & 7 for bit in bits[:5]))
    except ValueError:
        return None

    # Permuting five lane bits preserves the 32 consecutive source vectors.
    # Pick three independent Shared bank-group columns for the low lane bits,
    # so every eight-lane transaction uses all eight bank groups once.
    axes = next((axes for axes in combinations(range(5), 3)
                 if all(_independent_bank_axes(cols, axes) for cols in bank_columns)), None)
    if axes is None:
        return None
    order = (*axes, *(axis for axis in range(5) if axis not in axes))
    lanes = tuple((lane & ~31) | sum(((lane >> i) & 1) << axis for i, axis in enumerate(order))
                  for lane in range(128))
    planned = tuple(tuple(tuple(row[endpoint][lane] for lane in lanes) for row in rows)
                    for endpoint in range(2))
    before = _copy_cost(source, target)
    after = _copy_cost(*planned)
    if before == after or any(new > old for old, new in zip(before, after)):
        return None
    return planned, dict(lane_bit_order=list(order), global_sectors_32=list(after[:2]),
                         previous_global_sectors_32=list(before[:2]), shared_excess_wavefronts=after[2],
                         previous_shared_excess_wavefronts=before[2])


def plan_access(offset_tables, *, element_bytes, root_alignment, root_bytes,
                owner_masks=None, max_vector_bytes=16):
    """Plan reads or uniquely owned writes against an aligned root allocation.

    Each row gives the actual byte offset of one register slot in every lane.
    A 16-byte slot represents a caller-proved contiguous raw copy segment.
    With owner_masks, bit t denotes that lane t performs that slot's store.
    Otherwise every slot is loaded in every lane. Malformed and out-of-range
    accesses raise ValueError. A one-slot group of width one requires the
    byte-wise scalar path for the entire element.
    """
    if type(element_bytes) is not int or element_bytes not in (2, 4, 16):
        raise ValueError("access element must be FP16, FP32, or a proved 16-byte copy segment")
    if type(root_alignment) is not int or root_alignment < 1 or root_alignment & (root_alignment - 1):
        raise ValueError("root alignment must be a positive power of two")
    if type(root_bytes) is not int or not 0 <= root_bytes <= (1 << 64):
        raise ValueError("invalid root byte extent")
    if type(max_vector_bytes) is not int or max_vector_bytes not in (1, 2, 4, 8, 16):
        raise ValueError("invalid maximum vector width")

    try:
        tables = tuple(tuple(row) for row in offset_tables)
    except TypeError as error:
        raise ValueError("invalid physical lane table") from error
    lane_count = len(tables[0]) if tables else 0
    if tables and (lane_count == 0 or any(len(row) != lane_count for row in tables)):
        raise ValueError("incomplete physical lane table")
    if owner_masks is None:
        masks = (None,) * len(tables)
    else:
        try:
            masks = tuple(owner_masks)
        except TypeError as error:
            raise ValueError("invalid per-slot owner mask") from error
        if len(masks) != len(tables) or any(
            type(mask) is not int or mask < 0 or mask.bit_length() > lane_count for mask in masks
        ):
            raise ValueError("invalid per-slot owner mask")

    lanes_by_mask = {}
    written_bytes = set()
    alignment_bits = []
    classes = defaultdict(list)
    for slot, (table, mask) in enumerate(zip(tables, masks)):
        if any(type(offset) is not int or not 0 <= offset <= _MAX_BYTE_OFFSET for offset in table):
            raise ValueError("byte offset is outside the unsigned 32-bit map domain")
        if mask not in lanes_by_mask:
            lanes_by_mask[mask] = tuple(lane for lane in range(lane_count) if mask is None or mask >> lane & 1)
        # For a power-of-two width, every active offset and the root must
        # have all the bits below that width clear.
        low_bits = root_alignment
        for lane in lanes_by_mask[mask]:
            offset = table[lane]
            low_bits |= offset
            end = offset + element_bytes
            if end > root_bytes:
                raise ValueError("active access escapes the checked root allocation")
            if mask is not None:
                for byte in range(offset, end):
                    if byte in written_bytes:
                        raise ValueError("store owners overlap in the checked root allocation")
                    written_bytes.add(byte)
        alignment_bits.append(low_bits)
        if mask != 0:
            # Equal XOR deltas can change integer displacements through carry
            # or borrow. The integer signature proves the exact translation
            # over every lane, even lanes masked off for an eventual store.
            classes[tuple(offset - table[0] for offset in table)].append(slot)

    bases = []
    groups = []
    vector_widths = tuple(width for width in (16, 8, 4) if element_bytes < width <= max_vector_bytes)
    # Reserve room for byte offsets within even the last scalar element.
    displacement_limit = _MAX_DISPLACEMENT - (element_bytes - 1)
    for slots in classes.values():
        # Starting at the minimum original row makes every displacement
        # non-negative. Large displacements start a new original base.
        partitions = []
        for slot in sorted(slots, key=lambda index: (tables[index][0], index)):
            origin = tables[slot][0]
            if not partitions or origin - partitions[-1][0] > displacement_limit:
                partitions.append((origin, []))
            partitions[-1][1].append(slot)
        for origin, partition in partitions:
            base_index = len(bases)
            bases.append(tables[partition[0]])
            by_mask = defaultdict(lambda: defaultdict(deque))
            for slot in partition:
                by_mask[masks[slot]][tables[slot][0] - origin].append(slot)
            for mask, offsets in by_mask.items():
                for offset, pending in offsets.items():
                    while pending:
                        slot = pending[0]
                        alignment = alignment_bits[slot]
                        native_scalar = element_bytes <= max_vector_bytes and not alignment & (element_bytes - 1)
                        width = element_bytes if native_scalar else 1
                        chosen = (offset,)
                        for candidate in vector_widths:
                            if alignment & (candidate - 1):
                                continue
                            adjacent = tuple(range(offset, offset + candidate, element_bytes))
                            if all(offsets.get(address) for address in adjacent):
                                width = candidate
                                chosen = adjacent
                                break
                        groups.append(AccessGroup(tuple(offsets[address].popleft() for address in chosen),
                                                  base_index, offset, width, mask))
    return AccessPlan(tuple(bases), tuple(sorted(groups, key=lambda group: min(group.slots))))
