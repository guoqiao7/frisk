"""Checked address reuse and vector grouping for one Global memory operation."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


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
