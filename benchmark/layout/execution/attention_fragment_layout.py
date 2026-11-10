"""Proof-bound logical row carriers for D64 QK, ordered softmax and PV."""

from dataclasses import dataclass
from functools import lru_cache

from access_plan import linear
from attention_tile_mma import FRAGMENT_SLOTS, THREADS, TILE, fragment_coordinate


NATIVE_FRAGMENT = 'native'
ROW_LANE_REGISTER = 'q-row-lane-register'
FRAGMENT_LAYOUTS = (NATIVE_FRAGMENT, ROW_LANE_REGISTER)


@dataclass(frozen=True)
class AttentionFragment:
    name: str
    rows: tuple[int, ...]

    def query_offsets(self, descriptor_offsets):
        # Logical query row r is placed at descriptor-visible row R(r).
        # Columns stay fixed so all eight halfwords in each Copy16 stay together.
        return tuple(descriptor_offsets[physical * TILE + col]
                     for physical in self.rows for col in range(TILE))

    def accumulator_distribution(self, native):
        if self.name == NATIVE_FRAGMENT:
            return native
        # R is self-inverse. A native row m therefore carries logical row R(m).
        logical = [self.rows[row] * TILE + col for row, col in
                   (divmod(slot, TILE) for slot in native['logical'])]
        first = [0] * len(logical)
        for holder, slot in enumerate(logical):
            first[slot] = holder
        return dict(native, logical=logical, first=first)

    def evidence(self, accumulator_columns, score_columns, *, score_width):
        return dict(fragment_layout=self.name, query_native_row_map=list(self.rows),
                    fragment_row_gf2_columns=list(linear(self.rows)[1]),
                    logical_fragment_gf2_columns=list(self._logical_columns(accumulator_columns, TILE)),
                    logical_score_gf2_columns=list(self._logical_columns(score_columns, score_width)))

    def _logical_columns(self, physical_columns, columns):
        return tuple(self.rows[row] * columns + column
                     for row, column in (divmod(bit, columns) for bit in physical_columns))


@lru_cache(maxsize=len(FRAGMENT_LAYOUTS))
def checked_fragment_layout(name):
    if name not in FRAGMENT_LAYOUTS:
        raise ValueError('unsupported checked Attention fragment layout')
    if name == NATIVE_FRAGMENT:
        return AttentionFragment(name, tuple(range(TILE)))
    # Swap row bit 0 (owned by lane bit 2) and row bit 3 (owned by register
    # bit 1). No score key bits or PV output column bits participate.
    rows = tuple(row ^ 0b001001 if bool(row & 1) != bool(row & 8) else row for row in range(TILE))
    origin, _ = linear(rows)
    if origin or len(set(rows)) != TILE or any(rows[rows[row]] != row for row in range(TILE)):
        raise ValueError('Attention fragment row map is not an involutive GF(2) bijection')
    fragment = AttentionFragment(name, rows)

    for width in (8, 16, 32, 64, 128, 256):
        native, logical = [], []
        for thread in range(THREADS):
            for reg in range(width // 2):
                row, col = fragment_coordinate(thread, reg, kv_tile=width)
                # All four lanes of each canonical ordered reduction fiber
                # continue to hold one logical row and the same ordered keys.
                peer_row, peer_col = fragment_coordinate(thread & ~3, reg, kv_tile=width)
                if rows[row] != rows[peer_row] or col != peer_col + 2 * (thread & 3):
                    raise ValueError('Attention row map changes an ordered reduction fiber')
                native.append(row * width + col)
                logical.append(rows[row] * width + col)
        native_origin, physical_columns = linear(native)
        logical_origin, logical_columns = linear(logical)
        if (native_origin or logical_origin or len(set(logical)) != TILE * width
                or tuple(logical_columns) != fragment._logical_columns(physical_columns, width)):
            raise ValueError('Attention row map does not conjugate its native accumulator')
        if width == TILE:
            register_row_bit = 1
            lane_row_bit = FRAGMENT_SLOTS.bit_length() - 1 + 2
            if (logical_columns[register_row_bit] != physical_columns[lane_row_bit]
                    or logical_columns[lane_row_bit] != physical_columns[register_row_bit]
                    or all(a == b for a, b in zip(physical_columns, logical_columns, strict=True))):
                raise ValueError('Attention row map does not exchange native register and lane ownership')
    return fragment


def output_store_evidence(plans, *, vector_store):
    """Record physical output issue and sectors after logical prefix masking."""
    report = {}
    for rows, plan in sorted(plans.items()):
        groups = plan.groups[::2] if vector_store else plan.groups
        active_elements = sum(group.active_mask.bit_count() for group in plan.groups)
        if active_elements != rows * TILE:
            raise ValueError('Attention output proof lost logical rows')
        sector_counts = [0, 0]
        issue_warps = 0
        for group in groups:
            for warp in range(0, THREADS, 32):
                addresses = [group.destination_offsets[lane] for lane in range(warp, warp + 32)
                             if group.active_mask >> lane & 1]
                issue_warps += bool(addresses)
                for index, phase in enumerate((0, 16)):
                    sector_counts[index] += len({(address + phase) // 32 for address in addresses})
        report[rows] = dict(active_elements=active_elements, warp_issues=issue_warps,
                            global_sectors32=sector_counts)
    return report
