"""Native QK score holders that survive a checked key-column prefix."""

from dataclasses import dataclass
from functools import lru_cache

from attention_tile_mma import FRAGMENT_SLOTS, THREADS, TILE, _fragment_columns
from valid_domain import PrefixDomain


@dataclass(frozen=True)
class ScorePrefix:
    valid_columns: int
    kv_width: int
    holder_masks: tuple[int, ...]
    # Compare 2 * (tid & 3) against the slot's cutoff. Zero and seven
    # denote the empty and full thread sets, respectively.
    guard_cutoffs: tuple[int, ...]

    @property
    def held_slots(self):
        return tuple(slot for slot, mask in enumerate(self.holder_masks) if mask)


def _images(columns):
    result = [0]
    for column in columns:
        result += [value ^ column for value in result]
    return result


@lru_cache(maxsize=TILE)
def _native_score_prefix(valid_columns):
    columns = _fragment_columns()
    register_bits = FRAGMENT_SLOTS.bit_length() - 1
    origins = _images(columns[:register_bits])
    thread_offsets = _images(columns[register_bits:])
    domain = PrefixDomain((TILE, TILE), (TILE, valid_columns))
    masks = []
    cutoffs = []
    for origin in origins:
        # Fix the register axis and pull the logical prefix back along the
        # thread axes of the checked native QK/RS-P carrier.
        holders = tuple(origin ^ offset for offset in thread_offsets)
        mask = domain.activity_mask(holders)
        cutoff = min(7, max(0, valid_columns - origin % TILE))
        predicate = sum(1 << tid for tid in range(THREADS) if 2 * (tid & 3) < cutoff)
        if mask != predicate:
            raise ValueError("score-prefix pullback differs from the native lane predicate")
        masks.append(mask)
        cutoffs.append(cutoff)
    return ScorePrefix(valid_columns, TILE, tuple(masks), tuple(cutoffs))


def prove_score_prefix(fragment_columns, valid_columns, *, kv_width=TILE):
    """Bind score liveness to the same native carrier as the QK/PV plan."""
    if (type(valid_columns) is not int or not 1 <= valid_columns <= TILE
            or type(kv_width) is not int or kv_width != TILE):
        raise ValueError("score reuse requires a nonempty prefix of the native KV64 carrier")
    if (type(fragment_columns) not in (list, tuple)
            or any(type(column) is not int for column in fragment_columns)
            or tuple(fragment_columns) != _fragment_columns()):
        raise ValueError("score reuse requires the checked native GF(2) fragment")
    return _native_score_prefix(valid_columns)

