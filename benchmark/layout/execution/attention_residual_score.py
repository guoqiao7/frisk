"""Cache the complete static KV128 remainder as one checked QK carrier."""

from functools import lru_cache

from access_plan import linear
from attention_score_reuse import prove_score_prefix
from attention_tile_mma import FRAGMENT_SLOTS, ROOT_BYTES, THREADS, TILE, fragment_coordinate
from tiled_attention import _ALL_THREADS, _FP_ZERO, _TiledAttention
from valid_domain import PrefixDomain


def residual_score_holders(valid_columns, fragment_columns):
    """Pull a 65..127 key prefix back to the native KV128 register owners."""
    if type(valid_columns) is not int or not TILE < valid_columns < 2 * TILE:
        raise ValueError('static KV128 remainder requires 65 to 127 valid columns')
    if type(fragment_columns) is not tuple or any(type(basis) is not int for basis in fragment_columns):
        raise ValueError('static remainder requires integer native GF(2) columns')
    return _residual_score_holders(valid_columns, fragment_columns)


@lru_cache(maxsize=63)
def _residual_score_holders(valid_columns, fragment_columns):
    width = 2 * TILE
    slots = 2 * FRAGMENT_SLOTS
    table = tuple(row * width + column for thread in range(THREADS)
                  for register in range(slots)
                  for row, column in (fragment_coordinate(thread, register, kv_tile=width),))
    origin, columns = linear(table)
    if (origin or type(fragment_columns) is not tuple or any(type(basis) is not int for basis in fragment_columns)
            or tuple(columns) != fragment_columns):
        raise ValueError('static remainder requires the native KV128 GF(2) carrier')
    domain = PrefixDomain((TILE, width), (TILE, valid_columns))
    masks = tuple(domain.activity_mask(tuple(table[thread * slots + register] for thread in range(THREADS)))
                  for register in range(slots))
    # The low register coset is a complete native KV64. The high coset
    # restricts to exactly the existing guarded tail, with the same holders.
    narrow_table = tuple(row * TILE + column for thread in range(THREADS)
                         for register in range(FRAGMENT_SLOTS)
                         for row, column in (fragment_coordinate(thread, register),))
    narrow_origin, narrow_columns = linear(narrow_table)
    if narrow_origin or masks[:FRAGMENT_SLOTS] != (_ALL_THREADS,) * FRAGMENT_SLOTS:
        raise ValueError('static remainder changed a complete lower KV64')
    suffix = prove_score_prefix(tuple(narrow_columns), valid_columns - TILE)
    if masks[FRAGMENT_SLOTS:] != suffix.holder_masks:
        raise ValueError('static remainder changed the native guarded-tail projection')
    return masks


class _ResidualScoreAttention(_TiledAttention):
    def __init__(self, spec, *, tile_packet, vector_store, copy_owner_policy):
        if (type(spec) is not dict or type(spec.get('N')) is not int or spec['N'] < 2 * TILE
                or not TILE < spec['N'] % (2 * TILE) < 2 * TILE):
            raise ValueError('residual-score reuse requires N%128 in 65..127 and at least one full KV128')
        super().__init__(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                         kv_width=2 * TILE, retain_tail_scores=True, vector_store=vector_store)
        if self.residual_row is None or self.residual_row + TILE != self.tail_row:
            raise ValueError('static remainder must contain one complete KV64 followed by its tail')
        self.residual_masks = residual_score_holders(TILE + self.tail_prefix,
                                                    tuple(self.mma_tiles.score_fragment_columns))
        self.residual_left = None
        self.residual_right = None

    def _build_residual_scores(self):
        self.emit('bar.sync 0;')
        self._copy(self.roots[1], self.residual_row, self.mma_tiles.k_offset, guard=False)
        self._copy(self.roots[1], self.tail_row, self.mma_tiles.k_offset + ROOT_BYTES,
                   guard=True, prefix=self.tail_prefix)
        self._publish()
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32')
                  for _ in range(2 * FRAGMENT_SLOTS)]
        self.mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        scaled = tuple(self._value('mul.rn.f32', score, '0f3e000000', kind='f32') if mask else self.zero
                       for score, mask in zip(scores, self.residual_masks, strict=True))
        self.residual_left = scaled[:FRAGMENT_SLOTS]
        self.residual_right = scaled[FRAGMENT_SLOTS:]

    def _scores(self, tile_row, *, prefix, copy_v=False):
        if type(tile_row) is int and tile_row == self.residual_row:
            if prefix != TILE:
                raise ValueError('complete static remainder half requires 64 valid columns')
            if self.residual_left is None:
                if copy_v:
                    raise ValueError('static QK remainder must be produced before PV')
                self._build_residual_scores()
            if copy_v:
                self._publish_values(self.residual_row, TILE)
            return self.residual_left
        if type(tile_row) is int and tile_row == self.tail_row:
            if prefix != self.tail_prefix or self.residual_right is None:
                raise ValueError('static tail must use its already-produced native upper coset')
            if copy_v:
                self._publish_values(self.tail_row, self.tail_prefix)
            return self.residual_right
        return super()._scores(tile_row, prefix=prefix, copy_v=copy_v)

    def run(self):
        text, info = super().run()
        if self.residual_left is None or self.residual_right is None:
            raise ValueError('static remainder was never produced')
        info.update(retain_residual_scores=True, residual_score_rows=TILE + self.tail_prefix,
                    residual_score_evaluations=1, residual_native_qk_width=2 * TILE,
                    residual_score_cache_slots=sum(bool(mask) for mask in self.residual_masks),
                    residual_score_holder_counts=[mask.bit_count() for mask in self.residual_masks],
                    residual_shared_publication='K0 complete plus K1 guarded; wait0 before V replaces K1')
        return text, info


def emit_residual_score_attention(spec, *, tile_packet, vector_store=True, copy_owner_policy='gf2',
                                  factored_copy=False, simplify_tail_copy=False):
    if type(factored_copy) is not bool or type(simplify_tail_copy) is not bool:
        raise ValueError('residual copy strategies require boolean flags')
    emitter = _ResidualScoreAttention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                      copy_owner_policy=copy_owner_policy)
    emitter.factored_copy = factored_copy
    emitter.simplify_tail_copy = simplify_tail_copy
    return emitter.run()

