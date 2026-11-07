"""Single QK evaluation for a checked, complete short Attention score carrier."""

from functools import lru_cache
import hashlib

from access_plan import linear
from attention_tile_mma import (FRAGMENT_SLOTS, ROOT_BYTES, SHARED_ALIGNMENT, THREADS,
                                TILE, build_d64_wgmma_tiles, fragment_coordinate)
from backend import render_inline_execution
from native_static_support import check_static_shared
from tiled_attention import _FP_ZERO, _NEG_INF, _TiledAttention
from valid_domain import PrefixDomain


@lru_cache(maxsize=3)
def single_score_rows(width, fragment_columns):
    """Prove row fibers and native PV projections for one complete score tile."""
    if type(width) is not int or width not in (64, 128, 256):
        raise ValueError('single-score Attention requires a complete KV64, KV128 or KV256 carrier')
    registers = width // 2
    table = tuple(row * width + column for thread in range(THREADS)
                  for slot in range(registers)
                  for row, column in (fragment_coordinate(thread, slot, kv_tile=width),))
    origin, columns = linear(table)
    if origin or tuple(columns) != tuple(fragment_columns):
        raise ValueError('single-score ownership differs from the native GF(2) carrier')
    domain = PrefixDomain((TILE, width), (TILE, width))
    if set(table) != set(range(TILE * width)) or not all(domain.contains(slot) for slot in table):
        raise ValueError('single-score carrier does not bijectively cover the full valid domain')
    row0 = fragment_coordinate(0, 0)[0]
    rows = tuple(tuple(slot for slot in range(registers)
                       if fragment_coordinate(0, slot, kv_tile=width)[0] == row)
                 for row in (row0, row0 + 8))
    if any(len(slots) != width // 4 for slots in rows):
        raise ValueError('single-score reduction needs two native row fibers')
    for thread in range(THREADS):
        for slots in rows:
            previous_column = -1
            expected_row = fragment_coordinate(thread, slots[0], kv_tile=width)[0]
            for slot in slots:
                row, column = fragment_coordinate(thread, slot, kv_tile=width)
                block, narrow_slot = divmod(slot, FRAGMENT_SLOTS)
                narrow_row, narrow_column = fragment_coordinate(thread, narrow_slot)
                if ((row, column) != (narrow_row, block * TILE + narrow_column)
                        or row != expected_row or column <= previous_column):
                    raise ValueError('single-score row order differs from the checked PV subcarrier')
                previous_column = column
    return rows


class _SingleScoreAttention(_TiledAttention):
    """Retire scaled scores into exponents, then consume probabilities in order."""

    def __init__(self, spec, *, tile_packet, vector_store, copy_owner_policy):
        if type(spec) is not dict or type(spec.get('N')) is not int or spec['N'] not in (64, 128, 256):
            raise ValueError('single-score Attention supports aligned N64, N128 and N256')
        super().__init__(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                         kv_width=TILE, retain_tail_scores=False, vector_store=vector_store)
        previous_shared = self.mma_tiles.shared_bytes
        self.mma_tiles = build_d64_wgmma_tiles(kv_tile=self.length)
        self.kv_width = self.length
        self.score_rows = single_score_rows(self.length, tuple(self.mma_tiles.score_fragment_columns))
        old = f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{previous_shared}];'
        if self.decl.count(old) != 1:
            raise ValueError('single-score storage requires one checked Shared allocation')
        self.decl[self.decl.index(old)] = (
            f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{self.mma_tiles.shared_bytes}];')

    def _all_scores(self):
        self._copy(self.roots[0], self.query_row, self.mma_tiles.q_offset, guard=False)
        self.emit('bar.sync 0;')
        for block in range(self.length // TILE):
            self._copy(self.roots[1], block * TILE,
                       self.mma_tiles.k_offset + block * ROOT_BYTES, guard=False)
        self._publish()
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(self.length // 2)]
        self.mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        return [self._value('mul.rn.f32', value, '0f3e000000', kind='f32') for value in scores]

    def _statistics(self, scores):
        maxima = [self._value('mov.f32', _NEG_INF, kind='f32') for _ in self.rows]
        # Retain the original per-KV64 max order. The sum below uses the
        # complete per-lane tree, which factors into those exact KV64 subtrees.
        for block in range(self.length // TILE):
            for maximum, slots in zip(maxima, self.rows, strict=True):
                leaves = [scores[block * FRAGMENT_SLOTS + slot] for slot in slots]
                local, _ = self._tree('max.f32', leaves)
                self.emit(f'max.f32 {maximum}, {maximum}, {local};')
        maxima = [self._warp_reduce('max.f32', value, ordered=False) for value in maxima]
        exponent = []
        for block in range(self.length // TILE):
            begin = block * FRAGMENT_SLOTS
            exponent.extend(self._exponents(scores[begin:begin + FRAGMENT_SLOTS],
                                            maxima, (True,) * FRAGMENT_SLOTS))
        sums = []
        for slots in self.score_rows:
            local, _ = self._tree('add.rn.f32', [exponent[slot] for slot in slots])
            sums.append(self._warp_reduce('add.rn.f32', local, ordered=True))
        return exponent, sums

    def _consume_values(self, exponent, sums):
        accum = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(FRAGMENT_SLOTS)]
        for block in range(self.length // TILE):
            self._publish_values(block * TILE, TILE)
            begin = block * FRAGMENT_SLOTS
            packed = []
            # Packing each normalized pair immediately makes its FP32
            # temporaries dead before the next pair is formed.
            for slot in range(0, FRAGMENT_SLOTS, 2):
                low = self._value('div.rn.f32', exponent[begin + slot],
                                  sums[self.row_index[slot]], kind='f32')
                high = self._value('div.rn.f32', exponent[begin + slot + 1],
                                   sums[self.row_index[slot + 1]], kind='f32')
                packed.append(self._value('cvt.rn.f16x2.f32', high, low))
            self.mma_tiles.emit_pv(self.emit, self.tmp, self.smem_base, packed, accum)
        return accum

    def run(self):
        exponent, sums = self._statistics(self._all_scores())
        self._write_output(self._consume_values(exponent, sums))
        text = render_inline_execution(self.decl, self.code)
        info = dict(self.geometry, static_shared_bytes=self.mma_tiles.shared_bytes,
                    physical_shared_bytes=self.mma_tiles.shared_bytes,
                    qkv_root_offsets=[self.mma_tiles.q_offset, self.mma_tiles.k_offset,
                                      self.mma_tiles.v_offset],
                    fragment_gf2_columns=list(self.mma_tiles.fragment_columns),
                    score_gf2_columns=list(self.mma_tiles.score_fragment_columns),
                    execution_kv_width=self.length, retain_tail_score_slots=[],
                    retain_all_scores=True, vector_store=self.vector_store,
                    score_evaluations=1, score_cache_slots=self.length // 2,
                    score_cache_transition='scaled-qk-to-exponent',
                    shared_v_reuse='last-k64-after-wait0' if self.length > TILE else 'disjoint',
                    copy_owner_policy=self.copy_owner_policy,
                    copy_lane_bit_order=self.copy_plans[TILE].lane_bit_order,
                    passes=['global-max', 'canonical-row-sum', 'normalized-fp16-pv'],
                    reduction_order='hardware-first-owner-v1 with absent leaves pruned',
                    p_rounding='cvt.rn.f16x2.f32 after global normalization',
                    ftz=False, fp_fusion=False, mask='query-and-kv-prefix',
                    tail_copy_active_groups=[], tail_store_elements=0,
                    ttgir_sha256=hashlib.sha256(text.encode()).hexdigest())
        check_static_shared(info)
        return text, info


def emit_single_score_attention(spec, *, tile_packet, vector_store=True, copy_owner_policy='gf2', factored_copy=False):
    if type(factored_copy) is not bool:
        raise ValueError('copy schedule flag must be boolean')
    emitter = _SingleScoreAttention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                   copy_owner_policy=copy_owner_policy)
    emitter.factored_copy = factored_copy
    return emitter.run()
