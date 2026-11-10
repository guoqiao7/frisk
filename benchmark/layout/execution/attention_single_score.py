"""Single QK evaluation for a checked, complete short Attention score carrier."""

from functools import lru_cache
import hashlib

from access_plan import linear
from attention_fragment_layout import NATIVE_FRAGMENT
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

    def __init__(self, spec, *, tile_packet, vector_store, copy_owner_policy, tail_score_carrier=False):
        if type(tail_score_carrier) is not bool or type(spec) is not dict or type(spec.get('N')) is not int:
            raise ValueError('single-score Attention requires a checked shape and carrier strategy')
        length = spec['N']
        if tail_score_carrier:
            if not TILE < length < 2 * TILE:
                raise ValueError('padded KV128 score carrier requires one full KV64 and a nonempty tail')
        elif length not in (TILE, 2 * TILE, 4 * TILE):
            raise ValueError('single-score Attention supports aligned N64, N128 and N256')
        super().__init__(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                         kv_width=TILE, retain_tail_scores=tail_score_carrier, vector_store=vector_store)
        previous_shared = self.mma_tiles.shared_bytes
        self.kv_width = 2 * TILE if tail_score_carrier else length
        self.mma_tiles = build_d64_wgmma_tiles(kv_tile=self.kv_width)
        self.score_rows = single_score_rows(self.kv_width, self.mma_tiles.score_fragment_columns)
        self.score_masks = ()
        if tail_score_carrier:
            from attention_residual_score import residual_score_holders

            if self.fragment.name != NATIVE_FRAGMENT:
                raise ValueError('padded KV128 requires the native score and PV fragments')
            self.score_masks = residual_score_holders(length, self.mma_tiles.score_fragment_columns)
            if self.score_masks[FRAGMENT_SLOTS:] != self.score_prefix.holder_masks:
                raise ValueError('padded KV128 score domain differs from the emitted key predicate')
        old = f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{previous_shared}];'
        if self.decl.count(old) != 1:
            raise ValueError('single-score storage requires one checked Shared allocation')
        self.decl[self.decl.index(old)] = (
            f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{self.mma_tiles.shared_bytes}];')

    def _key_prefix(self, block):
        return min(TILE, self.length - block * TILE)

    def _copy_key_rows(self, root, block, offset):
        prefix = self._key_prefix(block)
        if prefix == TILE:
            self._copy(root, block * TILE, offset, guard=False)
        else:
            self._copy(root, block * TILE, offset, guard=True, prefix=prefix)

    def _all_scores(self):
        self._copy(self.roots[0], self.query_row, self.mma_tiles.q_offset, guard=bool(self.tail_prefix))
        if not self.score_masks:
            self.emit('bar.sync 0;')
        # Q and both K roots are disjoint. The final publication orders all three.
        for block in range(self.kv_width // TILE):
            self._copy_key_rows(self.roots[1], block, self.mma_tiles.k_offset + block * ROOT_BYTES)
        self._publish()
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(self.kv_width // 2)]
        self.mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        return [self._value('mul.rn.f32', value, '0f3e000000', kind='f32')
                if not self.score_masks or self.score_masks[slot] else self.zero
                for slot, value in enumerate(scores)]

    def _statistics(self, scores):
        maxima = [self._value('mov.f32', _NEG_INF, kind='f32') for _ in self.rows]
        guards = []
        # The wide lane tree factors into the original KV64 subtrees, including
        # the same guarded merge with the pruned tail before the lane butterfly.
        for block in range(self.kv_width // TILE):
            block_guards = self._guards(self._key_prefix(block))
            guards.extend(block_guards)
            start = block * FRAGMENT_SLOTS
            masked = [(_NEG_INF if guard is False else score if guard is True else
                       self._value('selp.b32', score, _NEG_INF, guard))
                      for score, guard in zip(scores[start:start + FRAGMENT_SLOTS], block_guards, strict=True)]
            for maximum, slots in zip(maxima, self.rows, strict=True):
                local, _ = self._tree('max.f32', [masked[slot] for slot in slots])
                self.emit(f'max.f32 {maximum}, {maximum}, {local};')
        maxima = [self._warp_reduce('max.f32', value, ordered=False) for value in maxima]
        exponent = []
        for begin in range(0, len(scores), FRAGMENT_SLOTS):
            end = begin + FRAGMENT_SLOTS
            exponent.extend(self._exponents(scores[begin:end], maxima, guards[begin:end]))
        sums = []
        for slots in self.score_rows:
            local, _ = self._tree('add.rn.f32', [exponent[slot] for slot in slots],
                                  [guards[slot] for slot in slots])
            sums.append(self._warp_reduce('add.rn.f32', local, ordered=True))
        return exponent, sums

    def _pack_probabilities(self, exponent, sums, block):
        guards = self._guards(self._key_prefix(block))
        begin = block * FRAGMENT_SLOTS
        packed = []
        # Odd prefixes can have a valid low half and an invalid high half.
        # Normalize each pair immediately so its FP32 temporaries die early.
        for slot in range(0, FRAGMENT_SLOTS, 2):
            pair = []
            for index in (slot, slot + 1):
                guard = guards[index]
                exp, denom = exponent[begin + index], sums[self.row_index[index]]
                if guard is False:
                    value = self.zero
                elif guard is True:
                    value = self._value('div.rn.f32', exp, denom, kind='f32')
                else:
                    value = self._value('mov.f32', _FP_ZERO, kind='f32')
                    self.emit(f'@{guard} div.rn.f32 {value}, {exp}, {denom};')
                pair.append(value)
            packed.append(self._value('cvt.rn.f16x2.f32', pair[1], pair[0]))
        return packed

    def _value_groups(self):
        return tuple((self.mma_tiles.v_offset,) for _ in range(self.mma_tiles.kv_tile // TILE))

    def _consume_values(self, exponent, sums):
        accum = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(FRAGMENT_SLOTS)]
        for block in range(self.kv_width // TILE):
            self._publish_values(block * TILE, self._key_prefix(block))
            packed = self._pack_probabilities(exponent, sums, block)
            self.mma_tiles.emit_pv(self.emit, self.tmp, self.smem_base, packed, accum)
        return accum

    def _tail_schedule_evidence(self):
        bar = 'bar.sync 0;'
        proxy = 'fence.proxy.async.shared::cta;'
        fence = 'wgmma.fence.sync.aligned;'
        commit = 'wgmma.commit_group.sync.aligned;'
        wait = 'wgmma.wait_group.sync.aligned 0;'
        store = 'st.shared.v4.b32'
        qk = 'wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16'
        pv = 'wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16'

        operations = []
        for line in self.code:
            opcode = line.split()[0]
            if line in (bar, proxy) or line.startswith('wgmma.') or 'st.shared.' in line:
                operations.append(opcode if line.startswith('wgmma.mma_async.') or opcode == store else line)
        groups = self._value_groups()
        expected = [store] * 12 + [bar, proxy, fence] + [qk] * 4 + [commit, wait]
        for roots in groups:
            expected.extend([bar] + [store] * (4 * len(roots)) + [bar, proxy, fence]
                            + [pv] * (4 * len(roots)) + [commit, wait])
        if operations != expected:
            raise ValueError('padded KV128 requires QK wait0 before V overwrite and published V before PV')
        k = self.mma_tiles.k_offset
        return dict(k_publications=1, qk_atoms=4, k_offsets=[k, k + ROOT_BYTES],
                    v_offsets=[offset for group in groups for offset in group],
                    wgmma_wait0=operations.count(wait), barriers=operations.count(bar),
                    proxy_fences=operations.count(proxy), kv_shared_store_bytes=4 * ROOT_BYTES,
                    kv_global_bytes=2 * self.length * TILE * 2)

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
                    execution_kv_width=self.kv_width, retain_tail_score_slots=[],
                    retain_all_scores=True, vector_store=self.vector_store,
                    score_evaluations=1, score_cache_slots=self.kv_width // 2,
                    score_cache_transition='scaled-qk-to-exponent',
                    shared_v_reuse='last-k64-after-wait0' if self.kv_width > TILE else 'disjoint',
                    copy_owner_policy=self.copy_owner_policy,
                    copy_lane_bit_order=self.copy_plans[TILE].lane_bit_order,
                    passes=['global-max', 'canonical-row-sum', 'normalized-fp16-pv'],
                    reduction_order='hardware-first-owner-v1 with absent leaves pruned',
                    p_rounding='cvt.rn.f16x2.f32 after global normalization',
                    ftz=False, fp_fusion=False, mask='query-and-kv-prefix',
                    tail_copy_active_groups=[], tail_store_elements=0,
                    ttgir_sha256=hashlib.sha256(text.encode()).hexdigest())
        if self.score_masks:
            schedule = self._tail_schedule_evidence()
            groups = self._value_groups()
            info.update(tail_score_carrier=True, score_valid_prefix=self.length,
                        score_tail_prefix=self.tail_prefix,
                        score_active_slots=sum(bool(mask) for mask in self.score_masks),
                        score_holder_counts=[mask.bit_count() for mask in self.score_masks],
                        value_publications=len(groups), pv_issue_groups=len(groups), pv_atoms=self.kv_width // 16,
                        shared_v_offsets=schedule['v_offsets'], score_tail_schedule=schedule,
                        factored_copy=self.factored_copy,
                        tail_copy_active_groups=[g.active_mask.bit_count()
                                                 for g in self.copy_plans[self.tail_prefix].groups],
                        tail_store_elements=sum(g.active_mask.bit_count()
                                                for g in self.store_plans[self.tail_prefix].groups))
        info.update(self._fragment_evidence())
        check_static_shared(info)
        return text, info


def emit_single_score_attention(spec, *, tile_packet, vector_store=True, copy_owner_policy='gf2',
                                factored_copy=False, tail_score_carrier=False):
    if type(factored_copy) is not bool:
        raise ValueError('copy schedule flag must be boolean')
    emitter = _SingleScoreAttention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                   copy_owner_policy=copy_owner_policy, tail_score_carrier=tail_score_carrier)
    emitter.factored_copy = factored_copy
    return emitter.run()
