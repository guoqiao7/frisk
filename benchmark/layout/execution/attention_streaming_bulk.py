"""Checked KV128 streaming with one co-published K/V tile and one PV group."""

from dataclasses import replace

from attention_residual_score import _ResidualScoreAttention
from attention_tile_mma import FRAGMENT_SLOTS, ROOT_BYTES, SHARED_ALIGNMENT, TILE, _checked_descriptors, _register_group
from native_static_support import STATIC_SHARED_LIMIT_BYTES
from tiled_attention import _FP_ZERO, _TiledAttention


class _StreamingBulkSchedule:
    def _configure_streaming_storage(self):
        prior = self.mma_tiles
        q = prior.q_offset
        k = prior.k_offset
        v = k + 2 * ROOT_BYTES
        spans = sorted(((q, q + ROOT_BYTES), (k, k + 2 * ROOT_BYTES),
                        (v, v + 2 * ROOT_BYTES)))
        end = spans[-1][1]
        if (self.kv_width != 2 * TILE
                or any(begin % SHARED_ALIGNMENT for begin, _ in spans)
                or any(first[1] > second[0] for first, second in zip(spans, spans[1:]))
                or end > STATIC_SHARED_LIMIT_BYTES):
            raise ValueError('streaming KV128 requires disjoint bounded Q/K128/V128 storage')
        value_descriptors = tuple(_checked_descriptors(v + block * ROOT_BYTES, k_major=False)
                                  for block in range(2))
        old = f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{prior.shared_bytes}];'
        if self.decl.count(old) != 1:
            raise ValueError('streaming storage requires one checked Shared declaration')
        self.decl[self.decl.index(old)] = f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{end}];'
        self.mma_tiles = replace(prior, v_offset=v, shared_bytes=end, _v=value_descriptors[0])
        self.bulk_value_descriptors = value_descriptors

    def _wide_scores(self, tile_row, *, copy_v=False):
        if not copy_v:
            return super()._wide_scores(tile_row)
        # The prior PV group has drained wait0 before any K or V overwrite.
        self.emit('bar.sync 0;')
        second_row = self._value('add.u32', tile_row, TILE)
        for block, row in enumerate((tile_row, second_row)):
            self._copy(self.roots[1], row, self.mma_tiles.k_offset + block * ROOT_BYTES, guard=False)
            self._copy(self.roots[2], row, self.mma_tiles.v_offset + block * ROOT_BYTES, guard=False)
        self._publish()
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(2 * FRAGMENT_SLOTS)]
        self.mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        scaled = [self._value('mul.rn.f32', score, '0f3e000000', kind='f32') for score in scores]
        return second_row, scaled[:FRAGMENT_SLOTS], scaled[FRAGMENT_SLOTS:]

    def _packed_probabilities(self, scores, maxima, sums, prefix):
        guards = self._guards(prefix)
        exponents = self._exponents(scores, maxima, guards)

        def normalized(slot):
            active = guards[slot]
            if active is False:
                return self.zero
            denominator = sums[self.row_index[slot]]
            if active is True:
                return self._value('div.rn.f32', exponents[slot], denominator, kind='f32')
            result = self._value('mov.f32', _FP_ZERO, kind='f32')
            self.emit(f'@{active} div.rn.f32 {result}, {exponents[slot]}, {denominator};')
            return result

        packed = []
        for slot in range(0, FRAGMENT_SLOTS, 2):
            low, high = normalized(slot), normalized(slot + 1)
            packed.append(self._value('cvt.rn.f16x2.f32', high, low))
        return packed

    def _emit_bulk_pv(self, packed, accum):
        packed = _register_group(packed, FRAGMENT_SLOTS)
        outputs = _register_group(accum, FRAGMENT_SLOTS)
        if set(outputs).intersection(packed):
            raise ValueError('streaming PV inputs alias their FP32 accumulator')
        args = []
        # Every packed RS input is defined before this group and remains
        # unchanged until wait0. The eight K16 atoms keep ascending key order.
        for block, descriptors in enumerate(self.bulk_value_descriptors):
            for atom, descriptor in enumerate(descriptors):
                value = descriptor.emit(self.emit, self.tmp, self.smem_base)
                begin = block * (FRAGMENT_SLOTS // 2) + atom * 4
                source = '{' + ', '.join(packed[begin:begin + 4]) + '}'
                args.append((source, value, '1'))
        self.mma_tiles._emit_group(self.emit, self.tmp, outputs, args)

    def _value_pass(self, maxima, sums):
        accum = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(FRAGMENT_SLOTS)]

        def full(_, row):
            _, left, right = self._wide_scores(row, copy_v=True)
            packed = (self._packed_probabilities(left, maxima, sums, TILE)
                      + self._packed_probabilities(right, maxima, sums, TILE))
            self._emit_bulk_pv(packed, accum)

        self._full_loop('row_value', full)
        if self.residual_row is not None:
            scores = self._scores(self.residual_row, prefix=TILE, copy_v=True)
            self.mma_tiles.emit_pv(self.emit, self.tmp, self.smem_base,
                                   self._packed_probabilities(scores, maxima, sums, TILE), accum)
        if self.tail_prefix:
            scores = self._tail_scores(copy_v=True)
            self.mma_tiles.emit_pv(self.emit, self.tmp, self.smem_base,
                                   self._packed_probabilities(scores, maxima, sums, self.tail_prefix), accum)
        return accum

    def run(self):
        text, info = super().run()
        info.update(shared_v_reuse='disjoint-q-k128-v128',
                    streaming_value_publication='K128-and-V128-before-QK',
                    streaming_pv_atoms=8, streaming_pv_issue_groups=1,
                    streaming_value_root_offsets=[self.mma_tiles.v_offset,
                                                  self.mma_tiles.v_offset + ROOT_BYTES])
        if info.get('retain_residual_scores'):
            info['residual_shared_publication'] = (
                'K0 complete plus K1 guarded; QK wait0 before disjoint V64 publication')
        return text, info


class _StreamingBulkAttention(_StreamingBulkSchedule, _TiledAttention):
    pass


class _StreamingResidualBulkAttention(_StreamingBulkSchedule, _ResidualScoreAttention):
    pass


def emit_streaming_bulk_attention(spec, *, tile_packet, vector_store=True, copy_owner_policy='gf2',
                                   retain_residual_scores=False, factored_copy=True,
                                   narrow_tail=False, simplify_tail_copy=False):
    if any(type(flag) is not bool for flag in
           (retain_residual_scores, factored_copy, narrow_tail, simplify_tail_copy)):
        raise ValueError('streaming schedule flags must be boolean')
    if retain_residual_scores:
        emitter = _StreamingResidualBulkAttention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                                  copy_owner_policy=copy_owner_policy)
    else:
        emitter = _StreamingBulkAttention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                          copy_owner_policy=copy_owner_policy, kv_width=2 * TILE,
                                          retain_tail_scores=True)
    if retain_residual_scores and narrow_tail:
        raise ValueError('residual caching consumes one complete KV128 carrier')
    emitter.factored_copy = factored_copy
    emitter.narrow_tail = narrow_tail
    emitter.simplify_tail_copy = simplify_tail_copy
    emitter._configure_streaming_storage()
    return emitter.run()

