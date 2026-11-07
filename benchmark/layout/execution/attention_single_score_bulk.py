"""Bulk V publication and one checked PV group for a complete score carrier."""

from attention_single_score import _SingleScoreAttention
from attention_tile_mma import FRAGMENT_SLOTS, ROOT_BYTES, TILE, _checked_descriptors, _register_group


class _SingleScoreBulkAttention(_SingleScoreAttention):
    def _value_roots(self):
        if self.length == TILE:
            return (self.mma_tiles.v_offset,)
        roots = tuple(self.mma_tiles.k_offset + block * ROOT_BYTES
                      for block in range(self.length // TILE))
        q = self.mma_tiles.q_offset
        if (roots[-1] + ROOT_BYTES > self.mma_tiles.shared_bytes
                or any(root < q + ROOT_BYTES and q < root + ROOT_BYTES for root in roots)):
            raise ValueError('bulk V publication exceeds checked K storage or aliases Q')
        return roots

    def _consume_values(self, exponent, sums):
        roots = self._value_roots()
        # QK has drained wait0. Publish the complete V footprint in the
        # no-longer-live K storage before any PV instruction is issued.
        self.emit('bar.sync 0;')
        for block, root in enumerate(roots):
            self._copy(self.roots[2], block * TILE, root, guard=False)
        self._publish()

        packed = []
        for block in range(self.length // TILE):
            begin = block * FRAGMENT_SLOTS
            for slot in range(0, FRAGMENT_SLOTS, 2):
                low = self._value('div.rn.f32', exponent[begin + slot],
                                  sums[self.row_index[slot]], kind='f32')
                high = self._value('div.rn.f32', exponent[begin + slot + 1],
                                   sums[self.row_index[slot + 1]], kind='f32')
                packed.append(self._value('cvt.rn.f16x2.f32', high, low))
        # All RS inputs are distinct and ready before the group. Their
        # definitions precede every issue, and no input is overwritten before
        # the final wait0. The key order is the original ascending K16 order.
        packed = _register_group(packed, self.length // 4)
        accum = _register_group(
            [self._value('mov.f32', '0f00000000', kind='f32') for _ in range(FRAGMENT_SLOTS)],
            FRAGMENT_SLOTS)
        if set(packed).intersection(accum):
            raise ValueError('bulk PV input aliases its FP32 accumulator')
        args = []
        for block, root in enumerate(roots):
            for atom, descriptor in enumerate(_checked_descriptors(root, k_major=False)):
                value = descriptor.emit(self.emit, self.tmp, self.smem_base)
                begin = block * (FRAGMENT_SLOTS // 2) + atom * 4
                source = '{' + ', '.join(packed[begin:begin + 4]) + '}'
                args.append((source, value, '1'))
        self.mma_tiles._emit_group(self.emit, self.tmp, accum, args)
        return accum

    def run(self):
        text, info = super().run()
        info.update(shared_v_reuse='all-k64-after-wait0' if self.length > TILE else 'disjoint',
                    shared_v_offsets=list(self._value_roots()),
                    value_publications=1, pv_issue_groups=1, pv_atoms=self.length // 16,
                    single_score_value_schedule='bulk-published-native-rs-v1')
        return text, info


def emit_single_score_bulk_attention(spec, *, tile_packet, vector_store=True, copy_owner_policy='gf2', factored_copy=False):
    if type(factored_copy) is not bool:
        raise ValueError('copy schedule flag must be boolean')
    emitter = _SingleScoreBulkAttention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                   copy_owner_policy=copy_owner_policy)
    emitter.factored_copy = factored_copy
    return emitter.run()
