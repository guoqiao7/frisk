"""Checked KV64 K residency with delayed V publication for one full tile and a tail."""

from dataclasses import replace

from attention_tile_mma import ROOT_BYTES, TILE, _checked_descriptors
from tiled_attention import _TiledAttention


SHORT_POLICIES = ('late-v-control', 'resident-k')
_PHASES = ('row_max', 'row_sum', 'row_value')
_WAIT0 = 'wgmma.wait_group.sync.aligned 0;'
_BAR = 'bar.sync 0;'
_PROXY = 'fence.proxy.async.shared::cta;'


def _check_short_order(events, resident):
    expected = [
        ('row_max', 'publish', 'k0'), ('row_max', 'qk', 'k0'),
        ('row_max', 'publish', 'ktail'), ('row_max', 'qk', 'ktail'),
    ]
    for phase in _PHASES[1:]:
        if not resident:
            expected.append((phase, 'publish', 'k0'))
        expected.append((phase, 'qk', 'k0'))
    expected.extend((('row_value', 'publish', 'v0'), ('row_value', 'pv', 'v0'),
                     ('row_value', 'publish', 'vtail'), ('row_value', 'pv', 'vtail')))
    if [(e['phase'], e['kind'], e['root']) for e in events] != expected:
        raise ValueError('short Attention K/V epoch order changed')
    for earlier, later in zip(events, events[1:]):
        if earlier['code_span'][1] > later['code_span'][0]:
            raise ValueError('short Attention epoch instructions overlap')
    for event in events:
        if event['kind'] == 'publish':
            if event['barriers'] != 2 or event['proxy_fences'] != 1:
                raise ValueError('short Attention publication lacks CTA and proxy ordering')
        elif event['wait0'] != 1 or event['mma_atoms'] != 4:
            raise ValueError('short Attention consumer did not drain its native MMA group')


class _ShortKStageAttention(_TiledAttention):
    def __init__(self, spec, *, policy, tile_packet, copy_owner_policy, vector_store, factored_copy):
        if (policy not in SHORT_POLICIES or type(spec) is not dict or type(spec.get('N')) is not int
                or not TILE < spec['N'] < 2 * TILE):
            raise ValueError('checked short K reuse requires one full KV64 tile and a nonempty tail')
        super().__init__(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                         vector_store=vector_store, factored_copy=factored_copy,
                         retain_tail_scores=True)
        self.policy = policy
        self.phase = None
        self.events = []
        self.published_keys = set()
        self.aliased = False

        # Rebind the disjoint old V root as Ktail. V gets an MN-major
        # descriptor over K0 only after the final reader drains wait0.
        baseline = self.mma_tiles
        k, tail = baseline.k_offset, baseline.v_offset
        if (baseline.q_offset, k, tail, baseline.shared_bytes) != (0, ROOT_BYTES, 2 * ROOT_BYTES, 3 * ROOT_BYTES):
            raise ValueError('checked short K residency requires three disjoint 8 KiB roots')
        self.mma_tiles = replace(baseline, v_offset=k, _v=_checked_descriptors(k, k_major=False))
        self.tail_mma = replace(self.mma_tiles, k_offset=tail,
                                _k=_checked_descriptors(tail, k_major=True))
        self.stage_offsets = dict(q=0, k0=k, ktail=tail, v0=k, vtail=k)

    def _full_loop(self, label, body):
        expected = 0 if self.phase is None else _PHASES.index(self.phase) + 1
        if expected >= len(_PHASES) or label != _PHASES[expected] or self.group_count != 1:
            raise ValueError('checked short K stage sequence changed')
        self.phase = label
        # Exactly one iteration in both arms, with the same logical key and
        # dyadic leaf index. The base max/sum/value arithmetic stays shared.
        body(self.zero, self.zero)

    def _begin(self, kind, root):
        label = f'kstage_{len(self.events)}_{kind}_{root}'
        self.emit(label + ':')
        return label, len(self.code)

    def _record(self, token, kind, root, **detail):
        label, start = token
        end = len(self.code)
        code = self.code[start:end]
        self.events.append(dict(kind=kind, root=root, phase=self.phase, offset=self.stage_offsets[root],
            label=label, code_span=[start, end], barriers=code.count(_BAR), proxy_fences=code.count(_PROXY),
            wait0=code.count(_WAIT0), mma_atoms=sum(x.startswith('wgmma.mma_async.') for x in code),
            **detail))

    def _publish_tile(self, tile, row, prefix):
        source = self.roots[1] if tile.startswith('k') else self.roots[2]
        token = self._begin('publish', tile)
        self.emit(_BAR)
        self._copy(source, row, self.stage_offsets[tile], guard=prefix != TILE, prefix=prefix)
        self._publish()
        self._record(token, 'publish', tile, valid_rows=prefix, global_bytes=prefix * TILE * 2,
                     shared_store_bytes=ROOT_BYTES)

    def _scores(self, tile_row, *, prefix, copy_v=False):
        full = prefix == TILE and tile_row == self.zero
        tail = prefix == self.tail_prefix and tile_row == self.tail_row and self.phase == 'row_max'
        if (self.aliased or not (full or tail)
                or copy_v != (full and self.phase == 'row_value')):
            raise ValueError('short Attention QK would read outside a live K epoch')
        tile = 'k0' if full else 'ktail'
        if self.policy == 'late-v-control' or tile not in self.published_keys:
            self._publish_tile(tile, tile_row, prefix)
            self.published_keys.add(tile)
        token = self._begin('qk', tile)
        scores = self._compute_scores(prefix, self.mma_tiles if full else self.tail_mma)
        self._record(token, 'qk', tile)
        if copy_v:
            self._publish_values(tile_row, prefix)
        return scores

    def _publish_values(self, row, prefix):
        full = prefix == TILE and row == self.zero
        tail = prefix == self.tail_prefix and row == self.tail_row
        previous = self.events[-1]
        if (self.phase != 'row_value' or not (full or tail) or self.retained_tail is None
                or (full and self.aliased) or (tail and not self.aliased)
                or (previous['kind'], previous['root'], previous['wait0']) !=
                    (('qk', 'k0', 1) if full else ('pv', 'v0', 1))):
            raise ValueError('short Attention V would overwrite an undrained reader')
        self._publish_tile('v0' if full else 'vtail', row, prefix)
        self.aliased = True

    def _emit_pv(self, packed, accum, prefix):
        tile = 'v0' if prefix == TILE else 'vtail'
        if not self.aliased or (self.events[-1]['kind'], self.events[-1]['root']) != ('publish', tile):
            raise ValueError('short Attention PV requires its published V epoch')
        token = self._begin('pv', tile)
        super()._emit_pv(packed, accum, prefix)
        self._record(token, 'pv', tile)

    def run(self):
        text, info = super().run()
        events = self.events
        _check_short_order(events, self.policy == 'resident-k')
        groups = [event for event in events if event['kind'] == 'publish']
        counters = {key: self.code.count(op) for key, op in
                    (('barriers', _BAR), ('proxy_fences', _PROXY), ('wgmma_wait0', _WAIT0))}
        info['shared_v_reuse'] = 'k0-after-final-qk-wait0'
        info['k_stage_reuse'] = dict(policy=self.policy, events=events, shared_roots=self.stage_offsets,
            ktail_descriptor_fields=[(d.offset, d.fields) for d in self.tail_mma._k],
            k0_publications=sum(e['root'] == 'k0' for e in groups),
            ktail_publications=sum(e['root'] == 'ktail' for e in groups),
            v_publications=sum(e['root'].startswith('v') for e in groups),
            kv_global_bytes=sum(e['global_bytes'] for e in groups),
            kv_shared_store_bytes=sum(e['shared_store_bytes'] for e in groups),
            final_qk_label=[e['label'] for e in events if e['kind'] == 'qk'][-1],
            first_v_publish_label=next(e['label'] for e in groups if e['root'] == 'v0'),
            **counters)
        return text, info


def emit_short_k_stage_attention(spec, *, policy, tile_packet, copy_owner_policy,
                                 vector_store, factored_copy):
    return _ShortKStageAttention(spec, policy=policy, tile_packet=tile_packet,
        copy_owner_policy=copy_owner_policy, vector_store=vector_store, factored_copy=factored_copy).run()
