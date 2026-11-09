"""Reuse the final K64 epoch across the checked aligned max-to-sum boundary."""

import re

from attention_tile_mma import ROOT_BYTES, TILE
from tiled_attention import _TiledAttention


POLICIES = ('boundary-control', 'boundary-reuse')
_LOOP_PHASES = ('row_max', 'row_sum', 'row_value')
_WAIT0 = 'wgmma.wait_group.sync.aligned 0;'
_BAR = 'bar.sync 0;'
_PROXY = 'fence.proxy.async.shared::cta;'
_BEGIN = 'kstage_boundary_sum_last_begin'
_PUBLISH = 'kstage_boundary_sum_last_publish'
_QK = 'kstage_boundary_sum_last_qk'
_END = 'kstage_boundary_sum_last_end'


def prove_boundary_sum(groups):
    """Verify the last leaf's carry against the ordered, left-complete tree."""
    if type(groups) is not int or not 2 <= groups <= 16:
        raise ValueError('boundary K reuse requires 2 to 16 complete KV64 groups')

    def canonical(first, count):
        if count == 1:
            return first
        split = 1 << ((count - 1).bit_length() - 1)
        return canonical(first, split), canonical(first + split, count - split)

    stack = [None] * groups.bit_length()
    last = groups - 1
    for tile in range(last):
        value, level = tile, 0
        while tile & (1 << level):
            value = (stack[level], value)
            level += 1
        stack[level] = value

    store_level = (last ^ (last + 1)).bit_length() - 1
    carries = tuple(range(store_level))
    value = last
    for level in carries:
        value = (stack[level], value)
    stack[store_level] = value

    tree = None
    for level, left in enumerate(stack):
        if groups & (1 << level):
            tree = left if tree is None else (left, tree)
    if tree != canonical(0, groups):
        raise ValueError('boundary reuse changed the ordered sum operand tree')
    return dict(last_group=last, carry_levels=carries, store_level=store_level, ordered_tree=tree)


def _counts(code):
    return dict(barriers=code.count(_BAR), proxy_fences=code.count(_PROXY),
                wgmma_wait0=code.count(_WAIT0),
                wgmma_atoms=sum(line.startswith('wgmma.mma_async.') for line in code),
                global_load_sites=sum('ld.global.v4.b32 ' in line for line in code),
                shared_store_sites=sum('st.shared.v4.b32 ' in line for line in code))


def _ordered_loop(code, phase, iterations):
    induction = re.fullmatch(r'mov\.u32 ([^ ,]+), 0;', code[0])
    if not induction or len(code) < 6:
        raise ValueError('boundary K reuse requires the original KV64 loop induction')
    tile = induction[1]
    branch = re.fullmatch(r'@([^ ]+) bra\.uni ' + phase + r'_loop;', code[-1])
    if (not branch or code[1] != phase + '_loop:'
            or not re.fullmatch(r'shl\.b32 [^ ,]+, ' + re.escape(tile) + r', 6;', code[2])
            or code[-3] != f'add.u32 {tile}, {tile}, 1;'
            or code[-2] != f'setp.lt.u32 {branch[1]}, {tile}, {iterations};'):
        raise ValueError('boundary K reuse requires a forward, bounded KV64 loop')


class _BoundaryKAttention(_TiledAttention):
    def __init__(self, spec, *, policy, tile_packet, copy_owner_policy, vector_store, factored_copy):
        if policy not in POLICIES:
            raise ValueError('unknown checked boundary K-stage policy')
        super().__init__(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                         vector_store=vector_store, factored_copy=factored_copy)
        if self.tail_prefix or self.residual_row is not None:
            raise ValueError('boundary K-stage policy requires aligned KV64')
        self.plan = prove_boundary_sum(self.group_count)
        self.policy = policy
        self.loops = {}
        self.events = []
        offsets = self.mma_tiles
        self.shared_roots = dict(q=offsets.q_offset, k=offsets.k_offset, v=offsets.v_offset)
        if tuple(self.shared_roots.values()) != (0, ROOT_BYTES, 2 * ROOT_BYTES):
            raise ValueError('boundary K-stage policy requires unchanged disjoint Q/K/V roots')

    def _full_loop(self, label, body, *, iterations=None):
        if len(self.loops) >= len(_LOOP_PHASES) or label != _LOOP_PHASES[len(self.loops)]:
            raise ValueError('boundary K-stage loop order changed')
        if iterations != (self.plan['last_group'] if label == 'row_sum' else None):
            raise ValueError('boundary K-stage loop does not cover the expected key interval')
        trips = self.group_count if iterations is None else iterations
        start = len(self.code)
        super()._full_loop(label, body, iterations=iterations)
        stop = len(self.code)
        code = self.code[start:stop]
        _ordered_loop(code, label, trips)
        counts = _counts(code)
        groups = 2 if label == 'row_value' else 1
        expected = dict(barriers=2, proxy_fences=1, wgmma_wait0=groups, wgmma_atoms=4 * groups,
                        global_load_sites=4 * groups, shared_store_sites=4 * groups)
        if counts != expected or code.index(_WAIT0) <= code.index(_PROXY):
            raise ValueError('boundary K-stage loop no longer publishes and drains the expected work')
        self.loops[label] = dict(kind='loop', phase=label, label=label + '_loop',
                                 code_span=[start, stop], trip_count=trips, **counts)
        self.events.append(self.loops[label])

    def _sum_boundary_scores(self):
        if list(self.loops) != ['row_max']:
            raise ValueError('boundary K-stage read is outside the completed max epoch')
        after_max = self.code[self.loops['row_max']['code_span'][1]:]
        if any(_counts(after_max).values()):
            raise ValueError('unexpected writes or asynchronous work after max before K reuse')
        last = self.plan['last_group']
        self.emit(_BEGIN + ':')
        start = len(self.code)
        publish = self.policy == 'boundary-control'
        if publish:
            self.emit(_PUBLISH + ':')
            self.emit(_BAR)
            self._copy(self.roots[1], last * TILE, self.mma_tiles.k_offset, guard=False)
            self._publish()
        self.emit(_QK + ':')
        qk_start = len(self.code)
        scores = self._compute_scores(TILE, self.mma_tiles)
        stop = len(self.code)
        self.emit(_END + ':')
        counts = _counts(self.code[start:stop])
        qk_counts = _counts(self.code[qk_start:stop])
        if (qk_counts != dict(barriers=0, proxy_fences=0, wgmma_wait0=1, wgmma_atoms=4,
                              global_load_sites=0, shared_store_sites=0)
                or counts != dict(qk_counts, barriers=2 * publish, proxy_fences=int(publish),
                                  global_load_sites=4 * publish, shared_store_sites=4 * publish)):
            raise ValueError('boundary K-stage read no longer matches the checked publication contract')
        self.events.append(dict(kind='boundary', phase='row_sum', label=_BEGIN,
                                code_span=[start, stop], trip_count=1, logical_group=last,
                                k_offset=self.mma_tiles.k_offset, **counts))
        return scores, self.plan

    def run(self):
        text, info = super().run()
        groups = self.group_count
        publish = int(self.policy == 'boundary-control')
        k_by_phase = dict(row_max=groups, row_sum=groups - 1 + publish, row_value=groups)
        k_total = sum(k_by_phase.values())
        count_names = ('barriers', 'proxy_fences', 'wgmma_wait0', 'wgmma_atoms',
                       'global_load_sites', 'shared_store_sites')
        total = {key: sum(event[key] * event['trip_count'] for event in self.events) for key in count_names}
        full_counts = _counts(self.code)
        if (full_counts['barriers'] != sum(e['barriers'] for e in self.events)
                or full_counts['proxy_fences'] != sum(e['proxy_fences'] for e in self.events)
                or full_counts['wgmma_wait0'] != sum(e['wgmma_wait0'] for e in self.events)
                or full_counts['wgmma_atoms'] != sum(e['wgmma_atoms'] for e in self.events)
                or total['global_load_sites'] != 4 * (k_total + groups)
                or total['shared_store_sites'] != 4 * (k_total + groups)
                or total['barriers'] != 2 * k_total or total['proxy_fences'] != k_total
                or total['wgmma_wait0'] != 4 * groups or total['wgmma_atoms'] != 16 * groups):
            raise ValueError('boundary K-stage dynamic work differs from checked epochs')
        info['k_stage_reuse'] = dict(
            policy=self.policy, events=self.events, shared_roots=self.shared_roots,
            max_loop_label=self.loops['row_max']['label'], sum_loop_label=self.loops['row_sum']['label'],
            boundary_begin_label=_BEGIN, boundary_publish_label=_PUBLISH if publish else None,
            boundary_qk_label=_QK, boundary_end_label=_END,
            k64_publications=k_total, k_publications_by_phase=k_by_phase, v_publications=groups,
            kv_global_bytes=(k_total + groups) * ROOT_BYTES,
            kv_shared_store_bytes=(k_total + groups) * ROOT_BYTES,
            ordered_tree=self.plan['ordered_tree'],
            sum_leaf_evaluation_order=[groups - 1, *range(groups - 1)],
            sum_leaf_insert_order=list(range(groups)), **total)
        return text, info


def emit_boundary_k_attention(spec, *, policy, tile_packet, copy_owner_policy,
                              vector_store, factored_copy):
    return _BoundaryKAttention(spec, policy=policy, tile_packet=tile_packet,
        copy_owner_policy=copy_owner_policy, vector_store=vector_store,
        factored_copy=factored_copy).run()
