
"""Checked D64 BHSD forward with bounded KV storage and full-row normalization."""

import hashlib

from attention_copy_layout import (attention_copy_layout_evidence, derive_attention_copy_layouts,
                                   native_copy_plan)
from attention_tile_mma import (FRAGMENT_SLOTS, ROOT_BYTES, SHARED_ALIGNMENT, THREADS,
                                TILE, build_d64_wgmma_tiles, fragment_coordinate,
                                shared_offsets_64x64)
from attention_fragment_layout import NATIVE_FRAGMENT, checked_fragment_layout, output_store_evidence
from attention_score_reuse import prove_score_prefix
from backend import Emitter, render_inline_execution, splat_f32_bits
from native_static_support import check_static_shared
from reduction_plan import verified_warp_reduction_trees
from valid_domain import plan_guarded_store


CONTRACT = 'frisk-sm90-strict-tiled-attention-v1'
_ENTRY = 'frisk_execution'
_FP_ZERO = '0f00000000'
_NEG_INF = '0fff800000'
_SHAPE = (TILE, TILE)
_ALL_THREADS = (1 << THREADS) - 1
_FRAGMENT_REQUEST = '_checked_attention_fragment_layout'


def checked_geometry(spec, *, kv_width=TILE, tail_score_carrier=False):
    """Admit compact BHSD buffers and bind one exact query-tile/grid mapping."""
    if (type(spec) is not dict or set(spec) != {'B', 'H', 'N', 'D', 'q_block', 'kv_block'}
            or any(type(n) is not int or n <= 0 for n in spec.values())):
        raise ValueError('expected the exact tiled BHSD specification')
    batch, heads, length, dim = (spec[k] for k in ('B', 'H', 'N', 'D'))
    if (dim, spec['q_block'], spec['kv_block']) != (64, TILE, TILE) or not TILE <= length <= 1025:
        raise ValueError('checked tiled Attention requires 64 <= N <= 1025, D64 and Q64/KV64 carriers')
    if type(tail_score_carrier) is not bool:
        raise ValueError('Attention tail score carrier flag must be boolean')
    if tail_score_carrier and (kv_width != 2 * TILE or not TILE < length < 2 * TILE):
        raise ValueError('tail score carrier requires physical KV128 and 65 to 127 logical keys')
    if (type(kv_width) is not int or kv_width not in (32, TILE, TILE * 2, TILE * 4)
            or (length < kv_width and not tail_score_carrier)):
        raise ValueError('execution KV width requires a complete checked KV64 or KV128 group')
    if kv_width == 32 and length % TILE:
        raise ValueError('compact KV32 requires complete KV64 pairs')
    if kv_width == 256 and length != 256:
        raise ValueError('KV256 is checked for one complete cached score tile')
    full, remainder = divmod(length, TILE)
    tiles = full + bool(remainder)
    grid = batch * heads * tiles
    if grid >= 1 << 31:
        raise ValueError('tiled BHSD grid exceeds the 32-bit CTA launch domain')
    head_bytes = [length * dim * element for element in (2, 2, 2, 4)]
    spans = [batch * heads * size for size in head_bytes]
    if any(size >= 1 << 64 for size in spans) or any(size % 16 for size in head_bytes):
        raise ValueError('dense BHSD stride outside the aligned 64-bit ABI')
    return dict(contract=CONTRACT, grid=[grid, 1, 1], block=[THREADS, 1, 1],
                entry=_ENTRY, alignment_bytes=16, head_bytes=head_bytes, root_spans=spans,
                q_tiles=tiles, full_kv_tiles=full, final_rows=remainder,
                argument_roles=['q', 'k', 'v', 'out', 'scratch'],
                argument_dtypes=['float16', 'float16', 'float16', 'float32', None],
                dynamic_shared_bytes=0, scratch_bytes=0)


def _output_distribution():
    slots = []
    for thread in range(THREADS):
        for register in range(FRAGMENT_SLOTS):
            row, col = fragment_coordinate(thread, register)
            slots.append(row * TILE + col)
    first = [0] * len(slots)
    for holder, slot in enumerate(slots):
        first[slot] = holder
    return dict(shape=list(_SHAPE), topology=[FRAGMENT_SLOTS, 32, 4, 1, 1],
                logical=slots, first=first, replication=1)


def _checked_tile_layouts(packet):
    """Bind staged execution to Frisk's native instruction certificates."""
    witness = Emitter(packet)
    values = witness.v
    operations = witness.data['operations']
    copies = [op for op in operations if op['name'] == 'frisk.copy']
    mmas = [op for op in operations if op['name'] == 'frisk.mma']
    reductions = [op for op in operations if op['name'] == 'frisk.reduce_tensor']
    stores = [op for op in operations if op['name'] == 'frisk.tile_store']
    if (len(copies), len(mmas), len(reductions), len(stores)) != (3, 2, 2, 1):
        raise ValueError('expected the inferred D64 Attention tile graph')
    if any(op['name'] == 'frisk.convert_layout' for op in operations):
        raise ValueError('staged D64 Attention requires the inferred coherent fragment')
    distribution = None
    expected_copy = dict(version=1, path='vector', target='sm_90a', vector_bytes=16,
                         valid_shape=list(_SHAPE), guarded=0, issuer='execution-map',
                         completion='synchronous')

    def offsets(value, expected, *, alignment):
        actual = witness.offsets(value)
        if (witness.storage(value)['alignment'] < alignment or len(actual) != len(expected)
                or any(actual.get(i) != offset for i, offset in enumerate(expected))):
            raise ValueError('inferred tile byte layout differs from the staged ABI')

    copy_targets = []
    for op in copies:
        source, target = op['operands'][:2]
        execution = op.get('execution')
        if (op['attrs'].get('frisk.copy_contract') != expected_copy
                or op['attrs'].get('frisk.writer_policy') != 'first_owner'
                or type(execution) is not dict or execution.get('shape') != list(_SHAPE)
                or execution.get('topology') != [TILE * TILE // THREADS, 32, 4, 1, 1]
                or execution.get('replication', 1) != 1):
            raise ValueError('inferred Copy16 ownership differs from the staged layout')
        if distribution is None:
            distribution = execution
        elif any(execution.get(key) != distribution.get(key) for key in ('logical', 'first')):
            raise ValueError('staged Attention requires one tied Q/K/V Copy16 owner map')
        offsets(source, range(0, ROOT_BYTES, 2), alignment=16)
        offsets(target, shared_offsets_64x64(), alignment=SHARED_ALIGNMENT)
        copy_targets.append(target)

    qk, pv = mmas
    if (qk['operands'][:2] != copy_targets[:2] or pv['operands'][1] != copy_targets[2]
            or stores[0]['operands'][0] != pv['results'][0]):
        raise ValueError('inferred D64 graph has different Q/K/V or output bindings')
    native = _output_distribution()

    def same_fragment(value):
        inferred = values[value].get('distributed', {})
        if any(inferred.get(k) != v for k, v in native.items()):
            raise ValueError('inferred fragment ownership differs from the D64 GF(2) witness')

    for value in (*qk['results'], *pv['results'], qk['operands'][2], pv['operands'][0], pv['operands'][2]):
        same_fragment(value)
    definitions = {result: op for op in operations for result in op['results']}

    def defining(value, name):
        op = definitions.get(value)
        if (op is None or op['name'] != name
                or op['attrs'].get('fastmath', '#arith.fastmath<none>') != '#arith.fastmath<none>'):
            raise ValueError('inferred tile does not match the exact Attention arithmetic')
        return op

    def constant(value, bits):
        if splat_f32_bits(defining(value, 'arith.constant')['attrs']['value']) != bits:
            raise ValueError('inferred Attention tile uses a different scalar constant')

    def broadcast(value, reduction):
        op = defining(value, 'linalg.broadcast')
        if op['attrs'].get('dimensions') != [1] or op['operands'][0] != reduction['results'][0]:
            raise ValueError('inferred Attention broadcast does not match its row statistic')

    for op in mmas:
        constant(op['operands'][2], 0)
    scale, = reductions[0]['operands']
    exp, = reductions[1]['operands']
    qk_acc, scale_constant = defining(scale, 'arith.mulf')['operands']
    constant(scale_constant, 0x3e000000)
    centered, = defining(exp, 'math.exp')['operands']
    score, max_broadcast = defining(centered, 'arith.subf')['operands']
    quotient, = defining(pv['operands'][0], 'arith.truncf')['operands']
    numerator, sum_broadcast = defining(quotient, 'arith.divf')['operands']
    if (qk_acc != qk['results'][0] or score != scale or numerator != exp
            or any(op['attrs']['dim'] != 1 for op in reductions)
            or [(op['attrs']['trans_a'], op['attrs']['trans_b']) for op in mmas] != [(0, -1), (0, 0)]):
        raise ValueError('inferred tile dataflow does not preserve the staged Attention contract')
    broadcast(max_broadcast, reductions[0])
    broadcast(sum_broadcast, reductions[1])
    for op in reductions:
        source, = op['operands']
        result, = op['results']
        same_fragment(source)
        verified_warp_reduction_trees(op, values[source]['distributed'], values[result]['distributed'])
    if [op['attrs']['kind'].strip('"') for op in reductions] != ['max', 'sum']:
        raise ValueError('inferred Attention tile must reduce max before sum')

    def descriptor(major, leading, step):
        return dict(major=major, leading=leading, stride=1024, swizzle=128,
                    entries=[value for k in range(0, TILE, 16) for value in (0, k, k * step, 0)])

    common = dict(version=1, target='sm_90a', atom=[64, 64, 16], grid=[1, 1], repeats=[1, 1, 4],
                  input_type='f16', accumulator_type='f32')
    kd = descriptor('k', 16, 2)
    expected = (dict(common, form='ss', packing='none', a_descriptor=kd, b_descriptor=kd),
                dict(common, form='rs', packing='f16x2-low-high',
                     b_descriptor=descriptor('mn', 1024, 128)))
    if tuple(op['attrs'].get('frisk.mma_contract') for op in mmas) != expected:
        raise ValueError('inferred QK/PV descriptors differ from the staged WGMMA proof')
    offsets(stores[0]['operands'][1], range(0, TILE * TILE * 4, 4), alignment=16)
    return distribution, values[pv['results'][0]]['distributed']


def _row_fibers():
    """Prove that native hardware order factors into two per-thread rows."""
    row0 = fragment_coordinate(0, 0)[0]
    rows = tuple(tuple(r for r in range(FRAGMENT_SLOTS) if fragment_coordinate(0, r)[0] == row)
                 for row in (row0, row0 + 8))
    row_index = {slot: half for half, slots in enumerate(rows) for slot in slots}
    columns = [fragment_coordinate(0, reg)[1] for reg in range(FRAGMENT_SLOTS)]
    if sorted(row_index) != list(range(FRAGMENT_SLOTS)) or any(len(row) != TILE // 4 for row in rows):
        raise ValueError('D64 carrier does not partition into two reduction fibers')
    for thread in range(THREADS):
        for half, slots in enumerate(rows):
            peer = thread & ~3
            want_row = fragment_coordinate(peer, slots[0])[0]
            for reg in slots:
                row, col = fragment_coordinate(thread, reg)
                if row != want_row or col != columns[reg] + (thread & 3) * 2:
                    raise ValueError('D64 fragment does not preserve ordered four-lane reduction fibers')
    return rows, row_index, columns


class _TiledAttention(Emitter):
    def __init__(self, spec, *, tile_packet, copy_owner_policy, kv_width=TILE,
                 retain_tail_scores=False, vector_store=False, narrow_tail=False,
                 simplify_tail_copy=False, factored_copy=False, shared_swizzle=128):
        super().__init__({'values': [], 'operations': []}, copy_owner_policy=copy_owner_policy)
        if any(type(flag) is not bool for flag in
               (retain_tail_scores, vector_store, narrow_tail, simplify_tail_copy, factored_copy)):
            raise ValueError('checked Attention strategies require boolean flags')
        self.geometry = checked_geometry(spec, kv_width=kv_width)
        self.length = spec['N']
        self.kv_width = kv_width
        self.vector_store = vector_store
        self.narrow_tail = narrow_tail
        self.simplify_tail_copy = simplify_tail_copy
        self.factored_copy = factored_copy
        self.group_tiles = kv_width // TILE
        self.group_count, residual = divmod(self.geometry['full_kv_tiles'], self.group_tiles)
        self.residual_row = self.group_count * kv_width if residual else None
        self.shared_swizzle = shared_swizzle
        self.mma_tiles = build_d64_wgmma_tiles(kv_tile=kv_width, shared_swizzle=shared_swizzle)
        self.rows, self.row_index, self.columns = _row_fibers()
        self.copy_plans = {}
        self.store_plans = {}
        fragment_name = tile_packet.get(_FRAGMENT_REQUEST, NATIVE_FRAGMENT)
        if _FRAGMENT_REQUEST in tile_packet:
            tile_packet = dict(tile_packet)
            tile_packet.pop(_FRAGMENT_REQUEST)
        self.fragment = checked_fragment_layout(fragment_name)
        self.copy_execution, output = _checked_tile_layouts(tile_packet)
        output = self.fragment.accumulator_distribution(output)
        self.copy_offsets = shared_offsets_64x64(self.shared_swizzle)
        alternate_fragment = fragment_name != NATIVE_FRAGMENT
        self.query_copy_offsets = self.fragment.query_offsets(self.copy_offsets) if alternate_fragment else self.copy_offsets
        self.query_copy_plans = {} if alternate_fragment else self.copy_plans
        for rows in {TILE, self.geometry['final_rows']} - {0}:
            self.copy_plans[rows] = self._checked_copy_plan(rows)
            if alternate_fragment:
                self.query_copy_plans[rows] = native_copy_plan(
                    self.copy_execution, rows, owner_policy=self.copy_owner_policy,
                    destination_offsets=self.query_copy_offsets)
            self.store_plans[rows] = plan_guarded_store(
                output, valid_shape=(rows, TILE), destination_row_stride_bytes=TILE * 4,
                destination_span_bytes=rows * TILE * 4)
        if any(group.owner_mask != _ALL_THREADS for plan in self.store_plans.values() for group in plan.groups):
            raise ValueError('D64 native output requires one physical owner per element')
        if self.geometry['final_rows']:
            for plans in (self.copy_plans, self.query_copy_plans) if alternate_fragment else (self.copy_plans,):
                full = plans[TILE].groups
                tail = plans[self.geometry['final_rows']].groups
                for a, b in zip(full, tail, strict=True):
                    if (a.rows, a.source_offsets, a.destination_offsets) != (b.rows, b.source_offsets, b.destination_offsets):
                        raise ValueError('boundary copy changed the physical ownership assignment')
        self.tail_prefix = self.geometry['final_rows']
        self.tail_row = self.length - self.tail_prefix
        self.score_prefix = None
        self.retained_tail = None
        if retain_tail_scores and self.tail_prefix:
            # Verify the predicate implemented by _guards against the image of
            # the native GF(2) score map restricted to the actual key prefix.
            self.score_prefix = prove_score_prefix(self.mma_tiles.fragment_columns, self.tail_prefix)
            for column, mask in zip(self.columns, self.score_prefix.holder_masks, strict=True):
                expected = sum(1 << tid for tid in range(THREADS)
                               if column + 2 * (tid & 3) < self.tail_prefix)
                if mask != expected:
                    raise ValueError('retained score domain differs from the emitted key predicate')
        if kv_width == 2 * TILE and self.mma_tiles.v_offset != self.mma_tiles.k_offset + ROOT_BYTES:
            raise ValueError('KV128 requires V to alias the second K64 root')
        self.smem_base = self.tmp('b32')
        self.decl.extend(('.reg .u32 tid;',
                          f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{self.mma_tiles.shared_bytes}];'))
        self.emit('mov.u32 tid, %tid.x;')
        self.emit(f'mov.u32 {self.smem_base}, frisk_shared;')
        self.cta = self._value('mov.u32', '%ctaid.x')
        self.lane = self._value('and.b32', 'tid', 3)
        self.lane_col = self._value('shl.b32', self.lane, 1)
        self.left_lane = [self._value('setp.eq.u32', self._value('and.b32', self.lane, mask), 0, kind='pred')
                          for mask in (1, 2)]
        tile_count = self.geometry['q_tiles']
        self.head = self._value('div.u32', self.cta, tile_count)
        head_tiles = self._value('mul.lo.u32', self.head, tile_count)
        query_index = self._value('sub.u32', self.cta, head_tiles)
        self.query_row = self._value('shl.b32', query_index, 6)
        self.roots = []
        for i, span in enumerate(self.geometry['head_bytes']):
            head_offset = self._value('mul.wide.u32', self.head, span, kind='b64')
            self.roots.append(self._value('add.u64', f'${i+1}', head_offset, kind='b64'))
        self.zero = self._value('mov.b32', 0)
        self.tail_guards = {}

    def _checked_copy_plan(self, rows):
        return native_copy_plan(self.copy_execution, rows, owner_policy=self.copy_owner_policy,
                                shared_swizzle=self.shared_swizzle)

    def _value(self, opcode, *args, kind='b32'):
        out = self.tmp(kind)
        self.emit(f"{opcode} {out}, {', '.join(map(str, args))};")
        return out

    def _tile_pointer(self, root, row, stride):
        offset = self._value('mul.wide.u32', row, stride, kind='b64')
        return self._value('add.u64', root, offset, kind='b64')

    def _row_predicate(self, tile_row, rows):
        local = self.map(rows)
        row = self._value('add.u32', tile_row, local)
        return self._value('setp.lt.u32', row, self.length, kind='pred')

    def _copy(self, root, tile_row, shared_offset, *, guard, prefix=TILE, storage_rows=TILE):
        from attention_copy_schedule import copy_end

        query = root == self.roots[0]
        if query and (shared_offset != self.mma_tiles.q_offset or storage_rows != TILE):
            raise ValueError('a checked query copy must publish its whole operand carrier')
        plans = self.query_copy_plans if query else self.copy_plans
        offsets = self.query_copy_offsets if query else self.copy_offsets
        if type(storage_rows) is not int or storage_rows not in (8, 16, 32, TILE) or storage_rows < prefix:
            raise ValueError('copy publication must cover the checked valid prefix')
        plan = plans[prefix]
        if shared_offset + copy_end(plan, storage_rows) > self.mma_tiles.shared_bytes:
            raise ValueError('copy publication escapes its checked Shared allocation')
        if self.factored_copy:
            from attention_copy_schedule import emit_factored_copy

            if emit_factored_copy(self, root, tile_row, shared_offset, guard=guard,
                                  prefix=prefix, storage_rows=storage_rows,
                                  plan=plan, native_offsets=offsets):
                return
        tile = self._tile_pointer(root, tile_row, TILE * 2)
        shared = self._value('add.u32', self.smem_base, shared_offset)
        for group in plan.groups:
            if min(group.rows) >= storage_rows:
                continue
            group_guard = guard
            if self.simplify_tail_copy and prefix < TILE and group.active_mask == _ALL_THREADS:
                group_guard = False
            dst_offset = self.map(group.destination_offsets)
            dst = self._value('add.u32', shared, dst_offset)
            vector = [self.tmp('b32') for _ in range(4)]
            names = '{' + ', '.join(vector) + '}'
            if group_guard:
                for reg in vector:
                    self.emit(f'mov.b32 {reg}, 0;')
            if group.active_mask:
                src_offset = self._value('cvt.u64.u32', self.map(group.source_offsets), kind='b64')
                src = self._value('add.u64', tile, src_offset, kind='b64')
                pred = self._row_predicate(tile_row, group.rows) if group_guard else None
                qualifier = f'@{pred} ' if pred else ''
                self.emit(f'{qualifier}ld.global.v4.b32 {names}, [{src}];')
            elif not group_guard:
                raise ValueError('inactive copy requires checked zero publication')
            store_guard = ''
            if max(group.rows) >= storage_rows:
                active = self._value('setp.lt.u32', self.map(group.rows), storage_rows, kind='pred')
                store_guard = f'@{active} '
            self.emit(f'{store_guard}st.shared.v4.b32 [{dst}], {names};')

    def _publish(self):
        self.emit('bar.sync 0;')
        self.emit('fence.proxy.async.shared::cta;')

    def _scores(self, tile_row, *, prefix, copy_v=False):
        # The prior QK/PV group has drained before any physical root overwrite.
        self.emit('bar.sync 0;')
        from attention_tile_mma import score_width_for_prefix

        width = score_width_for_prefix(prefix) if self.narrow_tail and prefix < TILE else TILE
        self._copy(self.roots[1], tile_row, self.mma_tiles.k_offset, guard=prefix != TILE,
                   prefix=prefix, storage_rows=width)
        if copy_v:
            self._copy(self.roots[2], tile_row, self.mma_tiles.v_offset, guard=prefix != TILE, prefix=prefix)
        self._publish()
        return self._compute_scores(prefix, self.mma_tiles)

    def _compute_scores(self, prefix, mma_tiles):
        from attention_tile_mma import score_width_for_prefix

        width = score_width_for_prefix(prefix) if self.narrow_tail and prefix < TILE else TILE
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(width // 2)]
        if width < TILE:
            mma_tiles.emit_qk_prefix(self.emit, self.tmp, self.smem_base, scores, prefix)
        else:
            mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        scaled = [self._value('mul.rn.f32', reg, '0f3e000000', kind='f32') for reg in scores]
        return scaled + [self.zero] * (FRAGMENT_SLOTS - len(scaled))

    def _wide_scores(self, tile_row):
        # A full KV128 is two adjacent, independently proved K64 copies.
        self.emit('bar.sync 0;')
        second_row = self._value('add.u32', tile_row, TILE)
        k = self.mma_tiles.k_offset
        self._copy(self.roots[1], tile_row, k, guard=False)
        self._copy(self.roots[1], second_row, k + ROOT_BYTES, guard=False)
        self._publish()
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(2 * FRAGMENT_SLOTS)]
        self.mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        scaled = [self._value('mul.rn.f32', reg, '0f3e000000', kind='f32') for reg in scores]
        return second_row, scaled[:FRAGMENT_SLOTS], scaled[FRAGMENT_SLOTS:]

    def _publish_values(self, row, prefix):
        # The native producer and previous PV consumer have completed wait0.
        self.emit('bar.sync 0;')
        self._copy(self.roots[2], row, self.mma_tiles.v_offset, guard=prefix != TILE, prefix=prefix)
        self._publish()

    def _tail_scores(self, *, copy_v=False):
        if self.retained_tail is None:
            return self._scores(self.tail_row, prefix=self.tail_prefix, copy_v=copy_v)
        if copy_v:
            self._publish_values(self.tail_row, self.tail_prefix)
        return self.retained_tail

    def _guards(self, prefix):
        if prefix == TILE:
            return (True,) * FRAGMENT_SLOTS
        if not self.tail_guards:
            for slot in self.columns:
                delta = prefix - slot
                if delta not in self.tail_guards:
                    self.tail_guards[delta] = (False if delta <= 0 else True if delta > 6 else
                                               self._value('setp.lt.u32', self.lane_col, delta, kind='pred'))
        return tuple(self.tail_guards[prefix - slot] for slot in self.columns)

    def _tree(self, op, values, valid=None):
        if valid is None:
            valid = (True,) * len(values)
        pairs = list(zip(values, valid, strict=True))
        while len(pairs) > 1:
            parents = []
            for (left, left_active), (right, right_active) in zip(pairs[::2], pairs[1::2], strict=True):
                # Row validity is a column prefix. A right leaf implies that
                # the corresponding lane has a valid left leaf.
                if right_active is False:
                    out = left
                elif right_active is True:
                    out = self._value(op, left, right, kind='f32')
                else:
                    out = self._value('mov.f32', left, kind='f32')
                    self.emit(f'@{right_active} {op} {out}, {left}, {right};')
                parents.append((out, left_active))
            pairs = parents
        return pairs[0]

    def _warp_reduce(self, opcode, value, *, ordered):
        for mask, left_pred in zip((1, 2), self.left_lane, strict=True):
            peer = self._value('shfl.sync.bfly.b32', value, mask, 31, -1)
            if ordered:
                left = self._value('selp.b32', value, peer, left_pred)
                right = self._value('selp.b32', peer, value, left_pred)
            else:
                left, right = value, peer
            value = self._value(opcode, left, right, kind='f32')
        return value

    def _full_loop(self, label, body, *, iterations=None):
        if iterations is None:
            iterations = self.group_count
        if type(iterations) is not int or not 1 <= iterations <= self.group_count:
            raise ValueError('invalid full KV loop bound')
        tile = self._value('mov.u32', 0)
        self.emit(f'{label}_loop:')
        first = tile if self.group_tiles == 1 else self._value('shl.b32', tile, 1)
        row = self._value('shl.b32', first, 6)
        body(first, row)
        self.emit(f'add.u32 {tile}, {tile}, 1;')
        again = self._value('setp.lt.u32', tile, iterations, kind='pred')
        self.emit(f'@{again} bra.uni {label}_loop;')

    def _max_pass(self, maxima):
        def update(scores, prefix):
            guards = self._guards(prefix)
            masked = [(_NEG_INF if p is False else score if p is True else
                       self._value('selp.b32', score, _NEG_INF, p))
                      for score, p in zip(scores, guards, strict=True)]
            for dst, slots in zip(maxima, self.rows, strict=True):
                value, _ = self._tree('max.f32', [masked[i] for i in slots])
                self.emit(f'max.f32 {dst}, {dst}, {value};')
        def full(_, row):
            if self.group_tiles == 1:
                update(self._scores(row, prefix=TILE), TILE)
            else:
                _, left, right = self._wide_scores(row)
                update(left, TILE)
                update(right, TILE)

        self._full_loop('row_max', full)
        if self.residual_row is not None:
            update(self._scores(self.residual_row, prefix=TILE), TILE)
        if self.tail_prefix:
            scores = self._tail_scores()
            if self.score_prefix is not None:
                self.retained_tail = tuple(score if mask else self.zero for score, mask in
                                           zip(scores, self.score_prefix.holder_masks, strict=True))
            update(scores, self.tail_prefix)
        return [self._warp_reduce('max.f32', maximum, ordered=False) for maximum in maxima]

    def _exponents(self, scores, maxima, guards):
        result = []
        for slot, (score, guard) in enumerate(zip(scores, guards, strict=True)):
            if guard is False:
                result.append(self.zero)
                continue
            center = self._value('sub.rn.f32', score, maxima[self.row_index[slot]], kind='f32')
            base2 = self._value('mul.rn.f32', center, '0f3fb8aa3b', kind='f32')
            exp = self._value('ex2.approx.f32', base2, kind='f32')
            result.append(exp if guard is True else self._value('selp.b32', exp, self.zero, guard))
        return result

    def _sum_boundary_scores(self):
        return None

    def _sum_pass(self, maxima):
        levels = self.geometry['full_kv_tiles'].bit_length()
        stack = [tuple(self._value('mov.f32', _FP_ZERO, kind='f32') for _ in self.rows)
                 for _ in range(levels)]
        add = 'add.rn.f32'

        def local(scores, prefix):
            valid = self._guards(prefix)
            exps = self._exponents(scores, maxima, valid)
            return [self._tree(add, [exps[i] for i in slots], [valid[i] for i in slots])
                    for slots in self.rows]

        def update(tile, row):
            first_level = self.group_tiles - 1
            if first_level == 0:
                values = [value for value, _ in local(self._scores(row, prefix=TILE), TILE)]
            else:
                _, left, right = self._wide_scores(row)
                values = [self._value(add, a, b, kind='f32') for (a, _), (b, _) in
                          zip(local(left, TILE), local(right, TILE), strict=True)]
            assignments = {}
            for level in range(first_level, len(stack)):
                registers = stack[level]
                assignments[level] = values
                if level + 1 < len(stack):
                    bit = self._value('and.b32', tile, 1 << level)
                    store = self._value('setp.eq.u32', bit, 0, kind='pred')
                    self.emit(f'@{store} bra.uni row_sum_store_{level};')
                    values = [self._value(add, previous, value, kind='f32')
                              for previous, value in zip(registers, values, strict=True)]
            for level in reversed(range(first_level, levels)):
                self.emit(f'row_sum_store_{level}:')
                for out, value in zip(stack[level], assignments[level], strict=True):
                    self.emit(f'mov.f32 {out}, {value};')
                self.emit('bra.uni row_sum_tile_done;')
            self.emit('row_sum_tile_done:')

        def join(left, right):
            parents = []
            for first, (second, active) in zip(left, right, strict=True):
                if active is True:
                    result = self._value(add, first, second, kind='f32')
                elif active is False:
                    result = first
                else:
                    result = self._value('mov.f32', first, kind='f32')
                    self.emit(f'@{active} {add} {result}, {first}, {second};')
                parents.append((result, True))
            return parents

        # Each loop iteration consumes an aligned dyadic subtree of one or
        # two original 64-key tiles. Join the remaining complete subtrees
        # right-to-left, with the odd full tile and guarded prefix at the end.
        boundary = self._sum_boundary_scores()
        if boundary is None:
            self._full_loop('row_sum', update)
        else:
            scores, plan = boundary
            values = [value for value, _ in local(scores, TILE)]
            self._full_loop('row_sum', update, iterations=plan['last_group'])
            # The precomputed last leaf is still the right operand at every
            # carry. Keep the same canonical merge and suffix order.
            for level in plan['carry_levels']:
                values = [self._value(add, previous, value, kind='f32')
                          for previous, value in zip(stack[level], values, strict=True)]
            for out, value in zip(stack[plan['store_level']], values, strict=True):
                self.emit(f'mov.f32 {out}, {value};')
        covered = self.group_count * self.group_tiles
        if self.residual_row is None and not (covered & (covered - 1)):
            totals = list(stack[-1])
            if self.tail_prefix:
                for reg, (value, guard) in zip(totals, local(self._tail_scores(), self.tail_prefix), strict=True):
                    if guard is not False:
                        qualifier = '' if guard is True else f'@{guard} '
                        self.emit(f'{qualifier}{add} {reg}, {reg}, {value};')
        else:
            suffix = None
            if self.residual_row is not None:
                suffix = local(self._scores(self.residual_row, prefix=TILE), TILE)
            if self.tail_prefix:
                tail = local(self._tail_scores(), self.tail_prefix)
                suffix = tail if suffix is None else join([reg for reg, _ in suffix], tail)
            for level, complete in enumerate(stack):
                if covered & (1 << level):
                    suffix = ([(reg, True) for reg in complete] if suffix is None else join(complete, suffix))
            totals = [value for value, _ in suffix]
        return [self._warp_reduce(add, value, ordered=True) for value in totals]

    def _emit_pv(self, packed, accum, prefix):
        self.mma_tiles.emit_pv(self.emit, self.tmp, self.smem_base, packed, accum)

    def _value_pass(self, maxima, sums):
        accum = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(FRAGMENT_SLOTS)]

        def update(scores, prefix):
            guards = self._guards(prefix)
            exps = self._exponents(scores, maxima, guards)
            probabilities = []
            for i, (exp, guard) in enumerate(zip(exps, guards, strict=True)):
                denom = sums[self.row_index[i]]
                if guard is False:
                    probabilities.append(self.zero)
                elif guard is True:
                    probabilities.append(self._value('div.rn.f32', exp, denom, kind='f32'))
                else:
                    probability = self._value('mov.f32', _FP_ZERO, kind='f32')
                    self.emit(f'@{guard} div.rn.f32 {probability}, {exp}, {denom};')
                    probabilities.append(probability)
            packed = [self._value('cvt.rn.f16x2.f32', high, low) for low, high in
                      zip(probabilities[::2], probabilities[1::2], strict=True)]
            self._emit_pv(packed, accum, prefix)

        def full(_, row):
            if self.group_tiles == 1:
                update(self._scores(row, prefix=TILE, copy_v=True), TILE)
            else:
                second_row, left, right = self._wide_scores(row)
                # KV128 QK is complete. The high K root can now stage each V64
                # in order, draining PV before the second overwrite.
                self._publish_values(row, TILE)
                update(left, TILE)
                self._publish_values(second_row, TILE)
                update(right, TILE)

        self._full_loop('row_value', full)
        if self.residual_row is not None:
            update(self._scores(self.residual_row, prefix=TILE, copy_v=True), TILE)
        if self.tail_prefix:
            update(self._tail_scores(copy_v=True), self.tail_prefix)
        return accum

    def _fragment_evidence(self):
        evidence = self.fragment.evidence(self.mma_tiles.fragment_columns,
                                          self.mma_tiles.score_fragment_columns,
                                          score_width=self.mma_tiles.kv_tile)
        return dict(evidence, output_store=output_store_evidence(self.store_plans,
                                                               vector_store=self.vector_store))

    def _write_output(self, accum):
        root = self._tile_pointer(self.roots[3], self.query_row, TILE * 4)
        groups = self.store_plans[TILE].groups
        if self.vector_store:
            for plan in self.store_plans.values():
                for low, high in zip(plan.groups[::2], plan.groups[1::2], strict=True):
                    if (low.rows != high.rows or low.active_mask != high.active_mask
                            or low.owner_mask != high.owner_mask
                            or any(a % 8 or b != a + 4 for a, b in
                                   zip(low.destination_offsets, high.destination_offsets, strict=True))):
                        raise ValueError('vector store requires adjacent, jointly active output owners')
            groups = groups[::2]
            accum = ['{' + low + ', ' + high + '}' for low, high in zip(accum[::2], accum[1::2], strict=True)]
        opcode = 'st.global.v2.b32' if self.vector_store else 'st.global.b32'
        output_predicates = {}
        for group, reg in zip(groups, accum, strict=True):
            offset = self._value('cvt.u64.u32', self.map(group.destination_offsets), kind='b64')
            addr = self._value('add.u64', root, offset, kind='b64')
            pred = None
            if self.geometry['final_rows']:
                key = tuple(group.rows)
                if not self.simplify_tail_copy or key not in output_predicates:
                    output_predicates[key] = self._row_predicate(self.query_row, group.rows)
                pred = output_predicates[key]
            qualifier = f'@{pred} ' if pred else ''
            self.emit(f'{qualifier}{opcode} [{addr}], {reg};')

    def run(self):
        if self.simplify_tail_copy and self.tail_prefix:
            complete = self._value('setp.le.u32', self.query_row, self.length - TILE, kind='pred')
            self.emit(f'@{complete} bra.uni query_copy_complete;')
            self._copy(self.roots[0], self.query_row, self.mma_tiles.q_offset, guard=True)
            self.emit('bra.uni query_copy_done;')
            self.emit('query_copy_complete:')
            self._copy(self.roots[0], self.query_row, self.mma_tiles.q_offset, guard=False)
            self.emit('query_copy_done:')
        else:
            self._copy(self.roots[0], self.query_row, self.mma_tiles.q_offset,
                       guard=bool(self.geometry['final_rows']))
        maxima = self._max_pass([self._value('mov.f32', _NEG_INF, kind='f32') for _ in self.rows])
        self._write_output(self._value_pass(maxima, self._sum_pass(maxima)))
        text = render_inline_execution(self.decl, self.code)
        info = dict(self.geometry, static_shared_bytes=self.mma_tiles.shared_bytes,
                    physical_shared_bytes=self.mma_tiles.shared_bytes,
                    qkv_root_offsets=[self.mma_tiles.q_offset, self.mma_tiles.k_offset, self.mma_tiles.v_offset],
                    fragment_gf2_columns=list(self.mma_tiles.fragment_columns),
                    execution_kv_width=self.kv_width,
                    score_gf2_columns=list(self.mma_tiles.score_fragment_columns),
                    retain_tail_score_slots=list(self.score_prefix.held_slots) if self.score_prefix else [],
                    vector_store=self.vector_store, narrow_tail=self.narrow_tail,
                    simplify_tail_copy=self.simplify_tail_copy, factored_copy=self.factored_copy,
                    shared_v_reuse='upper-k64-after-wait0' if self.group_tiles == 2 else 'disjoint',
                    copy_owner_policy=self.copy_owner_policy,
                    copy_lane_bit_order=self.copy_plans[TILE].lane_bit_order,
                    passes=['global-max', 'canonical-row-sum', 'normalized-fp16-pv'],
                    reduction_order='hardware-first-owner-v1 with absent leaves pruned',
                    p_rounding='cvt.rn.f16x2.f32 after global normalization', ftz=False,
                    fp_fusion=False, mask='query-and-kv-prefix',
                    tail_copy_active_groups=[g.active_mask.bit_count() for g in
                                             self.copy_plans[self.geometry['final_rows']].groups]
                    if self.geometry['final_rows'] else [],
                    tail_store_elements=sum(g.active_mask.bit_count() for g in
                                            self.store_plans[self.geometry['final_rows']].groups)
                    if self.geometry['final_rows'] else 0,
                    ttgir_sha256=hashlib.sha256(text.encode()).hexdigest())
        info.update(self.mma_tiles.shared_layout_evidence())
        info.update(self._fragment_evidence())
        check_static_shared(info)
        return text, info


def emit_tiled_attention(spec, *, tile_packet, copy_owner_policy='gf2', kv_width=TILE,
                         retain_tail_scores=False, vector_store=False, narrow_tail=False,
                         simplify_tail_copy=False, retain_all_scores=False, factored_copy=False,
                         bulk_value_publish=False, retain_residual_scores=False,
                         streaming_bulk_values=False, shared_swizzle=128,
                         fragment_layout=NATIVE_FRAGMENT, k_stage_policy=None, tail_score_carrier=False):
    fragment = checked_fragment_layout(fragment_layout)
    if type(tile_packet) is not dict or _FRAGMENT_REQUEST in tile_packet:
        raise ValueError('checked Attention requires the original execution replay packet')
    if fragment.name != NATIVE_FRAGMENT:
        tile_packet = dict(tile_packet, **{_FRAGMENT_REQUEST: fragment.name})
    if type(kv_width) is not int or any(type(flag) is not bool for flag in
            (retain_all_scores, retain_tail_scores, vector_store, narrow_tail, simplify_tail_copy,
             factored_copy, bulk_value_publish, retain_residual_scores, streaming_bulk_values,
             tail_score_carrier)):
        raise ValueError('Attention requires an integer KV width and boolean strategy flags')
    if type(shared_swizzle) is not int or shared_swizzle not in (64, 128):
        raise ValueError('checked Attention Shared swizzle must be 64 or 128')
    if shared_swizzle == 64 and (kv_width != TILE or retain_all_scores):
        raise ValueError('SW64 requires the checked streaming KV64 schedule')
    if tail_score_carrier:
        if (not retain_all_scores or shared_swizzle != 128 or fragment.name != NATIVE_FRAGMENT
                or k_stage_policy is not None):
            raise ValueError('tail score carrier requires native SW128 and full score retention')
        checked_geometry(spec, kv_width=kv_width, tail_score_carrier=True)
    if k_stage_policy in ('boundary-control', 'boundary-reuse'):
        from attention_k_boundary_reuse import emit_boundary_k_attention

        if (kv_width != TILE or shared_swizzle != 128 or fragment.name != NATIVE_FRAGMENT
                or retain_tail_scores or narrow_tail or simplify_tail_copy or retain_all_scores
                or bulk_value_publish or retain_residual_scores or streaming_bulk_values):
            raise ValueError('boundary K-stage policy requires native SW128 KV64 without tail options')
        return emit_boundary_k_attention(spec, policy=k_stage_policy, tile_packet=tile_packet,
            copy_owner_policy=copy_owner_policy, vector_store=vector_store, factored_copy=factored_copy)
    if k_stage_policy is not None:
        from attention_k_stage_reuse import emit_short_k_stage_attention

        if (kv_width != TILE or shared_swizzle != 128 or fragment.name != NATIVE_FRAGMENT
                or not retain_tail_scores or narrow_tail or simplify_tail_copy or retain_all_scores
                or bulk_value_publish or retain_residual_scores or streaming_bulk_values):
            raise ValueError('short K-stage policy requires native SW128 KV64 with retained tail scores')
        return emit_short_k_stage_attention(spec, policy=k_stage_policy, tile_packet=tile_packet,
            copy_owner_policy=copy_owner_policy, vector_store=vector_store, factored_copy=factored_copy)
    if retain_all_scores:
        from attention_single_score import emit_single_score_attention
        from attention_single_score_bulk import emit_single_score_bulk_attention

        if retain_tail_scores or narrow_tail or simplify_tail_copy or retain_residual_scores or streaming_bulk_values:
            raise ValueError('full score retention has no tail-copy strategy')
        if not tail_score_carrier and kv_width != spec['N']:
            raise ValueError('full score retention requires one complete native key tile')
        implementation = emit_single_score_bulk_attention if bulk_value_publish else emit_single_score_attention
        return implementation(spec, tile_packet=tile_packet, vector_store=vector_store,
                              copy_owner_policy=copy_owner_policy, factored_copy=factored_copy,
                              tail_score_carrier=tail_score_carrier)
    if bulk_value_publish:
        raise ValueError('bulk value publication requires a complete retained score tile')
    if streaming_bulk_values:
        from attention_streaming_bulk import emit_streaming_bulk_attention

        if kv_width != 128 or not retain_tail_scores:
            raise ValueError('streaming bulk values require the checked KV128 schedule')
        return emit_streaming_bulk_attention(spec, tile_packet=tile_packet, vector_store=vector_store,
            copy_owner_policy=copy_owner_policy, retain_residual_scores=retain_residual_scores,
            factored_copy=factored_copy, narrow_tail=narrow_tail, simplify_tail_copy=simplify_tail_copy)
    if retain_residual_scores:
        from attention_residual_score import emit_residual_score_attention

        if kv_width != 128 or not retain_tail_scores or narrow_tail:
            raise ValueError('residual score retention requires the full KV128 remainder')
        return emit_residual_score_attention(spec, tile_packet=tile_packet, vector_store=vector_store,
            copy_owner_policy=copy_owner_policy, factored_copy=factored_copy,
            simplify_tail_copy=simplify_tail_copy)
    if kv_width == 32:
        from attention_narrow import emit_narrow_attention

        if retain_tail_scores or narrow_tail or simplify_tail_copy:
            raise ValueError('compact KV32 admits aligned complete KV64 pairs')
        return emit_narrow_attention(spec, tile_packet=tile_packet, vector_store=vector_store,
                                     copy_owner_policy=copy_owner_policy, factored_copy=factored_copy)
    if kv_width not in (64, 128):
        raise ValueError('streaming Attention requires KV32, KV64 or KV128')
    return _TiledAttention(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                           kv_width=kv_width, retain_tail_scores=retain_tail_scores,
                           vector_store=vector_store, narrow_tail=narrow_tail,
                           simplify_tail_copy=simplify_tail_copy, factored_copy=factored_copy,
                           shared_swizzle=shared_swizzle).run()
