"""Checked aligned D64 Attention with compact 32-key native work tiles."""

from dataclasses import replace

from attention_tile_mma import (FRAGMENT_SLOTS, ROOT_BYTES, THREADS, TILE,
                                build_d64_wgmma_tiles, shared_offsets_64x64)
from tiled_attention import _FP_ZERO, _NEG_INF, _TiledAttention


_HALF = TILE // 2
_ALL_THREADS = (1 << THREADS) - 1


def _narrow_copy_groups(plan):
    """Bind the first 32 key/value rows to exactly one compact SW128 root."""
    if (plan.domain.carrier_shape != (TILE, TILE)
            or plan.domain.valid_shape != (TILE, TILE)):
        raise ValueError('KV32 requires a complete aligned native copy carrier')
    expected_offsets = shared_offsets_64x64()
    groups = []
    covered = set()
    for group in plan.groups:
        selected = tuple(row < _HALF for row in group.rows)
        if not any(selected):
            continue
        if group.active_mask != _ALL_THREADS or len(group.rows) != THREADS:
            raise ValueError('KV32 copy group has inactive or incomplete owners')
        for active, row, column, source, destination in zip(
                selected, group.rows, group.columns, group.source_offsets,
                group.destination_offsets, strict=True):
            logical = row * TILE + column
            if (not 0 <= row < TILE or not 0 <= column < TILE or column % 8
                    or source != logical * 2 or source % 16
                    or destination != expected_offsets[logical]):
                raise ValueError('KV32 copy changes its native ownership')
            if active:
                if not 0 <= destination <= ROOT_BYTES // 2 - 16 or destination in covered:
                    raise ValueError('KV32 copy group escapes its compact root or changes ownership')
                covered.add(destination)
        mask = sum(1 << lane for lane, active in enumerate(selected) if active)
        groups.append(replace(group, active_mask=mask))
    expected = {expected_offsets[slot] for slot in range(0, TILE * _HALF, 8)}
    if covered != expected:
        raise ValueError('KV32 copy does not initialize every native operand halfword')
    return tuple(groups)


class _NarrowAttention(_TiledAttention):
    def __init__(self, spec, *, tile_packet, copy_owner_policy, vector_store):
        if type(spec) is not dict or type(spec.get('N')) is not int or spec['N'] % TILE:
            raise ValueError('KV32 currently requires an aligned sequence length')
        super().__init__(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                         vector_store=vector_store)
        old_bytes = self.mma_tiles.shared_bytes
        self.mma_tiles = build_d64_wgmma_tiles(kv_tile=_HALF)
        self.kv_width = _HALF
        self.narrow_rows = tuple(tuple(slot for slot in row if slot < FRAGMENT_SLOTS // 2)
                                 for row in self.rows)
        if any(len(row) != _HALF // 4 for row in self.narrow_rows):
            raise ValueError('KV32 does not preserve the native row-fiber projection')
        self.narrow_copy_groups = _narrow_copy_groups(self.copy_plans[TILE])
        declaration = f'frisk_shared[{old_bytes}]'
        if sum(declaration in line for line in self.decl) != 1:
            raise ValueError('KV32 requires one checked static Shared allocation')
        self.decl = [line.replace(declaration, f'frisk_shared[{self.mma_tiles.shared_bytes}]')
                     for line in self.decl]

    def _copy32(self, root, row, shared_offset):
        if self.factored_copy:
            from attention_copy_schedule import emit_factored_copy
            if 32 not in self.copy_plans:
                self.copy_plans[32] = self._checked_copy_plan(32)
            if emit_factored_copy(self, root, row, shared_offset, guard=False, prefix=32, storage_rows=32):
                return
        tile = self._tile_pointer(root, row, TILE * 2)
        shared = self._value('add.u32', self.smem_base, shared_offset)
        for group in self.narrow_copy_groups:
            source_offset = self._value('cvt.u64.u32', self.map(group.source_offsets), kind='b64')
            source = self._value('add.u64', tile, source_offset, kind='b64')
            destination_offset = self.map(group.destination_offsets)
            destination = self._value('add.u32', shared, destination_offset)
            registers = [self.tmp('b32') for _ in range(4)]
            values = '{' + ', '.join(registers) + '}'
            qualifier = ''
            if group.active_mask != _ALL_THREADS:
                predicate = self._value('setp.lt.u32', self.map(group.rows), _HALF, kind='pred')
                qualifier = f'@{predicate} '
            self.emit(f'{qualifier}ld.global.v4.b32 {values}, [{source}];')
            self.emit(f'{qualifier}st.shared.v4.b32 [{destination}], {values};')

    def _scores32(self, row, *, copy_v=False):
        self.emit('bar.sync 0;')
        self._copy32(self.roots[1], row, self.mma_tiles.k_offset)
        if copy_v:
            self._copy32(self.roots[2], row, self.mma_tiles.v_offset)
        self._publish()
        scores = [self._value('mov.f32', _FP_ZERO, kind='f32')
                  for _ in range(FRAGMENT_SLOTS // 2)]
        self.mma_tiles.emit_qk(self.emit, self.tmp, self.smem_base, scores)
        return [self._value('mul.rn.f32', reg, '0f3e000000', kind='f32') for reg in scores]

    def _local32(self, scores, opcode):
        return [self._tree(opcode, [scores[slot] for slot in row])[0]
                for row in self.narrow_rows]

    def _max_pass(self, maxima):
        def full(_, row):
            left = self._local32(self._scores32(row), 'max.f32')
            second = self._value('add.u32', row, _HALF)
            right = self._local32(self._scores32(second), 'max.f32')
            for target, a, b in zip(maxima, left, right, strict=True):
                joined = self._value('max.f32', a, b, kind='f32')
                self.emit(f'max.f32 {target}, {target}, {joined};')

        self._full_loop('row_max', full)
        return [self._warp_reduce('max.f32', value, ordered=False) for value in maxima]

    def _sum_pass(self, maxima):
        add = 'add.rn.f32'
        count = self.geometry['full_kv_tiles']
        stack = [tuple(self._value('mov.f32', _FP_ZERO, kind='f32') for _ in self.rows)
                 for _ in range(count.bit_length())]

        def local(row):
            scores = self._scores32(row)
            exponents = self._exponents(scores, maxima, (True,) * len(scores))
            return self._local32(exponents, add)

        def full(tile, row):
            left = local(row)
            right = local(self._value('add.u32', row, _HALF))
            values = [self._value(add, a, b, kind='f32') for a, b in zip(left, right, strict=True)]
            assignments = []
            for level, registers in enumerate(stack):
                assignments.append(values)
                if level + 1 < len(stack):
                    bit = self._value('and.b32', tile, 1 << level)
                    store = self._value('setp.eq.u32', bit, 0, kind='pred')
                    self.emit(f'@{store} bra.uni narrow_sum_store_{level};')
                    values = [self._value(add, previous, value, kind='f32')
                              for previous, value in zip(registers, values, strict=True)]
            for level in reversed(range(len(stack))):
                self.emit(f'narrow_sum_store_{level}:')
                for target, value in zip(stack[level], assignments[level], strict=True):
                    self.emit(f'mov.f32 {target}, {value};')
                self.emit('bra.uni narrow_sum_tile_done;')
            self.emit('narrow_sum_tile_done:')

        self._full_loop('row_sum', full)
        suffix = None
        for level, complete in enumerate(stack):
            if count & (1 << level):
                suffix = (list(complete) if suffix is None else
                          [self._value(add, a, b, kind='f32')
                           for a, b in zip(complete, suffix, strict=True)])
        return [self._warp_reduce(add, value, ordered=True) for value in suffix]

    def _value_pass(self, maxima, sums):
        accum = [self._value('mov.f32', _FP_ZERO, kind='f32') for _ in range(FRAGMENT_SLOTS)]

        def half(row):
            scores = self._scores32(row, copy_v=True)
            exponents = self._exponents(scores, maxima, (True,) * len(scores))
            probabilities = [self._value('div.rn.f32', value, sums[self.row_index[slot]], kind='f32')
                             for slot, value in enumerate(exponents)]
            packed = [self._value('cvt.rn.f16x2.f32', high, low)
                      for low, high in zip(probabilities[::2], probabilities[1::2], strict=True)]
            self.mma_tiles.emit_pv(self.emit, self.tmp, self.smem_base, packed, accum)

        def full(_, row):
            half(row)
            half(self._value('add.u32', row, _HALF))

        self._full_loop('row_value', full)
        return accum


def emit_narrow_attention(spec, *, tile_packet, copy_owner_policy='gf2', vector_store=False, factored_copy=False):
    if type(factored_copy) is not bool:
        raise ValueError('copy schedule flag must be boolean')
    emitter = _NarrowAttention(spec, tile_packet=tile_packet, copy_owner_policy=copy_owner_policy,
                               vector_store=vector_store)
    emitter.factored_copy = factored_copy
    return emitter.run()
