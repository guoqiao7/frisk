import unittest
from types import SimpleNamespace
from unittest.mock import patch
from native_online_ring import prepare_ring


class KVStorageTest(unittest.TestCase):
    def setUp(self):
        self.emitter = SimpleNamespace(kcopy={'operands': [0, 2]}, vcopy={'operands': [1, 3]},
                                       storage=lambda v: {'root': v}, ring_stride=0)
        self.allocation = {
            'allocations': [{'root_id': k, 'offset': p, 'bytes': b, 'alignment': 1024}
                            for k,p,b in [(2, 8192, 8192), (3, 16384, 8192)]],
            'static_shared_bytes': 24576, 'physical_shared_bytes': 24576,
            'shared_limit_bytes': 65536, 'tma_barrier_bytes': 0,
        }

    @patch('native_online_ring._packed_groups', return_value=[object()])
    def test_alternate_versions_clear_live_scratch(self, _):
        self.allocation['static_shared_bytes'] += 2048
        self.allocation['physical_shared_bytes'] += 2048
        plan = prepare_ring(self.emitter, self.allocation)
        self.assertEqual(plan['native_kv_versions'], {2:[8192,26624], 3:[16384,34816]})
        self.assertEqual(plan['physical_shared_bytes'], 43008)
        self.assertEqual(self.allocation['physical_shared_bytes'], 26624)

    @patch('native_online_ring._packed_groups', return_value=[object()])
    def test_insufficient_space_keeps_original_allocation(self, _):
        self.allocation['shared_limit_bytes'] = 30000
        self.assertIs(prepare_ring(self.emitter, self.allocation), self.allocation)
        self.assertEqual(self.emitter.ring_stride, 0)

    @patch('native_online_ring._packed_groups', return_value=[])
    def test_uncertified_copy_keeps_original_allocation(self, _):
        self.assertIs(prepare_ring(self.emitter, self.allocation), self.allocation)
        self.assertEqual(self.emitter.ring_stride, 0)


class CausalDiagonalMaskTest(unittest.TestCase):
    def test_lane_dependent_register_offset_falls_back_before_emission(self):
        from native_online_launch import NativeOnlineEmitter
        emitter = SimpleNamespace(kv_tile=64, tensor_slots=lambda _: [[0, 64], [1, 66]])
        # The second register has a different column displacement in each lane.
        # No partial PTX may precede the generic mask on fallback.
        self.assertFalse(NativeOnlineEmitter.emit_short_causal_mask(emitter, 0))


class PartialQueryStoreTest(unittest.TestCase):
    def test_lane_dependent_row_carry_preserves_generic_store(self):
        from native_online_launch import NativeOnlineEmitter
        emitter = SimpleNamespace(value_dim=64)
        # The byte increment is constant but crosses a row in only one lane.
        self.assertFalse(NativeOnlineEmitter.emit_partial_q_store(
            emitter, 0, [[0, 248], [8, 256]], [], 2, 4))

    def test_two_vectors_in_one_row_share_the_guard(self):
        from native_online_launch import NativeOnlineEmitter
        code = []
        emitter = SimpleNamespace(value_dim=64, rows=224, qrow='q',
                                  map=lambda _: 'base', address=lambda *_: 'addr',
                                  tmp=lambda *args: 'pred' if args else 'row', emit=code.append)
        self.assertTrue(NativeOnlineEmitter.emit_partial_q_store(
            emitter, 0, [[0, 8], [16, 24]], list('abcd'), 2, 4))
        self.assertEqual(sum('setp.lt' in s for s in code), 1)
        self.assertTrue(any('[addr+16]' in s and '{c, d}' in s for s in code))


class CrossIterationPVTest(unittest.TestCase):
    def emitter(self):
        from native_online_launch import NativeOnlineEmitter
        e = object.__new__(NativeOnlineEmitter)
        e.cross_iteration_pv = True
        e.pv_op, e.qk_op = object(), object()
        e.pending_pv_by_slot = {0: ['PV0'], 1: ['PV1']}
        e.loop_version = 1
        e.loop_prime = False
        e.iv = 1
        e.indices = {1: 'index'}
        e.pending_normalizer = []
        e.code = []
        return e

    @patch('native_online_ring.prefetch_next')
    def test_retire_old_pv_before_refill(self, refill):
        e = self.emitter()
        refill.side_effect = lambda emitter, index: emitter.emit('refill')
        e.emit_mma_group(e.qk_op, ['QK1'], last=True)
        self.assertEqual(e.code[2:], [
            'PV0', 'wgmma.commit_group.sync.aligned;', 'QK1',
            'wgmma.commit_group.sync.aligned;', 'wgmma.wait_group.sync.aligned 1;',
            'cp.async.wait_group 0;', 'bar.sync 0;', 'fence.proxy.async.shared::cta;',
            'refill', 'wgmma.wait_group.sync.aligned 0;'])

    @patch('native_online_ring.prefetch_next')
    def test_post_qk_prefetch_keeps_pv_in_its_iteration(self, refill):
        e = self.emitter()
        e.cross_iteration_pv = False
        e.prefetch_after_qk = True
        refill.side_effect = lambda emitter, index: emitter.emit('refill')
        e.emit_mma_group(e.qk_op, ['QK'], last=True)
        self.assertEqual(e.code[2:], [
            'QK', 'wgmma.commit_group.sync.aligned;', 'cp.async.wait_group 0;',
            'bar.sync 0;', 'fence.proxy.async.shared::cta;', 'refill',
            'wgmma.wait_group.sync.aligned 0;'])
        e.code.clear()
        e.emit_mma_group(e.pv_op, ['PV'], last=True)
        self.assertEqual(e.code[2:], ['PV', 'wgmma.commit_group.sync.aligned;',
                                      'wgmma.wait_group.sync.aligned 0;'])

    def test_last_pv_is_carried_and_drained_on_the_correct_slot(self):
        e = self.emitter()
        e.emit_mma_group(e.pv_op, ['NEW_PV1'], last=True)
        self.assertEqual(e.code, [])
        for count, expected in ((4, 'NEW_PV1'), (5, 'PV0')):
            e.code.clear()
            e.ktiles = count
            e.drain_pv()
            self.assertEqual(e.code[-3:], [expected, 'wgmma.commit_group.sync.aligned;',
                                           'wgmma.wait_group.sync.aligned 0;'])


class IndependentPrefetchTest(unittest.TestCase):
    def test_each_operand_issued_once_on_its_requested_edge(self):
        from native_online_ring import issue_at_qk_edge
        for k_edge, v_edge in [('before', 'after'), ('after', 'before')]:
            emitter = SimpleNamespace(kv_prefetch_edges=(k_edge, v_edge), loop_version=1)
            with patch('native_online_ring.next_tile', return_value=('ahead', 'last')) as target, \
                 patch('native_online_ring.issue_guarded') as issue:
                for edge in ('before', 'after'):
                    issue_at_qk_edge(emitter, 'index', edge)
                    issue.assert_called_with(emitter, 'ahead', 'last',
                                             copy_k=edge == k_edge, copy_v=edge == v_edge)
                target.assert_called_once_with(emitter, 'index')
                self.assertEqual(issue.call_count, 2)

    def test_k_tail_skips_both_arrival_and_copy(self):
        from native_online_ring import issue_guarded
        code = []
        emitter = SimpleNamespace(loop_version=0, emit=code.append)
        with patch('native_online_ring.issue_tile', side_effect=lambda *a, **kw: code.append('tile')) as copy:
            issue_guarded(emitter, 'next', 'last', copy_v=False)
            copy.assert_called_once_with(emitter, 'next', 1, copy_k=True, copy_v=False)
        self.assertEqual(code, ['@last bra.uni frisk_ring_ready_0_k;', 'tile', 'frisk_ring_ready_0_k:'])

    def test_coincident_edges_reuse_one_group(self):
        from native_online_ring import issue_at_qk_edge
        for when in ('before', 'after'):
            emitter = SimpleNamespace(kv_prefetch_edges=(when, when))
            with patch('native_online_ring.prefetch_next') as group:
                for edge in ('before', 'after'):
                    issue_at_qk_edge(emitter, 'index', edge)
                group.assert_called_once_with(emitter, 'index')
