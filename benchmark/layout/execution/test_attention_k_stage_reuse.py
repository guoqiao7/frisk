"""Regression for checked K residency and the late V Shared alias."""

from copy import deepcopy
from unittest import TestCase, mock

from attention_k_stage_reuse import _check_short_order
from tiled_attention import _output_distribution, emit_tiled_attention
from valid_domain import dense_copy_execution


class AttentionKStageTests(TestCase):
    def test_true_tail_residency_has_the_same_mma_and_a_drained_v_alias(self):
        for n in (96, 127):
            spec = dict(B=1, H=1, N=n, D=64, q_block=64, kv_block=64)
            with self.subTest(n=n), mock.patch('tiled_attention._checked_tile_layouts',
                    return_value=(dense_copy_execution((64, 64)), _output_distribution())):
                results = [emit_tiled_attention(spec, tile_packet={}, retain_tail_scores=True,
                    vector_store=True, factored_copy=True, k_stage_policy=policy)
                    for policy in ('late-v-control', 'resident-k')]
                control, resident = [proof['k_stage_reuse'] for _, proof in results]
                self.assertEqual((control['k0_publications'], resident['k0_publications']), (3, 1))
                self.assertEqual(control['kv_global_bytes'] - resident['kv_global_bytes'], 16384)
                self.assertEqual(control['kv_shared_store_bytes'] - resident['kv_shared_store_bytes'], 16384)
                self.assertEqual((control['barriers'], resident['barriers']), (12, 8))
                self.assertEqual((control['wgmma_wait0'], resident['wgmma_wait0']), (6, 6))
                self.assertEqual(control['shared_roots'], resident['shared_roots'])
                self.assertEqual(resident['shared_roots']['k0'], resident['shared_roots']['v0'])
                self.assertNotEqual(resident['shared_roots']['ktail'], resident['shared_roots']['v0'])
                self.assertEqual([proof['static_shared_bytes'] for _, proof in results], [24576, 24576])
                text = results[-1][0]
                before, _, rest = text.partition(resident['first_v_publish_label'] + ':')
                self.assertIn('wgmma.wait_group.sync.aligned 0;', before[before.index(resident['final_qk_label']):])
                self.assertIn('bar.sync 0;', rest.split(r'\0A')[1])
                corrupt = deepcopy(resident['events'])
                next(row for row in corrupt if row['label'] == resident['final_qk_label'])['wait0'] = 0
                with self.assertRaisesRegex(ValueError, 'did not drain'):
                    _check_short_order(corrupt, resident=True)

    def test_rejects_unsupported_shapes_and_policies(self):
        spec = dict(B=1, H=1, N=128, D=64, q_block=64, kv_block=64)
        with self.assertRaisesRegex(ValueError, 'one full KV64 tile'):
            emit_tiled_attention(spec, tile_packet={}, retain_tail_scores=True, k_stage_policy='resident-k')
        with self.assertRaisesRegex(ValueError, 'with retained tail'):
            emit_tiled_attention(dict(spec, N=96), tile_packet={}, k_stage_policy='resident-k')
