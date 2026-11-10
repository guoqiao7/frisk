"""Regression tests for checked aligned KV64 max-to-sum K reuse."""

from unittest import TestCase, mock

from attention_k_boundary_reuse import POLICIES, prove_boundary_sum
from tiled_attention import _output_distribution, emit_tiled_attention
from valid_domain import dense_copy_execution


def _emit(n, policy, *, factored_copy=True):
    spec = dict(B=1, H=1, N=n, D=64, q_block=64, kv_block=64)
    return emit_tiled_attention(spec, tile_packet={}, vector_store=True, factored_copy=factored_copy,
                               k_stage_policy=policy)


class AttentionKBoundaryTests(TestCase):
    def test_verified_carry_preserves_exact_leaf_order(self):
        for groups in range(2, 17):
            with self.subTest(groups=groups):
                plan = prove_boundary_sum(groups)
                last = groups - 1
                expected = []
                while last & 1:
                    expected.append(len(expected))
                    last >>= 1
                self.assertEqual(plan['carry_levels'], tuple(expected))
                self.assertEqual(plan['store_level'], len(expected))
        self.assertEqual(prove_boundary_sum(7)['ordered_tree'], (((0, 1), (2, 3)), ((4, 5), 6)))

    def test_boundary_reuse_removes_only_the_matched_publication(self):
        for n, factored in ((512, True), (704, True), (704, False)):
            with self.subTest(n=n, factored=factored), mock.patch(
                    'tiled_attention._checked_tile_layouts',
                    return_value=(dense_copy_execution((64, 64)), _output_distribution())):
                (control_text, control_proof), (reuse_text, reuse_proof) = [
                    _emit(n, policy, factored_copy=factored) for policy in POLICIES]
                control, reuse = [p['k_stage_reuse'] for p in (control_proof, reuse_proof)]
                groups = n // 64
                self.assertEqual(control['k64_publications'], 3 * groups)
                self.assertEqual(reuse['k64_publications'], 3 * groups - 1)
                for key, expected in (('kv_global_bytes', 8192), ('kv_shared_store_bytes', 8192),
                        ('global_load_sites', 4), ('shared_store_sites', 4), ('barriers', 2),
                        ('proxy_fences', 1), ('wgmma_wait0', 0), ('wgmma_atoms', 0)):
                    self.assertEqual(control[key] - reuse[key], expected)
                self.assertEqual(control['ordered_tree'], reuse['ordered_tree'])
                self.assertEqual(control['sum_leaf_evaluation_order'], [groups - 1, *range(groups - 1)])
                self.assertEqual(control['sum_leaf_insert_order'], list(range(groups)))
                self.assertEqual([p['static_shared_bytes'] for p in (control_proof, reuse_proof)], [24576] * 2)
                for text, proof in ((control_text, control), (reuse_text, reuse)):
                    window = text[text.index(proof['boundary_begin_label'] + ':'):
                                  text.index(proof['boundary_end_label'] + ':')]
                    self.assertEqual(window.count('wgmma.mma_async.'), 4)
                    self.assertEqual(window.count('wgmma.wait_group.sync.aligned 0;'), 1)
                    self.assertEqual(window.count('ld.global.v4.b32'), 4 if proof is control else 0)

    def test_boundary_scope_fails_closed(self):
        for n in (64, 96, 1025):
            with self.subTest(n=n), mock.patch(
                    'tiled_attention._checked_tile_layouts',
                    return_value=(dense_copy_execution((64, 64)), _output_distribution())):
                with self.assertRaisesRegex(ValueError, 'aligned KV64|complete KV64 groups'):
                    _emit(n, 'boundary-reuse')
        with self.assertRaisesRegex(ValueError, 'native SW128 KV64'):
            emit_tiled_attention(dict(B=1, H=1, N=512, D=64, q_block=64, kv_block=64),
                                 tile_packet={}, shared_swizzle=64, k_stage_policy='boundary-reuse')
