"""Check the residency boundary and timing-free resource feedback."""

from unittest import TestCase

from attention_residency_cost import compiled_score, residency


class AttentionResidencyTests(TestCase):
    def test_one_more_cta_can_cross_the_3_cta_residency_class(self):
        def evidence(grid, registers):
            return residency(registers=registers, shared_bytes=41984, grid=grid)
        self.assertEqual(evidence(342, 146)['pressure'], 1)
        after = evidence(343, 146)
        self.assertEqual(after['ctas_per_sm'], 3)
        self.assertEqual(after['tail_ctas'], 1)
        self.assertAlmostEqual(after['pressure'], 4 / 3)
        self.assertEqual(evidence(343, 104)['pressure'], 1)

    def test_rounded_registers_and_reserved_shared_are_not_double_counted(self):
        proof = dict(grid=[384, 1, 1], block=[128, 1, 1], static_shared_bytes=40960)
        actual = dict(registers_per_thread=146, raw_shared_bytes=41984,
                      raw_local_bytes=0, stack_bytes_per_thread=0)
        result = compiled_score({'work': 300}, proof, actual)
        self.assertEqual(result['registers_allocated_per_cta'], 19456)
        self.assertEqual(result['shared_bytes_per_cta'], 41984)
        self.assertAlmostEqual(result['cost'], 400)
        self.assertFalse(result['local_or_stack'])
        actual['raw_local_bytes'] = 4
        self.assertTrue(compiled_score({'work': 300}, proof, actual)['local_or_stack'])
        with self.assertRaisesRegex(ValueError, 'checked plan'):
            compiled_score({'work': 300}, proof, dict(actual, raw_shared_bytes=24576))
