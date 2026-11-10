"""Regression checks for KV32's physically smaller Shared operand roots."""

from dataclasses import replace
import unittest

from attention_narrow import _narrow_copy_groups
from attention_tile_mma import shared_offsets_64x64
from valid_domain import dense_copy_execution, plan_guarded_copy_owners


def copy_plan(policy='gf2', rows=64):
    return plan_guarded_copy_owners(
        dense_copy_execution((64, 64)), valid_shape=(rows, 64),
        source_row_stride_bytes=128, source_span_bytes=rows * 128,
        destination_offsets=shared_offsets_64x64(), destination_span_bytes=8192,
        owner_policy=policy)


class NarrowCopyTests(unittest.TestCase):
    def test_selected_vectors_initialize_exact_compact_key_and_value_roots(self):
        for policy in ('execution', 'gf2'):
            with self.subTest(policy=policy):
                groups = _narrow_copy_groups(copy_plan(policy))
                sources = {offset for group in groups for lane, offset in enumerate(group.source_offsets)
                           if group.active_mask >> lane & 1}
                destinations = {offset for group in groups for lane, offset in enumerate(group.destination_offsets)
                                if group.active_mask >> lane & 1}
                self.assertEqual(sources, set(range(0, 4096, 16)))
                self.assertEqual(destinations, set(range(0, 4096, 16)))
                self.assertEqual(len(groups), 2 if policy == 'gf2' else 4)
                self.assertTrue(all(row < 32 for group in groups for lane, row in enumerate(group.rows)
                                    if group.active_mask >> lane & 1))

    def test_padded_half_root_cannot_keep_a_full64_issue_group(self):
        original = copy_plan()
        bad = replace(original.groups[0], destination_offsets=(4096, *original.groups[0].destination_offsets[1:]))
        with self.assertRaisesRegex(ValueError, 'compact root|native ownership'):
            _narrow_copy_groups(replace(original, groups=(bad, *original.groups[1:])))

    def test_partial_owners_and_incomplete_copy_coverage_are_rejected(self):
        original = copy_plan()
        bad = replace(original.groups[0], active_mask=(1 << 127) - 1)
        with self.assertRaisesRegex(ValueError, 'inactive or incomplete'):
            _narrow_copy_groups(replace(original, groups=(bad, *original.groups[1:])))
        with self.assertRaisesRegex(ValueError, 'every native operand'):
            _narrow_copy_groups(replace(original, groups=original.groups[1:]))
        with self.assertRaisesRegex(ValueError, 'complete aligned'):
            _narrow_copy_groups(copy_plan(rows=32))


if __name__ == '__main__':
    unittest.main()

