"""Regress physical MMA row owners and inverse writes for masked query tails."""

from unittest import TestCase

from attention_copy_layout import native_copy_plan
from attention_fragment_layout import NATIVE_FRAGMENT, ROW_LANE_REGISTER, checked_fragment_layout, output_store_evidence
from attention_tile_mma import shared_offsets_64x64
from tiled_attention import _output_distribution
from valid_domain import dense_copy_execution, plan_guarded_store


class AttentionFragmentLayoutTests(TestCase):
    def test_exchanges_lane_and_register_owners(self):
        canonical = _output_distribution()
        alternate = checked_fragment_layout(ROW_LANE_REGISTER)
        physical = alternate.accumulator_distribution(canonical)
        # At feature column zero, logical row 1 moves from lane 4 to
        # register 2 of lane 0. Logical row 8 moves in the other direction.
        for row, old, new in ((1, 4 * 32, 2), (8, 2, 4 * 32)):
            self.assertEqual(canonical['first'][row * 64], old)
            self.assertEqual(physical['first'][row * 64], new)
        self.assertEqual(sorted(physical['logical']), list(range(4096)))
        self.assertIs(checked_fragment_layout(NATIVE_FRAGMENT).accumulator_distribution(canonical), canonical)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            checked_fragment_layout('row-unknown')

    def test_copy_and_writeback_cover_the_same_logical_prefix(self):
        layout = checked_fragment_layout(ROW_LANE_REGISTER)
        distribution = layout.accumulator_distribution(_output_distribution())
        execution = dense_copy_execution((64, 64))
        for swizzle in (64, 128):
            table = shared_offsets_64x64(swizzle)
            query_offsets = layout.query_offsets(table)
            for rows in (1, 7, 9, 31, 63, 64):
                with self.subTest(swizzle=swizzle, rows=rows):
                    copy = native_copy_plan(execution, rows, owner_policy='gf2',
                                            destination_offsets=query_offsets)
                    writes = {(src, dst) for group in copy.groups
                              for lane, (src, dst) in enumerate(zip(group.source_offsets,
                                                                  group.destination_offsets, strict=True))
                              if group.active_mask >> lane & 1}
                    expected = {(2 * (row * 64 + col), table[layout.rows[row] * 64 + col])
                                for row in range(rows) for col in range(0, 64, 8)}
                    self.assertEqual(writes, expected)
                    stores = plan_guarded_store(distribution, valid_shape=(rows, 64),
                        destination_row_stride_bytes=256, destination_span_bytes=rows * 256)
                    written = {addr for group in stores.groups for lane, addr in
                               enumerate(group.destination_offsets) if group.active_mask >> lane & 1}
                    self.assertEqual(written, set(range(0, rows * 256, 4)))
                    evidence = output_store_evidence({rows: stores}, vector_store=True)
                    self.assertEqual(evidence[rows]['active_elements'], rows * 64)
