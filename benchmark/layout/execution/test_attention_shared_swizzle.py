"""Regression for descriptor-native SW64 on the D64 checked carrier."""
from unittest import TestCase

from attention_copy_layout import native_copy_plan
from attention_copy_schedule import prove_copy_schedule
from attention_tile_mma import (ROOT_BYTES, _checked_descriptors, build_d64_wgmma_tiles,
                                shared_offsets_64x64)
from valid_domain import dense_copy_execution


class SharedSwizzleTests(TestCase):
    def test_sw64_descriptor_fields_and_root_disjointness(self):
        mma = build_d64_wgmma_tiles(shared_swizzle=64)
        evidence = mma.shared_layout_evidence()
        self.assertEqual(evidence['shared_swizzle'], 64)
        self.assertEqual(mma.shared_bytes, 3 * ROOT_BYTES)
        k_fields = 1 << 63 | 32 << 32 | 1 << 16
        v_fields = 1 << 63 | 32 << 32 | 256 << 16
        q_starts = (0, 32, 4096, 4128)
        q, k, v = (evidence['shared_qkv_descriptor_fields'][name] for name in ('q','k','v'))
        self.assertEqual(q, [(start, k_fields) for start in q_starts])
        self.assertEqual(k, [(ROOT_BYTES + start, k_fields) for start in q_starts])
        self.assertEqual(v, [(2 * ROOT_BYTES + start, v_fields)
                             for start in (0, 1024, 2048, 3072)])
        for rows in (8, 16, 32):
            self.assertEqual(_checked_descriptors(ROOT_BYTES, k_major=True,
                             mn_extent=rows, swizzle=64), mma._k)
        for width in (32, 128, 256):
            with self.assertRaisesRegex(ValueError, 'complete KV64'):
                build_d64_wgmma_tiles(kv_tile=width, shared_swizzle=64)

    def test_sw64_copy_proof_preserves_vectors_and_high_planes(self):
        layout = shared_offsets_64x64(64)
        execution = dense_copy_execution((64, 64))
        for rows in (1, 9, 35, 64):
            plan = native_copy_plan(execution, rows, owner_policy='gf2', shared_swizzle=64)
            self.assertIsNotNone(prove_copy_schedule(plan, native_offsets=layout))
            pairs = {(src,dst) for group in plan.groups
                     for lane, (src,dst) in enumerate(zip(group.source_offsets,
                                                        group.destination_offsets,strict=True))
                     if group.active_mask >> lane & 1}
            self.assertEqual(pairs, {(slot * 2, layout[slot]) for slot in range(0, rows * 64, 8)})
        # Even one row occupies both planes, so row count * row bytes is
        # insufficient as a physical bounds check for a narrow SW64 copy.
        self.assertGreater(max(layout[:64]), 4096)
