import unittest

from access_plan import plan_access, plan_copy_owners


def rows(offsets, bases=(0, 64, 128, 192)):
    return [[base + offset for base in bases] for offset in offsets]


class AccessPlanTest(unittest.TestCase):
    def verify_offsets(self, plan, offsets, element_bytes, alignment, masks=None):
        used = []
        for group in plan.groups:
            base = plan.bases[group.base_index]
            self.assertIn(base, map(tuple, offsets))
            if masks is None:
                self.assertIsNone(group.owner_mask)
            for index, slot in enumerate(group.slots):
                delta = group.byte_offset + index * element_bytes
                self.assertEqual([x + delta for x in base], offsets[slot])
                if masks is not None:
                    self.assertEqual(group.owner_mask, masks[slot])
                used.append(slot)
            if group.width_bytes != 1:
                self.assertEqual(group.width_bytes, len(group.slots) * element_bytes)
                self.assertGreaterEqual(alignment, group.width_bytes)
                self.assertTrue(all(
                    (x + group.byte_offset) % group.width_bytes == 0
                    for lane, x in enumerate(base)
                    if masks is None or group.owner_mask >> lane & 1
                ))
        self.assertEqual(sorted(used), [
            i for i in range(len(offsets)) if masks is None or masks[i]
        ])

    def test_reordered_slots_cover_tail_without_overread(self):
        offsets = rows([12, 0, 8, 4, 24, 20, 16])
        plan = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=220)
        self.verify_offsets(plan, offsets, 4, 16)
        self.assertEqual(len(plan.bases), 1)
        self.assertEqual(
            [(g.slots, g.width_bytes) for g in sorted(plan.groups, key=lambda g: g.byte_offset)],
            [((1, 3, 2, 0), 16), ((6, 5), 8), ((4,), 4)],
        )

    def test_fp16_vector_widths_and_scalar_limit(self):
        offsets = rows(range(0, 16, 2))
        for alignment, max_bytes, expected in ((16, 16, 16), (16, 8, 8), (4, 16, 4), (16, 2, 2)):
            with self.subTest(alignment=alignment, max_bytes=max_bytes):
                plan = plan_access(offsets, element_bytes=2, root_alignment=alignment,
                                   root_bytes=208, max_vector_bytes=max_bytes)
                self.verify_offsets(plan, offsets, 2, alignment)
                self.assertEqual([g.width_bytes for g in plan.groups], [expected] * (16 // expected))

    def test_prepacked_copy_reuses_checked_aligned_segments(self):
        offsets = rows([16, 0])
        plan = plan_access(offsets, element_bytes=16, root_alignment=16, root_bytes=224)
        self.verify_offsets(plan, offsets, 16, 16)
        self.assertEqual(len(plan.bases), 1)
        self.assertEqual([g.width_bytes for g in plan.groups], [16, 16])
        with self.assertRaisesRegex(ValueError, "escapes"):
            plan_access(offsets, element_bytes=16, root_alignment=16, root_bytes=223)

    def test_copy_owner_reassigns_whole_paired_vectors(self):
        # Shared bank groups use the third through fifth bits of the source
        # vector ordinal. The original owner makes consecutive lane IDs
        # advance by eight vectors instead of one.
        source = [[16 * (8 * lane + slot) for lane in range(128)] for slot in range(8)]
        target = [[16 * ((index & ~31) | ((index & 3) << 3) | ((index >> 2) & 7))
                   for index in (offset // 16 for offset in row)] for row in source]
        (planned_source, planned_target), receipt = plan_copy_owners(source, target)
        pairs = lambda src, dst: sorted(pair for a, b in zip(src, dst) for pair in zip(a, b))
        self.assertEqual(pairs(source, target), pairs(planned_source, planned_target))
        self.assertEqual(receipt["lane_bit_order"], [2, 3, 4, 0, 1])
        self.assertEqual(receipt["previous_global_sectors_32"], [1024, 1024])
        self.assertEqual(receipt["global_sectors_32"], [512, 544])
        self.assertGreater(receipt["previous_shared_excess_wavefronts"], 0)
        self.assertEqual(receipt["shared_excess_wavefronts"], 0)
        self.assertIsNone(plan_copy_owners(planned_source, planned_target))

    def test_copy_owner_falls_back_for_gaps_or_nonaffine_destination(self):
        source = [list(range(0, 16 * 128, 16))]
        target = [source[0].copy()]
        target[0][0], target[0][1] = target[0][1], target[0][0]
        self.assertIsNone(plan_copy_owners(source, target))
        source[0][-1] += 16
        self.assertIsNone(plan_copy_owners(source, source))

    def test_weak_or_misaligned_layout_remains_byte_wise(self):
        for alignment, offsets in ((1, rows([0, 4])), (16, rows([1, 5]))):
            with self.subTest(alignment=alignment):
                plan = plan_access(offsets, element_bytes=4, root_alignment=alignment, root_bytes=204)
                self.verify_offsets(plan, offsets, 4, alignment)
                self.assertEqual([g.width_bytes for g in plan.groups], [1, 1])

    def test_alignment_checks_every_accessing_lane(self):
        offsets = rows([0, 4], bases=(0, 12, 32, 48))
        load = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=56)
        self.verify_offsets(load, offsets, 4, 16)
        self.assertEqual([g.width_bytes for g in load.groups], [4, 4])
        masks = (0b1101, 0b1101)
        store = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=56, owner_masks=masks)
        self.verify_offsets(store, offsets, 4, 16, masks)
        self.assertEqual([g.width_bytes for g in store.groups], [8])

    def test_xor_relation_is_not_constant_integer_delta(self):
        offsets = [[0, 4, 8, 12], [4, 0, 12, 8]]
        plan = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=16)
        self.verify_offsets(plan, offsets, 4, 16)
        self.assertEqual(len(plan.bases), 2)
        self.assertEqual([g.width_bytes for g in plan.groups], [4, 4])

    def test_stores_require_identical_owner_mask_for_vector(self):
        offsets = rows([0, 4, 8])
        for masks, widths in (((5, 5, 0), [8]), ((5, 10, 0), [4, 4])):
            with self.subTest(masks=masks):
                plan = plan_access(offsets, element_bytes=4, root_alignment=16,
                                   root_bytes=200, owner_masks=masks)
                self.verify_offsets(plan, offsets, 4, 16, masks)
                self.assertEqual([g.width_bytes for g in plan.groups], widths)
        dead = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=0, owner_masks=(0, 0, 0))
        self.assertEqual((dead.bases, dead.groups), ((), ()))

    def test_store_owner_exclusion_does_not_mask_reads(self):
        offsets = [[0, 4, 128, 132], [0, 4, 128, 132]]
        store = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=8, owner_masks=(1, 2))
        self.verify_offsets(store, offsets, 4, 16, (1, 2))
        with self.assertRaisesRegex(ValueError, "escapes"):
            plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=8)
        duplicate = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=136)
        self.verify_offsets(duplicate, offsets, 4, 16)

    def test_rejects_duplicate_and_partially_overlapping_active_writers(self):
        for offsets, masks in (([[0, 0]], (3,)), ([[0, 16], [32, 2]], (1, 2))):
            with self.subTest(offsets=offsets):
                with self.assertRaisesRegex(ValueError, "owners overlap"):
                    plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=36, owner_masks=masks)

    def test_out_of_range_and_invalid_masks_fail_closed(self):
        offsets = [[0, 4]]
        for masks in ((4,), (-1,), (True,), (), (1, 2), 1):
            with self.subTest(masks=masks):
                with self.assertRaisesRegex(ValueError, "owner mask"):
                    plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=8, owner_masks=masks)
        with self.assertRaisesRegex(ValueError, "escapes"):
            plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=7, owner_masks=(2,))
        with self.assertRaisesRegex(ValueError, "32-bit"):
            plan_access([[0, 1 << 32]], element_bytes=4, root_alignment=16, root_bytes=1 << 36)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            plan_access([[0, 4], [8]], element_bytes=4, root_alignment=16, root_bytes=12)

    def test_displacement_reserves_space_for_byte_fallback(self):
        offsets = [[0, 4], [0x7ffffffc, 0x80000000], [0x7fffffff, 0x80000003]]
        plan = plan_access(offsets, element_bytes=4, root_alignment=1, root_bytes=1 << 32)
        self.verify_offsets(plan, offsets, 4, 1)
        self.assertEqual(len(plan.bases), 2)
        self.assertEqual([g.byte_offset for g in plan.groups], [0, 0x7ffffffc, 0])


if __name__ == "__main__":
    unittest.main()
