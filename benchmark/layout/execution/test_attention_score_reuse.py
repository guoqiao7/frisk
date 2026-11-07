import unittest

from attention_score_reuse import prove_score_prefix
from attention_tile_mma import FRAGMENT_SLOTS, THREADS, TILE, build_d64_wgmma_tiles


class ScorePrefixTests(unittest.TestCase):
    def test_every_prefix_and_holder_matches_native_score_coordinates(self):
        columns = build_d64_wgmma_tiles().fragment_columns
        all_holders = (1 << THREADS) - 1
        for prefix in range(1, TILE + 1):
            with self.subTest(prefix=prefix):
                proof = prove_score_prefix(columns, prefix)
                self.assertEqual((proof.kv_width, proof.valid_columns), (64, prefix))
                self.assertEqual(len(proof.holder_masks), FRAGMENT_SLOTS)
                expected_held = []
                for slot, (mask, cutoff) in enumerate(zip(proof.holder_masks, proof.guard_cutoffs, strict=True)):
                    expected = 0
                    for thread in range(THREADS):
                        column = (thread % 4) * 2 + slot % 2 + (slot // 4) * 8
                        if column < prefix:
                            expected |= 1 << thread
                    self.assertEqual(mask, expected)
                    if mask:
                        expected_held.append(slot)
                    self.assertEqual(cutoff == 0, mask == 0)
                    self.assertEqual(cutoff == 7, mask == all_holders)
                self.assertEqual(proof.held_slots, tuple(expected_held))
                self.assertEqual(sum(mask.bit_count() for mask in proof.holder_masks), 64 * prefix)
        self.assertEqual(prove_score_prefix(columns, 1).held_slots, (0, 2))

    def test_non_native_maps_and_invalid_prefixes_fail_closed(self):
        columns = build_d64_wgmma_tiles().fragment_columns
        # Swapping axes preserves rank but changes native holder identities.
        altered = (columns[1], columns[0], *columns[2:])
        for invalid in (altered, columns[:-1], (True, *columns[1:])):
            with self.assertRaisesRegex(ValueError, "checked native GF"):
                prove_score_prefix(invalid, 1)
        for invalid in (0, -1, 65, True, 1.0):
            with self.assertRaisesRegex(ValueError, "nonempty prefix"):
                prove_score_prefix(columns, invalid)
        with self.assertRaisesRegex(ValueError, "native KV64"):
            prove_score_prefix(columns, 1, kv_width=128)


if __name__ == "__main__":
    unittest.main()

