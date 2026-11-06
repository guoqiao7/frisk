import unittest

from attention_tile_mma import (FRAGMENT_SLOTS, ROOT_BYTES, THREADS, build_d64_wgmma_tiles,
                                fragment_coordinate, shared_offsets_64x64)


class AttentionTileMmaTests(unittest.TestCase):
    def test_common_carrier_is_a_bijection_and_preserves_copy16(self):
        plan = build_d64_wgmma_tiles()
        offsets = shared_offsets_64x64()
        self.assertEqual((plan.q_offset, plan.k_offset, plan.v_offset), (0, 8192, 16384))
        self.assertEqual(plan.shared_bytes, 24576)
        self.assertEqual(set(offsets), set(range(0, ROOT_BYTES, 2)))
        for thread in range(THREADS):
            for register in range(FRAGMENT_SLOTS):
                row, column = fragment_coordinate(thread, register)
                hardware = thread * FRAGMENT_SLOTS + register
                logical = 0
                for bit, basis in enumerate(plan.fragment_columns):
                    if hardware >> bit & 1:
                        logical ^= basis
                self.assertEqual(row * 64 + column, logical)
        self.assertEqual(offsets[64:72], tuple(range(144, 160, 2)))
        self.assertEqual(offsets[72:80], tuple(range(128, 144, 2)))

    def test_only_aligned_disjoint_bounded_roots_are_admitted(self):
        for roots in (dict(q=0, k=1024, v=16384), dict(q=0, k=8192, v=16385),
                      dict(q=0, k=8192, v=49152), dict(q=0, k=8192), dict(q=0, k=8192, v=True)):
            with self.assertRaises(ValueError):
                build_d64_wgmma_tiles(roots)
        with self.assertRaises(ValueError):
            build_d64_wgmma_tiles(kv_tile=128)

    def test_wgmma_consumes_native_packs_and_preserves_order(self):
        plan = build_d64_wgmma_tiles()
        result = []
        temporary_names = iter(f"temp{i}" for i in range(100))
        temp = lambda kind: next(temporary_names)
        scores = [f"score{i}" for i in range(32)]
        output = [f"output{i}" for i in range(32)]
        packed = [f"p{i}" for i in range(16)]
        plan.emit_qk(result.append, temp, "smem", scores)
        plan.emit_pv(result.append, temp, "smem", packed, output)
        grouped = [i for i, text in enumerate(result) if text.startswith("wgmma.fence")]
        self.assertEqual(len(grouped), 2)
        for start in grouped:
            self.assertTrue(all(text.startswith("wgmma.mma_async.sync.aligned.m64n64k16")
                                for text in result[start + 1:start + 5]))
            self.assertEqual(result[start + 5:start + 7],
                             ["wgmma.commit_group.sync.aligned;", "wgmma.wait_group.sync.aligned 0;"])
        for atom, text in enumerate(result[grouped[1] + 1:grouped[1] + 5]):
            self.assertIn("{" + ", ".join(packed[atom * 4:(atom + 1) * 4]) + "}", text)
        self.assertFalse(any(text.startswith("mov.") for text in result))
        with self.assertRaises(ValueError):
            plan.emit_pv(result.append, temp, "smem", output[:16], output)
