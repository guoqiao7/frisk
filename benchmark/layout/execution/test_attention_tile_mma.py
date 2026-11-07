import unittest

from attention_tile_mma import (FRAGMENT_SLOTS, ROOT_BYTES, THREADS, build_d64_wgmma_tiles,
                                fragment_coordinate, score_width_for_prefix, shared_offsets_64x64)


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
            build_d64_wgmma_tiles(kv_tile=16)
        for roots in (dict(q=0, k=8192, v=8192), dict(q=16384, k=8192, v=16384),
                      dict(q=0, k=40960, v=49152)):
            with self.assertRaises(ValueError):
                build_d64_wgmma_tiles(roots, kv_tile=128)

    def test_wide_carrier_is_the_direct_sum_of_two_native_halves(self):
        narrow = build_d64_wgmma_tiles()
        wide = build_d64_wgmma_tiles(kv_tile=128)
        self.assertEqual(wide.fragment_columns, narrow.fragment_columns)
        self.assertEqual(len(wide.score_fragment_columns), 13)
        self.assertEqual(wide.score_fragment_columns[5], 1 << 6)
        seen = set()
        for thread in range(THREADS):
            for register in range(2 * FRAGMENT_SLOTS):
                hardware = thread * 2 * FRAGMENT_SLOTS + register
                logical = 0
                for bit, basis in enumerate(wide.score_fragment_columns):
                    if hardware >> bit & 1:
                        logical ^= basis
                row, column = fragment_coordinate(thread, register, kv_tile=128)
                narrow_row, narrow_column = fragment_coordinate(thread, register % FRAGMENT_SLOTS)
                self.assertEqual((row, column), (narrow_row, narrow_column + (register // FRAGMENT_SLOTS) * 64))
                self.assertEqual(logical, row * 128 + column)
                seen.add(logical)
        self.assertEqual(seen, set(range(64 * 128)))

    def test_wide_k_descriptor_covers_two_adjacent_sw128_roots(self):
        wide = build_d64_wgmma_tiles(kv_tile=128)
        self.assertEqual((wide.q_offset, wide.k_offset, wide.v_offset), (0, ROOT_BYTES, 2 * ROOT_BYTES))
        self.assertEqual(wide.shared_bytes, 3 * ROOT_BYTES)
        self.assertEqual(wide._k, wide._wide_k)
        one_root = shared_offsets_64x64()
        touched = set()
        for atom, desc in enumerate(wide._wide_k):
            stride = ((desc.fields >> 32) & 0x3fff) * 16
            self.assertEqual(stride, 8 * 128)
            self.assertEqual(desc.offset, wide.k_offset + atom * 32)
            self.assertEqual(desc.fields >> 62, 1)
            for key in range(128):
                base = desc.offset + (key // 8) * stride + (key % 8) * 128
                for depth in range(16):
                    address = (base + 2 * depth) ^ ((key % 8) << 4)
                    root, row = divmod(key, 64)
                    expected = wide.k_offset + root * ROOT_BYTES + one_root[row * 64 + atom * 16 + depth]
                    self.assertEqual(address, expected)
                    touched.add(address)
        self.assertEqual(touched, set(range(wide.k_offset, wide.shared_bytes, 2)))

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

    def test_wide_and_half_qk_share_pv_carrier_and_wait_before_reuse(self):
        plan = build_d64_wgmma_tiles(kv_tile=128)
        result = []
        temporary_names = iter(f"temp{i}" for i in range(100))
        temp = lambda kind: next(temporary_names)
        scores = [f"score{i}" for i in range(64)]
        output = [f"output{i}" for i in range(32)]
        packed = [f"p{i}" for i in range(16)]
        plan.emit_qk(result.append, temp, "smem", scores)
        plan.emit_qk_half(result.append, temp, "smem", scores[:32])
        plan.emit_pv(result.append, temp, "smem", packed, output)
        opcodes = [text.split(" ")[0] for text in result if text.startswith("wgmma.mma_async")]
        prefix = "wgmma.mma_async.sync.aligned."
        self.assertEqual(opcodes, [prefix + "m64n128k16.f32.f16.f16"] * 4
                         + [prefix + "m64n64k16.f32.f16.f16"] * 8)
        self.assertEqual(result.count("wgmma.wait_group.sync.aligned 0;"), 3)
        with self.assertRaises(ValueError):
            plan.emit_qk(result.append, temp, "smem", scores[:48])
        with self.assertRaises(ValueError):
            plan.emit_qk(result.append, temp, "smem", [scores[0]] * 64)


class AttentionCarrierStrategyTests(unittest.TestCase):
    def test_compact_kv32_restricts_the_native_carrier(self):
        plan = build_d64_wgmma_tiles(kv_tile=32)
        self.assertEqual((plan.q_offset, plan.k_offset, plan.v_offset, plan.shared_bytes),
                         (0, 8192, 12288, 16384))
        self.assertEqual(len(plan.score_fragment_columns), 11)
        coordinates = set()
        for thread in range(THREADS):
            for register in range(16):
                row, column = fragment_coordinate(thread, register, kv_tile=32)
                self.assertEqual((row, column), fragment_coordinate(thread, register))
                hardware = thread * 16 + register
                logical = 0
                for bit, basis in enumerate(plan.score_fragment_columns):
                    if hardware >> bit & 1:
                        logical ^= basis
                self.assertEqual(logical, row * 32 + column)
                coordinates.add(logical)
        self.assertEqual(coordinates, set(range(64 * 32)))
        self.assertEqual(len(plan._q), 4)
        self.assertEqual(len(plan._k), 4)
        self.assertEqual(len(plan._v), 2)
        self._check_key_descriptors(plan, 32)
        self._check_value_descriptors(plan, 32)

    def test_kv256_is_four_native_cosets_with_last_root_value_reuse(self):
        plan = build_d64_wgmma_tiles(kv_tile=256)
        self.assertEqual((plan.q_offset, plan.k_offset, plan.v_offset, plan.shared_bytes),
                         (0, 8192, 32768, 40960))
        self.assertEqual(len(plan.score_fragment_columns), 14)
        self.assertEqual(plan.score_fragment_columns[5:7], (64, 128))
        coordinates = set()
        for thread in range(THREADS):
            for register in range(128):
                row, column = fragment_coordinate(thread, register, kv_tile=256)
                base_row, base_column = fragment_coordinate(thread, register % FRAGMENT_SLOTS)
                self.assertEqual((row, column), (base_row, base_column + register // FRAGMENT_SLOTS * 64))
                hardware = thread * 128 + register
                logical = 0
                for bit, basis in enumerate(plan.score_fragment_columns):
                    if hardware >> bit & 1:
                        logical ^= basis
                self.assertEqual(logical, row * 256 + column)
                coordinates.add(logical)
        self.assertEqual(coordinates, set(range(64 * 256)))
        self._check_key_descriptors(plan, 256)
        self._check_value_descriptors(plan, 64)
        for roots in (dict(q=0, k=8192, v=16384), dict(q=8192,k=0,v=24576),
                      dict(q=0,k=32768,v=57344)):
            with self.assertRaises(ValueError):
                build_d64_wgmma_tiles(roots, kv_tile=256)

    def _check_key_descriptors(self, plan, width):
        descriptors = plan._wide_k if plan._wide_k is not None else plan._k
        original = shared_offsets_64x64()
        touched = set()
        for atom, descriptor in enumerate(descriptors):
            stride = ((descriptor.fields >> 32) & 0x3fff) * 16
            self.assertEqual(stride, 1024)
            for key in range(width):
                base = descriptor.offset + key // 8 * stride + key % 8 * 128
                for depth in range(16):
                    address = (base + 2 * depth) ^ ((key % 8) << 4)
                    block, row = divmod(key, 64)
                    expected = plan.k_offset + block * ROOT_BYTES + original[row * 64 + atom * 16 + depth]
                    self.assertEqual(address, expected)
                    touched.add(address)
        self.assertEqual(touched, set(range(plan.k_offset, plan.k_offset + width * 128, 2)))

    def _check_value_descriptors(self, plan, depth):
        original = shared_offsets_64x64()
        touched = set()
        for atom, descriptor in enumerate(plan._v):
            stride = ((descriptor.fields >> 32) & 0x3fff) * 16
            self.assertEqual(stride, 1024)
            for key in range(16):
                for feature in range(64):
                    offset = atom * 16 * 128 + feature * 2 + key % 8 * 128 + key // 8 * stride
                    offset ^= ((offset >> 7) & 7) << 4
                    address = plan.v_offset + offset
                    expected = plan.v_offset + original[(atom * 16 + key) * 64 + feature]
                    self.assertEqual(address, expected)
                    touched.add(address)
        self.assertEqual(touched, set(range(plan.v_offset, plan.v_offset + depth * 128, 2)))

    def test_native_prefix_shapes_preserve_full_carrier_owners(self):
        plan = build_d64_wgmma_tiles()
        for prefix in range(1, 65):
            width = score_width_for_prefix(prefix)
            self.assertIn(width, (8,16,32,64))
            self.assertGreaterEqual(width,prefix)
            self.assertTrue(width == 8 or width // 2 < prefix)
            code = []
            names = iter(f"t{i}" for i in range(100))
            plan.emit_qk_prefix(code.append,lambda kind: next(names),"smem",
                                [f"score{i}" for i in range(width // 2)],prefix)
            mma = [line for line in code if line.startswith("wgmma.mma_async")]
            self.assertEqual(len(mma),4)
            self.assertTrue(all(line.startswith(f"wgmma.mma_async.sync.aligned.m64n{width}k16")
                                for line in mma))
            self.assertEqual(code.count("wgmma.wait_group.sync.aligned 0;"),1)
            for thread in range(THREADS):
                for register in range(width // 2):
                    self.assertEqual(fragment_coordinate(thread,register,kv_tile=width),
                                     fragment_coordinate(thread,register))
        for prefix in (False,0,-1,65,1.5):
            with self.assertRaises(ValueError):
                score_width_for_prefix(prefix)
        with self.assertRaises(ValueError):
            plan.emit_qk_prefix(lambda line: None, lambda kind: "t", "smem", ["s"], 1)
        compact = build_d64_wgmma_tiles(kv_tile=32)
        with self.assertRaises(ValueError):
            compact.emit_qk_prefix(lambda line: None, lambda kind: "t", "smem",
                                   [f"s{i}" for i in range(32)], 33)

    def test_compact_and_full_score_native_emission(self):
        for width,score_count,packed_count,pv_atoms in ((32,16,8,2),(256,128,16,4)):
            plan = build_d64_wgmma_tiles(kv_tile=width)
            code = []
            names = iter(f"t{i}" for i in range(100))
            temp=lambda kind: next(names)
            plan.emit_qk(code.append,temp,"smem",[f"s{i}" for i in range(score_count)])
            plan.emit_pv(code.append,temp,"smem",[f"p{i}" for i in range(packed_count)],
                         [f"o{i}" for i in range(32)])
            mma=[line.split(" ")[0] for line in code if line.startswith("wgmma.mma_async")]
            self.assertEqual(mma,[f"wgmma.mma_async.sync.aligned.m64n{width}k16.f32.f16.f16"]*4
                             + ["wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"]*pv_atoms)
            self.assertEqual(code.count("wgmma.wait_group.sync.aligned 0;"),2)
        with self.assertRaises(ValueError):
            build_d64_wgmma_tiles(kv_tile=32).emit_qk_half(
                lambda line: None, lambda kind: "t", "smem", [f"s{i}" for i in range(32)])
