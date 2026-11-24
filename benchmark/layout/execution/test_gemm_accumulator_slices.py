"""Regression checks on layouts emitted by the normal multi-anchor solver."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from gemm_accumulator_slices import GemmLayouts, emit, rank_residencies, read_layouts, shared_bytes


class AccumulatorSlicesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        opt = os.environ.get("FRISK_OPT") or shutil.which("frisk-opt")
        if not opt:
            raise unittest.SkipTest("frisk-opt is required to produce the input layouts")
        fixture = Path(__file__).resolve().parents[3]/"test/Transforms/large-output-layout.mlir"
        cls.ir = subprocess.check_output([opt,str(fixture),"--frisk-multi-anchor-layout-pipeline",
                                          "-verify-each"],text=True)

    def test_partition_and_odd_k_tail(self):
        plan = read_layouts(self.ir)
        self.assertEqual(len(plan.origins),4)
        self.assertEqual(shared_bytes(plan),65536)
        self.assertIsNotNone(plan.reduced_writers)
        for sliced,count in ((True,4),(False,1)):
            src = emit(plan,208,sliced=sliced,wait_group=2)
            self.assertEqual(src.count("float acc["),count)
            self.assertEqual(src.count("kt=192; kt<208; kt+=16"),count)
            self.assertEqual(src.count("issue2(acc["),4)
            self.assertEqual(src.count("issue1(acc["),4)
        single = emit(plan,208,staged_k_tiles=1)
        self.assertEqual(single.count("kt=0; kt<208; kt+=16"),4)
        with self.assertRaises(ValueError):
            emit(plan,208,staged_k_tiles=1,wait_group=2)
        with self.assertRaises(ValueError):
            emit(plan,207)

    def test_all_widths_use_the_same_copy_footprint_as_the_emitter(self):
        plan = read_layouts(self.ir)
        rank = rank_residencies(plan, 208)
        self.assertEqual({row["resident_atoms"] for row in rank}, {1, 2, 3, 4})
        for row in rank:
            width = row["resident_atoms"]
            a = b = 0
            for i in range(0, len(plan.origins), width):
                rows, cols = set(), set()
                for r0, c0, r1, c1 in plan.origins[i:i+width]:
                    for r, c in ((r0, c0), (r1, c1)):
                        rows.update(range(r, r+64))
                        cols.update(range(c, c+plan.atom))
                a += len(rows) * 208 * 2
                b += len(cols) * 208 * 2
            self.assertEqual((a, b), (row["a_bytes"], row["b_bytes"]))
        source = emit(plan, 208, resident_atoms=3)
        self.assertEqual(source.count("float acc[3]"), 1)
        self.assertEqual(source.count("float acc[1]"), 1)
        self.assertEqual(emit(plan, 208), emit(plan, 208, resident_atoms=rank[0]["resident_atoms"]))
        self.assertEqual(emit(plan, 208, sliced=False), emit(plan, 208, resident_atoms=4))
        with self.assertRaisesRegex(ValueError, "either"):
            emit(plan, 208, sliced=False, resident_atoms=2)

    def test_saturated_and_single_wave_residency(self):
        origins = tuple((64*i, 0, 512+64*i, 0) for i in range(8))
        plan = GemmLayouts(1024, 128, 128, origins, None)
        thin = rank_residencies(plan, 128, grid_ctas=12)
        busy = rank_residencies(plan, 128, grid_ctas=132)
        self.assertEqual(thin[0]["target_ctas"], 1)
        self.assertEqual(busy[0]["target_ctas"], 2)
        self.assertTrue(any(c["residency_cost_units"] for c in busy))
        self.assertFalse(any(c["residency_cost_units"] for c in thin))
        longer = rank_residencies(plan, 224, grid_ctas=132)
        self.assertLess(longer[0]["resident_atoms"], thin[0]["resident_atoms"])
        self.assertEqual(busy[0]["estimated_spill_bytes"], 0)
        self.assertEqual(next(c for c in busy if c["resident_atoms"] == 8)["repeated_ab_bytes"], 0)
        for bad in (0, True, 9):
            with self.assertRaises(ValueError):
                emit(plan, 128, resident_atoms=bad)
        with self.assertRaises(ValueError):
            rank_residencies(plan, 128, sm_count=0)


    def test_incompatible_descriptor_or_writer(self):
        with self.assertRaisesRegex(ValueError,"descriptor"):
            read_layouts(self.ir.replace("swizzle = 32 : i64","swizzle = 0 : i64"))
        with self.assertRaisesRegex(ValueError,"row-major"):
            read_layouts(self.ir.replace("d0 * 2048 + d1 * 4","d0 * 4 + d1 * 1024"))

    def test_small_unspilled_tile_can_change_with_loop_work(self):
        plan = GemmLayouts(256, 128, 128, ((0, 0, 128, 0), (64, 0, 192, 0)), None)
        short = rank_residencies(plan, 96, grid_ctas=132)
        long = rank_residencies(plan, 320, grid_ctas=132)
        for candidate in short + long:
            self.assertEqual(candidate["estimated_spill_bytes"], 0)
        self.assertEqual(short[0]["resident_atoms"], 2)
        self.assertEqual(long[0]["resident_atoms"], 1)
        self.assertEqual(rank_residencies(plan, 320, grid_ctas=12)[0]["resident_atoms"], 2)

    def test_changed_atom_partition(self):
        lines = self.ir.splitlines()
        for i,line in enumerate(lines):
            if " = frisk.mma " in line:
                lines[i] = re.sub(r"grid = array<i64: 1, 2>","grid = array<i64: 2, 1>",line)
        with self.assertRaisesRegex(ValueError,"grid"):
            read_layouts("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
