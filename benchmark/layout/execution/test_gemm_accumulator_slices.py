"""Regression checks on layouts emitted by the normal multi-anchor solver."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from gemm_accumulator_slices import emit, read_layouts, shared_bytes


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

    def test_incompatible_descriptor_or_writer(self):
        with self.assertRaisesRegex(ValueError,"descriptor"):
            read_layouts(self.ir.replace("swizzle = 32 : i64","swizzle = 0 : i64"))
        with self.assertRaisesRegex(ValueError,"row-major"):
            read_layouts(self.ir.replace("d0 * 2048 + d1 * 4","d0 * 4 + d1 * 1024"))

    def test_changed_atom_partition(self):
        lines = self.ir.splitlines()
        for i,line in enumerate(lines):
            if " = frisk.mma " in line:
                lines[i] = re.sub(r"grid = array<i64: 1, 2>","grid = array<i64: 2, 1>",line)
        with self.assertRaisesRegex(ValueError,"grid"):
            read_layouts("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
