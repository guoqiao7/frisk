"""Checked GPU promotion regressions; prebuilt tools only, no CUDA execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

GENERATOR, OPT = sys.argv[1:3]
del sys.argv[1:3]


class PromotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.original = subprocess.check_output([GENERATOR, "identity_f32"], text=True)
        cls.source = cls.root / "source.mlir"
        cls.source.write_text(cls.original)
        cls.layout = subprocess.check_output([OPT, str(cls.source), "-frisk-layout-pipeline"], text=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def promote(self, text):
        self.source.write_text(text)
        output = self.root / "promoted.mlir"
        return subprocess.run([GENERATOR, "--promote", str(self.source), str(output)],
                              text=True, capture_output=True), output

    def test_matching_root_contract_preserved_as_explicit_views(self):
        result, path = self.promote(self.layout)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = path.read_text()
        self.assertIn("gpu.func", text)
        self.assertIn("frisk.layout_view", text)
        self.assertNotIn("frisk.storage_contract", text)
        self.assertIn("known_block_size = array<i32: 32, 1, 1>", text)

    def test_mismatched_root_view_rejected(self):
        lines = self.layout.splitlines()
        for index, line in enumerate(lines):
            if "frisk.layout_view" in line:
                lines[index] = line.replace("alignment = 4", "alignment = 8")
                break
        result, _ = self.promote("\n".join(lines))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("root contract cannot be represented", result.stderr)

    def test_unrepresented_root_contract_rejected(self):
        text = "\n".join(line for line in self.layout.splitlines()
                         if not ("frisk.layout_view %arg1" in line or "frisk.tile_store" in line))
        result, _ = self.promote(text)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrepresented storage root contract", result.stderr)

    def test_non_view_root_use_rejected(self):
        result, _ = self.promote(self.layout.replace("frisk.tile_load %0", "frisk.tile_load %arg0"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("root contract cannot be represented", result.stderr)

    def test_invalid_capacity_rejected_before_promotion(self):
        result, _ = self.promote(self.layout.replace("memref<32xf32", "memref<16xf32"))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr)

    def test_unknown_case_rejected(self):
        result = subprocess.run([GENERATOR, "unknown_f32"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
