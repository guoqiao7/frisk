import importlib.util
from pathlib import Path
import unittest
import tempfile

SOURCE = Path(__file__).resolve().parents[2] / "benchmark/layout/conversion/compile_conversion_case.py"
spec = importlib.util.spec_from_file_location("conversion_compiler", SOURCE)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


class CompilerTests(unittest.TestCase):
    def test_reference_digest_is_bound_to_build_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "reference.cu"
            source.write_text("built reference")
            manifest = root / "manifest"
            expected = compiler.digest(source)
            manifest.write_text("benchmark/reference.cu:" + expected + "\n")
            self.assertEqual(compiler.source_digest_from_manifest(manifest, "benchmark/reference.cu", source), expected)
            source.write_text("unbuilt edit")
            with self.assertRaisesRegex(ValueError, "does not match"):
                compiler.source_digest_from_manifest(manifest, "benchmark/reference.cu", source)
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                compiler.source_digest_from_manifest(manifest, "missing.cu", source)
    def test_mlir_ptx_string_decoding(self):
        self.assertEqual(compiler.extract_ptx('gpu.binary @x [#gpu.object<assembly = "a\\0Ab\\22c\\5C">]'), b'a\nb"c\\')

    def test_no_or_multiple_gpu_objects_rejected(self):
        for text in ('module {}', 'assembly = "a" assembly = "b"'):
            with self.assertRaises(ValueError):
                compiler.extract_ptx(text)

    def test_unresolved_adapter_rejected(self):
        for text in ('builtin.unrealized_conversion_cast', 'tensor<32xf32>', 'frisk.convert_layout'):
            with self.assertRaises(ValueError):
                compiler.check_adapter(text)


if __name__ == "__main__":
    unittest.main()
