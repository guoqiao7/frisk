"""Task22 importer tests against the explicitly selected build, never installed FFI."""
import argparse
import importlib
import importlib.util
import pathlib
import subprocess
import sys
import types
import unittest


def load_build(build_dir):
    package_dir = pathlib.Path(__file__).resolve().parents[2] / "python/frisk"
    products = sorted(build_dir.glob("frisk_ffi*.so"))
    if len(products) != 1:
        raise RuntimeError(f"expected one built frisk_ffi module, got {products}")
    package = types.ModuleType("frisk")
    package.__path__ = [str(package_dir)]
    sys.modules["frisk"] = package
    spec = importlib.util.spec_from_file_location("frisk.frisk_ffi", products[0])
    extension = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = extension
    spec.loader.exec_module(extension)
    assert pathlib.Path(extension.__file__).resolve().parent == build_dir
    helpers = importlib.import_module("frisk.common.utils")
    return extension.ir, helpers


class LegacyNormalizationTest(unittest.TestCase):
    def normalize(self, session, expected_error=None):
        source = str(session.module).replace(
            "module {", 'module attributes {frisk.target = "sm_90a"} {', 1)
        result = subprocess.run(
            [str(BUILD_DIR / "bin/frisk-opt"), "-frisk-normalize-layout-ir"],
            input=source, text=True, capture_output=True, check=False)
        if expected_error:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected_error, result.stderr)
        else:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def gemm(self, dtype="fp16", trans_a=False, trans_b=False,
             accumulator="fp32", semantics="tensor_v1"):
        session = HELPERS.create_session()
        a_shape = [16, 64] if trans_a else [64, 16]
        b_shape = [64, 16] if trans_b else [16, 64]
        _, entry = session.new_kernel("gemm", [
            session.memref_type(a_shape, dtype, space=IR.MemorySpace.SHARED),
            session.memref_type(b_shape, dtype, space=IR.MemorySpace.SHARED),
            session.memref_type([64, 64], accumulator)])
        c = session.alloc_buffer([64, 64], accumulator)
        session.create_gemm(entry.get_argument(0), entry.get_argument(1), c,
                            trans_a=trans_a, trans_b=trans_b, clear_accum=True,
                            legacy_semantics=semantics)
        session.builder.create_copy_op(c, entry.get_argument(2), [], [])
        return session

    def test_all_transposes_and_input_dtypes(self):
        for dtype in ("fp16", "bf16"):
            for trans_a in (False, True):
                for trans_b in (False, True):
                    with self.subTest(dtype=dtype, trans_a=trans_a, trans_b=trans_b):
                        output = self.normalize(self.gemm(dtype, trans_a, trans_b))
                        self.assertIn("frisk.mma", output)
                        self.assertIn("frisk.tile_store", output)
                        self.assertNotIn("frisk.gemm", output)

    def test_default_has_no_implicit_semantics_marker(self):
        session = self.gemm(semantics=None)
        self.assertNotIn("frisk.legacy_semantics", str(session.module))
        self.normalize(session, "legacy-math-contract")

    def test_legacy_f16_accumulator_stays_parseable_but_cannot_normalize(self):
        session = self.gemm(accumulator="fp16")
        source = str(session.module)
        parsed = subprocess.run([str(BUILD_DIR / "bin/frisk-opt")],
                                input=source, text=True, capture_output=True)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.normalize(session, "legacy-gemm-accumulator")

    def test_builder_rejects_effective_shape_mismatch(self):
        session = HELPERS.create_session()
        _, entry = session.new_kernel("bad", [
            session.memref_type([64, 16]), session.memref_type([8, 64]),
            session.memref_type([64, 64], "fp32")])
        with self.assertRaisesRegex(ValueError, "legacy-gemm-shape"):
            session.create_gemm(*(entry.get_argument(i) for i in range(3)))

    def test_reduce_clear_false_has_explicit_destination_merge(self):
        session = HELPERS.create_session()
        _, entry = session.new_kernel("reduce", [session.memref_type([4], "fp32")])
        src = session.alloc_buffer([4, 4], "fp32")
        dst = session.alloc_buffer([4], "fp32")
        session.builder.create_fill_op(src, 2.0, "fp32")
        session.builder.create_fill_op(dst, -0.0, "fp32")
        session.create_reduce(src, dst, "max", 1, clear=False,
                              legacy_semantics="tensor_v1")
        session.builder.create_copy_op(dst, entry.get_argument(0), [], [])
        output = self.normalize(session)
        self.assertIn("frisk.reduce_tensor", output)
        self.assertIn("arith.maximumf", output)
        self.assertIn("frisk.tile_store", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    BUILD_DIR = args.build_dir.resolve()
    IR, HELPERS = load_build(BUILD_DIR)
    unittest.main(argv=[sys.argv[0]])
