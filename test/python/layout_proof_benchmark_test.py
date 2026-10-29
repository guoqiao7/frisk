#!/usr/bin/env python3
"""Exercise the real proof microbenchmark; timings are never pass/fail gates."""

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--benchmark", type=Path, required=True)
ARGS, UNITTEST_ARGS = PARSER.parse_known_args()
BENCHMARK = ARGS.benchmark.resolve()
PROOF_CMAKE = Path(__file__).resolve().parents[2] / "benchmark/layout/proof/CMakeLists.txt"


@unittest.skipUnless(shutil.which("ninja"), "Ninja build tool unavailable")
class BuildTypeMetadataTest(unittest.TestCase):
    def generated_metadata(self, generator, *options):
        # Exercise exactly the benchmark compile-definition expression without
        # configuring LLVM or rebuilding production in additional build modes.
        definition = re.search(r'FRISK_PROOF_BUILD_TYPE="([^"\n]+)"',
                               PROOF_CMAKE.read_text(encoding="utf-8"))
        self.assertIsNotNone(definition)
        with tempfile.TemporaryDirectory(prefix="frisk-proof-build-type-") as tmp:
            source = Path(tmp)
            build = source / "build"
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\n"
                "project(BuildTypeMetadata NONE)\n"
                'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/type-$<CONFIG>.txt" '
                f'CONTENT "{definition.group(1)}")\n', encoding="utf-8")
            proc = subprocess.run(["cmake", "-S", str(source), "-B", str(build),
                                   "-G", generator, *options], text=True,
                                  capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return {p.name: p.read_text(encoding="utf-8")
                    for p in build.glob("type-*.txt")}

    def test_single_config_unspecified_and_release(self):
        self.assertEqual(self.generated_metadata("Ninja"), {"type-.txt": "unspecified"})
        self.assertEqual(self.generated_metadata("Ninja", "-DCMAKE_BUILD_TYPE=Release"),
                         {"type-Release.txt": "Release"})

    def test_multi_config_uses_active_configuration(self):
        help_result = subprocess.run(["cmake", "--help"], text=True,
                                     capture_output=True, check=False)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        if "Ninja Multi-Config" not in help_result.stdout:
            self.skipTest("Ninja Multi-Config generator unavailable")
        self.assertEqual(self.generated_metadata("Ninja Multi-Config",
                         "-DCMAKE_CONFIGURATION_TYPES=Debug;Release"),
                         {"type-Debug.txt": "Debug", "type-Release.txt": "Release"})


class LayoutProofBenchmarkTest(unittest.TestCase):
    def invoke(self, *args):
        self.assertTrue(BENCHMARK.is_file(), f"benchmark executable missing: {BENCHMARK}")
        return subprocess.run([str(BENCHMARK), *args], text=True,
                              capture_output=True, check=False)

    @classmethod
    def setUpClass(cls):
        cls.runs = []
        if BENCHMARK.is_file():
            for _ in range(2):
                proc = subprocess.run([str(BENCHMARK), "--samples", "2",
                                       "--iterations", "1"], text=True,
                                      capture_output=True, check=False)
                if proc.returncode:
                    raise AssertionError(proc.stderr)
                cls.runs.append(json.loads(proc.stdout))

    def result(self):
        self.assertTrue(self.runs, f"benchmark executable missing: {BENCHMARK}")
        return self.runs[0]

    def test_fixed_cases_states_and_capability_distinction(self):
        data = self.result()
        self.assertEqual((data["schema_version"], data["kind"], data["status"]),
                         (1, "layout-proof", "success"))
        expected = {}
        for kind in ("ownership", "broadcast"):
            for size, points in (("small", 128), ("boundary", 65536),
                                 ("overbudget", 131072)):
                expected[f"{kind}_{size}"] = (kind, points, "Proven",
                    "Unknown" if size == "overbudget" else "Proven")
        expected["broadcast_same_thread_unavailable"] = (
            "broadcast", 128, "Disproven", "Disproven")
        self.assertEqual([c["id"] for c in data["cases"]], list(expected))
        for case in data["cases"]:
            kind, points, symbolic, oracle = expected[case["id"]]
            self.assertEqual((case["proof_kind"], case["hardware_points"]), (kind, points))
            self.assertEqual(case["equivalent_work"], symbolic == oracle)
            self.assertNotIn("speedup", case)
            for method, state in (("symbolic", symbolic), ("enumeration", oracle)):
                result = case[method]
                self.assertEqual(result["status"], state)
                self.assertEqual(result["expected_status"], state)
                self.assertTrue(result["reason"])
            if symbolic == "Disproven":
                self.assertEqual(case["symbolic"]["counterexample"], [1, 0])
                self.assertEqual(case["enumeration"]["counterexample"], [1, 0])
            if oracle == "Unknown":
                self.assertIn("65536", case["enumeration"]["reason"])

    def test_raw_samples_iterations_units_and_sink(self):
        data = self.result()
        self.assertEqual(data["options"], {"samples": 2, "iterations_per_sample": 1,
            "warmup_iterations": 1, "enumeration_point_limit": 65536})
        self.assertIn("construction excluded", data["measurement_scope"])
        for case in data["cases"]:
            for method in ("symbolic", "enumeration"):
                result = case[method]
                self.assertEqual(result["time_unit"], "ns")
                self.assertEqual(result["sample_count"], 2)
                self.assertEqual(result["iterations_per_sample"], 1)
                self.assertEqual(result["measured_iterations"], 2)
                self.assertEqual(result["sink"], result["expected_sink"])
                self.assertGreater(result["sink"], 0)
                self.assertTrue(result["sink_verified"])
                for key in ("samples_ns", "samples_ns_per_iteration"):
                    self.assertEqual(len(result[key]), 2)
                    self.assertTrue(all(isinstance(v, (int, float)) and
                        math.isfinite(v) and v >= 0 for v in result[key]))
                self.assertEqual(result["samples_ns"], result["samples_ns_per_iteration"])

    def test_provenance_and_nontiming_repeatability(self):
        data = self.result()
        environment = data["environment"]
        self.assertEqual(environment["binary_sha256"],
                         hashlib.sha256(BENCHMARK.read_bytes()).hexdigest())
        self.assertEqual(Path(environment["binary_path"]), BENCHMARK)
        for key in ("compiler", "build_type", "cpu"):
            self.assertTrue(environment[key])
        for key in ("source_digest", "config_digest"):
            self.assertRegex(environment[key], r"^[a-f0-9]{64}$")
        stripped = copy.deepcopy(self.runs)
        for run in stripped:
            for case in run["cases"]:
                for method in ("symbolic", "enumeration"):
                    case[method].pop("samples_ns")
                    case[method].pop("samples_ns_per_iteration")
        self.assertEqual(stripped[0], stripped[1])

    def test_nonunit_batches(self):
        proc = self.invoke("--samples", "1", "--iterations", "2")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        for case in json.loads(proc.stdout)["cases"]:
            for method in ("symbolic", "enumeration"):
                result = case[method]
                self.assertEqual(result["iterations_per_sample"], 2)
                self.assertEqual(result["measured_iterations"], 2)
                self.assertEqual(result["samples_ns"][0] / 2,
                                 result["samples_ns_per_iteration"][0])
                self.assertEqual(result["sink"], result["expected_sink"])

    def test_invalid_input_rejected_without_json(self):
        for args in (("--samples", "0"), ("--iterations", "0"),
                     ("--samples", "-1"), ("--iterations", "nan"),
                     ("--samples", "101"), ("--iterations", "1001"),
                     ("--samples",), ("--unknown",),
                     ("--samples", "1", "--samples", "2")):
            with self.subTest(args=args):
                proc = self.invoke(*args)
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(proc.stdout, "")
                self.assertIn("usage:", proc.stderr)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *UNITTEST_ARGS])
