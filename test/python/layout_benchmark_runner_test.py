#!/usr/bin/env python3
"""Tests for the schema-locked Task 29 benchmark runner."""

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmark/layout/run_layout_benchmarks.py"
SHA = "a" * 64
SEEDS = [1, 7, 42, 20260816]
CASE_SPECS = {
    "identity_f32": 32,
    "shuffle32_f32": 32,
    "shared64_f32": 64,
}


def conversion_result(inferred=103.0, reference=100.0, layout=10.0, full=100.0):
    environment = {
        "gpu_uuid": "GPU-test", "gpu_name": "H100", "compute_capability": "9.0",
        "driver": "test", "toolkit": "test", "clock_policy": "unmodified",
        "compile_flags": ["-O3", "-arch=sm_90"], "device": 1,
        "background_load": "idle",
    }
    cases = []
    for case_id, threads in CASE_SPECS.items():
        cases.append({
            "id": case_id, "status": "success", "threads": threads, "dtype": "f32",
            "correctness": {"status": "pass", "seeds": SEEDS,
                            "boundary_values": True, "replication": True},
            "sanitizers": {"memcheck": "pass", "racecheck": "pass", "synccheck": "pass"},
            "warmup": 100, "iterations": 1000, "measurement_order": "alternating",
            "reference": {"origin": "independent-coordinate-reference",
                          "samples_ns": [reference] * 1000, "source_sha256": SHA},
            "inferred": {"origin": "frisk-conversion-adapter", "samples_ns": [inferred] * 1000},
            "compilation": {"cache": "disabled", "samples": [
                {"layout_pipeline_ns": layout, "full_compile_ns": full}]},
            "provenance": {
                "frisk_input_sha256": SHA, "adapter_ir_sha256": SHA,
                "lowered_ir_sha256": SHA, "ptx_sha256": SHA, "cubin_sha256": SHA,
                "source_digest": SHA, "commands": ["compiler --input case.mlir"],
                "compile_flags": ["-O3", "-arch=sm_90"],
            },
            "artifact_checks": {"unrealized_casts": 0, "unresolved_layouts": 0,
                                "placeholder_calls": 0},
        })
    return {"schema_version": 1, "kind": "conversion", "status": "success",
            "environment": environment, "manifest": list(CASE_SPECS), "cases": cases}


def baseline_for(result, review_status="accepted"):
    return {
        "schema_version": 1, "kind": "reference-baseline", "review_status": review_status,
        "environment": copy.deepcopy(result["environment"]),
        "manifest": copy.deepcopy(result["manifest"]),
        "cases": [{"id": c["id"], "origin": c["reference"]["origin"],
                   "samples_ns": copy.deepcopy(c["reference"]["samples_ns"]),
                   "source_sha256": c["reference"]["source_sha256"],
                   "provenance": copy.deepcopy(c["provenance"])} for c in result["cases"]],
    }


def solver_result():
    cases = []
    for operations in (10, 50, 100, 250):
        combinations = 2 if operations == 10 else 512
        mode = "exact" if combinations <= 256 else "beam"
        cases.append({
            "id": f"connected_{operations}", "operations": operations,
            "status": "success", "warmup": 5, "iterations": 30,
            "solution_hash": SHA, "deterministic": True,
            "samples": {k: [1] * 30 for k in
                        ("collection_ns", "preparation_ns", "solve_ns", "layout_pipeline_ns")},
            "statistics": {
                "variables": 1, "constraints": 1, "components": 1,
                "preparedCandidates": 2, "prunedCandidates": 0,
                "peakVariableDomain": 2, "peakComponentCombinations": combinations,
                "componentProductOverflow": False, "peakFrontierStates": 2,
                "expandedStates": 2, "conversionUseEdges": 0, "uniqueConversions": 0,
                "rematerializationUseEdges": 0, "uniqueRematerializations": 0,
                "conversionBytes": 0, "estimatedBankConflicts": 0,
                "estimatedMemoryTransactions": 0,
            },
            "components": [{"root": "v0", "combinations": combinations,
                            "productOverflow": False, "domains": [combinations], "mode": mode,
                            "beamWidth": 32, "expanded": 2,
                            "peakFrontier": 2, "mrvProbes": 1,
                            "expandedNodes": 2, "exhaustive": mode == "exact"}],
        })
    return {"schema_version": 1, "kind": "solver", "status": "success",
            "environment": {"cpu": "test", "build_type": "Release",
                            "compiler": "clang", "source_digest": SHA},
            "options": {"exact_combination_limit": 256, "beam_width": 32,
                        "max_expanded_states": 65536}, "cases": cases}


class RunnerTest(unittest.TestCase):
    def run_runner(self, *extra, conversion=None, solver=None, baseline=None):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            args = [sys.executable, str(RUNNER)]
            for name, value in (("solver", solver), ("conversion", conversion),
                                ("baseline", baseline)):
                if value is not None:
                    path = tmp / f"{name}.json"
                    path.write_text(json.dumps(value), encoding="utf-8")
                    args += [f"--{name}-json" if name != "baseline" else "--baseline", str(path)]
            output = tmp / "result.json"
            args += list(extra) + ["--output", str(output)]
            proc = subprocess.run(args, text=True, capture_output=True, check=False)
            parsed = json.loads(output.read_text()) if output.exists() else None
            return proc, parsed

    def test_cpu_only_check_is_partial_success(self):
        proc, report = self.run_runner(solver=solver_result(), *["--cpu-only", "--check"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["gates"]["solver"]["status"], "PASS")
        self.assertEqual(report["gates"]["overall"]["status"], "PARTIAL_PASS")
        self.assertEqual(report["gates"]["runtime"]["status"], "NOT_EVALUATED")

    def test_exact_thresholds_pass_per_case(self):
        result = conversion_result(inferred=103, reference=100, layout=10, full=100)
        proc, report = self.run_runner(conversion=result, baseline=baseline_for(result), *["--check"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["gates"]["overall"]["status"], "PASS")
        for case in report["conversions"]["cases"]:
            self.assertAlmostEqual(case["runtime_ratio"], 1.03)
            self.assertTrue(case["runtime_ok"])
            self.assertAlmostEqual(case["compile_max_ratio"], .10)
            self.assertTrue(case["compile_ok"])
            self.assertEqual(case["inferred_p95_ns"], 103)
        self.assertIn("raw", report["conversions"])
        self.assertEqual(len(report["conversions"]["raw"]["cases"][0]
                             ["inferred"]["samples_ns"]), 1000)

    def test_above_runtime_threshold_fails_only_affected_case(self):
        result = conversion_result(inferred=100)
        result["cases"][1]["inferred"]["samples_ns"] = [103.01] * 1000
        proc, report = self.run_runner(conversion=result, baseline=baseline_for(result), *["--check"])
        self.assertNotEqual(proc.returncode, 0)
        cases = {c["id"]: c for c in report["conversions"]["cases"]}
        self.assertFalse(cases["shuffle32_f32"]["runtime_ok"])
        self.assertTrue(cases["identity_f32"]["runtime_ok"])

    def test_each_compile_sample_is_gated_not_just_median(self):
        result = conversion_result(inferred=100)
        result["cases"][0]["compilation"]["samples"] = [
            {"layout_pipeline_ns": 1, "full_compile_ns": 100},
            {"layout_pipeline_ns": 11, "full_compile_ns": 100},
            {"layout_pipeline_ns": 1, "full_compile_ns": 100},
        ]
        proc, report = self.run_runner(conversion=result, baseline=baseline_for(result), *["--check"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(report["conversions"]["cases"][0]["compile_ok"])

    def test_pending_baseline_is_not_accepted(self):
        result = conversion_result()
        proc, report = self.run_runner(conversion=result,
                                       baseline=baseline_for(result, "pending"), *["--check"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(report["gates"]["runtime"]["status"], "INCOMPLETE")

    def test_environment_mismatch_is_not_comparable(self):
        result = conversion_result()
        baseline = baseline_for(result)
        baseline["environment"]["gpu_uuid"] = "GPU-other"
        proc, report = self.run_runner(conversion=result, baseline=baseline, *["--check"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(report["gates"]["runtime"]["status"], "NOT_COMPARABLE")

    def test_reference_source_mismatch_is_not_comparable(self):
        result = conversion_result()
        baseline = baseline_for(result)
        baseline["cases"][0]["source_sha256"] = "b" * 64
        proc, report = self.run_runner("--check", conversion=result, baseline=baseline)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(report["gates"]["runtime"]["status"], "NOT_COMPARABLE")

    def test_device_ordinal_and_background_load_do_not_define_environment_identity(self):
        result = conversion_result()
        baseline = baseline_for(result)
        baseline["environment"]["device"] = 0
        baseline["environment"]["background_load"] = "observed-other-work"
        proc, report = self.run_runner("--check", conversion=result, baseline=baseline)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(report["gates"]["runtime"]["status"], "PASS")

    def test_missing_baseline_is_incomplete(self):
        proc, report = self.run_runner(conversion=conversion_result(), *["--check"])
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(report["gates"]["runtime"]["status"], "INCOMPLETE")

    def test_invalid_samples_fail_closed(self):
        for invalid in ([], [0] * 1000, [-1] * 1000, [math.nan] * 1000,
                        [math.inf] * 1000):
            with self.subTest(invalid=invalid[:1]):
                result = conversion_result()
                result["cases"][0]["inferred"]["samples_ns"] = invalid
                proc, _ = self.run_runner(conversion=result, baseline=baseline_for(result), *["--check"])
                self.assertNotEqual(proc.returncode, 0)

    def test_duplicate_or_missing_cases_fail(self):
        for mutate in (lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
                       lambda r: r["cases"].pop()):
            result = conversion_result()
            mutate(result)
            proc, _ = self.run_runner(conversion=result, baseline=baseline_for(conversion_result()),
                                      *["--check"])
            self.assertNotEqual(proc.returncode, 0)

    def test_source_confusion_and_bad_artifacts_fail(self):
        mutations = [
            lambda c: c["reference"].update(origin="frisk-conversion-adapter"),
            lambda c: c["inferred"].update(origin="independent-coordinate-reference"),
            lambda c: c["artifact_checks"].update(unrealized_casts=1),
            lambda c: c["provenance"].update(ptx_sha256="not-a-sha"),
            lambda c: c["provenance"].update(commands=[]),
            lambda c: c["provenance"].update(compile_flags=["-O0"]),
        ]
        for mutation in mutations:
            result = conversion_result()
            mutation(result["cases"][0])
            proc, _ = self.run_runner(conversion=result, baseline=baseline_for(conversion_result()),
                                      *["--check"])
            self.assertNotEqual(proc.returncode, 0)

    def test_baseline_provenance_flags_must_match_baseline_environment(self):
        result = conversion_result()
        baseline = baseline_for(result)
        baseline["cases"][0]["provenance"]["compile_flags"] = ["-O0"]
        proc, report = self.run_runner("--check", conversion=result, baseline=baseline)
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(report["gates"]["schema"]["status"], "FAIL")

    def test_record_reference_writes_pending_reference_only_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            conversion = tmp / "conversion.json"
            baseline = tmp / "baseline.json"
            conversion.write_text(json.dumps(conversion_result()), encoding="utf-8")
            proc = subprocess.run([sys.executable, str(RUNNER), "--conversion-json", str(conversion),
                                   "--record-reference", "--output-baseline", str(baseline)],
                                  text=True, capture_output=True, check=False)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            data = json.loads(baseline.read_text())
            self.assertEqual(data["review_status"], "pending")
            self.assertNotIn("inferred", json.dumps(data))

    def test_malformed_tool_output_fails_and_tool_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            marker = tmp / "runs"
            tool = tmp / "tool.py"
            tool.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                f"p = Path({str(marker)!r})\n"
                "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\n"
                "print('not json')\n", encoding="utf-8")
            tool.chmod(0o755)
            output = tmp / "result.json"
            proc = subprocess.run([sys.executable, str(RUNNER), "--solver", str(tool),
                                   "--cpu-only", "--check", "--output", str(output)],
                                  text=True, capture_output=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(marker.read_text(), "x")
            self.assertEqual(json.loads(output.read_text())["gates"]["schema"]["status"], "FAIL")

    def test_solver_rejects_wrong_mode_and_expansion_budget(self):
        for mutation in (
            lambda s: s["cases"][0]["components"][0].update(mode="beam"),
            lambda s: s["cases"][0]["components"][0].update(expanded=65537),
            lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])),
            lambda s: s["cases"][0]["components"][0].update(root=""),
            lambda s: s["cases"][0]["components"][0].update(combinations=1),
        ):
            solver = solver_result()
            mutation(solver)
            proc, _ = self.run_runner("--cpu-only", "--check", solver=solver)
            self.assertNotEqual(proc.returncode, 0)

    def test_solver_uint64_product_overflow_contract(self):
        solver = solver_result()
        component = solver["cases"][0]["components"][0]
        component.update(domains=[1 << 63, 3], combinations=None,
                         productOverflow=True, mode="beam", exhaustive=False)
        proc, report = self.run_runner("--cpu-only", "--check", solver=solver)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("raw", report["solver"])
        self.assertIn("collection_p95_ns", report["solver"]["cases"][0])

    def test_malformed_shapes_produce_structured_failure(self):
        mutations = [
            lambda s: s["cases"][0].update(id=[]),
            lambda s: s["cases"][0]["components"].__setitem__(0, []),
        ]
        for mutation in mutations:
            solver = solver_result()
            mutation(solver)
            proc, report = self.run_runner("--cpu-only", "--check", solver=solver)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(report["gates"]["schema"]["status"], "FAIL")

    def test_record_reference_never_overwrites_existing_or_input_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            result = conversion_result()
            conversion = tmp / "conversion.json"
            baseline = tmp / "accepted.json"
            conversion.write_text(json.dumps(result), encoding="utf-8")
            accepted = baseline_for(result)
            accepted["environment"]["gpu_uuid"] = "GPU-other"
            original = json.dumps(accepted, sort_keys=True)
            baseline.write_text(original, encoding="utf-8")
            proc = subprocess.run([
                sys.executable, str(RUNNER), "--conversion-json", str(conversion),
                "--baseline", str(baseline), "--check", "--record-reference",
                "--output-baseline", str(baseline)], text=True, capture_output=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(baseline.read_text(), original)

    def test_record_reference_output_cannot_alias_aggregate_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            conversion = tmp / "conversion.json"
            output = tmp / "same.json"
            conversion.write_text(json.dumps(conversion_result()), encoding="utf-8")
            proc = subprocess.run([
                sys.executable, str(RUNNER), "--conversion-json", str(conversion),
                "--record-reference", "--output-baseline", str(output),
                "--output", str(output)], text=True, capture_output=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(output.exists())
            self.assertIn("alias", proc.stderr)

    def test_aggregate_output_never_overwrites_input_baseline(self):
        for recording in (False, True):
            with self.subTest(recording=recording), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                conversion = tmp / "conversion.json"
                baseline = tmp / "accepted.json"
                result = conversion_result()
                conversion.write_text(json.dumps(result), encoding="utf-8")
                original = json.dumps(baseline_for(result), sort_keys=True)
                baseline.write_text(original, encoding="utf-8")
                args = [sys.executable, str(RUNNER), "--conversion-json", str(conversion),
                        "--baseline", str(baseline), "--output", str(baseline)]
                if recording:
                    args += ["--record-reference", "--output-baseline", str(baseline)]
                proc = subprocess.run(args, text=True, capture_output=True, check=False)
                self.assertNotEqual(proc.returncode, 0)
                self.assertEqual(baseline.read_text(), original)
                self.assertIn("alias", proc.stderr)

    def test_aggregate_output_hardlink_alias_preserves_input_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            conversion = tmp / "conversion.json"
            baseline = tmp / "accepted.json"
            output_alias = tmp / "aggregate.json"
            result = conversion_result()
            conversion.write_text(json.dumps(result), encoding="utf-8")
            original = json.dumps(baseline_for(result), sort_keys=True)
            baseline.write_text(original, encoding="utf-8")
            output_alias.hardlink_to(baseline)
            proc = subprocess.run([
                sys.executable, str(RUNNER), "--conversion-json", str(conversion),
                "--baseline", str(baseline), "--output", str(output_alias)],
                text=True, capture_output=True, check=False)
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(baseline.read_text(), original)
            self.assertIn("alias", proc.stderr)


if __name__ == "__main__":
    unittest.main()
