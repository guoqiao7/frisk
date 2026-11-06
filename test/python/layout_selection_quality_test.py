"""Selection-quality runner contract tests (standard library, real CLI smoke)."""
import copy
import contextlib
import importlib.util
import json
import os
import hashlib
import io
import shutil
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "benchmark/layout/selection"
sys.path.insert(0, str(SCRIPTS))
try:
    import run_selection_quality as runner
    import selection_report as report
except ImportError:
    runner = report = None


def oracle(geometries, value=1, status="complete"):
    return {"status": status, "exhausted": status == "complete", "fatal": False,
            "geometry_set_kind": "all-optimal", "optimal_geometries": geometries,
            "best_cost": {"cost": [value] + [0] * 8, "saturated": False}}


def core_evidence():
    rows = []
    for m in runner.experiment_manifest("core"):
        status = "local-analysis-only" if m["phase"] == "local" else "success"
        r = {"status": status, "phase": m["phase"], "oracle": oracle(["1,2"]),
             "spec": {"consumer": m["consumer"], "mode": m["mode"]},
             "source_digest": "source", "config_digest": "config", "binary_sha256": "binary",
             "input_sha256": m["mode"], "solver_options": runner.SOLVER_OPTIONS,
             "output_abi_verified": m["phase"] != "local", "replay_stable": m["phase"] != "local",
             "native_snapshot_projection": {"sha256": m["mode"], "attempt": 1,
                                            "scope": "actual-native-final-attempt"},
             "oracle_comparison": "search-gap"}
        rows.append({"manifest": m, "status": status, "result": r})
    return rows


class ContractTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "selection runner/report implementation is missing")

    def test_manifest_keeps_64_native_and_36_consumer_graphs(self):
        native = runner.native_manifest()
        self.assertEqual(len(native), 64)
        self.assertEqual(len({r["id"] for r in native}), 64)
        consumers = runner.consumer_manifest()
        self.assertEqual(len(consumers), 36)
        self.assertEqual(len({r["id"] for r in consumers}), 36)
        self.assertEqual(len(runner.local_manifest()), 6)
        direct = runner.direct_manifest()
        self.assertEqual(len(direct), 16)
        for row in consumers:
            self.assertNotIn("restrict_grid", row)
            if row["consumer"].startswith("fixed"):
                self.assertIn(row["consumer_grid"], ["1,2", "2,1"])
        self.assertEqual([r["policy"] for r in native[:4]],
                         ["Auto", "Square", "FullRow", "FullCol"])

    def test_grid_candidates_are_input_based_not_winner_based(self):
        self.assertEqual(runner.geometry_candidates("C6", "ss"),
                         ["1,8", "2,4", "4,2", "8,1"])
        self.assertEqual(runner.geometry_candidates("C6", "rs"),
                         ["1,8", "2,4", "4,2", "8,1"])
        self.assertEqual(runner.geometry_candidates("C7", "ss"), ["1,4", "2,2"])
        row = runner.direct_manifest()[0]
        self.assertEqual(runner.branch_rows(row, ["1,1"])[0]["restrict_grid"], "1,1")

    def test_exit_json_contradiction_is_infrastructure_error(self):
        for code, record in [(1, {"status": "success", "fatal": False}),
                             (0, {"status": "success", "fatal": True}),
                             (0, {"status": "failed", "fatal": False}),
                             (1, {"status": "failed", "fatal": True}),
                             (2, {"status": "failed", "fatal": False})]:
            with self.assertRaises(runner.ValidationError):
                runner.validate_exit(record, code)
        runner.validate_exit({"status": "failed", "fatal": True}, 2)
        runner.validate_exit({"status": "failed", "fatal": False}, 1)
        runner.validate_exit({"status": "local-analysis-only", "fatal": False}, 0)

    def test_cost_shape_sum_and_full_vector_first_difference(self):
        c = {"cost": [1, 2] + [0] * 7, "saturated": False}
        runner.reconcile(c, [{"cost": c}])
        for invalid in [{"cost": [1], "saturated": False},
                        {"cost": [True] * 9, "saturated": False},
                        {"cost": [-1] * 9, "saturated": False}]:
            with self.assertRaises(runner.ValidationError):
                runner.cost_vector(invalid)
        with self.assertRaises(runner.ValidationError):
            runner.reconcile(c, [])
        diff = report.cost_difference([1, 2] + [0] * 7, [1, 3] + [0] * 7)
        self.assertEqual(diff["index"], 1)
        self.assertEqual(diff["left"], 2)
        self.assertIsNone(report.cost_difference([0] * 9, [0] * 9))

    def test_classification_strict_reversal_tie_same_layout_and_no_improvement(self):
        local = {"oracle": oracle(["1,2"])}
        joint = {"oracle": oracle(["2,1"], 1), "oracle_comparison": "equal-optimum",
                 "assignment": [1], "conversions": [], "selected_geometries": ["2,1"]}
        branches = {"1,2": {"oracle": oracle(["1,2"], 2), "assignment": [2]},
                    "2,1": {"oracle": oracle(["2,1"], 1), "selected_geometries": ["2,1"]}}
        self.assertEqual(report.classify(local, joint, branches)["classification"],
                         "strict-geometry-reversal")
        local["oracle"] = oracle(["1,2", "2,1"])
        self.assertEqual(report.classify(local, joint, branches)["classification"],
                         "local-tie-resolved")
        local["oracle"] = oracle(["2,1"])
        branches["2,1"]["assignment"] = [2]
        self.assertEqual(report.classify(local, joint, branches)["classification"],
                         "same-geometry-layout-conversion-change")
        branches["2,1"]["assignment"] = [1]
        branches["2,1"]["conversions"] = []
        self.assertEqual(report.classify(local, joint, branches)["classification"],
                         "no-observed-improvement")

    def test_incomplete_gap_missing_geometry_never_reversal(self):
        local = {"oracle": oracle(["1,2"])}
        joint = {"oracle": oracle(["2,1"])}
        branches = {"1,2": {"oracle": oracle(["1,2"], 2)}}
        for status in ["oracle-incomplete", "proof-incomplete"]:
            joint["oracle"]["status"] = status
            self.assertEqual(report.classify(local, joint, branches)["classification"],
                             "heuristic-only")
        joint["oracle"]["status"] = "complete"
        joint["oracle_comparison"] = "search-gap"
        self.assertEqual(report.classify(local, joint, branches)["classification"], "heuristic-only")
        joint.pop("oracle_comparison")
        self.assertEqual(report.classify(local, joint, {})["classification"], "not-comparable")

    def test_artifact_hash_tamper_missing_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "input.mlir").write_text("input")
            hashes = {"input.mlir": runner.sha256(path / "input.mlir")}
            runner.verify_artifacts(path, hashes)
            (path / "input.mlir").write_text("tamper")
            with self.assertRaises(runner.ValidationError):
                runner.verify_artifacts(path, hashes)
            for name in ["missing", "../escape", "/tmp/escape"]:
                with self.assertRaises(runner.ValidationError):
                    runner.verify_artifacts(path, {name: "0" * 64})

    def test_timeout_preserves_process_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.execute([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
                                    Path(tmp), 0.1)
            self.assertEqual(result["status"], "process-timeout")
            self.assertTrue(result["timeout"])
            self.assertIn("started", (Path(tmp) / "stdout.txt").read_text())
            self.assertIn("returncode", result)
            self.assertEqual(result["parent_timeout_seconds"], 0.1)

    def test_native_comparison_requires_all_identities_and_final_snapshot(self):
        base = {"status": "success", "phase": "native", "source_digest": "a",
                "config_digest": "b", "binary_sha256": "binary", "input_sha256": "c", "spec": {"consumer": "store"},
                "native_snapshot_projection": {"sha256": "d", "attempt": 1},
                "solver_options": runner.SOLVER_OPTIONS}
        controlled = copy.deepcopy(base)
        controlled["phase"] = "controlled"
        self.assertEqual(report.native_comparison(base, controlled)["status"], "same-domain")
        for key in ["source_digest", "config_digest", "binary_sha256", "input_sha256", "spec", "solver_options"]:
            changed = copy.deepcopy(controlled)
            changed[key] = "different"
            self.assertEqual(report.native_comparison(base, changed)["status"], "not-comparable")
        controlled["spec"]["consumer"] = "fixed0"
        self.assertEqual(report.native_comparison(base, controlled)["status"], "not-comparable")

    def test_timing_manifest_serial_48_and_non_timing_stats(self):
        rows = runner.timing_manifest()
        self.assertEqual(len(rows), 48)
        self.assertEqual(sum(r["warmup"] for r in rows), 8)
        self.assertEqual(runner.without_timings({"x": [{"timings": {"a": 1}, "cost": 2}]}),
                         {"x": [{"cost": 2}]})
        command = runner.timing_command(Path("/opt/frisk-opt"), Path("input.mlir"), Path("stats.json"), Path("out.mlir"))
        self.assertTrue(any("stats-file=stats.json" in arg for arg in command))
        self.assertFalse(any("trace-file" in arg or "profile-file" in arg for arg in command))

    def test_full_manifest_branches_and_core_gate(self):
        self.assertTrue(hasattr(runner, "experiment_manifest"), "batch matrix missing")
        full = runner.experiment_manifest("correctness")
        self.assertEqual(len([r for r in full if r["phase"] == "native"]), 64)
        self.assertEqual(len([r for r in full if r["consumer"] in ("reduce0", "reduce1", "fixed0", "fixed1") and not r.get("parent")]), 36)
        self.assertEqual(len({r["id"] for r in full}), len(full))
        for parent in runner.direct_manifest() + runner.consumer_manifest():
            actual = [r["restrict_grid"] for r in full if r.get("parent") == parent["id"]]
            self.assertEqual(actual, runner.geometry_candidates(parent["case"], parent["mode"]))
        self.assertFalse(runner.core_gate([])["passed"])
        self.assertGreater(len(runner.experiment_manifest("core")), 8)

    def test_bounded_batch_retains_failed_rows_in_manifest_order(self):
        self.assertTrue(hasattr(runner, "run_batch"), "bounded batch execution missing")
        with tempfile.TemporaryDirectory() as tmp:
            manifest = runner.native_manifest()[:3]
            rows = runner.run_batch(Path(tmp) / "missing-executable", manifest, Path(tmp), 1, 2)
            self.assertEqual([r["manifest"]["id"] for r in rows], [r["id"] for r in manifest])
            self.assertEqual([r["status"] for r in rows], ["process-launch-failure"] * 3)
            data = {"rows": rows}
            text = report.render(data)
            self.assertIn("Rows: 3", text)
            self.assertIn("process-launch-failure", text)

    def test_new_output_and_dataset_checksum_validation(self):
        self.assertTrue(hasattr(runner, "claim_output"), "new output / dataset publication missing")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "new"
            runner.claim_output(directory)
            with self.assertRaises(runner.ValidationError):
                runner.claim_output(directory)
            data = {"schema_version": 1, "kind": "selection-quality-batch", "identity": {}, "rows": []}
            runner.publish_dataset(directory, data)
            with self.assertRaises(runner.ValidationError):
                runner.load_dataset(directory / "results.json")
            (directory / "results.json").write_text("{}")
            with self.assertRaises(runner.ValidationError):
                runner.load_dataset(directory / "results.json")

    def test_core_gate_does_not_hide_complete_oracle_search_gaps(self):
        # Coverage gate is not a promise that default search is optimal.
        rows = core_evidence()
        self.assertTrue(runner.core_gate(rows)["passed"])
        summary = report.summarize(rows)
        self.assertTrue(all(c["classification"] == "heuristic-only" for c in summary["q2_classifications"]))

    def test_review_i1_all_26_oracles_and_both_native_witnesses_required(self):
        rows = core_evidence()
        self.assertTrue(runner.core_gate(rows)["passed"])
        self.assertEqual(len(runner.core_gate(rows).get("required_oracle_rows", [])), 26)
        for target in ("C1-ss-local", "C1-rs-local", "C1-ss-store-grid-1,2", "C1-ss-fixed0-1,2-grid-2,1", "C1-ss-native-Auto"):
            with self.subTest(target=target):
                missing = [r for r in rows if r["manifest"]["id"] != target]
                self.assertFalse(runner.core_gate(missing)["passed"])
                incomplete = copy.deepcopy(rows)
                next(r for r in incomplete if r["manifest"]["id"] == target)["status"] = "failed"
                self.assertFalse(runner.core_gate(incomplete)["passed"])

    def test_review_i1_formal_acceptance_exit_and_report_are_visible(self):
        self.assertTrue(hasattr(runner, "dataset_exit_code"), "shared formal return-code gate missing")
        for mode in ("correctness", "core", "merged"):
            data = {"mode": mode, "component_modes": ["correctness", "timing"], "rows": [],
                    "core_gate": runner.core_gate([]), "integrity_errors": []}
            self.assertEqual(runner.dataset_exit_code(data), 1)
            text = report.render(data)
            self.assertIn("未验收", text)
            self.assertIn("C1-ss-local", text)
        for mode in ("smoke", "timing"):
            self.assertEqual(runner.dataset_exit_code(dict(data, mode=mode)), 0)
        data["integrity_errors"] = [{"id": "bad-domain", "kind": "frozen-domain-integrity"}]
        self.assertIn("bad-domain", report.render(data))
        data = {"mode": "correctness", "rows": [{"manifest": runner.native_manifest()[0], "status": "fatal-correctness"}],
                "core_gate": {"passed": True, "missing_or_incomplete": []}, "integrity_errors": []}
        self.assertIn("Stage status: 未验收", report.render(data))

    def test_review_i2_unchanged_ties_only_compare_actual_same_geometry(self):
        local = {"oracle": oracle(["1,2", "2,1"])}
        joint = {"oracle": oracle(["1,2", "2,1"]), "selected_geometries": ["1,2"],
                 "assignment": ["grid1-layout"], "conversions": [], "existing_conversions": []}
        branches = {"1,2": copy.deepcopy(joint), "2,1": dict(joint, selected_geometries=["2,1"], assignment=["grid2-layout"])}
        self.assertEqual(report.classify(local, joint, branches)["classification"], "no-observed-improvement")
        branches["1,2"]["conversions"] = [{"remat": "arith.addf"}]
        self.assertEqual(report.classify(local, joint, branches)["classification"], "same-geometry-layout-conversion-change")
        branches["1,2"]["selected_geometries"] = ["2,1"]
        self.assertNotEqual(report.classify(local, joint, branches)["classification"], "same-geometry-layout-conversion-change")

    def test_review_i3_identity_requires_complete_hashset_and_digest(self):
        with self.assertRaises(runner.ValidationError):
            runner.check_identity({})
        bench = ROOT / "build-m6-release/benchmark/layout/selection/frisk_layout_selection_bench"
        valid = runner.identity(bench, ROOT / "build-m6-release/bin/frisk-opt")
        runner.check_identity(valid)
        for change in ("identity_sha256", "files_sha256", "compiled_source_digest", "solver_options"):
            bad = copy.deepcopy(valid)
            if change == "files_sha256":
                bad[change].pop(next(iter(bad[change])))
                bad["identity_sha256"] = hashlib.sha256(json.dumps(bad[change], sort_keys=True).encode()).hexdigest()
            else:
                bad[change] = "wrong"
            with self.subTest(change=change), self.assertRaises(runner.ValidationError):
                runner.check_identity(bad)

    def test_review_i3_missing_row_location_has_typed_schema_failure(self):
        try:
            runner.verify_saved_row(Path("/tmp"), {"status": "success", "files_sha256": {}})
        except Exception as error:
            self.assertIsInstance(error, runner.ValidationError)
        else:
            self.fail("missing archived location accepted")

    def test_canonical_encoding_requires_unbound_mma_and_bound_consumer(self):
        self.assertTrue(hasattr(runner, "canonical_encoding"), "canonical encoding audit missing")
        text = '%r = "frisk.mma"(%a) : (tensor<1xf32>) -> tensor<1xf32>\n%c = "frisk.convert_layout"(%r) : (tensor<1xf32>) -> tensor<1xf32, #frisk.distributed<map = 1>>\n'
        identity = runner.canonical_encoding(text)
        self.assertEqual(identity["attribute"], "#frisk.distributed<map = 1>")
        with self.assertRaises(runner.ValidationError):
            runner.canonical_encoding(text.replace("(tensor<1xf32>) -> tensor<1xf32,", "(tensor<1xf32, #frisk.distributed<map = 1>>) -> tensor<1xf32,"))

    def test_rss_is_per_child_and_timeout_reaps_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.execute([sys.executable, "-c", "print('rss')"], Path(tmp), 5, measure_rss=True)
            self.assertEqual(result["returncode"], 0)
            self.assertGreater(result["peak_rss_kib"], 0)
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.execute([sys.executable, "-c", "import time; time.sleep(10)"], Path(tmp), 0.05, measure_rss=True)
            self.assertTrue(result["timeout"])
            self.assertEqual(result["returncode"], -9)

    def test_report_has_separate_coverage_witness_and_timing_sections(self):
        text = report.render({"rows": []})
        for section in ("Native Q1 coverage", "Oracle witness validation", "Whole-CLI timing", "Decision example"):
            self.assertIn(section, text)


class RealExecutableTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, "selection runner/report implementation is missing")
        self.bench = Path(os.environ.get("FRISK_SELECTION_BENCH", ROOT / "build-m6-release/benchmark/layout/selection/frisk_layout_selection_bench"))
        self.assertTrue(self.bench.is_file(), f"real executable required: {self.bench}")

    def test_real_native_controlled_local_and_artifact_validation(self):
        with tempfile.TemporaryDirectory(prefix="frisk-selection-python-real-") as tmp:
            out = Path(tmp)
            native = runner.run_row(self.bench, runner.native_manifest()[0], out, 180)
            self.assertEqual(native["status"], "success", native)
            direct = runner.direct_manifest()[0]
            controlled = runner.run_row(self.bench, direct, out, 180)
            self.assertEqual(controlled["status"], "success", controlled)
            self.assertEqual(controlled["result"]["oracle"]["status"], "complete")
            for mutator in (
                    lambda r: r.update(schema_version=99),
                    lambda r: r["oracle"].update(exhausted=False),
                    lambda r: r["oracle"].update(unknown_assignments=1),
                    lambda r: r["oracle"].update(incumbent_actual_verified=False),
                    lambda r: r["oracle"]["limits"].update(nodes=999999),
                    lambda r: r["spec"].update(threads=42),
                    lambda r: r["oracle"].update(incumbent_validation="unverified")):
                invalid = copy.deepcopy(controlled["result"])
                mutator(invalid)
                with self.assertRaises(runner.ValidationError):
                    runner.validate_record(invalid, out / controlled["artifact_dir"], direct, self.bench)
            invalid = copy.deepcopy(controlled["result"])
            invalid["cost"] = {"cost": [0] * 9, "saturated": False}
            invalid["cost_contributions"] = [{"cost": invalid["cost"]}]
            with self.assertRaisesRegex(runner.ValidationError, "default below complete oracle"):
                runner.validate_record(invalid, out / controlled["artifact_dir"], direct, self.bench)
            self.assertEqual(report.native_comparison(native["result"], controlled["result"])["status"], "same-domain")
            branch = runner.branch_rows(direct, ["1,1"])[0]
            restricted = runner.run_row(self.bench, branch, out, 180)
            self.assertEqual(restricted["status"], "success", restricted)
            runner.validate_branches(controlled["result"], [restricted["result"]], out)
            local = {"id": "C0-ss-local", "case": "C0", "mode": "ss", "phase": "local", "consumer": "local", "policy": "Auto", "oracle": True}
            analyzed = runner.run_row(self.bench, local, out, 180)
            self.assertEqual(analyzed["status"], "local-analysis-only", analyzed)
            self.assertFalse(analyzed["result"]["output_abi_verified"])
            self.assertIn("result.json", native["files_sha256"])
            data = {"schema_version": 1, "kind": "selection-quality-batch", "mode": "smoke", "jobs": 1,
                    "parent_timeout_seconds": 180, "oracle_timeout_milliseconds_distinct_from_parent": 120000,
                    "execution_root": str(out), "identity": runner.identity(self.bench, ROOT / "build-m6-release/bin/frisk-opt"),
                    "manifest": runner.experiment_manifest("smoke"), "preflight": [], "native_entry_probes": [],
                    "integrity_errors": [], "rows": [native, controlled, restricted, analyzed]}
            runner.publish_dataset(out, data)
            runner.load_dataset(out / "results.json")
            for change in ("wrapper", "drop", "identity", "hashes", "rowfields", "command", "gate", "integrity", "manifest", "preflight", "probes"):
                bad = copy.deepcopy(data)
                if change == "wrapper":
                    bad["rows"][0]["status"] = "failed"
                elif change == "drop":
                    bad["rows"].pop()
                elif change == "identity":
                    bad["identity"] = {}
                elif change == "hashes":
                    bad["rows"][0]["files_sha256"] = {}
                elif change == "rowfields":
                    bad["rows"][0].pop("artifact_dir")
                elif change == "command":
                    bad["rows"][0]["process"]["command"][0] = "/wrong-bench"
                elif change == "gate":
                    bad["core_gate"]["passed"] = True
                elif change == "integrity":
                    bad["integrity_errors"] = [{"id": "invented", "error": "stale"}]
                elif change == "manifest":
                    bad["manifest"] = list(reversed(bad["manifest"]))
                elif change == "preflight":
                    bad["preflight"] = [native]
                elif change == "probes":
                    bad["native_entry_probes"] = [native]
                runner.publish_dataset(out, bad)
                if change == "gate":
                    # Publisher correctly recomputes gates; forge it after publication.
                    published = json.loads((out / "results.json").read_text())
                    published["core_gate"]["passed"] = True
                    runner.json_write(out / "results.json", published)
                    files = json.loads((out / "dataset-files.json").read_text())
                    files["results.json"] = runner.sha256(out / "results.json")
                    runner.json_write(out / "dataset-files.json", files)
                with self.subTest(change=change), self.assertRaises(runner.ValidationError):
                    runner.load_dataset(out / "results.json")
            runner.publish_dataset(out, data)
            archive = out / "relocated"
            archive.mkdir()
            for name in ("results.json", "report.md", "dataset-files.json", "identity.json", "manifest.json"):
                if (out / name).exists():
                    shutil.copy2(out / name, archive / name)
            for saved in data["rows"]:
                for key in ("artifact_dir", "process_dir"):
                    shutil.move(out / saved[key], archive / saved[key])
            runner.load_dataset(archive / "results.json")
            for saved in data["rows"]:
                for key in ("artifact_dir", "process_dir"):
                    shutil.move(archive / saved[key], out / saved[key])
            # Corrupt only a restricted non-geometry variable, then reconcile
            # every local hash/JSON reference. Cross-branch validation must
            # still reject the previously empty integrity report.
            branch_dir = out / restricted["artifact_dir"]
            names = ("branch-domains.json", "oracle.json", "result.json")
            original_bytes = {name: (branch_dir / name).read_bytes() for name in names}
            domain = json.loads(original_bytes["branch-domains.json"])
            next(v for v in domain["variables"] if v["kind"] != 2)["name"] += "-corrupt"
            runner.json_write(branch_dir / "branch-domains.json", domain)
            digest = runner.sha256(branch_dir / "branch-domains.json")
            raw = json.loads(original_bytes["result.json"])
            raw["frozen_domain"]["restricted_sha256"] = digest
            raw["oracle"]["restricted_domain_sha256"] = digest
            raw["artifacts_sha256"]["branch-domains.json"] = digest
            runner.json_write(branch_dir / "oracle.json", raw["oracle"])
            raw["artifacts_sha256"]["oracle.json"] = runner.sha256(branch_dir / "oracle.json")
            runner.json_write(branch_dir / "result.json", raw)
            bad = copy.deepcopy(data)
            saved_branch = next(r for r in bad["rows"] if r["manifest"].get("parent"))
            saved_branch["result"] = raw
            for name in names:
                saved_branch["files_sha256"][name] = runner.sha256(branch_dir / name)
            runner.publish_dataset(out, bad)
            with self.assertRaisesRegex(runner.ValidationError, "integrity errors differ"):
                runner.load_dataset(out / "results.json")
            for name, content in original_bytes.items():
                (branch_dir / name).write_bytes(content)
            corrupted = copy.deepcopy(native)
            corrupted["result"]["cost"]["cost"][0] += 1
            runner.publish_dataset(out, dict(data, rows=[corrupted] + data["rows"][1:]))
            with self.assertRaisesRegex(runner.ValidationError, "aggregate.*raw"):
                runner.load_dataset(out / "results.json")
            (out / native["artifact_dir"] / "input.mlir").write_text("corrupt")
            with self.assertRaises(runner.ValidationError):
                runner.verify_saved_row(out, native)

    def test_real_fixed_consumer_emit_and_whole_cli_timing_sample(self):
        self.assertTrue(hasattr(runner, "canonical_encoding"), "canonical encoding audit missing")
        with tempfile.TemporaryDirectory(prefix="frisk-selection-python-timing-") as tmp:
            out = Path(tmp)
            m = dict(runner.consumer_manifest()[1], phase="emit-input", oracle=False, id="fixed-input")
            emitted = runner.run_row(self.bench, m, out, 180)
            self.assertEqual(emitted["status"], "input-emitted", emitted)
            encoding = runner.canonical_encoding((out / "fixed-input/input.mlir").read_text())
            self.assertIn("#frisk.distributed", encoding["attribute"])
            m = dict(runner.native_manifest()[0], phase="emit-input", id="timing-input")
            native_input = runner.run_row(self.bench, m, out, 180)
            self.assertEqual(native_input["status"], "input-emitted", native_input)
            opt = ROOT / "build-m6-release/bin/frisk-opt"
            a = runner.run_timing_sample(opt, out / "timing-input/input.mlir", out / "sample0", 180)
            b = runner.run_timing_sample(opt, out / "timing-input/input.mlir", out / "sample1", 180)
            self.assertEqual(a["status"], "success", a)
            self.assertEqual(b["status"], "success", b)
            self.assertEqual(a["output_sha256"], b["output_sha256"])
            self.assertEqual(a["non_timing_statistics_sha256"], b["non_timing_statistics_sha256"])
            self.assertGreater(a["process"]["peak_rss_kib"], 0)

    def test_review_real_formal_cli_preflight_timeout_remains_unaccepted(self):
        with tempfile.TemporaryDirectory(prefix="frisk-selection-review-failure-") as tmp:
            out = Path(tmp) / "core-negative"
            with contextlib.redirect_stdout(io.StringIO()):
                code = runner.main(["core", "--output-dir", str(out), "--bench", str(self.bench),
                                    "--timeout", "0.000001", "--jobs", "1"])
            self.assertEqual(code, 1)
            data = runner.load_dataset(out / "results.json")
            self.assertEqual(len(data["rows"]), 28)
            self.assertFalse(data["core_gate"]["passed"])
            self.assertTrue(all(r["status"] == "preflight-blocked" for r in data["rows"] + data["native_entry_probes"]))
            self.assertIn("未验收", (out / "report.md").read_text())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(["report", "--dataset", str(out / "results.json"),
                                              "--output-dir", str(Path(tmp) / "formal-negative-report")]), 1)


if __name__ == "__main__":
    unittest.main()
