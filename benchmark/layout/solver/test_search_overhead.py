"""Evidence-integrity and per-child measurement tests; CPU only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import run_search_overhead as bench


class SearchOverheadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="frisk-search-bench-test-")
        self.root = Path(self.temporary.name)
        self.stage = self.root / "stage"
        (self.root / "CMakeLists.txt").write_text("project(test)\n")
        self.cpu = min(os.sched_getaffinity(0))
        self.real_checked = bench.checked_run
        self.patches = [mock.patch.object(bench, "ROOT", self.root),
                        mock.patch.object(bench, "checked_run", lambda path: self.real_checked(path, self.stage)),
                        mock.patch.object(bench, "render_fixture", lambda spec: "module {}\n")]
        for patch in self.patches:
            patch.start()
        fake = ("#!" + sys.executable + "\n" + r"""
import json
from pathlib import Path
import sys
arguments=sys.argv[1:]
options=next(x.split('=',1)[1] for x in arguments if x.startswith('--frisk-layout-pipeline='))
flags=dict(item.split('=',1) for item in options.split())
failure='EXPECTED_COST_FAILURE' in Path(arguments[0]).read_text()
if not failure:Path(arguments[arguments.index('-o')+1]).write_text('module {}\n')
stats={'schema_version':1,'status':'committed','timings':{'layout_pipeline_ns':7},
       'attempts':[{'status':'solved','statistics':{'expandedStates':4},'timings':{'solve_ns':5}}]}
if failure:stats.update(status='failed',failure={'kind':'cost-failure','reason':'layout cost evaluation failed'})
Path(flags['stats-file']).write_text(json.dumps(stats))
if 'trace-file' in flags:
    Path(flags['trace-file']).write_text(json.dumps({'truncated':False,'costExplanationComplete':not failure,'finalCost':None if failure else {'cost':[3,4]},'events':[{'domain':['A','B']}]}))
if 'profile-file' in flags:
    Path(flags['profile-file']).write_text(json.dumps({'status':'committed','entries_kind':'insertions','metrics':{'count':1 if flags.get('memoize-costs')!='false' else 2}}))
if failure:sys.exit(1)
""")
        self.baseline = self.root / "before"
        self.candidate = self.root / "after"
        for path in (self.baseline, self.candidate):
            path.write_text(fake); path.chmod(0o755)

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def register(self, off=False):
        run = self.stage / "run"
        with mock.patch.object(bench, "specifications", return_value=[{"id":"fixture", "timed":True}]):
            protocol = bench.registration(run, self.baseline, self.candidate, self.cpu,
                                          pairs=2, profile_pairs=1,
                                          candidate_off_option="memoize-costs=false" if off else None)
        return run, protocol

    def test_semantic_gate_keeps_cost_domain_and_search_counters(self):
        left = {"case":"x", "ir_sha256":"a", "non_timing_stats":{"cost":[1,2], "expandedStates":9}}
        for changed in ({"cost":[1,3], "expandedStates":9}, {"cost":[1,2], "expandedStates":10}):
            with self.assertRaisesRegex(bench.InvalidEvidence, "semantic mismatch"):
                bench.same_semantics(left, dict(left, non_timing_stats=changed))
        raw = {"cost":[1,2], "timings":{"solve_ns":3}, "children":[{"timings":{"verify_ns":9},"domain":[1,2]}]}
        self.assertEqual(bench.non_timing(raw), {"cost":[1,2],"children":[{"domain":[1,2]}]})

    def test_real_wait4_resources_and_owned_timeout(self):
        result = bench.execute([sys.executable, "-c", "print('child-ok')"], self.root / "child")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual((self.root / "child/stdout.txt").read_text(), "child-ok\n")
        self.assertGreater(result["wall_ns"], 0)
        self.assertGreater(result["peak_rss_kib"], 0)
        self.assertGreaterEqual(result["user_ns"] + result["system_ns"], 0)
        timed = bench.execute([sys.executable, "-c", "import time;time.sleep(30)"], self.root / "timed", timeout=.02)
        self.assertTrue(timed["timeout"])
        self.assertLess(timed["returncode"], 0)
        self.assertEqual(bench.read(self.root / "timed/process.json"), timed)

    def test_peak_rss_is_per_child_not_cumulative_high_water(self):
        large = bench.execute([sys.executable, "-c", "data=bytearray(64*1024*1024);print(len(data))"], self.root / "large")
        small = bench.execute([sys.executable, "-c", "print('small')"], self.root / "small")
        self.assertEqual(large["returncode"], 0)
        self.assertEqual(small["returncode"], 0)
        self.assertGreater(large["peak_rss_kib"], small["peak_rss_kib"] + 32768)

    def test_reproducible_balanced_random_pairs(self):
        orders = bench.schedule(10, 23)
        self.assertEqual(orders, bench.schedule(10, 23))
        self.assertEqual(sum(x == ["baseline","candidate"] for x in orders), 5)
        self.assertNotEqual(orders, bench.schedule(10, 24))
        with self.assertRaises(bench.InvalidEvidence):
            bench.schedule(0, 1)

    def test_path_escape_symlink_and_reuse_rejected(self):
        with self.assertRaises(bench.InvalidEvidence):
            self.real_checked(self.root / "outside", self.stage)
        self.stage.mkdir()
        alias = self.stage / "alias"; alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(bench.InvalidEvidence):
            self.real_checked(alias / "child", self.stage)
        run, _ = self.register()
        with self.assertRaisesRegex(bench.InvalidEvidence, "already exists"):
            bench.registration(run, self.baseline, self.candidate, self.cpu)

    def test_registered_input_binary_and_driver_guard(self):
        run, protocol = self.register()
        original = Path(protocol["cases"][0]["input"]).read_bytes()
        Path(protocol["cases"][0]["input"]).write_text("changed")
        with self.assertRaisesRegex(bench.InvalidEvidence, "input changed"):
            bench.check_registration(run)
        Path(protocol["cases"][0]["input"]).write_bytes(original)
        self.baseline.write_text(self.baseline.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(bench.InvalidEvidence, "baseline binary changed"):
            bench.check_registration(run)

    def test_matrix_has_native_attention_multiconsumer_and_explicit_v2_guards(self):
        rows = bench.specifications(True, "/tmp/example-profile.json")
        self.assertEqual(len(rows), 16)
        self.assertEqual(sum(row["timed"] for row in rows), 13)
        self.assertEqual({row["id"] for row in rows if not row["timed"]}, {"attention-fixed-weak-v2-guard", "diamond-32x32-v2-guard", "diamond-32x16-v2-guard"})
        self.assertTrue(any(row.get("shape") == [128,32] for row in rows))

    def test_arithmetic_control_has_ten_dependent_ops_without_storage_access(self):
        # Call the real renderer; setUp substitutes it for process-only tests.
        renderer = self.patches[2].temp_original
        text = renderer({"control":"ten-dependent-addf-tensor4xf32"})
        self.assertEqual(text.count("arith.addf"), 10)
        self.assertIn("%v9 = arith.addf %v8, %v8", text)
        self.assertNotIn("tile_load", text)
        self.assertNotIn("tile_store", text)

    def test_pair_summary_rejects_invalid_measurements_and_keeps_pairing(self):
        pairs = [{"observations": {"baseline":{"process":{"wall_ns":a}}, "candidate":{"process":{"wall_ns":b}}}} for a,b in [(20,10),(40,20)]]
        result = bench.paired_summary(pairs)
        self.assertAlmostEqual(result["paired_geomean_speedup"], 2)
        self.assertEqual(result["paired_bootstrap_95ci"], [2,2])
        pairs[0]["observations"]["candidate"]["process"]["wall_ns"] = 0
        with self.assertRaisesRegex(bench.InvalidEvidence, "nonpositive"):
            bench.paired_summary(pairs)

    def test_full_fake_phases_and_raw_tamper_rejected(self):
        run, _ = self.register(off=True)
        bench.correctness(run, timeout=10)
        gate = bench.verify_phase(run, "correctness")
        self.assertEqual(len(gate["rows"][0]["observations"]), 6)
        bench.measure(run, timeout=10)
        bench.measure(run, profile=True, timeout=10)
        bench.measure(run, profile=True, timeout=10, ablation=True)
        main = bench.verify_phase(run, "timing")
        diagnostic = bench.verify_phase(run, "profile")
        self.assertFalse(main["rows"][0]["pairs"][0]["observations"]["baseline"]["profile"])
        self.assertTrue(diagnostic["rows"][0]["pairs"][0]["observations"]["baseline"]["profile"])
        result = bench.report(run)
        self.assertTrue(result["no_gpu_performance_claim"])
        self.assertEqual(result["cases"][0]["wall_ns"]["pairs"], 2)
        self.assertIn("cache_ablation_profile_on", result)
        path = Path(gate["rows"][0]["canonical"]["directory"]) / "output.mlir"
        path.write_text("module { /* tampered */ }\n")
        with self.assertRaisesRegex(bench.InvalidEvidence, "artifact changed"):
            bench.report(run)

    def test_phase_summary_cannot_disagree_with_raw_invocation(self):
        run, _ = self.register()
        bench.correctness(run, timeout=10)
        path = run / "correctness.json"; record = bench.read(path)
        record["rows"][0]["observations"][1]["process"]["wall_ns"] += 1
        bench.write(path, record)
        with self.assertRaisesRegex(bench.InvalidEvidence, "summary differs"):
            bench.verify_phase(run, "correctness")

    def test_expected_cost_failure_stays_untimed_and_reported_na(self):
        run = self.stage / "run"
        failure = {"returncode":1,"stats_status":"failed","failure_kind":"cost-failure","failure_reason":"layout cost evaluation failed"}
        cases = [{"id":"fixture","timed":True},{"id":"failure","timed":False,"expected_failure":failure}]
        with mock.patch.object(bench,"specifications",return_value=cases), mock.patch.object(bench,"render_fixture",side_effect=lambda row: "module {} // EXPECTED_COST_FAILURE" if row["id"]=="failure" else "module {}"):
            bench.registration(run,self.baseline,self.candidate,self.cpu,pairs=2,profile_pairs=1,candidate_off_option="memoize-costs=false")
        bench.correctness(run,timeout=10)
        bench.measure(run,timeout=10)
        result=bench.report(run)
        self.assertEqual([row["case"] for row in result["cases"]],["fixture"])
        rejected=next(row for row in result["support_outcomes"] if row["case"]=="failure")
        self.assertIn("N/A",rejected["performance"])
        self.assertEqual(rejected["expected_failure"],failure)
        self.assertFalse(rejected["timed"])

    def test_expected_failure_does_not_accept_another_failure_kind(self):
        run = self.stage / "run"
        failure = {"returncode":1,"stats_status":"failed","failure_kind":"hard-conflict","failure_reason":"layout cost evaluation failed"}
        with mock.patch.object(bench,"specifications",return_value=[{"id":"failure","timed":False,"expected_failure":failure}]), mock.patch.object(bench,"render_fixture",return_value="module {} // EXPECTED_COST_FAILURE"):
            bench.registration(run,self.baseline,self.candidate,self.cpu,pairs=2,profile_pairs=1)
        with self.assertRaisesRegex(bench.InvalidEvidence,"failure outcome differs"):
            bench.correctness(run,timeout=10)
        self.assertEqual(bench.read(run/"correctness.json")["status"],"failed")

    def test_failed_cli_does_not_become_completed_measurement(self):
        run, _ = self.register()
        self.candidate.write_text("#!/bin/sh\nexit 7\n"); self.candidate.chmod(0o755)
        with self.assertRaisesRegex(bench.InvalidEvidence, "CLI failed"):
            bench.correctness(run, timeout=10)
        record = bench.read(run / "correctness.json")
        self.assertEqual(record["status"], "failed")
        self.assertTrue(record["no_performance_claim"])
        with self.assertRaisesRegex(bench.InvalidEvidence, "incomplete"):
            bench.verify_phase(run, "correctness")


if __name__ == "__main__":
    unittest.main()
