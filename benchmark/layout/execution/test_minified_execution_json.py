"""Real small CLI JSON-format contract; CPU only, no kernel compiler/CUDA."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import backend
from demand_repair_fixture import source

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get("FRISK_JOINT_ABLATION_BENCH", ROOT / "build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def small_case(directory, bench=BENCH):
    """Persist a complete small CLI replay comparison for tests/development."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    commands = []

    def run(label, args):
        completed = subprocess.run([str(bench), *map(str, args)], cwd=ROOT, env=env,
                                   capture_output=True, text=True, timeout=180)
        (directory / (label + ".log")).write_text(completed.stdout + completed.stderr)
        commands.append(dict(label=label, args=list(map(str, args)), exit_code=completed.returncode))
        if completed.returncode:
            raise AssertionError(label + ": " + completed.stdout + completed.stderr)

    spec = dict(key="json-format-small", case_id="no-conflict", graph_name="mixed",
                shape=[8, 16], tail="mixed", abi_stratum="weak-vector", dtype="float32",
                threads=128, split="CPU-JSON-format-contract-only")
    input_path = directory / "input.mlir"
    input_path.write_text(source(spec, ROOT))
    prepared = directory / "prepared"
    run("prepare", ["--prepare", input_path, "no-conflict", prepared, "--natural"])
    prepared_hashes = {p.name: digest(p) for p in prepared.iterdir() if p.is_file()}
    common = ["--prepared=" + str(prepared), "--case=no-conflict", "--selection=J",
              "--conversion=C", "--demand-ledger"]
    anchor = directory / "anchor"
    run("anchor", [*common, "--candidate-archive=2", "--output-dir=" + str(anchor)])
    rows = []
    packets = []
    emissions = []
    for name, flags in [("default", []), ("false", ["--minify-execution-json=false"]),
                        ("bare", ["--minify-execution-json"]),
                        ("true", ["--minify-execution-json=true"])]:
        output = directory / name
        run(name, [*common, "--assignment=" + str(anchor / "record.json"),
                   "--output-dir=" + str(output), *flags])
        packet = json.loads((output / "execution.json").read_text())
        record = json.loads((output / "record.json").read_text())
        assert record["status"] == "success" and record["budget_contract"]["used"] == 0
        assert record["complete_assignment_sha256"] == json.loads((anchor / "record.json").read_text())["complete_assignment_sha256"]
        ttgir, info = backend.emit(packet)
        (output / "emitted.ttgir").write_text(ttgir)
        (output / "emitted-config.json").write_text(json.dumps(info, sort_keys=True, indent=2) + "\n")
        packets.append(packet)
        emissions.append((ttgir, info))
        rows.append(dict(mode=name, execution_bytes=(output / "execution.json").stat().st_size,
                         execution_sha256=digest(output / "execution.json"),
                         TTGIR_sha256=digest(output / "emitted.ttgir"),
                         config_sha256=digest(output / "emitted-config.json"),
                         record_sha256=digest(output / "record.json"),
                         complete_assignment_sha256=record["complete_assignment_sha256"],
                         cost=record["cost"]))
    assert all(x == packets[0] for x in packets)
    assert all(x == emissions[0] for x in emissions)
    assert prepared_hashes == {p.name: digest(p) for p in prepared.iterdir() if p.is_file()}
    result = dict(status="passed", rows=rows, commands=commands,
                  prepared_hashes_unchanged=prepared_hashes,
                  parsed_packet_objects_equal=True, TTGIR_and_config_bytes_equal=True,
                  assignment_replay_zero_search=True, GPU_or_kernel_compiler_called=False)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class MinifiedExecutionJSONTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BENCH.is_file():
            raise RuntimeError("Build frisk_joint_ablation_bench before this CLI contract test")
        cls.temporary = tempfile.TemporaryDirectory(prefix="frisk-json-format-")
        cls.directory = Path(cls.temporary.name)
        cls.result = small_case(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_default_and_explicit_false_preserve_pretty_bytes(self):
        default = (self.directory / "default/execution.json").read_bytes()
        self.assertEqual(default, (self.directory / "false/execution.json").read_bytes())
        self.assertIn(b'\n  "', default)
        self.assertNotEqual(default, (self.directory / "bare/execution.json").read_bytes())

    def test_true_and_bare_only_remove_packet_whitespace(self):
        compact = (self.directory / "bare/execution.json").read_bytes()
        self.assertEqual(compact, (self.directory / "true/execution.json").read_bytes())
        self.assertNotIn(b"\n", compact)
        self.assertLess(len(compact), self.result["rows"][0]["execution_bytes"])
        self.assertTrue(self.result["parsed_packet_objects_equal"])
        self.assertTrue(self.result["TTGIR_and_config_bytes_equal"])
        self.assertTrue(all(row["cost"] == self.result["rows"][0]["cost"] for row in self.result["rows"]))

    def test_other_artifacts_keep_pretty_format_and_package_bindings(self):
        self.assertTrue(self.result["prepared_hashes_unchanged"])
        for path in ["prepared/domains.json", "prepared/prepared.json", "anchor/candidate-archive.json",
                     "bare/record.json", "true/record.json"]:
            self.assertIn(b'\n  "', (self.directory / path).read_bytes())
        self.assertTrue(self.result["assignment_replay_zero_search"])

    def test_invalid_boolean_is_rejected(self):
        result = subprocess.run([str(BENCH), "--minify-execution-json=maybe"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("minify-execution-json requires true or false", result.stderr)


if __name__ == "__main__":
    unittest.main()
