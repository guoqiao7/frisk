#!/usr/bin/env python3
"""Reproducible CPU-only selection-quality experiments. Standard library only.

Existing output directories are rejected (no resume). Run correctness/core/timing
separately. `report` re-verifies every archived byte before merging evidence.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import selection_report

ROOT = Path(__file__).resolve().parents[3]
SHAPES = ((64, 64, 16, 128), (128, 128, 16, 256), (128, 512, 16, 256),
          (256, 128, 16, 256), (128, 128, 32, 256), (256, 256, 16, 512),
          (512, 128, 16, 1024), (128, 128, 16, 512))
POLICIES = ("Auto", "Square", "FullRow", "FullCol")
SOLVER_OPTIONS = {"exact_combination_limit": 256, "beam_width": 32,
                  "max_expanded_states": 65536, "max_refinement_probes": 256}
ORACLE_LIMITS = {"nodes": 65536, "evaluations": 4096, "milliseconds": 120000}


class ValidationError(ValueError):
    """Evidence is missing, contradictory, or corrupt; not a support sample."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def geometry_candidates(case, mode):
    """Bounded input-derived matrix; actual preparation still validates support."""
    m, n, k, threads = SHAPES[int(case[1:])]
    result = []
    gm = 1
    while gm <= threads // 128:
        gn = threads // 128 // gm
        if m % (64 * gm) == 0 and n % (8 * gn) == 0 and (mode != "rs" or m * k * gn <= 65536):
            result.append(f"{gm},{gn}")
        gm *= 2
    return result


def row(case, mode, phase, consumer="store", policy="Auto", consumer_grid=None):
    suffix = f"-{consumer_grid}" if consumer_grid else ""
    name = f"{case}-{mode}-native-{policy}" if phase == "native" else f"{case}-{mode}-{consumer}{suffix}"
    r = {"id": name, "case": case, "mode": mode, "phase": phase, "consumer": consumer, "policy": policy}
    if phase in ("local", "controlled"):
        r["oracle"] = True
    if consumer_grid:
        r["consumer_grid"] = consumer_grid
    return r


def native_manifest():
    return [row(f"C{i}", mode, "native", policy=policy) for i in range(8)
            for mode in ("ss", "rs") for policy in POLICIES]


def direct_manifest():
    return [row(f"C{i}", mode, "controlled") for i in range(8) for mode in ("ss", "rs")]


def consumer_manifest():
    result = []
    for case in ("C1", "C2", "C3"):
        for mode in ("ss", "rs"):
            for axis in (0, 1):
                result.append(row(case, mode, "controlled", f"reduce{axis}"))
                result.extend(row(case, mode, "controlled", f"fixed{axis}", consumer_grid=g)
                              for g in geometry_candidates(case, mode))
    return result


def local_manifest():
    return [row(case, mode, "local", "local") for case in ("C1", "C2", "C3") for mode in ("ss", "rs")]


def branch_rows(parent, grids):
    return [dict(parent, id=parent["id"] + "-grid-" + g, parent=parent["id"], restrict_grid=g) for g in grids]


def timing_manifest():
    return [dict(row(case, "ss", "native", policy=policy), id=f"{case}-ss-{policy}-run-{repeat}",
                 warmup=repeat == 0, repeat=repeat) for case in ("C1", "C2")
            for policy in POLICIES for repeat in range(6)]


def auxiliary_manifests(mode):
    if mode == "timing":
        return [dict(row(case, "ss", "emit-input", policy=policy), id=f"input-{case}-{policy}")
                for case in ("C1", "C2") for policy in POLICIES], []
    fixed = [m for m in experiment_manifest(mode) if m["consumer"].startswith("fixed") and not m.get("parent")]
    return ([dict(m, id="emit-" + m["id"], phase="emit-input", oracle=False) for m in fixed],
            [dict(m, id="native-entry-" + m["id"], phase="native", oracle=False) for m in fixed])


def command_for(bench, manifest, output):
    command = [str(bench), "--case", manifest["case"], "--mode", manifest["mode"],
               "--phase", manifest["phase"], "--policy", manifest["policy"],
               "--consumer", manifest["consumer"], "--output-dir", str(output)]
    for key in ("consumer_grid", "restrict_grid"):
        if manifest.get(key):
            command.extend(["--" + key.replace("_", "-"), manifest[key]])
    if manifest.get("oracle"):
        command.append("--oracle")
    return command


def execute(command, logs, timeout, measure_rss=False):
    """Separate process group; preserve logs and return code even on timeout."""
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    if any(logs.iterdir()):
        raise ValidationError(f"refusing existing process artifacts: {logs}")
    start = time.monotonic()
    result = {"command": command, "parent_timeout_seconds": timeout, "timeout": False,
              "status": "exited", "returncode": None}
    if measure_rss:
        result["wall_observation"] = "parent monotonic whole-CLI duration; wait4 polling interval 0.005 seconds"
        result["peak_rss_scope"] = "per-child wait4 rusage.ru_maxrss, Linux KiB"
    with (logs / "stdout.txt").open("wb") as out, (logs / "stderr.txt").open("wb") as err:
        try:
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
            if measure_rss:
                while True:
                    pid, wait_status, usage = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(wait_status)
                        result["peak_rss_kib"] = usage.ru_maxrss
                        break
                    if time.monotonic() - start > timeout:
                        result.update(timeout=True, status="process-timeout")
                        os.killpg(process.pid, signal.SIGKILL)
                        _, wait_status, usage = os.wait4(process.pid, 0)
                        process.returncode = os.waitstatus_to_exitcode(wait_status)
                        result["peak_rss_kib"] = usage.ru_maxrss
                        break
                    time.sleep(0.005)
            else:
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    result.update(timeout=True, status="process-timeout")
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            result["returncode"] = process.returncode
        except OSError as error:
            result.update(status="process-launch-failure", error=str(error))
    result["wall_seconds"] = time.monotonic() - start
    json_write(logs / "process.json", result)
    return result


def validate_exit(record, code):
    status, fatal = record.get("status"), record.get("fatal")
    expected = 2 if status == "failed" and fatal is True else 1 if status == "failed" and fatal is False else 0
    if status not in ("failed", "success", "local-analysis-only", "input-emitted") or not isinstance(fatal, bool) or code != expected or (status != "failed" and fatal):
        raise ValidationError(f"exit/JSON contradiction: exit={code}, status={status}, fatal={fatal}")


def cost_vector(value):
    values = value.get("cost") if isinstance(value, dict) else None
    if not isinstance(values, list) or len(values) != 9 or any(type(v) is not int or v < 0 or v > 2**64 - 1 for v in values):
        raise ValidationError("cost must contain nine unsigned uint64 fields")
    if type(value.get("saturated")) is not bool:
        raise ValidationError("cost saturation flag missing")
    return values


def reconcile(cost, contributions):
    total = cost_vector(cost)
    sums = [0] * 9
    saturated = False
    for contribution in contributions:
        estimate = contribution.get("cost", contribution.get("estimate"))
        values = cost_vector(estimate)
        saturated |= estimate["saturated"]
        for i, value in enumerate(values):
            sums[i] += value
            if sums[i] > 2**64 - 1:
                sums[i] = 2**64 - 1
                saturated = True
    if sums != total or saturated != cost["saturated"]:
        raise ValidationError("cost contribution sum/saturation mismatch")


def verify_artifacts(directory, hashes):
    base = Path(directory).resolve()
    if not hashes or not isinstance(hashes, dict):
        raise ValidationError("missing artifact hash table")
    for name, digest in hashes.items():
        path = base / name
        if Path(name).is_absolute() or path.resolve().parent != base or not path.is_file() or sha256(path) != digest:
            raise ValidationError(f"missing, escaping, or hash-mismatched artifact: {name}")


def validate_record(record, directory, manifest, bench):
    if record.get("schema_version") != 1 or record.get("kind") != "layout-selection":
        raise ValidationError("unsupported harness schema/kind")
    spec = record.get("spec", {})
    for key, expected in (("id", manifest["case"]), ("mode", manifest["mode"]),
                          ("consumer", manifest["consumer"]), ("policy", manifest["policy"])):
        if spec.get(key) != expected:
            raise ValidationError(f"spec mismatch: {key}")
    m, n, k, threads = SHAPES[int(manifest["case"][1:])]
    invariants = {"m": m, "n": n, "k": k, "threads": threads, "input_type": "f16", "accumulator_type": "f32",
                  "trans_a": False, "trans_b": False, "shared_alignment": 1024, "global_alignment": 128,
                  "producer_a": "shared-alloc-layout-view" if manifest["mode"] == "ss" else "tensor-dense-one",
                  "producer_b": "shared-alloc-layout-view", "init": "tensor-dense-zero",
                  "input_scope": "shared/tensor-already-prepared-no-global-copy"}
    if any(spec.get(key) != expected for key, expected in invariants.items()):
        raise ValidationError("shape/threads/types/producer/ABI input contract mismatch")
    if record.get("phase") != manifest["phase"]:
        raise ValidationError("phase mismatch")
    if spec.get("consumer_grid") != ([int(x) for x in manifest["consumer_grid"].split(",")] if manifest.get("consumer_grid") else None):
        raise ValidationError("consumer grid mismatch")
    hashes = record.get("artifacts_sha256", {})
    verify_artifacts(directory, hashes)
    if Path(record.get("binary_path", "")).resolve() != Path(bench).resolve() or record.get("binary_sha256") != sha256(bench):
        raise ValidationError("benchmark binary changed")
    for field in ("source_digest", "config_digest", "binary_sha256"):
        if not isinstance(record.get(field), str) or len(record[field]) != 64:
            raise ValidationError(f"missing identity: {field}")
    if record["source_digest"] != hashes.get("source-manifest.txt"):
        raise ValidationError("source manifest identity mismatch")
    if record.get("input_sha256") != hashes.get("input.mlir") or record.get("output_sha256") != hashes.get("output.mlir"):
        raise ValidationError("input/output identity mismatch")
    source = Path(directory) / "source-manifest.txt"
    for line in source.read_text().splitlines():
        name, digest = line.rsplit(":", 1)
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest:
            raise ValidationError(f"source differs from compiled manifest: {name}")
    projection = record.get("native_snapshot_projection")
    if projection and hashes.get(projection["artifact"]) != projection["sha256"]:
        raise ValidationError("native projection identity mismatch")
    frozen = record.get("frozen_domain")
    if frozen:
        if frozen["sha256"] != hashes.get(frozen["artifact"]):
            raise ValidationError("frozen domain hash mismatch")
        if frozen.get("restricted_sha256") != hashes.get(frozen.get("branch_artifact")):
            raise ValidationError("restricted domain hash mismatch")
    if record.get("status") not in ("success", "local-analysis-only"):
        return
    if manifest["phase"] == "native":
        if projection is None or projection.get("scope") != "actual-native-final-attempt" or not projection.get("attempt"):
            raise ValidationError("native actual final-attempt projection missing")
        # Native solver options are recorded in the production trace, not top-level.
        trace = json.loads((Path(directory) / "trace.json").read_text())
        record["solver_options"] = trace_solver_options(trace)
    else:
        if record.get("solver_options") != SOLVER_OPTIONS or frozen is None:
            raise ValidationError("controlled budget/frozen domain missing or changed")
        if record.get("branch") != manifest.get("restrict_grid", "joint"):
            raise ValidationError("restriction mismatch")
    reconcile(record["cost"], record["cost_contributions"])
    if manifest["phase"] != "local" and (record.get("output_abi_verified") is not True or record.get("replay_stable") is not True):
        raise ValidationError("successful full graph lacks actual ABI/replay validation")
    o = record.get("oracle")
    if manifest.get("oracle") and not o:
        raise ValidationError("requested oracle missing")
    if not o:
        return
    if json.loads((Path(directory) / "oracle.json").read_text()) != o:
        raise ValidationError("oracle artifact and result oracle disagree")
    if o.get("limits") != ORACLE_LIMITS or o.get("fatal"):
        raise ValidationError("oracle budgets changed or fatal oracle")
    if o.get("frozen_domain_sha256") != frozen["sha256"] or o.get("restricted_domain_sha256") != frozen["restricted_sha256"]:
        raise ValidationError("oracle domain mismatch")
    if o.get("status") in ("complete", "infeasible") and (not o.get("exhausted") or o.get("unknown_assignments") != 0):
        raise ValidationError("incomplete/Unknown traversal cannot prove optimality/infeasibility")
    if o.get("best_cost") is not None:
        reconcile(o["best_cost"], o["cost_contributions"])
        if o.get("incumbent_actual_verified") is not True or record.get("oracle_incumbent_actual_verified") is not True:
            raise ValidationError("published incumbent lacks actual verification")
        scope = "local-analysis-only" if manifest["phase"] == "local" else "actual-IR-ABI-pipeline-replay"
        if o.get("incumbent_scope") != scope:
            raise ValidationError("oracle actual verification scope mismatch")
        prefix = o.get("incumbent_artifact_prefix")
        modes = {"": "reused-default-identical-assignment-and-conversion-plan",
                 "oracle-": "separate-clone-recollected-domain-identity-checked"}
        if prefix not in modes or o.get("incumbent_validation") != modes[prefix]:
            raise ValidationError("unknown oracle witness validation mode")
        if not prefix and any(o.get(key) != record.get(key) for key in ("assignment", "conversions", "existing_conversions")):
            raise ValidationError("reused-default incumbent assignment/plan differs")
        required = [prefix + "output.mlir"]
        if manifest["phase"] != "local":
            required += [prefix + "replay.mlir", prefix + "pipeline-statistics.json"]
            if hashes.get(prefix + "output.mlir") != hashes.get(prefix + "replay.mlir"):
                raise ValidationError("oracle output/replay bytes differ")
        if any(name not in hashes for name in required):
            raise ValidationError("oracle witness artifacts missing")
    if o.get("status") == "complete":
        if not selection_report.complete(record):
            raise ValidationError("complete oracle lacks all-optimal geometries/cost")
        selected, optimum = cost_vector(record["cost"]), cost_vector(o["best_cost"])
        if selected < optimum:
            raise ValidationError("fatal correctness: default below complete oracle")
        expected = "search-gap" if selected > optimum else "equal-optimum"
        if record.get("oracle_comparison") != expected:
            raise ValidationError("oracle/default comparison contradiction")


def trace_solver_options(trace):
    options = trace.get("solver_options", {})
    # The trace uses camelCase; the controlled schema uses snake_case.
    result = {"exact_combination_limit": options.get("exactCombinationLimit"),
              "beam_width": options.get("beamWidth"),
              "max_expanded_states": options.get("maxExpandedStates"),
              "max_refinement_probes": options.get("maxRefinementProbes")}
    if result != SOLVER_OPTIONS:
        raise ValidationError("native solver budgets differ or are missing")
    if options.get("refineBeamSolution") is not True:
        raise ValidationError("native refinement option changed")
    return result


def assess_harness(bench, manifest, directory, process):
    """The one status derivation shared by live execution and archive reload."""
    if process["status"] != "exited":
        return {"status": process["status"]}
    try:
        record = json.loads((Path(directory) / "result.json").read_text())
        validate_exit(record, process["returncode"])
        validate_record(record, directory, manifest, bench)
        if record.get("command") != process["command"]:
            raise ValidationError("harness command identity mismatch")
        if manifest["phase"] == "emit-input" and manifest["consumer"].startswith("fixed") and record["status"] == "input-emitted":
            canonical_encoding((Path(directory) / "input.mlir").read_text())
        return {"result": record, "status": "fatal-correctness" if record["fatal"] else record["status"]}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        return {"status": "infrastructure-error", "error": str(error)}


def run_row(bench, manifest, root, timeout):
    root = Path(root)
    directory = root / manifest["id"]
    command = command_for(bench, manifest, directory)
    logs = root / (manifest["id"] + "-process")
    process = execute(command, logs, timeout)
    result = {"manifest": manifest, "artifact_dir": directory.name, "process_dir": logs.name,
              "process": process, "status": process["status"]}
    result.update(assess_harness(bench, manifest, directory, process))
    result["files_sha256"] = {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file()} if directory.is_dir() else {}
    result["process_files_sha256"] = {p.name: sha256(p) for p in sorted(logs.iterdir()) if p.is_file()}
    return result


def verify_saved_row(root, row_result):
    for directory, key in (("artifact_dir", "files_sha256"), ("process_dir", "process_files_sha256")):
        if key not in row_result or not isinstance(row_result[key], dict) or directory not in row_result:
            raise ValidationError(f"missing raw hash table: {key}")
        location = archived_path(root, row_result[directory])
        hashes = row_result[key]
        actual = {p.name for p in location.iterdir() if p.is_file()} if location.is_dir() else set()
        if set(hashes) != actual:
            raise ValidationError(f"incomplete raw hash table: {key}")
        if key == "process_files_sha256" and set(hashes) != {"stdout.txt", "stderr.txt", "process.json"}:
            raise ValidationError("missing process stdout/stderr/return-code evidence")
        if hashes:
            verify_artifacts(location, hashes)
        elif row_result["status"] in ("success", "local-analysis-only", "input-emitted", "failed", "fatal-correctness", "timing-instability"):
            raise ValidationError("completed row cannot omit raw artifacts")


def archived_path(root, location):
    """Only artifact locations are relocated; original command strings stay exact."""
    if not isinstance(location, str) or not location:
        raise ValidationError("missing archived artifact location")
    return (Path(root) / location).resolve()


def validate_branches(joint, branches, root=None, directories=None):
    frozen = joint.get("frozen_domain") or {}
    for index, branch in enumerate(branches):
        b = branch.get("frozen_domain") or {}
        if not frozen.get("sha256") or frozen["sha256"] != b.get("sha256"):
            raise ValidationError("branches do not share the initial frozen domain")
        if len(frozen.get("geometry_candidates", [])) > 1 and frozen["sha256"] == b.get("restricted_sha256"):
            raise ValidationError("multi-geometry restriction did not change branch identity")
        if joint.get("solver_options") != branch.get("solver_options"):
            raise ValidationError("branch solver options differ")
        for key in ("source_digest", "config_digest", "binary_sha256", "input_sha256", "spec"):
            if joint.get(key) != branch.get(key):
                raise ValidationError(f"branch identity mismatch: {key}")
        # Frozen hash alone covers all unrestricted non-geometry candidates;
        # inspect branch snapshot as well, to detect unintended branch mutation.
        if root is not None or directories is not None:
            def relative_directory(record):
                command = record["command"]
                return Path(root) / Path(command[command.index("--output-dir") + 1]).name
            jdir = directories[0] if directories is not None else relative_directory(joint)
            bdir = directories[index + 1] if directories is not None else relative_directory(branch)
            initial = json.loads((jdir / "domains.json").read_text())
            restricted = json.loads((bdir / "branch-domains.json").read_text())
            left = [v for v in initial["variables"] if v["kind"] != 2]
            right = [v for v in restricted["variables"] if v["kind"] != 2]
            if left != right:
                raise ValidationError("geometry restriction changed non-geometry candidates")


def without_timings(value):
    if isinstance(value, dict):
        return {k: without_timings(v) for k, v in value.items() if k != "timings"}
    if isinstance(value, list):
        return [without_timings(v) for v in value]
    return value


def timing_command(opt, input_path, statistics, output):
    return [str(opt), str(input_path), f"--frisk-layout-pipeline=stats-file={statistics}", "-o", str(output)]


def experiment_manifest(mode):
    if mode == "smoke":
        direct = direct_manifest()[:1]
        return native_manifest()[:1] + direct + branch_rows(direct[0], ["1,1"]) + [row("C0", "ss", "local", "local")]
    if mode == "core":
        controlled = [r for r in direct_manifest() if r["case"] == "C1"]
        controlled += [r for r in consumer_manifest() if r["case"] == "C1" and r["mode"] == "ss"]
        native = [r for r in native_manifest() if r["case"] == "C1" and r["policy"] == "Auto"]
        local = [r for r in local_manifest() if r["case"] == "C1"]
    elif mode == "correctness":
        native, controlled, local = native_manifest(), direct_manifest() + consumer_manifest(), local_manifest()
    else:
        raise ValidationError("unknown experiment mode")
    result = native + local
    for parent in controlled:
        result.append(parent)
        result.extend(branch_rows(parent, geometry_candidates(parent["case"], parent["mode"])))
    return result


def core_gate(rows):
    required = experiment_manifest("core")
    by_id = {r["manifest"]["id"]: r for r in rows}
    failures = []
    for m in required:
        r = by_id.get(m["id"], {})
        record = r.get("result", {})
        expected = "local-analysis-only" if m["phase"] == "local" else "success"
        valid = r.get("status") == expected
        if m.get("oracle"):
            valid = valid and selection_report.complete(record)
        else:
            projection = record.get("native_snapshot_projection") or {}
            controlled = by_id.get(f"{m['case']}-{m['mode']}-store", {}).get("result", {})
            valid = (valid and record.get("output_abi_verified") is True and record.get("replay_stable") is True
                     and projection.get("scope") == "actual-native-final-attempt"
                     and selection_report.native_comparison(record, controlled)["status"] == "same-domain")
        if not valid:
            failures.append(m["id"])
    return {"passed": not failures, "required_oracle_rows": [r["id"] for r in required if r.get("oracle")],
            "required_native_rows": [r["id"] for r in required if not r.get("oracle")],
            "missing_or_incomplete": failures}


def dataset_exit_code(data):
    failed = bool(data.get("integrity_errors")) or any(r["status"] in selection_report.FAILED_STATUSES for r in data["rows"])
    if selection_report.is_formal(data) and not core_gate(data["rows"])["passed"]:
        failed = True
    return 1 if failed else 0


def run_batch(bench, manifest, output, timeout, jobs):
    if jobs < 1:
        raise ValidationError("jobs must be positive")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # executor.map yields input order, independent of completion order.
        return list(pool.map(lambda m: run_row(bench, m, output, timeout), manifest))


def claim_output(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValidationError(f"output must be NEW; resume is deliberately unsupported: {path}") from error


def identity(bench, opt):
    bench, opt = Path(bench).resolve(), Path(opt).resolve()
    files = [Path(__file__).resolve(), Path(selection_report.__file__).resolve(),
             ROOT / "test/python/layout_selection_quality_test.py", ROOT / "test/CMakeLists.txt",
             ROOT / "CMakeLists.txt", bench, opt]
    for binary in (bench, opt):
        for parent in binary.parents:
            cache = parent / "CMakeCache.txt"
            if cache.is_file():
                files.append(cache)
                break
    manifest = bench.parent / "source-manifest.txt"
    if not manifest.is_file():
        raise ValidationError("compiled source manifest is required alongside benchmark")
    files.append(manifest)
    for line in manifest.read_text().splitlines():
        name, digest = line.rsplit(":", 1)
        source = ROOT / name
        if not source.is_file() or sha256(source) != digest:
            raise ValidationError(f"compiled/source identity mismatch before execution: {name}")
        files.append(source)
    hashes = {str(p.resolve()): sha256(p) for p in sorted(set(files))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    return {"files_sha256": hashes, "bench_path": str(bench), "frisk_opt_path": str(opt), "git_head_context_only": git,
            "python": sys.version, "platform": sys.platform, "compiled_source_digest": sha256(manifest),
            "solver_options": SOLVER_OPTIONS, "oracle_limits": ORACLE_LIMITS,
            "identity_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def check_identity(expected):
    if not isinstance(expected, dict) or not expected.get("bench_path") or not expected.get("frisk_opt_path"):
        raise ValidationError("missing complete source/config/binary/script identity")
    actual = identity(Path(expected["bench_path"]), Path(expected["frisk_opt_path"]))
    if set(expected) != set(actual):
        raise ValidationError("incomplete identity schema")
    for key in actual:
        if key != "git_head_context_only" and expected[key] != actual[key]:
            raise ValidationError(f"run identity changed or incomplete: {key}")


def publish_dataset(output, data):
    output = Path(output)
    data["summary"] = selection_report.summarize(data["rows"])
    modes = {}
    for r in data["rows"]:
        validation = (r.get("result", {}).get("oracle") or {}).get("incumbent_validation")
        if validation:
            modes[validation] = modes.get(validation, 0) + 1
    data["summary"]["oracle_incumbent_validation_modes"] = modes
    data["core_gate"] = core_gate(data["rows"])
    json_write(output / "identity.json", data.get("identity", {}))
    json_write(output / "manifest.json", data.get("manifest", []))
    json_write(output / "results.json", data)
    (output / "report.md").write_text(selection_report.render(data))
    hashes = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "dataset-files.json"}
    json_write(output / "dataset-files.json", hashes)


def load_dataset(path):
    path = Path(path).resolve()
    hashes = json.loads((path.parent / "dataset-files.json").read_text())
    actual_files = {p.name for p in path.parent.iterdir() if p.is_file() and p.name != "dataset-files.json"}
    if not isinstance(hashes, dict) or set(hashes) != actual_files or not {"results.json", "report.md", "identity.json", "manifest.json"}.issubset(hashes):
        raise ValidationError("missing batch artifact hashes")
    verify_artifacts(path.parent, hashes)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("kind") != "selection-quality-batch":
        raise ValidationError("unsupported batch schema")
    if not {"mode", "identity", "manifest", "rows", "preflight", "native_entry_probes", "integrity_errors", "core_gate", "summary"}.issubset(data):
        raise ValidationError("incomplete batch schema")
    check_identity(data["identity"])
    if json.loads((path.parent / "identity.json").read_text()) != data["identity"] or json.loads((path.parent / "manifest.json").read_text()) != data["manifest"]:
        raise ValidationError("batch identity/manifest artifacts disagree")
    errors = []
    for config, rows, preflight, probes in batch_segments(data):
        mode = config["mode"]
        failed_preflight = any(r["status"] != "input-emitted" for r in preflight)
        if mode == "timing" and failed_preflight:
            raise ValidationError("timing samples cannot precede failed input emission")
        for r in preflight:
            reload_row(path.parent, r, config, data["identity"])
        for r in rows + probes:
            reload_row(path.parent, r, config, data["identity"], blocked=failed_preflight)
        if mode == "timing":
            stable = {}
            for r in rows:
                key = (r["manifest"]["case"], r["manifest"]["policy"])
                signature = (r.get("output_sha256"), r.get("non_timing_statistics_sha256"))
                if r["status"] in ("success", "timing-instability"):
                    expected = "timing-instability" if key in stable and stable[key] != (None, None) and signature != stable[key] else "success"
                    if r["status"] != expected:
                        raise ValidationError("aggregate timing stability differs from raw repeated evidence")
                stable.setdefault(key, signature)
        else:
            errors.extend(derive_integrity_errors(rows, preflight, probes, path.parent))
    if data.get("integrity_errors") != errors:
        raise ValidationError("aggregate integrity errors differ from revalidated frozen-domain/preflight evidence")
    if data.get("core_gate") != core_gate(data["rows"]):
        raise ValidationError("aggregate core gate differs from verified core evidence")
    if data.get("summary") != selection_report.summarize(data["rows"]):
        raise ValidationError("aggregate summary differs from verified rows")
    if (path.parent / "report.md").read_text() != selection_report.render(data):
        raise ValidationError("report differs from verified data")
    return data


def batch_segments(data):
    """Exact mode manifests, including auxiliary/failure rows; no silent resizing."""
    if data.get("mode") == "merged":
        configs = data.get("components")
        if not isinstance(configs, list) or not configs or any(not isinstance(c, dict) for c in configs) or data.get("component_modes") != [c.get("mode") for c in configs]:
            raise ValidationError("missing merged component schema")
    else:
        configs = [data]
    for key in ("rows", "preflight", "native_entry_probes", "manifest"):
        if not isinstance(data.get(key), list):
            raise ValidationError(f"missing batch list: {key}")
    cursors = [0, 0, 0]
    expected_manifest, seen, segments = [], set(), []
    for config in configs:
        mode = config.get("mode")
        if mode not in ("correctness", "core", "smoke", "timing"):
            raise ValidationError("unknown component mode")
        if (type(config.get("jobs")) is not int or config["jobs"] < 1 or
                type(config.get("parent_timeout_seconds")) not in (int, float) or
                not math.isfinite(config["parent_timeout_seconds"]) or config["parent_timeout_seconds"] <= 0 or
                not isinstance(config.get("execution_root"), str) or not Path(config["execution_root"]).is_absolute()):
            raise ValidationError("missing jobs/timeout/original execution-root schema")
        if mode == "timing" and config["jobs"] != 1:
            raise ValidationError("timing evidence must be serial")
        if mode != "timing" and config.get("oracle_timeout_milliseconds_distinct_from_parent") != 120000:
            raise ValidationError("oracle and parent timeouts missing or confused")
        main = timing_manifest() if mode == "timing" else experiment_manifest(mode)
        emits, probes = auxiliary_manifests(mode)
        groups = []
        for i, (key, expected) in enumerate(zip(("rows", "preflight", "native_entry_probes"), (main, emits, probes))):
            actual = data[key][cursors[i]:cursors[i] + len(expected)]
            if any(not isinstance(r, dict) for r in actual) or [r.get("manifest") for r in actual] != expected:
                raise ValidationError(f"exact {mode} manifest membership/order differs: {key}")
            cursors[i] += len(expected)
            groups.append(actual)
            for m in expected:
                if m["id"] in seen:
                    raise ValidationError("duplicate manifest row identity")
                seen.add(m["id"])
        expected_manifest.extend(main)
        segments.append((config, *groups))
    if any(cursors[i] != len(data[key]) for i, key in enumerate(("rows", "preflight", "native_entry_probes"))) or data["manifest"] != expected_manifest:
        raise ValidationError("batch manifest contains missing/extra rows")
    return segments


def validate_process(process, command, timeout):
    if (not isinstance(process, dict) or process.get("command") != command or process.get("parent_timeout_seconds") != timeout or
            type(process.get("wall_seconds")) not in (int, float) or not math.isfinite(process["wall_seconds"]) or process["wall_seconds"] < 0):
        raise ValidationError("process command/timeout/wall evidence mismatch")
    state, code = process.get("status"), process.get("returncode")
    if state == "exited":
        valid = process.get("timeout") is False and type(code) is int
    elif state == "process-timeout":
        valid = process.get("timeout") is True and code == -signal.SIGKILL
    elif state == "process-launch-failure":
        valid = process.get("timeout") is False and code is None and bool(process.get("error"))
    else:
        valid = False
    if not valid:
        raise ValidationError("invalid typed process outcome")


def reload_row(root, saved, config, run_identity, blocked=False):
    m = saved["manifest"]
    original = Path(config["execution_root"]) / m["id"]
    if "warmup" in m:
        command = timing_command(run_identity["frisk_opt_path"], Path(config["execution_root"]) / f"input-{m['case']}-{m['policy']}" / "input.mlir", original / "statistics.json", original / "output.mlir")
    else:
        command = command_for(run_identity["bench_path"], m, original)
    if blocked:
        if set(saved) != {"manifest", "status", "command", "reason"} or saved["status"] != "preflight-blocked" or saved["command"] != command or not saved["reason"]:
            raise ValidationError("invalid preflight-blocked row schema")
        return
    if saved["status"] == "preflight-blocked":
        raise ValidationError("row blocked despite fully validated preflight")
    verify_saved_row(root, saved)
    process_dir = archived_path(root, saved["process_dir"])
    process = json.loads((process_dir / "process.json").read_text())
    if process != saved.get("process"):
        raise ValidationError("aggregate process differs from raw process evidence")
    validate_process(process, command, config["parent_timeout_seconds"])
    directory = archived_path(root, saved["artifact_dir"])
    if "warmup" in m:
        actual = assess_timing(directory, process)
        allowed_status = actual["status"] == saved["status"] or actual["status"] == "success" and saved["status"] == "timing-instability"
        for key in ("output_sha256", "non_timing_statistics", "non_timing_statistics_sha256"):
            if actual.get(key) != saved.get(key):
                raise ValidationError("aggregate timing differs from raw IR/statistics")
    else:
        actual = assess_harness(Path(run_identity["bench_path"]), m, directory, process)
        allowed_status = actual["status"] == saved["status"]
        if actual.get("result") != saved.get("result"):
            raise ValidationError("aggregate result differs from raw harness result")
        if actual.get("result") and actual["result"]["source_digest"] != run_identity["compiled_source_digest"]:
            raise ValidationError("raw source identity differs from batch verified benchmark source manifest")
    if not allowed_status or (actual["status"] == "infrastructure-error" and not saved.get("error")):
        raise ValidationError("aggregate wrapper status differs from raw process/result evidence")


def canonical_preflight(bench, manifest, output, timeout, jobs):
    """Validate fixed canonical encodings before running/inspecting any winners."""
    consumers = [m for m in manifest if m["consumer"].startswith("fixed") and not m.get("parent")]
    emits = [dict(m, id="emit-" + m["id"], phase="emit-input", oracle=False) for m in consumers]
    json_write(Path(output) / "consumer-manifest.json", {"stage": "input-derived-before-winners", "consumers": consumers,
                                                       "encoding_identity": "canonical result fragment from Auto legal geometry; input.mlir hash is concrete identity"})
    results = run_batch(bench, emits, output, timeout, jobs)
    validated = []
    for m, r in zip(consumers, results):
        entry = dict(m, validation_status=r["status"], emit_row=r["manifest"]["id"])
        if r["status"] == "input-emitted":
            try:
                encoding = canonical_encoding((Path(output) / r["artifact_dir"] / "input.mlir").read_text())
                entry.update(encoding=encoding, input_sha256=r["result"]["input_sha256"])
            except (OSError, ValidationError) as error:
                r.update(status="infrastructure-error", error=str(error))
                entry.update(validation_status="infrastructure-error", error=str(error))
        validated.append(entry)
    json_write(Path(output) / "consumer-manifest-validated.json", {"stage": "emit-validated-before-winners", "consumers": validated})
    json_write(Path(output) / "consumer-preflight.json", results)
    return results


def canonical_encoding(text):
    lines = [line for line in text.splitlines() if '"frisk.convert_layout"' in line]
    mmas = [line for line in text.splitlines() if '"frisk.mma"' in line]
    if len(lines) != 1 or len(mmas) != 1 or "frisk.mma_contract" in mmas[0] or "#frisk.distributed" in mmas[0]:
        raise ValidationError("fixed fixture must keep exactly one unbound MMA")
    source, separator, target = lines[0].partition(" -> ")
    if not separator or "#frisk.distributed" in source or "#frisk.distributed" not in target:
        raise ValidationError("fixed fixture must bind consumer encoding, not producer")
    target = target.strip()
    attribute = target[target.index("#frisk.distributed"):-1]
    return {"attribute": attribute, "sha256": hashlib.sha256(attribute.encode()).hexdigest(), "consumer_tensor_type": target,
            "producer_binding": "unbound"}


def validate_matrix(rows, output):
    by_parent = {}
    for r in rows:
        parent = r["manifest"].get("parent")
        if parent:
            by_parent.setdefault(parent, []).append(r)
    errors = []
    for r in rows:
        m, record = r["manifest"], r.get("result", {})
        if m["phase"] != "controlled" or m.get("parent") or not record.get("frozen_domain"):
            continue
        try:
            prepared = [",".join(map(str, g["grid"])) for g in record["frozen_domain"]["geometry_candidates"]]
            expected = geometry_candidates(m["case"], m["mode"])
            if prepared != expected:
                raise ValidationError(f"prepared geometry mismatch: input-derived={expected}, actual={prepared}")
            branch_rows_with_domain = [b for b in by_parent.get(m["id"], []) if b.get("result", {}).get("frozen_domain")]
            branches = [b["result"] for b in branch_rows_with_domain]
            directories = [archived_path(output, saved["artifact_dir"]) for saved in [r] + branch_rows_with_domain]
            validate_branches(record, branches, directories=directories)
        except (ValidationError, OSError, KeyError, TypeError) as error:
            reason = f"artifact-read: {Path(error.filename).name}: {error.strerror}" if isinstance(error, OSError) and error.filename else str(error)
            errors.append({"id": m["id"], "kind": "frozen-domain-integrity", "error": reason})
    return errors


def derive_integrity_errors(rows, preflight, probes, output):
    errors = validate_matrix(rows, output)
    errors += [{"id": r["manifest"]["id"], "kind": "canonical-consumer-preflight", "error": r.get("error", r["status"])}
               for r in preflight if r["status"] != "input-emitted"]
    for r in probes:
        if r["status"] != "preflight-blocked" and (r["status"] != "failed" or r.get("result", {}).get("failure", {}).get("kind") != "unsupported-input-form"):
            errors.append({"id": r["manifest"]["id"], "kind": "native-entry-contract", "error": r["status"]})
    return errors


def blocked_rows(bench, manifest, output):
    return [{"manifest": m, "status": "preflight-blocked", "command": command_for(bench, m, output / m["id"]),
             "reason": "canonical consumer manifest not fully validated; no winner examined"} for m in manifest]


def run_correctness(args, output, run_identity):
    manifest = experiment_manifest(args.mode)
    json_write(output / "manifest.json", manifest)
    preflight = canonical_preflight(args.bench, manifest, output, args.timeout, args.jobs)
    # All emit results are validated before this point; do not read winners to
    # select, drop or replace a consumer. Failures are still represented below.
    if any(r["status"] != "input-emitted" for r in preflight):
        rows = blocked_rows(args.bench, manifest, output)
    else:
        rows = run_batch(args.bench, manifest, output, args.timeout, args.jobs)
    _, probes = auxiliary_manifests(args.mode)
    probe_results = run_batch(args.bench, probes, output, args.timeout, args.jobs) if not any(r["status"] != "input-emitted" for r in preflight) else blocked_rows(args.bench, probes, output)
    errors = derive_integrity_errors(rows, preflight, probe_results, output)
    return {"schema_version": 1, "kind": "selection-quality-batch", "mode": args.mode,
            "identity": run_identity, "jobs": args.jobs, "parent_timeout_seconds": args.timeout, "execution_root": str(output),
            "oracle_timeout_milliseconds_distinct_from_parent": 120000,
            "manifest": manifest, "preflight": preflight, "native_entry_probes": probe_results,
            "rows": rows, "integrity_errors": errors}


def run_timing_sample(opt, input_path, directory, timeout):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    stats, output = directory / "statistics.json", directory / "output.mlir"
    process = execute(timing_command(opt, input_path, stats, output), directory / "process", timeout, measure_rss=True)
    result = dict(assess_timing(directory, process), process=process)
    result["files_sha256"] = {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}
    return result


def assess_timing(directory, process):
    stats, output = Path(directory) / "statistics.json", Path(directory) / "output.mlir"
    result = {"status": process["status"]}
    if process["status"] == "exited":
        try:
            if process["returncode"] != 0:
                raise ValidationError("timing frisk-opt did not succeed")
            statistics = json.loads(stats.read_text())
            if statistics.get("status") != "committed":
                raise ValidationError("timing statistics not committed")
            semantic = without_timings(statistics)
            result.update(status="success", output_sha256=sha256(output), non_timing_statistics=semantic,
                          non_timing_statistics_sha256=hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest())
        except (OSError, ValueError) as error:
            result.update(status="infrastructure-error", error=str(error))
    return result


def active_workers():
    """Best-effort guard for unmanaged compilation/tests; explicit ack also required."""
    workers = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            executable = (proc / "exe").resolve().name
            if executable in ("frisk-opt", "frisk_layout_selection_bench", "FriskLayoutSelectionTests", "ninja", "cc1plus", "clang-18", "clang-19"):
                workers.append({"pid": int(proc.name), "executable": executable})
        except OSError:
            pass
    return workers


def run_timing(args, output, run_identity):
    if args.jobs != 1 or not args.workers_closed:
        raise ValidationError("timing requires --jobs 1 --workers-closed and closed external build/test workers")
    workers = active_workers()
    if workers:
        raise ValidationError(f"timing blocked by active workers: {workers}")
    # Generate the eight policy-specific inputs before the timed stage. No
    # oracle/trace/profile/benchmark work runs concurrently with a timing call.
    inputs, _ = auxiliary_manifests("timing")
    emitted = run_batch(args.bench, inputs, output, args.timeout, 1)
    if any(r["status"] != "input-emitted" or r["result"]["build_type"] != "Release" for r in emitted):
        raise ValidationError("timing requires successfully emitted Release inputs")
    # CMake cache belongs to the actual frisk-opt build, not just benchmark.
    opt_cache = next((p / "CMakeCache.txt" for p in args.frisk_opt.parents if (p / "CMakeCache.txt").is_file()), None)
    if opt_cache is None or "CMAKE_BUILD_TYPE:STRING=Release" not in opt_cache.read_text():
        raise ValidationError("timing frisk-opt must be a verifiable Release build")
    lookup = {(r["manifest"]["case"], r["manifest"]["policy"]): output / r["artifact_dir"] / "input.mlir" for r in emitted}
    manifest, rows, stable = timing_manifest(), [], {}
    json_write(output / "manifest.json", manifest)
    for m in manifest:
        directory = output / m["id"]
        sample = run_timing_sample(args.frisk_opt, lookup[m["case"], m["policy"]], directory, args.timeout)
        sample.update(manifest=m, artifact_dir=directory.name, process_dir=directory.name + "/process",
                      process_files_sha256={p.name: sha256(p) for p in (directory / "process").iterdir() if p.is_file()})
        key = (m["case"], m["policy"])
        signature = (sample.get("output_sha256"), sample.get("non_timing_statistics_sha256"))
        if sample["status"] == "success" and key in stable and stable[key] != (None, None) and signature != stable[key]:
            sample.update(status="timing-instability", error="IR/plan/non-timing statistics differ from warmup")
        stable.setdefault(key, signature)
        rows.append(sample)
    return {"schema_version": 1, "kind": "selection-quality-batch", "mode": "timing", "identity": run_identity,
            "jobs": 1, "execution_root": str(output), "timing_scope": "whole frisk-opt CLI: parse+production pipeline+IR output+statistics; no oracle/trace/profile",
            "cpu_isolation": "none; no significance claim from small timing differences", "preflight": emitted, "rows": rows,
            "manifest": manifest, "native_entry_probes": [], "parent_timeout_seconds": args.timeout, "integrity_errors": []}


def merge_reports(paths, output):
    datasets = [load_dataset(path) for path in paths]
    identities = {data["identity"].get("identity_sha256") for data in datasets}
    if len(identities) != 1:
        raise ValidationError("cannot merge different source/script/config/binary identities")
    rows, seen = [], set()
    preflight, probes, components = [], [], []
    for path, data in zip(paths, datasets):
        root = Path(path).resolve().parent
        if data["mode"] == "merged":
            components.extend(data["components"])
        else:
            components.append({key: data[key] for key in ("mode", "execution_root", "jobs", "parent_timeout_seconds")}
                              | ({"oracle_timeout_milliseconds_distinct_from_parent": 120000} if data["mode"] != "timing" else {}))
        for r in data["rows"]:
            if r["manifest"]["id"] in seen:
                raise ValidationError("overlapping row identities; do not cherry-pick repeats")
            seen.add(r["manifest"]["id"])
            archived = dict(r)
            for key in ("artifact_dir", "process_dir"):
                if key in r:
                    archived[key] = str(archived_path(root, r[key]))
            rows.append(archived)
        for key, target in (("preflight", preflight), ("native_entry_probes", probes)):
            for r in data.get(key, []):
                archived = dict(r)
                for location in ("artifact_dir", "process_dir"):
                    if location in r:
                        archived[location] = str(archived_path(root, r[location]))
                target.append(archived)
    merged = {"schema_version": 1, "kind": "selection-quality-batch", "mode": "merged", "identity": datasets[0]["identity"],
              "rows": rows, "preflight": preflight, "native_entry_probes": probes,
              "components": components, "component_modes": [c["mode"] for c in components], "manifest": [r["manifest"] for r in rows],
              "datasets": [{"path": str(Path(p).resolve()), "sha256": sha256(p)} for p in paths],
              "integrity_errors": [e for d in datasets for e in d.get("integrity_errors", [])]}
    publish_dataset(output, merged)
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("correctness", "core", "smoke", "timing", "report"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bench", type=Path, default=ROOT / "build-m6-release/benchmark/layout/selection/frisk_layout_selection_bench")
    parser.add_argument("--frisk-opt", type=Path, default=ROOT / "build-m6-release/bin/frisk-opt")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600, help="whole process budget, distinct from fixed 120 s oracle traversal")
    parser.add_argument("--workers-closed", action="store_true", help="acknowledge all external build/test workers are closed (timing only)")
    parser.add_argument("--dataset", type=Path, action="append", default=[], help="results.json to verify/merge (report only)")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    args.bench, args.frisk_opt = args.bench.resolve(), args.frisk_opt.resolve()
    try:
        if args.jobs < 1 or args.timeout <= 0:
            raise ValidationError("positive jobs/timeout required")
        if args.mode == "timing" and (args.jobs != 1 or not args.workers_closed):
            raise ValidationError("timing requires --jobs 1 --workers-closed")
        if args.mode == "report" and not args.dataset:
            raise ValidationError("report requires --dataset results.json (repeat to merge)")
        claim_output(args.output_dir)
        if args.mode == "report":
            data = merge_reports(args.dataset, args.output_dir)
        else:
            run_identity = identity(args.bench, args.frisk_opt)
            json_write(args.output_dir / "identity.json", run_identity)
            # Shared lock permits independent correctness runs. Exclusive timing
            # lock waits for all cooperating correctness workers to close.
            lock_name = hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]
            with Path(f"/tmp/frisk-selection-quality-{lock_name}.lock").open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX if args.mode == "timing" else fcntl.LOCK_SH)
                data = run_timing(args, args.output_dir, run_identity) if args.mode == "timing" else run_correctness(args, args.output_dir, run_identity)
                check_identity(run_identity)
                publish_dataset(args.output_dir, data)
        print(json.dumps({"output": str(args.output_dir), "rows": len(data["rows"]), "status_counts": data["summary"]["status_counts"], "core_gate": data["core_gate"]}, sort_keys=True))
        return dataset_exit_code(data)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"selection-quality infrastructure error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
