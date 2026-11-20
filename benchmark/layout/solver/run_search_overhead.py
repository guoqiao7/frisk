#!/usr/bin/env python3
"""CPU-only, paired layout-search measurements with archived semantic guards.

Register the supported workload protocol before formal comparison. Correctness/profile observations are
separate from fresh-process profile-off timing; no subprocess is a GPU runner.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "guoqiao/optimize_plan/search_overhead_optimization_artifacts"
OPTIONS = {"exact-combination-limit": 256, "beam-width": 32,
           "max-expanded-states": 65536, "refine-beam-solution": "true",
           "max-refinement-probes": 256, "boundary-frequency": "legacy"}
VARIANTS = ("baseline", "candidate")


class InvalidEvidence(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidEvidence(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    try:
        return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(InvalidEvidence("nonfinite JSON: " + x)))
    except (ValueError, OSError) as error:
        raise InvalidEvidence(str(path) + ": " + str(error)) from error


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".temporary")
    require(not temporary.exists(), "temporary file already exists: " + str(temporary))
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def checked_run(path, stage=STAGE):
    path = Path(path)
    require(path.is_absolute(), "run path must be absolute")
    require(path.resolve().is_relative_to(stage.resolve()) and path.resolve() != stage.resolve(), "run must be a child of this stage")
    require(not any(p.is_symlink() for p in [path, *path.parents] if p.exists()), "symlink run path forbidden")
    return path


def files_digest(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}


def non_timing(value):
    # Only documented timing containers are removed. Cost, budget and logical
    # probe counters remain observable and must be exactly equal.
    if isinstance(value, dict):
        return {k: non_timing(v) for k, v in value.items() if k != "timings"}
    if isinstance(value, list):
        return [non_timing(v) for v in value]
    return value


def source_identity():
    suffixes = {".cpp", ".cc", ".c", ".h", ".hpp", ".td", ".py", ".txt", ".cmake", ".mlir", ".in", ".cu"}
    files = {}
    for folder in ("include", "lib", "tools", "benchmark", "unittests", "test", "cmake"):
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
                files[str(p.relative_to(ROOT))] = sha(p)
    files["CMakeLists.txt"] = sha(ROOT / "CMakeLists.txt")
    return {"files": files, "digest": digest(files)}


def schedule(count, seed):
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0, "invalid pair count")
    orders = [list(VARIANTS if i % 2 == 0 else reversed(VARIANTS)) for i in range(count)]
    random.Random(seed).shuffle(orders)
    return orders


def specifications(stress=False, cost_profile=None, support_results=None):
    rows = [
        {"id": "attention-free", "fixture": "test/Transforms/attention-layout.mlir"},
        {"id": "attention-fixed-weak", "fixture": "test/Transforms/attention-layout-fixed-abi.mlir"},
        {"id": "attention-cpasync", "fixture": "test/Transforms/attention-layout-cp-async-abi.mlir"},
        {"id": "attention-tma", "fixture": "test/Transforms/attention-layout-tma-abi.mlir"},
        {"id": "diamond-32x32", "case_id": "diamond", "shape": [32, 32], "tail": "none", "origin": "original-retained"},
        {"id": "remat-128x16", "case_id": "remat", "shape": [128, 16], "tail": "none", "origin": "original-retained"},
    ]
    if stress:
        rows += [{"id": "mixed-128x32", "case_id": "no-conflict", "shape": [128, 32], "tail": "mixed", "origin": "original-retained"},
                 {"id": "conflict-256x8", "case_id": "conflict", "shape": [256, 8], "tail": "none", "origin": "original-retained"}]
    rows.append({"id": "arithmetic_chain_10", "control": "ten-dependent-addf-tensor4xf32"})
    rows += [{"id": "diamond-32x16", "case_id": "diamond", "shape": [32, 16], "tail": "none", "origin": "fixed-maximum-512-element-support-rule"},
             {"id": "remat-32x16", "case_id": "remat", "shape": [32, 16], "tail": "none", "origin": "fixed-maximum-512-element-support-rule"},
             {"id": "mixed-32x16", "case_id": "no-conflict", "shape": [32, 16], "tail": "mixed", "origin": "fixed-maximum-512-element-support-rule"},
             {"id": "conflict-64x8", "case_id": "conflict", "shape": [64, 8], "tail": "none", "origin": "fixed-maximum-512-element-support-rule"}]
    for row in rows:
        row["timed"] = True
    if cost_profile:
        for original in (rows[1], rows[4], next(x for x in rows if x["id"] == "diamond-32x16")):
            rows.append(dict(original, id=original["id"] + "-v2-guard", timed=False,
                             cost_profile=str(Path(cost_profile).resolve())))
    if support_results:
        observed = read(support_results)
        require(observed.get("status") == "complete" and observed.get("all_variants_semantically_equal") is True, "native support preflight incomplete or inequivalent")
        outcomes = {x["spec"]["id"]: x for x in observed["rows"]}
        require(len(outcomes) == len(observed["rows"]), "duplicate native support preflight case")
        for row in rows:
            if row.get("case_id"):
                require(row["id"] in outcomes, "native preflight missing: " + row["id"])
                outcome = outcomes[row["id"]]
                require(outcome["all_variants_semantically_equal"] is True, "native support variants differ")
                if not outcome["native_success"]:
                    first = outcome["observations"][0]
                    require(first["returncode"] != 0 and not first["ir_present"] and first["stats_status"] == "failed" and first["failure_kind"], "unclassified native failure")
                    failure_reason = read(Path(first["directory"]) / "stats.json")["failure"]["reason"]
                    require(isinstance(failure_reason, str) and failure_reason, "missing native failure reason")
                    row["expected_failure"] = {"returncode": first["returncode"], "stats_status": first["stats_status"], "failure_kind": first["failure_kind"], "failure_reason": failure_reason}
                    row["timed"] = False
    return rows


def render_fixture(spec):
    if spec.get("control") == "ten-dependent-addf-tensor4xf32":
        lines = ["module {", "func.func @chain0(%x: tensor<4xf32>) -> tensor<4xf32> {"]
        for index in range(10):
            previous = "%v" + str(index - 1) if index else "%x"
            lines.append("%v" + str(index) + " = arith.addf " + previous + ", " + previous + " : tensor<4xf32>")
        return "\n".join(lines + ["return %v9 : tensor<4xf32>", "}", "}", ""])
    if spec.get("fixture"):
        return (ROOT / spec["fixture"]).read_text()
    sys.path.insert(0, str(ROOT / "benchmark/layout/execution"))
    from calibration_fixture import fixture
    return fixture(dict(spec, key=spec["id"], split="search-overhead", abi_stratum="weak-vector", dtype="float32", threads=128))


def registration(run, baseline, candidate, cpu, pairs=10, profile_pairs=3, seed=20261101, stress=False, cost_profile=None, candidate_off_option=None, support_results=None):
    run = checked_run(run)
    require(not run.exists(), "run already exists")
    require(cpu in os.sched_getaffinity(0), "requested CPU outside allowed affinity")
    require(pairs > 0 and profile_pairs > 0, "pair counts must be positive")
    require(candidate_off_option in (None, "memoize-costs=false"), "unsupported candidate-only ablation option")
    baseline = Path(baseline).resolve(); candidate = Path(candidate).resolve()
    require(baseline.is_file() and os.access(baseline, os.X_OK), "baseline executable absent")
    if cost_profile:
        require(Path(cost_profile).is_file(), "cost profile absent")
    run.mkdir(parents=True)
    rows = specifications(stress, cost_profile, support_results)
    for row in rows:
        p = run / "inputs" / (row["id"] + ".mlir")
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(render_fixture(row))
        row["input"] = str(p); row["input_sha256"] = sha(p)
        if row.get("cost_profile"):
            row["cost_profile_sha256"] = sha(row["cost_profile"])
        row["orders"] = schedule(pairs, seed + len(row["id"]))
        row["profile_orders"] = schedule(profile_pairs, seed + len(row["id"]) + 1000)
        row["ablation_orders"] = [["candidate-off" if v == "baseline" else v for v in order] for order in row["profile_orders"]]
    protocol = {"schema_version": 1, "protocol_revision": 2, "kind": "cpu-search-overhead", "cpu": cpu, "pairs": pairs,
                "profile_pairs": profile_pairs, "warmups_per_variant_per_case": 1, "seed": seed,
                "options": OPTIONS, "cases": rows,
                "binaries": {"baseline": str(baseline), "candidate": str(candidate)},
                "baseline_sha256": sha(baseline), "registered_driver_sha256": sha(Path(__file__)),
                "measurement": "serial fresh-process, profile-off main; inclusive profiles separate",
                "cache_scope": "fresh process/invocation; OS/page warmups do not establish a persistent warm cache",
                "cpu_policy": "taskset pinned to one allowed CPU; frequency unmodified; no exclusive reservation",
                "candidate_off_option": candidate_off_option,
                "support_preflight": {"path": str(Path(support_results).resolve()), "sha256": sha(support_results)} if support_results else None,
                "support_policy": "retain every original case; expected failures are untimed N/A guards; fixed <=512-element analogues selected by support, never speed",
                "no_gpu_execution": True, "no_performance_claim_before_completed_validation": True}
    write(run / "registration.json", {"protocol": protocol, "protocol_sha256": digest(protocol), "created_ns": time.time_ns()})
    return protocol


def check_registration(run):
    reg = read(run / "registration.json"); p = reg["protocol"]
    require(reg["protocol_sha256"] == digest(p), "registration checksum mismatch")
    require(p["registered_driver_sha256"] == sha(Path(__file__)), "registered driver changed")
    require(p["options"] == OPTIONS and p["kind"] == "cpu-search-overhead", "unsupported protocol/options")
    require(sha(p["binaries"]["baseline"]) == p["baseline_sha256"], "baseline binary changed")
    require(p["cpu"] in os.sched_getaffinity(0), "registered CPU unavailable")
    if p.get("support_preflight"):
        require(sha(p["support_preflight"]["path"]) == p["support_preflight"]["sha256"], "native support preflight changed")
    ids = [x["id"] for x in p["cases"]]
    require(ids and len(set(ids)) == len(ids), "duplicate/empty case ids")
    for row in p["cases"]:
        require(not row.get("expected_failure") or row["timed"] is False, "failure case cannot enter performance samples")
        require(sha(row["input"]) == row["input_sha256"], "input changed: " + row["id"])
        if row.get("cost_profile"):
            require(sha(row["cost_profile"]) == row["cost_profile_sha256"], "cost profile changed")
    return p


def identity(protocol):
    return {"binaries": {k: sha(v) for k, v in protocol["binaries"].items()},
            "source": source_identity(), "driver_sha256": sha(Path(__file__)),
            "hostname": socket.gethostname(), "uname": list(os.uname()),
            "python": {"executable": sys.executable, "sha256": sha(sys.executable), "version": sys.version},
            "taskset_path": "/usr/bin/taskset", "taskset_sha256": sha("/usr/bin/taskset")}


def check_identity(protocol, frozen):
    require(identity(protocol) == frozen, "source/binary/driver/machine identity changed")


def execute(command, directory, timeout=600):
    directory = Path(directory)
    require(not directory.exists(), "process directory already exists")
    directory.mkdir(parents=True)
    result = {"command": command, "timeout_seconds": timeout, "timeout": False,
              "status": "exited", "returncode": None, "created_ns": time.time_ns(),
              "load_before": list(os.getloadavg()),
              "resource_scope": "per-child wait4 rusage; Linux ru_maxrss KiB; parent whole-CLI monotonic wall includes 1ms polling"}
    start = time.perf_counter_ns()
    with (directory / "stdout.txt").open("wb") as out, (directory / "stderr.txt").open("wb") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True)
        while True:
            pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid:
                break
            if (time.perf_counter_ns() - start) / 1e9 > timeout:
                result.update(timeout=True, status="process-timeout")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, usage = os.wait4(process.pid, 0)
                break
            time.sleep(.001)
        process.returncode = os.waitstatus_to_exitcode(status)
        result.update(returncode=process.returncode, wall_ns=time.perf_counter_ns() - start,
                      user_ns=round(usage.ru_utime * 1e9), system_ns=round(usage.ru_stime * 1e9),
                      peak_rss_kib=usage.ru_maxrss, load_after=list(os.getloadavg()))
    write(directory / "process.json", result)
    return result


def invocation(protocol, row, variant, directory, frozen, profile=False, trace=False, timeout=600):
    check_registration(Path(row["input"]).parents[1]); check_identity(protocol, frozen)
    directory = Path(directory)
    options = dict(OPTIONS, **{"stats-file": str(directory / "stats.json")})
    if profile:
        options["profile-file"] = str(directory / "profile.json")
    if trace:
        options["trace-file"] = str(directory / "trace.json")
    if row.get("cost_profile"):
        options["cost-profile"] = row["cost_profile"]
    if variant == "candidate-off":
        require(protocol.get("candidate_off_option") == "memoize-costs=false", "cache-off option not registered")
        options["memoize-costs"] = "false"
    flags = " ".join(k + "=" + str(v) for k, v in options.items())
    binary_variant = "candidate" if variant == "candidate-off" else variant
    command = ["/usr/bin/taskset", "-c", str(protocol["cpu"]), protocol["binaries"][binary_variant],
               row["input"], "--frisk-layout-pipeline=" + flags, "-verify-each", "-o", str(directory / "output.mlir")]
    process = execute(command, directory, timeout)
    check_registration(Path(row["input"]).parents[1]); check_identity(protocol, frozen)
    require(not process["timeout"], "CLI timeout: " + str(directory))
    expected = row.get("expected_failure")
    require(process["returncode"] == (expected["returncode"] if expected else 0), "CLI failed: " + str(directory))
    stats = read(directory / "stats.json")
    output = directory / "output.mlir"
    if expected:
        require(stats.get("status") == expected["stats_status"] and stats.get("failure", {}).get("kind") == expected["failure_kind"] and stats.get("failure", {}).get("reason") == expected["failure_reason"], "failure outcome differs from preregistration")
        require(not output.exists(), "failed pipeline published an output IR")
    else:
        require(stats.get("status") == "committed" and output.is_file(), "uncommitted pipeline result")
    result = {"case": row["id"], "variant": variant, "directory": str(directory), "process": process,
              "ir_sha256": None if expected else sha(output), "non_timing_stats": non_timing(stats),
              "outcome": "expected-failure" if expected else "success", "expected_failure": expected,
              "statistics": stats, "profile": profile, "trace": trace}
    if profile:
        result["profile_record"] = read(directory / "profile.json")
    if trace:
        record = read(directory / "trace.json")
        require(record.get("truncated") is False, "incomplete decision trace")
        if not expected:
            require(record.get("costExplanationComplete") is True, "incomplete cost explanation")
        result["trace_sha256_canonical"] = digest(record)
    write(directory / "result.json", result)
    return result


def same_semantics(left, right):
    for key in ("case", "ir_sha256", "non_timing_stats", "outcome", "expected_failure"):
        require(left.get(key) == right.get(key), "semantic mismatch: " + key)
    if left.get("trace") and right.get("trace"):
        require(left["trace_sha256_canonical"] == right["trace_sha256_canonical"], "decision trace mismatch")


def save_phase(run, name, value):
    value["artifact_hashes"] = files_digest(run / name)
    write(run / (name + ".json"), value)


def verify_phase(run, name):
    value = read(run / (name + ".json"))
    require(value.get("status") == "complete", "phase incomplete: " + name)
    require(files_digest(run / name) == value.get("artifact_hashes"), "phase artifact changed: " + name)
    require(value.get("registration_sha256") == sha(run / "registration.json"), "phase registration identity mismatch")
    require(value.get("identity_sha256") == sha(run / "measurement-identity.json"), "phase measurement identity mismatch")
    spec_by_id = {x["id"]: x for x in read(run / "registration.json")["protocol"]["cases"]}
    for row in value["rows"]:
        if name == "correctness":
            observations = row["observations"]
            variants = (*VARIANTS, "candidate-off") if read(run / "registration.json")["protocol"].get("candidate_off_option") else VARIANTS
            require(row.get("passed") is True and len(observations) == 2 * len(variants) and row["canonical"] == observations[0], "invalid correctness witness")
            require({(x["variant"], x["profile"]) for x in observations} == {(v, on) for v in variants for on in (False, True)}, "correctness observations missing")
        else:
            observations = row["warmups"] + [x for pair in row["pairs"] for x in pair["observations"].values()]
        for observed in observations:
            require(observed["trace"] is (name == "correctness"), "trace evidence mixed into measurement phase")
            if name != "correctness":
                require(observed["profile"] is (name in ("profile", "ablation")), "profile-on/off phases mixed")
            directory = Path(observed["directory"])
            require(directory.resolve().is_relative_to((run / name).resolve()), "observation directory escapes phase")
            require(observed == read(directory / "result.json"), "summary differs from archived invocation")
            require(observed["process"] == read(directory / "process.json"), "process resource summary mismatch")
            require(observed["statistics"] == read(directory / "stats.json") and observed["non_timing_stats"] == non_timing(observed["statistics"]), "statistics summary mismatch")
            expected = spec_by_id[observed["case"]].get("expected_failure")
            require(observed.get("expected_failure") == expected, "registered support outcome changed")
            if expected:
                require(observed.get("outcome") == "expected-failure" and observed["ir_sha256"] is None and not (directory / "output.mlir").exists(), "failed pipeline IR invariant violated")
                require(observed["process"]["returncode"] == expected["returncode"] and observed["statistics"]["status"] == expected["stats_status"] and observed["statistics"]["failure"]["kind"] == expected["failure_kind"] and observed["statistics"]["failure"]["reason"] == expected["failure_reason"], "failure status summary mismatch")
            else:
                require(observed.get("outcome") == "success" and observed["ir_sha256"] == sha(directory / "output.mlir"), "IR summary mismatch")
            if observed["profile"]:
                require(observed["profile_record"] == read(directory / "profile.json"), "profile summary mismatch")
            if observed["trace"]:
                require(observed["trace_sha256_canonical"] == digest(read(directory / "trace.json")), "trace summary mismatch")
        if name == "correctness":
            for observed in observations[1:]:
                same_semantics(row["canonical"], observed)
    return value


def correctness(run, timeout=600):
    p = check_registration(run)
    require(not (run / "correctness").exists() and not (run / "measurement-identity.json").exists(), "correctness already started")
    frozen = identity(p); write(run / "measurement-identity.json", frozen)
    rows = []
    try:
        for spec in p["cases"]:
            compared = []
            variants = (*VARIANTS, "candidate-off") if p.get("candidate_off_option") else VARIANTS
            for variant in variants:
                for on in (False, True):
                    d = run / "correctness" / spec["id"] / (variant + ("-profile-on" if on else "-profile-off"))
                    result = invocation(p, spec, variant, d, frozen, on, True, timeout)
                    if compared:
                        same_semantics(compared[0], result)
                    compared.append(result)
            rows.append({"case": spec["id"], "canonical": compared[0], "observations": compared, "passed": True})
        save_phase(run, "correctness", {"status": "complete", "rows": rows, "registration_sha256": sha(run / "registration.json"), "identity_sha256": sha(run / "measurement-identity.json")})
    except Exception as error:
        save_phase(run, "correctness", {"status": "failed", "rows": rows, "failure": str(error), "no_performance_claim": True})
        raise


def measure(run, profile=False, timeout=600, ablation=False):
    p = check_registration(run); frozen = read(run / "measurement-identity.json"); check_identity(p, frozen)
    gate = verify_phase(run, "correctness"); canonical_rows = {x["case"]: x["canonical"] for x in gate["rows"]}
    require([x["case"] for x in gate["rows"]] == [x["id"] for x in p["cases"]], "correctness matrix mismatch")
    name = "ablation" if ablation else "profile" if profile else "timing"
    variants = ("candidate-off", "candidate") if ablation else VARIANTS
    if ablation:
        require(profile and p.get("candidate_off_option"), "ablation needs registered candidate-off option and profiles")
    require(not (run / name).exists(), "measurement phase already started")
    rows = []
    try:
        for spec in p["cases"]:
            if not spec["timed"]:
                continue
            warmups = []
            for variant in variants:
                result = invocation(p, spec, variant, run / name / spec["id"] / ("warmup-" + variant), frozen, profile, False, timeout)
                same_semantics(canonical_rows[spec["id"]], result); warmups.append(result)
            pairs = []
            orders = spec["ablation_orders"] if ablation else spec["profile_orders"] if profile else spec["orders"]
            for index, order in enumerate(orders):
                pair = {}
                for variant in order:
                    result = invocation(p, spec, variant, run / name / spec["id"] / (str(index) + "-" + variant), frozen, profile, False, timeout)
                    same_semantics(canonical_rows[spec["id"]], result); pair[variant] = result
                pairs.append({"index": index, "order": order, "observations": pair})
            rows.append({"case": spec["id"], "warmups": warmups, "pairs": pairs})
        save_phase(run, name, {"status": "complete", "rows": rows, "profile_enabled": profile, "registration_sha256": sha(run / "registration.json"), "identity_sha256": sha(run / "measurement-identity.json")})
    except Exception as error:
        save_phase(run, name, {"status": "failed", "rows": rows, "failure": str(error), "no_performance_claim": True})
        raise


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def paired_summary(pairs, metric="wall_ns", seed=20261101, variants=VARIANTS):
    require(pairs, "empty paired samples")
    a = [x["observations"][variants[0]]["process"][metric] for x in pairs]
    b = [x["observations"][variants[1]]["process"][metric] for x in pairs]
    require(all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in a+b), "nonpositive/nonfinite measurement")
    logs = [math.log(x/y) for x, y in zip(a, b)]
    rng = random.Random(seed)
    boots = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(5000))
    return {"pairs": len(pairs), "baseline_median": statistics.median(a), "candidate_median": statistics.median(b),
            "baseline_p95": percentile(a, .95), "candidate_p95": percentile(b, .95),
            "paired_geomean_speedup": math.exp(statistics.mean(logs)), "paired_bootstrap_95ci": [boots[124], boots[4874]],
            "interval_scope": "paired subprocess blocks within this fixed case; not a workload-generalization interval"}


def report(run):
    p = check_registration(run); frozen = read(run / "measurement-identity.json"); check_identity(p, frozen)
    correct = verify_phase(run, "correctness"); timing = verify_phase(run, "timing")
    require([x["case"] for x in correct["rows"]] == [x["id"] for x in p["cases"]], "correctness matrix mismatch")
    expected = [x for x in p["cases"] if x["timed"]]
    require([x["case"] for x in timing["rows"]] == [x["id"] for x in expected], "timing matrix mismatch")
    rows = []
    for spec, row in zip(expected, timing["rows"]):
        require(len(row["pairs"]) == p["pairs"] and [x["order"] for x in row["pairs"]] == spec["orders"], "paired schedule mismatch")
        require(len(row["warmups"]) == 2 and {x["variant"] for x in row["warmups"]} == set(VARIANTS), "warmup mismatch")
        summary = {"case": row["case"], "wall_ns": paired_summary(row["pairs"], seed=p["seed"])}
        for variant in VARIANTS:
            samples = [pair["observations"][variant] for pair in row["pairs"]]
            summary[variant] = {"peak_rss_kib_median": statistics.median(x["process"]["peak_rss_kib"] for x in samples),
                                "peak_rss_kib_max": max(x["process"]["peak_rss_kib"] for x in samples),
                                "cpu_ns_median": statistics.median(x["process"]["user_ns"]+x["process"]["system_ns"] for x in samples),
                                "pipeline_ns_median": statistics.median(x["statistics"]["timings"]["layout_pipeline_ns"] for x in samples)}
        rows.append(summary)
    result = {"schema_version": 1, "status": "complete", "cases": rows,
              "descriptive_equal_case_geomean_speedup": math.exp(statistics.mean(math.log(x["wall_ns"]["paired_geomean_speedup"]) for x in rows)),
              "no_gpu_performance_claim": True, "cold_scope": p["cache_scope"],
              "source_identity_sha256": sha(run / "measurement-identity.json"), "registration_sha256": sha(run / "registration.json"),
              "correctness_sha256": sha(run / "correctness.json"), "timing_sha256": sha(run / "timing.json"),
              "profile_phase_complete": (run / "profile.json").exists(),
              "support_outcomes": [{"case": row["id"], "timed": row["timed"], "status": "expected-failure" if row.get("expected_failure") else "supported", "expected_failure": row.get("expected_failure"), "performance": "N/A: unsupported natural-domain cost evaluation" if row.get("expected_failure") else "measured" if row["timed"] else "semantic guard only"} for row in p["cases"]]}
    if result["profile_phase_complete"]:
        profile = verify_phase(run, "profile")
        require([x["case"] for x in profile["rows"]] == [x["id"] for x in expected], "profile matrix mismatch")
        require(all(len(x["pairs"]) == p["profile_pairs"] for x in profile["rows"]), "profile pair count mismatch")
        require(all([pair["order"] for pair in row["pairs"]] == spec["profile_orders"] for row, spec in zip(profile["rows"], expected)), "profile paired schedule mismatch")
        result["profile_sha256"] = sha(run / "profile.json")
    if (run / "ablation.json").exists():
        ablation = verify_phase(run, "ablation")
        require([x["case"] for x in ablation["rows"]] == [x["id"] for x in expected], "ablation matrix mismatch")
        require(all(len(row["pairs"]) == p["profile_pairs"] and [pair["order"] for pair in row["pairs"]] == spec["ablation_orders"] for row, spec in zip(ablation["rows"], expected)), "ablation pair schedule mismatch")
        result["ablation_sha256"] = sha(run / "ablation.json")
        result["cache_ablation_profile_on"] = [{"case": row["case"], "wall_ns": paired_summary(row["pairs"], seed=p["seed"], variants=("candidate-off", "candidate")), "scope": "candidate-only profile-on diagnostic, separate from main profile-off comparison"} for row in ablation["rows"]]
    write(run / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "correctness", "timing", "profile", "ablation", "report"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline"); parser.add_argument("--candidate")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--pairs", type=int, default=10); parser.add_argument("--profile-pairs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261101); parser.add_argument("--stress", action="store_true")
    parser.add_argument("--cost-profile"); parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--candidate-off-option", choices=("memoize-costs=false",))
    parser.add_argument("--support-results", help="archived development preflight results; never a speed-based filter")
    args = parser.parse_args(argv)
    try:
        run = checked_run(args.run)
        require(args.timeout > 0, "timeout must be positive")
        if args.phase == "register":
            require(args.baseline and args.candidate, "register needs both binary paths")
            value = registration(run, args.baseline, args.candidate, args.cpu, args.pairs, args.profile_pairs, args.seed, args.stress, args.cost_profile, args.candidate_off_option, args.support_results)
            print(canonical({"registered": len(value["cases"]), "protocol_sha256": digest(value)}))
        elif args.phase == "correctness":
            correctness(run, args.timeout)
        elif args.phase in ("timing", "profile", "ablation"):
            measure(run, args.phase != "timing", args.timeout, args.phase == "ablation")
        else:
            print(canonical(report(run)))
    except (InvalidEvidence, OSError, KeyError, TypeError) as error:
        if "run" in locals() and run.is_dir():
            write(run / ("failed-" + args.phase + "-" + str(time.time_ns()) + ".json"),
                  {"status": "failed", "phase": args.phase, "failure": str(error), "no_performance_claim": True})
        print("invalid search-overhead evidence: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
