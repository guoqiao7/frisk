#!/usr/bin/env python3
"""Validate and gate Frisk layout benchmark results (schema version 1)."""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


SOLVER_CASES = {f"connected_{n}": n for n in (10, 50, 100, 250)}
CONVERSION_CASES = {"identity_f32": 32, "shuffle32_f32": 32, "shared64_f32": 64}
SEEDS = [1, 7, 42, 20260816]
SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
COMPARABLE_ENVIRONMENT_KEYS = (
    "gpu_uuid", "gpu_name", "compute_capability", "driver", "toolkit", "clock_policy",
    "compile_flags",
)
STATISTICS = {
    "variables", "constraints", "components", "preparedCandidates", "prunedCandidates",
    "peakVariableDomain", "peakComponentCombinations", "componentProductOverflow",
    "peakFrontierStates", "expandedStates", "conversionUseEdges", "uniqueConversions",
    "rematerializationUseEdges", "uniqueRematerializations", "conversionBytes",
    "estimatedBankConflicts", "estimatedMemoryTransactions",
}
UINT64_MAX = (1 << 64) - 1


class InvalidResult(ValueError):
    pass


def strict_loads(text, source):
    try:
        return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
            InvalidResult(f"{source}: non-finite JSON number {value}")))
    except (json.JSONDecodeError, InvalidResult) as exc:
        raise InvalidResult(f"{source}: malformed JSON: {exc}") from exc


def load_file(path):
    try:
        return strict_loads(Path(path).read_text(encoding="utf-8"), str(path))
    except OSError as exc:
        raise InvalidResult(f"cannot read {path}: {exc}") from exc


def run_tool(path):
    try:
        completed = subprocess.run([path], text=True, capture_output=True, check=False)
    except OSError as exc:
        raise InvalidResult(f"cannot execute {path}: {exc}") from exc
    if completed.returncode:
        raise InvalidResult(f"tool {path} exited {completed.returncode}: {completed.stderr.strip()}")
    return strict_loads(completed.stdout, path)


def require(condition, message):
    if not condition:
        raise InvalidResult(message)


def require_object(value, label):
    require(isinstance(value, dict), f"{label} must be an object")
    return value


def require_int(value, label, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"{label} must be an integer >= {minimum}")


def require_sha(value, label):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None,
            f"{label} must be a 64-digit SHA-256")


def validate_samples(value, expected, label):
    require(isinstance(value, list) and len(value) == expected,
            f"{label} must contain exactly {expected} samples")
    for sample in value:
        require(isinstance(sample, (int, float)) and not isinstance(sample, bool)
                and math.isfinite(sample) and sample > 0,
                f"{label} samples must be positive finite numbers")
    return value


def unique_cases(result, required, label):
    cases = result.get("cases")
    require(isinstance(cases, list), f"{label}.cases must be an array")
    require(all(isinstance(case, dict) for case in cases), f"{label} cases must be objects")
    ids = [case.get("id") for case in cases]
    require(all(isinstance(case_id, str) for case_id in ids),
            f"{label} case ids must be strings")
    require(len(ids) == len(set(ids)), f"{label} contains duplicate case ids")
    require(set(ids) == set(required), f"{label} case set mismatch")
    return {case["id"]: case for case in cases}


def validate_solver(result):
    require_object(result, "solver")
    require(result.get("schema_version") == 1 and result.get("kind") == "solver",
            "invalid solver schema/kind")
    require(result.get("status") == "success", "solver status is not success")
    env = require_object(result.get("environment"), "solver.environment")
    for key in ("cpu", "build_type", "compiler"):
        require(isinstance(env.get(key), str) and env[key], f"missing solver environment {key}")
    require_sha(env.get("source_digest"), "solver.environment.source_digest")
    require(result.get("options") == {"exact_combination_limit": 256, "beam_width": 32,
                                      "max_expanded_states": 65536},
            "solver production options mismatch")
    cases = unique_cases(result, SOLVER_CASES, "solver")
    summaries = []
    for case_id, operations in SOLVER_CASES.items():
        case = cases[case_id]
        require(case.get("operations") == operations and case.get("status") == "success",
                f"{case_id}: operation count/status mismatch")
        require(case.get("warmup") == 5 and case.get("iterations") == 30,
                f"{case_id}: measurement count mismatch")
        require_sha(case.get("solution_hash"), f"{case_id}.solution_hash")
        require(case.get("deterministic") is True, f"{case_id}: solution is not deterministic")
        samples = require_object(case.get("samples"), f"{case_id}.samples")
        for key in ("collection_ns", "preparation_ns", "solve_ns", "layout_pipeline_ns"):
            validate_samples(samples.get(key), 30, f"{case_id}.{key}")
        stats = require_object(case.get("statistics"), f"{case_id}.statistics")
        require(STATISTICS <= set(stats), f"{case_id}: missing statistics counters")
        for key in STATISTICS - {"componentProductOverflow"}:
            require_int(stats[key], f"{case_id}.statistics.{key}")
        require(isinstance(stats["componentProductOverflow"], bool),
                f"{case_id}: componentProductOverflow must be boolean")
        components = case.get("components")
        require(isinstance(components, list) and components, f"{case_id}: components are empty")
        for component in components:
            component = require_object(component, f"{case_id}.component")
            require(isinstance(component.get("root"), str) and component["root"],
                    f"{case_id}: component root must be a nonempty string")
            overflow = component.get("productOverflow")
            combinations = component.get("combinations")
            require(isinstance(overflow, bool), f"{case_id}: invalid productOverflow")
            require((overflow and combinations is None) or
                    (not overflow and isinstance(combinations, int) and combinations >= 0),
                    f"{case_id}: combinations/overflow mismatch")
            domains = component.get("domains")
            require(isinstance(domains, list) and domains and
                    all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in domains),
                    f"{case_id}: invalid domains")
            domain_product = math.prod(domains)
            expected_overflow = domain_product > UINT64_MAX
            require(overflow == expected_overflow,
                    f"{case_id}: productOverflow does not match domain product")
            require((expected_overflow and combinations is None) or
                    (not expected_overflow and combinations == domain_product),
                    f"{case_id}: combinations does not match domain product")
            expected_mode = "beam" if expected_overflow or domain_product > 256 else "exact"
            require(component.get("mode") == expected_mode, f"{case_id}: search mode mismatch")
            require(component.get("beamWidth") == 32, f"{case_id}: beam width mismatch")
            for key in ("expanded", "peakFrontier", "mrvProbes", "expandedNodes"):
                require_int(component.get(key), f"{case_id}.{key}")
            require(component["expanded"] <= 65536 and component["expandedNodes"] <= 65536,
                    f"{case_id}: expansion budget exceeded")
            if expected_mode == "beam":
                require(component["peakFrontier"] <= 32, f"{case_id}: beam frontier exceeded")
            require(isinstance(component.get("exhaustive"), bool), f"{case_id}: exhaustive invalid")
        summary = {"id": case_id, "solution_hash": case["solution_hash"]}
        for phase in ("collection", "preparation", "solve", "layout_pipeline"):
            values = samples[f"{phase}_ns"]
            summary[f"{phase}_median_ns"] = statistics.median(values)
            summary[f"{phase}_p95_ns"] = percentile(values, .95)
        summaries.append(summary)
    return {"environment": env, "cases": summaries, "raw": result}


def validate_environment(env):
    env = require_object(env, "conversion.environment")
    expected_strings = ("gpu_uuid", "gpu_name", "driver", "toolkit", "background_load")
    for key in expected_strings:
        require(isinstance(env.get(key), str) and env[key], f"missing conversion environment {key}")
    require(env.get("compute_capability") == "9.0", "compute capability must be 9.0")
    require(env.get("clock_policy") == "unmodified", "clock policy must be unmodified")
    require_int(env.get("device"), "conversion.environment.device")
    require(isinstance(env.get("compile_flags"), list) and
            all(isinstance(x, str) for x in env["compile_flags"]), "invalid environment compile_flags")
    return env


def validate_provenance(provenance, label, expected_flags):
    provenance = require_object(provenance, label)
    for key in ("frisk_input_sha256", "adapter_ir_sha256", "lowered_ir_sha256",
                "ptx_sha256", "cubin_sha256", "source_digest"):
        require_sha(provenance.get(key), f"{label}.{key}")
    commands = provenance.get("commands")
    require(isinstance(commands, list) and commands and
            all(isinstance(x, str) and x.strip() for x in commands), f"{label}.commands invalid")
    flags = provenance.get("compile_flags")
    require(isinstance(flags, list) and all(isinstance(x, str) for x in flags),
            f"{label}.compile_flags invalid")
    require(flags == expected_flags, f"{label}.compile_flags disagree with environment")
    return provenance


def validate_conversion(result):
    require_object(result, "conversion")
    require(result.get("schema_version") == 1 and result.get("kind") == "conversion",
            "invalid conversion schema/kind")
    require(result.get("status") == "success", "conversion status is not success")
    env = validate_environment(result.get("environment"))
    require(result.get("manifest") == list(CONVERSION_CASES), "conversion manifest mismatch")
    cases = unique_cases(result, CONVERSION_CASES, "conversion")
    validated = {}
    for case_id, threads in CONVERSION_CASES.items():
        case = cases[case_id]
        require(case.get("status") == "success" and case.get("threads") == threads
                and case.get("dtype") == "f32", f"{case_id}: case metadata mismatch")
        correctness = require_object(case.get("correctness"), f"{case_id}.correctness")
        require(correctness == {"status": "pass", "seeds": SEEDS, "boundary_values": True,
                                "replication": True}, f"{case_id}: correctness evidence incomplete")
        require(case.get("sanitizers") == {"memcheck": "pass", "racecheck": "pass",
                                           "synccheck": "pass"},
                f"{case_id}: sanitizer evidence incomplete")
        require(case.get("warmup") == 100 and case.get("iterations") == 1000
                and case.get("measurement_order") == "alternating",
                f"{case_id}: measurement protocol mismatch")
        reference = require_object(case.get("reference"), f"{case_id}.reference")
        inferred = require_object(case.get("inferred"), f"{case_id}.inferred")
        require(reference.get("origin") == "independent-coordinate-reference",
                f"{case_id}: reference origin confusion")
        require(inferred.get("origin") == "frisk-conversion-adapter",
                f"{case_id}: inferred origin confusion")
        require_sha(reference.get("source_sha256"), f"{case_id}.reference.source_sha256")
        validate_samples(reference.get("samples_ns"), 1000, f"{case_id}.reference.samples_ns")
        validate_samples(inferred.get("samples_ns"), 1000, f"{case_id}.inferred.samples_ns")
        compilation = require_object(case.get("compilation"), f"{case_id}.compilation")
        require(compilation.get("cache") == "disabled", f"{case_id}: compile cache must be disabled")
        compile_samples = compilation.get("samples")
        require(isinstance(compile_samples, list) and compile_samples,
                f"{case_id}: compilation samples are empty")
        ratios = []
        for sample in compile_samples:
            sample = require_object(sample, f"{case_id}.compilation sample")
            values = [sample.get("layout_pipeline_ns"), sample.get("full_compile_ns")]
            require(all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        and math.isfinite(x) and x > 0 for x in values),
                    f"{case_id}: invalid compilation timing")
            ratios.append(values[0] / values[1])
        validate_provenance(case.get("provenance"), f"{case_id}.provenance",
                            env["compile_flags"])
        require(case.get("artifact_checks") == {"unrealized_casts": 0,
                                                "unresolved_layouts": 0,
                                                "placeholder_calls": 0},
                f"{case_id}: unresolved artifacts remain")
        validated[case_id] = {"raw": case, "compile_ratios": ratios}
    return env, validated


def validate_baseline(baseline):
    require_object(baseline, "baseline")
    require(baseline.get("schema_version") == 1 and baseline.get("kind") == "reference-baseline",
            "invalid baseline schema/kind")
    require(baseline.get("review_status") in ("pending", "accepted"),
            "invalid baseline review status")
    env = validate_environment(baseline.get("environment"))
    require(baseline.get("manifest") == list(CONVERSION_CASES), "baseline manifest mismatch")
    cases = unique_cases(baseline, CONVERSION_CASES, "baseline")
    for case_id, case in cases.items():
        require(case.get("origin") == "independent-coordinate-reference",
                f"{case_id}: baseline origin confusion")
        validate_samples(case.get("samples_ns"), 1000, f"baseline.{case_id}.samples_ns")
        require_sha(case.get("source_sha256"), f"baseline.{case_id}.source_sha256")
        validate_provenance(case.get("provenance"), f"baseline.{case_id}.provenance",
                            env["compile_flags"])
    return env, cases


def comparable_environment(environment):
    return {key: environment[key] for key in COMPARABLE_ENVIRONMENT_KEYS}


def percentile(values, quantile):
    """Return the nearest-rank percentile for a nonempty validated sample list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def pending_baseline(result):
    return {"schema_version": 1, "kind": "reference-baseline", "review_status": "pending",
            "environment": result["environment"], "manifest": result["manifest"],
            "cases": [{"id": case["id"], "origin": case["reference"]["origin"],
                       "samples_ns": case["reference"]["samples_ns"],
                       "source_sha256": case["reference"]["source_sha256"],
                       "provenance": case["provenance"]} for case in result["cases"]]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver")
    parser.add_argument("--solver-json")
    parser.add_argument("--conversion")
    parser.add_argument("--conversion-json")
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--record-reference", action="store_true")
    parser.add_argument("--output-baseline")
    parser.add_argument("--output")
    args = parser.parse_args()
    for executable, fixture, label in ((args.solver, args.solver_json, "solver"),
                                       (args.conversion, args.conversion_json, "conversion")):
        if executable and fixture:
            parser.error(f"choose only one --{label} or --{label}-json")
    if args.record_reference and not args.output_baseline:
        parser.error("--record-reference requires --output-baseline")
    if args.cpu_only and (args.conversion or args.conversion_json or args.record_reference):
        parser.error("--cpu-only cannot be combined with conversion inputs")
    return args


def paths_alias(first, second):
    if not first or not second:
        return False
    first_path, second_path = Path(first), Path(second)
    if first_path.resolve() == second_path.resolve():
        return True
    try:
        return os.path.samefile(first_path, second_path)
    except OSError:
        return False


def output_path_error(args):
    """Return a preflight error before any protected destination can be written."""
    if args.output and args.baseline and paths_alias(args.output, args.baseline):
        return "aggregate output must not alias the input baseline"
    if args.record_reference:
        if Path(args.output_baseline).exists():
            return "output baseline already exists; pending evidence never overwrites files"
        if args.baseline and paths_alias(args.output_baseline, args.baseline):
            return "output baseline must not alias the input baseline"
        if args.output and paths_alias(args.output_baseline, args.output):
            return "aggregate output must not alias the output baseline"
    return None


def main():
    args = parse_args()
    unsafe_path = output_path_error(args)
    if unsafe_path:
        print(f"error: unsafe output path alias: {unsafe_path}", file=sys.stderr)
        return 2
    report = {"schema_version": 1, "environment": {}, "solver": {}, "conversions": {},
              "supported_kernels": {}, "gates": {}}
    errors = []
    solver_data = conversion_data = None
    try:
        if args.solver or args.solver_json:
            solver_data = run_tool(args.solver) if args.solver else load_file(args.solver_json)
            report["solver"] = validate_solver(solver_data)
            report["gates"]["solver"] = {"status": "PASS"}
        else:
            report["gates"]["solver"] = {"status": "NOT_EVALUATED"}

        if args.cpu_only:
            report["gates"]["runtime"] = {"status": "NOT_EVALUATED"}
            report["gates"]["compile"] = {"status": "NOT_EVALUATED"}
            ok = report["gates"]["solver"]["status"] == "PASS"
            report["gates"]["overall"] = {"status": "PARTIAL_PASS" if ok else "INCOMPLETE"}
        elif args.conversion or args.conversion_json:
            conversion_data = (run_tool(args.conversion) if args.conversion
                               else load_file(args.conversion_json))
            environment, cases = validate_conversion(conversion_data)
            report["environment"] = environment
            report["supported_kernels"] = {case_id: True for case_id in CONVERSION_CASES}
            summaries = []
            baseline_cases = None
            baseline_status = "INCOMPLETE"
            if args.baseline:
                baseline = load_file(args.baseline)
                baseline_env, baseline_cases = validate_baseline(baseline)
                if (comparable_environment(baseline_env) != comparable_environment(environment)
                        or baseline.get("manifest") != conversion_data["manifest"]):
                    baseline_status = "NOT_COMPARABLE"
                    baseline_cases = None
                elif baseline["review_status"] != "accepted":
                    baseline_status = "INCOMPLETE"
                    baseline_cases = None
                else:
                    baseline_status = "PASS"
                    for case_id in CONVERSION_CASES:
                        if (baseline_cases[case_id]["source_sha256"] !=
                                cases[case_id]["raw"]["reference"]["source_sha256"]):
                            baseline_status = "NOT_COMPARABLE"
                            baseline_cases = None
                            break
            for case_id in CONVERSION_CASES:
                case = cases[case_id]
                compile_ratios = case["compile_ratios"]
                summary = {"id": case_id,
                           "same_run_reference_median_ns": statistics.median(
                               case["raw"]["reference"]["samples_ns"]),
                           "inferred_median_ns": statistics.median(
                               case["raw"]["inferred"]["samples_ns"]),
                           "same_run_reference_p95_ns": percentile(
                               case["raw"]["reference"]["samples_ns"], .95),
                           "inferred_p95_ns": percentile(
                               case["raw"]["inferred"]["samples_ns"], .95),
                           "compile_median_ratio": statistics.median(compile_ratios),
                           "compile_p95_ratio": percentile(compile_ratios, .95),
                           "compile_max_ratio": max(compile_ratios),
                           "compile_ok": all(ratio <= .10 for ratio in compile_ratios)}
                if baseline_cases is not None:
                    base_median = statistics.median(baseline_cases[case_id]["samples_ns"])
                    ratio = summary["inferred_median_ns"] / base_median
                    summary.update(baseline_reference_median_ns=base_median,
                                   baseline_reference_p95_ns=percentile(
                                       baseline_cases[case_id]["samples_ns"], .95),
                                   runtime_ratio=ratio, runtime_ok=ratio <= 1.03)
                summaries.append(summary)
            report["conversions"] = {"cases": summaries, "raw": conversion_data}
            compile_ok = all(case["compile_ok"] for case in summaries)
            runtime_ok = baseline_cases is not None and all(case["runtime_ok"] for case in summaries)
            report["gates"]["compile"] = {"status": "PASS" if compile_ok else "FAIL"}
            report["gates"]["runtime"] = {"status": ("PASS" if runtime_ok else
                                                        "FAIL" if baseline_cases is not None else
                                                        baseline_status)}
            report["gates"]["overall"] = {"status": "PASS" if compile_ok and runtime_ok else
                                                      "FAIL" if not compile_ok else
                                                      "INCOMPLETE" if baseline_cases is None else "FAIL"}
            if args.record_reference:
                atomic_write(args.output_baseline, pending_baseline(conversion_data))
        else:
            report["gates"]["runtime"] = {"status": "INCOMPLETE"}
            report["gates"]["compile"] = {"status": "INCOMPLETE"}
            report["gates"]["overall"] = {"status": "INCOMPLETE"}
    except InvalidResult as exc:
        errors.append(str(exc))
        report["gates"]["schema"] = {"status": "FAIL"}
        report["gates"]["overall"] = {"status": "FAIL"}

    if errors:
        report["errors"] = errors
    if args.output:
        atomic_write(args.output, report)
    else:
        json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
    if errors:
        return 2
    if args.check and report["gates"]["overall"]["status"] not in ("PASS", "PARTIAL_PASS"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
