#!/usr/bin/env python3
"""Real Frisk -> layout pipeline -> typed GPU ABI -> adapter -> PTX/cubin driver.

No baseline is written here. Every invocation uses an independent artifact
directory and retains input, commands, phase statistics and raw GPU samples.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time

CASES = ["identity_f32", "shuffle32_f32", "shared64_f32"]
REGRESSIONS = ["replicated64_f32"] + [f"shuffle32_{t}" for t in ("i8", "i16", "i32", "i64", "f16", "bf16", "f64")]
FLAGS = ["sm_90", "-O3", "-lineinfo", "bare-pointer-memref=true", "cache=disabled"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest_from_manifest(manifest, relative, source):
    entries = [line.rsplit(":", 1)[1] for line in Path(manifest).read_text().splitlines()
               if line.startswith(relative + ":")]
    if len(entries) != 1:
        raise ValueError("missing or duplicate build-manifest source entry: " + relative)
    if entries[0] != digest(source):
        raise ValueError("live source does not match built source manifest: " + relative)
    return entries[0]


def extract_ptx(text):
    matches = re.findall(r'assembly = "((?:[^"\\]|\\.)*)"', text)
    if len(matches) != 1:
        raise ValueError("expected exactly one serialized GPU assembly object")
    encoded, result, i = matches[0], bytearray(), 0
    while i < len(encoded):
        if encoded[i] != "\\":
            result.extend(encoded[i].encode("utf-8"))
            i += 1
        elif re.fullmatch(r"[0-9a-fA-F]{2}", encoded[i + 1:i + 3]):
            result.append(int(encoded[i + 1:i + 3], 16))
            i += 3
        else:
            escapes = {'"': 34, "\\": 92, "n": 10, "t": 9, "r": 13}
            result.append(escapes[encoded[i + 1]])
            i += 2
    return bytes(result)


def check_adapter(text):
    if any(token in text for token in ("unrealized_conversion_cast", "tensor<", "frisk.convert_layout", "frisk.tile_load", "frisk.tile_store", "func.call")):
        raise ValueError("adapter has unresolved tensor/layout or placeholder call")


def execute(command, directory, label):
    result = subprocess.run([str(arg) for arg in command], text=True, capture_output=True)
    (directory / f"{label}.stdout").write_text(result.stdout)
    (directory / f"{label}.stderr").write_text(result.stderr)
    (directory / f"{label}.command").write_text(shlex.join([str(arg) for arg in command]) + "\n")
    if result.returncode:
        raise RuntimeError(f"{label} failed with exit {result.returncode}; see {directory / (label + '.stderr')}")
    return result


def geometry(case):
    kind, dtype = case.rsplit("_", 1)
    size = {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}[dtype]
    return size, 64 if kind in ("shared64", "replicated64") else 32, 64 if kind == "shared64" else 32, 0 if kind == "identity" else 2 if kind == "shared64" else 1


def compile_case(case, args, root):
    directory = root / case
    directory.mkdir()
    generated = execute([args.generator, case], directory, "generate-input")
    original = directory / "input.mlir"
    original.write_text(generated.stdout)
    layout, promoted, adapter, lowered = [directory / name for name in ("layout.mlir", "promoted.mlir", "adapter.mlir", "lowered.mlir")]
    stats = directory / "layout-statistics.json"
    ptx, cubin = directory / "kernel.ptx", directory / "kernel.cubin"
    commands = []
    # Only actual compilation from the same input to its loadable cubin enters
    # this denominator. Tool construction/reference compilation/GPU work do not.
    start = time.perf_counter_ns()
    stages = [
        ("layout", [args.frisk_opt, original, f"-frisk-layout-pipeline=stats-file={stats}", "-o", layout]),
        ("promote", [args.generator, "--promote", layout, promoted]),
        ("adapter", [args.frisk_opt, promoted, "-frisk-test-lower-layout-conversions=executable", "-o", adapter]),
        ("backend", [args.mlir_opt, adapter, "--gpu-lower-to-nvvm-pipeline=cubin-chip=sm_90 cubin-format=isa kernel-bare-ptr-calling-convention=true opt-level=3", "-o", lowered]),
    ]
    for label, command in stages:
        execute(command, directory, label)
        commands.append(shlex.join([str(arg) for arg in command]))
        if label == "adapter":
            check_adapter(adapter.read_text())
    ptx.write_bytes(extract_ptx(lowered.read_text()))
    command = [args.ptxas, "-arch=sm_90", "-O3", "-lineinfo", ptx, "-o", cubin]
    execute(command, directory, "ptxas")
    full_ns = time.perf_counter_ns() - start
    commands.append(shlex.join([str(arg) for arg in command]))
    statistics = json.loads(stats.read_text())
    if statistics["status"] != "committed":
        raise ValueError("layout pipeline did not commit")
    provenance = {
        "frisk_input_sha256": digest(original), "promoted_ir_sha256": digest(promoted),
        "adapter_ir_sha256": digest(adapter), "lowered_ir_sha256": digest(lowered),
        "ptx_sha256": digest(ptx), "cubin_sha256": digest(cubin),
        "source_digest": args.source_digest, "commands": commands, "compile_flags": FLAGS,
        "configuration_digest": args.configuration_digest,
        "artifact_directory": str(directory), "tool_digests": args.tool_digests,
    }
    return {"id": case, "provenance": provenance,
            "compilation": {"cache": "disabled", "samples": [{"layout_pipeline_ns": statistics["timings"]["layout_pipeline_ns"], "full_compile_ns": full_ns}]},
            "artifact_checks": {"unrealized_casts": 0, "unresolved_layouts": 0, "placeholder_calls": 0}}


def runtime_command(case, record, args, samples):
    size, threads, volume, mode = geometry(case)
    return [args.runtime, Path(record["provenance"]["artifact_directory"]) / "kernel.cubin", f"inferred_{case}",
            str(size), str(threads), str(volume), str(mode), str(args.device), str(samples), case, args.reference]


def environment(args, root, label="environment-before"):
    query = execute(["nvidia-smi", "-i", str(args.device), "--query-gpu=uuid,name,compute_cap,driver_version,clocks.current.sm,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], root, label)
    uuid, name, capability, driver, clock, load, memory = [part.strip() for part in query.stdout.strip().split(",")]
    toolkit = execute([args.nvcc, "--version"], root, "toolkit").stdout.strip()
    return {"gpu_uuid": uuid, "gpu_name": name, "compute_capability": capability, "driver": driver,
            "toolkit": toolkit, "clock_policy": "unmodified", "compile_flags": FLAGS, "device": args.device,
            "background_load": f"utilization={load}%,memory={memory}MiB,sm_clock={clock}MHz"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generator", "runtime", "frisk-opt", "mlir-opt", "output-dir", "source-manifest", "build-configuration"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--nvcc", default="/usr/local/cuda/bin/nvcc")
    parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--sanitizer", default="/usr/local/cuda/bin/compute-sanitizer")
    parser.add_argument("--reference")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    args.reference = args.reference or str(Path(args.runtime).parent / "reference.cubin")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    args.tool_digests = {name: digest(getattr(args, name)) for name in ("generator", "runtime", "frisk_opt", "mlir_opt", "ptxas", "reference")}
    args.tool_digests["driver_script"] = digest(__file__)
    args.source_digest = digest(args.source_manifest)
    args.configuration_digest = digest(args.build_configuration)
    (root / "source-manifest.txt").write_bytes(Path(args.source_manifest).read_bytes())
    (root / "build-configuration.txt").write_bytes(Path(args.build_configuration).read_bytes())
    result = {"schema_version": 1, "kind": "conversion", "status": "failed", "manifest": CASES, "cases": [], "artifact_directory": str(root)}
    try:
        reference_source_digest = source_digest_from_manifest(args.source_manifest,
            "benchmark/layout/conversion/layout_conversion_bench.cu", Path(__file__).with_name("layout_conversion_bench.cu"))
        if not args.compile_only:
            result["environment"] = environment(args, root)
        records = {case: compile_case(case, args, root) for case in CASES + REGRESSIONS}
        result["cases"] = [records[case] for case in CASES]
        result["payload_regressions"] = [records[case] for case in REGRESSIONS]
        if args.compile_only:
            result["status"] = "compiled-not-executed"
        else:
            for case in CASES + REGRESSIONS:
                record = records[case]
                directory = Path(record["provenance"]["artifact_directory"])
                execute(runtime_command(case, record, args, 0), directory, "correctness")
                safety = {}
                for tool in ("memcheck", "racecheck", "synccheck"):
                    command = [args.sanitizer, "--tool", tool, "--error-exitcode", "97"] + runtime_command(case, record, args, 0)
                    output = execute(command, directory, tool)
                    log = output.stdout + output.stderr
                    if not re.search(r"(?:ERROR SUMMARY: 0 errors|RACECHECK SUMMARY: 0 hazards)", log):
                        raise ValueError(f"{case}: no positive sanitizer completion evidence for {tool}")
                    safety[tool] = "pass"
                record.update(status="success", sanitizers=safety)
            result["environment_measurement_before"] = environment(args, root, "environment-measurement-before")
            for case in CASES:
                record = records[case]
                directory = Path(record["provenance"]["artifact_directory"])
                measured = json.loads(execute(runtime_command(case, record, args, 1000), directory, "runtime").stdout)
                if measured.get("correctness") != "pass":
                    raise ValueError("missing runtime correctness evidence")
                record.update(threads=geometry(case)[1], dtype="f32", warmup=100, iterations=1000,
                              measurement_order="alternating",
                              correctness={"status": "pass", "seeds": [1, 7, 42, 20260816], "boundary_values": True,
                                           "replication": True},
                              replication_case="replicated64_f32",
                              reference={"origin": "independent-coordinate-reference", "samples_ns": measured["reference_samples_ns"],
                                         "source_sha256": reference_source_digest},
                              inferred={"origin": "frisk-conversion-adapter", "samples_ns": measured["inferred_samples_ns"]})
            result["environment_measurement_after"] = environment(args, root, "environment-measurement-after")
            if any(result[key]["gpu_uuid"] != result["environment"]["gpu_uuid"]
                   for key in ("environment_measurement_before", "environment_measurement_after")):
                raise ValueError("GPU identity changed during measurement")
            result["status"] = "success"
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        result["failure"] = str(error)
    (root / "raw-results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] in ("success", "compiled-not-executed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
