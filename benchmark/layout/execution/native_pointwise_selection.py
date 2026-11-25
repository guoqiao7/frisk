"""Select a native pointwise layout using the caller's representative launch.

prepare(packet, directory) must compile the packet using ordinary native vector
global I/O, check its outputs, and return (loaded_image, replay). replay must
enqueue the same number of launches on the current CUDA stream for all candidates.
Use fresh scratch space for every selection. Inputs, pipeline and native emitter
are fixed. For a single store, if both access prices choose shared redistribution,
also try the legal assignment with the fewest conversions, breaking ties by
estimated native service.
"""
from pathlib import Path
import hashlib
import json
import math
import random
import statistics
import subprocess
import tempfile
import time

import torch

# The caller may also supply a standalone CUDA compiler.
try:
    from triton.runtime.errors import OutOfResources
except ImportError:
    _COMPILE_FAILURES = (subprocess.CalledProcessError,)
else:
    _COMPILE_FAILURES = (subprocess.CalledProcessError, OutOfResources)


def _confirm(replays, baseline, challenger):
    """Twelve order-balanced blocks; ten must independently clear 2%."""
    device = torch.cuda.current_device()
    orders = random.Random(127133)
    stream = torch.cuda.current_stream(device)
    for name in (baseline, challenger):
        for _ in range(3):
            replays[name]()
    stream.synchronize()
    blocks = []
    for _ in range(12):
        order = [baseline, challenger]
        orders.shuffle(order)
        ratios = []
        for sequence in (order, list(reversed(order))):
            events = {}
            for name in sequence:
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record(stream)
                replays[name]()
                stop.record(stream)
                events[name] = (start, stop)
            stream.synchronize()
            times = {k: start.elapsed_time(stop) for k, (start, stop) in events.items()}
            if not all(v > 0 and math.isfinite(v) for v in times.values()):
                raise ValueError("nonpositive or nonfinite CUDA event interval")
            ratios.append(times[challenger] / times[baseline])
        blocks.append(statistics.geometric_mean(ratios))
    return blocks


def compile_pointwise_access(source, frisk_opt, output_dir, prepare):
    """Return the validated winning image, its execution packet and a receipt."""
    parent = Path(output_dir)
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="access-", dir=parent))
    source = Path(source).resolve()
    images, packets, replays, digests, elapsed = {}, {}, {}, {}, {}
    receipt = {"version": 2, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "directory": str(work.resolve()), "cubins": digests, "prepare_ms": elapsed}
    pointwise_ops = {"builtin.module", "frisk.kernel", "frisk.parallel", "frisk.end",
                     "frisk.storage_contract", "frisk.layout_view", "frisk.tile_load",
                     "frisk.tile_store", "frisk.convert_layout", "frisk.index"}

    def solve(name, price, extra=""):
        d = work / name
        d.mkdir()
        path = (d / "packet.json").resolve()
        flag = ("--frisk-protected-store-layout=anchor-solver=integer integer-free-bits=0 "
                "integer-oracle-limit=0 integer-global-access=" + price + extra)
        subprocess.run([str(frisk_opt), str(source), flag, "-verify-each",
                        "--frisk-export-execution=output=" + str(path) + " minify=true",
                        "-o", str(d / "selected.mlir")], check=True)
        packet = json.loads(path.read_bytes())
        ops = [op["name"] for op in packet["operations"]]
        if ("frisk.tile_load" not in ops or "frisk.tile_store" not in ops or
                any(op not in pointwise_ops and not op.startswith("arith.") for op in ops)):
            raise ValueError("paired pointwise selection requires a memory/elementwise graph")
        return packet

    def compile_arm(name, packet, start):
        packets[name] = packet
        images[name], replays[name] = prepare(packet, work / name)
        digests[name] = hashlib.sha256(images[name].asm["cubin"]).hexdigest()
        elapsed[name] = (time.perf_counter() - start) * 1000
    def has_shared(packet):
        return any(op.get("redistribution", {}).get("mode") == "Shared"
                   for op in packet["operations"])

    confirm_ms = 0.0

    def compare(incumbent, challenger):
        nonlocal confirm_ms
        if digests[incumbent] == digests[challenger]:
            return incumbent, []
        start = time.perf_counter()
        ratios = _confirm(replays, incumbent, challenger)
        confirm_ms += (time.perf_counter() - start) * 1000
        winner = challenger if sum(r < 0.98 for r in ratios) >= 10 else incumbent
        return winner, ratios

    def try_compatible(incumbent):
        start = time.perf_counter()
        try:
            candidate = solve("compatible", "native", " integer-pointwise-min-conversions=true")
            if has_shared(candidate):
                return incumbent, {"status": "still-shared"}
            compile_arm("compatible", candidate, start)
        except _COMPILE_FAILURES as exc:
            return incumbent, {"status": "unavailable", "error": str(exc)}
        if digests["compatible"] in (digests["scalar"], digests["native"]):
            return incumbent, {"status": "duplicate"}
        selected, ratios = compare(incumbent, "compatible")
        return selected, {"status": "selected" if selected == "compatible" else "rejected",
                          "block_ratios": ratios}

    for name in ("scalar", "native"):
        start = time.perf_counter()
        compile_arm(name, solve(name, name), start)
    winner, blocks = compare("scalar", "native")
    # A whole-graph compatibility trial targets single-sink delivery. Fanout
    # keeps the independent per-consumer choices from the two ordinary solves.
    single_store = sum(op["name"] == "frisk.tile_store"
                       for op in packets[winner]["operations"]) == 1
    trial = {"against": winner, "status": "existing-shared-free" if single_store else "multiple-stores",
             "block_ratios": []}
    if single_store and all(has_shared(packet) for packet in packets.values()):
        winner, details = try_compatible(winner)
        trial.update(details)
    receipt.update(winner=winner, block_ratios=blocks, confirm_ms=confirm_ms, compatibility=trial)
    (work / "selection.json").write_text(json.dumps(receipt, indent=2))
    return images[winner], packets[winner], receipt
