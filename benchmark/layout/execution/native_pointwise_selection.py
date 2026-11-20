"""Select a native pointwise layout using the caller's representative launch.

prepare(packet, directory) must compile the packet using ordinary native vector
global I/O, check its outputs, and return (loaded_image, replay). replay must
enqueue the same number of launches on the current CUDA stream for both arms.
Use fresh scratch space for every selection. Both candidates run the same IR,
pipeline and native emitter. Only the access pricing used in inference varies.
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


def _confirm(replays):
    """Twelve order-balanced blocks; ten must independently clear 2%."""
    device = torch.cuda.current_device()
    orders = random.Random(127133)
    stream = torch.cuda.current_stream(device)
    for replay in replays.values():
        for _ in range(3):
            replay()
    stream.synchronize()
    blocks = []
    for _ in range(12):
        order = list(replays)
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
            ratios.append(times["native"] / times["scalar"])
        blocks.append(statistics.geometric_mean(ratios))
    return blocks


def compile_pointwise_access(source, frisk_opt, output_dir, prepare):
    """Return the validated winning image, its execution packet and a receipt."""
    parent = Path(output_dir)
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="access-", dir=parent))
    source = Path(source).resolve()
    images, packets, replays, digests, elapsed = {}, {}, {}, {}, {}
    receipt = {"version": 1, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "directory": str(work.resolve()), "cubins": digests, "prepare_ms": elapsed}
    pointwise_ops = {"builtin.module", "frisk.kernel", "frisk.parallel", "frisk.end",
                     "frisk.storage_contract", "frisk.layout_view", "frisk.tile_load",
                     "frisk.tile_store", "frisk.convert_layout", "frisk.index"}
    for name in ("scalar", "native"):
        d = work / name
        d.mkdir()
        path = (d / "packet.json").resolve()
        start = time.perf_counter()
        flag = ("--frisk-protected-store-layout=anchor-solver=integer integer-free-bits=0 "
                "integer-oracle-limit=0 integer-global-access=" + name)
        subprocess.run([str(frisk_opt), str(source), flag, "-verify-each",
                        "--frisk-export-execution=output=" + str(path) + " minify=true",
                        "-o", str(d / "selected.mlir")], check=True)
        packet = json.loads(path.read_bytes())
        ops = [op["name"] for op in packet["operations"]]
        if ("frisk.tile_load" not in ops or "frisk.tile_store" not in ops or
                any(op not in pointwise_ops and not op.startswith("arith.") for op in ops)):
            raise ValueError("paired pointwise selection requires a memory/elementwise graph")
        packets[name] = packet
        images[name], replays[name] = prepare(packet, d)
        digests[name] = hashlib.sha256(images[name].asm["cubin"]).hexdigest()
        elapsed[name] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    blocks = [] if digests["scalar"] == digests["native"] else _confirm(replays)
    winner = "native" if sum(r < 0.98 for r in blocks) >= 10 else "scalar"
    receipt.update(winner=winner, block_ratios=blocks,
                   confirm_ms=(time.perf_counter() - start) * 1000)
    (work / "selection.json").write_text(json.dumps(receipt, indent=2))
    return images[winner], packets[winner], receipt
