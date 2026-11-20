"""Block-local schedule descriptors for a proven complete native call closure.

All branch/call boundaries cut dependency propagation. This describes static
physical-register order inside basic blocks, not a dynamic critical path,
cycles, issue width or a whole-CFG dependency reconstruction. It is a distinct
calibration family; the counts-only summary v1 remains unchanged.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import execution_cost_summary as complete
import native_cost_analysis as native

CONTRACT = "frisk-block-local-execution-cost-summary-v1"
FAMILY = "sm90-acyclic-block-local-schedule-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NATIVE_ANALYZER_SHA256 = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
SCHEDULE_FEATURES = (
    "block_dependency_depth_sum", "block_dependency_depth_max",
    "block_raw_dependency_depth_sum", "block_raw_dependency_depth_max",
    "block_load_use_exposure_32", "block_load_use_exposure_128",
    "block_unconsumed_loads", "block_count",
)
FEATURE_UNITS = {name: ("maximum block-local unit DAG path nodes" if name.endswith("_max") else "sum of block-local unit DAG path nodes, call-site expanded" if name.endswith("_sum") else "sum(max(0, window - first-use list distance)) within blocks, call-site expanded" if "exposure" in name else "static block-local load sites without a same-block register use, call-site expanded" if name == "block_unconsumed_loads" else "reachable basic blocks, call-site expanded") for name in SCHEDULE_FEATURES}


def _sha(x): return hashlib.sha256(x).hexdigest()

def _canonical(x): return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def _decode(x):
    base = x["opcode"].split(".")[0]
    if base not in {"PLOP3", "HFMA2"}: return native._decode(x)
    # PLOP3 has two predicate destinations; HFMA2 writes one packed 32-bit
    # destination. The arithmetic operation is irrelevant to register edges.
    operands = native._split_operands(x.get("operands", ""))
    destinations = 2 if base == "PLOP3" else 1
    if len(operands) <= destinations: raise ValueError("block-unknown-operand-form:" + base)
    writes = set().union(*(native._regs(s) for s in operands[:destinations]))
    reads = native._regs(native._execution_predicate(x))
    for s in operands[destinations:]: reads |= native._regs(s)
    if native._execution_predicate(x): reads |= writes
    return dict(opcode=x["opcode"], base=base, reads=reads, writes=writes, scope=None, store=False)


def _block_statistics(raw):
    decoded = [_decode(x) for x in raw if x["opcode"].split(".")[0] not in complete.CONTROL]
    last_writer, pending_readers = {}, {}
    memory_store, memory_reads = {}, {}
    barrier = None; phase = []
    depths, raw_depths, load_words, first_use = [], [], {}, {}
    edge_count = 0
    for i, x in enumerate(decoded):
        deps, raw_deps = set(), set()
        for reg in x["reads"]:
            if reg in last_writer:
                writer = last_writer[reg]; deps.add(writer); raw_deps.add(writer)
                if writer in load_words: first_use.setdefault(writer, i)
        for reg in x["writes"]:
            if reg in last_writer: deps.add(last_writer[reg])
            deps.update(pending_readers.get(reg, ()))
        if barrier is not None: deps.add(barrier)
        if x["base"] in {"BAR", "MEMBAR"}:
            deps.update(phase); phase = []; barrier = i
        scope = x["scope"]
        if scope:
            if scope in memory_store: deps.add(memory_store[scope])
            if x["store"]:
                deps.update(memory_reads.get(scope, ())); memory_reads[scope] = []; memory_store[scope] = i
            else: memory_reads.setdefault(scope, []).append(i)
        deps.discard(i); edge_count += len(deps)
        if edge_count > native.MAX_EDGES: raise ValueError("block-dependency-edge-budget")
        depths.append(1 + max((depths[p] for p in deps), default=0))
        raw_depths.append(1 + max((raw_depths[p] for p in raw_deps), default=0))
        for reg in x["writes"]: last_writer[reg] = i; pending_readers[reg] = set()
        for reg in x["reads"] - x["writes"]: pending_readers.setdefault(reg, set()).add(i)
        if x["base"] in native.LOADS: load_words[i] = x["writes"]
        phase.append(i)
    distance = [use - index for index, use in sorted(first_use.items())]
    depth, raw_depth = max(depths, default=0), max(raw_depths, default=0)
    return dict(block_dependency_depth_sum=depth, block_dependency_depth_max=depth,
                block_raw_dependency_depth_sum=raw_depth, block_raw_dependency_depth_max=raw_depth,
                block_load_use_exposure_32=sum(max(0, 32 - d) for d in distance),
                block_load_use_exposure_128=sum(max(0, 128 - d) for d in distance),
                block_unconsumed_loads=len(load_words) - len(first_use), block_count=1), dict(decoded_instructions=len(decoded), dependency_edges=edge_count)


def analyze_document(document, base_summary):
    """Pure block partition and descriptor. Requires the v1 complete proof."""
    try:
        if base_summary.get("status") != "complete" or base_summary.get("cost_family") != complete.SCALAR_FAMILY: raise ValueError("block-complete-scalar-family-required")
        regenerated = complete.analyze_document(document, entry=base_summary["proof"]["entry"])
        if regenerated.get("status") != "complete" or regenerated["features"] != base_summary["features"] or _canonical(regenerated["proof"]) != _canonical(base_summary["proof"]): raise ValueError("block-complete-proof-mismatch")
        functions = {f["function-name"]: f for f in document[1]}
        done = {}
        def visit(name):
            if name in done: return done[name]
            f = functions[name]; ins = {f["start"] + 16 * i: x for i, x in enumerate(f["sass-instructions"])}
            todo, reachable, edges, calls = [f["start"]], set(), {}, {}
            while todo:
                addr = todo.pop()
                if addr in reachable: continue
                reachable.add(addr); x = ins[addr]; base = x["opcode"].split(".")[0]
                if base == "BRA":
                    operands = x["operands"].split(","); successors = [int(operands[-1], 16)]
                    if x.get("predicate") or len(operands) > 1: successors.append(addr + 16)
                elif base in {"RET", "EXIT"}: successors = []
                else: successors = [addr + 16]
                if base == "CALL": calls[addr] = x["operands"]
                edges[addr] = sorted(set(successors)); todo.extend(successors)
            leaders = {f["start"]}
            for addr, out in edges.items():
                if ins[addr]["opcode"].split(".")[0] in complete.CONTROL: leaders.update(out)
            covered, blocks = set(), []
            for first in sorted(leaders):
                if first in covered: continue
                block, addr = [], first
                while True:
                    if addr in covered: raise ValueError("overlapping-basic-blocks")
                    covered.add(addr); block.append(addr)
                    out = edges[addr]
                    if len(out) != 1 or out[0] in leaders: break
                    addr = out[0]
                blocks.append(block)
            if covered != reachable: raise ValueError("incomplete-basic-block-partition")
            totals = {name: 0 for name in SCHEDULE_FEATURES}; details = []
            for block in blocks:
                stats, diag = _block_statistics([ins[a] for a in block])
                for key, value in stats.items():
                    totals[key] = max(totals[key], value) if key.endswith("_max") else totals[key] + value
                details.append(dict(first=block[0], last=block[-1], native_sites=len(block), **stats, **diag))
            for addr, callee in sorted(calls.items()):
                child = visit(callee)
                for key, value in child["features"].items():
                    totals[key] = max(totals[key], value) if key.endswith("_max") else totals[key] + value
            result = dict(features=totals, blocks=details)
            done[name] = result
            return result
        result = visit(base_summary["proof"]["entry"])
        return dict(status="complete", features=result["features"], proof={"functions": {k: done[k]["blocks"] for k in sorted(done)},
                    "block_boundary_semantics": "BSSY/BSYNC/BRA/CALL/RET/EXIT cut all register and memory dependency propagation; control sites are excluded from scalar DAG nodes; no cross-block, cross-call, dynamic path, latency or GPU issue inference"})
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return dict(status="unknown", features=None, reasons=[str(error)])


def summarize(directory, base_summary=None, *, nvdisasm="/usr/local/cuda/bin/nvdisasm"):
    """Refine a cached v1 summary after checking its lineage and CUBIN anew.

    A caller already holding a v1 result avoids repeating its disassembly.
    Attention v1 is returned unchanged: scalar block-DAG weights do not apply.
    """
    root = Path(directory)
    base_summary = complete.summarize(root, nvdisasm=nvdisasm) if base_summary is None else base_summary
    if base_summary.get("status") != "complete": return base_summary
    if base_summary.get("cost_family") == complete.ATTENTION_FAMILY: return base_summary
    try:
        if _sha(Path(__file__).read_bytes()) != SOURCE_SHA256 or _sha(Path(native.__file__).read_bytes()) != NATIVE_ANALYZER_SHA256: raise ValueError("block-analyzer-source-changed")
        original = dict(base_summary); expected = original.pop("summary_sha256")
        if _sha(_canonical(original)) != expected: raise ValueError("base-summary-payload-digest")
        provenance = base_summary["provenance"]
        if provenance["analyzer_sha256"] != complete.SOURCE_SHA256: raise ValueError("base-analyzer-source-mismatch")
        for name, expected_sha in provenance["dependencies"].items():
            if _sha((root / name).read_bytes()) != expected_sha: raise ValueError("base-dependency-mismatch:" + name)
        tool = Path(nvdisasm).resolve(strict=True)
        if _sha(tool.read_bytes()) != provenance["nvdisasm"]["sha256"]: raise ValueError("base-native-tool-mismatch")
        raw = subprocess.check_output([str(tool), "-json", str(root / "kernel.cubin")], stderr=subprocess.PIPE, timeout=120)
        if _sha(raw) != provenance["native_json_sha256"]: raise ValueError("native-document-identity-mismatch")
        extension = analyze_document(json.loads(raw), original)
        if extension["status"] != "complete": raise ValueError(extension["reasons"][0])
        result = dict(original)
        result.update(contract=CONTRACT, cost_family=FAMILY, features=dict(original["features"], **extension["features"]),
                      feature_names=original["feature_names"] + list(SCHEDULE_FEATURES),
                      feature_units=dict(original["feature_units"], **FEATURE_UNITS),
                      schedule_proof=extension["proof"],
                      provenance=dict(provenance, analyzer_sha256=SOURCE_SHA256,
                                      base_summary_sha256=base_summary["summary_sha256"], base_analyzer_sha256=complete.SOURCE_SHA256,
                                      native_decoder_sha256=NATIVE_ANALYZER_SHA256))
        result["execution_bounds"] = dict(original["execution_bounds"], feature_scope=list(complete.BASE_FEATURES))
        result["semantics"] += "; additional schedule descriptors are block-local conservative unit graphs with explicitly cut control boundaries, not a whole-program dynamic schedule"
        for name, expected_sha in provenance["dependencies"].items():
            if _sha((root / name).read_bytes()) != expected_sha: raise ValueError("base-dependency-changed-during-analysis:" + name)
        result["summary_sha256"] = _sha(_canonical(result))
        return result
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, IndexError) as error:
        return dict(contract=CONTRACT, status="unknown", cost_family=FAMILY, features=None, reasons=[str(error)], base_summary_sha256=base_summary.get("summary_sha256"))
