"""Complete native call-closure summaries, separate from native14.

The exact object is a static descriptor: every syntactically reachable site is
counted, and a callee descriptor is added at each reachable call site. Branches
are not assigned an observed frequency. These features require their own fitted
cost family; they cannot be substituted for straight-line native14 features.
No compilation, GPU access, timing-file read or filesystem write occurs here.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

CONTRACT = "frisk-complete-execution-cost-summary-v1"
SCALAR_FAMILY = "sm90-acyclic-call-expanded-occurrences-v1"
ATTENTION_FAMILY = "sm90-immediate-wait-wgmma-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MAX_SITES = 262144
BASE_FEATURES = (
    "native_instructions", "global_load_bytes", "global_store_bytes",
    "shared_transfer_bytes", "local_transfer_bytes", "barrier_instructions",
    "shuffle_instructions", "float_arithmetic_instructions",
    "special_math_instructions", "integer_logic_instructions",
    "control_instructions", "predicated_instructions", "call_sites",
    "callee_instructions",
)
ATTENTION_FEATURES = (
    "wgmma_m64n64k16_sites", "wgmma_m64n16k16_sites", "wgmma_fence_sites",
    "wgmma_commit_sites", "wgmma_wait0_sites", "native_hgmma_sites",
    "native_warpgroup_wait_sites", "ptx_f16_pack_sites", "native_f16_pack_sites",
    "declared_copy_payload_bytes", "vector_copy_operations", "cpasync_copy_sites",
    "cpasync_operand_bytes", "cpasync_commit_sites", "cpasync_wait0_sites",
    "tma_copy_sites", "tma_declared_transaction_bytes", "mbarrier_init_sites",
    "mbarrier_arrive_sites", "mbarrier_try_wait_sites",
    "native_mbarrier_try_wait_sites", "polling_sccs", "ptx_cta_barrier_sites",
    "async_proxy_fence_sites",
)
# These are explicit opcode classes, not an implicit zero-price catch-all.
FLOAT = {"FMUL", "FADD", "FFMA", "FMNMX", "FSEL", "HFMA2"}
SPECIAL = {"MUFU", "RRO"}
INTEGER = set("MOV UMOV IMAD UIMAD LOP3 ULOP3 SHF USHF SGXT PRMT SEL P2R R2P F2I I2F I2I F2F F2FP FLO POPC BREV BMSK UBMSK VIADD FCHK IADD3 UIADD3 LEA ULEA ISETP UISETP FSETP DSETP PSETP PLOP3 VOTE VOTEU".split())
SCALAR = FLOAT | SPECIAL | INTEGER | set("LDC ULDC S2R S2UR LDG LDS LDL STG STS STL SHFL BAR MEMBAR NOP".split())
ASYNC = {"LDGSTS", "LDGDEPBAR", "DEPBAR", "HGMMA", "WARPGROUP", "SYNCS", "FENCE", "UTMALDG"}
CONTROL = {"BRA", "BSSY", "BSYNC", "CALL", "RET", "EXIT"}
UNITS = {name: ("static operand bytes per thread-site, call-site expanded" if name.endswith("_bytes") else "reachable static instruction sites, call-site expanded") for name in BASE_FEATURES}
UNITS.update({
    "global_load_bytes": "LDG operand bytes per reachable thread-site, call-site expanded; excludes async copy",
    "global_store_bytes": "STG operand bytes per reachable thread-site, call-site expanded",
    "shared_transfer_bytes": "LDS+STS operand bytes per reachable thread-site; excludes async copy and tensor reads",
    "local_transfer_bytes": "LDL+STL operand bytes per reachable thread-site; never driver LOCAL frame bytes",
    "barrier_instructions": "BAR+MEMBAR static sites; async/tensor barriers have separate features",
    "callee_instructions": "callee static instruction sites expanded once per reachable caller CALL site",
    "float_arithmetic_instructions": "native float instruction sites including packed HFMA2; not FLOPs or f32 scalar element count",
})
UNITS.update({name: ("declared ABI payload bytes, not measured traffic" if name in {"declared_copy_payload_bytes", "tma_declared_transaction_bytes"} else "PTX operand bytes per site" if name == "cpasync_operand_bytes" else "static sites, operations or SCCs as named; no execution frequency") for name in ATTENTION_FEATURES})


def _sha(x):
    return hashlib.sha256(x).hexdigest()


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()


def _unknown(reason, **kw):
    return dict(contract=CONTRACT, status="unknown", cost_family=None,
                features=None, execution_bounds=None, reasons=[str(reason)], **kw)


def _width(opcode):
    parts = opcode.split(".")[1:]
    for size in (128, 64, 16, 8):
        if any(x in parts for x in (str(size), "U" + str(size), "S" + str(size))):
            return size // 8
    return 4


def _site_features(x):
    base = x["opcode"].split(".")[0]
    c = Counter(native_instructions=1)
    if base in {"LDG", "STG"}:
        c["global_load_bytes" if base == "LDG" else "global_store_bytes"] += _width(x["opcode"])
    if base in {"LDS", "STS"}: c["shared_transfer_bytes"] += _width(x["opcode"])
    if base in {"LDL", "STL"}: c["local_transfer_bytes"] += _width(x["opcode"])
    c["barrier_instructions"] += base in {"BAR", "MEMBAR"}
    c["shuffle_instructions"] += base == "SHFL"
    c["float_arithmetic_instructions"] += base in FLOAT
    c["special_math_instructions"] += base in SPECIAL
    c["integer_logic_instructions"] += base in INTEGER
    c["control_instructions"] += base in CONTROL
    # nvdisasm PR is a P2R data source, not an execution predicate.
    c["predicated_instructions"] += bool(x.get("predicate")) and not (base == "P2R" and x.get("predicate") == "PR")
    c["call_sites"] += base == "CALL"
    return c


def _scc(nodes, edges):
    """Iterative Kosaraju, bounded by raw sites/edges (no Python recursion)."""
    seen, order = set(), []
    for start in nodes:
        if start in seen: continue
        stack = [(start, False)]
        while stack:
            node, done = stack.pop()
            if done: order.append(node); continue
            if node in seen: continue
            seen.add(node); stack.append((node, True))
            stack.extend((v, False) for v in edges[node] if v not in seen)
    reverse = {x: [] for x in nodes}
    for x in nodes:
        for y in edges[x]: reverse[y].append(x)
    seen, components = set(), []
    for start in reversed(order):
        if start in seen: continue
        todo, component = [start], []
        while todo:
            x = todo.pop()
            if x in seen: continue
            seen.add(x); component.append(x)
            todo.extend(y for y in reverse[x] if y not in seen)
        components.append(sorted(component))
    return components


def analyze_document(document, *, entry="frisk_execution", allow_attention=False):
    """Pure complete-CFG core. Exact static features; separate dynamic bounds.

    The abstract CFG is a conservative per-thread graph: conditional BRA has
    both successors, CALL has a verified local callee and return continuation,
    and matched BSSY/BSYNC reconvergence has no unmodelled jump. All reachable
    exits/returns and all callee closures are checked. Dead padding after a RET
    is excluded by reachability, never by a first-EXIT prefix truncation.
    """
    try:
        if not isinstance(document, list) or len(document) != 2: raise ValueError("native-json-schema")
        meta, functions = document
        if not isinstance(meta, dict): raise ValueError("native-metadata-fields")
        if not isinstance(meta.get("SM"), dict) or not isinstance(meta.get("SchemaVersion"), dict): raise ValueError("native-metadata-fields")
        if meta.get("SM", {}).get("version") != {"major": 9, "minor": 0}: raise ValueError("architecture-not-sm90")
        if meta.get("SchemaVersion", {}).get("major") not in {12, 13}: raise ValueError("native-json-version")
        if not isinstance(functions, list) or not functions: raise ValueError("function-list")
        by_name = {}; all_addresses = set()
        for f in functions:
            if not isinstance(f, dict): raise ValueError("native-function-fields")
            name, start, raw = f["function-name"], f["start"], f["sass-instructions"]
            if not isinstance(name, str) or name in by_name or type(start) is not int or start < 0 or start % 16 or not isinstance(raw, list) or not raw or any(not isinstance(x, dict) for x in raw) or len(raw) > MAX_SITES: raise ValueError("function-shape-or-budget")
            if not isinstance(f.get("length"), int) or f["length"] < len(raw) * 16: raise ValueError("function-length")
            ins = {start + i * 16: x for i, x in enumerate(raw)}
            if all_addresses.intersection(ins): raise ValueError("overlapping-native-sites")
            all_addresses.update(ins); by_name[name] = dict(start=start, ins=ins)
        if entry not in by_name or len(all_addresses) > MAX_SITES: raise ValueError("entry-or-site-budget")
        done, active = {}, set()

        def visit(name):
            if name in done: return done[name]
            if name in active: raise ValueError("recursive-call-graph")
            if len(active) >= 32: raise ValueError("call-depth-budget")
            active.add(name)
            f = by_name[name]; ins = f["ins"]
            edges, calls, reachable, todo = {}, {}, set(), [f["start"]]
            joins = {}
            while todo:
                addr = todo.pop()
                if addr in reachable: continue
                if addr not in ins: raise ValueError("missing-native-target:" + hex(addr))
                reachable.add(addr); x = ins[addr]
                opcode = x.get("opcode", "")
                if not isinstance(opcode, str): raise ValueError("native-opcode-field")
                base = opcode.split(".")[0]
                operands = x.get("operands", ""); pred = x.get("predicate", "")
                if not isinstance(operands, str) or not isinstance(pred, str) or not opcode: raise ValueError("native-fields-or-code-gap")
                if pred and not re.fullmatch(r"@?!?(?:U?P[0-9]+|PT)|PR", pred): raise ValueError("unknown-execution-predicate")
                attrs = x.get("other-attributes", {})
                if not isinstance(attrs, dict): raise ValueError("native-attributes")
                if attrs.get("control-flow") == "True" and base not in CONTROL: raise ValueError("unknown-native-control:" + opcode)
                if base not in SCALAR | CONTROL | (ASYNC if allow_attention else set()): raise ValueError("unknown-native-opcode:" + opcode)
                next_addr = addr + 16
                if base == "BRA":
                    match = re.fullmatch(r"(?:(!?U?P[0-9]+),)?(0x[0-9a-fA-F]+)", operands)
                    if not match or opcode != "BRA": raise ValueError("indirect-or-unknown-branch")
                    successors = [int(match[2], 16)]
                    if pred or match[1]: successors.append(next_addr)
                elif base == "BSSY":
                    match = re.fullmatch(r"(B[0-9]+),(0x[0-9a-fA-F]+)", operands)
                    if not match or pred or opcode != "BSSY": raise ValueError("unknown-reconvergence-setup")
                    target = int(match[2], 16); end = ins.get(target - 16, {})
                    if target <= addr or target not in ins or end.get("opcode") != "BSYNC" or end.get("operands") != match[1] or end.get("predicate"): raise ValueError("unmatched-reconvergence")
                    joins.setdefault(target - 16, []).append((addr, match[1]))
                    successors = [next_addr]
                elif base == "BSYNC":
                    if pred or opcode != "BSYNC" or not re.fullmatch(r"B[0-9]+", operands): raise ValueError("unknown-reconvergence-join")
                    successors = [next_addr]
                elif base == "CALL":
                    if name != entry: raise ValueError("nested-local-call-abi-unproved")
                    if opcode != "CALL.REL.NOINC" or operands not in by_name: raise ValueError("nonlocal-or-unknown-call")
                    calls[addr] = operands; successors = [next_addr]
                elif base in {"EXIT", "RET"}:
                    if pred: raise ValueError("predicated-terminal")
                    if base == "EXIT" and (name != entry or opcode != "EXIT"): raise ValueError("callee-exit")
                    if base == "RET" and (name == entry or opcode != "RET.REL.NODEC" or not re.fullmatch(r"R[0-9]+," + re.escape(entry), operands)): raise ValueError("unknown-return-abi")
                    successors = []
                else: successors = [next_addr]
                edges[addr] = sorted(set(successors)); todo.extend(successors)
            for addr in reachable:
                if ins[addr]["opcode"] == "BSYNC" and addr not in joins: raise ValueError("join-without-setup")
            ranges = sorted((a, z, b) for z, setup in joins.items() for a, b in setup)
            # Crossing lifetimes or a live B-register redefinition are not in
            # the declared reconvergence domain.
            stack = []
            for start, end, breg in ranges:
                while stack and start > stack[-1][1]: stack.pop()
                if stack and (end >= stack[-1][1] or any(breg == t[2] for t in stack)): raise ValueError("crossing-reconvergence")
                stack.append((start, end, breg))
            # A matched textual setup is insufficient: reject edges entering a
            # live BSSY region while bypassing its setup. Dyadic interval
            # nesting gives a linear sweep instead of quadratic dominator sets.
            context, active_ranges, cursor = {}, [], 0
            for addr in sorted(reachable):
                while active_ranges and addr > active_ranges[-1][1]: active_ranges.pop()
                while cursor < len(ranges) and ranges[cursor][0] < addr:
                    region = ranges[cursor]; cursor += 1
                    while active_ranges and region[0] > active_ranges[-1][1]: active_ranges.pop()
                    if region[1] >= addr: active_ranges.append(region)
                context[addr] = {r[0] for r in active_ranges}
            for addr, successors in edges.items():
                for target in successors:
                    if context[target] - context[addr] - {addr}: raise ValueError("reconvergence-setup-bypassed")
            terminals = [a for a in reachable if not edges[a]]
            if not terminals: raise ValueError("terminal-missing")
            # Every reachable site must be able to reach a terminal, including
            # polling SCCs. A possible infinite poll still has unknown bound.
            reverse = {a: [] for a in reachable}
            for a, out in edges.items():
                for b in out: reverse[b].append(a)
            exits_reachable, queue = set(terminals), list(terminals)
            while queue:
                for a in reverse[queue.pop()]:
                    if a not in exits_reachable: exits_reachable.add(a); queue.append(a)
            if exits_reachable != reachable: raise ValueError("closed-nonterminating-region")
            cyclic = [c for c in _scc(sorted(reachable), edges) if len(c) > 1 or c[0] in edges[c[0]]]
            # Only the emitted two-site TMA parity polling loop is accepted;
            # execution frequency is explicitly unknown, never silently one.
            for component in cyclic:
                ops = [ins[a]["opcode"] for a in component]
                if not allow_attention or len(component) != 2 or sorted(ops) != ["BRA", "SYNCS.PHASECHK.TRANS64.TRYWAIT"]: raise ValueError("unproved-native-loop")
                branch = next(ins[a] for a in component if ins[a]["opcode"] == "BRA")
                wait_addr = next(a for a in component if ins[a]["opcode"] != "BRA")
                branch_addr = next(a for a in component if ins[a]["opcode"] == "BRA")
                predicate = re.match(r"(P[0-9]+)(?=,|\[)", ins[wait_addr].get("operands", ""))
                if not predicate or branch.get("predicate") != "@!" + predicate[1] or edges[wait_addr] != [branch_addr] or int(branch.get("operands", "0"), 16) != wait_addr: raise ValueError("unproved-poll-predicate")
            totals = Counter(); histogram = Counter(); predicated_histogram = Counter()
            for addr in sorted(reachable):
                totals.update(_site_features(ins[addr])); histogram[ins[addr]["opcode"]] += 1
                if _site_features(ins[addr])["predicated_instructions"]: predicated_histogram[ins[addr]["opcode"]] += 1
            callee_proofs = []
            for addr, callee in sorted(calls.items()):
                child = visit(callee)
                totals.update(child["features"]); totals["callee_instructions"] += child["features"]["native_instructions"]
                histogram.update(child["histogram"]); predicated_histogram.update(child["predicated_histogram"])
                callee_proofs.append({"site": addr, "callee": callee})
            active.remove(name)
            proof = {"name": name, "start": f["start"], "raw_sites": len(ins), "reachable_sites": len(reachable),
                     "unreachable_sites": len(ins) - len(reachable), "terminals": sorted(terminals),
                     "cfg_sha256": _sha(_canonical([[a, ins[a], edges[a]] for a in sorted(reachable)])),
                     "reconvergence_ranges": ranges, "calls": callee_proofs, "polling_sccs": cyclic,
                     "call_expanded_static_instructions": totals["native_instructions"]}
            result = {"features": totals, "histogram": histogram, "predicated_histogram": predicated_histogram, "proof": proof,
                      "has_poll": bool(cyclic) or any(done[c]["has_poll"] for c in calls.values())}
            done[name] = result
            return result
        result = visit(entry)
        features = {key: int(result["features"][key]) for key in BASE_FEATURES}
        return {"contract": CONTRACT, "status": "complete", "cost_family": ATTENTION_FAMILY if allow_attention else SCALAR_FAMILY,
                "feature_names": list(BASE_FEATURES), "feature_units": {k: UNITS[k] for k in BASE_FEATURES}, "features": features,
                "execution_bounds": {"scope": "per-thread scalar occurrences/operand bytes; not warp issue, cycles or active-lane counts",
                                     "lower": {k: 0 for k in BASE_FEATURES},
                                     "upper": None if result["has_poll"] else features.copy(),
                                     "reason": "TMA polling trip count is unbounded by static code" if result["has_poll"] else "acyclic per-thread CFG and nonrecursive local call graph; all possible branch bodies included"},
                "proof": {"entry": entry, "functions": [done[k]["proof"] for k in sorted(done)],
                          "opcode_histogram": dict(sorted(result["histogram"].items())),
                          "predicated_opcode_histogram": dict(sorted(result["predicated_histogram"].items())), "native_prefix_truncation": False},
                "semantics": "complete static reachable call-closure descriptor, not an exact dynamic cost or latency; no dependency-depth/exposure was invented; family-specific calibration required"}
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return _unknown(str(error))


def _ptx_instructions(text):
    out = []
    for line in text.splitlines():
        line = line.split("//", 1)[0].strip()
        match = re.fullmatch(r"(?:@!?[A-Za-z_][\w]*\s+)?([a-z][\w.:]*)(?:\s+(.*?))?;", line)
        if match: out.append((match[1], match[2] or ""))
    return out


def attention_features(execution, ptx, native_summary):
    """Validate the declared native SS/RS + immediate wait0 family.

    PTX and packet features remain separately named from native occurrences;
    source packing and lowered native packed instructions are both retained.
    """
    operations = execution["operations"]
    mma = [x["attrs"]["frisk.mma_contract"] for x in operations if x["name"] == "frisk.mma"]
    copies = [x["attrs"]["frisk.copy_contract"] for x in operations if x["name"] == "frisk.copy"]
    if not mma or not copies: raise ValueError("attention-contracts-missing")
    expected = Counter()
    for c in mma:
        atom, repeats = c.get("atom"), c.get("repeats")
        if c.get("target") != "sm_90a" or c.get("version") != 1 or c.get("input_type") != "f16" or c.get("accumulator_type") != "f32" or c.get("grid") != [1, 1] or atom not in ([64, 16, 16], [64, 64, 16]) or not isinstance(repeats, list) or len(repeats) != 3 or repeats[:2] != [1, 1] or not isinstance(repeats[2], int) or repeats[2] < 1: raise ValueError("attention-mma-contract-outside-domain")
        if c.get("form") == "ss":
            if repeats != [1, 1, 1] or c.get("packing") != "none": raise ValueError("attention-ss-contract")
        elif c.get("form") == "rs":
            if c.get("packing") != "f16x2-low-high": raise ValueError("attention-rs-packing")
        else: raise ValueError("attention-mma-form")
        expected["wgmma_m64n" + str(atom[1]) + "k16_sites"] += repeats[2]
    paths = {c.get("path") for c in copies}
    if len(paths) != 1 or not paths.issubset({"vector", "cp.async", "tma"}): raise ValueError("attention-copy-path")
    path = next(iter(paths)); features = {name: 0 for name in ATTENTION_FEATURES}
    for c in copies:
        if c.get("target") != "sm_90a" or c.get("version") != 1 or c.get("guarded") != 0 or c.get("valid_shape") != [64, 16]: raise ValueError("attention-copy-contract-outside-domain")
        completion = {"vector": "synchronous", "cp.async": "commit-wait-and-cta-visibility", "tma": "mbarrier-and-async-proxy"}[path]
        if c.get("completion") != completion or c.get("vector_bytes") != (1 if path == "vector" else 16): raise ValueError("attention-copy-completion")
        features["declared_copy_payload_bytes"] += 64 * 16 * 2
    if re.search(r"^\s*@[^\s]+\s+wgmma\.", ptx, re.M): raise ValueError("attention-predicated-wgmma")
    inst = _ptx_instructions(ptx)
    # No .func body or unvalidated external PTX routine can hide extra work.
    if len(re.findall(r"\.entry\s+frisk_execution\b", ptx)) != 1 or re.search(r"\.func\b", ptx): raise ValueError("attention-ptx-functions")
    group = []
    for i, (op, operands) in enumerate(inst):
        if op.startswith("wgmma."):
            group.append(op)
            if op.startswith("wgmma.mma_async."):
                match = re.fullmatch(r"wgmma\.mma_async\.sync\.aligned\.m64n(16|64)k16\.f32\.f16\.f16", op)
                if not match or inst[i+1:i+3] != [("wgmma.commit_group.sync.aligned", ""), ("wgmma.wait_group.sync.aligned", "0")]: raise ValueError("attention-not-immediate-wait0")
                features["wgmma_m64n" + match[1] + "k16_sites"] += 1
        if op == "wgmma.fence.sync.aligned": features["wgmma_fence_sites"] += 1
        if op == "wgmma.commit_group.sync.aligned": features["wgmma_commit_sites"] += 1
        if op == "wgmma.wait_group.sync.aligned" and operands == "0": features["wgmma_wait0_sites"] += 1
        if op == "mov.b32" and re.search(r",\s*\{[^}]*,[^}]*\}", operands): features["ptx_f16_pack_sites"] += 1
        if op.startswith("cp.async.") and op not in {"cp.async.ca.shared.global", "cp.async.commit_group", "cp.async.wait_group", "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"}: raise ValueError("attention-unknown-copy-instruction")
        if op == "cp.async.ca.shared.global":
            if not re.search(r",\s*16$", operands): raise ValueError("attention-cpasync-width")
            features["cpasync_copy_sites"] += 1; features["cpasync_operand_bytes"] += 16
        if op == "cp.async.commit_group": features["cpasync_commit_sites"] += 1
        if op == "cp.async.wait_group":
            if operands != "0": raise ValueError("attention-cpasync-wait")
            features["cpasync_wait0_sites"] += 1
        if op.startswith("cp.async.bulk.tensor."): features["tma_copy_sites"] += 1
        if op.startswith("mbarrier.init."): features["mbarrier_init_sites"] += 1
        if op.startswith("mbarrier.arrive.expect_tx."):
            if not re.search(r",\s*2048$", operands): raise ValueError("attention-tma-transaction-size")
            features["mbarrier_arrive_sites"] += 1; features["tma_declared_transaction_bytes"] += 2048
        if op.startswith("mbarrier.try_wait."): features["mbarrier_try_wait_sites"] += 1
        if op == "bar.sync": features["ptx_cta_barrier_sites"] += 1
        if op == "fence.proxy.async.shared::cta": features["async_proxy_fence_sites"] += 1
    n = sum(expected.values())
    if len(group) != 4 * n: raise ValueError("attention-wgmma-group-grammar")
    for i in range(n):
        g = group[i * 4:(i + 1) * 4]
        if g[0] != "wgmma.fence.sync.aligned" or not g[1].startswith("wgmma.mma_async.") or g[2:] != ["wgmma.commit_group.sync.aligned", "wgmma.wait_group.sync.aligned"]: raise ValueError("attention-wgmma-group-grammar")
    if any(features[k] != v for k, v in expected.items()) or features["wgmma_commit_sites"] != n or features["wgmma_wait0_sites"] != n: raise ValueError("attention-mma-packet-ptx-mismatch")
    h = native_summary["proof"]["opcode_histogram"]
    pred_hist = native_summary["proof"].get("predicated_opcode_histogram", {})
    if any(v and (k.startswith("HGMMA.") or k.startswith("WARPGROUP.")) for k, v in pred_hist.items()): raise ValueError("attention-predicated-native-tensor")
    features["native_hgmma_sites"] = sum(v for k, v in h.items() if k.startswith("HGMMA."))
    features["native_warpgroup_wait_sites"] = h.get("WARPGROUP.DEPBAR.LE", 0)
    features["native_f16_pack_sites"] = h.get("F2FP.F16.F32.PACK_AB", 0)
    features["native_mbarrier_try_wait_sites"] = h.get("SYNCS.PHASECHK.TRANS64.TRYWAIT", 0)
    features["polling_sccs"] = sum(len(f["polling_sccs"]) for f in native_summary["proof"]["functions"])
    if features["native_hgmma_sites"] != n or features["native_warpgroup_wait_sites"] != n: raise ValueError("attention-native-tensor-sequence-mismatch")
    if path == "vector":
        features["vector_copy_operations"] = len(copies)
        if features["cpasync_copy_sites"] or features["tma_copy_sites"]: raise ValueError("attention-vector-path-mismatch")
    if path == "cp.async":
        if sum(v for k, v in h.items() if k.startswith("LDGSTS.")) != len(copies) or h.get("LDGDEPBAR", 0) != len(copies) or h.get("DEPBAR.LE", 0) != len(copies): raise ValueError("attention-native-cpasync-sequence-mismatch")
        if any(features[k] != len(copies) for k in ("cpasync_copy_sites", "cpasync_commit_sites", "cpasync_wait0_sites")) or features["tma_copy_sites"]: raise ValueError("attention-cpasync-packet-ptx-mismatch")
    if path == "tma":
        if h.get("UTMALDG.2D", 0) != len(copies) or h.get("SYNCS.EXCH.64", 0) != len(copies) or h.get("SYNCS.ARRIVE.TRANS64", 0) != len(copies): raise ValueError("attention-native-tma-sequence-mismatch")
        if any(features[k] != len(copies) for k in ("tma_copy_sites", "mbarrier_init_sites", "mbarrier_arrive_sites", "mbarrier_try_wait_sites", "polling_sccs")) or features["cpasync_copy_sites"]: raise ValueError("attention-tma-packet-ptx-mismatch")
    return features, path


def summarize(directory, *, entry="frisk_execution", nvdisasm="/usr/local/cuda/bin/nvdisasm", document_provider=None):
    """Read-only CPU analysis with CUBIN, tools and full lineage SHA binding."""
    root = Path(directory)
    provenance = {"analyzer_sha256": SOURCE_SHA256, "dependencies": {}}
    try:
        if _sha(Path(__file__).read_bytes()) != SOURCE_SHA256: raise ValueError("analyzer-source-changed-since-import")
        required = ("kernel.cubin", "execution.json", "kernel.ttgir", "kernel.ptx", "execution_config.json", "compile.json", "image-identity.json")
        raw = {k: (root / k).read_bytes() for k in required}
        provenance["dependencies"] = {k: _sha(v) for k, v in raw.items()}
        identity = json.loads(raw["image-identity.json"]); receipt = json.loads(raw["compile.json"])
        for name in ("kernel.cubin", "execution.json", "kernel.ttgir", "kernel.ptx", "execution_config.json"):
            if identity.get(name) != _sha(raw[name]): raise ValueError("image-identity-mismatch:" + name)
        if receipt.get("cubin_sha256") != _sha(raw["kernel.cubin"]) or receipt.get("execution_sha256") != _sha(raw["execution.json"]): raise ValueError("compile-receipt-mismatch")
        tool = Path(nvdisasm).resolve(strict=True)
        tool_hash = _sha(tool.read_bytes())
        if document_provider is None:
            provenance["nvdisasm"] = {"path": str(tool), "sha256": tool_hash, "version": subprocess.check_output([str(tool), "--version"], text=True, timeout=30)}
            native_raw = subprocess.check_output([str(tool), "-json", str(root / "kernel.cubin")], stderr=subprocess.PIPE, timeout=120)
        else:
            provenance["nvdisasm"] = document_provider.tool(tool)
            if provenance["nvdisasm"]["sha256"] != tool_hash: raise ValueError("native-provider-tool-mismatch")
            native_raw = document_provider.output(provenance["nvdisasm"], ["-json", str(root / "kernel.cubin")], root / "kernel.cubin")
        provenance["native_json_sha256"] = _sha(native_raw)
        execution = json.loads(raw["execution.json"])
        attention = any(x.get("name") == "frisk.mma" for x in execution["operations"])
        result = (analyze_document(json.loads(native_raw), entry=entry, allow_attention=attention) if document_provider is None else
                  document_provider.pure(analyze_document, json.loads(native_raw), entry=entry, allow_attention=attention))
        if result["status"] == "complete" and attention:
            extra, path = (attention_features(execution, raw["kernel.ptx"].decode(), result) if document_provider is None else
                           document_provider.pure(attention_features, execution, raw["kernel.ptx"].decode(), result))
            result["features"].update(extra)
            result["feature_names"] += list(ATTENTION_FEATURES)
            result["feature_units"].update({k: UNITS[k] for k in ATTENTION_FEATURES})
            result["attention_abi"] = path
            # Native per-thread bounds do not apply to mixed PTX/ABI units.
            result["execution_bounds"]["feature_scope"] = list(BASE_FEATURES)
        for name, expected in provenance["dependencies"].items():
            if _sha((root / name).read_bytes()) != expected: raise ValueError("input-changed-during-analysis:" + name)
        if _sha(tool.read_bytes()) != tool_hash: raise ValueError("native-tool-changed-during-analysis")
        result["provenance"] = provenance
        result["resource_policy"] = "REG/LOCAL/SHARED driver facts remain separate; no driver frame bytes are interpreted as dynamic transfers"
        result["summary_sha256"] = _sha(_canonical(result))
        return result
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError, IndexError) as error:
        return _unknown(str(error), provenance=provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), sort_keys=True, indent=2))
