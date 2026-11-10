"""Post-ptxas surviving work and bounded native schedule evidence.

This is compilation feedback, not a pre-compilation layout predictor.  Counts
are static instruction occurrences/operand widths in one entry, not executed
lane counts, transactions, cycle estimates, or measured hardware ILP.  Native
control flow, subroutines, async/tensor operations and unknown opcodes fail
closed for scoring.  No GPU execution occurs in this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

FEATURES = (
    "native_instructions", "global_load_bytes", "global_store_bytes",
    "shared_load_bytes", "shared_store_bytes", "local_load_bytes",
    "local_store_bytes", "barrier_instructions", "shuffle_instructions",
    "float_arithmetic_instructions", "special_math_instructions",
    "integer_logic_instructions", "load_use_exposure_32",
    "load_use_exposure_128", "dependency_depth",
)
SCHEMA = 1
ANALYZER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DOMAIN = "sm90-scalar-straight-line-native-v2"
MAX_INSTRUCTIONS = 262144
MAX_EDGES = 2097152
CONTROL = {"BRA", "BRX", "BRXU", "JMP", "JMX", "CALL", "RET", "BSSY", "BSYNC", "BREAK", "CONT", "YIELD"}
# Explicit operand semantics below cover only these scalar opcodes.  Adding an
# opcode requires a semantic test; an unrecognised mnemonic is never free work.
ONE_DEST = {"LDC", "ULDC", "S2R", "S2UR", "MOV", "UMOV", "IMAD", "UIMAD", "LOP3", "ULOP3", "SHF", "USHF", "SGXT", "PRMT", "SEL", "FSEL", "P2R", "R2P", "F2I", "I2F", "I2I", "F2F", "F2FP", "FMUL", "FADD", "FFMA", "FMNMX", "MUFU", "RRO", "FLO", "POPC", "BREV", "BMSK", "UBMSK", "VIADD", "FCHK"}
CARRY_DEST = {"IADD3", "UIADD3", "LEA", "ULEA"}
TWO_DEST = {"ISETP", "UISETP", "FSETP", "DSETP", "PSETP", "SHFL", "VOTE", "VOTEU"}
LOADS = {"LDG": "global", "LDS": "shared", "LDL": "local"}
STORES = {"STG": "global", "STS": "shared", "STL": "local"}
FLOAT = {"FMUL", "FADD", "FFMA", "FMNMX", "FSEL"}
SPECIAL = {"MUFU", "RRO"}
INTEGER = (ONE_DEST | CARRY_DEST | TWO_DEST) - FLOAT - SPECIAL - {"LDC", "ULDC", "S2R", "S2UR", "SHFL"}
REGEX = re.compile(r"(?<![A-Za-z_0-9])(?P<reg>UR(?:Z|[0-9]+)|UP[0-9]+|R(?:Z|[0-9]+)|P[0-9]+|PT)(?P<width>\.(?:64|128))?(?![A-Za-z_0-9])")

def _sha(data):
    return hashlib.sha256(data).hexdigest()

def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

def _width(opcode):
    modifiers = opcode.split(".")[1:]
    for size in (128, 64, 16, 8):
        if str(size) in modifiers or "U" + str(size) in modifiers or "S" + str(size) in modifiers:
            return size // 8
    return 4

def _regs(operand, words=1):
    out = set()
    operand = re.sub(r"desc\[(UR[0-9]+)\]", r"desc[\1.64]", operand)
    for match in REGEX.finditer(operand):
        reg = match.group("reg")
        if reg in {"RZ", "URZ", "PT"}: continue
        prefix = re.match(r"[A-Z]+", reg).group()
        count = int(match.group("width")[1:]) // 32 if match.group("width") else words
        # Predicates never denote a vector of registers.
        if prefix in {"P", "UP"}: count = 1
        index = int(reg[len(prefix):])
        out.update(prefix + str(index + i) for i in range(count))
    return out

def _split_operands(text):
    # Commas inside grouped operands are not part of the bounded scalar domain.
    # nvdisasm12.8 omits the separating comma for some LDL instructions.
    text = re.sub(r"^((?:UR|R)[0-9]+)(?=\[)", r"\1,", text)
    return [s.strip() for s in text.split(",") if s.strip()]

def _execution_predicate(instruction):
    # nvdisasm12.8 emits the PR *data source* of P2R in its predicate field,
    # omitting PR from operands. It is not an instruction execution guard.
    predicate = instruction.get("predicate", "")
    return "" if instruction.get("opcode") == "P2R" and predicate == "PR" else predicate


def _predicate_move(instruction, operands):
    """Conservative aggregate-PR dependencies; not mask-exact hardware semantics.

    NVIDIA Binary Utilities documents the transfer direction but not complete
    mask/byte encoding. The narrow unsuffixed PR/GPR/immediate forms below read
    all mutable P0..P6; R2P also writes all and retains old state dependencies.
    This is a safe DAG overapproximation, never a precise latency description.
    Shifted registers, modifiers, uniform PR and other forms remain unsupported.
    """
    opcode = instruction["opcode"]
    predicate = _execution_predicate(instruction)
    if predicate and not re.fullmatch(r"@?!?P[0-6]", predicate):
        raise ValueError("unknown-predicate-move-guard:" + opcode)
    if opcode == "P2R" and instruction.get("predicate") == "PR":
        if len(operands) != 3:
            raise ValueError("unknown-predicate-move-form:" + opcode)
        operands = [operands[0], "PR", *operands[1:]]
    valid = (opcode == "R2P" and len(operands) == 3 and operands[0] == "PR"
             and re.fullmatch(r"R(?:[0-9]+|Z)", operands[1])) or (
             opcode == "P2R" and len(operands) == 4
             and re.fullmatch(r"R[0-9]+", operands[0]) and operands[1] == "PR"
             and re.fullmatch(r"R(?:[0-9]+|Z)", operands[2]))
    if not valid or not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)", operands[-1]):
        raise ValueError("unknown-predicate-move-form:" + opcode)
    mask = int(operands[-1], 16 if operands[-1].lower().startswith("0x") else 10)
    if not 0 <= mask <= 255:
        raise ValueError("unknown-predicate-move-mask:" + opcode)
    aggregate = {"P" + str(i) for i in range(7)}
    if opcode == "R2P":
        writes = set(aggregate)
        reads = aggregate | _regs(operands[1]) | _regs(predicate)
    else:
        writes = _regs(operands[0])
        reads = aggregate | _regs(operands[2]) | _regs(predicate)
        if predicate: reads |= writes
    return {"opcode": opcode, "base": opcode, "reads": reads, "writes": writes,
            "scope": None, "store": False}


def _decode(instruction):
    opcode = instruction.get("opcode", "")
    base = opcode.split(".")[0]
    operands = _split_operands(instruction.get("operands", ""))
    predicate = _execution_predicate(instruction)
    if base in {"P2R", "R2P"}:
        return _predicate_move(instruction, operands)
    if any(x in instruction.get("operands", "") for x in ("{", "}")):
        raise ValueError("grouped-operands:" + opcode)
    if base in CONTROL or instruction.get("other-attributes", {}).get("subroutine-call") == "True":
        raise ValueError("control-flow:" + opcode)
    if base == "EXIT":
        if predicate: raise ValueError("predicated-exit")
        return {"opcode": opcode, "base": base, "reads": set(), "writes": set(), "scope": None, "store": False}
    if base in {"BAR", "MEMBAR"}:
        if predicate or (base == "BAR" and not opcode.startswith("BAR.SYNC")):
            raise ValueError("unsupported-barrier:" + opcode)
        return {"opcode": opcode, "base": base, "reads": set(), "writes": set(), "scope": None, "store": False}
    if base == "NOP":
        return {"opcode": opcode, "base": base, "reads": set(), "writes": set(), "scope": None, "store": False}
    if base not in ONE_DEST | CARRY_DEST | TWO_DEST | LOADS.keys() | STORES.keys():
        raise ValueError("unknown-opcode:" + opcode)
    if not operands: raise ValueError("missing-operands:" + opcode)
    destinations = 0 if base in STORES else 2 if base in TWO_DEST else 1
    if base in CARRY_DEST:
        while destinations < len(operands) and re.fullmatch(r"(?:U?P[0-9]+|PT)", operands[destinations]):
            destinations += 1
    if len(operands) <= destinations:
        raise ValueError("missing-inputs:" + opcode)
    words = max(1, _width(opcode) // 4) if base in LOADS or ".WIDE" in opcode or base in {"LDC", "ULDC", "MOV", "UMOV"} else 1
    writes = set().union(*(_regs(s, words if i == 0 else 1) for i, s in enumerate(operands[:destinations]))) if destinations else set()
    reads = _regs(predicate)
    for i, operand in enumerate(operands[destinations:]):
        # Address suffix Rn.64 expands the pair. A 64/128bit store also reads
        # its complete data register range, not just the printed first register.
        data_words = max(1, _width(opcode) // 4) if base in STORES and i == 1 else 1
        reads |= _regs(operand, data_words)
    if predicate: reads |= writes  # conditional def retains the old value
    return {"opcode": opcode, "base": base, "reads": reads, "writes": writes, "scope": LOADS.get(base, STORES.get(base)), "store": base in STORES}

def _resources(text, entry):
    pieces = re.split(r"Function\s+([^:]+):", text)
    for i in range(1, len(pieces), 2):
        if pieces[i].strip() != entry: continue
        row = {name: int(value) for name, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):([0-9]+)", pieces[i+1])}
        if set(row) != {"REG", "STACK", "SHARED", "LOCAL"}: raise ValueError("incomplete-resource-record")
        return {"registers_per_thread": row["REG"], "stack_bytes_per_thread": row["STACK"], "raw_shared_bytes": row["SHARED"], "raw_local_bytes": row["LOCAL"], "spill_count": None,
                "meaning": "compiler-exact static resource fields; STACK/LOCAL are bytes, not spill counts; SHARED may include reserved bytes; nonzero STACK/LOCAL needs driver confirmation"}
    raise ValueError("entry-resource-record-missing")

def analyze_disassembly(document, resource_text, entry="frisk_execution"):
    """Analyse native JSON without invoking a tool (also the testable core)."""
    reasons = []
    if not isinstance(document, list) or len(document) != 2 or not isinstance(document[0], dict) or not isinstance(document[1], list):
        return {"status": "unsupported", "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["invalid-native-json-schema"]}, "features": None}
    metadata = document[0]
    if not isinstance(metadata.get("SM", {}), dict) or not isinstance(metadata.get("SchemaVersion", {}), dict) or any(not isinstance(f, dict) or not isinstance(f.get("function-name"), str) or not isinstance(f.get("sass-instructions"), list) or any(not isinstance(x, dict) or not isinstance(x.get("opcode", ""), str) or not isinstance(x.get("operands", ""), str) or not isinstance(x.get("predicate", ""), str) or not isinstance(x.get("other-attributes", {}), dict) for x in f.get("sass-instructions", [])) for f in document[1]):
        return {"status": "unsupported", "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["invalid-native-json-fields"]}, "features": None}
    if metadata.get("SM", {}).get("version") != {"major": 9, "minor": 0}: reasons.append("architecture-not-sm90")
    if metadata.get("SchemaVersion", {}).get("major") not in {12, 13}: reasons.append("unsupported-native-json-version")
    entries = [f for f in document[1] if f.get("function-name") == entry]
    if len(entries) != 1:
        return {"status": "unsupported", "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["entry-missing-or-ambiguous"]}, "features": None}
    raw = entries[0].get("sass-instructions", [])
    if not raw or len(raw) > MAX_INSTRUCTIONS: reasons.append("native-instruction-budget")
    main = []
    exit_seen = False
    for instruction in raw[:MAX_INSTRUCTIONS]:
        if not instruction.get("opcode"): reasons.append("native-code-gap"); continue
        main.append(instruction)
        if instruction["opcode"] == "EXIT" and not instruction.get("predicate"):
            exit_seen = True; break
    if not exit_seen: reasons.append("unconditional-exit-missing")
    counts = {name: 0 for name in FEATURES}
    histogram = {}
    decoded = []
    for instruction in main:
        op = instruction["opcode"]; base = op.split(".")[0]
        histogram[op] = histogram.get(op, 0) + 1
        counts["native_instructions"] += 1
        if base in LOADS: counts[LOADS[base] + "_load_bytes"] += _width(op)
        if base in STORES: counts[STORES[base] + "_store_bytes"] += _width(op)
        if base in {"BAR", "MEMBAR"}: counts["barrier_instructions"] += 1
        if base == "SHFL": counts["shuffle_instructions"] += 1
        if base in FLOAT: counts["float_arithmetic_instructions"] += 1
        if base in SPECIAL: counts["special_math_instructions"] += 1
        if base in INTEGER: counts["integer_logic_instructions"] += 1
        try: decoded.append(_decode(instruction))
        except ValueError as error: reasons.append(str(error))
    try: resources = _resources(resource_text, entry)
    except ValueError as error: reasons.append(str(error)); resources = None
    diagnostics = {"opcode_histogram": dict(sorted(histogram.items())), "static_occurrence_counts": counts.copy(), "resources": resources,
                   "instructions_after_first_unconditional_exit": len(raw) - len(main), "predicated_instructions": sum(bool(_execution_predicate(x)) for x in main)}
    support = {"status": "unsupported" if reasons else "supported", "domain": DOMAIN, "reasons": sorted(set(reasons))}
    if reasons:
        return {"status": "unsupported", "support": support, "features": None, "diagnostics": diagnostics, "native_metadata": metadata}
    # Physical register dependencies, memory ordering by address space, and full
    # CTA/memory fences. Edges are unit constraints, never hardware latencies.
    last_writer, pending_readers = {}, {}
    memory_store = {}; memory_reads = {}
    barrier = None; phase = []
    successors = [[] for _ in decoded]
    predecessors = []
    raw_depths = []; depths = []
    first_use = {}; load_words = {}; outstanding = set(); pending_peak = 0
    for i, instruction in enumerate(decoded):
        deps = set(); raw_deps = set()
        for reg in instruction["reads"]:
            if reg in last_writer:
                writer = last_writer[reg]; deps.add(writer); raw_deps.add(writer)
                if writer in load_words:
                    first_use.setdefault(writer, i)
                    outstanding.discard(writer)
        for reg in instruction["writes"]:
            if reg in last_writer: deps.add(last_writer[reg])
            deps.update(pending_readers.get(reg, ()))
        if barrier is not None: deps.add(barrier)
        if instruction["base"] in {"BAR", "MEMBAR", "EXIT"}:
            deps.update(phase)
            phase = []
            if instruction["base"] != "EXIT": barrier = i
        scope = instruction["scope"]
        if scope:
            if scope in memory_store: deps.add(memory_store[scope])
            if instruction["store"]:
                deps.update(memory_reads.get(scope, ()))
                memory_reads[scope] = []
                memory_store[scope] = i
            else: memory_reads.setdefault(scope, []).append(i)
        deps.discard(i)
        predecessors.append(sorted(deps))
        for parent in deps: successors[parent].append(i)
        if len(deps) + diagnostics.get("dependency_edges", 0) > MAX_EDGES:
            support["status"] = "unsupported"; support["reasons"].append("dependency-edge-budget")
            return {"status": "unsupported", "support": support, "features": None, "diagnostics": diagnostics, "native_metadata": metadata}
        diagnostics["dependency_edges"] = diagnostics.get("dependency_edges", 0) + len(deps)
        depths.append(1 + max((depths[p] for p in deps), default=0))
        raw_depths.append(1 + max((raw_depths[p] for p in raw_deps), default=0))
        for reg in instruction["writes"]:
            last_writer[reg] = i; pending_readers[reg] = set()
        for reg in instruction["reads"] - instruction["writes"]: pending_readers.setdefault(reg, set()).add(i)
        if instruction["base"] in LOADS:
            load_words[i] = instruction["writes"]; outstanding.add(i)
        pending_peak = max(pending_peak, len(outstanding))
        phase.append(i)
    distances = [use - index for index, use in sorted(first_use.items())]
    for window in (32, 128): counts["load_use_exposure_" + str(window)] = sum(max(0, window - distance) for distance in distances)
    counts["dependency_depth"] = max(depths, default=0)
    # Exact list-order ready set of the conservative unit DAG. This is a graph
    # statistic, not simultaneous GPU issue width or a claim of hardware ILP.
    ready = {i for i, parents in enumerate(predecessors) if not parents}
    indegree = [len(x) for x in predecessors]; widths = []
    for i in range(len(decoded)):
        if i not in ready: raise AssertionError("native dependency graph violates program order")
        widths.append(len(ready)); ready.remove(i)
        for child in successors[i]:
            indegree[child] -= 1
            if not indegree[child]: ready.add(child)
    first_math = next((i for i, x in enumerate(decoded) if x["base"] in FLOAT | SPECIAL), len(decoded))
    diagnostics.update({"raw_dependency_depth": max(raw_depths, default=0), "dependency_depth_unit": counts["dependency_depth"],
                        "ready_width_peak": max(widths, default=0), "ready_width_mean": sum(widths) / max(1, len(widths)),
                        "pending_load_peak": pending_peak, "load_first_use_distances": distances, "loads_without_register_use": len(load_words) - len(first_use),
                        "global_load_instructions_before_first_float_math": sum(x["base"] == "LDG" for x in decoded[:first_math])})
    return {"status": "available", "support": support, "features": counts, "diagnostics": diagnostics, "native_metadata": metadata,
            "semantics": "postcompile static native occurrences; bytes are declared operand width per instruction, predication is not lane-frequency modelled; dependency depth/ready width/exposure are unit graph/order statistics; aggregate PR edges conservatively overapproximate mutable P0..P6, not mask-exact hardware dependencies, cycles, dynamic work, or measured hardware ILP"}

def _tool(path):
    absolute = Path(path).resolve(strict=True)
    version = subprocess.check_output([str(absolute), "--version"], text=True, timeout=30)
    return {"path": str(absolute), "sha256": _sha(absolute.read_bytes()), "version": version}

def analyze(directory, cache_dir=None, *, entry="frisk_execution", nvdisasm="/usr/local/cuda/bin/nvdisasm", cuobjdump="/usr/local/cuda/bin/cuobjdump"):
    """Read kernel.cubin; optionally cache feedback bound to image+tools+source.

    Cache hits are verified by a payload digest and exact identity. Compilation
    is the caller's budgeted operation; no new image is compiled here. Failed
    tools/unsupported schedules return typed failure with no scoreable features.
    """
    source = Path(directory)
    cubin = source / "kernel.cubin" if source.is_dir() else source
    identity = {"schema_version": SCHEMA, "domain": DOMAIN, "entry": entry,
                "cubin_sha256": _sha(cubin.read_bytes()), "analyzer_sha256": ANALYZER_SOURCE_SHA256}
    if _sha(Path(__file__).read_bytes()) != ANALYZER_SOURCE_SHA256:
        return {"schema_version": SCHEMA, "status": "unavailable", "features": None, "identity": identity,
                "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["analyzer-source-changed-since-import"]}}
    try:
        identity["nvdisasm"] = _tool(nvdisasm); identity["cuobjdump"] = _tool(cuobjdump)
    except (OSError, subprocess.SubprocessError) as error:
        return {"schema_version": SCHEMA, "status": "unavailable", "features": None, "identity": identity,
                "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["native-tool-unavailable:" + type(error).__name__]}}
    key = _sha(_canonical(identity))
    cache = Path(cache_dir) / (key + ".json") if cache_dir is not None else None
    if cache is not None and cache.is_file():
        try:
            envelope = json.loads(cache.read_text()); payload = envelope["payload"]
            if payload["identity"] == identity and envelope["payload_sha256"] == _sha(_canonical(payload)):
                return dict(payload, cache_key=key, cache_hit=True)
        except (ValueError, KeyError, TypeError): pass  # corrupted cache is never trusted
    try:
        native = subprocess.check_output([identity["nvdisasm"]["path"], "-json", str(cubin)], text=True, stderr=subprocess.PIPE, timeout=60)
        resources = subprocess.check_output([identity["cuobjdump"]["path"], "--dump-resource-usage", str(cubin)], text=True, stderr=subprocess.PIPE, timeout=60)
        result = analyze_disassembly(json.loads(native), resources, entry)
        result.update({"native_json_sha256": _sha(native.encode()), "resource_text_sha256": _sha(resources.encode())})
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        result = {"status": "unavailable", "features": None, "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["native-analysis-failed:" + type(error).__name__]}}
    # A source image/tool changing while inspected invalidates the feedback.
    if _sha(cubin.read_bytes()) != identity["cubin_sha256"] or any(_sha(Path(identity[t]["path"]).read_bytes()) != identity[t]["sha256"] for t in ("nvdisasm", "cuobjdump")):
        result = {"status": "unavailable", "features": None, "support": {"status": "unsupported", "domain": DOMAIN, "reasons": ["native-input-changed-during-analysis"]}}
    payload = dict(result, schema_version=SCHEMA, identity=identity)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + ".tmp")
        temporary.write_text(json.dumps({"payload": payload, "payload_sha256": _sha(_canonical(payload))}, indent=2, sort_keys=True) + "\n")
        temporary.replace(cache)
    return dict(payload, cache_key=key, cache_hit=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory"); parser.add_argument("--cache-dir"); parser.add_argument("--entry", default="frisk_execution"); parser.add_argument("--output")
    args = parser.parse_args()
    result = analyze(args.directory, args.cache_dir, entry=args.entry)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output: Path(args.output).write_text(text)
    else: print(text, end="")
