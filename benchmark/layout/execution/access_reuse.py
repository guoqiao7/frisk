"""Checked GF(2) register-query-window locality potential for scalar f32.

Counts describe relative-base-zero address block sets in an exactly re-emitted
source program. They are neither native execution order, cache/TLB misses nor
latency. A whole weak ABI does not promise 32/128/4096-byte base alignment.
Warp-local sums and CTA unions remain distinct: warp image cosets may overlap.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import backend
import demand_coverage_audit as demand
import execution_support
import native_surviving_audit as survival
import storage_addresses

CONTRACT = "frisk-gf2-checked-address-window-locality-v1"
FAMILY = "gf2-128thread-register-window-locality-v1"
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
WINDOWS = (1, 2, 4, 8, 16, 32)
GRANULES = (32, 128, 4096)
SCOPES = ("warp", "cta")
FEATURE_NAMES = tuple(f"gf2_reuse_{scope}_b{b}_w{w}_{metric}" for scope in SCOPES for b in GRANULES for w in WINDOWS for metric in ("footprint", "potential"))
FOOTPRINT_FEATURES = tuple(k for k in FEATURE_NAMES if k.endswith("_footprint"))
UNITS = {k: ("sum of distinct relative-base-zero address blocks over source access/root, disjoint query windows and hardware scope; not misses" if k.endswith("_footprint") else "sum of per-query block counts minus query-window union counts; diagnostic repeat potential, not a nonnegative cost penalty") for k in FEATURE_NAMES}
SOURCE_FILES = ("backend.py", "redistribution.py", "shuffle_network.py", "execution_support.py", "storage_addresses.py", "demand_coverage_audit.py", "native_surviving_audit.py", "async_copy.py", "async_host.py", "shared_allocation.py", "access_plan.py", "reduction_plan.py")
SOURCE_HASHES = {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest() for n in SOURCE_FILES}
MAX_QUERY_SITES = 65536
MAX_ENUMERATED_BLOCK_TAGS = 8388608


def _sha(data): return hashlib.sha256(data).hexdigest()

def _canonical(data): return json.dumps(data, sort_keys=True, separators=(",", ":")).encode()

def _power(n): return type(n) is int and n > 0 and not n & (n - 1)


def _apply(columns, x):
    value = 0
    if type(x) is not int or x < 0 or x.bit_length() > len(columns): raise ValueError("map input exceeds proven bits")
    while x:
        low = x & -x; value ^= columns[low.bit_length() - 1]; x ^= low
    return value


def basis(columns):
    """Canonical XOR image basis, retaining exact cardinality via rank."""
    pivots = {}
    for column in columns:
        if type(column) is not int or column < 0: raise ValueError("nonnegative GF2 column required")
        while column:
            bit = column.bit_length() - 1
            if bit in pivots: column ^= pivots[bit]
            else: pivots[bit] = column; break
    # Reduced pivots make quotient coset representatives deterministic.
    for bit in sorted(pivots):
        for other in sorted(pivots):
            if other > bit and pivots[other] >> bit & 1: pivots[other] ^= pivots[bit]
    return tuple(pivots[b] for b in sorted(pivots, reverse=True))


def reduce_coset(value, image):
    for pivot in image:
        if value >> (pivot.bit_length() - 1) & 1: value ^= pivot
    return value


def _image_values(image):
    result = [0]
    for column in image: result += [x ^ column for x in result]
    return result


def count_windows(query_maps, window, granule, scope):
    """Exact address-set counts using image rank and quotient-coset unions.

    Each map has one affine constant and seven thread columns, in byte units.
    Four consecutive bytes of an f32 stay in one block at relative base zero.
    Generic fallback enumerates only projected image tags, never hardware slots.
    """
    q = len(query_maps)
    if not q or q > MAX_QUERY_SITES or window not in WINDOWS or granule not in GRANULES or scope not in SCOPES: raise ValueError("invalid locality window domain")
    if any(type(m) not in (tuple, list) or len(m) != 2 or type(m[0]) is not int or m[0] < 0 or m[0] % 4 or type(m[1]) not in (tuple, list) or len(m[1]) != 7 or any(type(c) is not int or c < 0 or c % 4 for c in m[1]) for m in query_maps): raise ValueError("f32-aligned byte affine maps required")
    shift = granule.bit_length() - 1; width = min(q, window)
    lanes = 5 if scope == "warp" else 7
    warp_ids = range(4) if scope == "warp" else range(1)
    sums, single_sums, method, enumerated = [0] * len(warp_ids), [0] * len(warp_ids), Counter(), 0
    # A global affine q-axis permits folding all aligned windows by image rank.
    constants = [m[0] for m in query_maps]
    common_threads = all(m[1] == query_maps[0][1] for m in query_maps)
    affine = None
    if _power(q) and common_threads:
        try: affine = backend.linear(constants)
        except ValueError: pass
    if affine is not None and _power(width):
        _, qcols = affine
        columns = query_maps[0][1]
        rank1 = len(basis([c >> shift for c in columns[:lanes]]))
        rankw = len(basis([c >> shift for c in columns[:lanes]] + [c >> shift for c in qcols[:width.bit_length() - 1]]))
        windows = q // width
        for wi in range(len(warp_ids)):
            sums[wi] = windows * (1 << rankw)
            single_sums[wi] = q * (1 << rank1)
        method["affine-image-rank"] = windows * len(warp_ids)
        ranks = dict(single_query_rank=rank1, window_rank=rankw)
    else:
        ranks = None
        for wi, warp in enumerate(warp_ids):
            for start in range(0, q, width):
                subset = query_maps[start:start + width]
                images = [basis([c >> shift for c in m[1][:lanes]]) for m in subset]
                offsets = [(m[0] ^ (_apply(m[1][5:], warp) if scope == "warp" else 0)) >> shift for m in subset]
                single_sums[wi] += sum(1 << len(image) for image in images)
                if all(image == images[0] for image in images):
                    cosets = {reduce_coset(offset, images[0]) for offset in offsets}
                    sums[wi] += len(cosets) * (1 << len(images[0])); method["common-image-quotient-cosets"] += 1
                else:
                    tags = set()
                    for offset, image in zip(offsets, images):
                        enumerated += 1 << len(image)
                        if enumerated > MAX_ENUMERATED_BLOCK_TAGS: raise ValueError("projected-block-tag-enumeration-budget")
                        tags.update(offset ^ x for x in _image_values(image))
                    sums[wi] += len(tags); method["bounded-projected-image-tag-union"] += 1
    footprint = sum(sums); potential = sum(single_sums) - footprint
    if potential < 0: raise AssertionError("union larger than summed members")
    return dict(footprint=footprint, potential=potential, effective_window=width, windows=(q + width - 1) // width,
                per_scope_footprint=sums, per_scope_potential=[a-b for a,b in zip(single_sums,sums)],
                methods=dict(method), projected_tags_enumerated=enumerated, ranks=ranks)


def _map_columns(enc):
    topology, shape = enc.get("topology"), enc.get("shape")
    if type(topology) is not list or len(topology) != 5 or topology[1:] != [32, 4, 1, 1] or not _power(topology[0]): raise ValueError("128-thread dyadic R topology required")
    if type(shape) is not list or len(shape) != 2 or any(not _power(x) for x in shape) or math.prod(shape) > 65536: raise ValueError("dyadic rank2 <=65536 logical elements required")
    if "symbolic" in enc:
        mapping = execution_support.SymbolicMap(enc)
        return 0, tuple(mapping.flatten(c) for c in mapping.columns), "checked-symbolic-map"
    logical = enc.get("logical")
    if type(logical) is not list or len(logical) != 128 * topology[0] or len(logical) > 65536: raise ValueError("explicit map exceeds bounded proof domain")
    if any(type(x) is not int or not 0 <= x < math.prod(shape) for x in logical): raise ValueError("explicit map logical range")
    base, columns = backend.linear(logical)
    if len(basis(columns)) != math.prod(shape).bit_length() - 1: raise ValueError("map does not cover logical image")
    replication = len(logical) // math.prod(shape)
    if type(enc.get("replication")) is not int or enc["replication"] != replication: raise ValueError("explicit map replication")
    return base, tuple(columns), "full-explicit-table-affine-verification-once"


def _storage(values, ident, shape):
    value = values[ident]; storage = value["storage"]; root = storage["root"]
    if storage_addresses._shape(value) != tuple(shape): raise ValueError("storage logical shape mismatch")
    table = storage_addresses.RowMajorAddresses(dict(contract=storage_addresses.CONTRACT, shape=list(shape), element_bits=32, base_bit=0, strides_bits=[32 * shape[1], 32]))
    storage_addresses._metadata(value, table)
    addresses = storage.get("addresses")
    if type(addresses) is storage_addresses.RowMajorAddresses:
        if addresses.complete_state() != table.complete_state(): raise ValueError("compact storage geometry mismatch")
    elif not storage_addresses._dense_equals(table, addresses): raise ValueError("noncontiguous or aliased storage")
    if root not in values or values[root]["storage"].get("root") != root: raise ValueError("canonical Global root missing")
    root_storage = values[root]["storage"]
    storage_addresses._metadata(values[root], table)
    root_addresses = root_storage.get("addresses")
    if type(root_addresses) is storage_addresses.RowMajorAddresses:
        if root_addresses.complete_state() != table.complete_state(): raise ValueError("root storage geometry mismatch")
    elif not storage_addresses._dense_equals(table, root_addresses): raise ValueError("root storage not contiguous")
    return root


def analyze_packet(packet, record, *, ordered_queries, retained_ssa):
    """Pure cost core; ordered queries must come from the verified emitter."""
    try:
        data = execution_support.normalize_packet(packet); values = {v["id"]: v for v in data["values"]}
        ledger = record.get("surviving_coverage", {})
        if ledger.get("proof_status") != "Proven" or ledger.get("work_basis") != survival.WORK_BASIS or ledger.get("native_exact") is not False: raise ValueError("proven-surviving-query-ledger-required")
        if any(o["name"] in {"frisk.mma", "frisk.copy", "frisk.reduce_tensor"} for o in data["operations"]): raise ValueError("attention-or-nonscalar-operation-outside-domain")
        features = {k: 0 for k in FEATURE_NAMES}; accesses = []; skipped = []
        for operation in data["operations"]:
            name = operation["name"]
            if name not in {"frisk.tile_load", "frisk.tile_store"}: continue
            read = name == "frisk.tile_load"
            ident = operation["results"][0] if read else operation["operands"][0]
            if ident not in retained_ssa:
                skipped.append(ident); continue
            value = values[ident]; enc = value["distributed"]
            if "xf32" not in value["type"].split(",")[0]: raise ValueError("f32 access required")
            base, columns, proof_method = _map_columns(enc)
            nr = enc["topology"][0]; rb = nr.bit_length() - 1
            if not read and len(basis(columns)) != rb + 7: raise ValueError("replicated output ownership outside first version")
            if not read and "symbolic" not in enc:
                first = enc.get("first")
                if type(first) is not list or len(first) != len(enc["logical"]) or any(type(first[slot]) is not int or first[slot] != owner for owner,slot in enumerate(enc["logical"])): raise ValueError("bijective output first-owner mismatch")
            storage = operation["operands"][0] if read else operation["operands"][1]
            root = _storage(values, storage, enc["shape"])
            signatures = ordered_queries.get(("read" if read else "write", ident), [])
            if not signatures or len(signatures) != len(set(signatures)) or len(signatures) > MAX_QUERY_SITES: raise ValueError("nonempty unique observed access query sequence required")
            maps = []
            for signature in signatures:
                if len(signature) != 8 or any(type(x) is not int or x < 0 for x in signature): raise ValueError("affine query signature required")
                qbase = base ^ _apply(columns[:rb], signature[0])
                qthreads = tuple((columns[rb+i] ^ _apply(columns[:rb], signature[i+1])) << 2 for i in range(7))
                maps.append((qbase << 2, qthreads))
            results = {}
            for scope in SCOPES:
                for block in GRANULES:
                    for window in WINDOWS:
                        counts = count_windows(maps, window, block, scope)
                        results[f"{scope}_b{block}_w{window}"] = counts
                        for metric in ("footprint", "potential"):
                            features[f"gf2_reuse_{scope}_b{block}_w{window}_{metric}"] += counts[metric]
            accesses.append(dict(ssa=ident, storage=storage, root=root, access="read" if read else "write", query_count=len(signatures),
                                 query_sequence_sha256=_sha(_canonical(signatures)), map_proof=proof_method,
                                 byte_query_maps_sha256=_sha(_canonical(maps)), counts=results,
                                 constant_byte_offset=maps[0][0], thread_byte_columns=list(maps[0][1])))
        if not accesses: raise ValueError("no-surviving-supported-global-accesses")
        return dict(contract=CONTRACT, status="complete", cost_family=FAMILY, features=features,
                    feature_names=list(FEATURE_NAMES), feature_units=UNITS.copy(), accesses=accesses, skipped_dead_access_ssa=skipped,
                    scope={"address_origin":"relative-base-zero; no weak-ABI runtime base alignment assumed", "order":"actual source emitter query order after whole-SSA-survival filtering, separately validated against full TTGIR and cfg", "hardware":"128 threads; warp counts sum four separate warp images, CTA counts union all thread image cosets", "windows":"nonoverlapping query-index windows; W_eff=min(W,Q), final partial window counted exactly", "not_claimed":["native instruction order", "hardware cache/TLB misses", "dynamic execution count", "latency or causal performance improvement"]})
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return dict(contract=CONTRACT, status="unsupported", cost_family=FAMILY, features=None, reasons=[str(error)])


class _TraceEmitter(backend.Emitter):
    """Observe eager access loops without changing a generated instruction."""
    def __init__(self, packet):
        self.observed = {}; self.ordered = {}
        self.load_values = {o["results"][0] for o in packet["operations"] if o["name"] == "frisk.tile_load"}
        def observe(value, indices):
            signature = demand.canonical_query(indices)
            if signature in self.observed.setdefault(value, set()): raise ValueError("duplicate-emitter-query")
            self.observed[value].add(signature)
            if value in self.load_values: self.ordered.setdefault(("read", value), []).append(signature)
        super().__init__(packet, demand_query_observer=observe)

    def load(self, operation):
        super().load(operation)
        value = operation["results"][0]
        self.ordered[("read", value)] = [(r, 0, 0, 0, 0, 0, 0, 0) for r in range(self.v[value]["distributed"]["topology"][0])]

    def store(self, operation):
        super().store(operation)
        value = operation["operands"][0]
        self.ordered[("write", value)] = [(r, 0, 0, 0, 0, 0, 0, 0) for r in range(self.v[value]["distributed"]["topology"][0])]


def _emission_trace(packet):
    """Exact original emitter/whole-SSA trace; no safety claim is cached."""
    emitter = _TraceEmitter(packet); text, cfg = emitter.run()
    distributed = {v["id"] for v in emitter.data["values"] if "distributed" in v}
    if not cfg.get("demand_lowering"):
        emitter.observed = {v:{(r,0,0,0,0,0,0,0) for r in range(emitter.v[v]["distributed"]["topology"][0])} for v in distributed}
    plans = [(o, emitter._checked_redistribution(o)) for o in emitter.data["operations"] if o["name"] == "frisk.convert_layout"]
    retained, _ = survival.surviving_ssa(emitter.data["operations"], distributed, emitter.observed, plans)
    return dict(text=text,cfg=cfg,retained=sorted(retained),
                observed=[[v,sorted(q)] for v,q in sorted(emitter.observed.items()) if v in retained],
                ordered=[[direction,v,list(qs)] for (direction,v),qs in sorted(emitter.ordered.items())])


def summarize(directory, *, document_provider=None):
    """Read-only CPU proof with full source and compiled-input SHA lineage."""
    root = Path(directory); provenance = dict(analyzer_sha256=SOURCE_SHA256, source_dependencies=SOURCE_HASHES.copy(), dependencies={})
    try:
        if _sha(Path(__file__).read_bytes()) != SOURCE_SHA256: raise ValueError("locality-source-changed-since-import")
        for name, expected in SOURCE_HASHES.items():
            if _sha((Path(__file__).parent/name).read_bytes()) != expected: raise ValueError("source-dependency-changed:" + name)
        required = ("execution.json", "record.json", "kernel.ttgir", "kernel.cubin", "image-identity.json", "compile.json", "emitted-config.json", "execution_config.json", "surviving-coverage-audit.json")
        raw = {n:(root/n).read_bytes() for n in required}; provenance["dependencies"] = {n:_sha(v) for n,v in raw.items()}
        packet, record = json.loads(raw["execution.json"]), json.loads(raw["record.json"])
        if record.get("surviving_coverage", {}).get("proof_status") != "Proven": raise ValueError("source-survival-outside-proven-domain")
        if any(o["name"] in {"frisk.mma", "frisk.copy", "frisk.reduce_tensor"} for o in packet["operations"]): raise ValueError("attention-or-nonscalar-operation-outside-domain")
        identity, compiled = json.loads(raw["image-identity.json"]), json.loads(raw["compile.json"])
        for name in ("execution.json", "kernel.ttgir", "kernel.cubin", "execution_config.json"):
            if identity.get(name) != _sha(raw[name]): raise ValueError("image-identity-mismatch:" + name)
        if compiled.get("execution_sha256") != _sha(raw["execution.json"]) or compiled.get("cubin_sha256") != _sha(raw["kernel.cubin"]): raise ValueError("compile-receipt-mismatch")
        audit = json.loads(raw["surviving-coverage-audit.json"])
        if audit.get("status") != "passed" or audit.get("native_exact") is not False: raise ValueError("survival-audit-not-passed")
        if document_provider is None:
            emitter = _TraceEmitter(packet); text, cfg = emitter.run()
            if _sha(text.encode()) != _sha(raw["kernel.ttgir"]): raise ValueError("full-reemitted-ttgir-mismatch")
            canonical_cfg = _canonical(json.loads(json.dumps(cfg)))
            if canonical_cfg != _canonical(json.loads(raw["emitted-config.json"])) or canonical_cfg != _canonical(json.loads(raw["execution_config.json"])): raise ValueError("full-reemitted-cfg-mismatch")
            distributed = {v["id"] for v in emitter.data["values"] if "distributed" in v}
            if not cfg.get("demand_lowering"):
                emitter.observed = {v:{(r,0,0,0,0,0,0,0) for r in range(emitter.v[v]["distributed"]["topology"][0])} for v in distributed}
            plans = [(o, emitter._checked_redistribution(o)) for o in emitter.data["operations"] if o["name"] == "frisk.convert_layout"]
            retained, _ = survival.surviving_ssa(emitter.data["operations"], distributed, emitter.observed, plans)
            observed = {v:q for v,q in emitter.observed.items() if v in retained}
            trace_values=emitter.data["values"];ordered=emitter.ordered
        else:
            trace=document_provider.pure(_emission_trace,packet)
            text,cfg=trace["text"],trace["cfg"]
            if _sha(text.encode()) != _sha(raw["kernel.ttgir"]): raise ValueError("full-reemitted-ttgir-mismatch")
            canonical_cfg = _canonical(cfg)
            if canonical_cfg != _canonical(json.loads(raw["emitted-config.json"])) or canonical_cfg != _canonical(json.loads(raw["execution_config.json"])): raise ValueError("full-reemitted-cfg-mismatch")
            retained=set(trace["retained"])
            observed={v:{tuple(q) for q in queries} for v,queries in trace["observed"]}
            ordered={(direction,v):[tuple(q) for q in queries] for direction,v,queries in trace["ordered"]}
            trace_values=execution_support.normalize_packet(packet)["values"]
        body = record["surviving_coverage"]["surviving"]
        demand.compare_queries(trace_values, demand.parse_values(body), observed, demand.parse_boundary_queries(body))
        if set(audit.get("retained_ssa", [])) != retained: raise ValueError("retained-ssa-audit-mismatch")
        result = analyze_packet(packet, record, ordered_queries=ordered, retained_ssa=retained)
        result["emission_proof"] = dict(ttgir_exact_sha256=_sha(text.encode()), full_cfg_canonical_sha256=_sha(canonical_cfg),
                                        query_order_sha256=_sha(_canonical([[direction,v,list(qs)] for (direction,v),qs in sorted(ordered.items())])),
                                        retained_ssa=sorted(retained), native_order_proven=False)
        for name, expected in provenance["dependencies"].items():
            if _sha((root/name).read_bytes()) != expected: raise ValueError("input-changed-during-analysis:" + name)
        for name, expected in SOURCE_HASHES.items():
            if _sha((Path(__file__).parent/name).read_bytes()) != expected: raise ValueError("source-changed-during-analysis:" + name)
        result["provenance"] = provenance
        result["summary_sha256"] = _sha(_canonical(result))
        return result
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        result = dict(contract=CONTRACT, status="unsupported", cost_family=FAMILY, features=None, reasons=[str(error)], provenance=provenance)
        result["summary_sha256"] = _sha(_canonical(result))
        return result
