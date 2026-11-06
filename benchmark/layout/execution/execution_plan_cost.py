"""Static execution-plan evidence and conservative, uncalibrated comparison."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from access_plan import AccessPlan
from resource_prediction import confirm_driver_resources, parse_compiler_resource_usage


CONTRACT = "frisk-execution-plan-cost-v1"
_ASM = re.compile(r'\btt\.elementwise_inline_asm\s+"((?:\\[0-9A-Fa-f]{2}|[^"\\])*)"')
_INSTRUCTION = re.compile(r"(@!?[%\w$]+\s+)?([a-z][\w.:]*)(?:\s+(.*?))?;")
_COUNTERS = (
    "global_load_sites", "global_store_sites", "shared_load_sites", "shared_store_sites",
    "conversion_shuffles", "reduction_shuffles", "other_shuffles",
    "cta_barriers", "async_proxy_fences", "async_copy_sites",
    "wgmma_fences", "wgmma_commits", "wgmma_waits", "shared_allocated_peak_bytes",
)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(obj):
    return _sha(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {field}")
    return value


def _geometry(value, field):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"invalid {field}")
    return [_integer(x, field, 1) for x in value]


def _body(ttgir):
    if not isinstance(ttgir, str):
        raise ValueError("expected emitted TTGIR text")
    entries = re.findall(r"\btt\.func\s+public\s+@([\w$.]+)\s*\(", ttgir)
    asm = _ASM.findall(ttgir)
    if len(entries) != 1 or len(asm) != 1 or ttgir.count("tt.elementwise_inline_asm") != 1:
        raise ValueError("one Frisk inline PTX entry required")
    body = re.sub(r"\\([0-9A-Fa-f]{2})", lambda m: chr(int(m[1], 16)), asm[0])
    return entries[0], body


def _instructions(body):
    result = []
    for raw in body.splitlines():
        text = raw.split("//", 1)[0].strip()
        match = _INSTRUCTION.fullmatch(text)
        if match:
            result.append((bool(match[1]), match[2], match[3] or ""))
        elif text and text not in ("{", "}") and not text.startswith(".") and not text.endswith(":"):
            raise ValueError(f"unknown Frisk inline PTX statement: {text[:60]}")
    return result


def _memory_width(opcode):
    parts = opcode.split(".")
    scalar = re.fullmatch(r"[bfus](8|16|32|64|128)", parts[-1])
    vectors = [int(part[1:]) for part in parts if part in ("v2", "v4", "v8")]
    if not scalar or len(vectors) > 1:
        raise ValueError(f"unknown inline PTX access width: {opcode}")
    width = int(scalar[1]) * (vectors[0] if vectors else 1) // 8
    if width not in (1, 2, 4, 8, 16):
        raise ValueError(f"unsupported inline PTX access width: {opcode}")
    return width


def _shared(body, metadata):
    main = re.findall(r"^\s*\.shared\s+\.align\s+\d+\s+\.b8\s+frisk_shared\[(\d+)\];$", body, re.M)
    barriers = re.findall(r"^\s*\.shared\s+\.align\s+8\s+\.b64\s+tma_barrier_\d+;$", body, re.M)
    if len(main) > 1 or len(main) + len(barriers) != len(re.findall(r"^\s*\.shared\b", body, re.M)):
        raise ValueError("unsupported inline Shared declaration")
    static = int(main[0]) if main else 0
    tma_bytes = 8 * len(barriers)
    allocated = ((static + 7) // 8 * 8 + tma_bytes) if tma_bytes else static
    if (metadata.get("static_shared_bytes", static) != static
            or metadata.get("physical_shared_bytes", allocated) != allocated):
        raise ValueError("Shared metadata differs from emitted allocation")
    allocation = metadata.get("shared_allocation_plan", {})
    if not isinstance(allocation, dict) or allocation.get("physical_shared_bytes", allocated) != allocated:
        raise ValueError("Shared allocation evidence differs from inline PTX")
    allocations = allocation.get("allocations")
    if allocations is None:
        peak = 0 if not allocated else None
    else:
        if type(allocations) is not list or any(type(row) is not dict for row in allocations):
            raise ValueError("invalid Shared allocation intervals")
        peak = _shared_live_peak(allocations, static)
    return allocated, allocation, peak


def _shared_live_peak(allocations, capacity):
    """Peak live root payload over the emitter's closed operation intervals."""
    events = Counter()
    regions = []
    for row in allocations:
        lifetime = row.get("lifetime")
        if lifetime == "whole-kernel-declared":
            return None
        if not isinstance(lifetime, (tuple, list)) or len(lifetime) != 2:
            raise ValueError("invalid Shared root lifetime")
        start, end = (_integer(t, "Shared lifetime endpoint") for t in lifetime)
        offset, size = (_integer(row.get(k), "Shared " + k) for k in ("offset", "bytes"))
        if end < start or offset + size > capacity:
            raise ValueError("Shared live root escapes emitted allocation")
        for a, b, lo, hi in regions:
            if start <= b and a <= end and offset < hi and lo < offset + size:
                raise ValueError("simultaneously live Shared roots overlap")
        regions.append((start, end, offset, offset + size))
        events[start] += size
        events[end + 1] -= size
    live = peak = 0
    for tick in sorted(events):
        live += events[tick]
        peak = max(peak, live)
    return peak


def access_sectors32(plan, *, element_bytes, root_alignment):
    """Count distinct 32B sectors for each warp issue and possible root phase."""
    if not isinstance(plan, AccessPlan) or element_bytes not in (2, 4, 16):
        raise ValueError("checked Global access plan required")
    if type(root_alignment) is not int or root_alignment < 1 or root_alignment & (root_alignment - 1):
        raise ValueError("invalid root alignment")
    phases = range(0, 32, min(32, root_alignment))
    result = {phase: 0 for phase in phases}
    for group in plan.groups:
        offsets = plan.bases[group.base_index]
        width = group.width_bytes
        count = element_bytes if width == 1 else 1
        for warp in range(0, len(offsets), 32):
            active = [lane for lane in range(warp, min(warp + 32, len(offsets)))
                      if group.owner_mask is None or group.owner_mask >> lane & 1]
            for i in range(count):
                for phase in phases:
                    sectors = set()
                    for lane in active:
                        start = offsets[lane] + group.byte_offset + i + phase
                        sectors.update(range(start // 32, (start + width - 1) // 32 + 1))
                    result[phase] += len(sectors)
    return dict(per_phase={str(k): v for k, v in result.items()}, upper=max(result.values(), default=0))


def _recorded_sector_upper(metadata, traffic):
    rows = metadata.get("global_access", ())
    if not rows or any("sector_issues32_upper" not in row for row in rows):
        return None
    recorded = {"global_load": Counter(), "global_store": Counter()}
    total = 0
    for row in rows:
        operation = row.get("operation")
        if operation not in ("load", "store", "copy"):
            raise ValueError("unknown Global traffic record")
        counts = recorded["global_load" if operation == "copy" else "global_" + operation]
        for width, count in row["groups_by_width"].items():
            width = int(width)
            count = _integer(count, "Global access group count")
            if width == 1:
                count *= _integer(row["element_bytes"], "Global access element width", 1)
            counts[width] += count
        total += _integer(row["sector_issues32_upper"], "Global sector upper bound")
    if any(recorded[k] != traffic[k] for k in recorded):
        return None
    return total


def _trace(metadata, shuffles, barriers, mma, waits):
    rows = metadata.get("execution_trace")
    if rows is None:
        return (*((0, 0, 0) if not shuffles else (None, None, None)), 0 if not barriers else None)
    if type(rows) is not list:
        raise ValueError("invalid operation emission trace")
    counts = Counter()
    summed = Counter()
    edge_barriers = 0
    for row in rows:
        if type(row) is not dict or not isinstance(row.get("operation"), str):
            raise ValueError("invalid operation emission trace row")
        for field in ("shuffles", "barriers", "wgmma", "waits"):
            summed[field] += _integer(row.get(field), "trace " + field)
        counts[row["operation"]] += row["shuffles"]
        if row["operation"] == "frisk.convert_layout":
            edge_barriers += row["barriers"]
    if (summed["shuffles"] > shuffles or summed["barriers"] > barriers
            or summed["wgmma"] > mma or summed["waits"] > waits):
        raise ValueError("operation trace exceeds emitted PTX counts")
    local_barriers = barriers - edge_barriers if summed["barriers"] == barriers else None
    if summed["shuffles"] != shuffles:
        return None, None, None, local_barriers
    conversion, reduction = counts["frisk.convert_layout"], counts["frisk.reduce_tensor"]
    return conversion, reduction, shuffles - conversion - reduction, local_barriers


def _wgmma(instructions):
    forms = Counter()
    fences = commits = waits = drain_waits = uncommitted = pending_groups = peak_groups = 0
    group_sizes = []
    complete = True
    branched = False
    for predicate, opcode, operands in instructions:
        branched |= opcode.split(".", 1)[0] in ("bra", "brx", "call", "ret", "exit")
        if not opcode.startswith("wgmma."):
            continue
        complete &= not predicate
        if opcode.startswith("wgmma.mma_async.sync.aligned."):
            match = re.match(r"\{[^}]+\},\s*([{\w])", operands)
            if match is None:
                raise ValueError("unsupported WGMMA operand layout")
            forms[("rs:" if match[1] == "{" else "ss:") + opcode] += 1
            uncommitted += 1
        elif opcode == "wgmma.fence.sync.aligned":
            fences += 1
        elif opcode == "wgmma.commit_group.sync.aligned":
            commits += 1
            group_sizes.append(uncommitted)
            complete &= uncommitted > 0
            uncommitted = 0
            pending_groups += 1
            peak_groups = max(peak_groups, pending_groups)
        elif opcode == "wgmma.wait_group.sync.aligned" and re.fullmatch(r"[0-7]", operands):
            waits += 1
            drain_waits += int(operands == "0")
            pending_groups = min(pending_groups, int(operands))
        else:
            raise ValueError(f"unsupported WGMMA instruction: {opcode}")
    complete &= uncommitted == 0 and pending_groups == 0 and (not forms or (fences > 0 and not branched))
    return {
        "sites": sum(forms.values()), "by_form_and_opcode": dict(sorted(forms.items())),
        "fences": fences, "commits": commits, "waits": waits, "drain_waits": drain_waits,
        "issues_per_commit": group_sizes if complete else None,
        "peak_outstanding_group_upper": peak_groups if complete else None,
        "lifecycle": "straight-line-counted" if complete else "unknown",
    }


def _native(native, ttgir_sha, entry, grid, block, shared_bytes, dynamic_bytes):
    result = dict(status="unknown", registers_per_thread=None, local_bytes=None,
                  resident_ctas_per_sm=None, theoretical_cta_waves=None, launchable=None)
    if native is None:
        return result
    if type(native) is not dict:
        raise ValueError("invalid native execution evidence")
    receipt = native["receipt"]
    raw = Path(native["cubin"]).read_bytes()
    cubin_sha = _sha(raw)
    if (raw[:4] != b"\x7fELF" or receipt["cubin_sha256"] != cubin_sha
            or receipt["ttgir_sha256"] != ttgir_sha or receipt["entry"] != entry
            or _geometry(receipt["grid"], "receipt grid") != grid
            or _geometry(receipt["block"], "receipt block") != block
            or receipt.get("dynamic_shared_bytes", dynamic_bytes) != dynamic_bytes):
        raise ValueError("native compile receipt differs from exact TTGIR, image or launch")
    compiler = native["compiler_exact"]
    identity = compiler["identity"]
    resource_text = compiler["raw_resource_usage"]
    if (compiler.get("kind") != "compiler-exact-postcompile" or identity.get("entry") != entry
            or identity.get("cubin_sha256") != cubin_sha
            or not re.fullmatch(r"[0-9a-f]{64}", identity.get("tool_sha256", ""))
            or compiler["raw_resource_sha256"] != _sha(resource_text.encode())):
        raise ValueError("compiler resource document differs from bound image")
    physical = parse_compiler_resource_usage(resource_text, entry)
    if physical != compiler["resources"]:
        raise ValueError("compiler resource payload differs from raw facts")
    result.update(status="compiler-exact", cubin_sha256=cubin_sha, tool_sha256=identity["tool_sha256"],
                  registers_per_thread=physical["registers_per_thread"], local_bytes=physical["local_bytes_per_thread"])
    driver = native.get("driver")
    if driver is None:
        return result
    if (driver.get("cubin_sha256") != cubin_sha or driver.get("entry") != entry
            or not isinstance(driver.get("device_identity"), str) or not driver["device_identity"]):
        raise ValueError("driver evidence differs from bound image or has no target identity")
    measured = driver["resources"]
    if (measured.get("source") != "CUDA-driver-loaded-function"
            or measured.get("threads_per_block") != math.prod(block)
            or measured.get("dynamic_shared_bytes") != dynamic_bytes
            or measured.get("total_shared_bytes_per_block") != shared_bytes + dynamic_bytes):
        raise ValueError("driver resource scope differs from requested launch")
    resolved = dict(kind="resolved-compiler-resources", compiler_exact=compiler,
                    raw_native_shared_bytes=physical["raw_shared_bytes"], shared_bytes=shared_bytes,
                    local_bytes=result["local_bytes"], registers_per_thread=result["registers_per_thread"])
    confirmed = confirm_driver_resources(resolved, measured, loaded_cubin_sha256=cubin_sha, loaded_entry=entry)
    limits, occupancy = measured["device_limits"], measured["residency"]
    active = _integer(occupancy.get("active_blocks_per_sm"), "resident CTA count")
    sms = _integer(limits.get("multiprocessor_count"), "SM count", 1)
    if (occupancy.get("kind") != "theoretical-resource-ceiling"
            or occupancy.get("api") != "cuOccupancyMaxActiveBlocksPerMultiprocessor"
            or occupancy.get("launchable_by_resources") is not (active > 0)
            or active > _integer(limits.get("max_blocks_per_sm"), "maximum resident CTAs", 1)
            or active * math.prod(block) > _integer(limits.get("max_threads_per_sm"), "maximum resident threads", 1)):
        raise ValueError("invalid driver occupancy ceiling")
    capacity = active * sms
    result.update(status="driver-confirmed", device_identity=driver["device_identity"], device_limits=limits,
                  local_bytes=confirmed["local_bytes"], resident_ctas_per_sm=active, launchable=active > 0,
                  theoretical_cta_waves=(math.prod(grid) + capacity - 1) // capacity if active else None)
    return result


def extract_features(ttgir, metadata, *, grid=(1, 1, 1), block=(128, 1, 1), native=None):
    """Extract structural facts, with optional bound compiler and CUDA driver facts.

    Counts are static PTX sites, not cycles, memory transactions, or measured
    occupancy. Candidate comparison requires the caller to hold the semantic
    workload and runtime contract fixed.
    """
    grid, block = _geometry(grid, "grid"), _geometry(block, "block")
    entry, body = _body(ttgir)
    if not isinstance(metadata, dict):
        raise ValueError("expected checked emitter metadata")
    attrs = [int(x) for x in re.findall(r'"ttg\.num-warps"\s*=\s*(\d+)\s*:\s*i32', ttgir)]
    if attrs != [math.prod(block) // 32] or math.prod(block) % 32 or not re.search(
        r'"ttg\.num-ctas"\s*=\s*1\s*:\s*i32', ttgir
    ):
        raise ValueError("TTGIR warp or cluster geometry differs from launch")
    for section in ("tiled_execution", "attention_bhsd"):
        launch = metadata.get(section, {})
        if ("grid" in launch and _geometry(launch["grid"], section + " grid") != grid
                or "block" in launch and _geometry(launch["block"], section + " block") != block):
            raise ValueError("emitter launch metadata differs from requested launch")
    instructions = _instructions(body)
    ops = Counter(op for _, op, _ in instructions)
    traffic = {name: Counter() for name in ("global_load", "global_store", "shared_load", "shared_store")}
    for _, opcode, _ in instructions:
        for asm, name in (("ld.global.", "global_load"), ("st.global.", "global_store"),
                          ("ld.shared.", "shared_load"), ("st.shared.", "shared_store")):
            if opcode.startswith(asm):
                traffic[name][_memory_width(opcode)] += 1
                break
    shared_bytes, allocation, live_peak = _shared(body, metadata)
    mma = _wgmma(instructions)
    shuffles = sum(count for opcode, count in ops.items() if opcode.startswith("shfl."))
    barriers = ops["bar.sync"] + ops["barrier.sync"]
    conversion, reduction, other, local_barriers = _trace(metadata, shuffles, barriers, mma["sites"], mma["waits"])
    layout_sectors = metadata.get("layout_sector_issues32")
    if layout_sectors is not None:
        layout_sectors = _integer(layout_sectors, "layout sector proxy")
    dynamic_bytes = _integer(metadata.get("dynamic_shared_bytes", 0), "dynamic Shared bytes")
    digest = _sha(ttgir.encode())
    publication = None
    if metadata.get("copy_path") == "vector" and any(
            record.get("operation") == "copy" for record in metadata.get("global_access", ())):
        non_publication = "\n".join(line for line in body.splitlines() if line.strip() not in (
            "bar.sync 0;", "fence.proxy.async.shared::cta;"))
        publication = dict(body_sha256=_sha(non_publication.encode()),
                           batched=bool(metadata.get("copy_batches")))
    return {
        "contract": CONTRACT, "static_status": "static-heuristic",
        "ttgir_sha256": digest, "metadata_sha256": _digest(metadata), "entry": entry,
        "scope": {"grid": grid, "block": block, "copy_path": metadata.get("copy_path"),
                  "global_alignments": metadata.get("global_alignments"),
                  "ftz": metadata.get("ftz"), "fp_fusion": metadata.get("fp_fusion"),
                  "reassociation": metadata.get("reassociation"), "p_rounding": metadata.get("p_rounding"),
                  "wgmma_atoms": mma["by_form_and_opcode"]},
        "static": {
            **{name + "_by_width_bytes": {str(k): v for k, v in sorted(counts.items())}
               for name, counts in traffic.items()},
            **{name + "_sites": sum(counts.values()) for name, counts in traffic.items()},
            "shuffle_sites": shuffles, "conversion_shuffles": conversion, "reduction_shuffles": reduction,
            "other_shuffles": other, "cta_barriers": barriers, "local_cta_barriers": local_barriers,
            "global_sector_issues32_upper": _recorded_sector_upper(metadata, traffic),
            "layout_sector_issues32": layout_sectors,
            "async_proxy_fences": sum(n for op, n in ops.items() if op.startswith("fence.proxy.async.shared")),
            "async_copy_sites": sum(n for op, n in ops.items() if op.startswith(("cp.async.ca.", "cp.async.cg.", "cp.async.bulk."))),
            "wgmma_fences": mma["fences"], "wgmma_commits": mma["commits"], "wgmma_waits": mma["waits"],
            "shared_allocated_peak_bytes": shared_bytes + dynamic_bytes,
            "shared_live_root_peak_bytes": live_peak,
        },
        "wgmma": mma, "shared_allocation": allocation, "copy_publication": publication,
        "native": _native(native, digest, entry, grid, block, shared_bytes, dynamic_bytes),
    }


def _pareto(pairs):
    better = any(a < b for a, b in pairs)
    worse = any(a > b for a, b in pairs)
    if better and worse:
        return "incomparable"
    if better:
        return "dominates"
    if worse:
        return "dominated"
    return "equal"


def copy_publication_only(candidate, control):
    """Require identical checked work after eliding only the two publication instructions."""
    new, old = (row.get("copy_publication") for row in (candidate, control))
    if (not new or not old or new["batched"] is not True or old["batched"] is not False
            or new["body_sha256"] != old["body_sha256"]
            or any(candidate[key] != control[key] for key in ("scope", "wgmma", "shared_allocation"))):
        return False
    fields = ("cta_barriers", "local_cta_barriers", "async_proxy_fences")
    after, before = (row["static"] for row in (candidate, control))
    if any(type(row.get(key)) is not int for row in (after, before) for key in fields):
        return False
    removed = [before[key] - after[key] for key in fields]
    return (removed[0] > 0 and all(n == removed[0] for n in removed)
            and {k: v for k, v in after.items() if k not in fields}
            == {k: v for k, v in before.items() if k not in fields})


def static_key(features, *, local=False):
    """Lexicographic provisional ordering, without latency or promotion claims."""
    if features.get("contract") != CONTRACT:
        raise ValueError("unknown execution plan evidence schema")
    counters = features["static"]
    sector = counters["global_sector_issues32_upper"]
    layout = counters["layout_sector_issues32"]
    sector_key = (0, sector) if sector is not None else (1, layout) if layout is not None else (2, 0)
    widths = tuple(sum(counters["global_" + kind + "_by_width_bytes"].get(str(width), 0)
                       for kind in ("load", "store")) for width in (1, 2, 4, 8))
    local_metrics = (
        counters["global_load_sites"] + counters["global_store_sites"], *widths,
        counters["reduction_shuffles"], counters["wgmma_waits"], counters["wgmma_fences"],
        counters["local_cta_barriers" if local else "cta_barriers"],
    )
    joint_metrics = (
        counters["conversion_shuffles"], counters["other_shuffles"],
        counters["shared_load_sites"] + counters["shared_store_sites"],
        counters["shared_allocated_peak_bytes"], counters["shared_live_root_peak_bytes"],
    )
    metrics = local_metrics if local else local_metrics + joint_metrics
    return (sector_key, *((int(value is None), value if value is not None else 0) for value in metrics))


def compare_features(candidate, incumbent):
    """Pareto comparison, with no promotion based on unknown resource fields."""
    if candidate.get("contract") != CONTRACT or incumbent.get("contract") != CONTRACT:
        raise ValueError("unknown execution plan evidence schema")
    if candidate["scope"] != incumbent["scope"]:
        return dict(relation="incomparable", structural_relation="incomparable", reason="different-scope")
    pairs = [(candidate["static"][key], incumbent["static"][key]) for key in _COUNTERS]
    unknown_structure = any(a is None or b is None for a, b in pairs)
    if any(row["wgmma"]["lifecycle"] == "unknown" for row in (candidate, incumbent)):
        unknown_structure = True
    structural = "incomparable" if unknown_structure else _pareto(pairs)
    ca, inc = candidate["native"], incumbent["native"]
    known = (ca["status"] == inc["status"] == "driver-confirmed"
             and ca["device_identity"] == inc["device_identity"] and ca["device_limits"] == inc["device_limits"]
             and ca["launchable"] and inc["launchable"])
    if unknown_structure or not known:
        return dict(relation="incomparable", structural_relation=structural,
                    reason="unknown-structural-or-bound-driver-resources")
    physical = [(ca[key], inc[key]) for key in ("registers_per_thread", "local_bytes", "theoretical_cta_waves")]
    physical.append((-ca["resident_ctas_per_sm"], -inc["resident_ctas_per_sm"]))
    return dict(relation=_pareto(pairs + physical), structural_relation=structural, reason="unweighted-pareto")


def pareto_frontier(features_by_arm, *, structural=False):
    """Return nondominated arms in caller order, retaining ties and unknowns."""
    field = "structural_relation" if structural else "relation"
    return [
        arm for arm, feature in features_by_arm.items()
        if not any(peer != arm and compare_features(other, feature)[field] == "dominates"
                   for peer, other in features_by_arm.items())
    ]
