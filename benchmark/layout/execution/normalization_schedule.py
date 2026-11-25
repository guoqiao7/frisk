"""Grid-aware straight-line SASS service estimate for the SM90 four-warp replay."""
import ctypes
from compiled_global_access import _ENCODED
from normalization_reuse import NormalizationReuse, reuse_scenarios, reuse_service, validate_reuse
import re


# Nominal cycles for dependent loads in the H100 PCIe profile. Measured by
# independent shared/local/L2-warm pointer chases, not by timing layouts.
_LOAD = {"LDG": ("global", 284), "LDL": ("local", 35),
         "LDS": ("shared", 30), "LDSM": ("shared", 30)}
_CONTROL = {"BRA", "BRX", "BRXU", "BSSY", "BREAK", "BSYNC", "CALL", "RET",
            "JMP", "JMX", "JMXU", "BRK", "CONT"}


def active_ctas(image):
    """Query occupancy for this loaded function, including dynamic shared memory."""
    image._init_handles()
    driver = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    status = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        ctypes.byref(count), ctypes.c_void_p(image.function), ctypes.c_int(128),
        ctypes.c_size_t(image.metadata.shared))
    if status or not 1 <= count.value <= 16:
        raise RuntimeError(f"invalid SM90 four-warp occupancy ({status}, {count.value})")
    return count.value


def instruction_stream(listing, entry):
    """Admit a contiguous straight-line entry up to the first unconditional exit."""
    sections = re.split(r"(?m)^\s*Function\s*:\s*(\S+)\s*$", listing)
    bodies = [s for name, s in zip(sections[1::2], sections[2::2]) if name == entry]
    if len(bodies) != 1:
        return None
    stream = []
    for pc, m in enumerate(_ENCODED.finditer(bodies[0])):
        address, pred, op, mods, _, high = m.groups()
        if int(address, 16) != pc * 16 or op in _CONTROL or (op == "EXIT" and pred):
            return None
        control = int(high, 16)
        # The per-warp issue count includes predicated straight-line instructions.
        # A predicated load is an upper estimate of the issued dependency.
        stream.append((op, (control >> 52) & 63, (control >> 46) & 7,
                       (control >> 41) & 15))
        if op == "EXIT":
            return stream
    return None


def wave_service(stream, resident):
    """One warp per CTA per scheduler. Interleave up to resident independent warps."""
    time = issue = 0
    waits = dict(global_wait=0, local_wait=0, shared_wait=0)
    ready = [0] * 6
    source = [None] * 6
    for op, mask, write_slot, stall in stream:
        latest = time
        cause = None
        while mask:
            bit = mask & -mask
            slot = bit.bit_length() - 1
            if ready[slot] > latest:
                latest, cause = ready[slot], source[slot]
            ready[slot] = 0
            mask -= bit
        if cause is not None:
            waits[cause + "_wait"] += latest - time
        time = latest
        if write_slot < 6 and op in _LOAD:
            kind, latency = _LOAD[op]
            if time + latency >= ready[write_slot]:
                ready[write_slot], source[write_slot] = time + latency, kind
        spacing = max(resident, stall)
        issue += spacing
        time += spacing
    return dict(issue=issue, **waits)


def schedule_profile(listing, entry, occupancy):
    stream = instruction_stream(listing, entry)
    if not stream:
        return None
    return dict(warp_issues=len(stream), resident_ctas=occupancy,
                waves=[wave_service(stream, n) for n in range(1, occupancy + 1)])


def grid_service(record, grid_ctas, sm_count):
    """Score the most heavily assigned SM, including a possibly partial final wave."""
    profile = record["schedule"]
    ctas = (grid_ctas + sm_count - 1) // sm_count
    full, tail = divmod(ctas, profile["resident_ctas"])
    sectors = record["read_sectors"] + record["write_sectors"] + record["spill_sectors"]

    def cost(n):
        scheduled = sum(profile["waves"][n - 1].values())
        # 128 bytes per SM service quantum, i.e. four 32-byte sectors.
        return max(scheduled, (n * sectors + 3) // 4)

    return full * cost(profile["resident_ctas"]) + (cost(tail) if tail else 0)


def check_grid(grid_ctas, sm_count):
    if type(grid_ctas) is not int or grid_ctas <= 0 or type(sm_count) is not int or sm_count <= 0:
        raise ValueError("normalization ranking requires positive integer grid_ctas and sm_count")


def rank_grid(records, baseline, grid_ctas, sm_count, *, reuse=None, l2_bytes=None):
    """Rank the fixed compiled set. Keep the physical winner on incomplete evidence."""
    check_grid(grid_ctas, sm_count)
    if reuse is not None or l2_bytes is not None:
        validate_reuse(reuse, l2_bytes)
    receipt = dict(grid_ctas=grid_ctas, sm_count=sm_count, baseline=baseline)
    if any(not r.get("schedule") or "physical_total" not in r for r in records.values()):
        return baseline, dict(receipt, status="incomplete-schedule")
    scores = {key: grid_service(r, grid_ctas, sm_count) for key, r in records.items()}
    reference = records[baseline]
    # Preserve the existing protection against serial reloads inside closed
    # reduction phases, now relative to the already accepted physical winner.
    eligible = [k for k, r in records.items()
                if r["interior_reloads"] - reference["interior_reloads"]
                <= max(0, sum(reference["ldg"]) - sum(r["ldg"]))]
    def select(costs):
        best = min(eligible, key=lambda k: (costs[k], records[k]["conversions"], k != baseline))
        return best if 100 * costs[best] < 98 * costs[baseline] else baseline

    grid_pick = select(scores)
    receipt.update(status="ranked", scores=scores, eligible=eligible)
    supplied = reuse is not None
    if not supplied:
        reuse = NormalizationReuse()

    evidence = dict(input_distance_bytes=reuse.input_distance_bytes,
                    output_distance_bytes=reuse.output_distance_bytes, l2_bytes=l2_bytes,
                    grid_winner=grid_pick, grid_scores=scores, supplied=supplied, scenarios=[])
    receipt["reuse"] = evidence
    if any("stg" not in r or any("global_wait" not in w or "shared_wait" not in w or "local_wait" not in w
                                 for w in r["schedule"]["waves"]) for r in records.values()):
        evidence["status"] = "incomplete-reuse-evidence"
        return grid_pick, receipt
    min_read = min(r["read_sectors"] for r in records.values())
    min_write = min(r["write_sectors"] for r in records.values())
    for rd, wr in reuse_scenarios(reuse, l2_bytes):
        costs = {k: reuse_service(r, grid_ctas, sm_count, rd, wr, min_read, min_write)
                 for k, r in records.items()}
        evidence["scenarios"].append(dict(input_miss=rd, output_miss=wr,
                                          winner=select(costs), scores=costs))
    scenarios = evidence["scenarios"]
    if len(scenarios) == 1:
        evidence["status"] = "specified"
        receipt["scores"] = scenarios[0]["scores"]
        return scenarios[0]["winner"], receipt
    winners = {s["winner"] for s in scenarios}
    if not supplied:
        evidence["status"] = ("stable-across-unknown-reuse" if len(winners) == 1
                              else "reuse-sensitive-grid-fallback")
        return grid_pick, receipt
    if len(winners) == 1:
        evidence["status"] = "stable-across-unknown-reuse"
        # There is no unique point estimate. Per-scenario costs are in the receipt.
        receipt.pop("scores")
        return scenarios[0]["winner"], receipt
    evidence["status"] = "reuse-sensitive-grid-fallback"
    return grid_pick, receipt
