"""Rank the four native FA3 orders on a fixed H100 PCIe execution contract."""
import heapq
from typing import NamedTuple

from fa3_event_schedule import SCHEDULES


def _after_prefix(query_tiles, key_tiles, causal, prefix):
    """Sum max(active_tiles - prefix, 0) over the query CTAs of one head."""
    full = max(key_tiles - prefix, 0)
    if not causal:
        return query_tiles * full
    ramp = min(query_tiles, key_tiles)
    active = max(ramp - prefix, 0)
    return active * (active + 1) // 2 + (query_tiles - ramp) * full


# Consumer-warpgroup clock64 cycles for startup, a middle iteration and drain.
# The first tuple is plain attention and the second is causal attention.
_PHASE_CYCLES = {
    "same_tile": ((5416, 3148, 3155), (5583, 3178, 3524)),
    "qk_next": ((4421, 2857, 662), (4371, 2764, 672)),
    "pv_first": ((4452, 3106, 618), (4412, 2919, 632)),
    "late_k": ((4436, 2863, 680), (4372, 2748, 686)),
}
_PROFILE_MHZ = 1740.0
# Unmasked, single-wave lookahead refill cost in us at this SM90 profile.
# Calibrated from paired frame differences at varying CTA occupancy.
_SINGLE_WAVE_REFILL_US = (0.025046, 0.034841, 0.432094)


class _Correction(NamedTuple):
    boundary: float
    transition: float
    causal_boundary: float
    causal_transition: float
    fringe_boundary: float
    fringe_transition: float
    ring_fill: float
    causal_ring_fill: float
    masked_columns: float
    diagonal: float


# Paired, uninstrumented time relative to same_tile, in microseconds per
# representative CTA. The runtime path does no fitting or benchmarking.
_CORRECTION_US = {
    "same_tile": _Correction(*(0.0,) * 10),
    "qk_next": _Correction(0.20989257, -0.23333253, 0.52351708, 0.10989898,
                           -0.47060305, 0.16159483, 1.04529756, -0.19362315,
                           -0.10781947, -0.05099360),
    "pv_first": _Correction(0.32612606, -0.22062590, 0.41223492, 0.07029977,
                            -0.27032423, 0.16744241, 0.95986832, 0.02392239,
                            -0.33981031, 0.11879702),
    "late_k": _Correction(0.06985313, -0.11074331, 0.66582055, -0.01136427,
                          -0.14724236, -0.02162187, 0.96458156, -0.00433930,
                          -0.03093422, -0.20818195),
}


# Measured CTA entry/epilogue time and residual single-iteration drain on SM90.
_CTA_BOUNDARY_US = 5.271379
_CAUSAL_SINGLE_TILE_US = {'same_tile': 0.0, 'qk_next': 0.627929, 'pv_first': 0.472612, 'late_k': 0.782232}


def _causal_wave_costs(rows, keys, heads, sm_count, versions, cta_dispatch="forward"):
    """Estimate when the last causal CTA completes at one resident CTA per SM.

    X is the query coordinate, the fastest moving grid dimension on this
    native launch. Reuse the next available SM slot instead of inserting a
    barrier between waves. A CTA can reach the KV tail only after its row
    reaches that key tile.
    """
    from fa3_query_tile import dispatch_query_tiles
    qt, kt = (rows + 127) // 128, (keys + 127) // 128
    blocks = qt * heads
    waves = (blocks + sm_count - 1) // sm_count
    partial = keys % 128
    per_order = {}
    for name, profile in _PHASE_CYCLES.items():
        same = name == "same_tile"
        start, rate, end = profile[1]
        w = _CORRECTION_US[name]
        # Rows past the diagonal ramp all use the same full KV extent.
        ramp = min(qt, kt)
        durations = []
        for row in range(min(qt, kt + 1)):
            trips = min(row + 1, kt)
            transitions = trips - 1
            fringe = bool(partial and rows > keys and trips == kt)
            diagonal = row < kt
            cycles = (start + rate * (max(transitions - 1, 0) if same else transitions)
                      + end * ((trips > 1) if same else 1))
            correction = (w.boundary + w.causal_boundary + w.diagonal * diagonal
                          + fringe * (w.fringe_boundary + w.masked_columns * (128 - partial) / 128)
                          + transitions * (w.transition + w.causal_transition + fringe * w.fringe_transition)
                          + min(transitions, versions) * (w.ring_fill + w.causal_ring_fill))
            durations.append(_CTA_BOUNDARY_US + cycles / _PROFILE_MHZ + correction
                             + (_CAUSAL_SINGLE_TILE_US[name] if trips == 1 else 0.0))
        slots = [0.0] * sm_count
        for row in dispatch_query_tiles(qt, heads, cta_dispatch=="reverse"):
            finish = slots[0] + durations[min(row, ramp)]
            heapq.heapreplace(slots, finish)
        per_order[name] = max(slots) / waves
    return per_order


def rank_fa3_events(plan, rows, keys, causal, budget=1, *, heads=16, sm_count=114, tile_work=None, cta_dispatch="forward"):
    """Rank before compiling. The times estimate relative cost per resident CTA."""
    if cta_dispatch not in ("forward","reverse") or (cta_dispatch=="reverse" and not causal):
        raise ValueError("invalid physical CTA dispatch")
    if type(budget) is not int or budget not in (1, 2):
        raise ValueError("the native FA3 schedule budget must be 1 or 2")
    if (plan["kind"] not in ("fa3_same_tile", "fa3_qk_next")
            or rows <= 128 or (rows % 128 and tile_work is None) or keys <= 128):
        raise ValueError("budgeted FA3 requires a native contract and full TMA output tiles")
    buffers = plan["buffers"]
    versions = buffers["k"]["versions"]
    if (versions, buffers["v"]["versions"], plan["wait_pending"]) != (2, 2, 0):
        raise ValueError("budgeted FA3 requires double K/V rings and wait_group 0")
    if any(type(n) is not int or n < 1 for n in (heads, sm_count)):
        raise ValueError("expected positive head and SM counts")

    causal = bool(causal)
    query_tiles = (rows + 127) // 128
    key_tiles = (keys + 127) // 128
    # Native callers supply the paid K/V tile work from their physical copy
    # contract. Plain plan callers retain the frozen phase-model API.
    if tile_work is not None and (query_tiles, key_tiles) != (tile_work.query_tiles, tile_work.key_tiles):
        raise ValueError("FA3 phase profile and physical transport cover different tiles")
    after = (tile_work.after if tile_work is not None else
             lambda prefix: _after_prefix(query_tiles, key_tiles, causal, prefix))
    longest = min(query_tiles, key_tiles) if causal else key_tiles
    # Underfilled grids end with the longest CTA. At higher wave counts, the
    # average number of active tiles accounts for more of the elapsed time.
    tail_share = min(1.0, sm_count / (query_tiles * heads))

    def effective_after(prefix):
        mean = after(prefix) / query_tiles
        return mean + tail_share * (max(0, longest - prefix) - mean)

    transitions = effective_after(1)
    middle = effective_after(2)
    separate_last = transitions - middle
    # min(T-1, versions) transitions precede reuse past the initial ring fill.
    ring_fill = transitions - effective_after(1 + versions)
    live_fringe = bool(keys % 128 and (not causal or rows > keys))
    masked_fraction = (128 - keys % 128) / 128 if live_fringe else 0.0
    diagonal_fraction = min(query_tiles, key_tiles) / query_tiles if causal else 0.0

    ctas = query_tiles * heads
    fill = min(1.0, ctas / sm_count)
    lookahead_grid_us = 0.0
    if ctas <= sm_count and not causal and not live_fringe:
        # All CTAs can run together. The K ring must wrap before a further
        # refill can affect an outstanding lookahead PV. Fractional occupancy
        # distinguishes nearly empty and nearly full grids.
        base_us, load_us, partial_us = _SINGLE_WAVE_REFILL_US
        refills = effective_after(1 + versions)
        lookahead_grid_us = refills * (base_us + load_us * fill
                                       + partial_us * fill * (1.0 - fill))

    costs = {}
    for name, order in SCHEDULES.items():
        same = order.frame == "same_tile"
        start, rate, end = _PHASE_CYCLES[name][int(causal)]
        weight = _CORRECTION_US[name]
        boundary = (weight.boundary + causal * weight.causal_boundary
                    + live_fringe * weight.fringe_boundary
                    + masked_fraction * weight.masked_columns
                    + diagonal_fraction * weight.diagonal)
        rate_delta = (weight.transition + causal * weight.causal_transition
                      + live_fringe * weight.fringe_transition)
        startup = start / _PROFILE_MHZ
        steady = (rate * (middle if same else transitions) / _PROFILE_MHZ
                  + rate_delta * transitions
                  + (weight.ring_fill + causal * weight.causal_ring_fill) * ring_fill)
        if not same:
            steady += lookahead_grid_us
        # Same-tile startup already includes its first PV, so T=1 has no
        # separate last iteration. Lookahead always ends with the final PV.
        drain = end * (separate_last if same else 1.0) / _PROFILE_MHZ
        costs[name] = dict(startup_us=startup, steady_us=steady, drain_us=drain,
                           boundary_correction_us=boundary,
                           total_us=startup + steady + drain + boundary)
    for cost in costs.values():
        cost["wave_adjustment_us"] = 0.0
    if causal and ctas > sm_count:
        wave_cost = _causal_wave_costs(rows, keys, heads, sm_count, versions, cta_dispatch)
        # Relative critical-path costs matter most when a short last wave
        # follows initial admission. Trust the steady profile more as more
        # complete waves contribute; this damping is empirically calibrated.
        excess_waves = (ctas - sm_count) / sm_count
        tail_weight = 1.0 / max(1.0, excess_waves) ** 2
        reference = costs["same_tile"]["total_us"]
        for name, cost in costs.items():
            baseline_delta = cost["total_us"] - reference
            tail_delta = wave_cost[name] - wave_cost["same_tile"]
            adjustment = tail_weight * (tail_delta - baseline_delta)
            cost["wave_adjustment_us"] = adjustment
            cost["total_us"] += adjustment
    ordered = sorted(costs, key=lambda name: costs[name]["total_us"])
    return dict(budget=budget, selected=ordered[:budget], costs=costs,
                steady_pairs_per_head=after(1),
                possible_k_refills_per_head=after(1 + versions),
                active_tiles_per_head=after(0),
                effective_transitions=transitions, effective_same_tile_middle=middle,
                effective_initial_ring_transitions=ring_fill,
                separate_last_fraction=separate_last, sm_count=sm_count,
                grid_fill=fill, lookahead_grid_us=lookahead_grid_us)
