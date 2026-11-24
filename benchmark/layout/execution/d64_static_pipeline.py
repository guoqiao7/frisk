"""H100 PCIe D64 packed self-attention with FP16 inputs and FP32 producer or solver output."""
import heapq
import math
from functools import lru_cache

# Each arm has (Q rows, KV rows, resident CTAs/SM). Capacities use the
# driver occupancy of the compiled kernel including dynamic shared memory.
PROFILE = {'cp-q128-wait4': (128, 64, 2),
 'cp-q256-wait4': (256, 64, 1),
 'cp-q64-producer-wait4': (64, 64, 4),
 'cp-q64-producer-wait4-late-v': (64, 64, 5),
 'q64-kv128-k-tma': (64, 128, 3),
 'q64-kv128-k-tma-post-qk-v-tma': (64, 128, 3),
 'q64-kv128-k-tma-v-tma': (64, 128, 3),
 'q64-kv64-k-tma': (64, 64, 5),
 'q64-kv64-k-tma-post-qk-v-tma': (64, 64, 5),
 'q64-kv64-k-tma-v-tma': (64, 64, 5)}
# Nonnegative launch, critical-path, and full-grid costs in microseconds.
COEFFICIENTS = {('cp-q128-wait4', False): (0.0, 5.2610006, 0.9038977, 0.0523, 0.7631536, 0.0),
 ('cp-q128-wait4', True): (4.2255145, 0.0, 0.8513189, 0.4754886, 0.502708, 3.7015716),
 ('cp-q256-wait4', False): (0.0, 7.1224754, 1.6100606, 0.0, 1.3529838, 0.0),
 ('cp-q256-wait4', True): (0.0, 6.8736294, 0.9634175, 0.9634175, 1.0189362, 0.0),
 ('cp-q64-producer-wait4', False): (0.0, 2.9733345, 0.7800528, 0.0, 0.4017791, 0.0),
 ('cp-q64-producer-wait4', True): (1.5527356, 1.0298691, 0.9113069, 0.0, 0.4706807, 0.8579902),
 ('cp-q64-producer-wait4-late-v', False): (0.0, 3.0904031, 0.4070152, 0.1958412, 0.2050862, 0.0),
 ('cp-q64-producer-wait4-late-v', True): (2.8394552, 0.0, 0.6794582, 0.0, 0.4050952, 0.6736912),
 ('q64-kv128-k-tma', False): (0.0, 2.6536907, 0.4790672, 0.3272621, 0.5841988, 0.0),
 ('q64-kv128-k-tma', True): (3.8992731, 1.6943027, 0.1831039, 0.469944, 0.6169334, 0.6411813),
 ('q64-kv128-k-tma-post-qk-v-tma', False): (0.0, 2.7814674, 0.0914305, 0.6012966, 0.1799768, 0.0),
 ('q64-kv128-k-tma-v-tma', False): (0.0, 3.0525389, 0.368923, 0.300035, 0.4530666, 0.0),
 ('q64-kv64-k-tma', False): (0.0, 3.9055029, 0.3245703, 0.0724103, 0.361228, 0.0),
 ('q64-kv64-k-tma', True): (2.8565063, 0.0, 0.5849628, 0.1176526, 0.3147737, 0.0512225),
 ('q64-kv64-k-tma-post-qk-v-tma', False): (0.0, 4.6434867, 0.1294011, 0.0, 0.4316168, 0.0),
 ('q64-kv64-k-tma-v-tma', False): (0.0, 4.5997957, 0.2090667, 0.0, 0.4196534, 0.0)}

@lru_cache(maxsize=256)
def features(rows, heads, causal, qtile, ktile, residency, sm_count):
    qtiles = (rows + qtile - 1) // qtile
    ktiles = (rows + ktile - 1) // ktile
    trips = [min(ktiles, (qtile*(i+1)+ktile-1)//ktile) for i in range(qtiles)] if causal else [ktiles]*qtiles
    count = qtiles*heads
    def critical(capacity):
        if not causal:
            blocks = (count + capacity - 1)//capacity
            return blocks, blocks*ktiles
        slots = [(0, 0, 0, i) for i in range(min(capacity, count))]
        for _ in range(heads):
            for work in trips:
                _, blocks, total, slot = slots[0]
                blocks += 1
                total += work
                heapq.heapreplace(slots,(3*blocks+total,blocks,total,slot))
        _, blocks, total, _ = max(slots)
        return blocks,total
    blocks, path = critical(sm_count*residency)
    _, single = critical(sm_count)
    work = sum(trips)*heads/sm_count
    return (1.0, blocks, path, single, work, count/sm_count)

# Predict log(time_KV64 / time_KV128) within each fixed transport and issue order.
# The features describe useful work, startup, queue drain and contention. The
# absolute model above supplies the common time scale across unlike families.
_PAIR_COEFFICIENTS = {('', False): (-0.032447854, 0.439336403, 0.366332158, 0.18269932, -0.135635608, 0.013247637),
 ('', True): (-0.04548539, -0.004554505, 0.211160774, 0.43072756, -0.022741757, 0.018302383),
 ('-post-qk-v-tma', False): (0.03256907, 0.303170517, -0.222996412, 0.05865711, -0.048086869, 0.124229668),
 ('-v-tma', False): (0.022435142, 0.229970067, -0.274520681, 0.131297619, -0.091945942, 0.177136365)}


_PAIR_ERROR = {('', False): 0.021160644,
 ('', True): 0.035428506,
 ('-post-qk-v-tma', False): 0.014976653,
 ('-v-tma', False): 0.012584482}


@lru_cache(maxsize=512)
def _resident_service(rows, heads, causal, kv, capacity, sms, contention):
    """Approximate head-major CTA dispatch on SM90. Time is measured in KV visits."""
    queries = (rows + 63) // 64
    max_visits = (rows + kv - 1) // kv
    jobs = queries * heads

    def service(cta):
        end_row = (cta % queries + 1) * 64
        visits = min(max_visits, (end_row + kv - 1) // kv) if causal else max_visits
        return 1 + visits  # one startup unit followed by the actual prefix

    # Stripe the first resident wave over SMs. Later CTAs fill whichever slot
    # retires first. Heap entries within an SM are deadlines in service units.
    queues = [[] for _ in range(min(sms, jobs))]
    next_cta = min(jobs, capacity * sms)
    for cta in range(next_cta):
        heapq.heappush(queues[cta % sms], service(cta))

    events = [(q[0] * (1 + contention * (len(q) - 1)), sm)
              for sm, q in enumerate(queues)]
    heapq.heapify(events)
    last = [0.0] * len(queues)
    busy = resident = 0.0
    while events:
        now, sm = heapq.heappop(events)
        q = queues[sm]
        elapsed = now - last[sm]
        busy += elapsed
        resident += elapsed * len(q)
        last[sm] = now
        completed = heapq.heappop(q)
        if next_cta < jobs:
            heapq.heappush(q, completed + service(next_cta))
            next_cta += 1
        if q:
            interval = (q[0] - completed) * (1 + contention * (len(q) - 1))
            heapq.heappush(events, (now + interval, sm))
    return now, resident / busy


@lru_cache(maxsize=256)
def _kv_pair_features(rows, heads, causal, sms, capacity64, capacity128):
    queries = (rows + 63) // 64
    visits, critical, occupancy = [], [], []
    for kv, cap in ((64, capacity64), (128, capacity128)):
        tiles = (rows + kv - 1) // kv
        mean_visits = (sum(min(tiles, ((i + 1) * 64 + kv - 1) // kv)
                           for i in range(queries)) / queries if causal else tiles)
        visits.append(mean_visits)
        # The fully shared issue-pipe limit estimates critical service. Partial
        # overlap separately estimates the number of CTAs actually concurrent.
        path, _ = _resident_service(rows, heads, causal, kv, cap, sms, 1.0)
        _, active = _resident_service(rows, heads, causal, kv, cap, sms, 0.5)
        critical.append(path)
        occupancy.append(active)
    return (1.0, math.log(visits[0] / (2 * visits[1])), 1 / visits[1],
            math.log(critical[0] / (2 * critical[1])),
            math.log(occupancy[0] / occupancy[1]), 1 / occupancy[1])


def _score_kv_pairs(costs, rows, heads, causal, sms):
    for (suffix, masked), coefficients in _PAIR_COEFFICIENTS.items():
        if masked != causal:
            continue
        small = 'q64-kv64-k-tma' + suffix
        large = 'q64-kv128-k-tma' + suffix
        if small not in costs or large not in costs:
            continue
        features = _kv_pair_features(rows, heads, causal, sms,
                                     PROFILE[small][2], PROFILE[large][2])
        log_ratio = sum(w * x for w, x in zip(coefficients, features))
        # An uncertain within-family comparison supplies no reason to change
        # the original order. Margins are median held-out errors in log time.
        if abs(log_ratio) <= _PAIR_ERROR[suffix, masked]:
            continue
        # Calibrate only the width. The best score of each transport/issue
        # family remains its cross-family estimate from the absolute model.
        best = min(costs[small], costs[large])
        if log_ratio > 0:
            costs[small], costs[large] = best * math.exp(log_ratio), best
        else:
            costs[small], costs[large] = best, best * math.exp(-log_ratio)


# Log K-TMA latency relative to late-V cp.async, calibrated on paired SM90a
# causal launches. Rows are KV64 and KV128. The last two terms account for
# a partially filled tensor-map tile and the amount of useful work it holds.
_CAUSAL_K_TMA_LOG_RATIOS = ((-0.0359645655, -0.0176160145, -0.0070397658, 0.0386276194, -0.0218351996, -0.0075375233), (0.0919023754, -0.0604063814, 0.0414451746, -0.0086150372, 0.001645747, -0.0548002185))


def _compare_causal_transport(costs, rows, heads, sm_count):
    baseline = "cp-q64-producer-wait4-late-v"
    alternatives = ("q64-kv64-k-tma", "q64-kv128-k-tma")
    if min(costs, key=costs.get) != baseline or not all(a in costs for a in alternatives):
        return
    qtile, kv_tile, residency = PROFILE[baseline]
    _, _, critical, _, total, grid = features(rows, heads, True, qtile, kv_tile, residency, sm_count)
    mean = total / grid
    tail = rows % PROFILE[alternatives[1]][1]
    terms = (1.0, math.log1p(mean), math.log1p(grid), math.log1p(critical / mean),
             float(bool(tail)), tail / PROFILE[alternatives[1]][1])
    for arm, coefficients in zip(alternatives, _CAUSAL_K_TMA_LOG_RATIOS):
        log_ratio = sum(feature * weight for feature, weight in zip(terms, coefficients))
        costs[arm] = costs[baseline] * math.exp(log_ratio)


# Paired SM90a log latency ratios. D64 compares V-TMA with the same K-TMA
# width and issue order. D128 compares whole plans with the late-V cp.async plan.
# Each row has weights and a training range. The runtime needs no device timings.
_TRANSPORT_LOG_RATIOS = {
    64: {
        "q64-kv64-k-tma-v-tma": ((-0.019203797, -0.01925757, -0.009790354, -0.002788265, 0.0, 0.000810602, -0.003559809), -0.107340289, -0.018144228),
        "q64-kv128-k-tma-v-tma": ((-0.152868674, -0.001924604, -0.000563472, 0.000112127, 0.0, -0.003120668, -0.001907654), -0.215147413, -0.062166918),
    },
    128: {
        'ring64': ((-0.07463309, 0.00051357, 0.0022406, 0.00093125, 0.00027612, -0.00134616, -0.17230874), -0.17657652, 0.04039765),
        'ring128': ((0.13965656, 0.00105237, 0.00363327, 0.00066229, -0.00263103, 0.00121146, 0.1550664), 0.05875509, 0.33050176),
        'q64-kv64-k-tma': ((-0.07900958, 1.72e-05, 0.0016466, 0.00052886, 0.00039845, -0.00192938, -0.24696095), -0.19150609, 0.05414185),
        'q64-kv64-k-tma-v-tma': ((-0.26207735, -0.00258794, -0.00093679, -0.00055197, 0.00109452, -0.00220835, -0.28266922), -0.42544688, -0.15081541),
        'q64-kv128-k-tma': ((0.12203522, 0.00081135, 0.00350379, 0.0005866, -0.00248304, 0.00139999, 0.17919876), 0.03732238, 0.31754803),
        'q64-kv128-k-tma-v-tma': ((-0.1384898, -0.00187103, 0.0011701, -0.00058724, -0.00157411, 0.00162242, 0.20766992), -0.26268832, 0.03795686),
    },
}


def _relative_transport_costs(rows, heads, kv_heads, causal, dim):
    # At Q64/KV128, a causal Q tile visits 1,1,2,2,... KV tiles.
    qtiles = (rows + 63) // 64
    mean = ((qtiles+1)**2//4)/qtiles if causal else (rows+127)//128
    terms = (1., math.log1p(mean), math.log1p(heads*qtiles/114),
             math.log2(heads/kv_heads), float(causal), float(bool(rows%128)), (rows%128)/128)
    costs = {}
    for arm, (weights, low, high) in _TRANSPORT_LOG_RATIOS[dim].items():
        width = 128 if '128' in arm else 64
        if (rows+width-1)//width < 4:
            continue
        ratio = sum(weight*term for weight,term in zip(weights,terms))
        costs[arm] = math.exp(min(high,max(low,ratio)))
    return costs


# H100 PCIe paired log(t_KV64 / t_KV128) at fixed K/V TMA and issue order.
_GROUPED_WIDTH = {
    False: (0.25, (0.0784643029, 0.9173767524, -0.1456857141, 0.0336832655, 0.0520671081, 0.0, -0.0059299213, 0.0)),
    True: (0.25, (0.1033767126, 0.2585103337, -0.0685941531, -0.0112613721, 0.0718456366, 0.0078968606, -0.0001930234, -0.3726572636)),
}


def _ends_in_singleton(depth, capacity):
    return depth > capacity and depth % capacity == 1


def _singleton_drain(blocks, sms, capacity):
    # A full resident wave followed by one CTA on an SM leaves no peer CTA
    # to hide that final CTA's load and dependency stalls.
    depth, extra = divmod(blocks, sms)
    affected = (sms - extra) * _ends_in_singleton(depth, capacity)
    affected += extra * _ends_in_singleton(depth + 1, capacity)
    return affected / min(sms, blocks)

@lru_cache(maxsize=512)
def _grouped_width_features(rows, heads, kv_heads, causal, sm_count, small_capacity, large_capacity):
    queries = (rows+63)//64
    blocks = heads*queries
    visits64 = (queries+1)/2 if causal else queries
    visits128 = ((queries+1)**2//4)/queries if causal else (rows+127)//128
    # Compare active work assuming equal service per useful key row.
    work = math.log(visits64/(2*visits128))
    drain = _singleton_drain(blocks,sm_count,small_capacity) - _singleton_drain(blocks,sm_count,large_capacity)
    path64,_ = _resident_service(rows,heads,causal,64,small_capacity,sm_count,0.5)
    path128,_ = _resident_service(rows,heads,causal,128,large_capacity,sm_count,0.5)
    critical = math.log(path64/(2*path128))
    return (1., work, 1/visits128, drain, critical, math.log2(heads/kv_heads),
            math.log1p(blocks/sm_count), math.log1p(rows*kv_heads*2*64*2/(50*1024**2)))


def _score_grouped_width(costs, rows, heads, kv_heads, causal, sm_count):
    # Compare widths only within the calibrated copy and prefetch contract.
    suffix = "-v-tma" if causal else "-post-qk-v-tma"
    small = "q64-kv64-k-tma" + suffix
    large = "q64-kv128-k-tma" + suffix
    if small not in costs or large not in costs:
        return None
    family_min = min(costs[small], costs[large])
    if family_min <= 0:
        return None
    prior = math.log(costs[small] / costs[large])
    weight, coefficients = _GROUPED_WIDTH[causal]
    terms = _grouped_width_features(rows, heads, kv_heads, causal, sm_count,
                                    PROFILE[small][2], PROFILE[large][2])
    log_ratio = weight * prior + sum(w * x for w, x in zip(coefficients, terms))
    if not causal and log_ratio < 0 <= prior:
        # At uniform loop length a singleton on idle/lightly loaded SMs
        # is insufficient. Check the last batch on the most loaded SMs
        # under striped block dispatch before overriding the wider tile.
        blocks = heads * ((rows + 63) // 64)
        depth = (blocks + sm_count - 1) // sm_count
        if not _ends_in_singleton(depth, PROFILE[large][2]) or _ends_in_singleton(depth, PROFILE[small][2]):
            log_ratio = prior
    # Preserve the comparison with other transfer/issue families.
    costs[small] = family_min * math.exp(max(0., log_ratio))
    costs[large] = family_min * math.exp(max(0., -log_ratio))
    return dict(kv64=small, kv128=large, prior_log_ratio=prior, log_ratio=log_ratio)


# H100 PCIe, ordinary D64 GQA, joint post-QK K/V TMA and KV128.
# Per-KV-iteration microseconds: base, work per CTA/SM, unfilled fraction
# of the last SM layer inside the first resident wave, and after that wave.
_GROUPED_KV128_SERVICE = {
    64: (0.476899, 0.867828, 0.347872, 0.383483),
    128: (0.399176, 1.593022, 1.008747, 1.593022),
}


def _grouped_kv128_service(query, blocks, sm_count, capacity):
    base, work, first_tail, later_tail = _GROUPED_KV128_SERVICE[query]
    depth = (blocks + sm_count - 1) // sm_count
    unfilled = (depth * sm_count - blocks) / sm_count
    # Tail coefficients are constrained to [0, work]. Increasing a grid
    # within a layer therefore never makes the predicted service cheaper.
    tail = first_tail if depth <= capacity else later_tail
    return base + work * (blocks / sm_count) + tail * unfilled


def _q128_width_log_ratio(rows, heads, sm_count):
    """Relative time of KV64 versus KV128 at Q128, with joint post-QK TMA."""
    short_visits = (rows + 63) // 64
    long_visits = (rows + 127) // 128
    layers = (heads * long_visits + sm_count - 1) // sm_count
    # Both widths have two resident Q128 CTAs per SM. Distinguish a full
    # resident pair from CTAs admitted after the first pair retires.
    resident_pair = layers == 2
    turnover = max(0, layers - 2) / layers
    # Native masks the final logits tile only when it is partial. A V
    # tensor-map box covers eight rows and repairs a shorter fringe once.
    narrow_avoids_mask = short_visits * 64 == rows and long_visits * 128 != rows
    v_fringe = (rows % 8 != 0) / long_visits
    return (0.12649841
            + 0.75461942 * math.log(short_visits / (2 * long_visits))
            - 0.33771836 / long_visits
            - 0.06933286 * resident_pair
            - 0.08603404 * turnover
            + 0.16185093 * v_fringe
            - 0.02195011 * narrow_avoids_mask)


def _score_grouped_query_tile(costs, rows, heads, sm_count, *, incumbent=None, paired_cost=None):
    """Compare two Q tiles at the same KV width, transports and issue edge."""
    incumbent = incumbent or min(costs, key=costs.get)
    if not (incumbent.startswith("q64-") and incumbent.endswith("-v-tma")):
        return None
    _, width, q64_capacity = PROFILE[incumbent]
    # Busiest-SM CTA count and wave count at driver-measured residency.
    # Q128 has two consumer warpgroups per CTA and two resident CTAs per SM.
    blocks64 = ((rows + 63) // 64) * heads
    blocks128 = ((rows + 127) // 128) * heads
    small = (blocks64 + sm_count - 1) // sm_count
    large = (blocks128 + sm_count - 1) // sm_count
    waves_small = (small + q64_capacity - 1) // q64_capacity
    waves_large = (large + 1) // 2
    paired = incumbent.replace("q64-", "q128-", 1)
    comparison = dict(q64=incumbent, q128=paired, critical_ctas=[small, large],
                      resident_waves=[waves_small, waves_large])
    if width == 128 and "-post-qk" in incumbent:
        service64 = _grouped_kv128_service(64, blocks64, sm_count, q64_capacity)
        service128 = _grouped_kv128_service(128, blocks128, sm_count, 2)
        # Both arms visit the same K/V tiles. Preserve the incumbent's cost
        # scale when comparing this pair with the other transport families.
        log_ratio = math.log(service128 / service64)
        comparison.update(cta_counts=[blocks64, blocks128],
                          service_us_per_iteration=[service64, service128])
    elif paired_cost is not None:
        log_ratio = math.log(paired_cost / costs[incumbent])
    else:
        log_ratio = (-0.140642 + 0.376642 * math.log(2 * large / small)
                     + 0.167392 * math.log(waves_large / waves_small)
                     + 0.152911 * (width / 64 - 1))
    costs[paired] = costs[incumbent] * math.exp(log_ratio)
    comparison["log_ratio"] = log_ratio
    return comparison


# H100 PCIe joint post-QK K/V TMA. Fixed first/final visits and steady
# interior visits are calibrated separately for each emitted tile pair.
_LIGHT_GRID_SERVICE = {
    (64, 64): (4.1517554, 0.5067115, 0.6425003, 0.1565498, 0.1796390, 0.4312745, 0.7232314, 0.1711417, 0.3251506),
    (64, 128): (4.4640136, 0.8180362, 2.1709374, 0.0741686, 0.4985423, 0.4058305, 0.5538914, 0.4059189, 0.7958888),
    (128, 64): (5.0879252, 0.7650655, 0.1745367, 0.1829804, 0.2931005, 0.5954801, 0.5897728, 0.4394560, 0.0000000),
    (128, 128): (6.4803148, 1.3356519, 0.8771308, 0.2638989, 0.4923300, 0.5675057, 0.3507102, 1.1770509, 0.0000000),
}


def _light_grid_tile_time(rows, heads, sm_count, query, width):
    query_tiles = (rows + query - 1) // query
    depth = heads * query_tiles / sm_count
    fill = min(1.0, depth)
    excess = max(0.0, depth - 1.0)
    interior = (rows + width - 1) // width - 2
    terms = (1.0, interior, fill,
                 interior * fill, interior * excess, rows % width != 0,
                 rows % 8 != 0, interior * (depth > 1), interior * max(0.0, depth - 2))
    return sum(cost * term for cost, term in zip(_LIGHT_GRID_SERVICE[query, width], terms))


def _apply_light_grid_service(scores, rows, heads, sm_count, pairs, kv_heads):
    # Fraction of SMs with no second Q128 CTA, capped at a single layer.
    blocks = heads * ((rows + 127) // 128)
    weight = min(1.0, 2.0 - blocks / sm_count)
    if weight <= 0:
        return None
    times = {}
    for width, pair in pairs:
        for query in (64, 128):
            arm = pair[f"q{query}"]
            times[arm] = _light_grid_tile_time(rows, heads, sm_count, query, width)
    shortest = min(times.values())
    if shortest <= 0:
        return None
    # Preserve the absolute family floor used to compare copy/issue paths.
    floor = min(scores.values())
    relative = {arm: (1 - weight) * scores[arm] / floor + weight * time / shortest
                for arm, time in times.items()}
    # CTAs from sibling query heads can issue concurrent TMA reads of the
    # same KV rows. Q64 issues more such requests than Q128. Calibrate the
    # relative Q cost at a fixed width, retaining both within-Q KV rankings.
    log_adjustment = 0.0
    group_size = heads // kv_heads if kv_heads is not None else 1
    if group_size > 1:
        log_adjustment = weight * (-0.05734939 + 0.03912539 * math.log2(group_size))
        for _, pair in pairs:
            relative[pair["q64"]] *= math.exp(log_adjustment)
    best = min(relative.values())
    scores.update({arm: floor * ratio / best for arm, ratio in relative.items()})
    return dict(weight=weight, service_us=times, kv_group_size=group_size,
                q64_over_q128_log_adjustment=log_adjustment)


# H100 PCIe, post-QK V prefetch, two slots, direct FP32 output. The first
# fit estimates SW32 service per KV visit. The second estimates the log time
# ratio measured at fixed geometry when V changes from SW32 to SW128.
_GQA_SW32_VISIT_COST = {
    (64, 64): (2.343587602, -0.035427881, 0.361818822, 0.224552814, 0.317799408, 0.354767103, 0.226687010),
    (64, 128): (1.725788197, -0.103915421, 0.632694818, 0.322855009, 0.528209699, 1.359765170, 0.506311323),
    (128, 64): (2.182256814, -0.065715303, 0.661983779, 0.492470790, 0.402120549, 1.049144661, 0.045054926),
    (128, 128): (1.759627754, 0.069553626, 1.257103189, 0.983999417, 0.492072192, 1.247351233, 0.138702711),
}

_GQA_V_LOG_TIME_RATIO = {
    (64, 64): (0.045898051, -0.213586271, 0.035898787, -0.059502571, -0.175611823),
    (64, 128): (0.118373275, -0.507071287, 0.016955909, -0.087358731, -0.197301281),
    (128, 64): (0.004929169, -0.034174695, 0.018553531, -0.020977121, -0.031552086),
    (128, 128): (0.004584806, -0.119384923, 0.041475969, -0.035530405, -0.033688801),
}


# H100 PCIe. Rows at each tile are SW32 then SW128, calibrated with paired
# timing at controlled CTA grids. Each slot term counts critical-SM visits.
_V_LAYER_SERVICE = {
    (64, 64): (
        (3.7269281470, 0.2098571203, 0.1037900542, 0.2931253085, 0.1628853579, 0.2465815117, 0.3315514650, 0.3361308313, 0.0913278200, -0.0140095485, -0.0037564689, 0.0479415387, -0.0171819536, 0.1557139697, 0.1557139697, -0.3443507283, 0.6268717175, -0.0470025545, -0.1426681521, 0.0479548780, 0.0505764072, 0.7019588915, 0.6870095588, -1.2945473485, -0.2447962833),
        (2.7526072446, 0.1796069710, 0.1124712255, 0.2957618676, 0.2037442576, 0.2808333875, 0.3847602705, 0.3743394200, 0.0615108423, 0.0208005629, 0.0019785684, 0.0026187736, -0.0006202090, 0.0669646912, 0.0669646912, -0.0611919325, 0.0396466829, 0.0106581155, -0.2052132043, -0.0024716387, -0.3182865116, 0.5298199047, 0.0812168362, 0.4490982683, -0.1659894385),
    ),
    (64, 128): (
        (3.5580500377, 0.2222757548, 0.2676028503, 0.4872800683, 0.4839563574, 0.6832614601, 0.2245755699, -0.0815484752, 0.2329653382, 0.0391483539, 0.0081391122, -0.1179891651, 0.6596846985, -0.5891781080, 0.2983213634, -0.0543154914, -0.6340396167, 0.0494546250, 0.0323774960, 0.1157493013, 1.0519498792, -0.7690372226, -0.0386263772),
        (3.8524879549, 0.1361941682, 0.2624198281, 0.4494999795, 0.4987722036, 0.7495237093, 0.1858361611, 0.0145038926, -0.0856647794, 0.0162314167, -0.0027745816, 0.5787986816, -0.1772087280, -0.0710440292, 0.1678225950, 0.0199005035, -0.5871058961, -0.0192809769, -0.2890136693, 0.6431160363, 0.0246757462, -0.3133408309, -0.3513544160),
    ),
    (128, 64): (
        (3.9168887913, 0.1416919346, 0.4216584575, 0.4117322018, 0.5622007014, 0.2429196634, -0.0493777250, 0.0205327273, 0.0454787972, 0.0033369109, 0.3029212593, -0.2566903333, -0.0527339315, 0.0778695021, -0.0023514588, -0.5364461166, -0.0050812885, -0.0850351540, -0.3718236957, 0.0357454371, 0.6217588652, -0.4049408330),
        (3.3689116851, 0.1286074917, 0.4190606974, 0.4110308213, 0.5689547130, 0.2599278542, -0.0021079866, -0.0100795036, 0.0104653942, 0.0033369905, 0.2547976424, -0.1899155851, -0.0683570849, 0.0569489452, 0.0033931910, -0.5461868645, -0.0120100557, -0.3625415840, 0.1365876993, -0.3479139251, 0.8453881314, -0.2929091832),
    ),
    (128, 128): (
        (4.2600964099, 0.2104673705, 0.7878321746, 0.7186057096, 1.1656334847, 0.4324946276, -0.1056122499, 0.0794089979, 0.0782409728, -0.0068856949, 0.2456758454, 0.2456758454, -0.1185772840, 0.0820354987, 0.0059949720, -0.8260555349, -0.0241113730, -0.1403418143, 0.0908637922, 0.0744087704, 0.0100060313, -0.7526837710),
        (3.7013000530, 0.1857616948, 0.7831407017, 0.7143260409, 1.1906949599, 0.4878823789, -0.0327743600, 0.0044550168, 0.0353621436, -0.0049519159, 0.1082504046, 0.1082504046, -0.1538050384, 0.0513403213, -0.0073631372, -0.8492660741, -0.0243356819, -0.4265439372, 0.6556095672, -0.3217424845, 0.0929875305, -0.5112567770),
    ),
}
_V_WIDTH_TIE_LOG_RATIO = 0.0076255275
_V_PAIRED_WIDTH = (-0.1641334774, 0.3843215693, 0.6457989491, 0.2818488272, -0.0894993404, 0.0743183142, -0.0027748872, 0.0959836406)


def _v_layer_services(rows, heads, group, sm_count, query, width, capacity):
    """Predict SW32/SW128 service on the SM that gets the last CTA."""
    visits = math.ceil(rows / width)
    qtiles = math.ceil(rows / query)
    blocks = heads * qtiles
    depth = blocks / sm_count
    layers = math.ceil(depth)
    waves, remainder = divmod(layers, capacity)
    slots = [waves + int(i < remainder) for i in range(capacity)]
    last_wave = math.ceil(layers / capacity)
    hole = layers - depth
    sharing = math.log2(group) / 4
    partial_v = int(rows % 8 != 0)
    padding = (visits * width - rows) / (visits * width)
    features = (
        1 / visits, 1., depth, *slots, hole, min(depth, 1.), sharing, partial_v,
        partial_v * depth, padding, (qtiles * query - rows) / (qtiles * query),
        hole * min(1., depth / capacity), sharing * min(depth / capacity, 2.),
        partial_v * hole, ((layers - 1) % capacity) / capacity, partial_v * sharing,
        layers / visits, last_wave / visits, depth / visits,
        int(rows % width != 0) * last_wave / visits, padding * layers,
    )
    gains = _V_LAYER_SERVICE[query, width]
    return tuple(visits * max(.25, sum(c * x for c, x in zip(coeffs, features)))
                 for coeffs in gains)


def _price_v_resident_layers(scores, pairs, base, rows, heads, kv_heads, sm_count):
    """Correct the paired V transfer ratio and its baseline on the same score scale."""
    before = dict(scores)
    service = {}
    group = heads / kv_heads
    for width, pair in pairs:
        for query in (64, 128):
            arm = pair[f"q{query}"]
            capacity = PROFILE[arm][2] if query == 64 else 2
            service[arm] = _v_layer_services(rows, heads, group, sm_count, query, width, capacity)
    scale = min(base.values()) / min(v[0] for v in service.values())
    depth = heads * math.ceil(rows / 64) / sm_count
    narrow, wide = (pair["q64"] for _, pair in pairs)
    narrow_cap, wide_cap = PROFILE[narrow][2], PROFILE[wide][2]
    # A single last wave has less leverage as the number of waves grows.
    weight = min(1., wide_cap / depth)
    base_adjust = {}
    ratio_adjust = {}
    for arm, (old_service, new_service) in service.items():
        modeled_ratio = min(1.15, max(.42, new_service / old_service))
        base_adjust[arm] = math.log(scale * old_service / base[arm])
        ratio_adjust[arm] = math.log(modeled_ratio / (before[arm] / base[arm]))
    # Both Q128 widths have two resident CTAs. Their critical layer cost is
    # common at saturation; keep the existing within-Q width comparison.
    group128 = [pair["q128"] for _, pair in pairs]
    width_weight = max(0., min(1., 2. - heads * math.ceil(rows / 128) / sm_count))
    for adjustments in (base_adjust, ratio_adjust):
        common = sum(adjustments[a] for a in group128) / len(group128)
        for arm in group128:
            adjustments[arm] = common + width_weight * (adjustments[arm] - common)
    without_v = {a: base[a] * math.exp(weight * base_adjust[a]) for a in scores}
    for a in scores:
        scores[a] = before[a] * math.exp(weight * (base_adjust[a] + ratio_adjust[a]))
    layer_scores = dict(scores)
    # Only an extra padded half-box AND an extra resident wave justify
    # replacing a small within-Q margin by its direct paired width estimate.
    small_width, large_width = (w for w, _ in pairs)
    small_visits = math.ceil(rows / small_width)
    large_visits = math.ceil(rows / large_width)
    layers = math.ceil(depth)
    extra = (large_visits * large_width - small_visits * small_width) / (small_visits * small_width)
    wave_diff = math.ceil(layers / wide_cap) - math.ceil(layers / narrow_cap)
    margin = math.log(scores[wide] / scores[narrow])
    width_feedback = max(0., 1. - abs(margin) / .05) if extra and wave_diff > 0 else 0.
    if width_feedback:
        terms = (1., 1 / small_visits, extra, wave_diff / layers, 1 / layers,
                 (layers - depth) / layers, math.log2(group) / 4, min(1., depth / wide_cap))
        paired_margin = sum(c * x for c, x in zip(_V_PAIRED_WIDTH, terms))
        half_delta = .5 * width_feedback * (paired_margin - margin)
        scores[narrow] *= math.exp(-half_delta)
        scores[wide] *= math.exp(half_delta)
    # Without a wave difference, keep the reference width if the new margin
    # is inside the held-out paired-width error band. Preserve the pair minimum.
    prior_margin = math.log(before[wide] / before[narrow])
    tied = wave_diff == 0 and abs(margin) < _V_WIDTH_TIE_LOG_RATIO and margin * prior_margin < 0
    if tied:
        scores[narrow], scores[wide] = scores[wide], scores[narrow]
    return dict(service_us={a: list(v) for a, v in service.items()}, layer_weight=weight,
                retained_width_in_tie=tied, width_weight=width_feedback,
                layer_scores_us=layer_scores, without_transfer_us=without_v,
                effective_time_ratio={a: layer_scores[a] / without_v[a] for a in scores},
                before_us=before, extra_half_box=bool(extra), extra_waves=wave_diff)



def _score_row_major_v(scores, pairs, rows, heads, kv_heads, sm_count):
    """Price the SW128 V path against the SW32 service model at the same geometry."""
    previous = dict(scores)
    group_sharing = math.log2(heads / kv_heads) / 4
    service = {}
    transfer = {}
    for width, pair in pairs:
        visits = math.ceil(rows / width)
        for query in (64, 128):
            arm = pair[f"q{query}"]
            ctas = heads * math.ceil(rows / query)
            depth = ctas / sm_count
            layers = math.ceil(depth)
            capacity = 2 if query == 128 else PROFILE[arm][2]
            live = min(depth, capacity)
            work_features = (
                1 / visits, 1., depth, layers - depth, math.ceil(layers / capacity),
                int(rows % width != 0) / visits, group_sharing,
            )
            transfer_features = (1., 1 / visits, live / capacity, min(live, 1.), group_sharing)
            work_coeffs = _GQA_SW32_VISIT_COST[query, width]
            ratio_coeffs = _GQA_V_LOG_TIME_RATIO[query, width]
            per_visit = sum(c * x for c, x in zip(work_coeffs, work_features))
            log_ratio = sum(c * x for c, x in zip(ratio_coeffs, transfer_features))
            service[arm] = visits * max(.25, per_visit)
            transfer[arm] = math.exp(min(math.log(1.10), max(math.log(.50), log_ratio)))

    # Keep the family on the common score scale used by the other transport paths.
    scale = min(previous.values()) / min(service.values())
    without_transfer = {}
    width_weights = {}
    for query in (64, 128):
        narrow, wide = (pair[f"q{query}"] for _, pair in pairs)
        capacity = 2 if query == 128 else PROFILE[wide][2]
        depth = heads * math.ceil(rows / query) / sm_count
        # A full resident grid already has a calibrated within-Q width ratio.
        # Below capacity, progressively use the measured transport sensitivity.
        weight = max(0., min(1., capacity - depth))
        width_weights[query] = weight
        service_ratio = {arm: scale * service[arm] / previous[arm] for arm in (narrow, wide)}
        mean_service = math.sqrt(service_ratio[narrow] * service_ratio[wide])
        mean_transfer = math.sqrt(transfer[narrow] * transfer[wide])
        for arm in (narrow, wide):
            without_transfer[arm] = previous[arm] * mean_service * (service_ratio[arm] / mean_service) ** weight
            scores[arm] = without_transfer[arm] * mean_transfer * (transfer[arm] / mean_transfer) ** weight

    resident = _price_v_resident_layers(scores, pairs, without_transfer, rows, heads, kv_heads, sm_count)
    resident["prior_without_transfer_us"] = without_transfer
    return dict(swizzle_bytes=128, tensor_map_rank=3, kv_slots=2, resident=resident,
                sw32_estimated_us=previous, sw32_service_us=service,
                without_transfer_us=resident["without_transfer_us"], time_ratio_to_sw32=resident["effective_time_ratio"],
                width_weights={f"q{q}": value for q, value in width_weights.items()})


def _score_joint_grouped_tiles(costs, rows, heads, sm_count, *, kv_heads=None):
    """Put every legal post-QK (Q, KV) combination on one time scale."""
    narrow = "q64-kv64-k-tma-post-qk-v-tma"
    wide = "q64-kv128-k-tma-post-qk-v-tma"
    original = min(costs, key=costs.get)
    other = None
    if original not in (narrow, wide) or narrow not in costs or wide not in costs:
        other = _score_grouped_query_tile(costs, rows, heads, sm_count, incumbent=original)
    if narrow not in costs or wide not in costs:
        return other, None

    # Q64 width costs include the calibrated grouped communication cost.
    # The KV128 service model supplies Q128/KV128 on that same scale.
    wide_pair = _score_grouped_query_tile(costs, rows, heads, sm_count, incumbent=wide)
    width_ratio = _q128_width_log_ratio(rows, heads, sm_count)
    narrow_cost = costs[wide_pair["q128"]] * math.exp(width_ratio)
    narrow_pair = _score_grouped_query_tile(costs, rows, heads, sm_count,
                                           incumbent=narrow, paired_cost=narrow_cost)
    scores = {name: costs[name] for pair in (narrow_pair, wide_pair)
              for name in (pair["q64"], pair["q128"])}
    light_grid = _apply_light_grid_service(scores, rows, heads, sm_count,
                                           ((64, narrow_pair), (128, wide_pair)), kv_heads)
    row_major_v = None
    if kv_heads is not None:
        row_major_v = _score_row_major_v(scores, ((64, narrow_pair), (128, wide_pair)),
                                        rows, heads, kv_heads, sm_count)
    if light_grid or row_major_v:
        costs.update(scores)
        for pair in (narrow_pair, wide_pair):
            pair["log_ratio"] = math.log(scores[pair["q128"]] / scores[pair["q64"]])
        width_ratio = math.log(scores[narrow_pair["q128"]] / scores[wide_pair["q128"]])
    winner = min(costs, key=costs.get)
    comparison = other
    if winner in scores:
        comparison = narrow_pair if winner in (narrow, narrow_pair["q128"]) else wide_pair
    joint = dict(estimated_us=scores, selected=min(scores, key=scores.get),
                 q128_kv64_over_kv128_log_ratio=width_ratio)
    if row_major_v:
        joint["v_transport"] = row_major_v
    if light_grid:
        joint["light_grid"] = light_grid
    return comparison, joint


def rank_grouped_attention(rows, heads, kv_heads, dim, causal, sm_count=114, *, output_bits=32):
    """Packed square GQA; head counts include the batch, with contiguous KV groups."""
    if (type(heads) is not int or type(kv_heads) is not int or kv_heads<1
            or heads<=kv_heads or heads%kv_heads or type(rows) is not int or rows<1
            or dim not in (64,128) or sm_count!=114):
        raise ValueError("expected a packed D64/D128 grouped launch on H100 PCIe")
    if dim==64:
        return rank_d64_pipelines(rows,heads,causal,sm_count,kv_heads=kv_heads,output_bits=output_bits)
    if output_bits!=32:
        raise ValueError("D128 FP16 GQA uses compile_static_fa3 with kv_heads")
    costs = {"cp-q64-producer-wait4-late-v":1.0}
    costs.update(_relative_transport_costs(rows,heads,kv_heads,bool(causal),128))
    return dict(selected=min(costs,key=costs.get), estimated_ratio=costs,
                kv_heads=kv_heads, query_heads=heads, sm_count=sm_count)


def rank_d64_pipelines(rows, heads, causal, sm_count=114, *, kv_heads=None, causal_v_tma=True, output_bits=32):
    """Rank legal compiled families without timing or solving any candidate."""
    if type(rows) is not int or rows <= 0 or type(heads) is not int or heads <= 0:
        raise ValueError("positive sequence and head counts are required")
    if sm_count != 114:
        raise ValueError("D64 static scheduling requires the H100 PCIe 114-SM profile")
    kv_heads = heads if kv_heads is None else kv_heads
    if type(kv_heads) is not int or not 1<=kv_heads<=heads or heads%kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    costs = {}
    if output_bits not in (16,32):
        raise ValueError("expected FP16 or FP32 output")
    is_causal = bool(causal)
    for arm, (q,k,capacity) in PROFILE.items():
        if output_bits==16 and arm.startswith("cp-") and arm!="cp-q64-producer-wait4-late-v":
            continue
        if (arm,is_causal) not in COEFFICIENTS:
            continue
        # The native two-version emitter requires at least four KV tiles.
        if not arm.startswith('cp-') and (rows+k-1)//k < 4:
            continue
        coefs = COEFFICIENTS[arm,is_causal]
        x = features(rows,heads,is_causal,q,k,capacity,sm_count)
        costs[arm] = sum(a*b for a,b in zip(x,coefs))
    _score_kv_pairs(costs, rows, heads, is_causal, sm_count)
    if is_causal:
        _compare_causal_transport(costs,rows,heads,sm_count)
        if causal_v_tma:
            for arm, ratio in _relative_transport_costs(rows,heads,kv_heads,True,64).items():
                same_k = arm.removesuffix("-v-tma")
                costs[arm] = costs[same_k] * ratio
    comparison = None
    if kv_heads < heads:
        comparison = _score_grouped_width(costs, rows, heads, kv_heads, is_causal, sm_count)
    geometry = joint = None
    if kv_heads < heads and not is_causal and output_bits == 32:
        geometry, joint = _score_joint_grouped_tiles(costs, rows, heads, sm_count, kv_heads=kv_heads)
    result = dict(selected=min(costs,key=costs.get),estimated_us=costs,sm_count=sm_count)
    if joint is not None:
        result["grouped_tiles"] = joint
    if geometry is not None:
        result["grouped_query_tile"] = geometry
    if comparison is not None:
        result["grouped_width"] = comparison
    return result
