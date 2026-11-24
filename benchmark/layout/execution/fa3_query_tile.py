"""H100 PCIe FA3 query-tile selection with one resident CTA per SM.

Use the actual Q/K tile ratio for causal trips. Return launch-time estimates;
the older Q128 event score has a different unit and is only ranked within Q128.

Q64 normalize produces FP16 O from FP32 numerator/denominator. Away from
zero and infinity the two binary64 roundings have relative error < 2**-52.
Write the exact quotient as A/B times a power of two, with <=24-bit integer
significands. At a normal binary32 midpoint the nonzero integer cross-product
difference puts the quotient at relative distance >= 2**-50; equality cannot
occur because a reduced dyadic quotient has at most 24 significant bits.
Thus rounding the wide estimate to binary32 preserves the reference on every
binade near a binary16 boundary. At binary32 underflow both paths narrow to
the same signed half zero. IEEE zero, infinity and NaN convert directly.
This argument is for the two casts in this emitter, not an FP32 output ABI.
"""
import heapq
import math

from fa3_event_schedule import SCHEDULES

# Per-configuration coefficients fitted to paired SM90a full-tile launches.
# Features: fixed launch, critical CTA count, critical KV trips, total trips per SM.
_COEFFICIENTS = {64: {'register': [[1.78152, 3.455516, 1.208659, 0.401946],
                   [1.033898, 3.844483, 1.454569, 0.0],
                   [1.283228, 3.506295, 1.679213, 0.0],
                   [1.008676, 3.889344, 1.439202, 0.0]],
      'tma': [[1.156471, 3.260474, 1.514383, 0.0],
              [1.221787, 3.589646, 1.373341, 0.0],
              [1.352104, 3.379022, 1.572389, 0.0],
              [1.164556, 3.68348, 1.348564, 0.0]]},
 128: {'register': [[0.0, 6.834194, 1.67479, 0.715598],
                    [0.271706, 6.618891, 1.677232, 0.492329],
                    [0.349305, 6.529308, 1.761773, 0.559462],
                    [0.2144, 6.771991, 1.606665, 0.580553]],
       'tma': [[0.0, 7.317388, 1.864051, 0.412462],
               [0.0, 7.868095, 1.976633, 0.196276],
               [0.0, 8.180168, 2.01599, 0.244688],
               [0.0, 7.89674, 1.92482, 0.290847]]}}


_TILE_LOG_RATIO_ERROR = 0.04045686


def dispatch_query_tiles(query_tiles, heads, reverse):
    """Logical query coordinates in the native linear CTA admission order."""
    if reverse:
        return (row for row in reversed(range(query_tiles)) for _ in range(heads))
    return (row for _ in range(heads) for row in range(query_tiles))


def launch_features(rows, keys, heads, causal, query_tile, sm_count=114, *, reverse=False):
    if query_tile not in (64,128) or min(rows,keys,heads,sm_count)<1:
        raise ValueError("invalid FA3 tile or launch geometry")
    qt=(rows+query_tile-1)//query_tile
    kt=(keys+127)//128
    trips=[min(kt,(min((i+1)*query_tile,rows)+127)//128) if causal else kt for i in range(qt)]
    # A CTA incurs one startup/drain plus its active KV trips. Assign the
    # next CTA to the first available SM (X varies fastest in the native grid).
    slots=[(0,0,0,i) for i in range(min(qt*heads,sm_count))]
    # Reverse maps consecutive physical CTAs to heads of the largest Q tile.
    for row in dispatch_query_tiles(qt, heads, reverse):
        count = trips[row]
        _,blocks,work,slot=slots[0]
        blocks+=1
        work+=count
        heapq.heapreplace(slots,(3*blocks+work,blocks,work,slot))
    _,blocks,work,_=max(slots)
    return (1.0,blocks,work,sum(trips)*heads/sm_count)


def rank_cta_dispatch(rows, keys, heads, causal, query_tile, *, mode="auto", sm_count=114):
    """Compare the critical SM's service in the two native physical grid orders."""
    if mode not in ("auto", "forward", "reverse") or (mode=="reverse" and not causal):
        raise ValueError("reverse CTA dispatch requires causal attention and a valid order")
    service = {}
    orders = ("forward","reverse") if causal else ("forward",)
    for order in orders:
        _, blocks, work, _ = launch_features(rows,keys,heads,causal,query_tile,sm_count,
                                            reverse=order=="reverse")
        service[order] = 3*blocks + work
    gain = math.log(service["forward"]/service.get("reverse",service["forward"]))
    selected = ("reverse" if gain > _TILE_LOG_RATIO_ERROR else "forward") if mode=="auto" else mode
    return dict(selected=selected, mode=mode, critical_service=service, sm_count=sm_count,
                log_ratio=gain, log_margin=_TILE_LOG_RATIO_ERROR)


def _launch_costs(features, query_tile, writer):
    return {name:sum(a*b for a,b in zip(features,coef))
            for name,coef in zip(SCHEDULES,_COEFFICIENTS[query_tile][writer])}


def launch_costs(rows,keys,heads,causal,query_tile,writeback,*,cta_dispatch="forward"):
    if cta_dispatch not in ("forward","reverse") or (cta_dispatch=="reverse" and not causal):
        raise ValueError("invalid physical CTA dispatch")
    features=launch_features(rows,keys,heads,causal,query_tile,reverse=cta_dispatch=="reverse")
    return _launch_costs(features,query_tile,writeback)


def select_query_geometry(rows, keys, heads, causal, writeback="auto", *,
                          cta_dispatch="auto", query_tile=None, sm_count=114, guarded_query=False):
    """Rank geometry and dispatch together in launch microseconds before layout."""
    if writeback not in ("auto","register","tma") or cta_dispatch not in ("auto","forward","reverse"):
        raise ValueError("unknown FA3 writer or dispatch")
    if cta_dispatch=="reverse" and not causal:
        raise ValueError("reverse dispatch requires causal attention")
    if query_tile not in (None,64,128):
        raise ValueError("unknown FA3 query geometry")
    writers=("register","tma") if writeback=="auto" else (writeback,)
    orders=(("forward","reverse") if causal else ("forward",)) if cta_dispatch=="auto" else (cta_dispatch,)
    candidates=[]
    for tile in ((64,128) if query_tile is None else (query_tile,)):
        eligible=[w for w in writers if rows%tile==0 or ((causal or guarded_query) and w=="register")]
        if not eligible:
            continue
        for order in orders:
            features=launch_features(rows,keys,heads,causal,tile,sm_count,reverse=order=="reverse")
            estimate=min(min(_launch_costs(features,tile,w).values()) for w in eligible)
            candidates.append(dict(query_tile=tile,cta_dispatch=order,estimated_us=estimate,
                                   eligible_writers=eligible,kv_trips_per_sm=features[3]))
    if not candidates:
        raise ValueError("automatic FA3 query tiling requires complete query tiles or a guarded causal writer")
    best=min(c["estimated_us"] for c in candidates)
    # Compare every legal pair against one minimum. The same uncertainty band
    # applies once; prefer lower aggregate K/V traffic, then ordinary grid order.
    near=[c for c in candidates if math.log(c["estimated_us"]/best)<=_TILE_LOG_RATIO_ERROR]
    winner=min(near,key=lambda c:(c["kv_trips_per_sm"],c["cta_dispatch"]!="forward",c["estimated_us"]))
    return dict(query_tile=winner["query_tile"],cta_dispatch=winner["cta_dispatch"],
                candidates=candidates,log_margin=_TILE_LOG_RATIO_ERROR,sm_count=sm_count)


def select_query_tile(rows,keys,heads,causal,writeback="auto"):
    return select_query_geometry(rows,keys,heads,causal,writeback)["query_tile"]
