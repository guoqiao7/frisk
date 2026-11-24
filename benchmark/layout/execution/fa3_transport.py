"""Effective transport of the certified SM90 FA3 main tile, shared by ranking and emission."""
from dataclasses import dataclass
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class FA3Transport:
    # Immutable even when a caller edits a returned receipt.
    encoded: str

    @property
    def facts(self):
        return json.loads(self.encoded)

    @property
    def fingerprint(self):
        return hashlib.sha256(self.encoded.encode()).hexdigest()


def register_store_work(holders, threads, valid_elements=None):
    """Count active warp-issued u32 stores and sectors on a row-aligned tile."""
    logical = holders["logical"]
    if valid_elements is None:
        valid_elements = len(logical)
    if not 0 < valid_elements <= len(logical) or valid_elements % 2:
        raise ValueError("invalid register store extent")
    if threads % 32 or len(logical) % threads:
        raise ValueError("invalid register store domain")
    per_thread = len(logical) // threads
    if per_thread % 2:
        raise ValueError("native register writer requires packed f16 pairs")
    issued = sectors = 0
    for warp in range(0, threads, 32):
        for reg in range(0, per_thread, 2):
            touched = set()
            for lane in range(warp, warp+32):
                i = lane*per_thread+reg
                first = logical[i]
                if first % 2 or logical[i+1] != first+1:
                    raise ValueError("output holder map cannot use the native u32 store")
                if first < valid_elements:
                    touched.add(first*2//32)
            issued += bool(touched)
            sectors += len(touched)
    return dict(bytes=valid_elements*2, warp_stores=issued, sector_requests=sectors,
                width_bytes=4, per_lane_stores=per_thread//2)


def resolve_verified_transport(data, plan, writeback, graph_id):
    """Called after verification of the packed FP16 graph and execution domains."""
    if writeback not in ("tma", "register"):
        raise ValueError("unsupported FA3 writeback")
    ops = data["operations"]
    values = {value["id"]: value for value in data["values"]}
    mma = {op["attrs"]["frisk.execution_role"]: op for op in ops if op["name"] == "frisk.mma"}
    copies = {op["attrs"]["frisk.execution_role"]: op for op in ops if op["name"] == "frisk.copy"}
    store, = [op for op in ops if op["name"] == "frisk.tile_store"]
    groups, roles = plan["groups"], plan["roles"]
    threads = groups[roles["qk"]]["threads"]
    anchors = {}
    for role, form in (("qk", "ss"), ("pv", "rs")):
        op = mma[role]
        attr = op["attrs"]
        cert = attr["frisk.mma_contract"]
        if (attr["frisk.execution_threads"], cert["form"], cert["grid"], cert["repeats"], cert["atom"]) != (
                threads, form, [threads//128, 1], [1, 1, plan["batch"]], [64, 128, 16]):
            raise ValueError("unsupported FA3 MMA layout")
        anchors[role] = dict(group=roles[role], threads=threads, form=form, grid=cert["grid"],
                             certificate_sha256=digest(cert))

    moves = {}
    for role, use, operand, major, blocks in (("q","qk",0,"k",threads//128), ("k","qk",1,"k",1), ("v","pv",1,"mn",1)):
        op = copies[role]
        consumer = mma[use]
        dst = op["operands"][1]
        if consumer["operands"][operand] != dst:
            raise ValueError("FA3 copy destination does not feed the MMA operand")
        cert = consumer["attrs"]["frisk.mma_contract"]
        desc = cert["a_descriptor" if operand == 0 else "b_descriptor"]
        if (desc["major"], desc["swizzle"]) != (major,128):
            raise ValueError("unsupported FA3 TMA descriptor")
        entries = iter(desc["entries"])
        starts = {(m,k): start for m,k,start,phase in zip(entries,entries,entries,entries) if phase == 0}
        gid = roles[role]
        moves[role] = dict(path="tma_tensor_4d", src=op["operands"][0], dst=dst, consumer=use,
                           issuer_group=gid, issuer_local=0, issuer_cta=groups[gid]["base"],
                           buffer=plan["buffers"][role], ready=plan["edges"][role+"_ready"],
                           free=plan["edges"].get(role+"_free"), completion="mbarrier_expect_tx_parity",
                           box=[64,plan["buffers"][role]["tile_bytes"]//256,1,1], boxes=2, element_bytes=2, swizzle=desc["swizzle"],
                           major=major, leading=desc["leading"], stride=desc["stride"],
                           starts=[starts[m*64,i*16] for m in range(blocks) for i in range(plan["batch"])],
                           generic_copy=op["attrs"]["frisk.copy_contract"])

    module, = (op for op in ops if op["name"]=="builtin.module")
    launch=module["attrs"]["frisk.online_launch"]
    head_group=launch["heads"]//launch["kv_heads"]
    if head_group>1:
        for role in ("k","v"):
            moves[role]["query_heads_per_kv_head"]=head_group
    holders = values[store["operands"][0]]["distributed"]
    gid = roles["pv"]
    staged = writeback == "tma"
    moves["o"] = dict(path="bulk_shared_to_global" if staged else "register_store_u32",
                      src=store["operands"][0], dst=store["operands"][1], holder_sha256=digest(holders),
                      topology=holders["topology"], issuer_group=gid, issuer_local=0 if staged else "all",
                      issuer_cta=groups[gid]["base"] if staged else None,
                      buffer=plan["buffers"]["o"] if staged else None,
                      completion="bulk_commit_wait_read_0" if staged else "synchronous",
                      staging="stmatrix_x4" if staged else None,
                      alias_barrier=2 if staged and plan["buffers"]["o"]["offset"] == plan["buffers"]["q"]["offset"] else None,
                      publish_barrier=3 if staged else None, async_proxy_fence=staged,
                      bulk_wait_groups=0 if staged else None, bulk_wait_reads=staged)
    if not staged:
        moves["o"]["store_work"] = register_store_work(holders, threads)
        module, = [op for op in ops if op["name"] == "builtin.module"]
        remaining = module["attrs"]["frisk.online_launch"]["rows"] % moves["q"]["box"][1]
        if remaining:
            moves["o"]["tail_store_work"] = register_store_work(holders, threads, remaining*128)
    facts = dict(version=1, scope="main_kernel bulk-tile body", graph_sha256=graph_id,
                 execution=plan, anchors=anchors, moves=moves, writeback=writeback,
                 slot="tile % versions", phase="(tile // versions) & 1")
    return FA3Transport(canonical(facts))


@dataclass(frozen=True)
class _TileWork:
    query_tiles: int
    key_tiles: int
    causal: bool
    queries_per_key: int = 1

    def after(self, prefix):
        from fa3_static_schedule import _after_prefix
        full,extra=divmod(self.query_tiles,self.queries_per_key)
        return (self.queries_per_key*_after_prefix(full,self.key_tiles,self.causal,prefix) +
                extra*max(0,(min(full+1,self.key_tiles) if self.causal else self.key_tiles)-prefix))


def rank_transport_events(transport, rows, keys, causal, budget, *, heads, cta_dispatch="forward"):
    """Select the calibrated profile by the actual paths, and bill its physical work."""
    from fa3_static_schedule import rank_fa3_events
    facts = transport.facts
    moves = facts["moves"]
    profile = tuple(moves[r]["path"] for r in ("q","k","v","o"))
    register = profile[-1] == "register_store_u32"
    if profile[:3] != ("tma_tensor_4d",)*3 or profile[-1] not in ("bulk_shared_to_global", "register_store_u32"):
        raise ValueError("no calibrated FA3 phase profile for this transport")
    plan = facts["execution"]
    # The frozen profile already measures these paths, including overlapping
    # service and waits. Adding serialized bandwidth time would bill them twice.
    q_height, k_height, v_height = (moves[r]["box"][1] for r in ("q","k","v"))
    tail = rows % q_height
    if k_height != v_height or (tail and not (register and (causal or moves["k"].get("query_heads_per_kv_head",1)>1))):
        raise ValueError("unsupported TMA tile extent for the calibrated phase profile")
    if q_height not in (64,128) or k_height != 128:
        raise ValueError("unsupported FA3 TMA extent")
    tiles = _TileWork((rows+q_height-1)//q_height, (keys+k_height-1)//k_height, bool(causal),k_height//q_height)
    if q_height==128:
        receipt=rank_fa3_events(plan,rows,keys,causal,budget,heads=heads,tile_work=tiles,cta_dispatch=cta_dispatch)
    else:
        from fa3_query_tile import launch_costs
        if type(budget) is not int or budget not in (1,2) or (plan["buffers"]["k"]["versions"],
                plan["buffers"]["v"]["versions"],plan["wait_pending"]) != (2,2,0):
            raise ValueError("Q64 scheduling requires a one/two-image budget and double K/V rings")
        costs=launch_costs(rows,keys,heads,causal,64,"register" if register else "tma",cta_dispatch=cta_dispatch)
        receipt=dict(budget=budget,selected=sorted(costs,key=costs.get)[:budget],
                     costs={k:dict(total_us=v) for k,v in costs.items()},score_unit="estimated launch us",
                     active_tiles_per_head=tiles.after(0))
    receipt["scored_dispatch"] = cta_dispatch
    ctas = heads*tiles.query_tiles
    iterations = heads*tiles.after(0)
    work = {}
    for role, move in moves.items():
        if role == "o" and register:
            service = move.get("store_work", {})
            # This calibration covers the measured full-tile holder footprint only.
            if (service.get("bytes"), service.get("warp_stores"), service.get("sector_requests"),
                    service.get("width_bytes")) != (q_height*256,q_height*2,q_height*16,4):
                raise ValueError("no calibrated register writeback for this holder footprint")
            partial = move.get("tail_store_work", {})
            if tail and partial.get("bytes") != tail*256:
                raise ValueError("register output tail disagrees with the launch")
            tails = heads if tail else 0
            def total(field):
                return (ctas-tails)*service[field] + tails*partial.get(field,0)
            work[role] = dict(path=move["path"], tiles=ctas, issued_instructions=total("warp_stores"),
                              requested_bytes=total("bytes"), sector_requests=total("sector_requests"))
            continue
        tile = move["buffer"]["tile_bytes"]
        instances = ctas if role in ("q","o") else iterations
        issues = move.get("boxes",1)
        if role != "o" and tile != issues*move["element_bytes"]*move["box"][0]*move["box"][1]:
            raise ValueError("TMA box geometry disagrees with expected transaction bytes")
        work[role] = dict(path=move["path"], tiles=instances, issued_instructions=instances*issues,
                          requested_bytes=instances*tile)
    if register and q_height==128:
        # Measured (register - bulk) per resident wave, in microseconds, on H100 PCIe.
        # Same-tile retires PV within the iteration; lookahead drains a final PV.
        # The phase estimate already includes bulk delivery. Replace its marginal
        # cost once per CTA, irrespective of the number of KV iterations.
        from fa3_event_schedule import SCHEDULES
        delta_us = {"same_tile": -0.709334, "lookahead": -1.789000}
        for name, cost in receipt["costs"].items():
            delta = delta_us[SCHEDULES[name].frame]
            # In a single resident wave a delayed K return can affect a
            # subsequent refill only after the double-buffered ring wraps.
            # Differential of differences against ordinary lookahead, per CTA.
            schedule = SCHEDULES[name]
            if (schedule.frame == "lookahead" and
                    schedule.steady.index("release_k") > schedule.steady.index("pv") and
                    schedule.steady.index("qk") < schedule.steady.index("pv") and
                    ctas <= receipt["sm_count"]):
                refill_share = min(1.0, receipt["possible_k_refills_per_head"] / tiles.query_tiles)
                delta -= 0.104 * refill_share
            cost["writeback_delta_us"] = delta
            cost["total_us"] += delta
        receipt["selected"] = sorted(receipt["costs"], key=lambda a: receipt["costs"][a]["total_us"])[:budget]
    family = "sm90_tma_qkv_register_o" if register else "sm90_tma_qkv_bulk_o"
    if q_height==64: family+="_q64"
    receipt.update(transport_sha256=transport.fingerprint, transport_profile=family,
                   transport_work=work, transport_cost_included_in="calibrated phase and boundary costs")
    return receipt
