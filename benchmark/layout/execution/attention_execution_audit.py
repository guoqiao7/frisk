"""Cross-check the solved attention domain and selected transport against one native image."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

from online_execution_contract import packet_contract


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError("attention execution contract: " + message)


def machine_ops(binary, symbol):
    with tempfile.TemporaryDirectory(prefix="frisk-fa-audit-") as tmp:
        path = Path(tmp, "kernel.cubin")
        path.write_bytes(binary)
        dis = subprocess.check_output(["cuobjdump", "--dump-sass", str(path)], text=True)
    functions = re.split(r"Function\s*:\s*(\S+)", dis)
    require(symbol in functions[1::2], f"missing entry {symbol}")
    body = functions[functions.index(symbol)+1]
    return Counter(re.findall(r"/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Za-z0-9_.]+)\s", body))


def mma_domain(data, plan):
    ops = data["operations"]
    by_form = {op["attrs"]["frisk.mma_contract"]["form"]:op for op in ops if op["name"]=="frisk.mma"}
    require(len([op for op in ops if op["name"]=="frisk.mma"]) == 2 and set(by_form)=={"ss","rs"}, "QK/PV anchor pair")
    anchors = {}
    for role, form in (("qk","ss"), ("pv","rs")):
        op = by_form[form]
        attr = op["attrs"]
        cert = attr["frisk.mma_contract"]
        gid = plan["roles"][role]
        threads = plan["groups"][gid]["threads"]
        require(attr["frisk.execution_threads"] == threads and
                cert["grid"][0]*cert["grid"][1]*128 == threads, role+" thread domain")
        anchors[role] = dict(group=gid, threads=threads, grid=cert["grid"], form=form,
                             certificate_sha256=digest(cert))
    return by_form, anchors


def check_tma_issue(ptx, desc, barriers):
    """Bind the PTX descriptor parameter, issuer predicate, slot and barrier at each issue."""
    param = f"frisk_execution_param_{desc['descriptor_arg']}"
    match = re.search(r"mov\.b64\s+(%\w+),\s*"+param+r";", ptx)
    if desc["abi"] == "value":
        require(match is not None, "by-value tensor-map parameter")
        reg = re.search(r"cvta\.param\.u64\s+(%\w+),\s*"+match[1]+r";",ptx)
    else:
        reg = re.search(r"ld\.param\.\w+\s+(%\w+),\s*\["+param+r"\];",ptx)
    require(reg is not None, "tensor-map parameter binding")
    shared = {}
    for name, base, offset in re.findall(r"add\.u32 (\w+), (\w+), (\d+);",ptx):
        if base=="smem" or base in shared:
            shared[name]=shared.get(base,0)+int(offset)
    issuers = set(re.findall(r"setp\.eq\.u32 (\w+), tid, 0;",ptx))
    issues = re.findall(r"@(\w+) cp\.async\.bulk\.tensor\.(\d)d[^\s]* \[(\w+)\], \["+
                        reg[1]+r", \{(\d+),[^}]*\}\], \[(\w+)\];",ptx)
    observed = set()
    for pred,rank,dest,col,bar in issues:
        require(pred in issuers and int(rank)==len(desc["box"]) and bar in shared and dest in shared,
                "TMA PTX issuer or address")
        index = barriers.index(shared[bar]) if shared[bar] in barriers else -1
        require(index >= 0 and shared[dest]==desc["slots"][index]+int(col)*desc["box"][1]*2,
                "TMA PTX slot/barrier pairing")
        observed.add(index)
    require(observed==set(range(len(barriers))), "missing PTX issue for a ring slot")


def online_moves(data, plan, info, expected, mma, ptx):
    alloc = info["shared_allocation_plan"]
    records = {v["root_id"]:v for v in alloc["allocations"]}
    rings = {int(k):v for k,v in alloc.get("native_kv_versions", {}).items()}
    values = {v["id"]:v for v in data["values"]}
    copies = {op["operands"][1]:op for op in data["operations"] if op["name"]=="frisk.copy"}
    moves = {}
    occupied = []
    transaction = 0
    for role, use, arg in (("q","ss",0),("k","ss",1),("v","rs",1)):
        src, dst = copies[mma[use]["operands"][arg]]["operands"]
        root = values[dst]["storage"]["root"]
        slot = records[root]
        offsets = rings.get(root, [slot["offset"]])
        for off in offsets:
            require(off % slot["alignment"] == 0, role+" slot alignment")
            require(0 <= off and off+slot["bytes"] <= info["dynamic_shared_bytes"], role+" shared range")
            occupied.append((off, off+slot["bytes"]))
        desc = info.get(role+"_tensor_map")
        path = "tma" if desc else "cp.async"
        require(path == expected["paths"][role], role+" scored copy path")
        require(len(offsets) == (2 if role!="q" and info["loop_optimizations"]["ring"] else 1),
                role+" lowered ring depth")
        if expected.get("kv_versions") and role != "q":
            require(len(offsets)==expected["kv_versions"], role+" scored ring depth")
        move = dict(src=src, dst=dst, path=path, issuer_group=0, issuer_local=0 if desc else "active lanes",
                    slot_offsets=offsets, versions=len(offsets), tile_bytes=slot["bytes"],
                    declared_versions=plan["buffers"][role]["versions"],
                    release="after final QK" if role=="q" else
                            info.get("pv_completion",{}).get("release","after consumer drain and CTA rendezvous"))
        if desc:
            kmap = info["k_tensor_map"]
            phase = kmap["phase"]
            require(offsets == desc["slots"] and desc["transaction_bytes"] == slot["bytes"], role+" tensor-map slots/bytes")
            barrier = kmap["barriers"]
            require(len(barrier)==len(offsets) and kmap["issuer_thread"]==0 and
                    phase=="(kv_iteration >> 1) & 1", role+" TMA issuer or phase")
            require(len(set(barrier))==len(barrier) and all(off%8==0 and
                alloc["tma_barrier_offset"] <= off <= info["dynamic_shared_bytes"]-8 for off in barrier),
                role+" transaction barrier allocation")
            check_tma_issue(ptx, desc, barrier)
            rank = len(desc["box"])
            require(f"cp.async.bulk.tensor.{rank}d" in ptx, role+" PTX tensor-map rank")
            swizzle = mma[use]["attrs"]["frisk.mma_contract"]["b_descriptor"]["swizzle"]
            require(desc["swizzle_bytes"] == swizzle, role+" descriptor swizzle")
            if role=="v" and "v_swizzle" in expected:
                require(swizzle==expected["v_swizzle"],"V scored swizzle")
            move.update(ready="mbarrier_expect_tx_parity", barrier_offsets=barrier, phase=phase,
                        swizzle_bytes=swizzle, map_rank=rank, map_abi=desc["abi"])
            transaction += desc["transaction_bytes"]
            if desc.get("tail_rows"):
                require("ld.global.v4.b32" in ptx and "st.shared.v4.b32" in ptx, "V tail repair")
                move["tail"] = dict(rows=desc["tail_rows"], path="guarded vector load + shared store")
        else:
            require(values[src]["storage"]["root"] in info["packed_copy_roots"] or
                    (root in rings and info["loop_optimizations"]["ring"]), role+" packed copy was not lowered")
            pending = 1 if role=="k" and info["defer_v_wait"] else 0
            require(f"cp.async.wait_group {pending};" in ptx, role+" PTX copy wait depth")
            move["ready"] = dict(protocol="cp.async.wait_group + CTA publication", pending=pending)
        moves[role] = move
    spans = sorted(occupied)
    require(all(left[1]<=right[0] for left,right in zip(spans,spans[1:])), "overlapping Q/K/V slots")
    if transaction:
        offsets = info["k_tensor_map"]["barriers"]
        require(all(hi<=b or lo>=b+8 for b in offsets for lo,hi in spans),"barrier aliases operand slot")
        require(re.search(r"mbarrier\.arrive\.expect_tx[^\n]*,\s*"+str(transaction)+r";", ptx),
                "PTX transaction byte count")
    if moves["v"]["path"]=="tma":
        joint=info["kv_completion"]
        require(joint["expected_bytes"]==transaction and joint["barriers"]==moves["k"]["barrier_offsets"] and
                joint["issuer_thread"]==0 and joint["phase"]==moves["k"]["phase"] and joint["slots"]==2,
                "joint K/V completion")
    store, = [op for op in data["operations"] if op["name"]=="frisk.tile_store"]
    src, dst = store["operands"]
    terminal = info["terminal_delivery"]
    require(terminal["requested"] == expected["writer"] and
            terminal["consumes_conversion"] == any(op["name"]=="frisk.convert_layout" and
                    src in op["results"] for op in data["operations"]), "scored output owner")
    writes = sorted(set(re.findall(r"\bst\.global\.([\w.]+)",ptx)))
    require(bool(writes), "missing register writeback")
    moves["o"] = dict(path="register_store", src=src, dst=dst, issuer_group=plan["roles"]["pv"],
                      issuer_local="writer lanes", versions=0, slot_offsets=[], ready="synchronous",
                      release="kernel exit", writer=terminal, holder_sha256=digest(values[src]["distributed"]), ptx_store_types=writes)
    return moves


def fa3_moves(plan, info, expected, cuda):
    transport = info["physical_transport"]
    require(info["transport_sha256"]==digest(transport) and transport["execution"]==plan and
            expected["transport_sha256"]==info["transport_sha256"] and expected["writeback"]==transport["writeback"], "FA3 scored transport")
    const = {k:int(v) for k,v in re.findall(r"constexpr int (\w+)=(-?\d+);",cuda)}
    from fa3_event_schedule import FA3EventSchedule
    schedule = info["event_schedule"]
    events = FA3EventSchedule(schedule["frame"],tuple(schedule["steady"]))
    require(events.cuda_definitions() in cuda, "FA3 event or K release order")
    reverse = re.search(r"constexpr bool REVERSE_Q=([01]);",cuda)
    require(reverse is not None and int(reverse[1])==int(info["cta_dispatch"]=="reverse"), "CTA dispatch")
    bulk = expected["writeback"]=="tma"
    require(("#define FRISK_REGISTER_WRITEBACK 1" in cuda) != bulk, "FA3 emitted writer")
    if bulk:
        out=transport["moves"]["o"]
        require((const["O_WAIT"],const["O_WAIT_READ"],const["O_PUBLISH_BARRIER"]) ==
                (out["bulk_wait_groups"],int(out["bulk_wait_reads"]),out["publish_barrier"]), "bulk completion")
    consumer=plan["groups"][plan["roles"]["qk"]]
    producer=plan["groups"][plan["roles"]["k"]]
    for k,v in dict(CTA=plan["cta"], C_BASE=consumer["base"], C_THREADS=consumer["threads"],
                    P_BASE=producer["base"], BATCH=plan["batch"], WAIT=plan["wait_pending"]).items():
        require(const[k]==v, k+" generated constant")
    moves = {}
    for role, physical in transport["moves"].items():
        r=role.upper()
        buf=physical["buffer"]
        gid=plan["roles"]["pv" if role=="o" else role]
        require(physical["issuer_group"]==gid, r+" issuer group")
        if physical["issuer_local"]!="all":
            require(const[r+"_ISSUER"]==physical["issuer_local"] and
                    physical["issuer_cta"]==plan["groups"][gid]["base"]+physical["issuer_local"], r+" issuer")
        offsets=[]
        if buf:
            require((buf["offset"],buf["versions"])==(const[r+"_OFFSET"],const[r+"_VER"]), r+" ring")
            offsets=[buf["offset"]+i*buf["tile_bytes"] for i in range(buf["versions"])]
        for edge in ("ready","free"):
            if physical.get(edge):
                require(physical[edge]==plan["edges"][role+"_"+edge] and
                        const[r+"_"+edge.upper()]==physical[edge]["first"], r+" "+edge)
        moves[role]=dict(physical, versions=len(offsets), slot_offsets=offsets,
                         ready=physical.get("ready") or physical["completion"],
                         release=physical.get("free") or ("after final QK" if role=="q" else physical["completion"]))
    arrivals = [0]*plan["barrier_slots"]
    for edge in plan["edges"].values():
        for idx in range(edge["first"],edge["first"]+edge["slots"]):
            arrivals[idx]=edge["arrivals"]
    lut = re.search(r"ARRIVALS\[\]=\{([0-9,]+)\}",cuda)
    require(lut is not None and list(map(int,lut[1].split(",")))==arrivals, "barrier participant counts")
    return moves


def audit_attention(data, info, asm, expected):
    plan=packet_contract(data)
    mma,anchors=mma_domain(data,plan)
    q=mma["ss"]["attrs"]
    require((q["m"],q["n"],plan["cta"])==(expected["q_tile"],expected["kv_tile"],info["threads"]),
            "scored geometry or CTA size")
    groups=[dict(base=g["base"],threads=g["threads"],local_tid="threadIdx.x - base") for g in plan["groups"]]
    fa3=plan["kind"]!="single"
    if fa3:
        require(info["physical_transport"]["anchors"]==anchors, "FA3 scored MMA certificates")
        moves=fa3_moves(plan,info,expected,asm["cuda"])
    else:
        require(plan==info["execution_contract"] and expected["batch"]==plan["batch"]==info["mma_group_size"],
                "single-group WGMMA wait batch")
        ptx=asm["ptx"]
        require("wgmma.wait_group.sync.aligned 0;" in ptx and "bar.sync 0;" in ptx and
                int(re.search(r"\.reqntid\s+(\d+)",ptx)[1])==plan["cta"],"PTX CTA or WGMMA drain")
        if expected.get("post_qk"):
            require(info["online_kv_schedule"]=="post-qk-prefetch","scored prefetch edge")
        moves=online_moves(data,plan,info,expected,mma,ptx)
    for role,move in moves.items():
        path=move["path"]
        move["instruction_family"]=("TMA" if path in ("tma","tma_tensor_4d") else
                                    "bulk" if path=="bulk_shared_to_global" else
                                    "STG" if role=="o" else "cp.async")
        base=groups[move["issuer_group"]]["base"]
        local=move["issuer_local"]
        move["issuer_cta"]=base+local if isinstance(local,int) else None
        move["slot_rule"]="tile % versions" if move["versions"]>1 else "single" if move["versions"] else "registers"
    symbol="main_kernel" if fa3 else "frisk_execution"
    ops=machine_ops(asm["cubin"],symbol)
    prefixes=["HGMMA.","WARPGROUP.DEPBAR"]
    if any("tma" in m["path"] for r,m in moves.items() if r!="o"):
        prefixes += ["UTMALDG.","SYNCS.PHASECHK"]
    if any(m["path"]=="cp.async" for m in moves.values()):
        prefixes += ["LDGSTS.", "DEPBAR."]
    prefixes.append("UBLKCP.G.S" if moves["o"]["path"]=="bulk_shared_to_global" else "STG.")
    counts={p:sum(v for k,v in ops.items() if k.startswith(p)) for p in prefixes}
    require(moves["o"]["path"]=="bulk_shared_to_global" or not any(k.startswith("UBLKCP.G.S") for k in ops), "unexpected bulk writer")
    require(all(counts.values()), "missing main-entry SASS family " + str([p for p,n in counts.items() if not n]))
    # Key the physical layout proof, independently of event order or output path.
    owners=[v for v in data["values"] if "distributed" in v or "storage" in v]
    layout_key=digest([groups,plan["roles"],q["m"],q["n"],anchors,owners])
    return dict(version=1, scope="main kernel", q_tile=q["m"], kv_tile=q["n"], groups=groups,
                roles=plan["roles"],anchors=anchors,moves=moves,wait_batch=plan["batch"],
                wait_pending=plan["wait_pending"],
                schedule=info.get("event_schedule",info.get("online_kv_schedule")),layout_sha256=layout_key, expected=expected,
                cubin_sha256=hashlib.sha256(asm["cubin"]).hexdigest(),symbol=symbol,sass_families=counts)
