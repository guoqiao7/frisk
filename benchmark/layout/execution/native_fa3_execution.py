"""Lower the admitted FA3 graph using Frisk certificates and an imported schedule."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile

from online_execution_contract import packet_contract
from fa3_transport import resolve_verified_transport, rank_transport_events
from fa3_event_schedule import SCHEDULES, resolve_events
from fa3_query_tile import eligible_writebacks, rank_cta_dispatch

# The canonical fixture is online_fixture(128,128,8192,16,16,execution_contract=plan)
# after --frisk-layout-pipeline --frisk-export-execution. This specialization
# verifies the complete SSA, arithmetic, layouts and memory maps before lowering.
# Optimizer receipts are diagnostic only. Execution plans are excluded so changes of placement, ring allocation and
# barrier numbering do not require another graph whitelist entry.
_GRAPH_SHA256 = 'b4b3baf0f8b488cc110e27c231bfc201ce779fd4f0ca3e7f919f06a3f187260c'


_GRAPH_Q64_SHA256 = "cb181ac47c2a0bb5fa103fad029737b9d7c8eb262583b3e6234c1af3c8d89c94"


def _query_tile(plan):
    return plan["buffers"]["q"]["tile_bytes"]//256


def _geometry(data):
    module, = [op for op in data['operations'] if op['name']=='builtin.module']
    launch = module['attrs']['frisk.online_launch']
    if set(launch) - {'kv_rows'} != {'rows','heads','kv_heads','causal'}:
        raise ValueError('unsupported packed FA3 launch')
    rows, heads = launch['rows'], launch['heads']
    kv = launch.get('kv_rows',rows)
    if (type(kv) is not int or kv <= 128 or type(rows) is not int or type(heads) is not int or rows <= 128
            or not (type(launch['kv_heads']) is int and 1 <= launch['kv_heads'] <= heads <= 65535 and heads % launch['kv_heads'] == 0) or max(rows,kv)*heads*128 >= 2**31
            or launch['causal'] not in (-1,0,1)):
        raise ValueError('invalid FA3 launch geometry')
    return rows, heads, bool(launch['causal'])


def _kv_heads(data):
    module, = (op for op in data['operations'] if op['name']=='builtin.module')
    return module['attrs']['frisk.online_launch']['kv_heads']


def _kv_rows(data):
    mod, = [op for op in data['operations'] if op['name']=='builtin.module']
    launch = mod['attrs']['frisk.online_launch']
    return launch.get('kv_rows',launch['rows'])


def _check_graph(data):
    ops=data['operations']
    graph=dict(data,operations=[dict(o,attrs={k:v for k,v in o['attrs'].items()
                      if k not in ('frisk.execution_contract','frisk.protected_graph')}) if o['name']=='builtin.module' else o for o in ops])
    # All SSA and layout information must match the certified graph. Only the
    # launch parameters and the validated, unique loop upper bound may vary.
    rows, heads, causal = _geometry(data)
    loop, = [op for op in ops if op['name']=='scf.for']
    bound, = [op for op in ops if op['name']=='arith.constant' and op['results']==[loop['operands'][1]]]
    if bound['attrs'] != {'value': (_kv_rows(data)+127)//128}:
        raise ValueError('FA3 loop bound disagrees with the launch shape')
    graph['operations'] = [
        dict(op,attrs=dict(op['attrs'],**{'frisk.online_launch':dict(rows=8192,heads=16,kv_heads=16,causal=0)}))
        if op['name']=='builtin.module' else dict(op,attrs={'value':64}) if op is bound else op
        for op in graph['operations']]
    digest=hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    expected = _GRAPH_Q64_SHA256 if _query_tile(packet_contract(data))==64 else _GRAPH_SHA256
    if digest!=expected:
        raise ValueError('FA3 specialization requires the certified packed FP16 graph and layout maps')

    return digest


def _prepare_transport(data, writeback):
    digest=_check_graph(data)
    return resolve_verified_transport(data, packet_contract(data), writeback, digest)


def _source(data,plan,writeback="tma",row_reciprocal=False,flush_exp_subnormals=False, *, event_schedule=None, transport=None):
    if transport is None:
        transport = _prepare_transport(data, writeback)
    facts = transport.facts
    if (facts["execution"] != plan or facts["graph_sha256"] != (_GRAPH_Q64_SHA256 if _query_tile(plan)==64 else _GRAPH_SHA256) or facts["writeback"] != writeback):
        raise ValueError("FA3 transport is incompatible with the certified packet or writer")
    events = resolve_events(plan['kind'], event_schedule)
    rows,heads,causal=_geometry(data)
    moves=facts["moves"]
    head_group=heads//_kv_heads(data)
    if any(moves[r].get("query_heads_per_kv_head",1)!=head_group for r in ("k","v")):
        raise ValueError("FA3 transport disagrees with the exported KV head mapping")
    staged=moves["o"]["path"]=="bulk_shared_to_global"
    if rows % _query_tile(plan) and staged:
        raise ValueError("partial FA3 output tiles require the guarded register writer")
    groups=plan["groups"]
    c=groups[moves["q"]["issuer_group"]]
    p=groups[moves["k"]["issuer_group"]]
    box=moves["k"]["box"]
    const=dict(CTA=plan['cta'],C_BASE=c['base'],C_THREADS=c['threads'],P_BASE=p['base'],
               C_REGS=c['regs'],P_REGS=p['regs'],TILE_BYTES=moves["k"]["buffer"]["tile_bytes"],
               QUERY_ROWS=moves["q"]["box"][1],Q_TILE_BYTES=moves["q"]["buffer"]["tile_bytes"],
               MAP_COLS=box[0],MAP_ROWS=box[1],MAP_SWIZZLE=moves["q"]["swizzle"],
               ROWS=rows,KV_ROWS=_kv_rows(data),TILES=(_kv_rows(data)+127)//128,HEADS=heads,CAUSAL=int(causal),
               BATCH=plan['batch'],WAIT=plan['wait_pending'],ARENA=plan['arena'],SLOTS=plan['barrier_slots'])
    for role in ("q","k","v"):
        m=moves[role]
        if (m["path"], m["box"], m["swizzle"], m["boxes"]) != ("tma_tensor_4d",[64,const["QUERY_ROWS"],1,1] if role=="q" else box,const["MAP_SWIZZLE"],2):
            raise ValueError("unsupported native FA3 TMA instruction geometry")
        const[role.upper()+"_ISSUER"]=m["issuer_local"]
    if staged:
        out=moves["o"]
        const.update(O_ISSUER=out["issuer_local"], O_ALIAS_BARRIER=out["alias_barrier"] if out["alias_barrier"] is not None else -1,
                     O_PUBLISH_BARRIER=out["publish_barrier"], O_WAIT=out["bulk_wait_groups"],
                     O_WAIT_READ=int(out["bulk_wait_reads"]))
    const['SAME_TILE_PV']=int(events.frame=='same_tile')
    arrivals=[0]*plan['barrier_slots']
    for role,move in moves.items():
        b=move["buffer"]
        if b is None:
            continue
        const[role.upper()+"_OFFSET"]=b["offset"]
        const[role.upper()+"_VER"]=b["versions"]
        for edge in ("ready","free"):
            e=move.get(edge)
            if e is not None:
                const[(role+"_"+edge).upper()]=e["first"]
                for i in range(e["slots"]):
                    arrivals[e["first"]+i]=e["arrivals"]
    arrays={'ARRIVALS':arrivals}
    for role,label,name in (("q","QA","QK_A_START"),("k","KB","QK_B_START"),("v","VB","PV_B_START")):
        m=moves[role]
        const['PV_SW' if role=="v" else 'SS_SW']=1  # SM90 SW128 descriptor encoding
        const[label+'_LEAD']=m["leading"]//16
        const[label+'_STRIDE']=m["stride"]//16
        arrays[name]=m["starts"]
    if not staged:
        store, = [op for op in data["operations"] if op["name"]=="frisk.tile_store"]
        value, = [v for v in data["values"] if v["id"]==store["operands"][0]]
        holders = value["distributed"]
        expected = [((tid//32)*16+(tid%32)//4+8*((i//2)%2))*128
                    + 8*(i//4)+2*(tid%4)+i%2 for tid in range(c["threads"]) for i in range(64)]
        if holders["logical"]!=expected or holders["topology"]!=[64,32,4,c["threads"]//128,1]:
            raise ValueError("register writeback disagrees with the output holder map")
    prefix='\n'.join(f'constexpr int {k}={v};' for k,v in const.items())
    lut='\n'.join('__device__ __constant__ int '+k+'[]={'+','.join(map(str,values))+'};' for k,values in arrays.items())
    if head_group>1:
        prefix += f"\n#define FRISK_KV_HEAD_RATIO {head_group}"
    if _query_tile(plan)==64:
        prefix += "\n#define FRISK_NARROW_OUTPUT_DIV 1"
    if flush_exp_subnormals:
        prefix += "\n#define FRISK_EXP_FTZ 1"
    if row_reciprocal:
        prefix += "\n#define FRISK_ROW_RECIPROCAL 1"
    if not staged:
        prefix += "\n#define FRISK_REGISTER_WRITEBACK 1"
    return prefix+'\n'+lut+'\n'+events.cuda_definitions()+'\n#include "native_fa3_sm90.cuh"\n#include "native_fa3_fringe.cuh"\n#include "native_fa3_host.cuh"\n'


class FA3Image:
    def __init__(self, directory, temporary, src, binary, rows, heads, kv_rows, scratch_floats, query_tile=128, kv_heads=None):
        self.kv_heads=heads if kv_heads is None else kv_heads
        self.query_tile=query_tile
        self.rows, self.heads, self.kv_rows = rows, heads, kv_rows
        self.directory=directory; self._temporary=temporary
        self.asm={'cubin':binary,'cuda':src}
        self._lib=ctypes.CDLL(str(directory/'fa3.so'))
        self.scratch_floats=scratch_floats
        self._run=self._lib.frisk_launch_workspace
        self._run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*7+[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self._run.restype=ctypes.c_int

    def __getitem__(self,grid):
        if tuple(grid)!=((self.rows+self.query_tile-1)//self.query_tile,self.heads,1): raise ValueError('unexpected FA3 launch grid')
        def launch(q,k,v,o):
            import torch
            if q.ndim!=4 or q.shape[0]*q.shape[1]!=self.heads or tuple(q.shape[2:])!=(self.rows,128):
                raise ValueError("expected dense BHSD with the compiled number of independent heads")
            pitches = []
            spans = []
            batch=q.shape[0]
            if self.kv_heads % batch:
                raise ValueError("batch does not divide the compiled KV head count")
            for t,n,total_heads in zip((q,k,v,o),(self.rows,self.kv_rows,self.kv_rows,self.rows),
                                       (self.heads,self.kv_heads,self.kv_heads,self.heads)):
                if (not t.is_cuda or t.device!=q.device or t.dtype!=torch.float16 or
                        tuple(t.shape)!=(batch,total_heads//batch,n,128) or t.data_ptr()%16 or
                        t.stride(3)!=1 or t.stride(2)!=128):
                    raise ValueError('expected aligned BHSD FP16 tensors with packed rows')
                stride = t.stride(1) if t.shape[1]>1 else (t.stride(0) if t.shape[0]>1 else n*128)
                if (stride%128 or stride<n*128 or stride*total_heads>=2**31 or
                        (t.shape[0]>1 and t.stride(0)!=t.shape[1]*stride)):
                    raise ValueError('heads require a uniform, nonoverlapping row pitch')
                pitches.append(stride//128)
                spans.append(2*((total_heads-1)*stride+n*128))
            if any(t.data_ptr()<o.data_ptr()+spans[3] and o.data_ptr()<t.data_ptr()+size
                   for t,size in zip((q,k,v),spans)):
                raise ValueError('output aliases an input')
            with torch.cuda.device(q.device):
                scratch = torch.empty(self.scratch_floats,dtype=torch.float32,device=q.device) if self.scratch_floats else None
                result=self._run(*(t.data_ptr() for t in (q,k,v,o)),torch.cuda.current_stream().cuda_stream,
                                 self.rows,self.kv_rows,self.heads,*pitches,None,scratch.data_ptr() if scratch is not None else None,self.scratch_floats)
            if result:raise RuntimeError(f'FA3 launch failed with CUDA status {result}')
        return launch


def compile_fa3(data, *, tilelang_root=None, output_directory=None, fa3_writeback="tma", row_reciprocal=False, flush_exp_subnormals=False, diagnostic_geometry=False, trace_ctas=False, fringe_policy="base", event_schedule=None, fa3_transport=None, cta_dispatch=None):
    plan=packet_contract(data)
    query_tile=_query_tile(plan)
    if _kv_heads(data)!=_geometry(data)[1] and (fringe_policy not in ("base","reverse") or diagnostic_geometry):
        raise ValueError("grouped FA3 requires the packed base kernel")
    if query_tile==64 and (fringe_policy!='base' or diagnostic_geometry):
        raise ValueError('Q64 uses its guarded base kernel')
    events=resolve_events(plan['kind'], event_schedule)
    if fa3_transport is None:
        fa3_transport = _prepare_transport(data, fa3_writeback)
    else:
        # Verify that a reused contract still meets this packet's layout proof.
        _check_graph(data)
    src=_source(data,plan,fa3_writeback,row_reciprocal,flush_exp_subnormals,event_schedule=events,transport=fa3_transport)
    if fringe_policy.startswith("split") and (_kv_rows(data)+63)//64>65535:
        raise ValueError("query fringe exceeds the CUDA split grid limit")
    policies = {"base":0, "warp":1, "body":2, "short":0, "warp-short":1, "split":3, "split-short":3, "reverse":0, "reverse-short":0}
    if fringe_policy not in policies or (fringe_policy=="body" and not diagnostic_geometry):
        raise ValueError("unsupported FA3 query fringe policy")
    if fringe_policy!="base" and fa3_writeback!="register":
        raise ValueError("query fringe requires register output")
    short_policy = fringe_policy.endswith('short')
    legacy_reverse = fringe_policy.startswith("reverse")
    if cta_dispatch is None:
        cta_dispatch = "reverse" if legacy_reverse else "forward"
    if legacy_reverse and cta_dispatch != "reverse":
        raise ValueError("conflicting CTA dispatch and legacy reverse fringe")
    if cta_dispatch == "auto":
        n, h, causal = _geometry(data)
        cta_dispatch = rank_cta_dispatch(n,_kv_rows(data),h,causal,query_tile)["selected"]
    if cta_dispatch not in ("forward","reverse"):
        raise ValueError("invalid CTA dispatch")
    reverse_order = cta_dispatch == "reverse"
    if reverse_order and (not _geometry(data)[2] or policies[fringe_policy]):
        raise ValueError("reverse dispatch requires a causal single base grid")
    src=f"constexpr bool REVERSE_Q={int(reverse_order)};\nconstexpr bool SHORT_KV={int(short_policy)};\nconstexpr int QUERY_FRINGE={policies[fringe_policy]};\n"+src
    if trace_ctas and (not diagnostic_geometry or fringe_policy!="base"):
        raise ValueError('CTA tracing requires diagnostic geometry')
    if diagnostic_geometry:
        if fa3_writeback!='register': raise ValueError('diagnostic grid needs guarded register stores')
        src='#define FRISK_DYNAMIC_GEOMETRY 1\n'+src
    if trace_ctas: src='#define FRISK_TRACE_CTAS 1\n'+src
    temporary=tempfile.TemporaryDirectory(prefix='frisk-fa3-')
    dest=Path(temporary.name)
    root=tilelang_root or os.environ.get('FRISK_TILELANG_ROOT')
    if not root: raise ValueError('FRISK_TILELANG_ROOT must point to the SM90 intrinsic headers')
    headers=Path(root); here=Path(__file__).resolve().parent
    (dest/'fa3.cu').write_text(src)
    command=['nvcc','-std=c++17','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','--shared','-Xcompiler=-fPIC',
             '-I'+str(here),'-I'+str(headers/'src'),'-I'+str(headers/'3rdparty/cutlass/include'),
             'fa3.cu','-lcuda','-o','fa3.so']
    subprocess.run(command,cwd=dest,check=True)
    subprocess.run(['cuobjdump','--extract-elf','all','fa3.so'],cwd=dest,check=True,stdout=subprocess.DEVNULL)
    images=[f for f in dest.glob('fa3.*.cubin') if 'Function main_kernel:' in
            subprocess.check_output(['cuobjdump','--dump-resource-usage',str(f)],text=True)]
    if len(images)!=1: raise RuntimeError('expected one native sm90a device image')
    binary=images[0].read_bytes()
    rows,heads,_=_geometry(data)
    nk=_kv_rows(data)
    scratch=130*(rows%128)*heads*((nk+63)//64) if fringe_policy.startswith("split") else 0
    image=FA3Image(dest,temporary,src,binary,rows,heads,nk,scratch,query_tile,_kv_heads(data))
    if output_directory:
        archive=Path(output_directory)
        archive.mkdir(parents=True,exist_ok=True)
        (archive/'kernel.cubin').write_bytes(binary)
        for name in ('fa3.so','fa3.cu'):
            shutil.copyfile(dest/name,archive/name)
    tail=rows%query_tile
    thin=policies[fringe_policy] in (1,2,3)
    main_blocks=rows//query_tile if thin else (rows+query_tile-1)//query_tile
    launches=[dict(kernel="main_kernel",grid=[main_blocks,heads,1],threads=plan["cta"])]
    if thin and tail and fringe_policy!="body":
        if fringe_policy.startswith("split"):
            launches.extend([dict(kernel="fa3_fringe_partial",grid=[tail,heads,(nk+63)//64],threads=256),
                             dict(kernel="fa3_merge_fringe",grid=[tail,heads,1],threads=128)])
        else:
            launches.append(dict(kernel="fa3_query_fringe",grid=[tail,heads,1],threads=256))
    info=dict(cta_dispatch=cta_dispatch,fringe_policy=fringe_policy,workspace_bytes=4*scratch,kernel_launches=launches,row_reciprocal=row_reciprocal,writeback=fa3_writeback,threads=plan['cta'],consumer_threads=plan['groups'][plan['roles']['qk']]['threads'],
                      dynamic_shared_bytes=plan['arena'],grid=[(rows+query_tile-1)//query_tile,heads,1],query_tile=query_tile,
                      physical_transport=fa3_transport.facts, transport_sha256=fa3_transport.fingerprint,
                      layout_execution_contract=plan, execution_contract=dict(plan,kind=events.contract_kind),
                      event_schedule=dict(frame=events.frame,steady=list(events.steady)))
    from attention_execution_audit import audit_attention
    info["execution_audit"] = audit_attention(data, info, image.asm,
        dict(q_tile=query_tile, kv_tile=128, writeback=fa3_writeback, transport_sha256=fa3_transport.fingerprint))
    return image,info


def solve_overlap_fa3(plan_file, opt, directory, *, multi_anchor=False, rows=8192, heads=16, causal=False, kv_rows=None, query_tile=128, kv_heads=None):
    """Export one certified packet; all native schedules consume this same packet."""
    from native_online_fixture import online_fixture
    from online_execution_contract import from_overlap, fa3_query_plan
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    plan=fa3_query_plan(from_overlap(json.loads(Path(plan_file).read_text())),query_tile)
    source=directory/'input.mlir'; solved=directory/'solved.mlir'; packet=directory/'execution.json'
    source.write_text(online_fixture(query_tile,128,rows,heads,heads if kv_heads is None else kv_heads,causal,kv_rows=kv_rows,execution_contract=plan))
    passes=(['--frisk-multi-anchor-layout-pipeline','--frisk-optimize-layout-conversions']
            if multi_anchor else ['--frisk-layout-pipeline'])
    subprocess.run([str(opt),str(source),*passes,
                    '--frisk-export-execution=output='+str(packet)+' minify=true',
                    '-verify-each','-o',str(solved)],check=True)
    return json.loads(packet.read_text())


def compile_overlap_fa3(plan_file, opt, directory, *, tilelang_root=None, multi_anchor=False, rows=8192, heads=16, causal=False, kv_rows=None, fa3_writeback="tma", query_tile=128, cta_dispatch=None, kv_heads=None):
    from backend import compile_execution
    directory=Path(directory).resolve()
    data=solve_overlap_fa3(plan_file,opt,directory,multi_anchor=multi_anchor,rows=rows,heads=heads,causal=causal,kv_rows=kv_rows,query_tile=query_tile,kv_heads=kv_heads)
    image,info=compile_execution(data,tilelang_root=tilelang_root,
                                 output_directory=directory,fa3_writeback=fa3_writeback,cta_dispatch=cta_dispatch)
    (directory/'metadata.json').write_text(json.dumps(info,indent=2))
    return image,info


def rank_fa3_writebacks(transports, rows, keys, causal, budget, *, heads, cta_dispatch="forward"):
    """Compare the original four event orders under each requested physical delivery."""
    if len(transports) > 1:
        if set(transports) != {"tma","register"}:
            raise ValueError("unexpected FA3 writeback family")
        bulk, direct = (transports[name].facts for name in ("tma","register"))
        if (any(bulk[k] != direct[k] for k in ("execution","anchors","graph_sha256")) or
                any(bulk["moves"][k] != direct["moves"][k] for k in ("q","k","v")) or
                bulk["moves"]["o"]["holder_sha256"] != direct["moves"]["o"]["holder_sha256"] or
                (bulk["writeback"],direct["writeback"]) != ("tma","register")):
            raise ValueError("writeback alternatives must share a solved layout and input transport")
    receipts = {w:rank_transport_events(t,rows,keys,causal,budget,heads=heads,cta_dispatch=cta_dispatch) for w,t in transports.items()}
    if len(receipts) == 1:
        writer, receipt = next(iter(receipts.items()))
        return receipt, [(name,writer,name) for name in receipt["selected"]]
    pairs = sorted(((w,a) for w,r in receipts.items() for a in r["costs"]),
                   key=lambda pair: receipts[pair[0]]["costs"][pair[1]]["total_us"])
    chosen = [(w+"-"+a,w,a) for w,a in pairs[:budget]]
    return dict(budget=budget,selected=[key for key,_,_ in chosen],writers=receipts), chosen


def _writer_transports(data, rows, tile, requested):
    writers = eligible_writebacks(rows, tile, requested)
    if not writers:
        raise ValueError("partial FA3 output tiles require the guarded register writer")
    transports = {writer: _prepare_transport(data, writer) for writer in writers}
    if any(t.facts["moves"]["q"]["box"][1] != tile for t in transports.values()):
        raise ValueError("FA3 Q transport extent disagrees with the solved query tile")
    return transports


def _emit_event_order(data, directory, tilelang_root, writer, transport, order, dispatch,
                      source_packet=None):
    from backend import compile_execution
    image, info = compile_execution(data, output_directory=directory, tilelang_root=tilelang_root,
                                    event_schedule=order, fa3_writeback=writer,
                                    fa3_transport=transport, cta_dispatch=dispatch)
    if source_packet is not None:
        info["source_packet"] = str(source_packet)
    if directory is not None:
        (directory / "metadata.json").write_text(json.dumps(info, indent=2))
    return image, info


def compile_budgeted_fa3(data, *, budget=1, directory=None, tilelang_root=None, fa3_writeback="tma", cta_dispatch="auto"):
    """Apply one shared compilation budget across all requested writers and event orders."""
    rows, heads, causal = _geometry(data)
    tile = _query_tile(packet_contract(data))
    transports = _writer_transports(data, rows, tile, fa3_writeback)
    keys = _kv_rows(data)
    dispatch = rank_cta_dispatch(rows,keys,heads,causal,tile,mode=cta_dispatch)
    receipt, chosen = rank_fa3_writebacks(transports,rows,keys,causal,budget,heads=heads,cta_dispatch=dispatch["selected"])
    receipt["cta_dispatch"] = dispatch
    root = Path(directory) if directory is not None else None
    images = {}
    for key, writer, name in chosen:
        dest = root / key if root is not None and budget > 1 else root
        images[key] = _emit_event_order(data, dest, tilelang_root, writer, transports[writer],
                                        name, dispatch["selected"])
    receipt["layout_proofs"] = sorted({info["execution_audit"]["layout_sha256"] for _,info in images.values()})
    if len(receipt["layout_proofs"]) != 1:
        raise ValueError("schedule candidates disagree on their shared solved layout")
    if root is not None:
        (root/"schedule.json").write_text(json.dumps(receipt,indent=2))
    return images, receipt


def compile_static_fa3(plan_file, opt, directory, *, rows, heads, causal, kv_rows=None,
                       fa3_writeback="auto", query_tile=None, cta_dispatch="auto", budget=1,
                       multi_anchor=True, tilelang_root=None, kv_heads=None):
    """Choose a geometry, solve one layout, and emit only the budgeted schedule(s)."""
    from fa3_query_tile import select_query_geometry
    geometry=select_query_geometry(rows,kv_rows or rows,heads,causal,fa3_writeback,
                                   query_tile=query_tile,cta_dispatch=cta_dispatch,guarded_query=kv_heads not in (None,heads))
    packet=solve_overlap_fa3(plan_file,opt,directory,rows=rows,heads=heads,causal=causal,
                             kv_rows=kv_rows,multi_anchor=multi_anchor,query_tile=geometry["query_tile"],kv_heads=kv_heads)
    layout_plan = packet_contract(packet)
    consumers = layout_plan["groups"][layout_plan["roles"]["qk"]]["threads"]
    if (_query_tile(layout_plan), consumers) != (geometry["query_tile"], 2*geometry["query_tile"]):
        raise ValueError("FA3 layout packet does not match the selected consumer geometry")
    images,receipt=compile_budgeted_fa3(packet,budget=budget,directory=directory,tilelang_root=tilelang_root,
                                       fa3_writeback=fa3_writeback,cta_dispatch=geometry["cta_dispatch"])
    receipt["geometry"]=geometry
    (Path(directory)/"schedule.json").write_text(json.dumps(receipt,indent=2))
    return images,receipt


def paired_screen_decision(primary, challenger, samples):
    """Retain the primary unless balanced measurement blocks favor a switch.

    Structured rows contain cycle, order and us. One block contains both
    launch orders on the same input. Legacy timing maps represent individual
    paired comparisons and have no launch-order metadata.
    """
    if not samples:
        raise ValueError("empty paired FA3 screen")
    keys = {primary, challenger}
    ordered = "us" in samples[0] and "order" in samples[0]
    blocks = {}
    effects = []
    for sample in samples:
        times = sample.get("us") if ordered else sample
        if (not isinstance(times, dict) or set(times) != keys
                or any(not math.isfinite(v) or v <= 0 for v in times.values())):
            raise ValueError("invalid paired FA3 measurements")
        effect = math.log(times[challenger]) - math.log(times[primary])
        if not ordered:
            effects.append(effect)
            continue
        order = tuple(sample.get("order", ()))
        if len(order) != 2 or set(order) != keys or "cycle" not in sample:
            raise ValueError("expected cycle and launch order for each FA3 pair")
        block = blocks.setdefault(sample["cycle"], {})
        if order[0] in block:
            raise ValueError("duplicate FA3 launch order within a cycle")
        block[order[0]] = effect

    incomplete = 0
    for block in blocks.values():
        if len(block) == 2:
            effects.append((block[primary] + block[challenger]) * 0.5)
        else:
            incomplete += 1
    wins = sum(effect < 0 for effect in effects)
    losses = sum(effect > 0 for effect in effects)
    count = wins + losses
    # One-sided sign test over complete blocks. Incomplete cycles never vote.
    tail = 1.0
    if wins > losses:
        tail = sum(math.comb(count, k) for k in range(losses + 1)) / (1 << count)
    midpoint = len(effects) // 2
    consistent = (midpoint > 0 and statistics.median(effects[:midpoint]) < 0
                  and statistics.median(effects[midpoint:]) < 0)
    switch = consistent and tail <= 0.05
    return dict(chosen=challenger if switch else primary, balanced=ordered,
                usable=len(effects), incomplete=incomplete, wins=wins, losses=losses,
                sign_tail=tail, both_halves_favor_challenger=consistent)


def select_budgeted_fa3(images, measure=None):
    """Select the single compiled image, or screen two with a caller's paired timings."""
    if len(images) == 1:
        return next(iter(images))
    if len(images) != 2 or measure is None:
        raise ValueError("two compiled FA3 candidates require paired measurements")
    primary, challenger = images
    return paired_screen_decision(primary, challenger, list(measure(images)))["chosen"]


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan');parser.add_argument('--opt',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--tilelang-root')
    parser.add_argument('--multi-anchor',action='store_true')
    parser.add_argument('--rows',type=int,default=8192)
    parser.add_argument('--kv-rows',type=int,help='key/value sequence length (defaults to query rows)')
    parser.add_argument('--heads',type=int,default=16)
    parser.add_argument('--kv-heads',type=int,help='packed KV heads across all batches (defaults to query heads)')
    parser.add_argument('--causal',action='store_true')
    parser.add_argument('--cta-dispatch',choices=('auto','forward','reverse'),
                        help='physical CTA order (budgeted default: auto, explicit replay: forward)')
    parser.add_argument("--writeback",choices=("tma","register","auto"),
                        help="physical output delivery for static scheduling; auto shares the total budget")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--event-schedules',nargs='+',choices=tuple(SCHEDULES),
                        help='share one solved packet and emit each event schedule in a separate subdirectory')
    mode.add_argument('--schedule-budget',type=int,choices=(1,2),default=1,
                      help='compile the static shortlist (default: one)')
    mode.add_argument('--replay-imported',action='store_true',help='replay the imported order explicitly')
    parser.add_argument("--query-tile",choices=("auto","64","128"),
                        help="choose the query geometry before solving layout (causal and partial-Q static default: auto)")
    args=parser.parse_args()
    replay=bool(args.replay_imported or args.event_schedules)
    cli_tile = 64 if args.query_tile == "64" else 128
    default_auto = not replay and (args.causal or args.rows % cli_tile != 0)
    args.writeback=args.writeback or ("auto" if default_auto else "tma")
    requested=args.query_tile or ("auto" if default_auto else "128")
    tile=None if requested=="auto" else int(requested)
    if not replay:
        compile_static_fa3(args.plan,args.opt,args.output,rows=args.rows,heads=args.heads,causal=args.causal,
                           kv_rows=args.kv_rows,kv_heads=args.kv_heads,fa3_writeback=args.writeback,query_tile=tile,
                           cta_dispatch=args.cta_dispatch or "auto",budget=args.schedule_budget,
                           multi_anchor=args.multi_anchor,tilelang_root=args.tilelang_root)
        raise SystemExit(0)
    from fa3_query_tile import select_query_geometry
    if tile is None:
        geometry=select_query_geometry(args.rows,args.kv_rows or args.rows,args.heads,args.causal,
                                       args.writeback,cta_dispatch=args.cta_dispatch or "forward",guarded_query=args.kv_heads not in (None,args.heads))
        query_tile=geometry["query_tile"]
        args.cta_dispatch=geometry["cta_dispatch"]
    else:
        query_tile=tile
    options=dict(query_tile=query_tile,multi_anchor=args.multi_anchor,rows=args.rows,heads=args.heads,causal=args.causal,kv_rows=args.kv_rows,kv_heads=args.kv_heads)
    if args.replay_imported and args.writeback == "auto":
        parser.error("imported single-image replay requires one output writer")
    if args.replay_imported:
        compile_overlap_fa3(args.plan,args.opt,args.output,tilelang_root=args.tilelang_root,fa3_writeback=args.writeback,cta_dispatch=args.cta_dispatch,**options)
    elif args.event_schedules:
        if len(set(args.event_schedules)) != len(args.event_schedules):
            parser.error('duplicate event schedules')
        directory=Path(args.output).resolve()
        data=solve_overlap_fa3(args.plan,args.opt,directory,**options)
        rows, heads, causal = _geometry(data)
        transports = _writer_transports(data, rows, query_tile, args.writeback)
        direction = rank_cta_dispatch(rows, _kv_rows(data), heads, causal, query_tile,
                                      mode=args.cta_dispatch or "forward")["selected"]
        receipts={}
        for writer, transport in transports.items():
            facts=transport.facts
            receipt=dict(transport_sha256=transport.fingerprint)
            # Explicit replay can also emit legal plans outside the profiled geometry.
            if rows % query_tile == 0 and all(facts["moves"][r]["buffer"]["versions"]==2 for r in ("k","v")):
                receipt=rank_transport_events(transport,rows,_kv_rows(data),causal,1,heads=heads,cta_dispatch=direction)
            receipt["replayed_orders"]=list(args.event_schedules)
            receipts[writer]=receipt
            for name in args.event_schedules:
                dest = directory / (writer + "-" + name if len(transports) > 1 else name)
                _emit_event_order(data, dest, args.tilelang_root, writer, transport, name, direction,
                                  directory / "execution.json")
        report = next(iter(receipts.values())) if len(receipts) == 1 else dict(writers=receipts)
        (directory/"schedule.json").write_text(json.dumps(report,indent=2))
