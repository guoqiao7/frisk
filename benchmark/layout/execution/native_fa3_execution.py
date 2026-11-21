"""Lower the admitted FA3 graph using Frisk certificates and an imported schedule."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from online_execution_contract import packet_contract

# The canonical fixture is online_fixture(128,128,8192,16,16,execution_contract=plan)
# after --frisk-layout-pipeline --frisk-export-execution. This specialization
# verifies the complete SSA, arithmetic, layouts and memory maps before lowering.
# Optimizer receipts are diagnostic only. Execution plans are excluded so changes of placement, ring allocation and
# barrier numbering do not require another graph whitelist entry.
_GRAPH_SHA256 = 'b4b3baf0f8b488cc110e27c231bfc201ce779fd4f0ca3e7f919f06a3f187260c'


def _geometry(data):
    module, = [op for op in data['operations'] if op['name']=='builtin.module']
    launch = module['attrs']['frisk.online_launch']
    if set(launch) - {'kv_rows'} != {'rows','heads','kv_heads','causal'}:
        raise ValueError('unsupported packed FA3 launch')
    rows, heads = launch['rows'], launch['heads']
    kv = launch.get('kv_rows',rows)
    if (type(kv) is not int or kv <= 128 or type(rows) is not int or type(heads) is not int or rows <= 128
            or not 1 <= heads == launch['kv_heads'] <= 65535 or max(rows,kv)*heads*128 >= 2**31
            or launch['causal'] not in (-1,0,1)):
        raise ValueError('invalid FA3 launch geometry')
    return rows, heads, bool(launch['causal'])


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
    if digest!=_GRAPH_SHA256:
        raise ValueError('FA3 specialization requires the certified packed FP16 graph and layout maps')


def _source(data,plan,writeback="tma",row_reciprocal=False,flush_exp_subnormals=False):
    _check_graph(data)
    rows,heads,causal=_geometry(data)
    if rows % 128 and writeback != "register":
        raise ValueError("partial FA3 output tiles require the guarded register writer")
    groups=plan['groups'];roles=plan['roles'];c=groups[roles['qk']];p=groups[roles['k']]
    mma=[o['attrs'] for o in data['operations'] if o['name']=='frisk.mma']
    ss,rs=(a['frisk.mma_contract'] for a in mma)
    for att,cert,role,form in zip(mma,(ss,rs),('qk','pv'),('ss','rs')):
        if (att['frisk.execution_role'],att['frisk.execution_threads'],cert['form'],cert['grid'],
            cert['repeats'],cert['atom'])!=(role,c['threads'],form,[2,1],[1,1,plan['batch']],[64,128,16]):
            raise ValueError('unsupported FA3 MMA layout')
    const=dict(CTA=plan['cta'],C_BASE=c['base'],C_THREADS=c['threads'],P_BASE=p['base'],
               C_REGS=c['regs'],P_REGS=p['regs'],TILE_BYTES=32768,ROWS=rows,KV_ROWS=_kv_rows(data),TILES=(_kv_rows(data)+127)//128,HEADS=heads,CAUSAL=int(causal),
               BATCH=plan['batch'],WAIT=plan['wait_pending'],ARENA=plan['arena'],SLOTS=plan['barrier_slots'])
    const['SAME_TILE_PV']=int(plan['kind']=='fa3_same_tile')
    for k,b in plan['buffers'].items():
        const[k.upper()+'_OFFSET']=b['offset'];const[k.upper()+'_VER']=b['versions']
    arrivals=[0]*plan['barrier_slots']
    for name,e in plan['edges'].items():
        const[name.upper()]=e['first']
        for i in range(e['slots']):arrivals[e['first']+i]=e['arrivals']
    arrays={'ARRIVALS':arrivals}
    for label,desc,direction,mnrows in (
            ('QA',ss['a_descriptor'],'k',2),('KB',ss['b_descriptor'],'k',1),('VB',rs['b_descriptor'],'mn',1)):
        if (desc['major'],desc['swizzle'])!=(direction,128): raise ValueError('unsupported TMA descriptor')
        const['PV_SW' if label=='VB' else 'SS_SW']=1  # GMMA layout_type for SW128
        const[label+'_LEAD']=desc['leading']//16;const[label+'_STRIDE']=desc['stride']//16
        it=iter(desc['entries']); starts={(m,k):start for m,k,start,phase in zip(it,it,it,it) if phase==0}
        name={'QA':'QK_A_START','KB':'QK_B_START','VB':'PV_B_START'}[label]
        arrays[name]=[starts[row*64,i*16] for row in range(mnrows) for i in range(plan['batch'])]
    if writeback not in ("tma","register"):
        raise ValueError("unsupported FA3 writeback")
    if writeback=="register":
        store, = [op for op in data["operations"] if op["name"]=="frisk.tile_store"]
        value, = [v for v in data["values"] if v["id"]==store["operands"][0]]
        holders = value["distributed"]
        expected = [((tid//32)*16+(tid%32)//4+8*((i//2)%2))*128
                    + 8*(i//4)+2*(tid%4)+i%2 for tid in range(c["threads"]) for i in range(64)]
        if holders["logical"]!=expected or holders["topology"]!=[64,32,4,2,1]:
            raise ValueError("register writeback disagrees with the output holder map")
    prefix='\n'.join(f'constexpr int {k}={v};' for k,v in const.items())
    lut='\n'.join('__device__ __constant__ int '+k+'[]={'+','.join(map(str,values))+'};' for k,values in arrays.items())
    if flush_exp_subnormals:
        prefix += "\n#define FRISK_EXP_FTZ 1"
    if row_reciprocal:
        prefix += "\n#define FRISK_ROW_RECIPROCAL 1"
    if writeback=="register":
        prefix += "\n#define FRISK_REGISTER_WRITEBACK 1"
    return prefix+'\n'+lut+'\n#include "native_fa3_sm90.cuh"\n#include "native_fa3_fringe.cuh"\n#include "native_fa3_host.cuh"\n'


class FA3Image:
    def __init__(self, directory, temporary, src, binary, rows, heads, kv_rows, scratch_floats):
        self.rows, self.heads, self.kv_rows = rows, heads, kv_rows
        self.directory=directory; self._temporary=temporary
        self.asm={'cubin':binary,'cuda':src}
        self._lib=ctypes.CDLL(str(directory/'fa3.so'))
        self.scratch_floats=scratch_floats
        self._run=self._lib.frisk_launch_workspace
        self._run.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*7+[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
        self._run.restype=ctypes.c_int

    def __getitem__(self,grid):
        if tuple(grid)!=((self.rows+127)//128,self.heads,1): raise ValueError('unexpected FA3 launch grid')
        def launch(q,k,v,o):
            import torch
            if q.ndim!=4 or q.shape[0]*q.shape[1]!=self.heads or tuple(q.shape[2:])!=(self.rows,128):
                raise ValueError("expected dense BHSD with the compiled number of independent heads")
            pitches = []
            spans = []
            for t,n in zip((q,k,v,o),(self.rows,self.kv_rows,self.kv_rows,self.rows)):
                if (not t.is_cuda or t.device!=q.device or t.dtype!=torch.float16 or
                        tuple(t.shape)!=(*q.shape[:2],n,128) or t.data_ptr()%16 or
                        t.stride(3)!=1 or t.stride(2)!=128):
                    raise ValueError('expected aligned BHSD FP16 tensors with packed rows')
                stride = t.stride(1) if t.shape[1]>1 else (t.stride(0) if t.shape[0]>1 else n*128)
                if (stride%128 or stride<n*128 or stride*self.heads>=2**31 or
                        (t.shape[0]>1 and t.stride(0)!=t.shape[1]*stride)):
                    raise ValueError('heads require a uniform, nonoverlapping row pitch')
                pitches.append(stride//128)
                spans.append(2*((self.heads-1)*stride+n*128))
            if any(t.data_ptr()<o.data_ptr()+spans[3] and o.data_ptr()<t.data_ptr()+size
                   for t,size in zip((q,k,v),spans)):
                raise ValueError('output aliases an input')
            with torch.cuda.device(q.device):
                scratch = torch.empty(self.scratch_floats,dtype=torch.float32,device=q.device) if self.scratch_floats else None
                result=self._run(*(t.data_ptr() for t in (q,k,v,o)),torch.cuda.current_stream().cuda_stream,
                                 self.rows,self.kv_rows,self.heads,*pitches,None,scratch.data_ptr() if scratch is not None else None,self.scratch_floats)
            if result:raise RuntimeError(f'FA3 launch failed with CUDA status {result}')
        return launch


def compile_fa3(data, *, tilelang_root=None, output_directory=None, fa3_writeback="tma", row_reciprocal=False, flush_exp_subnormals=False, diagnostic_geometry=False, trace_ctas=False, fringe_policy="base"):
    plan=packet_contract(data)
    src=_source(data,plan,fa3_writeback,row_reciprocal,flush_exp_subnormals)
    if fringe_policy.startswith("split") and (_kv_rows(data)+63)//64>65535:
        raise ValueError("query fringe exceeds the CUDA split grid limit")
    policies = {"base":0, "warp":1, "body":2, "short":0, "warp-short":1, "split":3, "split-short":3, "reverse":0, "reverse-short":0}
    if fringe_policy not in policies or (fringe_policy=="body" and not diagnostic_geometry):
        raise ValueError("unsupported FA3 query fringe policy")
    if fringe_policy!="base" and fa3_writeback!="register":
        raise ValueError("query fringe requires register output")
    if fringe_policy.startswith("reverse") and not _geometry(data)[2]:
        raise ValueError("descending work order requires causal attention")
    short_policy = fringe_policy.endswith('short')
    reverse_order = fringe_policy.startswith("reverse")
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
    image=FA3Image(dest,temporary,src,binary,rows,heads,nk,scratch)
    if output_directory:
        archive=Path(output_directory)
        archive.mkdir(parents=True,exist_ok=True)
        (archive/'kernel.cubin').write_bytes(binary)
        for name in ('fa3.so','fa3.cu'):
            shutil.copyfile(dest/name,archive/name)
    tail=rows%128
    thin=policies[fringe_policy] in (1,2,3)
    main_blocks=rows//128 if thin else (rows+127)//128
    launches=[dict(kernel="main_kernel",grid=[main_blocks,heads,1],threads=plan["cta"])]
    if thin and tail and fringe_policy!="body":
        if fringe_policy.startswith("split"):
            launches.extend([dict(kernel="fa3_fringe_partial",grid=[tail,heads,(nk+63)//64],threads=256),
                             dict(kernel="fa3_merge_fringe",grid=[tail,heads,1],threads=128)])
        else:
            launches.append(dict(kernel="fa3_query_fringe",grid=[tail,heads,1],threads=256))
    return image,dict(fringe_policy=fringe_policy,workspace_bytes=4*scratch,kernel_launches=launches,row_reciprocal=row_reciprocal,writeback=fa3_writeback,threads=plan['cta'],consumer_threads=plan['groups'][plan['roles']['qk']]['threads'],
                      dynamic_shared_bytes=plan['arena'],grid=[(rows+127)//128,heads,1],execution_contract=plan)


def compile_overlap_fa3(plan_file, opt, directory, *, tilelang_root=None, multi_anchor=False, rows=8192, heads=16, causal=False):
    from native_online_fixture import online_fixture
    from online_execution_contract import from_overlap
    from backend import compile_execution
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    plan=from_overlap(json.loads(Path(plan_file).read_text()))
    source=directory/'input.mlir'; solved=directory/'solved.mlir'; packet=directory/'execution.json'
    source.write_text(online_fixture(128,128,rows,heads,heads,causal,execution_contract=plan))
    passes=(['--frisk-multi-anchor-layout-pipeline','--frisk-optimize-layout-conversions']
            if multi_anchor else ['--frisk-layout-pipeline'])
    subprocess.run([str(opt),str(source),*passes,
                    '--frisk-export-execution=output='+str(packet)+' minify=true',
                    '-verify-each','-o',str(solved)],check=True)
    image,info=compile_execution(json.loads(packet.read_text()),tilelang_root=tilelang_root,
                                 output_directory=directory)
    (directory/'metadata.json').write_text(json.dumps(info,indent=2))
    return image,info


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan');parser.add_argument('--opt',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--tilelang-root')
    parser.add_argument('--multi-anchor',action='store_true')
    parser.add_argument('--rows',type=int,default=8192)
    parser.add_argument('--heads',type=int,default=16)
    parser.add_argument('--causal',action='store_true')
    args=parser.parse_args()
    compile_overlap_fa3(args.plan,args.opt,args.output,tilelang_root=args.tilelang_root,multi_anchor=args.multi_anchor,rows=args.rows,heads=args.heads,causal=args.causal)
