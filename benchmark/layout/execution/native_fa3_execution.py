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


def _check_graph(data):
    ops=data['operations']
    graph=dict(data,operations=[dict(o,attrs={k:v for k,v in o['attrs'].items()
                      if k not in ('frisk.execution_contract','frisk.protected_graph')}) if o['name']=='builtin.module' else o for o in ops])
    digest=hashlib.sha256(json.dumps(graph,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=_GRAPH_SHA256:
        raise ValueError('FA3 specialization requires the certified packed FP16 graph and layout maps')


def _source(data,plan):
    _check_graph(data)
    groups=plan['groups'];roles=plan['roles'];c=groups[roles['qk']];p=groups[roles['k']]
    mma=[o['attrs'] for o in data['operations'] if o['name']=='frisk.mma']
    ss,rs=(a['frisk.mma_contract'] for a in mma)
    for att,cert,role,form in zip(mma,(ss,rs),('qk','pv'),('ss','rs')):
        if (att['frisk.execution_role'],att['frisk.execution_threads'],cert['form'],cert['grid'],
            cert['repeats'],cert['atom'])!=(role,c['threads'],form,[2,1],[1,1,plan['batch']],[64,128,16]):
            raise ValueError('unsupported FA3 MMA layout')
    const=dict(CTA=plan['cta'],C_BASE=c['base'],C_THREADS=c['threads'],P_BASE=p['base'],
               C_REGS=c['regs'],P_REGS=p['regs'],TILE_BYTES=32768,TILES=64,
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
    prefix='\n'.join(f'constexpr int {k}={v};' for k,v in const.items())
    lut='\n'.join('__device__ __constant__ int '+k+'[]={'+','.join(map(str,values))+'};' for k,values in arrays.items())
    return prefix+'\n'+lut+'\n#include "native_fa3_sm90.cuh"\n#include "native_fa3_host.cuh"\n'


class FA3Image:
    def __init__(self, directory, temporary, src, binary):
        self.directory=directory; self._temporary=temporary
        self.asm={'cubin':binary,'cuda':src}
        self._lib=ctypes.CDLL(str(directory/'fa3.so'))
        self._run=self._lib.frisk_launch
        self._run.argtypes=[ctypes.c_void_p]*5
        self._run.restype=ctypes.c_int

    def __getitem__(self,grid):
        if tuple(grid)!=(64,16,1): raise ValueError('expected FA3 grid (64,16,1)')
        def launch(q,k,v,o):
            import torch
            if any(not t.is_cuda or t.device!=q.device or t.dtype!=torch.float16 or
                   tuple(t.shape)!=(1,16,8192,128) or not t.is_contiguous() or t.data_ptr()%16
                   for t in (q,k,v,o)): raise ValueError('expected aligned contiguous CUDA BHSD half tensors')
            if any(t.data_ptr()<o.data_ptr()+o.nbytes and o.data_ptr()<t.data_ptr()+t.nbytes for t in (q,k,v)):
                raise ValueError('output aliases an input')
            with torch.cuda.device(q.device):
                result=self._run(*(t.data_ptr() for t in (q,k,v,o)),torch.cuda.current_stream().cuda_stream)
            if result:raise RuntimeError(f'FA3 launch failed with CUDA status {result}')
        return launch


def compile_fa3(data, *, tilelang_root=None, output_directory=None):
    plan=packet_contract(data)
    src=_source(data,plan)
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
    image=FA3Image(dest,temporary,src,binary)
    if output_directory:
        archive=Path(output_directory)
        archive.mkdir(parents=True,exist_ok=True)
        (archive/'kernel.cubin').write_bytes(binary)
        for name in ('fa3.so','fa3.cu'):
            shutil.copyfile(dest/name,archive/name)
    return image,dict(threads=plan['cta'],consumer_threads=plan['groups'][plan['roles']['qk']]['threads'],
                      dynamic_shared_bytes=plan['arena'],grid=[64,16,1],execution_contract=plan)


def compile_overlap_fa3(plan_file, opt, directory, *, tilelang_root=None, multi_anchor=False):
    from native_online_fixture import online_fixture
    from online_execution_contract import from_overlap
    from backend import compile_execution
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    plan=from_overlap(json.loads(Path(plan_file).read_text()))
    source=directory/'input.mlir'; solved=directory/'solved.mlir'; packet=directory/'execution.json'
    source.write_text(online_fixture(128,128,8192,16,16,execution_contract=plan))
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
    args=parser.parse_args()
    compile_overlap_fa3(args.plan,args.opt,args.output,tilelang_root=args.tilelang_root,multi_anchor=args.multi_anchor)
