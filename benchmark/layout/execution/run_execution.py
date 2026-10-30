"""Replay original Frisk input through checked export and TTGIR on H100."""
import argparse,json,subprocess,time,hashlib,sys,tempfile,os
os.environ["TRITON_ALWAYS_COMPILE"]="1"
from pathlib import Path
import numpy as np
import torch,triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import get_ptxas
from triton._C import libtriton
from backend import emit
from async_host import validate_launch,TensorMaps,physical_abi
p=argparse.ArgumentParser();p.add_argument('--path',choices=['vector','cp.async','tma'],default='vector');p.add_argument('--root',default='/home/baopeihua/frisk');p.add_argument('--opt');p.add_argument('--output',required=True);p.add_argument('--compile-only',action='store_true');args=p.parse_args()
r=Path(args.root);output_root=Path(args.output);output_root.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='run-',dir=output_root));print('artifact_directory='+str(out),flush=True);opt=args.opt or str(r/'build-m6-release/bin/frisk-opt')
def execute(cmd,name):
 t=time.perf_counter_ns();q=subprocess.run(cmd,text=True,capture_output=True);(out/(name+'.stdout')).write_text(q.stdout);(out/(name+'.stderr')).write_text(q.stderr);(out/(name+'.command.json')).write_text(json.dumps(cmd));
 if q.returncode:raise RuntimeError(name+' failed')
 return time.perf_counter_ns()-t
source=r/('test/Transforms/attention-layout-'+{'vector':'fixed','cp.async':'cp-async','tma':'tma'}[args.path]+'-abi.mlir');(out/'input.mlir').write_bytes(source.read_bytes());start=time.perf_counter_ns();infer=execute([opt,str(source),'--frisk-layout-pipeline','-o',str(out/'selected.mlir')],'infer');execute([opt,str(out/'selected.mlir'),'-frisk-export-execution=output='+str(out/'execution.json'),'-o','/dev/null'],'export')
data=json.loads((out/'execution.json').read_text());text,info=emit(data);(out/'kernel.ttgir').write_text(text);(out/'execution_config.json').write_text(json.dumps(info,indent=2))
torch.cuda.set_device(1)
k=triton.compile(str(out/'kernel.ttgir'),target=GPUTarget('cuda',90,32),options={'num_warps':4,'enable_fp_fusion':False})
for ext,content in k.asm.items():
 f=out/('kernel.'+ext);f.write_bytes(content) if isinstance(content,bytes) else f.write_text(content)
physical_params=physical_abi(k.asm['ptx'],len(info['tma_plans']))
full=time.perf_counter_ns()-start
manifest={'artifact_directory':str(out),'launch_abi':{'logical_arguments':['Q:u8*','K:u8*','V:u8*','O:u8*'],'physical_arguments':physical_params,'grid':[1,1,1],'threads':128,'device':1,'global_alignments':info['global_alignments']},'schema_version':1,'status':'compiled','triton':triton.__version__,'inference_ns':infer,'full_compile_ns':full,'cache_policy':'TRITON_ALWAYS_COMPILE=1; backend recompiled every invocation','config':info,'metadata':str(k.metadata),'hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='manifest.json' and not f.name.startswith('failure_')}}
manifest['source_hashes']={str(f.relative_to(r)):hashlib.sha256(f.read_bytes()).hexdigest() for f in list((r/'benchmark/layout/execution').glob('*.py'))+[r/'lib/Conversion/FriskLayoutToGPU/ExportExecution.cpp',r/'include/Dialect/Frisk/Transforms/Passes.h',r/'include/Dialect/Frisk/Transforms/Passes.td']};manifest['tool_hashes']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(opt),Path(get_ptxas().path),Path(triton.__file__),Path(libtriton.__file__)]};manifest['environment']=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,name,driver_version,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip();(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('compiled',flush=True)
if args.compile_only:sys.exit(0)
# Independent CPU double-loop matrix products accumulated in FP32; no torch/TF32 reference.
def reference(q,k,v):
 q,k,v=[x.astype(np.float32) for x in (q,k,v)];s=np.zeros((64,64),dtype=np.float32)
 for t in range(16):s=np.float32(s+np.float32(q[:,t,None]*k[None,:,t]))
 s=np.float32(s*np.float32(.25));e=np.exp(np.float32(s-s.max(axis=1,keepdims=True)),dtype=np.float32);prob=np.float32(e/e.sum(axis=1,keepdims=True,dtype=np.float32)).astype(np.float16).astype(np.float32);o=np.zeros((64,16),dtype=np.float32)
 for t in range(64):o=np.float32(o+np.float32(prob[:,t,None]*v[None,t,:]))
 return o
cases=[]
for seed in range(10):
 rng=np.random.default_rng(seed);cases.append(('seed'+str(seed),*[rng.uniform(-1,1,(64,16)).astype(np.float16) for _ in range(3)]))
z=np.zeros((64,16),dtype=np.float16);one=np.ones_like(z);coord=np.linspace(-1,1,1024).reshape(64,16).astype(np.float16);hot=z.copy();hot[13,7]=1;cancel=np.tile(np.array([1,-1],np.float16)[:,None],(32,16));cases += [('zeroQK',z,z,coord),('constant',one,one,one),('signedcoords',coord,-coord,coord),('onehotV',coord,coord,hot),('cancelV',z,z,cancel)]
records=[];host_samples=[]
for name,q,kk,v in cases:
 ref=reference(q,kk,v)
 for offset in ((1,2,3) if args.path=='vector' else (0,128,256)):
  buffers=[]
  for x in (q,kk,v):
   buf=torch.full((x.nbytes+272,),165,dtype=torch.uint8,device='cuda');buf[offset:offset+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(buf)
  ob=torch.full((4096+272,),165,dtype=torch.uint8,device='cuda')
  pointers=[b.data_ptr()+offset for b in buffers]+[ob.data_ptr()+offset];validate_launch(info,pointers)
  maps=TensorMaps(info,pointers) if info['tma_plans'] else None
  descriptors=maps.device if maps else []
  host_samples.append({'case':name,'offset':offset,'descriptor_setup_ns':maps.setup_ns if maps else []});(out/'host_setup_ns.json').write_text(json.dumps(host_samples,indent=2))
  expected_inputs=[b.cpu().numpy().copy() for b in buffers]
  for repeat in range(3):
   ob.fill_(165)
   k[(1,1,1)](*[b[offset:] for b in buffers],ob[offset:],*descriptors)
   torch.cuda.synchronize();raw=ob.cpu().numpy();got=raw[offset:offset+4096].copy().view(np.float32).reshape(64,16);err=np.abs(got-ref);ok=np.isfinite(got).all() and np.all(err<=.002+.002*np.abs(ref));guards=np.all(raw[:offset]==165) and np.all(raw[offset+4096:]==165);input_unchanged=all(np.array_equal(b.cpu().numpy(),expected) for b,expected in zip(buffers,expected_inputs))
   rec={'case':name,'offset':offset,'repeat':repeat,'max_abs_error':float(err.max()),'coordinate':list(map(int,np.unravel_index(err.argmax(),err.shape))),'finite':bool(np.isfinite(got).all()),'numerics':bool(ok),'guards':bool(guards),'input_bytes_and_guards_unchanged':bool(input_unchanged)};records.append(rec);(out/'numerical_results.json').write_text(json.dumps(records,indent=2));print(rec,flush=True)
   if not(ok and guards and input_unchanged):np.save(out/'failure_got.npy',got);np.save(out/'failure_ref.npy',ref);raise AssertionError(rec)
import ctypes
cuda=ctypes.CDLL('libcuda.so.1');resources={}
for key,attribute in [('max_threads',0),('static_shared_bytes',1),('local_bytes',3),('registers',4)]:
 value=ctypes.c_int();status=cuda.cuFuncGetAttribute(ctypes.byref(value),attribute,ctypes.c_void_p(k.function));assert status==0;resources[key]=value.value
samples=[]
for _ in range(20):
 start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);start_event.record();k[(1,1,1)](*[b[offset:] for b in buffers],ob[offset:],*descriptors);end_event.record();end_event.synchronize();samples.append(start_event.elapsed_time(end_event)*1000)
(out/'gpu_timing_us.json').write_text(json.dumps(samples));(out/'physical_resources.json').write_text(json.dumps(resources,indent=2))
manifest.update(status='numerics_passed',registers=k.n_regs,spills=k.n_spills,physical_resources=resources,gpu_samples_us=samples);manifest['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file() and f.name!='manifest.json'};(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
