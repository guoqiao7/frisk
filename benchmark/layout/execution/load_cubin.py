import ctypes,json,sys,hashlib,faulthandler
faulthandler.enable()
from pathlib import Path
import torch,numpy as np
from async_host import TensorMaps,physical_abi,validate_launch
directory=Path(sys.argv[1]);info=json.loads((directory/'execution_config.json').read_text());physical=physical_abi((directory/'kernel.ptx').read_text(),len(info['tma_plans']));binary=(directory/'kernel.cubin').read_bytes();torch.cuda.set_device(1);torch.empty(1,device='cuda');cuda=ctypes.CDLL('libcuda.so.1');cuda.cuLaunchKernel.argtypes=[ctypes.c_void_p]+[ctypes.c_uint]*7+[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p];cuda.cuModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p];cuda.cuModuleLoadData.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p];cuda.cuModuleUnload.argtypes=[ctypes.c_void_p];module=ctypes.c_void_p();function=ctypes.c_void_p();assert cuda.cuModuleLoadData(ctypes.byref(module),ctypes.c_char_p(binary))==0;assert cuda.cuModuleGetFunction(ctypes.byref(function),module,b'frisk_execution')==0
rng=np.random.default_rng(0);q,k,v=[rng.uniform(-1,1,(64,16)).astype(np.float16) for _ in range(3)];s=np.zeros((64,64),np.float32)
for t in range(16):s=np.float32(s+q[:,t,None].astype(np.float32)*k[None,:,t].astype(np.float32))
e=np.exp(np.float32(s*.25-(s*.25).max(1,keepdims=True)),dtype=np.float32);p=np.float32(e/e.sum(1,keepdims=True)).astype(np.float16).astype(np.float32);ref=np.zeros((64,16),np.float32)
for t in range(64):ref=np.float32(ref+p[:,t,None]*v[None,t,:].astype(np.float32))
records=[]
for offset in ((1,2,3) if info['copy_path']=='vector' else (0,128,256)):
 buffers=[]
 for x in (q,k,v):
  b=torch.full((2320,),165,device='cuda',dtype=torch.uint8);b[offset:offset+2048].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(b)
 output=torch.full((4368,),165,device='cuda',dtype=torch.uint8);pointers=[b.data_ptr()+offset for b in buffers+[output]];validate_launch(info,pointers);maps=TensorMaps(info,pointers) if info['tma_plans'] else None;descriptors=[d.data_ptr() for d in maps.device] if maps else [];args=[ctypes.c_uint64(p) for p in pointers+descriptors+[0]];assert len(args)==len(physical);params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args]);expected=[b.cpu().numpy().copy() for b in buffers]
 for repeat in range(3):
  output.fill_(165);torch.cuda.synchronize();status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(0),params,ctypes.c_void_p(0));assert status==0,status;torch.cuda.synchronize();raw=output.cpu().numpy();got=raw[offset:offset+4096].copy().view(np.float32).reshape(64,16);assert np.isfinite(got).all() and np.all(np.abs(got-ref)<=.002+.002*np.abs(ref));assert np.all(raw[:offset]==165) and np.all(raw[offset+4096:]==165);assert all(np.array_equal(b.cpu().numpy(),x) for b,x in zip(buffers,expected))
print(json.dumps({'physical_params':physical,'cubin':str(directory/'kernel.cubin'),'sha256':hashlib.sha256(binary).hexdigest(),'offsets':([1,2,3] if info['copy_path']=='vector' else [0,128,256]),'repeats':3,'all_checked':True}))
assert cuda.cuModuleUnload(module)==0
