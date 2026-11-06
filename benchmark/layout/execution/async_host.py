"""Checked launch ABI and lifetime-owned CUDA tensor-map encoding."""
import ctypes,re,time

def validate_launch(info,pointers):
 if len(pointers)!=4 or any(not isinstance(p,int) or p<=0 for p in pointers):raise ValueError('four valid Global pointers required')
 for p,align in zip(pointers,info['global_alignments']):
  if p%align:raise ValueError('Global pointer violates declared root alignment')
 for p in info['tma_plans']:
  d=p['descriptor']
  if (pointers[info['root_order'].index(p['source_root'])]+d['source_offset'])%16:raise ValueError('TMA source offset alignment')

def physical_abi(ptx,descriptor_count):
 match=re.search(r'\.visible\s+\.entry\s+frisk_execution\s*\((.*?)\)',ptx,re.S)
 if not match:raise ValueError('PTX entry not found')
 params=re.findall(r'\.param\s+\.u64\s+\.ptr\s+\.global\s+\.align\s+1\s+(\w+)',match.group(1))
 if len(params)!=5+descriptor_count or match.group(1).count('.param')!=len(params):raise ValueError('unsupported physical PTX parameter ABI')
 return params

class TensorMaps:
 def __init__(self,info,pointers):
  validate_launch(info,pointers);self.host=[];self.device=[];self.setup_ns=[]
  import torch,numpy as np
  cuda=ctypes.CDLL('libcuda.so.1');fn=cuda.cuTensorMapEncodeTiled
  fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
  fn.restype=ctypes.c_int
  for p in info['tma_plans']:
   start=time.perf_counter_ns();d=p['descriptor'];raw=ctypes.create_string_buffer(128+63);addr=(ctypes.addressof(raw)+63)&~63
   dims=(ctypes.c_uint64*2)(*d['box']);strides=(ctypes.c_uint64*1)(*d['global_strides']);box=(ctypes.c_uint32*2)(*d['box']);steps=(ctypes.c_uint32*2)(1,1)
   ptr=pointers[info['root_order'].index(p['source_root'])]+d['source_offset']
   status=fn(addr,6,2,ptr,dims,strides,box,steps,0,{0:0,32:1}[d['swizzle_bytes']],0,0)
   if status:raise ValueError('cuTensorMapEncodeTiled failed '+str(status))
   payload=np.frombuffer(ctypes.string_at(addr,128),dtype=np.uint8).copy();device=torch.from_numpy(payload).to('cuda')
   if device.data_ptr()%64:raise ValueError('device descriptor alignment')
   self.host.append((raw,addr));self.device.append(device);self.setup_ns.append(time.perf_counter_ns()-start)
