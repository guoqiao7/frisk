"""Checked launch ABI and lifetime-owned CUDA tensor-map encoding."""
import ctypes,re,time
from functools import lru_cache

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

@lru_cache(maxsize=1)
def tensor_map_encoder():
 cuda=ctypes.CDLL('libcuda.so.1');fn=cuda.cuTensorMapEncodeTiled
 fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
 fn.restype=ctypes.c_int
 return fn


def encode_map(fn, addr, ptr, dims, strides, box, swizzle):
 rank=len(dims)
 u64=ctypes.c_uint64
 u32=ctypes.c_uint32
 status=fn(ctypes.c_void_p(addr),6,rank,ctypes.c_void_p(ptr),(u64*rank)(*dims),
           (u64*(rank-1))(*strides),(u32*rank)(*box),(u32*rank)(*([1]*rank)),
           0,{0:0,32:1,64:2,128:3}[swizzle],0,0)
 if status:raise ValueError('cuTensorMapEncodeTiled failed '+str(status))


def prepare_attention_k_map(plan,k):
 import torch
 if k.dtype!=torch.float16 or not k.is_cuda or k.data_ptr()%16:
  raise ValueError('invalid K tensor for the FP16 tensor map')
 cols,rows,heads=plan['dimensions']
 if (not k.is_contiguous() or k.numel()!=cols*rows*heads or
     plan['strides']!=[2*cols,2*cols*rows]):
  raise ValueError('packed K tensor extent disagrees with the encoded map')
 return _prepare_attention_map(plan,k)

def prepare_attention_v_map(plan,v):
 import torch
 if plan['swizzle_bytes']==128 and len(plan['dimensions'])==3:
  if plan['box'][0]!=64 or plan['box'][2]!=1 or plan.get('abi')!='value':
   raise ValueError('invalid packed D64 row-major V tensor map')
  return prepare_attention_k_map(plan,v)
 width,rows,segments,groups,heads=plan['dimensions']
 tail = plan.get('tail_rows', 0)
 if type(tail) is not int or not 0 <= tail < 8 or groups < 1:
  raise ValueError('invalid complete-group or tail extent in the V tensor map')
 head_rows = groups*8 + tail
 columns=16*segments
 if ((width,rows)!=(16,8) or segments not in (4,8) or not v.is_cuda or v.dtype!=torch.float16
     or not v.is_contiguous() or v.data_ptr()%16 or v.numel()!=heads*head_rows*columns
     or plan['strides']!=[2*columns,32,16*columns,head_rows*2*columns]):
  raise ValueError('packed V tensor extent disagrees with the encoded SW32 map')
 return _prepare_attention_map(plan,v)

def _prepare_attention_map(plan,tensor):
 import torch
 # A prepared CPU descriptor is also Triton's native by-value launch argument.
 from triton.backends.nvidia.driver import TmaDescKernelParam
 host=TmaDescKernelParam()
 addr=host.tma_desc_cpu_ptr()
 if addr%64:raise ValueError('host tensor map must be 64-byte aligned')
 fn=tensor_map_encoder()
 with torch.cuda.device(tensor.device):
  encode_map(fn,addr,tensor.data_ptr(),plan['dimensions'],plan['strides'],plan['box'],plan['swizzle_bytes'])
  if plan.get('abi','global')=='value':
   return host
  # The global ABI needs an immutable upload, complete before other streams use it.
  return host.desc.to(tensor.device,non_blocking=False)


class TensorMaps:
 def __init__(self,info,pointers):
  validate_launch(info,pointers);self.host=[];self.device=[];self.setup_ns=[]
  import torch,numpy as np
  fn=tensor_map_encoder()
  for p in info['tma_plans']:
   start=time.perf_counter_ns();d=p['descriptor'];raw=ctypes.create_string_buffer(128+63);addr=(ctypes.addressof(raw)+63)&~63
   ptr=pointers[info['root_order'].index(p['source_root'])]+d['source_offset']
   encode_map(fn,addr,ptr,d['box'],d['global_strides'],d['box'],d['swizzle_bytes'])
   payload=np.frombuffer(ctypes.string_at(addr,128),dtype=np.uint8).copy();device=torch.from_numpy(payload).to('cuda')
   if device.data_ptr()%64:raise ValueError('device descriptor alignment')
   self.host.append((raw,addr));self.device.append(device);self.setup_ns.append(time.perf_counter_ns()-start)
