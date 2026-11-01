"""Preregistered weak-ABI f32 mechanism graphs, seed 20261030.
Each arithmetic operation is separately rounded float32, no FMA/reassociation.
"""
import numpy as np
from transport_fixture import encoding,fixture as transport

def cases():
 return [dict(id=x,shape=[64,16],threads=128,producer_candidates=['A','B'],entry='raw-analysis-candidate-injection',formula=f) for x,f in [('no-conflict','p=q*k; O=p'),('conflict','p=q*k; a=p*p; b=p*p; O=a-b*k'),('diamond','p=q; a=p*p; b=p*p; O=a-b*v'),('remat','p=q*q; a=p*p; b=p*p; O=a-b*v')]]
def inputs():
 rng=np.random.default_rng(20261030)
 return tuple(rng.uniform(lo,hi,1024).astype(np.float32) for lo,hi in [(0.25,1.25),(1.5,2.5),(-0.75,-0.25)])
def branch_reference(case_id,q,k,v,perturb=-1):
 mul=lambda a,b:np.multiply(a,b,dtype=np.float32)
 sub=lambda a,b:np.subtract(a,b,dtype=np.float32)
 if case_id=='no-conflict':return mul(q,k)
 if case_id=='conflict':p=mul(q,k);factor=k
 elif case_id=='diamond':p=q;factor=v
 elif case_id=='remat':p=mul(q,q);factor=v
 else:raise ValueError('unknown graph')
 a=mul(p,p);b=mul(p,p)
 if perturb==0:a=np.add(a,np.float32(0.25),dtype=np.float32)
 if perturb==1:b=np.add(b,np.float32(0.25),dtype=np.float32)
 return sub(a,mul(b,factor))
def reference(case_id,q,k,v):return branch_reference(case_id,q,k,v)
def fixture(case_id):
 if case_id not in [c['id'] for c in cases()]:raise ValueError('unknown graph')
 text=transport();start=text.index(' %x=frisk.tile_load');end=text.index(' frisk.end',start)
 body=' %kv=frisk.layout_view %k {layout=#F}:memref<64x16xf32,1> -> memref<64x16xf32,1>\n %vv=frisk.layout_view %v {layout=#F}:memref<64x16xf32,1> -> memref<64x16xf32,1>\n %q0=frisk.tile_load %qv : memref<64x16xf32,1> -> tensor<64x16xf32>\n %k0=frisk.tile_load %kv : memref<64x16xf32,1> -> tensor<64x16xf32>\n %v0=frisk.tile_load %vv : memref<64x16xf32,1> -> tensor<64x16xf32>\n'
 if case_id=='no-conflict':body+='%p=arith.mulf %q0,%k0 : tensor<64x16xf32>\n';result='%p'
 else:
  if case_id=='conflict':body+='%p=arith.mulf %q0,%k0 : tensor<64x16xf32>\n';p='%p'
  elif case_id=='remat':body+='%p=arith.mulf %q0,%q0 : tensor<64x16xf32>\n';p='%p'
  else:p='%q0'
  body+=f'%a=arith.mulf {p},{p} : tensor<64x16xf32>\n%b=arith.mulf {p},{p} : tensor<64x16xf32>\n'
  factor='%k0' if case_id=='conflict' else '%v0'
  body+=f'%bk=arith.mulf %b,{factor} : tensor<64x16xf32>\n%out=arith.subf %a,%bk : tensor<64x16xf32>\n';result='%out'
 body+=f'frisk.tile_store {result},%ov : tensor<64x16xf32>,memref<64x16xf32,1>\n'
 return (text[:start]+body+text[end:]).replace('module attributes {frisk.target="sm_90a"}', 'module attributes {frisk.target="sm_90a",joint.A=#A,joint.B=#B}')
