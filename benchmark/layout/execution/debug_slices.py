import json,copy,sys
from pathlib import Path
import numpy as np,torch,triton
from triton.backends.compiler import GPUTarget
from backend import emit
r=Path('/home/baopeihua/frisk');a=r/'guoqiao/optimize_plan/p1_4b_artifacts';data=json.loads((a/'execution-run/execution.json').read_text());torch.cuda.set_device(1)
rng=np.random.default_rng(0);q,k,v=[rng.uniform(-1,1,(64,16)).astype(np.float16) for _ in range(3)];gpu=[torch.tensor(x,device='cuda') for x in (q,k,v)];score=q.astype(np.float32)@k.astype(np.float32).T;exp=np.exp(score*.25-(score*.25).max(1,keepdims=True));prob=exp/exp.sum(1,keepdims=True)
for target,ref in [(17,score),(31,prob)]:
 x=copy.deepcopy(data);ops=x['operations'];end=next(i for i,o in enumerate(ops) if target in o['results']);x['operations']=ops[:end+1]+[dict(name='frisk.tile_store',operands=[target,6],results=[],attrs={})]
 root=next(v for v in x['values'] if v['id']==6);root['storage']['addresses']=[dict(view=[i,j],root=[i,j],begin_bit=(i*64+j)*32,end_bit=(i*64+j+1)*32) for i in range(64) for j in range(64)]
 text,info=emit(x);path=a/f'slice-{target}.ttgir';path.write_text(text);ker=triton.compile(str(path),target=GPUTarget('cuda',90,32),options={'enable_fp_fusion':False});out=torch.empty((64,64),device='cuda');ker[(1,1,1)](*gpu,out);got=out.cpu().numpy();err=np.abs(got-ref);print(target,'err',err.max(),'got',got[:2,:8],'ref',ref[:2,:8],flush=True);np.save(a/f'slice-{target}.npy',got)
