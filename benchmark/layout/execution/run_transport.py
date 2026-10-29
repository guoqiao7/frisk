import json,subprocess
from pathlib import Path
import numpy as np,torch,triton
from triton.backends.compiler import GPUTarget
from backend import emit
from transport_fixture import fixture
r=Path('/home/baopeihua/frisk');a=r/'guoqiao/optimize_plan/p1_4b_artifacts/transport';a.mkdir(exist_ok=True);torch.cuda.set_device(1);records=[]
x=np.linspace(-1,1,1024,dtype=np.float32);x[:12]=[0,-0.0,2**-25,3*2**-25,-2**-25,2**-24,2**-14,1+2**-11,1+3*2**-11,-1-2**-11,.5,-.5]
for mode in ('base','replica','unit'):
 for swap,cast in ((False,False),(False,True),(True,False),(True,True)) if mode=='base' else ((False,False),):
  tag=f'{mode}-layout{int(swap)}-cast{int(cast)}';source=a/(tag+'.mlir');source.write_text(fixture(swap,cast,mode));export=a/(tag+'.json');q=subprocess.run([str(r/'build-m6-release/bin/frisk-opt'),str(source),'-frisk-export-execution=output='+str(export),'-o','/dev/null'],capture_output=True,text=True)
  if q.returncode:raise RuntimeError(q.stderr)
  text,info=emit(json.loads(export.read_text()));tt=a/(tag+'.ttgir');tt.write_text(text);k=triton.compile(str(tt),target=GPUTarget('cuda',90,32),options={'enable_fp_fusion':False});(a/(tag+'.cubin')).write_bytes(k.asm['cubin']);(a/(tag+'.ptx')).write_text(k.asm['ptx']);ref=x.astype(np.float16) if cast else x;rawref=ref.view(np.uint8)
  for offset in (1,2,3):
   qb=torch.full((x.nbytes+16,),165,dtype=torch.uint8,device='cuda');qb[offset:offset+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8)));ob=torch.full((ref.nbytes+16,),165,dtype=torch.uint8,device='cuda')
   expected_input=qb.cpu().numpy().copy()
   for repeat in range(3):
    ob.fill_(165)
    k[(1,1,1)](qb[offset:],qb[offset:],qb[offset:],ob[offset:])
    torch.cuda.synchronize();raw=ob.cpu().numpy();assert np.array_equal(raw[offset:offset+ref.nbytes],rawref);assert np.all(raw[:offset]==165) and np.all(raw[offset+ref.nbytes:]==165);assert np.array_equal(qb.cpu().numpy(),expected_input);records.append(dict(mode=mode,layout=int(swap),cast=cast,offset=offset,repeat=repeat,bit_exact=True,guards=True,input_bytes_and_guards_unchanged=True))
(a/'result.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
