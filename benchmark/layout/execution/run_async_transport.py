"""Bitexact probes: actual selected Copy followed by Shared-byte observation."""
import os
os.environ['TRITON_ALWAYS_COMPILE']='1'
import copy,json,tempfile,hashlib
from pathlib import Path
import torch,numpy as np,triton
from triton.backends.compiler import GPUTarget
from backend import Emitter
from async_host import TensorMaps,physical_abi
ROOT=Path(__file__).resolve().parents[3];ART=ROOT/'guoqiao/optimize_plan/p1_4c_artifacts'
class Probe(Emitter):
 def copy(self,o):
  super().copy(o)
  b=o['operands'][1];e=o['execution'];n=e['topology'][0];dst=self.offsets(b);output=sorted(self.roots)[3]
  for i in range(n):
   slots=[e['logical'][t*n+i] for t in range(128)];src=self.shared(b,self.map([dst[x] for x in slots]));target=self.address(output,self.map([x*2 for x in slots]));value=self.tmp();self.emit(f'ld.shared.b16 {value}, [{src}];');self.emit(f'st.global.b16 [{target}], {value};')
torch.cuda.set_device(1);records=[]
for path in ['cp.async','tma']:
 original=json.loads((ART/(path+'.json')).read_text());copies=[o for o in original['operations'] if o['name']=='frisk.copy']
 for ci,op in enumerate(copies):
  data=copy.deepcopy(original);data['operations']=[o for o in data['operations'] if o['name']=='frisk.storage_contract' or o==op];text,info=Probe(data).run();out=Path(tempfile.mkdtemp(prefix='run-',dir=ART/'transport'));(out/'kernel.ttgir').write_text(text)
  kernel=triton.compile(str(out/'kernel.ttgir'),target=GPUTarget('cuda',90,32),options={'num_warps':4,'enable_fp_fusion':False});physical_abi(kernel.asm['ptx'],len(info['tma_plans']))
  for ext,content in kernel.asm.items():
   p=out/('kernel.'+ext);p.write_bytes(content) if isinstance(content,bytes) else p.write_text(content)
  for offset in (0,128,256):
   payload=np.random.default_rng(ci).integers(0,65536,(1024,),dtype=np.uint16).view(np.uint8);buffers=[torch.full((2320,),165,dtype=torch.uint8,device='cuda') for _ in range(3)];output=torch.full((4368,),165,dtype=torch.uint8,device='cuda')
   for b in buffers:b[offset:offset+2048].copy_(torch.from_numpy(payload))
   expected=[b.cpu().numpy().copy() for b in buffers];ptrs=[b.data_ptr()+offset for b in buffers]+[output.data_ptr()+offset];maps=TensorMaps(info,ptrs) if info['tma_plans'] else None
   for repeat in range(3):
    output.fill_(165);kernel[(1,1,1)](*[b[offset:] for b in buffers],output[offset:],*(maps.device if maps else []));torch.cuda.synchronize();got=output.cpu().numpy();assert np.array_equal(got[offset:offset+2048],payload);assert np.all(got[:offset]==165) and np.all(got[offset+2048:]==165);assert all(np.array_equal(b.cpu().numpy(),x) for b,x in zip(buffers,expected));records.append({'path':path,'copy':ci,'offset':offset,'repeat':repeat,'bitexact':True,'guards':True,'inputs_unchanged':True,'artifact':str(out)})
(ART/'transport/results.json').write_text(json.dumps(records,indent=2));print('PASS',len(records))
