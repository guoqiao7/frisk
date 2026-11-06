"""Isolate native RS four-word packing after actual selected SSA P production."""
import json
from pathlib import Path
import torch,triton,numpy as np
from triton.backends.compiler import GPUTarget
from backend import Emitter
r=Path('/home/baopeihua/frisk');a=r/'guoqiao/optimize_plan/p1_4b_artifacts/packing';a.mkdir(exist_ok=True);data=json.loads((r/'guoqiao/optimize_plan/p1_4b_artifacts/execution-run/execution.json').read_text())
class Packing(Emitter):
 def mma(self,o):
  if o['attrs']['frisk.mma_contract']['form']=='ss':return super().mma(o)
  # Reuse production native map/contract validation, then discard only the
  # arithmetic instructions and output the exact pack inputs and packed words.
  n=len(self.code);super().mma(o);self.code=self.code[:n];src=self.regs(o['operands'][0]);root=6
  for i,z in enumerate(src):
   off=self.map([(t*32+i)*2 for t in range(128)]);addr=self.address(root,off);bits=self.tmp();self.emit(f'cvt.u32.u16 {bits}, {z};');self.emit(f'st.global.u8 [{addr}], {bits};');self.emit(f'shr.u32 {bits}, {bits}, 8;');self.emit(f'st.global.u8 [{addr}+1], {bits};')
  for pair in range(16):
   word=self.tmp();self.emit('mov.b32 '+word+', {'+', '.join(src[pair*2:pair*2+2])+'};');off=self.map([8192+(t*16+pair)*4 for t in range(128)]);addr=self.address(root,off)
   for byte in range(4):
    bits=self.tmp();self.emit(f'bfe.u32 {bits}, {word}, {byte*8}, 8;');self.emit(f'st.global.u8 [{addr}+{byte}], {bits};')
 def store(self,o):pass
text,info=Packing(data).run();path=a/'packing.ttgir';path.write_text(text);torch.cuda.set_device(1);ker=triton.compile(str(path),target=GPUTarget('cuda',90,32),options={'enable_fp_fusion':False});(a/'packing.cubin').write_bytes(ker.asm['cubin']);(a/'packing.ptx').write_text(ker.asm['ptx']);torch.manual_seed(7);args=[torch.rand((64,16),device='cuda',dtype=torch.float16)*2-1 for _ in range(3)];out=torch.full((16384+16,),165,device='cuda',dtype=torch.uint8)
for offset in (1,2,3):
 ker[(1,1,1)](*args,out[offset:]);torch.cuda.synchronize();raw=out.cpu().numpy();halves=raw[offset:offset+8192].copy().view(np.uint16).reshape(128,32);words=raw[offset+8192:offset+16384].copy().view(np.uint32).reshape(128,16);expected=halves[:,0::2].astype(np.uint32)|(halves[:,1::2].astype(np.uint32)<<16);assert np.array_equal(words,expected)
(a/'result.json').write_text(json.dumps({'offsets':[1,2,3],'packed_low_high_words':2048,'bit_exact':True,'actual_p_encoding':data['values'][32]['distributed']['encoding']}));print('packing low/high:2048 words exact at offsets1/2/3')
