"""Execute the standard protected online-attention fixture on an SM90 GPU.

Usage: python qualify_wide_online.py /path/to/frisk-opt
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from backend import compile_execution

def qualify(opt):
    repo = Path(__file__).resolve().parents[3]
    fixture = repo/'test/Transforms/protected-online-execution.mlir'
    with tempfile.TemporaryDirectory(prefix='frisk-online-') as directory:
        packet = Path(directory)/'execution.json'
        subprocess.run([opt,fixture,'--frisk-protected-layout-pipeline',
                        '--frisk-export-execution=output='+str(packet)+' minify=true',
                        '-verify-each','-o','/dev/null'],check=True)
        image, info = compile_execution(json.loads(packet.read_text()))
    assert info['threads']==256 and len(info['root_order'])==10
    torch.manual_seed(4519)
    q,k,v = [torch.randn((256,64),device='cuda',dtype=torch.float16)*0.5 for _ in range(3)]
    for kind in ('random','rescale','uniform'):
        if kind=='rescale':
            q[:,0] = 8
            k[:,0] = (torch.arange(256,device='cuda')//64)*4
        elif kind=='uniform':
            q.zero_()
            v = torch.arange(256,device='cuda',dtype=torch.float16)[:,None].expand(-1,64).contiguous()
        guard = torch.full((256*64+128,),-5831.,device='cuda')
        out = guard[64:-64].view(256,64)
        tiles = [x[i:i+64] for i in range(0,256,64) for x in (k,v)]
        image[(1,1,1)](q,*tiles,out)
        expected = (q.double()@k.double().T/8).softmax(1)@v.double()
        error = (out-expected).abs().max().item()
        assert torch.isfinite(out).all() and torch.all(guard[:64]==-5831.) and torch.all(guard[-64:]==-5831.)
        assert error == 0 if kind=='uniform' else error < 5e-4
        print(kind,error)

if __name__ == '__main__':
    qualify(sys.argv[1])
