"""Check the native local certificates, full Cartesian search, and loop export."""
import json
import re
import sys
from pathlib import Path

ir = Path(sys.argv[1]).read_text()
header = next(line for line in ir.splitlines() if 'frisk.protected_graph =' in line)
proofs = header.split('combinations = ')[0]
assert proofs.count('eligible = false, grid = array<i64: 1, 2>') == 2
assert proofs.count('eligible = true, grid = array<i64: 2, 1>') == 4
assert header.count('bindings_verified = true') == 4
assert list(map(int, re.findall(r'internal_conversions = (\d+)',header))) == [0,1,3,0]
assert 'selected = array<i64: 1, 1>' in header
packet = json.loads(Path(sys.argv[2]).read_text())
ops = packet['operations']
loop, = [op for op in ops if op['name'] == 'scf.for']
assert len(loop['body_arguments']) == 4
assert len(ops[loop['body_terminator']]['operands']) == 3
assert sum(op['name']=='scf.if' for op in ops) == 4
contracts = [op['attrs']['frisk.mma_contract'] for op in ops if op['name']=='frisk.mma']
assert [c['form'] for c in contracts] == ['ss','rs']
assert all(c['atom']==[64,64,16] and c['grid']==[2,1] for c in contracts)
assert [p['redistribution']['mode'] for p in ops if 'redistribution' in p] == ['Shared']
