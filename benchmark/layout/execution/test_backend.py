import unittest,json,sys,copy
from pathlib import Path
import backend
class BackendTests(unittest.TestCase):
 def test_complete_input_drives_operations(self):
  data=json.loads(Path(INPUT).read_text())
  text,info=backend.emit(data)
  self.assertIn('wgmma.mma_async.sync.aligned.m64n64k16',text)
  self.assertEqual(text.count('wgmma.mma_async.sync.aligned.m64n16k16'),4)
  self.assertIn('fence.proxy.async.shared::cta',text)
  self.assertIn('cvt.rn.f16.f32',text)
  self.assertEqual(text.count('mma_acc, 1, 1, 1;'),4)
  self.assertEqual(info['emitted_operations'],len(data['operations']))
 def test_full_coordinate_maps_include_replica_unit_and_atomic_refusal(self):
  for table in ([i for i in range(128)],[i^16 for i in range(128)],[i&63 for i in range(128)],[0]):
   base,bits=backend.linear(table)
   for i,value in enumerate(table):
    got=base
    for bit,basis in enumerate(bits):
     if i>>bit&1:got^=basis
    self.assertEqual(got,value)
  with self.assertRaises(ValueError):backend.linear([0,1,2,7])
 def test_wrong_geometry_and_packing_reject(self):
  original=json.loads(Path(INPUT).read_text())
  for field,value in [('atom',[32,64,16]),('packing','wrong'),('target','sm_80')]:
   data=copy.deepcopy(original);op=next(o for o in data['operations'] if o['name']=='frisk.mma');op['attrs']['frisk.mma_contract'][field]=value
   with self.assertRaises(ValueError):backend.emit(data)
 def test_ss_mn_major_is_rejected(self):
  data=json.loads(Path(INPUT).read_text());op=next(o for o in data['operations'] if o['name']=='frisk.mma');op['attrs']['frisk.mma_contract']['a_descriptor']['major']='mn'
  with self.assertRaisesRegex(ValueError,'SS.*K-major'):backend.emit(data)
 def test_copy_non_f16_is_rejected(self):
  data=json.loads(Path(INPUT).read_text());op=next(o for o in data['operations'] if o['name']=='frisk.copy');values={v['id']:v for v in data['values']}
  for identity in op['operands'][:2]:
   values[identity]['type']=values[identity]['type'].replace('xf16','xf32')
   for address in values[identity]['storage']['addresses']:address['begin_bit']*=2;address['end_bit']*=2
  with self.assertRaisesRegex(ValueError,'Copy requires.*16-bit'):backend.emit(data)
 def test_nonlast_broadcast_axis_is_explicitly_rejected(self):
  # Legal rank1->rank2 broadcast across dimension0, with128 replicas.
  # Correct output registers are r0,r1,r0,r1; the old implementation
  # silently selected r0,r0,r1,r1 by dividing flattened coords by2.
  source={'id':0,'type':'tensor<2xf32>','distributed':{'shape':[2],'topology':[2,32,4,1,1],'logical':[0,1]*128,'replication':128}}
  target={'id':1,'type':'tensor<2x2xf32>','distributed':{'shape':[2,2],'topology':[4,32,4,1,1],'logical':[0,1,2,3]*128,'replication':128}}
  emitter=backend.Emitter({'values':[source,target],'operations':[]});emitter.reg[0]=['r0','r1']
  with self.assertRaisesRegex(ValueError,'broadcast requires rank1.*dimensions=\[1\]'):
   emitter.broadcast({'operands':[0],'results':[1],'attrs':{'dimensions':[0]}})
if __name__=='__main__':
 arg=sys.argv[1];sys.argv=[sys.argv[0]];BackendTests.test_complete_input_drives_operations.__globals__['INPUT']=arg
 # Explicit test input retained separately from unittest options.
 sys.argv.append(arg);unittest.main(argv=[sys.argv[0]])
