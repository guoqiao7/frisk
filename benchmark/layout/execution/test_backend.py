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
  roots={values[identity]['storage']['root'] for identity in op['operands'][:2]}
  # Widen each complete storage family, including the canonical root. An
  # undersized root now correctly fails earlier in the allocation planner;
  # this fixture must stay in bounds to exercise the copy's bit-width rule.
  for value in values.values():
   if value.get('storage',{}).get('root') in roots:
    value['type']=value['type'].replace('xf16','xf32')
    for address in value['storage']['addresses']:address['begin_bit']*=2;address['end_bit']*=2
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
 def test_checked_attention_reductions_keep_an_eager_control(self):
  data=json.loads(Path(INPUT).read_text())
  old,eager=backend.emit(data,reduction_policy='eager')
  new,optimized=backend.emit(data)
  self.assertEqual([row['strategy'] for row in eager['reduction_plans']],['ordered-eager']*2)
  self.assertEqual([row['strategy'] for row in optimized['reduction_plans']],['ordered-butterfly']*2)
  self.assertTrue(all(row['butterfly_masks']==[[1,2],[1,2]] for row in optimized['reduction_plans']))
  self.assertLess(new.count('shfl.sync'),old.count('shfl.sync'))
  reduction=next(op for op in data['operations'] if op['name']=='frisk.reduce_tensor')
  root=reduction['fibers'][0]['nodes'][-1]
  root['left'],root['right']=root['right'],root['left']
  for mode in ('auto','eager'):
   with self.assertRaisesRegex(ValueError,'noncanonical reduction'):
    backend.emit(data,reduction_policy=mode)
 def test_aligned_fp16_uses_packed_four_word_transport(self):
  slots=list(range(1024))
  addresses=[dict(begin_bit=16*i,end_bit=16*(i+1),view=[i],root=[i]) for i in slots]
  storage={'id':0,'type':'memref<1024xf16, 1>','storage':{'root':0,'alignment':16,'addresses':addresses}}
  tensor={'id':1,'type':'tensor<1024xf16>','distributed':dict(shape=[1024],topology=[8,32,4,1,1],
                                                            logical=slots,first=slots,replication=1)}
  emitter=backend.Emitter({'values':[storage,tensor],'operations':[]})
  emitter.roots={0:0}
  emitter.load({'operands':[0],'results':[1]})
  emitter.store({'operands':[1,0]})
  self.assertEqual([x['groups_by_width'] for x in emitter.global_access],[{16:1},{16:1}])
  self.assertEqual(sum(line.startswith('ld.global.v4.b32') for line in emitter.code),1)
  self.assertEqual(sum(line.startswith('st.global.v4.b32') for line in emitter.code),1)
  self.assertFalse(any('global.u8' in line for line in emitter.code))
if __name__=='__main__':
 arg=sys.argv[1];sys.argv=[sys.argv[0]];BackendTests.test_complete_input_drives_operations.__globals__['INPUT']=arg
 # Explicit test input retained separately from unittest options.
 sys.argv.append(arg);unittest.main(argv=[sys.argv[0]])
