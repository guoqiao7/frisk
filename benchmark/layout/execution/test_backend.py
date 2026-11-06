import unittest,json,sys,copy
from pathlib import Path
import backend
class BackendTests(unittest.TestCase):
 def test_complete_input_drives_operations(self):
  data=json.loads(Path(INPUT).read_text())
  text,info=backend.emit(data,normalization_pack=False)
  self.assertIn('wgmma.mma_async.sync.aligned.m64n64k16',text)
  self.assertEqual(text.count('wgmma.mma_async.sync.aligned.m64n16k16'),4)
  self.assertIn('fence.proxy.async.shared::cta',text)
  self.assertIn('cvt.rn.f16.f32',text)
  self.assertEqual(text.count('mma_acc, 1, 1, 1;'),4)
  self.assertEqual(info['emitted_operations'],len(data['operations']))
 def test_normalization_pack_preserves_rounding_and_native_consumption(self):
  data=json.loads(Path(INPUT).read_text())
  packed,info=backend.emit(data)
  scalar,control=backend.emit(data,normalization_pack=False)
  pairs=info['normalization_pack_pairs']
  self.assertNotIn('normalization_pack_pairs',control)
  self.assertEqual(info['p_rounding'],control['p_rounding'])
  self.assertEqual(packed.count('cvt.rn.f16x2.f32'),pairs)
  self.assertEqual(scalar.count('cvt.rn.f16.f32'),pairs*2)
  self.assertNotIn('cvt.rn.f16.f32',packed)
  for instruction in ('div.rn.f32','wgmma.mma_async','wgmma.wait_group','bar.sync'):
   self.assertEqual(packed.count(instruction),scalar.count(instruction))
  self.assertEqual(info['emitted_operations'],control['emitted_operations'])
 def test_normalization_pack_requires_single_use_and_no_intervening_barrier(self):
  data=json.loads(Path(INPUT).read_text())
  emitter=backend.Emitter(data)
  operations=data['operations']
  ordinal,=emitter._normalization_pack_sites(operations,set())
  quotient=operations[ordinal]['results'][0]
  probability=operations[ordinal+1]['results'][0]
  for extra in (quotient,probability):
   self.assertFalse(emitter._normalization_pack_sites(
       [*operations,{'name':'test.use','operands':[extra]}],set()))
  self.assertFalse(emitter._normalization_pack_sites(operations,{ordinal+1}))
  fast_ops=copy.deepcopy(operations)
  fast_ops[ordinal]['attrs']['fastmath']='#arith.fastmath<fast>'
  self.assertFalse(emitter._normalization_pack_sites(fast_ops,set()))
  denominator=operations[ordinal]['operands'][1]
  logical=emitter.v[denominator]['distributed']['logical']
  logical[0],logical[1]=logical[1],logical[0]
  self.assertFalse(emitter._normalization_pack_sites(operations,set()))
 def test_guarded_reciprocal_preserves_exact_divide_and_packing_fallback(self):
  data=json.loads(Path(INPUT).read_text())
  original,baseline=backend.emit(data)
  optimized,info=backend.emit(data,normalization_division='guarded-row-reciprocal')
  proof,=info['normalization_reciprocal']
  pairs=baseline['normalization_pack_pairs']
  self.assertEqual((proof['row_denominators'],proof['pairs']),(2,pairs))
  self.assertEqual(optimized.count('rcp.rn.f64'),2)
  self.assertEqual(optimized.count('mul.rn.f64'),pairs*2)
  self.assertEqual(optimized.count('div.rn.f32'),pairs*2)
  original_divides=[line for line in original.split(r'\0A') if line.startswith(('div.rn.f32','cvt.rn.f16x2'))]
  fallback=optimized.split('exact_normalize_fallback:',1)[1].split('exact_normalize_merge:',1)[0]
  self.assertEqual([line for line in fallback.split(r'\0A') if line],original_divides)
  self.assertLess(optimized.index('exact_normalize_merge:'),optimized.rindex('wgmma.fence.sync.aligned'))
  self.assertEqual(original.count('wgmma.wait_group'),optimized.count('wgmma.wait_group'))
 def test_guarded_reciprocal_requires_checked_ordered_row_sum(self):
  data=json.loads(Path(INPUT).read_text())
  emitter=backend.Emitter(data)
  operations=emitter.data['operations']
  ordinal,=emitter._normalization_pack_sites(operations,set())
  self.assertEqual(emitter._normalization_reciprocal_sites(operations,{ordinal}),{ordinal})
  for name,edit in (
    ('sum',lambda operation: operation['attrs'].update(kind='max')),
    ('broadcast',lambda operation: operation['attrs'].update(dimensions=[0])),
    ('exp',lambda operation: operation.update(name='arith.addf'))):
   bad=copy.deepcopy(operations)
   if name=='sum':
    target=next(o for o in bad if o['name']=='frisk.reduce_tensor' and o['attrs']['kind'].strip('"')=='sum')
   elif name=='broadcast':
    denominator=bad[ordinal]['operands'][1]
    target=next(o for o in bad if o['results']==[denominator])
   else:
    target=next(o for o in bad if o['name']=='math.exp')
   edit(target)
   self.assertFalse(emitter._normalization_reciprocal_sites(bad,{ordinal}))
  with self.assertRaisesRegex(ValueError,'packed FP16'):
   backend.emit(data,normalization_pack=False,normalization_division='guarded-row-reciprocal')
 def test_exact_k_reuse_batches_qk_for_both_source_policies(self):
  data=json.loads(Path(INPUT).read_text())
  original,baseline=backend.emit(data,copy_batch=False)
  unbatched,info=backend.emit(data,copy_batch=False,shared_policy='reuse-k')
  batched,other=backend.emit(data,copy_batch=True,shared_policy='reuse-k')
  self.assertEqual((unbatched,info),(batched,other))
  self.assertLess(info['static_shared_bytes'],baseline['static_shared_bytes'])
  proof=info['attention_schedule']
  q,k,v=(proof['roots'][role] for role in ('q','k','v'))
  self.assertEqual((k['offset'],k['bytes']),(v['offset'],v['bytes']))
  self.assertLess(k['reader_operation'],v['copy_operation'])
  self.assertLess(v['copy_operation'],v['reader_operation'])
  self.assertEqual(proof['cta_barriers'],{'publish_qk':1,'before_v_overwrite':1,'publish_v':1})
  self.assertEqual(unbatched.count('wgmma.wait_group'),original.count('wgmma.wait_group'))
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
