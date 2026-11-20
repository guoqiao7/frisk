"""CPU-only checks of sealed calibration views and mechanical code compatibility."""
import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import compatible_native_profiles as c

class CompatibleNativeProfilesTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.native=c.ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts/native-fit-r3/profile.json'
  cls.local=c.ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts/local-fit-r1/profile.json'
  if not cls.native.is_file() or not cls.local.is_file() or not c.OLD_SNAPSHOT.is_dir():raise unittest.SkipTest('requires sealed server calibration/source evidence')
  cls.texts={n:((c.OLD_SNAPSHOT/n).read_text(),(c.ROOT/n).read_text()) for n in c.BANK_FILES if not n.endswith('native_cost_calibration.py')}
 def setUp(self):
  work=c.BASE/'development';work.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(prefix='profile-view-test-',dir=work);self.addCleanup(self.temp.cleanup);self.directory=Path(self.temp.name)
 def proof(self):return c.create_bank_compatibility_receipt(self.directory/'receipt.json',profile_path=self.local)
 def view(self,base=None):return c.create_view(self.directory/'view.json',base or self.local,self.proof())
 def test_real_original_profiles_load_without_rewriting(self):
  import torch
  self.assertFalse(torch.cuda.is_initialized());receipt=self.proof()
  for name,path in [('native',self.native),('local',self.local)]:
   before={p:c.sha(p) for p in [path,path.parent/'inputs.json',path.parent/'fit.json',path.parent/'training-ledger.json']};view=c.create_view(self.directory/(name+'.json'),path,receipt);loaded=c.load_profile(view);base=c._read(path)
   self.assertEqual(loaded['coefficients'],base['coefficients']);self.assertEqual(len(loaded['coefficients']),14)
   for key in ['features','training_maxima','model','calibration','ticks_per_microsecond','intercept_us']:
    self.assertEqual(loaded.get(key),base.get(key))
   self.assertEqual(loaded['analysis_contract']['symbolic_bank_implementation_sha256'],{n:c.sha(c.ROOT/n) for n in c.BANK_FILES})
   restored=copy.deepcopy(loaded);del restored['compatibility_view'];restored['analysis_contract']['symbolic_bank_implementation_sha256']=base['analysis_contract']['symbolic_bank_implementation_sha256'];self.assertEqual(restored,base)
   self.assertEqual(c.base_profile_path(view),path.resolve());self.assertTrue(all(c.sha(p)==h for p,h in before.items()))
  self.assertEqual(c.load_profile(self.directory/'local.json')['coefficients']['local_transfer_bytes'],6418);self.assertFalse(torch.cuda.is_initialized())
 def test_complete_recursive_dependency_chain(self):
  view=self.view();deps=c.dependency_hashes(view);inputs=c._read(self.local.parent/'inputs.json')
  expected=[view,self.directory/'receipt.json',self.local,self.local.parent/'inputs.json',self.local.parent/'fit.json',self.local.parent/'training-ledger.json',self.local.parent/'pair-ledger.json',self.native,self.native.parent/'inputs.json',self.native.parent/'fit.json',self.native.parent/'training-ledger.json',Path(c.__file__)]
  for key in ['known_completion','micro_completion','local_completion']:expected.append(Path(inputs[key]))
  for name in c.BANK_FILES:expected.extend([c.ROOT/name,c.OLD_SNAPSHOT/name])
  for p in expected:self.assertEqual(deps[str(p.resolve())],c.sha(p))
  self.assertGreater(len(deps),30)
 def test_original_paths_are_not_view_fit_directories(self):
  self.assertEqual(c.base_profile_path(self.native),self.native.resolve());self.assertEqual(c.base_profile_path(self.local),self.local.resolve());self.assertEqual(c.load_profile(self.native),c._read(self.native))
 def test_no_output_overwrite_or_outside_new_stage(self):
  proof=self.proof()
  with self.assertRaises(ValueError):c.create_bank_compatibility_receipt(proof,profile_path=self.local)
  with tempfile.TemporaryDirectory() as other:
   with self.assertRaises(ValueError):c.create_view(Path(other)/'view.json',self.local,proof)
 def test_receipt_rejects_arbitrary_resigning(self):
  view=self.view();receipt=self.directory/'receipt.json';value=c._read(receipt);key=c.PREFIX+'backend.py';value['payload']['new_bank_implementation_sha256'][key]='0'*64;value['payload_sha256']=c._digest(value['payload']);receipt.write_text(json.dumps(value));v=c._read(view);v['bank_compatibility_receipt_sha256']=c.sha(receipt);view.write_text(json.dumps(v))
  with self.assertRaises(ValueError):c.load_profile(view)
 def test_view_base_hash_mutation_rejected(self):
  view=self.view();value=c._read(view);value['base_profile_sha256']='0'*64;view.write_text(json.dumps(value))
  with self.assertRaises(ValueError):c.load_profile(view)
 def test_training_dependency_mutation_rejected(self):
  view=self.view();target=(self.local.parent/'pair-ledger.json').resolve();real=c.sha
  def changed(p):return '0'*64 if Path(p).resolve()==target else real(p)
  with patch.object(c,'sha',side_effect=changed):
   with self.assertRaises(ValueError):c.load_profile(view)
 def test_source_changes_after_receipt_even_comment_rejected(self):
  view=self.view();name=c.PREFIX+'backend.py';target=(c.ROOT/name).resolve();fake=self.directory/'changed-backend.py';fake.write_text((c.ROOT/name).read_text()+'\n# later mutation\n');regular=c._regular
  def redirected(p):return fake.resolve() if Path(p).absolute().resolve()==target else regular(p)
  with patch.object(c,'_regular',side_effect=redirected):
   with self.assertRaises(ValueError):c.load_profile(view)
 def test_original_native_calibration_source_byte_frozen(self):
  target=(c.ROOT/(c.PREFIX+'native_cost_calibration.py')).resolve();fake=self.directory/'changed-calibration.py';fake.write_text(target.read_text()+'\n# change\n');regular=c._regular
  def redirected(p):return fake.resolve() if Path(p).absolute().resolve()==target else regular(p)
  with patch.object(c,'_regular',side_effect=redirected):
   with self.assertRaises(ValueError):c._payload(self.local)
 def test_only_sealed_snapshot_allowed(self):
  with self.assertRaises(ValueError):c.create_bank_compatibility_receipt(self.directory/'x.json',profile_path=self.local,snapshot_root=self.directory)
 def test_actual_approved_projection(self):
  for name,(old,new) in self.texts.items():self.assertIn('projected_AST_sha256',c._projection(name,old,new))
 def test_backend_shared_instruction_mutation_rejected(self):
  name=c.PREFIX+'backend.py';old,new=self.texts[name];tree=ast.parse(new);function=c._function(tree,'convert','Emitter');strings=[n for n in ast.walk(function) if isinstance(n,ast.Constant) and isinstance(n.value,str) and '.shared.' in n.value];self.assertTrue(strings);strings[0].value+='MUTATED'
  with self.assertRaises(ValueError):c._projection(name,old,ast.unparse(tree))
 def test_redistribution_shared_scratch_mutation_rejected(self):
  name=c.PREFIX+'redistribution.py';old,new=self.texts[name];self.assertIn("volume*4 if mode=='Shared' else 0",new)
  with self.assertRaises(ValueError):c._projection(name,old,new.replace("volume*4 if mode=='Shared' else 0","volume*8 if mode=='Shared' else 0"))
 def test_audit_shared_calculation_mutation_rejected(self):
  name=c.PREFIX+'demand_coverage_audit.py';old,new=self.texts[name];tree=ast.parse(new);function=c._function(tree,'conversion_structure');shared=[n for n in ast.walk(function) if isinstance(n,ast.If) and c._dump(n.test)==c._dump(ast.parse("mode=='Shared'",mode='eval').body)];self.assertEqual(len(shared),1);shared[0].body.append(ast.Assign(targets=[ast.Name(id='value',ctx=ast.Store())],value=ast.Dict(keys=[],values=[])))
  with self.assertRaises(ValueError):c._projection(name,old,ast.unparse(ast.fix_missing_locations(tree)))
 def test_unreviewed_shuffle_side_effect_rejected(self):
  for name,(old,new) in self.texts.items():
   if name.endswith('backend.py'):new=new.replace('lane=self.map([o[\'lane\'] for o in owners]);candidates=[]','lane=self.map([o[\'lane\'] for o in owners]);candidates=[];self.shared.clear()')
   elif name.endswith('redistribution.py'):new=new.replace('actual=set(table);base=table[0];basis={}','actual=set(table);base=table[0];basis={};plan.clear()')
   else:new=new.replace('count=len(shuffle_source_support(plan,reg))','count=len(shuffle_source_support(plan,reg));plan.clear()')
   with self.assertRaises(ValueError):c._projection(name,old,new)
 def test_unrelated_domain_changes_rejected(self):
  for name,(old,new) in self.texts.items():
   if name.endswith('backend.py'):changed=new.replace('])>512 for p in plans.values()','])>1024 for p in plans.values()')
   elif name.endswith('redistribution.py'):changed=new.replace('0<nr<=512','0<nr<=1024')
   else:continue
   self.assertNotEqual(changed,new)
   with self.assertRaises(ValueError):c._projection(name,old,changed)
 def memo_source(self,text):
  tree=ast.parse(text);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Emitter')
  if not any(isinstance(n,ast.FunctionDef) and n.name=='_compute_offsets' for n in klass.body):
   original=c._function(tree,'offsets','Emitter');original.name='_compute_offsets';klass.body.insert(klass.body.index(original),ast.parse(c._OFFSET_WRAPPER).body[0]);c._function(tree,'__init__','Emitter').body.append(ast.parse('self._offset_cache={}').body[0])
  return tree
 def test_pure_instance_offset_memo_projection(self):
  name=c.PREFIX+'backend.py';old,new=self.texts[name];memo=ast.unparse(ast.fix_missing_locations(self.memo_source(new)));proof=c._projection(name,old,memo);self.assertTrue(proof['allowed_boundaries']['offsets_memo']['enabled'])
 def test_original_offset_calculation_change_rejected(self):
  name=c.PREFIX+'backend.py';old,new=self.texts[name];tree=self.memo_source(new);function=c._function(tree,'_compute_offsets','Emitter');values=[n for n in ast.walk(function) if isinstance(n,ast.Constant) and n.value==8];self.assertTrue(values);values[0].value=16
  with self.assertRaises(ValueError):c._projection(name,old,ast.unparse(ast.fix_missing_locations(tree)))
 def test_offset_wrapper_wrong_key_rejected(self):
  name=c.PREFIX+'backend.py';old,new=self.texts[name];tree=self.memo_source(new);wrapper=c._function(tree,'offsets','Emitter');wrapper.body[0].test=ast.parse("'all-values' not in self._offset_cache",mode='eval').body
  with self.assertRaises(ValueError):c._projection(name,old,ast.unparse(ast.fix_missing_locations(tree)))
 def test_offset_cache_nonempty_or_global_rejected(self):
  name=c.PREFIX+'backend.py';old,new=self.texts[name];tree=self.memo_source(new);init=c._function(tree,'__init__','Emitter');empty=c._dump(ast.parse('self._offset_cache={}').body[0]);assignment=next(n for n in init.body if c._dump(n)==empty);assignment.value=ast.parse("{'cached':{}}",mode='eval').body
  with self.assertRaises(ValueError):c._projection(name,old,ast.unparse(ast.fix_missing_locations(tree)))
  tree=self.memo_source(new);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Emitter');klass.body.insert(0,ast.parse('_offset_cache={}').body[0])
  with self.assertRaises(ValueError):c._projection(name,old,ast.unparse(ast.fix_missing_locations(tree)))
 def test_offset_tables_equal_and_emitter_local(self):
  from backend import Emitter
  if not hasattr(Emitter,'_compute_offsets'):self.skipTest('production memo not installed')
  addresses=[dict(view=[i//2,i%2],begin_bit=i*32,end_bit=i*32+32) for i in range(4)];packet=dict(values=[dict(id='storage',storage=dict(addresses=addresses))]);before=copy.deepcopy(packet);first=Emitter(packet);second=Emitter(packet)
  with patch.object(first,'_compute_offsets',wraps=first._compute_offsets) as build:
   a=first.offsets('storage');self.assertIs(a,first.offsets('storage'));self.assertEqual(build.call_count,1)
  self.assertEqual(a,{0:0,1:4,2:8,3:12});self.assertEqual(a,second.offsets('storage'));self.assertIsNot(a,second.offsets('storage'));self.assertEqual(packet,before)
 def test_duplicate_json_keys_rejected(self):
  p=self.directory/'duplicate.json';p.write_text('{"contract":"a","contract":"b"}')
  with self.assertRaises(ValueError):c.load_profile(p)
 def test_profile_coefficient_copy_cannot_replace_sealed_base(self):
  directory=self.directory/'unsealed-fit';directory.mkdir();value=c._read(self.local);value['coefficients']['local_transfer_bytes']+=1;(directory/'profile.json').write_text(json.dumps(value))
  with self.assertRaises((ValueError,FileNotFoundError)):c.create_view(self.directory/'bad.json',directory/'profile.json',self.proof())

 def test_network_helper_dependency_mutation_rejected(self):
  target=(c.ROOT/c._NETWORK_HELPER).resolve();real=c.sha
  with patch.object(c,'sha',side_effect=lambda p:'0'*64 if Path(p).resolve()==target else real(p)):
   with self.assertRaisesRegex(ValueError,'network helper'):self.proof()
 def test_network_anchor_seal_mutation_rejected(self):
  target=c._NETWORK_INDEX.resolve();real=c.sha
  with patch.object(c,'sha',side_effect=lambda p:'0'*64 if Path(p).resolve()==target else real(p)):
   with self.assertRaisesRegex(ValueError,'anchor seal'):self.proof()
 def test_reviewed_network_emitter_mutation_rejected(self):
  name=c.PREFIX+'backend.py';old,new=self.texts[name];tree=ast.parse(new);fn=c._function(tree,'_emit_shuffle_network','Emitter');fn.body.append(ast.parse('self.smem.clear()').body[0])
  with self.assertRaisesRegex(ValueError,'projection input is not pinned live source'):c._projection(name,old,ast.unparse(ast.fix_missing_locations(tree)))
 def test_new_receipts_stay_outside_prior_seal(self):
  self.assertEqual(c.BASE.name,'network_selection_execution_artifacts')
  self.assertFalse(self.directory.resolve().is_relative_to(c._NETWORK_INDEX.parent.resolve()))

if __name__=='__main__':unittest.main()
