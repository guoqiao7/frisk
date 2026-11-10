import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import archive_efficiency_bench as bench

class EfficiencyBenchTest(unittest.TestCase):
 def command_meta(self,root,name):
  p=root/'timers'/name;p.parent.mkdir(parents=True,exist_ok=True);bench.write(p,dict(exit_code=0,elapsed_ns=1))
  return dict(elapsed_ns=1,command_metadata=str(p),command_metadata_sha256=bench.sha(p))
 def matrix(self,root):
  groups=[];items=[]
  for i in range(6):
   spec=dict(key='case'+str(i),case_id='diamond',shape=[8,8],split='holdout');items.append(dict(spec=spec));modes={}
   for mode in ('old','new'):
    sources=[]
    for arm in (bench.SOURCE_ARMS if mode=='old' else ('candidate-0','candidate-1')):
     d=root/'generation'/mode/spec['key']/arm;d.mkdir(parents=True);bench.write(d/'execution.json',dict(i=i));(d/'emitted.ttgir').write_text('same-'+str(i));bench.write(d/'emitted-config.json',dict(i=i));bench.write(d/'record.json',dict(status='success'))
     sources.append(dict(path=str(d),arm=arm,assignment=spec['key']+arm,generation_hashes={n:bench.sha(d/n) for n in ('execution.json','emitted.ttgir','emitted-config.json','record.json')},selection_probes=0,refinement=0))
    stages={name:[self.command_meta(root,f'{mode}-{i}-{name}-{j}.json') for j in range(n)] for name,n in [('prepare',3 if mode=='old' else 1),('search',6 if mode=='old' else 1),('replay',0 if mode=='old' else 2)]}
    modes[mode]=dict(sources=sources,**stages,generation_work_wall_ns=30 if mode=='old' else 10,source_solves=6 if mode=='old' else 1,probes=0,refinement=0,prepare_wall_ns=len(stages['prepare']),selection_wall_ns=len(stages['search']),replay_wall_ns=len(stages['replay']),cli_wall_ns=sum(len(x) for x in stages.values()))
   groups.append(dict(spec=spec,modes=modes))
  return dict(items=items,budgets=dict(old_per_source_probes=65536),source={},tools={}),dict(status='generated',groups=groups)
 def runner(self,record):
  def run(args,log,**kwargs):
   args=[str(x) for x in args];record.append((args,kwargs))
   if 'old-worker' in args:
    directories=[dict(path=args[-1])];stats=None;outcomes={}
   else:
    cfg=bench.read(args[-1]);directories=cfg['directories'];cache=Path(cfg['cache']);initial=bench.cache_inventory(cache);cache.mkdir(parents=True,exist_ok=True);hits=calls=0;outcomes={}
    for row in directories:
     key=bench.sha(Path(row['path'])/'emitted.ttgir');p=cache/key
     if p.exists():hits+=1;outcomes[row['path']]=(False,True)
     else:p.write_text('compiled');calls+=1;outcomes[row['path']]=(True,False)
    stats=dict(requests=len(directories),compiler_api_calls=calls,exact_cache_hits=hits,compiler_internal_cache_hit=None)
    bench.write(cfg['result'],dict(status='complete',session=stats))
   for row in directories:
    d=Path(row['path']);(d/'kernel.cubin').write_bytes(b'\x7fELFtest');(d/'kernel.ptx').write_text('fake PTX');(d/'kernel.ttgir').write_bytes((d/'emitted.ttgir').read_bytes());(d/'execution_config.json').write_bytes((d/'emitted-config.json').read_bytes());bench.write(d/'compile.json',dict(wall_ns=1,compile_cache=dict(compiler_api_called=outcomes.get(row['path'],(True,False))[0],exact_cache_hit=outcomes.get(row['path'],(True,False))[1])))
   meta=Path(str(log)+'.command.json');meta.parent.mkdir(parents=True,exist_ok=True);bench.write(meta,dict(elapsed_ns=1,exit_code=0))
   return dict(elapsed_ns=1,exit_code=0,command_metadata=str(meta),command_metadata_sha256=bench.sha(meta))
  return run
 def test_old_flags_match_explicit_target_budget_without_archive(self):
  args=bench.source_command('/p',{'case_id':'diamond'},{'profile':None},'LI','/o')
  self.assertIn('--seed-beam-solution=false',args);self.assertFalse(any('--candidate-archive' in str(x) for x in args));self.assertGreaterEqual(bench.TIMEOUT,1200)
 def test_new_archive_cap_and_seed(self):
  args=bench.source_command('/p',{'case_id':'diamond'},{'profile':'/profile'},'J','/o',archive=True)
  self.assertIn('--candidate-archive=4',args);self.assertIn('--seed-beam-solution=true',args)
 def test_all_six_and_work_ledgers_required(self):
  with tempfile.TemporaryDirectory() as t:
   r,m=self.matrix(Path(t));self.assertTrue(bench.check_generated_matrix(r,m));m['groups'].pop()
   with self.assertRaises(ValueError):bench.check_generated_matrix(r,m)
 def test_same_program_cannot_be_substituted(self):
  with tempfile.TemporaryDirectory() as t:
   r,m=self.matrix(Path(t));m['groups'][0]['spec']=dict(m['groups'][0]['spec'],shape=[16,16])
   with self.assertRaises(ValueError):bench.check_generated_matrix(r,m)
 def test_hidden_archive_budget_increase_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   r,m=self.matrix(Path(t));m['groups'][0]['modes']['new']['probes']=65537
   with self.assertRaises(ValueError):bench.check_generated_matrix(r,m)
 def test_timer_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   r,m=self.matrix(Path(t));p=Path(m['groups'][0]['modes']['old']['prepare'][0]['command_metadata']);p.write_text('{}')
   with self.assertRaises(ValueError):bench.check_generated_matrix(r,m)
 def test_exact_generation_input_changes_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);r,m=self.matrix(root);p=Path(m['groups'][0]['modes']['new']['sources'][0]['path'])/'execution.json';p.write_text('{}')
   with self.assertRaises(ValueError):bench.compile_inputs(root,'cold','new',m['groups'])
 def test_cold_warm_are_private_fresh_and_warm_does_not_search(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);r,m=self.matrix(root);bench.write(root/'registration.json',r);bench.write(root/'generation-timing.json',dict(whole_phase_work_wall_ns=250,summed_mode_work_wall_ns=240,shared_outer_wall_ns=10));calls=[];runner=self.runner(calls)
   with patch.object(bench,'verify',return_value=(r,m)):
    cold=bench.compile_phase(root,'cold',runner);warm=bench.compile_phase(root,'warm',runner);result=bench.summarize(root)
   self.assertEqual(cold['old']['stats']['workers_started'],36);self.assertEqual(cold['new']['stats']['workers_started'],1)
   self.assertEqual(cold['new']['stats']['compiler_api_calls'],6);self.assertEqual(cold['new']['stats']['exact_cache_hits'],6)
   self.assertEqual(warm['new']['stats']['compiler_api_calls'],0);self.assertEqual(warm['new']['stats']['exact_cache_hits'],12)
   self.assertEqual(len(calls),74);self.assertTrue(all('--prepare' not in c[0] for c in calls))
   self.assertNotEqual(cold['old']['environment_overrides']['TRITON_CACHE_DIR'],cold['new']['environment_overrides']['TRITON_CACHE_DIR'])
   self.assertEqual(cold['new']['environment_overrides'],warm['new']['environment_overrides'])
   self.assertEqual(result['programs'],6);self.assertEqual(result['search']['old']['generation_work_wall_ns'],180);self.assertEqual(result['search']['new']['generation_work_wall_ns'],60);self.assertEqual(result['shared_generation_timing']['shared_outer_wall_ns'],10);self.assertTrue(result['no_statistical_confidence_interval']);self.assertTrue(result['compilation']['warm']['warm_compile_only'])
 def test_cold_refuses_preexisting_cache(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);r,m=self.matrix(root);(root/'cache/old/triton').mkdir(parents=True)
   with patch.object(bench,'verify',return_value=(r,m)):
    with self.assertRaises(ValueError):bench.compile_phase(root,'cold',self.runner([]))
 def test_warm_requires_complete_own_cold(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);r,m=self.matrix(root)
   with patch.object(bench,'verify',return_value=(r,m)):
    with self.assertRaises(ValueError):bench.compile_phase(root,'warm',self.runner([]))
 def test_cache_inventory_binds_content_not_just_existence(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'cache';a=bench.cache_inventory(p);self.assertFalse(a['exists']);p.mkdir();(p/'entry').write_text('a');b=bench.cache_inventory(p);(p/'entry').write_text('b');c=bench.cache_inventory(p)
   self.assertEqual(b['files'],c['files']);self.assertNotEqual(b['manifest_sha256'],c['manifest_sha256'])
 def test_registered_environment_allows_only_exact_private_cache(self):
  baseline=dict(environment={'TRITON_CACHE_DIR':None,'CUDA_VISIBLE_DEVICES':'1'},effective_knobs={'override':False},triton_cache_invalidating={})
  reg={'compiler':{'base_environment':baseline}};current=copy.deepcopy(baseline);current['environment']['TRITON_CACHE_DIR']='/private'
  self.assertTrue(bench.verify_compiler_environment(reg,'/private',current))
  for kind in ('cache','device','knob'):
   changed=copy.deepcopy(current)
   if kind=='cache':changed['environment']['TRITON_CACHE_DIR']='/other'
   elif kind=='device':changed['environment']['CUDA_VISIBLE_DEVICES']='0'
   else:changed['effective_knobs']['override']=True
   with self.assertRaises(ValueError):bench.verify_compiler_environment(reg,'/private',changed)
 def test_cpu_guard_rejects_context(self):
  import torch
  with patch.object(torch.cuda,'is_initialized',return_value=True):
   with self.assertRaises(ValueError):bench.assert_cpu_only()
class CurrentPreparationModeTest(unittest.TestCase):
 def fixture(self,root,*,rebuild=True):
  models=[dict(name=n,profile=None) for n in ('legacy','v5d','surviving')];old={}
  for m in models:
   p=root/'sealed-source'/m['name'];p.mkdir(parents=True)
   (p/'input.mlir').write_text('same observed algorithm')
   bench.write(p/'domains.json',{'old':True})
   bench.write(p/'prepared.json',dict(common_digest='old',domain_digest='old',coverage_digest='old',package_digest='old-'+m['name'],cost_model=m['name']))
   old[m['name']]=str(p)
  item=dict(spec=dict(key='observed',case_id='diamond'),prepared=old)
  return dict(models=models,items=[item],design=dict(preparation_mode='rebuild-current-same-source' if rebuild else 'proof-matched-sealed',execution_export='minified'),budgets=dict(old_per_source_probes=65536)),dict(status='registered',groups=[])
 def runner(self,*,mismatch=False,input_change=False):
  calls=[]
  def run(args,log,**kw):
   args=[str(x) for x in args];calls.append(args)
   if args[1]=='--prepare':
    p=Path(args[4]);p.mkdir(parents=True)
    (p/'input.mlir').write_text(Path(args[2]).read_text()+(' changed' if input_change else ''))
    bench.write(p/'domains.json',{'current':True})
    domain='mismatch' if mismatch and p.name=='v5d' else 'current'
    bench.write(p/'prepared.json',dict(common_digest=domain,domain_digest=domain,coverage_digest=domain,package_digest='new-'+p.name,cost_model=p.name))
   else:
    options={x.split('=',1)[0]:x.split('=',1)[1] for x in args if '=' in x}
    d=Path(options['--output-dir']);d.mkdir(parents=True);meta=bench.read(Path(options['--prepared'])/'prepared.json');replay='--assignment' in options
    record=dict(status='success',cost=dict(saturated=False,cost=[1]*9),complete_assignment_sha256='a'*64,prepared_digest=meta['package_digest'],cost_model=meta['cost_model'],
      total=dict(probes=1,refinement=0),candidate_archive_search=dict(total=dict(probes=1,refinement=0)),selection_bypassed=replay,budget_contract=dict(used=0,refinement_used=0))
    bench.write(d/'record.json',record);bench.write(d/'execution.json',{'minimal':'fake'})
    if '--candidate-archive' in options:bench.write(d/'candidate-archive.json',dict(entries=[dict(index=0,complete_assignment_sha256='a'*64)]))
   log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);meta=Path(str(log)+'.command.json');bench.write(meta,dict(command=args,elapsed_ns=1,exit_code=0))
   return dict(elapsed_ns=1,exit_code=0,command_metadata=str(meta),command_metadata_sha256=bench.sha(meta))
  return calls,run
 def execute(self,root,reg,m,runner):
  with patch.object(bench,'verify',return_value=(reg,m)),patch('backend.emit',return_value=('same IR',{'static_shared_bytes':0})),patch('demand_repair.check_search'),patch('partial_cost_qualification.check_archive'):
   return bench.generate(root,runner)
 def test_new_mode_searches_both_current_domains_and_minifies_symmetrically(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);reg,m=self.fixture(root);calls,run=self.runner();self.execute(root,reg,m,run)
   selections=[a for a in calls if '--prepare' not in a]
   self.assertEqual(len(selections),8)
   self.assertTrue(all('--minify-execution-json' in a for a in selections))
   self.assertTrue(all(any(x.startswith('--prepared='+str(root/'preparation')) for x in a) for a in selections))
   self.assertEqual({x['current_prepared']['package_digest'] for x in m['groups'][0]['modes']['old']['sources']},{'new-legacy','new-v5d','new-surviving'})
   self.assertEqual(len(m['groups'][0]['modes']['old']['preparation_domain_bridges']),3)
 def test_default_sealed_contract_still_rejects_changed_proof_domain(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);reg,m=self.fixture(root,rebuild=False);_,run=self.runner()
   with self.assertRaisesRegex(ValueError,'changed observed domain'):self.execute(root,reg,m,run)
   self.assertEqual(m['status'],'failed')
 def test_current_models_must_share_domain(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);reg,m=self.fixture(root);_,run=self.runner(mismatch=True)
   with self.assertRaisesRegex(ValueError,'domains differ'):self.execute(root,reg,m,run)
 def test_current_preparation_cannot_change_algorithm_input(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);reg,m=self.fixture(root);_,run=self.runner(input_change=True)
   with self.assertRaisesRegex(ValueError,'algorithm input'):self.execute(root,reg,m,run)
 def test_explicit_minify_is_optional_and_registration_mode_is_typed(self):
  plain=bench.source_command('/p',{'case_id':'diamond'},{'profile':None},'J','/o')
  small=bench.source_command('/p',{'case_id':'diamond'},{'profile':None},'J','/o',minify_execution=True)
  self.assertNotIn('--minify-execution-json',plain);self.assertIn('--minify-execution-json',small)
  with self.assertRaisesRegex(ValueError,'explicit bools'):bench.register('/unused',reprepare_current_domains='yes')

if __name__=='__main__':unittest.main()
