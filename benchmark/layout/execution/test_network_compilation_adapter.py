import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import network_selection_qualification as nq

CFG=dict(contract='fixed-ttgir-four-worker-batch-v1',mode='parallel',workers=4,
    cpus=[8,9,10,11],threads_per_worker=1,
    partition='exact-generated-ttgir-sha256-modulo-workers-v1',compiler_input_reuse=True)

class NetworkCompilationAdapterTest(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.calls=[]
 def tearDown(self):self.temp.cleanup()
 def run_dir(self,name='run',configuration=None):
  run=self.root/name;run.mkdir();cfg={'compilation_execution':copy.deepcopy(CFG)} if configuration is None else configuration
  protocol=dict(configuration=cfg)
  (run/'protocol.json').write_text(json.dumps(protocol));(run/'registration.json').write_text(json.dumps(dict(source={'frozen':'source'},tools={'frozen':'tools'})))
  (run/'manifest.json').write_text(json.dumps(dict(status='generated',rows=[dict(path=str(run/'a')),dict(path=str(run/'b'))])))
  return run,protocol
 def result(self):return dict(contract=CFG['contract'],status='complete',provider_mode='production',affinity=[8,9,10,11],
     request_cap=2,attempted_requests=2,succeeded_requests=2,failed_requests=0,not_attempted_requests=0,
     descendants_reaped=True,compiler_api_calls=1,group_count=1)
 def provider(self,**changes):
  def compile(directories,cache_dir,**kwargs):
   self.calls.append((directories,cache_dir,kwargs));return dict(self.result(),**changes)
  return types.SimpleNamespace(compile_many_parallel=compile)
 def test_schema_exact_opt_in_and_missing_serial(self):
  self.assertIsNone(nq.validate_compilation_execution({}))
  self.assertEqual(nq.validate_compilation_execution({'compilation_execution':CFG}),CFG)
  for key,value in (('workers',8),('workers',4.0),('cpus',[8,9,10,True]),('cpus',[4,5,6,7]),
                    ('threads_per_worker',True),('compiler_input_reuse',1),('partition','round-robin'),('mode','serial')):
   mutated=copy.deepcopy(CFG);mutated[key]=value
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):nq.validate_compilation_execution({'compilation_execution':mutated})
  with self.assertRaises(ValueError):nq.validate_compilation_execution({'compilation_execution':None})
  with self.assertRaises(ValueError):nq.validate_compilation_execution({'compilation_execution':dict(CFG,extra=True)})
 def test_init_rejects_invalid_before_registering(self):
  config=dict(compilation_execution=dict(CFG,workers=8))
  with patch.object(nq.engine,'read',return_value=config),patch.object(nq.engine,'init') as init,patch('predicate_cost_recalibration.validate_configuration') as model:
   with self.assertRaises(ValueError):nq.init(self.root/'run',self.root/'cfg.json')
   init.assert_not_called();model.assert_not_called()
 def test_default_missing_field_calls_unchanged_serial_engine(self):
  run,protocol=self.run_dir(configuration={});cache=self.root/'cache'
  with patch.object(nq.engine,'compile_batch',return_value='legacy-result') as compile,patch.object(nq.engine,'verify') as verify:
   self.assertEqual(nq.compile_batch(run,cache),'legacy-result');compile.assert_called_once_with(run,cache_dir=cache);verify.assert_not_called()
 def test_parallel_keeps_registration_full_result_manifest_hash_and_verify(self):
  run,protocol=self.run_dir();cache=self.root/'cache';old=(nq.engine.suite,nq.engine.BASE,nq.engine.SEALED)
  with patch.object(nq.engine,'verify',return_value=protocol) as verify,patch.dict(sys.modules,{'parallel_compile_batch':self.provider()}):
   result=nq.compile_batch(run,cache);self.assertEqual(result,self.result());self.assertEqual(verify.call_count,2)
  self.assertEqual((nq.engine.suite,nq.engine.BASE,nq.engine.SEALED),old)
  directories,got_cache,kwargs=self.calls[0];self.assertEqual(got_cache,cache);self.assertEqual(directories,[run/'a',run/'b'])
  self.assertEqual(kwargs['cpus'],(8,9,10,11));self.assertIs(kwargs['compiler_input_reuse'],True)
  self.assertEqual(kwargs['source'],{'frozen':'source'});self.assertEqual(kwargs['tools'],{'frozen':'tools'})
  self.assertEqual(kwargs['output'],run/'parallel-native-compilation')
  batch=json.loads((run/'batch-compilation.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
  self.assertEqual(batch['status'],'complete');self.assertEqual(batch['compilation_execution'],CFG)
  self.assertTrue(manifest['compiled']);self.assertEqual(manifest['batch_compilation_sha256'],nq.engine.sha(run/'batch-compilation.json'))
 def test_freshness_gates_before_dispatch(self):
  for i,change in enumerate((dict(status='registered'),dict(compiled=True))):
   run,protocol=self.run_dir('fresh'+str(i));m=json.loads((run/'manifest.json').read_text());m.update(change);(run/'manifest.json').write_text(json.dumps(m))
   with patch.object(nq.engine,'verify',return_value=protocol),patch.dict(sys.modules,{'parallel_compile_batch':self.provider()}):
    with self.assertRaises(ValueError):nq.compile_batch(run)
  self.assertFalse(self.calls)
 def test_incomplete_injected_or_overbudget_batch_cannot_mark_compiled(self):
  for i,change in enumerate((dict(status='failed'),dict(provider_mode='injected-cpu-test'),dict(attempted_requests=1),
      dict(request_cap=True),dict(compiler_api_calls=2),dict(descendants_reaped=False),dict(affinity=[4,5,6,7]))):
   run,protocol=self.run_dir('bad'+str(i))
   with patch.object(nq.engine,'verify',return_value=protocol),patch.dict(sys.modules,{'parallel_compile_batch':self.provider(**change)}):
    with self.assertRaises(ValueError):nq.compile_batch(run)
    self.assertFalse(json.loads((run/'manifest.json').read_text()).get('compiled',False))
    self.assertEqual(json.loads((run/'batch-compilation.json').read_text())['status'],'failed')
    before=len(self.calls)
    with self.assertRaises(FileExistsError):nq.compile_batch(run)
    self.assertEqual(len(self.calls),before)
 def test_real_exception_is_recorded_and_no_compiled_flag(self):
  run,protocol=self.run_dir()
  def failed(*args,**kwargs):raise RuntimeError('real injected failure')
  with patch.object(nq.engine,'verify',return_value=protocol),patch.dict(sys.modules,{'parallel_compile_batch':types.SimpleNamespace(compile_many_parallel=failed)}):
   with self.assertRaisesRegex(RuntimeError,'real injected failure'):nq.compile_batch(run)
  batch=json.loads((run/'batch-compilation.json').read_text());self.assertEqual(batch['status'],'failed');self.assertIn('RuntimeError',batch['error']);self.assertGreater(batch['wall_ns'],0)
  self.assertFalse(json.loads((run/'manifest.json').read_text()).get('compiled',False))
 def test_dispatch_cannot_replace_verified_registration(self):
  run,protocol=self.run_dir();mutated=copy.deepcopy(protocol);mutated['configuration']['compilation_execution']['workers']=8
  with patch.object(nq.engine,'verify',return_value=mutated),patch.dict(sys.modules,{'parallel_compile_batch':self.provider()}):
   with self.assertRaises(ValueError):nq.compile_batch(run)
  self.assertFalse(self.calls)

if __name__=='__main__':unittest.main()
