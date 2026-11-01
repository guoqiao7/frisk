"""Gate3 public entry regression tests; no GPU work."""
import json,subprocess,tempfile,unittest
from pathlib import Path
from joint_fixture import fixture
ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench'
class CommonPreparation(unittest.TestCase):
 def test_prepare_and_four_arms_share_domain(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'input.mlir';source.write_text(fixture('no-conflict'))
   prepared=root/'prepared'
   p=subprocess.run([str(BENCH),'--prepare',str(source),'no-conflict',str(prepared)],capture_output=True,text=True)
   self.assertEqual(p.returncode,0,p.stderr)
   identities=[]
   for selection in ('L','J'):
    for conversion in ('C','O'):
     out=root/(selection+conversion)
     p=subprocess.run([str(BENCH),'--prepared='+str(prepared),'--case=no-conflict','--selection='+selection,'--conversion='+conversion,'--output-dir='+str(out),'--oracle'],capture_output=True,text=True)
     self.assertEqual(p.returncode,0,p.stderr)
     record=json.loads((out/'record.json').read_text());identities.append(record['prepared_digest'])
     self.assertEqual(record['status'],'success');self.assertEqual(record['oracle']['status'],'complete')
   self.assertEqual(len(set(identities)),1)
class PreparedIdentity(unittest.TestCase):
 def test_scope_metadata_tamper_is_fatal(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'input.mlir';source.write_text(fixture('no-conflict'));prepared=root/'prepared'
   self.assertEqual(subprocess.run([str(BENCH),'--prepare',str(source),'no-conflict',str(prepared)],capture_output=True).returncode,0)
   metadata=json.loads((prepared/'prepared.json').read_text());metadata['producers']=[];(prepared/'prepared.json').write_text(json.dumps(metadata))
   p=subprocess.run([str(BENCH),'--prepared='+str(prepared),'--case=no-conflict','--selection=J','--conversion=C','--output-dir='+str(root/'out')],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0,'tampered scope metadata was accepted')
class AssignmentReplay(unittest.TestCase):
 def test_replay_keeps_complete_assignment_and_policy_is_checked(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'input.mlir';source.write_text(fixture('diamond'));prepared=root/'prepared'
   self.assertEqual(subprocess.run([str(BENCH),'--prepare',str(source),'diamond',str(prepared)],capture_output=True).returncode,0)
   common=[str(BENCH),'--prepared='+str(prepared),'--case=diamond','--output-dir=']
   original=root/'original';p=subprocess.run(common[:-1]+['--output-dir='+str(original),'--conversion=O'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
   record=json.loads((original/'record.json').read_text())
   for stage in ('independent','shared','remat','post'):
    d=root/stage;p=subprocess.run(common[:-1]+['--output-dir='+str(d),'--conversion=O','--assignment='+str(original/'record.json'),'--replay='+stage],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(json.loads((d/'record.json').read_text())['complete_assignment_sha256'],record['complete_assignment_sha256'])
   p=subprocess.run(common[:-1]+['--output-dir='+str(root/'wrong-policy'),'--conversion=C','--assignment='+str(original/'record.json')],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0,'cross-policy assignment was silently accepted')
class NaturalAttentionStages(unittest.TestCase):
 def test_real_copy_and_mma_l2_support_pins(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=ROOT/'test/Transforms/attention-layout-fixed-abi.mlir';prepared=root/'prepared'
   p=subprocess.run([str(BENCH),'--prepare',str(source),'attention',str(prepared)],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
   p=subprocess.run([str(BENCH),'--prepared='+str(prepared),'--case=attention','--selection=L','--conversion=O','--output-dir='+str(root/'out'),'--oracle'],capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr)
   record=json.loads((root/'out/record.json').read_text());self.assertEqual(record['status'],'success');self.assertEqual(record['stage2_oracle']['status'],'complete');self.assertTrue(record['selection_preserved_input_ir']);self.assertTrue(record['instruction_contracts_verified']);self.assertIsInstance(record['stage1_producer_optimum_tuple'],bool);self.assertGreaterEqual(len(record['instruction_bindings']),5)
   from backend import emit
   text,info=emit(json.loads((root/'out/execution.json').read_text()));self.assertIn('frisk_shared',text)
class ExecutableIdentity(unittest.TestCase):
 def test_debug_paths_are_provenance_and_instructions_are_identity(self):
  from run_joint_ablation import executable_ptx_fingerprint
  base='.version 8.0\n.target sm_90a\n.address_size 64\n.file 1 "A"\n.visible .entry frisk_execution() {\n.loc 1 2 3\n mov.u32 %r1, 1;\n ret;\n}\n'
  self.assertEqual(executable_ptx_fingerprint(base),executable_ptx_fingerprint(base.replace('"A"','"B"').replace('.loc 1 2 3','.loc 1 9 4')))
  self.assertNotEqual(executable_ptx_fingerprint(base),executable_ptx_fingerprint(base.replace('%r1, 1','%r1, 2')))
class FailureProtocol(unittest.TestCase):
 def test_signal_replay_integrity_and_numeric_failure_are_fatal(self):
  import run_joint_ablation as runner
  for phase,code in [('matrix',-11),('replay',2),('numerics',1)]:
   with self.subTest(phase=phase),tempfile.TemporaryDirectory() as temp:
    d=Path(temp)
    with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,code,phase,'record.json')
 def test_documented_budget_is_retained(self):
  import run_joint_ablation as runner
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);(d/'failure.json').write_text(json.dumps(dict(status='failed',fatal=False,failure_kind='search-budget',reason='no witness',limits={'probes':1})))
   outcome=runner.phase_outcome(d,3,'replay','record.json');self.assertFalse(outcome['fatal']);self.assertEqual(outcome['failure_kind'],'search-budget')
 def test_malformed_worker_result_is_fatal(self):
  import run_joint_ablation as runner
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);(d/'record.json').write_text('[]')
   with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,0,'replay','record.json')
class FatalManifest(unittest.TestCase):
 def test_every_phase_publishes_fatal_manifest(self):
  import run_joint_ablation as runner
  for phase,code in [('prepare',2),('matrix',-11),('replay',2),('numerics',1)]:
   with self.subTest(phase=phase),tempfile.TemporaryDirectory() as temp:
    root=Path(temp);manifest={'stage':'validating','matrix':[],'replays':[]}
    with self.assertRaises(runner.FatalPhaseError):runner.phase_or_abort(root,manifest,root,code,phase,'record.json')
    saved=json.loads((root/'manifest.json').read_text());self.assertEqual(saved['stage'],'fatal');self.assertTrue(saved['failure']['fatal']);self.assertEqual(saved['failure']['exit_code'],code)
class WorkerSuccessSchema(unittest.TestCase):
 def test_success_without_assignment_identity_is_fatal(self):
  import run_joint_ablation as runner
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);(d/'record.json').write_text(json.dumps({'status':'success'}))
   with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,0,'replay','record.json')
 def test_numerical_records_require_true_booleans_and_identity(self):
  import run_joint_ablation as runner
  import copy
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);(d/'kernel.cubin').write_bytes(b'fixture');base=dict(cubin_sha256=runner.digest(b'fixture'),gpu_uuid='GPU-test',physical_params=['arg'+str(i) for i in range(5)],physical_resources=dict(shared_bytes=0,local_bytes=0,registers_per_thread=1),case_id='no-conflict',input_hashes=['0'*64]*3,expected_sha256='1'*64,records=[dict(bitexact=True,guards=True,inputs_unchanged=True,offset=1,path='ordinary')])
   for bad in ({},dict(bitexact=True,guards=0,inputs_unchanged=True),dict(bitexact=True,guards='true',inputs_unchanged=True),dict(bitexact=1,guards=True,inputs_unchanged=True)):
    value=copy.deepcopy(base);value['records']=[bad];(d/'correctness.json').write_text(json.dumps(value))
    with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,0,'numerics','correctness.json')
   value=copy.deepcopy(base);del value['gpu_uuid'];(d/'correctness.json').write_text(json.dumps(value))
   with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,0,'numerics','correctness.json')
 def test_parent_timeout_is_retained_only_for_selection(self):
  import run_joint_ablation as runner
  import sys
  with tempfile.TemporaryDirectory() as temp:
   d=Path(temp);log=d/'worker.log';code=runner.command([sys.executable,'-c','import time;time.sleep(10)'],log,timeout_seconds=.05);metadata=json.loads(log.with_suffix('.log.command.json').read_text())
   outcome=runner.phase_outcome(d,code,'matrix','record.json',command_metadata=metadata);self.assertEqual(outcome['failure_kind'],'timeout');self.assertFalse(outcome['fatal'])
   for phase in ('numerics','safety'):
    with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,code,phase,'record.json',command_metadata=metadata)
   code=runner.command([sys.executable,'-c','raise SystemExit(124)'],log,timeout_seconds=2);metadata=json.loads(log.with_suffix('.log.command.json').read_text())
   with self.assertRaises(runner.FatalPhaseError):runner.phase_outcome(d,code,'matrix','record.json',command_metadata=metadata)
class ImmutableValidation(unittest.TestCase):
 def test_revalidating_frozen_run_preserves_original_manifest(self):
  import run_joint_ablation as runner
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'accepted';root.mkdir();path=root/'manifest.json';path.write_text(json.dumps(dict(stage='validated',identity=runner.source_identity())));before=path.read_bytes()
   with self.assertRaises(runner.FatalPhaseError):runner.validate(root)
   self.assertEqual(path.read_bytes(),before)
class NestedSchema(unittest.TestCase):
 def test_assignment_nested_missing_wrongtype_duplicate_and_range_are_typed_fatal(self):
  import copy
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'input.mlir';source.write_text(fixture('no-conflict'));prepared=root/'prepared';self.assertEqual(subprocess.run([str(BENCH),'--prepare',str(source),'no-conflict',str(prepared)],capture_output=True).returncode,0)
   original=root/'original';base=[str(BENCH),'--prepared='+str(prepared),'--case=no-conflict','--conversion=C'];self.assertEqual(subprocess.run(base+['--output-dir='+str(original)],capture_output=True).returncode,0);record=json.loads((original/'record.json').read_text())
   for index,kind in enumerate(('missing','type','duplicate','range')):
    modified=copy.deepcopy(record)
    if kind=='missing':del modified['assignment'][0]['candidate']
    elif kind=='type':modified['assignment'][0]['variable']='bad'
    elif kind=='duplicate':modified['assignment'][1]['variable']=modified['assignment'][0]['variable']
    else:modified['assignment'][0]['variable']=999999
    modified['complete_assignment_sha256']=__import__('hashlib').sha256(json.dumps({key:modified[key] for key in ('assignment','instruction_bindings','reduction_bindings')},indent=2,sort_keys=True).encode()).hexdigest();path=root/(kind+'.json');path.write_text(json.dumps(modified));out=root/('out'+str(index));p=subprocess.run(base+['--output-dir='+str(out),'--assignment='+str(path)],capture_output=True,text=True);self.assertEqual(p.returncode,2,(kind,p.stderr));failure=json.loads((out/'failure.json').read_text());self.assertTrue(failure['fatal'])
class StrongOffsets(unittest.TestCase):
 def test_input_and_output_alignment_are_independent(self):
  from run_joint_ablation import attention_offsets
  for path in ('cp.async','tma'):
   offsets=attention_offsets(path);self.assertEqual(set(offsets),{(q,o) for q in (0,128,256) for o in (1,2,3)});self.assertTrue(all(q%128==0 for q,o in offsets))
class AttentionReference(unittest.TestCase):
 def test_case_protocol_and_constant(self):
  import run_joint_ablation as runner
  import numpy as np
  cases=runner.attention_inputs();self.assertEqual(len(cases),15)
  name,q,k,v=next(x for x in cases if x[0]=='constant')
  np.testing.assert_array_equal(runner.attention_reference(q,k,v),np.ones((64,16),np.float32))
if __name__=='__main__':unittest.main()

