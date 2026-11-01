import unittest,hashlib,json
import run_joint_ablation as runner
class PairedAnalysis(unittest.TestCase):
 def analyze(self,records):
  self.assertTrue(callable(getattr(runner,'analyze_paired_batches',None)),'paired analysis missing')
  return runner.analyze_paired_batches(records)
 def records(self,ratio=1):
  result=[]
  for b in range(10):
   content=dict(batch=b,input_offset=1,output_offset=2,input_hashes=['x'],reference_sha256='y');pid=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
   for a in ('L-C','L-O','J-C','J-O'):
    samples=[(2 if a=='L-C' else 2/ratio)]*20;result.append(dict(batch=b,arm=a,pair_id=pid,pair_content=content.copy(),input_offset=1,output_offset=2,launch_count=100,graph_elapsed_ms=[v*100 for v in samples],samples_ms=samples,valid=True))
  return result
 def test_equal(self):
  x=self.analyze(self.records());self.assertEqual(x['comparisons']['L-C/J-O']['difference_ms_ci95'],[0.,0.]);self.assertEqual(x['n'],10)
 def test_twice(self):
  x=self.analyze(self.records(2));self.assertEqual(x['comparisons']['L-C/J-O']['ratio_ci95'],[2.,2.])
 def test_mismatch(self):
  x=self.records();x[1]['pair_id']='wrong'
  with self.assertRaises(ValueError):self.analyze(x)
 def test_missing_preserved(self):
  x=self.analyze(self.records()[:-1]);self.assertEqual(x['n'],9);self.assertFalse(x['formal']);self.assertEqual(len(x['invalid_batches']),1)
 def test_replay_not_independent(self):
  x=self.records()
  for r in x:r['samples_ms']*=100;r['graph_elapsed_ms']*=100
  self.assertEqual(self.analyze(x)['n'],10)
 def test_content_tamper(self):
  x=self.records();x[1]['pair_content']['output_offset']=9
  with self.assertRaises(ValueError):self.analyze(x)
 def test_count_mismatch(self):
  x=self.records();x[1]['launch_count']=50;x[1]['samples_ms']=[v/50 for v in x[1]['graph_elapsed_ms']]
  with self.assertRaises(ValueError):self.analyze(x)
 def test_sample_equation(self):
  x=self.records();x[1]['samples_ms'][0]+=1
  with self.assertRaises(ValueError):self.analyze(x)
class OracleSchemaCompatibility(unittest.TestCase):
 def test_complete_geometry_schema_remains_accepted(self):
  import sys
  from pathlib import Path
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'selection'))
  import selection_report
  record=dict(status='success',oracle=dict(status='complete',exhausted=True,fatal=False,geometry_set_kind='all-minimum-cost-geometries-independent-of-stable-producer-winner',optimal_geometries=['x'],best_cost={'cost':[0]*9}))
  self.assertTrue(selection_report.complete(record))
class ToolIdentity(unittest.TestCase):
 def test_tool_change_fails_closed(self):
  import tempfile
  from pathlib import Path
  import joint_measure
  self.assertTrue(callable(getattr(joint_measure,'validate_tool_hashes',None)),'tool hash guard missing')
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'tool';p.write_bytes(b'accepted');h=hashlib.sha256(p.read_bytes()).hexdigest();joint_measure.validate_tool_hashes({str(p):h});p.write_bytes(b'changed')
   with self.assertRaises(ValueError):joint_measure.validate_tool_hashes({str(p):h})
class SolveTiming(unittest.TestCase):
 def test_stage_host_timing_structure(self):
  import tempfile,json,subprocess
  from pathlib import Path
  from joint_fixture import fixture
  root=Path(__file__).resolve().parents[3];bench=root/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench'
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'raw.mlir').write_text(fixture('no-conflict'))
   subprocess.run([str(bench),'--prepare',str(p/'raw.mlir'),'no-conflict',str(p/'prepared')],check=True,capture_output=True)
   subprocess.run([str(bench),'--prepared='+str(p/'prepared'),'--case=no-conflict','--selection=L','--conversion=C','--output-dir='+str(p/'out')],check=True,capture_output=True)
   record=json.loads((p/'out/record.json').read_text());self.assertIn('solve_timing_ns',record)
   t=record['solve_timing_ns'];self.assertGreaterEqual(t['stage1'],0);self.assertGreaterEqual(t['stage2'],0);self.assertGreaterEqual(t['total'],t['stage1']+t['stage2'])
if __name__=='__main__':unittest.main()
