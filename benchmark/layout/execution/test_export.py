"""Checked exporter integration gate; failures must preserve existing output."""
import json,subprocess,tempfile,unittest,sys
from pathlib import Path
OPT=Path(sys.argv.pop(1)); ROOT=Path(__file__).resolve().parents[3]
class ExportTests(unittest.TestCase):
 def export(self,source,out):
  return subprocess.run([str(OPT),str(source),'-frisk-export-execution=output='+str(out)],capture_output=True,text=True)
 def test_selected_maps_and_reduction_proof(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);selected=p/'selected.mlir';out=p/'execution.json'
   q=subprocess.run([str(OPT),str(ROOT/'test/Transforms/attention-layout-fixed-abi.mlir'),'--frisk-layout-pipeline','-o',str(selected)],capture_output=True,text=True);self.assertEqual(q.returncode,0,q.stderr)
   q=self.export(selected,out);self.assertEqual(q.returncode,0,q.stderr)
   data=json.loads(out.read_text());self.assertEqual(data['schema_version'],1)
   reductions=[o for o in data['operations'] if o['name']=='frisk.reduce_tensor'];self.assertEqual(len(reductions),2)
   for o in reductions:
    self.assertEqual(len(o['fibers']),64)
    for f in o['fibers']:
     self.assertEqual(len(f['nodes']),127);self.assertEqual(len(f['broadcasts']),4)
   self.assertTrue(any(v.get('storage',{}).get('alignment')==1 for v in data['values']))
   self.assertTrue(any(v.get('distributed',{}).get('replication')==4 for v in data['values']))
 def test_changed_contracts_are_atomic(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);selected=p/'selected.mlir';out=p/'out.json'
   q=subprocess.run([str(OPT),str(ROOT/'test/Transforms/attention-layout-fixed-abi.mlir'),'--frisk-layout-pipeline','-o',str(selected)],capture_output=True,text=True);self.assertEqual(q.returncode,0,q.stderr)
   original=selected.read_text()
   for bad in (original.replace('threads = 128','threads = 64'), original.replace('sm_90a','sm_80'), original.replace('512, 4, 0, 32','512, 0, 0, 32')):
    selected.write_text(bad);out.write_text('preserve');q=self.export(selected,out)
    self.assertNotEqual(q.returncode,0,q.stderr);self.assertEqual(out.read_text(),'preserve')
 def test_unsupported_is_atomic(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);src=p/'unsupported.mlir';out=p/'execution.json';out.write_text('preserve')
   src.write_text('module { func.func @unsupported(%x: f32) -> f32 { %y = math.sqrt %x : f32 return %y : f32 } }')
   q=self.export(src,out);self.assertNotEqual(q.returncode,0);self.assertIn('unsupported execution operation',q.stderr);self.assertIn('stage=export-preflight code=unsupported-operation',q.stderr);self.assertEqual(out.read_text(),'preserve')
 def test_output_create_failure_is_typed_without_partial_file(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);selected=p/'selected.mlir';out=p/'missing'/'execution.json'
   q=subprocess.run([str(OPT),str(ROOT/'test/Transforms/attention-layout-fixed-abi.mlir'),'--frisk-layout-pipeline','-o',str(selected)],capture_output=True,text=True);self.assertEqual(q.returncode,0,q.stderr)
   q=self.export(selected,out);self.assertNotEqual(q.returncode,0);self.assertIn('stage=export-publication code=output-create',q.stderr);self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
