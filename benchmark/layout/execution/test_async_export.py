import unittest, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
class StrongExport(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build-m6-release');cls.inputs=Path(cls.tmp.name)
  for path,name in [('cp.async','cp-async'),('tma','tma')]:
   subprocess.run([str(ROOT/'build-m6-release/bin/frisk-opt'),str(ROOT/('test/Transforms/attention-layout-'+name+'-abi.mlir')),'--frisk-layout-pipeline','-o',str(cls.inputs/(path+'.selected.mlir'))],check=True,capture_output=True)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()

 def test_actual_strong_contract_export(self):
  for path in ['cp.async','tma']:
   with tempfile.TemporaryDirectory(dir=self.inputs) as tmp:
    p=subprocess.run([str(ROOT/'build-m6-release/bin/frisk-opt'),str(self.inputs/(path+'.selected.mlir')),'-frisk-export-execution=output='+tmp+'/out.json','-o','/dev/null'],capture_output=True,text=True)
    self.assertEqual(p.returncode,0,p.stderr[:500])
 def test_scope_and_missing_root_precondition_reject_atomically(self):
  import re
  source=(self.inputs/'cp.async.selected.mlir').read_text()
  lines=source.splitlines();decl=[l for l in lines if 'frisk.storage_contract %arg0 ' in l];self.assertTrue(decl)
  base=[l for l in lines if l not in decl]
  variants={'missing':base,'late':base[:-4]+[decl[0]]+base[-4:]}
  incompatible=source.replace('map = #map>, memory_space = #frisk<memory_space Global>, alignment = 128','map = #map1>, memory_space = #frisk<memory_space Global>, alignment = 128',1)
  variants['incompatible']=incompatible.splitlines()
  for kind,content in variants.items():
   with tempfile.TemporaryDirectory(dir=self.inputs) as tmp:
    inp=Path(tmp)/'input.mlir';out=Path(tmp)/'out.json';inp.write_text('\n'.join(content));out.write_text('preserve')
    p=subprocess.run([str(ROOT/'build-m6-release/bin/frisk-opt'),str(inp),'-frisk-export-execution=output='+str(out),'-o','/dev/null'],capture_output=True,text=True)
    self.assertNotEqual(p.returncode,0,kind);self.assertEqual(out.read_text(),'preserve')
if __name__=='__main__':unittest.main()
