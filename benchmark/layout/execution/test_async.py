import unittest,json,copy,tempfile,subprocess
from pathlib import Path
import backend
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'guoqiao/optimize_plan/p1_4c_artifacts'
class Async(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'build-m6-release');cls.inputs=Path(cls.tmp.name)
  for path,name in [('cp.async','cp-async'),('tma','tma')]:
   selected=cls.inputs/(path+'.mlir');export=cls.inputs/(path+'.json');opt=str(ROOT/'build-m6-release/bin/frisk-opt')
   subprocess.run([opt,str(ROOT/('test/Transforms/attention-layout-'+name+'-abi.mlir')),'--frisk-layout-pipeline','-o',str(selected)],check=True,capture_output=True)
   subprocess.run([opt,str(selected),'-frisk-export-execution=output='+str(export),'-o','/dev/null'],check=True,capture_output=True)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def data(self,path):return json.loads((self.inputs/(path+'.json')).read_text())
 def test_async_instructions(self):
  for path,instruction in [('cp.async','cp.async.ca.shared.global'),('tma','cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes')]:
   try:text,info=backend.emit(self.data(path))
   except ValueError as e:self.fail('valid async contract rejected: '+str(e))
   self.assertIn(instruction,text)
   self.assertEqual(info['copy_path'],path)
 def test_physical_pointer_abi(self):
  from async_host import physical_abi
  ptx='.visible .entry frisk_execution('+','.join('.param .u64 .ptr .global .align 1 p'+str(i) for i in range(8))+') {}'
  try:params=physical_abi(ptx,3)
  except ValueError as e:self.fail(str(e))
  self.assertEqual(len(params),8)
 def test_host_alignment_rejects_before_cuda(self):
  from async_host import validate_launch
  _,info=backend.emit(self.data('tma'))
  validate_launch(info,[128,256,384,1])
  for ptrs in [[1,256,384,1],[128,257,384,1],[128,256,385,1]]:
   with self.assertRaises(ValueError):validate_launch(info,ptrs)
 def test_host_validation_entrypoint(self):
  self.assertTrue(hasattr(backend,'validate_launch'),'strong pointer preflight missing')
 def test_invalid_contracts_reject(self):
  for path in ['cp.async','tma']:
   for field,value in [('version',2),('guarded',1),('issuer','wrong'),('completion','wrong'),('valid_shape',[16,64])]:
    d=self.data(path);next(o for o in d['operations'] if o['name']=='frisk.copy')['attrs']['frisk.copy_contract'][field]=value
    with self.assertRaises(ValueError):backend.emit(d)
 def test_root_bounds_and_physical_collisions_reject(self):
  for path in ['cp.async','tma']:
   for endpoint in [0,1]:
    d=self.data(path);o=next(o for o in d['operations'] if o['name']=='frisk.copy');v=next(v for v in d['values'] if v['id']==o['operands'][endpoint])
    for x in v['storage']['addresses']:x['begin_bit']+=2048*8;x['end_bit']+=2048*8
    if path=='tma':o['attrs']['frisk.copy_contract']['descriptor'][['source_offset','destination_offset'][endpoint]]+=2048
    with self.assertRaises(ValueError):backend.emit(d)
  d=self.data('cp.async');o=next(o for o in d['operations'] if o['name']=='frisk.copy');v=next(v for v in d['values'] if v['id']==o['operands'][1])
  for x in v['storage']['addresses']:x['begin_bit']%=128;x['end_bit']=x['begin_bit']+16
  with self.assertRaises(ValueError):backend.emit(d)
 def test_root_alignment_is_a_guarantee(self):
  for path in ['cp.async','tma']:
   d=self.data(path);o=next(o for o in d['operations'] if o['name']=='frisk.copy');v=next(v for v in d['values'] if v['id']==o['operands'][0]);root=next(vv for vv in d['values'] if vv['id']==v['storage']['root']);root['storage']['alignment']=1
   # Avoid the global-ABI guard masking a wrong-root guarantee by checking the plan itself.
   from async_copy import plan
   with self.assertRaises(ValueError):plan(o,{vv['id']:vv for vv in d['values']})
 def test_invalid_actual_maps_reject(self):
  for path in ['cp.async','tma']:
   for kind in ['alignment','offset','duplicate']:
    d=self.data(path);o=next(o for o in d['operations'] if o['name']=='frisk.copy');v=next(v for v in d['values'] if v['id']==o['operands'][0]);s=v['storage']
    if kind=='alignment':s['alignment']=1
    elif kind=='offset':s['addresses'][0]['begin_bit']+=8;s['addresses'][0]['end_bit']+=8
    else:s['addresses'][1]=copy.deepcopy(s['addresses'][0])
    with self.assertRaises(ValueError):backend.emit(d)
if __name__=='__main__':unittest.main()
