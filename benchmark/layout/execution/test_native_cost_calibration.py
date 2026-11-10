import unittest
import numpy as np
from native_cost_calibration import *
class NativeCalibrationTest(unittest.TestCase):
 def native(self):
  names=set(MODEL_FEATURES)-{'shared_transfer_bytes','local_transfer_bytes','symbolic_bank_excess'}
  return dict(status='available',identity=dict(schema_version=1,domain='test',analyzer_sha256='a',nvdisasm=dict(sha256='b',version='v'),cuobjdump=dict(sha256='c',version='v')),
   symbolic_banks=dict(contract='frisk-emitted-affine-shared-bank-v1',implementation_sha256={'test':'source'},shared_read_bank_excess=3,shared_write_bank_excess=2),
   features={**{n:1 for n in names},'shared_load_bytes':4,'shared_store_bytes':4,'local_load_bytes':0,'local_store_bytes':0})
 def profile(self):
  return dict(model='sm90-native-surviving-schedule-v1',analysis_contract=analysis_contract(self.native()),features=list(MODEL_FEATURES),coefficients={n:1 for n in MODEL_FEATURES},training_maxima={n:100 for n in MODEL_FEATURES})
 def test_analysis_identity_is_required(self):
  n=self.native();n['identity']['analyzer_sha256']='changed'
  self.assertEqual(score(n,self.profile())['reason'],'native-analysis-contract-mismatch')
 def test_missing_bank_proof_fallback(self):
  n=self.native();n.pop('symbolic_banks')
  self.assertEqual(score(n,self.profile())['reason'],'symbolic-bank-proof-unavailable')
 def test_signed_pair_nnls(self):
  from calibration_analysis import coordinate_fit
  x=np.array([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]])
  y=x@np.array([3.,5.]);r=coordinate_fit(x,y,ridge=0,signed=True,fit_intercept=False)
  self.assertTrue(r['converged']);self.assertTrue(np.allclose(r['coefficients'],[3.,5.]));self.assertEqual(r['intercept'],0.)
  with self.assertRaises(ValueError):coordinate_fit(x,y)
 def test_tied_features(self):
  f=model_features(self.native());self.assertEqual(f['shared_transfer_bytes'],8);self.assertEqual(f['local_transfer_bytes'],0)
 def test_model_unsupported_is_typed(self):
  self.assertIsNone(score(dict(status='unsupported'),self.profile())['cost'])
 def test_no_zero_price_communication(self):
  p=self.profile();p['coefficients']['shuffle_instructions']=0
  with self.assertRaises(ValueError):score(self.native(),p)
 def test_training_zero_activation_falls_back(self):
  p=self.profile();p['training_maxima']['global_load_bytes']=0
  self.assertEqual(score(self.native(),p)['reason'],'training-zero-feature-activation')
 def test_score_uses_only_static_features(self):
  n=self.native();p=self.profile();s=score(n,p);n['measured_us']=99999
  self.assertEqual(score(n,p)['cost'],s['cost']);self.assertFalse(s['uses_gpu_timing'])
 def test_micro_registered_and_reference(self):
  specs=micro_specs();self.assertEqual(len(specs),16)
  x=np.arange(128,dtype=np.uint32)
  for s in specs:
   self.assertIn('frisk_execution',micro_ptx(s));self.assertEqual(micro_reference(s,x).shape,(128,))
   if s['kind'] in ('shared','shuffle'):self.assertTrue(np.all(micro_reference(s,x)==0))
 def test_micro_reject_unknown(self):
  with self.assertRaises(ValueError):micro_ptx(dict(key='unknown'))
 def test_shared_floor_preserves_paired_covariance(self):
  labels=[];noise=np.linspace(-.001,.001,10);base=np.linspace(2.,2.1,10)
  for s in micro_specs():
   n=s['repeats'];f={k:0 for k in MODEL_FEATURES};f['barrier_instructions']=n*(2 if s['kind']=='shared' else s['kind']=='barrier');f['shuffle_instructions']=n*(s['kind']=='shuffle');f['shared_transfer_bytes']=n*8*(s['kind']=='shared')
   times=base.copy()
   if s['kind']=='barrier':times+=n*(.01+noise)
   if s['kind']=='shared':times+=2*n*(.01+noise)+n*8*.002
   if s['kind']=='shuffle':times+=n*.02
   labels.append(dict(spec=s,features=f,batch_medians_us=times.tolist()))
  result=communication_floors(labels)['shared_transfer_bytes']
  self.assertTrue(np.allclose(result['family_ci95'],[.002,.002]));self.assertAlmostEqual(result['floor_us_per_unit'],.0002)
 def test_nonpositive_marginal_rejected(self):
  names=list(MODEL_FEATURES);labels=[]
  for s in micro_specs():
   f={n:0 for n in names};f['barrier_instructions']=s['repeats']*(2 if s['kind']=='shared' else s['kind']=='barrier');f['shuffle_instructions']=s['repeats']*(s['kind']=='shuffle');f['shared_transfer_bytes']=s['repeats']*8*(s['kind']=='shared')
   labels.append(dict(spec=s,features=f,batch_medians_us=[2.]*10))
  with self.assertRaises(ValueError):communication_floors(labels)
if __name__=='__main__':unittest.main()
