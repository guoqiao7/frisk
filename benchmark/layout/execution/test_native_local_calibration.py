import unittest
import numpy as np
from unittest.mock import patch
import native_local_calibration as local
import native_cost_calibration as base
class LocalCalibrationTest(unittest.TestCase):
 def test_registered_matrix_covers_observed_units(self):
  rows=local.specifications();self.assertEqual(len(rows),22);self.assertEqual(len({r['key'] for r in rows}),22)
  self.assertEqual(max(r['words']*4 for r in rows),16384);self.assertEqual(max(r['repeats']*8 for r in rows),16384)
  for r in rows:
   if r['kind']=='local':self.assertTrue(any(c['kind']=='control' and c['repeats']==r['repeats'] for c in rows))
 def test_reference_has_input_dependence_and_wrap(self):
  for r in local.specifications():
   x=np.array([0,1,0xffffffff,0x80000000],dtype=np.uint32);ref=local.reference(r,x)
   expected=np.array([(int(a)+r['repeats']*local.STEP)%(1<<32) for a in x],dtype=np.uint32)
   self.assertTrue(np.array_equal(ref,expected));self.assertGreater(len(set(ref.tolist())),1)
 def test_independent_private_memory_recurrence(self):
  r=next(s for s in local.specifications() if s['key']=='local-w256-n8');x=np.array([0,13,0xabcdef98,0xffffffff],dtype=np.uint32);expected=[]
  for initial in x:
   a=int(initial);memory={}
   for _ in range(r['repeats']):
    index=a&(r['words']-1);self.assertLess(index,r['words']);memory[index]=a;a=(memory[index]+local.STEP)&0xffffffff
   expected.append(a)
  self.assertTrue(np.array_equal(local.reference(r,x),np.array(expected,dtype=np.uint32)))
 def test_ptx_never_uses_illegal_volatile_local(self):
  for s in local.specifications():
   p=local.ptx(s);self.assertNotIn('.volatile.local',p);self.assertEqual(p.count('bar.sync'),s['repeats'])
   n=s['repeats'] if s['kind']=='local' else 0
   self.assertEqual(p.count('st.local'),n);self.assertEqual(p.count('ld.local'),n)
 def test_unknown_spec_rejected(self):
  with self.assertRaises(ValueError):local.ptx(dict(key='fake'))
 def fake(self,s):
  n=s['repeats'] if s['kind']=='local' else 0
  return dict(status='available',features=dict(local_load_bytes=4*n,local_store_bytes=4*n,shared_load_bytes=0,shared_store_bytes=0,barrier_instructions=s['repeats']))
 def test_exact_native_traffic_and_zero_shared(self):
  s=local.specifications()[7]
  with patch.object(local,'analyze',return_value=self.fake(s)):
   n=local.native_evidence('.',s,None);self.assertEqual(n['symbolic_banks']['shared_read_bank_excess'],0);self.assertEqual(n['symbolic_banks']['shared_write_bank_excess'],0)
 def test_eliminated_local_is_not_calibrated(self):
  s=next(r for r in local.specifications() if r['kind']=='local');n=self.fake(s);n['features']['local_load_bytes']=0
  with patch.object(local,'analyze',return_value=n):
   with self.assertRaises(ValueError):local.native_evidence('.',s,None)
 def test_shared_protocol_not_mislabelled_bank_zero(self):
  s=local.specifications()[0];n=self.fake(s);n['features']['shared_load_bytes']=4
  with patch.object(local,'analyze',return_value=n):
   with self.assertRaises(ValueError):local.native_evidence('.',s,None)
 def synthetic(self,fee=.002):
  rows=[];noise=np.linspace(-.001,.001,10)
  for s in local.specifications():
   n=s['repeats'];bytes_=8*n if s['kind']=='local' else 0
   y=np.ones(10)*2+n*(.01+noise)+bytes_*fee
   rows.append(dict(spec=s,features={'local_transfer_bytes':bytes_},batch_medians_us=y.tolist()))
  return rows
 def test_floor_joint_bootstrap_removes_matched_bar_noise(self):
  f=local.local_floor(self.synthetic());self.assertTrue(np.allclose(f['family_ci95'],[.002,.002]));self.assertAlmostEqual(f['floor_us_per_unit'],.0002)
 def test_nonpositive_increment_does_not_get_fake_positive_fee(self):
  with self.assertRaises(ValueError):local.local_floor(self.synthetic(0.))
 def test_incremental_fit_preserves_thirteen_weights(self):
  x=np.zeros((4,len(local.FEATURES)));column=local.FEATURES.index('local_transfer_bytes');x[:,column]=[-2,-1,1,2]
  lower=np.zeros(len(local.FEATURES));lower[column]=.1;prior={'coefficients':{name:i+100 for i,name in enumerate(local.FEATURES)}}
  weights,fit=local.fit_one_local(x,x[:,column]*3,lower,prior)
  self.assertTrue(fit['converged']);self.assertAlmostEqual(weights['local_transfer_bytes']/1e6,3.1,places=5)
  for name in local.FEATURES:
   if name!='local_transfer_bytes':self.assertEqual(weights[name],prior['coefficients'][name])
 def test_protocol_delta_requires_native_bytes(self):
  rows=self.synthetic();rows[-1]['features']['local_transfer_bytes']-=4
  with self.assertRaises(ValueError):local.local_floor(rows)
if __name__=='__main__':unittest.main()
