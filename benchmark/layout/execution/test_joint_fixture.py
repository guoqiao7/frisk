import unittest,importlib.util,numpy as np
class JointFixture(unittest.TestCase):
 def test_preregistered_float32_branches(self):
  self.assertIsNotNone(importlib.util.find_spec('joint_fixture'),'preregistered fixture absent')
  from joint_fixture import cases,reference,inputs,branch_reference
  q,k,v=inputs()
  for c in cases():
   out=reference(c['id'],q,k,v);self.assertEqual(out.dtype,np.float32);self.assertTrue(np.isfinite(out).all())
   if c['id']!='no-conflict':
    self.assertTrue(np.any(out!=0))
    for branch in (0,1):self.assertTrue(np.any(branch_reference(c['id'],q,k,v,branch)!=out))
if __name__=='__main__':unittest.main()
