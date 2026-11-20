import unittest,math
import numpy as np
import partial_cost_fixture as f
class PartialFixtureTest(unittest.TestCase):
 def test_all_new_and_anchors(self):
  rows=f.specifications();self.assertTrue(f.validate_specifications(rows));self.assertEqual(sum(x['split']=='holdout' for x in rows),6)
 def test_observed_R1_is_not_fresh(self):
  rows=f.specifications();rows[0]=dict(f.observed.specifications()[0],key='reused',split='holdout')
  with self.assertRaises(ValueError):f.validate_specifications(rows)
 def test_flattened_workloads_are_not_renamed(self):
  rows=f.specifications();rows[0]=dict(rows[0],shape=[32,256])
  with self.assertRaises(ValueError):f.validate_specifications(rows)
 def test_new_references_finite_and_size(self):
  for s in f.specifications():
   if s['split']!='holdout':continue
   q,k,v=f.inputs(s,0);ref=f.reference(s,q,k,v)
   self.assertEqual(ref.size,math.prod(s['shape']));self.assertTrue(np.isfinite(ref).all())
 def test_common_physical_domain(self):
  for s in f.specifications():self.assertEqual(s['threads'],128)
if __name__=='__main__':unittest.main()
