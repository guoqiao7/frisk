import unittest
from surviving_cost_calibration import mark_execution_domain,FEATURES,FIXED_ZERO
class SurvivingCalibrationTest(unittest.TestCase):
 def test_existing_spaced_domain_kept(self):
  text='module attributes {frisk.execution_domain = "sm90-scalar-f32-dag-v1"} {}'
  self.assertEqual(mark_execution_domain(text,{'case_id':'diamond'}),text)
 def test_new_common_domain(self):
  text=mark_execution_domain('module attributes {} {}',{'case_id':'diamond'})
  self.assertEqual(text.count('frisk.execution_domain'),1)
 def test_twenty_feature_contract_and_zero_duplicate_bytes(self):
  self.assertEqual(len(FEATURES),20);self.assertEqual(len(set(FEATURES)),20)
  self.assertNotIn('global_load_issue_groups',FEATURES)
  self.assertEqual(FEATURES[-3:],('shared_warp_issues','shuffle_warp_issues','communication_alu_warp_issues'))
  self.assertIn('conversion_shared_bytes',FIXED_ZERO)
if __name__=='__main__':unittest.main()
