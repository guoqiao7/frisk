import unittest
from copy import deepcopy
import native_cost_fixture as f

class FixtureTests(unittest.TestCase):
    def test_fresh_shapes(self):
        s=f.specifications();self.assertEqual([x['shape'][0]*x['shape'][1] for x in s[:6]],[8192,8192,8192,16384,8192,4096]);self.assertTrue(f.validate_specifications(s))
    def test_previous_r4_is_observed(self):
        s=f.specifications();s[0]=dict(f.previous.specifications()[0],key=s[0]['key']);
        with self.assertRaisesRegex(ValueError,'observed'):f.validate_specifications(s)
    def test_equal_N_reshape_rejected(self):
        s=f.specifications();s[0]['shape']=[64,64]
        with self.assertRaisesRegex(ValueError,'observed'):f.validate_specifications(s)
    def test_fresh_equal_N_duplication_rejected(self):
        s=f.specifications();s[1].update(case_id='diamond',shape=[128,64])
        with self.assertRaisesRegex(ValueError,'duplicate flattened'):f.validate_specifications(s)
    def test_tail_is_identity(self):
        s=f.specifications()[0];self.assertNotEqual(f.workload_identity(s),f.workload_identity(dict(s,tail='exp1')))
    def test_sources_six(self):
        m=[dict(name=n,id=n) for n in f.MODEL_NAMES];a=f.arms(m);self.assertEqual(len(a),6);self.assertTrue(all(not x['incumbent_witness'] for x in a if x['selection']=='LI'))
    def test_fallback_declared(self):
        s=f.specifications();self.assertEqual(sum(f.planned_native_fallback(x) is None for x in s),4)
    def test_complete_protocol(self):
        m=[dict(name=n,id=n) for n in f.MODEL_NAMES];p=f.protocol(models=m,native=dict(id='native-v1',profile='/profile',analysis_version='v1'),profiles=[],training_inputs={'/known':'a'*64},preserved_evidence=[])
        self.assertEqual(p['budget']['portfolio_total_probes'],393216);self.assertEqual(p['acceptance']['promotion']['guard_comparisons'],9);self.assertEqual(len(p['experiment']['sources']),6)
        self.assertIn('all six',p['experiment']['primary']);self.assertIn('not evidence',p['experiment']['attribution'])
    def test_protocol_requires_identity(self):
        m=[dict(name=n,id=n) for n in f.MODEL_NAMES]
        with self.assertRaisesRegex(ValueError,'native'):f.protocol(models=m,native={},profiles=[],training_inputs={'/known':'a'*64},preserved_evidence=[])

if __name__=='__main__':unittest.main()
