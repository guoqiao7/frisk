import copy
import unittest
import network_selection_fixture as f

class NetworkFixtureTest(unittest.TestCase):
    def test_complete_fresh_and_observed(self):
        rows=f.specifications()
        self.assertEqual((len(rows),sum(s['split']=='holdout' for s in rows)),(21,6))
        self.assertTrue(f.validate_specifications(rows))
    def test_flattened_freshness(self):
        old={f.workload_identity(s) for s in f.historical_specifications()}
        self.assertFalse(old & {f.workload_identity(s) for s in f.fresh_specifications()})
    def test_changed_shape_is_not_new_fixture(self):
        rows=f.specifications();rows[0]['shape']=[256,256]
        with self.assertRaises(ValueError): f.validate_specifications(rows)
    def test_missing_or_substituted_regression_rejected(self):
        rows=f.specifications()
        with self.assertRaises(ValueError): f.validate_specifications(rows[:-1])
        rows[-1]=copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError): f.validate_specifications(rows)
    def test_preceding_holdouts_are_all_observed(self):
        self.assertTrue(all(s['split']!='holdout' for s in f.regression_specifications()))
        self.assertTrue(all(f.workload_identity(s) in {f.workload_identity(x) for x in f.historical_specifications()}
                            for s in f.regression_specifications()))

if __name__=='__main__': unittest.main()
