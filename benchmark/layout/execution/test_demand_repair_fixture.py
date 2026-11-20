import copy
from pathlib import Path
import unittest
import numpy as np
import demand_repair_fixture as fixture

ROOT=Path(__file__).resolve().parents[3]
MODELS=[{'name':'legacy','id':'sm90-layout-lex9-legacy-v2'},
        {'name':'demand','id':'sm90-layout-demand-test-v1'}]
PARAMETERS=dict(models=MODELS,profiles=[{'path':'frozen/profile.json','sha256':'1'*64}],
                feature_contract={'version':'test-demand-v1'},search_contract={'version':'test-search-v1'})

class DemandRepairFixtureTest(unittest.TestCase):
    def test_full_prior_inventory_includes_previous_six_holdouts(self):
        old=fixture.historical_specifications();self.assertEqual(len(old),29)
        known={fixture.program_identity(s) for s in old}
        self.assertTrue(all(fixture.program_identity(s) in known for s in fixture.development_specifications()))
        self.assertEqual(len(fixture.development_specifications()),6)
    def test_matrix_is_six_new_programs_and_three_observed_anchors(self):
        rows=fixture.specifications();self.assertEqual(len(rows),9)
        self.assertEqual(sum(s['split']=='holdout' for s in rows),6)
        self.assertEqual(sum(s['split']=='observed-anchor' for s in rows),3)
        self.assertEqual(sum(16 if s['case_id']!='attention' else 6 for s in rows),114)
    def test_new_programs_disjoint_from_training_and_development(self):
        held={fixture.program_identity(s) for s in fixture.specifications() if s['split']=='holdout'}
        old={fixture.program_identity(s) for s in fixture.historical_specifications()+fixture.training_specifications()+fixture.development_specifications()}
        self.assertFalse(held&old)
    def test_old_holdout_cannot_be_relabelled_fresh(self):
        row=dict(fixture.development_specifications()[0],key='new-key',split='holdout')
        with self.assertRaisesRegex(ValueError,'cannot be a new holdout'):fixture.validate_specifications([row])
    def test_equal_flattened_observed_workload_cannot_be_relabelled(self):
        # This exact shape is absent from old inventory, but diamond N1024 is known.
        row=dict(fixture.specifications()[0],shape=[128,8],key='different-shape')
        self.assertNotIn(fixture.program_identity(row),{fixture.program_identity(s) for s in fixture.historical_specifications()})
        with self.assertRaisesRegex(ValueError,'flattened workload'):fixture.validate_specifications([row])
    def test_all_new_pointwise_workload_sizes_unseen_within_family(self):
        held=[s for s in fixture.specifications() if s['split']=='holdout']
        self.assertEqual([int(np.prod(s['shape'])) for s in held],[4096,4096,4096,8192,1024,1024])
        old={fixture.workload_identity(s) for s in fixture.historical_specifications()}
        self.assertTrue(all(fixture.workload_identity(s) not in old for s in held))
    def test_duplicate_identity_rejected_even_different_key(self):
        row=fixture.specifications()[0]
        with self.assertRaisesRegex(ValueError,'duplicate'):fixture.validate_specifications([row,dict(row,key='different')])
    def test_natural_and_controlled_sources_keep_algorithm_and_shape(self):
        for s in fixture.specifications()[:6]:
            natural=fixture.source(s,ROOT);controlled=fixture.source(s,ROOT,True)
            self.assertNotIn('joint.A',natural);self.assertIn('joint.A',controlled)
            self.assertIn('tensor<'+str(s['shape'][0])+'x'+str(s['shape'][1])+'xf32>',natural)
            self.assertIn('frisk.tile_store',natural)
    def test_execution_domain_is_common_cost_independent_and_f32_only(self):
        for spec in fixture.specifications()[:6]:
            for controlled in (False,True):
                self.assertIn('frisk.execution_domain="sm90-scalar-f32-dag-v1"',fixture.source(spec,ROOT,controlled))
        for spec in fixture.specifications()[6:]:
            self.assertNotIn('frisk.execution_domain',fixture.source(spec,ROOT))
    def test_attention_is_observed_and_natural_only(self):
        for s in fixture.specifications()[6:]:
            self.assertIn('frisk.mma',fixture.source(s,ROOT))
            with self.assertRaises(ValueError):fixture.source(s,ROOT,True)
    def test_all_sixty_new_timing_inputs_have_independent_finite_reference(self):
        for s in fixture.specifications()[:6]:
            for i in range(10):
                q,k,v=fixture.inputs(s,i);ref=fixture.reference(s,q,k,v)
                self.assertEqual(ref.dtype,np.float32);self.assertEqual(ref.size,np.prod(s['shape']))
                self.assertTrue(np.isfinite(ref).all());self.assertTrue(fixture.check_numerics(s,ref,ref)['passed'])
    def test_exact_numerics_reject_signed_zero_change(self):
        s=fixture.specifications()[0];a=np.zeros(1,np.float32);b=-a
        self.assertFalse(fixture.check_numerics(s,a,b)['passed'])
    def test_transcendentals_keep_frozen_tolerance(self):
        s=fixture.specifications()[3];a=np.ones(1,np.float32)
        self.assertTrue(fixture.check_numerics(s,a+np.float32(1e-6),a)['passed'])
        self.assertFalse(fixture.check_numerics(s,a+np.float32(.01),a)['passed'])
    def test_protocol_requires_explicit_identity_contracts(self):
        with self.assertRaises(TypeError):fixture.protocol()
        for field in ['models','profiles','feature_contract','search_contract']:
            kw=copy.deepcopy(PARAMETERS);kw[field]=None
            with self.assertRaises(ValueError):fixture.protocol(**kw)
        kw=copy.deepcopy(PARAMETERS);kw['models'][1]['id']='TBD'
        with self.assertRaises(ValueError):fixture.protocol(**kw)
    def test_protocol_rejects_unhashed_profiles(self):
        for value in ['bad','F'*64,12]:
            kw=copy.deepcopy(PARAMETERS);kw['profiles'][0]['sha256']=value
            with self.assertRaises(ValueError):fixture.protocol(**kw)
    def test_model_search_factorial_arms_are_orthogonal(self):
        spec=fixture.specifications()[0]
        controlled=fixture.arms(spec,'controlled',MODELS);natural=fixture.arms(spec,'natural',MODELS)
        self.assertEqual(len(controlled),10);self.assertEqual(len(natural),6)
        self.assertEqual(len({a['arm'] for a in controlled}),10)
        for model in ['legacy','demand']:
            rows={a['search']:a for a in natural if a['model']==model}
            self.assertFalse(rows['local']['incumbent_witness'])
            self.assertFalse(rows['oldbeam']['incumbent_witness'])
            self.assertTrue(rows['witness']['incumbent_witness'])
        self.assertTrue(all(not a['incumbent_witness'] for a in controlled if a['force_producer']))
        with self.assertRaises(ValueError):fixture.arms(fixture.specifications()[-1],'controlled',MODELS)
    def test_predeclared_budgets_and_thresholds_not_relaxed(self):
        p=fixture.protocol(**PARAMETERS)
        self.assertEqual(p['budget']['max_expanded_states'],65536)
        self.assertEqual(p['budget']['max_refinement_probes'],256)
        self.assertEqual(p['measurement']['max_nodes'],4096)
        self.assertEqual(p['acceptance']['bounded_selection_quality']['geomean_regret_max'],1.03)
        self.assertEqual(p['acceptance']['bounded_selection_quality']['max_case_regret_max'],1.10)
        self.assertEqual(p['acceptance']['promotion']['geomean_time_ratio_upper_lt'],1/1.03)
    def test_protocol_deterministic_without_mutating_callers(self):
        before=copy.deepcopy(PARAMETERS);a=fixture.protocol(**PARAMETERS);b=fixture.protocol(**PARAMETERS)
        self.assertEqual(fixture.digest(a),fixture.digest(b));self.assertEqual(PARAMETERS,before)
if __name__=='__main__':unittest.main()
