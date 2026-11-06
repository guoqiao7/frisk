import copy
import unittest
import tempfile
from pathlib import Path
import demand_repair as driver

class DemandRepairDriverTest(unittest.TestCase):
    def test_feature_schema_is_eighteen_nonnegative_counts(self):
        x=driver.features({'cost':{'features':{'access_warp_issues':3}}})
        self.assertEqual(len(x),18);self.assertEqual(x['access_warp_issues'],3)
        self.assertEqual(x['shared_read_bank_excess'],0)
        for value in [{'unknown':1},{'access_warp_issues':-1},{'access_warp_issues':True}]:
            with self.assertRaises(ValueError):driver.features({'cost':{'features':value}})
    def test_complete_assignment_bridge_only_changes_package_identity(self):
        source={'assignment':[{'variable':1}],'instruction_bindings':[],'reduction_bindings':[],
                'conversion_policy':'independent','complete_assignment_sha256':'a'*64,'cost':{'cost':[1]},'prepared_digest':'old'}
        before=copy.deepcopy(source);result=driver.bridge_assignment(source,{'package_digest':'new'})
        self.assertEqual(set(result),{'assignment','instruction_bindings','reduction_bindings','conversion_policy','complete_assignment_sha256','prepared_digest'})
        self.assertEqual(result['prepared_digest'],'new');self.assertEqual(result['complete_assignment_sha256'],source['complete_assignment_sha256'])
        self.assertEqual(source,before)
        with self.assertRaises(ValueError):driver.bridge_assignment(dict(source,conversion_policy='full'),{'package_digest':'new'})
    def test_f32_domain_tag_is_idempotent_and_attention_is_not_tagged(self):
        source='module attributes {foo=true} {}';f32={'case_id':'diamond'};attention={'case_id':'attention'}
        value=driver.mark_execution_domain(source,f32)
        self.assertIn('frisk.execution_domain="sm90-scalar-f32-dag-v1"',value)
        self.assertEqual(driver.mark_execution_domain(value,f32),value)
        self.assertEqual(driver.mark_execution_domain(source,attention),source)
    def test_requested_search_flag_must_equal_native_record(self):
        record={'seed_beam_solution':False,'legal_return':True,'total':{'probes':12,'refinement':1},
                'budget_contract':{'kind':'shared-total-search-probes-v1','search_performed':True,'limit':65536,'refinement_limit':256,'used':12,'refinement_used':1},
                'search_components':[]}
        driver.check_search(record,{'incumbent_witness':False,'selection':'J'})
        with self.assertRaisesRegex(ValueError,'seed policy'):driver.check_search(record,{'incumbent_witness':True,'selection':'J'})
    def test_budget_record_cannot_be_missing_negative_bool_or_over_cap(self):
        base={'seed_beam_solution':True,'legal_return':True,'total':{'probes':1,'refinement':1}}
        for counters in [{'probes':-1,'refinement':0},{'probes':True,'refinement':0},{'probes':65537,'refinement':0},{'probes':1,'refinement':257},{}]:
            with self.assertRaises(ValueError):driver.check_search(dict(base,total=counters),{'incumbent_witness':True})
    def test_incumbent_certificate_and_seed_quota_are_enforced(self):
        base={'seed_beam_solution':True,'legal_return':True,'total':{'probes':12,'refinement':1},
              'budget_contract':{'kind':'shared-total-search-probes-v1','search_performed':True,'limit':65536,'refinement_limit':256,'used':12,'refinement_used':1},
              'search_components':[{'seed_probes':1,'seed_scores':1,'has_seed_witness':True,
                  'seed_start_cost':{'cost':[2]*9,'saturated':False},'selected_component_cost':{'cost':[1]*9,'saturated':False},'exhaustive':False,'beam_budget_fallback':False}]}
        driver.check_search(base,{'incumbent_witness':True,'selection':'J'})
        bad=copy.deepcopy(base);bad['search_components'][0]['selected_component_cost']['cost']=[3]*9
        with self.assertRaisesRegex(ValueError,'incumbent was lost'):driver.check_search(bad,{'incumbent_witness':True,'selection':'J'})
        bad=copy.deepcopy(base);bad['search_components'][0]['seed_probes']=8193
        with self.assertRaisesRegex(ValueError,'seed work'):driver.check_search(bad,{'incumbent_witness':True,'selection':'J'})
        bad=copy.deepcopy(base);bad['budget_contract']['used']=13
        with self.assertRaisesRegex(ValueError,'ledger disagrees'):driver.check_search(bad,{'incumbent_witness':True,'selection':'J'})
    def test_fitted_profile_collection_and_source_bindings_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);training=root/'training';fit=root/'fit';training.mkdir();fit.mkdir()
            for filename in ('profile.json','fit.json','training-ledger.json'):driver.write(fit/filename,{})
            registration=dict(source=driver.source_identity(),inputs={},tools={'hashes':{}},no_holdout_use=True)
            driver.write(training/'registration.json',registration)
            manifest=dict(status='complete',no_holdout_use=True,rows=[{}]*34,registration_sha256=driver.sha(training/'registration.json'))
            driver.write(training/'manifest.json',manifest)
            inputs=dict(training_directory=str(training),no_holdout_use=True,
                        training_manifest_sha256=driver.sha(training/'manifest.json'),registration_sha256=driver.sha(training/'registration.json'),
                        profile_sha256=driver.sha(fit/'profile.json'),fit_sha256=driver.sha(fit/'fit.json'),training_ledger_sha256=driver.sha(fit/'training-ledger.json'))
            driver.write(fit/'inputs.json',inputs)
            self.assertEqual(driver.verify_fitted_training(fit),training)
            driver.write(fit/'profile.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'fitted training file changed'):driver.verify_fitted_training(fit)
            driver.write(fit/'profile.json',{})
            registration['source']={'stale':True};driver.write(training/'registration.json',registration)
            manifest['registration_sha256']=driver.sha(training/'registration.json');driver.write(training/'manifest.json',manifest)
            inputs.update(training_manifest_sha256=driver.sha(training/'manifest.json'),registration_sha256=driver.sha(training/'registration.json'))
            driver.write(fit/'inputs.json',inputs)
            with self.assertRaisesRegex(ValueError,'source changed'):driver.verify_fitted_training(fit)
    def test_time_proxy_separates_fresh_anchors_and_never_picks_best_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);profile=root/'profile.json';driver.write(profile,dict(coefficients={k:(1000000 if k=='access_warp_issues' else 0) for k in driver.FEATURES}))
            driver.write(root/'fit.json',dict(selection_ticks_per_microsecond=1000000,intercept=1.))
            protocol=dict(models=[{'name':'legacy'},{'name':'demand','id':'frozen-demand','profile':str(profile)}],training={'directory':str(root)})
            def row(key,split,assignment,count,arm):
                return dict(model='demand',scope='natural',spec=dict(key=key,split=split),execution_identity='image',assignment_sha256=assignment,
                    features={k:(count if k=='access_warp_issues' else 0) for k in driver.FEATURES},arm=arm)
            rows=[row('fresh','holdout','a',1,'J-old'),row('fresh','holdout','a',1,'J-new'),row('fresh','holdout','b',7,'LI'),
                  row('anchor','observed-anchor','c',3,'J-new')]
            analyses={key:{'image_batch_medians_us':{'image':[4.]*10}} for key in ('natural-fresh','natural-anchor')}
            result=driver.time_proxy_diagnostics(protocol,rows,analyses);main=result['summaries']['holdout']['all_scopes'];anchors=result['summaries']['observed-anchor']['all_scopes']
            self.assertEqual((main['group_unique_images'],main['assignment_variants'],main['independent_programs']),(1,2,1))
            self.assertEqual(main['mae_us'],3.);self.assertEqual(anchors['mae_us'],0.);self.assertFalse(result['gate']);self.assertFalse(result['refit_on_holdout'])
            bad=copy.deepcopy(rows);bad[1]['features']['access_warp_issues']=9
            with self.assertRaisesRegex(ValueError,'same assignment'):driver.time_proxy_diagnostics(protocol,bad,analyses)
    def test_profile_arguments_are_explicit_and_do_not_change_caller(self):
        model={'name':'demand','profile':'/frozen/profile.json'};before=dict(model)
        self.assertEqual(driver.model_args(model),['--cost-profile=/frozen/profile.json'])
        self.assertEqual(driver.model_args({'name':'legacy','profile':None}),[])
        self.assertEqual(model,before)
if __name__=='__main__':unittest.main()
