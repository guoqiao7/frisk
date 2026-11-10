"""CPU-only contract, algebra and conservative-decision regression tests."""
import copy
from fractions import Fraction
import math
import unittest
import numpy as np
import decision_priority as priority
import joint_cost_model as model
import joint_cost_calibration as calibration


def test_domain():
    return dict(threads=128,grid=[1,1,1],sm_version=[9,0],dtype='float32',gpu_uuid='test-device',
        math_flags=dict(exp='ex2.approx.f32(mul.rn.f32(x, log2e-f32))',ftz=False,fp_fusion=False,p_rounding='cvt.rn.f16.f32',reassociation=False))


def fixture(arm,count=10,g=20,available=True):
    names=(set(model.NATIVE_FEATURES)-{'shared_transfer_bytes','local_transfer_bytes','symbolic_bank_excess'}) | {'shared_load_bytes','shared_store_bytes','local_load_bytes','local_store_bytes'}
    raw={n:0 for n in names};raw['native_instructions']=count
    native=dict(schema_version=13,status='available' if available else 'unsupported',
        support=dict(status='supported' if available else 'unsupported',domain='test',reasons=[] if available else ['control-flow']),
        identity=dict(schema_version=13,domain='test',analyzer_sha256='b'*64,entry='frisk_execution',cubin_sha256='a'*64,
            nvdisasm=dict(sha256='c'*64,version='test'),cuobjdump=dict(sha256='d'*64,version='test')),
        features=raw,symbolic_banks=dict(contract='frisk-emitted-affine-shared-bank-v1',implementation_sha256={},shared_read_bank_excess=0,shared_write_bank_excess=0))
    common=dict(contract='sm90-network-static-common-v1',grammar='ordered-surviving-network-source-issues-v1',frequency_mode='legacy',
        weight_provenance='migrated-engineering-proxy-not-refit',units='nonnegative-score-ticks-per-feature-unit-not-cycles',
        feature_order=list(model.STATIC_FEATURES),coefficients={n:1 for n in model.STATIC_FEATURES})
    lines=[common[n] for n in ('contract','grammar','frequency_mode','weight_provenance','units')]
    lines += [n+'='+str(common['coefficients'][n]) for n in model.STATIC_FEATURES]
    import hashlib
    common['sha256']=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()
    preceding=priority._seal(dict(schema_version=1,contract=priority.CONTRACT,implementation_sha256=priority.IMPLEMENTATION_SHA256,
        kind='candidate-evidence',arm=arm,static_cost_tuple=[1]*9,static_model_id='test',static_profile_sha256='0'*64,
        frequency_mode='legacy',prepared_digest='1'*64,native_sha256=model.digest(native),cubin_sha256='a'*64,
        gf2=dict(status='available',global_sectors32=g),local=dict(status='available',bytes_per_thread=100,traffic_bytes=None),uses_gpu_timing=False))
    static={n:0 for n in model.STATIC_FEATURES};static['global_sectors32']=g
    evidence=model._seal(dict(schema_version=1,contract=model.CONTRACT,kind='joint-candidate-evidence',
        implementation_sha256=model.implementation(),preceding=preceding,static20=static,static_contract=common,execution_domain=test_domain(),uses_gpu_timing=False))
    return native,evidence


def profile(native):
    return dict(execution_domain=test_domain(),schema_version=1,contract=model.CONTRACT,model=model.MODEL,features=list(model.FEATURES),feature_units=dict(model.UNITS),
        implementation_sha256=model.implementation(),analysis_contract=calibration.native_calibration.analysis_contract(native),
        coefficients_us_per_unit={n:1. for n in model.FEATURES},training_maxima={n:100000 for n in model.FEATURES},
        uses_observed_development_timings=True,hardware_time_bound=False,
        risk_calibration=dict(method='leave-one-graph-out-max-pair-log-ratio-residual-v1',log_ratio_error_envelope=1.,graph_groups=['a','b'],hardware_time_bound=False),
        identifiability=dict(interpretation='not physical'),dependencies=[])


class JointModelTests(unittest.TestCase):
    def test_equal_native_different_address_dominance(self):
        a,ea=fixture('a',g=100);b,eb=fixture('b',g=20);p=profile(a)
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a',['a','b'])
        self.assertEqual(result['payload']['selected_source'],'b')
        self.assertEqual(result['payload']['status'],'empirical-risk-selected')
        self.assertEqual(result['payload']['model_certificate']['status'],'nonnegative-family-argmin-proved')
        self.assertFalse(result['payload']['model_certificate']['bypasses_empirical_risk'])
        self.assertFalse(result['payload']['hardware_time_bound'])
        model.verify_selection(result,{'a':a,'b':b},{'a':ea,'b':eb},p,'a',['a','b'])
        changed=copy.deepcopy(result);changed['payload']['selected_source']='a'
        with self.assertRaises(ValueError):model.verify_selection(changed,{'a':a,'b':b},{'a':ea,'b':eb},p,'a',['a','b'])

    def test_native_unsupported_is_not_zero_cost(self):
        a,ea=fixture('a');b,eb=fixture('b',available=False)
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},profile(a),'a')
        self.assertEqual(result['payload']['status'],'fallback')
        self.assertIsNone(result['payload']['rows']['b']['features'])

    def test_risk_rejects_unconfident_tradeoff(self):
        a,ea=fixture('a',count=100,g=1);b,eb=fixture('b',count=10,g=20);p=profile(a)
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'b')
        self.assertEqual(result['payload']['status'],'empirical-risk-selected')
        p['risk_calibration']['log_ratio_error_envelope']=2.
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'a')
        self.assertIn('empirical-risk-threshold-not-met',result['payload']['reasons'])

    def test_tampered_native_and_units_rejected(self):
        a,e=fixture('a');a['features']['local_load_bytes']=True
        with self.assertRaises(ValueError):model.features(a,e)
        a,e=fixture('a');p=profile(a);p['feature_units']['local_transfer_bytes']='driver-frame-bytes'
        with self.assertRaises(ValueError):model.validate_profile(p)

    def test_training_zero_feature_forces_fallback(self):
        a,e=fixture('a');p=profile(a);p['training_maxima']['native_instructions']=0
        result=model.select({'a':a},{'a':e},p,'a')
        self.assertEqual(result['payload']['status'],'fallback')
        self.assertEqual(result['payload']['reasons'],['training-zero-feature-activation'])

    def test_margin_calibration_is_distinct_from_time_bound(self):
        a,ea=fixture('a',count=100,g=1);b,eb=fixture('b',count=10,g=20);p=profile(a)
        p['risk_calibration']=dict(method='out-of-program-selection-margin-v1',minimum_predicted_gain=.1,
            graph_groups=['train-a','train-b'],hardware_time_bound=False,material_loss_ratio=1.01,accepted_oof_proposals=1)
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'b')
        self.assertFalse(result['payload']['risk_certificate']['hardware_time_bound'])
        p['risk_calibration']['minimum_predicted_gain']=.9
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'a')

    def test_dispatch_without_complete_summary_retains_policy(self):
        a,ea=fixture('a',available=False);b,eb=fixture('b',available=False)
        result=model.dispatch({'a':a,'b':b},{'a':ea,'b':eb},{},'b',['b','a'],summaries={'a':None,'b':None})
        self.assertEqual(result['payload']['selected_source'],'b')
        self.assertIsNone(result['payload']['cost_family'])
        model.verify_dispatch(result,{'a':a,'b':b},{'a':ea,'b':eb},{},'b',['b','a'],summaries={'a':None,'b':None})

    def test_runtime_profile_does_not_open_training_timings(self):
        import tempfile,json
        from pathlib import Path
        a,_=fixture('a');p=profile(a)
        p['dependencies']=[dict(path='/no-such-archived-timings.json',sha256='0'*64,role='training')]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json';path.write_text(json.dumps(model._seal(p)))
            self.assertEqual(model.load_profile(path),p)
            with self.assertRaises(ValueError):model.load_profile(path,verify_training_dependencies=True)

    def test_independent_summary_family_and_dependency_binding(self):
        import execution_cost_summary as summary
        a,ea=fixture('a',available=False);b,eb=fixture('b',available=False,g=1)
        required={'kernel.cubin','execution.json','kernel.ttgir','kernel.ptx','execution_config.json','compile.json','image-identity.json'}
        dependencies={n:'0'*64 for n in required}
        for evidence in (ea,eb):
            preceding=evidence['payload']['preceding'];preceding['payload']['bindings']=dict(dependencies)
            preceding['payload_sha256']=model.digest(preceding['payload'])
            evidence['payload']['auxiliary_bindings']={}
            evidence['payload_sha256']=model.digest(evidence['payload'])
        ss=dict(contract=summary.CONTRACT,status='complete',cost_family='sm90-acyclic-call-expanded-occurrences-v1',
            feature_names=list(summary.BASE_FEATURES),feature_units={n:summary.UNITS[n] for n in summary.BASE_FEATURES},
            features={n:0 for n in summary.BASE_FEATURES},execution_bounds=dict(lower={},upper={}),
            provenance=dict(analyzer_sha256=model.implementation()['execution_cost_summary.py'],dependencies=dependencies,nvdisasm={'sha256':'z'}))
        ss['features']['native_instructions']=10;ss['summary_sha256']=model.digest(ss)
        p=profile(a);p['model']=model.SUMMARY_MODEL;p['cost_family']=ss['cost_family']
        names,units=model.feature_contract(p);p['features']=list(names);p['feature_units']=units
        p['training_maxima']={n:100000 for n in names};p['coefficients_us_per_unit']={n:1. for n in names}
        p['analysis_contract']=dict(contract=ss['contract'],analyzer_sha256=ss['provenance']['analyzer_sha256'],nvdisasm=ss['provenance']['nvdisasm'])
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a',summaries={'a':ss,'b':ss})
        self.assertEqual(result['payload']['selected_source'],'b')
        changed=copy.deepcopy(ss);changed['provenance']['dependencies']['kernel.ptx']='1'*64
        changed['summary_sha256']=model.digest({k:v for k,v in changed.items() if k!='summary_sha256'})
        with self.assertRaises(ValueError):model.summary_features(a,changed,ea,p)

    def test_zero_oof_acceptance_cannot_approve_default_override(self):
        a,ea=fixture('a',count=100,g=1);b,eb=fixture('b',count=10,g=20);p=profile(a)
        p['risk_calibration']=dict(method='out-of-program-selection-margin-v1',minimum_predicted_gain=.1,
            graph_groups=['a','b'],hardware_time_bound=False,accepted_oof_proposals=0)
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'a')
        self.assertEqual(result['payload']['reasons'],['empirical-acceptance-support-unavailable'])
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a',allow_empirical_extrapolation=True)
        self.assertEqual(result['payload']['selected_source'],'b')
        self.assertTrue(result['payload']['risk_certificate']['allow_empirical_extrapolation'])

    def test_model_dominance_never_bypasses_empirical_acceptance(self):
        a,ea=fixture('a',g=100);b,eb=fixture('b',g=20);p=profile(a)
        p['risk_calibration']=dict(method='out-of-program-selection-margin-v1',minimum_predicted_gain=.001,
            graph_groups=['a','b'],hardware_time_bound=False,accepted_oof_proposals=0)
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'a')
        self.assertEqual(result['payload']['model_certificate']['status'],'nonnegative-family-argmin-proved')
        self.assertIn('empirical-acceptance-support-unavailable',result['payload']['reasons'])

    def test_model_certificate_does_not_change_zero_weight_score_tie(self):
        base,eb=fixture('base',count=100,g=100);safe,es=fixture('safe',count=10,g=100);dom,ed=fixture('dom',count=10,g=1)
        p=profile(base);p['coefficients_us_per_unit']={n:float(n=='native_instructions') for n in model.FEATURES}
        p['risk_calibration']=dict(method='out-of-program-selection-margin-v1',minimum_predicted_gain=1e-9,
            graph_groups=['g'],hardware_time_bound=False,accepted_oof_proposals=1)
        result=model.select({'base':base,'safe':safe,'dom':dom},{'base':eb,'safe':es,'dom':ed},p,'base',['base','safe','dom'])
        self.assertEqual(result['payload']['selected_source'],'safe')
        self.assertEqual(result['payload']['model_certificate']['proved_candidate'],'dom')

    def test_multi_cta_domain_cannot_reuse_single_cta_risk(self):
        a,ea=fixture('a',g=100);b,eb=fixture('b',g=20);p=profile(a)
        for e in (ea,eb):
            e['payload']['execution_domain']['grid']=[257,1,1]
            e['payload_sha256']=model.digest(e['payload'])
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'a')
        self.assertEqual(result['payload']['status'],'fallback')
        self.assertTrue(all(r['reason']=='execution-domain-outside-calibration' for r in result['payload']['rows'].values()))

    def test_calibrated_order_cone_selects_without_identifying_prices(self):
        a,ea=fixture('a',count=100,g=1);b,eb=fixture('b',count=10,g=20);p=profile(a)
        row=[0]*len(model.FEATURES);row[model.FEATURES.index('native_instructions')]=-1;row[-1]=1
        witness=[[0,1] for _ in model.FEATURES];witness[0]=[2,1];witness[-1]=[1,1]
        p['ordering_cone']=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=list(model.FEATURES),constraints=[row],strict_feasible_witness=witness,
            leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0))
        p['coefficients_us_per_unit']={n:float(n=='gf2_global_sectors32') for n in model.FEATURES}
        result=model.select({'a':a,'b':b},{'a':ea,'b':eb},p,'a')
        self.assertEqual(result['payload']['selected_source'],'b')
        self.assertEqual(result['payload']['status'],'ordering-cone-certified')
        certificate=result['payload']['ordering_cone_certificate']
        self.assertTrue(certificate['common_offset_invariant'])
        self.assertFalse(certificate['hardware_time_bound'])
        self.assertTrue(certificate['comparisons']['a']['exact_dual_equality_verified'])
        a2,e2=fixture('a',count=100+10000,g=1+10000);b2,f2=fixture('b',count=10+10000,g=20+10000)
        shifted=model.select({'a':a2,'b':b2},{'a':e2,'b':f2},p,'a')
        self.assertEqual(shifted['payload']['selected_source'],'b')
        self.assertEqual(shifted['payload']['ordering_cone_certificate']['comparisons'],certificate['comparisons'])

    def test_order_cone_requires_nonempty_strict_witness_and_oof_support(self):
        a,ea=fixture('a');p=profile(a);row=[0]*len(model.FEATURES);row[0]=1
        bad=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=list(model.FEATURES),constraints=[row],strict_feasible_witness=[[1,1]]*len(model.FEATURES))
        with self.assertRaises(ValueError):model.verify_order_cone(bad,model.FEATURES)
        bad['status']='inconsistent';p['ordering_cone']=bad
        self.assertIsNone(model.ordering_cone_selection({'a':model.features(a,ea)['features']},p,'a',['a']))

    def test_weak_argmin_requires_strict_improvement_only_over_fallback(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=1,y=0),'winner':dict(x=0,y=0),'extra':dict(x=0,y=1)}
        result=model.ordering_cone_selection(f,p,'fallback',list(f))
        self.assertEqual(result['selected_source'],'winner')
        self.assertEqual(result['comparisons']['extra']['strict_calibration_mass'],[0,1])
        self.assertGreater(result['strict_fallback_improvement']['strict_calibration_mass'][0],0)

    def test_weak_argmin_prior_only_cannot_replace_fallback(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=0,y=1),'winner':dict(x=0,y=0),'extra':dict(x=1,y=0)}
        result=model.ordering_cone_selection(f,p,'fallback',list(f))
        self.assertEqual(result['selected_source'],'fallback')
        self.assertEqual(result['status'],'argmin-proved-without-strict-fallback-improvement')

    def test_weak_argmin_common_offset_changes_no_pair_certificate(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=1,y=0),'winner':dict(x=0,y=0),'extra':dict(x=0,y=1)}
        shifted={a:{k:v+1000 for k,v in row.items()} for a,row in f.items()}
        before=model.ordering_cone_selection(f,p,'fallback',list(f));after=model.ordering_cone_selection(shifted,p,'fallback',list(f))
        self.assertEqual(before,after)

    def test_weak_argmin_still_requires_oof_support(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=0,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=1,y=0),'winner':dict(x=0,y=0),'extra':dict(x=0,y=1)}
        self.assertIsNone(model.ordering_cone_selection(f,p,'fallback',list(f)))


    def test_strict_fallback_improvement_accepts_incomparable_opponent(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=2,y=1),'improvement':dict(x=1,y=1),'unresolved':dict(x=0,y=2)}
        result=model.ordering_cone_selection(f,p,'fallback',list(f))
        self.assertEqual(result['selected_source'],'improvement')
        self.assertEqual(result['status'],'calibrated-cone-fallback-improvement-proved')
        self.assertEqual(result['decision_rule_contract'],model.ORDER_SELECTION_RULE)
        self.assertFalse(result['argmin_proved'])
        self.assertEqual(result['complete_pool'],list(f))
        self.assertEqual(result['unresolved_opponents'],['unresolved'])
        self.assertGreater(result['strict_fallback_improvement']['strict_calibration_mass'][0],0)

    def test_strict_fallback_improvement_zero_contrast_is_rejected(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=1,y=1),'same':dict(x=1,y=1),'unresolved':dict(x=0,y=2)}
        result=model.ordering_cone_selection(f,p,'fallback',list(f))
        self.assertEqual(result['selected_source'],'fallback')
        self.assertFalse(result['argmin_proved'])
        self.assertEqual(result['status'],'not-proved')

    def test_strict_fallback_improvement_weak_prior_only_is_rejected(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=1,y=2),'weak':dict(x=1,y=1),'unresolved':dict(x=0,y=3)}
        result=model.ordering_cone_selection(f,p,'fallback',list(f))
        self.assertEqual(result['selected_source'],'fallback')
        self.assertEqual(result['status'],'not-proved')

    def test_strict_fallback_improvement_empty_oof_is_rejected(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=0,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=2,y=1),'improvement':dict(x=1,y=1),'unresolved':dict(x=0,y=2)}
        self.assertIsNone(model.ordering_cone_selection(f,p,'fallback',list(f)))

    def test_strict_fallback_improvement_uses_fixed_priority(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=3,y=3),'first':dict(x=2,y=3),'second':dict(x=1,y=3),'unresolved':dict(x=0,y=100)}
        result=model.ordering_cone_selection(f,p,'fallback',list(f))
        self.assertEqual(result['selected_source'],'first')
        self.assertEqual(result['strict_improvement_candidates'],['first','second'])
        self.assertFalse(result['argmin_proved'])

    def test_strict_fallback_improvement_common_offset_is_irrelevant(self):
        p=dict(features=['x','y'],ordering_cone=dict(contract=model.ORDER_CONE_CONTRACT,status='available',features=['x','y'],
            constraints=[[-1,0]],strict_feasible_witness=[[1,1],[0,1]],leave_program_out=dict(proved_pairs=1,contradicted_registered_pairs=0)))
        f={'fallback':dict(x=2,y=1),'improvement':dict(x=1,y=1),'unresolved':dict(x=0,y=2)}
        shifted={a:{n:v+987654321 for n,v in row.items()} for a,row in f.items()}
        self.assertEqual(model.ordering_cone_selection(f,p,'fallback',list(f)),model.ordering_cone_selection(shifted,p,'fallback',list(f)))


class CalibrationTests(unittest.TestCase):
    def test_nnls_active_boundary_and_kkt(self):
        weights,receipt=calibration.nnls([[1,0],[0,1],[1,1]],[2,-1,1])
        np.testing.assert_allclose(weights,[1.5,0],atol=1e-8)
        self.assertTrue(receipt['converged'])
        self.assertLess(receipt['scaled_kkt_residual'],1e-7)

    def test_exact_alias_and_gf2_independent_axis(self):
        matrix=[[1,2,3],[2,4,6],[1,2,7]]
        result=calibration.exact_linear_algebra(matrix,['native','alias','gf2'])
        self.assertEqual(result['rank'],2)
        self.assertEqual(result['exact_nullspace_basis'],[{'native':-2,'alias':1}])
        self.assertNotIn('gf2',result['alias_involved_features'])

    def test_graph_split_removes_cross_graph_program_alias(self):
        rows=[]
        for graph in ('a','b'):
            for i,identity in enumerate(('shared',graph+'1',graph+'2')):
                rows.append(dict(group=graph,graph=graph,arm=str(i),execution_identity=identity,
                    features={'work':i+1},median_us=float(i+1),batch_medians_us=[float(i+1)]*3))
        risk,_=calibration.graph_validation(rows,('work',))
        self.assertEqual(len(risk['folds']),2)
        self.assertTrue(all(f['alias_overlap_removed']==1 for f in risk['folds']))
        self.assertTrue(all(f['train_unique']==2 for f in risk['folds']))
        self.assertLess(risk['log_ratio_error_envelope'],1e-8)

    def test_auxiliary_training_state_is_scoped_even_after_failure(self):
        from unittest.mock import patch
        before=(calibration._NATIVE_MICRO_PAIRS,calibration._NATIVE_MICRO_MAXIMA,calibration._NATIVE_MICRO_DEPENDENCIES)
        with patch.object(calibration,'_run_bundle',side_effect=RuntimeError('synthetic offline failure')):
            with self.assertRaises(RuntimeError):calibration.run_bundle('unused','unused','unused')
        self.assertIs(calibration._NATIVE_MICRO_PAIRS,before[0])
        self.assertIs(calibration._NATIVE_MICRO_MAXIMA,before[1])
        self.assertIs(calibration._NATIVE_MICRO_DEPENDENCIES,before[2])

    def test_positive_cost_cone_keeps_opposite_but_drops_exact_positive_alias(self):
        rows=[dict(group='program',graph='g',arm=str(i),execution_identity=str(i),
            features=dict(base=i,opposite=3-i,redundant=2*i,constant=8)) for i in range(3)]
        axes,receipt=calibration.reduce_locality_axes(rows,['base'],['opposite','redundant','constant'])
        self.assertEqual(axes,['opposite'])
        self.assertEqual(receipt['ledger'][0]['status'],'retained-Q-dependent-nonnegative-redundancy-not-proved')
        self.assertEqual(receipt['ledger'][1]['status'],'dropped-exact-nonnegative-cone-combination')
        self.assertEqual(receipt['ledger'][1]['coefficients'],{'base':[2,1]})
        self.assertEqual(receipt['ledger'][2]['status'],'dropped-no-within-program-variation')

    def test_micro_proof_rejects_changed_stride_pointer_and_late_write(self):
        source=calibration.native_calibration.micro_ptx(dict(key='baseline',kind='baseline',repeats=0,stride=1))
        proof=calibration.micro_global_skeleton(source)
        self.assertEqual(len(proof['address_definitions']),6)
        for changed in (source.replace('mul.wide.u32 %i,%t,4;','mul.wide.u32 %i,%t,128;'),
                        source.replace('ld.param.u64 %q,[Q];','ld.param.u64 %q,[K];'),
                        source.replace('st.global.u32 [%op],%x;','mov.u64 %op,0;\nst.global.u32 [%op],%x;')):
            with self.assertRaises(ValueError):calibration.micro_global_skeleton(changed)

    def test_native_only_cli_runs_tiny_real_fit_without_undefined_names(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        rows=[];contract=None
        for graph in ('first','second'):
            for i in (1,2):
                native,evidence=fixture(graph+str(i),count=i,g=10)
                extracted=model.features(native,evidence);contract=extracted['analysis_contract']
                rows.append(dict(group=graph,graph=graph,spec={},arm=str(i),execution_identity=graph+str(i),
                    cubin_sha256='a'*64,path='synthetic',features=extracted['features'],feature_status='available',feature_reason=None,
                    static20={},native=native,evidence=evidence,median_us=float(i),batch_medians_us=[float(i)]*3))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(calibration,'collect',return_value=(rows,contract,[],{},{})),patch.object(calibration,'read',return_value={}),patch.object(calibration,'evaluate',return_value=dict(summary={},status_counts={})):result=calibration.run_calibration(root/'run',root/'out')
            self.assertEqual(result['completion']['status'],'complete')

    def test_native_only_cli_clears_and_recovers_auxiliary_state(self):
        from unittest.mock import patch
        original=(calibration._NATIVE_MICRO_PAIRS,calibration._NATIVE_MICRO_MAXIMA,calibration._NATIVE_MICRO_DEPENDENCIES)
        sentinels=([{'unused':1}],{'unused':1},[{'unused':1}])
        calibration._NATIVE_MICRO_PAIRS,calibration._NATIVE_MICRO_MAXIMA,calibration._NATIVE_MICRO_DEPENDENCIES=sentinels
        def check_cleared(*args):
            self.assertEqual(calibration._NATIVE_MICRO_PAIRS,[])
            self.assertEqual(calibration._NATIVE_MICRO_MAXIMA,{})
            self.assertEqual(calibration._NATIVE_MICRO_DEPENDENCIES,[])
            raise RuntimeError('synthetic')
        try:
            with patch.object(calibration,'_run_calibration',side_effect=check_cleared):
                with self.assertRaises(RuntimeError):calibration.run_calibration('unused','unused')
            self.assertIs(calibration._NATIVE_MICRO_PAIRS,sentinels[0])
        finally:calibration._NATIVE_MICRO_PAIRS,calibration._NATIVE_MICRO_MAXIMA,calibration._NATIVE_MICRO_DEPENDENCIES=original

    def test_empirical_dropped_axis_identity_is_rechecked(self):
        rows={'a':dict(features={'work':1}),'b':dict(features={'work':2})}
        documents={'a':dict(features={'axis':7}),'b':dict(features={'axis':9})}
        profile=dict(locality_reduction=dict(ledger=[dict(axis='axis',status='dropped-exact-nonnegative-cone-combination',coefficients={'work':[2,1]})]))
        self.assertTrue(model.locality_reduction_holds(rows,documents,profile,['a','b']))
        documents['b']['features']['axis']=10
        self.assertFalse(model.locality_reduction_holds(rows,documents,profile,['a','b']))

    def test_small_farkas_certificate_is_exact(self):
        certificate=calibration.small_farkas([[-1,2],[2,-1]])
        self.assertIsNotNone(certificate)
        self.assertTrue(all(Fraction(*v)>=0 for v in certificate['combined_delta']))


if __name__=='__main__':unittest.main()
