"""Semantic/certificate tests for partial native cost selection; no GPU use."""
import copy
import itertools
import random
import unittest
import native_cost_calibration as calibration
import partial_cost_selection as partial


def native(store=10,local=0):
    f={n:0 for n in ('native_instructions','global_load_bytes','global_store_bytes',
        'shared_load_bytes','shared_store_bytes','local_load_bytes','local_store_bytes',
        'barrier_instructions','shuffle_instructions','float_arithmetic_instructions',
        'special_math_instructions','integer_logic_instructions','load_use_exposure_32',
        'load_use_exposure_128','dependency_depth')}
    f.update(global_store_bytes=store,local_load_bytes=local)
    domain='sm90-scalar-straight-line-native-v1';cubin='1'*64
    return dict(schema_version=1,status='available',features=f,
        identity=dict(schema_version=1,domain=domain,entry='frisk_execution',
            cubin_sha256=cubin,analyzer_sha256='2'*64,
            nvdisasm=dict(sha256='3'*64,version='v'),cuobjdump=dict(sha256='4'*64,version='v')),
        support=dict(status='supported',domain=domain,reasons=[]),
        symbolic_banks=dict(contract='frisk-emitted-affine-shared-bank-v1',
            native_cubin_sha256=cubin,shared_read_bank_excess=0,shared_write_bank_excess=0,
            implementation_sha256={'test-source':'5'*64}))


def profile():
    p=dict(schema_version=1,model='sm90-native-surviving-schedule-v1',
        features=list(calibration.MODEL_FEATURES),coefficients={n:1 for n in calibration.MODEL_FEATURES},
        training_maxima={n:100 for n in calibration.MODEL_FEATURES},
        analysis_contract=calibration.analysis_contract(native()))
    p['coefficients']['local_transfer_bytes']=0;p['training_maxima']['local_transfer_bytes']=0
    return p


class IntervalTest(unittest.TestCase):
    def test_exact(self):
        x=partial.cost_interval(native(13),profile())
        self.assertEqual((x['status'],x['lower_ticks'],x['upper_ticks']),('exact',13,13))
        self.assertFalse(x['upper_unbounded']);self.assertFalse(x['uses_gpu_timing'])
        self.assertEqual(partial.verify_interval(native(13),profile(),x),x)

    def test_unknown_local_is_unbounded_not_zero_point_price(self):
        x=partial.cost_interval(native(13,100),profile())
        self.assertEqual((x['status'],x['lower_ticks'],x['upper_ticks']),('bounded',13,None))
        self.assertTrue(x['upper_unbounded']);self.assertEqual(x['unknown_features'],['local_transfer_bytes'])
        self.assertNotIn('local_transfer_bytes',x['known_terms'])
        self.assertFalse(x['unknown_terms']['local_transfer_bytes']['point_cost_available'])
        self.assertIsNone(x['native_score']['cost'])

    def test_new_local_calibration_makes_same_feature_exact(self):
        p=profile();p['training_maxima']['local_transfer_bytes']=100;p['coefficients']['local_transfer_bytes']=3
        x=partial.cost_interval(native(13,100),p)
        self.assertEqual((x['status'],x['lower_ticks'],x['upper_ticks']),('exact',313,313))

    def test_positive_untrained_prior_cannot_invent_an_interval_family(self):
        p=profile();p['coefficients']['local_transfer_bytes']=1
        x=partial.cost_interval(native(13,10),p)
        self.assertEqual(x['status'],'unsupported');self.assertIsNone(x['lower_ticks'])
        self.assertEqual(x['reason'],'unidentified-feature-has-nonzero-coefficient')

    def test_all_untrained_zero_columns_can_be_named(self):
        p=profile();p['training_maxima']['global_store_bytes']=0;p['coefficients']['global_store_bytes']=0
        x=partial.cost_interval(native(13,10),p)
        self.assertEqual(x['unknown_features'],['global_store_bytes','local_transfer_bytes'])
        self.assertEqual(x['lower_ticks'],0)

    def test_ordinary_unsupported_never_acquires_zero_lower_bound(self):
        for status in ('unsupported','unavailable'):
            n=native();n['status']=status;n['supported']=True
            x=partial.cost_interval(n,profile())
            self.assertEqual(x['status'],'unsupported');self.assertIsNone(x['lower_ticks'])

    def test_native_bank_tool_and_schema_failures_cannot_form_bound(self):
        for mutate in (
            lambda n:n['identity']['nvdisasm'].update(sha256='6'*64),
            lambda n:n['identity'].update(analyzer_sha256='6'*64),
            lambda n:n.update(schema_version=2),
            lambda n:n['support'].update(domain='other'),
            lambda n:n['support'].update(reasons=['control-flow:CALL']),
            lambda n:n['symbolic_banks'].update(native_cubin_sha256='6'*64),
            lambda n:n.pop('symbolic_banks'),
            lambda n:n['symbolic_banks'].update(implementation_sha256={'changed':'7'*64}),
        ):
            n=native(13,10);mutate(n);x=partial.cost_interval(n,profile())
            self.assertEqual(x['status'],'unsupported');self.assertIsNone(x['lower_ticks'])

    def test_invalid_profiles_are_integrity_errors(self):
        for value in (-1,float('nan'),float('inf'),True,1.5):
            for field in ('coefficients','training_maxima'):
                p=profile();p[field]['local_transfer_bytes']=value
                with self.assertRaises(ValueError):partial.cost_interval(native(13,10),p)
        p=profile();p['coefficients']['shuffle_instructions']=0
        with self.assertRaises(ValueError):partial.cost_interval(native(),p)
        p=profile();p['model']='unknown';n=native();n['status']='unavailable'
        with self.assertRaises(ValueError):partial.cost_interval(n,p)

    def test_malformed_native_feature_is_rejected(self):
        for value in (-1,float('nan'),float('inf'),True,1.5):
            n=native();n['features']['local_load_bytes']=value
            with self.assertRaises(ValueError):partial.cost_interval(n,profile())

    def test_bank_bool_and_unknown_feature_schema_are_rejected(self):
        n=native();n['symbolic_banks']['shared_read_bank_excess']=True
        with self.assertRaises(ValueError):partial.cost_interval(n,profile())
        n=native();n['features']['unknown_new_feature']=1
        with self.assertRaises(ValueError):partial.cost_interval(n,profile())

    def test_profile_and_native_evidence_are_bound(self):
        p=profile();n=native(13,10);x=partial.cost_interval(n,p)
        changed=copy.deepcopy(x);changed['lower_ticks']+=1
        with self.assertRaises(ValueError):partial.verify_interval(n,p,changed)
        n['features']['global_store_bytes']+=1
        with self.assertRaises(ValueError):partial.verify_interval(n,p,x)
        n=native(13,10);p['coefficients']['global_store_bytes']+=1
        with self.assertRaises(ValueError):partial.verify_interval(n,p,x)


class SelectionTest(unittest.TestCase):
    def intervals(self,items):
        p=profile()
        return {k:partial.cost_interval(native(*args),p) for k,args in items.items()}

    def test_exact_argmin_and_declared_ties(self):
        c=self.intervals({'slow':(20,0),'a':(10,0),'b':(10,0)})
        x=partial.select_partial_cost(c,fallback_arm='slow',priority=['b','slow','a'])
        self.assertEqual((x['status'],x['selected_source']),('exact-ranked','b'))
        self.assertEqual(x['certificate']['status'],'argmin-member-proven')
        self.assertFalse(x['certificate']['unique_optimum_proven'])

    def test_dominated_unknown_retained_and_certified(self):
        c=self.intervals({'slow':(20,0),'fast':(10,0),'unknown':(30,100)})
        x=partial.select_partial_cost(c,fallback_arm='slow')
        self.assertEqual((x['status'],x['selected_source']),('interval-certified','fast'))
        self.assertEqual(x['certificate']['bounded_candidates'],['unknown'])
        self.assertTrue(x['certificate']['unknown_lower_bound_comparisons'][0]['dominated'])
        self.assertEqual(set(x['certificate']['binding']['candidates']),set(c))

    def test_equal_unknown_lower_bound_prefers_exact_member(self):
        c=self.intervals({'unknown':(10,100),'exact':(10,0)})
        x=partial.select_partial_cost(c,fallback_arm='unknown',priority=['unknown','exact'])
        self.assertEqual(x['selected_source'],'exact')
        self.assertTrue(x['certificate']['argmin_membership_only'])
        self.assertFalse(x['certificate']['unique_optimum_proven'])

    def test_unknown_can_be_better_requires_fallback(self):
        c=self.intervals({'slow':(20,0),'fast':(10,0),'unknown':(1,100)})
        x=partial.select_partial_cost(c,fallback_arm='slow')
        self.assertEqual((x['status'],x['selected_source']),('fallback','slow'))
        self.assertIn('unknown-lower-bound-can-be-better:unknown',x['reasons'])
        self.assertEqual(x['certificate']['status'],'uncertified-fallback')

    def test_hard_unsupported_blocks_even_when_known_score_zero(self):
        p=profile();n=native();n['status']='unsupported'
        c={'exact':partial.cost_interval(native(0),p),'control':partial.cost_interval(n,p)}
        x=partial.select_partial_cost(c,fallback_arm='exact')
        self.assertEqual(x['status'],'fallback')
        self.assertEqual(x['certificate']['unsupported_candidates'],['control'])

    def test_no_exact_incumbent_is_not_a_fake_optimum(self):
        c=self.intervals({'unknown':(10,100)})
        x=partial.select_partial_cost(c,fallback_arm='unknown')
        self.assertEqual(x['status'],'fallback');self.assertIn('no-exact-incumbent',x['reasons'])

    def test_forced_domain_fallback_is_bound_and_explicit(self):
        c=self.intervals({'exact':(10,0),'fallback':(20,0)})
        x=partial.select_partial_cost(c,fallback_arm='fallback',force_fallback_reason='planned-control-flow')
        self.assertEqual((x['status'],x['selected_source']),('fallback','fallback'))
        self.assertEqual(x['certificate']['binding']['force_fallback_reason'],'planned-control-flow')

    def test_candidate_set_priority_and_fallback_are_complete(self):
        c=self.intervals({'a':(10,0),'b':(20,0)})
        x=partial.select_partial_cost(c,fallback_arm='b',priority=['b','a'])
        self.assertEqual(partial.verify_selection(c,x,fallback_arm='b',priority=['b','a']),x)
        with self.assertRaises(ValueError):partial.verify_selection(c,x,fallback_arm='a',priority=['b','a'])
        with self.assertRaises(ValueError):partial.verify_selection(c,x,fallback_arm='b',priority=['a','b'])
        with self.assertRaises(ValueError):partial.verify_selection({'b':c['b']},x,fallback_arm='b')
        for pr in (['a'],['a','a'],['a','c']):
            with self.assertRaises(ValueError):partial.select_partial_cost(c,fallback_arm='b',priority=pr)
        with self.assertRaises(ValueError):partial.select_partial_cost(c,fallback_arm='missing')

    def test_mixed_profiles_are_rejected(self):
        p=profile();a=partial.cost_interval(native(10),p);p['coefficients']['global_store_bytes']=2
        b=partial.cost_interval(native(10),p)
        with self.assertRaises(ValueError):partial.select_partial_cost({'a':a,'b':b},fallback_arm='a')

    def test_interval_tamper_detected_without_silently_repairing(self):
        c=self.intervals({'a':(10,0),'b':(20,100)})
        for mutate in (
            lambda v:v.update(lower_ticks=-1),lambda v:v.update(lower_ticks=float('nan')),
            lambda v:v.update(upper_ticks=30),lambda v:v.update(unknown_features=[]),
            lambda v:v['known_terms']['global_store_bytes'].update(count=1),
            lambda v:v['unknown_terms']['local_transfer_bytes'].update(coefficient_lower_ticks=-1),
            lambda v:v.update(uses_gpu_timing=True),lambda v:v.update(implementation_sha256='0'*64),
        ):
            modified=copy.deepcopy(c);mutate(modified['b'])
            with self.assertRaises(ValueError):partial.select_partial_cost(modified,fallback_arm='a')

    def test_complete_native_recompute_binds_receipt_changes(self):
        natives={'a':native(10),'b':native(20,100)};p=profile()
        x=partial.select_from_native(natives,p,fallback_arm='b')
        self.assertEqual(partial.verify_from_native(natives,p,x,fallback_arm='b'),x)
        changed=copy.deepcopy(x);changed['certificate']['incumbent_ticks']+=1
        with self.assertRaises(ValueError):partial.verify_from_native(natives,p,changed,fallback_arm='b')
        changed=copy.deepcopy(natives);changed['b']['features']['local_load_bytes']+=1
        with self.assertRaises(ValueError):partial.verify_from_native(changed,p,x,fallback_arm='b')

    def test_timing_payload_does_not_enter_ranking(self):
        n={'a':native(10),'b':native(20)};p=profile()
        before=partial.select_from_native(n,p,fallback_arm='b')
        n['a']['measured_us']=100000;n['b']['measured_us']=0
        after=partial.select_from_native(n,p,fallback_arm='b')
        self.assertEqual(before['selected_source'],after['selected_source'])
        self.assertNotEqual(before['certificate']['binding_sha256'],after['certificate']['binding_sha256'])

    def test_random_nonnegative_completions_preserve_certified_argmin(self):
        rng=random.Random(70131);certified=0;checked=0
        for _ in range(300):
            items={f'c{i}':(rng.randrange(0,30),0 if i==0 or rng.random()<.5 else rng.randrange(1,20)) for i in range(6)}
            intervals=self.intervals(items);decision=partial.select_partial_cost(intervals,fallback_arm='c0')
            if decision['status']=='fallback':continue
            certified+=1;chosen=decision['selected_source']
            # The unknown LOCAL coefficient is global/shared across candidates.
            for w in (0,1,7,100,10**9):
                costs={k:count+local*w for k,(count,local) in items.items()}
                self.assertEqual(costs[chosen],min(costs.values()));checked+=1
        self.assertGreater(certified,100);self.assertGreater(checked,500)

    def test_two_unknown_coefficients_exhaustive_completions(self):
        p=profile();p['training_maxima']['integer_logic_instructions']=0;p['coefficients']['integer_logic_instructions']=0
        a=native(5);b=native(9,3);b['features']['integer_logic_instructions']=2
        c=native(6,1);c['features']['integer_logic_instructions']=7
        decision=partial.select_from_native({'a':a,'b':b,'c':c},p,fallback_arm='c')
        self.assertEqual(decision['selected_source'],'a');self.assertEqual(decision['status'],'interval-certified')
        for w,v in itertools.product((0,1,2,1000),repeat=2):
            self.assertEqual(min(5,9+3*w+2*v,6+w+7*v),5)

if __name__=='__main__':unittest.main()
