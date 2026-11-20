import copy
import random
import unittest
import decision_priority as dp
import partial_cost_selection as partial
import native_cost_calibration as calibration


def profile(untrained_local=False):
    weights = {n: 1 for n in calibration.MODEL_FEATURES}
    maxima = {n: 100 for n in calibration.MODEL_FEATURES}
    if untrained_local:
        weights['local_transfer_bytes'] = maxima['local_transfer_bytes'] = 0
    return dict(model='sm90-native-surviving-schedule-v1', features=list(calibration.MODEL_FEATURES),
                coefficients=weights, training_maxima=maxima, analysis_contract={})


def interval(arm, base, p, local=0, unsupported=False):
    f = {n: 0 for n in calibration.MODEL_FEATURES}
    f['native_instructions'], f['local_transfer_bytes'] = base, local
    unknown = [n for n in calibration.MODEL_FEATURES if f[n] and p['training_maxima'][n] == 0]
    known = {n: dict(count=f[n], coefficient_ticks=p['coefficients'][n], contribution_ticks=f[n] * p['coefficients'][n]) for n in calibration.MODEL_FEATURES if n not in unknown}
    value = sum(v['contribution_ticks'] for v in known.values())
    return dict(schema_version=1, contract=partial.CONTRACT, implementation_sha256=partial.IMPLEMENTATION_SHA256,
                native_sha256=dp.digest({'arm': arm}), profile_sha256=dp.digest(p), uses_gpu_timing=False,
                bound_basis=partial.BASIS, status='unsupported' if unsupported else 'bounded' if unknown else 'exact',
                reason='native-analysis-unsupported' if unsupported else 'training-zero-feature-activation' if unknown else None,
                lower_ticks=None if unsupported else value, upper_ticks=None if unsupported or unknown else value,
                upper_unbounded=bool(unknown) and not unsupported, unknown_features=[] if unsupported else sorted(unknown),
                known_terms={} if unsupported else known,
                unknown_terms={} if unsupported else {n: dict(count=f[n], coefficient_lower_ticks=0, coefficient_upper_ticks=None, point_cost_available=False) for n in unknown}, features=f)


def evidence(arm, sectors, frame=0, primary=100, conversions=0):
    return dp._seal(dict(schema_version=1, contract=dp.CONTRACT, implementation_sha256=dp.IMPLEMENTATION_SHA256,
        kind='candidate-evidence', arm=arm, native_sha256=dp.digest({'arm': arm}),
        static_cost_tuple=[primary, 10, 0, 0, 1, 0, 0, conversions, 0],
        static_model_id='v6', static_profile_sha256='a' * 64, frequency_mode='legacy', prepared_digest='b' * 64,
        gf2=dict(status='available' if sectors is not None else 'unavailable', global_sectors32=sectors),
        local=dict(status='available' if frame is not None else 'unavailable', bytes_per_thread=frame),
        bindings={}, uses_gpu_timing=False))


def generic_matrix(rows, dimensions=('sector',)):
    return dp._seal(dict(schema_version=1, contract=dp.AFFINE_CONTRACT,
        implementation_sha256=dp.IMPLEMENTATION_SHA256, kind='affine-matrix',
        status='available', rows={a: dict(status='available', base_ticks=b, counts=dict(zip(dimensions, c))) for a, (b, c) in rows.items()},
        dimensions=list(dimensions), unsupported_candidates=[], uses_gpu_timing=False,
        coefficient_domain={n:dict(lower=0, upper=None, shared_across_all_candidates=True)for n in dimensions}, basis=dp.BASIS))


class DecisionPriorityTest(unittest.TestCase):
    def test_equal_native_cost_prefers_gf2_only_within_class(self):
        p = profile(); values = {a: interval(a, 100, p) for a in 'abc'}
        ev = {a: evidence(a, n) for a, n in zip('abc', (98304, 7680, 7680))}
        r = dp.native_priority(values, ev, list('abc'))
        self.assertEqual(r['payload']['priority'], list('bca'))
        self.assertEqual(dp.native_priority(values, ev, list('abc'), enabled=False)['payload']['priority'], list('abc'))
        values['a'] = interval('a', 99, p)
        r = dp.native_priority(values, ev, list('abc'))
        self.assertEqual(partial.select_partial_cost(values, fallback_arm='c', priority=r['payload']['priority'])['selected_source'], 'a')

    def test_missing_is_whole_class_not_pairwise(self):
        p = profile(); values = {a: interval(a, 100 if a != 'd' else 200, p) for a in 'abcd'}
        ev = {a: evidence(a, n) for a, n in zip('abcd', (10, 5, None, 0))}
        self.assertEqual(dp.native_priority(values, ev, list('abcd'))['payload']['priority'], list('abcd'))
        ev['c'] = evidence('c', 5); ev['d'] = evidence('d', None)
        self.assertEqual(dp.native_priority(values, ev, list('abcd'))['payload']['priority'], list('bcad'))

    def test_static_frame_does_not_cross_primary_and_precedes_conversion(self):
        ev = dict(a=evidence('a', 10, 6928, 100, 0), b=evidence('b', 10, 80, 100, 1), c=evidence('c', 0, 0, 101, 0))
        r = dp.static_fallback(ev, list('abc'))['payload']
        self.assertEqual(r['selected_source'], 'b'); self.assertFalse(r['full_tuple_minimum_preserved'])
        self.assertEqual(dp.static_fallback(ev, list('abc'), enabled=False)['payload']['selected_source'], 'a')
        ev['b'] = evidence('b', 10, None, 100, 1)
        self.assertEqual(dp.static_fallback(ev, list('abc'))['payload']['selected_source'], 'a')

    def test_class_outside_primary_missing_does_not_disable_frame(self):
        ev = dict(a=evidence('a', 1, 100, 100), b=evidence('b', 1, 0, 100, 1), c=evidence('c', 1, None, 101))
        self.assertEqual(dp.static_fallback(ev, list('abc'))['payload']['selected_source'], 'b')

    def test_factor_isolation_and_unknown_boundary(self):
        p = profile(True); values = dict(a=interval('a', 100, p), b=interval('b', 100, p, local=4), c=interval('c', 100, p))
        ev = dict(a=evidence('a', 10, 100), b=evidence('b', 0, 0, conversions=1), c=evidence('c', 5, 0, conversions=2))
        old_priority = list('abc')
        for frame_enabled in (False, True):
            fallback = dp.static_fallback(ev, old_priority, enabled=frame_enabled)['payload']['selected_source']
            for gf2_enabled in (False, True):
                order = dp.native_priority(values, ev, old_priority, enabled=gf2_enabled)['payload']['priority']
                decision = partial.select_partial_cost(values, fallback_arm=fallback, priority=order)
                self.assertEqual(decision['selected_source'], 'c' if gf2_enabled else 'a')
                self.assertEqual(decision['status'], 'interval-certified')
        values['b'] = interval('b', 99, p, local=4)
        for gf2_enabled in (False, True):
            order = dp.native_priority(values, ev, old_priority, enabled=gf2_enabled)['payload']['priority']
            self.assertEqual(partial.select_partial_cost(values, fallback_arm='b', priority=order)['selected_source'], 'b')

    def test_invalid_evidence_is_not_missing(self):
        p = profile(); values = {'a': interval('a', 1, p)}
        for bad in (-1, True, float('nan')):
            with self.assertRaises(ValueError):
                dp.native_priority(values, {'a': evidence('a', bad)}, ['a'])
        ev = {'a': evidence('a', 1)}; ev['a']['payload']['gf2']['global_sectors32'] = 0
        with self.assertRaises(ValueError): dp.native_priority(values, ev, ['a'])
        with self.assertRaises(ValueError): dp.native_priority(values, {'a': evidence('other', 1)}, ['a'])
        with self.assertRaises(ValueError): dp.native_priority(values, {'a': evidence('a', 1)}, ['a', 'a'])

    def test_receipt_recomputation_and_inputs_immutable(self):
        p = profile(); values = {a: interval(a, 5, p) for a in 'ab'}; ev = {a: evidence(a, n)for a, n in zip('ab',(9,1))}
        original = copy.deepcopy((values, ev)); r = dp.native_priority(values, ev, ['a','b'])
        self.assertEqual(dp.verify_native_priority(values, ev, ['a','b'], r), r)
        self.assertEqual(dp.native_priority(dict(reversed(list(values.items()))), dict(reversed(list(ev.items()))), ['a','b']), r)
        self.assertEqual((values, ev), original)
        r['payload']['priority'].reverse()
        with self.assertRaises(ValueError): dp.verify_native_priority(values, ev, ['a','b'], r)

    def test_affine_all_bounded_common_terms_cancel(self):
        p = profile(True); values = {a: interval(a, b, p, local=8)for a,b in [('a',10),('b',20)]}; ev = {a:evidence(a, 7)for a in values}
        m = dp.affine_matrix(values, ev, p); rows = m['payload']['rows']
        self.assertEqual(rows['a']['counts'], {'native-untrained:local_transfer_bytes':8, dp.GF2_DIMENSION:7})
        r = dp.select_affine(m, fallback_arm='b', priority=['b','a'])['payload']
        self.assertEqual(r['selected_source'], 'a'); self.assertEqual(r['status'], 'affine-certified')
        self.assertEqual(dp.verify_affine_matrix(values, ev, p, m), m)

    def test_affine_tradeoff_has_no_universal_winner(self):
        m = generic_matrix({'a':(1,[100]),'b':(2,[0])})
        r = dp.select_affine(m, fallback_arm='b', priority=['a','b'])['payload']
        self.assertEqual(r['status'], 'fallback'); self.assertEqual(r['selected_source'], 'b')
        self.assertEqual(set(r['necessity_counterexamples']), {'a','b'})

    def test_affine_cfg_unsupported_never_gets_lower_bound(self):
        p = profile(); values = dict(a=interval('a',1,p), b=interval('b',0,p,unsupported=True)); ev = {a:evidence(a,0)for a in values}
        m = dp.affine_matrix(values, ev, p)
        self.assertIsNone(m['payload']['rows']['b']['base_ticks']); self.assertIsNone(m['payload']['rows']['b']['counts'])
        self.assertEqual(dp.select_affine(m, fallback_arm='a', priority=['a','b'])['payload']['status'], 'fallback')

    def test_affine_exact_native_equal_sector_improves_completion(self):
        p = profile(); values = {a:interval(a,10,p)for a in 'abc'}; ev={a:evidence(a,n)for a,n in zip('abc',(98304,7680,7680))}
        m=dp.affine_matrix(values,ev,p);r=dp.select_affine(m,fallback_arm='a',priority=list('abc'))
        self.assertEqual(r['payload']['universal_argmin_members'], ['b','c'])
        self.assertEqual(r['payload']['selected_source'], 'b')
        self.assertEqual(dp.verify_affine_selection(m,r,fallback_arm='a',priority=list('abc')),r)

    def test_affine_domain_is_strict_and_not_a_point_fit(self):
        for key,value in [('lower',False),('upper',0),('shared_across_all_candidates',1)]:
            m=generic_matrix({'a':(1,[1]),'b':(2,[2])})
            m['payload']['coefficient_domain']['sector'][key]=value
            m=dp._seal(m['payload'])
            with self.assertRaises(ValueError):dp.select_affine(m,fallback_arm='b',priority=['a','b'])
        m=generic_matrix({'a':(1,[1]),'b':(2,[2])});m['payload']['schema_version']=True;m=dp._seal(m['payload'])
        with self.assertRaises(ValueError):dp.select_affine(m,fallback_arm='b',priority=['a','b'])

    def test_random_nonnegative_coefficients_and_necessity_witnesses(self):
        rng = random.Random(4107)
        for _ in range(120):
            rows={str(i):(rng.randrange(20),[rng.randrange(5)for _ in range(3)])for i in range(4)}
            m=generic_matrix(rows,('x','y','sector'));r=dp.select_affine(m,fallback_arm='0',priority=list(rows))['payload']
            expected=[a for a,(b,c)in rows.items()if all(b<=bb and all(v<=w for v,w in zip(c,cc))for bb,cc in rows.values())]
            self.assertEqual(r['universal_argmin_members'],expected)
            for _ in range(16):
                u=[rng.randrange(10000)for _ in range(3)];cost={a:b+sum(x*y for x,y in zip(c,u))for a,(b,c)in rows.items()}
                for a in expected:self.assertEqual(cost[a],min(cost.values()))
            for a,w in r['necessity_counterexamples'].items():
                u=[w['coefficients'][d]for d in m['payload']['dimensions']]
                b,c=rows[a];bb,cc=rows[w['opponent']]
                self.assertGreater(b+sum(x*y for x,y in zip(c,u)),bb+sum(x*y for x,y in zip(cc,u)))


if __name__ == '__main__': unittest.main()
