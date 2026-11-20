import copy
import math
import unittest
from pathlib import Path
import numpy as np
import gf2_decision_fixture as fixture
from selection_quality_fixture import program_identity


class GF2DecisionFixtureTest(unittest.TestCase):
    def configuration(self, main='joint-affine-frame'):
        return dict(static=dict(id='static-v6', profile='/fixture/static.json'),
                    native=dict(id='native-v1', profile='/fixture/native.json', analysis_version='native-v1'),
                    old_native=dict(id='native-old-v1', profile='/fixture/old-native.json'),
                    profiles=[dict(path='/fixture/static.json', sha256='0' * 64)],
                    training_inputs={'/fixture/known.json': '1' * 64},
                    preserved_evidence=[dict(path='/fixture/old-index.json', sha256='2' * 64)],
                    archive_capacity=4, main_policy=main)

    def test_fixed_large_cohort_and_all_regressions(self):
        rows = fixture.specifications()
        self.assertTrue(fixture.validate_specifications(rows))
        self.assertEqual(len(rows), 15)
        self.assertEqual([s['shape'] for s in rows[:6]], [[256, 128], [512, 64], [128, 256], [1024, 64], [256, 128], [256, 128]])
        self.assertEqual([math.prod(s['shape']) for s in rows[:6]], [32768, 32768, 32768, 65536, 32768, 32768])
        self.assertEqual(sum(s['split'] == 'holdout' for s in rows), 6)
        self.assertEqual(sum(s['split'] == 'observed-regression' for s in rows), 6)
        self.assertEqual(sum(s['split'] == 'observed-anchor' for s in rows), 3)
        self.assertTrue(all(s['key'].startswith('regression-r3-') for s in rows[6:]))

    def test_history_contains_r3_without_changing_old_fixture(self):
        history = fixture.historical_specifications()
        self.assertEqual(len(history), 47)
        old = fixture.previous.specifications()
        self.assertEqual(len(old), 9)
        self.assertEqual(len(fixture.previous.historical_specifications()), 41)
        self.assertEqual(sum(s['split'] == 'holdout' for s in old), 6)
        self.assertTrue({program_identity(s) for s in old}.issubset({program_identity(s) for s in history}))

    def test_r3_cannot_be_relabelled_fresh(self):
        rows = fixture.specifications()
        rows[0]['shape'] = [128, 128]
        with self.assertRaises(ValueError): fixture.validate_specifications(rows)

    def test_repeated_flat_workload_with_new_factors_is_rejected(self):
        for index, shape in [(0, [256, 64]), (3, [1024, 32])]:
            rows = fixture.specifications()
            rows[index]['shape'] = shape
            with self.assertRaises(ValueError): fixture.validate_specifications(rows)

    def test_regressions_cannot_be_dropped_or_changed(self):
        for change in ('drop', 'tail', 'shape', 'role'):
            rows = fixture.specifications()
            if change == 'drop': rows.pop(6)
            elif change == 'tail': rows[6]['tail'] = 'exp1'
            elif change == 'shape': rows[6]['shape'] = [256, 64]
            else: rows[6]['role'] = 'fresh-performance-holdout'
            with self.assertRaises(ValueError): fixture.validate_specifications(rows)

    def test_fresh_family_and_identity_duplicates_are_rejected(self):
        rows = fixture.specifications()
        rows[1] = dict(rows[0], key='holdout-refined-duplicate')
        with self.assertRaises(ValueError): fixture.validate_specifications(rows)
        rows = fixture.specifications()
        rows[0]['graph_name'] = 'unregistered'
        with self.assertRaises(ValueError): fixture.validate_specifications(rows)

    def test_support_domain_and_shape_types_are_enforced(self):
        changes = [('shape', [0, 128]), ('shape', [True, 32768]), ('shape', [3, 16384]),
                   ('shape', [1024, 128]), ('dtype', 'float16'), ('threads', 256),
                   ('abi_stratum', 'strong-cpasync')]
        for name, value in changes:
            rows = fixture.specifications()
            rows[0][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                fixture.validate_specifications(rows)

    def test_regression_source_and_numerics_reuse_accepted_semantics(self):
        root = Path(fixture.__file__).resolve().parents[3]
        old = {program_identity(s): s for s in fixture.previous.specifications()}
        for s in fixture.regression_specifications():
            original = old[program_identity(s)]
            self.assertEqual(fixture.source(s, root), fixture.previous.source(original, root))
            new_case = fixture.numerical_cases(s)[0]
            old_case = fixture.previous.numerical_cases(original)[0]
            self.assertEqual(new_case[0], old_case[0])
            for a, b in zip(new_case[1:], old_case[1:]):
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.shape, b.shape)
                self.assertEqual(a.tobytes(), b.tobytes())
            self.assertEqual(fixture.reference(s, *new_case[1:]).tobytes(),
                             fixture.previous.reference(original, *old_case[1:]).tobytes())

    def test_new_references_are_finite_at_the_declared_size(self):
        for s in fixture.fresh_specifications():
            case = fixture.numerical_cases(s)[0]
            value = fixture.reference(s, *case[1:])
            self.assertEqual(value.dtype, np.float32)
            self.assertEqual(value.size, math.prod(s['shape']))
            self.assertTrue(np.isfinite(value).all())

    def test_main_policy_requires_explicit_preregistration(self):
        for main in fixture.MAIN_POLICIES:
            p = fixture.protocol(self.configuration(main))
            self.assertEqual(p['acceptance']['promotion']['primary'], main)
            self.assertEqual(p['configuration']['main_policy'], main)
        for main in (None, '', 'local-affine-frame', 'unregistered'):
            c = self.configuration(main)
            with self.assertRaises(ValueError): fixture.protocol(c)
        c = self.configuration()
        del c['main_policy']
        with self.assertRaises(ValueError): fixture.protocol(c)

    def test_protocol_keeps_statistics_device_safety_and_budgets(self):
        c = self.configuration()
        snapshot = copy.deepcopy(c)
        old = fixture.previous.protocol(c)
        p = fixture.protocol(c)
        self.assertEqual(c, snapshot)
        self.assertEqual(p['stage'], fixture.STAGE)
        self.assertEqual(p['numerics'], old['numerics'])
        self.assertEqual(p['execution'], old['execution'])
        self.assertEqual(p['safety'], old['safety'])
        self.assertEqual({k: v for k, v in p['measurement'].items() if k != 'seed'},
                         {k: v for k, v in old['measurement'].items() if k != 'seed'})
        self.assertEqual(p['budget'], old['budget'])
        self.assertEqual(p['search_contract'], old['search_contract'])
        self.assertEqual(p['acceptance']['promotion']['guard_comparisons'], 15)
        self.assertEqual(len(p['specifications']), 15)
        self.assertEqual(len(p['historical_inventory']), 47)

    def test_same_pool_factorial_and_affine_assumptions_are_explicit(self):
        p = fixture.protocol(self.configuration())
        experiment = p['priority_experiment']
        self.assertEqual(len(set(p['experiment']['ablations'])), 13)
        self.assertEqual(set(p['experiment']['ablations']), set(fixture.POLICY_LABELS))
        self.assertEqual({(r['gf2_exact_score_tie'], r['compiled_local_fallback']) for r in experiment['factorial']},
                         {(False, False), (True, False), (False, True), (True, True)})
        self.assertTrue(experiment['same_pool'])
        self.assertTrue(experiment['same_compile_budget'])
        self.assertTrue(experiment['all_candidates_retained'])
        self.assertFalse(experiment['uses_gpu_timing'])
        self.assertFalse(experiment['case_or_shape_special_cases'])
        affine = experiment['affine_completion']
        self.assertIn('shared across all candidates', affine['parameters'])
        self.assertIn('proof of zero', affine['zero_counts'])
        self.assertIn('no numerical bound', affine['nondecomposable'])
        self.assertIn('not a hardware-time or latency bound', affine['interpretation'])
        self.assertIn('not a term', affine['frame'])

    def test_fixed_k4_budget_cannot_be_inflated_or_shrunk(self):
        for k in (2, 3, 5, 6, True):
            configuration = self.configuration()
            configuration['archive_capacity'] = k
            with self.assertRaises(ValueError):
                fixture.protocol(configuration)


if __name__ == '__main__':
    unittest.main()
