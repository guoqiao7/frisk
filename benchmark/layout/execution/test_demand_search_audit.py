import copy
import json
from pathlib import Path
import tempfile
import unittest
import demand_search_audit as audit

class DemandSearchAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return str(path)
    def record(self, probes=12):
        return dict(status='success', legal_return=True, cost_model='test-model',
            total=dict(probes=probes, refinement=1, relationship=probes-3, complete_scores=2, refinement_probes=1),
            budget_contract=dict(kind='shared-total-search-probes-v1', search_performed=True,
                limit=65536, refinement_limit=256, used=probes, refinement_used=1),
            solve_timing_ns=dict(total=400, stage1=0, stage2=0, incumbent_full_score=0),
            search_components=[dict(seed_probes=1, seed_scores=1)], cost=dict(cost=[10]*9, saturated=False))
    def row(self, arm, model='alpha', search='oldbeam', split='holdout'):
        path=self.root/arm; record=self.record(); record['cost_model']=model
        self.write(path/'record.json', record)
        metadata=self.write(path/'native.log.command.json', dict(elapsed_ns=1000, exit_code=0, timeout_expired=False))
        return dict(spec=dict(key='case', split=split), scope='natural', arm=arm, model=model, selection='J',
            treatment=dict(arm=arm, model=model, selection='J', search=search, incumbent_witness=search=='witness'),
            path=str(path), phases=[dict(stage='native-selection', command_metadata=metadata)])
    def test_four_search_arms_separate_model_cost_scales_and_cpu_metrics(self):
        rows=[self.row(model+'-'+search, model, search) for model in ('alpha','beta') for search in ('oldbeam','witness')]
        result=audit.summarize_search(rows)
        self.assertEqual(len(result['observations']), 4)
        pairs=result['pairwise_case_comparisons']; self.assertEqual(len(pairs), 6)
        self.assertEqual(sum(x['same_objective'] for x in pairs), 2)
        for item in pairs:
            self.assertEqual(item['metrics']['native_command_elapsed_ns'], dict(delta=0, ratio=1.))
            self.assertEqual(item['first_cost_component']['ratio'], 1. if item['same_objective'] else None)
    def test_checker_and_aggregates_keep_uninstrumented_seed_time_unknown(self):
        calls=[]; row=self.row('witness', search='witness')
        result=audit.summarize_search([row], lambda record,treatment:calls.append(treatment['arm']))
        self.assertEqual(calls,['witness']); self.assertIsNone(result['observations'][0]['seed_elapsed_ns'])
        self.assertEqual(result['by_arm_scope_split'][0]['seed_work'],2)
        self.assertIn('not an old historical executable',' '.join(result['limitations']))
    def failure(self, context=True, searched=True, probes=12):
        path=self.root/'failure';record=self.record(probes)
        if not searched:
            record['total']={k:0 for k in record['total']}
            record['budget_contract'].update(search_performed=False,used=0,refinement_used=0)
        envelope=dict(status='failed', failure_kind='unsupported', fatal=False, reason='unknown domain', limits=dict(probes=65536,refinement=256))
        record.update(envelope,budget_context_available=context)
        self.write(path/'failure.json',envelope);self.write(path/'failure-search-record.json',record)
        self.write(path/'failure-budget.json',dict(schema_version=1,budget_context_available=context,total=record['total'],budget_contract=record['budget_contract']))
        return path
    def test_failure_sidecars_validate_nonzero_shared_accounting(self):
        path=self.failure(); proof=audit.validate_failure_sidecars(path)
        self.assertTrue(proof['validated']);self.assertEqual(proof['total']['probes'],12)
        value=json.loads((path/'failure-budget.json').read_text());value['total']['relationship']+=1
        self.write(path/'failure-budget.json',value)
        with self.assertRaisesRegex(ValueError,'decomposition'):audit.validate_failure_sidecars(path)
    def test_presearch_failure_cannot_invent_budget_context_or_work(self):
        path=self.failure(context=False,searched=False)
        self.assertTrue(audit.validate_failure_sidecars(path)['validated'])
        path=self.failure(context=False,searched=True)
        with self.assertRaisesRegex(ValueError,'presearch'):audit.validate_failure_sidecars(path)
    def test_old_failure_envelope_must_not_gain_fields(self):
        path=self.failure();envelope=json.loads((path/'failure.json').read_text());envelope['total']={}
        self.write(path/'failure.json',envelope)
        with self.assertRaisesRegex(ValueError,'envelope changed'):audit.validate_failure_sidecars(path)
    def test_duplicate_observations_and_missing_timing_are_rejected(self):
        row=self.row('same')
        with self.assertRaisesRegex(ValueError,'duplicate'):audit.summarize_search([row,row])
        self.write(Path(row['phases'][0]['command_metadata']), dict(exit_code=0))
        with self.assertRaisesRegex(ValueError,'elapsed_ns'):audit.summarize_search([row])
    def test_invalid_success_probe_decomposition_is_not_hidden(self):
        row=self.row('bad');record=self.record();record['total']['relationship']+=1
        self.write(Path(row['path'])/'record.json',record)
        with self.assertRaisesRegex(ValueError,'decomposition'):audit.summarize_search([row])

if __name__=='__main__':
    unittest.main()
