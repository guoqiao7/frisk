"""CPU-only native replay/domain and failed-search ledger regressions."""
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
BENCH = Path(os.environ.get('FRISK_JOINT_ABLATION_BENCH', ROOT / 'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench'))

@unittest.skipUnless(BENCH.is_file(), 'native joint ablation benchmark is not built')
class NativeFailureBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='frisk-native-failure-')
        cls.base = Path(cls.temp.name)
        from calibration_fixture import fixture
        cls.source = fixture(dict(case_id='no-conflict', shape=[64, 16], tail='none'))
        cls.baseline = cls.prepare('baseline', cls.source)
        cls.baseline_output = cls.base / 'baseline-output'
        cls.invoke(['--prepared=' + str(cls.baseline), '--selection=J', '--conversion=C',
                    '--seed-beam-solution=false', '--output-dir=' + str(cls.baseline_output)], 0)
        cls.assignment = json.loads((cls.baseline_output / 'record.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def invoke(cls, args, code):
        result = subprocess.run([str(BENCH), *map(str, args)], text=True, capture_output=True, timeout=120)
        if result.returncode != code:
            raise AssertionError(f'expected exit {code}, got {result.returncode}: {result.stdout} {result.stderr}')
        return result

    @classmethod
    def prepare(cls, name, text):
        raw = cls.base / (name + '.mlir'); raw.write_text(text)
        prepared = cls.base / (name + '-prepared')
        # Controlled preparation preserves deliberately unknown domain metadata;
        # ordinary normalization correctly rejects it before selection.
        cls.invoke(['--prepare', raw, 'no-conflict', prepared], 0)
        return prepared

    def tagged(self, name, attribute):
        text = self.source.replace('module attributes {', 'module attributes {frisk.execution_domain=' + attribute + ', ', 1)
        return self.prepare(name, text)

    def failure(self, output, kind, context, searched):
        envelope = json.loads((output / 'failure.json').read_text())
        self.assertEqual(set(envelope), {'status', 'failure_kind', 'fatal', 'reason', 'limits'})
        self.assertEqual(envelope['failure_kind'], kind)
        budget = json.loads((output / 'failure-budget.json').read_text())
        record = json.loads((output / 'failure-search-record.json').read_text())
        self.assertEqual(budget['schema_version'], 1)
        self.assertIs(budget['budget_context_available'], context)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['failure_kind'], kind)
        self.assertEqual(record['total'], budget['total'])
        self.assertEqual(record['budget_contract'], budget['budget_contract'])
        total = budget['total']; contract = budget['budget_contract']
        self.assertIs(contract['search_performed'], searched)
        self.assertLessEqual(total['probes'], 65536)
        self.assertLessEqual(total['refinement'], 256)
        self.assertEqual(total['probes'], sum(total[k] for k in ('relationship', 'complete_scores', 'refinement_probes')))
        self.assertEqual(total['refinement'], total['refinement_probes'])
        self.assertEqual(contract['used'], total['probes'])
        self.assertEqual(contract['refinement_used'], total['refinement'])
        return total, record

    def test_unknown_domain_search_preserves_nonzero_probe_ledger(self):
        prepared = self.tagged('unknown-search', '"future-domain"')
        output = self.base / 'unknown-search-output'
        self.invoke(['--prepared=' + str(prepared), '--selection=J', '--conversion=C', '--output-dir=' + str(output)], 3)
        total, record = self.failure(output, 'unsupported', True, True)
        self.assertGreater(total['probes'], 0)
        self.assertFalse(record['legal_return'])
        self.assertEqual(record['selection'], 'J')
        self.assertIn('prepared_digest', record)

    def replay(self, name, prepared, expected, complete_record=None):
        source = self.assignment if complete_record is None else complete_record
        assignment = {k: source[k] for k in ('assignment', 'instruction_bindings', 'reduction_bindings', 'conversion_policy', 'complete_assignment_sha256')}
        assignment['prepared_digest'] = json.loads((prepared / 'prepared.json').read_text())['package_digest']
        path = self.base / (name + '-assignment.json'); path.write_text(json.dumps(assignment))
        output = self.base / (name + '-output')
        self.invoke(['--prepared=' + str(prepared), '--assignment=' + str(path), '--selection=J', '--conversion=C', '--output-dir=' + str(output)], 3)
        total, _ = self.failure(output, expected, True, False)
        self.assertEqual(total['probes'], 0)

    def test_unknown_domain_replay_is_typed_unsupported_before_scoring(self):
        self.replay('unknown-replay', self.tagged('unknown-replay', '"future-domain"'), 'unsupported')

    def test_nonstring_domain_replay_is_typed_unsupported(self):
        self.replay('nonstring-replay', self.tagged('nonstring-replay', '17 : i64'), 'unsupported')

    def test_disproven_domain_replay_is_typed_no_feasible(self):
        # Existing f32 storage remains legal for the carrier scorer. The
        # explicit v1 executor ABI additionally requires four alignment1 roots.
        strong = re.sub(r'alignment\s*=\s*1,', 'alignment=2,', self.source)
        self.assertNotEqual(strong, self.source)
        baseline = self.prepare('strong-baseline', strong)
        output = self.base / 'strong-baseline-output'
        self.invoke(['--prepared=' + str(baseline), '--selection=J', '--conversion=C', '--output-dir=' + str(output)], 0)
        assignment = json.loads((output / 'record.json').read_text())
        raw = strong.replace('module attributes {', 'module attributes {frisk.execution_domain="sm90-scalar-f32-dag-v1", ', 1)
        self.replay('disproven-replay', self.prepare('disproven-replay', raw), 'no-feasible', assignment)

    def test_presearch_failure_reports_absent_budget_context(self):
        output = self.base / 'missing-prepared-output'
        self.invoke(['--prepared=' + str(self.base / 'missing'), '--output-dir=' + str(output)], 2)
        total, _ = self.failure(output, 'fatal-integrity', False, False)
        self.assertEqual(total['probes'], 0)

if __name__ == '__main__':
    unittest.main()
