import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import selection_refinement_decision as adapter


class RefinementAdapterTest(unittest.TestCase):
    def row(self, arm='joint-c0', **extra):
        return dict(arm=arm, status='validated', spec={'shape': [16, 16]}, path='/not-opened', **extra)

    def test_validation_precedes_image_reads(self):
        with patch.object(adapter.model, 'candidate_evidence', side_effect=AssertionError('unexpected image read')):
            for rows in [[], [self.row(str(i)) for i in range(5)],
                         [dict(self.row(), status='generated')],
                         [self.row(), self.row()],
                         [self.row(), dict(self.row('joint-c1'), spec={'shape': [32, 16]})]]:
                with self.assertRaises(ValueError):
                    adapter.select_validated_pool(rows, {}, 'joint-c0')
            with self.assertRaises(ValueError):
                adapter.select_validated_pool([self.row(), self.row('joint-c1')], {}, 'joint-c0', ['joint-c0', 'joint-c0'])

    def test_whole_pool_preserves_unknowns_and_registered_fallback(self):
        rows = [self.row('joint-c0'), self.row('joint-c2')]
        summaries = {r['arm']: {'status': 'complete'} for r in rows}
        localities = {r['arm']: {'status': 'unsupported'} for r in rows}
        decision = {'payload': {'selected_source': 'joint-c2', 'uses_gpu_timing': False}}
        with patch.object(adapter.model, 'candidate_evidence', side_effect=lambda path, row: {'arm': row['arm']}), \
             patch.object(adapter, 'read', return_value={'status': 'unsupported'}), \
             patch.object(adapter.model, 'dispatch', return_value=decision) as dispatch, \
             patch.object(adapter.model, 'verify_dispatch', return_value=decision) as verify:
            result, evidence = adapter.select_validated_pool(rows, {}, 'joint-c2',
                summaries=summaries, localities=localities)
        self.assertIs(result, decision)
        self.assertEqual(set(evidence), {'joint-c0', 'joint-c2'})
        self.assertEqual(dispatch.call_args.args[3:5], ('joint-c2', ['joint-c2', 'joint-c0']))
        self.assertIs(dispatch.call_args.kwargs['localities'], localities)
        self.assertFalse(dispatch.call_args.kwargs['allow_empirical_extrapolation'])
        verify.assert_called_once()

    def seal(self, root, rows):
        index = root / 'index.json'
        value = dict(roots={'data': dict(real_absolute_path=str(root), identity=dict(
            device=root.stat().st_dev, inode=root.stat().st_ino, uid=os.getuid()))}, files=rows)
        index.write_text(json.dumps(value))
        return index, adapter.sha(index)

    def test_seal_detects_equal_length_tamper_and_unlisted_input(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); target = root / 'proof.json'; target.write_text('good')
            index, digest = self.seal(root, [dict(root_id='data', path='proof.json', bytes=4, sha256=adapter.sha(target))])
            inputs = adapter.SealedInputs(index, digest)
            self.assertEqual(inputs.check(target), adapter.sha(target))
            target.write_text('evil')
            with self.assertRaises(ValueError): inputs.check(target)
            with self.assertRaises(ValueError): inputs.check(root / 'other.json')

    def test_seal_rejects_traversal_duplicate_and_symlink_alias(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); target = root / 'proof'; target.write_text('proof')
            row = dict(root_id='data', path='proof', bytes=5, sha256=adapter.sha(target))
            for rows in [[dict(row, path='../escape')], [row, row]]:
                index, digest = self.seal(root, rows)
                with self.assertRaises(ValueError): adapter.SealedInputs(index, digest)
            index, digest = self.seal(root, [row]); inputs = adapter.SealedInputs(index, digest)
            alias = root / 'alias'; alias.symlink_to(target)
            with self.assertRaises(ValueError): inputs.check(alias)

    def test_fresh_output_cannot_escape_new_stage_or_replace_attempt(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); base = root / 'new'; base.mkdir()
            with patch.object(adapter, 'BASE', base), patch.object(adapter, 'DATA', base):
                with self.assertRaises(ValueError): adapter._fresh_output(root / 'old')
                out = adapter._fresh_output(base / 'attempt')
                self.assertTrue(out.is_dir())
                with self.assertRaises(FileExistsError): adapter._fresh_output(out)
                alias = root / 'alias'; alias.symlink_to(base, target_is_directory=True)
                with self.assertRaises(ValueError): adapter._fresh_output(alias / 'escape')

    def test_manifest_inventory_rejects_extra_spec_and_duplicate(self):
        pool = dict(spec={'shape': [16, 16]}, legacy='legacy-static', modes={
            'joint': {'arms': ['joint-c0']}, 'local': {'arms': ['local-c0']}})
        rows = [self.row(arm) for arm in ('legacy-static', 'joint-c0', 'local-c0')]
        self.assertTrue(adapter.verify_manifest_inventory(rows, {'group': pool}))
        for invalid in [rows + [dict(self.row('extra'), spec={'shape': [32, 16]})], rows + [rows[0]], rows[:-1]]:
            with self.assertRaises(ValueError): adapter.verify_manifest_inventory(invalid, {'group': pool})
        with self.assertRaises(ValueError):
            adapter.verify_manifest_inventory(rows, {'group': dict(pool, modes=dict(pool['modes'], extra={'arms': []}))})

    def test_process_audit_rejects_update_mode_and_symlink_labels(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'measurement').mkdir()
            target = root / 'measurement/values.json'; target.write_text('label')
            alias = root / 'input.json'; alias.symlink_to(target)
            code = """import sys
from pathlib import Path
import selection_refinement_decision as a
root=Path(sys.argv[1]); target=root/'measurement/values.json'; alias=root/'input.json'
a.install_label_read_guard()
for path,mode in [(target,'r+'),(target,'a+'),(alias,'r')]:
    try: path.open(mode).read()
    except PermissionError: pass
    else: raise AssertionError(str(path))
print('blocked aliases and update modes')
"""
            process = subprocess.run([sys.executable, '-B', '-c', code, name], cwd=Path(adapter.__file__).parent,
                                     capture_output=True, text=True, check=True)
            self.assertIn('blocked aliases and update modes', process.stdout)

    def test_process_audit_blocks_performance_labels(self):
        code = """import selection_refinement_decision as a
reads=a.install_label_read_guard()
for name in ['group-quality.json','measurement/case.json','case-raw.json','training-ledger.json']:
    try: open('/missing/'+name).read()
    except PermissionError: pass
    else: raise AssertionError(name)
print('blocked all labels')
"""
        process = subprocess.run([sys.executable, '-B', '-c', code], cwd=Path(adapter.__file__).parent,
                                 capture_output=True, text=True, check=True)
        self.assertIn('blocked all labels', process.stdout)


if __name__ == '__main__':
    unittest.main()
