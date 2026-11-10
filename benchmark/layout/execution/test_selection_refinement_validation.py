import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import selection_refinement_validation as validation


class RefinementValidationTest(unittest.TestCase):
    def test_epoch_restores_shared_engine_after_exception(self):
        original = validation.engine.suite, validation.engine.BASE
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with validation.epoch():
                self.assertEqual(validation.engine.suite.STAGE, validation.STAGE)
                raise RuntimeError('test')
        self.assertEqual((validation.engine.suite, validation.engine.BASE), original)

    def test_invalid_matrix_cannot_register_or_initialize_gpu(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); matrix = root / 'matrix.json'
            matrix.write_text(json.dumps(dict(schema_version=1, cases=[])))
            with patch.object(validation.adapter, 'SealedInputs', side_effect=AssertionError('unexpected seal/registration')):
                with self.assertRaises(ValueError): validation.init(root / 'run', matrix)
            self.assertFalse((root / 'run').exists())

    def test_execution_reuses_accepted_numerical_and_exact_compilation_provider(self):
        import native_compile_cache
        with patch.object(validation, 'verify_inventory', return_value={}), \
             patch.object(validation.engine.closure, 'execute', return_value={'status': 'passed'}) as call:
            result = validation.execute(Path('/new/cases/unused'), True, True, 'memcheck')
        self.assertEqual(result['status'], 'passed')
        self.assertIs(call.call_args.kwargs['numerical_fixture'], validation.numerical)
        self.assertIs(call.call_args.kwargs['compiler_provider'], native_compile_cache.verify_precompiled)
        self.assertEqual(call.call_args.kwargs['safety_tool'], 'memcheck')

    def test_inventory_rejects_same_count_program_replacement_and_duplicates(self):
        spec = {'key': 'known', 'shape': [16,16]}
        entry = dict(complete_assignment_sha256='a'*64, complete_program_sha256='b'*64,
                     cost={'cost': [1]}, cost_model='model')
        witness = dict(spec=spec, arm='joint-c0', prepared='/sealed/prepared', path='/new/cases/known-joint-c0',
                       prepared_sha256='c'*64, domain={'domain_digest':'d'}, entry=entry)
        key = (validation.numerical.digest(spec), 'joint-c0')
        row = dict(witness, archive_entry=entry, assignment_sha256='a'*64, complete_program_sha256='b'*64,
                   cost=entry['cost'], model_id='model', selection='J', conversion='C', model='surviving',
                   scope='natural', status='generated')
        row.pop('entry')
        self.assertTrue(validation.check_inventory_rows([row], {key:witness}))
        for changed in [dict(row, path='/new/cases/replacement'), dict(row, complete_program_sha256='f'*64),
                        dict(row, prepared='/other'), dict(row, arm='joint-c1')]:
            with self.assertRaises(ValueError): validation.check_inventory_rows([changed], {key:witness})
        with self.assertRaises(ValueError): validation.check_inventory_rows([row,row], {key:witness})

    def test_sanitizer_inventory_uses_registered_sanitizer_environment(self):
        protocol = dict(stage=validation.STAGE, native_budget={'requests': 0})
        manifest = {'rows': []}; registration = {'protocol_digest': validation.numerical.digest(protocol)}
        with patch.object(validation.engine.closure, 'verify_run', return_value=(manifest, registration)) as verify, \
             patch.object(validation.engine, 'verify', side_effect=AssertionError('ordinary environment guard used')), \
             patch.object(validation.engine, 'read', return_value=protocol), \
             patch.object(validation, 'expected_inventory', return_value={}):
            self.assertEqual(validation.verify_inventory('/new/run', sanitizer=True), protocol)
        verify.assert_called_once_with(Path('/new/run'), sanitizer=True)
        with patch.object(validation.engine.closure, 'verify_run', return_value=(manifest, {'protocol_digest': 'wrong'})), \
             patch.object(validation.engine, 'read', return_value=protocol):
            with self.assertRaises(ValueError): validation.verify_inventory('/new/run', sanitizer=True)

    def test_generate_refuses_existing_generation_before_side_effect(self):
        original = validation.engine.suite
        with patch.object(validation.engine, 'verify', return_value={}), \
             patch.object(validation.engine, 'read', return_value={'status': 'generated'}):
            with self.assertRaises(ValueError): validation.generate('/unused')
        self.assertIs(validation.engine.suite, original)


class MatrixModeValidationTest(unittest.TestCase):
    def test_mode_defaults_and_source_archive_binding_are_strict(self):
        for case,mode,recipe in [({},'J',False),({'mode':'J'},'J',True),({'mode':'LI'},'LI',True)]:
            case=dict(case,search='/new/search');pool={'search':{'retention_resolution':{}} if recipe else {}}
            record={'selection':mode}
            with patch.object(validation.adapter,'read',side_effect=[record,pool]), \
                 patch.object(validation.archive,'check_archive',return_value=[]) as check:
                self.assertEqual(validation._checked_archive(case),(record,pool,mode))
            check.assert_called_once_with(record,pool,mode,4,
                expected_policy=validation.archive.ROUTING_POLICY if recipe else validation.archive.POLICY)
        for mode in ('L','joint',None,True):
            with self.subTest(mode=mode),self.assertRaises(ValueError):validation._case_mode({'mode':mode})
        with patch.object(validation.adapter,'read',side_effect=[{'selection':'J'},{'search':{}}]), \
             patch.object(validation.archive,'check_archive',side_effect=AssertionError('cross-mode archive checked')):
            with self.assertRaisesRegex(ValueError,'selection mode'):
                validation._checked_archive({'mode':'LI','search':'/new/search'})

    def test_generate_keeps_joint_and_local_pools_separate_and_replay_selection_J(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);spec=dict(key='same',threads=128)
            cases=[dict(spec=spec,prepared='/sealed/prepared',search='/search/'+mode,mode=mode) for mode in ('J','LI')]
            protocol=dict(matrix=cases,native_budget={'requests':8});manifest={'status':'registered','rows':[]}
            entries=[dict(index=i) for i in range(4)]
            def read(path):
                return manifest if Path(path)==run/'manifest.json' else {'package_digest':'p'}
            def add_case(run,manifest,spec,arm,model,prepared,meta,directory,**kwargs):
                row=dict(spec=spec,arm=arm,selection='J',path=str(directory))
                manifest['rows'].append(row);return row
            with patch.object(validation.engine,'verify',return_value=protocol), \
                 patch.object(validation.engine,'read',side_effect=read), \
                 patch.object(validation,'_checked_archive',side_effect=lambda c:({},dict(entries=entries),c['mode'])), \
                 patch.object(validation.archive,'replay_args',return_value=['--network-objective=min-shfl']), \
                 patch.object(validation.engine,'add_case',side_effect=add_case) as replay:
                result=validation.generate(run)
            self.assertEqual(result['status'],'generated');self.assertEqual(len(result['rows']),8)
            self.assertEqual({r['arm'] for r in result['rows']},
                {prefix+str(i) for prefix in ('joint-c','local-c') for i in range(4)})
            for row in result['rows']:
                self.assertEqual(row['selection'],'J')
                self.assertEqual(row['source_archive_mode'],'LI' if row['arm'].startswith('local-') else 'J')
            for call in replay.call_args_list:
                self.assertIs(call.kwargs['replay_validator'],validation.archive.check_replay)
                self.assertEqual(call.kwargs['replay_arguments'],['--network-objective=min-shfl'])
                self.assertNotIn('selection',call.kwargs)

    def test_inventory_binds_source_mode_separately_from_fixed_J_replay(self):
        spec={'key':'known','shape':[16,16]}
        entry=dict(complete_assignment_sha256='a'*64,complete_program_sha256='b'*64,
                   cost={'cost':[1]},cost_model='model')
        witness=dict(spec=spec,arm='local-c0',prepared='/sealed/prepared',path='/new/cases/known-local-c0',
                     prepared_sha256='c'*64,domain={'domain_digest':'d'},entry=entry,
                     source_archive_mode='LI',source_mode_metadata_required=True)
        key=(validation.numerical.digest(spec),'local-c0')
        row=dict(witness,archive_entry=entry,assignment_sha256='a'*64,complete_program_sha256='b'*64,
                 cost=entry['cost'],model_id='model',selection='J',conversion='C',model='surviving',
                 scope='natural',status='generated')
        row.pop('entry');row.pop('source_mode_metadata_required')
        self.assertTrue(validation.check_inventory_rows([row],{key:witness}))
        missing=dict(row);missing.pop('source_archive_mode')
        for changed in [missing,dict(row,source_archive_mode='J'),dict(row,selection='LI'),
                        dict(row,arm='joint-c0'),dict(row,path='/new/cases/known-joint-c0')]:
            with self.assertRaises(ValueError):validation.check_inventory_rows([changed],{key:witness})

    def test_expected_inventory_uses_registered_modes_and_disjoint_paths(self):
        spec={'key':'known','shape':[16,16]};meta={name:'d' for name in ('common_digest','domain_digest','coverage_digest')}
        cases=[dict(spec=spec,prepared='/sealed/prepared',search='/search/'+m,mode=m) for m in ('J','LI')]
        with patch.object(validation.engine,'read',return_value=meta), \
             patch.object(validation,'_checked_archive',side_effect=lambda c:({},dict(entries=[dict(index=i) for i in range(4)]),c['mode'])), \
             patch.object(validation.adapter,'sha',return_value='c'*64):
            expected=validation.expected_inventory('/new/run',dict(matrix=cases))
        self.assertEqual(len(expected),8);self.assertEqual(len({w['path'] for w in expected.values()}),8)
        for witness in expected.values():
            mode='LI' if witness['arm'].startswith('local-') else 'J'
            self.assertEqual(witness['source_archive_mode'],mode)
            self.assertIs(witness['source_mode_metadata_required'],True)


if __name__ == '__main__':
    unittest.main()
