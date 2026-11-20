"""CPU-only Task 2 orchestration integration tests; GPU/compiler are mocked."""
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pipeline_overhead as task


class EpochTests(unittest.TestCase):
    def test_nested_epoch_restores_all_shared_globals_after_failure(self):
        original=(task.base.STAGE,task.adapter.BASE,task.adapter.DATA,task.base.engine.suite,task.base.engine.BASE)
        with self.assertRaisesRegex(RuntimeError,'scope test'):
            with task.epoch():
                self.assertEqual(task.base.STAGE,task.STAGE)
                self.assertEqual(task.adapter.BASE,task.HOME)
                self.assertEqual(task.adapter.DATA,task.DATA)
                self.assertEqual(task.base.engine.suite.STAGE,task.STAGE)
                self.assertEqual(task.base.engine.BASE,task.HOME)
                with task.epoch():self.assertEqual(task.base.engine.suite.STAGE,task.STAGE)
                raise RuntimeError('scope test')
        self.assertEqual((task.base.STAGE,task.adapter.BASE,task.adapter.DATA,task.base.engine.suite,task.base.engine.BASE),original)

    def test_empty_matrix_cannot_freeze_tools_or_start_gpu(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);matrix=root/'matrix.json';matrix.write_text('{"schema_version":1,"cases":[]}')
            with patch.object(task,'sha',return_value=task.TASK1_SHA), \
                 patch.object(task.base.engine.closure,'frozen_tools',side_effect=AssertionError('tools/GPU must not run')):
                with self.assertRaisesRegex(ValueError,'matrix required'):
                    task.init(root/'run',matrix,model_bundle=root/'unused',cache_root=root/'cache')
            self.assertFalse((root/'run').exists())


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name);self.stack=ExitStack()
        self.static=dict(schema_version=1,model='frozen-static')
        known_index=self.root/'known-index.json';known_index.write_text('{}')
        (self.root/'static-network-proxy-profile.json').write_text(json.dumps(self.static))
        self.prepared=self.root/'reference-prepared';self.prepared.mkdir()
        self.meta=dict(package_digest='package',common_digest='common',domain_digest='domain',coverage_digest='coverage')
        for file,value in [('prepared.json',self.meta),('domains.json',{})]:(self.prepared/file).write_text(json.dumps(value))
        (self.prepared/'input.mlir').write_text('module {}')
        self.spec=dict(key='known-attention',case_id='attention',shape=[64,16],threads=128,dtype='float16',split='observed-anchor',role='old')
        self.cases=[]
        for mode in ('J','LI'):
            search=self.root/('reference-search-'+mode);search.mkdir()
            for file in ('record.json','candidate-archive.json'):(search/file).write_text('{}')
            for file in ('prepared.mlir','materialized.mlir','output.mlir','reparsed.mlir','execution.json'):(search/file).write_text('same')
            self.cases.append(dict(spec=self.spec,prepared=str(self.prepared),search=str(search),mode=mode))
        self.matrix=self.root/'matrix.json';self.matrix.write_text(json.dumps(dict(schema_version=1,cases=self.cases)))
        profile=self.root/'model.json';profile.write_text('{}')
        self.bundle=self.root/'bundle.json';self.bundle.write_text(json.dumps(dict(profiles=dict(native=dict(path=str(profile))))))
        self.seal_path=self.root/'seal-index.json';self.seal_path.write_text('{}')
        self.local_profile=self.root/'local-profile.json';self.local_profile.write_text('{\"test\":\"local\"}')
        self.local_dependency=self.root/'local-dependency.json';self.local_dependency.write_text('dependency')
        self.profile_loader=self.stack.enter_context(patch('compatible_native_profiles.load_profile',return_value={}))
        self.dependency_loader=self.stack.enter_context(patch('compatible_native_profiles.dependency_hashes',return_value={str(self.local_dependency):hashlib.sha256(self.local_dependency.read_bytes()).hexdigest()}))
        outer=self
        class Seal:
            path=outer.seal_path
            def read(self,path):
                if Path(path).name=='manifest.json':return dict(rows=[dict(spec=outer.spec,prepared=str(outer.prepared),model='surviving')])
                return json.loads(Path(path).read_text())
            def check(self,path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.stack.enter_context(patch.object(task.adapter,'SealedInputs',Seal))
        self.stack.enter_context(patch.object(task.adapter,'KNOWN_INDEX',known_index))
        self.stack.enter_context(patch.object(task,'source_identity',return_value=dict(source_manifest={'pipeline.py':'frozen'},source_digest='snapshot')))
        self.stack.enter_context(patch.object(task.base.engine.closure,'frozen_tools',return_value=dict(hashes={},environment={})))
        actual_sha=task.sha
        self.stack.enter_context(patch.object(task,'sha',side_effect=lambda path:task.TASK1_SHA if Path(path)==task.TASK1_INDEX else actual_sha(path) if Path(path).is_file() else 'tool-sha'))
        self.stack.enter_context(patch.object(task.base,'_checked_archive',side_effect=lambda item:(dict(cost_profile=outer.static,prepared_digest='package'),dict(entries=[dict(index=i) for i in range(4 if item['mode']=='J' else 3)]),item['mode'])))
        self.stack.enter_context(patch.object(task,'HOME',self.root/'home'))
        self.stack.enter_context(patch.object(task,'DATA',self.root/'data'))
    def tearDown(self):self.stack.close();self.temporary.cleanup()
    def test_init_preregisters_planned_paths_exact_counts_providers_and_inputs(self):
        run=self.root/'data'/'run';protocol=task.init(run,self.matrix,model_bundle=self.bundle,cache_root=self.root/'cache',local_profile=self.local_profile)
        self.assertEqual(protocol['native_budget'],dict(capacity_per_pool=4,pools=2,requests=7))
        self.assertFalse(protocol['qualification_pass_claim']);self.assertEqual(protocol['role'],'known-development-not-fresh')
        self.assertEqual(protocol['generation_execution'],task._generation_contract())
        cfg=protocol['validation_execution']
        self.assertEqual(cfg['inventory_verifier'],'pipeline_overhead:verify_inventory')
        self.assertEqual(cfg['numerical_fixture'],'network_selection_fixture')
        self.assertEqual(cfg['contract'],'frisk-five-process-validation-batch-v1')
        self.assertEqual({case['mode'] for case in protocol['matrix']},{'J','LI'})
        for case in protocol['matrix']:
            self.assertTrue(Path(case['prepared']).is_relative_to(run));self.assertTrue(Path(case['search']).is_relative_to(run))
            self.assertEqual(case['spec']['role'],'known-overhead-regression')
            self.assertEqual(case['spec']['split'],'development');self.assertEqual(case['original_spec'],self.spec)
            self.assertEqual(case['reference_package_digest'],'package')
        self.assertEqual(protocol['matrix'][0]['prepared'],protocol['matrix'][1]['prepared'])
        self.assertNotEqual(protocol['matrix'][0]['search'],protocol['matrix'][1]['search'])
        registration=task.read(run/'registration.json');manifest=task.read(run/'manifest.json')
        self.assertEqual(manifest['registration_sha256'],task.sha(run/'registration.json'))
        self.assertEqual(registration['protocol_digest'],task.numerical.digest(protocol))
        self.assertEqual(registration['source']['source_digest'],'snapshot')
        self.assertIn(str(self.matrix),registration['immutable_inputs']);self.assertIn(str(self.bundle),registration['immutable_inputs'])
        self.assertEqual(protocol['local_profile'],str(self.local_profile))
        self.assertEqual(registration['immutable_inputs'][str(self.local_profile)],task.sha(self.local_profile))
        self.assertEqual(registration['immutable_inputs'][str(self.local_dependency)],task.sha(self.local_dependency))
        self.profile_loader.assert_called_once_with(self.local_profile)
        self.dependency_loader.assert_called_once_with(self.local_profile)
        for case in self.cases:
            for name in ('record.json','candidate-archive.json','prepared.mlir','materialized.mlir','output.mlir','reparsed.mlir','execution.json'):
                self.assertEqual(registration['immutable_inputs'][str(Path(case['search'])/name)],task.sha(Path(case['search'])/name))
        self.assertEqual(manifest['status'],'registered');self.assertFalse((run/'preparation').exists())
    def test_duplicate_or_unknown_program_rejected_before_run_creation(self):
        for cases in (self.cases+[self.cases[0]],[dict(self.cases[0],spec=dict(self.spec,key='unknown'))]):
            self.matrix.write_text(json.dumps(dict(schema_version=1,cases=cases)))
            with self.assertRaisesRegex(ValueError,'unknown/duplicate'):
                task.init(self.root/'data'/'bad',self.matrix,model_bundle=self.bundle,cache_root=self.root/'cache',local_profile=self.local_profile)
            self.assertFalse((self.root/'data'/'bad').exists())

    def test_invalid_local_dependency_aborts_before_fresh_output_or_tools(self):
        self.dependency_loader.side_effect=ValueError('strict LOCAL dependency changed')
        with patch.object(task.base.engine.closure,'frozen_tools',side_effect=AssertionError('no tool freeze')):
            with self.assertRaisesRegex(ValueError,'LOCAL dependency changed'):
                task.init(self.root/'data'/'invalid',self.matrix,model_bundle=self.bundle,cache_root=self.root/'cache',local_profile=self.local_profile)
        self.assertFalse((self.root/'data'/'invalid').exists())


class StageTests(unittest.TestCase):
    def test_generate_requires_completed_fresh_search_before_engine(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);task.write(run/'manifest.json',dict(status='registered',rows=[]))
            with patch.object(task.base,'generate') as generate:
                with self.assertRaisesRegex(ValueError,'preparation/search'):task.generate(run)
                generate.assert_not_called()
            task.write(run/'manifest.json',dict(status='registered',rows=[],prepared_and_searched=True))
            with patch.object(task.base,'generate',return_value={'status':'generated'}) as generate:
                task.write(run/'protocol.json',dict(generation_execution=task._generation_contract()))
                self.assertEqual(task.generate(run),{'status':'generated'})
                generate.assert_called_once_with(run)

    def test_prepare_search_deduplicates_preparation_but_retains_modes_and_receipts(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);reference=run/'reference';reference.mkdir();(reference/'input.mlir').write_text('module {}')
            meta=dict(package_digest='package',common_digest='c',domain_digest='d',coverage_digest='v')
            cases=[]
            for mode in ('J','LI'):
                old=run/('reference-search-'+mode);old.mkdir()
                task.write(old/'record.json',dict(selection=mode,solve_timing_ns=1))
                task.write(old/'candidate-archive.json',dict(entries=[],search=dict(total_wall_ns=1)))
                for file in ('prepared.mlir','materialized.mlir','output.mlir','reparsed.mlir','execution.json'):(old/file).write_text('same')
                cases.append(dict(spec=dict(key='known',case_id='diamond'),mode=mode,prepared=str(run/'preparation/known/prepared'),
                    search=str(run/'search'/mode),reference_prepared=str(reference),reference_search=str(old),reference_package_digest='package',
                    reference_domain={k:meta[k] for k in ('common_digest','domain_digest','coverage_digest')}))
            protocol=dict(matrix=cases,configuration=dict(static=dict(profile='static.json')))
            task.write(run/'registration.json',dict(immutable_inputs={str(Path(case['reference_search'])/'record.json'):task.sha(Path(case['reference_search'])/'record.json') for case in cases}))
            task.write(run/'manifest.json',dict(status='registered',rows=[]))
            calls=[]
            def invoke(argv,log):
                calls.append(argv)
                if '--prepare' in argv:
                    target=Path(argv[4]);target.mkdir();task.write(target/'prepared.json',meta)
                else:
                    target=Path(next(a for a in argv if a.startswith('--output-dir=')).split('=',1)[1]);target.mkdir(parents=True)
                    mode=next(a for a in argv if a.startswith('--selection=')).split('=',1)[1]
                    task.write(target/'record.json',dict(selection=mode,solve_timing_ns=99))
                    task.write(target/'candidate-archive.json',dict(entries=[],search=dict(total_wall_ns=99)))
                    for file in ('prepared.mlir','materialized.mlir','output.mlir','reparsed.mlir','execution.json'):(target/file).write_text('same')
                return dict(exit_code=0,log=str(log))
            with patch.object(task,'verify',return_value=protocol),patch.object(task.base.engine.demand,'invoke',side_effect=invoke), \
                 patch.object(task.archive,'search_args',return_value=['--registered-search']), \
                 patch.object(task.base,'_checked_archive',side_effect=lambda c:(task.read(Path(c['search'])/'record.json'),task.read(Path(c['search'])/'candidate-archive.json'),c['mode'])):
                receipts=task.prepare_search(run)
            self.assertEqual([r['stage'] for r in receipts],['prepare','search','search'])
            self.assertEqual(len(calls),3);self.assertEqual(sum('--prepare' in call for call in calls),1)
            self.assertEqual({r['mode'] for r in receipts if r['stage']=='search'},{'J','LI'})
            self.assertTrue(task.read(run/'manifest.json')['prepared_and_searched'])
            self.assertTrue((run/'preparation-search-receipts.json').is_file())

    def test_reference_paid_audit_flag_replays_presence_and_absence_exactly(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);reference=run/'reference';reference.mkdir();path=reference/'record.json'
            case=dict(reference_search=str(reference))
            for present in (False,True):
                search=dict(probes=65536)
                if present:search['paid_observation_audit']={}
                task.write(path,dict(candidate_archive_search=search))
                task.write(run/'registration.json',dict(immutable_inputs={str(path):task.sha(path)}))
                self.assertEqual(task.reference_audit_args(run,case),['--archive-audit-paid-observations=true'] if present else [])

    def test_reference_paid_audit_flag_rejects_unregistered_or_mutated_record(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);reference=run/'reference';reference.mkdir();path=reference/'record.json'
            case=dict(reference_search=str(reference));task.write(path,dict(candidate_archive_search={}))
            task.write(run/'registration.json',dict(immutable_inputs={}))
            with self.assertRaisesRegex(ValueError,'not fully registered'):task.reference_audit_args(run,case)
            task.write(run/'registration.json',dict(immutable_inputs={str(path):task.sha(path)}))
            task.write(path,dict(candidate_archive_search=dict(paid_observation_audit={})))
            with self.assertRaisesRegex(ValueError,'content differs'):task.reference_audit_args(run,case)

    def test_failed_compile_retains_receipt_and_never_marks_compiled(self):
        import parallel_compile_batch
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);task.write(run/'manifest.json',dict(status='generated',rows=[dict(path=str(run/'cases/row'))]))
            task.write(run/'registration.json',dict(source={},tools={}))
            protocol=dict(cache_root=str(run/'cache'),native_budget=dict(requests=1))
            failed=dict(status='failed',request_cap=1,succeeded_requests=0,failed_requests=1,not_attempted_requests=0,descendants_reaped=True)
            with patch.object(task,'verify_inventory',return_value=protocol), \
                 patch.object(parallel_compile_batch,'compile_many_parallel',return_value=failed):
                with self.assertRaisesRegex(ValueError,'budget/compile failure'):task.compile_batch(run)
            self.assertEqual(task.read(run/'batch-compilation.json')['status'],'failed')
            self.assertNotIn('compiled',task.read(run/'manifest.json'))

    def test_batch_and_serial_validation_use_explicit_registered_branches(self):
        import validation_batch
        protocol=dict(cache_root='/task2/cache',batched_validation=True)
        with patch.object(task,'verify_inventory',return_value=protocol), \
             patch.object(validation_batch,'validate',return_value='batch') as batched:
            self.assertEqual(task.validate('/task2/run'),'batch')
            batched.assert_called_once_with(Path('/task2/run'),safety_cache_dir=Path('/task2/cache/safety'),output=Path('/task2/run/batched-validation'))
        protocol['batched_validation']=False
        with patch.object(task,'verify_inventory',return_value=protocol), \
             patch.object(task.base.engine.quality,'validate',return_value='serial') as serial:
            self.assertEqual(task.validate('/task2/run'),'serial')
            serial.assert_called_once_with(Path('/task2/run'),worker_script=task.__file__)

    def test_inventory_delegation_preserves_sanitizer_and_artifact_flags(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);receipt=run/'preparation-search-inputs.json';task.write(receipt,{})
            task.write(run/'manifest.json',dict(prepared_and_searched=True,preparation_search_inputs_sha256=task.sha(receipt)))
            protocol=dict(stage=task.STAGE,matrix=[])
            with patch.object(task.base,'verify_inventory',return_value=protocol) as verify:
                self.assertEqual(task.verify_inventory(run,check_artifacts=False,sanitizer=True),protocol)
                verify.assert_called_once_with(run,check_artifacts=False,sanitizer=True)

    def test_inventory_rejects_receipt_and_generated_dependency_mutations(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);dependency=run/'generated.json';dependency.write_text('original')
            receipt=run/'preparation-search-inputs.json';task.write(receipt,{str(dependency):task.sha(dependency)})
            manifest=dict(prepared_and_searched=True,preparation_search_inputs_sha256=task.sha(receipt));task.write(run/'manifest.json',manifest)
            with patch.object(task.base,'verify_inventory',return_value=dict(matrix=[])):
                task.verify_inventory(run)
                dependency.write_text('modified')
                with self.assertRaisesRegex(ValueError,'content differs'):task.verify_inventory(run)
                dependency.write_text('original');task.write(receipt,{})
                with self.assertRaisesRegex(ValueError,'receipt differs'):task.verify_inventory(run)

    def test_inventory_recompares_full_archive_ignoring_only_named_clocks(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);reference=run/'reference';reference.mkdir()
            record=dict(selection='J',solve_timing_ns=1,cost=dict(cost=[7],features={'memory':9}))
            pool=dict(entries=[dict(index=0)],search=dict(total_wall_ns=1,probes=65536))
            task.write(reference/'record.json',record);task.write(reference/'candidate-archive.json',pool)
            receipt=run/'preparation-search-inputs.json';task.write(receipt,{})
            task.write(run/'manifest.json',dict(prepared_and_searched=True,preparation_search_inputs_sha256=task.sha(receipt)))
            protocol=dict(matrix=[dict(reference_search=str(reference))])
            fresh=copy.deepcopy(record);fresh['solve_timing_ns']=999
            newpool=copy.deepcopy(pool);newpool['search']['total_wall_ns']=999
            with patch.object(task.base,'verify_inventory',return_value=protocol), \
                 patch.object(task.base,'_checked_archive',side_effect=lambda case:(fresh,newpool,'J')):
                task.verify_inventory(run)
                fresh['cost']['features']['memory']=10
                with self.assertRaisesRegex(ValueError,'archive differs'):task.verify_inventory(run)
                fresh['cost']['features']['memory']=9;newpool['search']['probes']=65535
                with self.assertRaisesRegex(ValueError,'archive differs'):task.verify_inventory(run)

    def test_validated_row_callback_propagates_registered_sanitizer_flag(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);case=run/'cases'/'0';case.mkdir(parents=True)
            generation=case/'generation-inputs.json';task.write(generation,{})
            row=dict(path=str(case),generation_inputs_sha256=task.sha(generation),status='validated',spec=dict(key='known'),arm='joint-c0',archive_entry={})
            task.write(case/'record.json',{})
            task.write(case/'configuration.json',dict(spec=row['spec'],scope='natural',arm=row['arm'],model='surviving',entry='frisk_execution',grid=[1,1,1],threads=128))
            receipt=run/'preparation-search-inputs.json';task.write(receipt,{})
            manifest=dict(rows=[row],prepared_and_searched=True,preparation_search_inputs_sha256=task.sha(receipt))
            task.write(run/'manifest.json',manifest)
            protocol=dict(stage=task.STAGE,matrix=[],native_budget=dict(requests=1));task.write(run/'protocol.json',protocol)
            registration=dict(protocol_digest=task.numerical.digest(protocol))
            with patch.object(task.base.engine.closure,'verify_run',return_value=(manifest,registration)), \
                 patch.object(task.base,'expected_inventory',return_value={'known':{}}), \
                 patch.object(task.base,'check_inventory_rows',return_value=True), \
                 patch.object(task.archive,'check_replay'), \
                 patch.object(task.base.engine.closure,'verify_accepted_row',return_value={}) as accepted:
                task.verify_inventory(run,sanitizer=True)
                accepted.assert_called_once_with(case,row,sanitizer=True)




class GenerationProviderProtocolTests(unittest.TestCase):
    def test_missing_or_tampered_generation_contract_fails_before_provider(self):
        from emission_audit_session import current_factory
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);task.write(run/'manifest.json',dict(prepared_and_searched=True))
            for protocol in [{},dict(generation_execution=dict(task._generation_contract(),persistent_cache=True)),
                             dict(generation_execution=dict(task._generation_contract(),independent_audit_predicates=1))]:
                task.write(run/'protocol.json',protocol)
                with patch.object(task.base,'generate',side_effect=AssertionError('must not start generation')):
                    with self.assertRaisesRegex(ValueError,'generation execution contract differs'):task.generate(run)
                self.assertIsNone(current_factory())
    def test_explicit_generation_scope_restores_factory_after_success_and_error(self):
        from emission_audit_session import current_factory,EmissionSession
        from integrity_session import active
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);task.write(run/'manifest.json',dict(prepared_and_searched=True));task.write(run/'protocol.json',dict(generation_execution=task._generation_contract()))
            def generate(path):
                self.assertEqual(path,run);self.assertIs(current_factory(),EmissionSession);self.assertIsNotNone(active());return 'checked'
            with patch.object(task.base,'generate',side_effect=generate):self.assertEqual(task.generate(run),'checked')
            self.assertIsNone(current_factory());self.assertIsNone(active())
            with patch.object(task.base,'generate',side_effect=RuntimeError('generation failed')):
                with self.assertRaisesRegex(RuntimeError,'generation failed'):task.generate(run)
            self.assertIsNone(current_factory());self.assertIsNone(active())

if __name__=='__main__':unittest.main()
