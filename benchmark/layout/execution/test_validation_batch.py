"""CPU-only adversarial contract tests; no compiler or CUDA initialization."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import validation_batch as batch


CFG=dict(contract=batch.CONTRACT,numerical_fixture='test_validation_batch',
         inventory_verifier='test_validation_batch:inventory',integrity_session=True,safety_cache=True)


def inventory(*args,**kwargs):pass


def normal_payload(row):
    inputs=[dict(case=name,input_offset=i,output_offset=i,input_hashes=['q','k','v'],reference_sha256='reference')
            for name in ('seed0','seed1') for i in (1,2,3)]
    identity=dict(spec=row['spec'],cases=[dict(name='seed0',input_hashes=['q','k','v'])],offsets=[[1,1],[2,2],[3,3]])
    return dict(status='passed',cubin_sha256='same-cubin',gpu_uuid='same-gpu',physical_params=['q','k','v','o'],
                physical_resources=dict(registers_per_thread=32,local_bytes=0,shared_bytes=64),safety_input_identity=identity,
                image_identity_sha256=batch.sha(Path(row['path'])/'image-identity.json'),inputs=inputs,
                records=[],execution_environment=dict(canonical=dict(CUDA_MODULE_LOADING='LAZY')))


def safety_payload(normal):
    result=copy.deepcopy(normal);result.update(safety_subset=True,no_kernel_timing=True,runtime_abi_checked=True)
    result['inputs']=[v for v in result['inputs'] if v['case']=='seed0']
    result['records']=[dict(case=v['case'],input_offset=v['input_offset'],output_offset=v['output_offset'],
        passed=True,guards=True,inputs_unchanged=True,runtime_abi_checked=True,path='ordinary-0') for v in result['inputs']]
    return result


class BatchHarness(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.base=Path(self.temporary.name);self.calls=[];self.fail_tool=None
        self.registration=dict(source=dict(source_manifest={'fixture.py':'content'},source_digest='same-source'),
            tools=dict(hashes={'compiler':'tool-content'},environment=dict(CUDA_MODULE_LOADING='LAZY'),precision='fp32'))
        self.patches=[patch.object(batch.closure,'verify_run',side_effect=lambda run,**kw:(batch.read(Path(run)/'manifest.json'),self.registration)),
            patch.object(batch,'provider',return_value=inventory),patch.object(batch,'invoke_worker',side_effect=self.fake_worker),
            patch('cost_calibration.machine_artifacts',return_value={'status':'mock-cpu'})]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.temporary.cleanup()
    def run_directory(self,name='cold',count=2):
        run=self.base/name;(run/'cases').mkdir(parents=True)
        rows=[]
        for index in range(count):
            path=run/'cases'/str(index);path.mkdir()
            # identical executable, differing declared spec must remain separate proofs
            row=dict(path=str(path),spec=dict(key='case'+str(index),shape=[32,16],precision='fp32'),
                     status='generated',model='legacy',phases=[])
            batch.write(path/'execution_config.json',dict(copy_path='vector',precision='fp32',fp_fusion=False))
            batch.write(path/'image-identity.json',dict(mock='image',case=index))
            rows.append(row)
        batch.write(run/'manifest.json',dict(status='generated',compiled=True,rows=rows))
        batch.write(run/'registration.json',self.registration)
        batch.write(run/'protocol.json',dict(validation_execution=CFG,safety=dict(tools=list(batch.TOOLS))))
        return run
    def fake_worker(self,run,cfg,rows,phase,tool,output,**kwargs):
        self.calls.append((phase,tool,len(rows)));output=Path(output);output.mkdir()
        plan=dict(schema_version=1,contract=batch.CONTRACT,run=str(run),registration_sha256=batch.sha(Path(run)/'registration.json'),
                  protocol_sha256=batch.sha(Path(run)/'protocol.json'),configuration=cfg,phase=phase,tool=tool,
                  rows=[batch.row_projection(r) for r in rows],output=str(output),timeout_seconds=1800)
        batch.write(output/'plan.json',plan);plan_sha=batch.sha(output/'plan.json');attempts=[]
        fail=tool==self.fail_tool and self.fail_tool is not None
        for row in rows[:1] if fail else rows:
            path=Path(row['path']);result=normal_payload(row) if phase=='ordinary' else safety_payload(batch.read(path/'correctness.json'))
            result_path=path/('correctness.json' if phase=='ordinary' else 'safety-'+tool+'-correctness.json')
            batch.write(result_path,result)
            attempts.append(dict(path=row['path'],status='failed' if fail else 'complete',cubin_sha256=result['cubin_sha256'],
                                 result_path=str(result_path),result_sha256=batch.sha(result_path)))
        result=dict(contract=batch.CONTRACT,status='failed' if fail else 'complete',plan_sha256=plan_sha,
                    requested=len(rows),attempted=len(attempts),not_attempted=len(rows)-len(attempts),attempts=attempts)
        batch.write(output/'worker-result.json',result);(output/'process.log').write_text('mock CPU process')
        command=dict(exit_code=91 if fail else 0,status='failed' if fail else 'complete',launched=True,scope_complete=True,descendants_reaped=True)
        batch.write(output/'process.log.command.json',command)
        outcome=dict(command,log=str(output/'process.log'),command_metadata=str(output/'process.log.command.json'))
        return dict(status=result['status'],outcome=outcome,plan=plan,plan_sha256=plan_sha,result=result,
                    accounting={k:result[k] for k in ('requested','attempted','not_attempted')},result_sha256=batch.sha(output/'worker-result.json'))
    def validate(self,run):return batch.validate(run,safety_cache_dir=self.base/'cache')

    def test_cold_five_then_warm_one_and_rebind_provenance(self):
        cold=self.run_directory();a=self.validate(cold)
        self.assertEqual((a['ordinary_worker_processes'],a['actual_sanitizer_processes'],a['unique_safety_identities']),(1,4,2))
        self.assertFalse(a['safety_cache_hit']);self.assertEqual(len(self.calls),5)
        warm=self.run_directory('warm');b=self.validate(warm)
        self.assertEqual((b['ordinary_worker_processes'],b['actual_sanitizer_processes']),(1,0));self.assertTrue(b['safety_cache_hit'])
        self.assertEqual(len(self.calls),6)
        manifest=batch.read(warm/'manifest.json')
        for row in manifest['rows']:
            batch.verify_row_safety(row['path'],row)
            self.assertFalse(row['safety']['new_sanitizer_execution'])
            self.assertEqual(row['safety']['batch_tool_processes_in_this_run'],0)
        self.assertFalse((self.base/'cache'/'.writer.lock').exists())

    def alias_rows(self,run):
        manifest=batch.read(run/'manifest.json');first=manifest['rows'][0]
        for index,row in enumerate(manifest['rows']):
            row['spec']=copy.deepcopy(first['spec'])
            d=Path(row['path']);batch.write(d/'generation-inputs.json',dict(generation_source=str(run),candidate=index))
            batch.write(d/'image-identity.json',dict(kernel='same-cubin',generation_inputs_sha256=batch.sha(d/'generation-inputs.json')))
        batch.write(run/'manifest.json',manifest)
        return manifest['rows']

    def test_same_safety_identity_different_generation_lineage_cold_and_warm(self):
        cold=self.run_directory();rows=self.alias_rows(cold)
        images=[batch.sha(Path(row['path'])/'image-identity.json') for row in rows]
        self.assertNotEqual(*images)
        result=self.validate(cold)
        self.assertEqual(result['unique_safety_identities'],1);self.assertEqual(result['actual_sanitizer_processes'],4)
        self.assertTrue(all(count==1 for phase,tool,count in self.calls if phase=='safety'))
        manifest=batch.read(cold/'manifest.json');entry=Path(result['safety_bundle'])
        metadata,proofs,evidence=batch.bundle_payload(entry);key=next(iter(proofs))
        for tool in batch.TOOLS:
            self.assertEqual(proofs[key][tool]['image_identity_sha256'],images[0])
            self.assertEqual(evidence[tool]['covered_rows'][key],rows[0]['path'])
        for index,row in enumerate(manifest['rows']):
            batch.verify_row_safety(row['path'],row)
            self.assertEqual(row['safety']['current_image_identity_sha256'],images[index])
        warm=self.run_directory('warm-alias');warmrows=self.alias_rows(warm)
        warmimages=[batch.sha(Path(row['path'])/'image-identity.json') for row in warmrows]
        self.assertNotEqual(images,warmimages)
        before=len(self.calls);cached=self.validate(warm)
        self.assertTrue(cached['safety_cache_hit']);self.assertEqual(cached['actual_sanitizer_processes'],0)
        self.assertEqual(len(self.calls),before+1)
        manifest=batch.read(warm/'manifest.json')
        for index,row in enumerate(manifest['rows']):
            batch.verify_row_safety(row['path'],row)
            self.assertEqual(row['safety']['current_image_identity_sha256'],warmimages[index])
        # Even a coordinated receipt/hash rewrite cannot bind row two to row one.
        row=manifest['rows'][1];row['safety']['current_image_identity_sha256']=warmimages[0]
        proof=Path(row['path'])/'safety-batch-proof.json';batch.write(proof,row['safety']);row['artifact_hashes'][proof.name]=batch.sha(proof)
        with self.assertRaisesRegex(ValueError,'another current image'):batch.verify_row_safety(row['path'],row)

    def test_aliased_rows_do_not_accept_another_representative_sanitizer_provenance(self):
        run=self.run_directory();rows=self.alias_rows(run);original=self.fake_worker
        def wrong_representative(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[3]=='safety':
                tool=args[4];proof=Path(rows[0]['path'])/('safety-'+tool+'-correctness.json')
                value=batch.read(proof);value['image_identity_sha256']=batch.sha(Path(rows[1]['path'])/'image-identity.json');batch.write(proof,value)
            return result
        with patch.object(batch,'invoke_worker',side_effect=wrong_representative):
            with self.assertRaisesRegex(ValueError,'current sanitizer checked another image'):self.validate(run)
        self.assertEqual(batch.read(run/'manifest.json')['status'],'failed')
        self.assertEqual(list((self.base/'cache').iterdir()),[])

    def test_aliased_safety_key_still_requires_identical_full_ordinary_references(self):
        run=self.run_directory();rows=self.alias_rows(run);original=self.fake_worker
        def wrong_reference(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[3]=='ordinary':
                path=Path(rows[1]['path'])/'correctness.json';value=batch.read(path)
                value['inputs'][-1]['reference_sha256']='wrong-other-ordinary-case';batch.write(path,value)
            return result
        with patch.object(batch,'invoke_worker',side_effect=wrong_reference):
            with self.assertRaisesRegex(ValueError,'aliased safety identity changed actual inputs'):self.validate(run)
        self.assertFalse(any(phase=='safety' for phase,tool,count in self.calls))

    def test_registered_cache_and_session_ablations_preserve_cold_work(self):
        for index in range(2):
            run=self.run_directory('uncached'+str(index));protocol=batch.read(run/'protocol.json')
            protocol['validation_execution']['safety_cache']=False
            protocol['validation_execution']['integrity_session']=False
            batch.write(run/'protocol.json',protocol)
            result=self.validate(run)
            self.assertEqual(result['actual_sanitizer_processes'],4);self.assertFalse(result['safety_cache_hit'])
            self.assertIsNone(result['controller_integrity'])
            self.assertTrue(Path(result['safety_bundle']).is_relative_to(run/'batched-validation'))
        self.assertEqual(list((self.base/'cache').iterdir()),[])

    def test_failed_batch_is_counted_stops_and_cannot_publish(self):
        run=self.run_directory();self.fail_tool='racecheck'
        with self.assertRaisesRegex(RuntimeError,'failed validation worker'):self.validate(run)
        summary=batch.read(run/'batched-validation'/'summary.json')
        self.assertEqual(summary['status'],'failed');self.assertEqual(summary['actual_sanitizer_processes'],2)
        self.assertEqual(summary['phases'][-1]['accounting'],dict(requested=2,attempted=1,not_attempted=1))
        self.assertEqual(batch.read(run/'manifest.json')['status'],'failed')
        self.assertEqual(list((self.base/'cache').iterdir()),[])

    def test_corrupted_warm_proof_fails_closed_without_replacement_or_tools(self):
        cold=self.run_directory();a=self.validate(cold);entry=Path(a['safety_bundle']);path=entry/'proofs.json'
        old=path.stat();content=path.read_bytes();path.write_bytes(content.replace(b'same-gpu',b'fake-gpu'));os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
        warm=self.run_directory('warm');before=len(self.calls)
        with self.assertRaisesRegex(ValueError,'payload changed'):self.validate(warm)
        self.assertEqual(len(self.calls),before+1)
        self.assertTrue(entry.exists())

    def test_exact_context_mutations_change_bundle_key(self):
        run=self.run_directory();row=batch.read(run/'manifest.json')['rows'][0];normal=normal_payload(row)
        identity=batch.safety_identity(row,normal);key=batch.digest(batch.bundle_identity(self.registration,CFG,{'key':identity}))
        mutations=[('cubin_sha256','another'),('gpu_uuid','another'),('physical_params',['other']),
                   ('physical_resources',dict(registers_per_thread=33)),('safety_input_identity',dict(cases=[]))]
        for field,value in mutations:
            other=copy.deepcopy(identity);other[field]=value
            self.assertNotEqual(key,batch.digest(batch.bundle_identity(self.registration,CFG,{'key':other})),field)
        for field in ('source','tools'):
            other=copy.deepcopy(self.registration);other[field]['mutation']='changed'
            self.assertNotEqual(key,batch.digest(batch.bundle_identity(other,CFG,{'key':identity})),field)
        config=batch.read(Path(row['path'])/'execution_config.json');config['precision']='fp16';batch.write(Path(row['path'])/'execution_config.json',config)
        self.assertNotEqual(identity,batch.safety_identity(row,normal))

    def test_validated_batch_receipt_uses_explicit_sanitizer_environment(self):
        run=self.run_directory();self.validate(run);row=batch.read(run/'manifest.json')['rows'][0]
        actual=batch.closure.verify_run
        def guarded(path,*,sanitizer=False):
            if sanitizer is not True:raise ValueError('ordinary check rejected sanitizer environment')
            return actual(path,sanitizer=sanitizer)
        with patch.object(batch.closure,'verify_run',side_effect=guarded) as verify:
            with self.assertRaisesRegex(ValueError,'ordinary check'):batch.verify_row_safety(row['path'],row)
            batch.verify_row_safety(row['path'],row,sanitizer=True)
            self.assertTrue(verify.call_args.kwargs['sanitizer'])

    def test_changed_source_or_environment_cannot_reuse_row_receipt(self):
        run=self.run_directory();self.validate(run);row=batch.read(run/'manifest.json')['rows'][0]
        self.registration['tools']['environment']['CUDA_MODULE_LOADING']='EAGER'
        with self.assertRaisesRegex(ValueError,'scope changed'):batch.verify_row_safety(row['path'],row)

    def test_coordinated_failed_process_and_omitted_row_are_rejected(self):
        run=self.run_directory();result=self.validate(run);entry=Path(result['safety_bundle'])
        original=batch.read(entry/'process-evidence.json')
        for mutate in ('failed','omitted','escaped','digest'):
            evidence=copy.deepcopy(original);record=evidence['memcheck']
            if mutate=='failed':record['exit_code']=91
            if mutate=='omitted':record['worker']['attempts'].pop()
            if mutate=='escaped':record['command_metadata']['descendants_reaped']=False
            if mutate=='digest':record['proof_digests'][next(iter(record['proof_digests']))]='wrong'
            batch.write(entry/'process-evidence.json',evidence)
            metadata=batch.read(entry/'metadata.json');metadata['files']['process-evidence.json']=dict(bytes=(entry/'process-evidence.json').stat().st_size,sha256=batch.sha(entry/'process-evidence.json'));batch.write(entry/'metadata.json',metadata)
            with self.assertRaises(ValueError,msg=mutate):batch.bundle_payload(entry)

    def test_numerical_abi_guard_offset_reference_checks_remain_exact(self):
        run=self.run_directory();row=batch.read(run/'manifest.json')['rows'][0];normal=normal_payload(row)
        identity=batch.safety_identity(row,normal);proof=safety_payload(normal)
        batch.verify_safety_result(proof,normal,identity)
        for field,value in [('gpu_uuid','wrong'),('physical_params',['wrong']),('physical_resources',{}),('runtime_abi_checked',False),('safety_subset',False),('image_identity_sha256','old')]:
            mutated=copy.deepcopy(proof);mutated[field]=value
            with self.assertRaises(ValueError,msg=field):batch.verify_safety_result(mutated,normal,identity)
        for action in ('missing-record','offset','reference','guard','environment'):
            mutated=copy.deepcopy(proof)
            if action=='missing-record':mutated['records'].pop()
            if action=='offset':mutated['inputs'][0]['input_offset']=7
            if action=='reference':mutated['inputs'][0]['reference_sha256']='wrong'
            if action=='guard':mutated['records'][0]['guards']=False
            if action=='environment':mutated['execution_environment']['canonical']['LD_LIBRARY_PATH']='injected'
            with self.assertRaises(ValueError,msg=action):batch.verify_safety_result(mutated,normal,identity)


class WorkerTests(unittest.TestCase):
    def test_configuration_rejects_implicit_or_unregistered_controls(self):
        self.assertEqual(batch.configuration(CFG),CFG)
        for cfg in ({**CFG,'unknown':True},{**CFG,'integrity_session':1},{**CFG,'numerical_fixture':'../fixture'},
                    {**CFG,'inventory_verifier':'arbitrary.module:call'}):
            with self.assertRaises(ValueError):batch.configuration(cfg)

    def test_process_success_failure_timeout_and_double_fork_are_reaped(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            for name,script,timeout in [
                ('success','print("ok")',2),('failure','raise SystemExit(7)',2),
                ('timeout','import time;time.sleep(10)',.1),
                ('orphan','import os,time;pid=os.fork();\nif pid==0: os.setsid();time.sleep(.1);os._exit(0)',2),
                ('orphan-timeout','import os,time;pid=os.fork();\nif pid==0: os.setsid();time.sleep(10);os._exit(0)',.1)]:
                result=batch.invoke_process([sys.executable,'-B','-c',script],directory/(name+'.log'),timeout=timeout)
                self.assertTrue(result['launched'],name);self.assertTrue(result['scope_complete'],name)
                self.assertTrue(result['descendants_reaped'],name)
                self.assertEqual(result['status']=='complete',name in ('success','orphan'),name)
                if 'timeout' in name:self.assertTrue(result['timeout_expired'])
            self.assertFalse(batch.psutil.Process().children())

    def test_sanitizer_executes_exact_frozen_payload_without_cuda_alias(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);runtime=run/'canonical-cuda-runtime';runtime.mkdir()
            executable=runtime/'compute-sanitizer';executable.write_text('registered-tool')
            expected=batch.sha(executable)
            registration=dict(tools=dict(hashes={str(executable):expected},sanitizer_runtime=dict(directory=str(runtime),payload_hashes={str(executable):expected})))
            batch.write(run/'registration.json',registration)
            self.assertEqual(batch.frozen_sanitizer(run),str(executable))
            (run/'cases').mkdir();case=run/'cases'/'0';case.mkdir();batch.write(run/'protocol.json',{})
            outcome=dict(exit_code=91,status='failed',launched=True,scope_complete=True,descendants_reaped=True,log='retained',command_metadata='retained')
            with patch.object(batch,'invoke_process',return_value=outcome) as invoke:
                batch.invoke_worker(run,CFG,[dict(path=str(case),spec={})],'safety','memcheck',run/'worker')
                self.assertEqual(invoke.call_args.args[0][:5],[str(executable),'--tool','memcheck','--error-exitcode','91'])
                self.assertNotIn('/usr/local/cuda/bin/compute-sanitizer',invoke.call_args.args[0])
            executable.write_text('substituted-tool')
            with self.assertRaisesRegex(ValueError,'content differs'):batch.frozen_sanitizer(run)

    def test_sanitizer_rejects_unregistered_or_symlinked_payload_before_launch(self):
        with tempfile.TemporaryDirectory() as name:
            run=Path(name);runtime=run/'runtime';runtime.mkdir();executable=runtime/'compute-sanitizer';executable.write_text('tool')
            expected=batch.sha(executable)
            registration=dict(tools=dict(hashes={str(executable):expected},sanitizer_runtime=dict(directory=str(runtime),payload_hashes={str(executable):expected})))
            batch.write(run/'registration.json',registration)
            registration['tools']['sanitizer_runtime']['payload_hashes'][str(executable)]='0'*64
            batch.write(run/'registration.json',registration)
            with self.assertRaisesRegex(ValueError,'both frozen'):batch.frozen_sanitizer(run)
            registration['tools']['sanitizer_runtime']['payload_hashes'][str(executable)]=expected
            batch.write(run/'registration.json',registration)
            target=run/'replacement';executable.rename(target);executable.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'symlink'):batch.frozen_sanitizer(run)

    def test_invoke_worker_missing_report_keeps_execution_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);(run/'cases').mkdir();d=run/'cases'/'0';d.mkdir()
            (run/'registration.json').write_text('{}');(run/'protocol.json').write_text('{}')
            outcome=dict(exit_code=124,status='failed',launched=True,scope_complete=True,descendants_reaped=True,
                         log='retained.log',command_metadata='retained.json')
            with patch.object(batch,'invoke_process',return_value=outcome):
                result=batch.invoke_worker(run,CFG,[dict(path=str(d),spec={})],'ordinary',None,run/'worker',timeout_seconds=1)
            self.assertEqual(result['status'],'failed');self.assertIsNone(result['accounting']['attempted'])
            self.assertIsNone(result['accounting']['not_attempted']);self.assertIn('unknown',result['accounting']['interpretation'])
            self.assertTrue((run/'worker'/'invocation.json').is_file())

    def test_duplicate_paths_and_changed_worker_plan_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);(run/'cases').mkdir();d=run/'cases'/'0';d.mkdir();output=run/'worker';output.mkdir()
            row=dict(path=str(d),spec={})
            plan=dict(schema_version=1,contract=batch.CONTRACT,run=str(run),registration_sha256='r',protocol_sha256='p',
                configuration=CFG,phase='ordinary',tool=None,rows=[row,row],output=str(output),timeout_seconds=1)
            with self.assertRaisesRegex(ValueError,'duplicate'):batch.check_plan(plan)
            plan['rows']=[row];batch.write(output/'plan.json',plan)
            with patch.object(batch.closure,'execute') as execute:
                with self.assertRaisesRegex(ValueError,'hash differs'):batch.worker(output/'plan.json','not-the-plan-hash')
                execute.assert_not_called()

    def test_worker_fails_first_case_and_reports_rest_not_attempted(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary);(run/'cases').mkdir();output=run/'worker';output.mkdir()
            rows=[]
            for index in range(3):
                d=run/'cases'/str(index);d.mkdir();rows.append(dict(path=str(d),spec=dict(key=str(index))))
            plan=dict(schema_version=1,contract=batch.CONTRACT,run=str(run),registration_sha256='r',protocol_sha256='p',
                configuration=CFG,phase='ordinary',tool=None,rows=rows,output=str(output),timeout_seconds=2)
            batch.write(output/'plan.json',plan)
            with patch.object(batch,'check_live',return_value=({},{})),patch.object(batch,'provider',return_value=inventory), \
                 patch.object(batch.closure,'gpu_state',return_value=dict(foreign_pids=[])), \
                 patch.object(batch.closure,'execute',side_effect=ValueError('numeric guard failed')):
                status=batch.worker(output/'plan.json',batch.sha(output/'plan.json'))
            result=batch.read(output/'worker-result.json')
            self.assertEqual(status,1);self.assertEqual((result['attempted'],result['not_attempted']),(1,2))
            self.assertEqual(result['attempts'][0]['status'],'failed');self.assertTrue(result['integrity']['closed'])


if __name__=='__main__':unittest.main()
