"""Preregistered ordinary/sanitizer batches and exact same-source safety reuse.

The existing support_closure.execute performs every numerical, ABI, stream,
capture, resource and safety check. This module only schedules it. A successful
batch process is required for every contained sanitizer proof; a failed process
cannot publish a partial proof. Cache proofs are fully content-verified, never
accepted from file stat alone, and remain tied to complete source/tool/runtime
and fixture/input identities. Ordinary execution is always repeated.
"""
from __future__ import annotations
import argparse
import ctypes
from contextlib import nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
import psutil
import support_closure as closure
from integrity_session import VerificationSession

CONTRACT = 'frisk-five-process-validation-batch-v1'
CACHE_CONTRACT = 'frisk-same-source-safety-bundle-v1'
TOOLS = ('memcheck', 'racecheck', 'synccheck', 'initcheck')
ROOT = Path(__file__).resolve().parents[3]
MODULES = Path(__file__).resolve().parent
CONFIG_KEYS = {'contract','numerical_fixture','inventory_verifier','integrity_session','safety_cache'}
ROW_FIELDS = ('spec','scope','arm','model','selection','conversion','path','prepared','prepared_sha256',
              'domain','generation_inputs_sha256','assignment_sha256','complete_program_sha256',
              'model_id','cost','archive_entry','source_archive_mode')


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def serialized(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(path,value):Path(path).write_bytes(serialized(value))
def safe(path, *, directory=False, create=False):
    path=Path(path).absolute()
    if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('symlink validation/cache path rejected')
    if create:path.mkdir(parents=True,exist_ok=True)
    if directory and not path.is_dir():raise ValueError('validation/cache directory required')
    if not directory and not path.is_file():raise ValueError('validation/cache file required')
    return path


def configuration(value):
    if not isinstance(value,dict) or set(value)!=CONFIG_KEYS or value.get('contract')!=CONTRACT:
        raise ValueError('exact registered validation execution contract required')
    if type(value.get('integrity_session')) is not bool or type(value.get('safety_cache')) is not bool:
        raise ValueError('explicit validation session/cache booleans required')
    for field,pattern in [('numerical_fixture',r'[A-Za-z_][A-Za-z0-9_]*'),
                          ('inventory_verifier',r'[A-Za-z_][A-Za-z0-9_]*:[A-Za-z_][A-Za-z0-9_]*')]:
        if not isinstance(value[field],str) or re.fullmatch(pattern,value[field]) is None:
            raise ValueError('local fixture/inventory provider identifier required')
    return dict(value)


def provider(name, registration):
    module_name,separator,attribute=name.partition(':')
    path=safe(MODULES/(module_name+'.py'))
    relative=str(path.relative_to(ROOT))
    if registration['source']['source_manifest'].get(relative)!=sha(path):
        raise ValueError('validation provider is outside registered source snapshot')
    module=importlib.import_module(module_name)
    if Path(module.__file__).resolve()!=path:raise ValueError('validation provider import path differs')
    return getattr(module,attribute) if separator else module


def row_projection(row):return {name:row[name] for name in ROW_FIELDS if name in row}


def validate_rows(run,manifest):
    rows=manifest.get('rows')
    if not isinstance(rows,list) or not rows:raise ValueError('complete nonempty validation matrix required')
    seen=set()
    for row in rows:
        directory=safe(row['path'],directory=True)
        if directory.parent!=(Path(run)/'cases').absolute() or directory in seen:
            raise ValueError('unique registered case directories required')
        seen.add(directory)
        if row.get('status') not in ('generated','validated'):raise ValueError('generated/validated row required')
    return rows


def check_plan(plan):
    expected={'schema_version','contract','run','registration_sha256','protocol_sha256','configuration',
              'phase','tool','rows','output','timeout_seconds'}
    if not isinstance(plan,dict) or set(plan)!=expected or plan.get('schema_version')!=1 or plan.get('contract')!=CONTRACT:
        raise ValueError('validation worker plan schema differs')
    configuration(plan['configuration'])
    if plan['phase'] not in ('ordinary','safety') or (plan['phase']=='ordinary')!=(plan['tool'] is None):
        raise ValueError('validation worker phase/tool differs')
    if plan['phase']=='safety' and plan['tool'] not in TOOLS:raise ValueError('named registered sanitizer required')
    if type(plan['timeout_seconds']) is not int or not 1<=plan['timeout_seconds']<=3600:
        raise ValueError('bounded validation worker timeout required')
    run=safe(plan['run'],directory=True);safe(plan['output'],directory=True)
    if not isinstance(plan['rows'],list) or not plan['rows']:raise ValueError('nonempty explicit worker inventory required')
    paths=[]
    for row in plan['rows']:
        if row_projection(row)!=row:raise ValueError('unknown worker row field')
        directory=safe(row['path'],directory=True)
        if directory.parent!=run/'cases':raise ValueError('worker row outside registered run')
        paths.append(str(directory))
    if len(paths)!=len(set(paths)):raise ValueError('duplicate worker row')
    return run


def check_live(plan, *, sanitizer):
    run=check_plan(plan)
    manifest,registration=closure.verify_run(run,sanitizer=sanitizer)
    if sha(run/'registration.json')!=plan['registration_sha256'] or sha(run/'protocol.json')!=plan['protocol_sha256']:
        raise ValueError('validation plan registration/protocol changed')
    protocol=read(run/'protocol.json')
    if configuration(protocol.get('validation_execution'))!=plan['configuration']:
        raise ValueError('validation execution registration changed')
    if set(protocol.get('safety',{}).get('tools',[]))!=set(TOOLS) or len(protocol['safety']['tools'])!=4:
        raise ValueError('all four preregistered sanitizer tools required')
    rows=validate_rows(run,manifest);current={row['path']:row_projection(row) for row in rows}
    if any(current.get(row['path'])!=row for row in plan['rows']):
        raise ValueError('worker row differs from complete registered manifest')
    return manifest,registration


def worker(plan_path,expected_sha):
    path=safe(plan_path)
    if sha(path)!=expected_sha:raise ValueError('validation worker plan hash differs')
    plan=read(path);check_plan(plan);output=Path(plan['output']);sanitizer=plan['phase']=='safety'
    results=[];error=None;session=None;start=time.perf_counter_ns()
    try:
        scope=VerificationSession() if plan['configuration']['integrity_session'] else nullcontext()
        with scope as session:
            _,registration=check_live(plan,sanitizer=sanitizer)
            inventory=provider(plan['configuration']['inventory_verifier'],registration)
            inventory(plan['run'],check_artifacts=True,sanitizer=sanitizer)
            numerical=provider(plan['configuration']['numerical_fixture'],registration)
            from native_compile_cache import verify_precompiled
            for row in plan['rows']:
                begin=time.perf_counter_ns();attempt=dict(path=row['path'],status='running')
                try:
                    check_live(plan,sanitizer=sanitizer)
                    state=closure.gpu_state()
                    if state['foreign_pids']:raise RuntimeError('foreign GPU workload before validation case')
                    print(json.dumps(dict(event='row-start',path=row['path'],tool=plan['tool'])),flush=True)
                    result=closure.execute(row['path'],load_only=sanitizer,safety=sanitizer,safety_tool=plan['tool'],
                        numerical_fixture=numerical,compiler_provider=verify_precompiled)
                    result_path=Path(row['path'])/('safety-'+plan['tool']+'-correctness.json' if sanitizer else 'correctness.json')
                    attempt.update(status='complete',result_path=str(result_path),result_sha256=sha(result_path),
                                   cubin_sha256=result['cubin_sha256'],numerical_records=len(result['records']))
                    print(json.dumps(dict(event='row-complete',path=row['path'],tool=plan['tool'])),flush=True)
                except Exception as exc:
                    attempt.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                    results.append(dict(attempt,wall_ns=time.perf_counter_ns()-begin));raise
                results.append(dict(attempt,wall_ns=time.perf_counter_ns()-begin))
            check_live(plan,sanitizer=sanitizer)
            state=closure.gpu_state()
            if state['foreign_pids']:raise RuntimeError('foreign GPU workload after validation batch')
    except Exception as exc:error=dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
    result=dict(schema_version=1,contract=CONTRACT,status='complete' if error is None else 'failed',
        phase=plan['phase'],tool=plan['tool'],plan_sha256=expected_sha,requested=len(plan['rows']),
        attempted=len(results),not_attempted=len(plan['rows'])-len(results),attempts=results,
        error=error,wall_ns=time.perf_counter_ns()-start,
        integrity=session.summary() if isinstance(session,VerificationSession) else None)
    write(output/'worker-result.json',result)
    return 0 if error is None else 1


def invoke_process(argv,log,*,timeout):
    """Bounded isolated child tree, including reparented/setsid descendants.

    Linux subreaping and ECHILD close the parent-exit/fork snapshot race. An
    incomplete reap is a failed phase. We require no pre-existing children so
    waitpid(-1) can never consume another caller's process.
    """
    log=Path(log);metadata=Path(str(log)+'.command.json');start=time.perf_counter_ns()
    result=dict(argv=argv,timeout_seconds=timeout,exit_code=127,timeout_expired=False,
                launched=False,scope_complete=False,descendants_reaped=False,adopted_descendant_exits=[])
    process=None;libc=ctypes.CDLL(None,use_errno=True);prior=ctypes.c_int();enabled=False
    error=None;kill_begin=None;all_reaped=False;interrupted=None;tracked={}
    def family():
        values={}
        for proc in psutil.Process().children(recursive=True):
            try:values[proc.pid]=proc;tracked[proc.pid]=proc.create_time()
            except psutil.NoSuchProcess:pass
        for pid,created in list(tracked.items()):
            try:
                proc=psutil.Process(pid)
                if proc.create_time()==created:values[pid]=proc
            except psutil.NoSuchProcess:pass
        return values
    def stop(hard=False):
        signum=signal.SIGKILL if hard else signal.SIGTERM
        if process is not None:
            try:os.killpg(process.pid,signum)
            except ProcessLookupError:pass
        for proc in family().values():
            try:proc.send_signal(signum)
            except psutil.NoSuchProcess:pass
    try:
        if psutil.Process().children():raise RuntimeError('validation launcher requires an isolated process without existing children')
        if libc.prctl(37,ctypes.byref(prior),0,0,0)!=0 or libc.prctl(36,1,0,0,0)!=0:
            raise RuntimeError('Linux child-subreaper is required for validation')
        enabled=True
        with log.open('w') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            result.update(launched=True,pid=process.pid,owned_process_group=process.pid)
            while True:
                family();code=process.poll();elapsed=(time.perf_counter_ns()-start)/1e9
                if elapsed>timeout and kill_begin is None:
                    result['timeout_expired']=True;kill_begin=time.monotonic();stop()
                if code is not None:
                    while True:
                        try:pid,status=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:all_reaped=True;break
                        if pid==0:break
                        result['adopted_descendant_exits'].append(dict(pid=pid,exit_code=os.waitstatus_to_exitcode(status)))
                    if all_reaped:break
                if kill_begin is not None:
                    if time.monotonic()-kill_begin>1:stop(hard=True)
                    if time.monotonic()-kill_begin>10:raise RuntimeError('validation descendants could not be reaped after timeout')
                time.sleep(.05)
            result['parent_exit_code']=process.wait()
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc)
        if not isinstance(exc,Exception):interrupted=exc
    finally:
        if process is not None and not all_reaped:
            stop(hard=True);deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                stop(hard=True);process.poll()
                if process.returncode is not None:
                    while True:
                        try:pid,status=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:all_reaped=True;break
                        if pid==0:break
                        result['adopted_descendant_exits'].append(dict(pid=pid,exit_code=os.waitstatus_to_exitcode(status)))
                if all_reaped:break
                time.sleep(.05)
        if process is not None and process.returncode is not None:result['parent_exit_code']=process.returncode
        if enabled and (all_reaped or process is None):libc.prctl(36,prior.value,0,0,0)
        if error is None and any(p['exit_code']!=0 for p in result['adopted_descendant_exits']):
            error='owned descendant exited unsuccessfully'
        result.update(scope_complete=process is not None and all_reaped,descendants_reaped=all_reaped,
                      error=error,elapsed_ns=time.perf_counter_ns()-start)
        result['exit_code']=(124 if result['timeout_expired'] else result.get('parent_exit_code',127))
        if error is not None or not result['scope_complete']:result['exit_code']=94
        result['status']='complete' if result['exit_code']==0 and result['scope_complete'] else 'failed'
        result['descendant_accounting']='isolated Linux child-subreaper; direct parent reaped before waitpid ECHILD'
        if not log.exists():log.write_text('')
        write(metadata,result)
    if interrupted is not None:raise interrupted
    return dict(result,log=str(log),command_metadata=str(metadata))


def frozen_sanitizer(run):
    """Execute the content-verified registered canonical payload, never CUDA alias."""
    from integrity_session import verified_sha
    tools=read(Path(run)/'registration.json')['tools'];runtime=tools['sanitizer_runtime']
    directory=Path(runtime['directory'])
    if not directory.is_absolute():raise ValueError('frozen sanitizer runtime must be absolute')
    executable=safe(directory/'compute-sanitizer')
    name=str(executable);expected=tools['hashes'].get(name)
    if not isinstance(expected,str) or len(expected)!=64 or runtime['payload_hashes'].get(name)!=expected:
        raise ValueError('canonical sanitizer executable is not bound to both frozen tool identities')
    verified_sha(executable,expected)
    return str(executable)


def invoke_worker(run,configuration_value,rows,phase,tool,output,*,timeout_seconds=1800):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=dict(schema_version=1,contract=CONTRACT,run=str(Path(run).absolute()),
        registration_sha256=sha(Path(run)/'registration.json'),protocol_sha256=sha(Path(run)/'protocol.json'),
        configuration=configuration_value,phase=phase,tool=tool,
        rows=[row_projection(row) for row in rows],output=str(output.absolute()),timeout_seconds=timeout_seconds)
    check_plan(plan);write(output/'plan.json',plan);plan_sha=sha(output/'plan.json')
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'worker',str(output/'plan.json'),'--expected-sha',plan_sha]
    if tool is not None:argv=[frozen_sanitizer(run),'--tool',tool,'--error-exitcode','91']+argv
    outcome=invoke_process(argv,output/'process.log',timeout=timeout_seconds)
    record=dict(outcome=outcome,plan=plan,plan_sha256=plan_sha,result=None,result_sha256=None,
                status='failed',accounting=dict(requested=len(rows),attempted=None,not_attempted=None,
                interpretation='worker report unavailable; execution progress unknown'))
    try:
        result=read(output/'worker-result.json');record.update(result=result,result_sha256=sha(output/'worker-result.json'))
        attempts=result.get('attempts',[])
        if (result.get('contract')!=CONTRACT or result.get('plan_sha256')!=plan_sha or result.get('requested')!=len(rows)
                or result.get('attempted')!=len(attempts) or result.get('not_attempted')!=len(rows)-len(attempts)
                or [a.get('path') for a in attempts]!=[r['path'] for r in rows[:len(attempts)] ]
                or any(a.get('status') not in ('failed','complete') for a in attempts)):
            raise ValueError('validation worker result inventory/accounting differs')
        record['accounting']={k:result[k] for k in ('requested','attempted','not_attempted')}
        if outcome['exit_code']!=0 or outcome.get('scope_complete') is not True:
            raise RuntimeError('validation '+phase+' batch process failed')
        if result.get('status')!='complete' or result['not_attempted'] or any(a['status']!='complete' for a in attempts):
            raise ValueError('validation worker did not complete whole inventory')
        for attempt in attempts:
            if sha(attempt['result_path'])!=attempt['result_sha256']:raise ValueError('worker result payload changed')
        record['status']='complete'
    except Exception as exc:record['error']=type(exc).__name__+': '+str(exc)
    write(output/'invocation.json',record)
    return record


def safety_identity(row,correctness):
    directory=Path(row['path']);config=read(directory/'execution_config.json')
    return dict(cubin_sha256=correctness['cubin_sha256'],execution_config=config,
        physical_abi=correctness['physical_params'],safety_input_identity=correctness['safety_input_identity'],
        gpu_uuid=correctness['gpu_uuid'],physical_resources=correctness['physical_resources'])


def verify_safety_result(result,normal,identity, *, current_image=True):
    if (result.get('status')!='passed' or result.get('safety_subset') is not True
            or result.get('no_kernel_timing') is not True or result.get('runtime_abi_checked') is not True
            or result.get('cubin_sha256')!=identity['cubin_sha256'] or result.get('gpu_uuid')!=identity['gpu_uuid']
            or result.get('physical_params')!=identity['physical_abi']
            or result.get('physical_resources')!=identity['physical_resources']
            or result.get('safety_input_identity')!=identity['safety_input_identity']):
        raise ValueError('sanitizer result does not match exact current safety identity')
    if current_image and result.get('image_identity_sha256')!=normal['image_identity_sha256']:
        raise ValueError('current sanitizer checked another image provenance')
    wanted=[]
    names={case['name'] for case in identity['safety_input_identity']['cases']}
    for item in normal['inputs']:
        if item['case'] in names:wanted.append(item)
    if result.get('inputs')!=wanted:raise ValueError('sanitizer input/reference/offsets differ')
    records=result.get('records')
    expected=[(item['case'],item['input_offset'],item['output_offset']) for item in wanted]
    actual=[(r.get('case'),r.get('input_offset'),r.get('output_offset')) for r in records or []]
    if actual!=expected or any(not all(r.get(k) is True for k in ('passed','guards','inputs_unchanged','runtime_abi_checked')) for r in records or []):
        raise ValueError('sanitizer records are incomplete or failed')
    if result.get('execution_environment',{}).get('canonical')!=normal.get('execution_environment',{}).get('canonical'):
        raise ValueError('sanitizer canonical environment differs')
    return result


def bundle_identity(registration,configuration_value,identities):
    return dict(contract=CACHE_CONTRACT,validation_contract=CONTRACT,
        source=registration['source'],tools=registration['tools'],configuration=configuration_value,
        identities={key:identities[key] for key in sorted(identities)},tools_required=list(TOOLS))


def bundle_payload(entry,identity=None):
    entry=safe(entry,directory=True);metadata=read(safe(entry/'metadata.json'))
    if (set(metadata)!={'contract','identity','key','files'} or metadata['contract']!=CACHE_CONTRACT
            or metadata['key']!=digest(metadata['identity']) or identity is not None and metadata['identity']!=identity):
        raise ValueError('safety proof cache identity differs')
    identity=metadata['identity']
    if set(identity)!={'contract','validation_contract','source','tools','configuration','identities','tools_required'}:
        raise ValueError('safety proof identity schema differs')
    if identity['contract']!=CACHE_CONTRACT or identity['validation_contract']!=CONTRACT or identity['tools_required']!=list(TOOLS):
        raise ValueError('safety proof protocol differs')
    configuration(identity['configuration'])
    if not identity['identities'] or any(digest(value)!=key for key,value in identity['identities'].items()):
        raise ValueError('safety proof row keys differ')
    if set(metadata['files'])!={'proofs.json','process-evidence.json'}:raise ValueError('safety proof cache file set differs')
    if {p.name for p in entry.iterdir()}!={'metadata.json','proofs.json','process-evidence.json'}:
        raise ValueError('unexpected safety proof cache payload')
    for name,item in metadata['files'].items():
        path=safe(entry/name)
        if set(item)!={'bytes','sha256'} or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('safety proof cache payload changed')
    proofs=read(entry/'proofs.json');evidence=read(entry/'process-evidence.json')
    if set(proofs)!=set(identity['identities']) or set(evidence)!=set(TOOLS):raise ValueError('safety proof cache inventory differs')
    for tool,record in evidence.items():
        worker=record.get('worker',{});plan=record.get('plan',{});command=record.get('command_metadata',{})
        if (record.get('exit_code')!=0 or record.get('tool')!=tool or record.get('status')!='complete'
                or record.get('plan_sha256')!=hashlib.sha256(serialized(plan)).hexdigest()
                or worker.get('status')!='complete' or worker.get('plan_sha256')!=record['plan_sha256']
                or command.get('exit_code')!=0 or command.get('scope_complete') is not True
                or command.get('descendants_reaped') is not True):
            raise ValueError('safety proof process did not succeed and reap as a whole')
        if plan.get('configuration')!=identity['configuration'] or plan.get('tool')!=tool or plan.get('phase')!='safety':
            raise ValueError('cached safety worker contract differs')
        planned=[r['path'] for r in plan['rows']];attempts=worker.get('attempts',[])
        if (len(planned)!=len(set(planned)) or len(planned)!=len(identity['identities'])
                or worker.get('requested')!=len(planned) or worker.get('attempted')!=len(planned) or worker.get('not_attempted')!=0
                or [a['path'] for a in attempts]!=planned or any(a.get('status')!='complete' for a in attempts)
                or set(record.get('covered_rows',{}))!=set(identity['identities'])
                or set(record['covered_rows'].values())!=set(planned)
                or set(record.get('proof_digests',{}))!=set(identity['identities'])):
            raise ValueError('cached safety worker omitted or substituted a row')
        indexed={a['path']:a for a in attempts}
        for key,row_identity in identity['identities'].items():
            if (indexed[record['covered_rows'][key]].get('cubin_sha256')!=row_identity['cubin_sha256']
                    or record['proof_digests'][key]!=digest(proofs.get(key,{}).get(tool))):
                raise ValueError('cached worker proof binding differs')
    if any(set(bytool)!=set(TOOLS) for bytool in proofs.values()):raise ValueError('cached sanitizer tool subset differs')
    return metadata,proofs,evidence


def load_bundle(entry,identity,normals):
    metadata,proofs,evidence=bundle_payload(entry,identity)
    if set(normals)!=set(identity['identities']):raise ValueError('current normal inventory differs')
    for key,bytool in proofs.items():
        for tool,result in bytool.items():verify_safety_result(result,normals[key],identity['identities'][key],current_image=False)
    return metadata,proofs,evidence


def publish_bundle(parent,identity,proofs,evidence):
    parent=safe(parent,directory=True,create=True);key=digest(identity);entry=parent/key
    if entry.exists():raise FileExistsError('immutable safety proof entry appeared during publication')
    temporary=parent/('.partial-'+uuid.uuid4().hex);temporary.mkdir()
    write(temporary/'proofs.json',proofs);write(temporary/'process-evidence.json',evidence)
    files={name:dict(bytes=(temporary/name).stat().st_size,sha256=sha(temporary/name)) for name in ('proofs.json','process-evidence.json')}
    write(temporary/'metadata.json',dict(contract=CACHE_CONTRACT,identity=identity,key=key,files=files))
    temporary.rename(entry)
    return entry


def verify_row_safety(directory,row,*,sanitizer=False):
    safety=row.get('safety',{})
    if safety.get('status')!='verified-batch-safety-v1' or row.get('validation_execution_contract')!=CONTRACT:
        raise ValueError('registered batched safety receipt required')
    directory=Path(directory);run=directory.parent.parent
    path=safe(directory/'safety-batch-proof.json');proof=read(path)
    if row.get('artifact_hashes',{}).get(path.name)!=sha(path) or proof!=safety:
        raise ValueError('derived safety proof receipt changed')
    normal=read(directory/'correctness.json');identity=safety_identity(row,normal);key=digest(identity)
    entry=safe(safety['cache_entry'],directory=True)
    if sha(entry/'metadata.json')!=safety['cache_metadata_sha256']:raise ValueError('safety bundle metadata changed')
    metadata,proofs,evidence=bundle_payload(entry);all_identity=metadata['identity']
    _,registration=closure.verify_run(run,sanitizer=sanitizer)
    cfg=configuration(read(run/'protocol.json').get('validation_execution'))
    if all_identity['source']!=registration['source'] or all_identity['tools']!=registration['tools'] or all_identity['configuration']!=cfg:
        raise ValueError('safety bundle source/tool/environment/precision scope changed')
    if all_identity['identities'].get(key)!=identity or safety['row_identity_sha256']!=key:
        raise ValueError('safety bundle does not cover current exact executable/input context')
    if safety.get('identity')!=closure.safety_key(directory,normal):raise ValueError('derived original safety identity differs')
    for tool in TOOLS:verify_safety_result(proofs[key][tool],normal,identity,current_image=False)
    if safety['current_image_identity_sha256']!=normal['image_identity_sha256']:
        raise ValueError('safety reuse receipt bound to another current image')
    return dict(status='verified-batch-safety',key=key,bundle_key=metadata['key'],new_sanitizer_execution=safety['new_sanitizer_execution'])


def _validate(run, *, safety_cache_dir, output=None, worker_timeout_seconds=1800):
    """Five worker processes cold, one ordinary worker on exact warm proof hit.

    The protocol must preregister validation_execution with this module's exact
    configuration schema and a source-bound inventory verifier. Root workflow
    adapters retain control of candidate membership and total compile budgets.
    """
    run=safe(run,directory=True);manifest,registration=closure.verify_run(run);protocol=read(run/'protocol.json')
    cfg=configuration(protocol.get('validation_execution'));rows=validate_rows(run,manifest)
    if manifest.get('status')!='generated' or manifest.get('compiled') is not True:
        raise ValueError('fresh generated and precompiled matrix required')
    inventory=provider(cfg['inventory_verifier'],registration);inventory(run,check_artifacts=True,sanitizer=False)
    if set(protocol.get('safety',{}).get('tools',[]))!=set(TOOLS):raise ValueError('registered four-tool safety scope required')
    output=Path(output) if output is not None else run/'batched-validation'
    if output.exists():raise FileExistsError('validation refuses a previous attempt')
    output.mkdir(parents=True);cache=safe(safety_cache_dir,directory=True,create=True)
    if cache.stat().st_uid!=os.getuid():raise ValueError('same-user safety proof cache required')
    lock=cache/'.writer.lock';token=uuid.uuid4().hex
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as stream:stream.write(token)
    start=time.perf_counter_ns();phases=[];accepted={};normals={};identities={};error=None;cache_hit=False;entry=None
    manifest['status']='validating';closure.write(run/'manifest.json',manifest)
    def launch(worker_rows,phase,tool,directory):
        outcome=invoke_worker(run,cfg,worker_rows,phase,tool,directory,timeout_seconds=worker_timeout_seconds)
        phases.append(dict(phase=phase,tool=tool,worker_status=outcome['status'],accounting=outcome['accounting'],**outcome['outcome']))
        if outcome['status']!='complete':raise RuntimeError('failed validation worker: '+str(directory))
        return outcome
    try:
        ordinary=launch(rows,'ordinary',None,output/'ordinary')
        from cost_calibration import machine_artifacts
        for row in rows:
            directory=Path(row['path']);c=read(directory/'correctness.json')
            row['machine_evidence']=machine_artifacts(directory)
            row.update(image_identity_sha256=sha(directory/'image-identity.json'),correctness_sha256=sha(directory/'correctness.json'),
                cubin_sha256=c['cubin_sha256'],physical_resources=c['physical_resources'])
            row['phases'].append(dict(stage='batched-native-and-numerics',**ordinary['outcome'],batch_plan_sha256=ordinary['plan_sha256']))
            row['execution_identity']=digest(dict(cubin_sha256=c['cubin_sha256'],entry='frisk_execution',execution_config=read(directory/'execution_config.json'),physical_abi=c['physical_params']))
            identity=safety_identity(row,c);key=digest(identity)
            if key in normals and (normals[key]['inputs']!=c['inputs'] or identities[key]!=identity):raise ValueError('aliased safety identity changed actual inputs or context')
            # All three maps name the same first representative. Later rows
            # retain their own current image provenance in derived receipts.
            if key not in accepted:identities[key]=identity;normals[key]=c;accepted[key]=row
        closure.write(run/'manifest.json',manifest)
        identity=bundle_identity(registration,cfg,identities);entry=cache/digest(identity)
        if cfg['safety_cache'] and entry.exists():
            metadata,proofs,evidence=load_bundle(entry,identity,normals);cache_hit=True
        else:
            proofs={key:{} for key in accepted};evidence={};representatives=list(accepted.values())
            for tool in TOOLS:
                outcome=launch(representatives,'safety',tool,output/tool)
                evidence[tool]=dict(tool=tool,status='complete',exit_code=outcome['outcome']['exit_code'],
                    plan=outcome['plan'],plan_sha256=outcome['plan_sha256'],worker=outcome['result'],
                    process_log_sha256=sha(outcome['outcome']['log']),command_metadata=read(outcome['outcome']['command_metadata']))
                for key,row in accepted.items():
                    result=read(Path(row['path'])/('safety-'+tool+'-correctness.json'))
                    verify_safety_result(result,normals[key],identities[key]);proofs[key][tool]=result
                evidence[tool]['covered_rows']={key:row['path'] for key,row in accepted.items()}
                evidence[tool]['proof_digests']={key:digest(proofs[key][tool]) for key in accepted}
            # Even cache-disabled ablations publish an immutable per-run bundle
            # for audit; they never reuse an existing shared proof.
            if not cfg['safety_cache']:
                parent=output/'uncached-proof-bundles';parent.mkdir();entry=publish_bundle(parent,identity,proofs,evidence)
            else:entry=publish_bundle(cache,identity,proofs,evidence)
            metadata,proofs,evidence=load_bundle(entry,identity,normals)
        for row in rows:
            directory=Path(row['path']);key=digest(safety_identity(row,read(directory/'correctness.json')))
            receipt=dict(status='verified-batch-safety-v1',contract=CACHE_CONTRACT,identity=closure.safety_key(directory,read(directory/'correctness.json')),
                row_identity_sha256=key,cache_entry=str(entry),cache_metadata_sha256=sha(entry/'metadata.json'),
                current_image_identity_sha256=row['image_identity_sha256'],new_sanitizer_execution=not cache_hit,
                batch_tool_processes_in_this_run=0 if cache_hit else 4,process_count_scope='whole-validation-batch; do not sum per-row receipts',shared_processes_cover_multiple_rows=True,
                claim='exact source/tool/runtime/executable/ABI/input proof reuse; ordinary execution repeated; no new GPU timing')
            row['safety']=receipt;row['validation_execution_contract']=CONTRACT;write(directory/'safety-batch-proof.json',receipt)
            row['status']='validated';row['artifact_hashes']={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file()}
            verify_row_safety(directory,row)
        manifest['status']='validated';closure.write(run/'manifest.json',manifest)
        inventory(run,check_artifacts=True,sanitizer=False)
        closure.verify_run(run)
    except Exception as exc:
        error=dict(error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        manifest.update(status='failed',error=error['error']);closure.write(run/'manifest.json',manifest)
        raise
    finally:
        result=dict(schema_version=1,contract=CONTRACT,status='complete' if error is None else 'failed',
            rows=len(rows),unique_safety_identities=len(accepted),ordinary_worker_processes=sum(p['phase']=='ordinary' and p.get('launched') is True for p in phases),
            actual_sanitizer_processes=sum(p['phase']=='safety' and p.get('launched') is True for p in phases),safety_cache_hit=cache_hit,
            safety_bundle=str(entry) if entry else None,phases=phases,error=error,wall_ns=time.perf_counter_ns()-start,
            all_numerical_and_driver_checks_repeated=True,source_tool_and_environment_guards_preserved=True,
            trust_boundary='trusted registered local process; content at entry/exit and identity guards inside scope; not malicious concurrent-writer TOCTOU proof')
        write(output/'summary.json',result)
        if lock.is_symlink() or not lock.is_file() or lock.read_text()!=token:raise ValueError('safety cache writer ownership changed')
        lock.unlink()
    return result


def validate(run, *, safety_cache_dir, output=None, worker_timeout_seconds=1800):
    """Explicit complete controller + worker content sessions; default stays legacy.

    cfg is read before entry to choose the registered mode and is content-bound
    again inside _validate. No identity fact survives this invocation/session.
    """
    cfg=configuration(read(Path(run)/'protocol.json').get('validation_execution'))
    scope=VerificationSession() if cfg['integrity_session'] else nullcontext()
    session=None;result=None
    target=Path(output) if output is not None else Path(run)/'batched-validation'
    summary=target/'summary.json'
    try:
        with scope as session:
            result=_validate(run,safety_cache_dir=safety_cache_dir,output=output,worker_timeout_seconds=worker_timeout_seconds)
    except Exception as exc:
        if summary.is_file():
            value=read(summary);value.update(status='failed',boundary_error=type(exc).__name__+': '+str(exc),
                controller_integrity=session.summary() if isinstance(session,VerificationSession) else None)
            write(summary,value)
            manifest=read(Path(run)/'manifest.json');manifest.update(status='failed',error=value['boundary_error'])
            closure.write(Path(run)/'manifest.json',manifest)
        raise
    result['controller_integrity']=session.summary() if isinstance(session,VerificationSession) else None
    write(summary,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['worker'])
    parser.add_argument('plan',type=Path);parser.add_argument('--expected-sha',required=True)
    args=parser.parse_args();raise SystemExit(worker(args.plan,args.expected_sha))


if __name__=='__main__':main()
