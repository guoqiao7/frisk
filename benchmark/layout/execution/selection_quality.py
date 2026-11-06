"""Frozen model/selection qualification over exhaustive and natural domains.
Server-side only. Numerical/resource/sanitizer execution reuses support_closure.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import selection_quality_fixture as fixture
import support_closure as closure
from cost_calibration import source_identity, check_tools, check_inputs, features, compile_exact
from calibration_fixture import protocol_digest
from calibration_analysis import paired_analysis
ROOT=Path(__file__).resolve().parents[3]
BENCH=ROOT/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench'
OPT=ROOT/'build-m6-release/bin/frisk-opt'
sha=closure.sha
read=closure.read
write=closure.write
fresh=closure.fresh

def init(run):
    run=fresh(run);p=fixture.protocol();write(run/'protocol.json',p)
    profile=ROOT/fixture.PROFILE_RELATIVE
    if sha(profile)!=fixture.PROFILE_SHA256:raise ValueError('coefficient file changed')
    tools=closure.frozen_tools(OPT);tools['hashes'][str(BENCH)]=sha(BENCH)
    registration=dict(created_ns=time.time_ns(),protocol_sha256=sha(run/'protocol.json'),protocol_digest=protocol_digest(p),
        source=source_identity(),tools=tools,optimizer=str(OPT),profile_sha256=sha(profile))
    write(run/'registration.json',registration)
    write(run/'manifest.json',dict(schema_version=1,status='registered',registration_sha256=sha(run/'registration.json'),rows=[]))
    return run

def verify_protocol(run):
    closure.verify_run(run)
    p=read(run/'protocol.json');r=read(run/'registration.json')
    if protocol_digest(p)!=r['protocol_digest']:raise ValueError('protocol changed')
    return p

def invoke(args,log):
    result=closure.invoke(args,log)
    if result['exit_code']:raise RuntimeError('worker failed: '+str(log))
    return result

def generate(run):
    from execution_support import assess_file
    p=verify_protocol(run);manifest=read(run/'manifest.json')
    if manifest['status']!='registered':raise ValueError('fresh registered run required')
    manifest['status']='generating';write(run/'manifest.json',manifest)
    try:
        for spec in p['specifications']:
            scopes=['controlled','natural'] if spec['case_id']!='attention' else ['natural']
            for scope in scopes:
                base=fresh(run/'preparation'/(scope+'-'+spec['key']))
                raw=base/'input.mlir';raw.write_text(fixture.source(spec,ROOT,controlled=scope=='controlled'))
                common=None
                for model in ('legacy','calibrated'):
                    prepared=base/('prepared-'+model)
                    args=[BENCH,'--prepare',raw,spec['case_id'],prepared]
                    if scope=='natural' and spec['case_id']!='attention':args+=['--natural']
                    if model=='calibrated':args+=['--cost-profile='+str(ROOT/fixture.PROFILE_RELATIVE)]
                    invoke(args,base/('prepare-'+model+'.log'))
                    meta=read(prepared/'prepared.json')
                    keys=('common_digest','domain_digest','coverage_digest')
                    domain={k:meta[k] for k in keys}
                    if common is not None and domain!=common:raise ValueError('model changed common domain/proofs')
                    common=domain
                    arms=[('LI',None),('J',None)]
                    if scope=='controlled':arms += [('A','A'),('B','B')]
                    for label,force in arms:
                        arm=model+'-'+label;directory=run/'cases'/(scope+'-'+spec['key']+'-'+arm)
                        args=[BENCH,'--prepared='+str(prepared),'--case='+spec['case_id'],
                            '--selection='+('J' if force else label),'--conversion=C','--output-dir='+str(directory)]
                        if scope=='controlled':args+=['--oracle']
                        if model=='calibrated':args+=['--cost-profile='+str(ROOT/fixture.PROFILE_RELATIVE)]
                        if force:args+=['--force-producer='+force]
                        row=dict(spec=spec,scope=scope,arm=arm,model=model,selection=label,conversion='C',path=str(directory),status='generating',phases=[],
                            prepared=str(prepared),prepared_sha256=sha(prepared/'prepared.json'),domain=domain)
                        manifest['rows'].append(row);write(run/'manifest.json',manifest)
                        result=closure.invoke(args,base/(arm+'.log'))
                        row['phases'].append(dict(stage='selection',**result))
                        if result['exit_code']:
                            row['status']='failed'
                            if (directory/'failure.json').exists():row['failure']=read(directory/'failure.json')
                            raise RuntimeError('registered selection arm unavailable; no row excluded')
                        record=read(directory/'record.json')
                        if record['status']!='success' or record['cost'].get('saturated') is not False:raise ValueError('unknown/saturated cost')
                        if record['cost_model']!=meta['cost_model'] or record['prepared_digest']!=meta['package_digest']:raise ValueError('model/prepared package identity drift')
                        if model=='legacy' and record['cost_model']!='sm90-layout-lex9-legacy-v2':raise ValueError('wrong legacy semantic identity')
                        if model=='calibrated' and not record['cost_model'].startswith('sm90-layout-calibrated-v3-'):raise ValueError('wrong calibrated semantic identity')
                        write(directory/'calibration_spec.json',spec)
                        write(directory/'configuration.json',dict(spec=spec,model=model,scope=scope,arm=arm,entry='frisk_execution',grid=[1,1,1],threads=128))
                        result=assess_file(directory/'execution.json',include_emission=True)
                        tt=result.pop('ttgir',None);info=result.pop('execution_config',None)
                        write(directory/'support.json',result)
                        if result.get('status')!='supported':raise RuntimeError('registered backend arm unavailable: '+str(result))
                        (directory/'emitted.ttgir').write_text(tt);write(directory/'emitted-config.json',info)
                        names=('execution.json','record.json','output.mlir','calibration_spec.json','configuration.json','support.json','emitted.ttgir','emitted-config.json')
                        write(directory/'generation-inputs.json',{n:sha(directory/n) for n in names})
                        row.update(status='generated',generation_inputs_sha256=sha(directory/'generation-inputs.json'),assignment_sha256=record['complete_assignment_sha256'],
                            model_id=record['cost_model'],cost=record['cost'],oracle=record.get('oracle'),features=features(record) if model=='calibrated' else None)
                        write(run/'manifest.json',manifest)
        verify_protocol(run);manifest['status']='generated'
    except Exception as e:
        manifest.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:write(run/'manifest.json',manifest)
    return manifest

def execute(directory,load_only=False,safety=False,safety_tool=None):
    return closure.execute(directory,load_only=load_only,safety=safety,safety_tool=safety_tool,numerical_fixture=fixture)

def validate(run,*,worker_script=None):
    worker_script=__file__ if worker_script is None else str(worker_script)
    manifest=read(run/'manifest.json');verify_protocol(run)
    if manifest['status']!='generated':raise ValueError('generated matrix required')
    state=closure.gpu_state();write(run/'gpu-before.json',state)
    if state['foreign_pids']:raise RuntimeError('foreign workload before validation')
    manifest['status']='validating';write(run/'manifest.json',manifest);accepted={}
    try:
        for row in manifest['rows']:
            verify_protocol(run);directory=Path(row['path']);state=closure.gpu_state();write(directory/'gpu-before-configuration.json',state)
            if state['foreign_pids']:raise RuntimeError('foreign workload before configuration')
            result=invoke([sys.executable,worker_script,'execute',directory],directory/'compile-numerics.log')
            row['phases'].append(dict(stage='native-and-numerics',**result))
            from cost_calibration import machine_artifacts
            row['machine_evidence']=machine_artifacts(directory);c=read(directory/'correctness.json')
            row.update(image_identity_sha256=sha(directory/'image-identity.json'),correctness_sha256=sha(directory/'correctness.json'),cubin_sha256=c['cubin_sha256'],physical_resources=c['physical_resources'])
            write(run/'manifest.json',manifest)
            key=closure.safety_key(directory,c)
            if key in accepted:row['safety']=dict(status='exact-reuse',from_path=accepted[key],identity=key)
            else:
                row['safety']=[]
                for tool in verify_protocol(run)['safety']['tools']:
                    result=invoke(['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','91',sys.executable,worker_script,'execute',directory,'--load-only','--safety','--safety-tool',tool],directory/('safety-'+tool+'.log'))
                    row['safety'].append(dict(tool=tool,identity=key,correctness_sha256=sha(directory/('safety-'+tool+'-correctness.json')),**result))
                accepted[key]=str(directory)
            row['execution_identity']=fixture.digest(dict(cubin_sha256=c['cubin_sha256'],entry='frisk_execution',execution_config=read(directory/'execution_config.json'),physical_abi=c['physical_params']))
            row['status']='validated';row['artifact_hashes']={str(x.relative_to(directory)):sha(x) for x in directory.rglob('*') if x.is_file()}
            write(run/'manifest.json',manifest);print(json.dumps(dict(validated=directory.name)),flush=True)
        verify_protocol(run);state=closure.gpu_state();write(run/'gpu-after.json',state)
        if state['foreign_pids']:raise RuntimeError('foreign workload after validation')
        manifest['status']='validated'
    except Exception as e:
        manifest.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:write(run/'manifest.json',manifest)
    return manifest

def measurement_groups(run,phase=None):
    p=verify_protocol(run);manifest=read(run/'manifest.json')
    if manifest['status']!='validated':raise ValueError('all rows must be validated')
    expected={(scope,s['key'],model,label) for s in p['specifications'] for scope in (['controlled','natural'] if s['case_id']!='attention' else ['natural']) for model in ('legacy','calibrated') for label in (['LI','J','A','B'] if scope=='controlled' else ['LI','J'])}
    actual={(r['scope'],r['spec']['key'],r['model'],r['selection']) for r in manifest['rows']}
    if actual!=expected or len(actual)!=len(manifest['rows']):raise ValueError('registered matrix incomplete/duplicated')
    groups={};registered={s['key']:s for s in p['specifications']}
    for row in manifest['rows']:
        if row['spec']!=registered[row['spec']['key']] or row['conversion']!='C' or row['arm']!=row['model']+'-'+row['selection']:raise ValueError('registered specification/policy substituted')
        prepared=Path(row['prepared']);meta=read(prepared/'prepared.json')
        if sha(prepared/'prepared.json')!=row['prepared_sha256'] or any(meta[k]!=v for k,v in row['domain'].items()):raise ValueError('prepared domain changed')
        for name,key in (('input.mlir','input_sha256'),('domains.json','domain_file_sha256')):
            if sha(prepared/name)!=meta[key]:raise ValueError('prepared package bytes changed')
        directory=Path(row['path']);closure.verify_accepted_row(directory,row)
        config=read(directory/'configuration.json');record=read(directory/'record.json')
        if config['scope']!=row['scope'] or config['arm']!=row['arm'] or record['cost']!=row['cost'] or record['complete_assignment_sha256']!=row['assignment_sha256'] or record['cost_model']!=row['model_id']:raise ValueError('configuration/record substituted')
        identity=fixture.digest(dict(cubin_sha256=row['cubin_sha256'],entry='frisk_execution',execution_config=read(directory/'execution_config.json'),physical_abi=read(directory/'correctness.json')['physical_params']))
        if identity!=row['execution_identity']:raise ValueError('execution alias identity changed')
        c=read(Path(row['path'])/'correctness.json');row=dict(row,resource_comparison=c['resource_comparison'])
        groups.setdefault(row['scope']+'-'+row['spec']['key'],[]).append(row)
    aliases={};unique={}
    for key,rows in groups.items():
        byidentity={};mapping={}
        for row in sorted(rows,key=lambda r:r['arm']):
            identity=row['execution_identity']
            if identity not in byidentity:byidentity[identity]=row
            mapping[row['arm']]=dict(arm=byidentity[identity]['arm'],execution_identity=identity)
        aliases[key]=mapping;unique[key]=list(byidentity.values())
    path=run/'measurement-aliases.json'
    if path.exists() and read(path)!=aliases:raise ValueError('execution alias mapping changed')
    if not path.exists():write(path,aliases)
    return unique

def measurement_batch_plan(spec,b,provider=None):
    """Resolve one allocation/input phase without importing or touching CUDA."""
    if type(b) is not int or b<0:raise ValueError("nonnegative batch index required")
    if provider is None:
        return dict(input_batch=b,input_offset=1+b%3 if spec["abi_stratum"]=="weak-vector" else 128*(b%3),
                    output_offset=1+b%3,metadata={},arm_order=None)
    if not callable(provider):raise TypeError("explicit batch provider must be callable")
    value=provider(spec,b)
    if not isinstance(value,dict) or set(value)!={"input_batch","input_offset","output_offset","metadata","arm_order"}:
        raise ValueError("explicit batch provider schema differs")
    if (type(value['input_batch']) is not int or not 0<=value['input_batch']<(1<<31)
        or any(type(value[n]) is not int or not 0<=value[n]<=256 for n in ('input_offset','output_offset'))
        or not isinstance(value['metadata'],dict)
        or not isinstance(value['arm_order'],list) or not value['arm_order']
        or any(not isinstance(a,str) or not a for a in value['arm_order'])
        or len(set(value['arm_order']))!=len(value['arm_order'])):
        raise ValueError("explicit batch input/offset/order domain differs")
    # A JSON copy prevents a stateful provider from mutating already-validated
    # allocation identities. Wire types distinguish bool/int and int/float.
    return json.loads(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False))


def measurement_batch_plans(protocol,groups,provider=None):
    count=protocol['measurement']['batches']
    if type(count) is not int or count<1:raise ValueError('positive preregistered batch count required')
    explicit=protocol['measurement'].get('batch_provider_contract')
    if provider is None:
        if explicit is not None:raise ValueError('registered explicit schedule requires its provider')
    elif not callable(provider):raise TypeError('explicit batch provider must be callable')
    elif explicit!='explicit-preregistered-batches-v1':raise ValueError('unregistered explicit batch provider')
    registered=protocol['measurement'].get('batch_plan')
    if provider is not None and (not isinstance(registered,dict) or set(registered)!=set(groups)):
        raise ValueError('registered batch-plan group coverage differs')
    result={}
    for key,rows in groups.items():
        if not rows or any(row['spec']!=rows[0]['spec'] for row in rows):raise ValueError('common nonempty specification per group required')
        arms=[row['arm'] for row in rows]
        plans=[measurement_batch_plan(rows[0]['spec'],b,provider) for b in range(count)]
        if provider is not None:
            if not isinstance(registered[key],list) or len(registered[key])!=count:raise ValueError('registered batch count differs')
            for b,plan in enumerate(plans):
                if set(plan['arm_order'])!=set(arms) or len(plan['arm_order'])!=len(arms):raise ValueError('registered arm order differs')
                if json.dumps(plan,sort_keys=True,separators=(",",":"),allow_nan=False)!=json.dumps(registered[key][b],sort_keys=True,separators=(",",":"),allow_nan=False):
                    raise ValueError('provider changed the preregistered input/offset schedule')
        result[key]=plans
    return result


def measure(run,out,*,suite=None,groups=None,batch_provider=None):
    phase="qualification" if batch_provider is None else "explicit-preregistered-probe"
    if batch_provider is not None and not callable(batch_provider):raise TypeError("explicit batch provider must be callable")
    numeric=fixture if suite is None else suite
    import torch
    from run_joint_ablation import load_exact,attention_inputs,attention_reference
    from joint_measure import gpu_state,foreign_pids,measured_tensormaps
    from async_host import physical_abi,validate_launch,TensorMaps
    p=verify_protocol(run);groups=measurement_groups(run) if groups is None else groups;batch_plans=measurement_batch_plans(p,groups,batch_provider);out=fresh(out);identity=source_identity();registration=read(run/"registration.json");tools=registration["tools"];inputs_guard={str(run/"protocol.json"):sha(run/"protocol.json"),str(run/"registration.json"):sha(run/"registration.json"),str(run/"measurement-aliases.json"):sha(run/"measurement-aliases.json")};inputs_guard.update(registration["immutable_inputs"] if "immutable_inputs" in registration else {str(ROOT/fixture.PROFILE_RELATIVE):sha(ROOT/fixture.PROFILE_RELATIVE)});check_inputs(inputs_guard)
    write(out/"freeze.json",dict(run=str(run.resolve()),protocol_sha256=sha(run/"protocol.json"),phase=phase,source=identity,tools=tools,
                                 manifest_sha256=sha(run/"manifest.json"),input_hashes=inputs_guard,groups=groups))
    closure.activate_target_device()
    if foreign_pids(gpu_state()):raise RuntimeError("foreign GPU workload before timing")
    rng=np.random.default_rng(p["measurement"]["seed"]);summaries=[];observations=[]
    try:
        for key,rows in groups.items():
            entries={};setup=[];spec=rows[0]["spec"];arms=[r["arm"] for r in rows]
            for row in rows:
                d=Path(row["path"])
                for name,h in row["artifact_hashes"].items():
                    if sha(d/name)!=h:raise ValueError("validated artifact changed: "+str(d/name))
                start=time.perf_counter_ns();cuda,module,function,binary=load_exact(d);load_ns=time.perf_counter_ns()-start
                from physical_resources import query_physical_resources
                from resource_prediction import confirm_driver_resources
                confirm_driver_resources(read(d/"resolved-compiler-resources.json"),query_physical_resources(cuda,function),loaded_cubin_sha256=hashlib.sha256(binary).hexdigest(),loaded_entry="frisk_execution")
                info=read(d/"execution_config.json");abi=physical_abi((d/"kernel.ptx").read_text(),len(info["tma_plans"]))
                phases=[]
                for b in range(p["measurement"]["batches"]):
                    plan=batch_plans[key][b];input_batch=plan["input_batch"]
                    if spec["case_id"]=="attention":_,q,k,v=attention_inputs()[input_batch];ref=attention_reference(q,k,v)
                    else:q,k,v=numeric.inputs(spec,input_batch);ref=numeric.reference(spec,q,k,v)
                    io=plan["input_offset"];oo=plan["output_offset"]
                    start=time.perf_counter_ns();buffers=[]
                    for x in (q,k,v):
                        buf=torch.full((x.nbytes+272,),165,dtype=torch.uint8,device="cuda");buf[io:io+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(buf)
                    output=torch.full((ref.nbytes+272,),165,dtype=torch.uint8,device="cuda");pointers=[x.data_ptr()+io for x in buffers]+[output.data_ptr()+oo];validate_launch(info,pointers);closure.validate_runtime_regions(pointers,[q.nbytes,k.nbytes,v.nbytes,ref.nbytes])
                    maps=measured_tensormaps(info,pointers) if info["tma_plans"] else None
                    torch.cuda.synchronize()
                    if maps:
                        accepted=TensorMaps(info,pointers);torch.cuda.synchronize()
                        if not all(torch.equal(x,y) for x,y in zip(maps.device,accepted.device)):raise ValueError("TMA encoder mismatch")
                    args=[ctypes.c_uint64(x) for x in pointers+([x.data_ptr() for x in maps.device] if maps else [])+[0]]
                    if len(args)!=len(abi):raise ValueError("physical launch ABI")
                    params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
                    def launch(cuda=cuda,function=function,params=params):
                        status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                        if status:raise RuntimeError("cuLaunchKernel "+str(status))
                    content=dict(key=key,batch=b,input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)],
                                 reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),input_offset=io,output_offset=oo)
                    if batch_provider is not None:content.update(input_batch=input_batch,preregistered_batch=plan["metadata"],preregistered_arm_order=plan["arm_order"])
                    phases.append(dict(launch=launch,buffers=buffers,output=output,maps=maps,args=args,params=params,ref=ref,
                                       saved=[x.cpu().numpy().copy() for x in buffers],content=content,
                                       pair_id=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(",",":")).encode()).hexdigest()))
                    if batch_provider is not None:phases[-1]["actual_pointer_phases"]=[dict(role=role,mod32=pointer%32,mod128=pointer%128,mod4096=pointer%4096) for role,pointer in zip(("q","k","v","output"),pointers[:4])]
                    setup.append(dict(arm=row["arm"],batch=b,module_load_ns=load_ns,allocation_descriptor_upload_sync_ns=time.perf_counter_ns()-start,
                                      descriptor_encode_ns=maps.setup_ns if maps else [],descriptor_upload_ms=maps.upload_ms if maps else []))
                entries[row["arm"]]=dict(phases=phases,cuda=cuda,module=module,function=function,binary=binary)
            write(out/(key+"-setup.json"),setup)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            def check(ph):
                stream.synchronize();raw=ph["output"].cpu().numpy();oo=ph["content"]["output_offset"];ref=ph["ref"];got=raw[oo:oo+ref.nbytes].copy().view(ref.dtype).reshape(ref.shape)
                valid=numeric.check_numerics(spec,got,ref)["passed"]
                if not valid or not np.all(raw[:oo]==165) or not np.all(raw[oo+ref.nbytes:]==165):raise ValueError("measurement numerical/guard preflight")
                if not all(np.array_equal(x.cpu().numpy(),saved) for x,saved in zip(ph["buffers"],ph["saved"])):raise ValueError("input guard preflight")
            def events(fn,count=None):
                if count is None:count=p["measurement"]["replays"]
                result=[]
                for _ in range(count):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();fn();end.record();end.synchronize();result.append(begin.elapsed_time(end))
                return result
            pilots=[];preflight=[]
            with torch.cuda.stream(stream):
                for arm in arms:
                    for b,ph in enumerate(entries[arm]["phases"]):
                        ph["output"].fill_(165);ph["launch"]();check(ph);graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=stream):ph["launch"]()
                        ph["output"].fill_(165);graph.replay();check(ph);preflight.append(dict(arm=arm,batch=b,pair_id=ph["pair_id"],ordinary=True,capture=True,guards=True,inputs_unchanged=True))
                    launch=entries[arm]["phases"][0]["launch"]
                    for _ in range(p["measurement"]["warmup"]):launch()
                    stream.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(p["measurement"]["pilot_nodes"]):launch()
                    pilots.append(dict(arm=arm,graph_ms=events(graph.replay),nodes=p["measurement"]["pilot_nodes"]))
            count=max(1,min(p["measurement"]["max_nodes"],int(p["measurement"]["target_graph_ms"]/max(np.median(x["graph_ms"])/x["nodes"] for x in pilots))))
            write(out/(key+"-preflight.json"),preflight);write(out/(key+"-pilot.json"),dict(nodes=count,records=pilots))
            records=[]
            for b in range(p["measurement"]["batches"]):
                before=gpu_state()
                if foreign_pids(before):raise RuntimeError("foreign GPU workload before paired batch; no batch sampled")
                order=(batch_plans[key][b]["arm_order"] if batch_provider is not None else rng.permutation(arms).tolist());batch=[]
                with torch.cuda.stream(stream):
                    for arm in order:
                        ph=entries[arm]["phases"][b];launch=ph["launch"]
                        for _ in range(p["measurement"]["warmup"]):launch()
                        stream.synchronize();start=time.perf_counter_ns();graph=torch.cuda.CUDAGraph(keep_graph=True)
                        with torch.cuda.graph(graph,stream=stream):
                            for _ in range(count):launch()
                        stream.synchronize();capture_ns=time.perf_counter_ns()-start
                        cuda=entries[arm]["cuda"];cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)];actual=ctypes.c_size_t()
                        if cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(actual)) or actual.value!=count:raise ValueError("actual graph nodes mismatch")
                        start=time.perf_counter_ns();graph.instantiate();instantiate_ns=time.perf_counter_ns()-start;elapsed=events(graph.replay)
                        batch.append(dict(batch=b,arm=arm,pair_id=ph["pair_id"],pair_content=ph["content"],order=order,valid=True,
                                          launch_count=count,actual_graph_nodes=actual.value,graph_elapsed_ms=elapsed,samples_ms=[x/count for x in elapsed],
                                          capture_sync_ns=capture_ns,instantiate_ns=instantiate_ns,cubin_sha256=next(r["cubin_sha256"] for r in rows if r["arm"]==arm)))
                        if batch_provider is not None:batch[-1]["actual_pointer_phases"]=ph["actual_pointer_phases"]
                after=gpu_state();foreign=foreign_pids(before)+foreign_pids(after)
                for r in batch:r.update(valid=not foreign,foreign_pids=foreign,state_before=before,state_after=after)
                records+=batch;write(out/(key+"-raw.json"),records)
                print(json.dumps(dict(key=key,batch=b,foreign=foreign)),flush=True)
            analysis=paired_analysis(records,arms,seed=p["measurement"]["seed"])
            if batch_provider is not None:analysis.update(formal=analysis["n"]==p["measurement"]["batches"],expected_batches=p["measurement"]["batches"],scope="explicit registered probe; whole paired batches are independent units")
            summaries.append(dict(key=key,spec=spec,analysis=analysis));write(out/"analysis.json",summaries)
            if not analysis["formal"]:raise RuntimeError("insufficient valid paired batches; retained, no replacement")
            for row in rows:
                observations.append(dict(spec=spec,scope=row["scope"],candidate=row["arm"],arm=row["arm"],features=row["features"],median_us=analysis["arms"][row["arm"]]["median_us"],
                                         cubin_sha256=row["cubin_sha256"],measurement_sha256=sha(out/(key+"-raw.json")),physical_resources=row["physical_resources"],
                                         resource_comparison=row["resource_comparison"],record_path=str(Path(row["path"])/"record.json"),valid=True))
            write(out/"observations.json",observations)
        if source_identity()!=identity:raise ValueError("source changed during formal timing")
        check_tools(tools);check_inputs(inputs_guard);verify_protocol(run)
        evidence={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file()}
        write(out/"completion.json",dict(status="complete",run=str(run.resolve()),phase=phase,protocol_sha256=sha(run/"protocol.json"),
              manifest_sha256=sha(run/"manifest.json"),freeze_sha256=sha(out/"freeze.json"),
              evidence_hashes=evidence,source_unchanged=True,tools_unchanged=True,groups=len(groups)))
    except Exception as e:
        write(out/"failure.json",dict(status="failed",error=type(e).__name__+": "+str(e),partial_samples_retained=True));raise
    return out



def assess(run):
    from selection_quality_analysis import analyze_group,policy_ratio,summarize_suite,paired_policy_interval
    p=verify_protocol(run);groups=measurement_groups(run);m=run/'measurement'
    completion=read(m/'completion.json');freeze=read(m/'freeze.json')
    if completion.get('status')!='complete' or completion.get('freeze_sha256')!=sha(m/'freeze.json'):raise ValueError('incomplete measurement')
    for ledger in (completion,freeze):
        if ledger['run']!=str(run.resolve()) or ledger['phase']!='qualification' or ledger['protocol_sha256']!=sha(run/'protocol.json') or ledger['manifest_sha256']!=sha(run/'manifest.json'):raise ValueError('measurement run/protocol/phase/manifest mismatch')
    if freeze['source']!=source_identity():raise ValueError('measurement source drift')
    check_inputs(freeze['input_hashes']);check_tools(freeze['tools'])
    actual={str(x.relative_to(m)) for x in m.rglob('*') if x.is_file() and x!=m/'completion.json'}
    if set(completion['evidence_hashes'])!=actual:raise ValueError('measurement evidence file set changed')
    for name,digest in completion['evidence_hashes'].items():
        if sha(m/name)!=digest:raise ValueError('measurement evidence changed')
    manifest=read(run/'manifest.json');aliases=read(run/'measurement-aliases.json');analyses={};oracles=[]
    for key,unique in groups.items():
        rows=[r for r in manifest['rows'] if r['scope']+'-'+r['spec']['key']==key]
        selections={model+'-'+sel:aliases[key][model+'-'+sel]['arm'] for model in ('legacy','calibrated') for sel in ('LI','J')}
        candidates=[]
        for row in unique:
            scores={};score_sources={}
            if row['scope']=='controlled':
                for model in ('legacy','calibrated'):
                    matched=[r for r in rows if r['model']==model and r['selection'] in ('A','B') and r['execution_identity']==row['execution_identity']]
                    if not matched:raise ValueError('selected image outside complete forced candidate reference')
                    scores[model]=min(r['cost']['cost'] for r in matched)
                    score_sources[model]=[dict(arm=r['arm'],cost=r['cost']['cost']) for r in matched]
            candidates.append(dict(arm=row['arm'],cubin_sha256=row['cubin_sha256'],execution_identity=row['execution_identity'],model_scores=scores,score_sources=score_sources))
        result=analyze_group(read(m/(key+'-raw.json')),candidates,selections,
            expected_batches=p['measurement']['batches'],bootstrap_seed=p['measurement']['seed'],bootstrap_samples=p['measurement']['bootstrap_resamples'])
        result.update(spec=rows[0]['spec'],scope=rows[0]['scope'],candidates=candidates,
            reference_domain='exhausted two-producer-candidate complete assignment domain' if rows[0]['scope']=='controlled' else 'union of the four policy-selected executable images; not the entire initialized candidate domain')
        analyses[key]=result
        if rows[0]['scope']=='controlled':
            for row in rows:
                if row['selection'] not in ('LI','J'):continue
                r=read(Path(row['path'])/'record.json');o=r['oracle']
                passed=o['status']=='complete' and o['exhausted'] is True
                if row['selection']=='J':passed=passed and r['cost']['cost']==o['best_cost']['cost']
                else:
                    second=r['stage2_oracle'];first=r['stage1_oracle']
                    passed=passed and first['status']=='complete' and first['exhausted'] is True and r['stage1_producer_optimum_tuple'] is True and second['status']=='complete' and second['exhausted'] is True and r['cost']['cost']==second['best_cost']['cost']
                oracles.append(dict(key=key,arm=row['arm'],passed=bool(passed),record_sha256=sha(Path(row['path'])/'record.json')))
                if not passed:raise ValueError('exact oracle quality check failed')
    write(run/'group-quality.json',analyses);write(run/'oracle-quality.json',oracles)
    suites={}
    for scope in ('controlled','natural'):
        held={k:v for k,v in analyses.items() if v['scope']==scope and v['spec']['split']=='holdout'}
        suites[scope]=summarize_suite(held)
    resources=[]
    for row in manifest['rows']:
        d=Path(row['path']);c=read(d/'correctness.json');compiler=read(d/'resolved-compiler-resources.json')
        resources.append(dict(key=row['spec']['key'],scope=row['scope'],arm=row['arm'],physical=c['physical_resources'],confirmation=c['compiler_resource_confirmation'],comparison=c['resource_comparison'],plan=read(d/'resource-plan-before-native.json'),compiler=compiler,spill_count=None,compile_wall_ns=read(d/'compile.json')['wall_ns']))
    write(run/'resource-quality.json',resources)
    output=dict(schema_version=1,status='assessed',protocol_sha256=sha(run/'protocol.json'),source_digest=source_identity()['source_digest'],
        coverage=dict(configurations=len(manifest['rows']),validated=sum(r['status']=='validated' for r in manifest['rows']),groups=len(groups),unique_timed_images=sum(len(x) for x in groups.values()),controlled_oracle_checks=len(oracles)),
        suites=suites,prior_profile_transfer=True,static_physical_register_prediction='not-qualified; compiler/driver feedback used',
        limitations=['natural reference is selected-image union, not exhaustive candidate optimum','single CTA SM90 support domain','no new fitting; frozen prior coefficients evaluated under GF2 v3 semantics','kernel-only graph amortization; compile/load/search costs recorded separately'])
    thresholds=p['acceptance']['bounded_selection_quality']
    quality={}
    for scope,suite in suites.items():
        quality[scope]={}
        for policy,value in suite['policies'].items():
            observed=value['geomean_regret_ratio']<=thresholds['geomean_regret_max'] and value['max_case_regret_ratio']<=thresholds['max_case_regret_max']
            certain=value['geomean_regret_ci95'][1]<=thresholds['geomean_regret_max'] and value['max_case_regret_ci95'][1]<=thresholds['max_case_regret_max']
            failed=value['geomean_regret_ci95'][0]>thresholds['geomean_regret_max'] or value['max_case_regret_ci95'][0]>thresholds['max_case_regret_max']
            quality[scope][policy]=dict(status='qualified' if certain else 'failed' if failed else 'uncertain',observed_threshold_pass=observed)
    primary=suites['natural']['policy_comparisons']['calibrated-J/legacy-J']
    anchor_checks={k:policy_ratio(v,'calibrated-J','legacy-J') for k,v in analyses.items() if v['scope']=='natural' and v['spec']['split']!='holdout'}
    hold_checks={k:policy_ratio(v,'calibrated-J','legacy-J') for k,v in analyses.items() if v['scope']=='natural' and v['spec']['split']=='holdout'}
    promotion=p['acceptance']['calibrated_promotion']
    upper_limit=promotion['geomean_calibrated_over_legacy_ci95_upper_lt']
    guard_groups={k:v for k,v in analyses.items() if v['scope']=='natural'}
    familywise_guards={k:paired_policy_interval(v,'calibrated-J','legacy-J',alpha=.05/len(guard_groups)) for k,v in guard_groups.items()}
    no_regression=all(v['ratio_ci'][1]<=promotion['per_case_calibrated_over_legacy_ci95_upper_max'] for v in familywise_guards.values())
    regret_qualified=all(quality[scope]['calibrated-J']['status']=='qualified' for scope in ('controlled','natural'))
    promote=primary['ratio_ci95'][1]<upper_limit and no_regression and regret_qualified
    output.update(status='complete',quality_gates=quality,promotion=dict(primary=primary,anchors=anchor_checks,holdouts=hold_checks,per_case_familywise_guards=familywise_guards,
        no_per_case_regression=no_regression,bounded_regret_qualified=regret_qualified,passed=promote,
        recommendation='calibrated-eligible-for-explicit-default-change' if promote else 'retain-legacy-v2',
        rule='fixed six natural J holdouts: paired geomean upperCI<1/1.03; each holdout/Attention anchor familywise upperCI<=1.05; controlled and natural calibrated J regret qualified',
        production_default_changed=False))
    write(run/'quality-summary.json',output)
    return output

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('init','generate','validate','execute','measure','assess','run'))
    parser.add_argument('directory',type=Path);parser.add_argument('--out',type=Path)
    parser.add_argument('--load-only',action='store_true');parser.add_argument('--safety',action='store_true')
    parser.add_argument('--safety-tool',choices=('memcheck','racecheck','synccheck','initcheck'))
    a=parser.parse_args();d=a.directory.resolve()
    if a.command=='execute':
        try:execute(d,a.load_only,a.safety,a.safety_tool)
        except Exception as e:
            write(d/'worker-failure.json',dict(stage=getattr(e,'stage','worker'),code=getattr(e,'code','exception'),reason=type(e).__name__+': '+str(e)));raise
    elif a.command=='init':init(d)
    elif a.command=='generate':generate(d)
    elif a.command=='validate':validate(d)
    elif a.command=='measure':measure(d,a.out or d/'measurement')
    elif a.command=='assess':assess(d)
    elif a.command=='run':init(d);generate(d);validate(d);measure(d,d/'measurement');assess(d)
if __name__=='__main__':main()
