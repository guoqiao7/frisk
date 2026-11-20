"""Shared-budget archives, exact batch compilation and partial-cost qualification.
Strict numerics/sanitizers/timing reuse the accepted engines, never a copied GPU
loop.  Candidate decisions are complete-pool receipts fixed before event timing.
"""
import argparse,json,sys,time
from pathlib import Path
import gf2_decision_fixture as suite
import decision_priority as priority
import support_closure as closure
import selection_quality as quality
import demand_repair as demand
import demand_coverage_audit,native_surviving_audit
from cost_calibration import source_identity,check_inputs,check_tools
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'guoqiao/optimize_plan/gf2_decision_refinement_artifacts'
BENCH=demand.BENCH;OPT=demand.OPT
sha=closure.sha;read=closure.read;write=closure.write;fresh=closure.fresh
SEALED=(('model_selection_quality_artifacts','988dd10985b836ebf5a0c7fa204597f8782a6483ab9a057a4fac1ec2b6a501f7'),
 ('demand_cost_search_repair_artifacts','9b2fd7dbcdb60be193666a2c73bb14f1c1d776263f25134a09bebd48e57a77c5'),
 ('native_cost_repair_artifacts','66b28196c5065d5cc4504d3a069991b766718ec7d5618163533c120ba33ded9b'),
 ('partial_cost_efficiency_artifacts','eff4c3ca39681c91115e258638a34f5242ca1f4eb8d984a67835928657337fef'))


def preserved_indices():return [dict(path=str(ROOT/'guoqiao/optimize_plan'/n/'evidence-index.json'),sha256=h) for n,h in SEALED]

def audit_preserved(out):
    out=Path(out).resolve()
    if out.exists() or not out.is_relative_to(BASE.resolve()):raise ValueError('fresh new-stage audit path required')
    proofs=[]
    for item in preserved_indices():
        index=Path(item['path']);assert sha(index)==item['sha256'];data=read(index);contained=(ROOT/data['contained_root']).resolve();seen=set();total=0
        for row in data['entries']:
            raw=ROOT/row['path'];p=raw.resolve()
            if raw.is_symlink() or p in seen or not p.is_relative_to(contained) or any(x.is_symlink() for x in raw.parents if x.is_relative_to(contained)):raise ValueError('unsafe sealed entry')
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('old sealed content changed')
            seen.add(p);total+=row['bytes']
        if seen!={p.resolve() for p in contained.rglob('*') if p.is_file() and p!=index}:raise ValueError('sealed fileset changed')
        proofs.append(dict(**item,files=len(seen),bytes=total,exact_fileset=True))
    out.parent.mkdir(parents=True,exist_ok=True);write(out,dict(status='passed',indices=proofs));return proofs


def init(run,configuration, *, expected_local_weight=6418):
    from native_cost_calibration import MODEL_FEATURES
    from compatible_native_profiles import load_profile,base_profile_path,dependency_hashes
    configuration=Path(configuration).resolve();cfg=read(configuration)
    new=load_profile(cfg['native']['profile']);old=load_profile(cfg['old_native']['profile'])
    if ((expected_local_weight is not None and new['coefficients']['local_transfer_bytes']!=expected_local_weight)
        or new['training_maxima']['local_transfer_bytes']<=0):raise ValueError('registered empirically covered LOCAL coefficient required')
    if any(new['coefficients'][n]!=old['coefficients'][n] for n in MODEL_FEATURES if n!='local_transfer_bytes'):raise ValueError('incremental calibration changed frozen nonLOCAL coefficients')
    cfg['preserved_evidence']=preserved_indices();profiles=[];immutable={str(configuration):sha(configuration)}
    for section in ('static','native','old_native'):
        path=Path(cfg[section]['profile']).resolve();profiles.append(dict(path=str(path),sha256=sha(path)));immutable[str(path)]=sha(path)
        if section!='static':
            dependencies=dependency_hashes(path);base=Path(base_profile_path(path)).resolve()
            if str(base) not in dependencies:raise ValueError('native profile base is not dependency-bound')
            for filename,digest in dependencies.items():
                dependency=Path(filename)
                if not dependency.is_absolute() or sha(dependency)!=digest:raise ValueError('native profile dependency changed')
                immutable[str(dependency)]=digest
    cfg['profiles']=profiles
    cfg['native_profile_bases']={name:str(Path(base_profile_path(cfg[name]['profile'])).resolve()) for name in ('native','old_native')}
    for filename,digest in cfg['training_inputs'].items():
        path=Path(filename)
        if not path.is_absolute() or sha(path)!=digest:raise ValueError('registered training input changed')
        immutable[filename]=digest
    for item in preserved_indices():
        if sha(item['path'])!=item['sha256']:raise ValueError('sealed index changed')
        immutable[item['path']]=item['sha256']
    protocol=suite.protocol(cfg);run=fresh(run);write(run/'protocol.json',protocol);immutable[str(run/'protocol.json')]=sha(run/'protocol.json')
    tools=closure.frozen_tools(OPT)
    for path in (BENCH,Path('/usr/local/cuda/bin/nvdisasm')):tools['hashes'][str(path.resolve())]=sha(path)
    registration=dict(created_ns=time.time_ns(),source=source_identity(),tools=tools,optimizer=str(OPT),protocol_sha256=sha(run/'protocol.json'),protocol_digest=suite.digest(protocol),immutable_inputs=immutable)
    write(run/'registration.json',registration);write(run/'manifest.json',dict(status='registered',registration_sha256=sha(run/'registration.json'),rows=[]));return run


def verify(run):
    p=quality.verify_protocol(Path(run))
    if p['stage']!=suite.STAGE:raise ValueError('wrong protocol stage')
    return p


def check_archive(record,archive,mode,capacity):
    c=record['candidate_archive_search'];t=c['total'];local=c['local_end'];entries=archive['entries']
    if archive.get('schema_version')!=1 or c.get('version')!='shared-complete-candidate-archive-v1':raise ValueError('unsupported archive schema')
    if c['limit']!=65536 or c['refinement_limit']!=256 or c['capacity']!=capacity or archive['capacity']!=capacity:raise ValueError('archive budget substituted')
    for value in (t,local):
        if any(type(value[n]) is not int or value[n]<0 for n in ('probes','relationship','complete_scores','refinement','refinement_probes')):raise ValueError('invalid counters')
        if value['refinement']!=value['refinement_probes']:raise ValueError('refinement receipt drift')
        if value['probes']!=value['relationship']+value['complete_scores']+value['refinement']:raise ValueError('unaccounted archive work')
    if any(local[n]>t[n] for n in ('probes','relationship','complete_scores','refinement','refinement_probes')):raise ValueError('classified local counters exceed total')
    if t['probes']>65536 or t['refinement']>256 or local['probes']>t['probes'] or local['refinement']>t['refinement']:raise ValueError('shared budget reset or exceeded')
    quota=16384 if mode=='joint' else 65536;refine=256
    if c['local_terminal_probes']!=quota or c['local_terminal_refinement']!=refine or local['probes']>quota or local['refinement']>refine:raise ValueError('local quota exceeded')
    if mode=='local' and (c.get('joint_status') not in ('not-requested','not-requested-local-control','not-run','skipped','') or t!=local):raise ValueError('local baseline performed joint work')
    if not 1<=len(entries)<=capacity or c['retained']!=len(entries):raise ValueError('archive capacity/coverage mismatch')
    if archive.get('prepared_digest')!=record.get('prepared_digest'):raise ValueError('archive package substituted')
    if archive.get('search')!=c or record.get('total')!=t:raise ValueError('archive/record cumulative counters drift')
    contract=record.get('budget_contract',{})
    if contract.get('kind')!='shared-total-search-probes-v1' or contract.get('search_performed') is not True or contract.get('limit')!=65536 or contract.get('refinement_limit')!=256 or contract.get('used')!=t['probes'] or contract.get('refinement_used')!=t['refinement']:raise ValueError('outer budget receipt drift')
    if [e.get('index') for e in entries]!=list(range(len(entries))) or any(type(e.get('index')) is not int for e in entries):raise ValueError('invalid archive indices')
    identities=[e['complete_assignment_sha256'] for e in entries]
    if len(identities)!=len(set(identities)):raise ValueError('archive exact assignment duplicate')
    for e in entries:
        if e.get('certificate_verified') is not True or e['prepared_digest']!=archive['prepared_digest'] or e['conversion_policy']!='independent' or e['cost']['saturated']:raise ValueError('unverified archive candidate')
        if any(type(e[n]) is not int or e[n]<0 for n in ('probes_at_observation','refinement_at_observation')):raise ValueError('invalid observation counter')
        if e['probes_at_observation']>t['probes'] or e['refinement_at_observation']>t['refinement']:raise ValueError('free/outside-budget archive observation')
    if record['complete_assignment_sha256'] not in identities:raise ValueError('actual solver incumbent lost')
    if c['local_status']=='complete-witness':
        l=c['local']
        if l.get('selection_mode')!='informed-producer-first' or l.get('local_objective')!='own-operations-input-repairs-and-resources-v2' or l.get('incumbent_full_score_budget_charged') is not True or l.get('total')!=local:raise ValueError('local informed scope/budget receipt drift')
        if not any(e['local_incumbent'] for e in entries):raise ValueError('local witness lost')
    if c.get('joint_status')=='complete-witness' and c.get('joint',{}).get('total')!=t:raise ValueError('joint cumulative receipt drift')
    if mode=='joint' and c['joint_status']=='complete-witness' and not any(e['joint_incumbent'] for e in entries):raise ValueError('joint witness lost')
    for component in c.get('joint_components',[]):
        if component.get('has_seed_witness'):
            a=component.get('seed_start_cost');b=component.get('selected_component_cost')
            if a and b and (a['saturated'] or b['saturated'] or b['cost']>a['cost']):raise ValueError('constructed witness lost')
    return c


def static_fallback_entry(record,archive):
    return min(archive['entries'],key=lambda e:(e['cost']['cost'],e['complete_assignment_sha256']!=record['complete_assignment_sha256'],e['index']))


def add_case(run,m,spec,arm,model,prepared,meta,directory,*,assignment=None,entry=None,phases=None,replay_arguments=(),replay_validator=None):
    from execution_support import assess_file
    d=Path(directory);p=verify(run);cfg=p['configuration'];args=[BENCH,'--prepared='+str(prepared),'--case='+spec['case_id'],'--selection=J','--conversion=C','--demand-ledger','--minify-execution-json','--output-dir='+str(d)]
    if model=='surviving':args+=['--cost-profile='+cfg['static']['profile']]
    args+=list(replay_arguments)
    if assignment:args+=['--assignment='+str(assignment),'--seed-beam-solution=false']
    else:args+=['--seed-beam-solution=true']
    row=dict(spec=spec,scope='natural',arm=arm,model=model,selection='J',conversion='C',path=str(d),status='generating',phases=list(phases or []),prepared=str(prepared),prepared_sha256=sha(prepared/'prepared.json'),domain={k:meta[k] for k in ('common_digest','domain_digest','coverage_digest')})
    m['rows'].append(row);write(run/'manifest.json',m);result=closure.invoke(args,run/'generation-logs'/(spec['key']+'-'+arm+'.log'));row['phases'].append(dict(stage='assignment-replay' if assignment else'native-selection',**result,command_metadata_sha256=sha(result['command_metadata'])))
    if result['exit_code']:row['status']='failed';raise RuntimeError('declared candidate unavailable: '+arm)
    record=read(d/'record.json')
    if record['status']!='success' or record['cost']['saturated'] or record['prepared_digest']!=meta['package_digest']:raise ValueError('invalid full candidate')
    if assignment:
        if record.get('selection_bypassed') is not True or record['budget_contract']['used']!=0 or record['budget_contract']['refinement_used']!=0:raise ValueError('archive replay searched')
        if record['complete_assignment_sha256']!=entry['complete_assignment_sha256'] or record['cost']['cost']!=entry['cost']['cost']:raise ValueError('archive/replay complete identity or cost drift')
        if replay_validator is not None:replay_validator(record,entry)
    else:demand.check_search(record,dict(selection='J',incumbent_witness=True))
    from emission_audit_session import current_factory
    emission_factory=current_factory();sharing_receipt=None
    if emission_factory is None:
        cov=demand_coverage_audit.audit(read(d/'execution.json'),record)
        if spec['case_id']!='attention' and cov['status']!='passed':raise ValueError('emission coverage failed')
        write(d/'demand-coverage-audit.json',cov)
        if model=='surviving':write(d/'surviving-coverage-audit.json',native_surviving_audit.audit(read(d/'execution.json'),record))
        support=assess_file(d/'execution.json',include_emission=True);tt=support.pop('ttgir',None);info=support.pop('execution_config',None);write(d/'support.json',support)
        if not support.get('supported'):raise ValueError('backend unsupported')
        (d/'emitted.ttgir').write_text(tt);write(d/'emitted-config.json',info);write(d/'calibration_spec.json',spec);write(d/'configuration.json',dict(spec=spec,scope='natural',arm=arm,model=model,entry='frisk_execution',grid=[1,1,1],threads=128))
    else:
        packet=read(d/'execution.json')
        configuration=dict(spec=spec,scope='natural',arm=arm,model=model,entry='frisk_execution',grid=[1,1,1],threads=128)
        bindings=[d/'execution.json',d/'record.json',run/'protocol.json',run/'registration.json',run/'manifest.json',prepared/'prepared.json']
        if assignment:bindings.append(assignment)
        with emission_factory(packet,record,bindings=bindings,extra_identity=configuration) as shared:
            cov=shared.demand_audit()
            if spec['case_id']!='attention' and cov['status']!='passed':raise ValueError('emission coverage failed')
            write(d/'demand-coverage-audit.json',cov)
            if model=='surviving':write(d/'surviving-coverage-audit.json',shared.surviving_audit())
            support=shared.assess(include_emission=True);tt=support.pop('ttgir',None);info=support.pop('execution_config',None);write(d/'support.json',support)
            if not support.get('supported'):raise ValueError('backend unsupported')
            (d/'emitted.ttgir').write_text(tt);write(d/'emitted-config.json',info);write(d/'calibration_spec.json',spec);write(d/'configuration.json',configuration)
            sharing_receipt=shared.receipt(artifacts=dict(ttgir=d/'emitted.ttgir',typed_config=d/'emitted-config.json',configuration=d/'configuration.json'))
        write(d/'emission-sharing-receipt.json',sharing_receipt)
    names=('execution.json','record.json','output.mlir','calibration_spec.json','configuration.json','support.json','emitted.ttgir','emitted-config.json','demand-coverage-audit.json')
    if model=='surviving':names+=('surviving-coverage-audit.json',)
    if sharing_receipt is not None:names+=('emission-sharing-receipt.json',)
    write(d/'generation-inputs.json',{n:sha(d/n) for n in names})
    row.update(status='generated',generation_inputs_sha256=sha(d/'generation-inputs.json'),assignment_sha256=record['complete_assignment_sha256'],model_id=record['cost_model'],cost=record['cost'],features=record['cost'].get('features'),coverage=cov['status'])
    if 'complete_program_sha256' in record:row['complete_program_sha256']=record['complete_program_sha256']
    if entry:row['archive_entry']=entry
    write(run/'manifest.json',m);return row


def generate(run, *, archive_validator=None, archive_arguments=(), replay_argument_provider=None, replay_validator=None):
    run=Path(run);p=verify(run);cfg=p['configuration'];k=cfg.get('archive_capacity',4);m=read(run/'manifest.json')
    if m['status']!='registered':raise ValueError('fresh run required')
    m['status']='generating';write(run/'manifest.json',m);(run/'generation-logs').mkdir();pools={}
    try:
        for spec in p['specifications']:
            base=fresh(run/'preparation'/spec['key']);raw=base/'input.mlir';raw.write_text(suite.source(spec,ROOT));preps={};common=None
            for model,profile in [('surviving',cfg['static']['profile']),('legacy',None)]:
                prep=base/('prepared-'+model);args=[BENCH,'--prepare',raw,spec['case_id'],prep]+(['--cost-profile='+profile] if profile else[])
                if spec['case_id']!='attention':args+=['--natural']
                result=demand.invoke(args,base/('prepare-'+model+'.log'));meta=read(prep/'prepared.json');domain={n:meta[n] for n in ('common_digest','domain_digest','coverage_digest')}
                if common is not None and common!=domain:raise ValueError('reference/common domains differ')
                common=domain;preps[model]=(prep,meta,result)
            prep,meta,pr=preps['legacy'];legacy=add_case(run,m,spec,'legacy-static','legacy',prep,meta,run/'cases'/(spec['key']+'-legacy-static'),phases=[dict(stage='preparation',**pr,command_metadata_sha256=sha(pr['command_metadata']))]);group=dict(spec=spec,legacy='legacy-static',modes={})
            for mode in ('joint','local'):
                prep,meta,pr=preps['surviving'];search=run/'search'/(spec['key']+'-'+mode)
                args=[BENCH,'--prepared='+str(prep),'--case='+spec['case_id'],'--selection='+('J' if mode=='joint' else'LI'),'--conversion=C','--cost-profile='+cfg['static']['profile'],'--candidate-archive='+str(k),'--seed-beam-solution='+str(mode=='joint').lower(),'--demand-ledger','--minify-execution-json','--output-dir='+str(search)]
                args+=list(archive_arguments)
                result=demand.invoke(args,run/'generation-logs'/(spec['key']+'-'+mode+'-archive.log'));record=read(search/'record.json');archive=read(search/'candidate-archive.json');receipt=(archive_validator or check_archive)(record,archive,mode,k);arms=[]
                for entry in archive['entries']:
                    arm=mode+'-c'+str(entry['index']);assignment=base/(arm+'-assignment.json');write(assignment,entry)
                    add_case(run,m,spec,arm,'surviving',prep,meta,run/'cases'/(spec['key']+'-'+arm),assignment=assignment,entry=entry,replay_arguments=replay_argument_provider(entry) if replay_argument_provider else (),replay_validator=replay_validator,phases=[dict(stage='preparation',**pr,command_metadata_sha256=sha(pr['command_metadata'])),dict(stage='archive-search',**result,command_metadata_sha256=sha(result['command_metadata']))]);arms.append(arm)
                # Min full static cost within actual retained archive; exact tie
                # prefers the genuine solver incumbent. This does no new search.
                static=static_fallback_entry(record,archive)
                old_order=[mode+'-c'+str(e['index']) for e in sorted(archive['entries'],key=lambda e:(e['cost']['cost'],e['complete_assignment_sha256']!=record['complete_assignment_sha256'],e['index']))]
                group['modes'][mode]=dict(arms=arms,fallback=mode+'-c'+str(static['index']),static_order=old_order,search=str(search),record_sha256=sha(search/'record.json'),archive_sha256=sha(search/'candidate-archive.json'),receipt=receipt)
            pools['natural-'+spec['key']]=group;write(run/'candidate-pools.json',pools)
        verify(run);m.update(status='generated',candidate_pools_sha256=sha(run/'candidate-pools.json'))
    except Exception as e:m.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:write(run/'manifest.json',m)
    return m


def compile_batch(run, *, cache_dir=None):
    from native_compile_cache import compile_many
    run=Path(run);verify(run);m=read(run/'manifest.json');reg=read(run/'registration.json')
    if m['status']!='generated' or m.get('compiled'):raise ValueError('fresh generated matrix required')
    start=time.perf_counter_ns();result=compile_many([Path(r['path']) for r in m['rows']],run/'native-artifact-cache' if cache_dir is None else cache_dir,source=reg['source'],tools=reg['tools'])
    write(run/'batch-compilation.json',dict(status='complete',wall_ns=time.perf_counter_ns()-start,result=result));m['compiled']=True;m['batch_compilation_sha256']=sha(run/'batch-compilation.json');write(run/'manifest.json',m);verify(run)
    return result


def execute(directory,load_only=False,safety=False,safety_tool=None):
    from native_compile_cache import verify_precompiled
    return closure.execute(directory,load_only=load_only,safety=safety,safety_tool=safety_tool,numerical_fixture=suite,compiler_provider=verify_precompiled)


def strict_decision(analyses,profile,fallback):
    from native_cost_calibration import score
    feedback={k:score(v,profile) for k,v in analyses.items()};missing=[k for k,v in feedback.items() if v['status']!='supported']
    selected=fallback if missing else min(analyses,key=lambda k:(feedback[k]['cost'],k!=fallback,k))
    return dict(selected_source=selected,status='fallback' if missing else'exact-ranked',reasons=missing,uses_gpu_timing=False)


def calculate_decisions(pool,analyses,old,new,evidence, *, registered_labels=None):
    from partial_cost_selection import cost_interval,select_partial_cost,_profile
    old_weights,_=_profile(old);new_weights,_=_profile(new)
    if any(old_weights[n]!=new_weights[n] for n in old_weights if n!='local_transfer_bytes'):raise ValueError('rule ablation changed a frozen nonLOCAL coefficient')
    # Every rule uses this same fixed new profile; old binds calibration lineage only.
    result={'legacy-static':dict(selected_source=pool['legacy'],status='static',uses_gpu_timing=False)}
    for mode in ('joint','local'):
        info=pool['modes'][mode];subset={arm:analyses[arm] for arm in info['arms']};ev={arm:evidence[arm] for arm in info['arms']}
        original=info['fallback'];old_priority=[original]+sorted(a for a in subset if a!=original)
        intervals={a:cost_interval(n,new) for a,n in subset.items()}
        result[mode+'-static']=dict(selected_source=original,status='static',uses_gpu_timing=False)
        for rule,gf2,frame in suite.FACTORIAL_RULES:
            fallback=priority.static_fallback(ev,info['static_order'],enabled=frame)
            order=priority.native_priority(intervals,ev,old_priority,enabled=gf2)
            decision=select_partial_cost(intervals,fallback_arm=fallback['payload']['selected_source'],priority=order['payload']['priority'])
            result[mode+'-'+rule]=dict(decision,intervals=intervals,priority_receipt=order,fallback_receipt=fallback,
                fixed_original_priority=old_priority,rule=rule)
        fallback=priority.static_fallback(ev,info['static_order'],enabled=True)
        matrix=priority.affine_matrix(intervals,ev,new)
        decision=priority.select_affine(matrix,fallback_arm=fallback['payload']['selected_source'],priority=old_priority)
        payload=decision['payload']
        result[mode+'-affine-frame']=dict(selected_source=payload['selected_source'],status=payload['status'],uses_gpu_timing=False,
            fixed_original_priority=old_priority,intervals=intervals,affine_matrix=matrix,affine_selection=decision,fallback_receipt=fallback,
            rule='affine-frame')
    if set(result)!=set(suite.POLICY_LABELS if registered_labels is None else registered_labels):raise ValueError('registered decision policy matrix differs')
    return result


def analyze(run, *, evidence_provider=None, decision_provider=None):
    import native_cost_analysis as native
    from native_cost_calibration import attach_symbolic_banks
    from compatible_native_profiles import load_profile
    run=Path(run);p=verify(run);m=read(run/'manifest.json');cfg=p['configuration']
    if m['status']!='validated' or (run/'partial-decisions.json').exists() or (run/'measurement').exists():raise ValueError('validated/pre-timing analysis once required')
    old=load_profile(cfg['old_native']['profile']);new=load_profile(cfg['native']['profile']);pools=read(run/'candidate-pools.json');all_decisions={}
    for key,pool in pools.items():
        analyses={};hashes={};evidence={};evidence_hashes={}
        rows=[r for r in m['rows'] if r['spec']==pool['spec']]
        for row in rows:
            directory=Path(row['path']);closure.verify_accepted_row(directory,row);start=time.perf_counter_ns();analysis=native.analyze(directory,cache_dir=run/'native-analysis-cache')
            if analysis.get('identity',{}).get('cubin_sha256')!=row['cubin_sha256']:raise ValueError('native image substituted')
            if analysis.get('status')=='available':analysis=attach_symbolic_banks(analysis,directory)
            write(directory/'native-analysis.json',analysis);analyses[row['arm']]=analysis;hashes[row['arm']]=sha(directory/'native-analysis.json');row['native_analysis_sha256']=hashes[row['arm']];row['native_analysis_wall_ns']=time.perf_counter_ns()-start;row['artifact_hashes']['native-analysis.json']=hashes[row['arm']]
            if row['model']=='surviving':
                receipt=(evidence_provider or priority.candidate_evidence)(directory,row);write(directory/'decision-evidence.json',receipt);evidence[row['arm']]=receipt;evidence_hashes[row['arm']]=sha(directory/'decision-evidence.json');row['priority_evidence_sha256']=evidence_hashes[row['arm']];row['artifact_hashes']['decision-evidence.json']=evidence_hashes[row['arm']]
        choices=(decision_provider or calculate_decisions)(pool,analyses,old,new,evidence);all_decisions[key]=dict(spec=pool['spec'],choices=choices,analysis_hashes=hashes,priority_evidence_hashes=evidence_hashes)
    write(run/'partial-decisions.json',dict(schema_version=2,decision_contract=priority.CONTRACT,affine_contract=priority.AFFINE_CONTRACT,protocol_sha256=sha(run/'protocol.json'),candidate_pools_sha256=sha(run/'candidate-pools.json'),old_profile_sha256=sha(cfg['old_native']['profile']),local_profile_sha256=sha(cfg['native']['profile']),decided_before_measurement=True,uses_gpu_timing=False,groups=all_decisions));write(run/'manifest.json',m);return all_decisions


def measurement_groups(run, *, evidence_verifier=None, decision_provider=None, archive_validator=None):
    from compatible_native_profiles import load_profile
    run=Path(run);p=verify(run);cfg=p['configuration'];m=read(run/'manifest.json');pools=read(run/'candidate-pools.json');d=read(run/'partial-decisions.json')
    if m['status']!='validated' or d['decided_before_measurement'] is not True or d['uses_gpu_timing'] is not False or d['protocol_sha256']!=sha(run/'protocol.json') or d['candidate_pools_sha256']!=sha(run/'candidate-pools.json'):raise ValueError('decision/pool drift')
    if d['old_profile_sha256']!=sha(cfg['old_native']['profile']) or d['local_profile_sha256']!=sha(cfg['native']['profile']):raise ValueError('profile drift')
    old=load_profile(cfg['old_native']['profile']);new=load_profile(cfg['native']['profile']);groups={};aliases={};allseen=set()
    if sha(run/'candidate-pools.json')!=m['candidate_pools_sha256']:raise ValueError('registered pools changed')
    registered={'natural-'+spec['key']:spec for spec in p['specifications']}
    if set(pools)!=set(registered) or any(pool['spec']!=registered[key] for key,pool in pools.items()):raise ValueError('registered specifications substituted')
    for key,pool in pools.items():
        rows=[r for r in m['rows'] if r['spec']==pool['spec']];expected={pool['legacy']}|set(pool['modes']['joint']['arms'])|set(pool['modes']['local']['arms'])
        if len(rows)!=len(expected) or {r['arm'] for r in rows}!=expected:raise ValueError('archive candidate omitted or added')
        analyses={};images={};mapping={};evidence={}
        for mode,info in pool['modes'].items():
            search=Path(info['search'])
            if sha(search/'record.json')!=info['record_sha256'] or sha(search/'candidate-archive.json')!=info['archive_sha256']:raise ValueError('archive receipts changed')
            a=read(search/'candidate-archive.json');sr=read(search/'record.json');(archive_validator or check_archive)(sr,a,mode,cfg.get('archive_capacity',4))
            expected_order=[mode+'-c'+str(e['index']) for e in sorted(a['entries'],key=lambda e:(e['cost']['cost'],e['complete_assignment_sha256']!=sr['complete_assignment_sha256'],e['index']))]
            if info['arms']!=[mode+'-c'+str(e['index']) for e in a['entries']] or info['fallback']!=mode+'-c'+str(static_fallback_entry(sr,a)['index']) or info['static_order']!=expected_order:raise ValueError('archive membership/static fallback substituted')
            if set(pool['modes'])!={'joint','local'} or a['cost_model']!=cfg['static']['id']:raise ValueError('archive model/mode substituted')
            for e in a['entries']:
                matches=[row for row in rows if row['arm']==mode+'-c'+str(e['index'])]
                if len(matches)!=1 or matches[0].get('archive_entry')!=e or matches[0]['assignment_sha256']!=e['complete_assignment_sha256'] or matches[0]['cost']['cost']!=e['cost']['cost']:raise ValueError('archive entry replay substituted')
        for row in sorted(rows,key=lambda r:r['arm']):
            path=Path(row['path'])
            if path.is_symlink() or path.resolve().parent!=(run/'cases').resolve():raise ValueError('candidate path escapes run')
            allseen.add(str(path));closure.verify_accepted_row(path,row)
            config=read(path/'configuration.json')
            if row['conversion']!='C' or config['scope']!='natural' or config['arm']!=row['arm']:raise ValueError('candidate policy substituted')
            if sha(path/'native-analysis.json')!=d['groups'][key]['analysis_hashes'][row['arm']]:raise ValueError('native evidence changed')
            analyses[row['arm']]=read(path/'native-analysis.json');record=read(path/'record.json')
            if row['model']=='surviving':
                if sha(path/'decision-evidence.json')!=row['priority_evidence_sha256'] or sha(path/'decision-evidence.json')!=d['groups'][key]['priority_evidence_hashes'][row['arm']]:raise ValueError('priority evidence changed')
                evidence[row['arm']]=(evidence_verifier or priority.verify_candidate_evidence)(path,row,read(path/'decision-evidence.json'))
            if record['cost_model']!=row['model_id'] or record['complete_assignment_sha256']!=row['assignment_sha256'] or record['cost']!=row['cost']:raise ValueError('assignment changed')
            if 'complete_program_sha256' in record or 'complete_program_sha256' in row.get('archive_entry',{}):
                program=record.get('complete_program_sha256')
                if not isinstance(program,str) or len(program)!=64 or row.get('complete_program_sha256')!=program:raise ValueError('complete program identity changed')
                if row['model']=='surviving' and row.get('archive_entry',{}).get('complete_program_sha256')!=program:raise ValueError('archive complete program identity changed')
            for phase in row['phases']:
                if phase.get('command_metadata_sha256') and sha(phase['command_metadata'])!=phase['command_metadata_sha256']:raise ValueError('command work timer changed')
            prepared=Path(row['prepared']);meta=read(prepared/'prepared.json')
            if sha(prepared/'prepared.json')!=row['prepared_sha256'] or any(meta[k]!=value for k,value in row['domain'].items()):raise ValueError('prepared package changed')
            for name,field in (('input.mlir','input_sha256'),('domains.json','domain_file_sha256')):
                if sha(prepared/name)!=meta[field]:raise ValueError('prepared payload changed')
            c=read(path/'correctness.json');identity=suite.digest(dict(cubin_sha256=row['cubin_sha256'],entry='frisk_execution',execution_config=read(path/'execution_config.json'),physical_abi=c['physical_params']))
            if identity!=row['execution_identity']:raise ValueError('physical execution changed')
            images.setdefault(identity,dict(row,resource_comparison=c['resource_comparison']));mapping[row['arm']]=dict(arm=images[identity]['arm'],execution_identity=identity)
        if (decision_provider or calculate_decisions)(pool,analyses,old,new,evidence)!=d['groups'][key]['choices']:raise ValueError('partial decision certificate changed')
        groups[key]=list(images.values());aliases[key]=mapping
    if len(allseen)!=len(m['rows']) or len(groups)!=len(p['specifications']):raise ValueError('matrix coverage mismatch')
    value=dict(groups=aliases,partial_decisions_sha256=sha(run/'partial-decisions.json'));path=run/'measurement-aliases.json'
    if path.exists() and read(path)!=value:raise ValueError('aliases changed')
    if not path.exists():write(path,value)
    return groups


def assess(run, *, group_provider=None):
    from selection_quality_analysis import analyze_group,summarize_suite,paired_policy_interval
    run=Path(run);p=verify(run);groups=(group_provider or measurement_groups)(run);out=run/'measurement';c=read(out/'completion.json');f=read(out/'freeze.json')
    if c['status']!='complete' or c['freeze_sha256']!=sha(out/'freeze.json'):raise ValueError('incomplete measurement')
    for value in (c,f):
        if value['run']!=str(run.resolve()) or value['phase']!='qualification' or value['manifest_sha256']!=sha(run/'manifest.json') or value['protocol_sha256']!=sha(run/'protocol.json'):raise ValueError('measurement registration drift')
    demand.assert_source_inputs(f['source'],f['input_hashes'],f['tools'])
    if {str(x.relative_to(out)) for x in out.rglob('*') if x.is_file() and x.name!='completion.json'}!=set(c['evidence_hashes']):raise ValueError('raw file set changed')
    for name,h in c['evidence_hashes'].items():
        if sha(out/name)!=h:raise ValueError('raw hash changed')
    decisions=read(run/'partial-decisions.json')['groups'];aliases=read(run/'measurement-aliases.json')['groups'];analyses={}
    for key,rows in groups.items():
        selections={name:aliases[key][choice['selected_source']]['arm'] for name,choice in decisions[key]['choices'].items()}
        candidates=[dict(arm=r['arm'],cubin_sha256=r['cubin_sha256'],execution_identity=r['execution_identity'],model_scores={}) for r in rows]
        analyses[key]=analyze_group(read(out/(key+'-raw.json')),candidates,selections,expected_batches=10,bootstrap_seed=p['seed'],bootstrap_samples=10000);analyses[key]['spec']=rows[0]['spec']
    fresh={k:v for k,v in analyses.items() if v['spec']['split']=='holdout'};summary=summarize_suite(fresh,bootstrap_seed=p['seed']);name=p['configuration']['main_policy'];pol=summary['policies'][name]
    qualified=pol['geomean_regret_ci95'][1]<=1.03 and pol['max_case_regret_ci95'][1]<=1.10;failed=pol['geomean_regret_ci95'][0]>1.03 or pol['max_case_regret_ci95'][0]>1.10
    primary=summary['policy_comparisons'][name+'/legacy-static'];guards={k:paired_policy_interval(v,name,'legacy-static',alpha=.05/p['acceptance']['promotion']['guard_comparisons']) for k,v in analyses.items()};guardok=all(v['ratio_ci'][1]<=1.05 for v in guards.values())
    write(run/'group-quality.json',analyses);result=dict(status='complete',coverage=dict(configurations=len(read(run/'manifest.json')['rows']),groups=len(groups),unique_images=sum(len(x) for x in groups.values())),suite=summary,bounded_regret=dict(status='qualified' if qualified else'failed' if failed else'uncertain'),promotion=dict(primary=primary,guards=guards,no_regression=guardok,kernel_criteria_passed=qualified and guardok and primary['ratio_ci95'][1]<1/1.03,passed=False,efficiency_status='pending-separate-efficiency-acceptance',production_default_changed=False),same_budget_local_comparison=summary['policy_comparisons'][name+'/'+name.replace('joint-','local-',1)],note='final release also requires separate efficiency evidence; no unrestricted GPU optimality or external framework claim')
    write(run/'quality-summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['audit-preserved','init','generate','compile','validate','execute','analyze','measure','assess']);p.add_argument('directory',type=Path);p.add_argument('--configuration',type=Path);p.add_argument('--load-only',action='store_true');p.add_argument('--safety',action='store_true');p.add_argument('--safety-tool');a=p.parse_args();d=a.directory.resolve()
    if a.command=='audit-preserved':audit_preserved(d)
    elif a.command=='init':init(d,a.configuration)
    elif a.command=='generate':generate(d)
    elif a.command=='compile':compile_batch(d)
    elif a.command=='validate':
        if not read(d/'manifest.json').get('compiled'):raise ValueError('batch compile required')
        quality.validate(d,worker_script=__file__)
    elif a.command=='execute':
        try:execute(d,a.load_only,a.safety,a.safety_tool)
        except Exception as e:write(d/'worker-failure.json',dict(stage=getattr(e,'stage','worker'),code=getattr(e,'code','exception'),reason=type(e).__name__+': '+str(e)));raise
    elif a.command=='analyze':analyze(d)
    elif a.command=='measure':quality.measure(d,d/'measurement',suite=suite,groups=measurement_groups(d))
    else:assess(d)
if __name__=='__main__':main()
