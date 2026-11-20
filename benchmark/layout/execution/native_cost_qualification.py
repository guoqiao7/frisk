"""Bounded native-feedback qualification using the shared strict GPU engines.
No GPU work occurs at import/init/generate/analyze. Validation and measurement
must be explicitly invoked after source/profile/protocol registration is frozen.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import native_cost_fixture as suite
import demand_repair as demand
import demand_coverage_audit
import native_surviving_audit
import selection_quality as quality
import support_closure as closure
from cost_calibration import source_identity,check_inputs,check_tools
from demand_search_audit import summarize_search,validate_failure_sidecars

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts'
BENCH=demand.BENCH
OPT=demand.OPT
sha=closure.sha
read=closure.read
write=closure.write
fresh=closure.fresh
SEALED=(('model_selection_quality_artifacts','988dd10985b836ebf5a0c7fa204597f8782a6483ab9a057a4fac1ec2b6a501f7'),
        ('demand_cost_search_repair_artifacts','9b2fd7dbcdb60be193666a2c73bb14f1c1d776263f25134a09bebd48e57a77c5'))


def preserved_indices():
    return [dict(path=str(ROOT/'guoqiao/optimize_plan'/name/'evidence-index.json'),sha256=digest) for name,digest in SEALED]


def audit_preserved(out):
    """Read every old contained file; write evidence exclusively in the new root."""
    out=Path(out).resolve()
    if not out.is_relative_to(BASE.resolve()):raise ValueError('preservation audit output must stay in new stage')
    if out.exists():raise ValueError('fresh preservation audit path required')
    result=[]
    for item in preserved_indices():
        index=Path(item['path'])
        if sha(index)!=item['sha256']:raise ValueError('old sealed index changed')
        data=read(index);contained=(ROOT/data['contained_root']).resolve();seen=set();total=0
        for row in data['entries']:
            raw=ROOT/row['path'];p=raw.resolve()
            if raw.is_symlink() or p in seen or not p.is_relative_to(contained) or any(x.is_symlink() for x in raw.parents if x.is_relative_to(contained)):
                raise ValueError('unsafe or duplicate old evidence path')
            if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('old sealed evidence changed: '+str(p))
            seen.add(p);total+=row['bytes']
        actual={p.resolve() for p in contained.rglob('*') if p.is_file() and p!=index}
        if actual!=seen:raise ValueError('old sealed evidence fileset changed')
        result.append(dict(**item,files=len(seen),bytes=total,exact_fileset=True))
    out.parent.mkdir(parents=True,exist_ok=True);write(out,dict(schema_version=1,status='passed',old_evidence_read_only=True,indices=result))
    return result


def init(run,configuration):
    from native_cost_calibration import load_profile
    from native_cost_analysis import DOMAIN,FEATURES as NATIVE_FEATURES
    configuration=Path(configuration).resolve();cfg=read(configuration);models=cfg['models'];native=cfg['native']
    native_profile=load_profile(native['profile'])
    if native.get('analysis_version')!=DOMAIN:raise ValueError('native analyzer domain identity mismatch')
    if tuple(m['name'] for m in models)!=suite.MODEL_NAMES:raise ValueError('ordered static model identities required')
    if models[0]['id']!='sm90-layout-lex9-legacy-v2' or not models[1]['id'].startswith('sm90-layout-calibrated-v5d-') or not models[2]['id'].startswith('sm90-layout-calibrated-v6d-'):
        raise ValueError('unexpected static semantic model identities')
    profiles=[];immutable={str(configuration):sha(configuration)}
    for path in [m['profile'] for m in models if m.get('profile')]+[native['profile']]:
        path=Path(path).resolve();profiles.append(dict(path=str(path),sha256=sha(path)));immutable[str(path)]=sha(path)
    for filename in ('inputs.json','fit.json','training-ledger.json'):
        path=Path(native['profile']).resolve().parent/filename;immutable[str(path)]=sha(path)
    native_inputs=read(Path(native['profile']).resolve().parent/'inputs.json')
    for name in ('known_completion','micro_completion'):
        filename=native_inputs[name];immutable[filename]=native_inputs[name+'_sha256']
    for filename,digest in cfg['training_inputs'].items():
        path=Path(filename)
        if not path.is_absolute() or sha(path)!=digest:raise ValueError('training input changed or not absolute')
        immutable[filename]=digest
    if cfg.get('preserved_evidence',preserved_indices())!=preserved_indices():raise ValueError('sealed inventory cannot be changed')
    for item in preserved_indices():
        if sha(item['path'])!=item['sha256']:raise ValueError('old index changed')
        immutable[item['path']]=item['sha256']
    p=suite.protocol(models=models,native=native,profiles=profiles,training_inputs=cfg['training_inputs'],preserved_evidence=preserved_indices())
    p['feature_contract'].update(emitted=list(demand.FEATURES),surviving=list(native_surviving_audit.FEATURES),native_raw=list(NATIVE_FEATURES),native_priced=native_profile['features'],
        static_survival='conservative wholeSSA closure rooted at final output and physically emitted communication; no ptxas CSE or exact native instruction claim',
        native='postcompile static opcode/width/dependency/order counters with typed support limits; GF2 bank context must bind exact emitted TTGIR and native transfer bytes',
        physical_resources='compiler/driver feedback; localbytes and native local instructions do not identify spill count')
    run=fresh(run);write(run/'protocol.json',p);immutable[str(run/'protocol.json')]=sha(run/'protocol.json')
    tools=closure.frozen_tools(OPT)
    for path in (BENCH,Path('/usr/local/cuda/bin/nvdisasm')):tools['hashes'][str(path.resolve())]=sha(path)
    registration=dict(schema_version=1,created_ns=time.time_ns(),source=source_identity(),tools=tools,optimizer=str(OPT),protocol_sha256=sha(run/'protocol.json'),
                      protocol_digest=suite.digest(p),immutable_inputs=immutable)
    write(run/'registration.json',registration);write(run/'manifest.json',dict(schema_version=1,status='registered',registration_sha256=sha(run/'registration.json'),rows=[]))
    return run


def verify(run):
    p=quality.verify_protocol(Path(run))
    if p['stage']!='native-cost-repair-qualification':raise ValueError('wrong protocol stage')
    return p


def portfolio_budget(rows,models):
    expected={x['arm'] for x in suite.arms(models)}
    if len(rows)!=6 or {x['arm'] for x in rows}!=expected:raise ValueError('complete six-source portfolio required')
    total=refinement=seed=0
    for row in rows:
        record=read(Path(row['path'])/'record.json');demand.check_search(record,row['treatment'])
        total+=record['total']['probes'];refinement+=record['total']['refinement']
        seed+=sum(c['seed_probes']+c['seed_scores'] for c in record['search_components'])
    if total>393216 or refinement>1536:raise ValueError('portfolio total budget exceeded')
    return dict(sources=6,total_probes=total,total_refinement=refinement,seed_work=seed,total_probe_limit=393216,total_refinement_limit=1536,
                interpretation='six charged solves, not one65536-probe solve')


def generate(run):
    from execution_support import assess_file
    run=Path(run);p=verify(run);m=read(run/'manifest.json')
    if m['status']!='registered':raise ValueError('fresh registered run required')
    m['status']='generating';write(run/'manifest.json',m)
    try:
        for spec in p['specifications']:
            base=fresh(run/'preparation'/spec['key']);raw=base/'input.mlir';raw.write_text(suite.source(spec,ROOT))
            preparations={};common=None
            for model in p['models']:
                prep=base/('prepared-'+model['name']);args=[BENCH,'--prepare',raw,spec['case_id'],prep]+demand.model_args(model)
                if spec['case_id']!='attention':args+=['--natural']
                prepare_result=demand.invoke(args,base/('prepare-'+model['name']+'.log'));meta=read(prep/'prepared.json')
                domain={k:meta[k] for k in ('common_digest','domain_digest','coverage_digest')}
                if meta['cost_model']!=model['id'] or common is not None and domain!=common:raise ValueError('models have different common domains')
                common=domain;preparations[model['name']]=(model,prep,meta,prepare_result)
            for treatment in suite.arms(p['models']):
                name=treatment['arm'];model,prep,meta,prepare_result=preparations[treatment['model']];d=run/'cases'/('natural-'+spec['key']+'-'+name)
                args=[BENCH,'--prepared='+str(prep),'--case='+spec['case_id'],'--selection='+treatment['selection'],'--conversion=C',
                      '--seed-beam-solution='+str(treatment['incumbent_witness']).lower(),'--demand-ledger','--output-dir='+str(d)]+demand.model_args(model)
                row=dict(spec=spec,scope='natural',arm=name,model=model['name'],selection=treatment['selection'],treatment=treatment,conversion='C',path=str(d),
                         status='generating',phases=[dict(stage='preparation',command_metadata_sha256=sha(prepare_result['command_metadata']),**prepare_result)],prepared=str(prep),prepared_sha256=sha(prep/'prepared.json'),domain=common)
                m['rows'].append(row);write(run/'manifest.json',m)
                outcome=closure.invoke(args,base/(name+'.log'));row['phases'].append(dict(stage='native-selection',command_metadata_sha256=sha(outcome['command_metadata']),**outcome))
                if outcome['exit_code']:
                    row['status']='failed'
                    if (d/'failure.json').exists():row['failure']=read(d/'failure.json')
                    if outcome['exit_code']==3:row['failure_budget_proof']=validate_failure_sidecars(d)
                    raise RuntimeError('declared source failed; retained, never dropped')
                record=read(d/'record.json');demand.check_search(record,treatment)
                if record['status']!='success' or record['cost']['saturated'] or record['cost_model']!=model['id'] or record['prepared_digest']!=meta['package_digest']:
                    raise ValueError('invalid/saturated/substituted selection')
                cov=demand_coverage_audit.audit(read(d/'execution.json'),record,require_priced_features=model['name']=='v5d' and spec['case_id']!='attention')
                if spec['case_id']!='attention' and cov['status']!='passed':raise ValueError('exact execution coverage failed')
                write(d/'demand-coverage-audit.json',cov)
                if model['name']=='surviving':write(d/'surviving-coverage-audit.json',native_surviving_audit.audit(read(d/'execution.json'),record))
                support=assess_file(d/'execution.json',include_emission=True);tt=support.pop('ttgir',None);info=support.pop('execution_config',None);write(d/'support.json',support)
                if not support.get('supported'):raise ValueError('declared backend unsupported')
                (d/'emitted.ttgir').write_text(tt);write(d/'emitted-config.json',info);write(d/'calibration_spec.json',spec)
                write(d/'configuration.json',dict(spec=spec,scope='natural',arm=name,model=model['name'],treatment=treatment,entry='frisk_execution',grid=[1,1,1],threads=128))
                names=('execution.json','record.json','output.mlir','calibration_spec.json','configuration.json','support.json','emitted.ttgir','emitted-config.json','demand-coverage-audit.json')
                if model['name']=='surviving':names+=('surviving-coverage-audit.json',)
                write(d/'generation-inputs.json',{n:sha(d/n) for n in names})
                row.update(status='generated',generation_inputs_sha256=sha(d/'generation-inputs.json'),assignment_sha256=record['complete_assignment_sha256'],
                           model_id=record['cost_model'],cost=record['cost'],features=record['cost'].get('features'),coverage=cov['status'])
                write(run/'manifest.json',m)
        if len(m['rows'])!=54:raise ValueError('complete54 source matrix required')
        budgets={s['key']:portfolio_budget([r for r in m['rows'] if r['spec']['key']==s['key']],p['models']) for s in p['specifications']}
        write(run/'portfolio-budget.json',budgets);verify(run);m['status']='generated'
    except Exception as e:m.update(status='failed',error=type(e).__name__+': '+str(e));raise
    finally:write(run/'manifest.json',m)
    return m


def execute(directory,load_only=False,safety=False,safety_tool=None):
    return closure.execute(directory,load_only=load_only,safety=safety,safety_tool=safety_tool,numerical_fixture=suite)


def choose_native(spec,feedback,source_arms):
    """Pure all-or-nothing ranking, before any kernel performance is observed."""
    if set(feedback)!=set(source_arms) or len(source_arms)!=6:raise ValueError('complete common feedback pool required')
    reasons=[];planned=suite.planned_native_fallback(spec)
    if planned:reasons.append(planned)
    for arm in source_arms:
        result=feedback[arm]
        if result.get('uses_gpu_timing') is not False and result.get('status')=='supported':raise ValueError('native score lacks no-timing certificate')
        if result.get('status')!='supported':reasons.append(arm+':'+str(result.get('reason','native-analysis-unsupported')))
        elif type(result.get('cost')) is not int or result['cost']<0:raise ValueError('invalid native integer cost')
    priority=['surviving-J-new','legacy-J-new','v5d-J-new','surviving-LI','legacy-LI','v5d-LI']
    chosen='surviving-J-new' if reasons else min(priority,key=lambda arm:(feedback[arm]['cost'],priority.index(arm)))
    return dict(status='common-fallback' if reasons else'native-ranked',selected_source=chosen,reasons=sorted(set(reasons)),
                selections={'native-pool':chosen,'native-LI-opportunity':chosen,'native-J-opportunity':chosen},
                candidates=list(source_arms),uses_gpu_timing=False,tie_rule='surviving-J-new,legacy-J-new,v5d-J-new,surviving-LI,legacy-LI,v5d-LI')


def verify_decision(spec,feedback,sources,decision):
    expected=choose_native(spec,feedback,sources)
    for field,value in expected.items():
        if decision.get(field)!=value:raise ValueError('native decision differs from deterministic frozen feedback')
    return expected


def analyze(run):
    import native_cost_analysis as native
    from native_cost_calibration import load_profile,score,attach_symbolic_banks
    run=Path(run);p=verify(run);m=read(run/'manifest.json')
    if m['status']!='validated' or (run/'native-decisions.json').exists() or (run/'measurement').exists():raise ValueError('analyze once after validation and before timing')
    profile=load_profile(p['native']['profile']);decisions={}
    for spec in p['specifications']:
        rows=[r for r in m['rows'] if r['spec']['key']==spec['key']];portfolio_budget(rows,p['models']);feedback={};sources=[]
        for row in rows:
            d=Path(row['path']);closure.verify_accepted_row(d,row);start=time.perf_counter_ns();analysis=native.analyze(d,cache_dir=run/'native-cache')
            if analysis.get('identity',{}).get('cubin_sha256')!=row['cubin_sha256']:raise ValueError('native feedback cubin mismatch')
            if analysis.get('status')=='available':analysis=attach_symbolic_banks(analysis,d)
            result=score(analysis,profile)
            if result.get('status')=='supported' and result.get('profile_sha256')!=suite.digest(profile):raise ValueError('native score profile identity mismatch')
            write(d/'native-analysis.json',analysis);write(d/'native-score.json',result)
            feedback[row['arm']]=result;sources.append(row['arm']);row['native_analysis_wall_ns']=time.perf_counter_ns()-start
            row['native_analysis_sha256']=sha(d/'native-analysis.json');row['native_score_sha256']=sha(d/'native-score.json')
            for n in ('native-analysis.json','native-score.json'):row['artifact_hashes'][n]=sha(d/n)
        decision=choose_native(spec,feedback,sources)
        decision.update(spec=spec,feedback_hashes={r['arm']:dict(analysis=r['native_analysis_sha256'],score=r['native_score_sha256']) for r in rows})
        decisions['natural-'+spec['key']]=decision
    write(run/'native-decisions.json',dict(schema_version=1,protocol_sha256=sha(run/'protocol.json'),profile_sha256=sha(p['native']['profile']),
                                           decided_before_measurement=True,uses_gpu_timing=False,groups=decisions))
    write(run/'manifest.json',m);verify(run)
    return decisions


def measurement_groups(run):
    from native_cost_calibration import load_profile,score
    run=Path(run);p=verify(run);m=read(run/'manifest.json');decisions=read(run/'native-decisions.json')
    if m['status']!='validated' or decisions.get('decided_before_measurement') is not True or decisions['protocol_sha256']!=sha(run/'protocol.json') or decisions['profile_sha256']!=sha(p['native']['profile']):raise ValueError('frozen validated pool/decision required')
    profile=load_profile(p['native']['profile'])
    expected={(s['key'],a['arm']) for s in p['specifications'] for a in suite.arms(p['models'])}
    if {(r['spec']['key'],r['arm']) for r in m['rows']}!=expected or len(m['rows'])!=54:raise ValueError('matrix incomplete')
    groups={};aliases={}
    for spec in p['specifications']:
        key='natural-'+spec['key'];rows=[r for r in m['rows'] if r['spec']==spec];portfolio_budget(rows,p['models']);images={};mapping={}
        if len(rows)!=6:raise ValueError('specification substituted')
        for row in sorted(rows,key=lambda r:r['arm']):
            d=Path(row['path']);closure.verify_accepted_row(d,row);record=read(d/'record.json');demand.check_search(record,row['treatment'])
            treatment=next(a for a in suite.arms(p['models']) if a['arm']==row['arm'])
            if row['treatment']!=treatment or read(d/'configuration.json')['treatment']!=treatment:raise ValueError('source policy substituted')
            for phase in row['phases']:
                if phase.get('stage') in ('preparation','native-selection') and sha(phase['command_metadata'])!=phase.get('command_metadata_sha256'):raise ValueError('search/prepare command metadata changed')
            prep=Path(row['prepared']);meta=read(prep/'prepared.json')
            if sha(prep/'prepared.json')!=row['prepared_sha256'] or any(meta[k]!=v for k,v in row['domain'].items()):raise ValueError('prepared identity changed')
            for name,key2 in (('input.mlir','input_sha256'),('domains.json','domain_file_sha256')):
                if sha(prep/name)!=meta[key2]:raise ValueError('prepared payload changed')
            if record['cost_model']!=row['model_id'] or record['complete_assignment_sha256']!=row['assignment_sha256'] or record['cost']!=row['cost']:raise ValueError('selected assignment changed')
            if sha(d/'native-analysis.json')!=decisions['groups'][key]['feedback_hashes'][row['arm']]['analysis'] or sha(d/'native-score.json')!=decisions['groups'][key]['feedback_hashes'][row['arm']]['score']:raise ValueError('feedback changed after selection')
            if score(read(d/'native-analysis.json'),profile)!=read(d/'native-score.json'):raise ValueError('native score differs from frozen feedback/profile rescore')
            c=read(d/'correctness.json');identity=suite.digest(dict(cubin_sha256=row['cubin_sha256'],entry='frisk_execution',execution_config=read(d/'execution_config.json'),physical_abi=c['physical_params']))
            if identity!=row['execution_identity']:raise ValueError('execution identity changed')
            images.setdefault(identity,dict(row,resource_comparison=c['resource_comparison']));mapping[row['arm']]=dict(arm=images[identity]['arm'],execution_identity=identity)
        feedback={r['arm']:read(Path(r['path'])/'native-score.json') for r in rows}
        verify_decision(spec,feedback,[r['arm'] for r in rows],decisions['groups'][key])
        groups[key]=list(images.values());aliases[key]=mapping
    content=dict(groups=aliases,native_decisions_sha256=sha(run/'native-decisions.json'))
    path=run/'measurement-aliases.json'
    if path.exists() and read(path)!=content:raise ValueError('frozen aliases/decision changed')
    if not path.exists():write(path,content)
    return groups


def feedback_overhead(rows,groups):
    """Disclose full portfolio setup work and estimated kernel-only amortization.
    This uses already collected timers and performance; it never reruns search or
    uses latency to change selection. The break-even is descriptive, not a gate.
    """
    result={}
    for key,analysis in groups.items():
        candidates=[r for r in rows if 'natural-'+r['spec']['key']==key]
        preparation={};solve={};compile_wall={};native_wall={}
        for row in candidates:
            for phase in row['phases']:
                if phase.get('stage') not in ('preparation','native-selection'):continue
                path=Path(phase['command_metadata'])
                if sha(path)!=phase['command_metadata_sha256']:raise ValueError('overhead command timer changed')
                timer=read(path)
                if timer.get('exit_code')!=0 or type(timer.get('elapsed_ns')) is not int or timer['elapsed_ns']<0:raise ValueError('invalid setup timer')
                (preparation if phase['stage']=='preparation' else solve)[str(path)]=timer['elapsed_ns']
            d=Path(row['path']);compile_wall[row['arm']]=read(d/'compile.json')['wall_ns'];native_wall[row['arm']]=row['native_analysis_wall_ns']
        if len(preparation)!=3 or len(solve)!=6 or len(compile_wall)!=6:raise ValueError('incomplete setup work ledger')
        total=sum(preparation.values())+sum(solve.values())+sum(compile_wall.values())+sum(native_wall.values())
        baseline=next(r for r in candidates if r['arm']=='legacy-J-new')
        baseline_phases=[x for x in baseline['phases'] if x.get('stage') in ('preparation','native-selection')]
        baseline_setup=sum(read(x['command_metadata'])['elapsed_ns'] for x in baseline_phases)+compile_wall['legacy-J-new']
        extra=max(0,total-baseline_setup)
        base=analysis['selections']['legacy-J-new']['median_us'];chosen=analysis['selections']['native-pool']['median_us'];saved=base-chosen
        result[key]=dict(preparations=3,source_solves=6,native_compile_requests=6,prepare_wall_ns=sum(preparation.values()),selection_command_wall_ns=sum(solve.values()),
            compile_wall_ns=sum(compile_wall.values()),native_analysis_wall_ns=sum(native_wall.values()),portfolio_setup_wall_ns=total,observed_legacy_components_wall_ns=baseline_setup,
            additional_observed_component_work_ns=extra,median_kernel_saving_us=saved,estimated_break_even_launches=extra/(saved*1000) if saved>0 else None,
            no_kernel_saving=saved<=0,cache_policy='as-observed ordinary compiler/cache state; baseline components came from this same portfolio order, not an independent cold single-policy run; summed work is not parallel wall time',
            exclusions='numerical/sanitizer validation and graph pilot/event overhead are reported by shared engine, excluded from model setup amortization')
    return result


def assess(run):
    from selection_quality_analysis import analyze_group,summarize_suite,paired_policy_interval
    run=Path(run);p=verify(run);groups=measurement_groups(run);measurement=run/'measurement';completion=read(measurement/'completion.json');freeze=read(measurement/'freeze.json')
    if completion.get('status')!='complete' or completion['freeze_sha256']!=sha(measurement/'freeze.json'):raise ValueError('measurement incomplete')
    for x in (freeze,completion):
        if x['run']!=str(run.resolve()) or x['protocol_sha256']!=sha(run/'protocol.json') or x['manifest_sha256']!=sha(run/'manifest.json'):raise ValueError('measurement registration changed')
    demand.assert_source_inputs(freeze['source'],freeze['input_hashes'],freeze['tools'])
    actual={str(x.relative_to(measurement)) for x in measurement.rglob('*') if x.is_file() and x!=measurement/'completion.json'}
    if actual!=set(completion['evidence_hashes']):raise ValueError('measurement exact fileset changed')
    for name,digest in completion['evidence_hashes'].items():
        if sha(measurement/name)!=digest:raise ValueError('measurement raw changed')
    aliases=read(run/'measurement-aliases.json')['groups'];decisions=read(run/'native-decisions.json')['groups'];analyses={};m=read(run/'manifest.json')
    for key,unique in groups.items():
        source={arm:v['arm'] for arm,v in aliases[key].items()};selected=dict(source)
        selected.update({name:source[arm] for name,arm in decisions[key]['selections'].items()})
        candidates=[]
        for row in unique:
            score=read(Path(row['path'])/'native-score.json');scores={'native':[score['cost']]} if score['status']=='supported' else{}
            candidates.append(dict(arm=row['arm'],execution_identity=row['execution_identity'],cubin_sha256=row['cubin_sha256'],model_scores=scores))
        analyses[key]=analyze_group(read(measurement/(key+'-raw.json')),candidates,selected,expected_batches=p['measurement']['batches'],
                                   bootstrap_seed=p['seed'],bootstrap_samples=p['measurement']['bootstrap_resamples'])
        analyses[key].update(spec=unique[0]['spec'],native_decision=decisions[key])
    fresh={k:v for k,v in analyses.items() if v['spec']['split']=='holdout'};supported={k:v for k,v in fresh.items() if suite.planned_native_fallback(v['spec']) is None}
    summaries={'all-six':summarize_suite(fresh,bootstrap_seed=p['seed']),'predeclared-four':summarize_suite(supported,bootstrap_seed=p['seed'])}
    policy=summaries['all-six']['policies']['native-pool'];lowgeo,highgeo=policy['geomean_regret_ci95'];lowmax,highmax=policy['max_case_regret_ci95']
    qualified=highgeo<=1.03 and highmax<=1.10;failed=lowgeo>1.03 or lowmax>1.10
    primary=summaries['all-six']['policy_comparisons']['native-pool/legacy-J-new']
    guards={k:paired_policy_interval(v,'native-pool','legacy-J-new',alpha=.05/9) for k,v in analyses.items()}
    guard_ok=all(v['ratio_ci'][1]<=1.05 for v in guards.values());promote=qualified and guard_ok and primary['ratio_ci95'][1]<1/1.03
    write(run/'native-group-analysis.json',analyses);write(run/'search-quality.json',summarize_search(m['rows'],checker=demand.check_search))
    overhead=[]
    for row in m['rows']:
        d=Path(row['path']);overhead.append(dict(spec=row['spec'],arm=row['arm'],compile_wall_ns=read(d/'compile.json')['wall_ns'],native_analysis_wall_ns=row['native_analysis_wall_ns'],
                                               cubin_sha256=row['cubin_sha256'],physical_resources=row['physical_resources'],spill_count=None))
    write(run/'compile-feedback-overhead.json',dict(observations=overhead,by_program=feedback_overhead(m['rows'],analyses)))
    result=dict(schema_version=1,status='complete',coverage=dict(configurations=len(m['rows']),groups=len(groups),unique_images=sum(len(x) for x in groups.values())),
                suites=summaries,bounded_regret=dict(status='qualified' if qualified else'failed' if failed else'uncertain'),
                promotion=dict(primary=primary,guards=guards,no_regression=guard_ok,passed=promote,production_default_changed=False),
                native_support={k:v['status'] for k,v in decisions.items()},native_LI_J_opportunity_identical=True,
                limitations=['common candidate pool only; no full-domain optimality','six solves/compiles, not single-solve equal work','native ranking uses no holdout latency; compiler feedback has disclosed cost',
                             'static native occurrence/dependency proxies are not dynamic counts or cycle simulation','spill counts and achieved occupancy remain unknown'])
    write(run/'quality-summary.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['audit-preserved','init','generate','validate','execute','analyze','measure','assess']);parser.add_argument('directory',type=Path)
    parser.add_argument('--configuration',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--load-only',action='store_true');parser.add_argument('--safety',action='store_true');parser.add_argument('--safety-tool',choices=['memcheck','racecheck','synccheck','initcheck']);a=parser.parse_args();d=a.directory.resolve()
    if a.command=='audit-preserved':audit_preserved(d)
    elif a.command=='init':
        if a.configuration is None:parser.error('--configuration required')
        init(d,a.configuration)
    elif a.command=='generate':generate(d)
    elif a.command=='validate':quality.validate(d,worker_script=__file__)
    elif a.command=='execute':
        try:execute(d,a.load_only,a.safety,a.safety_tool)
        except Exception as e:write(d/'worker-failure.json',dict(stage=getattr(e,'stage','worker'),code=getattr(e,'code','exception'),reason=type(e).__name__+': '+str(e)));raise
    elif a.command=='analyze':analyze(d)
    elif a.command=='measure':quality.measure(d,a.out or d/'measurement',suite=suite,groups=measurement_groups(d))
    elif a.command=='assess':assess(d)

if __name__=='__main__':main()
