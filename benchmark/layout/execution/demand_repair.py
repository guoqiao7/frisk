"""Demand-cost/search repair driver: known-image training and fresh qualification.
The shared strict execution/timing engine is parameterized explicitly; no module
or fixture monkeypatching. All directories are fresh and evidence is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import demand_repair_fixture as suite
import demand_coverage_audit
from demand_search_audit import summarize_search,validate_failure_sidecars
import selection_quality as quality
import support_closure as closure
from calibration_analysis import FEATURES as BASE_FEATURES,coordinate_fit
from cost_calibration import source_identity,check_tools,check_inputs
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'guoqiao/optimize_plan/demand_cost_search_repair_artifacts'
BENCH=ROOT/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench'
OPT=ROOT/'build-m6-release/bin/frisk-opt'
FEATURES=tuple(BASE_FEATURES)+('shared_read_bank_excess','shared_write_bank_excess','global_load_issue_groups')
OLD_INDEX=ROOT/'guoqiao/optimize_plan/model_selection_quality_artifacts/evidence-index.json'
OLD_INDEX_SHA='988dd10985b836ebf5a0c7fa204597f8782a6483ab9a057a4fac1ec2b6a501f7'
sha=closure.sha
read=closure.read
write=closure.write
fresh=closure.fresh


def invoke(args,log):
    result=closure.invoke(args,log)
    if result['exit_code']:raise RuntimeError('worker failed; retained at '+str(log))
    return result


def features(record):
    value=record.get('cost',{}).get('features')
    if not isinstance(value,dict) or set(value)-set(FEATURES):raise ValueError('missing or unknown eighteen-feature ledger')
    result={name:value.get(name,0) for name in FEATURES}
    if any(type(v) is not int or v<0 for v in result.values()):raise ValueError('invalid feature count')
    return result


def assert_source_inputs(identity,inputs,tools):
    if source_identity()!=identity:raise ValueError('source changed during registered phase')
    check_inputs(inputs);check_tools(tools)


def native_tools_match_old(tools):
    # Frontend optimization changes intentionally; native compilation/driver
    # and numerical/safety payloads must retain their old exact bytes.
    allowed={str(OPT.resolve()),str(BENCH.resolve())}
    verified={};changed_frontends={}
    for filename,digest in tools['hashes'].items():
        if filename in allowed:changed_frontends[filename]=dict(old=digest,current=sha(filename));continue
        if sha(filename)!=digest:raise ValueError('known-label native tool changed: '+filename)
        verified[filename]=digest
    return dict(verified=verified,frontend_changes_allowed=changed_frontends)


def verify_old(known_run):
    known_run=Path(known_run).resolve()
    if sha(OLD_INDEX)!=OLD_INDEX_SHA:raise ValueError('sealed source index identity changed')
    index=read(OLD_INDEX);contained=(ROOT/index['contained_root']).resolve()
    entries=index['entries'];seen=set()
    for entry in entries:
        rawpath=ROOT/entry['path'];path=rawpath.resolve()
        if rawpath.is_symlink() or any(parent.is_symlink() for parent in rawpath.parents if parent.is_relative_to(contained)) or path in seen or not path.is_relative_to(contained):raise ValueError('invalid sealed evidence path')
        seen.add(path)
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('sealed known evidence changed: '+str(path))
    actual={p.resolve() for p in contained.rglob('*') if p.is_file() and p!=OLD_INDEX}
    if seen!=actual:raise ValueError('sealed known evidence fileset changed')
    completion=read(known_run/'measurement/completion.json');freeze=read(known_run/'measurement/freeze.json')
    if completion['status']!='complete' or completion['run']!=str(known_run) or freeze['run']!=str(known_run):raise ValueError('known measurement incomplete/run mismatch')
    if completion['freeze_sha256']!=sha(known_run/'measurement/freeze.json') or freeze['manifest_sha256']!=sha(known_run/'manifest.json'):raise ValueError('known measurement identity mismatch')
    check_inputs({str(known_run/'protocol.json'):freeze['protocol_sha256']})
    native=native_tools_match_old(freeze['tools'])
    count=sum(len(rows) for rows in freeze['groups'].values())
    if len(freeze['groups'])!=15 or count!=34:raise ValueError('all fifteen groups/thirty-four known images required')
    return freeze,dict(index_sha256=OLD_INDEX_SHA,entries=len(entries),native_tools=native,
        completion_sha256=sha(known_run/'measurement/completion.json'),freeze_sha256=sha(known_run/'measurement/freeze.json'))


def label(known_run,key,row):
    raw=read(known_run/'measurement'/(key+'-raw.json'))
    matches=[x for x in raw if x['arm']==row['arm']]
    if len(matches)!=10 or {x['batch'] for x in matches}!=set(range(10)):raise ValueError('known image lacks ten complete paired blocks')
    batches=[]
    for x in sorted(matches,key=lambda a:a['batch']):
        if x['valid'] is not True or x['cubin_sha256']!=row['cubin_sha256'] or len(x['samples_ms'])!=5:raise ValueError('known image invalid timing')
        if x['launch_count']!=x['actual_graph_nodes'] or not 1<=x['launch_count']<=4096:raise ValueError('known graph count invalid')
        if not np.allclose(x['samples_ms'],np.asarray(x['graph_elapsed_ms'])/x['launch_count'],rtol=1e-12,atol=0):raise ValueError('known event equation mismatch')
        content=x['pair_content'];expected=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if expected!=x['pair_id'] or content['batch']!=x['batch']:raise ValueError('known input/reference pair changed')
        if any(not np.isfinite(v) or v<=0 for v in x['samples_ms']):raise ValueError('invalid known event value')
        batches.append(float(np.median(x['samples_ms']))*1000)
    return dict(median_us=float(np.median(batches)),batch_medians_us=batches,raw_sha256=sha(known_run/'measurement'/(key+'-raw.json')),
        raw_path=str(known_run/'measurement'/(key+'-raw.json')),input_reference_offset_pairs=[x['pair_content'] for x in sorted(matches,key=lambda a:a['batch'])],
        cache='ordinary-cache kernel timings; not cold compilation',no_new_gpu_training=True)


def mark_execution_domain(text,spec):
    if spec['case_id']=='attention':return text
    marker='module attributes {'
    if text.count(marker)!=1:raise ValueError('known source lacks one module attribute block')
    if 'frisk.execution_domain='in text:return text
    return text.replace(marker,marker+'frisk.execution_domain="sm90-scalar-f32-dag-v1",',1)


def bridge_assignment(record,meta):
    result={k:record[k] for k in ('assignment','instruction_bindings','reduction_bindings','conversion_policy','complete_assignment_sha256')}
    if result['conversion_policy']!='independent':raise ValueError('only C labels may bridge')
    result['prepared_digest']=meta['package_digest']
    return result


def collect_training(known_run,out,*,development_preflight=False):
    """Re-score every known image under unit18; never tune/select new holdouts."""
    known_run=Path(known_run).resolve();out=fresh(out);freeze,oldproof=verify_old(known_run)
    unit=dict(schema_version=1,model='sm90-layout-demand-calibrated-v2',coefficients={k:(0 if k=='shared_bank_excess' else 1) for k in FEATURES},
        provenance='known-image raw feature extraction; unit eighteen weights only; fixed aggregate bank coefficient zero')
    write(out/'unit-profile.json',unit)
    identity=source_identity();tools=closure.frozen_tools(OPT);tools['hashes'][str(BENCH.resolve())]=sha(BENCH)
    inputs={str(OLD_INDEX):OLD_INDEX_SHA,str(out/'unit-profile.json'):sha(out/'unit-profile.json'),str(known_run/'measurement/completion.json'):sha(known_run/'measurement/completion.json')}
    registration=dict(stage='known-only-demand-feature-label-bridge',development_preflight=development_preflight,created_ns=time.time_ns(),source=identity,tools=tools,inputs=inputs,
        old=oldproof,images=34,f32_images=31,observed_attention_images=3,independent_programs=9,
        no_holdout_use=True,no_image_cherry_pick=True,no_new_gpu_training=True,
        fit=dict(method='nonnegative ridge NNLS',ridge=1e-6,max_iterations=20000,tolerance=1e-10,ticks_per_microsecond=1000000,
            fixed_coefficients={'shared_bank_excess':0},inactive_coefficient_ticks=1,intercept='nonnegative; diagnostics only; excluded from selection'))
    write(out/'registration.json',registration);rows=[];prepared={}
    write(out/'manifest.json',dict(status='collecting',registration_sha256=sha(out/'registration.json'),rows=[]))
    try:
        for key,oldrows in freeze['groups'].items():
            for oldrow in oldrows:
                source=Path(oldrow['path']);spec=oldrow['spec'];scope=oldrow['scope'];target=out/'images'/(key+'-'+oldrow['arm'])
                prepkey=(scope,spec['key'])
                if prepkey not in prepared:
                    base=fresh(out/'preparation'/key);oldprep=Path(oldrow['prepared']);raw=base/'input.mlir'
                    raw.write_text(mark_execution_domain((oldprep/'input.mlir').read_text(),spec))
                    p=base/'prepared';args=[BENCH,'--prepare',raw,spec['case_id'],p,'--cost-profile='+str(out/'unit-profile.json')]
                    if scope=='natural' and spec['case_id']!='attention':args+=['--natural']
                    invoke(args,base/'prepare.log');meta=read(p/'prepared.json')
                    if (p/'domains.json').read_bytes()!=(oldprep/'domains.json').read_bytes():raise ValueError('known-label candidate domain bytes changed')
                    for field in ('common_digest','domain_digest','coverage_digest'):
                        if meta[field]!=read(oldprep/'prepared.json')[field]:raise ValueError('known-label domain proof changed')
                    prepared[prepkey]=(p,meta)
                p,meta=prepared[prepkey];record=read(source/'record.json');bridge=out/(key+'-'+oldrow['arm']+'-assignment.json')
                write(bridge,bridge_assignment(record,meta))
                args=[BENCH,'--prepared='+str(p),'--case='+spec['case_id'],'--selection=J','--conversion=C','--assignment='+str(bridge),
                    '--cost-profile='+str(out/'unit-profile.json'),'--seed-beam-solution=false','--demand-ledger','--output-dir='+str(target)]
                invoke(args,out/(key+'-'+oldrow['arm']+'.log'));current=read(target/'record.json')
                if current['status']!='success' or current.get('selection_bypassed') is not True or current['complete_assignment_sha256']!=record['complete_assignment_sha256']:
                    raise ValueError('known full assignment replay changed or was searched')
                from backend import emit
                text,config=emit(read(target/'execution.json'))
                if text.encode()!=(source/'kernel.ttgir').read_bytes():raise ValueError('known-label emitted TTGIR bytes changed')
                config_bytes=json.dumps(config,indent=2).encode()
                if config_bytes!=(source/'execution_config.json').read_bytes():raise ValueError('known-label emitted config bytes changed')
                (target/'kernel.ttgir').write_bytes(text.encode());(target/'execution_config.json').write_bytes(config_bytes)
                oldcubin=source/'kernel.cubin'
                if sha(oldcubin)!=oldrow['cubin_sha256']:raise ValueError('old native timing image changed')
                shutil.copyfile(oldcubin,target/'kernel.cubin')
                if sha(target/'kernel.cubin')!=sha(oldcubin):raise ValueError('known image reuse byte mismatch')
                cov=demand_coverage_audit.audit(read(target/'execution.json'),current,require_priced_features=spec['case_id']!='attention')
                write(target/'demand-coverage-audit.json',cov);value=label(known_run,key,oldrow)
                row=dict(group=key,spec=dict(spec,split='train',role='previously-observed-known-training'),source_spec=spec,scope=scope,old_arm=oldrow['arm'],
                    source_path=str(source),path=str(target),source_record_sha256=sha(source/'record.json'),bridge_sha256=sha(bridge),target_prepared_sha256=sha(p/'prepared.json'),
                    complete_assignment_sha256=current['complete_assignment_sha256'],features=features(current),cubin_sha256=sha(oldcubin),
                    ttgir_sha256=sha(target/'kernel.ttgir'),execution_config_sha256=sha(target/'execution_config.json'),coverage=cov['status'],label=value,
                    native_image='exact old accepted bytes reused; no new native compilation',old_image_identity_sha256=sha(source/'image-identity.json'))
                rows.append(row);write(out/'manifest.json',dict(status='collecting',registration_sha256=sha(out/'registration.json'),rows=rows))
        if len(rows)!=34 or sum(x['spec']['case_id']!='attention' for x in rows)!=31:raise ValueError('complete known-image set required')
        if development_preflight:
            check_inputs(inputs);source_unchanged=source_identity()==identity;tools_unchanged=True
            try:check_tools(tools)
            except ValueError:tools_unchanged=False
        else:assert_source_inputs(identity,inputs,tools);source_unchanged=tools_unchanged=True
        write(out/'manifest.json',dict(status='preflight-complete' if development_preflight else 'complete',registration_sha256=sha(out/'registration.json'),rows=rows,
            source_unchanged=source_unchanged,tools_unchanged=tools_unchanged,no_holdout_use=True,development_preflight=development_preflight))
    except Exception as error:
        write(out/'failure.json',dict(status='failed',error=type(error).__name__+': '+str(error),rows_retained=len(rows)));raise
    return out


def fit_training(training,out):
    training=Path(training).resolve();manifest=read(training/'manifest.json');registration=read(training/'registration.json')
    if manifest['status']!='complete' or sha(training/'registration.json')!=manifest['registration_sha256']:raise ValueError('complete frozen training collection required')
    assert_source_inputs(registration['source'],registration['inputs'],registration['tools'])
    rows=manifest['rows']
    if len(rows)!=34 or any(r['spec']['split']!='train' for r in rows):raise ValueError('all known training images required')
    for row in rows:
        d=Path(row['path'])
        if sha(d/'kernel.cubin')!=row['cubin_sha256'] or sha(d/'kernel.ttgir')!=row['ttgir_sha256'] or sha(d/'execution_config.json')!=row['execution_config_sha256']:
            raise ValueError('training emission/image changed')
        if features(read(d/'record.json'))!=row['features']:raise ValueError('training feature ledger changed')
        if sha(row['label']['raw_path'])!=row['label']['raw_sha256']:raise ValueError('known timing label changed')
    x=np.asarray([[row['features'][f] for f in FEATURES] for row in rows],dtype=float);y=np.asarray([r['label']['median_us'] for r in rows])
    fixed=FEATURES.index('shared_bank_excess');x[:,fixed]=0
    fit=coordinate_fit(x,y,**{k:registration['fit'][k] for k in ('ridge','max_iterations','tolerance')})
    if not fit['converged']:raise ValueError('registered training NNLS did not converge')
    scale=registration['fit']['ticks_per_microsecond'];inactive=registration['fit']['inactive_coefficient_ticks']
    weights={name:(int(round(v*scale)) if active else inactive) for name,v,active in zip(FEATURES,fit['coefficients'],fit['active_columns'])};weights['shared_bank_excess']=0
    ledger=[dict(group=r['group'],candidate=r['old_arm'],assignment=r['complete_assignment_sha256'],features=r['features'],median_us=r['label']['median_us'],
        cubin_sha256=r['cubin_sha256'],timing_sha256=r['label']['raw_sha256']) for r in rows]
    digest=hashlib.sha256(json.dumps(ledger,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    out=fresh(out);profile=dict(schema_version=1,model='sm90-layout-demand-calibrated-v2',coefficients=weights,
        provenance='known-only34image NNLS; fixed aggregate bank0; training-ledger-sha256='+digest)
    write(out/'profile.json',profile);write(out/'training-ledger.json',ledger)
    fit.update(features=list(FEATURES),fixed_coefficients={'shared_bank_excess':0},training_ledger_sha256=digest,
        no_holdout_use=True,independent_programs=9,known_image_count=34,selection_ticks_per_microsecond=scale,coefficient_units='selection ticks per execution feature; not architectural latency',
        inactive_features=[name for name,active in zip(FEATURES,fit['active_columns']) if not active and name!='shared_bank_excess'],
        inactive_prior=dict(ticks=inactive,source='preregistered one-tick prior for training-zero feature; not learned/calibrated',holdout_activation='reported without refitting'),
        identifiability='NNLS coefficients not individually identifiable when rank deficient; fixed aggregate bank0 avoids duplicate directional bank column')
    assert_source_inputs(registration['source'],registration['inputs'],registration['tools'])
    write(out/'fit.json',fit);write(out/'inputs.json',dict(training_directory=str(training),training_manifest_sha256=sha(training/'manifest.json'),registration_sha256=sha(training/'registration.json'),
        profile_sha256=sha(out/'profile.json'),fit_sha256=sha(out/'fit.json'),training_ledger_sha256=sha(out/'training-ledger.json'),no_holdout_use=True))
    return out


def model_args(model):
    return ['--cost-profile='+model['profile']] if model.get('profile') else[]


def verify_fitted_training(fit_directory):
    fit_directory=Path(fit_directory).resolve();inputs=read(fit_directory/'inputs.json')
    if inputs.get('no_holdout_use') is not True:raise ValueError('fit is not known-only training')
    for filename,key in [('profile.json','profile_sha256'),('fit.json','fit_sha256'),('training-ledger.json','training_ledger_sha256')]:
        if sha(fit_directory/filename)!=inputs[key]:raise ValueError('frozen fitted training file changed: '+filename)
    training=Path(inputs['training_directory']).resolve()
    for filename,key in [('manifest.json','training_manifest_sha256'),('registration.json','registration_sha256')]:
        if sha(training/filename)!=inputs[key]:raise ValueError('frozen training collection changed: '+filename)
    manifest=read(training/'manifest.json');registration=read(training/'registration.json')
    if manifest.get('status')!='complete' or manifest.get('no_holdout_use') is not True or len(manifest['rows'])!=34:
        raise ValueError('complete all34 known training collection required')
    if manifest['registration_sha256']!=sha(training/'registration.json') or registration.get('no_holdout_use') is not True:
        raise ValueError('training registration mismatch')
    assert_source_inputs(registration['source'],registration['inputs'],registration['tools'])
    return training


def init(run,fit_directory):
    run=fresh(run);fit_directory=Path(fit_directory).resolve();profile=fit_directory/'profile.json'
    training=verify_fitted_training(fit_directory)
    fit=read(fit_directory/'fit.json')
    if fit.get('no_holdout_use') is not True or read(profile)['model']!='sm90-layout-demand-calibrated-v2':raise ValueError('known-only demand fit required')
    # Discover actual semantic IDs on an already-observed development program.
    # No fresh holdout is compiled or timed before preregistration.
    probe=fresh(run/'identity-probe');spec=suite.development_specifications()[0];raw=probe/'input.mlir';raw.write_text(suite.source(spec,ROOT))
    models=[]
    for name,pr in [('legacy',None),('demand',str(profile))]:
        prep=probe/name;descriptor=dict(name=name,profile=pr)
        invoke([BENCH,'--prepare',raw,spec['case_id'],prep,'--natural']+model_args(descriptor),probe/(name+'.log'))
        descriptor['id']=read(prep/'prepared.json')['cost_model'];models.append(descriptor)
    if models[0]['id']!='sm90-layout-lex9-legacy-v2' or not models[1]['id'].startswith('sm90-layout-calibrated-v5d-'):raise ValueError('unexpected frozen semantic model IDs')
    profiles=[dict(path=str(profile),sha256=sha(profile))]
    feature_contract=dict(version='frisk-affine-query-dag-v2',features=list(FEATURES),fixed_coefficients={'shared_bank_excess':0},
        scope='SM90f32pureDAG; splat tensor constants only; common execution_domain; register demand plus bounded cross-thread boundaries',
        query_budget=65536,ledger='per definition and constraint-keyed boundary affine8terms',
        alu_grammar='map/elect/pointer/mux/constantmov and scalar arithmetic query units; not complete PTX or optimized SASS instruction counts',
        load_issue_groups='actual scalar PTX order: four byte loads per f32 request; floating arithmetic, communication, barriers and stores split; addresses, assembly, literal moves and Register aliases do not',
        independent_oracle='actual backend128thread query observer; owner-election footprint; directional scalarSharedbanks',
        fallback='explicit reason and legacy carrier model; no unavailable feature zero substitution')
    search_contract=dict(version='frisk-budgeted-constructive-witness-v1',old='--seed-beam-solution=false',new='--seed-beam-solution=true',
        scope='same repaired binary/domain/fullgraph guards; old is seed-off ablation, not historical binary',
        wall_time='existing native command and solve timings reported separately; equal probe budget is not equal wall-time budget',
        witness_budget='same total65536/refinement256; constructive probes and fullscores charged; no hidden second solve')
    protocol=suite.protocol(models=models,profiles=profiles,feature_contract=feature_contract,search_contract=search_contract)
    protocol['training']=dict(directory=str(fit_directory),profile_sha256=sha(profile),fit_sha256=sha(fit_directory/'fit.json'),
        count=34,independent_programs=9,no_holdout_use=True,fixed_shared_bank_excess_coefficient=0,
        prediction_diagnostic=dict(intercept=float(fit['intercept']),ticks_per_microsecond=fit['selection_ticks_per_microsecond'],
            refit_on_holdout=False,gate=False,image_weighting='mean absolute error across assignment variants within group-image; equal group-image weights'))
    write(run/'protocol.json',protocol)
    immutable={str(run/'protocol.json'):sha(run/'protocol.json')}
    for name in ('profile.json','fit.json','inputs.json','training-ledger.json'):immutable[str(fit_directory/name)]=sha(fit_directory/name)
    for name in ('registration.json','manifest.json'):immutable[str(training/name)]=sha(training/name)
    tools=closure.frozen_tools(OPT);tools['hashes'][str(BENCH.resolve())]=sha(BENCH)
    registration=dict(schema_version=2,created_ns=time.time_ns(),source=source_identity(),tools=tools,optimizer=str(OPT),
        protocol_sha256=sha(run/'protocol.json'),protocol_digest=suite.digest(protocol),immutable_inputs=immutable)
    write(run/'registration.json',registration);write(run/'manifest.json',dict(schema_version=2,status='registered',registration_sha256=sha(run/'registration.json'),rows=[]))
    return run


def verify(run):
    quality.verify_protocol(run);protocol=read(run/'protocol.json')
    if protocol['stage']!='demand-cost-and-budget-search-repair':raise ValueError('wrong experiment protocol')
    return protocol


def check_search(record,treatment):
    if record.get('seed_beam_solution') is not treatment['incumbent_witness']:raise ValueError('requested and actual seed policy differ')
    if record.get('legal_return') is not True:raise ValueError('native selection not independently legal')
    counters=record.get('total')
    if not isinstance(counters,dict) or any(type(counters.get(k)) is not int or counters[k]<0 for k in ('probes','refinement')) or counters['probes']>65536 or counters['refinement']>256:raise ValueError('shared total search budget exceeded')
    contract=record.get('budget_contract');components=record.get('search_components')
    if not isinstance(contract,dict) or contract.get('kind')!='shared-total-search-probes-v1' or contract.get('search_performed') is not True:
        raise ValueError('missing native search budget contract')
    if contract.get('limit')!=65536 or contract.get('refinement_limit')!=256 or contract.get('used')!=counters['probes'] or contract.get('refinement_used')!=counters['refinement']:
        raise ValueError('native shared budget ledger disagrees')
    if not isinstance(components,list):raise ValueError('missing search component evidence')
    seed_work=0
    for component in components:
        for name in ('seed_probes','seed_scores'):
            if type(component.get(name)) is not int or component[name]<0:raise ValueError('invalid seed work counter')
            seed_work+=component[name]
        witness=component.get('has_seed_witness');start=component.get('seed_start_cost')
        if witness:
            selected=component['selected_component_cost']
            if start['saturated'] or selected['saturated'] or selected['cost']>start['cost']:raise ValueError('a better verified incumbent was lost')
        elif start is not None:raise ValueError('missing witness cannot claim seed score')
        if component.get('beam_budget_fallback') and component.get('exhaustive'):raise ValueError('fallback falsely claims exhaustive search')
    if seed_work>8192 or seed_work>counters['probes']:raise ValueError('seed work exceeds shared quota')
    if not treatment['incumbent_witness'] and seed_work:raise ValueError('seed disabled arm performed seed search')
    if treatment['selection']=='LI' and contract.get('producer_first_full_transition_charged') is not True:
        raise ValueError('local-to-full incumbent transition was not budgeted')



def generate(run):
    from execution_support import assess_file
    protocol=verify(run);manifest=read(run/'manifest.json');models=protocol['models']
    if manifest['status']!='registered':raise ValueError('fresh registered run required')
    manifest['status']='generating';write(run/'manifest.json',manifest)
    try:
        for spec in protocol['specifications']:
            for scope in (['controlled','natural'] if spec['case_id']!='attention' else['natural']):
                base=fresh(run/'preparation'/(scope+'-'+spec['key']));raw=base/'input.mlir';raw.write_text(suite.source(spec,ROOT,scope=='controlled'))
                preparations={};common=None
                for model in models:
                    prep=base/('prepared-'+model['name']);args=[BENCH,'--prepare',raw,spec['case_id'],prep]+model_args(model)
                    if scope=='natural' and spec['case_id']!='attention':args+=['--natural']
                    invoke(args,base/('prepare-'+model['name']+'.log'));meta=read(prep/'prepared.json')
                    domain={k:meta[k] for k in ('common_digest','domain_digest','coverage_digest')}
                    if meta['cost_model']!=model['id'] or common is not None and domain!=common:raise ValueError('model/domain identity differs')
                    common=domain;preparations[model['name']]=(model,prep,meta)
                for treatment in suite.arms(spec,scope,models):
                    name=treatment['arm'];model,prep,meta=preparations[treatment['model']];d=run/'cases'/(scope+'-'+spec['key']+'-'+name)
                    args=[BENCH,'--prepared='+str(prep),'--case='+spec['case_id'],'--selection='+treatment['selection'],'--conversion=C',
                        '--seed-beam-solution='+str(treatment['incumbent_witness']).lower(),'--demand-ledger','--output-dir='+str(d)]+model_args(model)
                    if scope=='controlled':args+=['--oracle']
                    if treatment['force_producer']:args+=['--force-producer='+treatment['force_producer']]
                    row=dict(spec=spec,scope=scope,arm=name,model=model['name'],selection=treatment['selection'],treatment=treatment,
                        conversion='C',path=str(d),status='generating',phases=[],prepared=str(prep),prepared_sha256=sha(prep/'prepared.json'),domain=common)
                    manifest['rows'].append(row);write(run/'manifest.json',manifest)
                    outcome=closure.invoke(args,base/(name+'.log'));row['phases'].append(dict(stage='native-selection',command_metadata_sha256=sha(outcome['command_metadata']),**outcome))
                    if outcome['exit_code']:
                        row['status']='failed'
                        if (d/'failure.json').exists():row['failure']=read(d/'failure.json')
                        if outcome['exit_code']==3:row['failure_budget_proof']=validate_failure_sidecars(d)
                        raise RuntimeError('registered arm unavailable; retained in declared coverage')
                    record=read(d/'record.json');check_search(record,treatment)
                    if record['status']!='success' or record['cost']['saturated'] is not False or record['cost_model']!=model['id'] or record['prepared_digest']!=meta['package_digest']:
                        raise ValueError('unknown/saturated/wrong model selection')
                    cov=demand_coverage_audit.audit(read(d/'execution.json'),record,require_priced_features=model['name']=='demand' and spec['case_id']!='attention')
                    if spec['case_id']!='attention' and cov['status']!='passed':raise ValueError('declared f32demand domain failed exact independent coverage')
                    write(d/'demand-coverage-audit.json',cov)
                    support=assess_file(d/'execution.json',include_emission=True);tt=support.pop('ttgir',None);info=support.pop('execution_config',None);write(d/'support.json',support)
                    if not support.get('supported'):raise ValueError('registered backend unsupported: '+str(support))
                    (d/'emitted.ttgir').write_text(tt);write(d/'emitted-config.json',info);write(d/'calibration_spec.json',spec)
                    write(d/'configuration.json',dict(spec=spec,scope=scope,arm=name,model=model['name'],treatment=treatment,entry='frisk_execution',grid=[1,1,1],threads=128))
                    names=('execution.json','record.json','output.mlir','calibration_spec.json','configuration.json','support.json','emitted.ttgir','emitted-config.json','demand-coverage-audit.json')
                    write(d/'generation-inputs.json',{n:sha(d/n) for n in names})
                    row.update(status='generated',generation_inputs_sha256=sha(d/'generation-inputs.json'),assignment_sha256=record['complete_assignment_sha256'],
                        model_id=record['cost_model'],cost=record['cost'],features=features(record) if model['name']=='demand' else None,coverage=cov['status'])
                    write(run/'manifest.json',manifest)
        verify(run)
        if len(manifest['rows'])!=114:raise ValueError('entire114configuration matrix required')
        manifest['status']='generated'
    except Exception as error:
        manifest.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:write(run/'manifest.json',manifest)
    return manifest


def execute(directory,load_only=False,safety=False,safety_tool=None):
    return closure.execute(directory,load_only=load_only,safety=safety,safety_tool=safety_tool,numerical_fixture=suite)


def measurement_groups(run):
    protocol=verify(run);manifest=read(run/'manifest.json')
    if manifest['status']!='validated':raise ValueError('every registered arm must validate')
    expected={(scope,s['key'],a['arm']) for s in protocol['specifications'] for scope in (['controlled','natural'] if s['case_id']!='attention' else['natural']) for a in suite.arms(s,scope,protocol['models'])}
    actual={(r['scope'],r['spec']['key'],r['arm']) for r in manifest['rows']}
    if actual!=expected or len(actual)!=len(manifest['rows']):raise ValueError('registered matrix missing/duplicated')
    specs={s['key']:s for s in protocol['specifications']};groups={};aliases={}
    for row in manifest['rows']:
        d=Path(row['path']);closure.verify_accepted_row(d,row)
        for phase in row['phases']:
            if phase.get('stage')=='native-selection' and sha(phase['command_metadata'])!=phase.get('command_metadata_sha256'):raise ValueError('native command timing evidence changed')
        if row['spec']!=specs[row['spec']['key']] or row['conversion']!='C':raise ValueError('specification substituted')
        config=read(d/'configuration.json');record=read(d/'record.json');check_search(record,row['treatment'])
        if config['treatment']!=row['treatment'] or config['arm']!=row['arm'] or config['scope']!=row['scope']:raise ValueError('policy substituted')
        if record['cost']!=row['cost'] or record['cost_model']!=row['model_id'] or record['complete_assignment_sha256']!=row['assignment_sha256']:raise ValueError('selection record substituted')
        prep=Path(row['prepared']);meta=read(prep/'prepared.json')
        if sha(prep/'prepared.json')!=row['prepared_sha256']:raise ValueError('prepared identity changed')
        for name,key in [('input.mlir','input_sha256'),('domains.json','domain_file_sha256')]:
            if sha(prep/name)!=meta[key]:raise ValueError('prepared package bytes changed')
        c=read(d/'correctness.json');identity=suite.digest(dict(cubin_sha256=row['cubin_sha256'],entry='frisk_execution',execution_config=read(d/'execution_config.json'),physical_abi=c['physical_params']))
        if identity!=row['execution_identity']:raise ValueError('exact execution identity changed')
        groups.setdefault(row['scope']+'-'+row['spec']['key'],[]).append(dict(row,resource_comparison=c['resource_comparison']))
    unique={}
    for key,rows in groups.items():
        images={};aliases[key]={}
        for row in sorted(rows,key=lambda x:x['arm']):
            images.setdefault(row['execution_identity'],row);aliases[key][row['arm']]=dict(arm=images[row['execution_identity']]['arm'],execution_identity=row['execution_identity'])
        unique[key]=list(images.values())
    p=run/'measurement-aliases.json'
    if p.exists() and read(p)!=aliases:raise ValueError('alias identity changed')
    if not p.exists():write(p,aliases)
    return unique


def time_proxy_diagnostics(protocol,rows,analyses):
    model=protocol['models'][1];profile=read(model['profile']);fit=read(Path(protocol['training']['directory'])/'fit.json')
    coefficients=profile['coefficients'];scale=fit['selection_ticks_per_microsecond'];intercept=fit['intercept']
    if set(coefficients)!=set(FEATURES) or not np.isfinite(scale) or scale<=0 or not np.isfinite(intercept) or intercept<0:
        raise ValueError('invalid frozen time proxy')
    images={}
    for row in rows:
        if row['model']!=model['name']:continue
        group=row['scope']+'-'+row['spec']['key'];identity=row['execution_identity'];key=(group,identity)
        observed=float(np.median(analyses[group]['image_batch_medians_us'][identity]))
        feature=row['features']
        if set(feature)!=set(FEATURES) or any(type(v) is not int or v<0 for v in feature.values()):raise ValueError('invalid prediction feature ledger')
        predicted=float(intercept+sum(feature[name]*coefficients[name] for name in FEATURES)/scale)
        if not np.isfinite(predicted) or predicted<0 or not np.isfinite(observed) or observed<=0:raise ValueError('invalid frozen time prediction')
        value=images.setdefault(key,dict(group=group,scope=row['scope'],spec=row['spec'],execution_identity=identity,observed_us=observed,variants={}))
        assignment=row['assignment_sha256'];variant=dict(assignment=assignment,features=feature,predicted_us=predicted,error_us=predicted-observed,relative_error=(predicted-observed)/observed,policies=[])
        if assignment in value['variants']:
            existing=value['variants'][assignment]
            if any(existing[k]!=variant[k] for k in ('features','predicted_us','error_us','relative_error')):raise ValueError('same assignment produced different frozen predictions')
        else:value['variants'][assignment]=variant
        value['variants'][assignment]['policies'].append(row['arm'])
    result=[]
    for value in images.values():
        value['variants']=list(value['variants'].values())
        errors=[v['error_us'] for v in value['variants']]
        value.update(mean_assignment_absolute_error_us=float(np.mean(np.abs(errors))),
            mean_assignment_absolute_relative_error=float(np.mean(np.abs(errors)))/value['observed_us'],
            max_assignment_absolute_error_us=max(abs(x) for x in errors),
            prediction_min_us=min(v['predicted_us'] for v in value['variants']),prediction_max_us=max(v['predicted_us'] for v in value['variants']))
        result.append(value)
    def summary(items):
        return dict(group_unique_images=len(items),assignment_variants=sum(len(x['variants']) for x in items),
            independent_programs=len({x['spec']['key'] for x in items}),
            mae_us=float(np.mean([x['mean_assignment_absolute_error_us'] for x in items])) if items else None,
            mean_absolute_relative_error=float(np.mean([x['mean_assignment_absolute_relative_error'] for x in items])) if items else None,
            max_absolute_error_us=max((x['max_assignment_absolute_error_us'] for x in items),default=None))
    buckets={}
    for split in ('holdout','observed-anchor'):
        selected=[x for x in result if x['spec']['split']==split]
        buckets[split]=dict(all_scopes=summary(selected),by_scope={scope:summary([x for x in selected if x['scope']==scope]) for scope in ('controlled','natural')})
    inactive=fit.get('inactive_features',[])
    unseen={name:dict(prior_ticks=coefficients[name],source='training-zero feature; frozen one-tick prior, not calibrated',
        fresh_assignment_activations=sum(v['features'][name]>0 for x in result if x['spec']['split']=='holdout' for v in x['variants']),
        fresh_max_count=max((v['features'][name] for x in result if x['spec']['split']=='holdout' for v in x['variants']),default=0)) for name in inactive}
    return dict(status='diagnostic-only',model=model['id'],intercept_us=float(intercept),ticks_per_microsecond=scale,
        training_identifiability=dict(design_rank=fit.get('design_rank'),design_columns=fit.get('design_columns'),active_columns=fit.get('active_columns'),
            inactive_prior=fit.get('inactive_prior'),inactive_feature_holdout_activations=unseen),
        refit_on_holdout=False,gate=False,rows=result,summaries=buckets,
        weighting='same-assignment policy aliases collapsed; mean absolute error across assignment variants within group-image; equal group-image weights',
        limitations=['only assignments scored by repaired model; no unscored baseline-image prediction invented',
            'group images and policy aliases are not independent programs; no generalization confidence claim',
            'observed Attention anchors are reported separately from fresh six-program holdout',
            'time proxy includes frozen training intercept for diagnostics; intercept is excluded from selection objective'])


def assess(run):
    from selection_quality_analysis import analyze_group,summarize_suite,policy_ratio,paired_policy_interval
    protocol=verify(run);groups=measurement_groups(run);m=run/'measurement';manifest=read(run/'manifest.json')
    completion=read(m/'completion.json');freeze=read(m/'freeze.json')
    if completion.get('status')!='complete' or completion['freeze_sha256']!=sha(m/'freeze.json'):raise ValueError('measurement incomplete')
    for value in (completion,freeze):
        if value['run']!=str(run.resolve()) or value['protocol_sha256']!=sha(run/'protocol.json') or value['manifest_sha256']!=sha(run/'manifest.json') or value['phase']!='qualification':raise ValueError('measurement registration drift')
    assert_source_inputs(freeze['source'],freeze['input_hashes'],freeze['tools'])
    evidence={str(p.relative_to(m)) for p in m.rglob('*') if p.is_file() and p!=m/'completion.json'}
    if set(completion['evidence_hashes'])!=evidence:raise ValueError('measurement exactfileset changed')
    for name,digest in completion['evidence_hashes'].items():
        if sha(m/name)!=digest:raise ValueError('measurement raw evidence changed')
    aliases=read(run/'measurement-aliases.json');analyses={};oracle=[];names=[x['name'] for x in protocol['models']]
    for key,unique in groups.items():
        rows=[r for r in manifest['rows'] if r['scope']+'-'+r['spec']['key']==key]
        treatment_arms=[a['arm'] for a in suite.arms(rows[0]['spec'],'natural',protocol['models'])]
        choices={name:aliases[key][name]['arm'] for name in treatment_arms};candidates=[]
        for row in unique:
            scores={}
            if row['scope']=='controlled':
                for model in names:
                    matched=[r for r in rows if r['model']==model and r['treatment']['force_producer'] and r['execution_identity']==row['execution_identity']]
                    if not matched:raise ValueError('selected image outside exact forced reference')
                    scores[model]=min(r['cost']['cost'] for r in matched)
            candidates.append(dict(arm=row['arm'],cubin_sha256=row['cubin_sha256'],execution_identity=row['execution_identity'],model_scores=scores))
        result=analyze_group(read(m/(key+'-raw.json')),candidates,choices,expected_batches=10,
            bootstrap_seed=protocol['measurement']['seed'],bootstrap_samples=protocol['measurement']['bootstrap_resamples'])
        result.update(spec=rows[0]['spec'],scope=rows[0]['scope'],reference_domain='exact two producer assignments' if rows[0]['scope']=='controlled' else'policy-selected exact image union; not full candidate optimum')
        analyses[key]=result
        if rows[0]['scope']=='controlled':
            for row in rows:
                if row['treatment']['force_producer']:continue
                record=read(Path(row['path'])/'record.json');full=record['oracle'];ok=full['status']=='complete' and full['exhausted'] is True
                if row['selection']=='J':ok=ok and record['cost']['cost']==full['best_cost']['cost']
                else:
                    first=record['stage1_oracle'];second=record['stage2_oracle']
                    ok=ok and first['status']=='complete' and first['exhausted'] and record['stage1_producer_optimum_tuple'] is True and second['status']=='complete' and second['exhausted'] and record['cost']['cost']==second['best_cost']['cost']
                oracle.append(dict(group=key,arm=row['arm'],passed=bool(ok),record_sha256=sha(Path(row['path'])/'record.json')))
                if not ok:raise ValueError('exact finite oracle failed')
    write(run/'group-quality.json',analyses);write(run/'oracle-quality.json',oracle)
    suites={scope:summarize_suite({k:g for k,g in analyses.items() if g['scope']==scope and g['spec']['split']=='holdout'},
        bootstrap_seed=protocol['measurement']['seed'],bootstrap_samples=protocol['measurement']['bootstrap_resamples']) for scope in ('controlled','natural')}
    gates={}
    for scope,values in suites.items():
        gates[scope]={}
        for policy,x in values['policies'].items():
            certain=x['geomean_regret_ci95'][1]<=1.03 and x['max_case_regret_ci95'][1]<=1.10
            failed=x['geomean_regret_ci95'][0]>1.03 or x['max_case_regret_ci95'][0]>1.10
            gates[scope][policy]=dict(status='qualified' if certain else'failed' if failed else'uncertain',
                observed_threshold_pass=x['geomean_regret_ratio']<=1.03 and x['max_case_regret_ratio']<=1.10)
    baseline=names[0];repaired=names[1];best_new=repaired+'-J-new';base_old=baseline+'-J-old'
    primary=suites['natural']['policy_comparisons'][best_new+'/'+base_old]
    natural={k:g for k,g in analyses.items() if g['scope']=='natural'}
    guards={k:paired_policy_interval(g,best_new,base_old,alpha=.05/len(natural)) for k,g in natural.items()}
    no_regression=all(x['ratio_ci'][1]<=1.05 for x in guards.values())
    bounded=all(gates[scope][best_new]['status']=='qualified' for scope in ('controlled','natural'))
    promoted=primary['ratio_ci95'][1]<1/1.03 and no_regression and bounded
    effects={}
    pairs={'cost_newsearch':(repaired+'-J-new',baseline+'-J-new'),'cost_oldsearch':(repaired+'-J-old',baseline+'-J-old'),
           'search_repaired':(repaired+'-J-new',repaired+'-J-old'),'search_baseline':(baseline+'-J-new',baseline+'-J-old'),
           'joint_repaired':(repaired+'-J-new',repaired+'-LI'),'joint_baseline':(baseline+'-J-new',baseline+'-LI')}
    for name,(a,b) in pairs.items():effects[name]=suites['natural']['policy_comparisons'][a+'/'+b]
    resources=[]
    for row in manifest['rows']:
        d=Path(row['path']);c=read(d/'correctness.json');resources.append(dict(group=row['scope']+'-'+row['spec']['key'],arm=row['arm'],
            physical=c['physical_resources'],confirmation=c['compiler_resource_confirmation'],comparison=c['resource_comparison'],
            spill_count=None,compile_wall_ns=read(d/'compile.json')['wall_ns']))
    write(run/'resource-quality.json',resources)
    prediction=time_proxy_diagnostics(protocol,manifest['rows'],analyses);write(run/'time-proxy-quality.json',prediction)
    search=summarize_search(manifest['rows'],checker=check_search);write(run/'search-quality.json',search)
    summary=dict(schema_version=2,status='complete',protocol_sha256=sha(run/'protocol.json'),source_digest=source_identity()['source_digest'],
        coverage=dict(configurations=114,validated=len(manifest['rows']),measurement_groups=15,group_unique_images=sum(len(x) for x in groups.values()),
            controlled_oracle_checks=len(oracle),exact_demand_configurations=sum(r['coverage']=='passed' for r in manifest['rows'])),
        suites=suites,quality_gates=gates,factorial_effects=effects,time_proxy_accuracy=prediction['summaries'],search_overhead_report='search-quality.json',promotion=dict(primary=primary,guards=guards,no_per_case_regression=no_regression,
            bounded_regret_qualified=bounded,passed=promoted,recommendation='repaired-eligible-for-explicit-default-change' if promoted else'preserve-existing-default',production_default_changed=False),
        limitations=['natural reference is selected six-policy image union','single CTA C supported execution domain','static physical GPR not qualified; compiler/driver feedback used',
            'training is thirty-four already-known exact images; no final holdout fitting','factorial effects separate cost and search; combined gain is not attributed to cost alone'])
    write(run/'quality-summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect-training','preflight-training','fit-training','init','generate','validate','execute','measure','assess','run'])
    p.add_argument('directory',type=Path);p.add_argument('--known-run',type=Path);p.add_argument('--training',type=Path);p.add_argument('--fit-directory',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--load-only',action='store_true');p.add_argument('--safety',action='store_true');p.add_argument('--safety-tool',choices=['memcheck','racecheck','synccheck','initcheck']);a=p.parse_args();d=a.directory.resolve()
    if a.command in ('collect-training','preflight-training'):
        if a.known_run is None:p.error('--known-run required')
        collect_training(a.known_run,d,development_preflight=a.command=='preflight-training')
    elif a.command=='fit-training':
        if a.training is None:p.error('--training required')
        fit_training(a.training,d)
    elif a.command=='init':
        if a.fit_directory is None:p.error('--fit-directory required')
        init(d,a.fit_directory)
    elif a.command=='generate':generate(d)
    elif a.command=='validate':quality.validate(d,worker_script=__file__)
    elif a.command=='execute':
        try:execute(d,a.load_only,a.safety,a.safety_tool)
        except Exception as error:
            write(d/'worker-failure.json',dict(stage=getattr(error,'stage','worker'),code=getattr(error,'code','exception'),reason=type(error).__name__+': '+str(error)));raise
    elif a.command=='measure':quality.measure(d,a.out or d/'measurement',suite=suite,groups=measurement_groups(d))
    elif a.command=='assess':assess(d)
    elif a.command=='run':
        if a.fit_directory is None:p.error('--fit-directory required')
        init(d,a.fit_directory);generate(d);quality.validate(d,worker_script=__file__);quality.measure(d,d/'measurement',suite=suite,groups=measurement_groups(d));assess(d)
if __name__=='__main__':main()
