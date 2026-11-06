"""Task-2 known-program pipeline: fresh preparation/search, exact cached execution.

All optimized providers are explicit and preregistered. No fresh performance
holdout, altered search/native budget or production cache source bypass.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import selection_refinement_validation as base
import selection_refinement_decision as adapter
import selection_retention_archive as archive
import network_selection_fixture as numerical
from cost_calibration import source_identity
from integrity_session import VerificationSession

ROOT = base.engine.ROOT
HOME = ROOT / 'guoqiao/optimize_plan/pipeline_overhead_artifacts'
DATA = Path('/data0/baopeihua-frisk-pipeline-overhead-20261105')
STAGE = 'pipeline-overhead-known-program-qualification'
TASK1_INDEX = ROOT / 'guoqiao/optimize_plan/selection_refinement_artifacts/evidence-index.json'
TASK1_SHA = 'd628e462e21b064d8ecdbb4c944751b954f34e378a6a56c3bbf5f954713be3a5'
CLOCK = {'/solve_timing_ns'} | {root+'/'+leaf for root in ('/candidate_archive_search','/search') for leaf in (
    'certificate_wall_ns','joint_wall_ns','local_wall_ns','total_wall_ns',
    'network_variant/plan_check_wall_ns','joint/solve_timing_ns','local/solve_timing_ns')}
read = adapter.read
write = adapter.write
sha = adapter.sha


@contextmanager
def epoch():
    saved = base.STAGE, adapter.BASE, adapter.DATA
    try:
        base.STAGE, adapter.BASE, adapter.DATA = STAGE, HOME, DATA
        with base.epoch(): yield
    finally:
        base.STAGE, adapter.BASE, adapter.DATA = saved


def semantic(value, path=''):
    if isinstance(value, dict): return {k:semantic(v,path+'/'+k) for k,v in value.items() if path+'/'+k not in CLOCK}
    if isinstance(value, list): return [semantic(v,path+'/'+str(i)) for i,v in enumerate(value)]
    return value


def verify(run):
    with epoch(): return base.engine.verify(Path(run))


def init(run, matrix, *, model_bundle, cache_root, batched=True, local_profile=None):
    run=Path(run).resolve();matrix=Path(matrix).resolve();bundle=Path(model_bundle).resolve();cache=Path(cache_root).resolve()
    if sha(TASK1_INDEX)!=TASK1_SHA: raise ValueError('task1 seal changed')
    value=read(matrix)
    if value.get('schema_version')!=1 or not isinstance(value.get('cases'),list) or not value['cases']:
        raise ValueError('complete known matrix required')
    local_profile=Path(local_profile or ROOT/'guoqiao/optimize_plan/network_selection_execution_artifacts/development/predicate-recalibration-r2/local-profile.json').resolve()
    from compatible_native_profiles import load_profile,dependency_hashes
    load_profile(local_profile)
    seal=adapter.SealedInputs();known=seal.read(Path('/data0/baopeihua-frisk-network-20261105/run-network-selection-r3/manifest.json'))
    known_pairs={(numerical.digest(r['spec']),r['prepared']) for r in known['rows'] if r['model']=='surviving'}
    immutable={str(matrix):sha(matrix),str(TASK1_INDEX):TASK1_SHA,str(bundle):sha(bundle),str(seal.path):adapter.KNOWN_INDEX_SHA256}
    immutable[str(local_profile)]=sha(local_profile)
    immutable.update(dependency_hashes(local_profile))
    cases=[];seen=set();static=adapter.KNOWN_INDEX.parent/'static-network-proxy-profile.json';immutable[str(static)]=seal.check(static)
    for item in value['cases']:
        if set(item)!={'spec','prepared','search','mode'}: raise ValueError('explicit matrix mode/schema required')
        spec=item['spec'];mode=base._case_mode(item);key=(spec['key'],mode)
        if key in seen or (numerical.digest(spec),item['prepared']) not in known_pairs: raise ValueError('unknown/duplicate program')
        seen.add(key);refprep=Path(item['prepared']);refsearch=Path(item['search']);record,pool,_=base._checked_archive(item)
        if record['cost_profile']!=read(static): raise ValueError('common frozen static model required')
        meta=seal.read(refprep/'prepared.json')
        if meta['package_digest']!=record['prepared_digest']: raise ValueError('reference preparation differs')
        for name in ('input.mlir','prepared.json','domains.json'): immutable[str(refprep/name)]=seal.check(refprep/name)
        for name in ('record.json','candidate-archive.json','prepared.mlir','materialized.mlir','output.mlir','reparsed.mlir','execution.json'): immutable[str(refsearch/name)]=sha(refsearch/name)
        cases.append(dict(spec=dict(spec,split='development',role='known-overhead-regression'),original_spec=spec,mode=mode,
            prepared=str(run/'preparation'/spec['key']/'prepared'),search=str(run/'search'/(spec['key']+'-'+mode)),
            reference_prepared=str(refprep),reference_search=str(refsearch),reference_domain={n:meta[n] for n in ('common_digest','domain_digest','coverage_digest')},
            reference_package_digest=meta['package_digest'],expected_requests=len(pool['entries'])))
    for item in read(bundle)['profiles'].values(): immutable[item['path']]=sha(item['path'])
    with epoch(): adapter._fresh_output(run)
    protocol=dict(schema_version=1,stage=STAGE,role='known-development-not-fresh',matrix=cases,
        specifications=[c['spec'] for c in cases],configuration=dict(static=dict(profile=str(static))),
        model_bundle=str(bundle),local_profile=str(local_profile),cache_root=str(cache),safety=dict(tools=['memcheck','racecheck','synccheck','initcheck']),
        native_budget=dict(capacity_per_pool=4,pools=len(cases),requests=sum(c['expected_requests'] for c in cases)),
        search_budget=dict(probes=65536,refinement=256,variant_score_reserve=1),
        generation_execution=_generation_contract(),
        compilation_execution=dict(contract='fixed-ttgir-four-worker-batch-v1',cpus=[8,9,10,11],compiler_input_reuse=True),
        validation_execution=dict(contract='frisk-five-process-validation-batch-v1',numerical_fixture='network_selection_fixture',
            inventory_verifier='pipeline_overhead:verify_inventory',integrity_session=True,safety_cache=True),
        batched_validation=batched,qualification_pass_claim=False)
    write(run/'protocol.json',protocol);tools=base.engine.closure.frozen_tools(base.engine.OPT)
    for path in (base.engine.BENCH,Path('/usr/local/cuda/bin/nvdisasm')): tools['hashes'][str(path.resolve())]=sha(path)
    registration=dict(created_ns=time.time_ns(),source=source_identity(),tools=tools,optimizer=str(base.engine.OPT),
        protocol_sha256=sha(run/'protocol.json'),protocol_digest=numerical.digest(protocol),immutable_inputs=immutable)
    write(run/'registration.json',registration);write(run/'manifest.json',dict(status='registered',registration_sha256=sha(run/'registration.json'),rows=[]))
    return protocol


def reference_audit_args(run, case):
    """Replay diagnostics only from the exact registered reference record."""
    from integrity_session import verified_sha
    path=Path(case['reference_search'])/'record.json'
    expected=read(Path(run)/'registration.json').get('immutable_inputs',{}).get(str(path))
    if not isinstance(expected,str) or len(expected)!=64:
        raise ValueError('reference audit record is not fully registered')
    verified_sha(path,expected)
    search=read(path).get('candidate_archive_search',{})
    if not isinstance(search,dict):raise ValueError('reference archive search diagnostics differ')
    return ['--archive-audit-paid-observations=true'] if 'paid_observation_audit' in search else []


def prepare_search(run):
    run=Path(run);p=verify(run);m=read(run/'manifest.json')
    if m['status']!='registered': raise ValueError('fresh registration required')
    (run/'preparation-logs').mkdir();(run/'search-logs').mkdir();receipts=[];done=set()
    with VerificationSession():
        for case in p['matrix']:
            spec=case['spec'];prep=Path(case['prepared'])
            if spec['key'] not in done:
                raw=prep.parent/'input.mlir';raw.parent.mkdir(parents=True);raw.write_bytes((Path(case['reference_prepared'])/'input.mlir').read_bytes())
                argv=[str(base.engine.BENCH),'--prepare',str(raw),spec['case_id'],str(prep),'--cost-profile='+p['configuration']['static']['profile']]
                if spec['case_id']!='attention':argv.append('--natural')
                receipt=base.engine.demand.invoke(argv,run/'preparation-logs'/(spec['key']+'.log'));receipts.append(dict(stage='prepare',spec=spec['key'],**receipt));done.add(spec['key'])
            meta=read(prep/'prepared.json')
            if {n:meta[n] for n in case['reference_domain']}!=case['reference_domain'] or meta['package_digest']!=case['reference_package_digest']:
                raise ValueError('fresh preparation semantic identity changed')
            args=[str(base.engine.BENCH),'--prepared='+str(prep),'--case='+spec['case_id'],'--selection='+case['mode'],'--conversion=C',
                '--cost-profile='+p['configuration']['static']['profile'],'--seed-beam-solution='+str(case['mode']=='J').lower(),
                '--demand-ledger','--minify-execution-json','--output-dir='+case['search']]+archive.search_args(policy=archive.ROUTING_POLICY)+reference_audit_args(run,case)
            receipt=base.engine.demand.invoke(args,run/'search-logs'/(spec['key']+'-'+case['mode']+'.log'));receipts.append(dict(stage='search',spec=spec['key'],mode=case['mode'],**receipt))
            record,pool,_=base._checked_archive(case);reference=Path(case['reference_search'])
            for name,new in [('record.json',record),('candidate-archive.json',pool)]:
                if semantic(new)!=semantic(read(reference/name)): raise ValueError('registered complete search result changed: '+name)
            for name in ('prepared.mlir','materialized.mlir','output.mlir','reparsed.mlir','execution.json'):
                if (Path(case['search'])/name).read_bytes()!=(reference/name).read_bytes(): raise ValueError('search program bytes changed: '+name)
        verify(run)
    generated={}
    for case in p['matrix']:
        for root in (Path(case['prepared']),Path(case['search'])):
            for file in root.rglob('*'):
                if file.is_file(): generated[str(file)]=sha(file)
    write(run/'preparation-search-inputs.json',generated)
    write(run/'preparation-search-receipts.json',receipts);m['prepared_and_searched']=True
    m['preparation_search_inputs_sha256']=sha(run/'preparation-search-inputs.json');write(run/'manifest.json',m)
    return receipts


def _generation_contract():
    from emission_audit_session import generation_execution_contract
    return generation_execution_contract()


def generate(run):
    run=Path(run)
    if read(run/'manifest.json').get('prepared_and_searched') is not True: raise ValueError('fresh preparation/search required')
    if numerical.digest(read(run/'protocol.json').get('generation_execution'))!=numerical.digest(_generation_contract()): raise ValueError('registered generation execution contract differs')
    from emission_audit_session import enabled
    with epoch(), VerificationSession(), enabled(): return base.generate(run)


def verify_inventory(run,check_artifacts=True,sanitizer=False):
    from integrity_session import verified_sha
    run=Path(run)
    with epoch():
        protocol=base.verify_inventory(run,check_artifacts=check_artifacts,sanitizer=sanitizer)
        manifest=read(run/'manifest.json');receipt=run/'preparation-search-inputs.json'
        if manifest.get('prepared_and_searched') is not True or verified_sha(receipt)!=manifest.get('preparation_search_inputs_sha256'):
            raise ValueError('fresh preparation/search receipt differs')
        for path,digest in read(receipt).items():verified_sha(path,digest)
        for case in protocol['matrix']:
            record,pool,_=base._checked_archive(case);reference=Path(case['reference_search'])
            if semantic(record)!=semantic(read(reference/'record.json')) or semantic(pool)!=semantic(read(reference/'candidate-archive.json')):
                raise ValueError('fresh archive differs from frozen selection')
        return protocol


def compile_batch(run):
    from parallel_compile_batch import compile_many_parallel
    run=Path(run);p=verify_inventory(run);m=read(run/'manifest.json');r=read(run/'registration.json')
    if m['status']!='generated' or m.get('compiled'): raise ValueError('fresh generated matrix required')
    start=time.perf_counter_ns()
    with VerificationSession():
        result=compile_many_parallel([Path(row['path']) for row in m['rows']],Path(p['cache_root'])/'native',
            output=run/'parallel-native-compilation',execution_modules=Path(__file__).parent,cpus=(8,9,10,11),
            compiler_input_reuse=True,source=r['source'],tools=r['tools'])
    count=p['native_budget']['requests']
    if not(result.get('status')=='complete' and result.get('request_cap')==count and result.get('succeeded_requests')==count
        and result.get('failed_requests')==0 and result.get('not_attempted_requests')==0 and result.get('descendants_reaped') is True):
        write(run/'batch-compilation.json',dict(status='failed',result=result));raise ValueError('native request budget/compile failure')
    write(run/'batch-compilation.json',dict(status='complete',result=result,wall_ns=time.perf_counter_ns()-start));m['compiled']=True
    m['batch_compilation_sha256']=sha(run/'batch-compilation.json');write(run/'manifest.json',m);verify_inventory(run);return result


def validate(run):
    from validation_batch import validate as batched
    run=Path(run);p=verify_inventory(run)
    if p['batched_validation']:return batched(run,safety_cache_dir=Path(p['cache_root'])/'safety',output=run/'batched-validation')
    with epoch():return base.engine.quality.validate(run,worker_script=__file__)


def execute(directory,load_only=False,safety=False,safety_tool=None):
    with epoch():return base.execute(directory,load_only,safety,safety_tool)


def analyze(run):
    from cost_evidence_cache import CostEvidenceCache,PositiveProofCache
    import joint_cost_model as model
    from compatible_native_profiles import load_profile
    import decision_priority as priority
    from partial_cost_selection import cost_interval
    run=Path(run);p=verify_inventory(run);m=read(run/'manifest.json')
    if m['status']!='validated' or (run/'cost-decisions.json').exists():raise ValueError('validated fresh analysis required')
    profiles=model.load_bundle(p['model_bundle']);cache=CostEvidenceCache(Path(p['cache_root'])/'cost');out=run/'cost-evidence';out.mkdir()
    local_path=Path(p['local_profile'])
    old=load_profile(local_path);groups={};newrows=[]
    for row in m['rows']:
        d=Path(row['path']);n=cache.native(d)
        if n.get('status')=='available':
            from native_cost_calibration import attach_symbolic_banks
            n=attach_symbolic_banks(n,d)
        native_path=d/'native-analysis.json'
        if native_path.exists() and read(native_path)!=n:raise ValueError('accepted native analysis substituted')
        write(native_path,n);enriched=dict(row,artifact_hashes=dict(row['artifact_hashes'],**{'native-analysis.json':sha(native_path)}))
        ev=model.candidate_evidence(d,enriched);s=cache.summary(d);l=cache.locality(d);a=cache.intervention(d,l)
        write(out/(d.name+'.json'),dict(native=n,evidence=ev,summary=s,locality=l,intervention=a));newrows.append(enriched)
    with PositiveProofCache(Path(p['cache_root'])/'proof') as proof:
        for case in p['matrix']:
            mode=case['mode'];rows=[row for row in newrows if row['spec']==case['spec'] and row['source_archive_mode']==mode]
            data={row['arm']:read(out/(Path(row['path']).name+'.json')) for row in rows};record,pool,_=base._checked_archive(case)
            static_order=[r['arm'] for r in sorted(rows,key=lambda r:(r['cost']['cost'],r['arm']))]
            incumbent=[base._arm(mode,e) for e in pool['entries'] if e['complete_program_sha256']==record['complete_program_sha256']]
            if len(incumbent)!=1:raise ValueError('unique complete incumbent required')
            order=incumbent+[arm for arm in sorted(data) if arm!=incumbent[0]]
            ev={arm:priority.candidate_evidence(next(r['path'] for r in rows if r['arm']==arm),next(r for r in rows if r['arm']==arm)) for arm in order}
            intervals={arm:cost_interval(data[arm]['native'],old) for arm in order};fallback=priority.static_fallback(ev,static_order,enabled=True)
            matrix=priority.affine_matrix(intervals,ev,old);choice=priority.select_affine(matrix,fallback_arm=fallback['payload']['selected_source'],priority=order)
            selected=choice['payload']['selected_source'];kw=dict(summaries={k:v['summary'] for k,v in data.items()},localities={k:v['locality'] for k,v in data.items()},interventions={k:v['intervention'] for k,v in data.items()},proof_provider=proof)
            decision=model.dispatch({k:v['native'] for k,v in data.items()},{k:v['evidence'] for k,v in data.items()},profiles,selected,order,**kw)
            model.verify_dispatch(decision,{k:v['native'] for k,v in data.items()},{k:v['evidence'] for k,v in data.items()},profiles,selected,order,**kw)
            groups[case['spec']['key']+'-'+mode]=dict(fallback=choice,decision=decision,arms=order,static_order=static_order)
        proof_stats=dict(proof.store.stats)
    write(run/'cost-decisions.json',dict(groups=groups,uses_gpu_timing=False,cache_stats=dict(cache.store.stats),proof_stats=proof_stats,rows=newrows));return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['init','search','generate','compile','validate','execute','analyze','verify'])
    parser.add_argument('directory',type=Path);parser.add_argument('--matrix',type=Path);parser.add_argument('--model-bundle',type=Path);parser.add_argument('--cache-root',type=Path);parser.add_argument('--local-profile',type=Path)
    parser.add_argument('--serial-validation',action='store_true');parser.add_argument('--load-only',action='store_true');parser.add_argument('--safety',action='store_true');parser.add_argument('--safety-tool')
    a=parser.parse_args();d=a.directory.resolve()
    if a.command=='init':init(d,a.matrix,model_bundle=a.model_bundle,cache_root=a.cache_root,batched=not a.serial_validation,local_profile=a.local_profile)
    elif a.command=='search':prepare_search(d)
    elif a.command=='generate':generate(d)
    elif a.command=='compile':compile_batch(d)
    elif a.command=='validate':validate(d)
    elif a.command=='execute':execute(d,a.load_only,a.safety,a.safety_tool)
    elif a.command=='analyze':analyze(d)
    else:verify_inventory(d)


if __name__=='__main__':main()
