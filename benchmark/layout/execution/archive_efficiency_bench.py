"""CPU-only full-suite search/compile efficiency comparison.

The old six-source path and one budgeted archive path use identical observed
programs. Cold/warm here means fresh/reused isolated *local* compiler caches;
ordinary Triton API cache-hit internals are unknown. Warm does not rerun search.
"""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from support_closure import sha,read,write,fresh
from cost_calibration import source_identity,check_tools,check_inputs
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts'
PREVIOUS=ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts'
OLD_RUN=PREVIOUS/'run-native-cost-repair-r1-20261103'
SEALED_SHA='66b28196c5065d5cc4504d3a069991b766718ec7d5618163533c120ba33ded9b'
BENCH=ROOT/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench'
SOURCE_ARMS=('legacy-LI','legacy-J-new','v5d-LI','v5d-J-new','surviving-LI','surviving-J-new')
TARGET={'backend':'cuda','arch':90,'warp_size':32}
OPTIONS={'num_warps':4,'enable_fp_fusion':False}
TIMEOUT=1800


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def cache_inventory(path):
    path=Path(path)
    if not path.exists():return dict(exists=False,files=0,bytes=0,manifest_sha256=digest({}))
    if path.is_symlink() or not path.is_dir():raise ValueError('regular cache directory required')
    files={}
    for p in sorted(path.rglob('*')):
        if p.is_symlink():raise ValueError('cache symlinks not allowed')
        if p.is_file():files[str(p.relative_to(path))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return dict(exists=True,files=len(files),bytes=sum(x['bytes'] for x in files.values()),manifest_sha256=digest(files))


def invoke(args,log,*,environment=None,timeout=TIMEOUT):
    """A timeout terminates this tool's dedicated child process group only."""
    log=Path(log);log.parent.mkdir(parents=True,exist_ok=True);start=time.perf_counter_ns();timed=False
    with log.open('wb') as stream:
        process=subprocess.Popen([str(x) for x in args],stdout=stream,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
        try:code=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed=True;os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
    record=dict(command=[str(x) for x in args],exit_code=code,elapsed_ns=time.perf_counter_ns()-start,timeout_seconds=timeout,timeout_expired=timed,child_pid=process.pid,owned_dedicated_process_group=True)
    write(Path(str(log)+'.command.json'),record)
    if code:raise RuntimeError('CPU efficiency command failed; retained at '+str(log))
    return dict(**record,command_metadata=str(log)+'.command.json',command_metadata_sha256=sha(str(log)+'.command.json'))


def assert_cpu_only():
    import torch
    if torch.cuda.is_initialized():raise ValueError('efficiency compilation must never initialize a CUDA context')
    return False


def register(out,previous=OLD_RUN,*,reprepare_current_domains=False,minify_execution=False):
    if type(reprepare_current_domains) is not bool or type(minify_execution) is not bool:raise ValueError('preparation/export modes must be explicit bools')
    from support_closure import frozen_tools,OPT
    from native_compile_cache import compilation_environment
    previous=Path(previous).resolve();index=PREVIOUS/'evidence-index.json'
    if previous!=OLD_RUN.resolve() or sha(index)!=SEALED_SHA:raise ValueError('exact sealed observed source suite required')
    old=read(previous/'manifest.json');protocol=read(previous/'protocol.json');specs={r['spec']['key']:r['spec'] for r in old['rows'] if r['spec']['split']=='holdout'}
    if len(specs)!=6 or any(s['case_id']=='attention' for s in specs.values()):raise ValueError('all six observed f32 programs required')
    items=[];inputs={str(index):SEALED_SHA,str(previous/'manifest.json'):sha(previous/'manifest.json'),str(previous/'protocol.json'):sha(previous/'protocol.json')}
    for key,spec in specs.items():
        rows=[r for r in old['rows'] if r['spec']==spec]
        if len(rows)!=6 or {r['arm'] for r in rows}!=set(SOURCE_ARMS):raise ValueError('observed six-source matrix incomplete')
        prepared={}
        for row in rows:
            prep=Path(row['prepared']);meta=read(prep/'prepared.json');prepared[row['model']]=str(prep)
            for name in ('prepared.json','domains.json','input.mlir'):inputs[str(prep/name)]=sha(prep/name)
            if row['domain']!={k:meta[k] for k in ('common_digest','domain_digest','coverage_digest')}:raise ValueError('sealed preparation differs')
        items.append(dict(spec=spec,prepared=prepared))
    models=protocol['models']
    for m in models:
        if m.get('profile'):inputs[m['profile']]=sha(m['profile'])
    tools=frozen_tools(OPT);tools['hashes'][str(BENCH.resolve())]=sha(BENCH);out=fresh(out)
    environment=compilation_environment();values=environment['environment']
    for name in ('TRITON_REMOTE_CACHE_BACKEND','TRITON_CACHE_MANAGER','TRITON_KERNEL_OVERRIDE'):
        if values.get(name):raise ValueError('uncontrolled remote/custom/override compiler cache configured')
    reg=dict(schema_version=1,stage='all-observed-six-program-CPU-efficiency',source=source_identity(),tools=tools,inputs=inputs,items=items,models=models,
        budgets=dict(old_sources=6,old_per_source_probes=65536,old_per_source_refinement=256,new_sources=1,new_total_probes=65536,new_total_refinement=256,archive_capacity=4,new_max_compile_requests=4,old_compile_requests=6,worker_timeout_seconds=TIMEOUT),
        compiler=dict(target=TARGET,options=OPTIONS,base_environment=environment),
        design=dict(cold='fresh isolated mode-private TRITON_CACHE_DIR, new exact cache and mode-private resource cache; no user cache cleared',warm='same mode compiler/exact/resource caches reused, fresh output directories; generated assignments reused without search',
            modes=['old-six-source','new-shared-archive'],all_six_programs=True,execution_export='minified' if minify_execution else 'pretty',preparation_mode='rebuild-current-same-source' if reprepare_current_domains else 'proof-matched-sealed',cache_internal_hits='unknown; do not infer from compile API call',repetitions='one cold and one warm descriptive observation; no statistical CI',gpu='no context, no GPU execution, explicit compilation target',fairness='same observed source/ABI/precision/domains/compiler; different disclosed search/request budgets are the optimization',prepare=('both modes search freshly prepared current-lowering domains; sealed packages supply only immutable observed input and declared historical delta' if reprepare_current_domains else 'current prepare timing reported separately and proof-matched to sealed packages; old search reads sealed packages')))
    write(out/'registration.json',reg);write(out/'manifest.json',dict(status='registered',registration_sha256=sha(out/'registration.json'),groups=[]));return out


def verify_compiler_environment(reg,expected_cache,current=None):
    from native_compile_cache import compilation_environment
    current=compilation_environment() if current is None else current
    current=json.loads(json.dumps(current));expected=reg['compiler']['base_environment']
    if current['environment'].get('TRITON_CACHE_DIR')!=expected_cache:raise ValueError('mode-private compiler cache environment differs')
    current['environment']['TRITON_CACHE_DIR']=expected['environment'].get('TRITON_CACHE_DIR')
    # Only this declared cache path override is allowed, including a provider
    # that happens to list the cache variable among invalidating settings.
    if 'TRITON_CACHE_DIR' in current.get('triton_cache_invalidating',{}):
        if 'TRITON_CACHE_DIR' not in expected.get('triton_cache_invalidating',{}):raise ValueError('compiler environment schema changed')
        current['triton_cache_invalidating']['TRITON_CACHE_DIR']=expected['triton_cache_invalidating']['TRITON_CACHE_DIR']
    if current!=expected:raise ValueError('registered compiler environment/knobs changed')
    return True


def verify(out):
    out=Path(out);reg=read(out/'registration.json');m=read(out/'manifest.json')
    if sha(out/'registration.json')!=m['registration_sha256'] or source_identity()!=reg['source']:raise ValueError('registered source changed')
    check_tools(reg['tools']);check_inputs(reg['inputs']);verify_compiler_environment(reg,reg['compiler']['base_environment']['environment'].get('TRITON_CACHE_DIR'));return reg,m


def source_command(prepared,spec,model,selection,directory,*,archive=False,minify_execution=False):
    args=[BENCH,'--prepared='+str(prepared),'--case='+spec['case_id'],'--selection='+selection,'--conversion=C','--seed-beam-solution='+str(selection=='J').lower(),'--demand-ledger','--output-dir='+str(directory)]
    if model.get('profile'):args+=['--cost-profile='+model['profile']]
    if archive:args+=['--candidate-archive=4']
    if minify_execution:args+=['--minify-execution-json']
    return args


def checked_generation(directory):
    from backend import emit
    d=Path(directory);record=read(d/'record.json')
    if record['status']!='success' or record['cost']['saturated']:raise ValueError('unavailable efficiency source candidate')
    text,config=emit(read(d/'execution.json'));(d/'emitted.ttgir').write_text(text);write(d/'emitted-config.json',config)
    return dict(path=str(d),assignment=record['complete_assignment_sha256'],generation_hashes={n:sha(d/n) for n in ('execution.json','record.json','emitted.ttgir','emitted-config.json')})


def bind_prepared(candidate,prepared):
    prepared=Path(prepared);meta=read(prepared/'prepared.json');record=read(Path(candidate['path'])/'record.json')
    if record.get('prepared_digest')!=meta['package_digest'] or record.get('cost_model')!=meta['cost_model']:
        raise ValueError('candidate does not use its current prepared package/model')
    candidate['current_prepared']=dict(path=str(prepared),package_digest=meta['package_digest'],cost_model=meta['cost_model'],
        hashes={name:sha(prepared/name) for name in ('prepared.json','domains.json','input.mlir')})
    return candidate


def generate(out,runner=invoke):
    import demand_repair
    from partial_cost_qualification import check_archive
    out=Path(out);reg,m=verify(out)
    if m['status']!='registered':raise ValueError('fresh registered CPU matrix required')
    models={r['name']:r for r in reg['models']};start=time.perf_counter_ns();m['status']='generating';write(out/'manifest.json',m)
    try:
        for item in reg['items']:
            spec=item['spec'];key=spec['key'];group=dict(spec=spec,modes={});m['groups'].append(group)
            prep_mode=reg.get('design',{}).get('preparation_mode','proof-matched-sealed')
            if prep_mode not in ('proof-matched-sealed','rebuild-current-same-source'):raise ValueError('unknown preparation contract')
            export=reg.get('design',{}).get('execution_export','pretty')
            if export not in ('pretty','minified'):raise ValueError('unknown execution export contract')
            rebuild=prep_mode=='rebuild-current-same-source';minify=export=='minified'
            common_current=None
            for mode in ('old','new'):
                mode_start=time.perf_counter_ns()
                sources=[];timers=[];prep_times=[];replay_times=[];search_times=[];current_prepared={};domain_bridges=[]
                for name in (('legacy','v5d','surviving') if mode=='old' else ('surviving',)):
                    oldprep=Path(item['prepared'][name]);meta=read(oldprep/'prepared.json');prep=out/'preparation'/mode/key/name
                    args=[BENCH,'--prepare',oldprep/'input.mlir',spec['case_id'],prep,'--natural']
                    if models[name].get('profile'):args+=['--cost-profile='+models[name]['profile']]
                    timer=runner(args,out/'logs'/(mode+'-'+key+'-'+name+'-prepare.log'));current=read(prep/'prepared.json')
                    fields=('common_digest','domain_digest','coverage_digest')
                    if rebuild:
                        if sha(prep/'input.mlir')!=sha(oldprep/'input.mlir'):raise ValueError('current preparation changed the observed algorithm input')
                        domain={k:current[k] for k in fields}
                        if common_current is not None and domain!=common_current:raise ValueError('current old/new/model preparation domains differ')
                        common_current=domain
                        domain_bridges.append(dict(model=name,historical_package=str(oldprep),current_package=str(prep),historical={k:meta[k] for k in fields},current=domain,
                            input_sha256=sha(prep/'input.mlir'),historical_package_sha256=sha(oldprep/'prepared.json'),current_package_sha256=sha(prep/'prepared.json'),
                            rule='same immutable algorithm input; both paths use current-lowering equal domains; old cost-work proofs are never replayed after lowering changes'))
                    elif any(current[k]!=meta[k] for k in fields):raise ValueError('new prepare changed observed domain')
                    current_prepared[name]=prep
                    prep_times.append(timer);timers.append(timer)
                if mode=='old':
                    for name in ('legacy','v5d','surviving'):
                        for selection in ('LI','J'):
                            arm=name+'-'+('LI' if selection=='LI' else'J-new');d=out/'generation'/mode/key/arm
                            t=runner(source_command(current_prepared[name] if rebuild else item['prepared'][name],spec,models[name],selection,d,minify_execution=minify),out/'logs'/(mode+'-'+key+'-'+arm+'.log'));search_times.append(t);timers.append(t)
                            record=read(d/'record.json');demand_repair.check_search(record,dict(selection=selection,incumbent_witness=selection=='J'))
                            candidate=checked_generation(d)
                            if rebuild:bind_prepared(candidate,current_prepared[name])
                            candidate.update(arm=arm,selection_probes=record['total']['probes'],refinement=record['total']['refinement']);sources.append(candidate)
                    if len(sources)!=6:raise ValueError('old six-source path incomplete')
                else:
                    prep=out/'preparation'/mode/key/'surviving';search=out/'search'/key
                    t=runner(source_command(prep,spec,models['surviving'],'J',search,archive=True,minify_execution=minify),out/'logs'/('new-'+key+'-archive.log'));search_times.append(t);timers.append(t)
                    record=read(search/'record.json');archive=read(search/'candidate-archive.json');check_archive(record,archive,'joint',4)
                    for entry in archive['entries']:
                        d=out/'generation'/mode/key/('candidate-'+str(entry['index']));assignment=out/'search'/(key+'-assignment-'+str(entry['index'])+'.json');write(assignment,entry)
                        args=[a for a in source_command(prep,spec,models['surviving'],'J',d,minify_execution=minify) if not str(a).startswith('--seed-beam-solution=')]+['--assignment='+str(assignment),'--seed-beam-solution=false']
                        t=runner(args,out/'logs'/('new-'+key+'-replay-'+str(entry['index'])+'.log'));replay_times.append(t);timers.append(t)
                        r=read(d/'record.json')
                        if r.get('selection_bypassed') is not True or r['budget_contract']['used']!=0 or r['budget_contract']['refinement_used']!=0 or r['complete_assignment_sha256']!=entry['complete_assignment_sha256']:raise ValueError('archive replay searched or changed identity')
                        candidate=checked_generation(d)
                        if rebuild:bind_prepared(candidate,prep)
                        candidate.update(arm='candidate-'+str(entry['index']),selection_probes=0,refinement=0);sources.append(candidate)
                    if not 1<=len(sources)<=4:raise ValueError('new archive compile cap exceeded')
                total_probes=sum(c['selection_probes'] for c in sources) if mode=='old' else record['candidate_archive_search']['total']['probes']
                refinement=sum(c['refinement'] for c in sources) if mode=='old' else record['candidate_archive_search']['total']['refinement']
                if total_probes>reg['budgets']['old_per_source_probes']*(6 if mode=='old' else 1) or refinement>256*(6 if mode=='old' else 1):raise ValueError('search shared total budget exceeded')
                group['modes'][mode]=dict(sources=sources,prepare=prep_times,search=search_times,replay=replay_times,preparation_domain_bridges=domain_bridges,probes=total_probes,refinement=refinement,
                    prepare_wall_ns=sum(t['elapsed_ns'] for t in prep_times),selection_wall_ns=sum(t['elapsed_ns'] for t in search_times),replay_wall_ns=sum(t['elapsed_ns'] for t in replay_times),
                    cli_wall_ns=sum(t['elapsed_ns'] for t in timers),source_solves=len(search_times),archive_capacity=4 if mode=='new' else None)
                write(out/'manifest.json',m)
                # Include this mode's emission preflight and manifest write.
                # The final receipt write belongs to shared outer accounting.
                group['modes'][mode]['generation_work_wall_ns']=time.perf_counter_ns()-mode_start
            print(json.dumps(dict(generated_observed_program=key)),flush=True)
        m.update(status='generated',generation_wall_ns=time.perf_counter_ns()-start);write(out/'manifest.json',m);verify(out)
        measured=sum(v['generation_work_wall_ns'] for g in m['groups'] for v in g['modes'].values())
        whole=time.perf_counter_ns()-start
        write(out/'generation-timing.json',dict(status='complete',whole_phase_work_wall_ns=whole,summed_mode_work_wall_ns=measured,shared_outer_wall_ns=whole-measured,
            scope='mode includes prepare/selection/replay/emission preflight/one own manifest write; shared outer includes final manifest and source/tool verification; excludes final timing receipt write and parent Python startup'))
    except Exception as error:m.update(status='failed',error=type(error).__name__+': '+str(error));write(out/'manifest.json',m);raise
    return m


def check_generated_matrix(reg,manifest):
    expected={x['spec']['key']:x['spec'] for x in reg['items']};groups=manifest.get('groups',[])
    if len(expected)!=6 or len(groups)!=6 or {g['spec']['key'] for g in groups}!=set(expected):raise ValueError('all six registered observed programs required')
    for g in groups:
        if g['spec']!=expected[g['spec']['key']] or set(g['modes'])!={'old','new'}:raise ValueError('observed program or mode substituted')
        for mode,v in g['modes'].items():
            count=len(v['sources'])
            if type(v.get('generation_work_wall_ns')) is not int or v['generation_work_wall_ns']<0:raise ValueError('mode generation wall ledger missing')
            if mode=='old' and (count!=6 or {r['arm'] for r in v['sources']}!=set(SOURCE_ARMS)):raise ValueError('old six sources incomplete')
            if mode=='new' and (not 1<=count<=4 or len({r['assignment'] for r in v['sources']})!=count):raise ValueError('new archive cap or exact assignment uniqueness differs')
            if len(v['prepare'])!=(3 if mode=='old' else 1) or len(v['search'])!=(6 if mode=='old' else 1) or len(v['replay'])!=(0 if mode=='old' else count):raise ValueError('prepare/search/replay work ledger incomplete')
            if v['source_solves']!=(6 if mode=='old' else 1) or type(v['probes']) is not int or not 0<=v['probes']<=65536*(6 if mode=='old' else 1) or type(v['refinement']) is not int or not 0<=v['refinement']<=256*(6 if mode=='old' else 1):raise ValueError('source or shared search budget differs')
            if reg.get('design',{}).get('preparation_mode')=='rebuild-current-same-source':
                bridges=v.get('preparation_domain_bridges',[])
                expected_models={'legacy','v5d','surviving'} if mode=='old' else {'surviving'}
                if {b['model'] for b in bridges}!=expected_models or len(bridges)!=len(expected_models):raise ValueError('current preparation domain bridge omitted')
                expected_paths={b['current_package'] for b in bridges}
                domains=[]
                for b in bridges:
                    meta=read(Path(b['current_package'])/'prepared.json')
                    if sha(Path(b['current_package'])/'prepared.json')!=b['current_package_sha256'] or sha(Path(b['current_package'])/'input.mlir')!=b['input_sha256']:
                        raise ValueError('current prepared domain bridge changed')
                    if sha(Path(b['historical_package'])/'prepared.json')!=b['historical_package_sha256']:raise ValueError('historical preparation changed')
                    if {k:meta[k] for k in ('common_digest','domain_digest','coverage_digest')}!=b['current']:raise ValueError('current domain identity changed')
                    domains.append(b['current'])
                for source in v['sources']:
                    bound=source.get('current_prepared',{})
                    if bound.get('path') not in expected_paths or set(bound.get('hashes',{}))!={'prepared.json','domains.json','input.mlir'}:raise ValueError('candidate current preparation binding missing')
                    for name,h in bound['hashes'].items():
                        if sha(Path(bound['path'])/name)!=h:raise ValueError('candidate prepared payload changed')
                    record=read(Path(source['path'])/'record.json')
                    if record.get('prepared_digest')!=bound.get('package_digest') or record.get('cost_model')!=bound.get('cost_model'):raise ValueError('candidate/current preparation mismatch')
                if any(x!=domains[0] for x in domains):raise ValueError('current source model domains differ')
            for stage in ('prepare','search','replay'):
                for timer in v[stage]:
                    if sha(timer['command_metadata'])!=timer['command_metadata_sha256']:raise ValueError('efficiency command timer changed')
        if reg.get('design',{}).get('preparation_mode')=='rebuild-current-same-source':
            domains=[b['current'] for v in g['modes'].values() for b in v['preparation_domain_bridges']]
            if any(x!=domains[0] for x in domains):raise ValueError('current old/new preparation domains differ')
    return True


def compile_inputs(out,temperature,mode,groups):
    if temperature not in ('cold','warm') or mode not in ('old','new'):raise ValueError('explicit cold/warm and old/new required')
    directories=[]
    for group in groups:
        for source in group['modes'][mode]['sources']:
            old=Path(source['path'])
            for name,h in source['generation_hashes'].items():
                if sha(old/name)!=h:raise ValueError('efficiency assignment input changed')
            d=fresh(Path(out)/mode/'cases'/(temperature+'-'+group['spec']['key']+'-'+source['arm']))
            for name in ('execution.json','emitted.ttgir','emitted-config.json'):shutil.copyfile(old/name,d/name)
            directories.append(dict(path=str(d),spec=group['spec'],arm=source['arm'],assignment=source['assignment'],emission_sha256=sha(d/'emitted.ttgir'),config_sha256=sha(d/'emitted-config.json'),input_hashes={n:sha(d/n) for n in ('execution.json','emitted.ttgir','emitted-config.json')}))
    return directories


def old_worker(directory):
    from cost_calibration import compile_exact
    directory=Path(directory);context=read(directory/'efficiency-context.json');registration=read(context['registration'])
    verify_compiler_environment(registration,context['triton_cache']);assert_cpu_only();start=time.perf_counter_ns();compile_exact(directory);assert_cpu_only();verify_compiler_environment(registration,context['triton_cache'])
    write(Path(directory)/'efficiency-worker.json',dict(kind='old-independent-worker',wall_ns=time.perf_counter_ns()-start,cuda_context_initialized=False,compiler_api_calls=1,compiler_internal_cache_hit=None,target=TARGET,options=OPTIONS))


def new_worker(configuration):
    from native_compile_cache import CompilerSession
    cfg=read(configuration);reg=read(cfg['registration']);verify_compiler_environment(reg,cfg['triton_cache']);assert_cpu_only();start=time.perf_counter_ns()
    with CompilerSession(cfg['cache'],source=reg['source'],tools=reg['tools']) as session:
        for row in cfg['directories']:session.compile(Path(row['path']));assert_cpu_only()
    assert_cpu_only();verify_compiler_environment(reg,cfg['triton_cache'])
    write(cfg['result'],dict(status='complete',wall_ns=time.perf_counter_ns()-start,cuda_context_initialized=False,session=session.summary()))


def compile_phase(out,temperature,runner=invoke):
    out=Path(out);reg,m=verify(out)
    if temperature not in ('cold','warm'):raise ValueError('cold or warm phase required')
    if m['status']!='generated':raise ValueError('complete generated suite required')
    check_generated_matrix(reg,m)
    resultfile=out/(temperature+'-compilation.json')
    if resultfile.exists():raise ValueError('fresh compile output required')
    if temperature=='warm' and not (out/'cold-compilation.json').exists():raise ValueError('warm requires completed own cold pass')
    results={};stage_start=time.perf_counter_ns()
    for mode in ('old','new'):
        triton_cache=out/'cache'/mode/'triton';exact=out/'cache'/mode/'exact';resource_cache=out/mode/'resource-feedback-cache' if mode=='old' else exact/'resource-feedback-cache';before={name:cache_inventory(path) for name,path in [('triton',triton_cache),('exact',exact),('resources',resource_cache)]}
        if temperature=='cold' and any(x['exists'] for x in before.values()):raise ValueError('cold compiler caches must not exist')
        triton_cache.mkdir(parents=True,exist_ok=True);env=os.environ.copy();env['TRITON_CACHE_DIR']=str(triton_cache);start=time.perf_counter_ns();directories=compile_inputs(out,temperature,mode,m['groups']);input_copy_ns=time.perf_counter_ns()-start;commands=[]
        if mode=='old':
            for row in directories:
                context=Path(row['path'])/'efficiency-context.json';write(context,dict(registration=str(out/'registration.json'),triton_cache=str(triton_cache)));row['input_hashes']['efficiency-context.json']=sha(context)
                commands.append(runner([sys.executable,__file__,'old-worker',row['path']],out/'logs'/(temperature+'-old-'+Path(row['path']).name+'.log'),environment=env,timeout=TIMEOUT))
            stats=dict(requests=len(directories),compiler_api_calls=len(directories),exact_cache_hits=0,compiler_internal_cache_hit=None,workers_started=len(directories))
        else:
            config=out/(temperature+'-new-worker.json');result=out/(temperature+'-new-worker-result.json');write(config,dict(registration=str(out/'registration.json'),directories=directories,cache=str(exact),triton_cache=str(triton_cache),result=str(result)))
            commands.append(runner([sys.executable,__file__,'new-worker',config],out/'logs'/(temperature+'-new-batch.log'),environment=env,timeout=TIMEOUT*6));worker=read(result);stats=dict(worker['session'],workers_started=1)
        work_wall_ns=time.perf_counter_ns()-start
        actual=[]
        for row in directories:
            d=Path(row['path']);info=read(d/'compile.json')
            for n,h in row['input_hashes'].items():
                if sha(d/n)!=h:raise ValueError('compile worker changed execution/generation inputs')
            if not (d/'kernel.cubin').read_bytes().startswith(b'\x7fELF'):raise ValueError('compiled native image missing ELF header')
            native_files=('kernel.cubin','kernel.ptx','kernel.ttgir','execution_config.json','compile.json')
            actual.append(dict(**row,cubin_sha256=sha(d/'kernel.cubin'),compile_wall_ns=info['wall_ns'],ptx_bytes=(d/'kernel.ptx').stat().st_size,cuda_context_initialized=False,
                compiler_api_called=info['compile_cache']['compiler_api_called'] if mode=='new' else True,
                exact_cache_hit=info['compile_cache']['exact_cache_hit'] if mode=='new' else False,
                artifact_hashes={n:sha(d/n) for n in native_files}))
        if len(actual)!=stats['requests'] or sum(r['compiler_api_called'] for r in actual)!=stats['compiler_api_calls'] or sum(r['exact_cache_hit'] for r in actual)!=stats['exact_cache_hits']:raise ValueError('compile request/API/exact-hit counters disagree')
        after={name:cache_inventory(path) for name,path in [('triton',triton_cache),('exact',exact),('resources',resource_cache)]}
        results[mode]=dict(wall_ns=work_wall_ns,post_work_audit_wall_ns=time.perf_counter_ns()-start-work_wall_ns,input_copy_wall_ns=input_copy_ns,directories=actual,stats=stats,commands=commands,cache_before=before,cache_after=after,environment_overrides={'TRITON_CACHE_DIR':str(triton_cache)},cuda_context_initialized=False)
        write(out/(temperature+'-'+mode+'-compilation.json'),results[mode]);print(json.dumps(dict(compiled=mode,temperature=temperature,requests=len(directories))),flush=True)
    write(resultfile,dict(status='complete',temperature=temperature,total_wall_ns=time.perf_counter_ns()-stage_start,modes=results,warm_search_rerun=False,cold_search_separate=True));verify(out);return results


def summarize(out):
    out=Path(out);reg,m=verify(out);check_generated_matrix(reg,m);cold=read(out/'cold-compilation.json');warm=read(out/'warm-compilation.json')
    if cold['status']!=warm['status'] or cold['status']!='complete':raise ValueError('both complete cold and warm passes required')
    for phase in (cold,warm):
        for mode,value in phase['modes'].items():
            for command in value['commands']:
                if sha(command['command_metadata'])!=command['command_metadata_sha256']:raise ValueError('compile command wall receipt changed')
            for row in value['directories']:
                for name,h in {**row['input_hashes'],**row['artifact_hashes']}.items():
                    if sha(Path(row['path'])/name)!=h:raise ValueError('compiled efficiency artifact changed')
    data={};search={mode:dict(generation_work_wall_ns=sum(g['modes'][mode]['generation_work_wall_ns'] for g in m['groups']),prepare_wall_ns=sum(g['modes'][mode]['prepare_wall_ns'] for g in m['groups']),selection_wall_ns=sum(g['modes'][mode]['selection_wall_ns'] for g in m['groups']),replay_wall_ns=sum(g['modes'][mode]['replay_wall_ns'] for g in m['groups']),cli_wall_ns=sum(g['modes'][mode]['cli_wall_ns'] for g in m['groups']),probes=sum(g['modes'][mode]['probes'] for g in m['groups']),refinement=sum(g['modes'][mode]['refinement'] for g in m['groups']),source_solves=sum(g['modes'][mode]['source_solves'] for g in m['groups'])) for mode in ('old','new')}
    for name,phase in [('cold',cold),('warm',warm)]:
        old=phase['modes']['old'];new=phase['modes']['new'];data[name]=dict(old_compile_wall_ns=old['wall_ns'],new_compile_wall_ns=new['wall_ns'],new_over_old_compile_ratio=new['wall_ns']/old['wall_ns'],old_stats=old['stats'],new_stats=new['stats'],new_over_old_cold_pipeline_ratio=(search['new']['generation_work_wall_ns']+new['wall_ns'])/(search['old']['generation_work_wall_ns']+old['wall_ns']) if name=='cold' else None,new_over_old_core_cli_compile_ratio=(search['new']['cli_wall_ns']+new['wall_ns'])/(search['old']['cli_wall_ns']+old['wall_ns']) if name=='cold' else None,warm_compile_only=name=='warm')
    by_program={}
    for group in m['groups']:
        key=group['spec']['key'];values={}
        for phase_name,phase in [('cold',cold),('warm',warm)]:
            values[phase_name]={}
            for mode in ('old','new'):
                selected=[row for row in phase['modes'][mode]['directories'] if row['spec']==group['spec']]
                values[phase_name][mode]=dict(requests=len(selected),compiler_api_calls=sum(row['compiler_api_called'] for row in selected),exact_cache_hits=sum(row['exact_cache_hit'] for row in selected),summed_per_request_compile_wall_ns=sum(row['compile_wall_ns'] for row in selected))
        by_program[key]=dict(spec=group['spec'],search={mode:{name:group['modes'][mode][name] for name in ('generation_work_wall_ns','prepare_wall_ns','selection_wall_ns','replay_wall_ns','probes','refinement','source_solves')} for mode in ('old','new')},compilation=values,
            timing_scope='per-request compile sums exclude process/session initialization and shared batch accounting; use aggregate phase wall for complete compilation-path comparison')
    result=dict(status='complete',by_program=by_program,shared_generation_timing=read(out/'generation-timing.json'),pipeline_scope='sum of measured per-mode generation work plus complete mode compilation work; shared outer generation and compilation audits separate; does not include registration or parent CLI startup',programs=len(m['groups']),descriptive_repetitions=1,no_statistical_confidence_interval=True,search=search,compilation=data,scope='all six previously observed programs; no GPU context/kernel execution; performance holdouts not read; cache-local cold/warm, warm search not rerun',registration_sha256=sha(out/'registration.json'))
    write(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','generate','compile-cold','compile-warm','summarize','old-worker','new-worker']);p.add_argument('directory',type=Path);p.add_argument('--reprepare-current-domains',action='store_true');p.add_argument('--minify-execution-json',action='store_true');a=p.parse_args()
    if (a.reprepare_current_domains or a.minify_execution_json) and a.action!='register':raise ValueError('preparation mode is fixed at registration only')
    if a.action=='register':register(a.directory,reprepare_current_domains=a.reprepare_current_domains,minify_execution=a.minify_execution_json)
    elif a.action=='generate':generate(a.directory)
    elif a.action=='compile-cold':compile_phase(a.directory,'cold')
    elif a.action=='compile-warm':compile_phase(a.directory,'warm')
    elif a.action=='summarize':print(json.dumps(summarize(a.directory)))
    elif a.action=='old-worker':old_worker(a.directory)
    else:new_worker(a.directory)
if __name__=='__main__':main()
