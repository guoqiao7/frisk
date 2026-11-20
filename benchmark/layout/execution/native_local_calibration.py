"""LOCAL-aware native calibration, preserving the original native scorer ABI.

All old performance observations are training/development data. New LOCAL
protocols use direct PTX and actual retained LDL/STL, not allocated LOCAL bytes
as a traffic proxy. No subprocess or CUDA work occurs on import.
"""
import argparse,ctypes,hashlib,itertools,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
import native_cost_calibration as base
from native_cost_analysis import analyze
from support_closure import sha,read,write,fresh
ROOT=base.ROOT
BASE=ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts'
PREVIOUS=ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts'
SEED=20261104
STEP=0x9e3779b9
SOURCE_SHA=sha(__file__)
OLD_RUNS=base.OLD_RUNS+(PREVIOUS/'run-native-cost-repair-r1-20261103',)
FEATURES=base.MODEL_FEATURES


def specifications():
    rows=[dict(key='baseline',kind='control',words=0,repeats=0)]
    for repeats in (8,32,128,512,1024,2048):rows.append(dict(key=f'control-{repeats}',kind='control',words=0,repeats=repeats))
    for words in (256,1024,4096):
        for repeats in ((8,32,128) if words==256 else (8,32,128,512,1024,2048)):
            rows.append(dict(key=f'local-w{words}-n{repeats}',kind='local',words=words,repeats=repeats))
    return rows


def ptx(spec):
    if spec not in specifications():raise ValueError('unregistered LOCAL protocol')
    words=spec['words'] if spec['kind']=='local' else 1
    head=['.version 8.0','.target sm_90','.address_size 64',
      '.visible .entry frisk_execution(.param .u64 Q,.param .u64 K,.param .u64 V,.param .u64 O,.param .u64 S) {',
      '.reg .b32 %t,%x,%y,%j,%a,%b;','.reg .b64 %q,%o,%i,%p,%z;',
      f'.local .align 4 .b8 scratch[{4*words}];',
      'ld.param.u64 %q,[Q];','ld.param.u64 %o,[O];','mov.u32 %t,%tid.x;',
      'mul.wide.u32 %i,%t,4;','add.u64 %p,%q,%i;','add.u64 %z,%o,%i;',
      'ld.global.u32 %x,[%p];','mov.u32 %b,scratch;']
    body=['bar.sync 0;',f'add.u32 %x,%x,{STEP};']
    if spec['kind']=='local':body=[f'and.b32 %j,%x,{words-1};','shl.b32 %j,%j,2;','add.u32 %a,%b,%j;',
      'st.local.u32 [%a],%x;','bar.sync 0;','ld.local.u32 %y,[%a];',f'add.u32 %x,%y,{STEP};']
    return '\n'.join(head+body*spec['repeats']+['st.global.u32 [%z],%x;','ret;','}'])+'\n'


def reference(spec,x):
    # Every read has an immediately preceding same-thread write to its index.
    # The observed value is x, followed by modulo2^32 STEP addition. The
    # dependency keeps each subsequent address input-dependent across threads.
    if spec not in specifications():raise ValueError('unregistered LOCAL protocol')
    return ((np.asarray(x,dtype=np.uint32).astype(np.uint64)+spec['repeats']*STEP)&0xffffffff).astype(np.uint32)


def native_evidence(directory,spec,cache):
    value=analyze(directory,cache_dir=cache)
    if value['status']!='available':raise ValueError('LOCAL protocol outside native analysis domain: '+str(value.get('support')))
    f=value['features'];n=spec['repeats'] if spec['kind']=='local' else 0
    if f['local_load_bytes']!=4*n or f['local_store_bytes']!=4*n:raise ValueError('LOCAL protocol was eliminated or changed width')
    if f['barrier_instructions']!=spec['repeats']:raise ValueError('matched barrier control changed')
    # Existing strict zero-Shared proof and original implementation identity.
    value=base.attach_micro_banks(value,dict(kind='baseline',repeats=0,stride=1))
    value['symbolic_banks']['local_protocol']=dict(spec=spec,source_sha256=SOURCE_SHA,zero_shared_verified=True)
    return value


def compile_protocols(out):
    from triton.backends.nvidia.compiler import get_ptxas
    from cost_calibration import source_identity
    out=fresh(out);exe=Path(get_ptxas().path)
    reg=dict(schema_version=1,stage='LOCAL-dependent-protocol-development-only',source=source_identity(),script_sha256=SOURCE_SHA,
        ptxas=str(exe),ptxas_sha256=sha(exe),specifications=specifications(),no_holdout_use=True,
        measurement=dict(batches=10,replays=5,nodes=1024,warmup=100,seed=SEED),
        local_floor=dict(fraction=.10,method='ten percent positive lower95% paired whole-batch bootstrap of matched BAR-control LOCAL protocol increment; includes address/dependency work, not universal hardware latency'),
        safety=dict(tools=['memcheck','racecheck','synccheck','initcheck'],coverage='each tool loads all exact22 cubins, 3 inputs/offsets each'),
        resource_units='driver LOCAL/STACK allocation bytes and static LDL/STL transfer widths are distinct; neither is spill count',
        initialization_proof='per thread and iteration idx=(x&(words-1))*4 lies inside allocated private array; st.local precedes ld.local same idx; all128threads reach every BAR')
    write(out/'registration.json',reg);rows=[]
    try:
        for spec in reg['specifications']:
            d=fresh(out/spec['key']);(d/'kernel.ptx').write_text(ptx(spec));args=[str(exe),'--gpu-name=sm_90','-O3',str(d/'kernel.ptx'),'-o',str(d/'kernel.cubin')]
            start=time.perf_counter_ns();r=subprocess.run(args,capture_output=True,text=True,timeout=60)
            write(d/'compile.json',dict(command=args,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_ns=time.perf_counter_ns()-start))
            if r.returncode:raise RuntimeError('LOCAL micro compilation failed')
            native=native_evidence(d,spec,out/'native-cache');write(d/'native-analysis.json',native)
            rows.append(dict(spec=spec,path=str(d),ptx_sha256=sha(d/'kernel.ptx'),cubin_sha256=sha(d/'kernel.cubin'),native_sha256=sha(d/'native-analysis.json'),features=base.model_features(native)))
            write(out/'manifest.json',dict(status='compiling',registration_sha256=sha(out/'registration.json'),rows=rows))
        write(out/'manifest.json',dict(status='compiled',registration_sha256=sha(out/'registration.json'),rows=rows))
    except Exception as error:write(out/'failure.json',dict(error=type(error).__name__+': '+str(error),rows=len(rows),no_gpu=True));raise
    return out


def verify_protocol(out):
    out=Path(out);m=read(out/'manifest.json');r=read(out/'registration.json')
    if m['status']!='compiled' or sha(out/'registration.json')!=m['registration_sha256'] or r['script_sha256']!=SOURCE_SHA or sha(__file__)!=SOURCE_SHA or sha(r['ptxas'])!=r['ptxas_sha256']:raise ValueError('LOCAL protocol/implementation identity changed')
    if [x['spec'] for x in m['rows']]!=specifications():raise ValueError('LOCAL micro matrix changed')
    for row in m['rows']:
        d=Path(row['path'])
        for filename,key in [('kernel.ptx','ptx_sha256'),('kernel.cubin','cubin_sha256'),('native-analysis.json','native_sha256')]:
            if sha(d/filename)!=row[key]:raise ValueError('LOCAL micro artifact changed')
        if (d/'kernel.ptx').read_text()!=ptx(row['spec']):raise ValueError('LOCAL PTX not registered template')
    return m,r


def load_entries(out,batches):
    import torch
    import support_closure as closure
    from run_joint_ablation import load_exact
    from physical_resources import query_physical_resources
    m,_=verify_protocol(out);closure.activate_target_device();entries={}
    if closure.gpu_state()['foreign_pids']:raise RuntimeError('foreign GPU job before LOCAL training')
    for row in m['rows']:
        d=Path(row['path']);cuda,module,function,binary=load_exact(d);resources=query_physical_resources(cuda,function)
        if row['spec']['kind']=='local' and resources['local_bytes']<4*row['spec']['words']:raise ValueError('retained LOCAL allocation smaller than address domain')
        phases=[]
        for batch in range(batches):
            x=np.random.default_rng(SEED+batch).integers(0,2**32,128,dtype=np.uint32);ref=reference(row['spec'],x);offset=4*(1+batch%3)
            ib=torch.full((x.nbytes+64,),165,dtype=torch.uint8,device='cuda');ib[offset:offset+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8)))
            ob=torch.full((x.nbytes+64,),165,dtype=torch.uint8,device='cuda');args=[ctypes.c_uint64(ib.data_ptr()+offset)]*3+[ctypes.c_uint64(ob.data_ptr()+offset),ctypes.c_uint64(0)]
            params=(ctypes.c_void_p*5)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
            def launch(cuda=cuda,function=function,params=params):
                code=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                if code:raise RuntimeError('LOCAL launch '+str(code))
            def check(ib=ib,ob=ob,x=x,ref=ref,offset=offset):
                torch.cuda.synchronize();raw=ob.cpu().numpy();inputbytes=ib.cpu().numpy()
                if not np.array_equal(raw[offset:offset+ref.nbytes].copy().view(np.uint32),ref):raise ValueError('LOCAL output mismatch')
                for a,size in [(raw,ref.nbytes),(inputbytes,x.nbytes)]:
                    if not np.all(a[:offset]==165) or not np.all(a[offset+size:]==165):raise ValueError('LOCAL buffer canary')
                if not np.array_equal(inputbytes[offset:offset+x.nbytes],x.view(np.uint8)):raise ValueError('LOCAL input modified')
            phases.append(dict(launch=launch,check=check,ib=ib,ob=ob,args=args,params=params,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),offset=offset))
        entries[row['spec']['key']]=dict(phases=phases,row=row,cuda=cuda,module=module,function=function,binary=binary,resources=resources)
    return entries


def check_protocols(out,tool=None):
    import support_closure as closure
    out=Path(out);entries=load_entries(out,3);records=[]
    for key,e in entries.items():
        for batch,p in enumerate(e['phases']):
            p['launch']();p['check']();records.append(dict(key=key,batch=batch,passed=True,guards=True,input_sha256=p['input_sha256'],reference_sha256=p['reference_sha256'],offset=p['offset'],cubin_sha256=e['row']['cubin_sha256']))
    if closure.gpu_state()['foreign_pids']:raise RuntimeError('foreign GPU job during LOCAL validation')
    name='safety-'+tool+'.json' if tool else 'correctness.json';q=out/name
    if q.exists():raise ValueError('fresh LOCAL validation output required')
    write(q,dict(status='passed',tool=tool,records=records,resources={k:e['resources'] for k,e in entries.items()},no_kernel_timing=True));return q


def measure_protocols(out):
    import torch
    import support_closure as closure
    from joint_measure import gpu_state,foreign_pids
    out=Path(out);m,reg=verify_protocol(out);entries=load_entries(out,10);allrows=[];rng=np.random.default_rng(SEED);keys=list(entries)
    if (out/'micro-raw.json').exists():raise ValueError('fresh LOCAL timing output required')
    for tool in reg['safety']['tools']:
        if read(out/('safety-'+tool+'.json'))['status']!='passed':raise ValueError('all LOCAL safety tools required before training timing')
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    try:
        for batch in range(10):
            before=gpu_state()
            if foreign_pids(before):raise RuntimeError('foreign GPU workload before LOCAL paired block')
            order=rng.permutation(keys).tolist();records=[]
            with torch.cuda.stream(stream):
                for key in order:
                    e=entries[key];p=e['phases'][batch];launch=p['launch'];launch();p['check']()
                    for _ in range(100):launch()
                    stream.synchronize();graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(1024):launch()
                    stream.synchronize();cuda=e['cuda'];cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)];actual=ctypes.c_size_t()
                    if cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(actual)) or actual.value!=1024:raise ValueError('LOCAL graph node mismatch')
                    graph.instantiate();elapsed=[]
                    for _ in range(5):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();graph.replay();end.record();end.synchronize();elapsed.append(begin.elapsed_time(end))
                    p['check']();records.append(dict(batch=batch,key=key,order=order,input_sha256=p['input_sha256'],reference_sha256=p['reference_sha256'],offset=p['offset'],cubin_sha256=e['row']['cubin_sha256'],graph_elapsed_ms=elapsed,nodes=1024,actual_graph_nodes=actual.value,samples_us=[x/1024*1000 for x in elapsed],numerics=True,guards=True))
            after=gpu_state();foreign=foreign_pids(before)+foreign_pids(after)
            for record in records:record.update(valid=not foreign,state_before=before,state_after=after)
            allrows+=records;write(out/'micro-raw.json',allrows)
            if foreign:raise RuntimeError('foreign workload invalidated complete LOCAL block; no replacement')
            print(json.dumps(dict(local_micro_batch=batch,images=len(keys))),flush=True)
        labels=[]
        for key,e in entries.items():
            values=[float(np.median(x['samples_us'])) for x in sorted((x for x in allrows if x['key']==key),key=lambda x:x['batch'])]
            d=Path(e['row']['path']);native=read(d/'native-analysis.json')
            labels.append(dict(spec=e['row']['spec'],path=str(d),cubin_sha256=e['row']['cubin_sha256'],native=native,features=base.model_features(native),batch_medians_us=values,median_us=float(np.median(values)),raw_sha256=sha(out/'micro-raw.json'),physical_resources=e['resources']))
        write(out/'labels.json',labels);verify_protocol(out)
        write(out/'completion.json',dict(status='complete',labels_sha256=sha(out/'labels.json'),raw_sha256=sha(out/'micro-raw.json'),registration_sha256=sha(out/'registration.json'),safety_hashes={tool:sha(out/('safety-'+tool+'.json')) for tool in reg['safety']['tools']},images=len(labels),batches=10,no_holdout_use=True))
    except Exception as error:write(out/'failure.json',dict(error=type(error).__name__+': '+str(error),partial_samples_retained=True));raise
    return out


def collect_known(out):
    out=fresh(out);rows=[];unsupported=[]
    for run in OLD_RUNS:
        freeze=read(run/'measurement/freeze.json');completion=read(run/'measurement/completion.json')
        if completion['status']!='complete' or completion['freeze_sha256']!=sha(run/'measurement/freeze.json'):raise ValueError('old timing freeze changed')
        for group,images in freeze['groups'].items():
            rawpath=run/'measurement'/(group+'-raw.json');raw=read(rawpath)
            if sha(rawpath)!=completion['evidence_hashes'][rawpath.name]:raise ValueError('known raw timing changed')
            for row in images:
                d=Path(row['path'])
                if sha(d/'kernel.cubin')!=row['cubin_sha256']:raise ValueError('known cubin changed')
                native=base.attach_symbolic_banks(analyze(d,cache_dir=out/'native-cache'),d)
                selected=sorted((x for x in raw if x['arm']==row['arm']),key=lambda x:x['batch'])
                if len(selected)!=10 or [x['batch'] for x in selected]!=list(range(10)) or any(not x['valid'] or x['cubin_sha256']!=row['cubin_sha256'] for x in selected):raise ValueError('known paired timings incomplete')
                value=dict(source_run=str(run),group=group,path=str(d),spec=row['spec'],cubin_sha256=row['cubin_sha256'],native=native,timing_path=str(rawpath),timing_sha256=sha(rawpath),image_arm=row['arm'])
                if native['status']!='available':unsupported.append(value);continue
                med=[float(np.median(x['samples_ms']))*1000 for x in selected];value.update(features=base.model_features(native),median_us=float(np.median(med)),batch_medians_us=med);rows.append(value)
    write(out/'known-labels.json',rows);write(out/'unsupported.json',unsupported)
    write(out/'completion.json',dict(status='complete',known_labels_sha256=sha(out/'known-labels.json'),unsupported_sha256=sha(out/'unsupported.json'),supported=len(rows),unsupported=len(unsupported),nonzero_local_images=sum(x['features']['local_transfer_bytes']>0 for x in rows),no_holdout_use=True,previous_holdouts='all three previous experiments are now observed training/development only'))
    return out


def local_floor(labels):
    by={x['spec']['key']:x for x in labels};indices=np.random.default_rng(SEED).integers(0,10,(10000,10));draws=[];comparisons=[]
    for row in labels:
        if row['spec']['kind']!='local':continue
        n=row['spec']['repeats'];control=by[f'control-{n}'];count=row['features']['local_transfer_bytes']-control['features']['local_transfer_bytes']
        if count!=8*n:raise ValueError('LOCAL calibration traffic count mismatch')
        delta=(np.asarray(row['batch_medians_us'])-control['batch_medians_us'])/count;replicates=np.median(delta[indices],axis=1);draws.append(replicates)
        comparisons.append(dict(key=row['spec']['key'],control=control['spec']['key'],static_transfer_bytes=count,median_us_per_byte=float(np.median(delta)),ci95=np.quantile(replicates,[.025,.975]).tolist()))
    interval=np.quantile(np.median(np.asarray(draws),axis=0),[.025,.975]).tolist();lower=max(0.,interval[0])
    if lower<=0:raise ValueError('LOCAL protocol increment not identifiable as positive')
    return dict(floor_us_per_unit=.10*lower,fraction=.10,family_ci95=interval,comparisons=comparisons,interpretation='matched-BAR dependent LOCAL roundtrip protocol increment including index/address/recurrence; calibration prior, not universal hardware lower latency',resource_units='native LDL/STL operand bytes per static instruction occurrence; driver LOCAL allocation and spills are separate')


def fit_one_local(design,target,lower,fixed_profile):
    from calibration_analysis import coordinate_fit
    column=FEATURES.index('local_transfer_bytes')
    solution=coordinate_fit(np.asarray(design)[:,[column]],np.asarray(target),ridge=1e-6,max_iterations=20000,tolerance=1e-10,signed=True,fit_intercept=False)
    if not solution['converged']:raise ValueError('incremental LOCAL NNLS did not converge')
    weights=dict(fixed_profile['coefficients']);weights['local_transfer_bytes']=int(math.ceil((solution['coefficients'][0]+lower[column])*1e6))
    if any(weights[name]!=fixed_profile['coefficients'][name] for name in FEATURES if name!='local_transfer_bytes'):raise AssertionError('nonlocal coefficient changed')
    return weights,solution


def fit(known,local,communication,out):
    from calibration_analysis import coordinate_fit
    known=Path(known);local=Path(local);communication=Path(communication);out=fresh(out)
    for directory,key,file in [(known,'known_labels_sha256','known-labels.json'),(local,'labels_sha256','labels.json'),(communication,'labels_sha256','labels.json')]:
        c=read(directory/'completion.json')
        if c['status']!='complete' or sha(directory/file)!=c[key]:raise ValueError('training collection changed')
    for directory in (local,communication):
        if sha(directory/'micro-raw.json')!=read(directory/'completion.json')['raw_sha256']:raise ValueError('micro raw timing changed')
    old=read(known/'known-labels.json');loc=read(local/'labels.json');comm=read(communication/'labels.json');floors=base.communication_floors(comm);floors['local_transfer_bytes']=local_floor(loc)
    rows=old+[dict(x,source_run='development-communication',group='isolated-communication') for x in comm]+[dict(x,source_run='development-LOCAL',group='isolated-LOCAL') for x in loc]
    groups={}
    for row in old:groups.setdefault((row['source_run'],row['group']),[]).append(row)
    for kind in ('barrier','shared','shuffle'):groups[('micro-communication',kind)]=[comm[0]]+[x for x in comm if x['spec']['kind']==kind]
    by={x['spec']['key']:x for x in loc}
    for words in (256,1024,4096):
        selected=[x for x in loc if x['spec']['kind']=='local' and x['spec']['words']==words]
        groups[('micro-LOCAL',str(words))]=selected+[by[f'control-{n}'] for n in sorted({x['spec']['repeats'] for x in selected})]
    fixed_profile_path=PREVIOUS/'native-fit-r3/profile.json';fixed_profile=base.load_profile(fixed_profile_path)
    empirical_floors=np.array([floors.get(name,{}).get('floor_us_per_unit',0.) for name in FEATURES])
    lower=np.array([fixed_profile['coefficients'][name]/1e6 for name in FEATURES]);lower[FEATURES.index('local_transfer_bytes')]=empirical_floors[FEATURES.index('local_transfer_bytes')]
    design=[];targets=[];pairs=[]
    for key,items in sorted(groups.items()):
        pair=list(itertools.combinations(items,2));scale=max(float(np.median([x['median_us'] for x in items])),1e-9)
        for a,b in pair:
            delta=np.array([a['features'][name]-b['features'][name] for name in FEATURES],dtype=float);value=float(np.median(np.asarray(a['batch_medians_us'])-b['batch_medians_us']));weight=1/(scale*np.sqrt(max(1,len(pair))))
            design.append(delta*weight);targets.append((value-delta@lower)*weight);pairs.append(dict(group=list(key),first=a['path'],second=b['path'],delta_us=value,normalization=scale,loss_weight=weight))
    x=np.array([[r['features'][name] for name in FEATURES] for r in rows],dtype=float);y=np.array([r['median_us'] for r in rows])
    for feature,target in zip(x,y):
        weight=np.sqrt(.05/len(rows))/max(target,1e-9);design.append(feature*weight);targets.append((target-feature@lower)*weight)
    matrix=np.asarray(design);target=np.asarray(targets)
    weights,solution=fit_one_local(matrix,target,lower,fixed_profile)
    coefficients=np.array([weights[name]/1e6 for name in FEATURES])
    # Full refit is retained only as a development diagnostic. It never
    # selects this run's main coefficients or reads fresh qualification labels.
    joint=coordinate_fit(matrix,target+matrix@(lower-empirical_floors),ridge=1e-6,max_iterations=20000,tolerance=1e-10,signed=True,fit_intercept=False)
    write(out/'joint-refit-development-only.json',dict(joint,coefficients_with_lower_bounds=(np.asarray(joint['coefficients'])+empirical_floors).tolist(),not_selected=True,no_holdout_use=True))
    contracts=[base.analysis_contract(row['native']) for row in rows]
    if any(c!=contracts[0] for c in contracts) or contracts[0]!=fixed_profile['analysis_contract']:raise ValueError('training analysis contracts differ')
    if weights['local_transfer_bytes']<=0:raise ValueError('LOCAL calibrated fee cannot be zero')
    profile=dict(schema_version=1,model='sm90-native-surviving-schedule-v1',features=list(FEATURES),coefficients=weights,ticks_per_microsecond=1000000,intercept_us=0.,training_maxima={name:int(v) for name,v in zip(FEATURES,x.max(axis=0))},analysis_contract=contracts[0],communication_floors={k:v for k,v in floors.items() if k!='local_transfer_bytes'},local_floor=floors['local_transfer_bytes'],calibration=dict(version='observed-and-isolated-LOCAL-v1',mode='local-only-13-nonlocal-coefficients-frozen',fixed_profile_sha256=sha(fixed_profile_path),implementation_sha256=SOURCE_SHA,known_images=len(old),communication_images=len(comm),local_images=len(loc),known_nonzero_LOCAL_images=sum(r['features']['local_transfer_bytes']>0 for r in old)),interpretation='paired relative train-only empirical native proxy; not cycles or universal latency; LOCAL transfers calibrated separately from allocated frame bytes',unsupported_rule='training-zero feature remains unknown; partial-model selector may prove dominance, otherwise explicit fallback',no_holdout_use=True)
    write(out/'profile.json',profile);write(out/'pair-ledger.json',pairs);write(out/'training-ledger.json',rows);write(out/'fit.json',dict(solution,fitted_parameter_names=['local_transfer_bytes'],fitted_parameters=1,fixed_nonlocal_parameters=13,feature_names=list(FEATURES),constrained_lower_bounds=lower.tolist(),coefficients_with_lower_bounds=coefficients.tolist(),objective='same-program relative paired differences, equal total pair mass/group; weak .05 relative absolute regularization',training_images=len(rows),native_known_images=len(old),LOCAL_protocol_images=len(loc),communication_images=len(comm),local_calibration=floors['local_transfer_bytes'],universal_parameter_identifiability=False))
    write(out/'inputs.json',dict(known_completion=str(known/'completion.json'),known_completion_sha256=sha(known/'completion.json'),micro_completion=str(communication/'completion.json'),micro_completion_sha256=sha(communication/'completion.json'),local_completion=str(local/'completion.json'),local_completion_sha256=sha(local/'completion.json'),profile_sha256=sha(out/'profile.json'),fit_sha256=sha(out/'fit.json'),ledger_sha256=sha(out/'training-ledger.json'),pair_ledger_sha256=sha(out/'pair-ledger.json'),fixed_profile=str(fixed_profile_path),fixed_profile_sha256=sha(fixed_profile_path),no_holdout_use=True))
    return out


def load_profile(path):
    path=Path(path);profile=base.load_profile(path);inputs=read(path.parent/'inputs.json')
    if sha(inputs['local_completion'])!=inputs['local_completion_sha256'] or sha(path.parent/'pair-ledger.json')!=inputs['pair_ledger_sha256']:raise ValueError('LOCAL calibration identity mismatch')
    if sha(inputs['fixed_profile'])!=inputs['fixed_profile_sha256']:raise ValueError('frozen nonlocal profile changed')
    old=read(inputs['fixed_profile'])
    if any(profile['coefficients'][name]!=old['coefficients'][name] for name in FEATURES if name!='local_transfer_bytes'):raise ValueError('incremental calibration changed nonlocal coefficients')
    return profile


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='action',required=True)
    for name in ('compile','check','measure','collect-known'):
        q=s.add_parser(name);q.add_argument('directory',type=Path)
        if name=='check':q.add_argument('--safety-tool',choices=['memcheck','racecheck','synccheck','initcheck'])
    q=s.add_parser('fit');q.add_argument('known',type=Path);q.add_argument('local',type=Path);q.add_argument('communication',type=Path);q.add_argument('out',type=Path);a=p.parse_args()
    if a.action=='compile':compile_protocols(a.directory)
    elif a.action=='check':check_protocols(a.directory,a.safety_tool)
    elif a.action=='measure':measure_protocols(a.directory)
    elif a.action=='collect-known':collect_known(a.directory)
    else:fit(a.known,a.local,a.communication,a.out)
if __name__=='__main__':main()
