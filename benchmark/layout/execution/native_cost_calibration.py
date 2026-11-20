"""Train-only native cost calibration with isolated communication experiments.
No CUDA/import side effects. Native counts are static occurrences, not cycles.
Old exact images/labels and new explicitly development-only PTX probes are used;
no qualification holdout is read by this module.
"""
import argparse,ctypes,hashlib,json,math,subprocess,sys,time
from pathlib import Path
import numpy as np
from support_closure import sha,read,write,fresh
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts'
OLD_RUNS=(ROOT/'guoqiao/optimize_plan/model_selection_quality_artifacts/run-model-selection-quality-r2-20261102T101000Z',
          ROOT/'guoqiao/optimize_plan/demand_cost_search_repair_artifacts/run-demand-repair-r4-20261103')
SEED=20261103
BANK_IMPLEMENTATION_SHA256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'benchmark/layout/execution/backend.py',ROOT/'benchmark/layout/execution/demand_coverage_audit.py',ROOT/'benchmark/layout/execution/redistribution.py')}
# Tied read/write byte coefficient: the experiment does not independently
# identify universal read and write latency. No double identical design columns.
MODEL_FEATURES=('native_instructions','global_load_bytes','global_store_bytes',
    'shared_transfer_bytes','local_transfer_bytes','barrier_instructions',
    'shuffle_instructions','float_arithmetic_instructions','special_math_instructions',
    'integer_logic_instructions','load_use_exposure_32','load_use_exposure_128','dependency_depth','symbolic_bank_excess')
FLOOR_FRACTION=.10


def model_features(native):
    f=native['features'] if 'features' in native else native
    names=set(MODEL_FEATURES)-{'shared_transfer_bytes','local_transfer_bytes','symbolic_bank_excess'}
    result={name:f[name] for name in names}
    result.update(shared_transfer_bytes=f['shared_load_bytes']+f['shared_store_bytes'],
                  local_transfer_bytes=f['local_load_bytes']+f['local_store_bytes'],
                  symbolic_bank_excess=native['symbolic_banks']['shared_read_bank_excess']+native['symbolic_banks']['shared_write_bank_excess'])
    if any(type(x) is not int or x<0 for x in result.values()):raise ValueError('invalid native features')
    return {name:result[name] for name in MODEL_FEATURES}


def analysis_contract(native):
    i=native['identity']
    result={k:i[k] for k in ('schema_version','domain','analyzer_sha256')}
    for name in ('nvdisasm','cuobjdump'):result[name]={k:i[name][k] for k in ('sha256','version')}
    result['symbolic_bank_contract']='frisk-emitted-affine-shared-bank-v1'
    result['symbolic_bank_implementation_sha256']=native['symbolic_banks']['implementation_sha256']
    return result


def bank_implementation_identity():
    if any(sha(ROOT/name)!=digest for name,digest in BANK_IMPLEMENTATION_SHA256.items()):raise ValueError('bank implementation changed since import')
    return dict(BANK_IMPLEMENTATION_SHA256)


def attach_symbolic_banks(native,directory):
    """Combine native surviving counts with Frisk affine Shared address proof.
    This is an emitted-address burden, not a decoded dynamic native bank count.
    Static native LDS/STS byte totals must equal materialized scalar transfers;
    otherwise conservatively reject the complete bank-aware score domain.
    """
    if native.get('status')!='available':return native
    import backend
    from demand_coverage_audit import canonical_query,conversion_structure
    directory=Path(directory);packet=read(directory/'execution.json');observed={}
    def observe(value,indices):observed.setdefault(value,set()).add(canonical_query(indices))
    emitter=backend.Emitter(packet,demand_query_observer=observe);tt,info=emitter.run()
    if hashlib.sha256(tt.encode()).hexdigest()!=sha(directory/'kernel.ttgir'):raise ValueError('bank proof TTGIR/native emission binding differs')
    demand=bool(info.get('demand_lowering'))
    if not demand:observed={v['id']:{(reg,0,0,0,0,0,0,0) for reg in range(v['distributed']['topology'][0])} for v in emitter.v.values() if 'distributed' in v}
    readbanks=writebanks=readbytes=writebytes=0
    for op in emitter.data['operations']:
        if op['name']!='frisk.convert_layout':continue
        source=op['operands'][0];target=op['results'][0];plan=backend.validate_redistribution(op,emitter.v)
        if plan['mode']!='Shared' or demand and not observed.get(target):continue
        f=conversion_structure(emitter,source,target,plan,observed.get(target,set()),demand)
        readbanks+=f['shared_read_bank_excess'];writebanks+=f['shared_write_bank_excess'];readbytes+=4*f['target_vectors'];writebytes+=4*f['source_vectors']
    f=native['features']
    if f['shared_load_bytes']!=readbytes or f['shared_store_bytes']!=writebytes:
        return dict(native,status='unsupported',support=dict(status='unsupported',domain=native['identity']['domain'],reasons=['symbolic/native-shared-transfer-byte-mismatch']),symbolic_banks=dict(expected_load_bytes=readbytes,expected_store_bytes=writebytes))
    return dict(native,symbolic_banks=dict(contract='frisk-emitted-affine-shared-bank-v1',implementation_sha256=bank_implementation_identity(),shared_read_bank_excess=readbanks,shared_write_bank_excess=writebanks,
        execution_sha256=sha(directory/'execution.json'),emitted_ttgir_sha256=sha(directory/'kernel.ttgir'),native_cubin_sha256=native['identity']['cubin_sha256'],
        semantics='GF2 emitted warp worst-bank serialization burden with byte-matched native transfers; not dynamic native conflict measurements'))


def attach_micro_banks(native,spec):
    if native.get('status')!='available':return native
    n=spec['repeats'] if spec['kind']=='shared' else 0
    if native['features']['shared_load_bytes']!=n*4 or native['features']['shared_store_bytes']!=n*4:raise ValueError('micro scalar transfer count differs')
    excess=4*(math.gcd(spec['stride'],32)-1)*n
    return dict(native,symbolic_banks=dict(contract='frisk-emitted-affine-shared-bank-v1',implementation_sha256=bank_implementation_identity(),shared_read_bank_excess=excess,shared_write_bank_excess=excess,
        derivation='four warps; scalar32b word addresses=t*stride or (t xor32)*stride; perwarp max distinct words/bank minus1',micro_spec=spec))


def micro_specs():
    specs=[dict(key='baseline',kind='baseline',repeats=0,stride=1)]
    for kind in ('barrier','shared','shuffle'):
        for repeats in (8,32,128):
            for stride in ((1,4,8) if kind=='shared' else (1,)):
                specs.append(dict(key=f'{kind}-{repeats}-s{stride}',kind=kind,repeats=repeats,stride=stride))
    return specs


def micro_ptx(spec):
    if spec not in micro_specs():raise ValueError('unregistered micro specification')
    body=[]
    for _ in range(spec['repeats']):
        if spec['kind']=='barrier':body+=['bar.sync 0;']
        elif spec['kind']=='shared':body+=['st.volatile.shared.u32 [%a], %x;','bar.sync 0;',
            'ld.volatile.shared.u32 %y, [%b];','bar.sync 0;','xor.b32 %x, %x, %y;']
        elif spec['kind']=='shuffle':body+=['shfl.sync.bfly.b32 %y, %x, 16, 31, 0xffffffff;','xor.b32 %x, %x, %y;']
    shared=128*4*spec['stride']
    return '\n'.join(['.version 8.0','.target sm_90','.address_size 64',
        '.visible .entry frisk_execution(.param .u64 Q,.param .u64 K,.param .u64 V,.param .u64 O,.param .u64 S)',
        '{','.reg .b32 %t,%a,%b,%j,%x,%y,%s;','.reg .b64 %q,%o,%i,%qp,%op;',
        f'.shared .align 4 .b8 scratch[{shared}];',
        'ld.param.u64 %q,[Q];','ld.param.u64 %o,[O];','mov.u32 %t,%tid.x;',
        'mul.wide.u32 %i,%t,4;','add.u64 %qp,%q,%i;','add.u64 %op,%o,%i;',
        'ld.global.u32 %x,[%qp];',f'mul.lo.u32 %a,%t,{4*spec["stride"]};',
        'xor.b32 %j,%t,32;',f'mul.lo.u32 %b,%j,{4*spec["stride"]};',
        'mov.u32 %s,scratch;','add.u32 %a,%s,%a;','add.u32 %b,%s,%b;']+body+['st.global.u32 [%op],%x;','ret;','}'])+'\n'


def micro_reference(spec,x):
    y=np.asarray(x,dtype=np.uint32).copy()
    for _ in range(spec['repeats']):
        if spec['kind'] in ('shared','shuffle'):
            y=np.bitwise_xor(y,y[np.arange(128)^(32 if spec['kind']=='shared' else 16)])
    return y


def compile_micros(out):
    from triton.backends.nvidia.compiler import get_ptxas
    from cost_calibration import source_identity
    out=fresh(out);ptxas=Path(get_ptxas().path)
    reg=dict(schema_version=1,stage='development-only-isolated-communication',source=source_identity(),
        script_sha256=sha(__file__),ptxas=str(ptxas),ptxas_sha256=sha(ptxas),specifications=micro_specs(),
        measurement=dict(batches=10,replays=5,nodes=1024,warmup=100,seed=SEED),
        floors=dict(fraction=FLOOR_FRACTION,confidence=.95,
          method='ten-percent positive lower paired-bootstrap incremental slope; conservative calibration prior, not a universal hardware lower bound'),
        no_holdout_use=True)
    write(out/'registration.json',reg)
    rows=[]
    for spec in reg['specifications']:
        d=fresh(out/spec['key']);(d/'kernel.ptx').write_text(micro_ptx(spec))
        command=[str(ptxas),'--gpu-name=sm_90','-O3',str(d/'kernel.ptx'),'-o',str(d/'kernel.cubin')]
        start=time.perf_counter_ns();r=subprocess.run(command,capture_output=True,text=True)
        write(d/'compile.json',dict(command=command,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr,wall_ns=time.perf_counter_ns()-start))
        if r.returncode:raise RuntimeError('micro compilation failed: '+str(d))
        rows.append(dict(spec=spec,path=str(d),ptx_sha256=sha(d/'kernel.ptx'),cubin_sha256=sha(d/'kernel.cubin')))
    write(out/'manifest.json',dict(status='compiled',registration_sha256=sha(out/'registration.json'),rows=rows))
    return out


def measure_micros(out):
    """Exact input/output/guard check, paired graph timing; no auto replacement."""
    import torch
    import support_closure as closure
    from run_joint_ablation import load_exact
    from native_cost_analysis import analyze
    from joint_measure import gpu_state,foreign_pids
    out=Path(out);manifest=read(out/'manifest.json');reg=read(out/'registration.json')
    if manifest['status']!='compiled' or sha(out/'registration.json')!=manifest['registration_sha256']:raise ValueError('fresh micro matrix required')
    if sha(__file__)!=reg['script_sha256'] or sha(reg['ptxas'])!=reg['ptxas_sha256']:raise ValueError('micro tools changed')
    closure.activate_target_device();entries={};allrows=[];rng=np.random.default_rng(SEED)
    for row in manifest['rows']:
        d=Path(row['path']);spec=row['spec']
        if sha(d/'kernel.cubin')!=row['cubin_sha256'] or sha(d/'kernel.ptx')!=row['ptx_sha256']:raise ValueError('micro image changed')
        native=attach_micro_banks(analyze(d,cache_dir=out/'native-cache'),spec);write(d/'native-analysis.json',native)
        if not native.get('supported',native.get('status') in ('available','supported')):raise ValueError('micro native unsupported: '+str(native))
        cuda,module,function,binary=load_exact(d)
        phases=[]
        for batch in range(10):
            x=np.random.default_rng(SEED+batch).integers(0,2**32,128,dtype=np.uint32);ref=micro_reference(spec,x)
            ib=torch.full((x.nbytes+64,),165,dtype=torch.uint8,device='cuda');ib[16:16+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8)))
            ob=torch.full((x.nbytes+64,),165,dtype=torch.uint8,device='cuda')
            args=[ctypes.c_uint64(ib.data_ptr()+16)]*3+[ctypes.c_uint64(ob.data_ptr()+16),ctypes.c_uint64(0)]
            params=(ctypes.c_void_p*5)(*[ctypes.cast(ctypes.byref(a),ctypes.c_void_p) for a in args])
            def launch(cuda=cuda,function=function,params=params):
                status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                if status:raise RuntimeError('micro launch '+str(status))
            def check(ib=ib,ob=ob,x=x,ref=ref):
                torch.cuda.synchronize();raw=ob.cpu().numpy()
                if not np.array_equal(raw[16:16+ref.nbytes].copy().view(np.uint32),ref):raise ValueError('micro output')
                if not np.all(raw[:16]==165) or not np.all(raw[16+ref.nbytes:]==165):raise ValueError('micro output guards')
                a=ib.cpu().numpy()
                if not np.all(a[:16]==165) or not np.all(a[16+x.nbytes:]==165) or not np.array_equal(a[16:16+x.nbytes],x.view(np.uint8)):raise ValueError('micro input guards')
            launch();check();phases.append(dict(launch=launch,check=check,ib=ib,ob=ob,args=args,params=params,input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        entries[spec['key']]=dict(phases=phases,native=native,module=module,function=function,binary=binary,cuda=cuda,row=row)
    keys=list(entries);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    try:
        for batch in range(10):
            before=gpu_state()
            if foreign_pids(before):raise RuntimeError('foreign job before micro paired batch')
            order=rng.permutation(keys).tolist();records=[]
            with torch.cuda.stream(stream):
                for key in order:
                    e=entries[key];ph=e['phases'][batch];launch=ph['launch'];launch();ph['check']()
                    for _ in range(100):launch()
                    stream.synchronize();graph=torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(1024):launch()
                    stream.synchronize();cuda=e['cuda'];cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)];actual=ctypes.c_size_t()
                    if cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(actual)) or actual.value!=1024:raise ValueError('micro graph node mismatch')
                    graph.instantiate();elapsed=[]
                    for _ in range(5):
                        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();graph.replay();end.record();end.synchronize();elapsed.append(begin.elapsed_time(end))
                    ph['check']();records.append(dict(batch=batch,key=key,order=order,input_sha256=ph['input_sha256'],cubin_sha256=e['row']['cubin_sha256'],graph_elapsed_ms=elapsed,nodes=1024,actual_graph_nodes=actual.value,samples_us=[v/1024*1000 for v in elapsed],numerics=True,guards=True))
            after=gpu_state();foreign=foreign_pids(before)+foreign_pids(after)
            for row in records:row.update(valid=not foreign,state_before=before,state_after=after)
            allrows+=records;write(out/'micro-raw.json',allrows)
            if foreign:raise RuntimeError('foreign job during micro paired batch; retained without replacement')
            print(json.dumps(dict(micro_batch=batch,images=len(keys))),flush=True)
        labels=[]
        for key,e in entries.items():
            raw=[r for r in allrows if r['key']==key]
            med=[float(np.median(r['samples_us'])) for r in sorted(raw,key=lambda r:r['batch'])]
            labels.append(dict(spec=e['row']['spec'],path=e['row']['path'],cubin_sha256=e['row']['cubin_sha256'],native=e['native'],features=model_features(e['native']),batch_medians_us=med,median_us=float(np.median(med)),raw_sha256=sha(out/'micro-raw.json')))
        write(out/'labels.json',labels);write(out/'completion.json',dict(status='complete',labels_sha256=sha(out/'labels.json'),raw_sha256=sha(out/'micro-raw.json'),registration_sha256=sha(out/'registration.json'),images=len(labels),batches=10,no_holdout_use=True))
    except Exception as error:
        write(out/'failure.json',dict(error=type(error).__name__+': '+str(error),partial_samples_retained=True));raise
    return out


def refresh_micros(old,out):
    from native_cost_analysis import analyze
    old=Path(old);completion=read(old/'completion.json');labels=read(old/'labels.json')
    if completion['status']!='complete' or sha(old/'labels.json')!=completion['labels_sha256'] or sha(old/'micro-raw.json')!=completion['raw_sha256']:raise ValueError('old micro timing changed')
    out=fresh(out);rows=[]
    for row in labels:
        d=Path(row['path']);native=attach_micro_banks(analyze(d,cache_dir=out/'native-cache'),row['spec'])
        if native['status']!='available':raise ValueError('micro reanalysis unavailable')
        prior=read(d/'native-analysis.json')
        if native['features']!=prior['features']:raise ValueError('micro native features changed since timing')
        rows.append(dict(row,native=native,features=model_features(native),cpu_reextraction=True))
    (out/'micro-raw.json').write_bytes((old/'micro-raw.json').read_bytes());write(out/'labels.json',rows)
    write(out/'provenance.json',dict(no_new_gpu=True,old_completion=str(old/'completion.json'),old_completion_sha256=sha(old/'completion.json'),old_registration_sha256=sha(old/'registration.json'),raw_bytes_equal=sha(out/'micro-raw.json')==sha(old/'micro-raw.json'),native_feature_counts_equal=True,added='independent GF2 bank burden and final analyzer identity'))
    write(out/'completion.json',dict(status='complete',labels_sha256=sha(out/'labels.json'),raw_sha256=sha(out/'micro-raw.json'),provenance_sha256=sha(out/'provenance.json'),images=len(rows),batches=10,no_new_gpu=True,no_holdout_use=True))
    return out


def collect_known(out):
    from native_cost_analysis import analyze
    out=fresh(out);rows=[];unsupported=[]
    for run in OLD_RUNS:
        freeze=read(run/'measurement/freeze.json');completion=read(run/'measurement/completion.json')
        if completion['status']!='complete' or completion['freeze_sha256']!=sha(run/'measurement/freeze.json'):raise ValueError('old timing freeze mismatch')
        for group,images in freeze['groups'].items():
            rawpath=run/'measurement'/(group+'-raw.json');raw=read(rawpath)
            if completion['evidence_hashes'][rawpath.name]!=sha(rawpath):raise ValueError('known raw timing changed')
            for row in images:
                d=Path(row['path'])
                if sha(d/'kernel.cubin')!=row['cubin_sha256']:raise ValueError('known cubin changed')
                # Analysis lives in new cache, never writes prior sealed images.
                native=attach_symbolic_banks(analyze(d,cache_dir=out/'native-cache'),d)
                selected=sorted((r for r in raw if r['arm']==row['arm']),key=lambda r:r['batch'])
                if len(selected)!=10 or [r['batch'] for r in selected]!=list(range(10)) or any(not r['valid'] or r['cubin_sha256']!=row['cubin_sha256'] for r in selected):raise ValueError('known paired timing incomplete')
                value=dict(source_run=str(run),group=group,path=str(d),spec=row['spec'],cubin_sha256=row['cubin_sha256'],native=native,timing_path=str(rawpath),timing_sha256=sha(rawpath),image_arm=row['arm'])
                if not native.get('supported',native.get('status') in ('available','supported')):
                    unsupported.append(value);continue
                med=[float(np.median(r['samples_ms']))*1000 for r in selected]
                value.update(features=model_features(native),median_us=float(np.median(med)),batch_medians_us=med)
                rows.append(value)
    write(out/'known-labels.json',rows);write(out/'unsupported.json',unsupported)
    write(out/'completion.json',dict(status='complete',known_labels_sha256=sha(out/'known-labels.json'),unsupported_sha256=sha(out/'unsupported.json'),supported=len(rows),unsupported=len(unsupported),no_holdout_use=True,previous_holdouts='all now observed and training-only'))
    return out


def communication_floors(labels):
    by={r['spec']['key']:r for r in labels};base=by['baseline'];rng=np.random.default_rng(SEED);resample=rng.integers(0,10,(10000,10))
    result={};proof=[]
    # Single primitive comparisons isolate BAR/shuffle. Shared removes measured
    # barrier marginal cost first. Floors are calibration priors, not proof of
    # additive non-overlap in an arbitrary full kernel.
    for kind,feature in [('barrier','barrier_instructions'),('shuffle','shuffle_instructions'),('shared','shared_transfer_bytes')]:
        slopes=[];family_draws=[]
        for repeats in (8,32,128):
            row=by[f'{kind}-{repeats}-s1'];delta=np.array(row['batch_medians_us'])-base['batch_medians_us']
            count=row['features'][feature]-base['features'][feature]
            if count<=0:raise ValueError('communication primitive eliminated by compiler')
            if kind=='shared':
                barriers=row['features']['barrier_instructions']-base['features']['barrier_instructions']
                paired_barrier=by[f'barrier-{repeats}-s1']
                barrier_count=paired_barrier['features']['barrier_instructions']-base['features']['barrier_instructions']
                if barrier_count<=0:raise ValueError('paired barrier primitive eliminated')
                # Preserve covariance and propagate uncertainty of the BAR
                # control through the SAME whole-batch bootstrap indices.
                barrier_delta=np.array(paired_barrier['batch_medians_us'])-base['batch_medians_us']
                delta=delta-(barriers/barrier_count)*barrier_delta
            samples=delta/count;draws=np.median(samples[resample],axis=1);low,high=np.quantile(draws,[.025,.975]);family_draws.append(draws)
            slopes.append(dict(repeats=repeats,unit_count=count,median=float(np.median(samples)),ci95=[float(low),float(high)]))
        measured=max(0.,float(np.median([s['median'] for s in slopes])))
        family_ci=np.quantile(np.median(np.asarray(family_draws),axis=0),[.025,.975]).tolist()
        lower=max(0.,float(family_ci[0]));floor=FLOOR_FRACTION*lower
        if floor<=0:raise ValueError('communication fee not identifiable as positive: '+kind)
        result[feature]=dict(floor_us_per_unit=floor,measured_slope_us=measured,fraction=FLOOR_FRACTION,positive_family_lower=lower,family_ci95=family_ci,comparisons=slopes,interpretation='paired dependent protocol increment including retention XOR/address work; not an isolated hardware latency',uncertainty='same-batch joint bootstrap including BAR control uncertainty for Shared')
    return result


def fit(known,micro,out):
    from calibration_analysis import coordinate_fit
    known=Path(known);micro=Path(micro);out=fresh(out)
    kc=read(known/'completion.json');mc=read(micro/'completion.json')
    if kc['status']!='complete' or mc['status']!='complete' or sha(known/'known-labels.json')!=kc['known_labels_sha256'] or sha(micro/'labels.json')!=mc['labels_sha256'] or sha(micro/'micro-raw.json')!=mc['raw_sha256']:raise ValueError('training evidence changed')
    old=read(known/'known-labels.json');micros=read(micro/'labels.json');floors=communication_floors(micros)
    rows=old+[dict(r,source_run='development-only-communication',group='isolated-communication',spec=r['spec']) for r in micros]
    x=np.array([[r['features'][name] for name in MODEL_FEATURES] for r in rows],dtype=float);y=np.array([r['median_us'] for r in rows])
    lower=np.array([floors.get(name,{}).get('floor_us_per_unit',0.) for name in MODEL_FEATURES])
    # Program-internal paired differences remove launch/inter-program intercepts.
    # Every old program-group has equal total loss mass; micro families each
    # contribute one equal-mass group. A fixed .05 relative absolute term is
    # diagnostic regularization, not an assertion of universal timing accuracy.
    groups={}
    for row in old:groups.setdefault((row['source_run'],row['group']),[]).append(row)
    for kind in ('barrier','shared','shuffle'):
        groups[('micro',kind)]=[micros[0]]+[r for r in micros if r['spec']['kind']==kind]
    design=[];targets=[];pair_ledger=[]
    for group,items in sorted(groups.items()):
        pairs=[(a,b) for i,a in enumerate(items) for b in items[i+1:]]
        scale=max(float(np.median([r['median_us'] for r in items])),1e-9)
        for first,second in pairs:
            delta=np.array([first['features'][name]-second['features'][name] for name in MODEL_FEATURES],dtype=float)
            target=float(np.median(np.array(first['batch_medians_us'])-second['batch_medians_us']))
            weight=1./(scale*np.sqrt(max(1,len(pairs))))
            design.append(delta*weight);targets.append((target-delta@lower)*weight)
            pair_ledger.append(dict(group=list(group),first=first['path'],second=second['path'],delta_us=target,normalization=scale,loss_weight=weight))
    # Equal-mass normalized absolute observations retain a weak general scale.
    for feature,target in zip(x,y):
        weight=np.sqrt(.05/len(rows))/max(target,1e-9)
        design.append(feature*weight);targets.append((target-feature@lower)*weight)
    solution=coordinate_fit(np.asarray(design),np.asarray(targets),ridge=1e-6,max_iterations=20000,tolerance=1e-10,signed=True,fit_intercept=False)
    if not solution['converged']:raise ValueError('native paired NNLS failed')
    coefficients=np.array(solution['coefficients'])+lower
    weights={name:int(math.ceil(v*1000000)) for name,v in zip(MODEL_FEATURES,coefficients)}
    # No zero cost for an unobserved instruction family: explicitly unsupported
    # feature extrapolation at scoring time, rather than fabricate a fee.
    contracts=[analysis_contract(r['native']) for r in old+micros]
    if any(c!=contracts[0] for c in contracts):raise ValueError('training native analysis contracts differ')
    profile=dict(schema_version=1,analysis_contract=contracts[0],model='sm90-native-surviving-schedule-v1',features=list(MODEL_FEATURES),coefficients=weights,
        ticks_per_microsecond=1000000,intercept_us=solution['intercept'],
        training_maxima={name:int(v) for name,v in zip(MODEL_FEATURES,x.max(axis=0))},
        communication_floors=floors,interpretation='train-only nonnegative empirical static-native proxy; not a cycle model; intercept diagnostics only',
        unsupported_rule='whole-program pool fallback if any candidate analysis unsupported or any training-zero feature activates',no_holdout_use=True)
    write(out/'profile.json',profile);write(out/'pair-ledger.json',pair_ledger);write(out/'fit.json',dict(solution,objective='equal-mass within-program paired relative differences plus fixed .05 relative absolute regularization',feature_names=list(MODEL_FEATURES),constrained_lower_bounds=lower.tolist(),coefficients_with_lower_bounds=coefficients.tolist(),training_images=len(rows),known_images=len(old),micro_images=len(micros),universal_parameter_identifiability=False))
    ledger=[{k:r[k] for k in ('source_run','group','spec','path','cubin_sha256','features','median_us','batch_medians_us')} for r in rows]
    write(out/'training-ledger.json',ledger)
    write(out/'inputs.json',dict(known_completion=str(known/'completion.json'),known_completion_sha256=sha(known/'completion.json'),micro_completion=str(micro/'completion.json'),micro_completion_sha256=sha(micro/'completion.json'),profile_sha256=sha(out/'profile.json'),fit_sha256=sha(out/'fit.json'),ledger_sha256=sha(out/'training-ledger.json'),no_holdout_use=True))
    return out


def load_profile(path):
    path=Path(path);profile=read(path);inputs=read(path.parent/'inputs.json')
    if inputs.get('no_holdout_use') is not True or sha(path)!=inputs['profile_sha256'] or sha(path.parent/'fit.json')!=inputs['fit_sha256'] or sha(path.parent/'training-ledger.json')!=inputs['ledger_sha256']:
        raise ValueError('native profile/training identity mismatch')
    for name in ('known_completion','micro_completion'):
        if sha(inputs[name])!=inputs[name+'_sha256']:raise ValueError('native training completion changed')
    if profile.get('features')!=list(MODEL_FEATURES) or profile.get('model')!='sm90-native-surviving-schedule-v1' or profile.get('no_holdout_use') is not True:
        raise ValueError('wrong native profile contract')
    return profile


def score(native,profile):
    if profile.get('model')!='sm90-native-surviving-schedule-v1' or profile.get('features')!=list(MODEL_FEATURES):raise ValueError('wrong native model')
    if not native.get('supported',native.get('status') in ('available','supported')):return dict(status='unsupported',reason='native-schedule-domain',cost=None)
    if native.get('symbolic_banks',{}).get('contract')!='frisk-emitted-affine-shared-bank-v1':return dict(status='unsupported',reason='symbolic-bank-proof-unavailable',cost=None)
    if analysis_contract(native)!=profile['analysis_contract']:return dict(status='unsupported',reason='native-analysis-contract-mismatch',cost=None)
    f=model_features(native)
    if any(f[n] and profile['training_maxima'][n]==0 for n in MODEL_FEATURES):return dict(status='unsupported',reason='training-zero-feature-activation',cost=None)
    weights=profile['coefficients']
    if set(weights)!=set(MODEL_FEATURES) or any(type(v) is not int or v<0 for v in weights.values()):raise ValueError('invalid native coefficients')
    if any(weights[n]<=0 for n in ('shared_transfer_bytes','barrier_instructions','shuffle_instructions')):raise ValueError('communication cannot be free')
    return dict(status='supported',cost=sum(f[n]*weights[n] for n in MODEL_FEATURES),features=f,profile_sha256=hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest(),uses_gpu_timing=False)


def main():
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest='action',required=True)
    for name in ('compile-micros','measure-micros','collect-known'):
        q=sub.add_parser(name);q.add_argument('out',type=Path)
    q=sub.add_parser('fit');q.add_argument('known',type=Path);q.add_argument('micro',type=Path);q.add_argument('out',type=Path)
    args=a.parse_args()
    if args.action=='compile-micros':compile_micros(args.out)
    elif args.action=='measure-micros':measure_micros(args.out)
    elif args.action=='collect-known':collect_known(args.out)
    elif args.action=='fit':fit(args.known,args.micro,args.out)
if __name__=='__main__':main()
