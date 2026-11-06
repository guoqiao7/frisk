"""Fresh-source execution bridge and independent Attention tile regression.

The old sealed run is read only. Identical single-tile programs reuse evidence
only after complete emission/config/input/physical-ABI checks. Multi-CTA programs
always receive new numerical and sanitizer execution, with distinct tile inputs.
No timing labels or performance-selection feedback are collected here.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COUNTS = (1, 2, 257)
TOOLS = ('memcheck', 'racecheck', 'synccheck', 'initcheck')
PRIOR_INDEX = ROOT/'guoqiao/optimize_plan/current_gpu_closure_artifacts/evidence-index.json'
PRIOR_INDEX_SHA256 = '6882b17c01a3879fa38683390673bcf2cc48afc0b90fd401aea6ffb65a150fc5'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def _prior_seal(prior):
    prior=Path(prior).resolve()
    if sha(PRIOR_INDEX)!=PRIOR_INDEX_SHA256:
        raise ValueError('accepted prior execution index changed')
    index=read(PRIOR_INDEX)
    if (Path(index['root']).resolve()!=ROOT.resolve() or prior.parent!=PRIOR_INDEX.parent.resolve()
        or prior.name!=index['accepted_run']):
        raise ValueError('only the sealed accepted 22-configuration run is a reuse source')
    entries={}
    for row in index['entries']:
        path=(ROOT/row['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()):raise ValueError('sealed path outside repository')
        if str(path) in entries and entries[str(path)]!=row['sha256']:raise ValueError('conflicting prior sealed entry')
        entries[str(path)]=row['sha256']
    return entries


def _sealed_prior_file(path,entries):
    path=Path(path)
    if path.is_symlink() or str(path.resolve()) not in entries or sha(path)!=entries[str(path.resolve())]:
        raise ValueError('old execution file differs from recorded seal: '+str(path))
    return entries[str(path.resolve())]


def source_identity():
    from cost_calibration import source_identity as current
    return current()


def _runtime_environment():
    return {key:os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','CUDA_DEVICE_ORDER','LD_LIBRARY_PATH')}


def _verify_environment(registration,tool):
    raw=_runtime_environment();canonical=dict(raw)
    if tool:
        original=registration['runtime_environment']['LD_LIBRARY_PATH']
        prefix=registration['tools']['sanitizer_runtime']['directory']
        allowed=prefix+(':'+original if original is not None else '')
        if raw['LD_LIBRARY_PATH']!=allowed:
            raise ValueError('unexpected sanitizer LD_LIBRARY_PATH injection')
        canonical['LD_LIBRARY_PATH']=original
    if canonical!=registration['runtime_environment']:
        raise ValueError('execution CUDA environment differs from registration')
    return dict(raw=raw,canonical=canonical,allowed_sanitizer_prefix=registration['tools']['sanitizer_runtime']['directory'] if tool else None)


def verify(run,tool=None):
    run=Path(run); registration=read(run/'registration.json')
    if source_identity()!=registration['source']:
        raise ValueError('source differs from the execution registration')
    if sha(__file__)!=registration['worker_sha256']:
        raise ValueError('execution worker changed')
    if digest(read(run/'manifest.json')['rows'])!=registration['rows_sha256']:
        raise ValueError('execution row matrix changed')
    if sha(registration['prior_path']+'/manifest.json')!=registration['prior_manifest_sha256'] or sha(registration['prior_path']+'/registration.json')!=registration['prior_registration_sha256']:
        raise ValueError('prior sealed execution identity changed')
    if registration['prior_evidence_index']!=str(PRIOR_INDEX) or registration['prior_evidence_index_sha256']!=PRIOR_INDEX_SHA256 or sha(registration['prior_evidence_index'])!=registration['prior_evidence_index_sha256']:
        raise ValueError('prior sealed evidence index changed')
    if sha(run/'single-tile-emission-bridge.json')!=registration['bridge_sha256']:
        raise ValueError('single-tile bridge changed')
    if sha(run/'protocol.json')!=registration['protocol_sha256']:
        raise ValueError('execution protocol changed')
    from cost_calibration import check_tools
    check_tools(registration['tools'])
    _verify_environment(registration,tool)
    return registration


def _check_old_inputs(directory,correctness):
    import support_closure_fixture as fixture
    spec=read(directory/'calibration_spec.json'); info=read(directory/'execution_config.json')
    expected={}
    for name,q,k,v in fixture.numerical_cases(spec):
        expected[name]=dict(inputs=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)],
                            reference=hashlib.sha256(fixture.reference(spec,q,k,v).tobytes()).hexdigest())
    records=correctness['inputs']
    if not records or {r['case'] for r in records}!=set(expected):
        raise ValueError('old numerical input matrix does not match current fixture')
    for row in records:
        value=expected[row['case']]
        if row['input_hashes']!=value['inputs'] or row['reference_sha256']!=value['reference']:
            raise ValueError('old inputs/reference differ from current numerical protocol')
    current=fixture.safety_input_identity(spec,info,correctness['physical_params'])
    if digest(current)!=digest(correctness['safety_input_identity']):
        raise ValueError('old safety protocol differs from current inputs/ABI')
    return dict(numerical_cases=len(expected),input_records=len(records),safety_input_identity_sha256=digest(current))


def prepare(run,prior):
    from backend import emit
    from tiled_execution import lift_emission
    from support_closure import verify_accepted_row, verify_declared_rows, safety_key
    run=Path(run); run.mkdir(parents=True,exist_ok=False); prior=Path(prior).resolve()
    prior_entries=_prior_seal(prior)
    for name in ('manifest.json','protocol.json','registration.json'):_sealed_prior_file(prior/name,prior_entries)
    old=read(prior/'manifest.json'); old_protocol=read(prior/'protocol.json')
    if old['status']!='validated':raise ValueError('validated prior execution matrix required')
    verify_declared_rows(old['rows'],old_protocol)
    if _runtime_environment()['CUDA_MODULE_LOADING']!='LAZY':
        raise ValueError('execution registration requires explicit CUDA_MODULE_LOADING=LAZY')
    identity=source_identity()
    from support_closure import frozen_tools,OPT
    tools=frozen_tools(OPT)
    runtime_environment=_runtime_environment()
    protocol=dict(schema_version=1,stage='network-backend-execution-regression',
        single_tile='all 22 old natural assignments; exact emission/config/input/ABI bridge or new execution',
        multi_tile=dict(abis=['weak-vector','strong-cpasync','strong-tma'],tile_counts=list(COUNTS),
            cases=['seed0+per-tile-f16-perturbation','seed1+per-tile-f16-perturbation'],
            numerics='existing independent FP32 stable-softmax/RNE-f16/PV reference; abs<=.002+.002*abs(reference)',
            offsets='count2: all old ABI offsets; count1 and257: last registered offset',
            paths=['ordinary-repeat0','ordinary-repeat1','nondefault-stream','capture3'],
            safety='first case at tile_count2 and257, same offsets, every named sanitizer; count1 also executed under every tool',
            guard_byte=165,inter_tile_padding=128,
            scope='independent batch of full fixed-shape tiles; no long-sequence or cross-CTA claim'),
        gpu_uuid='GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618',device_ordinal=1,
        safety_tools=list(TOOLS),no_kernel_timing=True)
    write(run/'protocol.json',protocol)
    rows=[]; bridges=[]
    for oldrow in old['rows']:
        directory=Path(oldrow['path'])
        if directory.resolve().parent!=(prior/'cases').resolve():raise ValueError('prior case outside accepted run')
        for name,value in oldrow['artifact_hashes'].items():
            if not (directory/name).resolve().is_relative_to(prior):raise ValueError('prior artifact escapes accepted run')
            if _sealed_prior_file(directory/name,prior_entries)!=value:raise ValueError('row artifact digest differs from recorded seal')
        correctness=verify_accepted_row(directory,oldrow)
        checked_inputs=_check_old_inputs(directory,correctness)
        packet=read(directory/'execution.json'); text,info=emit(packet)
        name=directory.name; target=run/'single'/name; target.mkdir(parents=True)
        (target/'emitted.ttgir').write_text(text);write(target/'emitted-config.json',info)
        equal_text=text==(directory/'kernel.ttgir').read_text()
        equal_config=digest(info)==digest(read(directory/'execution_config.json'))
        bridge=dict(name=name,old_path=str(directory),execution_sha256=sha(directory/'execution.json'),
            current_tt_sha256=sha(target/'emitted.ttgir'),old_tt_sha256=sha(directory/'kernel.ttgir'),
            text_equal=equal_text,complete_config_equal=equal_config,
            cubin_sha256=sha(directory/'kernel.cubin'),physical_abi=correctness['physical_params'],
            old_correctness_sha256=sha(directory/'correctness.json'),
            old_row_sha256=digest(oldrow),checked_inputs=checked_inputs,
            safety_key=safety_key(directory,correctness),old_safety=oldrow['safety'])
        for name2 in ('execution.json','calibration_spec.json'):
            shutil.copyfile(directory/name2,target/name2)
        if equal_text and equal_config:
            bridge['status']='EXACT_FULL_EMISSION_AND_PROTOCOL_REUSE'
        else:
            bridge['status']='CHANGED_REQUIRES_FRESH_EXECUTION'
            rows.append(dict(path=str(target.resolve()),kind='single',spec=oldrow['spec']))
        bridges.append(bridge)
        if oldrow['model']=='legacy' and oldrow['spec']['case_id']=='attention':
            for count in COUNTS:
                tile=run/'tiled'/(oldrow['spec']['abi_stratum']+'-'+str(count));tile.mkdir(parents=True)
                # 128 byte inter-tile gaps are independently checked canaries.
                from storage_addresses import global_root_spans
                spans=global_root_spans(packet,info['root_order'])
                strides=[span+128 for span in spans]
                tiled,config=lift_emission(packet,text,info,tile_count=count,root_stride_bytes=strides)
                (tile/'emitted.ttgir').write_text(tiled);write(tile/'emitted-config.json',config)
                shutil.copyfile(directory/'execution.json',tile/'execution.json')
                shutil.copyfile(directory/'calibration_spec.json',tile/'calibration_spec.json')
                rows.append(dict(path=str(tile.resolve()),kind='tiled',spec=oldrow['spec'],tile_count=count))
    if source_identity()!=identity:raise ValueError('source changed during execution preparation')
    write(run/'single-tile-emission-bridge.json',dict(rows=bridges,reused=sum(x['status'].startswith('EXACT') for x in bridges),changed=sum(x['status'].startswith('CHANGED') for x in bridges)))
    write(run/'registration.json',dict(source=identity,tools=tools,runtime_environment=runtime_environment,protocol_sha256=sha(run/'protocol.json'),
        bridge_sha256=sha(run/'single-tile-emission-bridge.json'),rows_sha256=digest(rows),prior_path=str(prior),
        prior_manifest_sha256=sha(prior/'manifest.json'),prior_registration_sha256=sha(prior/'registration.json'),
        prior_source_digest=read(prior/'registration.json')['source']['source_digest'],
        prior_evidence_index=str(prior.parent/'evidence-index.json'),prior_evidence_index_sha256=sha(prior.parent/'evidence-index.json'),
        worker_sha256=sha(__file__),created_ns=time.time_ns()))
    write(run/'manifest.json',dict(status='prepared',rows=rows))
    return dict(single=len(bridges),multi=len([x for x in rows if x['kind']=='tiled']),requires_execution=len(rows))


def compile_all(run):
    verify(run)
    import triton
    from triton.backends.compiler import GPUTarget
    from async_host import physical_abi
    run=Path(run);manifest=read(run/'manifest.json');reused={};records=[]
    if manifest['status']!='prepared':raise ValueError('fresh prepared compilation matrix required')
    if any((Path(row['path'])/'kernel.ttgir').exists() for row in manifest['rows']):
        raise FileExistsError('fresh compilation refuses overwriting an earlier attempt')
    for row in manifest['rows']:
        directory=Path(row['path']);text=(directory/'emitted.ttgir').read_text();info=read(directory/'emitted-config.json')
        key=hashlib.sha256(text.encode()).hexdigest()
        started=time.perf_counter_ns()
        (directory/'kernel.ttgir').write_text(text)
        if key in reused:
            source=reused[key]
            for name in ('kernel.ptx','kernel.cubin'):shutil.copyfile(source/name,directory/name)
            record=dict(exact_native_input_reuse=str(source),input_sha256=key,compiler_options=dict(num_warps=4,enable_fp_fusion=False))
        else:
            kernel=triton.compile(str(directory/'kernel.ttgir'),target=GPUTarget('cuda',90,32),options=dict(num_warps=4,enable_fp_fusion=False))
            (directory/'kernel.ptx').write_text(kernel.asm['ptx']);(directory/'kernel.cubin').write_bytes(kernel.asm['cubin'])
            reused[key]=directory
            record=dict(exact_native_input_reuse=None,input_sha256=key,compiler_options=dict(num_warps=4,enable_fp_fusion=False))
        abi=physical_abi((directory/'kernel.ptx').read_text(),len(info['tma_plans']))
        if row['kind']=='tiled' and '%ctaid.x' not in (directory/'kernel.ptx').read_text():raise ValueError('native PTX dropped independent CTA coordinate')
        write(directory/'execution_config.json',info)
        record.update(compile_environment={key:os.environ.get(key) for key in ('TRITON_ALWAYS_COMPILE','TRITON_CACHE_DIR','CUDA_MODULE_LOADING','CUDA_VISIBLE_DEVICES')},wall_ns=time.perf_counter_ns()-started,physical_abi=abi,cubin_sha256=sha(directory/'kernel.cubin'),
            files={name:sha(directory/name) for name in ('kernel.ttgir','kernel.ptx','kernel.cubin','execution_config.json','execution.json','calibration_spec.json')})
        write(directory/'compile.json',record);records.append(dict(path=str(directory),**record))
    verify(run);manifest['status']='compiled';write(run/'manifest.json',manifest);write(run/'compile-summary.json',records)
    return dict(rows=len(records),native_inputs=len(reused))


def _tile_cases(spec,count):
    import support_closure_fixture as fixture
    cases=fixture.numerical_cases(spec)
    if count is None:
        return [(name,[q],[k],[v],[fixture.reference(spec,q,k,v)]) for name,q,k,v in cases]
    result=[]
    for name,q,k,v in cases[:2]:
        arrays=[[],[],[]];references=[]
        for tile in range(count):
            # Exactly representable binary increments before f16 rounding, with
            # distinct complete byte payloads independently checked below.
            qq=(q.astype(np.float32)+np.float32(tile/1024)).astype(np.float16)
            kk=(k.astype(np.float32)+np.float32((tile%17)/512)).astype(np.float16)
            vv=(v.astype(np.float32)+np.float32(tile/2048)).astype(np.float16)
            for target,value in zip(arrays,(qq,kk,vv)):target.append(value)
            references.append(fixture.reference(spec,qq,kk,vv))
        if len({hashlib.sha256(qi.tobytes()+ki.tobytes()+vi.tobytes()).hexdigest() for qi,ki,vi in zip(*arrays)})!=count:
            raise ValueError('multi-CTA inputs must be distinct per tile')
        result.append((name,*arrays,references))
    return result


def execute_one(run,directory,tool=None):
    registration=verify(run,tool);run=Path(run);directory=Path(directory)
    rows=[row for row in read(run/'manifest.json')['rows'] if row['path']==str(directory.resolve())]
    if len(rows)!=1:raise ValueError('configuration missing from execution registration')
    row=rows[0]; compiled=read(directory/'compile.json')
    output_name=(('safety-'+tool+'-') if tool else '')+'correctness.json'
    if (directory/output_name).exists():raise FileExistsError('fresh execution refuses overwrite')
    if tool:
        accepted=read(directory/'correctness.json')
        if accepted['status']!='passed' or accepted['cubin_sha256']!=compiled['cubin_sha256']:
            raise ValueError('safety requires the exact normally accepted native image')
    for name,value in compiled['files'].items():
        if sha(directory/name)!=value:raise ValueError('compiled image input changed: '+name)
    from support_closure import activate_target_device,validate_runtime_regions
    from run_joint_ablation import load_exact
    from async_host import physical_abi,validate_launch,TensorMaps
    from tiled_execution import validate_tiled_launch,TiledTensorMaps
    from physical_resources import query_physical_resources
    import support_closure_fixture as fixture
    import torch
    target=activate_target_device();cuda,module,function,binary=load_exact(directory)
    if hashlib.sha256(binary).hexdigest()!=compiled['cubin_sha256']:raise ValueError('loaded image changed')
    info=read(directory/'execution_config.json');spec=row['spec'];tiled=row['kind']=='tiled'
    count=row.get('tile_count') if tiled else None;tiles=count or 1
    grid=[tiles,1,1];abi=physical_abi((directory/'kernel.ptx').read_text(),len(info['tma_plans']))
    physical=query_physical_resources(cuda,function)
    records=[];input_records=[]
    try:
        cases=_tile_cases(spec,count)
        if tool:cases=cases[:1]
        offsets=fixture.offsets(spec,info['copy_path'])
        if tiled and count!=2:offsets=offsets[-1:]
        for case_name,qs,ks,vs,refs in cases:
            arrays=[qs,ks,vs,refs]
            spans=[x[0].nbytes for x in arrays]
            strides=info['tiled_execution']['root_stride_bytes'] if tiled else spans
            required=[(tiles-1)*stride+span for stride,span in zip(strides,spans)]
            for offset,output_offset in offsets:
                positions=[offset]*3+[output_offset];buffers=[];saved=[]
                for i,(values,nbytes,start,stride) in enumerate(zip(arrays,required,positions,strides)):
                    buffer=torch.full((nbytes+start+272,),165,dtype=torch.uint8,device='cuda')
                    if i<3:
                        for tile,value in enumerate(values):buffer[start+tile*stride:start+tile*stride+value.nbytes].copy_(torch.from_numpy(value.view(np.uint8).reshape(-1)))
                    buffers.append(buffer)
                saved=[x.cpu().numpy().copy() for x in buffers[:3]]
                pointers=[x.data_ptr()+start for x,start in zip(buffers,positions)]
                if tiled:
                    launch_contract=validate_tiled_launch(info,pointers,[b.numel()-start for b,start in zip(buffers,positions)],grid=grid)
                    maps=TiledTensorMaps(info,pointers,[b.numel()-start for b,start in zip(buffers,positions)],grid=grid) if info['tma_plans'] else None
                else:
                    validate_launch(info,pointers);regions=validate_runtime_regions(pointers,spans);maps=TensorMaps(info,pointers) if info['tma_plans'] else None;launch_contract=dict(grid=grid,threads=128,**regions)
                args=[ctypes.c_uint64(pointer) for pointer in pointers+([d.data_ptr() for d in maps.device] if maps else [])+[0]]
                if len(args)!=len(abi):raise ValueError('physical ABI mismatch')
                params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
                input_records.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,
                    per_tile_inputs=[[hashlib.sha256(x.tobytes()).hexdigest() for x in values] for values in arrays[:3]],
                    per_tile_references=[hashlib.sha256(x.tobytes()).hexdigest() for x in refs],
                    descriptor_arrays=[dict(pointer=x.data_ptr(),bytes=x.numel(),alignment=64) for x in maps.device] if maps else [],
                    launch_contract=launch_contract))
                def launch():
                    status=cuda.cuLaunchKernel(function,*grid,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
                    if status:raise RuntimeError('cuLaunchKernel '+str(status))
                def check(path):
                    torch.cuda.synchronize();raw=buffers[3].cpu().numpy();mask=np.zeros(raw.size,dtype=bool);worst=0.0
                    for tile,reference in enumerate(refs):
                        start=output_offset+tile*strides[3];stop=start+reference.nbytes;mask[start:stop]=True
                        got=raw[start:stop].copy().view(np.float32).reshape(reference.shape)
                        checked=fixture.check_numerics(spec,got,reference)
                        if not checked['passed']:raise ValueError('numerical failure '+str(checked))
                        worst=max(worst,checked['max_abs_error'])
                    if not np.all(raw[~mask]==165):raise ValueError('output boundary/inter-tile canary corrupted')
                    if any(not np.array_equal(x.cpu().numpy(),before) for x,before in zip(buffers[:3],saved)):
                        raise ValueError('input bytes or input padding changed')
                    records.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,path=path,
                        tiles=tiles,passed=True,max_abs_error=worst,canaries=True,inputs_unchanged=True))
                for repeat in range(1 if tool else 2):
                    buffers[3].fill_(165);launch();check('ordinary'+str(repeat))
                if not tool:
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):buffers[3].fill_(165);launch()
                    check('nondefault-stream')
                    graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
                    with torch.cuda.graph(graph,stream=stream):
                        for _ in range(3):launch()
                    dot=directory/('capture-'+case_name+'-'+str(offset)+'-'+str(output_offset)+'.dot');graph.debug_dump(str(dot))
                    nodes=dot.read_text().count('frisk_execution')
                    if nodes!=3:raise ValueError('expected exact three-node kernel graph')
                    buffers[3].fill_(165);graph.replay();check('capture3');records[-1]['actual_graph_nodes']=nodes
        verify(run,tool)
        result=dict(status='passed',execution_environment=_verify_environment(registration,tool),target=target,kind=row['kind'],tile_count=tiles,tool=tool,
            physical_resources=physical,physical_params=abi,cubin_sha256=compiled['cubin_sha256'],
            source_digest=registration['source']['source_digest'],records=records,input_records=input_records,
            no_kernel_timing=True,scope='independent tiles' if tiled else 'single-tile changed-emission regression')
        write(directory/output_name,result)
        return dict(path=str(directory),records=len(records),tiles=tiles,status='passed')
    finally:
        torch.cuda.synchronize();cuda.cuModuleUnload(module)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','compile','execute'))
    parser.add_argument('run',type=Path);parser.add_argument('--prior',type=Path);parser.add_argument('--directory',type=Path)
    parser.add_argument('--tool',choices=TOOLS);args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.run,args.prior)
    elif args.command=='compile':result=compile_all(args.run)
    elif args.directory:result=execute_one(args.run,args.directory,args.tool)
    else:
        result=[]
        for row in read(args.run/'manifest.json')['rows']:result.append(execute_one(args.run,Path(row['path']),args.tool))
    print(json.dumps(result,allow_nan=False),flush=True)

if __name__=='__main__':main()
