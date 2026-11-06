"""Frozen-cubin paired measurement. Calibration and setup are never final samples."""
import ctypes,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np
ARMS=('L-C','L-O','J-C','J-O')
SEED=20261030
UUID='GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_tool_hashes(expected):
 if not expected:raise ValueError('empty tool manifest')
 for path,value in expected.items():
  if not Path(path).is_file() or sha(path)!=value:raise ValueError('frozen tool changed: '+path)

def analyze_paired_batches(records):
 groups={};invalid=[];valid=[]
 for row in records:
  b=row['batch'];arm=row['arm']
  content=row['pair_content'];computed=hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  if computed!=row['pair_id'] or content['batch']!=b or any(content[k]!=row[k] for k in ['input_offset','output_offset']):raise ValueError('raw pair content/hash mismatch')
  count=row['launch_count']
  if not isinstance(count,int) or count<1 or not row['samples_ms'] or len(row['samples_ms'])!=len(row['graph_elapsed_ms']) or not np.allclose(row['samples_ms'],np.asarray(row['graph_elapsed_ms'])/count,rtol=1e-12,atol=1e-15):raise ValueError('raw elapsed/count equation mismatch')
  if b not in range(10) or arm not in ARMS:raise ValueError('unexpected batch/arm')
  if arm in groups.setdefault(b,{}):raise ValueError('duplicate pair member')
  groups[b][arm]=row
 for b in range(10):
  group=groups.get(b,{})
  if len({x['launch_count'] for x in group.values()})>1:raise ValueError('unequal arm launch counts')
  if len({x['pair_id'] for x in group.values()})>1:raise ValueError('pair identity mismatch')
  if set(group)!=set(ARMS) or not all(x.get('valid',False) for x in group.values()):invalid.append(dict(batch=b,records=list(group.values())));continue
  values=[float(np.median(group[arm]['samples_ms'])) for arm in ARMS]
  if not all(np.isfinite(x) and x>0 for x in values):raise ValueError('invalid elapsed')
  valid.append(values)
 result=dict(n=len(valid),formal=len(valid)==10,invalid_batches=invalid,comparisons={},arms={},estimator='median of paired batch differences and baseline/target ratios',seed=SEED,resamples=10000)
 if not valid:return result
 matrix=np.array(valid);indices=np.random.default_rng(SEED).integers(0,len(valid),(10000,len(valid)))
 for i,arm in enumerate(ARMS):result['arms'][arm]=dict(median_ms=float(np.median(matrix[:,i])),p10_ms=float(np.percentile(matrix[:,i],10)),p90_ms=float(np.percentile(matrix[:,i],90)),batch_medians_ms=matrix[:,i].tolist())
 for i,j in [(0,1),(0,2),(0,3),(1,3),(2,3)]:
  delta=matrix[:,i]-matrix[:,j];ratio=matrix[:,i]/matrix[:,j]
  result['comparisons'][ARMS[i]+'/'+ARMS[j]]=dict(difference_ms=float(np.median(delta)),ratio=float(np.median(ratio)),difference_ms_ci95=np.percentile(np.median(delta[indices],axis=1),[2.5,97.5]).tolist(),ratio_ci95=np.percentile(np.median(ratio[indices],axis=1),[2.5,97.5]).tolist())
 return result

def gpu_state():
 def cmd(args):
  p=subprocess.run(args,capture_output=True,text=True);return dict(command=args,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
 return dict(time_ns=time.time_ns(),device=cmd(['nvidia-smi','-i',UUID,'--query-gpu=uuid,driver_version,pstate,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits']),processes=cmd(['nvidia-smi','-i',UUID,'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']))
def foreign_pids(state):
 assert all(state[k]['exit_code']==0 and state[k]['stdout'].strip() and not state[k]['stderr'].strip() for k in ['device']), 'invalid GPU device query'
 assert state['processes']['exit_code']==0 and state['processes']['stdout'].strip() and not state['processes']['stderr'].strip(),'invalid GPU process query'
 assert UUID in state['device']['stdout'],'missing target GPU identity'
 text=state['processes']['stdout']
 assert all(line.split(',')[0].strip().isdigit() for line in text.splitlines() if line.strip()),'invalid GPU PID response'
 return [int(line.split(',')[0]) for line in text.splitlines() if line.split(',')[0].strip().isdigit() and int(line.split(',')[0])!=os.getpid()]
def resolve(root,p):
 p=Path(p);return p if p.is_absolute() else root/p

def measured_tensormaps(info,pointers):
 import torch
 from types import SimpleNamespace
 maps=SimpleNamespace(host=[],device=[],setup_ns=[],upload_ms=[])
 cuda=ctypes.CDLL('libcuda.so.1');fn=cuda.cuTensorMapEncodeTiled
 fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_uint32,ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint64),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_uint32),ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int];fn.restype=ctypes.c_int
 for p in info['tma_plans']:
  start=time.perf_counter_ns();d=p['descriptor'];raw=ctypes.create_string_buffer(191);addr=(ctypes.addressof(raw)+63)&~63
  dims=(ctypes.c_uint64*2)(*d['box']);strides=(ctypes.c_uint64*1)(*d['global_strides']);box=(ctypes.c_uint32*2)(*d['box']);steps=(ctypes.c_uint32*2)(1,1)
  ptr=pointers[info['root_order'].index(p['source_root'])]+d['source_offset'];status=fn(addr,6,2,ptr,dims,strides,box,steps,0,{0:0,32:1}[d['swizzle_bytes']],0,0)
  if status:raise ValueError('descriptor encoding '+str(status))
  payload=np.frombuffer(ctypes.string_at(addr,128),dtype=np.uint8).copy();host=torch.from_numpy(payload).pin_memory();device=torch.empty(128,dtype=torch.uint8,device='cuda');maps.setup_ns.append(time.perf_counter_ns()-start)
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record();device.copy_(host,non_blocking=True);end.record();end.synchronize();maps.upload_ms.append(begin.elapsed_time(end))
  assert device.data_ptr()%64==0;maps.host.append((raw,host));maps.device.append(device)
 return maps

def measure(artifact_dir,batches=10,seed=SEED):
 import torch
 import run_joint_ablation as runner
 from async_host import TensorMaps,validate_launch,physical_abi
 assert batches==10 and seed==SEED,'fixed planned protocol'
 root=Path(__file__).resolve().parents[3];base=resolve(root,artifact_dir);manifest=json.loads((base/'manifest.json').read_text());assert manifest['stage']=='validated'
 assert json.loads((root/'guoqiao/optimize_plan/joint_conversion_ablation_artifacts/task-4-host-parity.json').read_text())['passed']==64
 out=root/'guoqiao/optimize_plan/joint_conversion_ablation_artifacts'/('run-task4-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));out.mkdir()
 # The generation manifest is immutable; measurement source is signed separately.
 snapshots=['task-1-source-snapshot.json','task-2-source-snapshot.json','task-3-source-snapshot.json','task-3-fixwave-source-snapshot.json','task-3-protocolguard-source-snapshot.json']
 accepted={}
 for name in snapshots:accepted.update(json.loads((out.parent/name).read_text()))
 changed=[name for name,content in accepted.items() if (root/name).read_text()!=content]
 assert set(changed)<=set(['benchmark/layout/execution/run_joint_ablation.py','benchmark/layout/selection/JointAblation.cpp','benchmark/layout/selection/selection_report.py']),'generation code changed: '+str(changed)
 tools=json.loads((out.parent/'task-4-current-tool-freeze.json').read_text());validate_tool_hashes(tools['current_hashes'])
 approved=json.loads((out.parent/'task-4-source-freeze.json').read_text())
 assert all(sha(root/name)==value for name,value in approved['file_hashes'].items()),'measurement source differs from reviewed freeze'
 identity=runner.source_identity(); frozen=dict(schema_version=1,base_manifest=str((base/'manifest.json').relative_to(root)),base_manifest_sha256=sha(base/'manifest.json'),generation_identity=manifest['identity'],measurement_identity=identity,measurement_only_delta=changed,accepted_snapshot_files=snapshots,reviewed_measurement_source=approved,current_tool_identity=tools,host_parity_sha256=sha(out.parent/'task-4-host-parity.json'),protocol=dict(batches=10,seed=seed,warmup=100,target_ms=200,replays=10,ordinary_samples=20,foreign_load_rule='any foreign compute PID on target UUID before/after batch invalidates the entire paired batch; failures retained, never replaced'))
 (out/'freeze.json').write_text(json.dumps(frozen,indent=2))
 own_processes=subprocess.run(['ps','-u',str(os.getuid()),'-o','pid,args'],capture_output=True,text=True).stdout
 heavy=[line for line in own_processes.splitlines() if any(x in line for x in ['FriskLayoutUnitTests','FriskLayoutSelectionTests','cmake --build','ctest --output','frisk_joint_ablation_bench','frisk_layout_selection_bench','ninja','task-4-cold-run.py','task-4-host-parity-run.py'])]
 assert not heavy,'active build/oracle/regression: '+str(heavy)
 (out/'exclusive-process-observation.txt').write_text(own_processes)
 torch.cuda.set_device(1);torch.empty(1,device='cuda');assert UUID in gpu_state()['device']['stdout'];assert not foreign_pids(gpu_state())
 rows={}
 for row in manifest['matrix']:
  assert row['status']=='success' and row.get('safety'),'numeric+safety prerequisite'
  rows.setdefault((row['case_id'],row['abi_stratum']),{})[row['arm']]=row
 result=[];rng=np.random.default_rng(seed)
 for (case,stratum),arms in rows.items():
  assert set(arms)==set(ARMS)
  entries={};calibration=[];setup=[]
  # Allocate every planned phase and descriptors before calibration/formal timing.
  for arm,row in arms.items():
   directory=resolve(root,row['path']);assert sha(directory/'kernel.cubin')==row['cubin_sha256']
   for name,h in row['artifact_hashes'].items():
    assert sha(directory/name)==h,'validated artifact changed: '+name
   t=time.perf_counter_ns();cuda,module,function,binary=runner.load_exact(directory);load_ns=time.perf_counter_ns()-t
   device=ctypes.c_int();assert cuda.cuCtxGetDevice(ctypes.byref(device))==0;raw_uuid=(ctypes.c_ubyte*16)();assert cuda.cuDeviceGetUuid(ctypes.byref(raw_uuid),device)==0
   import uuid
   assert 'GPU-'+str(uuid.UUID(bytes=bytes(raw_uuid)))==UUID,'actual CUDA selected device UUID mismatch'
   info=json.loads((directory/'execution_config.json').read_text());physical=physical_abi((directory/'kernel.ptx').read_text(),len(info['tma_plans']))
   phases=[]
   for b in range(10):
    if case=='attention':name,q,k,v=runner.attention_inputs()[b];ref=runner.attention_reference(q,k,v)
    else:q,k,v=runner.inputs();name='fixed';ref=runner.reference(case,q,k,v)
    io=(1+b%3) if stratum=='weak-vector' else (128*(b%3));oo=1+b%3
    t=time.perf_counter_ns();buffers=[]
    for x in (q,k,v):
     buf=torch.full((x.nbytes+272,),165,device='cuda',dtype=torch.uint8);buf[io:io+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(buf)
    output=torch.full((ref.nbytes+272,),165,device='cuda',dtype=torch.uint8);pointers=[x.data_ptr()+io for x in buffers]+[output.data_ptr()+oo];validate_launch(info,pointers)
    maps=measured_tensormaps(info,pointers) if info['tma_plans'] else None;torch.cuda.synchronize();setup_ns=time.perf_counter_ns()-t
    if maps:
     accepted_maps=TensorMaps(info,pointers);torch.cuda.synchronize()
     assert all(torch.equal(x,y) for x,y in zip(maps.device,accepted_maps.device)),'descriptor bytes differ from validated host encoder'
    args=[ctypes.c_uint64(x) for x in pointers+([x.data_ptr() for x in maps.device] if maps else [])+[0]];assert len(args)==len(physical)
    params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
    def launch(cuda=cuda,function=function,params=params):
     status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None)
     if status:raise RuntimeError('cuLaunchKernel '+str(status))
    pair_id=hashlib.sha256(json.dumps(dict(case=case,stratum=stratum,batch=b,input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)],input_offset=io,output_offset=oo),sort_keys=True).encode()).hexdigest()
    pair_content=dict(case_id=case,abi_stratum=stratum,batch=b,input_hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in (q,k,v)],reference_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),input_offset=io,output_offset=oo)
    pair_id=hashlib.sha256(json.dumps(pair_content,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    phases.append(dict(saved_inputs=[x.cpu().numpy().copy() for x in buffers],pair_content=pair_content,artifact_ids=dict(prepared_digest=row['prepared_digest'],assignment_sha256=row['complete_assignment_sha256'],config_sha256=sha(directory/'execution_config.json'),physical_abi=physical),launch=launch,buffers=buffers,output=output,maps=maps,args=args,params=params,ref=ref,pair_id=pair_id,input_offset=io,output_offset=oo))
    setup.append(dict(arm=arm,batch=b,load_ns=load_ns,allocation_descriptor_upload_sync_ns=setup_ns,descriptor_encode_pinned_allocation_ns=maps.setup_ns if maps else [],descriptor_upload_ms=maps.upload_ms if maps else []))
   entries[arm]=dict(phases=phases,module=module,binary=binary,cuda=cuda,function=function)
  (out/(case+'-'+stratum+'-setup.json')).write_text(json.dumps(setup,indent=2))
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  def event_samples(fn,count=20):
   values=[]
   for _ in range(count):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();fn();end.record();end.synchronize();values.append(start.elapsed_time(end))
   return values
  preflight=[]
  def check_phase(p):
   stream.synchronize();raw=p['output'].cpu().numpy();oo=p['output_offset'];ref=p['ref'];got=raw[oo:oo+ref.nbytes].copy().view(ref.dtype).reshape(ref.shape)
   assert np.all(np.isfinite(got))
   assert np.all(np.abs(got-ref)<=.002+.002*np.abs(ref)) if case=='attention' else np.array_equal(got,ref)
   assert np.all(raw[:oo]==165) and np.all(raw[oo+ref.nbytes:]==165)
   assert all(np.array_equal(x.cpu().numpy(),y) for x,y in zip(p['buffers'],p['saved_inputs'])),'QKV bytes/guards changed'
  with torch.cuda.stream(stream):
   for arm in ARMS:
    for b,p in enumerate(entries[arm]['phases']):
     p['output'].fill_(165);p['launch']();check_phase(p)
     graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph,stream=stream):p['launch']()
     p['output'].fill_(165);graph.replay();check_phase(p);preflight.append(dict(arm=arm,batch=b,pair_id=p['pair_id'],ordinary=True,capture=True,all_input_bytes_and_output_guards=True))
  (out/(case+'-'+stratum+'-preflight.json')).write_text(json.dumps(preflight,indent=2))
  # Correctness of nondefault stream and capture precedes all formal timing.
  with torch.cuda.stream(stream):
   for arm in ARMS:
    p=entries[arm]['phases'][0];launch=p['launch'];launch();stream.synchronize()
    graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
    with torch.cuda.graph(graph,stream=stream):launch()
    dot=out/(case+'-'+stratum+'-'+arm+'-capture.dot');graph.debug_dump(str(dot));assert dot.read_text().count('frisk_execution')==1
    p['output'].fill_(165);graph.replay();stream.synchronize();raw=p['output'].cpu().numpy();oo=p['output_offset'];ref=p['ref'];got=raw[oo:oo+ref.nbytes].copy().view(ref.dtype).reshape(ref.shape)
    assert np.all(np.abs(got-ref)<=.002+.002*np.abs(ref)) if case=='attention' else np.array_equal(got,ref)
    assert np.all(raw[:oo]==165) and np.all(raw[oo+ref.nbytes:]==165)
    for _ in range(100):launch()
    stream.synchronize();pilot=torch.cuda.CUDAGraph()
    with torch.cuda.graph(pilot,stream=stream):
     for _ in range(100):launch()
    stream.synchronize();pilot_elapsed=event_samples(pilot.replay,5);times=[x/100 for x in pilot_elapsed];calibration.append(dict(arm=arm,samples_ms=times,pilot_launch_count=100,pilot_graph_elapsed_ms=pilot_elapsed))
  # Shared count targets 200ms for the slowest ordinary estimate.
  count=max(1,int(200/max(np.median(x['samples_ms']) for x in calibration)))
  (out/(case+'-'+stratum+'-calibration.json')).write_text(json.dumps(dict(launch_count=count,records=calibration),indent=2))
  records=[]
  for b in range(10):
   pre=gpu_state();order=rng.permutation(ARMS).tolist();batch=[]
   with torch.cuda.stream(stream):
    for arm in order:
     p=entries[arm]['phases'][b];launch=p['launch']
     for _ in range(100):launch()
     stream.synchronize();ordinary=event_samples(launch)
     t=time.perf_counter_ns();graph=torch.cuda.CUDAGraph(keep_graph=True)
     with torch.cuda.graph(graph,stream=stream):
      for _ in range(count):launch()
     stream.synchronize();capture_ns=time.perf_counter_ns()-t
     cuda=entries[arm]['cuda'];cuda.cuGraphGetNodes.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)];node_count=ctypes.c_size_t();assert cuda.cuGraphGetNodes(ctypes.c_void_p(graph.raw_cuda_graph()),None,ctypes.byref(node_count))==0;assert node_count.value==count
     t=time.perf_counter_ns();graph.instantiate();instantiate_ns=time.perf_counter_ns()-t
     elapsed=event_samples(graph.replay,10)
     batch.append(dict(batch=b,arm=arm,pair_id=p['pair_id'],pair_content=p['pair_content'],artifact_ids=p['artifact_ids'],input_offset=p['input_offset'],output_offset=p['output_offset'],order=order,valid=True,launch_count=count,graph_elapsed_ms=elapsed,samples_ms=[x/count for x in elapsed],ordinary_ms=ordinary,capture_sync_ns=capture_ns,instantiate_ns=instantiate_ns,actual_graph_nodes=node_count.value,cubin_sha256=sha(resolve(root,arms[arm]['path'])/'kernel.cubin')))
   post=gpu_state();foreign=foreign_pids(pre)+foreign_pids(post)
   for record in batch:record.update(valid=not foreign,foreign_pids=foreign,state_before=pre,state_after=post)
   records.extend(batch);(out/(case+'-'+stratum+'-raw.json')).write_text(json.dumps(records,indent=2))
   print(json.dumps(dict(case=case,stratum=stratum,batch=b,foreign=foreign)),flush=True)
  analysis=analyze_paired_batches(records);result.append(dict(case_id=case,abi_stratum=stratum,analysis=analysis));(out/'analysis.json').write_text(json.dumps(result,indent=2))
 (out/'final-identity.json').write_text(json.dumps(runner.source_identity(),indent=2));assert runner.source_identity()==identity;validate_tool_hashes(tools['current_hashes'])
 print(str(out));return out
