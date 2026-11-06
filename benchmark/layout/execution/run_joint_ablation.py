"""Gate2 exact-cubin compilation, ordinary/current-stream/capture checks.
Every invocation requires a fresh artifact directory. No performance claims.
"""
import argparse,ctypes,hashlib,json,subprocess,sys,os,uuid,time
from async_host import physical_abi,validate_launch
from pathlib import Path
import numpy as np
from joint_fixture import inputs,reference

def digest(b):return hashlib.sha256(b).hexdigest()
def load_exact(directory):
 import torch
 binary=(directory/'kernel.cubin').read_bytes();cuda=ctypes.CDLL('libcuda.so.1')
 cuda.cuLaunchKernel.argtypes=[ctypes.c_void_p]+[ctypes.c_uint]*7+[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
 cuda.cuModuleLoadData.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
 cuda.cuModuleGetFunction.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p,ctypes.c_char_p]
 module=ctypes.c_void_p();function=ctypes.c_void_p()
 assert cuda.cuModuleLoadData(ctypes.byref(module),ctypes.c_char_p(binary))==0
 assert cuda.cuModuleGetFunction(ctypes.byref(function),module,b'frisk_execution')==0
 # module/binary/functions/params and all IO stay alive until synchronization.
 return cuda,module,function,binary

def execute(directory,case_id,transport=False,reload=False):
 import torch
 torch.cuda.set_device(1);torch.empty(1,device='cuda');cuda,module,function,binary=load_exact(directory)
 device=ctypes.c_int();assert cuda.cuCtxGetDevice(ctypes.byref(device))==0
 raw_uuid=(ctypes.c_ubyte*16)();assert cuda.cuDeviceGetUuid(ctypes.byref(raw_uuid),device)==0
 actual_uuid='GPU-'+str(uuid.UUID(bytes=bytes(raw_uuid)));assert actual_uuid=='GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618',actual_uuid
 info=json.loads((directory/'execution_config.json').read_text());physical=physical_abi((directory/'kernel.ptx').read_text(),len(info['tma_plans']))
 assert len(physical)==5 and info['copy_path']=='vector' and not info['tma_plans'],'only weak f32 stratum'
 resources={}
 for name,attribute in [('shared_bytes',1),('local_bytes',3),('registers_per_thread',4)]:
  value=ctypes.c_int();assert cuda.cuFuncGetAttribute(ctypes.byref(value),attribute,function)==0;resources[name]=value.value
 q,k,v=inputs();ref=q if transport else reference(case_id,q,k,v);records=[]
 for offset in (1,2,3):
  buffers=[]
  for x in (q,k,v):
   b=torch.full((x.nbytes+32,),165,dtype=torch.uint8,device='cuda');b[offset:offset+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8)));buffers.append(b)
  output=torch.full((ref.nbytes+32,),165,dtype=torch.uint8,device='cuda');saved=[b.cpu().numpy().copy() for b in buffers]
  pointers=[b.data_ptr()+offset for b in buffers+[output]];validate_launch(info,pointers)
  args=[ctypes.c_uint64(pointer) for pointer in pointers]+[ctypes.c_uint64(0)];assert len(args)==len(physical)
  params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args])
  def launch():
   stream=torch.cuda.current_stream().cuda_stream
   status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(stream),params,None)
   assert status==0,status
  def check():
   torch.cuda.synchronize();raw=output.cpu().numpy()
   assert np.array_equal(raw[offset:offset+ref.nbytes],ref.view(np.uint8)),'bitexact failure'
   assert np.all(raw[:offset]==165) and np.all(raw[offset+ref.nbytes:]==165),'output canary'
   assert all(np.array_equal(b.cpu().numpy(),before) for b,before in zip(buffers,saved)),'input/canary changed'
  for repeat in range(3):
   output.fill_(165);launch();check();records.append(dict(offset=offset,repeat=repeat,path='ordinary',bitexact=True,guards=True,inputs_unchanged=True))
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):output.fill_(165);launch()
  check();records.append(dict(offset=offset,path='nondefault-stream',stream=stream.cuda_stream,bitexact=True,guards=True,inputs_unchanged=True))
  graph=torch.cuda.CUDAGraph(keep_graph=True);graph.enable_debug_mode()
  with torch.cuda.graph(graph,stream=stream):
   for _ in range(3):launch()
  dot=directory/(('reload-' if reload else '')+'capture-'+str(offset)+'.dot');graph.debug_dump(str(dot));dottext=dot.read_text()
  # CUDA graph DOT emits one function-name label per actual kernel node.
  assert dottext.count('frisk_execution')==3, dottext
  output.fill_(165);graph.replay();check();records.append(dict(offset=offset,path='capture',target_kernel_nodes=3,stream=stream.cuda_stream,bitexact=True,guards=True,inputs_unchanged=True))
 result=dict(physical_resources=resources,gpu_uuid=actual_uuid,physical_params=physical,cubin_sha256=digest(binary),case_id=case_id,input_hashes=[digest(x.tobytes()) for x in (q,k,v)],expected_sha256=digest(ref.tobytes()),records=records)
 (directory/('reload-correctness.json' if reload else 'correctness.json')).write_text(json.dumps(result,indent=2));return result

def forced_main():
 parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--case',required=True);parser.add_argument('--load-only',action='store_true');parser.add_argument('--transport',action='store_true');a=parser.parse_args();d=a.directory
 if not a.load_only:
  for name in ('kernel.ttgir','kernel.ptx','kernel.cubin','execution_config.json','correctness.json'):
   if (d/name).exists():raise FileExistsError('fresh compile refuses overwrite: '+str(d/name))
  import triton
  from triton.backends.compiler import GPUTarget
  from backend import emit
  tt,info=emit(json.loads((d/'execution.json').read_text()));(d/'kernel.ttgir').write_text(tt)
  kernel=triton.compile(str(d/'kernel.ttgir'),target=GPUTarget('cuda',90,32),options={'enable_fp_fusion':False})
  root=Path(__file__).resolve().parents[3]
  sources={str(p.relative_to(root)):digest(p.read_bytes()) for folder in ('include','lib','benchmark') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.td','.py','.txt')}
  import torch
  from triton.backends.nvidia.compiler import get_ptxas
  identity=dict(source_manifest=sources,source_digest=digest(json.dumps(sources,sort_keys=True).encode()),execution_sha256=digest((d/'execution.json').read_bytes()),torch=torch.__version__,triton=triton.__version__,python=sys.version,ptxas=str(get_ptxas().path),fp_fusion=False,TRITON_ALWAYS_COMPILE=os.environ.get('TRITON_ALWAYS_COMPILE'))
  (d/'identity.json').write_text(json.dumps(identity,indent=2))
  (d/'kernel.cubin').write_bytes(kernel.asm['cubin']);(d/'kernel.ptx').write_text(kernel.asm['ptx']);(d/'execution_config.json').write_text(json.dumps(info,indent=2))
 else:
  accepted=json.loads((d/'correctness.json').read_text());assert digest((d/'kernel.cubin').read_bytes())==accepted['cubin_sha256'],'fixed-cubin identity changed'
 print(json.dumps(execute(d,a.case,a.transport,a.load_only)))


def executable_ptx_fingerprint(text):
 import re
 text=re.sub(r'/\*.*?\*/','',text,flags=re.S)
 lines=[]
 for line in text.splitlines():
  line=line.split('//',1)[0].strip()
  if line and not line.startswith(('.file ','.loc ')):lines.append(line)
 # Version 2 retains executable entry and all declarations/directives except
 # debug source locations. This deliberately does not rewrite raw artifacts.
 return digest(('\n'.join(lines)+'\n').encode())

def execution_evidence(directory):
 result={}
 if (directory/'kernel.ptx').exists():result.update(executable_ptx_fingerprint_version=2,normalized_executable_ptx_sha256=executable_ptx_fingerprint((directory/'kernel.ptx').read_text()),cubin_sha256=digest((directory/'kernel.cubin').read_bytes()))
 if (directory/'correctness.json').exists():result['correctness']=json.loads((directory/'correctness.json').read_text())
 for name in ('materialized','output'):
  if (directory/(name+'.mlir')).exists():
   text=(directory/(name+'.mlir')).read_text();result[name+'_operation_counts']={op:text.count(op) for op in ('frisk.convert_layout','arith.mulf','arith.subf','frisk.mma','frisk.copy')}
 return result

def source_identity():
 root=Path(__file__).resolve().parents[3]
 files={str(p.relative_to(root)):digest(p.read_bytes()) for folder in ('include','lib','benchmark','unittests') for p in (root/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.td','.py','.txt')}
 return dict(source_manifest=files,backend_digest=digest(json.dumps(files,sort_keys=True,separators=(',',':')).encode()))
def artifact_hashes(directory):
 return {str(p.relative_to(directory)):digest(p.read_bytes()) for p in directory.rglob('*') if p.is_file()}
class FatalPhaseError(RuntimeError):
 def __init__(self,result):
  self.result=result;super().__init__(json.dumps(result,sort_keys=True))

DOCUMENTED={'no-feasible','unknown','search-budget','unsupported','backend-unsupported','timeout'}
def success_schema(result,phase):
 import math
 def check(ok,reason):
  if not ok:raise ValueError(reason)
 def integer(value):return type(value) is int and value>=0
 def sha(value):return type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value)
 def text(value):return type(value) is str and bool(value)
 def id_list(values):return type(values) is list and all(integer(x) for x in values) and len(set(values))==len(values)
 def candidate(value):return type(value) is dict and sha(value.get('id')) and text(value.get('attribute'))
 check(type(result) is dict,'result object required')
 if phase=='prepare':
  for key in ('common_digest','domain_digest','coverage_digest','package_digest','input_sha256','domain_file_sha256'):check(sha(result.get(key)),'missing prepared identity '+key)
  for key in ('case','cost_model','candidate_preparation'):check(text(result.get(key)),'missing prepared metadata '+key)
  check(result.get('frequency_mode') in ('legacy','static-trip-count'),'frequency enum');check(id_list(result.get('producers')) and id_list(result.get('producer_instructions')),'scope IDs')
  check(type(result.get('candidate_proofs')) is list and bool(result['candidate_proofs']),'candidate proof coverage')
  for proof in result['candidate_proofs']:check(type(proof) is dict and integer(proof.get('variable')) and text(proof.get('attribute')) and type(proof.get('valid')) is bool and type(proof.get('capacity')) is bool,'candidate proof types')
 elif phase in ('matrix','replay'):
  check(result.get('status')=='success','successful selection status required')
  for key in ('prepared_digest','common_domain_digest','complete_assignment_sha256'):check(sha(result.get(key)),'missing selection identity '+key)
  check(type(result.get('assignment')) is list and bool(result['assignment']),'complete assignment required');seen=set()
  for row in result['assignment']:
   check(type(row) is dict and integer(row.get('variable')) and text(row.get('name')) and candidate(row.get('candidate')),'assignment entry types');check(row['variable'] not in seen,'duplicate assignment variable');seen.add(row['variable'])
  for kind in ('instruction_bindings','reduction_bindings'):
   check(type(result.get(kind)) is list,'binding array required');seen=set()
   for row in result[kind]:
    check(type(row) is dict and integer(row.get('constraint')) and text(row.get('binding')),'binding entry types');check(row['constraint'] not in seen,'duplicate binding');seen.add(row['constraint'])
  complete={key:result[key] for key in ('assignment','instruction_bindings','reduction_bindings')};check(digest(json.dumps(complete,indent=2,sort_keys=True).encode())==result['complete_assignment_sha256'],'complete assignment hash mismatch')
  cost=result.get('cost');check(type(cost) is dict and type(cost.get('cost')) is list and len(cost['cost'])==9 and all(integer(x) for x in cost['cost']) and type(cost.get('saturated')) is bool,'lex9 cost schema')
  check(result.get('conversion_policy') in ('independent','shared-only','full') and result.get('selection') in ('L','J'),'selection/policy enums');check(type(result.get('post_opt')) is bool and type(result.get('selection_bypassed')) is bool,'selection flags');check(text(result.get('scope')) and text(result.get('cost_model')) and result.get('frequency_mode') in ('legacy','static-trip-count'),'scope/model schema')
  check(result.get('selection_preserved_input_ir') is True and result.get('instruction_contracts_verified') is True,'IR/contract verification required');check(type(result.get('conversions')) is list,'conversion array required');check(type(result.get('failure_kind')) is str,'failure kind required')
 elif phase=='numerics':
  check(sha(result.get('cubin_sha256')) and text(result.get('gpu_uuid')) and result['gpu_uuid'].startswith('GPU-'),'GPU/cubin identity required');check(type(result.get('physical_params')) is list and len(result['physical_params']) in (5,8) and all(text(x) for x in result['physical_params']),'physical parameter ABI')
  resources=result.get('physical_resources');check(type(resources) is dict and all(integer(resources.get(key)) for key in ('shared_bytes','local_bytes','registers_per_thread')),'physical resource schema');check(type(result.get('records')) is list and bool(result['records']),'nonempty numerical records required')
  attention='copy_path' in result
  if attention:
   check(result['copy_path'] in ('vector','cp.async','tma') and integer(result.get('tma_descriptor_count')) and type(result.get('safety_subset')) is bool,'Attention ABI metadata');check(type(result.get('inputs')) is list and bool(result['inputs']),'Attention input identities')
   for row in result['inputs']:check(type(row) is dict and text(row.get('case')) and integer(row.get('input_offset')) and integer(row.get('output_offset')) and type(row.get('input_hashes')) is list and len(row['input_hashes'])==3 and all(sha(x) for x in row['input_hashes']) and sha(row.get('expected_sha256')),'Attention input identity schema')
  else:check(text(result.get('case_id')) and type(result.get('input_hashes')) is list and len(result['input_hashes'])==3 and all(sha(x) for x in result['input_hashes']) and sha(result.get('expected_sha256')),'f32 input/reference identities')
  for row in result['records']:
   check(type(row) is dict and row.get('guards') is True and row.get('inputs_unchanged') is True,'guard/input success must be explicit true booleans')
   if attention:check(row.get('numerics') is True and text(row.get('case')) and integer(row.get('input_offset')) and integer(row.get('output_offset')) and type(row.get('max_abs_error')) in (int,float) and math.isfinite(row['max_abs_error']) and row['max_abs_error']>=0,'Attention numerical success schema')
   else:check(row.get('bitexact') is True and integer(row.get('offset')) and row.get('path') in ('ordinary','nondefault-stream','capture'),'f32 bitexact success schema')
 else:raise ValueError('unknown success role')

def phase_outcome(directory,code,phase,result_name,command_metadata=None):
 directory=Path(directory)
 def fatal(reason):raise FatalPhaseError(dict(status='failed',fatal=True,failure_kind='fatal-'+phase,reason=reason,exit_code=code,path=str(directory)))
 failure=None
 if (directory/'failure.json').exists():
  try:failure=json.loads((directory/'failure.json').read_text())
  except (ValueError,OSError):fatal('malformed typed failure JSON')
  if not isinstance(failure,dict) or type(failure.get('fatal')) is not bool or failure.get('status')!='failed':fatal('malformed failure schema')
  if failure['fatal']:fatal(failure.get('reason','worker integrity failure'))
 if code==124 and type(command_metadata) is dict and command_metadata.get('timeout_expired') is True and command_metadata.get('exit_code')==124:
  if phase in ('prepare','matrix','replay'):return dict(status='failed',fatal=False,failure_kind='timeout',reason='parent subprocess timeout',exit_code=124,timeout_seconds=command_metadata.get('timeout_seconds'),timeout_origin='parent-TimeoutExpired')
  fatal('validation worker exceeded parent timeout')
 if code!=0:
  if code==3 and failure and failure.get('failure_kind') in DOCUMENTED:return dict(failure,exit_code=code)
  fatal('signal or undocumented exit' if code!=3 else 'documented exit missing valid typed outcome')
 if failure:fatal('success exit contradicts typed failure')
 try:result=json.loads((directory/result_name).read_text())
 except (ValueError,OSError):fatal('missing or malformed worker result')
 try:success_schema(result,phase)
 except Exception as error:fatal('worker success schema: '+str(error))
 if phase=='numerics' and (not (directory/'kernel.cubin').exists() or digest((directory/'kernel.cubin').read_bytes())!=result['cubin_sha256']):fatal('numerical evidence cubin mismatch')
 return dict(result,exit_code=0)

def command(args,log,timeout_seconds=600):
 args=[str(x) for x in args];start=time.perf_counter_ns();code=124;output='';timeout_expired=False
 try:
  result=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout_seconds);code=result.returncode;output=result.stdout
 except subprocess.TimeoutExpired as error:
  timeout_expired=True;output=str(error.stdout or '')+'\nTIMEOUT'
 elapsed=time.perf_counter_ns()-start;log.write_text(output);log.with_suffix(log.suffix+'.command.json').write_text(json.dumps(dict(argv=args,exit_code=code,elapsed_ns=elapsed,timeout_expired=timeout_expired,timeout_seconds=timeout_seconds,timing_kind='observed command wall duration; cache policy recorded separately'),indent=2));return code

def fatal_manifest(run,manifest,error):
 manifest.update(stage='fatal',failure=error.result);(run/'manifest.json').write_text(json.dumps(manifest,indent=2))

def phase_or_abort(run,manifest,directory,code,phase,result_name,command_metadata=None):
 try:return phase_outcome(directory,code,phase,result_name,command_metadata=command_metadata)
 except Exception as error:
  failure=error if isinstance(error,FatalPhaseError) else FatalPhaseError(dict(status='failed',fatal=True,failure_kind='fatal-'+phase,reason=type(error).__name__+': '+str(error),exit_code=code,path=str(directory)))
  fatal_manifest(run,manifest,failure);raise failure

def prepare(artifact_dir):
 from joint_fixture import cases,fixture
 from datetime import datetime,timezone
 root=Path(__file__).resolve().parents[3];run=artifact_dir/('run-task3-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]);run.mkdir(parents=True)
 bench=root/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench';rows=[];manifest=dict(schema_version=2,stage='preparing',run=str(run),identity=source_identity(),rows=rows)
 try:
  for case in cases()+[dict(id='attention',abi_stratum=x) for x in ('weak-vector','strong-cpasync','strong-tma')]:
   case_id=case['id'];stratum=case.get('abi_stratum','weak-vector');key=case_id if case_id!='attention' else case_id+'-'+stratum;d=run/key;d.mkdir();source=d/'raw.mlir'
   if case_id!='attention':text=fixture(case_id)
   else:text=(root/('test/Transforms/attention-layout-'+{'weak-vector':'fixed','strong-cpasync':'cp-async','strong-tma':'tma'}[stratum]+'-abi.mlir')).read_text()
   source.write_text(text);row=dict(case_id=case_id,key=key,abi_stratum=stratum,prepared=str(d/'prepared'),status='running',raw_sha256=digest(source.read_bytes()));rows.append(row)
   code=command([bench,'--prepare',source,case_id,d/'prepared'],d/'prepare.log');row['command_log']=str(d/'prepare.log');row['artifact_hashes']=artifact_hashes(d);outcome=phase_or_abort(run,manifest,d/'prepared',code,'prepare','prepared.json',json.loads((d/'prepare.log.command.json').read_text()));row.update(status='prepared' if code==0 else 'failed',failure_kind=outcome.get('failure_kind',''),exit_code=code,fatal=outcome.get('fatal',False))
 except Exception as error:
  failure=error if isinstance(error,FatalPhaseError) else FatalPhaseError(dict(status="failed",fatal=True,failure_kind="fatal-prepare",reason=type(error).__name__+": "+str(error)))
  row.update(failure.result);fatal_manifest(run,manifest,failure);raise failure
 manifest['stage']='prepared';(run/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(run=str(run),manifest=str(run/'manifest.json'))));return run

def _validate(artifact_dir,bench=None):
 root=Path(__file__).resolve().parents[3]
 if (artifact_dir/'manifest.json').exists():run=artifact_dir
 else:
  candidates=sorted(artifact_dir.glob('run-task3-*/manifest.json'));assert candidates,'prepare must run first';run=candidates[-1].parent
 manifest=json.loads((run/'manifest.json').read_text());assert manifest['identity']['backend_digest']==source_identity()['backend_digest'],'source changed since preparation';assert manifest['stage']=='prepared','fresh validate refuses existing matrix'
 bench=bench or root/'build-m6-release/benchmark/layout/selection/frisk_joint_ablation_bench';rows=[];replays=[];manifest.update(matrix=rows,replays=replays,stage='validating');current=None
 def invoke(args,log,d,phase,result_name):
  code=command(args,log);current['phase_commands']=current.get('phase_commands',[])+[dict(phase=phase,log=str(log),log_sha256=digest(log.read_bytes()),metadata=str(log)+'.command.json',metadata_sha256=digest(log.with_suffix(log.suffix+'.command.json').read_bytes()))];return phase_or_abort(run,manifest,d,code,phase,result_name,json.loads(log.with_suffix(log.suffix+'.command.json').read_text()))
 def numerical(row,d):
  args=[sys.executable,__file__,'--attention-execute',d] if row['case_id']=='attention' else [sys.executable,__file__,d,'--case',row['case_id']]
  invoke(args,d/'compile-correctness.log',d,'numerics','correctness.json');row.update(execution_evidence(d));row['artifact_hashes']=artifact_hashes(d)
 try:
  for case in manifest['rows']:
   if case['status']!='prepared':rows.append(case);continue
   assignments={}
   for selection in ('L','J'):
    for conversion in ('C','O'):
     arm=selection+'-'+conversion;d=run/case['key']/arm;current=dict(case_id=case['case_id'],abi_stratum=case['abi_stratum'],arm=arm,path=str(d),backend_digest=manifest['identity']['backend_digest'],status='running');rows.append(current)
     outcome=invoke([bench,'--prepared='+case['prepared'],'--case='+case['case_id'],'--selection='+selection,'--conversion='+conversion,'--output-dir='+str(d),'--oracle'],d.parent/(arm+'.log'),d,'matrix','record.json');current.update(outcome)
     if outcome['exit_code']==0:
      assignments.setdefault(outcome['complete_assignment_sha256'],d/'record.json');numerical(current,d)
     else:current['artifact_hashes']=artifact_hashes(d)
   for assignment_hash,assignment_path in assignments.items():
    for stage in ('independent','shared','remat','post'):
     d=run/case['key']/('replay-'+assignment_hash[:12]+'-'+stage);current=dict(case_id=case['case_id'],abi_stratum=case['abi_stratum'],stage=stage,path=str(d),backend_digest=manifest['identity']['backend_digest'],status='running');replays.append(current)
     outcome=invoke([bench,'--prepared='+case['prepared'],'--case='+case['case_id'],'--conversion=O','--assignment='+str(assignment_path),'--replay='+stage,'--output-dir='+str(d)],d.parent/(d.name+'.log'),d,'replay','record.json');current.update(outcome)
     if outcome['exit_code']==0:
      if outcome['complete_assignment_sha256']!=assignment_hash:raise FatalPhaseError(dict(status='failed',fatal=True,failure_kind='assignment-corruption',reason='replay assignment changed',path=str(d)))
      numerical(current,d)
     else:current['artifact_hashes']=artifact_hashes(d)
  accepted={}
  for current in rows+replays:
   if current.get('status')!='success':continue
   d=Path(current['path']);binary_sha=digest((d/'kernel.cubin').read_bytes());abi=json.loads((d/'execution_config.json').read_text());key=(binary_sha,json.dumps(abi,sort_keys=True))
   if key in accepted:current['safety']=dict(reused_from=accepted[key],cubin_sha256=binary_sha,exact_bytes_and_abi=True);continue
   checks=[]
   for tool in ('memcheck','racecheck','initcheck','synccheck'):
    launch=[sys.executable,__file__,'--attention-load',d,'--safety'] if current['case_id']=='attention' else [sys.executable,__file__,d,'--case',current['case_id'],'--load-only'];code=command(['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','91']+launch,d/('safety-'+tool+'.log'));checks.append(dict(tool=tool,exit_code=code,cubin_sha256=binary_sha,log_sha256=digest((d/('safety-'+tool+'.log')).read_bytes())))
    if code:raise FatalPhaseError(dict(status='failed',fatal=True,failure_kind='safety',reason='sanitizer worker failed',path=str(d),exit_code=code,safety=checks))
   current['safety']=checks;accepted[key]=str(d);current['artifact_hashes']=artifact_hashes(d)
 except Exception as error:
  failure=error if isinstance(error,FatalPhaseError) else FatalPhaseError(dict(status="failed",fatal=True,failure_kind="fatal-validation",reason=type(error).__name__+": "+str(error)))
  if current is not None:current.update(failure.result);current['artifact_hashes']=artifact_hashes(Path(current['path']))
  fatal_manifest(run,manifest,failure);raise failure
 manifest['stage']='validated';(run/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(run=str(run),rows=len(rows),replays=len(replays))))

def validate(artifact_dir,bench=None):
 run=artifact_dir
 if not (run/'manifest.json').exists():
  candidates=sorted(run.glob('run-task3-*/manifest.json'))
  if candidates:run=candidates[-1].parent
 try:initial=json.loads((run/'manifest.json').read_text())
 except Exception:initial={}
 immutable=type(initial) is dict and initial.get('stage') in ('validated','fatal')
 try:return _validate(run,bench)
 except Exception as error:
  try:manifest=json.loads((run/'manifest.json').read_text())
  except Exception:manifest={}
  if type(manifest) is not dict:manifest={}
  failure=error if isinstance(error,FatalPhaseError) else FatalPhaseError(dict(status='failed',fatal=True,failure_kind='fatal-validation-setup',reason=type(error).__name__+': '+str(error)))
  if immutable:
   attempt=run.parent/(run.name+'-rejected-validation-'+uuid.uuid4().hex[:8]);attempt.mkdir();fatal_manifest(attempt,dict(original_run=str(run)),failure)
  else:fatal_manifest(run,manifest,failure)
  raise failure

def attention_reference(q,k,v):
 q,k,v=[x.astype(np.float32) for x in (q,k,v)];scores=np.zeros((64,64),dtype=np.float32)
 for t in range(16):scores=np.float32(scores+np.float32(q[:,t,None]*k[None,:,t]))
 scores=np.float32(scores*np.float32(.25));e=np.exp(np.float32(scores-scores.max(axis=1,keepdims=True)),dtype=np.float32);prob=np.float32(e/e.sum(axis=1,keepdims=True,dtype=np.float32)).astype(np.float16).astype(np.float32);output=np.zeros((64,16),dtype=np.float32)
 for t in range(64):output=np.float32(output+np.float32(prob[:,t,None]*v[None,t,:]))
 return output

def attention_inputs():
 cases=[]
 for seed in range(10):
  rng=np.random.default_rng(seed);cases.append(('seed'+str(seed),*[rng.uniform(-1,1,(64,16)).astype(np.float16) for _ in range(3)]))
 z=np.zeros((64,16),dtype=np.float16);one=np.ones_like(z);coord=np.linspace(-1,1,1024).reshape(64,16).astype(np.float16);hot=z.copy();hot[13,7]=1;cancel=np.tile(np.array([1,-1],np.float16)[:,None],(32,16))
 return cases+[('zeroQK',z,z,coord),('constant',one,one,one),('signedcoords',coord,-coord,coord),('onehotV',coord,coord,hot),('cancelV',z,z,cancel)]

def compile_attention(directory):
 import triton
 from triton.backends.compiler import GPUTarget
 from backend import emit
 for name in ('kernel.ttgir','kernel.ptx','kernel.cubin','execution_config.json','correctness.json'):
  if (directory/name).exists():raise FileExistsError('fresh compile refuses overwrite: '+str(directory/name))
 tt,info=emit(json.loads((directory/'execution.json').read_text()));(directory/'kernel.ttgir').write_text(tt)
 kernel=triton.compile(str(directory/'kernel.ttgir'),target=GPUTarget('cuda',90,32),options={'num_warps':4,'enable_fp_fusion':False})
 (directory/'kernel.cubin').write_bytes(kernel.asm['cubin']);(directory/'kernel.ptx').write_text(kernel.asm['ptx']);(directory/'execution_config.json').write_text(json.dumps(info,indent=2));(directory/'identity.json').write_text(json.dumps(source_identity(),indent=2))

def attention_offsets(copy_path):
 if copy_path=='vector':return [(x,x) for x in (1,2,3)]
 if copy_path not in ('cp.async','tma'):raise ValueError('unknown copy path')
 return [(q,o) for q in (0,128,256) for o in (1,2,3)]

def execute_attention(directory,load_only=False,safety=False):
 import torch
 from async_host import TensorMaps
 torch.cuda.set_device(1);torch.empty(1,device='cuda')
 if not load_only:compile_attention(directory)
 cuda,module,function,binary=load_exact(directory);info=json.loads((directory/'execution_config.json').read_text());physical=physical_abi((directory/'kernel.ptx').read_text(),len(info['tma_plans']))
 device=ctypes.c_int();assert cuda.cuCtxGetDevice(ctypes.byref(device))==0;raw_uuid=(ctypes.c_ubyte*16)();assert cuda.cuDeviceGetUuid(ctypes.byref(raw_uuid),device)==0;actual_uuid='GPU-'+str(uuid.UUID(bytes=bytes(raw_uuid)));assert actual_uuid=='GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618'
 if load_only:assert digest(binary)==json.loads((directory/'correctness.json').read_text())['cubin_sha256'],'exact cubin changed'
 records=[];host=[];resources={}
 for name,attribute in [('shared_bytes',1),('local_bytes',3),('registers_per_thread',4)]:
  value=ctypes.c_int();assert cuda.cuFuncGetAttribute(ctypes.byref(value),attribute,function)==0;resources[name]=value.value
 cases=attention_inputs()[:1] if safety else attention_inputs()
 for case_name,q,k,v in cases:
  ref=attention_reference(q,k,v)
  for offset,output_offset in attention_offsets(info['copy_path']):
   buffers=[]
   for x in (q,k,v):
    buf=torch.full((x.nbytes+272,),165,dtype=torch.uint8,device='cuda');buf[offset:offset+x.nbytes].copy_(torch.from_numpy(x.view(np.uint8).reshape(-1)));buffers.append(buf)
   output=torch.full((4096+272,),165,dtype=torch.uint8,device='cuda');pointers=[b.data_ptr()+offset for b in buffers]+[output.data_ptr()+output_offset];validate_launch(info,pointers);maps=TensorMaps(info,pointers) if info['tma_plans'] else None
   args=[ctypes.c_uint64(x) for x in pointers+([d.data_ptr() for d in maps.device] if maps else [])+[0]];assert len(args)==len(physical);params=(ctypes.c_void_p*len(args))(*[ctypes.cast(ctypes.byref(x),ctypes.c_void_p) for x in args]);saved=[b.cpu().numpy().copy() for b in buffers]
   host.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,input_hashes=[digest(x.tobytes()) for x in (q,k,v)],expected_sha256=digest(ref.tobytes()),descriptor_setup_ns=maps.setup_ns if maps else []))
   def launch():
    status=cuda.cuLaunchKernel(function,1,1,1,128,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),params,None);assert status==0,status
   for repeat in range(3):
    output.fill_(165);launch();torch.cuda.synchronize();raw=output.cpu().numpy();got=raw[output_offset:output_offset+4096].copy().view(np.float32).reshape(64,16);error=np.abs(got-ref);numerics=np.isfinite(got).all() and np.all(error<=.002+.002*np.abs(ref));guards=np.all(raw[:output_offset]==165) and np.all(raw[output_offset+4096:]==165);unchanged=all(np.array_equal(b.cpu().numpy(),before) for b,before in zip(buffers,saved));records.append(dict(case=case_name,input_offset=offset,output_offset=output_offset,repeat=repeat,max_abs_error=float(error.max()),numerics=bool(numerics),guards=bool(guards),inputs_unchanged=unchanged));assert numerics and guards and unchanged,records[-1]
 result=dict(cubin_sha256=digest(binary),gpu_uuid=actual_uuid,physical_params=physical,physical_resources=resources,copy_path=info['copy_path'],tma_descriptor_count=len(info['tma_plans']),records=records,inputs=host,tolerance='finite abs(error)<=.002+.002*abs(reference)',safety_subset=safety)
 (directory/('safety-correctness.json' if safety else 'correctness.json')).write_text(json.dumps(result,indent=2));return result

from joint_measure import analyze_paired_batches

def main():
 if len(sys.argv)>1 and sys.argv[1]=="measure":
  from joint_measure import measure
  parser=argparse.ArgumentParser();parser.add_argument("command");parser.add_argument("--artifact-dir",type=Path,required=True);parser.add_argument("--batches",type=int,default=10);parser.add_argument("--seed",type=int,default=20261030);a=parser.parse_args();return measure(a.artifact_dir,a.batches,a.seed)
 if len(sys.argv)>1 and sys.argv[1] in ('--attention-execute','--attention-load'):
  print(json.dumps(execute_attention(Path(sys.argv[2]),load_only=sys.argv[1]=='--attention-load',safety='--safety' in sys.argv)));return
 if len(sys.argv)>1 and sys.argv[1] in ('prepare','validate'):
  parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','validate']);parser.add_argument('--artifact-dir',type=Path,required=True);a=parser.parse_args();return prepare(a.artifact_dir) if a.command=='prepare' else validate(a.artifact_dir)
 return forced_main()
if __name__=='__main__':main()
