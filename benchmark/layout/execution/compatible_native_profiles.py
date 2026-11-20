"""Verified code-contract views of sealed native calibrations, without refitting.

Only audited Shuffle implementation changes and two monotone register-domain
bounds may differ. Shared addressing/ownership/bank calculations are preserved.
A view changes the symbolic-bank implementation map, never the fitted model.
"""
from __future__ import annotations
import ast, copy, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'guoqiao/optimize_plan/network_selection_execution_artifacts'
OLD_SNAPSHOT=ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts/development/source-freeze-r2/source'
SEALED_INDICES=((ROOT/'guoqiao/optimize_plan/native_cost_repair_artifacts/evidence-index.json','66b28196c5065d5cc4504d3a069991b766718ec7d5618163533c120ba33ded9b'),(ROOT/'guoqiao/optimize_plan/partial_cost_efficiency_artifacts/evidence-index.json','eff4c3ca39681c91115e258638a34f5242ca1f4eb8d984a67835928657337fef'))
PREFIX='benchmark/layout/execution/'
BANK_FILES=tuple(PREFIX+n for n in ('native_cost_calibration.py','backend.py','demand_coverage_audit.py','redistribution.py'))
VIEW='frisk-sealed-native-profile-compatible-view-v1'
PROOF='frisk-shared-bank-content-proof-network-compact-compatibility-v5'

_NETWORK_ANCHOR=ROOT/'guoqiao/optimize_plan/gf2_decision_refinement_artifacts/development/source-freeze-final/source'
_NETWORK_INDEX=ROOT/'guoqiao/optimize_plan/gf2_decision_refinement_artifacts/evidence-index.json'
_NETWORK_INDEX_SHA='d9ab1a589446a9fbb1237d4dc47a4948eab2c06b56be92712b5fd6b87ff21774'
_NETWORK_HELPER=PREFIX+'shuffle_network.py'
_NETWORK_HELPER_SHA='3911df633237e13e4918e989cbad6e65c924e4baaffd8f8dafe83b07f0bd862a'
_NETWORK_ANCHOR_HASHES={'benchmark/layout/execution/backend.py': '9152c357eda2a84edfc4e4d771af7635ccfc42a58f5808787c1edba117245979', 'benchmark/layout/execution/redistribution.py': 'a1af6daa18f8e2c7f08eaac5784ec0e7423b411e5de5dfad841765af3ef5a829', 'benchmark/layout/execution/demand_coverage_audit.py': 'ae5450f0f174582bbdb54b6fb64bcbc510ded09a4d78289f6d21b8906af4558c'}
_NETWORK_APPROVED_AST={'benchmark/layout/execution/backend.py': {'Emitter.__init__': 'e4a9b0b97f629b277d484175f6ea6160e43c3040e0948b93292e8c6944d59e27', 'Emitter._emit_shuffle_network': 'feadb7b8e1c05441b7411532ec4cfa075dcba47b038705de165c67cc64d64f9b', 'Emitter.convert': '0f7c7a1cde8ce870bc9bac952a34cd5a06bf3f3241d4234750631531b902630f', 'Emitter.run': 'afab05bf16d7ffe4de90c9d50b418a9e6c9770d609c5e02535e522faca362da7', 'emit': '7d3d3f31909b2bb0fe620a301633982ff98eeaea1b565c36aa6ff8c3470f04ec'}, 'benchmark/layout/execution/redistribution.py': {'validate': '36f20a6c019d42075d2d0ca87021efd96af528b19d4b3d61fd64f0fbbac8bfa3'}, 'benchmark/layout/execution/demand_coverage_audit.py': {'conversion_structure': '56a31e8314c0436573769bf82145861f751a5a9fc0c67f50b574abc123b0a7e3'}}

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
IMPLEMENTATION_SHA256=sha(__file__)

def _canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def _digest(x):return hashlib.sha256(_canonical(x).encode()).hexdigest()
def _read(path):
 def unique(pairs):
  result={}
  for k,v in pairs:
   if k in result:raise ValueError('duplicate JSON key')
   result[k]=v
  return result
 return json.loads(_regular(path).read_text(),object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def _regular(path):
 p=Path(path).absolute()
 if not p.is_file() or any(x.is_symlink() for x in (p,*p.parents)):raise ValueError('regular nonsymlink file required: '+str(p))
 return p.resolve()
def _add(deps,path,expected=None):
 p=_regular(path);h=sha(p)
 if expected is not None and h!=expected:raise ValueError('dependency hash changed: '+str(p))
 if str(p) in deps and deps[str(p)]!=h:raise ValueError('dependency changed while loading')
 deps[str(p)]=h;return p

def _sealed(path,deps,indices):
 path=_regular(path)
 for index,h in SEALED_INDICES:
  index=Path(index)
  if path.is_relative_to(index.parent.resolve()):
   _add(deps,index,h)
   if str(index) not in indices:
    data=_read(index);indices[str(index)]={str((ROOT/row['path']).resolve()):row for row in data['entries']}
   row=indices[str(index)].get(str(path))
   if row is None or row['sha256']!=sha(path) or row['bytes']!=path.stat().st_size:raise ValueError('dependency differs from sealed evidence: '+str(path))
   return
 raise ValueError('base calibration dependency is outside sealed evidence: '+str(path))

def _completion(path,deps,indices):
 p=_regular(path);x=_read(p)
 names={'known_labels_sha256':'known-labels.json','unsupported_sha256':'unsupported.json','labels_sha256':'labels.json','raw_sha256':'micro-raw.json','provenance_sha256':'provenance.json','registration_sha256':'registration.json'}
 for field,name in names.items():
  if field in x:_add(deps,p.parent/name,x[field])
 for tool,h in x.get('safety_hashes',{}).items():
  if tool not in ('memcheck','racecheck','synccheck','initcheck'):raise ValueError('unknown LOCAL safety tool')
  _add(deps,p.parent/('safety-'+tool+'.json'),h)
 for name in list(deps):
  if Path(name).parent==p.parent:_sealed(name,deps,indices)

def _original(path,deps,indices,seen=None):
 path=_regular(path);seen=set() if seen is None else seen
 if path in seen:raise ValueError('cyclic profile dependency')
 seen.add(path);raw=_read(path)
 if raw.get('contract')==VIEW or 'compatibility_view' in raw:raise ValueError('nested or rewritten base profile')
 inputs=_read(path.parent/'inputs.json')
 import native_cost_calibration as base
 local='local_completion' in inputs
 if bool(raw.get('calibration'))!=local:raise ValueError('LOCAL provenance/inputs disagree')
 if local:
  import native_local_calibration as original
  profile=original.load_profile(path)
 else:profile=base.load_profile(path)
 if profile!=raw:raise ValueError('original loader changed calibration')
 for name,field in [(path.name,'profile_sha256'),('fit.json','fit_sha256'),('training-ledger.json','ledger_sha256')]:_add(deps,path.parent/name,inputs[field])
 _add(deps,path.parent/'inputs.json')
 if (path.parent/'pair-ledger.json').exists():_add(deps,path.parent/'pair-ledger.json',inputs.get('pair_ledger_sha256'))
 for key in ('known_completion','micro_completion')+ (('local_completion',) if local else ()):
  completion=_add(deps,inputs[key],inputs[key+'_sha256']);_completion(completion,deps,indices)
 if local:
  if 'pair_ledger_sha256' not in inputs:raise ValueError('LOCAL pair ledger is not bound')
  fixed=_add(deps,inputs['fixed_profile'],inputs['fixed_profile_sha256']);old=_original(fixed,deps,indices,seen)
  if any(profile['coefficients'][n]!=old['coefficients'][n] for n in profile['features'] if n!='local_transfer_bytes'):raise ValueError('frozen nonLOCAL coefficients changed')
  loader=_add(deps,ROOT/(PREFIX+'native_local_calibration.py'),profile['calibration']['implementation_sha256'])
  _add(deps,OLD_SNAPSHOT/(PREFIX+'native_local_calibration.py'),sha(loader))
 for name in list(deps):
  if name not in {str(Path(p).resolve()) for p,h in SEALED_INDICES} and Path(name).is_relative_to(Path(path).parent.parent):_sealed(name,deps,indices)
 _sealed(path,deps,indices);seen.remove(path);return profile

# Exact reviewed helper AST. It only reads the plan and mutates local containers.
_HELPER='''
def shuffle_source_support(plan, register):
 if plan.get('mode')!='Shuffle':raise ValueError('shuffle support requires Shuffle')
 dr=plan['destination_registers'];nr=plan['source_registers']
 if type(register) is not int or not 0<=register<dr or len(plan['destinations'])!=128*dr:
  raise ValueError('shuffle support destination domain')
 table=[plan['destinations'][t*dr+register]['source']['reg'] for t in range(128)]
 if any(type(x) is not int or not 0<=x<nr for x in table):raise ValueError('shuffle support source domain')
 actual=set(table);base=table[0];basis={}
 for bit in range(7):
  column=table[1<<bit]^base
  while column:
   pivot=column.bit_length()-1
   if pivot not in basis:basis[pivot]=column;break
   column^=basis[pivot]
 image=[0]
 for column in basis.values():image += [x^column for x in image]
 coset={base^x for x in image}
 return tuple(sorted(coset if coset==actual else actual))
'''
_BACKEND_SHUFFLE='''
lane=self.map([o['lane'] for o in owners]);candidates=[]
for source_register in shuffle_source_support(p,r):
 source=self.regs(a)[source_register]
 q=self.tmp();self.emit(f'shfl.sync.idx.b32 {q}, {source}, {lane}, 31, -1;');candidates.append((source_register,q))
chosen=self.map([o['reg'] for o in owners]);self.emit(f'mov.b32 {z}, {candidates[0][1]};')
for source_register,q in candidates[1:]:
 pred=self.tmp('pred');self.emit(f'setp.eq.u32 {pred}, {chosen}, {source_register};');self.emit(f'@{pred} mov.b32 {z}, {q};')
continue
'''
_AUDIT_SHUFFLE='''
from redistribution import shuffle_source_support
count=len(shuffle_source_support(plan,reg))
work=map_work([o['reg'] for o in owners])+1+2*(count-1)+map_work([o['lane'] for o in owners])
value['shuffle_warp_issues']+=4*count
'''

def _dump(node):return ast.dump(node,include_attributes=False)
class _Docs(ast.NodeTransformer):
 def _body(self,node):
  self.generic_visit(node)
  if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
  return node
 visit_Module=_body;visit_ClassDef=_body;visit_FunctionDef=_body;visit_AsyncFunctionDef=_body

def _tree(text):return _Docs().visit(ast.parse(text))
def _function(tree,name,cls=None):
 scope=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else tree.body
 found=[n for n in scope if isinstance(n,ast.FunctionDef) and n.name==name]
 if len(found)!=1:raise ValueError('required source function missing/duplicated: '+name)
 return found[0]
def _shuffle(node,variable):return isinstance(node,ast.If) and _dump(node.test)==_dump(ast.parse(variable+"=='Shuffle'",mode='eval').body)
def _replace_body(function,variable,expected_new):
 nodes=sorted((n for n in ast.walk(function) if _shuffle(n,variable)),key=lambda n:n.lineno)
 if len(nodes)!=len(expected_new):raise ValueError('Shuffle branch boundary changed')
 for node,approved in zip(nodes,expected_new):
  if approved is not None and _dump(ast.Module(body=node.body,type_ignores=[]))!=_dump(_tree(approved)):raise ValueError('unreviewed Shuffle branch implementation')
  if node.orelse:raise ValueError('Shuffle branch acquired else')
  node.body=[ast.Pass()]
def _guard(oldfn,newfn,marker):
 def find(fn):
  values=[n for n in ast.walk(fn) if isinstance(n,ast.If) and any(isinstance(x,ast.Constant) and x.value==marker for x in ast.walk(ast.Module(body=n.body,type_ignores=[])))]
  if len(values)!=1:raise ValueError('specific register guard boundary changed')
  return values[0]
 old=find(oldfn);new=find(newfn)
 if _dump(old)==_dump(new):return 'unchanged'
 candidate=copy.deepcopy(new);values=[n for n in ast.walk(candidate.test) if isinstance(n,ast.Constant) and n.value==512]
 if len(values)!=1:raise ValueError('only exact256-to512 boundary extension allowed')
 values[0].value=256
 if _dump(candidate)!=_dump(old):raise ValueError('nonmonotone register domain change')
 new.test=copy.deepcopy(old.test);return '256-to512'
def _require(fn,message):
 found=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value==message]
 if len(found)!=1:raise ValueError('specific redistribution guard missing/duplicated')
 return found[0]

_OFFSET_WRAPPER="""
def offsets(self,v):
 if v not in self._offset_cache:self._offset_cache[v]=self._compute_offsets(v)
 return self._offset_cache[v]
"""

def _offset_memo(old,new):
 """Erase only a pure instance memo; preserve the original computation AST."""
 oldfn=_function(old,'offsets','Emitter')
 klass=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='Emitter')
 functions=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_offsets']
 if not functions:return dict(enabled=False)
 if len(functions)!=1:raise ValueError('offset computation duplicated')
 compute=functions[0];wrapper=_function(new,'offsets','Emitter')
 if _dump(wrapper)!=_dump(_function(_tree(_OFFSET_WRAPPER),'offsets')):raise ValueError('offset memo must be the reviewed instance-cache wrapper')
 reference=copy.deepcopy(compute);reference.name='offsets'
 if _dump(reference)!=_dump(oldfn):raise ValueError('original offset/address calculation changed')
 init=_function(new,'__init__','Emitter');empty=_tree('self._offset_cache={}').body[0]
 assignments=[n for n in init.body if _dump(n)==_dump(empty)]
 if len(assignments)!=1:raise ValueError('offset memo requires one fresh empty instance dictionary')
 init.body.remove(assignments[0]);klass.body.remove(wrapper);compute.name='offsets'
 return dict(enabled=True,contract='per-emitter-immutable-storage-offset-table-v1',original_computation_AST_sha256=hashlib.sha256(_dump(oldfn).encode()).hexdigest(),wrapper_AST_sha256=hashlib.sha256(_dump(wrapper).encode()).hexdigest(),key='storage value id',global_or_cross_instance_cache=False)

def _network_projection(name,new):
 """Mechanically project reviewed transport-only edits onto sealed v2 code.

 Exact whole-function AST pins prevent an edited Shared path being erased by
 the projection. Shared conversion statements are also compared directly before
 any replacement. The separate pure planner is byte-pinned and dependency bound.
 This proves Shared-bank feature compatibility, not time-model requalification.
 """
 present=any(isinstance(n,ast.Name) and n.id=='plan_shuffle_network' for n in ast.walk(new))
 if not present:return dict(enabled=False)
 if name not in _NETWORK_APPROVED_AST:raise ValueError('unreviewed network projection source')
 if sha(_regular(_NETWORK_INDEX))!=_NETWORK_INDEX_SHA:raise ValueError('network compatibility anchor seal changed')
 anchor_path=_regular(_NETWORK_ANCHOR/name)
 if sha(anchor_path)!=_NETWORK_ANCHOR_HASHES[name]:raise ValueError('network compatibility anchor source changed')
 from shared_semantics_projection import dependencies
 active_helper=dependencies()[0]['helper_sha256'][_NETWORK_HELPER]
 if sha(_regular(ROOT/_NETWORK_HELPER))!=active_helper:raise ValueError('unreviewed shuffle network helper')
 anchor=_tree(anchor_path.read_text());details={}
 for qualified,expected in _NETWORK_APPROVED_AST[name].items():
  cls,fn=qualified.rsplit('.',1) if '.' in qualified else (None,qualified)
  current=_function(new,fn,cls)
  actual=hashlib.sha256(_dump(current).encode()).hexdigest()
  if actual!=expected:raise ValueError('unreviewed network transport function: '+qualified)
  details[qualified]=actual
 if name.endswith('backend.py'):
  current=_function(new,'convert','Emitter');previous=_function(anchor,'convert','Emitter')
  def shared_tail(fn):
   indexes=[i for i,n in enumerate(fn.body) if isinstance(n,ast.If) and _dump(n.test)==_dump(ast.parse("p['mode'] in ('Register','Shuffle')",mode='eval').body)]
   if len(indexes)!=1:raise ValueError('Shared convert boundary changed')
   return ast.Module(body=fn.body[indexes[0]+1:],type_ignores=[])
  if _dump(shared_tail(current))!=_dump(shared_tail(previous)):raise ValueError('Shared conversion transport changed')
  details['shared_tail_AST_sha256']=hashlib.sha256(_dump(shared_tail(current)).encode()).hexdigest()
  imports=[n for n in new.body if isinstance(n,ast.ImportFrom) and n.module=='shuffle_network']
  if len(imports)!=1 or _dump(imports[0])!=_dump(ast.parse('from shuffle_network import plan_shuffle_network').body[0]):raise ValueError('unreviewed network import')
  new.body.remove(imports[0])
 elif name.endswith('demand_coverage_audit.py'):
  def branch(tree):
   fn=_function(tree,'conversion_structure');found=[n for n in ast.walk(fn) if _shuffle(n,'mode') is False and isinstance(n,ast.If) and _dump(n.test)==_dump(ast.parse("mode=='Shared'",mode='eval').body)]
   if len(found)!=1:raise ValueError('Shared audit boundary changed')
   return ast.Module(body=found[0].body,type_ignores=[])
  if _dump(branch(new))!=_dump(branch(anchor)):raise ValueError('Shared bank feature calculation changed')
  details['shared_bank_branch_AST_sha256']=hashlib.sha256(_dump(branch(new)).encode()).hexdigest()
 for qualified in _NETWORK_APPROVED_AST[name]:
  cls,fn=qualified.rsplit('.',1) if '.' in qualified else (None,qualified)
  scope=next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name==cls).body if cls else new.body
  current=_function(new,fn,cls);at=scope.index(current)
  if fn=='_emit_shuffle_network':scope.pop(at)
  else:scope[at]=copy.deepcopy(_function(anchor,fn,cls))
 return dict(enabled=True,anchor_sha256=_NETWORK_ANCHOR_HASHES[name],reviewed_functions=details,
             helper_sha256=active_helper,historical_helper_sha256=_NETWORK_HELPER_SHA,scope='Shared-bank features on valid common-domain inputs; no new latency calibration')

def _projection(name,oldtext,newtext):
 from shared_semantics_projection import project
 old=_tree(oldtext);new=ast.parse(newtext);epoch=project(name,new);new=_Docs().visit(new);details={'content_proof_epoch':epoch,'shuffle_network':_network_projection(name,new)}
 if name.endswith('backend.py'):
  details['offsets_memo']=_offset_memo(old,new)
  for node in new.body:
   if isinstance(node,ast.ImportFrom) and node.module=='redistribution':node.names=[x for x in node.names if not (x.name=='shuffle_source_support' and x.asname is None)]
  of=_function(old,'convert','Emitter');nf=_function(new,'convert','Emitter');_replace_body(of,"p['mode']",[None]);_replace_body(nf,"p['mode']",[_BACKEND_SHUFFLE])
  details['demand_boundary']=_guard(_function(old,'demand_program','Emitter'),_function(new,'demand_program','Emitter'),'large cross-thread demand boundary exceeds the bounded conversion backend')
 elif name.endswith('redistribution.py'):
  helper=_function(new,'shuffle_source_support')
  if _dump(helper)!=_dump(_function(_tree(_HELPER),'shuffle_source_support')):raise ValueError('unreviewed/nonpure support helper')
  new.body.remove(helper);of=_function(old,'validate');nf=_function(new,'validate');og=_require(of,'bounded domain');ng=_require(nf,'bounded domain')
  if _dump(og)!=_dump(ng):
   candidate=copy.deepcopy(ng);limits=[n for n in ast.walk(candidate.args[0]) if isinstance(n,ast.Constant) and n.value==512]
   if len(limits)!=2:raise ValueError('ordinary bounded domain must extend exactly two256 limits')
   for n in limits:n.value=256
   if _dump(candidate)!=_dump(og):raise ValueError('ordinary domain changed beyond monotone register cap')
   ng.args=copy.deepcopy(og.args);details['ordinary_registers']='256-to512'
  oldwork=_require(of,'shuffle code-size limit');newwork=_require(nf,'shuffle emitted-work limit')
  approved=_tree("require(mode!='Shuffle' or sum(len(shuffle_source_support(p,r)) for r in range(dr))<=65536,'shuffle emitted-work limit')").body[0].value
  if _dump(newwork)!=_dump(approved):raise ValueError('unreviewed Shuffle-only work guard')
  newwork.args=copy.deepcopy(oldwork.args)
 elif name.endswith('demand_coverage_audit.py'):
  _replace_body(_function(old,'conversion_structure'),'mode',[None,None]);_replace_body(_function(new,'conversion_structure'),'mode',[_AUDIT_SHUFFLE,"value['bytes']=value['shuffle_warp_issues']*32*4"])
 else:raise ValueError('unrecognized projected source')
 if _dump(old)!=_dump(new):raise ValueError('Shared or nonallowlisted source semantics changed: '+name)
 return dict(projected_AST_sha256=hashlib.sha256(_dump(old).encode()).hexdigest(),allowed_boundaries=details)

def _payload(profile_path):
 if sha(__file__)!=IMPLEMENTATION_SHA256:raise ValueError('compatibility verifier changed since import')
 deps={};indices={};_add(deps,__file__,IMPLEMENTATION_SHA256);profile_path=_regular(profile_path);profile=_original(profile_path,deps,indices);oldmap=profile['analysis_contract']['symbolic_bank_implementation_sha256']
 if set(oldmap)!=set(BANK_FILES):raise ValueError('original four-file bank identity required')
 oldmap=dict(oldmap);newmap={};proof={}
 for name in BANK_FILES:
  old=_add(deps,OLD_SNAPSHOT/name,oldmap[name]);_sealed(old,deps,indices);current=_add(deps,ROOT/name);newmap[name]=sha(current)
  if name.endswith('native_cost_calibration.py'):
   if newmap[name]!=oldmap[name]:raise ValueError('native calibration implementation must be byte identical')
   proof[name]=dict(byte_identical=True)
  else:proof[name]=_projection(name,old.read_text(),current.read_text())
 for name in _NETWORK_ANCHOR_HASHES:_add(deps,_NETWORK_ANCHOR/name,_NETWORK_ANCHOR_HASHES[name])
 _add(deps,_NETWORK_INDEX,_NETWORK_INDEX_SHA)
 from shared_semantics_projection import dependencies,helper_proofs
 epoch,extra=dependencies();helper_contract=helper_proofs()
 for path,digest in extra.items():_add(deps,path,digest)
 _add(deps,ROOT/_NETWORK_HELPER,epoch['helper_sha256'][_NETWORK_HELPER])
 # LOCAL's original validation code remains frozen even when the anchor is old native.
 loader=ROOT/(PREFIX+'native_local_calibration.py');oldloader=OLD_SNAPSHOT/(PREFIX+'native_local_calibration.py');_add(deps,loader,sha(oldloader));_add(deps,oldloader);_sealed(oldloader,deps,indices)
 return dict(schema_version=1,contract=PROOF,verifier_sha256=IMPLEMENTATION_SHA256,anchor_profile_path=str(profile_path),anchor_profile_sha256=sha(profile_path),old_snapshot_root=str(OLD_SNAPSHOT.resolve()),old_bank_implementation_sha256=oldmap,new_bank_implementation_sha256=newmap,proof=proof,helper_proofs=helper_contract,dependencies=deps,interpretation='Shared-only mechanical source compatibility on the common domain: complete pinned module AST projection through sealed network and sparse anchors; direct Shared transport, addressing/bank arithmetic and allocator lifetime checks; pure GF2 query/canonical-section equivalence and content-bound per-emission proof reuse with whole-packet exit seal. Source definitions and network alternatives changed. This bridge does not migrate transport timing, native analyzer semantics, fitted coefficients, hardware-time claims or global selection qualification. Any predicate-DAG recalibration is a separate known-only experiment.')

def _receipt(path):
 envelope=_read(path)
 if set(envelope)!={'payload','payload_sha256'} or _digest(envelope['payload'])!=envelope['payload_sha256']:raise ValueError('compatibility receipt envelope mismatch')
 payload=envelope['payload']
 if payload.get('contract')!=PROOF or payload.get('verifier_sha256')!=IMPLEMENTATION_SHA256:raise ValueError('wrong compatibility verifier contract')
 expected=_payload(payload['anchor_profile_path'])
 if _canonical(payload)!=_canonical(expected):raise ValueError('source/training dependency or compatibility proof changed')
 return payload

def _fresh(path,value):
 path=Path(path).absolute();parent=path.parent.resolve()
 if not parent.is_relative_to(BASE.resolve()) or path.exists() or path.is_symlink():raise ValueError('fresh new-stage output required')
 parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
 return path.resolve()

def create_bank_compatibility_receipt(path,*,profile_path,snapshot_root=None):
 if snapshot_root is not None and Path(snapshot_root).resolve()!=OLD_SNAPSHOT.resolve():raise ValueError('only the sealed source snapshot is allowed')
 payload=_payload(profile_path);return _fresh(path,dict(payload=payload,payload_sha256=_digest(payload)))

def base_profile_path(path):
 path=_regular(path);value=_read(path)
 if value.get('contract')=='frisk-predicate-native-recalibration-v1':
  from predicate_cost_recalibration import base_profile_path as strict_base
  return strict_base(path)
 return _regular(value['base_profile_path']) if value.get('contract')==VIEW else path

def create_view(path,base_profile_path,receipt_path):
 basepath=_regular(base_profile_path);receiptpath=_regular(receipt_path);proof=_receipt(receiptpath);profile=_original(basepath,{},{});
 if profile['analysis_contract']['symbolic_bank_implementation_sha256']!=proof['old_bank_implementation_sha256']:raise ValueError('profile and compatibility proof have different original bank contracts')
 return _fresh(path,dict(schema_version=1,contract=VIEW,base_profile_path=str(basepath),base_profile_sha256=sha(basepath),bank_compatibility_receipt_path=str(receiptpath),bank_compatibility_receipt_sha256=sha(receiptpath)))

def _load(path):
 path=_regular(path);view=_read(path);deps={}
 if view.get('contract')=='frisk-predicate-native-recalibration-v1':
  from predicate_cost_recalibration import load_profile as strict_load,dependency_hashes as strict_dependencies
  return strict_load(path),strict_dependencies(path)
 if view.get('contract')!=VIEW:return _original(path,deps,{}),deps
 if set(view)!={'schema_version','contract','base_profile_path','base_profile_sha256','bank_compatibility_receipt_path','bank_compatibility_receipt_sha256'} or type(view['schema_version'])is not int or view['schema_version']!=1:raise ValueError('wrong profile view schema')
 _add(deps,path);basepath=_add(deps,view['base_profile_path'],view['base_profile_sha256']);receiptpath=_add(deps,view['bank_compatibility_receipt_path'],view['bank_compatibility_receipt_sha256']);proof=_receipt(receiptpath);profile=_original(basepath,deps,{})
 if profile['analysis_contract']['symbolic_bank_implementation_sha256']!=proof['old_bank_implementation_sha256']:raise ValueError('view base bank identity differs')
 deps.update(proof['dependencies']);profile=copy.deepcopy(profile);profile['analysis_contract']['symbolic_bank_implementation_sha256']=copy.deepcopy(proof['new_bank_implementation_sha256']);profile['compatibility_view']=dict(contract=VIEW,path=str(path),sha256=sha(path),base_profile_path=str(basepath),base_profile_sha256=sha(basepath),receipt_path=str(receiptpath),receipt_sha256=sha(receiptpath),refitted=False,hardware_latency_guarantee=False)
 return profile,deps

def load_profile(path):return _load(path)[0]
def dependency_hashes(path):return _load(path)[1]


def validate_shared_semantics_bridge(path):
 """Strict Shared-only source bridge for known-label feature re-extraction."""
 return _receipt(path)
