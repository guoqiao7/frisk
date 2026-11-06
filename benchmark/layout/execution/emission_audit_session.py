"""Explicit one-case checked-emission observations shared by independent audits.

Legacy modules and their globals are never modified. Private function views run
exactly the original audit code objects with a closed, case-local backend view.
Every predicate/ledger check still runs; only identical successful strong proofs
and checked emission facts are reused. This module is a separately bound producer.
"""
from __future__ import annotations
import builtins
from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
import importlib
import json
from pathlib import Path
import types
import io
import pickle
import time
import struct

CONTRACT='frisk-case-local-emission-audit-witness-v1'
_ACTIVE=ContextVar('frisk_emission_audit_factory',default=None)
_MODULES=('backend','redistribution','execution_support','storage_addresses',
          'shared_allocation','async_copy','async_host','shuffle_network',
          'demand_coverage_audit','native_surviving_audit','native_static_support')

def generation_execution_contract():
    return dict(contract=CONTRACT,scope='one-add-case',emission_observer='complete-ordered-affine-query-trace',
                independent_audit_predicates=True,default_apis='unchanged-independent',
                source_binding='full-producer-files-and-live-function-bindings',persistent_cache=False)

@contextmanager
def enabled():
    if _ACTIVE.get() is not None:raise ValueError('fresh nonnested emission provider required')
    token=_ACTIVE.set(EmissionSession)
    try:yield
    finally:_ACTIVE.reset(token)

def current_factory():return _ACTIVE.get()

def _sha(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('regular emission identity required')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _private(function, replacements, *, importer=None):
    # A new function object retains the complete original code and closure.
    # No assignment is made to a legacy module/function/class/global namespace.
    namespace=dict(function.__globals__);namespace.update(replacements)
    if importer is not None:
        namespace['__builtins__']=dict(vars(builtins),__import__=importer)
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=copy.copy(function.__kwdefaults__)
    return result

def _json_snapshot(value):
    # Wire input has ordinary JSON values. C codecs avoid recursive Python
    # deepcopy of large owner tables while retaining a fully independent copy.
    return json.loads(json.dumps(value,separators=(',',':'),allow_nan=False))

def _snapshot_classes():
    from storage_addresses import RowMajorAddresses
    from execution_support import LogicalTable,SymbolicMap,_ImageBasis
    return {(value.__module__,value.__name__):value for value in
            (RowMajorAddresses,LogicalTable,SymbolicMap,_ImageBasis)}

def _snapshot_dump(state):
    # Private, case-local bytes only. Never accept a path or external payload.
    # The object whitelist is also enforced while serializing, not just loading.
    allowed=_snapshot_classes();classes=tuple(allowed.values())
    row_major=allowed[('storage_addresses','RowMajorAddresses')]
    class Writer(pickle.Pickler):
        def reducer_override(self,obj):
            if type(obj) is row_major:return (row_major,(obj.description(),))
            if type(obj) in classes or any(obj is cls for cls in classes):return NotImplemented
            raise ValueError('unknown object in emission state snapshot')
    stream=io.BytesIO();Writer(stream,protocol=5).dump(state);return stream.getvalue()

def _snapshot_load(blob):
    # This loader is reachable only for the session's just-created, sealed
    # in-memory bytes. No arbitrary imports/globals or pickle persistence.
    allowed=_snapshot_classes()
    class Reader(pickle.Unpickler):
        def find_class(self,module,name):
            if (module,name) not in allowed:raise ValueError('unknown emission snapshot global')
            return allowed[(module,name)]
    return Reader(io.BytesIO(blob)).load()

def _bindings(module, digest):
    functions=[];constants=[]
    def bind(name,value):
        if callable(value):
            functions.append((name,value,getattr(value,'__code__',None),
                              digest((getattr(value,'__defaults__',None),getattr(value,'__kwdefaults__',None)))))
    for name,value in sorted(vars(module).items()):
        bind(name,value)
        if isinstance(value,type) and value.__module__==module.__name__:
            for attr,method in sorted(vars(value).items()):
                if isinstance(method,(staticmethod,classmethod)):method=method.__func__
                bind(name+'.'+attr,method)
        if name.isupper():
            fingerprint=digest(value)
            if fingerprint is not None:constants.append((name,fingerprint))
    return functions,constants

class _EmitterView:
    def __init__(self,session,observer):
        session._ensure_emission();self._session=session;self._observer=observer;self._ran=False
        # Each consumer owns a deep snapshot, including normalized lazy maps,
        # register declarations and code. No mutable Emitter state is shared.
        self._emitter=object.__new__(session.backend.Emitter)
        clone_start=time.perf_counter_ns()
        self._emitter.__dict__.update(_snapshot_load(session._state))
        from storage_addresses import RowMajorByteOffsets
        self._emitter._offset_cache={key:RowMajorByteOffsets(value[1]) if value[0]=='row-major-byte-offsets'
                                    else copy.deepcopy(value[1]) for key,value in session._offset_cache_state.items()}
        self._emitter.demand_query_observer=None
        self._emitter._redistribution_validator=session.backend.validate_redistribution
        proof=session.redistribution.ProofCache(validator=session.backend.validate_redistribution)
        proof.receipts=dict(session._proof_receipts)
        self._emitter._redistribution_proofs=proof
        session.stats['snapshot_clone_ns']+=time.perf_counter_ns()-clone_start
    def __getattr__(self,name):return getattr(self._emitter,name)
    def run(self):
        if self._ran:raise ValueError('one emission replay per audit consumer required')
        self._ran=True;session=self._session;trace_start=time.perf_counter_ns()
        if self._observer is not None:
            for value,indices in session._events:
                self._observer(value,struct.unpack('<128I',indices))
        session.stats['observer_replay_ns']+=time.perf_counter_ns()-trace_start
        session.stats['emission_replays']+=1
        return session._text,copy.deepcopy(session._info)

class _BackendView:
    def __init__(self,session):self.session=session
    def __getattr__(self,name):return getattr(self.session.backend,name)
    def Emitter(self,packet,*,demand_query_observer=None,demand_query_budget=65536,shuffle_network_policy='auto'):
        session=self.session
        if session.digest(packet)!=session._packet_digest:raise ValueError('emission packet differs from case witness')
        if demand_query_budget!=65536 or shuffle_network_policy!='auto':raise ValueError('unregistered emission options')
        return _EmitterView(session,demand_query_observer)
    def emit(self,packet,**kwargs):
        session=self.session;session._ensure_emission()
        if kwargs or session.digest(packet)!=session._normalized_digest:raise ValueError('assessment normalization differs from emission witness')
        session.stats['emission_replays']+=1
        return session._text,copy.deepcopy(session._info)

class _DemandView:
    def __init__(self,session):self.session=session
    def __getattr__(self,name):return getattr(self.session.demand_module,name)
    def audit(self,packet,record,*,require_priced_features=False):
        return self.session._demand(packet,record,require_priced_features=require_priced_features)
    def access_coverage(self,emitter,observed,info):
        return self.session._access_coverage(emitter,observed,info)

class EmissionSession:
    def __init__(self,packet,record,*,bindings=(),extra_identity=None):
        self.backend=importlib.import_module('backend');self.redistribution=importlib.import_module('redistribution')
        self.digest=self.redistribution.content_digest
        self.demand_module=importlib.import_module('demand_coverage_audit')
        self.surviving_module=importlib.import_module('native_surviving_audit')
        self.support_module=importlib.import_module('execution_support')
        self._caller_packet=packet;self._caller_record=record;self._caller_extra=extra_identity
        self._packet_digest=self.digest(packet);self._record_digest=self.digest(record)
        if self._packet_digest is None or self._record_digest is None:raise ValueError('known complete typed emission content required')
        self._packet=_json_snapshot(packet);self._record=_json_snapshot(record)
        # The wire decoder has no aliases beyond interned keys. An ordinary
        # caller may supply aliasing; retain its exact typed representation.
        if self.digest(self._packet)!=self._packet_digest:self._packet=copy.deepcopy(packet)
        if self.digest(self._record)!=self._record_digest:self._record=copy.deepcopy(record)
        self._extra=copy.deepcopy(extra_identity);self._extra_digest=self.digest(extra_identity)
        if self._extra_digest is None:raise ValueError('known complete configuration identity required')
        self._files={str(Path(path).absolute()):_sha(path) for path in bindings}
        modules=[importlib.import_module(name) for name in _MODULES]
        modules.append(importlib.import_module(__name__))
        self._sources={str(Path(module.__file__).resolve()):_sha(module.__file__) for module in modules}
        self._module_bindings=[(module,_bindings(module,self.digest)) for module in modules]
        self._ready=False;self._closed=False;self._failed=False;self._proofs={};self._access_rows={}
        self._events=[];self._state=None;self._offset_cache_state={};self._proof_receipts={};self._witness_seal=None
        self.stats=dict(emitter_runs=0,emission_replays=0,strong_proof_hits=0,strong_proof_misses=0,guard_ns=0,consumer_ns=0,real_emit_ns=0,snapshot_capture_ns=0,snapshot_clone_ns=0,observer_replay_ns=0,access_row_hits=0,access_row_misses=0)
        backend_view=_BackendView(self);demand_view=_DemandView(self)
        self._demand_code=_private(self.demand_module.audit,{'backend':backend_view,'access_coverage':self._access_coverage})
        self._surviving_code=_private(self.surviving_module.audit,{'backend':backend_view,'emission':demand_view})
        original_import=builtins.__import__
        def local_import(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='backend' and tuple(fromlist)==('emit',) and level==0:return backend_view
            return original_import(name,globals,locals,fromlist,level)
        self._assess_code=_private(self.support_module.assess,{},importer=local_import)
        self._check()
    def __enter__(self):self._check();return self
    def __exit__(self,kind,error,trace):
        try:
            if kind is None:self._check()
        finally:self._closed=True
    def _check(self):
        try:self._verify()
        except Exception:
            self._failed=True;raise
    def _verify(self):
        if self._closed or self._failed:raise ValueError('closed or failed emission witness')
        if self.digest(self._caller_packet)!=self._packet_digest or self.digest(self._caller_record)!=self._record_digest:
            raise ValueError('caller packet/record changed during emission proof')
        if self.digest(self._caller_extra)!=self._extra_digest:raise ValueError('caller configuration identity changed')
        if self.digest(self._packet)!=self._packet_digest or self.digest(self._record)!=self._record_digest or self.digest(self._extra)!=self._extra_digest:
            raise ValueError('owned emission/configuration snapshot changed')
        for name,expected in {**self._sources,**self._files}.items():
            if _sha(name)!=expected:raise ValueError('emission source/configuration/manifest changed: '+name)
        for module,expected in self._module_bindings:
            if _bindings(module,self.digest)!=expected:raise ValueError('live emission/proof implementation changed: '+module.__name__)
        if self._ready and self._seal()!=self._witness_seal:raise ValueError('checked emission observation changed')
        for proof,expected in self._proofs.values():
            if self.digest(proof)!=expected:raise ValueError('strong proof receipt changed')
        for row,expected in self._access_rows.values():
            if self.digest(row)!=expected:raise ValueError('access footprint receipt changed')
    def _seal(self):
        return self.digest((self._text,self._info,self._events,self._state,self._offset_cache_state,self._proof_receipts,self._normalized_digest))
    def _ensure_emission(self):
        if self._ready:return
        try:
            def observe(value,indices):
                self.demand_module.canonical_query(indices)
                self._events.append((value,struct.pack('<128I',*indices)))
            emitter=self.backend.Emitter(self._packet,demand_query_observer=observe)
            self._normalized_digest=self.digest(emitter.data)
            if self._normalized_digest is None:raise ValueError('uncacheable normalized execution content')
            emit_start=time.perf_counter_ns()
            self._text,info=emitter.run();self.stats['emitter_runs']+=1
            self.stats['real_emit_ns']+=time.perf_counter_ns()-emit_start
            capture_start=time.perf_counter_ns()
            if self.digest(emitter.data)!=self._normalized_digest:raise ValueError('emitter mutated normalized packet')
            self._info=copy.deepcopy(info)
            self._state=_snapshot_dump({name:value for name,value in vars(emitter).items()
                                      if name not in ('demand_query_observer','_redistribution_validator','_redistribution_proofs','_offset_cache')})
            from storage_addresses import RowMajorByteOffsets
            for key,value in emitter._offset_cache.items():
                if type(value) is RowMajorByteOffsets:self._offset_cache_state[key]=('row-major-byte-offsets',len(value))
                elif type(value) is dict:self._offset_cache_state[key]=('plain-offset-map',copy.deepcopy(value))
                else:raise ValueError('unknown mutable offset-cache state')
            self._proof_receipts=dict(emitter._redistribution_proofs.receipts)
            self._witness_seal=self._seal()
            if self._witness_seal is None:raise ValueError('uncacheable complete emission observation')
            self._ready=True
            self.stats['snapshot_capture_ns']+=time.perf_counter_ns()-capture_start
        except Exception:
            self._failed=True;raise
    def _access_coverage(self,emitter,observed,info):
        # The original helper is a sum/list over independent memory operations.
        # Run its unchanged code on one operation at a time. Read rows depend on
        # that SSA's query set; write rows always enumerate the full register axis.
        # All other packet/config/source inputs are sealed by this one-case scope.
        result=[]
        for operation in emitter.data['operations']:
            name=operation['name']
            if name not in ('frisk.tile_load','frisk.tile_store'):continue
            read=name=='frisk.tile_load';value=operation['results'][0] if read else operation['operands'][0]
            signatures=observed.get(value,set()) if read else None
            key=self.digest(('one-memory-operation-footprint-v1',operation,signatures,info))
            if key is not None and key in self._access_rows:
                self.stats['access_row_hits']+=1
                row,seal=self._access_rows[key]
                if self.digest(row)!=seal:raise ValueError('access footprint receipt changed')
                result.append(copy.deepcopy(row));continue
            self.stats['access_row_misses']+=1
            class OneOperation:
                def __init__(self,original):
                    self.original=original;self.data=dict(original.data,operations=[operation])
                def __getattr__(self,name):return getattr(self.original,name)
            rows=self.demand_module.access_coverage(OneOperation(emitter),observed,info)
            if len(rows)!=1:raise ValueError('independent access row proof changed')
            row=rows[0];result.append(row)
            if key is not None and len(self._access_rows)<256:self._access_rows[key]=(copy.deepcopy(row),self.digest(row))
        return result
    def _demand(self,packet,record,*,require_priced_features=False):
        if self.digest(packet)!=self._packet_digest or self.digest(record)!=self._record_digest:
            raise ValueError('strong emission audit inputs differ')
        key=bool(require_priced_features)
        if type(require_priced_features) is not bool:raise ValueError('boolean pricing predicate required')
        if key in self._proofs:
            self.stats['strong_proof_hits']+=1
            proof,proof_seal=self._proofs[key]
            if self.digest(proof)!=proof_seal:raise ValueError('strong proof receipt changed')
            return copy.deepcopy(proof)
        self.stats['strong_proof_misses']+=1
        proof=self._demand_code(packet,record,require_priced_features=require_priced_features)
        # A fallback is returned normally but never saved as a successful proof.
        if proof.get('status')=='passed':self._proofs[key]=(copy.deepcopy(proof),self.digest(proof))
        return proof
    def _call(self,function,*args,**kwargs):
        try:
            start=time.perf_counter_ns();self._check();self.stats['guard_ns']+=time.perf_counter_ns()-start
            start=time.perf_counter_ns();result=function(*args,**kwargs);self.stats['consumer_ns']+=time.perf_counter_ns()-start
            start=time.perf_counter_ns();self._check();self.stats['guard_ns']+=time.perf_counter_ns()-start
            return result
        except Exception:
            self._failed=True;raise
    def demand_audit(self,*,require_priced_features=False):
        return self._call(self._demand,self._packet,self._record,require_priced_features=require_priced_features)
    def surviving_audit(self,*,require_priced_features=True):
        return self._call(self._surviving_code,self._packet,self._record,require_priced_features=require_priced_features)
    def assess(self,*,include_emission=False):
        return self._call(self._assess_code,self._packet,include_emission=include_emission)
    def receipt(self,*,artifacts=None):
        if artifacts is not None:
            if not self._ready or set(artifacts)!={'ttgir','typed_config','configuration'}:
                self._failed=True;raise ValueError('complete generated artifact binding required')
            encoded=lambda x:(json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
            data={'ttgir':self._text.encode(),'typed_config':encoded(self._info),'configuration':encoded(self._extra)}
            paths={name:str(Path(path).absolute()) for name,path in artifacts.items()}
            canonical=[str(Path(path).resolve()) for path in paths.values()]
            occupied={str(Path(path).resolve()) for path in (*self._files,*self._sources)}
            if len(set(canonical))!=len(canonical) or set(canonical)&occupied:
                self._failed=True;raise ValueError('generated artifacts cannot replace an existing binding')
            for name,path in paths.items():self._files[path]=hashlib.sha256(data[name]).hexdigest()
        self._check()
        if not self._ready:raise ValueError('successful checked emission required')
        return dict(schema_version=1,contract=CONTRACT,packet_typed_sha256=self._packet_digest,
                    record_typed_sha256=self._record_digest,configuration_identity_sha256=self._extra_digest,
                    normalized_packet_typed_sha256=self._normalized_digest,
                    emitted_ttgir_sha256=hashlib.sha256(self._text.encode()).hexdigest(),
                    complete_typed_config_sha256=self.digest(self._info),ordered_observer_trace_sha256=self.digest(self._events),
                    complete_observation_sha256=self._witness_seal,sources=dict(self._sources),input_files=dict(self._files),
                    stats=dict(self.stats),strong_proof_receipts={str(key):value[1] for key,value in self._proofs.items()},
                    snapshot_codec='private-sealed-memory-restricted-class-snapshot-v1',
                    access_row_receipts={key:value[1] for key,value in self._access_rows.items()},access_row_limit=256,
                    producer='emission_audit_session',private_view_replacements={
                     'demand_coverage_audit.audit':['backend','access_coverage (unchanged single-operation helper)'],
                     'native_surviving_audit.audit':['backend','emission'],
                     'execution_support.assess':['function-local backend.emit import only']},
                    original_audit_code_objects=True,negative_proofs_cached=False,persistent=False)
