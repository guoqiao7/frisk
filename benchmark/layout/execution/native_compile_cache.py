"""Exact CPU compilation batching; independent of numerical/GPU execution.

One CompilerSession is one explicit cache writer. Raw cache entries are immutable
and keyed by exact TTGIR/configuration plus the complete compilation context.
A cache miss means a Triton compile API call; its internal ordinary cache hit is
not exposed and is never invented here. Resource plans are rebuilt per execution.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from native_static_support import check_static_shared

ROOT=Path(__file__).resolve().parents[3]
SCHEMA=1
CONTRACT='frisk-exact-native-compile-cache-v1'
IMPLEMENTATION_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TARGET={'backend':'cuda','arch':90,'warp_size':32}
OPTIONS={'num_warps':4,'enable_fp_fusion':False}
NATIVE_REQUIRED={'kernel.ttgir','execution_config.json','kernel.ptx','kernel.cubin'}
NATIVE_OPTIONAL={'kernel.llir'}
EMISSION_INPUTS={'emitted.ttgir','emitted-config.json'}
RESOURCE_FILES={'resource-plan-before-native.json','resource-pressure-ptx-only.json','resolved-compiler-resources.json'}
RECEIPT='compile-cache-receipt.json'
BACKEND_FILES=('benchmark/layout/execution/backend.py','benchmark/layout/execution/redistribution.py',
               'benchmark/layout/execution/shared_allocation.py','benchmark/layout/execution/shuffle_network.py',
               'benchmark/layout/execution/execution_support.py','benchmark/layout/execution/tiled_execution.py',
               'benchmark/layout/execution/storage_addresses.py',
               'benchmark/layout/execution/resource_prediction.py',
               'benchmark/layout/execution/register_pressure.py','benchmark/layout/execution/native_compile_cache.py',
               'benchmark/layout/execution/native_static_support.py',
               'benchmark/layout/execution/access_plan.py','benchmark/layout/execution/reduction_plan.py',
               'benchmark/layout/execution/execution_schedule.py','benchmark/layout/execution/execution_plan_cost.py')


def _canonical(x):
    try:return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    except (TypeError,ValueError) as error:raise ValueError('finite JSON compilation identity required') from error


def _copy(x):return json.loads(_canonical(x))
def _sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _digest(x):return hashlib.sha256(_canonical(x)).hexdigest()
def _read(p):return json.loads(Path(p).read_text())
def _write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))


def _safe(path,*,directory=False,create=False):
    p=Path(path).absolute()
    if any(q.is_symlink() for q in [p,*p.parents]):raise ValueError('symlink compilation path rejected')
    if create:p.mkdir(parents=True,exist_ok=True)
    if directory and not p.is_dir():raise ValueError('existing compilation directory required')
    if not directory and not p.is_file():raise ValueError('existing regular compilation file required')
    return p


def _manifest(directory,names):
    return {name:dict(bytes=_safe(directory/name).stat().st_size,sha256=_sha(directory/name)) for name in sorted(names)}


def _verify_manifest(directory,manifest):
    if not isinstance(manifest,dict):raise ValueError('artifact manifest missing')
    for name,item in manifest.items():
        if name not in NATIVE_REQUIRED|NATIVE_OPTIONAL|RESOURCE_FILES|EMISSION_INPUTS|{'compile.json'}:raise ValueError('unexpected compile artifact name')
        p=_safe(directory/name)
        if not isinstance(item,dict) or set(item)!={'bytes','sha256'} or type(item['bytes']) is not int or p.stat().st_size!=item['bytes'] or _sha(p)!=item['sha256']:
            raise ValueError('compilation artifact integrity differs: '+name)


def _source_check(source,*,force=False):
    if not isinstance(source,dict) or not isinstance(source.get('source_manifest'),dict):raise ValueError('source snapshot missing')
    files=source['source_manifest']
    if source.get('source_digest')!=_digest(files):raise ValueError('source snapshot digest differs')
    for name,digest in files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source manifest path')
        p=_safe(ROOT/rel)
        from integrity_session import active, verified_sha
        actual=verified_sha(p,digest,force=force) if active() is not None else _sha(p)
        if actual!=digest:raise ValueError('registered source changed: '+name)


def _tool_check(tools,*,force=False):
    if not isinstance(tools,dict) or not isinstance(tools.get('hashes'),dict) or not tools['hashes']:raise ValueError('tool snapshot missing')
    for name,digest in tools['hashes'].items():
        from integrity_session import active, verified_sha
        if not Path(name).is_absolute():raise ValueError('registered tool changed: '+name)
        actual=verified_sha(_safe(name),digest,force=force) if active() is not None else _sha(_safe(name))
        if actual!=digest:raise ValueError('registered tool changed: '+name)


def compilation_environment():
    """Only actual compiler settings, never unrelated credentials/environment."""
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton import knobs
    names=('CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','LD_LIBRARY_PATH','LD_PRELOAD','CC',
        'TRITON_KERNEL_OVERRIDE','TRITON_KERNEL_DUMP','TRITON_STORE_BINARY_ONLY','TRITON_ALWAYS_COMPILE',
        'USE_IR_LOC','TRITON_ENABLE_ASAN','TRITON_DISABLE_LINE_INFO','TRITON_FRONT_END_DEBUGGING',
        'TRITON_ALLOW_NON_CONSTEXPR_GLOBALS','TRITON_OVERRIDE_ARCH','TRITON_F32_DEFAULT','TRITON_DEFAULT_FP_FUSION',
        'NVPTX_ENABLE_DUMP','DISABLE_PTXAS_OPT','TRITON_MOCK_PTX_VERSION','TRITON_LIBDEVICE_PATH',
        'TRITON_LIBCUDA_PATH','TRITON_PTXAS_PATH','TRITON_PTXAS_BLACKWELL_PATH','TRITON_CACHE_DIR',
        'TRITON_HOME','TRITON_OVERRIDE_DIR','TRITON_DUMP_DIR','TRITON_CACHE_MANAGER','TRITON_REMOTE_CACHE_BACKEND')
    values={name:os.environ.get(name) for name in names}
    if values['CUDA_MODULE_LOADING'] is None:values['CUDA_MODULE_LOADING']='LAZY'
    # Arbitrary override files cannot be represented by a few environment flags.
    if knobs.compilation.override:raise ValueError('native compile cache requires IR override disabled')
    return dict(environment=values,triton_cache_invalidating=_copy(get_cache_invalidating_env_vars()),
        effective_knobs=dict(override=knobs.compilation.override,enable_asan=knobs.compilation.enable_asan,
            disable_line_info=knobs.compilation.disable_line_info,use_ir_loc=knobs.compilation.use_ir_loc,
            disable_ptxas_opt=knobs.nvidia.disable_ptxas_opt,mock_ptx_version=knobs.nvidia.mock_ptx_version,
            override_arch=knobs.runtime.override_arch,default_fp_fusion=knobs.language.default_fp_fusion))


def _backend_identity():
    if _sha(Path(__file__))!=IMPLEMENTATION_SHA256:raise ValueError('compiler cache implementation changed since import')
    return {name:_sha(_safe(ROOT/name)) for name in BACKEND_FILES}


def _production_payload_paths():
    import triton
    from triton import knobs
    base=Path(triton.__file__).resolve().parent
    files={str(p.resolve()) for p in base.rglob('*.py') if p.is_file()}
    mathlib=Path(knobs.nvidia.libdevice_path) if knobs.nvidia.libdevice_path else base/'backends/nvidia/lib/libdevice.10.bc'
    if not mathlib.is_file():raise ValueError('Triton libdevice payload unavailable')
    files.add(str(mathlib.resolve()))
    if knobs.build.impl is not None:raise ValueError('custom Triton build implementation is outside this cache contract')
    return sorted(files)


def _production_provider_identity(*,force=False):
    import triton
    from integrity_session import active,verified_sha
    paths=_production_payload_paths()
    if active() is not None:
        active().inventory('Triton-Python-and-math-payload',paths,_production_payload_paths)
    files={name:(verified_sha(_safe(name),force=force) if active() is not None else _sha(_safe(name))) for name in paths}
    return dict(kind='installed-Triton-compile-API',triton_version=triton.__version__,python_and_math_payload_hashes=files)


def _large_identity_witness(context):
    """Metadata guard, never a replacement for the content hashes in the key.

    Re-list compiler payloads so additions/removals are detected too. ctime
    catches content writes even when a writer restores the original mtime.
    Full content verification brackets a session and any changed witness.
    Every newly compiled artifact additionally gets a full content recheck
    before publication: file metadata alone is insufficient on coarse clocks.
    """
    paths=set(context['tools']['hashes'])
    if context['mode']=='production':
        import triton
        identity=context['provider_identity']
        live=_production_payload_paths()
        if set(live)!=set(identity['python_and_math_payload_hashes']) or triton.__version__!=identity['triton_version']:
            raise ValueError('Triton compiler payload set/version changed during batch')
        paths.update(live)
    result={}
    for name in sorted(paths):
        st=_safe(name).stat()
        result[name]=(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    return result


def _providers(compiler,emitter,predictor,resolver,provider_identity):
    injected=any(x is not None for x in (compiler,emitter,predictor,resolver))
    if injected:
        if any(x is None for x in (compiler,emitter,predictor,resolver)) or not isinstance(provider_identity,dict) or not provider_identity:
            raise ValueError('injected compiler requires four explicit providers and a nonempty identity')
        return 'injected-test',compiler,emitter,predictor,resolver,_copy(provider_identity)
    if provider_identity is not None:raise ValueError('production provider identity is implementation-defined')
    import triton
    from triton.backends.compiler import GPUTarget
    from backend import emit
    from resource_prediction import predict_resources,resolve_resources
    def compile_native(path,target,options):
        result=triton.compile(str(path),target=GPUTarget(target['backend'],target['arch'],target['warp_size']),options=options)
        return {k:result.asm[k] for k in ('ptx','cubin','llir') if k in result.asm}
    return 'production',compile_native,emit,predict_resources,resolve_resources,_production_provider_identity()


def _context(*,source,tools,environment,target,options,mode,provider_identity):
    if source is None:
        from cost_calibration import source_identity
        source=source_identity()
    else:source=_copy(source)
    if tools is None:
        from support_closure import frozen_tools,OPT
        tools=frozen_tools(OPT)
    else:tools=_copy(tools)
    _source_check(source);_tool_check(tools)
    if environment is None:environment=compilation_environment()
    context=dict(schema_version=SCHEMA,contract=CONTRACT,mode=mode,provider_identity=provider_identity,
        source=source,tools=tools,environment=_copy(environment),target=_copy(TARGET if target is None else target),options=_copy(OPTIONS if options is None else options),
        backend_sha256=_backend_identity(),implementation_sha256=IMPLEMENTATION_SHA256)
    t=context['target'];o=context['options']
    if set(t)!={'backend','arch','warp_size'} or t['backend']!='cuda' or type(t['arch']) is not int or type(t['warp_size']) is not int or t['arch']<=0 or t['warp_size']<=0:
        raise ValueError('explicit CUDA target required')
    if mode=='production' and (t!=TARGET or o!=OPTIONS):raise ValueError('registered scalar backend requires exact SM90 options')
    return context


def _normalized_config(value):
    if not isinstance(value,dict):raise ValueError('full emitted configuration must be an object')
    def unique(pairs):
        result={}
        for key,item in pairs:
            if key in result:raise ValueError('emission configuration JSON key collision')
            result[key]=item
        return result
    # The wire contract is JSON. Backend diagnostics use integer SSA keys and
    # tuples, which the existing execution_config writer serializes to string
    # keys and arrays. Preserve that exact wire representation and key order.
    try:return json.loads(json.dumps(value,allow_nan=False),object_pairs_hook=unique)
    except (TypeError,ValueError) as error:raise ValueError('invalid full emission configuration JSON') from error


def _identity(context,ttgir,config):
    if not isinstance(ttgir,str) or not isinstance(config,dict):raise ValueError('emitter did not return exact text and full configuration')
    return dict(schema_version=SCHEMA,context=context,ttgir_sha256=hashlib.sha256(ttgir.encode()).hexdigest(),
                execution_config_sha256=_digest(config),execution_config=config)


def _validate_entry(entry,identity):
    entry=_safe(entry,directory=True);meta=_read(_safe(entry/'metadata.json'));key=_digest(identity)
    if set(meta)!={'schema_version','key','identity','artifacts'} or meta['schema_version']!=SCHEMA or meta['key']!=key or meta['identity']!=identity:
        raise ValueError('native cache identity mismatch')
    names=set(meta['artifacts'])
    if not NATIVE_REQUIRED<=names or names-NATIVE_REQUIRED-NATIVE_OPTIONAL:raise ValueError('native cache payload file list differs')
    actual={p.name for p in entry.iterdir()}
    if actual!=names|{'metadata.json'} or any(not p.is_file() or p.is_symlink() for p in entry.iterdir()):raise ValueError('native cache exact fileset differs')
    _verify_manifest(entry,meta['artifacts'])
    if _sha(entry/'kernel.ttgir')!=identity['ttgir_sha256'] or _read(entry/'execution_config.json')!=identity['execution_config']:
        raise ValueError('native cache emitted input differs')
    return meta


def _resource_semantics(value):
    """Drop cache/path timing metadata, retain independently parsed resource facts."""
    result=_copy(value)
    native=result.get('compiler_exact')
    if not isinstance(native,dict):raise ValueError('resolved compiler evidence missing')
    for name in ('cache_hit','cache_key','source_cubin_path','command'):native.pop(name,None)
    return result



# Optional secondary cache. Full-config identities remain the public cache key.
COMPILER_INPUT_CONTRACT='frisk-exact-compiler-input-cache-v1'
COMPILER_INPUT_REQUIRED={'kernel.ttgir','kernel.ptx','kernel.cubin'}

def _compiler_input_identity(context,ttgir):
    if not isinstance(ttgir,str):raise ValueError('exact compiler input text required')
    return dict(schema_version=SCHEMA,contract=COMPILER_INPUT_CONTRACT,context=context,
        ttgir_sha256=hashlib.sha256(ttgir.encode()).hexdigest())


def _validate_compiler_input(entry,identity):
    entry=_safe(entry,directory=True);meta=_read(_safe(entry/'metadata.json'));key=_digest(identity)
    if (not isinstance(meta,dict) or set(meta)!={'schema_version','contract','key','identity','artifacts'}
        or meta['schema_version']!=SCHEMA or meta['contract']!=COMPILER_INPUT_CONTRACT
        or meta['key']!=key or meta['identity']!=identity):raise ValueError('compiler-input cache identity differs')
    names=set(meta['artifacts'])
    if not COMPILER_INPUT_REQUIRED<=names or names-COMPILER_INPUT_REQUIRED-NATIVE_OPTIONAL:
        raise ValueError('compiler-input cache payload file list differs')
    if ({p.name for p in entry.iterdir()}!=names|{'metadata.json'}
        or any(not p.is_file() or p.is_symlink() for p in entry.iterdir())):
        raise ValueError('compiler-input cache exact fileset differs')
    _verify_manifest(entry,meta['artifacts'])
    if _sha(entry/'kernel.ttgir')!=identity['ttgir_sha256']:raise ValueError('compiler-input cached text differs')
    if not (entry/'kernel.cubin').read_bytes().startswith(b'\x7fELF'):raise ValueError('compiler-input CUBIN is malformed')
    return meta


def _publish_compiler_input(parent,identity,ttgir,asm):
    parent=_safe(parent,directory=True,create=True);key=_digest(identity);entry=parent/key
    if entry.exists() or entry.is_symlink():raise FileExistsError('immutable compiler-input destination appeared under writer lock')
    temporary=parent/('.partial-'+key+'-'+uuid.uuid4().hex);temporary.mkdir(exist_ok=False)
    (temporary/'kernel.ttgir').write_text(ttgir);(temporary/'kernel.ptx').write_text(asm['ptx']);(temporary/'kernel.cubin').write_bytes(asm['cubin'])
    if 'llir' in asm:(temporary/'kernel.llir').write_text(asm['llir'])
    names=COMPILER_INPUT_REQUIRED|({'kernel.llir'} if 'llir' in asm else set())
    meta=dict(schema_version=SCHEMA,contract=COMPILER_INPUT_CONTRACT,key=key,identity=identity,artifacts=_manifest(temporary,names))
    _write(temporary/'metadata.json',meta);_validate_compiler_input(temporary,identity)
    if entry.exists() or entry.is_symlink():raise FileExistsError('immutable compiler-input destination appeared before publication')
    temporary.rename(entry);return entry,_validate_compiler_input(entry,identity)


def _verify_compiler_input_provenance(directory,p,context,ttgir):
    proof=p.get('compiler_input_cache')
    if proof is None:
        if type(p.get('exact_cache_hit')) is not bool or p.get('compiler_api_called') is not (not p['exact_cache_hit']):
            raise ValueError('compile accounting requires full hit, proven compiler-input hit, or actual API call')
        return
    fields={'contract','identity','key','cache_entry','cache_metadata_sha256','compiler_input_hit'}
    identity=_compiler_input_identity(context,ttgir)
    if (not isinstance(proof,dict) or set(proof)!=fields or proof['contract']!=COMPILER_INPUT_CONTRACT
        or proof['identity']!=identity or proof['key']!=_digest(identity)
        or type(proof['compiler_input_hit']) is not bool or p['exact_cache_hit'] is not False
        or p['compiler_api_called'] is not (not proof['compiler_input_hit'])):
        raise ValueError('compiler-input provenance differs')
    entry=_safe(proof['cache_entry'],directory=True);meta=_validate_compiler_input(entry,identity)
    if _sha(entry/'metadata.json')!=proof['cache_metadata_sha256']:raise ValueError('compiler-input cache metadata changed')
    native_names=set(p['native_artifacts'])-{'execution_config.json'}
    if set(meta['artifacts'])!=native_names or any(meta['artifacts'][n]!=p['native_artifacts'][n] for n in native_names):
        raise ValueError('compiler-input and full-config native payload differ')
    _verify_manifest(directory,meta['artifacts'])


class CompilerSession:
    def __init__(self,cache_dir,*,source=None,tools=None,target=None,options=None,environment=None,
                 compiler=None,emitter=None,predictor=None,resolver=None,provider_identity=None,compiler_input_reuse=False):
        if type(compiler_input_reuse) is not bool:raise ValueError('compiler_input_reuse must be explicit bool')
        self.compiler_input_reuse=compiler_input_reuse;self.compiler_input_hits=0
        start=time.perf_counter_ns()
        mode,self.compiler,self.emitter,self.predictor,self.resolver,provider=_providers(compiler,emitter,predictor,resolver,provider_identity)
        self.context=_context(source=source,tools=tools,environment=environment,target=target,options=options,mode=mode,provider_identity=provider)
        self.large_identity_full_checks=0;self.large_identity_stat_checks=0
        self.large_identity_witness=None
        self._assert_large_identity(full=True)
        self.cache=_safe(cache_dir,directory=True,create=True);self.lock=self.cache/'.writer.lock';self.token=uuid.uuid4().hex
        fd=os.open(str(self.lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'w') as f:f.write(self.token)
        self.closed=False;self.rows=[];self.requests=0;self.hits=0;self.api_calls=0;self.api_ns=0
        self.initialization_ns=time.perf_counter_ns()-start
        self.environment_live=environment is None

    def __enter__(self):return self
    def __exit__(self,kind,value,tb):self.close()

    def _assert_large_identity(self,*,full=False):
        before=_large_identity_witness(self.context);self.large_identity_stat_checks+=1
        if full or before!=self.large_identity_witness:
            _tool_check(self.context['tools'],force=True)
            if self.context['mode']=='production' and _production_provider_identity(force=True)!=self.context['provider_identity']:
                raise ValueError('Triton Python/math implementation changed during batch')
            self.large_identity_full_checks+=1
            after=_large_identity_witness(self.context)
            if before!=after:raise ValueError('compiler tool payload changed while validating its content identity')
            self.large_identity_witness=after

    def _assert_current(self,*,full=False):
        _source_check(self.context['source'],force=full)
        if _backend_identity()!=self.context['backend_sha256']:raise ValueError('backend changed during compilation batch')
        if self.environment_live and compilation_environment()!=self.context['environment']:raise ValueError('compiler environment changed during batch')
        self._assert_large_identity(full=full)

    def close(self):
        if self.closed:return
        try:self._assert_current(full=True)
        finally:
            if self.lock.is_symlink() or not self.lock.is_file() or self.lock.read_text()!=self.token:
                raise ValueError('native cache writer ownership changed')
            self.lock.unlink();self.closed=True

    def compile(self,directory):
        if self.closed:raise ValueError('compiler session is closed')
        start=time.perf_counter_ns();directory=_safe(directory,directory=True)
        owned=NATIVE_REQUIRED|NATIVE_OPTIONAL|RESOURCE_FILES|{'compile.json',RECEIPT,'correctness.json'}
        if any((directory/name).exists() or (directory/name).is_symlink() for name in owned):raise FileExistsError('fresh batch compilation refuses overwrite')
        self._assert_current()  # Detect tool changes before invoking the compiler.
        execution_path=_safe(directory/'execution.json');execution_sha=_sha(execution_path);execution=_read(execution_path)
        ttgir,config=self.emitter(execution);config=_normalized_config(config)
        check_static_shared(config)  # Reject before compiler invocation, output writes or cache publication.
        identity=_identity(self.context,ttgir,config);key=_digest(identity);entry=self.cache/key
        generation=_manifest(directory,EMISSION_INPUTS)
        if _safe(directory/'emitted.ttgir').read_bytes()!=ttgir.encode() or _read(_safe(directory/'emitted-config.json'))!=config:
            raise ValueError('checked generation emission differs from current backend')
        self.requests+=1
        before=self.predictor(execution)
        _write(directory/'resource-plan-before-native.json',dict(prediction=before,information_stage='execution-export-before-native-compile-or-retrieval',created_ns=time.time_ns()))
        (directory/'kernel.ttgir').write_text(ttgir);_write(directory/'execution_config.json',config)
        hit=entry.exists();compiler_input_hit=False;compiler_input_proof=None
        if hit:
            meta=_validate_entry(entry,identity);self.hits+=1
        else:
            temporary=self.cache/('.partial-'+key+'-'+uuid.uuid4().hex);temporary.mkdir(exist_ok=False)
            ci_identity=_compiler_input_identity(self.context,ttgir) if self.compiler_input_reuse else None
            ci_parent=self.cache/'compiler-input';ci_entry=ci_parent/_digest(ci_identity) if ci_identity else None
            if ci_identity and (ci_entry.exists() or ci_entry.is_symlink()):
                ci_meta=_validate_compiler_input(ci_entry,ci_identity)
                # Bind the exact bytes we use, not only an earlier filesystem read.
                ci_payload={name:_safe(ci_entry/name).read_bytes() for name in ci_meta['artifacts']}
                actual={name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in ci_payload.items()}
                if actual!=ci_meta['artifacts']:raise ValueError('compiler-input payload changed after validation')
                asm={'ptx':ci_payload['kernel.ptx'].decode('utf-8'),'cubin':ci_payload['kernel.cubin']}
                if 'kernel.llir' in ci_payload:asm['llir']=ci_payload['kernel.llir'].decode('utf-8')
                compiler_input_hit=True;self.compiler_input_hits+=1
            else:
                api_start=time.perf_counter_ns();self.api_calls+=1
                try:asm=self.compiler(directory/'kernel.ttgir',_copy(self.context['target']),_copy(self.context['options']))
                finally:self.api_ns+=time.perf_counter_ns()-api_start
            if not isinstance(asm,dict) or not {'ptx','cubin'}<=set(asm) or set(asm)-{'ptx','cubin','llir'}:
                raise ValueError('compiler did not return the registered native artifact set')
            if not isinstance(asm['ptx'],str) or not isinstance(asm['cubin'],bytes) or not asm['cubin'].startswith(b'\x7fELF'):
                raise ValueError('compiler native artifacts malformed')
            if 'llir' in asm and not isinstance(asm['llir'],str):raise ValueError('compiler LLIR malformed')
            if (_sha(execution_path)!=execution_sha or (directory/'kernel.ttgir').read_bytes()!=ttgir.encode() or
                _read(directory/'execution_config.json')!=config):
                raise ValueError('compiler changed its owned emitted inputs')
            self._assert_current(full=True)  # Full content check before every native publication, even on coarse-timestamp filesystems.
            if ci_identity:
                if not compiler_input_hit:ci_entry,ci_meta=_publish_compiler_input(ci_parent,ci_identity,ttgir,asm)
                compiler_input_proof=dict(contract=COMPILER_INPUT_CONTRACT,identity=ci_identity,key=_digest(ci_identity),
                    cache_entry=str(ci_entry),cache_metadata_sha256=_sha(ci_entry/'metadata.json'),compiler_input_hit=compiler_input_hit)
            (temporary/'kernel.ttgir').write_text(ttgir);_write(temporary/'execution_config.json',config)
            (temporary/'kernel.ptx').write_text(asm['ptx']);(temporary/'kernel.cubin').write_bytes(asm['cubin'])
            if 'llir' in asm:(temporary/'kernel.llir').write_text(asm['llir'])
            native_names=NATIVE_REQUIRED|({'kernel.llir'} if 'llir' in asm else set())
            meta=dict(schema_version=SCHEMA,key=key,identity=identity,artifacts=_manifest(temporary,native_names))
            if ci_identity and {name:item for name,item in meta['artifacts'].items() if name!='execution_config.json'}!=ci_meta['artifacts']:
                raise ValueError('compiler-input and full-config payload differ before publication')
            _write(temporary/'metadata.json',meta);_validate_entry(temporary,identity)
            if entry.exists():raise FileExistsError('immutable cache destination appeared while writer held lock')
            temporary.rename(entry);_validate_entry(entry,identity)
        for name in meta['artifacts']:
            if name not in ('kernel.ttgir','execution_config.json'):
                target=directory/name
                if target.exists() or target.is_symlink():raise FileExistsError('native output appeared during compilation')
                shutil.copyfile(entry/name,target)
        _verify_manifest(directory,meta['artifacts'])
        if _sha(execution_path)!=execution_sha:raise ValueError('execution changed during native compilation')
        if _backend_identity()!=self.context['backend_sha256']:raise ValueError('backend changed during compilation')
        ptx=(directory/'kernel.ptx').read_text();pressure=self.predictor(execution,ptx)
        _write(directory/'resource-pressure-ptx-only.json',dict(prediction=pressure,information_stage='PTX-only-pre-RA-information',
            wallclock_stage='after-native-compile-or-exact-cache-retrieval',created_ns=time.time_ns(),reads_native_resources=False))
        resolved=self.resolver(execution,ptx,directory/'kernel.cubin',cache_dir=self.cache/'resource-feedback-cache',expected_cubin_sha256=_sha(directory/'kernel.cubin'))
        _write(directory/'resolved-compiler-resources.json',resolved)
        result=dict(wall_ns=time.perf_counter_ns()-start,cubin_sha256=_sha(directory/'kernel.cubin'),execution_sha256=execution_sha,
            source=self.context['source'],tools=self.context['tools'],tool_identity_boundary='full tool/payload contents at session entry/exit and before native publication; cache-hit file identity and payload-set guard per request; worker fully rechecks receipt before loading',
            cache='exact TTGIR+full-config+context cache; ordinary Triton internal cache hit unknown',
            compile_cache=dict(contract=CONTRACT,key=key,exact_cache_hit=hit,compiler_api_called=not hit and not compiler_input_hit,compiler_internal_cache_hit=None),
            session_context_sha256=_digest(self.context))
        if self.compiler_input_reuse:
            result['compile_cache'].update(exact_full_hit=hit,compiler_input_hit=compiler_input_hit,compiler_input_contract=COMPILER_INPUT_CONTRACT)
        _write(directory/'compile.json',result)
        names=set(meta['artifacts'])|RESOURCE_FILES|{'compile.json'}
        payload=dict(schema_version=SCHEMA,contract=CONTRACT,context=self.context,context_sha256=_digest(self.context),
            execution_sha256=execution_sha,generation_emission=generation,identity=identity,key=key,cache_entry=str(entry),cache_metadata_sha256=_sha(entry/'metadata.json'),
            native_artifacts=meta['artifacts'],artifacts=_manifest(directory,names),exact_cache_hit=hit,
            compiler_api_called=not hit and not compiler_input_hit,compiler_internal_cache_hit=None)
        if compiler_input_proof is not None:payload['compiler_input_cache']=compiler_input_proof
        _write(directory/RECEIPT,dict(payload=payload,payload_sha256=_digest(payload)))
        self.rows.append(dict(path=str(directory),key=key,exact_cache_hit=hit,compiler_api_called=not hit and not compiler_input_hit,compile_wall_ns=result['wall_ns'],receipt_sha256=_sha(directory/RECEIPT)))
        if self.compiler_input_reuse:self.rows[-1].update(exact_full_hit=hit,compiler_input_hit=compiler_input_hit)
        return result

    def summary(self):
        result=dict(schema_version=SCHEMA,contract=CONTRACT,context_sha256=_digest(self.context),mode=self.context['mode'],
            requests=self.requests,exact_cache_hits=self.hits,compiler_api_calls=self.api_calls,
            compiler_api_wall_ns=self.api_ns,session_initialization_wall_ns=self.initialization_ns,
            large_identity_full_checks=self.large_identity_full_checks,large_identity_stat_checks=self.large_identity_stat_checks,
            tool_integrity_policy='full content hashes at entry/exit and native publication; full recheck on any per-request identity change',
            compiler_internal_cache_hit=None,cache_interpretation='API calls may themselves hit ordinary Triton cache; never a cold-compile claim',rows=_copy(self.rows))
        if self.compiler_input_reuse:result.update(compiler_input_reuse=True,compiler_input_hits=self.compiler_input_hits,exact_full_hits=self.hits)
        return result


def compile_many(directories,cache_dir,**kwargs):
    with CompilerSession(cache_dir,**kwargs) as session:
        for directory in directories:session.compile(directory)
    return dict(status='complete',**session.summary())


def verify_precompiled(directory,*,expected_source=None,expected_tools=None,expected_environment=None,
                       allow_injected=False,emitter=None,predictor=None,resolver=None,provider_identity=None):
    """Verify every receipt/artifact and current execution resources, never compile.

    Expected source/tools may come from the caller's already-verified frozen
    registration. With no expectations, live identities are obtained once here.
    Injection is explicit test mode and rejected by the production default.
    """
    directory=_safe(directory,directory=True);envelope=_read(_safe(directory/RECEIPT))
    if not isinstance(envelope,dict) or set(envelope)!={'payload','payload_sha256'} or _digest(envelope['payload'])!=envelope['payload_sha256']:
        raise ValueError('compile receipt payload digest differs')
    p=envelope['payload']
    if not isinstance(p,dict) or not isinstance(p.get('context'),dict):raise ValueError('compile receipt payload/context missing')
    context=p['context']
    if p.get('schema_version')!=SCHEMA or p.get('contract')!=CONTRACT or p.get('context_sha256')!=_digest(context):raise ValueError('compile receipt schema/context differs')
    injected=context.get('mode')=='injected-test'
    if injected:
        if not allow_injected or any(x is None for x in (emitter,predictor,resolver)) or provider_identity!=context['provider_identity']:
            raise ValueError('production verification rejects injected compiler artifacts')
    else:
        if context.get('mode')!='production' or any(x is not None for x in (emitter,predictor,resolver,provider_identity)):
            raise ValueError('production provider contract differs')
        from backend import emit
        from resource_prediction import predict_resources,resolve_resources
        emitter=emit;predictor=predict_resources;resolver=resolve_resources
        if context['target']!=TARGET or context['options']!=OPTIONS:raise ValueError('precompiled target/options differ')
        if context['provider_identity']!=_production_provider_identity():raise ValueError('precompiled Triton Python/math payload differs')
    if expected_source is None:
        if injected:expected_source=context['source'];_source_check(expected_source)
        else:
            from cost_calibration import source_identity
            expected_source=source_identity()
    if expected_tools is None:
        if injected:expected_tools=context['tools'];_tool_check(expected_tools)
        else:
            from support_closure import frozen_tools,OPT
            expected_tools=frozen_tools(OPT)
    if expected_environment is None:expected_environment=context['environment'] if injected else compilation_environment()
    _source_check(expected_source);_tool_check(expected_tools)
    if context['source']!=expected_source or context['tools']!=expected_tools or context['environment']!=expected_environment:
        raise ValueError('precompiled source/tool/environment context differs')
    if context.get('implementation_sha256')!=IMPLEMENTATION_SHA256 or context.get('backend_sha256')!=_backend_identity():
        raise ValueError('precompiled backend implementation changed')
    execution_path=_safe(directory/'execution.json')
    if _sha(execution_path)!=p['execution_sha256']:raise ValueError('precompiled execution identity differs')
    if not isinstance(p.get('generation_emission'),dict) or set(p['generation_emission'])!=EMISSION_INPUTS:
        raise ValueError('precompiled checked-emission receipt missing')
    _verify_manifest(directory,p['generation_emission'])
    execution=_read(execution_path);ttgir,config=emitter(execution);config=_normalized_config(config)
    check_static_shared(config)  # A valid artifact hash cannot expand this backend's support domain.
    identity=_identity(context,ttgir,config)
    if identity!=p['identity'] or _digest(identity)!=p['key']:raise ValueError('precompiled emitted identity differs')
    names=set(p['native_artifacts'])
    if not NATIVE_REQUIRED<=names or names-NATIVE_REQUIRED-NATIVE_OPTIONAL:raise ValueError('precompiled native payload set differs')
    if set(p['artifacts'])!=names|RESOURCE_FILES|{'compile.json'}:raise ValueError('precompiled artifact manifest set differs')
    _verify_manifest(directory,p['native_artifacts']);_verify_manifest(directory,p['artifacts'])
    if (directory/'kernel.ttgir').read_bytes()!=ttgir.encode() or _read(directory/'execution_config.json')!=config:
        raise ValueError('precompiled bytes differ from current emitted execution')
    if (directory/'emitted.ttgir').exists() and _safe(directory/'emitted.ttgir').read_bytes()!=ttgir.encode():raise ValueError('generation emitted text differs')
    if (directory/'emitted-config.json').exists() and _read(_safe(directory/'emitted-config.json'))!=config:raise ValueError('generation emitted configuration differs')
    if ('kernel.llir' in names)!=((directory/'kernel.llir').exists() or (directory/'kernel.llir').is_symlink()):raise ValueError('unexpected/missing optional LLVM artifact')
    before=_read(directory/'resource-plan-before-native.json');pressure=_read(directory/'resource-pressure-ptx-only.json')
    ptx=(directory/'kernel.ptx').read_text()
    if before['prediction']!=predictor(execution) or pressure['prediction']!=predictor(execution,ptx):raise ValueError('precompiled resource plans differ from current execution')
    current=resolver(execution,ptx,directory/'kernel.cubin',cache_dir=directory.parents[1]/'resource-feedback-cache',expected_cubin_sha256=_sha(directory/'kernel.cubin'))
    if _resource_semantics(current)!=_resource_semantics(_read(directory/'resolved-compiler-resources.json')):
        raise ValueError('precompiled native resource facts differ')
    _verify_compiler_input_provenance(directory,p,context,ttgir)
    compiled=_read(directory/'compile.json')
    if compiled['execution_sha256']!=p['execution_sha256'] or compiled['cubin_sha256']!=_sha(directory/'kernel.cubin') or compiled['session_context_sha256']!=p['context_sha256']:
        raise ValueError('compile metadata identity differs')
    counts=compiled['compile_cache']
    if counts.get('exact_cache_hit') is not p['exact_cache_hit'] or counts.get('compiler_api_called') is not p['compiler_api_called']:
        raise ValueError('compile accounting receipt differs')
    if 'compiler_input_hit' in counts:
        actual_ci=p.get('compiler_input_cache',{}).get('compiler_input_hit',False)
        if counts.get('exact_full_hit') is not p['exact_cache_hit'] or counts.get('compiler_input_hit') is not actual_ci or counts.get('compiler_input_contract')!=COMPILER_INPUT_CONTRACT:
            raise ValueError('compiler-input accounting receipt differs')
    return dict(status='verified-precompiled',contract=CONTRACT,key=p['key'],receipt_sha256=_sha(directory/RECEIPT),
        execution_sha256=p['execution_sha256'],cubin_sha256=_sha(directory/'kernel.cubin'),
        ttgir_sha256=identity['ttgir_sha256'],execution_config_sha256=identity['execution_config_sha256'],
        generation_emission=p['generation_emission'],context_sha256=p['context_sha256'],target=context['target'],options=context['options'],
        exact_cache_hit=p['exact_cache_hit'],compiler_api_called=p['compiler_api_called'],verification_compiled=False,
        resources_reconfirmed_for_current_execution=True,provider_mode=context['mode'])
