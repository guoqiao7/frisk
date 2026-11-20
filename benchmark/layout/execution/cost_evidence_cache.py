"""Private content-addressed CPU evidence and exact positive-proof proposals.

Every lookup hashes actual inputs, source and tools; metadata is used only to
reject concurrent mutation. Cached tool stdout is trusted user-owned local
compiler evidence, just as the existing native-analysis cache: it is not an
independent proof that nvdisasm implements CUDA semantics. This is not a GPU,
ABI, correctness or compiler-cache compatibility certificate.
"""
from __future__ import annotations
import base64
from collections import Counter
import hashlib
import inspect
import json
import types
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import time
import uuid

CONTRACT = 'frisk-private-cost-evidence-cache-v1'
SOURCE = Path(__file__).resolve()
IMPLEMENTATION_SHA256 = hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _code_value(value):
    """Complete typed code constants; frozensets have semantic set order."""
    kind=type(value)
    if kind is types.CodeType:
        fields={name:_code_value(getattr(value,name)) for name in dir(value)
                if name.startswith('co_') and not callable(getattr(value,name))}
        return dict(type='code',fields=fields)
    if value is None:return dict(type='NoneType')
    if value is Ellipsis:return dict(type='ellipsis')
    if kind in (bool,int,str):return dict(type=kind.__name__,value=value)
    if kind is float:return dict(type='float',ieee754=struct.pack('>d',value).hex())
    if kind is complex:return dict(type='complex',real=struct.pack('>d',value.real).hex(),imag=struct.pack('>d',value.imag).hex())
    if kind is bytes:return dict(type='bytes',hex=value.hex())
    if kind is tuple:return dict(type='tuple',items=[_code_value(x) for x in value])
    if kind is frozenset:
        return dict(type='frozenset',items=sorted((_code_value(x) for x in value),key=canonical))
    raise ValueError('unsupported code identity value: '+kind.__name__)


def code_fingerprint(code):
    """Hash every noncallable co_* field, independently of hash randomization.

    marshal serializes frozenset constants in hash-dependent iteration order.
    Preserve every code field, nested code object and typed constant while
    sorting only the mathematically unordered frozenset members.
    """
    if type(code) is not types.CodeType:raise ValueError('code object required')
    return digest(dict(contract='frisk-complete-stable-code-fingerprint-v1',code=_code_value(code)))


def snapshot(path):
    path = Path(path).absolute()
    if path.resolve() != path or path.is_symlink():
        raise ValueError('unaliased regular input required: ' + str(path))
    before = path.stat(); h = hashlib.sha256()
    if not stat.S_ISREG(before.st_mode): raise ValueError('regular input required')
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    after = path.stat()
    fields = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if fields(before) != fields(after): raise ValueError('input changed while hashing: ' + str(path))
    return dict(path=str(path),sha256=h.hexdigest(),bytes=after.st_size)


CPU_ENVIRONMENT_CONTRACT = 'frisk-controlled-cpu-cost-extraction-environment-v1'
CPU_ENVIRONMENT_NAMES = frozenset(('PATH','HOME','USER','LOGNAME','SHELL','BASH_ENV','ENV',
    'SHELLOPTS','BASHOPTS','IFS','LANG','LANGUAGE','TZ','LOCPATH','GCONV_PATH',
    'GLIBC_TUNABLES','TMPDIR','TMP','TEMP','XDG_CACHE_HOME','XDG_CONFIG_HOME',
    'XDG_DATA_HOME','XDG_RUNTIME_DIR','GOTO_NUM_THREADS','ASAN_OPTIONS',
    'LSAN_OPTIONS','TSAN_OPTIONS','UBSAN_OPTIONS'))
CPU_ENVIRONMENT_PREFIXES = ('LC_','LD_','PYTHON','OMP_','OPENMP_','OPENBLAS_',
    'MKL_','BLIS_','VECLIB_','NUMEXPR_','KMP_','TBB_','MALLOC_','CUDA_',
    'NVIDIA_','NV_','TRITON_','FRISK_')


def cpu_environment():
    """Complete environment actually forwarded to the read-only CPU tools.

    Pure extraction modules do not read other environment names; Python/hash,
    locale, dynamic-loader and numerical-thread settings remain bound. SSH
    transport bookkeeping is neither forwarded nor part of semantic identity.
    Changes to this explicit contract are themselves source-key changes.
    """
    return {name:value for name,value in os.environ.items()
            if name in CPU_ENVIRONMENT_NAMES or name.startswith(CPU_ENVIRONMENT_PREFIXES)}


def environment():
    # Record no potentially secret variable values in public receipts.
    return digest(dict(contract=CPU_ENVIRONMENT_CONTRACT,variables=cpu_environment()))


def tool_runtime(path):
    """Bind load-time ELF dependencies, not randomized ldd addresses."""
    loader=Path('/usr/bin/ldd').resolve(strict=True)
    result=subprocess.run([str(loader),str(path)],capture_output=True,text=True,timeout=30,env=cpu_environment())
    if result.returncode:raise ValueError('cannot certify native tool load-time dependencies')
    libraries={}
    for line in result.stdout.splitlines():
        for word in line.split():
            if word.startswith('/'):
                target=Path(word).resolve(strict=True)
                libraries[word]=snapshot(target)
    return dict(loader=snapshot(loader),libraries=libraries)


class PrivateStore:
    def __init__(self, root):
        raw = Path(root).absolute()
        if raw.resolve() != raw: raise ValueError('unaliased cache directory required')
        raw.mkdir(mode=0o700, parents=True, exist_ok=True)
        mode = raw.stat()
        if raw.is_symlink() or mode.st_uid != os.geteuid() or mode.st_mode & 0o077:
            raise ValueError('cache must be private to the current user')
        self.root=raw; self.stats=Counter()

    def path(self, kind, identity):
        if not kind.replace('-', '').isalnum(): raise ValueError('fixed cache namespace required')
        directory=self.root/kind; directory.mkdir(mode=0o700,exist_ok=True)
        if directory.is_symlink() or directory.stat().st_uid!=os.geteuid() or directory.stat().st_mode & 0o077:
            raise ValueError('unsafe cache namespace')
        return directory/(digest(identity)+'.json')

    def load(self, kind, identity):
        path=self.path(kind,identity)
        if not path.exists(): self.stats[kind+'.misses']+=1; return None
        info=path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.is_symlink() or info.st_uid!=os.geteuid() or info.st_nlink!=1 or info.st_mode & 0o077:
            raise ValueError('unsafe cache object')
        try:
            if info.st_size>256*1024*1024: raise ValueError('cache object limit')
            raw=path.read_bytes(); envelope=json.loads(raw)
            if set(envelope)!={'contract','identity','payload','payload_sha256'} or envelope['contract']!=CONTRACT or envelope['identity']!=identity or envelope['payload_sha256']!=digest(envelope['payload']):
                raise ValueError('cache content/identity mismatch')
            self.stats[kind+'.hits']+=1; self.stats['cache_read_bytes']+=len(raw)
            return envelope['payload']
        except (ValueError,KeyError,TypeError):
            self.stats[kind+'.corrupt']+=1
            # Preserve corrupt regular bytes for audit, without following links.
            bad=self.root/'invalid';bad.mkdir(mode=0o700,exist_ok=True)
            if bad.is_symlink() or bad.stat().st_mode & 0o077:raise ValueError('unsafe invalid cache directory')
            target=bad/(path.stem+'-'+uuid.uuid4().hex+'.json')
            descriptor=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'wb') as stream:stream.write(path.read_bytes())
            return None

    def save(self, kind, identity, payload):
        path=self.path(kind,identity);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        raw=canonical(dict(contract=CONTRACT,identity=identity,payload=payload,payload_sha256=digest(payload)))+b'\n'
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            with os.fdopen(descriptor,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()
        self.stats['cache_write_bytes']+=len(raw)


def _native_with_provider(directory, cache_dir=None, *, entry='frisk_execution', nvdisasm='/usr/local/cuda/bin/nvdisasm', cuobjdump='/usr/local/cuda/bin/cuobjdump', document_provider=None):
    """Read kernel.cubin; optionally cache feedback bound to image+tools+source.

    Cache hits are verified by a payload digest and exact identity. Compilation
    is the caller's budgeted operation; no new image is compiled here. Failed
    tools/unsupported schedules return typed failure with no scoreable features.
    """
    import native_cost_analysis as _native
    source = Path(directory)
    cubin = source / 'kernel.cubin' if source.is_dir() else source
    identity = {'schema_version': _native.SCHEMA, 'domain': _native.DOMAIN, 'entry': entry, 'cubin_sha256': _native._sha(cubin.read_bytes()), 'analyzer_sha256': _native.ANALYZER_SOURCE_SHA256}
    if _native._sha(Path(_native.__file__).read_bytes()) != _native.ANALYZER_SOURCE_SHA256:
        return {'schema_version': _native.SCHEMA, 'status': 'unavailable', 'features': None, 'identity': identity, 'support': {'status': 'unsupported', 'domain': _native.DOMAIN, 'reasons': ['analyzer-source-changed-since-import']}}
    try:
        identity['nvdisasm'] = _native._tool(nvdisasm) if document_provider is None else document_provider.tool(nvdisasm)
        identity['cuobjdump'] = _native._tool(cuobjdump) if document_provider is None else document_provider.tool(cuobjdump)
    except (OSError, subprocess.SubprocessError) as error:
        return {'schema_version': _native.SCHEMA, 'status': 'unavailable', 'features': None, 'identity': identity, 'support': {'status': 'unsupported', 'domain': _native.DOMAIN, 'reasons': ['native-tool-unavailable:' + type(error).__name__]}}
    key = _native._sha(_native._canonical(identity))
    cache = Path(cache_dir) / (key + '.json') if cache_dir is not None else None
    if cache is not None and cache.is_file():
        try:
            envelope = json.loads(cache.read_text())
            payload = envelope['payload']
            if payload['identity'] == identity and envelope['payload_sha256'] == _native._sha(_native._canonical(payload)):
                return dict(payload, cache_key=key, cache_hit=True)
        except (ValueError, KeyError, TypeError):
            pass
    try:
        if document_provider is None:
            native = subprocess.check_output([identity['nvdisasm']['path'], '-json', str(cubin)], text=True, stderr=subprocess.PIPE, timeout=60)
            resources = subprocess.check_output([identity['cuobjdump']['path'], '--dump-resource-usage', str(cubin)], text=True, stderr=subprocess.PIPE, timeout=60)
        else:
            native = document_provider.output(identity['nvdisasm'], ['-json', str(cubin)], cubin).decode()
            resources = document_provider.output(identity['cuobjdump'], ['--dump-resource-usage', str(cubin)], cubin).decode()
        result = _native.analyze_disassembly(json.loads(native), resources, entry) if document_provider is None else document_provider.pure(_native.analyze_disassembly, json.loads(native), resources, entry)
        result.update({'native_json_sha256': _native._sha(native.encode()), 'resource_text_sha256': _native._sha(resources.encode())})
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        result = {'status': 'unavailable', 'features': None, 'support': {'status': 'unsupported', 'domain': _native.DOMAIN, 'reasons': ['native-analysis-failed:' + type(error).__name__]}}
    if _native._sha(cubin.read_bytes()) != identity['cubin_sha256'] or any((_native._sha(Path(identity[t]['path']).read_bytes()) != identity[t]['sha256'] for t in ('nvdisasm', 'cuobjdump'))):
        result = {'status': 'unavailable', 'features': None, 'support': {'status': 'unsupported', 'domain': _native.DOMAIN, 'reasons': ['native-input-changed-during-analysis']}}
    payload = dict(result, schema_version=_native.SCHEMA, identity=identity)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + '.tmp')
        temporary.write_text(json.dumps({'payload': payload, 'payload_sha256': _native._sha(_native._canonical(payload))}, indent=2, sort_keys=True) + '\n')
        temporary.replace(cache)
    return dict(payload, cache_key=key, cache_hit=False)


class CostEvidenceCache:
    """One API for cold/shared native decoding and verified warm extraction."""
    INPUTS = {
        'native': ('kernel.cubin',),
        'summary': ('kernel.cubin','execution.json','kernel.ttgir','kernel.ptx','execution_config.json','compile.json','image-identity.json'),
        'locality': ('execution.json','record.json','kernel.ttgir','kernel.cubin','image-identity.json','compile.json','emitted-config.json','execution_config.json','surviving-coverage-audit.json'),
        'intervention': ('kernel.ptx','kernel.cubin','correctness.json','image-identity.json','native-analysis.json','execution_config.json'),
    }

    def __init__(self, root, *, nvdisasm='/usr/local/cuda/bin/nvdisasm', cuobjdump='/usr/local/cuda/bin/cuobjdump'):
        self.store=PrivateStore(root);self.nvdisasm=nvdisasm;self.cuobjdump=cuobjdump

    @property
    def stats(self):return dict(self.store.stats)

    def sources(self):
        # Bind loaded producer code, rather than relabeling it with newer disk
        # bytes after an in-process source edit.
        if snapshot(SOURCE)['sha256']!=IMPLEMENTATION_SHA256:
            raise ValueError('cache-source-changed-since-import')
        # Conservative transitive Python closure. No stat-only content reuse.
        return {p.name:snapshot(p)['sha256'] for p in sorted(SOURCE.parent.glob('*.py'))}

    def tool(self, path):
        path=Path(path).resolve(strict=True);state=snapshot(path)
        identity=dict(contract=CONTRACT,kind='tool-version',tool=state,runtime=tool_runtime(path),environment_sha256=environment())
        version=self.store.load('toolversion',identity)
        if version is None:
            self.store.stats['tool_version_processes']+=1
            version=subprocess.check_output([str(path),'--version'],text=True,timeout=30,env=cpu_environment())
            if snapshot(path)!=state:raise ValueError('tool changed during version query')
            self.store.save('toolversion',identity,version)
        if snapshot(path)!=state or environment()!=identity['environment_sha256'] or tool_runtime(path)!=identity['runtime']:
            raise ValueError('tool runtime changed during identity query')
        return dict(path=str(path),sha256=state['sha256'],version=version)

    def decoder_image(self, cubin, image):
        """Make the decoder pathname itself content-addressed, never assumed inert.

        Tool invocations are executed on this exact private pathname on misses;
        their cached stdout key therefore still binds the actual complete argv.
        Source input bytes/path and current provenance are checked separately.
        """
        directory=self.store.root/'decoder-images';directory.mkdir(mode=0o700,exist_ok=True)
        info=directory.lstat()
        if directory.is_symlink() or info.st_uid!=os.geteuid() or info.st_mode & 0o077:
            raise ValueError('unsafe decoder image directory')
        target=directory/(image['sha256']+'.cubin')
        if target.exists():
            info=target.lstat()
            if target.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid!=os.geteuid() or info.st_nlink!=1 or info.st_mode & 0o077:
                raise ValueError('unsafe decoder image')
            existing=snapshot(target)
            if existing['sha256']==image['sha256'] and existing['bytes']==image['bytes']:
                return existing
            self.store.stats['decoder_image.corrupt']+=1
        raw=Path(cubin).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=image['sha256'] or len(raw)!=image['bytes']:
            raise ValueError('decoder input mutated before staging')
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.tmp')
        descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            with os.fdopen(descriptor,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,target)
        finally:
            if temporary.exists():temporary.unlink()
        self.store.stats['decoder_image.write_bytes']+=len(raw)
        return snapshot(target)

    def output(self, tool, arguments, cubin):
        image=snapshot(cubin);actual=self.tool(tool['path'])
        if actual!=tool:raise ValueError('provider tool identity differs')
        # Whitelist the two established read-only decoders. Any other command
        # remains pathname-specific and gains no inferred semantic identity.
        arguments=list(arguments);decoder=image
        expected={'nvdisasm':'-json','cuobjdump':'--dump-resource-usage'}
        if Path(actual['path']).name in expected and arguments==[expected[Path(actual['path']).name],str(cubin)]:
            decoder=self.decoder_image(cubin,image);arguments=[arguments[0],decoder['path']]
        identity=dict(contract=CONTRACT,kind='native-tool-stdout',tool=actual,
            argv=[actual['path']]+arguments,image=decoder,runtime=tool_runtime(actual['path']),environment_sha256=environment(),provider_sha256=snapshot(SOURCE)['sha256'])
        payload=self.store.load('toolstdout',identity)
        if payload is not None:
            try:raw=base64.b64decode(payload['base64'],validate=True)
            except (ValueError,KeyError,TypeError):raw=None
            if raw is not None and hashlib.sha256(raw).hexdigest()==payload.get('bytes_sha256'):
                if snapshot(cubin)!=image or snapshot(decoder['path'])!=decoder or self.tool(actual['path'])!=actual or tool_runtime(actual['path'])!=identity['runtime'] or environment()!=identity['environment_sha256']:raise ValueError('decoding context changed during cached lookup')
                return raw
        self.store.stats['native_tool_processes']+=1
        raw=subprocess.check_output(identity['argv'],stderr=subprocess.PIPE,timeout=120,env=cpu_environment())
        if snapshot(cubin)!=image or snapshot(decoder['path'])!=decoder or self.tool(actual['path'])!=actual or tool_runtime(actual['path'])!=identity['runtime'] or environment()!=identity['environment_sha256']:raise ValueError('native decoding input changed')
        self.store.save('toolstdout',identity,dict(base64=base64.b64encode(raw).decode(),bytes_sha256=hashlib.sha256(raw).hexdigest()))
        return raw

    def pure(self, function, *arguments, **keywords):
        """Reuse deterministic cores, then let callers recheck current provenance.

        Function arguments are fully content-hashed, not inferred from filenames.
        All local Python source is conservative transitive dependency identity.
        """
        source=self.sources(); runtime=environment()
        bound=inspect.signature(function).bind(*arguments,**keywords);bound.apply_defaults()
        identity=dict(contract=CONTRACT,kind='pure-analysis',sources=source,
            function=function.__module__+'.'+function.__qualname__,
            bytecode_sha256=code_fingerprint(function.__code__),
            arguments_sha256=digest(dict(bound.arguments)),
            environment_sha256=runtime,python=sys.version)
        value=self.store.load('pureanalysis',identity)
        if value is None:
            start=time.perf_counter_ns();value=function(*arguments,**keywords)
            self.store.stats['pureanalysis.compute_wall_ns']+=time.perf_counter_ns()-start
            # JSON roundtrip is part of the public provider contract.
            value=json.loads(canonical(value));self.store.save('pureanalysis',identity,value)
        if self.sources()!=source or environment()!=runtime:
            raise ValueError('pure analysis source/environment changed')
        return value

    def _call(self, kind, directory, function, *, extra=None):
        import native_cost_analysis as native
        import execution_cost_summary as complete
        import access_reuse as locality
        import address_intervention as intervention
        root=Path(directory).resolve(strict=True);started=time.perf_counter_ns()
        try:inputs={n:snapshot(root/n) for n in self.INPUTS[kind]}
        except (OSError,ValueError):
            self.store.stats[kind+'.uncacheable']+=1
            return function()
        source=self.sources()
        imported=dict(native=native.ANALYZER_SOURCE_SHA256,summary=complete.SOURCE_SHA256,
            locality=locality.SOURCE_SHA256,locality_sources=locality.SOURCE_HASHES,
            intervention=intervention.SOURCE_SHA256,intervention_sources=intervention.SUPPORT_SOURCES)
        tools={} if kind=='locality' else dict(nvdisasm=self.tool(self.nvdisasm))
        if kind=='native':tools['cuobjdump']=self.tool(self.cuobjdump)
        runtimes={name:tool_runtime(value['path']) for name,value in tools.items()}
        identity=dict(contract=CONTRACT,kind=kind,inputs=inputs,sources=source,imported_sources=imported,
            tools=tools,tool_runtimes=runtimes,extra=extra,environment_sha256=environment(),python=sys.version,
            callable_bytecode_sha256=code_fingerprint(function.__code__))
        def stable():
            return ({n:snapshot(root/n) for n in self.INPUTS[kind]}==inputs and self.sources()==source
                and environment()==identity['environment_sha256']
                and {name:self.tool(value['path']) for name,value in tools.items()}==tools
                and {name:tool_runtime(value['path']) for name,value in tools.items()}==runtimes)
        self.store.stats['identity_wall_ns']+=time.perf_counter_ns()-started
        payload=self.store.load(kind,identity)
        if payload is not None:
            if not stable():raise ValueError('inputs changed during cache lookup')
            return payload
        start=time.perf_counter_ns();payload=function();self.store.stats[kind+'.compute_wall_ns']+=time.perf_counter_ns()-start
        if not stable():raise ValueError('inputs changed during extraction')
        stable=payload.get('status') in ('available','complete') or (kind=='native' and payload.get('status')=='unsupported' and 'native_json_sha256' in payload and 'resource_text_sha256' in payload)
        if stable:self.store.save(kind,identity,payload)
        return json.loads(canonical(payload))

    def native(self,directory):
        import native_cost_analysis as module
        return self._call('native',directory,lambda:_native_with_provider(directory,nvdisasm=self.nvdisasm,cuobjdump=self.cuobjdump,document_provider=self))

    def summary(self,directory):
        import execution_cost_summary as module
        return self._call('summary',directory,lambda:module.summarize(directory,nvdisasm=self.nvdisasm,document_provider=self))

    def locality(self,directory):
        import access_reuse as module
        return self._call('locality',directory,lambda:module.summarize(directory,document_provider=self))

    def intervention(self,directory,locality):
        import address_intervention as module
        return self._call('intervention',directory,lambda:module.summarize(directory,locality,nvdisasm=self.nvdisasm,document_provider=self),extra=dict(locality_sha256=digest(locality)))


class PositiveProofCache:
    """A lookup proposal, never a replacement for the core Fraction verifier.

    Use as a context manager; numerical implementation/environment snapshots are
    checked at both boundaries. Negative numerical searches are not cached.
    """
    def __init__(self, root):
        self.store=PrivateStore(root);self.active=False

    def context(self):
        if snapshot(SOURCE)['sha256']!=IMPLEMENTATION_SHA256:
            raise ValueError('cache-source-changed-since-import')
        import numpy as np
        import sympy
        import numpy.linalg._umath_linalg as linear
        import numpy.core._multiarray_umath as array
        files=[SOURCE,SOURCE.parent/'joint_cost_model.py',SOURCE.parent/'joint_cost_calibration.py',Path(linear.__file__),Path(array.__file__)]
        for row in Path('/proc/self/maps').read_text().splitlines():
            name=row.split()[-1]
            if name.startswith('/') and any(word in name.lower() for word in ('blas','lapack')) and Path(name).is_file():files.append(Path(name))
        return dict(contract=CONTRACT,kind='positive-exact-order-proposal',
            files={str(p):snapshot(p)['sha256'] for p in set(files)},numpy=np.__version__,sympy=sympy.__version__,python=sys.version,
            environment_sha256=environment())

    def __enter__(self):
        self.identity=self.context();self.active=True;return self

    def __exit__(self,*error):
        self.active=False
        if self.context()!=self.identity:raise ValueError('numerical source/environment changed during proof session')

    @property
    def stats(self):return dict(self.store.stats)

    def key(self,rows,target):
        if not self.active:raise ValueError('active verified proof session required')
        return dict(self.identity,rows=rows,target=target)

    def __call__(self,rows,target):
        import joint_cost_model as model
        key=self.key(rows,target);proof=self.store.load('orderproof',key)
        if proof is None:return None
        try:return model.verify_order_proof(rows,target,proof)
        except (ValueError,KeyError,TypeError,IndexError):
            self.store.stats['orderproof.invalid_exact_certificate']+=1;return None

    def record(self,rows,target,proof):
        import joint_cost_model as model
        checked=model.verify_order_proof(rows,target,proof)
        self.store.save('orderproof',self.key(rows,target),checked)
