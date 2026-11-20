"""Exact compilation-cache tests with explicit CPU-only fake providers."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import native_compile_cache as cache
from native_static_support import StaticSharedUnsupported


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Providers:
    def __init__(self):self.calls=0;self.resolve_ids=[];self.predict_ids=[];self.fail=False;self.mutate=False
    def emit(self,e):return 'module { '+e['code']+' }\n',copy.deepcopy(e['config'])
    def compiler(self,path,target,options):
        self.calls+=1
        if self.fail:raise RuntimeError('explicit fake compile failure')
        text=path.read_text();token=digest([text,target,options])
        if self.mutate:path.write_text('changed')
        return dict(ptx='// fake PTX '+token+'\n',cubin=bytes([127])+b'ELF'+token.encode(),llir='; fake LLVM '+token+'\n')
    def predict(self,e,ptx=None):
        self.predict_ids.append(e['id'])
        return dict(logical_id=e['id'],ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest() if ptx is not None else None)
    def resolve(self,e,ptx,cubin,*,cache_dir,expected_cubin_sha256):
        self.resolve_ids.append(e['id']);assert sha(cubin)==expected_cubin_sha256
        return dict(schema_version=2,kind='resolved-compiler-resources',prediction=self.predict(e,ptx),
            compiler_exact=dict(identity=dict(schema_version=2,cubin_sha256=expected_cubin_sha256,entry='frisk_execution',tool_sha256='0'*64),
              resources=dict(registers_per_thread=32,raw_shared_bytes=0,stack_bytes_per_thread=0,raw_local_bytes=0,local_bytes_per_thread=0,needs_driver_local_query=False,spill_count=None),
              raw_resource_usage='fake record',raw_resource_sha256=hashlib.sha256(b'fake record').hexdigest(),source_cubin_path=str(cubin),command=['fake',str(cubin)],cache_hit=len(self.resolve_ids)>1,cache_key='fake'),
            registers_per_thread=32,shared_bytes=0,raw_native_shared_bytes=0,local_bytes=0,resource_sources={'registers':'compiled-cubin'},driver_confirmation_required=True)


class CompileCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='frisk-native-cache-test-');self.base=Path(self.temporary.name)
        self.tool=self.base/'tool';self.tool.write_bytes(b'fake native tool')
        self.source=dict(schema_version=1,source_manifest={},source_digest=digest({}))
        self.tools=dict(hashes={str(self.tool):sha(self.tool)})
        self.env={'compiler_flags':'default'};self.providers=Providers();self.identity={'kind':'explicit-unit-fake-v1'}
        self.cache=self.base/'cache'
    def tearDown(self):self.temporary.cleanup()
    def kwargs(self,**extra):
        x=dict(source=self.source,tools=self.tools,environment=self.env,provider_identity=self.identity,
               compiler=self.providers.compiler,emitter=self.providers.emit,predictor=self.providers.predict,resolver=self.providers.resolve)
        x.update(extra);return x
    def output(self,name,*,code='same',config=None,execution_id=None):
        d=self.base/name;d.mkdir()
        info={'static_shared_bytes':0,'threads':128,'all_metadata':['complete',1]} if config is None else copy.deepcopy(config)
        e=dict(id=execution_id or name,code=code,config=info)
        (d/'execution.json').write_text(json.dumps(e));tt,info=self.providers.emit(e)
        (d/'emitted.ttgir').write_text(tt);(d/'emitted-config.json').write_text(json.dumps(info));return d
    def verify(self,d,**extra):
        x=dict(allow_injected=True,emitter=self.providers.emit,predictor=self.providers.predict,
               resolver=self.providers.resolve,provider_identity=self.identity)
        x.update(extra);return cache.verify_precompiled(d,**x)
    def test_batch_deduplicates_exact_text_and_complete_config(self):
        a=self.output('a');b=self.output('b');summary=cache.compile_many([a,b],self.cache,**self.kwargs())
        self.assertEqual((summary['requests'],summary['exact_cache_hits'],summary['compiler_api_calls']),(2,1,1))
        self.assertEqual(self.providers.calls,1);self.assertIsNone(summary['compiler_internal_cache_hit'])
        self.assertEqual((a/'kernel.cubin').read_bytes(),(b/'kernel.cubin').read_bytes())
        self.assertEqual(json.loads((a/'resource-plan-before-native.json').read_text())['prediction']['logical_id'],'a')
        self.assertEqual(json.loads((b/'resource-plan-before-native.json').read_text())['prediction']['logical_id'],'b')
        self.assertEqual(self.providers.resolve_ids,['a','b'])
        self.assertFalse(self.verify(a)['verification_compiled']);self.assertTrue(self.verify(b)['exact_cache_hit'])
        self.assertEqual(self.providers.calls,1);self.assertEqual(self.providers.resolve_ids,['a','b','a','b'])
    def test_backend_integer_keys_and_tuples_follow_existing_json_wire_contract(self):
        original=self.providers.emit
        def wire(e):
            text,info=original(e);info['ssa_map']={12:[1,2],9:[3,4]};info['shape']=(32,32);return text,info
        a=self.output('a');b=self.output('b')
        for d in (a,b):
            text,info=wire(json.loads((d/'execution.json').read_text()));(d/'emitted-config.json').write_text(json.dumps(info))
        summary=cache.compile_many([a,b],self.cache,**self.kwargs(emitter=wire))
        self.assertEqual(summary['compiler_api_calls'],1)
        kwargs=dict(allow_injected=True,emitter=wire,predictor=self.providers.predict,resolver=self.providers.resolve,provider_identity=self.identity)
        self.assertEqual(cache.verify_precompiled(b,**kwargs)['status'],'verified-precompiled')
        stored=json.loads((a/'execution_config.json').read_text())
        self.assertEqual(stored['ssa_map'],{'12':[1,2],'9':[3,4]});self.assertEqual(stored['shape'],[32,32])

    def test_config_key_collision_is_rejected_not_silently_dropped(self):
        old=self.providers.emit
        def collision(e):
            tt,info=old(e);info['bad']={1:'number','1':'string'};return tt,info
        with self.assertRaises(ValueError):cache.compile_many([self.output('a')],self.cache,**self.kwargs(emitter=collision))
        self.assertEqual(self.providers.calls,0)

    def test_same_sass_or_feature_values_do_not_define_cache_identity(self):
        a=self.output('a');b=self.output('b',config={'static_shared_bytes':0,'threads':128,'all_metadata':['different',1]})
        summary=cache.compile_many([a,b],self.cache,**self.kwargs())
        self.assertEqual(summary['compiler_api_calls'],2);self.assertNotEqual(summary['rows'][0]['key'],summary['rows'][1]['key'])
        self.assertEqual((a/'kernel.cubin').read_bytes(),(b/'kernel.cubin').read_bytes())
    def test_changed_ir_changes_key_and_compiles(self):
        a=self.output('a');b=self.output('b',code='new')
        summary=cache.compile_many([a,b],self.cache,**self.kwargs());self.assertEqual(summary['compiler_api_calls'],2)
    def test_exact_cache_survives_session_boundary(self):
        a=self.output('a');b=self.output('b')
        first=cache.compile_many([a],self.cache,**self.kwargs());second=cache.compile_many([b],self.cache,**self.kwargs())
        self.assertEqual(first['compiler_api_calls'],1);self.assertEqual(second['compiler_api_calls'],0);self.assertEqual(self.providers.calls,1)
    def test_target_options_environment_source_and_tool_changes_invalidate_key(self):
        a=self.output('first');original=cache.compile_many([a],self.cache,**self.kwargs())['rows'][0]['key']
        changes=[{'target':{'backend':'cuda','arch':91,'warp_size':32}}, {'options':{'num_warps':8,'enable_fp_fusion':False}},
                 {'environment':{'compiler_flags':'changed'}}, {'provider_identity':{'kind':'unit-fake-v2'}}]
        for i,change in enumerate(changes):
            result=cache.compile_many([self.output('variant'+str(i))],self.cache,**self.kwargs(**change))
            self.assertEqual(result['compiler_api_calls'],1);self.assertNotEqual(result['rows'][0]['key'],original)
        self.tool.write_bytes(b'changed actual tool');changed=dict(hashes={str(self.tool):sha(self.tool)})
        result=cache.compile_many([self.output('changedtool')],self.cache,**self.kwargs(tools=changed))
        self.assertEqual(result['compiler_api_calls'],1);self.assertNotEqual(result['rows'][0]['key'],original)
    def test_source_manifest_digest_is_not_self_attested(self):
        source=copy.deepcopy(self.source);source['source_digest']='0'*64
        with self.assertRaises(ValueError):cache.compile_many([self.output('a')],self.cache,**self.kwargs(source=source))
    def test_existing_outputs_are_never_overwritten(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs());before=(a/'kernel.cubin').read_bytes()
        with self.assertRaises(FileExistsError):cache.compile_many([a],self.cache,**self.kwargs())
        self.assertEqual(before,(a/'kernel.cubin').read_bytes());self.assertEqual(self.providers.calls,1)
    def test_corrupt_same_key_is_rejected_not_recompiled(self):
        a=self.output('a');result=cache.compile_many([a],self.cache,**self.kwargs());entry=self.cache/result['rows'][0]['key']
        (entry/'kernel.cubin').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):cache.compile_many([self.output('b')],self.cache,**self.kwargs())
        self.assertEqual(self.providers.calls,1)
    def test_cache_exact_fileset_rejects_extra_file(self):
        a=self.output('a');result=cache.compile_many([a],self.cache,**self.kwargs());entry=self.cache/result['rows'][0]['key'];(entry/'extra').write_text('hidden')
        with self.assertRaises(ValueError):cache.compile_many([self.output('b')],self.cache,**self.kwargs())
        self.assertEqual(self.providers.calls,1)
    def test_cache_metadata_identity_tamper_is_rejected(self):
        a=self.output('a');result=cache.compile_many([a],self.cache,**self.kwargs());entry=self.cache/result['rows'][0]['key'];p=entry/'metadata.json';value=json.loads(p.read_text());value['identity']['execution_config']['threads']=64;p.write_text(json.dumps(value))
        with self.assertRaises(ValueError):cache.compile_many([self.output('b')],self.cache,**self.kwargs())
    def test_symlink_output_or_cache_paths_are_rejected(self):
        a=self.output('a');link=self.base/'link';link.symlink_to(a,target_is_directory=True)
        with self.assertRaises(ValueError):cache.compile_many([link],self.cache,**self.kwargs())
        link=self.base/'cachelink';link.symlink_to(self.cache,target_is_directory=True)
        with self.assertRaises(ValueError):cache.compile_many([a],link,**self.kwargs())
    def test_symlink_cache_payload_is_rejected(self):
        a=self.output('a');result=cache.compile_many([a],self.cache,**self.kwargs());entry=self.cache/result['rows'][0]['key'];saved=entry/'kernel.cubin';outside=self.base/'outside';outside.write_bytes(saved.read_bytes());saved.unlink();saved.symlink_to(outside)
        with self.assertRaises(ValueError):cache.compile_many([self.output('b')],self.cache,**self.kwargs())
    def test_single_writer_lock_rejects_concurrent_session(self):
        first=cache.CompilerSession(self.cache,**self.kwargs())
        try:
            with self.assertRaises(FileExistsError):cache.CompilerSession(self.cache,**self.kwargs())
        finally:first.close()
        with cache.CompilerSession(self.cache,**self.kwargs()) as fresh:self.assertFalse(fresh.closed)
    def test_failed_compilation_is_not_published_or_retried_in_same_output(self):
        a=self.output('a');self.providers.fail=True
        with self.assertRaises(RuntimeError):cache.compile_many([a],self.cache,**self.kwargs())
        self.assertFalse((a/cache.RECEIPT).exists());self.assertFalse(any(x.is_dir() and len(x.name)==64 for x in self.cache.iterdir()))
        self.assertTrue(any(x.name.startswith('.partial-') for x in self.cache.iterdir()))
        self.providers.fail=False
        with self.assertRaises(FileExistsError):cache.compile_many([a],self.cache,**self.kwargs())
    def test_compiler_mutating_input_is_detected(self):
        self.providers.mutate=True
        with self.assertRaises(ValueError):cache.compile_many([self.output('a')],self.cache,**self.kwargs())
    def test_changed_tool_during_compile_is_never_published(self):
        old=self.providers.compiler
        def changed(path,target,options):
            asm=old(path,target,options);self.tool.write_bytes(b'changed during compile');return asm
        with self.assertRaises(ValueError):cache.compile_many([self.output('a')],self.cache,**self.kwargs(compiler=changed))
        self.assertFalse(any(x.is_dir() and len(x.name)==64 for x in self.cache.iterdir()))

    def test_optional_llir_absence_remains_a_strict_artifact_set(self):
        old=self.providers.compiler
        def binary_only_ir(path,target,options):
            asm=old(path,target,options);asm.pop('llir');return asm
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs(compiler=binary_only_ir))
        self.assertFalse((a/'kernel.llir').exists());self.assertEqual(self.verify(a)['status'],'verified-precompiled')
        (a/'kernel.llir').write_text('unregistered extra LLVM')
        with self.assertRaises(ValueError):self.verify(a)

    def test_production_verifier_refuses_fake_provider_even_with_valid_hashes(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs())
        with self.assertRaises(ValueError):cache.verify_precompiled(a)
    def test_execution_mutation_cannot_reuse_receipt(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs());p=a/'execution.json';e=json.loads(p.read_text());e['id']='different';p.write_text(json.dumps(e))
        with self.assertRaises(ValueError):self.verify(a)
    def test_raw_artifact_and_generation_emission_mutations_rejected(self):
        for index,name in enumerate(['kernel.cubin','kernel.ptx','kernel.ttgir','execution_config.json','kernel.llir','emitted.ttgir','emitted-config.json']):
            d=self.output('a'+str(index));cache.compile_many([d],self.cache,**self.kwargs());p=d/name;p.write_bytes(p.read_bytes()+b'bad')
            with self.assertRaises((ValueError,json.JSONDecodeError)):self.verify(d)
    def test_checked_generation_emission_cannot_disappear(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs());(a/'emitted.ttgir').unlink()
        with self.assertRaises(ValueError):self.verify(a)

    def test_generation_mismatch_rejected_before_compiler_work(self):
        a=self.output('a');(a/'emitted.ttgir').write_text('wrong')
        with self.assertRaises(ValueError):cache.compile_many([a],self.cache,**self.kwargs())
        self.assertEqual(self.providers.calls,0)

    def test_receipt_digest_and_complete_artifact_set_rejected(self):
        for i in range(2):
            a=self.output('a'+str(i));cache.compile_many([a],self.cache,**self.kwargs());p=a/cache.RECEIPT;receipt=json.loads(p.read_text())
            if i==0:receipt['payload']['key']='0'*64
            else:
                receipt['payload']['artifacts'].pop('kernel.ptx');receipt['payload_sha256']=digest(receipt['payload'])
            p.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):self.verify(a)
    def test_resource_evidence_is_reconfirmed_not_only_hash_checked(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs());p=a/'resolved-compiler-resources.json';value=json.loads(p.read_text());value['registers_per_thread']=99;p.write_text(json.dumps(value))
        receipt=json.loads((a/cache.RECEIPT).read_text());receipt['payload']['artifacts'][p.name]={'bytes':p.stat().st_size,'sha256':sha(p)};receipt['payload_sha256']=digest(receipt['payload']);(a/cache.RECEIPT).write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):self.verify(a)
    def test_verification_compares_expected_context(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs())
        with self.assertRaises(ValueError):self.verify(a,expected_environment={'compiler_flags':'changed'})
        with self.assertRaises(ValueError):self.verify(a,provider_identity={'kind':'wrong'})
        self.tool.write_bytes(b'tool has changed')
        with self.assertRaises(ValueError):self.verify(a)
    def test_static_shared_supported_boundaries_compile_and_verify(self):
        for size in (0, 32768, 49152):
            with self.subTest(static_shared_bytes=size):
                a=self.output('static-'+str(size),config={'static_shared_bytes':size,'shared_limit_bytes':227328})
                result=cache.compile_many([a],self.cache,**self.kwargs())
                self.assertEqual(result['compiler_api_calls'],1)
                self.assertEqual(self.verify(a)['status'],'verified-precompiled')
                self.assertEqual(json.loads((a/'execution_config.json').read_text()),
                                 {'static_shared_bytes':size,'shared_limit_bytes':227328})
        self.assertEqual(self.providers.calls,3)

    def test_static_shared_outside_backend_domain_never_invokes_compiler_or_publishes(self):
        a=self.output('overcap',config={'static_shared_bytes':65536,'shared_limit_bytes':227328})
        with self.assertRaises(StaticSharedUnsupported):cache.compile_many([a],self.cache,**self.kwargs())
        self.assertEqual(self.providers.calls,0)
        self.assertFalse(list(self.cache.iterdir()))
        self.assertEqual({x.name for x in a.iterdir()}, {'execution.json','emitted.ttgir','emitted-config.json'})
        self.assertEqual(self.providers.predict_ids,[])

    def test_invalid_static_allocation_never_invokes_compiler(self):
        for index,config in enumerate(({}, {'static_shared_bytes':True}, {'static_shared_bytes':-1}, {'static_shared_bytes':1.0})):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    cache.compile_many([self.output('invalid-static-'+str(index),config=config)],self.cache,**self.kwargs())
        self.assertEqual(self.providers.calls,0)
        self.assertFalse(list(self.cache.iterdir()))

    def test_precompiled_verifier_rechecks_static_domain_after_valid_execution_hash(self):
        a=self.output('verified-static');cache.compile_many([a],self.cache,**self.kwargs())
        execution=json.loads((a/'execution.json').read_text());execution['config']['static_shared_bytes']=65536
        (a/'execution.json').write_text(json.dumps(execution))
        (a/'emitted-config.json').write_text(json.dumps(execution['config']))
        p=a/cache.RECEIPT;receipt=json.loads(p.read_text())
        receipt['payload']['execution_sha256']=sha(a/'execution.json')
        receipt['payload']['generation_emission']={name:{'bytes':(a/name).stat().st_size,'sha256':sha(a/name)} for name in cache.EMISSION_INPUTS}
        receipt['payload_sha256']=digest(receipt['payload']);p.write_text(json.dumps(receipt))
        with self.assertRaises(StaticSharedUnsupported):self.verify(a)
        self.assertEqual(self.providers.calls,1)

    def test_closed_session_cannot_compile(self):
        with cache.CompilerSession(self.cache,**self.kwargs()) as session:pass
        with self.assertRaises(ValueError):session.compile(self.output('a'))
    def test_injection_requires_all_providers_and_identity(self):
        with self.assertRaises(ValueError):cache.CompilerSession(self.cache,compiler=self.providers.compiler)

class SessionIdentityGuardTest(unittest.TestCase):
    setUp=CompileCacheTest.setUp
    tearDown=CompileCacheTest.tearDown
    kwargs=CompileCacheTest.kwargs
    output=CompileCacheTest.output
    # Shared fixture/helpers, not duplicate execution of the base test methods.
    def test_large_tool_content_reads_do_not_scale_with_hit_count(self):
        from unittest.mock import patch
        original=cache._sha
        counts=[]
        for size in (1,12):
            reads=[]
            def watched(path):
                if Path(path)==self.tool:reads.append(self.tool.stat().st_size)
                return original(path)
            with patch.object(cache,'_sha',side_effect=watched):
                with cache.CompilerSession(self.base/('cache-'+str(size)),**self.kwargs()) as session:
                    for i in range(size):session.compile(self.output('guard-'+str(size)+'-'+str(i)))
            counts.append(sum(reads))
        self.assertEqual(counts[0],counts[1])
        self.assertGreater(counts[0],0)  # Content hashes still run, including close.

    def test_tool_mutation_before_request_never_invokes_compiler(self):
        d=self.output('changed-tool')
        with self.assertRaisesRegex(ValueError,'tool changed'):
            with cache.CompilerSession(self.cache,**self.kwargs()) as session:
                self.tool.write_bytes(b'different executable')
                session.compile(d)
        self.assertEqual(self.providers.calls,0)
        self.assertFalse((d/'kernel.ttgir').exists())
        self.assertFalse(list(self.cache.glob('*/metadata.json')))
        self.assertFalse((self.cache/'.writer.lock').exists())

    def test_same_mtime_content_mutation_during_compile_never_publishes(self):
        import os
        compiler=self.providers.compiler
        def mutation(path,target,options):
            result=compiler(path,target,options);before=self.tool.stat()
            self.tool.write_bytes(b'X'*before.st_size)
            os.utime(self.tool,ns=(before.st_atime_ns,before.st_mtime_ns))
            return result
        with self.assertRaisesRegex(ValueError,'tool changed'):
            with cache.CompilerSession(self.cache,**self.kwargs(compiler=mutation)) as session:
                session.compile(self.output('restored-mtime'))
        self.assertEqual(self.providers.calls,1)
        self.assertFalse(list(self.cache.glob('*/metadata.json')))
        self.assertFalse((self.cache/'.writer.lock').exists())

    def test_unchanged_content_touch_is_revalidated_and_allowed(self):
        import os
        with cache.CompilerSession(self.cache,**self.kwargs()) as session:
            before=self.tool.stat();os.utime(self.tool,ns=(before.st_atime_ns,before.st_mtime_ns+1000000000))
            session.compile(self.output('touch-tool'))
            checks=session.large_identity_full_checks
        self.assertGreaterEqual(checks,2)
        self.assertEqual(self.providers.calls,1)

    def test_provider_payload_addition_detected_without_gpu_or_hash_read(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        context=dict(mode='production',tools=self.tools,provider_identity=dict(triton_version='fake-version',python_and_math_payload_hashes={str(self.tool):sha(self.tool)}))
        extra=self.base/'new-compiler.py';extra.write_text('new code')
        with patch.dict('sys.modules',{'triton':SimpleNamespace(__version__='fake-version')}),patch.object(cache,'_production_payload_paths',return_value=[str(self.tool),str(extra)]):
            with self.assertRaisesRegex(ValueError,'payload set/version changed'):cache._large_identity_witness(context)

if __name__=='__main__':unittest.main()
