"""CPU injected-provider tests for exact compiler-input reuse, artifact staging."""
import copy,json,unittest
from pathlib import Path
import native_compile_cache as cache
import test_native_compile_cache as previous
from test_native_compile_cache import digest,sha
from native_static_support import StaticSharedUnsupported

class CompilerInputReuseTest(unittest.TestCase):
    setUp=previous.CompileCacheTest.setUp;tearDown=previous.CompileCacheTest.tearDown;kwargs=previous.CompileCacheTest.kwargs;output=previous.CompileCacheTest.output;verify=previous.CompileCacheTest.verify
    def different(self,name,**kwargs):
        return self.output(name,config={'static_shared_bytes':0,'threads':128,'owner_diagnostic':name},**kwargs)
    def run_enabled(self,dirs,**kwargs):return cache.compile_many(dirs,self.cache,**self.kwargs(compiler_input_reuse=True,**kwargs))
    def ci_entry(self,d):
        p=json.loads((d/cache.RECEIPT).read_text())['payload']['compiler_input_cache'];return Path(p['cache_entry'])
    def reseal(self,d,edit):
        p=d/cache.RECEIPT;v=json.loads(p.read_text());edit(v['payload']);v['payload_sha256']=digest(v['payload']);p.write_text(json.dumps(v))
    def test_three_way_accounting_and_per_request_resources(self):
        a=self.different('a');b=self.different('b');c=self.output('c',config=json.loads((a/'emitted-config.json').read_text()))
        x=self.run_enabled([a,b,c]);self.assertEqual((x['requests'],x['exact_cache_hits'],x['compiler_input_hits'],x['compiler_api_calls']),(3,1,1,1))
        self.assertEqual(self.providers.calls,1);self.assertEqual(self.providers.resolve_ids,['a','b','c'])
        self.assertNotEqual(x['rows'][0]['key'],x['rows'][1]['key']);self.assertEqual(x['rows'][0]['key'],x['rows'][2]['key'])
        self.assertEqual([q['compiler_input_hit'] for q in x['rows']],[False,True,False])
        for d in (a,b,c):self.assertEqual(self.verify(d)['status'],'verified-precompiled')
        self.assertEqual(self.providers.calls,1)
        self.assertEqual(json.loads((b/'resource-plan-before-native.json').read_text())['prediction']['logical_id'],'b')
        self.assertEqual(json.loads((b/'execution_config.json').read_text())['owner_diagnostic'],'b')
    def test_default_is_full_config_only(self):
        x=cache.compile_many([self.different('a'),self.different('b')],self.cache,**self.kwargs())
        self.assertEqual(x['compiler_api_calls'],2);self.assertNotIn('compiler_input_hits',x);self.assertFalse((self.cache/'compiler-input').exists())
    def test_secondary_hit_survives_session_boundary(self):
        a=self.different('a');b=self.different('b');self.run_enabled([a]);x=self.run_enabled([b]);self.assertEqual((x['compiler_api_calls'],x['compiler_input_hits']),(0,1));self.verify(b)
    def test_full_hit_needs_no_secondary_entry(self):
        a=self.output('a');cache.compile_many([a],self.cache,**self.kwargs())
        b=self.output('b');x=self.run_enabled([b]);self.assertEqual((x['exact_cache_hits'],x['compiler_input_hits'],x['compiler_api_calls']),(1,0,0));self.verify(b)
    def test_opt_in_type_is_strict(self):
        for bad in (1,0,None,'true'):
            with self.assertRaises(ValueError):cache.CompilerSession(self.cache,**self.kwargs(compiler_input_reuse=bad))
    def test_exact_TTGIR_not_identical_native_bytes(self):
        old=self.providers.compiler
        def same(path,target,options):
            old(path,target,options);return dict(ptx='same',cubin=b'\x7fELFsame',llir='same')
        a=self.different('a');b=self.different('b',code='changed')
        x=self.run_enabled([a,b],compiler=same);self.assertEqual((x['compiler_input_hits'],x['compiler_api_calls']),(0,2))
        self.assertEqual((a/'kernel.cubin').read_bytes(),(b/'kernel.cubin').read_bytes())
    def test_all_compiler_context_axes_invalidate_secondary(self):
        self.run_enabled([self.different('a')])
        cases=[dict(environment={'changed':'compiler-env'}),dict(provider_identity={'kind':'other'}),dict(target={'backend':'cuda','arch':91,'warp_size':32}),dict(options={'num_warps':8,'enable_fp_fusion':False}),dict(source={**self.source,'schema_version':2})]
        for i,change in enumerate(cases):
            x=self.run_enabled([self.different('other'+str(i))],**change);self.assertEqual((x['compiler_input_hits'],x['compiler_api_calls']),(0,1))
        self.tool.write_bytes(b'changedtool');x=self.run_enabled([self.different('tool-request')],tools={'hashes':{str(self.tool):sha(self.tool)}});self.assertEqual(x['compiler_api_calls'],1)
    def test_static_support_rechecked_before_secondary_hit(self):
        self.run_enabled([self.different('a')]);bad=self.output('b',config={'static_shared_bytes':65536,'threads':128})
        with self.assertRaises(StaticSharedUnsupported):self.run_enabled([bad])
        self.assertEqual(self.providers.calls,1);self.assertEqual(self.providers.resolve_ids,['a'])
    def test_generation_rechecked_before_secondary_hit(self):
        self.run_enabled([self.different('a')]);bad=self.different('b');(bad/'emitted.ttgir').write_text('changed')
        with self.assertRaises(ValueError):self.run_enabled([bad])
        self.assertEqual(self.providers.calls,1)
    def test_secondary_payload_hash_tamper_rejects_without_recompile(self):
        a=self.different('a');self.run_enabled([a]);entry=self.ci_entry(a);(entry/'kernel.cubin').write_bytes(b'\x7fELFchanged')
        with self.assertRaises(ValueError):self.run_enabled([self.different('b')])
        self.assertEqual(self.providers.calls,1)
    def test_secondary_exact_fileset_rejects_extra(self):
        a=self.different('a');self.run_enabled([a]);(self.ci_entry(a)/'hidden').write_text('extra')
        with self.assertRaises(ValueError):self.run_enabled([self.different('b')])
    def test_secondary_symlink_rejected(self):
        a=self.different('a');self.run_enabled([a]);entry=self.ci_entry(a);outside=self.base/'outside';outside.write_bytes((entry/'kernel.cubin').read_bytes());(entry/'kernel.cubin').unlink();(entry/'kernel.cubin').symlink_to(outside)
        with self.assertRaises(ValueError):self.run_enabled([self.different('b')])
    def test_secondary_identity_tamper_rejected(self):
        a=self.different('a');self.run_enabled([a]);entry=self.ci_entry(a);p=entry/'metadata.json';m=json.loads(p.read_text());m['identity']['context']['options']['num_warps']=8;p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):self.run_enabled([self.different('b')])
    def test_optional_llir_is_exact_in_both_caches(self):
        old=self.providers.compiler
        def no_llir(*a):
            x=old(*a);x.pop('llir');return x
        a=self.different('a');b=self.different('b');x=self.run_enabled([a,b],compiler=no_llir);self.assertEqual(x['compiler_input_hits'],1);self.assertFalse((b/'kernel.llir').exists());self.verify(b)
        (self.ci_entry(b)/'kernel.llir').write_text('unregistered')
        with self.assertRaises(ValueError):self.verify(b)
    def test_secondary_provenance_cannot_disappear(self):
        a=self.different('a');b=self.different('b');self.run_enabled([a,b]);self.reseal(b,lambda p:p.pop('compiler_input_cache'))
        with self.assertRaises(ValueError):self.verify(b)
    def test_secondary_hit_cannot_be_relabelled_as_API(self):
        a=self.different('a');b=self.different('b');self.run_enabled([a,b]);self.reseal(b,lambda p:p['compiler_input_cache'].update(compiler_input_hit=False))
        with self.assertRaises(ValueError):self.verify(b)
    def test_secondary_context_binding_rechecked_by_loader(self):
        a=self.different('a');b=self.different('b');self.run_enabled([a,b]);self.reseal(b,lambda p:p['compiler_input_cache']['identity']['context'].update(options={'num_warps':8,'enable_fp_fusion':False}))
        with self.assertRaises(ValueError):self.verify(b)
    def test_secondary_and_full_payload_must_match(self):
        a=self.different('a');b=self.different('b');self.run_enabled([a,b]);entry=self.ci_entry(b);p=entry/'metadata.json';m=json.loads(p.read_text());(entry/'kernel.ptx').write_text('consistent altered file');m['artifacts']['kernel.ptx']={'bytes':(entry/'kernel.ptx').stat().st_size,'sha256':sha(entry/'kernel.ptx')};p.write_text(json.dumps(m))
        self.reseal(b,lambda p:p['compiler_input_cache'].update(cache_metadata_sha256=sha(entry/'metadata.json')))
        with self.assertRaises(ValueError):self.verify(b)
    def test_request_resource_plan_remains_independent(self):
        a=self.different('a');b=self.different('b');self.run_enabled([a,b]);p=b/'resolved-compiler-resources.json';x=json.loads(p.read_text());x['prediction']['logical_id']='a';p.write_text(json.dumps(x))
        self.reseal(b,lambda v:v['artifacts'].update({p.name:{'bytes':p.stat().st_size,'sha256':sha(p)}}))
        with self.assertRaises(ValueError):self.verify(b)
    def test_failed_API_never_publishes_secondary(self):
        self.providers.fail=True
        with self.assertRaises(RuntimeError):self.run_enabled([self.different('a')])
        self.assertFalse((self.cache/'compiler-input').exists());self.assertFalse((self.cache/'.writer.lock').exists())

    def test_validated_secondary_single_read_fault_never_publishes_full_entry(self):
        from unittest.mock import patch
        a=self.different('seed-read-fault');self.run_enabled([a]);entry=self.ci_entry(a)
        healthy=(entry/'kernel.cubin').read_bytes()
        for filename in sorted(cache.COMPILER_INPUT_REQUIRED|cache.NATIVE_OPTIONAL):
            with self.subTest(filename=filename):
                b=self.different('fault-'+filename);before={p.name for p in self.cache.iterdir() if p.is_dir() and len(p.name)==64}
                original=cache._validate_compiler_input;read=Path.read_bytes;state={'armed':False,'fired':False}
                def validated(path,identity):
                    meta=original(path,identity)
                    if path==entry:state['armed']=True
                    return meta
                def corrupted(path):
                    value=read(path)
                    if state['armed'] and path==entry/filename and not state['fired']:
                        state['fired']=True;return value+b'one transient read fault'
                    return value
                with patch.object(cache,'_validate_compiler_input',side_effect=validated),patch.object(Path,'read_bytes',corrupted):
                    with self.assertRaisesRegex(ValueError,'payload changed after validation'):self.run_enabled([b])
                self.assertTrue(state['fired']);self.assertEqual(before,{p.name for p in self.cache.iterdir() if p.is_dir() and len(p.name)==64})
                self.assertFalse((b/cache.RECEIPT).exists());self.assertFalse((self.cache/'.writer.lock').exists());self.assertEqual((entry/'kernel.cubin').read_bytes(),healthy)
                config=json.loads((b/'emitted-config.json').read_text());c=self.output('healthy-'+filename,config=config)
                result=self.run_enabled([c]);self.assertEqual((result['exact_cache_hits'],result['compiler_input_hits'],result['compiler_api_calls']),(0,1,0))
                self.assertEqual((c/'kernel.cubin').read_bytes(),healthy);self.assertEqual(self.verify(c)['status'],'verified-precompiled')
                d=self.output('full-hit-'+filename,config=config);result=self.run_enabled([d]);self.assertEqual(result['exact_cache_hits'],1)
                self.assertEqual((d/'kernel.cubin').read_bytes(),healthy);self.verify(d)
        self.assertEqual(self.providers.calls,1)

    def test_secondary_changed_after_validation_never_publishes_full_entry(self):
        from unittest.mock import patch
        a=self.different('seed-post-validation');self.run_enabled([a]);entry=self.ci_entry(a);original=cache._validate_compiler_input
        before={p.name for p in self.cache.iterdir() if p.is_dir() and len(p.name)==64}
        def changed(path,identity):
            meta=original(path,identity)
            if path==entry:(entry/'kernel.cubin').write_bytes(b'\x7fELFchanged-after-validation')
            return meta
        b=self.different('changed-post-validation')
        with patch.object(cache,'_validate_compiler_input',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'payload changed after validation'):self.run_enabled([b])
        self.assertEqual(before,{p.name for p in self.cache.iterdir() if p.is_dir() and len(p.name)==64});self.assertFalse((b/cache.RECEIPT).exists())
        with self.assertRaises(ValueError):self.run_enabled([self.different('later-corrupt-secondary')])
        self.assertEqual(self.providers.calls,1)

    def test_full_temp_write_fault_cannot_resign_compiler_input_payload(self):
        from unittest.mock import patch
        a=self.different('seed-write-fault');self.run_enabled([a]);entry=self.ci_entry(a);healthy=(entry/'kernel.cubin').read_bytes();write=Path.write_bytes
        before={p.name for p in self.cache.iterdir() if p.is_dir() and len(p.name)==64};state={'fired':False}
        def changed(path,value):
            if path.name=='kernel.cubin' and path.parent.parent==self.cache and path.parent.name.startswith('.partial-'):
                state['fired']=True;value=b'\x7fELFwrite-fault'
            return write(path,value)
        b=self.different('fault-before-publication')
        with patch.object(Path,'write_bytes',changed):
            with self.assertRaisesRegex(ValueError,'payload differ before publication'):self.run_enabled([b])
        self.assertTrue(state['fired']);self.assertEqual(before,{p.name for p in self.cache.iterdir() if p.is_dir() and len(p.name)==64});self.assertFalse((b/cache.RECEIPT).exists());self.assertEqual((entry/'kernel.cubin').read_bytes(),healthy)
        c=self.output('healthy-after-write-fault',config=json.loads((b/'emitted-config.json').read_text()));result=self.run_enabled([c])
        self.assertEqual((result['exact_cache_hits'],result['compiler_input_hits'],result['compiler_api_calls']),(0,1,0));self.verify(c);self.assertEqual(self.providers.calls,1)

if __name__=='__main__':unittest.main()
