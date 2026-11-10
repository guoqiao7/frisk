import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import struct
import sys
import unittest
from unittest.mock import patch
import cost_evidence_cache as cache
import joint_cost_model as model


class CostEvidenceCacheTest(unittest.TestCase):
    def test_loaded_cache_producer_source_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            provider=cache.CostEvidenceCache(Path(directory)/'cache')
            with patch.object(cache,'IMPLEMENTATION_SHA256','different-loaded-source'):
                with self.assertRaisesRegex(ValueError,'cache-source-changed-since-import'):
                    provider.sources()
                with self.assertRaisesRegex(ValueError,'cache-source-changed-since-import'):
                    cache.PositiveProofCache(Path(directory)/'proof').context()

    def test_declared_cpu_environment_is_exactly_forwarded(self):
        with patch.dict(os.environ,{'SSH_CONNECTION':'random-connection','SSH_CLIENT':'random-client',
             '_':'arbitrary-shell-export','FRISK_CPU_OPTION':'semantic','LC_ALL':'C'},clear=True):
            self.assertEqual(cache.cpu_environment(),{'FRISK_CPU_OPTION':'semantic','LC_ALL':'C'})
            first=cache.environment()
            with patch.dict(os.environ,{'SSH_CONNECTION':'another-port','_':'another-binary'}):
                self.assertEqual(cache.environment(),first)
            with patch.dict(os.environ,{'FRISK_CPU_OPTION':'changed'}):
                self.assertNotEqual(cache.environment(),first)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);provider=cache.CostEvidenceCache(root/'cache');image=root/'kernel.cubin';image.write_bytes(b'image')
            tool={'path':'/usr/local/cuda-12.8/bin/nvdisasm','sha256':'fixture','version':'fixture'}
            with patch.object(provider,'tool',return_value=tool),patch.object(cache,'tool_runtime',return_value={}),patch.object(cache.subprocess,'check_output',return_value=b'output') as process:
                provider.output(tool,['-json',str(image)],image)
                self.assertEqual(process.call_args.kwargs['env'],cache.cpu_environment())

    def test_complete_code_fingerprint_is_stable_across_hash_seeds(self):
        program=("import sys,json,marshal,hashlib;sys.path.insert(0,"+repr(str(Path(cache.__file__).parent))+
                 ");import cost_evidence_cache as c;f=compile(\"def f(v): return v in {'alpha','beta','gamma','delta'}\",'<fixed>','exec');"+
                 "print(json.dumps([c.code_fingerprint(f),hashlib.sha256(marshal.dumps(f)).hexdigest()]))")
        values=[]
        for seed in ('1','2','43'):
            output=subprocess.check_output([sys.executable,'-B','-c',program],env=dict(os.environ,PYTHONHASHSEED=seed),text=True)
            values.append(json.loads(output))
        self.assertEqual(len({v[0] for v in values}),1)
        self.assertGreater(len({v[1] for v in values}),1)

    def test_complete_code_fingerprint_preserves_typed_constants_and_fields(self):
        source="def f(v): return v in {'alpha','beta'}"
        a=compile(source,'<fixed>','exec')
        for changed in (compile(source.replace('beta','gamma'),'<fixed>','exec'),
                        compile(source,'<different-file>','exec'),
                        compile('\n'+source,'<fixed>','exec'),
                        compile('def f(v): return 1','<fixed>','exec'),
                        compile('def f(v): return 1.0','<fixed>','exec')):
            self.assertNotEqual(cache.code_fingerprint(a),cache.code_fingerprint(changed))
        self.assertNotEqual(cache.code_fingerprint(compile('x=1','<fixed>','exec')),
                            cache.code_fingerprint(compile('x=True','<fixed>','exec')))
        nan_a=struct.unpack('>d',bytes.fromhex('7ff8000000000001'))[0]
        nan_b=struct.unpack('>d',bytes.fromhex('7ff8000000000002'))[0]
        constant=compile('x=0.0','<fixed>','exec')
        self.assertNotEqual(cache.code_fingerprint(constant.replace(co_consts=(nan_a,None))),
                            cache.code_fingerprint(constant.replace(co_consts=(nan_b,None))))
        for field,old in ((n,getattr(a,n)) for n in dir(a) if n.startswith('co_') and not callable(getattr(a,n))):
            self.assertIn(field,cache._code_value(a)['fields'])

    def test_content_not_stat_is_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'input';path.write_bytes(b'abcd');old=path.stat()
            first=cache.snapshot(path);path.write_bytes(b'efgh');os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns))
            self.assertNotEqual(first['sha256'],cache.snapshot(path)['sha256'])

    def test_private_store_identity_corruption_and_return_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            store=cache.PrivateStore(Path(directory)/'cache');identity={'input':'a'};value={'body':[1,2]}
            store.save('test',identity,value);result=store.load('test',identity);result['body'].append(3)
            self.assertEqual(store.load('test',identity),value)
            self.assertIsNone(store.load('test',{'input':'b'}))
            path=store.path('test',identity);path.write_text('{bad json')
            self.assertIsNone(store.load('test',identity));self.assertEqual(store.stats['test.corrupt'],1)
            self.assertEqual(len(list((store.root/'invalid').glob('*.json'))),1)

    def test_shared_or_linked_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bad=root/'shared';bad.mkdir(mode=0o755)
            with self.assertRaises(ValueError):cache.PrivateStore(bad)
            store=cache.PrivateStore(root/'private');p=store.path('test',{'a':1});outside=root/'outside';outside.write_text('{}');p.symlink_to(outside)
            with self.assertRaises(ValueError):store.load('test',{'a':1})
            p.unlink();os.link(outside,p)
            with self.assertRaises(ValueError):store.load('test',{'a':1})

    def test_core_rechecks_provider_proof_and_recomputes_none(self):
        rows=[[-1,0]];target=[-1,0];proof=model.exact_order_proof(rows,target)
        with patch.object(model,'exact_order_proof',wraps=model.exact_order_proof) as solve:
            self.assertEqual(model._order_proof(rows,target,lambda a,b:proof),proof);solve.assert_not_called()
            self.assertEqual(model._order_proof(rows,target,lambda a,b:None),proof);self.assertEqual(solve.call_count,1)
        for key,value in [('strict_calibration_mass',[0,1]),('strict_calibration_mass',[1.0,1]),('hardware_time_bound',0),('hardware_time_bound',True),('lambdas',{'0':[-1,1]}),('nonnegative_prior_slack',{'1':[2,1]})]:
            invalid=copy.deepcopy(proof);invalid[key]=value
            with self.assertRaises(ValueError):model._order_proof(rows,target,lambda a,b:invalid)

    def test_proof_cache_positive_only_and_exact_poison_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cache';rows=[[-1,0]];target=[-1,0]
            with cache.PositiveProofCache(path) as provider:
                proof=model._order_proof(rows,target,provider)
                self.assertEqual(provider(rows,target),proof)
                self.assertIsNone(provider(rows,[1,0]))
                key=provider.key(rows,target);invalid=copy.deepcopy(proof);invalid['lambdas']={'0':[2,1]}
                provider.store.save('orderproof',key,invalid)
                self.assertIsNone(provider(rows,target))
                self.assertEqual(model._order_proof(rows,target,provider),proof)
                with patch.object(model,'exact_order_proof',return_value=None) as solver:
                    self.assertIsNone(model._order_proof(rows,[1,0],provider))
                    self.assertIsNone(model._order_proof(rows,[1,0],provider));self.assertEqual(solver.call_count,2)
            with self.assertRaises(ValueError):provider(rows,target)

    def test_proof_session_rejects_environment_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            provider=cache.PositiveProofCache(Path(directory)/'cache')
            with self.assertRaises(ValueError):
                with provider:
                    with patch.dict(os.environ,{'FRISK_CACHE_TEST':'different'}):
                        provider.__exit__(None,None,None)

    def test_whole_function_cache_binds_inputs_sources_and_providers(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'case';case.mkdir();(case/'kernel.cubin').write_bytes(b'fake-cubin')
            provider=cache.CostEvidenceCache(root/'cache');calls=[]
            function=lambda: calls.append(1) or {'status':'available','features':{'n':3}}
            with patch.object(provider,'tool',return_value={'path':'/tool','sha256':'x','version':'v'}), patch.object(cache,'tool_runtime',return_value={}):
                first=provider._call('native',case,function);second=provider._call('native',case,function)
                self.assertEqual(first,second);self.assertEqual(len(calls),1)
                (case/'kernel.cubin').write_bytes(b'other-cbin');provider._call('native',case,function);self.assertEqual(len(calls),2)
                with patch.object(provider,'sources',return_value={'changed.py':'new'}):provider._call('native',case,function)
                self.assertEqual(len(calls),3)
                with patch.dict(os.environ,{'FRISK_CACHE_INPUT':'new'}):provider._call('native',case,function)
                self.assertEqual(len(calls),4)

    def test_pure_core_unifies_defaults_and_rechecks_input_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            provider=cache.CostEvidenceCache(Path(directory)/'cache');calls=[]
            def core(value,*,factor=3):calls.append(value);return {'result':value*factor}
            self.assertEqual(provider.pure(core,7),{'result':21})
            self.assertEqual(provider.pure(core,7,factor=3),{'result':21})
            self.assertEqual(calls,[7]);provider.pure(core,8);self.assertEqual(calls,[7,8])
            with patch.object(provider,'sources',return_value={'different':'source'}):provider.pure(core,7)
            self.assertEqual(calls,[7,8,7])
            self.assertEqual(provider.stats['pureanalysis.hits'],1)

    def test_content_addressed_decoder_image_is_actual_argv_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);provider=cache.CostEvidenceCache(root/'cache')
            a=root/'a.cubin';b=root/'b.cubin';a.write_bytes(b'actual-image');b.write_bytes(a.read_bytes())
            x=provider.decoder_image(a,cache.snapshot(a));y=provider.decoder_image(b,cache.snapshot(b))
            self.assertEqual(x,y);self.assertEqual(Path(x['path']).read_bytes(),a.read_bytes())
            Path(x['path']).write_bytes(b'corrupt')
            self.assertEqual(provider.decoder_image(a,cache.snapshot(a)),x)
            self.assertEqual(provider.stats['decoder_image.corrupt'],1)
            Path(x['path']).unlink();Path(x['path']).symlink_to(a)
            with self.assertRaises(ValueError):provider.decoder_image(a,cache.snapshot(a))

    def test_tool_stdout_shared_across_input_paths_and_mutation_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);provider=cache.CostEvidenceCache(root/'cache')
            a=root/'a.cubin';b=root/'b.cubin';a.write_bytes(b'first');b.write_bytes(b'first')
            tool={'path':'/usr/local/cuda-12.8/bin/nvdisasm','sha256':'fixture','version':'fixture'}
            def decode(argv,**kwargs):return Path(argv[-1]).read_bytes()+b'-decoded'
            with patch.object(provider,'tool',return_value=tool),patch.object(cache,'tool_runtime',return_value={}),patch.object(cache.subprocess,'check_output',side_effect=decode) as process:
                self.assertEqual(provider.output(tool,['-json',str(a)],a),b'first-decoded')
                self.assertEqual(provider.output(tool,['-json',str(b)],b),b'first-decoded');self.assertEqual(process.call_count,1)
                b.write_bytes(b'other');self.assertEqual(provider.output(tool,['-json',str(b)],b),b'other-decoded');self.assertEqual(process.call_count,2)
                argv=process.call_args[0][0];self.assertIn('/decoder-images/',argv[-1])

    def test_transient_or_incomplete_unknown_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=root/'case';case.mkdir();(case/'kernel.cubin').write_bytes(b'cubin')
            provider=cache.CostEvidenceCache(root/'cache');calls=[]
            def function():calls.append(1);return {'status':'unavailable','features':None}
            with patch.object(provider,'tool',return_value={'path':'/tool','sha256':'x','version':'v'}), patch.object(cache,'tool_runtime',return_value={}):
                provider._call('native',case,function);provider._call('native',case,function)
            self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
