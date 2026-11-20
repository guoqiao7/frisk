"""Case-local sharing regressions independent of GPU, old receipts or benchmarks."""
import ast
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import backend
import demand_coverage_audit as demand
import emission_audit_session as sharing
import execution_support as support
import native_surviving_audit as surviving
from test_execution_support import packet


def exact_fixture():
    data=packet()
    for i,value in enumerate(data['values'][-2:]):value['layout_variable_ids']=[i]
    values=[dict(variable=i,queries=[dict(terms=[0]*8)]) for i in range(2)]
    accesses=[dict(query_vectors=1,warp_issues=4,global_sectors32=20,shared_bank_excess=0,address_alu_warp_issues=alu)
              for alu in (124,360)]
    ledger=dict(exact=True,demand_lowering=False,values=values,boundaries=[],accesses=accesses,
                arithmetic_warp_issues=4,address_alu_warp_issues=484,global_load_issue_groups=1,
                materialized_query_vectors=2,scratch_max_bytes=0)
    retained=copy.deepcopy(ledger);retained['global_load_issue_groups']=None
    record=dict(demand_coverage=ledger,cost=dict(cost=[0]*9),surviving_coverage=dict(
        proof_status='Proven',work_basis=surviving.WORK_BASIS,native_exact=False,emitted=copy.deepcopy(ledger),
        surviving=retained,retained_variables=[0,1],removed_variables=[],retained_communication=[],removed_query_vectors=0))
    return json.loads(json.dumps(data,sort_keys=True)),record


def fallback_fixture():
    return json.loads(json.dumps(packet(),sort_keys=True)),dict(demand_coverage=dict(exact=False,reason='test outside exact demand'),
                         surviving_coverage=dict(proof_status='Unknown',reason='test outside exact survival'))


class EmissionAuditSessionTests(unittest.TestCase):
    def test_legacy_default_branch_is_original_full_ast(self):
        source=Path(sharing.__file__).with_name('partial_cost_qualification.py').read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='add_case')
        branch=next(n for n in fn.body if isinstance(n,ast.If) and ast.unparse(n.test)=='emission_factory is None')
        observed=hashlib.sha256(ast.dump(ast.Module(body=branch.body,type_ignores=[]),include_attributes=False).encode()).hexdigest()
        self.assertEqual(observed,'51a010c43930ed1dbb42b8d23a716b202d8812864071f6cd8345df71f3901ad9')
    def test_context_is_explicit_nonnested_and_restores_default(self):
        self.assertIsNone(sharing.current_factory())
        with self.assertRaisesRegex(RuntimeError,'boundary'):
            with sharing.enabled():
                self.assertIs(sharing.current_factory(),sharing.EmissionSession)
                with self.assertRaisesRegex(ValueError,'nonnested'):
                    with sharing.enabled():pass
                raise RuntimeError('boundary')
        self.assertIsNone(sharing.current_factory())
    def test_original_audit_code_full_result_and_single_emission(self):
        data,record=exact_fixture();before=(demand.backend,surviving.backend,surviving.emission,builtins.__import__)
        original=(demand.audit(data,record),surviving.audit(data,record,require_priced_features=False),support.assess(data,include_emission=True))
        with sharing.EmissionSession(data,record) as session:
            self.assertIs(session._demand_code.__code__,demand.audit.__code__)
            self.assertIs(session._surviving_code.__code__,surviving.audit.__code__)
            self.assertIs(session._assess_code.__code__,support.assess.__code__)
            current=(session.demand_audit(),session.surviving_audit(require_priced_features=False),session.assess(include_emission=True))
            self.assertEqual(current,original);self.assertEqual(session.stats['emitter_runs'],1)
            self.assertEqual(session.stats['emission_replays'],3);self.assertEqual(session.stats['strong_proof_hits'],1)
            self.assertEqual(session.stats['access_row_hits'],2);self.assertEqual(session.stats['access_row_misses'],2)
        self.assertEqual((demand.backend,surviving.backend,surviving.emission,builtins.__import__),before)
    def test_explicit_fallback_never_becomes_successful_memo(self):
        data,record=fallback_fixture()
        with sharing.EmissionSession(data,record) as session:
            self.assertEqual(session.demand_audit(),demand.audit(data,record))
            self.assertEqual(session.surviving_audit(),surviving.audit(data,record))
            self.assertEqual(session.stats['strong_proof_hits'],0);self.assertEqual(session.stats['strong_proof_misses'],2)
            self.assertEqual(session.receipt()['strong_proof_receipts'],{})
            self.assertEqual(session.stats['emitter_runs'],1)
    def test_scalar_coverage_predicate_is_still_checked(self):
        data,record=exact_fixture();record['demand_coverage']['accesses'][0]['global_sectors32']+=1
        session=sharing.EmissionSession(data,record)
        with self.assertRaisesRegex(ValueError,'backend access footprint differs'):session.demand_audit()
        self.assertEqual(session._proofs,{})
        with self.assertRaisesRegex(ValueError,'closed or failed'):session.assess(include_emission=True)
    def test_survival_predicate_cannot_be_hidden_by_valid_strong_proof(self):
        data,record=exact_fixture();record['surviving_coverage']['surviving']['accesses'][0]['global_sectors32']+=1
        session=sharing.EmissionSession(data,record);self.assertEqual(session.demand_audit()['status'],'passed')
        with self.assertRaisesRegex(ValueError,'surviving access ledger differs'):session.surviving_audit(require_priced_features=False)
        self.assertEqual(session.stats['strong_proof_hits'],1)
    def test_failed_assembly_is_not_cached_or_reused(self):
        data,record=fallback_fixture();data['operations'][-1]['name']='malformed-op';session=sharing.EmissionSession(data,record)
        with self.assertRaises(ValueError):session.demand_audit()
        self.assertFalse(session._ready)
        with self.assertRaisesRegex(ValueError,'closed or failed'):session.surviving_audit()
    def test_bool_int_packet_record_and_configuration_mutations_fail_closed(self):
        for name in ('packet','record','config'):
            with self.subTest(name=name):
                data,record=fallback_fixture();config=dict(flag=False);session=sharing.EmissionSession(data,record,extra_identity=config)
                target,key,new={'packet':(data,'schema_version',True),'record':(record['demand_coverage'],'exact',0),'config':(config,'flag',0)}[name]
                old=target[key];target[key]=new
                with self.assertRaisesRegex(ValueError,'changed'):session.demand_audit()
                target[key]=old
                with self.assertRaisesRegex(ValueError,'closed or failed'):session.demand_audit()
    def test_complete_file_bytes_checked_even_if_size_and_mtime_restored(self):
        data,record=fallback_fixture()
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);paths=[root/f for f in ('source.py','manifest.json','configuration.json')]
            for path in paths:path.write_bytes(b'abcd')
            for path in paths:
                session=sharing.EmissionSession(data,record,bindings=paths);stat=path.stat();path.write_bytes(b'efgh');os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
                with self.assertRaisesRegex(ValueError,'source/configuration/manifest changed'):session.demand_audit()
                path.write_bytes(b'abcd')
    def test_live_proof_replacement_cannot_inherit_witness(self):
        data,record=fallback_fixture();session=sharing.EmissionSession(data,record)
        with patch.object(backend,'validate_redistribution',lambda *_:None):
            with self.assertRaisesRegex(ValueError,'live emission/proof implementation changed'):session.assess()
    def test_returned_config_mutation_does_not_pollute_other_consumers(self):
        data,record=fallback_fixture()
        with sharing.EmissionSession(data,record) as session:
            result=session.assess(include_emission=True);result['execution_config']['root_order'].append(999)
            self.assertEqual(session.assess(include_emission=True),support.assess(data,include_emission=True))
            self.assertEqual(session.stats['emitter_runs'],1)
    def test_private_trace_snapshot_and_successful_proof_each_sealed(self):
        for variant in ('trace','snapshot','strong','access'):
            with self.subTest(variant=variant):
                data,record=exact_fixture();session=sharing.EmissionSession(data,record);session.demand_audit()
                if variant=='trace':session._info['_emission_trace']={'mutated':True}
                elif variant=='snapshot':session._state=b'bad snapshot'
                elif variant=='strong':session._proofs[False][0]['status']='explicit-fallback'
                else:next(iter(session._access_rows.values()))[0]['global_sectors32']+=1
                with self.assertRaisesRegex(ValueError,'changed'):session.receipt()
    def test_generated_config_bytes_and_existing_input_binding_are_strict(self):
        data,record=fallback_fixture();cfg=dict(threads=128)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);paths={name:root/(name+'.json') for name in ('ttgir','typed_config','configuration')}
            session=sharing.EmissionSession(data,record,extra_identity=cfg)
            result=session.assess(include_emission=True);paths['ttgir'].write_text(result['ttgir']);paths['typed_config'].write_text(json.dumps(result['execution_config'],indent=2)+'\n');paths['configuration'].write_text(json.dumps(cfg,indent=2)+'\n')
            bound=session.receipt(artifacts=paths);self.assertEqual(len(bound['input_files']),3)
            paths['typed_config'].write_text('{}')
            with self.assertRaisesRegex(ValueError,'source/configuration/manifest changed'):session.receipt()
            another=sharing.EmissionSession(data,record,bindings=[paths['configuration']],extra_identity=cfg);another.assess()
            with self.assertRaisesRegex(ValueError,'cannot replace an existing binding'):another.receipt(artifacts=paths)
    def test_snapshot_loader_has_no_unrestricted_global_resolution(self):
        with self.assertRaisesRegex(ValueError,'unknown emission snapshot global'):sharing._snapshot_load(pickle.dumps(eval))
    def test_distinct_sessions_do_not_inherit_emissions_or_proofs(self):
        data,record=fallback_fixture()
        with sharing.EmissionSession(data,record) as a:a.assess();self.assertEqual(a.stats['emitter_runs'],1)
        with self.assertRaisesRegex(ValueError,'closed or failed'):a.assess()
        with sharing.EmissionSession(data,record) as b:self.assertEqual(b.stats['emitter_runs'],0);b.assess();self.assertEqual(b.stats['emitter_runs'],1)

if __name__=='__main__':unittest.main()
