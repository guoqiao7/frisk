"""Independent algebra and content-invalidation tests for CPU emit reuse."""
import copy
import random
import unittest
from unittest.mock import patch
import backend
from execution_support import SymbolicMap,normalize_distribution
from redistribution import ProofCache,content_digest,validate
from shuffle_network import plan_shuffle_network,shuffle_network_candidates,network_resource_proxy
from test_execution_support import distribution,packet
from test_shuffle_network_codegen import permutation_case


class CanonicalMapReuse(unittest.TestCase):
    def test_first_section_xor_expansion_matches_exact_solver(self):
        rng=random.Random(20261105)
        for shape in [[],[1],[2],[2,4],[4,2],[1,4,2],[4,4,4],[256,256]]:
            bits=sum(x.bit_length()-1 for x in shape)
            hardware=max(bits,7);topology=[1<<(hardware-7),32,4,1,1]
            rows=[1<<i for i in range(bits)]
            for _ in range(5*bits):
                if bits>1:
                    i,j=rng.sample(range(bits),2);rows[i]^=rows[j]
            mapping=SymbolicMap(distribution(shape,topology,rows))
            expected=[mapping.first(i) for i in range(mapping.volume)]
            with patch.object(mapping,'first',wraps=mapping.first) as called:
                self.assertEqual(mapping.first_table(),expected)
                self.assertEqual(called.call_count,bits)
            for i in range(mapping.holders):
                self.assertEqual(mapping.logical(i),mapping.flatten(mapping.packed(i)))

    def test_rectangular_maps_with_nonzero_dependent_columns(self):
        rng=random.Random(20261106)
        for _ in range(15):
            shape=[4,8];bits=5;hardware=10
            rows=[(1<<i)|sum(rng.randrange(2)<<j for j in range(bits,hardware)) for i in range(bits)]
            mapping=SymbolicMap(distribution(shape,[8,32,4,1,1],rows))
            self.assertEqual(mapping.first_table(),[mapping.first(i) for i in range(mapping.volume)])
            for i in range(mapping.holders):self.assertEqual(mapping.logical(i),mapping.flatten(mapping.packed(i)))

    def test_byte_lookup_public_dependency_replacement_falls_back(self):
        mapping=SymbolicMap(distribution([2,4],[1,32,4,1,1],[1,2,4]))
        mapping.rows=(mapping.rows[0]^mapping.rows[1],)+mapping.rows[1:]
        for i in range(mapping.holders):self.assertEqual(mapping.logical(i),mapping.flatten(mapping.packed(i)))
        mapping.shape=(4,2)
        for i in range(mapping.holders):self.assertEqual(mapping.logical(i),mapping.flatten(mapping.packed(i)))
        for value in [True,-1,mapping.holders]:
            with self.assertRaises(IndexError):mapping.logical(value)

    def test_new_raw_content_has_independent_normalization(self):
        raw=distribution([4],[1,32,4,1,1],[1,2]);a=normalize_distribution(raw)
        raw['symbolic']['rows_hex']=['2','1'];b=normalize_distribution(raw)
        self.assertNotEqual(list(a['logical']),list(b['logical']))
        self.assertEqual(a['first'],[0,1,2,3]);self.assertEqual(b['first'],[0,2,1,3])


class ProofContentTests(unittest.TestCase):
    def fixture(self):return permutation_case(3,[(0,0)])

    def test_equal_content_new_objects_hit_but_mutated_return_fails(self):
        operation,values=self.fixture();cache=ProofCache()
        result=cache.validate(operation,values)
        copied=copy.deepcopy((operation,values))
        self.assertIs(cache.validate(*copied),copied[0]['redistribution'])
        self.assertEqual((cache.misses,cache.hits),(1,1))
        result['scratch_bytes']=4
        with self.assertRaises(ValueError):cache.validate(operation,values)
        self.assertEqual(cache.misses,2)

    def test_owner_bool_source_type_and_list_tuple_cannot_reuse(self):
        changes=[]
        operation,values=self.fixture();cache=ProofCache();cache.validate(operation,values)
        operation['redistribution']['destinations'][0]['source']['reg']=False
        with self.assertRaises(ValueError):cache.validate(operation,values)
        operation,values=self.fixture();cache=ProofCache();cache.validate(operation,values)
        values[0]['type']=values[0]['type'].replace('f32','f16')
        with self.assertRaises(ValueError):cache.validate(operation,values)
        operation,values=self.fixture();cache=ProofCache();cache.validate(operation,values)
        values[0]['distributed']['topology']=tuple(values[0]['distributed']['topology'])
        with self.assertRaises(ValueError):cache.validate(operation,values)
        self.assertNotEqual(content_digest([1,2]),content_digest((1,2)))
        self.assertNotEqual(content_digest({'a':1}),content_digest({'a':True}))

    def test_unknown_objects_do_not_execute_reduce_or_get_cached(self):
        class Unknown:
            def __reduce__(self):raise AssertionError('must not invoke unknown pickle reducer')
        operation,values=self.fixture();operation['attrs']={'opaque':Unknown()};cache=ProofCache()
        cache.validate(operation,values);cache.validate(operation,values)
        self.assertEqual((cache.hits,cache.misses,cache.uncacheable),(0,2,2))

    def test_failed_proofs_not_stored_and_receipts_bounded(self):
        operation,values=self.fixture();cache=ProofCache(max_entries=1);cache.validate(operation,values)
        operation['attrs']={'new':'content'};cache.validate(operation,values);cache.validate(operation,values)
        self.assertEqual((len(cache.receipts),cache.hits,cache.misses),(1,0,3))
        operation['redistribution']['scratch_bytes']=4
        for _ in range(2):
            with self.assertRaises(ValueError):cache.validate(operation,values)
        self.assertEqual(len(cache.receipts),1)

    def test_transforming_validator_is_not_replaced_by_success_marker(self):
        operation,values=self.fixture();calls=[]
        def transformed(operation,values):
            calls.append(1)
            result=copy.deepcopy(validate(operation,values));result['checked_transform']=len(calls)
            return result
        cache=ProofCache(validator=transformed)
        self.assertEqual(cache.validate(operation,values)['checked_transform'],1)
        self.assertEqual(cache.validate(operation,values)['checked_transform'],2)
        self.assertEqual((cache.hits,cache.misses,len(cache.receipts)),(0,2,0))

    def test_unknown_packet_carrier_disables_emitter_receipts(self):
        class Unknown:
            def __reduce__(self):raise AssertionError('must not invoke unknown reducer')
        operation,values=self.fixture()
        emitter=backend.Emitter(dict(values=list(values.values()),opaque=Unknown()))
        self.assertIsNone(emitter._packet_content_seal)
        with patch.object(backend,'validate_redistribution',wraps=validate) as checked:
            emitter._checked_redistribution(operation);emitter._checked_redistribution(operation)
            self.assertEqual(checked.call_count,2)
        self.assertEqual(len(emitter._redistribution_proofs.receipts),0)

    def test_validator_implementation_changes_clear_receipts(self):
        operation,values=self.fixture();cache=ProofCache();cache.validate(operation,values)
        def changed(*args):raise ValueError('new proof contract')
        cache.validator=changed
        with self.assertRaisesRegex(ValueError,'new proof contract'):cache.validate(operation,values)
        self.assertEqual(cache.hits,0)
        emitter=backend.Emitter(dict(values=list(values.values())))
        emitter.reg[0]=['source'+str(i) for i in range(8)];emitter.convert(operation)
        with patch.object(backend,'validate_redistribution',side_effect=ValueError('replacement proof')):
            with self.assertRaisesRegex(ValueError,'replacement proof'):emitter.convert(operation)

    def test_declaration_objective_proxy_and_typed_tamper(self):
        operation,values=permutation_case(4,[(i,i) for i in range(4)])
        plan=operation['redistribution'];default=plan_shuffle_network(plan)
        self.assertEqual(default,plan_shuffle_network(plan,objective='min-shfl'))
        alternatives=[x for x in shuffle_network_candidates(plan) if x['counts']['total_work']<=x['legacy_work'] and x['counts']['shfl']<=x['legacy_shfl']]
        work=plan_shuffle_network(plan,objective='min-work')
        self.assertEqual(work,min(alternatives,key=lambda x:(x['counts']['total_work'],x['counts']['shfl'],x['kind'])))
        self.assertNotEqual(default,work)
        plan['shuffle_network_objective']='min-work';plan['shuffle_network']=work;plan['shuffle_network_resource_proxy']=network_resource_proxy(work)
        cache=ProofCache();cache.validate(operation,values);cache.validate(operation,values)
        plan['shuffle_network_resource_proxy']['physical_register_prediction']=0
        with self.assertRaisesRegex(ValueError,'resource proxy'):cache.validate(operation,values)
        plan['shuffle_network_resource_proxy']=network_resource_proxy(work);plan['shuffle_network_objective']='min-shfl'
        with self.assertRaisesRegex(ValueError,'network declaration'):cache.validate(operation,values)
        with self.assertRaises(ValueError):plan_shuffle_network(plan,objective='unknown')


class WidthHoistTests(unittest.TestCase):
    @staticmethod
    def large_packet():
        data=packet()
        for value in data['values']:
            if 'distributed' in value:
                value['distributed'].update(topology=[512,32,4,1,1],replication=512,
                    logical=[t for t in range(128) for _ in range(512)],first=[t*512 for t in range(128)])
        return data

    def test_one_descriptor_proof_for_many_load_queries(self):
        data=self.large_packet();calls=[];original=backend.content_digest
        def observed(value):
            if type(value) is dict and 'addresses' in value:calls.append(value)
            return original(value)
        with patch.object(backend,'content_digest',side_effect=observed):_,info=backend.emit(data)
        self.assertEqual(info['demand_lowering']['load_vectors'],512)
        self.assertEqual(len(calls),2)  # Initial full-content proof and final content recheck.
        data['values'][0]['storage']['addresses'][0]['end_bit']+=1
        with self.assertRaisesRegex(ValueError,'unsupported packed storage|f32 storage'):backend.emit(data)

    def test_observer_after_plans_mutates_register_proof_is_rejected(self):
        data=self.large_packet();value=copy.deepcopy(data['values'][4]);value['id']=6;data['values'].append(value)
        for operation in data['operations']:
            if operation['name']=='arith.mulf':operation['operands']=[6,6]
        def owner(thread):return dict(reg=0,lane=thread%32,warp=thread//32,warp_group=0)
        plan=dict(mode='Register',threads=128,warps=4,source_registers=512,destination_registers=512,
            volume=128,scratch_bytes=0,payload=dict(bits=32,words=[dict(bit_offset=0,live_bits=32)]),
            source_slots=data['values'][4]['distributed']['logical'],global_writers=[owner(t) for t in range(128)],
            destinations=[dict(slot=t,source=owner(t)) for t in range(128) for _ in range(512)])
        conversion=dict(name='frisk.convert_layout',operands=[4],results=[6],attrs={},redistribution=plan)
        index=next(i for i,o in enumerate(data['operations']) if o['name']=='arith.mulf');data['operations'].insert(index,conversion)
        changed=[]
        def observe(value,indices):
            if not changed:plan['scratch_bytes']=4;changed.append(True)
        with self.assertRaisesRegex(ValueError,'execution content changed during emission'):
            backend.emit(data,demand_query_observer=observe)
        self.assertTrue(changed)

    def test_observer_mutation_is_not_silently_accepted(self):
        data=self.large_packet();holder={};changed=[]
        def observer(value,indices):
            if not changed:
                holder['emitter'].storage(0)['addresses'][0]['end_bit']+=8;changed.append(True)
        emitter=backend.Emitter(data,demand_query_observer=observer);holder['emitter']=emitter
        with self.assertRaisesRegex(ValueError,'storage content changed'):emitter.run()


if __name__=='__main__':unittest.main()
