"""CPU-only exact compact-address codec, typed seals and dense fallback regressions."""
import copy
import unittest
from storage_addresses import (RowMajorAddresses, RowMajorByteOffsets, normalize_storage_values,
                               global_root_spans, _linear_expression)
from execution_support import normalize_packet
from redistribution import content_digest
from shared_allocation import _extent, plan_shared_allocation
import backend
from test_execution_support import packet

def descriptor(shape=(2,64)):
    return dict(contract='contiguous-identity-addresses-v1',shape=list(shape),element_bits=32,base_bit=0,strides_bits=[32*shape[1],32])

def layout(shape=(2,64),expression=None):
    m,n=shape
    return '#frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = ['+str(m)+', '+str(n)+'], outputs = ["byte_offset", "bit_offset"], output_extents = ['+str(4*m*n)+', 8], map = affine_map<(d0, d1) -> ('+(expression or '(d0 * '+str(n)+' + d1) * 4')+', 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>'

def fixture(compact=True):
    result=packet();result['schema_version']=2 if compact else 1
    for value in result['values'][:4]:
        value['type']='memref<2x64xf32, 1>';storage=value['storage'];storage['layout']=layout()
        if compact:storage.pop('addresses');storage['addresses_compact']=descriptor()
        else:storage['addresses']=list(RowMajorAddresses(descriptor()))
    return result

class CompactAddressesTests(unittest.TestCase):
    def test_exact_rows_offsets_extent_for_boundaries_and_singletons(self):
        for shape in [(1,1),(1,64),(128,1),(2,64),(512,128),(256,256)]:
            table=RowMajorAddresses(descriptor(shape));offsets=table.byte_offsets()
            for i in range(len(table)):
                coordinate=list(divmod(i,shape[1]))
                self.assertEqual(table[i],dict(begin_bit=32*i,end_bit=32*(i+1),root=coordinate,view=coordinate))
                self.assertEqual(offsets[i],4*i)
            self.assertEqual(_extent(dict(addresses=table)),(4*len(table),32*len(table)))
            self.assertEqual(dict(offsets),{i:4*i for i in range(len(table))})
    def test_descriptor_schema_and_exact_types_fail_closed(self):
        mutations=[lambda d:d.update(extra=0),lambda d:d.pop('shape'),lambda d:d.update(contract='v2'),lambda d:d.update(shape=[True,64]),lambda d:d.update(shape=[0,64]),lambda d:d.update(shape=[2,64,1]),lambda d:d.update(shape=(2,64)),lambda d:d.update(shape=[1024,128]),lambda d:d.update(element_bits=True),lambda d:d.update(element_bits=16),lambda d:d.update(base_bit=32),lambda d:d.update(base_bit=False),lambda d:d.update(strides_bits=[64,32]),lambda d:d.update(strides_bits=[2048,True])]
        for mutate in mutations:
            value=descriptor();mutate(value)
            with self.subTest(value=value),self.assertRaises(ValueError):RowMajorAddresses(value)
    def test_snapshot_and_returned_coordinates_are_independent(self):
        original=descriptor();table=RowMajorAddresses(original);original['shape'][0]=9
        self.assertEqual(len(table),128);a=table[0];a['root'][0]=99
        self.assertEqual(a['view'],[0,0]);self.assertEqual(table[0]['root'],[0,0]);self.assertIs(copy.deepcopy(table),table)
        with self.assertRaises(AttributeError):table._count=4
        with self.assertRaises(AttributeError):table.byte_offsets()._count=4
        self.assertEqual(table[-1],table[127]);self.assertEqual(table[1:3],[table[1],table[2]])
        for key in [True,1.0,'0']:
            with self.assertRaises(TypeError):table[key]
        for key in [128,-129]:
            with self.assertRaises(IndexError):table[key]
        for key in [-1,128,True,1.0]:
            with self.assertRaises(KeyError):table.byte_offsets()[key]
    def test_normalize_idempotent_materialize_and_dense_noop(self):
        raw=fixture();normalized=normalize_packet(raw)
        self.assertEqual(normalized['schema_version'],1);self.assertIs(type(normalized['values'][0]['storage']['addresses']),RowMajorAddresses)
        self.assertNotIn('addresses_compact',normalized['values'][0]['storage']);self.assertIs(normalize_packet(normalized),normalized)
        dense=fixture(False);self.assertIs(normalize_packet(dense),dense)
        self.assertEqual(normalize_packet(raw,materialize=True),dense)
        self.assertEqual(normalize_packet(normalized,materialize=True),dense)
        raw['values'][0]['storage']['addresses_compact']['shape'][0]=1
        self.assertEqual(len(normalized['values'][0]['storage']['addresses']),128)
    def test_context_mutations_are_rejected(self):
        changes=[lambda p:p.update(schema_version=1),lambda p:p['values'][0]['storage'].update(addresses=[]),lambda p:p['values'][0]['storage'].update(root=True),lambda p:p['values'][0]['storage'].update(root=99),lambda p:p['values'][0]['storage'].update(alignment=True),lambda p:p['values'][0]['storage'].update(alignment=2),lambda p:p['values'][0].update(type='memref<2x64xf16, 1>'),lambda p:p['values'][0].update(type='memref<2x64xf32, 3>'),lambda p:p['values'][0]['storage'].update(layout=layout().replace('Global','Shared')),lambda p:p['values'][0]['storage'].update(layout=layout().replace('vector_granularity = 1','vector_granularity = 2')),lambda p:p['values'][0]['storage'].update(layout=layout(expression='d0 * 4 + d1 * 8')),lambda p:p['values'][0]['storage'].update(layout=layout().replace('[512, 8]','[511, 8]')),lambda p:p['values'][0].update(id=1)]
        for change in changes:
            value=fixture();change(value)
            with self.subTest(change=change),self.assertRaises(ValueError):normalize_packet(value)
    def test_coordinate_identity_alias_with_compact_or_exact_dense_root(self):
        raw=fixture();alias=copy.deepcopy(raw['values'][0]);alias['id']=9;raw['values'].append(alias)
        normalized=normalize_packet(raw);self.assertEqual(normalized['values'][-1]['storage']['root'],0)
        raw['values'][0]=fixture(False)['values'][0];self.assertEqual(len(normalize_packet(raw)['values'][-1]['storage']['addresses']),128)
        raw['values'][0]['storage']['addresses'][0]['root'][0]=1
        with self.assertRaises(ValueError):normalize_packet(raw)
    def test_affine_positive_literal_grammar(self):
        for expression in ['(d0 * 64 + d1) * 4','d0 * 256 + d1 * 4','4 * (64 * d0 + d1)','((d0 * 64) + (d1)) * 4']:
            self.assertEqual(_linear_expression(expression),(256,4,0))
            raw=fixture();raw['values'][0]['storage']['layout']=layout(expression=expression);normalize_packet(raw)
        for expression in ['d0*d1','0*d0+d1','d0*-1+d1','(d0//1)*256+d1*4','s0+d1','__import__("os")','(d0+1','d0 d1','d0 ** 2','9223372036854775808*d0']:
            with self.subTest(expression=expression),self.assertRaises(ValueError):_linear_expression(expression)
    def test_explicit_identity_memref_descriptor_and_space(self):
        for type_ in ['memref<2x64xf32, strided<[64, 1]>, 1>','memref<2x64xf32, strided<[64, 1], offset: 0>, #frisk<memory_space Global>>','memref<2x64xf32, #frisk<memory_space Global>>']:
            raw=fixture();raw['values'][0]['type']=type_;normalize_packet(raw)
        for type_ in ['memref<2x64xf32, strided<[65, 1]>, 1>','memref<2x64xf32, strided<[64, 1], offset: 1>, 1>']:
            raw=fixture();raw['values'][0]['type']=type_
            with self.assertRaises(ValueError):normalize_packet(raw)
    def test_complete_typed_digest_and_entry_state_mutations(self):
        normalized=normalize_packet(fixture());old=content_digest(normalized);self.assertIsNotNone(old)
        normalized['values'][0]['storage']['root']=1;self.assertNotEqual(content_digest(normalized),old)
        normalized=normalize_packet(fixture());table=normalized['values'][0]['storage']['addresses'];old=content_digest(normalized)
        object.__setattr__(table,'_count',64);self.assertNotEqual(content_digest(normalized),old)
        with self.assertRaises(ValueError):normalize_packet(normalized)
        class Unknown:
            def __reduce__(self):raise AssertionError('must never invoke an unknown reducer')
        self.assertIsNone(content_digest(Unknown()))
    def test_full_small_emission_and_tiling_byte_equal(self):
        from tiled_execution import lift_emission
        dense=fixture(False);compact=fixture();a,ac=backend.emit(dense);b,bc=backend.emit(compact)
        self.assertEqual(a,b);self.assertEqual(ac,bc);self.assertEqual(global_root_spans(compact,bc['root_order']),[512]*4)
        for count in [1,2,257]:self.assertEqual(lift_emission(dense,a,ac,tile_count=count),lift_emission(compact,b,bc,tile_count=count))
        self.assertEqual(plan_shared_allocation(dense),plan_shared_allocation(compact))
    def test_observer_changes_to_compact_carrier_are_rejected(self):
        raw=fixture();emitter=None
        def observer(_):
            if emitter is not None:object.__setattr__(emitter.data['values'][0]['storage']['addresses'],'_count',64)
        emitter=backend.Emitter(raw,demand_query_observer=observer)
        # Directly exercise the exact entry/exit seal even when this small graph
        # uses the ordinary lowering path rather than demand-query callbacks.
        original=emitter._packet_content_seal
        object.__setattr__(emitter.data['values'][0]['storage']['addresses'],'_count',64)
        self.assertNotEqual(content_digest(emitter.data),original)
        with self.assertRaises(ValueError):emitter.run()

if __name__=='__main__':unittest.main()
