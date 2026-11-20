import copy
import types
import unittest
import backend


def packet(registers=512):
    values=[]
    for identity in range(4):
        addresses=[dict(view=[i],root=[i],begin_bit=32*i,end_bit=32*i+32) for i in range(registers)]
        values.append(dict(id=identity,type='memref<'+str(registers)+'xf32,1>',storage=dict(root=identity,alignment=1,layout='memory_space Global',addresses=addresses)))
    logical=[r for t in range(128) for r in range(registers)]
    values.append(dict(id=4,type='tensor<'+str(registers)+'xf32>',distributed=dict(shape=[registers],topology=[registers,32,4,1,1],logical=logical,first=list(range(registers)),replication=128)))
    operations=[dict(name='frisk.storage_contract',operands=[i],results=[],attrs={}) for i in range(4)]
    operations += [dict(name='frisk.tile_load',operands=[0],results=[4],attrs={}),dict(name='frisk.tile_store',operands=[4,3],results=[],attrs={})]
    return dict(schema_version=1,values=values,operations=operations)


class CountingEmitter(backend.Emitter):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.offset_builds=[]
    def _compute_offsets(self,value):
        self.offset_builds.append(value)
        return super()._compute_offsets(value)


class OffsetMemoTest(unittest.TestCase):
    def test_storage_value_and_view_ids_have_separate_entries(self):
        data=packet(8);view=copy.deepcopy(data['values'][0]);view['id']=5
        for a in view['storage']['addresses']:a['begin_bit']+=128;a['end_bit']+=128
        data['values'].append(view);e=CountingEmitter(data)
        root=e.offsets(0);alias=e.offsets(5)
        self.assertEqual(root[0],0);self.assertEqual(alias[0],16)
        self.assertIs(e.offsets(0),root);self.assertIs(e.offsets(5),alias)
        self.assertEqual(e.offset_builds,[0,5]);self.assertEqual(set(e._offset_cache),{0,5})

    def test_new_emitters_never_share_a_cache(self):
        a=CountingEmitter(packet(8));b=CountingEmitter(packet(8))
        self.assertIsNot(a.offsets(0),b.offsets(0))
        self.assertEqual(a.offset_builds,[0]);self.assertEqual(b.offset_builds,[0])
        self.assertIsNot(a._offset_cache,b._offset_cache)

    def test_exception_is_not_cached(self):
        data=packet(8);data['values'][0]['storage']['addresses'][0]['end_bit']=24;e=CountingEmitter(data)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError,'unsupported packed storage'):e.offsets(0)
        self.assertEqual(e.offset_builds,[0,0]);self.assertEqual(e._offset_cache,{})

    def test_complete_TTGIR_and_info_equal_original_bound_function(self):
        for registers in [8,512]:
            data=packet(registers);before=copy.deepcopy(data);memo=CountingEmitter(data);reference=CountingEmitter(copy.deepcopy(data))
            reference.offsets=types.MethodType(CountingEmitter._compute_offsets,reference)
            expected=reference.run();actual=memo.run()
            self.assertEqual(actual,expected)
            self.assertEqual(data,before)
            self.assertEqual(set(memo.offset_builds),{0,3})
            self.assertEqual(len(memo.offset_builds),2)
            self.assertEqual(len(reference.offset_builds),registers+1 if registers>256 else 2)
            cached={value:dict(table) for value,table in memo._offset_cache.items()}
            self.assertEqual(cached,{value:backend.Emitter._compute_offsets(memo,value) for value in cached})

if __name__=='__main__':unittest.main()
