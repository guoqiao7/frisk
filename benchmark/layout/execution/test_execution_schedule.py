"""Proof boundaries for synchronous Copy scheduling and Shared reuse."""

import copy
import unittest

from execution_schedule import (attention_late_schedule, attention_publication, place_reusable_roots,
                                shared_schedule, verify_attention_publication)


def storage(ident, space):
    return dict(id=ident,storage=dict(root=ident,layout='memory_space '+space,
                addresses=[dict(begin_bit=0,end_bit=16)]))


def op(name, operands=(), attrs=None):
    return dict(name=name,operands=list(operands),attrs=attrs or {})


def packet():
    contract={'frisk.copy_contract':dict(path='vector',completion='synchronous',vector_bytes=16)}
    operations=[op('frisk.copy',(0,2),contract),op('frisk.copy',(1,3),contract),
                op('frisk.mma',(2,)),op('arith.mulf'),op('frisk.mma',(3,)),op('frisk.tile_store',(4,5))]
    return dict(values=[storage(0,'Global'),storage(1,'Global'),storage(2,'Shared'),
                        storage(3,'Shared'),dict(id=4),storage(5,'Global')],operations=operations)


def attention_packet():
    copy_contract = {'frisk.copy_contract': dict(path='vector', completion='synchronous', vector_bytes=16)}
    operations = [op('frisk.copy', (source, source + 4), copy_contract) for source in range(3)]
    operations.extend([
        op('frisk.mma', (4, 5, 7), {'frisk.mma_contract': {'form': 'ss'}, 'm': 64,
                                    'n': 128, 'k': 16, 'trans_b': -1}),
        op('frisk.reduce_tensor', (8,)),
        op('arith.subf', (8, 9)),
        op('frisk.reduce_tensor', (10,)),
        op('arith.truncf', (10, 11)),
        op('frisk.mma', (12, 6, 13), {'frisk.mma_contract': {'form': 'rs'}, 'm': 64,
                                     'n': 16, 'k': 128, 'trans_b': 0}),
        op('frisk.tile_store', (14, 3)),
    ])
    for entry, results in zip(operations, ([], [], [], [8], [9], [10], [11], [12], [14], [])):
        entry['results'] = results
    values = [storage(i, 'Global') for i in range(4)]
    values.extend(storage(i, 'Shared') for i in range(4, 7))
    values.extend({'id': i} for i in range(7, 15))
    return dict(operations=operations, values=values)


class ScheduleTest(unittest.TestCase):
    def test_late_copy_reuses_after_last_reader_with_barrier(self):
        original=packet()
        ordered, intervals=shared_schedule(original)
        self.assertEqual([x['name'] for x in ordered],['frisk.copy','frisk.mma','arith.mulf',
                                                     'frisk.copy','frisk.mma','frisk.tile_store'])
        self.assertEqual(intervals,{2:(0,1),3:(3,4)})
        roots,barriers=place_reusable_roots({2:(32,256),3:(32,256)},{2:16,3:16},intervals,32)
        self.assertEqual([r['offset'] for r in roots],[0,0])
        self.assertEqual(barriers,[3])
        self.assertEqual(original,packet())

    def test_intersecting_and_inclusive_lifetimes_cannot_alias(self):
        roots,barriers=place_reusable_roots({1:(32,256),2:(16,128)},{1:32,2:16},{1:(0,3),2:(3,4)},48)
        self.assertEqual([r['offset'] for r in roots],[0,32])
        self.assertEqual(barriers,[])

    def test_unknown_effect_duplicate_copy_partial_copy_and_alias_are_rejected(self):
        cases=[]
        p=packet();p['operations'].insert(3,op('frisk.tile_load',(2,)));cases.append(p)
        p=packet();p['operations'].insert(1,p['operations'][0]);cases.append(p)
        p=packet();p['values'].append(storage(6,'Shared'))
        p['values'][-1]['storage']['root']=2
        p['values'][-1]['storage']['addresses'][0]['end_bit']=8
        p['operations'][0]['operands'][1]=6;cases.append(p)
        p=packet();p['operations'][-1]['operands'][1]=1;cases.append(p)
        for p in cases:
            with self.subTest(p=p):
                with self.assertRaises(ValueError):
                    shared_schedule(p)

    def test_noncomplete_or_async_initialization_rejected(self):
        p=packet();p['operations'][0]=copy.deepcopy(p['operations'][0])
        p['operations'][0]['attrs']['frisk.copy_contract']['path']='cp.async'
        with self.assertRaisesRegex(ValueError,'synchronous'):
            shared_schedule(p)


    def test_attention_proves_exact_k_reuse_and_publication_order(self):
        _, intervals, roles = attention_late_schedule(attention_packet())
        roots, barriers = place_reusable_roots(
            {4: (16, 128), 5: (32, 256), 6: (32, 256)}, {4: 16, 5: 16, 6: 16},
            intervals, 48, reuse_after={6: 5})
        self.assertEqual([entry['offset'] for entry in roots], [0, 16, 16])
        self.assertEqual(barriers, [7])
        proof = attention_publication(roles, dict(allocations=roots, reuse_barriers=barriers),
                                      [{'operations': [0, 1]}], reuse_k=True)
        publish = ['bar.sync 0;', 'fence.proxy.async.shared::cta;']
        store = 'st.shared.v4.b32 [address], data;'
        mma = ['wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 out, a, b, c;',
               'wgmma.commit_group.sync.aligned;', 'wgmma.wait_group.sync.aligned 0;']
        code = {0: [store], 1: [store, *publish], 2: mma, 7: [publish[0], store, *publish], 8: mma}
        checked = verify_attention_publication(proof, code)
        self.assertEqual(checked['wgmma_wait_group_0'], {'qk': 1, 'pv': 1})
        self.assertEqual(checked['cta_barriers'], {'publish_qk': 1, 'before_v_overwrite': 1, 'publish_v': 1})
        self.assertEqual(checked['proxy_fences'], {'publish_qk': 1, 'publish_v': 1})
        with self.assertRaisesRegex(ValueError, 'fully drained'):
            verify_attention_publication(proof, code | {2: mma[:-1]})
        with self.assertRaisesRegex(ValueError, 'publication differs'):
            verify_attention_publication(proof, code | {7: [store, *publish]})
        with self.assertRaisesRegex(ValueError, 'publication differs'):
            verify_attention_publication(proof, code | {7: [publish[0], store, publish[0]]})
        with self.assertRaisesRegex(ValueError, 'unexpected CTA barrier'):
            attention_publication(roles, dict(allocations=roots, reuse_barriers=[]),
                                  [{'operations': [0, 1]}], reuse_k=True)
        wrong_order = copy.deepcopy(roles)
        wrong_order['v']['reader_operation'] = wrong_order['v']['copy_operation']
        with self.assertRaisesRegex(ValueError, 'expected ordered'):
            attention_publication(wrong_order, dict(allocations=roots, reuse_barriers=barriers),
                                  [{'operations': [0, 1]}], reuse_k=True)
        wrong_lifetime = copy.deepcopy(roots)
        wrong_lifetime[1]['lifetime'][1] = roles['v']['copy_operation']
        with self.assertRaisesRegex(ValueError, 'lifetime differs'):
            attention_publication(roles, dict(allocations=wrong_lifetime, reuse_barriers=barriers),
                                  [{'operations': [0, 1]}], reuse_k=True)

    def test_exact_reuse_rejects_unequal_live_or_misaligned_intervals(self):
        for sizes, alignments, intervals in (
                ({1: (32, 256), 2: (16, 128)}, {1: 16, 2: 16}, {1: (0, 2), 2: (3, 4)}),
                ({1: (32, 256), 2: (32, 256)}, {1: 16, 2: 16}, {1: (0, 3), 2: (3, 4)}),
                ({0: (16, 128), 1: (32, 256), 2: (32, 256)}, {0: 16, 1: 16, 2: 32},
                 {0: (0, 2), 1: (1, 2), 2: (3, 4)})):
            with self.subTest(intervals=intervals):
                with self.assertRaisesRegex(ValueError, 'dead, equal-sized aligned'):
                    place_reusable_roots(sizes, alignments, intervals, 64, reuse_after={2: 1})

    def test_late_schedule_requires_pv_to_depend_on_both_softmax_reductions(self):
        data = attention_packet()
        next(entry for entry in data['operations'] if entry['name'] == 'arith.truncf')['operands'] = [10]
        with self.assertRaisesRegex(ValueError, 'PV input must depend'):
            attention_late_schedule(data)
