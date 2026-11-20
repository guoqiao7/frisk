import unittest
from copy import deepcopy
from online_execution_contract import _bind_fa3_operations, bind_module, packet_contract, single_group, validate


class OnlineContractTest(unittest.TestCase):
    def test_fa3_stage_and_consumer_issue_order(self):
        orders = {
            'fa3_same_tile': ([5,6,7,8,9,10,11,12,13,14,15,16,18], 1),
            'fa3_qk_next': ([5,6,18,7,8,9,10,12,15,11,13,14,16], 2),
        }
        for expected,(consumer,pv_stage) in orders.items():
            ops=[]
            for group,ids in ((0,[0,1,2,3]),(1,[4,17]),(0,consumer),(0,[19,20,21])):
                for order,op in enumerate(ids):
                    stage = None
                    if op in consumer or op==17: stage = pv_stage if op==18 else 1
                    if op==4: stage = 0
                    ops.append(dict(operation_id=op,group_id=group,stage=stage,order=order))
            kind,roles=_bind_fa3_operations(list(reversed(ops)))
            self.assertEqual(kind,expected)
            self.assertEqual(roles,dict(q=0,k=1,v=1,qk=0,pv=0,reduce=0))
            # Relabeling only PV's stage leaves an incompatible issue order.
            next(o for o in ops if o['operation_id']==18)['stage']=3-pv_stage
            with self.assertRaisesRegex(ValueError,'issue order'): _bind_fa3_operations(ops)

    def test_single_wait_is_a_contract(self):
        for batch in (1,4):
            p=single_group(batch)
            ir=bind_module('module attributes {frisk.target="sm_90a"} {}',p)
            self.assertIn('batch='+str(batch)+':i64',ir)
            with self.assertRaises(ValueError): bind_module(ir,p)

    def test_single_rejects_ring_or_group_mismatch(self):
        p=single_group()
        p['buffers']['v']['versions']=2
        with self.assertRaises(ValueError): validate(p)
        p=single_group()
        ops=[{'name':'builtin.module','attrs':{'frisk.execution_contract':p}},
             {'name':'frisk.mma','attrs':{'frisk.execution_threads':384}}]
        with self.assertRaises(ValueError): packet_contract({'operations':ops})

    def test_ready_release_are_distinct(self):
        p=dict(kind='fa3_qk_next',version=1,batch=8,wait_pending=0,cta=384,
               roles=dict(q=0,k=1,v=1,qk=0,pv=0,reduce=0),
               groups=[dict(base=0,threads=256,increase=1,regs=240),
                       dict(base=256,threads=128,increase=0,regs=24)],
               buffers={key:dict(offset=offset,versions=versions,tile_bytes=32768)
                        for key,offset,versions in [('q',0,1),('k',32768,2),('v',98304,2),('o',0,1)]},
               arena=163840,barrier_slots=9,
               edges={name:dict(first=i,slots=v,arrivals=a,distance=d)
                      for name,i,v,a,d in [('q_ready',0,1,1,0),('k_ready',1,2,1,0),
                                           ('v_ready',3,2,1,0),('k_free',5,2,256,2),('v_free',7,2,256,2)]})
        validate(p)
        for key,value in [('distance',1),('slots',1),('arrivals',384),('first',6)]:
            bad=deepcopy(p);bad['edges']['v_free'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): validate(bad)
