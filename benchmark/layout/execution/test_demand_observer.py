import copy
import json
from pathlib import Path
import sys
import unittest
import backend

class DemandObserverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=json.loads(Path(PACKET).read_text());cls.eager=json.loads(Path(EAGER).read_text())
        cls.old={};exec(compile(Path(PREIMAGE).read_text(),PREIMAGE,'exec'),cls.old)
    def test_default_and_observed_emit_equal_preimage_bytes_and_config(self):
        for packet in (self.packet,self.eager):
            before=self.old['emit'](packet);events=[]
            self.assertEqual(backend.emit(packet),before)
            self.assertEqual(backend.emit(packet,demand_query_observer=lambda v,q:events.append((v,q))),before)
    def test_each_materialized_cache_query_observed_once(self):
        events=[];text,info=backend.emit(self.packet,demand_query_observer=lambda v,q:events.append((v,q)))
        demand=info['demand_lowering']
        self.assertEqual(len(events),demand['materialized_query_vectors'])
        self.assertEqual(len(set(events)),len(events))
        counts={v:sum(x==v for x,q in events) for v,q in events}
        self.assertEqual(counts,demand['queried_register_vectors'])
    def test_query_is_immutable_and_input_is_not_mutated(self):
        before=json.dumps(self.packet,sort_keys=True);events=[]
        def observe(v,q):
            self.assertIsInstance(q,tuple);self.assertEqual(len(q),128)
            with self.assertRaises(TypeError):q[0]=0
            events.append((v,q))
        backend.emit(self.packet,demand_query_observer=observe)
        self.assertTrue(events);self.assertEqual(json.dumps(self.packet,sort_keys=True),before)
    def test_actual_queries_have_full_domain_checked_affine_signatures(self):
        events=[];backend.emit(self.packet,demand_query_observer=lambda v,q:events.append((v,q)))
        for v,q in events:
            base,bits=backend.linear(q);self.assertEqual(len(bits),7)
            for t,value in enumerate(q):
                expected=base
                for bit,column in enumerate(bits):
                    if (t>>bit)&1:expected^=column
                self.assertEqual(expected,value)
    def test_eager_emission_does_not_invent_demand_events(self):
        events=[];_,info=backend.emit(self.eager,demand_query_observer=lambda v,q:events.append((v,q)))
        self.assertNotIn('demand_lowering',info);self.assertEqual(events,[])
    def test_query_compilation_budget_rejects_real_recursive_materialization(self):
        with self.assertRaisesRegex(ValueError,'query budget exceeded'):
            backend.emit(self.packet,demand_query_budget=1)
        with self.assertRaisesRegex(ValueError,'carrier budget exceeded'):
            backend.emit(self.eager,demand_query_budget=1)
        for value in (0,65537,True):
            with self.assertRaises(ValueError):backend.emit(self.packet,demand_query_budget=value)
    def test_invalid_observer_rejected_before_lowering(self):
        with self.assertRaises(TypeError):backend.emit(self.packet,demand_query_observer=[])
    def test_observer_error_is_not_silently_accepted(self):
        def fail(v,q):raise RuntimeError('observer-audit-failure')
        with self.assertRaisesRegex(RuntimeError,'observer-audit-failure'):backend.emit(self.packet,demand_query_observer=fail)
if __name__=='__main__':
    PACKET,EAGER,PREIMAGE=sys.argv[1:4]
    unittest.main(argv=[sys.argv[0]])
