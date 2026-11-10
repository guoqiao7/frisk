"""Actual compact-export demand lowering: layout/ownership/order regressions."""
import copy
import json
from pathlib import Path
import sys
import unittest
import backend
from execution_support import normalize_packet, LogicalTable

class DemandBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=json.loads(Path(PACKET).read_text())
        cls.normal=normalize_packet(cls.packet)

    def test_actual_large_packet_is_compact_and_emits_only_consumed_queries(self):
        self.assertEqual(self.packet['schema_version'],2)
        large=[v for v in self.normal['values'] if v.get('distributed',{}).get('topology',[0])[0]>256]
        self.assertTrue(large)
        self.assertTrue(all(isinstance(v['distributed']['logical'],LogicalTable) for v in large))
        text,info=backend.emit(self.packet)
        demand=info['demand_lowering']
        self.assertTrue(demand['preserves_checked_layouts'])
        self.assertTrue(demand['per_thread_reads_before_store'])
        self.assertLess(demand['materialized_query_vectors'],demand['declared_register_vectors'])
        self.assertLess(len(text),1000000)
        self.assertNotIn('shfl.sync',text)
        self.assertNotIn('st.shared',text)

    def test_all_global_reads_precede_the_first_global_write(self):
        text,info=backend.emit(self.packet)
        self.assertLess(text.rfind('ld.global.u8'),text.find('st.global.u8'))
        self.assertGreater(text.count('ld.global.u8'),0)
        self.assertGreater(text.count('st.global.u8'),0)

    def test_emission_is_deterministic_without_mutating_input(self):
        before=json.dumps(self.packet,sort_keys=True)
        first=backend.emit(self.packet)
        second=backend.emit(self.packet)
        self.assertEqual(first,second)
        self.assertEqual(before,json.dumps(self.packet,sort_keys=True))

    def test_two_stores_and_nonfinal_store_refuse(self):
        packet=copy.deepcopy(self.packet)
        store=next(o for o in packet['operations'] if o['name']=='frisk.tile_store')
        packet['operations'].insert(0,copy.deepcopy(store))
        with self.assertRaisesRegex(ValueError,'pure DAG and one final store'):
            backend.emit(packet)

    def test_pure_operand_layout_mismatch_cannot_be_silently_reindexed(self):
        packet=copy.deepcopy(self.packet)
        op=next(o for o in packet['operations'] if o['name']=='arith.mulf')
        ident=op['operands'][0]
        value=next(v for v in packet['values'] if v['id']==ident)
        rows=value['distributed']['symbolic']['rows_hex']
        rows[0],rows[1]=rows[1],rows[0]
        # Keep a valid full-rank map but break the operation's operand contract.
        with self.assertRaises(ValueError):backend.emit(packet)

if __name__=='__main__':
    PACKET=sys.argv[1]
    unittest.main(argv=[sys.argv[0]])
