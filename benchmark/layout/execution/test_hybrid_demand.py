"""CPU checks of actual hybrid demand packet, communication and old-path bytes.
Usage: python test_hybrid_demand.py ACTUAL_R1_FAILING_EXECUTION_JSON
"""
import ast
import copy
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import backend
from execution_support import normalize_packet
from redistribution import validate

ROOT=Path(__file__).resolve().parents[3]
PREIMAGE=ROOT/"guoqiao/optimize_plan/model_selection_quality_artifacts/development/hybrid-demand-repair/backend.py.preimage"
OLD_RUN=ROOT/"guoqiao/optimize_plan/current_gpu_closure_artifacts/run-current-gpu-closure-20261102T081806Z/cases"


def previous_backend():
    module=types.ModuleType("backend_before_hybrid")
    exec(compile(PREIMAGE.read_text(),str(PREIMAGE),"exec"),module.__dict__)
    return module


def method_ast(text,name):
    cls=next(x for x in ast.parse(text).body if isinstance(x,ast.ClassDef) and x.name=="Emitter")
    return ast.dump(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)


class HybridDemand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet=json.loads(Path(PACKET).read_text())
        cls.normal=normalize_packet(cls.packet)
        cls.text,cls.info=backend.emit(cls.packet)
        cls.cross=[o for o in cls.normal["operations"] if o["name"]=="frisk.convert_layout" and o["redistribution"]["mode"]!="Register"]

    def test_actual_prior_rejection_is_now_checked_hybrid(self):
        with self.assertRaisesRegex(ValueError,"same-thread Register routes"):
            previous_backend().emit(self.packet)
        self.assertEqual(self.info["demand_lowering"]["version"],"hybrid-demand-slots-v2")
        self.assertEqual(len(self.cross),4)
        self.assertTrue(all(o["redistribution"]["mode"]=="Shared" for o in self.cross))

    def test_each_cross_thread_ssa_materialized_exactly_once(self):
        seen=[];original=backend.Emitter.convert
        def checked(instance,operation):
            seen.append(operation["results"][0]);return original(instance,operation)
        with patch.object(backend.Emitter,"convert",checked):
            text,info=backend.emit(self.packet)
        expected=[o["results"][0] for o in self.cross]
        self.assertEqual(seen,expected)
        self.assertEqual(len(seen),len(set(seen)))
        metadata=info["demand_lowering"]["materialized_cross_thread_boundaries"]
        self.assertEqual([x["value"] for x in metadata],expected)
        self.assertTrue(all((x["mode"],x["source_registers"],x["destination_registers"])==("Shared",8,8) for x in metadata))
        self.assertEqual((text,info),(self.text,self.info))

    def test_four_shared_exchanges_keep_eight_unconditional_barriers(self):
        self.assertEqual(self.text.count("bar.sync 0;"),8)
        self.assertEqual(self.text.count("st.shared.b32"),32)
        self.assertEqual(self.text.count("ld.shared.b32"),32)
        lines=self.text.split("\\0A")
        barriers=[i for i,line in enumerate(lines) if line=="bar.sync 0;"]
        self.assertEqual(len(barriers),8)
        start=0
        for i in range(0,len(barriers),2):
            before=lines[start:barriers[i]];exchange=lines[barriers[i]+1:barriers[i+1]]
            self.assertEqual(sum("st.shared.b32" in x for x in before),8)
            self.assertEqual(sum("ld.shared.b32" in x for x in before),0)
            self.assertEqual(sum("ld.shared.b32" in x for x in exchange),8)
            self.assertFalse(any("st.shared" in x or "ld.global" in x for x in exchange))
            start=barriers[i+1]+1

    def test_shared_scratch_is_serial_max_not_sum(self):
        self.assertEqual(self.info["route_counts"],dict(Register=1,Shuffle=0,Shared=4))
        self.assertEqual(self.info["scratch_max_bytes"],4096)
        self.assertEqual(self.info["static_shared_bytes"],4096)
        self.assertEqual(self.info["physical_shared_bytes"],4096)
        self.assertEqual(self.info["scratch_offset"],0)
        self.assertEqual(self.info["shared_allocation_plan"]["scratch_reuse"],"one serial segment, max across completed redistributions, not sum")

    def test_large_replicated_v_is_only_demanded_in_eight_vectors(self):
        demand=self.info["demand_lowering"]
        large=[v for v in self.normal["values"] if v.get("distributed",{}).get("topology",[0])[0]>256]
        self.assertEqual(len(large),1)
        value=large[0];self.assertEqual(value["id"],11)
        self.assertEqual(value["distributed"]["topology"][0],1024)
        self.assertEqual(demand["queried_register_vectors"][11],8)
        self.assertEqual(demand["load_vectors"],24)
        self.assertLess(demand["materialized_query_vectors"],demand["declared_register_vectors"])

    def test_global_inputs_read_before_any_global_output(self):
        self.assertLess(self.text.rfind("ld.global.u8"),self.text.find("st.global.u8"))
        demand=self.info["demand_lowering"]
        self.assertTrue(demand["preserves_checked_layouts"])
        self.assertTrue(demand["per_thread_reads_before_store"])
        self.assertTrue(demand["requires_disjoint_output"])
        self.assertFalse(self.info["fp_fusion"])
        self.assertFalse(self.info["reassociation"])

    def test_emission_is_deterministic_and_does_not_mutate_packet(self):
        before=json.dumps(self.packet,sort_keys=True)
        self.assertEqual(backend.emit(self.packet),backend.emit(self.packet))
        self.assertEqual(json.dumps(self.packet,sort_keys=True),before)

    def test_existing_conversion_implementation_is_unchanged(self):
        self.assertEqual(method_ast(Path(backend.__file__).read_text(),"convert"),method_ast(PREIMAGE.read_text(),"convert"))

    def test_large_compact_source_cannot_be_relabelled_cross_thread(self):
        packet=copy.deepcopy(self.packet)
        conversion=next(o for o in packet["operations"] if o["name"]=="frisk.convert_layout" and "source_slots_kind" in o["redistribution"])
        conversion["redistribution"]["mode"]="Shared"
        with self.assertRaisesRegex(ValueError,"compact source Register-only route"):
            backend.emit(packet)

    def test_existing_explicit_cross_thread_domain_still_rejects_over_256(self):
        # Probe the real ordinary-proof validator before any slot expansion.
        # A proposed 512-register Shared boundary cannot enter the hybrid path.
        values={v["id"]:dict(v) for v in self.normal["values"]}
        operation=self.cross[0];source=operation["operands"][0]
        values[source]=dict(values[source],distributed=dict(values[source]["distributed"]))
        values[source]["distributed"]["topology"]=list(values[source]["distributed"]["topology"])
        values[source]["distributed"]["topology"][0]=512
        with self.assertRaisesRegex(ValueError,"bounded domain"):
            validate(operation,values)

    def test_hybrid_guard_refuses_future_large_cross_thread_plan(self):
        # The exporter/validator currently reject it earlier; independently guard
        # against a future validator widening its proof without codegen support.
        emitter=backend.Emitter(self.packet);original=backend.validate_redistribution
        def widened(operation,values):
            plan=original(operation,values)
            return dict(plan,source_registers=512) if plan["mode"]=="Shared" else plan
        with patch.object(backend,"validate_redistribution",widened):
            with self.assertRaisesRegex(ValueError,"large cross-thread demand boundary"):
                emitter.demand_program()

    def test_previous_register_only_paths_keep_identical_ttgir_and_config(self):
        previous=previous_backend()
        for name in ("diamond-32x32-calibrated","remat-128x16-calibrated","mixed-128x32-calibrated","conflict-256x8-calibrated"):
            with self.subTest(name=name):
                packet=json.loads((OLD_RUN/name/"execution.json").read_text())
                actual=backend.emit(packet);before=previous.emit(packet)
                self.assertEqual(actual,before)
                self.assertEqual(actual[1]["demand_lowering"]["version"],"same-thread-demand-slots-v1")
                self.assertNotIn("materialized_cross_thread_boundaries",actual[1]["demand_lowering"])


if __name__=="__main__":
    PACKET=sys.argv[1]
    unittest.main(argv=[sys.argv[0]])
