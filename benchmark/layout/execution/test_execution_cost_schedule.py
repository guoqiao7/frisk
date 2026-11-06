"""Semantic tests for the optional block-local descriptor, never hardware latency."""
import copy
import unittest
import execution_cost_schedule as schedule
import execution_cost_summary as complete
from test_execution_cost_summary import document, function, op


def analyze(doc):
    base = complete.analyze_document(doc)
    result = schedule.analyze_document(doc, base)
    if result["status"] != "complete": raise AssertionError(result)
    return result


class BlockScheduleTest(unittest.TestCase):
    def test_same_work_different_first_use_order_has_distinct_exposure(self):
        load = op("LDG.E", "R0,[R2.64]")
        use = op("FADD", "R3,R0,R1")
        immediate = document([load, use, *[op("NOP") for _ in range(40)], op("EXIT")])
        delayed = document([load, *[op("NOP") for _ in range(40)], use, op("EXIT")])
        self.assertEqual(complete.analyze_document(immediate)["features"], complete.analyze_document(delayed)["features"])
        early, late = analyze(immediate)["features"], analyze(delayed)["features"]
        self.assertGreater(early["block_load_use_exposure_32"], late["block_load_use_exposure_32"])
        self.assertGreater(early["block_load_use_exposure_128"], late["block_load_use_exposure_128"])

    def test_cross_branch_load_use_is_cut_not_fabricated(self):
        d = document([op("LDG.E", "R0,[R2.64]"), op("BRA", "0x30"), op("NOP"), op("FADD", "R3,R0,R1"), op("EXIT")])
        f = analyze(d)["features"]
        self.assertEqual(f["block_count"], 2)
        self.assertEqual(f["block_unconsumed_loads"], 1)
        self.assertEqual(f["block_load_use_exposure_128"], 0)
        self.assertEqual(f["block_dependency_depth_max"], 1)

    def test_callee_stats_are_added_per_call_but_max_is_not_summed(self):
        callee = function("helper", 0x100, [op("LDG.E", "R0,[R2.64]"), op("FADD", "R3,R0,R1"), op("RET.REL.NODEC", "R6,frisk_execution")])
        one = analyze(document([op("CALL.REL.NOINC", "helper"), op("EXIT")], callee))["features"]
        two = analyze(document([op("CALL.REL.NOINC", "helper"), op("CALL.REL.NOINC", "helper"), op("EXIT")], callee))["features"]
        self.assertEqual(two["block_load_use_exposure_32"], 2 * one["block_load_use_exposure_32"])
        self.assertEqual(two["block_dependency_depth_max"], one["block_dependency_depth_max"])
        self.assertEqual(two["block_dependency_depth_sum"], 2 * one["block_dependency_depth_sum"])

    def test_complete_proof_cannot_be_substituted(self):
        d = document([op("MOV", "R0,0x4"), op("EXIT")]); base = complete.analyze_document(d)
        changed = copy.deepcopy(d); changed[1][0]["sass-instructions"][0]["operands"] = "R0,0x8"
        r = schedule.analyze_document(changed, base)
        self.assertEqual(r["status"], "unknown")
        self.assertIn("block-complete-proof-mismatch", r["reasons"])

    def test_immediate_address_change_is_not_schedule_information(self):
        first = document([op("LOP3.LUT", "R2,R2,0x4,RZ,0xc0,!PT"), op("EXIT")])
        second = document([op("LOP3.LUT", "R2,R2,0x10,RZ,0xc0,!PT"), op("EXIT")])
        self.assertEqual(analyze(first)["features"], analyze(second)["features"])

    def test_packed_float_and_predicate_logic_register_edges(self):
        d = document([op("HFMA2.MMA", "R129,-RZ,RZ,0,0"), op("ISETP.NE.AND", "P0,PT,R129,RZ,PT"), op("PLOP3.LUT", "P1,PT,P0,PT,PT,0xa8,0"), op("FADD", "R0,R1,R2", "@P1"), op("EXIT")])
        self.assertGreaterEqual(analyze(d)["features"]["block_raw_dependency_depth_max"], 4)

    def test_unknown_or_other_family_is_not_accepted(self):
        d = document([op("EXIT")]); base = complete.analyze_document(d)
        base["cost_family"] = "native14"
        self.assertEqual(schedule.analyze_document(d, base)["status"], "unknown")


if __name__ == "__main__": unittest.main()
