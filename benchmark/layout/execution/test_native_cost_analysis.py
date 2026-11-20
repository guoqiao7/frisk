import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import native_cost_analysis as native

META = {"SM": {"version": {"major": 9, "minor": 0}}, "SchemaVersion": {"major": 12, "minor": 8}}
RESOURCE = "Resource usage:\n Function frisk_execution:\n REG:32 STACK:0 SHARED:1024 LOCAL:0 CONSTANT[0]:568\n"

def document(instructions):
    return [copy.deepcopy(META), [{"function-name": "frisk_execution", "sass-instructions": instructions}]]

def inst(opcode, operands="", predicate=None):
    row = {"opcode": opcode, "operands": operands}
    if predicate: row["predicate"] = predicate
    return row

def run(instructions):
    return native.analyze_disassembly(document(instructions + [inst("EXIT")]), RESOURCE)

class NativeCostAnalysisTest(unittest.TestCase):
    def test_surviving_native_counts_ignore_padding(self):
        data = document([inst("LDG.E.U8", "R0,desc[UR4][R2.64]"), inst("FMUL", "R1,R0,R0"), inst("STG.E.U8", "desc[UR4][R2.64],R1"), inst("EXIT"), inst("BRA", "0x30"), inst("NOP")])
        result = native.analyze_disassembly(data, RESOURCE)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["features"]["native_instructions"], 4)
        self.assertEqual(result["features"]["global_load_bytes"], 1)
        self.assertEqual(result["features"]["global_store_bytes"], 1)
        self.assertEqual(result["diagnostics"]["instructions_after_first_unconditional_exit"], 2)
        self.assertEqual(result["features"]["load_use_exposure_32"], 31)
        self.assertEqual(result["features"]["load_use_exposure_128"], 127)

    def test_wide_address_and_descriptor_pairs(self):
        decoded = native._decode(inst("LDG.E.128", "R8,desc[UR4][R2.64]"))
        self.assertEqual(decoded["writes"], {"R8", "R9", "R10", "R11"})
        self.assertEqual(decoded["reads"], {"UR4", "UR5", "R2", "R3"})
        store = native._decode(inst("STL.64", "[R1+0x10],R12"))
        self.assertEqual(store["reads"], {"R1", "R12", "R13"})

    def test_ldl_missing_comma_native12_8(self):
        decoded = native._decode(inst("LDL.LU.64", "R20[R1+0xfc]"))
        self.assertEqual(decoded["writes"], {"R20", "R21"})
        self.assertEqual(decoded["reads"], {"R1"})
        result = run([inst("STL.64", "[R1+0x10],R12"), inst("LDL.LU.64", "R20[R1+0x10]"), inst("FADD", "R0,R20,R21")])
        self.assertEqual(result["features"]["local_store_bytes"], 8)
        self.assertEqual(result["features"]["local_load_bytes"], 8)

    def test_carry_and_shuffle_multi_output(self):
        carry = native._decode(inst("IADD3", "R2,P0,R0,UR8,RZ"))
        self.assertEqual(carry["writes"], {"R2", "P0"})
        shuffle = native._decode(inst("SHFL.IDX", "P1,R15,R17,RZ,0x1f"))
        self.assertEqual(shuffle["writes"], {"P1", "R15"})
        self.assertEqual(shuffle["reads"], {"R17"})
        fchk = native._decode(inst("FCHK", "P0,R8,R9"))
        self.assertEqual(fchk["writes"], {"P0"})
        self.assertEqual(fchk["reads"], {"R8", "R9"})

    def test_predication_retains_old_dest(self):
        result = native._decode(inst("FMUL", "R1,R2,R3", "@!P0"))
        self.assertEqual(result["reads"], {"P0", "R1", "R2", "R3"})
        report = run([inst("FMUL", "R1,R2,R3", "@!P0")])
        self.assertEqual(report["status"], "available")
        self.assertEqual(report["diagnostics"]["predicated_instructions"], 1)

    def test_shared_barrier_and_positive_features(self):
        result = run([inst("STS.U16", "[R0],R1"), inst("BAR.SYNC.DEFER_BLOCKING", "0x0"), inst("LDS", "R2,[R0]"), inst("SHFL.IDX", "PT,R3,R2,RZ,0x1f")])
        self.assertEqual(result["features"]["shared_store_bytes"], 2)
        self.assertEqual(result["features"]["shared_load_bytes"], 4)
        self.assertEqual(result["features"]["barrier_instructions"], 1)
        self.assertEqual(result["features"]["shuffle_instructions"], 1)
        self.assertEqual(result["features"]["dependency_depth"], 5)

    def test_load_distance_exposure_direction(self):
        close = run([inst("LDG", "R0,[R10.64]"), inst("FMUL", "R1,R0,R0")])
        far = run([inst("LDG", "R0,[R10.64]"), inst("MOV", "R2,0x0"), inst("MOV", "R3,0x0"), inst("FMUL", "R1,R0,R0")])
        self.assertLess(far["features"]["load_use_exposure_32"], close["features"]["load_use_exposure_32"])
        self.assertEqual(far["diagnostics"]["load_first_use_distances"], [3])

    def test_dead_native_load_diagnostic_not_exposure(self):
        result = run([inst("LDG", "R0,[R10.64]"), inst("MOV", "R0,0x0")])
        self.assertEqual(result["features"]["global_load_bytes"], 4)
        self.assertEqual(result["features"]["load_use_exposure_32"], 0)
        self.assertEqual(result["diagnostics"]["loads_without_register_use"], 1)

    def test_read_before_overwrite_and_alias_dependencies(self):
        result = run([inst("MOV", "R1,0x0"), inst("FMUL", "R2,R1,R1"), inst("MOV", "R1,0x1")])
        # unit DAG has MOV->FMUL->overwrite (WAR), then EXIT fence
        self.assertEqual(result["features"]["dependency_depth"], 4)
        self.assertEqual(result["diagnostics"]["raw_dependency_depth"], 2)

    def test_aggregate_predicate_moves_and_nvdisasm_pr_field(self):
        predicates={"P"+str(i) for i in range(7)}
        r2p=native._decode(inst("R2P", "PR,R2,0x1f"))
        self.assertEqual(r2p["writes"],predicates)
        self.assertEqual(r2p["reads"],predicates|{"R2"})
        for row in [inst("P2R","R44,RZ,0x40","PR"),inst("P2R","R44,PR,RZ,0x40")]:
            p2r=native._decode(row)
            self.assertEqual(p2r["reads"],predicates)
            self.assertEqual(p2r["writes"],{"R44"})
            self.assertEqual(run([row])["diagnostics"]["predicated_instructions"],0)
        guarded=native._decode(inst("P2R","R4,PR,R2,0xff","@!P0"))
        self.assertEqual(guarded["reads"],predicates|{"R2","R4"})
    def test_aggregate_predicate_edges_are_not_omitted(self):
        # load -> predicate definition -> aggregate transfer -> use is retained.
        result=run([inst("LDG","R2,[R10.64]"),inst("ISETP.NE.AND","P0,PT,R2,RZ,PT"),inst("P2R","R4,RZ,0x1","PR"),inst("FMUL","R5,R4,R4")])
        self.assertEqual(result["status"],"available")
        self.assertGreaterEqual(result["diagnostics"]["raw_dependency_depth"],4)
        self.assertEqual(result["features"]["integer_logic_instructions"],2)
    def test_unproved_predicate_move_forms_fail_closed(self):
        for row in [inst("R2P.B1","PR,R2,0x1f"),inst("R2P","PR,R2.B1,0x1f"),inst("R2P","UPR,UR2,0x1f"),inst("R2P","PR,R2,0x100"),inst("P2R","R4,PR,RZ,-1"),inst("P2R","R4,RZ,0x40"),inst("P2R.B1","R4,PR,RZ,0xff"),inst("P2R","R4,RZ,0xff","PRX")]:
            with self.subTest(row=row):
                self.assertEqual(run([row])["status"],"unsupported")
    def test_control_flow_and_unknown_fail_closed(self):
        for row in [inst("BRA", "0x40", "@!P0"), inst("CALL.REL.NOINC", "0x80"), inst("HGMMA.64x64x16.F32", "R0,R1,R2"), inst("NEW_HARDWARE_OP", "R0,R1")]:
            result = run([row])
            self.assertEqual(result["status"], "unsupported")
            self.assertIsNone(result["features"])
            self.assertTrue(result["support"]["reasons"])

    def test_predicated_exit_and_missing_exit_fail_closed(self):
        for rows in [[inst("EXIT", predicate="@P0")], [inst("MOV", "R0,0x0")]]:
            result = native.analyze_disassembly(document(rows), RESOURCE)
            self.assertEqual(result["status"], "unsupported")
            self.assertIsNone(result["features"])

    def test_bad_schema_architecture_resources(self):
        data = document([inst("EXIT")]); data[0]["SM"]["version"]["major"] = 8
        self.assertEqual(native.analyze_disassembly(data, RESOURCE)["status"], "unsupported")
        self.assertEqual(native.analyze_disassembly(document([inst("EXIT")]), "")["status"], "unsupported")
        self.assertEqual(native.analyze_disassembly({}, RESOURCE)["status"], "unsupported")

    def test_analysis_work_limits_are_typed(self):
        with mock.patch.object(native, "MAX_INSTRUCTIONS", 1):
            result = run([inst("MOV", "R0,0x0")])
            self.assertEqual(result["status"], "unsupported")
            self.assertIn("native-instruction-budget", result["support"]["reasons"])
        with mock.patch.object(native, "MAX_EDGES", 0):
            result = run([inst("MOV", "R0,0x0"), inst("FMUL", "R1,R0,R0")])
            self.assertEqual(result["status"], "unsupported")
            self.assertIn("dependency-edge-budget", result["support"]["reasons"])

    def test_malformed_native_fields_fail_closed(self):
        for data in [[META, [None]], [META, [{"function-name": "frisk_execution", "sass-instructions": "bad"}]], [META, [{"function-name": "frisk_execution", "sass-instructions": [{"opcode": 1}]}]]]:
            result = native.analyze_disassembly(data, RESOURCE)
            self.assertEqual(result["status"], "unsupported")
            self.assertIsNone(result["features"])

    def test_entry_selection_is_exact(self):
        data = document([inst("EXIT")]); data[1].append({"function-name": "other", "sass-instructions": [inst("NEW_HARDWARE_OP")]})
        self.assertEqual(native.analyze_disassembly(data, RESOURCE)["status"], "available")
        data[1].append(copy.deepcopy(data[1][0]))
        self.assertEqual(native.analyze_disassembly(data, RESOURCE)["status"], "unsupported")

    def test_cache_payload_tamper_invalidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); image = root / "kernel.cubin"; image.write_bytes(b"image")
            tool = root / "tool"; tool.write_bytes(b"tool")
            identity = {"path": str(tool), "sha256": native._sha(b"tool"), "version": "test"}
            def invoke(command, **kwargs):
                return json.dumps(document([inst("EXIT")])) if "-json" in command else RESOURCE
            with mock.patch.object(native, "_tool", return_value=identity), mock.patch.object(native.subprocess, "check_output", side_effect=invoke) as calls:
                first = native.analyze(root, root / "cache")
                second = native.analyze(root, root / "cache")
                self.assertFalse(first["cache_hit"]); self.assertTrue(second["cache_hit"])
                self.assertEqual(calls.call_count, 2)
                path = next((root / "cache").glob("*.json")); corrupted = json.loads(path.read_text())
                corrupted["payload"]["features"]["native_instructions"] = 0
                path.write_text(json.dumps(corrupted))
                third = native.analyze(root, root / "cache")
                self.assertFalse(third["cache_hit"]); self.assertEqual(third["features"]["native_instructions"], 1)
                self.assertEqual(calls.call_count, 4)
                image.write_bytes(b"different")
                fourth = native.analyze(root, root / "cache")
                self.assertNotEqual(third["cache_key"], fourth["cache_key"])

if __name__ == "__main__": unittest.main()
