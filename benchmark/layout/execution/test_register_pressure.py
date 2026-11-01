import unittest
from register_pressure import analyze_ptx_pressure


def ptx(body):
    return ".version 8.7\n.target sm_90a\n.address_size 64\n.visible .entry frisk_execution(.param .u64 ptr) {\n"+body+"\n}"


class RegisterPressureTest(unittest.TestCase):
    def analyze(self,body):
        row=analyze_ptx_pressure(ptx(body))
        self.assertEqual(row["status"],"available",row)
        return row["features"]
    def test_dead_typed_values_are_not_allocation(self):
        row=self.analyze(".reg .b32 r<100>;\n.reg .b64 addr;\nld.param.u64 addr,[ptr];\nmov.u32 r0,1;\nmov.u32 r1,2;\nst.global.u32 [addr],r0;\nret;")
        self.assertEqual(row["virtual_declared_register32"],102)
        self.assertEqual(row["virtual_live_register32_peak"],3)
        self.assertGreaterEqual(row["syntactic_dce_instruction_count"],1)
    def test_64bit_addresses_count_two_words(self):
        row=self.analyze(".reg .b64 a,b;\nld.param.u64 a,[ptr];\nadd.u64 b,a,8;\nst.global.u64 [a],b;\nret;")
        self.assertEqual(row["virtual_live_register32_peak"],4)
        self.assertEqual(row["virtual_live_u64_register32_peak"],4)
    def test_predicated_write_preserves_old_value(self):
        row=self.analyze(".reg .b32 r;\n.reg .b64 a;\n.reg .pred p;\nld.param.u64 a,[ptr];\nmov.u32 r,7;\nsetp.eq.u32 p,%tid.x,0;\n@p mov.u32 r,3;\nst.global.u32 [a],r;\nret;")
        self.assertEqual(row["syntactic_dce_instruction_count"],0)
        self.assertEqual(row["virtual_live_predicate_peak"],1)
    def test_direct_cfg_loop_reaches_fixed_point(self):
        row=self.analyze(".reg .b64 a;\n.reg .b32 r;\n.reg .pred p;\nld.param.u64 a,[ptr];\nmov.u32 r,0;\nloop:\nadd.u32 r,r,1;\nsetp.lt.u32 p,r,8;\n@p bra loop;\nst.global.u32 [a],r;\nret;")
        self.assertEqual(row["cfg_edges"],3)
        self.assertGreater(row["virtual_live_register32_peak"],0)
    def test_unreachable_heavy_branch_does_not_count_peak(self):
        row=self.analyze(".reg .b64 a,b;\n.reg .b32 r;\nbra done;\nld.param.u64 a,[ptr];\nadd.u64 b,a,8;\nst.global.u64 [a],b;\ndone:\nret;")
        self.assertEqual(row["virtual_live_register32_peak"],0)
        self.assertLess(row["reachable_basic_blocks"],row["basic_blocks"])
    def test_vector_and_mma_inout_tuple(self):
        row=self.analyze(".reg .b64 a;\n.reg .f32 x,y;\nld.param.u64 a,[ptr];\nmov.f32 x,0f00000000;\nmov.f32 y,0f00000000;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {x,y},a,a,1,1,1,0,0;\nst.global.v2.f32 [a],{x,y};\nret;")
        self.assertGreaterEqual(row["virtual_live_register32_peak"],4)
        self.assertEqual(row["syntactic_dce_instruction_count"],0)
    def test_unknown_opcode_and_ambiguous_or_undeclared_symbols_fail_closed(self):
        for code in (".reg .b32 r;\nmadeup.u32 r,0;\nret;", ".reg .b32 r;\n.reg .f32 r;\nret;",
                     ".reg .b32 r;\nadd.u32 r,r,%missing;\nret;", ".reg .b32 r;\nadd.u32 r,r,missing;\nret;", "bra missing;\nret;", "call f;\nret;"):
            with self.subTest(code=code):
                row=analyze_ptx_pressure(ptx(code));self.assertEqual(row["status"],"unavailable");self.assertIsNone(row["features"])
    def test_analysis_budget_failure_is_not_zero_pressure(self):
        row=analyze_ptx_pressure(ptx(".reg .b32 r<10>;\nret;"),max_registers=2)
        self.assertEqual(row["status"],"unavailable");self.assertIsNone(row["physical_registers_per_thread"])
    def test_huge_declaration_rejected_before_expansion(self):
        row=analyze_ptx_pressure(ptx(".reg .b32 r<999999999999999>;\nret;"))
        self.assertEqual(row["status"],"unavailable")
    def test_implicit_carry_and_reduction_barrier_fail_closed(self):
        for body in (".reg .b32 r;\nadd.cc.u32 r,r,1;\nret;", ".reg .b32 r;\naddc.u32 r,r,1;\nret;", ".reg .b32 r;\nbar.red.popc.u32 r,0;\nret;"):
            with self.subTest(body=body):self.assertEqual(analyze_ptx_pressure(ptx(body))["status"],"unavailable")
    def test_address_dependency_includes_pointer_base(self):
        row=self.analyze(".reg .b64 base,address;\n.reg .b32 r;\nld.param.u64 base,[ptr];\nadd.u64 address,base,8;\nld.global.u32 r,[address];\nst.global.u32 [base],r;\nret;")
        self.assertGreaterEqual(row["virtual_live_address_register32_peak"],4)
    def test_no_physical_register_claim_or_external_resource_input(self):
        row=analyze_ptx_pressure(ptx("ret;"))
        self.assertIsNone(row["physical_registers_per_thread"])
        self.assertIn("not physical",row["meaning"])

    def test_long_reverse_address_chain_is_linear_in_dependency_visits(self):
        count=1000
        code=[f".reg .b64 a<{count}>;", "ld.param.u64 a0,[ptr];"]
        code += [f"add.u64 a{i},a{i-1},8;" for i in range(1,count)]
        code += [f"st.global.u32 [a{count-1}],7;", "ret;"]
        result=analyze_ptx_pressure(ptx("\n".join(code)))
        self.assertEqual(result["status"],"available",result)
        work=result["analysis_work"]
        self.assertEqual(work["address_dependency_node_visits"],count)
        self.assertEqual(work["address_dependency_edge_visits"],count-1)
        self.assertLessEqual(work["liveness_instruction_visits"],4*(count+2))
        self.assertLessEqual(work["total"],result["analysis_limits"]["max_analysis_work"])
    def test_dependency_and_global_work_caps_return_unavailable(self):
        body=".reg .b64 a,b,c;\nld.param.u64 a,[ptr];\nadd.u64 b,a,8;\nadd.u64 c,b,8;\nst.global.u32 [c],7;\nret;"
        edge=analyze_ptx_pressure(ptx(body),max_dependency_edges=1)
        self.assertEqual(edge["status"],"unavailable");self.assertIn("edge budget",edge["reason"])
        loop=".reg .b64 a;\n.reg .b32 r;\n.reg .pred p;\nld.param.u64 a,[ptr];\nmov.u32 r,0;\nloop:\nadd.u32 r,r,1;\nsetp.lt.u32 p,r,8;\n@p bra loop;\nst.global.u32 [a],r;\nret;"
        budget=analyze_ptx_pressure(ptx(loop),max_analysis_work=32)
        self.assertEqual(budget["status"],"unavailable");self.assertIn("total analysis work budget",budget["reason"])
        self.assertIn("liveness",budget["reason"])
        self.assertIsNone(budget["features"])
        self.assertLessEqual(budget["analysis_work"]["total"],32)
    def test_invalid_work_budgets_are_rejected(self):
        for options in ({"max_dependency_edges":0},{"max_analysis_work":True}):
            with self.assertRaises(ValueError):analyze_ptx_pressure(ptx("ret;"),**options)


if __name__=="__main__":unittest.main()
