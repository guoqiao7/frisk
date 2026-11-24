import unittest

from compiled_global_access import global_access, physical_choice, UnsupportedGlobalAccess


class GlobalAccessTest(unittest.TestCase):
    def test_width_and_closed_phase(self):
        listing = """Function : norm
/*0000*/ LDG.E.128 R4, desc[UR4][R2.64];
/*0010*/ LDL R2,[R1];
/*0020*/ BAR.SYNC 0x0;
/*0030*/ LDL.64 R2,[R1];
/*0040*/ BAR.SYNC 0x0;
/*0050*/ STG.E.64 desc[UR4][R2.64],R4;
/*0060*/ EXIT;"""
        a = global_access(listing,"norm")
        self.assertEqual((a.loads,a.stores,a.interior_reloads),((0,0,1),(0,1,0),1))
        with self.assertRaises(UnsupportedGlobalAccess):
            global_access(listing.replace("LDG.E.128","@P0 LDG.E.128"),"norm")
        with self.assertRaises(UnsupportedGlobalAccess):
            global_access(listing.replace("LDL R2,[R1];","BRA 0x0060;"),"norm")

    def test_immediate_local_raw_requires_register_and_wait_in_closed_phase(self):
        from compiled_global_access import exposed_reload_uses
        def body(consumer="FMUL R0, R5, R2", mask=4, gap=False, io=False, closed=True):
            items = [("BAR.SYNC 0x0", 0), ("LDL.64 R4,[R1]", 2 << 46)]
            if gap:
                items.append(("NOP", 0))
            items.append((consumer, mask << 52))
            if io:
                items.append(("LDG.E R3,[R2]", 0))
            if closed:
                items.append(("BAR.SYNC 0x0", 0))
            items.append(("EXIT", 0))
            return "\n".join(f"/*{16*i:04x}*/ {op}; /* 0x{'0'*16} */ /* 0x{hi:016x} */"
                             for i,(op,hi) in enumerate(items))
        self.assertEqual(exposed_reload_uses(body()), 1)
        for text in (body(mask=0), body(consumer="FMUL R5, R6, R7"), body(gap=True),
                     body(io=True), body(closed=False), body(consumer="@P0 FMUL R0, R5, R5")):
            self.assertEqual(exposed_reload_uses(text), 0)
        self.assertIsNone(exposed_reload_uses(body().replace("/*0010*/","/*0050*/")))
        self.assertIsNone(exposed_reload_uses(body().replace("FMUL R0, R5, R2","BRA 0x0040")))

    def test_uniform_fp32_issues_require_a_complete_executed_path(self):
        def disasm(arithmetic):
            ops = ["LDG.E R1,[R2]", "SHFL.BFLY PT,R2,R1,1,31", *arithmetic,
                   "STG.E [R2],R1", "EXIT", "FADD R1,R2,R3"]
            word = " /* 0x" + "0"*16 + " */"
            return "Function : norm\n" + "\n".join(
                f"/*{16*i:04x}*/ {op};{word}{word}" for i,op in enumerate(ops))

        listing = disasm(["FFMA R1,R1,R2,R3", "MUFU.RSQ R2,R1"])
        self.assertEqual(global_access(listing, "norm").fp32_issues, 2)
        self.assertEqual(global_access(disasm([]), "norm").fp32_issues, 0)
        predicated = global_access(disasm(["@P0 FMUL R1,R1,R2"]), "norm")
        self.assertIsNone(predicated.fp32_issues)
        self.assertEqual(predicated.exposed_reloads, 0)
        self.assertIsNone(global_access(disasm(["MUFU.RCP64H R1,R1"]), "norm").fp32_issues)
        self.assertIsNone(global_access(listing.replace("/*0020*/", "/*0028*/"), "norm").fp32_issues)

    def test_fp32_cost_is_compared_only_if_complete(self):
        base = dict(local_words=1,ldg=[1,0,0],interior_reloads=0,conversions=0)
        records = {"incumbent":dict(base,physical_total=20,dependency_total=20,compute_total=24),
                   "wide":dict(base,physical_total=19,dependency_total=19,compute_total=31)}
        self.assertEqual(physical_choice(records,"incumbent"), "incumbent")
        del records["wide"]["compute_total"]
        self.assertEqual(physical_choice(records,"incumbent"), "wide")

    def test_partial_dependency_evidence_keeps_previous_vector_comparison(self):
        base = dict(local_words=1,ldg=[1,0,0],interior_reloads=0,conversions=0)
        records = {"incumbent":dict(base,physical_total=20,dependency_total=20),
                   "wide":dict(base,physical_total=10,dependency_total=42)}
        self.assertEqual(physical_choice(records,"incumbent"),"incumbent")
        del records["wide"]["dependency_total"]
        self.assertEqual(physical_choice(records,"incumbent"),"wide")

    def test_serial_reload_guard(self):
        def record(cost, loads, middle):
            return dict(local_words=1,physical_total=cost,ldg=[loads,0,0],
                        interior_reloads=middle,conversions=0)
        r={"incumbent":record(100,10,2),"wide":record(80,6,7)}
        self.assertEqual(physical_choice(r,"incumbent"),"incumbent")
        r["wide"]["interior_reloads"]=6
        self.assertEqual(physical_choice(r,"incumbent"),"wide")
        del r["wide"]["physical_total"]
        self.assertEqual(physical_choice(r,"incumbent"),"incumbent")


if __name__ == "__main__":
    unittest.main()
