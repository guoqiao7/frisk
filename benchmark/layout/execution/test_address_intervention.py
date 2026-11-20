"""Independent source-DU and native-static invariants, not dynamic equality."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import address_intervention as proof
from test_execution_cost_summary import document, function, op


def source(mask=4,base=0,data_constant=1,suffix=0,predicate=False):
    pred="@p " if predicate else ""
    return f""".version 8.7
.target sm_90a
.address_size 64
.visible .entry frisk_execution(.param .u64 arg0,.param .u64 arg1)
.reqntid 128
{{
.reg .b64 %rd<3>;
.reg .b32 tid, a, bit, value, result;
.reg .b64 ptr;
.reg .pred p;
ld.param.u64 %rd1, [arg0];
ld.param.u64 %rd2, [arg1];
mov.u32 tid, %tid.x;
mov.u32 a, {base};
bfe.u32 bit, tid, 0, 1;
neg.s32 bit, bit;
and.b32 bit, bit, {mask};
xor.b32 a, a, bit;
cvt.u64.u32 ptr, a;
add.u64 ptr, %rd1, ptr;
ld.global.u8 value, [ptr+{suffix}];
add.u32 result, value, {data_constant};
setp.eq.u32 p, tid, 0;
{pred}st.global.u8 [%rd2+0], result;
ret;
}}
.file 1 "unrelated-debug-name"
"""


class SourceInterventionTest(unittest.TestCase):
    def available(self,ptx):
        result=proof.source_signature(ptx);self.assertEqual(result["status"],"available",result);return result

    def test_only_global_address_constants_may_vary(self):
        a=self.available(source(4));b=self.available(source(16,128))
        self.assertEqual(a["nonaddress_sha256"],b["nonaddress_sha256"])
        self.assertEqual(a["preserved_nonaddress_instructions"],b["preserved_nonaddress_instructions"])
        self.assertNotEqual(a["normalized_source_sha256"],b["normalized_source_sha256"])
        self.assertGreater(a["removed_address_instructions"],0)

    def test_virtual_temporary_alpha_names_and_debug_metadata_do_not_matter(self):
        a=source();b=a.replace("value","payload42").replace("result","out17").replace("unrelated-debug-name","different-debug-name")
        self.assertEqual(self.available(a)["nonaddress_sha256"],self.available(b)["nonaddress_sha256"])

    def test_data_math_predicate_and_byte_terminal_are_preserved(self):
        original=self.available(source())
        mutations=[source(data_constant=2),source(suffix=1),source(predicate=True),source().replace("setp.eq.u32 p, tid, 0","setp.eq.u32 p, tid, 1"),source().replace("add.u32 result, value, 1","sub.u32 result, value, 1")]
        for ptx in mutations:
            value=self.available(ptx)
            self.assertNotEqual(value["nonaddress_sha256"],original["nonaddress_sha256"])

    def test_address_value_used_as_data_is_not_removed(self):
        a=source(4).replace("add.u32 result, value, 1","add.u32 result, a, 1")
        b=source(16).replace("add.u32 result, value, 1","add.u32 result, a, 1")
        self.assertNotEqual(self.available(a)["nonaddress_sha256"],self.available(b)["nonaddress_sha256"])

    def test_address_value_used_by_predicate_is_not_removed(self):
        a=source(4).replace("setp.eq.u32 p, tid, 0","setp.eq.u32 p, a, 0")
        b=source(16).replace("setp.eq.u32 p, tid, 0","setp.eq.u32 p, a, 0")
        self.assertNotEqual(self.available(a)["nonaddress_sha256"],self.available(b)["nonaddress_sha256"])

    def test_shared_or_barrier_changes_are_not_global_interventions(self):
        a=source().replace("ret;","bar.sync 0;\nret;")
        b=source().replace("ret;","bar.sync 1;\nret;")
        self.assertNotEqual(self.available(a)["nonaddress_sha256"],self.available(b)["nonaddress_sha256"])
        a=source(4).replace("ld.global.u8 value","ld.shared.u8 value")
        b=source(16).replace("ld.global.u8 value","ld.shared.u8 value")
        self.assertNotEqual(self.available(a)["nonaddress_sha256"],self.available(b)["nonaddress_sha256"])

    def test_unknown_specials_symbols_parameters_and_pure_arity_are_rejected(self):
        for name in ("%clock", "%clock64", "%globaltimer", "%laneid", "%ctaid.x"):
            result=proof.source_signature(source().replace("%tid.x",name))
            self.assertEqual(result["status"],"unsupported",result)
            self.assertIn("unknown external",result["reasons"][0])
        for text in [source().replace("mov.u32 a, 0;","mov.u32 a, unknown_symbol;"),source().replace("[arg0]","[undeclared_argument]"),source().replace("and.b32 bit, bit, 4;","and.b32 bit, bit, 4, 0;"),source().replace("neg.s32 bit, bit;","neg.s32 bit;")]:
            self.assertEqual(proof.source_signature(text)["status"],"unsupported")

    def test_declared_shared_symbol_is_never_a_global_address_root(self):
        text=source().replace(".reg .pred p;",".reg .pred p;\n.shared .align 4 .b8 scratch[128];").replace("mov.u32 a, 0;","mov.u32 a, scratch;")
        result=proof.source_signature(text)
        self.assertEqual(result["status"],"unsupported",result)
        self.assertIn("address-root-unknown-external-symbol",result["reasons"][0])

    def test_data_dependent_address_or_source_branches_are_unsupported(self):
        ptx=source().replace("ret;","cvt.u64.u32 ptr, value;\nld.global.u8 value, [ptr+0];\nret;")
        self.assertEqual(proof.source_signature(ptx)["status"],"unsupported")
        for text in [source().replace("ret;","bra label;\nret;"),source().replace("ret;","call foo;\nret;"),source().replace("mov.u32 tid, %tid.x;","")]:
            self.assertEqual(proof.source_signature(text)["status"],"unsupported")


class NativeStaticTest(unittest.TestCase):
    def native(self,mask="0x4"):
        return document([op("LOP3.LUT",f"R2,R2,{mask},RZ,0xc0,!PT"),op("LDG.E.U8","R0,[R2.64]"),op("EXIT")])

    def test_only_entry_mask_candidate_is_normalized(self):
        a=proof.native_signature(self.native("0x4"));b=proof.native_signature(self.native("0x10"))
        self.assertEqual(a["status"],"available");self.assertEqual(a["skeleton_sha256"],b["skeleton_sha256"])
        a=proof.native_signature(self.native("0x7fffff"));b=proof.native_signature(self.native("0xffffff"))
        self.assertNotEqual(a["skeleton_sha256"],b["skeleton_sha256"])

    def test_physical_register_opcode_truth_and_predicate_changes_are_locked(self):
        original=proof.native_signature(self.native())["skeleton_sha256"]
        for mutate in [lambda d:d[1][0]["sass-instructions"][1].update(opcode="LDG.E"),lambda d:d[1][0]["sass-instructions"][0].update(operands="R3,R2,0x4,RZ,0xc0,!PT"),lambda d:d[1][0]["sass-instructions"][0].update(operands="R2,R2,0x4,RZ,0x78,!PT"),lambda d:d[1][0]["sass-instructions"][1].update(predicate="@P0")]:
            d=self.native();mutate(d);result=proof.native_signature(d);self.assertEqual(result["status"],"available");self.assertNotEqual(result["skeleton_sha256"],original)

    def test_entire_callee_is_locked_even_for_mask_constants(self):
        def doc(mask):
            return document([op("CALL.REL.NOINC","helper"),op("EXIT")],function("helper",0x100,[op("LOP3.LUT",f"R2,R2,{mask},RZ,0xc0,!PT"),op("RET.REL.NODEC","R6,frisk_execution")]))
        a=proof.native_signature(doc("0x4"));b=proof.native_signature(doc("0x10"))
        self.assertEqual(a["status"],"available");self.assertNotEqual(a["skeleton_sha256"],b["skeleton_sha256"])
        self.assertNotEqual(a["full_callee_sha256"],b["full_callee_sha256"])


class CompiledBindingTest(unittest.TestCase):
    def fixture(self,directory):
        from test_access_reuse import ReceiptTest
        root=ReceiptTest().case(directory)
        (root/'kernel.ptx').write_text(source())
        identity=json.loads((root/'image-identity.json').read_text());identity['kernel.ptx']=hashlib.sha256((root/'kernel.ptx').read_bytes()).hexdigest();(root/'image-identity.json').write_text(json.dumps(identity))
        cubin=hashlib.sha256((root/'kernel.cubin').read_bytes()).hexdigest()
        correct=dict(cubin_sha256=cubin,image_identity_sha256=hashlib.sha256((root/'image-identity.json').read_bytes()).hexdigest(),no_kernel_timing=True,physical_resources=dict(function_attributes=dict(registers_per_thread=16,local_bytes_per_thread=0,shared_bytes=0),threads_per_block=128,dynamic_shared_bytes=0,device_limits=dict(warp_size=32)))
        (root/'correctness.json').write_text(json.dumps(correct))
        raw=json.dumps(document([op('LOP3.LUT','R2,R2,0x4,RZ,0xc0,!PT'),op('LDG.E.U8','R0,[R2.64]'),op('EXIT')])).encode()
        old=dict(identity=dict(cubin_sha256=cubin),native_json_sha256=hashlib.sha256(raw).hexdigest())
        (root/'native-analysis.json').write_text(json.dumps(old))
        locality=proof.access_reuse.summarize(root)
        self.assertEqual(locality['status'],'complete',locality)
        return root,locality,raw

    def test_raw_native_and_accepted_image_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root,locality,raw=self.fixture(directory)
            result=proof.summarize(root,locality,native_raw=raw)
            self.assertEqual(result['status'],'available',result)
            self.assertFalse(result['claims']['dynamic_lane_masks_or_slowpath_counts_equal'])
            changed=proof.summarize(root,locality,native_raw=raw+b' ')
            self.assertEqual(changed['status'],'unsupported');self.assertIn('native-document-old-binding',changed['reasons'])

    def test_different_cubin_native_analysis_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root,locality,raw=self.fixture(directory)
            old=json.loads((root/'native-analysis.json').read_text());old['identity']['cubin_sha256']='0'*64;(root/'native-analysis.json').write_text(json.dumps(old))
            result=proof.summarize(root,locality,native_raw=raw)
            self.assertEqual(result['status'],'unsupported');self.assertIn('native-analysis-cubin-binding',result['reasons'])


if __name__=="__main__":unittest.main()

