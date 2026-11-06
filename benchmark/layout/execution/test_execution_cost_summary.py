"""CPU proof failures and semantic invariants for complete cost summaries."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import execution_cost_summary as summary


def op(opcode, operands="", predicate=""):
    return dict(opcode=opcode, operands=operands, predicate=predicate)


def function(name, start, instructions):
    return {"function-name": name, "start": start, "length": 16 * len(instructions), "sass-instructions": instructions}


def document(entry, *others):
    return [{"SM": {"version": {"major": 9, "minor": 0}}, "SchemaVersion": {"major": 12}}, [function("frisk_execution", 0, entry), *others]]


def helper(instructions=None):
    return function("helper", 0x100, instructions or [op("LDG.E.128", "R4,[R2.64]"), op("FMUL", "R0,R1,R2"), op("RET.REL.NODEC", "R6,frisk_execution")])


class CompleteNativeProofTest(unittest.TestCase):
    def complete(self, doc, **kwargs):
        result = summary.analyze_document(doc, **kwargs)
        self.assertEqual(result["status"], "complete", result)
        return result

    def unknown(self, doc, reason=None, **kwargs):
        result = summary.analyze_document(doc, **kwargs)
        self.assertEqual(result["status"], "unknown", result)
        self.assertIsNone(result["features"])
        if reason: self.assertIn(reason, result["reasons"][0])
        return result

    def test_callee_expanded_per_call_dead_ret_padding_excluded(self):
        d = document([op("CALL.REL.NOINC", "helper"), op("CALL.REL.NOINC", "helper"), op("EXIT"), op("BRA", "0x30")], helper())
        r = self.complete(d)
        self.assertEqual(r["features"]["native_instructions"], 9)
        self.assertEqual(r["features"]["callee_instructions"], 6)
        self.assertEqual(r["features"]["global_load_bytes"], 32)
        self.assertEqual(r["features"]["call_sites"], 2)
        self.assertEqual(r["execution_bounds"]["upper"]["native_instructions"], 9)
        self.assertFalse(r["proof"]["native_prefix_truncation"])

    def test_reachable_code_after_first_exit_is_included(self):
        r = self.complete(document([op("BRA", "0x30", "@P0"), op("MOV", "R1,R0"), op("EXIT"), op("LDL.64", "R2,[R1]"), op("EXIT")]))
        self.assertEqual(r["features"]["native_instructions"], 5)
        self.assertEqual(r["features"]["local_transfer_bytes"], 8)

    def test_backward_address_edge_can_still_be_dag(self):
        r = self.complete(document([op("BRA", "0x30", "@P0"), op("EXIT"), op("NOP"), op("BRA", "0x10")]))
        self.assertEqual(r["features"]["native_instructions"], 3)
        self.assertIsNotNone(r["execution_bounds"]["upper"])

    def test_unreachable_unknown_opcode_does_not_poison_closure(self):
        r = self.complete(document([op("EXIT"), op("BRX", "R0")]))
        self.assertEqual(r["features"]["native_instructions"], 1)
        self.assertEqual(r["proof"]["functions"][0]["unreachable_sites"], 1)

    def test_matched_reconvergence(self):
        r = self.complete(document([op("BSSY", "B0,0x40"), op("BRA", "0x30", "@P0"), op("MOV", "R0,R1"), op("BSYNC", "B0"), op("EXIT")]))
        self.assertEqual(r["features"]["control_instructions"], 4)

    def test_unmatched_reconvergence_fails_closed(self):
        self.unknown(document([op("BSSY", "B0,0x20"), op("NOP"), op("EXIT")]), "unmatched-reconvergence")
        self.unknown(document([op("BSYNC", "B0"), op("EXIT")]), "join-without-setup")

    def test_setup_bypass_is_not_a_valid_reconvergence_proof(self):
        self.unknown(document([op("BRA", "0x20", "@P0"), op("BSSY", "B0,0x40"), op("MOV", "R0,R1"), op("BSYNC", "B0"), op("EXIT")]), "setup-bypassed")

    def test_packed_float_is_a_site_not_two_scalar_flops(self):
        r = self.complete(document([op("HFMA2.MMA", "R129,-RZ,RZ,0,0"), op("EXIT")]))
        self.assertEqual(r["features"]["float_arithmetic_instructions"], 1)
        self.assertIn("not FLOPs", r["feature_units"]["float_arithmetic_instructions"])

    def test_missing_and_indirect_targets_fail_closed(self):
        self.unknown(document([op("BRA", "0x500")]), "missing-native-target")
        self.unknown(document([op("BRX", "R1")]), "unknown-native-opcode")
        self.unknown(document([op("CALL.REL.NOINC", "missing"), op("EXIT")]), "nonlocal-or-unknown-call")
        self.unknown(document([op("CALL.ABS.NOINC", "helper"), op("EXIT")], helper()), "nonlocal-or-unknown-call")

    def test_recursive_call_fails_closed(self):
        self.unknown(document([op("CALL.REL.NOINC", "helper"), op("EXIT")], helper([op("CALL.REL.NOINC", "helper"), op("RET.REL.NODEC", "R6,frisk_execution")])), "nested-local-call-abi-unproved")

    def test_return_and_exit_are_not_interchangeable(self):
        self.unknown(document([op("RET.REL.NODEC", "R6,frisk_execution")]), "unknown-return-abi")
        self.unknown(document([op("CALL.REL.NOINC", "helper"), op("EXIT")], helper([op("EXIT")])), "callee-exit")
        self.unknown(document([op("EXIT", predicate="@P0")]), "predicated-terminal")
        self.unknown(document([op("MOV", "R0,R1")]), "missing-native-target")

    def test_unknown_opcode_never_has_zero_price(self):
        self.unknown(document([op("MAGIC"), op("EXIT")]), "unknown-native-opcode")
        self.unknown(document([op("HGMMA.64x16x16.F32"), op("EXIT")]), "unknown-native-opcode")

    def test_only_proven_tma_poll_has_unbounded_dynamic_count(self):
        d = document([op("SYNCS.PHASECHK.TRANS64.TRYWAIT", "P0,[R0],R1"), op("BRA", "0x0", "@!P0"), op("EXIT")])
        r = self.complete(d, allow_attention=True)
        self.assertIsNone(r["execution_bounds"]["upper"])
        self.assertEqual(r["features"]["native_instructions"], 3)
        self.assertEqual(r["proof"]["functions"][0]["polling_sccs"], [[0, 16]])
        self.unknown(d)
        self.unknown(document([op("MOV", "R0,R1"), op("BRA", "0x0", "@P0"), op("EXIT")]), "unproved-native-loop", allow_attention=True)
        self.unknown(document([op("BRA", "0x0")]), "terminal-missing")

    def test_poll_guard_must_use_the_wait_result(self):
        self.unknown(document([op("SYNCS.PHASECHK.TRANS64.TRYWAIT", "P1[UR0],RZ"), op("BRA", "0x0", "@!P0"), op("EXIT")]), "unproved-poll-predicate", allow_attention=True)

    def test_aggregate_pr_not_execution_guard(self):
        r = self.complete(document([op("P2R", "R0,RZ,0xff", "PR"), op("EXIT")]))
        self.assertEqual(r["features"]["predicated_instructions"], 0)

    def test_bad_json_returns_typed_unknown(self):
        for bad in [None, [], [[], []], [{"SM": []}, []], document([None]), document([op(2)]), document([op("EXIT")]) + [{}]]:
            self.unknown(bad)


class AttentionContractTest(unittest.TestCase):
    def setUp(self):
        self.packet = {"operations": [
            {"name": "frisk.mma", "attrs": {"frisk.mma_contract": dict(version=1, target="sm_90a", input_type="f16", accumulator_type="f32", grid=[1, 1], atom=[64, 16, 16], repeats=[1, 1, 1], form="rs", packing="f16x2-low-high")}},
            *[{"name": "frisk.copy", "attrs": {"frisk.copy_contract": dict(version=1, target="sm_90a", guarded=0, valid_shape=[64, 16], path="vector", completion="synchronous", vector_bytes=1)}} for _ in range(3)]]}
        self.ptx = ".visible .entry frisk_execution() {\nmov.b32 t0, {h0,h1};\nwgmma.fence.sync.aligned;\nwgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {a0}, {r0}, desc, p0, 1, 1, 1;\nwgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\nret;\n}\n"
        self.native = {"proof": {"opcode_histogram": {"HGMMA.64x16x16.F32": 1, "WARPGROUP.DEPBAR.LE": 1, "F2FP.F16.F32.PACK_AB": 1}, "functions": [{"polling_sccs": []}]}}

    def test_ss_rs_packing_and_wait_are_separate_features(self):
        f, path = summary.attention_features(self.packet, self.ptx, self.native)
        self.assertEqual(path, "vector")
        self.assertEqual(f["wgmma_wait0_sites"], 1)
        self.assertEqual(f["ptx_f16_pack_sites"], 1)
        self.assertEqual(f["native_f16_pack_sites"], 1)
        self.assertEqual(f["declared_copy_payload_bytes"], 6144)

    def test_delayed_or_nonzero_wait_rejected(self):
        for ptx in [self.ptx.replace("aligned 0;", "aligned 2;"), self.ptx.replace("wgmma.commit_group", "mov.u32 r0, 0;\nwgmma.commit_group"), self.ptx.replace("wgmma.fence.sync.aligned;", "")]:
            with self.assertRaises(ValueError): summary.attention_features(self.packet, ptx, self.native)

    def test_predicated_tensor_sequence_rejected(self):
        ptx = self.ptx.replace("wgmma.mma_async.", "@p0 wgmma.mma_async.")
        with self.assertRaisesRegex(ValueError, "predicated-wgmma"): summary.attention_features(self.packet, ptx, self.native)
        native = copy.deepcopy(self.native); native["proof"]["predicated_opcode_histogram"] = {"HGMMA.64x16x16.F32": 1}
        with self.assertRaisesRegex(ValueError, "predicated-native-tensor"): summary.attention_features(self.packet, self.ptx, native)

    def test_packet_compiler_native_mismatch_rejected(self):
        native = copy.deepcopy(self.native); native["proof"]["opcode_histogram"]["HGMMA.64x16x16.F32"] = 2
        with self.assertRaisesRegex(ValueError, "native-tensor"): summary.attention_features(self.packet, self.ptx, native)
        packet = copy.deepcopy(self.packet); packet["operations"][0]["attrs"]["frisk.mma_contract"]["packing"] = "opaque"
        with self.assertRaisesRegex(ValueError, "packing"): summary.attention_features(packet, self.ptx, self.native)

    def test_unsupported_async_copy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown-copy"): summary.attention_features(self.packet, self.ptx + "cp.async.bulk.shared::cta.global foo;\n", self.native)


class ProvenanceTest(unittest.TestCase):
    def setup_case(self, directory):
        root = Path(directory)
        for name, data in {"kernel.cubin": b"fake-cubin", "execution.json": b'{"operations":[]}', "kernel.ttgir": b"fake-ttgir", "kernel.ptx": b"fake-ptx", "execution_config.json": b"{}"}.items():
            (root / name).write_bytes(data)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        (root / "image-identity.json").write_text(json.dumps({p.name: sha(p) for p in root.iterdir()}))
        (root / "compile.json").write_text(json.dumps({"cubin_sha256": sha(root / "kernel.cubin"), "execution_sha256": sha(root / "execution.json")}))
        tool = root / "tool"; tool.write_text("fake nvdisasm, not an executable")
        return root, tool

    def test_full_lineage_bound_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root, tool = self.setup_case(directory)
            def run(args, **kwargs):
                return "fake version" if args[1] == "--version" else json.dumps(document([op("EXIT")])).encode()
            with mock.patch.object(summary.subprocess, "check_output", side_effect=run):
                first = summary.summarize(root, nvdisasm=tool); second = summary.summarize(root, nvdisasm=tool)
            self.assertEqual(first["status"], "complete", first)
            self.assertEqual(first, second)
            self.assertEqual(set(first["provenance"]["dependencies"]), {"kernel.cubin", "execution.json", "kernel.ttgir", "kernel.ptx", "execution_config.json", "compile.json", "image-identity.json"})

    def test_changed_input_and_wrong_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, tool = self.setup_case(directory)
            (root / "kernel.ptx").write_text("changed")
            self.assertEqual(summary.summarize(root, nvdisasm=tool)["reasons"], ["image-identity-mismatch:kernel.ptx"])
        with tempfile.TemporaryDirectory() as directory:
            root, tool = self.setup_case(directory)
            def run(args, **kwargs):
                if args[1] == "--version": return "fake version"
                (root / "kernel.cubin").write_bytes(b"changed during tool analysis")
                return json.dumps(document([op("EXIT")])).encode()
            with mock.patch.object(summary.subprocess, "check_output", side_effect=run):
                result = summary.summarize(root, nvdisasm=tool)
            self.assertEqual(result["status"], "unknown")
            self.assertIn("input-changed-during-analysis:kernel.cubin", result["reasons"])


if __name__ == "__main__": unittest.main()
