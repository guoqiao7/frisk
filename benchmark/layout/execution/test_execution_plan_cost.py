import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from access_plan import plan_access
from execution_plan_cost import access_sectors32, compare_features, extract_features, pareto_frontier, static_key
from resource_prediction import parse_compiler_resource_usage


def ttgir(*lines):
    asm = r"\0A".join(("{", *lines, "}"))
    return (
        'module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {\n'
        '  tt.func public @frisk_execution() {\n'
        f'    %done = tt.elementwise_inline_asm "{asm}" {{pure = false}}\n'
        '    tt.return\n  }\n}\n'
    )


def trace(op, *, shuffles=0, barriers=0, wgmma=0, waits=0):
    return dict(operation=op, shuffles=shuffles, barriers=barriers, wgmma=wgmma, waits=waits)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class ExecutionPlanCostTest(unittest.TestCase):
    def test_traffic_widths_and_exact_trace_split(self):
        text = ttgir(
            "ld.global.v4.b32 {a,b,c,d}, [ptr];", "ld.global.u8 a, [ptr];",
            "@pred st.global.v2.f32 [ptr], {a,b};", "st.shared.v4.b32 [smem], {a,b,c,d};",
            "shfl.sync.bfly.b32 a, b, 1, 31, -1;", "shfl.sync.idx.b32 a, b, 1, 31, -1;",
            "bar.sync 0;", "fence.proxy.async.shared::cta;",
        )
        metadata = dict(execution_trace=[trace("frisk.reduce_tensor", shuffles=1),
                                         trace("frisk.convert_layout", shuffles=1, barriers=1)])
        row = extract_features(text, metadata)["static"]
        self.assertEqual(row["global_load_by_width_bytes"], {"1": 1, "16": 1})
        self.assertEqual(row["global_store_by_width_bytes"], {"8": 1})
        self.assertEqual(row["shared_store_by_width_bytes"], {"16": 1})
        self.assertEqual((row["reduction_shuffles"], row["conversion_shuffles"], row["other_shuffles"]), (1, 1, 0))
        self.assertEqual((row["cta_barriers"], row["local_cta_barriers"], row["async_proxy_fences"]), (1, 0, 1))

    def test_partial_trace_preserves_unknown(self):
        text = ttgir("shfl.sync.idx.b32 a, b, 1, 31, -1;")
        for metadata in ({}, dict(execution_trace=[trace("frisk.reduce_tensor")])):
            row = extract_features(text, metadata)["static"]
            self.assertEqual(row["shuffle_sites"], 1)
            self.assertIsNone(row["reduction_shuffles"])
            self.assertIsNone(row["conversion_shuffles"])
        with self.assertRaisesRegex(ValueError, "trace exceeds"):
            extract_features(text, dict(execution_trace=[trace("frisk.reduce_tensor", shuffles=2)]))

    def test_wgmma_lifecycle_counts_issued_slices(self):
        mma = "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"
        instructions = (
            "wgmma.fence.sync.aligned;", f"{mma} {{a,b}}, sa, sb, acc, 1, 1, 0, 0;",
            f"{mma} {{a,b}}, {{a,b,c,d}}, sb, acc, 1, 1, 1;",
            "wgmma.commit_group.sync.aligned;", "wgmma.wait_group.sync.aligned 0;",
        )
        row = extract_features(ttgir(*instructions), {})["wgmma"]
        self.assertEqual(row["by_form_and_opcode"], {"ss:" + mma: 1, "rs:" + mma: 1})
        self.assertEqual(row["issues_per_commit"], [2])
        self.assertEqual(row["peak_outstanding_group_upper"], 1)
        self.assertEqual(row["drain_waits"], 1)
        unbalanced = extract_features(ttgir(*instructions[:-1]), {})["wgmma"]
        self.assertEqual(unbalanced["lifecycle"], "unknown")
        self.assertIsNone(unbalanced["peak_outstanding_group_upper"])

    def test_shared_live_peak_is_separate_from_allocated_peak(self):
        text = ttgir(".shared .align 16 .b8 frisk_shared[512];")
        metadata = dict(static_shared_bytes=512, physical_shared_bytes=512, shared_allocation_plan={
            "physical_shared_bytes": 512, "allocations": [
                {"offset": 0, "bytes": 256, "lifetime": [0, 2]},
                {"offset": 0, "bytes": 256, "lifetime": [3, 5]},
            ],
        })
        row = extract_features(text, metadata)["static"]
        self.assertEqual(row["shared_allocated_peak_bytes"], 512)
        self.assertEqual(row["shared_live_root_peak_bytes"], 256)
        invalid = copy.deepcopy(metadata)
        invalid["shared_allocation_plan"]["allocations"][1]["lifetime"][0] = 2
        with self.assertRaisesRegex(ValueError, "live Shared roots overlap"):
            extract_features(text, invalid)
        metadata["shared_allocation_plan"]["allocations"][0]["lifetime"] = "whole-kernel-declared"
        self.assertIsNone(extract_features(text, metadata)["static"]["shared_live_root_peak_bytes"])
        with self.assertRaisesRegex(ValueError, "Shared metadata"):
            extract_features(text, dict(physical_shared_bytes=513))

    def test_sector_proof_counts_issues_and_observes_phase(self):
        offsets = [[i * 8 for i in range(32)], [i * 8 + 4 for i in range(32)]]
        proof = plan_access(offsets, element_bytes=4, root_alignment=16, root_bytes=256)
        sectors = access_sectors32(proof, element_bytes=4, root_alignment=16)
        self.assertEqual(sectors, dict(per_phase={"0": 8, "16": 9}, upper=9))
        row = dict(operation="load", element_bytes=4, groups_by_width={8: 1}, sector_issues32_upper=9)
        text = ttgir("ld.global.v2.f32 {a,b}, [ptr];")
        self.assertEqual(extract_features(text, {"global_access": [row]})["static"]["global_sector_issues32_upper"], 9)
        self.assertIsNone(extract_features(ttgir("ld.global.f32 a, [ptr];"), {
            "global_access": [row],
        })["static"]["global_sector_issues32_upper"])

    def test_local_key_excludes_conversion_and_shared(self):
        base = "ld.global.v2.f32 {a,b}, [ptr];"
        incumbent = extract_features(ttgir(base), {})
        other = extract_features(ttgir(base, ".shared .align 16 .b8 frisk_shared[16];",
                                        "shfl.sync.idx.b32 a, b, 1, 31, -1;", "bar.sync 0;"),
                                 {"execution_trace": [trace("frisk.convert_layout", shuffles=1, barriers=1)]})
        self.assertEqual(static_key(other, local=True), static_key(incumbent, local=True))
        self.assertLess(static_key(incumbent), static_key(other))
        comparison = compare_features(incumbent, other)
        self.assertEqual(comparison["structural_relation"], "dominates")
        self.assertEqual(comparison["relation"], "incomparable")
        self.assertEqual(pareto_frontier({"incumbent": incumbent, "other": other}), ["incumbent", "other"])

    def test_native_binding_and_driver_waves(self):
        text = ttgir("ld.global.v2.f32 {a,b}, [ptr];")
        grid = [240, 1, 1]
        with tempfile.TemporaryDirectory() as tmp:
            cubin = Path(tmp) / "kernel.cubin"
            cubin.write_bytes(b"\x7fELFsynthetic-image")
            cubin_sha = sha(cubin.read_bytes())
            entry = "frisk_execution"
            raw = "Resource usage:\n Function frisk_execution:\n  REG:186 STACK:0 SHARED:0 LOCAL:0\n"
            evidence = {
                "cubin": cubin,
                "receipt": dict(cubin_sha256=cubin_sha, ttgir_sha256=sha(text.encode()),
                                entry=entry, grid=grid, block=[128, 1, 1]),
                "compiler_exact": dict(kind="compiler-exact-postcompile", raw_resource_usage=raw,
                                       raw_resource_sha256=sha(raw.encode()),
                                       resources=parse_compiler_resource_usage(raw, entry),
                                       identity=dict(entry=entry, cubin_sha256=cubin_sha, tool_sha256="a"*64)),
            }
            known = extract_features(text, {}, grid=grid, native=evidence)["native"]
            self.assertEqual(known["registers_per_thread"], 186)
            self.assertIsNone(known["theoretical_cta_waves"])
            evidence["driver"] = dict(cubin_sha256=cubin_sha, entry=entry, device_identity="gpu-test", resources={
                "source": "CUDA-driver-loaded-function", "threads_per_block": 128, "dynamic_shared_bytes": 0,
                "registers_per_thread": 186, "shared_bytes": 0, "local_bytes": 0, "total_shared_bytes_per_block": 0,
                "device_limits": dict(multiprocessor_count=114, max_blocks_per_sm=32, max_threads_per_sm=2048,
                                      reserved_shared_bytes_per_block=1024),
                "residency": dict(kind="theoretical-resource-ceiling", api="cuOccupancyMaxActiveBlocksPerMultiprocessor",
                                  active_blocks_per_sm=2, launchable_by_resources=True),
            })
            known = extract_features(text, {}, grid=grid, native=evidence)["native"]
            self.assertEqual((known["status"], known["resident_ctas_per_sm"], known["theoretical_cta_waves"]),
                             ("driver-confirmed", 2, 2))
            with self.assertRaisesRegex(ValueError, "compile receipt"):
                extract_features(text, {}, grid=[228, 1, 1], native=evidence)
            evidence["compiler_exact"]["resources"]["registers_per_thread"] = 32
            with self.assertRaisesRegex(ValueError, "raw facts"):
                extract_features(text, {}, grid=grid, native=evidence)

    def test_geometry_mismatch_rejected(self):
        text = ttgir("mov.u32 $0, 0;")
        with self.assertRaisesRegex(ValueError, "launch metadata"):
            extract_features(text, {"tiled_execution": {"grid": [4, 1, 1]}})
        with self.assertRaisesRegex(ValueError, "warp or cluster"):
            extract_features(text, {}, block=[64, 1, 1])


if __name__ == "__main__":
    unittest.main()
