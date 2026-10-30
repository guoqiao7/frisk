# P1.4c complete asynchronous Attention implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. This is ONE delivery task; its transport paths are internal validation steps.

**Goal:** Complete cp.async and TMA Attention on the existing contract-driven backend, retaining weak-alignment baseline.

**Architecture:** Export selected actual SSA/layout/Copy contracts through the existing JSON bridge. Extend the same scalar TTGIR/PTX emitter and runner for two explicitly strong-alignment input ABIs. cp.async uses checked vector issuer addresses, commit/wait and CTA visibility; TMA uses actual-map-derived tensor descriptors, elected issuer and one-shot mbarrier completion, with no overlap.

**Tech Stack:** Frisk C++/MLIR, Python, Triton3.4.0 selective backend, bundled ptxas12.8.93, H100 sm90a.

## Global Constraints

- All source/evidence stays under /home/baopeihua/frisk on H100. No worktree, staging, commit, push, or local retained copies. Baseline HEAD441dae548a6732f700b6863b5e266cd57e98dadc.
- One backend/runner/manifest, one execution plan, one implementation report. cp.async and TMA are internal steps of ONE task. No hand-written substitute Attention or silent layout reselection.
- Global alignment1 original ABI remains unchanged; strong ABI variants are explicit. Consume actual Global roots/Shared maps and selected Copy contracts. Reject unsupported contracts/alignment/offset/descriptor before launch.
- Single CTA128threads Q/K/V64x16f16 O64x16f32. Same SS QK scale.25 stableSoftmax P RNEf16 RS PV Store algorithm. Shared swizzle and actual footprint retained.
- f16-rounded finite[-1,1] input. Independent CPU FP32QK/stableSoftmax/P RNEf16/FP32PV. All O finite and abs(got-ref)<=.002+.002abs(ref); never relax.
- seeds0-9, zeroQK, constant, signedcoords, onehotV, cancelV. cancelV rows alternate +/-1 with zeroQK and exactzero reference. Each path3legal offsets x3repeats eachcase, reset O sentinel each launch, verify full output/QKV bytes/guards. Baseline offsets1/2/3.
- cp.async must implement width/issuer/commit/wait/CTA visibility. TMA must implement descriptor hostsetup/elected issuer/mbarrier init/expectedbytes/wait/visibility. One-shot no overlap; no multistage, warp specialization, cluster multicast, or TMA store.
- Exact final cubin memcheck/racecheck/synccheck/initcheck each seed0 x3legal buffer configurations x3repeats; describe coverage limits. Standalone async transport slices bitexact plus full chains.
- TRITON_ALWAYS_COMPILE=1, fresh run-* directories, no self-hashed manifest. Record source/input/selectedIR/contracts/TTGIR/LLVM/PTX/cubin/tool/config/ABI/GPUUUID. Verify actual PTX physical ABI including null scratch and TMA descriptor arguments/alignment/lifetime.
- Record physical registers/shared/local/spills, descriptor+barrier storage, host setup, inference/compile/GPU rawtimings; no performance benefit claim.
- GPU1 UUIDGPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618 preferred only after occupancy check. Single source/build/GPU owner. New C++ tests Debug+Release, affected gates, final full Release including selection relinking, CTest/lit.
- Meaningful behavioral RED before production change, GREEN after. Independent task review then strongest whole-source/data/docs review, fix and covered retest and re-review. Close all launched jobs.

### Task 1: Complete P1.4c asynchronous Attention delivery

**Files:** Modify benchmark/layout/execution/backend.py and run_execution.py, lib/Conversion/FriskLayoutToGPU/ExportExecution.cpp as needed; add focused async helper(s) and tests in existing execution directory; add explicit strongABI Attention fixtures under test/Transforms; extend unittests/Dialect/Frisk/Layout/ExecutionExportTest.cpp as needed. Maintain report guoqiao/inference_detail/p1_4c_implementation_report.md and evidence guoqiao/optimize_plan/p1_4c_artifacts.

**Interfaces:** Existing schema1 JSON actual SSA/storage/distributed/operation exports; emit(data) -> (TTGIR, config); same runner accepts explicit input/path choice. Preserve baseline calls. Async transport helper validates and derives instruction/descriptor plans from exported contracts and concrete maps; host helper validates pointer/config before CUDA descriptor encoding/launch. Version schema only if genuinely necessary and document it.

- [x] Capture tracked source/tool/input baseline hashes and actual selected Copy schemas; inspect AGENTS and TDD skill before code.
- [x] Write failing positive/negative tests for strongABI contract consumption and unsupported alignment/offset/maps; record real behavioral RED.
- [x] Extend export/backend with minimal explicit cp.async support; checked vector groups and issuer addresses, commit/wait0/barrier. Test bitexact transport then complete chain immediately.
- [x] Add failing descriptor/mbarrier/physicalABI tests; implement TMA from actual root/Shared maps and selected contract, CUDA tensorMap hostsetup with aligned lifetime-owned descriptor storage. Test transport then complete chain.
- [x] Generalize existing runner/manifest without duplicated infrastructure; full frozen numerical/safety matrices for both paths and baseline, rejected negative inputs before launch.
- [x] Run new C++ Debug/Release and affected tests, final full Release/CTest/lit, exact final cubin four sanitizers. Preserve all commands/results/hashes and raw timing/resource/setup samples.
- [x] Self-review, write implementation report; independent task review fixes/retests; strongest whole source/data/docs review fixes/retests; root identity/reference audit; close jobs and update same plan/ledger/roadmap.

## Completion

P1.4c complete. Independent task review and strongest whole source/data/docs review Approved; root reference/identity audit verified. All jobs closed. Two Minor observations retained in whole-source-review.md: redundant existence assertion and config Shared payload naming (physical total recorded accurately). No production source changes after final evidence freeze; no stage/commit/push/worktree.
