# P1.4b Complete Attention implementation report

Status: implementation and required execution/regression gates complete. Independent task and whole-source reviews found three support-domain issues; all were fixed with failing/passing regression evidence. Independent task review and strongest whole-source re-review are Approved; no Critical or Important findings remain. Records: p1_4b_artifacts/task-review.md and p1_4b_artifacts/whole-source-review.md.

## Current reproducible artifact

- Directory: `guoqiao/optimize_plan/p1_4b_artifacts/post-review-execution/run-zopay3ho`
- Manifest SHA256: `614eb8d62d93ac2291668442234641e7414a43cff6dfb025e613444399fa5440`
- Cubin SHA256: `c2c32e0a3355a843ae7560ee70b3913ec3dd7b123461d780f259dbe3f187bb62`
- Source/tool/config/input/selected IR/contract JSON/TTGIR/LLVM/PTX/cubin hashes: current manifest; external identity in `current-artifact.json`.
- Exact-current-binary sanitizer commands/return codes/log hashes: `current-sanitizer-results.json`; binary/loader association: `current-sanitizer-artifact.json`.

Replay from `/home/baopeihua/frisk`:

```
/home/baopeihua/anaconda3/envs/torch/bin/python benchmark/layout/execution/run_execution.py --output guoqiao/optimize_plan/p1_4b_artifacts/replay
```

Each invocation creates a fresh `run-*` directory and forces Triton compilation with `TRITON_ALWAYS_COMPILE=1`. Unsupported inputs fail without publishing a new executable. The kernel has four logical byte-pointer arguments Q/K/V/O; Triton appends a fifth physical global-scratch pointer (null, size0). Launch is one CTA of128threads on device1.

## Implemented compilation boundary

Original fixed-ABI `test/Transforms/attention-layout-fixed-abi.mlir` passes through existing Frisk inference, `frisk-export-execution`, schema1 JSON, actual-operation scalarization into TTGIR, pinned Triton LLVM/PTX/cubin lowering, and H100 launch. No fixed hand-written Attention substitution or regular Triton dot/layout selection is used.

C++ preflights the complete operation set, single sm_90a kernel/128-thread parallel, and `verifyMaterializedLayouts`. It exports actual SSA IDs, `ExecutionEnumeration` coordinates/first owners, real `StorageAliasFootprint` addresses, and C++ `buildReductionLayoutProof` fibers/nodes/broadcast holders before atomic JSON rename. Backend finite-domain equivalence checks permit only proven GF(2)-equivalent address transforms. Unsupported dtype/topology/maps/contracts fail explicitly.

The generic emitter walks actual operations and scalarizes thread/register ownership. Triton handles the scalar ABI, inline assembly and lower compiler pipeline; it does not select native tensor layouts. Source clone/conversion semantics are preserved by actual SSA walking; separate fixtures exercise the same backend load/convert/store implementation. This is a bounded single-CTA backend, not general Frisk GPU support.

Global alignment1 uses byte loads/stores with actual pointer offsets1/2/3. Shared roots are allocated from actual footprint/alignment and retain32B swizzle. Copy supports explicit f16/16-bit endpoints; non-f16 Copy is rejected. SS WGMMA supports the selected K-major descriptors and rejects other majors. Actual SS atom64x64x16 and RS64x16x16 repeatedK4 consume descriptor start/leading/stride/swizzle/phase and checked native accumulator/register maps. RS packs low/high halfwords into four32-bit operands per slice. Descriptor MN-major maps RS transB1. CTA barrier and proxy fence publish Shared; WGMMA fence/commit/wait0 serialize each operation. No overlap/cp.async/TMA is introduced.

Reduce consumes the exported canonical_fiber_tree_v1 binary dependency order and first_owner leaves. Holder-relative trees are checked equal before uniform scalar/shuffle generation; completed roots reach every output replica. Broadcast explicitly supports rank1→rank2 dimensions=[1] with matching source extent and verifies same-thread source slots. Other axes/ranks are rejected. tensor.empty produces no reads.

Math settings: explicit mul.rn/sub.rn/div.rn; exp=ex2.approx.f32 after mul.rn by FP32 log2(e); no FTZ modifiers, fusion or reassociation; P=cvt.rn.f16.f32. Frozen numerical gate in frozen_numerics.json was not relaxed.

## Final validation

| Gate | Result / evidence |
|---|---|
| Complete Attention |135/135 individually checked launches:15cases×3offsets×3repeats; fresh O sentinel each launch; all O finite, whole output within frozen .002+.002abs(ref), output guards and all Q/K/V bytes/guards unchanged |
| Cases |seed0–9, zeroQK, constant, signedcoords, onehotV, cancelV |
| Maximum error |1.4137476682662964e-05 at seed3 coordinate[48, 11]; cancelV error0 |
| Independent reference |Explicit NumPy FP32 QK products/reduction, stable softmax, RNE-f16 P, FP32 PV; no torch/TF32 matmul |
| New-backend transport/cast |54/54 individually checked launches; two distinct bound distributions, replica/first_owner and unit-axis fixtures; actual load→Shared convert→store; RNE halfway ties/±0/subnormals; offsets1/2/3; per-launch sentinel, exact bytes and unchanged input guards (`transport/result.json`) |
| Isolated packing |2048 low/high32-bit words exact at offsets1/2/3, checked as low\|high<<16 after actual P production (`packing/result.json`) |
| SS/softmax slices |SS maxerror9.54e-7; normalized softmax9.32e-9 (`debug-slices.log`) |
| Exact-current-binary sanitizers |memcheck/racecheck/synccheck/initcheck allreturn0 and0errors/hazards; each tool covers seed0×3offsets×3separate repeats=9launches, separately from full135numerics |
| New C++ tests |Release2/2;Debug2/2 (`cpp-release.log`,`cpp-debug.log`) |
| Final complete CPU regression |Release gtest577/577 (67suites,272.909s); ctest10/10 (17.73s); lit73/73 (60.65s) |
| Focused Python |Exporter3/3; backend6/6 after review fixes; numerical cancellation fixture1/1 |

Sanitizers supplement explicit async proxy/barrier/fence/commit/wait contracts and initialized operands; their summaries are not a proof of every WGMMA asynchronous behavior. No scheduling overlap is attempted.

## Toolchain, resources and timings

Device1 UUID `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618` was selected after GPU0 had67GB occupied and GPU1 was0%/42MiB. Triton3.4.0 / Torch2.8.0+cu128. Actual wheel ptxas12.8.93 path/hash and native libtriton.so hash are in versions.json/current manifest. This native hash pins the packaged LLVM implementation; a separate LLVM semantic version is not exposed by its Python interface. System CUDA nvcc12.8.61 and Compute Sanitizer2025.1 are separate identities. Exact selective make_llir source is saved in selective_lowering_source.py.txt; starting with TTGIR skips layout-selection stages.

CUDA function attributes:64registers,6144static Shared bytes,0local bytes/spills. Triton dynamic Shared metadata is0; it is not total physical Shared. Final invocation inference=7382952996ns; full input→cubin wall time=8608972132ns (includes required host/CUDA setup within the driver). Twenty raw GPU event samples are saved; median=21.5840us. These small-tile functional timings do not imply layout performance benefit.

## Review fixes and historical failures

1. Initial full GPU seed0 failed by.291107 although finite/guards passed. SS and softmax slices isolated RS MN-major transB0; a failing regression preceded descriptor-derived transB1. Historical arrays/summary remain under debug-rs-transpose, outside current artifact identity.
2. Task review found SS non-K-major and32-bit Copy silently accepted beyond implemented instruction domains. Both behavioral REDs are in review-fixes-red.log; narrowed-domain fixes pass5/5 in review-fixes-green.log.
3. Whole-source review found broadcast dimensions ignored. A legal dims0 replicated counterexample failed expected rejection; explicit rank/axis guards now pass6/6 (`broadcast-domain-red.log`,`broadcast-domain-green.log`). Transport validation was also strengthened to check every launch and input guard, producing54records.
4. cancelV originally alternated columns. The exact old expression fails K-axis cancellation; the current fixture alternates rows with zeroQK/uniformP, and its independent reference is exactly0 (`cancellation-red.log`,`cancellation-green.log`).
5. Exact-cubin ctypes helper initially returned11 because it supplied four parameters to a physical five-parameter Triton entry, causing a host parameter-array overread. It now passes null scratch; all four final exact-binary sanitizer runs return0. The failed helper runs are diagnostic failures, not acceptance evidence.
6. First ctest attempt exposed missing test_pass executables and stale selection source manifests. Full Release build corrected infrastructure without changing existing expectations; final full tests passed. No unchanged passing CPU suite was rerun after Python-only review fixes.

Exporter behavioral RED/GREEN and later review regressions are retained. Strict per-feature TDD ordering was not preserved for all later transport extensions: backend's initial absent-module import error was not adequate behavioral RED, and some transport code preceded its dedicated fixture. This process limitation is recorded rather than inventing historical evidence.

## Preservation and source scope

`preservation-audit.json` confirms all3pre-existing dirty file hashes unchanged. Intended tracked edits are Passes.h/Passes.td and the conversion/unit-test CMake registrations; new source is ExportExecution.cpp, ExecutionExportTest.cpp and benchmark/layout/execution/*.py. Existing dirty artifact directories remain. No staging, commit, push, local files or worktree creation occurred. All launched build/test/GPU jobs have ended.
