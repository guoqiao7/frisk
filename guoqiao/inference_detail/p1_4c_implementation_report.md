# P1.4c complete asynchronous Attention implementation report

Status: complete implementation and all GPU/CPU validation. Independent task and strongest whole source/data/docs reviews Approved; root reference/identity audit verified. No Critical or Important findings remain. All launched jobs are closed.

## Architecture and boundary

One schema1 actual-SSA/storage/execution/Copy/MMA/reduction JSON bridge, one existing scalar TTGIR/PTX backend and one runner now execute weak vector, cp.async, and TMA Attention. No alternate hand-written Attention, silent layout reselection, worktree, local retained copy, staging, commit, or push. Baseline HEAD 441dae548a6732f700b6863b5e266cd57e98dadc. All edits/evidence under /home/baopeihua/frisk.

New explicit attention-layout-cp-async-abi.mlir and attention-layout-tma-abi.mlir declare Q/K/V root alignment128 and leave O alignment1. They bind complete checked selected Copy contracts, a contiguous-register execution map, and actual row-major Global views. Source maps/Shared layouts/MMA maps pass the same inference and materialized verifier. Weak alignment1 fixture remains unchanged. Legal strong offsets0/128/256 respect entire declared ABI, not merely instruction alignment; baseline offsets1/2/3 remain supported. Host preflight rejects misaligned pointers before descriptor encoding/launch.

Copy helper checks every contract field/version/target/validshape/guard/issuer/completion, f16 16-bit exact full coordinate domain, declared root alignment/direction, actual view interval equality at its exported root coordinate, root bounds and disjoint destination intervals. cp.async additionally validates width4/8/16, contiguous register groups for both endpoints, vector alignment, first-owner election and exact logical/physical coverage. Actual selected width16 issues one vector per thread, commit_group, wait_group0, CTA barrier and async-proxy fence. There is no overlap.

TMA supports actual full2D64x16f16 root boxes only, descriptor reversed box[16,64], source strides[32], interleave none, swizzle0/32 and checked aligned offsets/full actual root maps. Every source/destination address is checked against descriptor formula, including canonical32B swizzle. Distinct8B Shared mbarriers are initialized by tid0, proxy-fenced and CTA-published; tid0 arrives with expected2048bytes and issues bulk.tensor.2d; all threads acquire-wait phase0 and meet a CTA barrier. No cluster multicast/TMA store, multistage, warp specialization or overlap. Existing actual-map WGMMA SS QK, scale.25, stableSoftmax, RNEf16P, RS PV and byte-safe O store remain the complete chain.

C++ export previously ignored explicit root alignment when directly analyzing aliases: strong root block arguments appeared alignment1 and were rejected. Fix reuses collectStorageRootContracts validation and dominance for view definitions, or own complete root declaration for external ABI serialization. It does not infer root guarantees from a child's demand. verifyMaterializedLayouts preflight and validated root contract compatibility remain mandatory. Schema1 unchanged.

## Descriptor and physical ABI

async_host.py encodes CUDA tensor maps with cuTensorMapEncodeTiled using full actual descriptor/root address. Each descriptor has128bytes in owned64-byte-aligned host storage, plus an immutable128byte device allocation; both live through launch completion. Final PTX physical ABI is parsed and checked: vector/cp.async Q,K,V,O,nullscratch (5 pointers); TMA Q,K,V,O,descQ,descK,descV,nullscratch (8 pointers). PTX pointers use .u64.ptr.global.align1; host preconditions enforce actual strong alignment. Descriptor arguments are device pointers, not host tensor-map addresses. Standalone loader constructs exactly as many kernelParams as final PTX and verifies count before launch.

## Current acceptance evidence

Authoritative current identity is p1_4c_artifacts/current-artifacts.json (external manifest/cubin SHA256, no manifest selfhash). Each fresh run-* includes input, selectedIR, actual execution JSON, config, TTGIR, LLVM/PTX/cubin, source/tool hashes, GPU UUID, compile/inference raw timing and physical resources. TRITON_ALWAYS_COMPILE=1 remains forced. current-sanitizer-results.json records exact same final cubin SHA256, all commands, return codes and log hashes.

- Complete numerical matrix:405/405 launches total (135 each vector/cp.async/TMA),15cases x3legal offsets x3repeats. Each launch resets O sentinel; checks all finite output against independent FP32QK/stableSoftmax/RNEf16P/FP32PV with abs<=.002+.002abs(ref), output guards and complete Q/K/V bytes+guards unchanged. Cases seeds0-9, zeroQK, constant, signedcoords, onehotV,cancelV. Maximum abs error for all paths1.4137476682662964e-05; cancelV exactly0.
- Standalone async transport:54/54 bitexact checks over both paths x all3actual Copy Shared maps x3offsets x3repeats. Probe observes Shared bytes after actual Copy and stores to O; arbitrary16-bit payloads include nonfinite encodings. Each launch checks payload/guards/unchanged inputs. It is transport instrumentation, not a substitute Attention. transport/results.json and each kernel artifact retained.
- Exact final cubin memcheck/racecheck/synccheck/initcheck:12/12 return0, zero errors/hazards. Each tool/path covers seed0 x3legal configurations x3repeats with full output/guards/input bytes checks. Sanitizers supplement explicit proxy/barrier contracts; they do not prove every asynchronous hardware behavior or every numerical case.
- New C++ actual strong export test: meaningful RED cpp-red.log (valid selected strong ABI fails export); Release3/3 and Debug3/3 ExecutionExportTest pass in build-m6-release-cpp-green.log/build-cpp-green.log.
- Async positive/negative Python tests8/8; export strong/scoping/absent/incompatible-root atomic preservation2/2. Baseline backend6/6, exporter3/3 and numerical fixture1/1 are rerun; python-validation-commands.json captures exact commands/results.

## TDD and fixes

strong-export-red.log and cpp-red.log record real valid-strong-export rejection before exporter edits. async-red.log records valid contracts rejected by old alignment1 backend; async-green.log records instructions/negative validation. physical-abi-red.log records actual generated pointer-qualified PTX rejected by too-narrow parser, followed by actual successful compilation/launches. root-bounds-red.log reproduces review findings: source/destination+2048B outside unchanged root and cross-thread physical collisions were accepted; root-bounds-green.log7/7 after exact-root interval and collision guards. Review also found overwritten host setup timings; runner now accumulates45case/config records (TMA135descriptor timings), not just final config.

The host entrypoint existence assertion alone was a weak RED; actual host alignment rejection, actual ABI parsing and GPU descriptor encoding/launch tests now provide behavioral coverage, but this process limitation is stated explicitly. Initial transport observer used values[3] (Q) instead of actual O root; unchanged-input validation caught it. Fixed observer resolves output from actual sorted root order; final54/54 acceptance is after correction. Historical failed attempts are not acceptance evidence.

## Toolchain/resources/timings

GPU1 GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618 selected after occupancy check (0%,40MiB); GPU0 occupied67384MiB. Triton3.4/Torch2.8cu128, bundled actual ptxas12.8.93, system CUDA12.8.61, ComputeSanitizer2025.1. Actual optimizer/ptxas/Triton/native compiler hashes are in manifests; native libtriton hash pins packaged LLVM.

All paths64registers,0spills,0local bytes. Physical Shared vector/cp.async6144B; TMA6168B =6144data+3x8Bbarriers. Descriptor storage per active config384Bhost payload plus alignment backing and384Bdevice; host/device lifetime retained. Fortyfive case/config setup samples (135descriptor timings for TMA) and20GPU event samples per final path are retained. Final vector/cp.async/TMA raw inference/fullcompile and medianGPU samples are summarized in current-summary.json (exact values belong to the latest frozen manifests). These are small-tile functional timing samples, not a performance benefit claim.

## Source scope and completed gates

Changed backend.py/run_execution.py/ExportExecution.cpp/ExecutionExportTest.cpp. Added focused async_copy.py/async_host.py, exact load_cubin.py, run_async_transport.py, test_async.py/test_async_export.py and two explicit strong ABI fixtures. Existing original fixture is unchanged. Project source edits and retained acceptance evidence stay in this repository; no git index/commit/push/worktree actions were performed during P1.4c implementation.

Final whole Release build completed, with selection targets relinked. Full gtest578/578 (67suites,276.374s), selection20/20 (172.864s), CTest10/10, and lit75/75 (60.78s) passed. GPU/descriptor/sanitizer jobs ended. Independent task review, strongest full source/data/docs review, root reference/identity audit and roadmap/ledger integration are complete.

## Additional review fixes

Root alignment guarantee: root-alignment-red.log records the actual plan accepting a rootalignment1/viewalignment128 counterexample. Async helper now requires equal root/view declared guarantee before applying root footprint proofs; clean-input regression8/8 passes. Tests no longer read untracked execution archives: they infer and export both checked source fixtures into self-cleaning temporary directories under the repository build directory. Physical-ABI parser unit uses a controlled pointer-qualified PTX fixture; actual final PTX is independently checked on every full run/exact cubin load. Strong scope/missing/incompatible root atomic tests regenerate selectedIR from source (2/2).

Initial full lit observed the two new fixtures before RUN comments were added and reported73pass/2unresolved (no RUN lines); comments now exercise inference, verification and idempotent replay, with fresh complete lit rerun75/75 passing. Archived initial log is final-release-lit-initial.log. No production change resulted from that test registration issue.

## Final source identity and sanitizer association

After the last code change (test temporary directories constrained to the repository), every path was freshly compiled and the full135 numerical launches repeated; final identity is current-artifacts.json and raw current-summary.json. All source/tool/artifact SHA256s were verified against disk (final-identity-audit.json). A proposed test-only sanitizer association was rejected by its own exact-cubin hash assertion: fresh compilation changes cubin debug/file metadata. Consequently all four sanitizers were rerun and passed on every final frozen cubin; no binary-equality association is claimed. Current exact sanitizer commands/log hashes and cubin SHA256s are recorded in current-sanitizer-results.json.

Final focused tests: async8/8; source-generated strong export2/2; baseline backend6/6; original exporter3/3; cancellation fixture1/1. Latest exact transport54/54 rerun after root alignment fix. New C++3/3 in both Debug and Release; full final Release578/578 across67suites, selection20/20, CTest10/10, full lit75/75. No CPU baseline expectations were weakened. Full Release build relinked selection support/bench targets before gates.

Final job closure: all launched build, CPU test, inference, GPU, transport and sanitizer processes have exited. Final identity verification passed for every current source/tool/artifact hash and all12 sanitizer results. Source remained frozen through the approved strongest whole-source/data/docs review. Independent task review: Approved (task-review.md), no Critical/Important findings; minor redundant host-entrypoint existence assertion is retained and its weak RED is explicitly disclosed.

Primary instruction/API references: [PTX ISA8.7 cp.async bulk tensor](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor), [CUDA12.8 Tensor Map encoding](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html). The implementation's narrower accepted map/alignment domain is checked independently of these hardware minimums.

## Final acceptance and review disposition

Task review: Approved (`p1_4c_artifacts/task-review.md`). Strongest whole-source/data/docs review: Approved (`whole-source-review.md`). Root independent reference/identity audit: verified (`root-reference-identity-audit.json`). No Critical/Important issues remain. Two nonblocking Minor observations are retained: the host-entrypoint existence assertion is redundant; config.static_shared_bytes represents data payload6144B while TMA physical Shared totals6168B including24B barriers. Physical resource evidence and this report use the correct total.

The complete task is accepted: both async full Attention paths and original weak-alignment regression passed. Final artifact/validation/review pointers and document hashes are in `p1_4c_artifacts/acceptance.json`; no self-hashed manifest. Source and evidence remain only on H100. No further staging, commit, push, or worktree action occurred after baseline441dae5.

Resource field clarification: config.static_shared_bytes currently denotes the 6144-byte Storage-root payload. physical_resources.static_shared_bytes is the actual total (6168 bytes for TMA, including24 barrier bytes). Use the latter for total hardware resource reporting.
