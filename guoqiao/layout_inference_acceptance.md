# Layout inference acceptance — M6 evidence ledger

Updated 2026-10-10. **Overall M6: PASS for the declared scope, including full regressions and independent final review.**
The final CPU/SM90 benchmark gate passes after the checked affine optimization;
the independent reference baseline has been reviewed and accepted. Historical
first-run compile failures are retained below, not overwritten.
No thresholds were relaxed; no automatic commit/push/worktree was used.

## Evidence and provenance

- Working tree is intentionally dirty (Task24–29); Git HEAD alone is not a
  reproducibility identifier. Source/config manifests and per-input, promoted,
  adapter, lowered, PTX/cubin and tool SHA-256 values accompany conversion data.
- First real GPU data: `/tmp/frisk-task29-conversion-gpu1-escalated.json`;
  artifact root `build-m6-release/benchmark/layout/conversion/measurements/run-l2h8x_r0`.
  Each case retains commands, stdout/stderr, input and all generated artifacts.
  Source/config manifests are in that root, not inferred from a later checkout.
- Hardware: device1, H100 PCIe SM90, UUID
  `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618`, driver580.178.04,
  CUDA12.8.61. Recorded pre-run load0%,4MiB; clock policy unmodified.
  No clocks/power/process management was performed. Sandbox child NVML failed;
  scoped escalated actual run succeeded.
- CPU Release evidence `/tmp/frisk-task29-solver-release.json` is exploratory:
  the run overlapped CPU builds. It includes source/config/binary digests,
  stable solution hashes, raw samples and phase median/p95. Isolated final
  measurement has now completed at `/tmp/frisk-task29-solver-final.json`;
  all final regressions also pass, as recorded below.
- Task27/28 evidence remains in `.superpowers/sdd/task-27-report.md` and
  `.superpowers/sdd/task-28-report.md`; their prior independent reviews passed.
  Their historical totals are not presented as current final-gate reruns.
- Task28 canonical corpus SHA-256:
  `bb4a69f647305e9017ecdbb063e87a1a4cf39ec33cf5ee77ec0dd3936842fd44`;
  exporter SHA-256:
  `5ebad2735a4fa6368b4901dd518fee73fcc4aa7ff2f9e204b134f68cac9e7453`.
  Native export/re-export commands, source revision and dependency evidence are
  documented in [tilelang_layout_reference.md](tilelang_layout_reference.md).
- Final unit binary SHA-256: `79d03d72a2549aa5ba06ec4e3009ce896771a57f97ec34b8ebfc3add93f262e7`.
- Final CPU raw-data SHA-256: `d276b9e1485ac76aded92d5384c9a2b24560040c627f1c68efb10fa9a7bc122a`.
- Task27–29 reviewed source/test delta SHA-256: `076d8f0138653276d9684577290a6076778cd5a5b6684dbb36bdc90da69f9196`
  (`/tmp/frisk-m6-whole-review.diff`, sequential deltas against pre-task dirty snapshots).
  These source/binary identities bind the unit/property/negative-test rows in
  the matrix; GPU rows additionally bind to their per-input/artifact manifest.

## Independent review checkpoint

- Task27: `/tmp/frisk-m6-task27-review.md`, spec and quality PASS.
- Task28: `/tmp/frisk-m6-task28-review.md`, spec and quality PASS after the
  artifact-publication correction.
- Task29 runner: `/tmp/frisk-m6-task29-runner-review.md`, scoped PASS;
  22 Python tests. This does not accept any generated benchmark result.
- Task29 statistics: `/tmp/frisk-m6-task29-statistics-rereview.md`, scoped
  spec and quality **PASS** after the finalized-identity-snapshot fix.
  Latest 12 statistics tests pass, including reverse variable/constraint
  insertion without mutation of the failed caller graph.
- Task29 GPU: `/tmp/frisk-m6-task29-gpu-review.md`, scoped code spec and
  quality **PASS** after memory-space rejection and negative regressions.
  Root reran all 16 statistics/adapter units successfully. The revised harness
  binds reference source to its build manifest and captures load near and after
  measurements. A separate post-rebuild evidence review explicitly approved
  the exact final pending reference data for acceptance; binding hashes below.
- Task29 affine evaluator: `/tmp/frisk-m6-task29-affine-review.md`, scoped
  spec and quality **PASS**; root reran all 7 affine regressions successfully.
  The only remaining suggestion is additional lower-bound overflow and
  successful negative-divisor boundary tests. The reviewer found no matching
  arithmetic defect; the whole review retained this as a nonblocking follow-up.
- Whole M6 final review: `/tmp/frisk-m6-whole-review.md`, spec, quality and
  technical readiness PASS after reconciling final regression evidence and
  documentation, with no Critical/Important findings. A second nonblocking follow-up is
  suppressing expected git-error noise in the negative exporter fixture.
  Neither suggestion was silently discarded or used to waive a failing gate.

## Final measured gate and accepted baseline

`/tmp/frisk-task29-conversion-final.json`, artifacts
`build-m6-release/benchmark/layout/conversion/measurements/run-tcglhe9n`,
contains the final rebuilt 11 correctness cases, 33 clean sanitizer runs and
1,000 samples per origin for each mandatory runtime case. Device visibility
was not remapped (`CUDA_VISIBLE_DEVICES` unset). Load snapshots were 0%, 2%,
6% before the run, near timing and after timing; all report 4MiB and 1620MHz.
These snapshots are disclosed, not interpreted as continuous idleness or
attributed to any particular process. Builds and other task benchmarks were
stopped during these measurements.

| Case | Reference median (ns) | Inferred median (ns) | Runtime ratio | Layout/full compile (ms) | Compile ratio |
|---|---:|---:|---:|---:|---:|
| identity_f32 | 4064 | 4096 | 1.007874 | 3.129421 / 82.375553 | 3.798968% |
| shuffle32_f32 | 3840 | 3840 | 1.000000 | 3.283414 / 82.635025 | 3.973393% |
| shared64_f32 | 3840 | 3872 | 1.008333 | 5.860232 / 87.896523 | 6.667194% |

Two further independent uncached compile-only repetitions are retained at
`/tmp/frisk-task29-conversion-final-repeat1.json` and `...-repeat2.json`.
Their per-case ratios are 4.077%/3.810%/6.801% and
4.007%/3.705%/7.165%, respectively; no repetition was dropped for exceeding a
threshold. The earlier failing, pre-optimization runs remain separate controls.

Final isolated CPU pipeline medians for 10/50/100/250 real operations are
18.767/98.726/211.075/615.492ms. Each corpus has five warmups, thirty samples,
stable solution hashes, and the unchanged 256/32/65536 configuration; raw
phase samples and median/p95 are in `/tmp/frisk-task29-solver-final.json`.
Multiple connected ten-operation components are used, not a claim about one
connected 250-operation component. CPU-only runner returns `PARTIAL_PASS` as
designed; the combined runner below returns all four gates **PASS**.

Independent performance review `/tmp/frisk-m6-task29-gpu-review.md` verified
all 66 artifact hashes, the current source manifest and seven tool digests,
correctness/safety logs, exact reference samples and matching environment.
Only `review_status` was changed from `pending` to `accepted` when adding
`benchmark/layout/baseline/sm90_layout_baseline.json`; reference data were
not recomputed or replaced by inferred samples.

- Raw final GPU SHA-256: `3ba20f77e87ddb78828f3c0830f9c8dfc57c1f6c512d7a42d34e6321518b8202`.
- Reviewed pending baseline SHA-256: `02211cfa7ac0aacbf6af8792d49f311f9d1dea581a2a6424381759a953c634a6`.
- Accepted baseline SHA-256: `a5a8edd1fda0a237eccfe24e8d95ddca683ca5685022220682ed51cf5c69c0c2`.
- Status-independent canonical baseline-data SHA-256: `2092c8b21ec01008783d6398130aba5048bc7e86c3ae5b4f61a99cca724eb617` (remove only `review_status`, sorted JSON keys, compact separators).

```bash
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 benchmark/layout/run_layout_benchmarks.py \
  --solver-json /tmp/frisk-task29-solver-final.json \
  --conversion-json /tmp/frisk-task29-conversion-final.json \
  --baseline benchmark/layout/baseline/sm90_layout_baseline.json \
  --check --output /tmp/frisk-m6-final-benchmark-gate.json
```

Exit 0; solver, runtime, compile and overall benchmark gates PASS.

## Final full regression

| Check | Fresh result | Raw evidence |
|---|---|---|
| Debug all-target build | PASS, exit 0 | `/tmp/frisk-m6-final-build.log` |
| Complete unit suite | **405/405 PASS**; shards 102+101+101+101, no disabled/failing/error tests | `/tmp/frisk-m6-final-unit-{0,1,2,3}.{log,xml}` |
| Complete lit suite | **58/58 PASS** | `/tmp/frisk-m6-final-lit.log` |
| Debug CTest | **7/7 PASS**, including 12 exporter, 22 runner and 4 compiler Python checks | `/tmp/frisk-m6-final-ctest.log` |
| Release promotion CTest | **1/1 PASS**, six real-generator integration checks | `/tmp/frisk-m6-final-generator-ctest.log` |
| Python normalization | **5/5 PASS**, explicitly loaded freshly built FFI | `/tmp/frisk-m6-final-normalization.log` |
| Production legacy API isolation | PASS | `/tmp/frisk-m6-final-api.log` |
| SM90 layout-tool SS/RS/output/rejection smoke | PASS | `/tmp/frisk-m6-final-tool.log` |
| Task28 semantic differential | 20 equivalent + 3 expected rejections PASS again within full unit run | unit shards above; pinned corpus digest above |
| Source whitespace check | `git diff --check`, exit 0 | final controller command |

Unit shards all returned exit 0. No focused filter substituted for the complete
unit/lit suites. Final regression ran after isolated timing measurements; their
CPU load is not part of the accepted measurement window.

## First real runtime and complete-compile results

Runtime uses independent coordinate reference cubins and actual adapter-generated
inferred cubins, with the same driver launch ABI and stream/event timing.
Each required case uses100 warmups and1000 alternating measurements. The
complete compile starts from the same Frisk input and ends at its cubin; no
reference build, dependency build, GPU execution or deliberate wait enters it.

| Case | Correctness / safety | Inferred/reference median | Layout/full compile | Gate |
|---|---|---:|---:|---|
| identity_f32,32 threads | PASS |1.000000|10.650527/87.756022ms=12.14%|Compile FAIL |
| shuffle32_f32,32 threads | PASS |0.992063|10.764240/89.348535ms=12.05%|Compile FAIL |
| shared64_f32,64 threads | PASS |1.000000|21.088304/104.216375ms=20.24%|Compile FAIL |

All11 compiled cases passed nonuniform seeded and boundary-bit correctness:
the three above, `replicated64_f32`, and shuffle transport for
i8/i16/i32/i64/f16/bf16/f64. All33 memcheck/racecheck/synccheck invocations
reported positive zero-error completion. This validates these cases, not arbitrary
Frisk kernels. Shared capacity/alignment/unique-writer/uniform-barrier checks
remain static adapter obligations as well as runtime checks.

The first reference record `/tmp/frisk-task29-reference-pending.json` remains
historical **pending review** evidence, not the accepted baseline. At that
checkpoint, runner replay with the pending baseline exited1:
compile FAIL, runtime INCOMPLETE, overall FAIL
(`/tmp/frisk-task29-gpu-first-gate.json`). Raw paired runtime ratios meet1.03,
and accepted runtime status still required baseline/performance review.

Independent compile-only repeats preserve the failure: identity11.18%/11.27%,
shuffle9.92%/10.61%, shared19.58%/19.09%; raw files
`/tmp/frisk-task29-conversion-compile-repeat{1,2}.json`.

The first uncontended optimized compile-only sample
(`/tmp/frisk-task29-conversion-optimized1.json`) is below the unchanged 10%
threshold: identity 3.281470/79.801279ms = 4.11%, shuffle
3.447411/89.417047ms = 3.86%, shared 6.299364/86.897415ms = 7.25%.
This replaces per-point affine Attribute construction with checked direct
integer evaluation, without removing proof calls or reducing their budgets.
At that checkpoint, boundary tests and the final GPU/regression run were still
pending. They have since passed; the final table above, not this intermediate
sample, defines acceptance.

## Global Constraint matrix

PASS below applies only to the named supported domain and actual evidence;
OUT_OF_SCOPE preserves the pre-agreed full-instruction-lowering boundary.

| Global constraint | Current evidence / command family | Status and uncovered boundary |
|---|---|---|
| SM90/SM90a target isolation | SM90 target/unit suites; explicit device1 runtime; no SM90 string found in generic candidate solver/relaxation/propagation scan | PASS; no broader target claim |
| Full WGMMA/TMA/mbarrier lowering excluded | Static instruction/copy/reduction contract suites; conversion-only generator | OUT_OF_SCOPE for full instruction lowering; not a substitute for required conversion runtime |
| Pinned TileLang reference; no TVM/TIRX dependency | Task28 actual pinned revision5e149e31674658f94779c7d0c6039549a1853123 export,20equivalent+3expected rejection, byte-identical re-export, audit digests | PASS, including final differential rerun; no production upstream dependency |
| Local/Register uses RankedTensor SSA | Normalization/materialization/layout pipeline tests, legacy API scan | PASS in final suite and production boundary check |
| Shared/Global MemRef plus explicit binding | Storage root/alias/capacity tests; checked promotion with exact explicit root views; malformed-space/capacity rejection | PASS in final units and promotion tests |
| Conversion/rematerialization ordinary cost candidates | LayoutConversionChoice/Hoist tests; physical leader accounting statistics tests | PASS; no zero-conversion priority introduced |
| Hard constraints dominate cost; unknown never proves | Solver property/oracle, candidate/relaxation/contract negative suites and checked affine failure tests | PASS; proof budgets unchanged |
| Stable identities and deterministic decisions | Task27 insertion-order/process tests; independent nonempty statistics replay; four CPU stable hashes; finalized identity regression | PASS in final unit/lit/CPU runs |
| User errors are diagnosed, not assertions | Negative unit/lit tests; statistics write-error rollback; malformed adapter/promotion tests | PASS for tested errors; arithmetic failure is conservative |
| Same-task design synchronization | Design §25 records collectors, checked evaluator and test-only executable ABI | PASS; implementation scope and measured limits recorded |
| Long-term plan/comparison maintenance | Implementation plan §29 and comparison M6 summary updated with final evidence | PASS; no unsupported superiority claim |
| Preserve user workspace | No reset/checkout/commit/push/worktree; unrelated Task24–28 changes retained, build.sh untouched | PASS; direct source main, HEAD36df49f unchanged |
| Zero race/OOB/invalid ownership/unresolved/nondeterminism |11GPU correctness cases+33sanitizers; artifact checks; final rejection/ownership suites and stable IR hashes | PASS for measured/tested cases, not arbitrary program proof |
| Runtime<=1.03, compile<=.10, exact256/beam32/65536 | Final combined runner above, accepted independent reference, actual component products/frontiers/budgets and same-sample cubin compilation | PASS for the stated conversion cases and CPU corpus; no threshold change |

## Reproduction commands

Use project Python `/home/baopeihua/anaconda3/envs/torch/bin/python3.10`.

```bash
cmake -S . -B build-m6-release -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir \
  -DFRISK_ENABLE_CUDA_BENCHMARKS=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DPython3_EXECUTABLE=/home/baopeihua/anaconda3/envs/torch/bin/python3.10
cmake --build build-m6-release --target frisk_layout_conversion_bench frisk_layout_solver_bench --parallel 12
build-m6-release/benchmark/layout/conversion/frisk_layout_conversion_bench --device 1
build-m6-release/benchmark/layout/conversion/frisk_layout_conversion_bench --compile-only
build-m6-release/benchmark/layout/solver/frisk_layout_solver_bench
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter=LayoutStatisticsTest.*
```

Full regression commands (the four unit shards together cover the entire suite):

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 12
cmake --build build --parallel 12
for shard in 0 1 2 3; do
  GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=$shard \
    build/unittests/Dialect/Frisk/FriskLayoutUnitTests
done
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -sv build/test
ctest --test-dir build --output-on-failure
ctest --test-dir build-m6-release -R FriskLayoutConversionGeneratorTest --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/sm90_layout_tool_test.py --build-dir build
git diff --check
```

GPU commands may require scoped execution permission. Raw `/tmp` logs and build
artifacts are local evidence, not tracked source; the accepted reference samples,
provenance and fixed corpus are repository files. No speed comparison against
TileLang or Triton is supported by these semantic or conversion-only tests.
