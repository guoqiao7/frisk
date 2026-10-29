# P0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development task-by-task with independent spec/quality review. Steps use checkbox syntax. User-approved source-directory/no-commit instructions override skill defaults for worktrees and commits.

**Goal:** 实施已批准的 O01→O03→O02 契约，使固定 ABI Attention 可验证、选择过程可解释、beam 解在原预算内得到局部改进。

**Architecture:** 根契约及 ABI 测试沿用现有 Storage 机制。独立 trace 记录器观察传播/搜索/实际成本贡献，不改变求解。邻域改进复用完整关系与成本检查。

**Tech Stack:** C++17、MLIR/LLVM、GoogleTest、lit/FileCheck、Python 验收工具。

## Global Constraints

- 完整设计：`guoqiao/optimize_plan/p0_implementation_design.md`，用户已确认。
- 代码基线 `219c0cf`；设计基线 `b237911`。当前 main 源目录直接修改，不创建 worktree，不 commit/push。
- 默认 exact=256、beam=32、每分量总尝试预算=65536 不变；不改九项成本模型、硬语义或支持域。
- 新生产行为必须有先失败测试；纯新增验收若原有实现通过则如实记录。
- 单个实现者负责构建，最多 `--parallel 12`；根代理协调，禁止并发 Ninja。
- 每个任务完成后独立 spec+quality 复审；全部完成后整体复审与完整回归。
- 基线本轮已验证：Debug 构建成功、417 unit、61 lit、Attention 单独推断成功。完整单测约 829 秒，局部迭代可用 Release 加速，最终 Debug 必须重建验证。

### Task 1: O01 fixed external ABI integration — complete, independently approved

**Files:** Modify `unittests/Dialect/Frisk/Layout/AttentionLayoutIntegrationTest.cpp`; create `test/Transforms/attention-layout-fixed-abi.mlir`; change unit CMake only if a new fixture definition is needed. Production fixes, if demonstrated, restricted to generic Storage/candidate rules. Do not modify trace/solver behavior.

**Interfaces:** Consume `frisk.storage_contract`, `analyzeStorageAlias`, `getStorageAliasFootprint`/`buildStorageAliasFootprint`, `buildFriskLayoutPipeline`, `verifyMaterializedLayouts`. Reuse current real Attention dataflow helper, extend it to follow proven static subviews where necessary.

- [x] Add fixture from existing complete Attention with explicit whole-root row-major Global contracts; no tensor encodings and no replaced intermediate computation. Root f16 map `(i,j)->(2*(16*i+j),0)`, f32 map `(i,j)->(4*(16*i+j),0)`; internal Shared remains unbound.
- [x] Add tests for root alignment 1 and explicit 16; all Q/K/V/O byte addresses after inference match an independent arithmetic oracle. Test shape `64×16` view into `128×16` root at row 16 and a misaligned static view into a sufficiently large row-major root. Expected oracle pattern:

```cpp
for (const auto &entry : footprint.entries) {
  const auto i = entry.view[0], j = entry.view[1];
  EXPECT_EQ(entry.begin, 8 * elementBytes * ((i + rowOffset) * rootColumns + j + columnOffset));
  EXPECT_EQ(entry.end - entry.begin, 8 * elementBytes);
}
```

- [x] Assert actual SSA chain, contract preservation, true scalar fallback when required, materialized verifier, print/parse replay, byte-identical rerun. Add incompatible hard view binding test and assert pipeline failure leaves original IR unchanged.
- [x] Build `cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12`; run `build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='AttentionLayoutIntegrationTest.*:StorageRootContractTest.*'`. Expected new cases pass or identify a real generic defect. If defect, record failing output before minimal fix and repeat.
- [x] Run focused lit with `/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py -sv build-m6-release/test/Transforms/attention-layout-fixed-abi.mlir`; inspect commands and output. Report exact tests, evidence and changed paths in `/tmp/frisk-p0-task-1-report.md`. No commit.

### Task 2: O03 bounded trace recorder and cost contribution foundation — complete, independently approved

**Files:** Create `include/Dialect/Frisk/Analysis/LayoutTrace.h`, `lib/Dialect/Frisk/Analysis/LayoutTrace.cpp`, `unittests/Dialect/Frisk/Layout/LayoutTraceTest.cpp`; modify their CMake files. Create a small cost-contribution interface header if needed; modify `LayoutCostModel.h/.cpp`, `LayoutConversionChoice.h/.cpp`, `LayoutTarget.h`, SM90 `SM90CostModel.h/.cpp`, `SM90LayoutTarget.cpp` for optional actual-path observation. No pipeline/search behavior yet.

**Interfaces:** `LayoutTraceOptions` owns trace file and limits; independent `LayoutTrace` receives versioned JSON events and cost contributions, publishes atomic JSON. Interface must support lazy/off recording, event and serialized-byte caps, bounded final summary and omittedCost. Optional cost observer accepts `(kind, stableOwner, CostEstimate)` from existing charge sites; null observer is original evaluation. Lock exact signatures in the report before Task 3 consumes them.

- [x] Add recorder tests before implementation: default/invalid limits, whole-event truncation, large attributes, byte cap including JSON envelope, escaped strings, stable repeated serialization, arithmetic saturation, omitted contributions, atomic publication failure and identical output-path rejection.
- [x] Add cost tests comparing evaluation with/without observer and summing actual contributions, including Shared aliases and shared conversion/rematerialization leaders. Assertion pattern:

```cpp
CostEstimate total;
for (const auto &part : parts) total = addLayoutCosts(total, part.cost);
EXPECT_EQ(total.cost, evaluated.cost);
EXPECT_EQ(total.saturated, evaluated.saturated);
```

- [x] Observe expected missing-feature failures, implement minimal recorder and observer plumbing. Preserve original model values; no separate explanation model, no extra candidate or proof work. Contributions must be owner-stable and Shared root counting deterministic.
- [x] Run `LayoutTraceTest.*`, `LayoutCostModelTest.*`, `SM90CostModelTest.*`, `LayoutConversionChoiceTest.*` using current unit executable after rebuild. Record exact filter matches and results; self-review and hand off interfaces in `/tmp/frisk-p0-task-2-report.md`. No commit. Final 52/52 tests pass including Static/SM90Address suites; independent review approved after trace-off traversal and symlink-path fixes.

### Task 3: O03 connect trace to propagation, search and transactions — complete, independently approved

**Files:** Modify `LayoutSolver.h`, `LayoutCandidateSolver.cpp`, `LayoutPropagation.cpp`, `LayoutRelaxation.cpp`, relevant collection code, `LayoutStatistics.h/.cpp` if needed, `Transforms/Passes.td`, `Passes.h`, `PassPipelines.h/.cpp`, `LayoutInfer.cpp`; extend trace tests and add `test/Transforms/layout-trace.mlir` plus a Python JSON checker if useful.

**Interfaces:** Consume Task 2 recorder and cost observer. Trace lifetime belongs to one standalone pass/full pipeline transaction; use an explicit optional pointer through collection/attempts/solver. Existing callers retain defaults. Trace must survive failed attempts without retaining invalid MLIR pointers.

- [x] Before implementation add pass/pipeline tests expecting `trace-file`, `trace-max-events`, `trace-max-bytes` options; default empty, limits 4096 events/8388608 bytes, minimum 4096 bytes. Observe unknown-option or missing trace failure.
- [x] Record immutable candidate IDs, graph provenance, prepared/strict/common domain snapshots and support tables. Removed supports distinguish Disproven/Unknown/no support. Record MRV selection and per-depth counts, at most 4 retained and 4 discarded states, with partial/complete cost labels. Reuse existing checks instead of re-running proof work for diagnostics.
- [x] In final complete evaluation collect actual cost contributions (Task 2), selected assignments/contracts/conversion plans; publish after verification. Add trace file failure to staged transaction handling. Reject same stats/trace destination; do not assert IR committed before file handling succeeds. Empty trace option allocates no snapshots and does no explanation work.
- [x] Test standalone and full pipeline, success/failure/analysis-only, retry attempt isolation, bounded output, JSON parse and cost sums, path collision and rollback. Compare trace on/off exact final IR, selected assignments and search counters; repeat semantic JSON comparison excluding explicitly non-deterministic fields.
- [x] Run focused trace/statistics/relaxation tests and lit. Run actual Attention trace with default cap; assert full stage domains and final cost breakdown are not truncated. Record evidence in `/tmp/frisk-p0-task-3-report.md`. No commit. 59/59 focused units and 4/4 lit passed; after review fix, 26/26 covering units and 4/4 lit passed again. Mandatory solver options survive minimum-cap truncation; independent review approved.

### Task 4: O02 deterministic bounded neighborhood refinement — complete, independently approved

**Files:** Modify `LayoutSolver.h`, `LayoutCandidateSolver.cpp`, `LayoutStatistics.h/.cpp`, pass/pipeline option plumbing; extract focused refinement helper if needed. Extend `LayoutCandidateSolverTest.cpp`, `LayoutSolverPropertyTest.cpp`, `AttentionLayoutIntegrationTest.cpp`, trace/statistics tests and lit option checks.

**Interfaces:** Append defaulted solver options `bool refineBeamSolution=true; uint64_t maxRefinementProbes=256;`. Only complete beam components refine. Add separate refinement counters/stop reason; existing expanded includes MRV+refinement, existing MRV remains MRV-only. Observe refinement via Task 3 trace.

- [x] Add failing tests before implementation: beam misses a lower-cost one-variable neighbor; default enables refinement and false/zero disables it; exact unchanged; no remaining budget preserves incumbent; mid-scan budget retains best verified improvement; full conversion cost checked; equal cost no move; invalid/Unknown neighbors rejected; cost failure explicit; two-variable barrier remains local optimum; insertion permutation deterministic.
- [x] Implement best-improvement sweeps over canonical variable/candidate order. Each changed candidate consumes one local and one total probe before complete constraint checks. Cost uses existing full target plus conversion choice; accept strictly smaller vectors only, canonical key ties among improving neighbors. Stop on no improvement or limits; rebuild final contracts/edges. No candidate creation, no new relaxation, no fake exhaustive=true.
- [x] Add actual free-layout Attention comparison on/off, validate whole graph and IR. Expected disabled baseline memoryTransactions=608; enabled complete vector must be lexicographically no worse than `[8293,576,64,0,428,6144,6,0,0]`. Do not require same fixed-ABI costs.
- [x] Rebuild and run solver/property/trace/statistics and Attention filters plus affected lit. Record commands, RED/GREEN output, counters and cost vectors in `/tmp/frisk-p0-task-4-report.md`. No commit. 61/61 unit, final 53/53 covered subset and 6/6 lit pass; independent review approved. Actual refined vector `[8293,544,0,0,428,6144,6,0,0]`, 70 added probes within original budget.

### Task 5: Joint gates, repeatable report and documentation

**Files:** Update three long-term maintenance documents, P0 design/status and this checklist. Add reproducible P0 result report under `guoqiao/optimize_plan/`; add a benchmark/checking script only if it is necessary to rerun the acceptance data.

- [x] Rebuild Debug `FriskLayoutUnitTests frisk-opt` and all required CTest/Python targets; execute full unit suite (four explicit GTEST shards permitted), full lit, existing CTest and three Python normalization/API/tool checks. All exit codes and XML summaries must be checked. No GPU Attention claim. Post-fix final build and all gates exit 0: 450 unit (XML failures/errors/disabled 0), 64 lit, 7 Debug CTest, 1 Release generator CTest, 5 normalization tests plus API/tool checks.
- [x] In Release run at least 5 samples each of trace/refinement on/off for the same Attention input, sequentially with no competing benchmark. Report median/range, model cost, attempts, refinement and trace sizes; compare fixed ABI separately. Do not equate smaller model cost to measured GPU speed. All 48 calls (8 configurations × one warmup and five measurements) and semantic checks pass; existing multi-component benchmark also passes 30 samples/case.
- [x] Whole-source independent review against the approved design. Fix Critical/Important findings with regression tests and re-review. Resolve earlier task review uncertainties. Added and reviewed saturated-global non-worsening guard; source re-review approved with no remaining blockers.
- [x] Publish actual scope, commands and results; mark only completed items. Verify `git diff --check`, source/test/doc links and no build artifacts added. Leave changes uncommitted and report baseline commits separately. Results and repeatable commands: [implementation report](p0_implementation_report.md).
