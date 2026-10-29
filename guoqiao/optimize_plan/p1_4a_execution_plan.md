# P1.4a Minimum Shape Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute task-by-task with independent task and final review.

**Goal:** Support static positive power-of-two tensor shapes including unit axes through ownership, inserted-axis Broadcast, rank>=2 to rank>=1 Reduce and necessary static views.

**Architecture:** Keep bit-linear maps: extent one contributes zero logical bits. Repair zero-row/zero-column boundary handling in consumers while preserving proof status and actual materialized verification. Reuse existing transpose and static tensor expand/collapse paths where already correct; add narrowly scoped support where real tests expose a gap.

**Tech Stack:** C++17, MLIR/LLVM, GoogleTest, lit, Python remote evidence.

## Global Constraints

- Only /home/baopeihua/frisk on H100 is edited. Preserve existing dirty source. No local copies, worktrees, staging, commits, pushes, GPU runs.
- Static strictly positive power-of-two extents include 1. Rank0, dynamic, zero extents and ragged extents remain unsupported. Reduce source rank>=2 and result rank>=1. Broadcast inserts sorted distinct axes and preserves remaining dimensions; no scalar/rank0 Broadcast.
- Unknown remains Unknown; do not replace unproved layouts with successful proofs. Keep ABI, targets, search budgets and cost model defaults unchanged.
- At most one source/build owner. All review is read-only except review report files. Capture real dirty baseline, RED/GREEN, both build configurations for new tests, one full final Release suite and related lit/CTest. No GPU performance claims.
- Existing symbolic bit/enumeration budgets remain unchanged. Empty logical bits must not enter GF2 APIs requiring nonempty dimensions; ownership/replication and same-thread semantics must still be checked.

## Task 1: Integrated minimum shape closure

**Files:** Primary production changes: lib/Dialect/Frisk/Analysis/BroadcastLayoutConstraints.cpp, ReductionLayoutProof.cpp; inspect necessary zero-bit handling in ExecutionLayoutProof.cpp, LinearExecutionLayout.cpp, LayoutRelations.cpp, LayoutConstraint.cpp, SM90LayoutTarget.cpp, FriskOps.cpp and bit-map verifier. Change only files whose failures require it. Tests: SymbolicBroadcastLayoutTest.cpp, ReduceLayoutIntegrationTest.cpp, SymbolicExecutionLayoutTest.cpp, LayoutPipelineTest.cpp and focused view tests or new UnitAxisShapeTest.cpp registered in Layout/CMakeLists.txt. New lit fixture test/Transforms/layout-unit-axis.mlir if CLI coverage needs one.

**Interfaces:** Preserve public signatures validBroadcastTypes, projectBroadcastInputEncoding, proveBroadcastLayout, projectReductionEncoding, proveReductionLayout and materialized verifier. Extent 1 maps to width 0. Necessary views are static tensor expand_shape/collapse_shape and transpose preserving element count and existing positional map semantics, without arbitrary reshape generalization.

- [x] Add meaningful failing regression tests before production edits. Example behavioral assertions: validBroadcastTypes(tensor<1xf32>,tensor<1x8xf32>,{1}) is true; actual-coordinate oracle agrees with proof; Reduce tensor<1x8xf32> over dim1 produces tensor<1xf32>; Reduce tensor<8x1xf32> over dim1 preserves 8 coordinates; unit-only tensor<1x1xf32> reduces to tensor<1xf32>. Include inserted unit axes, retained unit axes and all-unit input/output, rank3 multi-axis Broadcast, zero hardware register bits and replication; preserve negative rank0/dynamic/zero/ragged/axes/topology tests.
- [x] Build only tests first and record expected assertion RED, including integration failure before implementation. Commands use /home/baopeihua/anaconda3/envs/torch/bin/cmake --build build-m6-release --target FriskLayoutUnitTests -j 2 and the test binary with exact filters/XML output.
- [x] Change invalid extent check `e <= 1` to `e <= 0` in supported shape validators only after RED. Guard zero-dimensional linear algebra explicitly: validated ownership and equal matching threads imply same-thread Broadcast coverage when no retained logical bits. Projection must preserve a legal encoding and correct replication for unit-only result. Derive actual code from local APIs; do not force APInt(0) or call GF2Matrix::get(0,...).
- [x] Add integration tests for infer/materialize/actual-only verifier/text replay/idempotence for unit-axis Broadcast/Reduce and necessary static views; assert real encodings/contracts, preserve failed transactional IR. Test wrong replication or same-thread coverage fails, and near-neighbor unsupported shapes reject.
- [x] Run new and affected suites in Debug and Release; relevant lit in both; capture exact commands, return codes, XML and SHA256 source manifest. Write /tmp/frisk-p14a-20261029-57dtdpio/task-1-report.md with RED/GREEN, coverage, changed files and concerns. Close all jobs. Independent task review required before Task2.

## Task 2: Final regression and durable acceptance

- [x] Controller runs full final Release unit suite, selection suite, relevant Release lit and CTest guards; count actual XML/list outcomes, no invented totals. Reuse focused Debug evidence.
- [x] Save actual baseline-relative unified diff and source identities, raw logs/XML and task review under guoqiao/optimize_plan/p1_4a_artifacts. Write guoqiao/inference_detail/p1_4a_implementation_report.md with supported boundaries and exact evidence.
- [x] Strongest available independent whole-change reviewer evaluates final source, tests, report, evidence and open task findings. Fix any blocking findings with covering RED/GREEN and re-review.
- [x] Update ledger with accepted scope, jobs closed and next phase. Keep workspace as-is as explicitly requested; no integration menu or commit workflow.
