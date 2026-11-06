# P1.2a–P1.2b Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 完成获批的 Copy 集成/诊断、可关闭开销观测及有界 MMA grid 联合选择，不扩展 GPU 执行域。

**Architecture:** Copy 构造返回可追踪证明结果；调用级 profile 只观察实际计算；MMA geometry 是第五个显式 Instruction 变量，数据布局与指令计划由同一支持表联合选择，实际 IR 独立复验。

**Tech Stack:** C++17、MLIR/TableGen、LLVM ADT/JSON、GTest/lit、Python 测量。

## Global Constraints

- 用户已确认阶段契约，见 [设计 §2–3](p1_2_to_p1_4_design_proposal.md)。直接源目录 main；保留 P0/P1.1；不创建 worktree、不自动 commit/push。
- exact=256、beam=32、总 probes=65536、邻域 probes=256；实际地址表 65536 点。每个 instruction role 至多 4 候选、完整 tuple 证明至多 256 次、支持 tuples 至多 256。
- 不覆盖显式布局、外部字节地址、对齐、线程数和已有 policy 硬契约；不把 Unknown 当作 Proven。
- 候选准备后冻结，传播只删；物化、独立实际 IR 重验、文本重放、确定性和事务回滚必须保持。
- 仅一个实现者/构建者修改共享源码；root 维护文档和最终验收。每任务先 RED/GREEN，再独立 spec+quality review。新测试须 Debug/Release 验证。
- 不改九项成本的公式或增加任意权重；MMA 项改为读取选中绑定。profile 不重复计算，不改变候选、成本、IR、搜索探测数。
- 本轮仅 P1.2a/b；P1.3/P1.4 不实施。无 GPU 性能或执行安全结论。

## Task 1: Copy 诊断与固定 ABI Copy→MMA 集成

**Files:** 修改 `include/Dialect/Frisk/Target/SM90/SM90CopyConstraints.h`、`lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`、`include/Dialect/Frisk/Analysis/LayoutTarget.h`、`lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`、`lib/Dialect/Frisk/Analysis/InstructionLayoutConstraints.cpp`，按需接入现有 LayoutTrace；扩展 `unittests/Dialect/Frisk/Layout/SM90CopyContractTest.cpp`，新增 `CopyMmaIntegrationTest.cpp` 并注册同目录 CMake；新增 `test/Transforms/copy-mma-contract.mlir` 和 errors fixture。

**Interfaces:** 保留 `FailureOr<Attribute> buildSM90CopyContract(...)`；新增下列受检接口。通用 target 的详细结果接口默认调用旧 builder，SM90 Copy override 使用详细构造结果，不能在 caller 重做一次证明。

```cpp
struct InstructionContractBuildResult {
  Attribute binding;
  LayoutProof proof;
  std::string reasonCode;
  std::optional<unsigned> operand;
};
InstructionContractBuildResult buildSM90CopyContractDetailed(
    const LayoutConstraintGraph &, const LayoutConstraint &, ArrayRef<Attribute>);
```

- [x] 在现有 Copy fixture 加入失败细分测试并运行 RED。核心断言如下，fixture 中分别只变更一个事实：

```cpp
EXPECT_EQ(result.proof.status, ProofStatus::Disproven);
EXPECT_EQ(result.reasonCode, "copy-alignment");
EXPECT_FALSE(result.binding);
// 对实际地址/枚举预算不足的输入断言 Unknown，而不是 Disproven。
```

- [x] 重构 builder 内各拒绝分支，直接携带实际下层 proof 与坐标、端点；稳定 code 区分 shape/方向/目标/对齐/地址连续性/writer/descriptor/预算/decoder。不强行把混合 proof 的 Unknown 转成 Disproven。旧 wrapper 只转换详细结果。

```text
validate role/type/target/alias facts
request actual footprint once; propagate its proof on failure
prove mandatory vector width; preserve source/destination proof
for permitted wider widths: use existing proof results; choose largest proven
validate selected path and TMA descriptor using actual addresses
return binding + Proven or structured reason at the rejection site
```

- [x] 接入支持表和有界 trace：拒绝事件含 path、status、reason_code、端点/坐标；合法候选与最终 selection 可区分。不把正常候选拒绝作为逐条用户错误；最终无支持时给有据原因；显式 binding 篡改仍验证失败。
- [x] 新增真实链：Global A/B `[64,16]f16` 行主序 alignment128 → Shared alignment1024 → SS MMA M=N64/K16、transB、128 threads → Global store。完整 Copy binding 由 helper 生成；分别固定 vector/cp.async/tma，另测自由选择。

```text
collect graph and construct complete checked Copy bindings
run production pipeline, verify materialized copy and mma contracts
enumerate all external logical coordinates: byte offsets unchanged
print/parse/rerun: semantic bindings/layout unchanged
mutate just issuer / completion / descriptor: fail and original IR unchanged
```

- [x] 补齐单条件负例：alignment1、起点偏移一个 f16、同线程非连续、TMA 外层 stride、重复 writer、错误目标/方向；分别检查合法 fallback 或无 fallback 的原子失败。guarded async 支持域不扩大。
- [x] Release 构建 `cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12`；运行 unit filters `SM90CopyContractTest.*:CopyMmaIntegrationTest.*:GuardedCopyTest.*` 及新 lit。按实际测试 suite 名校正命令，报告实际命令/RED/GREEN/警告。
- [x] 独立 spec+quality review 后记录接口、稳定 reason codes、源码清单和证据；不提交。

## Task 2: 单次推断热点与缓存观测

**Files:** 新增 `include/Dialect/Frisk/Analysis/LayoutProfile.h`、`lib/Dialect/Frisk/Analysis/LayoutProfile.cpp`、`unittests/Dialect/Frisk/Layout/LayoutProfileTest.cpp`，注册 Analysis/unit CMake；修改 `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`、`include/Dialect/Frisk/Transforms/Passes.{h,td}`、Transforms 的 `PassPipelines.{h,cpp}`；在 Analysis 的 `LayoutAliasAnalysis.cpp`、`ExecutionLayoutProof.cpp`、`LayoutCostModel.cpp`、`LayoutConversionChoice.cpp`、`LayoutCandidateSolver.cpp`、`InstructionLayoutConstraints.cpp`、`LayoutRelations.cpp` 的真实调用/cache 点接入；新增 `test/Transforms/layout-profile.mlir` 与 Python JSON checker。

**Interfaces:** 独立 `LayoutProfile` 汇总固定种类的 count/inclusive_ns/hits/misses/entries，RAII scope 只在 enabled 时读取时钟。pass/pipeline 新 `profile-file` 默认为空。生命周期必须限定在一次推断调用并支持嵌套/多线程隔离；不引入数据缓存。可用受限 thread-local 当前 observer + RAII 恢复，不捕获跨调用 IR 指针。发布沿用 stats/trace 的临时文件原子替换和路径冲突检查语义。

```cpp
enum class LayoutProfileKind {
  AddressEnumeration, ExecutionEnumeration, AccessCost,
  ConversionClassification, ConversionChoice, InstructionSupport,
  InstructionContract, RelationQuery, RelationCache, AliasPairCache,
  FootprintCache, Count
};
// public observation payload is finite; no per-coordinate/event log.
```

- [x] 先新增开关、JSON、嵌套 scope 和缓存计数测试，观察缺少 profile 行为的 RED；测试不得依赖毫秒下限，时间只要求非负。
- [x] 实现固定 schema `version/status/timing_kind=inclusive/metrics`；注明包含关系，禁止把嵌套时间求和冒充总耗时；区分入口调用和缓存 miss 计算。

```text
if profile disabled: no clock, no event construction, no extra proof
on measured actual function entry: increment calls, start optional timer
on real cache lookup: increment hit or miss
on real insertion: count entry; no synthetic query
on exit: add elapsed duration, restore nested observer on pipeline exit
```

- [x] 真实记录 address/execution enumeration、access cost、conversion classification/choice、联合关系查询；relation/footprint cache 实际 hit/miss/entry。错误路径和第二次调用不能串统计。
- [x] 开/关两份 IR、prepared domains、solution、完整九项成本和探测数相等；trace 存在时逐字节一致。检查 profile 与 stats/trace 路径别名、无法写文件、失败回滚；原有 pass/pipeline 默认行为不变。
- [x] Release focused unit/lit 通过并独立复审。串行测量归 Task 4：root 最后串行运行两类 Attention 和小图的 off/on（1 warmup + 5 samples），保存原始计数/时间/RSS/身份，报告开销而不承诺加速。不新增无证据缓存。

## Task 3: MMA Auto、显式 geometry 与有界联合选择

**Files:** 修改 `include/Dialect/Frisk/IR/FriskEnums.td`、`lib/Dialect/Frisk/IR/FriskOps.cpp`/`FriskLayoutOps.cpp` 的 verifier、Target/SM90 下的 `SM90MmaLayoutProof.{h,cpp}`、`SM90GemmConstraints.{h,cpp}`、`SM90LayoutTarget.cpp`、`SM90CostModel.cpp`；Analysis 下的 `MmaLayoutConstraints.cpp`、`LayoutConstraint.{h,cpp}`、`InstructionLayoutConstraints.cpp`、统计与 trace；`lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`。新增独立 `unittests/Dialect/Frisk/Layout/SM90MmaGeometryTest.cpp`、`MmaAutoIntegrationTest.cpp` 与 `test/Transforms/mma-auto-layout.mlir`/errors，更新受影响图结构期望。

**Interfaces:** 保留 legacy `getSM90MmaGeometry`；增加 `FailureOr<SmallVector<SM90MmaGeometry>> getSM90MmaGeometries(MmaOp, int64_t)`。geometry 候选使用专用受检 dictionary（kind/version/shape/threads/grid/atom/repeats），集中 encode/decode helpers；最终完整 `MmaContractAttr` 仍是物化真值。数据角色 0–3 不变，角色 4 是 Instruction-kind、Mma anchor、无 use/转换边。受检 decode 必须验证 shape/threads/policy/支持域，不能只读 grid。

- [x] 验证新 Auto policy 可解析，旧 gemm 拒绝；128×128/K16/256threads 产生两种合法 geometry；默认 Square 保持原值。新增 enum 后显式防止旧 gemm switch 落入 Auto。初次 parser RED 使用错误语法，属于测试搭建错误；后补正确语法的旧二进制对照，不追认为实现前 RED，详见报告。

```text
groups = threads/128
legacy: preserve original policy selection exactly
Auto: enumerate gM*gN=groups in stable order
require M divisible by 64*gM; select largest supported atomN dividing N/gN
derive repeats; keep only current dtype/shape/thread supported geometries
cap grid count at four; no all-atom expansion or thread search
```

- [x] 更新完整路径：collector 新 geometry 变量 → 准备几何/fragment/storage → 约束图 verifier 检查第五角色 → geometry 条件化 tuple 枚举 → assignment/support → 物化 → 独立 IR verifier。旧 policy geometry 单例，Copy 四角色不变。
- [x] 先按实际 fragment 坐标证明筛选 compatible role candidates，再枚举同 geometry 的 storage；不能按 attribute 模板相等排除合法 reordered carrier seed。预检查/完整证明分别计数，失败的完整证明也占 256 预算；超预算返回明确 Unknown/Unsupported，不保留部分表冒充完备无解、不静默丢 hard seeds。

```text
for geometry in frozen geometry domain:
  prove A(if RS), init, result fragments against selected geometry
  visit compatible role combinations; reject inconsistent repeated vars
  if completeProofAttempts == 256 and another tuple needs proof: Unknown
  ++completeProofAttempts
  build/verify binding with selected geometry; append only Proven tuple
```

- [x] 实际 contract verifier 从 binding 解码并证明合法几何；Auto 允许合法集合，legacy 保持 policy 一致。显式 mma_contract 固定 geometry/repeats/packing/descriptor，重跑不可覆盖。
- [x] MMA CostVector 第一项从选中已验证 contract 计算；其他成本公式不变。核心 SS 例 `(1,2)` atom64/repeats(2,1,1) 项为4；`(2,1)` atom128/repeats(1,1,1) 项为2，solver 与同域精确 oracle 一致。
- [x] 正/反例：SS/RS，128/256/512/1024threads 合法子域，错误 grid乘积/repeats/packing/descriptor/policy/threads、carrier 重排硬 seed、显式非默认绑定、错误第五角色、候选与 tuple 预算、text replay/rerun/rollback。含四个 grid 的合法域与硬 seed：已知 RequireEncoding 的槽位不应加入不可能通过其硬约束的生成模板，导致人为第五候选溢出；不得删除该 hard seed。MMA→Reduce(dim0/1)→store、固定 consumer encoding/转换与精确 oracle 比较。
- [x] legacy 最终语义和完整成本保持；允许新 singleton geometry 导致图 ID/count/trace 改变。报告每项差异，不要求旧 trace 字节相同。未找到 consumer 驱动严格反转则如实列剩余项，不改权重制造反转。
- [x] Release focused unit/lit 通过，独立 spec+quality review；报告新 graph/tuple APIs 与精确预算统计。不提交。

## Task 4: 完整回归、串行测量与维护

**Files:** 新增 `guoqiao/optimize_plan/p1_2_implementation_report.md`、`p1_2_measurement_results.json`；更新当前计划、阶段设计与长期维护文档。必要的复现驱动放 `benchmark/layout/proof/`，测试放 `test/python/`。

- [x] root 保存修改前 source archive、Release binary/config 至 `/tmp/frisk-p12-baseline-vZeQIU/`；HEAD `b237911`，完整 dirty baseline 包含 P0/P1.1。二进制 SHA256 `ad69cb595ee62f689a03819f2e5f4863a98560d0b4c3a5d3a88e26c075659853`。
- [x] 所有修改及独立 scoped review 结束后冻结源码，构建 Debug/Release；禁止链接与使用同一个 frisk-opt 的 lit 同时运行。

```bash
cmake --build build --target all FriskLayoutUnitTests --parallel 12
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt frisk_layout_proof_bench --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py -sv -j4 build/test
ctest --test-dir build --output-on-failure
ctest --test-dir build-m6-release -R 'FriskLayoutProofBenchmarkTest|FriskLayoutConversionGeneratorTest' --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/sm90_layout_tool_test.py --build-dir build
```

- [x] 新增测试 Release 与 Debug 均通过；全套 Debug 最终使用16分片，505 个唯一用例全部通过，无跳过。69 lit、8 Debug CTest、2 个选定 Release CTest、5 Python normalization 与 API/tool 守门通过；Release 60 项受影响测试及测试同步后的 3 项通过（重叠计数不相加）。
- [x] 独立全量增量源码审查通过；显式预算状态问题及测试期望同步已复审关闭。review diff 相对 dirty baseline，不把已有 P0/P1.1 当作本轮新增；最终测试和测量仍按独立门槛验收。
- [x] 所有 jobs 结束后串行测量：54 次 CLI 调用（各组1warmup+5samples）和9次 trace 调用完成，两类 Attention/小图的新旧 IR/完整成本相同，profile 开关 IR/stats/trace 等价。保存命令、源码/二进制哈希、RSS、原始时间、median/range 与驱动。只测 CPU，不跑 GPU。
- [x] 文档同步实际完成子集/未完成目标/失败及修复；最终证据/文档独立复审 APPROVED，无遗留发现项，`git diff --check` 通过，保留未提交状态。原始数据见 `p1_2_measurement_results.json`。
