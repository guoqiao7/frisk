# P1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 用受限符号证明减少 ownership/Broadcast 枚举，保持地址证明安全并量化效果。

**Architecture:** 公用受检 bit-linear 硬件矩阵表示；ownership 基于秩/核，Broadcast 基于同线程像空间包含。枚举接口保留为生产地址表及独立测试 oracle。

**Tech Stack:** C++17、LLVM APInt/GF2Matrix、MLIR、GTest、lit、Python 测量工具。

## Global Constraints

- 直接在源目录 main 工作；保留 P0 未提交改动；不创建 worktree、不自动 commit/push。
- 不改 CostVector、exact=256、beam=32、总探测预算=65536、邻域预算=256 或外部 ABI。
- `enumerateExecutionLayout` 保留真实表及 65536 点上限；符号 Proven 不解除 vector/storage/Reduce/Copy/MMA 的枚举/地址限制。
- 不扩展 Broadcast 形状支持域，不新增 GPU lowering 或全局长期缓存；Unknown 不视为 Proven。
- 每个任务先 RED 再 GREEN；源文件用 apply_patch；仅一个实现者/构建者同时修改共享目录。独立 reviewer 不重复跑完整测试。

## Task 1: 基线与 ownership 符号证明

**Files:** 新增 `include/Dialect/Frisk/Analysis/LinearExecutionLayout.h`、`lib/Dialect/Frisk/Analysis/LinearExecutionLayout.cpp`；修改 `ExecutionLayoutProof.{h,cpp}`、Analysis/CMakeLists.txt、ExecutionLayoutProofTest.cpp；可新增独立 SymbolicExecutionLayoutTest.cpp 并注册 unit CMake。

**Interfaces:** 公共 `proveExecutionOwnership(DistributedEncodingAttr, RankedTensorType, StringRef)` 签名不变；给 Task 2 提供受检完整 hardware/按 carrier 分块矩阵的内部分析接口，置 LinearExecutionLayout.h，实际接口写入任务报告。复用 GF2Matrix，避免复制矩阵解析。

- [x] Root 在修改前保存基线二进制、源码快照，运行两个 Attention fixture 各 1 warmup+5 samples；记录现有计时/RSS。结果见实施报告 §1–2，12 次调用均成功。
- [x] 实现者先修改现有超界测试，加入以下行为断言并运行，看到旧 ownership 为 Unknown 的预期 RED：

```cpp
auto large = encoding({"register"}, {2}, {4,32,1024,1,1}, {1,2}, 2, 32768);
EXPECT_EQ(proveExecutionOwnership(large, type(), "first_owner").status, ProofStatus::Proven);
ExecutionEnumeration points;
EXPECT_EQ(enumerateExecutionLayout(large, type(), points).status, ProofStatus::Unknown);
EXPECT_EQ(proveExecutionOwnership(large, type(), "all").status, ProofStatus::Disproven);
```

- [x] 按设计 §3.1 实现受检 canonical map、秩/复制/核证明和确定性坐标反例；非适用子类保守处理。示意判据（须处理空矩阵和溢出）：

```text
validate attributes, hardware bits and logical bits
M = zero-extend omitted carriers and reorder supplied columns
if rank(M) != logicalBits: Disproven + uncovered coordinate
if replication != 2^(hardwareBits-logicalBits): Disproven
if policy == all && rank(M) != hardwareBits: Disproven + nonzero kernel point
else Proven (symbolic ownership)
```

- [x] 加入小域矩阵族与枚举 oracle 差分、carrier 省略/重排、零位域、畸形属性、复制错误、反例求值、现有 vector 限制测试。
- [x] 运行 `cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12`；运行 `build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='ExecutionLayoutProofTest.*:SymbolicExecutionLayoutTest.*'`。记录实际结果与警告，不修改无关代码清理旧警告。
- [x] 独立 spec+quality review 通过，报告保留 RED/GREEN 命令/输出与供 Task 2 使用的精确接口，不提交。

## Task 2: Broadcast 同线程符号证明

**Files:** 修改 `lib/Dialect/Frisk/Analysis/BroadcastLayoutConstraints.cpp`、`unittests/Dialect/Frisk/Layout/BroadcastLayoutTest.cpp`；按需新建 SymbolicBroadcastLayoutTest.cpp 并注册 unit CMake；复用 Task 1 的 LinearExecutionLayout 接口。

**Interfaces:** `proveBroadcastLayout` 与 `projectBroadcastInputEncoding` 公共签名不变。输入维度、轴、dtype、CTA 与线程规则不变；用 Task 1 的受检矩阵，失败返回 LayoutProof 的确定性实际坐标。

- [x] 先写超 65536 持有者合法广播测试并运行，观察旧证明的 Unknown；再加入同线程不可取反例。
- [x] 以如下判据替换全持有者枚举；源与目标 ownership 先通过，按真实 source logical bit 顺序删去目标广播轴：

```text
R = source register columns
D = [projected destination register columns,
     projected destination thread columns XOR source thread columns]
same-thread available iff rank([R,D]) == rank(R)
```

- [x] 失败时用不属于 image(R) 的 D 列构造目标硬件基向量，返回目标逻辑坐标并独立验证该线程无源 register 可取。投影入口仅将“是否覆盖”检查换为 ownership；不改 reduction 投影算法支持域。
- [x] 小域独立 oracle 对每个目标持有者删除广播轴并穷举同线程源 register；覆盖正确/错误轴、register 基重排/XOR/复制、lane/warp 匹配与不匹配、畸形编码，以及两个超界路径状态差别。
- [x] 构建 Release，运行 `--gtest_filter='BroadcastLayoutTest.*:SymbolicBroadcastLayoutTest.*:ExecutionLayoutProofTest.*:SymbolicExecutionLayoutTest.*:AttentionLayoutIntegrationTest.*'`，lit 运行 attention-layout、attention-layout-fixed-abi、attention-layout-trace 与 broadcast 相关 fixture（以 rg --files 查实际名称）。
- [x] 独立 spec+quality review 通过，报告测试结果与支持域变化，不提交。

## Task 3: 可重复测量、完整验收与维护文档

**Files:** 新增 `benchmark/layout/proof/CMakeLists.txt`、`benchmark/layout/proof/layout_proof_bench.cpp`、`test/python/layout_proof_benchmark_test.py`；修改 `benchmark/layout/CMakeLists.txt` 接入子目录及 CTest。Root 更新 `guoqiao/optimize_plan/p1_1_implementation_report.md`、长期维护文档和本计划复选框，并在 `p1_1_measurement_results.json` 保存原始测量证据。

**Interfaces:** 使用生产 ownership/Broadcast API 与独立枚举 oracle；不向生产类添加测试专用开关。微基准 JSON 记录案例身份、点数、状态、迭代数、ns 和 sink 验证，不能优化掉实际调用。

目标名 `frisk_layout_proof_bench`，固定小域/65536 边界/131072 超界 ownership 与 Broadcast 六个主案例，并含广播同线程不可取负例。每例双方先验证预期状态，再记录重复测量；JSON 区分两种方法的结果，超界 Unknown/Proven 不计算等价加速比。测试使用实际可执行程序，核对案例/状态、非负时间/样本数、重复非计时元数据一致；不设置墙钟时间阈值。基准增加构建类型/编译器/二进制身份，配置和运行命令写入报告。Root 的全量验收及串行性能采样不是实现者的任务；实现者仅构建 Release 微基准并运行其 focused CTest/Python 验收，避免重复全套测试。

- [x] 先写可运行微基准验收（预期不存在目标/结果而 RED），实现固定小域/边界/超界的 ownership 和 Broadcast 对照；从上界枚举 Unknown 与符号 Proven 区分能力，不将它们计作等价耗时加速。7 项 Python/1 项 focused CTest 通过，构建配置及 Ninja 可用性问题修正并独立复审通过。
- [x] Root 运行 fresh Debug build、完整 GTest（可 4 分片）、完整 lit、CTest、生产 API 边界/Python 检查。修正 Debug 负例构造后 465/465 unit、64/64 lit、8/8 Debug CTest、2/2 Release CTest、5/5 Python normalization 与 API/tool 守门通过。
- [x] 所有构建/测试结束后串行运行 Release 微基准和两个 Attention 各 1 warmup+5 samples。记录原始样本/median/range/RSS、阶段计时与相同输入输出语义；结果及原始数据见实施报告 §7–8。
- [x] 检查候选变化、最终成本/布局/探测数、trace 稳定/成本一致、实际 verifier 与 rerun。两个 Attention 的前后 IR、非计时 stats 及完整 trace 一致；不推断所有其他输入的候选不变。
- [x] 全量源码独立复审；重要问题修复后重跑覆盖测试及必要全量守门。Debug 负例构造问题已修正、复审关闭并完成全量回归；报告区分 CPU 证明收益、整体时间持平及未测 GPU。
- [x] 更新设计状态、执行计划、实施计划、策略对比、优化清单；git diff --check 通过，保留未提交状态，不推送。
