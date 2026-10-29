> 2026-10-29 P1.4a：最小形状支持已完整验收，功能回归、独立源码及最终证据/文档复审均通过。静态正二次幂单位轴、插入轴Broadcast、rank≥2到rank≥1 Reduce及静态expand/collapse/transpose闭环；575 Release主单测、20 selection，双配置最终15聚焦/5lit及辅助守门通过。GPU基础后端及后续路线未完成。见[实施报告](inference_detail/p1_4a_implementation_report.md)。

# Frisk Layout Inference System Implementation Plan

> 2026-10-29：P1.3c已完成验收，完整源码、驱动、数据与文档独立复审通过。默认legacy保留；显式`boundary-frequency=static-trip-count`统一转换/clone与受限外提的频率口径。Release主单测570、选择质量20，相关Debug/lit及辅助守门通过；18次语义核验与36次串行CPU调用通过，身份复验不变。下一独立任务为最小形状支持，GPU和跨框架比较尚未实施。见[阶段报告](inference_detail/p1_3c_implementation_report.md)。

> 2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。 见[完整Attention资源报告](inference_detail/p1_3b_attention_resource_report.md)。

## 选择质量实验 → P1.3（2026-10-26，Q1/Q2已验收，继续P1.3）

用户指定的Auto/固定策略对照及消费者联合选择实验已完成，见[报告](inference_detail/p1_2_selection_quality_report.md)。按[实验设计](optimize_plan/p1_2_selection_quality_design.md)执行原生四policy＋共同冻结域geometry限制的双层对照：64原生配置、36消费者图，独立有界oracle区分局部打平、严格反转与搜索差异。保留原始数据后继续P1.3a→b→c，不改写本次实验基线。

- [x] 核对现有代码、历史证据与 P1.3 契约，写入实验设计并自审。
- [x] 用户确认方案 A 书面契约，编写[执行计划](optimize_plan/p1_2_selection_quality_execution_plan.md)，保存脏源码/工具基线。
- [x] 实施驱动/测试/实验，形成 `inference_detail/p1_2_selection_quality_report.md`，独立最终复审通过。
- [x] 实验报告验收后，依次细化并实施 P1.3a、P1.3b、P1.3c；不同时改变候选与成本模型来影响实验基线。

按任务实施和独立复审后逐项验收；以下Task1/2描述保留阶段检查点，最终验收以Task4为准。不覆盖下方P1.2历史状态，不自动commit/push。

阶段进度：Task 1 原生实验驱动已完成并经独立复审，Release 9项、Debug原7项＋修复定向3项通过；64项输入生成已验证，但不等于64项推断实验已全部执行。C1 SS/RS 的实际ABI/重放与跨构建一致性样例通过。固定消费者原始Convert输入的验证入口限制已记入实验设计§5.1，该组用受控求解/物化后完整流水线验收，不改变生产验证器。

Task 2 已完成并独立复审通过：冻结域/geometry限制、独立有界oracle、完整成本对账、原生/受控域身份匹配与oracle方案独立副本物化重验。Release 20项、Debug定向8项通过，13个持久化CLI样例中12个请求的oracle全部完成，且实际走过独立副本验证。

Task 3 批量驱动/分类/报告工具已完成，25项Python测试及真实CLI集成通过；独立复审关闭核心验收门槛、同geometry分类、报告重载身份校验三项问题。Task 4 正式230行矩阵、Debug/Release新增各20项、原MMA/流水线24项和48次串行计时已完成，严格重载/合并及独立最终复审通过。165个完整oracle与默认最优成本一致，1个C6 RS joint超时单列；消费者3个严格反转、6个平局消除、27个未改善。数据/源码/IR/日志归档见[报告](inference_detail/p1_2_selection_quality_report.md)，现在推进P1.3，不将Q1/Q2完成等同于整个P1完成。

P1.3a已完成模型身份/覆盖表和实际优化IR算术诊断，实现任务、整体源码及最终证据/文档复审均通过；完成524个不同Debug单测的分阶段覆盖、70 lit及最终修复定向验收，并保存12次语义核验＋36次串行观测。两类Attention的Mul/Sub/Exp/Div/Trunc各4096 element-op，旧基线/新默认/诊断开启的IR、完整成本与非计时求解统计保持一致。详见[报告](inference_detail/p1_3a_implementation_report.md)及[执行计划](optimize_plan/p1_3a_execution_plan.md)；P1.3b进展见下方阶段记录，P1.3c现已完成实现、功能回归、正式观测及最终独立复审。

## P1.2–P1.4 契约（2026-10-24；P1.2a/b 已完成）

用户确认的方案 A 已完成 P1.2a→P1.2b。审计、源码依据、实例和验收标准见[阶段设计](optimize_plan/p1_2_to_p1_4_design_proposal.md)，任务见 [执行计划](optimize_plan/p1_2_execution_plan.md)，实际结果见 [实施报告](optimize_plan/p1_2_implementation_report.md)。以下未勾选项尚未实施，不将 O04–O11 的所有长期目标标为完成。

- [x] 审计现有候选/指令证明、Copy、shape、成本/转换及可执行后端；核实 P1.1 后续编号此前未定义。审计阶段未重新构建或运行测试，后续实施证据单独登记。
- [x] 建立 P1.2–P1.4 与 O04–O11 的映射、拟议契约、验收案例及剩余项，登记长期文档。
- [x] 用户确认具体契约，编写 [P1.2 执行计划](optimize_plan/p1_2_execution_plan.md)；本轮仅实施 P1.2a→P1.2b。
- [x] P1.2a：固定外部 ABI 的 vector/cp.async/TMA Copy→MMA 集成、细分证明状态/原因/坐标、默认关闭且调用隔离的热点/缓存观测。
- [x] P1.2b：显式 Auto、第五 Instruction geometry 角色、有界 grid 联合选择及完整成本 oracle；旧 policy/显式绑定保持，成本读取实际选中绑定；预算不足保留 Unknown。
- [x] P1.3a：模型身份/覆盖与实际优化IR逐opcode算术计数；最终证据/文档复审通过。
- [x] P1.3b：实际 IR 存活资源诊断的实现、回归及实验已完成；最终独立证据/文档复审已通过，全局峰值不写入可加成本目标。
- P1.3b [具体执行契约](optimize_plan/p1_3b_execution_plan.md)的纯分析器、通用pipeline集成及规范ranked-Tensor Broadcast原子SSA窄特例已实现并经独立源码复审。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过；15次语义核验先于36次串行CPU观测（6次预热、30次采样）。两类完整Attention均available，各16个Tensor SSA、3个Shared根，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节；旧/新与开关对照的IR、完整成本及非计时求解统计不变，旧arith诊断不变，最终IR重放核验IR与诊断等价。模型仍为 `sm90-layout-lex9-legacy-v1`，不推断物理寄存器/occupancy；P1整体及O06全目标尚未完成。最终独立证据/文档复审已通过，见[阶段报告](inference_detail/p1_3b_implementation_report.md)与[完整Attention资源报告](inference_detail/p1_3b_attention_resource_report.md)。
- [x] P1.3c：显式开启的静态 for 次数模型，保留legacy默认和原通信/外提合法性边界；完整回归、正式观测与独立最终复审通过。
- [x] P1.4a：静态非零 rank 的单位轴、广播/Reduce/投影及完整链；不扩张成通用 ragged/dynamic 支持。
- [ ] P1.4b：消费实际物化契约的固定 shape/ABI Attention GPU lowering、数值和安全验收，保留弱对齐语义。
- [ ] P1.4c/d：cp.async 与 TMA 独立执行验收、同算法布局性能对照及后续成本校准依据。

### P1.3b历史检查点（2026-10-27，已由2026-10-28实现与实验结果取代）

P1.3b [具体执行契约](optimize_plan/p1_3b_execution_plan.md)的Task1纯分析器及Task2通用pipeline集成已通过独立复审：真实SSA峰值、Shared根去重/声明与活跃容量分离、capture和保守异步生命周期检查；诊断开关不改变布局/成本。Debug/Release最终定向各87项、相关lit各4项、全Release545项及相关Python/工具回归通过。两类Attention探针推断成功，IR与P1.3a相同，但第20、25个Broadcast令资源报告unavailable。named Broadcast隐式region窄特例仍待确认，完整资源/计时实验及整体最终复审未完成。见[阶段报告](inference_detail/p1_3b_implementation_report.md)。

以下为P1.2历史验收记录：

最终验收：505/505 Debug unit、69/69 lit、60 项 Release 受影响测试（以及测试同步后的 3 项）、8 Debug/2 选定 Release CTest、5 Python normalization 与 API/tool 守门通过，独立源码复审通过。54 次串行调用与 9 次 trace 调用保持 legacy 最终 IR/九项成本；profile 开关保持 IR、非计时 stats 和完整 trace。两类 Attention 的 profile 中位开销约 0.15%/0.10%；小图约 1.33%，只作为本机 CPU 观测，不声称 GPU 加速。Auto 消费者案例未展示赢家反转，全 atom/tile/线程搜索仍不在本轮范围。

直接在源目录 main 工作，保留未提交 P0/P1.1，不自动提交或推送。完整 Attention GPU 后端是新增实质工作，不能以已有 conversion 微基准、推断 IR 或手写替代 kernel 宣称完成。

## P1.1 增量（2026-10-21，已完成实现与验收）

采用方案 A 分阶段推进 P1，本轮仅实施 O07 的 ownership/Broadcast 受限符号证明与编译开销验收。具体边界见 [P1.1 设计](optimize_plan/p1_1_implementation_design.md)，任务见 [执行计划](optimize_plan/p1_1_execution_plan.md)。保留 P0 未提交成果，不自动提交或推送。

- [x] 保存实际 P0 工作区基线，测量阶段时间与 RSS；两个 Attention 各 1 次预热/5 次测量成功，详见 [P1.1 实施报告](optimize_plan/p1_1_implementation_report.md)。
- [x] ownership 秩/核证明及枚举 oracle 差分；16 项局部与当时 457 项 Release 全量 unit 通过，独立 spec/quality 复审通过，后纳入下述最终 Debug 回归。
- [x] Broadcast 同线程像空间包含证明及超界正反例；32,768 组小矩阵差分、40 项局部/Attention unit、4 项 lit 和独立复审通过；具体地址表仍受枚举上限约束。
- [x] 完整回归、独立复审、微基准与 Attention 对照、维护文档同步。修正 Debug 负例构造后 465/465 unit、64/64 lit、8/8 Debug CTest、2/2 Release CTest、5/5 Python normalization 与 API/tool 守门通过；7 案例微基准及两类 Attention 各 1 warmup/5 samples 完成。局部证明降低开销，但 Attention 时间基本持平（中位数约下降 0.21%/0.17%），IR/非计时 stats/完整 trace 一致；原始测量随报告保存，不宣称 GPU 加速。

地址表枚举仍保持 65536 点上限；不扩展形状、GPU lowering 或 CostVector。P1.2–P1.4 尚未实施。

## P0 增量（2026-10-14，已完成实现与验收）

实现前基线：`219c0cf`，设计提交 `b237911`，本地 main，未推送。具体边界见 [P0 设计](optimize_plan/p0_implementation_design.md)，按 O01→O03→O02 完成；实际命令、样本和验收见 [实施报告](optimize_plan/p0_implementation_report.md)。以下增量不改变既有里程碑的历史证据。

- [x] P0-1：固定外部行主序 Attention；验证对齐、静态偏移、硬冲突、地址与数据流、重放/回滚。仅补测试，无生产修改；16 项局部 unit、1 项新增 lit 及独立复审通过，已纳入最终全量回归。
- [x] P0-2：独立有界 trace；阶段候选、支持/拒绝、搜索事件、九项成本来源、文件失败和开关等价。基础模块 52 项、接入 59 项局部 unit 与 4 项 lit 及复审修复覆盖测试通过；最终两类 Attention 日志未截断，完整成本可复核，trace 开关语义相同。
- [x] P0-3：beam 合法解的确定性单变量改进；默认至多 256 邻域探测且共用 65536 总预算；完整成本与约束、统计和漏选回归。补齐整体复审发现的饱和全局不退化保护，最终方案/转换/成本日志一致。自由布局 Attention 完整向量从 `[8293,608,64,0,428,6144,6,0,0]` 改进至 `[8293,544,0,0,428,6144,6,0,0]`，新增 70 次探测、4 次接受；独立源码复审通过。
- [x] P0-4：修复后 450/450 unit、64/64 lit、7/7 Debug CTest、1/1 Release generator、5/5 Python normalization 及 API/tool 守门通过。8 组开关配置各 5 次测量，另有 4 组多分量基准各 30 次采样；维护文档及报告已同步。较低模型成本不等于 GPU 加速，自由布局推断时间中位数约增加 25%。

实现直接在源目录进行，未创建 worktree。按 [执行计划](optimize_plan/p0_execution_plan.md)测试驱动实施并逐任务/整体复审；不扩大形状/指令支持域，不实现 GPU lowering，后续实现仍未提交或推送。

> **M6 完成（2026-10-10）：Task27–29 实现及分项复审通过，最终 405/405 unit、58/58 lit、7/7 CTest、Python/生产 API 隔离通过。真实 GPU 的 11 个正确性用例、33 次 sanitizer 检查通过；三个必测用例 runtime 比值不超过 1.00834，布局编译占比 3.80%/3.97%/6.67%，参考基线已独立评审接受。整体源码及最终文档复审通过，按已声明支持域完成 M6 验收。首轮失败与修复过程保留在 §29.2，最终验收见 §29.3 及 `layout_inference_acceptance.md`。未 commit/push。**

> **当前进展（2026-09-28）：Task 26 按确认的方案 A 完成有序候选扩展、安全转换提升和共享重计算，分块/整体复审通过；357 unit / 53 lit / 4 CTest / 5 Python normalization、SM90 工具与旧生产接口隔离验收通过，契约、范围与结果见 §26.1–26.5。Task 24/25/26 修改保留在源目录 main，未 commit/push；HEAD 仍为 `36df49f`。Task 22/23 已推送至 origin/main。以下旧状态保留为历史记录。**

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Frisk 中实现一个面向 NVIDIA SM90/SM90a 的 MLIR-native 布局系统闭环，覆盖组合布局代数、Distributed/Storage IR、约束传播、有限候选求解、显式 conversion、正确性验证和性能守门。

**Architecture:** 采用“基础能力 + 纵向切片 + 逐类扩展”。Local/Register 使用带 `DistributedEncodingAttr` 的 RankedTensor SSA，Shared/Global 使用 MemRef 与 `StorageLayoutAttr` binding；Op 只收集约束，solver 只选择 assignment，materializer 在求解后统一改写 IR。现有 `LayoutAttr + DenseMap<Value, Attribute>` 通过 adapter 提供已知正确子集的迁移回归；不能作为新版规则的唯一真值来源，不能为差分一致保留已知错误。

**Tech Stack:** C++17、LLVM/MLIR ODS/TableGen、MLIR Pass/Dialect Conversion、Affine/Presburger、SCF、MemRef、Tensor、GPU/NVGPU/NVVM、LLVM ADT/APInt、CMake/Ninja、llvm-lit/FileCheck、CTest、CUDA/Nsight 性能工具。

> **最新验收（2026-09-27）：Task 23 按确认的方案 A 与 §23.1–23.8 完成，独立复审及 272 unit / 49 lit / 4 CTest / 5 Python 测试、旧生产接口隔离扫描通过，实际记录见 §23.9。9 变量核心及 Copy/Parallel128/MMA/Reduce/Global 写回整链成功，补齐 Task 21/22 的历史集成缺口；M5 Task 24–26 尚未完成。本轮仍在源目录 main，未 commit/push。以下旧状态保留为历史记录。**

> **执行状态（2026-09-19）：M0–M3 与 M4 Task 18–20 已完成。Task 21 已按确认契约完成受限 Reduce 实现、独立复审与 197 unit / 36 lit / 4 CTest 回归；完整 MMA→Reduce→tile_store 的成功验收仍因 9 变量超过保留的 8 变量上限而未完成，详见 §21.8，不将测试全通过等同于原契约全部验收。Task 22 尚未实施。Task 19/20 已提交到源目录 main 的 `c028e82`，未 push；Task 21 未提交、未推送。**

> **最新进展（2026-09-20）：Task 22 方案 A 及具体契约已确认，生命周期 normalization、持久根契约、事务 pipeline 和旧 oracle 隔离已实现，已通过最终独立复审与全量验收，记录见 §22.13。此前 Task 19/20 的 `c028e82` 与 Task 21 的 `14b39e9` 已推送到 `origin/main`（guoqiao7/frisk），以上未提交/未推送为历史时点记录。本轮修改尚未提交或推送，保留独立的 Task19–21 对比报告。**

> **设计更新（2026-09-19）：Task 20 的 SS＋RS 具体契约 v1 已由用户确认；§20.1–20.11 是契约，§20.12 是执行计划，§20.13 记录实际实现、验证证据和限制。通用 encoding＋操作级指令 binding 的适配同时写入设计总文档 §19。**

> **集成与后续（2026-09-19）：用户要求先提交既有修改，再进入 Task 21。Task 19/20 的代码、测试、维护文档和汇报已提交到本地 main：`c028e82`，提交前重新验证 165 unit / 34 lit / 4 CTest 全部通过；未 push。以上“未提交”描述是当时的实施记录，以本条更新为准。Task 21 的 §21.1–21.7 已由用户确认并实施；实际证据与尚未解决的完整链验收冲突见 §21.8。**

> **工作区说明：M3 从 `feature/m3-distributed-layout`（`be71d91`，实现验收基线 `642c30b`）于 2026-09-11 合入 `main`，合并提交 `b120d06400a14a703a44dac1a37a0b38d8110935` 已在此前按用户要求推送至 `origin`（guoqiao7/frisk）。Task 18 历史上使用 `.worktrees/m4-task18`，集成记录见 §18.10。本轮 Task 19 遵照用户新要求直接在主源码 main 中实现，不使用 worktree，不改动独立的 `guoqiao/inference_detail/README.md`，不自动 commit/push。27 lit、74 unit、4 CTest 是 M3 基线；Task 18/19 Gate 分别见 §18.9/§19.9。**

## Global Constraints

- 首个目标仅为 NVIDIA SM90/SM90a；不得在通用 solver 中散布 target 字符串判断。
- 完整 WGMMA/TMA/mbarrier 指令 lowering 不在本计划范围；本计划实现布局契约、静态成本和 conversion 微基准所需的最小 lowering adapter。
- 本轮 TileLang 审计及 Task28 已建立的新语义参考固定为 `5e149e31674658f94779c7d0c6039549a1853123`；旧 `6623b12d232b343648a5ba99992e3e6f0d6376d2` 保留作历史迁移参考。上游升级不自动改变 Frisk 契约；每次须记录新旧行为与采用理由，不得链接或 vendor TVM/TIRX 类型。
- Local/Register 的最终公共表示必须是 RankedTensor SSA；不得继续扩展 local-memory MemRef 作为长期寄存器模型。
- Shared/Global 保持 MemRef；XOR storage mapping 通过 `frisk.layout_view`/binding 表达，不强塞进 MemRef semi-affine layout。
- `frisk.convert_layout` 和 rematerialization 是普通候选，不以“零转换方案无解”为启用条件。
- Hard constraint 不得被代价模型违反；Affine/Presburger 的 `unknown` 不得当作证明成功。
- 所有 worklist、constraint、candidate 和 tie-break 必须使用稳定 ID，结果不得依赖 DenseMap/指针遍历顺序。
- 用户 IR 错误、unsupported shape 和求解冲突必须返回带位置的诊断；不得用 assertion 处理。
- 每项公共语义或架构边界发生变化时，必须在同一任务中同步维护 `guoqiao/layout_inference_design.md`。
- 长期同时维护本计划和 `guoqiao/layout_inference_strategy_comparison.md`：计划记完成度/设计/验收，比较记版本证据/取舍/差异假设。Task 18 确认实施后同步设计总文档和属性/API 注释；公共地址与对齐契约见 §18.2–18.3。
- 现有用户修改属于用户；不得覆盖 `build.sh` 或其他与当前任务无关的工作区变更。
- 正确性零容忍：race、OOB、invalid ownership、未解析 LayoutVar 和 nondeterministic IR 必须为 0。
- 性能守门：受支持 kernel 固定环境中位运行时间回退不超过 3%；沿用 Task23 的 exactCombinationLimit=256、beamWidth=32、maxExpandedStates=65536，256 是精确搜索组合数阈值而非候选硬上限；对具备真实完整编译路径的用例，布局推断不超过完整编译时间的 10%，无完整路径只报告阶段耗时，不冒充该比例通过。

---

## 0. 执行规则

### 0.1 每个任务的固定循环

```text
确认前置任务和接口
  -> 写一个会失败的最小测试
  -> 运行并确认失败原因正确
  -> 写满足该测试的最小实现
  -> 运行局部测试
  -> 运行本里程碑回归
  -> 检查设计文档是否漂移
  -> 提交一个可独立评审的 commit
```

不得把多个未验证任务合并后一次测试。一个任务的退出条件未满足时，不进入依赖它的任务。

### 0.2 基础构建命令

首次配置或 CMake 结构变化后：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
```

日常构建：

```bash
cmake --build build --target FriskIR FriskTransforms frisk-opt check-frisk --parallel 32
```

现有基线：

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

### 0.3 文档维护检查

每个任务提交前运行：

```bash
git diff --check
rg -n "LayoutAttr|DistributedEncodingAttr|StorageLayoutAttr|convert_layout|CostVector" \
  guoqiao/layout_inference_design.md \
  guoqiao/layout_inference_implementation_plan.md \
  guoqiao/layout_inference_strategy_comparison.md
```

如果实现改变了属性字段、映射方向、hard/soft 边界、候选优先级、conversion 策略、里程碑范围或性能阈值，必须先更新设计文档再提交。

每项布局任务还必须留下以下审计记录：

1. Frisk 与上游固定 SHA、审计日期、源码符号/永久链接；区分旧版已有和本区间新增。
2. 旧规则的问题、当前 Frisk 的适配约束，以及采用/调整/拒绝的决定；源版本升级不能只更换 SHA。
3. 已实现、设计待确认、未来目标分别标记；测试源码存在不等于本轮执行通过。
4. 新旧行为的正例、反例、失败策略与实际验证命令；旧版错误用独立语义不变量修正，不能强制复制。
5. 差异化写成可检验假设，注明当前证据和否证条件；不把通用 alias/GF(2)/while/worklist 当作首创。

审计历史与精确上游证据集中在[比较文档 §0、§13、§15](./layout_inference_strategy_comparison.md)；下方 M4 记录审计缺口、已确认设计和实现验收。Task28 已真实导出并验证固定 TileLang corpus（§28.2），不等同于运行上游完整测试集；本轮尚无 Triton 运行测试或 GPU 性能结论。

## 1. 里程碑与依赖

| 里程碑              | 任务   | 可独立验收的结果                                                                  |
| ------------------- | ------ | --------------------------------------------------------------------------------- |
| M0 基础设施         | 1–4   | 原生测试可由 CTest/lit 运行；`frisk-opt` 可加载 dialect/pass；MemoryEffect 正确 |
| M1 布局代数         | 5–9   | Affine/BitLinear/Product 与 Distributed/Storage Attr 可解析、组合和验证           |
| M2 Storage 切片     | 10–13 | alloc/view/copy 完成 constraint → propagation → binding → verifier             |
| M3 Distributed 切片 | 14–17 | Tensor 多 consumer 可选择共同布局或物化`frisk.convert_layout`                   |
| M4 完整传播迁移     | 18–22 | alias/region/Copy/Fill/Gemm/Reduce 进入新 solver；旧生产路径退役                  |
| M5 候选与 SM90      | 23–26 | 全局候选选择、SM90 契约、conversion 优化和 controlled relaxation 完整             |
| M6 硬化验收         | 27–29 | property/differential/performance/compile-time 守门通过，文档与实现一致           |

## 2. 目标代码结构

```text
include/Dialect/Frisk/IR/
  FriskLayoutAttrs.td
  FriskLayoutAttrInterfaces.td
  FriskLayoutOpInterfaces.td
  FriskLayoutInterfaces.h
  FriskLayoutOps.td

include/Dialect/Frisk/Analysis/
  LayoutCommon.h
  LayoutAlgebra.h
  LayoutConstraint.h
  LayoutSolver.h
  LayoutVerifier.h
  LayoutTarget.h
  LegacyLayoutAdapter.h
  LayoutStatistics.h

include/Dialect/Frisk/Target/SM90/
  SM90LayoutTarget.h

include/Dialect/Frisk/Transforms/
  LayoutTypeConverter.h
  PassPipelines.h
  Passes.h
  Passes.td

lib/Dialect/Frisk/IR/
  FriskLayoutAttrs.cpp
  FriskLayoutInterfaces.cpp
  FriskLayoutOps.cpp

lib/Dialect/Frisk/Analysis/
  LayoutAlgebra.cpp
  LayoutConstraint.cpp
  LayoutPropagation.cpp
  LayoutCandidateSolver.cpp
  LayoutCostModel.cpp
  LayoutVerifier.cpp
  LegacyLayoutAdapter.cpp
  LayoutStatistics.cpp

lib/Dialect/Frisk/Target/SM90/
  SM90LayoutTarget.cpp
  SM90CopyConstraints.cpp
  SM90GemmConstraints.cpp
  SM90ReduceConstraints.cpp
  SM90CostModel.cpp

lib/Dialect/Frisk/Transforms/
  NormalizeLayoutIR.cpp
  LayoutInfer.cpp
  LayoutTypeConverter.cpp
  MaterializeLayouts.cpp
  OptimizeLayoutConversions.cpp
  PassPipelines.cpp

lib/Conversion/FriskLayoutToGPU/
  CMakeLists.txt
  TestLayoutConversionLowering.cpp

tools/frisk-opt/
  CMakeLists.txt
  frisk-opt.cpp

test/Dialect/Frisk/layout/
test/Transforms/
unittests/Dialect/Frisk/Layout/
benchmark/layout/conversion/
benchmark/layout/solver/
```

文件按职责拆分。`FriskOps.cpp` 中现有 GEMM/shared layout 构造只在迁移任务中移动，不提前做无关重构。

---

## M0：基础设施与现状冻结

### Task 1: 将现有原生测试纳入 CTest 并冻结基线

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `test_pass/CMakeLists.txt`
- Create: `guoqiao/layout_baseline.md`
- Verify: `test_pass/attr_test.cpp`
- Verify: `test_pass/reduce_layout_test.cpp`

**Interfaces:**

- Consumes: 当前 `FriskIR`、`frisk_attr_test`、`frisk_reduce_layout_test`。
- Produces: `ctest --test-dir build` 可发现并运行两个 legacy baseline；后续所有任务使用同一入口做回归。

- [X] **Step 1: 记录当前测试发现状态**

Run:

```bash
ctest --test-dir build -N
```

Expected: 输出 `Total Tests: 0`，证明当前 CMake 尚未注册测试。

- [X] **Step 2: 运行两个现有可执行测试并保存基线结论**

Run:

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

Expected: 两个进程退出码均为 0；将覆盖的 target/shape/reduce case 和当前已知限制写入 `guoqiao/layout_baseline.md`，不得复制大段日志。

- [X] **Step 3: 注册 CTest**

在顶层 `CMakeLists.txt` 的 `project` 命令后加入：

```cmake
include(CTest)
enable_testing()
```

在 `test_pass/CMakeLists.txt` 两个 executable 定义后加入：

```cmake
add_test(NAME FriskAttrTest COMMAND frisk_attr_test)
add_test(NAME FriskReduceLayoutTest COMMAND frisk_reduce_layout_test)
```

- [X] **Step 4: 重新配置并验证测试发现**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target frisk_attr_test frisk_reduce_layout_test --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 2`。

- [X] **Step 5: 提交基线**

```bash
git add CMakeLists.txt test_pass/CMakeLists.txt guoqiao/layout_baseline.md
git commit -m "test: register legacy layout baselines"
```

### Task 2: 编译并注册真正的布局推断 pass

**Files:**

- Modify: `include/Dialect/Frisk/Transforms/Passes.h:1-18`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td:1-17`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt:1-13`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp:1-25`
- Create: `test_pass/layout_pass_test.cpp`
- Modify: `test_pass/CMakeLists.txt`

**Interfaces:**

- Consumes: MLIR `OperationPass<FunctionOpInterface>`、生成的 `impl::FriskInferLayoutsBase`。
- Produces: `std::unique_ptr<Pass> createFriskInferLayoutsPass()` 和注册参数 `frisk-infer-layouts`。

- [X] **Step 1: 写会失败的 pass 构造测试**

创建 `test_pass/layout_pass_test.cpp`，核心断言为：

```cpp
#include "Dialect/Frisk/Transforms/Passes.h"

int main() {
  std::unique_ptr<mlir::Pass> pass = mlir::frisk::createFriskInferLayoutsPass();
  if (!pass || pass->getArgument() != "frisk-infer-layouts")
    return 1;
  return 0;
}
```

Run:

```bash
cmake --build build --target frisk_layout_pass_test --parallel 32
```

Expected: FAIL，原因是 target/`createFriskInferLayoutsPass` 尚不存在。

- [X] **Step 2: 统一 pass 名称和构造器**

将 `Passes.td` 定义改为：

```tablegen
def FriskInferLayouts : InterfacePass<"frisk-infer-layouts", "FunctionOpInterface"> {
  let summary = "Infer and materialize Frisk layouts";
  // MLIR 21 pass TableGen expects a call expression, including `()`.
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
}
```

删除 `Passes.h` 中旧的 `createLayoutInferPass()` 手写声明，改为：

```cpp
std::unique_ptr<Pass> createFriskInferLayoutsPass();
```

- [X] **Step 3: 实现最小可运行 pass**

`LayoutInfer.cpp` 使用生成基类：

```cpp
namespace {
class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  void runOnOperation() override {
    getOperation()->setAttr("frisk.layout_inference_ran",
                            UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<Pass> createFriskInferLayoutsPass() {
  return std::make_unique<FriskInferLayoutsPass>();
}
```

将 `LayoutInfer.cpp` 加入 `FriskTransforms` source list，并链接 `MLIRFuncDialect`、`MLIRIR`、`MLIRPass`、`FriskIR`。

- [X] **Step 4: 注册并运行测试**

在 `test_pass/CMakeLists.txt` 新增 `frisk_layout_pass_test`，链接 `FriskTransforms`，并注册 CTest。

Run:

```bash
cmake --build build --target frisk_layout_pass_test --parallel 32
ctest --test-dir build -R FriskLayoutPassTest --output-on-failure
```

Expected: `FriskLayoutPassTest` PASS。

- [X] **Step 5: 提交 pass 驱动**

```bash
git add include/Dialect/Frisk/Transforms lib/Dialect/Frisk/Transforms test_pass
git commit -m "feat: add executable layout inference pass"
```

### Task 3: 增加 `frisk-opt` 与 MLIR lit 测试入口

**Files:**

- Modify: `CMakeLists.txt`
- Create: `tools/CMakeLists.txt`
- Create: `tools/frisk-opt/CMakeLists.txt`
- Create: `tools/frisk-opt/frisk-opt.cpp`
- Create: `test/CMakeLists.txt`
- Create: `test/lit.cfg.py`
- Create: `test/lit.site.cfg.py.in`
- Create: `test/Transforms/infer-layouts-smoke.mlir`

**Interfaces:**

- Consumes: `FriskDialect`、`registerFriskPasses()`、`MlirOptMain`。
- Produces: `build/bin/frisk-opt` 和 `check-frisk` target。

- [X] **Step 1: 写命令行 smoke test**

创建：

```mlir
// RUN: frisk-opt %s -pass-pipeline='builtin.module(func.func(frisk-infer-layouts))' | FileCheck %s

module {
  func.func @smoke() {
    return
  }
}

// CHECK: frisk.layout_inference_ran
```

Run:

```bash
cmake --build build --target check-frisk --parallel 32
```

Expected: FAIL，原因是 `check-frisk`/`frisk-opt` 尚不存在。

- [X] **Step 2: 实现 opt driver**

`tools/frisk-opt/frisk-opt.cpp`：

```cpp
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect>();
  mlir::registerAllPasses();
  mlir::frisk::registerFriskPasses();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Frisk optimizer\n", registry));
}
```

对应 CMake 必须同时链接 `registerAllDialects()`/`registerAllPasses()` 引用的
MLIR 静态库集合；仅链接 `MLIROptLib` 不会传递所有注册实现：

```cmake
add_llvm_executable(frisk-opt frisk-opt.cpp)
set_target_properties(frisk-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
llvm_update_compile_flags(frisk-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(frisk-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  FriskIR
  FriskTransforms
)
```

- [X] **Step 3: 配置 lit**

`test/lit.cfg.py` 必须配置：

```python
config.name = "FRISK"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.frisk_obj_root, "test")
llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    [ToolSubst("frisk-opt", unresolved="fatal")],
    [config.frisk_tools_dir, config.llvm_tools_dir],
)
```

`test/CMakeLists.txt` 使用 `configure_lit_site_cfg`，并按以下方式创建目标：

```cmake
# 顶层必须在 add_subdirectory(test) 前提供 Python3_EXECUTABLE，避免依赖
# LLVM 构建树中 llvm-lit 脚本自身可能不可移植的 shebang。
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_lit_testsuite(check-frisk "Running Frisk regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS frisk-opt
)
```

顶层增加 `add_subdirectory(tools)` 与 `add_subdirectory(test)`。

- [X] **Step 4: 验证 driver 和 lit**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target frisk-opt check-frisk --parallel 32
```

Expected: lit 报告 `1 passed`。

- [X] **Step 5: 提交测试入口**

```bash
git add CMakeLists.txt tools test
git commit -m "test: add frisk-opt and lit harness"
```

### Task 4: 修正 MemoryEffect 与容器 Op 副作用传播

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Create: `test_pass/memory_effect_test.cpp`
- Modify: `test_pass/CMakeLists.txt`
- Modify: `guoqiao/layout_inference_design.md` only if the implemented effect model differs from Section 8.2

**Interfaces:**

- Consumes: MLIR `MemoryEffectOpInterface`、`RecursiveMemoryEffects`。
- Produces: Alloc/Copy/Fill/Gemm/Reduce 的准确 Allocate/Read/Write effect；Kernel/Parallel/Block/For 递归暴露 region effects。

- [X] **Step 1: 写失败的 effect 测试**

测试构造 Copy、Fill、Gemm、Reduce、Alloc，并断言：

```cpp
if (mlir::isMemoryEffectFree(copy) || mlir::isMemoryEffectFree(gemm) ||
    mlir::isMemoryEffectFree(reduce))
  return 1;
if (!mlir::hasEffect<mlir::MemoryEffects::Write>(fill) ||
    !mlir::hasEffect<mlir::MemoryEffects::Allocate>(alloc))
  return 1;
```

Run:

```bash
cmake --build build --target frisk_memory_effect_test --parallel 32
```

Expected: FAIL，因为现有 ODS 把多个内存 Op 标记为 `Pure`。

- [X] **Step 2: 修正 ODS effects**

采用以下边界：

```tablegen
// 容器
Frisk_Op<"kernel", [RecursiveMemoryEffects,
                     ImplicitFriskTerminator]>
Frisk_Op<"parallel", [RecursiveMemoryEffects,
                       HasParent<"KernelOp">,
                       ImplicitFriskTerminator]>
Frisk_Op<"block", [RecursiveMemoryEffects, ImplicitFriskTerminator]>
Frisk_Op<"for", [RecursiveMemoryEffects, ImplicitFriskTerminator]>

// 空终止符必须显式无副作用，否则容器递归 effect 查询会返回 unknown。
Frisk_Op<"end", [Pure, Terminator, ReturnLike]>

// 数据操作：删除 Pure，按 operand/result 声明
Arg<AnyMemRef, "", [MemRead]>:$src
Arg<AnyMemRef, "", [MemWrite]>:$dst
Res<AnyMemRef, "", [MemAlloc]>:$result
```

Gemm 的 A/B 为 `MemRead`，C 为 `MemRead, MemWrite`；Reduce 的 src 为 `MemRead`、dst 为 `MemWrite`；Copy/FIll 使用现有 operand effect 并删除 `Pure`。

- [X] **Step 3: 生成、编译并运行 effect 测试**

Run:

```bash
cmake --build build --target FriskTableGen FriskIR frisk_memory_effect_test --parallel 32
ctest --test-dir build -R FriskMemoryEffectTest --output-on-failure
```

Expected: PASS。

- [X] **Step 4: 运行 M0 全量回归**

Run:

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target check-frisk --parallel 32
```

Expected: 所有 CTest 和 lit tests PASS。

- [X] **Step 5: 提交 MemoryEffect 修复**

```bash
git add include/Dialect/Frisk/IR/FriskOps.td test_pass guoqiao/layout_inference_design.md
git commit -m "fix: model Frisk memory effects"
```

### M0 Gate

必须同时满足：

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target check-frisk --parallel 32
build/bin/frisk-opt --help-list | rg "frisk-infer-layouts"
```

Expected: CTest/lit 全部通过，pass 出现在完整帮助列表中。MLIR 21 的普通
`--help` 不展开 pass 清单，因此这里必须使用 `--help-list`。未满足时不得进入 M1。

### M0 执行记录（2026-08-30）

状态：**完成**。

独立提交：

```text
29fdf5e test: register legacy layout baselines
487e4e8 feat: add executable layout inference pass
4c57a18 test: add frisk-opt and lit harness
7af9e04 fix: model Frisk memory effects
```

最终 Gate 结果：

```text
ctest --test-dir build --output-on-failure
  4/4 passed

cmake --build build --target check-frisk --parallel 32
  Total Discovered Tests: 1
  Passed: 1

build/bin/frisk-opt --help-list | rg "frisk-infer-layouts"
  --frisk-infer-layouts - Infer and materialize Frisk layouts
```

实施中确认并回写计划的工具链约束：

1. MLIR 21 pass TableGen 的 `constructor` 必须是带 `()` 的调用表达式。
2. 调用 `registerAllDialects()`/`registerAllPasses()` 的静态链接 driver 必须链接 MLIR dialect、conversion、extension 及核心 transform 库集合。
3. lit 配置前必须提供 `Python3_EXECUTABLE`；`frisk-opt` 必须显式输出到 `build/bin`。
4. RecursiveMemoryEffects 容器中的 `frisk.end` 必须显式 `Pure`，否则递归 effect 查询会返回 unknown。
5. MLIR 21 的 pass 注册验收使用 `frisk-opt --help-list`；普通 `--help` 不展示 pass 清单。

---

## M1：组合布局代数与新 Attr

### Task 5: 建立布局公共类型和生成式 Interface

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutCommon.h`
- Create: `include/Dialect/Frisk/IR/FriskLayoutAttrInterfaces.td`
- Create: `include/Dialect/Frisk/IR/FriskLayoutOpInterfaces.td`
- Create: `include/Dialect/Frisk/IR/FriskLayoutInterfaces.h`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutInterfaces.cpp`
- Modify: `include/Dialect/Frisk/IR/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutInterfaceTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/CMakeLists.txt`
- Create: `unittests/CMakeLists.txt`
- Modify: `CMakeLists.txt`

**Interfaces:**

- Produces:

```cpp
enum class LayoutKind { Distributed, Storage, Instruction };
enum class ProofStatus { Proven, Disproven, Unknown };

struct LayoutProof {
  ProofStatus status;
  SmallVector<int64_t> counterexample;
  std::string reason;
};

struct CostVector {
  uint64_t instructionPathAndWork;
  uint64_t memoryTransactions;
  uint64_t bankConflictDegree;
  uint64_t conversionBytesAndSync;
  uint64_t spillRiskAndRegisters;
  uint64_t sharedBytesAndOccupancy;
  uint64_t replication;
  uint64_t codeSize;
  uint64_t deterministicTieBreak;
};

class LayoutMapAttrInterface;
class LayoutEncodingAttrInterface;
class LayoutConstraintOpInterface;
class LayoutConstraintBuilder;
```

- [X] **Step 1: 写 Interface 生成失败测试**

测试包含新头文件并做静态检查：

```cpp
static_assert(std::is_enum_v<mlir::frisk::LayoutKind>);
static_assert(std::is_enum_v<mlir::frisk::ProofStatus>);
```

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
```

Expected: FAIL，原因是头文件和 unit test target 尚不存在。

- [X] **Step 2: 定义公共结果类型**

`LayoutCommon.h` 定义上述枚举，并增加：

```cpp
struct LayoutDimension {
  StringAttr name;
  int64_t extent;
};

using LayoutMapAttr = LayoutMapAttrInterface;
```

`extent == ShapedType::kDynamic` 只允许出现在 affine outer；BitLinear 的 bit width 必须静态。

- [X] **Step 3: 定义 Attr/Encoding 接口**

`FriskLayoutAttrInterfaces.td` 至少声明：

```tablegen
def LayoutMapAttrInterface : AttrInterface<"LayoutMapAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let methods = [
    InterfaceMethod<"Return canonical map", "::mlir::FailureOr<::mlir::Attribute>",
                    "canonicalizeMap", (ins)>,
    InterfaceMethod<"Verify named dimensions", "::mlir::LogicalResult",
                    "verifyMap", (ins "::mlir::Location":$loc)>
  ];
}

def LayoutEncodingAttrInterface : AttrInterface<"LayoutEncodingAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let methods = [
    InterfaceMethod<"Expand encoding", "::mlir::FailureOr<::mlir::Attribute>",
                    "getCanonicalMap", (ins "::mlir::ShapedType":$type)>,
    InterfaceMethod<"Verify encoding", "::mlir::LogicalResult",
                    "verifyForType", (ins "::mlir::ShapedType":$type,
                                           "::mlir::Location":$loc)>,
    InterfaceMethod<"Return encoding kind", "::mlir::frisk::LayoutKind",
                    "getKind", (ins)>
  ];
}
```

`FriskLayoutOpInterfaces.td` 声明 `collectLayoutConstraints(LayoutConstraintBuilder&)`；此时只生成声明，Task 11 再提供 builder 实现。

- [X] **Step 4: 接入 TableGen 和 unit test**

为 Attr interface 和 Op interface 分别生成 decl/def，避免两个 generator 写同一文件；`FriskIR` 增加 `FriskLayoutInterfaces.cpp`。Unit test 使用：

```cmake
add_unittest(FriskUnitTests FriskLayoutUnitTests
  LayoutInterfaceTest.cpp
)
target_link_libraries(FriskLayoutUnitTests PRIVATE FriskIR MLIRIR)
```

Frisk 是外部 MLIR 工程；若 LLVM 配置未导出 `llvm_gtest` target，`unittests/CMakeLists.txt`
必须先从 `${LLVM_MAIN_SRC_DIR}/../third-party/unittest` 引入与当前 LLVM 源码匹配的
gtest/gmock，再调用 `add_unittest`。不能依赖系统 gtest 或裸 `-lllvm_gtest`。

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
```

Expected: unit test PASS。

- [X] **Step 5: 提交接口骨架**

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk unittests CMakeLists.txt
git commit -m "feat: define layout interfaces and proof types"
```

### Task 6: 实现 GF(2) Matrix 核心

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutAlgebra.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutAlgebra.cpp`
- Create: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/CMakeLists.txt`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/GF2MatrixTest.cpp`

**Interfaces:**

- Produces:

```cpp
class GF2Matrix {
public:
  static FailureOr<GF2Matrix> get(unsigned rows, unsigned cols,
                                  ArrayRef<APInt> rowBits);
  unsigned getNumRows() const;
  unsigned getNumColumns() const;
  APInt apply(const APInt &input) const;
  GF2Matrix transpose() const;
  FailureOr<GF2Matrix> compose(const GF2Matrix &rhs) const;
  unsigned rank() const;
  SmallVector<APInt> kernelBasis() const;
  FailureOr<GF2Matrix> inverse() const;
  FailureOr<GF2Matrix> rightInverse() const;
  bool operator==(const GF2Matrix &rhs) const;
};
```

- [X] **Step 1: 写矩阵红灯测试**

至少覆盖 identity、XOR、rank-deficient、inverse 和 compose 顺序：

```cpp
TEST(GF2MatrixTest, ComposeUsesRhsThenLhs) {
  GF2Matrix distributed = makeMatrix({0b01, 0b11}, 2);
  GF2Matrix storage = makeMatrix({0b10, 0b11}, 2);
  auto composed = storage.compose(distributed);
  ASSERT_TRUE(succeeded(composed));
  EXPECT_EQ(composed->apply(APInt(2, 0b11)),
            storage.apply(distributed.apply(APInt(2, 0b11))));
}
```

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
```

Expected: FAIL，`GF2Matrix` 未定义。

- [X] **Step 2: 实现构造、apply、transpose 和 compose**

内部使用 `SmallVector<APInt> rows`。乘加规则固定为：

```cpp
bool bit = (rows[row] & input).popcount() & 1;
```

`lhs.compose(rhs)` 表示 `lhs(rhs(x))`；维度不匹配返回 failure，不截断 APInt。

- [X] **Step 3: 实现消元、kernel、inverse/right-inverse**

使用确定性 Gauss-Jordan：pivot 按 column 从小到大、row 从小到大选择。`inverse()` 仅方阵满秩成功；`rightInverse()` 仅输出空间被覆盖时成功。

- [X] **Step 4: 增加穷举 oracle 并验证**

对输入 bit 数不超过 8 的矩阵枚举所有输入，比较 `compose/apply`；对可逆矩阵验证 `A.inverse()(A(x)) == x`。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='GF2MatrixTest.*'
```

Expected: 所有 GF2Matrix tests PASS。

- [X] **Step 5: 提交 GF(2) 核心**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis unittests
git commit -m "feat: add deterministic GF2 matrix algebra"
```

### Task 7: 实现 `BitLinearLayoutMapAttr`

**Files:**

- Create: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `include/Dialect/Frisk/IR/FriskAttributes.td`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `test/Dialect/Frisk/layout/bit-linear-attr.mlir`
- Create: `unittests/Dialect/Frisk/Layout/BitLinearLayoutTest.cpp`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`

**Interfaces:**

- Produces `BitLinearLayoutMapAttr`，字段固定为：

```text
input_names       : ArrayAttr<StringAttr>
input_bit_widths  : DenseI64ArrayAttr
output_names      : ArrayAttr<StringAttr>
output_bit_widths : DenseI64ArrayAttr
matrix            : DenseIntElementsAttr<tensor<rows x cols x i1>>
```

- [X] **Step 1: 写 parse/verify 红灯测试**

测试包含一个合法 XOR map 和三个非法 map：matrix shape 错、重复 dimension name、bit width 为 0。

```mlir
// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s
module attributes {
  frisk.map = #frisk.bit_linear<
    inputs = ["lane", "register"], input_bits = [2, 1],
    outputs = ["m", "n"], output_bits = [1, 2],
    matrix = dense<[[1, 0, 0], [0, 1, 1], [0, 0, 1]]> : tensor<3x3xi1>>
} {}
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，attribute 尚未定义。

- [X] **Step 2: 定义 ODS Attr 和 verifier**

Attr 实现 `LayoutMapAttrInterface`。Verifier 检查：名称唯一、name/width 数量一致、所有 width > 0、matrix 行数等于输出总 bit、列数等于输入总 bit。

- [X] **Step 3: 接入 GF2 algebra**

实现：

```cpp
FailureOr<GF2Matrix> BitLinearLayoutMapAttr::getMatrixValue() const;
FailureOr<Attribute> BitLinearLayoutMapAttr::canonicalizeMap() const;
LayoutProof checkInjective(BitLinearLayoutMapAttr map);
LayoutProof checkSurjective(BitLinearLayoutMapAttr map);
FailureOr<BitLinearLayoutMapAttr>
composeBitLinear(BitLinearLayoutMapAttr lhs, BitLinearLayoutMapAttr rhs);
```

Canonical form 保持命名维度顺序，但删除 width 为 0 的禁止状态、规范 DenseElements 存储并拒绝重复名称；不得按 hash 顺序重排。

- [X] **Step 4: 验证 parse/print、compose 和 replication**

Unit test 检查全零 input column 被识别为 kernel/replication，FileCheck 检查 parse-print 稳定。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='BitLinearLayoutTest.*'
```

Expected: unit/lit tests PASS。

- [X] **Step 5: 提交 BitLinear Attr**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test unittests
git commit -m "feat: add bit-linear layout attribute"
```

### Task 8: 实现 Affine outer 与 Product layout

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutAlgebra.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutAlgebra.cpp`
- Create: `unittests/Dialect/Frisk/Layout/ProductLayoutTest.cpp`
- Create: `test/Dialect/Frisk/layout/product-layout.mlir`

**Interfaces:**

- Produces:

```text
AffineLayoutMapAttr(input_names, input_extents,
                    output_names, output_extents, affine_map)
ProductLayoutMapAttr(outer, inner, split_extents)
```

```cpp
FailureOr<Attribute> composeLayoutMaps(Attribute lhs, Attribute rhs);
FailureOr<Attribute> projectLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
FailureOr<Attribute> permuteLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
LayoutProof checkCoverage(Attribute map, ArrayRef<int64_t> logicalShape);
LayoutProof checkInjectivity(Attribute map, ArrayRef<int64_t> domainShape);
```

- [X] **Step 1: 写 non-power-of-two 与 carry 红灯测试**

测试概念语义 `logical = outer * innerExtent + inner`；Attr 中将 Affine outer
规范存为已缩放的 `outer_base = outer * innerExtent`，实际求值为
`logical = outer_base + inner`。覆盖 shape 6 的 ragged predicate，以及 outer base
未对齐时禁止直接拼接 GF(2) 内层。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，Affine/Product API 尚不存在。

- [X] **Step 2: 实现 `AffineLayoutMapAttr` verifier**

检查 AffineMap dim 数、symbol 数、name/extent 数量、正静态 extent 或 `ShapedType::kDynamic`。动态 extent 只允许作为 symbol/bounds，不得进入 BitLinear matrix。

- [X] **Step 3: 实现 Product compose 和 canonicalization**

固定组合流程：

```text
按名字对齐维度
-> 验证 inner extent 为 2 的幂
-> 验证 outer base 不向 inner bit 产生 carry
-> 分别 compose outer/inner
-> 合并 predicate
-> canonicalize factor order
```

Product verifier 无法证明无 carry时直接拒绝构造；独立 proof API 对动态/超限情形返回
`ProofStatus::Unknown`，不得生成不安全 Product，也不得把 `Unknown` 当作成功。

- [X] **Step 4: 运行枚举 oracle**

对每维 extent 不超过 8 的 product map 枚举 `(outer, inner)`，比较 canonical map 与逐层求值；覆盖 transpose、projection、padding、ragged。

实现额外设置 65,536 个 domain point 的硬上限：完整 BitLinear domain 优先使用
GF(2) rank 精确证明；静态 Affine/Product 仅作为 verifier/测试的有界参考证明，动态、
含 symbol 或超限时返回 `Unknown`，不得进入 solver 的逐候选热循环。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ProductLayoutTest.*'
```

Expected: PASS。

- [X] **Step 5: 提交组合代数**

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests
git commit -m "feat: compose affine and bit-linear layouts"
```

### Task 9: 实现 Distributed/Storage Encoding 与 legacy adapter

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `include/Dialect/Frisk/IR/FriskAttributes.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Create: `include/Dialect/Frisk/Analysis/LegacyLayoutAdapter.h`
- Create: `lib/Dialect/Frisk/Analysis/LegacyLayoutAdapter.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutEncodingTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LegacyLayoutAdapterTest.cpp`
- Create: `test/Dialect/Frisk/layout/layout-encoding.mlir`

**Interfaces:**

- Produces:

```cpp
FailureOr<DistributedEncodingAttr>
convertLegacyDistributed(LayoutAttr legacy, ShapedType type,
                         Location loc);

FailureOr<StorageLayoutAttr>
convertLegacyStorage(LayoutAttr legacy, MemRefType type,
                     Location loc);
```

Encoding 字段：

```text
DistributedEncodingAttr(map, topology, replication)
StorageLayoutAttr(map, memory_space, alignment, vector_granularity)
```

ODS assembly format 固定为命名字段；后续测试中的 `#smem_layout` 和 `#enc` 必须在文件顶部完整定义，例如：

```mlir
#smem_layout = #frisk.storage<
  map = #smem_map,
  memory_space = #frisk<memory_space Shared>,
  alignment = 128,
  vector_granularity = 16>
#enc = #frisk.distributed<
  map = #blocked_map,
  topology = [8, 32, 4, 1, 1],
  replication = 1>
```

`topology` 的字段顺序固定为 `(register, lane, warp, warp_group, cta)`；parser、printer 和 verifier 共同使用这一顺序，禁止依靠调用点注释猜测。

- [X] **Step 1: 写 type verifier 和 baseline conversion 红灯测试**

测试：encoding logical shape 不匹配、Storage map 非单射、shared alignment 非正、现有 `sm90_ss` Gemm A/B/C legacy layout 转换。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，新 Encoding/adapter 不存在。

- [X] **Step 2: 实现 Encoding verifier**

Distributed 检查 carrier names 只来自 `register/lane/warp/warp_group/cta`、coverage、replication 与 topology；对 BitLinear map 精确验证
`replication = product(topology) / 2^rank(B_D)`。Storage 检查 memory space、live
domain injectivity、`(byte_offset, bit_offset)`、byte 单位的 alignment/vector
granularity。

- [X] **Step 3: 实现显式支持集的 legacy adapter**

Adapter 仅支持当前 SM90 baseline 中可证明的静态 2 次幂 fragment、linear/padded
shared 和能从完整真值表恢复 GF(2) basis 的 swizzle。SM80 旧路径继续回归，但不作为
当前 SM90-only adapter 的支持承诺。无法识别时：

```cpp
return emitError(loc)
       << "legacy layout cannot be represented by the canonical layout algebra";
```

不得猜测、不得退化为 linear layout。对 swizzle 使用现有语义逐点枚举构造 GF(2) basis，并验证全域后才返回。

实现细节：adapter 单独编译为 `FriskLegacyLayoutAdapter`，依赖 `FriskIR` 和
`FriskAnalysis`，从而避免 `FriskIR <-> FriskAnalysis` 循环。多结果 legacy storage
tuple 采用由各结果最大 extent 确定的 row-major flatten；只有单结果 map 可进入
Affine fallback。枚举上限为 65,536，超限返回 failure。

- [X] **Step 4: 对现有 baseline 做等价枚举**

对每个 `(thread, register)` 和 logical shared point，比较 legacy 与新 canonical map；测试失败必须打印首个反例。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutEncodingTest.*:LegacyLayoutAdapterTest.*'
```

Expected: `sm90_ss` 和 `sm90_rs` 所有已转换 operand PASS；其中 `sm90_rs` A 的
legacy `replicate=2` 必须成为一个零 carrier bit，并由 GF(2) rank 验证为 2。

- [X] **Step 5: 维护设计文档并提交**

如果最终 Attr 参数或 canonicalization 与设计文档 Section 5/6 不同，先同步修改。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests \
  guoqiao/layout_inference_design.md
git commit -m "feat: add distributed and storage encodings"
```

### M1 Gate

Run:

```bash
cmake --build build --target FriskIR FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
```

Expected: 全部通过；现有 SM90 layout baseline 的新旧枚举无差异；无 `unknown` 被当作 proven 的测试路径。

**执行结果（2026-08-30）：PASS。** `FriskLayoutUnitTests` 运行 6 个 suite、
24 个测试全部通过；lit 4/4 通过；CTest 4/4 通过。`sm90_ss` 与 `sm90_rs`
adapter 对每个 logical point 完成新旧差分，动态 Affine proof 的 `Unknown` 路径有独立
测试且未被接受为 proven。

---

## M2：Storage 纵向切片

### Task 10: 新增 `frisk.layout_view` Storage anchor

**Files:**

- Create: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `test/Dialect/Frisk/layout/layout-view.mlir`

**Interfaces:**

- Produces:

```mlir
%view = frisk.layout_view %source
  {layout = #smem_layout}
  : memref<64x64xf16, #frisk<memory_space Shared>>
    -> memref<64x64xf16, #frisk<memory_space Shared>>
```

`layout` 在推断前可省略，在 materialization 后必须存在。

- [X] **Step 1: 写 parse/verify 红灯测试**

覆盖合法 view、source/result type 不同、global view 绑定 shared layout、非单射 storage map。

```mlir
// expected-error@+1 {{source and result must have identical memref types}}
%bad = "frisk.layout_view"(%arg0) :
  (memref<64x64xf16>) -> memref<32x128xf16>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，Op 尚不存在。

- [X] **Step 2: 定义 Op 和 ViewLike 语义**

ODS 约束：一个 `AnyMemRef` operand、同类型 `AnyMemRef` result、可选 `StorageLayoutAttr`，实现 `ViewLikeOpInterface` 和无副作用 view 语义。C++ verifier 调用 `layout.verifyForType(getResult().getType(), getLoc())`。

```cpp
Value LayoutViewOp::getViewSource() { return getSource(); }
```

- [X] **Step 3: 实现 canonicalization**

只允许以下折叠：

```text
layout_view(layout_view(x, L), L) -> layout_view(x, L)
layout_view(x, identity storage)  -> x，仅当无下游 layout anchor 依赖该 Op
```

两个不同 storage binding 不得折叠或覆盖。

- [X] **Step 4: 运行 IR tests**

Run:

```bash
cmake --build build --target FriskIR check-frisk --parallel 32
```

Expected: layout-view tests PASS，非法 case 输出指定诊断。

- [X] **Step 5: 提交 layout view**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test
git commit -m "feat: add storage layout view anchor"
```

### Task 11: 建立 LayoutVar、Constraint 和 Provenance graph

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`
- Create: `include/Dialect/Frisk/Analysis/LayoutTarget.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutConstraintTest.cpp`

**Interfaces:**

- Produces:

```cpp
using LayoutVarID = uint32_t;
using LayoutConstraintID = uint32_t;
using ProvenanceID = uint32_t;

enum class LayoutState { Uninitialized, CandidateSet, Resolved, Conflict };
enum class ConstraintStrength { Hard, Soft };
enum class ConstraintKind {
  SameLayout,
  TransformLayout,
  RequireEncoding,
  InstructionContract,
  StorageAccess,
  AliasLayout,
  Ownership,
  ResourceLimit,
  Preference
};

struct LayoutCandidate {
  Attribute value;
  ProvenanceID provenance;
  uint64_t stableOrdinal;
};

struct LayoutVar {
  LayoutVarID id;
  LayoutKind kind;
  Type shapedType;
  SmallVector<LayoutCandidate> candidates;
  LayoutState state;
};

enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution;
  uint64_t bytes;
  uint64_t synchronizationCost;
};

class LayoutConstraintGraph {
public:
  LayoutVarID addVariable(LayoutKind kind, Type type, StringRef stableName);
  LayoutConstraintID addConstraint(ConstraintKind kind,
                                   ConstraintStrength strength,
                                   ArrayRef<LayoutVarID> vars,
                                   Operation *source, StringRef rule,
                                   StringRef reason);
  LogicalResult verifyInvariants(Location loc) const;
};

struct CandidateAssignment;

class LayoutTarget {
public:
  virtual ~LayoutTarget() = default;
  virtual void enumerateCandidates(
      const LayoutVar &var,
      SmallVectorImpl<LayoutCandidate> &out) const = 0;
  virtual LogicalResult verifyCandidate(const LayoutVar &var,
                                        Attribute candidate,
                                        Location loc) const = 0;
  virtual FailureOr<CostVector>
  evaluate(const CandidateAssignment &assignment) const = 0;
};
```

- [X] **Step 1: 写 stable-ID 和 invariant 红灯测试**

测试相同逻辑输入在不同插入顺序下，按 `stableName` finalize 后 ID/constraint 顺序一致；重复变量名、无效 var 引用和空 hard constraint 必须失败。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，graph 尚不存在。

- [X] **Step 2: 实现 graph ownership**

Graph 拥有 vars、constraints、provenance；`Operation*` 只用于诊断，不参与排序或 hash。稳定名称来自：

```text
unambiguously encoded fully-qualified ancestor symbol path / block ordinal / operation ordinal / result-or-operand ordinal / layout kind
```

- [X] **Step 3: 实现 provenance chain**

```cpp
struct LayoutProvenance {
  ProvenanceID id;
  std::optional<ProvenanceID> parent;
  Operation *source;
  std::string rule;
  std::string reason;
};
```

提供 `printProvenanceChain(ProvenanceID, raw_ostream&)`，检测 parent cycle。

- [X] **Step 4: 验证确定性和诊断**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutConstraintTest.*'
```

Expected: PASS；两种插入顺序 dump 完全相同。

- [X] **Step 5: 提交 constraint graph**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis unittests
git commit -m "feat: add deterministic layout constraint graph"
```

### Task 12: 实现 Storage constraint collector 和 strict/common propagation

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Modify: `lib/Dialect/Frisk/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutPropagationTest.cpp`
- Create: `test/Transforms/storage-propagation.mlir`

**Interfaces:**

- Consumes the `LayoutConversionEdge` and `EdgeResolutionKind` definitions from Task 11; produces edge enumeration and cost evaluation:

```cpp
class LayoutConstraintBuilder {
public:
  LayoutVarID getOrCreateStorageVar(Value anchor);
  LayoutVarID getOrCreateDistributedVar(Value value);
  LogicalResult require(LayoutVarID var, Attribute encoding,
                        Operation *source, StringRef rule);
  LogicalResult same(LayoutVarID lhs, LayoutVarID rhs,
                     Operation *source, StringRef rule);
};

FailureOr<LayoutConstraintGraph> collectLayoutConstraints(Operation *root,
                                                          LayoutTarget &target);
LogicalResult propagateStrict(LayoutConstraintGraph &graph);
LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph);
```

- [X] **Step 1: 写 storage propagation 红灯测试**

输入包含两个 `layout_view` 和一个 Copy；src 有 storage binding、dst 未绑定。预期 pass 后 dst 获得数学等价 binding。另一个 split case 给 src/dst 不兼容 hard seed，预期诊断包含两条 provenance。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，pass 仍只写 smoke attr。

- [X] **Step 2: 收集 Storage variables 和 constraints**

Collector 规则：

```text
layout_view with layout -> RequireEncoding(hard)
layout_view without layout -> unresolved Storage var
same-source compatible views -> AliasLayout(hard)
copy whole-tile src/dst -> StorageAccess(hard relation) + coalescing Preference
```

Task 12 只接受 whole-tile、静态 shape copy；其他 case 返回明确 unsupported diagnostic。

- [X] **Step 3: 实现 bootstrap SM90 storage candidates**

`SM90LayoutTarget` 首版只生成：linear、transpose、必要 padding、32B/64B/128B XOR swizzle。每个候选必须先通过 Storage verifier；候选按固定枚举序号排序。

这里的 candidate generation 仅用于初始化尚无 hard seed 的变量 domain；它发生在 fixed-point propagation 之前，不执行全局优选。完整候选生成、剪枝和 beam search 由 Task 23 接管。

- [X] **Step 4: 实现单调传播**

Strict 只处理 singleton seed、SameLayout 和 exact AliasLayout。Common 只执行交集/关系投影：

```cpp
while (!worklist.empty()) {
  LayoutConstraintID id = worklist.pop_front();
  ChangeResult changed = applyConstraint(graph, id);
  if (changed == ChangeResult::Change)
    enqueueAdjacentConstraintsInStableOrder(graph, id, worklist);
}
```

空集合进入 Conflict 并输出两条 seed provenance；传播循环内部不得新增 candidate。

- [X] **Step 5: 验证收敛和顺序独立**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutPropagationTest.*'
```

Expected: PASS；随机打乱初始 constraint vector 后 canonical graph dump 不变。

- [X] **Step 6: 提交 storage propagation**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk test unittests
git commit -m "feat: propagate storage layout constraints"
```

### Task 13: 物化 Storage binding 并验证第一条端到端链路

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutVerifier.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp`
- Create: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `test/Transforms/storage-materialization.mlir`
- Create: `test/Transforms/storage-conflict.mlir`
- Create: `unittests/Dialect/Frisk/Layout/LayoutVerifierTest.cpp`

**Interfaces:**

- Produces:

```cpp
struct LayoutSolution {
  DenseMap<LayoutVarID, Attribute> assignments;
  SmallVector<LayoutConversionEdge> conversions;
};

struct BootstrapSolverLimits {
  unsigned maxVariables = 8;
  unsigned maxDomainSize = 4;
};

FailureOr<LayoutSolution> solveBootstrapLayoutGraph(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    BootstrapSolverLimits limits = {});

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target,
                                      Location loc);

LogicalResult materializeLayouts(Operation *root,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution);

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target);
```

- [X] **Step 1: 写未物化和错误物化红灯测试**

覆盖 unresolved layout_view、错误 memory space、非单射 map、alias views 不一致。诊断检查：

```mlir
// expected-error@+1 {{unresolved storage layout for layout variable}}
%v = "frisk.layout_view"(%arg0) : (memref<64x64xf16>) -> memref<64x64xf16>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，verifier/materializer 尚不存在。

- [X] **Step 2: 实现严格受限的 bootstrap resolver**

为使 M2/M3 在完整求解器之前可执行，实现只面向纵向切片的 exhaustive resolver：component 最多 8 个变量、每个 domain 最多 4 个候选；只接受满足全部 hard constraint 的 assignment，并按 candidate stable ordinal 序列决定唯一结果。超过边界直接诊断 `bootstrap layout solver limit exceeded`，不得截断或猜测。

M2 不枚举 conversion/rematerialization；Task 15 扩展该 resolver 处理一个多 consumer conversion edge，Task 23 必须以正式 CostVector/beam solver 替换并删除它。

- [X] **Step 3: 实现 solved graph verifier**

逐 assignment 检查 kind、type、coverage/injectivity、alias 和 Copy relation；物化后二次验证还会枚举静态 logical domain，证明元素 bit range 不重叠且 storage footprint 不超过底层 MemRef strided/affine type 的静态容量。失败信息包含稳定 var 名、candidate、constraint rule 和 provenance chain。

- [X] **Step 4: 实现 Storage materialization**

Materializer 只给现有 `LayoutViewOp` 设置求解出的 `layout` attr；若 solution 中的 Attr 与已有 hard binding 不等价则立即失败，不覆盖。

- [X] **Step 5: 将 pass 变成真正 orchestrator**

`FriskInferLayoutsPass::runOnOperation()` 顺序固定为：

```cpp
auto graph = collectLayoutConstraints(getOperation(), target);
propagateStrict(*graph);
propagateCommonToFixedPoint(*graph);
auto solution = solveBootstrapLayoutGraph(*graph, target);
verifySolvedLayoutGraph(*graph, solution, target, getOperation().getLoc());
materializeLayouts(getOperation(), *graph, solution);
verifyMaterializedLayouts(getOperation(), target);
```

任一步失败调用 `signalPassFailure()`；删除 M0 的 smoke attr。

- [X] **Step 6: 运行端到端测试**

Run:

```bash
cmake --build build --target FriskTransforms FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutVerifierTest.*'
```

Expected: Storage 输入推断并打印 canonical binding；冲突 case 输出两条 provenance；所有测试 PASS。

- [X] **Step 7: 同步设计并提交**

如果实际 pass 阶段边界与设计 Section 8 不同，先更新设计文档。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests \
  guoqiao/layout_inference_design.md
git commit -m "feat: materialize and verify storage layouts"
```

### M2 Gate

Run:

```bash
build/bin/frisk-opt test/Transforms/storage-materialization.mlir \
  -frisk-infer-layouts -verify-each
cmake --build build --target check-frisk FriskLayoutUnitTests --parallel 32
```

Expected: alloc/view/copy Storage slice 完整通过；`LayoutInfer.cpp` 不再是 pass 壳；无 conversion 或 Tensor 逻辑被提前混入。

---

## M3：Distributed Tensor 与 conversion 纵向切片

### 实际实现与原计划的调整记录

本节按已验收代码维护，而不仅标记复选框。下列调整已落实到 Task 14–17 的正文：

| 原计划的简写或假设                                    | 实际实现及原因                                                                                                                                                                |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| convert 两端 encoding 必须不同，但又要求消除 identity | identity 是合法输入；仅拒绝 solver 请求新插入 identity，清理替换须保持精确 SSA 类型。                                                                                         |
| `tensor.transpose`                                  | 本工具链没有该 Op；使用已注册的`linalg.transpose` Tensor DPS，建模输入 use、init 和 result。                                                                                |
| 两种 Storage 布局要求两个 Distributed 候选/转换       | `StorageAccess` 证明 `S(D(h))` 的合法性，不把 coalescing 当硬约束；linear/transpose 两个 store 可以共同布局零转换。另用独立硬编码的活跃 transpose consumer 验证一次转换。 |
| 候选集合从始至终只缩小                                | 收集阶段先由硬 seed 和 target templates 双向投影生成闭包；之后 strict/common propagation 才单调裁剪。                                                                         |
| 逐 Op 改类型、while 所有位置一组相等                  | 按原始 Value 定型并重建 detached ModuleOp；while 的输入 tuple 与 condition/result tuple 分开建模，验证成功后一次提交。                                                        |
| 最小 lowering 可直接视为可执行 GPU 链路               | 当前仅生成测试用 GPU-level IR；Tensor↔thread-vector 桥接尚需 M6 消除，不能宣称 GPU runtime 验收完成。                                                                        |

完成步骤中的 `Expected: FAIL` 是当时测试驱动实现的红灯预期，不是当前验收结果。
本次状态同步不改变 M4–M6 的未完成任务或将其提前标为已实现。

### Task 14: 新增 Tensor carrier 和 `frisk.convert_layout`

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `test/Dialect/Frisk/layout/tile-carriers.mlir`
- Create: `test/Dialect/Frisk/layout/convert-layout.mlir`

**Interfaces:**

- Produces:

```mlir
%tile = frisk.tile_load %view
  : memref<64x64xf16, #frisk<memory_space Shared>>
    -> tensor<64x64xf16>

%converted = frisk.convert_layout %tile
  : tensor<64x64xf16, #src> -> tensor<64x64xf16, #dst>

frisk.tile_store %tile, %view
  : tensor<64x64xf16, #enc>,
    memref<64x64xf16, #frisk<memory_space Shared>>
```

首版 load/store 只表示 whole-tile、静态 shape；切片和动态 offset 在 M4 后扩展。

- [X] **Step 1: 写 Op verifier 红灯测试**

覆盖：load shape/dtype 不匹配、store shape 不匹配、convert source/target shape 或 element type 不同、identity convert。

```mlir
// expected-error@+1 {{source and target must have identical shape and element type}}
%bad = "frisk.convert_layout"(%x) :
  (tensor<64x64xf16, #src>) -> tensor<32x128xf16, #dst>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，新 Ops 尚不存在。

- [X] **Step 2: 定义 carrier Ops**

`TileLoadOp` 具有 MemRead，`TileStoreOp` 具有 MemWrite，`ConvertLayoutOp` 为 pure。Convert verifier 要求 source/target 均有合法 `DistributedEncodingAttr`；identity convert 是合法输入并交给 canonicalizer 删除，materializer 后续拒绝主动生成 identity convert。parser verifier 测试使用 canonicalization 前合法表示时不得崩溃。

- [X] **Step 3: 实现基础 canonicalization**

```text
convert_layout(x, src == dst) -> x
convert_layout(convert_layout(x, A -> B), B -> A) -> x
```

删除 identity/inverse 时同时检查 canonical map 等价与替换值/result 的完整 SSA 类型相等；仅数学映射等价、但 encoding 不同，不足以直接替换。

- [X] **Step 4: 运行 IR tests**

Run:

```bash
cmake --build build --target FriskIR check-frisk --parallel 32
```

Expected: carrier/convert tests PASS。

- [X] **Step 5: 提交 Tensor carrier**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test
git commit -m "feat: add distributed tensor carriers"
```

### Task 15: 收集 Distributed constraints 并支持多 consumer

已实现并通过独立审查；接口和边界见 [Task 15 analysis](m3_task15_analysis.md)。

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Create: `unittests/Dialect/Frisk/Layout/DistributedPropagationTest.cpp`
- Create: `test/Transforms/distributed-propagation.mlir`
- Create: `test/Transforms/multi-consumer-layout.mlir`

实际还新增 `lib/Dialect/Frisk/Analysis/DistributedLayoutConstraints.cpp`、
`include/Dialect/Frisk/Analysis/LayoutRelations.h`、对应 `LayoutRelations.cpp`、
`lib/Dialect/Frisk/Target/SM90/SM90DistributedCandidates.cpp` 和
`test/Transforms/distributed-propagation-errors.mlir`；修改 `LayoutConstraint.h`、
`LayoutSolver.h`、`LayoutVerifier.cpp`、`LayoutInfer.cpp`、`Passes.td` 及相关 CMake。
Op 规则收集、共享关系判断与 SM90 候选枚举分别放在这些文件，未集中堆入原计划列出的文件。

**Interfaces:**

- Extends:

```cpp
LayoutVarID LayoutConstraintBuilder::getOrCreateDistributedVar(Value value);
LayoutVarID LayoutConstraintBuilder::getOrCreateDistributedUse(OpOperand &use);
LogicalResult LayoutConstraintBuilder::convertible(
    LayoutVarID src, LayoutVarID dst, OpOperand &use, bool existing = false);
LogicalResult LayoutConstraintBuilder::transform(
    LayoutVarID src, LayoutVarID dst, Attribute coordinateTransform,
    Operation *source, StringRef rule);
LogicalResult LayoutConstraintBuilder::storageAccess(
    LayoutVarID distributed, LayoutVarID storage, AccessKind access,
    Operation *source, StringRef rule);

FailureOr<LayoutSolution> solveBootstrapLayoutGraph(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    BootstrapSolverLimits limits);
```

- [X] **Step 1: 写双 consumer 红灯测试**

构造一个 `tile_load` 结果被两个 `tile_store` 使用，两个 destination view 分别绑定 linear 和 transpose storage。实际回归检查 producer 和两个 consumer use 均保留 4 个候选，求解选择共同布局、0 个转换，而不是由第一个 consumer 锁定 producer。StorageAccess 不将访存合并度当硬约束。

另一个 `@contract` 用例使用 2x2 非 identity `linalg.transpose`、独立硬编码的 XOR 结果布局和使用原 producer 的 store；函数返回 transpose 结果，使 consumer 在 cleanup 后仍存活，并保留恰好一次输入 use 转换。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，collector 尚未创建 Distributed var。

- [X] **Step 2: 收集基础 Distributed rules**

规则作用于 producer 定义与 consumer 实际期望的 use/slot，而非强制原 producer 与所有消费者布局相同：

```text
tile_load -> StorageAccess(result definition, storage view)
tile_store -> Convertible(producer, consumer use) + StorageAccess(consumer use, storage view)
arith/math elementwise -> 同布局 results；tensor operand use 经 Convertible 满足结果布局
linalg.transpose (Tensor DPS) -> TransformLayout(input use, result, permutation)；init use 经 Convertible 满足 result 布局
existing tensor encoding -> RequireEncoding(hard)
existing convert_layout -> source/target RequireEncoding(hard) + Convertible edge
defined func -> entry argument / function result slot / return use；标量位置不改
arith.constant(DenseElementsAttr), tensor.empty -> Tensor definition binding
```

对于未知 layout-bearing Op，pass 必须报 `operation has no layout constraint model`；Tensor call、external Tensor signature 和非 dense Tensor constant 明确拒绝。SCF if/for/while 的收集同样位于 `DistributedLayoutConstraints.cpp`，随 Task 16 一并实现。

真实定义由 `LayoutVar::value` 标识，独立 consumer 期望由 `use` 标识，函数返回布局由 `functionResult` 标识；已有 result/join slot 可直接作为 consumer 期望，避免冗余变量。已编码 Tensor 的 hard binding 不因插入消费边转换而被覆盖。

- [X] **Step 3: 生成基础 distributed candidates**

SM90 bootstrap 默认模板只有 blocked、lane-striped、warp-striped、fully-replicated 四类，展开并验证后去重；小形状可能得到少于 4 个不同默认候选。候选域还可来自已有合法硬编码和关系投影，不意味着拒绝所有不属于四类模板的显式布局。当前只接受非零 rank、每个逻辑 extent 大于 1 的静态 2 次幂 Tensor tile；extent=1 受 M1 named zero-bit dimension 限制而明确拒绝。

- [X] **Step 4: 扩展 propagation 为双向关系投影**

先把已有硬 seed 投影到闭包；再逐个为尚无候选的变量枚举 target templates，每初始化一个域就继续投影到闭包。生成阶段允许添加候选；随后 strict/common propagation 执行双向支持检查和单调裁剪，避免丢掉其他 consumer 的合法选择。

Transpose 按 permutation/逆 permutation 投影逻辑 bit-row blocks，并按实际目标候选的 output names 判断兼容性。Distributed↔Storage 的 `StorageAccess` 仅检查已验证映射的逻辑域/element type 匹配以证明 `S(D(h))`，不从 Storage 顺序强制推导 Distributed 排列；原 M2 Storage↔Storage copy 仍要求兼容的 storage map。

- [X] **Step 5: 扩展 bootstrap resolver，闭合第一条多 consumer 路径**

在既有 8-variable/4-candidate 上限内，允许每个可转换 consumer edge 枚举 `KeepCommonLayout` 或 `Convert`。目标顺序固定为：先满足 hard constraint，再最小化 conversion 数，最后比较 assignment/edge stable key；本阶段不宣称性能最优，也不枚举 rematerialization。求解结果必须把选中的 conversion 写入 `LayoutSolution::conversions`，供 Task 16 直接消费。

上限精确为每连通分量 8 个变量、每个变量域 4 个候选，超过直接诊断、不截断。已有显式 conversion 不再插入或计为新增转换。`analysis-only` 执行 collect/solve/verify 而不改 IR，`dump-analysis` 输出稳定图和选边。分析支持的 single-CTA topology 范围宽于 Task 17 测试 adapter，分析成功不等于该 adapter 能降低。

- [X] **Step 6: 验证 fixed-point 与多 consumer**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='DistributedPropagationTest.*'
```

Expected: PASS；双 store 保留候选并选择零转换，独立硬编码的活跃 transpose consumer 选择一次 conversion；顺序打乱结果不变。

- [X] **Step 7: 提交 Distributed propagation**

```bash
git add lib/Dialect/Frisk test unittests
git commit -m "feat: propagate distributed layout constraints"
```

### Task 16: 物化 Tensor encoding、SCF 类型和 conversion edge

已实现并通过独立审查；实现与验证记录见 [m3_task16_materialization.md](m3_task16_materialization.md)。
下方 SCF while 规则已按实际 MLIR 的双 tuple 契约修正。

**Files:**

- Modify: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Create: `include/Dialect/Frisk/Transforms/LayoutTypeConverter.h`
- Create: `lib/Dialect/Frisk/Transforms/LayoutTypeConverter.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `test/Transforms/materialize-distributed.mlir`
- Create: `test/Transforms/materialize-scf.mlir`
- Create: `test/Transforms/materialize-conversion.mlir`

实际还修改 `DistributedLayoutConstraints.cpp`、`LayoutSolver.h` 和 `LayoutPropagation.cpp`
以加入 SCF 关系及 `LayoutCollectionMode::RelationsOnly`，新增
`test/Transforms/materialize-scalar-cfg.mlir` 和
`unittests/Dialect/Frisk/Layout/LayoutMaterializationTest.cpp`，同步相关 CMake。

**Interfaces:**

- Produces:

```cpp
class LayoutTypeConverter : public TypeConverter {
public:
  LayoutTypeConverter(const LayoutConstraintGraph &graph,
                      const LayoutSolution &solution);
  FailureOr<RankedTensorType> convertLayoutBearingTensor(Value value) const;
};

LogicalResult materializeDistributedLayouts(
    Operation *root, const LayoutConstraintGraph &graph,
    const LayoutSolution &solution,
    ArrayRef<LayoutConversionEdge> conversions);
```

- [X] **Step 1: 写 SCF 和 conversion 红灯测试**

覆盖 `scf.if` 两个 yield、`scf.for` init/block argument/yield/result、`scf.while` 不同 arity 的两组 tuple，以及多 consumer 的一次转换。实际文本用例在 transpose 输入 use 前插入 conversion，独立 tile_store 继续使用原 producer：

```mlir
// CHECK: %[[CVT:.*]] = frisk.convert_layout %[[PRODUCER]]
// CHECK-NEXT: {{.*}}linalg.transpose ins(%[[CVT]]
// CHECK: frisk.tile_store %[[PRODUCER]],
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，Tensor type 尚未被改写。

- [X] **Step 2: 实现非 region Op 的 type rewrite**

`LayoutTypeConverter` 按原始 SSA Value 查 solution；同一个原始 Tensor Type 的不同 Value 可得到不同布局，因此继承的 Type-only 转换路径明确拒绝 Tensor，不能按 Type 缓存布局。

实际实现仅接受 ModuleOp 根：预检 solution 并快照选中 use 的 owner/operand index，在 detached ModuleOp 中用 `Operation::create` 和 `IRMapping` 重建整个 body。保留属性、native properties、位置、successor、region 和标量参数；同步函数签名以及 DenseElementsAttr 的结果类型。逐 block 保留原操作顺序，调度时检查显式 operand 与嵌套 region 捕获值，使非 dominance 打印顺序的标量 CFG 也可处理；不是原地逐个 setType，也不是只克隆少数结果 Op。

- [X] **Step 3: 实现 SCF 一致重写**

固定规则：

```text
scf.if: each yield operand encoding == corresponding result encoding
scf.for: init == iter_arg == yield == result
scf.while: init == before args == after yield; condition args == after args == results
```

这里的 init/yield/condition 等式指完成选中 use 转换后的 operand 编码，不要求其原 producer 定义预先相等。For 的 iter_arg/result、while 的 after args/results 直接硬相等；需要转换的 init、yield、condition use 分别在相应消费点插入。While 的两组 tuple 可以有不同 arity/type，predicate 不计入 forwarded operands，也不把 before/after 无关槽位强行绑定。不得修改 join 后类型逃避 hard constraint。

- [X] **Step 4: 物化 solver 选中的 conversion edge**

只有 `LayoutSolution` 显式列出的 edge 才插入 `frisk.convert_layout`。source/target 相等时视为 solver/materializer bug 并失败；materializer 不重新比较成本。

物化后先验证 staged MLIR，再用 `RelationsOnly` 重收集关系，把每个定义、实际 operand use、函数结果绑定到 IR 中实际存在的 encoding。此处不枚举候选、不传播、不重新求解，不能通过“假设未来会插入 conversion”掩盖缺失转换；同时保留 M2 alias/copy/capacity 检查。全部成功后只执行一次 module body transfer；任何失败都保留原 IR。提交后原 graph/solution 借用的 Value/Operation/OpOperand 身份失效，不可继续解引用。

- [X] **Step 5: 运行 materialization tests**

Run:

```bash
cmake --build build --target FriskTransforms check-frisk --parallel 32
```

Expected: Tensor/SCF/conversion tests PASS，`-verify-each` 无错误。

- [X] **Step 6: 提交 Distributed materialization**

```bash
git add include/Dialect/Frisk/Transforms lib/Dialect/Frisk/Transforms test
git commit -m "feat: materialize distributed layouts and conversions"
```

### Task 17: 实现基础 conversion cleanup 和最小测试 lowering adapter

**Files:**

- Create: `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `lib/Conversion/FriskLayoutToGPU/CMakeLists.txt`
- Create: `lib/Conversion/FriskLayoutToGPU/TestLayoutConversionLowering.cpp`
- Modify: `lib/Conversion/CMakeLists.txt`
- Create: `test/Transforms/optimize-layout-conversions.mlir`
- Create: `test/Conversion/FriskLayoutToGPU/warp-shuffle.mlir`
- Create: `test/Conversion/FriskLayoutToGPU/shared-exchange.mlir`

实际还新增 `include/Conversion/FriskLayoutToGPU/LayoutConversionPlan.h` 及
`lib/Conversion/FriskLayoutToGPU/LayoutConversionPlan.cpp`，让 emitter 与单元测试共享
owner/payload plan；新增 `LayoutConversionPlanTest.cpp` 及 register-selection、
replica-locality、scratch-budget、existing-scratch、unaccounted-scratch、unsupported
测试文件。清理模式复用 `FriskLayoutOps.cpp` canonicalization，接线还修改 `Passes.h`、
`lib/CMakeLists.txt`、`tools/frisk-opt/CMakeLists.txt` 和 unit-test CMake。

**Interfaces:**

- Produces:

```cpp
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass();
std::unique_ptr<Pass> createTestLowerLayoutConversionsPass();
```

已实现。详细 adapter 边界和 M6 Tensor↔thread-vector 桥接职责见 [Task 17 implementation](m3_task17_conversions.md)。测试 lowering 仅支持静态、单 CTA、BitLinear→BitLinear redistribution：同线程优先 register extraction，同 warp 优先 shuffle，跨 warp 使用真实 workgroup attribution + barrier exchange。两端 execution topology 必须匹配，lane=32，gpu.func 单 entry block 且 known_block_size 匹配；不支持的 scope/topology/dtype 明确诊断。

公共 SSA 保持原始精确 RankedTensor 类型，内部使用 `builtin.unrealized_conversion_cast`
连接每线程 vector；该表示桥接并非可执行的 tile_load/store lowering，M6 仍需接管和消除。
此 adapter 不接受嵌套 SCF 转换，不支持循环 scratch 复用、跨 CTA/cluster 或 sub-byte。
额外明确限制为至多 1024 threads、两端各至多 256 registers/thread、逻辑 volume≤65536、
source+destination carrier visits≤262144、保守 shuffle-word count≤65536；均是 adapter 限额。

- [X] **Step 1: 写 cleanup/lowering 红灯测试**

覆盖 identity、A→B→A、相邻 A→B→C 合并，一个 lane permutation conversion 产生 `gpu.shuffle`，以及一个跨 warp permutation 产生 workgroup scratch store/barrier/load。动态 encoded tensor 在正常 IR 路径由 M1 verifier 先拒绝；programmatic planner negative 单独验证动态输入。可到达测试 adapter 的 unsupported 情况使用 `test lowering requires a static single-CTA redistribution` 前缀，未禁用 verifier。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，passes 尚不存在。

- [X] **Step 2: 实现 type-safe cleanup patterns**

只实现数学上可证明的：identity elimination、相邻 conversion compose、inverse pair elimination。Pattern 必须调用 canonical map equality，不比较 Attr 指针。

- [X] **Step 3: 实现单 warp redistribution 计划**

计算：

```text
R = rightInverse(D_src) compose D_dst
```

按 named input 解码每个目标 carrier 对应的 source lane/register，并验证 source coverage/bounds；replica 优先同 thread/warp。所有 source-register candidate shuffle 均在全 warp 无条件执行，之后按 destination 请求选择 register。i8/i16/i32/i64/f16/bf16/f32/f64 经 bit-preserving i32 words 传输；64-bit 拆分/重组。

- [X] **Step 4: 实现单 CTA shared exchange**

先用 canonical maps 为每个 live logical element证明唯一 source owner 和所有 destination owners，再以 logical linear index 分配 scratch slot：source owner 写入，执行一次 `gpu.barrier`，destination owner 读取。scratch 字节数按 element bit width、tile volume 和 alignment 精确计算；sub-byte 暂不支持。global writer election 与 local replica preference 分离。缺失 source/writer/slot proof 失败报告 logical coordinate；容量/类型限制给出具体原因。默认 49152-byte adapter budget（非 SM90 最大值）计入原 dtype byte size、既有 attribution、padding 和全部新增 scratch；dynamic/non-attributed workgroup storage 拒绝。

- [X] **Step 5: 注册 passes 并运行测试**

Run:

```bash
cmake --build build --target FriskTransforms FriskLayoutToGPU FriskLayoutUnitTests check-frisk --parallel 32
```

Expected: cleanup、single-warp 和 shared-exchange FileCheck PASS；unsupported case 使用预期诊断失败；生成 IR 通过 `-verify-each`。

- [X] **Step 6: 提交 conversion MVP**

```bash
git add include/Dialect/Frisk/Transforms lib/Conversion \
  lib/Dialect/Frisk/Transforms test
git commit -m "feat: optimize and test-lower layout conversions"
```

### M3 Gate

状态：通过（2026-09-10）。27 lit、74 unit、4 CTest 全部通过；活跃 2x2 transpose
consumer 保留恰好一个选定 conversion，重复 infer/cleanup 输出逐字一致。
详细范围、命令和限制见 [M3 execution notes](m3_execution_notes.md)。

Run:

```bash
cmake --build build --target check-frisk FriskLayoutUnitTests frisk_attr_test \
  frisk_reduce_layout_test frisk_layout_pass_test frisk_memory_effect_test --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
build/bin/frisk-opt test/Transforms/multi-consumer-layout.mlir \
  -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each
```

以上命令从已配置的主工作区 build 运行；`--split-input-file` 可选。该 lit 用例还
检查活跃 consumer 及再次 infer/cleanup 后的输出逐字一致。构建命令同时刷新 CTest
的四个 legacy test target；`check-frisk` 不代替执行独立的 layout unit 二进制。

Expected: 多 consumer IR 合法；共同布局或 conversion 由 solution 明确决定；SCF 类型一致；单 warp 与单 CTA shared-exchange conversion 可以 lower，非支持路径明确失败。

---

## M4：完整约束传播与旧 IR 迁移

### 2026-09-12 实现审计与阶段划分

本轮读 Frisk `b120d06`、TileLang `5e149e3`、Triton `42c5e89` 的源码与相关测试；完整 SHA、永久链接及比较结论见[比较文档](./layout_inference_strategy_comparison.md)。TileLang 另对照旧 `6623b12`，未改用户的 TileLang checkout。

| 阶段     | Task   | 本阶段目的                                | 本轮状态                                                                         |
| -------- | ------ | ----------------------------------------- | -------------------------------------------------------------------------------- |
| 第一阶段 | 18     | 坐标 alias、显式 region graph、可审计收敛 | 用户已确认，已编码；最终 Gate 见 §18.9                                          |
| 第二阶段 | 19     | Copy/Fill/Parallel 进入新接口与约束       | 已按确认契约实现；独立复审及 Gate 记录见 §19.9                                  |
| 第三阶段 | 20–21 | Tensor MMA/GEMM 与 Reduce 规则适配        | Task 20 已通过 Gate；Task 21 受限实现与回归完成，完整链成功验收待解决，见 §21.8 |
| 第四阶段 | 22     | normalization、旧生产逻辑退役和集成       | 已按方案 A 实现，最终验收见 §22.13；Task 21 完整链验收缺口仍保留                |

审计起点结论与具体修正（以下 Frisk 缺口描述固定于 `b120d06`，实施后状态见 §18.9）：

| 项目                  | 当前实际实现/上游证据                                                                                                            | 决定                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Frisk storage alias   | 只沿 layout_view 找 root；邻接 AliasLayout 链、候选直接复制、Attr==                                                              | Task 18 改坐标证明；不能声称 cast/subview 已支持                                                        |
| Frisk SCF             | M3 已有 if/for/while，while 输入/输出两 tuple 已分开                                                                             | 保留语义并显式记录边，不重做或弱化为同槽位强相等                                                        |
| Frisk propagation     | 候选增长 closure 后执行多轮全图扫描，无可核验 pop/deletion 统计                                                                  | 分离有界候选准备与单调删减 worklist                                                                     |
| Frisk actual verifier | 从实际 IR 重置 singleton 并核验；事务提交已存在                                                                                  | 保留无推测、无 future conversion 的验证边界；纠正 RelationsOnly 提前复制 alias seed 的模式契约          |
| TileLang alias        | bit-aware reshape、same-data 分组、widening 同线程/连续槽位证明旧版就已有                                                        | 借鉴 storage bits 不变量，但不把同存储等同坐标恒等；dtype reinterpret 本阶段仍拒绝                      |
| TileLang 更新         | PartialFragment、reducer unset→narrow→wide、dst-steering、epoch 语境扩展、attempt 隔离；spill-aware 默认评分和 opt-in I/O 模型 | Op/reducer 规则由 Task 19–21 审阅适配，cost policy 留给 M5/Task 23；Task 18 不引入 reducer/cost solver |
| Triton 当前           | while 正向双 tuple；物理 BufferRegion 用于 effect/ConSan/同步；PaddedSharedEncoding 已组合 padding/linear                        | 不能把这些单独宣称为 Frisk 创新；比较统一约束的范围、证明与失败契约                                     |
| 旧 Frisk Op           | 旧局部方法还在，但新 pass 尚未完整使用 Op interface models                                                                       | 迁移按当前语义与独立 oracle；不能只搬旧 args 下标/映射代码                                              |

本轮基线验证命令（均 exit 0）：

```bash
cmake --build build --target check-frisk FriskLayoutUnitTests \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
```

结果：27/27 lit、74/74 unit（12 suites）、4/4 CTest。它们确认审计起点可回归，不包含尚未编写的 Task 18 用例；没有上游运行差分或性能结论。

验证过程备注：最终复跑时，一次与重新构建并行启动的 CTest 报 `permission denied / BAD_COMMAND`，4 个进程均未进入测试；检查日志与可执行权限后，在相关目标链接完成、经授权的沙箱外环境重跑，4/4 通过。未为此修改代码；不将该次启动失败当作功能断言失败，也不省略其验证记录。文档检查另核对了 65 个当前源码/本地链接及 1 个历史源码链接，`git diff --check` 通过。

### Task 18: 完成 alias/view 与 region graph

**状态：设计 v1 已获用户确认（2026-09-12），实现、独立复审与 Gate 已完成；2026-09-16 按用户授权提交并合入本地 main，未推送。实际差异、已知限制及证据见 §18.9，集成记录见 §18.10。**

- [X] 固定三方源码版本并完成实现审计，修正比较文档。
- [X] 写明 Task 18 的支持边界、地址契约、候选/传播算法、失败语义与验收矩阵。
- [X] 用户确认以下设计与支持边界。
- [X] 红灯测试 → 实现 → 独立评审 → Gate。
- [X] 2026-09-16 经用户授权提交并本地集成；远端推送不在本次授权范围。

#### 18.1 范围与设计选择

采用“坐标关系驱动的别名分析”。支持同元素类型/同 memory space 的静态 `memref.cast`、`memref.subview`（非零 offset、正 stride、降秩、嵌套），并保留 `layout_view` 的纯绑定语义。这里的“静态”是所需的 root shape、路径 metadata、边界和单射性都能证明，不只是 result type 看起来静态。

第一阶段不支持动态 offset/size/stride、dtype reinterpret、memory-space cast、未知 view 语义、非正 stride、无法证明的 root alias/容量；这些不是可忽略边，而应在相关布局链上给出明确诊断。不扩展任意 memref.reshape、非结构化 CFG、跨过程 alias 或 memref loop-carried alias 分析；不同函数实参之间的 MayAlias 不升级为 MustAlias，也不据 root ID 不同断言 NoAlias。Region 部分保持 M3 的 Tensor SCF 范围。

不扩展 GEMM/Reduce、成本求解、conversion hoist/remat 或完整硬件 lowering；不放宽现有 Copy/TileLoad/TileStore 的“直接 layout_view operand + whole tile”支持边界。静态 Tensor shape 与 bootstrap solver 大小限制继续生效。

三种方案比较：

| 方案                                                                        | 问题/取舍                                                                                                    | 决定                                                       |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| 同 root 直接 union/Attr==                                                   | offset、stride、rank reduction 后坐标不同；不同文本可语义相等；局部重叠不具备等值传递性                      | 拒绝                                                       |
| 引入隐藏 canonical-root LayoutVar，所有 view 只连 root                      | 关系简洁，但没有显式 root binding 时，隐藏 root assignment 无法由实际物化 IR 独立取得，还需新增物化/验证规则 | 暂不引入                                                   |
| 保存每个实际 view→root 坐标映射，对同 root 的实际 views 做完整逐对兼容检查 | 多一些 pair constraints，但不用隐藏 assignment；可复用现有有限域与 actual-only verifier                      | **本阶段推荐**；候选生成仍可从有限 root 模板前向投影 |

逐对检查不等于每次 worklist pop 枚举笛卡尔积：先缓存每个 endpoint/candidate 的 root 坐标与物理 bit 区间索引，再按共同坐标/排序区间检查。图构建须记录 pair 数与证明规模；第一阶段沿用小 component 限制，不声称大图复杂度已经解决。

#### 18.2 坐标、地址原点与容量契约

规范化记录：

```text
view v:
  root identity + stable root key
  T_v: view logical coordinates -> root logical coordinates
  root static descriptor (shape / offset / strides / element bits / memory space)
  root accessible span proof + view live domain + provenance
```

坐标变换由 Op 路径组合，不从 result descriptor 的 offset/strides 反解根坐标：

- cast：同一 logical element 的恒等变换；可从静态 source 恢复中间 cast 隐去的事实，不能用动态 source 猜静态范围。实际 layout_view endpoint 仍须具有静态 ranked shape；允许中间 cast 丢信息、后续静态 endpoint 恢复，不扩展当前属性 verifier 为动态 shape 的值上下文验证器。
- subview：保留维度满足 `q_i = offset_i + stride_i * x_j`；被降掉的维度固定为 `offset_i`。
- 使用 MLIR Op 实际 rank-reduction mask；不能“删除所有 size=1 的维度”，因为这些维度也可能保留。
- 逐层证明 `0 <= offset`、`offset + (size-1)*stride < sourceExtent`，检查整数溢出、正 live extent；嵌套时组合 T。
- root 的静态 descriptor 必须可证明单射，拒绝 `strides=[1,1]` 等重叠歧义。未知 view 不透明穿透；ViewLike interface 只定位 source，不自动提供该 Op 的坐标语义。

**统一地址契约（已确认并接入 Task 18）**：Storage map 输出相对于共同 root 的 **aligned pointer** 的 `(byte_offset, bit_offset)`。物理区间为：

```text
A_v(x) = 8 * byte_offset_v(x) + bit_offset_v(x)
I_v(x) = [A_v(x), A_v(x) + elementBits)
```

subview 的 descriptor offset 只用于坐标路径证明，不能在已 root-relative 的 map 输出上再加一次。显式 root binding 的 map 也使用同一原点。无 binding 时，root 的默认 linear 模板使用 `elementBits * (rootOffset + Σ rootStride_i*q_i)`；其他 root 模板必须声明相同地址原点、证明容量，不能沿用“忽略 root offset”行为。布局绑定表达新的物理布局契约，未绑定 root descriptor 用于确定 logical domain/可访问 span，并不强迫所有显式 swizzle 等于原线性地址公式。

root nonzero offset 的前缀不是可随意使用的空间：可访问边界采用 `[elementBits*rootOffset, elementBits*(rootOffset + Σ(extent_i-1)*stride_i + 1))`，所有实际 map 区间须落在其中。对静态 alloc/alloca/get_global 追到真实根；外部参数只使用类型保证的该区间，不能推断更大的实际 allocation。中间 cast/subview 的 type span 不得扩大 root 的保证范围；binding 不改变 allocation 大小。

容量是可访问地址上界/区间，不是元素总字节数；带 padding 时二者不同。单独通过 result memref type 的容量检查不够，还须证明 view live domain 位于 root 内。packed dtype 按 bit 区间计算，byte 容量向上取整，不能丢弃 bit offset。

**具体例子（已由 `layout-alias.mlir` 和 `StorageAliasIntegrationTest` 覆盖）：**

```text
root: 4x8xi32，offset=0，strides=[8,1]，linear S_r(r,c)=32r+4c bytes
view: subview offsets=[1,2], sizes=[2,3], strides=[1,2]
T_v(i,j)=(i+1, 2j+2)
S_v(i,j)=32i+8j+40 bytes
地址：{40,48,56,72,80,88}；每个元素占 4 bytes，root 容量 128 bytes。
```

同一 root 的另一个视图若也覆盖 `(1,2)`，该点必须为 byte 40；若它把另一个 root 元素放到 byte 40，则是非预期碰撞，也要拒绝。这个 view 的 map 与 root 的 map 文本不同是正常的，直接复制 `32i+4j` 则错误。

若 root 另有 offset=3，其默认 linear 地址整体加 12 bytes，root 可访问区间从 byte 12 开始；本例 view 首元素为 byte 52，不能再加 subview descriptor offset。

#### 18.3 AliasLayout 的兼容性与图不变量

对同一可证明 root 的任意两个实际 view 候选：

1. 若 `T_a(x)==T_b(y)`，两者必须有相同物理 bit 起点及元素区间。
2. 若 root 坐标不同，两个物理元素区间不得相交，即使 logical views 不相交。
3. 每个候选本身须通过 live-domain injectivity、bit/output bounds 和 root 容量验证。
4. alignment/vector-granularity 是独立保证：按 root pointer 的真实对齐依据、投影后地址和实际使用条件共同验证；offset 可整除不等于绝对地址对齐。可信依据限定为 allocation/global 的显式对齐、已验证 ABI/参数前置契约，以及显式 whole-root binding 声明的前置契约（须记录 provenance，不能称运行时已证明）。没有依据不能推导更强对齐，只能降低推断保证或拒绝相应向量候选。child 的对齐要求不是 root 对齐证据；不得仅复制较强保证或因地址相同就认定所有 metadata 相等。

所有同 root pairs 都覆盖，不能保留旧的邻接链。反例：`A∩B=∅, B∩C=∅, A∩C≠∅`，只连 A–B–C 会漏检 A/C 冲突；仅跳过逻辑不相交 pairs 还会漏检不同元素的物理碰撞。

逐对约束验证的是**实际绑定视图覆盖域的联合一致性**，不是宣称未被任何 view 覆盖的 root 元素已经有唯一布局。若存在 root 的 whole-domain binding，它自然作为一个实际 endpoint 约束全部子视图；若不存在，不在 verifier 中求一个隐藏 root 解。此限制不授权对未绑定 root 的普通访问进行新布局 lowering。

Alias payload 显式绑定各 endpoint 的 T/root/domain。现有 finalize 会排序 AliasLayout endpoints：新实现必须连 payload 一起 remap/交换，不能单独排序 vars。stable key 包含 root、endpoint、canonical transform；不得依赖 Value 指针排序。graph invariant 检查两端 storage kind、dtype/space、rank/domain、同 root 证据及有效的 var/constraint IDs。

求解与物化核验共享**关系语义/证明实现**，但 actual verifier 的输入必须独立从当前 IR 重建；不能信任此前候选缓存或 solution 的未来转换计划。

#### 18.4 有限候选准备、XOR 投影与反向规则

先固定候选 origins，再冻结域，最后执行删减传播。禁止把任意 affine 正反向组合直接塞进现有增长 closure：

- origins 来自显式 seeds 及每个 root/必要 endpoint 一次性的 target enumeration；用 stable origin ID 去重。
- root 模板可按 `S_v = S_r ∘ T_v` 投影到实际 endpoint。显式 whole-root seed 和可证明全域可逆的 seed 也可成为 root origin，保留 M2 同域 alias 传播能力。
- 切片不是满域双射。child seed 只能在既定有限候选中筛选 parent；不能“求逆”得到未观测元素的唯一布局。
- 从 child 向其他 view 转移只有在后者覆盖域完全包含在该 child 的已知域且逆像唯一时才允许；否则不生成扩展。每个 `(origin, endpoint)` 最多生成一次规范投影，不经循环反复拼表达式。
- 显式 seeds 本身保留为候选，兼容性交给 hard constraints；不能只因不属于默认模板就悄悄替换。无可用有限扩展候选时诊断“支持的候选域内无解”，不宣称数学上所有布局无解。
- 新 alias 投影不得使已有 Tensor 变换 closure 无界；保持 M3 有限变换族/去重，并为新增关系明确 origin 上界。无法证明有限闭包的新增变换不在本阶段接入。

非零 offset/非二次幂 stride 与 XOR 不能直接按 homogeneous GF(2) 合成：整数加法可能进位。例如 input bit 可写为 `(x floordiv 2^k) mod 2`，XOR output bit 是所选 bits 之和 mod 2。推荐把受支持的有限 bit-linear 映射精确转换为 affine floorDiv/mod 表达式，再与 T 组合，最后验证实际 live domain；零偏移纯 bit-linear 可保持快速路径。

限制 bit width、表达式规模、算术溢出和 proof domain；沿用当前 65536 live-point 枚举上界作为首阶段精确验证界，超限返回 Unknown/unsupported，不增加截断采样“证明”。ProductLayoutMap 不是任意 composition wrapper，本阶段不扩展任意 affine×GF(2) 组合支持。无法表示的 proposal 可丢弃；显式 hard seed 无法证明或最终没有可证候选时必须诊断。

候选兼容缓存按 graph/endpoint/origin 作用域管理，IR 改写后失效。RelationsOnly **不调用 target enumeration、不复制 alias seed、不投影、不求候选闭包**；只记录真实 bindings/RequireEncoding 和关系，随后用实际 IR assignment 设置 singleton 验证。

#### 18.5 Region graph：保留 M3 语义，显式记录流向

不再采用旧草案的固定 `incoming/blockArgument/yielded/result` 四字段一组结构。使用带 kind 的边，引用现有约束，避免重复构造 Convertible：

| SCF 情况           | 需要保留的映射                                                               |
| ------------------ | ---------------------------------------------------------------------------- |
| if                 | 每个 branch yield use → 对应 result slot；没有伪造 block argument           |
| for                | init use → iterarg；iterarg 与 result SameLayout；yield use → carried slot |
| while 输入 tuple I | init use → before arg；after yield use → before arg                        |
| while 输出 tuple O | condition forwarded use → after arg/result；after arg 与 result SameLayout  |
| while predicate    | i1 条件不是 forwarded tensor，不建立布局边                                   |

I/O tuple 的 arity/type 可以不同，严禁按相同 index 把 before 与 after 强行等值。保留 zero-result、scalar condition 及已有嵌套 SCF 行为。

拟新增接口示意（名字/字段可在批准后的实现评审中细化，语义不得弱化）：

```cpp
enum class RegionLayoutEdgeKind {
  IfYield, ForInit, ForBackedge, ForResult,
  WhileInit, WhileBackedge, WhileCondition, WhileResult
};
struct RegionLayoutEdge {
  RegionLayoutEdgeKind kind;
  LayoutVarID source;
  LayoutVarID target;
  LayoutConstraintID constraint;
  Operation *owner;
  OpOperand *use; // 槽位等值边允许为空；转换边必须是实际 operand use。
  unsigned slot;
  std::string stableKey;
};
```

collector 建边与约束一次完成；finalize 后同步 remap IDs。graph dump 公开 kind/slot/stable key/constraint，pointer 只用于有效期内的定位。转换仍由 M3 solution 授权，在实际 init/yield/condition/consumer use 物化；本任务不增加跨 side effects 移动或隐式转换。

#### 18.6 单调删减 worklist 与可核验统计

graph finalize、候选准备完成后冻结 domains，建立各变量去重且按稳定 constraint ID 排序的 hard adjacency。stable FIFO 初始放入本阶段每条需执行 hard constraint 一次；inQueue 去重。pop 时先清 inQueue，只有实际 domain 缩小时才唤醒受影响变量的相邻约束。

Strict/Common 可共享 kernel，但各自保留现有关系启用条件；分别报告统计，避免把两次初始队列当成一次。Strict 的 adjacency/degree 必须包含本阶段所有潜在可执行关系，不能因当前尚无 singleton 就删除边；singleton 启用条件在 pop 时判断，后续 domain change 仍会唤醒它。删减阶段不得再 enumerate/project/addCandidate。每阶段定义：

```text
N0 = 初始候选总数
Nf = 最终候选总数（失败时为已删减后的实际状态）
C0 = 初始入队的约束数
deletedCandidates = N0 - Nf <= N0
domainChanges = 实际缩域事件数
popUpperBound = C0 + Σ_events degree(changedVar)
queuePops <= popUpperBound
静态宽松界 = C0 + Σ_v |D_v(initial)| * degree(v)
```

一次删除多个候选，deletedCandidates 按实际数量累计，domainChanges 按真实变量变化事件计。冲突路径也先计数再返回。degree 只计算该阶段能够被调度的硬约束；soft preference 不混入一致性证明。

统计还包含最终候选数、enqueue/最大队列长度、按 var 的 change 次数，以及候选准备 origins/projected candidates/proof evaluations（与删减统计分开）。测试核对逐项账本和静态界，不用“最多循环 N 次”掩盖振荡。确定性比较以稳定逻辑身份和相同输入域为前提，不要求不同语义/不同图有相同统计。

#### 18.7 验收矩阵（具体执行记录见 §18.9）

| 类别                    | 最小正例                                                                                           | 必须拒绝/揭露的反例                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| cast/view 链            | 同 dtype 静态 cast、nested layout_view；中间 cast 丢信息后在静态 endpoint 恢复                     | 动态 source 无法证明、动态 shape endpoint、未知 ViewLike、dtype/memory-space reinterpret                               |
| subview                 | 上述 4×8 例子；嵌套正 stride、降秩；size=1 维度保留/删除分别覆盖                                  | 越 root 边界、非正 stride、零 live extent、算术溢出、动态 offset/size/stride                                           |
| 地址原点/容量           | root offset=3 首元素 byte 52；root 容量大于 view span；packed bit 区间                             | offset 重复叠加、利用 root 前缀、仅凭扩大后的 view type 通过容量检查                                                   |
| 对齐                    | root 对齐契约 + 投影地址共同支持较弱的 view 对齐                                                   | 仅凭相对 offset 整除推断更强绝对对齐；child 要求倒推 root 保证                                                         |
| alias 兼容              | 不同 Attr 文本但同实际地址；不同视图的部分重叠                                                     | A/B/C 非邻接冲突；不同 logical 元素物理区间碰撞；非单射 root                                                           |
| 投影与反向              | XOR+offset=1/stride，逐点独立预期；切片兼容多个 parent 候选不误锁定                                | 把 offset 当 XOR；逆切片伪造全域；表达式/证明域超限却接受                                                              |
| candidate/RelationsOnly | counting target 证明 actual verifier 不枚举；带 bound/unbound storage aliases                      | seed 复制或 projection 偷渡到 RelationsOnly；依赖未物化 root/未来 conversion 通过                                      |
| SCF                     | nested if/for/while；不同 arity/type I/O tuples、zero result；实际 carried/result 缩域后回边重调度 | predicate 进入布局域、before/after 误强绑、漏掉 yield/condition 约束；不能为制造缩域而把 Convertible 偷换为 SameLayout |
| 统计/确定性             | 同一稳定图不同插入顺序，assignment/IR/统计相同；逐次删除/pop 可对账                                | 超删除界、伪造 change、全图扫描计成依赖 pop、固定 N 次掩盖未收敛                                                       |
| 独立核验/事务性         | 物化后重收集实际绑定，所有 pair/region constraints 成立                                            | 篡改一个 view 地址被 actual verifier 拒绝；任意失败保持原 IR 逐字不变                                                  |

证明失败需区分 Disproven（首个 view/root 坐标、bit 区间或边界反例）和 Unknown（具体动态项/表示限制/预算）；均不得当作硬关系成功。诊断包含 root/view stable key、源位置和 provenance 链，不以 assertion 处理用户 IR。

#### 18.8 预期文件与实施顺序（确认后执行）

**Files（原设计清单；实际职责细分见 §18.9）：**

- Create: `include/Dialect/Frisk/Analysis/LayoutAliasAnalysis.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutAliasAnalysis.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`、`LayoutSolver.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`、`LayoutRelations.cpp`、`LayoutPropagation.cpp`、`LayoutVerifier.cpp`、`DistributedLayoutConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`（root-relative 候选适配，不引入 cost solver）
- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`、`FriskLayoutOps.td` 及对应 verifier/实现（公共地址契约）
- Modify: Analysis/unit CMake；`guoqiao/layout_inference_design.md`、本计划和比较文档
- Create: `unittests/Dialect/Frisk/Layout/AliasRegionTest.cpp`
- Create: `test/Transforms/layout-alias.mlir`、`layout-scf-fixed-point.mlir`

- [X] **Step 1：确认设计后建立隔离实现工作区，编写红灯测试。** 先复用 M3 regression，再添加 offset/stride/rank-reduction 与真实 region 边用例，确认因能力缺失失败。
- [X] **Step 2：实现坐标规范化与 AliasLayout 证明。** 定义 root-relative 契约、容量与 pair invariants，优先支持显式 bindings 的独立核验，再实现有限候选投影。
- [X] **Step 3：实现显式 region edges。** 复用 M3 SCF 关系，不重复建转换边；补全 dump/finalize remap/invariant 测试。
- [X] **Step 4：替换传播调度并公开统计。** 冻结候选域后执行 stable worklist；验证两阶段删除/pop 上界、冲突计数与确定性。
- [X] **Step 5：接入 actual-only verifier 与原子物化。** 不允许用 speculative root/候选修补实际 IR；运行篡改布局和 rollback 用例。
- [X] **Step 6：完整 Gate 和独立审查。** 除下述命令外，逐项完成 §18.7；将实测数量、实现限制和偏离设计的理由同步两份长期文档。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
git diff --check
```

Expected：新增 alias/region 测试及全部 M0–M3 regression PASS；实际 IR 核验通过，失败路径不改 IR；统计满足推导界。不以旧 27/74/4 数量充当新增功能验收。

- [X] **Step 7：按授权提交与集成。** 2026-09-16 提交代码、测试及同步文档并合入本地 main；不包含用户独立 README，未推送远端。

#### 18.9 确认后的实现记录（2026-09-12）

工作树：`.worktrees/m4-task18`，分支 `feature/m4-task18`，起点 `b120d06400a14a703a44dac1a37a0b38d8110935`。此前两份审计文档的修改复制到隔离树继续维护，属于本次交付文档；主工作树原文件保留不覆盖。用户独立 README 未复制、未修改，不纳入任务提交。

实际分工与原设计的细化：

- `LayoutAliasAnalysis.{h,cpp}` 负责静态路径规范化、root span、精确 bit 区间、XOR 的整数 carry 投影、包含域内的唯一逆像。`StorageAliasCandidates.cpp` 独立承载有限 origin 初始化，避免继续扩大 Tensor 增长 closure。
- `LayoutVar` 持有 endpoint 对应的 T/root，finalize 排序只 remap endpoint ID，不存在变换与端点脱配。`LayoutRelations.cpp` 复用 endpoint footprint 与 pair proof 缓存；actual verifier 重建新图，缓存不跨 IR 生命周期。
- root 默认 linear 模板在 alias 初始化适配层按 descriptor 构造，先验证 root proof budget，再调用 target 枚举，避免超预算大 root 进入模板算术；未修改 SM90 target 的公开枚举接口。其他 target 模板只有在满足 root 原点、容量和元数据证明时保留。临时 proposal 不是图中的隐藏 root assignment。
- `RegionLayoutEdge` 引用已有 hard relation，Strict/Common 共享 stable FIFO kernel。两阶段统计分开；origin/投影/footprint/pair-proof 次数单独输出。
- vector granularity 的安全契约补充为：1 是 scalar/packed 基线；较大值须满足字节元素、对齐的 scalar 子块，或完整最内层逻辑行的连续向量块。偏移、stride 或尾部破坏保证时，推断投影逐级减小 vector；显式 hard binding 不允许静默降级。alignment 仍表示 root pointer，并在 dump 中记录 allocation/global 依据或 whole-root 声明前置条件（不是运行时证明）。
- 原矩阵“回边确实多次缩域”需要语义澄清：M3 的合法单 CTA Tensor 编码之间允许 Convertible，真实回边不会仅因编码不同而删除候选。本次测试真实 for 图的 carried/result 缩域后回边重新入队；另用 SameLayout 链独立核对多次缩域事件。没有将回边改成 SameLayout，也未增加 conversion 移动权限。

新增测试分为 `LayoutAliasAnalysisTest`、`StorageAliasIntegrationTest`、`AliasRegionTest`，以及 `layout-alias.mlir`、`layout-scf-fixed-point.mlir`。旧 `storage-conflict.mlir` 的前两例原先同时违反容量与 alias；改用容量内 reversed map 专测 alias 冲突，保留第三例 14 bytes/8 bytes 容量错误，并检查新增反例坐标诊断。

TDD/评审证据：先复现切片地址错误 `0/8/20`（应为 `40/56/88`）、非邻接冲突漏检、RelationsOnly seed 偷渡、缺失 region dump/统计；随后补齐。独立评审另发现向量保证未按实际地址验证、alignment provenance 缺失和 public graph 自身 ID 未验证；均以反例/检查补强。两项分模块复审及一次整体复审完成；整体复审保留以下跨 pass 生命周期限制，无未处理的 Task 18 范围内阻断项。

保守边界保持：65536 live points、最多 4096 affine expression nodes、受限 bit width/matrix 大小、signed 64-bit 可表达的 root bit span；预算溢出/不支持表示为 Unknown，hard seed 不得通过。ProductLayoutMap 的一般 alias 投影、动态路径、reinterpret、跨过程/region-carried memref alias 均未扩展。未运行上游运行差分或 GPU 性能实验，创新表述仍是可核验的体系差异，不是性能领先结论。

**已核验的跨 pass 限制：** whole-root binding 声明的对齐前置条件依赖该 binding 仍存在于 actual IR。`layout_view` 保持 Pure，现有 canonicalize/DCE 可能删除未被使用的 root binding；若 sibling child 仍使用该对齐保证，`infer → canonicalize → infer` 会因缺少 root evidence 保守报错，而 `infer → infer` 通过。新增 `layout-alias-contract-lifetime.mlir` 固定此行为。需要跨任意规范化保留对齐时，应使用 allocation/global 显式对齐依据或保留实际 whole-root 契约；Task 22 需设计 durable contract。Task 18 不伪造副作用、不放宽 actual-only verifier，也不静默修改显式 hard binding。

回归维护记录：新增 own-ID 不变量后，M3 的乱序构图测试需在 reverse constraints 后重新赋予合法数组 ID；测试仍比较同一 assignment/conversion use。该调整修复测试构造，不放宽生产不变量。最终全量验证此前曾为 102/103；修正后复跑为 103/103，再加入 root-budget 回归进入最终 Gate。

**最终 Gate（2026-09-12，均 exit 0）：**

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
git diff --check
```

结果：**30/30 lit、104/104 unit（15 suites）、4/4 CTest**；相对 M3 基线新增 3 lit、30 unit。命令均在隔离工作树执行；CTest 在目标链接完成后运行。日志中的预期负例诊断不代表测试失败。

| 验收组                                                                                                       | 直接证据                                                                                                              |
| ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| cast/subview、降秩、packed、offset=3、XOR carry、逆切片限制、Unknown                                         | `LayoutAliasAnalysisTest` 14 tests，含独立逐点地址预期与失败域                                                      |
| root-relative 物化、全部 3 pairs、缓存复用、RelationsOnly counting target、篡改/rollback、向量及 root-budget | `StorageAliasIntegrationTest` 5 tests；4×8 切片结果 40/56/88，offset=3 lit 结果 52/100                             |
| region tuple/ID/方向、Strict/Common 完整账本、失败计数、确定性与回边                                         | `AliasRegionTest` 11 tests；12 种构造顺序相同 assignment/统计；真实 for 9 pops ≤ 10 event bound ≤ 18 static bound |
| 可运行 pass 与重复物化                                                                                       | `layout-alias.mlir`、`layout-scf-fixed-point.mlir`；SCF 一次/两次 pass 输出逐字 diff 相等                         |
| 前置契约被规范化删除                                                                                         | `layout-alias-contract-lifetime.mlir`：重复 infer 成功，插入 canonicalize 后缺失证据被明确拒绝                      |

#### 18.10 本地集成记录（2026-09-16）

- 用户明确授权提交当前修改并合入 main；未要求 push。
- `05e8cfa`：先独立提交主工作树中的 M4 审计/Task 18 设计基线，保留审计先于实现的历史。
- `9c28b67`：提交 Task 18 实现、测试和三份同步文档。
- 使用 `git merge --no-ff --no-commit feature/m4-task18` 集成。代码无冲突；两份审计文档冲突采用核对后的 Task 18 最新版本，审计原文可由 `05e8cfa` 恢复。
- 合并前在功能工作树重跑 Gate，合并后在 main 工作树重新构建及验收；两次均为 30/30 lit、104/104 unit、4/4 CTest。`git diff --check` 通过。
- `guoqiao/inference_detail/README.md` 未修改、未暂存、未提交；本次没有 fetch/pull/push 或远端分支操作。

### Task 19: 迁移 Copy、Fill 和 Parallel 约束

**状态（2026-09-16）：具体契约 v1 已获用户书面确认，§19.8 A–F 已完成，独立复审及全量 Gate 通过。代码尚未提交或推送，实际接口、设计适配和支持边界见 §19.9。**

#### 19.1 当前代码审计与设计取舍

审计起点为 `ef85a0d`，直接在主源码目录工作，不新建 worktree。本任务不自动 commit/push。

- `LayoutPropagation.cpp::collectLayoutConstraints` 已收集整块静态 Copy，但只接受直接 `layout_view` 操作数；现有 Storage–Storage `StorageAccess` 在 `LayoutRelations.cpp` 中要求 map 相同。这是 M2 的受限实现，不是复制操作的一般正确性条件。
- `FriskOps.td` 的 Copy 和 Fill 都只有 MemRef 操作数。不能按 memory space 把某个 MemRef 暗中当作 Tensor，也不能凭空插入一个无法在最终 IR 中找到的执行布局变量。
- `FillOp::verify` 当前只接受与元素类型匹配的 FloatAttr；本任务不顺带扩展整数 Fill 或任意标量表达式。
- `ParallelOp::inferLayout` 保留旧的 DenseMap 递归路径；新 `FriskInferLayoutsPass` 当前已经走 collect/propagate/solve/verify/materialize，不调用旧方法。本任务补充隔离证据，不把现有隔离重复计为新增能力。
- `Ownership`、`ResourceLimit` 目前有枚举名，但 bootstrap solver 尚不执行其语义。仅生成这两种约束并打印出来，不算完成迁移；传播、求解和 actual-only verifier 必须共同执行它们。

方案比较：

| 方案                                                       | 取舍                                                             | 决定         |
| ---------------------------------------------------------- | ---------------------------------------------------------------- | ------------ |
| 继续要求 Copy 两端布局相同，Fill 只检查存储单射            | 改动小，但没有表达真正的执行者，也错误限制独立存储的复制         | 不采用       |
| 将 Copy/Fill 改为 Tensor 新算子族                          | 可直接使用 Tensor encoding，但扩展前端、结果类型和旧调用迁移范围 | 本任务不采用 |
| 保留现有 MemRef 算子，新增实际可物化的执行布局与写入者契约 | 存储与执行分布分开，能接入统一约束和独立核验                     | 采用         |

#### 19.2 公共表示：存储布局与操作执行布局分开

每个 Copy/Fill 新增一个操作级 Distributed 执行变量，逻辑 shape/dtype 来自完整复制/填充区域。内部可用同 shape 的 RankedTensorType 描述布局域，但不新增虚构的 Tensor SSA value，也不改变 MemRef 类型。

选定结果写回原操作的以下属性，并由 Op verifier 检查属性类型与基本格式：

```text
frisk.execution_layout : DistributedEncodingAttr
frisk.execution_threads : i64
frisk.writer_policy : "all" | "first_owner"
frisk.vector_bytes : i64
```

- 无这些属性的旧 IR 仍可作为推断输入；推断成功后四项必须齐全。
- 显式给出的属性是硬要求。不能覆盖显式 encoding、改变显式 writer policy，或静默降低显式 vector width。
- 不存在 enclosing Parallel 时，默认执行环境为单 CTA、32 线程，并在操作上物化 `execution_threads = 32`；显式给出的合法线程数可替代默认值。
- 存在 enclosing Parallel 时，执行线程数必须等于该作用域的 `threads`；不能用操作上的属性覆盖外层线程数。
- LayoutVar 增加专门的操作级绑定标记，区别于 Value、OpOperand use 和函数结果槽位；stable identity 包含所属操作的稳定位置及属性角色，不依赖指针排序。
- MaterializeLayouts 对该变量写操作属性，actual-only verifier 从同一操作属性读取实际 assignment；不得沿用候选缓存或想象未物化的执行布局。
- Writer policy 和 vector width 在本任务中是明确输入/确定性默认契约，不作为完整性能成本搜索的新维度。默认 vector width 为 1；默认 writer policy 为 `first_owner`。
- 新增执行变量继续受 M3 候选生成边界约束：非零 rank、各维为大于 1 的静态二次幂。不能因 Storage 端可表示任意静态 shape 就宣称执行端也支持；不满足时明确拒绝，不 padding 或修改 shape。这是新增执行模型的首版限制，须在迁移兼容性测试中单独记录。
- bootstrap solver 的每连通分量 8 个变量、每域 4 个候选限制保持不变；执行变量计入限制。不为通过测试而隐藏变量或任意截断到前 4 个候选。

#### 19.3 Copy：逻辑对应，不是物理地址相等

支持边界仍为：同 shape、同元素类型、静态非空整块区域，src/dst 为直接 `layout_view` 结果，两个 copy map 均无输入及输出，indices 为空。被绑定的 MemRef 可以是 Task 18 已支持的静态切片；不扩大到 Copy 自带动态 indices、部分区域或 Local MemRef 冒充寄存器布局。

设执行布局为 `D(h)=q`，源和目的存储布局为 `S_src(q)`、`S_dst(q)`，复制契约为：

```text
同一个逻辑坐标 q：读取 S_src(q)，写入 S_dst(q)。
```

建立两条带明确角色的 `StorageAccess`：执行布局→src 为 Read，执行布局→dst 为 Write；建立执行变量上的 Ownership 与 ResourceLimit。不得建立 src/dst 的 AliasLayout，也不得仅因复制就添加 Storage–Storage SameLayout。

例如两块独立的 `2×2xf32` 存储，源地址公式为 `8i+4j`、目的为 `4i+8j`，只要各自合法且执行访问满足契约，Copy 可以成立。它不是把两个地址公式声明为相等，而是按逻辑元素搬运。

双向信息利用限定为候选准备：已知 src 可向 dst 提供有限合法存储方案，已知 dst 同样可向 src 提供方案；跨不同 root 不直接复制绝对地址或对齐声明。只允许可证明的地址原点重定位和目标域重绑定，重新检查目标 root 容量、单射性、对齐与向量访问。无法证明的生成建议丢弃；显式绑定不得静默丢弃。不同而合法的显式两端布局不能因候选传播被强绑为相同。

与 Task 18 相同，候选准备结束后冻结域；不把双向生成放进删减 worklist，不做无界来回投影。双向测试检查“对端获得合法候选且解满足访问关系”，不再把两个 Attribute 相等作为通用正确性标准。

Copy 的读写依赖不是 alias 布局一致性的替代品。同一已知 root 的区域如存在非恒等重叠，首版拒绝，避免将可能需要临时缓冲或有序搬运的操作当成无序并行复制；完全不相交或逐元素相同的视图可接受。不同 root ID 不构成运行时 NoAlias 证明；独立 MemRef 参数间的不重叠要求属于 Copy 的调用前置契约，必须在文档和诊断中与已证明的同 root 信息区分，不宣称本任务完成跨过程 alias 分析。

#### 19.4 Fill 与 Copy 的实际写入者契约

`replication` 描述同一逻辑元素的硬件持有者数量，不直接等于实际写入次数。给执行坐标规定与 map 输入顺序无关的规范顺序：

```text
(cta, warp_group, warp, lane, register) 的字典序。
```

两种 writer policy 的含义：

- `all`：每个持有者都执行写入。只有每个 live logical point 恰好有一个持有者时合法；replication > 1 的 all-writer store 拒绝。
- `first_owner`：对每个逻辑坐标只允许规范顺序最小的持有者执行写入，其他持有者不写。必须精确证明覆盖完整、被选写入者唯一；这一选择显式保存在实际 IR，不能只留在 solver 中或假定未来 lowering 会自动处理。

例如 32 线程填充 4 个元素，执行布局可能每个元素有 8 个持有者。`all` 会重复写入，应报错；`first_owner` 明确选出每个元素的一个写入者，才允许通过。Copy 的重复读取可以合法，但目的写入仍必须经过同样的 writer policy 核验。

Fill 建立执行布局→目标存储的 Write StorageAccess，以及 Ownership/ResourceLimit；标量常数不创建布局变量。执行坐标枚举及逻辑域验证各限制为最多 65536 点，乘法/位移先做溢出检查。超限返回 Unknown/unsupported，不抽样证明。

这补充了原计划“禁止 replicated store”的准确含义：禁止同一元素的多写入者，不禁止带显式唯一写入策略的 replicated read/持有布局。首版不实现 reducer 的 partial addend、原子写入或跨操作写竞争分析。

#### 19.5 向量访问契约

`vector_bytes` 仅接受 `{1,2,4,8,16}`。它是操作的执行访问宽度，不等于 StorageLayout 的 `vector_granularity`，不能只比较两个数值大小就认为合法。

- 宽度 1 是标量/packed 基线；较宽访问首版仅支持字节整齐的标量元素。
- 必须根据 `S(D(h))` 验证每个实际执行者的连续寄存器元素形成完整向量块，物理地址连续、块起点满足绝对对齐、无越界和尾部缺失。元素大于向量宽度时只允许可证明对齐的标量子块。
- Copy 两端都需证明同一执行分组有效，不能源端连续就默认目的端连续。Fill 检查唯一写入者选择后的分组，不能把由不同线程负责的元素拼成一个向量块。
- 对齐证据沿用 Task 18 的 root provenance。操作要求不能提升根指针的已知对齐。
- 不满足硬 vector width 的候选在冻结后的传播中删除；显式编码没有其他选择时报告冲突，不修改 shape、allocation 或显式属性。候选的独立合法性证明与操作访问证明分别记录来源。

#### 19.6 Parallel：执行环境约束，不是最终布局生成器

保留现有 Parallel 的 Kernel 父操作与 region 结构，不引入新的嵌套 Parallel 语义。`ranges` 的静态正范围和区域参数契约由 verifier 检查，不把它们误当作 Copy/Fill 的 tile shape。

首版支持 `threads ∈ {32,64,128,256,512,1024}`，单 CTA。Distributed topology 的线程数按 `lane × warp × warp_group` 计算，不乘 register。要求 lane=32、CTA=1，线程数与作用域精确一致；输入格式错误或乘积溢出直接诊断。

128 线程的默认候选采用 lane=32、warp=4、warp_group=1，但这只是候选生成所用的资源环境，不代表自动选择 WGMMA。显式合法拓扑可以采用等价的分组方式，由实际线程乘积和 map 契约共同验证。

区域内 Copy/Fill 执行变量以及当前支持的 Tensor 值/具体消费边界都受最近的 Parallel 环境约束。区域外生产的 Tensor 不能只因跨边界使用就改写生产者编码；必要转换仍在实际 use 上显式物化。未建模的 Tensor region 边界继续明确拒绝，不隐式扩大 M3 支持范围。

ResourceLimit 是真正的硬约束：影响候选筛选、求解可行性和最终核验。目标模型依据线程环境生成有限拓扑候选，不再只生成固定 32/128 线程后指望其他线程数自行成立。Parallel 自身不调用旧递归推断，也不直接写最终执行布局。

#### 19.7 求解、物化、失败语义与文件分工

通用规则与 SM90 规则分开：collector 负责操作角色与绑定；Analysis 中的纯 helper 负责有界执行/向量访问证明，SM90 target 依据资源环境生成候选；统一 relation evaluator 负责硬关系执行，不能在 solver 中硬编码 Copy 例外。实现采用下述 Analysis helper 分层，避免 Analysis → Target → Analysis 循环链接。

已实现的主要接口与 payload 如下：

```cpp
// OperationLayoutConstraints.h：在图 finalize 之前创建实际操作绑定。
LogicalResult collectOperationLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph,
    LayoutConstraintBuilder &builder);

LogicalResult collectParallelResourceConstraints(LayoutConstraintGraph &graph);

// ExecutionLayoutProof.h：不改 IR 的有界证明，返回 Proven/Disproven/Unknown。
LayoutProof proveExecutionOwnership(
    DistributedEncodingAttr execution, RankedTensorType logicalType,
    StringRef writerPolicy);
LayoutProof proveExecutionVectorAccess(
    DistributedEncodingAttr execution, const StorageAliasInfo &storageInfo,
    const StorageAliasFootprint &storage,
    unsigned vectorBytes, AccessKind access, StringRef writerPolicy);

struct OperationExecutionBinding {
  int64_t threads = 32;
  unsigned vectorBytes = 1;
  std::string writerPolicy = "first_owner";
};
// LayoutVar 新增：optional<OperationExecutionBinding> operationExecution;
//                 int64_t requiredThreads = 0;
// LayoutConstraint 新增：int64_t requiredThreads = 0;
```

`CopyAccess` 是新增的 Storage–Storage 硬关系：核验同 shape/dtype、合法 footprint，以及同 root 的逐点恒等或完全不相交。它也供既有 `StorageAliasCandidates` 有限 origin 准备遍历使用；投影通过 `rebaseStorageCopyCandidate` 重建目的地址原点/容量，并削弱到目的 root 实际具备的对齐保证。它不等于 `AliasLayout`、`SameLayout`，也不证明不同 MemRef 参数运行时不重叠。

新增/修改职责：

- `include/Dialect/Frisk/IR/FriskOps.td`、`lib/Dialect/Frisk/IR/FriskOps.cpp`：操作属性及语法/类型 verifier，Parallel 资源输入检查；保留旧 adapter 方法。
- `include/Dialect/Frisk/Analysis/OperationLayoutConstraints.h`、`lib/Dialect/Frisk/Analysis/OperationLayoutConstraints.cpp`：Copy/Fill/Parallel 收集与执行绑定，替代 LayoutPropagation 中内嵌的受限 Copy 收集器。
- `include/Dialect/Frisk/Analysis/ExecutionLayoutProof.h`、`lib/Dialect/Frisk/Analysis/ExecutionLayoutProof.cpp`：纯执行证明 helper；未新增原提纲中的 SM90CopyConstraints 文件，数学证明不依赖目标候选枚举。
- `LayoutConstraint.{h,cpp}`、`LayoutSolver.h`、`LayoutRelations.{h,cpp}`、`LayoutPropagation.cpp`、`LayoutVerifier.cpp`：操作级身份、带类型的 unary/binary payload、graph invariants、冻结后的硬约束删减、solver/actual verifier 共用关系语义。
- `SM90DistributedCandidates.cpp`：依据线程环境生成有限执行候选；复用未改动的 `StorageAliasCandidates.cpp` 调度 CopyAccess 的有限双向建议；RelationsOnly 禁止调用这些生成逻辑。
- `MaterializeLayouts.cpp`：在 detached module 中物化操作属性，与现有 Tensor/Storage 改写一起核验后原子提交。
- 对应 CMake 文件、新单元测试及 lit 文件：新增源文件和测试注册；不改无关构建依赖。

所有新绑定都要在 graph finalize 中保持稳定身份。新增 unary 约束必须接入 Strict/Common 的计数与调度，不得通过“关系没有第二个端点所以默认成功”。实际核验不允许为缺失属性补默认值、重新枚举候选或选择 writer policy。

任何失败均保持原 IR 不变。缺失或篡改 execution_layout/threads/writer_policy/vector_bytes、根对齐证据丢失、资源或写入者冲突都必须有对应负例。

#### 19.8 实施顺序与验收清单

使用 writing-plans、test-driven-development 和分阶段执行/复审流程；保持用户指定的源目录工作方式，不创建 worktree。先完成本设计的书面复核，再进入下面的红灯测试与实现。

- [X] **A：执行绑定与 Ownership。** 在 `CopyFillConstraintTest.cpp` 先测试未物化绑定被拒绝、32 线程/4 元素的 all-writer 失败与 first-owner 成功、缺失元素覆盖失败；再实现操作级变量、Ownership 和 actual-only 属性读取。
- [X] **B：Copy 双向与向量访问。** 先测试 src 已知/dst 未知、dst 已知/src 未知、两端合法异构存储、不同根偏移不直接复制、同 root 非恒等重叠拒绝；向量宽度测试 3 非法、16 字节未对齐失败、目的不连续失败、标量基线成功。再替换旧 Copy 收集器并实现有界证明。
- [X] **C：Parallel。** 先测试 128 线程候选、32/128 显式编码冲突、其他合法线程数、非法线程数和 Tensor 消費边界，再接入 ResourceLimit 与目标候选环境。
- [X] **D：独立核验与事务性。** 从合法结果分别篡改四项操作属性，测试 actual verifier 拒绝且 target enumeration 次数为 0；测试推断一次/两次输出一致，失败前后原 IR 逐字一致。
- [X] **E：旧路径隔离与回归。** 用真实 pass 调用检查 legacy Parallel inference 的调用次数为 0，单独运行旧 adapter 测试；统计只反映实际调用，不添加永远为零的伪计数。生产新路径与 legacy 行为不要求文本一致，差异按逻辑坐标和硬契约解释。
- [X] **F：全量 Gate 与文档。** 运行以下命令，记录实际新增测试数量、支持边界及设计偏差；同步比较文档，不以 M3/Task 18 的旧测试数代替本任务验收。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
git diff --check
```

红灯测试先运行新增测试过滤器及新 lit 文件，确认因缺失行为而失败，再编写生产实现。完成标准是全量测试通过、独立复审无未处理阻断项、两份长期文档与代码一致；未运行上游运行差分或 GPU benchmark 时不得声明行为全面兼容或性能领先。

以下保留原始迁移提纲以追踪计划演进，复选框不再代表当前验收状态，以 §19.8 A–F 与 §19.9 为准；若与上面 v1 契约冲突，以上面的具体契约为准。特别是 Copy 的 SameLayout 仅能用于真正同编码的执行槽位，不能再用于要求两个独立存储地址公式相同；Step 6 的 git 命令需用户另行授权。

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `test/Transforms/infer-copy-fill-parallel.mlir`
- Create: `unittests/Dialect/Frisk/Layout/CopyFillConstraintTest.cpp`

**Interfaces:**

- Produces：

```cpp
LogicalResult collectCopyConstraints(CopyOp op,
                                     LayoutConstraintBuilder &builder,
                                     LayoutTarget &target);
LogicalResult collectFillConstraints(FillOp op,
                                     LayoutConstraintBuilder &builder);
LogicalResult collectParallelConstraints(ParallelOp op,
                                         LayoutConstraintBuilder &builder);
```

- [ ] **Step 1: 写正反向推断红灯测试**

Copy 分别测试已知 src 推 dst、已知 dst 反推 src；Fill 测试 unique writer/replication；Parallel 测试 threads=128 seed warp-group topology，threads 与 encoding 不一致时报错。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，collector 尚未覆盖三类 Op。

- [ ] **Step 2: 实现 Copy 约束**

Copy 根据 operand role 创建 Storage 或 Distributed var，建立 `StorageAccess`/`SameLayout`/`Preference`。首版 vector width 只接受 `{1, 2, 4, 8, 16}` bytes，alignment 不满足时删除候选，不修改 shape。

- [ ] **Step 3: 实现 Fill/Parallel 约束**

Fill 对每个 live logical point要求唯一 writer，允许明确 replicated read 但不允许 replicated store。Parallel 将 thread count/topology 写入 ResourceLimit/Ownership，不直接生成最终 layout。

- [ ] **Step 4: 删除 Parallel 的递归 `DenseMap` 调度使用**

保留旧方法定义供 adapter tests 使用，但生产 `FriskInferLayoutsPass` 不再调用 `ParallelOp::inferLayout`。增加测试统计 legacy 方法调用次数为 0。

- [ ] **Step 5: 运行迁移测试**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: Copy/Fill/Parallel 新测试和 legacy tests 均 PASS。

- [ ] **Step 6: 提交基础 Op 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: migrate copy fill and parallel constraints"
```

#### 19.9 实施记录与设计适配（2026-09-16）

实现起点 `ef85a0d`，在源目录 main 工作，未创建 worktree，未自动提交或推送。以下内容是 Task 19 的实际增量，不表示 M4 整体完成。

| 已实现能力                        | 实现位置                                                                                             | 直接验证                                                                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 操作级执行变量、四项实际 IR 契约  | `OperationLayoutConstraints`、`LayoutConstraint`、`MaterializeLayouts`                         | Fill 推断后可读取全部属性；删除任意一项、篡改 encoding/threads/writer/vector 均拒绝；actual verifier 的目标枚举次数为 0          |
| Copy 的读写坐标关系与有限双向建议 | `LayoutRelations` 的 CopyAccess/StorageAccess、`LayoutAliasAnalysis::rebaseStorageCopyCandidate` | 2×2 f32 的行主序源/列主序目的可复制；已知任一端可向另一端提供重定位候选；同 root 恒等和不相交成功，位移重叠失败                 |
| 实際写入者与向量访问证明          | `ExecutionLayoutProof`                                                                             | all-writer replica 冲突、first_owner 规范选择、缺失覆盖、输入名字重排、寄存器向量连续性/对齐/尾部、packed 标量基线、65536 点上界 |
| Parallel 线程环境与实际 use 转换  | `collectParallelResourceConstraints`、`SM90DistributedCandidates`                                | 六种合法线程数；128 线程作用域拒绝 32 线程显式属性/编码；外部 32 线程 Tensor 在内部消费边界插入转换，生产者保持原编码            |
| 硬约束闭环和可核验统计            | Strict/Common worklist、bootstrap solver、actual verifier                                            | Ownership/ResourceLimit 不再是未执行的枚举名；宽向量要求真实删除候选；冻结候选与 monotone stats 上界保持有效                     |
| 旧路径隔离与文本回放              | 真实 legacy 调用计数、Kernel/Parallel/Copy printer/parser                                            | 显式调用 legacy 后计数增加；运行新 pass 后不增加；打印后重新解析、第二次推断与第一次逐字一致                                     |

具体例子：在 `threads=128` 的 Parallel 中对 `memref<4xf32>` 做 Fill，默认执行布局可以让每个元素有 32 个持有者，但物化的 `writer_policy="first_owner"` 只允许每个元素的首个规范持有者写入。把属性显式设为 `all` 会因重复写入被拒绝。128 线程对应的默认 topology 是 `[register=1, lane=32, warp=4, warp_group=1, cta=1]`，不是已选择 WGMMA 指令。

适配与复审记录：

- 证明模块放入 Analysis，collector 和 target 分开；没有机械新增一个反向依赖 Target 的 SM90 helper。
- 为支持真实 Kernel/Parallel 输入，Task 18 的静态存储根识别补充了类型与签名一致的 Kernel 入口 MemRef；没有扩展到未知 region 参数或动态别名。
- 回放测试暴露并修正旧 Kernel 入口参数名丢失、空签名解析、Parallel induction 参数类型/终结符、Copy 双重解析方括号和 extents/属性打印位置问题。Copy 的 map 使用 ODS getter 获取，extents 用 `srcExtents`/`dstExtents` 的 DenseI64Array 保存；不是扩大 Copy 的推断支持范围。
- Copy 候选重建后的地址上界按目的 descriptor 的实际可达容量规范化。旧 lit 中 16/160 改为 14/134 的两例仍保持相同物理地址公式，并未扩大 allocation 或更改 shape。
- 对齐负例保留 Task 18 的完整根 binding 前置契约语义：仅删除 allocation alignment，若完整根 `layout_view` 仍显式声明相同保证，不能称为证据全丢失。测试同时将 allocation 保证移除、whole-root storage binding 的 alignment/vector_granularity 降为 1，并保留操作 vector_bytes=16，要求 actual verifier 拒绝且不枚举候选。操作自己的向量需求不能充当根指针对齐依据。
- 纯证明独立复审指出：相同 root 不足以证明 footprint 属于当前 view；标量宽度也必须检查区间不重叠。新增两个先红后绿的反例，核对 view→root 坐标并在宽度 1 快速返回前核验物理区间。复审已确认这两项解决。
- 全任务独立复审补出了零输入 Kernel 回放反例，已修正；最新代码复审无未处理的 Critical/Important 项。首次核心红灯涉及未物化绑定、all-writer 误接收及 Copy 强制相同地址图；不是只编写通过测试。

验证命令沿用 §19.8，2026-09-16 最终结果：

- 构建 `FriskLayoutUnitTests check-frisk frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test frisk_memory_effect_test` 成功。
- **129/129 单元测试通过**（17 个 suite）：相对 Task 18 的 104 项新增 25 项，其中 `CopyFillConstraintTest` 16 项、`ExecutionLayoutProofTest` 9 项。
- **32/32 lit 通过**：相对 Task 18 新增 `infer-copy-fill-parallel.mlir`、`infer-copy-fill-parallel-errors.mlir` 两份，覆盖图输出、属性物化、打印解析后二次推断逐字一致和错误诊断。
- **4/4 CTest 通过**，包括旧 attribute、reduce、layout pass 与 memory effect 回归；单元中的 legacy adapter 测试一并通过。
- `git diff --check` 无错误。构建日志中的既有 Ninja 日志恢复提示未阻断构建；没有借此删除或重建用户工作区。

新增单元位于 `unittests/Dialect/Frisk/Layout/CopyFillConstraintTest.cpp` 和 `ExecutionLayoutProofTest.cpp`。负例测试主动产生的 expected error 不代表测试失败，以测试程序退出码和汇总为准。

仍不支持：动态/部分区域 Copy、执行域非二次幂或含 extent=1、一般跨过程 NoAlias 证明、跨操作竞争分析、GPU 写入谓词/向量指令生成、TMA/cp.async、GEMM/Reduce 迁移和完整性能成本求解。`first_owner` 是已经验证并物化的执行契约，未来 lowering 仍须消费它。未运行 TileLang/Triton 执行差分或 GPU benchmark，本轮不宣称全面兼容、独创性已证明或性能领先。

### Task 20: 增加内部 Tensor MMA Op 并迁移 Gemm 约束

**状态（2026-09-19）：已按用户确认的以下 SS＋RS 契约完成实现与测试。全量 165 unit / 34 lit / 4 CTest 通过；实际实现、复审修正与边界见 §20.13。Task 21/22 及 GPU lowering 未包含在本次交付内。**

#### 20.1 实现前审计与方案选择

工作基线是 `ef85a0d` 加当前未提交的 Task 19 实现与文档，而不是只有该提交的干净源码。在 `/home/baopeihua/frisk` 的主工作区继续，不创建 worktree，不自动 commit/push，不改写独立汇报文档。

本轮只读检查发现：

- `FriskLayoutOps.td` 尚无 `MmaOp`；`FriskLayoutAttrs.td` 只有通用 Distributed/Storage encoding，没有实现设计目标中的 Mma/DotOperand 专用 encoding。
- `GemmOp::verify` 要求 A/B/C 同元素类型，且按未转置形状检查维度；不能作为新操作的数学 verifier。
- 旧 `GemmOp::inferLayout` 用 Local MemRef 表示寄存器 fragment，直接更新 `DenseMap<Value, Attribute>`；新接口必须使用 Tensor SSA 和图中的实际消费位置，不能机械复用旧容器。
- `InstructionContract` 目前只有枚举名。现有 relation evaluator 主要执行 unary/binary 关系，不能把 A/B/accumulator 的联合契约拆成若干可能不相容的两两“存在某个方案”。
- Task 19 的操作执行变量专门绑定 Copy/Fill；MMA 已有 Tensor accumulator/result，不借用这类变量虚构另一份结果编码。
- `LayoutTarget` 目前只枚举/核验单个布局，没有指令级联合接口；需要补充有界生成与纯核验的分层，避免 Analysis 反向链接 SM90。
- Task 22 已单独安排旧 Local MemRef 生命周期 normalization；本任务不提前自动改写旧 `frisk.gemm`。

采用“内部 Tensor MMA ＋显式指令契约 ＋通用 canonical maps”的方案。只实现 SS 会遗漏寄存器 A 的迁移验证；同时加入旧 IR 自动转换及 GPU pipeline 会跨越 Task 22 和 lowering 的边界。本任务完成 SS、RS 的布局正确性闭环，不生成 WGMMA/fence/commit/wait 指令序列。

本轮重新核对了既有审计快照的相关源码，HEAD 分别为 TileLang `5e149e31674658f94779c7d0c6039549a1853123`、Triton `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c`；没有刷新到浮动 main，也没有运行上游测试。具体证据与取舍见 §20.10 及比较文档的 Task 20 设计增量。

#### 20.2 `frisk.mma` 的数学语义、类型与副作用

新操作固定表达：

```text
result[i,j] = init[i,j] + sum(k = 0 .. K-1) A_eff[i,k] * B_eff[k,j]

A_eff[i,k] = trans_a ? A[k,i] : A[i,k]
B_eff[k,j] = trans_b ? B[j,k] : B[k,j]
```

这定义逻辑对应，不承诺浮点归约顺序或比目标指令更强的数值精度。没有隐含的 alpha/beta、清零开关、额外输出写回或动态 clear 条件。

- 操作数顺序固定为 A、B、init，唯一结果为 result。A/B 在类型层可为 RankedTensor 或 MemRef；init/result 必须为 RankedTensor。
- 首版数学 verifier 接受静态正 extent 的 rank-2 浮点矩阵；A/B 元素类型相同，init/result 的元素类型相同且位宽不小于输入。特定 dtype 是否有目标指令由 target 判断。
- 属性 `m/n/k` 为正的 signless i64；`trans_a/trans_b` 为 BoolAttr，缺省 false；`policy` 为已有 GemmWarpPolicy，缺省 Square。维度按上述逻辑转置精确匹配；矩形转置负例必须覆盖。
- init/result 的逻辑 shape 必须都是 `[M,N]`。输入阶段允许缺失 encoding 或两者 encoding 不同；最终通过 init 的实际消费位置转换，使该位置与 result 的 accumulator encoding 一致，不能直接改写 init 的其他使用者。
- MemRef 输入只有 Read effect，不写 A/B、不分配存储；Tensor 输入是 SSA 依赖，result 不是 MemRef 写入。带 MemRef 输入的 MMA 不可标为 Pure，不可因漏报 memory effect 而跨越写操作随意移动。
- `clear_accum=true` 的未来迁移应传入显式零 Tensor，普通累加传入实际旧 accumulator；动态 clear 的语义与转换留给 Task 22。不得用 `tensor.empty` 充当已初始化的零。
- 这是高层同步数学操作：result 在该操作之后可用。未来异步 lowering 必须建立这一可见性，当前布局验证不声称已经证明 pipeline 同步安全。

原 Task 20 示例仍有效：`A:128×64xbf16`、`B:64×128xbf16`、`init/result:128×128xf32`。示例只展示操作类型，不省略测试所需的合法 shared layout、根对齐契约和目标环境。

#### 20.3 SM90a 首版支持集与明确拒绝项

数学操作不内嵌 PTX mnemonic；目标规则按以下受限表选择契约：

| 项目       | 本任务设计边界                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 目标       | 最近的`frisk.target` 必须明确为 `"sm_90a"`；缺失、`sm_90` 基础能力或其他目标在 MMA target 校验中拒绝。不改变 Task 19 普通操作的默认目标行为 |
| 数据类型   | A/B 同为 f16 或同为 bf16；init/result 为 f32。暂不接入 f16 accumulator、TF32、FP8、整数或稀疏路径                                                 |
| SS         | A/B 为 Shared MemRef，必须是直接`layout_view` 结果；根和静态别名链沿用 Task 18                                                                  |
| RS         | A 为 RankedTensor，B 为上述 Shared MemRef；禁止将 Local MemRef 当作 Tensor fragment                                                               |
| Tensor B   | 通用类型语法可表达，但本 SM90a target 拒绝。不得偷偷插入 shared allocation/store 把它变成另一种路径                                               |
| 执行域     | 保留现有 Distributed 的非零 rank、每维大于 1 的二次幂限制；M 是 64 的倍数，K 是 16 的倍数，N 至少为 8                                             |
| 原子与拼接 | 原子为`64×n_atom×16`，`n_atom ∈ {8,16,32,64,128,256}`；整 tile 可以在 M/N/K 上重复原子，不 padding、不改 shape、不处理尾块                 |
| 线程       | 单 CTA，`threads ∈ {128,256,512,1024}`；外层 Parallel 的线程数必须一致，无 Parallel 时缺省 128，显式合法线程数可替代缺省值                     |
| 控制流     | 保留 M3 的 SSA/SCF 布局规则；Task 20 不新增 warpgroup 分歧、动态线程参与或异步 token 分析                                                         |

硬件支持不等于 Frisk 当前表示支持：例如硬件存在的 `n=24` 原子不属于上述二次幂子集，不能通过改 shape 或忽略 encoding verifier 来接受。

SS 支持数学上的 `trans_a/trans_b` 四种组合。RS 的 `trans_a=true` 通过 A fragment 的逻辑坐标交换和必要的输入布局转换表达，不伪造硬件 RS 的 A-transpose immediate。逻辑 transpose 与硬件 major mode 是两层信息：major mode 从最终地址关系解码，不能把 BoolAttr 原样当作 PTX immediate。

#### 20.4 线程组织、policy 和寄存器坐标

每组固定 4 个连续 warp，lane extent=32、CTA extent=1，warp_group extent=`threads/128`。线程编号解释为 `((warp_group * 4 + warp) * 32 + lane)`，不能只检查线程数乘积而放过不符合组边界的任意拓扑。

设 group 数为 G，在有限因子对中选择 `[gM,gN]`，满足：

```text
gM * gN = G
M % (64*gM) == 0
N % (8*gN) == 0
```

- FullRow：在合法因子对中取最大 gM。
- FullCol：取最大 gN。
- Square：最小化整数值 `abs(M/gM - N/gN)`；相同值按 `(gM,gN)` 字典序稳定选择。这是 Frisk 首版确定性几何策略，不复现旧代码的浮点打分，也不是性能最优结论。
- 分组后取能够整除 `N/gN` 的最大受支持 `n_atom`。记录重复次数 `[M/(64*gM), N/(n_atom*gN), K/16]`；不搜索全部指令宽度组合。
- 没有合法分组就报错；不截断线程数、不忽略 policy。128 线程时只有 `[1,1]`，不同 policy 得到相同分组是合法退化情形；256 线程的 `128×128` 用例须展示 FullRow/FullCol 的差别。

group 坐标固定为 `groupM = warp_group / gN`、`groupN = warp_group % gN`，各 group 负责连续的 `[M/gM,N/gN]` 子块。原子外层寄存器槽按 accumulator 的 `(mRepeat,nRepeat,innerSlot)`、RS A 的 `(mRepeat,kRepeat,innerSlot)` 字典序展开，最右侧变化最快；innerSlot 顺序取硬件原子规范。K 重复更新同一 accumulator，不新增一份 result 槽。线程环境只约束 MMA 的消费槽与结果；不得为满足指令而改写外部 A/init 生产者的显式线程/encoding 契约。

结果与 init 消费槽使用按硬件 accumulator 原子展开的 DistributedEncoding，逐点验证完整覆盖和唯一持有者。RS 的 A 使用按寄存器输入原子展开的 DistributedEncoding；不同 N 方向的 group 可以持有相同 A 元素，这种只读 replication 必须被精确描述，不能沿用 Fill 的 all-writer 判定。

**packing 的语义必须明确：**现有 Distributed map 的 `register` 坐标在 16-bit A 上表示逻辑元素槽，不直接表示 32-bit 物理寄存器编号。每个 A 原子的 8 个元素槽要按契约组合成 4 个 32-bit packed registers，低/高半字顺序纳入指令核验；f32 accumulator 每槽对应一个 32-bit 寄存器。不能把“覆盖了同一批元素”当成“寄存器顺序满足指令”。原子外层的 M/K 重复及 N-group replication 同样进入规范 map。

共享与寄存器布局优先展开为当前已支持的 canonical BitLinear/Affine 表示，不在本任务顺带放宽一般 Product map 的 pass 支持边界。单个布局的逻辑点与硬件元素槽枚举均不超过 65536，任何溢出/超限返回 Unknown 并使硬检查失败。该点数上界不是寄存器分配、spill 或 occupancy 证明；完整资源成本与调度仍属于后续任务。

#### 20.5 Shared descriptor：必须证明能访问实际存储

对每个 Shared 操作数分别执行两层检查：先通过 Task 18 的根容量、坐标、单射和别名一致性证明，再验证它能够由本任务支持的指令描述符方案表示。普通合法 StorageLayout 不自动等于合法 WGMMA operand。

设计支持 K-major/MN-major 的无 swizzle 及 32/64/128-byte swizzle 规范模板。生成阶段每个未绑定 operand 只选一个默认 major：若按其数学角色和 transpose 解释后，原二维类型的最后一维是 K，则选 K-major，否则选 MN-major；在该 major 下最多提供四种 swizzle 模板。显式另一 major 的布局仍可被精确核验，不强制改回默认值。

描述符方案包含：major、swizzle 模式、leading/stride byte offset，以及该操作数各原子调用所需的 root-relative 起点和 swizzle phase。物化的是与 runtime root pointer 组合的**静态描述符方案**，不是假装知道运行时 shared pointer 的完整 64-bit descriptor。

必须满足：

1. 所有实际原子的起点和描述符偏移可编码、对齐、无截断和溢出；不能通过掩去高位把不合法偏移变合法。
2. 对每个原子实际访问的逻辑坐标，按描述符规则重建的地址与所选 StorageLayout 的 root-relative bit 地址相等，且不越出根可访问范围。
3. 静态切片的偏移、步长和 K-panel 间距来自实际布局/根坐标关系。不能根据切片自身的 M/N extent 猜测它在更大父 buffer 中的 panel stride。
4. 不重复叠加 descriptor offset；基址非零时同样逐点核验。数学 transpose 只改变逻辑坐标解释，不直接修改已有存储。
5. 无 swizzle 需要根及原子起点满足 16-byte 对齐。为使首版静态 swizzle phase 可证明，32/64/128-byte swizzle 分别要求根具有 256/512/1024-byte 对齐保证，再根据原子 root-relative 偏移计算 phase；这是 Frisk 的保守支持边界，不是声称硬件要求所有根都如此对齐。
6. 根对齐证据沿用 Task 18：allocation/global 属性或完整根 layout_view 的显式前置契约。生成候选、子视图要求和 MMA 指令需求不得自行增强根保证；没有足够证据时丢弃该建议，显式 binding 则报错。

显式任意地址 map 只有被上述模板和有界逐点证明共同覆盖时才能通过。对于数学上合法但当前无法解码的 layout，明确报告 unsupported，不宣称它在硬件上必然不合法，也不自动新建 shared buffer 搬运。

A/B 都是只读输入，可以重叠；若有共同 root，依然执行 AliasLayout 一致性检查，但不错误套用 Copy 的非恒等读写重叠禁止规则。

#### 20.6 图中的角色、真正的联合约束与有限候选

每个 MMA 的联合端点按固定角色排序为 `[A-slot, B-slot, init-use, result]`：SS 的 A/B slot 为 Storage var；RS 的 A-slot 为 Distributed use var、B-slot 为 Storage var；init-use/result 为 Distributed var。

关系如下：

- Tensor A 的生产者 → A 消费槽：Convertible，必要时在 MMA 的真实 operand 上插转换。
- init 生产者 → init 消费槽：Convertible。
- init 消费槽 ↔ result：SameLayout，不要求 init 的所有外部使用者都变成该编码。
- 四个角色共同关联一个 InstructionContract；Distributed 角色同时受真实 ResourceLimit 约束。
- Shared A/B 的访问维度与 result 不同，不得用 Task 19 的同 shape StorageAccess 直接连接。它们的 Read effect 和访问证明属于 MMA 的角色化联合契约。

InstructionContract 不是四条独立的 RequireEncoding。候选准备阶段形成经过目标纯证明核验的有限合法 tuple 表；一条 tuple 同时包含四个角色的编码及对应指令方案。只有**同一 tuple**能够支持当前各角色选择时才成立。

候选流程必须区分生成和删除：

1. 收集显式 encoding/storage binding、操作数学属性和目标/线程环境，创建稳定角色身份；显式项是硬要求。
2. 按所选 policy/原子组织准备 MMA 专用 canonical accumulator/A-fragment 和有限 shared 建议，复用 Task 18/19 的有限 alias/Copy origin 传播；不能给 MMA 结果继续填充无关的通用 SIMT 默认布局，也不能把 target 专用建议当作用户 RequireEncoding。
3. 不沿 InstructionContract 做无界“正反向生新布局”。专用建议各生成一次，通用候选准备收敛后冻结全部域；若同一存储参与多个 MMA，收集全部已允许的有限来源，而不是按访问先后覆盖。
4. 保留每 component 8 vars、每域 4 candidates 的 bootstrap 上限。超限明确报错，不截断前四项，不移除变量来绕过限制。每个四端点契约最多检查 `4^4=256` 个编码组合；重复端点必须使用同一 assignment。
5. 每个编码组合通过确定性的 descriptor 解码获得至多一个规范指令方案；有多个等价解码时采用固定键序，不额外引入没有上界的隐藏选择。已显式绑定方案时，只核验该方案。
6. Strict/Common 在原有队列上增加 hyperedge 支持检查：某个候选若没有任何与其余当前域同时相容的 tuple，就删除；InstructionContract 即使没有 singleton 也必须执行。domain 删除会重排所有相关 unary/binary/hyperedge，旧统计公式按真实邻接关系计数。
7. Solver 对部分 assignment 检查是否仍有合法 tuple 扩展，对完整 assignment 要求整条 tuple 成立；不能沿用把所有关系拆成首端点与其他端点两两检查的循环。

合法 tuple 表、proof cache 和 provenance 只在当前 graph 生命周期内有效；finalize 重排后所有角色 ID/constraint ID 正确 remap。稳定键按 IR 位置和角色，不按指针。图不变量检查角色数量、类型、source-op、重复端点一致性和 binding 归属。

仍采用现有 bootstrap 的最少显式转换数＋稳定排序选择，不调用未完成的完整 CostVector 性能优化。policy、原子宽度和 swizzle 排序只能称确定性策略。

#### 20.7 实际 IR 上的指令绑定与分层接口

新操作的数学属性不存 PTX 字符串；所选目标实现写入独立的类型化属性：

```text
frisk.mma_contract : MmaInstructionContractAttr
frisk.execution_threads : signless i64
```

`MmaInstructionContractAttr` 的 v1 必需字段是：schema version、target feature、SS/RS form、输入/累加类型、原子 `[64,n_atom,16]`、warp-group grid、M/N/K repeat counts、A packing 规则，以及 SS 的 A/B 或 RS 的 B descriptor plan。每个 plan 保存 major、swizzle、leading/stride offsets，和以操作数原子坐标为键的 root-relative start/phase 表。表规模受上述 tile/枚举预算约束，禁止含 pointer、SSA 地址或仅在内存中存在的缓存 ID。

Tensor result/消费位置仍使用 `DistributedEncodingAttr`，Shared 仍使用 `StorageLayoutAttr`。本任务将原提纲的“Mma/DotOperand”落实为**指令角色＋可展开的通用 encoding＋操作契约**，不新增另一套 Tensor encoding 子类型；不宣称总体设计中的 `MmaEncodingAttr`、`DotOperandEncodingAttr` 类已经实现。这一选择避免无关地重写全部 M3 Tensor 类型转换，但不省略专用 fragment/packing 证明。

显式 mma_contract 是完整硬绑定，不接受随意缺字段的“部分契约”；推断不能覆盖它。未指定时可由求解生成；actual-only 验证时两项属性必须齐全，不允许补默认值或重新选择方案。

新增职责与 API 契约：

- Analysis 的 `MmaLayoutConstraints` 只负责操作角色、SSA use、数学上下文和图关系。
- `LayoutTarget` 增加独立的指令候选准备入口和纯 `verifyInstructionContract` 入口；默认实现拒绝未知 contract，不通过抽象层直接调用 SM90 函数。
- `SM90GemmConstraints` 提供候选建议、规范方案构造；`SM90MmaLayoutProof` 提供按确定方案验证 fragment/descriptor 的纯函数。这两者不得修改 IR。
- graph 内的 `InstructionLayoutContract` payload 记录角色、实际操作 binding 和有界 tuple 证据；`LayoutSolution` 增加以稳定 constraint ID 为键的 selected instruction binding。合法方案不是普通 Storage/Distributed 变量，不伪装成 Tensor shape。
- propagation/solver 只做通用有限 tuple 支持检查，不识别 `sm_90a` 字符串或直接构造 WGMMA map；最终 verifier 通过传入的 target 重新验证实际绑定。
- RelationsOnly 模式在候选生成和 tuple 枚举之前返回结构图；读取实际 IR 的 singleton assignments 和实际 mma_contract，调用纯证明。不得调用 target enumeration/方案搜索，也不得复用上次求解的合法 tuple 表。
- 未识别的指令类别依旧 fail-closed；原 `RejectsUnsupportedHardConstraint` 测试改为未知指令类别负例，不能因 InstructionContract 被接入就默认所有类别合法。

#### 20.8 物化、独立核验和失败诊断

物化在 detached module 中完成：写 Tensor encoding、插入实际 use 转换、绑定 storage layout、写 mma_contract/threads，随后进行 MLIR verifier 和 actual-only 指令/布局证明，成功后才替换原 module body。

独立验证从实际类型和操作属性重建 A/B/init/result，检查目标 feature、线程 scope、数学 transpose、policy、packing、descriptor plan 与 map 一致性。不能把自报“正确”的 descriptor 属性当成地址证明；即使 descriptor 和某一个 operand 都各自合法，也要检查它们属于同一操作、同一 shape 和同一 warp-group 分组。

诊断使用稳定规则名，例如 `mma-shape`、`sm90-mma-target`、`sm90-mma-thread-group`、`sm90-mma-fragment`、`sm90-mma-descriptor`、`mma-joint-contract`；注明 A/B/init/result 角色及 relevant dtype/shape，地址或持有者失败尽可能给出逻辑/硬件坐标。Unknown 单列为无法证明，不冒充数学反例。

任何失败保持原 IR 不变；同一输入两次推断及打印/解析后再次推断必须稳定。当前不声称发现任意语义篡改：例如用户同时合法修改数学属性与完整布局/契约，得到的是另一个合法程序；测试要篡改为确实违反当前契约的组合。

#### 20.9 测试分组与完成标准

实施采用先红后绿及独立复审，先逐个打通 SS/RS vertical slice，再扩展合法模板；不能仅新增 Op 并打印一个 contract 就标记完成。

- [X] **A：数学 IR 与副作用。** rank/正 shape、M/N/K、矩形 transpose、f16/bf16→f32、init/result 关系、MemRef Read/no Write、零 Tensor 初始化，printer/parser。
- [X] **B：独立硬件坐标 oracle。** 固定原子的 A packed halves、accumulator 坐标、不同 lane/warp/寄存器顺序；多原子 M/K 拼接与 N-group replication。测试参考表不能调用待测候选生成器构造“期望值”。
- [X] **C：SS/RS 与 descriptor。** 两种输入 dtype、所有数学 transpose 组合、K/MN major、四种 swizzle；合法及不满足支持边界的组合分类测试。覆盖父 buffer 切片的 K-panel stride、非零起点、根对齐不足、phase 错误、地址区间越界；B Tensor/Local MemRef 明确拒绝。
- [X] **D：线程与 policy。** 128/256/512/1024、group 连续性、FullRow/FullCol/Square、无合法分组、显式 threads 与 Parallel 冲突；`sm_90a` 成功、缺失 target/`sm_90`/SM80 新路径拒绝。
- [X] **E：联合关系与收敛。** 构造“每对看似都有支持，但不存在完整合法 tuple”的反例；重复端点、稳定 remap、多 MMA 共用 storage、显式绑定冲突、8 vars/4 candidates 边界及超限、最多 256 个组合和点数预算。256 是四个合法候选域的组合上界，不人为制造域合法却有更多组合的假场景；损坏或重复的 tuple payload 单独按图不变量拒绝。统计证明冻结后只删不增。
- [X] **F：实际物化。** 外部 init/A encoding 保持不变，在 MMA 消费处转换；零拷贝 shared binding；删除 contract/threads、篡改 fragment/packing/descriptor/分组后拒绝；target 生成及 tuple 枚举计数均为 0；失败 IR 逐字不变、二次推断逐字一致。
- [X] **G：legacy 分类差分。** 用旧 adapter 比较适用子集的逻辑持有关系与地址，不比较 Attribute 字面相等；旧 FP16 accumulator 用例只能作为旧行为回归，不能冒充本次 FP32 accumulator 的数值等价测试。SM80、B-local、错误 group 组织、过时 ABI 分类记录；不保留已知错误求一致。
- [X] **H：全量 Gate、独立复审和文档。** 新单元 `GemmConstraintTest.cpp`、`SM90MmaLayoutProofTest.cpp`；新增 `infer-gemm-layout.mlir`、错误/回放 lit 与必要 IR verifier 测试；同步本文和比较文档的实际实现/差异/计数。

验收至少包含 `128×128×64` 的 SS/RS f16/bf16→f32 例子及 `64×64×16` 原子例子；它们必须在未放宽既有 solver 限制的真实 pass 中成功，不只通过独立 helper。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
git diff --check
```

实现后记录新增测试数量、实际 Gate 和每项未支持边界；没有执行上游/GPU benchmark 时，不宣称完整数值 lowering 验证、上游全面兼容或性能领先。

#### 20.10 上游核对与 Frisk 的适配原则

本轮已读的固定快照证据：

- TileLang [`src/op/gemm.cc`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/gemm.cc)：`k_pack`/`wg_wait` 在 annotations；可能读取 C 与确定 read-before-write C 对 clear 条件有不同判断；WGMMA 不随意沿用不符合指令要求的 shared layout。Frisk 用显式 init SSA 表达旧值依赖，不复制 positional ABI 或 `completed_` 调度。
- TileLang [`src/cuda/op/gemm.cc`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/cuda/op/gemm.cc)：WGMMA 分组把 4 个 warp 视为不可拆单元。Frisk 保留这一硬边界，但用自己的显式 group grid 和稳定 policy 定义，不强求旧浮点评分结果一致。
- TileLang [`test_tilelang_cuda_wgmma_operand_layout.py`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/cuda/test_tilelang_cuda_wgmma_operand_layout.py)：K-panel stride 从实际布局提取，切片不能按自身 extent 重建父存储间距。Frisk 将其作为 descriptor/address 一致性反例输入，不宣称该正确性要求是 Frisk 独创。
- Triton [`AccelerateMatmul.cpp`](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)：MMA encoding、accumulator 转换和 MMAv3 的 A-register/shared、B-shared 路径已有实现。Frisk 的设计差别在于将该类选择形成有限域联合硬契约，并在物化后对实际 IR 独立核验，不是声称 Triton 没有这些 operand/转换能力。

硬件规则依据为 2026-09-19 访问的 [NVIDIA PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：§9.7.17.5.1 的 WGMMA fragment 与 shared canonical layout、§9.7.17.5.1.2.2 Matrix Descriptor Format，以及 §9.7.17.5.2 指令定义。实现注释和测试按该版本章节及图像文件名定位，不依赖会随版本变化的图号；已有 A 链路报告中的旧 Figure 148 不能当作新版图号。

#### 20.11 文件分工及原提纲适配

实际新增 `Analysis/MmaLayoutConstraints.{h,cpp}`、`Analysis/InstructionLayoutConstraints.{h,cpp}`、`Target/SM90/SM90GemmConstraints.{h,cpp}`、`Target/SM90/SM90MmaLayoutProof.{h,cpp}`，以及硬件、通用联合约束、真实集成单元 suite 和 lit。IR 层在 `FriskLayoutOps` 定义 MmaOp，在 `FriskLayoutAttrs` 定义操作指令契约及 descriptor-plan 属性；类型化字段的结构 verifier 与目标合法性 verifier 分开。

修改 `LayoutTarget`、`LayoutConstraint`、`LayoutPropagation`、`LayoutVerifier`、`DistributedLayoutConstraints`、`OperationLayoutConstraints`、`MaterializeLayouts` 和相关 CMake，接入 instruction hyperedge、解中的操作 binding 及实际 singleton 核验。原预计放进 `LayoutRelations` 的通用 tuple 准备/支持检查独立放入 `InstructionLayoutConstraints`，不把四元约束压成二元关系。保留 Analysis→IR、Target→Analysis 的依赖方向；Analysis 使用 target 抽象接口，不增加 Analysis→SM90 链接。

#### 20.12 执行计划与记录（2026-09-19）

**Goal:** 将 §20.1–20.11 的 MMA 布局契约接入真实推断、物化和独立验证。

**Architecture:** IR 定义数学操作与类型化契约；SM90 提供有界候选和纯指令证明；Analysis 只执行通用角色图和有限 tuple。依赖保持 Target→Analysis→IR。

**Tech Stack:** C++17、MLIR/TableGen、GoogleTest、lit/FileCheck、CMake。

**Global constraints:** 原目录 main，保留 Task 19 修改，不 commit/push/worktree；8 variables/component、4 candidates/domain、256 编码组合、65536 枚举点；目标与 dtype、线程、SS/RS 和失败边界严格遵守 §20.3–20.8。

执行使用 subagent-driven-development：独立的 IR/硬件证明子任务由专门实现者处理，主执行者负责通用图集成；分别测试和复审后执行全量 Gate。提交步骤由用户“不自动提交”要求替代为保留工作区修改。

- [X] **20A：IR 和 SM90 纯证明。** 修改 `include/Dialect/Frisk/IR/FriskLayoutOps.td`、`FriskLayoutAttrs.td` 及对应 `lib/Dialect/Frisk/IR` 实现；新增 `include/Dialect/Frisk/Target/SM90/SM90GemmConstraints.h`、`SM90MmaLayoutProof.h` 和对应 cpp。先添加 `unittests/Dialect/Frisk/Layout/GemmConstraintTest.cpp`、`SM90MmaLayoutProofTest.cpp` 的数学 verifier、已知 fragment 坐标、SS/RS 地址反例并记录 RED；实现后分别 GREEN。目标桥接口是 `prepareSM90MmaCandidates(LayoutConstraintGraph &) -> LogicalResult`、`buildSM90MmaContract(const LayoutConstraintGraph &, const LayoutConstraint &, ArrayRef<Attribute>) -> FailureOr<Attribute>`、`verifySM90MmaContract(const LayoutConstraintGraph &, const LayoutConstraint &, ArrayRef<Attribute>, Attribute) -> LayoutProof`。它们不改 IR，最后一个不生成候选或搜索方案。
- [X] **20B：通用联合图。** 修改 `LayoutConstraint.h/.cpp`、`LayoutTarget.h`、`LayoutVerifier.h/.cpp`、`LayoutPropagation.cpp`；新增 `Analysis/MmaLayoutConstraints.h/.cpp`。`InstructionLayoutContract` 保存 source、完整实际 binding 及角色顺序对应的 `{encodings,binding}` tuples；`LayoutSolution::instructionBindings` 按 constraint ID 记录方案。用手工有限域测试无 singleton 删除、两两可行但整体不可行、重复端点、未知类别和 ID 重排，先 RED 再实现。tuple 检查对每个已赋值端点要求相等，未赋值端点要求仍在当前域，完整 assignment 必须支持同一 tuple。测试命令 `cmake --build build --target FriskLayoutUnitTests --parallel 16` 后运行 `build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='InstructionConstraintTest.*'`。
- [X] **20C：真实 pass 和物化。** 修改 `DistributedLayoutConstraints.cpp`、`OperationLayoutConstraints.cpp`、`SM90LayoutTarget.cpp`、`MaterializeLayouts.cpp` 及 CMake 接线。target 抽象提供上述 prepare/build/verify 三个对应 virtual 方法，默认拒绝未知指令。RelationsOnly 在 prepare/build 前返回；实际 binding 读取属性并调用纯 verify。先用 `test/Transforms/infer-gemm-layout.mlir` 与错误用例触发未支持操作，再实现 A/B/init/result 角色、线程约束、独立 encoding、方案写回和原子回滚。分别核验 `64×64×16` 和 `128×128×64` SS/RS、两种 dtype、transpose/phase/线程篡改、二次运行与打印解析回放。
- [X] **20D：复审、Gate 和文档。** 对 20A、20B/20C 做契约与代码质量复审；运行 §20.9 全部命令，记录新增测试、实际结果、限制及与原契约差异；同步比较文档。未跑的 GPU/上游测试明确标未执行。

#### 20.13 实现、复审与验证记录（2026-09-19）

**工作区与交付范围。** 直接在 `/home/baopeihua/frisk` 的 `main` 修改，保留进入任务前的 Task 19 未提交内容，没有新 worktree、commit、merge 或 push。本轮实现的是静态数学 MMA 的布局推断/验证闭环，不是可运行 WGMMA lowering。Task 21/22 未开始。

**实际接入：**

1. `MmaOp` 验证 rank/static shape、矩形 transpose、浮点输入/累加匹配和已有 encoding；MemRef 只读，init 显式 SSA。`MmaInstructionContractAttr`、`MmaDescriptorPlanAttr` 使用严格 Dictionary schema，拒绝缺字段、未知字段及不匹配的 packing/form。
2. `SM90GemmConstraints` 生成 canonical fragment、有限 shared 建议及确定性的 descriptor 方案；`SM90MmaLayoutProof` 根据 PTX 坐标、packed 顺序和实际 root-relative 地址独立证明，不用候选缓存代替证明。Generated proposal 不增强根对齐。
3. `InstructionLayoutConstraints` 提供冻结域的有界 tuple 准备和完整 tuple 支持检查。重复端点只枚举一次；四个 4 候选域正好至多 256 次；Strict/Common 不依赖 singleton 才触发；partial solver 必须有同一完整 tuple 扩展。
4. `MmaLayoutConstraints` 收集四个真实角色；方案写入 `LayoutSolution::instructionBindings`。Tensor producer 与 use 分离、init-use/result SameLayout；多 MMA 共用 storage 合并有限建议来源。原 ResourceLimit 与 8 vars/4 candidates 上限保持。
5. 事务物化写 encoding/storage/typed contract/threads，必要时仅在 Tensor 消费点插转换。actual-only 重建实际 singleton，绕过候选生成、tuple 枚举及方案构造，纯验证通过才提交。

**测试覆盖与证据分层：**

- 新增 36 个单元：`GemmConstraintTest.cpp` 7、`SM90MmaLayoutProofTest.cpp` 8、`InstructionConstraintTest.cpp` 9、`GemmLayoutIntegrationTest.cpp` 11，以及 legacy adapter 的 RS replication 分类 1。新增 2 份 lit：`infer-gemm-layout.mlir`、`infer-gemm-layout-errors.mlir`。
- 真实推断覆盖 `64×64×16` 的 SS/RS × f16/bf16 × transA/transB 共 16 种组合，以及 `128×128×64` 的两种 form × 两种 dtype 共 4 种组合。大 tile 同时验证二次推断、打印/解析回放及 actual-only 无生成调用。
- Helper 独立坐标表覆盖 A packed halves、accumulator、M/K repeat、N-group replication、全部受支持 atom N；descriptor 覆盖 K/MN × 无/32/64/128-byte swizzle 共 8 模板、父 K-panel 间距、非零 start/phase、对齐/越界反例。线程与 policy helper 覆盖 128/256/512/1024 和三种 policy；不宣称全部 descriptor × transpose × 多 warp-group 的端到端笛卡尔积已运行。
- 实际 IR 负例覆盖缺属性、线程/target 冲突、合法 typed grid/descriptor 篡改、仍满足通用 coverage 的 fragment 位交换。错误 RS packing 在 typed checked-constructor 层拒绝；直接注入原始 Dictionary 也不能绕过 actual-only 的 typed 属性要求。测试累计检查候选枚举/准备/方案构造次数均为 0。
- 外部 A/init 编码保留，并在消费点产生两个转换；失败事务保留原 IR 逐字不变；两个 MMA 共用存储域、未知指令、重复端点、tuple 去重/remap、256 组合与第 5 候选拒绝、逻辑/硬件点数预算均有测试。

**复审与修正。** 分别进行了通用图集成和硬件/全链路独立源码复审，未发现已确立的 Critical/Important 正确性缺陷。验收覆盖缺口是实际 fragment/packing/descriptor/grid 篡改，已补测试；另修正数学 verifier 对未知 Tensor encoding 的漏检、失败角色标注，以及显式 binding 失败时丢失 Unknown/reason/坐标的诊断。早期预算失败现在明确为 Unknown；无可构造方案报告 bounded decoder 无法证明，不冒充硬件不可能的数学反例。生成 API 仍返回 `FailureOr<Attribute>`，一般解码失败只能给有界构造失败说明，不能承诺总有逐点反例。

**测试先后记录。** 未注册 `frisk.mma`、未知 InstructionContract 被错误接受、未知 Tensor encoding、角色诊断、Unknown 诊断均观察到预期 RED 后修正。部分硬件 helper 用例虽先写测试，但首次可运行构建时已为 GREEN，不能宣称每个 helper 都单独完成过可执行 RED。实际 packing 负例最初误用会 assert 的 `get()`；改为 `getChecked()` 验证结构拒绝，这属于测试构造修正，不是生产路径接受了非法 packing。

首次全量单元运行在新增 Parallel fixture 构造中触发空 region `front()` 断言；单测复现和带符号栈定位到测试自身。原因是现有 `ParallelOp::build` 只添加 region、不创建 block；测试现已显式构造两个 index 参数的 block 和 `EndOp` 再放入 MMA。未因此改动旧 Parallel builder 或放宽生产 verifier；该次中止不能计作全量通过，后续重新执行完整 Gate。

**Legacy 分类而非机械一致：**

| 历史输入/行为                            | Task 20 处理与证据                                                                                                            |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 适用的旧 SM90 SS/RS storage 与 fragment  | 保留原 adapter 持有关系/地址回归；不以属性文本相等或旧 f16 accumulator 冒充新 f32 数值等价                                    |
| 旧 RS A 的 replication=2                 | 新增分类测试：旧候选被新指令证明拒绝；128 线程新 canonical A 为 replication=1，不能沿用旧冗余组织；B storage 地址仍可独立对照 |
| SM80 或仅`sm_90` target                | 新 MMA 只接受显式`sm_90a`；旧 SM80 路径仅保留 legacy 回归，不扩展本阶段 target                                              |
| Local A/B、Tensor B                      | Local MemRef 不当寄存器；RS 必须 Tensor A，B 必须 Shared。明确拒绝，生命周期 normalization 留到 Task 22                       |
| 非四 warp 分组/错误线程 scope            | 按合法 grid、连续 warp-group 和实际 Parallel scope 检查；不为旧 policy 输出放宽硬件边界                                       |
| 旧 positional ABI、clear/annotation 行为 | 不复制位置下标或完成标记；新 op 用显式 init SSA 表示旧值依赖，不提供隐式 clear。旧 ABI 自动转换不在本任务                     |

**最终 Gate：** 以下命令重新执行并通过，单元总数较 Task 19 的 129 增至 165。全量单元耗时 207207 ms，23 个 suite；其中 MMA 集成 suite 为 196996 ms，不是 GPU 性能数据。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
git diff --check
```

| 验证项                   | 最终实际结果                                                       |
| ------------------------ | ------------------------------------------------------------------ |
| 构建及 lit               | exit 0；34/34 PASS（9.46 s）                                       |
| 全量单元                 | exit 0；165/165 PASS（207.207 s）                                  |
| CTest legacy executables | exit 0；4/4 PASS                                                   |
| `git diff --check`     | exit 0，无 whitespace 错误                                         |
| 独立复审                 | 通用图与硬件/全链路各自审阅；覆盖/诊断问题修正后复核，无剩余阻塞项 |

构建观察到已有 Ninja `premature end of file; recovering` 和 CMake CMP0116 OLD 弃用提示，不宣称零警告构建。负例测试中的预期错误诊断不计为失败；最终通过以测试汇总与 exit code 为准。

**未执行/不支持：** 未运行 TileLang/Triton 上游测试、GPU 数值内核或性能 benchmark；未升级固定上游 SHA。一般 Product 指令布局、动态/非二次幂/尾块、Tensor B、Local fragment、f16 accumulator、TF32/FP8/int/sparse、SM80 新路径、异步 token/同步 lowering、寄存器分配/spill/occupancy 和完整 CostVector 均未实现。本任务 65536 点是证明预算，不是寄存器性能预算；policy/swizzle 排序不代表性能最优。

以下保留最初的 Task 20 迁移提纲用于追踪演进，复选框不代表当前验收进度；以上述 v1 契约和 A–H 为准。其中“DotOperand/Mma encoding”按 §20.7 落实，寄存器 B 不在 SM90a 支持集，旧 Gemm 自动 normalization 留到 Task 22。原 Step 6 不自动执行，提交和推送仍需用户另行授权。

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90GemmConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/GemmConstraintTest.cpp`
- Create: `test/Transforms/infer-gemm-layout.mlir`

**Interfaces:**

- Produces internal operand type constraint and op：

```tablegen
def Frisk_MemRefOrRankedTensor :
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>;
```

```mlir
%result = frisk.mma %a, %b, %init
  {m = 128, n = 128, k = 64, trans_a = false, trans_b = false,
   policy = #frisk<gemm_warp_policy Square>}
  : (memref<128x64xbf16, #frisk<memory_space Shared>>,
     memref<64x128xbf16, #frisk<memory_space Shared>>,
     tensor<128x128xf32>) -> tensor<128x128xf32>
```

```cpp
LogicalResult collectGemmConstraints(MmaOp op,
                                     LayoutConstraintBuilder &builder,
                                     LayoutTarget &target);
```

- [ ] **Step 1: 写 MMA verifier 和 constraint 红灯测试**

覆盖 BF16/FP16 `m64nNxk16` SM90 候选、A/B major/transpose、accumulator shape、128-thread warp-group、非法 dtype/shape，以及不同 warp policy。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，内部 MMA Op/constraints 尚不存在。

- [ ] **Step 2: 定义 target-neutral `MmaOp`**

Op 只保存数学 tile 语义和 policy，不保存 PTX mnemonic。A/B 可为 Shared MemRef view 或 RankedTensor；init/result 必须是 RankedTensor。Verifier 只检查 shape/dtype 的数学一致性，SM90 合法性放 target rules。

- [ ] **Step 3: 审核新旧 Gemm 语义后提取 canonical candidates**

以 `FriskOps.cpp` 的旧 Gemm target/layout 构造为迁移输入（历史定位 1285–1406 行，实施时按符号重新定位），对照固定新版 TileLang 的参数/annotations、read-before-write 和硬件契约审核后迁入 `SM90GemmConstraints.cpp`。输出 `InstructionContract`、`RequireEncoding` 候选和 provenance；不得直接写 layout map，也不得机械搬用旧 positional args 下标。

```text
A/B shared -> Storage candidates + WGMMA descriptor contract
A/B tensor -> DotOperand distributed candidates
result/init -> Mma/Distributed accumulator candidates
```

- [ ] **Step 4: 与 legacy Gemm 枚举差分**

对现有 `sm80_ss`、`sm90_ss` 和 local/shared 组合先分类：已知正确且适用当前 SM90 契约的 case 要有数学等价候选；SM80-only case 仅作 adapter 回归，不能因此扩展本阶段 target。旧 bug、非法 owner/地址、已过时协议按独立坐标和硬件契约修正，记录新旧输出、原因和反例，不要求复现旧错误。新版新增行为单独测试；新系统可保留其他合法候选，未运行成本实验不能称其更优。

- [ ] **Step 5: 运行 tests**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='GemmConstraintTest.*'
```

Expected: PASS；非法 SM90 contract 输出目标规则名称。

- [ ] **Step 6: 提交 Gemm 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: model gemm as layout constraints"
```

### Task 21: 增加 Tensor Reduce Op 并迁移 Reduce ownership

**状态（2026-09-19）：用户已确认 §21.1–21.7；受限实现、独立复审及 197 unit / 36 lit / 4 CTest 回归已完成，完整链成功验收仍未完成（9 变量与 8 变量上限冲突，见 §21.8）。实施基线为本地 main `c028e82`，直接在源目录实施，未创建 worktree、未 commit/push。原 Step 1–6 保留为历史提纲，以已确认的具体契约及实际验收记录为准。**

#### 21.1 实现审计：哪些可以复用，哪些不能直接迁移

| 核对对象                                       | 实际发现                                                                                                                                                                       | 对 Task 21 的影响                                                                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `FriskOps_Reduce.cpp::ReduceOp::inferLayout` | 旧输入/输出都是 Local MemRef；依赖已有 src layout，主要进行 src→dst 的维度消除与 replication condense                                                                         | 复用逻辑投影思想，不复用 Local MemRef/DenseMap 生产路径，不宣称已有一般双向推断                                                               |
| `FriskOps.cpp::ReduceOp::verify`             | 旧 kind 为 add/mul/min/max；检查删维 shape 和同 dtype，旧操作还有目的端/clear 语义                                                                                             | 新 Tensor Reduce 的 sum 命名、纯值语义与旧 add/clear 的自动映射必须分开，自动 normalization 留给 Task 22                                      |
| `test_pass/reduce_layout_test.cpp`           | 8 个参数用例及 GEMM→Reduce legacy 用例；case 5 注释明确指出 src map 有问题；case 7 的实际 index 为`d1*4+d1`、thread 为 `d2`，均不编码 batch 维 d0；case 6–8 使用 16 线程 | 不能以旧测试 PASS 证明输入持有关系合法。case 7 不同 batch 会落到相同物理槽；这些用例要分类或重新构造独立 oracle，不能机械复制期望 replication |
| `LayoutRelations.cpp::permuteEncoding`       | 当前 TransformLayout 接收等 rank 的轴置换，反向用逆置换；归约删维并非双射                                                                                                      | 不把 Reduce 塞进现有 permutation payload，不伪造逆矩阵补回已消失的 reduction 维                                                               |
| Task 19 execution proof                        | 已能验证静态覆盖、重复持有者、first_owner 和线程数；同一元素的多个持有者默认表达同值副本                                                                                       | 需要另证“每个不同逻辑输入恰好贡献一次”；普通 coverage/replication 检查不能单独证明 sum 没有多算                                             |
| Task 20 instruction tuples                     | 四角色及 MMA source 的不变量是明确契约，不是任意 arity 的通用计算节点                                                                                                          | Reduce 采用有名的二端点关系和独立方案绑定；不复制端点凑成四元 MMA tuple                                                                       |
| 当前公共布局表示                               | Distributed pass 支持静态非零 rank、每维大于 1 的二次幂 BitLinear；已有 8 vars / 4 candidates 和 65536 点预算                                                                  | 原提纲中的非二次幂 guarded/shared fallback 不能只加一个分支就实现，需要先扩展表示与边界证明                                                   |

固定上游证据再次核对本地审计副本的 HEAD：TileLang `5e149e31674658f94779c7d0c6039549a1853123`、Triton `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c`。没有刷新浮动 main，也没有运行上游测试。GitHub 固定源码网页本轮读取返回 cache miss，源码判断来自同 SHA 本地副本，不将其他本地 TileLang 分支误当该快照。

- TileLang `src/op/reduce.cc::ComputeReducerLayout/InferLayout` 仍是传统 Fragment 的源驱动投影、压缩和目的包含性检查。`src/layout/layout.h::PartialFragmentNode` 明确区分 addend lanes 与 equal-value copy groups，并禁止将 partial 当作普通 replica；这与 local.reducer epoch 是另外一层机制。
- Triton `inferReduceOpEncoding` 使用 SliceEncoding；`ReduceOpToLLVM.cpp` 先处理寄存器内归约，再处理 warp 内归约，其余跨 warp/block 情形借助布局转换与同步。它已经具有分层归约实现，不能把 register/warp/shared 分层本身宣称为 Frisk 独创。
- 本地 MLIR `arith.maximumf/minimumf` 与[官方 Arith 文档](https://mlir.llvm.org/docs/Dialects/ArithOps/)均明确 NaN 传播及正负零次序；新 kind 不能无说明地混用 maxnum/minnum。

#### 21.2 方案取舍

| 方案                                                       | 收益                                                                          | 问题与结论                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| A：坐标投影＋贡献者证明＋有界通信方案（推荐）              | 保持 Tensor/Storage 双域、有限候选、实际 IR 独立验证；能接上 Task 20 MMA 输出 | 首版限定现有静态 BitLinear 支持集，需要新增归约专用关系，但改动边界可控 |
| B：只迁移旧 dst map/replication 公式                       | 实现最短                                                                      | 无法证明重复输入未被多算，难以独立核验通信需求，不作为完整 Task 21 交付 |
| C：同时引入 PartialFragment/epoch、非二次幂及 GPU lowering | 支持范围更广                                                                  | 扩展类型/语义/调度多个系统，超出当前单任务，单独规划                    |

推荐 A。此选择包含对原提纲的明确缩限：非二次幂 fallback 改为可测试的 unsupported 边界，待支持表示/guard 的后续任务补齐；不把 Shared 通信方案误说成已经支持 ragged shape。

#### 21.3 已确认的数学操作与支持集

```mlir
// 已注册操作的未编码输入示例；布局由推断补齐。
%result = frisk.reduce_tensor %src {kind = "sum", dim = 1}
  : tensor<64x64xf32> -> tensor<64xf32>
```

1. 单个 RankedTensor 输入与单个 RankedTensor 结果；沿 dim 归约，结果 shape 等于删除该维后的 shape；无 keep_dims、隐式 init/clear、目的 MemRef 或写回副作用。操作是纯 Tensor 值计算，外部累加另用显式算术操作表达。
2. 首版 kind 为 `sum/max/min`，dtype 为 f16/bf16/f32，输入与结果同 dtype；没有隐式 f32 提升。旧 `add` 不作为新 op 的第二套公开拼写，旧 `mul`、abs/bitwise、自定义 combiner、argmax/argmin 不包含在首版。
3. `sum` 按每个不同逻辑输入恰好一次定义，允许并行重结合，选中方案固定确定性的归约树；不承诺与串行从左到右相加逐 bit 一致。这是新操作本身的显式语义选择，不为普通 arith.addf 擅自添加重结合许可。CPU 测试须按指定树核对，不能用浮点交换律假设代替证明。
4. `max/min` 采用 maximumf/minimumf 的 NaN 传播和正负零语义，不采用“忽略 NaN”的 maxnum/minnum。NaN payload 不作为跨不同实现的逐 bit 保证。
5. 首版输入 rank 至少 2，结果 rank 至少 1；所有 extent 为大于 1 的静态二次幂。rank-1→scalar、extent=1、动态、非二次幂明确拒绝，不伪造 shape=1 的输出绕过既有 verifier。
6. 限单 CTA、32/64/128/256/512/1024 线程，lane=32。显式 `frisk.execution_threads` 与外层 Parallel 必须一致；没有 Parallel 时优先显式值，再继承可确定的输入编码/生产者图线程要求，否则缺省 32。例如 Task 20 的 128 线程 MMA→Reduce 默认继承 128，不强行先转成 32。多个互相冲突的来源不取任意第一个。
7. target 按当前 SM90 target 模型处理：支持 `sm_90/sm_90a`；缺失时沿用普通操作默认 SM90，不为 Reduce 额外要求 MMA 的 `sm_90a`。但同图 MMA 仍独立要求显式 sm_90a，其他显式 target 拒绝。
8. result 的多个持有者必须表示已经完成的同一个归约值，而不是尚未合并的 partial。只有 layout 无法证明动态 payload 相等；相等是 Distributed Tensor 的语义契约，不将任意不等的 runtime 部分和当作 replica。

#### 21.4 投影与贡献者证明

设 `Dsrc(h)` 为源硬件位置 h 对应的逻辑坐标，`P` 为删除 dim 的逻辑投影；输出坐标 y 的归约集合为：

```text
F(y) = {x | P(x) = y}
```

证明必须区分三个对象：F(y) 中不同的输入元素、同一个输入 x 的多个物理副本，以及完成后输出 y 的多个物理副本。只有第一类是 sum 的不同加数。

推荐先由 `P∘Dsrc` 构造自然输出布局，再压缩投影后冗余的 register 位：按旧 register 位顺序选取线性独立的列，保留 lane/warp/warp_group/CTA 的组织，重新计算 register extent 和实际 replication。不能只删输出行后沿用旧 replication；不能只删除零列而忽略非零相关列。通用投影 API 可复用，但 Reduce 的删维、重命名和 register 商空间处理应有独立 helper。

对每个 y，按 `(cta,warp_group,warp,lane,register)` 的字典序选出每个不同 x 的最小源持有者一次，并按此物理键排序。`canonical_fiber_tree_v1` 每层将相邻项两两合并，奇数尾项原样进入下一层，合并结果保留在左项的代表持有者；重复直到一个根，再将完成值从根分发到 Ddst 声明的所有输出持有者。证明检查输入完整、无重复贡献、各次合并的贡献集合不相交、最终集合恰为 F(y)，并检查完整结果的分发覆盖。first_owner 是本版本的确定性基线，不宣称通信量最小；不能让输出副本数量乘进求和。

以每行 4 个不同值 `[1,2,3,4]`、每个值均有两个物理副本为例，行 sum 必须是 10，不能是 20；两个输出持有者应各持有 10，而不是分别持有未完成的 3 和 7。即使 max 测试对重复加数不敏感，sum 反例也必须覆盖。

通信范围按实际树边及完成值分发边推导：同线程为 register，跨 lane 同 warp 为 warp，跨 warp/warp_group 为 CTA shared-tree。只记录依赖与通信要求，不生成 shuffle、shared alloc 或 barrier。含 shared-tree 的证明不能表述为已经完成可执行同步/共享容量分配验证；其含义是单 CTA 内存在该有限通信依赖，后续 lowering 必须落实资源和同步。

#### 21.5 图接线、有限候选与方案绑定

- 新增有名二端点 `ReductionLayout` 关系，端点固定为 `[src-use,result]`，payload 保存 source op、dim 和实际归约绑定；结构检查覆盖角色、类型、归属、ID remap。保留现有 TransformLayout 的轴置换语义，不影响 Task 20 四元不变量。
- src producer→src-use 使用已有 Convertible；自然 result→其他消费者使用既有 use/转换关系。不能直接修改外部 producer 来满足这一次归约，也不能无条件认定所有 input/output encoding 组合可行。
- 正向只从已允许的源候选生成自然输出布局；反向根据输出要求筛选已有源候选，不把投影当可逆映射，不凭空恢复源布局。下游消费者的不同布局要求通过消费者边转换；若 Reduce 自身结果的硬编码与所有合法源候选的自然投影均不兼容，则报冲突，不能靠消费者转换掩盖，也不能覆盖硬绑定。自然布局的逻辑维名称允许按结果轴位置作语义等价重命名，不据此接受任意结果映射。
- 候选准备需把 Reduce 正向投影接入现有有限来源闭包，支持 reduce→reduce、transpose→reduce 和 MMA→reduce；result 不先塞无关默认候选。来源包含显式编码、前驱合法传播和既有 target 默认族，来源去重并记录 provenance。冻结后 Strict/Common 只删不增，每次删除重排相关约束；部分和完整 assignment 都检查 pair 支持。
- 保留 8 vars/component、4 candidates/domain；每条二端点关系最多 16 对布局，不通过复制变量或截断候选绕过限制。单个输入/输出的逻辑与硬件枚举分别最多 65536 点；贡献关系按源点和目标点线性建索引，不构造 65536² 的笛卡尔积。超预算返回 Unknown。
- 规划 `ReductionLayoutConstraints.{h,cpp}`、`ReductionLayoutProof.{h,cpp}`（Analysis）与 `SM90ReduceConstraints.{h,cpp}`（Target）。前者处理逻辑 fiber/候选关系/贡献证明，后者处理线程组织与通信方案；仍保持 Target→Analysis→IR，不在通用 solver 散布 target 字符串。
- 规划操作属性 `frisk.reduction_contract : ReductionContractAttr` 和 `frisk.execution_threads : i64`。采用完整、严格的 v1 schema：version、target、kind、axis、dtype、输入/输出 shape、threads、算法版本 `canonical_fiber_tree_v1`、输入副本选择规则、输出完成值分发规则、通信范围。规范算法从实际编码唯一确定详细依赖，无 pointer/cache ID；无法支持的算法字段拒绝，不静默更换方案。
- 解中以稳定 constraint ID 保存 reduction binding，物化到真实操作；显式完整属性只验证、不覆盖。最后 actual-only 从实际输入/结果类型和属性检查 shape、投影、贡献集合、通信范围与线程一致性，允许执行纯确定性证明，禁止候选准备、pair 枚举、重新搜索方案或复用旧解缓存。
- 沿用 detached module 原子物化、打印解析与二次推断稳定性；写回前失败保持原 IR 不变。后续 tile_store 的唯一写入者检查复用既有 ownership 约定；Reduce 本身不是 MemRef 写操作。

#### 21.6 验收分组与旧行为分类

- [X] 数学 verifier：所有 kind/dtype、dim 越界、删维 shape、输入/结果类型不匹配、rank-0/动态/非二次幂拒绝、纯副作用及 printer/parser；sum/max/min 的语义差异和 NaN/±0 分类有独立 CPU oracle。
- [X] 布局代数：reduce 首/中/尾轴、register 位压缩、相关列/XOR 混合、coverage、replication 重算及自定义维名；不能由待测候选生成器生成唯一“期望值”。
- [X] 贡献证明：线程内、跨 lane、跨 warp、跨 warp-group，输入/输出副本分离；sum 漏算/重复、错误 partial、跨行混合、缺失结果分发、thread scope 和预算负例；CPU 模拟检查每个输出完整贡献集合。
- [X] 真实推断的已通过子集：`64×64→64`、3D 删中轴、`128×128` MMA→按行 Reduce、Reduce→Global tile_store、连续两次 Reduce、外部显式编码转换、SCF/Parallel 环境；保留 bootstrap 上限。
- [ ] 原契约完整链成功验收：Task 20 `128×128` MMA 结果→按行 Reduce→tile_store。当前只验证 9 变量超限拒绝，分段成功不替代此项；须另行确认预算或等价压缩策略。
- [X] 传播与独立核验：输出要求反向删源候选、无支持域失败、固定点统计、稳定 ID；缺少/篡改 reduction_contract、线程/通信范围、源/结果编码均 fail-closed；actual-only 生成计数为 0，事务回滚和回放逐字稳定。
- [X] Legacy 分类：case 1–4 作为适用候选，仍先检查完整持有关系；case 5 的不一致 replication 不迁入正确性 oracle；case 7 的 batch 丢失作为反例并另建合法 3D 样例；case 6/8 的 16 线程不伪装成新支持集，补真实 32-thread 同值复制后的独立样例。旧 add→sum、mul/clear、SM80 GEMM 来源均记录支持/延期，不以旧文本相等为验收。
- [X] 全量回归 Gate：新增 `ReduceConstraintTest.cpp`、`ReductionLayoutProofTest.cpp`、`ReduceLayoutIntegrationTest.cpp` 和 `infer-reduce-layout{,-errors}.mlir`，197 unit / 36 lit / 4 CTest 全通过；独立复审与未完成项见 §21.8。未执行 GPU/上游实验；本项通过不消除上述完整链验收缺口。

#### 21.7 确认点与实施顺序

用户已确认方案 A 的支持集和语义——**sum/max/min、静态二次幂 Tensor、贡献者去重、自然投影及寄存器压缩、register/warp/CTA 通信契约和实际 IR 复验；sum 允许并行重结合，非二次幂 fallback、PartialFragment/epoch 与 GPU lowering 延后**。

按四步落地：数学 IR 与独立投影/贡献 oracle → 有名二端点关系及候选闭包 → SM90 通信属性与事务物化/actual-only → MMA→Reduce 集成、legacy 分类、全量回归和文档。上述受限实现与复审修正已完成；165/34/4 是已提交基线，本轮实际回归为 197/36/4。完整链成功验收尚未完成。

#### 21.8 实际实现与验收边界（实施记录）

**工作区。** 基线为本地 `main c028e82`，本轮直接修改原目录；保留既有 Task 21 设计修改，不创建 worktree，不自动 commit/push，不修改独立汇报 README。

**实现接线。**

| 层次       | 实际代码与责任                                                                                                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| IR         | `FriskLayoutOps.td/.cpp` 新增纯 `ReduceTensorOp`；`FriskLayoutAttrs.td/.cpp` 新增完整、严格的 `ReductionContractAttr`                                                                   |
| 基础证明   | `ExecutionLayoutProof.h/.cpp` 提取共享的有界坐标枚举；`ReductionLayoutProof.h/.cpp` 做自然投影、register 子空间基选取、规范贡献树和完整输出分发                                             |
| 图关系     | `ReductionLayoutConstraints.h/.cpp` 收集 `[src-use,result]`、解析线程环境、冻结后构建最多 16 对合法编码和 binding；`LayoutConstraint` 检查实际 SSA 角色、axis、来源、pair 唯一性与稳定 ID |
| Target     | `SM90ReduceConstraints.h/.cpp` 解释 SM90/SM90a、线程与通信范围；通过 `LayoutTarget::buildReductionContract/verifyReductionContract` 接入，不把 target 字符串判断塞进通用 solver             |
| 传播和求解 | 正向生成自然输出，反向仅以已有 pair 删减；Strict/Common 重排受影响约束；部分和完整 assignment 都要求 pair 支持；`LayoutSolution::reductionBindings` 按稳定 constraint ID 保存方案             |
| 物化与复验 | `MaterializeLayouts.cpp` 写 `frisk.reduction_contract` 和 `frisk.execution_threads`；实际 IR 路径不调用候选生成、pair 准备或 binding 搜索；失败不更新原 IR                                |

`ReductionContractAttr` 的 v1 字典准确包含 12 个字段：`version/target/kind/axis/dtype/source_shape/result_shape/threads/algorithm/input_policy/output_policy/scope`。算法固定为 `canonical_fiber_tree_v1`，输入策略固定 `first_owner`，输出策略固定 `broadcast_complete`；scope 为 `register/warp/cta_shared_tree`。物理位置序列、相邻合并树和广播目的地由实际编码唯一重建，不存 SSA 指针或缓存编号。

**明确的验收差异：完整 MMA→Reduce→store 链。** 实施时发现原 §21.6 同时要求完整链成功和保持 8 vars/component，两者在当前图模型下冲突。SS MMA 有 A/B storage、init producer/use、result 共 5 个变量；Reduce 新增 source-use/result 2 个；store 新增 use/storage 2 个，总数为 9，且处于同一连通分量。已向用户说明这是原契约遗漏，8 是 bootstrap 求解规模限制而非硬件限制。未获得放宽上限的明确授权，因此保持 8，分别验证 `128×128 MMA→Reduce`、`Reduce→Global tile_store`；完整链验证 9 个真实变量及超限拒绝，不能宣称完整链成功。后续若需完整链，应单独确认预算或等价变量压缩策略。

**验证方法。** 数学、贡献证明、图集成分别由 `ReduceConstraintTest.cpp`、`ReductionLayoutProofTest.cpp`、`ReduceLayoutIntegrationTest.cpp` 覆盖；lit 正反例为 `infer-reduce-layout{,-errors}.mlir`。CPU oracle 按声明的树模拟求和/最大/最小，包含 NaN、正负零和重结合反例；这不是 GPU 数值执行。旧 case 1–4 独立反演 thread/slot 关系，case 5 错误 replication 和 case 7 batch 冲突不作为正确性 oracle，case 6/8 明确区分旧 16 线程与新 32 线程同值副本适配。

**复审记录。** 基础证明首轮无 Critical/Important；按 Minor 建议增加物理顺序不同于逻辑顺序的完整树 oracle、逻辑/硬件预算边界以及 API 数值语义前置条件说明。图复审发现下游候选反向污染自然输出、线程环境越过已确定转换边界的问题；已用回归复现并修正。硬编码端点不再接受不可能的传播候选，归约自然结果不接受反向消费者建议；线程追溯在已确定的 encoding/执行环境处停止，但仍核对 SameLayout 同伴的独立冲突。最终代码复审无新增 Critical/Important 实现缺陷；完整链未成功仍是 Important 验收缺口。另有 Minor 测试建议保留：目前 fanout 回归硬绑定了 Reduce 结果，未来可增加未编码结果的下游强需求用例，独立保护仅针对 `reductionResult` 的候选屏障。

**测试命令与当前证据。**

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
git diff --check
```

本轮 build/check-frisk 返回 0，lit 36/36、CTest 4/4 通过；全量 unit 的 26 个 suite、197/197 项全部通过（262.577 秒，退出码 0）。新增单元为 4 项数学/schema、13 项投影/贡献证明、15 项图集成，共 32 项；新增 2 份 lit 文件。既有 Ninja `premature end of file; recovering` 和 CMake CMP0116 弃用提示使构建重新执行较多目标，不作为 Task 21 功能失败，也未为此改写构建缓存或策略。CPU 布局与树 oracle、打印解析和独立验证不代替 GPU 数值/性能测试；本轮未执行 GPU 或上游框架实验。

以下为原始迁移提纲，实施以以上已确认的具体契约及实际记录为准；原 Step 6 不作为自动提交授权。

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90ReduceConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/ReduceConstraintTest.cpp`
- Create: `test/Transforms/infer-reduce-layout.mlir`

**Interfaces:**

- Produces：

```mlir
%result = frisk.reduce_tensor %src
  {kind = "sum", dim = 1}
  : tensor<64x64xf32> -> tensor<64xf32>
```

```cpp
LogicalResult collectReduceConstraints(ReduceTensorOp op,
                                       LayoutConstraintBuilder &builder,
                                       LayoutTarget &target);
```

- [ ] **Step 1: 写 ownership 红灯测试**

覆盖 reduce dim projection、lane shuffle、replicated result、elected owner、thread count 与 replicate 不整除、非 2 次幂 reduce extent 的 guarded/shared fallback。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，新 Reduce Op/constraints 不存在。

- [ ] **Step 2: 定义 `ReduceTensorOp`**

Verifier 检查 result shape 等于删除 reduce dim 后的 shape、dtype/kind 支持、dim 合法。Op 不直接选择 shuffle/shared tree。

- [ ] **Step 3: 将现有 Reduce map 逻辑转成 Transform/Ownership constraints**

`FriskOps_Reduce.cpp:235-369` 的 dimension projection、replicate condense 语义迁到 target-neutral transform；SM90 rule 决定 shuffle/shared tree candidate。无法证明 unique owner 时删除 store candidate并给反例。

- [ ] **Step 4: 与 legacy Reduce cases 差分**

复用 `reduce_layout_test.cpp` 已知正确的 case table，比较 output coordinate、replication 和 owner，而不是 Attr 文本。对照新版 TileLang 区分普通 Fragment 的同值 replica 与 PartialFragment 的 addend/copy groups；传统 Reduce 与 reducer epoch 是不同机制，不因后者更新就宣称前者全面双向化。先列出本任务实际接入子集和不支持诊断；旧错误用独立 reduction/owner 不变量修正并记录差异，不强求 legacy 等价。

- [ ] **Step 5: 运行 tests**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ReduceConstraintTest.*'
```

Expected: PASS；legacy executable 仍 PASS。

- [ ] **Step 6: 提交 Reduce 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: model reduce ownership constraints"
```

### Task 22: 实现 legacy normalization 并退役旧生产路径

**状态（2026-09-20）：基线为源目录 main `14b39e9`；用户已确认方案 A 及 §22.3–22.11 具体契约；生命周期转换、持久契约、pipeline 与旧 oracle 隔离已完成，独立复审通过，最终 242 unit / 48 lit / 4 CTest / 5 Python 测试方法及隔离扫描全通过。原 Step 1–7 保留为历史提纲；不自动提交或推送。**

#### 22.1 首轮实现审计

| 核对位置                                                                                                                 | 源码事实                                                                                  | Task 22 必须解决的问题                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `Passes.td`、`LayoutInfer.cpp`、`frisk-opt.cpp`                                                                    | 已有 infer/optimize pass，但没有 normalize pass 或具名 layout pipeline                    | 新增 pass、公开构造入口和显式 pipeline 注册，不能依赖链接器保留静态初始化                                                            |
| `FriskOps.cpp::GemmOp::verify`                                                                                         | 旧 verifier 要求 A/B/C 同 dtype，按未转置形状检查矩阵维度                                 | 新 MMA 的 f16/bf16 输入、f32 累加不能通过简单改名从这类已验证旧 IR 得到；必须区分导入接口兼容扩展与数值精度变化                      |
| `ffi/ir.cpp::create_gemm_op`                                                                                           | Python C++ 绑定同样要求 A/B/C 同 dtype，且按未转置形状推导 M/N/K                          | 只修改 C++ verifier 不足以使 Python 入口生成新支持的导入 IR；需明确同步范围及验证方式                                                |
| 旧 Gemm 的`clear_accum`、旧 Reduce 的 `clear`                                                                        | 都是 BoolAttr；旧布局推断没有解释执行时如何清零/与旧目的值合并                            | 不能从布局代码推断已经验证的运行时数值语义；需明确 init、覆盖与累加契约，禁止把未初始化 alloc 或 tensor.empty 当零                   |
| `FriskOps.td::ReduceOp`                                                                                                | dst 目前只标记 MemWrite，但若 clear=false 表示与旧 dst 合并，就需要读取旧值               | 必须同时核对数学语义与内存效应，不能只生成一个 reduce_tensor 就丢掉旧值依赖                                                          |
| Local MemRef 生命周期                                                                                                    | 原计划要求可证明无 alias escape 的完整 def-use slice；当前尚无相应转换器                  | 明确 alloc、初始化、Copy/Fill、Gemm/Reduce、读取/写回、SCF 携带与别名的支持矩阵；未知使用、读前未初始化、部分改写必须拒绝            |
| `FriskOps.td/.cpp`、`FriskOps_Reduce.cpp`、`FriskInterfaces.td`                                                    | 三个旧 inferLayout 方法与旧 DenseMap 接口仍在；新 infer pass 已经不调用它们               | 退役是删除生产 API 与依赖，不是再次宣称新 pass 首次与旧方法隔离                                                                      |
| `test_pass/attr_test.cpp`、`reduce_layout_test.cpp`、`LegacyLayoutAdapterTest.cpp`、`CopyFillConstraintTest.cpp` | 仍直接调用旧 inferLayout，不能直接删方法后保留测试原样                                    | 将 baseline oracle 隔离到测试专用支持代码或固定独立 fixture，保留迁移分类覆盖；不能靠删测试或让生产 pass 调用旧 oracle 过关          |
| `layout-alias-contract-lifetime.mlir`                                                                                  | Pure 的 whole-root layout_view 可被 canonicalize/DCE 删除，其子视图随后失去根对齐前置证据 | Task 18 遗留的 durable contract 需要纳入 Task 22：只保存已有且合法作用域内的保证，不得凭指令要求扩大根对齐，也不能让未知证据变成成功 |
| Task 21 集成与 bootstrap solver                                                                                          | MMA→Reduce→store 仍为 9 个连通变量，现有限制为 8                                        | normalization 不会自动消除该限制；不得将 normalize 成功等同于整个 pipeline 成功，也不得绕过图或擅自提高上限                          |

本轮检查的是 Frisk 自身源码与已有测试，未重新获取、运行或更新 TileLang/Triton 快照。旧归约的 clear 数值行为、旧 f16 累加与新 f32 累加的等价性均没有新增 GPU 证据。

#### 22.2 已确认的范围选择与实施原则

| 方案                                              | 做法                                                                                                                                       | 取舍                                                                                                |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- |
| A：严格语义边界的受限完整生命周期迁移（用户已选） | 为可证明的 Local fragment 生命周期建立导入契约；数值类型不静默变化，初始化显式化；补 pipeline、持久契约及旧 API 退役，unsupported 明确拒绝 | 保持 Task 19–21 目标边界与当前求解预算；不承诺所有旧程序都可迁移，Task 21 完整链成功验收缺口仍保留 |
| B：同时扩大数值与求解支持集                       | 除 normalization 外，实现旧 f16 累加语义的目标路径、更多旧 Reduce 变体，以及较大图求解                                                     | 覆盖面更大，但会改变 Task 20/21 契约并进入 Task 23 范围，应拆出额外设计与验收                       |
| C：仅注册 pipeline 并拒绝所有旧操作               | 新入口存在但不转换真实 Local fragment                                                                                                      | 不能满足本 Task 的核心交付，不采用                                                                  |

用户已选择 A，以下是本方案的设计底线（不是已实施事实）：

- 不把旧 f16 C 自动提升成 f32 后声称等价；若扩展旧 Buffer 导入接口以允许显式 f32 C，必须在 verifier 与前端入口明确一致地表达，原不支持的数值路径给出诊断。
- 将 clear 的覆盖/累加、Reduce kind 的映射、NaN/有符号零及并行重结合许可写成数学契约；无初始化证据时不得制造零值，无法证明语义兼容时不自动转换。
- 先列清直线代码、SCF if/for/while、cast/subview/逃逸的支持与拒绝条件，再设计整个生命周期的转换；不以只改写一个 Op 冒充完整 alias group 的 normalization。
- 使用事务式改写，区分 normalize pass 的成功与后续 infer/optimize 的成功；失败与重跑稳定性均需可测试。
- 保持现有 8 vars/component、4 candidates/domain，不把 Task 21 的未完成成功用例改成已完成；大图求解单独确认。
- 将根前置条件的保留、作用域和实际 IR 独立验证作为公共契约设计；不通过伪造 layout_view 副作用或重复偏移来保住证据。
- 旧推断只允许作为测试 oracle 的隔离实现；生产接口、生成代码与链接关系一起退役，并保留全部受影响测试的有效覆盖。

下面补充逐项转换规则、公共接口和验收用例。后续阶段按“生命周期 normalization → pipeline/持久契约 → 旧接口退役与全量回归”组织；不将历史提纲中的 commit 命令视为自动提交授权。

#### 22.3 导入语义 v1：显式同意数学契约，不隐式改变精度

**已确认的入口约定：** 需要迁移的 legacy Gemm/Reduce 必须在自身或最近的 `func.func/frisk.kernel/module` 上显式声明 `frisk.legacy_semantics = "tensor_v1"`。查找由内向外；没有属性的作用域继续向外查找，遇到已声明但类型/值不合法的属性立即报错，不跳过后回退外层。Copy/Fill、已经是新 Tensor 的程序不要求此标注。旧操作 verifier 不强制此标记，检查发生在 normalization 导入边界。

这是对当前缺少完整旧数值 lowering 的直接应对：标注表示导入者接受本节数学语义，不声称已经通过 GPU 实验证明所有旧行为与新行为等价。缺少标注诊断 `legacy-math-contract: explicit tensor_v1 semantics required`；不得自动补上来绕过确认。即使 dead legacy 计算最终没有结果使用，也先验证其导入语义，不靠 DCE 隐藏不支持输入。

| 操作/属性                 | v1 转换规则                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Gemm`clear_accum=true`  | 使用 C 形状和 f32 dtype 的显式正零 Tensor 作 init；不读取 C 的旧内容，产生`mma(A_eff,B_eff,zero)`                                                                |
| Gemm`clear_accum=false` | 使用已经初始化的 C 当前 Tensor 作 init，产生`mma(A_eff,B_eff,C_current)`；未初始化即失败                                                                         |
| Reduce`kind=add`        | 变成`reduce_tensor kind=sum`；接受 Task 21 的并行重结合及选中布局下的规范树，不承诺旧串行顺序 bitwise 等价                                                       |
| Reduce`kind=max/min`    | 采用 Task 21 的 NaN 传播与有符号零语义，不按旧 kind 拼写推断 maxnum/minnum 行为                                                                                    |
| Reduce`clear=true`      | 覆盖 dst，直接绑定归约结果；不额外与零或 ±∞ 合并，以免额外改变 ±0/NaN 行为                                                                                      |
| Reduce`clear=false`     | 先完整归约，再与已初始化的 dst 当前值逐元素合并：`arith.addf/maximumf/minimumf(dst_current,reduced)`；固定为“旧 dst 在左”，不把旧 dst 当源副本或任意插入归约树 |
| Reduce`mul` 或未知 kind | 拒绝，不能映射成 sum，也不顺带扩展 Task 21 支持集                                                                                                                  |

Gemm 迁移只接受相同 dtype 的 f16/bf16 A/B 和 f32 C，目标必须显式为 `sm_90a`；C、RS 的 A 必须来自下面的可提升 Local 生命周期。B 必须 Shared，A 可 Shared 或 Local；不支持 Local B。M/N/K 按实际 transpose 后形状核对，policy 与支持的线程配置保留；不改变 Task 20 的支持集。

旧 C 为 f16/bf16 的程序仍可作为 legacy IR/旧 oracle 输入存在，但 normalization 明确报告 `legacy-gemm-accumulator: f32 required; implicit promotion is forbidden`。不通过 f32 临时累加再截断伪装成旧 f16 累加。

为能构造合法的 f32 C 导入 IR，旧 `GemmOp::verify` 与 `ffi/ir.cpp::create_gemm_op` 同步扩展：保留同 dtype 的合法旧输入；额外接受上述低精度 A/B＋f32 C 组合，统一按 transpose 解释维度并核对 M/N/K。Python `create_gemm` 包装层同步可选 `legacy_semantics` 关键字，默认不标注；Reduce builder 也提供同名可选标注入口。数学形状/类型检查与目标路径可迁移性分开，不把 verifier 接受等同于 normalize 成功。

Gemm/Reduce 保留真实 memory effect：Gemm 读 A/B、写 C，clear=false 再读 C；Reduce 读 src、写 dst，clear=false 再读 dst。若暂时采用保守的无条件目的 Read，其语义也不得少报实际 Read，但测试必须能检查新增的 Reduce 目的读取；实现计划选择精确的条件 effect。其他依赖这些操作的递归 effect 检查继续有效。

#### 22.4 Local fragment 支持集与逐操作转换

以每个 Local allocation 的完整 def-use slice 为单位，不局部转换同一根的一半使用。候选根为函数/Kernel/Parallel 或受支持顺序 region 内的 `frisk.alloc_buffer`、`memref.alloc/alloca`，类型为 Local 静态 identity MemRef，offset=0，rank>0，各维>1 且为二次幂，dtype 为 f16/bf16/f32，逻辑元素不超过 65536。Local 参数、返回的 Local 引用、global、reinterpret、动态/非 identity descriptor 不作为可提升根。

同根别名首版只支持保持静态 shape/dtype/memory space/identity descriptor 的 `memref.cast`（含类型完全相同的 cast）；将其折叠到同一根状态。Local subview、降维、reshape、带布局的 Local layout_view、scalar load/store、未知调用、取地址/逃逸，以及根作为既有 SCF MemRef iter_arg/结果传递均拒绝。这里的保守 Local 支持集不缩减 Task 18 对 Shared/Global 静态 cast/subview 的已有支持。

为每个根维护 `Uninitialized` 或 `Initialized(Tensor SSA value)`，所有别名共享该状态。每个读取先检查状态；每个完整写入生成或绑定新值。未知使用即 `legacy local fragment escapes supported normalization boundary`，读前未初始化即 `legacy-fragment-uninitialized`；不得用 tensor.empty、undef 或自动补零消除错误。

| 原操作                              | 归一化行为                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Local alloc                         | 建立 Uninitialized 状态；所有使用转换并验证成功后删除 allocation                                                       |
| Local Fill                          | 用原 dtype/常数 bit pattern 的 dense splat 初始化或覆盖 Tensor；保留 -0，不强行改成 +0                                 |
| 整块 Copy，Storage→Local           | 在原 Copy 位置生成`tile_load`，源必须具有合法 layout_view；保留读取相对其他写入的顺序                                |
| 整块 Copy，Local→Storage           | 读取当前 Tensor，在原 Copy 位置生成`tile_store`，目标必须具有合法 layout_view                                        |
| 整块 Copy，Local→Local             | 先读取源的当前值，再更新目的状态；两个根可暂时引用同一不可变值，后续覆盖其中一个不改变另一个；同根自复制仍要求已初始化 |
| Storage→Storage Copy、Storage Fill | 保留原操作及 Task 19 契约，必要时仅为合法端点补 layout_view                                                            |
| legacy Gemm                         | 按 §22.3 消费 A/B/init，生成 MMA，将结果设为 C 的新当前值                                                             |
| legacy Reduce                       | 按 §22.3 读取 src，生成 Reduce 及必要的显式目的合并，将结果设为 dst 的新当前值                                        |
| Local dealloc                       | 仅接受同块、支配关系可验证的最终 dealloc，后面不存在使用；提升成功后移除，不支持条件释放或 use-after-free              |

Copy 首版仍为同 shape/dtype、空索引和空 map 的静态整块复制；不构造动态 slice。不得把任意直接 MemRef 当成硬件 fragment。新 `tile_load/store` 如果已经访问待提升 Local 根，或者 legacy 与新 Tensor 对同一 Local 根混用，则诊断不支持；不会建立两套并行状态。

待移除的 Local Fill/Copy 如果带显式执行布局、writer policy 或 vector width，首版拒绝迁移，以免删除其硬要求。待提升 allocation 的旧布局标注也不得被静默丢弃。已知 Gemm/Reduce 的 target、policy、执行线程数有明确迁移规则：`frisk.threads` 只作为旧 Gemm 的线程别名，转换到 `frisk.execution_threads`；两者或与外层 Parallel 不一致则拒绝。无法解释的 Frisk 语义属性报错，普通位置/调试元数据可保留。

Shared/Global 继续使用 MemRef：`frisk.alloc_buffer` 在合法静态空间下改为同 shape/dtype/memory space 的 `memref.alloc`，保留已有正数对齐，0 表示无额外保证而不是推断成 16/128/1024。复用已有 layout_view，不覆盖其显式 layout；对未绑定的合法根或 Task 18 已支持的视图才插入未编码 layout_view。未知别名、动态边界和 dtype 重解释仍拒绝。

#### 22.5 控制流与初始化：明确产生循环携带值

支持单块函数/Kernel 内的顺序代码及嵌套 `scf.if/for/while`；不修改函数签名，不跨过程提升，不将一般 CFG 或并发 region 当作顺序块处理。原有无关标量值、控制条件、操作顺序和 side effect 保留。

- **If：** 两分支从同一入口状态开始；对被写的外部 Local 根按稳定根序追加 Tensor 结果及 yield。若分支不写该根，则传递入口旧值。两个分支都必须能提供已初始化值；无 else 等价于旧值路径，不能给未初始化的另一分支补零。分支内新建、不逃逸的根只在分支内提升。
- **For：** 只为被修改的外部根追加 init/iter_arg/result/yield；入口必须已初始化，即使静态看起来至少迭代一次，也不靠 trip-count 特判绕过该约定。零次迭代返回入口值，循环体每次从 iter_arg 开始，尾部 yield 最新值。只读外部根捕获当前 Tensor，不额外生成携带槽。
- **While：** 保留原 before/input 与 after/result 两套可能不同的 tuple，给被修改根分别追加对应 Tensor 槽；before 的最新状态送入 condition/after/result，after 的最新状态经 yield 回到 before。条件第一次为假时，结果仍包含执行过 before 的状态，不能错误地当成普通 for 的零次执行。循环入口的携带根必须已初始化。
- **已有 Tensor 携带值：** 保留原槽顺序与类型，只追加新槽；M3/M4 的 SameLayout、Convertible、region edge 语义不变。新增槽采用 IR 根出现顺序，不用 DenseMap 遍历顺序。
- **Parallel：** 仅提升其 body 内创建且不逃逸的 Local 根；拒绝跨 Parallel 边界捕获或写回外部 Local 根，避免把潜在并发访问当作一个顺序 SSA 状态。Shared/Global 捕获按既有规则保留。
- **旧 `frisk.for`：** 涉及待提升根时，将静态合法的正 step 半开区间 `[lower,upper)` 改成 `scf.for`，把 `frisk.end` 替为 yield，并按上述规则加携带值；不进行展开。未知语义的 `frisk.block`、其他自定义 region、并行/异步控制流若捕获待提升根则拒绝，不凭名称假定可内联。

Local allocation 在循环体内创建、使用且不逃逸时，每次执行从 Uninitialized 开始，只提升该局部生命周期，不能把一次迭代的状态自动带到下一次。

转换完成只说明形成了合法 SSA。若新增携带值令图超过 8 变量，后续 infer 仍明确失败；不合并应独立的 producer/use，不展开循环绕过预算，不截断候选。

#### 22.6 可持久核验的存储前置契约

**已确认的公共操作：** `frisk.storage_contract %root {layout = #frisk.storage<...>} : memref<...>`，无结果，声明该根的完整存储映射和根指针对齐前置条件。只保存 alignment 不够，因为原 whole-root binding 还约束了子视图的地址一致性。

它是编译期可观察的前置条件声明，不是生成一个可随意丢弃的值，不标记 Pure；首版以保守、不可随意删除的契约操作保留，不能伪报对实际数据的 MemRead/MemWrite，也不改变 `layout_view` 的 Pure 属性。该声明不执行 runtime 对齐检查，不提升 allocation 的真实对齐能力；要求由调用者/原始声明满足，未来 lowering 需显式消费或证明可消除该声明。

来源只允许已有显式合法的 whole-root layout_view 或用户显式写入的 storage_contract，不能从子视图需求、MMA descriptor 对齐要求或候选偏好反推根保证。自动提取条件为同 shape、view→root 恒等、同 dtype/space；声明放在原绑定位置之前，不提升出分支/循环。Root 为当前支持的 allocation/global/函数或 Kernel 入口参数，不能把子视图伪装成新根。

作用域按 dominance 核验：只有支配某个使用点的声明可为该点提供证据；分支内声明不能给兄弟分支或分支外使用提供保证。Task 18 原先全图归集 whole-root 保证的路径也要加同样的作用域检查。对本阶段不能证明一致作用域的组合保守拒绝，不把条件保证变成无条件保证。

实际实现额外检查每层执行 region 均为已知的 func/Kernel/Parallel/scf.if/for/while 且具有 SSA dominance；Module 仅作容器。未知或 graph region 中 MLIR 的通用 dominance 不代表执行先后，不能用来借用根保证；其中的显式 storage_contract 和 whole-root 契约保存保守拒绝。此限制与无 Local 状态的同位置 Storage 端点绑定相互独立。

图中将声明作为已确定的前置事实，增加有名 unary `RootStorageContract` 约束关联被约束的 Storage 变量，payload 引用实际声明及根坐标关系。它不创建可选布局变量或额外 producer/use：此处没有新选择域；已存在的真实 LayoutVar 仍全部按原规则计数。候选准备可从显式根 map 投影有限建议，冻结后仅筛选；每个候选必须与声明的完整根地址表相容。不是仅把 metadata 存起来却不参加求解。

实际 IR 路径重新读取、验证并比较每个支配声明，使用 Task 18 的有界坐标/bit interval 证明；声明之间在可共同生效的区域必须相容。无有效消费者的声明也必须检查自身 schema、根归属、map 范围及对齐格式，不能靠死代码掩盖畸形输入。超预算返回 Unknown，不生成候选、不使用上次求解缓存。

`normalize` 保存已有显式 whole-root 声明；完整 pipeline 在 infer 之后、conversion 优化之前再次保存实际物化的 whole-root 绑定，防止后续 DCE 丢掉证据。按同根、同位置作用域和同契约确定性去重，不重复追加；绑定仍被使用时保留 layout_view，绑定被 DCE 删除后前置事实仍在。不得因此新增可搜索的域或改变 8/4 上限。

验收同时保留两个事实：不经过保存步骤的旧直接 infer→canonicalize→infer 序列不能凭空恢复已丢失证据；经过新 normalize/pipeline 保存后的序列应成功。删除 storage_contract 并删除原 whole-root binding 后，依赖该证据的实际 IR 必须失败。

#### 22.7 Pass、事务和最终核验

公开入口保持 `createNormalizeLayoutIRPass()` / `frisk-normalize-layout-ir`，以及 `buildFriskLayoutPipeline(OpPassManager &pm)` / `frisk-layout-pipeline`；另提供显式 `registerFriskLayoutPipelines()`，在 frisk-opt 启动时调用并供其他宿主显式使用，不依赖静态初始化碰巧链接。当前 FFI 没有已实现的 pass-manager 注册入口，本阶段不凭空扩展 Python pass-manager API；Python 导入测试通过本次构建的 frisk-opt 验证生成 IR。

`normalize` 先在 detached ModuleOp 上完成完整预检与改写，运行 MLIR verifier、残留 legacy 检查及导入契约检查，通过才替换原 module body/attrs。已归一化 IR 再次运行结果逐字稳定；不能转换的一个根会导致整个 normalize 失败，原模块不留部分转换结果。

具名 pipeline 使用一个事务包装 pass，在 clone 上执行：

```text
normalize → infer/materialize → 保存实际根契约
          → optimize-layout-conversions → actual-only 最终核验 → 成功才提交
```

这是对原提纲简单串联三个 pass 的明确适配：单个 pass 原子性不自动保证后续 pass 失败时撤销前面的修改。包装 pass 构造内层 PassManager，不递归调用自身；保留位置诊断，任一步失败都使外层原模块不变。

直接运行独立 normalize 成功后，用户再单独运行 infer 失败，normalize 的先前结果仍然存在；只有上述具名 pipeline 提供整条流程的原子性。最终 actual-only 在优化后检查真实布局、存储契约、MMA/Reduce/Copy/Fill 属性，不假定优化一定保留证据。

生产 infer 的入口补明确的 legacy 残留诊断：遇到旧 Gemm/Reduce 或待归一化的 fragment 用法，应要求先执行 normalize/pipeline，不能因旧操作没有 Tensor operand 就静默略过。生产中不调用任何旧 inferLayout 或 oracle。

#### 22.8 旧 API 退役与测试 oracle 隔离

- 从 ODS、C++ 和头文件删除 `ParallelOp/GemmOp/ReduceOp::inferLayout`、旧 `LayoutInterface::inferLayout(DenseMap&)`、旧 Parallel 调用计数 API 和相关生成/注释残留。旧 Gemm/Reduce 操作本身保留为导入 IR，不删除 Python 前端使用的操作名字。
- 将需要保留的旧推断公式及测试计数迁入 `test/Support/LegacyLayoutOracle.{h,cpp}`，使用测试命名空间下的自由函数；必要纯 helper 随其迁入。现有 `LegacyLayoutAdapter` 继续只做表示转换，不承担生产推断。
- 在 `test/Support/CMakeLists.txt` 用普通测试库注册，不用把库收集到 `DEEPPIPE_LIBS` 的 `add_frisk_library`，防止被 frisk_ffi 自动链接。只由 test_pass 和 unit targets 链接；frisk-opt、FriskIR、FriskTransforms、frisk_ffi 均不得依赖测试 oracle。
- 更新原 attr/reduce 测试、LegacyLayoutAdapterTest 与 CopyFill 隔离测试，不删除旧错误布局分类或把新 target 生成结果当成唯一旧 oracle。旧生产调用计数测试改为测试 oracle 计数及生产源码/链接边界检查。
- 补跨行源码扫描，避免原提纲单行 rg 漏掉分行签名；并用符号/链接检查确认旧公共方法与测试 oracle 不在生产入口中。不是禁止生产代码使用任何 DenseMap，而是禁止旧 Value→Attribute 推断接口与其执行路径。

#### 22.9 文件与接口分工（已按此分工落地）

| 文件/区域                                                                                         | 责任                                                                                                      |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `Transforms/NormalizeLayoutIR.cpp`                                                              | normalize pass、克隆事务、阶段调用及残留检查；不把所有生命周期规则塞进单一 runOnOperation                 |
| `Transforms/LegacyFragmentNormalization.{h,cpp}`                                                | 私有 Local root/alias 预检、初始化状态、逐操作重写及 SCF/旧 for 携带值转换                                |
| `IR/LegacyImportSemantics.{h,cpp}`                                                              | Gemm 共享 shape/dtype 检查、tensor_v1 查找与合法值检查；供 ODS verifier 和 FFI 使用，不依赖 target solver |
| `IR/FriskOps.td/.cpp`、`ffi/ir.cpp`、`python/frisk/common/utils.py`                         | 导入接口、条件 memory effect、旧方法退役及 Python 一致性                                                  |
| `IR/FriskLayoutOps.td/.cpp`                                                                     | `StorageContractOp` 的语法、结构验证与可持久声明语义                                                    |
| `Analysis/StorageRootContracts.{h,cpp}`                                                         | 根声明解析、作用域/dominance、去重依据、完整地址相容性；无候选搜索的公共验证 helper                       |
| `Analysis/LayoutConstraint*`、`LayoutRelations*`、`LayoutPropagation*`、`LayoutVerifier*` | RootStorageContract 的图不变量、候选投影/筛选和 actual-only 接线，不改变旧 alias/region 关系语义          |
| `Transforms/PreserveStorageContracts.{h,cpp}`                                                   | 从实际 whole-root view 提取声明并稳定去重；normalize 与 pipeline 共用，不从 unknown 候选创造保证          |
| `Transforms/PassPipelines.cpp`、公开 `PassPipelines.h`、`Passes.td/.h`、CMake、frisk-opt    | 显式注册具名 pipeline、事务包装和优化后复验；FFI 只同步导入 builder，不新增 Python pass-manager           |
| `test/Support/LegacyLayoutOracle.*` 及关联测试/CMake                                            | 旧公式只作测试参考，生产库不链接                                                                          |

文件表中的目录分别相对于 `lib/Dialect/Frisk`；公开头文件放在匹配的 `include/Dialect/Frisk` 下，Transforms 私有 helper 头留实现目录。测试新增 `LegacyNormalizationTest.cpp`、`StorageRootContractTest.cpp`、`LayoutPipelineTest.cpp`，lit 分为 normalization 正/反例、region、durable-contract 与 no-legacy；Python 新增 `test/python/legacy_normalization_test.py`，显式从本次 build 导入 FFI，不能误用环境里旧安装包。

#### 22.10 验收分组与不能宣称完成的边界

- [X] **导入数学：** C++/Python 同步接受 f16/bf16 A/B＋f32 C，四种 transpose 组合按有效形状核对；旧 f16 C 可作为旧 IR 但 normalize 拒绝，且 IR 不变；未声明/错声明 tensor_v1、错误类型/target/kind 均明确失败。
- [X] **初始化与 clear：** Fill/Copy 初始化、Gemm true 不读旧 C、false 读旧 C；Reduce true 覆盖和 false 合并分别检查 SSA；CPU 独立模型覆盖有限精确值、±0、NaN 和明确的旧 dst 合并次序，不以布局文本相等代替数值契约。
- [X] **完整生命周期：** whole-tile load/store、Local→Local 分叉后单边重写、相同根别名、最终 dealloc；未知用户、scalar 访问、Local subview、动态 cast、逃逸和混合新旧 Local 用法拒绝；任一根失败原 module 不变。
- [X] **控制流：** if 两边写/一边透传/缺初始化；for 零次与多次、只读捕获、嵌套、旧 frisk.for；while before/after 不同原 tuple、首次条件为假仍保留 before 写入；Parallel 内部根成功、跨边界 Local 捕获拒绝。分别验收 normalize-only 和预算内 pipeline，不假称所有循环规模都能求解。
- [X] **持久契约：** normalize/infer/保存/优化/打印解析/二次 pipeline 稳定；canonicalize 后根 map 与对齐仍可核验；畸形、冲突、错误根、越界、缺失、篡改、非支配分支声明均有负例；actual-only 不枚举候选、不调用旧 oracle。
- [X] **端到端：** 有真实外部写回的 legacy SS/RS MMA→store（预算内）、Fill/Copy→Reduce→store（预算内）、Copy/Fill/Parallel 和已有 Tensor 输入通过具名 pipeline；混合合法和非法函数全 module 回滚。完整 SS MMA→Reduce→store 的 normalize 可成功，但 pipeline 因 9>8 拒绝且原 module 不变。
- [X] **退役：** 生产 ODS/生成 API/源码无三种旧 inferLayout，生产二进制不链接 oracle；旧测试通过隔离 API 保留覆盖，4 个原 CTest 保持有效。
- [X] **全量 Gate：** 重建 FriskIR/Transforms/frisk-opt/frisk_ffi、全部 unit、全部 lit、原 4 CTest 和新增 Python 导入测试；单独运行 unit 二进制，不能以 check-frisk 代替。记录实际新增数量，不预填通过数。

完整验证命令（实际执行记录见 §22.13）：

```bash
cmake --build build --target FriskIR FriskTransforms FriskLayoutUnitTests \
  frisk-opt frisk_ffi check-frisk frisk_attr_test frisk_reduce_layout_test \
  frisk_layout_pass_test frisk_memory_effect_test --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_api_boundary_test.py --source-root . --build-dir build
git diff --check
```

本任务不扩展 8/4 求解预算、Task 20/21 数值/目标支持集、GPU lowering 或性能成本模型。M4 验收报告必须保留 Task 21 完整链成功尚未实现这一项；Task 22 的受限 normalization 和退役通过，不等于整个 M4 无缺口。

#### 22.11 设计确认点与实施顺序

用户已确认方案 A，并以“进入实现与测试”确认上述具体契约，包括 **显式 `tensor_v1` 导入语义声明** 和 **不可被当作无用值删除的 `storage_contract` 前置条件操作**。其他转换、SCF、事务和测试规则按上文执行。

确认后按三个可独立复审阶段实施：22A 导入语义与完整生命周期 normalization；22B 根契约、具名原子 pipeline 与 actual-only；22C 旧 API/oracle 隔离、前端/回归与文档。每阶段先写对应失败测试再实现，最后进行全量验收；不自动 commit/push，也不改动独立 Task19–21 对比汇报。

#### 22.12 实施检查表（已完成）

各项采用先失败测试、后实现、针对性复审；工作直接在源目录，不自动提交。

1. **22A.1 导入基础。** 在 `LegacyImportSemantics` 集中 shape/dtype 与标注检查；修改 Gemm/Reduce 条件 effects、FFI 和 Python builder。先测试四种 transpose、混合 accumulator dtype、marker 查找及 clear effects，再构建 IR、FFI 和针对性单元测试。
2. **22A.2 生命周期转换。** 新增 `LegacyFragmentNormalization` 与 `NormalizeLayoutIR`，先覆盖直线 Fill/Copy/Gemm/Reduce、失败回滚，再增量覆盖 if/for/while、Parallel 和旧 for。公开 `createNormalizeLayoutIRPass()`；完整生命周期与未初始化错误不得用局部重写替代。
3. **22B.1 持久契约。** 新增 `StorageContractOp`、`StorageRootContracts`、`PreserveStorageContracts`；先验证 schema/作用域/地址相容性，再接 unary `RootStorageContract` 的图不变量、候选准备和 actual-only。已有 whole-root 对齐证据也按 dominance 生效。
4. **22B.2 原子 pipeline。** 新增 `PassPipelines` 与显式注册；测试 clone 上 normalize/infer/preserve/optimize/actual-only 的提交与整体失败回滚。预算内 MMA→store、Reduce→store 成功；9 变量全链仍失败且保留原模块。
5. **22C.1 隔离旧 oracle。** 搬迁旧公式到普通测试库 `test/Support/LegacyLayoutOracle`，删除三类生产 inferLayout API、旧接口和计数器；先增加源码/链接隔离断言，再更新旧测试调用，不削弱原覆盖。
6. **22C.2 最终验收。** 独立审查全部改动；执行 §22.10 的完整 build、unit、lit、CTest、Python 和 diff 检查，同步设计与比较文档的实际完成状态。

#### 22.13 实际实现与验收记录（已完成）

实现直接位于源目录 main，基线 `14b39e9`，未创建工作树、未自动提交或推送。具体实现采用 §22.3–22.12 的已确认契约，而非下面保留的原始简化提纲：

- `LegacyImportSemantics` 统一 C++/FFI 的有效 M/N/K 和 dtype 校验；Python 显式可选 `legacy_semantics`，Gemm/Reduce 按 clear 精确声明目的 Read。旧数学输入仍可解析，导入拒绝与旧 verifier 的合法性分开。
- `LegacyFragmentNormalization` 对完整 Local 根维护 Tensor 状态，处理 Fill/整块 Copy、Gemm/Reduce、if/for/while、旧静态 for、Parallel 内部根；未初始化、未知别名、逃逸和跨函数/Kernel/Parallel 捕获明确拒绝。normalize 在 clone 上全部验证成功才提交。
- `StorageRootContracts`、`PreserveStorageContracts` 和 unary `RootStorageContract` 保存并检查完整根地址/对齐事实，不增加候选变量。支持已有分析可证明的静态 strided 根；whole-root view 依据 shape/dtype/space 和坐标恒等判断，不要求 MemRef descriptor 类型完全相同。
- `buildFriskLayoutPipeline`/`registerFriskLayoutPipelines` 注册事务入口。保存根契约发生在 infer 后、优化前，最终重新从实际 IR 复验；9 变量完整链只通过 normalize，pipeline 超限时整个 module 回滚。
- 旧 Parallel/Gemm/Reduce `inferLayout`、ODS interface 与生产计数器退役。历史公式和纯 helper 搬入普通测试静态库 `test/Support/LegacyLayoutOracle*`，不进入生产 FFI 库列表；原有分类/非法布局测试继续有效。

**实现中必要的适配：** Python 真正 dump→parse 的红灯暴露了 Copy 空索引 builder 仍生成 rank identity map，以及 alloc_buffer 解析缺省 Local memorySpace/打印遗漏附加属性的问题；已分别修复并增加回归。仅对 Storage 的 Copy/Fill/alloc，即使位于未知 region 或一般 CFG，也可在原位置绑定合法端点；这不提升 Local 状态、不扩展 solver 的 region 支持。Local 跨嵌套 Kernel 捕获已专门拒绝。

**最终新鲜验证：** 所列全目标构建成功；独立单元可执行文件 **242/242**（276.991 秒）、**48/48 lit**（9.49 秒）、原 **4/4 CTest**（0.03 秒）、**5/5 Python 测试方法**（0.201 秒）通过；源码、生成头文件、frisk-opt/FFI 二进制的旧接口/oracle 隔离扫描及 `git diff --check` 通过。相较 Task 21 的 197 unit / 36 lit，新增 **45 项单元、12 份 lit**；其中一个数值单元包含 30 组场景。独立审查及修复后复审通过，无未解决发现。构建仍出现已有 Ninja `premature end of file; recovering` 警告，本任务未修改构建元数据。

数值回归使用独立 CPU 模型核对 Reduce 的 30 组有限精确值、±0、NaN、三种 kind 和 clear 组合；不等于运行过 GPU lowering。未刷新上游快照或运行 TileLang/Triton 实验，不声称性能优势或已证明研究独创性。8 vars/component、4 candidates/domain 和 Task 21 完整链成功验收缺口保持不变。

最终独立审查还发现并补测两处边界：未知/non-SSA region 的通用 dominance 可能将后出现的声明当作支配证据，现统一使用上述保守 scope 检查；残留旧 LayoutAttr 扫描原先仅检查操作属性/结果，现递归检查所有 region 的块参数类型（含未使用参数），normalize 与直接 infer 共用。两者均已观察修复前失败用例，并通过修复后全量复验与独立复审。

以下为原始迁移提纲，保留用于追踪计划演进，旧复选框和简化 pipeline 示例不代表当前实现；以 §22.3–22.13 为准，提交命令不自动执行：

**Files:**

- Create: `lib/Dialect/Frisk/Transforms/NormalizeLayoutIR.cpp`
- Create: `include/Dialect/Frisk/Transforms/PassPipelines.h`
- Create: `lib/Dialect/Frisk/Transforms/PassPipelines.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskOps.cpp`
- Modify: `lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp`
- Modify: `include/Dialect/Frisk/IR/FriskInterfaces.td`
- Create: `test/Transforms/normalize-legacy-layout-ir.mlir`
- Create: `test/Transforms/no-legacy-layout-map.mlir`

**Interfaces:**

- Produces: `std::unique_ptr<Pass> createNormalizeLayoutIRPass()`，参数 `frisk-normalize-layout-ir`。
- Produces: `void buildFriskLayoutPipeline(OpPassManager &pm)`，注册参数 `frisk-layout-pipeline`。
- Pipeline contract:

```text
legacy Buffer IR
  -> frisk-normalize-layout-ir
  -> Tensor Mma/Reduce + layout_view/tile load/store
  -> frisk-infer-layouts
```

- [ ] **Step 1: 写 normalization 红灯测试**

输入使用现有 local MemRef Gemm C、Reduce src/dst；输出必须含 `frisk.mma`、`frisk.reduce_tensor`、Tensor SSA 和必要 bridge，不得含 local fragment 作为新 solver LayoutVar。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，normalization pass 尚不存在。

- [ ] **Step 2: 实现 local MemRef 到 Tensor SSA normalization**

按 def-use slice 重写完整 fragment 生命周期。仅支持可证明无 alias escape 的 local alloc；地址被未知 Op 使用时诊断：

```text
legacy local fragment escapes supported normalization boundary
```

不得部分转换同一 alias group。

- [ ] **Step 3: 注册默认布局 pipeline**

在 `PassPipelines.h/.cpp` 提供：

```cpp
void buildFriskLayoutPipeline(OpPassManager &pm) {
  pm.addPass(createNormalizeLayoutIRPass());
  pm.addPass(createFriskInferLayoutsPass());
  pm.addPass(createOptimizeLayoutConversionsPass());
}
```

在 pass 注册实现中增加：

```cpp
PassPipelineRegistration<> layoutPipeline(
    "frisk-layout-pipeline",
    "Normalize, infer, materialize, and optimize Frisk layouts",
    [](OpPassManager &pm) { buildFriskLayoutPipeline(pm); });
```

`frisk-opt` 启动时调用 `registerFriskPasses()` 后必须同时注册上述 pipeline；`PassPipelines.cpp` 加入 `FriskTransforms` source list，不能依赖静态初始化碰巧被链接保留。

- [ ] **Step 4: 删除旧生产接口**

从 ODS/C++ 删除 `ParallelOp::inferLayout`、`GemmOp::inferLayout`、`ReduceOp::inferLayout` 及旧 `LayoutInterface::inferLayout(DenseMap&)`。Legacy adapter 和 baseline oracle 保留在 Analysis/test，不得被生产 pass 调用。

- [ ] **Step 5: 验证仓库无生产 DenseMap 路径**

Run:

```bash
rg -n "inferLayout.*DenseMap|DenseMap<Value, Attribute>.*layout" \
  include/Dialect lib/Dialect
```

Expected: 无匹配；允许匹配仅存在于 `LegacyLayoutAdapter` 和 test 文件时，应使用更窄路径再次确认生产目录为 0。

- [ ] **Step 6: 运行 M4 全量回归**

Run:

```bash
cmake --build build --target FriskIR FriskTransforms FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: 所有 tests PASS；legacy input 经 normalization 后通过 `-verify-each`。

- [ ] **Step 7: 同步设计文档并提交**

确认设计 Section 5、7、12 不再把旧 interface 写成生产路径。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test \
  guoqiao/layout_inference_design.md
git commit -m "refactor: normalize legacy fragments to tensor layout IR"
```

### M4 Gate

Run:

```bash
build/bin/frisk-opt test/Transforms/no-legacy-layout-map.mlir \
  -frisk-layout-pipeline -verify-each
rg -n "inferLayout.*DenseMap|DenseMap<Value, Attribute>.*layout" \
  include/Dialect lib/Dialect
```

Expected: pipeline 成功；生产目录旧接口匹配数为 0；Copy/Fill/Gemm/Reduce/Parallel/SCF 都由新 graph 处理。

---

## M5：有限候选、SM90 契约与 conversion 优化

### Task 23: 实现连通分量候选求解和 CostVector

**状态（2026-09-27）：用户确认的方案 A 及 §23.1–23.8 已完成实现、独立复审和全量验收，结果见 §23.9。源码目录 main HEAD 为 `9bc8349`，工作区包含已完成但未提交的 Task 22；保留既有修改，本轮未提交或推送。**

#### 23.1 审计：不能直接照搬原始接口提纲

| 当前源码事实                                                                                                           | 本任务需要适配的地方                                                           |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `LayoutVerifier.cpp::solveBootstrapLayoutGraph` 对 hard component 做 DFS，拒绝超过 8 个变量或任一域超过 4 个候选     | 替换为正式的有界求解，不只是把 8 改成更大的常数；删除 bootstrap 入口与限制结构 |
| 现有求解只比较新增 conversion 数，然后用 stableOrdinal/文本决定同分顺序，未调用`LayoutTarget::evaluate`              | 真正接入 CostVector；不再把零转换当作独立于其他成本的最高优先级                |
| SM90`evaluate` 只累加变量 ID 到 tie-break，相同变量的不同 assignment 得分相同                                        | 改为明确的静态基础成本，并采用与候选插入顺序无关的完整 assignment key          |
| `LayoutConversionEdge` 已有 bytes/synchronizationCost，但当前求解没有填充                                            | 本阶段必须计算保守、非零的转换估计，不能声称已经存在可直接使用的边成本         |
| `CandidateAssignment` 已存 Attribute 值；`LayoutSolution` 还包含 MMA instructionBindings、Reduce reductionBindings | 保留真实编码和两类 binding；原提纲不能覆盖这些已落地接口或只返回候选下标       |
| MMA/Reduce 准备阶段另外限制关系的证明组合规模                                                                          | 与搜索预算分开命名、诊断；更大的 solver 不代表扩大指令/归约证明支持集          |
| Task 24 计划做 placement/rematerialization，Task 25 做 shape-aware SM90 cost                                           | 本任务完成搜索和可扩展代价接线，不提前声称完成这两个任务                       |

#### 23.2 方案比较与推荐

- **方案 A（推荐）：小组件精确搜索＋大组件确定性 beam＋明确静态基础成本。** 删除 solver 的 8/4 硬上限，保留独立证明预算；接通完整 CostVector 顺序及已有 conversion，验收完整 9 变量链。与原 Task 23 范围一致。
- **方案 B：全部使用有总节点预算的精确搜索。** 预算内可证明最优，但遇到较大候选乘积容易直接耗尽预算，不能提供原计划要求的 beam 路径。
- **方案 C：本任务同时引入完整 SM90 性能模型与重计算。** 成本更丰富，但跨入 Task 24/25，验证范围显著扩大；不能仅凭名称相同把它算成 Task 23 的正常实现步骤。

以下方案 A 具体契约已经用户确认，是本轮实现与验收依据。

#### 23.3 候选、硬约束与稳定身份

1. 沿用已存在的候选准备、strict/common 单调删减与支持表；搜索期间不再生成候选。不合并真实 producer/use 来规避预算，也不删除 RootStorageContract。
2. 求解入口先检查图不变量、target/type 合法性及硬要求，再按 hard constraint 建组件；Convertible 属于需要实际满足的硬关系，不能为了拆小组件将其忽略。孤立变量也必须求解。现有 soft preference 不新增跨组件耦合；首版成本按组件可分解，不能对任意跨组件非可加代价宣称全局最优。
3. 候选使用完整结构身份去重并稳定排序。允许 canonical hash 作加速索引，但必须验证实际编码相等；保留维度命名、topology、replication 等类型契约，不把数学映射相同但 SSA 类型不等的编码折叠。现有 `layoutCandidateKey` 实际是文本序列化，不能称其已提供数学语义归一化。
4. 搜索使用稳定候选表；如暴露 candidateIndex，它引用求解开始时的该表，不能引用随后删减的可变下标。solution 仍以 Attribute assignment 为实际物化依据，保留 provenance 与 instruction/reduction bindings。
5. 候选排序不依赖 DenseMap 迭代、地址或插入次序，也不使用 stableOrdinal 暗中编码性能偏好。最终同分按稳定变量名及完整编码 key 排序；64-bit tie-break 不能替代无碰撞的完整 key。
6. 记录候选拒绝：变量、稳定候选身份、对应约束/target 检查及原因。区分全局域删减、某个部分 assignment 不相容、beam 丢弃：后两者不能伪造为候选本身非法。已有传播统计继续保留。

#### 23.4 搜索预算与失败含义

公开 `SolverOptions`，默认值为：

```cpp
uint64_t exactCombinationLimit = 256;
unsigned beamWidth = 32;
uint64_t maxExpandedStates = 65536; // 每个组件尝试的候选扩展总数
```

原提纲 `maxCandidatesPerComponent=256` 易被误读为变量数或单域容量；这里明确改名为精确搜索的组合数阈值。以上参数必须为正；普通 infer pass 和具名 pipeline 使用相同默认值，提供可测试的配置入口。

- 计算去重/硬剪枝前后候选乘积，采用溢出安全算术；溢出记录为超过可表示范围，不能回绕成小图。
- 剪枝后乘积 ≤256 时精确搜索；超过时启用 beam32，而非截取前 256 个组合。超过阈值输出组件、变量数、域大小、原始/剪枝后乘积、剪枝量和 beamWidth。
- 搜索变量顺序为当前剩余合法候选数最少、hard degree 最大、stable ID 最小；每次扩展检查已就绪关系，并检查 MMA tuple/Reduce pair 是否仍存在合法延伸。
- 部分状态按可证明的成本下界与稳定 assignment key 排序；没有可信下界的项使用非负成本的零下界，不能拿高估值剪掉可能更优的解。完整 assignment 必须重新计算完整成本并满足全部硬关系。
- 每尝试一个候选扩展就计数，失败的扩展及 MRV 判断剩余合法域时的试探也计入 maxExpandedStates，避免“只统计成功节点”的假预算；搜索不依赖与组件深度相等的无界 C++ 递归栈。
- 精确且未截断的搜索才可报告“当前固定候选域内最优”或“已穷尽无解”。beam 截断后找到合法解只报告可行近似解，不承诺最优。
- beam 丢弃或节点预算耗尽后没有完整解，诊断 `search-incomplete`/`search-budget-exceeded`，不能称为已证明无解。节点预算中断不提交未完成搜索的部分 assignment；IR 保持不变。
- 所有解仍须经过 `verifySolvedLayoutGraph` 和实际 IR 复验。硬证明为 Unknown 的候选不能当成合法；诊断保留“不被证明”的原因，不借高成本代替硬错误。

取消的是 solver 的 maxVariables=8/maxDomainSize=4。MMA/Reduce 等本地证明组合预算本阶段不扩张；它们超限时明确报告 proof-preparation budget，而非继续冒用 bootstrap solver 的错误名称。

#### 23.5 CostVector v1：完整比较接口，不伪造完整硬件模型

九字段按声明顺序作字典序比较，合法性在成本之前完成。加法、乘法采用明确的饱和上限，并记录 saturation；不得溢出回绕导致高成本方案胜出。tie-break 只能在前八项相同后作用，SM90 首版该数值字段可保持中性，最终由完整稳定 key 解决同分。`exhaustive` 仅说明该组件搜索未截断；跨组件的可加最优性要求所有参与比较的总成本不发生饱和，不能据此声称饱和总和仍全局最优（饱和可能压平高优先级差异）。

Task 23 的交付分为两层：

| 层次              | 本阶段交付                                                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 通用引擎          | 真正调用 target cost，支持九字段排序、可分解成本累加、合法下界、同分稳定性；用受控 target 测试所有字段优先级                            |
| SM90 静态基础估计 | 非零 conversion 搬运/同步上界、可从编码得到的寄存器槽数与 replication、实际 Shared 根占用跨度、显式转换数量等结构项；不是测得的执行周期 |

计费规则必须能审计：每条实际 conversion 边计一次；Shared 同根别名按所选合法 footprint 的最大根覆盖跨度计费，不重复当成多份 allocation；寄存器槽/replication 的基础估计按定义位置或操作 execution binding 计，不把同一个值的每个消费节点都当成额外常驻副本。`spillRiskAndRegisters` 使用编码给出的每线程寄存器坐标槽数之和，`replication` 使用各计费定义的 `replication-1` 之和，`codeSize` 首版仅计既有及新增的非恒等转换操作数。consumer conversion 的临时搬运开销由边成本承担。无活跃区间分析时，“槽数之和”只能称静态代理量，不能称精确 spill 或 occupancy。

`instructionPathAndWork`、memoryTransactions、bankConflictDegree 的完整硬件估计留 Task 25；本阶段 SM90 未建模项统一中性并在 cost coverage 输出中明确标注，不把它们解释为已证明零开销。测试 target 可提供非零值验证“更好的主计算路径可以胜过零转换”。当前 soft coalescing 偏好不能冒充已经落地的事务计数模型。

保留/适配现有 `LayoutTarget::evaluate`，使其获得评估所需的图/变量上下文；不得仍然接收信息不足的接口、再通过全局隐藏状态补上下文。新增 partial lower-bound 查询时默认给出安全零下界；完整评估失败应明确诊断，不返回零成本。组件独立求解依赖非负可分解成本契约，测试覆盖该前提。

转换首版只支持既有 KeepCommonLayout/Convert：

- 相同编码无新增转换；不同编码先由现有关系证明可转换，再评估；不能用成本让不可转换关系通过。
- 首版对非恒等转换统一采用单 CTA shared staging 的保守估计，不假定已经识别最优 shuffle。设 `Psrc/Pdst` 为编码的完整物理槽数、`b=ceil(elementBits/8)`、`T` 为 CTA 线程数，记录 `bytes=b*(Psrc+Pdst)`，覆盖按槽写入/读取的搬运上界，`synchronizationCost=2` 表示两次 CTA 同步的估计上界；恒等转换二者为 0。估计来源标记 `cta-staging-upper-bound-v1`，不能因暂未生成 lowering 而记为零，也不由此承诺实际 scratch 已经分配。
- `conversionBytesAndSync=bytes+4*T*synchronizationCost`，将一次 CTA 同步折算为每线程一个 32-bit 单位的固定静态惩罚。已有合法关系支持同一 CTA 内不同活跃线程子集之间的转换（如 32→128）；本轮接线明确 `T=max(Tsrc,Tdst)`，保持既有执行边界语义，不新增“两端线程数必须相等”的限制。这是公开的比较尺度，不是硬件周期或实测延迟；不可按测试用例调权重，更精细路径/频次校准留后续任务。
- 已有显式 convert 只计一次成本、不重复插入；新边将 bytes/synchronizationCost 写入 solution。loop 频次、critical path、hoist、rematerialization 不在本任务内。

#### 23.6 接线、观测与事务

新增 `LayoutCandidateSolver.cpp`、`LayoutCostModel.cpp`；正式求解 API 放 `LayoutSolver.h`，保留现有 solution 的实际 assignment、conversion、MMA/Reduce binding，并附加成本/搜索统计。从 `LayoutVerifier.cpp` 移走搜索实现，验证器继续独立检查真实解；共享硬关系查询可提取为小 helper，不复制两套不同的合法性规则。

生产 infer、pipeline 和受影响测试全部切换正式 solver。删除 `solveBootstrapLayoutGraph`、`BootstrapSolverLimits` 和生产旧调用，不通过同名包装保留两套求解入口。

dump-analysis 增加每组件搜索模式、域大小、扩展数、硬拒绝数、beam 丢弃数、是否穷尽、所选 CostVector、cost coverage 和转换估计。统计排序稳定，可供 FileCheck 核验；“未找到解”与“证明无解”使用不同消息。

求解只产生方案，不改动 IR；沿用现有物化与 Task 22 模块级事务。搜索失败、非法 solution、最终实际 IR 复验失败均不得留下部分更新。actual-only 仍不枚举候选、不调用求解器，不把成本当合法性依据。

#### 23.7 验收契约

- [X] 小图联合选择胜过逐变量贪心；含 SameLayout/Convertible、多角色 InstructionContract、ReductionLayout、alias/root contract/region 的回归保持语义正确。
- [X] 相同输入打乱变量、候选、约束插入次序仍得到相同规范解、binding 和转换位置；去重不破坏支持表或候选拒绝 provenance。
- [X] 256 及边界上下的 exact/beam 模式、beam 截断丢失唯一解、节点预算耗尽、候选乘积和成本溢出分别测试，不能混淆 unsat 与 incomplete。
- [X] 固定随机种子的 ≤4096 组合小图用独立 exhaustive oracle 比较；提高 exact 阈值或使用足够宽的 beam 验证最优一致，默认窄 beam 仅验证合法性与确定性。
- [X] CostVector 九字段逐项优先级、主计算优于零转换、同主成本下较轻转换优先、已有 conversion 不重插、相同根不重复计费、未知完整成本不默认为零。
- [X] 真实 solver 接受超过 8 个变量的可解组件，以及非 MMA/Reduce 证明端点上的合法 >4 候选域；独立证明预算仍有明确拒绝测试。
- [X] 将现有 9 变量 SS MMA→Reduce→Global store 改为完整成功验收，包含实际外部写回、MMA/Reduce binding、actual-only 与重跑稳定；另设显式小搜索预算测试替代原 8 变量回滚负例，不直接删掉回滚覆盖。
- [X] 追加报告中含 Global→Shared Copy、Parallel128、MMA、Reduce、Global 写回的整链集成例；只在实际通过后更新其“尚受 8 变量限制”的当前状态，保留历史记录。
- [X] 重建全部相关目标，独立跑完整 unit、lit、原 4 CTest、Python 导入与旧接口隔离测试，执行 diff 检查和独立复审；记录本轮实际数量，不预填通过数。

#### 23.8 范围与确认点

用户已确认方案 A：用组合/搜索预算取代 solver 的 8/4 限制，采用默认 exact256＋beam32＋每组件65536候选扩展预算，完成完整链验收；本阶段提供可审计的静态基础成本，不将 Task 24/25 的重计算、布局转换调度或完整 SM90 性能模型提前计作完成。

按“成本/稳定身份与统计 → exact/beam solver 与完整 bindings → 生产替换与整链验收 → 全量回归及文档”实施。原提纲中的提交命令不代表本轮获得提交授权。

#### 23.9 实际实现与验收记录（2026-09-27）

实现已接入生产，独立复审及最终全量 Gate 已通过；本轮未提交或推送。

| 部件                                                       | 实际变化                                                                                                                       |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `LayoutSolver.h` / `LayoutCandidateSolver.cpp`         | 正式`solveLayoutGraph`、组件 exact/beam、MRV、独立预算、结构去重、拒绝记录、稳定全赋值 key；移除旧 bootstrap 入口及 8/4 限制 |
| `LayoutRelations.cpp` / `LayoutVerifier.cpp`           | 搜索与验证复用硬关系查询；验证器不负责搜索，仍独立核验真实解和 MMA/Reduce binding                                              |
| `LayoutCostModel.cpp` / `LayoutTarget.h` / SM90 target | 九项字典序、饱和成本、图上下文评估、安全部分下界、conversion 上界和静态定义/根占用计费                                         |
| infer / named pipeline                                     | 三项同默认值的搜索选项、组件及总成本统计、conversion bytes/sync/source；沿用事务物化及 actual-only 验证                        |
| 本地证明准备                                               | MMA/Reduce 域预算保持原值；错误名称改为 instruction/reduction proof budget，与搜索规模限制分开                                 |

搜索前保留原 strict/common 传播，solver 自身检查 target/type、存储容量、硬 seed/unary 与 tuple/pair 单候选支持，再冻结域；搜索中的 relation cache 只记录该冻结域内的规范候选下标。MRV 试探也计入 expanded，不将某个分支拒绝当作全局候选删除。原始域乘积溢出与剪枝后乘积溢出分别记录，原始大图被硬约束缩为单解时仍可进入 exact。

实施中经回归确认并适配的边界：

- 保留原有同 CTA 不同线程子集之间的合法转换，成本同步项取 `max(Tsrc,Tdst)`；非单 CTA 仍拒绝。
- 公共入口规范化未 finalized 的有效图；已 finalized 的生产图不重复清空传播统计。空图仍校验预算/异常关系；无类型非法图不先解引用类型。
- `exhaustive` 为组件是否穷尽；不能把有限 beam 或发生饱和的跨组件总成本声称为全局最优。
- Shared 部分别名以根原点到最大占用末端计费，例如 `[4,12)`、`[20,28)` 只计 28 bytes，不计 16、24 或完整 allocation 的 32 bytes。

测试覆盖包括：256 边界及相邻模式、65 变量乘积溢出与 hard seed 缩至单解、宽/窄 beam 区分、三条联合关系使 beam 丢失唯一解、24 个固定随机样例各 64 种赋值的独立穷举对照、规范同分与真实 >4 候选、主计算优先于零转换、较轻转换优选及既有转换只计一次。成本模块独立复审和求解器复审已完成，已修正公共入口稳定性、参数检查及 conversion dump 问题；无遗留 Important 项。

最终 Gate：272/272 unit（35 suites，382.771 秒）、49/49 lit、4/4 CTest、5/5 Python 测试方法；legacy production boundary 扫描、旧 bootstrap API 源码扫描及 `git diff --check` 全通过。新增 28 项成本/搜索单元用例和 2 项整链/预算单元用例；原有 9 变量失败用例改为成功验收，不删除回滚覆盖。重点 cost/solver/verifier 32/32、pipeline/Reduce 25/25 也独立运行通过。数值及性能边界未扩展为 GPU 执行保证。

实际命令（在源目录运行）：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt frisk_ffi frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test frisk_memory_effect_test --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -sv build/test
ctest --test-dir build --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
git diff --check
```

构建期间仍出现既有 Ninja 日志恢复和 CMake CMP0116 弃用告警；构建及测试退出码均为 0。本任务没有改动构建环境政策、运行 TileLang/Triton 测试或进行 GPU 性能实验。

以下保留原始提纲用于追踪；具体语义以确认后的 §23.1–23.8 为准：

**Files:**

- Create: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Create: `lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Modify: `include/Dialect/Frisk/Analysis/LayoutTarget.h`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutCandidateSolverTest.cpp`
- Create: `test/Transforms/layout-candidate-selection.mlir`

**Interfaces:**

- Consumes the `CostVector` definition from Task 5; produces assignment and solver APIs:

```cpp
bool operator<(const CostVector &lhs, const CostVector &rhs);

struct CandidateAssignment {
  DenseMap<LayoutVarID, unsigned> candidateIndex;
  SmallVector<LayoutConversionEdge> conversions;
  CostVector cost;
};

struct SolverOptions {
  unsigned maxCandidatesPerComponent = 256;
  unsigned beamWidth = 32;
};

FailureOr<LayoutSolution> solveLayoutGraph(LayoutConstraintGraph &graph,
                                           LayoutTarget &target,
                                           SolverOptions options);
```

- [ ] **Step 1: 写全局选择红灯测试**

构造三个变量的 component，使逐点最小选择违反中间 SameLayout，而全局 assignment 有唯一合法最小值；另测相同 cost 依赖 stable tie-break，不依赖 candidate 插入顺序。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，candidate solver 不存在。

- [ ] **Step 2: 构建连通分量和 hard pruning**

只使用 hard constraint 建 component。候选先 canonical hash 去重，再逐个验证；pruning 必须记录：

```cpp
struct CandidateRejection {
  LayoutVarID var;
  unsigned candidateIndex;
  LayoutConstraintID constraint;
  std::string reason;
};
```

正式 solver 必须先覆盖 bootstrap 已支持的 `KeepCommonLayout/Convert` edge resolution：从 edge 中已有的 bytes/synchronization 上界构造保守 conversion cost，并保持 M3 的 conversion materialization 接口。完成生产接线后，删除 `solveBootstrapLayoutGraph`、`BootstrapSolverLimits` 及生产 pass 对它们的调用；用相同的 M2/M3 case 验证正式 solver 保持语义结果，并允许因完整 CostVector 选择更优的合法 assignment。Rematerialization 和更精确的 critical-path conversion cost 留到 Task 24。

- [ ] **Step 3: 实现有上限 beam search**

变量选择顺序固定为：candidate 数最少、hard degree 最大、stable ID 最小。每扩展一个变量立即做 partial hard check 和 lower-bound cost；beam 使用 `CostVector + stable assignment key` 排序。

超过 256 个原始组合时不静默截断：打印 component ID、vars、每个 domain size、pruning 数和 beamWidth。

- [ ] **Step 4: 实现 CostVector 比较**

非法候选在进入 cost 前已经删除。比较顺序按照字段声明；`deterministicTieBreak` 只能解决其他字段完全相等，不能编码性能偏好。

- [ ] **Step 5: 验证最优性和确定性**

对总组合数不超过 4096 的随机小图，用 exhaustive solver 作为 oracle，比较 beam/exact assignment；首版测试图设置 beamWidth 足够覆盖 exact solution。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutCandidateSolverTest.*'
```

Expected: PASS。

- [ ] **Step 6: 提交候选 solver**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis test unittests
git commit -m "feat: select global layout candidate assignments"
```

### Task 24: 将 conversion placement 和 rematerialization 纳入普通候选

#### 24.1 已确认设计：方案 A（2026-09-27）

用户确认先写入契约，再实施与测试。直接在源目录工作，不创建 worktree；本轮不自动提交或推送。以下细化取代本节旧步骤中未明确的策略，Task 25/26 不提前计作完成。

**目标与架构：** 在 Task 23 的有限布局赋值搜索内，对每个完整赋值求解可分离的边界方案，再联合比较 CostVector。保持共同布局、consumer conversion、同块共享 conversion、便宜 producer rematerialization 都是普通方案，不以零转换不可行为前提。物化前记录可验证的计划，物化失败不改变原 IR。

**全局约束：**

- 保留 exact256 / beam32 / 65536 布局扩展预算；边界方案采用确定性、有限的同源/同目标/同块分组。组间不共享临时值或重计算切片，成本可相加；不宣称跨组或跨 region 全局最优。
- 同组比较各 consumer 独立转换、一次共享转换、各 consumer 的受限重计算组合。共享转换插在最早 consumer 前，源定义必须支配该位置。首版不跨 block、if、loop 移动或共享。
- 显式已有 conversion 保留其语义与位置，计费但不重复生成；相同 encoding 不生成转换。Hard seed、SameLayout、AliasLayout、MMA/Reduce/Storage 契约均不可用重计算或转换覆盖。
- 重计算只接受无 region、单 Tensor 结果、静态同形的受支持便宜操作。首版白名单为 dense arith.constant、arith 的 add/sub/mul、整数按位操作和 select；还必须由 MemoryEffectOpInterface 证明无副作用。不包含 load/store、MMA、Reduce、随机、时钟、除法、未知操作或 tensor.empty。
- backward slice 限制为最多 8 个不同操作（共享子表达式只算一次）；第 9 个操作出现时整条重计算候选不枚举。只复制 consumer 所在块中的生产者；块参数或不可复制的定义作为叶子，不穿越 region。叶子必须支配插入点；叶子布局不符时插入的转换必须一并验证与计费。
- 克隆保留操作属性、标量依赖及运算顺序，不做浮点重结合；每个 Tensor 克隆结果均重新检查目标布局、硬 seed 与资源约束。原有 producer 和其他 users 不修改；本任务不依赖 DCE 获得成本收益。

#### 24.2 成本契约

- 新增生产侧的有限、可解释交换分类，不依赖 `frisk::test` 的测试 lowering ABI：同线程 register、同 warp shuffle、跨 warp shared；证明/枚举预算不足或布局不支持分类时回退到 Task 23 的 CTA staging 上界，并记录 provenance。
- 使用元素 bit width（按字节向上取整）、tile volume/物理 carrier 数、是否 staging、barrier 数和插入位置的 loop depth。循环权重采用 `2^depth` 的静态代理，饱和运算；它不是 trip count 或 cycle 预测，动态循环也不假装已知执行次数。
- `conversionBytesAndSync` 在 Task 24 明确为“边界修复成本”：转换搬运/同步或重计算工作与其叶子转换的同量纲静态代理。便宜克隆每个输出物理元素计一次元素字节工作；不把克隆工作单独加到更高优先级的 `instructionPathAndWork`，否则转换的该项为 0 会机械地禁止任何重计算。主计算路径的 target 评分优先级不变。
- 克隆寄存器/replication/codeSize 增量也计费；共享转换只计一次，独立方案逐次计费。仅当重计算总边界成本严格低于该 use 的直接转换时枚举；同分优先不克隆，再按稳定图 key。
- 部分布局赋值的边界成本下界取 0，避免将随后可共享/替代的转换重复计入下界。完整赋值执行组内有限比较；不因 beam 的不完整搜索声称不可行性已证明。

#### 24.3 方案表示、验证与物化

- `LayoutConversionEdge` 扩展 placement、共享组 leader、交换模式、loop depth、成本来源和有序 rematerialization slice；每个不一致的实际 use 仍保留一条有授权 constraint ID 的记录。
- 新建 `LayoutConversionChoice.h/.cpp`，负责 slice 构造、组内方案比较及方案核验；`LayoutCostModel.cpp` 只负责数值估计；solver 不直接改 IR。
- verifier 从原 SSA、完整 assignments 和硬约束重新检查计划，拒绝伪造/遗漏/重复 use、错误插入点、越块共享、过大切片和硬 seed 改写；不能仅相信成本字段。
- `MaterializeLayouts.cpp` 在 detached module 内按计划生成共享 conversion 或克隆切片，重用同组转换 SSA；之后执行 MLIR 与实际布局验证，成功才替换原 module body。循环 carried slot 和函数签名保持既有约束。

#### 24.4 实施与验收清单

使用 writing-plans、TDD 与实现后独立代码复审。代码任务按以下顺序推进，未验证前不打完成标记。

- [X] **24-A 成本分类。** 修改 `LayoutCostModel.h/.cpp` 和 `LayoutConstraint.h`，在 `LayoutCostModelTest.cpp` 先加入 register/shuffle/shared、fallback、位宽、loop depth、饱和的红灯测试；保持未知路径非零。运行 `cmake --build build --target FriskLayoutUnitTests --parallel 12` 与 `build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='*LayoutCost*'`。
- [X] **24-B 候选选择。** 新建 `LayoutConversionChoice.h/.cpp`、`LayoutConversionChoiceTest.cpp` 并登记 CMake。先断言同块两个同目标 use 只付一次成本、便宜常量/算术 slice 选择 Rematerialize，再实现分组、8-op 限制及叶子计费。接入 solver 完整赋值评分，下界不重复计算边界成本。
- [X] **24-C 验证与物化。** 扩展 `LayoutVerifier.cpp`、`MaterializeLayouts.cpp`。测试真实 solver 输出的转换位置、多个 use 同一个 SSA、克隆结果/属性、8/9 op、循环隔离、hard seed，以及篡改方案失败后原 IR 字节级不变。
- [X] **24-D 集成。** 新建 `test/Transforms/layout-conversion-choice.mlir`；验证主路径 100/零转换 与 10/带转换的两个成本方向、相同主路径选择便宜边界、确定性与推断重跑。保留完整 Copy/Parallel/MMA/Reduce/store 回归。
- [X] **24-E 文档与 Gate。** 同步本计划、`layout_inference_design.md` §8.7–8.9 和 `layout_inference_strategy_comparison.md`，区分实际完成与未来跨 region 优化。运行全量 unit、lit、CTest、Python normalization 和 legacy boundary 扫描；独立复审后记录确切命令与数量，不以历史 Task 23 测试替代。

示例验收断言（具体建图沿用现有 test fixture）：

```cpp
EXPECT_EQ(solution.conversions[0].resolution, EdgeResolutionKind::Rematerialize);
EXPECT_EQ(firstConsumer.getOperand(0), secondConsumer.getOperand(0)); // shared conversion
EXPECT_TRUE(failed(materializeLayouts(module, graph, tampered)));
EXPECT_EQ(before, print(module)); // transaction rollback
```

#### 24.5 原始任务分解（实施时同步勾选）

**Files:**

- Modify: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`
- Create: `include/Dialect/Frisk/Analysis/LayoutConversionChoice.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutConversionChoice.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutCostModel.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutConversionChoiceTest.cpp`
- Create: `test/Transforms/layout-conversion-choice.mlir`

**Interfaces:**

- Produces:

```cpp
enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution;
  uint64_t bytes;
  uint64_t synchronizationCost;
  LayoutConstraintID constraint;
  ConversionExchangeKind exchange;
  uint64_t loopDepth;
  std::string costProvenance;
  Operation *insertionPoint;
  LayoutConstraintID sharedLeader;
  SmallVector<Operation *> rematerialization;
};

bool isRematerializable(Operation *op);
FailureOr<LayoutConversionChoices> chooseLayoutConversions(
    const LayoutConstraintGraph &, const CandidateAssignment &, LayoutTarget &);
FailureOr<CostEstimate> evaluateLayoutConversionCost(
    const LayoutVar &, Attribute source, Attribute target, LayoutConversionEdge &);
FailureOr<CostEstimate> evaluateLayoutCloneCost(
    const LayoutVar &, Attribute encoding, Operation *insertionPoint);
```

- [X] **Step 1: 写“零转换不是绝对优先”的红灯测试**

构造：

```text
方案 A：零 conversion，instructionPathAndWork = 100
方案 B：一次 conversion，instructionPathAndWork = 10，conversion cost = 8
```

预期选择 B。再构造主路径相同而 conversion 更重的 case，预期选择零 conversion。

这里的 8 是原始示意值；实际 Task 24 回归使用合法的 64xf32 编码与 512 单位 shuffle 成本，验证 100/10 的主路径优先级以及同主路径时选择零转换。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

历史红灯预期：solver 尚未枚举 edge resolution。实际 Task 23 已具备两个主路径成本方向的能力，本任务保留该回归，并以共享转换、常量重计算、8-op 切片与方案篡改测试证明新增能力的红灯。

- [X] **Step 2: 枚举 edge resolution**

对于 `Convertible` soft conflict 同时生成：共同布局、consumer-edge conversion、pure producer rematerialization。Hard seed 冲突不得通过 conversion 掩盖。

- [X] **Step 3: 计算 conversion 成本**

至少包含元素 bit 数、tile volume、是否跨 warp、是否需要 shared staging、barrier 数和所在 loop depth。未知 lowering 使用保守上界并标记 provenance，不能记为 0。

- [X] **Step 4: 限制 rematerialization**

只允许 `MemoryEffectOpInterface` 证明无副作用、无 region、无随机/时钟语义、成本低于 conversion 的 backward slice。最大 slice op 数首版设为 8；超过时不枚举。

- [X] **Step 5: 验证选择与 IR 物化**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutConversionChoiceTest.*'
```

Expected: 两个成本方向测试均 PASS；FileCheck 中 conversion 数和位置与 solution 一致。

- [X] **Step 6: 同步 conversion 设计**

确认 `layout_inference_design.md` Section 8.7–8.9 与实际普通候选策略一致。

- [ ] **Step 7: 用户确认后提交（不属于本轮自动执行）**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis \
  test unittests guoqiao/layout_inference_design.md
git commit -m "feat: solve layout conversion placement globally"
```

#### 24.6 实际实现与边界说明

实现沿用 Task 23 的 `LayoutSolution::conversions` 承载边界记录，无需重建 solver 公共结构；新增 `LayoutConversionChoice.h/.cpp` 负责选择与计划核验。`KeepCommonLayout` 表示赋值相同，无需生成记录；`Convert/Rematerialize` 每个实际 use 保留一条记录，同组 leader 标识实际共享的一个转换。`dump-analysis` 的 `conversions` 数量仍是边界记录数，不等于去重后的物理转换操作数。

对固定赋值，同一组的转换源、目标和块相同，所以一个共享转换已经能服务整组。保留共享转换再增加克隆不会降低非负成本；因此只需比较“一次共享转换”和“逐 use 最便宜方案之和”，不需要枚举 `2^users` 个组合。只有第四字段严格更低的合法 slice 才参与；整体仍按九字段字典序比较。先选边界最优值，再参与原 exact/beam 的赋值评分；这是受限可分离模型的最优选择，不是通用 placement 的全局最优保证。

数字例子：64 个 f32 元素，每端 64 个 physical carriers 的同 warp shuffle，估计 512 字节、0 barrier；同一块两个 consumer 共享一次仍为 512，不是 1024。单个常量克隆工作为 256，选择重计算；两个 consumer 各自克隆合计 512 时，共享转换凭较低资源/codeSize 或稳定保守 tie-break 胜出。跨 warp shared/未知路径另加 `4 * maxThreads * 2` 的同步惩罚，循环内边界总工作乘 `2^depth`。这些数字是成本模型单位，不是实测 cycle。

分类预算为最多 65536 个逻辑元素、两端合计最多 262144 个 physical carriers；只对 lane=32 且相同 active-thread topology 的可规范化 BitLinear 映射做 owner 分类。其余已合法的转换使用带 `cta-staging-upper-bound-v1` 来源的保守估计，不把 unknown 当作零开销。超出预算不会触发巨大 owner 表枚举。

重计算只复制原 consumer 同块内的白名单 slice，保持所有原操作及其其他 users；叶子和标量值须支配插入点，同一叶子在同一 slice 内只转换一次。跨块纯 producer 不复制，故循环体可以重算体内常量，但不能因“纯”而复制循环外 producer。循环 init/backedge/result 的既有类型和关系不变。克隆数值估计已统一放在 `LayoutCostModel.cpp`。

独立复审已检查选择、cost、dominance、硬约束、克隆和事务边界；未发现 Critical/Important 问题，补充的四项覆盖/职责划分意见已修正并复审关闭。完整验收数字见 §24.7，不以中间测试结果代替。

#### 24.7 最终验收（2026-09-27）

| 验证范围                                     | 最终结果                                 |
| -------------------------------------------- | ---------------------------------------- |
| 指定构建目标（unit/CLI/FFI/4 个 CTest 程序） | 成功                                     |
| 全量单元测试                                 | 287/287，36 suites，410.369 秒           |
| 全量 lit                                     | 50/50                                    |
| CTest                                        | 4/4                                      |
| Python legacy normalization                  | 5/5                                      |
| 旧生产接口隔离扫描                           | PASS                                     |
| 独立复审及修正复审                           | APPROVE，无遗留 Critical/Important/Minor |
| 差异空白检查                                 | `git diff --check` 通过                |

实际命令（源目录执行）：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt frisk_ffi \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_output=xml:/tmp/frisk-task24-final-unit.xml
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -sv build/test
ctest --test-dir build --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_api_boundary_test.py --source-root . --build-dir build
git diff --check
```

相对 Task 23 新增 15 个单元用例（12 个 choice/物化测试与成本覆盖增量）和 1 个 lit 文件。覆盖共享实际 SSA、共享转换优于多个廉价克隆、单常量/算术重计算、成功且去重的叶子转换、昂贵叶子拒绝、8/9 个唯一操作边界、hard seed、未知/昂贵操作拒绝、循环位置与外部 producer 隔离、i1/i16/i32 字节取整、分类预算回退、饱和、稳定选择与方案篡改回滚。既有 MMA/Reduce、9 变量核心及完整 Copy/Parallel/MMA/Reduce/store 回归仍通过。

红灯与修正轨迹：初始共享/重计算/切片/方案核验测试按预期失败，成本分类与循环权重也先验证失败再实现。第一次全量回归发现旧 capture 测试仍要求两次转换，以及新证书校验抢先覆盖缺失转换诊断；现已将 capture 断言加强为“一次转换、相同 operand SSA、仍在原 parallel 块内”，并恢复具体缺失边/hard conflict 诊断优先级。定向 30/30 通过后，重新执行了上述最终全量 Gate，不把第一次失败运行算作通过。

构建期间存在既有 Ninja `premature end of file; recovering` 警告并导致广泛重编译；指定目标最终正常构建。默认 all-target 的实验 `exp/layout/gemm_layout_tool.cpp` 仍引用退役的 `ParallelOp::inferLayout`，不属于本轮 Gate，未在 Task 24 扩展修复。成本仅为静态代理，没有执行 GPU benchmark 或更新 TileLang/Triton 固定快照。Task 24 修改尚未提交/推送。

### Task 25: 完成数据驱动的 SM90 WGMMA/TMA 布局契约

#### 25.1 已确认设计：方案 A（2026-09-28）

用户确认先写入具体契约，再进入实现与测试。直接在源目录实施，保留未提交的 Task 24 修改；本任务不自动 commit/push，不创建 worktree。状态：已按确认契约完成实现、独立复审与最终 Gate，证据见 §25.4–25.5；未提交/推送。

**Goal:** 接通有来源的 SM90 目标规则、WGMMA/Copy 指令路径及 shape/address-aware 成本，并使选中的契约能在实际 IR 上独立复验。

**Architecture:** 复用现有有限候选、四角色 InstructionContract tuple、CostVector 和事务式物化。Copy 的四角色为 source storage、destination storage、execution layout、instruction path；路径是独立的 Instruction 变量，不能按相同布局去重掉不同路径。MMA 的既有四角色和逐坐标证明保持不变。目标规则只在 SM90 层解释，通用求解器不判断芯片名称。

**Tech Stack:** 当前固定 LLVM/MLIR 工具链、C++17、TableGen/MLIR attributes、GoogleTest、lit/FileCheck；硬件依据为 NVIDIA PTX 与 CUDA Tensor Map API。交付布局与指令契约，不交付完整 GPU async lowering 或性能结论。

#### 25.2 全局实现契约

- **目标解析：** `SM90TargetInfo` 从作用域内 NVVM target/chip/features、适用的 DLTI 设备属性与迁移期 `frisk.target` 读取信息。同作用域冲突、错误类型与不支持芯片必须报错；不能把 sm_90 静默提升为 sm_90a。无显式目标沿用普通布局的 SM90 默认环境，但 WGMMA 仍要求显式 architecture-specific 能力。字符串比较封装在 target adapter。
- **WGMMA：** 本任务数据表覆盖 FP16/BF16 输入、FP32 累加、SS/RS、m64nNk16 atom 与已支持的 warp-group grid；固定 128 threads/warp-group，不限制整个 block 只能 128 threads。复用 canonical distributed fragment、真实 root-relative descriptor 和证明预算。TF32/I8 不新增，非法数学 MMA 不伪装成已实现的 SIMT fallback。
- **Copy：** source/destination/execution/path 联合候选，路径为 TMA、cp.async、vector/scalar；已有 execution/vector/writer/layout 硬要求保持。TMA 首版为静态 whole-tile、非 interleave、单位 element stride 的 global→shared tiled copy；rank 规则覆盖 1–5，不能把 unsupported direction 当作 TMA。没有合法 TMA 时删除该候选，仍保留满足地址与 ownership 的普通 Copy。
- **TMA：** 验证 dtype、rank、box（每维 1–256）、global byte strides、inner box bytes、独立对齐证据、none/32B/64B/128B swizzle 和真实 shared 地址映射；分别表示 global base、shared base 和 phase 约束，不把 128B 写成所有指针的统一要求。第一版不切分超 box 的 tile，不支持未知动态 descriptor；这些只排除 TMA，不自动排除正确的 Copy。mbarrier/proxy 的 lowering 义务显式保存在 binding 中，不把 metadata 当成已经插入的同步。
- **cp.async：** 仅对 global→shared、同线程可证明的对齐连续 4/8/16-byte 分组开放，不能替代已指定更强 vector/ownership 要求。其他情形保留 vector/scalar 路径。
- **静态 ragged：** 仅扩展 whole-tile Copy，静态正 extent，逻辑点与 padded carrier 域均受 65536 预算控制。执行 map 覆盖向上取二次幂的 carrier 域，实际有效范围来自原 Copy shape；每个实际元素恰有一个 elected writer，尾部 carrier 必须屏蔽。guarded shape 进入 Copy 契约和复验，不能作为普通无 guard Tensor encoding 使用。第一版 ragged 采用 scalar path，明确拒绝不相容的显式宽向量；不扩展动态 shape、部分 Copy、Fill 或 ragged MMA。
- **成本：** `instructionPathAndWork` 按实际指令 atom/访问批次数与固定可解释 setup 工作计算；`memoryTransactions` 按地址分组统计 32B sector（未知 base phase 取保守值）；`bankConflictDegree` 按 warp/访问批次的 shared 4B bank 地址计算，读广播与不同地址冲突分开。成本是版本化静态代理，不能声称 cycle。MMA、Copy、Reduce 和普通 StorageAccess 各按实际执行位置计一次，不与 Task 24 边界工作重复收费。
- **保守性：** 证明预算不足不能把未知路径当免费或宣称硬件绝对非法。成本饱和，不回绕；部分赋值使用不高估的下界。路径必须先合法，再比较成本。TMA/cp.async/vector 同一布局下是不同选择，不使用永远偏好某条路径的序号。
- **复验：** Copy binding 必须包含版本、路径、支持域、向量/box/swizzle、guard 与同步义务；从原 SSA、实际 storage、execution map、target 重建，不相信序列化的成本或缓存标志。事务失败不改原模块。实际 IR 缺失、篡改、重复或冲突契约均拒绝。
- **兼容：** 保留 Task 19–24 的硬约束、alias/SCF 语义和回滚；Task 26 的通用有序 relaxation、跨 region hoist 和共享 rematerialization 不计入本任务。

规则来源：[CUDA tiled Tensor Map](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__TENSOR__MEMORY.html)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。只使用 SM90 支持的规则子集，不因最新版文档出现其他架构模式就自动加入。

#### 25.3 实施步骤与验收清单

使用 writing-plans、subagent-driven-development、TDD 和实现后独立复审。按责任分工，构建由主 agent 协调，防止多个 Ninja 同时修改 build。

- [X] **25-A 目标与 WGMMA 表。** 新建 `SM90TargetInfo.h/.cpp`、`SM90InstructionTable.h/.cpp` 与 `SM90ContractTest.cpp`；修改既有 `SM90MmaLayoutProof.cpp`、`SM90GemmConstraints.cpp`、目标入口与 CMake。先写 FP16/BF16、SS/RS、128-thread 单位、major、sm_90/sm_90a、target 冲突测试，看到失败再接入表与目标解析；既有 MMA map/descriptor 回归必须保持。
- [X] **25-B Copy 路径和 descriptor。** 新建 `SM90CopyConstraints.h/.cpp`、`SM90CopyContractTest.cpp`。扩展 Copy 收集为四角色 InstructionContract，增加 path 变量和 `frisk.copy_contract`；通用 tuple/support、graph invariant、物化/实际 verifier 不再假定所有 instruction 都是 MMA。先验证 TMA 合法、非法 alignment/box 保留 fallback、cp.async 4/8/16、硬属性与篡改失败。
- [X] **25-C 静态 guarded Copy。** 修改 `OperationLayoutConstraints.cpp`、`ExecutionLayoutProof.cpp`、执行属性 verifier 和 distributed candidate 入口，新增 `GuardedCopyTest.cpp`。先用 3x5、7x18 与单元素轴证明旧实现拒绝，再实现 padded execution domain、有效域 mask、scalar guard 与复验；测试尾部遗漏/伪造/越界与明确宽向量冲突。
- [X] **25-D 地址/形状成本。** 新建 `SM90CostModel.h/.cpp`、`SM90CostModelTest.cpp`，在 target evaluate 接入前三项。测试连续和跨 sector、同 bank 不同地址和读广播、不同 shape 的路径工作、未知 alignment 的上界、溢出与 partial lower bound；验证至少一项真实 solver 决策随成本变化，而不只调用独立评分函数。
- [X] **25-E 工具与完整链。** 更新 `exp/layout/gemm_layout_tool.cpp` 和其 CMake，去除退役 `ParallelOp::inferLayout`，通过新 pipeline 生成实际 canonical layouts。新建 `test/Transforms/sm90-layout-contracts.mlir`，验证合法/非法/fallback、serialize/reparse/replay 与完整 Copy→MMA→Reduce→store；旧基准通过独立 oracle 对照，不恢复生产旧 API。
- [X] **25-F Gate 与长期文档。** 同步设计文档和策略比较，逐项复审；执行指定构建、全量 unit/lit/CTest/Python 与 legacy API 隔离检查，记录确切结果。未通过项不标完成，不提交/推送。

主要验证命令：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt frisk_layout_tool --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='SM90ContractTest.*:SM90CopyContractTest.*:GuardedCopyTest.*:SM90CostModelTest.*'
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_output=xml:/tmp/frisk-task25-final-unit.xml
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -sv build/test
ctest --test-dir build --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
git diff --check
```

验收关键断言（与具体 fixture 配合）：

```cpp
EXPECT_EQ(selectedPath, "tma");                 // 合法 descriptor + 更低模型成本
EXPECT_NE(fallbackPath, "tma");                // 未证明 TMA，不影响合法普通 Copy
EXPECT_TRUE(failed(verifyMaterializedLayouts(module, target))); // 篡改实际 binding
EXPECT_EQ(before, print(module));               // 失败事务保持原 IR
EXPECT_LT(contiguous.cost.memoryTransactions, strided.cost.memoryTransactions);
```

#### 25.4 实际实现与评分口径

代码已经接入以下路径，最终 Gate 结果在 §25.5 单独记录：

1. `SM90TargetInfo` 统一解释 `frisk.target`、`nvvm.target`、GPU module targets 和适用 builtin module 的 DLTI GPU device。芯片声明取最近作用域，但 DLTI 独立查找并与该芯片核对，不因 GPU module 上有 target 就跳过外层设备约束。支持两个 NVPTX CUDA triple；显式 PTX 版本限定在固定工具链支持集，禁用/未知版本和相互冲突的声明报错。无 target 默认普通 SM90，不默认启用 WGMMA。
2. `SM90InstructionTable` 固化已有 FP16/BF16×SS/RS×N={8,16,32,64,128,256} 的 m64nNk16 规则，共 24 行。block 支持 128/256/512/1024 threads，每 warp-group 固定 128。表筛选之后仍逐点验证 fragment、packing、major、真实根地址和 descriptor；不是用“命中表项”代替证明。
3. `SM90CopyConstraints` 为 source storage、destination storage、execution、path 构造四角色关系，物化 `frisk.copy_contract`。binding 包括 path、target、vector_bytes、valid_shape、guarded、issuer、completion；TMA 另有 box/global_strides/source_offset/destination_offset/swizzle_bytes/interleave。普通向量取满足硬要求且已证明的最大 1/2/4/8/16B 宽度；cp.async 只接受 4/8/16B、Global→Shared 与相应 PTX 能力。TMA 首版 dtype 为 F16/BF16/F32；不能证明的路径删除，正确普通路径保留。
4. `proveGuardedExecutionVectorAccess` 共用原有 footprint 检查，仅用实际 shape 为内存地址建索引。3×5 使用 4×8 carrier，7×18 使用 8×32 carrier；尾部没有访问地址。1×8、1×1 用零 output bits 表达单位轴，不额外填充；一般 GF2 空矩阵 API 未扩大。ragged Copy 的 vector_bytes 必须为 1，Fill/dynamic/partial Copy 支持域不变。
5. `SM90CostModel` 在 Task 24 静态资源/边界成本之上增加前三项，通用 solver 不解释 SM90 名称。MMA 工作量是 group×M/N/K repeat 的 atom 数；Reduce 是输入元素数减输出元素数的合并工作代理；StorageAccess 根据实际 execution ownership 分组。该版本没有把 Reduce 内部同步、scratch 生命周期或硬件 overlap 当作已精确建模。

Copy 工作代理 `sm90-shape-work-v1`：普通 Copy 按真实 load+store warp issue 数；cp.async 按 source issue 数加固定 2 单位提交/等待工作；TMA 按 `4 + ceil(tileBytes/1024)` 估计固定 setup 与 bulk 工作。4、2、1024 是版本化代理参数，不是 latency 或保证的机器指令计数。32×64 F32、128B/1024B 根对齐的 fixture 中，宽 16B 普通 Copy 为 32 单位，cp.async 为 18，TMA 为 12，因此选 TMA；8×16 F32 同类 fixture 中三者分别为 2、3、5，选普通向量。不是给 TMA 一个永远最小的序号。

地址评分 `warp-issue-32B-sectors-v1` 在每个 warp/register-vector batch 中按实际地址统计 Global sector；TMA 统计整块地址集合。未知根对齐相位枚举所有与保证一致的相位并取上界。Shared 评分 `shared-128B-phase-32x4B-distinct-word-conflicts-v2` 将 8B/16B 每线程访问分别分为 16/8 lane 的独立 128B 阶段，每阶段累计 `max(distinct words per bank)-1`；同 word 读取广播不重复计数，不能将一个连续 float4 warp 的四个独立阶段误算为 4-way bank conflict。packed 写使用 distinct-word 带宽代理，不承担写竞态证明（ownership/地址证明负责）。大于 16B 的原生整数类型只给保守代理，不声称原生指令 lowering。

下界保留静态资源值、将前三项和未定边界代价置零；不能提前假定尚未选择的 Copy 路径。各访问本地计数有枚举上限，跨约束求和仍经饱和 `addLayoutCosts`。`costCoverage()` 标明代理版本与未建模 latency/cache/overlap。

工具 `exp/layout/frisk_layout_tool` 已改用完整事务 pipeline，返回真实 canonical storage/distributed/MMA 属性与 IR；返回 Tensor 使 MMA 不被无用代码消除。支持 SS/RS、FP16/BF16、SM90a，输出 schema 为 `frisk-canonical-layout-v1`。旧 `--ldmatrix`/`--lane-vector` 仅发退役提示，不伪造指令保证；只接受单 CTA grid。Python 展示入口已适配，不再把另一套 PyTorch benchmark 当作当前布局性能。生产隔离检查也覆盖该工具，旧算法只在独立测试 oracle 中对照。

#### 25.5 验证与复审记录（2026-09-28）

先验证缺少目标 API、Copy path/binding、非零成本、ragged 支持、canonical swizzle 的失败，再实施。独立审查发现并修复：GPU target 遮蔽适用 DLTI、NVPTX triple 前缀误判、宽 Shared 访问跨 128B 阶段误计 bank；相应测试均已 red→green。工具 smoke 增加实际属性断言，捕获并修复“进程成功但 MMA 被优化删除”的空输出。

第一轮完整单元回归暴露 Copy 新属性未进入 normalize 白名单，以及地址成本接入后旧双消费者测试仍要求零转换；已分别修正，并加强为“恰好一次 shuffle 且 Global sector 少于共同布局”的断言。最终完整构建成功，325/325 unit、51/51 lit、4/4 CTest、5/5 Python normalization、新工具 SS/RS/目标拒绝/输出存活 smoke、包含新工具的旧接口生产隔离检查全部通过。单元测试使用 GoogleTest 原生四分片，分别 82/81/81/81 项，均无失败；不是用定向测试代替全量。仍未执行 GPU benchmark，没有新增运行时性能承诺。

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt frisk_layout_tool -j 16
cmake --build build -j 16
for shard in 0 1 2 3; do
  GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=$shard \
    build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
      --gtest_output=xml:/tmp/frisk-task25-final-unit-$shard.xml
done
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -sv build/test
ctest --test-dir build --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/sm90_layout_tool_test.py --build-dir build
git diff --check
```

相对 Task 24 新增 38 个单元用例：目标/规则表 12、Copy 11、guard 6、地址评分 8、完整 Copy 成本 1；新增 1 个 lit 文件。包括 TMA rank 1–5 与三种 dtype、rank 6/box 512/不合格 inner granularity/40B stride/错误方向/dtype fallback、cp.async 4/8/16B 实际对齐、明确宽向量冲突、descriptor offset 与 guard/完成义务篡改，以及实际 IR 重建。地址测试有精确值：连续 f32 Global 4 个 sector，未知相位保守 5，32B stride 为 32；Shared 同 bank 不同 word 31，读广播 0，连续 8/16B 向量各阶段均为 0。既有独立 legacy oracle 和完整 Copy→Parallel→MMA→Reduce→Global store/重跑/回滚测试保留通过。

三轮独立复审（目标表、guard foundation、整体验收及成本修复复审）已关闭所有重要问题。剩余非阻塞覆盖建议是增加非零且对齐的 alias 起点正例；当前已验证真实 root-relative 计算、地址相位门槛和 descriptor offset 篡改拒绝，不将缺少额外正例写成已测试。Ninja 仍有既有 `premature end of file; recovering` 警告，完整 all-target 已正常完成；此前实验工具的退役接口构建错误已消除。当前源目录 main 的 Task 24/25 修改未 commit/push。

计划中的 `--target=sm_90a --block-m=128 --block-n=128 --block-k=64 --threads=128 --a-space=shared --b-space=shared --dtype=fp16 --ldmatrix` 工具命令也已成功执行，输出经 JSON 检查确有两个 storage layout、MMA contract、accumulator encoding 和可观察的返回结果。`--ldmatrix` 输出退役提示，不代表已选择 ldmatrix 指令。

以下保留原始任务清单；具体实施以上述已确认契约为准。原计划将已存在的 `SM90LayoutTarget.h` 标为 Create，应为 Modify；TF32/I8 枚举不代表这些 MMA 路径已实现。

**Files:**

- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90GemmConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90ReduceConstraints.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp`
- Create: `include/Dialect/Frisk/Target/SM90/SM90LayoutTarget.h`
- Create: `unittests/Dialect/Frisk/Layout/SM90ContractTest.cpp`
- Create: `test/Transforms/sm90-layout-contracts.mlir`

**Interfaces:**

- Produces数据表：

```cpp
enum class SM90ElementType { F16, BF16, F32, TF32, I8 };

struct WGMMAContract {
  int64_t m;
  int64_t n;
  int64_t k;
  SM90ElementType aType;
  SM90ElementType bType;
  SM90ElementType accType;
  bool transA;
  bool transB;
  unsigned threads = 128;
};

struct TMAContract {
  unsigned rank;
  unsigned swizzleBytes;
  unsigned baseAlignmentBytes;
  unsigned innermostGranularityBytes;
};
```

- [X] **Step 1: 写已知合法/非法契约红灯测试**

至少覆盖 BF16/FP16 WGMMA、A/B major mode、128-thread warp-group、32B/64B/128B swizzle、128B base alignment、TMA box/granularity、ragged fallback。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，规则表/成本尚不完整。

- [X] **Step 2: 固化 target 查询**

`SM90LayoutTarget` 从 `#nvvm.target`/DLTI 查询 chip/features；迁移期允许读取 `frisk.target = "sm_90"`，但立即规范化为 target object。通用 Analysis 不允许比较字符串。

- [X] **Step 3: 实现 WGMMA candidate expansion**

每条 contract 展开为 accumulator、dot operand、shared storage canonical maps；展开后调用通用 verifier。MMA Attr 只是 contract view，必须能返回 canonical `DistributedEncodingAttr`。

- [X] **Step 4: 实现 TMA/Copy candidates**

TMA 不合法时保留 cp.async/vector copy candidate；不能把 fallback 当 hard error。Swizzle/padding 必须同时计入 allocation size、bank 和 descriptor legality。

- [X] **Step 5: 实现 shape-aware target cost**

成本来自实际 tile volume、transaction、bank、barrier 和资源估计，不使用一个对所有 shape 固定的“WGMMA 永远优先”常数。Static contract 未命中时只删除对应 candidate，不影响仍正确的 fallback。

- [X] **Step 6: 与现有 layout tool/legacy baseline 差分**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests frisk_layout_tool check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='SM90ContractTest.*'
./build/exp/layout/frisk_layout_tool \
  --target=sm_90a --block-m=128 --block-n=128 --block-k=64 \
  --threads=128 --a-space=shared --b-space=shared --dtype=fp16 --ldmatrix
```

Expected: tests PASS；核心 baseline canonical map 等价；非法 TMA case 仍有合法 fallback。

- [ ] **Step 7: 提交 SM90 规则库（未执行：本轮明确不自动 commit/push，等待用户另行要求）**

```bash
git add include/Dialect/Frisk/Target lib/Dialect/Frisk/Target test unittests
git commit -m "feat: add SM90 layout contract library"
```

### Task 26: 实现 controlled relaxation、conversion hoist 和受控重计算

> **当前状态：方案 A 已完成实现、独立复审与完整验收，实际支持边界见 §26.4，结果见 §26.5。Task25 的“Task26 尚未实施”是此前验收快照，不代表本节最新状态。未 commit/push。**

#### 26.1 已确认设计：方案 A（2026-09-28）

用户确认先将契约写入设计，再进入实现与测试。直接在源目录 main 工作，保留 Task24/25 未提交修改，不创建 worktree，不自动 commit/push。下方旧清单作为历史意图保留，具体以本节契约为准；完成标记已依据最终验收更新。

**Goal:** 在不放宽硬语义的前提下，提供可诊断的有序候选扩展、安全 conversion hoist 和求解阶段计费的共享重计算。

**Architecture:** 普通候选和显式转换/重计算先联合求解；失败用结构化状态分类。仅可证明的空候选域进入从原始图快照开始的独立重试。改写保留 Task24 的有限切片与计划核验，转换优化使用独立事务和实际 IR 验证。

**Tech Stack:** 现有 LLVM/MLIR、C++17、GoogleTest、lit/FileCheck。使用 writing-plans、TDD、subagent-driven-development 与独立代码复审；测试构建统一协调。

#### 26.2 全局契约

1. **普通求解优先。** Task25 的 TMA/cp.async/vector、guarded Copy，以及已有 padding/replication 都保留普通候选身份，不能为制造回退路径而移除。普通方案成功时零 relaxation 尝试。
2. **失败分类。** 空候选域、硬约束冲突、证明 unknown/unsupported、目标错误、预算耗尽、beam 不完整分别处理；只有有证明依据的空域允许重试。硬编码、SameLayout、alias、ownership、线程数、显式 vector 要求和真实 allocation 容量均不可更改。
3. **独立有序重试。** 顺序为 Replication、StoragePaddingOrNarrowerVector、GuardedRagged、TMAFallback、WGMMAFallback。每次基于原始图，不带入前次新增候选；准备结束后冻结域，传播只删减。不支持或已经在普通候选内的阶段明确记录，不重复宣称新增能力。WGMMA 没有已实现替代路径时返回 unsupported，不生成虚假 SIMT 契约。
4. **目标扩展边界。** 新增候选必须通过现有目标/容量/硬关系证明。首版局部 replication 采用有限倍增候选，仍满足既定活跃线程拓扑与唯一写入者；padding 仅使用真实容量内的布局，不扩容 allocation；窄向量不得低于显式硬要求。通用引擎不识别 SM90 芯片字符串。
5. **记录。** 每次尝试记录阶段、触发变量/约束、原因、新增候选数量、结局；选中方案记录最终 CostEstimate。原问题无解时无基准成本，不能伪造 incrementalCost；有明确基准时以分别非负的 increase/decrease 表示变化，禁止无符号回绕。release IR 不写非必要 debug attr。
6. **安全提升。** 有限白名单 pure elementwise、已支持且可证明的 broadcast/extend 布局关系可作转换前移；依赖必须支配新位置，不能越过 effects/barrier/未知 region/ownership 边界。循环提升只处理 source 在循环外、执行语义已证明安全的转换；不能仅凭 Pure 移动 shuffle/Shared 通信跨越控制流。无执行域证明时保守保留原位置，不改变 iter_arg/result 关系。改写必须通过实际布局验证并有非劣成本依据。
7. **共享重计算。** 保留 Task24 的白名单、同块和最多 8 个不同操作限制。相同源、目标布局、切片及可用依赖的多个 use，可共享一次已选克隆；输入转换和切片只计费一次。共享选择进入 solver 计划和计划核验，物化只执行计划，不能事后无条件克隆。跨块/region 共享克隆不扩展。
8. **原子性。** optimizer 独立运行也先克隆模块，检查 dominance、MLIR verifier 和 actual-layout verifier，再提交；失败保持原 IR。cleanup 不承担修复错误推断的责任。重复执行稳定，无指数克隆。

实现时进一步明确：编码本身也是硬 seed，不能由“没有额外属性”推断可改写。物化只有在原结果未编码、调用方明确启用后续优化时才生成临时 `frisk.inferred_result_layout` 授权；优化器核对它与当前编码一致，结束前移除。默认独立 materialize/infer 不输出该标记；完整事务 pipeline 在内存中传递它。不支持的 broadcast 不新增操作语义；已有 arith extend 使用明确白名单。Shared/shuffle 不跨循环推测执行。

#### 26.3 实施清单与接口

> 执行：按 subagent-driven-development 分工；主 agent 负责 relaxation 与集成、构建和 Gate，独立子任务负责转换提升、共享克隆及复审。用户指定的源目录与长期文档位置覆盖技能的 worktree/独立文档默认值。

- [X] **26-A 结构化求解失败与有序扩展。** 新建 `include/Dialect/Frisk/Analysis/LayoutRelaxation.h`、`lib/Dialect/Frisk/Analysis/LayoutRelaxation.cpp`、`unittests/Dialect/Frisk/Layout/LayoutRelaxationTest.cpp`；修改 `LayoutSolver.h`、`LayoutTarget.h`、`LayoutCandidateSolver.cpp`、`LayoutInfer.cpp`、SM90 target 和 CMake。先验证正常解不重试、预算/unknown/hard conflict 不重试、独立快照与顺序，再实现并回归。

```cpp
enum class LayoutSolveFailureKind {
  None, InvalidInput, EmptyDomain, HardConflict, Unsupported,
  SearchBudget, SearchIncomplete, CostFailure
};
struct LayoutSolveFailure {
  LayoutSolveFailureKind kind = LayoutSolveFailureKind::None;
  LayoutVarID variable = UINT32_MAX;
  LayoutConstraintID constraint = UINT32_MAX;
  std::string reason;
};
// 原有调用保持兼容：追加可选输出指针，不依赖诊断字符串分类。
// solveLayoutGraph(graph, target, options, LayoutSolveFailure *failure = nullptr)
// attemptControlledRelaxation(graph, target, options, records, failure)
```

- [X] **26-B 安全 conversion hoist。** 修改 `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`，新增 `test/Transforms/hoist-layout-conversion.mlir` 和独立单元测试。先证明现有 pass 不提升，再实现白名单/成本检查/线程内循环提升/原子验证；负例覆盖 memory/barrier、未知 region、通信转换、非支配输入和 hard encoding。不得把任意 Pure 都视作可移动。
- [X] **26-C 共享已选重计算。** 修改 `LayoutConversionChoice.cpp`、`MaterializeLayouts.cpp` 与相关单元测试，新建 `test/Transforms/layout-relaxation.mlir`。先断言多个 use 应共享同一 SSA 克隆失败，再在同块组比较共享转换、独立修复和共享重计算三种计划，更新核验与物化缓存。负例覆盖源/目标不同、hard seed、跨块、切片上限与篡改 leader。
- [X] **26-D 集成与完整 Gate。** 先构建指定目标并运行定向 unit/lit，后运行全量 unit/lit/CTest/Python/tool/API 隔离。独立复审每个实现块和整体契约，修复重要问题后重跑覆盖测试。同步 `layout_inference_design.md`、`layout_inference_strategy_comparison.md` 与本节实际记录；不将未执行项目标为通过。

定向与最终命令：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt frisk_layout_tool -j 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutRelaxationTest.*:LayoutConversionChoiceTest.*:LayoutConversionHoistTest.*'
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -sv build/test
ctest --test-dir build --output-on-failure
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/sm90_layout_tool_test.py --build-dir build
git diff --check
```

#### 26.4 实际实现边界

- **失败分类与重试。** `LayoutSolveFailure` 不依赖诊断字符串。普通图先验证不变量、准备联合指令/Reduce 支持表，再传播与求解。某处为空不能跳过其他分量的错误；同一分量内仍对非空残余图执行有预算的联合搜索，不能把逐边有支持当作整体有解。残余图仅调用部分赋值成本下界，不为缺失端点的转换/指令计算完整成本，也不产出最终解。无法投影的部分 n 元约束保守返回 Unsupported。未知证明、硬 seed 冲突、预算耗尽、beam 不完整和成本失败均不触发重试。
- **五级调度不等于五种新增硬件能力。** 每一级从原始快照开始，失败不污染调用方图。SM90 实际新增至多四个有限 replication 提案，寄存器位数相对最小值增加 1/2，保持线程数、覆盖和写入者契约，carrier 数不超过 65536；普通候选中已有的 padding/窄向量、guarded Copy、TMA/cp.async/vector 不重新标成“松弛成果”。指令/Reduce 联合表端点及显式硬编码不由通用扩展器修改；没有新 allocation 扩容或 WGMMA→SIMT lowering。
- **可核验记录。** 记录触发变量/约束、阶段、提案数、无新候选/拒绝/选中以及原因；选中时附最终九维成本。无可行 baseline 时 `baselineCost` 为空，打印 `baseline=unavailable`，不做无符号成本相减。耗尽时重放原始诊断及附注，并追加重试摘要。
- **转换前移。** 只接受紧邻的、单结果单 use、一个唯一 Tensor 输入的便宜 elementwise/arith extend；必须有物化器对原本无编码结果的内部授权，显式编码不能被改写。要求依赖支配、相同 shape、成本非劣和实际布局复验；不新建 broadcast 语义。循环提升仅接受循环外来源的线程内 Register 交换，不提升 shuffle/Shared 通信，不越过 effect/barrier/未知 region/Frisk ownership 域。
- **共享重计算。** 同源、同目标布局、同块的多个 use 比较共享转换、独立修复与共享重计算。共享方案严格改善相关成本才选中，持平保留转换；仍限已有白名单和最多 8 个不同操作。leader/follower、切片与位置由 verifier 重算核对；物化缓存同一实际 SSA，叶子转换和克隆只生成、计费一次。不同源/目标/块不合并。
- **事务与清理。** 独立 optimizer 也在模块克隆上操作，通过 MLIR 与实际布局验证后才提交；失败不改变原 IR。完整 pipeline 内传递临时授权，结束移除；默认独立 infer/materialize 不泄漏授权标记。未开展 GPU 性能测试，不把静态成本改进等同于运行时间收益。

#### 26.5 验证与复审记录

执行顺序为设计契约 → 失败测试 → 实现 → 定向回归 → 分块及整体复审 → 全量验收。测试不依赖 GPU；有限域测试使用真实求解器，SM90 replication 正例使用真实 encoding 与目标验证器。普通 Copy 的 bulk/guard/fallback lit 另检查没有 relaxation 记录；未用此测试冒充新的硬件回退实现。

复审发现并补回归的问题包括：早遇空域跳过其他分量错误、二元关系 Unknown 被折叠为冲突、空域使无关指令表未准备、已 finalized 图被修改后未经不变量检查、以及同分量中逐边可行但联合不可行的环。最后一项用 `{p,q}`、`{q,r}`、`{p,r}` 的 SameLayout 环复现；修复为有预算的残余联合搜索。随后补测残余 Convertible 的目的端为空时不得误用完整成本 API。耗尽诊断单独保存原始位置和附注，避免后续失败提案覆盖原始证据。

共享重计算覆盖一次计费/一次克隆/同 SSA、叶子转换去重、leader/位置篡改回滚、不同源/目标/块不共享，以及原有硬 seed/8-op 上限。转换提升覆盖单输入/extend/Register 循环正例和显式编码、双输入、通信、effect/barrier、未知 region、非支配输入、MLIR/实际布局验证失败不提交等反例。

**最终结果（2026-09-28）：**

- 显式构建 `FriskLayoutUnitTests frisk-opt frisk_layout_tool` 和默认 all 目标均成功；默认 all 不含 unit 目标，不能用其成功代替单测程序重建。
- 定向 56/56：17 relaxation、16 conversion choice、11 hoist、12 solver。全量单元测试四分片 90+89+89+89=**357/357**，XML 的 failures/errors 均为 0。新增 Convertible fixture 最终使用真实 SM90 encoding，直接检查原图返回 EmptyDomain、修复后成功；不把 StringAttr 占位 encoding 的错误拒绝当成有效验收。
- **53/53 lit、4/4 CTest、5/5 Python normalization**；SM90 tool 的 SS/RS 可观察结果与非法目标/多 CTA 拒绝、旧生产接口隔离检查全部通过。现有大 tile MMA、Copy→Parallel→MMA→Reduce→store 整链与 replay 回归保持通过。
- B/C 分块复审和 A 修复后复审、最终整体源码复审均无剩余 Critical/Important 问题；`git diff --check` 通过。未执行 GPU benchmark、完整异步 lowering 或上游框架性能比较。

全量 unit 使用 `GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=0/1/2/3` 分别运行测试程序，并写独立 XML；与不分片运行覆盖同一组测试。生产代码最终构建后运行所有分片，测试夹具修正后重建并重跑受影响分片及全部定向测试。日志位于 `/tmp/frisk-task26-final-unit-{0,1,2,3}.{log,xml}`、`/tmp/frisk-task26-focused-final4.log`、`/tmp/frisk-task26-final-lit2.log`；这些临时文件仅作本次证据，长期依据为本节命令、仓库测试及结果记录。

以下为原始清单，WGMMA fallback 和增量成本等表述由上述契约收紧：

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutRelaxationTest.cpp`
- Create: `test/Transforms/layout-relaxation.mlir`
- Create: `test/Transforms/hoist-layout-conversion.mlir`

**Interfaces:**

- Produces:

```cpp
enum class RelaxationKind {
  Replication,
  StoragePaddingOrNarrowerVector,
  GuardedRagged,
  TMAFallback,
  WGMMAFallback
};

FailureOr<LayoutSolution> attemptControlledRelaxation(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    SolverOptions options, SmallVectorImpl<RelaxationRecord> &records);
```

- [X] **Step 1: 写 relaxation 顺序红灯测试**

覆盖：普通 conversion 候选存在时不进入 relaxation；TMA alignment 不满足选择 cp.async；ragged shape 选择 guarded candidate；WGMMA contract 合法时不得为减少 conversion 回退 SIMT。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，relaxation records 不存在。

- [X] **Step 2: 实现严格有序 relaxation**

仅当普通 candidate domain 为空时依次尝试：replication、padding/窄 vector、guarded ragged、TMA fallback、最后 WGMMA illegal fallback。每一步从原始 graph 快照重建 domain，不能累积未选松弛。

- [X] **Step 3: 记录原因与可核验成本（按确认契约修订，不伪造不可行基准）**

```cpp
struct RelaxationRecord {
  RelaxationKind kind;
  LayoutVarID variable = UINT32_MAX;
  LayoutConstraintID trigger = UINT32_MAX;
  uint64_t addedCandidates = 0;
  RelaxationOutcome outcome = RelaxationOutcome::NoNewCandidates;
  std::string reason;
  std::optional<CostEstimate> baselineCost;
  std::optional<CostEstimate> selectedCost;
};
```

诊断/debug IR 可打印 records；release IR 不保留非必要 debug attr。

- [X] **Step 4: 实现 conversion hoist（按 §26.2/26.4 的安全子集）**

实现已授权的单输入 pure elementwise/extend 前移和 Register loop-invariant conversion 提升；当前没有可用 Tensor broadcast 模型，不新增其语义。不得越过 memory effects、未知 region、barrier 或 ownership boundary；支配、成本与实际布局核验后才提交。

- [X] **Step 5: 实现受控 rematerialization rewrite**

只重写 solver 已选择 `Rematerialize` 的 slice；使用 `IRMapping`，共享已有 rematerialized value，避免指数复制；完成后删除死 conversion 并运行 canonicalizer。

Task 26 复用 Task 24 同块、每 slice 最多 8 个操作的已选重计算，在求解评分、计划核验和物化缓存中补齐安全的跨 use 共享；不重新无条件搜索/克隆，叶子转换和克隆只计费一次。

- [X] **Step 6: 验证 relaxation/hoist**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutRelaxationTest.*'
```

Expected: PASS；每个 case 的 relaxation kind、conversion count 和位置稳定。

- [ ] **Step 7: 同步设计并提交（设计已同步；commit/push 未获本轮授权，保留未执行）**

```bash
git add lib/Dialect/Frisk test unittests guoqiao/layout_inference_design.md
git commit -m "feat: relax and optimize layout assignments safely"
```

### M5 Gate

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
build/bin/frisk-opt test/Transforms/layout-conversion-choice.mlir \
  -frisk-layout-pipeline -verify-each
```

Expected: exhaustive 小图 oracle、全局成本选择、SM90 contracts、conversion/rematerialization 和 relaxation 全部通过；不存在“零转换绝对优先”行为。

---

## M6：正确性、差分测试与性能验收

### M6.1 已确认方案 A 与实施顺序（2026-10-09）

**Goal:** 为现有布局系统建立可复现的性质、差分、真实 conversion 执行与性能证据，不借验收扩大生产语义或削弱已有硬约束。

**Architecture:** Task27 建立独立性质 oracle、确定性与失败复现；Task28 在隔离的固定版本 TileLang 环境导出语义，Frisk 在无 TileLang 运行依赖的测试中读取 corpus；Task29 聚合真实统计、CPU benchmark 和经过实际 adapter 的 GPU conversion 路径，按明确适用域判定 Gate。三个 Task 顺序推进，每个 Task 单独进行代码/契约复审，修复重要问题后再进入下一项。

**Tech Stack:** 现有 C++17、LLVM/MLIR、GoogleTest、lit/FileCheck、LLVM JSON；隔离 Python/TileLang 参考环境；可选 CUDA 12.8、SM90 与 compute-sanitizer。默认构建不增加 CUDA 或 TileLang 依赖。

源目录直接实施，保留 Task24–26 及报告的未提交修改，不使用 Frisk worktree，不自动 commit/push。长期方案直接维护本文、设计总文档和策略比较文档；原各 Task 的提交步骤只在用户另行授权后执行。

**审计起点：** Task26 的记录为 357 unit / 53 lit / 4 CTest / 5 Python normalization；这不是本轮已重跑的结果。2026-10-08 只读环境检查确认两张 H100 PCIe、compute capability 9.0、driver 580.178.04、nvcc 12.8。沙箱内 NVML 不可访问，沙箱外只读查询成功；实际 GPU 测试仍需对应执行权限。固定 TileLang `5e149e31674658f94779c7d0c6039549a1853123` 源码已找到，但缺少子模块初始化及构建产物。`TestLayoutConversionLowering.cpp` 仍有 Tensor↔vector 的 `UnrealizedConversionCastOp`，不能当成已完成可执行 lowering。

### M6.2 三处原提纲修订

1. **256 的含义。** 与 §23.4 一致，去重/硬剪枝后的组合数不超过 256 才默认精确搜索，超过则 beam32；每组件每次尝试最多 65536 次扩展。分别记录单变量域、组件组合乘积、frontier 和扩展数，不把它们混成 `peakComponentCandidates`。乘积溢出单独标记，不静默截断到 256；指令/alias/Reduce 自身证明预算不因此放宽。
2. **差分并不要求支持上游全部功能。** 共同支持域比较实际坐标语义；Frisk 明确不支持的 dtype-changing view、一般非二次幂 Tensor/broadcast、PartialFragment 等作为单独的 expected-rejection 用例，核对既有边界和诊断，不悄悄实现新语义。合法的多个同值持有者不是错误；重复载体记录、冲突载体映射和非法写入者才是错误。
3. **编译占比必须有真实分母。** 对受支持可执行用例，计时覆盖输入 IR 到最终可加载 GPU 产物的实际编译流程；分子包含该次流程的布局候选准备、传播/搜索、物化、布局优化及验证，不仅是 solver。外部依赖构建、测试、故意等待、无关 kernel 编译和 GPU 执行不进入分母。没有完整编译路径的用例只报告各阶段耗时，不能写成 10% 门槛通过。

3% runtime 回退、10% 布局编译占比门槛不变。若实测失败，记录失败、定位并在已批准范围内改进；不得通过修改阈值、删除失败用例或伪造 baseline 让 Gate 变绿。

### M6.3 性质与差分的共同证据规则

- 测试生成器固定 seeds `{1, 7, 42, 20260816}`；输出 seed、case 编号、稳定变量/约束身份及首个反例，不输出地址，不依赖 DenseMap 顺序。
- GF(2) 矩阵性质使用独立逐位枚举检查，不用同一生产 compose/inverse 实现同时生成期望与被测结果。不可逆或非满行秩必须拒绝相应 inverse/rightInverse 请求，不能丢弃失败样例只保留成功矩阵。
- 小图 oracle 独立枚举完整赋值，核验全部硬关系和九维成本。exact 或未发生截断的搜索才检查固定域最优性；有截断的 beam 仅承诺合法性和稳定结果，不能把它未找到解当成已证明无解。
- 图扰动仅改变变量/约束/候选的插入顺序，保留稳定语义身份；不能通过随意交换有依赖或有副作用的 IR 操作构造“等价程序”。canonical 输出排除时间、地址和枚举顺序噪声，但不能去掉实际赋值、binding 或边界计划差异。
- 缩减只接受保持同一结构化 failure signature 的合法重放用例。稳定尝试删除 constraint、variable、candidate 至不动点，得到在这些单步删除操作下的 1-minimal 结果，不承诺全局最小。无悬空 SSA/use/alias/region 端点；重编号后仍保留可比的稳定身份；非法图不能替代原始语义反例。
- corpus 内容必须来自运行 pinned TileLang 语义对象后的枚举。拒绝用 Frisk 输出填充期望值、从当前默认 site-packages 导入另一版本、或手抄公式冒充实际导出。独立 reference 环境不写入用户现有 TileLang checkout。

### M6.4 GPU 与基线的共同证据规则

- `inferred_*` 必须由 Frisk 输入和实际 conversion adapter 产生；先消除占位 Tensor/vector 桥接，使用真实 tile load/store 与每线程 vector 传输，再 lower 到可执行 GPU 产物。不得用第二套手写 CUDA 分配公式替代 adapter；生成链需记录输入、编译命令和产物 hash。
- `reference_*` 为独立实现，与 inferred 路径共用输入、输出解释和校验器；身份、warp shuffle、跨 warp Shared exchange 三类至少各有一个实际执行用例。先验证输出，再做 memcheck/racecheck/synccheck；工具缺失、权限失败或 case 未执行均不是通过。
- Shared 路径保留已证明的唯一写入者、容量、对齐和 uniform barrier 义务。不能仅验证最终输出就声称无 race/OOB，不能将某个测试的无错误推广到任意程序。
- GPU 热身 100 次、测量 1000 次；CPU solver 热身 5 次、测量 30 次。记录原始样本、median/p95、硬件 UUID/驱动/toolkit、编译 flags、时钟策略和是否有其他负载。计时区间不打印 IR；正确性/安全工具时间不混入性能时间。
- baseline 来自相同环境下独立 reference kernel 的测量；环境或 case manifest 不匹配返回不可比较，不自动覆盖 baseline。首次记录保留为待复审证据，经过性能评审后才能作为接受基线；缺失基线不填 0，也不把 inferred 首次结果作为 reference。
- 当前源码有未提交修改，仅记录 Git HEAD 不足以复现，产物记录须同时包含相关源码/输入/配置的内容摘要。使用独立 Release 构建目录 `build-m6-release`，不把现有 Debug `build` 原地改成 Release；不自动更改 GPU 时钟、功耗或终止他人进程。
- WGMMA/TMA 只有静态契约的路径不纳入虚构 runtime 基线；其静态验证状态与 conversion adapter 的真实 GPU 状态分开列出。

### M6.5 完成条件与记录方式

- [x] 用户确认方案 A 和上述原提纲口径修订。
- [x] 同步本文、设计总文档与比较文档中的契约口径。
- [x] Task27 实现、局部回归与独立复审完成。
- [x] Task28 真实导出、重导出字节一致、比较/预期拒绝测试及独立复审完成。
- [x] Task29 CPU、GPU 正确性/安全、runtime、适用域完整编译占比及独立复审完成。
- [x] 全量构建/CTest/lit/unit/Python/API 隔离重跑；逐项填写 `guoqiao/layout_inference_acceptance.md`。

验收记录每项必须为 PASS、FAIL、BLOCKED 或明确的 OUT_OF_SCOPE，附命令、适用 case、原始产物和原因。OUT_OF_SCOPE 仅用于事先排除的生产能力，不能用于掩盖 M6 必需的三个 conversion case 或真实 TileLang 导出未运行。总体 M6 Gate 缺一项证据即保持未完成，不因 CPU 通过、工具代码写完或 GPU 可见而标完成。

### Task 27: 建立 property、随机图和确定性测试

#### 27.1 本轮实施契约

本 Task 不新建第二套生产布局推断器。测试复用 `GF2Matrix`、布局 map 接口、`solveLayoutGraph`、`attemptControlledRelaxation` 和 verifier；独立 oracle 仅在测试支持库中存在。

实施分为三个可独立验收的单元：

1. **27-A 代数性质。** 新增 `LayoutPropertyTest.cpp`，对每个 seed 至少生成 32 个最多 8 input bits 的矩阵/布局样例，枚举有限输入核验恒等、组合、逆、右逆、canonicalize 幂等与 Product 分阶段求值。覆盖奇异矩阵、非满射及形状不匹配的拒绝，输出最小坐标反例。修改单元测试 CMake 后先确认缺失测试能力或故意破坏性质能触发失败，再补齐测试/必要修复并运行局部测试。
2. **27-B solver 与确定性。** 新增 `LayoutSolverPropertyTest.cpp`；每 seed 至少 16 张最多 8 vars/16 constraints 的小图，最多 4 candidates/var（穷举最多 65536 赋值）。每图 20 次打乱插入顺序，对照独立 oracle；精确 oracle 测试显式配置足够搜索预算，另设默认 exact256/beam32/65536 边界用例。覆盖 empty、hard conflict、Unknown、SearchBudget、SearchIncomplete、CostFailure，保持 Task26 的重试资格。新增 diagnostics/determinism lit，检查稳定诊断和跨进程布局 IR 字节一致。
3. **27-C 重放与缩减。** 新增 `tools/layout-reduce/{CMakeLists.txt,layout-reduce.cpp}` 及相邻的测试支持序列化/缩减模块、`LayoutFailureReductionTest.cpp`。输入包含 schema、seed、options、稳定变量/候选/关系描述和 failure signature；需要 SSA/use/alias/region 信息时同时保存宿主 MLIR，以稳定操作路径重建引用，禁止序列化指针。共同 parser 服务 unit 和 CLI，不复制解析逻辑；不支持的 payload 明确拒绝，不能丢弃字段后声称可重放。测试覆盖 round-trip、错误 schema/引用、不同失败类别不得替换、缩减幂等和删除不再能缩小。

缩减的原提纲接口保留为内部算法入口，但工具必须同时拥有宿主 IR 和序列化元数据；仅返回含裸指针的 graph 不是完整文件格式。所有删除操作先重建并验证图，再运行 oracle/失败谓词。必要的支持代码放在 `tools/layout-reduce/` 的独立 library target，不将测试 target 或假成本模型链接进生产 `FriskAnalysis`。

典型回归断言为：

```cpp
// GF2Matrix::compose 的方向是 lhs(rhs(x))。
ASSERT_TRUE(succeeded(composed));
EXPECT_EQ(composed->apply(x), lhs.apply(rhs.apply(x)));
// 生产 solver 的 exact 结果对比独立全赋值枚举，而不是第二次调用生产 solver。
ASSERT_TRUE(succeeded(solution));
EXPECT_FALSE(solution->cost.cost < oracleCost);
EXPECT_FALSE(oracleCost < solution->cost.cost);
```

局部命令使用下方 Step5；补充 `--gtest_filter='LayoutFailureReductionTest.*'` 及 CLI round-trip lit。诊断中的空原始域允许以结构化 `EmptyDomain` 加变量/provenance 记录表达，不伪造一个从未存在的“被拒候选”。

**Files:**

- Create: `unittests/Dialect/Frisk/Layout/LayoutPropertyTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutSolverPropertyTest.cpp`
- Create: `test/Transforms/layout-diagnostics.mlir`
- Create: `test/Transforms/layout-determinism.mlir`
- Create: `tools/layout-reduce/CMakeLists.txt`
- Create: `tools/layout-reduce/layout-reduce.cpp`
- Create: `tools/layout-reduce/LayoutFailureCase.h`
- Create: `tools/layout-reduce/LayoutFailureCase.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutFailureReductionTest.cpp`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Modify: `tools/CMakeLists.txt`

**Interfaces:**

- Produces固定 seed property runner 和失败图最小化工具：

```cpp
struct LayoutPropertyOptions {
  uint64_t seed;
  unsigned cases;
  unsigned maxBits;
  unsigned maxVars;
};

FailureOr<LayoutConstraintGraph>
reduceFailingLayoutGraph(const LayoutConstraintGraph &graph,
                         function_ref<bool(const LayoutConstraintGraph &)> fails);
```

- [x] **Step 1: 写会暴露顺序问题的随机测试**

固定 seeds `{1, 7, 42, 20260816}`，生成最多 8 input bits、8 vars、16 constraints 的可枚举小图；对每张图执行 20 次稳定 shuffle，并比较 canonical solution dump。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: 在最小化工具和完整确定性保障实现前，测试应至少缺少目标或失败。

- [x] **Step 2: 增加代数 properties**

必须验证：

```text
identity compose A == A
(A compose B)(x) == A(B(x))
inverse(A)(A(x)) == x when invertible
rightInverse covers every required logical point
canonicalize(canonicalize(A)) == canonicalize(A)
Product evaluation == outer/inner staged evaluation
```

- [x] **Step 3: 增加 solver properties**

小图与独立穷举 oracle 比较合法性和最小 CostVector，最优性断言仅适用于 exact/未截断搜索；所有 hard constraint 在 solution 上重新执行；空 domain 必须有结构化失败与变量/provenance 记录，不能凭空编造候选拒绝。

- [x] **Step 4: 实现失败图最小化**

按稳定顺序尝试删除 constraint、var、candidate，保留合法且仍触发同一 failure signature 的子图，迭代至 1-minimal；输出可被共享 parser 重新加载的文本，不输出地址，不承诺全局最小。

- [x] **Step 5: 验证 diagnostics 和 determinism**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests layout-reduce check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutPropertyTest.*:LayoutSolverPropertyTest.*'
```

Expected: 所有固定 seed PASS；重复执行 canonical output 字节相同。

- [ ] **Step 6: 提交 property suite（本轮不自动执行，等待另行授权）**

```bash
git add unittests test tools
git commit -m "test: add layout property and determinism coverage"
```

#### 27.2 实施与独立复审（2026-10-09）

已新增代数、solver、失败重放三组共 16 项单测以及 diagnostics/determinism/reduce 三项 lit。四 seeds 各 32 个矩阵、32 个 Product 样例逐点枚举；四 seeds 各 16 张有限图，每图 20 次插入顺序扰动，以独立全赋值 oracle 比较九维成本，另外分别检查九字段可决定最优值、默认 exact/beam 边界和非重试失败。精确 oracle 专用配置不改变生产默认预算。

`evaluateLayoutMap` 是已有私有求值器的公开包装，不新增布局语义。失败工具 schema1 明确限定 `finite-same-require-v1`，最多 8 vars/16 hard SameLayout 或 RequireEncoding/4 字符串候选；零成本测试 target 与生产库隔离。CLI 与单测共享 parser，保存宿主 MLIR、稳定身份和结构化 failure signature。指令表、SSA/use/alias/region、Convertible 或硬件成本载荷明确拒绝，不宣称完整 GPU 图可重放。所有合法单步删除迭代至 1-minimal，不承诺全局最小。

局部证据：`cmake --build build --target FriskLayoutUnitTests layout-reduce check-frisk --parallel 12` exit0，56/56 lit。独立复审发现 raw reducer 重建时可能丢失候选 ordinal，已先用两个红灯反例复现，再保留稳定候选身份，并检查重建起点及最终返回图仍满足谓词；增加第三个规范化敏感拒绝测试。修复后包含 16 项新增测试的 45/45 focused unit 和 3/3 受影响 lit 通过，复审结论 spec/quality PASS。日志 `/tmp/frisk-task27-review-{red,green-unit,green-lit}.log`，复审 `/tmp/frisk-m6-task27-review.md`。随后主代理全量重跑 373/373 unit（四分片 94+93+93+93）及 56/56 lit 均 exit0，日志 `/tmp/frisk-task27-full-unit-{0,1,2,3}.{log,xml}`、`/tmp/frisk-task27-full-lit.log`。Task27 已完成；Task28/29 和 M6 最终 Gate 仍未完成，未 commit/push。

### Task 28: 建立固定 TileLang 语义差分 corpus

#### 28.1 本轮实施契约

本 Task 必须真实构建并执行 pinned TileLang，不以源码阅读或从 Frisk 生成的 JSON 代替语义导出。采用隔离 reference checkout、隔离 Python 环境和精确依赖版本；执行前核对 HEAD、工作区状态、submodule revision、Python 模块路径和实际加载 native library 路径。不修改用户现有 TileLang checkout，也不把 TVM/TIRX 链入 Frisk。

首版 schema 使用版本 2，替代下方历史 v1 提纲；必需字段为 `schema_version`、完整 `tilelang_commit`、依赖 revisions、exporter/source digest、cases。每 case 必须记录稳定 name、operation/shape/dtype、carrier extents、枚举后的 distributed/storage bit coordinates，以及 `frisk_expectation`：

- `equivalent`：Frisk 构建实际布局/契约并枚举对比，不能直接把 corpus 坐标作为 Frisk 的输出。比较 coverage、逻辑持有者集合、同值 copy group、有效域及按所声明策略选出的 writer。存在多个合法 owner 策略时比较策略兼容性，不伪造上游没有暴露的 writer 选择。
- `reject`：记录既有不支持原因及稳定诊断类别；Frisk 必须实际拒绝对应输入，不能仅因 JSON 标注 reject 就算通过。单独统计，不计入语义等价通过数。

输入和结果的完成值/部分贡献必须区分。PartialFragment 若仅用于 expected-rejection，不要求 Frisk 新增表示；不能将 addends 作为普通副本比较。reader 拒绝重复 case、重复物理 carrier、同一 carrier 冲突映射、越界 bit offset、错误覆盖及写入者，但允许由明确同值 copy group 解释的多个逻辑点持有者。

实施步骤与 Gate：

1. **28-A reader 与反例。** 新增 corpus reader/测试支持模块及 `TileLangDifferentialTest.cpp`，先用 schema 错误、revision 错误、carrier 冲突、合法复制、位偏移越界的最小 JSON 做失败/成功测试；parser 不执行 JSON 中的命令或路径。
2. **28-B exporter 与真实 corpus。** 新增 `export_tilelang_layouts.py`；在单独子进程导入 pinned 模块，逐项构造下面 Step3 的用例、枚举、排序后写 JSON。审计日志另存，时间戳和临时绝对路径不进入 canonical corpus。两次导出用 `cmp` 比较完全一致；失败不覆盖已有 corpus。
3. **28-C Frisk 差分与范围记录。** 共享支持域至少覆盖 FP16/BF16 SM90 GEMM、linear/transpose/32B/64B/128B Shared、Copy vector width、dim0/1 Reduce 和静态 guarded Copy；其余类别逐项判定等价或预期拒绝，不能遗漏。首个不一致输出 logical/carrier 坐标、bit offset 或 owner 反例。完成 `guoqiao/tilelang_layout_reference.md`，记录采用与不采用的语义、重导出命令和升级规则。

测试在不安装 TileLang 的普通 Frisk CI 中读取已经审计的静态 JSON；exporter 的来源验证/错误路径另以 Python 测试覆盖。若固定版本暂时构建不出，Task28 保持 BLOCKED，不提交手填 corpus 冒充完成。

**参考版本决策（2026-09-12）：** 本任务尚未执行。以下 schema/exporter/命令的拟定主基线升级为本轮审计的 `5e149e3`；这不是已经更新或运行了 corpus。旧 `6623b12` 可另存带版本标签的历史回归，不混入新版 schema。PartialFragment/combine 语义若纳入 corpus，须先扩展 schema 区分 addends 与同值 copy groups；不得继续仅以 replication 数字代表全部语义。

**Files:**

- Create: `tools/layout-reference/export_tilelang_layouts.py`
- Create: `unittests/Dialect/Frisk/Layout/TileLangCorpus.h`
- Create: `unittests/Dialect/Frisk/Layout/TileLangCorpus.cpp`
- Create: `test/python/tilelang_exporter_test.py`
- Create: `test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json`
- Create: `unittests/Dialect/Frisk/Layout/TileLangDifferentialTest.cpp`
- Create: `guoqiao/tilelang_layout_reference.md`

**Interfaces:**

- 历史 corpus schema v1 提纲（实现使用 §28.1 的 v2，下面的空数组不是可接受的验收数据）：

```json
{
  "schema_version": 1,
  "tilelang_commit": "5e149e31674658f94779c7d0c6039549a1853123",
  "cases": [
    {
      "name": "sm90_bf16_gemm_128x128x64",
      "shape": [128, 128, 64],
      "dtype": "bf16",
      "carrier_extents": {"register": 8, "lane": 32, "warp": 4},
      "distributed_points": [],
      "storage_points": [],
      "replication": 1,
      "writer_owners": []
    }
  ]
}
```

测试只读取提交到仓库的 JSON，不要求 CI 安装 TileLang。

- [x] **Step 1: 写 corpus reader 红灯测试**

测试拒绝 schema 版本错误、commit 不匹配、重复物理 carrier/冲突映射和越界 storage bit offset；由同值 copy group 明确解释的合法多个 owner 必须接受。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，corpus/reader 尚不存在。

- [x] **Step 2: 实现固定版本 exporter**

脚本启动时执行：

```python
expected = "5e149e31674658f94779c7d0c6039549a1853123"
if args.revision != expected:
    raise SystemExit(f"unsupported TileLang revision: {args.revision}")
actual = subprocess.check_output(
    ["git", "-C", args.tilelang, "rev-parse", "HEAD"], text=True
).strip()
if actual != expected:
    raise SystemExit(f"TileLang revision mismatch: {actual} != {expected}")
```

Exporter 输出枚举语义，不输出 TileLang Attr 文本或 Python 对象 repr；JSON key 排序、数字格式稳定。

- [x] **Step 3: 生成首批 corpus**

至少包含：

```text
SM90 BF16/FP16 GEMM
linear/transpose/32B/64B/128B shared swizzle
copy vector widths
reduce dim 0/1
non-power-of-two broadcast
ragged padding guard
dtype-changing view
unused/floating fragment
owner compatibility
```

不得直接从 `/home/baopeihua/tilelang` 当前 checkout 导入模块，也不假定其中已有新版对象。在独立 reference clone 中 checkout pinned commit，不创建 Frisk 或参考 worktree；exporter 必须检查精确 `HEAD`、依赖和实际加载路径。以下路径是任务实施时新建的示例，已存在时先检查而非覆盖：

```bash
git clone --no-checkout https://github.com/tile-ai/tilelang.git \
  /tmp/frisk-m6-tilelang-reference
git -C /tmp/frisk-m6-tilelang-reference checkout --detach \
  5e149e31674658f94779c7d0c6039549a1853123
python tools/layout-reference/export_tilelang_layouts.py \
  --tilelang /tmp/frisk-m6-tilelang-reference \
  --revision 5e149e31674658f94779c7d0c6039549a1853123 \
  --output test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json
```

Expected: 完成固定依赖构建后的 exporter 成功，第二次执行产生完全相同文件。reference checkout 保留用于复核，不自动删除。上述命令仅展示版本与导出入口，完整依赖构建命令在实际 reference 文档中记录，不能省略构建而宣称运行成功。

Exporter 还必须把 pinned checkout 插入独立 Python 进程的首个 import path，并拒绝从系统 site-packages 导入 TileLang；启动日志打印实际模块/native library 路径和 commit，文档审计时一并核对。

- [x] **Step 4: 实现语义比较**

Frisk 对每个 case 枚举 canonical map，比较 coverage、logical ownership、replication、writer owner 和 storage bit offset。允许不同构造文本，只比较语义；失败打印首个 carrier/logical coordinate 反例。

- [x] **Step 5: 运行差分测试**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='TileLangDifferentialTest.*'
```

Expected: 所有 equivalent case 语义比较 PASS，所有 reject case 的实际拒绝符合预期；两类分别报告，不把拒绝项计成已支持功能。

- [x] **Step 6: 记录参考策略（文档已完成；以下 commit 等待另行授权，未执行）**

`guoqiao/tilelang_layout_reference.md` 记录 commit、schema、重新生成命令、采用/拒绝的语义和更新审计流程。

```bash
git add tools/layout-reference test/Dialect/Frisk/layout/Inputs \
  unittests/Dialect/Frisk/Layout/TileLangDifferentialTest.cpp \
  guoqiao/tilelang_layout_reference.md
git commit -m "test: add pinned TileLang layout corpus"
```

#### 28.2 实施结果（2026-10-09，独立复审通过）

隔离参考 `/tmp/frisk-m6-tilelang-reference` 已真实构建并运行 pinned TileLang `5e149e31674658f94779c7d0c6039549a1853123`、TVM `907a88c8791ccf33b9874821bc875e7abf624367` 及全部固定递归依赖。原用户 TileLang checkout 未修改；独立 venv 使用 FFI 0.1.12，并逐项核验 Python 依赖、实际加载模块及 native library 路径/摘要。默认 Frisk 构建仅使用静态 JSON，无 TileLang/TVM 链接依赖。

实际 corpus 共 23 个 case：20 个等价比较、3 个真实拒绝，覆盖四个 FP16/BF16 WGMMA A/C、五种 Shared、三种 Copy vector width、两个 Reduce 轴的输入/结果、owner/floating/unused、静态 guarded Copy，以及非二次幂 Tensor、dtype view、PartialFragment 的拒绝边界。Copy/Reduce/guard/floating 使用 native LayoutInference；GEMM 使用 native emitter。floating 比较上游推断结果和 Frisk 显式 canonical map，不宣称默认选择策略相同；unused 测试明确 seed 的保留。writer 是显式兼容策略，不伪称上游固定选出了该 writer。

reader 严格检查 schema/revision、完整载体覆盖、同值 copy groups、partial addends、位地址、writer 及资源边界；Frisk 布局由独立构造和实际 proof 验证，坐标表仅作期望数据。局部 9/9 C++ tests（其中一个枚举全部 23 case）、12/12 无 site-packages Python tests、5/5 CTest 及旧生产接口隔离检查通过。全量 382/382 unit（96+96+95+95）及 56/56 lit 通过，日志 `/tmp/frisk-task28-full-unit-{0,1,2,3}.{log,xml}`、`/tmp/frisk-task28-full-lit.log`。

独立复审发现 exporter 在 audit 写入失败或输出路径重叠时可能破坏旧 corpus。先以回归测试复现，修复为校验目标身份、暂存并 fsync 两份输出、先发布 audit、最后替换 corpus；清理错误不会在替换 corpus 后误报导出失败。不承诺两个文件的原子事务：最后的 corpus rename 失败时旧 corpus 保留，较新的 audit 可由摘要识别不匹配。复审 spec/quality PASS，无未关闭的重要问题，见 `/tmp/frisk-m6-task28-review.md`。

修复后再次执行两次真实导出，corpus 和 audit 分别 `cmp` exit0；最终 corpus SHA-256 `bb4a69f647305e9017ecdbb063e87a1a4cf39ec33cf5ee77ec0dd3936842fd44`，exporter SHA-256 `5ebad2735a4fa6368b4901dd518fee73fcc4aa7ff2f9e204b134f68cac9e7453`。源码、依赖、命令和范围见 `guoqiao/tilelang_layout_reference.md`；修复与导出证据 `/tmp/frisk-m6-task28-i1-*.log` 和 `/tmp/frisk-m6-tilelang-export.audit.json`。Task28 完成，不代表 GPU benchmark 或 M6 最终 Gate 通过；未 commit/push。

### Task 29: 建立性能、编译开销守门并完成系统验收

#### 29.1 本轮实施契约

本 Task 分为四个可单独验收的单元；下方原 Steps 是目录级清单，本节明确计量和执行边界。

**29-A 统计与 CPU benchmark。** 修改 `LayoutInfer.cpp`、`LayoutCandidateSolver.cpp`、`LayoutSolver.h`、`Passes.td` 及具名 pipeline 的选项传递；新增 `LayoutStatistics.h/.cpp`、`LayoutStatisticsTest.cpp` 和 `test/Transforms/layout-statistics.mlir`。统计以同一输入的一次尝试/最终计划为边界，重试有独立 attempt 记录，不把原始候选、重试提案和选中域混加。公开以下无歧义字段：

```cpp
struct LayoutStatistics {
  uint64_t variables = 0, constraints = 0, components = 0;
  uint64_t preparedCandidates = 0, prunedCandidates = 0;
  uint64_t peakVariableDomain = 0;
  uint64_t peakComponentCombinations = 0;
  bool componentProductOverflow = false;
  uint64_t peakFrontierStates = 0, expandedStates = 0;
  uint64_t conversionUseEdges = 0, uniqueConversions = 0;
  uint64_t rematerializationUseEdges = 0, uniqueRematerializations = 0;
  uint64_t conversionBytes = 0;
  uint64_t estimatedBankConflicts = 0, estimatedMemoryTransactions = 0;
  std::chrono::nanoseconds solveTime{0};
};
```

`preparedCandidates` 为该 attempt 搜索前去重后的域总量，`prunedCandidates` 为该 attempt 准备快照到搜索域的实际删减；原始生成提案数、各阶段传播/搜索/物化/优化/验证耗时另记录在 JSON 的 preparation/attempts/timings 对象。组件组合数是域乘积而非内存中的候选计数，溢出使用 flag/字符串或空数值表达，不回绕。frontier 统计实际保留的 frontier，MRV probes 和扩展节点单独计数。

转换 use 边数与实际唯一转换数分开；共享转换/共享重计算按经核验的 leader 去重计费，不将两个 use 当成两次物理搬运。失败 JSON 含结构化 failure kind 和稳定变量/约束身份，不捏造成功解、零成本或完成比例。

输出选项为 infer pass 的 `stats-file`，示例 `--frisk-infer-layouts='stats-file=/tmp/frisk-m6-stats.json'`；完整 pipeline 显式传递同一选项。默认不写文件；完整 pipeline 的统计区分已求解、已物化和已提交，不能在后续失败时留下“最终成功”记录。临时文件写完再替换指定输出，写入失败是可诊断错误，不影响原 IR；不新增未经用户选择的隐式落盘路径。

CPU benchmark 使用含 10/50/100/250 个实际布局相关操作的固定 IR corpus，包含小组件和真实相连的 producer/use 图；不把 250 个孤立常量当成唯一大图样例。分别测收集、候选准备、求解与完整布局 pipeline，预热 5 次、测量 30 次，报告 median/p95、稳定解 hash、组合数、frontier 和扩展数。打印/JSON 序列化在计时区间外，生产默认搜索配置不为跑分改成单候选或绕过证明。

**29-B 最小可执行 conversion adapter。** 修改 `lib/Conversion/FriskLayoutToGPU/TestLayoutConversionLowering.cpp` 及必要的相邻 helper，补真实 tile load/store 到每线程 vector 的 ABI；新增 `test/Conversion/layout-conversion-executable.mlir`、adapter 回归单测和 benchmark 生成入口。生成入口保存原始 Frisk IR、adapter 结果、lower 后 IR 与最终 PTX/cubin 的 hash、生成命令和编译 flags。它可以复用现有 `planRedistribution`，但 inferred 路径必须运行实际 adapter 生成逻辑，不能只调用 planner 后手写另一套 kernel。

最小首批 runtime case 均为静态单 CTA F32，分别覆盖 identity、32-thread warp shuffle、至少 64-thread 的跨 warp Shared exchange；补充已有 payload transport 支持类型的数值/位模式回归，但不因此承诺这些类型都有完整性能 baseline。适配器输入只能是已验证布局/容量/线程拓扑，最终 GPU 编译产物不得残留 Tensor/vector `UnrealizedConversionCastOp`、未解析 Frisk 布局或占位调用。无完整 lowering 的 MMA/TMA 不进入该测试生成入口。

参考 kernel 必须独立于 inferred adapter 的源/目的查找表，按明确坐标定义实现并由相同语义校验器比较。使用非全零、非全同值的输入区分元素，验证多组固定 seeds、边界值及合法复制；不能用常量数据掩盖错误映射。Shared scratch 正确计量与对齐、唯一 writer、uniform barrier 由静态检查加 GPU memcheck/racecheck/synccheck 联合验证。若 adapter 因动态 lookup 导致慢于 reference，应定位并优化该受限 adapter，而非将 reference 改慢。

**29-C runner、基线与完整编译。** 新增 `benchmark/layout/run_layout_benchmarks.py` 和独立 Python runner 测试。runner 接受 solver 和 conversion 工具输出，验证 schema、case manifest、环境、成功状态、非空样本、正的有限时间值、重复 case、产物 provenance。每个适用 case 单独计算：

```python
runtime_ok = inferred_median_ns / reference_median_ns <= 1.03
compile_ok = layout_pipeline_ns / full_compile_ns <= 0.10
```

runtime 逐 case 判定，不用所有 kernel 的平均值掩盖某项回退。布局计时包含收集/准备/传播/搜索/重试/物化/转换优化/布局验证；完整编译从同一 Frisk 输入进入编译驱动开始，到其 GPU 产物生成完成结束。两者属于相同编译样本，工具预先构建；禁止在分母加入依赖安装、reference 编译、故意等待、GPU 执行或无关工作。默认使用无产物缓存命中的独立编译，若使用缓存，另建有明确标签的实验，不能混算。

`--cpu-only --check` 模式报告局部 Gate，不要求 GPU baseline；非 CPU-only 的 `--check` 模式要求全部预先列明的 GPU case、产物、安全与编译证据，缺项返回未完成而不是默认通过。runner 测试通过固定的人工 JSON 输入覆盖恰好 1.03/0.10、超过阈值、环境不匹配、缺失基线、零时间、NaN/Infinity、case 缺失和推断/参考来源混淆；这些测试数据不得进入真实 baseline。

`sm90_layout_baseline.json` 仅在 reference 实测及评审后生成/更新，不预置虚构的 runtime 数字；先保存待审原始结果。GPU 选择使用明确设备，记录 UUID/时钟策略及后台负载，参考/推断交替测量降低漂移；不通过自动调时钟或杀进程获取“稳定环境”。如果 H100 忙碌或权限受限，记录具体限制，不把跨环境数据拿来凑门槛。

**29-D 完整 Gate 与长期记录。** 新增 `guoqiao/layout_inference_acceptance.md`；逐个 Global Constraint 映射测试/命令、源码与输入 digest、结果和未覆盖范围。全量 unit/lit/CTest/Python、生产 legacy API 隔离必须重新运行。同步设计/比较文档时，仅报告真实测量支持的结论，不将固定 TileLang 语义对齐解释为比 TileLang/Triton 更快。任何安全、3% 或适用域 10% 失败都保留 FAIL，M6 不标完成。

局部执行顺序：统计/runner 红灯 → 最小实现与 CPU 局部绿灯 → adapter 无桥接及位模式回归 → 生成真实 GPU 路径 → 输出/安全检查 → reference/inferred 性能与完整编译 → 独立复审 → 全量 Gate。下方命令中的 Python 均使用项目对应环境；不能在缺依赖时静默改用系统中的另一个 TileLang。

**Files:**

- Modify: `CMakeLists.txt`
- Create: `include/Dialect/Frisk/Analysis/LayoutStatistics.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutStatistics.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/PassPipelines.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Conversion/FriskLayoutToGPU/TestLayoutConversionLowering.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutStatisticsTest.cpp`
- Create: `test/Transforms/layout-statistics.mlir`
- Create: `test/Conversion/layout-conversion-executable.mlir`
- Create: `test/python/layout_benchmark_runner_test.py`
- Create: `benchmark/CMakeLists.txt`
- Create: `benchmark/layout/CMakeLists.txt`
- Create: `benchmark/layout/solver/CMakeLists.txt`
- Create: `benchmark/layout/solver/layout_solver_bench.cpp`
- Create: `benchmark/layout/conversion/CMakeLists.txt`
- Create: `benchmark/layout/conversion/generate_conversion_cases.cpp`
- Create: `benchmark/layout/conversion/compile_conversion_case.py`
- Create: `benchmark/layout/conversion/layout_conversion_bench.cu`
- Create: `benchmark/layout/run_layout_benchmarks.py`
- Create: `benchmark/layout/baseline/sm90_layout_baseline.json`
- Create: `guoqiao/layout_inference_acceptance.md`
- Modify: `guoqiao/layout_inference_design.md` if measured limits require an approved change

**Interfaces:**

- Produces：

```cpp
// 采用 §29.1 的 LayoutStatistics。
// prepared/pruned/domain/combinations/frontier/expanded 分开记录；
// unique conversion/rematerialization 与 use edge 数分开记录。
```

Infer pass option `--frisk-infer-layouts='stats-file=<path>'` 输出稳定 JSON；完整 pipeline 显式传递，默认不写文件。除 `timings` 等明确测量字段外，相同输入输出应字节一致。

- [x] **Step 1: 写 statistics 红灯测试**

用固定小图检查 vars/constraints/candidates/conversions 数量和稳定 JSON key；连续两次运行除时间字段外完全一致。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，statistics option 尚不存在。

- [x] **Step 2: 实现统计和 solver microbenchmark**

`layout_solver_bench` 使用固定真实 IR corpus，分别测 10、50、100、250 个布局相关 Op；预热 5 次、测量 30 次，报告 median/p95、单域/组合/frontier 峰值及扩展数和解 hash。不得在计时区间打印 IR；不以孤立常量掩盖相连图开销，见 §29.1。

- [x] **Step 3: 实现可选 CUDA conversion microbenchmark**

CMake option：

```cmake
option(FRISK_ENABLE_CUDA_BENCHMARKS
       "Build SM90 layout conversion benchmarks" OFF)
```

启用时要求 `nvcc` 和 SM90；至少比较 identity、单 warp shuffle、shared exchange 三类 conversion。每类 case 同时提供人工编写的 `reference_*` kernel 和通过 `TestLayoutConversionLowering.cpp` 生成/调用的 `inferred_*` 路径；两者使用同一输入和校验器。每个 case 校验输出，再预热 100 次、测量 1000 次；记录 GPU、driver、clock policy 和编译 flags。

顶层仅在 `FRISK_ENABLE_CUDA_BENCHMARKS=ON` 时执行 `enable_language(CUDA)` 并加入 conversion 子目录；默认 CPU/CI 构建不得依赖 CUDA toolkit。

- [x] **Step 4: 实现 benchmark runner 和 baseline 比较（22 项 Python 测试、独立复审通过；实际 GPU 基线仍待 Step 6）**

Runner 读取/写入：

```json
{
  "environment": {},
  "solver": {},
  "conversions": {},
  "supported_kernels": {}
}
```

baseline JSON 中 conversion/runtime 数值必须来自同一固定环境下的 `reference_*` kernel，不能把新 solver 首次运行结果自封为基线。正确性失败立即退出；已有 lowering 的 `inferred_*` 对相应 `reference_*` median 回退超过 3% 失败；适用用例的布局 pipeline 占真实完整编译时间超过 10% 失败。候选组合超过 256 仅触发 beam32，不自动算失败；验证实际搜索模式、frontier 与扩展预算，预算耗尽不能冒充正确性通过。时间噪声在 3% 内视为持平，环境不匹配则不可比较。

本计划的 runtime 守门范围只覆盖 conversion adapter 和仓库中已经存在、能走新布局 pipeline 的 kernel；尚无 WGMMA/TMA lowering 的路径只做静态 contract/cost 验证，不伪造 runtime 数据。

- [x] **Step 5: 运行 CPU/solver 验收**

Run:

```bash
cmake --build build --target frisk_layout_solver_bench check-frisk FriskLayoutUnitTests --parallel 32
python benchmark/layout/run_layout_benchmarks.py \
  --solver build/benchmark/layout/solver/frisk_layout_solver_bench \
  --cpu-only --check
```

Expected: CPU correctness、确定性及搜索模式/预算 Gate PASS，输出阶段耗时；本命令本身不提供 GPU 安全/runtime 或完整编译占比通过证据。

- [x] **Step 6: 在 SM90 环境运行 conversion/runtime 验收**

Run:

```bash
cmake -S . -B build-m6-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir \
  -DFRISK_ENABLE_CUDA_BENCHMARKS=ON
cmake --build build-m6-release --target frisk_layout_conversion_bench --parallel 12
python benchmark/layout/run_layout_benchmarks.py \
  --conversion build-m6-release/benchmark/layout/conversion/frisk_layout_conversion_bench \
  --record-reference \
  --output-baseline /tmp/frisk-sm90-layout-reference-pending.json
python benchmark/layout/run_layout_benchmarks.py \
  --conversion build-m6-release/benchmark/layout/conversion/frisk_layout_conversion_bench \
  --baseline benchmark/layout/baseline/sm90_layout_baseline.json \
  --check
```

Expected: 第一条 runner 命令仅在尚不存在的独立路径保存 pending reference，不会生成 accepted baseline，也不能覆盖已存在文件。须先独立核验参考来源、环境、安全、原始样本和内容摘要，再保持实测数据不变、将评审状态设为 accepted 并保存到仓库基线文件；第二条命令才能正式判定 runtime Gate。2026-10-10 的实际接受记录及摘要见验收文档。没有 SM90 机器或未完成性能评审时不得将本步骤标记完成。

- [x] **Step 7: 执行完整最终验证**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target FriskIR FriskTransforms frisk-opt \
  FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
git diff --check
```

Expected: 所有构建、CTest、lit、unit tests 和 diff check PASS。

- [x] **Step 8: 编写验收记录并同步设计**

`layout_inference_acceptance.md` 必须列出每个 Global Constraint 的证据命令、结果、未覆盖项和后续 WGMMA/TMA lowering 边界。若实测要求改变阈值，必须先获得架构评审再修改设计文档，不能为让 benchmark 通过而直接放宽。

- [ ] **Step 9: 提交性能守门和验收记录（本轮不自动执行，等待另行授权）**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis \
  lib/Dialect/Frisk/Transforms benchmark guoqiao/layout_inference_acceptance.md \
  guoqiao/layout_inference_design.md
git commit -m "test: gate layout correctness and performance"
```

#### 29.2 历史实施检查点（2026-10-10，最终结果见 §29.3）

以下保留当时的失败和阶段性判断，不表示当前仍有这些未完成项；后续结果不覆盖或删除早期证据。

- runner 已完成 schema/逐 case 阈值/来源一致性/原始证据保留及基线保护，22 项 Python 测试与独立复审通过。新参考数据只能写到独立的 pending 文件，不能自动覆盖 accepted baseline；汇总输出也不能与基线路径或其别名重合。
- GPU adapter/驱动代码的局部复审已通过：显式拒绝非 Global 地址空间，补充 4 项 adapter 单元测试、6 项存储契约入口提升测试和 4 项编译驱动测试。首次性能数据之后新增了测量窗口前后负载快照与构建清单绑定；最终基线须使用修订后工具重新运行和评审，不能直接把代码复审通过等同于基线接受。
- 统计与 CPU benchmark 已实现并通过局部独立复审，最新 12 项统计测试通过。修复覆盖失败分类、外部收集器生命周期、最终冲突身份，以及未排序调用者图与已排序求解图之间的错误身份映射；失败时不修改调用者图。早先一次“10 项通过”的记录使用了加强断言前的二进制；加强后的测试曾因预期 `EmptyDomain` 与实际 `HardConflict` 不符而失败，现分别测试真实 coordinator 冲突身份，以及最终失败记录不覆盖 attempt 历史，未为适配测试改变生产分类。
- 第一份 Debug CPU 探索结果已完整生成，10/50/100/250 个相连操作 corpus 均有 30 个测量样本和稳定结果 hash；2026-10-10 用 runner 重放检查得到 `PARTIAL_PASS`。原始数据 `/tmp/frisk-task29-solver.json`，重放结果 `/tmp/frisk-m6-task29-exploratory-cpu-gate.json`。这是修复前二进制的探索证据，不能替代最终版本的性能验收，也不能用于 10% 完整编译占比结论。
- 显式 executable adapter 已将真实 tile load/store 接入每线程传输，11 个 identity/shuffle/Shared、payload 与复制用例通过真实 cubin 执行，以及 memcheck/racecheck/synccheck。首轮原始数据 `/tmp/frisk-task29-conversion-gpu1-escalated.json`，逐次产物位于 `build-m6-release/benchmark/layout/conversion/measurements/run-l2h8x_r0/`。三个必测用例的同轮 inferred/reference median 比值分别为 1.000、0.99206、1.000；参考数据尚待独立复审，不能据此宣称正式 runtime Gate 已验收。
- 首轮三个必测用例的布局 pipeline / 完整编译占比分别为 **12.14%、12.05%、20.24%，均 FAIL**；实际 runner 结果 `/tmp/frisk-task29-gpu-first-gate.json` 保留失败。继续定位重复证明开销，不改变 10% 门槛，不增加无关分母工作。尚无接受的 GPU baseline，最终全量回归和验收记录仍待完成，M6 保持未完成。
- 仿射求值局部优化后的第一轮无并行构建干扰 compile-only 数据 `/tmp/frisk-task29-conversion-optimized1.json` 为 **4.11%、3.86%、7.25%**；布局阶段分别 3.28/3.45/6.30 ms。优化消除逐点属性构造，保留证明与枚举预算；算术边界回归、最终二进制 GPU 重测和全量验收仍在进行，因此该轮不标记最终 Gate 通过。

#### 29.3 最终验收结果（2026-10-10）

- 统计/solver benchmark、runner、可执行 adapter/编译驱动及仿射求值优化均通过独立代码与契约复审。最终完整构建成功，**405/405 unit、58/58 lit、7/7 Debug CTest** 通过；Release 的入口提升 CTest 1/1（内部 6 项）通过，Python 规范化 5/5、旧 API 生产隔离和 SM90 工具 SS/RS/输出存活/拒绝检查通过。完整单元分片为 102+101+101+101，无失败、错误或 disabled 用例。
- 首轮编译占比失败定位为有限仿射坐标求值反复构造属性的开销。改成带溢出检查的直接整数求值，未删证明、未缩减枚举/搜索预算、未改变分母工作。7 项算术边界测试通过；三个必测用例优化前后的输入、布局 IR、adapter IR、PTX/cubin 逐字节一致。Shared 后端包装中的一次计时元数据差异单独记录，不被解释为生成代码变化。
- 最终真实 H100 运行：11 个正确性/位模式/复制用例、33 次 memcheck/racecheck/synccheck 通过。三个必测用例的 inferred/reference median 比值为 **1.007874、1.000000、1.008333**，布局 pipeline / 同样本完整编译为 **3.798968%、3.973393%、6.667194%**。两次另外的无缓存完整编译重复均通过，最大占比 7.165%。阈值仍为 3% 和 10%。
- 独立性能评审接受最终 reference 的原始数据；`benchmark/layout/baseline/sm90_layout_baseline.json` 仅改变 pending→accepted 状态，不修改样本。使用最终 CPU/GPU JSON 及此基线重放 runner，solver/runtime/compile/overall 全部 PASS。CPU 10/50/100/250 个实际操作的最终独立测量、原始样本及阶段 median/p95 均保留，生产配置保持 exact256/beam32/65536。
- 逐 Global Constraint 的命令、来源/输入/产物摘要、测试结果、历史 FAIL 和未交付范围见 `guoqiao/layout_inference_acceptance.md`。整体源码及最终文档复审通过，无 Critical/Important 问题；两项非阻塞建议保留记录：更多有符号算术边界测试、导出器预期拒绝测试的 stderr 降噪。
- 完整 WGMMA/TMA/mbarrier 指令 lowering 仍不在本轮范围；没有声称 Frisk 比 TileLang/Triton 更快。源目录 main 直接实施，未 commit/push，未创建 worktree，原用户修改保留。

### M6 Gate

必须具备以下证据：

```text
CTest/lit/unit/property 全部通过
TileLang 固定 corpus 真实导出且可重现；equivalent 与 expected-rejection 分别通过
相同输入重复编译产生相同布局 IR
race/OOB/invalid ownership/unresolved var 为 0
exactCombinationLimit = 256；超限使用 beam width = 32，不截断候选到 256
每组件每次尝试扩展 <= 65536；域/组合/frontier/扩展分别统计
适用用例布局 pipeline / 真实完整编译时间 <= 10%；无完整路径只报告阶段耗时
受支持 SM90 runtime case 中位回退 <= 3%
尚无 lowering 的 WGMMA/TMA 路径仅声明静态契约通过
```

任一必需项缺少证据时，布局系统不能标记完成。必须包含实际 adapter 的 identity/shuffle/Shared GPU 正确性、安全和 runtime 证据；不能以 OUT_OF_SCOPE 或 CPU-only PASS 代替这三个必需 case。

---

## 3. 最终 Definition of Done

以下完成状态限定于本文已声明的支持域及验收矩阵，不代表任意 kernel 或完整 WGMMA/TMA lowering 已获验证。

- [x] M0–M6 的全部 Gate 均有最新命令输出。
- [x] `frisk-layout-pipeline` 对所有受支持 Frisk kernel 生成合法、已解析、可验证的布局 IR。
- [x] 生产源码中不存在 `inferLayout(builder, DenseMap<Value, Attribute>&)`。
- [x] Local/Register 的生产 IR 不再以 local MemRef 作为求解载体。
- [x] Distributed/Storage layout 映射方向、Attr 字段和 verifier 与设计文档一致。
- [x] Hard conflict 输出 provenance chain 和坐标反例。
- [x] 多 consumer 可以选择共同布局、conversion 或 rematerialization。
- [x] conversion 是普通成本候选，且不会被零转换绝对优先规则压制。
- [x] SM90 WGMMA/TMA/shared swizzle contracts 有正反例测试。
- [x] TileLang 固定语义 corpus、property tests 和顺序确定性测试通过。
- [x] CPU/solver 与 SM90 conversion/runtime 性能守门通过。
- [x] `guoqiao/layout_inference_design.md`、对比文档、实施计划和验收记录不存在语义冲突。
- [x] 完整 WGMMA/TMA lowering 的未交付边界在验收记录中明确保留。

## 4. 计划执行方式

### 后续方案 A：审计修复与单 tile Attention（2026-10-10 已完成）

目标与完整语义以设计文档开头的“后续增量契约”为准。技术栈继续使用 C++/MLIR、GF(2) 布局代数、GoogleTest/lit；不引入 TileLang/TVM 生产依赖。

全局约束：SM90a、128 线程；Q/K/V=`64×16xf16`、scores=`64×64xf32`、O=`64×16xf32`；SS QKᵀ、稳定 f32 Softmax、cast_f16、RS PV；exact256/beam32/65536 不变。无 causal/dropout/动态尾块/通用 reshape/完整 Attention GPU lowering；不绕过实际布局 verifier。源目录 main，保留构建产物，不自动 commit/push。

- [x] A0：先写设计契约、实施步骤及比较口径。
- [x] A1：两处 Storage 修复及独立复审。
  - 在 `unittests/Dialect/Frisk/Layout/LegacyNormalizationTest.cpp`、`LayoutAliasAnalysisTest.cpp` 及 `test/Transforms/storage-audit-local-global-packed.mlir` 增加跨函数/分支 Global 与 packed BitLinear/affine 等价反例；旧代码两项单测及 lit 失败，修复后 30 项局部 unit、8 项 lit、根代理完整 pipeline 连续重跑通过，独立 spec/quality 复审通过。
  - 修改 `lib/Dialect/Frisk/Transforms/PreserveStorageContracts.cpp` 与 `lib/Dialect/Frisk/Analysis/LayoutAliasAnalysis.cpp`：分离 canonical identity 与就地 SSA 根，允许零输出位宽常量，不放宽其他边界。
  - 构建 `FriskLayoutUnitTests frisk-opt`，运行对应 gtest filter 和新增 lit；复审 spec/quality 后进入 A2。
- [x] A2：广播关系及独立复审。
  - 新建 `BroadcastLayoutConstraints.h/.cpp` 封装广播坐标证明/投影，接入 constraint kind、collector、冻结候选准备、传播、solver 和实际 IR 验证；具体接口以代码声明为准，禁止伪造 SameLayout/transpose。
  - 新建 `BroadcastLayoutTest.cpp` 与 `test/Transforms/infer-broadcast-layout.mlir`：先证明未实现失败，再验证正反坐标关系、不同广播轴、硬绑定冲突、源端转换、重跑、rollback。
  - 实际新增 8 项广播单测；81 项局部 unit、4 项 lit、根代理重跑 8 项广播单测通过，独立 spec/quality 复审通过。涵盖 4×8/32 线程和 8×16/128 线程真实 load→稳定 Softmax→store、坐标 oracle、实际 verifier 零候选生成、冲突和回滚。广播规则无 Attention 特定形状例外。
- [x] A3：完整 Attention 集成及最终复审。
  - 新建 `test/Transforms/attention-layout.mlir`，真实 Global Q/K/V→三次 Copy→SS MMA→scale/max/broadcast/sub/exp/sum/broadcast/div→truncf→RS MMA→Global Store；输入不预填 Tensor encoding。
  - 新增 C++ 集成测试检查完整数据流、契约、实际 IR 验证、连续重跑与 parse 重放、失败原子回滚；修复只限此链暴露的通用规则集成问题。
  - 执行 `cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12`、全量 unit、`llvm-lit -sv build/test`、`ctest --test-dir build --output-on-failure`、三个已有 Python normalization/API/tool 检查；逐项记录结果，更新维护文档。

复现完整输入（本例只验收布局推断，不执行 GPU）：

```bash
build/bin/frisk-opt test/Transforms/attention-layout.mlir \
  --frisk-layout-pipeline -verify-each -o /tmp/frisk-attention.mlir
```

`AttentionLayoutIntegrationTest.cpp` 与 lit 共用该输入，逐边检查 Q/K/V Copy、SS/RS MMA、稳定 Softmax、truncf、真实 O Store；测试不会用常量替换中间计算。A3 首次集成在 A1/A2 后已通过，因此只补充验收测试，没有宣称额外“先红后绿”的生产修复。

最终验收记录：

- Debug 显式构建 `FriskLayoutUnitTests frisk-opt` 后完成全量构建，均 exit 0。全量 unit 分 4 片执行，105+104+104+104 = **417/417**，XML 汇总 failures/errors/disabled 均为 0；包含新增 2 项 Storage、8 项 Broadcast、2 项 Attention 测试。
- 全量 lit **61/61**；Debug CTest **7/7**；Release 的 `FriskLayoutConversionGeneratorTest` **1/1**。不将 Release 的未构建其他 CTest targets 算作已验收。
- Python normalization **5/5**，生产 legacy API 隔离、SM90 SS/RS 工具输出与目标拒绝检查通过。
- A1/A2/A3 各自独立 spec/quality 复审及整体源码复审通过，无 Critical/Important 问题。保留两项非阻塞说明：Attention 测试个别 typed attribute 缺失时诊断可更友好；Release 重编既有测试时出现 APInt 编译警告，Debug Ninja 日志有既有恢复提示，不宣称构建零警告。
- 输入/输出布局真实物化、原有完整算子链回归、实际 verifier、重跑/文本重放、失败原子回滚通过。不包括 Attention GPU 数值/性能、完整 WGMMA/TMA lowering 或 M6 微基准性能重新认证。本轮没有 commit/push。

可重复的全量验证命令（分片可分别并行执行）：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12
cmake --build build --parallel 12
for shard in 0 1 2 3; do
  GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=$shard \
    build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
    --gtest_output=xml:/tmp/frisk-attention-final-unit-$shard.xml
done
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py -sv -j 4 build/test
ctest --test-dir build --output-on-failure
ctest --test-dir build-m6-release -R FriskLayoutConversionGeneratorTest --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/sm90_layout_tool_test.py --build-dir build
```

本轮临时证据为 `/tmp/frisk-attention-final-{build,lit,ctest,release-ctest,python}.log` 和 `/tmp/frisk-attention-final-unit-{0,1,2,3}.{log,xml}`；长期依据为上述仓库测试、命令和结果记录。三份维护文档先写契约，再同步实际实现边界与结果。

以上增量采用测试驱动与子代理逐任务独立复审。下面 Task1–29 顺序描述保留为历史里程碑执行约定。

严格按 Task 1 → Task 29 顺序执行。每完成一个 Task：

1. 运行该任务局部测试；
2. 运行所在里程碑已有回归；
3. 检查设计文档漂移；
4. 用户授权后提交独立 commit（本轮按源目录工作约定不自动 commit/push）；
5. 进行一次 spec compliance review 和一次 code quality review；
6. 只有 review 通过才进入下一个 Task。

建议在每个 M0–M6 Gate 后由用户进行一次阶段验收。M2、M3、M4、M5 的 Gate 是架构关口，不建议并行跨越。
