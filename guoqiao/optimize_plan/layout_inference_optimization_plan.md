> 2026-10-29 P1.4a：最小形状支持已完整验收，功能回归、独立源码及最终证据/文档复审均通过。静态正二次幂单位轴、插入轴Broadcast、rank≥2到rank≥1 Reduce及静态expand/collapse/transpose闭环；575 Release主单测、20 selection，双配置最终15聚焦/5lit及辅助守门通过。GPU基础后端及后续路线未完成。见[实施报告](../inference_detail/p1_4a_implementation_report.md)。

# Frisk 布局推断系统优化清单与推进建议

> 2026-10-29：P1.3c已完成验收，完整源码、驱动、数据与文档独立复审通过。默认legacy保留；显式`boundary-frequency=static-trip-count`统一转换/clone与受限外提的频率口径。Release主单测570、选择质量20，相关Debug/lit及辅助守门通过；18次语义核验与36次串行CPU调用通过，身份复验不变。下一独立任务为最小形状支持，GPU和跨框架比较尚未实施。见[阶段报告](../inference_detail/p1_3c_implementation_report.md)。

> 2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。 见[完整Attention资源报告](../inference_detail/p1_3b_attention_resource_report.md)。

> 历史检查点（2026-10-27 P1.3b阶段进展，已由2026-10-28结果取代）：实际SSA/Shared root存活诊断与默认关闭的pipeline接入已通过独立复审，Debug/Release定向各87项、lit各4项和全Release545项通过。仅补充观测，不更改可加成本目标。完整Attention资源验收仍待Broadcast隐式region窄特例确认；O06全目标、P1.3b整体与P1.3c均不能标为完成。见[阶段报告](../inference_detail/p1_3b_implementation_report.md)。

> 历史检查点（2026-10-26 P1.3a；后续资源诊断进展见2026-10-28记录）：模型身份/九项覆盖与实际IR算术诊断、回归/串行观测已完成，最终独立证据复审通过。诊断与搜索评分分开；两类Attention各操作4096逻辑element-op，开启后布局/成本不变，CPU中位差异很小且不作显著性判断。见[报告](../inference_detail/p1_3a_implementation_report.md)。这只完成O05的可解释性基础，不完成时延校准、O06活跃峰值或O10静态循环频率；继续P1.3b→c，P1.4仍未实施。

> 2026-10-26：方案A选择质量实验已完成实现、测试/采集和独立最终复审，见[报告](../inference_detail/p1_2_selection_quality_report.md)。消费者36图中出现3个严格反转、6个平局消除；原生Store的Auto与FullRow成本全部打平，而C1/C2的Auto编译耗时约为FullRow的2.25/2.24倍。新增的是有限域可核验选择质量证据，不是新的权重/候选、GPU收益或全域最优保证；C6 RS joint oracle超时明确保留。继续按已批准P1.3a→b→c实施。下方历史状态不覆盖本条更新。

> 2026-10-24：P1.2a→P1.2b 已按 [执行计划](p1_2_execution_plan.md)完成，见 [实施报告](p1_2_implementation_report.md)。交付 O08 的固定 ABI Copy→MMA 集成/细分诊断、O07 的可关闭热点/缓存观测、O04 的显式 Auto/有限 grid 联合选择；505 unit / 69 lit 与辅助守门及独立源码复审通过，54 次 CPU 采样调用与 9 次 trace 核对完成。不是 O04/O07/O08 的全部长期目标：全 atom/consumer 赢家反转/自动 tile/新增缓存/GPU 执行仍未交付。P1.3/P1.4 未开始，下方原始审计快照保留。

> 2026-10-24：按用户选择的方案 A 完成剩余 P1 的只读实现审计，形成[P1.2–P1.4 阶段契约提案](p1_2_to_p1_4_design_proposal.md)，具体设计待确认。P1.2 对应 O04/O08 的受限增量及 O07 观测；P1.3 对应 O05/O06 的诊断基础和 O10 静态频率；P1.4 对应 O09 单位轴、O11 执行闭环及 O05/O06 校准证据。各项未纳入部分明确保留，不宣称整个 P1 已完成；本轮未改生产代码、构建或测试。

> 2026-10-21：P1 方案 A 的 P1.1（O07 受限符号证明与开销验收）已完成。ownership/Broadcast 符号证明、差分测试、完整回归与独立复审通过；部分超界关系可证明，证明微基准降低开销，两类 Attention 全链时间基本持平且 IR/stats/trace 不变。见[具体设计](p1_1_implementation_design.md)与[实测报告](p1_1_implementation_report.md)。O07 的其他热点/缓存工作及 O04/O05/O06/O08/O09/O10/O11 尚未在本轮实施。以下原始审查快照保留其历史语境。

> 2026-10-14 更新：O01/O03/O02 已按获批的 [P0 方案 A](p0_implementation_design.md)完成实现与验收，结果、边界与开销见 [实施报告](p0_implementation_report.md)。下文保留 2026-10-13 的审查快照；其中 P0 的“现状/待补强”是实施前情况，不代表当前仍未实现。P1/P2 本轮未实施。

审查日期：2026-10-13。依据：当前工作区源码（包含尚未提交的 Attention 增量）、已有测试和推断报告。

本文是后续工作的建议清单，不是已批准的实现契约；不改变 Task 1–29 及 Attention 增量的既有完成状态。本轮仅审查并编写文档，没有重新执行编译、单测或 GPU 测试。下述数字引用已有推断记录。

## 1. 当前系统到了哪一步

当前系统已经能在给定 shape、dtype、线程数和硬约束的前提下，联合选择 Tensor 线程分布、Storage 布局、Copy/MMA/Reduce 契约，并处理布局转换和有限重计算。它已有坐标证明、别名一致性检查、region 约束、有限候选传播、有界搜索、失败诊断、实际 IR 验证及回滚机制。

已有完整的静态单 tile Attention **布局推断**用例：

```text
Global Q/K/V → Copy 到 Shared → SS MMA(QKᵀ)
            → scale → ReduceMax → Broadcast → Sub → Exp
            → ReduceSum → Broadcast → Div → cast_f16
            → RS MMA(PV) → Global O
```

但需要区分三个层次：

| 层次                           | 当前证据与边界                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| 能否产生合法布局 IR            | 上述 Attention 链已有集成验收；并不代表任意 shape、算子和控制流组合都支持                   |
| 能否选到较好的布局             | 有代价模型和有界搜索，但已有漏掉更低估算代价方案的记录，不保证最优                          |
| 能否生成并运行高性能 Attention | 完整 Attention 的 GPU lowering、数值和性能闭环尚未交付；已有 GPU 验收主要覆盖布局转换微基准 |

因此，下面既有需要改进的现有能力，也有需要新增的能力，不能全部称为正确性缺陷。

主要依据：[Attention 流水线报告](../summary/layout_inference_pipeline_report.md)、[实施计划及验收边界](../layout_inference_implementation_plan.md)、[Attention 输入 IR](../../test/Transforms/attention-layout.mlir)。

## 2. 优化项总览

P0 表示建议下一轮优先处理；P1 表示在此基础上分批增强；P2 表示涉及调度、后端或外层搜索的能力扩展。优先级不等于已发现缺陷的严重等级。

| 编号 | 优先级 | 优化点                          | 性质               | 预期结果                                |
| ---- | ------ | ------------------------------- | ------------------ | --------------------------------------- |
| O01  | P0     | 固定真实输入输出内存契约        | 集成验收补强       | 普通外部矩阵不必迁就推断出的布局        |
| O02  | P0     | 改进有界搜索的选择质量          | 已观察到的优化机会 | 在受控预算内减少漏掉好方案的情况        |
| O03  | P0     | 输出可复核的决策记录            | 工具能力增强       | 解释为何选、为何拒绝、代价来自哪里      |
| O04  | P1     | 优化候选生成与 MMA 几何选择     | 候选覆盖增强       | 不只在少量预设方案中比较                |
| O05  | P1     | 完善工作量、访存与异步路径代价  | 模型覆盖不足       | 模型排序更接近实际执行                  |
| O06  | P1     | 按存活区间估计寄存器和 Shared   | 资源估计偏粗       | 区分累加槽位与真实峰值资源              |
| O07  | P1     | 降低证明和求解开销              | 可扩展性增强       | 扩大可证明范围，控制编译耗时            |
| O08  | P1     | 完善 Copy 路径的契约与集成覆盖  | 规则及诊断增强     | 在条件满足时可靠使用向量化/cp.async/TMA |
| O09  | P1     | 扩大 shape、广播和视图支持范围  | 新支持域           | 从规则小 tile 走向更常见的算子输入      |
| O10  | P1     | 改进跨循环和多消费者转换决策    | 现有能力增强       | 避免重复转换与不划算的重计算            |
| O11  | P1     | 建立 Attention GPU 执行闭环     | 后端及验证扩展     | 证明推断结果能正确执行并产生性能收益    |
| O12  | P2     | 自动选择 tile、线程数和流水级数 | 外层配置搜索       | 回答“怎样分块计算更快”                |
| O13  | P2     | 支持线程角色分工和流水重叠      | 调度能力扩展       | 不同操作可使用不同线程子集              |
| O14  | P2     | 建立系统级对比与消融评测        | 研究与维护完善     | 用证据说明 Frisk 的独立价值             |

## 3. P0：建议优先推进的三项

### O01：固定真实输入输出内存契约

**现状。** 当前 Attention 输入未固定完整的 Global Storage 布局，也没有给出更强的 Global 根地址对齐保证。系统可以在允许的候选中为 Q/K/V/O 选择存储形式。这验证了联合推断，但尚不能替代“调用者按固定行主序提供矩阵”的接口验收。

**具体例子。** 外部提供 `Q[64,16] : f16`，要求行主序，则应保持：

```text
Q[i,j] 的字节地址 = Q_base + 2 × (16 × i + j)
```

推断可以改变 Shared 中 Q 的布局，也可以选择搬运线程分布；不能为了降低代价，将调用者已有的 Global Q 静默改成另一种地址映射。输出 O 同理。若确实需要预打包，必须显式执行并计入成本。

**建议。** 保留现有自由布局用例，新增固定行主序 Q/K/V/O 的端到端推断用例；分别覆盖未知对齐、调用者明确保证的对齐、带静态偏移的子视图。对齐是输入事实，不是优化器可凭空增强的属性。

**验收。** 推断前后外部逐坐标字节地址不变；内部布局仍可优化；硬绑定冲突必须诊断，不得覆盖用户契约。未知对齐用例保持保守处理。该项是在已有 Storage 根契约机制上补强集成，不是重建契约系统。

实现入口：[Storage 契约保存](../../lib/Dialect/Frisk/Transforms/PreserveStorageContracts.cpp)、[别名分析](../../lib/Dialect/Frisk/Analysis/LayoutAliasAnalysis.cpp)、[Attention 集成测试](../../unittests/Dialect/Frisk/Layout/AttentionLayoutIntegrationTest.cpp)。

### O02：改进有界搜索的选择质量

**现状。** 已有 Attention 记录中，传播后组合空间为 5184；默认精确枚举阈值为 256，beam 宽度为 32。搜索给出合法方案，但事后复算找到更低代价的合法替代：

```text
已选方案：             [8293, 608, 64, 0, 428, 6144, 6, 0, 0]
仅替换 Q Global 候选： [8293, 576, 64, 0, 428, 6144, 6, 0, 0]
```

第一项相同，第二项内存事务估算由 608 降至 576，因此按当前字典序，后者更好。这是搜索质量问题，不是已选方案非法，也不代表 576 是全局最优或 GPU 必然更快。该比较属于已有事后验证记录，并非本轮重新运行的 beam 轨迹。

**审查线索。** SM90 完整方案调用 `evaluateSM90LayoutCost`，部分方案的 `lowerBound` 调用 `evaluateStaticLayoutCost`。部分排序与最终目标覆盖不同，值得优先检查；尚不能仅凭这一点断言它就是漏选的唯一原因。

**建议。** 先将该反例固化为可重放测试，再记录较优路径在哪一步被丢弃。评估增加部分方案的已确定访存成本、联合支持表上的保守最小成本，以及有界局部改进等办法。用于剪枝的下界必须证明不会高估；若只是排序启发式，应与下界接口明确区分。

**验收。** 在同一候选集合、同一代价模型下比较新旧搜索；小图用精确枚举作参考。对该 Attention 例应找回已知更优方案或更好方案，同时报告探测次数、耗时和内存。第一轮保持默认 256/32/65536 预算，不以单纯扩大预算作为优化结论。

实现入口：[候选求解器](../../lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp)、[SM90 目标接口](../../lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp)、[求解器属性测试](../../unittests/Dialect/Frisk/Layout/LayoutSolverPropertyTest.cpp)。

### O03：输出可复核的决策记录

**现状。** 系统已有统计 JSON、候选拒绝信息、失败来源、阶段计时和搜索预算记录。缺口不是“没有日志”，而是缺少稳定、完整地串联候选变化、联合支持、搜索淘汰和逐项代价的汇报接口。

**具体例子。** 对一次 Copy，希望直接知道：TMA 被拒绝是因为 Global 对齐不足，还是地址不能由描述符表达；某个线程分布是坐标不合法，还是证明点数超过预算；最终 `memoryTransactions=608` 分别由哪些访问贡献。

**建议。** 增加可选、带版本号的决策 trace，记录稳定变量编号、候选来源、删减原因、契约支持、有限数量的备选方案以及按操作分解的九项代价。严格区分“证明不成立”“无法证明”“搜索预算耗尽”“合法但未被选择”。

**验收。** 无需临时 C++ 辅助程序即可生成 Attention 各阶段报告；分项成本可核对回总成本；相同输入的语义记录稳定。默认不输出全部搜索树，详细模式须限制体积并测量额外开销。

实现入口：[统计接口](../../include/Dialect/Frisk/Analysis/LayoutStatistics.h)、[候选求解器](../../lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp)、[统计测试](../../unittests/Dialect/Frisk/Layout/LayoutStatisticsTest.cpp)。

## 4. P1：增强覆盖、模型和可执行性

### O04：优化候选生成与 MMA 几何选择

**现状。** 默认 Distributed 候选有四类分配方式，但完整候选还包括显式绑定、MMA 等指令要求、关系投影和受控松弛，不能理解为系统总共只有四种布局。当前 MMA 几何主要依据形状及 policy 确定一个方案，再据此构造契约，并非对所有合法几何作成本比较。

**建议。** 根据访存连续性、Reduce 轴、广播关系和 consumer 指令角色生成少量高价值候选；对允许多个方案的 MMA，探索有限个 warpgroup 划分和指令 atom 组合，与输入/输出布局联合评估。显式 policy 的语义必须保留，不能把硬要求改成软偏好。

**例子。** 两种合法 MMA 划分可能使后续 Reduce 的跨线程通信不同；只看矩阵长宽的平衡程度，无法表达这一后续影响。

**验收。** 增加一个“局部 MMA 选择与整图较优选择不同”的可重放例子；报告候选数、支持表大小和求解开销。候选准备后仍冻结域，传播仍单调删减；去重不能误合并对齐、资源或指令语义不同的方案。

实现入口：[Distributed 候选](../../lib/Dialect/Frisk/Target/SM90/SM90DistributedCandidates.cpp)、[MMA 几何与证明](../../lib/Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.cpp)。

### O05：完善工作量、访存与异步路径代价

**现状。** 模型已有 Global sector、Shared bank、指令路径、转换等估算，但明确未覆盖延迟、缓存和执行重叠。当前 Attention 的工作量项没有逐项计入 Mul/Exp/Div；Shared 冲突项反映相关 StorageAccess，不能当作完整 MMA Shared 读取行为的统计。

**例子。** Attention 的 `8293` 中，两个 Reduce 的元素工作代理占 `8064`。它不是机器指令数，更不能直接当作耗时。跨不同算法或指令路径时，用一个工作单位同时代表不同类型操作，可能造成排序偏差，需实测判断。

**建议。** 先明确每个成本分量覆盖的操作与单位，再补齐普通算术、MMA 相关访问和同步成本；为代价模型加版本，保留“未建模”标记。异步收益需结合依赖和流水调度，不能仅因使用 TMA 就固定减分。评估字典序对实际性能的适用范围，不直接替换成未经校准的任意加权和。

**验收。** 逐操作贡献可复算；sector 表示模型中的访问区域计数，不宣称等于 DRAM 实际流量。模型改动用固定微基准检验排序相关性，并报告无法解释的案例；不只证明新的估算数更小。

实现入口：[SM90 成本模型](../../lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp)、[通用成本模型](../../lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp)、[SM90 成本测试](../../unittests/Dialect/Frisk/Layout/SM90CostModelTest.cpp)。

### O06：按存活区间估计寄存器和 Shared

**现状。** 当前寄存器项主要累加各值的布局槽位；Shared 项按存储根计入跨度。这些是资源风险代理，不是最终物理寄存器峰值或实测 occupancy。

**例子。** A、B 两个 Tensor 各有每线程 32 个槽位：若 A 使用结束后才产生 B，峰值可能接近一份；若二者同时存活，则可能需要两份。只累加无法区分。Shared 缓冲区复用和多级流水也存在相同问题。

**建议。** 加入 SSA 存活区间、别名根共享、转换临时值和重计算延长存活的影响；对分支与循环保守估计峰值。结合 dtype、目标分配粒度和后端寄存器结果，再估计资源对 occupancy 的限制。

**验收。** 成对测试能区分重叠/不重叠存活、Shared 可复用/不可复用；别名不重复计费。最终与后端资源报告核对，并单独报告估算误差，不能把槽位数直接命名为物理寄存器数。

实现入口：[通用成本模型](../../lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp)、[成本模型测试](../../unittests/Dialect/Frisk/Layout/LayoutCostModelTest.cpp)。

### O07：降低证明和求解开销

**P1.1 更新（2026-10-21）。** 已用受检完整硬件位矩阵、秩/核与同线程像空间包含替换 ownership/Broadcast 的枚举证明。131072 点正例从 Unknown 变为 Proven；实际地址表、Reduce 投影等仍保留 65536 点限制，因此不等于完整 Attention 的 Eall 已获支持。32,768 对 Broadcast 小矩阵差分及最终 465 unit/64 lit 通过。局部微基准改善明显，Attention 整体约 13.59→13.56 秒、8.21→8.19 秒，视为持平；逐函数热点、缓存命中率和更广泛求解开销优化尚未完成。详见[报告](p1_1_implementation_report.md)。

**现状。** 多项坐标、别名和执行证明使用有界枚举。Attention 中 `Eall` 的硬件槽位数为 `1024×32×4=131072`，超过相关证明的 65536 点预算，因此不能取得支持。这是未能证明，不是数学上已证伪。系统已有关联缓存和 footprint 缓存，不能将建议简化成“新增缓存”。

**建议。** 先测候选准备、地址枚举、联合关系和搜索各自的耗时与缓存命中率。对可表达的子类，利用已有 GF(2) 代数，以秩、像空间、核及坐标关系完成符号证明；对含进位、分段或非线性地址关系保留适当的枚举后备。检查缓存键是否覆盖类型、根对齐和目标等语义条件。

**验收。** 小域上符号证明与枚举 oracle 一致；至少有原先因点数预算返回 Unknown 的可证明案例被解决；报告时间和内存变化。不得通过移除预算、将 Unknown 当作 Proven 来扩大支持域，也不要求保留性能明显不佳的全复制候选。

实现入口：[布局代数](../../include/Dialect/Frisk/Analysis/LayoutAlgebra.h)、[执行证明](../../lib/Dialect/Frisk/Analysis/ExecutionLayoutProof.cpp)、[别名分析](../../lib/Dialect/Frisk/Analysis/LayoutAliasAnalysis.cpp)、[solver 基准](../../benchmark/layout/solver/layout_solver_bench.cpp)。

### O08：完善 Copy 路径的契约与集成覆盖

**现状。** vector、cp.async、TMA 已有候选和合法性规则。当前 Attention 未选择异步路径，不能解释成 Frisk 完全不支持它们。Global 对齐证据不足、单线程地址不连续、源地址不能由当前 TMA 描述符表示，都会阻止某条路径。

**例子。** 即使 Global 和 Shared 的起点都充分对齐，如果同一线程的两个 f16 元素相距 32 字节，也不能将它们视为连续 4 字节向量搬运。另一方面，Shared 足够对齐并不能补足 Global 的对齐保证。

**建议。** 在 O01 的固定外部布局下，分别构造合法向量、cp.async、TMA 的完整 Copy→MMA 用例；为对齐、访问连续性、描述符可表达性、writer/issuer 和完成机制提供细分诊断。扩展规则前先明确输入支持域。

**验收。** 每条路径都有正例，以及只破坏一个条件的反例；必要时可硬绑定路径验证合法性，但默认优化测试不应要求永远选择 TMA。缺证据时保守回退或说明失败；不得放松实际地址验证来“选中异步”。GPU 异步安全由 O11 验收。

实现入口：[SM90 Copy 契约](../../lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp)、[Copy 契约测试](../../unittests/Dialect/Frisk/Layout/SM90CopyContractTest.cpp)。

### O09：扩大 shape、广播和视图支持范围

**现状。** 当前广播规则要求静态形状、维度大于 1 且为 2 的幂，并校验插入轴和同线程可取关系。别名支持同元素类型、可静态证明的 cast/subview；不能将这些能力等同于通用 reshape、动态切片或 dtype 重解释。

**例子。** 已支持的 `tensor<64xf32> → tensor<64x64xf32>` 广播，不能直接推广为任意 `tensor<1x64xf32> → tensor<63x64xf32>` 广播。大小为 1 的轴、非 2 的幂边界与操作本身的形状契约都需单独处理。

**建议顺序。** 先支持单位维度和明确的标量/广播形式；再支持静态非整齐尺寸、padding/mask 和尾块；之后才评估受限 reshape/transpose/view 组合。动态偏移与 dtype 重解释另立契约，避免一次性扩大过多证明边界。

**验收。** 新支持域有坐标正反例、越界/唯一写者检查和完整算子链测试。非整齐 Attention 除 Copy 掩码外，还要正确处理 Softmax 的无效位置与 Reduce 中性值；不能只让 IR verifier 接受形状。

实现入口：[广播规则](../../lib/Dialect/Frisk/Analysis/BroadcastLayoutConstraints.cpp)、[广播测试](../../unittests/Dialect/Frisk/Layout/BroadcastLayoutTest.cpp)、[Guarded Copy 测试](../../unittests/Dialect/Frisk/Layout/GuardedCopyTest.cpp)。

### O10：改进跨循环和多消费者转换决策

**现状。** 系统已有 conversion 共享、有限纯计算切片重计算、有限 hoist 和循环深度成本代理。下一步应增强现有决策，而不是另建一套转换机制。

**例子。** 一个布局转换位于循环中，循环执行 100 次；若源值循环不变且转换可安全外提，执行一次通常值得考虑。若源是每次更新的 `iter_arg`，外提就可能改变语义。重计算也可能减少转换，却增加工作量与峰值寄存器。

**建议。** 在可获知时使用静态迭代次数或显式频率信息，完善支配关系、纯度、生命周期和跨 consumer 共享决策；同时纳入 O06 的资源影响。

**验收。** 覆盖循环不变/变化值、if/for/while、多 consumer 和副作用边界；验证转换执行次数、最终值语义、实际 IR 与重跑稳定性。未知迭代次数仍采用明确标记的保守代理，不能假定一次执行。

实现入口：[转换选择](../../lib/Dialect/Frisk/Analysis/LayoutConversionChoice.cpp)、[转换外提测试](../../unittests/Dialect/Frisk/Layout/LayoutConversionHoistTest.cpp)。

### O11：建立 Attention GPU 执行闭环

**现状。** 当前完整 Attention 的成果是布局推断与 IR 验证。M6 的布局转换 GPU 证据不能替代完整 Copy、MMA、Reduce、Broadcast、Softmax 和 Store 链的数值及异步安全验收。

**建议。** 首先完成一个固定 shape、固定外部 ABI 的可执行 Attention lowering，落实已选择的指令与布局契约；分开验收简单搬运路径和异步路径。之后再扩大形状、路径和调度范围。该项包含实质性的后端工作，不是给布局推断补一条测试即可完成。

**验收分三层。**

1. 数值：与匹配当前语义的参考结果比较，特别保留 Softmax 后 `cast_f16` 的语义，明确误差阈值和输入分布。
2. 安全：检查越界、未初始化读取、并发竞争及同步；异步路径验证 wait/barrier 和数据可见性。
3. 性能：固定设备、构建配置、预热和采样方法，对比不同布局方案，并同时报告运行时间、编译时间和资源使用。

只有这三层形成证据，才能称“布局选择改善了 Attention 性能”。在此之前，较低 CostVector 只代表模型偏好。

起点：[Attention 输入 IR](../../test/Transforms/attention-layout.mlir)、[Attention 集成测试](../../unittests/Dialect/Frisk/Layout/AttentionLayoutIntegrationTest.cpp)、[已有转换微基准](../../benchmark/layout/conversion/layout_conversion_bench.cu)。

## 5. P2：回答更高层的性能问题

### O12：自动选择 tile、线程数和流水级数

**现状。** 当前系统主要回答“给定 tile 和线程数，怎样分布数据、选用布局和契约”；不会自动搜索完整算子的分块尺寸、线程数量和流水级数。

**建议。** 将自动配置放在外层：

```text
生成有限配置（tile / threads / stages）
  → 构造对应程序与缓冲区
  → 对每份程序运行布局推断
  → 检查资源和可执行性
  → 编译测量或可信模型评估
  → 选择配置
```

**关键边界。** 不同 tile 的工作量不能直接比较单 tile 的 CostVector，需要核算完成整个算子的 tile 数、总搬运、边界、启动和预打包成本。Attention 分块还可能涉及在线 Softmax、跨 tile 状态与累加语义，并非修改几个 shape 属性。

**验收。** 对同一完整问题尺寸比较多个合法配置，数值一致，能输出最终配置及选择依据；联合记录搜索成本和运行收益。建议在 O01、O11 完成后推进，避免在不可执行程序上调参。

### O13：支持线程角色分工和流水重叠

**现状。** 指定 128 个线程不表示每条操作都必须由 128 个线程执行相同工作。当前已有 owner/writer、Reduce 分工和 TMA elected issuer 等语义；缺少的是自动规划通用的 producer/consumer 线程角色及多阶段流水。

**例子。** 将部分 warp 用于准备下一块数据，其余 warp 计算当前块，要求显式处理参与线程集合、缓冲区阶段、barrier 和生命周期。不能只给两个操作写不同的线程数就认为支持了 warp specialization。

**建议。** 将参与线程集合和调度契约与数据布局分开建模，再联合检查可取值关系、通信与同步。先从固定角色、固定阶段的受限模式做起，不直接搜索任意线程分工。

**验收。** 角色参与范围可解释；跨角色传值有合法同步和存储路径；有死锁、竞争、尾轮收尾及数值测试，并用 O11 的执行闭环测量实际重叠收益。此项不应通过放松既有 SameLayout/region 不变量实现。

### O14：建立系统级对比与消融评测

**现状。** 已有固定 TileLang 参考、属性测试及成本/转换基准；这些验证了部分语义与性能边界，还不足以证明整套 Frisk 推断系统在完整算子上的优势。

**建议。** 建立 GEMM、Reduce/Softmax、Attention、多消费者与循环等测试集。比较 TileLang/Triton 时固定版本、硬件、shape、dtype、输入输出布局和数值语义；分别报告“同一程序配置下的布局选择”和“各系统完整调优”的结果，避免混淆布局、分块与调度贡献。

**应验证的差异点。** 坐标/字节地址证明对跨视图一致性的作用；联合约束对局部贪心的改进；转换/重计算联合选择的收益；有界搜索的质量、耗时和解释性。已有设计特点不自动等于新的学术贡献，需要基线和消融支撑。

**验收。** 报告支持率、失败类型、编译时间、资源、运行时间及搜索最优性差距（仅对可精确求解的小图）；保留退化案例。每项实施前后同步长期设计、计划与对比文档，不对未检查的竞争系统版本作结论。

入口：[参考导出工具](../../tools/layout-reference/export_tilelang_layouts.py)、[基准运行工具](../../benchmark/layout/run_layout_benchmarks.py)、[长期对比文档](../layout_inference_strategy_comparison.md)。

## 6. 推荐推进顺序与共同守门条件

建议按以下阶段推进，每个阶段再独立确认具体实现契约：

| 阶段                  | 工作内容                                                             | 阶段结束应能回答的问题                                       |
| --------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------ |
| 0：固化基线           | 重跑已有回归，保存源码版本、输入、候选及成本模型版本；提交须另行授权 | 后续改进与哪个版本比较？                                     |
| 1：可信且可解释的选择 | O01、O03、O02                                                        | 固定真实输入输出后，为什么选这个方案，是否漏掉已知更优方案？ |
| 2：扩大能力并控制开销 | 按实际失败及 profiling 结果选择 O04、O07、O08、O09、O10              | 更多程序能否在合理编译时间内获得合法方案？                   |
| 3：执行与成本校准     | O11 与 O05、O06 迭代推进                                             | 估算偏好是否转化为真实数值正确性和性能收益？                 |
| 4：外层优化与评测     | O12、O13；逐步完善 O14                                               | 能否自动选择计算配置，并说明收益来自哪里？                   |

所有阶段共同遵守：

- 不覆盖显式布局、根地址和对齐硬契约；不以较低成本抵消不合法性。
- 不把 Unknown 当作 Proven；不把搜索未找到解等同于数学上无解。
- 保留候选冻结、单调传播、确定性、实际 IR 验证、文本重放和失败原子回滚。
- 优化前后同时比较正确性、选择质量、编译时间和内存；分别记录候选、模型、预算变化，避免归因混淆。
- 新能力先写入[设计文档](../layout_inference_design.md)和[实施计划](../layout_inference_implementation_plan.md)，完成后同步[体系对比](../layout_inference_strategy_comparison.md)及验收证据。本文的建议不能直接勾为已完成。

**下一轮最小建议范围：固定行主序 Attention 用例 + 决策 trace + 已知 608→576 漏选回归。** 先把“真实边界下选得合理且讲得清楚”做扎实，再扩展自动分块和线程角色调度。
