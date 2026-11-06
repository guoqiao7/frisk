> 2026-10-29 P1.4a：最小形状支持已完整验收，功能回归、独立源码及最终证据/文档复审均通过。静态正二次幂单位轴、插入轴Broadcast、rank≥2到rank≥1 Reduce及静态expand/collapse/transpose闭环；575 Release主单测、20 selection，双配置最终15聚焦/5lit及辅助守门通过。GPU基础后端及后续路线未完成。见[实施报告](../inference_detail/p1_4a_implementation_report.md)。

# P1.3：九项布局成本的身份、单位与覆盖边界

> 2026-10-29：P1.3c已完成验收，完整源码、驱动、数据与文档独立复审通过。默认legacy保留；显式`boundary-frequency=static-trip-count`统一转换/clone与受限外提的频率口径。Release主单测570、选择质量20，相关Debug/lit及辅助守门通过；18次语义核验与36次串行CPU调用通过，身份复验不变。下一独立任务为最小形状支持，GPU和跨框架比较尚未实施。见[阶段报告](../inference_detail/p1_3c_implementation_report.md)。

> 2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。 见[完整Attention资源报告](../inference_detail/p1_3b_attention_resource_report.md)。

## 完整身份与兼容性

默认 SM90 完整模型 ID 为 `sm90-layout-lex9-legacy-v1`，唯一来源是
`getSM90LegacyCostModelID()`。它标识下表整套公式、转换/clone 代理、legacy
循环深度频率和九项字典序，不是硬件性能保证，也不等同于 JSON 格式版本。
trace 和 stats 的根字段 `cost_model_id` 发布此身份；`schema_version` 仍为 1。
P1.3a仅添加身份元数据。P1.3c新增显式`sm90-layout-lex9-static-trip-count-v1`，只替换动态边界频率；默认legacy评分、候选、硬约束与预算保留。

target 通过 `LayoutTarget::costModelID()` 提供身份；旧自定义 target 默认
`unversioned-target`，有自己的合法 ID 时保留自己的身份，不能冒充 SM90。
合法 ID 是 1–64 字节的 ASCII 字母、数字、`.`、`_`、`-`；空、不合法或过长
ID 整体回退 `unversioned-target`，不截短成看似有效的版本。
成功、失败、尚未求解以及可选 trace 事件被截断时，保留实际已知的 target 身份。
身份位于强制摘要，不依赖会截短的旧 `cost_coverage` 文本；4096-byte 最小
trace 上限不变。尚无已知 target 的独立空报告明确为 unversioned。

## P1.3c显式静态次数模型

三个入口`frisk-layout-pipeline`、`frisk-infer-layouts`、`frisk-optimize-layout-conversions`均接受`boundary-frequency=legacy|static-trip-count`，默认legacy。`LayoutTarget::boundaryFrequencyMode()`提供实际模式，不从模型ID字符串推断；物化复验与后优化沿用同一模式。

static-trip-count对常量有符号for、正step使用扩宽算术计算`max(0,ceil((ub-lb)/step))`；动态、unsigned、while及不可解码的LoopLike用factor=2代理，该代理不是上界。嵌套相乘，任何已知零次祖先优先为精确零；其他溢出单独标记饱和。这里只建模循环次数，未建模分支概率。

只有下表④的动态转换/clone工作乘次数；⑤⑥资源、⑧代码大小与实际IR普通算术诊断不乘次数。静态模式的受限register-only外提要求严格动态收益：零次/一次保留原位置，安全的百次循环可外提。legacy的循环种类、权重及外提打平规则保留。

原`conversionBytes`仍是唯一Convert leader的未加权流量，保留既有聚合行为。新`weightedConversionBytes`、`weightedConversionBytesUsesProxy`和`weightedConversionBytesSaturated`使用独立受检运算，不计clone/sync/code size。`finalCost`及这些选择统计是后优化之前的选中方案，不是hoist后IR重新评分；实际IR资源/算术诊断另列。

## 九项覆盖表（默认legacy；static模式只替换④的频率）

字段顺序就是比较优先级：逐项字典序比较，不是不同单位之间的加权求和。
各个贡献在同一字段内使用 uint64 饱和相加，`saturated` 独立标记溢出。
未赋值的零字段意味着未建模，不能解释成测得免费。

| 顺序 / CostVector 字段 | 当前实际计入的代理及单位 | 不代表什么 / 未覆盖范围 | 主要源码 |
|---|---|---|---|
| ① `instructionPathAndWork` | Copy 的静态 setup/work：TMA 为 `4 + ceil(bytes/1024)`，cp.async 为 2；非 bulk access 按物理 warp / register-vector 发射组计数，vector Copy 分别计 load/store，cp.async 只在 read 侧计一次。MMA 为选定几何的 `gM*gN*mRepeat*nRepeat*kRepeat`；Reduce 为源与结果逻辑元素数之差。不同代理共用此字段，并非统一机器指令计数。 | 不计普通 arith/math 的执行时延，也不是 Copy/MMA/Reduce 的周期或吞吐率模型。clone 的字节工作在④，不在此估计其 opcode 时延。 | `SM90CostModel.cpp::accessCost/evaluateSM90LayoutCost` |
| ② `memoryTransactions` | Global access 对每个发射组独立计算实际地址范围触及的不同 32B sector 数，再求和；TMA 使用 bulk 分组。未知 base phase 在 root alignment 允许的相位中取最坏值。guarded Copy 的 inactive 点不产生地址。 | 不是 DRAM 实际字节/事务、cache miss 或带宽；不跨不同 access/warp 发射组去重，不预测 cache 命中与复用。 | `SM90CostModel.cpp::accessCost` |
| ③ `bankConflictDegree` | 非 bulk Shared access 在独立的至多 128B phase 内，按 32 个 4B bank 分组 distinct word，以 `max(distinct words per bank)-1` 求冲突代理，再对 phase 累加并取允许 base phase 的最坏值。读 broadcast 和 packed write 的同一 word 只计一次。 | 不是冲突时延，也不是 subword 写入正确性证明。不覆盖 TMA 的 Shared bank 成本；不会展开 MMA 的全部 Shared operand 访问，因此不是完整 MMA Shared 带宽模型。 | `SM90CostModel.cpp::accessCost` |
| ④ `conversionBytesAndSync` | 非 identity 转换按 bounded owner proof 分类：register 取目标 carriers，shuffle/shared/conservative 取源与目标 carriers 之和，乘 `ceil(elementBits/8)`；shared/conservative 再加 `4*max(sourceThreads,targetThreads)*2` 同步代理，其他分类同步为 0。clone 为输出 topology 全部 carrier 槽位乘元素字节宽度。仅此动态代理字段按 insertion point 的 enclosing `scf.for/scf.while` 深度乘 `2^depth`（饱和）。 | 字节与同步工作混合的排序代理，不是实际 scratch allocation、传输时间或普通算术时延。legacy 深度频率不是 trip count、分支概率或 profiling；不推断循环执行次数。 | `LayoutCostModel.cpp::evaluateLayoutConversionCost/evaluateLayoutCloneCost/weightLayoutBoundaryCost` |
| ⑤ `spillRiskAndRegisters` | 每个非 use、非 instruction 的 distributed 变量加 encoding `topology[0]` 的逻辑 register-carrier 槽位数；被保留的 clone 另加输出 encoding 的同一槽位数。 | 不是程序点活跃峰值、物理寄存器分配、dtype 位宽折算、spill 数或 occupancy。本字段不使用 P1.3b 的独立存活诊断。 | `LayoutCostModel.cpp::evaluateStaticLayoutCost/evaluateLayoutCloneCost` |
| ⑥ `sharedBytesAndOccupancy` | Shared storage 的可证明地址 footprint，以各访问区间终点的向上取整字节位置形成 root span；同一 alias root 取最大 span、去重计一次，不同 root 相加。无 alias 信息时使用布局 footprint。 | 不是按活跃区间复用后的后端 allocation、occupancy 或峰值活跃 Shared；span 不等于去掉 hole 后的有效 payload。 | `LayoutCostModel.cpp::evaluateStaticLayoutCost` |
| ⑦ `replication` | 与⑤相同的 distributed 变量与 clone，分别累加 `replication-1`。 | 不是复制字节、动态执行次数或物理资源占用；use 端不重复收费。 | `LayoutCostModel.cpp::evaluateStaticLayoutCost/evaluateLayoutCloneCost` |
| ⑧ `codeSize` | 每个非 identity 转换静态计 1，每个被收费的 clone op 静态计 1；已共享的转换/重计算组按选定 leader 处理，不按每个 consumer 重复计物理项。循环深度不放大此字段。 | 不是机器码字节数、opcode 权重或动态指令数。 | `LayoutCostModel.cpp::evaluateLayoutConversionCost/evaluateLayoutCloneCost`；`LayoutConversionChoice.cpp` 的分组收费 |
| ⑨ `deterministicTieBreak` | 九项成本比较中的最后一个保留字段。当前 SM90/static/conversion/clone 生产评分不向此字段写入非零值，故当前真实 SM90 贡献为 0。 | 不能宣称此字段正在编码候选序号。稳定候选 key、搜索顺序及其他稳定排序实现确定性；自定义 target 可以按自己的模型给此字段赋值。 | `LayoutCostModel.cpp::operator<`；`LayoutCandidateSolver.cpp` 的稳定候选排序 |

静态资源 lower bound 使用 `evaluateStaticLayoutCost`，不是新增公式。
③的 128B phase 描述适用于常规标量/向量宽度。native 元素宽于 16B 时，现有实现
使用保守的 distinct-word 代理，不声称对应最终指令 lowering；宽于 128B 时，phase
的 lane 数下限钳制为 1，单 lane 本身就可能跨越超过 128B，因此不保证 phase 总宽度
仍不超过 128B。这是原有评分的支持域说明，不是新评分规则。
转换/重计算的共享与候选选择由 `LayoutConversionChoice.cpp` 处理；trace 记录已存在
的最终贡献，不重新评分。当前 scope 的完整 `cost-contribution` 加 `omittedCost`
仍逐项等于 `finalCost`；历史尝试不能混入最终尝试的成本。

## 旧分项标记与完整 ID 的映射

这些已有标记是完整 `sm90-layout-lex9-legacy-v1` 内部子模型的说明，不是另一个完整
模型 ID；旧 target coverage/provenance 文本继续保留。

| 旧标记 | 当前完整模型内的含义 |
|---|---|
| `bounded-owner-proof-v1/{identity,register,shuffle,shared}` | 有界物理 owner 证明及转换类别（④、⑧）。 |
| `cta-staging-upper-bound-v1` | owner 证明预算/支持域不足时的非零 staging 上界回退（④、⑧），不是已生成的 lowering scratch。 |
| `pure-slice-byte-work-v1` | 可重计算 pure slice 的每个 clone 输出-carrier 字节工作及静态资源（④、⑤、⑦、⑧）。 |
| `loop-depth proxy` | 仅④的 legacy `2^depth`，完整默认模型ID保持legacy；P1.3c显式静态模型使用独立完整ID。 |
| `register/replication slots`；`shared root spans` | ⑤、⑦和⑥；已有 coverage 文本未为其单独编号。 |
| `sm90-shape-work-v1` | ①的 Copy setup、MMA 几何与 Reduce 工作量代理。 |
| `warp-issue-32B-sectors-v1` | ②的逐发射 sector 计数与未知 base phase 上界。 |
| `shared-128B-phase-32x4B-distinct-word-conflicts-v2` | ③的 Shared phase/distinct-word bank 代理。旧 v1 不是本完整模型 ID 的当前公式。 |

## 明确未建模

本模型没有硬件 latency、cache 层次/命中、跨操作 overlap、调度、pipeline、占用率反馈、
完整 MMA Shared 访问、普通算术 opcode 时延、物理寄存器分配或 spill 预测。
诸如 Mul/Exp/Div 即使改变了实际 IR 的普通算术工作，也不能从九项代理推导成统一运行时间。
P1.3a Task 2 已增加默认关闭的 `ir-diagnostics`，独立诊断实际优化 IR 的静态逻辑
element-op，不把该统计混入 CostVector、剪枝、tie-break 或布局拒绝。
阶段状态与最终验收见[实施报告](../inference_detail/p1_3a_implementation_report.md)；
P1.3b 已接入独立的实际IR资源诊断，但仍不改变表中任何评分公式；
单块支持域按真实SSA区间给逻辑槽位/字节峰值，Shared按root去重给活跃峰值与声明容量。
规范ranked-Tensor Broadcast现按原子SSA边界分析，其他非规范region仍unavailable。
两类完整Attention均available：各16个Tensor SSA、3个Shared根，每线程峰值106逻辑槽/424字节，
Shared活跃峰值/声明容量均6144字节；15次语义核验与36次串行CPU观测已完成，
旧/新和诊断开关对照保持IR、完整成本、非计时求解统计及旧arith诊断不变，最终IR重放核验IR与诊断等价。
最终独立证据/文档复审已通过；模型仍为 `sm90-layout-lex9-legacy-v1`，O06全目标未完成；
不能把新峰值当作⑤/⑥、lower bound、物理寄存器或occupancy。
以上资源诊断数字为P1.3b历史验收。P1.3c静态循环频率现已实现并通过功能回归，正式观测和最终数据/文档复审均通过；见[P1.3c阶段报告](../inference_detail/p1_3c_implementation_report.md)。

历史检查点（2026-10-27，已被2026-10-28结果取代）：当时实际Attention的Broadcast隐式region仍令资源诊断unavailable，完整验收待确认窄特例。原探针证据保留在[P1.3b阶段报告](../inference_detail/p1_3b_implementation_report.md)。
