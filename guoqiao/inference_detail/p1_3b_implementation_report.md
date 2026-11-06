# P1.3b：实际 IR 的存活资源诊断

2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。 详见[完整Attention资源报告](p1_3b_attention_resource_report.md)。

## 1. 本轮解决什么问题

旧 CostVector 中的资源字段是可加代理，不能回答“程序执行到某一步时，同时需要保存多少 Tensor 数据”。本轮在转换优化后的真实 IR 上补充只读诊断，不改变候选、评分、搜索预算或默认推断结果。

只在开启 `ir-diagnostics=true` 时计算，结果位于 stats/trace 的 `actual_ir_diagnostics.resources`。模型仍是 `sm90-layout-lex9-legacy-v1`。关闭时该诊断保持null，合法但资源分析不支持的if/for仍可成功推断。

## 2. Tensor：看同时存活，不看累计出现

设 A、B 各占每线程32个逻辑槽：

```text
定义 A → 最后使用 A → 定义 B → 最后使用 B    峰值32
定义 A → 定义 B → 同时使用 A、B              峰值64
B = convert(A)                             转换边界至少64
```

一个结果被两个消费者使用仍是一个 SSA 值。输入与输出在操作边界同时计数，不假设后端能原地复用寄存器。32槽 f32 是128逻辑字节，32槽 f16 是64逻辑字节；两者同时存活是64槽、192逻辑字节。

这不是物理寄存器分配结果，未包含执行临时量，不据此宣称 spill 或 occupancy。

CLI实际验收例子来自 `test/Transforms/layout-ir-diagnostics.mlir`，逻辑计算如下（省略已推断类型编码）：

```mlir
%m = arith.mulf %a, %b : tensor<8x8xf32>
%e = math.exp %m : tensor<8x8xf32>
%d = arith.divf %e, %m : tensor<8x8xf32>
return %d, %m : tensor<8x8xf32>, tensor<8x8xf32>
```

最终选中布局中每个值每线程1个f32槽，真实诊断如下；entry编号0，以上四个操作编号1至4。

| SSA | 存活区间 | 每线程逻辑字节 |
| --- | --- | --- |
| a、b | 各[0,1] | 各4 |
| m | [1,4] | 4 |
| e | [2,3] | 4 |
| d | [3,4] | 4 |

第1步同时保存a、b及新结果m，第3步同时保存m、e及新结果d，因此峰值是3槽/12字节，最早出现在第1步；不是累计5槽/20字节。Shared声明和活跃容量均为0。此完整JSON已通过CLI精确断言、开关对照、trace截断和重跑检查。

## 3. Shared：区分别名、存活峰值和声明容量

同一个4KiB底层存储的两个 view 只计一个 root；容量按完整根地址跨度，而不是子 view 大小。

两个不同4KiB roots 若错开使用，逻辑存活峰值可以是4KiB，但声明容量仍是8KiB。本分析没有实现后端 allocation 复用。cp.async/TMA 等无法在该层证明完成的使用会保守保活至域末，或明确报告不可用。

## 4. 支持边界

单块无嵌套控制流，以及固定单CTA `frisk.parallel ranges=[1]` 计算域；capture 显式作为 live-in。if/for/while、多块及无法证明的布局/别名/生命周期返回结构化 `unavailable`，不以0冒充结果，也不让原本合法的布局推断失败。不同函数/计算域独立报告，不合计成全模块峰值。

## 5. 历史验收与 Attention 探针（2026-10-27）

以下为窄特例获批前的阶段证据，保留原始测试数量、配置失败与unavailable探针；当前实现与实验状态以2026-10-28最终完成记录及完整Attention资源报告为准，最终独立证据/文档复审已通过。

已完成第一轮纯分析器验收：16项资源测试、2项JSON生命周期测试、15项别名测试、2项encoding测试，在Debug和Release中各35/35通过。包括上文32/64、转换共享、混合dtype、Shared同根去重/异根错开、offset/stride/降维、capture、异步保活/逃逸、溢出和不可用状态。先有行为RED再修正实现；没有把编译失败作为缺失功能的证明。

Task2接入验收：Debug/Release各87项通过，相关lit各4项通过。实际postoptimizer保留clone/共享转换的结果与独立SSA def-use区间及逐边界求和一致；开关保持IR、完整成本、非计时求解统计，失败清空诊断并回滚原IR。独立复审通过。

额外回归：全量Release545/545通过（65 suites，266.658秒，无disabled/skipped/errors/failures）；4项Python CTest通过，既有Debug配置的5项normalization测试和API/SM90工具守门通过。初次将后面三项指向没有构建Python扩展/独立工具的Release目录而失败，改用已有Debug产物后通过；这是配置问题，不作为产品功能RED。未重建那些未修改的Debug Python/工具产物，也未重跑整个Debug主单测。

两类Attention实测探针均 `status=committed`，输出IR逐字节等于P1.3a对应输出；但resources是 `unavailable`，原因精确落在第20、25个 `linalg.broadcast`。这里没有发布虚假的资源峰值，也没有影响原有布局推断成功。探针不是CPU性能采样。

历史待办检查点（2026-10-27；资源区间与实验已补齐，整体最终复审已通过）：两类Attention各SSA区间/峰值/声明容量；三输入的完整旧/新开关成本和统计对照、36次串行CPU观测，以及P1.3b整体最终复审。P1.3c尚未开始。

历史契约检查点（2026-10-27，2026-10-28已授权并实现）：`linalg.broadcast` 是带隐式标量yield region的named op。如果按“所有带region操作均不可用”处理，会拒绝Attention。建议仅对规范Broadcast增加原子Tensor操作特例，计外层输入/输出边界，继续拒绝if/for/while及任意其他region。确认前暂未实现该特例。

本报告不代表P1.3c静态循环次数模型或P1.4 GPU执行已完成。


## 2026-10-28：最终完成记录

2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。

两个Broadcast的窄特例已通过规范正例与8种非规范形态测试，包括复审发现的零region绕过入口；该问题先产生实际行为RED，再修复并以最终源码完成新一轮回归。此前全Release549通过及中止的旧lit仅作为历史，不替代最终代码的验证。所有失败、配置告警和被替代运行均保留。

三输入的旧/新与开关对照、最终IR重放共15次；所有语义门槛通过后off/on各1预热+5采样，共36次串行调用。两类Attention从最终IR逐SSA/Shared root独立核算通过。数据与源码/工具/驱动身份见[测量JSON](../optimize_plan/p1_3b_measurement_results.json)，原始日志/IR/stats/trace/审计与复审见[证据索引](../optimize_plan/p1_3b_artifacts/README.md)。

P1.3c未开始，O06全局资源目标仍未全部完成；本轮保持源目录main和所有既有改动，不自动commit/push。
