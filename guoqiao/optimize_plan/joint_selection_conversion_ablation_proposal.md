# 联合选择及转换消融方案

日期：2026-10-30。状态：方案，尚未实现或运行。本阶段作为一个完整交付任务，内部连续完成控制接口、共同后端/用例、测量与统一报告，不按算子或开关另立任务。

代码起点：P1.4c已提交 `dd577f6c1e0c87fbe5b5bf204fd250c00a613ce1`。提交前只删除test_async_export.py末尾一个空行，Python AST不变，原接受源码及说明保留在p1_4c_artifacts/precommit-format*。既有GPU证据为功能起点，不是本阶段性能结果。

## 目标与问题

在相同算法、ABI、dtype/精度、候选域、硬约束、执行后端及总搜索预算下，回答：

1. 先选producer再补全consumer的顺序方案，与全图一次联合选择，是否产生不同的合法决策和实际执行效果？
2. 共享转换、有限重计算与后转换优化这一组合，能否减少实际通信/重复工作，并转化为GPU收益？
3. 哪些结果只是模型分数下降、局部平局消除或编译器最终折叠，哪些有可重复的GPU改善？

不以加速或严格选择反转为预设结果；无差异、退化、不支持和预算耗尽均保留。使用GF(2)或Triton低层编译本身不作为创新/性能优势结论。

## 当前必须承认的实现边界

- SolverOptions没有local/joint开关。refineBeamSolution只是beam结果的局部细化，不能作为局部布局选择基线。
- chooseLayoutConversions已在完整assignment评分时执行同源/目标/block共享与至多8-op白名单纯计算slice rematerialization；verifyLayoutConversionChoices会重算默认策略。保守模式必须贯穿评分、求解、复验、物化和oracle，不能只跳过末尾pass。
- frisk-optimize-layout-conversions是另一个后优化阶段，包含规范折叠/组合、受限elementwise commute和register-only循环外提。应与前述计划选择分开记录。
- 当前执行backend的f32非identity转换全部使用Shared staging及两次CTA barrier；模型Register/Shuffle分类尚不等于实际机器指令。必须先统一补齐路由再冻结后端，且所有消融组共用该实现。
- 当前MMA执行支持单warpgroup128线程、native固定fragment/grid[1,1]，不能把P1.2中256线程C1/C2/C3的geometry反转直接当成本阶段GPU证据。
- 当前export不支持scf循环，Reduce/Broadcast有明确有限支持域。因此主GPU矩阵不纳入循环频率/外提或未支持的Reduce维度；不为本阶段顺带扩展多warpgroup、流水或任意shape。

## 主对照：四组2×2实验

以下是计划中的实验模式名称，尚非已有CLI选项。

| 组 | 布局选择 | 转换处理 |
|---|---|---|
| L-C | producer局部目标先选，固定后补全全图 | 必要显式转换，禁跨use共享/remat与后优化 |
| J-C | 全图一次联合选择 | 同L-C |
| L-O | 同L-C的两阶段顺序选择 | 现有完整共享/remat计划和后优化 |
| J-O | 全图一次联合选择 | 同L-O，即完整配置 |

### L和J的准确含义

L是新定义的两阶段顺序基线，不是声称复现现有Triton/TileLang或某个生产greedy算法。第一阶段保留完整图硬合法性，只按预声明producer范围计分，选择producer输出encoding及指令geometry；局部打平仅按producer稳定ID裁决，不让consumer assignment顺序或未来软成本暗中破平局。保存一份完整合法witness。

第二阶段固定producer决策，使用完整目标优化剩余变量。这避免将一个任意糟糕的consumer补全当成局部基线。J则一次优化全图。L/J差异包含决策顺序和预算分配效应，报告不得写成“完全相同算法只改目标”。

默认exact=256、beam=32、总探测=65536、refinement=256保持。L两个阶段共享同一总计数器；建议第一阶段最多占总probe/refinement一半，未用预算加剩余额度交第二阶段，不能各跑一份完整预算。扩展/关系探测/完整评分/refinement采用同一计量规则，记录分阶段与总消耗。

第一阶段保存的合法witness可在第二阶段预算耗尽时保留，但合法返回、搜索完成及第二阶段改进分别标记。没有合法witness则报告失败；不得使用oracle替换默认解。局部cost scope的partial lower bound必须同scope一致，无法证明时使用安全零下界，不以完整图下界误剪局部目标。

主机制小图应控制到exact阈值以内，使预算截断不主导结论；Attention作为有界真实图另列。

### C和O的准确含义

C保留必要且合法的转换，按独立use物化；identity不硬造工作。正确性所需first_owner、内存同步和外部ABI不能关闭。O采用现有共享/重计算计划及后优化。

这是一个转换优化组合消融，不能凭四组结果单独声称共享、remat或后优化各贡献多少。任务内部再固定同一assignment，分步回放“独立显式转换→共享→允许remat→后优化”，仅在对应机制实际适用的行执行，分别存计划及实际IR/机器指令变化。

如果O靠remat避开C无法证明或降低的转换，该行属于支持能力差异，不纳入配对性能/选择收益。四组配对主表只包含共同可执行可行域，其他行另表报告。

## 共同候选域、策略身份与复验

先准备一次候选及约束，必要relaxation只在公共准备阶段完成，冻结传播前快照后复制给四组。固定候选稳定ID、硬绑定、完整关系证明和支持状态，不允许各组单独relax添加候选。受控A/B白名单要明确标为实验注入，不冒称自然候选枚举已覆盖。

每个实验保存selection scope、producer集合、conversion policy、post-opt状态、成本模型ID及频率模式。策略必须在score/solve/verify/materialize/oracle/replay全过程一致，默认生产行为不变。不能通过修改最终IR绕开计划复验，也不能重放默认pipeline而重新执行Full策略。

主矩阵固定当前lex9/legacy口径，不调权重，不进行成本校准或搜索优化。物化前的selection cost与物化/后优化后的实际IR诊断分开；不将原finalCost冒称重新评估后的最终机器成本。

## 统一后端及最小用例集

### 必要的转换路由补齐（同一任务的前置内部步骤）

优先复用现有LayoutConversionPlan/planRedistribution的owner映射和payload逻辑：同线程转换实际使用寄存器重排；同warp转换实际使用受检shuffle；跨warp转换使用Shared。核验完整owner、副本、first_owner、shuffle参与范围、barrier及scratch生命周期/预算，不能只根据分类字符串发指令。

所有组使用完全相同的路由和分配实现。后端完成一次语义/bitexact/安全验证后冻结，再生成四组正式artifact；不得只给O组使用更好的lowering。声明Shared数据根与scratch分配策略，不顺带新增资源峰值目标或流水调度。

### 主机制图

从现有128线程[64,16]f32、topology[8,32,4,1,1]的A/B映射出发，使用已支持mulf/subf和固定输出ABI，先构造少量预登记图族：

- 无冲突对照：单producer/单consumer，检验四组本应一致或成本打平的情形。
- 固定consumer冲突：同一producer的A/B编码consumer支路都影响最终O，检验提前固定producer与完整联合选择的差异。
- 多use/diamond：同源同目标重复边界与可复用纯计算slice，检验共享/remat及其转换成本。优先移植现有已验证转换选择见证，不使用当前remat白名单之外的exp/div/truncf来制造案例。

固定consumer可借鉴SelectionCase的受控pre-layout图。若输入初始Convert源尚无encoding，要登记为实验分析入口；经求解/物化/actual verifier后才进入执行，不能称其原始文本已通过生产native入口。

既有transport fixture将source A和target B硬固定，只能作为转换正确性对照；必须重新开放producer候选才是选择实验。所有支路都可观测，四组完全同一计算图。反转统计区分严格反转、局部平局消除、仅consumer补全变化和无变化。

每个主图冻结前验证至少两份完整合法assignment可以实际lower并运行。用例未产生选择差异时如实记录，不改权重、ABI或计算图迎合预期。

### 完整Attention

原固定ABI Attention保留为整链回归和自然选择域对照。当前不承诺它存在多个可执行MMA geometry；若四组完全相同，这一行仍应保留。已有P1.2较大geometry反转只能继续列CPU选择证据，除非另行明确扩展后端并独立验证。

主因果矩阵固定vector弱对齐ABI。cp.async/TMA的强QKV128/O1作为独立分层回归/补充矩阵，每层内部四组ABI完全相同；不混合强弱ABI计算收益。

## 验证与测量

### 正确性和有限域oracle

- 每组完整实际IR、Storage逐坐标ABI、策略计划、parse/replay、TTGIR/PTX/cubin数值和安全验证均通过后才能计时。保存失败类型：NoFeasible/Unknown/预算不足/后端不支持/超时/数值失败，不从报告中删除。
- 复用SelectionOracle bounded框架（最多65536节点、4096完整评分、120秒）作为独立验证；相同policy/scope核验L两个阶段及J。只有complete+exhausted才能宣称有限域最优，截断标best-so-far。
- 不将局部与完整图成本直接相减；L最终补全与J均用同一完整口径复评。lex9差异报告首次不同分量和值，不把异单位向量直接算“百分比改进”。
- 独立transport/packing/cast与完整Attention沿用既有数值口径；新f32图有独立逐op参考并在运行前冻结误差/精度规则。

### GPU和编译开销

当前runner每artifact20个event样本只适合功能观测。正式性能测量预编译、预加载和统一预热；kernel计时排除JIT、descriptor setup/上传和graph capture，另报其开销。

主小tile可复用固定版本triton.testing.do_bench_cudagraph(return_mode='all')，保留所有批次样本。四组同一配对批次使用相同输入、offset与GPU状态，采用交错/随机顺序，串行测量并保存UUID、时钟、负载、工具/config身份。记录预热、graph内重复数、至少10个独立配对批次；必要时按预先固定最大批次数扩展，不能只在有利结果时增加采样。

分别报告graph replay时间、batch内每launch摊销时间，以及普通单launch观测；CUDA Graph内部重复launch不视为独立统计样本。置信区间使用独立配对批次差异/比值，不伪造大量独立样本。不同时跑构建、oracle或其他重负载任务。

保留中位数、分位数、原始样本、配对不确定性，接近噪声不宣称提升。编译分列局部/联合选择、完整input→cubin冷编译、host descriptor setup；报告真实register/Shared/local/spill、转换分类和实际指令/同步数量。TMA总Shared以physical_resources为准，不以config中的6144B数据payload冒充全部静态分配。

## 一个任务内的推进和交付

1. 连续完成实验控制、共同候选冻结和必要转换路由，建立小图与Attention统一入口；先验证语义/复验和共同可执行域，再冻结运行矩阵。
2. 在冻结矩阵上执行四组、固定assignment回放、独立oracle和GPU测量；实现/验证进程全部关闭后串行正式计时。
3. 完成新/受影响回归、独立源码及数据复审，交付一份报告和一个持久证据索引，覆盖成功/退化/无改善/不支持案例。

验收要求：策略因果边界清楚、共同域与预算可核验、至少两份真实合法执行方案、转换/重计算优化具有真实测试覆盖、数据完整可复放。若某机制没有适用见证，明确标“未覆盖”，不宣称该机制验证完成。性能提升、严格joint反转或胜过Triton/TileLang不是功能验收的强制门槛。

本阶段不实施成本/资源校准、搜索加速、外层tile/线程/流水搜索或跨框架调优。完成后按总路线继续成本与资源校准→搜索开销优化→冻结公平对比。

## 参考实现依据

- include/Dialect/Frisk/Analysis/LayoutSolver.h
- lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp
- lib/Dialect/Frisk/Analysis/LayoutConversionChoice.cpp
- lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp
- lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp
- include/Conversion/FriskLayoutToGPU/LayoutConversionPlan.h
- benchmark/layout/selection/{SelectionCase,SelectionControlled,SelectionOracle}.cpp
- benchmark/layout/execution/{backend,transport_fixture}.py
- guoqiao/inference_detail/p1_2_selection_quality_report.md
- guoqiao/inference_detail/p1_4c_implementation_report.md
- Triton计时API：https://triton-lang.org/main/python-api/generated/triton.testing.do_bench_cudagraph.html；本地3.4.0源码已核实return_mode='all'。
