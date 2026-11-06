# P1.3c 静态循环频率实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。用户于2026-10-29授权总路线，并允许拆为多次可独立复核的任务；本轮交付P1.3c，最小形状支持为下一项。不重复已验收P1.3b。

**Goal:** 新增显式可选的静态循环频率模型，统一转换、重计算边界计费与受限外提决策，保留默认legacy行为。

**Architecture:** 独立只读frequency helper负责常量循环次数和未知代理；模型选择随target/pass实例传递。九项成本结构与可加求解框架保留，仅动态边界项使用新频率；最终IR的现有诊断保持独立。

**Tech Stack:** C++17、当前安装的MLIR/LLVM、GoogleTest、lit、Python证据驱动。

## Global Constraints

- 直接在`/home/baopeihua/frisk`修改，保留全部已有改动；不worktree/stage/commit/push。每次只有一个源码/构建拥有者；复审只读。本轮不运行GPU、不实施P1.4或资源目标替换。
- 候选、硬契约、外部ABI及默认预算不变：exact=256、beam=32、max probes=65536、refinement probes=256。Unknown不能当作Proven或Disproven。
- 默认模型保持legacy，原模型ID为`sm90-layout-lex9-legacy-v1`。新模型ID为`sm90-layout-lex9-static-trip-count-v1`，CLI选项为`boundary-frequency=legacy|static-trip-count`。不得使用全局可变开关或悄悄改默认。
- 新模型只乘动态conversion/clone byte-work项；静态code size、资源、普通算术诊断不乘频率。原`conversionBytes`保持每个唯一实际Convert leader的未加权流量含义。
- 当前支持域为常量有符号`scf.for`正step；动态/unsigned/while/无法可靠解码的循环每层factor=2并标记proxy，proxy不是次数上界。保持既有register-only、无副作用、支配及循环不变外提条件；不扩大shuffle/Shared/if/while外提支持。
- 同一模型实例用于选择、计划复验、物化和后优化。计划新增元数据必须纳入一致性校验。文本重放、确定性、失败回滚保持成立。
- 测试驱动：先记录有意义RED，再GREEN；新增单测在Debug和Release验证。最终一次完整Release单测和相关lit/CTest，独立任务复审及阶段整体复审。源码身份基于实际脏工作区快照，不能用HEAD代替。
- 本轮基线为`/tmp/frisk-p13c-20261029-wrrdckp5/tree`，324文件；旧Release工具为同目录`frisk-opt`。工作日志、报告和review diff放同目录，最终复制到持久artifacts。

## Task 1: 无副作用的循环频率分析

**Files:** 新建`include/Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h`、`lib/Dialect/Frisk/Analysis/LayoutBoundaryFrequency.cpp`、`unittests/Dialect/Frisk/Layout/LayoutBoundaryFrequencyTest.cpp`；只修改相应两处CMakeLists注册。此任务不接入生产评分和pass。

**Interfaces:**

```cpp
enum class LayoutBoundaryFrequencyMode { Legacy, StaticTripCount };
struct LayoutBoundaryFrequency {
  uint64_t factor = 1;
  bool usesProxy = false;
  bool saturated = false;
};
LayoutBoundaryFrequency getLayoutBoundaryFrequency(
    Operation *insertionPoint, LayoutBoundaryFrequencyMode mode);
```

声明位于`mlir::frisk`，头文件仅依赖cstdint及Operation前置声明。枚举/结构/函数名为后续集成契约。分析从insertionPoint的parent开始，不包含insertionPoint本身；空指针返回factor=1。

静态已知次数计算为`max(0,ceil((ub-lb)/step))`，使用足够宽APInt或受检无符号运算，避免signed subtraction/round-up overflow。支持当前MLIR的常量index/可可靠解码的有符号整数，不能静默截断宽常量；超出实现能证明的表示返回proxy。检查unsigned比较属性；当前依赖无该语法也必须保守处理同名属性，而不能推断signed。

遍历所有祖先循环后组合：任何可证明零次的祖先优先令整体factor=0、usesProxy=false、saturated=false，即使内层未知或此前中间乘积溢出。其他已知次数与proxy相乘，超过uint64最大值饱和并标记；刚好等于UINT64_MAX不是溢出。Legacy helper返回每层2的proxy，只反映深度、不观察常量零次；后续legacy评分继续调用旧weight函数以保留其历史饱和细节。

- [x] 在合法MLIR上添加可区分旧代理和真实次数的失败测试。包含0/1/100次、负下界、非整除step、3×5、3×未知、0×未知、0×溢出、多层溢出、极端有符号边界、while/动态边界、unsigned标记、空指针及insertionPoint自身排除。
- [x] 记录RED实际命令和失败断言；缺符号编译失败不能替代全部行为RED。
- [x] 实现公共helper，复用当前LLVM工具，不引入逐迭代枚举或任意循环改写。
- [x] 构建并运行Debug/Release的`LayoutBoundaryFrequencyTest.*`，记录测试数量/输出/XML/文件SHA；关闭所有构建测试进程后交回拥有权。
- [x] 自审并写`/tmp/frisk-p13c-20261029-wrrdckp5/task-1-report.md`，由独立任务复审通过后进入Task 2。

## Task 2: 成本、目标、pass、转换优化与诊断一致接入

**Files:** `LayoutCostModel.h/.cpp`、`LayoutTarget.h`、`LayoutConstraint.h`中conversion edge、`LayoutConversionChoice.cpp`、SM90 target/cost model接口及实现、`LayoutStatistics.h/.cpp`、`LayoutTrace.cpp`、`Passes.h/.td`、`PassPipelines.h/.cpp`、`LayoutInfer.cpp`、`MaterializeLayouts.cpp`及公开物化API、`OptimizeLayoutConversions.cpp`；相应现有单测和新增`test/Transforms/layout-boundary-frequency.mlir`及必要Python checker。新逻辑按既有职责放置，不做无关重构。

**Interfaces:** 消费Task 1 helper。`LayoutTarget`新增默认返回Legacy的const虚拟frequency accessor；SM90 factory和已有cost/pass/pipeline factory增加末尾默认Legacy参数，保留旧调用源兼容。CLI在三个入口（infer、optimizer、transactional pipeline）统一解析上述精确两个值；无效值在任何IR写入之前报错。

Task 1复审后的集成澄清：Legacy的循环种类必须与旧`getLayoutLoopDepth`一致，只计`scf.for/scf.while`；Task 2允许窄改`LayoutBoundaryFrequency.cpp`及其测试，使helper的Legacy模式也遵守这个范围，StaticTripCount仍将其他LoopLike作明确proxy。加入非for/while LoopLike回归，避免weighted统计与legacy评分描述不同执行频率。旧weight函数的饱和细节继续原样保留。

新静态权重路径仅修改`conversionBytesAndSync`，检查factor乘法；factor=0不抹掉此前其他字段的saturation。Legacy沿用旧`weightLayoutBoundaryCost(cost, depth)`。转换与clone通过同一target mode走helper。保留loopDepth用于兼容诊断，并为edge记录factor/proxy/saturation；所有新增字段参与计划相等/复验。

统计增加名字明确的`weightedConversionBytes`与其proxy/saturation状态，仅统计唯一Convert leader，不混入clone、同步或code size；trace写出mode与边界频率元数据，保持JSON自有字符串和完整cost_model_id。若修复旧未加权加法溢出，必须单独回归并解释行为变化。solver finalCost仍表示后优化前的选中解，不冒称已重算hoist后的IR成本。

外提在StaticTripCount模式要求严格动态收益：0次和1次循环不新增循环外转换，100次合法register-only转换可由100降到1；保留所有安全条件。Legacy tie行为不变。commute、clone/recompute候选、choice复验必须共享同一mode；不扩展现有8-op纯计算切片范围。

- [x] 添加行为RED：默认等同显式legacy；static 100次只改变动态项；静态0/1不外提；100次安全外提；iter_arg、barrier、shuffle/Shared仍不外提。
- [x] 接入target/cost/choice/optimizer并运行聚焦测试，确认静态资源与code size不乘频率。
- [x] 接入CLI/transaction/pipeline/trace/stats，测试非法值、早期失败模型身份、重跑、回滚及跨模式计划元数据不一致拒绝。
- [x] 每种模型分别运行有限域oracle、partial bound、refined≤unrefined；多consumer只对唯一转换leader计加权流量，原始字节保持含义。覆盖饱和及JSON生命周期。
- [x] Debug/Release运行新增和受影响测试，相关lit；写`task-2-report.md`并关闭所有jobs，独立任务复审。

Task 2过程注：已记录新增CLI和一次循环外提的有效RED，部分其余回归未逐个先于实现运行。上方勾选表示功能、覆盖与独立复审通过，不声称严格逐测试先红。详细时序与初次测试fixture错误保留在实施报告。

## Task 3: 阶段验收与持久交付（控制者）

- [x] 在最终源码上执行一次完整Release单测、受影响Debug单测、相关lit/CTest/API守门，统计真实结果及任何失败修复。
- [x] 小循环、free Attention、fixed-ABI Attention各核对旧基线/default/explicit legacy/new static/replay。无循环Attention应保持物化IR及数值成本；模型ID与新增元数据允许预期差异。循环变化逐项解释。
- [x] 所有构建/测试关闭后串行记录编译开销样本：每种输入/模型1次预热+5次计时，报告原始样本与中位数，不作GPU或统计显著性结论。
- [x] 保存baseline/source hashes、命令、测试输出、原始样本及reviews；同步覆盖表、阶段状态与实施报告。整体复审以真实基线diff为准，修复后只重跑覆盖改动的验证。
- [x] 阶段交付明确P1.3c完成范围和后续最小形状支持未实现；不将完整路线标为完成。

## 总路线的可独立验收任务

1. 本轮P1.3c：静态循环频率与一致决策。
2. 最小形状支持：静态正二次幂中的单位轴、插入轴Broadcast、rank≥2到rank≥1 Reduce、必要静态view；保留rank0/dynamic/ragged边界。
3. GPU基础后端：先搬运/Store、SS MMA、Reduce/Broadcast/math、RS packing切片，再固定ABI Attention整链；实际物化IR驱动生成artifact，验证弱对齐及同步。
4. GPU异步路径：强对齐独立ABI的cp.async和TMA分别验收；循环benchmark需要重叠时，另立缓冲区生命周期与流水验收任务。
5. 联合选择及转换消融：共同候选/后端的至少两种合法执行方案，分别控制局部/联合/转换策略，保留无改善与退化。
6. 成本与资源校准：基于实测拟合/校验排序，完整方案资源复评，明确全局峰值与分量耦合；校准集与验证集分开。
7. 搜索开销优化：依据热点做缓存/去重/安全剪枝，在同预算下复验质量、确定性及编译时间。
8. 冻结与公平对比：固定源码/工具链/输入/算法/ABI/精度/调优预算；分别报告固定schedule机制消融与原生框架调优。冻结manifest不等于自动commit/push。

每项开始前依据已交付接口形成独立执行计划，完成实现、测试、独立复审、报告后交付。后续设计决定由已授权目标和实测证据推进，不为例行实现选择反复要求确认。
