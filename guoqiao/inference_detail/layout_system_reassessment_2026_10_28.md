# Frisk 布局推断系统能力与研究价值审视

日期：2026-10-28。评估对象为 P1.3b 验收后的 Frisk 新布局推断路径。

**当前系统有扎实的正确性基础和可研究的优化空间，但尚未功能完备，也没有证明相对当前 TileLang、Triton 的 GPU 性能优势。最值得推进的是带合法性约束的跨算子指令与布局联合选择，并把该选择兑现为真实执行收益。**

## 当前判断

| 问题 | 判断 | 依据 |
| --- | --- | --- |
| 能否完成有意义的推断 | 能，在明确的有限支持域内 | Copy、SS/RS MMA、Reduce、Broadcast、转换及固定 ABI Attention 已形成完整推断链 |
| 是否功能完备 | 尚未 | 一般 shape、单位轴、动态情况、完整新路径 Attention GPU lowering 与调度仍有缺口 |
| 是否足够健壮 | 支持域内有较强验证，工业化覆盖仍不足 | 有实际 IR 复验、Unknown 分类、失败回滚、性质/差分测试；549项测试不等于任意输入正确或形式化证明 |
| 理论上能否获得收益 | 有具体机制支持这种可能 | 联合选择可能减少转换、同步、访存和复制；已有消费者改变 geometry 赢家的模型内见证 |
| 是否已有跨框架优势 | 尚未建立 | 完整 Attention 主要有推断/诊断证据；M6 的真实 GPU 证据覆盖受限 conversion，不是完整 Attention 对当前上游的公平对照 |
| 创新是否成立 | 有研究假设，尚不能作独创性结论 | GF(2)、显式布局、成本排序、转换提升及重计算均有相关工作，需要证明具体算法差异和收益 |

## 当前上游基线需要重新认识

本次通过 GitHub 连接器读取的公开快照为 Triton `ff868dc6596ce2e4b476ec2754f27f3444e90c16`、TileLang `714cbf150d90412ca390eeded575dd1c962d960b`，均为2026-10-28获取的 main 提交。这是源码评估版本，不替换原有 M6 差分回归快照，也不表示已构建、测试了新上游。

### Triton 的 Linear Layout 是表示和代数基础

你的理解基本正确。Triton 已将 Linear Layout 用作重要基础，以 GF(2) 映射表示硬件位置与逻辑索引，支撑布局转换及代码生成。但“怎样表示布局”“选择哪种布局”“如何生成高效指令”是不同层次。[Linear Layouts 论文](https://arxiv.org/abs/2505.23819)、[LinearLayout 接口](https://github.com/triton-lang/triton/blob/ff868dc6596ce2e4b476ec2754f27f3444e90c16/include/triton/Tools/LinearLayout.h)

当前 `RemoveLayoutConversions` 有锚点传播、冲突消解、重计算和转换提升。所审阅的 `resolveConflicts` 分支仍按操作偏好 encoding 作启发式选择。这为研究特定跨算子决策留下空间，但不能据此说整个 Triton 没有全局优化。[固定源码](https://github.com/triton-lang/triton/blob/ff868dc6596ce2e4b476ec2754f27f3444e90c16/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)

Linear Layout 的内部二进制域不等于 Triton 只能运行二次幂输入。外部问题可以借助分块和 mask 处理尾部；官方向量加示例就覆盖非整块长度。因此不能把“支持一般输入形状”笼统当作 Frisk 对 Triton 的独有优势。[官方示例](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)

### TileLang 已有完整方案评分

当前 TileLang free-mode 按连通分量尝试不同推断根，比较完成传播的布局方案。成本接口有默认 register-count 和可选 `io-aware`；后者比较访存语句的带宽与发射开销代理，代码还考虑特定寄存器数组降为 local memory 的流量。[成本接口](https://github.com/tile-ai/tilelang/blob/714cbf150d90412ca390eeded575dd1c962d960b/src/transform/layout_inference/layout_cost_model.h)、[成本实现](https://github.com/tile-ai/tilelang/blob/714cbf150d90412ca390eeded575dd1c962d960b/src/transform/layout_inference/layout_cost_model.cc)、[推断实现](https://github.com/tile-ai/tilelang/blob/714cbf150d90412ca390eeded575dd1c962d960b/src/transform/layout_inference/layout_inference.cc)

比较应落到具体搜索空间、约束、目标函数和结果。不能继续以“TileLang只传播，Frisk才有成本或联合选择”为研究前提。

### 单靠布局代数不足以确立创新

除了 Triton 的 GF(2) 工作，已有整数集合关系统一 CuTe/Triton 布局的研究，也有2026年10月提出的区分 logical、physical、thread-value 空间的布局代数。Affine 与 BitLinear 的组合有工程价值，但单凭这种组合不足以证明没有相近工作。[整数集合关系研究](https://arxiv.org/abs/2511.10374)、[布局代数新工作](https://arxiv.org/abs/2610.13679)

## Frisk 已经做对的部分

Storage 与 Distributed 分离，让存储地址、Tensor 线程持有者、指令执行角色分别表达，再通过指令契约连接。Copy 的 vector/cp.async/TMA 有显式候选，MMA geometry 也进入联合支持关系；可以表达搬运、计算和消费者之间的兼容性。

硬约束、支持表、有界求解、实际 IR 重验及事务回滚构成清晰的正确性流程。预算不足、无法证明、确实冲突分别报告；显式 ABI、同根 alias 和复制所有权也有独立约束。

P1.3b 把实际资源观测与旧评分分开。两类 Attention 均报告16个 Tensor SSA、3个 Shared root、每线程106逻辑槽/424字节峰值和6144字节 Shared 活跃峰值，并从最终 IR 的 def-use 独立核对。这是逻辑诊断，不是物理寄存器或 occupancy。

这些是扎实基础；组合成研究贡献，还需新的决策能力及可归因的执行收益。

## 功能与健壮性的剩余边界

| 范围 | 当前限制及影响 |
| --- | --- |
| Tensor/Broadcast 形状 | 生产规则仍要求非零rank、静态且每轴大于1的二次幂形状；P1.3b 的规范 region 特例只补资源诊断，没有交付一般单位轴广播 |
| 动态与 alias | 有些静态可恢复 view 能证明；一般动态边界、无法证明的 alias 和 dtype-changing view 未完整支持 |
| MMA 搜索空间 | Auto 最多4个 grid，每个 grid 按现有规则选一个 atomN；不是全 atom/tile/线程/流水级数搜索 |
| 指令联合证明 | 每个角色候选数目前不能超过4；扩大候选会触及证明和编译预算 |
| 近似搜索 | exact阈值256、beam宽32、展开预算65536、邻域探测256；大域不保证最优，有限域最优也不是硬件最优 |
| 控制流 | 有布局传播和region关系；资源峰值诊断主要覆盖单块或固定单CTA域，两者不能混淆 |
| GPU 执行 | 有受限 conversion 的真实验证；完整新路径 Attention 的 WGMMA/TMA、同步、数值与安全闭环仍待完成 |
| 资源驱动选择 | SSA峰值尚不参与评分；非可加全局峰值不能直接放进当前分量独立求解器 |
| 编译开销 | 最新主机观测中，自由/固定ABI Attention 的整个Release CLI中位约13.36/7.48秒，是CPU编译开销，需要按热点和收益审视 |

健壮性应表述为“在明确支持域内有较强验证”。549项 Release、50项 Debug、13项 lit，加上性质/差分和回滚验证，提供有价值的证据；最近通过零region Broadcast 反例修复入口漏洞，也说明边界测试仍有必要。

下一轮应针对 shape/rank/dtype、复制所有权、偏移/步长/降维 alias、多consumer、循环携带值、Unknown/预算、饱和算术、文本重放和失败后状态做组合验证。用独立小域oracle、变形性质和失败缩减检验；域外输入稳定拒绝或明确Unknown。随后把CPU布局验证接到最终GPU数值、越界、未初始化读取和同步安全检验。

## 当前成本较低为何不保证更快

九项 `CostVector` 是字典序代理，不是执行时间。SM90首项混合 Copy设置/发射、MMA几何重复次数和Reduce元素差；首项的一次改善优先于后续访存、转换和资源项。符合这个排序不等于排序能预测运行时间。

普通 Exp/Div/Mul 时延、完整 MMA Shared 访问、cache、物理寄存器与spill、occupancy和重叠未建模；循环转换仍按 `2^depth` 而非真实trip count计费；资源字段是槽位和及root容量和。P1.3b 的存活诊断没有自动修正这些评分机制。

现有选择质量实验有积极信号：36个消费者图出现3个严格geometry反转、6个平局消除。但原生Store组Auto与FullRow打平，C1/C2的Auto CPU编译耗时约为FullRow的2.25/2.24倍。这支持继续研究消费者影响，也要求计入选优代价，不能提前写成GPU加速。

## 理论收益最可能来自融合链

优先寻找局部最优与整图最优不一致的场景：

```text
MMA → Reduce → Broadcast → Elementwise → MMA
```

一个便于单次MMA的geometry可能让下游需要跨warp转换；另一种增加少量局部工作，却减少整图搬运、同步或复制。多consumer共享转换、廉价重计算、循环内重复转换也有类似空间。

量级示例：64×64xf32完整tile若通过Shared各写读一次，每次约32KiB逻辑流量；64次重复约2MiB。消除这样的转换存在收益机制，但具体指令、同步和时间须由真实lowering/GPU测量确认。增加重计算或延长其他值存活也可能抵消收益。

对已被成熟实现很好映射的算子，或硬约束下可执行布局几乎唯一的情况，新的求解器未必有更好的方案。合理目标是先在明确融合、多consumer或循环工作负载上建立优势，再检验推广性。

## 值得聚焦的研究贡献

| 候选贡献 | 已有基础 | 需要补的内容和证据 |
| --- | --- | --- |
| 跨算子指令与布局联合选择 | Storage、Distributed、Copy路径、MMA grid与转换计划已有关联 | 给出相对逐根传播/固定锚点的算法差异，在共同合法域展示可执行的赢家反转与GPU收益 |
| 面向通信和资源的联合目标 | 有访问成本、转换计划和SSA/root存活诊断 | 处理全局峰值与分量耦合，以实际register/scratch/指令校准排序，不能把106直接替换旧第五项 |
| 选优到执行的可核验一致性 | 有支持表、契约、实际IR复验、trace与回滚 | 将所选布局、descriptor、writer、转换对应到生成artifact，明确合法性保证与近似成本的区别，再做GPU差分/安全验证 |

这些仍是研究假设。GF(2)、候选、beam、成本排序、重计算、hoist、MLIR或日志更详细，单独存在都不足以作为独有创新。最有说服力的路线是让上述联合决策真实可执行、可解释，并经消融证明收益。

## 性能实验应区分收益来源

先做两层比较：

1. **同候选、同后端的机制消融。** 在Frisk内部固定ABI、合法候选、tile/线程/schedule和后端，对比固定局部geometry、联合选择、转换共享/重计算等策略，归因到算法。
2. **原生框架对照。** 固定当前Triton/TileLang提交，使用合理优化配置及相同调优预算；TileLang报告默认和io-aware。统一算法、dtype/精度、外部布局、输入范围与计时；原生最优结果与固定schedule消融分别报告。

优先测试MMA→Reduce/Broadcast→MMA、多consumer、循环重复转换、固定外部布局搬运/转置。保留成功、无改善、退化和不支持。单CTA与足量独立tiles/多CTA分开，防止启动开销和GPU利用不足掩盖布局效果。

每例记录正确性/安全、GPU延迟分布、转换/同步指令、register/Shared/scratch/spill、完整编译时间及调优预算。以可执行候选中的实测最优作有限参照，分别检查模型选错、搜索未找好、候选没覆盖好解、或lowering未兑现计划。报告支持率、几何平均加速、最坏退化与噪声区间，不只展示赢家。

若改Global布局，调用方重排/适配成本须纳入端到端结果。也不能把当前单tile显式Softmax与不同online算法或精度的FlashAttention直接比较，再归因于layout inference。

## 建议推进顺序

下一阶段主目标应是：**证明跨算子联合选择在真实GPU执行中产生可归因收益。**

先收尾受限P1.3c静态循环频率，补齐后端验收所需的最小单位轴支持；再将固定ABI Attention的实际物化IR打通到受限正确GPU执行。先简单Copy及必要MMA/Reduce/Broadcast同步，随后分别验证cp.async、TMA。在至少两种合法可执行布局上做消融，再依据失配案例调整成本和资源目标，最后扩展候选、一般shape和流水调度。

不必等通用功能全部齐备才验证性能假设，但必须有真实执行和公平对照。没有胜过基线时，应按结果收缩主张或改进方法，不能靠扩大静态分数差、放宽数值门槛或更换输入制造优势。

## 依据与范围

Frisk依据是P1.3b当前脏源码及314文件身份，不是仅Git HEAD。主要位置：`LayoutCostModel.cpp:221,236,353` 的排序/频率/槽位代理；`SM90CostModel.cpp:149` 的工作代理；`LayoutSolver.h:49` 的预算；`InstructionLayoutConstraints.cpp:43` 的支持表边界；`SM90CopyConstraints.cpp:19` 的路径候选；`SM90MmaLayoutProof.cpp:55` 的Auto几何；`BroadcastLayoutConstraints.cpp:15` 的形状限制；`LayoutIRResources.cpp:349` 的未建模资源标记；`SM90LayoutTarget.cpp:295` 的下界；`TestLayoutConversionLowering.cpp:17` 的测试适配器边界。文件均在服务器 `/home/baopeihua/frisk` 下。

本次是源码和既有证据的重新评估，结合当前上游源码及论文；没有新跑GPU对照、广域fuzz或全量回归。P1.3b验收保持成立，本报告不扩张其支持域，不修改现有评分、回归快照或已批准门槛。
