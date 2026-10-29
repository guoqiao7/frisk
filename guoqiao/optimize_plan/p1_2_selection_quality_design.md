# P1.2 选择质量实验设计与 P1.3 衔接

日期：2026-10-26。状态：方案A的Q1/Q2实现、230行正确性矩阵、48次串行计时及独立最终复审完成；结果见[选择质量报告](../inference_detail/p1_2_selection_quality_report.md)。按[执行计划](p1_2_selection_quality_execution_plan.md)继续P1.3a→b→c，以下设计保留实验前契约。3个严格反转、6个平局消除是本次有限域结果，不推广为GPU性能结论。

## 1. 目标、基线与分阶段交付

P1.2 已有合法性、精确小域、重放和 profile 验收，不能把重跑它们当作新的选择质量研究。本轮要回答：

1. Auto 相对 Square/FullRow/FullCol 在哪些输入上改变 geometry 和完整代理成本？
2. 加入下游后，联合选择是否优于先固定局部最优 geometry？改善来自候选覆盖、布局协调还是搜索差异？

当前源目录 main，HEAD `b237911`，包含未提交 P0/P1.1/P1.2。起点能力及哈希参考 [P1.2 报告](p1_2_implementation_report.md)。开始实验前重新记录实际源码、配置、工具身份并保存快照；不能用 HEAD 单独冒充完整基线。直接在源目录工作，不创建 worktree、不自动 commit/push，保留原有修改。

交付顺序固定为：

1. Q1/Q2 实验驱动、可复现数据和 `guoqiao/inference_detail/p1_2_selection_quality_report.md`。
2. P1.3a 模型身份、覆盖说明及实际 IR 算术计数，独立设计细化、实现、实验与复审。
3. P1.3b 实际 IR 存活资源诊断，同样独立验收。
4. P1.3c 显式选择的静态循环频率模型，同样独立验收。

P1.3 沿用 [既有阶段契约 §4](p1_2_to_p1_4_design_proposal.md#4-p13-契约把评分口径与实际资源分清)，但不把三项尚未实现的能力预先算入 Q1/Q2 的结果。P1.4/GPU lowering、本轮之外的成本权重修改、自动 tile/线程搜索不纳入。

## 2. 方案选择

**方案 A（推荐）：原生策略对照＋同域受控实验。** 原生生产 pipeline 比较四种 policy 的实际行为；另在同一 Auto 冻结候选域上，仅限制 geometry，分离候选生成和联合选择的影响。小域增加独立穷举，报告搜索缺口。工程量较大，但结论可归因。

**方案 B：只做四种 policy 的原生对照。** 驱动较简单，能形成选中布局/成本/耗时表，但各策略候选域不同，无法据此证明 geometry 联合选择本身的收益。也不能严格区分消费者作用和搜索截断。

以下具体契约按方案 A 编写，不预设 Auto 总是更优，也不以“必须找到反转”为完成条件。

## 3. 审计发现及方法约束

- [MMA 几何](../../lib/Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.cpp)：旧 policy 选择单例，Auto 最多四个 grid；每 grid 取现有最大合法 atomN，RS 还有复制后的硬件点数限制。
- [已有实验原型](../../unittests/Dialect/Frisk/Layout/MmaAutoIntegrationTest.cpp)：已有四 policy、小域遍历、SS/RS 和两轴固定 consumer 用例，但不是参数化选择质量报告。部分孤立 MMA 测试没有 observable output，不能直接复制为完整 pipeline 的测量输入。
- [目标评分](../../lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp)：MMA 首项为 `gM*gN*mRepeat*nRepeat*kRepeat`；Reduce 的工作量项为源元素数减结果元素数，并未完整刻画所有归约通信。
- [转换选择](../../lib/Dialect/Frisk/Analysis/LayoutConversionChoice.cpp)与[通用评分](../../lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp)：完整 assignment 必须累计 target cost 和 conversion/rematerialization choice cost，各一次。不能遗漏转换，也不能重复计费。

当前成本按九项字典序比较，不是加权和。128×128×16、256线程的两个 grid，其 MMA 首项为 4/2；仅减少第四项转换代价不能抵消已存在的首项差距。加入 128×512×16、256线程：

| grid | atom | repeats | MMA 首项 |
| --- | --- | --- | ---: |
| (1,2) | (64,256,16) | (2,1,1) | 4 |
| (2,1) | (64,256,16) | (1,2,1) | 4 |

此例更适合检查其他成本项能否影响选择，特别是 RS 复制和固定 consumer 转换；这只是由源码公式得到的实验假设，不是已测到的反转。Store 等其他首项贡献也必须核对，不能仅凭 MMA 项打平就宣称全图首项打平。

## 4. Q1：原生四策略与同域 geometry 对照

### 4.1 有限矩阵

固定 SM90a、f16→f32、transA=false/transB=false、单 CTA。下表每项分别测试 SS 和 RS，再比较四种 policy，共 **64 个原生配置**。线程数是实验输入，不是优化器搜索结果。

| ID | M | N | K | threads | 目的 |
| --- | ---: | ---: | ---: | ---: | --- |
| C0 | 64 | 64 | 16 | 128 | 单 geometry 控制组 |
| C1 | 128 | 128 | 16 | 256 | 已有 4 对 2 工作量例 |
| C2 | 128 | 512 | 16 | 256 | 两 grid 的 MMA 首项相等 |
| C3 | 256 | 128 | 16 | 256 | 另一方向矩形 |
| C4 | 128 | 128 | 32 | 256 | K repeats 变化 |
| C5 | 256 | 256 | 16 | 512 | 较大结果及更多 warpgroup |
| C6 | 512 | 128 | 16 | 1024 | 四 grid 有限域 |
| C7 | 128 | 128 | 16 | 512 | 线程数与可用 grid 子域 |

不预先把全部配置判为可支持。每个 manifest 行都必须保留，报告实际成功、硬冲突、Unsupported、Unknown/证明预算、搜索预算等状态；不能静默删掉失败行或用较小 shape 替换。只有可证明的硬空域才叫无解。

### 4.2 输入及费用边界

主图是 `MMA → Global Store`，输出逐坐标固定行主序 f32、alignment=128；SS 内部 Shared 根 alignment=1024，RS 的 A 为逻辑 Tensor。优先复用已有 SS/RS 测试的输入构造并在 manifest 中记录具体 producer，成对比较时不得更换 producer、初始化、硬 seed 或对齐。MMA 必须保持 live，不能因没有真实 Store 而被消除。

这是 **Shared/Tensor 输入已经准备好的 MMA 子图**。Shared 内部布局仍可选择；输出 Global ABI 固定。没有把 Global→Shared 搬运加入目标，也不证明这些静态 fixture 可直接运行 GPU 数值测试。结果不能外推为完整 Attention 或把省略的输入搬运费用称为零。现有 Copy→MMA 固定 ABI 测试继续作为回归，不与本表混为一种输入契约。

### 4.3 两层比较

原生层分别运行四 policy 的生产 pipeline；记录候选/支持表摘要、实际绑定、完整成本、传播/搜索/松弛统计、转换及实际 IR。各 policy 候选域不同，不宣称 Auto 域必然包含所有旧策略域。

受控层先对相同输入的 Auto 准备一次有限候选和支持表，记录候选身份及 proof 元数据，再建立受限快照：

- `joint`：允许全部已准备的合法 geometry；
- `geometry=g`：只允许某一个 geometry，其他数据角色的初始候选不变；相关支持 tuple 同步限制，并重新检查图不变量、传播与求解。

冻结域限制只存在于实验驱动，不发明生产 IR 的“geometry-only”属性。不用完整 `mma_contract` 代替该限制，因为它还会固定 descriptor/packing 等其他选择。每个成功解按现有接口物化成完整真实 binding，再独立实际 IR 验证、打印/解析和重跑。不会把某个受限分支的更优结果冒充生产 Auto 原本返回的结果。

如果原生 Auto 需要扩展/松弛，而受控图尚不能忠实复现其成功候选域，标记该层 `not-comparable`，保留原生结果；不得悄悄混用不同 attempt 的域。支持表本身超预算时不能拿残缺表做实验。

## 5. Q2：局部选择与消费者联合选择

选取 C1/C2/C3，各测试 SS/RS，共六个基础输入。每个输入设置：

- `U0/U1`：MMA→Reduce(sum, dim=0/1)→Store，消费者编码自由。
- `F0(g)/F1(g)`：MMA→显式 ConvertLayout 到固定消费者编码→Reduce(sum, dim=0/1)→Store。固定编码依次取**实验前**生成且验证的各合法 geometry 的 canonical result fragment；按稳定 geometry 身份排列，不根据运行赢家挑选。编码等价的行可合并执行，但 manifest 保留映射。

这些基础输入最多两个 grid，因此消费者矩阵最多 **36 行**。每行的 Global 输出固定与 shape 对应的行主序 f32 和 alignment=128。固定的是消费者输入，不是 MMA 结果；显式转换须计费。dim=0/1 的不同输出图不能直接相减成本来声称改善。

### 5.1 局部基线的定义

实施时确认的入口边界：`ConvertLayoutOp::verify` 要求源/目标都有 DistributedEncodingAttr，而固定消费者实验故意不绑定 MMA 源结果；`NormalizeLayoutIR` 又在推断前执行验证。因此 F0/F1 的原始输入是 **pre-layout analysis fixture**，不能称为可直接输入完整 CLI 的未绑定程序。该组按本节受控实验路径调用现有 collector/solver/materializer，得到完整绑定后，必须通过独立 actual IR verifier 和完整流水线重放；保存原始分析输入与物化后流水线输入，明确区分。不会隐藏预绑定步骤来冒称 native Auto，也不修改生产验证器。Q1 的64个 native Store 图及计时组不受影响。

先在孤立 MMA 分析图上求局部最优 geometry 集合 `G_local`，使用完整的本地九项成本并保留所有打平的最优 geometry。孤立图只作局部分析，不作为会经过 DCE 的完整 pipeline 时间样本。

局部基线在完整消费者图中**只固定 geometry**，仍允许其他布局重新优化。这是“先决定 geometry、再尽可能优化下游”的基线，不冻结整份局部绑定。局部/完整图的费用范围不同，只比较选中 geometry 集合；真正的成本差必须在同一个完整图上比较。

每个消费者图运行全部 geometry 的受限分支与不受限 joint，且全部来自该消费者图自己的同一冻结域。若局部选中 geometry 不在该图的证明域中，单列“局部决策不可延伸/比较域不同”，不计作成本反转。

### 5.2 结果分类

1. **严格 geometry 反转**：局部与完整图均有完整 oracle；局部最优集合和完整图最优集合不相交，且 joint 的完整图成本严格小于所有 `g∈G_local` 的完整图最优成本。
2. **局部打平后由消费者区分**：局部多个 geometry 打平，消费者只区分它们；不称严格反转。
3. **同 geometry 下布局/转换变化**：报告布局、转换与成本差，不冒充 grid 反转。
4. **未观察到改善/反转**：作为有效结论，列出覆盖范围和首个决定排序的成本分量。
5. **仅启发式差异**：oracle 不完整或默认搜索有 gap，只报告已找到的方案，不能判定全局最优或严格反转。

不更改权重、字段优先级、候选上限或证明预算来制造结果。对字典序向量不计算一个虚构的“总成本降低百分比”；给出全向量及首个不同分量。

“同 geometry 下布局/转换变化”只比较实际选中同一个grid的方案。若完整成本相等，明确称为打平绑定变化，不自动归因于消费者收益；另一个grid的绑定不同本来就是预期现象，不能用来证明该分类。

## 6. Oracle、预算与正确性门槛

生产预算保持 exact=256、beam=32、总 probes=65536、邻域 probes=256、每角色最多4候选、每 op 最多256次完整 tuple 证明、实际枚举最多65536点。

实验 oracle 在同一冻结域上独立遍历 assignment、核对 hard relations，并调用实际目标评分和转换选择各一次。不复用生产搜索的排序/剪枝来假称独立搜索 oracle；它校验当前模型下的搜索结果，不是独立验证成本模型符合硬件。

这里的域包括生产 collector 已准备完成的 **Proven 支持表**。完整 oracle 只证明该有限已证明支持域内的最优/无解，不证明被解码器排除或仍 Unknown 的指令替代方案在硬件上不合法。支持表预算耗尽时，生产实现清空残缺表并失败，不能拿该残缺表继续实验；表准备成功也不等于所有未收录方案都已被证明不可能。报告必须明确这一口径。普通 unary/pair 关系遇到 Unknown 仍标 proof-incomplete，不将它改成 Disproven。

为控制成本，oracle 每个受控图最多访问65536个递归节点、评估4096个完整合法 assignment，另有单次120秒的安全超时。任何上限触发都标记 `oracle-incomplete`，保存已访问数量和当前最好值，但不得称其为最优或穷尽无解。未触发上限也必须确认遍历确实结束才能标记完整。至少 C1 的 SS/RS 直接图及 SS 两轴自由/固定消费者核心组要获得完整 oracle；核心未完成则报告阶段未验收，不私自扩大预算。

核心验收按实际比较行展开：8个C1完整图各有joint＋两个geometry受限分支，共24个oracle行，加上C1 SS/RS两个local基线，共 **26个oracle行**；另核对两个native Auto输入的域身份与成功证据。不能只检查8个joint而遗漏其比较基线。正式correctness/report应明确显示通过或“未验收”及缺失行，核心未完成时返回失败；smoke和单独timing只表示诊断/采集完成，不冒充整阶段验收。

默认求解器与 oracle 比较时：完整 oracle 成本是下界与最优证据；默认 beam 若更差，报告真实 search gap，不用多次重启取最小值遮蔽。若默认成本低于已证完整 oracle、出现硬约束非法、成本漏计或 actual verifier 失败，则是正确性问题：保留失败输入、先修复和复审，再重跑受影响矩阵。不得将这种情况作为普通不支持样本通过验收。

测试覆盖 manifest 完整性、政策/域限制隔离、成本只计一次、局部打平分类、oracle 截断不冒充完整、异常/超时记录、结果 JSON 校验及确定性。所有完整图的成功配置均须真实物化、逐坐标外部输出地址核验和完整 pipeline 重放；孤立局部分析另标 `local-analysis-only`，不计为 pipeline 成功。不只验证手工合成的成本表。

## 7. 数据、计时与报告

拟新增实验专用 C++ harness（沿用 `benchmark/layout/` 结构），复用生产 collector/solver/materializer/verifier；Python 驱动负责固定 manifest、独立进程、超时、保存产物和汇总。实验统计/schema 属于 harness，不为方便实验修改生产模型。新增测试走 RED/GREEN、Debug/Release；源码和报告独立复审后关闭阶段。

每行保存输入与输出 IR、完整绑定、geometry 候选及身份、冻结域/支持表摘要、九项总成本及贡献、转换/重计算计划、实际转换计数、证明/搜索统计、typed failure、oracle 覆盖及状态。支持表较大时保存独立文件与摘要，不能只保存选中赢家而丢掉比较依据。JSON 保留 source/config/input/binary/artifact hashes 和实际命令，报告由数据生成后逐项核验。

正确性、带 trace/profile、oracle 与计时分开运行。串行 Release 性能组先限 C1/C2 的 SS 四 policy，共8配置，各1次预热＋5次记录；采用完整 `frisk-opt` CLI 的 wall time 和峰值 RSS，包含解析、生产 pipeline、IR 输出与必需 stats 发布，不包含另跑的 oracle、trace/profile 或报告汇总。重复语义样本须保持 IR/计划稳定。其他 manifest 行的单次用时仅作诊断，不排名性能；没有 CPU 隔离/随机化时不据小幅差异宣称显著加速。

最终报告必须包含：64行原生矩阵的成功/拒绝分类、消费者矩阵覆盖、至少一个完整逐项决策例、同域和异域比较的区别、默认搜索与 oracle 的差异、反转/打平/无改善的真实分类、CPU 开销及剩余限制。无 GPU、无新增 TileLang/Triton 测量，不以 Frisk 内部实验声称跨框架领先。

## 8. P1.3 的后续执行边界

Q1/Q2 报告验收后再进入 P1.3，先细化该子阶段的接口、数据口径与测试计划，不同时改三个子系统。

| 阶段 | 实现边界 | 核心实验 |
| --- | --- | --- |
| P1.3a | cost_model_id/覆盖表；实际 IR opcode/dtype element-op 诊断，不加入排序 | Attention Mul/Exp/Div，多 use不重复计原op，clone据实际IR计数；诊断开关不改布局/成本 |
| P1.3b | 支持域内真实SSA存活与Shared-root合并；诊断不替换可加目标 | 各32槽的值错开/重叠；转换边界输入输出同时存活；两alias共4KiB；逻辑live与declared分开 |
| P1.3c | 显式开启的静态for频率；legacy默认保留 | 0/1/100次、3×5、未知/溢出、合法外提；0次不能增加循环外工作，模型间不直接比较数字 |

Q1/Q2 的结果可暴露模型缺口，但不能授权临时更改 P1.3 契约或扩大 P1.4。发现需要新语义或不可加目标时单列设计，不把实验调参当成修复。

## 9. 设计检查与当前状态

- [x] 核对当前代码、P1.2 已完成证据、P1.3 契约和用户指定顺序。
- [x] 提出 A/B 路线；按 A 写明有限矩阵、比较域、成本口径、失败类型与预算。
- [x] 自审：区分子图与完整输入搬运、geometry与完整binding、局部打平与严格反转、oracle与真实硬件性能；不修改默认模型。
- [x] 用户审阅并确认本文具体契约（2026-10-26）。
- [x] 使用 writing-plans 形成逐任务执行计划。
- [ ] 实施 Q1/Q2 并独立验收报告。
- [ ] Q1/Q2 报告验收后，按 P1.3a→b→c 分阶段推进。

已保存当前脏源码/Release 工具基线；按执行计划推进驱动、测试和测量，不预先视为通过。未实施 P1.3，未进行 git 提交或推送。
