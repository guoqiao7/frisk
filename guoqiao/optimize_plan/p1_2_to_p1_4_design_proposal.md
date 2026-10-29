# P1.2–P1.4：实现审计、阶段划分与契约提案

日期：2026-10-24。**状态：用户确认的 P1.2a→P1.2b 已完成实现与回归、串行测量及源码复审。** 505 unit / 69 lit 与辅助守门通过；范围、原始数据及边界修复见 [实施报告](p1_2_implementation_report.md)，执行步骤见 [P1.2 执行计划](p1_2_execution_plan.md)。P1.3/P1.4 本轮未实施；下文审计结果保留为实施前基线，不把 O04–O11 全部长期目标标为完成。

## 1. 基线与审计结论

以下保留实施前审计：起点为源目录 `main`、HEAD `b237911` 加未提交 P0/P1.1 工作区增量。P1.1 已完成，不重做；其 465 unit / 64 lit 等结果引用[上一轮报告](p1_1_implementation_report.md)，审计阶段没有重新构建、执行测试或 GPU 测量。后续实际实现与新验收结果见 [P1.2 实施报告](p1_2_implementation_report.md)。继续保留已有改动，不创建 worktree、不自动 commit/push。

此前文档仅提及 P1.2–P1.4，未定义对应任务。本提案首次建立与 O04–O11 的明确映射。**完成下述受限阶段，不等于优化清单中每项长期目标均已完成。**

| 优化项 | 现有实现及实际缺口 | 本提案归属 |
| --- | --- | --- |
| O04 候选/MMA 几何 | 四类普通 Distributed 默认候选之外，还有显式绑定、投影及指令候选；MMA 生成、证明、评分均重算同一个几何，尚非多几何联合选择 | P1.2b：有限 grid 搜索；通用 consumer 驱动候选、全 atom 搜索后续另立契约 |
| O05 成本覆盖 | 已有分项版本字符串、sector/bank/Copy/MMA/Reduce 代理；没有独立普通算术计费或完整 MMA Shared 读成本、执行重叠模型 | P1.3a：模型身份、覆盖与算术计数；P1.4d：执行后校准。未经校准不改变全部工作量权重 |
| O06 资源峰值 | 当前是槽位累加、Shared 根跨度去重；不是存活峰值，也不是物理寄存器/occupancy | P1.3b：实际 IR 上的受限存活分析；P1.4d 与后端资源报告核对。全局峰值参与搜索另立契约 |
| O07 编译开销 | P1.1 已改进局部证明，Attention 总耗时基本持平；没有逐函数热点和 cache hit-rate 证据 | P1.2a：可关闭的热点/缓存观测；各阶段保留 CPU/RSS 回归。新增缓存须由测量决定 |
| O08 Copy 契约 | vector/cp.async/TMA 已有证明、候选及单操作物化测试；缺少固定 ABI 异步 Copy→MMA 的明确正例和细分构造失败原因 | P1.2a：集成/诊断；P1.4c：实际异步搬运执行 |
| O09 shape/view | 零位输出表示与 ownership 已部分支持；收集、广播、Reduce 投影仍拒绝单位轴；并非缺少全部基础代数 | P1.4a：非零 rank 的单位维度；ragged Attention、rank-0、动态视图/重解释后续单列 |
| O10 转换/循环 | 已有同块共享转换、至多 8-op 纯切片重计算、受限 register-only for 外提；频率只是 `2^loopDepth` | P1.3c：静态 for 次数与一致计费；不自动扩展通信外提及跨块共享 |
| O11 GPU 闭环 | 已有 rank-one TileLoad→ConvertLayout→TileStore 的可执行测试适配器，不是完整 Attention 后端 | P1.4b/c/d：完整简单搬运链、异步变体、数值/安全/性能闭环 |

关键源码证据（相对本仓库；行号可能随实施变化）：

- [MMA 几何选择及纯证明](../../lib/Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.cpp)：`getSM90MmaGeometry` 选单一 grid；`verifySM90MmaContract` 要求 binding 等于该默认选择（审计时 25–48、198–215 行）。
- [联合支持表](../../lib/Dialect/Frisk/Analysis/InstructionLayoutConstraints.cpp)及[图不变量](../../lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp)：四角色、每角色至多四候选、至多 256 tuples；同 encoding tuple 不允许两个不同 binding，支持查询返回首个匹配。
- [SM90 评分](../../lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp)：MMA 成本重算默认几何；Reduce 工作量仅为源/结果元素数之差（146–157 行），不能声称已计入全部归约通信。
- [目标接口](../../include/Dialect/Frisk/Analysis/LayoutTarget.h)与[求解器](../../lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp)：成本要求硬连通分量可分、非负，partial lower bound 不高估；全函数资源峰值不满足可加性。
- [Copy 契约](../../lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp)已有实际地址、对齐、issuer/completion 校验；构造返回 `FailureOr`，细分失败原因会丢失。
- [广播](../../lib/Dialect/Frisk/Analysis/BroadcastLayoutConstraints.cpp)、[Distributed 收集](../../lib/Dialect/Frisk/Analysis/DistributedLayoutConstraints.cpp)、[Reduce 投影](../../lib/Dialect/Frisk/Analysis/ReductionLayoutProof.cpp)是单位轴支持的主要入口阻塞点。
- [现有转换 lowering](../../lib/Conversion/FriskLayoutToGPU/TestLayoutConversionLowering.cpp)与[编译驱动](../../benchmark/layout/conversion/compile_conversion_case.py)限制 rank-one 测试 ABI；默认结构模式还允许非执行用的 `unrealized_conversion_cast`，不能据此宣称 Attention 可执行。

## 2. 推荐顺序与取舍

推荐“**选择空间 → 决策模型与诊断 → 支持域与真实执行**”，每个子阶段独立验收：

```text
P1.2a Copy 集成 + 开销观测 → P1.2b 有界 MMA grid 联合选择
                                  ↓
P1.3a 模型/工作量说明 → P1.3b 存活资源诊断 → P1.3c 静态循环频率
                                  ↓
P1.4a 单位轴 → P1.4b 简单 Copy 的完整 Attention GPU 链
                         → P1.4c cp.async / TMA 各自验收 → P1.4d 校准与对照
```

这主要是验收顺序，不表示所有子阶段均有算法依赖：Copy 集成和单位轴可以独立；GPU 简单搬运也不依赖单位轴。成本校准则确实依赖可执行路径。

另两种路线：GPU 后端优先能更早取得执行证据，但会推迟推断系统本身的改进；同时改候选、成本和后端会使选择变化难以归因，回归范围也最大。本提案推荐分层推进。

共同约束：

1. 不覆盖显式布局、外部字节地址、对齐、线程数和已有 policy 硬契约；不把 Unknown 当作 Proven。
2. 默认搜索预算仍为 exact=256、beam=32、总 probes=65536、邻域 probes=256；实际地址表仍有 65536 点限制。新计划变量会影响组合数，必须报告，不靠放大搜索预算取得验收结果。
3. 候选在准备阶段生成并冻结，传播只删除；所有选中计划必须物化并由实际 IR 独立重验。文本重放、确定性和事务回滚不可省略。
4. 保留 P0/P1.1 作为历史基线。候选/模型/图结构变化可能使新 trace 或成本变化，须逐项解释；仅观测开关和等价缓存才要求选择不变。
5. 每阶段先 RED/GREEN，再局部与完整回归、独立复审，保存源码/二进制身份和原始样本。新单测必须在 Debug 和 Release 下验证，避免再次以非法属性构造触发 Debug 断言。

## 3. P1.2 契约：可验证的选择空间

### 3.1 P1.2a：Copy 集成与编译开销观测

**Copy 范围。** 不新增搬运指令、不扩展 guarded async；复用既有固定 Global 根契约，在同一 `Copy(A/B) → SS MMA → Store` 链中分别验证 vector、cp.async、TMA。

- 正例起点：A/B 为 `[64,16]f16` 固定行主序，Global 根显式保证 128-byte 对齐，Shared 根 1024-byte 对齐，128 线程，M=N=64/K=16、B 转置。具体 Shared layout 必须同时通过 Copy 地址证明与 MMA descriptor 证明；不能只在两个独立测试中各自合法。
- 三条路径各有明确绑定的集成用例；绑定须由受检 helper 构造完整契约，不伪造 path-only 字典。另有自由选择用例，只要求合法，不强制默认选择 TMA。
- 每个用例验证 Global 逐坐标地址不变、选中 Copy 与 MMA 契约一致、实际物化/解析/重跑成功。此处验收是布局推断，不是 GPU 异步安全。
- 单条件负例：仅削弱 Global 对齐；仅将 view 起点移一项 f16；同线程地址不连续；外层 stride 不满足描述符条件；篡改 issuer/completion/descriptor；writer 重复；错误目标或方向。有合法 fallback 时保留，无合法 fallback 时原子失败。

**诊断契约。** 为 Copy 构造增加带 `ProofStatus`、稳定 reason code、相关端点/坐标的结果；兼容现有返回接口，但不能仅在最后补一条泛化字符串。区分“条件不成立”“缺少事实/decoder 未支持”“证明预算不足”和“合法但未选中”，受现有 trace 容量上限控制。

**O07 观测契约。** 添加默认关闭、单次推断生命周期的 profile：区分 address/execution enumeration、access cost、conversion classification/choice、联合关系查询等调用次数与时间；已有 relation/footprint cache 记录 hit/miss/entry。计时标明包含关系，不把嵌套时间相加；不能为统计而重复求值。观测开关不得改变 IR、候选、完整成本或探测次数。先测两类 Attention 与小图，记录开启额外开销；不在没有热点证据时盲目增加全局缓存。

### 3.2 P1.2b：有限 MMA grid 的显式联合选择

**policy。** 新增显式 `Auto` 入口用于新 `frisk.mma`；默认仍为 `Square`，已有 Square/FullRow/FullCol 含义保持不变。旧 `frisk.gemm` 不新增 Auto 推断承诺，不允许 Auto 意外进入旧规则。显式已物化 `mma_contract` 是硬绑定，重跑不得重新择优覆盖。

**候选。** 首轮只枚举现有目标/类型/线程/shape 支持域内的 grid；每个 grid 仍采用现规则允许的最大 atomN，不枚举全部 atom 组合。单 CTA、128/256/512/1024 线程，grid 数最多 4；不新增线程数选择、tile 搜索或 WGMMA fallback。普通 SIMT 四类默认候选不在本子阶段重写。

**计划身份。** 为 MMA 增加独立 Instruction-kind geometry 变量，旧 policy 只有一个 geometry 候选，Auto 可有多个。约束从“四个数据角色”扩展为“四个数据角色＋一个 geometry 角色”；Copy 仍四角色。SSA 角色 0–3 不变，角色 4 必须校验 anchor、类型和线程环境，不将其当成 Tensor/use 或转换边。

同一 Tensor encoding 不等于同一指令计划；geometry 作为选择身份进入支持 tuple、解、trace 和物化契约，不能靠“首个匹配 tuple”暗中选方案。实际验证从绑定解码 geometry 并证明合法性：Auto 验证合法集合；旧 policy 另验证原 policy 一致性。MMA 成本读取**选中且受检的绑定**，不得继续重算默认 geometry。

**预算。** 保留每角色至多 4 个候选、每操作至多 256 次完整 tuple 证明及 256 个支持 tuple；先按 geometry 证明并筛选兼容的 A/init/result fragment，再枚举可配套的 storage，避免直接展开 `4^5`。兼容性以实际坐标证明为准，不能因为合法显式 seed 的 carrier 顺序或表示不同于生成模板而丢弃它。fragment 预检查与完整 tuple 证明分别计数，拒绝的完整 tuple 也占预算。预算不足报告 Unsupported/Unknown，不丢弃 hard seed、不把截断视为无解。新增单例 geometry 变量也会改变旧图的节点编号/数量，因此不要求跨版本 trace 逐字节相同；旧 policy 的最终布局及数值成本应保持语义一致。

**具体验收。** `M=N=128,K=16,threads=256` 的 SS f16→f32 案例中验证：

| grid | atom | repeats | 现工作量公式的 MMA 项 |
| --- | --- | --- | ---: |
| `(1,2)` | `(64,64,16)` | `(2,1,1)` | 4 |
| `(2,1)` | `(64,128,16)` | `(1,1,1)` | 2 |

两种 geometry 均需独立验证 fragment/descriptor，Auto 的选择必须与同域精确 oracle 的最小完整成本一致。错误 grid 乘积、repeats、packing、descriptor、policy/线程冲突均拒绝。

该例展示“形状启发式与已有成本选择的区别”，**不能冒称为 Reduce 通信导致局部最优反转**。另外验收 MMA→Reduce(dim=0/1)→store、固定 consumer 编码/转换的联合图；小域对照精确 oracle，记录影响选择的真实成本项。若未找到 consumer 导致的严格反转例，明确列为 O04 剩余目标，不为了制造反转临时修改 O05 权重。

## 4. P1.3 契约：把评分口径与实际资源分清

### 4.1 P1.3a：模型身份、覆盖和算术计数

- 增加完整、短小的 `cost_model_id`，与 JSON schema version 分开；截断 trace 仍保留 ID。现有子模型版本不是空白，不重新发明同义机制。
- 发布九项成本的单位与覆盖表；“未建模”不等于“零开销”。当前字典序及各项公式保留为 legacy 模型。
- 在实际物化并优化后的 IR 上，按 opcode/dtype 输出普通算术的逻辑 element-op 计数，例如 `Mul/Exp/Div` 各计算多少元素。每个实际 op 计一次，多 consumer 不重复计原操作；仍存在的原计算和实际新增 clone 分别统计。
- 计数为诊断，不把一个 Exp 与一个 Mul 无校准地当作同等机器耗时，不将诊断加入 `cost-contribution` 求和。MMA Shared 访问、latency/cache/overlap 等未覆盖项目显式列出。
- 验收 trace 截断仍有身份、贡献之和仍等于 finalCost、重复 use 不重复计数、clone/转换实际 IR 与统计一致。只开启诊断时，布局和成本不变。

### 4.2 P1.3b：存活资源诊断，暂不替换搜索目标

第一支持域是无嵌套控制流的单块计算，以及固定单 CTA、`frisk.parallel ranges=[1]` 的 Attention 计算块。跨域 capture 显式作为 live-in；遇到不支持的 if/for/while 输出结构化 `unavailable` 和原因，不能忽略 region 后给出貌似完整的峰值。

指标在转换优化后的**实际 IR**上计算：

- `logical_live_carrier_slots_per_thread`、按 dtype 计算的 carrier bytes；说明采用各值 per-thread 最大槽位相加的保守口径，不是物理寄存器或已实现的 occupancy。
- 每个真实 Tensor SSA 计一次；use 角色、函数返回契约槽、Copy/Fill 执行伪变量不当作常驻 Tensor。执行临时量未建模须标记。
- 操作执行边界同时计尚需读取的输入和已产生的输出，不假设寄存器原地复用；转换输入/输出、共享转换结果、clone 中间值按真实 def-use 区间计入。
- Shared alias 按底层 root 合并使用区间，容量取实际根跨度；同时报告 declared allocation 与 logical-live root bytes。不同根使用时间错开不等于后端已复用 allocation；异步使用无法闭合时保守保活或报告 unavailable。

示例验收：A 的最后使用已结束后才独立定义 B，各 32 槽，峰值 32；同时存活为 64；`B=convert(A)` 在转换边界至少 64。一个转换结果被两个 consumer 共用只计一次。两个 alias 共享 4 KiB root 不能计 8 KiB；两个不同 root 各 4 KiB 即便逻辑生命周期错开，也仍报告 8 KiB declared allocation。

**不得直接将全函数峰值写入 CostVector 第五/六项。** 两个独立分量各有 32 槽峰值、时间不重叠，全函数峰值为 32，分量峰值之和为 64；当前分量独立求解及可加 trace 不适用。旧槽位和也不能作为新峰值的 lower bound。即使把 lower bound 改为零，也未解决分量耦合。因此本子阶段完成资源分析，不宣称完成“以真实资源峰值优化选择”的 O06 全目标。

### 4.3 P1.3c：静态循环次数参与转换决策

新增明确命名、可选择的 boundary-frequency 模型；legacy `2^depth` 保留作为 A/B 基线。采用同一公共 helper 驱动转换选择、clone 边界计费和受限外提检查，避免选择时与改写后使用不同口径。

- 对常量边界、正 step、当前明确支持的有符号 `scf.for` 使用溢出安全的 `max(0,ceil((ub-lb)/step))`。unsigned 比较、动态边界和 while 不猜测，使用明确标记的未知循环 factor=2 代理；它不是次数上界。
- 嵌套已知次数相乘；精确零外层循环优先得到零，即便内层未知或中间乘积很大。其他溢出用既有饱和标记。
- 频率仅乘当前动态边界/clone-byte-work 项；静态 code size 和资源字段不乘次数。原 `conversionBytes` 仍表示每次搬运字节，另增名字明确的加权统计。
- 保持现有 register-only、无副作用、支配与循环不变条件；不把 shuffle/Shared 通信移出循环，不把 iter_arg 当作循环不变值，不扩展 if/while 外提或任意跨块重计算。
- 已知零次循环不因“外提更便宜”增加循环外工作；一次循环成本打平时保留原位置。新的决策版本必须在 trace 中可识别。

验收：100 次循环的边界工作乘 100，合法外提后乘 1；3×5 为 15；3×未知为标记代理的 6；零×未知为 0；覆盖负下界、非整除 step、极大次数/饱和、iter_arg、多 consumer、屏障/副作用、重跑与失败回滚。针对每个模型版本分别验证精确 oracle、partial bound 和 refined≤unrefined，不能直接比较不同版本的数字并宣称改进。

本阶段默认仍为 legacy；新的静态次数模型须显式选择，未来默认值变更另行批准。资源诊断不能暗中作为 tie-break 或否决条件。

## 5. P1.4 契约：受限支持域与真实 Attention 执行

P1.4 是多个后端里程碑，不是补一条测试。每个子阶段单独给出实现计划和证据；通过 P1.4a 不能勾选 GPU 完成，通过简单搬运链也不能勾选 TMA 安全。

### 5.1 P1.4a：单位维度，不扩张成通用广播

允许非零 rank、静态正二次幂形状中的 extent=1；保留 `linalg.broadcast` 的**插入轴**语义、dtype/有序轴和同线程可取规则。不是 NumPy 同 rank 扩张，不支持 rank-0、非二次幂 Tensor、动态 shape 或任意 reshape。

复用 P1.1 零位域表示，修正收集、投影和零行矩阵分支，不放松 GF2Matrix 的非空前置条件。Reduce 首轮允许 rank≥2→rank≥1 的单位轴/单位结果，sum/max/min 的单元素 fiber 为恒等贡献；副本仍只按 first_owner 计一次。rank1→rank0 不纳入。

具体正例：`[1,64]→[1,8,64] axes=[1]`、`[64]→[1,64] axes=[0]`、`[1]→[1,8] axes=[1]`、`[1]→[1,1] axes=[1]`。错误/重复轴、保留维不匹配、线程不一致、覆盖不足仍拒绝。全部逻辑轴为 1、32 个线程副本时，first_owner 合法而 all-writer 必须拒绝。

完整链：固定布局 load `[1,64]`→Reduce(dim=1) `[1]`→Broadcast `[1,64]`→elementwise→store；另测 `[1,1]` Reduce 和合法静态降维 subview 的字节地址不变。小域枚举与符号方法逐例比对，仍保留实际表预算。

### 5.2 P1.4b：简单 Copy 的完整 GPU Attention

第一执行目标忠实消费现有[固定 ABI Attention](../../test/Transforms/attention-layout-fixed-abi.mlir)：一个 CTA、128 线程，Q/K/V 为行主序 `[64,16]f16`，O 为 `[64,16]f32`；SS QKᵀ→scale 0.25→max/sub/exp/sum/div→**P 转 f16**→RS PV→store。没有 batch、mask、在线 Softmax、自动 tile/线程数或流水重叠。

**ABI 不变。** 原根只保证 alignment=1。必须使用安全的弱对齐访问（例如受检 byte load/store 与组装）或确实保留 alignment=1 的后端操作；不能把 CUDA 分配器的实际强对齐冒充 IR 前提。GPU 验收包含偏移 1/2/3 字节的外部缓冲区与保护区。强对齐版本可另建、单独命名，不能替代此弱对齐验收。

**真实 lowering。** 增加独立实验性 lowering/preflight，输入是经实际 verifier 验证的物化 IR，不是只读取 shape 后调用一个固定手写 Attention kernel。优先使用本地 MLIR GPU/vector/arith/math/NVVM 及必要的局部指令发射；实施时先核对安装的 LLVM/NVVM 能力和固定版本 ISA。

lowering 必须消费：

- 实际 Storage map、root-relative 地址和对齐，按物理 footprint 分配 Shared，不把带 swizzle 的 Shared 偷换成普通 row-major。
- 实际 Distributed encoding 与 selected geometry/descriptor/packing；区分 f16 逻辑槽位和 packed 寄存器。每个实际 Tensor op 都有对应执行，不遗留 tensor、Frisk op、unrealized cast 或占位计算调用。
- Copy writer/执行分配、转换计划、canonical reduction tree/first_owner/输出副本分发，以及同线程 Broadcast 的源寄存器选择；`tensor.empty` 不能成为未初始化读取。
- SS/RS MMA 的完成与可见性约束。**简单 Copy 不代表无异步同步需求：WGMMA 自身仍需正确 fence/commit/wait，消费者不能过早读取结果。** 首版串行完成，不做计算/搬运重叠。

先验收 Copy/Store、SS MMA/Store、Reduce/Broadcast/math、RS packing，再验收整条链；切片通过不等于完整算子通过。首个后端只承诺所列 fixed-shape/契约子集，其他合法布局若暂不能 lowering，必须明确拒绝而非重新择优或隐式改变布局。

**数值契约提案。** 固定输入为先舍入至 f16 的有限 `[-1,1]` 值；独立参考包含 FP32 QK、稳定 Softmax、P 的 round-to-nearest-ties-to-even f16 转换、FP32 PV。至少 seeds 0–9，加全零 Q/K、常量、带符号坐标模式、one-hot V 与抵消型 V。运输与孤立 cast 测试 bit-exact；完整 O 的初始门槛为全部有限且 `abs(got-ref) <= 2e-3 + 2e-3*abs(ref)`，保留最大误差及坐标。门槛是待批准设计，不是实测结论；不得见到失败后自行放宽。记录 exp 实现、舍入、subnormal、fast-math/reassociation 口径；通用 NaN/Inf 或更广输入范围不由这组有限输入结果推出。

**安全与身份。** GPU 运行必须加载本次 IR 生成的 artifact；记录 source/input/selected-layout/contract/PTX/cubin/tool/config hashes、完整命令及 launch ABI。检查 O sentinel、边界保护和重复运行；运行 memcheck/racecheck/synccheck，并对初始化读取问题进行 initcheck 或相应明确检查。工具未覆盖某类异步语义时须说明，不能将“未报告”当作完整证明。缺 GPU/工具时阶段记为未验收，不以 CPU IR 测试替代。

### 5.3 P1.4c：cp.async 与 TMA 各自闭环

使用明确强对齐的独立 ABI 变体，保留弱对齐 fallback。两个子阶段分别完成已选 Copy 路径的发射、数据可见性、同步及完整 Attention 数值/安全测试。

cp.async 必须兑现 width、issuer、commit/wait 与 CTA 可见性；TMA 必须兑现实际 descriptor、elected issuer、mbarrier 与完成字节数/可见性，host descriptor 的 setup/ABI 增量显式记录。先做一次性、无重叠执行，不引入多级环形缓冲、warp specialization、cluster multicast 或 TMA 输出 Store。阶段环绕/尾轮测试在未来增加可复用流水后单列，不能以单次 TMA 成功预先宣称流水安全。

### 5.4 P1.4d：成本校准与性能对照

- 在相同算法、P 的 f16 cast、外部 ABI、schedule 和编译参数下，比较至少两种**实际可执行**的合法布局；若某个固定实例只有一种可执行布局，需补充有两个方案的受支持 fixture，不能编造性能差异。
- 报告 GPU UUID、工具链、预热与原始样本、寄存器/Shared/spill、layout inference 与完整同输入→可加载 artifact 的编译时间。单 CTA 小问题可能受启动开销影响，不推广为大尺寸 Attention 吞吐结论。
- O05 校准先检验各分项排序相关性与失配例，再提出成本 v2 权重/模型；这份提案不授权凭猜测给 Exp、TMA 或重叠收益赋任意分值。全局资源峰值进入搜索仍需要单独的可分解性/全局目标设计。
- 功能与性能结论分开：正确执行且无已知安全错误可以通过功能验收；只有公平实测改善才能称布局选择带来 GPU 收益，退化也必须记录。

## 6. 不在本次契约中自动完成的项目

全 atom/MMA 指令族搜索、通用 consumer 驱动候选；所有控制流下的全局峰值资源优化；任意跨块/跨循环通信外提；ragged/动态 Attention 与 masked Softmax；一般 reshape/dtype reinterpretation；自动 tile/线程数/流水级数；warp specialization/多阶段重叠；对 TileLang/Triton 最新版本的性能优越性结论。

这不是将它们从优化清单删除，而是避免将若干受限实现包装成 O04–O11 全面完成。后续状态按“已实现子集 / 已验证子集 / 明确剩余项”维护。

## 7. 维护、验收与下一步确认

长期[实施计划](../layout_inference_implementation_plan.md)、[体系对比](../layout_inference_strategy_comparison.md)和[设计](../layout_inference_design.md)同步登记本契约，用户已确认 P1.2a→P1.2b 实施。按子阶段维护执行计划和报告；保留成功/失败、支持域、开销及对照数据，不覆盖历史记录。P1.3/P1.4 本轮不实施。

本轮未刷新 TileLang/Triton 快照，也未作新上游比较。Frisk 的研究方向仍是可证明的 Storage/Distributed 联合关系、显式指令选择、可核验成本与实际执行契约一致；这些是待消融检验的系统设计点，不自动构成独创性或性能优势证明。

**确认记录：用户已确认，先实施 P1.2a→P1.2b。** P1.3 的资源峰值先作为诊断，P1.4 的 GPU 后端逐子阶段验收；后两阶段本轮不实施。已编写 P1.2 执行计划，不自动提交/推送。
