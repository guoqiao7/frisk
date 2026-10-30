> 2026-10-29 P1.4c补齐cp.async/TMA完整单CTA Attention执行证据：同一Frisk决策/受检发射和Triton低层编译框架，强对齐路径与弱对齐基础分别验收；保留算法/精度/外部地址约束。功能与安全通过，单次无重叠；未宣称性能优势，后续联合选择/转换消融、校准及公平对比尚未完成。见[报告](inference_detail/p1_4c_implementation_report.md)。

> 2026-10-29 P1.4b已建立受限固定ABI Attention真实执行证据：Frisk保有布局/契约决策，通用逐操作标量发射复用Triton低层编译；并未证明所有Frisk映射可直接转为Triton原生Tensor编码。功能与安全通过，跨布局性能消融、成本校准和公平跨框架比较仍未完成。见[报告](inference_detail/p1_4b_implementation_report.md)。

# Triton、TileLang 与 Frisk 布局推断对比及 Frisk 架构决策

> 2026-10-29：P1.3c已完成验收，完整源码、驱动、数据与文档独立复审通过。默认legacy保留；显式`boundary-frequency=static-trip-count`统一转换/clone与受限外提的频率口径。Release主单测570、选择质量20，相关Debug/lit及辅助守门通过；18次语义核验与36次串行CPU调用通过，身份复验不变。下一独立任务为最小形状支持，GPU和跨框架比较尚未实施。见[阶段报告](inference_detail/p1_3c_implementation_report.md)。

> 2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。 见[完整Attention资源报告](inference_detail/p1_3b_attention_resource_report.md)。

> 历史检查点（P1.3b，2026-10-27，已由2026-10-28结果取代）：纯资源分析器与通用pipeline接入已通过独立复审，按真实SSA存活区间统计逻辑carrier峰值，Shared别名按root合并并区分声明容量/逻辑活跃字节；不改变可加成本目标，不代表物理寄存器或occupancy。Debug/Release定向各87项、lit各4项及全Release545项通过。两类Attention仍成功推断且IR不变，但完整资源验收尚未完成（Broadcast隐式region特例待确认）。未刷新上游比较快照或GPU数据，不能据此声称跨框架性能优势。见[阶段报告](inference_detail/p1_3b_implementation_report.md)。

> P1.3a（2026-10-26，已验收，最终独立复审通过）：新增完整模型身份 `sm90-layout-lex9-legacy-v1`、九项单位/覆盖说明，以及实际优化IR按opcode/结果dtype计数的默认关闭诊断。两类Attention的Mul/Sub/Exp/Div/Trunc各4096 element-op；旧基线和开关对照保持IR/完整成本/非计时求解统计一致。这里新增的是“评分代理与实际IR工作量的分离及可核验观测”，不是新性能权重、物理资源估计或独创性证明。未刷新上游快照，未运行GPU。见[报告](inference_detail/p1_3a_implementation_report.md)。

> 2026-10-26（方案 A 的Q1/Q2已验收，独立最终复审通过）：[选择质量报告](inference_detail/p1_2_selection_quality_report.md)完成230行正确性矩阵和48次串行计时。原生四策略64/64成功，Auto对Square有9/16组成本更低、对FullRow16/16打平；共同冻结域的消费者实验出现3个严格geometry反转、6个局部平局消除、27个未观察到改善。C2 RS自由Reduce反转由结果载体槽位代理导致，固定消费者反转由转换代理导致，不称为已测GPU通信收益。165个完整oracle无默认搜索gap；C6 RS joint的120秒截断单列、不称最优。Auto在C1/C2的编译耗时约为FullRow的2.25/2.24倍。没有刷新上游快照或GPU数据，不据此声称独创性/跨框架性能优势。现在继续P1.3a→b→c，不将其能力预先算入本轮实验。

> P1.2（2026-10-24，已完成）：Frisk 增加可审计 Copy→MMA 集成/细分诊断、调用级 profile，以及显式 geometry 身份参与联合支持表、解、绑定评分与实际 IR 重验的有界 Auto grid 选择。505 unit / 69 lit 与辅助守门、独立源码复审通过；三类输入的 legacy 最终 IR/完整成本不变，profile 开关等价，原始 CPU 样本见 [实施报告](optimize_plan/p1_2_implementation_report.md)。这验证了 Frisk 内部“计划身份—联合证明—成本—实际绑定”的一致性，不把 Auto 搜索或缓存观测本身称为独创；未展示 consumer 导致赢家反转，未刷新 TileLang/Triton 快照，未做 GPU 性能或跨框架消融。P1.3/P1.4 尚未实施，下方提案保留为审计历史。

> P1.2–P1.4（2026-10-24，待确认设计）：本轮只读审计 Frisk 后提出[分阶段契约](optimize_plan/p1_2_to_p1_4_design_proposal.md)，未刷新 TileLang/Triton 快照、运行新测试或取得 GPU 数据。拟研究“显式指令计划与数据布局联合选择”“可核验代理成本与实际存活资源分离”“从证明契约到真实执行的一致性”；这些是待实施、待消融检验的设计方向，不是新能力、独创性或性能领先的既成结论。尤其不能以添加候选替代修改默认几何验证，也不能将非可加全局峰值直接放进当前分量求解器。

> P1.1（2026-10-21，已完成）：Frisk 在既有 GF(2) 基础上以秩/核及像空间包含证明 ownership/Broadcast，经小域独立枚举差分检验，同时保留地址表枚举边界。465 unit、64 lit 及辅助验收通过；局部证明调用明显降低开销，但两类 Attention 全链时间基本持平，前后 IR/非计时 stats/完整 trace 相同。这是本系统可核验性与编译开销的增量，不把线性代数方法本身宣称为独有创新；本轮不刷新 TileLang/Triton 参考版本，不作 GPU 性能比较。见 [设计](optimize_plan/p1_1_implementation_design.md)和[实测报告](optimize_plan/p1_1_implementation_report.md)。

> P0 增量方案 A（2026-10-14，已验收）：在同候选域、成本模型及总预算下，补齐固定外部 ABI、可复核有界 trace、单变量邻域改进与饱和全局不退化保护。自由布局 Attention 新增 70 次探测，内存事务估算 608→544，bank 代价 64→0；5 次重复采样中 CPU 推断中位时间约增加 25%，trace 额外开销约 0.4%–0.6%。450 unit/64 lit 等完整守门与独立复审通过；范围、固定 ABI 对照和全部样本见 [实施报告](optimize_plan/p0_implementation_report.md)。这些是 Frisk 本轮实现证据，不等于全局最优或 GPU 加速；未刷新 TileLang/Triton 快照，也不声称它们缺少类似机制。

> **M6 完成（2026-10-10）：Task27–29 已实施并通过分项复审，最终全量回归通过。固定 TileLang 的 20 个共同域等价比较与 3 个实际拒绝通过；真实 GPU conversion 的 11 个正确性用例及 33 次 sanitizer 检查通过。Task29 经不删减证明的仿射求值优化，最终布局编译占比为 3.80%/3.97%/6.67%，runtime 比值不超过 1.00834，独立 reference 基线已接受。首轮 FAIL 保留为历史证据，整体源码及最终文档复审通过。这不构成比 TileLang/Triton 更快或完整推断策略等价的结论；口径及摘要见实现计划 §29.3 和验收文档。**

### M6 已确认的比较与证据规则

1. **版本和来源可核验。** TileLang 保持 `5e149e31674658f94779c7d0c6039549a1853123`，Triton 保持本文原固定快照，不自动追踪最新 main。TileLang 参考必须在隔离环境真实构建/运行并记录依赖及加载路径；禁止从 Frisk 结果反填 corpus 或把手工翻译公式冒充上游实际输出。
2. **比较语义，不要求实现相同。** schema v2 分开记录逻辑覆盖、carrier 所有权、同值 copy group、完成值/部分贡献、写入策略和存储 bit offset。多个合法持有者不等于重复写入；对上游未定义的 writer 选择只核对明确策略的兼容性，不宣称两边策略完全相同。
3. **支持与拒绝分开报告。** 共同支持域进行 `equivalent` 比较；Frisk 尚不支持的 dtype-changing view、一般非二次幂 Tensor/broadcast 或 PartialFragment 做真实 `reject` 测试，不计作等价支持，也不为追平 corpus 扩大生产语义。保留不采用某项上游语义的依据。
4. **确定性和最优性的证据不同。** 256 是 exact→beam 的组合阈值，不是候选硬上限；beam32/扩展预算65536 保持。独立小图 oracle 验证 exact/未截断解，截断 beam 不承诺最优或穷尽无解。测试的是 Frisk 自身契约，不由此推出比上游更优。
5. **性能结论必须经过实际生成链。** H100 conversion 对照必须运行 Frisk adapter 生成的 inferred 路径与独立 reference kernel，不能用静态 CostVector 或两份手写 CUDA 替代。3% 回退及适用域 10% 完整编译占比维持；没有完整 codegen 的 MMA/TMA 只报告静态契约，无需也不得伪造 runtime。通过这些门槛只证明固定 case/环境的结果，不证明总体优于 TileLang/Triton。

Task27 完成时的历史检查点（2026-10-09）：已新增独立逐位代数 oracle、全赋值九维成本 oracle、插入顺序/跨进程确定性测试及有限失败缩减工具，包含 16 项新增测试的 45 项局部 unit 与全量 56 项 lit 通过，并通过独立复审。缩减文件只支持明确的 `finite-same-require-v1` 测试域，不能重放任意 GPU/alias/region 图。此证据用于核验 Frisk 自身稳定性与失败分类，不是与 TileLang/Triton 的语义差分或性能优势证据；当时真实 corpus 和 GPU 验收尚待 Task28/29，现已完成。

下文记录 Task28 实际 corpus 范围及对齐/拒绝项；Task29 的运行环境、性能结果和验收命令见实现计划 §29.3 及 `layout_inference_acceptance.md`。已批准的比较方法与实测结论分别记录，不将有限共同域证据推广为整个框架的性能或推断策略等价结论。

Task28 实测结果（2026-10-09，独立复审通过）：已真实构建固定 TileLang/TVM 并从实际 native 对象及 LayoutInference 导出 23 个 case，20 个共同域语义比较通过，3 个 Frisk 现有拒绝边界通过，两次最终导出字节一致。包含 FP16/BF16 WGMMA A/C、Shared linear/transpose/32B/64B/128B、4B/8B/16B Copy、dim0/1 Reduce、guarded Copy、合法复制以及 floating/显式 unused 布局。拒绝项为非二次幂一般 Tensor、dtype-changing view、PartialFragment；不计入已支持功能。Copy/Reduce 采用明确源布局 seed；floating 的参考推断结果与 Frisk 显式 map 比较，不表示两边默认优选策略相同。writer 使用兼容合法策略，不伪称上游选择了特定 writer。reader/exporter 的 9 项 C++、12 项 Python 测试通过，全量 382 unit / 56 lit / 5 CTest 通过；导出发布缺陷经回归修复并复审关闭。真实坐标对齐只支持这些固定 case 的契约结论，未测 kernel runtime，不证明速度领先。可复现命令、native 路径/摘要和升级规则见 [固定参考说明](./tilelang_layout_reference.md)，实现记录见计划 §28.2。

> Task26 方案 A 已完成（2026-09-28）：357 unit / 53 lit / 4 CTest / 5 Python normalization、SM90 工具及旧生产接口隔离检查通过，独立复审的重要问题已关闭。保留 Task25 普通候选竞争，仅可证明空域进入独立快照有序扩展；预算/unknown/硬冲突不作为松弛理由。共享重计算在求解计划中计费，转换提升受硬编码来源、执行域、dominance 与实际 IR 复验限制。本轮差异化关注可核验失败分类及优化计划/真实改写的一致性，不把 fallback、hoist 或 rematerialization 本身称为独有创新。不刷新上游快照、不声称 GPU 性能优势；具体验收见实现计划 §26.5，未 commit/push。

Task26 的比较口径进一步收紧为三个可核验机制（实际边界见实现计划 §26.4）：

- **失败有类型，重试不改变硬语义。** 普通候选先求解；只有可证明空域进入原始快照上的有序扩展。其他分量或同分量非空残余图的硬冲突、unknown、预算/beam 失败不能借空域被忽略。SM90 新增的是有限 replication 提案，不宣称已有 WGMMA→SIMT fallback。
- **提升有授权和执行域边界。** 原本无编码的结果由物化器授予内部重写许可；显式编码继续是 hard seed。单输入 elementwise/extend 前移及线程内循环不变转换要求 dominance、成本非劣和 actual-layout 验证；不推测执行 shuffle/Shared 通信。
- **共享重计算先选择计费，再生成一次。** 同块同源同目标的多个 use 比较共享转换、独立修复、共享克隆；选择计划含 leader 和切片，实际改写核对计划并复用同一 SSA。可证明静态重复搬运/克隆计费减少，但未证明 GPU 运行时间领先。

这些不是对 TileLang/Triton “不存在对应优化”的断言。与固定快照的差异仍是 Frisk 将失败分类、有限候选、成本化边界计划、实际 IR 复验和事务提交组合为可追溯闭环；上游结论沿用已有审计证据，不在本任务推测其最新 main 行为。

> 历史增量（2026-09-28）：Task 25 的 SM90 规则/Copy 路径/guard/地址评分已实现，325 unit / 51 lit / 4 CTest / 5 Python normalization 及工具、隔离检查通过；重要复审问题已关闭。新增能力见下文 Task 25 小节和实现计划 §25.4–25.5。未刷新 TileLang/Triton 快照，未做 GPU 性能比较，未 commit/push。

> 最新验收（2026-09-27）：Task 23 已完成，正式 exact/beam 求解器与静态 CostVector 接入，272 unit / 49 lit / 4 CTest / 5 Python 及独立复审通过，旧 9/8 变量集成缺口已补齐。本轮未刷新上游快照、未做性能对比、未提交或推送。

> 文档性质：技术对比 + Frisk 架构决策记录
>
> Frisk 目标平台：NVIDIA SM90/SM90a
>
> 首次审阅：2026-08-16；本次源码审计：2026-09-12
>
> 状态（2026-09-19）：M0–M3 与 M4 Task 18–20 已完成；Task 21 的受限 Reduce 实现、独立复审及 197 unit / 36 lit / 4 CTest 回归已完成，完整 MMA→Reduce→tile_store 成功验收仍受 9 变量与 8 变量上限冲突限制，见实现计划 §21.8。Task 22 尚未实施。Task 19/20 已提交为本地 main `c028e82`，未 push；Task 21 未提交、未推送。远端同步状态不由本文推断。

> 最新进展（2026-09-20）：Task 22 已按用户确认的方案 A 契约实现，独立复审与全量验收已通过，记录见实现计划 §22.13。上一轮已经将 `c028e82` 与 `14b39e9` 推送到 `origin/main`；上文未推送为历史记录，本轮修改未提交或推送。Task 21 的 9/8 变量完整链验收缺口仍保留。
>
> 实现更新（2026-09-19）：Task 20 的 SS＋RS 具体契约 v1 已确认并实施；实际适配、有限联合约束与验证边界见下节及实现计划 §20.13。本轮核对固定上游快照的 GEMM 源码，没有升级快照或运行上游测试。

> 后续进展（2026-09-19）：Task 19/20 既有修改已按用户要求提交到本地 main `c028e82`，提交前重跑 165 unit / 34 lit / 4 CTest 全通过；未 push。以上“未提交”为历史时点记录。Task 21 已经用户确认并实施，下面区分已验证能力与尚未完成的完整链验收，不将分段成功计作端到端成功。
>
> 关联文档：[Frisk 布局推断系统设计方案](./layout_inference_design.md)、[GF(2) 与组合布局说明](./gf2_layout_guide.md)

## 0. 审计基线、证据边界与维护约定

本轮按用户要求先审计与设计，再确认实施。下表保留审计起点；Frisk Task 18 的实现增量另见 §6.1、§13 和实现计划 §18.9。上游比较仅指固定快照，不表示今后 main 的永久状态。

| 项目 | 本轮源码快照 | 审计/验证范围 |
| --- | --- | --- |
| Frisk | `b120d06400a14a703a44dac1a37a0b38d8110935`，M3 合并后的 main | 读取新旧布局路径；本轮构建及 27 lit、74 unit、4 CTest 全部通过 |
| TileLang | `5e149e31674658f94779c7d0c6039549a1853123`，2026-09-12 | 官方 main 获取后核对 HEAD；读取推断、代数、Op、reducer verifier 和测试；未运行其测试 |
| Triton | `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c`，2026-09-11（提交时区 -07:00） | 官方 main 获取后核对 HEAD；读取布局传播、SCF、代数、存储区域分析和测试；未运行其测试 |

历史记录：2026-08-16 比较采用 TileLang `6623b12d232b343648a5ba99992e3e6f0d6376d2`。本轮逐项核对旧版本，区分“旧版已有”和“本区间新增”；旧快照保留为迁移回归参考，不再代替新版语义。Triton 原文使用浮动 main，无法据此证明具体机制在哪次提交首次出现；本轮只陈述固定新版已具备什么。

维护规则：

- 每个相关实现任务同步本文件和[实现计划](./layout_inference_implementation_plan.md)：版本、源码证据、行为取舍、测试、未完成项缺一不可。
- “源码存在”“本轮测试通过”“设计目标”“性能/创新假设”分别标注；不得互相替代。
- 上游升级必须记录旧/新 SHA 和行为变化；不能只替换链接就宣称差分 corpus 已更新。
- 不采用“另一框架完全没有”作为创新证据；缺失结论限定到已审阅 pass/接口和支持子集。
- 本轮没有 TileLang/Triton 运行实验、GPU benchmark 或端到端性能结论。上表 27/74/4 是 M3 基线；Task 18 的新增测试结果单独记录，不能混用。

## 1. 结论摘要

### Task 23 求解器实现与验收（2026-09-27，已完成）

本轮以 Frisk 工作区（main `9bc8349`＋未提交 Task 22）为起点，未刷新或运行 TileLang/Triton。起点审计核实了旧 bootstrap 的 8 变量/4 候选限制、转换数优先及未被调用的 target CostVector；当时 `LayoutConversionEdge` 的字节/同步字段没有实际填充值。Task 23 现已替换该求解器并接通静态成本，不把此前“已有代价结构体”表述为“当时已有性能求解器”。

用户确认的固定候选域小组件 exact、大组件确定性 beam 已实现，明确区分穷尽无可证明方案、beam 未找到解和搜索预算耗尽，并保留操作级证明、实际 IR 复验与模块事务。具体契约及验收见[实现计划 §23](./layout_inference_implementation_plan.md#task-23-实现连通分量候选求解和-costvector)，独立复审与 272 unit / 49 lit / 4 CTest / 5 Python、生产接口隔离扫描均通过。Task 22/23 后续已提交并推送至 `origin/main`，Task 24 的工作基线为 `36df49f`。

可追踪的设计重点是：候选组合选择与硬件合法性证明分离；搜索不完整不能冒充非法性证明；联合 SSA/use、Storage、MMA tuple 和 Reduce pair 的既有关系求解；CostVector 区分已建模静态项和未来硬件估计。Beam/exact、字典序代价及稳定排序本身不是创新性证明，也不能由此宣称比固定快照的上游系统更快或更优。Task 24/25 的 placement/rematerialization 与完整 SM90 成本不属于 Task 23。

**Task 24 方案 A（2026-09-27，已完成）：** 在完整布局赋值评分期间比较同块共享 conversion 和最多 8 个便宜纯操作的 rematerialization，记录可验证的计划后事务式物化。与上文固定快照的 Triton backward rematerialization 相比，本任务强调边界选择在赋值评分中可见，以及 SSA/use 与既有双域硬约束一同验证；不声称“重计算”本身是 Frisk 独创。首版不跨 block/region/loop 提升，成本为带来源与循环深度权重的静态代理，不是完整 SM90 性能模型。最终独立复审及 287 unit / 50 lit / 4 CTest / 5 Python、旧接口隔离扫描均通过，命令、修正轨迹与边界见实现计划 §24.7；本轮未提交/推送，未运行上游框架或 GPU benchmark。

本轮实际差异：选择以硬关系连通分量为范围，输出实际 Attribute assignment、指令/归约 binding 和逐 use conversion；小图 exact、大图有界 beam，失败区分无可证明方案、截断未找到解与预算耗尽。已有 32→128 活跃线程子集转换继续合法，成本取两端最大线程覆盖；Shared 别名只按根最大占用末端计一次。这与旧 Frisk 局部 layoutMap 更新不同，但并不证明上游缺少全局分析或代价选择。9 变量核心及 Copy/Parallel128/MMA/Reduce/Global 写回已在 25 项集成回归中成功；全量 Gate 及精确命令见实现计划 §23.9。

### Task 22 方案 A 的实现与验收（2026-09-20）

基线为 Frisk `14b39e9`。方案 A 的生命周期转换、持久契约、事务 pipeline 和旧入口退役已落地，独立复审通过，最终 242 unit / 48 lit / 4 CTest / 5 Python 测试方法及隔离扫描通过；本轮未刷新或运行上游框架。实际代码、适配及测试证据以[实现计划 §22.13](./layout_inference_implementation_plan.md#task-22-实现-legacy-normalization-并退役旧生产路径)为准，不把测试文件存在等同于执行通过。

旧 Buffer→Tensor 迁移不能只替换类型或 Op 名字：原 Gemm verifier 与 Python 绑定要求同 dtype，而新 MMA 目标采用 f16/bf16 输入与 f32 累加。本任务同步扩展导入 builder/verifier，并以显式数学声明处理旧 clear；不静默提升累加精度，不把尚未初始化的 Local buffer 当零。新增完整 Local 生命周期和 SCF 状态转换，不能归功于此前已有的 Tensor region graph。

已落实的体系区分点仍是 Frisk 自身的“双域表示＋操作契约＋实际 IR 独立验证”：真实存储继续使用 MemRef，寄存器值生命周期转成 Tensor；whole-root map/对齐前置事实经持久声明保存，不从消费者需求反推根保证。旧方法已从生产 API 删除，baseline 仅保留测试角色。这是本仓库可核验的工程实现，不能单凭 Tensor 化、SSA 或旧路径退役声称上游没有相同机制或已证明研究独创性。

已确认契约增加以下明确选择：

- `frisk.legacy_semantics = "tensor_v1"` 作为 legacy Gemm/Reduce 的显式数学导入声明，区分“导入者接受新语义”与“已经证明所有旧数值实现等价”；f16 C 不静默提升成 f32。旧 Buffer builder/verifier 同步支持显式 f32 C，不能只让内部 C++ fixture 能构造。
- clear=true 表示覆盖；clear=false 必须有已初始化旧值，Reduce 先归约再按同 kind 与旧 dst 合并，并补全真实目的 Read effect。SCF 的初始化/分支合并/循环携带值单独证明，不把原有 Tensor region graph 当作现成 Buffer promotion。
- `frisk.storage_contract` 保存完整根 map 与对齐前置条件，并按 dominance 限定作用域；其不可随意删除来自声明本身的语义，不伪造 layout_view 内存副作用。验证同时检查映射相容性，不只保存对齐数值。
- 对原简单 pass 串联计划作事务适配：具名 pipeline 在 clone 上执行 normalize、infer、根契约保存、conversion 优化及最终 actual-only，全部成功才提交；单 pass 原子性不冒充整条 pipeline 原子性。
- 旧公式只保留在不被生产二进制链接的测试 oracle 中。使用普通测试库而非收集到 FFI 链接列表的生产库，避免源码位置看似隔离而链接仍污染生产路径。

本阶段保留 8 变量求解预算；normalize、预算内 pipeline 成功和完整 MMA→Reduce→store 成功分别验收。根声明是硬前置事实，不新增可选布局域，不移除真实 producer/use 来绕过预算。不把 Task 21 的分段证据重新包装成端到端证据，也不在缺少 GPU/上游实验时声明数值等价或性能领先。

### Task 21 实现增量与验收边界（2026-09-19）

详细审计、候选方案、数学语义与验收点见[实现计划 Task 21](./layout_inference_implementation_plan.md#task-21-增加-tensor-reduce-op-并迁移-reduce-ownership)的 §21.1–21.7。Frisk 基线为 `c028e82`；本轮再次核实本地审计副本的 TileLang/Triton HEAD 与 §0 的固定 SHA 一致，没有升级快照或运行上游测试。GitHub 固定源码网页读取失败时使用该本地副本核对，不据其他旧分支推断新版行为。

| 本轮源码证据 | Frisk 的适配选择 | 不能据此作出的结论 |
| --- | --- | --- |
| 旧 Frisk Reduce 只处理 Local MemRef，依赖 src layout 投影/condense；旧 kind 为 add/mul/min/max | 新 `reduce_tensor` 使用纯 Tensor SSA、sum/max/min；旧 clear/Buffer 生命周期转换留 Task 22 | 旧接口改名不等于已完成新 solver 集成 |
| 旧测试 case 5 标注 src map 有问题；case 7 的 register/thread 映射遗漏 batch 坐标；另有 16-thread 案例 | 先验证完整源持有关系；把错误或超支持范围样例分类，补独立贡献 oracle | legacy PASS 不代表所有旧布局都可作为正确性真值 |
| TileLang `ComputeReducerLayout/InferLayout` 仍进行普通 Fragment 源驱动投影、condense 和目的包含性检查 | 正向生成有限自然输出，反向仅筛选已存在源候选，区分信息生成与约束删减 | 新 reducer epoch 机制不代表传统 Reduce 已全面双向化 |
| TileLang PartialFragment 区分 addend lanes 与 equal-value copy groups，禁止 partial 走普通 replica 等价捷径 | 新证明分别建模不同逻辑输入、输入同值副本与完成输出副本；sum 每个逻辑输入恰贡献一次 | Frisk 不能声称发现了上游尚未考虑的“部分和不等于副本”问题；也不声称本轮实现 epoch |
| Triton Reduce 结果使用 SliceEncoding；lowering 已有寄存器内、warp 内及其余布局转换/同步步骤 | 复用 Frisk 通用 BitLinear 表示，配有名二端点归约关系和操作级可验证通信契约 | 分层归约或显式转换本身不是 Frisk 独创；当前 Frisk 仍不交付对应 GPU lowering |
| 当前 Frisk TransformLayout 只实现等 rank 置换，公共 Distributed 不支持一般 ragged/动态 tile | 不伪造 Reduce 逆变换；经用户确认，原非二次幂 fallback 延后，静态受限子集先闭环 | 不能以 Shared 通信方案或现有 Product 代数库存在推断 ragged 已支持 |

用户已确认并实施的方案是“坐标 fiber 的精确贡献证明＋有限 pair 支持筛选＋实际 IR 独立复验”。自然输出删维并压缩 register 子空间，保留线程维的复制语义；源副本按 first_owner 选代表参与相邻配对确定性树，完成后分发到全部输出持有者。公共 API 不新增 PartialFragment 类型，也不借用 Task 20 的四端点 MMA tuple 假装二端点归约。实现采用通用 Distributed encoding、独立 `ReductionLayout` 关系及类型化 `ReductionContractAttr`；12 字段契约和实际代码定位见实现计划 §21.8。新增 32 项单元与 2 份 lit，全量 197 unit / 36 lit / 4 CTest 通过；不宣称性能最优、研究独创性已验证或数值 GPU 验收已完成。

已确认的数值语义：sum 允许并行重结合、不保证串行 bitwise 相等；max/min 采用 NaN 传播及正负零有序的 maximumf/minimumf 语义，参考已核对的[MLIR Arith 定义](https://mlir.llvm.org/docs/Dialects/ArithOps/)。不复制 TileLang `nan_propagate` 缺省或旧 clear 行为而不说明差异。非二次幂、rank-0、一般 Product、PartialFragment/epoch、GPU lowering 与完整成本优化不在首版。

**支持边界的实施发现：** 完整 SS MMA→Reduce→tile_store 在现有生产者/use 分离模型下需要 9 个连通变量，与原契约的 8 变量上限冲突。未放宽上限；将 MMA→Reduce 与 Reduce→store 分段验证，完整链保留超限拒绝证据。这不是相较上游的能力优势，不能把两个切片的成功拼成端到端成功。实际回归结果与复审修正记录见实现计划 §21.8；本轮仍未进行 GPU 或上游运行实验。

固定源码锚点：[TileLang Reduce](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/reduce.cc)、[PartialFragment](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/layout/layout.h)、[Triton Reduce encoding](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/IR/Dialect.cpp)、[Triton Reduce lowering](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp)。

### Task 20 实现增量（2026-09-19）

具体契约和实测记录见[实现计划 Task 20](./layout_inference_implementation_plan.md#task-20-增加内部-tensor-mma-op-并迁移-gemm-约束)。工作基线是 `ef85a0d` 加未提交的 Task 19 修改；下表记录已经接入的代码选择，测试结果单独记录，不把静态证明当作 GPU 执行证据。

| 已核对的证据 | Task 20 已实现选择 | 比较边界 |
| --- | --- | --- |
| 旧 Frisk Gemm 使用 Local MemRef/DenseMap，verifier 同 dtype 且未按 transpose 检查形状 | 新增 Tensor accumulator/result 的 `frisk.mma`，显式表达 `result = init + A_eff × B_eff`；消费处插转换 | 不直接沿用旧 verifier，不提前实现 Task 22 的旧 Buffer 生命周期 normalization |
| TileLang 固定快照的 GEMM 使用更新后的 annotations、clear 读依赖语义及 WGMMA shared 要求 | 用 SSA init 表达累加依赖；shared 地址按实际角色证明，不复制位置参数 ABI 或推断完成标记 | 不是声称 TileLang 没有严格指令布局检查 |
| TileLang WGMMA partition 以四个 warp 为不可拆单元；其 operand-layout 测试检查父 buffer 的 K-panel stride | 固定合法 warp-group grid；从实际 root/layout 重建 descriptor 地址，不按切片 extent 猜 panel 间距 | 四 warp 和实际 stride 是正确性要求，不单独作为 Frisk 创新点 |
| Triton MMAv3 已有 MMA encoding、accumulator conversion、A register/shared 与 B shared 路径 | SS＋RS 都接入同一有限域联合 InstructionContract，保留显式转换 | 不声称 Triton 缺少这些能力；差异是 Frisk 的统一约束表示及求解后核验方式 |
| 实现前 Frisk InstructionContract 尚无执行语义，unary/binary 关系不能保证四个角色属于同一指令方案 | 增加 `[A-slot,B-slot,init-use,result]` 的有界合法 tuple，传播删除无完整支持的候选，求解选择整条方案 | 已接入 Strict/Common 与部分 assignment 检查；不是完整 CostVector 或大图全局最优求解 |
| M3 已有通用 Distributed/Storage encoding，尚无专用 Mma/DotOperand encoding 类 | 复用通用 map，在操作上写类型化 `mma_contract`；按专用 fragment、packed 半字顺序、descriptor 证明 | 是对原计划 encoding 类层次的明确适配；不把普通元素覆盖误当成硬件指令兼容 |

实现支持明确的 `sm_90a`、f16/bf16 输入和 f32 accumulator；SS 是 A/B shared，RS 是 Tensor A/shared B。只处理当前布局表示允许的静态二次幂 tile、单 CTA、128/256/512/1024 线程，以及有界规范 descriptor 模板。Tensor B、Local MemRef fragment、动态尺寸、未证明的地址方案和超出 bootstrap 预算的组件明确拒绝；不静默改变 shape、补 shared buffer 或扩大候选上限。无目标标注不为 MMA 猜测 `sm_90a`，但不改变 Task 19 普通操作的默认目标行为。

物化后的纯验证只读取实际 Tensor/Storage encoding、线程环境和完整指令属性，重新证明 fragment/packing/descriptor 的联合一致性；不枚举候选，不使用上次求解 tuple 作为证明。测试覆盖事务回滚、打印解析回放、无 singleton 的联合不相容反例以及独立硬件坐标 oracle。`SameLayout` 连接 init 的实际消费槽和 result，不强迫外部 init 的全部使用者一致。

源码锚点：`Analysis/InstructionLayoutConstraints.cpp` 的有限 tuple 和完整支持检查，`Analysis/MmaLayoutConstraints.cpp` 的真实角色，`Target/SM90/SM90GemmConstraints.cpp` 的有界建议/方案构造，`SM90MmaLayoutProof.cpp` 的纯 fragment/descriptor 证明，以及 `Transforms/MaterializeLayouts.cpp` 的 typed binding 与 actual-only 事务验收。通用 Analysis 不依赖 SM90 实现库。

与最初设计的明确适配：不新增 `MmaEncodingAttr`/`DotOperandEncodingAttr` Tensor 子类型；Distributed 保持 BitLinear、Storage 使用既有 Affine/BitLinear，操作级两个 typed 属性使用严格 Dictionary schema。多原子重复可在当前 BitLinear fragment 中表达，但未扩展一般 Product 指令布局。生成失败/证明预算不足报告 Unknown，不因目标建议不足就声称所有硬件实现均不可能。

旧规则分类差分只覆盖适用的持有关系与 storage 地址。新分类测试记录旧 RS A replication=2 与 128 线程新 canonical A replication=1 的区别；旧布局不能因为来自 baseline 就绕过新硬件证明。SM80/`sm_90`、Local B、错误四-warp 组织明确不进入新路径；旧 f16 accumulator、positional ABI/clear 不冒充新 f32/init SSA 语义等价，自动 normalization 留到 Task 22。

上游证据固定为 TileLang `5e149e31674658f94779c7d0c6039549a1853123` 的 [`src/op/gemm.cc`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/gemm.cc)、[`src/cuda/op/gemm.cc`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/cuda/op/gemm.cc)、[`test_tilelang_cuda_wgmma_operand_layout.py`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/cuda/test_tilelang_cuda_wgmma_operand_layout.py)，以及 Triton `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c` 的 [`AccelerateMatmul.cpp`](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)。硬件依据为本日访问的 [NVIDIA PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)，具体章节见实现计划 §20.10。

已落地的架构差异是“Storage/Distributed 双域＋有限联合硬约束＋实际 IR 独立证明”的组合；不能据此断言上游绝无相似机制，或未经实验宣称创新性/性能领先。当前不实现完整 CostVector、GPU lowering 或异步 pipeline，也没有上游运行差分、GPU 数值测试和性能优势证据；可表示静态指令契约不等于已生成可执行内核。

本轮新增 36 个单元与 2 份 lit，最终 **165/165 unit、34/34 lit、4/4 CTest** 通过。实际推断覆盖 16 个小 tile SS/RS/dtype/transpose 组合和 4 个 `128×128×64` SS/RS/dtype 组合；8 类 descriptor、4 种线程数与 policy 的更广矩阵在 helper 层验证，不宣称完整端到端组合穷举。详细命令、首次全量测试 fixture 修正、Unknown 诊断修正及复审闭环见实现计划 §20.13。

### Task 19 实现增量（2026-09-16）

本次以 Frisk `ef85a0d` 为实现起点，先审计 Copy/Fill/Parallel 接口、确认具体契约，再按[实现计划 §19.1–19.9](./layout_inference_implementation_plan.md#task-19-迁移-copyfill-和-parallel-约束)实现与测试。下表记录已经落地的设计选择；本轮未重新获取或运行 TileLang/Triton，上游比较继续限定到 §0 的固定快照。

| 实现前审计发现 | Task 19 已实现选择 | 与既定体系比较的关系 |
| --- | --- | --- |
| Frisk Copy/Fill 目前是 MemRef 操作，没有承载执行布局的 Tensor SSA 结果 | 将执行布局、线程环境、writer policy 和向量宽度显式物化在原操作上；内存布局仍在 layout_view 上 | 保留 Frisk 的 Storage/Distributed 双域，不把旧 Buffer 接口机械改称 Tensor encoding |
| M2 Copy 对两端 storage map 强制相同 | 用同一逻辑坐标的读/写访问关系连接各自存储；有限候选可以双向准备，但不同存储不构成 alias 或硬布局等值 | 吸收既有审计中 TileLang 的方向性信息利用，采用图中的硬正确性关系与候选建议分离，不宣称全面复现其 dst-steering |
| replicated 持有不等于 replicated 写入 | 区分 all-writer 与显式 first-owner，逐点核验唯一实际写入者；不假设未来 lowering 自动选 owner | 延续三方比较中的 read/write ownership 区分；不把 TileLang partial addend 当作同值副本，也不扩展 reducer epoch |
| Ownership/ResourceLimit 枚举尚无完整求解语义；新 pass 已不调用旧 Parallel inference | 真正接入传播/求解/最终核验，线程拓扑不直接等于最终布局；增加真实旧调用隔离测试 | 可核验差异是约束、物化属性与最终 IR 的一致性，不是声称其他系统没有线程约束 |

首版仍限制静态 whole-tile Copy/Fill、单 CTA 和已支持的布局表示；执行域各维必须是大于 1 的二次幂，bootstrap 的 8 variables/component、4 candidates/domain 限制不变。不扩展 Tensor 版 Copy/Fill、TMA/cp.async lowering、GEMM/Reduce 或完整成本求解。`first_owner` 是已核验并物化的显式执行契约，不等于已生成 GPU 写入谓词；未来 lowering 必须消费该契约。

新增的 `CopyAccess` 只约束逻辑点的复制及已知同 root 的不重叠/恒等条件，不要求两端物理 map 相等，也不替代 `AliasLayout`。例如 `2×2xf32` 的源地址 `8i+4j` 与目的地址 `4i+8j` 可以复制；不同 MemRef 参数仍需调用者保证非重叠，不能把不同 root ID 当作 NoAlias 证明。候选准备可在两端之间重定位有限存储建议，但冻结后只删除候选，不继续生成。

源码证据：`OperationLayoutConstraints` 收集实际操作绑定；`ExecutionLayoutProof` 核验规范 writer 与逐线程寄存器向量；`LayoutRelations` 统一执行 CopyAccess/Ownership/ResourceLimit；`SM90DistributedCandidates` 按线程环境生成有限候选；`MaterializeLayouts` 事务式物化四项属性并独立重建核验。纯证明放在 Analysis，避免 Analysis/Target 循环依赖。覆盖不同物理布局、双向候选、写入竞争、向量过滤、Parallel capture 转换、属性篡改、旧路径隔离和文本回放的测试及 Gate 结果见实现计划 §19.9。

差异定位是“存储与执行分开建模 + 有限候选建议与硬证明分开 + 实际 IR 契约可独立重验”，不是声称 TileLang/Triton 不支持 owner 选择、别名或向量访问，也不是性能优越性的证明。未运行上游差分或 GPU benchmark。

三者的核心路线可以概括为：

```text
TileLang = 以 Buffer 为中心的兼容布局闭包
Triton   = 带 Encoding 的 Tensor SSA + 硬件锚点驱动的多轮布局改写
           + 显式 ConvertLayout 的插入、消除、提升与重计算
Frisk目标 = MLIR 类型化的 Distributed/Storage 双域布局
           + 全图约束传播 + 有限候选联合选择
           + 求解后显式物化 conversion
```

需要先修正两个容易产生误解的说法：

1. **TileLang 并不是在前端创建每个 Op 时立即确定布局。**每个 TileOperator 确实实现自己的 `InferLayout` 规则，但最新布局系统是在 pipeline planning、software pipeline rewrite 等变换之后，由独立的全局 `LayoutInference` pass 收集 TileOp，再执行 strict、common fixed-point、free-mode 和 alias finalize。
2. **Triton 也不是所有 Op 各自独立选择“自己的最优布局”。**TTIR 转成 TTGIR 时先赋予默认 encoding，随后 coalesce、matmul acceleration、dot operand optimization 等 pass 建立高价值布局锚点；`RemoveLayoutConversions` 再从锚点传播布局、解决冲突并重写 IR。普通 elementwise/shape Op 更多是在传播中跟随锚点，而不是各自运行一个完整代价搜索。

Frisk 的目标方案不是“以 Triton 为主、再复制 TileLang 的规则”，而是把三层明确分开：

| 层次 | Frisk 的选择 |
| --- | --- |
| IR 表示 | 采用 MLIR/Triton 风格的类型化 Tensor SSA、MemRef 和显式 conversion |
| 规则传播 | 吸收 TileLang 的 strict/common/free 分阶段思想，但改成候选集合上的单调传播 |
| 决策机制 | 使用 Frisk 自己的全局约束图，对布局、硬件路径和 conversion placement 联合选择 |
| 布局代数 | 使用“仿射外层 × GF(2) 位线性内层”，而不是迁移 TVM PrimExpr，也不只复制 Triton encoding 类层次 |
| 目标支持 | 首阶段只实现 SM90/SM90a，通用 solver 与 SM90 rule library 分离 |

因此，Frisk 的准确定位是：

> **表示层接近 Triton，传播阶段借鉴 TileLang，决策层与组合布局代数形成 Frisk 自己的架构。**

这里所说的“创新”是待检验的架构假设，不是已证明的独创性或性能优势；当前完成度见 §6.1，验证与否证条件见 §10。

## 2. 比较前必须统一的概念

### 2.1 “布局”不是单一概念

三套系统使用了不同 IR 载体。如果把 register、shared memory、global memory 都笼统称为“内存布局”，会掩盖关键差异。Frisk 应至少区分两个布局域：

#### Distributed layout

描述一个逻辑 Tensor 的元素如何分布到执行资源：

```text
(register, lane, warp, warp-group, CTA) -> logical tensor coordinates
```

它回答：

- 某个逻辑元素由哪个 lane 持有；
- 一个 lane 的第几个寄存器槽保存该元素；
- 元素是否被多个线程复制；
- 哪个线程拥有最终 store/reduce 的写权限。

“Register”在这里不是一种可以被 MemRef 任意寻址的普通内存，而是 SSA value 在 lowering 后的物理承载方式。因此 Frisk 目标架构使用：

```text
RankedTensorType + DistributedEncodingAttr
```

来表达寄存器分布，而不是继续用 `memref<..., local>` 模拟可寻址的寄存器数组。

#### Storage layout

描述逻辑坐标如何映射到真实可寻址存储：

```text
logical tensor coordinates -> physical bit/byte address -> bank/segment
```

它回答：

- shared memory 是否转置、padding 或 XOR swizzle；
- 地址、stride、alignment 和 allocation size；
- TMA box、descriptor 和 swizzle 是否合法；
- 同一 storage 的 view/alias 是否保持一致。

Frisk 使用 `MemRef` 保持可寻址存储语义，并在 allocation/layout view/binding 上关联 `StorageLayoutAttr`。首阶段它主要服务 shared memory；global memory 通常继续使用 MemRef 自身的 strided/affine layout，只有存在显式物理重排时才需要额外 storage binding。

### 2.2 Conversion 的含义

`convert_layout` 不是改变数学 Tensor 的值，而是改变同一逻辑值在执行资源上的分布。它可能 lower 成：

- register shuffle；
- register → shared → register；
- warp/warp-group 间交换；
- 重计算 producer，直接生成目标布局；
- 在特殊情况下退化为带同步的较重搬运。

因此 conversion 同时具有正确性语义和性能代价。它必须显式、可验证，也必须进入全局成本选择。

### 2.3 “推断时机”需要分成三个时刻

不能只用“早”或“晚”描述布局推断，应区分：

1. **布局规则何时可见**：高层 TileOp、TTIR Op 还是目标化 GPU Op；
2. **布局何时成为 IR 的一部分**：Buffer annotation、Tensor type encoding 或独立 binding；
3. **冲突何时被解决并物化**：推断过程中、专用 propagation pass，还是 lowering 前。

## 3. 三者总体流程

### 3.1 TileLang 总体流程

以本轮 TileLang `5e149e3` 为基线，以下仅列 CUDA 布局相关关键阶段（不是完整 pass 清单）：

```text
High-level Tile IR
  -> warp specialization / Blackwell annotation preparation
  -> pipeline planning
  -> software pipeline rewrite
  -> LayoutInference
       0. floating fragment -> fully replicated
       1. strict inference
       2. common BFS fixed-point
       3. free-mode root search
       4. alias finalize
  -> ReducerPlanAndMaterialize
  -> LowerTileOp
  -> TIRX/CUDA lowerings
```

总体逻辑是：

> 收集 TileOp 对 Buffer 的布局要求，通过多轮兼容性传播，为每个 Buffer/alias group 得到一个最终布局；布局冻结后再降低 TileOp。

### 3.2 Triton 总体流程

Triton NVIDIA 后端的关键逻辑不是一个单独的“布局推断 pass”，而是多轮目标化和布局优化：

```text
TTIR: unencoded tensor SSA
  -> TTIR to TTGIR TypeConverter
       tensor -> default BlockedEncoding
       type mismatch -> ConvertLayout materialization
  -> Coalesce
       为昂贵 load/store 选择合并访存布局
  -> RemoveLayoutConversions
       锚点传播、冲突消解、IR 重写
  -> OptimizeThreadLocality
  -> AccelerateMatmul
       选择 MMA encoding / dot operand / shared contract
       插入必要 ConvertLayout
  -> RemoveLayoutConversions
  -> OptimizeDotOperands / descriptor optimizations
  -> scheduling / pipelining / TMA lowering
  -> RemoveLayoutConversions（后续再次清理）
  -> TTGIR to LLVM/PTX
```

总体逻辑是：

> 让 load/store、dot/MMA、descriptor 等高价值 Op 成为布局锚点，通过带 encoding 的 SSA 类型表达布局差异，以显式 conversion 保证不同局部最优布局可以共存，再通过传播、rematerialization 和 conversion cleanup 减少代价。

### 3.3 Frisk 目标流程

Frisk 目标方案采用“先分析、后选择、再物化”：

```text
Frisk high-level MLIR
  -> layout IR normalization + verifier
  -> 建立 Distributed value graph、Storage alias graph、region edges
  -> 收集 Op constraints 和 SM90 instruction contracts
  -> hard seed
  -> strict propagation
  -> common bidirectional fixed-point
  -> component candidate generation + hard pruning
  -> 布局 / 指令路径 / conversion edge 联合代价选择
  -> controlled relaxation（仅处理普通候选域无解的情况）
  -> solved graph verification
  -> materialize encodings, storage bindings, convert_layout
  -> conversion canonicalization / hoist / rematerialization
  -> SM90 pipeline materialization
  -> Vector/MemRef/GPU/NVGPU/NVVM lowering
```

总体逻辑是：

> Op 只声明关系、合法集合和性能偏好，不直接锁定最终布局。求解器在整个布局连通分量上同时考虑 producer、consumer、storage、硬件指令和 conversion 位置，最后一次性提交类型与 binding。

## 4. TileLang 的具体布局推断逻辑

### 4.1 IR 载体和推断对象

TileLang 的布局推断建立在 TVM/TIRX Buffer 和 TileOperator 之上：

- shared memory 使用一般 `Layout`；
- fragment/local buffer 使用 `Fragment`；
- 推断状态主要是 `Map<Buffer, Layout>`；
- 同一底层 `buffer->data` 的不同 Buffer view 被归入 alias group；
- parallel loop 的布局和 predicate 以 annotation 形式回写到 IR。

这是一种 **Buffer-centric** 模型。一个 Buffer 在最终 layout map 中对应一个确定布局，consumer 一般需要共同接受这个布局。

### 4.2 每个 Op 提供规则，全局 pass 负责调度

TileOperator 的 `InferLayout` 会根据：

- 当前已经推断的 Buffer layout；
- target 和线程范围；
- TVM Analyzer 的等价性证明；
- 当前 inference level；

返回对 Buffer layout map 的更新。

但调用顺序和收敛过程由全局 `LayoutInference` pass 控制，而不是前端创建 Op 时立即执行。因此它同时具有：

- **局部规则**：Copy、Gemm、Reduce、Parallel 等 Op 知道自己的布局关系；
- **全局调度**：use-list、BFS queue、connected component 和 alias propagation 决定传播顺序。

### 4.3 五个阶段

#### 阶段 0：floating fragment 复制

如果 fragment buffer 在 TileOp 之外被普通控制流或表达式访问，TileLang 无法从 TileOp 契约推断线程访问模式，于是将其设为 fully replicated，保证任意线程都可读取。

#### 阶段 1：strict inference

传播由硬件/语义唯一决定的布局事实，并形成 `strict_layout_map`。后续 common/free 推断不能随意破坏这些事实。

#### 阶段 2：common BFS fixed-point

通过 Buffer use-list/worklist 调用 `InferLayout(kCommon)`；新布局及 reducer widening 会唤醒相关 users。不能据此断言所有事实更新都实现统一依赖唤醒：当前 containment 替换和 swizzle merge 分支并不统一重新入队全部 users，也未给所有 Layout 提供同一个有限格证明。

此阶段主要寻找兼容布局闭包。已有布局与新布局相遇时会：

- 接受数学上相同的布局；
- 对非 strict fragment，在可证明 containment 时采用包含更多映射的布局；
- 对 compatible shared swizzle 尝试合并到更小粒度；
- 无法兼容时报告布局冲突。

#### 阶段 3：free-mode

对未完全解析的连通分量尝试不同 root 的 native plan，并为符合条件的 reducer root 增加 scalar candidate。新版 `RunOneAttempt` 隔离队列、layout map 与 cloned op 状态，失败候选不能污染下一次尝试。

新版采用可插拔 `LayoutCostModel`：默认 `register-count` 先比较估计 spill traffic，再比较寄存器数；`tl.layout_cost_model="io-aware"` 为 opt-in，增加全局访问 bandwidth/issue 成本。I/O 模型利用 CuTe algebra 推导访问并采用 probe-then-prove，不能继续称 TileLang“只有寄存器启发式、没有访问代数或 I/O 成本”。它仍是有限 component attempt 评分，不是把任意 conversion placement 纳入的目标级全局最优求解。见 §15 固定源码。

#### 阶段 4：alias finalize

对共享同一底层 storage Var 的 Buffer：

- shape 相同则传播同一布局；
- shape/dtype 不同则根据 storage bit ratio reshape；
- 对支持的 full-buffer reshape/reinterpret 验证 storage bits 与映射兼容性。

这不是任意 offset/strided slice 的统一证明：前端 `T.view/T.reshape` 重建同 data 的 Tensor，检查总 storage bits，但不复制 source 的 strides/elem_offset。same-data 分组也不等于逻辑坐标恒等。位宽感知 reshape、widening 的同线程/连续且对齐槽位证明在旧 `6623b12` 就已存在。

### 4.4 冲突处理

TileLang 主要通过“找到一个大家都能接受的 Buffer 布局”消除冲突。其常规机制并不是为同一个逻辑 Buffer 保留多个 SSA encoding，再在 use 边界插入通用 conversion。

典型结果为：

```text
兼容              -> 合并/传播
fragment containment -> 选择可包含的布局
shared swizzle 可合并 -> 合并粒度
free-mode 有可行 root -> 按所选 cost policy 比较 attempt
仍不兼容           -> 编译期冲突或 unsupported
```

### 4.5 本轮确认的新机制与迁移影响

- **PartialFragment**：区分未归约的 addend lanes 与同值 copy groups；相同 storage algebra 不代表相同 reduction 语义。普通 Fragment 的 replica 处理不能直接复用。
- **Reducer 单调 widening**：`unset → narrow → wide`；strict annotation 冲突仍报错。dst-steering 决定哪个 Op 有首次提交无约束 dst 的权利，不是硬件 thread owner。
- **作用域与控制流**：过滤含外层自由变量的 open fragment；reducer epoch verifier 新支持 serial/pipelined loop 与受限 conditional refinement，检查 loop/then/else/while 语境。这不是通用 SSA block-argument/yield join 求解，但也不能写成“不支持控制流”。
- **Parallel**：free-mode 重访 frozen loop 时重新验证晚到约束；只读 broadcast 可退到 canonical fully replicated，written buffer 不能这样处理。
- **Copy/GEMM/Reduce**：Copy/GEMM 部分 positional 参数迁到 annotations；GEMM 新增 definite read-before-write region 语义。传统 Reduce 的主要 src→dst/containment 规则并未因 reducer 更新而全面双向化。旧 Frisk Op 迁移必须按当前 API 和语义重审。
- **逆映射**：新增 inverse round-trip 检查，可拒绝明确错误的逆；其有限域检查和 Unknown 处理有边界，不能称任意映射的完整反演证明。

证据与现有测试入口见 §15；这里只审阅测试源码，没有执行 TileLang 测试。

### 4.6 优势与局限

优势：

- TileOp 语义强，Copy/Gemm/Reduce 的规则靠近高层调度信息；
- Buffer 模型与 TIR lowering 直接衔接；
- strict/common/free 机制直观，兼容性强；
- 对高性能手写 tile kernel，布局通常由少数强算子约束，求解开销较低。

局限：

- 一 Buffer 一布局使多 consumer 的不同局部最优难以同时表达；
- 通用冲突恢复能力弱，很多冲突只能合并或失败；
- 类型系统不会自动验证每一条 SSA use 的 encoding 一致性；
- 推断代码绑定 TVM Analyzer、PrimExpr、Buffer、TIRX 和 TileLang op hierarchy；
- 若直接迁移到 Frisk，会把 MLIR 降格为 TVM 数据结构的承载壳。

## 5. Triton 的具体布局推断与优化逻辑

### 5.1 IR 载体

TritonGPU IR 将 distributed layout 作为 `RankedTensorType` 的 encoding。典型 encoding 包括：

- `BlockedEncodingAttr`；
- MMA encoding；
- DotOperand encoding；
- Slice encoding；
- shared/memdesc 相关 encoding；
- 底层可归一到 LinearLayout 的映射。

这使布局成为 MLIR 类型的一部分：两个 shape/dtype 相同但 encoding 不同的 Tensor 是不同类型，不能在不转换的情况下错误连接。

shared memory 则通过可寻址的 memdesc/storage 类型和相应 encoding 表达，不能简单理解为“所有内容都用 Tensor encoding”。Triton 与 Frisk 真正值得复用的思想，是区分 **分布式值布局** 和 **存储布局**，并让转换边界在 IR 中显式存在。

### 5.2 TTIR 转 TTGIR：先给默认布局

TypeConverter 保留已有 encoding，为未编码 `RankedTensorType` 添加默认 blocked encoding。当前 source materialization 可用 `UnrealizedConversionCastOp`，target materialization 使用 `ConvertLayoutOp`，二者不能混为同一路径。

这一步建立的是合法、统一的初始 TTGIR，不代表已经得到整张图的最终最优布局。

### 5.3 目标化 pass 建立布局锚点

#### Coalesce

对昂贵 load/store，Triton 根据 axis/contiguity 信息选择适合 memory coalescing 和 vectorization 的 blocked layout。实现通常先：

1. 把 memory op operands 转到 coalesced layout；
2. 创建使用该布局的新 memory op；
3. 再把结果转回原布局；

随后由 `RemoveLayoutConversions` 尝试把这个好布局向前后传播，从而消除临时 conversion。

#### AccelerateMatmul

当目标支持 Tensor Core 时，该 pass 会：

- 为 accumulator/result 选择 MMA encoding；
- 为 A/B 建立 DotOperand 或 shared operand contract；
- 插入从旧 encoding 到新 encoding 的 `ConvertLayoutOp`；
- 在 SM90 路径上结合 shared descriptor、WGMMA/TMA 相关 lowering。

因此 dot/MMA 是强布局锚点，而不是普通 elementwise Op 自己决定布局。

#### 其他锚点

当前 `RemoveLayoutConversions` 源码把 descriptor、昂贵 load/store、dot、atomic、特定 gather/reshape、部分 target-specific Op 识别为 layout anchor。锚点表示“这个布局具有应当保留的硬件价值”。

### 5.4 RemoveLayoutConversions 的主算法

当前实现的注释明确给出四步：

1. 找出希望保留布局的 anchor ops；
2. 从 tensor function arguments 和 anchor 的 tensor results 向 descendants 传播 encoding（无 result 的 store 虽可被判定为 anchor，却不直接成为 forward seed）；
3. 一个 value 可能得到多个候选 encoding，随后解决冲突并在必要处插入 conversion；
4. 按 dominance/structured-region 顺序重写 IR。

需要注意：当前冲突选择仍包含启发式逻辑。例如在已审阅源码中，load/store 倾向 blocked encoding，其他冲突倾向 MMA encoding，并留下“需要更完善 heuristic”的 TODO。因此不能把 Triton 描述成已经拥有统一、精确的全局最优布局求解器。

### 5.5 Conversion 优化不是简单删除

Triton 的 conversion 优化包含：

- identity/冗余转换消除；
- backward rematerialization：重算便宜 producer，直接产生目标 encoding；
- 把 conversion 提升到 broadcast/extend 之前，使搬运 Tensor 更小；
- 将 conversion 移入条件分支，避免无条件执行；
- dot operand conversion hoist；
- 对 region/block argument/yield/result 进行一致重写。

这说明 Triton 的真正优势不只是“允许冲突”，而是：

> 把冲突显式化之后，能够使用 SSA、dominance、slice analysis 和 canonicalization 对 conversion 做系统优化。

### 5.6 Region、存储分析与代数的实际边界

`RemoveLayoutConversions` 使用单调**增长**候选的 forward worklist；`SmallSetVector<Attribute, 8>` 中的 8 是 inline capacity，不是候选硬上限。随后还有独立的 rewrite、backward rematerialization/cleanup 循环，并非同一个有限域删减求解器。

`scf.while` 已支持两套不同 arity/type 的 tuple：init/after-yield 对 before args；condition-forwarded values 对 after args/results，跳过 predicate。反向 rematerialization 对 while block args 和部分 loop results 仍有限制。不能把局部限制描述成 Triton 没有 while 支持。`getValueAs` 的转换可放在定义后，并非一律紧贴 operand use。

当前 `BufferRegionAnalysis` 已区分 allocation frame、view offset 和精确 physical address sets；动态 index 可以形成合法 subbuffer 的 MAY 集合，未知 producer 提升为 Unknown。这些能力用于 memory effect、ConSan 与同步分析，**本轮未发现它参与 RemoveLayoutConversions 候选选择，不能将两者说成一个布局 solver**。

`LinearLayout` 本体为 GF(2) 映射，且允许非单射以表达复制；pseudoinverse 选择规范代表不等于证明唯一 writer。Triton 的 `PaddedSharedEncoding` 已组合 padding 与 linear component，不能把 GF(2) 本体限制扩大为“Triton 无法表达 padding”。固定源码与静态测试见 §15。

### 5.7 优势与局限

优势：

- Tensor encoding 与 MLIR SSA 类型系统结合，错误连接能够被 verifier 捕获；
- 不同 consumer 可以拥有不同局部布局，冲突不必直接编译失败；
- coalescing、MMA、descriptor 等硬件优化可以独立迭代；
- conversion 可以利用 dominance、region 和 rematerialization 系统优化；
- 已经形成成熟的 NVIDIA lowering 与性能经验。

局限：

- 布局决策分散在多个 pass 中，pass 顺序本身承担隐式策略；
- 某个 pass 先建立局部最优，后续 pass 再修补 conversion，可能产生局部最优陷阱；
- 冲突消解仍存在 target/pass-specific heuristic；
- encoding 类型层次和转换清理代码复杂，新硬件特例容易扩散；
- conversion cleanup 能减少已有代价，但不能保证此前锚点选择是全局最优。

## 6. Frisk 的目标布局推断逻辑

### 6.1 当前实现状态与目标架构必须分开

审计起点 Frisk `b120d06` 已完成 M0–M3，原文“LayoutInfer 是空壳”已经过时。下表加入 Task 18 增量（实现提交 `9c28b67`，已合入 main）及 Task 19/20 增量（本地 main 提交 `c028e82`）：

| 已有部件/功能 | 已核对实现 | 当前边界 |
| --- | --- | --- |
| 布局代数与属性 | GF(2) 矩阵、Affine/BitLinear/Product、LayoutProof、Distributed/Storage attrs | 代数库能力不等于所有 map/shape 已接入 pass；Unknown 不可作为硬证明 |
| Storage 纵向切片 | layout_view、whole-tile Copy；Task 18 增加静态 cast/subview 坐标链、有限 origin 投影、全部同-root pairs 的 bit 区间证明 | 同 dtype/space、可静态证明；动态路径、reinterpret、一般 Product alias 投影保守拒绝；不引入隐藏 root assignment |
| Distributed 纵向切片 | Tensor encoding、transpose/同形 elementwise 关系、tile_load/store、显式 conversion | 静态受限形状；bootstrap 每 component 至多 8 vars、每域至多 4 candidates，不是 M5 的通用全局搜索 |
| 结构化控制流 | if/for/while 实际 use/slot 约束与转换物化；Task 18 显式带 kind/slot 的 region edges | 保留 while I/O 两套 tuple；回边仍允许 Convertible，不为制造缩域而改为相等约束 |
| 约束求解/物化 | stable IDs、hard constraints、provenance、确定性有限枚举；Task 18 stable FIFO 删减及真实 degree 上界统计；Task 20 四角色 instruction tuple；detached module 原子提交 | 指令联合合法性已接入，完整 CostVector/指令路径性能联合优化未实现；候选有限初始化不是通用大图最优解证明 |
| 独立核验与测试 adapter | 从实际 IR 重建关系、singleton assignment 核验；受限 conversion lowering 测试 | 不是完整 WGMMA/TMA/mbarrier lowering，没有端到端性能优越性证据 |
| Copy/Fill/Parallel 迁移 | Copy 两端分别读写；Fill 唯一写入者契约；Parallel 的真实 ResourceLimit；四项执行属性物化并独立重验 | 仅受限静态整块、单 CTA；不是完整 lowering；Reduce 尚待 Task 21 |
| Tensor MMA/GEMM 迁移 | `frisk.mma`；SM90a SS/RS f16/bf16→f32；四角色联合约束；fragment/packing/descriptor 实际 IR 纯证明 | 静态二次幂、有界单 CTA；不含 Tensor B、Local fragment、一般 Product/尾块、异步指令 lowering 或数值 GPU 验收 |
| 旧 Op 规则 | legacy 方法保留供 adapter 回归；真实计数验证新 pass 不调用旧 Parallel 递归路径 | 采用公共 collector 接入已迁移规则，尚未实现 Task 22 的完整 Op interface 迁移 |

源码锚点：`LayoutAliasAnalysis.cpp` 的 root/坐标/bit 证明；`StorageAliasCandidates.cpp` 的有限 origins；`LayoutPropagation.cpp` 的 strict/common FIFO；`LayoutRelations.cpp` 的 AliasLayout 证明缓存；`DistributedLayoutConstraints.cpp` 的 SCF 规则；`MaterializeLayouts.cpp` 的实际值重绑和事务提交。上游证据见 §15。

Task 18 对应的审计缺口已在实现分支处理：endpoint-owned 坐标 payload 随 ID remap；同-root 关系由邻接链改为全部 pairs；容量改查 root accessible bit span；RelationsOnly 在任何投影/枚举之前返回。此前 RelationsOnly seed 复制是模式契约缺口，不能倒推旧版已经发生错误验过；本次另有 actual verifier 非邻接冲突反例。

审计 baseline：27 lit、74 unit、4 CTest 全通过。Task 18 新增 30 个 alias/region/integration 单元和 3 份 lit，最终 Gate 为 30/30 lit、104/104 unit、4/4 CTest；命令及边界见实现计划 §18.9。§6.2–6.5 仍是目标分解，不能整体当作现状；完整硬件 pipeline 超出当前范围。

Task 19 再新增 25 个单元（16 个操作集成、9 个执行证明）及 2 份 lit，最终 Gate 为 **32/32 lit、129/129 unit、4/4 CTest**；命令、反例和设计适配见实现计划 §19.9。独立复审的 footprint/view 一致性、物理区间重叠与空 Kernel 回放问题均已修正。

Task 20 再新增 36 个单元（15 个数学/硬件证明、9 个通用联合约束、11 个 MMA 集成、1 个 legacy 分类）及 2 份 lit，最终 Gate 为 **34/34 lit、165/165 unit、4/4 CTest**；命令及不支持边界见实现计划 §20.13。

### 6.2 表示层：Distributed 与 Storage 分离

#### Register/local value

```mlir
tensor<64x64xf32, #frisk.distributed<...>>
```

使用 SSA Tensor 表示逻辑值，`DistributedEncodingAttr` 表达 lane/warp/register ownership。它不是“专门描述硬件寄存器编号”，而是描述逻辑元素到并行执行资源的分布；真实 PTX register allocation 仍由后端完成。

#### Shared/global storage

```mlir
memref<64x64xf16, ..., #gpu.address_space<workgroup>>
```

使用 MemRef 表示真实可寻址 storage，`StorageLayoutAttr` 通过 allocation/layout view/binding 表达 padding、transpose、XOR swizzle 和 bit address。global 通常使用标准 MemRef layout；shared 是首阶段 StorageLayoutAttr 的主要使用者。

#### 显式边界

- distributed → distributed：`frisk.convert_layout`；
- distributed ↔ storage：layout-aware load/store/copy；
- view/transpose/reshape：优先只改变坐标解释；只有 carrier distribution 或真实 storage mapping 改变时才产生搬运。

### 6.3 布局代数：仿射外层 × GF(2) 位线性内层

Frisk 不应让一个表达系统勉强承担所有问题，而应把布局分成：

```text
ProductLayout = AffineOuterLayout × BitLinearInnerLayout
```

#### 仿射外层

负责：

- tile/block/warp-group 的 quotient 和 remainder；
- 非 2 次幂 shape；
- dynamic/ragged boundary 和 predicate；
- padding、stride、base offset；
- logical view 与 storage view 的组合。

它可以复用 MLIR Affine、Presburger、MemRef layout 和 canonicalization 能力。

#### GF(2) 位线性内层

负责固定小 tile 内的位级分布：

- register/lane/warp bits 到 logical coordinate bits；
- XOR swizzle；
- transpose/permute/compose；
- rank、kernel、image、inverse/pseudoinverse；
- coverage、injectivity 和 replication 的精确证明。

#### 组合方式

先由仿射外层确定逻辑点属于哪个大 tile、边界是否有效，再由 GF(2) 内层确定这个 tile 内由哪个 lane/register 持有，或映射到哪个 shared bank/offset：

```text
logical coordinate
  -> affine quotient/remainder
  -> (outer tile coordinate, inner tile coordinate, predicate)
  -> GF(2) transform(inner coordinate / hardware bits)
  -> distributed owner or storage inner offset
  -> affine base + stride + transformed inner offset
```

相比 TileLang，这避免把全部证明绑定到 TVM PrimExpr/Analyzer；相比只使用 Triton LinearLayout，它不要求位线性代数同时承担动态边界、非 2 次幂和物理 padding。详细说明见 [GF(2) 与组合布局说明](./gf2_layout_guide.md)。

### 6.4 Op 不返回最终布局，而是声明约束

新的 Op interface 应返回 `LayoutConstraintSet`，而不是原地修改 `DenseMap<Value, Attribute>`。约束至少包括：

| 约束 | 含义 | 示例 |
| --- | --- | --- |
| `RequireEncoding` | 必须属于指定合法集合 | WGMMA result、用户 mandatory annotation |
| `SameDistributed` | 两个 SSA value 分布相同 | elementwise operand/result |
| `TransformDistributed` | 经过坐标变换后的分布关系 | transpose、reshape、broadcast |
| `StorageCompatible` | alias/view 必须保持物理一致 | shared view、subview |
| `InstructionContract` | 布局必须匹配目标指令 | WGMMA、TMA、ldmatrix |
| `UniqueOwner` | 写入或 reduce 结果拥有唯一 owner | store、reduce |
| `Convertible` | 可在边界插入显式 conversion | 多 consumer 冲突 |
| `Preference` | 可违反但需要计入代价 | coalescing、bank conflict、replication |

Op 可以提供候选生成 hook，但不能遍历任意 IR 或直接提交全局选择。SM90 rule library 负责返回合法候选、指令契约、资源限制和局部成本。

### 6.5 完整推断阶段

#### A. Normalization 与前置验证

- 规范 view/transpose/reshape；
- 修正 MemoryEffect；
- 建立 `#nvvm.target`/DLTI target 信息；
- 将 register fragment MemRef 逐步转为 Tensor SSA；
- 验证 shape、dtype、memory space、region terminator。

#### B. 建立图

- 每个 Tensor SSA value 建 Distributed `LayoutVar`；
- 每个 allocation/layout view/kernel storage 参数建 Storage `LayoutVar`；
- 建立 def-use、alias/view、SCF block argument/yield/loop-carried edges；
- 记录每个 use 的布局要求，不做最终选择。

#### C. Hard seeds

按语义优先级加入：

- 用户 mandatory annotation；
- WGMMA/TMA 的不可违反指令契约；
- 已存在且通过 verifier 的 encoding/binding；
- 显式 conversion 的 source/target；
- scalar/size-one 的 replicated seed。

#### D. Strict propagation

只传播唯一事实，例如 exact same-layout、可逆 transpose、WGMMA parent/operand 关系、alias storage、SCF 类型一致关系。不创建默认布局、不 padding，也不物化 conversion。

#### E. Common fixed-point

在候选集合上运行双向 worklist：

- producer → consumer 传播；
- WGMMA/TMA/store 等强 consumer → producer 反向传播；
- alias/view 传播 storage mapping；
- region join 传播候选交集；
- elementwise/shape transform 传播坐标变换后的候选。

候选 domain 一旦建立，只允许交集、投影和剪枝，保证阶段内单调、可终止、与遍历顺序无关。

#### F. 连通分量候选生成与 hard pruning

SM90 rule library 生成有限候选：

- distributed：blocked、lane/warp-striped、replicated、WGMMA-compatible；
- storage：linear、transpose、padding、32B/64B/128B swizzle；
- copy：vector load/store、cp.async、TMA；
- reduce：shuffle、shared tree、replicated result、elected owner；
- GEMM：合法 WGMMA shape/dtype/major mode；
- conversion edge：直接保留、显式转换、便宜 producer rematerialization。

随后删除违反 shape、alignment、coverage、injectivity、TMA box、WGMMA contract 或 ownership 的候选。

#### G. 联合代价选择

优化单元是 layout constraint graph 的连通分量。一个 assignment 同时包含：

```text
Distributed layout choices
+ Storage layout choices
+ instruction path choices
+ conversion/rematerialization choices
```

第一版采用 hard pruning + 动态规划/有上限的 beam search，不引入无界 ILP/SMT。评估维度包括：

```text
illegal_or_unsupported              // 必须为 0
instruction_path_and_estimated_work // WGMMA/TMA/cp.async/SIMT 与实际 shape
global_memory_transactions
shared_bank_conflicts
conversion_bytes_and_sync_on_critical_path
spill_risk_and_registers
shared_bytes_and_occupancy
replication_and_code_size
deterministic_tiebreak
```

成本选择不能把“conversion 数量最少”当作绝对目标。例如一个 conversion 如果能让主 GEMM 命中 WGMMA，通常优于零 conversion 的 SIMT 路径；反之，为收益很小的局部 coalescing 在热循环内引入重型 shared conversion，也可能得不偿失。

#### H. Controlled relaxation

**显式 conversion 本身不应只在“零转换无解”时才出现。**它是普通候选域中的一等决策变量。只有普通合法候选全部失败时，才进入 relaxation：

1. 增加受控 replication；
2. 增加 shared padding 或降低 vector width；
3. 采用 guarded/ragged fallback；
4. TMA 回退 cp.async/vector copy；
5. WGMMA 路径不合法时回退到 warp MMA/SIMT。

每次 relaxation 必须带 provenance 和新增成本，不能静默改变 mandatory annotation、writer ownership 或 alias 语义。

#### I. 求解结果验证

检查：

- Attr 格式与 canonical form；
- tensor shape/dtype 与 encoding；
- distributed coverage/replication/owner；
- storage injectivity/alignment/allocation size；
- alias/view 一致性；
- WGMMA/TMA/copy contract；
- conversion source/target 的合法性和必要性；
- 所有 layout-bearing value 已解析。

诊断应报告 source location、冲突候选、触发约束、传播链和一个具体坐标反例。

#### J. 物化与 conversion 优化

- 将 Distributed 结果写入 Tensor encoding；
- 将 Storage 结果写入 allocation/layout binding；
- 同步更新 region block arguments、yields 和 results；
- 在求解选择的边界插入 `frisk.convert_layout`；
- 删除 identity/相邻可合并 conversion；
- hoist loop-invariant conversion；
- 对便宜 pure slice 做受控 rematerialization。

conversion cleanup 只优化已经验证的选择，不能承担“修复错误推断”的职责。

#### K. SM90 lowering

长期目标流水线为（不是当前可运行 pipeline，也不属于本计划完整指令 lowering 的交付承诺）：

```text
frisk-normalize-layout-ir
  -> frisk-infer-layouts
  -> frisk-verify-layouts
  -> frisk-materialize-sm90-pipeline
  -> convert-frisk-to-vector-memref-gpu-nvgpu
  -> convert-nvgpu-to-nvvm
  -> LLVM/PTX
```

通用 solver 不直接构造 WGMMA/TMA Op；SM90 adapter 把已经选定并验证的 instruction contract 转换到 NVGPU/NVVM 或必要的 target-specific op。

## 7. 各环节横向对比

| 环节 | TileLang | Triton | Frisk 目标方案 |
| --- | --- | --- | --- |
| 基础 IR | TVM/TIRX Buffer + TileOp | MLIR TTIR/TTGIR Tensor SSA + memdesc | MLIR Frisk Tensor SSA + MemRef |
| Register 表示 | local/fragment Buffer + Fragment layout | RankedTensorType encoding | RankedTensorType + DistributedEncodingAttr |
| Shared 表示 | Buffer + Layout | memdesc/storage encoding | MemRef + StorageLayoutAttr binding |
| 初始布局 | 用户 annotation、TileOp strict rule | TTIR→TTGIR 默认 BlockedEncoding | hard seeds；未解析值不急于给默认布局 |
| 推断时机 | pipeline rewrite 后、LowerTileOp 前 | TTGIR 转换及多个 GPU optimization pass 中反复调整 | normalization 后统一分析，SM90 pipeline materialization 前提交 |
| 局部规则 | 每个 TileOperator 的 `InferLayout` | coalesce、dot/MMA、descriptor 等 target pass | Op constraint interface + SM90 rule library |
| 全局状态 | `Map<Buffer, Layout>` | value → encoding candidates，类型直接携带结果 | LayoutVar domain + constraint/provenance graph |
| 传播方式 | strict + BFS common + free root search | anchor → descendants propagation，多轮 pass | strict + 双向 fixed-point + component candidate solve |
| 反向推断 | TileOp 可依据已知 buffer 推另一端 | inferSrcEncoding / backward rematerialization；SCF 穿越有边界 | 一等能力；consumer contract 可反推 producer/storage |
| Alias/view | 同 data 分组、位宽感知 reshape，非任意 slice 证明 | memdesc view + 独立物理 BufferRegion 分析 | Task 18 显式坐标关系 + 全部同-root pairs 的 storage hard constraints；支持子集与 Unknown 明确 |
| 多 consumer | 尽量找到一个共同 Buffer 布局 | 可保留局部不同 encoding，以 conversion 分隔 | Task 24 在赋值评分中比较同块共享转换与各 use 的受限重计算，不跨 region 共享 |
| 冲突处理 | containment、swizzle merge、free root；仍冲突则失败 | heuristic 选 encoding并插 conversion | hard conflict 诊断；soft conflict 进入全局候选选择 |
| Conversion | 不是通用核心抽象 | 显式 ConvertLayout，随后多轮优化 | 显式且是求解变量；求解后统一物化 |
| 成本重点 | 默认 spill/register；可选 I/O-aware component attempt 评分 | pass-local hardware heuristic + conversion cleanup | Task 23/24 的边界工作、资源和循环深度代理，加 Task 25 的 atom/Copy 工作、32B sector、按 128B 阶段统计的 bank 代理；不是 cycle 模型 |
| Region/loop | TIR loop/annotation、reducer epoch 语境验证；非通用 SSA join 图 | if/for/while 正向传播/重写；while 区分双 tuple，反向 remat 有边界 | M3 SCF 语义 + Task 18 显式边与有限域收敛统计 |
| 验证 | Analyzer/ICHECK/专用 validator | MLIR verifier + target-specific checks | MLIR verifier + 数学证书 + provenance diagnostic |
| 代数 | PrimExpr/IndexMap/Fragment/PartialFragment，部分 CuTe 访问代数 | encoding + LinearLayout；已有 padding/linear 组合 | Affine outer × GF(2) inner，实际支持范围见 §6.1 |
| Target 耦合 | TileOp/layout 与 TVM CUDA lowering 紧密 | target-specific encoding/pass 较多 | 通用 solver 与 SM90 rule/adapter 分层 |
| 默认失败策略 | 找不到共同布局时冲突 | 插 conversion 或 target fallback | 先全局选成本最低合法解；无普通解才受控 fallback |

### Task 25 方案 A 实现增量（2026-09-28）

既有 FP16/BF16 WGMMA 证明已整理成 24 条目标规则；Copy 新增独立 TMA/cp.async/vector 路径变量，与 source storage、destination storage、execution 联合求解。binding 同时记录 descriptor、guard 和未来 lowering 的同步义务，并从物化后实际 SSA/地址独立重建。静态 3×5 等 Copy 使用二次幂 carrier 和原始有效域保护，首版只走 scalar；不等同于 TileLang 动态通用布局能力。

前三项地址/形状代价现已接入真实 solver，而非离线评分：同类已对齐 fixture 中 32×64 F32 Copy 选 TMA，8×16 选普通向量；转置存储的独立消费者可以为减少 sector 接受一次 shuffle。硬约束先排除非法高速路径，随后才比较工作量与转换代价。未知对齐相位保守处理；Shared bank 按硬件独立 128B 阶段计数，读广播不等同于不同地址冲突。

可验证的区分点是 **双域＋指令路径的联合关系、有限域证据、事务物化与实际 IR 复验**，不是声称 TMA、swizzle、guard 或成本模型本身独有。具体支持域、代理公式、325 unit 等 Gate 见实现计划 §25.1–25.5。完整 GPU lowering、动态 ragged、通用 relaxation、运行时收益及大规模编译成本仍待后续验证；此次没有扩大对固定 TileLang/Triton 快照的能力结论。

## 8. 同一个 SM90 WGMMA 数据流在三者中的处理

本节是机制示意；尤其 §8.3 为 Frisk 未来设计，不是本轮编译/运行结果。

考虑：

```text
global A/B
  -> shared A/B（希望 TMA + 128B swizzle）
  -> WGMMA
  -> accumulator
      ├─ consumer 1: elementwise epilogue
      └─ consumer 2: coalesced global store
```

假设 WGMMA accumulator 的最佳 distributed layout 与最终 store 的最佳 coalesced layout 不同。

### 8.1 TileLang

1. TMA/Copy、Gemm、Parallel 等 TileOp 被全局 pass 收集；
2. Gemm strict rule 推出 accumulator fragment 以及 A/B shared layout 要求；
3. Copy rule 将 shared layout 与 global copy/thread mapping 关联；
4. epilogue 和 store consumer 尝试接受/传播同一 accumulator Buffer layout；
5. 如果 fragment 关系可以 containment/replication 兼容，则选共同布局；
6. free-mode 比较 root attempts，按默认 spill/register 或 opt-in I/O-aware policy 评分；
7. 如果 store 所需布局与 accumulator 布局无法兼容，也没有 Op-specific lowering 处理该差异，则发生冲突或走专门 fallback。

这里的核心目标是：**尽量让一个 Buffer layout 同时满足整条链。**

### 8.2 Triton

1. 初始 TTGIR Tensor 具有 BlockedEncoding；
2. Coalesce 为 load/store 建立合并访存布局，并暂时插入 conversions；
3. AccelerateMatmul 为 dot result/operands 建立 MMA 相关 encoding，并插入 conversions；
4. MMA、昂贵 store、descriptor 成为 anchors；
5. `RemoveLayoutConversions` 从多个 anchor 传播 encoding；
6. accumulator/epilogue 可以继续使用 MMA-compatible encoding；
7. store 边界若仍需要不同 encoding，则保留 conversion；如果 epilogue 很便宜，可能 rematerialize 或移动 conversion；
8. lowering 将剩余 conversion 转成 register shuffle 或 shared-memory exchange。

这里的核心目标是：**允许多个硬件局部最优共存，再优化它们之间的显式边界。**

### 8.3 Frisk

Frisk 不在首次看到 WGMMA 或 store 时立即提交布局，而是建立一个联合约束分量：

```text
Storage(A_shared) --TMA/WGMMA contract--+
Storage(B_shared) --TMA/WGMMA contract--+--> Distributed(acc)
                                              | same/transform
                                              +--> epilogue
                                              |
                                              +--convertible--> store layout
```

候选至少包括：

- A：accumulator 和 epilogue 保持 WGMMA encoding，store 前转换；
- B：便宜 epilogue 在 store encoding 下 rematerialize；
- C：整段使用一个兼容布局，完全不转换，但可能降低 store coalescing；
- D：改变 shared swizzle/copy path，但仍满足 WGMMA contract；
- E：WGMMA/TMA 不合法时才考虑 cp.async 或更低性能计算路径。

solver 比较整条 critical path 的 WGMMA/TMA 命中、global transaction、bank conflict、conversion bytes/sync、register 和 occupancy 后选择。通常 WGMMA result 是强锚点，因此更可能选择 A 或 B，而不会为了“零 conversion”破坏主计算路径。

这里的核心目标是：

> **在一次联合决策中同时选择硬件路径、两类布局和转换位置，而不是先固定局部布局再完全依赖 cleanup 修补。**

## 9. Frisk 这样设计的好处

### 9.1 相比直接迁移 TileLang

1. **保留 MLIR 类型安全。**不同 distributed layout 是不同 Tensor type，错误 use 不能被一个无类型 `DenseMap` 静默掩盖。
2. **多 consumer 不再要求一 Buffer 一布局。**同一 SSA value 可以在不同 use 边界显式转换，不必为了兼容最弱 consumer 牺牲主路径。
3. **控制流处理更自然。**SCF block argument、yield、loop-carried value 和 result 的 encoding 通过类型一致性约束统一处理。
4. **alias 与 storage 语义分开。**MemRef/AliasAnalysis 处理真实 storage；Tensor SSA 处理 register ownership，避免把两类关系混在 Buffer map 中。
5. **复用成熟基础设施。**Dialect Conversion、DataFlow、Dominance、SideEffect、Affine、Presburger、Vector、GPU、NVGPU、NVVM 都可以直接接入。
6. **长期独立。**Frisk 只审计 TileLang 的语义、规则和测试，不依赖 TVM/TIRX ABI。

### 9.2 相比机械复制 Triton

1. **布局选择更集中。**锚点、storage、distributed、conversion 和资源成本在同一 constraint graph 中可见，减少 pass 顺序造成的隐式决策。
2. **候选冲突有明确 provenance。**不仅知道“插了 conversion”，还知道由哪两个 Op、哪条指令契约和哪次传播导致。
3. **组合代数覆盖边界语义。**GF(2) 负责精确位线性内层，Affine/Presburger 负责非 2 次幂、dynamic、padding 与 ragged，不让一种代数承担所有问题。
4. **target 特例受控。**SM90 rule library 产生候选和成本，通用 solver 不散布 `if (sm90)`。
5. **conversion 是前置决策变量。**不是只有布局都定完之后才被动清理，可以直接比较“共同布局”和“局部好布局 + conversion”的端到端成本。

### 9.3 对长期扩展的价值

- 新增 Op：实现 constraint interface 和 verifier，不必修改全局遍历逻辑；
- 新增硬件：增加 target rule library/adapter，不改核心布局图；
- 升级 MLIR：IR、analysis、conversion 和 lowering 仍遵循标准接口；
- 接 autotuning：有限候选和稳定 cost key 已经形成清晰入口；
- 做性能回归：可以按 candidate、conversion、register、bank conflict 输出统计，而不是只比较最终 PTX。

## 10. 差异化假设与可核验创新边界

双域区分、GF(2)、padding、显式 conversion、alias/view、while、fixed-point、候选评分本身都不是 Frisk 独有。TileLang 的 PartialFragment/位宽视图与 Triton 的 BufferRegion/PaddedSharedEncoding 进一步说明：不能通过省略对方能力建立“创新”。

| 假设 | 与固定参考实现比较的具体点 | Frisk 当前状态 | 验证/否证条件 |
| --- | --- | --- | --- |
| H1：坐标化 alias 与 SSA region 关系在同一 hard-constraint 图内协作 | 对比 TileLang buffer 兼容闭包及 Triton 分离的布局传播/物理区域分析；比较作用范围，不说对方没有坐标分析 | Task 18 坐标 alias/显式边已编码，支持受限静态域；尚无性能结论 | 偏移/步长/降秩、三视图冲突、真实回边重调度；actual-only verifier 无隐藏 assignment。Tensor/storage 经访问关系协作，不代表任意跨域组合全覆盖 |
| H2：可审计的有限域终止与冲突解释 | 公开冻结候选数、实际删除数、queue pops、degree 上界，关联 seed→alias/region→冲突 | Task 18 worklist/stats、finite origins、proof cache 已编码 | 插入顺序扰动下 assignment/统计相同；重复 pass IR 相同；真实事件逐项对账，Unknown 拒绝。该性质是当前受限实现的工程差异，不是对上游整体的否定 |
| H3：组合代数扩大受支持布局域而保持证明一致 | 比较具体 Affine/GF(2) 组合域与 TileLang IndexMap/CuTe、Triton padded/linear 的交集及差集 | M1 代数已有，pass 子集受限 | 同坐标语义、coverage/injectivity/owner 与反例；非零 offset XOR 必须正确处理 carry。若只换一种文本表达，不构成新增能力 |
| H4：布局、指令路径和 conversion placement 联合选择改善结果 | 与固定 pipeline/attempt policy 比较合法候选及最终代价，不假定全局最优 | Task 23 有界搜索、Task 24 同块边界评分、Task 25 Copy 路径和物理地址评分已有真实选择用例；尚未验证完整 SM90 路径性能或 GPU 收益 | 固定硬件/输入/编译选项，报告 runtime、访存、bank、寄存器/共享内存和编译时间；超出计划回退阈值或无收益则不宣称性能优势 |

语义差分是验证方法，不是独创性结论。旧 Frisk/旧 TileLang 仅为已知正确子集的回归 oracle；新版机制、历史 bug 修正须用独立的坐标/owner/address 不变量及行为变更记录核验，不能为了“和旧版一致”保留错误。Task28 已建立并运行固定版本 corpus，20 个等价比较和 3 个实际拒绝分别通过；具体支持域及策略差异见本文 M6 摘要，不能推广为任意上游布局或性能等价。

Task 18 审计时点的历史边界：当时只实现 H1/H2 的受限切片，不扩展 GEMM/Reduce、cost solver、硬件 lowering；whole-root alignment 前置契约若被 Pure/DCE 删除，actual verifier 会保守拒绝。此后 Task 19–24 的实际增量见本文摘要；Task 22 已提供受支持 normalization/pipeline 的持久根契约，不能再把上述历史限制当作当前全部功能，也不由这些工程增量推断研究独创性或性能收益。

## 11. Frisk 架构决策

### 11.1 已决定采用

| 决策 | 结论 | 原因 |
| --- | --- | --- |
| Register carrier | SSA Tensor + DistributedEncodingAttr | 类型安全、def-use/region 可组合、显式 conversion |
| Storage carrier | MemRef + StorageLayoutAttr binding | 保留可寻址、alias、alignment 和 memory effect 语义 |
| Op interface | 返回 constraint/candidate，不直接改最终 map | 支持冲突、候选、代价和 provenance |
| 推断框架 | strict + common fixed-point + component solve | 兼顾确定事实传播和多 consumer 全局选择 |
| Conversion | 一等候选变量，求解后显式物化 | 不把零转换绝对化，也不依赖事后修补 |
| 布局代数 | Affine outer × GF(2) inner | 同时覆盖动态/不规则外层与位线性硬件内层 |
| Target 策略 | 首阶段只支持 SM90/SM90a | 收敛规则和性能验证范围 |
| TileLang 关系 | 最新审计 commit 作语义 oracle，不链接/不 vendor | 吸收经验，保持长期独立 |
| Triton 关系 | 参考类型、anchor、conversion/remat 机制，不复制 pass 树 | 复用 MLIR 思想，避免特例扩散 |

### 11.2 明确不采用

- 不迁移 TVM/TIRX/PrimExpr/Buffer 作为 Frisk layout ABI；
- 不继续以 `memref<..., local>` 作为长期 register value 模型；
- 不让 Op 访问并修改一个全局 `DenseMap<Value, Attribute>` 后立即锁定布局；
- 不要求整张图零 conversion；
- 不允许 conversion cleanup 修复不合法的推断结果；
- 不在通用 solver 中散布 SM90 intrinsic 构造；
- 首版不引入无界 ILP/SMT，也不追求任意 target 的抽象完备性。

### 11.3 Frisk 对 TileLang/Triton 的取舍

```text
保留 TileLang：
  - TileOp 强语义
  - strict/common/free 分阶段思想
  - alias/storage-bit 一致性
  - 语义差分 corpus

拒绝 TileLang：
  - TVM/TIRX 类型依赖
  - Buffer 单布局作为唯一公共模型
  - 冲突主要靠合并或失败

保留 Triton：
  - Tensor encoding
  - 显式 ConvertLayout
  - anchor propagation
  - dominance/region/rematerialization 优化

拒绝机械复制 Triton：
  - 大量特化 encoding/pass 树
  - 依赖 pass 顺序形成的隐式全局策略
  - 先局部定布局、再主要依靠 cleanup 修补
```

## 12. 风险、性能边界与验收标准

### 12.1 主要风险

| 风险 | 后果 | 控制方式 |
| --- | --- | --- |
| 候选组合爆炸 | 编译时间不可控 | 连通分量、hard pruning、canonical hash、beam 上限、统计诊断 |
| Cost model 判断错误 | 选择多余 conversion 或低性能路径 | 保守规则、benchmark 白名单、feature flag、稳定 fallback |
| Tensor SSA 改造范围大 | 影响前端与 lowering | 过渡 op/adapter，分阶段替换 local MemRef |
| GF(2)/Affine compose 错误 | ownership、OOB、bank mapping 错误 | property test、枚举小 tile、SM90 contract verifier |
| conversion lowering 过重 | shared traffic、同步和寄存器增加 | 估算 bytes/sync/critical path，hoist/remat，性能守门 |
| target rule 与 NVGPU 不匹配 | 合法布局无法 lowering | 固定 LLVM commit，SM90 adapter 测试，明确 unsupported |

### 12.2 正确性验收

- 所有 layout-bearing SSA value 和 storage anchor 均已解析；
- region/block argument/yield/result 类型一致；
- distributed coverage、replication、unique owner 正确；
- storage injectivity、allocation size、alignment、alias 正确；
- WGMMA/TMA/mbarrier contract 完整；
- ragged/non-power-of-two/sub-byte dtype 无 OOB/race；
- 推断结果与遍历顺序无关。

### 12.3 性能验收

每个关键 kernel 至少记录：

- WGMMA/TMA/cp.async 是否命中；
- runtime、吞吐与端到端延迟；
- global transactions/coalescing；
- shared bank conflict；
- `convert_layout` 数量、bytes、同步及所在循环层级；
- register、spill、shared bytes、occupancy；
- solver candidate 数与编译时间。

重要边界：

- 不为了追求“创新布局”破坏已验证的 WGMMA/TMA 主路径；
- 对等 kernel 不应无原因增加热循环 conversion；
- 与手写/TileLang/Triton baseline 稳定回退的候选不能默认开启；
- 任何性能结论必须在固定 shape、dtype、stage、clock/测量协议下比较。

## 13. 2026-09-12 审计决策记录

| 证据/缺口 | 决策 | 对应任务与状态 |
| --- | --- | --- |
| TileLang bit-aware alias/widening 旧版已有；same data 不足以表达任意 slice | 采用 storage-bit/owner 不变量，调整为显式坐标关系；拒绝直接 union 为布局相等 | Task 18 已编码；dtype reinterpret 本阶段拒绝 |
| TileLang 新 PartialFragment、dst-steering、epoch 语境扩展与 annotations ABI | 迁移前逐 Op 建新旧行为表；区分同值 replica 和 partial addend，区分读/写规则 | Task 19–21 后续设计输入，不在本轮实现 |
| Triton while 两套 tuple 已有，backward remat 有局限 | 保留 Frisk M3 两套 tuple；显式记录 region edges，不扩大转换移动权限 | Task 18 已编码，真实 SCF 回归保留 Convertible |
| Triton BufferRegion 已建模物理区域，但不是布局候选 solver | 借鉴 root/frame/地址集合与 Unknown；适配到 Frisk alias hard relation | Task 18 已编码；不照搬 memdesc 的专用 same-encoding 限制 |
| Frisk Attr==、邻接 alias 链、root capacity 与地址原点约定不足 | 共同 root 地址契约、逐对坐标/bit 区间证明，处理非相邻重叠和非预期碰撞 | Task 18 实现分支已修复，并有先失败后通过的集成用例 |
| Frisk 增长 closure 与多轮扫描尚无统计契约 | 有限候选初始化与单调删减分开；stable worklist、真实 degree 统计 | Task 18 已编码；Strict/Common 逐项账本、冲突与确定性测试 |
| 原比较称 pass 空壳、成本模型只有寄存器，或把目标写成完成 | 本轮修正完成度和固定证据，创新改成待检验假设 | 本文与实现计划已同步 |
| 旧文档曾对 conversion 普通候选/失败后 fallback 有歧义 | 维持计划现有决策：conversion 为普通候选，controlled fallback 仅处理普通域无解 | 历史问题已在现行设计/计划澄清，不再列为待修订缺陷 |

[Task 18 详细设计、验收矩阵与实现记录](./layout_inference_implementation_plan.md#task-18-完成-aliasview-与-region-graph)保存了先审计设计、用户确认、红灯测试、实现与复审的过程。复审补强向量地址保证、对齐依据 provenance 和 region/图 ID 不变量；长期文档、公共属性/Op 注释同步，push/merge 仍须用户另行指示。

## 14. 最终结论

TileLang、Triton、Frisk 不应被简化成“早推断”和“晚推断”的区别。更准确的差异是：

- TileLang 以 Buffer 兼容闭包为中心，布局在高层 TileOp lowering 前冻结；
- Triton 以 encoded SSA 和硬件锚点为中心，在多个 TTGIR pass 中迭代布局，并用显式 conversion 连接局部最优；
- Frisk 应以 MLIR-native 的双域布局和全局约束图为中心，在物化前联合选择布局、指令路径和 conversion placement。

Frisk 的价值不是单独“比 TVM 更好”或“比 Triton 更新”，而是利用 MLIR 将以下能力组合成一套可长期扩展的系统：

```text
类型安全的 SSA
+ 明确的 storage/alias 语义
+ TileOp/SM90 强约束
+ Affine × GF(2) 精确代数
+ 全局有限候选选择
+ 显式且可优化的 conversion
+ 可解释验证与性能守门
```

在该架构下，Frisk 可以吸收 TileLang 和 Triton 的成熟经验，同时保持自己的 IR、求解器、目标规则库和长期演进路径。

## 15. 主要参考源码与文档

### TileLang

- [TileLang CUDA pipeline（固定审计 commit）](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/tilelang/cuda/pipeline.py)
- [TileLang LayoutInference（固定审计 commit）](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_inference.cc)
- [TileLang layout 实现目录](https://github.com/tile-ai/tilelang/tree/5e149e31674658f94779c7d0c6039549a1853123/src/layout)

源码定位与静态测试（以下均已审阅，未执行上游测试）：

- [alias / reducer widening / attempt 调度](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_inference.cc#L311)：`propagate_alias`、`RunOneAttempt`。
- [PartialFragment](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/layout/layout.h#L256)、[reinterpret/reshape 与 inverse 检查](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/layout/layout.cc#L140)、[前端 view/reshape](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/tilelang/language/customize.py#L60)。
- [代价接口](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_cost_model.h#L38)、[代价实现](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_cost_model.cc)、[默认 policy](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/config.h#L38)。
- [epoch 语境验证](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/verify_reducer_epoch.cc#L104)、[Parallel 规则](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/parallel.cc#L331)。
- [Copy 新 ABI](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/copy.cc#L619)、[CUDA Copy 区域规则](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/cuda/op/copy.cc#L584)、[GEMM 读写/契约](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/gemm.cc#L164)、[传统 Reduce](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/reduce.cc#L195)。
- [位宽 view 测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/language/test_tilelang_language_view.py)、[reducer v2 测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/language/test_tilelang_language_reducer_v2.py)、[attempt 隔离测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/transform/test_tilelang_transform_reducer_scalar_candidates.py)、[推断/成本测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/transform/test_tilelang_transform_layout_inference.py)、[inverse 测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/layout/test_tilelang_layout_inverse.py)。
- [历史 LayoutInference（6623b12，仅作新旧比较）](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/transform/layout_inference.cc)。

### Triton

- [TTIR → TTGIR TypeConverter](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp)
- [Coalesce](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)
- [AccelerateMatmul](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)
- [RemoveLayoutConversions](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)
- [LinearLayout](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/include/triton/Tools/LinearLayout.h)
- [NVIDIA backend pipeline](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/third_party/nvidia/backend/compiler.py)

- [while forward 与 conflict heuristic](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L287)、[backward slice 限制](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/Utility.cpp#L900)。
- [BufferRegion 契约](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/include/triton/Analysis/BufferRegion.h#L39)、[物理地址与 view 分析](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Analysis/BufferRegion.cpp)、[ConSan 使用点](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonInstrument/IR/Utility.cpp#L698)、[Membar 使用点](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Analysis/Membar.cpp#L38)。
- [PaddedSharedEncoding](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L208)、[memdesc verifier](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/IR/Ops.cpp#L579)。
- [while/转换静态测试](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/test/TritonGPU/combine.mlir#L1896)、[物理布局 alias 测试](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/test/Analysis/test-buffer-region-layout-alias.mlir)、[动态/重解释 alias 测试](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/test/Analysis/test-buffer-region-alias.mlir)、[LinearLayout 单测](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/unittest/Tools/LinearLayoutTest.cpp)。

### MLIR

- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [DataFlow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/)
- [Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [MemRef Dialect](https://mlir.llvm.org/docs/Dialects/MemRef/)
- [NVGPU Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [NVVM Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)

### Frisk 当前代码

- [LayoutAttr](../include/Dialect/Frisk/IR/FriskAttributes.td)
- [新布局 Op/Attr 接口入口](../include/Dialect/Frisk/IR/FriskLayoutInterfaces.h)（接口定义存在不等于全部 Op model 已迁移）
- [当前 LayoutInfer pass](../lib/Dialect/Frisk/Transforms/LayoutInfer.cpp)
- [当前 Op 局部布局逻辑](../lib/Dialect/Frisk/IR/FriskOps.cpp)
- [当前 Reduce 布局逻辑](../lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp)

- [布局属性](../include/Dialect/Frisk/IR/FriskLayoutAttrs.td)、[代数/证明](../include/Dialect/Frisk/Analysis/LayoutAlgebra.h)。
- [约束图](../include/Dialect/Frisk/Analysis/LayoutConstraint.h)、[传播/alias root](../lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp)、[关系核验](../lib/Dialect/Frisk/Analysis/LayoutRelations.cpp)。
- [SCF 与 Tensor 约束](../lib/Dialect/Frisk/Analysis/DistributedLayoutConstraints.cpp)、[求解 verifier](../lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp)、[原子物化/实际 IR 核验](../lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp)。
- [SCF 回归](../test/Transforms/materialize-scf.mlir)、[传播单测](../unittests/Dialect/Frisk/Layout/LayoutPropagationTest.cpp)。

本地链接用于后续维护；上述“当前实现”结论固定到 Frisk `b120d06`，不能用之后的文件内容反推此次已实现范围。

## 后续增量：简化 Attention 验收（2026-10-10，方案 A 已完成）

增量前审计证明既有子集可以推断 Copy→MMA→Reduce→Store，但当时 `linalg.broadcast` 无规则，不能据 M6 完成宣称 Softmax/Attention 已支持。本轮已修复跨函数/分支 Global 契约 SSA 根及 packed 零输出位宽两处合法输入拒绝，并新增独立 `BroadcastLayout` 关系。

本增量以 SM90a/128 线程、Q/K/V=`64×16xf16` 验收真实 `cast_f16(softmax(QKᵀ×0.25))V` 的布局推断：第一次 SS MMA、第二次 RS MMA，f32 稳定 Softmax，Global 输入/输出。广播使用坐标投影约束、有限候选传播、显式转换与实际 IR 复验，而非按 Op 顺序修改外部布局 map，也不复制上游 encoding 分类。保留 exact256/beam32/65536。

实际候选策略是有限输出布局种子→删除广播轴、压缩冗余 register 基→输入 use；输入只过滤输出支持，不伪造广播的唯一逆映射。实际证明逐输出持有者检查所需源坐标在同线程可用，支持源复制及寄存器重编号。合法时显式插入源 use 转换，实际 IR verifier 不依赖求解阶段候选表。静态非零 rank、各轴大于 1 的 2 次幂边界保持明确；广播不是不同 shape 的 SameLayout。

真实示例见 [`attention-layout.mlir`](../test/Transforms/attention-layout.mlir)，从无 Tensor encoding 输入完成整链物化、实际布局复验、重复推断及文本重放；失败保持原 IR。最终 **417 unit / 61 lit / 7 Debug CTest / 1 Release generator CTest / 5 Python normalization** 及 API/tool 检查通过，独立复审无阻塞问题。

这是 Frisk 自身完整算子链验收，不是与 TileLang/Triton 的完整策略等价、性能优势或 Attention GPU 执行证据。未新增上游版本审计或比较性能测量；无 causal/dropout/尾块/online Attention/通用 reshape/完整 WGMMA/TMA lowering。
