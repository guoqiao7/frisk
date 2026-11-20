> 阶段更新（2026-11-05）：冻结源码摘要`7cf13ffa82ef`的新网络后端已完成本轮实现、执行回归、选择质量与完整费用复验。新6组的regret门槛及21组相对legacy退化保护通过；至少3%的性能晋升条件未通过，默认继续使用legacy。
> 当前范围、原始证据与限制见[本轮完整报告](../inference_detail/network_selection_execution_report.md)。后文按时间保留历史记录，其中“当前”以对应阶段日期为准。

## 2026-11-05：新网络后端验收

- 正式比较固定C转换模式、128线程、单CTA：183候选完成数值与安全门禁，三Attention ABI在回归范围内。独立tile多CTA另覆盖grid=1/2/257；不包含长序列KV循环或跨CTA通信。
- 新留出GM regret上界1.013448、最坏上界1.044640；21组退化保护通过。J/legacy时间比0.998293，95%区间[0.994599,1.003304]，尚无预登记3%收益。已有回归和新留出分别报告。
- 首次／热缓存prepare+validate实际15386.618／10748.236秒，缩短约30.1%，包含严格安全证明复用与额外核验。热搜索仍重跑，完整热准备本身更长；缓存收益与验证费用分项保存。
- native14仍有75个unsupported候选及已观察排序缺口；LOCAL额外选择效应未由该矩阵识别。已有GF(2)访存轴的非冗余证据已登记，不等同于新性能增益。
- 本阶段已完成所选支持域验收；Triton/TileLang同算法、ABI、精度与调优预算的外部比较尚未开展。新版本保留legacy基线与实验策略，所有失败／中止及原始计时保留。

---

> 阶段更新（2026-11-03）：存活需求 v6、通信边际费用和 native/GF(2) 联合评分已实现，并完成六个新性能留出复验。整体选择质量未通过，默认仍为 legacy，尚不进入 Triton/TileLang 外部公平对比冻结。
> 六来源候选池的 kernel 平均收益与额外搜索/编译开销已分别记录；conflict 的 LOCAL 未覆盖触发整池回退是当前必要修复。
> 详见[本阶段完整实现与验收报告](../inference_detail/native_cost_repair_report.md)。以下保留之前版本的原始记录。

## 2026-11-03：需求覆盖与同预算搜索复验（整体未通过）

本轮执行覆盖与预算内搜索修复已经实现并复验：114配置、16002数值/保护记录、144次实际sanitizer全部通过；36次受控Oracle和预算核对通过。原始计时150个配对批次/1800事件，无无效或替换批次。

整体模型及选择质量仍未通过：组合主时间比0.940789，但exp32×32退化44.9%；固定新搜索时，新成本模型反慢6.38%。自然域regret和单例保护失败，默认保持legacy-v2，不进入公平对比版本冻结。收益主要来自搜索修复，不能称新模型已经取得性能优势。新模型CPU solve合计约为legacy的9.25倍，仍需优化。

完整报告：`guoqiao/inference_detail/demand_cost_search_repair_report.md`；证据与保留的CPU失败历史在`guoqiao/optimize_plan/demand_cost_search_repair_artifacts/`。新性能标签只在冻结后的R4读取，没有调权或换case；以下原文逐字节保留，作为历史状态。

---

## 2026-11-02：模型及选择质量验收完成，整体资格未通过

84/84配置通过严格GPU闭环；11508条普通数值/保护/stream/graph/ABI记录、136真实sanitizer进程、50配置安全精确复用通过。15组/34组内执行镜像完成150配对批次、1700event，24受控精确oracle通过；恢复前29配置与首轮47重叠包保持。

calibrated-J在六个新自然程序相对legacy-J的几何平均时间比1.279 [95%CI 1.261,1.289]，晋升失败，默认保留legacy。legacy-J相对充分知情LI在该自然已选镜像并集快1.362×，但受控diamond两种J最差regret1.1126、未过1.10门槛；保留默认不代表已全面合格。Attention三层四臂同镜像，无独立J收益。

已证明两类缺口：mixed成本计价覆盖4192而真实demand逻辑warp-value覆盖128（32.75×，不是时延比）；六个自然calJ模型成本均高于合法LI赋值。后者在GPU上多数仍比calLI快，必须先对齐执行覆盖/排序，再做同预算incumbent保底与新留出验收。当前不进入跨框架版本冻结。

详见[本阶段报告](../inference_detail/model_selection_quality_report.md)；旧内容完整保留于下方。

---

## 2026-11-02：支持域与当前GPU闭环完成

当前GF(2)/legacy-v2/calibrated-v3版本的11自然程序/形状/ABI×2模型全部22配置完成推断→紧凑/普通导出→native→GPU闭环；3654数值/保护/stream/graph/ABI记录通过，60真实sanitizer进程与7精确安全复用。四个calibrated大图Eall导出缺口已用符号Register计划和同线程需求驱动标量化解决，保持selected布局与原形状。

支持域分五层登记，GPU已验证范围是SM90a单CTA128线程、四根f16/f32/固定行主序与分层ABI；保留更广shape/控制流/多CTA/非dyadic等边界。资源实测不当作时延收益，无GPU性能比较或静态GPR预测达标声明。详细支持矩阵、数值/安全/身份与失败记录见[本阶段报告](../inference_detail/current_gpu_closure_report.md)。下一阶段仍为模型/选择质量资格与公平框架，未自动开展或提交。

## 2026-11-01：GF(2) 符号化成本覆盖修复

原五个自然候选域 cost-failure 已通过商空间读计数、最小 first_owner 截面与正交补转换证明修复；保留原形状、候选域和搜索预算。正式 64 CLI 对照中，新版 32 次均成功，五原失败配置的缓存开/关 10 次均 committed；其余 11 配置的 IR、成本和搜索行为保持一致。

成本语义版本为 legacy/static-v2、calibrated-v3；冻结 v2 系数文件格式与 SHA 保持不变。访问逻辑 footprint 65536、非 dyadic fallback 及其他执行/证明上限仍明确保留。本轮不形成 GPU 性能结论。完整数学依据、验证与源码桥接见[本阶段报告](../inference_detail/symbolic_access_cost_repair_report.md)。

> 2026-11-01 搜索开销优化更新：执行闭环状态：本轮优化CPU推断，已登记支持域的IR与转换保持一致，无新增GPU执行或kernel性能测量。
> 已先提交旧修改为`9f79155`，本阶段新代码留供复核。613主单测、45选择质量、CTest10/10、14驱动测试通过；438 CLI（408成功+30预期失败）及90主配对完整验收，IR/non-timing stats/trace与预算/目标保持一致。free/weak Attention CLI中位数13.221→1.751s、7.394→1.416s，配对7.559×/5.221×；9成功用例均提升。3小图进程RSS峰增加约2.48–3.07MiB，4原大shape+v2diamond仍为cost-failure/N/A，未删样本或改预算。
> [统一报告](../inference_detail/search_overhead_optimization_report.md)；证据为`guoqiao/optimize_plan/search_overhead_optimization_artifacts/evidence-index.json`。旧24Attention开发观测及失败run不混正式统计；下方原文逐字节保留。

> 2026-10-31 资源预测修复更新：GPU闭环状态：66精确配置的资源、数值和安全验证完成；首次native编译成本仍存在，缓存不会消除首次编译。
> Shared按实际root、对齐、串行max scratch和TMA barrier作编译前结构预测：known42+prospective24全部0B误差；寄存器撤下未验证SSA点模型，PTX仅typed虚拟压力，精确数量由编译器与loaded driver交叉确认66/66。纯静态物理寄存器点预测coverage0/24、MAE=None、not-qualified，不能称预测0误差。136项兼容Python回归、额外backend6/export3、CTest10/10通过；216次sanitizer实跑与12配置精确复用。
> [详细报告](../inference_detail/resource_prediction_repair_report.md)；证据为`guoqiao/optimize_plan/resource_prediction_repair_artifacts/evidence-index.json`。下方保留更新前原文，历史结论不回写。

> 2026-10-31 成本与资源校准、充分知情局部基线已完成实现及有限域实验：v2显式profile接入正常推断，计入算术、route与Copy/转换CTA同步；LI-C纳入producer自身访存和必要输入修复，保留旧L，LI-O对无法局部归属的输入修复整域明确Unsupported。见[实施报告](../inference_detail/cost_resource_calibration_report.md)、[证据索引](cost_resource_calibration_artifacts/evidence-index.json)、[独立复审](cost_resource_calibration_artifacts/whole-review.md)。
>
> 当前源码603主单测、45选择测试、CTest10/10及Python等检查通过；新run共13251条数值记录、200次实际sanitizer进程，训练90pairs、最终120pairs，全部有效。测量后仅静态SASS谓词parser诊断修复，以AST/原始文件完整性桥区分测量版与最终版，不修改原样本或profile。
>
> 研究结论与准入边界：有限候选排序的7训练/3留出可区分对均正确，但LI/J与新旧模型主比较未显示额外kernel收益。3个留出spec的时间预测MAE4.562µs、最大相对误差36.28%；资源预测MAE27.03个寄存器/3906B Shared，不能宣称通用外推可靠或替代后端分配。默认仍legacy，新profile显式可选。此前旧L收益须按基线信息范围解释。
>
> 后续主线为搜索开销优化，再冻结公平比较；冻结之前仍须补强跨形状时延/资源泛化及自然工作负载证据。以下历史内容完整保留，由本条更新当前状态，不将本次有限域实施等同于O05/O06全部长期目标完成。

> 2026-10-30 联合选择及转换消融已验收：同一受检后端、共同候选域的 L-C/J-C/L-O/J-O 四臂完成7行各10个有效配对批次（共70 pairs），完整数值/安全/回归及 fresh whole 独立复审 Approved。见[统一报告](../inference_detail/joint_selection_conversion_ablation_report.md)、[完整证据索引](joint_conversion_ablation_artifacts/evidence-index.json)及[独立复审](joint_conversion_ablation_artifacts/whole-review.md)。
>
> 观察与限制：conflict/diamond 的 C/O 组合比值约1.34；remat 的 L/J 差异属于局部平局消除，弱ABI Attention 的差异含 producer 自身 access 被局部scope排除的效应；强cp.async/TMA同SASS控制未显示可靠布局收益。不宣称严格反转、跨ABI收益或胜过Triton/TileLang，不从2×2单独归因共享/remat/post-opt。精确配对CI、机器身份与开销口径以统一报告为准。
>
> next = 成本/资源校准 → 搜索开销优化 → 冻结公平比较。以下旧内容（含既有roadmap修改）原样保留为阶段历史；其中“联合消融未开始”等旧状态由本条验收更新取代。

> 2026-10-29 P1.4c已验收：同一backend/runner的cp.async与TMA完整Attention各135次通过，基础弱对齐135次回归通过；54次bitexact运输、12组最终cubin四种Sanitizer及CPU回归通过。两层独立复审Approved，root身份/参考核验通过。详见[报告](../inference_detail/p1_4c_implementation_report.md)。

> 2026-10-29 P1.4b已验收：实际Frisk操作/映射/契约受检导出，逐操作生成低层标量TTGIR及PTX发射，复用Triton 3.4.0编译与运行基础；不由Triton重新选择原生Tensor布局。完整Attention135次、运输/cast54次逐次检查、最终cubin四种Sanitizer及CPU回归通过。详见[报告](../inference_detail/p1_4b_implementation_report.md)。

# GPU 执行闭环阶段任务清单

日期：2026-10-29。状态：按用户反馈保留2个独立交付任务；P1.4b已完整验收，P1.4c已完整验收。代码检查点为 `19d56e60aadd4864e902c33e3c04587c87d68dc4`，P1.4a已验收。

任务按完整可执行能力划分。桥接验证、搬运、MMA、Softmax和运行器均为任务内部实施步骤及测试，不再单独立项、发布计划或形成交付报告。

## 整体路线和维护方式

Frisk布局推断/选择/证明/物化 → 受检显式布局TTGIR桥接 → 固定版本Triton后端 → LLVM/PTX/cubin → H100运行核验。

- Frisk保有布局、转换/重计算和指令geometry决策；选择性运行Triton后端pass，核验地址、元素归属和指令契约，记录后端新增通信/scratch及必要执行变化。
- GF(2)共同基础不等于接口完全兼容。桥接必须处理维序/bit宽、硬件拓扑、replication、单位轴，以及Storage映射方向、byte/bit单位、root offset、footprint和alignment。Affine/Product只接入已证明可表达的子集。
- 版本兼容和最小GPU探针在任务一开头完成，结论直接驱动同一任务的后续实现；不能表达的部分记录并作局部扩展或明确拒绝，不静默重选Frisk布局。
- 保持统一的后端入口、映射适配层、运行器和artifact格式。SS/RS、Reduce和Copy复用相同地址/寄存器语义；按代码职责组织模块，不按里程碑复制框架或为每个算子建立独立工具链。
- 每个独立任务只维护一份执行计划、一份实施验收报告及共享证据索引。内部检查点持续更新同一清单，通过切片测试后继续整合，不逐项发起新任务。复审覆盖实际变更，阶段最终做整体复审。
- 代码和证据仅在H100 `/home/baopeihua/frisk`保存，保留已有内容。当前提交是检查点，不自动提交或推送后续修改。
- 不把Unknown当Proven。先preflight支持域，在独立模块中转换后发布；不支持的操作/映射/契约明确拒绝，失败保留输入。

## 任务一：P1.4b 基础完整Attention执行闭环

**目标：** 原始Frisk Attention经过推断、受检桥接和Triton后端，生成可加载cubin，在H100上正确执行。完成时交付一个完整、可复核的执行入口。

**固定目标：** `test/Transforms/attention-layout-fixed-abi.mlir`，单CTA、128线程；Q/K/V为行主序 `[64,16]f16`，O为 `[64,16]f32`，Global根alignment=1。

算法为SS QKᵀ → scale 0.25 → max/sub/exp/sum/div稳定Softmax → P以RNE转f16 → RS PV → Store。保持实际Storage/Distributed map、geometry、descriptor、packing、转换与clone语义。

### 内部实施步骤（不另立任务）

1. **确定接入并建立统一骨架。** 固定Triton及其LLVM、CUDA/PTX/ptxas、GPU UUID和pass集合；先完成最小显式布局load/store探针，检查固定Attention所需的表示能力。优先独立进程/文本IR连接现有Frisk与Triton，避免直接混链LLVM。建立preflight、导出入口、runner和版本化manifest。若需局部后端扩展，在同一执行计划中登记和实现。
2. **连续补齐整条执行链。** 实现Storage/Distributed受检转换，接入Copy/Load/Store/Convert、SS MMA、Reduce/Broadcast/math和RS packing/MMA。以搬运、SS、Softmax、RS切片测试定位问题，通过后立即整合到同一Attention路径，不将切片测试当作阶段交付。
3. **完成真实GPU整链验证和资源记录。** runner加载本次IR生成的artifact，验证数值、安全、弱对齐和重复执行；记录完整编译/GPU耗时及物理资源，完成最终回归、独立整体复审和统一报告。

### 必须落实的实现语义

- 从真实Storage map生成Global/Shared地址，Shared按物理footprint和对齐分配，保留swizzle。唯一writer、读取副本、scratch生命周期与同步均受检。
- Global alignment=1必须兑现为安全的弱对齐操作或byte运输/组装，不借分配器实际对齐加强ABI；测试偏移1/2/3字节及前后保护区。
- SS/RS消费选定atom/grid/repeats、descriptor与accumulator编码，处理初始化、转置、K分片和累加；RS严格核验.f16x2的元素位置、低高半字及寄存器组。
- Reduce消费canonical_fiber_tree_v1、first_owner及输出副本分发；高层tt.reduce不自动视为同一归约树，不匹配时显式展开或补降低。Broadcast读取已证明的同线程源槽位；tensor.empty不产生未初始化读取。
- 冻结exp、舍入、subnormal、fast-math/reassociation口径。WGMMA正确处理Shared对异步代理可见性、fence/commit/wait及warpgroup一致执行。首版串行完成，不做搬运计算重叠。
- TTGIR可包含其合法tensor中间表示；最终降低产物不得遗留Frisk op、未解析tensor、unrealized cast或替代实际输入的固定手写Attention调用。

### 统一验收

- 从原始输入至本次cubin，再到GPU输出的一条命令可重放，保存source/input/selected-layout/contract/TTGIR/PTX/cubin/tool/config哈希、命令和launch ABI。
- 映射使用小域坐标oracle/代数等价检查，覆盖两种不同合法分布、swizzle、复制及单位轴；不支持的映射和错误契约原子拒绝。至少有受支持fixture可运行两种合法方案，供后续消融使用，不强称固定Attention本身必然有两个方案。
- 搬运/转换和孤立packing/cast bit-exact。完整Attention至少seeds0–9、零Q/K、常量、带符号坐标、one-hot V及抵消V。
- 实现前冻结数值门槛。沿用既有提案：输入先舍入f16且有限[-1,1]；独立参考包含FP32 QK、稳定Softmax、P的RNE f16转换、FP32 PV；建议O全部有限且abs(got-ref)<=2e-3+2e-3*abs(ref)。门槛仍为提案，不能观测失败后自行放宽；记录最大误差及坐标。
- 输出sentinel、保护区和重复运行通过；运行memcheck/racecheck/synccheck及适用initcheck/初始化检查，说明工具对异步执行的覆盖边界。
- 新C++单测双配置，GPU切片和整链测试及最终完整CPU回归通过，所有重要复审问题关闭。相同源码的通过检查不无理由重复。
- 保存layout inference、完整编译、GPU耗时原始样本，记录真实寄存器/Shared/spill，与已有逻辑资源诊断区分。功能通过不等同于布局性能获益。

**完成定义：** 现有固定ABI完整Attention从Frisk输入在H100正确、安全、可追溯地运行；弱对齐路径确实通过，完整操作和契约被消费。仅完成桥接或单个切片不能将任务一标为完成。

## 任务二：P1.4c 异步搬运完整闭环

**目标：** 在任务一的同一后端、runner和测试框架上，增加cp.async与TMA完整Attention变体。它们属于同一独立任务，在同一报告中分别记录验收结果。

### 内部实施步骤（不另立任务）

1. **cp.async路径。** 使用显式强对齐独立ABI，兑现已选width、issuer、commit/wait及CTA可见性；保留基础弱对齐版本和拒绝/回退负例。先单次无重叠执行，完成搬运切片后直接接入完整Attention验证。
2. **TMA路径。** 在同一框架补tensor descriptor、host setup、elected issuer、mbarrier初始化/完成字节数/等待及可见性。记录host ABI/setup成本，覆盖对齐/offset/descriptor负例，完成一次性无重叠的完整Attention验证。
3. **统一回归和报告。** 两条路径分别核对实际指令、数值、安全、资源与耗时，并对基础路径回归；整体复审和一个阶段报告关闭任务。

**完成定义：** cp.async和TMA两条路径都在明确的强对齐ABI下通过完整Attention数值与安全验收，且基础弱对齐版本保持有效。只通过其中一条或只通过搬运切片时如实记录进度，不将任务二整体标为完成。

多阶段缓冲复用、循环重叠、warp specialization、cluster multicast和TMA输出Store不纳入本次最小执行闭环。后续确需流水时再规划生命周期、barrier phase、阶段环绕和尾轮收尾验证。

## 与后续路线的衔接

完成基础及所需异步路径后，进入联合选择及转换消融 → 成本/资源校准 → 搜索开销优化 → 冻结公平比较。本阶段不新增成本模型或搜索预算优化任务。

共同后端用于隔离Frisk选择贡献；原生Triton/TileLang完整调优另做端到端比较。单CTA小tile的功能与耗时不推导大尺寸吞吐优势。

## 当前交付状态

- [x] P1.4b 基础完整Attention执行闭环
- [x] P1.4c 异步搬运完整闭环

原P1.4b.0–b.6及P1.4c.1/c.2编号停止作为独立任务使用。其必要验证已合并到以上内部步骤。P1.4b和P1.4c均已完整验收；后续联合选择/转换消融等按新任务推进，不因本次验收自动实施。


> 2026-10-30：P1.4c已提交dd577f6。下一阶段仍作为一个完整任务；[联合选择及转换消融方案](joint_selection_conversion_ablation_proposal.md)已整理，尚未实施。


## 任务1：候选保留与成本决策联合修复（受限源域）

已实现GF(2)通信切分/所有权子空间归档、完整静态摘要与有理排序锥/严格fallback改善；最终显式recipe为 `calibrated-source-grammar-v1`，生产默认仍legacy。Exp开发实测耗时降低9.64%；强制全域GF2在Mixed掉旧优选导致2.527×退化，未隐去。保守源域路由已使Mixed/Div完整恢复旧K4和预算，Exp J/LI保持GF2；Exp两模式最终同镜像，不能声称J胜LI。六池严格选择/镜像桥接和GPU缺项已验收，scalar统一族/Attention校准仍fallback，未作新留出、全域晋升或Triton/TileLang公平比较。完整报告：`/home/baopeihua/frisk/guoqiao/inference_detail/selection_refinement_report.md`。下一阶段为任务2开销收敛。


任务2（2026-11-06）：任务2已复验登记的单 CTA 路径，含 Attention 三种 ABI；两次各 42 个普通验证请求。 见 [全链路开销报告](/home/baopeihua/frisk/guoqiao/inference_detail/pipeline_overhead_report.md)。
