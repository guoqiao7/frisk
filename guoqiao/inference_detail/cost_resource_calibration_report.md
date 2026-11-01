# 成本与资源校准及充分知情局部基线实施报告

日期：2026-10-31。状态：实现与有限域执行、选择及实验闭环已完成；留出验证表明通用时延/资源外推精度不足，默认仍为legacy。最终独立复审与证据闭合见本文末和证据索引。所有实现和证据仅保存在 H100 `/home/baopeihua/frisk`，未暂存、提交或推送。

## 实施范围与版本

本阶段沿用同一 Frisk 受检导出、标量映射适配和固定 Triton 低层编译后端，不由 Triton 重新选择 Tensor 布局。工作区基于 `dd577f6c1e0c87fbe5b5bf204fd250c00a613ce1` 上一阶段已验收但未提交的修改继续实现。阶段前像、当前365份源码清单、40文件独立回放通过的补丁见[证据目录](../optimize_plan/cost_resource_calibration_artifacts/)；其中新增7源、修改33源，无删除，不混入上一阶段修改。

- [执行计划](../optimize_plan/cost_resource_calibration_execution_plan.md)
- [独立整体审查](../optimize_plan/cost_resource_calibration_artifacts/whole-review.md)
- [阶段补丁](../optimize_plan/cost_resource_calibration_artifacts/stage-implementation.patch)
- [源码封存审计](../optimize_plan/cost_resource_calibration_artifacts/stage-source-package-audit.json)

## 可用于正常推断的 v2 模型

保留默认 legacy 行为；显式 `cost-profile` 可用于正常 `frisk-infer-layouts`、`frisk-layout-pipeline` 及实验驱动。profile严格验证已知schema、完整15项非负整数系数、重复字段、输入大小和嵌套界限；完整规范化SHA256及频率模式绑定准备包、trace和最终记录，模型ID含参数摘要与leg/stc后缀。失败保持原始IR，不把非法配置回退成免费成本。

v2将第一成本分量改为非负整数selection ticks，其余legacy分量保留作tie和诊断。tick是训练得到的有限域排序分数，不能称为硬件周期或指令数；拟合截距只用于误差分析，不进入方案排序。全图SSA峰值非可加，本阶段不将其直接替换进仍按硬component求解的可加目标；v2的partial lower bound保守取零。

| 特征组 | 明确计数 |
|---|---|
| 访存与指令路径 | access_warp_issues、32B Global sector代理、Shared额外bank冲突代理、Copy setup单位、MMA指令工作代理、Reduce元素工作 |
| 普通算术 | ALU、div、exp、cast的每warp标量issue代理，包含实际分布的复制载体执行 |
| 布局修复 | Register／Shuffle／Shared／Conservative分别计转换字节代理；转换、重计算及其叶片转换用同一目标比较 |
| CTA同步 | 受支持Global→Shared Copy的vector/cp.async各1次、TMA各2次；Shared转换2次；按同一频率口径加权 |

最终被选择的方案携带可复算feature账本，未采用的转换/clone方案不污染累计。同步feature不包含mbarrier轮询、异步wait、proxy fence等所有同步。常量折叠、缓存、依赖延迟、重叠、完整MMA Shared读取bank与标量控制等未建模部分明确保留；未知受支持域外Tensor算术不能默认为免费。

v2后优化保留语义canonicalization，暂不运行尚未证明与新目标一致的旧commute/hoist。本阶段主性能矩阵固定C，隔离此差异；O只作训练和机制/功能覆盖，不从O结果单独归因参数权重收益。默认legacy O保持既有行为。

## 充分知情的局部基线 LI

保留原受限L，增加LI与J。LI第一阶段保留完整硬合法域，按真实操作归属计入producer自身的算术、指令、访存、路径、直接资源及必要输入转换；其本地input/Storage辅助选择可以变化。未来consumer的边界继续排除。第二阶段仅固定producer输出encoding、geometry/path，再用全图目标补全；记录补全后的local rescore，不声称所有自身辅助决定被固定。预算、稳定producer tuple tie与独立小域oracle沿用统一接口。

Shared资源若聚合本地与未来访问且无法分离，明确拒绝。LI-O若原准备域存在潜在非identity自身输入修复，在传播和强制候选之前整域返回Unsupported；direct solver、oracle和评分接口一致守门，不按assignment失败悄悄缩小候选域。主C矩阵不受此O覆盖边界影响。

R3真实remat图复核：旧L选A，LI与J选B。A的自身输入约束2/3均纳入，未来7/8/10排除；legacy局部转换代理A=18432、B=0，unit-v2局部分数A=16420、B=32。LI/J全图成本相同。此项说明旧受限局部基线存在信息盲点，不能把消除自身输入修复的收益当成相对LI的联合选择优势。精确scope、assignment、oracle和SHA见[r3-cpu-smoke](../optimize_plan/cost_resource_calibration_artifacts/r3-cpu-smoke/summary.json)。

## 预登记、训练与留出

最终接受数据目录为 `cost_resource_calibration_artifacts/run-calibration-r3-20261031T105856Z`。协议、feature semantics和unit-profile与首次登记逐字节一致；R3修复发生在任何正式样本之前。9个训练spec：64×16 no-conflict/conflict/remat，32×16 mul/div/exp链，固定64×16 Attention的弱vector、强cp.async、强TMA三个ABI层。3个留出spec：32×32 diamond、128×16 remat、64×32混合exp/div/mul/sub。

所有同spec的候选属于同一split。强制A/B覆盖受控图合法候选；Attention使用自然候选，有限模型oracle最优不等于GPU全域最优。最终四臂为legacy-LI、legacy-J、calibrated-LI、calibrated-J，同batch加入实际候选参考，弱Attention额外保留旧L诊断。模型比较始终在相同算法、ABI、精度和后端内；训练跨路径覆盖不构成跨ABI性能收益结论。

训练固定非负ridge/active-set NNLS及预登记超参，采用训练列最大值缩放；ticks_per_microsecond=1000000。留出不用于拟合、选择超参或更改实验设计。fit/assessment均核验本run、phase、profile、protocol、实际source/tool、manifest及完整测量file-set身份。

训练结果：9组各10个有效配对批次，共90 pairs、42个arm观察，无foreign/invalid、无补样本。NNLS收敛，KKT violation约1.99e-13，design rank=11/13，训练时间RMSE=0.382814µs；分析截距=0.513962µs。Register/Shuffle/Conservative转换未激活，其系数1是显式fallback；共线项及拟合为0的活跃项不能解释为真实操作免费，也不能声称15参数都已独立识别。

冻结profile：`run-calibration-r3-20261031T105856Z/fit/profile.json`，原文件SHA256 `6b4947af82c2748fc8db75268ddb621fc45852d238b63deee12cf17c6cde7388`；训练ledger SHA256 `cf23be2de94cc6493620d9e92c8380e086f1cb6a24818b9751b18bf9fb99aa53`。独立训练审计复算90pairs、81组配对ratio/CI及全部身份、特征求和与profile dot-score通过。

## 实际资源与预测

在物化及最终IR上复用SSA存活诊断。对实际loaded CUfunction查询registers、静态Shared、每线程local bytes和CUDA driver理论residency；理论residency不是单CTA执行时的实测occupancy，local bytes也不是spill计数。实际Shared必须匹配emitter分配，scratch及TMA barrier分别保存；保留最终cubin、LLIR、PTX、SASS及机器资源原始报告，静态opcode计数不冒充动态执行次数。

资源经验映射只从训练集拟合，最终验证使用同一冻结模型。训练MAE：寄存器2.139个/线程，Shared 971.7B；最大误差分别6.747个和2645.3B。SSA根峰值缺少转换scratch等后端分项，残差必须保留；emitter分项用于解释和driver精确核对，不偷塞进SSA预测器当作预测成功。资源映射本阶段作为校准诊断，未宣称替代后端分配或资源对性能的非线性影响。

## 最终留出与实际选择结果

最终12组各10个有效配对批次，共120pairs、76个arm观察；无foreign/invalid、未补样本。全部205组配对ratio及逐比较95%bootstrap CI由独立脚本重新计算通过。以下是C主矩阵GPU时间中位数，单位µs；ratio使用逐batch配对中位数，不能由表中总中位数相除代替。

| spec | legacy LI | legacy J | v2 LI | v2 J |
|---|---:|---:|---:|---:|
| train-no-conflict-64x16 | 5.055 | 5.060 | 5.056 | 5.058 |
| train-conflict-64x16 | 6.864 | 6.860 | 6.866 | 6.866 |
| train-remat-64x16 | 7.411 | 7.416 | 7.407 | 7.409 |
| train-mul-32x16 | 1.708 | 1.704 | 1.703 | 1.708 |
| train-div-32x16 | 2.528 | 2.529 | 2.524 | 2.528 |
| train-exp-32x16 | 1.737 | 1.733 | 1.733 | 1.736 |
| train-attention-weak-vector | 6.380 | 6.427 | 6.418 | 6.387 |
| train-attention-strong-cpasync | 6.332 | 6.360 | 6.291 | 6.346 |
| train-attention-strong-tma | 6.507 | 6.518 | 6.483 | 6.489 |
| holdout-diamond-32x32 | 6.896 | 6.891 | 6.885 | 6.890 |
| holdout-remat-128x16 | 21.085 | 21.094 | 21.099 | 21.088 |
| holdout-mixed-64x32 | 15.486 | 15.506 | 15.493 | 15.485 |

LI/J、legacy-J/calibrated-J及legacy-LI/calibrated-LI的全部12行主CI均包含1；本次没有确认联合选择相对充分知情LI或新模型相对legacy的额外kernel收益。弱Attention旧L/LI配对ratio=1.056369，95%CI[1.052721,1.058815]，说明旧局部scope遗漏自身访问的影响。remat的同batch候选A比B慢，而LI与J均已选择B；这应解释为补齐局部信息消除了旧盲点。

独立候选排序审计按完整assignment合并同batch重复arm，机器码相同或CI含1按tie处理：训练7个、留出3个可区分候选对全部排序正确。此为有限已测候选集的探索性排序结果，不是10个独立完整问题或GPU全域最优证明，也未显示比legacy更优。详见 `root-ranking-audit.json`。

**外推精度不通过。** 3个留出spec的时间误差由12个可定价arm观察统计：MAE=4.562242µs，最大相对误差36.27996%。这些重复arm不是12个独立问题。对应v2-J的实际值与预测如下：

| 留出spec | 实测µs | 预测µs | 相对误差 |
|---|---:|---:|---:|
| holdout-diamond-32x32 | 6.890 | 6.670 | -3.19% |
| holdout-remat-128x16 | 21.088 | 13.837 | -34.38% |
| holdout-mixed-64x32 | 15.485 | 10.376 | -33.00% |

留出资源误差基于同3spec的18个arm观察：寄存器MAE=27.0316个/线程、最大106.0544个；Shared MAE=3906.37B、最大6741.33B。mixed64×32强制B候选实测168个寄存器、预测61.95；remat128×16选中方案实测96个寄存器/8192B Shared，预测约56.15个/1450.67B。Shared根代理缺少转换scratch，不能据这些回归承诺物理资源上界。

所有103份实际产物的driver Shared分配与emitter静态分项精确匹配；51份具有v2特征的C产物，其已建模CTA barrier数与独立SASS BAR静态计数逐一相等。该结构核验不同于SSA经验回归的泛化成功。寄存器较少也不必更快，例如remat128×16的B使用96个寄存器，仍比80个寄存器的A更快；转换代价与寄存器压力需共同评估。

结论：新模型和充分知情基线可用、有限候选排序与实际执行闭环通过；训练拟合不能外推为可靠通用时延/资源预测。本阶段不把该profile切为默认，不据此宣称性能优势，冻结公平比较之前仍须补强资源结构和跨形状预测质量。

## 验证、证据与历史边界

最终R3源码全Release主单测603/70 suites通过（286.567s），完整selection45/3 suites通过（180.131s），CTest10/10通过；另有成本/trace27、LI10及Python校准27等定向检查，存在覆盖交叠，不相加为独立测试总数。既有执行器Python51、backend6、export3通过。source delta完整40文件patch独立回放SHA逐一通过。

初始候选54/54均通过新run精确cubin数值安全验收：7542条数值记录，148次实际sanitizer执行，另17份仅按raw cubin字节、完整ABI/config相同复用安全证据。强Attention包含输入0/128/256与独立弱输出1/2/3偏移。最终49份重选产物也全部通过：5709条数值记录、52次实际sanitizer进程、36份精确复用。阶段合计13251条数值记录、200次实际安全检查进程，不将复用重复计算为新进程。

历史失败保留：R1测量到fit/assessment身份绑定、R2 trace 4096字节envelope与feature ledger、R3 producer自身输入修复计费均经独立复核修正。首次run在发现R3后、修复前停止，17行历史验证不用于本次acceptance、正式样本为0。控制脚本一度误指历史目录，被工具身份检查阻止进入GPU；该操作覆盖了旧首份pre-validation状态文件，原文件不可恢复，已如实记录，不声称历史失败文件逐字完整。最终新run全部独立验收，不使用旧17行安全记录。

后续路线仍为：搜索开销优化 → 冻结版本开展Triton/TileLang公平比较。当前不作跨框架优越性、任意形状/控制流或大尺寸多CTA吞吐结论。


## 测量后静态诊断修复与两种源码身份

独立SASS审计发现原parse_sass漏识别PT/UPT谓词，10份cp.async产物原静态计数1689，校正为1704，补计15条`@!PT LDS RZ,[RZ]`。它们是never predicate的静态指令，不是15次动态Shared读。仅修复parser正则与唯一新增测试，27项Python校准测试通过；runtime删除parse_sass后old/new AST完全相同，test删除新增方法后AST相同。

所有原始测量、machine-evidence、profile和fit保持不变；103份校正sidecar位于`run-calibration-r3-20261031T105856Z/audits/parse-sass-predicate-fix/`，逐一签原SASS/cubin与两个parser身份。2706个原artifact/测量/fit文件及完整file-set独立复核0变化。原测量源码摘要`32601fa0e2dfa0fa86f2adb4157abb249d60fa9eb333cb7665c5bc21786ad199`与最终当前源码不同，历史guard保持failclosed，不能冒称当前源码直接通过旧freeze；测量版本源码、最终源码与受限桥均封存。该修复不参与C++选择、模型feature/fit、实际资源查询或GPU计时，不重拟合或重采样。实际测量bench二进制另行封存；最终重链接工具的`.text`与其逐字节一致，`.rodata`差异仅单个64字符十六进制source-manifest摘要字符串，另有ELF build-id变化，见`root-postdiag-native-sections.json`。

## 最终交付与审查

证据索引为 `guoqiao/optimize_plan/cost_resource_calibration_artifacts/evidence-index.json`，索引在最终review后生成/校验，不自签自身。源码、原测量版本、最终版本、AST桥、source delta回放、实际工具身份、raw样本、失败历史与独立审计统一收录。最终独立Assessment以`whole-review.md`末尾为准；“实现与证据验收”不等于“通用外推精度验收”。
