# 搜索开销优化：统一实施与验收计划

日期：2026-11-01（Asia/Shanghai）。服务器 `/home/baopeihua/frisk`。先前修改已按用户要求提交为 `9f79155ac08f82133f4ddb6554af552feefca8d2`，未推送；本阶段作为一次实现和复核交付，不自动追加提交。代码和证据不保留本地副本。

## 起点与目标

380份源码起点及5长期文档原文保存在 `search_overhead_optimization_artifacts`，旧资源/校准/联合消融的封存产物和报告不改。保存旧工具与新构建的提交基线工具及source/config/动态依赖身份。

四Attention的固定CPU、profile开启开发观测显示：free/weak总wall约13.24/7.43秒，solve约11.63/6.09秒，access评分1169/1519次、inclusive约1.37/1.98秒。profile嵌套时间不能相加；这不是正式性能比较。除重复access外，静态partial/full成本反复执行不可变(type,encoding)验证，需要单独计量和复用。

## 实现边界

只减少重复纯计算，不改变候选域、合法性、MRV/tie顺序、预算、beam/refinement策略、模型、转换选择或计费。两种graph-local成功查询缓存：

1. 未加权StorageAccess原始代理：key覆盖约束、执行/存储type、encoding/layout、完整selected Copy binding/null、access kind及root alignment。命中仍正常向observer发送贡献；v2仍逐次应用原profile/frequency。
2. 仅不可变(type,attribute)成功验证；不缓存失败/诊断，不缓存Shared root聚合、全assignment或加权成本。

各缓存最多4096项，满时正常计算；仅finalized graph生效，结构更改与finalize入口先清并失效，失败finalize不得保留旧ready状态。IR/alias合同改变须重建或re-finalize图；类型/attr、Copy路径和alignment不得串。正常pass/pipeline提供 `memoize-costs=false`，仅影响计算复用，不能影响逻辑trace或预算计数。profile新增缓存与实际验证计数，明确entries为累计插入而非峰值内存。

## 实验与守门

由一个CPU driver冻结输入、工具、协议与seed，复用正常 `frisk-opt --frisk-layout-pipeline`，保留每次stdout/stderr、IR/stats/trace、wall/user/system/per-child maxRSS与哈希。主矩阵为四Attention及diamond32x32/remat128x16，扩展mixed128x32和多consumer压力控制；具体输入字节、全部flags/shape及可用性在正式测量前登记，不因结果选择性删行。

主性能：相同CPU亲和、fresh process、相同默认256/32/65536与refinement256、相同legacy模型，1次warmup与10个固定seed随机AB/BA配对，串行执行；profile关闭，stats分阶段时间单列且不相加。warmup只热OS/page cache，不能称跨进程持久cache。3对profile开启归因及final cache-off消融另列，不混入主wall。CPU测量期间无构建、测试或GPU运行；记录系统与工具/source冻结身份。

语义守门：baseline/final/final-off的IR、去timings完整stats、完整logical trace、最终assignment/转换/指令binding/完整cost与logical expanded/MRV/refinement计数一致；profile on/off也一致。选2例显式旧冻结v2 profile确认贡献重放不变。负例覆盖预算耗尽、Unknown、HardConflict、非法输入；缓存容量/disabled、失败诊断、跨图、type/path/alignment与重新finalize均需有测试。公开主测的每行结果、CPU/RSS及未提升/退化，不以更高预算取得收益，不将CPU编译收益等同kernel收益。

## 交付

适当完整C++/CTest、Python driver与相关lit/trace回归；独立源码、协议、数据和最终文档/索引复审；一份中文报告、一份证据索引、可复放源码差异及五文档前缀更新。保持旧 sealed 范围字节不变，测量/最终源身份若不同必须明确映射，不回写旧manifest。最终只提交前置88文件的已授权commit；本阶段新修改留供复核，除非用户另行要求提交。

## 正式计时前的支持域修订

原run `20261101T003017Z` 在24个Attention语义观测后遇到diamond baseline cost-failure，25 CLI/0formal timing，原记录保留。独立定位该diamond自然Eall131072 carriers超过原访存枚举65536支持上限。30开发预检确认原4大shape+v2大diamond三变体均同失败，固定≤512小shape三变体成功且完整语义一致。按支持域而非速度修订：原5case保留untimed失败守门，新9timed（4Attention+4小shape机制图+算术控制）+2成功v2guard，合计16case/96正确性CLI。90主pairs/198 timing CLI及72profile/72ablation保持，合计438接受CLI。新harness14fake tests通过，fresh登记与工具/源freeze在任何正式性能样本前重做；不修改native code/domain/model/budget，不混用旧24观测。见stage `amendment.json`。

## 接受运行与数据验收

接受RUN为`20261101T005848Z`；protocol SHA `fcbeb84cc6ea2cce4cc3503916b03237770513b728c30bb368da2d4e52e94e29`、registration SHA `b82e9337e7c4a7cf5b62870d7e2bb7ab2ecec1f231126f1539df3d3fb1eb0bd9`。438 CLI全部按预登记完成：408成功、30预期失败，90主pairs/27profile/27ablation。9主用例wall全部提升，free/weak配对7.5587×/5.2207×；3小图RSS峰增加2.48–3.07MiB，5失败行N/A不进汇总。独立数据重算/2886产物/376源/5动态库审查PASS，旧16223封存文件不变。正式后仅补joint consumer重链接、未改FriskOpt字节或source。最终源码包、5文档原文后缀已独立复放通过；完整交付范围与最终批准以whole-review和总证据索引为准。
