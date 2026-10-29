# P1.2 选择质量方案 A 执行计划

日期：2026-10-26。用户已确认[具体契约](p1_2_selection_quality_design.md)。先 Q1/Q2，再 P1.3a→b→c。

**Goal:** 可复核地区分策略候选覆盖、消费者影响和搜索误差，不预设 Auto 更优或 geometry 反转。

**Architecture:** 独立 `benchmark/layout/selection` C++ 支持库构造真实 Frisk 图，执行原生流水线及冻结域受控求解；独立 DFS oracle 校验搜索。Python 管理有限 manifest、产物、串行 CLI 计时和报告。生产模型、预算不变。

**Tech Stack:** C++17/MLIR/LLVM JSON、Frisk 分析与流水线、GoogleTest、Python3 unittest、CMake/Ninja。

**Global Constraints:** 直接源目录 main，保留已有修改；不 worktree/stage/commit/push/GPU/P1.4。基线 `/tmp/frisk-selection-baseline-Z6CUX3/tree` 是当前脏源码快照，同目录有原 Release 工具/配置；不能用 HEAD 代替。每任务 RED/GREEN、独立复审；仅一个实现/构建拥有者。禁止为实验结果修改生产权重/预算；正确性错误不能降为普通不支持行。

基线 SHA-256：`source.tar=ea98f040a5ec7b84684fb8a13d53e875c7bb9a220990984ed1ea8126bb423c15`；Release `frisk-opt=3c1f549298693d549ebad304b59bf43b0a830be631052d5c096b82524607bf8d`；Release `CMakeCache.txt=e4acb24cf609aebc914141e60915a236d1aed20effb8aada0d7b6785952aaffb`。归档哈希标识本次快照，不替代后续按文件内容生成的源码身份。

## Task 1: 输入矩阵与原生流水线实验

**Files:** 新建 `benchmark/layout/selection/{CMakeLists.txt,SelectionCase.h,SelectionCase.cpp,SelectionExperiment.h,SelectionExperiment.cpp,layout_selection_bench.cpp}`、`unittests/Dialect/Frisk/Layout/LayoutSelectionQualityTest.cpp`；修改 `benchmark/layout/CMakeLists.txt`。不改生产 include/lib。交付可独立运行的 native 实验。

- [x] RED：64 行 manifest 无重漏、C0–C7/K/线程/SS/RS/四 policy；真实 C1 SS/RS Store 与两轴 free/fixed consumer；外部 ABI；无效 case/flag/grid 拒绝。记录实现前失败。
- [x] 支持库命名空间 `mlir::frisk::selection`；`CaseSpec` 包含 case id、M/N/K/threads、SS/RS、policy、consumer、可选 consumer grid；consumer 为 Store/Reduce0/Reduce1/Fixed0/Fixed1/Local。`buildFixture(MLIRContext&,const CaseSpec&,std::string &error)` 返回 owning ModuleOp；`nativeCases()` 返回64行。具体 C++ 类型可按库习惯调整，跨任务头文件保持稳定。
- [x] C0=(64,64,16,128)、C1=(128,128,16,256)、C2=(128,512,16,256)、C3=(256,128,16,256)、C4=(128,128,32,256)、C5=(256,256,16,512)、C6=(512,128,16,1024)、C7=(128,128,16,512)。SS Shared roots align1024，RS A Tensor，f16输入/f32 zero init，无转置；Global输出f32行主序align128。无Global→Shared输入Copy。fixed consumer 按指定合法geometry生成canonical result fragment并插入显式ConvertLayout，不固定MMA输出。参考 `MmaAutoIntegrationTest.cpp`，不能硬编码K16。
- [x] CLI `--case C1 --mode ss --policy Auto --consumer store --phase native --output-dir <new-dir>`；consumer接收store/reduce0/reduce1/fixed0/fixed1/local，fixed必须`--consumer-grid 1,2`；`--list-cases`输出JSON；`--phase emit-input`仅发布输入。未知/不相容参数拒绝；输出目录必须新建，不覆盖。Task2增加controlled/local phase。
- [x] native调用真实`buildFriskLayoutPipeline`采集stats/trace；成功后独立actual IR verifier、逐坐标输出字节地址核验、print/parse后再次完整pipeline并确认稳定。local不走DCE流水线，标local-analysis-only。失败保留input/诊断/typed failure，内部证明/成本错误fatal。
- [x] 已查明入口边界：fixed consumer 的原始 ConvertLayout 源未编码，生产验证器拒绝（`FriskLayoutOps.cpp:248`、`NormalizeLayoutIR.cpp:21`）。Task1保留此analysis fixture和结构测试，native显式报unsupported-input-form、不做隐藏预绑定；Task2负责低层求解/物化后完整pipeline重验。64个Store native样本不受影响。上面的fixed“输入验证”指目标编码和fixture结构，不虚称整个未物化IR已通过验证。
- [x] 每行JSON保存schema/version、规格、状态、命令、source/config/binary/input/output hashes；保存input/output/replay IR、完整stats/trace。source-manifest沿用solver benchmark，覆盖支持库/头文件/生产源码/构建配置。绑定、geometry、成本贡献、转换实际计数从真实trace/IR提取，不捏造缺失数据。Task2补frozen domain/oracle。
- [x] 注册独立`FriskLayoutSelectionTests`链接支持库，不把测试API放进生产库。Debug/Release构建并跑新测试、C1 SS/RS落盘样例：`cmake --build build-m6-release --target frisk_layout_selection_bench FriskLayoutSelectionTests --parallel 12`，Debug使用`build`；记录实际exe路径、RED/GREEN日志。
- [x] 独立spec/quality review，修完Critical/Important再交接；不自动提交。

## Task 2: 同域 geometry 限制与独立有界 oracle

**Files:** 新建 `benchmark/layout/selection/{SelectionControlled.cpp,SelectionOracle.h,SelectionOracle.cpp}`；扩展Task1支持库/CLI/测试/CMake。不改生产include/lib。

- [x] RED：真实C1 Auto两geometry；joint/单geometry只缩geometry和相关tuples，不改其他初始候选；完整/节点截断/赋值截断/超时；target evaluate+conversion各计一次；fixed consumer计费；局部tie集合保留。
- [x] CLI `--phase controlled [--restrict-grid 1,2] [--oracle]`、`--phase local --oracle`。controlled一律Auto。冻结candidate preparation后/传播前完整图和支持表到`domains.json`并hash；各分支从同一未改图出发，限制后检查invariants再传播求解。不增加geometry-only生产属性、不用完整mma_contract代替。不从已物化IR生成另一分支，明确图/IR生命周期和clone映射。
- [x] 默认解使用原预算。oracle稳定变量序DFS，完整赋值检查所有hard relations，评分target evaluate+chooseLayoutConversions各一次；不复用生产排序/成本剪枝。保存全部最优geometry集合，说明九项成本tie定义。上限65536递归节点/4096完整合法评分/120秒，任一截断标incomplete；边界恰好穷尽正确处理，只有确实遍历完可称complete/infeasible。测试允许缩小、不允许扩大上限。
- [x] hard relation 的 bool 接口不能区分 Unknown/Disproven；有未证明关系时明确 oracle proof-incomplete，不能把 Unknown 当成证明无解。已完整证明的 instruction/reduction support table 可用于缺失支持判定；unary/pair proof 保留状态。实际受支持小图无 Unknown 也要有针对负例测试。
- [x] 保存assignment、9项成本/贡献、conversion/remat、proof/search stats、oracle visited/evaluated/status、完整冻结域身份。成功完整解物化、独立验证、ABI/完整pipeline重放；local只分析。native需松弛而controlled不能复现最终域时not-comparable，不混attempt。
- [x] fixed consumer 原始图标pre-layout-analysis-fixture，低层collector允许分析；先物化再做generic/actualIR验证和完整pipeline重放。保存原始及物化后流水线输入；不能把这一结果标为原始native CLI推断。无需修改ConvertLayout生产验证器。
- [x] 默认成本低于完整oracle/非法解fatal，高于oracle报告gap而不替换默认结果。local/full成本不直接相减，只将local几何放入同一完整图重新优化。
- [x] Debug/Release新增controlled/oracle定向测试和CLI样例，Release全部selection tests；最终Task4再跑完整Debug新测试。Task1的7项Debug实测549.5秒，按未改变/受影响路径复用阶段证据，避免每个任务无差别重复慢native矩阵。正式Release实验确认C1 SS/RS direct及C1 SS两轴free/全部fixed核心oracle完整。核心incomplete阻塞验收，不暗改预算。
- [x] 独立review关闭重要问题。

## Task 3: 矩阵运行、产物校验与报告工具

**Files:** 新建 `benchmark/layout/selection/run_selection_quality.py`、`test/python/layout_selection_quality_test.py`；必要时拆`selection_report.py`；CMake注册测试。

- [x] RED：64 native；Q2 C1/C2/C3×SS/RS×两轴free/每合法grid fixed最多36行；先生成consumer编码manifest；失败/超时不漏行；JSON/hash/贡献核验；严格反转/tie消除/samegeometry/无改善/heuristic-only分类。
- [x] 各行独立进程保留命令、退出码、超时、日志；native+controlled joint/eachgrid+local。不得缩shape或漏失败行。实现选择拒绝已有目录，不提供 resume。
- [x] 使用真实CLI约定：成功exit0、已分类非致命失败exit1、fatal/参数错误exit2；JSON和退出码矛盾必须报基础设施错误。fixed消费者的native入口拒绝单独保留，主要结论来自controlled；不要求无效原始Convert输入直接通过CLI。Python驱动/报告脚本自身哈希也纳入复现身份，不能只记录C++二进制。
- [x] ordinary native/controlled 用 `native-snapshot-projection.json` 的实际最终成功attempt图/支持表/传播前域身份做匹配，同时核对输入、源码、配置、预算；controlled 的 pending-native-comparison 只有匹配后才能成为same-domain。fixed/local为入口不同，不冒充native同域；冻结域内部joint/geometry分支另用自己的域身份验证。不要用丰富proof诊断文件的哈希直接代替native可比投影。
- [x] 分类遵设计§5.2；完整local/fulloracle且最优geometry集合不相交才严格反转；输出全向量及首个差异字段，不虚构加权总分百分比。不同域/预算截断单列；same-geometry 只比较实际选中同一 grid，成本打平的绑定变化不声称收益。
- [x] 实现独立串行timing调用Release `frisk-opt`，C1/C2 SS×四policy×(1warmup+5measure)=48次；wholeCLI wall/RSS含stats，不开trace/profile/oracle，不与构建/测试重叠；重复IR/计划/非计时stats稳定。正式48次运行由Task4验收，不以工具测试代替。
- [x] 非计时正确性矩阵可用有界`--jobs`独立进程并行，记录并发数并按manifest稳定排序输出；每行仍保留独立oracle预算和超时。计时模式必须等待其他任务全部结束并强制串行，不能将并行正确性阶段的wall time当性能排名。
- [x] 输出全部原始数据/hash JSON和Markdown。Python分类小表测试外，还用真实exe/临时目录跑集成；Release CTest执行25项Python测试通过。重载严格核验完整身份、manifest、raw状态和分支不变量；正式门槛为26完整oracle＋2个native同域见证。
- [x] 独立review关闭3项重要问题；源代码批准不等于正式实验已验收。

## Task 4: 完整实验、验收与后续 P1.3

**Files:** `guoqiao/inference_detail/p1_2_selection_quality_report.md`、`guoqiao/optimize_plan/p1_2_selection_quality_results.json`及可复现原始产物目录；同步本计划/设计/长期实现计划/体系对比。root负责。

- [x] 源码冻结后Debug/Release全部selection各20项、Python25项及相关MMA/pipeline旧24项通过；复用Task1/2已构建且源码未变的二进制，Task3仅Python/CMake注册；历史505/69不冒充本轮新验证。
- [x] 完整64 native/36消费者及其他受限分支，共230行；核心26完整oracle和2个native身份见证通过。165完整oracle无默认gap；C6 RS joint oracle超时单列，预算未增大。
- [x] 关闭其他测试/构建后48次串行计时，全部成功且IR/计划/stats稳定；持久化源码、raw JSON/IR/trace、测试日志。报告含C1逐项成本、固定转换及C2严格反转算例；明确Auto编译开销，无GPU/跨框架结论。
- [x] 独立最终source/evidence/docs review通过，无未关闭Critical/Important；维护文档同步，3项非阻塞分析/诊断边界在报告中明确保留。
- [ ] 报告验收后细化既有P1.3a接口/测试并另写执行计划，实施/实验/复审；随后P1.3b、P1.3c。未完成不声称整体完成。

## 自审

覆盖设计§3–8；native/controlled分离，local不走DCE，fixed consumer不固定MMA；评分和oracle搜索分离；预算不变。Task1–3与Task4的Q1/Q2实验、回归、最终复审全部完成；继续独立细化/实施P1.3a→b→c，不能将Q1/Q2完成当作P1.3完成。
