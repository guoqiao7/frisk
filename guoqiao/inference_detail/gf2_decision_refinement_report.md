# Frisk 决策修复、LOCAL 校准与搜索编译开销验收报告

日期：2026-11-04。代码和证据仅在 H100_direct 的 `/home/baopeihua/frisk`，没有保留本地代码副本。

**本轮三项修改已完成，并通过当前声明支持域的内部验收。** 修复后的主策略在六个新留出程序上，相对 Frisk legacy 的时间比为 0.660491，即约 **1.514×**；相对相同65536 probes／256 refinement／K4预算的充分知情局部基线，约 **3.884×**。有限归档加legacy联合池的平均遗憾约0.310%、最坏约1.668%，全部15项回归守卫通过。独立与正式统计共6000项数值检查，差异为0。

完整六程序的CPU开销对照中，生成加冷编译1001.494s→420.173s（2.384×），热编译428.208s→93.087s（4.600×）。这是单次描述性结果，保留监测覆盖限制；并不表示所有大形状都已低开销，最大mixed候选的一次完整编译仍约605s。

综合记录 `release-qualification.json` 的状态为 **QUALIFIED_FOR_DECLARED_DOMAIN_WITH_LIMITATIONS**。它合并独立GPU与CPU验收；正式GPU报告中保留的“待单独效率验收／promotion=false”是中间阶段字段，本综合记录不覆盖它。**生产默认选择策略仍为legacy，代码未提交。** 这满足本轮内部门槛，可进入当前支持域的冻结对比准备；尚未进行与Triton、TileLang的同算法、ABI、精度和调优预算比较，不声称已优于它们。

下文 `development/`、`run-gf2-decision-refinement-r1-20261104/` 及 `release-qualification.json` 相对证据根 `/home/baopeihua/frisk/guoqiao/optimize_plan/gf2_decision_refinement_artifacts`；Python实现位于 `/home/baopeihua/frisk/benchmark/layout/execution`。

| 验收内容 | 完整结果 |
|---|---|
| 源码测试 | 680项C++单测、53项选择测试、14组CTest、80项IR测试均通过 |
| GPU功能 | 15规格／129候选全部通过；17685条正常记录含12879 ordinary、2403非默认流、2403三节点capture |
| GPU安全 | 96独立协议组×4工具＝384次直接Sanitizer；33行严格exact-safe复用；1296条安全记录 |
| 编译与证据 | 129请求＝96 API＋33 exact hits；80种原始CUBIN；114候选证据；195个策略选择 |
| 计时 | 150配对批次／96执行配置／960图像块／4800正式event；非法、替换批次和边界外部PID观测均为0 |
| 历史保全 | 四套旧封存41351文件／32928590880字节逐文件与文件集校验通过，另核native及partial外部报告 |

## 已定位并修复的问题

上一轮完整实验已封存于 `partial_cost_efficiency_artifacts`，状态为 IMPLEMENTED_RETESTED_NOT_QUALIFIED。功能和开销检查通过，但新选择的整体 GPU 性能未通过。尤其 conflict 的多个候选具有完全相同的原生 14 维特征，原生分数无法区分其访存分布；mixed 的控制流使直线原生计分失效，原静态后备在主成本相同的候选中选择了 LOCAL 占用更大的实现。这是信息缺失与后备决策问题，单纯重新拟合相同 14 个系数无法解决。旧失败、被中断的尝试及更正说明均保留，不把已观察的 R3 当成新留出集。

### 完整池上的决策与不确定性

1. 保留原静态见证和 interval-only 部分成本选择，作为同池消融，均使用冻结的 14 维模型。严格原生策略只有上一轮已封存的历史结果，本轮 13 个策略标签没有重新运行 strict。原生精确分数相同时，只有全部必要证据齐全，才使用 GF(2) 推导的 emitted-demand global-sector burden 排序。它描述 ABI 允许相位下按 query 取最大值的需求负担，不是 Ncu 实测动态事务量。
2. 新主策略 `joint-affine-frame` 对每个候选使用成本完成族 `c_i = b_i + sum(a_ik*u_k) + beta*s_i`。已校准部分为 b；合法的未覆盖非负费用与 GF(2) 扇区负担使用共享、非负但未拟合的系数。某候选的固定贡献和每一未知系数的计数都不大于整个池的所有竞争者时，它是该完成族的 argmin 成员。共同的未知项可消去；出现权衡则回退，不猜一个参数，不从新留出时间拟合 beta。
3. 普通 CFG 不支持、工具或证据契约不成立时，不制造数值下界。回退仅在静态主成本最小的一组中，以 driver 确认的 LOCAL footprint 辅助决策，再使用原剩余 tuple 和见证规则。这里保证主成本最小，并不声称保持原完整九项字典序结果。任一必要占用证据缺失时，对整组跳过该细化；未知不是零。
4. 证书绑定完整候选池、源代码、profile、原生分析、GF(2) ledger、编译 image 与资源证明，可纯函数重新计算。结论是指定模型族内的条件性最优成员，不是 GPU 时间下界或无限搜索空间的最优证明。

对应实现：`decision_priority.py`、`partial_cost_selection.py`、`partial_cost_qualification.py`。独立数学审查含 1200 组矩阵、28800 组系数完成、3882 个反例、172 组不支持回退，以及各 400 组精确成本／静态主成本细化检查。

### LOCAL 校准及复用边界

真实 LOCAL 校准来自本任务已完成且已封存的 `partial_cost_efficiency_artifacts/local-fit-r1`，仅修改 `local_transfer_bytes` 的系数到 6418 ticks；其他 13 个系数保持原值。本轮继续冻结 14 个系数，不借新性能标签重新拟合。 本轮 13 种规则消融都使用同一份已校准模型，旧模型只绑定校准沿袭，不重新运行旧／新 LOCAL 系数消融。上一轮 R3 中两模型选中相同候选，尚未确认 LOCAL 系数本身的额外 GPU 收益；不能把本轮决策改进或帧大小后备的收益归因于该系数。训练覆盖 94 个 group-images，其中 22 个新 LOCAL 微基准，66 项常规与 264 项 Sanitizer 检查、1100 个 event 样本，配套寄存器与屏障对照。正覆盖不等于通用外推精度已经通过。

6418 是模型计分单位，不是一个字节的硬件访问延迟。编译器 LOCAL 分配帧、静态 LDL/STL 字节、动态访问量及 spill 数量是不同概念；帧大小只在保守后备中使用，不能替代已校准的传输项。匹配协议的增量时间区间也不能冒充拟合系数的置信区间。

稀疏 Shuffle 与地址表 memo 改动会改变实现身份。`compatible_native_profiles.py` 通过严格 AST／内容证明限定允许的差异，仅重绑 Shared bank 分析实现哈希：Shared 分析语义、训练输入、特征、系数及其他契约保持不变。它不重新训练，也不放宽 Shared 语义。活动视图位于 `development/native-profile-compatibility-offsets-r2`，依赖链注册到正式 protocol。

## 搜索与编译开销优化

### 一份受同预算约束的候选归档

新路径用一次 v6 准备和共享的 65536 probes／256 refinement 预算，替代旧六来源各自有预算的重复求解；CPU 两流程的搜索组织与总工作量并不相等。GPU 同预算选择质量对照则要求 J 与充分知情 LOCAL 各有 65536 probes／256 refinement／K4。J 先分配 16384 probes 给充分知情的局部见证，然后使用总 65536 probes／256 refinement 的剩余预算；局部基线自己拥有相同总预算。每个模式最多保留 K4，锁定实际局部／联合见证，再用全赋值精确去重与差异性补充。完整归档 replay 不重新搜索，所有计数有累计收据。对比仍保留旧六来源流程，不把预算重置或候选缩减隐藏在缓存中。

### 精确支持集降低 Shuffle 展开

旧展开按 source-register × destination-register 生成 Shuffle。本轮按每个目标寄存器在全部 128 threads 中真正需要的源寄存器支持集生成，再按实际源寄存器编号选择结果。利用 GF(2) 仿射像加速求支持，并对有限映射验证；不能只检查单 warp 或假定 rank 固定。C++ 的工作计数、Python lowering 与 surviving-demand 审计一起对齐。新普通 Shuffle／受限转换路径支持源／目标最多 512 寄存器，并受每条转换路线实际展开工作不超过 65536、逻辑元素 65536、holders 262144 及静态 Shared 49152B 等边界约束；工作上限不是整个 kernel 的总指令上限。它们不是整个后端的统一上限：既有大复制 Register 例外仍沿独立的 demand／符号源 domain 保留，不把声明 holder 槽位全部展开。

CPU 覆盖 rank0..7、复制布局、跨 warp 支持集及 512 边界。真实 GPU 位级微基准按任意 uint32 搬运契约，以固定的两组内容及 IEEE 特殊位模式验证；240 项常规记录及四种 Sanitizer 的 192 项记录全部通过，含非默认 stream 与 CUDA graph。它验证数据搬运语义，不是完整 kernel 的性能证明。五步位交换网络仅有探索，不在交付实现与 GPU 加速声明中。

### 地址表、输出与编译身份开销

- `Emitter.offsets` 以本次 Emitter 的 storage-value id 缓存，原计算完整保留在 `_compute_offsets`。异常不缓存，不使用跨编译全局缓存。真实 N65536 同 packet 的一次 CPU 对照，1537 次表构造降为 4 次，50.7815s 降为 7.7070s，完整 TTGIR／config 字节一致。这是发射局部一次性测量，不能写成整个搜索编译流程 6.59 倍。
- 可选 `--minify-execution-json` 在最终输出处省略空白，默认及 false 的旧格式仍字节兼容。解析对象、TTGIR、config 不变。小型真实 CLI 对照从 289686B 降为 94018B。大型旧 packet 的空白格式投影约节省 66.8%，本轮整体验收已使用真实新 CLI 产物。更激进的 compact-map 方案因 CPU 发射变慢被拒绝，未纳入交付。
- 精确编译缓存以完整 TTGIR／config／target／options／生产源／工具／环境及 Triton payload 内容绑定。每次实际执行仍有资源与 image 验证。SDK 大文件不再在每次 exact hit 重复通读；session 进出和每个新编译产物发布前执行完整内容校验，命中路径检查元数据、payload 文件集合，worker 加载前再完整校验。38 项测试及 5 项独立变更测试通过，包含相同 mtime／大小、编译中内容变化、无效产物不得发布。旧 r1 暴露的时间戳问题保留，r2 已修复。

## 冻结与验收设计

正式源 digest：`8f3bb5005b651713f94fdf81e830d24e45badfac7613dd188e083ed74c39d61e`。`development/source-freeze-final/freeze.json` 包含可重放 patch、完整源码及五个构建产物的身份。Git HEAD 保持原值，尚未提交。

整体验证：680 项 C++ 单测、53 项选择测试、14 组 CTest、80 项 IR 测试全部通过。对应 `development/integrated-validation-r1/summary.json`，源文件前后哈希相同。

正式运行 `run-gf2-decision-refinement-r1-20261104` 注册 15 个规格、13 种同池策略：新留出 diamond256x128、remat512x64、conflict128x256、mixed1024x64、div256x128、exp256x128；另有 R3 六个 observed regressions 和三种 Attention ABI 锚点。新留出与 47 个历史完整／flattened 程序去重，不通过重命名恢复新鲜性。mixed N65536 保留，不选择更容易的形状。

所有候选必须完整生成，归档候选 replay 不增加搜索（legacy-static 参考另有已声明的独立 J 搜索），并完成发射／需求覆盖、CPU 编译、数值与 canary、非默认 stream／graph、四种 Sanitizer 或严格 exact-safe reuse，然后分析并锁定所有决策，经独立 pretiming 审查后才测时间。每组 10 paired batches、每 image 5 events，不替换无效批次。

门槛保持预注册：六个新留出有限候选联合池的 regret 上界 GM≤1.03、最坏≤1.10；主策略对 legacy 的时间比区间上界小于 1/1.03；15 个 guard 用 alpha=.05/15，逐项上界≤1.05。无限空间最优及 Triton／TileLang 优势均未由该实验证明。

新 CPU 开销实验以相同当前 lowering、准备域、minify 与编译选项重做完整六个已观察程序：old-six 和 new-archive 各自私有 cache，以 CPU0..3 亲和约束启动，资格验收使用 CPU4..191，内存和 IO 共享。要求完整 generation、generation+cold compile、warm compile-only 均降低。一轮描述性测量无统计 CI。共同注册／父进程启动／外部 analysis／安全验证／kernel 时间另列；API 次数、exact hits 与未知的编译器内部命中分别统计。

## 新留出GPU结果、消融与解释

主统计仅汇总六个新留出，另外九组只作观察过的回归／ABI守卫。参考最优仅指登记的有限归档与legacy联合池，不是无限布局空间的最优。采用完整配对批次bootstrap（10000次），没有把同一块的5个event当独立程序样本。

| 指标 | 点估计与区间 | 预设标准 |
|---|---|
| 主／legacy时间比 | 0.660491，95%CI[0.659312,0.663008] | 上界<1/1.03；通过 |
| 平均遗憾 | 1.003096，95%CI[1.000292,1.003804] | 上界≤1.03；通过 |
| 最坏遗憾 | 1.016680，95%CI[1.001451,1.019543] | 上界≤1.10；通过 |
| 15项回归守卫 | 全部通过，最坏上界为R3 conflict约1.003430 | Bonferroni α=.05/15、逐项上界≤1.05 |
| 主／充分知情LOCAL时间比 | 0.257481，95%CI[0.251800,0.258051] | 同K4／probes／refinement预算，约3.884× |

| 新留出 | 主选中位数 µs | legacy µs | LOCAL µs | 有限池最快 µs | 配对regret |
|---|---:|---:|---:|---:|---:|
| conflict-128x256 | 48.042 | 67.153 | 311.236 | 48.042 | 1.000000 |
| diamond-256x128 | 47.098 | 70.787 | 47.098 | 47.098 | 1.000000 |
| div-256x128 | 154.493 | 454.672 | 416.360 | 154.493 | 1.000000 |
| exp-256x128 | 50.066 | 68.762 | 312.912 | 49.892 | 1.002006 |
| mixed-1024x64 | 347.960 | 347.960 | 1325.624 | 342.544 | 1.016680 |
| remat-512x64 | 49.398 | 67.917 | 404.464 | 49.398 | 1.000000 |

前四列是各自边际中位数，regret来自配对比率及候选比较的bootstrap，因此不必恰好等于两列中位数之比。mixed主策略与legacy同物理图像，未获得额外速度；与有限池最快仍有约1.67%的配对差距。三种Attention锚点的主策略与legacy也同物理图像，时间比为1。

| 同一联合K4池的规则 | 平均regret | 最大regret |
|---|---:|---:|
| joint-interval-only | 1.643608 | 6.496125 |
| joint-gf2-tie | 1.203250 | 2.979063 |
| joint-frame-only | 1.370203 | 6.496125 |
| joint-tie-frame | 1.003096 | 1.016680 |
| joint-affine-frame | 1.003096 | 1.016680 |

GF(2)细化修正新conflict中原interval-only约6.496倍的遗憾；driver帧后备修正新div中原规则约2.979倍的遗憾。两种修复组合相对interval-only时间比0.610302，95%CI[0.609151,0.612024]，约1.639×。它们在相同算法、ABI、精度、后端与同一实际候选池内比较，能够隔离决策机制的影响。

本轮affine-frame与tie-frame全部选中相同实际图像，时间比1；**没有确认affine完成族额外GPU速度收益**。它的交付价值是可重算的模型族证书与对共享未知费用的正确处理，不能把数学支持范围扩展伪称实测加速。114个归档区间实际为64 exact／50 unsupported（无bounded），联合8组affine-certified／7组fallback，LOCAL5组certified／10组fallback。可计分候选中38个有已知正LOCAL传输特征，说明新覆盖实际参与评分；但所有13策略使用同一已校准模型，没有新的旧／新LOCAL系数独立消融。

资源语义也已逐项核验：129配置中77行（114归档中69行）cuobjdump rawLOCAL=0但driver LOCAL帧>0。一个diamond legacy配置为rawLOCAL0、STACK2504B、driver帧2504B，而SASS LDL＋STL静态字节9776B。这四者不能混用，尤其不能把rawLOCAL0当作无分配、无传输或无spill。spill数若没有可靠字段仍保持unknown。

独立统计先于正式assess。suite／guards472项及逐组5528项数组／字段比较全部差0，195策略的实际identity、regret和中位数、1170策略pair及96图像向量一致。model_scores／maps为空，因此没有给出通用全模型rank-inversion指标，也没有验收通用绝对时延外推或跨GPU精度。

正常功能检查的capture门禁是实际cuGraphGetNodes==3；其独立numeric字段没有另外持久化，证据是冻结worker门禁与完整成功记录。性能测量使用另一套已登记pilot自适应节点数：实际每图2—754节点，累计148610个capture节点、743050次正式节点执行；480个pilot event单列，不算入4800个正式event。没有固定为1024节点或3节点的性能假设。

计时启动的第一次编排额外设置CUDA_VISIBLE_DEVICES，与注册null不一致，被环境门禁在创建measurement目录前拒绝，2.011s内退出且无性能样本。原失败保留。第二次完全恢复5项注册环境，沿用冻结worker的torch.cuda.set_device(1)，在GPU1 UUID `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618` 完成唯一完整测量，175.076s。没有替换或重跑已采样批次；GPU外部PID只在已登记批次边界观测，不宣称连续系统级排他证明。

已登记measurement.confidence文字仍继承“nine／.05/9”，这是描述性元数据瑕疵。实际acceptance明确15组，冻结的正式及独立代码均使用.05/15（单组置信度99.6667%、family α=.05），比旧文字更严格。没有测后修改协议、阈值、形状、样本或系数。独立流程审查明确记录此差异。

## 完整六程序开销结果（独立验收完成）

两个流程均使用当前冻结的后端、当前重建准备域、相同 minify 选项。六个已观察程序全部完成，并非前缀外推。

| 范围 | 六来源 old | 共享归档 new | old/new |
|---|---:|---:|---:|
| 生成（含各模式准备、搜索、replay 和发射前检） | 452.081s | 264.737s | 1.708× |
| 冷编译完整 mode wall | 549.413s | 155.437s | 3.535× |
| 生成＋冷编译完整 mode wall | 1001.494s | 420.173s | 2.384× |
| 热编译 only，不重跑生成 | 428.208s | 93.087s | 4.600× |

36→6 次 solve，156330→52034 probes，3193→990 refinement；编译请求每相位 36→24，worker 36→1。冷相位 old36/new24 次 API、双方 exact hits=0；热相位 old36 次 API（内部命中未知），new0 次 API／24 exact hits。new 冷相位 API 累计16.252s，不应把155.437s整个 mode wall 写成 native compiler 自身时间；其余包含发射、证明／身份、产物与会话工作。SDK full content checks 冷26次（进出＋24发布），热2次（进出）；statchecks 分别50和26。

一次描述性观察，无统计置信区间；内存／IO共享，old后new固定顺序，OS page cache未宣称隔离。共同生成 outer1.942s、注册和父CLI启动／其他阶段audit另列。以上不是完整选择加安全验收的端到端时间，也不与不同 GPU 形状混合计算 break-even。分析、数值／Sanitizer 和 kernel 时间单列。旧R3数值仅历史，不能当作本轮对照。两轮ratio也不能直接比较：本轮两侧都使用新后端，旧六来源也获得底层改进；ratio变化不是同一基线上的退化或加速证明。

监视器因把线程mask要求为精确{0,1,2,3}，误报合法{1}；实际warm进程未被停止、未重启、未被改亲和，完成同一次完整计时。原监视错误和日志保留。续监视按CPU0..3非空子集核验；从误报记录到续监视启动间隔111.613s，原最后／续第一次精确采样时间未记录，不编造。补存快照和纠正sidecar，不能声称每个时刻每个线程均连续被观察。

完整数据：`development/efficiency-current-r1/summary.json`，SHA `c2bb233c657c5939888f930a433b2b420a7f618c7014081d792826834a677711`。独立全产物／域／计数／cache链审查另存。

独立开销验收结论为 `DESCRIPTIVE_CPU_OVERHEAD_REDUCTION_PASS_WITH_MONITORING_LIMITATION`，报告 `development/independent-current-efficiency-acceptance.json`，SHA `45d4794fa14a733527f3f2205ea2128cbd78f4a3fb043229daff35685c82e8d9`。独立核验全部预算／域桥、120编译记录、48缓存回执及精确context；各程序生成均下降。跨模式12组相同emission的CUBIN容器差异严格限定为 `.debug_line`、`.rela.debug_line`、`.nv_debug_ptx_txt` 三个非ALLOC调试节，175个非debug节payload、ABI、符号与资源metadata一致；同mode冷暖CUBIN全部字节相同。最初要求跨路径CUBIN逐字一致的过强审计断言及其失败保留，未改变原始编译结果。

监测限制补充：8个旧warm worker无亲和采样；111.612735s只是failure记录至续watch启动事件差，精确采样缺口未知。续监视器也不能直接reap原warm OS exit，因此不能声称亲自观察该原父CLI的exit0；全部37个内部worker命令返回0、完整phase产物／计时／同PID持续运行和无重跑替换链成立。最终单次CPU结论带此限制，不把CPU验收当GPU质量或LOCAL系数边际收益验收。

本轮15规格的129候选已全部生成、编译并通过GPU功能与安全验收，范围包括两个模式及legacy参考，大形状实包最高约85MB。`development/qualification-generation-boundary-observation-r1.json` 保存注册created_ns至生成manifest mtime的6009.727s（约100分钟）墙钟边界；它包含注册收尾、父进程启动和验证，不是perf_counter测得的纯生成时间。不能用小形状六程序的420.173s外推本次更大完整验收的成本，也不能把整个129候选的资格验收成本当作一个生产候选的成本。大形状的发射／需求证明开销仍显著。

注册元数据说明：输入cfg继承的development_only=true、frozen_for_gpu=false为描述字段，不是执行门禁，也不是验收完成凭据；实际冻结和有效依赖以effective protocol、registration、source/binary freeze及独立前置证明为准。profiles/preserved_evidence在init重建，包含活动兼容视图与四套旧封存索引。相关补充在`development/registration-metadata-clarification-r1.json`；CMake未生成compile_commands，已另存完整build.ninja与只读ninja -t commands，并由独立审查核验。未更改原cfg或验收标准。

## 大形状可扩展性检查（当前冻结版本的限制）

最大mixed1024x64的8个归档候选CPU重发射与正式TTGIR逐字一致。实际Shuffle／内部选择计数：J0=L0为16384／15872，J1/J2/J3为0，L1为49152／47616（三条Shuffle路线），L2为32768／31744（两条），L3为16384／15872。每条路线的每warp支持集及all128thread union都是32，没有假设中的跨warp四倍冗余。当前这些程序的物化边界查询全部统一，潜在dense receiver mux分支未触发；Register走大需求递归，不能把普通convert的全NR选择计入它们。未发现新的语义或成本记账缺陷。

J0完整单请求编译175.612s；最大L1完整单请求604.895s。二者不是API净时间。L1的TTGIR约14.76MB；J0和L1的ELF `.text.frisk_execution` 分别约2.55MB／4.52MB，包含同节div慢路径helper。原生静态记录包含控制与NOP，不能当动态指令数；原始CUBIN还含调试信息，不能当纯执行text。这里没有Ncu动态证据，不能据此直接归因于ICache瓶颈。

每条转换路线的65536工作cap不限制整个kernel的总静态工作或PTXAS复杂度。稀疏化已避免无支持源的展开，但大形状仍有显著、真实且有界的IR和编译代价；小形状六程序2.38倍冷路径收益不能外推到所有大形状。

只读审查还给出当前J0五位交换网络的CPU索引等价证明：65536 holders零差异，理论1280次SHFL＋3840次选择。该网络没有纳入生产实现，没有原生编译／GPU验证，不能算本轮实测加速，也不能用此前不在相同支持域运行的dense512作实测对照。它只是后续降低结构工作量的数学候选。

证据：`development/largest-compile-scalability-audit-r1.json`（a9d5c40b…）、`largest-compile-native-and-cost-supplement-r1.json`（86d2ee86…）、对应说明及bit-swap-reference CPU脚本／结果。

完整129候选的正式批编译收据：`run-gf2-decision-refinement-r1-20261104/batch-compilation.json`，SHA `79f37c758a13a22cf59b098fbc0908ed680c9d7322f2c092d072b8a22def0a22`。wall3492.757s，129 requests＝96 API＋33 exact hits，API累计1964.112s；默认Triton内部cache状态未知，不宣称此资格阶段是全冷编译。SDK fullchecks98、statchecks227，session初始化3.839s。它包含大形状与两个模式／参考，不能与六程序小形状的mode计时混合。

纯原生分析预取在编译期间并行，使用原有image＋工具＋analyzer源码绑定cache；仅原子写cache，不写case／manifest、不评分、不attach bank、不运行GPU，观察到compiled=True／验证启动后合作式停止，允许已开始的调用收尾；本次PID已退出，正式analyze在其退出后序列化启动。这个采样式检查不是强实时原子handoff，API的固定临时文件名也不是通用多writer锁。126/129已处理（47hits／79misses、79独立cache条目），剩3个TMA候选留正式阶段；无瞬态工具故障。编排wall626.209s含等待，API累计wall146.635s，控制进程CPU151.976s不含工具子进程CPU。后续正式analyze是温缓存，预取工作没有消失，不能称cold分析加速。完成收据`development/native-analysis-prefetch-r1/completion.json` SHA `2043ccea4c10e65ce5c21060849ed5f6f582edc4cc5759902ad4356233f8c5aa`。最终决策仍在全部GPU验证后绑定并锁定。

预取独立数据审查：`development/independent-native-prefetch-data-review.json`（SHA `8cde48a2e7add3e52201014918331e8d2e1e3dfde5f03a4d1c9a4276d7477689`），1059个文件核验通过。控制检查与API调用之间存在TOCTOU；没有逐次调用Unix时点或完整系统调用trace，不作强实时或全写入轨迹证明。当前可证明合作式停止、worker退出、三个TMA候选未预取以及与正式analyze无重叠；这些事实足以支持本次私有cache的串行使用，不泛化为多writer安全。

## 最终证据、保留的失败及当前边界

最终综合记录为 `release-qualification.json`（SHA `c3303b6cff9af6144dd736b1580cec06bfd66f2b7b042411eed65c437a7f9490`）。GPU独立最终报告为 `development/independent-final-GPU-evidence-acceptance-r1.json`（0eb3d7c9…），CPU独立最终报告为 `development/independent-current-efficiency-acceptance.json`（45d4794f…）。正式GPU报告 `run-gf2-decision-refinement-r1-20261104/quality-summary.json`（b3416088…）原样保留中间promotion待效率字段，由综合记录确认两组件合格，不改变默认选择策略。

以下诊断完整保留：上一轮R3整体GPU质量失败和全部更早失败／中断；本轮cache-identity-guard-r1暴露的粗时间戳发布漏洞（r2已修复，并通过38原测＋5独立变更测）；CPU监视器将合法亲和子集误报、8个旧warm worker无采样的限制；跨目录CUBIN逐字相等的过强审计断言（严格调试节证明后澄清）；measure-r1环境前置拒绝无样本；继承配置bool／列表与九守卫描述文字瑕疵。未以删除失败、替换批次或复用已观察R3为新留出来获得通过。

本轮仅承诺上述支持域和内部候选集下的功能、选择及开销门槛。大形状IR／编译成本仍显著，五位交换网络尚未实现；复杂native CFG仍采用明确不支持与静态／资源后备；绝对成本外推、通用模型排序及跨GPU性能未验收。CPU六程序与GPU六新留出为不同cohort，旧六来源和新归档的搜索组织／预算分配也不同，不能组合计算净应用加速或混合break-even。本报告不替代后续Triton／TileLang公平对比。

四套历史封存41351文件／32928590880B已逐文件与精确文件集核验通过，另核native和partial两个外部报告，记录位于 `development/old-sealed-preservation-final.json`（8ee16425…）及 `old-external-report-preservation-final.json`（be845ac1…）。Git HEAD仍为 `9f79155ac08f82133f4ddb6554af552feefca8d2`，index SHA仍为 `4602fb66327ad70bb8f959bd766b712cf0ca8daa3bde2d72c037613dc745aa92`，与任务开始一致；生产代码recipe digest为 `8f3bb5005b651713f94fdf81e830d24e45badfac7613dd188e083ed74c39d61e`。没有创建提交。最终证据索引保存为证据根的 `evidence-index.json`，包含本报告外部只读引用。
