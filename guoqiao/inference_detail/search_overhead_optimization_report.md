# 搜索开销优化实施与验收报告

日期：2026-11-01（Asia/Shanghai）。状态：实现、回归、438次CLI、源码与数据独立复审通过，阶段在明确支持域内完成。服务器 `/home/baopeihua/frisk`；无本地代码副本，无GPU运行。

## 前置提交与范围

按用户要求先将此前联合消融、校准、资源修复的代码/测试/报告共88文件提交为 `9f79155ac08f82133f4ddb6554af552feefca8d2`，未推送。大型产物和build目录未纳入Git，封存文件原始字节保留；两处既有EOF空行在stage check按显式允许blank-at-eof口径处理，没有改写已封存报告/源码。本阶段之后的优化修改留供复核，未自动追加提交。

本轮只复用重复的纯查询，不改变候选域、合法性、MRV/tie、精确/beam/refinement算法、预算、成本模型或转换计费。通用时延/寄存器预测和跨框架公平性能比较仍是独立限制，不通过本轮CPU收益宣称kernel加速。

## 热点与实现

优化前同CPU四Attention开发profile：free wall13.24s/solve11.63s，access1169次/inclusive1.37s；weak wall7.43s/solve6.09s，access1519次/inclusive1.98s。cp.async/TMA约1.30/1.74s，总工作负载不同，不能跨ABI相除。这是一次开发归因样本，不是正式配对结论；各profile inclusive嵌套时间不可相加。

`evaluateStaticLayoutCost`在partial/full评分中反复验证相同不可变(type,encoding)，`accessCost`对重复layout/Copy配置重建execution枚举、逻辑坐标地址表、warp/bank分组及对齐phase。两个graph-owned缓存：

- 成功静态验证按完整Type+Attribute复用；失败仍执行并保留原诊断。Shared根去重/最大跨度、资源聚合和observer顺序仍逐次计算。
- 成功原始StorageAccess按constraint、两端Type、encoding/layout、完整Copy binding（null独立）、AccessKind、root alignment复用。原始项尚未应用profile/frequency；每次命中仍向observer记录charge，v2仍按原规则重新加权。

每个缓存最多4096项，仅finalized graph生效；容量满正常计算，禁用正常计算。结构变动和finalize入口先清缓存、置非finalized；失败finalize不残留旧ready状态。IR/alias/root合同变更必须重建或re-finalize图，candidate pruning不改变这些事实。类型、属性和Copy binding是不可变MLIR对象；此缓存没有跨图或跨进程持久状态。

正常pass/pipeline支持 `memoize-costs=false`，只用于关闭计算复用，逻辑trace不记录该非语义选择。profile增加 `access_cost_cache`、`static_cost_validation_cache` 和 `static_cost_validation`；前两项count=hits+misses，entries指插入次数而非峰值内存。关闭时查询计数为0、实际decoder仍计量。

## 冻结协议与完整等价验证

原登记运行 `run-search-overhead-20261101T003017Z` 在correctness阶段执行25次CLI：四Attention的24观测全部完整等价，首个diamond自然pipeline在baseline返回cost-failure后停止。0次正式性能计时，performance_eligible=false，全部原文件及旧driver/test字节保留，不混入最终统计。

该diamond的独立静态归因：自然域var6含Eall `[1024,32,4,1,1]`，131072 carriers；其第一完整赋值经过StorageAccess c14，而原始访存枚举上限65536，Unknown导致cost-failure。候选仍满足普通type和128线程硬约束，合法性与成本支持域不同。这是既有自然成本支持缺口，本阶段没有改候选、预算或把Unknown当免费。其他较大机制图的失败只按原始证据各自登记，不以diamond单例归因代替全部诊断。

30次开发支持预检确认：原4大shape与原v2 diamond的baseline/final/off全部相同cost-failure；固定≤512元素小shape及small v2 diamond全部成功并且IR/stats/trace一致。按支持边界修订而不按速度选样本，具体前后关系在`amendment.json`。原登记SHA和旧运行继续作为历史，未回写。任何接受性能样本之前将新登记、重新冻结全部工具/源。

最终矩阵计划16case：9timed为四Attention、diamond32×16/remat32×16/mixed32×16/conflict64×8和10操作无StorageAccess算术链；2个成功显式旧v2 profile守门（weak Attention及small diamond）；另5个原大shape/v2diamond保留untimed expected-cost-failure。失败行须reason/kind、exit、无IR、完整non-timing stats与未截断trace全相同，并在最终报告列N/A、不进入速度汇总。v2使用既有707B冻结profile，不重新拟合。

主默认256/32/65536、refinement256、legacy模型和频率不变；baseline/final/final-off各有profile on/off语义守门，共96 CLI。性能主测仅profile-off，fresh process+CPU0，各variant一warmup与10随机配对，9行90配对共198 CLI；wall/user/system与每子进程maxRSS分列。profile归因与final on/off消融各3配对、各72 CLI另列，全部接受阶段共438 CLI。warmup只热OS/page cache，不能称跨进程缓存复用；不将嵌套时间相加。正式计时无并行构建、测试或GPU运行。

接受运行：`run-search-overhead-20261101T005848Z`。protocol SHA `fcbeb84cc6ea2cce4cc3503916b03237770513b728c30bb368da2d4e52e94e29`，registration SHA `b82e9337e7c4a7cf5b62870d7e2bb7ab2ecec1f231126f1539df3d3fb1eb0bd9`。candidate native binary仍为 `0d6956bb0c2162903a94d736b90dcd602ef4cfd420516a5895467d3985e81994`，与首轮native回归相同；修订仅改变两个Python harness文件。Torch Python3.10.18与preflight/measurement/控制器argv完全一致，阶段前后校验共同动态库身份。

新96次语义守门全部通过：66次成功pipeline输出、30次预期cost-failure，其完整IR（或明确缺席）、非计时stats与trace完全等价。5失败行不算成功编译、不进入性能汇总。首次run的system-Python预检与实际Torch身份差异仅作诊断桥保留，未回写原快照；最终接受run没有该差异。

## 正式性能结果

下表wall为每variant十个主样本的中位数，加速比为十个配对ratio的几何平均，故不必等于两个中位数直接相除。95% CI为固定用例内配对子进程块bootstrap，不是不同工作负载的泛化区间。仅主profile-off数据，9行均提升；等权9行描述性几何平均为2.10088×，不推广为实际业务平均收益。

| 用例 | 原版wall(s) | 新版wall(s) | 配对加速比 | 95% CI |
|---|---:|---:|---:|---|
| Attention 自由布局 | 13.22085 | 1.75117 | 7.5587× | [7.5436, 7.5745] |
| Attention 弱ABI | 7.39414 | 1.41561 | 5.2207× | [5.2090, 5.2336] |
| Attention cp.async | 1.30218 | 1.05460 | 1.2334× | [1.2312, 1.2353] |
| Attention TMA | 1.74078 | 1.32041 | 1.3182× | [1.3149, 1.3218] |
| 10操作算术控制 | 0.03489 | 0.03122 | 1.1352× | [1.1138, 1.1601] |
| diamond 32×16 | 3.05954 | 1.80433 | 1.6959× | [1.6929, 1.6991] |
| remat 32×16 | 3.28435 | 1.87068 | 1.7543× | [1.7513, 1.7575] |
| mixed 32×16 | 4.28419 | 2.25046 | 1.9023× | [1.8977, 1.9066] |
| conflict 64×8 | 5.39376 | 2.79071 | 1.9341× | [1.9226, 1.9461] |

solver内部计时、整条CLI CPU(user+system)时间及进程RSS另列。计时均为主profile-off中位数；RSS为十次计时子进程峰值的最大值。层级timings有嵌套、不同样本中位数不具有可加性，不能相加解释wall。

| 用例 | 原版→新版solve(s) | 原版→新版CPU(s) | 原版→新版RSS峰(MiB) |
|---|---:|---:|---:|
| Attention 自由布局 | 11.601341→0.142872 | 13.21815→1.75002 | 72.133→72.211 |
| Attention 弱ABI | 6.056506→0.082815 | 7.39236→1.41487 | 73.918→74.730 |
| Attention cp.async | 0.309157→0.060069 | 1.30111→1.05352 | 73.691→73.637 |
| Attention TMA | 0.464599→0.049986 | 1.73990→1.31970 | 74.402→74.387 |
| 10操作算术控制 | 0.019706→0.015766 | 0.03434→0.03041 | 55.027→55.098 |
| diamond 32×16 | 2.864036→1.607314 | 3.05825→1.80340 | 60.121→63.191 |
| remat 32×16 | 3.086976→1.673377 | 3.28267→1.86946 | 60.332→63.266 |
| mixed 32×16 | 4.081599→2.046299 | 4.28300→2.24932 | 60.609→63.086 |
| conflict 64×8 | 5.197516→2.593909 | 5.39179→2.78956 | 62.574→62.254 |

小算术控制的user CPU中位数约23.26→24.64ms，system约11.49→6.09ms；不能写成每项CPU指标都下降，也不把该行全部wall差只归因某一缓存。

内存代价如实保留：diamond/remat/mixed的RSS样本峰分别增加3144/3004/2536KiB，约3.07/2.93/2.48MiB；这不是GPU寄存器或Shared变化。缓存每个最多4096项，但进程RSS包含全部解析、证明、allocator与输出，不能把RSS差值全算成缓存对象大小。

## 归因、消融和残余开销

独立profile与同一candidate的on/off消融各27配对，均开启profile、只作诊断，不混主数据。on/off整条wall配对加速在free/weak分别7.4769×/5.2332×，与主比较方向一致；开启观测本身会改变小开销，不能用该数值替代上表。

| 用例 | 静态实际验证 off→on | raw access查询/命中/未命中 |
|---|---:|---:|
| Attention 自由布局 | 14209→18 | 1169/1132/37 |
| Attention 弱ABI | 8749→13 | 1519/1491/28 |
| Attention cp.async | 908→11 | 63/53/10 |
| Attention TMA | 663→9 | 14/7/7 |
| 10操作算术控制 | 7254→3 | 0/0/0 |
| diamond 32×16 | 4919→5 | 455/439/16 |
| remat 32×16 | 6379→5 | 464/448/16 |
| mixed 32×16 | 8846→5 | 624/608/16 |
| conflict 64×8 | 6251→5 | 464/448/16 |

free的实际验证14209→18，raw access1169查询只有37次未命中；weak为8749→13与1519查询/28次未命中。查询及逻辑搜索次数不减，减少的是重复decoder/访存成本计算，observer仍每次重放。off两cache count/hits/misses/entries均0，真实decoder继续执行。entries为累计插入，不推断live/peak缓存内存。

free solve11.60134→0.142872s、weak6.05651→0.082815s；准备、IR物化、转换后优化和验证基本保持原量级，因此整条CLI收益低于单独solve比。4小机制图新版solve仍约1.61/1.67/2.05/2.59s，不能称搜索全部开销消失；owner/转换分类和首次计算等仍有后续优化空间。本阶段没有分别只开一种缓存的2×2消融，不把联合收益精确分摊到两种缓存。

测量条件：AMD EPYC 9654 96-Core Processor，主机`amax9518`，单CPU亲和0，Release/O3、相同编译配置，Torch Python3.10.18。fresh进程，OS/page warmup只热文件/系统缓存，graph内缓存每次从空开始；parent wall含1ms轮询粒度，小算术控制的毫秒级差值须结合CPU/solve和CI看。没有kernel时间、JIT或GPU性能结论，也没有测v2的时间提升。

5个原失败case（diamond32×32、remat128×16、mixed128×32、conflict256×8、large v2diamond）仍为N/A。本轮不扩展这个自然成本支持域；冻结跨框架比较前需明确修复或限制该域，不可用当前缓存收益掩盖。




## 回归与交付证据

完整Release主单测613项、选择质量45项、CTest10/10均通过，其中新增3项验证缓存与7项raw access缓存回归；修订driver轻量测试14项通过。独立4负例×三变体和arith正例on/off共14 CLI均一致，off profile满足两cache查询全0而actual validation>0。预算确在search-budget退出；hard-conflict/unknown-scope fixture由normalization拒绝，invalid-options为invalid-input，不误称后三者都是solver HardConflict/Unknown。arith正例on实际decoder3/off7254，只有计数证据不作主时间估计。初次统一编译因新测试缺少getStorageAliasFootprint声明失败，原日志保留；只补测试include/合法rootKey，生产代码首次编译无相关错误。Unknown失败测试也从先被静态layout上限拒绝的大域改为小合法layout+unsupported symbolic alias，确保触达真正access footprint Unknown路径，未放松检查顺序。

全部438接受CLI中408次exit0、30次预期cost-failure；90主pairs、27profilepairs、27cache-on/off pairs。独立数据审计重新核2886份phase产物、376份按harness配方的源文件和5项动态库；逐块重算全部bootstrap/CPU/RSS，与报告一致，0Critical/0未关闭Important。旧3阶段16223封存文件约3.858GB、3旧index自身和7份旧报告/计划/proposal逐SHA未变，380本阶段起点与9f79155 Git blobs逐字节一致。

正式测量后补重链接未测`frisk_joint_ablation_bench`消费端，仅构建不改源。已测FriskOpt在前后均为`0d6956bb0c2162903a94d736b90dcd602ef4cfd420516a5895467d3985e81994`，不将新consumer身份混入已测两工具。源码包/最终索引详见下节交付文件。

前置资源修复及其他sealed目录/报告不回写。本阶段起点380源文件与5长期文档前像、baseline tools/source/config/dynamic依赖保存新stage。完整final源码包382文件（14修改+2新增），harness身份配方376源文件当前与测量时逐SHA一致；旧失败run到接受版本仅两个Python harness变化，旧字节与身份桥已保存。源码patch已独立于Git状态在server临时目录git apply并逐字节复放382/382，五文档patch也已复放且原字节后缀不变。交付将含可复放patch、final源码清单、一个证据索引和独立whole-review；阶段成功不以每一行都出现正收益为条件，未提升和退化均保留。

## 交付与后续边界

当前HEAD仍`9f79155ac08f82133f4ddb6554af552feefca8d2`，这是前置提交；本阶段新修改未暂存/提交/推送。统一计划、`search_overhead_optimization_artifacts/evidence-index.json`、`whole-review.md`、`source-final-manifest.json`、`stage-implementation.patch`与`source-package-audit.json`分别记录最终证据、独立审查和起点→最终字节复放。5长期文档按前缀更新，原文后缀逐字节保留。

本次收益是相同候选/预算/目标/选择轨迹下的CPU推断与搜索开销减少，支持域内输出布局和转换未改变。当前成本/资源泛化限制、5自然大形状失败、其余热点与跨框架GPU收益仍未解决。公平比较应使用冻结版本、同算法/ABI/精度与调优预算，不能把这轮编译优化当成胜过Triton/TileLang的kernel证据。
