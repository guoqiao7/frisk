# Frisk layout：选择质量、执行闭环与完整开销验收报告

**两项实现及所选支持域的执行、质量和完整开销复验已完成。新留出选择质量与21组退化保护通过；3%性能晋升门槛未通过，默认继续使用legacy。**

## 版本、范围与统计方法

冻结源码474份，摘要 `7cf13ffa82ef7c6d1d82d4e1fe00bacbf7b7f0b5f7b1f2d261204eeafd192281`；完整源码／五个二进制见 [freeze-r3](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-source-freeze-r3/freeze.json)。[r4运行描述符](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-source-freeze-r4/freeze.json)沿用相同代码和二进制，仅更新外部控制器／数据盘位置，SHA256 `cca1413f456e7a6906d734b5235bbfb31823eba749595fda7ef6b0b8a831fe3d`。配置为 [qualification-configuration-r2.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/qualification-configuration-r2.json)，SHA256 `b2f43033ab39b0bf963be4fe4833a85cd5b15352856716132637ab4f69d11c28`。

正式冷运行根为 `/data0/baopeihua-frisk-network-20261105/run-network-selection-r3`。183个请求全部采用C转换模式、128线程、单CTA；新留出是6个FP32图：diamond512×128、remat1024×64、conflict256×256、mixed16×16、div512×128、exp512×128。另15个组是已观察回归，含三个Attention ABI。历史O模式消融、独立多CTA正确性和旧mixed1024×64诊断不并入新留出推断。

主规则固定为 `joint-affine-frame`，15个预登记规则复用原候选池，不按测量标签改策略。137个完整执行身份各有10个成对块、每块5个event样本，共1370条raw记录、210个组级paired blocks和6850个event值。bootstrap在每个固定程序内重采样完整paired block，10,000次、seed20261107；不把5次event当独立样本，不重采样程序。置信区间仅针对这个登记cohort与设备／协议，不代表新程序总体或全布局最优。

独立统计不导入生产统计函数，10项合成检验通过，931项输入哈希前后稳定。入口：[独立质量结果](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/quality-independent-result.json)，SHA256 `610a117ed8e8658229f9644ce42477990fa12a7020c29b00db1fd4a0a2d1391e`。最终 [独立总报告](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/report.md)及 [completion.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/completion.json)已闭合，后者SHA256 `5f55277d8fe98a22ddf55102c5069068c471ad3b2547286d6f5adf4c31a18406`。生产汇总在 [quality-summary.json](/data0/baopeihua-frisk-network-20261105/run-network-selection-r3/quality-summary.json)。

## 1. 主结论：质量合格，性能晋升不合格

以下时间比均小于1有利于分子策略，regret以有限实测archive＋legacy为参考，参考在重采样中重新计算。

|预登记判断|结果|门槛|结论|
|---|---|---|---|
|6个新留出：主规则GM regret|1.007143，95%CI [1.001709,1.013448]|上界≤1.03|通过|
|6个新留出：最坏case regret|1.038182，95%CI [1.005740,1.044640]|上界≤1.10|通过|
|全21组：主规则／legacy退化guard|21／21通过；最大上界1.020171|各上界≤1.05，21组Bonferroni|通过|
|6个新留出：主规则／legacy配对GM时间比|0.998293，95%CI [0.994599,1.003304]|上界严格<1／1.03≈0.970874|未通过|

J、LI和legacy共享同一个新网络后端与执行ABI，该比较隔离选择策略的贡献。主规则相对legacy只有约0.17%较低时间的点估计，区间包含1，未建立预登记的3%收益。因此 `kernel_criteria_passed=false`、`promotion.passed=false`、`production_default_changed=false`；不能用已有局部加速或资源改善替代该结论。

新留出逐例结果如下，避免把15个旧回归的显著收益混入新留出平均值：

|新留出|主规则regret点估计|J／legacy配对时间比|J／LI配对时间比|
|---|---:|---:|---:|
|diamond512×128|1.004286|1.000077|1.000000|
|remat1024×64|1.000000|0.994983|1.000000|
|conflict256×256|1.000957|1.000184|1.000000|
|mixed16×16|1.000000|1.000000|0.948186|
|div512×128|1.038182|1.000339|1.000000|
|exp512×128|1.000000|0.994198|1.000000|

充分知情LI对比：新留出J／LI的GM为0.991172，95%CI [0.990671,0.992539]。差异仅来自mixed16×16，其余五例选择同物理image；不能据此声称J在全部新形状均有优势。 15个已观察回归组的J／legacy成对时间比GM为0.769256，仅作旧回归描述性统计，不并入6个新留出的主推断。

参考有限已测池，事后每程序选一个固定最佳image的配对GM／legacy为0.990716，约0.93%时间余量；允许每block重选的更乐观oracle为0.984570，约1.54%余量。这两者均使用测量后选择，属于adaptive描述性结果，不是选择有效的收益保证、理论上界，亦不能由legacy regret取倒数替代。它们表明当前有限池的观察余量不大，不证明扩大搜索域或未来实验不可能达到3%。见 [finite-pool-oracle-headroom.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/finite-pool-oracle-headroom.json)。

## 2. 同预算、消融与剩余模型缺口

J／LI各例上限相同：65536 probes、256 refinements、native K4；真实网络变体重评分计费，不能增加第5个native候选。实际工作量不同：

|21组累计|J|LI|
|---|---:|---:|
|实际probes／单例最大|141754／13043|87761／8091|
|实际refinements／单例最大|2232／201|1764／174|
|逐例unused probe caps之和|1234502|1288495|
|逐例unused refinement caps之和|3144|3612|
|实际native请求|81|81|
|真实MinWork变体重评分|0|1|

unused仅用于逐例记账，不允许跨case转移。加21个legacy请求共183个；这不是42个池全部展开完整网络frontier。 另盲登记的旧mixed1024×64四点诊断完整报告MinShuffle／MinWork资源取舍；MinWork在J0、L1分别慢约10.755%、1.170%，不能拿降低REG或提高理论驻留上限当成延迟收益。见 [四点诊断](/data0/baopeihua-frisk-network-20261105/minwork-resource-latency-r1/diagnostic-report.md)。LI包含自身producer工作、incoming repairs与资源，completion仍允许已声明的辅助变量变化，不是naive repair-free局部基线，也不是全图oracle。计数与别名证据：[paid-budget-and-alias-review.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/independent-pretiming-r3-review-r1/paid-budget-and-alias-review.json)。

**规则和LOCAL消融。** 新留出中每mode七条规则全部选择同一物理程序，因此无法从这里识别tie／frame／affine／resource规则各自的额外收益。在全部21组，J的frame／tie-frame／affine／resource仅在已观察mixed512×64改变J-static选择：约304.970→123.936µs，成对比0.397280，约2.52×。该例native分析unsupported，收益来自资源frame后备规则，不能归因于native预测。J-interval／gf2与J-static全同；LI-interval／gf2／frame／tie-frame仅在已观察exp128×64不同，LI-affine／resource与LI-static全21同物理选择。

另行盲登记的LOCAL已知／未知两对控制，在J、LI各21组均为相同logical／physical alias；新留出两对均比值1、family2调整区间[1,1]。本cohort没有识别出LOCAL知情开关的额外选择效应，不等于LOCAL流量／校准普遍无用。unsupported例保持在全cohort，不冒充有效LOCAL样本。见 [LOCAL completion](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/local-supplement/completion.json)，SHA256 `119ab7892b25ce0e7a53a22c9defbbbe6685c5a84ec0c3cf8b46194820469def`。

**native覆盖和排序。** native14来自旧136个known images重校准；`local_transfer_bytes`系数为6422分值tick／静态LDL或STL操作数字节，模型标度为1µs=10⁶tick。它是经验费用，并非driver LOCAL frame字节项或通用硬件每字节延迟；未用新留出回馈拟合，static20仍是工程代理。183个请求的当前成本区间为108 exact、75 unsupported、0 bounded；`exact`指受支持语法内特征与模型费用确定，绝不等于精确延迟。42个J／LI mode中24个池完整可评分、18个不可完整评分，后者保留在主质量和fallback统计，没有筛掉困难例再报好结果。在双方相同完整域／相同物理pair上：

|同域cohort|完整mode|全部pair／可区分pair|native正确／逆序／模型tie|static正确／逆序／模型tie|
|---|---:|---:|---:|---:|
|6个新留出|8|48／21|19／0／2|20／1／0|
|15个已观察回归|16|96／63|60／2／1|39／14／10|
|全21描述性合计|24|144／84|79／2／3|59／15／10|

可区分依据预登记3% practical tolerance和mode内Bonferroni区间；J／LI可能共享pair，计数不是相互独立Bernoulli试验。native在已观察域改善多数排序，但新留出同域没有显示比static更多正确pair。两次native逆序均在已观察exp256×128的J／LI池，实际较慢约2.52×／2.82×；模型偏好LOCAL frame504／472B的程序，实际更快image为1776B，REG均255。资源相关性不能确定原因，排序逆序也不等于最终affine主选择必错。这是必须保留的残余模型缺口，不能用exact标签抹去。详见 [native-ranking-common-domain-and-inversions.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/native-ranking-common-domain-and-inversions.json)。

**事后代数诊断将两类缺口分开。** 84个可区分pair中3对native14完整等向量（fresh exp512×128 J两对、observed exp128×64 J一对），故任何14D重赋权仍不能严格区分；零差约束的单坐标\(\lambda=e_{pair}\)给出精确Farkas否定证书。Frisk已有GF(2)地址分析的sector轴\(g\)却分别为慢／快104448／30720、13056／3840。在96×14候选出现计数矩阵上，有\(d^TX=0\)、\(d^Tg=73728,73728,9216\)，所以在有理数或实数域均有\(\operatorname{rank}[X,g]=\operatorname{rank}X+1\)。这是普通计数列空间证明，不是GF(2)权重计算；只证明native14表示缺信息，不能外推整个Frisk系统无救、一定需要新schedule轴，或未知sector系数必为正。另两次大逆序并非等向量／坐标支配：当前static LOCAL-transfer项贡献压过shuffle等权衡，存在反事实重赋权可区分，但未拟合／应用，也不保证其它pair。既有GF2轴非本轮新发明，未单独证明新的留出性能收益；全部profile／主选择保持冻结。见 [等向量与贡献收据](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/algebraic-pair-diagnostic/algebraic-diagnostic.json)、[列空间非冗余收据](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/algebraic-pair-diagnostic/column-space-nonredundancy.json)。

池覆盖与池内选择另按逐block有限union oracle分解：`own_best/union_best × selected/own_best = selected/union_best`逐block成立。fresh J覆盖项的逐例成对中位数均为1（每批仍可有小幅变化），仍存在池内选择损失；LI另有覆盖损失。中位数／GM归约后不保持简单乘法，不把这套描述性block-oracle指标与主regret门槛混算。逐例结果见 [percase-decomposition-and-ranking.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-quality-independent-review-r3/percase-decomposition-and-ranking.json)。

全21组另作描述性核对：主规则regret几何均值1.008928，最坏为已观察exp128×64的1.109022（区间[1.079564,1.120866]）。这不属于新6组的登记门槛，也说明“新留出质量通过”不能扩写为全部已知case均在10%以内。两个代表性分解为：

|程序／mode|池覆盖损失中位数|池内选择损失中位数|
|---|---:|---:|
|新div512×128，J|1.000000|1.038182|
|新div512×128，LI|1.038182|1.000000|
|已观察exp128×64，J|1.105571|1.000000|
|已观察exp128×64，LI|1.000000|1.109022|

前者的J池有更快程序但后备选择未选中；后者的J主要缺在保留池，而LI保留了更快程序却未由主规则选中。两种缺口需要分别处理，不能只改一组权重后宣称全部搜索与选择问题已解决。

## 3. 执行与资源证据

最终冻结源码的703项C++、56项selection、80项lit、16／16CTest组通过；子集和CTest组不重复相加。474源码、五个主二进制和Git index前后一致，见 [final-cpu-validation-r3/completion.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-cpu-validation-r3/completion.json)。

正式冷运行183请求全部完成数值及安全门禁。非safety记录23355条，已包含3537条capture，另5427是输入／reference tuples而非5427次GPU检查。安全身份以(CUBIN,完整config,ABI)区分：137个不同身份、101个不同CUBIN，实际运行548个四工具进程，另46请求严格exact-reuse；不能把137称作不同CUBIN数。driver确认全部183请求的loaded-image资源；raw compiler LOCAL=0仍需driver查询，不据此声称无spill。详见 [覆盖与资源审核](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/independent-pretiming-r3-review-r1/coverage-native-resource-safety.json)。

独立当前执行域另覆盖22 single＋9 tiled配置、三Attention ABI及grid1／2／257。完整packet／storage／TTGIR／typedconfig／ABI与历史image按严格桥接核对；唯一有9处metadata差异的conflict256×8legacy走新compile与真实GPU，84普通／12安全记录、四工具零错。强Q／K／V标签输入／参考／分离证明当前再生一致；旧真实GPU证据明确保持历史身份，不说31配置全fresh重测。外审：[independent-selected31-review-r1/result.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/independent-selected31-review-r1/result.json)。

七份旧封存另经完整内容审计：57,401文件、46,824,368,663字节一致，无新增／缺失／内容差异；15个历史外部live路径的旧版本均在封存快照中找到相同SHA。它们保持历史来源，见 [audit-completion.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/old-seven-seals-content-review-r1/audit-completion.json)。

## 4. 首次／热缓存完整费用

下列完整费用对应21组／183候选的研究验收，包括全部数值、安全与选择复核；它不是一次kernel部署的编译时延。持久化compile.json内层墙钟的p50／p95／max约为8.210／109.209／236.961秒，包含缓存命中和不同程序，不是重复测量的时延置信区间。

本稿中的“cold”是本轮私有原生缓存首次空的登记运行，未清除OS page cache，不是机器或所有工具的首次启动。三段完整控制进程均actual exit0、scope_complete、后代已回收，无越核：

|完整控制进程|cold active wall秒|采样进程树RSS峰值GiB|warm|
|---|---:|---:|---|
|prepare：登记＋生成＋编译|6183.077672|12.724|6603.510050秒；12.770GiB|
|validate：数值／安全＋native／决策＋复核|9203.540349|3.673|4144.725869秒；3.049GiB|
|measure：正式测量＋assess|287.165449|2.089|不并入premeasure暖对比|
|上述三段active wall之和|15673.783470（约4.35小时）|不相加各段峰值|热仅两段合计10748.235919秒|

这是三个已闭合控制器active时间之和，不是从首次开发开始的历时，段间审查／调度等待也未冒充active工作。RSS是同一时刻进程和，可能重复计共享页，不是唯一物理内存。来源：[qualification-controller-whole-r3](/data0/baopeihua-frisk-network-20261105/qualification-controller-whole-r3)。

阶段内wall如下，属于上述外层的嵌套分解，不能再次加到外层总时间：

|WORK阶段|cold秒|warm|
|---|---:|---|
|01 register|74.705372|74.270176|
|02 generate|4892.233161|4909.999885|
|02b 全生成集合等值复核|无独立阶段|3.074621|
|03 compile batch|1213.812379|1612.229583|
|04 functional-safety|8818.191314|3705.656969|
|05 native-decisions|347.372195|374.024844|
|06 reverify-groups|35.639082|61.676390|
|07 measure|231.585063|不纳入premeasure比较|
|08 assess|54.030834|不纳入premeasure比较|

计划／证明／发射被这些实际命令覆盖，但没有单独完整计时，不能写作0或用减法猜测各自秒数。source search timers另累计1756.613秒，是generate内的嵌套观察；子字段不严格穷尽total，不把余量命名为某个未测组件。完整阶段记录：[WORK/state.json](/data0/baopeihua-frisk-network-20261105/qualification-workflow-r3/state.json)。

183／183原生请求、0失败／0未尝试：101次编译器API、46 full-key hits、36 exact compiler-input hits；一级137项、二级101项、实际101个不同CUBIN。批次外层1213.812秒，四worker内层1205.510秒，API累计2525.573秒、逐请求累计4052.371秒均是嵌套并行耗时，不能相加到外层；四核合计CPU4460.864秒。API调用也不单独证明一次冷PTXAS。当前CI2＋4worker是一个已实现组合，不是serial／parallel×CI2 on/off全因子性能实验，cache hit数不等于可归因的全链路提速。

完整readiness仅比较prepare＋validate：首次 **15386.618021秒**、热缓存 **10748.235919秒**，描述性比率1.431548×，耗时缩短约30.1%。冷07／08测量与assess另计287.165449秒，热流程未重测GPU性能。热流程仍约2.99小时；这是两个实际epoch的一次完整观察，不是重复实验置信区间或纯缓存因果效应。

生产parallel batch实际1205.510125→385.105672秒（描述性3.130336×），但含183次额外串行verify_precompiled重发射与审查的03整体1213.812379→1612.229583秒，完整热prepare也更长。没有把phase-minus-batch残差硬归因为单一组件。热183请求全部full-key命中，API0、CI2-only0、失败0、未尝试0；全部183普通数值／stream／capture重新执行，安全证明183项严格匹配cold完整key，新增sanitizer进程0。普通执行命令累计2938.115511→2946.662734秒，基本未降；cold548个实际sanitizer进程累计4934.026550秒，热复用验证本身仍计入04费用。

compile.json异构请求内层墙钟分布（p50／p95／max）：冷8.209715／109.208972／236.961158秒，热4.048884／18.790354／40.605286秒。完整流程采样树RSS峰值13,661,941,760→13,711,671,296字节，没有改善；native parallel batch峰值11,928,715,264→7,037,788,160字节。42份archive内嵌search total为1756.612973→1752.909199秒，嵌套于generate，不能再加入外层墙钟。生成和重复验证仍是主要剩余开销。

183项物理镜像／执行身份／driver资源事实一致，native-analysis完整内容一致；21×15选择及数学证书经验证的合法provenance重绑定后相同。完整证书字节不相同：cache receipt、image identity、correctness及仅cache_hit布尔值改变的resolved-resource记录各自重新绑定，再验证其派生SHA。每epoch证书独立通过验证，未知差异不获豁免。COLD＋WORK的11312文件mtime均早于热启动，原字节与pretiming SHA不变；最终read-only hook记录0写入。

证据：[完整费用与内存分析](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/warm-full-pipeline-acceptance-r1/efficiency.json)、[冷／热验收报告](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/warm-full-pipeline-acceptance-r1/report.md)、[completion.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/warm-full-pipeline-acceptance-r1/completion.json)，completion SHA256 `80cf7c8da4df27e7d058ee879c421c5ae344dea93775285dfcc69c24539328ce`。外围post-run审计r1／r2过严拒绝的13.559222／83.997905秒及r3通过的96.445419秒分别保留，不混入控制器active时间或抹去失败。一次性CPU测试、独立复算和历史封存审核也分别保留各自收据，未冒充单kernel部署费用。

### 已核验的CPU局部优化

严格连续Global f32地址codec采用整数行主序证明和定长描述符，保留不满足条件时的原路径。三个固定大程序的C++导出回放观察如下；材料化packet语义、完整TTGIR／配置保持等值：

|固定回放|此前秒|当前秒|
|---|---:|---:|
|diamond，J-c0|28.746|1.710|
|remat，J-c0|29.499|1.803|
|已观察mixed，LI-c1|32.725|5.521|

这是对应程序在不同时间的描述性CPU观察，不作为全域或GPU加速的统计证明。证据：[whole-export-equivalence-r1/result.json](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/compact-source-epoch-r2/whole-export-equivalence-r1/result.json)。整21组生成另观察到旧ed568的9853.117秒至当前4892.233秒；183个TTGIR／完整typed configuration等值，162个加权程序身份不变，但162份prepared_digest因连续存储证明reason文字改变而不同，归档不是逐字节相同。组件比例不可相乘为整个工作流或GPU收益，完整内存峰值仍以上表实际观测为准。

此前ed568中止运行、home-r2中止运行及全部失败／false-miss费用原位保留，作为开发／中止成本单列；不并入当前成功cold比较，也不删除来美化总投入。组件级CPU对照可证明对应输出等值的局部收益，但不同epoch／cohort的2.8×与3.4×不能相乘成系统收益。

## 5. 实现贡献、支持域与交付状态

本轮legacy旧模型ID为`sm90-layout-lex9-legacy-v2`；历史v4保留为明确标注的旧口径基线。新的20维静态合同固定为`ordered-surviving-network-source-issues-v1`，分值tick与native经验标度各按合同解释；跨模型贡献统一用相同算法／ABI下的GPU时间比检验。

本轮新增MinWork／有限三轴frontier、实际发射static20工作合同、同预算代码变体归档、compiled-resource条件选择、LOCAL重校准与完整身份链；同时优化typed proof复用、严格连续Global f32地址codec、CI2和四worker控制。GF(2)像核商空间、一般Gauss／Ship／fused网络及resource基础有旧阶段来源；static20是迁移工程代理，不是延迟重新拟合，b32定义数不是物理REG预测。完整归属、数学条件与固定Triton commit差异见 [实现与边界审查](/home/baopeihua/frisk/guoqiao/optimize_plan/network_selection_execution_artifacts/development/final-report-implementation-review-r1/report.md)，不在这里重写为通用创新或外部性能优越性宣称。

支持域仍有明确边界：专用shuffle要求同warp、完整32bit、相等R且R为≤512的2幂、完整affine owner重放；资源证书仅对共享独立非负加性模型成立。当前确有受限native H100 WGMMA（f16输入／f32累加）、cp.async／TMA路径，但WGMMA／cp.async立即wait，不是完整异步流水。多CTA只是独立固定tile，不含长序列Attention循环KV、跨CTA归约或persistent调度；正式性能结论严格为本轮C模式单CTA。与Triton／TileLang同算法、完整ABI、精度和调优预算的外部比较尚未完成。

本轮所选支持域的实现、执行、选择质量及费用验收已经完成。新留出bounded质量通过，性能晋升未通过，默认legacy保留。模型覆盖、已观察排序与保留池、搜索和重复验证开销的残余缺口如上保留。最终两根内容索引入口为`guoqiao/optimize_plan/network_selection_execution_artifacts/evidence-index.json`，同时覆盖HOMEBASE与数据盘任务根；源码、文档及原始数据以索引内快照为复核依据。外部公平比较尚未执行。
