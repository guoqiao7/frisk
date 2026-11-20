# 任务1：候选保留与成本决策联合修复

本报告记录 H100_direct 上的实现与开发验证。**所有旧 network-selection R3 程序均已观察，本轮包括新增候选在内仍是开发数据，不是新留出，不构成 Triton/TileLang 公平对比或性能晋升。默认生产策略保持 legacy。** 原始失败和未通过项保留。

## 实现

### 1. 归档按执行结构保留，预算不增加

新增 GF(2) 通信切分和逐操作所有权子空间视角，避免仅用 static20 总量支配删除不同执行结构。对一个子空间对，采用

\[
d(U,V)=\dim U+\dim V-2\dim(U\cap V)
       =2\operatorname{rank}[U,V]-\operatorname{rank}U-\operatorname{rank}V.
\]

保留矩阵的规范列空间、寄存器核维及每个转换边界的包含缺陷，缓存编码与子空间距离；不枚举硬件槽位。死的纯计算及未使用 nonvolatile load 不能人为制造多样性。完整程序身份不因总量特征相等而合并。

K4、总 probe 65536、refinement 256、网络变体的一个预留评分及 local/joint/static pins 保持。活跃 reservoir 上限32，加一个 incoming。旧 `legacy-hamming-v2` 和失败的 `paid-static-feature-frontier-v1` 消融保留；强制全域结构归档为 `paid-feature-frontier-v1`。 这里legacy是归档保留规则，路由不把共同static20成本合同改成另一模型；严格旧LOCAL基线仍在同一保留池上计算。生产默认成本模型 `sm90-layout-lex9-legacy-v2` 与研究中的显式network/static20配置分开记录。

exp128×64 的旧 LI c1 已在 J 搜索 probe1810/refinement36 付费观察，但原归档删除了它。结构策略将其完整 assignment `a932923e…` 恢复到 slot3。J 的 exp/mixed/div 实际 probes 为3368/6844/13043，refinement84/105/162，新旧计数完全相同；LI 对应2088/4240/8091及66/78/129，也完全相同。LI保留本地pin，无额外J搜索。

### 2. 完整静态摘要、单位和支持域

旧 native14 的108 available、75 unsupported 状态不改。新增独立的完整静态摘要：183/183，其中162 scalar acyclic call-expanded summaries、21当前Attention summaries。它们按可达 CFG/callee/返回/同步条件校验，不能把跨单位摘要直接塞进native14价格；TMA polling、动态predicate/slowpath次数等仍不被声称有时间上界。

为144个scalar J/LI程序补 GF(2) 寄存器窗口局部性：投影像空间秩和商空间陪集统计 warp/CTA 的32B、128B、4KiB几何地址块足迹。全域采用3,836,280次image-rank与45,360次quotient-coset窗口计算，投影标签枚举0。4KiB不是实际TLB页粒度，relative-base-zero足迹不是硬件miss数。

144份源地址受控干预证据形成71组，其中15个非平凡组含40个候选：只删除纯、exclusive Global地址DU切片，保留非地址/数据/predicate/通信，再锁定全native opcode/reg/CFG/callee和driver资源。它不证明native DU、动态mask、slowpath次数或cache行为相同。

### 3. 用精确有理排序锥处理成本不确定性

GF(2)用于位映射，成本排序在整数/有理数上处理，两者不能混淆。对已登记、差异至少2%且配对CI排除1的训练排序，令

\[
A_i=f(\mathrm{fast})-f(\mathrm{slow}),\qquad A_iw\le-1,\quad w\ge0.
\]

先给出严格可行的非负有理见证，拒绝空锥或矛盾数据。目标差分若有

\[
d=\sum_i\lambda_i A_i-\mu,\quad\lambda_i,\mu\ge0,
\]

则在该校准模型族内有弱界；若 `sum(lambda)>0`，得到严格改善。所有等式用 Fraction 独立复核，浮点NNLS只提出候选证书。共同工作或公共成本offset在差分中消去，不影响此证明。

优先证明池内argmin；若第三个候选不可辨识，也允许选择**严格改善原fallback**的方案，固定priority消歧，结果明确 `argmin_proved=False` 并列出未解决对手。既不把“无法证明全局最优”误作“不能改进”，也不把保守改进宣称为最优。

native注册163条material非零排序，跨程序留出122条可证明；这不是全部234个不同execution pair。58个零contrast对中52个同CUBIN、6个异CUBIN，均没有遗漏的≥2%且CI排1约束；合成material-zero反例会正确触发矛盾。地址边际层8条注册排序、6条跨程序证明，也不是全部82个可比对。

scalar全图家族存在两行精确Farkas矛盾；Attention没有OOF接受支持；全局NNLS的接受支持也未通过。这些路径继续fallback，未通过项不因静态摘要完整而消失。单CTA、128线程、精度、GPU等执行域外也回退。删局部性轴需要精确非负有理组合证书，并在实际池复验，不只凭矩阵秩删列。

## 开发实测和发现

### Exp：保留与决策两步均起作用

R16只使用旧R3数据建模。在旧42池中，LI exp128从c0改到c1，其余41池不变。新J K4恢复该方案后，严格改进规则从c0选c3；c3对c2仍不可辨识，两组满足全部训练约束的有理权重能给出相反排序，已保留证据。

选择先冻结再测量：3个固定seed × 输入偏移1/2/3 × 输出偏移1/2/3 × 4次平衡重复，每形状108完整4臂block，三组合计324 block/1296条候选记录；每block的5次event先取中位数，bootstrap按完整block在27cell内重采，不能把event当独立样本。

exp固定c3/c0时间比 **0.9035712**，95%区间 **[0.9028070,0.9043229]**，三比较familywise区间 **[0.9026419,0.9044830]**，约降低9.64%／1.1067×。c3也是当前K4的最佳固定arm。该结论仅限这些已知程序与固定输入/ABI条件，不是未见shape或外部编译器性能保证。

### Div：原“统一反选”判断需修正

旧9批把seed和指针偏移混在一起；逐批oracle regret不能证明固定c2显著更快。独立固定输入×输入/输出偏移交叉测量108对后，c0/c2总体几何比0.996746，描述性95%区间[0.996285,0.997212]；输入偏移1/2时c2快，偏移3时c0快，三个seed方向一致。新K4中c0与表观点最佳c1是同CUBIN/同ABI，0.097%点差不能认作布局损失。

两个arm的32/128B低位相同，但4KiB低位全不相同；独立buffer和高位地址/cache映射仍有混杂，不能归因TLB或cache单一机制。新地址数据没有用来重拟合本轮模型。

### Mixed：强制全域GF2归档未通过保留性

新K4内c0确为最佳，但旧系统 `joint-affine-frame` 所选c2（assignment7ab60eae…，CUBINc741bc70…）已掉出新池。因此c0/c0=1不能证明对旧系统无退化。

额外固定旧/新策略诊断采用已验收的旧c2与新c0，108完整pair、27cell×4、每cell2AB2BA。新/旧比 **2.5272122**，95%区间 **[2.5258527,2.5285774]**；约308.89/122.23µs，慢152.72%，3seed、27cell均退化。这是额外池外基线诊断，不是同调优预算公平比较；R16模型和规则没有据此调参。这个失败必须保留，不能用Exp收益掩盖。

### 保守路由修复已知 Mixed 退化

新增显式实验recipe `--archive-retention=calibrated-source-grammar-v1`，生产默认仍为legacy，R16全域GF2失败消融仍可重现。搜索前只解析一次完整源程序：受检单kernel、静态f32/rank2、精确Global存储/view、闭合的纯load/mul/sub/exp/constant DAG、唯一最终store和完整候选域通过时才用GF2结构归档；DIV、未知math（即使无DIV）、call/control/MMA/async/reduce/f16/dynamic/未知encoding、额外region/kernel或alias不符均完整沿用旧归档。

这只证明**源grammar eligibility**，明确 `native_output_guaranteed=false`；没有声称ptxas必保持native15。编译后整个池的实际native/schema/domain和风险检查仍在。条件按实际IR、类型、region、alias与全部候选域判断，没有graph名或具体shape特判，也不以小R门槛误拒R8192 demand carrier。

65项C++和30项严格archive/route测试通过。12次J/LI CPU归档逐条完整entry等值：Exp实际GF2分支保持R16池，Mixed/Div实际legacy-v2分支恢复旧default池。旧Mixed c2已回到K4；预算、local/J轨迹、pins和min-work变体证明均保持。gate前没有额外搜索或额外评分。实际v2/v3格式由effective策略决定，requested/effective与单次resolver事实独立记录，未知路径不能夹带虚假v3评分声明。

J三组12项已与R2或旧R3的完整TTGIR和typed execution config严格匹配。此前全域GF2的2.527×失败没有改写成通过；新route是受限域采用加旧路径保留，仍需真正独立性能留出。Exp LI完整K4已单独完成新增GPU验收，两个此前缺少镜像的可选项不再留空。


## GPU、源码与证据

J的12个新K4请求全部完成编译、1260项普通/stream/capture数值检查及48个实际sanitizer进程；另对Exp LI完整4请求补420项检查与16个实际sanitizer进程，新增两项LI可选程序已闭环。合计16次候选验证请求、1680项检查和64个成功sanitizer进程；请求可能包含严格相同镜像，不能称16个独特内核。R2原manifest、镜像和已验收字节保持。R16仅两个主机决策文件相对GPU epoch变化，做了明确source/TTGIR/config/image桥接；没有重绑定生产cache身份。首次r1在sanitizer前置环境校验失败的记录原样保留；修复只允许原冻结sanitizer prefix，源码/工具/immutable/protocol检查没有放宽。

完整旧network封存审计再次通过29,836文件、24,919,125,134字节和1,980目录；更早7个index SHA也不变。R16测量源、R2 GPU源及对应工具已分别保存，不能混写为同一次最终源码运行。

证据入口：

- `/home/baopeihua/frisk/guoqiao/optimize_plan/selection_refinement_artifacts/development`
- `/data0/baopeihua-frisk-selection-refinement-20261105`
- R16配置与固定决定：`/home/baopeihua/frisk/guoqiao/optimize_plan/selection_refinement_artifacts/development/known-k4-decision-freeze-r16.json`
- 新K4实测：`/home/baopeihua/frisk/guoqiao/optimize_plan/selection_refinement_artifacts/development/known-k4-performance-acceptance-r16-r1`
- mixed池外对照：`/home/baopeihua/frisk/guoqiao/optimize_plan/selection_refinement_artifacts/development/mixed-old-policy-external-acceptance-r16-r1`
- 严格审查：`/home/baopeihua/frisk/guoqiao/optimize_plan/selection_refinement_artifacts/development/joint-artifact-independent-review-r3`、`joint-statistics-independent-review-r4`、`mixed-external-independent-review-r5`

## 最终有限域验收与后续边界

|程序|最终归档路由|J选择|充分知情LI选择|
|---|---|---|---|
|Exp128×64|源域合格，GF2结构K4|c3|c1|
|Mixed512×64|完整legacy K4|c2|c0|
|Div512×128|完整legacy K4|c0|c0|

最终六池都通过严格旧native/LOCAL基线加载、完整程序与已验镜像桥接和无性能标签推断。Exp LI c1对三个对手有完整条件argmin证书；其完整program、CUBIN、TTGIR、typedcfg和物理ABI与J c3相同，**这里没有J胜过充分知情LI的内核性能证据**。Div的J/LI也为同镜像。Mixed保持旧系统原有J/LI路径，本轮没有把新增J优势归给保守路由。

最终CPU验证为703项基础布局回归（最终二进制与实际跑过的703项逐字节相同）、65项selection C++、245项相关Python测试；额外独立数学/原始记录审查另有凭据，不混成独特测试总数。新J/LI来源mode绑定，执行载体的固定assignment replay仍为J，不能据此伪称LI做了joint搜索。

**本次任务1的受限源域实现、已知失败止损及开发验收完成；不等于全域模型和保留策略已解决，更不是性能晋升。** 强制全域GF2仍有2.527×Mixed失败，故保留为消融；新source gate只是eligibility而非native保持定理，未知源域完整用旧归档，编译后模型检查仍必须通过。Scalar统一成本族的矛盾、Attention无OOF接受支持、全局NNLS泛化接受未通过继续显式fallback。新形状、新输入分布、实际规模Attention或整卡吞吐不在本轮性能结论内。

下一阶段是任务2的搜索/规划/证明/发射/编译/验证开销收敛；本轮没有实现新的持久cache工程。随后需要在明确支持域、算法/ABI/精度/总调优预算和最终源码上，用真正新留出重新验收，再决定正式冻结及Triton/TileLang外部公平比较。默认legacy未晋升。

代码及证据均在H100_direct，未建立本地代码副本，本轮未另作git commit。

