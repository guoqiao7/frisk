# Frisk GF(2) Shuffle 网络优化与验收报告

日期：2026-11-04。代码、构建和实验全部在 H100_direct 的 `/home/baopeihua/frisk`；未建立本地代码副本。最终源码 recipe 为 `987b5a462e0bf4d85bf9640290539e8f78e3533e0fc0f4b8ce07c82586a5a9a6`，446 个源码文件。代码未提交，布局选择默认仍为 legacy；Shuffle lowering 默认采用新的 auto 网络。

## 结论与适用范围

本轮已完成用户要求的 Shuffle 次数膨胀优化。对于已观察的 mixed-1024×64 中 J0／L1 两个完整布局赋值，SHFL 在源代码与原生 SASS 中均下降约33.03倍；相同生成代码的原生编译 API 分别降低3.81倍／5.10倍；最终源码下重新测得内核加速约2.09倍／2.02倍。全部功能、安全、成本一致性和本轮运行门槛通过。

这是同算法、完整赋值、ABI、精度与线程组织的 lowering 消融，不是新留出集的布局选择质量验收，也不是与 Triton／TileLang 的公平性能比较。原先封存的选择质量结论仍属于其原源码版本，不自动延伸到本轮。综合记录 `release-qualification.json` 状态为 `IMPLEMENTED_AND_VALIDATED_SHUFFLE_LOWERING_WITH_DECLARED_LIMITS`。

证据根为 `/home/baopeihua/frisk/guoqiao/optimize_plan/gf2_shuffle_network_artifacts`；下文相对路径均相对此目录。

## 问题定位和目标实例

目标 shape 是1024×64，即65536个逻辑元素；当前最小 GPU 闭环使用一个 CTA、4个warp／128线程。相关转换每线程512个逻辑槽位，不等于512个物理寄存器。J0 指 joint-c0，L1 指 local-c1；这里 local 是局部搜索归档标签，与 LOCAL 内存不是一个概念。

原实现对每个目标槽位取回32个可能的源寄存器，再用谓词搬运选择结果。一条路线展开512×32＝16384条 SHFL 与512×31＝15872次逻辑选择。L1有三条路线，所以累计49152条 SHFL／47616次选择。每路线的工作cap不能限制全kernel代码大小或PTXAS复杂度。已有稀疏支持集排除了无关来源，但没有合成可共享中间结果的完整重排网络。

## 论文与 Triton 参考，以及 Frisk 实现

阅读了用户文件 `E:\Desktop\paper\Linear_layout.pdf` 第5.4节（PDF第8—10页，第9页完整视觉核查），SHA `c6eaa53ed2c1badfb1c6680e5eb3140bc0b3bd26c7efbf85d56c8836e7be4e4f`；对应 *Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2*，arXiv2505.23819v5，2026-03-06。

参考原理是先组织线程内寄存器重排，再交换确定的数据，避免每个接收目标枚举所有来源。实际安装 Triton 为3.4.0，官方tag对应 `c817b9b63d40ead1ed023b7663f5ea14f676f4bc`；这不是声称wheel二进制由该commit精确构建。该版使用前后lane-dependent寄存器重排，并对复杂交叉情形保守回退。另研究当前main `1887644bdea84fde07825813158782fa526871fd` 的寄存器／lane位排列、Ship与合并Swap。其常规路径策略仍偏保守，不能把本轮实验说成对该框架的性能胜出。

Primary code references:

- [Triton3.4分解](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/lib/Analysis/Utility.cpp#L211)
- [Triton3.4发射](https://github.com/triton-lang/triton/blob/c817b9b63d40ead1ed023b7663f5ea14f676f4bc/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp#L352)
- [Triton main分解](https://github.com/triton-lang/triton/blob/1887644bdea84fde07825813158782fa526871fd/lib/Analysis/Utility.cpp#L400)
- [Triton main Ship／Swap](https://github.com/triton-lang/triton/blob/1887644bdea84fde07825813158782fa526871fd/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp#L243)

实现独立编写，没有复制 Triton 源码，也没有引入其布局选择依赖。现有 Triton3.4仅作为底层编译基础设施。Frisk除位排列专用分解外，还对任意已证明可逆的稠密 GF(2) holder 映射实现 Gauss–Jordan／CNOT 剪切网络，保留一般映射的表达能力；不宣称这些基础线代算法是新数学发明。

### 完整映射证明与网络

将warp内holder写成 x＝r＋R·lane，拟合 f_w(x)＝A·x xor c_w。先逐holder核验真实canonical owner函数、同warp、完整仿射性和A的可逆性；不能只因某个支持集是GF(2)像就假定整个函数线性。当前执行支持32-bit f32字、128线程、等长且为2幂的NR／DR≤512。不能满足证明或不够优的映射保留原exact-support路径。

位排列分解选择不相交的寄存器位／lane位交换S，使B＝S·A成为分块排列，A＝S·B。寄存器排列只重命名，不发指令。Ship用一次xor-SHFL和三次selp处理一个寄存器对；合并Swap利用C·D·C剪切恒等式合并通信阶段，并折入剩余lane排列。

当lane残余为恒等、R个32-bit字、m个混合位交换时：

| 方案 | 静态SHFL | 逻辑选择／selp |
|---|---:|---:|
| 原逐目标枚举 | R·2^m | R·(2^m−1) |
| 逐位Ship | mR/2 | 3mR/2 |
| 合并Swap | (1−2^-m)R | 2mR |

实际每条路线R512／m5：原16384 SHFL；Ship1280；合并Swap496。三条L1路线选择合并方案，总计1488。旧选择使用setp＋谓词mov，新方案使用selp；不能因旧PTX字面selp计数为0而称旧实现没有选择。

一般可逆GF(2)映射通过基本行变换生成剪切／交换。寄存器内变换可静态重命名，register←lane用条件寄存器对交换，lane←register对静态控制位为1的寄存器发xor-SHFL，lane←lane用核验后的lane变换。pullback发射必须采用消元正序；反序会错，已有非交换反例测试。所有网络都用唯一holder ID在整个有限域模拟。

### 双重不增限制

早期版本只限制总静态work，发现纯静态反例：R8，columns `[1,38,226,7,21,249,120,225]`，offset0。原路径64SHFL／456work；Gauss网络92SHFL／392work。数值正确但通信反而增加。

最终同时要求 `total_work <= legacy_work` 且 `SHFL <= legacy_SHFL`，再按SHFL、total_work、kind确定性选择。不满足则回退。513组C++／Python完整计划差分为0，348接受／165回退；所有接受方案满足双约束。反例当前正确回退，未根据GPU性能标签调规则。

实际四条路线在固定独立32-bit transport接口、仅允许Register＋SHFL、不跨route重计算或用Shared替代等条件下，每lane需要496个独立远端word、本地16个；每warp动态SHFL至少496。当前无条件直线网络达到该受限通信界。循环可以减少静态文本，故这不证明任意算法静态指令数最低，也不证明全kernel延迟、选择数、编译时间最优。

## 集成、成本与校准边界

Python规划器：`benchmark/layout/execution/shuffle_network.py`；发射接入 `backend.py`，实际PTX计数在发射时与计划逐项断言。C++独立规划器 `ShuffleNetwork.h/.cpp` 接入转换导出及新ordered demand计分，Python重新计算整个导出声明并拒绝篡改。`demand_coverage_audit.py` 从实际发射语法独立计数，比较C++边界、访问与surviving特征。历史冻结v4语义仍明确保留为旧语义，不能把它称新实现成本。

寄存器排列是别名；相同谓词在单网络内复用；单bit谓词2条、parity谓词4条指令；identity lane-xor不发通信。真实每条合并路线为496SHFL＋5120selp＋10条谓词设置，总5626work，原59904。所有Shuffle均无条件执行、全warp收敛，运输中不执行浮点算术。

新增规划器已进入exact native cache的实现身份。生产cache仍绑定完整源码、生成代码／config、工具和环境，未为本实验放宽源身份。

旧Shared-bank校准兼容校验最初拒绝新AST，现使用严格两阶段机械投影：先绑定上轮封存8f3中间源码，核对已审查transport函数的完整AST及Shared直接分支，再沿用已有稀疏／offset memo证明。规划器文件、锚点index和源码全部加入依赖。27项严格测试及独立攻击检查通过。最终活动视图在 `development/native-profile-views-r2`，14个系数原样，LOCAL传输系数仍6418。这仅证明Shared-bank特征兼容，没有重新拟合成本模型或验收其绝对外推／全局选择质量。

## 编译开销：测量边界与结果

原编译实验源码为 `e2ab47ed5de77ad4c5d0e09dc2d99586418d0720f11f4955e9e6421999e06616`，完整446文件、5个二进制及archive保存在 `development/source-freeze-r1`。固定同一J0／L1完整assignment，两策略各一轮，顺序J0 legacy→auto、L1 auto→legacy；CPU8..11，私有初始空Triton cache，导入前设置ALWAYS_COMPILE，真实PTXAS调用已观测。编译不初始化CUDA，不导入Torch或加载driver库。

最终增加双guard后源码为987b5a，保存于 `development/source-freeze-r2`。对四程序重新完整发射，TTGIR逐字和完整序列化JSON对象全部一致；桥 `development/final-source-emission-equivalence-r1/summary.json` SHA `da9d9ff9da90f16263bab1468bdc61806b9a7e9f9f8644c72deb717da96adfc7`。因此这些原生编译输入和后续加载CUBIN仍对应最终生成程序；原e2ab的emit／phase计时保留原版本范围，不能冒称最终源码整条pipeline已重新计时。生产cache上下文未被重写。

| 用例 | 原生编译API old→new | 倍率 | PTXAS等待 old→new | 倍率 |
|---|---:|---:|---:|---:|
| J0 | 122.621→32.179s | 3.811× | 121.923→31.605s | 3.858× |
| L1 | 559.441→109.704s | 5.100× | 558.437→109.130s | 5.117× |

两例API合计682.061→141.882s，4.807×；实际PTXAS等待680.360→140.736s，4.834×。原epoch的解析＋emit＋API＋物化phase合计730.096→190.228s，3.838×，其中J0 149.357→59.274、L1 580.739→130.954s。imports、完整身份检查、父进程和监测等另列，不是完整应用端到端时间。

emit合计44.085→44.428s，基本未改善；大包规范化、验证和发射仍有20—25s量级成本。改进主要发生在已实测的PTXAS，而不是凭代码大小推测。PTXAS指标是透明包装原subprocess.run测得的等待wall，不是CPU时间；API减该等待只是其余未细分费用，不能称LLVM独占时间。

本CPU数据为单次描述，无CI。184次CPU监测无亲和违规，固定次序，内存／IO／OS page cache不完全隔离。空私有cache＋force＋真实PTXAS表明本次发生了原生汇编工作，不能声称整个操作系统冷启动。历史605s来自另一整请求边界，不能直接与新API109.7s相除作为匹配加速。

## 原生机器码与资源

| 图像 | SASS SHFL old→new | ELF执行段bytes old→new | REG/thread old→new | driver LOCAL frame/thread old→new |
|---|---:|---:|---:|---:|
| J0 | 16384→496 | 2546816→1329024 | 32→255 | 19032→2672B |
| L1 | 49152→1488 | 4519680→1889408 | 32→32 | 58832→8496B |

源级通信下降在实际SASS中精确保留。执行段按ELF SHF_EXECINSTR统计，没有把重叠函数长度相加，也未用含debug的整个CUBIN大小替代。

J0寄存器增加，理论资源驻留上限由1.0变0.125；L1仍1.0。这是资源上限，不是Ncu实测occupancy；本实验grid只有一个CTA，不证明多CTA吞吐。不能概括为“所有资源下降”。raw cuobjdump LOCAL四图像均0，但STACK和driver frame非零；frame不是动态LOCAL流量，也不能推spill数量，spill_count保持unknown。没有从静态代码推ICache瓶颈。

## 最终GPU时间：预注册配对实验

所有GPU操作只在GPU1 UUID `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618`，GPU0外部作业未停止。四CUBIN／输入／ABI／工具／源码先绑定，数值和四工具安全门禁通过后才计时。最终GPU worker为 `shuffle_full_gpu_compare_r2.py`；环境显式CUDA_MODULE_LOADING=LAZY、CUDA_VISIBLE_DEVICES未设置（null）。

这两例是同一逻辑工作负载的不同完整布局实现，几何平均只汇总这两份实现。

每个程序20个完整paired batch，每image block5个event；每interval回放32节点graph，并实际cuGraphGetNodes验证。批内策略顺序精确平衡，合计40pairs／80blocks／400intervals／12800个正式kernel节点。无invalid、replacement或边界foreign PID。每策略先warmup10launch＋2graph回放，另计。

统计预先注册：median5作为block时间，20个auto/legacy配对比率再取median；10000次整pair bootstrap，95%percentile CI；不能把五个event当五个独立batch。以下延迟是各100个interval边际中位数，paired ratio不必等于两列中位数之比。

| 程序 | old中位数 µs | new中位数 µs | paired new/old，95%CI | 加速，95%CI |
|---|---:|---:|---|---|
| J0 | 1408.369 | 672.406 | 0.477383 [0.477159,0.477677] | 2.09475× [2.09346,2.09574] |
| L1 | 2263.566 | 1118.128 | 0.494238 [0.493744,0.495041] | 2.02332× [2.02003,2.02534] |

两例几何平均时间比0.485738，CI[0.485459,0.486181]，加速2.05872×，CI[2.05685,2.05991]。每例lower-latency和≤1.05回归门槛均过。区间仅刻画这两个固定程序在本次实验内的变化，不是程序总体CI或跨运行／跨GPU保证；没有familywise覆盖声明。

独立raw-first统计先于root第二份计算，全部30542项数值对照差0，两个bootstrap索引hash、四个布尔准入和完整资源对象相同。

## 验收覆盖

- 最终689项C++单测、53项选择测试、80项IR测试与15组CTest全部通过；49项针对性检查是子集，不能再累加为不同用例。
- 513组C++／Python完整计划JSON差分为0；另有实际PTX解释及大边界全holder验证。目标四路线及两种备选网络均验证过完整索引。
- 最终11种原生primitive：330条常规记录＋264条四工具安全记录，包含随机uint32、13种IEEE位模式、offset1/2/3、复制后备、稠密affine warp offset、非默认流及实际3节点capture。没有遍历全部2^32数值。
- 真实四图像：420条正常记录和48条四工具安全记录；正常210个old/new配对及安全24个配对全部逐位一致，既有精度阈值未放宽，输入／canary／不别名检查均过。
- 全量正常共750条，安全共312条；两套实验各自的spec、ABI和scope单独保存。

## 保留的失败与更正

初期15组CTest有两组失败：旧精确compat白名单拒绝新网络及selection bench未刷新source manifest；已通过机械证明和重建修复。双guard后一次Python集合在ninja链接过程中遇到暂时PermissionError，源不是权限修复，最终完全链接后15组CTest全过。原失败日志未删除。

CPU r1 controller重复timestamp字段，worker又遇到Python int-key config与JSON string-key表示差异，均在native／GPU之前失败；完整序列化对象证明无字段差异后才修r2，未筛掉字段。原worker自行退出，未实际kill外部或自有作业。后处理一度错误要求sm_90拼写，核对真实Triton sm_90a后更正断言，未重跑数据。

首轮真实GPU normal已完成数值launch，但Torch初始化将环境CUDA_MODULE_LOADING从null设LAZY，结束门禁拒绝，未产生性能样本。诊断确认唯一该项漂移。保留原worker／注册／失败，另用只有5个输出目录字面变化的新worker与显式LAZY重新注册，参数／统计／阈值没有变化。

独立stats原字段all_resources_decreased采用≤语义，实际False；解释为“资源全部非增加”也为False，不能概括成严格下降，未据此新加准入规则。

早期compat临时测试目录曾位于旧stage，退出后已清理；BASE现用于新stage。最终对五套历史封存53609文件／45080249482字节和三个外部报告逐文件hash、精确文件集全过，不声称从未创建过临时路径。

## 当前边界与后续冻结对比

本轮交付是网络lowering及其成本／执行闭环。对不能证明或不够优的映射保留旧路径；未优化test-only直接LLVM参考pass。尚未新增subword packing、跨warp网络、非双射quotient算法，也未解决全部发射／验证时间。

新的全局搜索／模型选择质量应在新source和新的性能留出上复验后再冻结外部公平对比。未重新验收绝对时间外推、跨GPU效果、多CTA吞吐或与Triton／TileLang的相对性能。本报告不把旧1.51×布局选择结果叠加到本轮2.06× lowering结果。

最终source freeze位于 `development/source-freeze-r2/freeze.json`，SHA `1cb947cad2c0e01e3239a6f009481cac99b157bdc91abe187319bccb11359698`。CPU summary SHA `e8e61033d589f15d81afc4bff14bc3ee75f0f5de528382d41e2700553b76b8d3`；独立GPU结果 `502bef80506976dc22551fe1331f415a92c0a49d7fe02f5f010af5843ccb8015`；统计对照 `f92a8e3cf0db9148ce9f580f399467874566fa71a1f9d9aa2e3fd90eb1a33e39`；完整组件见 `release-qualification.json`。最终 `evidence-index.json` 将包含全部新stage文件及本报告的外部只读引用。

Git HEAD仍为 `9f79155ac08f82133f4ddb6554af552feefca8d2`，index SHA仍为 `4602fb66327ad70bb8f959bd766b712cf0ca8daa3bde2d72c037613dc745aa92`。未创建提交。
