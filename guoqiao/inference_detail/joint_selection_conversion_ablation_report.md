# 联合选择及转换消融：当前统一报告

更新：2026-10-30T07:56:41.003164+00:00

本主文只描述当前接受证据。旧实现修复、失败运行与阶段记录完整保存在文末历史附录；历史段落不作为当前结论。

## 当前状态

Gate1–3 的四臂、共同域、固定 assignment 回放、精确 cubin 数值和安全证据已完成独立复核。Gate4 当前完成源冻结、完整回归、64份 host 字节一致性桥、28条隔离冷编译和实际机器资源记录。正式配对测量结果如下，最终独立 whole review 状态需结合索引中的最新审查记录。

## 因果边界与身份

四臂为 L-C、J-C、L-O、J-O。L 的第一阶段保留完整硬合法性、只计 producer 范围并用 producer 稳定 ID 破平局；第二阶段固定 producer encoding 与指令 geometry 后补全。J 一次优化全图。L/J 包含顺序及共享预算分配效应。C 独立必要转换；O 为共享、有限 remat 与 post-opt 组合。默认 exact=256、beam=32、总 probes=65536、refinement=256；L 两阶段共用总额且第一阶段至多一半。未调权重、ABI、算法或图以迎合结果。

主机制图使用128线程、[64,16] f32、topology[8,32,4,1,1]、受控A/B候选、固定四根弱 vector ABI。自然 Attention 弱 vector 与强 cp.async/TMA 分层呈现，禁止跨 ABI 算收益。每主图至少两份完整可执行 assignment 的数值/安全证据见 Gate2 最终 artifacts；每实际 cubin 按自身 raw hash 和 ABI 对应 safety，不以归一化机器身份替代安全证据。

共同传播前候选、稳定ID、关系证明和支持状态由一次 preparation 冻结再克隆。L2 是同一共同图加 producer/instruction pins 的子域，common_digest 与 restriction_digest 分列。仅 complete+exhausted oracle 可作相应有限域最优证明；L2 最优不冒充 unrestricted 全图最优。oracle 上限65536节点/4096完整评分/120秒；不以 oracle 替代默认 solver 返回。

接受生成 corpus：`run-task3-20261030T053924Z-574ed1d8/manifest.json`。当前 host bridge64 assignment/export 不变；strong O1 supplement 覆盖输入0/128/256、输出1/2/3相位组合，详见 `task-3-final-index.json`。Gate4 只新增测量/统计、host chrono 诊断与旧 report schema 兼容；未改生成数学。历史生成源码身份、当前测量源码、历史/当前 tool hash 分署，不覆写接受 manifest。

当前测量源码冻结：`task-4-source-freeze.json`；全部64 assignment及已有 selected/materialized/output/execution 文件逐字节相同见 `task-4-host-parity.json`。

## 选择与机制结果

无冲突、conflict、diamond 的 L/J assignment 不变。remat 的 L/J 差异属于 producer-local 平局消除；不得表述为严格 producer-local 优劣反转。weak Attention 的 L/J 差异来自 producer Copy 自己的 access(bank/global)成本被 producer-only scope 排除，是受限 scope 效应，不能归因于跨 consumer 转换。strong Attention 若 producer变化而完整 cost相等，记 cost-neutral change。详见 `task-3-selection-classification.json` 和两份 access attribution JSON。

固定 assignment 的 independent→shared→remat→post 回放显示真实机制变化；conflict/diamond 可减少重复边界，remat 白名单 slice 有真实物化及后优化证据。四臂只给出转换优化组合效果，不隔离共享/remat/post各自性能贡献。完整分步 IR、route、资源与精确执行在 `task-3-mechanism-summary.json` 和36份 replay artifacts。Register、普通 Shuffle、receiver-dependent Shuffle 与 Shared 的独立正确性控制保留 Gate2 证据。

| case / ABI | arm | full lex9 cost | total probes / refinement | materialized→output converts | raw cubin |
|---|---|---|---|---|---|
| no-conflict / weak-vector | L-C | [128, 4112, 0, 0, 32, 0, 0, 0, 0] | 74 / 0 | 0→0 | feb78ebe3b88 |
| no-conflict / weak-vector | L-O | [128, 4112, 0, 0, 32, 0, 0, 0, 0] | 74 / 0 | 0→0 | 82514b762858 |
| no-conflict / weak-vector | J-C | [128, 4112, 0, 0, 32, 0, 0, 0, 0] | 41 / 0 | 0→0 | feb78ebe3b88 |
| no-conflict / weak-vector | J-O | [128, 4112, 0, 0, 32, 0, 0, 0, 0] | 41 / 0 | 0→0 | 82514b762858 |
| conflict / weak-vector | L-C | [128, 4112, 0, 27648, 64, 0, 0, 3, 0] | 154 / 0 | 3→3 | da853064a5e0 |
| conflict / weak-vector | L-O | [128, 4112, 0, 17408, 80, 0, 0, 3, 0] | 154 / 0 | 1→0 | 10be71faae3c |
| conflict / weak-vector | J-C | [128, 4112, 0, 27648, 64, 0, 0, 3, 0] | 83 / 0 | 3→3 | da853064a5e0 |
| conflict / weak-vector | J-O | [128, 4112, 0, 17408, 80, 0, 0, 3, 0] | 83 / 0 | 1→0 | 10be71faae3c |
| diamond / weak-vector | L-C | [128, 4112, 0, 27648, 56, 0, 0, 3, 0] | 131 / 0 | 3→3 | 7eec88d8d83d |
| diamond / weak-vector | L-O | [128, 4112, 0, 13312, 64, 0, 0, 2, 0] | 131 / 0 | 1→0 | 893b8fdb28a7 |
| diamond / weak-vector | J-C | [128, 4112, 0, 27648, 56, 0, 0, 3, 0] | 71 / 0 | 3→3 | 7eec88d8d83d |
| diamond / weak-vector | J-O | [128, 4112, 0, 13312, 64, 0, 0, 2, 0] | 71 / 0 | 1→0 | 893b8fdb28a7 |
| remat / weak-vector | L-C | [128, 4236, 0, 46080, 64, 0, 0, 5, 0] | 154 / 0 | 5→5 | b7b4d2da033e |
| remat / weak-vector | L-O | [128, 4236, 0, 22528, 72, 0, 0, 3, 0] | 154 / 0 | 2→2 | e74f731e4b88 |
| remat / weak-vector | J-C | [128, 4236, 0, 27648, 64, 0, 0, 3, 0] | 83 / 0 | 3→3 | d3defbe18bfa |
| remat / weak-vector | J-O | [128, 4236, 0, 18432, 64, 0, 0, 2, 0] | 83 / 0 | 2→2 | 9d1dd98bafdb |
| attention / weak-vector | L-C | [8293, 2144, 288, 0, 428, 6144, 6, 0, 0] | 2793 / 0 | 0→0 | deb7e4af4db4 |
| attention / weak-vector | L-O | [8293, 2144, 288, 0, 428, 6144, 6, 0, 0] | 2793 / 0 | 0→0 | 8939ef277bf4 |
| attention / weak-vector | J-C | [8293, 800, 0, 0, 428, 6144, 6, 0, 0] | 1838 / 0 | 0→0 | a9400963f4e3 |
| attention / weak-vector | J-O | [8293, 800, 0, 0, 428, 6144, 6, 0, 0] | 1838 / 0 | 0→0 | 97c2befb0afb |
| attention / strong-cpasync | L-C | [8119, 704, 48, 0, 428, 6144, 6, 0, 0] | 1910 / 0 | 0→0 | c74cb1dea28d |
| attention / strong-cpasync | L-O | [8119, 704, 48, 0, 428, 6144, 6, 0, 0] | 1910 / 0 | 0→0 | 89dc84025525 |
| attention / strong-cpasync | J-C | [8119, 704, 48, 0, 428, 6144, 6, 0, 0] | 955 / 0 | 0→0 | c74cb1dea28d |
| attention / strong-cpasync | J-O | [8119, 704, 48, 0, 428, 6144, 6, 0, 0] | 955 / 0 | 0→0 | 89dc84025525 |
| attention / strong-tma | L-C | [8119, 704, 0, 0, 428, 6144, 6, 0, 0] | 1642 / 0 | 0→0 | 9eabe7154c87 |
| attention / strong-tma | L-O | [8119, 704, 0, 0, 428, 6144, 6, 0, 0] | 1642 / 0 | 0→0 | 2ca9e32e6cce |
| attention / strong-tma | J-C | [8119, 704, 0, 0, 428, 6144, 6, 0, 0] | 821 / 0 | 0→0 | 9eabe7154c87 |
| attention / strong-tma | J-O | [8119, 704, 0, 0, 428, 6144, 6, 0, 0] | 821 / 0 | 0→0 | 2ca9e32e6cce |

### L/J 分类与首次完整成本差分

九分量顺序为：instructionPathAndWork, memoryTransactions, bankConflictDegree, conversionBytesAndSync, spillRiskAndRegisters, sharedBytesAndOccupancy, replication, codeSize, deterministicTieBreak。以下分量索引为零起始cost[i]，只比较相同policy的最终完整图lex9成本，不将producer-local成本与full成本相减，也不对混合单位向量算百分比。无strict reversal或consumer-only主矩阵见证时如实标无；producer变化并不自动意味着严格反转。

| case / ABI | policy | classification | 首次不同分量 | L值 | J值 |
|---|---|---|---|---|---|
| no-conflict / weak-vector | C | unchanged | none | equal | equal |
| no-conflict / weak-vector | O | unchanged | none | equal | equal |
| conflict / weak-vector | C | unchanged | none | equal | equal |
| conflict / weak-vector | O | unchanged | none | equal | equal |
| diamond / weak-vector | C | unchanged | none | equal | equal |
| diamond / weak-vector | O | unchanged | none | equal | equal |
| remat / weak-vector | C | full-cost-difference-after-local-tie | cost[3] | 46080 | 27648 |
| remat / weak-vector | O | full-cost-difference-after-local-tie | cost[3] | 22528 | 18432 |
| attention / weak-vector | C | restricted producer-own Copy/access scope effect | cost[1] | 2144 | 800 |
| attention / weak-vector | O | restricted producer-own Copy/access scope effect | cost[1] | 2144 | 800 |
| attention / strong-cpasync | C | unchanged | none | equal | equal |
| attention / strong-cpasync | O | unchanged | none | equal | equal |
| attention / strong-tma | C | unchanged | none | equal | equal |
| attention / strong-tma | O | unchanged | none | equal | equal |

同一受控A/B候选配置的固定见证，独立于正式选择运行验证，用于证明每个主机制图确有两份完整可执行assignment；它们标注experimental injection，不能冒称自然枚举或把固定transport当选择收益。下表A/B均legal_return，存在exact-cubin普通/current-stream/capture数值与safety证据。Gate3四臂仍从自己的共同prepared domain求解，未以这些固定见证替换默认solver。

| 主图 | 完整A witness | 完整B witness |
|---|---|---|
| no-conflict | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/no-conflict-A-independent` (cubin 996259904821) | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/no-conflict-B-independent` (cubin 8a963cb8a441) |
| conflict | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/conflict-A-independent` (cubin da853064a5e0) | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/conflict-B-independent` (cubin ed2d055c9937) |
| diamond | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/diamond-A-independent` (cubin 7eec88d8d83d) | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/diamond-B-independent` (cubin 62b0bbada35c) |
| remat | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/remat-A-independent` (cubin b7b4d2da033e) | `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task2-final-006/remat-B-independent` (cubin d3defbe18bfa) |

## 数值、安全与回归

基础 corpus：28四臂+36固定回放，64份成功执行，25个不同 raw cubin、100项四工具 sanitizer 检查；另有 strong O1 四个 exact-cubin/ABI 家族、1620数值记录、16项 sanitizer。基础数值记录数4320；所有复用仅 exactbytes+ABI 相同。失败及被修复旧运行保留，不计作最终成功证据。新 f32 主图采用独立逐op bitexact参考；Attention保留finite abs(error)<=.002+.002*abs(reference)，transport/packing/cast/Attention既有容差未放宽。

完整 Unit595项/69suites(284.158s)、Selection38项(177.578s)运行命令、stdout/stderr/exit 见 `task-4-regressions.json`，Selection38项通过；最终 CTest10/10、Python43项通过见 `task-4-final-v4-regressions.json`；backend6项、export3项通过。此前 identity/schema CTest失败日志保留，并由重新configure/relink和严格新旧完整几何集合schema兼容修复。

## 冷编译、资源与机器身份

唯一最终冷编译为 `task-4-cold-final-20261030T074650Z`，身份入口 `task-4-final-cold-identity.json`，identity.json SHA256=7b0212659f9d4b79e1cdd4f6bc68704fe443f40ee00d648ebb2bcc3e4947c6ac。它绑定immutable final-v4工具清单、自身清单hash、actualbench/native工具before/after一致与全source哈希一致。原cold28及两次失败attempt原样保留为historical，不混入当前表；原cold阶段工具map未完整持久化，故全28行重测，不择优选取观测。每条冷编译独立子进程及隔离 TRITON_CACHE_DIR；不清 OS cache，不运行 oracle/数值/sanitizer。计时包括 fresh common preparation、一次真实 L/J选择/物化/export、TTGIR emission、Triton→cubin；总 wall 止于 compile child 返回，之后串行解析实际 cubin/SASS。每条只有一次描述性开销观测，不据此宣称统计编译优势。L stage1/stage2 在同一次 solve 内 chrono记录，总 solve包含验证/桥接等开销。

实际寄存器、physical Shared与local来自exact accepted cubin；SASS解析使用实际 cuobjdump，raw及normalized指纹分列。全部64份实际机器记录及完整SASS opcode histogram见 `task-4-actual-resources.json`；local/LDL/STL作为实际local与spill访存证据，与PTX文本route counts分别标记。TMA physical Shared6168B，不能用6144B payload代替。raw cubin可能仅debug不同；PTX差异亦可能被ptxas折叠。所有no-conflict四臂实际SASS相同；strong cp.async层四臂SASS相同，strong TMA层四臂SASS相同。相同机器代码是zero-effect sanity，观察到非零性能差异时应检查顺序/时钟/capture偏差，不能据此声称布局收益。

| case / ABI | arm | cold total ms | L1 / L2 / solve ms | regs / Shared / local | SHFL / LDS / STS / BAR | SASS hash |
|---|---|---|---|---|---|---|
| no-conflict / weak-vector | L-C | 1047.172 | 51.275 / 48.464 / 136.777 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 5f71bb00cbdf |
| no-conflict / weak-vector | L-O | 1124.767 | 50.121 / 47.570 / 132.927 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 5f71bb00cbdf |
| no-conflict / weak-vector | J-C | 979.776 | 0.000 / 0.000 / 60.993 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 5f71bb00cbdf |
| no-conflict / weak-vector | J-O | 1057.870 | 0.000 / 0.000 / 61.639 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 5f71bb00cbdf |
| conflict / weak-vector | L-C | 1192.635 | 53.459 / 47.834 / 141.029 | 48 / 4096 / 0 | 0 / 24 / 24 / 6 | 7392521619d1 |
| conflict / weak-vector | L-O | 1169.576 | 57.907 / 52.548 / 155.878 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 56f14f2337e7 |
| conflict / weak-vector | J-C | 1119.726 | 0.000 / 0.000 / 62.936 | 48 / 4096 / 0 | 0 / 24 / 24 / 6 | 7392521619d1 |
| conflict / weak-vector | J-O | 1077.268 | 0.000 / 0.000 / 66.380 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 56f14f2337e7 |
| diamond / weak-vector | L-C | 1184.237 | 52.428 / 47.821 / 139.435 | 48 / 4096 / 0 | 0 / 24 / 24 / 6 | 15478ce407bc |
| diamond / weak-vector | L-O | 1140.338 | 53.447 / 47.803 / 140.855 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 4d91715c1578 |
| diamond / weak-vector | J-C | 1102.850 | 0.000 / 0.000 / 62.502 | 48 / 4096 / 0 | 0 / 24 / 24 / 6 | 15478ce407bc |
| diamond / weak-vector | J-O | 1066.438 | 0.000 / 0.000 / 62.069 | 32 / 0 / 0 | 0 / 0 / 0 / 0 | 4d91715c1578 |
| remat / weak-vector | L-C | 1288.348 | 56.854 / 50.596 / 150.973 | 48 / 4096 / 0 | 0 / 40 / 40 / 10 | 908c8a6dcb11 |
| remat / weak-vector | L-O | 1352.618 | 57.693 / 51.312 / 153.255 | 48 / 4096 / 0 | 0 / 16 / 16 / 4 | 0a963f6d7aeb |
| remat / weak-vector | J-C | 1112.646 | 0.000 / 0.000 / 61.956 | 48 / 4096 / 0 | 0 / 24 / 24 / 6 | ea6be21d87b0 |
| remat / weak-vector | J-O | 1214.485 | 0.000 / 0.000 / 67.072 | 40 / 4096 / 0 | 0 / 16 / 16 / 4 | b28a0852b4d5 |
| attention / weak-vector | L-C | 7454.190 | 4688.410 / 346.535 / 5110.028 | 64 / 6144 / 0 | 256 / 0 / 24 / 3 | bad7bf174067 |
| attention / weak-vector | L-O | 7804.787 | 4703.204 / 348.010 / 5126.858 | 64 / 6144 / 0 | 256 / 0 / 24 / 3 | bad7bf174067 |
| attention / weak-vector | J-C | 8556.354 | 0.000 / 0.000 / 6063.656 | 64 / 6144 / 0 | 256 / 0 / 24 / 3 | a7357757fa6f |
| attention / weak-vector | J-O | 8961.861 | 0.000 / 0.000 / 6081.969 | 64 / 6144 / 0 | 256 / 0 / 24 / 3 | a7357757fa6f |
| attention / strong-cpasync | L-C | 2623.195 | 184.874 / 309.515 / 549.558 | 64 / 6144 / 0 | 256 / 15 / 0 / 3 | 9fa134691822 |
| attention / strong-cpasync | L-O | 2923.316 | 185.705 / 312.558 / 554.081 | 64 / 6144 / 0 | 256 / 15 / 0 / 3 | 9fa134691822 |
| attention / strong-cpasync | J-C | 2382.638 | 0.000 / 0.000 / 318.138 | 64 / 6144 / 0 | 256 / 15 / 0 / 3 | 9fa134691822 |
| attention / strong-cpasync | J-O | 2666.047 | 0.000 / 0.000 / 310.003 | 64 / 6144 / 0 | 256 / 15 / 0 / 3 | 9fa134691822 |
| attention / strong-tma | L-C | 2982.124 | 76.050 / 465.524 / 617.405 | 64 / 6168 / 0 | 256 / 0 / 0 / 6 | d4a71f646fa6 |
| attention / strong-tma | L-O | 3385.668 | 77.136 / 467.731 / 620.062 | 64 / 6168 / 0 | 256 / 0 / 0 / 6 | d4a71f646fa6 |
| attention / strong-tma | J-C | 2834.169 | 0.000 / 0.000 / 463.781 | 64 / 6168 / 0 | 256 / 0 / 0 / 6 | d4a71f646fa6 |
| attention / strong-tma | J-O | 3232.788 | 0.000 / 0.000 / 465.136 | 64 / 6168 / 0 | 256 / 0 / 0 / 6 | d4a71f646fa6 |

## 配对 GPU 测量

固定协议：GPU1 UUID GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618；恰好10预定独立配对batch，seed20261030随机四臂顺序、同batch相同输入/reference与独立输入/输出offset。每phase100 untimed warmup；100launch pilotgraph单独校准共同count，以slowest接近200ms。正式每臂10 replay raw CUDAevent样本；cuGraphGetNodes核实际节点count，capture和instantiate在samples之外且分列。perlaunch ms=graph_elapsed_ms/count，先collapse每batch median，bootstrap10配对batch(10000resamples、逐比较95%percentileCI（未作同时区间/多重比较校正）)，不是把graph内部launch当独立n。估计量为配对batch差值和baseline/target比值的中位数。普通singlelaunch events另列，可能含hostenqueue空隙，不冒充graph amortized纯kernel时间。

JIT、模块加载、allocation、host descriptor encode、异步上传完成event、capture/instantiate均排除于正式kernel样本并单列。全部10phase先ordinary+capture数值、QKV全bytes与Oguards检查；TMA测量hostencoder还与已验证encoder逐descriptor exactbytes对照。实际cuCtx deviceUUID与nvidia-smi均检查。query失败/空/格式异常failclosed；batch前后任何foreign computePID使整个pair invalid且保留，不选择性扩样；不足10valid则不作正式性能结论。

测量目录：`guoqiao/optimize_plan/joint_conversion_ablation_artifacts/run-task4-20261030T072322Z`。下表仅展示L-C/J-O，全部五组预声明比较及逐臂median/p10/p90、全部原始样本见analysis.json和每行raw.json。

| case / ABI | valid n | L-C median us | J-O median us | paired ratio [95%CI] | paired difference us [95%CI] |
|---|---|---|---|---|---|
| no-conflict / weak-vector | 10 | 5.2371 | 5.2417 | [1.0002508001164556, [0.9976434605050711, 1.0021114147638228]] | [0.0013193704491653267, [-0.012362939073972366, 0.011042786543923912]] |
| conflict / weak-vector | 10 | 7.0223 | 5.2042 | [1.3481661441060675, [1.3446368713239214, 1.3555807835772222]] | [1.812463195018414, [1.8005750678731685, 1.8431518499858555]] |
| diamond / weak-vector | 10 | 7.0130 | 5.2196 | [1.3435339949826508, [1.3385112850710676, 1.349187701207708]] | [1.79195321563519, [1.7695784449177263, 1.8205229924161732]] |
| remat / weak-vector | 10 | 8.3770 | 6.8967 | [1.214952318211295, [1.213551580673668, 1.2162148394006103]] | [1.4832002131959015, [1.471955827039686, 1.4907955778859818]] |
| attention / weak-vector | 10 | 6.8854 | 6.5386 | [1.0505230388320888, [1.049605856634168, 1.0530598491311434]] | [0.3320825520250325, [0.32556262942642566, 0.3445892291116578]] |
| attention / strong-cpasync | 10 | 6.4304 | 6.4554 | [1.0002917556291395, [0.9979962316465598, 1.0015257832303117]] | [0.0019054642733649409, [-0.012919545244308595, 0.009931830528232474]] |
| attention / strong-tma | 10 | 6.5866 | 6.6167 | [0.9989165568971663, [0.9959478699338873, 1.0000576411064617]] | [-0.007156503478844035, [-0.027066564235905897, 0.00040038536865865185]] |

### 完整四臂时间与全部预声明比较

时间为每batch内部replay median之后跨10batch的median，方括号为batch medians的p10/p90，单位us。差值/比值使用配对batch估计，不能由两个跨batch中位数直接相除代替。以下为观察数据，独立统计重算已PASS，最终报告/索引闭环审查已通过。

| case / ABI | L-C us [p10,p90] | L-O | J-C | J-O |
|---|---|---|---|---|
| no-conflict / weak-vector | 5.23709 [5.22268,5.25194] | 5.23861 [5.22818,5.25136] | 5.23565 [5.22051,5.24209] | 5.24171 [5.22328,5.25093] |
| conflict / weak-vector | 7.02234 [7.01166,7.03142] | 5.21267 [5.20157,5.21657] | 7.01856 [6.99351,7.03016] | 5.20417 [5.16810,5.23101] |
| diamond / weak-vector | 7.01300 [6.99201,7.03905] | 5.22149 [5.20297,5.22712] | 7.02127 [7.00101,7.04027] | 5.21958 [5.21265,5.23059] |
| remat / weak-vector | 8.37698 [8.35398,8.39592] | 6.89897 [6.89320,6.90469] | 7.47987 [7.45924,7.48754] | 6.89672 [6.88867,6.90606] |
| attention / weak-vector | 6.88536 [6.75545,7.00641] | 6.89304 [6.77328,7.02680] | 6.54806 [6.43960,6.66969] | 6.53863 [6.43145,6.67534] |
| attention / strong-cpasync | 6.43039 [6.40610,6.54022] | 6.44853 [6.38730,6.53311] | 6.44368 [6.40034,6.53699] | 6.45537 [6.40537,6.53069] |
| attention / strong-tma | 6.58663 [6.52208,6.67205] | 6.61270 [6.53146,6.68634] | 6.60534 [6.53620,6.68731] | 6.61665 [6.52436,6.69584] |

| case / ABI | paired contrast baseline/target | ratio [95% CI] | difference us [95% CI] |
|---|---|---|---|
| no-conflict / weak-vector | L-C/L-O | 0.999970 [0.997584,1.001554] | -0.000157 [-0.012692,0.008148] |
| no-conflict / weak-vector | L-C/J-C | 1.000868 [0.999007,1.002713] | 0.004547 [-0.005201,0.014182] |
| no-conflict / weak-vector | L-C/J-O | 1.000251 [0.997643,1.002111] | 0.001319 [-0.012363,0.011043] |
| no-conflict / weak-vector | L-O/J-O | 1.000198 [0.998973,1.001891] | 0.001035 [-0.005395,0.009810] |
| no-conflict / weak-vector | J-C/J-O | 0.998598 [0.996949,1.001244] | -0.007344 [-0.016004,0.006496] |
| conflict / weak-vector | L-C/L-O | 1.347781 [1.345812,1.349109] | 1.811757 [1.802257,1.820492] |
| conflict / weak-vector | L-C/J-C | 1.000723 [0.999116,1.002758] | 0.005073 [-0.006200,0.019288] |
| conflict / weak-vector | L-C/J-O | 1.348166 [1.344637,1.355581] | 1.812463 [1.800575,1.843152] |
| conflict / weak-vector | L-O/J-O | 1.000786 [0.998911,1.004842] | 0.004077 [-0.005692,0.025050] |
| conflict / weak-vector | J-C/J-O | 1.346339 [1.341920,1.356208] | 1.802791 [1.789626,1.846404] |
| diamond / weak-vector | L-C/L-O | 1.343233 [1.341857,1.347863] | 1.792080 [1.785476,1.814959] |
| diamond / weak-vector | L-C/J-C | 0.999907 [0.997217,1.001467] | -0.000651 [-0.019546,0.010288] |
| diamond / weak-vector | L-C/J-O | 1.343534 [1.338511,1.349188] | 1.791953 [1.769578,1.820523] |
| diamond / weak-vector | L-O/J-O | 1.000043 [0.995721,1.001376] | 0.000219 [-0.022364,0.007174] |
| diamond / weak-vector | J-C/J-O | 1.345521 [1.341047,1.347861] | 1.800765 [1.783536,1.815075] |
| remat / weak-vector | L-C/L-O | 1.214881 [1.212303,1.216145] | 1.483500 [1.464206,1.490440] |
| remat / weak-vector | L-C/J-C | 1.121551 [1.117632,1.122483] | 0.908815 [0.879945,0.915592] |
| remat / weak-vector | L-C/J-O | 1.214952 [1.213552,1.216215] | 1.483200 [1.471956,1.490796] |
| remat / weak-vector | L-O/J-O | 1.000562 [0.999308,1.001322] | 0.003875 [-0.004779,0.009111] |
| remat / weak-vector | J-C/J-O | 1.083863 [1.082364,1.085144] | 0.578286 [0.568542,0.586936] |
| attention / weak-vector | L-C/L-O | 0.999622 [0.996894,1.000799] | -0.002619 [-0.021299,0.005423] |
| attention / weak-vector | L-C/J-C | 1.051619 [1.049078,1.051995] | 0.337455 [0.316081,0.342314] |
| attention / weak-vector | L-C/J-O | 1.050523 [1.049606,1.053060] | 0.332083 [0.325563,0.344589] |
| attention / weak-vector | L-O/J-O | 1.052665 [1.049413,1.053902] | 0.341481 [0.328498,0.353479] |
| attention / weak-vector | J-C/J-O | 1.000801 [0.998712,1.001724] | 0.005235 [-0.008607,0.011216] |
| attention / strong-cpasync | L-C/L-O | 0.999545 [0.997303,1.003872] | -0.003012 [-0.017373,0.025245] |
| attention / strong-cpasync | L-C/J-C | 1.000296 [0.998440,1.001986] | 0.001906 [-0.010037,0.012868] |
| attention / strong-cpasync | L-C/J-O | 1.000292 [0.997996,1.001526] | 0.001905 [-0.012920,0.009932] |
| attention / strong-cpasync | L-O/J-O | 0.999282 [0.996980,1.001914] | -0.004697 [-0.019584,0.012271] |
| attention / strong-cpasync | J-C/J-O | 0.999778 [0.997066,1.001340] | -0.001426 [-0.018976,0.008586] |
| attention / strong-tma | L-C/L-O | 0.997846 [0.995135,1.000097] | -0.014296 [-0.031795,0.000648] |
| attention / strong-tma | L-C/J-C | 0.997896 [0.996738,1.000384] | -0.013736 [-0.021765,0.002526] |
| attention / strong-tma | L-C/J-O | 0.998917 [0.995948,1.000058] | -0.007157 [-0.027067,0.000400] |
| attention / strong-tma | L-O/J-O | 0.999313 [0.998155,1.003402] | -0.004587 [-0.012229,0.022519] |
| attention / strong-tma | J-C/J-O | 1.000250 [0.997558,1.002075] | 0.001601 [-0.016171,0.013773] |

七行各10/10有效配对批次，70pairs/280arm records，2800graph replay events与5600ordinary events；全部actual节点数=count，全部10phase预检通过，foreign/invalid为0。source/tool末尾复验一致。一次控制面capacity事件没有重启既有remote session35357或测量进程，没有删除、替换或扩充批次。

观察：no-conflict四臂同SASS且全部配对CI包含ratio1，零效应控制未显示显著偏离。conflict与diamond的L/J同policy比较CI包含1，而C/O转换优化组合的配对ratio约1.34；remat的L-C/J-C ratio约1.122，来源是local tie消除，L-O/J-O CI包含1。weak Attention L/J ratio约1.052，须保留producer自身access被scope排除的归因限制。强cp.async/TMA各层四臂同SASS，所有预声明ratio CI均包含1，不将微小点估计差异解释为布局收益。以上独立重算已PASS，保留scope与机制归因限制，不作跨ABI或跨框架结论。

### setup / capture / ordinary launch

模块加载只按每臂一次列出；allocation+descriptor+upload同步的setup wall不冒充纯hostencode。descriptor encode含pinned allocation，upload为实际CUDAevent完成时间。capture与显式instantiate均在正式event之外。ordinary统计严格先取每batch20个event的median，再取10个batch median的median，不pool成200个独立样本。ordinary单launch event包含host enqueue间隙的可能影响，与graph摊销口径分列。weak Attention的L-C ordinary约29.84us、L-O约15.73us，但二者same SASS、graph均约6.89us且配对CI包含1；这是host enqueue敏感的观测异常，不能解释为C/O kernel收益，原始数据全部保留。完整逐phase记录保留在setup/raw JSON。

| case / ABI | setup median ms | descriptor encode median us | upload median us | capture median ms | instantiate median ms | ordinary L-C/L-O/J-C/J-O us |
|---|---|---|---|---|---|---|
| no-conflict / weak-vector | 0.200 | n/a | n/a | 331.370 | 83.364 | 14.928 / 14.840 / 14.648 / 14.664 |
| conflict / weak-vector | 0.114 | n/a | n/a | 241.240 | 48.301 | 15.984 / 14.776 / 15.920 / 14.720 |
| diamond / weak-vector | 0.115 | n/a | n/a | 239.079 | 48.421 | 15.880 / 14.896 / 15.928 / 15.000 |
| remat / weak-vector | 0.114 | n/a | n/a | 202.166 | 38.634 | 17.296 / 15.784 / 16.432 / 15.896 |
| attention / weak-vector | 0.129 | n/a | n/a | 233.360 | 47.667 | 29.840 / 15.728 / 15.288 / 15.320 |
| attention / strong-cpasync | 0.127 | n/a | n/a | 248.588 | 51.451 | 15.136 / 15.424 / 15.336 / 15.320 |
| attention / strong-tma | 0.373 | 29.855 | 13.440 | 245.881 | 50.689 | 15.752 / 15.752 / 15.976 / 15.784 |

## 结论范围与审查

没有预设严格反转、性能提升或胜过Triton/TileLang；本实验不支持跨框架优越性结论，也不能从2×2独立归因三个转换优化机制。无差异、编译器折叠、退化、不支持和预算耗尽均应保留；当前主矩阵未凭模型分数直接推断机器收益。当前7行共同可执行矩阵无unsupported、数值失败、安全失败或预算耗尽行；历史失败原样保留。无收益/近噪声行包括no-conflict、strong cp.async/TMA，以及conflict/diamond的同policy L/J与remat的L-O/J-O。没有以本组逐比较CI确认的明显退化行。scf循环、多warpgroup、任意shape/Reduce域及循环频率外提不在本GPU支持范围，不把未覆盖机制称为已验证。默认生产调用路径和选项保持不变。

Gate1–3 独立 spec/quality 复审及root数值/身份审计见证据索引中的对应review/audit文件。最终测量source digest为`95badf028ba969fa2c3e5a3bfc223ac4206a22898f765623ff57cc15ee9b3ea1`，source-freeze SHA为`e6578a39168c3656f2169faa8122a9338ded77b863d8c3757aef62f8fbb5d242`。工具hash guard的小delta另保留`task-4-toolguard-delta.patch`，oldhash精确匹配原checkpoint；当前measurement源码94992a90…/测试f67c95a2…已由同一fresh whole reviewer复审，无Critical/Important。Gate4 whole-source checkpoint由fresh agent `/root/joint_ablation_delivery/whole_review` (gpt-6-astra high)独立审查：Approved，0 Critical/Important，35hash匹配；证据 `whole-review.md`。后续final measurement/source-delta checkpoint已独立复算全部35组paired estimates/CIs、全部raw身份/seeded顺序/节点count、64份SASS与开销汇总并全部PASS，未发现Critical/Important。两个分项verdict为source Approved、data/statistics PASS；整体最终独立审查已通过，详见whole-review.md最终Assessment；evidence-index.json记录最终文件哈希。不把实现者自审当独立验收。最终交付是本报告和一个evidence-index.json。

## 历史附录（原记录完整保留）

以下内容是旧阶段与修复流水，非当前验收主文。

---

# Joint selection and conversion ablation report


## Task 1 — policy, scoped search, producer-first coordinator, oracle interfaces

Implementation source frozen for independent review on 2026-10-30. Exact 21-file list: `../optimize_plan/joint_conversion_ablation_artifacts/task-1-changed-files.json`; per-file SHA256: `task-1-source-digests.json`. Combined sorted JSON digest: `23dabfa2ed8f207553021447b1ad9ceb29f522bd12236be1647610b27cc9f93a`. No staging, commit, push, worktree or local copy; baseline roadmap preserved. No GPU experiment has run in this gate.

### Implemented behavior and causal limits

- Independent preserves identity and existing explicit boundary semantics; nonidentity repair edges remain independent and disable rematerialization. SharedOnly merges matching repair groups but disables rematerialization. Full remains default. Score, solver, explicit expected-policy verification, materialization and bounded oracle carry this policy. Materialization now takes independent expectedPolicy defaulting to Full; experimental callers must provide the fixed arm's policy, never derive it from a mutable solution tag.
- Producer-first stage 1 checks full hard feasibility and complete conversion legality but sums only declared producer charges. Future access/conversion/sharing/Shared-root charges are excluded. Unknown contribution kinds, target aggregate fallback and missing observer totals fail closed. Scope permits Distributed definitions and instruction geometry; Storage producers and operand-use variables are rejected.
- For the fixed legacy SM90 model, real `frisk.mma` work has a checked narrow dependency: `gM*gN*mRepeat*nRepeat*kRepeat` depends on geometry, with fixed shapes and execution threads. The MMA output and geometry must both be registered. SS Storage operands remain full feasibility dependencies and are never local cost producers. Generic instruction costs retain full endpoint dependency rules. The manifest includes accepted/excluded dependencies and this scope restriction.
- Local partial bound is zero. Exact/beam/refinement use producer stableName plus candidate stableOrdinal after local cost ties; consumer completion keys do not rank producer tuples. The oracle implements its own producer tuple comparison, independently of production less/MRV/beam.
- One opt-in SharedSearchBudget spans components and both L stages. Relationship/MRV probes, complete-score attempts and refinement candidates are exclusive total categories; refinement is a subset of total. Final certificate regeneration and final diagnostic full reevaluation are separately documented diagnostic work, not search probes. Null sharedBudget retains existing component budget behavior. Stage1 half-cap and stage2 total-cap preserve unused allowance.
- Stage1 saves a full witness. Stage2 pins producer assignments and selected instruction bindings/geometry on a graph copy and uses full cost. Typed SearchBudget failure retains the original witness and regenerates its full-policy cost/plan; other failures do not become successful fallback. legal_return, search_complete and stage2_improved are separate. Oracle never substitutes its answer.
- Oracle candidate iteration now independently checks target candidate validity and Storage capacity. FrozenProofs domain fingerprint/coverage strengthening remains Gate 3 work; this gate does not claim adversarial proof-package identity validation.

### RED → GREEN evidence

All paths below are under `guoqiao/optimize_plan/joint_conversion_ablation_artifacts/`.

| Critical behavior | RED evidence | Final result |
|---|---|---|
| Independent no sharing/no remat, SharedOnly no remat, identity no work, default Full unchanged | task-1-policy-red.log missing requested contract | Five policy tests PASS, task-1-policy-green.log and final-check-1.log |
| Cross-policy chosen-plan and actual materialization rejection | policy RED; task-1-materialize-negative-red.log missing independent materialization contract | O under C and C under O reject; forged tag does not bypass plan reconstruction; final-check-1.log PASS |
| Producer tie ignores consumer ordering/future costs | task-1-solver-red.log assertion selected a vs stable b | PASS final-check-1.log |
| Local bound never invokes global lowerBound | task-1-solver-red.log bound count 2 vs expected 0 | PASS final-check-1.log |
| Full hard legality during local choice | finite hard-table test | PASS final-check-1.log |
| Shared budget across disconnected components/run calls; default null preserves semantics | task-1-solver-red.log missing shared accounting | PASS, category sum equals total, final-check-1.log |
| Unknown and missing charge coverage fail closed | task-1-scope-negative-red.log expected rejection failed | PASS final-check-1.log |
| Refinement accepts lower stable producer tuple after strict descent | task-1-tie-validity-red.log plateau assertion failed | PASS final-check-1.log |
| Genuine stage2 exhaustion preserves full legal witness | finite 30-probe test, stage1 <=15, remaining stage2 exhausted during full score | 3 assignments and instruction binding retained, verifySolved passes; final-focused-0.log |
| Geometry remains pinned, no-witness fails | finite instruction domain; tiny shared budget | PASS final-focused-0.log |
| Oracle stable producer tuple and independent candidate rejection | task-1-oracle-tie-red.log / task-1-tie-validity-red.log | PASS final-focused-0.log |
| Scope dependency manifest table | task-1-scope-table-red.log missing table | PASS final-focused-0.log |
| Real SS MMA geometry-only work; missing geometry/Storage/future-use negatives | task-1-real-mma-scope-red.log work=0 and missing geometry accepted | PASS final-check-1.log |

### Validation provenance

`task-1-final-release.log` contains full Release unit suites passing 591/591 and selection 26/26 on the pre-real-MMA-dependency snapshot. Those broad results are supporting regression evidence, **not** final-source full-suite certification; its exact complete source digest was not captured before the final scope adjustment. The final source digest above corresponds to Release rebuild `task-1-final-check-0.log`, final affected unit checks 67/67 in `task-1-final-check-1.log`, and final selection checks 6/6 in `task-1-final-focused-0.log`. A redundant second full selection run was intentionally terminated after the earlier complete suite passed; `task-1-final-checks.json` retains its SIGTERM exit rather than hiding it. Final full Release suite is deferred to the planned final gate.

Final focused Debug build/check results are complete in `task-1-final-focused.json`: build exit 0; affected unit checks 67/67 and selection checks 6/6 PASS (`task-1-final-focused-{1,2,3}.log`). Release focused selection is `task-1-final-focused-0.log` (6/6 PASS). All these focused checks use the final 21-file digest recorded above. Existing GCC APInt warnings and CMake CMP0116 deprecation warning appear in build logs; no new warning category was intentionally introduced. Independent source/spec review is pending; this section is not a review approval.

Task 1 self-review status: DONE_WITH_CONCERNS pending independent gate review. `git diff --check` passed. Caller contract for experimental scope/solveAblation is a copied **already finalized frozen graph**: scope IDs refer to that prepared domain. The API currently relies on this caller contract rather than independently rejecting an unfinalized experimental graph; adversarial scope-ID validation should be reviewed. Frozen-proof domain/coverage identity is explicitly deferred to Gate 3. No implementation/build/oracle/GPU process remains running at Task 1 handoff.


### Task 1 independent review fixes — source frozen for re-review

The five review findings were addressed in one TDD repair cycle. `task-1-review-fixes-red.log` records intended failures before production changes; `task-1-review-fixes-green.log` records initial green. The same 21-file changed-source list remains authoritative. Updated combined source digest: `438251a4f187f8d705c18020fa4f2be5a9a6df752289d4bea605d9df1396e4b1`; per-file digests are `task-1-review-fixes-source-digests.json` and the refreshed `task-1-source-digests.json`.

1. **Frozen identity is now enforced.** Experimental solver calls, producer-first/joint experiment coordinator calls, direct local scoring and scoped oracle reject unfinalized graphs. The default production solver still auto-finalizes. Scope IDs are never silently remapped. The rejection regression uses zproducer/aconsumer insertion order and two different constraint kinds, independently finalizes a copy and proves that both variable identity and constraint kind at ID0 actually change. Coordinator/oracle reject before probes or mutation. Existing producer/consumer fixtures keep their original names, finalize first and resolve the producer by stableName; custom target costs now also resolve the true producer name. Finite geometry/consumer fixtures were similarly corrected; previously prefixed names used to sidestep sorting were removed.
2. **Saturated global producer ties are compared after component aggregation.** If original/refined lex9 sums tie, local mode compares the complete sorted producer tuple and retains the smaller stable ordinal tuple; the full-graph default comparison is unchanged. The new two-component regression saturates a strict local work improvement into a global tie and proves that the original stable producer assignment and fallback flag survive.
3. **Scope evidence is generated by the real scorer observer.** A single contribution rule now drives inclusion and evidence for every actual callback. Rows contain the actual cost kind, actual variable/constraint owner ID and stable owner identity, stable dependency IDs/names, included flag, reason and observed cost. No instruction/use/function-result variable is guessed to have a registers charge; no constraint kind is mislabeled as a cost kind. Compound excluded Shared-root/access/conversion/rematerialization charges conservatively list the full graph as dependencies, rather than presenting leader identity as a complete dependency proof. MMA geometry work retains the reviewed narrow legacy rule. The table test checks actual schema, registered dependency names and included register-charge sum against stage1's real cost.
4. **Experiment ceilings are enforced.** solveAblation rejects budgets above 65536 total or 256 refinement before search; lower test limits remain supported. Both raised-limit negatives and existing lowered-budget witness/no-witness cases pass.
5. **Ninja warning provenance is recorded honestly.** The exact warning `ninja: warning: premature end of file; recovering` appears twice in the earlier Debug build `task-1-final-focused-1.log`, at lines 1 and 34, immediately around CMake regeneration. The contexts are preserved in `task-1-ninja-recovery-warning.json`. This is a Ninja build metadata recovery warning, distinct from GCC APInt and CMake CMP0116 warnings. Existing logs do not identify the truncated metadata file or prove the originating cause; no stronger attribution is claimed. That build completed and tests passed. No build directory/log cleanup was used to conceal the warning.

Affected Release tests after repair passed 69/69; selection/oracle tests passed 12/12 (`task-1-review-fixes-release.log`). Affected Debug tests also passed 69/69 and 12/12 (`task-1-review-fixes-debug.log`). The last change corrected the raw-ID rejection test to use its original producer ID0 and added only the API contract comment; exact final-snapshot rebuild/check logs are `task-1-review-fixes-final-{build-m6-release,build}.log`, with commands/exits in `task-1-review-fixes-final-commands.json` (completion pending at the time of this source-freeze entry). No unrelated full suite was rerun. FrozenProofs identity strengthening remains assigned to Gate 3.

Final review-fix snapshot validation completed: all six build/test commands in task-1-review-fixes-final-commands.json exited 0. Release and Debug each passed 69 affected unit tests and 12 selection/oracle tests on combined digest 438251a4f187f8d705c18020fa4f2be5a9a6df752289d4bea605d9df1396e4b1. Final diff check exit 0. Source remains frozen and no build/test process remains running. The finalized-scope reliance concern from the original Task 1 handoff is now resolved by executable guards; independent re-review remains pending.

### Task 1 independent rereview outcome

Fresh independent reviewer approved both spec compliance and code quality (task-1-rereview.md), with all initial Important findings closed. Final exact source digest and six successful command outputs are in task-1-review-fixes-final-commands.json. The final Debug log again contains Ninja premature-end/recovering warnings at lines 2 and 23; the actual affected metadata file/root cause is not established by existing logs. Build and tests exited successfully; this is retained as tool-output noise, not represented as a warning-free build.


## Gate 2: common executable routes and mechanism witnesses

The common backend now exports and independently validates the existing checked redistribution plan. Register conversions use registers, Shuffle uses converged unconditional shuffles followed by receiver register selection, and Shared uses one elected writer per logical slot plus publish/reuse barriers. Existing Shared roots precede route-aware maximum sequential scratch; Register/Shuffle reserve zero conversion scratch. This retains the existing sequential reuse strategy. Conversion-internal right-inverse writers and the external `first_owner` ABI are separately checked.

`task-2-final-manifest.json` indexes `run-task2-final-006`: all four controlled f32 graph families have two complete legal forced producer assignments, each under Independent and Full policy (16 executable assignments). All materialize, verify, export, compile and execute. Every selected repair in these graph records has `existing_conversion=false`; sharing and bounded mulf rematerialization occur in actual Full plans (including remat-A). The raw entry is explicitly an experimental candidate-injection analysis fixture, not a production-native preverified input. Each common domain has exactly two combinations; each record freezes producer ID/stable name, complete assignment, domain/proofs and contributions. The asymmetric conflict/diamond formulas are nonzero and each branch independently changes output. `joint_fixture.py` uses seed20261030, separate NumPy float32 mul/sub operations, no fusion/reassociation, and preregistered input/expected byte hashes.

There are four additional real route controls: Register, ordinary Shuffle, receiver-lane-dependent source-register Shuffle, and four-lane-replica Shared. All 20 exact cubins pass byte-exact outputs, unchanged input bytes and canaries at offsets1/2/3. Each offset checks three ordinary launches, a nondefault-stream launch, and replay of an inspected three-target-node CUDA Graph. All 80 exact-cubin sanitizer invocations (memcheck, racecheck, synccheck, initcheck) pass; safety copies preserve the accepted cubin SHA and never recompile. GPU UUID is `GPU-028d5ad6-2ff8-f6e3-130f-9eefb0447618`; GPU0 was not used. The runner parses actual PTX parameter ABI and launches on the current Torch stream.

Loaded resources show no-conflict A uses zero Shared, while primary nonidentity Shared cases use4096 bytes; actual local bytes are zero. These are implementation/resource observations, not performance claims. Raw cubin and PTX hashes are preserved. No-conflict A Independent/Full have identical TTGIR and normalized executable PTX (`3375689475153cdbcf56884a0ce6481438947b3338a6e6a3a9d2047e1c72a880`), while raw cubin hashes differ because of `.file`/DWARF path data; this is not computational work difference.

The final GPU run records one consistent runtime source digest `184c1965fda7f5241c28ca4d49af1637b36955f87e134edc4dc3f2ab865d9779`; tool versions, export identities, loaded resources and build/cache identities are retained. Ten changed source/test files are frozen in `task-2-source-snapshot.json` and `task-2-source-sha256.json`, with `task-2-netdiff.patch` relative to Task1/unchanged HEAD. RED/GREEN logs cover route validation/emission, scratch and replica election, export fields and numerical branch references. Release export/plan regression passes17 tests; CPU route/reference/Attention-backend regressions pass4+1+6. Debug build and all36 affected Debug tests passed. The optional full Debug selection run was stopped with SIGTERM after611 seconds under the coordinator’s instruction; nine tests had passed, with ControlledCliAndRealFrozenOracleReplay active. Its partial stdout and stop record are retained, and no full-selection pass is claimed. The final global selection regression remains Gate4 work. Independent review requested a host-only artifact-lifecycle fix, addressed below. Earlier failed fresh run directories remain as diagnostic evidence and are excluded from accepted final006. No four-arm selection/oracle/performance claim is made at this gate.


### Gate 2 host-only review fixes

The fresh independent review (`task-2-review.md`) approved route/ownership, actual sharing/remat and all20/80 numerical/safety evidence, but requested artifact-lifecycle fixes. A targeted six-case RED first demonstrated that nonempty directories and invalid enum values were accepted, and error handling was not controlled. The CLI now validates known case/producer/policy strings and required graph structure before output, rejects nonempty directories without changing existing bytes, checks mkdir, and checks every write/flush/close result. All six cases now return controlled status2 and pass preservation/no-output checks (`task-2-host-{red,green}.log`); a missing-arithmetic fixture also returns2 without creating output. All16 fresh successful host runs retain byte-identical materialized IR, execution JSON and domain JSON (`run-task2-host-success-001`). Backend, references and GPU execution paths are unchanged; GPU reruns were unnecessary for this host-only fix.

Accepted GPU runs retain their original source digest `184c1965fda7f5241c28ca4d49af1637b36955f87e134edc4dc3f2ab865d9779` and raw hashes. The subsequent host-only CLI source digest is `8969ed0d02c7093b664d4200f5c92fb6f9747fa294cdf7d339c1791c4976f160`. The original GPU source snapshot remains `task-2-gpu-source-snapshot.json`; current ten-file hashes and host diff are separate. Control IR identity is now explicitly the actual `input.mlir` path, role `preselected-transport-input`, and SHA256 of its bytes; main graph IR identity names materialized `output.mlir`. Former control output hashes were null because no output.mlir exists, not a valid hash of a different file.

Build logs are preserved with their warnings: Debug Ninja emitted “premature end of file; recovering”, and CMake emitted CMP0116 OLD-behavior deprecation output. No unverified cause is attributed to the Ninja recovery, and the build logs are not described as warning-free. Host-only rereview is pending.

### Task 2 independent rereview outcome

Fresh reviewer approved spec compliance and task quality with no open findings. The six failed-input/output paths reject safely; 16 successful host exports (48 files) match the GPU006 inputs byte for byte. All 20 actual IR paths/hashes now match the manifest. Only host CLI changed among 176 runtime-source files; accepted GPU source and host-fix source identities remain separate. Full evidence in task-2-review.md.


## Gate 3 implementation and validation in progress (2026-10-30)

The frozen implementation review package is `task-3-source-snapshot.json` and `task-3-review.patch` under the artifact root. The patch combines approved Task 1 snapshot with approved Task 2 snapshot for the eight changed/new source files. No source, build or CPU test process is running concurrently with the serialized GPU validation.

The common pre-propagation package binds input MLIR, complete domains/support/proof tables, producer scope and actual legacy cost/frequency metadata. Frozen proofs reject domain/coverage drift, reconcile target candidate validity and capacity, and support explicit verified subdomain projection. L2 restricts only declared producer encodings and instruction geometry/path; consumer/storage roles remain available for full-objective completion. The natural Attention direct materialization path reuses existing `preserveStorageContracts` without invoking selection again.

CPU evidence: `task-3-selection-regressions.log` records 33/33 affected selection tests before final local refinements; `task-3-final-selection-green.log` records coordinator/proof tests; later `task-3-restriction-green.log` records 13 finite restriction/validity tests, and `task-3-rootcontracts-green.log` records all six public Python tests including natural Copy/MMA two-stage materialization and CPU emission, prepared metadata tampering, complete-assignment policy-safe replay and executable identity. RED logs are retained for each corrected integrity, pinning and emission failure. Earlier fresh run `run-task3-20261030T053148Z-b6b4d88f` stopped fatally at missing Shared-root declarations after 36 numerical successes; it remains a failed evidence row set and is not the accepted matrix.

Current candidate matrix: `run-task3-20261030T053924Z-574ed1d8/manifest.json`. All 28 four-arm rows and 36 fixed-assignment replay rows have produced exact cubins and passed numerical validation. Validation subsequently completed: 28 four-arm rows plus 36 replay rows (64 exact-cubin numerical successes), 25 distinct raw cubins each passed four sanitizer tools (100 checks), and 39 rows reused only exactly equal cubin bytes plus ABI. The final manifest SHA256 is 272edd64a3858ea80c591575c9a6301708f6ba40fb2928d040b859b1cbe98214. This remains pending fresh review, and no formal timing was run. Attention preserves natural candidate domains, 128-thread single-native-geometry support and separate weak-vector / strong-cp.async / strong-TMA ABI strata. All three strata's L producer tuples match complete bounded local oracles. Strong cp.async/TMA strata have equal four-arm full costs. Weak-vector L=[8293,2144,288,0,428,6144,6,0,0] and J=[8293,800,0,0,428,6144,6,0,0], within either conversion policy. L's equal oracle comparison is relative to its fixed-producer L2 oracle, whereas J compares with the global oracle; these references must not be conflated. Copy/access scope attribution is examined separately.

The controlled remat graph has a local-objective tie resolved differently by joint selection; it is not a strict reversal. Its first differing final full-objective component is `conversionBytesAndSync` (index 3): C L=46080/J=27648; O L=22528/J=18432 (later register component L=72/J=64). These are model costs, not physical Shared allocation or machine speed. Other controlled graphs have no within-policy L/J full-objective difference.

Producer-access attribution is frozen separately in `task-3-forced-producer-access-attribution.json`, referencing all 16 accepted forced-A/B record hashes. Only diamond has producer-owned access charges: A=[32,1028,0,0,0,0,0,0,0], B=[32,1152,0,0,0,0,0,0,0]. Its forced A/B full-objective contrast therefore includes the excluded producer access baseline; it cannot be attributed wholly to cross-consumer conversions. The four selected diamond arms all use the same producer/access charge. No-conflict/conflict/remat have no producer-owned access contribution.

Raw PTX/cubin hashes remain provenance. The version-2 executable PTX fingerprint retains declarations/directives/instructions and removes comments/source file/location directives; it is explicitly versioned separately from Gate 2's entry-only fingerprint. Identical executable fingerprints do not imply identical raw cubin bytes or transferable sanitizer acceptance.


Weak Attention cost attribution is recorded in `task-3-attention-weak-attribution.json`: full L/J contribution differences belong to access charges; the local objective excludes access, while producer Copy execution/path choices are fixed before full completion. This is a restricted-scope/tie baseline effect with no conversion objective difference, not evidence of cross-consumer conversion savings.

LLVM IR was recovered for all 64 artifacts from cache-eligible Triton compilation using the original saved TTGIR and options. Every returned cubin was asserted byte-identical to the accepted cubin before saving `kernel.llir`; no kernel or raw accepted artifact was replaced. `task-3-llir-tool-recovery.json` records exact cubin matches, llir hashes, observed recovery times (not cold compilation), and actual ptxas, loaded libtriton.so, Python, benchmark and sanitizer executable hashes. The manifest includes these additional artifact/tool identities.


### Gate 3 repair wave and exact-identity supplement

The independent review's four Important findings were repaired together. Prepare, matrix, replay and worker phases now share a typed fatal/documented outcome contract: malformed/inconsistent data, signals, numerical or safety failures persist a fatal partial manifest and abort; proven NoFeasible, Unknown, no-witness search budget and supported explicit backend outcomes remain visible typed failed rows. Complete nested prepared/assignment schemas validate required types, enums, ranges and unique IDs before graph access. Tests recompute malformed assignments' hashes so rejection exercises schema validation rather than merely detecting stale hashes. Complete and incumbent comparisons now explicitly distinguish `global` and `fixed-producer-L2` references.

Repair evidence: `task-3-fixwave-selection-green.log` passes 18 C++ coordinator/oracle/outcome tests; `task-3-fixwave-final-public-green.log` passes 12 Python tests, including persistent fatal manifests for every phase, documented-budget retention, signal/replay/numerical failures, malformed worker results, nested assignment negatives and independent O1 offsets. Source snapshot and patch are `task-3-fixwave-source-snapshot.json` and `task-3-fixwave-review.patch`. The repaired host source digest is c2a42016113b35fad0e18b0b767bc950196b457494cfef01fa31f0a653cad127. `run-task3-hostfix-0ad78784ab/manifest.json` proves all 64 complete assignments and 320 saved materialization/export files remain byte-identical to the accepted GPU-source artifacts; GPU-source and repaired-host provenance remain separate.

Strong QKV128/O1 coverage was supplemented for every four distinct accepted raw cubin/ABI family in `run-task3-strong-o1-00e3242c3b/manifest.json`. Each image executes 15 cases ×3 aligned QKV offsets ×3 independent O offsets ×3 resets =405 numerical launches, totaling 1620. QKV offsets are 0/128/256 and output offsets are 1/2/3, recorded independently with output canaries and unchanged inputs. Each image also passes memcheck/racecheck/initcheck/synccheck on seed0 ×3 input offsets ×3 output offsets ×3 repeats =27 launches/tool/image: 16 supplemental checks, not 405 launches per sanitizer. These executions load the same accepted cubin bytes and ABI into new evidence directories; the original aligned-output runs remain intact.

`task-3-selection-classification.json` independently compares complete assignments and declared producer tuples. No-conflict/conflict/diamond are unchanged between L/J within either policy. Both strong Attention strata are also unchanged in complete assignment identity, not merely equal in cost. Weak Attention changes Copy producers 33/35/37 and differs in omitted own-access charges; remat changes producer 10 after a local objective tie. No strict reversal or performance benefit is asserted. Mechanism coverage is established from actual fixed-assignment plan slices (conflict two-mulf, diamond one-mulf, remat one-mulf slices, all ≤8 operations), IR/LLIR/PTX/resource evidence, rather than case names.

Command sidecars now record argv, exit status and observed wall duration. These host/cache-observed durations do not recover the original missing cold compilation timings; formal cold compile and performance measurements remain Gate 4 work. Accepted CUDA Graph sanitizer logs contain Torch debug-dump UserWarnings (`cudaGraphDebugDotPrint`) with zero sanitizer errors/hazards; the logs are retained and are not described as warning-free. The final evidence index is `task-3-final-index.json`, including base validation, LLVM/tool recovery, repaired-host byte equivalence, O1 supplementation, attribution, classifications and the root's independent base-data audit. All source/build/GPU jobs are stopped pending rereview.


### Final Python protocol guards (host-only)

The last rereview's two Python boundary findings are repaired without rebuilding or executing a GPU kernel. The runner distinguishes its own caught `TimeoutExpired` provenance from an arbitrary child's exit124. Only prepare/selection/replay parent timeouts become retained documented timeout rows; numerical/safety timeouts remain fatal. Success results now require role-specific typed identities, complete assignment/binding hashes, lex9/flags, and explicit true numerical/guard/input booleans; empty records, missing assignment identity, integers or strings in place of booleans and malformed worker data fail closed. Parsing and subsequent exceptions persist fatal manifests, while a rejected repeat validation of an already frozen run writes fresh sibling failure evidence and preserves the original manifest.

`task-3-protocolguard-red.log` and `task-3-protocolguard-immutable-red.log` retain failures; `task-3-protocolguard-final-green.log` passes all 16 Python tests in 22.560s. The frozen two-file delta is `task-3-protocolguard-review.patch` with snapshot `task-3-protocolguard-source-snapshot.json`; current host protocol source digest is e885b74850a717005feb50882d7fe9bc0846e30fbe9821ad30cc06982a0dc4d2. The accepted base GPU source, 64-row host byte-equivalence source and strong O1 supplemental source retain their original separate provenance and unchanged manifests. `task-3-protocolguard-role-schema-audit.json` additionally validates seven prepared packages, 28 matrix records, 36 replay records and four strong numerical evidence objects against the new schemas on CPU. The root's independent base and host/O1 data audits both report no failures and are referenced in `task-3-final-index.json`. Final source rereview is pending; all jobs are stopped and Gate 4 has not started.

### Task 3 final independent rereview

Independent reviewer approved spec compliance and code quality; all findings in Gate3 scope are closed (task-3-rereview.md final superseding assessment). The strict worker-schema guards, trustworthy timeout classification and immutable rejected-validation sibling were verified. Current source digest e885b74850a717005feb50882d7fe9bc0846e30fbe9821ad30cc06982a0dc4d2 is separate from immutable base/host/O1 evidence identities. Gate4 formal timing and cold compilation remain pending.

