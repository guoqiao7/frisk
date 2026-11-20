# Frisk 布局推断：决策修复、LOCAL 校准与开销优化阶段报告

**状态：实现、功能与CPU开销已验收；GPU选择质量FAILED，不晋升默认策略。** 本文记录完整R3成功与失败结果；默认继续legacy，全部资料仅在H100服务器。

## 1. 目标、冻结身份与适用范围

本阶段处理三个问题：部分候选存在未训练特征时，整池回退丢失可证明更优的已知候选；LOCAL 静态流量缺少独立正成本校准；六来源重复搜索与编译的 CPU 开销偏高。

已完成实现、训练集校准、CPU与GPU功能回归及独立审核。正式R3确认CPU开销下降，但新留出GPU质量失败，详见第8节。

| 项目 | 固定身份或范围 |
| --- | --- |
| 最终源码 | source-freeze-r2；digest 49763e080865fcf4f6de49e416c1ea57b49624e12211e360c14cd33b89b40904 |
| 冻结收据 | /home/baopeihua/frisk/guoqiao/optimize_plan/partial_cost_efficiency_artifacts/development/source-freeze-r2/freeze.json |
| 冻结收据 SHA256 | f4d88d4afa33b0cf48ecfda6394b99daedcb6304ca5c0bf147739141c012eb91 |
| 当前正式资格运行 | /home/baopeihua/frisk/guoqiao/optimize_plan/partial_cost_efficiency_artifacts/run-partial-cost-efficiency-r3-20261103 |
| 当前效率运行 | /home/baopeihua/frisk/guoqiao/optimize_plan/partial_cost_efficiency_artifacts/development/efficiency-benchmark-r3 |
| 资格协议 SHA256 | 69d0c8a4be045e8e8c345e8d0e029d60e7e858b7415ffececb0175446f69b48e |
| 资格协议 digest | 65e3363213a5e7081d1895a82e10081efc32d7e2a5df36509059d6facddb274e |
| 静态搜索模型 | sm90-layout-calibrated-v6d-1850a5d94517c090e16c713c307017c4-leg |
| LOCAL native profile | sm90-native-local-aware-v1-0449658d7769af15 |
| 支持范围 | SM90、有界单 CTA 后端；未建模 CFG/MMA 保持 typed unsupported/fallback |

R3 沿用 source-freeze-r2，未因 R2 中止修改产品源码、模型、输入或阈值。旧实验已经观察过的程序属于训练/开发回归，不再作为新性能留出。本阶段不声称相对 Triton/TileLang 的性能优势；外部公平比较仍需相同算法、ABI、精度与调优预算。

## 2. 决策机制与数学边界

实现入口为 /home/baopeihua/frisk/benchmark/layout/execution/partial_cost_selection.py。每个候选保留完整证据，并区分：

| 状态 | 含义 | 处理 |
| --- | --- | --- |
| exact | 当前模型及支持域内所需特征均有已识别系数 | 使用模型分数 |
| bounded | 唯一缺口为训练最大值为零、系数也为零的特征激活 | 该系数作为未知非负量，保留已知费用形成下界 |
| unsupported | 分析、证据绑定、契约或非可分解语义不支持 | 无合法数学区间，不能自动变成有界候选 |

设 L_i 为已知费用，f_ij 为未知特征计数，w_j 为未知单位系数：

```text
C_i(w) = L_i + Σ_j f_ij · w_j，f_ij ≥ 0，w_j ≥ 0。
```

选择 exact 候选 e，仅当所有 bounded 竞争者都满足 L_i ≥ C_e，且不存在无法建立合法数学界的 unsupported 竞争者时，才证明 e 对该非负加性模型族属于 argmin。若下界不足或普通 unsupported 仍存在，则回退到该模式自己的静态 incumbent。候选没有被按结果删除，证书保留全部竞争者。

证明只针对声明的模型族及已获取候选池，不是硬件时间下界，不保证模型与实际时间一致，也不证明布局全集最优。相等只证明 argmin 成员关系，不证明唯一最优。未知系数可在候选间全局共享；上述下界是充分条件，不能据此虚构额外硬件延迟先验。

独立审核重算 18 个已观察“程序×profile”组合及 400 组非负系数补全，验证普通 unsupported 未被伪装成界、未知候选未被遗漏。证据为 development/independent-partial-mathematics-review.json，SHA256 e85f933032cb4d7bd3a6a9021cf23754e6b99265e8daaeadb2659ae45bca6c51，未读取 fresh GPU 标签。

Frisk 的基础仍是自身的 GF(2) 布局、消费者敏感需求传播、转换所有权与跨 SSA 联合选择。GF(2) 的像、核、商空间和正交补等数学工具不作排他性创新声明。本阶段贡献是将部分可识别成本的可复核决策、有限预算候选保存及后端证据契约组合进该系统。

## 3. LOCAL 实验、模型与独立重算

保持旧 v1 的 14 特征格式，仅拟合 LOCAL 一项；13 个非 LOCAL 系数逐项不变。14 参数联合 refit 仅保留为开发诊断，没有选入最终 profile。

| 项目 | 已完成结果 |
| --- | --- |
| 训练镜像 | 94＝56 旧 native-supported＋16 通信＋22 LOCAL micro |
| 已观察程序中的非零 LOCAL | 2 个镜像；unsupported 原因仍保留 |
| micro 覆盖 | 1/4/16 KiB private frame，n=8…2048，实际 LDL/STL 保留 |
| 正常检查 | 66 条全部通过 |
| sanitizer | 4 个实际进程，264 条记录通过 |
| 时间样本 | 10 配对批次、220 image-block、1100 event |
| 协议增量 family 95% CI | [0.017317260699201142, 0.017584960914973635] µs/静态传输 byte |
| 非负先验 floor | 低置信端点的 10%：0.0017317260699201142 µs/静态 byte |
| 最终 LOCAL 系数 | 6418 ticks/static LOCAL transfer byte |
| 独立重算 | 单参数 rank=1；residual coefficient 0.004685733732187896；与生产差异 1.734723475976807e-18 |
| 零 LOCAL 不变性 | 77 个训练镜像分数严格不变 |
| 训练流量上限 | 70220 static LOCAL transfer bytes |
| profile SHA256 | 0449658d7769af15cd58d59bc62bde01bd5d1b336dc19268d5ba1b894ce30165 |

micro 增量包含匹配 BAR、地址计算、递推依赖等协议影响。该 CI 不是硬件 LOCAL 延迟区间，也不是最终拟合系数的置信区间。driver private allocation、静态 LDL/STL 传输字节与 spill 次数严格区分；spill 次数仍未知，不能从 frame 大小反推。

最终拟合是一个带非负约束和弱正则的配对残差参数；它是排序代理，不宣称绝对时间外推已验收。训练未使用新的整程序性能留出结果，超出训练覆盖的输入仍受注册支持规则约束。

独立审核不调用生产拟合/特征辅助函数，重算原始 event、配对关系、系数及零 LOCAL 不变性。证据根目录为 /home/baopeihua/frisk/guoqiao/optimize_plan/partial_cost_efficiency_artifacts：

- local-training-r2/{registration.json,manifest.json,completion.json,correctness.json,micro-raw.json,safety-*.json}
- local-fit-r1/{profile.json,inputs.json,fit.json,pair-ledger.json,training-ledger.json,joint-refit-development-only.json}
- development/local-calibration-independent-audit-r2.json，SHA256 3a585f0e5197f80121b77b24b8f847ad4a87325ca88e43067b05a76b428dfd77
- development/independent-LOCAL-acceptance-review.json，SHA256 1b8686e0306043fa4eb3391d17ad88c015e7842b966e7217ca9cbc052f6476b9

## 4. 一个协调流程内的有限候选 archive

旧路径为三个模型各运行 LI/J，共六来源搜索，理论总上限 393216 probes、1536 refinement、6 次编译请求。新生产候选获取使用一个 v6 prepared domain；额外 legacy prepare 只是资格基线。

联合模式不是“内部只有一次 solve”。它包含 fully-informed producer-local witness 的局部与完整修复阶段，再运行 J，共用一个不重置的 SharedSearchBudget。

| 项目 | joint archive | fully-informed local archive 对照 |
| --- | --- | --- |
| 总 probe | ≤65536 | ≤65536 |
| 总 refinement | ≤256 | ≤256 |
| witness probe quota | 16384 | 使用完整总预算 |
| witness refinement | 可用共享 256；J 用实际剩余 | 使用完整总预算 |
| 默认 archive/编译 K | 4 | 4 |
| archive 可配置范围 | 2–6 | 2–6 |

observer 默认 nullptr，未启用时保留原路径。启用时，仅在已经付费、成功且合法的完整 assignment 评分点采样，不另跑六次来源搜索。实际获得的 local/J 完整 incumbent 强制保留；无 witness、预算耗尽、preparation 不支持与 integrity failure 必须明确区分。

额外候选来自在线有限集合：按通用费用前沿和 assignment 多样性选取；主费用通常不超过较优 incumbent 的两倍，零主费用检查完整费用 tuple；强制 incumbent 不受此筛选排挤。去重基于完整 attribute assignment，不因特征或费用向量相同合并。这是启发式有限池，不保证全局最优。

entry 包含完整 assignment、指令/规约 tuple、来源、观察时预算与完整费用。导出前验证，之后可用 --assignment 零搜索 replay、物化与 ABI 导出。证书重验、replay 和物化仍计 CPU 时间，单列而不冒充免费设置。

duplicate 只统计命中当前有限池的观测；frontier/capacity discard 是观测事件，不是全部历史状态唯一计数。local component cost 可能是局部目标，最终完整目标以导出 entry.cost 为准。

开发回归覆盖六旧程序×LI/J 共 12 次协调调用、48 个 entry 的零搜索 replay/独立 strong20 检查；旧快 local witness 的 TTGIR 均保留，K=2 也保留两个 incumbent。此结果不是新留出选择质量结论。入口：

- development/archive-implementation-summary.json
- development/archive-final-binary-bridge/summary.json
- development/root-real-archive-contract-check.json

## 5. 精确编译缓存与批处理

native_compile_cache.py 以完整 TTGIR UTF-8 字节、规范化完整配置、目标/选项、注册源码/工具 payload、环境和有效 knobs 建立身份。命中时验证不可变文件集合和所有原生工件哈希；锁与原子发布避免不完整 entry 复用。资源规划和支持证明按当前 execution 重建，不直接复用其他 execution 的证书。

计数口径为 requests＝exact_cache_hits＋compiler_API_calls。API 调用可能命中 Triton 内部缓存；未知内部命中保持 null，API 次数不能当作 PTXAS 次数。warm 效率仅指编译缓存变暖，不表示搜索已缓存。

production CPU smoke 两次精确请求中，1 次 compiler API 调用、1 次 exact hit，原生工件逐字节相同，CUDA 未初始化。这证明缓存契约可用，不证明完整设置开销加速。

证据 development/native-compile-cache-production-smoke-r2/summary.json，SHA256 e8294328e88506a2b61b828d8ad73a7e93bfb3780c98c167608aadaab80e411a。独立缓存审核为 development/independent-cache-efficiency-source-review.json；最终跨阶段收据审核为 development/independent-cross-stage-schema-review-r2.json，SHA256 9ccc91f4ef71f7e503bafe559fea4079cf00c1332314d2ce34c2bbec5c913dd1。完整 cold/warm 数字待 R3。

## 6. 静态 Shared 修复与保留的失败

### 6.1 R1 效率实验是真实的编译失败

mixed 256×64 candidate-3 发射静态 .shared[65536]，PTXAS 静态上限为 49152 bytes；之前错误地拿 227328-byte total/opt-in 容量代替了当前静态后端能力。

旧路径完成 36 个 cold 请求；新路径完成 15 个，第 16 个失败，余 8 个未尝试。没有完整 cold 对比，没有 warm 结果，不能从成功前缀推完整开销收益。失败目录不改写，详见 development/efficiency-benchmark-final-failure-analysis.json。

### 6.2 修复范围与真实候选回归

C++ verifyLayoutDemandBackendSupport 只收紧当前 explicit scalar f32 执行域中 Shared scratch 上限为 49152。Python native_static_support.py 在直接编译、批量编译、缓存检索前做同一检查。一般 SM90 硬件容量不变，未来 dynamic Shared 后端不受该静态能力声明限制。

backend.py、native_cost_calibration.py、demand_coverage_audit.py、redistribution.py 四个契约逐字节未变，不因无关 bank contract 变化重新拟合 profile。

C++ 新测试先证明 GF(2) 双射和 first_owner 所有权合法，再验证 Shared32768 允许、Shared65536 后端不支持；同形状 Register65536 仍允许。拒绝的是当前后端，不是硬件布局数学非法。

| 真实实例 | 最终冻结二进制 CPU 重放 |
| --- | --- |
| 旧 div，Shared32768 | execution.json、TTGIR、配置逐字节不变，零搜索 |
| 已观察 mixed，Shared65536 | exit3；backend unsupported/static49152；零搜索，无 emission/编译/GPU |
| fresh mixed，Shared131072 | 同样在编译前拒绝；只有 CPU 能力信息，没有新 GPU 性能标签 |

既有 C++ CLI 将不满足声明执行域的证明记为该域内 no-feasible，reason 明确 backend unsupported；不能据此解释为一般 SM90 不可行。

最终证据为 development/static-shared-backend-fix/real-cases-r4-final-bin/summary.json，SHA256 49d6126e84742a31dff46a22e7c2573c13b86a5f5e5ae61358727e260117d6b5。绑定 joint binary SHA256 bfaa103b2e43988f9b4d2f361f7aa2268aae20ed7454ee0d61b41123d926df49。隔离审核将 fresh131072 的 CPU 能力暴露与未见 GPU 性能严格分开。

### 6.3 R2 是调查误读导致的主动中止

R2 在 native 编译和 GPU 前被 root 中止。最初认为生成行缺 conversion；对 live、R1/R2 快照及 replay 源码的逐字节和 AST 核查证明，它们本来就含 selection='J'、conversion='C'。这是调查误读，不是产品缺陷，没有由此修改源码。

旧中止记录仍保留，必须结合 development/source-r2-stop-reason-correction.json 阅读；更正 SHA256 b6cd25569cb3be4a514dbad443fafe3c67ba6f8e04026244a97252616bdb306f。R3 继续使用相同 source-freeze-r2、模型与协议。隔离审核证明记录中的 R1 整程序阶段没有新 GPU 时间；它不声称排除未记录的外部实验。

## 7. 已完成验证与证据入口

以下路径均相对 /home/baopeihua/frisk/guoqiao/optimize_plan/partial_cost_efficiency_artifacts。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| C++ build | production Ninja＋显式 GoogleTest targets | development/static-shared-backend-fix |
| LayoutDemand 专项 | 28/28，含新增 2 回归 | development/static-shared-backend-fix/focused-final.xml |
| 全布局单测 | 677/677 | development/static-shared-backend-fix/05-all-unit.xml |
| 选择/Oracle/archive | 53/53 | development/static-shared-backend-fix/06-all-selection.xml |
| 最终 R2 集成 | Ninja、13/13 CTest、80/80 lit | development/integrated-validation-r2/summary.json |
| Python static support/cache | 36/36 | development/static-shared-python-domain-r2/summary.json |
| Python support/qualification/fixture | 33/33 | development/root-static-support-integration-tests-r2/summary.json |
| 部分成本单测 | 25＋2 个已观察回归 | development/partial-decision-validation-r1/summary.json |
| 独立数学 | 18 组合＋400 系数补全 | development/independent-partial-mathematics-review.json |
| 独立 LOCAL | 94 镜像、262 pairs、356 rows、1 拟合维度 | development/independent-LOCAL-acceptance-review.json |
| 跨阶段 schema/收据 | 审核通过，未代替实际 GPU 验证 | development/independent-cross-stage-schema-review-r2.json |
| source-R2 隔离/预注册 | 快照及旧失败隔离审核通过 | development/independent-source-r2-isolation-preregistration-review.json |

关键摘要 SHA256：

- integrated-validation-r2/summary.json：e10a356299f990b5072ca7889b38a05e4bcd13150b9ae4b43df0680b7e351313
- static-shared-backend-fix/implementation-summary.json：2551ba4cf1fa797d6c2db1d63d09b763291d1888fa55e0b5f45465baecd80f88
- static-shared-python-domain-r2/summary.json：82d8deee1fdab874909c87b213f6a153f66638c31d97fd12fa7634cea040dce9
- independent-source-r2-isolation-preregistration-review.json：e00db3c981bdf49abc7fe962cbdd0277248e5bf4996b05c784f923bca761cb64

上述测试不代替正式 GPU 正确性、安全检查、原始时间统计、选择 regret 或完整开销验收。

## 8. 正式 R3 结果：功能与开销通过，选择质量失败

### 8.1 GPU 质量与独立复算

75候选/9规格全部通过。3159输入偏移组合、12015正确性记录（9477 ordinary＋1269 nondefault＋1269 capture）、216个真实Sanitizer进程/792安全记录、21行精确安全复用均独立对账。资格编译75请求=54 API＋21精确命中。固定27个partial decisions为11 exact-ranked、1 interval-certified、15 fallback；99个成本区间与证书全部复验。完整测量90配对批次、540 image-block、2700 events，无无效批次或替换。

| 质量指标 | 结果与95% CI | 验收 |
| --- | --- | --- |
| 主策略/legacy耗时 | 1.403909 [1.399773,1.418369] | 失败，整体约慢40.4% |
| 六fresh有限池regret GM | 1.803837 [1.790669,1.810135] | 失败，门槛1.03 |
| 最差regret | 12.691973 [12.580029,12.783982] | 失败，门槛1.10 |
| 9项回归guard | conflict失败，其余8通过 | 失败 |
| J/同预算充分知情LOCAL | 0.594462 [0.593821,0.603157] | J相对LOCAL较快，但不能替代legacy与regret门槛 |
| strict→partial | 0.995252 [0.993360,0.996040] | remat由回退变证书，六例总体改善约0.47% |
| partial→LOCAL校准 | 1.0 [1,1] | 本轮选择完全相同，未确认额外性能收益 |

| fresh程序 | 主选择median µs | 池内最快median µs | 配对regret |
| --- | ---: | ---: | ---: |
| diamond128×128 | 12.073 | 12.061 | 1.0000 |
| remat256×64 | 13.636 | 13.636 | 1.0000 |
| conflict128×128 | 157.508 | 12.387 | 12.6920 |
| mixed512×64 | 309.680 | 127.056 | 2.4889 |
| div128×128 | 77.123 | 75.827 | 1.0160 |
| exp128×64 | 11.737 | 10.936 | 1.0734 |

regret是配对批次比率的统计量，不能用两列median直接相除重建。compiled reference只限注册archive＋baseline，并非全布局GPU最优。R3从此是已观察数据，不再标为fresh用于调参后验收。

失败归因：conflict的c0/c2/c3拥有完全相同14维native特征及23631078 ticks，native按声明顺序选c0；但GF(2)已识别的global-sector负担为98304/7680/7680，warp内相邻lane地址跨度512B/4B/4B。只重调这14个系数无法区分同向量的快慢候选。c1为23875002 ticks，实际却约12.39µs。它是native评分/平局决策遗失已有地址分布信息，非搜索没找到快候选、LOCAL缺失或证书求解错误；尚无硬件计数器单因果实验，不能把12.8事务比直接解释成12.7时间因果。

mixed四个J候选因原生CFG超域均回退，静态主费用相同149814272，最终conversion count的0/1平局规则选c1，未利用已编译LOCAL footprint 6928B/80B（c1/c3）；footprint不是流量或spill次数。exp的最快local-c2不在J池，存在K4候选覆盖余差。

额外资格生成2545.648s、75候选批量编译3347.899s、完整验证3402.197s、native分析170.179s、计时101.464s，含LOCAL与legacy额外公平对照，不能冒称主策略部署费用。大mixed对照的Shuffle仍按256²展开；每target实际source-reg support仅32/rank5，等价稀疏化可将每route65536条shuffle降到8192，尚未实现，不声称8倍编译/GPU加速。

证据：run-partial-cost-efficiency-r3-20261103/{quality-summary.json,group-quality.json,partial-decisions.json,measurement/completion.json}；development/independent-{premeasurement-functional-audit-r3,partial-raw-statistics-r3,per-case-raw-details-r3,selection-failure-classification-r3}.json；development/{selection-failure-diagnosis-r3,shuffle-code-size-diagnosis-r3}.{json,md}。

### 8.2 完整 CPU 开销：独立验收通过

六个已观察程序全部完成，固定 CPU 0..3；同一冻结代码、工具和输入域，旧三模型×LI/J 六来源对比单 v6 共享预算归档。以下为单次描述性测量，未给出统计 CI，不是全机器独占结果；root 资格任务使用 CPU 4..191，内存/I/O 仍共享。

| 模式内完整费用 | 旧流程 | 新流程 | 新/旧 | 加速 |
| --- | ---: | ---: | ---: | ---: |
| 生成（prepare/search/replay/emission/I/O） | 471.825 s | 277.580 s | 0.5883 | 1.70× |
| 冷编译 | 1050.879 s | 171.357 s | 0.1631 | 6.13× |
| 生成＋冷编译 | 1522.704 s | 448.938 s | 0.2948 | 3.39× |
| 热编译，仅 compile | 457.605 s | 107.730 s | 0.2354 | 4.25× |

实际搜索为36→6次、probes156303→52034、refinement3166→990；新生成包含92.844s零搜索重放。编译每轮36→24请求、36→1 worker。冷轮旧36/new24 API，双方精确命中均为0；热轮旧仍36 API（Triton内部命中未知），新0 API＋24 exact hits。**冷轮收益不能归因精确缓存命中**；只能归于更少来源/请求、批量会话及不同受限候选池避开旧巨大编译的组合，未做单因素因果消融。

生成共同外层1.931s、cold编译后置审计1.631s以及共同注册/父CLI启动单列；上表是各模式完整生成与编译路径之和，不是包含共同开销的整个workflow端到端比率。热轮没有重跑搜索。native分析、数值/四Sanitizer和GPU kernel时间另列。未用这些旧输入的设置费用混合不同fresh输入的kernel时间计算摊销次数。

完整120请求（72 old、48 new）、74个worker、2707文件哈希、6 archive/24零搜索重放、cold→warm缓存链独立对账通过。全部已观察worker/线程亲和性为0..3，CPU编译无CUDA context。24个新warm键与CUBIN逐项对应cold。逐程序费用及普通内部缓存未知状态保留在原始summary。

效率阈值解释登记在完整cold/warm结束前，但generation及部分CPU结果已经可见；它是对既定“降低开销”的操作解释，不冒称所有CPU规则均在实验前预注册。全部六例聚合generation、冷模式路径和热compile均下降，因此当前完整CPU效率验收通过；GPU选择质量仍独立待验收。

证据：development/efficiency-benchmark-r3/summary.json（SHA256 d26951dfc9d103880dddd439239db46f11fb7c56334aea4b7f72452926b55226）；development/efficiency-workflow-r3.json（c8dc171394db6aaaf98fc4ffeb94df2b06fed14aa03f075a4cec63de14116c8f）；development/independent-complete-efficiency-acceptance-r3.json（097721fdd57badb99977c5be9fbf99680cfa318872ab3376263e39f752f8ee9b）。

最终状态：IMPLEMENTED_RETESTED_NOT_QUALIFIED。本轮性能失败完整保留，默认继续legacy。后续在新的源版本与留出集修复通用平局信息损失及稀疏Shuffle，不在R3上调参后冒称fresh。

## 9. 封存

本阶段contained证据树：/home/baopeihua/frisk/guoqiao/optimize_plan/partial_cost_efficiency_artifacts；evidence-index.json封存全部文件与本报告外部哈希。旧三个sealed阶段保持精确文件集与哈希，见development/old-sealed-preservation-final.json。未提交或推送代码，HEAD与Git index保持原值。后续实现使用新证据目录，禁止回写此阶段。

最终统计交叉核验：development/independent-final-statistical-crosscheck-r3.json，SHA256 87779ce0e5dbd474e0513fa8dd4d31a399cc42f46dacad9b5a91b31082694b10；172项汇总、1522项逐组数值比较的最大差均为0，151项familywise图像排序关系一致。
