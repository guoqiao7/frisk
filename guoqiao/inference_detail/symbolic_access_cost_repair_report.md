# GF(2) 符号化访问成本与转换证明修复

日期：2026-11-01（Asia/Shanghai）。仓库：H100_direct:/home/baopeihua/frisk。基线为搜索开销优化交付后的未提交工作树，Git HEAD 9f79155ac08f82133f4ddb6554af552feefca8d2。本阶段只在服务器修改，不创建本地副本，不提交或推送。

## 结论与问题定位

这五个失败是同一类分析覆盖缺口的真实实例：合法布局的硬件 holder 数超出访问成本分析 65536 枚举预算，首个完整候选的成本失败被求解器 fail-closed 传播为整个搜索失败。本轮以 GF(2) 商空间、规范截面和正交补替代相关枚举；没有删除复制布局、改小原始形状、提高搜索预算、跳过未知费用或改变算法/ABI/精度。

| 保留的原配置 | 第一个已重建的超限访问 | hardware topology | holders |
|---|---|---|---:|
| diamond-32x32，legacy | c14，vars 6/7 | [1024,32,4,1,1] | 131072 |
| remat-128x16，legacy | c16，vars 7/8 | [2048,32,4,1,1] | 262144 |
| mixed-128x32，legacy | c15，vars 7/8 | [4096,32,4,1,1] | 524288 |
| conflict-256x8，legacy | c16，vars 7/8 | [2048,32,4,1,1] | 262144 |
| diamond-32x32-v2-guard | c13，vars 1/9 | [1024,32,4,1,1] | 131072 |

第五项保留历史 fixture ID；本轮加载原样冻结 v2 系数文件，实际成本语义已经版本化为 calibrated-v3。逐例方法、冻结 trace/input SHA 与赋值重建在 ../optimize_plan/symbolic_access_cost_artifacts/research/five-natural-cost-failures-attribution.json；不能仅从 diamond 类推其余案例。

## 实现与数学性质

1. **读：陪集分区与精确重数。** 对执行映射 L:H→V，先证明满射及复制度一致。一次物理 issue 内自由位为 F，令 U=im(L|F)。其逻辑值集合是 L(k)+U；每个陪集的 issue 重数是 2^[dim(H)−dim(F)−dim(V)+dim(U)]。只枚举现有已证明 footprint 的逻辑地址，按陪集分组，再乘重数。bank phase 另用自身自由子空间。这样会去掉同 issue 内重复地址，但仍收费跨 warp/issue 的复制工作。TMA 保留单 bulk issue。
2. **写：核的规范截面。** 按 canonical hardware 低位到高位选独立列并保存 preimage。所有被舍弃 dependent bit 设零，给出每个逻辑值数值最小的硬件 owner，与旧 first_owner 穷举一致。它不仅处理零列，也处理多个非零列相关的复制。
3. **转换：像包含与正交补。** 目标可变列及固定 carrier 的源/目标差列若全在源 register 像中，则是 Register；扩大源自由子空间到 register+lane 后成立则是 Shuffle，否则在共同完整 CTA 内是 Shared。利用 v∈im(U) ⇔ 对所有 n∈ker(Uᵀ)，n·v=0，每个范围只做一次消元，后续使用奇偶点积。删除原转换分类的 65536 逻辑点、262144 holder 预算；保留实际表示、物理拓扑和覆盖前提。

地址本身仍取真实 storage footprint，保留 strided/view、未知 base byte phase 的最坏值、packed 元素及 active tail 语义。常规二次幂 chunk/phase 的访问成本不再展开 hardware holders。所有计分乘数采用显式饱和，并保留 observer/cache/v2 系数应用顺序。

详细证明和进一步数学应用在 ../optimize_plan/symbolic_access_cost_artifacts/mathematical-design.md。包括 bank 子空间交公式的前提，以及为什么 rank 相同不意味着 candidate 可全局合并。

## Triton 参考与进一步设计

官方源码固定为 Triton commit 0861b90f695e3c866f6f7eadd4be8339378ab017（2026-10-31），研究中核对了：

- [LinearLayout.cpp 的自由变量与零基处理](https://github.com/triton-lang/triton/blob/0861b90f695e3c866f6f7eadd4be8339378ab017/lib/Tools/LinearLayout.cpp#L1167)：RREF/free-variable 是处理复制的参考。
- [GenericSwizzling.cpp 的子空间交](https://github.com/triton-lang/triton/blob/0861b90f695e3c866f6f7eadd4be8339378ab017/lib/Tools/GenericSwizzling.cpp#L244)：满足地址线性前提时可用交空间维度描述 bank 冲突。
- [NVIDIA load/store lowering](https://github.com/triton-lang/triton/blob/0861b90f695e3c866f6f7eadd4be8339378ab017/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/LoadStoreOpToLLVM.cpp#L1014)：广播和冗余线程处理依访问契约而定，跨 CTA 冗余读不能随意抹掉。

本轮进一步在 Frisk 的精确 issue 成本和 first_owner 契约下推导商空间重数、最小 preimage 与完整转换范围证明；不是直接接入 Triton 成本模型，也不宣称这些数学工具本身具有首创性或已证明领先性能。

## 成本模型版本与充分知情基线

访问压缩在旧支持域内保持原评分。符号转换分类则可能把原先非零 Conservative 估计证明为更便宜的路线，因而完整语义 ID 升级：sm90-layout-lex9-legacy-v2、sm90-layout-lex9-static-trip-count-v2，以及 sm90-layout-calibrated-v3-<profilehash>-leg/stc。

profileModel 仍为 sm90-layout-calibrated-v2，schema/serialize/parse/hash 不变。冻结系数文件 SHA256 为 6b4947af82c2748fc8db75268ddb621fc45852d238b63deee12cf17c6cde7388。未重新拟合系数，也未扩大旧模型精度结论。充分知情局部基线的版本识别已同步；使用真实 MMA 图和四种实际工厂目标验证 geometry-only 依赖，未知目标继续保留全部角色依赖。

## 验证记录

- 最终构建：all、FriskLayoutUnitTests、FriskLayoutSelectionTests；388 份构建范围源码在构建前后完全一致。
- 完整 native 单元 630/630（72 suites）；选择测试 45/45；CTest 10/10；搜索驱动测试 14/14。三个补查文件修改后，消费者/单元目标构建为 no-op，五个工具 SHA 全部不变；再次 CTest 10/10、相关 native 128/128（12 suites）通过。
- 新增 17 个 native 测试，包括随机满秩混合 GF(2) 映射与独立旧穷举 oracle、非零相关列、carrier 顺序重排、first_owner、向量/phase、TMA/cp.async、65536 边界、大复制、饱和、拒绝错误覆盖/复制度以及充分知情基线契约。
- 额外独立穷举：42 个满秩 2×3 映射，5376 个 read/free/active 分组对照，168 个最小 owner 对照，3528 个源/目标 scope 包含对照，全部一致。
- 正式 CLI 矩阵固定在 ../optimize_plan/symbolic_access_cost_artifacts/run-natural-support-20261101T150407Z：16 原配置 × 新旧 binary × cache on/off = 64 次真实 native 执行；54 exit0，10 次旧版原失败配置保持 cost-failure。新版 32/32 成功，其中原五失败配置 10/10 committed。
- 原 11 成功配置输出 IR、布局赋值、成本向量和逻辑搜索保持相同；原新元数据差异由独立逐 path 审查，只允许已明确版本化的 modelID/coverage。新旧内部 32 个 cache pair 严格等价。

正式 FriskOpt SHA256：8387de44b068d40d0f7a5a9018b8427f8d812ddae080159a9436f46c1aa22b6f。测量用 376 文件 recipe 源码 digest：6667939d001eb8c15cbd166b606412fbeba6ef967c62878820fba2483b0133e3。两 binary、五个动态运行库和测量范围源码在正式运行前后完全一致。结构化逻辑事件/成本贡献零遗漏；trace header coverage 摘要受既有 128 字符上限约束并标记 summaryTextTruncated，solver-options 事件保留完整 coverage 文本。

四个转换 fixture 已用真实 LLVM lit/FileCheck 执行全部 38 条 RUN，4/4 通过；补查中更新的三个测试文件在正式矩阵结束后修改，仅涉及模型 ID、coverage 与 provenance 期望。`formal-to-final-source-bridge.json` 和 `build-to-final-test-only-bridge.json` 分别明确 376 文件测量 recipe 与 388 文件构建/归档口径，其他源码全部相同。FriskOpt 字节 SHA 不变，未把补查文件冒充成参与正式测量的版本。

最终 14 个源码/测试文件补丁已通过 git apply --check、应用及逐字节 roundtrip；前像与上一阶段搜索优化冻结源码完全一致。独立组件审查状态 APPROVED_BOUNDED_SYMBOLIC_ACCESS_AND_SCOPE_PROOFS，0 Critical/0 Important；正式矩阵逐 path 语义审查状态 PASS_BOUNDED_SEMANTIC_SUPPORT_REGRESSION。原始 result.json 保留严格字节跨版本比较的 needs-review 状态，由 `run-natural-support-20261101T150407Z/reviewed-semantic-result.json` 解释允许的具体元数据差异，不覆盖原始数据。

## 保留的边界

访问成本仍依赖已证明的 storage footprint；逻辑/padded 域上限 65536，代数表示上限 256 bit，外层合法性/资源数值检查仍生效。非二次幂 chunk/phase（如 i24/i40/i48/i80）保留旧精确 bounded fallback；超限时不会当作零成本。Copy vector、Reduction、MMA、GPU export 的其他独立枚举上限未取消。真实未建模成本仍 fail-closed，typed 成本失败诊断尚未改造。

理论收益是分析复杂度和覆盖范围不再由常规复制 holder 展开决定；本轮是 CPU 正确性与编译支持验证，单次 wall 只诊断，不形成速度比、GPU 加速或优于 Triton/TileLang 的性能结论。

## 证据与原始阶段保护

本阶段 artifacts 根目录为 guoqiao/optimize_plan/symbolic_access_cost_artifacts。四个旧阶段索引字节未变，旧 stage 内共 20550 个封存文件、4813984283 字节全部按原索引校验通过。活跃工作树源码属于新版本，旧阶段源码仍通过各自冻结快照恢复。初次测试夹具零位宽错误、未重链 consumer 身份错误及开发运行的归档 binary 执行权限遗漏均保留原始日志；均不计入本阶段正式通过样本。

## 最终交付状态

本轮限定范围实现与正确性验收完成。补丁、最终 388 文件源码快照、正式 FriskOpt、逐例原始运行、数学穷举和独立审查归档于本阶段 artifacts，统一清单为 evidence-index.json（不自包含自身哈希）。原 Git HEAD 不变，暂存区为空，本阶段未提交或推送。后续公平性能比较需要使用新语义模型身份并单独冻结版本；本报告不扩展为 GPU 性能验收。
