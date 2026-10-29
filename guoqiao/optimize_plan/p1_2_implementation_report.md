# P1.2a–P1.2b 实施与验收报告

状态：P1.2a→P1.2b 已完成实现、完整回归、串行测量及独立源码/证据/文档复审，最终 readiness APPROVED，无遗留发现项。执行范围见 [计划](p1_2_execution_plan.md)，原始数据见 [测量 JSON](p1_2_measurement_results.json)。本轮未实施 P1.3/P1.4，未自动提交或推送。

## 1. 复现基线

- 直接在源目录 `main` 工作，HEAD `b23791101d095023cdd3033c8751b2fbbd81b4e6`；开始时暂存区为空。基线包含未提交 P0/P1.1，因此仅 checkout 此提交不能复现本轮起点。
- 修改前完整源码快照：`/tmp/frisk-p12-baseline-vZeQIU/source.tar`，SHA256 `1802dadefe62c24b60e1501de00195948d1c20e7be413dd6f15b1ec1844f1c1e`。快照包含 include/lib/unittests/test/tools/benchmark/guoqiao 和根 CMakeLists。
- 修改前 Release 二进制：`/tmp/frisk-p12-baseline-vZeQIU/frisk-opt`，SHA256 `ad69cb595ee62f689a03819f2e5f4863a98560d0b4c3a5d3a88e26c075659853`；对应配置同目录 `CMakeCache.txt`。动态链接检查仅发现系统/C++ runtime，无随本轮重编译变化的 Frisk/MLIR 动态库。
- 不创建工作树、不自动提交或推送；原有修改保留。P1.3/P1.4 不在本轮实现范围。

## 2. 验收状态

| 项目 | 当前状态 |
| --- | --- |
| Copy 细分诊断与固定 ABI Copy→MMA | 已实现，Release 局部 52/52 unit、2/2 新 lit 通过；独立复审通过，诊断坐标建议已补齐 |
| 单次推断 profile、缓存命中与开关等价 | 已实现并通过复审；三类输入 IR/非计时 stats/完整 trace 开关等价，54 次串行调用完成 |
| Auto 与第五 geometry 变量、联合选择 | 已实现并通过局部测试；显式 RS/SS 超预算状态问题已修复，完整源码复审通过 |
| 完整 Debug/Release 回归及独立复审 | 505/505 Debug unit、69/69 lit、8/8 Debug CTest、2/2 Release CTest、Python/API/tool 通过；源码复审通过 |
| 串行 CPU 推断开销与身份记录 | 54 次采样调用＋9 次 trace 调用通过；原始值、哈希、驱动与检查结果已入库 |

本轮不运行 GPU Attention，也不把静态成本下降称为运行性能提升。

## 3. Copy 构造与集成（局部验收完成）

增加 `InstructionContractBuildResult` 及 `buildInstructionContractDetailed`；Copy 专用入口为 `buildSM90CopyContractDetailed`。结果同时携带 binding、LayoutProof、稳定 reason code 和可选 operand。原 `FailureOr<Attribute>` 接口保留为适配器，支持表使用已有证明结果，不再重复验证生成的 Copy 契约；旧 target 的显式 binding 仍走原有 verification-only 语义。

诊断区分 shape、direction、target、alignment、address-contiguity、writer、descriptor、budget、decoder 和 binding。已有下层 proof 的状态、原因和坐标保留；理由分类目前依赖既有 reason 字符串，不改变证明真假。存在真实 Unknown 的无支持候选集仍报告 Unknown，不误称硬件不可实现。

trace 新增惰性的 `instruction-rejected` / `instruction-supported`，含 path、状态、code、端点与坐标；最终选择仍由 solution 事件确定。沿用原 event/byte 上限。新增测试直接解析 JSON，检查这些字段、截断和已支持但未选中计划与最终解的区分。

固定 ABI 用例为 Global A/B `[64,16]f16` 行主序 alignment128 → Shared alignment1024 → SS MMA 64×64×16、transB、128线程 → Global `[64,64]f32`。三条 Copy 路径分别使用受检 helper 生成的完整 binding，不伪造 path-only 字典。测试逐坐标检查声明及实际物化 view 的外部地址，检查数据流、实际 verifier 和打印/解析/再次推断的稳定性。

负例覆盖根对齐、偏移一个 f16、同线程非连续地址、TMA stride、错误方向/目标、重复 writer、issuer/completion/descriptor 篡改；区分有合法 fallback 与硬绑定下的原子失败。真实超 65536 hardware 点的 proof 为 Unknown；另外测试逐项保留下层坐标。

局部复现命令：

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='SM90CopyContractTest.*:CopyMmaIntegrationTest.*:GuardedCopyTest.*:InstructionConstraintTest.*:LayoutTraceTest.*:GemmLayoutIntegrationTest.MissingAndTamperedContractsAreRejected'
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 /home/baopeihua/mlir-tutorial/llvm-project/llvm/utils/lit/lit.py -v build-m6-release/test/Transforms/copy-mma-contract.mlir build-m6-release/test/Transforms/copy-mma-contract-errors.mlir
```

最终局部结果：52/52 unit（24.987 秒）、2/2 lit（4.90 秒），构建与命令均 exit0。独立任务复审 spec PASS、quality APPROVED，未发现 Critical/Important。非阻塞建议为最终 TMA destination-descriptor 失败补充 mismatch 坐标，已在 §6 整体修正中实现和验证。

RED/修复记录：原对齐案例实际丢失错误原因；仅加兼容接口时详细状态/端点/坐标测试继续失败；实现后通过。接入详细 target 接口时，既有显式绑定测试发现额外 builder 调用，修复后保持零 builder 调用。测试开发中修正了物化后内部 storage declarations 的计数假设、混合预算 Unknown 的断言，以及副本数不一致的负例元数据；没有放宽生产证明。中间一次 JSON `Expected` 的 const/bool 编译错误也已修正。

较大范围重编译出现既有 CMP0116 deprecation 和未修改测试中 LLVM APInt 的 GCC `-Wfree-nonheap-object` 警告；Task 1 最终增量编译无警告。不把大范围构建描述为无警告。详细过程与独立复审保留在 `/tmp/frisk-p12-task1-report.md`、`/tmp/frisk-p12-task1-review.md`；Debug/全量验收见 §7。

## 4. 单次推断 profile

独立 pass 和完整 pipeline 增加默认空的 `profile-file` 选项。例如：

```bash
build-m6-release/bin/frisk-opt test/Transforms/attention-layout-fixed-abi.mlir --frisk-layout-pipeline='profile-file=/tmp/layout.profile.json' -verify-each -o /tmp/layout.mlir
```

观测对象是固定的 11 项计数器，不保存 IR 指针或额外缓存；仅一次调用有效。显式 pipeline observer 传给内层 infer，独立调用则隔离；RAII 恢复 thread-local 状态。关闭时不增加 profile 的时钟读取或求值，原有 stats 计时保持不变。

| metric | 真实观测位置 |
| --- | --- |
| address_enumeration | 实际 storage 地址候选枚举 |
| execution_enumeration | 实际线程/寄存器坐标枚举 |
| access_cost | SM90 访问成本计算 |
| conversion_classification / conversion_choice | 转换分类 / 选择 |
| instruction_support / instruction_contract | 支持表查询 / 完整 tuple 构造 |
| relation_query | 约束求值 |
| relation_cache / alias_pair_cache / footprint_cache | solver 关系缓存 / alias pair 证明缓存 / 地址表缓存 |

JSON 固定 `version=1`、`timing_kind=inclusive`、`entries_kind=insertions`。每项含 `count/inclusive_ns/hits/misses/entries`，采用饱和计数。`entries` 是累计成功插入次数，不是 live/peak 容量；缓存重建/清空后可大于当前大小。计时包含嵌套调用，不得直接相加当作总时间；整体耗时仍读取 stats 或外部计时。

输出采用临时 sibling 完整写出再原子替换；写出失败不提交 IR。profile 与 stats/trace 的规范化路径、软链接及硬链接冲突在改写前检查；没有新增“多个输出文件共同原子提交”的承诺。

局部验收包括 6 个新 Profile 测试：嵌套/线程隔离、真实缓存 hit/miss/insertion、清空后 insertion 语义、路径别名、临时文件清理、独立及 pipeline 写失败回滚、成功/失败后的 observer 恢复、重复调用。开关比较检查 IR、除 timings 外的完整 stats 和逐字节 trace；后两者覆盖候选、完整成本和探测数。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt -j4
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutProfile*:LayoutStatistics*:LayoutTrace*:LayoutAliasAnalysis*:LayoutConversionChoice*:LayoutCostModel*:LayoutCandidateSolver*:InstructionConstraint*:ExecutionLayoutProof*:SM90CostModel*' --gtest_brief=1
python3 /data0/xiebaokang/rocm-llvm-project/build/bin/llvm-lit -v build-m6-release/test/Transforms/layout-profile.mlir build-m6-release/test/Transforms/layout-trace.mlir build-m6-release/test/Transforms/layout-statistics.mlir build-m6-release/test/Transforms/layout-relaxation.mlir build-m6-release/test/Transforms/copy-mma-contract.mlir
```

结果：105/105 unit、5/5 lit，均 exit0；独立 spec PASS、quality APPROVED，无发现项。CLI RED 是旧工具明确拒绝未知 `profile-file`；实现中修正 JSON 构造/字符串转换/缺少 include 的编译错误。一次探索命令使用不存在的 lit 文件名而出现 discovery warning，最终命令已更正。测试中的两条 solver negative diagnostics 属于已有负例，未当作新编译警告。

源码复审及局部验证证据：`/tmp/frisk-p12-task2-report.md`、`/tmp/frisk-p12-task2-review.md`。小图/Copy-MMA smoke 只证明统计入口被触发，不据此作开销或加速结论；两类 Attention 与小图的统一串行测量见 §8。

## 5. 有界 MMA geometry 联合选择（局部验收完成，任务复审通过）

显式开启方式为 `policy = #frisk<gemm_warp_policy Auto>`；省略 policy 仍是 Square，FullRow/FullCol 不变，旧 gemm、旧推断 helper 和 normalization 入口均拒绝 Auto。

MMA 原四个数据角色增加第五个 Instruction-kind geometry 角色。候选 dictionary 保存 kind/version/shape/threads/grid/atom/repeats，并由实际 operation 的合法域校验；最终 IR 仍以完整 `MmaInstructionContractAttr` 为真值，不新增一个与它可能脱节的隐藏选择。独立 materialized verifier 从实际 binding 重建第五角色并再次检查。

`getSM90MmaGeometries` 稳定枚举至多四个 grid，每 grid 保留现有最大合法 atomN，不搜索新线程数或全 atom。旧单值 API 保留确定性策略语义，对 Auto 明确失败，避免误取第一个候选。目标中立的 fragment precheck 按真实 carrier 坐标判断；Unknown 不用于删掉候选。生成候选前尊重 SameLayout 连通的硬约束，保留重排但等价的 seed，互相冲突的硬约束仍失败。

先按 geometry 筛选 fragment，再枚举配套 storage，避免直接展开 `4^5`。每 op 完整构造/证明尝试至多 256，失败也计数；需要第 257 次时返回明确 Unknown 并清空不完整表。每 role 仍至多四候选，不静默丢第五个硬 seed。stats 的 `preparation` 分别导出 `instructionCombinations` 和 `instructionFragmentPrechecks`，trace 记录 geometry 身份、预检查与完整尝试，profile 仍只包围实际 builder 调用。

成本从选中的受检 binding 读取，不再重算旧默认几何。128×128×16、256线程的 SS 核心例：

| grid | atom | repeats | MMA 第一项贡献 |
| --- | --- | --- | ---: |
| (1,2) | (64,64,16) | (2,1,1) | 4 |
| (2,1) | (64,128,16) | (1,1,1) | 2 |

下游 Reduce→store 的独立穷举遍历使用相同冻结域、硬约束和**完整 target＋conversion/rematerialization 成本**。四个案例均选择 (2,1)：

| Reduce axis | 固定 Square consumer | 完整九项成本向量 | 合法穷举 assignment 数 |
| --- | --- | --- | ---: |
| 0 | 否 | `[16290,32,0,0,160,8192,63,0,0]` | 128 |
| 0 | 是 | `[16290,32,0,133120,208,8192,31,1,0]` | 32 |
| 1 | 否 | `[16274,16,0,0,130,8192,3,0,0]` | 128 |
| 1 | 是 | `[16274,16,0,133120,196,8192,7,1,0]` | 32 |

这些均是静态代理成本，第四项不能直接当作实际 GPU 时间或纯字节数。固定 consumer 案例验证实际存在非恒等转换、外部 storage 不变和完整 pipeline 重放。每例 8 次 fragment 预检查、16 次完整尝试；RS 独立案例为 12 次预检查。**没有展示 consumer 导致赢家反转，不宣称已经找到该证据，也没有修改权重制造它。**

最终局部验证：37/37 focused unit；随后仅强化 schema 篡改断言的 2/2；23/23 既有集成/证明回归（与前者重叠，不相加）；8/8 focused lit。live fixture 同时含 MMA→store、两轴 Reduce→store、三个固定外部 ABI，完整生产 pipeline 两次运行后文本一致；其图为 16 variables / 29 constraints，16 完整尝试 / 8 预检查 / 118 generated proposals。证据在 `/tmp/frisk-p12-task3-focused.xml` 和 `/tmp/frisk-p12-task3-mma-stats.json`。

过程记录：最初 parser RED 用了错误的 dotted enum 语法，属于测试搭建错误，不算干净的功能 RED；另有测试移除 ODS 默认 property 导致 getter crash，已修正测试。根代理后来用保存的旧二进制运行正确语法的新 live fixture，旧版 exit1，明确只接受 Square/FullRow/FullCol；这是新增能力的基线对照，不追认成实现前 TDD。四-grid 重排 hard seed 曾真实失败于候选超限，追踪到 SameLayout peer 未过滤，修复后 SS/RS 均通过。oracle 曾遗漏 conversion scoring，已修正为一次完整计费；没有改生产权重。一次 lit 与链接重叠的失败已在构建结束后重跑，不当作通过。

完整 handoff：`/tmp/frisk-p12-task3-report.md`；独立任务复审 `/tmp/frisk-p12-task3-review.md` spec PASS、quality Approved。最终全量验收与 CPU 对照分别见 §7、§8。

## 6. 整体验证中的边界修正

完整增量审查使用实际 dirty baseline，而不是 HEAD；未把 P0/P1.1 当作本轮新增。审查发现：Auto 的生成域会排除 RS 复制后超 65536 hardware points 的 grid，但显式 contract 解码也复用了这个列表，使“尚无法证明”错误变成“几何不合法”。

具体反例：RS f16→f32，M=N128/K512，256线程，grid(1,2)、atom(64,64,16)、repeats(2,1,32)。A/B 各 65536 个逻辑元素，仍在逻辑支持域；A 因复制而有 131072 个硬件点。旧 Square 的 fragment proof 返回 Unknown，新 Auto 显式计划却会返回 Disproven。这是证明状态分类缺陷，不是错误执行被放行。

修复要求：保留默认生成域和 65536 上限，识别有效但超预算的显式 geometry 并保留 Unknown/budget 原因；不能把它替换为其他 grid，也不能将真正非法的 geometry 一律改成 Unknown。另集中处理 TMA destination mismatch 的坐标诊断建议。

首次 Debug 完整构建通过。首轮四分片全量 unit 和完整 lit 在发现该问题后主动中断（exit130），不计通过；同轮 CTest/Python/API/tool 检查通过，但没有用来代替修复后的重跑。最终结果见 §7。

第一轮修正把 semantic geometry 解码与默认候选的证明预算过滤分开，并保留 TMA unswizzled alternative 的代表坐标。两项新回归真实 RED 为 0/2，通过修复后为 2/2；扩展 Release 38/38、Debug 36/36（两项昂贵 oracle 在 Release 通过）、两套各 4/4 lit 均通过。修复复审认可原 RS 和 TMA 问题已纠正，但发现提前解码会遮蔽旧的逻辑点数上限诊断：显式 SS M256/N512/K16、128线程、grid(1,1) 的结果有 131072 元素，应保持 Unknown/logical budget。补充回归和修正前，随后启动的 root 构建也主动停止（exit130），没有误计为最终通过。

后续修正已完成：先执行原有 overflow-safe 逻辑点数检查，再解码显式 geometry；保留 RS 复制后硬件点数 Unknown 路径。新增逻辑上限回归在改生产代码前真实失败（2 项中 1 失败），支持域内非法 grid 对照原本即通过。修复后逻辑上限、非法 grid、原 RS 硬件上限、TMA 坐标共 4 项在 Release/Debug 各通过 4/4。两轮没有放大候选域或证明预算；TMA 坐标明确标记为 `swizzle_bytes=0` 的代表 mismatch，不冒称单点否定所有 swizzle。

最终源码复审 APPROVED：原 I1、检查顺序回归和 M1 均关闭，无 Critical/Important 遗留。证据见 `/tmp/frisk-p12-whole-fix-report.md` 和 `/tmp/frisk-p12-whole-review.md`。源码冻结后完整回归、串行测量与最终文档核对分别验收，不把局部通过等同于全部完成。

## 7. 最终冻结版本与验收命令

最终 Debug 全部默认目标与单测、Release 单测/工具/证明微基准均重新构建成功（exit0）：

```bash
cmake --build build --target all FriskLayoutUnitTests --parallel 12
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt frisk_layout_proof_bench --parallel 12
```

对应日志 `/tmp/frisk-p12-final2-debug-build.log`、`/tmp/frisk-p12-final2-release-build.log`。包含历史 CMP0116 和 Ninja 构建日志恢复提示，不宣称全程无警告；此前主动中断的构建不计成功。

最终版本身份：

- main/HEAD 仍为 `b23791101d095023cdd3033c8751b2fbbd81b4e6`，未提交增量包含 P0/P1.1/P1.2。
- Release `frisk-opt` SHA256：`3c1f549298693d549ebad304b59bf43b0a830be631052d5c096b82524607bf8d`。
- Release/Debug CMakeCache SHA256：`e4acb24cf609aebc914141e60915a236d1aed20effb8aada0d7b6785952aaffb` / `1ca319f6b37618d9661ca2559c97c666e1bc29fcf8e3fe3414206b00ad8cdb08`。
- 源码规范化 tar 流 SHA256：`ceee454c3d267621339e7164af20561296623fd29d90233e20a74a6de10048ac`（包含下述测试期望同步）。命令如下，不含文档、构建产物和 Python cache；与 §1 原始快照 hash 不是同一种归档算法。

```bash
tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --exclude=__pycache__ --exclude='*.pyc' -cf - include lib unittests test tools benchmark CMakeLists.txt | sha256sum
```

回归使用冻结后的二进制，期间无编译/链接：首轮 Debug 单测按 4 个分片运行，最后的测试同步后以 `GTEST_TOTAL_SHARDS=16 GTEST_SHARD_INDEX=0..15` 重跑全部用例并保存 XML，不过滤慢用例；69 个 lit；Debug 全部 CTest；Release 只选实际构建的 `FriskLayoutProofBenchmarkTest|FriskLayoutConversionGeneratorTest`，不将未构建的 Release test_pass 计为已测。Python normalization/API boundary/SM90 tool 使用 `/home/baopeihua/anaconda3/envs/torch/bin/python3.10`。

最终结果：**505/505 Debug unit、69/69 lit、8/8 Debug CTest、2/2 选定 Release CTest、5 项 Python normalization，以及 API boundary/SM90 tool 全部通过**。16 个单测进程均 exit0，逐个 XML 检查 505 个唯一用例，无 failure/error/disabled/skipped。Release 受影响集合 60/60 通过，测试同步后的 3/3 通过；局部/全量中重叠的用例不相加。

完整 lit 已结束：69/69 通过（670.38 秒）；Release 受影响集合 60/60 通过（XML `/tmp/frisk-p12-postfix-release-focused.xml`）。同轮 Debug 发现三处旧测试仍把“数据节点数”当作“含 geometry 的总节点数”，分别在 `LayoutPipelineTest` 两项和 `ReduceLayoutIntegrationTest.MmaResultInherits128Threads`。这轮不能称为全量通过。

测试同步只涉及这两个测试文件，不改生产代码：直接 SS 链为 3 Storage＋4 Distributed＋1 Instruction；RS 为 2＋6＋1；SS→Reduce→store 为 3＋6＋1。保留原数据数量和每角色最多四候选，核对第五角色的 anchor/type/threads/no-SSA 及 legacy Square geometry 单例，检查实际绑定与重放；geometry 因子为 1，不增加原有效组合上界。三处修改均有实际失败日志，独立源码复审及双配置 GREEN 通过，完整 Debug 复测亦通过。

测试同步后 Release 单测目标构建成功，受影响 3/3 用例通过（16.73 秒）；Debug 单测目标构建成功，未重复运行定向 3 项，而是由随后的全量 16 分片同时覆盖。Release `frisk-opt` 哈希不变；此前完整 lit/辅助检查仍对应相同生产源码与工具，不把测试文件的变化描述为新的生产修复。

最终日志：`/tmp/frisk-p12-final3-unit-{0..15}.{log,xml}`、`/tmp/frisk-p12-postfix-{lit,ctest,release-ctest,python}.log`、`/tmp/frisk-p12-postfix-release-focused.xml`；测试同步证据 `/tmp/frisk-p12-countfix-report.md`。首轮失败和主动中断日志保留，不用覆盖日志隐藏过程。

## 8. CPU 开销测量与热点

三个输入为 `attention-layout.mlir`、`attention-layout-fixed-abi.mlir` 和 `layout-solver-cost.mlir`，其 SHA256 均与修改前 source archive 内的输入一致。每个输入分别运行旧版 profile-off、新版 profile-off、新版 profile-on，每组 1 次预热＋5 次记录，共 54 次独立 CLI 调用；所有本任务构建/测试作业关闭后串行执行。另用 9 次独立 trace 调用核对成本分项之和、新版 profile 开关等价及 legacy 最终完整成本。

典型单次命令如下（不同样本使用独立输出路径）：

```bash
/usr/bin/time -f '%M' -o /tmp/sample.rss build-m6-release/bin/frisk-opt \
  test/Transforms/attention-layout-fixed-abi.mlir \
  --frisk-layout-pipeline='stats-file=/tmp/sample.stats.json profile-file=/tmp/sample.profile.json' \
  -verify-each -o /tmp/sample.mlir
```

profile-off 去掉 `profile-file`，旧版使用 §1 保存的二进制，trace 检查额外传 `trace-file`。时间为外部整条 CLI 的 wall time（含解析、验证、输出和文件发布），RSS 为 `/usr/bin/time` 的峰值 KiB；不是 GPU runtime，也不将 profile 的嵌套计时相加。原始样本、配置/输入/二进制身份及运行命令保存在 [测量数据](p1_2_measurement_results.json)。

环境：AMD EPYC 9654，2 sockets/192 online CPUs，GCC 11.4.0，Release，MLIR 位于 `/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir`。没有锁频、CPU 绑核或对其他用户负载作隔离；5 个固定顺序样本用于检查本次开销及发现热点，不足以声明小幅差异具有统计显著性。

### 8.1 完整 CLI 耗时

54 次调用全部 exit0；下表是排除预热后的 5 样本中位数，单位秒。变化率分别为 `new_off / old_off - 1` 和 `new_on / new_off - 1`。

| 输入 | 旧版 off | 新版 off | 新版 on | 新版 off 变化 | profile 开启变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 自由布局 Attention | 13.567934 | 13.479138 | 13.499130 | −0.65% | +0.15% |
| 固定 ABI Attention | 8.183015 | 8.050704 | 8.059028 | −1.62% | +0.10% |
| 小图 | 0.168684 | 0.168711 | 0.170962 | +0.02% | +1.33% |

新版 off 的样本范围依次为 `[13.463387,13.523589]`、`[8.046929,8.060685]`、`[0.167252,0.169805]` 秒；所有配置的完整范围和原始值在 JSON 中。两类 Attention 本机样本略降，小图基本持平；没有做逐项消融或随机交错采样，不将差异归因于某一修改，也不据此宣称通用加速。

三种配置正式样本合并后的峰值 RSS 范围分别为 71568–73892、74428–75996、55600–57064 KiB。profile 未引入数据缓存；RSS 的小幅波动不等同于 observer 的精确内存占用。

### 8.2 观测到的调用与缓存

下表给出 profile-on 的固定调用次数和五次 inclusive 时间中位数。次数在五个样本间一致。

| 观测项 | 自由 Attention：次数 / 秒 | 固定 ABI：次数 / 秒 |
| --- | ---: | ---: |
| access_cost | 1169 / 1.730830 | 1519 / 2.547542 |
| address_enumeration | 266 / 0.311370 | 538 / 0.359842 |
| execution_enumeration | 2157 / 0.121896 | 3182 / 0.169890 |
| instruction_contract | 150 / 0.115728 | 336 / 0.218322 |
| relation_query | 1433 / 0.083734 | 1406 / 0.099707 |

关系缓存：自由布局 `227538 hits / 443 misses / 443 insertions`；固定 ABI `121263 / 387 / 387`。地址表缓存：自由布局 `3612 / 57 / 57`；固定 ABI `3559 / 61 / 61`。这些是现有缓存的实际观察值，不是新加入缓存的成绩。

两类 Attention 未触发 conversion_classification 与 alias_pair_cache；小图触发 108 次转换分类，inclusive 中位时间 0.007704 秒。alias pair 的正向计数由单测覆盖，不能把本次样本的零计数说成该类计算普遍无成本。

在本轮覆盖的计时项中，access_cost 最大，后续 O07 可以继续拆解它及未覆盖的验证阶段；不能把这张表当成完整 CPU 火焰图，更不能将嵌套时间相加后从 wall time 相减来定位“剩余开销”。本轮只补观测，没有据此引入全局缓存。

### 8.3 新旧决策与 profile 等价性

9 次 trace 调用全部通过现有 checker 的成本贡献求和核验。三个输入的新旧最终 IR 字节相同，完整九项成本相同；新版 profile 开/关的 IR、全部非计时 stats 以及 trace 均一致（trace 另以 `cmp` 检查逐字节相等）。每组六次运行的 IR/非计时 stats 也一致，没有把 timing 字段拿来要求确定性。

| 输入 | 旧→新图节点 / 约束 | 旧→新原始提案 | 旧→新 trace 事件 | 最终九项成本（新旧相同） |
| --- | --- | --- | --- | --- |
| 自由 Attention | 38→40 / 78→78 | 329→331 | 169→326 | `[8293,544,0,0,428,6144,6,0,0]` |
| 固定 ABI Attention | 38→40 / 82→82 | 324→326 | 101→444 | `[8293,800,0,0,428,6144,6,0,0]` |
| 小图 | 4→4 / 5→5 | 38→38 | 16→16 | `[4,20,0,0,2,0,0,0,0]` |

Attention 各有两个 MMA，新增两个 singleton geometry 及两个提案；MMA 约束变为五角色，但约束条数不增加。新 trace 增加细分支持/拒绝和预检查事件，所以**不要求新旧版本 trace 相同**。新增统计项 `instructionCombinations / instructionFragmentPrechecks` 分别为自由图 `150 / 5`、固定 ABI 图 `336 / 5`、小图 `0 / 0`。这里是整图所有指令约束的合计，固定图的 336 不表示单个 op 突破 256 次完整证明上限。

数据 JSON 保存了原始测量/trace 驱动源码，可提取后从仓库根目录重跑：

```bash
python3 /tmp/frisk-p12-measure.py /tmp/frisk-p12-baseline-vZeQIU/frisk-opt \
  build-m6-release/bin/frisk-opt /tmp/frisk-p12-postfix-measure
python3 /tmp/frisk-p12-traces.py /tmp/frisk-p12-baseline-vZeQIU/frisk-opt \
  build-m6-release/bin/frisk-opt /tmp/frisk-p12-postfix-measure /tmp/frisk-p12-postfix-traces
```

重跑时应使用新的输出目录，驱动拒绝覆盖既有目录。JSON 留存全部样本、命令、profile、stats、身份和 trace 摘要；完整 IR/trace/XML 以及旧二进制快照位于本机 `/tmp`，不是远端仓库中的已提交产物。

## 9. 完成边界与后续

本轮完成 O08 的 Copy 集成/诊断、O07 的可关闭观测、O04 的有限 MMA grid 联合选择。没有增加 Copy 指令种类、放大证明/搜索预算或改变九项成本权重；没有自动选择 tile/线程数、遍历全部 atom、证明 GPU 异步执行安全，也没有展示 consumer 导致 grid 赢家反转。

P1.3 的模型覆盖/实际 IR 存活资源诊断/循环频率，以及 P1.4 的单位轴与 Attention GPU 执行闭环继续按阶段契约保留。未刷新 TileLang/Triton 参考快照，本轮结论仅限 Frisk 自身受限布局推断，不作为独创性或跨框架性能领先证据。
