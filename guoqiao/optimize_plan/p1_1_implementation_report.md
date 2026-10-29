# P1.1 实施与验证记录

日期：2026-10-21。状态：按方案 A 完成 P1.1 实现、测试与测量。契约见 [设计](p1_1_implementation_design.md)，逐任务见 [执行计划](p1_1_execution_plan.md)。

本轮将 ownership 与 Broadcast 同线程可取检查改为受限 GF(2) 符号证明，解决部分超过 65536 点的可证明案例，并保留实际访问表的原有安全上限。465 项 unit、64 项 lit 及辅助守门通过；证明微基准明显降低调用开销，但两类 Attention 的完整推断耗时基本持平。P1.2–P1.4 尚未实施，修改未提交、未推送。

## 1. 实际基线

HEAD 为 `b237911`，但实际测试基线包含已验收且未提交的 P0 工作区增量，不能用 HEAD 单独重建。本轮未创建工作树、未提交或推送，保留已有修改。

- 基线快照：`/tmp/frisk-p1-baseline-xpDmX2/source.tar`，包含 include/lib/test/unittests/benchmark/tools/guoqiao；另保存 tracked.diff 和相关 untracked 清单。
- Release 基线 executable：`/tmp/frisk-p1-baseline-xpDmX2/frisk-opt`，SHA256 `08c8f4dd9f0dcd5532daa0d4b0ae126b0e21b194250c0349c6e2e11fc4f0ccdb`，与 P0 最终测量二进制一致。
- 采样产物：`/tmp/frisk-p1-baseline-xpDmX2/measure/results.json` 及对应 IR/stats/stderr/RSS；临时目录不是仓库长期附件，下表保留核心原始样本。
- 两个 fixture 各 1 次预热 + 5 次测量；全部 exit 0；各自 6 次 IR hash 相同。默认 refinement 开启、trace 关闭、stats 开启；固定 ABI 使用现有未知更强对齐 fixture。

## 2. 优化前测量

所有样本串行执行；这期间没有本任务的并行构建/测试。未绑定 CPU，也不宣称机器独占。wall time 包含进程启动、pipeline、实际验证、IR/stats 文件写入；峰值 RSS 是 `/usr/bin/time -f %M` 报告的子进程 KiB，不是所有子进程或系统内存总和。

| 输入 | 5 次 wall time（秒） | 中位数（秒） | 5 次 max RSS（KiB） |
| --- | --- | --- | --- |
| 自由布局 Attention | 13.556042, 13.591026, 13.602007, 13.583617, 13.593573 | 13.591026 | 72768, 72184, 72160, 72448, 72780 |
| 固定 ABI Attention | 8.197328, 8.205570, 8.209231, 8.207742, 8.199948 | 8.205570 | 74684, 75312, 75020, 75568, 75108 |

已有统计中的阶段中位数（毫秒）：

| 计时字段 | 自由布局 | 固定 ABI | 口径 |
| --- | ---: | ---: | --- |
| collection_ns | 0.185 | 4.519 | 收集图 |
| preparation_ns | 302.430 | 459.700 | 准备候选/支持表，包含其中证明 |
| attempt.propagation_ns | 6.244 | 27.548 | 传播 |
| attempt.solve_ns | 11885.738 | 6679.162 | 求解，包含检查/证明/成本，不是纯搜索循环 |
| attempt.verification_ns | 40.901 | 33.611 | attempt 验证 |
| propagation_search_retries_ns | 11938.294 | 6746.743 | 包含上述 attempt 阶段的外层计时 |
| verification_ns | 377.548 | 281.484 | pass 外层验证 |
| layout_pipeline_ns | 13455.224 | 8110.870 | pipeline 总计时 |

这些嵌套字段不能相加成独立阶段总耗时。观察结果是求解阶段占主要时间，但其中包含重复证明、成本评估等，不足以判定某个证明函数就是主要瓶颈。本环境未发现 perf/gdb/valgrind 可执行工具，本轮将使用生产阶段计时与独立证明微基准，不能宣称获得全流水线逐函数热点或缓存命中率。

基线复现单次命令（分别将 fixture 替换为 attention-layout-fixed-abi.mlir 可测固定 ABI）：

```bash
/usr/bin/time -f '%M' -o /tmp/p11-before.rss \
  /tmp/frisk-p1-baseline-xpDmX2/frisk-opt \
  test/Transforms/attention-layout.mlir \
  --frisk-layout-pipeline='stats-file=/tmp/p11-before.stats.json' \
  -verify-each -o /tmp/p11-before.mlir
```

本轮 Python runner 使用 perf_counter_ns 包围该命令；每个配置首轮仅预热，其后五轮计入统计。

## 3. Ownership 局部实现与验收

增加 `LinearExecutionLayout` 受检规范表示，按 register/lane/warp/warp_group/cta 展开完整硬件位域，包括映射省略的 carrier 零列。ownership 使用秩验证覆盖、指数验证 replication、核验证唯一写者；缺覆盖返回真实不可达逻辑坐标，重复写者返回与零点同值的非零硬件坐标。结构分析的 Proven 明确不是 ownership 证书。

采用最多 256 个硬件位/逻辑位的代数上限，不构造全硬件点数。维持原显式 carrier/逻辑维度最多 16 位、lane≤32 的检查；处理零位域及不能以 signed i64 表示的复制因子。`enumerateExecutionLayout` 未修改，仍是独立的有界 oracle；vector/storage 消费者仍受其限制。

例如硬件 topology `[4,32,1024,1,1]`、逻辑 shape `[4]`，只由两位 register 映射四个逻辑值，其余 carrier 不影响值：共有 131072 个硬件持有者，每值复制 32768 次。旧枚举返回 Unknown；新 first_owner 证明返回 Proven，all-writers 返回 Disproven，而具体枚举/访问表仍返回 Unknown。这个例子只证明数学 ownership，不宣称该 topology 是 SM90 可执行资源配置。

TDD：先修改超界断言，旧实现实际返回 Unknown，测试 exit 1；实现后 16/16 局部测试通过。差分覆盖所有 64 个 2×3 矩阵在三种 replication 下的行为和全部 16 个 2×2 矩阵，并独立核验反例；包括重排/省略 carrier、零位域、巨大域、畸形属性和原访问限制。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ExecutionLayoutProofTest.*:SymbolicExecutionLayoutTest.*'
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests
```

局部与最后增量构建 exit 0，focused 16/16；补充 Release 全量 457/457 通过（182.172 秒，非性能样本），固定 TileLang corpus 20 个等价/3 个预期拒绝通过。独立 spec/quality 复审通过；详细过程 `/tmp/frisk-p11-task1-{report,review}.md`。既有 CMP0116 与 GCC APInt 警告、负例诊断保留，不声称零警告。此时 Broadcast 与最终 P1.1 守门尚未完成。

## 4. Broadcast 局部实现与验收

源与目标首先分别通过 ownership，再验证现有形状/轴/dtype、单 CTA、相同线程 topology。将输出坐标删除广播轴后，令 R 为源 register 映射，D 为目标 register 映射与双方 thread 映射之差，检查 `rank([R,D]) == rank(R)`。这等价于目标所需的每个源值都能由该线程的某个源 register 提供，允许 register 编号/数量/基不同。

以直观的两线程示意说明区别：源线程 0 持有 `{x0,x1}`，源线程 1 持有 `{x2,x3}`。如果输出线程 0 要广播 `{x0,x2}`，虽然整个硬件域覆盖四个源值，x2 仍不在该线程内，因此必须拒绝。新证明检查的是每个线程的可取空间，不是仅比较全局覆盖或寄存器编号。

不满足包含关系时，选取被源 register 像空间的左核分离的目标硬件列，生成实际不可取输出坐标。测试独立枚举该线程所有源 register 核验反例，并核验重复返回确定性。

两项超界测试先实际失败：旧实现对合法和非法关系都返回 Unknown；新实现分别返回 Proven 和 Disproven。原 `enumerateExecutionLayout` 仍 Unknown；带插入轴的自动投影仍调用有界 reduction 投影，不宣称所有超界候选都可自动生成。空轴 identity 投影仅需 coverage，可以使用新证书。

差分测试穷举全部 64 个 2×3 源矩阵与全部 512 个 3×3 目标矩阵，共 32,768 对；oracle 从真实枚举表删除广播轴，穷举同线程源 register，不调用矩阵代数。另测 XOR/依赖 register 基、复制、零 register 位、carrier 重排/省略、多轴、错误轴、跨 lane/warp/CTA、65536 点边界、超界、错误 ownership/畸形输入和独立的 256 位预算。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='BroadcastLayoutTest.*:SymbolicBroadcastLayoutTest.*:ExecutionLayoutProofTest.*:SymbolicExecutionLayoutTest.*:AttentionLayoutIntegrationTest.*'
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py -sv \
  build-m6-release/test/Transforms/attention-layout.mlir \
  build-m6-release/test/Transforms/attention-layout-fixed-abi.mlir \
  build-m6-release/test/Transforms/attention-layout-trace.mlir \
  build-m6-release/test/Transforms/infer-broadcast-layout.mlir
```

结果：32/32 证明局部 unit、40/40 含 Attention unit、4/4 lit，构建/测试均 exit 0；独立 spec/quality 复审通过，无发现项。40 项耗时 116.158 秒、lit 62.86 秒，两者有并发，不作性能样本。实际 lit 使用当前环境 Python 执行同一 LLVM 构建的 llvm-lit wrapper；直接运行 wrapper 首次遇到旧 shebang 不可访问，改用显式 Python 后通过，没有修改测试规避失败。详细过程 `/tmp/frisk-p11-task2-{report,review}.md`。

## 5. 独立证明微基准

新增 `frisk_layout_proof_bench`：固定 7 个案例，分别测量 ownership/Broadcast 的 128、65536、131072 个硬件点，以及 128 点的同线程不可取负例。共同支持域分别返回 Proven/Proven 或 Disproven/Disproven；131072 点返回符号 Proven、枚举 Unknown，明确标记 `equivalent_work=false`，不计算等价加速比。

ownership 对照使用保留的真实枚举；Broadcast 对照独立建立每线程的源值集合，再检查每个输出持有者所需的源值，不调用生产 Broadcast 或 GF(2) 证明。集合和表每次重新建立，无全局缓存。这是等价语义 oracle，不是逐字复刻旧 Broadcast 函数，因此不能将微基准比值冒充旧生产函数的实测加速比。

JSON 保留逐样本 batch ns、每次调用 ns、次数、实际/预期状态、反例、sink 核验、源码/配置摘要、编译器、构建类型、CPU 与二进制 SHA256。构造输入及计算二进制 hash 不计入证明调用时间；没有耗时阈值测试。CLI 限制 samples=1..100、iterations=1..1000，非法参数拒绝。

TDD 先因缺少目标得到预期 RED，再通过 5 项真实二进制测试。复审后补充 2 项配置测试，记录未指定构建类型为 `unspecified`，多配置生成器记录实际 Debug/Release；Ninja 不可用时仅跳过相应配置 fixture，不跳过真实二进制测试。最终局部 7/7 Python、1/1 Release CTest 通过，独立复审无遗留项。详细记录 `/tmp/frisk-p11-task3-{report,review}.md`。

```bash
cmake --build build-m6-release --target frisk_layout_proof_bench --parallel 12
python3 test/python/layout_proof_benchmark_test.py \
  --benchmark build-m6-release/benchmark/layout/proof/frisk_layout_proof_bench -v
```

## 6. 最终回归过程记录

- 首轮 lit 与显式 unit/opt 构建发生调度重叠，`frisk-opt` 链接期间导致 exit 126（Permission denied）。保留 `/tmp/frisk-p11-final-lit.log`，中断该轮；链接完成后同一失败命令成功，再独立重跑完整 lit。该失败不算通过，也未修改测试规避。
- Debug unit 发现新 malformed 负例使用 `Attr::get` 直接构造非法 topology，触发 MLIR 属性构造断言（exit 134），未进入证明函数；Release 关闭断言而未暴露。focused 复现日志 `/tmp/frisk-p11-debug-symbolic-repro.log`。已修正两份测试：结构非法通过 checked builder/parser 验证拒绝，结构合法但不满足证明条件的案例继续调用生产证明；不关闭断言、不跳过测试。修正后 Debug/Release 各 32/32 focused 通过。修复中另记录 checked-builder 重载编译错误（显式转换为 `Attribute` 后解决）和 Ninja 日志恢复警告；没有改动生产代码。
- 修复后完整 Debug unit **465/465** 通过：四分片 117+116+116+116，XML 核对 465 个唯一案例，failures/errors/disabled 均为 0。四分片均 exit 0，日志/XML 为 `/tmp/frisk-p11-postfix-unit-{0..3}.{log,xml}`；分片耗时 229.167/434.595/901.452/404.259 秒，存在并发，只作验收记录。旧版本尚未结束的分片被中断，未用旧记录拼接此次通过结论。
- 链接完成后的完整 lit 重跑 64/64 通过（exit 0，677.10 秒，非性能样本），日志 `/tmp/frisk-p11-final-lit-rerun.log`；包括自由/固定 ABI 的实际 verifier、重跑一致性，以及 trace 开关、重复、逐项成本和 refinement 预算检查。
- 整体源码复审核对了 vector/storage、Reduce、Copy/TMA、MMA 与成本模型消费者：仍需要真实有界表/地址证明，没有因为 ownership 的符号 Proven 放行未知地址。测试修复复审通过，当前无 Critical/Important/Minor 遗留项；最终阶段验收仍以完整测试、测量和文档核对为准。
- 修复后的辅助守门全部 exit 0：8/8 Debug CTest、2/2 Release CTest（proof benchmark 与 conversion generator）、5/5 Python normalization、生产 API 隔离与 SM90 工具检查。日志 `/tmp/frisk-p11-postfix-{ctest,release-ctest,python}.log`。

完整测试复现命令（unit 可按 `GTEST_TOTAL_SHARDS=4`、`GTEST_SHARD_INDEX=0..3` 分片；不能在 lit 运行期间重新链接其工具）：

```bash
cmake --build build --parallel 12
cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py -sv -j4 build/test
ctest --test-dir build --output-on-failure
ctest --test-dir build-m6-release \
  -R 'FriskLayoutConversionGeneratorTest|FriskLayoutProofBenchmarkTest' --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 \
  test/python/sm90_layout_tool_test.py --build-dir build
```

## 7. 证明微基准结果

所有测试与构建结束后串行运行，每种方法 1 次预热、30 个样本，每个样本连续调用 10 次。下表为每次调用的中位数与最小–最大值，单位 **微秒**；构造输入不计时，结果消费计时，枚举表每次重建。先测符号再测枚举，未绑定 CPU，不能视为随机化统计实验。

| 案例 / 硬件点数 | 符号（中位数；范围） | 枚举 oracle（中位数；范围） | 结果 |
| --- | --- | --- | --- |
| ownership / 128 | 0.671；0.667–0.905 | 2.967；2.960–3.655 | 都 Proven |
| ownership / 65536 | 2.624；2.621–2.880 | 2038.676；2035.777–2057.177 | 都 Proven |
| ownership / 131072 | 2.943；2.935–3.524 | 0.051；0.050–0.091 | Proven / Unknown，不是等价加速对照 |
| Broadcast / 128 | 3.241；3.221–3.384 | 10.676；10.665–12.064 | 都 Proven |
| Broadcast / 65536 | 11.634；11.615–11.989 | 7049.790；7042.000–7060.840 | 都 Proven |
| Broadcast / 131072 | 12.747；12.732–13.270 | 0.051；0.050–0.070 | Proven / Unknown，不是等价加速对照 |
| 同线程不可取 / 128 | 3.214；3.202–3.320 | 6.542；6.528–6.976 | 都 Disproven，反例 `[1,0]` |

这些结果证明指定案例的证明调用成本降低、超界证明能力增强；超界枚举非常快是因为立即返回 Unknown。没有实测 GPU 运行时间，也没有由此推断完整推断耗时的加速比例。

```bash
cmake -S . -B build-m6-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-m6-release --target frisk_layout_proof_bench --parallel 12
build-m6-release/benchmark/layout/proof/frisk_layout_proof_bench \
  --samples 30 --iterations 10 > /tmp/frisk-p11-final-proof.json
```

配置沿用现有 `MLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir`、同 LLVM 根，`CMAKE_CXX_COMPILER=/usr/bin/c++`，GCC 11.4.0，CPU 标识 `znver4`。二进制 SHA256 `464b9665d3ca8ba00950606290d63339fd3c98c6a3093d80686fb41faff60b64`；source digest `5d06af501df0083f31d180eba177d3e6c456bb2fa117a5691c026003d291133f`；config digest `ae32e6b7dbb1f4c702209fc7d7c0ed8e763945e9606e649804075b1e69222cdb`。JSON status=success，各方法实际状态及 sink 均与预期一致。

## 8. Attention 前后对照

采用与 §2 相同的命令、输入和默认搜索选项，当前 Release `frisk-opt` SHA256 为 `ad69cb595ee62f689a03819f2e5f4863a98560d0b4c3a5d3a88e26c075659853`。两个配置各 1 次预热 + 5 次测量，12 次全部 exit 0；与微基准顺序执行，无本任务的并发测试/构建。

**结论：完整 Attention 推断耗时基本持平，不能宣称显著加速。** 自由布局中位数下降约 0.21%，固定 ABI 下降约 0.17%；只有五个未绑定 CPU 的样本，没有统计显著性结论。固定 ABI 最后一次为 8.989 秒，未剔除，也没有将其原因归咎于某个未经测量的热点。

| 输入 | 优化后 5 次 wall time（秒） | 中位数：前 → 后（秒） | 范围：前 → 后（秒） |
| --- | --- | --- | --- |
| 自由布局 | 13.570202, 13.562972, 13.548375, 13.572402, 13.560489 | 13.591026 → 13.562972 | 13.556042–13.602007 → 13.548375–13.572402 |
| 固定 ABI | 8.181473, 8.185688, 8.191583, 8.220970, 8.989170 | 8.205570 → 8.191583 | 8.197328–8.209231 → 8.181473–8.989170 |

| 输入 | 优化后 5 次峰值 RSS（KiB） | RSS 中位数：前 → 后（KiB） |
| --- | --- | --- |
| 自由布局 | 71600, 71592, 71328, 71820, 71944 | 72448 → 71600 |
| 固定 ABI | 75432, 74648, 74416, 74668, 76056 | 75108 → 74668 |

阶段中位数，单位毫秒，`前 → 后`；嵌套字段仍不能相加：

| 字段 | 自由布局 | 固定 ABI |
| --- | ---: | ---: |
| collection | 0.185 → 0.186 | 4.519 → 4.559 |
| preparation | 302.430 → 295.191 | 459.700 → 453.736 |
| attempt.propagation | 6.244 → 3.514 | 27.548 → 26.253 |
| attempt.solve | 11885.738 → 11881.809 | 6679.162 → 6680.546 |
| attempt.verification | 40.901 → 39.894 | 33.611 → 32.395 |
| propagation_search_retries | 11938.294 → 11930.174 | 6746.743 → 6746.827 |
| verification | 377.548 → 372.461 | 281.484 → 277.478 |
| layout_pipeline | 13455.224 → 13427.818 | 8110.870 → 8094.914 |

局部证明改善没有转化为明显的整体收益；从阶段统计看 solve 仍占主要时间，但它同时包含成本与证明，不能据此断言某个具体函数为瓶颈。本轮没有增加逐函数采样或缓存命中率测量。

前后每个配置共 12 份 IR 的哈希一致，所有 samples 去除 `timings` 后的 stats 完全一致：

| 指标（前后相同） | 自由布局 | 固定 ABI |
| --- | ---: | ---: |
| 变量 / 约束 | 38 / 78 | 38 / 82 |
| 原始 proposals / prepared / pruned | 329 / 61 / 9 | 324 / 65 / 18 |
| 连通分量 / 笛卡尔组合 | 1 / 5184 | 1 / 216 |
| 搜索模式 | beam | exact |
| MRV probes / 邻域 probes | 3034 / 70 | 1525 / 0 |
| 邻域接受 / sweeps | 4 / 5 | 0 / 0 |
| 最终九项成本 | `[8293,544,0,0,428,6144,6,0,0]` | `[8293,800,0,0,428,6144,6,0,0]` |

IR SHA256：自由布局 `effdffc56e504038e713d49183fc2ad19211a95abd9b563654bdd2b128012cb1`，固定 ABI `7d4bca857c49f7e8489d638213a2bf46cac9e187366bb12f6f691efea90de21a`。因此两个验收输入没有因本轮优化改变最终布局或搜索预算行为；不据此声称所有其他输入的候选永远不变。

复现单次命令：

```bash
/usr/bin/time -f '%M' -o /tmp/p11-after.rss \
  build-m6-release/bin/frisk-opt test/Transforms/attention-layout.mlir \
  --frisk-layout-pipeline='stats-file=/tmp/p11-after.stats.json' \
  -verify-each -o /tmp/p11-after.mlir
```

完整微基准 JSON、Attention 前后原始 wall/RSS/阶段时间/IR hash、调用命令和非计时统计已保存为仓库内的 [测量数据](p1_1_measurement_results.json)，避免仅依赖临时文件。原始临时产物还包括 `/tmp/frisk-p11-final-{proof.json,comparison.json}` 与 `/tmp/frisk-p11-final-measure/`。

## 9. Trace 复核与交付边界

测量结束后，分别使用保存的旧二进制和当前二进制，对两类 Attention 各运行一次 trace 开启的完整 pipeline（另计 4 次，均 exit 0，不计入性能样本）。结果：

- 每次 trace 开启的 IR 与相应 trace 关闭的测量输出逐字节一致，非计时 stats 一致。
- 复用 `test/Transforms/Inputs/check-layout-trace.py` 的 `attention` 检查，逐项累加成本贡献得到 finalCost，图节点及阶段候选完整，trace 未截断。
- 两类输入的旧/新 **完整 trace JSON 均一致**，包括 graph、domains、solution 和 refinement-stop；不是仅比较候选数量。新版自身重复 trace 的确定性由完整 lit 中的 attention-layout-trace 验证。

复现命令（分别替换二进制和 fixture，使用不同输出前缀）：

```bash
build-m6-release/bin/frisk-opt test/Transforms/attention-layout.mlir \
  --frisk-layout-pipeline='stats-file=/tmp/p11-trace.stats.json trace-file=/tmp/p11-trace.json' \
  -verify-each -o /tmp/p11-trace.mlir
python3 test/Transforms/Inputs/check-layout-trace.py /tmp/p11-trace.json attention
python3 test/Transforms/Inputs/check-layout-trace.py \
  /tmp/p11-trace.stats.json compare-stats /tmp/p11-after.stats.json
```

实际核对日志 `/tmp/frisk-p11-final-traces.log`，IR/stats/trace 保存在 `/tmp/frisk-p11-final-traces/`。独立源码、测试修复及最终证据/文档复审均通过，未遗留发现项；复审记录 `/tmp/frisk-p11-whole-review.md`。

本轮未改变九项 CostVector、搜索预算、外部 ABI 或广播形状支持域；未增加全局长期缓存、GPU lowering、Copy/TMA 支持域、tile/线程数自动调优。没有 GPU 性能结论，也没有证明系统整体开销已解决。后续应据新的具体契约推进 P1.2–P1.4，而不是把 O07 全部目标视为完成。

工作仍位于源目录 `main`，HEAD 保持 `b237911`，staging 为空，保留已有 P0 修改及构建目录；本轮未创建 worktree、未 commit/push。设计、实施计划、体系对比及优化清单已同步。
