# P0 方案 A 实施与验证记录

状态：O01/O03/O02 实现与验收完成（2026-10-14）。逐任务及整体源码复审通过；修复后完整回归和串行重复采样通过。改动保留在源目录 main，尚未提交或推送。

## 1. 可复现基线

- 代码基线：`219c0cf52ac261aae3dfda77b6c92e42d6473d5a`。
- 设计提交：`b237911`；用户已确认[详细契约](p0_implementation_design.md)。
- 实现直接位于源目录 main；后续变更尚未提交或推送。构建产物不属于提交内容。
- 本机基线构建：Debug、GCC 11.4.0、CMake 4.2.1；LLVM/MLIR 构建版本 `21.0.0git`，构建头文件记录 LLVM revision `7271c2f8f92f7ea4f5da02cd249fab5cf24920ec`。这是构建产物的版本记录，不宣称外部 LLVM 源工作区已审计为干净状态。

本轮基线验证命令：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_brief=1 --gtest_output=xml:/tmp/frisk-p0-baseline-unit.xml
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py \
  -sv -j 4 build/test
build/bin/frisk-opt test/Transforms/attention-layout.mlir \
  --frisk-layout-pipeline='stats-file=/tmp/frisk-p0-baseline-attention.json' \
  -verify-each -o /tmp/frisk-p0-baseline-attention.mlir
```

结果：构建 exit 0；417/417 unit（XML failures/errors/disabled 均为 0）；61/61 lit；Attention 推断 exit 0。unit 总计 829.188 秒；此轮同时运行过其他测试，不将这些时间当作后续性能比较样本。构建仍有既有 Ninja 日志恢复提示，不宣称零警告。

Attention 独立推断统计：38 variables、78 constraints、1 component；61 prepared candidates，删减 9；组合空间 5184；beam width 32，3034 MRV probes、538 expanded nodes；内存事务估算 608、bank 冲突估算 64；无转换/重计算。这验证了后续比较基线仍可复现，不是 GPU 性能结果。

## 2. O01 局部实现证据

新增固定行主序完整 Attention fixture，以及五个集成用例：未知更强对齐、显式 16 字节对齐、静态行偏移、非对齐子视图、与根契约冲突的硬布局绑定。四个外部矩阵均逐坐标核对实际地址；保持真实完整 SSA 链，检查实际布局验证、print/parse、重复 pipeline 与失败回滚。

测试证明：非对齐 f16 子视图允许单元素 2 字节访问，但不能把它当作连续 4/8/16 字节向量访问。未知更强对齐的用例保持 `vector_bytes=1` 保守路径。

当前未修改生产推断规则；正确构造的用例在已有实现上通过。最初的编译/断言失败属于测试构造错误，已修正，不宣称修复了生产缺陷。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='AttentionLayoutIntegrationTest.*:StorageRootContractTest.*'
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py \
  -sv build-m6-release/test/Transforms/attention-layout-fixed-abi.mlir
```

局部结果：16/16 unit、1/1 lit，构建 exit 0。独立 spec/quality 复审通过，无 Critical/Important/Minor 问题；该阶段尚未执行最终全量回归，修复后的最终结果见第 7 节。Release 重建遇到既有 CMake policy/APInt 编译警告，未修改无关代码处理它们。

## 3. O03 基础模块局部证据

已实现独立有界 JSON 记录器和可选成本观察接口。成本贡献来自原有实际计费位置，Shared 根和选中的转换/重计算 leader 只收一次；重试使用独立 costScope，历史费用不混入最终总成本。事件/字节超限后保留摘要及 omittedCost；关闭记录时不构造事件。

新增记录器测试覆盖延迟构造、完整事件截断、JSON 转义及 UTF-8、饱和加法、attempt 隔离、原子文件发布和路径冲突。成本测试覆盖观察前后结果相同、根去重、选中转换及重计算费用核对。独立复审发现并修复两项问题：关闭日志仍全图遍历的额外开销，以及符号链接与 `..` 组合路径的同目标误判。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests -j 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutTraceTest.*:LayoutCostModelTest.*:StaticLayoutCostTest.*:SM90CostModelTest.*:SM90AddressCostTest.*:LayoutConversionChoiceTest.*'
```

最终局部结果：52/52 unit、构建 exit 0，独立 spec/quality 复审通过。路径回归在修复前真实失败、修复后通过。基线与当前 SM90 测试 fixture 使用相同编译选项均复现六项 APInt `-Wfree-nonheap-object` 警告；既有 CMP0116 OLD 警告也保留，未屏蔽或做无关修改。本节只列基础模块证据，完整流水线 trace 验收见第 4 节。

## 4. O03 流水线接入证据

已在 standalone pass 和完整 pipeline 接入可选 trace，覆盖真实 prepared/strict/common 域、约束支持、三值拒绝原因、每层有限搜索摘要、最终布局/契约/转换方案。日志不重复求解或证明；失败诊断求解与正式选择明确区分。报告发布失败时保留原 IR，固定摘要在截断后仍保留实际求解选项。

复现改进前选择的 Release 命令（显式关闭已实现的邻域改进）：

```bash
build-m6-release/bin/frisk-opt test/Transforms/attention-layout.mlir \
  -frisk-layout-pipeline='refine-beam-solution=false trace-file=/tmp/frisk-attention.trace.json stats-file=/tmp/frisk-attention.stats.json' \
  -o /tmp/frisk-attention.inferred.mlir
python3 test/Transforms/Inputs/check-layout-trace.py /tmp/frisk-attention.trace.json attention
```

局部验收包括 59/59 unit、4/4 lit；复审补齐固定摘要中的求解选项后，26/26 覆盖 unit、4/4 lit 再次通过。独立 spec/quality 复审通过。测试核对 trace 开关及重复运行的 IR 字节、全部非计时 stats，以及不排除任何字段的语义 trace 相等。

| 改进前输入 | trace 字节/事件/贡献项 | prepared→strict→common | 九项成本 |
| --- | --- | --- | --- |
| 自由布局 Attention | 429750 / 92 / 36，未截断 | 61→52→52，均含 38 个变量 | `[8293,608,64,0,428,6144,6,0,0]` |
| 固定行主序 ABI Attention | 396840 / 101 / 36，未截断 | 65→47→47，均含 38 个变量 | `[8293,800,0,0,428,6144,6,0,0]` |

表中日志字节数来自接入阶段，后续增加固定摘要中的 refinement 选项后略有增加；对应成本与候选域不变。两者均按实际贡献重新求和验证全部九项成本与 saturation。真实双分量重试测试还验证：历史 attempt 已计费 7，最终 attempt 计费 14；极小日志预算下 omittedCost 仍为 14，而不是混入历史后的 21。这些是模型成本与日志正确性证据，不是 GPU 执行测量。

## 5. O02 局部实现与结果

默认 `refine-beam-solution=true`、`max-refinement-probes=256`，可关闭或设探测数为 0；精确搜索不改。仅对 beam 已找到的完整合法解，在传播后的冻结域内进行单变量 best-improvement。每次尝试同时消耗局部预算与原分量总预算；完整验证约束和九项成本（含转换/重计算），相同成本不移动，合法邻居计费失败明确报 CostFailure 并回滚。

自由布局 Attention 的实际选择变化：

| 项目 | 关闭改进 | 开启改进 |
| --- | --- | --- |
| 完整成本向量 | `[8293,608,64,0,428,6144,6,0,0]` | `[8293,544,0,0,428,6144,6,0,0]` |
| MRV 探测 / 展开节点 | 3034 / 538 | 3034 / 538 |
| 邻域探测 / 接受 / 扫描轮数 | 0 / 0 / 0 | 70 / 4 / 5 |
| 总探测 | 3034 | 3104（低于 65536） |
| 停止 / 穷举标志 | disabled / false | no-improvement / false |

新增用例覆盖九维字典序、最佳而非首个改善、规范化平局、多轮/不移动/双变量障碍、非法及 Unknown、无预算/部分扫描耗尽、转换成本抵消收益、最终计划重建及成本失败回滚。独立有限图 oracle 用四个固定种子、每个 12 图、每图 20 种插入排列核对 `oracle ≤ refined ≤ unrefined`、合法性与确定性，不要求局部解等于全局最优。

局部结果为 61/61 unit，最后一次仅增强测试后的 53/53 覆盖复测，以及 6/6 lit；独立 spec/quality 复审通过。真实 Attention 还通过完整 SSA 数据流、已解图、物化布局和 MLIR 验证。此阶段（加入第 6 节全局保护前）完整路径 lit 的默认 trace 为 460941 字节、168 个事件、36 项成本贡献，无截断；关闭改进为 429856 字节、92 个事件。源文件路径进入 provenance，使用相对/绝对输入路径时字节数可以不同。

## 6. 整体复审与饱和保护

整体复审发现并修复了跨分量饱和加法的顺序问题：局部分量字典序改善可能在全局高优先级项饱和后变成退化。新增真实失败回归后，保留未改进方案与已计算成本，在最终选择前比较全局饱和总成本；若更差，整体退回原方案。精确/禁用分量保持原计费流程，可能变化的 beam 分量延后最终选中计费；不增加候选、搜索或为日志重算证明。

局部搜索历史仍由 start/end/probes/accepted 表示；`refinementGlobalFallback` 和 `selectedCost` 明确最终选中结果。数值与实际转换回归同时核对布局、指令绑定、转换计划、trace 开关/截断、贡献求和、预算和失败语义。修复后 56/56 覆盖 unit、4/4 lit 通过，Attention 向量与 70 次探测不变；新增全局选择事件后，其完整路径 trace 为 461245 字节、169 个事件、36 项贡献、零遗漏。

该修复通过整体源码复审确认，无剩余 Critical/Important 问题。修复前 448 项 unit 和其他守门结果仅作历史证据；以下验收全部在修复后重新运行。

## 7. 最终完整回归

| 守门 | 修复后结果 |
| --- | --- |
| Debug 显式目标及全部目标构建 | exit 0 |
| 全量单测 | 450/450；四分片 113+113+112+112；XML failures/errors/disabled 均为 0 |
| 完整 lit | 64/64，exit 0 |
| Debug CTest | 7/7 |
| Release conversion generator CTest | 1/1 |
| Python normalization | 5/5 |
| 旧推断生产 API 隔离 | PASS |
| SM90 工具 SS/RS、输出存活及目标拒绝 | PASS |

复现命令：

```bash
cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12
cmake --build build --parallel 12
for shard in 0 1 2 3; do
  GTEST_TOTAL_SHARDS=4 GTEST_SHARD_INDEX=$shard \
    build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1 \
    --gtest_output=xml:/tmp/frisk-p0-unit-$shard.xml
done
/usr/bin/python3 /data0/xiebaokang/rocm-llvm-project/llvm/utils/lit/lit.py -sv -j 4 build/test
ctest --test-dir build --output-on-failure
ctest --test-dir build-m6-release -R FriskLayoutConversionGeneratorTest --output-on-failure
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_normalization_test.py --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/legacy_api_boundary_test.py --source-root . --build-dir build
/home/baopeihua/anaconda3/envs/torch/bin/python3.10 test/python/sm90_layout_tool_test.py --build-dir build
```

本轮四分片并发执行，其余命令相同；原始证据位于 `/tmp/frisk-p0-postfix-{build,lit,ctest,release-ctest,python}.log` 和 `/tmp/frisk-p0-postfix-unit-{0..3}.{log,xml}`。保留既有编译警告及负例诊断，不宣称零警告；本轮未执行 GPU Attention。

## 8. 串行 CPU 测量

环境：AMD EPYC 9654（两路，共 192 个可见 CPU）、GCC 11.4.0、Release；未绑定 CPU、不承诺独占整机。所有本轮构建/回归进程结束后才开始，两个测量工具之间也不并发。只说明 CPU 推断开销和模型选择，不说明 GPU 性能。

### 8.1 多个小分量

使用现有 `frisk_layout_solver_bench`，每组 5 次预热、30 次测量，trace 关闭、默认 SolverOptions（含本轮新增的 true/256）。每个分量含 10 个真实依赖操作，不是孤立常量。四组均成功、输出哈希稳定。

| 操作数 / 分量数 | 求解中位数 / p95（ms） | 完整 pipeline 中位数 / p95（ms） |
| --- | --- | --- |
| 10 / 1 | 19.856 / 20.061 | 20.644 / 20.842 |
| 50 / 5 | 104.911 / 105.699 | 108.667 / 109.359 |
| 100 / 10 | 226.045 / 230.477 | 233.706 / 236.680 |
| 250 / 25 | 669.470 / 674.831 | 695.784 / 698.423 |

```bash
cmake --build build-m6-release --target frisk_layout_solver_bench --parallel 12
build-m6-release/benchmark/layout/solver/frisk_layout_solver_bench > /tmp/frisk-p0-solver-bench.json
```

本轮原始结果 `/tmp/frisk-p0-final-solver-bench.json`，可执行文件 SHA-256 为 `738953fe9ca1dc2663eb076f1ac5a7261e317d5c9e43b11a671b1a788465ba09`。这是当前版本的扩展规模测量；没有据此宣称严格线性复杂度或相对旧版本更快。

### 8.2 Attention 开关对照

8 组配置各 1 次预热、5 次正式测量，共 48 次串行调用全部成功。每组重复 IR、非计时 stats 和 trace 内容稳定；同一 refinement 设置下 trace 开关的 IR、非计时 stats 完全相同。所有有 trace 的样本均通过完整候选域、贡献求和及全局选择核对；默认上限未截断。可执行文件在采样前后 SHA-256 相同。

| 输入 | refinement / trace | 五次中位数（s） | 最小–最大（s） | trace 字节 |
| --- | --- | --- | --- | --- |
| 自由布局 | off / off | 10.871059 | 10.847020–10.892804 | — |
| 自由布局 | off / on | 10.918432 | 10.905480–10.921648 | 429856 |
| 自由布局 | on / off | 13.587561 | 13.584200–13.632467 | — |
| 自由布局 | on / on | 13.653714 | 13.641993–13.684305 | 461245 |
| 固定 ABI | off / off | 8.225482 | 8.209775–8.269853 | — |
| 固定 ABI | off / on | 8.277844 | 8.252229–8.724325 | 402842 |
| 固定 ABI | on / off | 8.217649 | 8.194719–8.240881 | — |
| 固定 ABI | on / on | 8.268749 | 8.250542–8.279380 | 402840 |

自由布局保持 §5 的成本变化：608→544（内存事务估算降低约 10.5%），bank 代价 64→0；增加 70 次探测。关闭 trace 比较，中位推断时间增加约 **25.0%**，这是搜索更好方案的 CPU 开销，不宣称编译更快。trace 自身中位额外开销约 0.4%–0.6%；样本有限且有系统噪声，不作为普遍上界。

固定 ABI 的分量走 **exact**，因此 refinement 开关都为 0 次邻域探测，MRV/总探测均 1525、展开节点 358，成本始终 `[8293,800,0,0,428,6144,6,0,0]`。其小幅计时差异不归因于算法改善。固定 ABI 的 trace 两个字节差异来自布尔配置拼写，不是布局变化。

原始五次耗时（秒，按执行顺序，不含预热）：

```text
free off/off : 10.881570 10.892804 10.871059 10.863071 10.847020
free off/on  : 10.905480 10.921648 10.909719 10.920143 10.918432
free on/off  : 13.632467 13.584200 13.587561 13.585540 13.632145
free on/on   : 13.641993 13.667778 13.653714 13.645335 13.684305
fixed off/off: 8.213493 8.269853 8.209775 8.237464 8.225482
fixed off/on : 8.724325 8.291555 8.277844 8.273066 8.252229
fixed on/off : 8.217649 8.235216 8.194719 8.240881 8.210160
fixed on/on  : 8.279380 8.276674 8.268749 8.250542 8.265615
```

本轮详细 JSON、每次命令/IR/stats/trace 位于 `/tmp/frisk-p0-measure-bfcb0xos/`（汇总 `results.json`），日志 `/tmp/frisk-p0-final-measure.log`。`frisk-opt` SHA-256：`08c8f4dd9f0dcd5532daa0d4b0ae126b0e21b194250c0349c6e2e11fc4f0ccdb`。临时目录不是长期复现前提，原始样本及可复跑命令已保存在本文。

可独立复跑的串行命令如下；`run=0` 为预热，统计时只取 1–5。所有组合都写 stats，trace 组合额外写决策日志。测量包含进程启动、推断、验证、物化及报告/IR 写入；不包含外部 Python 核对。本轮用 Python `perf_counter_ns` 计壁钟，下面用 `/usr/bin/time` 记录等价范围。

```bash
p0_reports=$(mktemp -d /tmp/frisk-p0-repro.XXXXXX)
for fixture in attention-layout attention-layout-fixed-abi; do
  for refine in false true; do
    for trace in false true; do
      for run in 0 1 2 3 4 5; do
        p0_prefix="$p0_reports/$fixture-$refine-$trace-$run"
        p0_opts=("refine-beam-solution=$refine" "stats-file=$p0_prefix.stats.json")
        if [ "$trace" = true ]; then
          p0_opts+=("trace-file=$p0_prefix.trace.json")
        fi
        /usr/bin/time -f '%e' -o "$p0_prefix.time" \
          build-m6-release/bin/frisk-opt "test/Transforms/$fixture.mlir" \
          --frisk-layout-pipeline="${p0_opts[*]}" -verify-each -o "$p0_prefix.mlir"
        if [ "$trace" = true ]; then
          /usr/bin/python3 test/Transforms/Inputs/check-layout-trace.py "$p0_prefix.trace.json" attention
        fi
      done
    done
  done
done
```

可继续用 `diff` 比较同配置的重复 IR 和 trace 开关 IR，用上述 checker 的 `compare-stats`/`compare-trace` 模式核对非计时 stats 与完整语义 trace。为复现完全相同的 provenance 字节数，需使用同一绝对输入路径；本轮测量使用 `/home/baopeihua/frisk/test/Transforms/` 下的绝对路径。
