# P1.3a：成本模型身份与实际IR工作量诊断

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。逐任务RED/GREEN及独立复审；用户已批准方案A和P1.3a→b→c顺序，无需重新询问是否执行。

**Goal:** 让用户知道九项代理成本代表什么、哪些成本未覆盖，以及优化后的实际IR包含多少普通算术；不改变任何选择结果。

**Architecture:** 目标接口提供完整短模型ID，trace强制摘要与stats携带身份；纯只读分析器统计实际IR的静态逻辑element-op，pipeline在最终验证成功后按需调用。工作量诊断与可加CostVector完全分离。

**Tech Stack:** C++17、MLIR/LLVM JSON、GoogleTest、lit、Python3证据核验。

## Global Constraints

- 直接在源目录main修改，保留已有改动；不worktree/stage/commit/push/GPU/P1.4。每次只有一个代码/构建拥有者；独立复审只读。
- 已批准来源：[阶段契约§4.1](p1_2_to_p1_4_design_proposal.md#41-p13a模型身份覆盖和算术计数)。Q1/Q2已验收且原始数据冻结，不能用新版本数据覆盖它。
- CostVector九项、字典序、候选、约束、target evaluate/lowerBound、转换/重计算评分及预算完全不变。默认exact=256、beam=32、expanded=65536、refinement=256；地址表65536点。诊断不参与剪枝、tie-break或拒绝布局。
- 算术统计是实际IR上的**静态语法计数**，每个实际op一次，不按use数、线程数、循环次数或分支概率加权。不同opcode不相加成机器时间；P1.3c的频率模型属于后续任务。
- 不实现P1.3b存活分析或P1.3c频率改变。本计划执行完继续依次细化它们，不将整个P1标为完成。
- 基线是`/tmp/frisk-p13a-baseline-eKTIuI/tree`实际脏源码快照；同目录`frisk-opt`为旧Release工具，SHA256=`3c1f549298693d549ebad304b59bf43b0a830be631052d5c096b82524607bf8d`。不得用HEAD冒充实际基线。持久化源码快照在Q1/Q2证据归档。

## 具体接口与诊断契约

### 模型身份

SM90当前完整模型ID固定为`sm90-layout-lex9-legacy-v1`，独立于JSON的`schema_version=1`；它标识整套九项公式、转换/clone代理、legacy循环频率及字典序，旧的分项coverage字符串继续保留。

```cpp
// LayoutTarget.h：旧自定义target保持源兼容，不冒充SM90。
virtual StringRef costModelID() const { return "unversioned-target"; }
// SM90CostModel.h/.cpp：唯一的SM90身份来源。
StringRef getSM90LegacyCostModelID();
// SM90LayoutTarget.cpp override返回上述helper。
// LayoutTraceSummary / LayoutStatisticsReport：
std::string costModelID = "unversioned-target";
```

trace根对象和stats根对象使用`cost_model_id`。成功、失败、未执行求解及trace截断均保留实际已知target身份。底层solve/coordinator及standalone/pipeline入口不得只在成功时赋值；自定义target保留自己的ID或明确unversioned，不硬写SM90。模型ID最多64个ASCII字母/数字/`.`/`_`/`-`；无效/过长身份明确回退`unversioned-target`，不截断成看似完整的新ID。4096-byte最小trace摘要预算仍需成立。

### 实际IR诊断

新增`LayoutIRDiagnostics.h/.cpp`，公开纯函数：

```cpp
llvm::json::Object collectLayoutIRArithmetic(Operation *operation);
```

分析函数不改IR、不读取求解中的候选/赋值、不调用评分、不推断物理指令。支持范围：arith/math dialect内带Elementwise trait、单结果的操作，排除常量；包括算术、比较和cast，分别按真实opcode及结果element dtype分组，不能合并成同一单位耗时。静态RankedTensor结果计各维extent的安全乘积；int/float/index标量计1。动态/unranked/vector/多结果等未支持情况进入有序`unavailable_operations`并令整体`status=partial`，不补零冒充完整。仅对arith/math的相关计算记录这些未支持项；常量/Frisk MMA/Reduce/Broadcast/ConvertLayout/存储/控制流包装不算普通算术。控制流内arith/math的**语法op**可被访问一次，明确不估算执行次数。

输出结构固定：

上述动态/零元素等测试针对纯诊断helper的MLIR输入，不扩大Frisk布局推断支持域；pipeline仍先通过既有验证/求解，不能借诊断放宽shape限制。

```json
{
  "schema_version": 1,
  "status": "available",
  "counting": "static-logical-result-elements-once-per-ir-op",
  "execution_frequency": "not-applied",
  "entries": [
    {"opcode": "arith.mulf", "dtype": "f32", "op_count": 1,
     "element_ops": 64, "saturated": false}
  ],
  "unavailable_operations": []
}
```

entries按opcode/dtype排序；unavailable按IR walk顺序，含opcode和稳定reason（如`dynamic-shape`、`unsupported-result-type`、`multiple-results`）。无地址/指针或不稳定名字。计数使用uint64饱和，并保留每组`saturated`；乘积先检查零维，再乘其他维，不能依赖可能溢出的`getNumElements()`，不能分配与元素数成比例的容器。

pipeline增加默认关闭的`ir-diagnostics`选项；C++ overload最后追加默认`bool irDiagnostics=false`，保持既有调用源兼容。只有完整normalize/infer/preserve/optimize与最终actual verifier成功后，才对staged实际IR调用一次分析器。stats增加`actual_ir_diagnostics`（默认/失败为null），启用成功时内容：

```json
{"stage":"post-conversion-optimization", "arithmetic": {"schema_version":1}}
```

其中arithmetic为上面的完整对象。trace复用同一对象记录可截断的`actual-ir-diagnostics`事件，**不是**`cost-contribution`；它被截断也不影响强制模型ID和成本对账。standalone infer只发布模型ID，不声称自己已经经过完整转换优化。用户若仅启用诊断而不指定文件，C++报告仍可取；不引入额外stdout、IR属性或隐藏全局状态。失败时不发布成功诊断，保持文件错误/IR事务回滚规则。

文件事务沿用既有逐文件原子发布，不新增多文件共同提交承诺：IR及caller报告在任何写失败后不提交/不保留成功诊断；若stats已成功写出、后续trace/profile写失败，已发布stats仍是该次**已验证staged IR**的记录，不是整个调用或IR提交成功的证明。

## Task 1: 完整模型身份与九项覆盖说明

**Files:** 修改`include/Dialect/Frisk/Analysis/{LayoutTarget.h,LayoutTrace.h,LayoutStatistics.h}`、`include/Dialect/Frisk/Target/SM90/SM90CostModel.h`；`lib/Dialect/Frisk/Analysis/{LayoutTrace.cpp,LayoutStatistics.cpp,LayoutCandidateSolver.cpp,LayoutRelaxation.cpp}`中实际需要的身份传播点；`lib/Dialect/Frisk/Target/SM90/{SM90CostModel.cpp,SM90LayoutTarget.cpp}`；`lib/Dialect/Frisk/Transforms/{LayoutInfer.cpp,PassPipelines.cpp}`；测试`LayoutTraceTest.cpp`、`LayoutStatisticsTest.cpp`、`SM90CostModelTest.cpp`；新增`guoqiao/optimize_plan/p1_3_cost_model_coverage.md`。

**Interfaces:** 提供上述`costModelID()`、`getSM90LegacyCostModelID()`和两份报告身份字段，供Task2复用，不改变评分函数签名或行为。

若trace/stats需要共用ID校验，允许在已有`LayoutCostModel.h/.cpp`增加纯元数据helper，避免两个不一致的校验实现；不得顺带改动任何评分公式。

- [x] RED：先对当前trace/stats JSON断言存在完整ID，观察真实失败；新接口可最小声明/空stub保证测试可运行，不能以编译错误代替RED。

```cpp
auto parsed = llvm::json::parse(trace.serialize());
ASSERT_TRUE(parsed);
EXPECT_EQ(parsed->getAsObject()->getString("cost_model_id"),
          "sm90-layout-lex9-legacy-v1");
EXPECT_EQ(parsed->getAsObject()->getInteger("schema_version"), 1);
```

- [x] 实现ID来源/传播/JSON输出。以target提供的完整身份更新summary，不能依赖可截断coverage或仅可选事件；invalid/too-long ID明确unversioned，保留4096-byte约束。
- [x] 测试正常/事件截断/字节截断/失败/无求解记录、最大摘要字段、custom target、不合法长ID、重复序列化稳定；完整cost-contribution加omitted仍等于finalCost。
- [x] 发布九项覆盖表：①Copy/MMA/Reduce/access工作量代理（未计普通算术时延）；②逐发射32B-sector代理（非DRAM流量）；③Shared phase内distinct-word bank代理（不含全部MMA Shared访问）；④owner/staging字节＋同步及clone-byte-work、legacy深度频率；⑤逻辑carrier槽位和（非活跃峰值/物理寄存器/spill）；⑥去重Shared根跨度（非occupancy/后端allocation复用）；⑦replication−1代理和；⑧转换/clone静态代码量代理（非机器码字节）；⑨确定性保留字段与实际实现情况。逐项核对现源码，说明未建模的latency/cache/overlap/MMA Shared普通算术，并列出旧子模型版本与完整ID映射。
- [x] Debug/Release定向构建与测试，给出命令/RED/GREEN/实际数量/日志；独立spec/quality复审关闭重要问题后Task2。禁止自动commit。

Task 1 验收：Debug/Release各42项通过；复审发现的外部trace早退身份遗漏已用8场景回归修复，wide-element bank口径也已澄清。独立spec/quality复审通过；详见`/tmp/frisk-p13a-task1-report.md`及`/tmp/frisk-p13a-task1-review.md`。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutTraceTest.*:LayoutStatisticsTest.*:SM90CostModelTest.*'
cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutTraceTest.*:LayoutStatisticsTest.*:SM90CostModelTest.*'
```

## Task 2: 实际IR算术计数及pipeline按需接入

**Files:** 新增`include/Dialect/Frisk/Analysis/LayoutIRDiagnostics.h`、`lib/Dialect/Frisk/Analysis/LayoutIRDiagnostics.cpp`和`unittests/Dialect/Frisk/Layout/LayoutIRDiagnosticsTest.cpp`；修改Analysis/单测CMake、`LayoutStatistics.h/.cpp`、`PassPipelines.h/.cpp`；新增`test/Transforms/layout-ir-diagnostics.mlir`、`test/Transforms/Inputs/check-layout-ir-diagnostics.py`；必要时扩展已有conversion/Attention集成测试，不改选择语义。

**Interfaces:** 使用Task1模型ID；产出`collectLayoutIRArithmetic(Operation*)`、`actual_ir_diagnostics`和默认关闭的pipeline `ir-diagnostics`。

- [x] RED：真实parsed IR里`tensor<8x8xf32>`的mulf→exp→divf，每op预期64个element-op；mulf两个consumer仍只计1个op/64，真实复制一个mulf才计2个op/128。记录功能缺失断言，不以坏IR/缺include为RED。

```mlir
func.func @arithmetic(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %m = arith.mulf %a, %b : tensor<8x8xf32>
  %e = math.exp %m : tensor<8x8xf32>
  %d = arith.divf %e, %m : tensor<8x8xf32>
  return %d, %m : tensor<8x8xf32>, tensor<8x8xf32>
}
```

- [x] 实现按op访问、按opcode/result dtype分组的纯分析，严格遵循上面的支持域/稳定排序/partial/saturation契约。多use不重计；只按实际留下来的op统计，删除的原op不算，保留的原op和clone各算。ConvertLayout不是算术，其结果下游算术仍按逻辑shape而不是carrier副本计算。
- [x] RED/GREEN补不同dtype/f16 cast、标量、动态shape、多结果/不支持结果类型、零元素与超uint64乘积饱和、已有for/if内静态语法计数（明确frequency未应用）、无算术空表、重复调用稳定且IR不变。
- [x] pipeline追加默认false参数/CLI选项，成功最终验证后收集一次，stats保存同一对象，trace可选事件复用；默认/失败null，不影响trace强制ID、成本贡献和事务。增加实际pipeline失败/文件写失败回滚及截断trace仍有ID/成本对账测试。
- [x] 实际转换优化/重计算集成：至少一例真正保留clone的优化后IR，一例共享转换多consumer。独立遍历最终IR核对op数量/shape，证明计数对应实际结果，不从solver计划反填；diagnostics off/on的IR、bindings、完整finalCost、非计时求解stats（剔除新增诊断本身）完全一致。重放稳定。
- [x] Debug/Release新测试＋受影响Trace/Statistics/Conversion/Pipeline定向测试，新增lit真实CLI开关和JSON核验；独立复审关闭重要问题。不要运行正式性能采集，root在最终源码冻结后负责。

Task 2 验收：Debug/Release各62项、相关lit各6项通过；复审建议的回滚区分性测试加强后两种构建定向各1项通过（仅测试修改，不冒称完整62项重跑）。固定ABI Attention CLI与trace核验通过。独立spec/quality复审已通过，证据在`/tmp/frisk-p13a-task2-{report,review}.md`；最终阶段仍需Task 3。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests frisk-opt --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutIRDiagnosticsTest.*:LayoutTraceTest.*:LayoutStatisticsTest.*:LayoutPipelineTest.*:LayoutConversionChoiceTest.*'
cmake --build build --target FriskLayoutUnitTests frisk-opt --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutIRDiagnosticsTest.*:LayoutTraceTest.*:LayoutStatisticsTest.*'
build/bin/frisk-opt test/Transforms/attention-layout-fixed-abi.mlir '-frisk-layout-pipeline=ir-diagnostics=true stats-file=/tmp/frisk-p13a-fixed-stats.json trace-file=/tmp/frisk-p13a-fixed-trace.json' -o /tmp/frisk-p13a-fixed-output.mlir
```

## Task 3: P1.3a最终回归、观测开销与报告（root）

**Files:** 新建`guoqiao/inference_detail/p1_3a_implementation_report.md`及`guoqiao/optimize_plan/p1_3a_measurement_results.json`/证据目录；更新本计划、长期实现计划/体系比较/优化清单。保留Q1/Q2原文数据，不重新运行旧230行来混入新模型身份。

- [x] 冻结源码，完整Debug主unit及lit、Release受影响测试、新/现Python/CTest守门；实际计数，不复制旧505/69数字。完整selection慢Debug已在Q1/Q2验收，本阶段按受影响模型身份/trace路径选择Release及定向Debug，不无差别重复18分钟核心循环。
- [x] 对小图、自由Attention、固定ABI Attention跑旧基线/新默认及新诊断开关：输出IR/binding/完整成本不变；新增模型身份和诊断字段单列，不要求跨版本JSON逐字节一样。开启诊断不改变其他非计时求解统计。
- [x] 全部构建/测试关闭后，Release每个输入×诊断off/on×(1预热＋5测量)=36次串行CLI；记录原始wall/RSS、输入/源码/二进制/config身份，真实diag counts与实际优化IR。无GPU或编译加速预设，小差异不作显著性宣称。
- [x] 独立最终source/evidence/docs review；关闭重要问题，报告ID、九项表、真实Mul/Exp/Div例子及不计时延/不改选择的边界。然后细化并实施P1.3b，再P1.3c。

Task 3 已完成测量及回归：524个不同Debug测试的集合覆盖（521个修复前完整覆盖＋最终修复定向新增3项）、70 lit、最终修复26单测/3身份/25 Python两种构建及工具诊断lit；不是一次性最终二进制全量重跑。最终集成发现并修复JSON动态字符串借用，以及受控实验已知目标身份遗漏；独立修复复审、整体源码及最终证据/文档复审均通过，无遗留发现项。原始数据/边界见[报告](../inference_detail/p1_3a_implementation_report.md)。P1.3a已验收，继续P1.3b→c。

## 自审

覆盖既有§4.1全部要求：模型ID与schema分开且截断保留；九项单位/未覆盖说明；实际优化IR的多use/clone计数；诊断不进cost contributions；开关等价及成本对账；报告和独立复审。P1.3b/c未提前实施；不存在扩大权重、预算或GPU范围的步骤。状态：Task 1–3均已验收，P1.3a完成。
