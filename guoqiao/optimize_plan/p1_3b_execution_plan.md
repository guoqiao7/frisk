# P1.3b：实际 IR 的存活资源诊断

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development。按任务测试驱动实施并独立复审，用户已批准 P1.3a→b→c，不重新询问是否继续。

**Goal:** 用真实 SSA 与 Shared root 的存活区间解释资源峰值，区分逻辑峰值、声明容量和旧的可加资源代理。

**Architecture:** 独立只读分析器对有限支持域建立稳定区间和执行边界，输出自有字符串 JSON；在现有默认关闭的 `ir-diagnostics` 中复用最终已验证 staged IR，增加 `resources` 子对象。诊断与评分分离。

**Tech Stack:** C++17、MLIR IR/Frisk attrs 与 alias analysis、LLVM JSON、GoogleTest、lit/Python。

## Global Constraints

- 直接在源目录 main 修改，保留已有改动；不 worktree/stage/commit/push/GPU/P1.4。每次一个实现/构建拥有者，独立复审只读。
- 来源是[已批准契约§4.2](p1_2_to_p1_4_design_proposal.md#42-p13b存活资源诊断暂不替换搜索目标)。P1.3a 已验收，报告/源数据不改写。
- CostVector 九项、字典序、候选、约束、target evaluate/lowerBound、转换/重计算评分及预算完全不变。模型 ID 仍为 `sm90-layout-lex9-legacy-v1`。资源诊断不参与选择、剪枝、tie-break 或拒绝合法布局。
- 不将非可加全局峰值写入 CostVector 第五/六项；旧槽位和也不是新峰值的 lower bound。不推断物理寄存器、spill、occupancy 或后端 Shared allocation 复用。
- 不实现 P1.3c 频率改变；现有算术计数仍是每实际 op 一次，不乘频率。诊断关闭时不进行此分析。
- 基线是 `/tmp/frisk-p13b-baseline-6AVomg/tree`（P1.3a 的311文件源快照），同目录 Release `frisk-opt` SHA256 为 `da119bda3bef8b1bb502bb978e831608273e5c9e3b806c91e4f7e7488a550356`，不是 HEAD。

## 具体分析契约

### 支持域与不可用结果

入口为 `llvm::json::Object collectLayoutIRResources(Operation *operation);`，声明放在现有 `LayoutIRDiagnostics.h`，实现独立放在 `LayoutIRResources.cpp`，避免把算术计数与 liveness 混成一个函数。

- Module 中的单块 `func.func` / `frisk.kernel` 无嵌套 region 计算体；external declaration 不产生计算域。
- 一个 kernel 包含一个 `frisk.parallel ranges=[1]` 时分析其单块计算体；外层为 allocation、view/contract、标量准备等容器代码。该域显式报告 captures 为 live-in，包含其可达外部 Shared roots。其他范围、多个 parallel、多块、if/for/while 或其他嵌套 region 不支持，不能跳过后仍给父域完整峰值。
- 非并行外壳若还有域外 Tensor 计算，不能把内部 parallel 峰值冒称整个 kernel 峰值：该 kernel 标记 unavailable，说明 `tensor-work-outside-parallel`。Tensor 参数/跨域 capture 本身不因此拒绝，仍作为 live-in。
- 每个计算域独立给出结果；不把不同函数/CTA 域的峰值求和或宣称全模块峰值。任一域 unavailable 时顶层为 partial（全部不可用为 unavailable），域内数值峰值为 null，不补零。空模块是 available 空 scopes。
- 稳定 reason：`unsupported-control-flow`、`multiple-blocks`、`unsupported-parallel-domain`、`tensor-work-outside-parallel`、`unsupported-tensor-layout`、`unsupported-storage-alias`、`unsupported-shared-lifetime`。按 IR 顺序给出 scope/opcode；所有字符串自己持有，不借用被销毁的 report/context。
- 分析纯只读；失败是诊断状态，不使原本合法的 pipeline 失败。复用 alias/type 验证时局部捕获诊断，不额外输出 error/stdout。既有 alias 证明预算保留，不扩大推断支持域。

### Tensor SSA 区间与边界

计算域 entry 点编号0，操作按块顺序编号1..N（包括 terminator）。Tensor 稳定 ID 按实际 block arguments、首次出现的 captures、操作结果的 IR 顺序分配，不输出地址或借用指针。

- 只计真实 RankedTensor SSA：非空静态 shape、int/float dtype、可验证的 DistributedEncoding、CTA topology=1。未支持/缺少编码不能按0处理。
- 每值每线程逻辑 carrier 槽位是 encoding `topology[0]`；逻辑字节是槽位×`ceil(elementBits/8)`，包括 i1 向上取整；不推断 f16 packing 或物理寄存器复用。
- 使用到的参数/capture 为 `[0,last_use]`，标记 `live_in=true`；不使用的参数无需成为 live-in。结果从其定义操作到最后使用；即使结果没有 use，也在定义执行边界计一次。不按 use 数创建多个值。
- 操作边界同时包含尚需读取的输入和新输出，结束后才能移除最后使用的值；不能假设输入输出原地复用。返回只延长既有 SSA 的区间，返回契约不是另一个常驻值。
- Copy/Fill 的 execution-layout 属性、use 角色、layout contract 等伪变量不计 Tensor。ConvertLayout 真实输入/输出各自存活，实际 clone 和共享转换结果按实际 SSA 计数。
- 槽位峰值和字节峰值可在不同点发生，分别记录最早达到峰值的位置。使用检查过的 uint64 饱和计算；不能用饱和总和直接减去离开区间的项导致错误恢复。首版允许按 IR 边界重新求活跃集合之和，不按 Tensor 元素逐点展开。

具体验收（每个真实 Tensor 各32槽）：

```text
define A → consume A → define B → consume B      peak = 32
define A → define B → consume(A,B)               peak = 64
B = convert(A)                                  peak >= 64 at conversion
B = convert(A); consumer1(B); consumer2(B)       B is one SSA, not two values
function argument A → return A                  peak = 32, not 64
```

f32 32槽对应128逻辑字节，f16 32槽对应64逻辑字节；同时存活为64槽/192字节。复制布局也按实际 topology 槽位，而不是逻辑元素数乘 consumer 数。

### Shared root 容量与使用区间

- 使用 `analyzeStorageAlias` 追到同元素类型的真实 root；静态 cast/subview/layout_view 合并，两个 alias 不当成两个 allocation。不得自己写一套不带偏移/步长/降维的简化 alias 逻辑。
- 容量取完整 root descriptor 的可访问地址上界向上取整字节（`ceil(upperBit/8)`），相对 aligned pointer，含 descriptor offset/padding 前缀；不是子 view 的 payload 或形状乘积，也不重复叠加 view offset。根已验证 Storage mapping 的访问仍在这个 root span 内。
- `declared_shared_root_bytes` 对域所属函数/kernel 的内部静态 Shared allocations 和该计算域引用的外部 Shared roots 去重求和。外部引用包括域内metadata及capture；仅在parallel外壳metadata提及、但未被计算域引用的外部root不计入该域容量。外部 root 标明 `origin=external`，它是调用方声明容量，不是本函数实际分配。未使用的内部 allocation 仍在 declared 容量中。
- `logical_live_shared_root_bytes` 使用 root 的合并实际读写区间：从第一次实际存储访问至最后一次。allocation、alias/view 和 storage_contract 本身不是 payload 访问；没有实际访问时 live interval 为 null。两个alias的区间合并为一个保守连续区间。
- 对 `frisk.copy_contract.path=cp.async/tma`，本分析不证明低层异步完成，相关 Shared roots 从首次使用保活到域末并标记 `held_to_scope_end=true`；未知 Shared 用户同样保守保活。显式异步 token 逃逸、Shared dealloc/无法闭合的生命周期或根归属不明时，域 unavailable。`frisk.mma` 是同步数学 Tensor 操作，不能仅因将来用 WGMMA lowering 就错误改变当前 IR 语义。
- `declared_shared_root_bytes` 不随使用区间缩小。两个不同4KiB roots错开使用，logical-live峰值可为4KiB，但 declared 仍是8KiB；两个同根alias均访问4KiB，两个指标都不因此翻倍。

### JSON 与 pipeline

资源返回体固定有以下字段；数值仅在域 available 时存在有效值，unavailable 时为 null：

```json
{
  "schema_version": 1,
  "status": "available",
  "counting": "conservative-static-ssa-and-shared-root-liveness",
  "physical_registers": "not-modeled",
  "execution_temporaries": "not-modeled",
  "backend_allocation_reuse": "not-modeled",
  "scopes": [{
    "scope": "func@sample", "status": "available",
    "logical_live_carrier_slots_per_thread": 64,
    "logical_live_carrier_bytes_per_thread": 256,
    "logical_live_shared_root_bytes": 4096,
    "declared_shared_root_bytes": 8192,
    "peak_slots_at": 2, "peak_bytes_at": 2, "peak_shared_at": 3,
    "saturated": false,
    "tensor_values": [], "shared_roots": [], "unavailable_operations": []
  }]
}
```

`tensor_values` 含 `id,dtype,slots_per_thread,bytes_per_thread,begin,end,live_in`；`shared_roots` 含 `id,origin,bytes,begin,end,held_to_scope_end`。未使用 root 的 begin/end 为 null。scope 名与 opcode 为自有字符串，重复调用/打印解析后稳定；JSON 可在 MLIRContext 销毁后安全序列化。per-value/整体饱和传播不可遗漏。

原 `actual_ir_diagnostics` 增加 `resources`，`stage` 和 `arithmetic` 不变；不新增开关、stdout、IR属性或全局缓存。最终验证成功且 `ir-diagnostics=true` 时各纯分析调用一次，同一完整对象放入 stats/trace。失败/关闭为 null，沿用 P1.3a 的逐文件发布及 IR 回滚规则。resource unavailable 仍可带 available arithmetic，但不能变成合法性拒绝。

## Task 1: 纯资源分析器与独立例子

**Files:** `include/Dialect/Frisk/Analysis/LayoutIRDiagnostics.h`；新增 `lib/Dialect/Frisk/Analysis/LayoutIRResources.cpp`、`unittests/Dialect/Frisk/Layout/LayoutIRResourcesTest.cpp`；修改 Analysis/Layout unit 的 CMakeLists。

**Interfaces:** 在现有头文件追加 `llvm::json::Object collectLayoutIRResources(Operation *operation);`，仅实现上述纯分析；Task2接入现有pipeline。

- [x] 建立真实 parsed/built IR 的编码32槽正例，先用最小空stub跑出32/64断言的行为RED，不能以编译失败代替。
- [x] 两遍构造真实 SSA/Shared root 区间，按实际边界求保守峰值；独立测试上列32/64/转换64/共享一次/return不重计/f32-f16字节。例子使用真实 Frisk encoding，不能从 solver 估计反填。

```cpp
auto resources = collectLayoutIRResources(*module);
auto *scope = resources.getArray("scopes")->front().getAsObject();
EXPECT_EQ(scope->getInteger("logical_live_carrier_slots_per_thread"), 64);
EXPECT_EQ(scope->getInteger("logical_live_carrier_bytes_per_thread"), 192);
```

- [x] Shared同根alias4KiB不翻倍、不同根错开peak4KiB/declared8KiB、无use的allocation、带offset/stride/dropped axis的可证明alias、外部root声明、async保活/无法证明unavailable。
- [x] 固定单CTA parallel capture作为live-in，if/for/while、多块、ranges!=[1]及缺失encoding明确unavailable；不跳过控制流给完整峰值。测试使用大静态合法表示验证饱和，不分配与Tensor元素数成比例的表。
- [x] 重复调用稳定/IR不变/上下文销毁后JSON仍有效；Debug/Release资源单测＋必要alias/encoding定向测试，记录命令/RED/GREEN/实际数量；独立复审后Task2。无性能采样或自动提交。

```bash
cmake --build build-m6-release --target FriskLayoutUnitTests --parallel 12
build-m6-release/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutIRResourcesTest.*:LayoutIRDiagnosticsLifetimeTest.*:LayoutAliasAnalysisTest.*'
cmake --build build --target FriskLayoutUnitTests --parallel 12
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='LayoutIRResourcesTest.*:LayoutIRDiagnosticsLifetimeTest.*:LayoutAliasAnalysisTest.*'
```

## Task 2: pipeline集成、真实优化后IR与开关等价

**Files:** `lib/Dialect/Frisk/Transforms/PassPipelines.cpp`；`LayoutIRDiagnosticsTest.cpp`/`LayoutConversionChoiceTest.cpp`及新资源单测；`test/Transforms/layout-ir-diagnostics.mlir`及其JSON checker，必要时新 `layout-ir-resources.mlir`；不改评分器。

**Interfaces:** 消费Task1 `collectLayoutIRResources`，在现有启用诊断成功分支增加 `resources` 子对象，现有 bool 参数及两份JSON报告接口不变。

- [x] RED：实际pipeline启用后必须有resources，而关闭仍为null；合法但不支持资源分析的for/if程序仍能成功布局推断且resources明确unavailable。
- [x] 与arith同一最终IR、同一发布对象；测试失败及文件错误回滚、截断trace保留模型ID/完整成本对账、完整IR/绑定/完整成本/除诊断与timings外的stats开关相同。
- [x] 真实postoptimizer共享转换、保留clone的资源区间与独立最终IR def-use遍历一致；重跑稳定，不从计划/graph变量构造资源数。
- [x] Debug/Release资源/诊断/trace/statistics/conversion定向测试与新/既有诊断lit；独立复审。记录Ninja/CMake既有警告，不为确认环境重复无变化的构建。

## Task 3: Attention实验、维护报告与最终复审（root）

**Files:** 新建 `guoqiao/inference_detail/p1_3b_implementation_report.md`、`guoqiao/optimize_plan/p1_3b_measurement_results.json`及证据目录；更新长期实现计划/体系对比/优化清单/九项覆盖边界及本计划。

- [x] 受影响Debug/Release及全Release主unit、新/相关lit与Python/CTest回归；P1.3a全Debug覆盖作为未修改既有核心的基线，不无差别重跑23分钟无关Debug Attention路径。新增与变化路径必须在Debug验证。
- [x] 对32/64小例、Shared alias例、两类完整Attention输出真实资源区间/峰值/声明容量；给出旧CostVector资源和与新峰值的区别，不声称物理寄存器/occupancy。
- [x] 三类输入（小图、自由Attention、固定ABI）旧P1.3a默认/新默认/新诊断的IR/完整成本/非计时求解统计等价；旧arith诊断逐字段不变。最终所有构建/测试关闭后，新版off/on各1warmup＋5测量，共36次串行CLI，保存原始命令/输出/源和工具身份。
- [x] 独立整体source/evidence/docs复审，关闭重要问题，归档并标明支持域。P1.3c属于后续阶段，本轮未实施；O06全目标与P1整体不标为完成。

## 自审

§4.2的32/64、转换边界、共享一次、alias合并、声明/逻辑峰值分离、capture及控制流unavailable均有具体验收。JSON字符串独立拥有，避免重复P1.3a已修复的生命周期问题。模型/预算完全不变；未知异步行为只保守分析或unavailable。

历史检查点（2026-10-27，已由2026-10-28实现与实验结果取代）：Task1纯分析器与Task2通用pipeline集成均已通过独立复审。Task1首轮Debug/Release各35项、修复后各21项通过；Task2最终各87项与4lit通过；Task3全量Release545项、4项Python CTest及Debug既有Python/工具守门通过。证据见[阶段归档](p1_3b_artifacts/README.md)。两类Attention探针推断成功且IR与P1.3a相同，但第20、25个Broadcast令资源报告unavailable；资源特例及后续正式实验仍待确认，不能宣称整个P1.3b完成。

历史确认前提案（2026-10-27；2026-10-28已获授权并实现）：上述“不支持其他嵌套region”与Attention的named `linalg.broadcast`隐式标量yield region存在冲突。建议仅将规范Broadcast视作一个原子Tensor操作，按外层SSA边界计数，不泛化到其他region。确认前保留unavailable，不宣称Attention资源诊断已完成。


## 2026-10-28：Broadcast 窄特例与最终验收

用户在当前接续会话要求“交付一份完整 Attention 的资源诊断报告，并完成 P1.3b 验收”，明确接续前述窄特例与完整验收建议。本轮可直接实施，无需再次确认同一设计决定。

仅接受真实、规范的 ranked-Tensor `linalg.broadcast`：隐式单块标量 region 只将输入标量 yield 到结果；分析器将其视为原子 Tensor 操作，按外层 SSA 输入/输出的闭区间计数，不另计标量 block arguments。任意非规范 body、其他嵌套 region、控制流和不支持的计算域仍按原契约 unavailable；诊断不影响推断合法性、评分与模型身份。

当时获批的执行顺序（现已完成，实际结果见下方最终实验检查点）：先完成行为 RED/GREEN、受影响 Debug/Release 回归和独立源码复审；所有构建/测试结束后，再进行15次语义核验、36次串行 CPU 观测。两类 Attention 由独立、限定 fixture 的最终 IR def-use/Shared root 核算器逐值核对，不只从诊断 JSON 重算峰值。历史 Task1/2 和 2026-10-27 的 unavailable 探针保留作为修复前证据。本轮不实施 P1.3c/P1.4，不自动提交或推送。


## 最终实验检查点（2026-10-28）

2026-10-28：P1.3b已完成验收，最终独立复审通过。 规范ranked-Tensor Broadcast按原子SSA边界统计，非规范region仍unavailable；两类完整Attention均available，每线程峰值106逻辑槽/424字节，Shared活跃峰值/声明容量均6144字节。最终Debug定向50、全Release549（65 suites）、lit Debug5/Release8、CTest4及Python守门通过。15语义核验与36串行CPU观测通过，IR/完整成本/非计时求解统计/旧arith诊断不变，身份复验未变化。不涉及P1.3c/GPU或物理寄存器/occupancy，不自动提交/推送。

源码、最终证据与文档独立复审均已通过，最后复审检查项已完成。正式驱动15次语义核验先于36次串行CPU观测；两份实际IR审计16 SSA/3 roots及所有峰值、区间、容量逐项一致。报告与原始数据均已生成。
