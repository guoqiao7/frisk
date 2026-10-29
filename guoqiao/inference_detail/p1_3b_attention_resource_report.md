# 完整 Attention 资源诊断与 P1.3b 验收报告

日期：2026-10-28。目标为 SM90a，固定一个 CTA、128线程的完整单 tile Attention。P1.3b 已完成验收，源码、数据与文档的最终独立复审通过。本报告的资源是最终已验证IR的保守逻辑存活诊断，不是GPU实测寄存器、后端Shared复用、spill或occupancy。

## 1. 主要结果

| 输入 | Tensor SSA数 | 每线程槽位峰值 | 每线程逻辑字节峰值 | Shared活跃峰值/字节 | Shared声明容量/字节 |
| --- | --- | --- | --- | --- | --- |
| 自由布局 Attention | 16 | 106 | 424 | 6144 | 6144 |
| 固定行主序ABI Attention | 16 | 106 | 424 | 6144 | 6144 |

两类输入均成功推断，资源状态均为 `available`。旧P1.3a、新版关闭诊断、新版开启诊断的输出IR逐字节相同，完整九项成本与非计时求解统计相同；旧算术诊断不变，最终IR重放稳定。独立核算器从实际IR重新构造16个Tensor的def-use区间和3个Shared根的访问区间，逐项核对JSON，不只复算诊断自身的区间。

槽位和字节峰值首次位于操作20，操作25再次达到峰值；Shared峰值首次位于操作15。两套布局的逻辑峰值相同，不代表地址映射、访存成本或GPU性能相同。

## 2. 完整计算链与分析位置

Q/K/V为 `[64,16]f16`，O为 `[64,16]f32`：

```text
Global Q/K/V → Copy到Shared → SS MMA(QKᵀ，f32累加) → ×0.25
  → ReduceMax → Broadcast → Sub → Exp
  → ReduceSum → Broadcast → Div → P转f16
  → RS MMA(PV，f32累加) → Store O
```

自由输入允许选择外部存储布局；固定ABI输入保持Q/K/V/O的外部行主序。外部Global Q/K/V/O根地址只声明alignment=1；Shared allocations及其contracts声明alignment=1024。三次Copy均为同步vector路径，`vector_bytes=1`。本轮没有改变ABI、搬运路径、MMA geometry、候选或选择。

分析位于 `post-conversion-optimization`，在staged IR最终验证成功后收集，开关为 `ir-diagnostics=true`。域是单CTA的 `frisk.parallel ranges=[1]`，捕获值作为live-in。编号0为入口，1–32按真实块内操作编号，包括元数据与terminator。

## 3. Broadcast窄特例

修复前，两类Attention均推断成功，但操作20、25的 `linalg.broadcast` 含隐式标量region，触发一般region拒绝规则，资源报告为unavailable。

现在只将规范ranked-Tensor Broadcast作为原子Tensor操作：检查真实op及规范region，按外层输入、destination和结果的SSA边界计数；隐式标量参数/yield不新增Tensor载体。非规范body、其他region、if/for/while、多块和不支持的并行域仍明确不可用。资源只用于诊断，不参与评分、剪枝、tie-break或合法性判断；模型仍为 `sm90-layout-lex9-legacy-v1`。

## 4. 全部Tensor实际存活区间

两类输入分别核算后，以下区间、dtype与槽位逐项一致。区间含首尾：执行边界同时保存待读取的输入和新结果。域外 `%cst*` 捕获常量保守地从入口计数。

| ID | 实际SSA | 含义 | shape/dtype | 槽位/线程 | 字节/线程 | 闭区间 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `%cst_1` | SS零累加器 | 64x64/f32 | 32 | 128 | [0,16] |
| 1 | `%cst_0` | scale常量0.25 | 64x64/f32 | 32 | 128 | [0,17] |
| 2 | `%cst` | RS零累加器 | 64x16/f32 | 8 | 32 | [0,28] |
| 3 | `%6` | QKᵀ scores | 64x64/f32 | 32 | 128 | [16,17] |
| 4 | `%7` | scaled scores | 64x64/f32 | 32 | 128 | [17,21] |
| 5 | `%8` | 行最大值 | 64/f32 | 2 | 8 | [18,20] |
| 6 | `%9` | Broadcast destination | 64x64/f32 | 32 | 128 | [19,20] |
| 7 | `%broadcasted` | 广播最大值 | 64x64/f32 | 32 | 128 | [20,21] |
| 8 | `%10` | 减去最大值 | 64x64/f32 | 32 | 128 | [21,22] |
| 9 | `%11` | 指数结果 | 64x64/f32 | 32 | 128 | [22,26] |
| 10 | `%12` | 行指数和 | 64/f32 | 2 | 8 | [23,25] |
| 11 | `%13` | Broadcast destination | 64x64/f32 | 32 | 128 | [24,25] |
| 12 | `%broadcasted_4` | 广播指数和 | 64x64/f32 | 32 | 128 | [25,26] |
| 13 | `%14` | 归一化概率 | 64x64/f32 | 32 | 128 | [26,27] |
| 14 | `%15` | f16概率P | 64x64/f16 | 32 | 64 | [27,28] |
| 15 | `%16` | PV输出 | 64x16/f32 | 8 | 32 | [28,31] |

`tensor.empty`也按当前保守SSA模型计carrier槽位，不表示后端一定为未初始化destination分配同量物理寄存器。lowering消除、常量实现和寄存器复用未建模。f16逻辑槽计2字节，f32计4字节，不推断物理packing。

## 5. 峰值来源

操作20（广播最大值）同时存活：

```text
RS零累加器 %cst           8槽 /  32字节
scaled scores %7        32槽 / 128字节
行最大值 %8               2槽 /   8字节
destination %9          32槽 / 128字节
结果 %broadcasted       32槽 / 128字节
合计                    106槽 / 424字节
```

操作25以exp、行求和、第二个destination和广播结果形成同样峰值。输入、destination和结果在操作边界保守同存活，不假设原地复用。一个SSA被多个consumer使用仍只计一次。106是每线程逻辑carrier上界，不能读作106个物理寄存器。

## 6. Shared根的容量与存活

| 根 | 用途 | 容量/字节 | 首次访问 | 最后访问 | 保活至域末 |
| --- | --- | --- | --- | --- | --- |
| %alloc | Q | 2048 | 13 | 16 | False |
| %alloc_2 | K | 2048 | 14 | 16 | False |
| %alloc_3 | V | 2048 | 15 | 28 | False |

三个 `[64,16]f16` allocation各2048字节。操作15完成第三次Copy时三根同时活跃，峰值6144字节；Q/K在操作16被SS MMA最后使用，V保留至操作28的RS MMA。view按底层root合并，不重复计容量。

本例声明容量也为6144字节，两个指标恰好相同。已有真实IR回归中，两个4KiB根错开使用时活跃峰值4KiB、声明容量8KiB；同一4KiB根的两个alias不会计为8KiB。本次同步vector Attention不证明TMA/WGMMA lowering的异步安全。

## 7. 完整资源时间线

以下来自最终IR独立重算，完整活跃SSA集合保存在原始审计JSON。入口包含域外Tensor captures。

| 操作 | 实际opcode | 结果 | 活跃槽位/线程 | 逻辑字节/线程 | 活跃Shared/字节 |
| --- | --- | --- | --- | --- | --- |
| 0 | entry | — | 72 | 288 | 0 |
| 1 | frisk.storage_contract | — | 72 | 288 | 0 |
| 2 | frisk.layout_view | %0 | 72 | 288 | 0 |
| 3 | frisk.storage_contract | — | 72 | 288 | 0 |
| 4 | frisk.layout_view | %1 | 72 | 288 | 0 |
| 5 | frisk.storage_contract | — | 72 | 288 | 0 |
| 6 | frisk.layout_view | %2 | 72 | 288 | 0 |
| 7 | frisk.storage_contract | — | 72 | 288 | 0 |
| 8 | frisk.layout_view | %3 | 72 | 288 | 0 |
| 9 | frisk.storage_contract | — | 72 | 288 | 0 |
| 10 | frisk.layout_view | %4 | 72 | 288 | 0 |
| 11 | frisk.storage_contract | — | 72 | 288 | 0 |
| 12 | frisk.layout_view | %5 | 72 | 288 | 0 |
| 13 | frisk.copy | — | 72 | 288 | 2048 |
| 14 | frisk.copy | — | 72 | 288 | 4096 |
| 15 | frisk.copy | — | 72 | 288 | 6144 |
| 16 | frisk.mma | %6 | 104 | 416 | 6144 |
| 17 | arith.mulf | %7 | 104 | 416 | 2048 |
| 18 | frisk.reduce_tensor | %8 | 42 | 168 | 2048 |
| 19 | tensor.empty | %9 | 74 | 296 | 2048 |
| 20 | linalg.broadcast | %broadcasted | 106 | 424 | 2048 |
| 21 | arith.subf | %10 | 104 | 416 | 2048 |
| 22 | math.exp | %11 | 72 | 288 | 2048 |
| 23 | frisk.reduce_tensor | %12 | 42 | 168 | 2048 |
| 24 | tensor.empty | %13 | 74 | 296 | 2048 |
| 25 | linalg.broadcast | %broadcasted_4 | 106 | 424 | 2048 |
| 26 | arith.divf | %14 | 104 | 416 | 2048 |
| 27 | arith.truncf | %15 | 72 | 224 | 2048 |
| 28 | frisk.mma | %16 | 48 | 128 | 2048 |
| 29 | frisk.storage_contract | — | 8 | 32 | 0 |
| 30 | frisk.layout_view | %17 | 8 | 32 | 0 |
| 31 | frisk.tile_store | — | 8 | 32 | 0 |
| 32 | frisk.end | — | 0 | 0 | 0 |

## 8. 与旧九项成本的区别

| 输入 | 最终完整CostVector |
| --- | --- |
| 自由布局 | `[8293, 544, 0, 0, 428, 6144, 6, 0, 0]` |
| 固定ABI | `[8293, 800, 0, 0, 428, 6144, 6, 0, 0]` |

九项与字典序均不变。第五项 `spillRiskAndRegisters`仍为旧可加代理，不能与106槽等同；第六项 `sharedBytesAndOccupancy`没有被全局存活峰值替换。分量峰值不能直接相加为全函数峰值，本阶段不宣称已用真实峰值优化选择，也不将O06长期目标全部标完。

## 9. 验收

- 行为RED在修复前复现；正例使用验证通过的真实MLIR，非规范region和一般控制流仍不可用。
- 当前源码的Debug/Release定向测试、相关lit与Python/CTest守门通过。最终全Release `549`项、`65`个suite通过，无失败、错误、跳过或禁用。分阶段数量与精确命令见实施记录，不重复累计历史测试。
- 三输入各旧默认/旧诊断/新默认/新诊断/最终IR重放，共15次语义核验。每个原输入的前四种配置保持IR、绑定、完整成本及非计时求解统计相同，旧算术诊断不变；重放单独核验输出IR及实际IR诊断稳定，不要求重放的求解统计等同首次推断。两Attention全部SSA/root独立核对通过。
- 32/64槽、转换边界同存活、共享转换SSA不重计、Shared同根去重/异根错开的真实IR例子均由行为测试覆盖。
- 所有构建测试结束后才做36次串行CLI观测；源码、两工具、输入、驱动和独立审计脚本均冻结，并在结束时复验身份未变。

[实施及测试记录](../optimize_plan/p1_3b_artifacts/completion/implementation-report.md)；[独立IR核算器](../optimize_plan/p1_3b_artifacts/completion/audit_attention_ir.py)；[原始测量JSON](../optimize_plan/p1_3b_measurement_results.json)；[证据索引](../optimize_plan/p1_3b_artifacts/README.md)。

## 10. CPU诊断开销观测

每输入off/on各1次预热、5次测量，串行共36次（6预热、30有效样本）。wall time包含进程启动；RSS为5个有效样本中的进程峰值最大值。

| 输入 | off中位/秒 | on中位/秒 | on/off | off峰值RSS/KiB | on峰值RSS/KiB |
| --- | --- | --- | --- | --- | --- |
| small | 0.152433 | 0.153024 | 1.0039 | 56428 | 56636 |
| free | 13.356795 | 13.355220 | 0.9999 | 73796 | 72752 |
| fixed | 7.477001 | 7.460543 | 0.9978 | 76424 | 75916 |

主机无CPU隔离、采样顺序未随机化、样本数有限；这些是本次运行的观测，不作统计显著性或GPU性能结论。每个样本均复核IR、成本与统计等价。

## 11. 身份与复现

- 旧P1.3a工具：`da119bda3bef8b1bb502bb978e831608273e5c9e3b806c91e4f7e7488a550356`。
- 本次Release工具：`fa4120115d2e0053e91f76b3c8d9b4dc55c9c3c318c7178ec235c79311c16e29`。
- 驱动：`5990ee89d7d29a6ba7a9939903febc2205f0facccf2115b96fee228ede95b108`。
- 独立审计：`39b26fe0182d6736de48ca001dc60e3752fa692ef9fd373a044b1e096055d23f`。
- 自由Attention最终IR：`effdffc56e504038e713d49183fc2ad19211a95abd9b563654bdd2b128012cb1`。
- 固定ABI最终IR：`7d4bca857c49f7e8489d638213a2bf46cac9e187366bb12f6f691efea90de21a`。

源码身份是测量JSON中的逐文件清单，含此前未提交P0/P1增量；仅checkout Git HEAD不能复现。归档保留源码快照、构建配置、RED/GREEN、回归、原始IR/stats/trace、独立审计与复审。外部LLVM/MLIR、系统runtime及工具二进制不随源码归档提供，重建后应重新记录身份。

原工作区复现一个输入：

```bash
build-m6-release/bin/frisk-opt test/Transforms/attention-layout-fixed-abi.mlir \
  '-frisk-layout-pipeline=ir-diagnostics=true stats-file=/tmp/attention-resources.json' \
  -verify-each -o /tmp/attention-resources.mlir
```

## 12. 范围

本阶段交付规范Broadcast原子统计、完整Attention资源诊断、SSA/root逐项核算、开关等价、回归与CPU观测。P1.3c循环频率、P1.4 GPU执行、物理寄存器/occupancy、Shared后端复用与基于全局峰值的搜索优化仍属于后续阶段。
