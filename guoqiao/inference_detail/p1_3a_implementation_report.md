# P1.3a：成本身份与实际 IR 算术诊断

状态：实现、回归、串行实验与最终源码/证据/文档复审均已通过，无未关闭发现项。具体契约见[执行计划](../optimize_plan/p1_3a_execution_plan.md)。本报告不把此前 Q1/Q2 的测试或测量算作本阶段结果。

## 1. 解决的问题

布局成本向量回答“在当前代理模型下选哪个合法布局”，不等于完整程序的执行时间。P1.3a 补充两种可核验信息：

- **模型身份**：trace 和 stats 的 `cost_model_id` 标识整套评分公式，不再依赖可能被截短的 coverage 文本。当前 SM90 为 `sm90-layout-lex9-legacy-v1`，JSON schema 仍是 1。
- **实际算术工作**：在完整流水线最终验证成功后，按真实 opcode 和结果 dtype 统计优化后 IR 的逻辑 element-op。它与九项成本分开，不参与求解、剪枝或选择。

九项的单位、实现依据与未覆盖部分见[覆盖表](../optimize_plan/p1_3_cost_model_coverage.md)。例如，sector 数不是 DRAM 流量，carrier 槽位和不是活跃峰值，未计入 Exp 时延不表示 Exp 免费。

## 2. 算术诊断的口径

对 `tensor<8x8xf32>` 的 `arith.mulf → math.exp → arith.divf`，各操作处理 64 个逻辑元素，分别报告 64，不把三种操作相加成统一时间。`mulf` 有两个 consumer 仍只有一个实际操作；若优化后确实留下原操作和一个重计算副本，则报告两个操作、128 个 element-op。若原操作被删除，则不再计它。

这是静态语法计数：不乘循环次数、线程数或布局副本数。常量及 MMA/Reduce/ConvertLayout 不算普通算术；动态形状等未支持情况明确标为 partial，而不是悄悄补零。循环频率属于后续 P1.3c，资源存活峰值属于 P1.3b。

实际流水线的使用入口：

```bash
build-m6-release/bin/frisk-opt test/Transforms/attention-layout-fixed-abi.mlir \
  '-frisk-layout-pipeline=ir-diagnostics=true stats-file=/tmp/attention.stats.json' \
  -o /tmp/attention.mlir
```

结果在 `actual_ir_diagnostics.arithmetic.entries`。关闭时整个 `actual_ir_diagnostics` 为 null；启用后，trace 也可记录独立的 `actual-ir-diagnostics` 事件，事件被截断不影响强制模型 ID 或成本对账。

注意输出沿用逐文件原子发布：如果 stats 已成功写出、后续 trace/profile 写失败，原 IR 回滚且 caller 报告的诊断清空，但先写出的 stats 仍可包含已验证暂存 IR 的诊断。它不是多个输出均成功或 IR 已提交的证明。

## 3. 完整 Attention 的实际结果

自由布局和固定外部 ABI 两个 Attention 都走完实际流水线，并经过打印/解析/重跑。两者的算术诊断相同：

| 实际操作 | 结果 dtype | IR 中操作数 | 逻辑 element-op |
| --- | --- | ---: | ---: |
| `arith.mulf`：scale | f32 | 1 | 4096 |
| `arith.subf`：减去行最大值 | f32 | 1 | 4096 |
| `math.exp` | f32 | 1 | 4096 |
| `arith.divf`：归一化 | f32 | 1 | 4096 |
| `arith.truncf`：P 转 f16 | f16 | 1 | 4096 |

每个结果都是 64×64；dtype 按结果记录，所以 Trunc 是 f16。原始实际 IR 和 stats 随[实验数据](../optimize_plan/p1_3a_measurement_results.json)归档，驱动还独立核对了最终 IR 中这五种操作的形状、dtype 和数量。这个核对器只用于这组已知 fixture，不是通用 MLIR parser。

开启诊断**没有改变选择**。旧基线、新默认、新诊断三组完整成本分别一致：

```text
小图 Copy：       [4, 20, 0, 0, 2, 0, 0, 0, 0]
自由 Attention：  [8293, 544, 0, 0, 428, 6144, 6, 0, 0]
固定 ABI：        [8293, 800, 0, 0, 428, 6144, 6, 0, 0]
```

三组均未饱和。两种 Attention 的 Global 事务代理不同，但普通算术逻辑工作相同，正好说明这些诊断不能互相替代。小图只有搬运，没有所支持的普通算术，返回 available 空表，而不是假装统计了 Copy 的机器指令。

## 4. 回归与修复记录

本阶段保持既有九项公式、字典序、候选、约束及预算。默认关闭算术诊断；最终需要验证开关前后的实际 IR、绑定、完整成本及非计时求解统计一致。

基线是实施前实际脏源码快照，而不是仅 HEAD：`/tmp/frisk-p13a-baseline-eKTIuI/tree`；旧 Release `frisk-opt` SHA256 为 `3c1f549298693d549ebad304b59bf43b0a830be631052d5c096b82524607bf8d`。其源码已随[Q1/Q2 证据](../optimize_plan/p1_2_selection_quality_artifacts/README.md)归档，旧数据不被新测量覆盖。

| 验证范围 | 实际证据 |
| --- | --- |
| 模型身份任务 | Debug/Release 各 42 项通过；两轮独立复审 |
| 算术/转换/流水线任务 | Debug/Release 各 62 项、相关 lit 各 6 项通过；加强回滚测试后定向各 1 项通过 |
| Debug 主单测完整覆盖 | 先覆盖当时全部 521 项：任务内 62 项＋其余四分片 115/115/115/114 项；最终修复定向 26 项包含 3 项新增生命周期测试。对照最终注册列表，**524 个不同测试全部覆盖，无跳过** |
| CLI 回归 | 全部 70 项 lit 通过；最终工具重链接后诊断 lit 在两种构建再各通过 1 项 |
| 选择实验兼容性 | 修复前 Release 20 项、Debug 定向 2 项通过；最终元数据修复后，两种构建各 3 项显式身份测试和 25 项真实 Python 驱动测试通过 |
| 辅助守门 | Debug 其余 8 项 CTest 通过，选择驱动那项在修复后通过；Release proof benchmark 与修复后的选择驱动通过；Python normalization 5 项及 API/工具检查通过 |

这里明确区分**完整既有覆盖＋最终修复定向验证**，没有声称一次性重跑了最终二进制的全部 524 项或完整最终 70 项。逐测试集合对账在证据中的 `unit-coverage.json`；详细命令、日志及旧/新文件摘要见[证据说明](../optimize_plan/p1_3a_artifacts/README.md)。

本轮发现并修复的实际问题：

- 外部 trace 在非法 trace/profile 参数早退时保留旧模型身份；新增 fresh/stale × 四种参数错误的 8 场景回归。
- JSON 借用 report 的模型 ID 或 MLIR context 的 opcode 字符串，原对象销毁后可能产生非法 UTF-8。通过原字符串原地修改与上下文释放测试复现，改为 JSON 自有字符串；真实选择实验的 Debug/Release Python 驱动随后通过。
- 手动组装的受控实验 stats/早退 trace 未初始化已知目标身份；改为从 target 取得，未改搜索行为，也没有重写旧 Q1/Q2 数据。
- 回滚测试原先对已物化的固定点重跑，区分性不足；改成用独立成功模块初始化报告，再对全新未处理 IR 制造失败，确认没有误提交。

构建保留既有 Ninja EOF 恢复、CMake CMP0116 deprecation，以及身份任务早期未修改地址测试的 LLVM/APInt 编译警告。一次 Python 守门初始调用缺少必填参数，纠正后通过；测量核对器最初误读了 encoding 内的 i1 矩阵类型，采样前已修正并验证。没有通过忽略解码错误或删掉失败测试来验收。

## 5. 诊断观测开销

所有构建/测试结束后，先运行 12 次语义核验调用，再对三类输入、off/on 各做 1 次预热＋5 次测量，共 36 次串行调用。测量包括 CLI 解析、完整流水线、验证、IR/stats 写出；不启用 profile 或 trace。各配置的 IR、非计时 stats 和诊断结果重复稳定；前后源码/工具身份相同。

| 输入 | off 中位秒 | on 中位秒 | on/off 变化 | off/on 最大 RSS（KiB） |
| --- | ---: | ---: | ---: | ---: |
| 小图 Copy | 0.173354 | 0.171851 | −0.87% | 56836 / 56864 |
| 自由 Attention | 13.386144 | 13.394233 | +0.06% | 73852 / 73752 |
| 固定 ABI Attention | 8.037111 | 8.030604 | −0.08% | 76632 / 76300 |

这些小差异只描述本机样本：没有 CPU 隔离或随机化执行顺序，样本数为 5，不能解释成显著加速/退化，更不是 GPU 性能结论。保留全部原始 wall/RSS、命令和输出，不以中位数隐藏单个样本。

复现入口（必须在没有其他构建/测试/采样工作时运行，输出目录必须全新）：

```bash
python3 measure.py /path/to/baseline/frisk-opt /path/to/current/frisk-opt \
  /tmp/new-p13a-results --workers-closed
```

驱动在 `measurements.tar.gz` 的 `evidence/measure.py` 中；旧工具 SHA256 为上节基线值，新 Release 工具为 `da119bda3bef8b1bb502bb978e831608273e5c9e3b806c91e4f7e7488a550356`。采样保存了 311 个源码/输入/测试/构建脚本文件的摘要，以及两个 CMakeCache 和驱动摘要。归档未打包工具二进制或外部 LLVM，因此重新构建要记录新的工具身份，不能冒称原二进制复现。

## 6. 剩余范围

本阶段没有改变评分公式，也没有实现活跃资源峰值或静态循环次数模型；后续按 P1.3b→P1.3c 推进。全局资源峰值仍不得直接写入可加 CostVector，普通算术计数也不是统一时延权重。

本阶段未运行 GPU、未自动提交或推送，保留已有未提交修改。
