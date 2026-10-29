# P1.1：受限符号证明与编译开销验收

日期：2026-10-21。状态：用户批准方案 A 后先记录契约，再完成实施与测试，结果见 [实施报告](p1_1_implementation_report.md)。对应 [优化清单](layout_inference_optimization_plan.md) O07 的受限子阶段；不代表 O04–O11 或 O07 全部目标完成。

## 1. 范围与基线

基线为本地 main 的 `b237911` 加已验收、未提交的 P0 工作区增量。保留原改动和构建目录；直接在源目录工作，不创建 worktree，不自动 commit/push。实施前保存基线 executable、工作区差异及测量记录，避免把 HEAD 误当 P0 实现基线。

先测量，再实现静态 bit-linear ownership 与 Broadcast 的符号证明，最后做正确性、完整推断和时间/内存对照。不改九项 CostVector 含义、字典序、exact=256、beam=32、总探测预算=65536、邻域预算=256 或外部 ABI。不新增 MMA 几何、Copy 指令、单位维度广播、动态/非二次幂支持、通用 reshape、GPU lowering。

## 2. 支持域与安全边界

- ownership：静态正二次幂逻辑形状、合法五维硬件 topology 和 BitLinearLayoutMap；保留现有 carrier 名称、位宽/拓扑匹配、lane 上限、矩阵维度、复制因子和 writer policy 检查。符号计算按位数设界，不按点数分配；整数乘法/移位必须检查溢出。
- 输入 carrier 可重排、可省略；被省略但 topology 非单位的 carrier 必须作为零列加入完整硬件域，不能漏算复制或唯一写者。
- 符号分析的硬件总位数和逻辑总位数分别最多 256，超过返回 Unknown；沿用显式 carrier 和逻辑维度最多 16 位的现有检查。以 APInt 表示完整位域，复制因子按指数比较，不乘出可能溢出的全硬件点数。
- Broadcast：保持当前非零 rank、每轴大于 1 且为二次幂、相同 dtype、有序插入轴、同一 CTA 且线程 topology 相同的支持域；允许不同 register 基、register 数量和复制。不是源/目标 register 编号相等。
- `enumerateExecutionLayout` 继续提供真实表，保留 65536 点上限；需要表的 vector/storage/Reduce/Copy/MMA 证明仍遵守各自限制。符号 ownership Proven 不构造表，也不使 TMA/向量访问自动合法。
- 非适用子类保持旧枚举或保守失败/Unknown，不把未证实条件当作 Proven。畸形输入不能因为缺少矩阵/属性而崩溃；先验证结构再做代数。保留三值语义与可核验失败反例。

## 3. 证明方法

### 3.1 Ownership

将输入矩阵规范到完整硬件位空间，得到 `y=Mh`。输出位宽总和为 m，硬件位宽总和为 n。

- 覆盖全部逻辑坐标：`rank(M)=m`。
- 覆盖成立时，每个逻辑值持有者数量为 `2^(n-m)`，必须等于编码 replication。
- `first_owner`：覆盖和复制检查通过即存在规范最小持有者；本接口不返回持有者枚举表。
- `all`：还必须 `ker(M)={0}`，即 `rank(M)=n`，否则存在重复写者。
- 缺覆盖时构造不在像空间的逻辑坐标；重复写者可用非零核向量构造硬件坐标，使用既有 `(cta,warp_group,warp,lane,register)` 顺序。反例不必与旧枚举发现的第一个相同，但必须可独立核验且确定。

复用已有 GF2Matrix，不另建通用线性代数框架；零位宽/零列情形单独正确处理，不能违反 GF2Matrix 的非空矩阵前置条件。

### 3.2 Broadcast

源映射为 `x=A_r r_s + A_t t`；删除输出广播轴后的映射为 `x=B_r r_d + B_t t`。在双方 ownership 和线程域通过后，同线程可取当且仅当：

```text
image([B_r, B_t XOR A_t]) ⊆ image(A_r)
```

用秩/列空间判定，不遍历所有持有者。若包含关系失败，构造对应目标线程和寄存器，给出实际无法取源值的输出逻辑坐标。投影准备中仅需要 coverage 的检查可改用 ownership 证明；实际 reduction 投影本身若仍有枚举界，不宣称一并解除。

## 4. 测量与可核验性

实施前对自由布局和固定 ABI Attention 各进行 1 次预热、5 次串行测量，保存 IR、stats、wall time、进程 max RSS、binary hash。读取已有 collection/preparation、propagation、search/solve、verification 阶段计时；嵌套计时不相加为独立热点。

增加可重复的证明微基准，比较同一合法输入的符号证明与保留的枚举 oracle，包含 ownership、Broadcast、小域、65536 边界和超界案例；记录状态、点数、重复次数和时间。若没有系统采样 profiler，只报告阶段测量与隔离微基准，不声称已取得全流水线逐函数热点或缓存命中率。

不新增全局长期缓存或不完整 cache key。proof.reason 区分符号路径与枚举预算失败，既有 trace 拒绝记录按实际结果输出；不更改 trace schema。证明优化可能让原 Unknown 获得支持，因此全域候选/选择不强求永远与旧系统相同；共同可证明域必须语义一致，任何候选变化都要解释。

## 5. 验收

1. TDD：超界 ownership/Broadcast 先出现预期失败，再实现。保留旧枚举 oracle，测试正反例、输入 carrier 重排/省略、复制、错误 topology/矩阵、非满秩、线程不匹配、同线程不可取及反例有效性。
2. 小域确定性矩阵族或固定种子生成样本逐一与枚举比对，不仅比较正例；Broadcast oracle 枚举同一线程的所有源 register，与代数实现相独立。
3. 超过 65536 硬件点的合法 ownership 和 Broadcast 可 Proven，原枚举依然 Unknown；非法同线程广播仍拒绝。vector/storage 等消费表路径保持安全限制。
4. 自由/固定 ABI Attention 全链 IR、实际 verifier、重跑稳定、trace 成本与预算守门通过；全量 unit/lit/CTest 和相关 Python 检查通过。测试数量以最终日志为准。
5. 同机器同配置重复测量时间与峰值 RSS，对照源码与二进制身份。报告改善、持平或退化，不承诺固定加速比例、不作 GPU 性能结论。
6. 独立逐任务及整体复审；同步长期设计、实施计划、体系对比与优化清单。对比文档只更新 Frisk 本地证据，不宣称上游不存在相似方法。

## 6. 实施顺序

按 [执行计划](p1_1_execution_plan.md) 完成：基线测量 → ownership → Broadcast → 完整回归/证明微基准/Attention 对照。P1.2–P1.4 仍需各自具体契约，不在本轮实现。
