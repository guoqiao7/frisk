# Frisk 完整布局推断过程与 Attention 实例

本文以当前已通过的简化 Attention 为例，说明 Frisk 如何从没有指定 Tensor 布局的程序，得到带有存储布局、线程分布和指令契约的 IR。重点是每一步解决什么问题、信息如何传播，以及最后实际选出了什么。

代码与实测口径：2026-10-11 的源码工作区，包含已完成的 Attention 增量。输入为 [`attention-layout.mlir`](../../test/Transforms/attention-layout.mlir)。本例验证布局推断和 IR 合法性，不代表已经完成 Attention 的 GPU lowering、数值验证或性能测试。

**若要直接查看每一步的具体中间结果，阅读第 13 节。**其中列出了全部 38 个变量的候选集、实际删减内容、联合合法组合、最终逐算子布局，以及固定其他选择后的成本对比。前 12 节解释这些结果的含义。

**若要按第 3 章的编译流水线逐步追踪状态变化，阅读第 14 章。**第 13 章是结果明细，第 14 章将这些结果按实际处理顺序串起来，区分分析图的变化、IR 的变化和最终事务提交。

## 1 输入是什么 输出是什么

本例计算：

```text
S = Q × Kᵀ × 0.25
M[i] = max_j S[i,j]
E[i,j] = exp(S[i,j] - M[i])
Z[i] = sum_j E[i,j]
P[i,j] = E[i,j] / Z[i]
O = cast_f16(P) × V
```

配置固定为 SM90a、128 线程、单头、单静态 tile：

| 数据         | 类型与形状     | 所在位置                              |
| ------------ | -------------- | ------------------------------------- |
| Q、K、V 输入 | `64×16xf16` | Global，经三次 Copy 搬到 Shared       |
| S、E、P      | `64×64xf32` | Tensor，表示线程持有的寄存器值        |
| M、Z         | `64xf32`     | Reduce 输出 Tensor                    |
| cast_f16(P)  | `64×64xf16` | 第二次 MMA 的寄存器 A 操作数          |
| O            | `64×16xf32` | 第二次 MMA 输出 Tensor，再写到 Global |

第一次 MMA 是 SS：Q 和 K 都来自 Shared；第二次是 RS：P 来自寄存器、V 来自 Shared。这里的 MMA 使用 Tensor Core 指令布局契约，不是 CUDA Core 标量乘加布局。

输入已经说明数学操作、shape、dtype、memory space、128 线程，以及 Shared 分配的 1024 字节对齐；但没有指定 Tensor encoding，也没有为各 `layout_view` 指定完整 Storage layout。

推断完成后增加三类信息：

1. **存储布局**：逻辑坐标对应底层内存的哪个字节、哪个 bit。
2. **分布式布局**：哪个 warp、哪个 lane 的哪个逻辑寄存器槽持有某个元素。
3. **操作契约**：Copy 选哪条路径，MMA 用哪种 atom/descriptor，Reduce 用什么作用域和结果复制策略。

推断不计算 Q/K/V 的数值。它确定执行这些运算所需的数据分布，并证明这些分布可以衔接。

## 2 两种布局通过逻辑坐标连接

用两个函数理解整个系统：

```text
D(register, lane, warp, ...) = 逻辑坐标
S(逻辑坐标)                 = (byte_offset, bit_offset)
```

`D` 对应 `DistributedEncodingAttr`，`S` 对应 `StorageLayoutAttr`。一个线程访问内存时，组合关系为 `S(D(...))`。

例如某线程的寄存器槽持有 `P[8,3]`：Tensor 布局回答“谁持有它”；写入 O 或从 Shared 读取数据时，Storage 布局再回答“对应地址在哪里”。两者不是同一种布局，因此不能用一个 SameLayout 把线程分布和字节地址简单绑在一起。

`layout_view` 是存储布局的绑定入口，不是数据搬运。真正搬运数据的是 Copy、tile_load、tile_store 等操作。本例改变 Shared 中的布局，必须让 Copy 和后面的 MMA 对同一份存储达成一致。

## 3 完整推断流水线

生产入口是 `--frisk-layout-pipeline`，其实际顺序为：

```text
克隆原始 Module，建立事务副本
  ↓
NormalizeLayoutIR：将支持的旧表示规范化
  ↓
InferLayouts
  ├─ 校验目标和输入
  ├─ 收集统一约束图
  ├─ 准备有限候选及联合合法组合
  ├─ 单调删减传播
  ├─ 有界搜索与代价选择
  ├─ 必要时受控松弛并独立重试
  ├─ 验证求解结果
  └─ 物化布局和必要转换，再验证实际 IR
  ↓
保留底层 Storage 契约
  ↓
优化布局转换
  ↓
再次执行 MLIR verifier 和实际布局 verifier
  ↓
全部成功才替换原始 Module；失败保留原 IR
```

入口实现见 [`PassPipelines.cpp`](../../lib/Dialect/Frisk/Transforms/PassPipelines.cpp)；推断阶段见 [`LayoutInfer.cpp`](../../lib/Dialect/Frisk/Transforms/LayoutInfer.cpp)。

上述每个步骤在本例中的输入、处理结果及 IR 状态，见第 14 章。

### 3.1 规范化不是先替用户决定布局

规范化把受支持的旧 Local/MemRef 表示转换为 Tensor SSA 等统一表示，清理旧布局推断入口的依赖。它不等于把所有 MemRef 都改成 Tensor：Global/Shared 仍然保留 MemRef。

本例输入本来就使用 `frisk.mma`、`frisk.reduce_tensor` 和 Tensor 算术，因此不需要通过旧 GEMM/Reduce 导入才能成立。完整 pipeline 仍经过规范化及其校验。

### 3.2 收集约束时不立刻选唯一布局

遍历操作时，系统主要记录“什么必须成立”。例如先记录 MMA 的 A/B/累加器必须组成合法指令组合，而不是看到 Q 就永久确定 Q 的 Shared layout。

因此后面的 MMA 可以影响前面的 Copy 和存储选择；Reduce、Broadcast 和第二次 MMA 的要求也可以沿关系传播。实际算法是图上的固定点与搜索，不是按源代码顺序推断一次便结束。

## 4 将 Attention 变成统一约束图

图中的节点是待确定的布局变量，不是简单地“一条 op 一个节点”。本例实际有 **38 个变量、78 条约束、1 个连通分量**。

变量包括 7 个 Storage 变量、3 个 Copy execution 变量、3 个 Copy instruction-path 变量，以及 25 个 Tensor 结果或使用位置的 Distributed 变量。

### 4.1 为什么 Tensor 结果和使用位置可以分开

设 `%exp` 同时交给 ReduceSum 和 Div。生产者只有一个真实布局，但两个消费者可能需要不同布局：

```text
%exp 的结果布局
  ├─ Convertible → ReduceSum 输入 use 的布局
  └─ Convertible → Div 输入 use 的布局
```

相同时直接使用；不同时必须证明转换合法，并将其代价算入方案。选中不同布局后，才在对应 use 前插入 `frisk.convert_layout`。不能为了照顾某个消费者，直接把生产者及所有其他消费者的布局一起覆盖。

### 4.2 本例中的主要约束

| 约束                          | 含义                                                 | Attention 中的具体位置                                              |
| ----------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------- |
| `SameLayout`                | 两个端点必须具有相同布局                             | MMA 的 accumulator 输入 use 与 result；truncf 的输入 use 与输出分布 |
| `Convertible`               | 可以同布局，也可以通过已证明合法的转换连接           | Tensor def-use，例如`%exp` 到 ReduceSum、Div                      |
| `InstructionContract`       | 多个角色必须共同组成一个合法指令方案                 | 三次 Copy、两次 MMA                                                 |
| `ReductionLayout`           | 输入贡献完整且不重复，结果遵守自然投影与复制契约     | ReduceMax、ReduceSum                                                |
| `BroadcastLayout`           | 输出坐标删除广播轴后，对应输入元素必须在同线程可取得 | 两次`[64] → [64,64]` 广播                                        |
| `StorageAccess`             | 线程分布与存储地址组合后的访问合法                   | Copy 读写和最终 tile_store                                          |
| `CopyAccess`、`Ownership` | 逻辑点正确搬运，选出的 writer 唯一                   | Global→Shared 的三次 Copy                                          |
| `ResourceLimit`             | 执行布局符合线程域等要求                             | 本例 Distributed 变量需要匹配 128 线程                              |
| `Preference`                | 可用于成本评价的软偏好，不替代合法性                 | 内存访问合并偏好                                                    |

本次图中还有必要的类型、容量与目标证明；不是只检查上表关系的名字存在。

`AliasLayout` 用于同一底层存储的多个 view，要求共同坐标的实际 bit 地址一致。本例每个根只有一个相关 view，初始图没有 AliasLayout 边。显式固定布局时会出现 `RequireEncoding`；这个无预填布局的输入也不依赖它。不要把系统支持的约束全部说成本例实际出现的约束。

## 5 准备有限候选

### 5.1 候选来自哪里

候选有三类主要来源：程序显式指定的布局、目标规则生成的合法提案、沿约束投影得到的布局。

本例中 MMA 和 Copy 的 SM90 规则提供重要种子。例如：

- 第一次 SS MMA 提供 A/B Shared descriptor 与 `64×64xf32` accumulator 的合法组合。
- 第二次 RS MMA 提供 A 寄存器布局、V Shared descriptor 和输出 accumulator 的合法组合。
- Copy path 初始可以提出 `vector`、`cp.async`、`tma`，但提出不等于通过合法性检查。
- Reduce 从输入分布投影结果布局；Broadcast 从输出候选投影输入 use 布局。
- 仍没有候选的普通变量才由 target 默认布局补齐，再继续投影。

已有硬绑定不能被传播出的替代候选冲掉；Reduce 的自然结果也不能被消费者的反向 Convertible 提案任意改写。

### 5.2 广播为什么只做有依据的反向投影

对于 `B[i,j] = M[i]`，知道输出 B 的布局，就可以删掉 `j` 坐标，推导各线程必须持有哪些 M 元素。

反过来，只知道 M 的布局，并不能唯一确定 B 的列如何分布。因此当前策略是：

```text
硬约束、邻接关系、target 默认候选提供 B 的有限候选
  → 删除广播轴
  → 压缩冗余的 register 基
  → 得到 M 输入 use 的候选
```

M 的候选会用于过滤不兼容的 B 候选，但不会无限生成“各种可能的广播逆映射”。保留 lane/warp 的复制信息十分重要，后面的例子会说明原因。

### 5.3 准备与传播是两个阶段

准备阶段允许有依据地增加候选。准备结束后，传播阶段只删除不受支持的候选，不再边求解边无限增长。

本次实测：准备过程中共有 **329 次原始候选提案**，经过验证和去重，各变量的候选条目总和为 **61**。329 包含重复提案，不是 329 个不同布局；61 也不是整图方案数。

相关实现见 [`LayoutPropagation.cpp`](../../lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp)。

## 6 用合法性证明删掉不能采用的候选

基本规则是：某变量的候选，如果在相关约束的其他端点中找不到任何合法支持，就删除它。删除后把受影响的约束重新加入 worklist，直到没有变化。

MMA/Copy 不是仅检查若干两两关系：还要检查联合指令组合。Reduce 同样有已证明合法的输入/输出配对。

本次实际传播记录：

```text
strict：61 → 52，删除 9 个候选条目
common：52 → 52，无进一步删除
```

变化落在三次 Copy 的 execution/path 变量上。最终 Copy 都是 `vector`，不是 TMA 或 cp.async。

这里有一个具体的拒绝原因：输入没有提供更强的 Global 根对齐契约，图中证据只有 1 字节。当前 TMA descriptor 规则要求 Global 源至少 16 字节对齐；cp.async 需要可证明的至少 4 字节访问宽度。本例最终只能证明 `vector_bytes=1`，因此不能因为硬件是 SM90a 就自动选择这些异步路径。Shared 的 1024 字节对齐不能代替 Global 源的对齐保证。

布局证明使用 `Proven / Disproven / Unknown` 三态。Unknown 表示当前规则或预算内无法证明，不当成合法，也不伪装为已找到数学反例。

## 7 从剩余候选中选择完整方案

### 7.1 本例为什么进入 beam search

传播后，38 个变量中有 6 个各剩 2 个候选，4 个各剩 3 个，其余 28 个各剩 1 个：

```text
候选条目总数 = 6×2 + 4×3 + 28×1 = 52
组合数       = 2⁶ × 3⁴ = 5184
```

5184 是候选域的笛卡尔积大小，不是已经证明合法的完整解数量。

默认精确枚举阈值是 256，5184 超过该阈值，因此使用宽度 32 的 beam search。每个连通分量仍受 65536 次候选扩展尝试的预算限制；这一搜索预算与其他证明中同为 65536 的枚举上限不是同一个计数器。

搜索优先选当前可行候选较少的变量，即 MRV；同等情况下结合约束连接程度和稳定顺序。扩展部分方案时检查硬约束，计算部分代价下界；每轮排序后保留有限个较优状态。完整方案才进行完整成本评价。

本例记录为：

| 指标                         | 实测值      | 含义                                   |
| ---------------------------- | ----------- | -------------------------------------- |
| search mode                  | `beam`    | 不是全组合精确枚举                     |
| beam width / peak frontier   | 32 / 32     | 每层保留状态的上限及本次峰值           |
| `mrvProbes` / `expanded` | 3034 / 3034 | 包括为选择变量而尝试的候选扩展         |
| `expandedNodes`            | 538         | 选定变量后形成的子状态累计数           |
| `beam-discarded`           | 288         | 排序后超出 beam 宽度而丢弃的状态累计数 |
| `exhaustive`               | false       | 没有穷尽所有组合                       |
| relaxation attempts          | 0           | 首次正常求解成功，无需松弛重试         |

因此结论是“在有界搜索中找到并证明了一个合法方案”，不是“证明它在所有可能布局中全局最优”。

### 7.2 代价不是九个字段随意加权

`CostVector` 按声明顺序做字典序比较：先比较第一个不同的字段，更小者优先。前面字段不同，后面字段再好也不能抵消它。

本次选中方案的总成本为：

```text
[8293, 608, 64, 0, 428, 6144, 6, 0, 0]
```

| 顺序 | 字段                    | 本次值 | 用途                                                        |
| ---- | ----------------------- | ------ | ----------------------------------------------------------- |
| 1    | instructionPathAndWork  | 8293   | 指令路径与工作量估计                                        |
| 2    | memoryTransactions      | 608    | 内存事务估计                                                |
| 3    | bankConflictDegree      | 64     | Shared bank conflict 估计                                   |
| 4    | conversionBytesAndSync  | 0      | 布局转换的搬运与同步代价                                    |
| 5    | spillRiskAndRegisters   | 428    | 寄存器槽及压力代理指标                                      |
| 6    | sharedBytesAndOccupancy | 6144   | Shared 占用等代理指标；本例三块输入各 2048 字节             |
| 7    | replication             | 6      | 复制代价的累计代理指标，不是某个 Tensor 的 replication 属性 |
| 8    | codeSize                | 0      | 代码体积相关估计                                            |
| 9    | deterministicTieBreak   | 0      | 成本向量中的末项；搜索另有稳定顺序保证确定性                |

这些是静态模型值，不是 GPU 实测周期、带宽或物理寄存器分配结果。当前模型也没有完整模拟 cache、延迟和执行重叠。

规则与实现见 [`LayoutCommon.h`](../../include/Dialect/Frisk/Analysis/LayoutCommon.h)、[`LayoutCandidateSolver.cpp`](../../lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp) 和 [`LayoutCostModel.cpp`](../../lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp)。

### 7.3 正常求解失败时怎样重试

只有符合条件的 `EmptyDomain` 才能进入受控松弛。硬绑定冲突、未知证明、无效图和搜索预算耗尽，不会通过“放宽一点”被当作成功处理。

每种松弛都从原始候选快照建立独立尝试，新增内容必须来自 target 已实现的有限规则；RequireEncoding 以及联合指令/归约端点受到保护。重试仍执行传播、搜索和验证，不降低合法性要求。机制中列出某种 fallback 名称，也不表示 target 已实现该 fallback。

本例 `relaxations=[]`，第一次正常求解就成功，没有触发这一分支；三次 Copy 选 vector 是普通候选竞争与合法性过滤的结果，不是本例发生了 TMA 松弛重试。

## 8 沿 Attention 数据流看最终布局

以下按数据流解释最终方案，便于理解；不是声称求解器按此顺序做贪心决策。

### 8.1 三次 Copy 和第一次 SS MMA

三次 Copy 的最终路径均为同步 `vector`，执行线程数 128，`writer_policy=first_owner`，`vector_bytes=1`。Copy 自身获得 execution layout，因为它没有一个 Tensor 结果可以承载“由哪些线程搬哪些元素”的信息。

第一次 MMA 的实际契约摘录：

```text
form    = ss
atom    = [64, 64, 16]
repeats = [1, 1, 1]
trans_b = true
A/B descriptor: major=k, swizzle=0, leading=1024, stride=128
```

K 的逻辑转置由 `trans_b=true` 表达，不需要先生成一个实体转置 Tensor。Q/K 的 Shared 地址映射和 descriptor 一起通过检查后，MMA 才能采用这一方案。

### 8.2 分数矩阵怎样分到 128 个线程

第一次 MMA 输出的实际 topology 为：

```text
[register, lane, warp, warp_group, cta] = [32, 32, 4, 1, 1]
replication = 1
```

也就是每线程 32 个逻辑元素槽，128 个线程合计 4096 个槽，正好覆盖 `64×64`。这里的 register 数是布局模型中的逻辑槽，不直接等于后端分配的 32-bit 物理寄存器数。

将本次 GF(2) 矩阵展开后，设寄存器槽 `r∈[0,31]`、lane `l∈[0,31]`、warp `w∈[0,3]`，实际映射为：

```text
i = floor(l / 4) + 8 × ((r >> 1) & 1) + 16 × w
j = (r & 1) + 2 × (l & 3) + 8 × floor(r / 4)
```

例如 warp 0 中：

| lane | r=0    | r=1    | r=2    | r=3    |
| ---- | ------ | ------ | ------ | ------ |
| 0    | S[0,0] | S[0,1] | S[8,0] | S[8,1] |
| 1    | S[0,2] | S[0,3] | S[8,2] | S[8,3] |
| 2    | S[0,4] | S[0,5] | S[8,4] | S[8,5] |
| 3    | S[0,6] | S[0,7] | S[8,6] | S[8,7] |

寄存器槽继续变化就覆盖剩余列；lane 4–7 对应另外两行。乘以 0.25、减法、Exp 和 Div 在最终方案中沿用这个分布。

### 8.3 ReduceMax 如何产生可广播的行最大值

沿列维 `dim=1` 归约，需要每一行的 64 个贡献值完整参与且不重复。

从上面的映射可见，同一行由同一 warp 内的一组 4 个 lane 持有；每个 lane 持有该行的 16 列。最终 ReduceMax 契约为：

```text
scope         = warp
algorithm     = canonical_fiber_tree_v1
input_policy  = first_owner
output_policy = broadcast_complete
threads       = 128
```

自然投影删除列坐标，并压缩不再影响行号的 register 基，结果为：

```text
M: tensor<64xf32>
topology    = [2, 32, 4, 1, 1]
replication = 4
i = floor(l / 4) + 8 × r' + 16 × w，r'∈[0,1]
```

每线程现在持有 2 个行最大值。例如 warp 0 的 lane 0、1、2、3 都持有 M[0] 和 M[8]。128×2=256 个持有槽，对应 64 个逻辑值，每值复制 4 份。

这不是把输入重复加进归约，而是在完成归约后让需要结果的线程都持有完整结果。`input_policy` 与 `output_policy` 处理的是不同问题。

### 8.4 广播为什么不用额外转换

第一次广播要求：

```text
B[i,j] = M[i]
```

例如 warp 0、lane 1 的槽 r=0 对应 B[0,2]，该线程已经有 M[0]；槽 r=2 对应 B[8,2]，该线程已经有 M[8]。

广播证明会检查每一个输出持有者：删去列坐标后所需的 M[i]，是否存在于同一线程的某个源寄存器槽中。源槽编号不必等于输出槽编号。

所以本例可以直接采用分数矩阵的二维布局作为广播输出布局，再与 S 做逐元素减法。**广播的兼容性来自实际坐标和同线程可用性，而不是把 `[64]` 与 `[64,64]` 当成 SameLayout。**

这里“没有转换”只表示不需要额外 `convert_layout`；Reduce 本身仍有跨 lane 归约通信，不能把零转换说成零通信。

实现见 [`BroadcastLayoutConstraints.cpp`](../../lib/Dialect/Frisk/Analysis/BroadcastLayoutConstraints.cpp)。

### 8.5 ReduceSum 与归一化沿用同样的分布关系

`math.exp` 输出仍是上述二维布局。ReduceSum 沿同一轴归约，因此得到相同 topology、replication 和 warp 作用域的 Z 布局，只是归约运算从 max 改为 sum。

第二次广播满足 `BZ[i,j]=Z[i]`，Div 的两个输入与输出能够使用同一二维分布。到这里，稳定 Softmax 整条链没有额外的 Tensor layout conversion。

### 8.6 f16 转换与第二次 RS MMA

`arith.truncf` 将归一化后的 P 从 f32 转为 f16。这是数值类型转换，不是 `convert_layout`。本例前后保留同一逻辑坐标分布，dtype 改变不会被错误解释成存储别名重解释。

这个分布也满足第二次 RS MMA 的 A operand 要求，因此不必将 P 写回 Shared 或重新分配到另一组线程。

第二次 MMA 的实际契约为：

```text
form    = rs
atom    = [64, 16, 16]
repeats = [1, 1, 4]
packing = f16x2-low-high
B descriptor: major=mn, swizzle=32, leading=256, stride=256
```

Op 的 K=64，而基本 atom 的 K=16，所以契约沿 K 重复 4 次。V 的 Shared 布局采用与 B descriptor 相符的 32B swizzle。这一选择必须同时满足前面的 Copy 和后面的 MMA，不能由两条 op 分别独立决定。

O 的 Tensor topology 为 `[8,32,4,1,1]`，replication=1，即 128×8=1024 个逻辑槽覆盖 `64×16`，随后由 tile_store 写到真正的 Global O 参数。

## 9 一个必须注意的接口边界

本例没有预先固定 Global 的物理布局，因此求解器也会为 Global Q/K/V/O 选择 Storage layout，并通过 `storage_contract` 保留下来。**本例不是“外部已经按普通连续行主序准备好数据，编译器仍能无条件直接读取”的验收。**

本次输出中，Q/K 的字节映射可化简为：

```text
offset_QK(i,j) = 16×i + 2×(j mod 8) + 1024×floor(j/8)
```

O 的字节映射为：

```text
offset_O(i,j) = 4×(64×j + i)
```

它们不同于普通 `64×16` 行主序的 `element_bytes×(16×i+j)`。V 的 Global/Shared 也带有本次选择的 swizzle 地址映射。

这些契约是调用边界需要遵守的存储前提，不是编译器已经在运行时把外部数据重排好了。若真实调用方要求固定行主序，必须在输入布局契约中明确固定，然后对那套输入重新推断和验收，不能直接套用本例的零转换或成本结论。

这也解释了为什么“完整布局推断成功”与“可直接运行一个外部框架的 Attention Tensor”是两个不同里程碑。

## 10 物化 优化与最终验证

### 10.1 把求解结果写回 IR

物化阶段将选中的布局写入：

- Tensor type 的 `#frisk.distributed` encoding。
- `layout_view` 的 `#frisk.storage` 属性及根 `storage_contract`。
- Copy 的 execution layout、path、vector width、writer policy。
- MMA 的 instruction contract、descriptor、packing/repeats。
- Reduce 的 reduction contract。

若某个 Convertible use 选中不同于生产者的布局，物化阶段插入合法转换；适用时也可以执行已选择的重计算方案。本例求解结果是 **0 条转换边、0 次重计算、0 conversion bytes**，最终 IR 中没有 `frisk.convert_layout`。

之后仍会运行转换优化阶段。不能因为 pipeline 包含转换消除、提升或共享重计算机制，就声称本例一定触发了这些优化。

### 10.2 为什么实际 IR 还要重新验证

求解图中的解合法，不代表改写 IR 时一定没有接错值、漏改类型或保留过期契约。

因此 `verifyMaterializedLayouts` 从实际 IR 重建关系，读取已经存在的 encoding 和 contract，检查存储容量、地址、线程持有关系、指令组合等。这个路径不重新生成候选或替错误结果找另一套合法布局。

最终还要经过普通 MLIR verifier，检查 SSA 支配、类型和操作自身约束。两类检查全部成功后，事务副本才提交回原始 Module。

### 10.3 失败和重放怎样处理

将 `max-expanded-states` 设置为 1 时，这个完整输入会报 `search-budget-exceeded`。对应单元测试检查失败原因，并确认原始无 encoding IR 字节不变。

正常推断结果再跑同一 pipeline，以及打印后重新解析再推断，输出保持字节一致。首次输出会增加布局，后续推断不应不断增加转换或更换契约。

相关实现见 [`MaterializeLayouts.cpp`](../../lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp)；完整集成断言见 [`AttentionLayoutIntegrationTest.cpp`](../../unittests/Dialect/Frisk/Layout/AttentionLayoutIntegrationTest.cpp)。

## 11 如何复现本文的结果

在仓库根目录运行；以下使用本次已有的 Release 工具：

```bash
# 生成最终 IR 和机器可读统计。
build-m6-release/bin/frisk-opt test/Transforms/attention-layout.mlir \
  --frisk-layout-pipeline='stats-file=/tmp/frisk-attention-summary-stats.json' \
  -verify-each -o /tmp/frisk-attention-summary.mlir

# 查看约束、传播删减、搜索方式和完整成本向量。
# 本输入已是规范 Tensor IR，可直接使用 infer 的 analysis-only 模式。
build-m6-release/bin/frisk-opt test/Transforms/attention-layout.mlir \
  --frisk-infer-layouts='analysis-only dump-analysis' \
  -o /dev/null 2> /tmp/frisk-attention-summary-analysis.txt

# 将 GF(2) 矩阵展开成布尔行列，方便检查第 8 节的坐标映射。
build-m6-release/bin/frisk-opt /tmp/frisk-attention-summary.mlir \
  --mlir-print-elementsattrs-with-hex-if-larger=-1 \
  -o /tmp/frisk-attention-summary-readable.mlir

# 重跑完整 pipeline，验证输出稳定。
build-m6-release/bin/frisk-opt /tmp/frisk-attention-summary.mlir \
  --frisk-layout-pipeline -verify-each -o /tmp/frisk-attention-summary-twice.mlir
diff /tmp/frisk-attention-summary.mlir /tmp/frisk-attention-summary-twice.mlir
```

也可以将工具路径替换为已经构建的 `build/bin/frisk-opt`。统计中的耗时随机器负载变化；本文不使用它作为性能基准。后续规则或成本模型变化后，应按相同输入重新记录候选数和选择结果。

本轮实现的全量验收记录为 417 项 unit、61 项 lit、7 项 Debug CTest、1 项 Release generator CTest、5 项 Python normalization 以及 API/tool 检查通过，详见[实现计划的后续方案 A](../layout_inference_implementation_plan.md)。这组结果是已完成实现的回归记录，不代表本文编写时重新运行了全部测试。

## 12 一句话概括新系统

Frisk 先把整个程序的存储、线程分布和指令要求表达为统一约束，在有限候选中先证明合法性、再按成本选方案，最后将方案写入并重新验证实际 IR。本例的关键衔接是：**SS MMA 的二维分布 → 带 4 份复制的行归约结果 → 同线程广播 → 保持二维分布的 Softmax → 兼容 RS MMA 的 P**，从而在当前输入契约下完成整条 Attention 的布局推断，且不需要额外 Tensor 布局转换。

## 13 Attention 推断各阶段的详细中间结果

本节记录同一输入的一次实际分析。下面的布局简称是为了阅读而起的名字，不是新增的 IR attribute。变量编号取自这次图的稳定排序；它们不代表算子的执行顺序。

快照通过当前分析 API 获取：`RelationsOnly` 收集、`InitializeForRelaxation` 候选准备、`propagateStrict`、`propagateCommonToFixedPoint`、`solveLayoutGraph`、物化、保留存储契约、转换优化。common 的 API 内部会先跑 strict，因此两个传播快照分别从同一份 prepared 图开始，未在生产路径额外插入传播步骤。分步结果通过实际布局 verifier，统一打印格式后与生产 pipeline 的最终 IR 一致。

### 13.1 各阶段结束时得到了什么

| 阶段            | 实际得到的结果                                                                               | 此时是否已经改写 Tensor 类型 |
| --------------- | -------------------------------------------------------------------------------------------- | ---------------------------- |
| 原始输入        | 7 个无 layout 的 view，16 个无 encoding 的 Tensor 结果；3 Copy、2 MMA、2 Reduce、2 Broadcast | 否                           |
| Normalize       | 与解析后的原始输入打印结果完全相同；本例已是规范 Tensor 表示                                 | 否                           |
| 仅收集关系      | 38 个变量、78 条约束；候选条目为 0                                                           | 否                           |
| 候选准备完成    | 61 个候选条目；Copy/MMA 联合支持表和 Reduce 配对已经生成                                     | 否                           |
| strict 传播完成 | 52 个条目；3 个 execution 候选和 6 个 Copy 路径候选被删除                                    | 否                           |
| common 传播完成 | 仍为 52 个条目，各变量候选集与 strict 后相同                                                 | 否                           |
| 求解完成        | 每个变量选中 1 个布局；选中 5 个指令契约、2 个归约契约；0 条转换边                           | 否，结果仍在 solution 中     |
| 物化完成        | 16 个 Tensor 结果都有 encoding；7 个 view 都有 Storage layout；操作契约落到 IR               | 是                           |
| 保留根存储契约  | 新增 7 个`frisk.storage_contract`，对应 Q/K/V/O 与三块 Shared 根                           | 是                           |
| 转换优化完成    | 没有可删除的 layout conversion；3 个常量移到 kernel 外的 Module 作用域，临时推断标记移除     | 是                           |
| 最终验证并提交  | MLIR verifier、实际布局 verifier 通过；事务提交，整条数据流保留                              | 是                           |

“16 个 Tensor 结果”包括两个 `tensor.empty` 及三个常量，不包括 MemRef 结果。`tensor.empty` 是 Broadcast 的目标占位，不是用来替代实际计算的数值输入。

### 13.2 本次出现的具体布局字典

#### Tensor 和 Copy execution 布局

以下 `r` 为线程内逻辑寄存器槽，`l` 为 lane，`w` 为 warp。`bit(r,k)=(r >> k)&1`，所有除法均取下整。

| 简称     | 逻辑 shape | topology        | replication | 坐标映射                                                   |
| -------- | ---------- | --------------- | ----------- | ---------------------------------------------------------- |
| `D_S`  | 64×64     | [32,32,4,1,1]   | 1           | i=l/4+8×bit(r,1)+16w；j=bit(r,0)+2×(l mod 4)+8×(r/4)    |
| `D_R`  | 64         | [2,32,4,1,1]    | 4           | i=l/4+8r+16w                                               |
| `D_O`  | 64×16     | [8,32,4,1,1]    | 1           | i=l/4+8×bit(r,1)+16w；j=bit(r,0)+2×(l mod 4)+8×bit(r,2) |
| `E0`   | 64×16     | [8,32,4,1,1]    | 1           | i=l/2+16w；j=r+8×(l mod 2)                                |
| `E1`   | 64×16     | [8,32,4,1,1]    | 1           | i=l/16+2r+16w；j=l mod 16                                  |
| `E2`   | 64×16     | [8,32,4,1,1]    | 1           | i=l/4+8r；j=w+4×(l mod 4)                                 |
| `Eall` | 64×16     | [1024,32,4,1,1] | 128         | i=r/16；j=r mod 16，与 lane/warp 无关                      |

`D_S` 同时用于 f32 分数/Softmax 和 f16 的 P，它描述坐标分布，dtype 由 Tensor 类型单独携带。`D_O` 与 `E0/E1/E2` 虽然 topology 相同，矩阵不同，因此不是同一布局。

例如 E1 中 warp 0、lane 0 的 8 个槽依次搬运 `(0,0),(2,0),…,(14,0)`；lane 1 搬运 `(0,1),(2,1),…,(14,1)`。而 E0 的 lane 0 持有 `(0,0)…(0,7)`。这就是三种 Copy execution 候选的实际区别。

#### Storage 布局

所有下面的地址单位都是 byte，`bit_offset=0`。定义 `xor` 为按位异或：

```text
F0(i,j)  = 16i + 2×(j mod 8) + 1024×floor(j/8)
Fv0(i,j) = 16×(i mod 8) + 256×floor(i/8)
           + 2×(j mod 8) + 128×floor(j/8)
F32(i,j) = x xor (((x >> 7) & 1) << 4)，x=32i+2j

FoRow(i,j) = 4×(16i+j)
FoCol(i,j) = 4×(64j+i)
FoXor(i,j) = x xor (((x >> 7) & 3) << 4)，x=4×(16i+j)
```

| 简称                      | 适用变量   | memory space | alignment | 映射                  |
| ------------------------- | ---------- | ------------ | --------- | --------------------- |
| `Gq0 / Gq32`            | Global Q/K | Global       | 1         | F0 / F32              |
| `Sq0 / Sq32`            | Shared Q/K | Shared       | 1024      | F0 / F32              |
| `Gv0 / Gv32`            | Global V   | Global       | 1         | Fv0 / F32             |
| `Sv0 / Sv32`            | Shared V   | Shared       | 1024      | Fv0 / F32             |
| `GoRow / GoCol / GoXor` | Global O   | Global       | 1         | FoRow / FoCol / FoXor |

Q/K/V 每块容量为 2048 字节，O 为 4096 字节；这些候选的 vector granularity 都是 1。`Gq32` 与 `Gv32` 在当前 shape 内可以化简为同一个地址函数，但其原始 affine 表达式不完全相同；表中合并的是坐标含义，不是宣称 attribute 身份相等。

### 13.3 约束收集后具体有哪些变量

下表完整列出 38 个变量。名字对应输入文件中的 SSA 名；`use` 表示消费者使用位置。第一个候选集是准备阶段结束时的真实结果，不是人为指定的初始布局。

| ID | 变量含义                          | prepared 候选         | strict 后  | common 后 | solution 选中 |
| -- | --------------------------------- | --------------------- | ---------- | --------- | ------------- |
| 0  | `%qg` Global Q view             | Gq0, Gq32             | 不变       | 不变      | Gq0           |
| 1  | `%kg` Global K view             | Gq0, Gq32             | 不变       | 不变      | Gq0           |
| 2  | `%scores` SS MMA 输出           | D_S                   | 不变       | 不变      | D_S           |
| 3  | SS MMA 的 accumulator use         | D_S                   | 不变       | 不变      | D_S           |
| 4  | `%scale` 常量 0.25              | D_S                   | 不变       | 不变      | D_S           |
| 5  | `%scaled` 乘法输出              | D_S                   | 不变       | 不变      | D_S           |
| 6  | `%max` 行最大值                 | D_R                   | 不变       | 不变      | D_R           |
| 7  | ReduceMax 输入 use                | D_S                   | 不变       | 不变      | D_S           |
| 8  | `%e0` Broadcast 目标占位        | D_S                   | 不变       | 不变      | D_S           |
| 9  | `%bm` 最大值广播输出            | D_S                   | 不变       | 不变      | D_S           |
| 10 | 第一次 Broadcast 输入 use         | D_R                   | 不变       | 不变      | D_R           |
| 11 | `%centered` 减法输出            | D_S                   | 不变       | 不变      | D_S           |
| 12 | `%exp` 指数输出                 | D_S                   | 不变       | 不变      | D_S           |
| 13 | `%sum` 行求和输出               | D_R                   | 不变       | 不变      | D_R           |
| 14 | ReduceSum 输入 use                | D_S                   | 不变       | 不变      | D_S           |
| 15 | `%e1` 第二个 Broadcast 目标占位 | D_S                   | 不变       | 不变      | D_S           |
| 16 | `%vg` Global V view             | Gv0, Gv32             | 不变       | 不变      | Gv32          |
| 17 | `%bs` 求和广播输出              | D_S                   | 不变       | 不变      | D_S           |
| 18 | 第二次 Broadcast 输入 use         | D_R                   | 不变       | 不变      | D_R           |
| 19 | `%normalized` Div 输出          | D_S                   | 不变       | 不变      | D_S           |
| 20 | `%p` truncf 输出                | D_S                   | 不变       | 不变      | D_S           |
| 21 | truncf 输入 use                   | D_S                   | 不变       | 不变      | D_S           |
| 22 | `%ozero` 第二次 MMA 初始零值    | D_O                   | 不变       | 不变      | D_O           |
| 23 | `%result` RS MMA 输出           | D_O                   | 不变       | 不变      | D_O           |
| 24 | RS MMA 的 A operand use           | D_S                   | 不变       | 不变      | D_S           |
| 25 | RS MMA 的 accumulator use         | D_O                   | 不变       | 不变      | D_O           |
| 26 | `%ov` Global O view             | GoRow, GoCol, GoXor   | 不变       | 不变      | GoCol         |
| 27 | tile_store 的 Tensor 输入 use     | D_O                   | 不变       | 不变      | D_O           |
| 28 | `%qv` Shared Q view             | Sq0, Sq32             | 不变       | 不变      | Sq0           |
| 29 | `%kv` Shared K view             | Sq0, Sq32             | 不变       | 不变      | Sq0           |
| 30 | `%vv` Shared V view             | Sv0, Sv32             | 不变       | 不变      | Sv32          |
| 31 | Q Copy execution                  | E0, E1, E2, Eall      | E0, E1, E2 | 不变      | E1            |
| 32 | Q Copy instruction path           | vector, cp.async, tma | vector     | 不变      | vector        |
| 33 | K Copy execution                  | E0, E1, E2, Eall      | E0, E1, E2 | 不变      | E1            |
| 34 | K Copy instruction path           | vector, cp.async, tma | vector     | 不变      | vector        |
| 35 | V Copy execution                  | E0, E1, E2, Eall      | E0, E1, E2 | 不变      | E1            |
| 36 | V Copy instruction path           | vector, cp.async, tma | vector     | 不变      | vector        |
| 37 | `%zero` 第一次 MMA 初始零值     | D_S                   | 不变       | 不变      | D_S           |

可以从表中看出：Tensor 部分在候选准备完成时就已经全部是单候选；本例后续搜索主要在存储布局和 Copy execution 之间作选择。**“在 prepared 阶段已唯一”不等于用户提前指定；这些布局是指令规则与坐标投影共同准备出来的。**

为方便对照图，关键关系的真实端点如下：

```text
SS MMA joint contract   [28,29,3,2]   = [Qshared,Kshared,init-use,scores]
ReduceMax               [7,6]        = [scaled-use,max]
Broadcast Max           [10,9]       = [max-use,bm]
ReduceSum               [14,13]      = [exp-use,sum]
Broadcast Sum           [18,17]      = [sum-use,bs]
RS MMA joint contract   [24,30,25,23] = [P-use,Vshared,init-use,result]
Output storage access  [27,26]       = [result-use,Oglobal]
Q Copy joint contract  [0,28,31,32]
K Copy joint contract  [1,29,33,34]
V Copy joint contract  [16,30,35,36]
```

### 13.4 候选准备还产出了哪些联合合法组合

联合支持表已经在准备阶段完成证明，但单个变量的 domain 此时还可以包含不被任何 tuple 支持的候选。strict 阶段负责把这种候选从 domain 中删除。

| 操作      | 支持表大小 | 准备阶段已证明的组合                                                 |
| --------- | ---------- | -------------------------------------------------------------------- |
| Q Copy    | 12         | {Gq0,Gq32} × {Sq0,Sq32} × {E0,E1,E2} × {vector}                   |
| K Copy    | 12         | {Gq0,Gq32} × {Sq0,Sq32} × {E0,E1,E2} × {vector}                   |
| V Copy    | 12         | {Gv0,Gv32} × {Sv0,Sv32} × {E0,E1,E2} × {vector}                   |
| SS MMA    | 4          | (Sq0,Sq0)、(Sq0,Sq32)、(Sq32,Sq0)、(Sq32,Sq32)，init/result 均为 D_S |
| RS MMA    | 2          | (D_S,Sv0,D_O,D_O)、(D_S,Sv32,D_O,D_O)                                |
| ReduceMax | 1          | D_S → D_R，max/warp/broadcast_complete                              |
| ReduceSum | 1          | D_S → D_R，sum/warp/broadcast_complete                              |

这也说明 Copy **不要求源与目标有相同 Storage layout**。例如 Gq0→Sq32 是支持表中的合法搬运；读、写地址可以不同，但必须对应同一个逻辑元素。

MMA 支持表中的 descriptor 具体参数为：

| Shared 布局 | 使用角色     | major | swizzle | leading | stride |
| ----------- | ------------ | ----- | ------- | ------- | ------ |
| Sq0         | SS 的 Q 或 K | k     | 0       | 1024    | 128    |
| Sq32        | SS 的 Q 或 K | k     | 32      | 16      | 256    |
| Sv0         | RS 的 V      | mn    | 0       | 256     | 128    |
| Sv32        | RS 的 V      | mn    | 32      | 256     | 256    |

两种 Reduce 使用相同的布局对，但 kind 不同，因此保留各自的契约，不能把两次运算合并成一次。

### 13.5 strict 与 common 的逐项结果

每次 Copy 的删除内容完全相同：

```text
execution: {E0,E1,E2,Eall} → {E0,E1,E2}  删除 1 个
path:      {vector,cp.async,tma} → {vector} 删除 2 个
三次 Copy 合计：3×(1+2)=9
```

删除理由分别是：

- `Eall` 的 hardware domain 是 `1024×32×4=131072` 个持有者，超过当前 execution proof 的 65536 点预算，不能构成已证明的 Copy tuple。这里是 **Unknown / 预算内不可证明**，不是已经证明这种复制布局数学上错误。
- `cp.async` 没有满足当前源对齐、访问宽度要求的支持组合。
- `tma` 没有满足 Global 源对齐等 descriptor 要求的支持组合。

strict 的实际记录为 `deleted=9、domainChanges=6、queuePops=74`；改变的是 6 个变量，不是 9 个变量。common 为 `deleted=0、domainChanges=0、queuePops=71`。

Q/K/V 的 Storage 候选没有在这个阶段删成唯一方案；这一步只排除无合法支持的选项，还没有做最后的成本取舍。

### 13.6 求解结束时的完整选择

真正选中的有分歧部分为：

```text
Global:  Q=Gq0, K=Gq0, V=Gv32, O=GoCol
Shared:  Q=Sq0, K=Sq0, V=Sv32
Copy:    三次都是 execution=E1, path=vector, vector_bytes=1
SS MMA:  (Sq0,Sq0,D_S,D_S)，atom=[64,64,16]，repeats=[1,1,1]
RS MMA:  (D_S,Sv32,D_O,D_O)，atom=[64,16,16]，repeats=[1,1,4]
Reduce:  Max 和 Sum 都是 D_S→D_R，scope=warp
Convert: 0
Rematerialize: 0
Relaxation retry: 0
```

剩余单候选变量的选择与 13.3 表一致。搜索记录为 5184 个域组合、beam width 32、3034 次 MRV probe、538 个子状态扩展、288 个 beam 丢弃状态，最终完整成本 `[8293,608,64,0,428,6144,6,0,0]`。

#### 用实际成本复算解释具体取舍

下面不是逐轮 beam 日志，而是**固定最终方案的其他变量，只替换表中一个变量**后，重新验证完整求解图、重新生成相关指令 binding，并按相同成本函数复算的对照。所有列出的对照方案均通过完整图 verifier。

前三项分别为指令工作、内存事务、bank conflict；未列出的六项在这些对照中均为 `[0,428,6144,6,0,0]`。

| 单项改动                       | 前三项成本      | 与实际选中方案比较                 |
| ------------------------------ | --------------- | ---------------------------------- |
| 不改动，实际方案               | [8293,608,64]   | 基准                               |
| O 从 GoCol 改为 GoRow          | [8293,864,64]   | 内存事务更高                       |
| O 从 GoCol 改为 GoXor          | [8293,848,64]   | 内存事务更高                       |
| Q Copy execution 从 E1 改为 E0 | [8293,1056,128] | 内存事务更高，bank conflict 也更高 |
| Q Copy execution 从 E1 改为 E2 | [8293,800,64]   | 内存事务更高                       |
| V Shared 从 Sv32 改为 Sv0      | [8293,608,96]   | 前两项相同，bank conflict 更高     |
| V Global 从 Gv32 改为 Gv0      | [8293,640,64]   | 内存事务更高                       |
| Q Shared 从 Sq0 改为 Sq32      | [8293,608,32]   | **比实际方案更低成本**       |
| Q Global 从 Gq0 改为 Gq32      | [8293,576,64]   | **比实际方案更低成本**       |

前几行可以直观看到 GoCol、E1、V 的 swizzle 选择在当前组合中有什么成本意义。最后两行则明确暴露了一个边界：**这次宽度 32 的 beam search 不仅没有全局最优保证，实际也存在已经通过完整图证明、成本更低的候选组合没有被最终选中。**

因此不能把“最终选中 Sq0/Gq0”解释为“它们必然比 Sq32/Gq32 便宜”。完整成本到完整状态才充分评价，有限 beam 会提前裁掉部分路径；本次没有保存每一层的完整 frontier，不能指定更优组合究竟在哪一层被舍弃。该观察属于搜索质量边界，不影响已输出方案的布局合法性。本轮只记录结果，不修改搜索策略或预算。

### 13.7 物化后每条计算实际得到的类型

下面按原始程序数据流列出，不按变量 ID 排序。记 `T<shape,dtype,D>` 为带该 Distributed encoding 的 Tensor，`M<shape,dtype,S>` 为绑定 Storage layout 的 MemRef view。

| 步骤 | 操作                        | 该步输出或写入结果                                                | 线程分布与额外动作                                 |
| ---- | --------------------------- | ----------------------------------------------------------------- | -------------------------------------------------- |
| 1    | 三块 Shared alloc 与 view   | Qshared=M<64×16,f16,Sq0>；Kshared 同；Vshared=M<64×16,f16,Sv32> | 每块 2048B，alignment=1024                         |
| 2    | Copy Q                      | 按 Sq0 写入 Qshared                                               | E1、128 线程、vector、first_owner                  |
| 3    | Copy K                      | 按 Sq0 写入 Kshared                                               | 同上                                               |
| 4    | Copy V                      | 按 Sv32 写入 Vshared                                              | 同上；不是 TMA                                     |
| 5    | 初始零值`%zero`           | T<64×64,f32,D_S>，全部逻辑值为 0                                 | 每线程 32 槽，无复制                               |
| 6    | SS MMA QKᵀ                 | `%scores`: T<64×64,f32,D_S>                                    | SS atom 64×64×16，trans_b=true                   |
| 7    | 常量`%scale`              | T<64×64,f32,D_S>，全部逻辑值为 0.25                              | 与 scores 布局一致                                 |
| 8    | `%scaled = scores×scale` | T<64×64,f32,D_S>                                                 | 无 layout conversion                               |
| 9    | ReduceMax                   | `%max`: T<64,f32,D_R>                                           | 每线程 2 槽，每逻辑行值复制 4 份                   |
| 10   | `%e0 = tensor.empty`      | T<64×64,f32,D_S>                                                 | Broadcast 的输出占位，不承诺已有数值               |
| 11   | Broadcast Max               | `%bm`: T<64×64,f32,D_S>，逻辑关系 bm[i,j]=max[i]               | 从同线程 D_R 源槽取得数据                          |
| 12   | Sub                         | `%centered`: T<64×64,f32,D_S>                                  | centered[i,j]=scaled[i,j]−max[i]                  |
| 13   | Exp                         | `%exp`: T<64×64,f32,D_S>                                       | 分布不变                                           |
| 14   | ReduceSum                   | `%sum`: T<64,f32,D_R>                                           | warp 归约，输出复制 4 份                           |
| 15   | `%e1 = tensor.empty`      | T<64×64,f32,D_S>                                                 | 第二次 Broadcast 的输出占位                        |
| 16   | Broadcast Sum               | `%bs`: T<64×64,f32,D_S>，bs[i,j]=sum[i]                        | 无 layout conversion                               |
| 17   | Div                         | `%normalized`: T<64×64,f32,D_S>                                | normalized[i,j]=exp[i,j]/sum[i]                    |
| 18   | truncf                      | `%p`: T<64×64,f16,D_S>                                         | dtype 改变，坐标分布不变                           |
| 19   | `%ozero`                  | T<64×16,f32,D_O>，全部逻辑值为 0                                 | 每线程 8 槽，无复制                                |
| 20   | RS MMA PV                   | `%result`: T<64×16,f32,D_O>                                    | P 的 use 仍为 D_S；V=Sv32；K 方向 4 次 atom repeat |
| 21   | tile_store                  | 按 GoCol 写入 Global O                                            | 输入 use 为 D_O，不先转另一种 Tensor layout        |

这里记录的是类型、布局、契约及逻辑运算关系，不是 Q/K/V 的具体浮点数值。推断不运行 Attention，因此不会产生一份实际的 scores/max/exp 数值矩阵。

#### 同一个线程在各步持有什么

以 warp 0、lane 1 为例，二维 D_S 的前四个槽与归约 D_R 的两个槽关系为：

```text
SS MMA / scale:
  r0: scaled[0,2]    r1: scaled[0,3]
  r2: scaled[8,2]    r3: scaled[8,3]

ReduceMax 后 D_R:
  r'0: max[0]       r'1: max[8]

Broadcast Max 后 D_S:
  r0: max[0]        r1: max[0]
  r2: max[8]        r3: max[8]

Sub / Exp 后 D_S:
  r0: exp(scaled[0,2]-max[0])
  r1: exp(scaled[0,3]-max[0])
  r2: exp(scaled[8,2]-max[8])
  r3: exp(scaled[8,3]-max[8])

ReduceSum 后 D_R:
  r'0: sum[0]       r'1: sum[8]

Broadcast Sum 后 D_S:
  r0: sum[0]        r1: sum[0]
  r2: sum[8]        r3: sum[8]

Div / truncf 后 D_S:
  r0: P_f16[0,2]   r1: P_f16[0,3]
  r2: P_f16[8,2]   r3: P_f16[8,3]

RS MMA 后 D_O:
  r0: O[0,2]       r1: O[0,3]
  r2: O[8,2]       r3: O[8,3]
```

表里只列前四槽帮助追踪坐标，不表示每个线程只有四个值；D_S 总计 32 槽，D_O 总计 8 槽。两次 MMA 是协同矩阵运算，不能把最后四行误解成只拿该线程同编号的 P 做独立标量乘法。

### 13.8 保留契约与优化之后发生了什么

单独 materialize 完成时，7 个 view 已绑定 Storage layout，但 `storage_contract` 数量还是 0。后续 preserve 阶段才为 Q、K、V、O 和三块 Shared 分配增加 7 个根契约；这是两个不同阶段。

转换优化前后，Copy/MMA/Reduce/Broadcast/Store 数量不变，layout conversion 始终为 0；物化产生的 16 个 `frisk.inferred_result_layout` 临时标记全部移除。优化后的可见变化是常量提升和临时标记清理：

```text
Module:
  zero_O : T<64×16,f32,D_O>
  scale  : T<64×64,f32,D_S>
  zero_S : T<64×64,f32,D_S>
  kernel attention:
    Shared allocations
    parallel 128 threads:
      Global / Shared root storage contracts 与 layout_views
      Copy Q, Copy K, Copy V
      SS MMA → Mul → ReduceMax → Broadcast → Sub → Exp
             → ReduceSum → Broadcast → Div → truncf
             → RS MMA → Global tile_store
```

上面的结构省略了 attribute 长文本，但每个布局的坐标映射、候选集和最终取值都已在本节给出。最终 IR 经实际布局验证，并与正常生产 pipeline 输出在统一打印格式后相同。

### 13.9 快照证据与复现边界

本次临时证据为：

- `/tmp/frisk-attention-stages-trace.json`：prepared/strict/common 各变量的完整 attribute、联合支持表、selected assignment、分阶段 IR 和事后对照成本。
- `/tmp/frisk-attention-stages-stats.json`：正常生产 pipeline 的统计。
- `/tmp/frisk-attention-stages-final.mlir`：正常生产 pipeline 的最终结果。
- `/tmp/frisk-attention-trace.cpp`：只调用现有分析 API 的临时诊断程序，不属于生产实现。

长期阅读以本节的完整候选表和结果为准；临时文件不是仓库必须依赖的输入。第 11 节命令可重新导出生产结果和总统计，但当前 CLI 的 `dump-analysis` **不会**逐项打印完整候选 attribute，也不会记录每一轮 beam 的全部 frontier。本节没有将事后候选成本复算冒充成搜索过程的逐轮历史。

## 14 沿完整流水线追踪 Attention 的中间结果

本章严格对应第 3 章的流水线。这里的“每一步”是**编译器处理阶段**，不是在 GPU 上依次计算 QKᵀ、Softmax、PV。相同的 Attention 程序将依次表现为：无布局 IR、约束图、候选域、已选方案、带布局 IR、可提交的最终 IR。

阅读时区分三个对象：

- **原始 Module**：调用者交给 pipeline 的程序，提交前不改动。
- **工作副本 IR**：规范化、物化和优化实际修改的程序。
- **分析数据**：约束图、候选集合和求解结果；它们不等于已经写进 Tensor 类型的 encoding。

下文沿用第 13.2 节的布局简称。`D_S` 是 `64×64` 的分布，每线程 32 槽、每元素一份；`D_R` 是归约后 `64` 的分布，每线程 2 槽、每元素复制 4 份；`D_O` 是 `64×16` 的分布，每线程 8 槽、每元素一份。三者均使用 128 线程。`G*` 表示 Global 存储布局，`S*` 表示 Shared 存储布局，`E*` 表示 Copy 的线程执行分布。

### 14.1 克隆原始 Module：先建立可丢弃的工作副本

输入包含三个 Global→Shared Copy，以及下面这条尚未编码的 Tensor 链：

```text
scores[64,64]     = SS_MMA(Qshared, Kshared, zero)
scaled[64,64]     = scores * 0.25
max[64]          = ReduceMax(scaled, dim=1)
centered[64,64]   = scaled - Broadcast(max)
exp[64,64]        = Exp(centered)
sum[64]          = ReduceSum(exp, dim=1)
normalized[64,64] = exp / Broadcast(sum)
p[64,64]         = cast_f16(normalized)
result[64,16]    = RS_MMA(p, Vshared, ozero)
Store(result, Oglobal)
```

入口执行 `original->clone()`。此时的具体状态是：

| 对象             | 克隆后的状态                  |
| ---------------- | ----------------------------- |
| 原始 Module      | 保持输入状态                  |
| 工作副本         | 与原始 Module 内容相同        |
| Tensor 结果      | 16 个，均无 encoding          |
| `layout_view`  | 7 个，均未绑定 Storage layout |
| 根 Storage 契约  | 0 个                          |
| 约束图、候选和解 | 尚未建立                      |

例如 `%scores` 仍是 `tensor<64x64xf32>`，而不是 `tensor<64x64xf32, D_S>`。副本只是保证后续失败可以不影响原程序，并没有产生布局决策。

### 14.2 NormalizeLayoutIR：本例规范化后内容不变

此步骤处理支持范围内的旧表示，使后续分析面对统一的 Tensor/Storage 表示。本例已经使用 Tensor 形式的 MMA、Reduce、Broadcast 和算术运算，Global/Shared 则保留 MemRef，因此没有需要改写的旧算子。

分阶段快照的结果是 `normalizedSameAsInput = true`：解析后以相同格式打印，规范化前后的 IR 相同。这不要求源码注释、空格或原始 SSA 名称逐字保留。

关键中间结果仍为：

```text
%scores : tensor<64x64xf32>       encoding = 未指定
%max    : tensor<64xf32>          encoding = 未指定
%p      : tensor<64x64xf16>       encoding = 未指定
%result : tensor<64x16xf32>       encoding = 未指定
%qv、%kv、%vv                    Storage layout = 未指定
```

**交给下一步的是规范化后的无布局 IR，不是已经选好 MMA 布局的 IR。**

### 14.3 校验目标和输入：确认可以进入当前推断体系

`InferLayouts` 首先检查没有遗留的不受支持旧布局表示，并检查目标边界。本例通过检查，使用 SM90 target 模型继续分析。

后续分析可以从 IR 取得以下事实：

| 已知事实          | 本例取值                          |
| ----------------- | --------------------------------- |
| 目标              | `sm_90a`                        |
| parallel 线程数   | 128，即 4 个 warp                 |
| Q/K/V             | `64×16xf16`，Global 输入       |
| Shared Q/K/V      | 各 2048 字节，显式 alignment=1024 |
| Global 根对齐证据 | 没有更强保证，按 1 字节处理       |
| 两次 MMA          | SS：64×64×16；RS：64×16×64    |

这里的“通过”只意味着输入和目标在入口支持范围内，**不是已经证明所有 Copy/MMA 布局组合合法**。此时还没有选定 Copy 路径，工作副本 IR 也没有变化。

### 14.4 收集统一约束图：把程序改写成“待满足的关系”

这一阶段的产物不是另一份 MLIR，而是一张含 **38 个变量、78 条约束**的分析图，其中 71 条是硬约束，7 条是软偏好。

| 变量类别       | 数量 | 具体对象                                 |
| -------------- | ---- | ---------------------------------------- |
| Storage        | 7    | Q/K/V Global、Q/K/V Shared、O Global     |
| Copy execution | 3    | 三次 Copy 各自的线程执行布局             |
| Copy path      | 3    | 三次 Copy 各自的指令路径                 |
| Distributed    | 25   | Tensor 结果布局及独立建模的输入 use 布局 |

以几个真实子关系说明此时记录了什么：

```text
Copy Q:
  (Qglobal, Qshared, Q执行布局, Q路径) 必须组成合法 Copy 契约

SS MMA:
  (Qshared, Kshared, accumulator-use, scores) 必须组成合法 SS 契约
  accumulator-use 与 scores 必须 SameLayout

ReduceMax / Broadcast:
  scaled-result --Convertible--> ReduceMax-input-use
  ReduceMax-input-use --ReductionLayout--> max-result
  max-result --Convertible--> Broadcast-input-use
  Broadcast-input-use --BroadcastLayout--> bm-result

Store:
  result --Convertible--> Store-input-use
  (Store-input-use, Oglobal) 必须满足 StorageAccess
```

这时知道“Q 的 Copy 和第一条 MMA 共用同一个 Qshared 布局变量”，但还不知道应该取 `Sq0` 还是 `Sq32`。所以后面的 MMA 能约束前面的 Copy，不需要依靠源代码遍历顺序猜测。

为观察这个边界，诊断程序使用 `RelationsOnly` 收集了候选总数为 **0** 的纯关系图。生产入口使用 `InitializeForRelaxation`，在收集流程中继续完成候选准备；**纯关系快照不是生产流水线额外执行的一次 pass**。工作副本 IR 此时仍未编码。

### 14.5 准备有限候选：关系图第一次有了具体可选值

目标规则提出 MMA/Copy 等种子，关系投影传播这些候选，必要时再由目标默认规则补齐。准备结束得到 **61 个候选条目**；过程中统计到的 329 次原始提案包含重复项。

最关键的候选域如下。完整 38 变量逐项表见第 13.3 节。

| 对象                                                    | 准备结束后的候选                  |
| ------------------------------------------------------- | --------------------------------- |
| Q、K Global                                             | 各`{Gq0, Gq32}`                 |
| Q、K Shared                                             | 各`{Sq0, Sq32}`                 |
| V Global / Shared                                       | `{Gv0, Gv32}` / `{Sv0, Sv32}` |
| O Global                                                | `{GoRow, GoCol, GoXor}`         |
| 每次 Copy execution                                     | `{E0, E1, E2, Eall}`            |
| 每次 Copy path                                          | `{vector, cp.async, tma}`       |
| scores、scaled、bm、centered、exp、bs、normalized、p 等 | `{D_S}`                         |
| max、sum 及 Broadcast 的源 use                          | `{D_R}`                         |
| ozero、result、Store 的输入 use                         | `{D_O}`                         |

这里已经能看见 Reduce/Broadcast 对接的结果：`64×64` 分布采用 `D_S`，对应的行归约分布采用 `D_R`。但这些只是图中变量的候选，**`%max` 的实际类型仍然是没有 encoding 的 `tensor<64xf32>`**。

准备还产生联合支持表，不是任意候选都能随意拼接：

| 关系                 | 已准备的合法支持组合                                                    |
| -------------------- | ----------------------------------------------------------------------- |
| SS MMA               | 4 个：A/B 各选 no-swizzle 或 swizzle32，acc/result 均为`D_S`          |
| RS MMA               | 2 个：A 为`D_S`，B 为 `Sv0` 或 `Sv32`，acc/result 为 `D_O`      |
| 每次 Copy            | 12 个：2 个 Global × 2 个 Shared × 3 个 execution × 1 个 vector 路径 |
| ReduceMax、ReduceSum | 各 1 个输入/输出配对：`D_S → D_R`                                    |

注意 Copy 的变量域里暂时还有 `Eall`、`cp.async`、`tma`，但合法支持表没有它们。**“提过这个候选”和“已证明能组成合法操作”是两件事**；下一步会把这种没有支持的候选删掉。

### 14.6 单调删减传播：61 个条目缩小到 52 个

候选准备到此结束。传播开始后不再增加候选，只根据约束删除没有支持的条目，直到固定点。

三次 Copy 发生完全相同的变化：

```text
execution: {E0, E1, E2, Eall} → {E0, E1, E2}
path:      {vector, cp.async, tma} → {vector}
```

每次删除 3 个，共删除 9 个。`Eall` 的硬件槽位域为 `1024×32×4 = 131072`，当前相关证明枚举上限为 65536，未取得所需证明；不能将这种未证明状态当作合法支持。异步路径则缺少足够的 Global 对齐/访问宽度证据：本例只能证明 `vector_bytes=1`，Shared 的 1024 字节对齐不能替代 Global 保证。

其余候选域不变。传播记录是：

| 子阶段 | 输入条目 | 输出条目 | 删除条目 | domain changes | worklist pops |
| ------ | -------- | -------- | -------- | -------------- | ------------- |
| strict | 61       | 52       | 9        | 6              | 74            |
| common | 52       | 52       | 0        | 0              | 71            |

这里把 strict 和 common 的观察边界拆开展示；生产的 common 固定点入口包含 strict 处理，不是为了生成文档又执行一套生产传播。

传播结束还不能说“全部布局已确定”。Q/K/V 存储、O 存储以及 Copy execution 仍有选择空间：

```text
6 个变量各剩 2 个候选
4 个变量各剩 3 个候选
28 个变量各剩 1 个候选
候选条目总数 = 52
候选域组合数 = 2^6 × 3^4 = 5184
```

5184 是待考虑域的笛卡尔积，不代表 5184 个组合均已通过整图证明。此时工作副本 IR 仍然未改变。

### 14.7 有界搜索：从剩余候选中选出一套完整方案

单个连通分量的组合数 5184 超过默认精确枚举阈值 256，因此本例进入宽度 32 的 beam search。搜索统计为：候选探测 `mrvProbes=3034`、实际扩展节点 `expandedNodes=538`、frontier 峰值 32，且 `exhaustive=false`。

搜索的核心产物是**每个变量的唯一取值，加上与这些取值匹配的操作契约**：

| 决策对象          | 实际选中结果                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------- |
| Q/K Global        | `Gq0`                                                                                            |
| Q/K Shared        | `Sq0`                                                                                            |
| V Global / Shared | `Gv32` / `Sv32`                                                                                |
| O Global          | `GoCol`                                                                                          |
| 三次 Copy         | 均为`E1 + vector`，`vector_bytes=1`，同步、`first_owner`                                     |
| 第一条 MMA        | SS，atom`[64,64,16]`，repeats `[1,1,1]`，A/B descriptor 均 swizzle=0                           |
| 两次 Reduce       | `D_S → D_R`，warp 作用域，`canonical_fiber_tree_v1`，结果 `broadcast_complete`              |
| 第二条 MMA        | RS，atom`[64,16,16]`，repeats `[1,1,4]`，B descriptor swizzle=32，A packing=`f16x2-low-high` |
| Tensor 布局       | Softmax 主链`D_S`，两个归约结果 `D_R`，输出 `D_O`                                            |
| def-use 衔接      | 无需额外 layout conversion，也无需重物化                                                           |

为了让存储选择具体可见，选中布局的字节地址关系为：

```text
Q/K 的 Global 与 Shared：byte(i,j) = 16*i + 2*(j%8) + 1024*(j/8 的整数商)
V 的 Global 与 Shared：  x = 32*i + 2*j
                        byte(i,j) = x XOR (((x >> 7) & 1) << 4)
O 的 Global：           byte(i,j) = 4*(64*j + i)
```

这些布局的 bit offset 都是 0。Q/K/V 的 Global 和 Shared 地址函数对应相同逻辑排布，但 memory space 和对齐契约不同，仍需要真实 Copy 搬运。

总代价向量为：

```text
[8293, 608, 64, 0, 428, 6144, 6, 0, 0]
   │    │   │  │    │     │  │  │  └─ deterministicTieBreak
   │    │   │  │    │     │  │  └──── codeSize
   │    │   │  │    │     │  └─────── replication
   │    │   │  │    │     └────────── sharedBytesAndOccupancy
   │    │   │  │    └──────────────── spillRiskAndRegisters
   │    │   │  └───────────────────── conversionBytesAndSync
   │    │   └──────────────────────── bankConflictDegree
   │    └──────────────────────────── memoryTransactions
   └───────────────────────────────── instructionPathAndWork
```

这是目标模型的估算向量，不是 GPU 实测时间，也不是 8293 条实际机器指令的承诺。比较使用字典序，不是将九项简单相加。

**这一结果是本次搜索返回的合法方案，不能写成全局最优方案。**第 13.6 节的事后复算已找到代价更低的合法替代，例如仅把 Q Global 改为 `Gq32` 可将第二项从 608 降为 576。这说明有限宽度搜索确实可能漏解；现有快照没有保存逐轮 frontier，不能杜撰该替代方案在哪一轮被丢弃。

到此仍然只有分析数据中的选中方案，Tensor 类型和 view 属性尚未改写。

### 14.8 受控松弛：本例不进入重试分支

第 3 章将“必要时松弛”画在搜索之后，表示条件分支，而不是每次必须再运行一个步骤。实际由 `attemptControlledRelaxation` 协调初始传播、求解、验证和必要的独立重试。

本例的实测记录是：

```text
attempts = [ { kind: "initial", status: "solved" } ]
relaxations = []
```

即第一次尝试已经有合法解，没有第二轮候选域，没有放宽任何关系，也没有修改用户给出的硬契约。上一节的方案直接交给后续验证。

若出现允许处理的空域，系统才会从原始分析快照建立相应重试；不是直接在已经删空的域上随意加候选。本例未执行这条路径，因而没有可列出的“松弛后中间结果”。

### 14.9 验证求解结果：检查“选中的组合”而不是只看每个候选

`verifySolvedLayoutGraph` 检查最终赋值与联合绑定是否共同满足图中的硬关系。本例通过，推断报告在这一阶段可进入 `solved` 状态。

几个具体检查结果是：

| 被验证的连接                 | 本例选中值及结果                                                              |
| ---------------------------- | ----------------------------------------------------------------------------- |
| Copy Q → SS MMA             | 两者引用同一个`Sq0`；Copy 写入排布与 MMA A descriptor 一致                  |
| Copy V → RS MMA             | 两者引用同一个`Sv32`；Copy 写入排布与 MMA B descriptor 一致                 |
| SS accumulator-use → result | `D_S = D_S`，满足 SameLayout                                                |
| ReduceMax → Broadcast       | 归约结果与 Broadcast 源 use 均为`D_R`，广播输出为 `D_S`，同线程取值可证明 |
| ReduceSum → Broadcast       | 同样满足`D_R → D_S` 的广播关系                                             |
| RS accumulator-use → result | `D_O = D_O`，满足 SameLayout                                                |
| RS result → Store           | 输入 use 采用`D_O`，与 `GoCol` 组合满足存储访问要求                       |

例如 warp 0、lane 1 在 `D_R` 下持有行 0 和行 8 的归约结果；同线程 `D_S` 的前四槽对应 `(0,2)、(0,3)、(8,2)、(8,3)`。广播分别读取该线程已有的行 0、行 0、行 8、行 8 的值，因此不需要额外跨线程转换。

**这一验证通过，证明的是分析方案可行；还没有证明后面的 IR 改写实现一定正确。**所以物化后必须再次针对实际 IR 检查，而不能只复用这里的成功结论。

### 14.10 物化及实际 IR 验证：第一次把选中布局写入程序

`materializeLayouts` 根据完整方案修改工作副本。此时出现本流水线最重要的 IR 状态变化：

| 项目                                      | 物化前                | 物化后                                       |
| ----------------------------------------- | --------------------- | -------------------------------------------- |
| 16 个 Tensor 结果                         | 没有 encoding         | 全部具有选中的 Distributed encoding          |
| 7 个 layout_view                          | 没有 Storage binding  | 全部绑定选中 Storage layout                  |
| Copy/MMA/Reduce                           | 输入中的数学/形状信息 | 补入 execution、instruction/reduction 等契约 |
| `frisk.convert_layout`                  | 0 个                  | 0 个，本例不需要插入                         |
| `frisk.inferred_result_layout` 临时标记 | 0 个                  | 16 个，供 pipeline 内部优化使用              |
| `frisk.storage_contract` 根契约         | 0 个                  | 仍为 0 个，下一阶段才保留                    |

下面列出物化后所有 16 个 Tensor 结果的类型。`D_*` 是本文简称，不是可直接解析的完整 attribute 文本。

| 结果                                            | 物化后的类型               |
| ----------------------------------------------- | -------------------------- |
| `%zero`、`%scores`、`%scale`、`%scaled` | `tensor<64x64xf32, D_S>` |
| `%max`                                        | `tensor<64xf32, D_R>`    |
| `%e0`、`%bm`、`%centered`、`%exp`       | `tensor<64x64xf32, D_S>` |
| `%sum`                                        | `tensor<64xf32, D_R>`    |
| `%e1`、`%bs`、`%normalized`               | `tensor<64x64xf32, D_S>` |
| `%p`                                          | `tensor<64x64xf16, D_S>` |
| `%ozero`、`%result`                         | `tensor<64x16xf32, D_O>` |

两处具体变化可以简写为：

```text
物化前：
  qv = layout_view(qs)                 layout 未指定
  max = reduce_tensor(scaled, dim=1)   tensor<64xf32>

物化后：
  qv = layout_view(qs, layout=Sq0)
  max = reduce_tensor(scaled:D_S, dim=1,
                      execution_threads=128,
                      contract=warp / canonical_fiber_tree_v1 / broadcast_complete)
        : tensor<64xf32, D_R>
```

随后 `verifyMaterializedLayouts` 从**这份实际 IR** 重新收集关系，固定已经写入的布局和契约，检查它们是否真的匹配。它不是重新生成一批候选，替错误的物化结果另找一个可行解。本例该验证通过，报告状态进入 `materialized`。

此时仍未生成 GPU 机器码，也没有算出 scores/max/exp 的浮点数值；只是它们的执行布局已成为程序的一部分。原始 Module 仍未替换。

### 14.11 保留底层 Storage 契约：从 view 绑定落实到存储根

物化只保证当前 view 携带正确布局。接下来 `preserveStorageContracts` 将布局要求保留在对应根上，避免后续 view 简化等改写丢失底层存储约定。

本例新增 7 个根契约：

| 存储根               | 保留的布局契约 |
| -------------------- | -------------- |
| Global Q、K          | 各`Gq0`      |
| Global V             | `Gv32`       |
| Global O             | `GoCol`      |
| Shared Q、K 的 alloc | 各`Sq0`      |
| Shared V 的 alloc    | `Sv32`       |

例如 `qv` 已经要求 `Sq0`；保留步骤再让其底层 `%qs` 的存储契约明确同一排布。本步之后：7 个 view 仍绑定，根契约从 **0 → 7**，16 个临时推断标记仍在，Tensor encoding 不变。

特别注意 Global O：`GoCol` 意味着 `O[i,j]` 位于 `4*(64*j+i)` 字节处。这里保留的是对调用者/后续 lowering 的真实存储约定，**不是把一个原本固定 row-major 的外部缓冲区免费转成 column-major**。本例输入未预先固定该 Storage 契约；如应用要求固定 ABI，应在输入中明确约束。

### 14.12 优化布局转换：本例清理标记和提升常量，不改变选择

进入 `OptimizeLayoutConversions` 时已经没有 layout conversion，因此不存在“把多个转换合成一个”的实际案例。本例观察到的变化为：

- 16 个 `frisk.inferred_result_layout` 临时标记全部清除，Tensor 类型中的 encoding 保留。
- 三个常量提升到 Module 层：`zero_O:D_O`、`scale:D_S`、`zero_S:D_S`。
- 7 个根 Storage 契约保留，Copy/MMA/Reduce 的选中契约保留。

优化前后都有 3 个 Copy、2 个 MMA、2 个 Reduce、2 个 Broadcast、1 个 tile_store，layout conversion 数量都是 0。

优化后的结构简写如下；数学数据流未变：

```text
Module:
  zero_O : tensor<64x16xf32, D_O>
  scale  : tensor<64x64xf32, D_S>
  zero_S : tensor<64x64xf32, D_S>
  attention:
    Shared allocations
    parallel(128 threads):
      7 个根 Storage 契约与 7 个带布局的 view
      Copy Q/K/V：E1 + vector
      SS MMA → D_S 的 Softmax 主链（两次 D_R 归约及广播）
             → truncf:D_S → RS MMA:D_O → Store:GoCol
```

这一步没有重新搜索另一套更低代价的存储布局，不能把前面 beam search 的有限性理解为会由转换优化自动补救。

### 14.13 最终验证与事务提交：通过后才替换原始 Module

pipeline 对优化后的工作副本再次运行两层验证：MLIR verifier 检查类型、SSA 等 IR 合法性，实际布局 verifier 检查最终 encoding、Storage 绑定和操作契约的一致性。

本例通过最终验证，阶段快照经统一打印格式后与正常生产 pipeline 的最终 IR 相同。正常生产统计的关键结果是：

```text
status = committed
attempts = 1，initial solved
relaxations = []
variables = 38，constraints = 78，components = 1
preparedCandidates = 61，prunedCandidates = 9
conversionUseEdges = 0，uniqueConversions = 0，conversionBytes = 0
uniqueRematerializations = 0
```

如果要求输出统计文件，还必须成功写出该文件，才执行原始 Module 的属性和 body 替换。任一步失败都不提交工作副本；调用者的原始 IR 保持输入状态。这里的 `committed` 是 **IR 事务提交**，与 Git commit/push 无关。

因此，在全部成功之前，外部仍持有原始无布局 Module；全部成功之后，外部一次性得到第 14.12 节所描述的完整带布局 Module。

### 14.14 汇总：每个阶段究竟多出了什么

| 流水线边界         | 分析数据的中间结果                         | 工作副本 IR 的中间结果                                      |
| ------------------ | ------------------------------------------ | ----------------------------------------------------------- |
| 克隆               | 尚无图                                     | 输入副本：16 个未编码 Tensor 结果、7 个未绑定 view          |
| 规范化             | 尚无图                                     | 本例内容不变                                                |
| 入口校验           | 目标与输入边界检查通过                     | 不变                                                        |
| 收集关系           | 38 变量、78 约束；纯关系观察快照无候选     | 不变                                                        |
| 准备候选           | 61 个候选条目及联合合法支持表              | 不变                                                        |
| strict/common 传播 | 删掉 9 个，剩 52 个；候选域组合数 5184     | 不变                                                        |
| 搜索               | 每个变量选定一个值；绑定操作契约和完整代价 | 不变                                                        |
| 松弛分支           | 本例未触发，仍为 initial 解                | 不变                                                        |
| 求解结果验证       | 所选整图方案通过检查                       | 不变                                                        |
| 物化与实际验证     | 确认写出的布局符合关系                     | 16 个已编码结果、7 个已绑定 view、16 个临时标记、0 个根契约 |
| 保留根契约         | 不重新求解                                 | 根契约增加到 7 个                                           |
| 转换优化           | 不重新求解                                 | 临时标记清零、三个常量提升，encoding 与根契约保留           |
| 最终验证及提交     | 成功统计为`committed`                    | 验证后的工作副本替换原始 Module                             |

这条流水线最关键的分界是：**先在图上证明并选择，后把选择写入 IR，再对实际 IR 重新验证，最后才提交。**本例各阶段都未产生额外 layout conversion；这说明选中的两个 MMA、Softmax 归约/广播链和最终 Store 可以直接按上述布局衔接，不意味着可以省略 Copy、Reduce 或 MMA 本身的执行工作。
