# Task 19–21 迁移对比汇报：从旧 Op 布局推断到统一约束与可验证契约

整理日期：2026-09-19。实现核对基线：`main` 的 `14b39e9`；Task 19/20 提交为 `c028e82`，Task 21 提交为 `14b39e9`。

本文比较旧 Frisk 的实际推断路径与三个任务的实际增量，不把后续规划当成已经实现的能力。测试数字引用已完成的验收记录，本次文档整理未重新运行测试。

后续更新（2026-09-27）：上文整理日期及第 1–10 节为 Task 19–21 的历史汇报；Task 22 的完整例子见第 11 节。Task 23 已解除旧求解器 8/4 上限，完整链实际测试通过，当前边界以 §11.8 更新为准。

## 1. 汇报结论：不是简单搬代码，而是重新表达和验证旧规则

Task 19、20、21 的共同目标，是让原先分散在各个 Op 中的布局规则进入新 Frisk 的统一推断流程。但三个任务的工作并不完全相同：

- Task 19 保留 Copy、Fill、Parallel 的操作形式，补充执行分工、写入责任和线程资源约束。
- Task 20 新增 Tensor 形式的 `frisk.mma`，把原有矩阵指令布局知识改造成联合指令契约，并新增实际寄存器/存储地址证明。
- Task 21 新增纯 Tensor 形式的 `frisk.reduce_tensor`，在原有删维布局思想上，增加精确贡献者证明和通信契约。

最重要的变化，可以概括为：

> 旧路径主要由 Op 根据形状、目标和已有布局计算局部布局；新路径将布局要求表达为有限约束，统一选择相容方案，并在方案写入真实程序后重新证明其正确性。

需要先澄清三点，避免汇报时夸大增量：

1. 旧 Frisk 本来就有 Tensor Core 相关布局规则。Task 20 不是第一次支持 Tensor Core，也不是新增 CUDA Core 矩阵乘布局。
2. 统一图、有限求解、显式转换和事务物化等基础设施主要来自 M0–M3 与 Task 18。Task 19–21 是将更多操作接入这些基础设施，并补齐相应的约束语义和证明。
3. 旧 `frisk.gemm`、旧 Buffer Reduce 及其推断方法仍然保留。将旧程序自动转换为新 Tensor 操作、退役旧生产路径，是 Task 22 的工作，不能说本次已经完成所有旧 IR 迁移。

## 2. 整体对比：究竟改变了什么

本文中的“旧逻辑”区分两个来源：一是 legacy 的 `LayoutAttr + DenseMap<Value, Attribute>` Op 推断；二是新框架在 M2/M3 时已经存在、但尚未完整适配这些 Op 的受限规则。二者不能混为一谈。

| 对比维度 | 旧路径或任务前的状态 | Task 19–21 的实际变化 |
| --- | --- | --- |
| 布局表达 | legacy 用统一的 `LayoutAttr`，Local MemRef 也用于表达寄存器片段 | Copy/Fill 保留 MemRef 并增加执行属性；MMA/Reduce 的寄存器值使用 Tensor encoding；存储与分布分开 |
| 推断方式 | Gemm/Reduce 等局部方法计算布局并更新共享映射表 | Op 要求进入约束图，经过有限候选准备、传播、求解和实际 IR 验证 |
| 多端点相容性 | 旧 Gemm 按一套局部规则构造 A/B/C 布局 | MMA 用四角色联合契约选择完整合法组合；Reduce 用专门的输入—输出合法组合 |
| 消费者特殊要求 | legacy 布局表以 Value 为键保存布局 | 复用新框架的生产者/消费位置分离，在需要的使用点插入转换，不随意修改显式生产者布局 |
| 副本含义 | legacy 已有 replication 表示，但表示副本不等于证明执行正确 | Copy/Fill 检查谁实际写入；Reduce 检查谁实际贡献、输出是否是完整值 |
| 硬件要求 | legacy 已根据目标、warp policy 构造 shared/fragment 布局 | 新增可物化的操作契约，进一步核验 packed 寄存器、实际 descriptor 地址、线程及通信范围 |
| 验证依据 | legacy 有 shape、目标、映射兼容性等检查，但没有本次同等的操作级证明闭环 | 从实际类型、存储绑定和操作属性重新核验；不靠旧候选缓存或假定未来会插转换 |
| 支持范围 | 旧路径包含不同目标、操作种类和部分 legacy 特例 | 新路径选择明确受限子集，不能简单理解为旧能力的全面超集 |

例如，旧 Gemm 最后对 A/B/C 使用 `try_emplace` 写入布局表，不能把这种“给每个值保存一个布局”的行为等同于“从多个候选中证明四角色联合相容”。但也不能据此说旧 Gemm 完全没有指令规则：其 shared 和 fragment 构造本来就有目标相关知识。

### 2.1 什么是“Tensor 形式”

这里的 Tensor 是编译器 IR 中的“有形状、有元素类型的多维数据值”，不是指 PyTorch 对象，也不是指 Tensor Core 硬件。

例如 `tensor<64x64xf32>` 表示一个包含 `64×64` 个 f32 元素的二维值。Frisk 可以进一步给它附加 encoding，描述这些元素由哪些线程、哪些寄存器槽持有。可以把它理解为：

```text
Tensor 的 shape 和 dtype：这批数据是什么。
Tensor 的 Distributed encoding：这批数据如何分配给线程和寄存器。
```

“Tensor 形式的操作”是指用这种值表达计算输入或结果。例如：

```mlir
%r = frisk.reduce_tensor %x {kind = "sum", dim = 1}
  : tensor<64x64xf32> -> tensor<64xf32>
```

这条操作的含义是“根据 `%x` 计算出一个新值 `%r`”，而不是“找到某个目的地址，把它原来的内容改掉”。`%x` 在语义上仍表示原来的值。

需要区分两个层次：一般 MLIR Tensor 类型本身不保证数据一定驻留寄存器；新 Frisk 在当前布局体系中，将带 Distributed encoding 的 Tensor 用作线程分布的寄存器 tile 表示。后续如何分配物理寄存器、是否出现 spill，仍属于后端工作。

### 2.2 Tensor 与 MemRef 的主要区别：数据值和存储引用

MemRef 可以理解为“指向某块存储的带形状引用”。通过它可以读取或修改存储内容，多个视图也可能引用同一底层存储。

| 对比项 | Tensor 形式 | MemRef 形式 |
| --- | --- | --- |
| 表达重点 | 一次计算产生的数据值 | 一块可被读写的存储或其视图 |
| 更新表达 | 计算产生新结果值，旧值的含义不变 | 写操作改变所引用存储的内容 |
| Frisk 中的典型角色 | 寄存器片段、MMA 累加值、Reduce 输入/结果 | shared/global buffer、Copy/Fill 的目标 |
| 布局重点 | 逻辑元素如何分配到线程与寄存器槽 | 逻辑元素对应哪个内存地址 |
| 依赖分析重点 | 谁产生这个值、哪些操作消费这个值 | 谁读写这块存储、不同引用是否可能别名 |

两者在 MLIR 中都可以使用 SSA 名字，不能说“只有 Tensor 才是 SSA”。区别是：MemRef 的 SSA 名字固定了一个存储引用，但该存储里的内容仍然可以变化；Tensor 的 SSA 名字表达的是一个固定含义的数据值。

SSA 即“静态单赋值”：每个结果名字在 IR 中有一个定义。循环中则通过循环携带参数传递下一轮的值，并不是要求每轮永远使用同一份初始数据。

### 2.3 为什么要把寄存器计算改成 Tensor：让依赖和布局边界更明确

旧 Frisk 用 Local MemRef 表达寄存器片段，容易把“一个计算值”和“可被多次读写的存储”放在同一种模型中。转成 Tensor，不是改变矩阵乘法或归约的数学目的，而是让编译器更直接地看到计算过程。

以两次矩阵乘加为例，下面是省略类型、属性和具体语法的语义示意，不是可直接运行的 IR：

```text
缓冲区式表达：
    将 C 初始化为 0
    C ← C + A0 × B0
    C ← C + A1 × B1

Tensor 值式表达：
    C0 = 零 Tensor
    C1 = mma(A0, B0, C0)
    C2 = mma(A1, B1, C1)
```

第一种表达必须结合写入顺序、别名和内存效应，判断某次读取 C 时看到的是哪一版内容。第二种表达直接说明：第二次 MMA 消费第一次的结果 C1，最后产生 C2。

它对新布局系统有四个具体作用：

1. **更容易建立生产者—消费者关系。** MMA 产生什么值、Reduce 读取什么值，直接体现在操作数和结果上，不必仅靠追踪某个 buffer 的历次写入来恢复数据流。
2. **布局可以跟随明确的计算值。** C1、C2 都有自己的类型和编码；有关联时由 `SameLayout`、`ReductionLayout` 等约束明确表达，而不是因为复用了一个 buffer 名字就默认所有阶段都是同一关系。
3. **不同使用者可以有不同消费要求。** 如果同一个值的某个使用者需要另一布局，可以只在那个使用点插入 `convert_layout`，其他使用者仍使用原值。转换保持逻辑数据不变，但可能需要真实数据交换，并不保证零成本。
4. **更容易验证计算前后的一致性。** 验证器能够检查“实际输入编码、实际结果编码、操作契约”是否匹配；Reduce 也能明确规定结果副本必须表示完整归约值，而不是存储中某个尚未完成的中间状态。

这些好处不意味着 Tensor 天然免除所有分析。读取 shared 的 MMA 仍依赖内存读语义和真实地址证明；循环仍需明确数据流；布局合法性仍由相应约束和验证器证明。

### 2.4 具体对应到 Task 19、20、21

- **Task 19 没有把所有操作转成 Tensor。** Copy/Fill 的职责就是读取或修改真实存储，因此继续使用 MemRef，通过额外执行属性说明线程分工。
- **Task 20 主要将累加值和结果改成 Tensor。** `frisk.mma` 显式接收 init 并返回 result；SS 路径的 A/B 仍是 shared MemRef，RS 路径的 A 是 Tensor、B 是 shared MemRef。因此它是混合输入形式，不是所有操作数都必须是 Tensor，也不能笼统称为无内存效应的纯操作。
- **Task 21 的 Reduce 是纯 Tensor 计算。** 输入一个值，返回归约后的值，不隐式清零目的内存或写回。需要保存结果时，再接显式 `tile_store`。

也就是说，新体系不是“用 Tensor 取代全部 MemRef”，而是各司其职：计算值用 Tensor 表达，真实存储继续用 MemRef 表达，必要时用 `tile_load/tile_store` 明确连接两者。MMA 能直接读取 shared，因此也不是所有 MemRef 输入都必须先经过 `tile_load`。

### 2.5 不要把 Tensor 形式理解成额外分配或性能保证

在 IR 中产生 C1、C2 两个不同的值，不代表运行时一定额外申请两块内存，也不代表一定复制整个矩阵。只要生命周期与指令要求允许，后续寄存器分配可以复用物理资源；若两个值仍同时需要，也不能随意覆盖。

同样，Tensor 形式不会自动让程序使用 Tensor Core：普通加法、转置、Reduce 都可以采用 Tensor 形式；是否使用 Tensor Core 由具体操作与目标指令契约决定。

本次转换的直接价值是表达准确、布局约束清晰、结果可验证，为后续优化提供基础，而不是已经证明寄存器更少、显存占用更低或运行速度更快。旧 Buffer 程序自动转换成这些 Tensor 操作，仍需 Task 22 处理，不能直接替换类型名字就完成。

## 3. Task 19：从存储布局适配，扩展到可验证的线程执行分工

### 3.1 保留了什么

Copy 仍然复制 MemRef 中的数据；Fill 仍然向 MemRef 填常数；Parallel 仍然表达并行区域和线程数。没有为了迁移而把 Copy/Fill 强制改成 Tensor 算子。

旧 Parallel 已会根据线程数为部分 buffer 准备布局，并递归处理内部 Parallel/Gemm。因此，“旧版完全不知道线程数”是不准确的。

Task 19 前的新 pass 已经不调用旧 Parallel 的递归推断。此次补充真实调用计数测试来确认隔离，而不是第一次建立这条隔离边界。

### 3.2 修改和新增了什么

**第一，纠正早期新版 Copy 的过强限制。** M2 的受限 Copy 关系要求源和目的的存储 map 相同。Task 19 改为按同一个逻辑坐标分别计算两边地址：复制要求逻辑元素对应，不要求物理排布相同。

同时区分 Copy 与 Alias：复制两块独立存储，不等于它们是同一存储的两个视图；已知同根的非恒等重叠复制则保守拒绝。

**第二，给 Copy/Fill 增加真实执行契约。** 存储布局回答“元素在哪”，执行布局回答“哪个线程负责哪个元素”。新增并物化：

- `frisk.execution_layout`：线程、寄存器与逻辑元素的对应关系。
- `frisk.execution_threads`：执行线程数。
- `frisk.writer_policy`：实际写入者的选择规则。
- `frisk.vector_bytes`：需要满足的访问宽度。

原来只有名字、没有完整求解语义的 `Ownership`、`ResourceLimit`，现在真正参与筛选、求解和最终核验。

**第三，检查唯一写入与向量访问。** 多个线程持有同一元素，不代表它们都能写。要求较宽访问时，还需证明同一线程连续持有、地址连续、根对齐充分、没有缺失尾部；不是看到总字节数够大就认为可以向量化。

**第四，线程数成为贯穿全流程的约束。** Parallel 的线程环境参与候选构造、约束检查与实际结果复验；显式冲突不能被静默改写。外部 Tensor 必要时在 Parallel 内部使用点转换，保留生产者的原有编码。

### 3.3 例子：行主序复制到列主序

有两块独立的 `2×2` f32 存储 A、B，A 按行存放，B 按列存放。两边字节偏移分别为：

```text
A(i,j) = 8i + 4j
B(i,j) = 4i + 8j
```

逻辑元素 `(0,1)` 应从 A 的偏移 4 读取，写到 B 的偏移 8。复制后逻辑矩阵相同，只是物理存放顺序不同。

早期 M2 的“map 必须相同”会排除这种情况；Task 19 可以在其他契约满足时接受。注意，这不是逻辑矩阵转置，也不表示本阶段已经生成 GPU 搬运代码。

再假设这个元素被两个线程持有：选择 `first_owner` 后只允许一个规范持有者写入；若要求所有持有者都写，则违反唯一写入契约。这里新增的是对实际执行责任的验证，而不只是更多布局模板。

## 4. Task 20：从局部构造 GEMM 布局，扩展到联合指令契约

### 4.1 旧版原本已经有什么

旧 `GemmOp::inferLayout` 已经能够识别 Ampere/Hopper，结合线程数和 warp policy，构造 shared 布局以及 A/B/C 的寄存器 fragment；Hopper 路径包含 WGMMA 相关选择。

这些就是原有 Tensor Core 布局知识。Task 20 沿用其可验证的硬件思想和迁移样例，而不是从零发明矩阵片段的线程分布。

旧操作的主要表达方式是 A/B/C MemRef，C 位于 Local memory space；推断结果写入 Value→Layout 映射表。旧 verifier 要求 A/B/C 同 dtype，并按未转置形状检查尺寸，不能直接充当新 Tensor 操作的完整数学检查。

### 4.2 修改和新增了什么

**第一，新增 `frisk.mma`，明确数据依赖。** 数学语义为：

```text
result = init + A_eff × B_eff
```

init 与 result 是 Tensor SSA 值，A_eff/B_eff 根据转置属性解释。不再用 Local MemRef 模拟新路径的累加器；首次计算应传显式零 Tensor，而不是靠隐藏清零行为。

目标路径支持 f16/bf16 输入和 f32 累加；SS 表示 A/B 都在 shared，RS 表示 A 为 Tensor 寄存器片段、B 在 shared。当前目标要求明确的 `sm_90a`。

**第二，将指令相容性变成四角色联合约束。** Task 20 让原先仅有枚举名的 `InstructionContract` 具有实际语义，关联 `[A 消费角色, B 消费角色, init 消费位置, result]`。

求解时必须存在一套完整方案同时支持四个角色，不能把每个角色单独合法误认为组合合法。每个角色最多 4 个候选，对应最多 `4⁴=256` 个编码组合；这不是无限搜索或完整性能最优求解。

**第三，从布局公式进一步走到实际指令证明。** 新路径检查：

- 每个线程及寄存器槽是否对应指令规定的矩阵坐标。
- RS 的 16 位 A 元素低半字/高半字打包顺序是否正确。
- warp-group 的组织及大 tile 的基本片段拼接是否符合契约。
- shared descriptor 重建的地址是否与真实存储绑定、切片起点、父存储间距和根对齐一致。

**第四，将方案写入操作并独立验证。** 通用 Tensor/Storage encoding 继续使用，新增操作级 `MmaInstructionContractAttr` 和 `MmaDescriptorPlanAttr`；没有新增一整套专用 MMA Tensor encoding 子类。验证不重新搜索另一条指令方案来掩盖当前绑定的错误。

### 4.3 例子：单独合法，不代表能拼成一条指令

假设合法方案只有两套，下面的名字仅作机制示意：

| 方案 | A | B | init 消费布局 | result 布局 |
| --- | --- | --- | --- | --- |
| 一 | A1 | B1 | C1 | C1 |
| 二 | A2 | B2 | C2 | C2 |

A1 和 B2 分别都在某套合法方案中，但 `A1+B2+C1+C1` 没有完整方案支持，必须拒绝。新增联合约束明确表达并检查这种配套要求，而不只是为各个 Value 写入编码。

如果外部 init 已有其他布局，新框架在 MMA 消费处插入转换，保持外部值不变；MMA 的 init 消费位置与 result 必须满足 `SameLayout`。

另外，shared 切片变小不意味着父存储变紧凑。Task 20 的实际地址证明会拒绝“按切片尺寸猜 descriptor 间距”的错误方案，即便其逻辑矩阵形状看起来完全正确。

## 5. Task 21：从结果布局投影，扩展到完整贡献与通信证明

### 5.1 旧版原本已经有什么

旧 `ReduceOp::inferLayout` 并非没有检查：它检查源布局元数据、静态归约维、replication、线程与副本规模关系，并会对已有目的布局进行兼容性检查。

其主要流程是：在 Local MemRef 源布局已知的情况下，消除归约维、压缩复制维、推导目的 thread/index 映射，然后更新布局表。

这类规则主要回答“归约后的结果如何分布”。它与“每个不同输入是否恰好贡献一次、合并依赖是否正确、所有输出副本是否完整”不是同一份证明。

### 5.2 修改和新增了什么

**第一，新增纯 Tensor 归约。** `frisk.reduce_tensor` 接收一个 Tensor，返回删去归约维度的新 Tensor，没有目的 MemRef、隐式 init/clear 或内存写回。支持 `sum/max/min`，同类型 f16/bf16/f32。

这是有意调整语义，而非旧接口原样复制：旧 `add` 对应新 `sum` 的概念，但尚未自动改写；旧 `mul/clear` 不在首版迁移范围。sum 明确允许并行重结合；max/min 明确传播 NaN 并区分正负零。

**第二，重新实现自然输出布局投影。** 在通用 BitLinear 表示中删除归约坐标，按顺序保留投影后独立的寄存器位，压缩零列和非零相关列，重新计算寄存器数量及 replication。沿用的是删维思想，不是机械翻译旧 replication 公式。

**第三，增加精确贡献证明。** 每个不同逻辑输入只选一个规范物理持有者，按确定顺序构建相邻合并树，最后将完整结果分发到所有输出持有者。验证覆盖遗漏、重复、跨行混合和不完整分发。

**第四，新增归约专用关系与通信绑定。** `ReductionLayout` 是输入消费位置与结果之间的二端点关系，不伪装成可逆转置，也不硬套 MMA 的四角色约束。正向准备自然输出候选，反向只筛选已有合法组合；每条关系最多 16 对布局。

`ReductionContractAttr` 记录可重建的算法与策略，以及 register、warp 或 CTA shared-tree 通信范围。实际 IR 验证从真实编码重建依赖，不重新枚举候选。

### 5.3 例子：4 个加数各有 2 个副本，和仍然是 10

```text
不同逻辑输入：1、2、3、4
物理持有情况：每个输入都有两个同值副本
正确行和：   1+2+3+4 = 10
错误行和：   两份副本都参与计算，得到 20
```

旧布局投影可以描述结果的持有关系，但不能仅凭这份投影就证明没有重复加数。Task 21 增加的贡献证明会检查每个逻辑输入只参与一次。

如果输出也有两个持有者，它们都必须持有 10；不能把两个尚未合并的部分和 3、7 当作两个同值副本。这里的新增能力不是“更多 Reduce 布局”，而是明确区分不同加数、输入副本和完整输出副本。

## 6. 三个任务共同新增的闭环能力

三个任务依托既有新框架，分别补齐操作规则，形成如下闭环：

```text
操作语义及硬要求
      ↓
有限候选与合法组合 → 传播删减 → 求解
                                  ↓
                        写入临时程序的实际属性
                                  ↓
                        按实际 IR 独立重新验证
                                  ↓
                     成功才更新原程序，失败则不改写
```

共同收益是：

- 规则不再只是一段局部布局生成函数，而是可参与跨操作相容性检查的约束。
- 写入者、指令选择、归约算法等执行约定显式留在 IR 中，不只存在于临时推断表。
- 硬要求不能被候选建议或代价选择悄悄覆盖；证明超预算不能当作正确。
- 可以测试“属性被删掉或篡改后是否拒绝”“二次推断是否稳定”“失败是否保持原程序”。

但不能把所有基础设施都算作这三个任务新建，也不能把“静态契约可验证”说成“GPU 指令已经生成并执行正确”。Task 19–21 的重点是操作级规则和证明的接入与增强。

## 7. 支持边界：新路径不是旧路径的无条件超集

| 项目 | 当前支持与限制 |
| --- | --- |
| Copy/Fill | 静态受限整块访问与已有存储视图；未增加整数 Fill、任意动态索引或通用异步复制 |
| Parallel | 单 CTA、32/64/128/256/512/1024 线程；保留对实际线程组织的检查 |
| MMA | 显式 sm_90a、SS/RS、f16/bf16 输入与 f32 累加；不支持 Tensor B，也未继承旧路径所有 Ampere/Local MemRef 形式 |
| Reduce | 静态二次幂、各维大于 1、输入至少二维；暂不支持标量输出、动态/ragged、mul 或部分和专用类型 |
| 求解预算 | Task 21 交付时为每组件最多 8 变量、每域最多 4 候选；Task 23 已改为有界 exact/beam，见 §11.8 |
| 后续工作 | 旧 Buffer 程序自动 normalization、完整 GPU lowering、同步/资源落实和性能成本优化尚未由这三项交付 |

尤其要保留 Task 21 当时的验收差异：`MMA→Reduce` 与 `Reduce→Global tile_store` 已分别验证成功，但完整 `MMA→Reduce→tile_store` 有 9 个连通变量，超过当时的 8 变量上限。

其数量是 MMA 的 5 个变量，加 Reduce 的 2 个，再加 store 的 2 个。当时完整链测试验证超限拒绝，不能把分段成功拼成端到端成功。该历史缺口后来由 Task 23 的完整成功测试补齐，而非追溯计作 Task 21 已完成。

## 8. 验证结果：增强点有测试，边界也有拒绝用例

以下是各任务实施后的全量回归规模，不是每个任务各自新增的测试数：

| 任务 | 全量单元测试 | 全量 lit | CTest | 本任务新增单元测试 |
| --- | --- | --- | --- | --- |
| Task 19 | 129/129 | 32/32 | 4/4 | 25 |
| Task 20 | 165/165 | 34/34 | 4/4 | 36 |
| Task 21 | 197/197 | 36/36 | 4/4 | 32 |

针对上述区别，代表性测试包括：

- Task 19：不同物理存储布局的 Copy；副本不能全部写；向量宽度与显式布局冲突；新 pass 不调用旧 Parallel 推断。
- Task 20：四角色联合支持；SS/RS、类型与转置组合；descriptor、半字打包、线程组织被篡改后拒绝；外部编码保持不变。
- Task 21：重复加数、错误 partial、漏分发；投影独立坐标 oracle；线程环境继承；MMA/Reduce 分段集成和完整链超限。

旧测试也不是无条件照抄。Task 20 对旧 RS replication 进行硬件契约分类；Task 21 将错误 replication、丢失 batch 坐标的旧样例作为反例，适用样例则用独立计算核对。原则是保留已证明正确的规则，而不是为了与旧输出相同而保留错误。

以上是 CPU 侧布局分析、IR 变换和静态证明测试，不是 GPU 数值/性能验收。没有据此得出性能领先 TileLang/Triton 的结论。

## 9. 可直接用于口头汇报的总结

> Task 19–21 不只是把旧 Frisk 的 Op 推断函数搬进新目录，而是将其改造成统一约束体系中的可验证规则。Task 19 重点补齐 Copy、Fill、Parallel 的线程分工、写入责任和访问契约；Task 20 在已有 Tensor Core 布局知识上，新增 Tensor MMA、四角色联合约束，以及寄存器和真实存储地址的指令兼容证明；Task 21 在已有归约投影思想上，新增 Tensor Reduce、精确贡献者证明与通信契约。三个任务共同实现了“生成方案、统一选择、写入 IR、独立复验”的操作级闭环。当前不是旧能力的全面超集，也尚未完成旧 IR 自动转换或 GPU 指令生成；Task 21 的完整 MMA→Reduce→store 链仍受现有求解预算限制。

## 10. 代码与材料索引

旧路径仍可在当前仓库中核对；其中公共 verifier 等已有本轮增量，查任务前行为时应结合 `ef85a0d` 基线和实施审计记录，不能把当前文件的所有检查都当作 legacy 已有。

| 对象 | 主要证据 |
| --- | --- |
| 旧 Parallel/Gemm 局部推断、旧 Op verifier | [FriskOps.cpp](../../lib/Dialect/Frisk/IR/FriskOps.cpp) 的 `ParallelOp::inferLayout`、`GemmOp::inferLayout/verify` 等 |
| 旧 Reduce 删维、replication 压缩与兼容性检查 | [FriskOps_Reduce.cpp](../../lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp) |
| Task 19 新约束与执行证明 | [OperationLayoutConstraints.cpp](../../lib/Dialect/Frisk/Analysis/OperationLayoutConstraints.cpp)、[ExecutionLayoutProof.cpp](../../lib/Dialect/Frisk/Analysis/ExecutionLayoutProof.cpp) |
| Task 20 联合约束与指令证明 | [InstructionLayoutConstraints.cpp](../../lib/Dialect/Frisk/Analysis/InstructionLayoutConstraints.cpp)、[SM90MmaLayoutProof.cpp](../../lib/Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.cpp) |
| Task 21 归约约束与贡献证明 | [ReductionLayoutConstraints.cpp](../../lib/Dialect/Frisk/Analysis/ReductionLayoutConstraints.cpp)、[ReductionLayoutProof.cpp](../../lib/Dialect/Frisk/Analysis/ReductionLayoutProof.cpp) |
| 实际 IR 物化与复验 | [MaterializeLayouts.cpp](../../lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp)、[LayoutVerifier.cpp](../../lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp) |

详细汇报：[Task 19](./task19_report.md)、[Task 20](./task20_report.md)、[Task 21](./task21_report.md)。

维护依据：[实现计划](../layout_inference_implementation_plan.md) §19.1/19.9、§20.1/20.13、§21.1/21.8；[布局系统设计](../layout_inference_design.md)；[三方策略比较](../layout_inference_strategy_comparison.md)。

## 11. 完整例子：Copy → Gemm → Reduce → 写回，新旧系统分别怎样推断

本节补充 Task 22 完成后的状态。前文“旧推断仍在生产保留、normalization 尚待实现”是 Task 19–21 时点的记录；目前旧公式已迁入测试专用 [LegacyLayoutOracle.cpp](../../test/Support/LegacyLayoutOracle.cpp)，生产入口使用新 pipeline。

### 11.1 同一个计算任务

在一个 128 线程的 Parallel 中，读取两个矩阵，执行矩阵乘加，再逐行求和：

```text
A：64×16，f16，Global 输入
B：16×64，f16，Global 输入
C：64×64，f32，中间累加值
Y：64，   f32，Global 输出

C[i,j] = 1 + Σ(k=0..15) A[i,k] × B[k,j]
Y[i]   = Σ(j=0..63) C[i,j]
```

若 A、B 全为 1，则每个 C 元素为 17，每个 Y 元素为 1088。下面的代码均为省略具体语法的推断示意。

旧 Buffer 形式：

```text
target = sm_90a
parallel(128 threads) {
    As = alloc Shared[64,16] f16，显式 alignment=1024
    Bs = alloc Shared[16,64] f16，显式 alignment=1024
    C  = alloc Local [64,64] f32
    R  = alloc Local [64]    f32

    Copy(A_global, As)
    Copy(B_global, Bs)
    Fill(C, 1.0)
    Gemm(As, Bs, C, transA=false, transB=false,
         policy=Square, clear_accum=false)
    Reduce(C, R, kind=add, dim=1, clear=true)
    Copy(R, Y_global)
}
```

这里选 SS MMA：A/B 都在 Shared，C 表示寄存器累加片段。Shared allocation 的对齐是输入中真实声明的保证，不是由 MMA 的需求倒推出来的。

**精度边界：** Task 22 扩展后的旧操作导入 verifier 支持 f16 A/B＋f32 C；原始旧 verifier 要求 A/B/C 同 dtype。下面旧侧对照的是历史布局推断规则，不声称这段混合精度文本在旧版本中可以原样通过，更不把旧 f16 C 自动提升成 f32。

### 11.2 旧系统：Gemm 决定 A/B/C，Reduce 再由 C 推出 R

旧 Gemm 读取目标、128 线程、M/N/K、transpose 和 warp policy，调用目标相关公式，分别构造：

```text
L_As：Shared A 的布局
L_Bs：Shared B 的布局
L_C ：累加片段的线程/索引布局
```

然后写入共享布局表：

```cpp
layoutMap.try_emplace(As, L_As);
layoutMap.try_emplace(Bs, L_Bs);
layoutMap.try_emplace(C,  L_C);
```

旧 Reduce 被调用且找到 `layoutMap[C]` 后，再执行：

```text
C 的线程映射
  → 消去列维，将相应持有关系转入复制维
  → 压缩复制维并推导结果索引
  → 得到 L_R
  → 写入/检查 layoutMap[R]
```

这条推断依赖可概括为：

```text
Gemm 的目标公式 → L_As、L_Bs、L_C
                                  └→ Reduce 局部投影 → L_R
```

旧规则已有 Tensor Core 布局知识、replication 表示及部分兼容性检查。局限不在于“完全不会推断”，而在于这些结果主要以每个 Buffer 一个表项组织：

- `C` 的生产布局与 Reduce 消费 `C` 的布局，没有成为两个可协调的端点。
- Copy 的访问分工、Gemm 的指令组合、Reduce 的贡献者没有进入同一张操作约束图共同检查。
- `try_emplace` 在表项已存在时不会执行联合候选求解；不能将“不覆盖旧值”解释为“已证明旧布局满足新操作”。
- 旧 Reduce 找不到源布局时可以直接返回 success，不代表这条链已经推断完整。

另外，保留的旧 Parallel 实现并没有自动调度完整的 Copy→Gemm→Reduce→Copy 求解链；这里展示的是各旧规则之间的依赖，不是假设旧系统已有同样的完整 pipeline。

### 11.3 Task 22：先把隐式 Buffer 更新变成明确的生产/消费关系

为导入操作声明 `frisk.legacy_semantics="tensor_v1"` 后，归一化得到：

```text
parallel(128 threads) {
    // Global/Shared 仍为真实存储；补充或复用 Storage view。
    Copy(A_global_view, As_view)
    Copy(B_global_view, Bs_view)

    c0 = TensorConstant<64×64×f32>(1.0)
    c1 = mma(As_view, Bs_view, c0)
    r  = reduce_tensor(c1, kind=sum, dim=1)
    tile_store(r, Y_global_view)
}
```

转换前后最关键的对应关系是：

| 原操作 | 转换后 | 对布局推断的作用 |
| --- | --- | --- |
| Global→Shared Copy | 保留 Copy，端点绑定 Storage view | 读写地址与执行线程分工能够进入新图 |
| Local `Fill(C,1)` | 产生 `c0` Tensor | 明确 MMA 的初始累加值，不隐式读取未知 C |
| Gemm 修改 C | `c1 = mma(As,Bs,c0)` | init、结果及各消费位置可以分别建立布局变量 |
| Reduce 读取 C、修改 R | `r = reduce_tensor(c1)` | 明确 MMA 结果到 Reduce 输入的依赖 |
| Local→Global Copy | `tile_store(r,Y)` | 将结果消费布局与输出地址映射连接 |

这里的 **Local Fill→常量属于 Task 22**；它不是 Task 19 对 Shared/Global Fill 的线程写入规则。Buffer 引用原本也是 SSA 值，真正改变的是“引用背后会变的内容”现在由 `c0/c1/r` 三个明确的计算值表示。

### 11.4 Task 19：Copy 的布局不能只看目的 Shared buffer

以 `Copy(A_global, As)` 为例，Global A 若采用普通行主序，则：

```text
S_global(i,k) = 2 × (16 × i + k) 字节
```

Shared As 可以采用另一种合法映射，例如满足后续 MMA 要求的 swizzle 布局。两端布局不必相等，但必须复制同一个逻辑元素：

```text
线程/寄存器位置 h
      │ 执行布局 E_copy
      ▼
逻辑坐标 (i,k)
      ├→ S_global(i,k)：从哪里读
      └→ S_shared(i,k)：写到哪里
```

Task 19 给 Copy 增加的正是这些要求：

- 执行分工必须覆盖要复制的逻辑元素。
- 源、目的访问必须对应同一逻辑坐标，而非强迫两端物理布局相同。
- 若执行布局含副本，写入责任必须合法，不能让全部副本无条件写同一地址。
- 访问的向量宽度、对齐与 Parallel 的 128 线程环境必须相容。

**与旧路径的区别：** 不只是拿到 `L_As` 后认为 Copy 可以照做；Copy 自己也对 `As` 的候选布局提出约束。随后 Gemm 又对同一个 Shared 存储提出指令要求，二者必须共同成立。这里的 swizzle 是候选说明，不是宣称完整链当前已经选中了某个具体 swizzle。

### 11.5 Task 20：MMA 选择的是完整合法组合，不是三个独立表项

新 MMA 的核心联合关系是：

```text
(As 的实际 Storage 布局,
 Bs 的实际 Storage 布局,
 c0 在 MMA 消费位置的 Distributed 布局,
 c1 的 Distributed 布局)
                 ↓
       必须属于同一个合法 MMA 组合
```

同一个 `As_view` 同时受到上一步 Copy 访问要求和这里的 MMA 指令要求约束，不能由两段代码各自决定一份互不相干的布局。

Task 20 的证明还会核对实际 Shared 字节地址、descriptor 条件、线程组织和累加片段的寄存器映射。硬件契约不满足的候选不能靠“另一个候选代价更低”或“旧公式曾经生成过它”来放行。

对 `c0`，新系统区分：

```text
c0 的生产布局 ── Convertible ── MMA 实际消费 c0 的布局
```

能共用就共用；需要且允许转换时在消费点转换，不修改显式生产者布局来掩盖冲突。选定的指令组合写入 `mma_contract`，后续从真实 IR 重新核验。

**与旧路径的区别：** 旧系统已有 Shared/fragment 公式；新增的是四角色联合相容性、实际地址/寄存器证明，以及它们与其他操作约束共同求解的接线。

### 11.6 Task 21：Reduce 不仅推导结果布局，还证明每个 C 元素恰好贡献一次

MMA 输出的 `c1` 布局首先满足矩阵指令要求，但 Reduce 按行求和需要检查另一类关系：

```text
c1 的生产布局
      │ Convertible
Reduce 消费 c1 的布局
      │ ReductionLayout
r 的自然输出布局
      │ Convertible
tile_store 消费 r 的布局
```

这里不强制 MMA 输出布局和 Reduce 消费布局天然相同，也不直接拿 Reduce 的输出要求覆盖 MMA 结果；二者通过明确的使用点关系协调。

对某一行，本例有 64 个不同的 C 元素。A/B 全为 1 时，这些元素都等于 17，正确结果为：

```text
64 × 17 = 1088
```

若某个候选布局使每个 C 元素有两份物理副本，正确贡献者数量仍应是 64，不能变成 128。若输出也有两份副本，每份都必须持有完整的 1088，不能各持有部分和 544。

Task 21 据实际候选布局检查逻辑输入的唯一贡献、归约树、完整结果分发和通信范围，并将方案写入 `reduction_contract`。

**与旧路径的区别：** 旧投影主要回答“R 由谁持有”；新关系还回答“这些持有者得到的是否是完整且不重算的行和”。上面的两份副本用于说明验证条件，不是本例某个已选布局的实际复制数。

### 11.7 把四个任务放回同一条推断链

```text
Task 22：Buffer 生命周期 → 明确的 Tensor 生产/消费关系
                                  ↓
Task 19：Global→Shared Copy 的地址、分工、写入和线程约束
                                  ↓ 共享 As/Bs 的 Storage 变量
Task 20：As/Bs/init/result 的 MMA 联合指令约束
                                  ↓ Convertible
Task 21：Reduce 输入/输出的投影、贡献和通信约束
                                  ↓ Convertible + StorageAccess
                        归约结果写回 Global
                                  ↓
Task 22：保存根存储契约、优化、实际 IR 复验、事务提交
```

旧逻辑主要形成 `As/Bs/C/R → LayoutAttr` 的局部推断结果；新逻辑把“Copy 是否能搬、MMA 是否能算、Reduce 是否算全、结果是否能正确写回”表达为可共同检查的关系。源布局不必等于消费布局，存储布局不必等于执行布局；需要满足的是它们之间的操作契约。

Task 22 还保存实际 whole-root 存储映射与对齐前置条件，防止纯 view 被优化删除后丢失验证依据。最终检查读取真实类型和操作属性，而不是信任旧 `layoutMap` 或上次求解缓存；具名 pipeline 任一步失败都不改动原模块。

### 11.8 Task 22 的历史限制与 Task 23 的完整链验收

**Task 22 交付时，这个例子覆盖四个任务的作用，但整链布局求解受限。Task 23 已解除这一规模限制并验证完整链成功。** 即使先不计 Global→Shared Copy 引入的额外变量，SS MMA→Reduce→store 核心就已有：

| 部分 | 变量 |
| --- | --- |
| MMA | As Storage、Bs Storage、init 生产位置、init 消费位置、MMA 结果，共 5 个 |
| Reduce | 输入消费位置、归约结果，新增 2 个 |
| Store | 输入消费位置、输出 Storage，新增 2 个 |
| 合计 | 9 个，属于同一连通分量 |

Task 22 可以完成这个核心的旧 IR 归一化，但当时的 bootstrap solver 最多接受每连通分量 8 个变量，因此完整 pipeline 会超限拒绝并回滚。加入前面的 Copy 约束不会消除这个问题。这是历史验收边界，不再是当前求解器的变量数限制。

2026-09-27 Task 23 的实际测试现在覆盖：

- `NineVariableFullChainMaterializesAndReplays`：9 变量核心完整求解，保留 MMA/Reduce binding，实际写回外部 Global 参数，实际 IR 复验与重跑一致。
- `GlobalCopyParallelMmaReduceStoreMaterializesAndReplays`：两次 Global→Shared Copy、Parallel128、MMA、Reduce、Global 写回完整成功，并验证序列化后重新解析、重跑一致。
- `ExplicitSearchBudgetRollsBackFullChainNormalization`：将搜索预算显式设为 1 后仍验证原模块不变，不因解除旧 8 变量限制而删除回滚覆盖。

上述用例所在的 pipeline/Reduce 集成测试本轮 25/25 通过。对应证据见 [LayoutPipelineTest.cpp](../../unittests/Dialect/Frisk/Layout/LayoutPipelineTest.cpp) 和 [ReduceLayoutIntegrationTest.cpp](../../unittests/Dialect/Frisk/Layout/ReduceLayoutIntegrationTest.cpp)。求解器改为固定候选域上的 exact/beam，MMA/Reduce 的操作级证明预算未放宽。

**因此，Task 19–22 完成统一表示、联合约束与可验证的操作契约；Task 23 再把大连通分量的统一选择接通。完整链成功依靠统一求解器，不是恢复逐个 Op 的旧式局部猜测。这里的验收是布局推断、物化与复验，不代表已生成并执行 GPU 内核。**
