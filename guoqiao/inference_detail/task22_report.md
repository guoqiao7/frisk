# Task 22 工作汇报：让旧 Buffer 程序接入新的布局推断体系

整理日期：2026-09-21。范围：M4 第四阶段 Task 22，采用已确认的方案 A。

当前状态：受限范围内的实现、独立复审和全量回归已完成。本报告总结现有实现，不新增功能；测试数字引用本轮实施结束时的验收记录，本次整理报告未重新运行全量测试。

## 1. 一句话概括

Task 22 为新旧 Frisk 之间补上了“程序转换入口”：将符合条件的旧 Local Buffer 程序转换为新 Tensor 程序，再交给统一布局推断；同时保存不能丢失的存储契约，并退出旧的生产推断路径。

与前几个任务的关系可以概括为：

| 任务 | 解决的问题 |
| --- | --- |
| Task 19 | Copy、Fill、Parallel 在新体系中如何表达访问分工和写入责任 |
| Task 20 | Tensor MMA 的输入、累加器和指令布局如何匹配 |
| Task 21 | Tensor Reduce 的布局和归约贡献如何验证 |
| Task 22 | 旧 Buffer 程序如何正确转换到上述新体系，并统一执行、验证和回滚 |

因此，本次不是新增一种 Tensor Core 矩阵乘算法，也不是重新实现 Task 20/21 的布局规则，而是让旧程序在明确的语义边界内使用这些规则。

## 2. 为什么还需要这一步转换？

旧程序常用 Local MemRef 表示中间计算结果，例如“分配 C，然后 Gemm 不断修改 C”。MemRef 描述一块可读写的存储，同一个名字在不同时间可能代表不同内容。

新体系用 Tensor SSA 表示计算值。SSA 可以理解为“每个名字只定义一次”：计算产生新值，后续操作使用明确的那个版本，而不是隐式读取某块可能已经被改写的内存。

二者的区别是：

```text
旧表达：修改 C → 再次修改 C → 从 C 读取结果
新表达：C0 → 计算得到 C1 → 再计算得到 C2 → 使用 C2
```

显式区分这些版本后，编译器才能清楚地判断哪个值由谁产生、由谁使用、是否需要布局转换，以及循环究竟传递哪个值。

但 Tensor 化不等于“把所有内存都换成 Tensor”。Task 22 仍然保持两类表示：

- Shared/Global：继续用 MemRef 表示真实存储，用 `layout_view` 绑定存储布局。
- 可提升的 Local 中间值：转换为 Tensor，交给分布式布局规则描述线程持有关系。

Tensor 也不代表已经分配好物理寄存器或生成了 GPU 指令；后续 lowering 仍需完成这些工作。

## 3. 能转换完整的 Local Buffer 生命周期

新增 `frisk-normalize-layout-ir`，负责归一化，即把不同形式的旧程序整理为新体系能够处理的统一表示。

它处理的不是孤立的 Gemm 或 Reduce，而是同一块 Local Buffer 从分配、初始化、读写到释放的完整过程：

| 旧操作 | 转换后的含义 |
| --- | --- |
| Local allocation | 建立一个“尚未初始化”的状态，完成转换后删除原分配 |
| Local Fill | 产生一个具有相同数值的 Tensor 常量，保留负零等细节 |
| Shared/Global → Local Copy | 在原读取位置产生 `tile_load` |
| Local → Shared/Global Copy | 在原写入位置产生 `tile_store` |
| Local → Local Copy | 目的端绑定源端当前的 Tensor 值 |
| 旧 Gemm / Reduce | 产生 `mma` / `reduce_tensor` 及必要的显式合并 |
| 合法的最终 dealloc | 随被提升的 Local 生命周期一起移除 |

### 3.1 具体例子：填充矩阵，再逐行求和并写回

下面使用便于阅读的伪代码，表达已通过集成测试的 `4×4` f32 用例，不是可直接粘贴的完整 MLIR。

转换前：

```text
声明接受 tensor_v1 导入语义
src = 分配 Local[4,4]
dst = 分配 Local[4]
Fill(src, 2.0)
Reduce(src, dst, kind=add, dim=1, clear=true)
Copy(dst, GlobalOutput)
```

转换后：

```text
src_value = Tensor 常量，形状 [4,4]，所有元素为 2.0
dst_value = reduce_tensor(src_value, kind=sum, dim=1)
output_view = 为 GlobalOutput 绑定存储布局
tile_store(dst_value, output_view)
```

数学上，每行有 4 个 2.0，所以输出应为 `[8,8,8,8]`。

变化在于：中间计算结果由 Tensor 值表达，最后写回 Global 才通过显式 store 表达内存修改。测试已经验证这类程序能够完成布局 pipeline，并保留真实的外部写回；这里的数学说明不是声称已经运行 GPU 内核得到该输出。

### 3.2 复制后再修改，不会误改另一个值

例如旧程序执行：

```text
Fill(x, 1)
Copy(x, y)
Fill(x, 2)
```

转换后，`y` 仍绑定旧的全 1 Tensor，`x` 改为绑定新的全 2 Tensor。因为 Tensor 值不可变，两个 Buffer 曾经引用同一个值，并不意味着后续更新一个会修改另一个。

如果读取尚未初始化的 Buffer，转换会明确失败，不会自动补零，也不会用未定义值掩盖问题。同一个根的受支持 cast 别名共享初始化和当前值状态。

## 4. 明确旧 Gemm/Reduce 的数学导入语义

旧布局推断代码主要描述数据如何分布，不能单凭它证明旧操作所有数值行为都与新操作一致。因此，旧 Gemm/Reduce 的迁移要求显式声明：

```mlir
frisk.legacy_semantics = "tensor_v1"
```

该声明可放在操作或相应的函数、Kernel、module 上，按最近的属性声明生效。它表示“导入者接受这套数学契约”，不是“编译器已经证明所有历史 GPU 实现数值等价”。Python builder 提供可选参数，但不会默认替用户声明。

### 4.1 Gemm：清零和累加必须区分

采用受支持的 f16/bf16 A、B 和 f32 C 时，转换规则为：

```text
clear_accum=true： C_new = mma(A, B, 显式 f32 正零 Tensor)
clear_accum=false：C_new = mma(A, B, C_old)
```

第一种不读取旧 C；第二种必须先有已初始化的 C。旧 C 如果是 f16/bf16，不会被静默提升成 f32，而是在归一化时拒绝。

同时，C++ verifier 和 Python/FFI builder 已同步支持低精度 A/B 配合 f32 C，并根据 transpose 后的有效形状检查 M/N/K，避免内部测试能构造、前端却无法导入。

矩阵乘迁移要求显式 `sm_90a`，支持已确定的两条路径：

- SS：A、B 来自 Shared，C 的 Local 生命周期转换成 Tensor。
- RS：A 的 Local 生命周期转换成 Tensor，B 来自 Shared，C 同样转换成 Tensor。

这些仍然接入 Task 20 的 Tensor Core MMA 布局契约，不是新增 CUDA Core 矩阵乘路径。

### 4.2 Reduce：覆盖与累加的区别能够直接看到

假设源矩阵每行的和为 8，旧目的值为 `[10,10,10,10]`。

```text
clear=true：
  reduced = reduce_tensor(src)
  dst_new = reduced                 → [8,8,8,8]

clear=false：
  reduced = reduce_tensor(src)
  dst_new = add(old_dst, reduced)    → [18,18,18,18]
```

`clear=false` 必须先完整归约，再与旧目的值合并，且旧目的值位于合并操作左侧。max/min 对应生成 `arith.maximumf/minimumf`；旧 add 映射到新 sum，mul 不在迁移支持集。

sum 接受新体系的并行重结合契约，不承诺与旧串行求和逐 bit 相等；max/min 保留约定的 NaN 传播和正负零语义。

Gemm/Reduce 的内存 effect 也同步修正：清零/覆盖时不读旧目的值，累加/合并时明确声明目的 Read，避免优化器遗漏真实依赖。

## 5. 循环和分支中的可变状态，也能转成显式 Tensor

Local Buffer 如果在循环中被反复修改，不能只删除内存操作，必须把每轮之间传递的值保留下来。

例如，以下是循环归一化的语义示意：

```text
旧程序：
  Fill(acc, 0)
  for i in 0..N:
    Reduce(src, acc, add, clear=false)
  使用 acc

新程序：
  acc0 = 零 Tensor
  result = for i in 0..N，循环携带 acc_iter，初值为 acc0:
    r = reduce_tensor(src_value)
    acc_next = add(acc_iter, r)
    yield acc_next
  使用 result
```

这里的 `acc_iter` 就是 `iter_arg`：当前这一轮收到的累加值。`yield` 将新值送给下一轮；循环结束后产生最终结果。即使 N 为 0，结果也应是入口的 acc0。

本次还处理了这些容易出错的情况：

- if：两分支合并值；某分支不写时传递入口旧值，缺失初始化的路径不能补零。
- for：只为被修改的外部 Local 根新增循环携带值，只读值可直接捕获。
- while：分别扩展 before 与 after 的原有参数列表；第一次条件为假时，before 已经发生的更新也必须保留。
- Parallel：允许其内部创建的受支持 Local 根，不允许跨 Parallel 边界捕获 Local 可变状态。
- 旧静态正步长 `frisk.for`：涉及 Local 转换时可改为 `scf.for`。

这些是归一化能力，不意味着任意复杂循环都能通过后续求解。新增 Tensor 值仍受现有图规模预算限制。

## 6. 保存优化后不能丢失的根存储契约

Task 22 新增 `frisk.storage_contract`，保存某个底层存储根的完整布局映射和对齐前置条件。

它解决的是此前一个实际问题：`layout_view` 是无副作用的视图操作，如果完整根视图没有直接使用者，优化可能将它删除，但子视图的合法性验证仍需要其中的根对齐证据。

### 6.1 具体例子：子视图仍然需要根对齐保证

假设外部传入一个含 4 个 i32 的 Shared 根存储，完整根声明：

```text
元素 i 的字节偏移 = 4 × i
根地址至少按 4 字节对齐
```

程序只使用从第 1 个元素开始、长度为 2 的子视图。子视图的地址规则为：

```text
子元素 j 的地址 = 根地址 + 4 × (j + 1)
```

如果完整根 view 被删掉，而根参数本身没有提供额外对齐证据，就不能凭空继续认定它满足 4 字节对齐。

新流程会在原来的合法作用域保存 `storage_contract`。后续即使纯 view 被消除，编译器仍可从实际 IR 读取根映射和对齐声明，重新检查子视图是否一致。

需要强调：

- 它是调用者或原始声明必须满足的前置条件，不是运行时对齐检查，也不会真的把内存地址变得更对齐。
- 它不能根据 MMA 或子视图的需求，倒推出一个原本不存在的根保证。
- 分支内的声明不能供分支外使用；未知执行作用域中的声明保守拒绝。
- 它在图中是固定的 `RootStorageContract` 约束，不是新增的可搜索布局变量，不用于绕过变量预算。

这让“布局为什么合法”的证据能够跨过优化保留下来，而不是依赖编译器内存中的隐藏状态。

## 7. 提供统一入口，失败时不留下半转换程序

新增具名入口 `frisk-layout-pipeline`，组织以下步骤：

```text
旧 IR
  → 生命周期归一化
  → 新布局推断与物化
  → 保存实际根存储契约
  → 优化布局转换
  → 根据实际 IR 独立复验
  → 全部成功后更新原模块
```

整个过程先在模块副本上执行。任一步失败，原模块保持不变。例如模块里第一个函数合法、第二个函数有不支持的 Local 用法，不会留下“前一半已经转换、后一半仍是旧程序”的状态。

独立运行 normalize 本身也具有事务性。但如果用户先单独执行 normalize 成功，再单独执行 infer 失败，先前成功的 normalize 不会被撤销；只有具名 pipeline 提供整条流程的原子性。

调用方式：

```bash
# 只转换旧程序表示。
build/bin/frisk-opt input.mlir -frisk-normalize-layout-ir -verify-each

# 执行完整布局流程。
build/bin/frisk-opt input.mlir -frisk-layout-pipeline -verify-each
```

## 8. 旧生产推断正式退役，但旧测试参考仍保留

本次删除了生产 ODS/C++ 中旧 Parallel/Gemm/Reduce 的 `inferLayout` 方法、旧接口和相关计数入口。新生产流程不再执行旧的 `Value → Attribute` 布局表更新逻辑。

旧公式没有直接丢弃，而是迁入 `test/Support/LegacyLayoutOracle*`。这里的 oracle 指“测试参考实现”，用于继续核对历史布局公式和错误分类。

隔离不仅体现在目录上，还体现在构建和链接上：该库只供测试使用，并检查生产源码、生成头文件、frisk-opt 和 FFI 二进制中不存在退役推断接口或旧 oracle 链接。

旧 `frisk.gemm/reduce` 操作名字仍可作为导入格式存在。“旧操作还能解析”和“旧推断仍在生产运行”是两回事；直接将残留旧操作送入新 infer 会得到要求先归一化的诊断。

前端联调还修复了整块 Copy 空索引的 map 构造，以及 alloc_buffer 的 Local memorySpace 解析和附加属性打印问题，验证了从本次构建的 Python FFI 生成 IR、再交给 frisk-opt 的真实路径。

## 9. 验收结果与当前限制

实施结束时记录的结果如下，具体来源见[实现计划 §22.13](../layout_inference_implementation_plan.md#2213-实际实现与验收记录已完成)：

| 验证项 | 结果 |
| --- | --- |
| 独立单元测试程序 | 242/242 通过，较 Task 21 新增 45 项 |
| lit 回归 | 48/48 通过，新增 12 份 |
| 原有 CTest | 4/4 通过 |
| Python 导入测试 | 5/5 测试方法通过，包含多个 dtype/transpose 子场景 |
| 生产接口与链接隔离 | 源码、生成头文件和二进制检查通过 |
| 独立代码审查 | 修复后复审通过，无未解决发现 |

其中，CPU 数值模型检查了 Reduce 的 30 组场景，覆盖三种 kind、clear true/false、有限精确值、正负零和 NaN。最终审查还补上了未知 region 中错误借用支配证据、以及遗漏块参数旧布局编码的负例。

本阶段仍有明确边界：

1. Local 提升仅支持规定的静态 identity allocation、整块操作和受支持 cast；不支持 Local 参数、subview、动态形状、scalar 访问、逃逸或任意控制流。
2. Gemm 不隐式提升 f16/bf16 accumulator，不扩展 Task 20 的硬件支持集；Reduce 不支持 mul 迁移。
3. 每连通分量最多 8 个变量、每个候选域最多 4 个候选的预算不变。
4. 完整 SS MMA→Reduce→tile_store 需要 9 个变量：归一化能够成功，但完整 pipeline 仍超限拒绝并回滚。预算内的 SS/RS MMA→store 和 Fill→Reduce→store 已通过，不能把它们拼称为完整链成功。
5. 本次没有完成 GPU 指令 lowering、GPU 数值测试或性能调优，也没有新增 TileLang/Triton 上游运行对比。

因此，Task 22 的已确认受限契约完成，不等于整个 M4 的所有成功验收项都已完成。

## 10. 汇报时可以这样总结

> Task 19–21 建立了新体系里的操作布局规则，Task 22 则补齐旧程序进入这个体系的转换入口。它把可证明安全的 Local Buffer 生命周期转换成 Tensor 值，明确初始化、累加和循环传值；用持久根契约保住优化后仍需核验的存储依据；通过事务 pipeline 统一完成转换、推断和复验，失败时保留原程序。同时，旧推断退出生产，只作为测试参考保留。当前受限实现和回归已完成，但大图求解预算以及 GPU 执行验收仍属于后续工作。

实现与证据入口：

- [Task 22 实现计划及验收记录](../layout_inference_implementation_plan.md#task-22-实现-legacy-normalization-并退役旧生产路径)
- [生命周期归一化实现](../../lib/Dialect/Frisk/Transforms/LegacyFragmentNormalization.cpp)
- [事务 pipeline](../../lib/Dialect/Frisk/Transforms/PassPipelines.cpp)
- [根存储契约分析](../../lib/Dialect/Frisk/Analysis/StorageRootContracts.cpp)
- [端到端与回滚测试](../../unittests/Dialect/Frisk/Layout/LayoutPipelineTest.cpp)
- [Python 导入回归](../../test/python/legacy_normalization_test.py)
