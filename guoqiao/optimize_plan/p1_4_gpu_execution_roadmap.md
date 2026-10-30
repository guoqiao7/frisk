> 2026-10-29 P1.4c已验收：同一backend/runner的cp.async与TMA完整Attention各135次通过，基础弱对齐135次回归通过；54次bitexact运输、12组最终cubin四种Sanitizer及CPU回归通过。两层独立复审Approved，root身份/参考核验通过。详见[报告](../inference_detail/p1_4c_implementation_report.md)。

> 2026-10-29 P1.4b已验收：实际Frisk操作/映射/契约受检导出，逐操作生成低层标量TTGIR及PTX发射，复用Triton 3.4.0编译与运行基础；不由Triton重新选择原生Tensor布局。完整Attention135次、运输/cast54次逐次检查、最终cubin四种Sanitizer及CPU回归通过。详见[报告](../inference_detail/p1_4b_implementation_report.md)。

# GPU 执行闭环阶段任务清单

日期：2026-10-29。状态：按用户反馈保留2个独立交付任务；P1.4b已完整验收，P1.4c已完整验收。代码检查点为 `19d56e60aadd4864e902c33e3c04587c87d68dc4`，P1.4a已验收。

任务按完整可执行能力划分。桥接验证、搬运、MMA、Softmax和运行器均为任务内部实施步骤及测试，不再单独立项、发布计划或形成交付报告。

## 整体路线和维护方式

Frisk布局推断/选择/证明/物化 → 受检显式布局TTGIR桥接 → 固定版本Triton后端 → LLVM/PTX/cubin → H100运行核验。

- Frisk保有布局、转换/重计算和指令geometry决策；选择性运行Triton后端pass，核验地址、元素归属和指令契约，记录后端新增通信/scratch及必要执行变化。
- GF(2)共同基础不等于接口完全兼容。桥接必须处理维序/bit宽、硬件拓扑、replication、单位轴，以及Storage映射方向、byte/bit单位、root offset、footprint和alignment。Affine/Product只接入已证明可表达的子集。
- 版本兼容和最小GPU探针在任务一开头完成，结论直接驱动同一任务的后续实现；不能表达的部分记录并作局部扩展或明确拒绝，不静默重选Frisk布局。
- 保持统一的后端入口、映射适配层、运行器和artifact格式。SS/RS、Reduce和Copy复用相同地址/寄存器语义；按代码职责组织模块，不按里程碑复制框架或为每个算子建立独立工具链。
- 每个独立任务只维护一份执行计划、一份实施验收报告及共享证据索引。内部检查点持续更新同一清单，通过切片测试后继续整合，不逐项发起新任务。复审覆盖实际变更，阶段最终做整体复审。
- 代码和证据仅在H100 `/home/baopeihua/frisk`保存，保留已有内容。当前提交是检查点，不自动提交或推送后续修改。
- 不把Unknown当Proven。先preflight支持域，在独立模块中转换后发布；不支持的操作/映射/契约明确拒绝，失败保留输入。

## 任务一：P1.4b 基础完整Attention执行闭环

**目标：** 原始Frisk Attention经过推断、受检桥接和Triton后端，生成可加载cubin，在H100上正确执行。完成时交付一个完整、可复核的执行入口。

**固定目标：** `test/Transforms/attention-layout-fixed-abi.mlir`，单CTA、128线程；Q/K/V为行主序 `[64,16]f16`，O为 `[64,16]f32`，Global根alignment=1。

算法为SS QKᵀ → scale 0.25 → max/sub/exp/sum/div稳定Softmax → P以RNE转f16 → RS PV → Store。保持实际Storage/Distributed map、geometry、descriptor、packing、转换与clone语义。

### 内部实施步骤（不另立任务）

1. **确定接入并建立统一骨架。** 固定Triton及其LLVM、CUDA/PTX/ptxas、GPU UUID和pass集合；先完成最小显式布局load/store探针，检查固定Attention所需的表示能力。优先独立进程/文本IR连接现有Frisk与Triton，避免直接混链LLVM。建立preflight、导出入口、runner和版本化manifest。若需局部后端扩展，在同一执行计划中登记和实现。
2. **连续补齐整条执行链。** 实现Storage/Distributed受检转换，接入Copy/Load/Store/Convert、SS MMA、Reduce/Broadcast/math和RS packing/MMA。以搬运、SS、Softmax、RS切片测试定位问题，通过后立即整合到同一Attention路径，不将切片测试当作阶段交付。
3. **完成真实GPU整链验证和资源记录。** runner加载本次IR生成的artifact，验证数值、安全、弱对齐和重复执行；记录完整编译/GPU耗时及物理资源，完成最终回归、独立整体复审和统一报告。

### 必须落实的实现语义

- 从真实Storage map生成Global/Shared地址，Shared按物理footprint和对齐分配，保留swizzle。唯一writer、读取副本、scratch生命周期与同步均受检。
- Global alignment=1必须兑现为安全的弱对齐操作或byte运输/组装，不借分配器实际对齐加强ABI；测试偏移1/2/3字节及前后保护区。
- SS/RS消费选定atom/grid/repeats、descriptor与accumulator编码，处理初始化、转置、K分片和累加；RS严格核验.f16x2的元素位置、低高半字及寄存器组。
- Reduce消费canonical_fiber_tree_v1、first_owner及输出副本分发；高层tt.reduce不自动视为同一归约树，不匹配时显式展开或补降低。Broadcast读取已证明的同线程源槽位；tensor.empty不产生未初始化读取。
- 冻结exp、舍入、subnormal、fast-math/reassociation口径。WGMMA正确处理Shared对异步代理可见性、fence/commit/wait及warpgroup一致执行。首版串行完成，不做搬运计算重叠。
- TTGIR可包含其合法tensor中间表示；最终降低产物不得遗留Frisk op、未解析tensor、unrealized cast或替代实际输入的固定手写Attention调用。

### 统一验收

- 从原始输入至本次cubin，再到GPU输出的一条命令可重放，保存source/input/selected-layout/contract/TTGIR/PTX/cubin/tool/config哈希、命令和launch ABI。
- 映射使用小域坐标oracle/代数等价检查，覆盖两种不同合法分布、swizzle、复制及单位轴；不支持的映射和错误契约原子拒绝。至少有受支持fixture可运行两种合法方案，供后续消融使用，不强称固定Attention本身必然有两个方案。
- 搬运/转换和孤立packing/cast bit-exact。完整Attention至少seeds0–9、零Q/K、常量、带符号坐标、one-hot V及抵消V。
- 实现前冻结数值门槛。沿用既有提案：输入先舍入f16且有限[-1,1]；独立参考包含FP32 QK、稳定Softmax、P的RNE f16转换、FP32 PV；建议O全部有限且abs(got-ref)<=2e-3+2e-3*abs(ref)。门槛仍为提案，不能观测失败后自行放宽；记录最大误差及坐标。
- 输出sentinel、保护区和重复运行通过；运行memcheck/racecheck/synccheck及适用initcheck/初始化检查，说明工具对异步执行的覆盖边界。
- 新C++单测双配置，GPU切片和整链测试及最终完整CPU回归通过，所有重要复审问题关闭。相同源码的通过检查不无理由重复。
- 保存layout inference、完整编译、GPU耗时原始样本，记录真实寄存器/Shared/spill，与已有逻辑资源诊断区分。功能通过不等同于布局性能获益。

**完成定义：** 现有固定ABI完整Attention从Frisk输入在H100正确、安全、可追溯地运行；弱对齐路径确实通过，完整操作和契约被消费。仅完成桥接或单个切片不能将任务一标为完成。

## 任务二：P1.4c 异步搬运完整闭环

**目标：** 在任务一的同一后端、runner和测试框架上，增加cp.async与TMA完整Attention变体。它们属于同一独立任务，在同一报告中分别记录验收结果。

### 内部实施步骤（不另立任务）

1. **cp.async路径。** 使用显式强对齐独立ABI，兑现已选width、issuer、commit/wait及CTA可见性；保留基础弱对齐版本和拒绝/回退负例。先单次无重叠执行，完成搬运切片后直接接入完整Attention验证。
2. **TMA路径。** 在同一框架补tensor descriptor、host setup、elected issuer、mbarrier初始化/完成字节数/等待及可见性。记录host ABI/setup成本，覆盖对齐/offset/descriptor负例，完成一次性无重叠的完整Attention验证。
3. **统一回归和报告。** 两条路径分别核对实际指令、数值、安全、资源与耗时，并对基础路径回归；整体复审和一个阶段报告关闭任务。

**完成定义：** cp.async和TMA两条路径都在明确的强对齐ABI下通过完整Attention数值与安全验收，且基础弱对齐版本保持有效。只通过其中一条或只通过搬运切片时如实记录进度，不将任务二整体标为完成。

多阶段缓冲复用、循环重叠、warp specialization、cluster multicast和TMA输出Store不纳入本次最小执行闭环。后续确需流水时再规划生命周期、barrier phase、阶段环绕和尾轮收尾验证。

## 与后续路线的衔接

完成基础及所需异步路径后，进入联合选择及转换消融 → 成本/资源校准 → 搜索开销优化 → 冻结公平比较。本阶段不新增成本模型或搜索预算优化任务。

共同后端用于隔离Frisk选择贡献；原生Triton/TileLang完整调优另做端到端比较。单CTA小tile的功能与耗时不推导大尺寸吞吐优势。

## 当前交付状态

- [x] P1.4b 基础完整Attention执行闭环
- [x] P1.4c 异步搬运完整闭环

原P1.4b.0–b.6及P1.4c.1/c.2编号停止作为独立任务使用。其必要验证已合并到以上内部步骤。P1.4b和P1.4c均已完整验收；后续联合选择/转换消融等按新任务推进，不因本次验收自动实施。
