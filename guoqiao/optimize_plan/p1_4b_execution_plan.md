# P1.4b Complete Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development; implement one complete task with internal checkpoints, test-driven-development and independent task plus whole-source reviews.

**Goal:** Execute the original fixed-ABI Frisk Attention end-to-end on H100 with traceable checked lowering and full correctness/safety evidence.

**Architecture:** Preserve Frisk selection, maps and operation contracts; export checked explicit-layout IR through a process boundary to pinned Triton lowering. Use one shared backend, map adapter, runtime and artifact schema for all operations. Unsupported inputs fail atomically; input IR remains untouched.

**Tech Stack:** Existing Frisk LLVM/MLIR, Triton 3.4.0, CUDA 12.8, H100, Python runner and C++ MLIR passes.

The frozen `p1_4b_artifacts/frozen_numerics.json` is authoritative; legacy proposal wording quoted below does not permit changing its gate.

## Global Constraints

Only edit /home/baopeihua/frisk on H100; preserve pre-existing changes. No worktree, staging, commit or push. No P1.4c cp.async/TMA. Source baseline is p1_4b_artifacts/baseline.json, not HEAD alone.

Numerical gate is frozen before production implementation: finite f16-rounded inputs in [-1,1]; independent FP32 QK, stable softmax, RNE f16 P, FP32 PV reference; all O finite and abs(got-ref)<=2e-3+2e-3*abs(ref). No tolerance relaxation based on observed errors. exp/subnormal/reassociation settings recorded explicitly.

# GPU 执行闭环阶段任务清单

日期：2026-10-29。状态：按用户反馈将原9项合并为2个独立交付任务，GPU后端实现尚未开始。代码检查点为 `19d56e60aadd4864e902c33e3c04587c87d68dc4`，P1.4a已验收。

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


## Task 1: Complete fixed-ABI Attention GPU execution

**Files and responsibilities:**
- Existing lib/Conversion/FriskLayoutToGPU and its registration/CMake: checked atomic preflight and Frisk export. Add focused files here as required by the adapter boundary; preserve existing conversion experiment.
- benchmark/layout/execution/: one compiler, map adapter, runtime and validation entry point, with names established by the initial probe and recorded below before implementation.
- test/Conversion/ and unittests/: rejection, coordinate oracle, contracts and pass tests matching existing repository conventions.
- guoqiao/optimize_plan/p1_4b_artifacts/: baseline, probe, frozen config, RED/GREEN logs, selected IR, manifests, binaries, raw timings, reviews and progress ledger.
- guoqiao/inference_detail/p1_4b_implementation_report.md: sole implementation and acceptance report.

**Interfaces:** Original fixed-ABI MLIR is the authoritative input. Export consumes selected/verified Frisk operations and contracts and produces an explicit-layout TTGIR plus schema-versioned manifest. Compiler consumes this pair and produces LLVM/PTX/cubin and exact launch ABI; runner consumes only those recorded artifacts and produces raw validation/timing/safety records. Every unsupported mapping or contract returns nonzero without publishing an executable artifact.

- [ ] Inspect actual selected maps/contracts and Triton interfaces; save explicit-layout load/store probe and version/GPU UUID evidence. Probe must execute on H100 before selecting production bridge details.
- [ ] Freeze exact supported map domain, implementation file names and compiler/pass options in this plan. Write failing tests for map coordinate equivalence, two legal distributions, swizzle, replicas/unit axes and atomic refusal. Implement checked adapters until these pass.
- [ ] Write failing byte-offset transport/cast/packing tests before implementing transport. For offsets 1,2,3 assert exact input/output bytes and unchanged guards; confirm IR/PTX does not strengthen alignment1. Use real Shared footprint/swizzle and unique-writer ownership.
- [ ] Write failing SS/RS instruction-contract tests before implementing descriptor/accumulator/packing and synchronization. Assert actual geometry/maps and fence/commit/wait/proxy visibility; save GPU isolated arithmetic results and PTX instruction evidence.
- [ ] Write failing reduction/broadcast/math tests before implementing canonical_fiber_tree_v1 with first_owner distribution; confirm input-defined tree order, same-thread broadcast slots, and no uninitialized empty reads.
- [ ] Integrate original Attention operation walk in the same backend. Run frozen complete numerical suite for seeds0-9 and zero QK, constant, signed coordinate, one-hot V and cancellation V; assert full output, no sampling. Save maximum errors and coordinates.
- [ ] Run sentinels/guards/repeats, offsets1/2/3 and memcheck/racecheck/synccheck/initcheck or explicit initialization evidence; document tool coverage boundary for WGMMA.
- [ ] Emit replay command, source/input/selected-layout/contract/TTGIR/PTX/cubin/tool/config hashes, launch ABI, physical registers/shared/spill, inference/full compile timings and raw GPU samples.
- [ ] Run new C++ tests both Debug and Release and final full Release once plus impacted gates. Record exact commands/counts and logs; do not repeat already passing unchanged checks without cause.
- [ ] Produce review package against baseline including untracked source. Independent task reviewer gives spec and code-quality verdicts, then strongest model whole-source/data/document review; fix all material findings and rerun covering checks.
- [ ] Update single report, close all launched jobs, verify pre-existing changes preserved. Keep server working tree as requested, without integration menu or commit.

## Progress ledger

Final state: P1.4b complete. Independent task review and strongest whole-source re-review approved; no open Critical/Important findings. Current artifact: post-review-execution/run-zopay3ho. All launched jobs ended; working tree retained without stage/commit/push.

The checklist above preserves the original procedural plan. Strict test-before-implementation ordering was not achieved for every initial feature; this historical process deviation is disclosed in the sole report, not retroactively marked fulfilled. Functional acceptance gates and all review-driven RED/GREEN fixes passed.


### Selected process interface (2026-10-29)

`ExportExecution.cpp` exports actual SSA operations, checked finite execution/Storage oracles and C++ canonical reduction dependency proofs to schema1 JSON. `benchmark/layout/execution/backend.py` scalarizes actual thread/register maps and emits generic operation-driven PTX fragments through side-effect scalar TTGIR inline assembly. This keeps Frisk descriptors/phase/packing/trees exact and avoids native warp_group_dot descriptor reinterpretation. Triton lowers the scalar TTGIR ABI and inline assembly to LLVM/PTX/cubin; it does not select layouts. Static shared roots use exported footprints and root alignments, not fixed allocation assumptions. Every actual clone/conversion must execute or reject explicitly.

`run_execution.py` is the single replay compiler/runtime. Math settings freeze before GPU numerical execution: mul.rn/sub.rn/div.rn; exp=ex2.approx.f32(mul.rn.f32(x, float32(log2(e)))); no FTZ instruction modifiers; cvt.rn.f16.f32 P; explicit proof tree arithmetic with no fusion/reassociation. CPU independent FP32-loop reference, f16-rounded P; no torch matmul/TF32 reference.

Progress: explicit linear-layout byte probe passed offsets1/2/3, 3 repeats, guards. Exporter RED and GREEN saved; initial exporter integration 2/2 passes. Actual full-graph scalar TTGIR compile is being tested; no complete acceptance yet.

Progress update: original fixed ABI complete GPU graph numerics passed135/135 fresh launches with per-launch sentinels and unchanged input bytes/guards. Two bound legal distributions plusreplica/unitaxis transport/cast18/18 offset records bit-exact; isolated packing2048words bit-exact. New C++testsDebug2/2,Release2/2; completeReleasegtest577/577;ctest10/10;lit73/73. Four sanitizers previously0; final-cubin matching sanitizer replay underway. Source frozen for independent review; acceptance remains pending review. Sole report: guoqiao/inference_detail/p1_4b_implementation_report.md.

Final implementation evidence: current-artifact.json identifies latest135-launch artifact; transport54individually checked launches; exact-current4sanitizers0; C++Debug/Release2/2, fullReleasegtest577/577,ctest10/10,lit73/73. Task review2domain issues andwhole-source broadcast issue corrected withRED/GREEN. Final report consolidated; controller records final independent review verdict. No jobs remain.

Acceptance ledger: complete on2026-10-29; task-review.md approved after2 domain fixes, whole-source-review.md approved after broadcast and transport-validation fixes. Final135 Attention launches,54 transport/cast launches, packing3offsets,4exact-cubin sanitizers; fullRelease577/577(67suites),ctest10/10,lit73/73;newC++2/2both configurations. No commits.
