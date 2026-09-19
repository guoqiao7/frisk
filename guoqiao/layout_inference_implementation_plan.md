# Frisk Layout Inference System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Frisk 中实现一个面向 NVIDIA SM90/SM90a 的 MLIR-native 布局系统闭环，覆盖组合布局代数、Distributed/Storage IR、约束传播、有限候选求解、显式 conversion、正确性验证和性能守门。

**Architecture:** 采用“基础能力 + 纵向切片 + 逐类扩展”。Local/Register 使用带 `DistributedEncodingAttr` 的 RankedTensor SSA，Shared/Global 使用 MemRef 与 `StorageLayoutAttr` binding；Op 只收集约束，solver 只选择 assignment，materializer 在求解后统一改写 IR。现有 `LayoutAttr + DenseMap<Value, Attribute>` 通过 adapter 提供已知正确子集的迁移回归；不能作为新版规则的唯一真值来源，不能为差分一致保留已知错误。

**Tech Stack:** C++17、LLVM/MLIR ODS/TableGen、MLIR Pass/Dialect Conversion、Affine/Presburger、SCF、MemRef、Tensor、GPU/NVGPU/NVVM、LLVM ADT/APInt、CMake/Ninja、llvm-lit/FileCheck、CTest、CUDA/Nsight 性能工具。

> **执行状态（2026-09-19）：M0–M3 与 M4 Task 18–20 已完成。Task 21 已按确认契约完成受限 Reduce 实现、独立复审与 197 unit / 36 lit / 4 CTest 回归；完整 MMA→Reduce→tile_store 的成功验收仍因 9 变量超过保留的 8 变量上限而未完成，详见 §21.8，不将测试全通过等同于原契约全部验收。Task 22 尚未实施。Task 19/20 已提交到源目录 main 的 `c028e82`，未 push；Task 21 未提交、未推送。**

> **设计更新（2026-09-19）：Task 20 的 SS＋RS 具体契约 v1 已由用户确认；§20.1–20.11 是契约，§20.12 是执行计划，§20.13 记录实际实现、验证证据和限制。通用 encoding＋操作级指令 binding 的适配同时写入设计总文档 §19。**

> **集成与后续（2026-09-19）：用户要求先提交既有修改，再进入 Task 21。Task 19/20 的代码、测试、维护文档和汇报已提交到本地 main：`c028e82`，提交前重新验证 165 unit / 34 lit / 4 CTest 全部通过；未 push。以上“未提交”描述是当时的实施记录，以本条更新为准。Task 21 的 §21.1–21.7 已由用户确认并实施；实际证据与尚未解决的完整链验收冲突见 §21.8。**

> **工作区说明：M3 从 `feature/m3-distributed-layout`（`be71d91`，实现验收基线 `642c30b`）于 2026-09-11 合入 `main`，合并提交 `b120d06400a14a703a44dac1a37a0b38d8110935` 已在此前按用户要求推送至 `origin`（guoqiao7/frisk）。Task 18 历史上使用 `.worktrees/m4-task18`，集成记录见 §18.10。本轮 Task 19 遵照用户新要求直接在主源码 main 中实现，不使用 worktree，不改动独立的 `guoqiao/inference_detail/README.md`，不自动 commit/push。27 lit、74 unit、4 CTest 是 M3 基线；Task 18/19 Gate 分别见 §18.9/§19.9。**

## Global Constraints

- 首个目标仅为 NVIDIA SM90/SM90a；不得在通用 solver 中散布 target 字符串判断。
- 完整 WGMMA/TMA/mbarrier 指令 lowering 不在本计划范围；本计划实现布局契约、静态成本和 conversion 微基准所需的最小 lowering adapter。
- 本轮 TileLang 审计及 Task 28 拟建立的新语义参考固定为 `5e149e31674658f94779c7d0c6039549a1853123`；旧 `6623b12d232b343648a5ba99992e3e6f0d6376d2` 保留作历史迁移参考。上游升级不自动改变 Frisk 契约；每次须记录新旧行为与采用理由，不得链接或 vendor TVM/TIRX 类型。
- Local/Register 的最终公共表示必须是 RankedTensor SSA；不得继续扩展 local-memory MemRef 作为长期寄存器模型。
- Shared/Global 保持 MemRef；XOR storage mapping 通过 `frisk.layout_view`/binding 表达，不强塞进 MemRef semi-affine layout。
- `frisk.convert_layout` 和 rematerialization 是普通候选，不以“零转换方案无解”为启用条件。
- Hard constraint 不得被代价模型违反；Affine/Presburger 的 `unknown` 不得当作证明成功。
- 所有 worklist、constraint、candidate 和 tie-break 必须使用稳定 ID，结果不得依赖 DenseMap/指针遍历顺序。
- 用户 IR 错误、unsupported shape 和求解冲突必须返回带位置的诊断；不得用 assertion 处理。
- 每项公共语义或架构边界发生变化时，必须在同一任务中同步维护 `guoqiao/layout_inference_design.md`。
- 长期同时维护本计划和 `guoqiao/layout_inference_strategy_comparison.md`：计划记完成度/设计/验收，比较记版本证据/取舍/差异假设。Task 18 确认实施后同步设计总文档和属性/API 注释；公共地址与对齐契约见 §18.2–18.3。
- 现有用户修改属于用户；不得覆盖 `build.sh` 或其他与当前任务无关的工作区变更。
- 正确性零容忍：race、OOB、invalid ownership、未解析 LayoutVar 和 nondeterministic IR 必须为 0。
- 性能守门：受支持 kernel 固定环境中位运行时间回退不超过 3%；常规 component candidate 上限 256，首版 beam width 32，布局推断不超过完整编译时间的 10%。

---

## 0. 执行规则

### 0.1 每个任务的固定循环

```text
确认前置任务和接口
  -> 写一个会失败的最小测试
  -> 运行并确认失败原因正确
  -> 写满足该测试的最小实现
  -> 运行局部测试
  -> 运行本里程碑回归
  -> 检查设计文档是否漂移
  -> 提交一个可独立评审的 commit
```

不得把多个未验证任务合并后一次测试。一个任务的退出条件未满足时，不进入依赖它的任务。

### 0.2 基础构建命令

首次配置或 CMake 结构变化后：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
```

日常构建：

```bash
cmake --build build --target FriskIR FriskTransforms frisk-opt check-frisk --parallel 32
```

现有基线：

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

### 0.3 文档维护检查

每个任务提交前运行：

```bash
git diff --check
rg -n "LayoutAttr|DistributedEncodingAttr|StorageLayoutAttr|convert_layout|CostVector" \
  guoqiao/layout_inference_design.md \
  guoqiao/layout_inference_implementation_plan.md \
  guoqiao/layout_inference_strategy_comparison.md
```

如果实现改变了属性字段、映射方向、hard/soft 边界、候选优先级、conversion 策略、里程碑范围或性能阈值，必须先更新设计文档再提交。

每项布局任务还必须留下以下审计记录：

1. Frisk 与上游固定 SHA、审计日期、源码符号/永久链接；区分旧版已有和本区间新增。
2. 旧规则的问题、当前 Frisk 的适配约束，以及采用/调整/拒绝的决定；源版本升级不能只更换 SHA。
3. 已实现、设计待确认、未来目标分别标记；测试源码存在不等于本轮执行通过。
4. 新旧行为的正例、反例、失败策略与实际验证命令；旧版错误用独立语义不变量修正，不能强制复制。
5. 差异化写成可检验假设，注明当前证据和否证条件；不把通用 alias/GF(2)/while/worklist 当作首创。

审计历史与精确上游证据集中在[比较文档 §0、§13、§15](./layout_inference_strategy_comparison.md)；下方 M4 记录审计缺口、已确认设计和实现验收。Task 28 corpus 尚未生成，本轮也未运行 TileLang/Triton 测试或 GPU 性能实验。

## 1. 里程碑与依赖

| 里程碑              | 任务   | 可独立验收的结果                                                                  |
| ------------------- | ------ | --------------------------------------------------------------------------------- |
| M0 基础设施         | 1–4   | 原生测试可由 CTest/lit 运行；`frisk-opt` 可加载 dialect/pass；MemoryEffect 正确 |
| M1 布局代数         | 5–9   | Affine/BitLinear/Product 与 Distributed/Storage Attr 可解析、组合和验证           |
| M2 Storage 切片     | 10–13 | alloc/view/copy 完成 constraint → propagation → binding → verifier             |
| M3 Distributed 切片 | 14–17 | Tensor 多 consumer 可选择共同布局或物化`frisk.convert_layout`                   |
| M4 完整传播迁移     | 18–22 | alias/region/Copy/Fill/Gemm/Reduce 进入新 solver；旧生产路径退役                  |
| M5 候选与 SM90      | 23–26 | 全局候选选择、SM90 契约、conversion 优化和 controlled relaxation 完整             |
| M6 硬化验收         | 27–29 | property/differential/performance/compile-time 守门通过，文档与实现一致           |

## 2. 目标代码结构

```text
include/Dialect/Frisk/IR/
  FriskLayoutAttrs.td
  FriskLayoutAttrInterfaces.td
  FriskLayoutOpInterfaces.td
  FriskLayoutInterfaces.h
  FriskLayoutOps.td

include/Dialect/Frisk/Analysis/
  LayoutCommon.h
  LayoutAlgebra.h
  LayoutConstraint.h
  LayoutSolver.h
  LayoutVerifier.h
  LayoutTarget.h
  LegacyLayoutAdapter.h
  LayoutStatistics.h

include/Dialect/Frisk/Target/SM90/
  SM90LayoutTarget.h

include/Dialect/Frisk/Transforms/
  LayoutTypeConverter.h
  PassPipelines.h
  Passes.h
  Passes.td

lib/Dialect/Frisk/IR/
  FriskLayoutAttrs.cpp
  FriskLayoutInterfaces.cpp
  FriskLayoutOps.cpp

lib/Dialect/Frisk/Analysis/
  LayoutAlgebra.cpp
  LayoutConstraint.cpp
  LayoutPropagation.cpp
  LayoutCandidateSolver.cpp
  LayoutCostModel.cpp
  LayoutVerifier.cpp
  LegacyLayoutAdapter.cpp
  LayoutStatistics.cpp

lib/Dialect/Frisk/Target/SM90/
  SM90LayoutTarget.cpp
  SM90CopyConstraints.cpp
  SM90GemmConstraints.cpp
  SM90ReduceConstraints.cpp
  SM90CostModel.cpp

lib/Dialect/Frisk/Transforms/
  NormalizeLayoutIR.cpp
  LayoutInfer.cpp
  LayoutTypeConverter.cpp
  MaterializeLayouts.cpp
  OptimizeLayoutConversions.cpp
  PassPipelines.cpp

lib/Conversion/FriskLayoutToGPU/
  CMakeLists.txt
  TestLayoutConversionLowering.cpp

tools/frisk-opt/
  CMakeLists.txt
  frisk-opt.cpp

test/Dialect/Frisk/layout/
test/Transforms/
unittests/Dialect/Frisk/Layout/
benchmark/layout/conversion/
benchmark/layout/solver/
```

文件按职责拆分。`FriskOps.cpp` 中现有 GEMM/shared layout 构造只在迁移任务中移动，不提前做无关重构。

---

## M0：基础设施与现状冻结

### Task 1: 将现有原生测试纳入 CTest 并冻结基线

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `test_pass/CMakeLists.txt`
- Create: `guoqiao/layout_baseline.md`
- Verify: `test_pass/attr_test.cpp`
- Verify: `test_pass/reduce_layout_test.cpp`

**Interfaces:**

- Consumes: 当前 `FriskIR`、`frisk_attr_test`、`frisk_reduce_layout_test`。
- Produces: `ctest --test-dir build` 可发现并运行两个 legacy baseline；后续所有任务使用同一入口做回归。

- [X] **Step 1: 记录当前测试发现状态**

Run:

```bash
ctest --test-dir build -N
```

Expected: 输出 `Total Tests: 0`，证明当前 CMake 尚未注册测试。

- [X] **Step 2: 运行两个现有可执行测试并保存基线结论**

Run:

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

Expected: 两个进程退出码均为 0；将覆盖的 target/shape/reduce case 和当前已知限制写入 `guoqiao/layout_baseline.md`，不得复制大段日志。

- [X] **Step 3: 注册 CTest**

在顶层 `CMakeLists.txt` 的 `project` 命令后加入：

```cmake
include(CTest)
enable_testing()
```

在 `test_pass/CMakeLists.txt` 两个 executable 定义后加入：

```cmake
add_test(NAME FriskAttrTest COMMAND frisk_attr_test)
add_test(NAME FriskReduceLayoutTest COMMAND frisk_reduce_layout_test)
```

- [X] **Step 4: 重新配置并验证测试发现**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target frisk_attr_test frisk_reduce_layout_test --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 2`。

- [X] **Step 5: 提交基线**

```bash
git add CMakeLists.txt test_pass/CMakeLists.txt guoqiao/layout_baseline.md
git commit -m "test: register legacy layout baselines"
```

### Task 2: 编译并注册真正的布局推断 pass

**Files:**

- Modify: `include/Dialect/Frisk/Transforms/Passes.h:1-18`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td:1-17`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt:1-13`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp:1-25`
- Create: `test_pass/layout_pass_test.cpp`
- Modify: `test_pass/CMakeLists.txt`

**Interfaces:**

- Consumes: MLIR `OperationPass<FunctionOpInterface>`、生成的 `impl::FriskInferLayoutsBase`。
- Produces: `std::unique_ptr<Pass> createFriskInferLayoutsPass()` 和注册参数 `frisk-infer-layouts`。

- [X] **Step 1: 写会失败的 pass 构造测试**

创建 `test_pass/layout_pass_test.cpp`，核心断言为：

```cpp
#include "Dialect/Frisk/Transforms/Passes.h"

int main() {
  std::unique_ptr<mlir::Pass> pass = mlir::frisk::createFriskInferLayoutsPass();
  if (!pass || pass->getArgument() != "frisk-infer-layouts")
    return 1;
  return 0;
}
```

Run:

```bash
cmake --build build --target frisk_layout_pass_test --parallel 32
```

Expected: FAIL，原因是 target/`createFriskInferLayoutsPass` 尚不存在。

- [X] **Step 2: 统一 pass 名称和构造器**

将 `Passes.td` 定义改为：

```tablegen
def FriskInferLayouts : InterfacePass<"frisk-infer-layouts", "FunctionOpInterface"> {
  let summary = "Infer and materialize Frisk layouts";
  // MLIR 21 pass TableGen expects a call expression, including `()`.
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
}
```

删除 `Passes.h` 中旧的 `createLayoutInferPass()` 手写声明，改为：

```cpp
std::unique_ptr<Pass> createFriskInferLayoutsPass();
```

- [X] **Step 3: 实现最小可运行 pass**

`LayoutInfer.cpp` 使用生成基类：

```cpp
namespace {
class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  void runOnOperation() override {
    getOperation()->setAttr("frisk.layout_inference_ran",
                            UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<Pass> createFriskInferLayoutsPass() {
  return std::make_unique<FriskInferLayoutsPass>();
}
```

将 `LayoutInfer.cpp` 加入 `FriskTransforms` source list，并链接 `MLIRFuncDialect`、`MLIRIR`、`MLIRPass`、`FriskIR`。

- [X] **Step 4: 注册并运行测试**

在 `test_pass/CMakeLists.txt` 新增 `frisk_layout_pass_test`，链接 `FriskTransforms`，并注册 CTest。

Run:

```bash
cmake --build build --target frisk_layout_pass_test --parallel 32
ctest --test-dir build -R FriskLayoutPassTest --output-on-failure
```

Expected: `FriskLayoutPassTest` PASS。

- [X] **Step 5: 提交 pass 驱动**

```bash
git add include/Dialect/Frisk/Transforms lib/Dialect/Frisk/Transforms test_pass
git commit -m "feat: add executable layout inference pass"
```

### Task 3: 增加 `frisk-opt` 与 MLIR lit 测试入口

**Files:**

- Modify: `CMakeLists.txt`
- Create: `tools/CMakeLists.txt`
- Create: `tools/frisk-opt/CMakeLists.txt`
- Create: `tools/frisk-opt/frisk-opt.cpp`
- Create: `test/CMakeLists.txt`
- Create: `test/lit.cfg.py`
- Create: `test/lit.site.cfg.py.in`
- Create: `test/Transforms/infer-layouts-smoke.mlir`

**Interfaces:**

- Consumes: `FriskDialect`、`registerFriskPasses()`、`MlirOptMain`。
- Produces: `build/bin/frisk-opt` 和 `check-frisk` target。

- [X] **Step 1: 写命令行 smoke test**

创建：

```mlir
// RUN: frisk-opt %s -pass-pipeline='builtin.module(func.func(frisk-infer-layouts))' | FileCheck %s

module {
  func.func @smoke() {
    return
  }
}

// CHECK: frisk.layout_inference_ran
```

Run:

```bash
cmake --build build --target check-frisk --parallel 32
```

Expected: FAIL，原因是 `check-frisk`/`frisk-opt` 尚不存在。

- [X] **Step 2: 实现 opt driver**

`tools/frisk-opt/frisk-opt.cpp`：

```cpp
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect>();
  mlir::registerAllPasses();
  mlir::frisk::registerFriskPasses();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Frisk optimizer\n", registry));
}
```

对应 CMake 必须同时链接 `registerAllDialects()`/`registerAllPasses()` 引用的
MLIR 静态库集合；仅链接 `MLIROptLib` 不会传递所有注册实现：

```cmake
add_llvm_executable(frisk-opt frisk-opt.cpp)
set_target_properties(frisk-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
llvm_update_compile_flags(frisk-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(frisk-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  FriskIR
  FriskTransforms
)
```

- [X] **Step 3: 配置 lit**

`test/lit.cfg.py` 必须配置：

```python
config.name = "FRISK"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.frisk_obj_root, "test")
llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    [ToolSubst("frisk-opt", unresolved="fatal")],
    [config.frisk_tools_dir, config.llvm_tools_dir],
)
```

`test/CMakeLists.txt` 使用 `configure_lit_site_cfg`，并按以下方式创建目标：

```cmake
# 顶层必须在 add_subdirectory(test) 前提供 Python3_EXECUTABLE，避免依赖
# LLVM 构建树中 llvm-lit 脚本自身可能不可移植的 shebang。
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_lit_testsuite(check-frisk "Running Frisk regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS frisk-opt
)
```

顶层增加 `add_subdirectory(tools)` 与 `add_subdirectory(test)`。

- [X] **Step 4: 验证 driver 和 lit**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target frisk-opt check-frisk --parallel 32
```

Expected: lit 报告 `1 passed`。

- [X] **Step 5: 提交测试入口**

```bash
git add CMakeLists.txt tools test
git commit -m "test: add frisk-opt and lit harness"
```

### Task 4: 修正 MemoryEffect 与容器 Op 副作用传播

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Create: `test_pass/memory_effect_test.cpp`
- Modify: `test_pass/CMakeLists.txt`
- Modify: `guoqiao/layout_inference_design.md` only if the implemented effect model differs from Section 8.2

**Interfaces:**

- Consumes: MLIR `MemoryEffectOpInterface`、`RecursiveMemoryEffects`。
- Produces: Alloc/Copy/Fill/Gemm/Reduce 的准确 Allocate/Read/Write effect；Kernel/Parallel/Block/For 递归暴露 region effects。

- [X] **Step 1: 写失败的 effect 测试**

测试构造 Copy、Fill、Gemm、Reduce、Alloc，并断言：

```cpp
if (mlir::isMemoryEffectFree(copy) || mlir::isMemoryEffectFree(gemm) ||
    mlir::isMemoryEffectFree(reduce))
  return 1;
if (!mlir::hasEffect<mlir::MemoryEffects::Write>(fill) ||
    !mlir::hasEffect<mlir::MemoryEffects::Allocate>(alloc))
  return 1;
```

Run:

```bash
cmake --build build --target frisk_memory_effect_test --parallel 32
```

Expected: FAIL，因为现有 ODS 把多个内存 Op 标记为 `Pure`。

- [X] **Step 2: 修正 ODS effects**

采用以下边界：

```tablegen
// 容器
Frisk_Op<"kernel", [RecursiveMemoryEffects,
                     ImplicitFriskTerminator]>
Frisk_Op<"parallel", [RecursiveMemoryEffects,
                       HasParent<"KernelOp">,
                       ImplicitFriskTerminator]>
Frisk_Op<"block", [RecursiveMemoryEffects, ImplicitFriskTerminator]>
Frisk_Op<"for", [RecursiveMemoryEffects, ImplicitFriskTerminator]>

// 空终止符必须显式无副作用，否则容器递归 effect 查询会返回 unknown。
Frisk_Op<"end", [Pure, Terminator, ReturnLike]>

// 数据操作：删除 Pure，按 operand/result 声明
Arg<AnyMemRef, "", [MemRead]>:$src
Arg<AnyMemRef, "", [MemWrite]>:$dst
Res<AnyMemRef, "", [MemAlloc]>:$result
```

Gemm 的 A/B 为 `MemRead`，C 为 `MemRead, MemWrite`；Reduce 的 src 为 `MemRead`、dst 为 `MemWrite`；Copy/FIll 使用现有 operand effect 并删除 `Pure`。

- [X] **Step 3: 生成、编译并运行 effect 测试**

Run:

```bash
cmake --build build --target FriskTableGen FriskIR frisk_memory_effect_test --parallel 32
ctest --test-dir build -R FriskMemoryEffectTest --output-on-failure
```

Expected: PASS。

- [X] **Step 4: 运行 M0 全量回归**

Run:

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target check-frisk --parallel 32
```

Expected: 所有 CTest 和 lit tests PASS。

- [X] **Step 5: 提交 MemoryEffect 修复**

```bash
git add include/Dialect/Frisk/IR/FriskOps.td test_pass guoqiao/layout_inference_design.md
git commit -m "fix: model Frisk memory effects"
```

### M0 Gate

必须同时满足：

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target check-frisk --parallel 32
build/bin/frisk-opt --help-list | rg "frisk-infer-layouts"
```

Expected: CTest/lit 全部通过，pass 出现在完整帮助列表中。MLIR 21 的普通
`--help` 不展开 pass 清单，因此这里必须使用 `--help-list`。未满足时不得进入 M1。

### M0 执行记录（2026-08-30）

状态：**完成**。

独立提交：

```text
29fdf5e test: register legacy layout baselines
487e4e8 feat: add executable layout inference pass
4c57a18 test: add frisk-opt and lit harness
7af9e04 fix: model Frisk memory effects
```

最终 Gate 结果：

```text
ctest --test-dir build --output-on-failure
  4/4 passed

cmake --build build --target check-frisk --parallel 32
  Total Discovered Tests: 1
  Passed: 1

build/bin/frisk-opt --help-list | rg "frisk-infer-layouts"
  --frisk-infer-layouts - Infer and materialize Frisk layouts
```

实施中确认并回写计划的工具链约束：

1. MLIR 21 pass TableGen 的 `constructor` 必须是带 `()` 的调用表达式。
2. 调用 `registerAllDialects()`/`registerAllPasses()` 的静态链接 driver 必须链接 MLIR dialect、conversion、extension 及核心 transform 库集合。
3. lit 配置前必须提供 `Python3_EXECUTABLE`；`frisk-opt` 必须显式输出到 `build/bin`。
4. RecursiveMemoryEffects 容器中的 `frisk.end` 必须显式 `Pure`，否则递归 effect 查询会返回 unknown。
5. MLIR 21 的 pass 注册验收使用 `frisk-opt --help-list`；普通 `--help` 不展示 pass 清单。

---

## M1：组合布局代数与新 Attr

### Task 5: 建立布局公共类型和生成式 Interface

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutCommon.h`
- Create: `include/Dialect/Frisk/IR/FriskLayoutAttrInterfaces.td`
- Create: `include/Dialect/Frisk/IR/FriskLayoutOpInterfaces.td`
- Create: `include/Dialect/Frisk/IR/FriskLayoutInterfaces.h`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutInterfaces.cpp`
- Modify: `include/Dialect/Frisk/IR/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutInterfaceTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/CMakeLists.txt`
- Create: `unittests/CMakeLists.txt`
- Modify: `CMakeLists.txt`

**Interfaces:**

- Produces:

```cpp
enum class LayoutKind { Distributed, Storage, Instruction };
enum class ProofStatus { Proven, Disproven, Unknown };

struct LayoutProof {
  ProofStatus status;
  SmallVector<int64_t> counterexample;
  std::string reason;
};

struct CostVector {
  uint64_t instructionPathAndWork;
  uint64_t memoryTransactions;
  uint64_t bankConflictDegree;
  uint64_t conversionBytesAndSync;
  uint64_t spillRiskAndRegisters;
  uint64_t sharedBytesAndOccupancy;
  uint64_t replication;
  uint64_t codeSize;
  uint64_t deterministicTieBreak;
};

class LayoutMapAttrInterface;
class LayoutEncodingAttrInterface;
class LayoutConstraintOpInterface;
class LayoutConstraintBuilder;
```

- [X] **Step 1: 写 Interface 生成失败测试**

测试包含新头文件并做静态检查：

```cpp
static_assert(std::is_enum_v<mlir::frisk::LayoutKind>);
static_assert(std::is_enum_v<mlir::frisk::ProofStatus>);
```

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
```

Expected: FAIL，原因是头文件和 unit test target 尚不存在。

- [X] **Step 2: 定义公共结果类型**

`LayoutCommon.h` 定义上述枚举，并增加：

```cpp
struct LayoutDimension {
  StringAttr name;
  int64_t extent;
};

using LayoutMapAttr = LayoutMapAttrInterface;
```

`extent == ShapedType::kDynamic` 只允许出现在 affine outer；BitLinear 的 bit width 必须静态。

- [X] **Step 3: 定义 Attr/Encoding 接口**

`FriskLayoutAttrInterfaces.td` 至少声明：

```tablegen
def LayoutMapAttrInterface : AttrInterface<"LayoutMapAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let methods = [
    InterfaceMethod<"Return canonical map", "::mlir::FailureOr<::mlir::Attribute>",
                    "canonicalizeMap", (ins)>,
    InterfaceMethod<"Verify named dimensions", "::mlir::LogicalResult",
                    "verifyMap", (ins "::mlir::Location":$loc)>
  ];
}

def LayoutEncodingAttrInterface : AttrInterface<"LayoutEncodingAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let methods = [
    InterfaceMethod<"Expand encoding", "::mlir::FailureOr<::mlir::Attribute>",
                    "getCanonicalMap", (ins "::mlir::ShapedType":$type)>,
    InterfaceMethod<"Verify encoding", "::mlir::LogicalResult",
                    "verifyForType", (ins "::mlir::ShapedType":$type,
                                           "::mlir::Location":$loc)>,
    InterfaceMethod<"Return encoding kind", "::mlir::frisk::LayoutKind",
                    "getKind", (ins)>
  ];
}
```

`FriskLayoutOpInterfaces.td` 声明 `collectLayoutConstraints(LayoutConstraintBuilder&)`；此时只生成声明，Task 11 再提供 builder 实现。

- [X] **Step 4: 接入 TableGen 和 unit test**

为 Attr interface 和 Op interface 分别生成 decl/def，避免两个 generator 写同一文件；`FriskIR` 增加 `FriskLayoutInterfaces.cpp`。Unit test 使用：

```cmake
add_unittest(FriskUnitTests FriskLayoutUnitTests
  LayoutInterfaceTest.cpp
)
target_link_libraries(FriskLayoutUnitTests PRIVATE FriskIR MLIRIR)
```

Frisk 是外部 MLIR 工程；若 LLVM 配置未导出 `llvm_gtest` target，`unittests/CMakeLists.txt`
必须先从 `${LLVM_MAIN_SRC_DIR}/../third-party/unittest` 引入与当前 LLVM 源码匹配的
gtest/gmock，再调用 `add_unittest`。不能依赖系统 gtest 或裸 `-lllvm_gtest`。

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
```

Expected: unit test PASS。

- [X] **Step 5: 提交接口骨架**

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk unittests CMakeLists.txt
git commit -m "feat: define layout interfaces and proof types"
```

### Task 6: 实现 GF(2) Matrix 核心

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutAlgebra.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutAlgebra.cpp`
- Create: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/CMakeLists.txt`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/GF2MatrixTest.cpp`

**Interfaces:**

- Produces:

```cpp
class GF2Matrix {
public:
  static FailureOr<GF2Matrix> get(unsigned rows, unsigned cols,
                                  ArrayRef<APInt> rowBits);
  unsigned getNumRows() const;
  unsigned getNumColumns() const;
  APInt apply(const APInt &input) const;
  GF2Matrix transpose() const;
  FailureOr<GF2Matrix> compose(const GF2Matrix &rhs) const;
  unsigned rank() const;
  SmallVector<APInt> kernelBasis() const;
  FailureOr<GF2Matrix> inverse() const;
  FailureOr<GF2Matrix> rightInverse() const;
  bool operator==(const GF2Matrix &rhs) const;
};
```

- [X] **Step 1: 写矩阵红灯测试**

至少覆盖 identity、XOR、rank-deficient、inverse 和 compose 顺序：

```cpp
TEST(GF2MatrixTest, ComposeUsesRhsThenLhs) {
  GF2Matrix distributed = makeMatrix({0b01, 0b11}, 2);
  GF2Matrix storage = makeMatrix({0b10, 0b11}, 2);
  auto composed = storage.compose(distributed);
  ASSERT_TRUE(succeeded(composed));
  EXPECT_EQ(composed->apply(APInt(2, 0b11)),
            storage.apply(distributed.apply(APInt(2, 0b11))));
}
```

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
```

Expected: FAIL，`GF2Matrix` 未定义。

- [X] **Step 2: 实现构造、apply、transpose 和 compose**

内部使用 `SmallVector<APInt> rows`。乘加规则固定为：

```cpp
bool bit = (rows[row] & input).popcount() & 1;
```

`lhs.compose(rhs)` 表示 `lhs(rhs(x))`；维度不匹配返回 failure，不截断 APInt。

- [X] **Step 3: 实现消元、kernel、inverse/right-inverse**

使用确定性 Gauss-Jordan：pivot 按 column 从小到大、row 从小到大选择。`inverse()` 仅方阵满秩成功；`rightInverse()` 仅输出空间被覆盖时成功。

- [X] **Step 4: 增加穷举 oracle 并验证**

对输入 bit 数不超过 8 的矩阵枚举所有输入，比较 `compose/apply`；对可逆矩阵验证 `A.inverse()(A(x)) == x`。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='GF2MatrixTest.*'
```

Expected: 所有 GF2Matrix tests PASS。

- [X] **Step 5: 提交 GF(2) 核心**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis unittests
git commit -m "feat: add deterministic GF2 matrix algebra"
```

### Task 7: 实现 `BitLinearLayoutMapAttr`

**Files:**

- Create: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `include/Dialect/Frisk/IR/FriskAttributes.td`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `test/Dialect/Frisk/layout/bit-linear-attr.mlir`
- Create: `unittests/Dialect/Frisk/Layout/BitLinearLayoutTest.cpp`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`

**Interfaces:**

- Produces `BitLinearLayoutMapAttr`，字段固定为：

```text
input_names       : ArrayAttr<StringAttr>
input_bit_widths  : DenseI64ArrayAttr
output_names      : ArrayAttr<StringAttr>
output_bit_widths : DenseI64ArrayAttr
matrix            : DenseIntElementsAttr<tensor<rows x cols x i1>>
```

- [X] **Step 1: 写 parse/verify 红灯测试**

测试包含一个合法 XOR map 和三个非法 map：matrix shape 错、重复 dimension name、bit width 为 0。

```mlir
// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s
module attributes {
  frisk.map = #frisk.bit_linear<
    inputs = ["lane", "register"], input_bits = [2, 1],
    outputs = ["m", "n"], output_bits = [1, 2],
    matrix = dense<[[1, 0, 0], [0, 1, 1], [0, 0, 1]]> : tensor<3x3xi1>>
} {}
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，attribute 尚未定义。

- [X] **Step 2: 定义 ODS Attr 和 verifier**

Attr 实现 `LayoutMapAttrInterface`。Verifier 检查：名称唯一、name/width 数量一致、所有 width > 0、matrix 行数等于输出总 bit、列数等于输入总 bit。

- [X] **Step 3: 接入 GF2 algebra**

实现：

```cpp
FailureOr<GF2Matrix> BitLinearLayoutMapAttr::getMatrixValue() const;
FailureOr<Attribute> BitLinearLayoutMapAttr::canonicalizeMap() const;
LayoutProof checkInjective(BitLinearLayoutMapAttr map);
LayoutProof checkSurjective(BitLinearLayoutMapAttr map);
FailureOr<BitLinearLayoutMapAttr>
composeBitLinear(BitLinearLayoutMapAttr lhs, BitLinearLayoutMapAttr rhs);
```

Canonical form 保持命名维度顺序，但删除 width 为 0 的禁止状态、规范 DenseElements 存储并拒绝重复名称；不得按 hash 顺序重排。

- [X] **Step 4: 验证 parse/print、compose 和 replication**

Unit test 检查全零 input column 被识别为 kernel/replication，FileCheck 检查 parse-print 稳定。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='BitLinearLayoutTest.*'
```

Expected: unit/lit tests PASS。

- [X] **Step 5: 提交 BitLinear Attr**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test unittests
git commit -m "feat: add bit-linear layout attribute"
```

### Task 8: 实现 Affine outer 与 Product layout

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutAlgebra.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutAlgebra.cpp`
- Create: `unittests/Dialect/Frisk/Layout/ProductLayoutTest.cpp`
- Create: `test/Dialect/Frisk/layout/product-layout.mlir`

**Interfaces:**

- Produces:

```text
AffineLayoutMapAttr(input_names, input_extents,
                    output_names, output_extents, affine_map)
ProductLayoutMapAttr(outer, inner, split_extents)
```

```cpp
FailureOr<Attribute> composeLayoutMaps(Attribute lhs, Attribute rhs);
FailureOr<Attribute> projectLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
FailureOr<Attribute> permuteLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
LayoutProof checkCoverage(Attribute map, ArrayRef<int64_t> logicalShape);
LayoutProof checkInjectivity(Attribute map, ArrayRef<int64_t> domainShape);
```

- [X] **Step 1: 写 non-power-of-two 与 carry 红灯测试**

测试概念语义 `logical = outer * innerExtent + inner`；Attr 中将 Affine outer
规范存为已缩放的 `outer_base = outer * innerExtent`，实际求值为
`logical = outer_base + inner`。覆盖 shape 6 的 ragged predicate，以及 outer base
未对齐时禁止直接拼接 GF(2) 内层。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，Affine/Product API 尚不存在。

- [X] **Step 2: 实现 `AffineLayoutMapAttr` verifier**

检查 AffineMap dim 数、symbol 数、name/extent 数量、正静态 extent 或 `ShapedType::kDynamic`。动态 extent 只允许作为 symbol/bounds，不得进入 BitLinear matrix。

- [X] **Step 3: 实现 Product compose 和 canonicalization**

固定组合流程：

```text
按名字对齐维度
-> 验证 inner extent 为 2 的幂
-> 验证 outer base 不向 inner bit 产生 carry
-> 分别 compose outer/inner
-> 合并 predicate
-> canonicalize factor order
```

Product verifier 无法证明无 carry时直接拒绝构造；独立 proof API 对动态/超限情形返回
`ProofStatus::Unknown`，不得生成不安全 Product，也不得把 `Unknown` 当作成功。

- [X] **Step 4: 运行枚举 oracle**

对每维 extent 不超过 8 的 product map 枚举 `(outer, inner)`，比较 canonical map 与逐层求值；覆盖 transpose、projection、padding、ragged。

实现额外设置 65,536 个 domain point 的硬上限：完整 BitLinear domain 优先使用
GF(2) rank 精确证明；静态 Affine/Product 仅作为 verifier/测试的有界参考证明，动态、
含 symbol 或超限时返回 `Unknown`，不得进入 solver 的逐候选热循环。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ProductLayoutTest.*'
```

Expected: PASS。

- [X] **Step 5: 提交组合代数**

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests
git commit -m "feat: compose affine and bit-linear layouts"
```

### Task 9: 实现 Distributed/Storage Encoding 与 legacy adapter

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `include/Dialect/Frisk/IR/FriskAttributes.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Create: `include/Dialect/Frisk/Analysis/LegacyLayoutAdapter.h`
- Create: `lib/Dialect/Frisk/Analysis/LegacyLayoutAdapter.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutEncodingTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LegacyLayoutAdapterTest.cpp`
- Create: `test/Dialect/Frisk/layout/layout-encoding.mlir`

**Interfaces:**

- Produces:

```cpp
FailureOr<DistributedEncodingAttr>
convertLegacyDistributed(LayoutAttr legacy, ShapedType type,
                         Location loc);

FailureOr<StorageLayoutAttr>
convertLegacyStorage(LayoutAttr legacy, MemRefType type,
                     Location loc);
```

Encoding 字段：

```text
DistributedEncodingAttr(map, topology, replication)
StorageLayoutAttr(map, memory_space, alignment, vector_granularity)
```

ODS assembly format 固定为命名字段；后续测试中的 `#smem_layout` 和 `#enc` 必须在文件顶部完整定义，例如：

```mlir
#smem_layout = #frisk.storage<
  map = #smem_map,
  memory_space = #frisk<memory_space Shared>,
  alignment = 128,
  vector_granularity = 16>
#enc = #frisk.distributed<
  map = #blocked_map,
  topology = [8, 32, 4, 1, 1],
  replication = 1>
```

`topology` 的字段顺序固定为 `(register, lane, warp, warp_group, cta)`；parser、printer 和 verifier 共同使用这一顺序，禁止依靠调用点注释猜测。

- [X] **Step 1: 写 type verifier 和 baseline conversion 红灯测试**

测试：encoding logical shape 不匹配、Storage map 非单射、shared alignment 非正、现有 `sm90_ss` Gemm A/B/C legacy layout 转换。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，新 Encoding/adapter 不存在。

- [X] **Step 2: 实现 Encoding verifier**

Distributed 检查 carrier names 只来自 `register/lane/warp/warp_group/cta`、coverage、replication 与 topology；对 BitLinear map 精确验证
`replication = product(topology) / 2^rank(B_D)`。Storage 检查 memory space、live
domain injectivity、`(byte_offset, bit_offset)`、byte 单位的 alignment/vector
granularity。

- [X] **Step 3: 实现显式支持集的 legacy adapter**

Adapter 仅支持当前 SM90 baseline 中可证明的静态 2 次幂 fragment、linear/padded
shared 和能从完整真值表恢复 GF(2) basis 的 swizzle。SM80 旧路径继续回归，但不作为
当前 SM90-only adapter 的支持承诺。无法识别时：

```cpp
return emitError(loc)
       << "legacy layout cannot be represented by the canonical layout algebra";
```

不得猜测、不得退化为 linear layout。对 swizzle 使用现有语义逐点枚举构造 GF(2) basis，并验证全域后才返回。

实现细节：adapter 单独编译为 `FriskLegacyLayoutAdapter`，依赖 `FriskIR` 和
`FriskAnalysis`，从而避免 `FriskIR <-> FriskAnalysis` 循环。多结果 legacy storage
tuple 采用由各结果最大 extent 确定的 row-major flatten；只有单结果 map 可进入
Affine fallback。枚举上限为 65,536，超限返回 failure。

- [X] **Step 4: 对现有 baseline 做等价枚举**

对每个 `(thread, register)` 和 logical shared point，比较 legacy 与新 canonical map；测试失败必须打印首个反例。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutEncodingTest.*:LegacyLayoutAdapterTest.*'
```

Expected: `sm90_ss` 和 `sm90_rs` 所有已转换 operand PASS；其中 `sm90_rs` A 的
legacy `replicate=2` 必须成为一个零 carrier bit，并由 GF(2) rank 验证为 2。

- [X] **Step 5: 维护设计文档并提交**

如果最终 Attr 参数或 canonicalization 与设计文档 Section 5/6 不同，先同步修改。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests \
  guoqiao/layout_inference_design.md
git commit -m "feat: add distributed and storage encodings"
```

### M1 Gate

Run:

```bash
cmake --build build --target FriskIR FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
```

Expected: 全部通过；现有 SM90 layout baseline 的新旧枚举无差异；无 `unknown` 被当作 proven 的测试路径。

**执行结果（2026-08-30）：PASS。** `FriskLayoutUnitTests` 运行 6 个 suite、
24 个测试全部通过；lit 4/4 通过；CTest 4/4 通过。`sm90_ss` 与 `sm90_rs`
adapter 对每个 logical point 完成新旧差分，动态 Affine proof 的 `Unknown` 路径有独立
测试且未被接受为 proven。

---

## M2：Storage 纵向切片

### Task 10: 新增 `frisk.layout_view` Storage anchor

**Files:**

- Create: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `test/Dialect/Frisk/layout/layout-view.mlir`

**Interfaces:**

- Produces:

```mlir
%view = frisk.layout_view %source
  {layout = #smem_layout}
  : memref<64x64xf16, #frisk<memory_space Shared>>
    -> memref<64x64xf16, #frisk<memory_space Shared>>
```

`layout` 在推断前可省略，在 materialization 后必须存在。

- [X] **Step 1: 写 parse/verify 红灯测试**

覆盖合法 view、source/result type 不同、global view 绑定 shared layout、非单射 storage map。

```mlir
// expected-error@+1 {{source and result must have identical memref types}}
%bad = "frisk.layout_view"(%arg0) :
  (memref<64x64xf16>) -> memref<32x128xf16>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，Op 尚不存在。

- [X] **Step 2: 定义 Op 和 ViewLike 语义**

ODS 约束：一个 `AnyMemRef` operand、同类型 `AnyMemRef` result、可选 `StorageLayoutAttr`，实现 `ViewLikeOpInterface` 和无副作用 view 语义。C++ verifier 调用 `layout.verifyForType(getResult().getType(), getLoc())`。

```cpp
Value LayoutViewOp::getViewSource() { return getSource(); }
```

- [X] **Step 3: 实现 canonicalization**

只允许以下折叠：

```text
layout_view(layout_view(x, L), L) -> layout_view(x, L)
layout_view(x, identity storage)  -> x，仅当无下游 layout anchor 依赖该 Op
```

两个不同 storage binding 不得折叠或覆盖。

- [X] **Step 4: 运行 IR tests**

Run:

```bash
cmake --build build --target FriskIR check-frisk --parallel 32
```

Expected: layout-view tests PASS，非法 case 输出指定诊断。

- [X] **Step 5: 提交 layout view**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test
git commit -m "feat: add storage layout view anchor"
```

### Task 11: 建立 LayoutVar、Constraint 和 Provenance graph

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`
- Create: `include/Dialect/Frisk/Analysis/LayoutTarget.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutConstraintTest.cpp`

**Interfaces:**

- Produces:

```cpp
using LayoutVarID = uint32_t;
using LayoutConstraintID = uint32_t;
using ProvenanceID = uint32_t;

enum class LayoutState { Uninitialized, CandidateSet, Resolved, Conflict };
enum class ConstraintStrength { Hard, Soft };
enum class ConstraintKind {
  SameLayout,
  TransformLayout,
  RequireEncoding,
  InstructionContract,
  StorageAccess,
  AliasLayout,
  Ownership,
  ResourceLimit,
  Preference
};

struct LayoutCandidate {
  Attribute value;
  ProvenanceID provenance;
  uint64_t stableOrdinal;
};

struct LayoutVar {
  LayoutVarID id;
  LayoutKind kind;
  Type shapedType;
  SmallVector<LayoutCandidate> candidates;
  LayoutState state;
};

enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution;
  uint64_t bytes;
  uint64_t synchronizationCost;
};

class LayoutConstraintGraph {
public:
  LayoutVarID addVariable(LayoutKind kind, Type type, StringRef stableName);
  LayoutConstraintID addConstraint(ConstraintKind kind,
                                   ConstraintStrength strength,
                                   ArrayRef<LayoutVarID> vars,
                                   Operation *source, StringRef rule,
                                   StringRef reason);
  LogicalResult verifyInvariants(Location loc) const;
};

struct CandidateAssignment;

class LayoutTarget {
public:
  virtual ~LayoutTarget() = default;
  virtual void enumerateCandidates(
      const LayoutVar &var,
      SmallVectorImpl<LayoutCandidate> &out) const = 0;
  virtual LogicalResult verifyCandidate(const LayoutVar &var,
                                        Attribute candidate,
                                        Location loc) const = 0;
  virtual FailureOr<CostVector>
  evaluate(const CandidateAssignment &assignment) const = 0;
};
```

- [X] **Step 1: 写 stable-ID 和 invariant 红灯测试**

测试相同逻辑输入在不同插入顺序下，按 `stableName` finalize 后 ID/constraint 顺序一致；重复变量名、无效 var 引用和空 hard constraint 必须失败。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，graph 尚不存在。

- [X] **Step 2: 实现 graph ownership**

Graph 拥有 vars、constraints、provenance；`Operation*` 只用于诊断，不参与排序或 hash。稳定名称来自：

```text
unambiguously encoded fully-qualified ancestor symbol path / block ordinal / operation ordinal / result-or-operand ordinal / layout kind
```

- [X] **Step 3: 实现 provenance chain**

```cpp
struct LayoutProvenance {
  ProvenanceID id;
  std::optional<ProvenanceID> parent;
  Operation *source;
  std::string rule;
  std::string reason;
};
```

提供 `printProvenanceChain(ProvenanceID, raw_ostream&)`，检测 parent cycle。

- [X] **Step 4: 验证确定性和诊断**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutConstraintTest.*'
```

Expected: PASS；两种插入顺序 dump 完全相同。

- [X] **Step 5: 提交 constraint graph**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis unittests
git commit -m "feat: add deterministic layout constraint graph"
```

### Task 12: 实现 Storage constraint collector 和 strict/common propagation

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Modify: `lib/Dialect/Frisk/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutPropagationTest.cpp`
- Create: `test/Transforms/storage-propagation.mlir`

**Interfaces:**

- Consumes the `LayoutConversionEdge` and `EdgeResolutionKind` definitions from Task 11; produces edge enumeration and cost evaluation:

```cpp
class LayoutConstraintBuilder {
public:
  LayoutVarID getOrCreateStorageVar(Value anchor);
  LayoutVarID getOrCreateDistributedVar(Value value);
  LogicalResult require(LayoutVarID var, Attribute encoding,
                        Operation *source, StringRef rule);
  LogicalResult same(LayoutVarID lhs, LayoutVarID rhs,
                     Operation *source, StringRef rule);
};

FailureOr<LayoutConstraintGraph> collectLayoutConstraints(Operation *root,
                                                          LayoutTarget &target);
LogicalResult propagateStrict(LayoutConstraintGraph &graph);
LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph);
```

- [X] **Step 1: 写 storage propagation 红灯测试**

输入包含两个 `layout_view` 和一个 Copy；src 有 storage binding、dst 未绑定。预期 pass 后 dst 获得数学等价 binding。另一个 split case 给 src/dst 不兼容 hard seed，预期诊断包含两条 provenance。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，pass 仍只写 smoke attr。

- [X] **Step 2: 收集 Storage variables 和 constraints**

Collector 规则：

```text
layout_view with layout -> RequireEncoding(hard)
layout_view without layout -> unresolved Storage var
same-source compatible views -> AliasLayout(hard)
copy whole-tile src/dst -> StorageAccess(hard relation) + coalescing Preference
```

Task 12 只接受 whole-tile、静态 shape copy；其他 case 返回明确 unsupported diagnostic。

- [X] **Step 3: 实现 bootstrap SM90 storage candidates**

`SM90LayoutTarget` 首版只生成：linear、transpose、必要 padding、32B/64B/128B XOR swizzle。每个候选必须先通过 Storage verifier；候选按固定枚举序号排序。

这里的 candidate generation 仅用于初始化尚无 hard seed 的变量 domain；它发生在 fixed-point propagation 之前，不执行全局优选。完整候选生成、剪枝和 beam search 由 Task 23 接管。

- [X] **Step 4: 实现单调传播**

Strict 只处理 singleton seed、SameLayout 和 exact AliasLayout。Common 只执行交集/关系投影：

```cpp
while (!worklist.empty()) {
  LayoutConstraintID id = worklist.pop_front();
  ChangeResult changed = applyConstraint(graph, id);
  if (changed == ChangeResult::Change)
    enqueueAdjacentConstraintsInStableOrder(graph, id, worklist);
}
```

空集合进入 Conflict 并输出两条 seed provenance；传播循环内部不得新增 candidate。

- [X] **Step 5: 验证收敛和顺序独立**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutPropagationTest.*'
```

Expected: PASS；随机打乱初始 constraint vector 后 canonical graph dump 不变。

- [X] **Step 6: 提交 storage propagation**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk test unittests
git commit -m "feat: propagate storage layout constraints"
```

### Task 13: 物化 Storage binding 并验证第一条端到端链路

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutVerifier.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp`
- Create: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `test/Transforms/storage-materialization.mlir`
- Create: `test/Transforms/storage-conflict.mlir`
- Create: `unittests/Dialect/Frisk/Layout/LayoutVerifierTest.cpp`

**Interfaces:**

- Produces:

```cpp
struct LayoutSolution {
  DenseMap<LayoutVarID, Attribute> assignments;
  SmallVector<LayoutConversionEdge> conversions;
};

struct BootstrapSolverLimits {
  unsigned maxVariables = 8;
  unsigned maxDomainSize = 4;
};

FailureOr<LayoutSolution> solveBootstrapLayoutGraph(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    BootstrapSolverLimits limits = {});

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target,
                                      Location loc);

LogicalResult materializeLayouts(Operation *root,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution);

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target);
```

- [X] **Step 1: 写未物化和错误物化红灯测试**

覆盖 unresolved layout_view、错误 memory space、非单射 map、alias views 不一致。诊断检查：

```mlir
// expected-error@+1 {{unresolved storage layout for layout variable}}
%v = "frisk.layout_view"(%arg0) : (memref<64x64xf16>) -> memref<64x64xf16>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，verifier/materializer 尚不存在。

- [X] **Step 2: 实现严格受限的 bootstrap resolver**

为使 M2/M3 在完整求解器之前可执行，实现只面向纵向切片的 exhaustive resolver：component 最多 8 个变量、每个 domain 最多 4 个候选；只接受满足全部 hard constraint 的 assignment，并按 candidate stable ordinal 序列决定唯一结果。超过边界直接诊断 `bootstrap layout solver limit exceeded`，不得截断或猜测。

M2 不枚举 conversion/rematerialization；Task 15 扩展该 resolver 处理一个多 consumer conversion edge，Task 23 必须以正式 CostVector/beam solver 替换并删除它。

- [X] **Step 3: 实现 solved graph verifier**

逐 assignment 检查 kind、type、coverage/injectivity、alias 和 Copy relation；物化后二次验证还会枚举静态 logical domain，证明元素 bit range 不重叠且 storage footprint 不超过底层 MemRef strided/affine type 的静态容量。失败信息包含稳定 var 名、candidate、constraint rule 和 provenance chain。

- [X] **Step 4: 实现 Storage materialization**

Materializer 只给现有 `LayoutViewOp` 设置求解出的 `layout` attr；若 solution 中的 Attr 与已有 hard binding 不等价则立即失败，不覆盖。

- [X] **Step 5: 将 pass 变成真正 orchestrator**

`FriskInferLayoutsPass::runOnOperation()` 顺序固定为：

```cpp
auto graph = collectLayoutConstraints(getOperation(), target);
propagateStrict(*graph);
propagateCommonToFixedPoint(*graph);
auto solution = solveBootstrapLayoutGraph(*graph, target);
verifySolvedLayoutGraph(*graph, solution, target, getOperation().getLoc());
materializeLayouts(getOperation(), *graph, solution);
verifyMaterializedLayouts(getOperation(), target);
```

任一步失败调用 `signalPassFailure()`；删除 M0 的 smoke attr。

- [X] **Step 6: 运行端到端测试**

Run:

```bash
cmake --build build --target FriskTransforms FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutVerifierTest.*'
```

Expected: Storage 输入推断并打印 canonical binding；冲突 case 输出两条 provenance；所有测试 PASS。

- [X] **Step 7: 同步设计并提交**

如果实际 pass 阶段边界与设计 Section 8 不同，先更新设计文档。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests \
  guoqiao/layout_inference_design.md
git commit -m "feat: materialize and verify storage layouts"
```

### M2 Gate

Run:

```bash
build/bin/frisk-opt test/Transforms/storage-materialization.mlir \
  -frisk-infer-layouts -verify-each
cmake --build build --target check-frisk FriskLayoutUnitTests --parallel 32
```

Expected: alloc/view/copy Storage slice 完整通过；`LayoutInfer.cpp` 不再是 pass 壳；无 conversion 或 Tensor 逻辑被提前混入。

---

## M3：Distributed Tensor 与 conversion 纵向切片

### 实际实现与原计划的调整记录

本节按已验收代码维护，而不仅标记复选框。下列调整已落实到 Task 14–17 的正文：

| 原计划的简写或假设                                    | 实际实现及原因                                                                                                                                                                |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| convert 两端 encoding 必须不同，但又要求消除 identity | identity 是合法输入；仅拒绝 solver 请求新插入 identity，清理替换须保持精确 SSA 类型。                                                                                         |
| `tensor.transpose`                                  | 本工具链没有该 Op；使用已注册的`linalg.transpose` Tensor DPS，建模输入 use、init 和 result。                                                                                |
| 两种 Storage 布局要求两个 Distributed 候选/转换       | `StorageAccess` 证明 `S(D(h))` 的合法性，不把 coalescing 当硬约束；linear/transpose 两个 store 可以共同布局零转换。另用独立硬编码的活跃 transpose consumer 验证一次转换。 |
| 候选集合从始至终只缩小                                | 收集阶段先由硬 seed 和 target templates 双向投影生成闭包；之后 strict/common propagation 才单调裁剪。                                                                         |
| 逐 Op 改类型、while 所有位置一组相等                  | 按原始 Value 定型并重建 detached ModuleOp；while 的输入 tuple 与 condition/result tuple 分开建模，验证成功后一次提交。                                                        |
| 最小 lowering 可直接视为可执行 GPU 链路               | 当前仅生成测试用 GPU-level IR；Tensor↔thread-vector 桥接尚需 M6 消除，不能宣称 GPU runtime 验收完成。                                                                        |

完成步骤中的 `Expected: FAIL` 是当时测试驱动实现的红灯预期，不是当前验收结果。
本次状态同步不改变 M4–M6 的未完成任务或将其提前标为已实现。

### Task 14: 新增 Tensor carrier 和 `frisk.convert_layout`

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `test/Dialect/Frisk/layout/tile-carriers.mlir`
- Create: `test/Dialect/Frisk/layout/convert-layout.mlir`

**Interfaces:**

- Produces:

```mlir
%tile = frisk.tile_load %view
  : memref<64x64xf16, #frisk<memory_space Shared>>
    -> tensor<64x64xf16>

%converted = frisk.convert_layout %tile
  : tensor<64x64xf16, #src> -> tensor<64x64xf16, #dst>

frisk.tile_store %tile, %view
  : tensor<64x64xf16, #enc>,
    memref<64x64xf16, #frisk<memory_space Shared>>
```

首版 load/store 只表示 whole-tile、静态 shape；切片和动态 offset 在 M4 后扩展。

- [X] **Step 1: 写 Op verifier 红灯测试**

覆盖：load shape/dtype 不匹配、store shape 不匹配、convert source/target shape 或 element type 不同、identity convert。

```mlir
// expected-error@+1 {{source and target must have identical shape and element type}}
%bad = "frisk.convert_layout"(%x) :
  (tensor<64x64xf16, #src>) -> tensor<32x128xf16, #dst>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，新 Ops 尚不存在。

- [X] **Step 2: 定义 carrier Ops**

`TileLoadOp` 具有 MemRead，`TileStoreOp` 具有 MemWrite，`ConvertLayoutOp` 为 pure。Convert verifier 要求 source/target 均有合法 `DistributedEncodingAttr`；identity convert 是合法输入并交给 canonicalizer 删除，materializer 后续拒绝主动生成 identity convert。parser verifier 测试使用 canonicalization 前合法表示时不得崩溃。

- [X] **Step 3: 实现基础 canonicalization**

```text
convert_layout(x, src == dst) -> x
convert_layout(convert_layout(x, A -> B), B -> A) -> x
```

删除 identity/inverse 时同时检查 canonical map 等价与替换值/result 的完整 SSA 类型相等；仅数学映射等价、但 encoding 不同，不足以直接替换。

- [X] **Step 4: 运行 IR tests**

Run:

```bash
cmake --build build --target FriskIR check-frisk --parallel 32
```

Expected: carrier/convert tests PASS。

- [X] **Step 5: 提交 Tensor carrier**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test
git commit -m "feat: add distributed tensor carriers"
```

### Task 15: 收集 Distributed constraints 并支持多 consumer

已实现并通过独立审查；接口和边界见 [Task 15 analysis](m3_task15_analysis.md)。

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Create: `unittests/Dialect/Frisk/Layout/DistributedPropagationTest.cpp`
- Create: `test/Transforms/distributed-propagation.mlir`
- Create: `test/Transforms/multi-consumer-layout.mlir`

实际还新增 `lib/Dialect/Frisk/Analysis/DistributedLayoutConstraints.cpp`、
`include/Dialect/Frisk/Analysis/LayoutRelations.h`、对应 `LayoutRelations.cpp`、
`lib/Dialect/Frisk/Target/SM90/SM90DistributedCandidates.cpp` 和
`test/Transforms/distributed-propagation-errors.mlir`；修改 `LayoutConstraint.h`、
`LayoutSolver.h`、`LayoutVerifier.cpp`、`LayoutInfer.cpp`、`Passes.td` 及相关 CMake。
Op 规则收集、共享关系判断与 SM90 候选枚举分别放在这些文件，未集中堆入原计划列出的文件。

**Interfaces:**

- Extends:

```cpp
LayoutVarID LayoutConstraintBuilder::getOrCreateDistributedVar(Value value);
LayoutVarID LayoutConstraintBuilder::getOrCreateDistributedUse(OpOperand &use);
LogicalResult LayoutConstraintBuilder::convertible(
    LayoutVarID src, LayoutVarID dst, OpOperand &use, bool existing = false);
LogicalResult LayoutConstraintBuilder::transform(
    LayoutVarID src, LayoutVarID dst, Attribute coordinateTransform,
    Operation *source, StringRef rule);
LogicalResult LayoutConstraintBuilder::storageAccess(
    LayoutVarID distributed, LayoutVarID storage, AccessKind access,
    Operation *source, StringRef rule);

FailureOr<LayoutSolution> solveBootstrapLayoutGraph(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    BootstrapSolverLimits limits);
```

- [X] **Step 1: 写双 consumer 红灯测试**

构造一个 `tile_load` 结果被两个 `tile_store` 使用，两个 destination view 分别绑定 linear 和 transpose storage。实际回归检查 producer 和两个 consumer use 均保留 4 个候选，求解选择共同布局、0 个转换，而不是由第一个 consumer 锁定 producer。StorageAccess 不将访存合并度当硬约束。

另一个 `@contract` 用例使用 2x2 非 identity `linalg.transpose`、独立硬编码的 XOR 结果布局和使用原 producer 的 store；函数返回 transpose 结果，使 consumer 在 cleanup 后仍存活，并保留恰好一次输入 use 转换。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，collector 尚未创建 Distributed var。

- [X] **Step 2: 收集基础 Distributed rules**

规则作用于 producer 定义与 consumer 实际期望的 use/slot，而非强制原 producer 与所有消费者布局相同：

```text
tile_load -> StorageAccess(result definition, storage view)
tile_store -> Convertible(producer, consumer use) + StorageAccess(consumer use, storage view)
arith/math elementwise -> 同布局 results；tensor operand use 经 Convertible 满足结果布局
linalg.transpose (Tensor DPS) -> TransformLayout(input use, result, permutation)；init use 经 Convertible 满足 result 布局
existing tensor encoding -> RequireEncoding(hard)
existing convert_layout -> source/target RequireEncoding(hard) + Convertible edge
defined func -> entry argument / function result slot / return use；标量位置不改
arith.constant(DenseElementsAttr), tensor.empty -> Tensor definition binding
```

对于未知 layout-bearing Op，pass 必须报 `operation has no layout constraint model`；Tensor call、external Tensor signature 和非 dense Tensor constant 明确拒绝。SCF if/for/while 的收集同样位于 `DistributedLayoutConstraints.cpp`，随 Task 16 一并实现。

真实定义由 `LayoutVar::value` 标识，独立 consumer 期望由 `use` 标识，函数返回布局由 `functionResult` 标识；已有 result/join slot 可直接作为 consumer 期望，避免冗余变量。已编码 Tensor 的 hard binding 不因插入消费边转换而被覆盖。

- [X] **Step 3: 生成基础 distributed candidates**

SM90 bootstrap 默认模板只有 blocked、lane-striped、warp-striped、fully-replicated 四类，展开并验证后去重；小形状可能得到少于 4 个不同默认候选。候选域还可来自已有合法硬编码和关系投影，不意味着拒绝所有不属于四类模板的显式布局。当前只接受非零 rank、每个逻辑 extent 大于 1 的静态 2 次幂 Tensor tile；extent=1 受 M1 named zero-bit dimension 限制而明确拒绝。

- [X] **Step 4: 扩展 propagation 为双向关系投影**

先把已有硬 seed 投影到闭包；再逐个为尚无候选的变量枚举 target templates，每初始化一个域就继续投影到闭包。生成阶段允许添加候选；随后 strict/common propagation 执行双向支持检查和单调裁剪，避免丢掉其他 consumer 的合法选择。

Transpose 按 permutation/逆 permutation 投影逻辑 bit-row blocks，并按实际目标候选的 output names 判断兼容性。Distributed↔Storage 的 `StorageAccess` 仅检查已验证映射的逻辑域/element type 匹配以证明 `S(D(h))`，不从 Storage 顺序强制推导 Distributed 排列；原 M2 Storage↔Storage copy 仍要求兼容的 storage map。

- [X] **Step 5: 扩展 bootstrap resolver，闭合第一条多 consumer 路径**

在既有 8-variable/4-candidate 上限内，允许每个可转换 consumer edge 枚举 `KeepCommonLayout` 或 `Convert`。目标顺序固定为：先满足 hard constraint，再最小化 conversion 数，最后比较 assignment/edge stable key；本阶段不宣称性能最优，也不枚举 rematerialization。求解结果必须把选中的 conversion 写入 `LayoutSolution::conversions`，供 Task 16 直接消费。

上限精确为每连通分量 8 个变量、每个变量域 4 个候选，超过直接诊断、不截断。已有显式 conversion 不再插入或计为新增转换。`analysis-only` 执行 collect/solve/verify 而不改 IR，`dump-analysis` 输出稳定图和选边。分析支持的 single-CTA topology 范围宽于 Task 17 测试 adapter，分析成功不等于该 adapter 能降低。

- [X] **Step 6: 验证 fixed-point 与多 consumer**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='DistributedPropagationTest.*'
```

Expected: PASS；双 store 保留候选并选择零转换，独立硬编码的活跃 transpose consumer 选择一次 conversion；顺序打乱结果不变。

- [X] **Step 7: 提交 Distributed propagation**

```bash
git add lib/Dialect/Frisk test unittests
git commit -m "feat: propagate distributed layout constraints"
```

### Task 16: 物化 Tensor encoding、SCF 类型和 conversion edge

已实现并通过独立审查；实现与验证记录见 [m3_task16_materialization.md](m3_task16_materialization.md)。
下方 SCF while 规则已按实际 MLIR 的双 tuple 契约修正。

**Files:**

- Modify: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Create: `include/Dialect/Frisk/Transforms/LayoutTypeConverter.h`
- Create: `lib/Dialect/Frisk/Transforms/LayoutTypeConverter.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `test/Transforms/materialize-distributed.mlir`
- Create: `test/Transforms/materialize-scf.mlir`
- Create: `test/Transforms/materialize-conversion.mlir`

实际还修改 `DistributedLayoutConstraints.cpp`、`LayoutSolver.h` 和 `LayoutPropagation.cpp`
以加入 SCF 关系及 `LayoutCollectionMode::RelationsOnly`，新增
`test/Transforms/materialize-scalar-cfg.mlir` 和
`unittests/Dialect/Frisk/Layout/LayoutMaterializationTest.cpp`，同步相关 CMake。

**Interfaces:**

- Produces:

```cpp
class LayoutTypeConverter : public TypeConverter {
public:
  LayoutTypeConverter(const LayoutConstraintGraph &graph,
                      const LayoutSolution &solution);
  FailureOr<RankedTensorType> convertLayoutBearingTensor(Value value) const;
};

LogicalResult materializeDistributedLayouts(
    Operation *root, const LayoutConstraintGraph &graph,
    const LayoutSolution &solution,
    ArrayRef<LayoutConversionEdge> conversions);
```

- [X] **Step 1: 写 SCF 和 conversion 红灯测试**

覆盖 `scf.if` 两个 yield、`scf.for` init/block argument/yield/result、`scf.while` 不同 arity 的两组 tuple，以及多 consumer 的一次转换。实际文本用例在 transpose 输入 use 前插入 conversion，独立 tile_store 继续使用原 producer：

```mlir
// CHECK: %[[CVT:.*]] = frisk.convert_layout %[[PRODUCER]]
// CHECK-NEXT: {{.*}}linalg.transpose ins(%[[CVT]]
// CHECK: frisk.tile_store %[[PRODUCER]],
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，Tensor type 尚未被改写。

- [X] **Step 2: 实现非 region Op 的 type rewrite**

`LayoutTypeConverter` 按原始 SSA Value 查 solution；同一个原始 Tensor Type 的不同 Value 可得到不同布局，因此继承的 Type-only 转换路径明确拒绝 Tensor，不能按 Type 缓存布局。

实际实现仅接受 ModuleOp 根：预检 solution 并快照选中 use 的 owner/operand index，在 detached ModuleOp 中用 `Operation::create` 和 `IRMapping` 重建整个 body。保留属性、native properties、位置、successor、region 和标量参数；同步函数签名以及 DenseElementsAttr 的结果类型。逐 block 保留原操作顺序，调度时检查显式 operand 与嵌套 region 捕获值，使非 dominance 打印顺序的标量 CFG 也可处理；不是原地逐个 setType，也不是只克隆少数结果 Op。

- [X] **Step 3: 实现 SCF 一致重写**

固定规则：

```text
scf.if: each yield operand encoding == corresponding result encoding
scf.for: init == iter_arg == yield == result
scf.while: init == before args == after yield; condition args == after args == results
```

这里的 init/yield/condition 等式指完成选中 use 转换后的 operand 编码，不要求其原 producer 定义预先相等。For 的 iter_arg/result、while 的 after args/results 直接硬相等；需要转换的 init、yield、condition use 分别在相应消费点插入。While 的两组 tuple 可以有不同 arity/type，predicate 不计入 forwarded operands，也不把 before/after 无关槽位强行绑定。不得修改 join 后类型逃避 hard constraint。

- [X] **Step 4: 物化 solver 选中的 conversion edge**

只有 `LayoutSolution` 显式列出的 edge 才插入 `frisk.convert_layout`。source/target 相等时视为 solver/materializer bug 并失败；materializer 不重新比较成本。

物化后先验证 staged MLIR，再用 `RelationsOnly` 重收集关系，把每个定义、实际 operand use、函数结果绑定到 IR 中实际存在的 encoding。此处不枚举候选、不传播、不重新求解，不能通过“假设未来会插入 conversion”掩盖缺失转换；同时保留 M2 alias/copy/capacity 检查。全部成功后只执行一次 module body transfer；任何失败都保留原 IR。提交后原 graph/solution 借用的 Value/Operation/OpOperand 身份失效，不可继续解引用。

- [X] **Step 5: 运行 materialization tests**

Run:

```bash
cmake --build build --target FriskTransforms check-frisk --parallel 32
```

Expected: Tensor/SCF/conversion tests PASS，`-verify-each` 无错误。

- [X] **Step 6: 提交 Distributed materialization**

```bash
git add include/Dialect/Frisk/Transforms lib/Dialect/Frisk/Transforms test
git commit -m "feat: materialize distributed layouts and conversions"
```

### Task 17: 实现基础 conversion cleanup 和最小测试 lowering adapter

**Files:**

- Create: `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `lib/Conversion/FriskLayoutToGPU/CMakeLists.txt`
- Create: `lib/Conversion/FriskLayoutToGPU/TestLayoutConversionLowering.cpp`
- Modify: `lib/Conversion/CMakeLists.txt`
- Create: `test/Transforms/optimize-layout-conversions.mlir`
- Create: `test/Conversion/FriskLayoutToGPU/warp-shuffle.mlir`
- Create: `test/Conversion/FriskLayoutToGPU/shared-exchange.mlir`

实际还新增 `include/Conversion/FriskLayoutToGPU/LayoutConversionPlan.h` 及
`lib/Conversion/FriskLayoutToGPU/LayoutConversionPlan.cpp`，让 emitter 与单元测试共享
owner/payload plan；新增 `LayoutConversionPlanTest.cpp` 及 register-selection、
replica-locality、scratch-budget、existing-scratch、unaccounted-scratch、unsupported
测试文件。清理模式复用 `FriskLayoutOps.cpp` canonicalization，接线还修改 `Passes.h`、
`lib/CMakeLists.txt`、`tools/frisk-opt/CMakeLists.txt` 和 unit-test CMake。

**Interfaces:**

- Produces:

```cpp
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass();
std::unique_ptr<Pass> createTestLowerLayoutConversionsPass();
```

已实现。详细 adapter 边界和 M6 Tensor↔thread-vector 桥接职责见 [Task 17 implementation](m3_task17_conversions.md)。测试 lowering 仅支持静态、单 CTA、BitLinear→BitLinear redistribution：同线程优先 register extraction，同 warp 优先 shuffle，跨 warp 使用真实 workgroup attribution + barrier exchange。两端 execution topology 必须匹配，lane=32，gpu.func 单 entry block 且 known_block_size 匹配；不支持的 scope/topology/dtype 明确诊断。

公共 SSA 保持原始精确 RankedTensor 类型，内部使用 `builtin.unrealized_conversion_cast`
连接每线程 vector；该表示桥接并非可执行的 tile_load/store lowering，M6 仍需接管和消除。
此 adapter 不接受嵌套 SCF 转换，不支持循环 scratch 复用、跨 CTA/cluster 或 sub-byte。
额外明确限制为至多 1024 threads、两端各至多 256 registers/thread、逻辑 volume≤65536、
source+destination carrier visits≤262144、保守 shuffle-word count≤65536；均是 adapter 限额。

- [X] **Step 1: 写 cleanup/lowering 红灯测试**

覆盖 identity、A→B→A、相邻 A→B→C 合并，一个 lane permutation conversion 产生 `gpu.shuffle`，以及一个跨 warp permutation 产生 workgroup scratch store/barrier/load。动态 encoded tensor 在正常 IR 路径由 M1 verifier 先拒绝；programmatic planner negative 单独验证动态输入。可到达测试 adapter 的 unsupported 情况使用 `test lowering requires a static single-CTA redistribution` 前缀，未禁用 verifier。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，passes 尚不存在。

- [X] **Step 2: 实现 type-safe cleanup patterns**

只实现数学上可证明的：identity elimination、相邻 conversion compose、inverse pair elimination。Pattern 必须调用 canonical map equality，不比较 Attr 指针。

- [X] **Step 3: 实现单 warp redistribution 计划**

计算：

```text
R = rightInverse(D_src) compose D_dst
```

按 named input 解码每个目标 carrier 对应的 source lane/register，并验证 source coverage/bounds；replica 优先同 thread/warp。所有 source-register candidate shuffle 均在全 warp 无条件执行，之后按 destination 请求选择 register。i8/i16/i32/i64/f16/bf16/f32/f64 经 bit-preserving i32 words 传输；64-bit 拆分/重组。

- [X] **Step 4: 实现单 CTA shared exchange**

先用 canonical maps 为每个 live logical element证明唯一 source owner 和所有 destination owners，再以 logical linear index 分配 scratch slot：source owner 写入，执行一次 `gpu.barrier`，destination owner 读取。scratch 字节数按 element bit width、tile volume 和 alignment 精确计算；sub-byte 暂不支持。global writer election 与 local replica preference 分离。缺失 source/writer/slot proof 失败报告 logical coordinate；容量/类型限制给出具体原因。默认 49152-byte adapter budget（非 SM90 最大值）计入原 dtype byte size、既有 attribution、padding 和全部新增 scratch；dynamic/non-attributed workgroup storage 拒绝。

- [X] **Step 5: 注册 passes 并运行测试**

Run:

```bash
cmake --build build --target FriskTransforms FriskLayoutToGPU FriskLayoutUnitTests check-frisk --parallel 32
```

Expected: cleanup、single-warp 和 shared-exchange FileCheck PASS；unsupported case 使用预期诊断失败；生成 IR 通过 `-verify-each`。

- [X] **Step 6: 提交 conversion MVP**

```bash
git add include/Dialect/Frisk/Transforms lib/Conversion \
  lib/Dialect/Frisk/Transforms test
git commit -m "feat: optimize and test-lower layout conversions"
```

### M3 Gate

状态：通过（2026-09-10）。27 lit、74 unit、4 CTest 全部通过；活跃 2x2 transpose
consumer 保留恰好一个选定 conversion，重复 infer/cleanup 输出逐字一致。
详细范围、命令和限制见 [M3 execution notes](m3_execution_notes.md)。

Run:

```bash
cmake --build build --target check-frisk FriskLayoutUnitTests frisk_attr_test \
  frisk_reduce_layout_test frisk_layout_pass_test frisk_memory_effect_test --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
build/bin/frisk-opt test/Transforms/multi-consumer-layout.mlir \
  -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each
```

以上命令从已配置的主工作区 build 运行；`--split-input-file` 可选。该 lit 用例还
检查活跃 consumer 及再次 infer/cleanup 后的输出逐字一致。构建命令同时刷新 CTest
的四个 legacy test target；`check-frisk` 不代替执行独立的 layout unit 二进制。

Expected: 多 consumer IR 合法；共同布局或 conversion 由 solution 明确决定；SCF 类型一致；单 warp 与单 CTA shared-exchange conversion 可以 lower，非支持路径明确失败。

---

## M4：完整约束传播与旧 IR 迁移

### 2026-09-12 实现审计与阶段划分

本轮读 Frisk `b120d06`、TileLang `5e149e3`、Triton `42c5e89` 的源码与相关测试；完整 SHA、永久链接及比较结论见[比较文档](./layout_inference_strategy_comparison.md)。TileLang 另对照旧 `6623b12`，未改用户的 TileLang checkout。

| 阶段     | Task   | 本阶段目的                                | 本轮状态                                                            |
| -------- | ------ | ----------------------------------------- | ------------------------------------------------------------------- |
| 第一阶段 | 18     | 坐标 alias、显式 region graph、可审计收敛 | 用户已确认，已编码；最终 Gate 见 §18.9                             |
| 第二阶段 | 19     | Copy/Fill/Parallel 进入新接口与约束       | 已按确认契约实现；独立复审及 Gate 记录见 §19.9                      |
| 第三阶段 | 20–21 | Tensor MMA/GEMM 与 Reduce 规则适配        | Task 20 已通过 Gate；Task 21 受限实现与回归完成，完整链成功验收待解决，见 §21.8 |
| 第四阶段 | 22     | normalization、旧生产逻辑退役和集成       | 未开始；依赖前三阶段 Gate                                           |

审计起点结论与具体修正（以下 Frisk 缺口描述固定于 `b120d06`，实施后状态见 §18.9）：

| 项目                  | 当前实际实现/上游证据                                                                                                            | 决定                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Frisk storage alias   | 只沿 layout_view 找 root；邻接 AliasLayout 链、候选直接复制、Attr==                                                              | Task 18 改坐标证明；不能声称 cast/subview 已支持                                                        |
| Frisk SCF             | M3 已有 if/for/while，while 输入/输出两 tuple 已分开                                                                             | 保留语义并显式记录边，不重做或弱化为同槽位强相等                                                        |
| Frisk propagation     | 候选增长 closure 后执行多轮全图扫描，无可核验 pop/deletion 统计                                                                  | 分离有界候选准备与单调删减 worklist                                                                     |
| Frisk actual verifier | 从实际 IR 重置 singleton 并核验；事务提交已存在                                                                                  | 保留无推测、无 future conversion 的验证边界；纠正 RelationsOnly 提前复制 alias seed 的模式契约          |
| TileLang alias        | bit-aware reshape、same-data 分组、widening 同线程/连续槽位证明旧版就已有                                                        | 借鉴 storage bits 不变量，但不把同存储等同坐标恒等；dtype reinterpret 本阶段仍拒绝                      |
| TileLang 更新         | PartialFragment、reducer unset→narrow→wide、dst-steering、epoch 语境扩展、attempt 隔离；spill-aware 默认评分和 opt-in I/O 模型 | Op/reducer 规则由 Task 19–21 审阅适配，cost policy 留给 M5/Task 23；Task 18 不引入 reducer/cost solver |
| Triton 当前           | while 正向双 tuple；物理 BufferRegion 用于 effect/ConSan/同步；PaddedSharedEncoding 已组合 padding/linear                        | 不能把这些单独宣称为 Frisk 创新；比较统一约束的范围、证明与失败契约                                     |
| 旧 Frisk Op           | 旧局部方法还在，但新 pass 尚未完整使用 Op interface models                                                                       | 迁移按当前语义与独立 oracle；不能只搬旧 args 下标/映射代码                                              |

本轮基线验证命令（均 exit 0）：

```bash
cmake --build build --target check-frisk FriskLayoutUnitTests \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 32
build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
```

结果：27/27 lit、74/74 unit（12 suites）、4/4 CTest。它们确认审计起点可回归，不包含尚未编写的 Task 18 用例；没有上游运行差分或性能结论。

验证过程备注：最终复跑时，一次与重新构建并行启动的 CTest 报 `permission denied / BAD_COMMAND`，4 个进程均未进入测试；检查日志与可执行权限后，在相关目标链接完成、经授权的沙箱外环境重跑，4/4 通过。未为此修改代码；不将该次启动失败当作功能断言失败，也不省略其验证记录。文档检查另核对了 65 个当前源码/本地链接及 1 个历史源码链接，`git diff --check` 通过。

### Task 18: 完成 alias/view 与 region graph

**状态：设计 v1 已获用户确认（2026-09-12），实现、独立复审与 Gate 已完成；2026-09-16 按用户授权提交并合入本地 main，未推送。实际差异、已知限制及证据见 §18.9，集成记录见 §18.10。**

- [X] 固定三方源码版本并完成实现审计，修正比较文档。
- [X] 写明 Task 18 的支持边界、地址契约、候选/传播算法、失败语义与验收矩阵。
- [X] 用户确认以下设计与支持边界。
- [X] 红灯测试 → 实现 → 独立评审 → Gate。
- [X] 2026-09-16 经用户授权提交并本地集成；远端推送不在本次授权范围。

#### 18.1 范围与设计选择

采用“坐标关系驱动的别名分析”。支持同元素类型/同 memory space 的静态 `memref.cast`、`memref.subview`（非零 offset、正 stride、降秩、嵌套），并保留 `layout_view` 的纯绑定语义。这里的“静态”是所需的 root shape、路径 metadata、边界和单射性都能证明，不只是 result type 看起来静态。

第一阶段不支持动态 offset/size/stride、dtype reinterpret、memory-space cast、未知 view 语义、非正 stride、无法证明的 root alias/容量；这些不是可忽略边，而应在相关布局链上给出明确诊断。不扩展任意 memref.reshape、非结构化 CFG、跨过程 alias 或 memref loop-carried alias 分析；不同函数实参之间的 MayAlias 不升级为 MustAlias，也不据 root ID 不同断言 NoAlias。Region 部分保持 M3 的 Tensor SCF 范围。

不扩展 GEMM/Reduce、成本求解、conversion hoist/remat 或完整硬件 lowering；不放宽现有 Copy/TileLoad/TileStore 的“直接 layout_view operand + whole tile”支持边界。静态 Tensor shape 与 bootstrap solver 大小限制继续生效。

三种方案比较：

| 方案                                                                        | 问题/取舍                                                                                                    | 决定                                                       |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| 同 root 直接 union/Attr==                                                   | offset、stride、rank reduction 后坐标不同；不同文本可语义相等；局部重叠不具备等值传递性                      | 拒绝                                                       |
| 引入隐藏 canonical-root LayoutVar，所有 view 只连 root                      | 关系简洁，但没有显式 root binding 时，隐藏 root assignment 无法由实际物化 IR 独立取得，还需新增物化/验证规则 | 暂不引入                                                   |
| 保存每个实际 view→root 坐标映射，对同 root 的实际 views 做完整逐对兼容检查 | 多一些 pair constraints，但不用隐藏 assignment；可复用现有有限域与 actual-only verifier                      | **本阶段推荐**；候选生成仍可从有限 root 模板前向投影 |

逐对检查不等于每次 worklist pop 枚举笛卡尔积：先缓存每个 endpoint/candidate 的 root 坐标与物理 bit 区间索引，再按共同坐标/排序区间检查。图构建须记录 pair 数与证明规模；第一阶段沿用小 component 限制，不声称大图复杂度已经解决。

#### 18.2 坐标、地址原点与容量契约

规范化记录：

```text
view v:
  root identity + stable root key
  T_v: view logical coordinates -> root logical coordinates
  root static descriptor (shape / offset / strides / element bits / memory space)
  root accessible span proof + view live domain + provenance
```

坐标变换由 Op 路径组合，不从 result descriptor 的 offset/strides 反解根坐标：

- cast：同一 logical element 的恒等变换；可从静态 source 恢复中间 cast 隐去的事实，不能用动态 source 猜静态范围。实际 layout_view endpoint 仍须具有静态 ranked shape；允许中间 cast 丢信息、后续静态 endpoint 恢复，不扩展当前属性 verifier 为动态 shape 的值上下文验证器。
- subview：保留维度满足 `q_i = offset_i + stride_i * x_j`；被降掉的维度固定为 `offset_i`。
- 使用 MLIR Op 实际 rank-reduction mask；不能“删除所有 size=1 的维度”，因为这些维度也可能保留。
- 逐层证明 `0 <= offset`、`offset + (size-1)*stride < sourceExtent`，检查整数溢出、正 live extent；嵌套时组合 T。
- root 的静态 descriptor 必须可证明单射，拒绝 `strides=[1,1]` 等重叠歧义。未知 view 不透明穿透；ViewLike interface 只定位 source，不自动提供该 Op 的坐标语义。

**统一地址契约（已确认并接入 Task 18）**：Storage map 输出相对于共同 root 的 **aligned pointer** 的 `(byte_offset, bit_offset)`。物理区间为：

```text
A_v(x) = 8 * byte_offset_v(x) + bit_offset_v(x)
I_v(x) = [A_v(x), A_v(x) + elementBits)
```

subview 的 descriptor offset 只用于坐标路径证明，不能在已 root-relative 的 map 输出上再加一次。显式 root binding 的 map 也使用同一原点。无 binding 时，root 的默认 linear 模板使用 `elementBits * (rootOffset + Σ rootStride_i*q_i)`；其他 root 模板必须声明相同地址原点、证明容量，不能沿用“忽略 root offset”行为。布局绑定表达新的物理布局契约，未绑定 root descriptor 用于确定 logical domain/可访问 span，并不强迫所有显式 swizzle 等于原线性地址公式。

root nonzero offset 的前缀不是可随意使用的空间：可访问边界采用 `[elementBits*rootOffset, elementBits*(rootOffset + Σ(extent_i-1)*stride_i + 1))`，所有实际 map 区间须落在其中。对静态 alloc/alloca/get_global 追到真实根；外部参数只使用类型保证的该区间，不能推断更大的实际 allocation。中间 cast/subview 的 type span 不得扩大 root 的保证范围；binding 不改变 allocation 大小。

容量是可访问地址上界/区间，不是元素总字节数；带 padding 时二者不同。单独通过 result memref type 的容量检查不够，还须证明 view live domain 位于 root 内。packed dtype 按 bit 区间计算，byte 容量向上取整，不能丢弃 bit offset。

**具体例子（已由 `layout-alias.mlir` 和 `StorageAliasIntegrationTest` 覆盖）：**

```text
root: 4x8xi32，offset=0，strides=[8,1]，linear S_r(r,c)=32r+4c bytes
view: subview offsets=[1,2], sizes=[2,3], strides=[1,2]
T_v(i,j)=(i+1, 2j+2)
S_v(i,j)=32i+8j+40 bytes
地址：{40,48,56,72,80,88}；每个元素占 4 bytes，root 容量 128 bytes。
```

同一 root 的另一个视图若也覆盖 `(1,2)`，该点必须为 byte 40；若它把另一个 root 元素放到 byte 40，则是非预期碰撞，也要拒绝。这个 view 的 map 与 root 的 map 文本不同是正常的，直接复制 `32i+4j` 则错误。

若 root 另有 offset=3，其默认 linear 地址整体加 12 bytes，root 可访问区间从 byte 12 开始；本例 view 首元素为 byte 52，不能再加 subview descriptor offset。

#### 18.3 AliasLayout 的兼容性与图不变量

对同一可证明 root 的任意两个实际 view 候选：

1. 若 `T_a(x)==T_b(y)`，两者必须有相同物理 bit 起点及元素区间。
2. 若 root 坐标不同，两个物理元素区间不得相交，即使 logical views 不相交。
3. 每个候选本身须通过 live-domain injectivity、bit/output bounds 和 root 容量验证。
4. alignment/vector-granularity 是独立保证：按 root pointer 的真实对齐依据、投影后地址和实际使用条件共同验证；offset 可整除不等于绝对地址对齐。可信依据限定为 allocation/global 的显式对齐、已验证 ABI/参数前置契约，以及显式 whole-root binding 声明的前置契约（须记录 provenance，不能称运行时已证明）。没有依据不能推导更强对齐，只能降低推断保证或拒绝相应向量候选。child 的对齐要求不是 root 对齐证据；不得仅复制较强保证或因地址相同就认定所有 metadata 相等。

所有同 root pairs 都覆盖，不能保留旧的邻接链。反例：`A∩B=∅, B∩C=∅, A∩C≠∅`，只连 A–B–C 会漏检 A/C 冲突；仅跳过逻辑不相交 pairs 还会漏检不同元素的物理碰撞。

逐对约束验证的是**实际绑定视图覆盖域的联合一致性**，不是宣称未被任何 view 覆盖的 root 元素已经有唯一布局。若存在 root 的 whole-domain binding，它自然作为一个实际 endpoint 约束全部子视图；若不存在，不在 verifier 中求一个隐藏 root 解。此限制不授权对未绑定 root 的普通访问进行新布局 lowering。

Alias payload 显式绑定各 endpoint 的 T/root/domain。现有 finalize 会排序 AliasLayout endpoints：新实现必须连 payload 一起 remap/交换，不能单独排序 vars。stable key 包含 root、endpoint、canonical transform；不得依赖 Value 指针排序。graph invariant 检查两端 storage kind、dtype/space、rank/domain、同 root 证据及有效的 var/constraint IDs。

求解与物化核验共享**关系语义/证明实现**，但 actual verifier 的输入必须独立从当前 IR 重建；不能信任此前候选缓存或 solution 的未来转换计划。

#### 18.4 有限候选准备、XOR 投影与反向规则

先固定候选 origins，再冻结域，最后执行删减传播。禁止把任意 affine 正反向组合直接塞进现有增长 closure：

- origins 来自显式 seeds 及每个 root/必要 endpoint 一次性的 target enumeration；用 stable origin ID 去重。
- root 模板可按 `S_v = S_r ∘ T_v` 投影到实际 endpoint。显式 whole-root seed 和可证明全域可逆的 seed 也可成为 root origin，保留 M2 同域 alias 传播能力。
- 切片不是满域双射。child seed 只能在既定有限候选中筛选 parent；不能“求逆”得到未观测元素的唯一布局。
- 从 child 向其他 view 转移只有在后者覆盖域完全包含在该 child 的已知域且逆像唯一时才允许；否则不生成扩展。每个 `(origin, endpoint)` 最多生成一次规范投影，不经循环反复拼表达式。
- 显式 seeds 本身保留为候选，兼容性交给 hard constraints；不能只因不属于默认模板就悄悄替换。无可用有限扩展候选时诊断“支持的候选域内无解”，不宣称数学上所有布局无解。
- 新 alias 投影不得使已有 Tensor 变换 closure 无界；保持 M3 有限变换族/去重，并为新增关系明确 origin 上界。无法证明有限闭包的新增变换不在本阶段接入。

非零 offset/非二次幂 stride 与 XOR 不能直接按 homogeneous GF(2) 合成：整数加法可能进位。例如 input bit 可写为 `(x floordiv 2^k) mod 2`，XOR output bit 是所选 bits 之和 mod 2。推荐把受支持的有限 bit-linear 映射精确转换为 affine floorDiv/mod 表达式，再与 T 组合，最后验证实际 live domain；零偏移纯 bit-linear 可保持快速路径。

限制 bit width、表达式规模、算术溢出和 proof domain；沿用当前 65536 live-point 枚举上界作为首阶段精确验证界，超限返回 Unknown/unsupported，不增加截断采样“证明”。ProductLayoutMap 不是任意 composition wrapper，本阶段不扩展任意 affine×GF(2) 组合支持。无法表示的 proposal 可丢弃；显式 hard seed 无法证明或最终没有可证候选时必须诊断。

候选兼容缓存按 graph/endpoint/origin 作用域管理，IR 改写后失效。RelationsOnly **不调用 target enumeration、不复制 alias seed、不投影、不求候选闭包**；只记录真实 bindings/RequireEncoding 和关系，随后用实际 IR assignment 设置 singleton 验证。

#### 18.5 Region graph：保留 M3 语义，显式记录流向

不再采用旧草案的固定 `incoming/blockArgument/yielded/result` 四字段一组结构。使用带 kind 的边，引用现有约束，避免重复构造 Convertible：

| SCF 情况           | 需要保留的映射                                                               |
| ------------------ | ---------------------------------------------------------------------------- |
| if                 | 每个 branch yield use → 对应 result slot；没有伪造 block argument           |
| for                | init use → iterarg；iterarg 与 result SameLayout；yield use → carried slot |
| while 输入 tuple I | init use → before arg；after yield use → before arg                        |
| while 输出 tuple O | condition forwarded use → after arg/result；after arg 与 result SameLayout  |
| while predicate    | i1 条件不是 forwarded tensor，不建立布局边                                   |

I/O tuple 的 arity/type 可以不同，严禁按相同 index 把 before 与 after 强行等值。保留 zero-result、scalar condition 及已有嵌套 SCF 行为。

拟新增接口示意（名字/字段可在批准后的实现评审中细化，语义不得弱化）：

```cpp
enum class RegionLayoutEdgeKind {
  IfYield, ForInit, ForBackedge, ForResult,
  WhileInit, WhileBackedge, WhileCondition, WhileResult
};
struct RegionLayoutEdge {
  RegionLayoutEdgeKind kind;
  LayoutVarID source;
  LayoutVarID target;
  LayoutConstraintID constraint;
  Operation *owner;
  OpOperand *use; // 槽位等值边允许为空；转换边必须是实际 operand use。
  unsigned slot;
  std::string stableKey;
};
```

collector 建边与约束一次完成；finalize 后同步 remap IDs。graph dump 公开 kind/slot/stable key/constraint，pointer 只用于有效期内的定位。转换仍由 M3 solution 授权，在实际 init/yield/condition/consumer use 物化；本任务不增加跨 side effects 移动或隐式转换。

#### 18.6 单调删减 worklist 与可核验统计

graph finalize、候选准备完成后冻结 domains，建立各变量去重且按稳定 constraint ID 排序的 hard adjacency。stable FIFO 初始放入本阶段每条需执行 hard constraint 一次；inQueue 去重。pop 时先清 inQueue，只有实际 domain 缩小时才唤醒受影响变量的相邻约束。

Strict/Common 可共享 kernel，但各自保留现有关系启用条件；分别报告统计，避免把两次初始队列当成一次。Strict 的 adjacency/degree 必须包含本阶段所有潜在可执行关系，不能因当前尚无 singleton 就删除边；singleton 启用条件在 pop 时判断，后续 domain change 仍会唤醒它。删减阶段不得再 enumerate/project/addCandidate。每阶段定义：

```text
N0 = 初始候选总数
Nf = 最终候选总数（失败时为已删减后的实际状态）
C0 = 初始入队的约束数
deletedCandidates = N0 - Nf <= N0
domainChanges = 实际缩域事件数
popUpperBound = C0 + Σ_events degree(changedVar)
queuePops <= popUpperBound
静态宽松界 = C0 + Σ_v |D_v(initial)| * degree(v)
```

一次删除多个候选，deletedCandidates 按实际数量累计，domainChanges 按真实变量变化事件计。冲突路径也先计数再返回。degree 只计算该阶段能够被调度的硬约束；soft preference 不混入一致性证明。

统计还包含最终候选数、enqueue/最大队列长度、按 var 的 change 次数，以及候选准备 origins/projected candidates/proof evaluations（与删减统计分开）。测试核对逐项账本和静态界，不用“最多循环 N 次”掩盖振荡。确定性比较以稳定逻辑身份和相同输入域为前提，不要求不同语义/不同图有相同统计。

#### 18.7 验收矩阵（具体执行记录见 §18.9）

| 类别                    | 最小正例                                                                                           | 必须拒绝/揭露的反例                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| cast/view 链            | 同 dtype 静态 cast、nested layout_view；中间 cast 丢信息后在静态 endpoint 恢复                     | 动态 source 无法证明、动态 shape endpoint、未知 ViewLike、dtype/memory-space reinterpret                               |
| subview                 | 上述 4×8 例子；嵌套正 stride、降秩；size=1 维度保留/删除分别覆盖                                  | 越 root 边界、非正 stride、零 live extent、算术溢出、动态 offset/size/stride                                           |
| 地址原点/容量           | root offset=3 首元素 byte 52；root 容量大于 view span；packed bit 区间                             | offset 重复叠加、利用 root 前缀、仅凭扩大后的 view type 通过容量检查                                                   |
| 对齐                    | root 对齐契约 + 投影地址共同支持较弱的 view 对齐                                                   | 仅凭相对 offset 整除推断更强绝对对齐；child 要求倒推 root 保证                                                         |
| alias 兼容              | 不同 Attr 文本但同实际地址；不同视图的部分重叠                                                     | A/B/C 非邻接冲突；不同 logical 元素物理区间碰撞；非单射 root                                                           |
| 投影与反向              | XOR+offset=1/stride，逐点独立预期；切片兼容多个 parent 候选不误锁定                                | 把 offset 当 XOR；逆切片伪造全域；表达式/证明域超限却接受                                                              |
| candidate/RelationsOnly | counting target 证明 actual verifier 不枚举；带 bound/unbound storage aliases                      | seed 复制或 projection 偷渡到 RelationsOnly；依赖未物化 root/未来 conversion 通过                                      |
| SCF                     | nested if/for/while；不同 arity/type I/O tuples、zero result；实际 carried/result 缩域后回边重调度 | predicate 进入布局域、before/after 误强绑、漏掉 yield/condition 约束；不能为制造缩域而把 Convertible 偷换为 SameLayout |
| 统计/确定性             | 同一稳定图不同插入顺序，assignment/IR/统计相同；逐次删除/pop 可对账                                | 超删除界、伪造 change、全图扫描计成依赖 pop、固定 N 次掩盖未收敛                                                       |
| 独立核验/事务性         | 物化后重收集实际绑定，所有 pair/region constraints 成立                                            | 篡改一个 view 地址被 actual verifier 拒绝；任意失败保持原 IR 逐字不变                                                  |

证明失败需区分 Disproven（首个 view/root 坐标、bit 区间或边界反例）和 Unknown（具体动态项/表示限制/预算）；均不得当作硬关系成功。诊断包含 root/view stable key、源位置和 provenance 链，不以 assertion 处理用户 IR。

#### 18.8 预期文件与实施顺序（确认后执行）

**Files（原设计清单；实际职责细分见 §18.9）：**

- Create: `include/Dialect/Frisk/Analysis/LayoutAliasAnalysis.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutAliasAnalysis.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`、`LayoutSolver.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`、`LayoutRelations.cpp`、`LayoutPropagation.cpp`、`LayoutVerifier.cpp`、`DistributedLayoutConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`（root-relative 候选适配，不引入 cost solver）
- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`、`FriskLayoutOps.td` 及对应 verifier/实现（公共地址契约）
- Modify: Analysis/unit CMake；`guoqiao/layout_inference_design.md`、本计划和比较文档
- Create: `unittests/Dialect/Frisk/Layout/AliasRegionTest.cpp`
- Create: `test/Transforms/layout-alias.mlir`、`layout-scf-fixed-point.mlir`

- [X] **Step 1：确认设计后建立隔离实现工作区，编写红灯测试。** 先复用 M3 regression，再添加 offset/stride/rank-reduction 与真实 region 边用例，确认因能力缺失失败。
- [X] **Step 2：实现坐标规范化与 AliasLayout 证明。** 定义 root-relative 契约、容量与 pair invariants，优先支持显式 bindings 的独立核验，再实现有限候选投影。
- [X] **Step 3：实现显式 region edges。** 复用 M3 SCF 关系，不重复建转换边；补全 dump/finalize remap/invariant 测试。
- [X] **Step 4：替换传播调度并公开统计。** 冻结候选域后执行 stable worklist；验证两阶段删除/pop 上界、冲突计数与确定性。
- [X] **Step 5：接入 actual-only verifier 与原子物化。** 不允许用 speculative root/候选修补实际 IR；运行篡改布局和 rollback 用例。
- [X] **Step 6：完整 Gate 和独立审查。** 除下述命令外，逐项完成 §18.7；将实测数量、实现限制和偏离设计的理由同步两份长期文档。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
git diff --check
```

Expected：新增 alias/region 测试及全部 M0–M3 regression PASS；实际 IR 核验通过，失败路径不改 IR；统计满足推导界。不以旧 27/74/4 数量充当新增功能验收。

- [X] **Step 7：按授权提交与集成。** 2026-09-16 提交代码、测试及同步文档并合入本地 main；不包含用户独立 README，未推送远端。

#### 18.9 确认后的实现记录（2026-09-12）

工作树：`.worktrees/m4-task18`，分支 `feature/m4-task18`，起点 `b120d06400a14a703a44dac1a37a0b38d8110935`。此前两份审计文档的修改复制到隔离树继续维护，属于本次交付文档；主工作树原文件保留不覆盖。用户独立 README 未复制、未修改，不纳入任务提交。

实际分工与原设计的细化：

- `LayoutAliasAnalysis.{h,cpp}` 负责静态路径规范化、root span、精确 bit 区间、XOR 的整数 carry 投影、包含域内的唯一逆像。`StorageAliasCandidates.cpp` 独立承载有限 origin 初始化，避免继续扩大 Tensor 增长 closure。
- `LayoutVar` 持有 endpoint 对应的 T/root，finalize 排序只 remap endpoint ID，不存在变换与端点脱配。`LayoutRelations.cpp` 复用 endpoint footprint 与 pair proof 缓存；actual verifier 重建新图，缓存不跨 IR 生命周期。
- root 默认 linear 模板在 alias 初始化适配层按 descriptor 构造，先验证 root proof budget，再调用 target 枚举，避免超预算大 root 进入模板算术；未修改 SM90 target 的公开枚举接口。其他 target 模板只有在满足 root 原点、容量和元数据证明时保留。临时 proposal 不是图中的隐藏 root assignment。
- `RegionLayoutEdge` 引用已有 hard relation，Strict/Common 共享 stable FIFO kernel。两阶段统计分开；origin/投影/footprint/pair-proof 次数单独输出。
- vector granularity 的安全契约补充为：1 是 scalar/packed 基线；较大值须满足字节元素、对齐的 scalar 子块，或完整最内层逻辑行的连续向量块。偏移、stride 或尾部破坏保证时，推断投影逐级减小 vector；显式 hard binding 不允许静默降级。alignment 仍表示 root pointer，并在 dump 中记录 allocation/global 依据或 whole-root 声明前置条件（不是运行时证明）。
- 原矩阵“回边确实多次缩域”需要语义澄清：M3 的合法单 CTA Tensor 编码之间允许 Convertible，真实回边不会仅因编码不同而删除候选。本次测试真实 for 图的 carried/result 缩域后回边重新入队；另用 SameLayout 链独立核对多次缩域事件。没有将回边改成 SameLayout，也未增加 conversion 移动权限。

新增测试分为 `LayoutAliasAnalysisTest`、`StorageAliasIntegrationTest`、`AliasRegionTest`，以及 `layout-alias.mlir`、`layout-scf-fixed-point.mlir`。旧 `storage-conflict.mlir` 的前两例原先同时违反容量与 alias；改用容量内 reversed map 专测 alias 冲突，保留第三例 14 bytes/8 bytes 容量错误，并检查新增反例坐标诊断。

TDD/评审证据：先复现切片地址错误 `0/8/20`（应为 `40/56/88`）、非邻接冲突漏检、RelationsOnly seed 偷渡、缺失 region dump/统计；随后补齐。独立评审另发现向量保证未按实际地址验证、alignment provenance 缺失和 public graph 自身 ID 未验证；均以反例/检查补强。两项分模块复审及一次整体复审完成；整体复审保留以下跨 pass 生命周期限制，无未处理的 Task 18 范围内阻断项。

保守边界保持：65536 live points、最多 4096 affine expression nodes、受限 bit width/matrix 大小、signed 64-bit 可表达的 root bit span；预算溢出/不支持表示为 Unknown，hard seed 不得通过。ProductLayoutMap 的一般 alias 投影、动态路径、reinterpret、跨过程/region-carried memref alias 均未扩展。未运行上游运行差分或 GPU 性能实验，创新表述仍是可核验的体系差异，不是性能领先结论。

**已核验的跨 pass 限制：** whole-root binding 声明的对齐前置条件依赖该 binding 仍存在于 actual IR。`layout_view` 保持 Pure，现有 canonicalize/DCE 可能删除未被使用的 root binding；若 sibling child 仍使用该对齐保证，`infer → canonicalize → infer` 会因缺少 root evidence 保守报错，而 `infer → infer` 通过。新增 `layout-alias-contract-lifetime.mlir` 固定此行为。需要跨任意规范化保留对齐时，应使用 allocation/global 显式对齐依据或保留实际 whole-root 契约；Task 22 需设计 durable contract。Task 18 不伪造副作用、不放宽 actual-only verifier，也不静默修改显式 hard binding。

回归维护记录：新增 own-ID 不变量后，M3 的乱序构图测试需在 reverse constraints 后重新赋予合法数组 ID；测试仍比较同一 assignment/conversion use。该调整修复测试构造，不放宽生产不变量。最终全量验证此前曾为 102/103；修正后复跑为 103/103，再加入 root-budget 回归进入最终 Gate。

**最终 Gate（2026-09-12，均 exit 0）：**

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
git diff --check
```

结果：**30/30 lit、104/104 unit（15 suites）、4/4 CTest**；相对 M3 基线新增 3 lit、30 unit。命令均在隔离工作树执行；CTest 在目标链接完成后运行。日志中的预期负例诊断不代表测试失败。

| 验收组                                                                                                       | 直接证据                                                                                                              |
| ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| cast/subview、降秩、packed、offset=3、XOR carry、逆切片限制、Unknown                                         | `LayoutAliasAnalysisTest` 14 tests，含独立逐点地址预期与失败域                                                      |
| root-relative 物化、全部 3 pairs、缓存复用、RelationsOnly counting target、篡改/rollback、向量及 root-budget | `StorageAliasIntegrationTest` 5 tests；4×8 切片结果 40/56/88，offset=3 lit 结果 52/100                             |
| region tuple/ID/方向、Strict/Common 完整账本、失败计数、确定性与回边                                         | `AliasRegionTest` 11 tests；12 种构造顺序相同 assignment/统计；真实 for 9 pops ≤ 10 event bound ≤ 18 static bound |
| 可运行 pass 与重复物化                                                                                       | `layout-alias.mlir`、`layout-scf-fixed-point.mlir`；SCF 一次/两次 pass 输出逐字 diff 相等                         |
| 前置契约被规范化删除                                                                                         | `layout-alias-contract-lifetime.mlir`：重复 infer 成功，插入 canonicalize 后缺失证据被明确拒绝                      |

#### 18.10 本地集成记录（2026-09-16）

- 用户明确授权提交当前修改并合入 main；未要求 push。
- `05e8cfa`：先独立提交主工作树中的 M4 审计/Task 18 设计基线，保留审计先于实现的历史。
- `9c28b67`：提交 Task 18 实现、测试和三份同步文档。
- 使用 `git merge --no-ff --no-commit feature/m4-task18` 集成。代码无冲突；两份审计文档冲突采用核对后的 Task 18 最新版本，审计原文可由 `05e8cfa` 恢复。
- 合并前在功能工作树重跑 Gate，合并后在 main 工作树重新构建及验收；两次均为 30/30 lit、104/104 unit、4/4 CTest。`git diff --check` 通过。
- `guoqiao/inference_detail/README.md` 未修改、未暂存、未提交；本次没有 fetch/pull/push 或远端分支操作。

### Task 19: 迁移 Copy、Fill 和 Parallel 约束

**状态（2026-09-16）：具体契约 v1 已获用户书面确认，§19.8 A–F 已完成，独立复审及全量 Gate 通过。代码尚未提交或推送，实际接口、设计适配和支持边界见 §19.9。**

#### 19.1 当前代码审计与设计取舍

审计起点为 `ef85a0d`，直接在主源码目录工作，不新建 worktree。本任务不自动 commit/push。

- `LayoutPropagation.cpp::collectLayoutConstraints` 已收集整块静态 Copy，但只接受直接 `layout_view` 操作数；现有 Storage–Storage `StorageAccess` 在 `LayoutRelations.cpp` 中要求 map 相同。这是 M2 的受限实现，不是复制操作的一般正确性条件。
- `FriskOps.td` 的 Copy 和 Fill 都只有 MemRef 操作数。不能按 memory space 把某个 MemRef 暗中当作 Tensor，也不能凭空插入一个无法在最终 IR 中找到的执行布局变量。
- `FillOp::verify` 当前只接受与元素类型匹配的 FloatAttr；本任务不顺带扩展整数 Fill 或任意标量表达式。
- `ParallelOp::inferLayout` 保留旧的 DenseMap 递归路径；新 `FriskInferLayoutsPass` 当前已经走 collect/propagate/solve/verify/materialize，不调用旧方法。本任务补充隔离证据，不把现有隔离重复计为新增能力。
- `Ownership`、`ResourceLimit` 目前有枚举名，但 bootstrap solver 尚不执行其语义。仅生成这两种约束并打印出来，不算完成迁移；传播、求解和 actual-only verifier 必须共同执行它们。

方案比较：

| 方案 | 取舍 | 决定 |
| --- | --- | --- |
| 继续要求 Copy 两端布局相同，Fill 只检查存储单射 | 改动小，但没有表达真正的执行者，也错误限制独立存储的复制 | 不采用 |
| 将 Copy/Fill 改为 Tensor 新算子族 | 可直接使用 Tensor encoding，但扩展前端、结果类型和旧调用迁移范围 | 本任务不采用 |
| 保留现有 MemRef 算子，新增实际可物化的执行布局与写入者契约 | 存储与执行分布分开，能接入统一约束和独立核验 | 采用 |

#### 19.2 公共表示：存储布局与操作执行布局分开

每个 Copy/Fill 新增一个操作级 Distributed 执行变量，逻辑 shape/dtype 来自完整复制/填充区域。内部可用同 shape 的 RankedTensorType 描述布局域，但不新增虚构的 Tensor SSA value，也不改变 MemRef 类型。

选定结果写回原操作的以下属性，并由 Op verifier 检查属性类型与基本格式：

```text
frisk.execution_layout : DistributedEncodingAttr
frisk.execution_threads : i64
frisk.writer_policy : "all" | "first_owner"
frisk.vector_bytes : i64
```

- 无这些属性的旧 IR 仍可作为推断输入；推断成功后四项必须齐全。
- 显式给出的属性是硬要求。不能覆盖显式 encoding、改变显式 writer policy，或静默降低显式 vector width。
- 不存在 enclosing Parallel 时，默认执行环境为单 CTA、32 线程，并在操作上物化 `execution_threads = 32`；显式给出的合法线程数可替代默认值。
- 存在 enclosing Parallel 时，执行线程数必须等于该作用域的 `threads`；不能用操作上的属性覆盖外层线程数。
- LayoutVar 增加专门的操作级绑定标记，区别于 Value、OpOperand use 和函数结果槽位；stable identity 包含所属操作的稳定位置及属性角色，不依赖指针排序。
- MaterializeLayouts 对该变量写操作属性，actual-only verifier 从同一操作属性读取实际 assignment；不得沿用候选缓存或想象未物化的执行布局。
- Writer policy 和 vector width 在本任务中是明确输入/确定性默认契约，不作为完整性能成本搜索的新维度。默认 vector width 为 1；默认 writer policy 为 `first_owner`。
- 新增执行变量继续受 M3 候选生成边界约束：非零 rank、各维为大于 1 的静态二次幂。不能因 Storage 端可表示任意静态 shape 就宣称执行端也支持；不满足时明确拒绝，不 padding 或修改 shape。这是新增执行模型的首版限制，须在迁移兼容性测试中单独记录。
- bootstrap solver 的每连通分量 8 个变量、每域 4 个候选限制保持不变；执行变量计入限制。不为通过测试而隐藏变量或任意截断到前 4 个候选。

#### 19.3 Copy：逻辑对应，不是物理地址相等

支持边界仍为：同 shape、同元素类型、静态非空整块区域，src/dst 为直接 `layout_view` 结果，两个 copy map 均无输入及输出，indices 为空。被绑定的 MemRef 可以是 Task 18 已支持的静态切片；不扩大到 Copy 自带动态 indices、部分区域或 Local MemRef 冒充寄存器布局。

设执行布局为 `D(h)=q`，源和目的存储布局为 `S_src(q)`、`S_dst(q)`，复制契约为：

```text
同一个逻辑坐标 q：读取 S_src(q)，写入 S_dst(q)。
```

建立两条带明确角色的 `StorageAccess`：执行布局→src 为 Read，执行布局→dst 为 Write；建立执行变量上的 Ownership 与 ResourceLimit。不得建立 src/dst 的 AliasLayout，也不得仅因复制就添加 Storage–Storage SameLayout。

例如两块独立的 `2×2xf32` 存储，源地址公式为 `8i+4j`、目的为 `4i+8j`，只要各自合法且执行访问满足契约，Copy 可以成立。它不是把两个地址公式声明为相等，而是按逻辑元素搬运。

双向信息利用限定为候选准备：已知 src 可向 dst 提供有限合法存储方案，已知 dst 同样可向 src 提供方案；跨不同 root 不直接复制绝对地址或对齐声明。只允许可证明的地址原点重定位和目标域重绑定，重新检查目标 root 容量、单射性、对齐与向量访问。无法证明的生成建议丢弃；显式绑定不得静默丢弃。不同而合法的显式两端布局不能因候选传播被强绑为相同。

与 Task 18 相同，候选准备结束后冻结域；不把双向生成放进删减 worklist，不做无界来回投影。双向测试检查“对端获得合法候选且解满足访问关系”，不再把两个 Attribute 相等作为通用正确性标准。

Copy 的读写依赖不是 alias 布局一致性的替代品。同一已知 root 的区域如存在非恒等重叠，首版拒绝，避免将可能需要临时缓冲或有序搬运的操作当成无序并行复制；完全不相交或逐元素相同的视图可接受。不同 root ID 不构成运行时 NoAlias 证明；独立 MemRef 参数间的不重叠要求属于 Copy 的调用前置契约，必须在文档和诊断中与已证明的同 root 信息区分，不宣称本任务完成跨过程 alias 分析。

#### 19.4 Fill 与 Copy 的实际写入者契约

`replication` 描述同一逻辑元素的硬件持有者数量，不直接等于实际写入次数。给执行坐标规定与 map 输入顺序无关的规范顺序：

```text
(cta, warp_group, warp, lane, register) 的字典序。
```

两种 writer policy 的含义：

- `all`：每个持有者都执行写入。只有每个 live logical point 恰好有一个持有者时合法；replication > 1 的 all-writer store 拒绝。
- `first_owner`：对每个逻辑坐标只允许规范顺序最小的持有者执行写入，其他持有者不写。必须精确证明覆盖完整、被选写入者唯一；这一选择显式保存在实际 IR，不能只留在 solver 中或假定未来 lowering 会自动处理。

例如 32 线程填充 4 个元素，执行布局可能每个元素有 8 个持有者。`all` 会重复写入，应报错；`first_owner` 明确选出每个元素的一个写入者，才允许通过。Copy 的重复读取可以合法，但目的写入仍必须经过同样的 writer policy 核验。

Fill 建立执行布局→目标存储的 Write StorageAccess，以及 Ownership/ResourceLimit；标量常数不创建布局变量。执行坐标枚举及逻辑域验证各限制为最多 65536 点，乘法/位移先做溢出检查。超限返回 Unknown/unsupported，不抽样证明。

这补充了原计划“禁止 replicated store”的准确含义：禁止同一元素的多写入者，不禁止带显式唯一写入策略的 replicated read/持有布局。首版不实现 reducer 的 partial addend、原子写入或跨操作写竞争分析。

#### 19.5 向量访问契约

`vector_bytes` 仅接受 `{1,2,4,8,16}`。它是操作的执行访问宽度，不等于 StorageLayout 的 `vector_granularity`，不能只比较两个数值大小就认为合法。

- 宽度 1 是标量/packed 基线；较宽访问首版仅支持字节整齐的标量元素。
- 必须根据 `S(D(h))` 验证每个实际执行者的连续寄存器元素形成完整向量块，物理地址连续、块起点满足绝对对齐、无越界和尾部缺失。元素大于向量宽度时只允许可证明对齐的标量子块。
- Copy 两端都需证明同一执行分组有效，不能源端连续就默认目的端连续。Fill 检查唯一写入者选择后的分组，不能把由不同线程负责的元素拼成一个向量块。
- 对齐证据沿用 Task 18 的 root provenance。操作要求不能提升根指针的已知对齐。
- 不满足硬 vector width 的候选在冻结后的传播中删除；显式编码没有其他选择时报告冲突，不修改 shape、allocation 或显式属性。候选的独立合法性证明与操作访问证明分别记录来源。

#### 19.6 Parallel：执行环境约束，不是最终布局生成器

保留现有 Parallel 的 Kernel 父操作与 region 结构，不引入新的嵌套 Parallel 语义。`ranges` 的静态正范围和区域参数契约由 verifier 检查，不把它们误当作 Copy/Fill 的 tile shape。

首版支持 `threads ∈ {32,64,128,256,512,1024}`，单 CTA。Distributed topology 的线程数按 `lane × warp × warp_group` 计算，不乘 register。要求 lane=32、CTA=1，线程数与作用域精确一致；输入格式错误或乘积溢出直接诊断。

128 线程的默认候选采用 lane=32、warp=4、warp_group=1，但这只是候选生成所用的资源环境，不代表自动选择 WGMMA。显式合法拓扑可以采用等价的分组方式，由实际线程乘积和 map 契约共同验证。

区域内 Copy/Fill 执行变量以及当前支持的 Tensor 值/具体消费边界都受最近的 Parallel 环境约束。区域外生产的 Tensor 不能只因跨边界使用就改写生产者编码；必要转换仍在实际 use 上显式物化。未建模的 Tensor region 边界继续明确拒绝，不隐式扩大 M3 支持范围。

ResourceLimit 是真正的硬约束：影响候选筛选、求解可行性和最终核验。目标模型依据线程环境生成有限拓扑候选，不再只生成固定 32/128 线程后指望其他线程数自行成立。Parallel 自身不调用旧递归推断，也不直接写最终执行布局。

#### 19.7 求解、物化、失败语义与文件分工

通用规则与 SM90 规则分开：collector 负责操作角色与绑定；Analysis 中的纯 helper 负责有界执行/向量访问证明，SM90 target 依据资源环境生成候选；统一 relation evaluator 负责硬关系执行，不能在 solver 中硬编码 Copy 例外。实现采用下述 Analysis helper 分层，避免 Analysis → Target → Analysis 循环链接。

已实现的主要接口与 payload 如下：

```cpp
// OperationLayoutConstraints.h：在图 finalize 之前创建实际操作绑定。
LogicalResult collectOperationLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph,
    LayoutConstraintBuilder &builder);

LogicalResult collectParallelResourceConstraints(LayoutConstraintGraph &graph);

// ExecutionLayoutProof.h：不改 IR 的有界证明，返回 Proven/Disproven/Unknown。
LayoutProof proveExecutionOwnership(
    DistributedEncodingAttr execution, RankedTensorType logicalType,
    StringRef writerPolicy);
LayoutProof proveExecutionVectorAccess(
    DistributedEncodingAttr execution, const StorageAliasInfo &storageInfo,
    const StorageAliasFootprint &storage,
    unsigned vectorBytes, AccessKind access, StringRef writerPolicy);

struct OperationExecutionBinding {
  int64_t threads = 32;
  unsigned vectorBytes = 1;
  std::string writerPolicy = "first_owner";
};
// LayoutVar 新增：optional<OperationExecutionBinding> operationExecution;
//                 int64_t requiredThreads = 0;
// LayoutConstraint 新增：int64_t requiredThreads = 0;
```

`CopyAccess` 是新增的 Storage–Storage 硬关系：核验同 shape/dtype、合法 footprint，以及同 root 的逐点恒等或完全不相交。它也供既有 `StorageAliasCandidates` 有限 origin 准备遍历使用；投影通过 `rebaseStorageCopyCandidate` 重建目的地址原点/容量，并削弱到目的 root 实际具备的对齐保证。它不等于 `AliasLayout`、`SameLayout`，也不证明不同 MemRef 参数运行时不重叠。

新增/修改职责：

- `include/Dialect/Frisk/IR/FriskOps.td`、`lib/Dialect/Frisk/IR/FriskOps.cpp`：操作属性及语法/类型 verifier，Parallel 资源输入检查；保留旧 adapter 方法。
- `include/Dialect/Frisk/Analysis/OperationLayoutConstraints.h`、`lib/Dialect/Frisk/Analysis/OperationLayoutConstraints.cpp`：Copy/Fill/Parallel 收集与执行绑定，替代 LayoutPropagation 中内嵌的受限 Copy 收集器。
- `include/Dialect/Frisk/Analysis/ExecutionLayoutProof.h`、`lib/Dialect/Frisk/Analysis/ExecutionLayoutProof.cpp`：纯执行证明 helper；未新增原提纲中的 SM90CopyConstraints 文件，数学证明不依赖目标候选枚举。
- `LayoutConstraint.{h,cpp}`、`LayoutSolver.h`、`LayoutRelations.{h,cpp}`、`LayoutPropagation.cpp`、`LayoutVerifier.cpp`：操作级身份、带类型的 unary/binary payload、graph invariants、冻结后的硬约束删减、solver/actual verifier 共用关系语义。
- `SM90DistributedCandidates.cpp`：依据线程环境生成有限执行候选；复用未改动的 `StorageAliasCandidates.cpp` 调度 CopyAccess 的有限双向建议；RelationsOnly 禁止调用这些生成逻辑。
- `MaterializeLayouts.cpp`：在 detached module 中物化操作属性，与现有 Tensor/Storage 改写一起核验后原子提交。
- 对应 CMake 文件、新单元测试及 lit 文件：新增源文件和测试注册；不改无关构建依赖。

所有新绑定都要在 graph finalize 中保持稳定身份。新增 unary 约束必须接入 Strict/Common 的计数与调度，不得通过“关系没有第二个端点所以默认成功”。实际核验不允许为缺失属性补默认值、重新枚举候选或选择 writer policy。

任何失败均保持原 IR 不变。缺失或篡改 execution_layout/threads/writer_policy/vector_bytes、根对齐证据丢失、资源或写入者冲突都必须有对应负例。

#### 19.8 实施顺序与验收清单

使用 writing-plans、test-driven-development 和分阶段执行/复审流程；保持用户指定的源目录工作方式，不创建 worktree。先完成本设计的书面复核，再进入下面的红灯测试与实现。

- [x] **A：执行绑定与 Ownership。** 在 `CopyFillConstraintTest.cpp` 先测试未物化绑定被拒绝、32 线程/4 元素的 all-writer 失败与 first-owner 成功、缺失元素覆盖失败；再实现操作级变量、Ownership 和 actual-only 属性读取。
- [x] **B：Copy 双向与向量访问。** 先测试 src 已知/dst 未知、dst 已知/src 未知、两端合法异构存储、不同根偏移不直接复制、同 root 非恒等重叠拒绝；向量宽度测试 3 非法、16 字节未对齐失败、目的不连续失败、标量基线成功。再替换旧 Copy 收集器并实现有界证明。
- [x] **C：Parallel。** 先测试 128 线程候选、32/128 显式编码冲突、其他合法线程数、非法线程数和 Tensor 消費边界，再接入 ResourceLimit 与目标候选环境。
- [x] **D：独立核验与事务性。** 从合法结果分别篡改四项操作属性，测试 actual verifier 拒绝且 target enumeration 次数为 0；测试推断一次/两次输出一致，失败前后原 IR 逐字一致。
- [x] **E：旧路径隔离与回归。** 用真实 pass 调用检查 legacy Parallel inference 的调用次数为 0，单独运行旧 adapter 测试；统计只反映实际调用，不添加永远为零的伪计数。生产新路径与 legacy 行为不要求文本一致，差异按逻辑坐标和硬契约解释。
- [x] **F：全量 Gate 与文档。** 运行以下命令，记录实际新增测试数量、支持边界及设计偏差；同步比较文档，不以 M3/Task 18 的旧测试数代替本任务验收。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
git diff --check
```

红灯测试先运行新增测试过滤器及新 lit 文件，确认因缺失行为而失败，再编写生产实现。完成标准是全量测试通过、独立复审无未处理阻断项、两份长期文档与代码一致；未运行上游运行差分或 GPU benchmark 时不得声明行为全面兼容或性能领先。

以下保留原始迁移提纲以追踪计划演进，复选框不再代表当前验收状态，以 §19.8 A–F 与 §19.9 为准；若与上面 v1 契约冲突，以上面的具体契约为准。特别是 Copy 的 SameLayout 仅能用于真正同编码的执行槽位，不能再用于要求两个独立存储地址公式相同；Step 6 的 git 命令需用户另行授权。

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `test/Transforms/infer-copy-fill-parallel.mlir`
- Create: `unittests/Dialect/Frisk/Layout/CopyFillConstraintTest.cpp`

**Interfaces:**

- Produces：

```cpp
LogicalResult collectCopyConstraints(CopyOp op,
                                     LayoutConstraintBuilder &builder,
                                     LayoutTarget &target);
LogicalResult collectFillConstraints(FillOp op,
                                     LayoutConstraintBuilder &builder);
LogicalResult collectParallelConstraints(ParallelOp op,
                                         LayoutConstraintBuilder &builder);
```

- [ ] **Step 1: 写正反向推断红灯测试**

Copy 分别测试已知 src 推 dst、已知 dst 反推 src；Fill 测试 unique writer/replication；Parallel 测试 threads=128 seed warp-group topology，threads 与 encoding 不一致时报错。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，collector 尚未覆盖三类 Op。

- [ ] **Step 2: 实现 Copy 约束**

Copy 根据 operand role 创建 Storage 或 Distributed var，建立 `StorageAccess`/`SameLayout`/`Preference`。首版 vector width 只接受 `{1, 2, 4, 8, 16}` bytes，alignment 不满足时删除候选，不修改 shape。

- [ ] **Step 3: 实现 Fill/Parallel 约束**

Fill 对每个 live logical point要求唯一 writer，允许明确 replicated read 但不允许 replicated store。Parallel 将 thread count/topology 写入 ResourceLimit/Ownership，不直接生成最终 layout。

- [ ] **Step 4: 删除 Parallel 的递归 `DenseMap` 调度使用**

保留旧方法定义供 adapter tests 使用，但生产 `FriskInferLayoutsPass` 不再调用 `ParallelOp::inferLayout`。增加测试统计 legacy 方法调用次数为 0。

- [ ] **Step 5: 运行迁移测试**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: Copy/Fill/Parallel 新测试和 legacy tests 均 PASS。

- [ ] **Step 6: 提交基础 Op 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: migrate copy fill and parallel constraints"
```

#### 19.9 实施记录与设计适配（2026-09-16）

实现起点 `ef85a0d`，在源目录 main 工作，未创建 worktree，未自动提交或推送。以下内容是 Task 19 的实际增量，不表示 M4 整体完成。

| 已实现能力 | 实现位置 | 直接验证 |
| --- | --- | --- |
| 操作级执行变量、四项实际 IR 契约 | `OperationLayoutConstraints`、`LayoutConstraint`、`MaterializeLayouts` | Fill 推断后可读取全部属性；删除任意一项、篡改 encoding/threads/writer/vector 均拒绝；actual verifier 的目标枚举次数为 0 |
| Copy 的读写坐标关系与有限双向建议 | `LayoutRelations` 的 CopyAccess/StorageAccess、`LayoutAliasAnalysis::rebaseStorageCopyCandidate` | 2×2 f32 的行主序源/列主序目的可复制；已知任一端可向另一端提供重定位候选；同 root 恒等和不相交成功，位移重叠失败 |
| 实際写入者与向量访问证明 | `ExecutionLayoutProof` | all-writer replica 冲突、first_owner 规范选择、缺失覆盖、输入名字重排、寄存器向量连续性/对齐/尾部、packed 标量基线、65536 点上界 |
| Parallel 线程环境与实际 use 转换 | `collectParallelResourceConstraints`、`SM90DistributedCandidates` | 六种合法线程数；128 线程作用域拒绝 32 线程显式属性/编码；外部 32 线程 Tensor 在内部消费边界插入转换，生产者保持原编码 |
| 硬约束闭环和可核验统计 | Strict/Common worklist、bootstrap solver、actual verifier | Ownership/ResourceLimit 不再是未执行的枚举名；宽向量要求真实删除候选；冻结候选与 monotone stats 上界保持有效 |
| 旧路径隔离与文本回放 | 真实 legacy 调用计数、Kernel/Parallel/Copy printer/parser | 显式调用 legacy 后计数增加；运行新 pass 后不增加；打印后重新解析、第二次推断与第一次逐字一致 |

具体例子：在 `threads=128` 的 Parallel 中对 `memref<4xf32>` 做 Fill，默认执行布局可以让每个元素有 32 个持有者，但物化的 `writer_policy="first_owner"` 只允许每个元素的首个规范持有者写入。把属性显式设为 `all` 会因重复写入被拒绝。128 线程对应的默认 topology 是 `[register=1, lane=32, warp=4, warp_group=1, cta=1]`，不是已选择 WGMMA 指令。

适配与复审记录：

- 证明模块放入 Analysis，collector 和 target 分开；没有机械新增一个反向依赖 Target 的 SM90 helper。
- 为支持真实 Kernel/Parallel 输入，Task 18 的静态存储根识别补充了类型与签名一致的 Kernel 入口 MemRef；没有扩展到未知 region 参数或动态别名。
- 回放测试暴露并修正旧 Kernel 入口参数名丢失、空签名解析、Parallel induction 参数类型/终结符、Copy 双重解析方括号和 extents/属性打印位置问题。Copy 的 map 使用 ODS getter 获取，extents 用 `srcExtents`/`dstExtents` 的 DenseI64Array 保存；不是扩大 Copy 的推断支持范围。
- Copy 候选重建后的地址上界按目的 descriptor 的实际可达容量规范化。旧 lit 中 16/160 改为 14/134 的两例仍保持相同物理地址公式，并未扩大 allocation 或更改 shape。
- 对齐负例保留 Task 18 的完整根 binding 前置契约语义：仅删除 allocation alignment，若完整根 `layout_view` 仍显式声明相同保证，不能称为证据全丢失。测试同时将 allocation 保证移除、whole-root storage binding 的 alignment/vector_granularity 降为 1，并保留操作 vector_bytes=16，要求 actual verifier 拒绝且不枚举候选。操作自己的向量需求不能充当根指针对齐依据。
- 纯证明独立复审指出：相同 root 不足以证明 footprint 属于当前 view；标量宽度也必须检查区间不重叠。新增两个先红后绿的反例，核对 view→root 坐标并在宽度 1 快速返回前核验物理区间。复审已确认这两项解决。
- 全任务独立复审补出了零输入 Kernel 回放反例，已修正；最新代码复审无未处理的 Critical/Important 项。首次核心红灯涉及未物化绑定、all-writer 误接收及 Copy 强制相同地址图；不是只编写通过测试。

验证命令沿用 §19.8，2026-09-16 最终结果：

- 构建 `FriskLayoutUnitTests check-frisk frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test frisk_memory_effect_test` 成功。
- **129/129 单元测试通过**（17 个 suite）：相对 Task 18 的 104 项新增 25 项，其中 `CopyFillConstraintTest` 16 项、`ExecutionLayoutProofTest` 9 项。
- **32/32 lit 通过**：相对 Task 18 新增 `infer-copy-fill-parallel.mlir`、`infer-copy-fill-parallel-errors.mlir` 两份，覆盖图输出、属性物化、打印解析后二次推断逐字一致和错误诊断。
- **4/4 CTest 通过**，包括旧 attribute、reduce、layout pass 与 memory effect 回归；单元中的 legacy adapter 测试一并通过。
- `git diff --check` 无错误。构建日志中的既有 Ninja 日志恢复提示未阻断构建；没有借此删除或重建用户工作区。

新增单元位于 `unittests/Dialect/Frisk/Layout/CopyFillConstraintTest.cpp` 和 `ExecutionLayoutProofTest.cpp`。负例测试主动产生的 expected error 不代表测试失败，以测试程序退出码和汇总为准。

仍不支持：动态/部分区域 Copy、执行域非二次幂或含 extent=1、一般跨过程 NoAlias 证明、跨操作竞争分析、GPU 写入谓词/向量指令生成、TMA/cp.async、GEMM/Reduce 迁移和完整性能成本求解。`first_owner` 是已经验证并物化的执行契约，未来 lowering 仍须消费它。未运行 TileLang/Triton 执行差分或 GPU benchmark，本轮不宣称全面兼容、独创性已证明或性能领先。

### Task 20: 增加内部 Tensor MMA Op 并迁移 Gemm 约束

**状态（2026-09-19）：已按用户确认的以下 SS＋RS 契约完成实现与测试。全量 165 unit / 34 lit / 4 CTest 通过；实际实现、复审修正与边界见 §20.13。Task 21/22 及 GPU lowering 未包含在本次交付内。**

#### 20.1 实现前审计与方案选择

工作基线是 `ef85a0d` 加当前未提交的 Task 19 实现与文档，而不是只有该提交的干净源码。在 `/home/baopeihua/frisk` 的主工作区继续，不创建 worktree，不自动 commit/push，不改写独立汇报文档。

本轮只读检查发现：

- `FriskLayoutOps.td` 尚无 `MmaOp`；`FriskLayoutAttrs.td` 只有通用 Distributed/Storage encoding，没有实现设计目标中的 Mma/DotOperand 专用 encoding。
- `GemmOp::verify` 要求 A/B/C 同元素类型，且按未转置形状检查维度；不能作为新操作的数学 verifier。
- 旧 `GemmOp::inferLayout` 用 Local MemRef 表示寄存器 fragment，直接更新 `DenseMap<Value, Attribute>`；新接口必须使用 Tensor SSA 和图中的实际消费位置，不能机械复用旧容器。
- `InstructionContract` 目前只有枚举名。现有 relation evaluator 主要执行 unary/binary 关系，不能把 A/B/accumulator 的联合契约拆成若干可能不相容的两两“存在某个方案”。
- Task 19 的操作执行变量专门绑定 Copy/Fill；MMA 已有 Tensor accumulator/result，不借用这类变量虚构另一份结果编码。
- `LayoutTarget` 目前只枚举/核验单个布局，没有指令级联合接口；需要补充有界生成与纯核验的分层，避免 Analysis 反向链接 SM90。
- Task 22 已单独安排旧 Local MemRef 生命周期 normalization；本任务不提前自动改写旧 `frisk.gemm`。

采用“内部 Tensor MMA ＋显式指令契约 ＋通用 canonical maps”的方案。只实现 SS 会遗漏寄存器 A 的迁移验证；同时加入旧 IR 自动转换及 GPU pipeline 会跨越 Task 22 和 lowering 的边界。本任务完成 SS、RS 的布局正确性闭环，不生成 WGMMA/fence/commit/wait 指令序列。

本轮重新核对了既有审计快照的相关源码，HEAD 分别为 TileLang `5e149e31674658f94779c7d0c6039549a1853123`、Triton `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c`；没有刷新到浮动 main，也没有运行上游测试。具体证据与取舍见 §20.10 及比较文档的 Task 20 设计增量。

#### 20.2 `frisk.mma` 的数学语义、类型与副作用

新操作固定表达：

```text
result[i,j] = init[i,j] + sum(k = 0 .. K-1) A_eff[i,k] * B_eff[k,j]

A_eff[i,k] = trans_a ? A[k,i] : A[i,k]
B_eff[k,j] = trans_b ? B[j,k] : B[k,j]
```

这定义逻辑对应，不承诺浮点归约顺序或比目标指令更强的数值精度。没有隐含的 alpha/beta、清零开关、额外输出写回或动态 clear 条件。

- 操作数顺序固定为 A、B、init，唯一结果为 result。A/B 在类型层可为 RankedTensor 或 MemRef；init/result 必须为 RankedTensor。
- 首版数学 verifier 接受静态正 extent 的 rank-2 浮点矩阵；A/B 元素类型相同，init/result 的元素类型相同且位宽不小于输入。特定 dtype 是否有目标指令由 target 判断。
- 属性 `m/n/k` 为正的 signless i64；`trans_a/trans_b` 为 BoolAttr，缺省 false；`policy` 为已有 GemmWarpPolicy，缺省 Square。维度按上述逻辑转置精确匹配；矩形转置负例必须覆盖。
- init/result 的逻辑 shape 必须都是 `[M,N]`。输入阶段允许缺失 encoding 或两者 encoding 不同；最终通过 init 的实际消费位置转换，使该位置与 result 的 accumulator encoding 一致，不能直接改写 init 的其他使用者。
- MemRef 输入只有 Read effect，不写 A/B、不分配存储；Tensor 输入是 SSA 依赖，result 不是 MemRef 写入。带 MemRef 输入的 MMA 不可标为 Pure，不可因漏报 memory effect 而跨越写操作随意移动。
- `clear_accum=true` 的未来迁移应传入显式零 Tensor，普通累加传入实际旧 accumulator；动态 clear 的语义与转换留给 Task 22。不得用 `tensor.empty` 充当已初始化的零。
- 这是高层同步数学操作：result 在该操作之后可用。未来异步 lowering 必须建立这一可见性，当前布局验证不声称已经证明 pipeline 同步安全。

原 Task 20 示例仍有效：`A:128×64xbf16`、`B:64×128xbf16`、`init/result:128×128xf32`。示例只展示操作类型，不省略测试所需的合法 shared layout、根对齐契约和目标环境。

#### 20.3 SM90a 首版支持集与明确拒绝项

数学操作不内嵌 PTX mnemonic；目标规则按以下受限表选择契约：

| 项目 | 本任务设计边界 |
| --- | --- |
| 目标 | 最近的 `frisk.target` 必须明确为 `"sm_90a"`；缺失、`sm_90` 基础能力或其他目标在 MMA target 校验中拒绝。不改变 Task 19 普通操作的默认目标行为 |
| 数据类型 | A/B 同为 f16 或同为 bf16；init/result 为 f32。暂不接入 f16 accumulator、TF32、FP8、整数或稀疏路径 |
| SS | A/B 为 Shared MemRef，必须是直接 `layout_view` 结果；根和静态别名链沿用 Task 18 |
| RS | A 为 RankedTensor，B 为上述 Shared MemRef；禁止将 Local MemRef 当作 Tensor fragment |
| Tensor B | 通用类型语法可表达，但本 SM90a target 拒绝。不得偷偷插入 shared allocation/store 把它变成另一种路径 |
| 执行域 | 保留现有 Distributed 的非零 rank、每维大于 1 的二次幂限制；M 是 64 的倍数，K 是 16 的倍数，N 至少为 8 |
| 原子与拼接 | 原子为 `64×n_atom×16`，`n_atom ∈ {8,16,32,64,128,256}`；整 tile 可以在 M/N/K 上重复原子，不 padding、不改 shape、不处理尾块 |
| 线程 | 单 CTA，`threads ∈ {128,256,512,1024}`；外层 Parallel 的线程数必须一致，无 Parallel 时缺省 128，显式合法线程数可替代缺省值 |
| 控制流 | 保留 M3 的 SSA/SCF 布局规则；Task 20 不新增 warpgroup 分歧、动态线程参与或异步 token 分析 |

硬件支持不等于 Frisk 当前表示支持：例如硬件存在的 `n=24` 原子不属于上述二次幂子集，不能通过改 shape 或忽略 encoding verifier 来接受。

SS 支持数学上的 `trans_a/trans_b` 四种组合。RS 的 `trans_a=true` 通过 A fragment 的逻辑坐标交换和必要的输入布局转换表达，不伪造硬件 RS 的 A-transpose immediate。逻辑 transpose 与硬件 major mode 是两层信息：major mode 从最终地址关系解码，不能把 BoolAttr 原样当作 PTX immediate。

#### 20.4 线程组织、policy 和寄存器坐标

每组固定 4 个连续 warp，lane extent=32、CTA extent=1，warp_group extent=`threads/128`。线程编号解释为 `((warp_group * 4 + warp) * 32 + lane)`，不能只检查线程数乘积而放过不符合组边界的任意拓扑。

设 group 数为 G，在有限因子对中选择 `[gM,gN]`，满足：

```text
gM * gN = G
M % (64*gM) == 0
N % (8*gN) == 0
```

- FullRow：在合法因子对中取最大 gM。
- FullCol：取最大 gN。
- Square：最小化整数值 `abs(M/gM - N/gN)`；相同值按 `(gM,gN)` 字典序稳定选择。这是 Frisk 首版确定性几何策略，不复现旧代码的浮点打分，也不是性能最优结论。
- 分组后取能够整除 `N/gN` 的最大受支持 `n_atom`。记录重复次数 `[M/(64*gM), N/(n_atom*gN), K/16]`；不搜索全部指令宽度组合。
- 没有合法分组就报错；不截断线程数、不忽略 policy。128 线程时只有 `[1,1]`，不同 policy 得到相同分组是合法退化情形；256 线程的 `128×128` 用例须展示 FullRow/FullCol 的差别。

group 坐标固定为 `groupM = warp_group / gN`、`groupN = warp_group % gN`，各 group 负责连续的 `[M/gM,N/gN]` 子块。原子外层寄存器槽按 accumulator 的 `(mRepeat,nRepeat,innerSlot)`、RS A 的 `(mRepeat,kRepeat,innerSlot)` 字典序展开，最右侧变化最快；innerSlot 顺序取硬件原子规范。K 重复更新同一 accumulator，不新增一份 result 槽。线程环境只约束 MMA 的消费槽与结果；不得为满足指令而改写外部 A/init 生产者的显式线程/encoding 契约。

结果与 init 消费槽使用按硬件 accumulator 原子展开的 DistributedEncoding，逐点验证完整覆盖和唯一持有者。RS 的 A 使用按寄存器输入原子展开的 DistributedEncoding；不同 N 方向的 group 可以持有相同 A 元素，这种只读 replication 必须被精确描述，不能沿用 Fill 的 all-writer 判定。

**packing 的语义必须明确：**现有 Distributed map 的 `register` 坐标在 16-bit A 上表示逻辑元素槽，不直接表示 32-bit 物理寄存器编号。每个 A 原子的 8 个元素槽要按契约组合成 4 个 32-bit packed registers，低/高半字顺序纳入指令核验；f32 accumulator 每槽对应一个 32-bit 寄存器。不能把“覆盖了同一批元素”当成“寄存器顺序满足指令”。原子外层的 M/K 重复及 N-group replication 同样进入规范 map。

共享与寄存器布局优先展开为当前已支持的 canonical BitLinear/Affine 表示，不在本任务顺带放宽一般 Product map 的 pass 支持边界。单个布局的逻辑点与硬件元素槽枚举均不超过 65536，任何溢出/超限返回 Unknown 并使硬检查失败。该点数上界不是寄存器分配、spill 或 occupancy 证明；完整资源成本与调度仍属于后续任务。

#### 20.5 Shared descriptor：必须证明能访问实际存储

对每个 Shared 操作数分别执行两层检查：先通过 Task 18 的根容量、坐标、单射和别名一致性证明，再验证它能够由本任务支持的指令描述符方案表示。普通合法 StorageLayout 不自动等于合法 WGMMA operand。

设计支持 K-major/MN-major 的无 swizzle 及 32/64/128-byte swizzle 规范模板。生成阶段每个未绑定 operand 只选一个默认 major：若按其数学角色和 transpose 解释后，原二维类型的最后一维是 K，则选 K-major，否则选 MN-major；在该 major 下最多提供四种 swizzle 模板。显式另一 major 的布局仍可被精确核验，不强制改回默认值。

描述符方案包含：major、swizzle 模式、leading/stride byte offset，以及该操作数各原子调用所需的 root-relative 起点和 swizzle phase。物化的是与 runtime root pointer 组合的**静态描述符方案**，不是假装知道运行时 shared pointer 的完整 64-bit descriptor。

必须满足：

1. 所有实际原子的起点和描述符偏移可编码、对齐、无截断和溢出；不能通过掩去高位把不合法偏移变合法。
2. 对每个原子实际访问的逻辑坐标，按描述符规则重建的地址与所选 StorageLayout 的 root-relative bit 地址相等，且不越出根可访问范围。
3. 静态切片的偏移、步长和 K-panel 间距来自实际布局/根坐标关系。不能根据切片自身的 M/N extent 猜测它在更大父 buffer 中的 panel stride。
4. 不重复叠加 descriptor offset；基址非零时同样逐点核验。数学 transpose 只改变逻辑坐标解释，不直接修改已有存储。
5. 无 swizzle 需要根及原子起点满足 16-byte 对齐。为使首版静态 swizzle phase 可证明，32/64/128-byte swizzle 分别要求根具有 256/512/1024-byte 对齐保证，再根据原子 root-relative 偏移计算 phase；这是 Frisk 的保守支持边界，不是声称硬件要求所有根都如此对齐。
6. 根对齐证据沿用 Task 18：allocation/global 属性或完整根 layout_view 的显式前置契约。生成候选、子视图要求和 MMA 指令需求不得自行增强根保证；没有足够证据时丢弃该建议，显式 binding 则报错。

显式任意地址 map 只有被上述模板和有界逐点证明共同覆盖时才能通过。对于数学上合法但当前无法解码的 layout，明确报告 unsupported，不宣称它在硬件上必然不合法，也不自动新建 shared buffer 搬运。

A/B 都是只读输入，可以重叠；若有共同 root，依然执行 AliasLayout 一致性检查，但不错误套用 Copy 的非恒等读写重叠禁止规则。

#### 20.6 图中的角色、真正的联合约束与有限候选

每个 MMA 的联合端点按固定角色排序为 `[A-slot, B-slot, init-use, result]`：SS 的 A/B slot 为 Storage var；RS 的 A-slot 为 Distributed use var、B-slot 为 Storage var；init-use/result 为 Distributed var。

关系如下：

- Tensor A 的生产者 → A 消费槽：Convertible，必要时在 MMA 的真实 operand 上插转换。
- init 生产者 → init 消费槽：Convertible。
- init 消费槽 ↔ result：SameLayout，不要求 init 的所有外部使用者都变成该编码。
- 四个角色共同关联一个 InstructionContract；Distributed 角色同时受真实 ResourceLimit 约束。
- Shared A/B 的访问维度与 result 不同，不得用 Task 19 的同 shape StorageAccess 直接连接。它们的 Read effect 和访问证明属于 MMA 的角色化联合契约。

InstructionContract 不是四条独立的 RequireEncoding。候选准备阶段形成经过目标纯证明核验的有限合法 tuple 表；一条 tuple 同时包含四个角色的编码及对应指令方案。只有**同一 tuple**能够支持当前各角色选择时才成立。

候选流程必须区分生成和删除：

1. 收集显式 encoding/storage binding、操作数学属性和目标/线程环境，创建稳定角色身份；显式项是硬要求。
2. 按所选 policy/原子组织准备 MMA 专用 canonical accumulator/A-fragment 和有限 shared 建议，复用 Task 18/19 的有限 alias/Copy origin 传播；不能给 MMA 结果继续填充无关的通用 SIMT 默认布局，也不能把 target 专用建议当作用户 RequireEncoding。
3. 不沿 InstructionContract 做无界“正反向生新布局”。专用建议各生成一次，通用候选准备收敛后冻结全部域；若同一存储参与多个 MMA，收集全部已允许的有限来源，而不是按访问先后覆盖。
4. 保留每 component 8 vars、每域 4 candidates 的 bootstrap 上限。超限明确报错，不截断前四项，不移除变量来绕过限制。每个四端点契约最多检查 `4^4=256` 个编码组合；重复端点必须使用同一 assignment。
5. 每个编码组合通过确定性的 descriptor 解码获得至多一个规范指令方案；有多个等价解码时采用固定键序，不额外引入没有上界的隐藏选择。已显式绑定方案时，只核验该方案。
6. Strict/Common 在原有队列上增加 hyperedge 支持检查：某个候选若没有任何与其余当前域同时相容的 tuple，就删除；InstructionContract 即使没有 singleton 也必须执行。domain 删除会重排所有相关 unary/binary/hyperedge，旧统计公式按真实邻接关系计数。
7. Solver 对部分 assignment 检查是否仍有合法 tuple 扩展，对完整 assignment 要求整条 tuple 成立；不能沿用把所有关系拆成首端点与其他端点两两检查的循环。

合法 tuple 表、proof cache 和 provenance 只在当前 graph 生命周期内有效；finalize 重排后所有角色 ID/constraint ID 正确 remap。稳定键按 IR 位置和角色，不按指针。图不变量检查角色数量、类型、source-op、重复端点一致性和 binding 归属。

仍采用现有 bootstrap 的最少显式转换数＋稳定排序选择，不调用未完成的完整 CostVector 性能优化。policy、原子宽度和 swizzle 排序只能称确定性策略。

#### 20.7 实际 IR 上的指令绑定与分层接口

新操作的数学属性不存 PTX 字符串；所选目标实现写入独立的类型化属性：

```text
frisk.mma_contract : MmaInstructionContractAttr
frisk.execution_threads : signless i64
```

`MmaInstructionContractAttr` 的 v1 必需字段是：schema version、target feature、SS/RS form、输入/累加类型、原子 `[64,n_atom,16]`、warp-group grid、M/N/K repeat counts、A packing 规则，以及 SS 的 A/B 或 RS 的 B descriptor plan。每个 plan 保存 major、swizzle、leading/stride offsets，和以操作数原子坐标为键的 root-relative start/phase 表。表规模受上述 tile/枚举预算约束，禁止含 pointer、SSA 地址或仅在内存中存在的缓存 ID。

Tensor result/消费位置仍使用 `DistributedEncodingAttr`，Shared 仍使用 `StorageLayoutAttr`。本任务将原提纲的“Mma/DotOperand”落实为**指令角色＋可展开的通用 encoding＋操作契约**，不新增另一套 Tensor encoding 子类型；不宣称总体设计中的 `MmaEncodingAttr`、`DotOperandEncodingAttr` 类已经实现。这一选择避免无关地重写全部 M3 Tensor 类型转换，但不省略专用 fragment/packing 证明。

显式 mma_contract 是完整硬绑定，不接受随意缺字段的“部分契约”；推断不能覆盖它。未指定时可由求解生成；actual-only 验证时两项属性必须齐全，不允许补默认值或重新选择方案。

新增职责与 API 契约：

- Analysis 的 `MmaLayoutConstraints` 只负责操作角色、SSA use、数学上下文和图关系。
- `LayoutTarget` 增加独立的指令候选准备入口和纯 `verifyInstructionContract` 入口；默认实现拒绝未知 contract，不通过抽象层直接调用 SM90 函数。
- `SM90GemmConstraints` 提供候选建议、规范方案构造；`SM90MmaLayoutProof` 提供按确定方案验证 fragment/descriptor 的纯函数。这两者不得修改 IR。
- graph 内的 `InstructionLayoutContract` payload 记录角色、实际操作 binding 和有界 tuple 证据；`LayoutSolution` 增加以稳定 constraint ID 为键的 selected instruction binding。合法方案不是普通 Storage/Distributed 变量，不伪装成 Tensor shape。
- propagation/solver 只做通用有限 tuple 支持检查，不识别 `sm_90a` 字符串或直接构造 WGMMA map；最终 verifier 通过传入的 target 重新验证实际绑定。
- RelationsOnly 模式在候选生成和 tuple 枚举之前返回结构图；读取实际 IR 的 singleton assignments 和实际 mma_contract，调用纯证明。不得调用 target enumeration/方案搜索，也不得复用上次求解的合法 tuple 表。
- 未识别的指令类别依旧 fail-closed；原 `RejectsUnsupportedHardConstraint` 测试改为未知指令类别负例，不能因 InstructionContract 被接入就默认所有类别合法。

#### 20.8 物化、独立核验和失败诊断

物化在 detached module 中完成：写 Tensor encoding、插入实际 use 转换、绑定 storage layout、写 mma_contract/threads，随后进行 MLIR verifier 和 actual-only 指令/布局证明，成功后才替换原 module body。

独立验证从实际类型和操作属性重建 A/B/init/result，检查目标 feature、线程 scope、数学 transpose、policy、packing、descriptor plan 与 map 一致性。不能把自报“正确”的 descriptor 属性当成地址证明；即使 descriptor 和某一个 operand 都各自合法，也要检查它们属于同一操作、同一 shape 和同一 warp-group 分组。

诊断使用稳定规则名，例如 `mma-shape`、`sm90-mma-target`、`sm90-mma-thread-group`、`sm90-mma-fragment`、`sm90-mma-descriptor`、`mma-joint-contract`；注明 A/B/init/result 角色及 relevant dtype/shape，地址或持有者失败尽可能给出逻辑/硬件坐标。Unknown 单列为无法证明，不冒充数学反例。

任何失败保持原 IR 不变；同一输入两次推断及打印/解析后再次推断必须稳定。当前不声称发现任意语义篡改：例如用户同时合法修改数学属性与完整布局/契约，得到的是另一个合法程序；测试要篡改为确实违反当前契约的组合。

#### 20.9 测试分组与完成标准

实施采用先红后绿及独立复审，先逐个打通 SS/RS vertical slice，再扩展合法模板；不能仅新增 Op 并打印一个 contract 就标记完成。

- [x] **A：数学 IR 与副作用。** rank/正 shape、M/N/K、矩形 transpose、f16/bf16→f32、init/result 关系、MemRef Read/no Write、零 Tensor 初始化，printer/parser。
- [x] **B：独立硬件坐标 oracle。** 固定原子的 A packed halves、accumulator 坐标、不同 lane/warp/寄存器顺序；多原子 M/K 拼接与 N-group replication。测试参考表不能调用待测候选生成器构造“期望值”。
- [x] **C：SS/RS 与 descriptor。** 两种输入 dtype、所有数学 transpose 组合、K/MN major、四种 swizzle；合法及不满足支持边界的组合分类测试。覆盖父 buffer 切片的 K-panel stride、非零起点、根对齐不足、phase 错误、地址区间越界；B Tensor/Local MemRef 明确拒绝。
- [x] **D：线程与 policy。** 128/256/512/1024、group 连续性、FullRow/FullCol/Square、无合法分组、显式 threads 与 Parallel 冲突；`sm_90a` 成功、缺失 target/`sm_90`/SM80 新路径拒绝。
- [x] **E：联合关系与收敛。** 构造“每对看似都有支持，但不存在完整合法 tuple”的反例；重复端点、稳定 remap、多 MMA 共用 storage、显式绑定冲突、8 vars/4 candidates 边界及超限、最多 256 个组合和点数预算。256 是四个合法候选域的组合上界，不人为制造域合法却有更多组合的假场景；损坏或重复的 tuple payload 单独按图不变量拒绝。统计证明冻结后只删不增。
- [x] **F：实际物化。** 外部 init/A encoding 保持不变，在 MMA 消费处转换；零拷贝 shared binding；删除 contract/threads、篡改 fragment/packing/descriptor/分组后拒绝；target 生成及 tuple 枚举计数均为 0；失败 IR 逐字不变、二次推断逐字一致。
- [x] **G：legacy 分类差分。** 用旧 adapter 比较适用子集的逻辑持有关系与地址，不比较 Attribute 字面相等；旧 FP16 accumulator 用例只能作为旧行为回归，不能冒充本次 FP32 accumulator 的数值等价测试。SM80、B-local、错误 group 组织、过时 ABI 分类记录；不保留已知错误求一致。
- [x] **H：全量 Gate、独立复审和文档。** 新单元 `GemmConstraintTest.cpp`、`SM90MmaLayoutProofTest.cpp`；新增 `infer-gemm-layout.mlir`、错误/回放 lit 与必要 IR verifier 测试；同步本文和比较文档的实际实现/差异/计数。

验收至少包含 `128×128×64` 的 SS/RS f16/bf16→f32 例子及 `64×64×16` 原子例子；它们必须在未放宽既有 solver 限制的真实 pass 中成功，不只通过独立 helper。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_brief=1
ctest --test-dir build --output-on-failure
git diff --check
```

实现后记录新增测试数量、实际 Gate 和每项未支持边界；没有执行上游/GPU benchmark 时，不宣称完整数值 lowering 验证、上游全面兼容或性能领先。

#### 20.10 上游核对与 Frisk 的适配原则

本轮已读的固定快照证据：

- TileLang [`src/op/gemm.cc`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/gemm.cc)：`k_pack`/`wg_wait` 在 annotations；可能读取 C 与确定 read-before-write C 对 clear 条件有不同判断；WGMMA 不随意沿用不符合指令要求的 shared layout。Frisk 用显式 init SSA 表达旧值依赖，不复制 positional ABI 或 `completed_` 调度。
- TileLang [`src/cuda/op/gemm.cc`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/cuda/op/gemm.cc)：WGMMA 分组把 4 个 warp 视为不可拆单元。Frisk 保留这一硬边界，但用自己的显式 group grid 和稳定 policy 定义，不强求旧浮点评分结果一致。
- TileLang [`test_tilelang_cuda_wgmma_operand_layout.py`](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/cuda/test_tilelang_cuda_wgmma_operand_layout.py)：K-panel stride 从实际布局提取，切片不能按自身 extent 重建父存储间距。Frisk 将其作为 descriptor/address 一致性反例输入，不宣称该正确性要求是 Frisk 独创。
- Triton [`AccelerateMatmul.cpp`](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)：MMA encoding、accumulator 转换和 MMAv3 的 A-register/shared、B-shared 路径已有实现。Frisk 的设计差别在于将该类选择形成有限域联合硬契约，并在物化后对实际 IR 独立核验，不是声称 Triton 没有这些 operand/转换能力。

硬件规则依据为 2026-09-19 访问的 [NVIDIA PTX ISA 9.4](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)：§9.7.17.5.1 的 WGMMA fragment 与 shared canonical layout、§9.7.17.5.1.2.2 Matrix Descriptor Format，以及 §9.7.17.5.2 指令定义。实现注释和测试按该版本章节及图像文件名定位，不依赖会随版本变化的图号；已有 A 链路报告中的旧 Figure 148 不能当作新版图号。

#### 20.11 文件分工及原提纲适配

实际新增 `Analysis/MmaLayoutConstraints.{h,cpp}`、`Analysis/InstructionLayoutConstraints.{h,cpp}`、`Target/SM90/SM90GemmConstraints.{h,cpp}`、`Target/SM90/SM90MmaLayoutProof.{h,cpp}`，以及硬件、通用联合约束、真实集成单元 suite 和 lit。IR 层在 `FriskLayoutOps` 定义 MmaOp，在 `FriskLayoutAttrs` 定义操作指令契约及 descriptor-plan 属性；类型化字段的结构 verifier 与目标合法性 verifier 分开。

修改 `LayoutTarget`、`LayoutConstraint`、`LayoutPropagation`、`LayoutVerifier`、`DistributedLayoutConstraints`、`OperationLayoutConstraints`、`MaterializeLayouts` 和相关 CMake，接入 instruction hyperedge、解中的操作 binding 及实际 singleton 核验。原预计放进 `LayoutRelations` 的通用 tuple 准备/支持检查独立放入 `InstructionLayoutConstraints`，不把四元约束压成二元关系。保留 Analysis→IR、Target→Analysis 的依赖方向；Analysis 使用 target 抽象接口，不增加 Analysis→SM90 链接。

#### 20.12 执行计划与记录（2026-09-19）

**Goal:** 将 §20.1–20.11 的 MMA 布局契约接入真实推断、物化和独立验证。

**Architecture:** IR 定义数学操作与类型化契约；SM90 提供有界候选和纯指令证明；Analysis 只执行通用角色图和有限 tuple。依赖保持 Target→Analysis→IR。

**Tech Stack:** C++17、MLIR/TableGen、GoogleTest、lit/FileCheck、CMake。

**Global constraints:** 原目录 main，保留 Task 19 修改，不 commit/push/worktree；8 variables/component、4 candidates/domain、256 编码组合、65536 枚举点；目标与 dtype、线程、SS/RS 和失败边界严格遵守 §20.3–20.8。

执行使用 subagent-driven-development：独立的 IR/硬件证明子任务由专门实现者处理，主执行者负责通用图集成；分别测试和复审后执行全量 Gate。提交步骤由用户“不自动提交”要求替代为保留工作区修改。

- [x] **20A：IR 和 SM90 纯证明。** 修改 `include/Dialect/Frisk/IR/FriskLayoutOps.td`、`FriskLayoutAttrs.td` 及对应 `lib/Dialect/Frisk/IR` 实现；新增 `include/Dialect/Frisk/Target/SM90/SM90GemmConstraints.h`、`SM90MmaLayoutProof.h` 和对应 cpp。先添加 `unittests/Dialect/Frisk/Layout/GemmConstraintTest.cpp`、`SM90MmaLayoutProofTest.cpp` 的数学 verifier、已知 fragment 坐标、SS/RS 地址反例并记录 RED；实现后分别 GREEN。目标桥接口是 `prepareSM90MmaCandidates(LayoutConstraintGraph &) -> LogicalResult`、`buildSM90MmaContract(const LayoutConstraintGraph &, const LayoutConstraint &, ArrayRef<Attribute>) -> FailureOr<Attribute>`、`verifySM90MmaContract(const LayoutConstraintGraph &, const LayoutConstraint &, ArrayRef<Attribute>, Attribute) -> LayoutProof`。它们不改 IR，最后一个不生成候选或搜索方案。
- [x] **20B：通用联合图。** 修改 `LayoutConstraint.h/.cpp`、`LayoutTarget.h`、`LayoutVerifier.h/.cpp`、`LayoutPropagation.cpp`；新增 `Analysis/MmaLayoutConstraints.h/.cpp`。`InstructionLayoutContract` 保存 source、完整实际 binding 及角色顺序对应的 `{encodings,binding}` tuples；`LayoutSolution::instructionBindings` 按 constraint ID 记录方案。用手工有限域测试无 singleton 删除、两两可行但整体不可行、重复端点、未知类别和 ID 重排，先 RED 再实现。tuple 检查对每个已赋值端点要求相等，未赋值端点要求仍在当前域，完整 assignment 必须支持同一 tuple。测试命令 `cmake --build build --target FriskLayoutUnitTests --parallel 16` 后运行 `build/unittests/Dialect/Frisk/FriskLayoutUnitTests --gtest_filter='InstructionConstraintTest.*'`。
- [x] **20C：真实 pass 和物化。** 修改 `DistributedLayoutConstraints.cpp`、`OperationLayoutConstraints.cpp`、`SM90LayoutTarget.cpp`、`MaterializeLayouts.cpp` 及 CMake 接线。target 抽象提供上述 prepare/build/verify 三个对应 virtual 方法，默认拒绝未知指令。RelationsOnly 在 prepare/build 前返回；实际 binding 读取属性并调用纯 verify。先用 `test/Transforms/infer-gemm-layout.mlir` 与错误用例触发未支持操作，再实现 A/B/init/result 角色、线程约束、独立 encoding、方案写回和原子回滚。分别核验 `64×64×16` 和 `128×128×64` SS/RS、两种 dtype、transpose/phase/线程篡改、二次运行与打印解析回放。
- [x] **20D：复审、Gate 和文档。** 对 20A、20B/20C 做契约与代码质量复审；运行 §20.9 全部命令，记录新增测试、实际结果、限制及与原契约差异；同步比较文档。未跑的 GPU/上游测试明确标未执行。

#### 20.13 实现、复审与验证记录（2026-09-19）

**工作区与交付范围。** 直接在 `/home/baopeihua/frisk` 的 `main` 修改，保留进入任务前的 Task 19 未提交内容，没有新 worktree、commit、merge 或 push。本轮实现的是静态数学 MMA 的布局推断/验证闭环，不是可运行 WGMMA lowering。Task 21/22 未开始。

**实际接入：**

1. `MmaOp` 验证 rank/static shape、矩形 transpose、浮点输入/累加匹配和已有 encoding；MemRef 只读，init 显式 SSA。`MmaInstructionContractAttr`、`MmaDescriptorPlanAttr` 使用严格 Dictionary schema，拒绝缺字段、未知字段及不匹配的 packing/form。
2. `SM90GemmConstraints` 生成 canonical fragment、有限 shared 建议及确定性的 descriptor 方案；`SM90MmaLayoutProof` 根据 PTX 坐标、packed 顺序和实际 root-relative 地址独立证明，不用候选缓存代替证明。Generated proposal 不增强根对齐。
3. `InstructionLayoutConstraints` 提供冻结域的有界 tuple 准备和完整 tuple 支持检查。重复端点只枚举一次；四个 4 候选域正好至多 256 次；Strict/Common 不依赖 singleton 才触发；partial solver 必须有同一完整 tuple 扩展。
4. `MmaLayoutConstraints` 收集四个真实角色；方案写入 `LayoutSolution::instructionBindings`。Tensor producer 与 use 分离、init-use/result SameLayout；多 MMA 共用 storage 合并有限建议来源。原 ResourceLimit 与 8 vars/4 candidates 上限保持。
5. 事务物化写 encoding/storage/typed contract/threads，必要时仅在 Tensor 消费点插转换。actual-only 重建实际 singleton，绕过候选生成、tuple 枚举及方案构造，纯验证通过才提交。

**测试覆盖与证据分层：**

- 新增 36 个单元：`GemmConstraintTest.cpp` 7、`SM90MmaLayoutProofTest.cpp` 8、`InstructionConstraintTest.cpp` 9、`GemmLayoutIntegrationTest.cpp` 11，以及 legacy adapter 的 RS replication 分类 1。新增 2 份 lit：`infer-gemm-layout.mlir`、`infer-gemm-layout-errors.mlir`。
- 真实推断覆盖 `64×64×16` 的 SS/RS × f16/bf16 × transA/transB 共 16 种组合，以及 `128×128×64` 的两种 form × 两种 dtype 共 4 种组合。大 tile 同时验证二次推断、打印/解析回放及 actual-only 无生成调用。
- Helper 独立坐标表覆盖 A packed halves、accumulator、M/K repeat、N-group replication、全部受支持 atom N；descriptor 覆盖 K/MN × 无/32/64/128-byte swizzle 共 8 模板、父 K-panel 间距、非零 start/phase、对齐/越界反例。线程与 policy helper 覆盖 128/256/512/1024 和三种 policy；不宣称全部 descriptor × transpose × 多 warp-group 的端到端笛卡尔积已运行。
- 实际 IR 负例覆盖缺属性、线程/target 冲突、合法 typed grid/descriptor 篡改、仍满足通用 coverage 的 fragment 位交换。错误 RS packing 在 typed checked-constructor 层拒绝；直接注入原始 Dictionary 也不能绕过 actual-only 的 typed 属性要求。测试累计检查候选枚举/准备/方案构造次数均为 0。
- 外部 A/init 编码保留，并在消费点产生两个转换；失败事务保留原 IR 逐字不变；两个 MMA 共用存储域、未知指令、重复端点、tuple 去重/remap、256 组合与第 5 候选拒绝、逻辑/硬件点数预算均有测试。

**复审与修正。** 分别进行了通用图集成和硬件/全链路独立源码复审，未发现已确立的 Critical/Important 正确性缺陷。验收覆盖缺口是实际 fragment/packing/descriptor/grid 篡改，已补测试；另修正数学 verifier 对未知 Tensor encoding 的漏检、失败角色标注，以及显式 binding 失败时丢失 Unknown/reason/坐标的诊断。早期预算失败现在明确为 Unknown；无可构造方案报告 bounded decoder 无法证明，不冒充硬件不可能的数学反例。生成 API 仍返回 `FailureOr<Attribute>`，一般解码失败只能给有界构造失败说明，不能承诺总有逐点反例。

**测试先后记录。** 未注册 `frisk.mma`、未知 InstructionContract 被错误接受、未知 Tensor encoding、角色诊断、Unknown 诊断均观察到预期 RED 后修正。部分硬件 helper 用例虽先写测试，但首次可运行构建时已为 GREEN，不能宣称每个 helper 都单独完成过可执行 RED。实际 packing 负例最初误用会 assert 的 `get()`；改为 `getChecked()` 验证结构拒绝，这属于测试构造修正，不是生产路径接受了非法 packing。

首次全量单元运行在新增 Parallel fixture 构造中触发空 region `front()` 断言；单测复现和带符号栈定位到测试自身。原因是现有 `ParallelOp::build` 只添加 region、不创建 block；测试现已显式构造两个 index 参数的 block 和 `EndOp` 再放入 MMA。未因此改动旧 Parallel builder 或放宽生产 verifier；该次中止不能计作全量通过，后续重新执行完整 Gate。

**Legacy 分类而非机械一致：**

| 历史输入/行为 | Task 20 处理与证据 |
| --- | --- |
| 适用的旧 SM90 SS/RS storage 与 fragment | 保留原 adapter 持有关系/地址回归；不以属性文本相等或旧 f16 accumulator 冒充新 f32 数值等价 |
| 旧 RS A 的 replication=2 | 新增分类测试：旧候选被新指令证明拒绝；128 线程新 canonical A 为 replication=1，不能沿用旧冗余组织；B storage 地址仍可独立对照 |
| SM80 或仅 `sm_90` target | 新 MMA 只接受显式 `sm_90a`；旧 SM80 路径仅保留 legacy 回归，不扩展本阶段 target |
| Local A/B、Tensor B | Local MemRef 不当寄存器；RS 必须 Tensor A，B 必须 Shared。明确拒绝，生命周期 normalization 留到 Task 22 |
| 非四 warp 分组/错误线程 scope | 按合法 grid、连续 warp-group 和实际 Parallel scope 检查；不为旧 policy 输出放宽硬件边界 |
| 旧 positional ABI、clear/annotation 行为 | 不复制位置下标或完成标记；新 op 用显式 init SSA 表示旧值依赖，不提供隐式 clear。旧 ABI 自动转换不在本任务 |

**最终 Gate：** 以下命令重新执行并通过，单元总数较 Task 19 的 129 增至 165。全量单元耗时 207207 ms，23 个 suite；其中 MMA 集成 suite 为 196996 ms，不是 GPU 性能数据。

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
git diff --check
```

| 验证项 | 最终实际结果 |
| --- | --- |
| 构建及 lit | exit 0；34/34 PASS（9.46 s） |
| 全量单元 | exit 0；165/165 PASS（207.207 s） |
| CTest legacy executables | exit 0；4/4 PASS |
| `git diff --check` | exit 0，无 whitespace 错误 |
| 独立复审 | 通用图与硬件/全链路各自审阅；覆盖/诊断问题修正后复核，无剩余阻塞项 |

构建观察到已有 Ninja `premature end of file; recovering` 和 CMake CMP0116 OLD 弃用提示，不宣称零警告构建。负例测试中的预期错误诊断不计为失败；最终通过以测试汇总与 exit code 为准。

**未执行/不支持：** 未运行 TileLang/Triton 上游测试、GPU 数值内核或性能 benchmark；未升级固定上游 SHA。一般 Product 指令布局、动态/非二次幂/尾块、Tensor B、Local fragment、f16 accumulator、TF32/FP8/int/sparse、SM80 新路径、异步 token/同步 lowering、寄存器分配/spill/occupancy 和完整 CostVector 均未实现。本任务 65536 点是证明预算，不是寄存器性能预算；policy/swizzle 排序不代表性能最优。

以下保留最初的 Task 20 迁移提纲用于追踪演进，复选框不代表当前验收进度；以上述 v1 契约和 A–H 为准。其中“DotOperand/Mma encoding”按 §20.7 落实，寄存器 B 不在 SM90a 支持集，旧 Gemm 自动 normalization 留到 Task 22。原 Step 6 不自动执行，提交和推送仍需用户另行授权。

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90GemmConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/GemmConstraintTest.cpp`
- Create: `test/Transforms/infer-gemm-layout.mlir`

**Interfaces:**

- Produces internal operand type constraint and op：

```tablegen
def Frisk_MemRefOrRankedTensor :
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>;
```

```mlir
%result = frisk.mma %a, %b, %init
  {m = 128, n = 128, k = 64, trans_a = false, trans_b = false,
   policy = #frisk<gemm_warp_policy Square>}
  : (memref<128x64xbf16, #frisk<memory_space Shared>>,
     memref<64x128xbf16, #frisk<memory_space Shared>>,
     tensor<128x128xf32>) -> tensor<128x128xf32>
```

```cpp
LogicalResult collectGemmConstraints(MmaOp op,
                                     LayoutConstraintBuilder &builder,
                                     LayoutTarget &target);
```

- [ ] **Step 1: 写 MMA verifier 和 constraint 红灯测试**

覆盖 BF16/FP16 `m64nNxk16` SM90 候选、A/B major/transpose、accumulator shape、128-thread warp-group、非法 dtype/shape，以及不同 warp policy。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，内部 MMA Op/constraints 尚不存在。

- [ ] **Step 2: 定义 target-neutral `MmaOp`**

Op 只保存数学 tile 语义和 policy，不保存 PTX mnemonic。A/B 可为 Shared MemRef view 或 RankedTensor；init/result 必须是 RankedTensor。Verifier 只检查 shape/dtype 的数学一致性，SM90 合法性放 target rules。

- [ ] **Step 3: 审核新旧 Gemm 语义后提取 canonical candidates**

以 `FriskOps.cpp` 的旧 Gemm target/layout 构造为迁移输入（历史定位 1285–1406 行，实施时按符号重新定位），对照固定新版 TileLang 的参数/annotations、read-before-write 和硬件契约审核后迁入 `SM90GemmConstraints.cpp`。输出 `InstructionContract`、`RequireEncoding` 候选和 provenance；不得直接写 layout map，也不得机械搬用旧 positional args 下标。

```text
A/B shared -> Storage candidates + WGMMA descriptor contract
A/B tensor -> DotOperand distributed candidates
result/init -> Mma/Distributed accumulator candidates
```

- [ ] **Step 4: 与 legacy Gemm 枚举差分**

对现有 `sm80_ss`、`sm90_ss` 和 local/shared 组合先分类：已知正确且适用当前 SM90 契约的 case 要有数学等价候选；SM80-only case 仅作 adapter 回归，不能因此扩展本阶段 target。旧 bug、非法 owner/地址、已过时协议按独立坐标和硬件契约修正，记录新旧输出、原因和反例，不要求复现旧错误。新版新增行为单独测试；新系统可保留其他合法候选，未运行成本实验不能称其更优。

- [ ] **Step 5: 运行 tests**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='GemmConstraintTest.*'
```

Expected: PASS；非法 SM90 contract 输出目标规则名称。

- [ ] **Step 6: 提交 Gemm 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: model gemm as layout constraints"
```

### Task 21: 增加 Tensor Reduce Op 并迁移 Reduce ownership

**状态（2026-09-19）：用户已确认 §21.1–21.7；受限实现、独立复审及 197 unit / 36 lit / 4 CTest 回归已完成，完整链成功验收仍未完成（9 变量与 8 变量上限冲突，见 §21.8）。实施基线为本地 main `c028e82`，直接在源目录实施，未创建 worktree、未 commit/push。原 Step 1–6 保留为历史提纲，以已确认的具体契约及实际验收记录为准。**

#### 21.1 实现审计：哪些可以复用，哪些不能直接迁移

| 核对对象 | 实际发现 | 对 Task 21 的影响 |
| --- | --- | --- |
| `FriskOps_Reduce.cpp::ReduceOp::inferLayout` | 旧输入/输出都是 Local MemRef；依赖已有 src layout，主要进行 src→dst 的维度消除与 replication condense | 复用逻辑投影思想，不复用 Local MemRef/DenseMap 生产路径，不宣称已有一般双向推断 |
| `FriskOps.cpp::ReduceOp::verify` | 旧 kind 为 add/mul/min/max；检查删维 shape 和同 dtype，旧操作还有目的端/clear 语义 | 新 Tensor Reduce 的 sum 命名、纯值语义与旧 add/clear 的自动映射必须分开，自动 normalization 留给 Task 22 |
| `test_pass/reduce_layout_test.cpp` | 8 个参数用例及 GEMM→Reduce legacy 用例；case 5 注释明确指出 src map 有问题；case 7 的实际 index 为 `d1*4+d1`、thread 为 `d2`，均不编码 batch 维 d0；case 6–8 使用 16 线程 | 不能以旧测试 PASS 证明输入持有关系合法。case 7 不同 batch 会落到相同物理槽；这些用例要分类或重新构造独立 oracle，不能机械复制期望 replication |
| `LayoutRelations.cpp::permuteEncoding` | 当前 TransformLayout 接收等 rank 的轴置换，反向用逆置换；归约删维并非双射 | 不把 Reduce 塞进现有 permutation payload，不伪造逆矩阵补回已消失的 reduction 维 |
| Task 19 execution proof | 已能验证静态覆盖、重复持有者、first_owner 和线程数；同一元素的多个持有者默认表达同值副本 | 需要另证“每个不同逻辑输入恰好贡献一次”；普通 coverage/replication 检查不能单独证明 sum 没有多算 |
| Task 20 instruction tuples | 四角色及 MMA source 的不变量是明确契约，不是任意 arity 的通用计算节点 | Reduce 采用有名的二端点关系和独立方案绑定；不复制端点凑成四元 MMA tuple |
| 当前公共布局表示 | Distributed pass 支持静态非零 rank、每维大于 1 的二次幂 BitLinear；已有 8 vars / 4 candidates 和 65536 点预算 | 原提纲中的非二次幂 guarded/shared fallback 不能只加一个分支就实现，需要先扩展表示与边界证明 |

固定上游证据再次核对本地审计副本的 HEAD：TileLang `5e149e31674658f94779c7d0c6039549a1853123`、Triton `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c`。没有刷新浮动 main，也没有运行上游测试。GitHub 固定源码网页本轮读取返回 cache miss，源码判断来自同 SHA 本地副本，不将其他本地 TileLang 分支误当该快照。

- TileLang `src/op/reduce.cc::ComputeReducerLayout/InferLayout` 仍是传统 Fragment 的源驱动投影、压缩和目的包含性检查。`src/layout/layout.h::PartialFragmentNode` 明确区分 addend lanes 与 equal-value copy groups，并禁止将 partial 当作普通 replica；这与 local.reducer epoch 是另外一层机制。
- Triton `inferReduceOpEncoding` 使用 SliceEncoding；`ReduceOpToLLVM.cpp` 先处理寄存器内归约，再处理 warp 内归约，其余跨 warp/block 情形借助布局转换与同步。它已经具有分层归约实现，不能把 register/warp/shared 分层本身宣称为 Frisk 独创。
- 本地 MLIR `arith.maximumf/minimumf` 与[官方 Arith 文档](https://mlir.llvm.org/docs/Dialects/ArithOps/)均明确 NaN 传播及正负零次序；新 kind 不能无说明地混用 maxnum/minnum。

#### 21.2 方案取舍

| 方案 | 收益 | 问题与结论 |
| --- | --- | --- |
| A：坐标投影＋贡献者证明＋有界通信方案（推荐） | 保持 Tensor/Storage 双域、有限候选、实际 IR 独立验证；能接上 Task 20 MMA 输出 | 首版限定现有静态 BitLinear 支持集，需要新增归约专用关系，但改动边界可控 |
| B：只迁移旧 dst map/replication 公式 | 实现最短 | 无法证明重复输入未被多算，难以独立核验通信需求，不作为完整 Task 21 交付 |
| C：同时引入 PartialFragment/epoch、非二次幂及 GPU lowering | 支持范围更广 | 扩展类型/语义/调度多个系统，超出当前单任务，单独规划 |

推荐 A。此选择包含对原提纲的明确缩限：非二次幂 fallback 改为可测试的 unsupported 边界，待支持表示/guard 的后续任务补齐；不把 Shared 通信方案误说成已经支持 ragged shape。

#### 21.3 已确认的数学操作与支持集

```mlir
// 已注册操作的未编码输入示例；布局由推断补齐。
%result = frisk.reduce_tensor %src {kind = "sum", dim = 1}
  : tensor<64x64xf32> -> tensor<64xf32>
```

1. 单个 RankedTensor 输入与单个 RankedTensor 结果；沿 dim 归约，结果 shape 等于删除该维后的 shape；无 keep_dims、隐式 init/clear、目的 MemRef 或写回副作用。操作是纯 Tensor 值计算，外部累加另用显式算术操作表达。
2. 首版 kind 为 `sum/max/min`，dtype 为 f16/bf16/f32，输入与结果同 dtype；没有隐式 f32 提升。旧 `add` 不作为新 op 的第二套公开拼写，旧 `mul`、abs/bitwise、自定义 combiner、argmax/argmin 不包含在首版。
3. `sum` 按每个不同逻辑输入恰好一次定义，允许并行重结合，选中方案固定确定性的归约树；不承诺与串行从左到右相加逐 bit 一致。这是新操作本身的显式语义选择，不为普通 arith.addf 擅自添加重结合许可。CPU 测试须按指定树核对，不能用浮点交换律假设代替证明。
4. `max/min` 采用 maximumf/minimumf 的 NaN 传播和正负零语义，不采用“忽略 NaN”的 maxnum/minnum。NaN payload 不作为跨不同实现的逐 bit 保证。
5. 首版输入 rank 至少 2，结果 rank 至少 1；所有 extent 为大于 1 的静态二次幂。rank-1→scalar、extent=1、动态、非二次幂明确拒绝，不伪造 shape=1 的输出绕过既有 verifier。
6. 限单 CTA、32/64/128/256/512/1024 线程，lane=32。显式 `frisk.execution_threads` 与外层 Parallel 必须一致；没有 Parallel 时优先显式值，再继承可确定的输入编码/生产者图线程要求，否则缺省 32。例如 Task 20 的 128 线程 MMA→Reduce 默认继承 128，不强行先转成 32。多个互相冲突的来源不取任意第一个。
7. target 按当前 SM90 target 模型处理：支持 `sm_90/sm_90a`；缺失时沿用普通操作默认 SM90，不为 Reduce 额外要求 MMA 的 `sm_90a`。但同图 MMA 仍独立要求显式 sm_90a，其他显式 target 拒绝。
8. result 的多个持有者必须表示已经完成的同一个归约值，而不是尚未合并的 partial。只有 layout 无法证明动态 payload 相等；相等是 Distributed Tensor 的语义契约，不将任意不等的 runtime 部分和当作 replica。

#### 21.4 投影与贡献者证明

设 `Dsrc(h)` 为源硬件位置 h 对应的逻辑坐标，`P` 为删除 dim 的逻辑投影；输出坐标 y 的归约集合为：

```text
F(y) = {x | P(x) = y}
```

证明必须区分三个对象：F(y) 中不同的输入元素、同一个输入 x 的多个物理副本，以及完成后输出 y 的多个物理副本。只有第一类是 sum 的不同加数。

推荐先由 `P∘Dsrc` 构造自然输出布局，再压缩投影后冗余的 register 位：按旧 register 位顺序选取线性独立的列，保留 lane/warp/warp_group/CTA 的组织，重新计算 register extent 和实际 replication。不能只删输出行后沿用旧 replication；不能只删除零列而忽略非零相关列。通用投影 API 可复用，但 Reduce 的删维、重命名和 register 商空间处理应有独立 helper。

对每个 y，按 `(cta,warp_group,warp,lane,register)` 的字典序选出每个不同 x 的最小源持有者一次，并按此物理键排序。`canonical_fiber_tree_v1` 每层将相邻项两两合并，奇数尾项原样进入下一层，合并结果保留在左项的代表持有者；重复直到一个根，再将完成值从根分发到 Ddst 声明的所有输出持有者。证明检查输入完整、无重复贡献、各次合并的贡献集合不相交、最终集合恰为 F(y)，并检查完整结果的分发覆盖。first_owner 是本版本的确定性基线，不宣称通信量最小；不能让输出副本数量乘进求和。

以每行 4 个不同值 `[1,2,3,4]`、每个值均有两个物理副本为例，行 sum 必须是 10，不能是 20；两个输出持有者应各持有 10，而不是分别持有未完成的 3 和 7。即使 max 测试对重复加数不敏感，sum 反例也必须覆盖。

通信范围按实际树边及完成值分发边推导：同线程为 register，跨 lane 同 warp 为 warp，跨 warp/warp_group 为 CTA shared-tree。只记录依赖与通信要求，不生成 shuffle、shared alloc 或 barrier。含 shared-tree 的证明不能表述为已经完成可执行同步/共享容量分配验证；其含义是单 CTA 内存在该有限通信依赖，后续 lowering 必须落实资源和同步。

#### 21.5 图接线、有限候选与方案绑定

- 新增有名二端点 `ReductionLayout` 关系，端点固定为 `[src-use,result]`，payload 保存 source op、dim 和实际归约绑定；结构检查覆盖角色、类型、归属、ID remap。保留现有 TransformLayout 的轴置换语义，不影响 Task 20 四元不变量。
- src producer→src-use 使用已有 Convertible；自然 result→其他消费者使用既有 use/转换关系。不能直接修改外部 producer 来满足这一次归约，也不能无条件认定所有 input/output encoding 组合可行。
- 正向只从已允许的源候选生成自然输出布局；反向根据输出要求筛选已有源候选，不把投影当可逆映射，不凭空恢复源布局。下游消费者的不同布局要求通过消费者边转换；若 Reduce 自身结果的硬编码与所有合法源候选的自然投影均不兼容，则报冲突，不能靠消费者转换掩盖，也不能覆盖硬绑定。自然布局的逻辑维名称允许按结果轴位置作语义等价重命名，不据此接受任意结果映射。
- 候选准备需把 Reduce 正向投影接入现有有限来源闭包，支持 reduce→reduce、transpose→reduce 和 MMA→reduce；result 不先塞无关默认候选。来源包含显式编码、前驱合法传播和既有 target 默认族，来源去重并记录 provenance。冻结后 Strict/Common 只删不增，每次删除重排相关约束；部分和完整 assignment 都检查 pair 支持。
- 保留 8 vars/component、4 candidates/domain；每条二端点关系最多 16 对布局，不通过复制变量或截断候选绕过限制。单个输入/输出的逻辑与硬件枚举分别最多 65536 点；贡献关系按源点和目标点线性建索引，不构造 65536² 的笛卡尔积。超预算返回 Unknown。
- 规划 `ReductionLayoutConstraints.{h,cpp}`、`ReductionLayoutProof.{h,cpp}`（Analysis）与 `SM90ReduceConstraints.{h,cpp}`（Target）。前者处理逻辑 fiber/候选关系/贡献证明，后者处理线程组织与通信方案；仍保持 Target→Analysis→IR，不在通用 solver 散布 target 字符串。
- 规划操作属性 `frisk.reduction_contract : ReductionContractAttr` 和 `frisk.execution_threads : i64`。采用完整、严格的 v1 schema：version、target、kind、axis、dtype、输入/输出 shape、threads、算法版本 `canonical_fiber_tree_v1`、输入副本选择规则、输出完成值分发规则、通信范围。规范算法从实际编码唯一确定详细依赖，无 pointer/cache ID；无法支持的算法字段拒绝，不静默更换方案。
- 解中以稳定 constraint ID 保存 reduction binding，物化到真实操作；显式完整属性只验证、不覆盖。最后 actual-only 从实际输入/结果类型和属性检查 shape、投影、贡献集合、通信范围与线程一致性，允许执行纯确定性证明，禁止候选准备、pair 枚举、重新搜索方案或复用旧解缓存。
- 沿用 detached module 原子物化、打印解析与二次推断稳定性；写回前失败保持原 IR 不变。后续 tile_store 的唯一写入者检查复用既有 ownership 约定；Reduce 本身不是 MemRef 写操作。

#### 21.6 验收分组与旧行为分类

- [x] 数学 verifier：所有 kind/dtype、dim 越界、删维 shape、输入/结果类型不匹配、rank-0/动态/非二次幂拒绝、纯副作用及 printer/parser；sum/max/min 的语义差异和 NaN/±0 分类有独立 CPU oracle。
- [x] 布局代数：reduce 首/中/尾轴、register 位压缩、相关列/XOR 混合、coverage、replication 重算及自定义维名；不能由待测候选生成器生成唯一“期望值”。
- [x] 贡献证明：线程内、跨 lane、跨 warp、跨 warp-group，输入/输出副本分离；sum 漏算/重复、错误 partial、跨行混合、缺失结果分发、thread scope 和预算负例；CPU 模拟检查每个输出完整贡献集合。
- [x] 真实推断的已通过子集：`64×64→64`、3D 删中轴、`128×128` MMA→按行 Reduce、Reduce→Global tile_store、连续两次 Reduce、外部显式编码转换、SCF/Parallel 环境；保留 bootstrap 上限。
- [ ] 原契约完整链成功验收：Task 20 `128×128` MMA 结果→按行 Reduce→tile_store。当前只验证 9 变量超限拒绝，分段成功不替代此项；须另行确认预算或等价压缩策略。
- [x] 传播与独立核验：输出要求反向删源候选、无支持域失败、固定点统计、稳定 ID；缺少/篡改 reduction_contract、线程/通信范围、源/结果编码均 fail-closed；actual-only 生成计数为 0，事务回滚和回放逐字稳定。
- [x] Legacy 分类：case 1–4 作为适用候选，仍先检查完整持有关系；case 5 的不一致 replication 不迁入正确性 oracle；case 7 的 batch 丢失作为反例并另建合法 3D 样例；case 6/8 的 16 线程不伪装成新支持集，补真实 32-thread 同值复制后的独立样例。旧 add→sum、mul/clear、SM80 GEMM 来源均记录支持/延期，不以旧文本相等为验收。
- [x] 全量回归 Gate：新增 `ReduceConstraintTest.cpp`、`ReductionLayoutProofTest.cpp`、`ReduceLayoutIntegrationTest.cpp` 和 `infer-reduce-layout{,-errors}.mlir`，197 unit / 36 lit / 4 CTest 全通过；独立复审与未完成项见 §21.8。未执行 GPU/上游实验；本项通过不消除上述完整链验收缺口。

#### 21.7 确认点与实施顺序

用户已确认方案 A 的支持集和语义——**sum/max/min、静态二次幂 Tensor、贡献者去重、自然投影及寄存器压缩、register/warp/CTA 通信契约和实际 IR 复验；sum 允许并行重结合，非二次幂 fallback、PartialFragment/epoch 与 GPU lowering 延后**。

按四步落地：数学 IR 与独立投影/贡献 oracle → 有名二端点关系及候选闭包 → SM90 通信属性与事务物化/actual-only → MMA→Reduce 集成、legacy 分类、全量回归和文档。上述受限实现与复审修正已完成；165/34/4 是已提交基线，本轮实际回归为 197/36/4。完整链成功验收尚未完成。

#### 21.8 实际实现与验收边界（实施记录）

**工作区。** 基线为本地 `main c028e82`，本轮直接修改原目录；保留既有 Task 21 设计修改，不创建 worktree，不自动 commit/push，不修改独立汇报 README。

**实现接线。**

| 层次 | 实际代码与责任 |
| --- | --- |
| IR | `FriskLayoutOps.td/.cpp` 新增纯 `ReduceTensorOp`；`FriskLayoutAttrs.td/.cpp` 新增完整、严格的 `ReductionContractAttr` |
| 基础证明 | `ExecutionLayoutProof.h/.cpp` 提取共享的有界坐标枚举；`ReductionLayoutProof.h/.cpp` 做自然投影、register 子空间基选取、规范贡献树和完整输出分发 |
| 图关系 | `ReductionLayoutConstraints.h/.cpp` 收集 `[src-use,result]`、解析线程环境、冻结后构建最多 16 对合法编码和 binding；`LayoutConstraint` 检查实际 SSA 角色、axis、来源、pair 唯一性与稳定 ID |
| Target | `SM90ReduceConstraints.h/.cpp` 解释 SM90/SM90a、线程与通信范围；通过 `LayoutTarget::buildReductionContract/verifyReductionContract` 接入，不把 target 字符串判断塞进通用 solver |
| 传播和求解 | 正向生成自然输出，反向仅以已有 pair 删减；Strict/Common 重排受影响约束；部分和完整 assignment 都要求 pair 支持；`LayoutSolution::reductionBindings` 按稳定 constraint ID 保存方案 |
| 物化与复验 | `MaterializeLayouts.cpp` 写 `frisk.reduction_contract` 和 `frisk.execution_threads`；实际 IR 路径不调用候选生成、pair 准备或 binding 搜索；失败不更新原 IR |

`ReductionContractAttr` 的 v1 字典准确包含 12 个字段：`version/target/kind/axis/dtype/source_shape/result_shape/threads/algorithm/input_policy/output_policy/scope`。算法固定为 `canonical_fiber_tree_v1`，输入策略固定 `first_owner`，输出策略固定 `broadcast_complete`；scope 为 `register/warp/cta_shared_tree`。物理位置序列、相邻合并树和广播目的地由实际编码唯一重建，不存 SSA 指针或缓存编号。

**明确的验收差异：完整 MMA→Reduce→store 链。** 实施时发现原 §21.6 同时要求完整链成功和保持 8 vars/component，两者在当前图模型下冲突。SS MMA 有 A/B storage、init producer/use、result 共 5 个变量；Reduce 新增 source-use/result 2 个；store 新增 use/storage 2 个，总数为 9，且处于同一连通分量。已向用户说明这是原契约遗漏，8 是 bootstrap 求解规模限制而非硬件限制。未获得放宽上限的明确授权，因此保持 8，分别验证 `128×128 MMA→Reduce`、`Reduce→Global tile_store`；完整链验证 9 个真实变量及超限拒绝，不能宣称完整链成功。后续若需完整链，应单独确认预算或等价变量压缩策略。

**验证方法。** 数学、贡献证明、图集成分别由 `ReduceConstraintTest.cpp`、`ReductionLayoutProofTest.cpp`、`ReduceLayoutIntegrationTest.cpp` 覆盖；lit 正反例为 `infer-reduce-layout{,-errors}.mlir`。CPU oracle 按声明的树模拟求和/最大/最小，包含 NaN、正负零和重结合反例；这不是 GPU 数值执行。旧 case 1–4 独立反演 thread/slot 关系，case 5 错误 replication 和 case 7 batch 冲突不作为正确性 oracle，case 6/8 明确区分旧 16 线程与新 32 线程同值副本适配。

**复审记录。** 基础证明首轮无 Critical/Important；按 Minor 建议增加物理顺序不同于逻辑顺序的完整树 oracle、逻辑/硬件预算边界以及 API 数值语义前置条件说明。图复审发现下游候选反向污染自然输出、线程环境越过已确定转换边界的问题；已用回归复现并修正。硬编码端点不再接受不可能的传播候选，归约自然结果不接受反向消费者建议；线程追溯在已确定的 encoding/执行环境处停止，但仍核对 SameLayout 同伴的独立冲突。最终代码复审无新增 Critical/Important 实现缺陷；完整链未成功仍是 Important 验收缺口。另有 Minor 测试建议保留：目前 fanout 回归硬绑定了 Reduce 结果，未来可增加未编码结果的下游强需求用例，独立保护仅针对 `reductionResult` 的候选屏障。

**测试命令与当前证据。**

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk \
  frisk_attr_test frisk_reduce_layout_test frisk_layout_pass_test \
  frisk_memory_effect_test --parallel 16
build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
git diff --check
```

本轮 build/check-frisk 返回 0，lit 36/36、CTest 4/4 通过；全量 unit 的 26 个 suite、197/197 项全部通过（262.577 秒，退出码 0）。新增单元为 4 项数学/schema、13 项投影/贡献证明、15 项图集成，共 32 项；新增 2 份 lit 文件。既有 Ninja `premature end of file; recovering` 和 CMake CMP0116 弃用提示使构建重新执行较多目标，不作为 Task 21 功能失败，也未为此改写构建缓存或策略。CPU 布局与树 oracle、打印解析和独立验证不代替 GPU 数值/性能测试；本轮未执行 GPU 或上游框架实验。

以下为原始迁移提纲，实施以以上已确认的具体契约及实际记录为准；原 Step 6 不作为自动提交授权。

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90ReduceConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/ReduceConstraintTest.cpp`
- Create: `test/Transforms/infer-reduce-layout.mlir`

**Interfaces:**

- Produces：

```mlir
%result = frisk.reduce_tensor %src
  {kind = "sum", dim = 1}
  : tensor<64x64xf32> -> tensor<64xf32>
```

```cpp
LogicalResult collectReduceConstraints(ReduceTensorOp op,
                                       LayoutConstraintBuilder &builder,
                                       LayoutTarget &target);
```

- [ ] **Step 1: 写 ownership 红灯测试**

覆盖 reduce dim projection、lane shuffle、replicated result、elected owner、thread count 与 replicate 不整除、非 2 次幂 reduce extent 的 guarded/shared fallback。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，新 Reduce Op/constraints 不存在。

- [ ] **Step 2: 定义 `ReduceTensorOp`**

Verifier 检查 result shape 等于删除 reduce dim 后的 shape、dtype/kind 支持、dim 合法。Op 不直接选择 shuffle/shared tree。

- [ ] **Step 3: 将现有 Reduce map 逻辑转成 Transform/Ownership constraints**

`FriskOps_Reduce.cpp:235-369` 的 dimension projection、replicate condense 语义迁到 target-neutral transform；SM90 rule 决定 shuffle/shared tree candidate。无法证明 unique owner 时删除 store candidate并给反例。

- [ ] **Step 4: 与 legacy Reduce cases 差分**

复用 `reduce_layout_test.cpp` 已知正确的 case table，比较 output coordinate、replication 和 owner，而不是 Attr 文本。对照新版 TileLang 区分普通 Fragment 的同值 replica 与 PartialFragment 的 addend/copy groups；传统 Reduce 与 reducer epoch 是不同机制，不因后者更新就宣称前者全面双向化。先列出本任务实际接入子集和不支持诊断；旧错误用独立 reduction/owner 不变量修正并记录差异，不强求 legacy 等价。

- [ ] **Step 5: 运行 tests**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ReduceConstraintTest.*'
```

Expected: PASS；legacy executable 仍 PASS。

- [ ] **Step 6: 提交 Reduce 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: model reduce ownership constraints"
```

### Task 22: 实现 legacy normalization 并退役旧生产路径

**Files:**

- Create: `lib/Dialect/Frisk/Transforms/NormalizeLayoutIR.cpp`
- Create: `include/Dialect/Frisk/Transforms/PassPipelines.h`
- Create: `lib/Dialect/Frisk/Transforms/PassPipelines.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskOps.cpp`
- Modify: `lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp`
- Modify: `include/Dialect/Frisk/IR/FriskInterfaces.td`
- Create: `test/Transforms/normalize-legacy-layout-ir.mlir`
- Create: `test/Transforms/no-legacy-layout-map.mlir`

**Interfaces:**

- Produces: `std::unique_ptr<Pass> createNormalizeLayoutIRPass()`，参数 `frisk-normalize-layout-ir`。
- Produces: `void buildFriskLayoutPipeline(OpPassManager &pm)`，注册参数 `frisk-layout-pipeline`。
- Pipeline contract:

```text
legacy Buffer IR
  -> frisk-normalize-layout-ir
  -> Tensor Mma/Reduce + layout_view/tile load/store
  -> frisk-infer-layouts
```

- [ ] **Step 1: 写 normalization 红灯测试**

输入使用现有 local MemRef Gemm C、Reduce src/dst；输出必须含 `frisk.mma`、`frisk.reduce_tensor`、Tensor SSA 和必要 bridge，不得含 local fragment 作为新 solver LayoutVar。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，normalization pass 尚不存在。

- [ ] **Step 2: 实现 local MemRef 到 Tensor SSA normalization**

按 def-use slice 重写完整 fragment 生命周期。仅支持可证明无 alias escape 的 local alloc；地址被未知 Op 使用时诊断：

```text
legacy local fragment escapes supported normalization boundary
```

不得部分转换同一 alias group。

- [ ] **Step 3: 注册默认布局 pipeline**

在 `PassPipelines.h/.cpp` 提供：

```cpp
void buildFriskLayoutPipeline(OpPassManager &pm) {
  pm.addPass(createNormalizeLayoutIRPass());
  pm.addPass(createFriskInferLayoutsPass());
  pm.addPass(createOptimizeLayoutConversionsPass());
}
```

在 pass 注册实现中增加：

```cpp
PassPipelineRegistration<> layoutPipeline(
    "frisk-layout-pipeline",
    "Normalize, infer, materialize, and optimize Frisk layouts",
    [](OpPassManager &pm) { buildFriskLayoutPipeline(pm); });
```

`frisk-opt` 启动时调用 `registerFriskPasses()` 后必须同时注册上述 pipeline；`PassPipelines.cpp` 加入 `FriskTransforms` source list，不能依赖静态初始化碰巧被链接保留。

- [ ] **Step 4: 删除旧生产接口**

从 ODS/C++ 删除 `ParallelOp::inferLayout`、`GemmOp::inferLayout`、`ReduceOp::inferLayout` 及旧 `LayoutInterface::inferLayout(DenseMap&)`。Legacy adapter 和 baseline oracle 保留在 Analysis/test，不得被生产 pass 调用。

- [ ] **Step 5: 验证仓库无生产 DenseMap 路径**

Run:

```bash
rg -n "inferLayout.*DenseMap|DenseMap<Value, Attribute>.*layout" \
  include/Dialect lib/Dialect
```

Expected: 无匹配；允许匹配仅存在于 `LegacyLayoutAdapter` 和 test 文件时，应使用更窄路径再次确认生产目录为 0。

- [ ] **Step 6: 运行 M4 全量回归**

Run:

```bash
cmake --build build --target FriskIR FriskTransforms FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: 所有 tests PASS；legacy input 经 normalization 后通过 `-verify-each`。

- [ ] **Step 7: 同步设计文档并提交**

确认设计 Section 5、7、12 不再把旧 interface 写成生产路径。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test \
  guoqiao/layout_inference_design.md
git commit -m "refactor: normalize legacy fragments to tensor layout IR"
```

### M4 Gate

Run:

```bash
build/bin/frisk-opt test/Transforms/no-legacy-layout-map.mlir \
  -frisk-layout-pipeline -verify-each
rg -n "inferLayout.*DenseMap|DenseMap<Value, Attribute>.*layout" \
  include/Dialect lib/Dialect
```

Expected: pipeline 成功；生产目录旧接口匹配数为 0；Copy/Fill/Gemm/Reduce/Parallel/SCF 都由新 graph 处理。

---

## M5：有限候选、SM90 契约与 conversion 优化

### Task 23: 实现连通分量候选求解和 CostVector

**Files:**

- Create: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Create: `lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Modify: `include/Dialect/Frisk/Analysis/LayoutTarget.h`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutCandidateSolverTest.cpp`
- Create: `test/Transforms/layout-candidate-selection.mlir`

**Interfaces:**

- Consumes the `CostVector` definition from Task 5; produces assignment and solver APIs:

```cpp
bool operator<(const CostVector &lhs, const CostVector &rhs);

struct CandidateAssignment {
  DenseMap<LayoutVarID, unsigned> candidateIndex;
  SmallVector<LayoutConversionEdge> conversions;
  CostVector cost;
};

struct SolverOptions {
  unsigned maxCandidatesPerComponent = 256;
  unsigned beamWidth = 32;
};

FailureOr<LayoutSolution> solveLayoutGraph(LayoutConstraintGraph &graph,
                                           LayoutTarget &target,
                                           SolverOptions options);
```

- [ ] **Step 1: 写全局选择红灯测试**

构造三个变量的 component，使逐点最小选择违反中间 SameLayout，而全局 assignment 有唯一合法最小值；另测相同 cost 依赖 stable tie-break，不依赖 candidate 插入顺序。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，candidate solver 不存在。

- [ ] **Step 2: 构建连通分量和 hard pruning**

只使用 hard constraint 建 component。候选先 canonical hash 去重，再逐个验证；pruning 必须记录：

```cpp
struct CandidateRejection {
  LayoutVarID var;
  unsigned candidateIndex;
  LayoutConstraintID constraint;
  std::string reason;
};
```

正式 solver 必须先覆盖 bootstrap 已支持的 `KeepCommonLayout/Convert` edge resolution：从 edge 中已有的 bytes/synchronization 上界构造保守 conversion cost，并保持 M3 的 conversion materialization 接口。完成生产接线后，删除 `solveBootstrapLayoutGraph`、`BootstrapSolverLimits` 及生产 pass 对它们的调用；用相同的 M2/M3 case 验证正式 solver 保持语义结果，并允许因完整 CostVector 选择更优的合法 assignment。Rematerialization 和更精确的 critical-path conversion cost 留到 Task 24。

- [ ] **Step 3: 实现有上限 beam search**

变量选择顺序固定为：candidate 数最少、hard degree 最大、stable ID 最小。每扩展一个变量立即做 partial hard check 和 lower-bound cost；beam 使用 `CostVector + stable assignment key` 排序。

超过 256 个原始组合时不静默截断：打印 component ID、vars、每个 domain size、pruning 数和 beamWidth。

- [ ] **Step 4: 实现 CostVector 比较**

非法候选在进入 cost 前已经删除。比较顺序按照字段声明；`deterministicTieBreak` 只能解决其他字段完全相等，不能编码性能偏好。

- [ ] **Step 5: 验证最优性和确定性**

对总组合数不超过 4096 的随机小图，用 exhaustive solver 作为 oracle，比较 beam/exact assignment；首版测试图设置 beamWidth 足够覆盖 exact solution。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutCandidateSolverTest.*'
```

Expected: PASS。

- [ ] **Step 6: 提交候选 solver**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis test unittests
git commit -m "feat: select global layout candidate assignments"
```

### Task 24: 将 conversion placement 和 rematerialization 纳入普通候选

**Files:**

- Modify: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`
- Modify: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutConversionChoiceTest.cpp`
- Create: `test/Transforms/layout-conversion-choice.mlir`

**Interfaces:**

- Produces:

```cpp
enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution;
  uint64_t bytes;
  uint64_t synchronizationCost;
};

bool isRematerializable(Operation *op);
FailureOr<CostVector> evaluateConversionEdge(const LayoutConversionEdge &edge,
                                             LayoutTarget &target);
```

- [ ] **Step 1: 写“零转换不是绝对优先”的红灯测试**

构造：

```text
方案 A：零 conversion，instructionPathAndWork = 100
方案 B：一次 conversion，instructionPathAndWork = 10，conversion cost = 8
```

预期选择 B。再构造主路径相同而 conversion 更重的 case，预期选择零 conversion。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，solver 尚未枚举 edge resolution。

- [ ] **Step 2: 枚举 edge resolution**

对于 `Convertible` soft conflict 同时生成：共同布局、consumer-edge conversion、pure producer rematerialization。Hard seed 冲突不得通过 conversion 掩盖。

- [ ] **Step 3: 计算 conversion 成本**

至少包含元素 bit 数、tile volume、是否跨 warp、是否需要 shared staging、barrier 数和所在 loop depth。未知 lowering 使用保守上界并标记 provenance，不能记为 0。

- [ ] **Step 4: 限制 rematerialization**

只允许 `MemoryEffectOpInterface` 证明无副作用、无 region、无随机/时钟语义、成本低于 conversion 的 backward slice。最大 slice op 数首版设为 8；超过时不枚举。

- [ ] **Step 5: 验证选择与 IR 物化**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutConversionChoiceTest.*'
```

Expected: 两个成本方向测试均 PASS；FileCheck 中 conversion 数和位置与 solution 一致。

- [ ] **Step 6: 同步 conversion 设计并提交**

确认 `layout_inference_design.md` Section 8.7–8.9 与实际普通候选策略一致。

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis \
  test unittests guoqiao/layout_inference_design.md
git commit -m "feat: solve layout conversion placement globally"
```

### Task 25: 完成数据驱动的 SM90 WGMMA/TMA 布局契约

**Files:**

- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90GemmConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90ReduceConstraints.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp`
- Create: `include/Dialect/Frisk/Target/SM90/SM90LayoutTarget.h`
- Create: `unittests/Dialect/Frisk/Layout/SM90ContractTest.cpp`
- Create: `test/Transforms/sm90-layout-contracts.mlir`

**Interfaces:**

- Produces数据表：

```cpp
enum class SM90ElementType { F16, BF16, F32, TF32, I8 };

struct WGMMAContract {
  int64_t m;
  int64_t n;
  int64_t k;
  SM90ElementType aType;
  SM90ElementType bType;
  SM90ElementType accType;
  bool transA;
  bool transB;
  unsigned threads = 128;
};

struct TMAContract {
  unsigned rank;
  unsigned swizzleBytes;
  unsigned baseAlignmentBytes;
  unsigned innermostGranularityBytes;
};
```

- [ ] **Step 1: 写已知合法/非法契约红灯测试**

至少覆盖 BF16/FP16 WGMMA、A/B major mode、128-thread warp-group、32B/64B/128B swizzle、128B base alignment、TMA box/granularity、ragged fallback。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，规则表/成本尚不完整。

- [ ] **Step 2: 固化 target 查询**

`SM90LayoutTarget` 从 `#nvvm.target`/DLTI 查询 chip/features；迁移期允许读取 `frisk.target = "sm_90"`，但立即规范化为 target object。通用 Analysis 不允许比较字符串。

- [ ] **Step 3: 实现 WGMMA candidate expansion**

每条 contract 展开为 accumulator、dot operand、shared storage canonical maps；展开后调用通用 verifier。MMA Attr 只是 contract view，必须能返回 canonical `DistributedEncodingAttr`。

- [ ] **Step 4: 实现 TMA/Copy candidates**

TMA 不合法时保留 cp.async/vector copy candidate；不能把 fallback 当 hard error。Swizzle/padding 必须同时计入 allocation size、bank 和 descriptor legality。

- [ ] **Step 5: 实现 shape-aware target cost**

成本来自实际 tile volume、transaction、bank、barrier 和资源估计，不使用一个对所有 shape 固定的“WGMMA 永远优先”常数。Static contract 未命中时只删除对应 candidate，不影响仍正确的 fallback。

- [ ] **Step 6: 与现有 layout tool/legacy baseline 差分**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests frisk_layout_tool check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='SM90ContractTest.*'
./build/exp/layout/frisk_layout_tool \
  --target=sm_90 --block-m=128 --block-n=128 --block-k=64 \
  --threads=128 --a-space=shared --b-space=shared --dtype=fp16 --ldmatrix
```

Expected: tests PASS；核心 baseline canonical map 等价；非法 TMA case 仍有合法 fallback。

- [ ] **Step 7: 提交 SM90 规则库**

```bash
git add include/Dialect/Frisk/Target lib/Dialect/Frisk/Target test unittests
git commit -m "feat: add SM90 layout contract library"
```

### Task 26: 实现 controlled relaxation、conversion hoist 和受控重计算

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutRelaxationTest.cpp`
- Create: `test/Transforms/layout-relaxation.mlir`
- Create: `test/Transforms/hoist-layout-conversion.mlir`

**Interfaces:**

- Produces:

```cpp
enum class RelaxationKind {
  Replication,
  StoragePaddingOrNarrowerVector,
  GuardedRagged,
  TMAFallback,
  WGMMAFallback
};

FailureOr<LayoutSolution> attemptControlledRelaxation(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    SolverOptions options, SmallVectorImpl<RelaxationRecord> &records);
```

- [ ] **Step 1: 写 relaxation 顺序红灯测试**

覆盖：普通 conversion 候选存在时不进入 relaxation；TMA alignment 不满足选择 cp.async；ragged shape 选择 guarded candidate；WGMMA contract 合法时不得为减少 conversion 回退 SIMT。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，relaxation records 不存在。

- [ ] **Step 2: 实现严格有序 relaxation**

仅当普通 candidate domain 为空时依次尝试：replication、padding/窄 vector、guarded ragged、TMA fallback、最后 WGMMA illegal fallback。每一步从原始 graph 快照重建 domain，不能累积未选松弛。

- [ ] **Step 3: 记录原因和增量成本**

```cpp
struct RelaxationRecord {
  RelaxationKind kind;
  LayoutConstraintID trigger;
  CostVector incrementalCost;
  std::string reason;
};
```

诊断/debug IR 可打印 records；release IR 不保留非必要 debug attr。

- [ ] **Step 4: 实现 conversion hoist**

允许越过 pure elementwise、broadcast/extend 和 loop-invariant producer；不得越过 memory effects、未知 region、barrier 或 ownership boundary。Hoist 前后都运行 dominance 和 verifier。

- [ ] **Step 5: 实现受控 rematerialization rewrite**

只重写 solver 已选择 `Rematerialize` 的 slice；使用 `IRMapping`，共享已有 rematerialized value，避免指数复制；完成后删除死 conversion 并运行 canonicalizer。

- [ ] **Step 6: 验证 relaxation/hoist**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutRelaxationTest.*'
```

Expected: PASS；每个 case 的 relaxation kind、conversion count 和位置稳定。

- [ ] **Step 7: 同步设计并提交**

```bash
git add lib/Dialect/Frisk test unittests guoqiao/layout_inference_design.md
git commit -m "feat: relax and optimize layout assignments safely"
```

### M5 Gate

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
build/bin/frisk-opt test/Transforms/layout-conversion-choice.mlir \
  -frisk-layout-pipeline -verify-each
```

Expected: exhaustive 小图 oracle、全局成本选择、SM90 contracts、conversion/rematerialization 和 relaxation 全部通过；不存在“零转换绝对优先”行为。

---

## M6：正确性、差分测试与性能验收

### Task 27: 建立 property、随机图和确定性测试

**Files:**

- Create: `unittests/Dialect/Frisk/Layout/LayoutPropertyTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutSolverPropertyTest.cpp`
- Create: `test/Transforms/layout-diagnostics.mlir`
- Create: `test/Transforms/layout-determinism.mlir`
- Create: `tools/layout-reduce/CMakeLists.txt`
- Create: `tools/layout-reduce/layout-reduce.cpp`
- Modify: `tools/CMakeLists.txt`

**Interfaces:**

- Produces固定 seed property runner 和失败图最小化工具：

```cpp
struct LayoutPropertyOptions {
  uint64_t seed;
  unsigned cases;
  unsigned maxBits;
  unsigned maxVars;
};

FailureOr<LayoutConstraintGraph>
reduceFailingLayoutGraph(const LayoutConstraintGraph &graph,
                         function_ref<bool(const LayoutConstraintGraph &)> fails);
```

- [ ] **Step 1: 写会暴露顺序问题的随机测试**

固定 seeds `{1, 7, 42, 20260816}`，生成最多 8 input bits、8 vars、16 constraints 的可枚举小图；对每张图执行 20 次稳定 shuffle，并比较 canonical solution dump。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: 在最小化工具和完整确定性保障实现前，测试应至少缺少目标或失败。

- [ ] **Step 2: 增加代数 properties**

必须验证：

```text
identity compose A == A
(A compose B)(x) == A(B(x))
inverse(A)(A(x)) == x when invertible
rightInverse covers every required logical point
canonicalize(canonicalize(A)) == canonicalize(A)
Product evaluation == outer/inner staged evaluation
```

- [ ] **Step 3: 增加 solver properties**

小图与 exhaustive solver 比较合法性和最小 CostVector；所有 hard constraint 在 solution 上重新执行；空 domain 必须有至少一个 rejection/provenance record。

- [ ] **Step 4: 实现失败图最小化**

按稳定顺序尝试删除 constraint、var、candidate，保留仍触发同一 failure signature 的最小子图；输出可被 unit test parser 重新加载的文本，不输出地址。

- [ ] **Step 5: 验证 diagnostics 和 determinism**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests layout-reduce check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutPropertyTest.*:LayoutSolverPropertyTest.*'
```

Expected: 所有固定 seed PASS；重复执行 canonical output 字节相同。

- [ ] **Step 6: 提交 property suite**

```bash
git add unittests test tools
git commit -m "test: add layout property and determinism coverage"
```

### Task 28: 建立固定 TileLang 语义差分 corpus

**参考版本决策（2026-09-12）：** 本任务尚未执行。以下 schema/exporter/命令的拟定主基线升级为本轮审计的 `5e149e3`；这不是已经更新或运行了 corpus。旧 `6623b12` 可另存带版本标签的历史回归，不混入新版 schema。PartialFragment/combine 语义若纳入 corpus，须先扩展 schema 区分 addends 与同值 copy groups；不得继续仅以 replication 数字代表全部语义。

**Files:**

- Create: `tools/layout-reference/export_tilelang_layouts.py`
- Create: `test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json`
- Create: `unittests/Dialect/Frisk/Layout/TileLangDifferentialTest.cpp`
- Create: `guoqiao/tilelang_layout_reference.md`

**Interfaces:**

- Corpus schema：

```json
{
  "schema_version": 1,
  "tilelang_commit": "5e149e31674658f94779c7d0c6039549a1853123",
  "cases": [
    {
      "name": "sm90_bf16_gemm_128x128x64",
      "shape": [128, 128, 64],
      "dtype": "bf16",
      "carrier_extents": {"register": 8, "lane": 32, "warp": 4},
      "distributed_points": [],
      "storage_points": [],
      "replication": 1,
      "writer_owners": []
    }
  ]
}
```

测试只读取提交到仓库的 JSON，不要求 CI 安装 TileLang。

- [ ] **Step 1: 写 corpus reader 红灯测试**

测试拒绝 schema 版本错误、commit 不匹配、重复 logical point owner 和越界 storage offset。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，corpus/reader 尚不存在。

- [ ] **Step 2: 实现固定版本 exporter**

脚本启动时执行：

```python
expected = "5e149e31674658f94779c7d0c6039549a1853123"
if args.revision != expected:
    raise SystemExit(f"unsupported TileLang revision: {args.revision}")
actual = subprocess.check_output(
    ["git", "-C", args.tilelang, "rev-parse", "HEAD"], text=True
).strip()
if actual != expected:
    raise SystemExit(f"TileLang revision mismatch: {actual} != {expected}")
```

Exporter 输出枚举语义，不输出 TileLang Attr 文本或 Python 对象 repr；JSON key 排序、数字格式稳定。

- [ ] **Step 3: 生成首批 corpus**

至少包含：

```text
SM90 BF16/FP16 GEMM
linear/transpose/32B/64B/128B shared swizzle
copy vector widths
reduce dim 0/1
non-power-of-two broadcast
ragged padding guard
dtype-changing view
unused/floating fragment
owner compatibility
```

不得直接从 `/home/baopeihua/tilelang` 当前 checkout 导入模块，也不假定其中已有新版对象。在独立 reference clone 中获取 pinned commit，再创建 detached worktree；clone 的 main 不作为语义参考，exporter 必须检查 worktree 的精确 `HEAD`。以下路径是任务实施时新建的示例，已存在时先检查而非覆盖：

```bash
git clone --no-checkout https://github.com/tile-ai/tilelang.git \
  /tmp/tilelang-layout-reference-repo
git -C /tmp/tilelang-layout-reference-repo worktree add --detach \
  /tmp/tilelang-layout-reference-5e149e3 \
  5e149e31674658f94779c7d0c6039549a1853123
python tools/layout-reference/export_tilelang_layouts.py \
  --tilelang /tmp/tilelang-layout-reference-5e149e3 \
  --revision 5e149e31674658f94779c7d0c6039549a1853123 \
  --output test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json
```

Expected: exporter 成功，第二次执行产生完全相同文件。生成和审计结束后可执行：

```bash
git -C /tmp/tilelang-layout-reference-repo worktree remove \
  /tmp/tilelang-layout-reference-5e149e3
```

Exporter 还必须把 pinned worktree 插入独立 Python 进程的首个 import path，并拒绝从系统 site-packages 导入 TileLang；启动日志打印实际导入模块路径和 commit，文档审计时一并核对。

- [ ] **Step 4: 实现语义比较**

Frisk 对每个 case 枚举 canonical map，比较 coverage、logical ownership、replication、writer owner 和 storage bit offset。允许不同构造文本，只比较语义；失败打印首个 carrier/logical coordinate 反例。

- [ ] **Step 5: 运行差分测试**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='TileLangDifferentialTest.*'
```

Expected: corpus 全部 PASS。

- [ ] **Step 6: 记录参考策略并提交**

`guoqiao/tilelang_layout_reference.md` 记录 commit、schema、重新生成命令、采用/拒绝的语义和更新审计流程。

```bash
git add tools/layout-reference test/Dialect/Frisk/layout/Inputs \
  unittests/Dialect/Frisk/Layout/TileLangDifferentialTest.cpp \
  guoqiao/tilelang_layout_reference.md
git commit -m "test: add pinned TileLang layout corpus"
```

### Task 29: 建立性能、编译开销守门并完成系统验收

**Files:**

- Modify: `CMakeLists.txt`
- Create: `include/Dialect/Frisk/Analysis/LayoutStatistics.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutStatistics.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Create: `benchmark/CMakeLists.txt`
- Create: `benchmark/layout/CMakeLists.txt`
- Create: `benchmark/layout/solver/CMakeLists.txt`
- Create: `benchmark/layout/solver/layout_solver_bench.cpp`
- Create: `benchmark/layout/conversion/CMakeLists.txt`
- Create: `benchmark/layout/conversion/layout_conversion_bench.cu`
- Create: `benchmark/layout/run_layout_benchmarks.py`
- Create: `benchmark/layout/baseline/sm90_layout_baseline.json`
- Create: `guoqiao/layout_inference_acceptance.md`
- Modify: `guoqiao/layout_inference_design.md` if measured limits require an approved change

**Interfaces:**

- Produces：

```cpp
struct LayoutStatistics {
  uint64_t variables;
  uint64_t constraints;
  uint64_t components;
  uint64_t generatedCandidates;
  uint64_t prunedCandidates;
  uint64_t peakComponentCandidates;
  uint64_t conversions;
  uint64_t conversionBytes;
  uint64_t estimatedBankConflicts;
  uint64_t estimatedMemoryTransactions;
  std::chrono::nanoseconds solveTime;
};
```

Pass option `-frisk-layout-stats-file=<path>` 输出稳定 JSON；默认不写文件。

- [ ] **Step 1: 写 statistics 红灯测试**

用固定小图检查 vars/constraints/candidates/conversions 数量和稳定 JSON key；连续两次运行除时间字段外完全一致。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，statistics option 尚不存在。

- [ ] **Step 2: 实现统计和 solver microbenchmark**

`layout_solver_bench` 使用固定 corpus，分别测 10、50、100、250 Op graph；预热 5 次、测量 30 次，报告 median/p95、candidate peak 和解 hash。不得在计时区间打印 IR。

- [ ] **Step 3: 实现可选 CUDA conversion microbenchmark**

CMake option：

```cmake
option(FRISK_ENABLE_CUDA_BENCHMARKS
       "Build SM90 layout conversion benchmarks" OFF)
```

启用时要求 `nvcc` 和 SM90；至少比较 identity、单 warp shuffle、shared exchange 三类 conversion。每类 case 同时提供人工编写的 `reference_*` kernel 和通过 `TestLayoutConversionLowering.cpp` 生成/调用的 `inferred_*` 路径；两者使用同一输入和校验器。每个 case 校验输出，再预热 100 次、测量 1000 次；记录 GPU、driver、clock policy 和编译 flags。

顶层仅在 `FRISK_ENABLE_CUDA_BENCHMARKS=ON` 时执行 `enable_language(CUDA)` 并加入 conversion 子目录；默认 CPU/CI 构建不得依赖 CUDA toolkit。

- [ ] **Step 4: 实现 benchmark runner 和 baseline 比较**

Runner 读取/写入：

```json
{
  "environment": {},
  "solver": {},
  "conversions": {},
  "supported_kernels": {}
}
```

baseline JSON 中 conversion/runtime 数值必须来自同一固定环境下的 `reference_*` kernel，不能把新 solver 首次运行结果自封为基线。比较规则：正确性失败立即退出；已有 lowering 的 `inferred_*` 对相应 `reference_*` median 回退超过 3% 失败；布局推断占完整编译时间超过 10% 失败；peak component candidates 超过 256 失败。时间噪声在 3% 内视为持平。

本计划的 runtime 守门范围只覆盖 conversion adapter 和仓库中已经存在、能走新布局 pipeline 的 kernel；尚无 WGMMA/TMA lowering 的路径只做静态 contract/cost 验证，不伪造 runtime 数据。

- [ ] **Step 5: 运行 CPU/solver 验收**

Run:

```bash
cmake --build build --target frisk_layout_solver_bench check-frisk FriskLayoutUnitTests --parallel 32
python benchmark/layout/run_layout_benchmarks.py \
  --solver build/benchmark/layout/solver/frisk_layout_solver_bench \
  --baseline benchmark/layout/baseline/sm90_layout_baseline.json \
  --check
```

Expected: correctness/compile-time/candidate gates PASS。

- [ ] **Step 6: 在 SM90 环境运行 conversion/runtime 验收**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir \
  -DFRISK_ENABLE_CUDA_BENCHMARKS=ON
cmake --build build --target frisk_layout_conversion_bench --parallel 32
python benchmark/layout/run_layout_benchmarks.py \
  --conversion build/benchmark/layout/conversion/frisk_layout_conversion_bench \
  --record-reference \
  --output-baseline benchmark/layout/baseline/sm90_layout_baseline.json
python benchmark/layout/run_layout_benchmarks.py \
  --conversion build/benchmark/layout/conversion/frisk_layout_conversion_bench \
  --baseline benchmark/layout/baseline/sm90_layout_baseline.json \
  --check
```

Expected: 第一条 runner 命令在记录 GPU/driver/clock/flags 后生成 reference baseline；第二条 GPU correctness PASS，且受支持 case 不超过已批准回退阈值。baseline 的新增或更新必须由性能评审确认；没有 SM90 机器时不得生成 baseline，也不得把本步骤标记完成。

- [ ] **Step 7: 执行完整最终验证**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target FriskIR FriskTransforms frisk-opt \
  FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
git diff --check
```

Expected: 所有构建、CTest、lit、unit tests 和 diff check PASS。

- [ ] **Step 8: 编写验收记录并同步设计**

`layout_inference_acceptance.md` 必须列出每个 Global Constraint 的证据命令、结果、未覆盖项和后续 WGMMA/TMA lowering 边界。若实测要求改变阈值，必须先获得架构评审再修改设计文档，不能为让 benchmark 通过而直接放宽。

- [ ] **Step 9: 提交性能守门和验收记录**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis \
  lib/Dialect/Frisk/Transforms benchmark guoqiao/layout_inference_acceptance.md \
  guoqiao/layout_inference_design.md
git commit -m "test: gate layout correctness and performance"
```

### M6 Gate

必须具备以下证据：

```text
CTest/lit/unit/property 全部通过
TileLang 固定 corpus 全部通过
相同输入重复编译产生相同布局 IR
race/OOB/invalid ownership/unresolved var 为 0
常规 component candidates <= 256
beam width = 32
布局推断编译时间占比 <= 10%
受支持 SM90 runtime case 中位回退 <= 3%
尚无 lowering 的 WGMMA/TMA 路径仅声明静态契约通过
```

任一项缺少证据时，布局系统不能标记完成。

---

## 3. 最终 Definition of Done

- [ ] M0–M6 的全部 Gate 均有最新命令输出。
- [ ] `frisk-layout-pipeline` 对所有受支持 Frisk kernel 生成合法、已解析、可验证的布局 IR。
- [ ] 生产源码中不存在 `inferLayout(builder, DenseMap<Value, Attribute>&)`。
- [ ] Local/Register 的生产 IR 不再以 local MemRef 作为求解载体。
- [ ] Distributed/Storage layout 映射方向、Attr 字段和 verifier 与设计文档一致。
- [ ] Hard conflict 输出 provenance chain 和坐标反例。
- [ ] 多 consumer 可以选择共同布局、conversion 或 rematerialization。
- [ ] conversion 是普通成本候选，且不会被零转换绝对优先规则压制。
- [ ] SM90 WGMMA/TMA/shared swizzle contracts 有正反例测试。
- [ ] TileLang 固定语义 corpus、property tests 和顺序确定性测试通过。
- [ ] CPU/solver 与 SM90 conversion/runtime 性能守门通过。
- [ ] `guoqiao/layout_inference_design.md`、对比文档、实施计划和验收记录不存在语义冲突。
- [ ] 完整 WGMMA/TMA lowering 的未交付边界在验收记录中明确保留。

## 4. 计划执行方式

严格按 Task 1 → Task 29 顺序执行。每完成一个 Task：

1. 运行该任务局部测试；
2. 运行所在里程碑已有回归；
3. 检查设计文档漂移；
4. 提交独立 commit；
5. 进行一次 spec compliance review 和一次 code quality review；
6. 只有 review 通过才进入下一个 Task。

建议在每个 M0–M6 Gate 后由用户进行一次阶段验收。M2、M3、M4、M5 的 Gate 是架构关口，不建议并行跨越。
