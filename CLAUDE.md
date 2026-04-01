# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Frisk is an MLIR-based high-performance operator compiler targeting GPU platforms (NVIDIA sm_70–sm_100, AMD CDNA). It defines a custom `frisk` MLIR dialect with ops for kernel structure (`KernelOp`, `ParallelOp`, `BlockOp`, `ForOp`), compute (`GemmOp`, `ReduceOp`), and memory (`AllocBufferOp`, `CopyOp`, `FillOp`). The compiler is in early development; the current focus is porting GEMM layout inference from tilelang into MLIR `LayoutAttr`-based representations.

## Build Commands

Requires a custom LLVM/MLIR install from `DeepGenGroup/rocm-llvm-project` (branch `deepgen-dev`). `MLIR_DIR` is hardcoded in `build.sh` to `/home/frank/llvm/install/lib/cmake/mlir`.

```bash
# Clean build (wipes build/, reconfigures, builds, copies .so into python/frisk/)
./build.sh

# Incremental build (reuses existing build tree, ninja -j32)
./rebuild.sh

# Build a single library target inside build/
cd build && ninja FriskIR

# Run native C++ tests (built as executables, not via lit yet)
./build/frisk_attr_test
./build/frisk_reduce_layout_test

# Python smoke tests (set PYTHONPATH first)
export PYTHONPATH=$(pwd)/python:$PYTHONPATH
python test/python/bind_test.py
python test/python/map_expr_test.py
```

## Architecture

### Dialect definition (TableGen → C++)

- **TableGen sources**: `include/Dialect/Frisk/IR/*.td` — defines `FriskDialect`, ops, enums (`MemorySpace`, `GemmWarpPolicy`), attributes (`LayoutAttr`), interfaces (`LayoutInterface`).
- **Generated headers** land in `build/include/` and are included via `#include "Dialect/Frisk/IR/..."`.
- **C++ implementations**: `lib/Dialect/Frisk/IR/` — `FriskOps.cpp` contains op verifiers, custom assembly format, and the bulk of `GemmOp::inferLayout` and target detection logic. `FriskOps_Reduce.cpp` has `ReduceOp::inferLayout`. `FriskAttributes.cpp` / `FriskEnums.cpp` are mostly TableGen-generated glue.
- The single library target `FriskIR` (defined in `lib/Dialect/Frisk/IR/CMakeLists.txt`) compiles all dialect code plus `LayoutUtils.cpp`.

### Layout inference pipeline

The core of the ongoing work. `LayoutAttr` encodes `(inputShape, forwardIndex, forwardThread?, replicateSize?)` where index/thread maps are `AffineMapAttr`. Layout inference is invoked per-op:

- `GemmOp::inferLayout` (in `FriskOps.cpp`) detects the target arch from a `frisk.target` string attribute on the enclosing module, determines `GemmInst` (MMA vs WGMMA), computes warp partitions, then dispatches to arch-specific fragment/shared-memory layout builders that construct `AffineExpr`-based mappings.
- `ReduceOp::inferLayout` (in `FriskOps_Reduce.cpp`) derives destination layouts by compressing the reduced dimension out of the source `LayoutAttr`, using utilities from `LayoutUtils`.
- `LayoutUtils.h/.cpp` provides `computeUsedExtentForDim`, `compressReplicateDimInMap`, and `inferFragmentIndexFromThreadMap` — reusable affine-analysis helpers that mirror tilelang's `CondenseReplicateVar` / `DivideUnusedIterators`.

### Transforms / Passes

- `Passes.td` defines `FriskLayoutInfer` as an `InterfacePass<FunctionOpInterface>`. The implementation in `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp` is currently a stub.

### Python bindings (pybind11)

- `ffi/ir.cpp` is the main binding file (~1600 lines). It exposes `context`, `builder`, `module`, every Frisk op type, `MemorySpace`/`GemmWarpPolicy` enums, and `AffineMap`/`AffineExpr` manipulation to Python.
- `ffi/ffi.cpp` is the pybind11 module entry point (`frisk_ffi`).
- `python/frisk/__init__.py` re-exports everything from `frisk_ffi.ir`. `python/frisk/common/utils.py` provides `FriskHelper` / `create_session` convenience wrappers.
- The compiled `.so` is copied from `build/` into `python/frisk/` by the build scripts.

### Test structure

- `test/python/` — Python integration tests using the pybind11 bindings (`bind_test.py` builds a full kernel IR, `map_expr_test.py` exercises affine map manipulation).
- `test_pass/` — Native C++ test executables (`attr_test.cpp` validates attribute construction/parsing and GEMM layout inference across targets; `reduce_layout_test.cpp` validates reduce layout inference with multiple split patterns).
- `exp/layout/` — Experimental tools (`gemm_layout_tool.cpp` for standalone layout analysis, `analyze_bank_conflicts.py`).

## Coding Conventions

- C++17, LLVM/MLIR style: two-space indent, `UpperCamelCase` types/ops, `lowerCamelCase` functions.
- Include order: `<std>`, LLVM (`llvm/`), MLIR (`mlir/`), local (`Dialect/Frisk/...`).
- Prefer `llvm::SmallVector`, `mlir::LogicalResult`, `AffineExpr`/`AffineMap` over STL equivalents.
- Python code follows PEP 8 with snake_case.
- The `frisk` dialect C++ namespace is `mlir::frisk`; enums live in `mlir::frisk::attr`.

## Key Environment Variable

```bash
export MLIR_DIR=/home/frank/llvm/install/lib/cmake/mlir
```
