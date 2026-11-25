# Frisk

This project is an high-performance operator compiler designed to unlock the full potential of computational kernels across heterogeneous hardware. Built upon the MLIR (Multi-Level Intermediate Representation) framework, it leverages modern compiler technologies to generate highly optimized code. Currently in its early development phase, the compiler initially targets GPU platforms from NVIDIA, AMD, and HYGON, with a foundational architecture poised for future expansion to other accelerators.



## ✨ Installation

### Prerequisites
Before compiling from source, ensure the following dependencies are met:
- Pythonversion must be ≥ 3.9
- CMakeversion must be ≥ 3.18
- Ninjabuild system
- pybind11 library
- LLVM/MLIR(needs to be compiled from source)
### Compiling LLVM/MLIR
```bash
# Clone the repository
git clone https://github.com/DeepGenGroup/rocm-llvm-project
cd rocm-llvm-project
git checkout deepgen-dev

# Compile
mkdir build && cd build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_INSTALL_PREFIX=/path/to/llvm/install

# Build (add -jN for parallel build, e.g., -j8 for 8 threads)
ninja
```

### Building from Source
Clone and build the project:
```bash
git clone https://github.com/Xiebaokang/frisk
cd frisk
# Execute the build script
./rebuild.sh
```
Note:After installation, you need to set the Python path:
```bash
export PYTHONPATH=~/install/path/frisk/python:$PYTHONPATH
```



## 🚀 Example

Here's a simple example demonstrating how to create a GEMM IR expression:
```bash
cd /path/to/frisk
python test/python/bind_test.py
```
This will generate and execute a GEMM kernel using the Frisk compiler.


## Layout inference and native execution

Use `frisk-opt` to normalize and solve layout IR. The SM90 multi-anchor path is
`--frisk-multi-anchor-layout-pipeline`. `--frisk-optimize-layout-conversions`
materializes consumer deliveries and `--frisk-export-execution` exports the packet.
The MLIR passes live in `lib/Dialect/Frisk/Transforms`. Target-specific legality
and cost models live in `lib/Dialect/Frisk/Target/SM90`.

The H100 research backend is in `benchmark/layout/execution`. Use
`backend.compile_execution` for a solved packet or a native dense GEMM contract.
For attention, `attention_pipeline_selection` exposes static D64 MHA and grouped
selection and measured tuning. They share the physical plans in
`attention_pipeline_candidates`. Use `native_fa3_execution.compile_static_fa3`
for imported FA3 schedules. Its CLI also replays explicitly named event orders.
A solved layout fixes the execution domain and transport before an order is emitted.

For static SM90 layout inspection, build `frisk_layout_tool` as described in
[the layout tool guide](exp/layout/README.md). Compiler regressions live in
`test/Transforms` and `unittests`. The Python backend and scoring tests live
alongside their implementations.

The [H100 paper regression](benchmark/layout/paper/README.md) contains the
fixed-pipeline comparisons, native kernel measurements, certified anchor
experiments, raw paired timings, and hardware replay instructions.
