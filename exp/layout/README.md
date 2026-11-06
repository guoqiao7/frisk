# Task 25 canonical tool

The tool now runs the transactional Frisk layout pipeline and reports actual
canonical storage/distributed attributes and the WGMMA contract as JSON schema
`frisk-canonical-layout-v1`. Use `--target=sm_90a`; `sm_90` does not enable WGMMA.
Supported inputs are FP16/BF16 with FP32 accumulation: A shared/local (SS/RS),
B shared, C local. Unsupported configurations fail instead of changing dtype.

```bash
cmake --build build --target frisk_layout_tool -j 12
build/exp/layout/frisk_layout_tool --target=sm_90a --block-m=64 --block-n=8 --block-k=16
python3 test/python/sm90_layout_tool_test.py --build-dir build
```

`--ldmatrix` and `--lane-vector` are deprecated reporting flags, not instruction
requirements. Only an omitted or all-one grid is accepted; layouts describe one CTA.
The Python wrapper displays the canonical schema without running an unrelated
PyTorch benchmark. Old bank-statistic JSON and the archived outputs below are
historical examples, not current inference or performance evidence. Independent
legacy-map comparisons live in `LegacyLayoutAdapterTest`, not production code.

## Historical experiment documentation

# GEMM Layout Experiments

This directory contains standalone experiments for validating the effect of
`frisk.gemm` layout inference. The toolkit focuses on measuring shared-memory
bank conflicts before and after layout inference, using PyTorch’s default
row-major layout as the baseline.

## Build the analysis tool

Configure and rebuild Frisk as usual; the CMake target `frisk_layout_tool` is
available automatically after a normal `./build.sh` or `./rebuild.sh` run:

```bash
cd /home/frank/mlir/frisk
./rebuild.sh frisk_layout_tool
```

The resulting binary lives at `build/exp/layout/frisk_layout_tool`.

## Bank-conflict benchmarking

`analyze_bank_conflicts.py` drives the C++ tool and (optionally) executes a
PyTorch GEMM as the baseline reference:

```bash
cd /home/frank/mlir/frisk
python exp/layout/analyze_bank_conflicts.py \
    --tool ./build/exp/layout/frisk_layout_tool \
    --target sm_80 \
    --shape 128,128,64 \
    --threads 128 \
    --a-space shared \
    --b-space shared \
    --dtype fp16 \
    --ldmatrix
```

What the script does:

1. Runs `frisk_layout_tool` with the requested configuration to infer layouts,
   then parses the emitted JSON to extract layout attributes and predicted bank
   conflict metrics.
2. Treats row-major accesses (PyTorch’s default layout) as the baseline and
   reports their bank conflicts for comparison.
3. If PyTorch is available (`pip install torch`), executes the equivalent
   `torch.matmul` (using CUDA when available, CPU otherwise) to record the
   achieved TFLOPS. This is a convenience baseline so layout experiments can be
   correlated with real GEMM performance.
4. The new `--lane-vector N` option (or the shorthand `--ldmatrix`, which forces
   `N >= 4`) models vectorized loads such as `ldmatrix.x4`, so the bank
   statistics reflect realistic warp memory transactions instead of scalar
   loads.

Sample output highlights (abridged):

```
== Config ==
target      : sm_80
shape       : M=128 N=128 K=64
dtype       : fp16
threads     : 128

Operand A (shared):
  layout index     : affine_map<(d0, d1) -> (d0 * 72 + d1 xor ... )>
  bank stats       : avg_max=1.00  max=1
  baseline (torch) : avg_max=8.00  max=16

Operand B (shared):
  ...

PyTorch baseline throughput: 115.3 TFLOPS (device: cuda:0)
```

Use this report to validate that the inferred layout eliminates shared-memory
bank conflicts relative to the baseline and to correlate with the PyTorch
reference performance.

### Recommended configuration to see layout benefits

Vectorized FP16 shared-memory loads show the starkest contrast between a
row-major layout and Frisk’s inferred swizzles. Use:

```
python exp/layout/analyze_bank_conflicts.py \
    --tool ./build/exp/layout/frisk_layout_tool \
    --target sm_80 \
    --shape 128,128,64 \
    --threads 128 \
    --dtype fp16 \
    --a-space shared \
    --b-space shared \
    --ldmatrix
```

The baseline row-major tensor will typically report `max=8` conflicts (each
lane issues a vectorized 8-byte load), while the Frisk layout keeps every bank
single-ported (`max=1`), showcasing the benefit of layout inference.
