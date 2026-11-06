# Frisk Layout Legacy Baseline

> 冻结日期：2026-08-30
>
> 作用：记录 M0 开始时已有布局测试的语义覆盖。后续布局系统迁移必须通过这些用例，除非同一任务明确更新设计、测试和本基线。

## 运行入口

```bash
cmake --build build --target frisk_attr_test frisk_reduce_layout_test --parallel 32
ctest --test-dir build --output-on-failure
```

M0 修改前，两个测试二进制均退出 0，但 CTest 尚未注册任何测试：

```text
ctest --test-dir build -N
Total Tests: 0
```

## `FriskAttrTest` 覆盖

- `MemorySpaceAttr`：Local、Global、Shared 构造，以及 Shared parse。
- `GemmWarpPolicyAttr`：Square、FullRow、FullCol 构造，以及 FullRow parse。
- `LayoutAttr`：二维线性 affine layout 构造。
- GEMM legacy layout inference：`M=N=128`、`K=64`、128 threads、FP16。
  - SM80：shared/shared、register/shared、shared/register。
  - SM90：shared/shared、register/shared。
  - 检查 A/B storage 或 fragment layout 以及 C accumulator fragment layout 能被推断。
- Reduce legacy layout inference：从二维 fragment 推导一维 reduce result layout。

## `FriskReduceLayoutTest` 覆盖

- 8x8 tensor-core 基础 fragment 的按行、按列归约。
- 单 warp 16x8、双 warp 32x8 fragment 的按行归约。
- 输入已经 replication 的归约。
- 被归约维度不参与 thread map 的情况。
- 三维 batch tile 的中间维与末维归约。
- GEMM C fragment 到 Reduce 的集成路径：
  - SM80 128x128，dim=1，128 threads。
  - SM80 64x64，dim=0，128 threads。
  - SM90 128x128，dim=1，128 threads。
  - SM80 128x64，dim=1，256 threads。

## 已知限制

- 这些是直接调用现有 C++ Attr/Op 推断接口的 legacy host tests，不经过独立 layout inference pass。
- Register fragment 仍由 MemRef/legacy `LayoutAttr` 表达，不是最终的 RankedTensor SSA 与 `DistributedEncodingAttr`。
- Shared XOR storage、TMA/WGMMA 指令契约、显式 conversion、候选求解和完整 verifier 尚未覆盖。
- 测试不执行 Frisk 到 NVGPU/NVVM/PTX 的 lowering，也不在 GPU 上验证正确性或性能。
- 当前主要是正向固定样例，不构成 coverage、ownership、OOB、确定性和随机图性质测试。
