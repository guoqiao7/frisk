# SM90 layout inspection

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
