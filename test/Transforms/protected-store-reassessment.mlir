// RUN: python3 %S/Inputs/check-protected-store-routing.py history frisk-opt %s
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x128xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x128xf16, 3> -> memref<16x128xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %2 = frisk.mma %0, %1, %cst {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy FullCol>} : memref<128x16xf16, 3>, memref<16x128xf16, 3>, tensor<128x128xf32> -> tensor<128x128xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128x128xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [128, 128], outputs = ["byte_offset", "bit_offset"], output_extents = [65536, 8], map = affine_map<(d0, d1) -> (d0 * 512 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128x128xf32, 1> -> memref<128x128xf32, 1>
    frisk.tile_store %2, %3 : tensor<128x128xf32>, memref<128x128xf32, 1>
    return
  }
}