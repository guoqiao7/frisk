// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-multi-anchor-layout-pipeline -verify-each | FileCheck %s
// A scalar C store has 4096 warp issues. The canonical WGMMA owner map
// touches eight 32-byte sectors per issue; the column store adds 256 units.
// CHECK: access_units = 37120
// CHECK: frisk.mma
// CHECK: frisk.tile_store
// CHECK: frisk.reduce_tensor
// CHECK-SAME: canonical_fiber_tree_v1
// CHECK: frisk.tile_store
module attributes {frisk.target = "sm_90a"} {
  func.func @selection(%a: memref<256x96xf16, 1>, %b: memref<96x512xf16, 1>) {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<256x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<256x16xf16, 3> -> memref<256x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x512xf32>
    %2 = frisk.mma %0, %1, %cst {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 256 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<256x16xf16, 3>, memref<16x512xf16, 3>, tensor<256x512xf32> -> tensor<256x512xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<256x512xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x512xf32, 1> -> memref<256x512xf32, 1>
    frisk.tile_store %2, %3 : tensor<256x512xf32>, memref<256x512xf32, 1>
    %rs = frisk.reduce_tensor %2 {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<256x512xf32> -> tensor<512xf32>
    %redout = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %dst = frisk.layout_view %redout {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %rs, %dst : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}
