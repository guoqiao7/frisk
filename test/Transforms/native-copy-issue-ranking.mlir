// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer native-resident-atoms=1 native-delivery=dd native-sm-count=114 native-grid-ctas=32" -verify-each | FileCheck %s --check-prefix=THIN
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer native-resident-atoms=1 native-delivery=dd native-sm-count=114 native-grid-ctas=132" -verify-each | FileCheck %s --check-prefix=BUSY
// The same protected domain reverses its selection when the launch fills the GPU.
// THIN: = frisk.mma
// THIN-SAME: grid = array<i64: 2, 1>
// THIN-SAME: row_partition = "interleaved64"
// THIN-SAME: frisk.native_slice_traffic = {copy_range_visits = 128
// THIN-SAME: idle_copy_lanes = 0
// BUSY: = frisk.mma
// BUSY-SAME: grid = array<i64: 1, 2>
// BUSY-SAME: frisk.native_slice_traffic = {copy_range_visits = 98
// BUSY-SAME: idle_copy_lanes = 8192
module attributes {frisk.target = "sm_90a"} {
  func.func @selection(%a: memref<256x256xf16, 1>, %b: memref<256x512xf16, 1>) {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<256x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<256x16xf16, 3> -> memref<256x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x512xf32>
    %lo = arith.constant 0 : index
    %hi = arith.constant 256 : index
    %step = arith.constant 16 : index
    %2 = scf.for %iv = %lo to %hi step %step iter_args(%acc = %cst) -> (tensor<256x512xf32>) {
      %at = memref.subview %a[0,%iv][256,16][1,1] : memref<256x256xf16, 1> to memref<256x16xf16, strided<[256,1], offset: ?>, 1>
      %bt = memref.subview %b[%iv,0][16,512][1,1] : memref<256x512xf16, 1> to memref<16x512xf16, strided<[512,1], offset: ?>, 1>
      memref.copy %at, %alloc : memref<256x16xf16, strided<[256,1], offset: ?>, 1> to memref<256x16xf16, 3>
      memref.copy %bt, %alloc_0 : memref<16x512xf16, strided<[512,1], offset: ?>, 1> to memref<16x512xf16, 3>
    %next = frisk.mma %0, %1, %acc {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 256 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<256x16xf16, 3>, memref<16x512xf16, 3>, tensor<256x512xf32> -> tensor<256x512xf32>
      scf.yield %next : tensor<256x512xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<256x512xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x512xf32, 1> -> memref<256x512xf32, 1>
    frisk.tile_store %2, %3 : tensor<256x512xf32>, memref<256x512xf32, 1>
    %rs = frisk.reduce_tensor %2 {dim = 1 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<256x512xf32> -> tensor<256xf32>
    %redout = memref.alloc() {alignment = 128 : i64} : memref<256xf32, 1>
    %dst = frisk.layout_view %redout {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [256], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256xf32, 1> -> memref<256xf32, 1>
    frisk.tile_store %rs, %dst : tensor<256xf32>, memref<256xf32, 1>
    return
  }
}
