// RUN: frisk-opt %s --split-input-file --frisk-protected-store-layout="striped-shared-replay=true" -verify-each | FileCheck %s
// Both local optima survive. Only the matrix store receives a conversion.
// CHECK: eligible = true
// CHECK-SAME: grid = array<i64: 1, 2>
// CHECK-SAME: eligible = true
// CHECK-SAME: grid = array<i64: 2, 1>
// CHECK: func.func @dual_store
// CHECK: [[ACC:%[[:alnum:]]+]] = scf.for
// CHECK: frisk.mma
// CHECK-SAME: grid = array<i64: 1, 2>
// CHECK: [[ROUTED:%[[:alnum:]]+]] = frisk.convert_layout [[ACC]]
// CHECK: frisk.tile_store [[ROUTED]],
// CHECK-SAME: frisk.store_score = array<i64: 16384, 8192, 4160, 1>
// CHECK: [[COL:%[[:alnum:]]+]] = frisk.reduce_tensor [[ACC]]
// CHECK: frisk.tile_store [[COL]],
// CHECK-SAME: frisk.store_score = array<i64: 128, 64, 160, 0>
module attributes {frisk.target = "sm_90a"} {
  func.func @dual_store(%a: memref<128x128xf16, 1>, %b: memref<128x512xf16, 1>) {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x512xf32>
    %lo = arith.constant 0 : index
    %hi = arith.constant 128 : index
    %step = arith.constant 16 : index
    %2 = scf.for %iv = %lo to %hi step %step iter_args(%acc = %cst) -> (tensor<128x512xf32>) {
      %at = memref.subview %a[0,%iv][128,16][1,1] : memref<128x128xf16, 1> to memref<128x16xf16, strided<[128,1], offset: ?>, 1>
      %bt = memref.subview %b[%iv,0][16,512][1,1] : memref<128x512xf16, 1> to memref<16x512xf16, strided<[512,1], offset: ?>, 1>
      memref.copy %at, %alloc : memref<128x16xf16, strided<[128,1], offset: ?>, 1> to memref<128x16xf16, 3>
      memref.copy %bt, %alloc_0 : memref<16x512xf16, strided<[512,1], offset: ?>, 1> to memref<16x512xf16, 3>
    %next = frisk.mma %0, %1, %acc {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<128x16xf16, 3>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
      scf.yield %next : tensor<128x512xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128x512xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [128, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [262144, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128x512xf32, 1> -> memref<128x512xf32, 1>
    frisk.tile_store %2, %3 : tensor<128x512xf32>, memref<128x512xf32, 1>
    %rs = frisk.reduce_tensor %2 {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x512xf32> -> tensor<512xf32>
    %redout = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %dst = frisk.layout_view %redout {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %rs, %dst : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}
// -----

// With register-sourced A, consumer savings cannot admit the other grid.
// CHECK: eligible = false, grid = array<i64: 1, 2>
// CHECK-SAME: eligible = true
// CHECK-SAME: grid = array<i64: 2, 1>
// CHECK: func.func @rs_guard
// CHECK: frisk.mma
// CHECK-SAME: grid = array<i64: 2, 1>
module attributes {frisk.target = "sm_90a"} {
  func.func @rs_guard() {
    %cst = arith.constant dense<1.000000e+00> : tensor<128x16xf16>
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %0 = frisk.layout_view %alloc : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x512xf32>
    %1 = frisk.mma %cst, %0, %cst_0 {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : tensor<128x16xf16>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
    %2 = frisk.reduce_tensor %1 {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x512xf32> -> tensor<512xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %2, %3 : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}