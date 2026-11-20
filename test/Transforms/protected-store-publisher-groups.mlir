// RUN: frisk-opt %s --frisk-protected-store-layout="striped-shared-replay=true" -verify-each -o %t
// RUN: FileCheck %s < %t
// Below the accumulator register limit, the working set alone must not win
// over a route requiring more warpgroup/stripe publications.
// CHECK: eligible = false, grid = array<i64: 1, 2>
// CHECK-SAME: eligible = true
// CHECK-SAME: writer_exposed_peak_slots = 0
// CHECK-SAME: writer_peak_slots = 192
// CHECK-SAME: writer_publisher_group_stripes = 2
// CHECK-SAME: eligible = true
// CHECK-SAME: row_partition = "interleaved64"
// CHECK-SAME: writer_exposed_peak_slots = 0
// CHECK-SAME: writer_peak_slots = 128
// CHECK-SAME: writer_publisher_group_stripes = 4
// CHECK: frisk.mma
// CHECK-SAME: grid = array<i64: 2, 1>
// CHECK-NOT: row_partition
// CHECK: frisk.convert_layout
// CHECK: frisk.tile_store
module attributes {frisk.target = "sm_90a"} {
  func.func @selection(%a: memref<256x272xf16, 1>, %b: memref<272x128xf16, 1>) {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<256x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<256x16xf16, 3> -> memref<256x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x128xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x128xf16, 3> -> memref<16x128xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32>
    %lo = arith.constant 0 : index
    %hi = arith.constant 272 : index
    %step = arith.constant 16 : index
    %2 = scf.for %iv = %lo to %hi step %step iter_args(%acc = %cst) -> (tensor<256x128xf32>) {
      %at = memref.subview %a[0,%iv][256,16][1,1] : memref<256x272xf16, 1> to memref<256x16xf16, strided<[272,1], offset: ?>, 1>
      %bt = memref.subview %b[%iv,0][16,128][1,1] : memref<272x128xf16, 1> to memref<16x128xf16, strided<[128,1], offset: ?>, 1>
      memref.copy %at, %alloc : memref<256x16xf16, strided<[272,1], offset: ?>, 1> to memref<256x16xf16, 3>
      memref.copy %bt, %alloc_0 : memref<16x128xf16, strided<[128,1], offset: ?>, 1> to memref<16x128xf16, 3>
    %next = frisk.mma %0, %1, %acc {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 256 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<256x16xf16, 3>, memref<16x128xf16, 3>, tensor<256x128xf32> -> tensor<256x128xf32>
      scf.yield %next : tensor<256x128xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<256x128xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 128], outputs = ["byte_offset", "bit_offset"], output_extents = [131072, 8], map = affine_map<(d0, d1) -> (d0 * 512 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x128xf32, 1> -> memref<256x128xf32, 1>
    frisk.tile_store %2, %3 : tensor<256x128xf32>, memref<256x128xf32, 1>
    return
  }
}