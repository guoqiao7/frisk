// RUN: frisk-opt %s --frisk-protected-store-layout="striped-shared-replay=true" -verify-each | FileCheck %s --check-prefix=ON
// RUN: frisk-opt %s --frisk-protected-store-layout="score-spill-risk=false striped-shared-replay=true" -verify-each | FileCheck %s --check-prefix=OFF
// Both candidates are local optima. Under pressure, the 2x1 candidate
// pays a higher routed-store cost than the 1x2 candidate.
// ON: grid = array<i64: 1, 2>
// ON-SAME: store_cost = 12368
// ON-SAME: grid = array<i64: 2, 1>
// ON-SAME: store_cost = 15488
// ON: frisk.mma
// ON-SAME: grid = array<i64: 1, 2>
// ON: frisk.convert_layout
// ON: frisk.tile_store
// ON-SAME: frisk.spill_pressure = array<i64: 256, 192, 16, 16>
// OFF: frisk.mma
// OFF-SAME: grid = array<i64: 2, 1>
// OFF: frisk.convert_layout
// OFF: frisk.tile_store
// OFF-SAME: frisk.spill_pressure = array<i64: 256, 320, 16, 4176>
module attributes {frisk.target = "sm_90a"} {
  func.func @selection(%a: memref<128x128xf16, 1>, %b: memref<128x512xf16, 1>) {
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
    return
  }
}