// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-native-stripes=true" -verify-each | FileCheck %s --check-prefix=SHORT
// RUN: sed "s/constant 80 : index/constant 272 : index/" %s | frisk-opt --frisk-protected-store-layout="anchor-solver=integer integer-native-stripes=true" -verify-each | FileCheck %s --check-prefix=LONG
// RUN: sed -e "s/constant 80 : index/constant 0 : index/" -e "s/gemm_warp_policy Auto/gemm_warp_policy FullCol/" %s | frisk-opt --frisk-protected-store-layout="anchor-solver=integer integer-native-stripes=true" -verify-each | FileCheck %s --check-prefix=EMPTY
// SHORT: main_loop_units = 20 : i64
// SHORT-SAME: reduction_units = 0 : i64
// LONG: main_loop_units = 0 : i64
// LONG-SAME: reduction_units = 48 : i64
// EMPTY: main_loop_units = 0 : i64
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x512xf32>
    %lo = arith.constant 0 : index
    %hi = arith.constant 80 : index
    %step = arith.constant 16 : index
    %2 = scf.for %iv = %lo to %hi step %step iter_args(%acc = %cst) -> (tensor<128x512xf32>) {
    %next = frisk.mma %0, %1, %acc {frisk.serial_column_width = 256 : i64, frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<128x16xf16, 3>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
      scf.yield %next : tensor<128x512xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128x512xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [128, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [262144, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128x512xf32, 1> -> memref<128x512xf32, 1>
    frisk.tile_store %2, %3 : tensor<128x512xf32>, memref<128x512xf32, 1>
    %rs = frisk.reduce_tensor %2 {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "max"} : tensor<128x512xf32> -> tensor<512xf32>
    %redout = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %dst = frisk.layout_view %redout {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %rs, %dst : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}
