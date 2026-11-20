// REQUIRES: frisk-z3
// RUN: split-file %s %t
// RUN: frisk-opt %t/ss.mlir --frisk-protected-store-layout="anchor-solver=integer integer-oracle-limit=250000" -verify-each -o %t/ss.out
// RUN: frisk-opt %t/rs.mlir --frisk-protected-store-layout="anchor-solver=integer integer-oracle-limit=250000" -verify-each -o %t/rs.out
// RUN: frisk-opt %t/chain.mlir --frisk-protected-store-layout="anchor-solver=integer integer-oracle-limit=250000" -verify-each -o %t/chain.out
// RUN: frisk-opt %t/fanout-small.mlir --frisk-protected-store-layout="anchor-solver=integer integer-oracle-limit=250000" -verify-each -o %t/fanout-small.out
// RUN: frisk-opt %t/write1.mlir --frisk-protected-store-layout="anchor-solver=integer integer-oracle-limit=250000" -verify-each -o %t/write1.out
// RUN: python3 %S/Inputs/check-multi-anchor-integer.py %t
// RUN: not frisk-opt %t/ss.mlir --frisk-protected-store-layout="anchor-solver=integer integer-oracle-limit=1" 2>&1 | FileCheck %s --check-prefix=LIMIT
// LIMIT: enumeration audit reached its node limit

//--- ss.mlir
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x512xf32>
    %2 = frisk.mma %0, %1, %cst {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<128x16xf16, 3>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
    %3 = frisk.reduce_tensor %2 {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x512xf32> -> tensor<512xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %4 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %3, %4 : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}
//--- rs.mlir
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
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
//--- chain.mlir
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x512xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x512xf16, 3> -> memref<16x512xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x512xf32>
    %2 = frisk.mma %0, %1, %cst {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<128x16xf16, 3>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
    %p = arith.constant dense<1.0> : tensor<128x16xf16>
    %r = frisk.mma %p, %1, %2 {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy Auto>} : tensor<128x16xf16>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
    %3 = frisk.reduce_tensor %r {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x512xf32> -> tensor<512xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %4 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %3, %4 : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}
//--- fanout-small.mlir
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x128xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x128xf16, 3> -> memref<16x128xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %2 = frisk.mma %0, %1, %cst {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy FullCol>} : memref<128x16xf16, 3>, memref<16x128xf16, 3>, tensor<128x128xf32> -> tensor<128x128xf32>
    %p = arith.constant dense<1.0> : tensor<128x16xf16>
    %r = frisk.mma %p, %1, %2 {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy Auto>} : tensor<128x16xf16>, memref<16x128xf16, 3>, tensor<128x128xf32> -> tensor<128x128xf32>
    %p2 = arith.constant dense<2.0> : tensor<128x16xf16>
    %r2 = frisk.mma %p2, %1, %2 {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy Auto>} : tensor<128x16xf16>, memref<16x128xf16, 3>, tensor<128x128xf32> -> tensor<128x128xf32>
    %sum = arith.addf %r, %r2 : tensor<128x128xf32>
    %3 = frisk.reduce_tensor %sum {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x128xf32> -> tensor<128xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128xf32, 1>
    %4 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [128], outputs = ["byte_offset", "bit_offset"], output_extents = [512, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128xf32, 1> -> memref<128xf32, 1>
    frisk.tile_store %3, %4 : tensor<128xf32>, memref<128xf32, 1>
    return
  }
}
//--- write1.mlir
module attributes {frisk.target = "sm_90a"} {
  func.func @selection() {
    %alloc = memref.alloc() {alignment = 1024 : i64} : memref<128x16xf16, 3>
    %0 = frisk.layout_view %alloc : memref<128x16xf16, 3> -> memref<128x16xf16, 3>
    %alloc_0 = memref.alloc() {alignment = 1024 : i64} : memref<16x128xf16, 3>
    %1 = frisk.layout_view %alloc_0 : memref<16x128xf16, 3> -> memref<16x128xf16, 3>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %2 = frisk.mma %0, %1, %cst {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy Auto>} : memref<128x16xf16, 3>, memref<16x128xf16, 3>, tensor<128x128xf32> -> tensor<128x128xf32>
    %p = arith.constant dense<1.0> : tensor<128x16xf16>
    %r = frisk.mma %p, %1, %2 {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 128 : i64, policy = #frisk<gemm_warp_policy Auto>} : tensor<128x16xf16>, memref<16x128xf16, 3>, tensor<128x128xf32> -> tensor<128x128xf32>
    %buf0 = memref.alloc() {alignment=128:i64} : memref<128x128xf32,1>
    %view0 = frisk.layout_view %buf0 {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[128,128],outputs=["byte_offset","bit_offset"],output_extents=[65536,8],map=affine_map<(i,j)->((i*512+j*4),0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<128x128xf32,1> -> memref<128x128xf32,1>
    frisk.tile_store %2, %view0 : tensor<128x128xf32>, memref<128x128xf32,1>
    %3 = frisk.reduce_tensor %r {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x128xf32> -> tensor<128xf32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128xf32, 1>
    %4 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [128], outputs = ["byte_offset", "bit_offset"], output_extents = [512, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128xf32, 1> -> memref<128xf32, 1>
    frisk.tile_store %3, %4 : tensor<128xf32>, memref<128xf32, 1>
    return
  }
}
