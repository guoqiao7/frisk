// REQUIRES: frisk-z3
// RUN: split-file %s %t
// RUN: frisk-opt %t/fan.mlir --frisk-protected-store-layout="anchor-solver=integer integer-native-stripes=false" -verify-each | FileCheck %s --check-prefix=LOCAL
// RUN: frisk-opt %t/fan21.mlir --frisk-protected-store-layout="anchor-solver=integer integer-native-stripes=false" -verify-each | FileCheck %s --check-prefix=REMOTE
// RUN: frisk-opt %t/store21.mlir --frisk-protected-store-layout="anchor-solver=integer integer-native-stripes=false" -verify-each | FileCheck %s --check-prefix=DIRECT
// RUN: frisk-opt %t/store21.mlir --frisk-multi-anchor-layout-pipeline -verify-each | FileCheck %s --check-prefix=STRIPE
// LOCAL: reduction_units = 0 : i64, selected = array<i64: 0>
// LOCAL: frisk.mma
// LOCAL-SAME: grid = array<i64: 1, 2>
// REMOTE: reduction_units = 48 : i64
// REMOTE: frisk.mma
// REMOTE-SAME: grid = array<i64: 2, 1>
// DIRECT: distinct_conversions = 0 : i64
// DIRECT-NOT: frisk.convert_layout
// DIRECT: frisk.tile_store
// STRIPE: conversion_units = 4160 : i64, distinct_conversions = 1 : i64
// STRIPE: frisk.mma
// STRIPE-SAME: grid = array<i64: 2, 1>
// STRIPE: %[[DELIVERED:[a-zA-Z0-9_]+]] = frisk.convert_layout {{.*}} {frisk.native_shared_stripes}
// STRIPE: frisk.tile_store %[[DELIVERED]],

//--- fan.mlir
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
    %rs = frisk.reduce_tensor %2 {dim = 0 : i64, frisk.execution_threads = 256 : i64, kind = "sum"} : tensor<128x512xf32> -> tensor<512xf32>
    %redout = memref.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
    %dst = frisk.layout_view %redout {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0"], input_extents = [512], outputs = ["byte_offset", "bit_offset"], output_extents = [2048, 8], map = affine_map<(d0) -> (d0 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<512xf32, 1> -> memref<512xf32, 1>
    frisk.tile_store %rs, %dst : tensor<512xf32>, memref<512xf32, 1>
    return
  }
}
//--- fan21.mlir
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
    %next = frisk.mma %0, %1, %acc {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy FullRow>} : memref<128x16xf16, 3>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
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
//--- store21.mlir
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
    %next = frisk.mma %0, %1, %acc {frisk.execution_threads = 256 : i64, k = 16 : i64, m = 128 : i64, n = 512 : i64, policy = #frisk<gemm_warp_policy FullRow>} : memref<128x16xf16, 3>, memref<16x512xf16, 3>, tensor<128x512xf32> -> tensor<128x512xf32>
      scf.yield %next : tensor<128x512xf32>
    }
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<128x512xf32, 1>
    %3 = frisk.layout_view %alloc_1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [128, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [262144, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128x512xf32, 1> -> memref<128x512xf32, 1>
    frisk.tile_store %2, %3 : tensor<128x512xf32>, memref<128x512xf32, 1>
    return
  }
}
