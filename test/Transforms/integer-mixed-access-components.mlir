// REQUIRES: frisk-z3
// RUN: split-file %s %t
// RUN: cat %t/header %t/transpose %t/sum %t/footer > %t/tr-first.mlir
// RUN: cat %t/header %t/sum %t/transpose %t/footer > %t/sum-first.mlir
// RUN: frisk-opt %t/tr-first.mlir --frisk-protected-store-layout="anchor-solver=integer integer-global-access=native" -verify-each | FileCheck %s
// RUN: frisk-opt %t/tr-first.mlir --frisk-protected-store-layout="anchor-solver=integer integer-global-access=pointwise" -verify-each | FileCheck %s
// RUN: frisk-opt %t/sum-first.mlir --frisk-protected-store-layout="anchor-solver=integer integer-global-access=native" -verify-each | FileCheck %s
// RUN: sed 's/frisk.layout_view %rin/frisk.layout_view %in/' %t/tr-first.mlir > %t/one-root.mlir
// RUN: frisk-opt %t/one-root.mlir --frisk-protected-store-layout="anchor-solver=integer integer-global-access=native" -verify-each | FileCheck %s
//
// Both components share the CTA, but use independent SSA values. Together
// they must retain the separate optima, even when loading the same root: 272 + 72 for the transpose, and
// 163 + 36 for dim0 Sum. Only the column-major write needs redistribution.
// CHECK: access_units = 435 : i64
// CHECK-SAME: conversion_units = 72 : i64
// CHECK-SAME: distinct_conversions = 1 : i64
// CHECK-SAME: reduction_units = 36 : i64
// CHECK: %[[CV:[^ ]+]] = frisk.convert_layout
// CHECK-NEXT: frisk.tile_store %[[CV]], {{.*}} : tensor<64x16xf32

//--- header
#vec = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[16],outputs=["byte_offset","bit_offset"],output_extents=[64,8],map=affine_map<(i)->(i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#row = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,16],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*64+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#col = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,16],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(j*256+i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
  "frisk.kernel"() <{sym_name="mixed",function_type=(memref<64x16xf32, 1>,memref<64x16xf32, 1>,memref<64x16xf32, 1>,memref<16xf32, 1>)->()}> ({
  ^bb0(%in:memref<64x16xf32, 1>,%out:memref<64x16xf32, 1>,%rin:memref<64x16xf32, 1>,%rout:memref<16xf32, 1>):
    frisk.storage_contract %in {layout=#row} : memref<64x16xf32, 1>
    frisk.storage_contract %out {layout=#col} : memref<64x16xf32, 1>
    frisk.storage_contract %rin {layout=#row} : memref<64x16xf32, 1>
    frisk.storage_contract %rout {layout=#vec} : memref<16xf32, 1>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
//--- transpose
      %a = frisk.layout_view %in {layout=#row} : memref<64x16xf32, 1> -> memref<64x16xf32, 1>
      %b = frisk.layout_view %out {layout=#col} : memref<64x16xf32, 1> -> memref<64x16xf32, 1>
      %x = frisk.tile_load %a : memref<64x16xf32, 1> -> tensor<64x16xf32>
      frisk.tile_store %x, %b : tensor<64x16xf32>, memref<64x16xf32, 1>
//--- sum
      %ri = frisk.layout_view %rin {layout=#row} : memref<64x16xf32, 1> -> memref<64x16xf32, 1>
      %ro = frisk.layout_view %rout {layout=#vec} : memref<16xf32, 1> -> memref<16xf32, 1>
      %rx = frisk.tile_load %ri : memref<64x16xf32, 1> -> tensor<64x16xf32>
      %rr = frisk.reduce_tensor %rx {dim = 0 : i64, frisk.execution_threads = 128 : i64, kind = "sum"} : tensor<64x16xf32> -> tensor<16xf32>
      frisk.tile_store %rr, %ro : tensor<16xf32>, memref<16xf32, 1>
//--- footer
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
