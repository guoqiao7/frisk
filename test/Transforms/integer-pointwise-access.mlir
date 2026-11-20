// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-oracle-limit=250000" -verify-each | FileCheck %s
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-oracle-limit=0 integer-global-access=native" -verify-each | FileCheck %s --check-prefix=VECTOR
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-oracle-limit=0 integer-global-access=scalar" -verify-each | FileCheck %s --check-prefix=SCALAR
// VECTOR: access_units = 816 : i64
// VECTOR-SAME: conversion_units = 136 : i64
// SCALAR: access_units = 960 : i64
// SCALAR-SAME: conversion_units = 136 : i64
// Row-major input and first output; the second output is column-major.
// All tensor types are open. Only storage maps and the CTA width are supplied.
// CHECK: anchors = []
// CHECK-SAME: distinct_conversions = 1
// CHECK-SAME: memory_domains =
// CHECK: %[[X:[0-9]+]] = frisk.tile_load
// CHECK: %[[V:[0-9]+]] = arith.addf %[[X]],
// CHECK-NEXT: frisk.tile_store %[[V]],
// CHECK-NEXT: %[[CV:[0-9]+]] = frisk.convert_layout %[[V]]
// CHECK-NEXT: frisk.tile_store %[[CV]],
#column = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(j*128+i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#row = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="access",function_type=(memref<32x64xf32, 1>,memref<32x64xf32, 1>,memref<32x64xf32, 1>)->()}> ({
^bb0(%A:memref<32x64xf32, 1>,%B:memref<32x64xf32, 1>,%C:memref<32x64xf32, 1>):
frisk.storage_contract %A {layout=#row} : memref<32x64xf32, 1>
frisk.storage_contract %B {layout=#row} : memref<32x64xf32, 1>
frisk.storage_contract %C {layout=#column} : memref<32x64xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#row} : memref<32x64xf32, 1> -> memref<32x64xf32, 1>
%b = frisk.layout_view %B {layout=#row} : memref<32x64xf32, 1> -> memref<32x64xf32, 1>
%c = frisk.layout_view %C {layout=#column} : memref<32x64xf32, 1> -> memref<32x64xf32, 1>
%x = frisk.tile_load %a : memref<32x64xf32, 1> -> tensor<32x64xf32>
%one = arith.constant dense<1.0> : tensor<32x64xf32>
%v = arith.addf %x, %one : tensor<32x64xf32>
frisk.tile_store %v, %b : tensor<32x64xf32>, memref<32x64xf32, 1>
frisk.tile_store %v, %c : tensor<32x64xf32>, memref<32x64xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
