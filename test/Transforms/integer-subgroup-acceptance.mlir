// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-subgroup-candidates=true" -verify-each | FileCheck %s
// CHECK: subgroup_challenger_candidates = {{[1-9][0-9]*}} : i64
// CHECK-SAME: subgroup_challenger_certified = true
// CHECK-SAME: subgroup_promoted = false
// CHECK-SAME: subgroup_selected = 0 : i64
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="access",function_type=(memref<32x32xf32, 1>,memref<32x32xf32, 1>,memref<32x32xf32, 1>,memref<32x32xf32, 1>)->()}> ({
^bb0(%A:memref<32x32xf32, 1>,%B:memref<32x32xf32, 1>,%C:memref<32x32xf32, 1>,%D:memref<32x32xf32, 1>):
frisk.storage_contract %A {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1>
frisk.storage_contract %B {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1>
frisk.storage_contract %C {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1>
frisk.storage_contract %D {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1> -> memref<32x32xf32, 1>
%b = frisk.layout_view %B {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1> -> memref<32x32xf32, 1>
%c = frisk.layout_view %C {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1> -> memref<32x32xf32, 1>
%d = frisk.layout_view %D {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,32],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*128+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<32x32xf32, 1> -> memref<32x32xf32, 1>
%x = frisk.tile_load %a : memref<32x32xf32, 1> -> tensor<32x32xf32>
frisk.tile_store %x, %b : tensor<32x32xf32>, memref<32x32xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
