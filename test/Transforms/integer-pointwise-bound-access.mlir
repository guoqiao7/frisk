// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-oracle-limit=250000" -verify-each | FileCheck %s
// CHECK: distinct_conversions = 0
// CHECK: = frisk.tile_load
// CHECK: = arith.addf
// CHECK-NEXT: frisk.tile_store
// CHECK-NEXT: frisk.tile_store
#column = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(j*128+i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#row = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#L0 = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[4,5,2],outputs=["dim0","dim1"],output_bits=[5,6],matrix=dense<"0x01100004400004400004400024000200"> : tensor<11x11xi1>>,topology=[16,32,4,1,1],replication=1>
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
%x = frisk.tile_load %a {frisk.fixed_access_layout=#L0} : memref<32x64xf32, 1> -> tensor<32x64xf32, #L0>
%one = arith.constant dense<1.0> : tensor<32x64xf32, #L0>
%v = arith.addf %x, %one : tensor<32x64xf32, #L0>
frisk.tile_store %v, %b {frisk.fixed_access_layout=#L0} : tensor<32x64xf32, #L0>, memref<32x64xf32, 1>
frisk.tile_store %v, %c {frisk.fixed_access_layout=#L0} : tensor<32x64xf32, #L0>, memref<32x64xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
