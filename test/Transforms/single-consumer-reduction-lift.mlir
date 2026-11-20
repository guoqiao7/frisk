// REQUIRES: frisk-z3
// RUN: frisk-opt %s -split-input-file --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0" -verify-each | FileCheck %s
// Only the load and rank-one store are pinned. The two-dimensional working
// layout must be proposed by the solver. The second store has an XOR row.
// CHECK: %[[LOAD:[a-zA-Z0-9_]+]] = frisk.tile_load
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = frisk.convert_layout %[[LOAD]]
// CHECK-NEXT: %[[SUM:[a-zA-Z0-9_]+]] = frisk.reduce_tensor %[[INPUT]]
// CHECK-SAME: scope = "warp"
// CHECK-NEXT: frisk.tile_store %[[SUM]]
#R = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[6,5,2],outputs=["dim0","dim1"],output_bits=[5,8],matrix=dense<"0x00040001400800020280000002800020000802800000"> : tensor<13x13xi1>>,topology=[64,32,4,1,1],replication=1>
#sC = #frisk.distributed<map=#frisk.bit_linear<inputs=["register", "lane", "warp"],input_bits=[4, 5, 2],outputs=["dim1"],output_bits=[8],matrix=dense<"0x8000088000180001100001"> : tensor<8x11xi1>>,topology=[16,32,4,1,1],replication=8>
#input = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,256],outputs=["byte_offset","bit_offset"],output_extents=[32768,8],map=affine_map<(i,j)->(i*1024+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#output = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[256],outputs=["byte_offset","bit_offset"],output_extents=[1024,8],map=affine_map<(i)->(i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<32x256xf32, 1>,memref<256xf32, 1>)->()}> ({
^bb0(%A:memref<32x256xf32, 1>,%O:memref<256xf32, 1>):
frisk.storage_contract %A {layout=#input} : memref<32x256xf32, 1>
frisk.storage_contract %O {layout=#output} : memref<256xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#input} : memref<32x256xf32, 1> -> memref<32x256xf32, 1>
%o = frisk.layout_view %O {layout=#output} : memref<256xf32, 1> -> memref<256xf32, 1>
%x = frisk.tile_load %a {frisk.fixed_access_layout=#R} : memref<32x256xf32, 1> -> tensor<32x256xf32>
%result = frisk.reduce_tensor %x {dim=0:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<32x256xf32> -> tensor<256xf32>
frisk.tile_store %result, %o {frisk.fixed_access_layout=#sC} : tensor<256xf32>, memref<256xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}

// -----

// CHECK: %[[LOAD:[a-zA-Z0-9_]+]] = frisk.tile_load
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = frisk.convert_layout %[[LOAD]]
// CHECK-NEXT: %[[SUM:[a-zA-Z0-9_]+]] = frisk.reduce_tensor %[[INPUT]]
// CHECK-SAME: scope = "warp"
// CHECK-NEXT: frisk.tile_store %[[SUM]]
#R = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[6,5,2],outputs=["dim0","dim1"],output_bits=[5,8],matrix=dense<"0x00040001400800020280000002800020000802800000"> : tensor<13x13xi1>>,topology=[64,32,4,1,1],replication=1>
#sC = #frisk.distributed<map=#frisk.bit_linear<inputs=["register", "lane", "warp"],input_bits=[4, 5, 2],outputs=["dim1"],output_bits=[8],matrix=dense<"0x8000088000180001108001"> : tensor<8x11xi1>>,topology=[16,32,4,1,1],replication=8>
#input = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,256],outputs=["byte_offset","bit_offset"],output_extents=[32768,8],map=affine_map<(i,j)->(i*1024+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#output = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[256],outputs=["byte_offset","bit_offset"],output_extents=[1024,8],map=affine_map<(i)->(i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<32x256xf32, 1>,memref<256xf32, 1>)->()}> ({
^bb0(%A:memref<32x256xf32, 1>,%O:memref<256xf32, 1>):
frisk.storage_contract %A {layout=#input} : memref<32x256xf32, 1>
frisk.storage_contract %O {layout=#output} : memref<256xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#input} : memref<32x256xf32, 1> -> memref<32x256xf32, 1>
%o = frisk.layout_view %O {layout=#output} : memref<256xf32, 1> -> memref<256xf32, 1>
%x = frisk.tile_load %a {frisk.fixed_access_layout=#R} : memref<32x256xf32, 1> -> tensor<32x256xf32>
%result = frisk.reduce_tensor %x {dim=0:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<32x256xf32> -> tensor<256xf32>
frisk.tile_store %result, %o {frisk.fixed_access_layout=#sC} : tensor<256xf32>, memref<256xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}

// -----

// The declared result type can also fix the writeback.
// CHECK: %[[LOAD:[a-zA-Z0-9_]+]] = frisk.tile_load
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = frisk.convert_layout %[[LOAD]]
// CHECK-NEXT: %[[SUM:[a-zA-Z0-9_]+]] = frisk.reduce_tensor %[[INPUT]]
// CHECK-SAME: scope = "warp"
// CHECK-NEXT: frisk.tile_store %[[SUM]]
#R = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[6,5,2],outputs=["dim0","dim1"],output_bits=[5,8],matrix=dense<"0x00040001400800020280000002800020000802800000"> : tensor<13x13xi1>>,topology=[64,32,4,1,1],replication=1>
#sC = #frisk.distributed<map=#frisk.bit_linear<inputs=["register", "lane", "warp"],input_bits=[4, 5, 2],outputs=["dim1"],output_bits=[8],matrix=dense<"0x8000088000180001100001"> : tensor<8x11xi1>>,topology=[16,32,4,1,1],replication=8>
#input = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,256],outputs=["byte_offset","bit_offset"],output_extents=[32768,8],map=affine_map<(i,j)->(i*1024+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#output = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[256],outputs=["byte_offset","bit_offset"],output_extents=[1024,8],map=affine_map<(i)->(i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<32x256xf32, 1>,memref<256xf32, 1>)->()}> ({
^bb0(%A:memref<32x256xf32, 1>,%O:memref<256xf32, 1>):
frisk.storage_contract %A {layout=#input} : memref<32x256xf32, 1>
frisk.storage_contract %O {layout=#output} : memref<256xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#input} : memref<32x256xf32, 1> -> memref<32x256xf32, 1>
%o = frisk.layout_view %O {layout=#output} : memref<256xf32, 1> -> memref<256xf32, 1>
%x = frisk.tile_load %a {frisk.fixed_access_layout=#R} : memref<32x256xf32, 1> -> tensor<32x256xf32>
%result = frisk.reduce_tensor %x {dim=0:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<32x256xf32> -> tensor<256xf32, #sC>
frisk.tile_store %result, %o : tensor<256xf32, #sC>, memref<256xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
