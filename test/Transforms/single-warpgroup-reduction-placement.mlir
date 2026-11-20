// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0" -verify-each | FileCheck %s
// Fixed I/O, unencoded arithmetic. The load fans out into the max and sub.
// A single shared conversion must serve both before the first reduction.
// CHECK: reduction_units = 96 : i64
// CHECK: %[[INPUT:[a-zA-Z0-9_]+]] = frisk.tile_load
// CHECK: %[[EARLY:[a-zA-Z0-9_]+]] = frisk.convert_layout %[[INPUT]]
// CHECK: frisk.reduce_tensor %[[EARLY]]
// CHECK-SAME: scope = "warp"
// CHECK: arith.subf %[[EARLY]]
// CHECK: frisk.reduce_tensor
// CHECK-SAME: scope = "warp"
// CHECK-NOT: frisk.convert_layout
// CHECK: frisk.tile_store
#R = #frisk.distributed<map = #frisk.bit_linear<inputs = ["register", "lane", "warp"], input_bits = [5, 5, 2], outputs = ["dim0", "dim1"], output_bits = [6, 6], matrix = dense<"0x000240004800080001012000200004800010"> : tensor<12x12xi1>>, topology = [32, 32, 4, 1, 1], replication = 1>
#C = #frisk.distributed<map = #frisk.bit_linear<inputs = ["register", "lane", "warp"], input_bits = [5, 5, 2], outputs = ["dim0", "dim1"], output_bits = [6, 6], matrix = dense<"0x012000200004804000000120000480080001"> : tensor<12x12xi1>>, topology = [32, 32, 4, 1, 1], replication = 1>
#row = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#col = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(j*256+i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<64x64xf32, 1>,memref<64x64xf32, 1>)->()}> ({
^bb0(%A:memref<64x64xf32, 1>,%O:memref<64x64xf32, 1>):
frisk.storage_contract %A {layout=#row} : memref<64x64xf32, 1>
frisk.storage_contract %O {layout=#col} : memref<64x64xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#row} : memref<64x64xf32, 1> -> memref<64x64xf32, 1>
%o = frisk.layout_view %O {layout=#col} : memref<64x64xf32, 1> -> memref<64x64xf32, 1>
%x = frisk.tile_load %a {frisk.fixed_access_layout=#R} : memref<64x64xf32, 1> -> tensor<64x64xf32>
%empty = tensor.empty() : tensor<64x64xf32>
%mx = frisk.reduce_tensor %x {dim=0:i64,kind="max",frisk.execution_threads=128:i64} : tensor<64x64xf32> -> tensor<64xf32>
%bmax = linalg.broadcast ins(%mx : tensor<64xf32>) outs(%empty : tensor<64x64xf32>) dimensions = [0]
%diff = arith.subf %x, %bmax : tensor<64x64xf32>
%e = math.exp %diff : tensor<64x64xf32>
%z = frisk.reduce_tensor %e {dim=0:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<64x64xf32> -> tensor<64xf32>
%den = linalg.broadcast ins(%z : tensor<64xf32>) outs(%empty : tensor<64x64xf32>) dimensions = [0]
%result = arith.divf %e, %den : tensor<64x64xf32>
frisk.tile_store %result, %o {frisk.fixed_access_layout=#C} : tensor<64x64xf32>, memref<64x64xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
