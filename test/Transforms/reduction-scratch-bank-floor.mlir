// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-scratch-backend=gluon34 integer-scratch-bank-floor=false" -verify-each | FileCheck %s --check-prefix=ISSUE
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-scratch-backend=gluon34" -verify-each | FileCheck %s --check-prefix=WAVE
// Only the scratch publisher's price changes. Fixed I/O and both parents are identical.
// ISSUE: %[[IN:[0-9]+]] = frisk.tile_load
// ISSUE-NEXT: %[[EARLY:[0-9]+]] = frisk.convert_layout %[[IN]]
// ISSUE-NEXT: %[[SUM:[0-9]+]] = frisk.reduce_tensor %[[EARLY]]
// ISSUE-NEXT: frisk.tile_store %[[SUM]]
// WAVE: %[[IN:[0-9]+]] = frisk.tile_load
// WAVE-NEXT: %[[SUM:[0-9]+]] = frisk.reduce_tensor %[[IN]]
// WAVE-NEXT: %[[LATE:[0-9]+]] = frisk.convert_layout %[[SUM]]
// WAVE-NEXT: frisk.tile_store %[[LATE]]
#C = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[8,5,2],outputs=["dim0","dim1"],output_bits=[9,6],matrix=dense<"0x0100010040004000402000200020002000000400040004000402000200"> : tensor<15x15xi1>>,topology=[256,32,4,1,1],replication=1>
#sR = #frisk.distributed<map=#frisk.bit_linear<inputs=["register", "lane", "warp"],input_bits=[6, 5, 2],outputs=["dim0"],output_bits=[9],matrix=dense<"0x00040001c000200008000280002000"> : tensor<9x13xi1>>,topology=[64,32,4,1,1],replication=16>
#input = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[512,64],outputs=["byte_offset","bit_offset"],output_extents=[131072,8],map=affine_map<(i,j)->(j*2048+i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#output = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[512],outputs=["byte_offset","bit_offset"],output_extents=[2048,8],map=affine_map<(i)->(i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<512x64xf32, 1>,memref<512xf32, 1>)->()}> ({
^bb0(%A:memref<512x64xf32, 1>,%O:memref<512xf32, 1>):
frisk.storage_contract %A {layout=#input} : memref<512x64xf32, 1>
frisk.storage_contract %O {layout=#output} : memref<512xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#input} : memref<512x64xf32, 1> -> memref<512x64xf32, 1>
%o = frisk.layout_view %O {layout=#output} : memref<512xf32, 1> -> memref<512xf32, 1>
%x = frisk.tile_load %a {frisk.fixed_access_layout=#C} : memref<512x64xf32, 1> -> tensor<512x64xf32>
%result = frisk.reduce_tensor %x {dim=1:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<512x64xf32> -> tensor<512xf32>
frisk.tile_store %result, %o {frisk.fixed_access_layout=#sR} : tensor<512xf32>, memref<512xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
