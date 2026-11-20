// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-scratch-backend=gluon34 integer-oracle-limit=250000" -verify-each | FileCheck %s --check-prefix=LATE
// RUN: sed 's/load_words=1977:i64/load_words=1:i64/;s/store_words=1880:i64/store_words=1:i64/' %s | frisk-opt --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-scratch-backend=gluon34 integer-oracle-limit=250000" -verify-each | FileCheck %s --check-prefix=EARLY
// RUN: sed 's/axis=0:i64/axis=1:i64/' %s | not frisk-opt --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-scratch-backend=gluon34 integer-oracle-limit=250000" 2>&1 | FileCheck %s --check-prefix=AXIS
// RUN: sed 's/{layout=#C, cubin=/{layout=#R, cubin=/' %s | not frisk-opt --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-scratch-backend=gluon34 integer-oracle-limit=250000" 2>&1 | FileCheck %s --check-prefix=DUP
// RUN: not frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer" 2>&1 | FileCheck %s --check-prefix=BACKEND
// LATE: compiled_spill_units = 0 : i64
// LATE: = frisk.tile_load
// LATE-NEXT: %[[VAL:[0-9]+]] = frisk.reduce_tensor
// LATE-NEXT: {{.*}} = frisk.convert_layout %[[VAL]]
// EARLY: compiled_spill_units = 32 : i64
// EARLY: = frisk.tile_load
// EARLY-NEXT: %[[VAL:[0-9]+]] = frisk.convert_layout
// EARLY-NEXT: {{.*}} = frisk.reduce_tensor %[[VAL]]
// AXIS: reduction or axis does not match
// DUP: duplicate or unrecognized candidate layout
// BACKEND: compiled Sum spill evidence requires the Gluon integer scoring path
#output = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[2048],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i)->(i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#input = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[32,2048],outputs=["byte_offset","bit_offset"],output_extents=[262144,8],map=affine_map<(i,j)->(i*8192+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#R = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[9,5,2],outputs=["dim0","dim1"],output_bits=[5,11],matrix=dense<"0x0020004000808000000101000200000200040008001004000800100020004000"> : tensor<16x16xi1>>,topology=[512,32,4,1,1],replication=1>
#C = #frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[9,5,2],outputs=["dim0","dim1"],output_bits=[5,11],matrix=dense<"0x0100020000020004000800100020004000800400080010002000400080000001"> : tensor<16x16xi1>>,topology=[512,32,4,1,1],replication=1>
#sC = #frisk.distributed<map=#frisk.bit_linear<inputs=["register", "lane", "warp"],input_bits=[7, 5, 2],outputs=["dim1"],output_bits=[11],matrix=dense<"0x0004000200018001800040002000100008000400"> : tensor<11x14xi1>>,topology=[128,32,4,1,1],replication=8>
module attributes {frisk.compiled_sum_spill = {backend="gluon34", threads=128:i64, stages=1:i64, axis=0:i64, read_storage=#input, write_storage=#output, read=#R, write=#sC, arms=[{layout=#R, cubin="62a261cc2bafb5947f509c8b1459f82197dcd470aafffdea49bf80ae6c823fbf", load_words=0:i64, store_words=0:i64}, {layout=#C, cubin="0db34b794a25798637689fb9c84e0fdcabb1b54499616ed2700e94835cc5c802", load_words=1977:i64, store_words=1880:i64}]}, frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<32x2048xf32, 1>,memref<2048xf32, 1>)->()}> ({
^bb0(%A:memref<32x2048xf32, 1>,%O:memref<2048xf32, 1>):
frisk.storage_contract %A {layout=#input} : memref<32x2048xf32, 1>
frisk.storage_contract %O {layout=#output} : memref<2048xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#input} : memref<32x2048xf32, 1> -> memref<32x2048xf32, 1>
%o = frisk.layout_view %O {layout=#output} : memref<2048xf32, 1> -> memref<2048xf32, 1>
%x = frisk.tile_load %a {frisk.fixed_access_layout=#R} : memref<32x2048xf32, 1> -> tensor<32x2048xf32>
%result = frisk.reduce_tensor %x {dim=0:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<32x2048xf32> -> tensor<2048xf32>
frisk.tile_store %result, %o {frisk.fixed_access_layout=#sC} : tensor<2048xf32>, memref<2048xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
