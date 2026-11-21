// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer" -verify-each -o %t
// RUN: %python %S/Inputs/check-normalization-price.py %t frisk-opt

module attributes {frisk.normalization_trial={},frisk.target="sm_90a"} {
"frisk.kernel"() <{sym_name="k",function_type=(memref<8x128xf32, 1>,memref<8x128xf32, 1>)->()}> ({
^bb0(%A:memref<8x128xf32, 1>,%O:memref<8x128xf32, 1>):
frisk.storage_contract %A {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[8,128],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*512+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<8x128xf32, 1>
frisk.storage_contract %O {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[8,128],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*512+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<8x128xf32, 1>
"frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
^bb0(%i:index):
%a = frisk.layout_view %A {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[8,128],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*512+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<8x128xf32, 1> -> memref<8x128xf32, 1>
%o = frisk.layout_view %O {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[8,128],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*512+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<8x128xf32, 1> -> memref<8x128xf32, 1>
%x = frisk.tile_load %a : memref<8x128xf32, 1> -> tensor<8x128xf32>
%empty = tensor.empty() : tensor<8x128xf32>
%sq = arith.mulf %x, %x : tensor<8x128xf32>
%z = frisk.reduce_tensor %sq {dim=1:i64,kind="sum",frisk.execution_threads=128:i64} : tensor<8x128xf32> -> tensor<8xf32>
%count = arith.constant dense<128.0> : tensor<8xf32>
%eps = arith.constant dense<1.0e-5> : tensor<8xf32>
%mean = arith.divf %z, %count : tensor<8xf32>
%off = arith.addf %mean, %eps : tensor<8xf32>
%inv = math.rsqrt %off : tensor<8xf32>
%scale = linalg.broadcast ins(%inv : tensor<8xf32>) outs(%empty : tensor<8x128xf32>) dimensions = [1]
%result = arith.mulf %x, %scale : tensor<8x128xf32>
frisk.tile_store %result, %o : tensor<8x128xf32>, memref<8x128xf32, 1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}
