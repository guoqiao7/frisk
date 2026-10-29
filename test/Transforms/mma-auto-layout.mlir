// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-layout-pipeline -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// Live Auto MMA and both reduction axes, with fixed external row-major ABI.
// CHECK: frisk.kernel @mma_auto
// CHECK: frisk.storage_contract
// CHECK: frisk.mma
// CHECK-SAME: grid = array<i64: 2, 1>
// CHECK-SAME: policy = #frisk<gemm_warp_policy Auto>
// CHECK: frisk.tile_store
// CHECK: frisk.reduce_tensor
// CHECK-SAME: dim = 0
// CHECK: frisk.tile_store
// CHECK: frisk.reduce_tensor
// CHECK-SAME: dim = 1
// CHECK: frisk.tile_store
#matrix = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[128,128],outputs=["byte_offset","bit_offset"],output_extents=[65536,8],map=affine_map<(i,j)->(4*(128*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#vector = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[128],outputs=["byte_offset","bit_offset"],output_extents=[512,8],map=affine_map<(i)->(4*i,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
  "frisk.kernel"() <{sym_name="mma_auto",function_type=(memref<128x128xf32,1>,memref<128xf32,1>,memref<128xf32,1>)->()}> ({
  ^bb0(%o:memref<128x128xf32,1>,%o0:memref<128xf32,1>,%o1:memref<128xf32,1>):
    frisk.storage_contract %o {layout=#matrix} : memref<128x128xf32,1>
    frisk.storage_contract %o0 {layout=#vector} : memref<128xf32,1>
    frisk.storage_contract %o1 {layout=#vector} : memref<128xf32,1>
    %ar=memref.alloc() {alignment=1024:i64} : memref<128x16xf16,3>
    %br=memref.alloc() {alignment=1024:i64} : memref<16x128xf16,3>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=256:i64}> ({
    ^bb0(%i:index):
      %a=frisk.layout_view %ar : memref<128x16xf16,3> -> memref<128x16xf16,3>
      %b=frisk.layout_view %br : memref<16x128xf16,3> -> memref<16x128xf16,3>
      %out=frisk.layout_view %o : memref<128x128xf32,1> -> memref<128x128xf32,1>
      %out0=frisk.layout_view %o0 : memref<128xf32,1> -> memref<128xf32,1>
      %out1=frisk.layout_view %o1 : memref<128xf32,1> -> memref<128xf32,1>
      %c=arith.constant dense<0.0> : tensor<128x128xf32>
      %r="frisk.mma"(%a,%b,%c) {m=128:i64,n=128:i64,k=16:i64,policy=#frisk<gemm_warp_policy Auto>} : (memref<128x16xf16,3>,memref<16x128xf16,3>,tensor<128x128xf32>) -> tensor<128x128xf32>
      frisk.tile_store %r, %out : tensor<128x128xf32>, memref<128x128xf32,1>
      %r0="frisk.reduce_tensor"(%r) {kind="sum",dim=0:i64} : (tensor<128x128xf32>) -> tensor<128xf32>
      frisk.tile_store %r0, %out0 : tensor<128xf32>, memref<128xf32,1>
      %r1="frisk.reduce_tensor"(%r) {kind="sum",dim=1:i64} : (tensor<128x128xf32>) -> tensor<128xf32>
      frisk.tile_store %r1, %out1 : tensor<128xf32>, memref<128xf32,1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
