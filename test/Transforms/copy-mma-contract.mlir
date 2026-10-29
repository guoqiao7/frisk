// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-layout-pipeline -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// The ABI is fixed, not a proposal that the Copy/MMA solver may rewrite.
// CHECK: frisk.kernel @copy_mma
// CHECK: frisk.storage_contract
// CHECK-SAME: alignment = 128
// CHECK: frisk.copy
// CHECK-SAME: frisk.copy_contract = {
// CHECK: frisk.copy
// CHECK-SAME: frisk.copy_contract = {
// CHECK: frisk.mma
// CHECK-SAME: form = "ss"
// CHECK-SAME: trans_b = true
// CHECK: frisk.tile_store
#input = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,16],outputs=["byte_offset","bit_offset"],output_extents=[2048,8],map=affine_map<(i,j)->(2*(16*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#output = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(4*(64*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
  "frisk.kernel"() <{sym_name="copy_mma",function_type=(memref<64x16xf16,1>,memref<64x16xf16,1>,memref<64x64xf32,1>)->()}> ({
  ^bb0(%a:memref<64x16xf16,1>,%b:memref<64x16xf16,1>,%o:memref<64x64xf32,1>):
    frisk.storage_contract %a {layout=#input} : memref<64x16xf16,1>
    frisk.storage_contract %b {layout=#input} : memref<64x16xf16,1>
    frisk.storage_contract %o {layout=#output} : memref<64x64xf32,1>
    %as = memref.alloc() {alignment=1024:i64} : memref<64x16xf16,3>
    %bs = memref.alloc() {alignment=1024:i64} : memref<64x16xf16,3>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %ag = frisk.layout_view %a : memref<64x16xf16,1> -> memref<64x16xf16,1>
      %bg = frisk.layout_view %b : memref<64x16xf16,1> -> memref<64x16xf16,1>
      %av = frisk.layout_view %as : memref<64x16xf16,3> -> memref<64x16xf16,3>
      %bv = frisk.layout_view %bs : memref<64x16xf16,3> -> memref<64x16xf16,3>
      frisk.copy %ag[], %av[] {srcExtents=array<i64:64,16>,dstExtents=array<i64:64,16>}
        : memref<64x16xf16,1>, memref<64x16xf16,3>
      frisk.copy %bg[], %bv[] {srcExtents=array<i64:64,16>,dstExtents=array<i64:64,16>}
        : memref<64x16xf16,1>, memref<64x16xf16,3>
      %zero = arith.constant dense<0.0> : tensor<64x64xf32>
      %r = "frisk.mma"(%av,%bv,%zero) {m=64:i64,n=64:i64,k=16:i64,trans_b=true}
        : (memref<64x16xf16,3>,memref<64x16xf16,3>,tensor<64x64xf32>)->tensor<64x64xf32>
      %ov = frisk.layout_view %o : memref<64x64xf32,1> -> memref<64x64xf32,1>
      frisk.tile_store %r, %ov : tensor<64x64xf32>, memref<64x64xf32,1>
      frisk.end
    }) : ()->()
    frisk.end
  }) : ()->()
}
