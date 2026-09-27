// RUN: frisk-opt %s -frisk-layout-pipeline -frisk-layout-pipeline | FileCheck %s
// Already-normalized storage programs use the same explicit pipeline entry.
// CHECK-LABEL: func.func @storage
// CHECK: frisk.storage_contract
// CHECK: memref.store
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
func.func @storage(%r:memref<4xi32,3>, %x:i32) {
  %zero = arith.constant 0:index
  %rv = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  memref.store %x, %v[%zero] : memref<2xi32,strided<[1],offset:1>,3>
  return
}
