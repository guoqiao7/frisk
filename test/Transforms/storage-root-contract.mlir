// RUN: frisk-opt %s -frisk-infer-layouts -canonicalize -frisk-infer-layouts | FileCheck %s
// Root evidence survives DCE of the original whole-root view.
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
  outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
  memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
// CHECK-LABEL: func.func @durable
// CHECK: frisk.storage_contract
// CHECK: frisk.layout_view
// CHECK: memref.store
func.func @durable(%r:memref<4xi32,3>, %x:i32) {
  frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
  %zero = arith.constant 0 : index
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  memref.store %x, %v[%zero] : memref<2xi32,strided<[1],offset:1>,3>
  return
}
