// RUN: frisk-opt %s -frisk-normalize-layout-ir -canonicalize -frisk-infer-layouts | FileCheck %s
// RUN: frisk-opt %s -frisk-normalize-layout-ir -frisk-normalize-layout-ir | FileCheck %s --check-prefix=ONCE
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
  outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
  memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
// CHECK-LABEL: func.func @preserve
// CHECK: frisk.storage_contract
// CHECK: memref.store
// ONCE-LABEL: func.func @preserve
// ONCE-COUNT-1: frisk.storage_contract
// ONCE: return
func.func @preserve(%r:memref<4xi32,3>, %x:i32) {
  %zero = arith.constant 0 : index
  %rv = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  memref.store %x, %v[%zero] : memref<2xi32,strided<[1],offset:1>,3>
  return
}
