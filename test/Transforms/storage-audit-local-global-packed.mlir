// RUN: frisk-opt %s -frisk-normalize-layout-ir -frisk-infer-layouts -frisk-infer-layouts | FileCheck %s
// RUN: frisk-opt %s -frisk-normalize-layout-ir -frisk-infer-layouts -frisk-normalize-layout-ir -frisk-infer-layouts | FileCheck %s
#global = #frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[4],outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>
#packed = #frisk.storage<map=#frisk.bit_linear<inputs=["d"],input_bits=[2],outputs=["byte_offset","bit_offset"],output_bits=[0,3],matrix=dense<[[1,0],[0,1],[0,0]]>:tensor<3x2xi1>>,memory_space=#frisk<memory_space Shared>,alignment=1,vector_granularity=1>
module {
  memref.global "private" @g : memref<4xi32,1> = dense<0>
  // CHECK-LABEL: func.func @first
  // CHECK: %[[FIRST:.*]] = memref.get_global @g
  // CHECK: frisk.storage_contract %[[FIRST]]
  // CHECK: return
  func.func @first() {
    %g = memref.get_global @g : memref<4xi32,1>
    %v = frisk.layout_view %g {layout=#global} : memref<4xi32,1> -> memref<4xi32,1>
    return
  }
  // CHECK-LABEL: func.func @second
  // CHECK: %[[SECOND:.*]] = memref.get_global @g
  // CHECK: frisk.storage_contract %[[SECOND]]
  // CHECK: return
  func.func @second() {
    %g = memref.get_global @g : memref<4xi32,1>
    %v = frisk.layout_view %g {layout=#global} : memref<4xi32,1> -> memref<4xi32,1>
    return
  }
  // CHECK-LABEL: func.func @packed
  // CHECK: frisk.storage_contract %arg0
  // CHECK: frisk.layout_view
  // CHECK: return
  func.func @packed(%r:memref<4xi1,3>) {
    %v = frisk.layout_view %r {layout=#packed} : memref<4xi1,3> -> memref<4xi1,3>
    return
  }
}
