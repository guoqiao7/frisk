// RUN: frisk-opt %s -allow-unregistered-dialect -frisk-infer-layouts -split-input-file -verify-diagnostics
// RUN: frisk-opt %s -allow-unregistered-dialect -frisk-normalize-layout-ir -split-input-file -verify-diagnostics

#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
func.func @later_declaration(%r: memref<4xi32,3>) {
  "test.region"() ({
    %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
    %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
    // expected-error@+1 {{storage-root-contract: unsupported region scope}}
    frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
    "test.end"() : () -> ()
  }) : () -> ()
  return
}
