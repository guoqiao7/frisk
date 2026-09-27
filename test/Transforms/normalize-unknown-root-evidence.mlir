// RUN: frisk-opt %s -allow-unregistered-dialect -frisk-normalize-layout-ir -verify-diagnostics
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
func.func @cannot_preserve_unknown_scope(%r: memref<4xi32,3>) {
  "test.region"() ({
    // expected-error@+1 {{storage-root-contract: unsupported region scope}}
    %v = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
    "test.end"() : () -> ()
  }) : () -> ()
  return
}
