// RUN: frisk-opt %s --frisk-layout-pipeline -split-input-file -verify-diagnostics

module attributes {frisk.target="sm_90a"} {
  func.func @hard_alignment() {
    %s = memref.alloc() {alignment=1:i64} : memref<8x16xf32,1>
    %d = memref.alloc() {alignment=1024:i64} : memref<8x16xf32,3>
    %sv = frisk.layout_view %s : memref<8x16xf32,1> -> memref<8x16xf32,1>
    %dv = frisk.layout_view %d : memref<8x16xf32,3> -> memref<8x16xf32,3>
    // expected-error@+1 {{copy-joint-contract: no proven instruction tuple in frozen candidate domains; counterexample: copy-alignment: root alignment evidence is insufficient for vector access; operand 0}}
    frisk.copy %sv[], %dv[] {srcExtents=array<i64:8,16>,dstExtents=array<i64:8,16>,frisk.vector_bytes=16:i64}
      : memref<8x16xf32,1>, memref<8x16xf32,3>
    return
  }
}

// -----

module attributes {frisk.target="sm_90a"} {
  func.func @guarded_async_not_admitted() {
    %s = memref.alloc() {alignment=128:i64} : memref<3x5xf16,1>
    %d = memref.alloc() {alignment=1024:i64} : memref<3x5xf16,3>
    %sv = frisk.layout_view %s : memref<3x5xf16,1> -> memref<3x5xf16,1>
    %dv = frisk.layout_view %d : memref<3x5xf16,3> -> memref<3x5xf16,3>
    // expected-error@+1 {{copy-joint-contract: no proven instruction tuple in frozen candidate domains; counterexample: copy-shape: guarded Copy only admits the scalar vector path}}
    frisk.copy %sv[], %dv[] {srcExtents=array<i64:3,5>,dstExtents=array<i64:3,5>,frisk.copy_contract={path="cp.async"}}
      : memref<3x5xf16,1>, memref<3x5xf16,3>
    return
  }
}
