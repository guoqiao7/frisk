// RUN: frisk-opt %s --frisk-infer-layouts --split-input-file --verify-diagnostics

func.func @wrong_transpose(%a: memref<16x64xf16, 3>, %b: memref<16x64xf16, 3>, %init: tensor<64x64xf32>) {
  // expected-error@+1 {{mma-shape: matrix extents disagree}}
  %r = "frisk.mma"(%a, %b, %init) {m = 64 : i64, n = 64 : i64, k = 16 : i64}
    : (memref<16x64xf16, 3>, memref<16x64xf16, 3>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return
}

// -----

module attributes {frisk.target = "sm_90a"} {
  func.func @wrong_group(%a: memref<64x16xf16, 3>, %b: memref<16x64xf16, 3>, %init: tensor<64x64xf32>) {
    %av = frisk.layout_view %a : memref<64x16xf16, 3> -> memref<64x16xf16, 3>
    %bv = frisk.layout_view %b : memref<16x64xf16, 3> -> memref<16x64xf16, 3>
    // expected-error@+1 {{sm90-mma-thread-group: expected 128, 256, 512 or 1024 threads}}
    %r = "frisk.mma"(%av, %bv, %init) {m = 64 : i64, n = 64 : i64, k = 16 : i64, frisk.execution_threads = 32 : i64}
      : (memref<64x16xf16, 3>, memref<16x64xf16, 3>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return
  }
}
