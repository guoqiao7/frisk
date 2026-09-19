// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: frisk-opt %s --frisk-infer-layouts -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-infer-layouts -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// GRAPH: hard instruction-contract
// CHECK-LABEL: func.func @ss
// CHECK: frisk.mma
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: frisk.mma_contract = #frisk.mma_contract
module attributes {frisk.target = "sm_90a"} {
  func.func @ss() {
    %a = memref.alloc() {alignment = 1024 : i64} : memref<64x16xf16, 3>
    %b = memref.alloc() {alignment = 1024 : i64} : memref<16x64xf16, 3>
    %av = frisk.layout_view %a : memref<64x16xf16, 3> -> memref<64x16xf16, 3>
    %bv = frisk.layout_view %b : memref<16x64xf16, 3> -> memref<16x64xf16, 3>
    %zero = arith.constant dense<0.0> : tensor<64x64xf32>
    %result = "frisk.mma"(%av, %bv, %zero) {m = 64 : i64, n = 64 : i64, k = 16 : i64}
      : (memref<64x16xf16, 3>, memref<16x64xf16, 3>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return
  }
  // CHECK-LABEL: func.func @rs
  // CHECK: frisk.mma
  // CHECK-SAME: form = "rs"
  func.func @rs() {
    %a = arith.constant dense<1.0> : tensor<64x16xbf16>
    %b = memref.alloc() {alignment = 1024 : i64} : memref<16x64xbf16, 3>
    %bv = frisk.layout_view %b : memref<16x64xbf16, 3> -> memref<16x64xbf16, 3>
    %zero = arith.constant dense<0.0> : tensor<64x64xf32>
    %result = "frisk.mma"(%a, %bv, %zero) {m = 64 : i64, n = 64 : i64, k = 16 : i64}
      : (tensor<64x16xbf16>, memref<16x64xbf16, 3>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return
  }
}
