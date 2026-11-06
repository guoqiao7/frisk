// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: frisk-opt %s --frisk-infer-layouts -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-infer-layouts -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// GRAPH: hard reduction-layout
// GRAPH: candidate-preparation {{.*}}reduction-pairs=
// CHECK-LABEL: func.func @rows
// CHECK: frisk.reduce_tensor
// CHECK-SAME: frisk.execution_threads = 32 : i64
// CHECK-SAME: frisk.reduction_contract = #frisk.reduction_contract
module {
  func.func @rows() {
    %a = arith.constant dense<1.0> : tensor<64x64xf32>
    %r = "frisk.reduce_tensor"(%a) {kind = "sum", dim = 1 : i64}
      : (tensor<64x64xf32>) -> tensor<64xf32>
    return
  }
  // CHECK-LABEL: func.func @middle_then_first
  // CHECK: frisk.reduce_tensor
  // CHECK: frisk.reduce_tensor
  func.func @middle_then_first() {
    %a = arith.constant dense<1.0> : tensor<4x8x16xbf16>
    %r = "frisk.reduce_tensor"(%a) {kind = "max", dim = 1 : i64}
      : (tensor<4x8x16xbf16>) -> tensor<4x16xbf16>
    %s = "frisk.reduce_tensor"(%r) {kind = "min", dim = 0 : i64}
      : (tensor<4x16xbf16>) -> tensor<16xbf16>
    return
  }
}
