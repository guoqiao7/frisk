// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0" -verify-each | FileCheck %s
// No memory operation supplies a seed. Use the target's legal domain.
// CHECK: access_units = 0
// CHECK-SAME: anchors = []
// CHECK-SAME: distinct_conversions = 0
// CHECK: arith.addf
module attributes {frisk.target="sm_90a"} {
  func.func @add(%x: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %v = arith.addf %x, %x : tensor<32x64xf32>
    return %v : tensor<32x64xf32>
  }
}
