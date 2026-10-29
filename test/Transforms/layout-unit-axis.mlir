// RUN: frisk-opt %s --frisk-infer-layouts -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-infer-layouts -verify-each > %t.twice
// RUN: diff %t.once %t.twice
// CHECK-LABEL: func.func @unit
// CHECK: tensor.expand_shape
// CHECK: linalg.transpose
// CHECK: tensor.collapse_shape
// CHECK: frisk.reduce_tensor
// CHECK-SAME: frisk.reduction_contract
// CHECK: linalg.broadcast
module {
func.func @unit() {
  %a=arith.constant dense<1.0> : tensor<8xf32>
  %e=tensor.expand_shape %a [[0,1]] output_shape [1,8] : tensor<8xf32> into tensor<1x8xf32>
  %empty=tensor.empty() : tensor<8x1xf32>
  %t=linalg.transpose ins(%e:tensor<1x8xf32>) outs(%empty:tensor<8x1xf32>) permutation=[1,0]
  %c=tensor.collapse_shape %t [[0,1]] : tensor<8x1xf32> into tensor<8xf32>
  %r=frisk.reduce_tensor %e {kind="sum",dim=1:i64} : tensor<1x8xf32> -> tensor<1xf32>
  %dest=tensor.empty() : tensor<1x1xf32>
  %b=linalg.broadcast ins(%r:tensor<1xf32>) outs(%dest:tensor<1x1xf32>) dimensions=[1]
  %rr=frisk.reduce_tensor %b {kind="max",dim=0:i64} : tensor<1x1xf32> -> tensor<1xf32>
  return
}
}
