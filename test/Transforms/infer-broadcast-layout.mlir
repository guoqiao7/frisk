// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: frisk-opt %s --frisk-infer-layouts -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-infer-layouts -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// GRAPH: hard broadcast-layout
// CHECK-LABEL: func.func @stable_softmax
// CHECK: frisk.tile_load
// CHECK: frisk.reduce_tensor
// CHECK-SAME: frisk.reduction_contract
// CHECK: linalg.broadcast
// CHECK: arith.subf
// CHECK: math.exp
// CHECK: frisk.reduce_tensor
// CHECK-SAME: frisk.reduction_contract
// CHECK: linalg.broadcast
// CHECK: arith.divf
// CHECK: frisk.tile_store
module {
  func.func @stable_softmax(%src: memref<4x8xf32, 1>, %dst: memref<4x8xf32, 1>) {
    %sv = frisk.layout_view %src : memref<4x8xf32, 1> -> memref<4x8xf32, 1>
    %dv = frisk.layout_view %dst : memref<4x8xf32, 1> -> memref<4x8xf32, 1>
    %x = frisk.tile_load %sv : memref<4x8xf32, 1> -> tensor<4x8xf32>
    %max = frisk.reduce_tensor %x {kind = "max", dim = 1 : i64} : tensor<4x8xf32> -> tensor<4xf32>
    %e0 = tensor.empty() : tensor<4x8xf32>
    %bm = linalg.broadcast ins(%max : tensor<4xf32>) outs(%e0 : tensor<4x8xf32>) dimensions = [1]
    %centered = arith.subf %x, %bm : tensor<4x8xf32>
    %exp = math.exp %centered : tensor<4x8xf32>
    %sum = frisk.reduce_tensor %exp {kind = "sum", dim = 1 : i64} : tensor<4x8xf32> -> tensor<4xf32>
    %e1 = tensor.empty() : tensor<4x8xf32>
    %bs = linalg.broadcast ins(%sum : tensor<4xf32>) outs(%e1 : tensor<4x8xf32>) dimensions = [1]
    %y = arith.divf %exp, %bs : tensor<4x8xf32>
    frisk.tile_store %y, %dv : tensor<4x8xf32>, memref<4x8xf32, 1>
    return
  }
}
