// RUN: frisk-opt %s '-frisk-layout-pipeline=stats-file=%t.off.json trace-file=%t.off.trace.json' -o %t.off.mlir
// RUN: frisk-opt %s '-frisk-layout-pipeline=ir-diagnostics=true stats-file=%t.on.json trace-file=%t.on.trace.json' -o %t.on.mlir
// RUN: diff %t.off.mlir %t.on.mlir
// RUN: frisk-opt %s '-frisk-layout-pipeline=ir-diagnostics=true stats-file=%t.cap.json trace-file=%t.cap.trace.json trace-max-events=1 trace-max-bytes=4096' -o %t.cap.mlir
// RUN: diff %t.on.mlir %t.cap.mlir
// RUN: frisk-opt %t.on.mlir '-frisk-layout-pipeline=ir-diagnostics=true stats-file=%t.replay.json' -o %t.replay.mlir
// RUN: diff %t.on.mlir %t.replay.mlir
// RUN: python3 %S/Inputs/check-layout-ir-diagnostics.py %t
func.func @arithmetic(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %m = arith.mulf %a, %b : tensor<8x8xf32>
  %e = math.exp %m : tensor<8x8xf32>
  %d = arith.divf %e, %m : tensor<8x8xf32>
  return %d, %m : tensor<8x8xf32>, tensor<8x8xf32>
}
