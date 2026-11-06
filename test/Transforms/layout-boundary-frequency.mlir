// RUN: frisk-opt %s --frisk-layout-pipeline -o %t.default.mlir
// RUN: frisk-opt %s --frisk-layout-pipeline="boundary-frequency=legacy" -o %t.legacy.mlir
// RUN: diff %t.default.mlir %t.legacy.mlir
// RUN: frisk-opt %s --frisk-layout-pipeline="boundary-frequency=static-trip-count trace-file=%t.json stats-file=%t.stats.json" -o %t.static.mlir
// RUN: python3 %S/check-boundary-frequency.py %t.json %t.stats.json
// RUN: frisk-opt %s --frisk-layout-pipeline="boundary-frequency=static-trip-count trace-file=%t.again.json" -o %t.again.mlir
// RUN: diff %t.static.mlir %t.again.mlir
// RUN: diff %t.json %t.again.json
// RUN: not frisk-opt %s --frisk-infer-layouts="boundary-frequency=invalid" -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not frisk-opt %s --frisk-optimize-layout-conversions="boundary-frequency=invalid" -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not frisk-opt %s --frisk-layout-pipeline="boundary-frequency=invalid" -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
// INVALID: invalid boundary-frequency: expected legacy or static-trip-count
// Fixture-local, hard register-exchange boundary executed 100 times.
#a = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [2, 5], outputs = ["d0"], output_bits = [7], matrix = dense<[[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]> : tensor<7x7xi1>>
#b = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [2, 5], outputs = ["d0"], output_bits = [7], matrix = dense<[[0,1,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]> : tensor<7x7xi1>>
#A = #frisk.distributed<map = #a, topology = [4,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [4,32,1,1,1], replication = 1>
func.func @loop_boundary(%x: tensor<128xf32, #A>, %init: tensor<128xf32, #B>) -> tensor<128xf32, #B> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %result = scf.for %i = %c0 to %c100 step %c1 iter_args(%acc = %init) -> tensor<128xf32, #B> {
    %converted = linalg.transpose ins(%x : tensor<128xf32, #A>) outs(%acc : tensor<128xf32, #B>) permutation = [0]
    %sum = arith.addf %converted, %acc : tensor<128xf32, #B>
    scf.yield %sum : tensor<128xf32, #B>
  }
  return %result : tensor<128xf32, #B>
}
