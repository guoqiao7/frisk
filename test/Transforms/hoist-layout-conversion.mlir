// RUN: frisk-opt %s -frisk-optimize-layout-conversions -verify-each | FileCheck %s
#a = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [2, 5], outputs = ["d0"], output_bits = [7], matrix = dense<[[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]> : tensor<7x7xi1>>
#b = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [2, 5], outputs = ["d0"], output_bits = [7], matrix = dense<[[0,1,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]> : tensor<7x7xi1>>
#A = #frisk.distributed<map = #a, topology = [4,32,1,1,1], replication = 1>
#B = #frisk.distributed<map = #b, topology = [4,32,1,1,1], replication = 1>
func.func @commute(%x: tensor<128xf32, #A>) -> tensor<128xf32, #B> {
  // This is the private authorization emitted for non-hard inferred producers.
  %sum = arith.addf %x, %x {frisk.inferred_result_layout = #A} : tensor<128xf32, #A>
  %result = frisk.convert_layout %sum : tensor<128xf32, #A> -> tensor<128xf32, #B>
  return %result : tensor<128xf32, #B>
}
// CHECK-LABEL: func.func @commute
// CHECK: %[[C:.*]] = frisk.convert_layout %arg0
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[C]], %[[C]]
// CHECK-NEXT: return %[[SUM]]

func.func @register_loop(%x: tensor<128xf32, #A>, %init: tensor<128xf32, #B>) -> tensor<128xf32, #B> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> tensor<128xf32, #B> {
    %converted = frisk.convert_layout %x : tensor<128xf32, #A> -> tensor<128xf32, #B>
    %sum = arith.addf %converted, %acc : tensor<128xf32, #B>
    scf.yield %sum : tensor<128xf32, #B>
  }
  return %result : tensor<128xf32, #B>
}
// CHECK-LABEL: func.func @register_loop
// CHECK: %[[C:.*]] = frisk.convert_layout %arg0
// CHECK-NEXT: %{{.*}} = scf.for
// CHECK-NOT: frisk.convert_layout
// CHECK: arith.addf %[[C]],
