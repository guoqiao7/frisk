// RUN: frisk-opt %s --frisk-infer-layouts | FileCheck %s
// RUN: frisk-opt %s --frisk-infer-layouts --frisk-infer-layouts | FileCheck %s

#a_map = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [1, 5], outputs = ["d"], output_bits = [6], matrix = dense<[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]> : tensor<6x6xi1>>
#b_map = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [1, 5], outputs = ["d"], output_bits = [6], matrix = dense<[[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0]]> : tensor<6x6xi1>>
#a = #frisk.distributed<map = #a_map, topology = [2,32,1,1,1], replication = 1>
#b = #frisk.distributed<map = #b_map, topology = [2,32,1,1,1], replication = 1>

// CHECK-LABEL: func.func @shared
// CHECK: %[[X:.*]] = frisk.convert_layout %arg0
// CHECK-NEXT: %[[T0:.*]] = linalg.transpose ins(%[[X]]
// CHECK-NOT: frisk.convert_layout
// CHECK: %[[T1:.*]] = linalg.transpose ins(%[[X]]
// CHECK: return %[[T0]], %[[T1]]
func.func @shared(%x: tensor<64xf32, #a>) -> (tensor<64xf32, #b>, tensor<64xf32, #b>) {
  %init = tensor.empty() : tensor<64xf32, #b>
  %t0 = linalg.transpose ins(%x : tensor<64xf32, #a>) outs(%init : tensor<64xf32, #b>) permutation = [0]
  %t1 = linalg.transpose ins(%x : tensor<64xf32, #a>) outs(%init : tensor<64xf32, #b>) permutation = [0]
  return %t0, %t1 : tensor<64xf32, #b>, tensor<64xf32, #b>
}

// CHECK-LABEL: func.func @rematerialize
// CHECK: %[[A:.*]] = arith.constant dense<3.000000e+00>
// CHECK-NOT: frisk.convert_layout
// CHECK: %[[B:.*]] = arith.constant dense<3.000000e+00>
// CHECK-NOT: frisk.convert_layout
// CHECK: return
func.func @rematerialize() -> (tensor<64xf32, #a>, tensor<64xf32, #b>) {
  %c = arith.constant dense<3.0> : tensor<64xf32>
  %a = tensor.empty() : tensor<64xf32, #a>
  %b = tensor.empty() : tensor<64xf32, #b>
  %t0 = linalg.transpose ins(%c : tensor<64xf32>) outs(%a : tensor<64xf32, #a>) permutation = [0]
  %t1 = linalg.transpose ins(%c : tensor<64xf32>) outs(%b : tensor<64xf32, #b>) permutation = [0]
  return %t0, %t1 : tensor<64xf32, #a>, tensor<64xf32, #b>
}

// CHECK-LABEL: func.func @regions
// CHECK: scf.if
// CHECK: frisk.convert_layout %arg1
// CHECK: scf.yield
// CHECK: } else {
// CHECK: frisk.convert_layout %arg1
// CHECK: scf.yield
func.func @regions(%p: i1, %x: tensor<64xf32, #a>) -> tensor<64xf32, #b> {
  %r = scf.if %p -> tensor<64xf32, #b> {
    %init = tensor.empty() : tensor<64xf32, #b>
    %t = linalg.transpose ins(%x : tensor<64xf32, #a>) outs(%init : tensor<64xf32, #b>) permutation = [0]
    scf.yield %t : tensor<64xf32, #b>
  } else {
    %init = tensor.empty() : tensor<64xf32, #b>
    %t = linalg.transpose ins(%x : tensor<64xf32, #a>) outs(%init : tensor<64xf32, #b>) permutation = [0]
    scf.yield %t : tensor<64xf32, #b>
  }
  return %r : tensor<64xf32, #b>
}
