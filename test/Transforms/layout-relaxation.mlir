// RUN: frisk-opt %s --frisk-infer-layouts | FileCheck %s --check-prefix=SHARED --implicit-check-not=frisk.convert_layout
// RUN: frisk-opt %s --frisk-infer-layouts --frisk-infer-layouts | FileCheck %s --check-prefix=SHARED --implicit-check-not=frisk.convert_layout
// RUN: frisk-opt %s --frisk-infer-layouts | FileCheck %s --check-prefix=COUNT

#a_map = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [1, 5], outputs = ["d"], output_bits = [6], matrix = dense<[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]> : tensor<6x6xi1>>
#b_map = #frisk.bit_linear<inputs = ["register", "lane"], input_bits = [1, 5], outputs = ["d"], output_bits = [6], matrix = dense<[[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1],[1,0,0,0,0,0]]> : tensor<6x6xi1>>
#a = #frisk.distributed<map = #a_map, topology = [2,32,1,1,1], replication = 1>
#b = #frisk.distributed<map = #b_map, topology = [2,32,1,1,1], replication = 1>

// Whichever layout is selected for the original, the other layout must get
// one shared selected clone, not a shared conversion or two per-use clones.
// SHARED-LABEL: func.func @shared_rematerialization
// SHARED: %[[A0:.*]] = linalg.transpose ins(%[[A:[a-zA-Z0-9_]+]]
// SHARED: %[[A1:.*]] = linalg.transpose ins(%[[A]]
// SHARED: %[[B0:.*]] = linalg.transpose ins(%[[B:[a-zA-Z0-9_]+]]
// SHARED: %[[B1:.*]] = linalg.transpose ins(%[[B]]
// SHARED: return %[[A0]], %[[A1]], %[[B0]], %[[B1]]
// COUNT-LABEL: func.func @shared_rematerialization
// COUNT-COUNT-2: arith.constant dense<3.000000e+00>
// COUNT-NOT: arith.constant
// COUNT: return
func.func @shared_rematerialization() -> (tensor<64xf32, #a>, tensor<64xf32, #a>, tensor<64xf32, #b>, tensor<64xf32, #b>) {
  %c = arith.constant dense<3.0> : tensor<64xf32>
  %a = tensor.empty() : tensor<64xf32, #a>
  %b = tensor.empty() : tensor<64xf32, #b>
  %a0 = linalg.transpose ins(%c : tensor<64xf32>) outs(%a : tensor<64xf32, #a>) permutation = [0]
  %a1 = linalg.transpose ins(%c : tensor<64xf32>) outs(%a : tensor<64xf32, #a>) permutation = [0]
  %b0 = linalg.transpose ins(%c : tensor<64xf32>) outs(%b : tensor<64xf32, #b>) permutation = [0]
  %b1 = linalg.transpose ins(%c : tensor<64xf32>) outs(%b : tensor<64xf32, #b>) permutation = [0]
  return %a0, %a1, %b0, %b1 : tensor<64xf32, #a>, tensor<64xf32, #a>, tensor<64xf32, #b>, tensor<64xf32, #b>
}
