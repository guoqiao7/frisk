// RUN: if frisk-opt %s --frisk-infer-layouts='max-expanded-states=1' > %t.first 2>&1; then exit 1; fi
// RUN: if frisk-opt %s --frisk-infer-layouts='max-expanded-states=1' > %t.second 2>&1; then exit 1; fi
// RUN: diff %t.first %t.second
// RUN: FileCheck %s < %t.first
// CHECK: search-budget-exceeded:
// CHECK-NOT: no feasible
// CHECK-NOT: 0x
func.func @budget_diagnostic(%in: memref<8x8xf32,1>, %out: memref<8x8xf32,1>) {
  %a = frisk.layout_view %in : memref<8x8xf32,1> -> memref<8x8xf32,1>
  %b = frisk.layout_view %out : memref<8x8xf32,1> -> memref<8x8xf32,1>
  %v = frisk.tile_load %a : memref<8x8xf32,1> -> tensor<8x8xf32>
  frisk.tile_store %v, %b : tensor<8x8xf32>, memref<8x8xf32,1>
  return
}
