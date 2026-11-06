// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.first
// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.second
// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.third
// RUN: diff %t.first %t.second
// RUN: diff %t.first %t.third
// RUN: FileCheck %s < %t.first
// Cross-process byte comparison retains actual layout types and operation
// bindings, not merely solver cost or a filtered diagnostic summary.
// CHECK: func.func @deterministic
// CHECK: frisk.tile_load
// CHECK-SAME: #frisk.distributed
// CHECK: frisk.tile_store
func.func @deterministic(%in: memref<8x8xf32,1>, %out: memref<8x8xf32,1>) {
  %a = frisk.layout_view %in : memref<8x8xf32,1> -> memref<8x8xf32,1>
  %b = frisk.layout_view %out : memref<8x8xf32,1> -> memref<8x8xf32,1>
  %v = frisk.tile_load %a : memref<8x8xf32,1> -> tensor<8x8xf32>
  %n = arith.negf %v : tensor<8x8xf32>
  frisk.tile_store %n, %b : tensor<8x8xf32>, memref<8x8xf32,1>
  return
}
