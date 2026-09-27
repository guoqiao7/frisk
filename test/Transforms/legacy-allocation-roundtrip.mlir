// RUN: frisk-opt %s | frisk-opt | FileCheck %s
// Historical custom printer emitted scope-only attrs and its parser attempted
// to add a null memorySpace for Local memrefs. Keep semantic metadata visible.
// CHECK: frisk.alloc_buffer
// CHECK-SAME: frisk.legacy_annotation = "must-not-disappear"
// CHECK: frisk.alloc_buffer
// CHECK-SAME: alignment = 16
func.func @allocations() {
  %local = frisk.alloc_buffer {frisk.legacy_annotation="must-not-disappear"} -> memref<4x4xf32>
  %shared = frisk.alloc_buffer {alignment=16:i64} -> memref<4x4xf16,3>
  return
}
