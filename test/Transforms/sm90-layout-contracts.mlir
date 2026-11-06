// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-layout-pipeline -verify-each > %t.twice
// RUN: diff %t.once %t.twice
// RUN: frisk-opt %s --frisk-infer-layouts='dump-analysis' -verify-each 2>&1 | FileCheck %s --check-prefix=NORETRY --implicit-check-not='relaxation '
// NORETRY: solver total cost=
// NORETRY: func.func @bulk
// NORETRY: path = "tma"
// NORETRY: func.func @ragged
// NORETRY: guarded = true
// NORETRY: func.func @unaligned_fallback
// NORETRY: path = "vector"

module attributes {nvvm.target = #nvvm.target<chip = "sm_90", features = "+ptx80">} {
  // CHECK-LABEL: func.func @bulk
  // CHECK: frisk.copy
  // CHECK-SAME: completion = "mbarrier-and-async-proxy"
  // CHECK-SAME: guarded = false
  // CHECK-SAME: path = "tma"
  func.func @bulk() {
    %s = memref.alloc() {alignment = 128 : i64} : memref<32x64xf32,1>
    %d = memref.alloc() {alignment = 1024 : i64} : memref<32x64xf32,3>
    %sv = frisk.layout_view %s : memref<32x64xf32,1> -> memref<32x64xf32,1>
    %dv = frisk.layout_view %d : memref<32x64xf32,3> -> memref<32x64xf32,3>
    "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:32,64>,dstExtents=array<i64:32,64>}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<32x64xf32,1>,memref<32x64xf32,3>)->()
    return
  }
  // CHECK-LABEL: func.func @ragged
  // CHECK: frisk.copy
  // CHECK-SAME: guarded = true
  // CHECK-SAME: path = "vector"
  // CHECK-SAME: valid_shape = array<i64: 3, 5>
  // CHECK-SAME: vector_bytes = 1
  func.func @ragged(%s:memref<3x5xf32,1>, %d:memref<3x5xf32,3>) {
    %sv = frisk.layout_view %s : memref<3x5xf32,1> -> memref<3x5xf32,1>
    %dv = frisk.layout_view %d : memref<3x5xf32,3> -> memref<3x5xf32,3>
    "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:3,5>,dstExtents=array<i64:3,5>}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<3x5xf32,1>,memref<3x5xf32,3>)->()
    return
  }
  // CHECK-LABEL: func.func @unaligned_fallback
  // CHECK: frisk.copy
  // CHECK-SAME: path = "vector"
  func.func @unaligned_fallback(%s:memref<8x16xf32,1>, %d:memref<8x16xf32,3>) {
    %sv = frisk.layout_view %s : memref<8x16xf32,1> -> memref<8x16xf32,1>
    %dv = frisk.layout_view %d : memref<8x16xf32,3> -> memref<8x16xf32,3>
    "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:8,16>,dstExtents=array<i64:8,16>}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<8x16xf32,1>,memref<8x16xf32,3>)->()
    return
  }
}
