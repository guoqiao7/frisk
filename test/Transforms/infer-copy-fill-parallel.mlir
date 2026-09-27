// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s --check-prefix=GRAPH
// RUN: frisk-opt %s --frisk-infer-layouts -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once --frisk-infer-layouts -verify-each > %t.twice
// RUN: diff %t.once %t.twice
// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.pipeline

// GRAPH-DAG: operation-execution {{.*}} threads=128 writer=first_owner vector-bytes=1
// GRAPH-DAG: hard copy-access
// GRAPH-DAG: hard ownership
// GRAPH-DAG: hard resource-limit
// GRAPH: propagation strict initial=
// GRAPH: propagation common initial=

// CHECK: frisk.kernel @copy_fill
// CHECK: ^bb0(%{{.*}}: memref<4xf32, 1>, %{{.*}}: memref<4xf32, 3>):
// CHECK: frisk.parallel
// CHECK: threads = 128
// CHECK: frisk.fill
// CHECK-SAME: frisk.execution_layout = #frisk.distributed
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: frisk.vector_bytes = 1 : i64
// CHECK-SAME: frisk.writer_policy = "first_owner"
// CHECK: frisk.copy
// CHECK-SAME: frisk.execution_layout = #frisk.distributed
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: frisk.vector_bytes = 1 : i64
// CHECK-SAME: frisk.writer_policy = "first_owner"
"frisk.kernel"() <{sym_name = "copy_fill", function_type = (memref<4xf32, 1>, memref<4xf32, 3>) -> ()}> ({
^bb0(%src: memref<4xf32, 1>, %dst: memref<4xf32, 3>):
  "frisk.parallel"() <{ranges = array<i64: 4>, threads = 128 : i64}> ({
  ^bb0(%i: index):
    %s = frisk.layout_view %src : memref<4xf32, 1> -> memref<4xf32, 1>
    %d = frisk.layout_view %dst : memref<4xf32, 3> -> memref<4xf32, 3>
    frisk.fill %d {value = 0.0 : f32} : memref<4xf32, 3>
    "frisk.copy"(%s, %d) <{srcMap = affine_map<() -> ()>, dstMap = affine_map<() -> ()>, srcExtents = array<i64: 4>, dstExtents = array<i64: 4>}> {operandSegmentSizes = array<i32: 1, 1, 0, 0>} : (memref<4xf32, 1>, memref<4xf32, 3>) -> ()
    "frisk.end"() : () -> ()
  }) : () -> ()
  "frisk.end"() : () -> ()
}) : () -> ()

// Also replay a zero-input kernel; its body captures a function argument.
func.func @zero_input_kernel(%p: memref<4xf32, 3>) {
  "frisk.kernel"() <{sym_name = "captured", function_type = () -> ()}> ({
    "frisk.parallel"() <{ranges = array<i64: 4>, threads = 32 : i64}> ({
    ^bb0(%i: index):
      %v = frisk.layout_view %p : memref<4xf32, 3> -> memref<4xf32, 3>
      frisk.fill %v {value = 0.0 : f32} : memref<4xf32, 3>
      "frisk.end"() : () -> ()
    }) : () -> ()
    "frisk.end"() : () -> ()
  }) : () -> ()
  return
}
