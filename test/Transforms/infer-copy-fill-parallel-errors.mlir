// RUN: frisk-opt %s --frisk-infer-layouts --split-input-file --verify-diagnostics

func.func @invalid_width(%p: memref<4xf32, 3>) {
  %v = frisk.layout_view %p : memref<4xf32, 3> -> memref<4xf32, 3>
  // expected-error@+1 {{frisk.vector_bytes}}
  frisk.fill %v {value = 0.0 : f32, frisk.vector_bytes = 3 : i64} : memref<4xf32, 3>
  return
}

// -----

"frisk.kernel"() <{sym_name = "thread_conflict", function_type = (memref<4xf32, 3>) -> ()}> ({
^bb0(%p: memref<4xf32, 3>):
  "frisk.parallel"() <{ranges = array<i64: 4>, threads = 128 : i64}> ({
  ^bb0(%i: index):
    %v = frisk.layout_view %p : memref<4xf32, 3> -> memref<4xf32, 3>
    // expected-error@+1 {{execution_threads conflicts with enclosing parallel thread count}}
    frisk.fill %v {value = 0.0 : f32, frisk.execution_threads = 32 : i64} : memref<4xf32, 3>
    "frisk.end"() : () -> ()
  }) : () -> ()
  "frisk.end"() : () -> ()
}) : () -> ()

// -----

func.func @unsupported_extent(%p: memref<3xf32, 3>) {
  %v = frisk.layout_view %p : memref<3xf32, 3> -> memref<3xf32, 3>
  // expected-error@+1 {{operation execution layout requires static power-of-two tile extents greater than one}}
  frisk.fill %v {value = 0.0 : f32} : memref<3xf32, 3>
  return
}
