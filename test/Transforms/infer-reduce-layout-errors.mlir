// RUN: frisk-opt %s --frisk-infer-layouts --split-input-file --verify-diagnostics

func.func @wrong_kind(%a: tensor<4x4xf32>) {
  // expected-error@+1 {{kind must be sum/max/min}}
  %r = frisk.reduce_tensor %a {kind = "add", dim = 1 : i64} : tensor<4x4xf32> -> tensor<4xf32>
  return
}

// -----

func.func @wrong_axis(%a: tensor<4x4xf32>) {
  // expected-error@+1 {{rank >= 2 and valid axis required}}
  %r = frisk.reduce_tensor %a {kind = "sum", dim = 2 : i64} : tensor<4x4xf32> -> tensor<4xf32>
  return
}

// -----

func.func @ragged(%a: tensor<4x3xf32>) {
  // expected-error@+1 {{static power-of-two extents greater than one}}
  %r = frisk.reduce_tensor %a {kind = "sum", dim = 1 : i64} : tensor<4x3xf32> -> tensor<4xf32>
  return
}

// -----

func.func @scalar(%a: tensor<4xf32>) {
  // expected-error@+1 {{rank >= 2 and valid axis required}}
  %r = frisk.reduce_tensor %a {kind = "sum", dim = 0 : i64} : tensor<4xf32> -> tensor<f32>
  return
}

// -----

func.func @wrong_dtype(%a: tensor<4x4xf16>) {
  // expected-error@+1 {{matching f16/bf16/f32 element types}}
  %r = frisk.reduce_tensor %a {kind = "sum", dim = 1 : i64} : tensor<4x4xf16> -> tensor<4xf32>
  return
}

// -----

func.func @wrong_shape(%a: tensor<4x8xf32>) {
  // expected-error@+1 {{result must delete the reduction axis}}
  %r = frisk.reduce_tensor %a {kind = "sum", dim = 1 : i64} : tensor<4x8xf32> -> tensor<8xf32>
  return
}

// -----

"frisk.kernel"() <{sym_name="scope_conflict", function_type=()->()}> ({
  "frisk.parallel"() <{ranges=array<i64:4>, threads=128:i64}> ({
  ^bb0(%i:index):
    %a = arith.constant dense<1.0> : tensor<4x4xf32>
    // expected-error@+1 {{execution_threads conflicts with enclosing parallel}}
    %r = frisk.reduce_tensor %a {kind="sum",dim=1:i64,frisk.execution_threads=32:i64} : tensor<4x4xf32> -> tensor<4xf32>
    frisk.end
  }) : () -> ()
  frisk.end
}) : () -> ()
