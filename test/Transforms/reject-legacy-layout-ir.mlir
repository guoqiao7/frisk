// RUN: frisk-opt %s -frisk-infer-layouts --verify-diagnostics
func.func @legacy(%a:memref<4x4xf32>, %b:memref<4xf32>) {
  // expected-error@+1 {{legacy layout IR requires frisk-normalize-layout-ir or frisk-layout-pipeline}}
  "frisk.reduce"(%a, %b) {kind="add", dim=1:i64, clear=true} : (memref<4x4xf32>,memref<4xf32>) -> ()
  return
}
