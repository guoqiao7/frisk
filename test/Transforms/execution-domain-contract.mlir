// RUN: frisk-opt %s -frisk-normalize-layout-ir -split-input-file -verify-diagnostics | FileCheck %s
// CHECK: module attributes {frisk.execution_domain = "sm90-scalar-f32-dag-v1"}
module attributes {frisk.execution_domain = "sm90-scalar-f32-dag-v1"} {
  func.func @valid() { return }
}
// -----
// expected-error@+1 {{execution-domain-contract: expected module-level sm90-scalar-f32-dag-v1}}
module attributes {frisk.execution_domain = "unknown-domain"} {}
// -----
// expected-error@+1 {{execution-domain-contract: expected module-level sm90-scalar-f32-dag-v1}}
module attributes {frisk.execution_domain = 1 : i64} {}
// -----
module {
  // expected-error@+1 {{execution-domain-contract: expected module-level sm90-scalar-f32-dag-v1}}
  func.func @misplaced() attributes {frisk.execution_domain = "sm90-scalar-f32-dag-v1"} { return }
}
