// RUN: frisk-opt %s -allow-unregistered-dialect -frisk-normalize-layout-ir -verify-diagnostics
// RUN: frisk-opt %s -allow-unregistered-dialect -frisk-infer-layouts -verify-diagnostics
// An unused block argument remains an IR type contract. Neither normalization
// nor direct inference may overlook an obsolete layout hidden in such a type.
module {
  // expected-error@+1 {{legacy-layout-attribute}}
  "test.region"() ({
  ^bb0(%unused: tensor<4xf32, #frisk.layout<[4], affine_map<(d)->(d)>>>):
    "test.end"() : () -> ()
  }) : () -> ()
}
