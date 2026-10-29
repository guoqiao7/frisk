// Actual Attention, default trace/search budgets. No summary-only substitute.
// RUN: frisk-opt %S/attention-layout.mlir -frisk-layout-pipeline='stats-file=%t.off.stats.json' -o %t.off.mlir
// RUN: frisk-opt %S/attention-layout.mlir -frisk-layout-pipeline='stats-file=%t.on.stats.json trace-file=%t.on.trace.json' -o %t.on.mlir
// RUN: python3 %S/Inputs/check-layout-trace.py %t.on.trace.json attention
// RUN: python3 %S/Inputs/check-layout-trace.py %t.on.trace.json refinement-enabled
// RUN: diff %t.off.mlir %t.on.mlir
// RUN: python3 %S/Inputs/check-layout-trace.py %t.off.stats.json compare-stats %t.on.stats.json
// RUN: frisk-opt %S/attention-layout.mlir -frisk-layout-pipeline='trace-file=%t.repeat.trace.json' -o %t.repeat.mlir
// RUN: diff %t.on.mlir %t.repeat.mlir
// RUN: python3 %S/Inputs/check-layout-trace.py %t.on.trace.json compare-trace %t.repeat.trace.json
// RUN: frisk-opt %S/attention-layout.mlir -frisk-layout-pipeline='refine-beam-solution=false trace-file=%t.unrefined.trace.json' -verify-each -o %t.unrefined.mlir
// RUN: python3 %S/Inputs/check-layout-trace.py %t.unrefined.trace.json refinement-disabled
// RUN: frisk-opt %S/attention-layout.mlir -frisk-layout-pipeline='max-refinement-probes=0 trace-file=%t.zero.trace.json' -verify-each -o %t.zero.mlir
// RUN: python3 %S/Inputs/check-layout-trace.py %t.zero.trace.json refinement-disabled
// RUN: diff %t.unrefined.mlir %t.zero.mlir
module {}
