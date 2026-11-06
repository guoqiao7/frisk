// RUN: frisk-opt %s -frisk-infer-layouts='analysis-only profile-file=%t.json' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-profile.py %t.json solved
// RUN: frisk-opt %s -frisk-layout-pipeline='profile-file=%t.pipeline.json' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-profile.py %t.pipeline.json committed
// RUN: not frisk-opt %s -frisk-infer-layouts='profile-file=%t.same stats-file=%t.same' -o /dev/null 2>&1 | FileCheck %s --check-prefix=STATS
// RUN: not frisk-opt %s -frisk-layout-pipeline='profile-file=%t.same trace-file=%t.same' -o /dev/null 2>&1 | FileCheck %s --check-prefix=TRACE
// RUN: not frisk-opt %S/layout-relaxation.mlir -frisk-infer-layouts='profile-file=%t.fail.json max-expanded-states=1' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-profile.py %t.fail.json failed
// RUN: frisk-opt %S/layout-relaxation.mlir -frisk-layout-pipeline='trace-file=%t.off.trace' -o %t.off.mlir
// RUN: frisk-opt %S/layout-relaxation.mlir -frisk-layout-pipeline='trace-file=%t.on.trace profile-file=%t.small.json' -o %t.on.mlir
// RUN: diff %t.off.mlir %t.on.mlir
// RUN: diff %t.off.trace %t.on.trace
// RUN: python3 %S/Inputs/check-layout-profile.py %t.small.json committed conversion_choice conversion_classification relation_query relation_cache
// RUN: frisk-opt %S/copy-mma-contract.mlir -frisk-layout-pipeline='profile-file=%t.copy.json' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-profile.py %t.copy.json committed address_enumeration execution_enumeration access_cost instruction_support instruction_contract footprint_cache
// STATS: profile-file and stats-file must name different output paths
// TRACE: profile-file and trace-file must name different output paths
module {}
