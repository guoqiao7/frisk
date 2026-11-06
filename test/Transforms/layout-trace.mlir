// RUN: frisk-opt %s -frisk-infer-layouts='analysis-only trace-file=%t.json trace-max-events=4096 trace-max-bytes=8388608' -o /dev/null
// RUN: cat %t.json | FileCheck %s --check-prefix=SOLVED
// RUN: frisk-opt %s -frisk-layout-pipeline='trace-file=%t.pipeline.json trace-max-events=4096 trace-max-bytes=8388608' -o /dev/null
// RUN: cat %t.pipeline.json | FileCheck %s --check-prefix=VERIFIED
// RUN: python3 %S/Inputs/check-layout-trace.py %t.json full
// RUN: python3 %S/Inputs/check-layout-trace.py %t.pipeline.json full
// RUN: frisk-opt %s -frisk-layout-pipeline='trace-file=%t.cap.json trace-max-events=1 trace-max-bytes=4096' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-trace.py %t.cap.json capped
// RUN: frisk-opt %S/layout-relaxation.mlir -frisk-layout-pipeline='trace-file=%t.nonempty.json' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-trace.py %t.nonempty.json full
// RUN: frisk-opt %S/layout-relaxation.mlir -frisk-infer-layouts='trace-file=%t.partial.json trace-max-events=1' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-trace.py %t.partial.json capped
// RUN: frisk-opt %S/layout-relaxation.mlir -o %t.before.mlir
// RUN: frisk-opt %S/layout-relaxation.mlir -frisk-infer-layouts='analysis-only trace-file=%t.analysis.json' -o %t.analysis.mlir
// RUN: diff %t.before.mlir %t.analysis.mlir
// RUN: python3 %S/Inputs/check-layout-trace.py %t.analysis.json full
// RUN: not frisk-opt %S/layout-relaxation.mlir -frisk-infer-layouts='trace-file=%t.fail.json max-expanded-states=1' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-trace.py %t.fail.json failed
// RUN: not frisk-opt %S/layout-relaxation.mlir -frisk-layout-pipeline='trace-file=%t.pipeline.fail.json max-expanded-states=1' -o /dev/null
// RUN: python3 %S/Inputs/check-layout-trace.py %t.pipeline.fail.json failed
// RUN: not frisk-opt %s -frisk-infer-layouts='trace-max-events=0' -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not frisk-opt %s -frisk-layout-pipeline='trace-max-bytes=4095' -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: not frisk-opt %s -frisk-infer-layouts='trace-file=%t.same stats-file=%t.same' -o /dev/null 2>&1 | FileCheck %s --check-prefix=COLLISION
// SOLVED: "analysis_status":"solved"
// SOLVED-SAME: "verification_status":"solved-graph-verified"
// VERIFIED: "verification_status":"pipeline-verified"
// INVALID: trace-max-events must be positive and trace-max-bytes must be at least 4096
// COLLISION: trace-file and stats-file must name different output paths
module {}
