// RUN: frisk-opt %s -frisk-infer-layouts='analysis-only stats-file=%t.json' -o /dev/null
// RUN: cat %t.json | FileCheck %s --check-prefix=SOLVED
// RUN: frisk-opt %s -frisk-layout-pipeline='stats-file=%t.pipeline.json' -o /dev/null
// RUN: cat %t.pipeline.json | FileCheck %s --check-prefix=COMMITTED
// SOLVED: "attempts": [
// SOLVED: "preparedCandidates": 0
// SOLVED: "schema_version": 1
// SOLVED: "status": "solved"
// COMMITTED: "status": "committed"
module {}
