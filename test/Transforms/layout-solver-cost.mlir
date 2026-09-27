// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' 2>&1 | FileCheck %s
// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' > %t.first 2>&1
// RUN: frisk-opt %s --frisk-infer-layouts='analysis-only dump-analysis' > %t.second 2>&1
// RUN: diff %t.first %t.second
// RUN: if frisk-opt %s --frisk-infer-layouts='max-expanded-states=1' > %t.err 2>&1; then exit 1; fi
// RUN: FileCheck %s --check-prefix=BUDGET < %t.err
// RUN: if frisk-opt %s --frisk-layout-pipeline='max-expanded-states=1' > %t.err 2>&1; then exit 1; fi
// RUN: FileCheck %s --check-prefix=BUDGET < %t.err
// RUN: if frisk-opt %s --frisk-infer-layouts='beam-width=0' > %t.err 2>&1; then exit 1; fi
// RUN: FileCheck %s --check-prefix=INVALID < %t.err
// RUN: frisk-opt %s --frisk-layout-pipeline='beam-width=8 exact-combination-limit=16 max-expanded-states=4096' -verify-each > %t.out
// CHECK: cost coverage: cta-staging-upper-bound-v1
// CHECK-SAME: unmodeled=instructionPathAndWork,memoryTransactions,bankConflictDegree
// CHECK: solver component {{.*}}: mode=exact beam-width=32 variables=
// CHECK-SAME: raw={{[0-9]+}} post-prune={{[0-9]+}} raw-overflow=0 product-overflow=0 domains=[
// CHECK-SAME: expanded={{[0-9]+}} hard-rejected={{[0-9]+}} beam-discarded=0
// CHECK-SAME: exhaustive=1 cost=[0,0,0,
// CHECK-SAME: saturated=0
// BUDGET: search-budget-exceeded:
// INVALID: invalid layout solver options
func.func @copy_tensor(%in: memref<8x8xf32,1>, %out: memref<8x8xf32,1>) {
  %a = frisk.layout_view %in : memref<8x8xf32,1> -> memref<8x8xf32,1>
  %b = frisk.layout_view %out : memref<8x8xf32,1> -> memref<8x8xf32,1>
  %v = frisk.tile_load %a : memref<8x8xf32,1> -> tensor<8x8xf32>
  frisk.tile_store %v, %b : tensor<8x8xf32>, memref<8x8xf32,1>
  return
}
