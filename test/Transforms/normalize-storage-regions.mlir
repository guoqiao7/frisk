// RUN: frisk-opt %s -frisk-normalize-layout-ir | FileCheck %s
// Storage endpoint binding is local to the original operation. This does not
// assign sequential SSA state to the unknown custom region or general CFG.
// CHECK-LABEL: func.func @custom
// CHECK: frisk.block
// CHECK: frisk.layout_view
// CHECK: frisk.fill
func.func @custom(%dst:memref<4xf32,3>) {
  "frisk.block"() <{ranges=array<i64:4>}> ({
  ^bb0(%i:index):
    frisk.fill %dst {value=2.0:f32} : memref<4xf32,3>
    "frisk.end"() : () -> ()
  }) : () -> ()
  return
}
// CHECK-LABEL: func.func @cfg
// CHECK: cf.cond_br
// CHECK: frisk.layout_view
// CHECK: frisk.fill
func.func @cfg(%dst:memref<4xf32,3>, %c:i1) {
  cf.cond_br %c, ^write, ^done
^write:
  frisk.fill %dst {value=3.0:f32} : memref<4xf32,3>
  cf.br ^done
^done:
  return
}
