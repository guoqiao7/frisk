// RUN: frisk-opt %s -frisk-normalize-layout-ir -frisk-normalize-layout-ir | FileCheck %s
// CHECK-LABEL: func.func @lifecycle
// CHECK-NOT: memref.alloc
// CHECK: arith.constant dense<-0.000000e+00> : tensor<4xf32>
// CHECK: scf.for {{.*}} iter_args
// CHECK: scf.if {{.*}} -> (tensor<4xf32>)
// CHECK: scf.yield {{.*}} : tensor<4xf32>
// CHECK: scf.yield {{.*}} : tensor<4xf32>
// CHECK: scf.yield {{.*}} : tensor<4xf32>
// CHECK-NOT: frisk.fill
// CHECK-NOT: memref.dealloc
func.func @lifecycle(%n:index, %cond:i1) {
  %lb = arith.constant 0:index
  %step = arith.constant 1:index
  %x = memref.alloc() : memref<4xf32>
  frisk.fill %x {value=-0.0:f32} : memref<4xf32>
  scf.for %i = %lb to %n step %step {
    scf.if %cond {
      frisk.fill %x {value=2.0:f32} : memref<4xf32>
    } else {
      frisk.fill %x {value=3.0:f32} : memref<4xf32>
    }
  }
  memref.dealloc %x : memref<4xf32>
  return
}
