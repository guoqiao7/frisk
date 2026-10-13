// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each > %t.once
// RUN: FileCheck %s < %t.once
// RUN: frisk-opt %t.once -verify-each -o /dev/null
// RUN: frisk-opt %t.once --frisk-layout-pipeline -verify-each > %t.twice
// RUN: diff %t.once %t.twice

// Complete, initially unencoded static Attention inference, not GPU lowering.
// CHECK: arith.constant dense<2.500000e-01>
// CHECK: frisk.kernel @attention
// CHECK: frisk.parallel
// CHECK: frisk.copy
// CHECK: frisk.copy
// CHECK: frisk.copy
// CHECK: frisk.mma
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: form = "ss"
// CHECK-SAME: trans_b = true
// CHECK: arith.mulf
// CHECK: frisk.reduce_tensor
// CHECK-SAME: dim = 1 : i64
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: frisk.reduction_contract = #frisk.reduction_contract
// CHECK-SAME: kind = "max"
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [1]
// CHECK: arith.subf
// CHECK: math.exp
// CHECK: frisk.reduce_tensor
// CHECK-SAME: dim = 1 : i64
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: frisk.reduction_contract = #frisk.reduction_contract
// CHECK-SAME: kind = "sum"
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [1]
// CHECK: arith.divf
// CHECK: arith.truncf
// CHECK: frisk.mma
// CHECK-SAME: frisk.execution_threads = 128 : i64
// CHECK-SAME: form = "rs"
// CHECK: frisk.tile_store
module attributes {frisk.target = "sm_90a"} {
  "frisk.kernel"() <{sym_name="attention",
      function_type=(memref<64x16xf16,1>,memref<64x16xf16,1>,memref<64x16xf16,1>,memref<64x16xf32,1>)->()}> ({
  ^bb0(%q:memref<64x16xf16,1>, %k:memref<64x16xf16,1>, %v:memref<64x16xf16,1>, %o:memref<64x16xf32,1>):
    %qs = memref.alloc() {alignment=1024:i64} : memref<64x16xf16,3>
    %ks = memref.alloc() {alignment=1024:i64} : memref<64x16xf16,3>
    %vs = memref.alloc() {alignment=1024:i64} : memref<64x16xf16,3>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %qg = frisk.layout_view %q : memref<64x16xf16,1> -> memref<64x16xf16,1>
      %kg = frisk.layout_view %k : memref<64x16xf16,1> -> memref<64x16xf16,1>
      %vg = frisk.layout_view %v : memref<64x16xf16,1> -> memref<64x16xf16,1>
      %qv = frisk.layout_view %qs : memref<64x16xf16,3> -> memref<64x16xf16,3>
      %kv = frisk.layout_view %ks : memref<64x16xf16,3> -> memref<64x16xf16,3>
      %vv = frisk.layout_view %vs : memref<64x16xf16,3> -> memref<64x16xf16,3>
      frisk.copy %qg[], %qv[] {srcExtents=array<i64:64,16>,dstExtents=array<i64:64,16>}
        : memref<64x16xf16,1>, memref<64x16xf16,3>
      frisk.copy %kg[], %kv[] {srcExtents=array<i64:64,16>,dstExtents=array<i64:64,16>}
        : memref<64x16xf16,1>, memref<64x16xf16,3>
      frisk.copy %vg[], %vv[] {srcExtents=array<i64:64,16>,dstExtents=array<i64:64,16>}
        : memref<64x16xf16,1>, memref<64x16xf16,3>
      %zero = arith.constant dense<0.0> : tensor<64x64xf32>
      %scores = "frisk.mma"(%qv,%kv,%zero) {m=64:i64,n=64:i64,k=16:i64,trans_b=true}
        : (memref<64x16xf16,3>,memref<64x16xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
      %scale = arith.constant dense<0.25> : tensor<64x64xf32>
      %scaled = arith.mulf %scores, %scale : tensor<64x64xf32>
      %max = frisk.reduce_tensor %scaled {kind="max",dim=1:i64} : tensor<64x64xf32> -> tensor<64xf32>
      %e0 = tensor.empty() : tensor<64x64xf32>
      %bm = linalg.broadcast ins(%max : tensor<64xf32>) outs(%e0 : tensor<64x64xf32>) dimensions = [1]
      %centered = arith.subf %scaled, %bm : tensor<64x64xf32>
      %exp = math.exp %centered : tensor<64x64xf32>
      %sum = frisk.reduce_tensor %exp {kind="sum",dim=1:i64} : tensor<64x64xf32> -> tensor<64xf32>
      %e1 = tensor.empty() : tensor<64x64xf32>
      %bs = linalg.broadcast ins(%sum : tensor<64xf32>) outs(%e1 : tensor<64x64xf32>) dimensions = [1]
      %normalized = arith.divf %exp, %bs : tensor<64x64xf32>
      %p = arith.truncf %normalized : tensor<64x64xf32> to tensor<64x64xf16>
      %ozero = arith.constant dense<0.0> : tensor<64x16xf32>
      %result = "frisk.mma"(%p,%vv,%ozero) {m=64:i64,n=16:i64,k=64:i64}
        : (tensor<64x64xf16>,memref<64x16xf16,3>,tensor<64x16xf32>) -> tensor<64x16xf32>
      %ov = frisk.layout_view %o : memref<64x16xf32,1> -> memref<64x16xf32,1>
      frisk.tile_store %result, %ov : tensor<64x16xf32>, memref<64x16xf32,1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
