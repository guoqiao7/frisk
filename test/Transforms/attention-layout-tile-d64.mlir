// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each -o %t.selected
// RUN: FileCheck %s --input-file=%t.selected
// RUN: frisk-opt %t.selected -frisk-export-execution=output=%t.json -o /dev/null
// RUN: FileCheck %s --check-prefix=EXPORT --input-file=%t.json

// The D64 copy endpoints exceed four proposals before hard root checks. The
// complete legal join fits the existing proof budget and yields a SW128 tile.
// CHECK-COUNT-3: frisk.copy {{.*}}vector_bytes = 16 : i64
// CHECK: frisk.mma
// CHECK-SAME: atom = array<i64: 64, 64, 16>
// CHECK-SAME: form = "ss"
// CHECK-SAME: repeats = array<i64: 1, 1, 4>
// CHECK: frisk.mma
// CHECK-SAME: atom = array<i64: 64, 64, 16>
// CHECK-SAME: form = "rs"
// CHECK-SAME: repeats = array<i64: 1, 1, 4>
// EXPORT: "form": "ss"
// EXPORT: "form": "rs"

#input_root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(2*(64*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=16,vector_granularity=1>
#o_root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(4*(64*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=16,vector_granularity=1>
module attributes {frisk.target = "sm_90a"} {
  "frisk.kernel"() <{sym_name="attention_q64_full_kv",function_type=(memref<64x64xf16,1>,memref<64x64xf16,1>,memref<64x64xf16,1>,memref<64x64xf32,1>)->()}> ({
  ^bb0(%q:memref<64x64xf16,1>, %k:memref<64x64xf16,1>, %v:memref<64x64xf16,1>, %o:memref<64x64xf32,1>):
    frisk.storage_contract %q {layout=#input_root} : memref<64x64xf16,1>
    frisk.storage_contract %k {layout=#input_root} : memref<64x64xf16,1>
    frisk.storage_contract %v {layout=#input_root} : memref<64x64xf16,1>
    frisk.storage_contract %o {layout=#o_root} : memref<64x64xf32,1>
    %qs = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
    %ks = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
    %vs = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %qg = frisk.layout_view %q : memref<64x64xf16,1> -> memref<64x64xf16,1>
      %qv = frisk.layout_view %qs : memref<64x64xf16,3> -> memref<64x64xf16,3>
      %kg = frisk.layout_view %k : memref<64x64xf16,1> -> memref<64x64xf16,1>
      %kv = frisk.layout_view %ks : memref<64x64xf16,3> -> memref<64x64xf16,3>
      %vg = frisk.layout_view %v : memref<64x64xf16,1> -> memref<64x64xf16,1>
      %vv = frisk.layout_view %vs : memref<64x64xf16,3> -> memref<64x64xf16,3>
      frisk.copy %qg[], %qv[] {frisk.vector_bytes=16:i64,frisk.writer_policy="first_owner",frisk.copy_contract={version=1:i64,path="vector",target="sm_90a",vector_bytes=16:i64,valid_shape=array<i64:64,64>,guarded=false,issuer="execution-map",completion="synchronous"},srcExtents=array<i64:64,64>,dstExtents=array<i64:64,64>} : memref<64x64xf16,1>, memref<64x64xf16,3>
      frisk.copy %kg[], %kv[] {frisk.vector_bytes=16:i64,frisk.writer_policy="first_owner",frisk.copy_contract={version=1:i64,path="vector",target="sm_90a",vector_bytes=16:i64,valid_shape=array<i64:64,64>,guarded=false,issuer="execution-map",completion="synchronous"},srcExtents=array<i64:64,64>,dstExtents=array<i64:64,64>} : memref<64x64xf16,1>, memref<64x64xf16,3>
      frisk.copy %vg[], %vv[] {frisk.vector_bytes=16:i64,frisk.writer_policy="first_owner",frisk.copy_contract={version=1:i64,path="vector",target="sm_90a",vector_bytes=16:i64,valid_shape=array<i64:64,64>,guarded=false,issuer="execution-map",completion="synchronous"},srcExtents=array<i64:64,64>,dstExtents=array<i64:64,64>} : memref<64x64xf16,1>, memref<64x64xf16,3>
      %zero = arith.constant dense<0.0> : tensor<64x64xf32>
      %scores = "frisk.mma"(%qv,%kv,%zero) {m=64:i64,n=64:i64,k=64:i64,trans_b=true} : (memref<64x64xf16,3>,memref<64x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
      %scale = arith.constant dense<0x3E000000> : tensor<64x64xf32>
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
      %ozero = arith.constant dense<0.0> : tensor<64x64xf32>
      %result = "frisk.mma"(%p,%vv,%ozero) {m=64:i64,n=64:i64,k=64:i64} : (tensor<64x64xf16>,memref<64x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
      %ov = frisk.layout_view %o : memref<64x64xf32,1> -> memref<64x64xf32,1>
      frisk.tile_store %result, %ov : tensor<64x64xf32>, memref<64x64xf32,1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
