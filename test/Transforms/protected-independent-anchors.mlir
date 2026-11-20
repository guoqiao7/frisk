// RUN: frisk-opt %s --frisk-protected-store-layout="max-combinations=1" -verify-each | FileCheck %s
// RUN: not frisk-opt %s --frisk-protected-store-layout="max-combinations=0" 2>&1 | FileCheck %s --check-prefix=LIMIT
// CHECK: frisk.protected_functions = {first = [
// CHECK-SAME: second = [
// CHECK: func.func @first
// CHECK: frisk.mma
// CHECK: func.func @second
// CHECK: frisk.mma
// LIMIT: local-optimum combinations exceed max-combinations
module attributes {frisk.target="sm_90a"} {
func.func @first() {
%q0 = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%q = frisk.layout_view %q0 : memref<64x64xf16,3> -> memref<64x64xf16,3>
%k0 = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%k = frisk.layout_view %k0 : memref<64x64xf16,3> -> memref<64x64xf16,3>
%z = arith.constant dense<0.0> : tensor<64x64xf32>
%s = "frisk.mma"(%q,%k,%z) {m=64:i64,n=64:i64,k=64:i64,frisk.execution_threads=128:i64,trans_b=true, policy=#frisk<gemm_warp_policy Auto>} : (memref<64x64xf16,3>,memref<64x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
%storage = memref.alloc() {alignment=128:i64} : memref<64x64xf32,1>
%view = frisk.layout_view %storage {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<64x64xf32,1> -> memref<64x64xf32,1>
frisk.tile_store %s, %view : tensor<64x64xf32>, memref<64x64xf32,1>
return
}
func.func @second() {
%p = arith.constant dense<1.0> : tensor<64x64xf16>
%v0 = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%v = frisk.layout_view %v0 : memref<64x64xf16,3> -> memref<64x64xf16,3>
%z2 = arith.constant dense<0.0> : tensor<64x64xf32>
%out = "frisk.mma"(%p,%v,%z2) {m=64:i64,n=64:i64,k=64:i64,frisk.execution_threads=128:i64, policy=#frisk<gemm_warp_policy Auto>} : (tensor<64x64xf16>,memref<64x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
%storage = memref.alloc() {alignment=128:i64} : memref<64x64xf32,1>
%view = frisk.layout_view %storage {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<64x64xf32,1> -> memref<64x64xf32,1>
frisk.tile_store %out, %view : tensor<64x64xf32>, memref<64x64xf32,1>
return
}
}
