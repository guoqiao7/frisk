// RUN: frisk-opt %s -split-input-file -verify-diagnostics -o /dev/null
func.func @legacy_auto(%a:memref<64x16xf16,3>,%b:memref<16x64xf16,3>,%c:memref<64x64xf32>) {
  // expected-error @+1 {{Auto policy requires frisk.mma}}
  frisk.gemm (%a,%b,%c) {transA=false,transB=false,M=64:i64,N=64:i64,K=16:i64,policy=#frisk<gemm_warp_policy Auto>,clear_accum=true} : memref<64x16xf16,3>,memref<16x64xf16,3>,memref<64x64xf32>
  return
}
