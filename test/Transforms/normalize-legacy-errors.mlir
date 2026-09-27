// RUN: frisk-opt %s -frisk-normalize-layout-ir -split-input-file -verify-diagnostics
func.func @uninitialized(%dst:memref<4xf32,3>) {
  %x = memref.alloc() : memref<4xf32>
  // expected-error@+1 {{legacy-fragment-uninitialized}}
  "frisk.copy"(%x, %dst) <{srcMap=affine_map<()->()>, dstMap=affine_map<()->()>, srcExtents=array<i64:4>, dstExtents=array<i64:4>}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<4xf32>,memref<4xf32,3>) -> ()
  return
}
// -----
// expected-error@+1 {{legacy-layout-attribute}}
module attributes {test.debug_contract = [{nested = #frisk.layout<[4], affine_map<(d)->(d)>>}]} {}
// -----
func.func @missing_contract(%a:memref<64x16xf16,3>, %b:memref<16x64xf16,3>) {
  %c = memref.alloc() : memref<64x64xf32>
  // expected-error@+1 {{legacy-math-contract: explicit tensor_v1 semantics required}}
  frisk.gemm (%a,%b,%c) {transA=false,transB=false,M=64:i64,N=64:i64,K=16:i64,policy=#frisk<gemm_warp_policy Square>,clear_accum=true} : memref<64x16xf16,3>,memref<16x64xf16,3>,memref<64x64xf32>
  return
}
// -----
func.func @custom_local_capture() {
  %x=memref.alloc():memref<4xf32>
  "frisk.block"() <{ranges=array<i64:4>}> ({
  ^bb0(%i:index):
    // expected-error@+1 {{legacy local fragment escapes supported normalization boundary}}
    frisk.fill %x {value=2.0:f32}:memref<4xf32>
    "frisk.end"() : () -> ()
  }) : () -> ()
  return
}
// -----
// expected-error@+1 {{general CFG is not a sequential fragment lifetime}}
func.func @cfg_local(%c:i1) {
  %x=memref.alloc():memref<4xf32>
  cf.cond_br %c, ^write, ^done
^write:
  frisk.fill %x {value=2.0:f32}:memref<4xf32>
  cf.br ^done
^done:
  return
}
// -----
func.func @kernel_capture() {
  %x=memref.alloc():memref<4xf32>
  frisk.fill %x {value=1.0:f32}:memref<4xf32>
  "frisk.kernel"() <{sym_name="nested",function_type=()->()}> ({
    // expected-error@+1 {{Local root crosses a function or Kernel boundary}}
    frisk.fill %x {value=2.0:f32}:memref<4xf32>
    "frisk.end"() : () -> ()
  }) : () -> ()
  return
}
