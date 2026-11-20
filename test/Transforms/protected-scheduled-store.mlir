// RUN: frisk-opt %s --frisk-layout-pipeline -verify-each -o %t.base
// RUN: python3 %S/Inputs/check-paired-mainloop.py %t.base frisk-opt
// RUN: frisk-opt %t.base --frisk-protected-store-layout=fixed-anchors=true -verify-each -o %t.overlap
// RUN: FileCheck %s --check-prefix=OVERLAP < %t.overlap
// RUN: sed 's/copy_overlap = true/copy_overlap = false/g' %t.base | frisk-opt --frisk-protected-store-layout=fixed-anchors=true -verify-each -o %t.serial
// RUN: FileCheck %s --check-prefix=SERIAL < %t.serial
// RUN: frisk-opt %t.serial --frisk-protected-store-layout=fixed-anchors=true -o %t.again
// RUN: diff %t.serial %t.again
// RUN: sed 's/sm_count = 114/sm_count = 0/g' %t.base > %t.invalid
// RUN: not frisk-opt %t.invalid --frisk-protected-store-layout=fixed-anchors=true 2>&1 | FileCheck %s --check-prefix=INVALID
// OVERLAP: frisk.mma
// OVERLAP: frisk.mma
// OVERLAP-NOT: frisk.convert_layout
// OVERLAP: frisk.tile_store
// OVERLAP-SAME: frisk.scheduled_store_score = array<i64: 704512, 352256, 923, 1580, 635024, 734632>
// OVERLAP-SAME: frisk.store_score = array<i64: 1024, 512, 264, 0>
// SERIAL: frisk.convert_layout
// SERIAL-SAME: frisk.protected_store_route
// SERIAL: frisk.tile_store
// SERIAL-SAME: frisk.store_score = array<i64: 1024, 512, 264, 1>
// INVALID: invalid frisk.h100_store_schedule

// Three query CTAs per head. Only one row of the final 64-row tile is valid.
#schedule = {rows = 129 : i64, instances = 8 : i64, sm_count = 114 : i64,
             mma_issues = 8 : i64, mma_waits = 2 : i64, copy_overlap = true}
module attributes {frisk.target="sm_90a"} {
func.func @tile(%oldM: tensor<64xf32>, %oldL: tensor<64xf32>, %oldO: tensor<64x64xf32>) {
%q0 = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%q = frisk.layout_view %q0 : memref<64x64xf16,3> -> memref<64x64xf16,3>
%k0 = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%k = frisk.layout_view %k0 : memref<64x64xf16,3> -> memref<64x64xf16,3>
%z = arith.constant dense<0.0> : tensor<64x64xf32>
%s = "frisk.mma"(%q,%k,%z) {m=64:i64,n=64:i64,k=64:i64,frisk.execution_threads=128:i64,trans_b=true} : (memref<64x64xf16,3>,memref<64x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
%scale = arith.constant dense<0.125> : tensor<64x64xf32>
%x = arith.mulf %s, %scale : tensor<64x64xf32>
%mx = frisk.reduce_tensor %x {kind="max",dim=1:i64,frisk.execution_threads=128:i64} : tensor<64x64xf32> -> tensor<64xf32>
%newM = arith.maximumf %mx, %oldM : tensor<64xf32>
%diff = arith.subf %oldM, %newM : tensor<64xf32>
%alpha = math.exp %diff : tensor<64xf32>
%empty = tensor.empty() : tensor<64x64xf32>
%bm = linalg.broadcast ins(%newM : tensor<64xf32>) outs(%empty : tensor<64x64xf32>) dimensions = [1]
%dx = arith.subf %x, %bm : tensor<64x64xf32>
%e = math.exp %dx : tensor<64x64xf32>
%sum = frisk.reduce_tensor %e {kind="sum",dim=1:i64,frisk.execution_threads=128:i64} : tensor<64x64xf32> -> tensor<64xf32>
%p = arith.truncf %e : tensor<64x64xf32> to tensor<64x64xf16>
%v0 = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%v = frisk.layout_view %v0 : memref<64x64xf16,3> -> memref<64x64xf16,3>
%oldScaled = arith.mulf %oldL, %alpha : tensor<64xf32>
%newL = arith.addf %oldScaled, %sum : tensor<64xf32>
%ba = linalg.broadcast ins(%alpha : tensor<64xf32>) outs(%empty : tensor<64x64xf32>) dimensions = [1]
%z2 = arith.mulf %oldO, %ba : tensor<64x64xf32>
%out = "frisk.mma"(%p,%v,%z2) {m=64:i64,n=64:i64,k=64:i64,frisk.execution_threads=128:i64} : (tensor<64x64xf16>,memref<64x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
%storage = memref.alloc() {alignment=128:i64} : memref<64x64xf32,1>
%view = frisk.layout_view %storage {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[16384,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<64x64xf32,1> -> memref<64x64xf32,1>
%bl = linalg.broadcast ins(%newL : tensor<64xf32>) outs(%empty : tensor<64x64xf32>) dimensions = [1]
%norm = arith.divf %out, %bl : tensor<64x64xf32>
frisk.tile_store %norm, %view {frisk.h100_store_schedule = #schedule} : tensor<64x64xf32>, memref<64x64xf32,1>
return
}
}
