// Four global roots are Q, K, V and O. The launch maps each query tile to a CTA.
#map = affine_map<(d0, d1) -> ((d0 mod 8) * 128 + (d0 floordiv 8) * 1024 + (d1 mod 64) * 2 + (d1 floordiv 64) * 32768 + (((((d1 mod 64) * 2) floordiv 16) mod 2 + (d0 mod 8 + (d0 floordiv 8) * 8 + ((d1 mod 64) * 2) floordiv 128) mod 2) mod 2 - (((d1 mod 64) * 2) floordiv 16) mod 2) * 16 + (((((d1 mod 64) * 2) floordiv 32) mod 2 + (((d0 mod 8) * 128 + (d0 floordiv 8) * 1024 + (d1 mod 64) * 2) floordiv 256) mod 2) mod 2 - (((d1 mod 64) * 2) floordiv 32) mod 2) * 32 + (((((d1 mod 64) * 2) floordiv 64) mod 2 + (((d0 mod 8) * 128 + (d0 floordiv 8) * 1024 + (d1 mod 64) * 2) floordiv 512) mod 2) mod 2 - (((d1 mod 64) * 2) floordiv 64) mod 2) * 64, 0)>
#map1 = affine_map<(d0, d1) -> ((d0 mod 8) * 128 + (d0 floordiv 8) * 1024 + (d1 mod 64) * 2 + (d1 floordiv 64) * 8192 + (((((d1 mod 64) * 2) floordiv 16) mod 2 + (d0 mod 8 + (d0 floordiv 8) * 8 + ((d1 mod 64) * 2) floordiv 128) mod 2) mod 2 - (((d1 mod 64) * 2) floordiv 16) mod 2) * 16 + (((((d1 mod 64) * 2) floordiv 32) mod 2 + (((d0 mod 8) * 128 + (d0 floordiv 8) * 1024 + (d1 mod 64) * 2) floordiv 256) mod 2) mod 2 - (((d1 mod 64) * 2) floordiv 32) mod 2) * 32 + (((((d1 mod 64) * 2) floordiv 64) mod 2 + (((d0 mod 8) * 128 + (d0 floordiv 8) * 1024 + (d1 mod 64) * 2) floordiv 512) mod 2) mod 2 - (((d1 mod 64) * 2) floordiv 64) mod 2) * 64, 0)>
#map2 = affine_map<(d0, d1) -> ((d1 mod 16) * 2 + (d1 floordiv 16) * 256 + (d0 mod 8) * 32 + (d0 floordiv 8) * 1024 + (((((d1 mod 16) * 2) floordiv 16) mod 2 + (((d1 mod 16) * 2 + (d1 floordiv 16) * 256 + (d0 mod 8) * 32) floordiv 128) mod 2) mod 2 - (((d1 mod 16) * 2) floordiv 16) mod 2) * 16, 0)>
#map3 = affine_map<(d0, d1) -> (d0 * 256 + d1 * 4, 0)>
#E64=#frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp","warp_group"],input_bits=[4,5,2,1],outputs=["dim0","dim1"],output_bits=[6,6],matrix=dense<[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]>:tensor<12x12xi1>>,topology=[16,32,4,2,1],replication=1>
#E256=#frisk.distributed<map=#frisk.bit_linear<inputs=["register","lane","warp","warp_group"],input_bits=[6,5,2,1],outputs=["dim0","dim1"],output_bits=[8,6],matrix=dense<[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]>:tensor<14x14xi1>>,topology=[64,32,4,2,1],replication=1>
module attributes {frisk.target="sm_90a", frisk.online_launch = {rows=1024:i64, heads=8:i64, kv_heads=8:i64, causal=false}} {
"frisk.kernel"() <{sym_name="online_q256",function_type=(memref<256x64xf16,1>,memref<64x64xf16,1>,memref<64x64xf16,1>,memref<256x64xf32,1>)->()}> ({
^bb0(%Q:memref<256x64xf16,1>,%K0:memref<64x64xf16,1>,%V0:memref<64x64xf16,1>,%O:memref<256x64xf32,1>):
frisk.storage_contract %Q {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[256,64],outputs=["byte_offset","bit_offset"],output_extents=[32768,8],map=affine_map<(i,j)->(i*128+j*2,0)>>,memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>} : memref<256x64xf16,1>
frisk.storage_contract %K0 {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(i*128+j*2,0)>>,memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>} : memref<64x64xf16,1>
frisk.storage_contract %V0 {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,64],outputs=["byte_offset","bit_offset"],output_extents=[8192,8],map=affine_map<(i,j)->(i*128+j*2,0)>>,memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>} : memref<64x64xf16,1>
frisk.storage_contract %O {layout=#frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[256,64],outputs=["byte_offset","bit_offset"],output_extents=[65536,8],map=affine_map<(i,j)->(i*256+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>} : memref<256x64xf32,1>
%qs = memref.alloc() {alignment=1024:i64} : memref<256x64xf16,3>
%ks = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
%vs = memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
"frisk.parallel"() <{ranges=array<i64:1>,threads=256:i64}> ({
^bb0(%i:index):
%gQ = frisk.layout_view %Q : memref<256x64xf16,1> -> memref<256x64xf16,1>
%gK0 = frisk.layout_view %K0 : memref<64x64xf16,1> -> memref<64x64xf16,1>
%gV0 = frisk.layout_view %V0 : memref<64x64xf16,1> -> memref<64x64xf16,1>
%gO = frisk.layout_view %O : memref<256x64xf32,1> -> memref<256x64xf32,1>
%sqs = frisk.layout_view %qs {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 64], outputs = ["byte_offset", "bit_offset"], output_extents = [32768, 8], map = #map>, memory_space = #frisk<memory_space Shared>, alignment = 1024, vector_granularity = 1>} : memref<256x64xf16,3> -> memref<256x64xf16,3>
%sks = frisk.layout_view %ks {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [64, 64], outputs = ["byte_offset", "bit_offset"], output_extents = [8192, 8], map = #map1>, memory_space = #frisk<memory_space Shared>, alignment = 1024, vector_granularity = 1>} : memref<64x64xf16,3> -> memref<64x64xf16,3>
%svs = frisk.layout_view %vs {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [64, 64], outputs = ["byte_offset", "bit_offset"], output_extents = [8192, 8], map = #map2>, memory_space = #frisk<memory_space Shared>, alignment = 1024, vector_granularity = 1>} : memref<64x64xf16,3> -> memref<64x64xf16,3>
frisk.copy %gQ[], %sqs[] {frisk.execution_threads=256:i64,frisk.execution_layout=#E256,frisk.writer_policy="first_owner",frisk.vector_bytes=1:i64,srcExtents=array<i64:256,64>,dstExtents=array<i64:256,64>} : memref<256x64xf16,1>, memref<256x64xf16,3>
%zero = arith.constant dense<0.0> : tensor<256x64xf32>
%zeroR = arith.constant dense<0.0> : tensor<256xf32>
%scale = arith.constant dense<0.125> : tensor<256x64xf32>
%empty = tensor.empty() : tensor<256x64xf32>
%neg = arith.constant dense<0xFF800000> : tensor<256xf32>
%idx0 = arith.constant 0 : index
%idx1 = arith.constant 1 : index
%bound = arith.constant 16 : index
%mfinal,%lfinal,%ofinal = scf.for %iv = %idx0 to %bound step %idx1 iter_args(%carryM=%neg,%carryL=%zeroR,%carryO=%zero) -> (tensor<256xf32>,tensor<256xf32>,tensor<256x64xf32>) {
frisk.copy %gK0[], %sks[] {frisk.execution_threads=256:i64,frisk.execution_layout=#E64,frisk.writer_policy="first_owner",frisk.vector_bytes=1:i64,srcExtents=array<i64:64,64>,dstExtents=array<i64:64,64>} : memref<64x64xf16,1>, memref<64x64xf16,3>
frisk.copy %gV0[], %svs[] {frisk.execution_threads=256:i64,frisk.execution_layout=#E64,frisk.writer_policy="first_owner",frisk.vector_bytes=1:i64,srcExtents=array<i64:64,64>,dstExtents=array<i64:64,64>} : memref<64x64xf16,1>, memref<64x64xf16,3>
%s1 = "frisk.mma"(%sqs,%sks,%zero) {m=256:i64,n=64:i64,k=64:i64,frisk.execution_threads=256:i64,policy=#frisk<gemm_warp_policy Square>,trans_b=true} : (memref<256x64xf16,3>,memref<64x64xf16,3>,tensor<256x64xf32>) -> tensor<256x64xf32>
%x1 = arith.mulf %s1, %scale : tensor<256x64xf32>
%mx1 = frisk.reduce_tensor %x1 {kind="max",dim=1:i64,frisk.execution_threads=256:i64} : tensor<256x64xf32> -> tensor<256xf32>
%newM1 = arith.maximumf %carryM, %mx1 : tensor<256xf32>
%diff1 = arith.subf %carryM, %newM1 : tensor<256xf32>
%alpha1 = math.exp %diff1 : tensor<256xf32>
%ba1 = linalg.broadcast ins(%alpha1 : tensor<256xf32>) outs(%empty : tensor<256x64xf32>) dimensions = [1]
%acc1 = arith.mulf %carryO, %ba1 : tensor<256x64xf32>
%lscaled1 = arith.mulf %carryL, %alpha1 : tensor<256xf32>
%bm1 = linalg.broadcast ins(%newM1 : tensor<256xf32>) outs(%empty : tensor<256x64xf32>) dimensions = [1]
%dx1 = arith.subf %x1, %bm1 : tensor<256x64xf32>
%exp1 = math.exp %dx1 : tensor<256x64xf32>
%sum1 = frisk.reduce_tensor %exp1 {kind="sum",dim=1:i64,frisk.execution_threads=256:i64} : tensor<256x64xf32> -> tensor<256xf32>
%p1 = arith.truncf %exp1 : tensor<256x64xf32> to tensor<256x64xf16>
%newL1 = arith.addf %lscaled1, %sum1 : tensor<256xf32>
%out1 = "frisk.mma"(%p1,%svs,%acc1) {m=256:i64,n=64:i64,k=64:i64,frisk.execution_threads=256:i64,policy=#frisk<gemm_warp_policy Square>} : (tensor<256x64xf16>,memref<64x64xf16,3>,tensor<256x64xf32>) -> tensor<256x64xf32>
scf.yield %newM1,%newL1,%out1 : tensor<256xf32>,tensor<256xf32>,tensor<256x64xf32>
}
%denom3 = linalg.broadcast ins(%lfinal : tensor<256xf32>) outs(%empty : tensor<256x64xf32>) dimensions = [1]
%out = arith.divf %ofinal, %denom3 : tensor<256x64xf32>
frisk.tile_store %out, %gO : tensor<256x64xf32>, memref<256x64xf32,1>
frisk.end
}) : () -> ()
frisk.end
}) : () -> ()
}

