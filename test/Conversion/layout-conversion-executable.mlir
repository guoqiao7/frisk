// RUN: frisk-opt %s -frisk-test-lower-layout-conversions='executable' | FileCheck %s --implicit-check-not=unrealized_conversion_cast --implicit-check-not=frisk.tile --implicit-check-not=frisk.convert_layout --implicit-check-not='tensor<'
// RUN: sed 's/known_block_size=array<i32:32,1,1>/known_block_size=array<i32:64,1,1>/' %s | not frisk-opt -frisk-test-lower-layout-conversions='executable' 2>&1 | FileCheck %s --check-prefix=TOPOLOGY
// TOPOLOGY: requires known_block_size matching topology
// RUN: sed 's/memref<32xf32>/memref<32xf32, #gpu.address_space<workgroup>>/g' %s | not frisk-opt -frisk-test-lower-layout-conversions='executable' 2>&1 | FileCheck %s --check-prefix=SPACE
// SPACE: executable ABI requires static global rank-one memrefs
#a = #frisk.bit_linear<inputs=["lane"], input_bits=[5], outputs=["m"], output_bits=[5], matrix=dense<[[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]> : tensor<5x5xi1>>
#b = #frisk.bit_linear<inputs=["lane"], input_bits=[5], outputs=["m"], output_bits=[5], matrix=dense<[[0,1,0,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]> : tensor<5x5xi1>>
#A = #frisk.distributed<map=#a, topology=[1,32,1,1,1], replication=1>
#B = #frisk.distributed<map=#b, topology=[1,32,1,1,1], replication=1>
gpu.module @kernels {
  gpu.func @shuffle32_f32(%input:memref<32xf32>, %output:memref<32xf32>) kernel attributes {known_block_size=array<i32:32,1,1>} {
    %x = frisk.tile_load %input : memref<32xf32> -> tensor<32xf32,#A>
    %y = frisk.convert_layout %x : tensor<32xf32,#A> -> tensor<32xf32,#B>
    frisk.tile_store %y, %output : tensor<32xf32,#B>, memref<32xf32>
    gpu.return
  }
}
// CHECK: gpu.func @shuffle32_f32
// CHECK: memref.load
// CHECK: gpu.shuffle
// CHECK: memref.store
// CHECK: gpu.return
