// RUN: frisk-opt %s --frisk-layout-pipeline --frisk-export-execution="output=%t.json" -o /dev/null
// RUN: FileCheck %s < %t.json

// A checked static f32 RMSNorm retains ordered sum, epsilon addition, and rsqrt.
// CHECK: "name": "frisk.reduce_tensor"
// CHECK: "name": "arith.addf"
// CHECK: "name": "math.rsqrt"
// CHECK: "name": "frisk.tile_store"

#a = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[4,256],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(4*(256*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=16,vector_granularity=1>
module attributes {frisk.target = "sm_90a"} {
  "frisk.kernel"() <{sym_name="rmsnorm",function_type=(memref<4x256xf32,1>,memref<4x256xf32,1>,memref<4x256xf32,1>,memref<4x256xf32,1>)->()}> ({
  ^bb0(%a:memref<4x256xf32,1>, %k:memref<4x256xf32,1>, %v:memref<4x256xf32,1>, %y:memref<4x256xf32,1>):
    frisk.storage_contract %a {layout=#a} : memref<4x256xf32,1>
    frisk.storage_contract %k {layout=#a} : memref<4x256xf32,1>
    frisk.storage_contract %v {layout=#a} : memref<4x256xf32,1>
    frisk.storage_contract %y {layout=#a} : memref<4x256xf32,1>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %ag = frisk.layout_view %a : memref<4x256xf32,1> -> memref<4x256xf32,1>
      %x = frisk.tile_load %ag : memref<4x256xf32,1> -> tensor<4x256xf32>
      %inv = arith.constant dense<0.00390625> : tensor<4xf32>
      %sq = arith.mulf %x, %x : tensor<4x256xf32>
      %sum = frisk.reduce_tensor %sq {kind="sum",dim=1:i64} : tensor<4x256xf32> -> tensor<4xf32>
      %ms = arith.mulf %sum, %inv : tensor<4xf32>
      %eps = arith.constant dense<1.000000e-05> : tensor<4xf32>
      %ve = arith.addf %ms, %eps : tensor<4xf32>
      %invstd = math.rsqrt %ve : tensor<4xf32>
      %te = tensor.empty() : tensor<4x256xf32>
      %tb = linalg.broadcast ins(%invstd : tensor<4xf32>) outs(%te : tensor<4x256xf32>) dimensions = [1]
      %out = arith.mulf %x, %tb : tensor<4x256xf32>
      %og = frisk.layout_view %y : memref<4x256xf32,1> -> memref<4x256xf32,1>
      frisk.tile_store %out, %og : tensor<4x256xf32>, memref<4x256xf32,1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
