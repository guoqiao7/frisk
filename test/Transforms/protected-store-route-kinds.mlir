// RUN: python3 %S/Inputs/check-protected-store-routing.py consumers frisk-opt %s
#mapS = #frisk.bit_linear<inputs=["register","lane","warp"],input_bits=[2,5,2],outputs=["dim0","dim1"],output_bits=[2,7],matrix=dense<[[1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]]> : tensor<9x9xi1>>
#S = #frisk.distributed<map=#mapS,topology=[4,32,4,1,1],replication=1>
#a = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[4,128],outputs=["byte_offset","bit_offset"],output_extents=[2048,8],map=affine_map<(i,j)->(4*(128*i+j),0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target = "sm_90a"} {
  "frisk.kernel"() <{sym_name="rmsnorm",function_type=(memref<4x128xf32,1>,memref<4x128xf32,1>,memref<4x128xf32,1>,memref<4x128xf32,1>)->()}> ({
  ^bb0(%a:memref<4x128xf32,1>, %k:memref<4x128xf32,1>, %v:memref<4x128xf32,1>, %y:memref<4x128xf32,1>):
    frisk.storage_contract %a {layout=#a} : memref<4x128xf32,1>
    frisk.storage_contract %k {layout=#a} : memref<4x128xf32,1>
    frisk.storage_contract %v {layout=#a} : memref<4x128xf32,1>
    frisk.storage_contract %y {layout=#a} : memref<4x128xf32,1>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %ag = frisk.layout_view %a {layout=#a} : memref<4x128xf32,1> -> memref<4x128xf32,1>
      %x = frisk.tile_load %ag : memref<4x128xf32,1> -> tensor<4x128xf32, #S>
      %og = frisk.layout_view %y {layout=#a} : memref<4x128xf32,1> -> memref<4x128xf32,1>
      frisk.tile_store %x, %og : tensor<4x128xf32, #S>, memref<4x128xf32,1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
