// REQUIRES: frisk-z3
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-global-access=native integer-oracle-limit=1000" -verify-each | FileCheck %s --check-prefix=SERVICE
// RUN: frisk-opt %s --frisk-protected-store-layout="anchor-solver=integer integer-free-bits=0 integer-global-access=native integer-pointwise-min-conversions=true integer-oracle-limit=1000" -verify-each | FileCheck %s --check-prefix=DIRECT

// Same open tensor and access domains. The exhaustive oracle verifies both
// lexicographic objectives. Only the compatibility candidate reuses the load.
// SERVICE: access_units = 272
// SERVICE-SAME: best_without_conversions = 392
// SERVICE-SAME: conversion_units = 72
// SERVICE-SAME: distinct_conversions = 1
// SERVICE: frisk.convert_layout
// DIRECT: access_units = 392
// DIRECT-SAME: best_without_conversions = 392
// DIRECT-SAME: conversion_units = 0
// DIRECT-SAME: distinct_conversions = 0
// DIRECT-SAME: minimize_conversions = true
// DIRECT: %[[VAL:[^ ]+]] = frisk.tile_load
// DIRECT-NEXT: frisk.tile_store %[[VAL]],
#row = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,16],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(i*64+j*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
#col = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[64,16],outputs=["byte_offset","bit_offset"],output_extents=[4096,8],map=affine_map<(i,j)->(j*256+i*4,0)>>,memory_space=#frisk<memory_space Global>,alignment=128,vector_granularity=1>
module attributes {frisk.target="sm_90a"} {
  "frisk.kernel"() <{sym_name="transpose",function_type=(memref<64x16xf32, 1>,memref<64x16xf32, 1>)->()}> ({
  ^bb0(%in:memref<64x16xf32, 1>,%out:memref<64x16xf32, 1>):
    frisk.storage_contract %in {layout=#row} : memref<64x16xf32, 1>
    frisk.storage_contract %out {layout=#col} : memref<64x16xf32, 1>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %a = frisk.layout_view %in {layout=#row} : memref<64x16xf32, 1> -> memref<64x16xf32, 1>
      %b = frisk.layout_view %out {layout=#col} : memref<64x16xf32, 1> -> memref<64x16xf32, 1>
      %x = frisk.tile_load %a : memref<64x16xf32, 1> -> tensor<64x16xf32>
      frisk.tile_store %x, %b : tensor<64x16xf32>, memref<64x16xf32, 1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
}
