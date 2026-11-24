// RUN: frisk-opt %s -verify-each -o /dev/null
// RUN: sed "s/d0 \* 4096/d0 * 4/" %s | not frisk-opt 2>&1 | FileCheck %s
// CHECK: wide storage layout disproven: same byte/bit address as [0, 1]; coordinate [1, 0]
module {
  func.func @wide(%arg0: memref<128x1024xf32, 1>) {
    frisk.storage_contract %arg0 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [128, 1024], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> (d0 * 4096 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<128x1024xf32, 1>
    return
  }
}
