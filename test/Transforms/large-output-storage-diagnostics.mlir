// RUN: frisk-opt %s -split-input-file -verify-diagnostics
func.func @check(%arg: memref<256x512xf32, 1>) {
  // expected-error@+1 {{same byte/bit address as [0, 511]; coordinate [1, 0]}}
    %3 = frisk.layout_view %arg {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> (d0 * 2044 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x512xf32, 1> -> memref<256x512xf32, 1>
  return
}
// -----
func.func @check(%arg: memref<256x512xf32, 1>) {
  // expected-error@+1 {{last element exceeds the declared address extent or root interval; coordinate [255, 511]}}
    %3 = frisk.layout_view %arg {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 4 + 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x512xf32, 1> -> memref<256x512xf32, 1>
  return
}
// -----
func.func @check(%arg: memref<256x512xf32, 1>) {
  // expected-error@+1 {{element overlaps the element at [0, 0]; coordinate [0, 1]}}
    %3 = frisk.layout_view %arg {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> (d0 * 2048 + d1 * 2, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x512xf32, 1> -> memref<256x512xf32, 1>
  return
}
// -----
func.func @check(%arg: memref<256x512xf32, 1>) {
  // expected-error@+1 {{wide storage proof unknown: unsupported wide strided storage map; dense interval enumeration limit is 65536}}
    %3 = frisk.layout_view %arg {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [256, 512], outputs = ["byte_offset", "bit_offset"], output_extents = [524288, 8], map = affine_map<(d0, d1) -> ((d0 mod 2) * 2048 + d1 * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 128, vector_granularity = 1>} : memref<256x512xf32, 1> -> memref<256x512xf32, 1>
  return
}
