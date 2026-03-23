#ifndef FRISK_UTILS_LAYOUT_UTILS_H
#define FRISK_UTILS_LAYOUT_UTILS_H

#include <cstdint>
#include <optional>

#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::frisk {

struct CompressedReplicateInfo {
  AffineMapAttr mapAttr;
  int64_t replicateExtent = 1;
};

/// Estimate how many distinct values an affine map may produce from the
/// dimension at `placeholderPos` when that dimension ranges over
/// `[0, placeholderExtent)`. The analysis mirrors tilelang's
/// `CondenseReplicateVar` by collecting the used iterator splits of the target
/// dimension (via its `mod`/`floordiv` usage) and returning the product of the
/// participating split extents. A return value of 1 means the dimension does
/// not affect the map. Returns `std::nullopt` when the map references the
/// placeholder through unsupported affine constructs or when overflow occurs.
std::optional<int64_t>
computeUsedExtentForDim(AffineMapAttr mapAttr, unsigned placeholderPos,
                        int64_t placeholderExtent);

/// Compress the placeholder dimension of `mapAttr` the same way tilelang's
/// `CondenseReplicateVar` rewrites the replicate iterator: the participating
/// splits of `placeholderPos` are packed into a new contiguous replicate
/// symbol, while all non-placeholder dimensions are preserved.
///
/// This helper currently expects `placeholderPos` to be the final dimension of
/// the map, which matches the reduce remapping pipeline in Frisk.
std::optional<CompressedReplicateInfo>
compressReplicateDimInMap(OpBuilder &builder, AffineMapAttr mapAttr,
                          unsigned placeholderPos, int64_t placeholderExtent);

/// Infer a fragment-style forward index from `threadMapAttr` by flattening the
/// pieces of each logical dimension that are not consumed by the thread map.
/// This mirrors tilelang's `infer_fragment_index(DivideUnusedIterators(...))`.
std::optional<AffineMapAttr>
inferFragmentIndexFromThreadMap(OpBuilder &builder, AffineMapAttr threadMapAttr,
                                ArrayRef<int64_t> inputShape);

} // namespace mlir::frisk

#endif // FRISK_UTILS_LAYOUT_UTILS_H
