#ifndef FRISK_IR_CONTIGUOUSSTORAGELAYOUT_H
#define FRISK_IR_CONTIGUOUSSTORAGELAYOUT_H
#include "mlir/IR/BuiltinTypes.h"
#include "Dialect/Frisk/Analysis/LayoutCommon.h"
#include <cstdint>
#include <optional>
namespace mlir::frisk {
class StorageLayoutAttr;
/// Exact rank-two Global f32 row-major function. This symbolic proof is not an
/// enumerated footprint: interval(i,j)=[32*(i*N+j),32*(i*N+j+1)).
struct ContiguousIdentityStorage {
  int64_t rows, columns;
  uint64_t elements, bytes;
};
/// Canonical row-major Global f32 address function; the caller checks root
/// alignment and view identity when using it as physical address evidence.
std::optional<ContiguousIdentityStorage>
matchRowMajorF32Storage(StorageLayoutAttr layout, MemRefType type);
/// Unsupported expressions/metadata deliberately retain the original proof.
std::optional<ContiguousIdentityStorage>
matchContiguousIdentityStorage(StorageLayoutAttr layout, MemRefType type);
inline bool hasWideStorageDomain(ArrayRef<int64_t> shape) {
  return shape.size() == 2 && shape[0] > 0 && shape[1] > 0 &&
         shape[0] > 65536 / shape[1];
}
/// Prove all element intervals of a large positive rank-two strided map
/// are disjoint and contained in this half-open bit interval.
LayoutProof proveWideStridedStorageRange(StorageLayoutAttr layout, MemRefType type,
                                  uint64_t lowerBit, uint64_t upperBit);
} // namespace mlir::frisk
#endif
