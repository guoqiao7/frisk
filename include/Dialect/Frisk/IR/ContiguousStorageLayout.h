#ifndef FRISK_IR_CONTIGUOUSSTORAGELAYOUT_H
#define FRISK_IR_CONTIGUOUSSTORAGELAYOUT_H
#include "mlir/IR/BuiltinTypes.h"
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
/// Unsupported expressions/metadata deliberately retain the original proof.
std::optional<ContiguousIdentityStorage>
matchContiguousIdentityStorage(StorageLayoutAttr layout, MemRefType type);
/// Prove all element intervals of a large positive rank-two strided map
/// are disjoint and contained in this half-open bit interval.
bool proveWideStridedStorageRange(StorageLayoutAttr layout, MemRefType type,
                                  uint64_t lowerBit, uint64_t upperBit);
} // namespace mlir::frisk
#endif
