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
} // namespace mlir::frisk
#endif
