#ifndef FRISK_ATTRIBUTES_H
#define FRISK_ATTRIBUTES_H

#include <cstdint>
#include <optional>
#include <string>

#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"

#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/IR/FriskLayoutInterfaces.h"

namespace mlir::frisk {
class FriskDialect;
} // namespace mlir::frisk

#define GET_ATTRDEF_CLASSES
#include "Dialect/Frisk/IR/FriskAttributes.h.inc"

namespace mlir::frisk {
// Emit a compact description of the layout/fragment that mirrors tilelang's
// LayoutNode::DebugOutput()/FragmentNode::DebugOutput() helpers.
void printLayoutDebug(LayoutAttr layout, llvm::raw_ostream &os);

// Convenience wrapper that returns the debug string instead of writing it to
// an ostream.
std::string layoutDebugString(LayoutAttr layout);

LayoutProof checkInjective(BitLinearLayoutMapAttr map);
LayoutProof checkSurjective(BitLinearLayoutMapAttr map);

/// Evaluate one coordinate using the same evaluator as coverage/injectivity
/// proofs. Callers must supply a non-null, verified canonical map and in-domain
/// coordinates; this wrapper does not validate the map or enforce its declared
/// input extents. Unsupported map kinds fail closed.
FailureOr<SmallVector<int64_t>> evaluateLayoutMap(
    Attribute map, ArrayRef<int64_t> coordinates);

std::optional<attr::MemorySpace> getFriskMemorySpace(MemRefType type);
FailureOr<uint64_t> getMemRefStaticCapacityBytes(MemRefType type);
FailureOr<uint64_t> getStorageFootprintBytes(StorageLayoutAttr layout,
                                             MemRefType type);

FailureOr<BitLinearLayoutMapAttr>
composeBitLinear(BitLinearLayoutMapAttr lhs, BitLinearLayoutMapAttr rhs);
} // namespace mlir::frisk

#endif // FRISK_ATTRIBUTES_H
