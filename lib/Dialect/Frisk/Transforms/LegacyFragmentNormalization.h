#ifndef FRISK_TRANSFORMS_LEGACYFRAGMENTNORMALIZATION_H
#define FRISK_TRANSFORMS_LEGACYFRAGMENTNORMALIZATION_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
namespace mlir::frisk {
/// Mutates a transaction clone only. Promotes complete Local lifetimes to SSA
/// and imports legacy mathematical operations under the tensor_v1 contract.
LogicalResult normalizeLegacyFragments(ModuleOp module);
/// Reject surviving import operations, Local legacy effects, obsolete layout
/// attributes (including nested attributes/types), and unknown Frisk contracts.
LogicalResult verifyNoLegacyLayoutIR(Operation *scope);
}
#endif
