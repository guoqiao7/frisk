#ifndef FRISK_TRANSFORMS_PRESERVESTORAGECONTRACTS_H
#define FRISK_TRANSFORMS_PRESERVESTORAGECONTRACTS_H
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
namespace mlir::frisk {
/// Capture actual explicit whole-root views at their original scope. Call on a
/// transaction clone: failure may leave partially inserted declarations.
LogicalResult preserveStorageContracts(Operation *scope);
}
#endif
