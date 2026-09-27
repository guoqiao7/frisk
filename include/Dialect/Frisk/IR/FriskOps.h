#ifndef FRISK_OPS_H
#define FRISK_OPS_H

#include "Dialect/Frisk/IR/FriskDialect.h"

namespace mlir::frisk {
/// Basic format checks only; semantic execution proofs live in Analysis.
LogicalResult verifyOperationExecutionAttributes(Operation *op);
bool isSupportedExecutionThreadCount(int64_t threads);
} // namespace mlir::frisk

#endif // FRISK_OPS_H
