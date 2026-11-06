#ifndef FRISK_LEGACY_IMPORT_SEMANTICS_H
#define FRISK_LEGACY_IMPORT_SEMANTICS_H

#include <optional>
#include <string>

#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class Operation;

namespace frisk {

/// Returns an explanatory error for an invalid legacy Gemm contract.
/// Effective A/B shapes account for the transpose flags.
std::optional<std::string> getLegacyGemmValidationError(
    Type aType, Type bType, Type cType, bool transA, bool transB, int64_t m,
    int64_t n, int64_t k);

/// Requires tensor_v1 on `op` or the nearest declaring function, Frisk kernel,
/// or module. A malformed/unknown nearer declaration never falls back outward.
LogicalResult verifyLegacyTensorV1Semantics(Operation *op);

} // namespace frisk
} // namespace mlir

#endif // FRISK_LEGACY_IMPORT_SEMANTICS_H
