#ifndef FRISK_ANALYSIS_LAYOUTIRDIAGNOSTICS_H
#define FRISK_ANALYSIS_LAYOUTIRDIAGNOSTICS_H
#include "llvm/Support/JSON.h"
namespace mlir {
class Operation;
namespace frisk {
/// Pure static logical-element counts of actual arith/math IR operations.
llvm::json::Object collectLayoutIRArithmetic(Operation *operation);
/// Pure conservative liveness of actual tensor SSA and shared storage roots.
llvm::json::Object collectLayoutIRResources(Operation *operation);
} // namespace frisk
} // namespace mlir
#endif
