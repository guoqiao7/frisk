#include "Dialect/Frisk/IR/LegacyImportSemantics.h"

#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {

std::optional<std::string> getLegacyGemmValidationError(
    Type aType, Type bType, Type cType, bool transA, bool transB, int64_t m,
    int64_t n, int64_t k) {
  auto a = dyn_cast<MemRefType>(aType);
  auto b = dyn_cast<MemRefType>(bType);
  auto c = dyn_cast<MemRefType>(cType);
  if (!a || !b || !c)
    return "legacy-gemm-shape: A, B, and C must be memrefs";
  if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2)
    return "legacy-gemm-shape: A, B, and C must have rank 2";
  if (!a.hasStaticShape() || !b.hasStaticShape() || !c.hasStaticShape())
    return "legacy-gemm-shape: A, B, and C must have static shapes";
  if (m <= 0 || n <= 0 || k <= 0)
    return "legacy-gemm-shape: M, N, and K must be positive";

  Type inputType = a.getElementType();
  Type bElementType = b.getElementType();
  Type accumulatorType = c.getElementType();
  if (inputType != bElementType)
    return "legacy-gemm-type: A and B must have the same element type";
  bool sameType = accumulatorType == inputType;
  bool lowPrecisionF32 =
      (inputType.isF16() || inputType.isBF16()) && accumulatorType.isF32();
  if (!sameType && !lowPrecisionF32)
    return "legacy-gemm-type: C must match A/B or be f32 for f16/bf16 inputs";

  int64_t aM = a.getDimSize(transA ? 1 : 0);
  int64_t aK = a.getDimSize(transA ? 0 : 1);
  int64_t bK = b.getDimSize(transB ? 1 : 0);
  int64_t bN = b.getDimSize(transB ? 0 : 1);
  if (aM != m || aK != k || bK != k || bN != n)
    return "legacy-gemm-shape: effective A/B shapes must match M/N/K";
  if (c.getDimSize(0) != m || c.getDimSize(1) != n)
    return "legacy-gemm-shape: C shape must be MxN";
  return std::nullopt;
}

LogicalResult verifyLegacyTensorV1Semantics(Operation *op) {
  constexpr StringLiteral attrName = "frisk.legacy_semantics";
  Operation *candidate = op;
  while (candidate) {
    bool declarationScope = candidate == op || isa<func::FuncOp, KernelOp, ModuleOp>(candidate);
    if (declarationScope) {
      if (Attribute attr = candidate->getAttr(attrName)) {
        auto value = dyn_cast<StringAttr>(attr);
        if (value && value.getValue() == "tensor_v1")
          return success();
        return op->emitOpError(
            "legacy-math-contract: explicit tensor_v1 semantics required");
      }
    }
    candidate = candidate->getParentOp();
  }
  return op->emitOpError(
      "legacy-math-contract: explicit tensor_v1 semantics required");
}

} // namespace mlir::frisk
