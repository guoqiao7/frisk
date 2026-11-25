#pragma once
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::frisk {
class DistributedEncodingAttr;
using RoutePrice =
    llvm::function_ref<FailureOr<int64_t>(RankedTensorType, RankedTensorType)>;

using StoreMap = llvm::function_ref<DistributedEncodingAttr(
    RankedTensorType, DistributedEncodingAttr)>;

// Solve on a transaction clone whose instruction domains contain ONLY
// certified instruction-local minima. Materialize the selected SSA uses once.
FailureOr<DictionaryAttr> solveCertifiedMultiAnchor(
    ModuleOp module, const CertifiedInstructionDomains &domains,
    RoutePrice price, StoreMap storeMap, unsigned freeBits, unsigned timeoutMs,
    unsigned oracleLimit, bool scoreReduction, StringRef globalAccess, bool subgroupCandidates,
    bool minimizeConversions, const DenseMap<Attribute, int64_t> &sliceUnits, StringRef nativeDelivery);
} // namespace mlir::frisk
