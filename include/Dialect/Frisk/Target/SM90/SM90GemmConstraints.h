#ifndef FRISK_SM90_GEMM_CONSTRAINTS_H
#define FRISK_SM90_GEMM_CONSTRAINTS_H
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
namespace mlir::frisk {
FailureOr<DistributedEncodingAttr> buildSM90MmaFragment(MLIRContext *context,
    const SM90MmaGeometry &g, bool operandA, bool transA);
LogicalResult prepareSM90MmaCandidates(LayoutConstraintGraph &graph);
FailureOr<Attribute> buildSM90MmaContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings);
} // namespace mlir::frisk
#endif
