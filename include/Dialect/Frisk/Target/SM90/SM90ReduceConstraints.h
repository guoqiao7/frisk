#ifndef FRISK_TARGET_SM90_REDUCECONSTRAINTS_H
#define FRISK_TARGET_SM90_REDUCECONSTRAINTS_H
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
namespace mlir::frisk {
FailureOr<Attribute> buildSM90ReductionContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, Attribute source, Attribute result);
LayoutProof verifySM90ReductionContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, Attribute source, Attribute result,
    Attribute binding);
} // namespace mlir::frisk
#endif
