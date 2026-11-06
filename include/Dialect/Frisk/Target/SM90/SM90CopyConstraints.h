#ifndef FRISK_SM90_COPY_CONSTRAINTS_H
#define FRISK_SM90_COPY_CONSTRAINTS_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
namespace mlir::frisk {
LogicalResult prepareSM90CopyCandidates(LayoutConstraintGraph &graph);
InstructionContractBuildResult buildSM90CopyContractDetailed(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    ArrayRef<Attribute> encodings);
FailureOr<Attribute> buildSM90CopyContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings);
LayoutProof verifySM90CopyContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings, Attribute binding);
bool isSM90CopyPath(Attribute candidate);
} // namespace mlir::frisk
#endif
