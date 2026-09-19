#ifndef FRISK_ANALYSIS_INSTRUCTIONLAYOUTCONSTRAINTS_H
#define FRISK_ANALYSIS_INSTRUCTIONLAYOUTCONSTRAINTS_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"

namespace mlir::frisk {
/// Select only from frozen domains; a tuple supports all roles simultaneously.
const InstructionLayoutTuple *findInstructionSupport(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &partialAssignment);
LogicalResult prepareInstructionTuples(LayoutConstraintGraph &graph,
                                       LayoutTarget &target, Location loc);
} // namespace mlir::frisk
#endif
