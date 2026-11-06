#ifndef FRISK_ANALYSIS_INSTRUCTIONLAYOUTCONSTRAINTS_H
#define FRISK_ANALYSIS_INSTRUCTIONLAYOUTCONSTRAINTS_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"

namespace mlir::frisk {
class LayoutTrace;
/// Select only from frozen domains; a tuple supports all roles simultaneously.
const InstructionLayoutTuple *findInstructionSupport(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &partialAssignment);
LogicalResult prepareInstructionTuples(LayoutConstraintGraph &graph,
                                       LayoutTarget &target, Location loc,
                                       LayoutTrace *trace = nullptr);
} // namespace mlir::frisk
#endif
