#ifndef FRISK_ANALYSIS_REDUCTIONLAYOUTCONSTRAINTS_H
#define FRISK_ANALYSIS_REDUCTIONLAYOUTCONSTRAINTS_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
namespace mlir::frisk {
LogicalResult collectReductionLayoutConstraints(Operation *root,
    LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder);
LogicalResult prepareReductionPairs(LayoutConstraintGraph &graph,
                                    LayoutTarget &target, Location loc);
const ReductionLayoutPair *findReductionSupport(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &partialAssignment);
} // namespace mlir::frisk
#endif
