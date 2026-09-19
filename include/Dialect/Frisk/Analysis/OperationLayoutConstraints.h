#ifndef FRISK_ANALYSIS_OPERATIONLAYOUTCONSTRAINTS_H
#define FRISK_ANALYSIS_OPERATIONLAYOUTCONSTRAINTS_H

#include "Dialect/Frisk/Analysis/LayoutSolver.h"

namespace mlir::frisk {
LogicalResult collectOperationLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder);
/// Adds the environment to existing Distributed definitions/use slots.
LogicalResult collectParallelResourceConstraints(LayoutConstraintGraph &graph);
} // namespace mlir::frisk
#endif
