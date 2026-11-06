#ifndef FRISK_ANALYSIS_MMALAYOUTCONSTRAINTS_H
#define FRISK_ANALYSIS_MMALAYOUTCONSTRAINTS_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
namespace mlir::frisk {
LogicalResult collectMmaLayoutConstraints(Operation *root,
    LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder);
} // namespace mlir::frisk
#endif
