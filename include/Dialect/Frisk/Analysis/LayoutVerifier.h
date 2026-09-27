#ifndef FRISK_ANALYSIS_LAYOUTVERIFIER_H
#define FRISK_ANALYSIS_LAYOUTVERIFIER_H

#include "Dialect/Frisk/Analysis/LayoutSolver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::frisk {

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target, Location loc);

LogicalResult materializeLayouts(Operation *root,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution);

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTVERIFIER_H
