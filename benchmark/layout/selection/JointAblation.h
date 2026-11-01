#ifndef FRISK_JOINT_ABLATION_H
#define FRISK_JOINT_ABLATION_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "llvm/Support/JSON.h"
namespace mlir::frisk::selection {
FailureOr<LayoutSolution> solveAblation(LayoutConstraintGraph &,LayoutTarget &,
    SelectionMode,SolverOptions,llvm::json::Object &record);
}
#endif
