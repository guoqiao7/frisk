#ifndef FRISK_SM90_COST_MODEL_H
#define FRISK_SM90_COST_MODEL_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
namespace mlir::frisk {
StringRef getSM90LegacyCostModelID();
FailureOr<CostEstimate> evaluateSM90LayoutCost(const LayoutConstraintGraph &graph,
    const CandidateAssignment &assignment,
    LayoutCostObserver *observer = nullptr, LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);
} // namespace mlir::frisk
#endif
