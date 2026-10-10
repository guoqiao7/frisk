#ifndef FRISK_ANALYSIS_LAYOUTCONVERSIONCHOICE_H
#define FRISK_ANALYSIS_LAYOUTCONVERSIONCHOICE_H

#include "Dialect/Frisk/Analysis/LayoutSolver.h"

namespace mlir::frisk {
struct LayoutConversionChoices {
  SmallVector<LayoutConversionEdge> edges;
  CostEstimate cost;
};

/// A conservative whitelist AND a MemoryEffectOpInterface proof. This only
/// establishes operation eligibility, not slice, dominance or layout legality.
bool isRematerializable(Operation *op);

/// Minimize independent same-source/target/block boundary groups for a complete
/// (possibly component-local) assignment. Never modifies IR or candidate domains.
FailureOr<LayoutConversionChoices> chooseLayoutConversions(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment,
    LayoutTarget &target);

bool sameLayoutConversionPlan(const LayoutConversionEdge &a,
                              const LayoutConversionEdge &b);
LogicalResult verifyLayoutConversionChoices(const LayoutConstraintGraph &graph,
    const LayoutSolution &solution, LayoutTarget &target, Location loc);
} // namespace mlir::frisk
#endif
