#ifndef FRISK_ANALYSIS_LAYOUTCOSTMODEL_H
#define FRISK_ANALYSIS_LAYOUTCOSTMODEL_H

#include "Dialect/Frisk/Analysis/LayoutTarget.h"

namespace mlir::frisk {
CostEstimate addLayoutCosts(const CostEstimate &lhs, const CostEstimate &rhs);

/// Static proxies only: registers, replication and occupied Shared root spans.
/// Zero fields are unmodeled, not measured to be free. Accepts component-local
/// assignments and never enumerates candidates or changes IR.
FailureOr<CostEstimate>
evaluateStaticLayoutCost(const LayoutConstraintGraph &graph,
                         const CandidateAssignment &assignment);

/// cta-staging-upper-bound-v1: caller must first prove Convertible. This is
/// neither hardware cycles nor the scratch allocation of a lowering plan.
FailureOr<CostEstimate> evaluateLayoutConversionCost(
    const LayoutVar &sourceVar, Attribute source, Attribute target,
    LayoutConversionEdge &edge);
} // namespace mlir::frisk
#endif
