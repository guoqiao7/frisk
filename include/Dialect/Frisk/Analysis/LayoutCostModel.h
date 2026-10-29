#ifndef FRISK_ANALYSIS_LAYOUTCOSTMODEL_H
#define FRISK_ANALYSIS_LAYOUTCOSTMODEL_H

#include "Dialect/Frisk/Analysis/LayoutTarget.h"

namespace mlir::frisk {
/// Metadata only: accept a complete 1..64-byte ASCII model identifier, or
/// explicitly mark it unversioned. Never truncate an invalid identity.
StringRef normalizeLayoutCostModelID(StringRef id);

CostEstimate addLayoutCosts(const CostEstimate &lhs, const CostEstimate &rhs);

/// Counts enclosing scf.for/scf.while operations. This is a bounded dynamic
/// execution-frequency proxy, not a trip-count or cycle estimate.
uint64_t getLayoutLoopDepth(Operation *insertionPoint);

/// Weights only the dynamic conversion proxy by 2^loopDepth, saturating.
/// Static code-size and resource fields are intentionally unchanged.
CostEstimate weightLayoutBoundaryCost(CostEstimate cost, uint64_t loopDepth);

/// Static proxies only: registers, replication and occupied Shared root spans.
/// Zero fields are unmodeled, not measured to be free. Accepts component-local
/// assignments and never enumerates candidates or changes IR.
FailureOr<CostEstimate>
evaluateStaticLayoutCost(const LayoutConstraintGraph &graph,
                         const CandidateAssignment &assignment,
                         LayoutCostObserver *observer = nullptr);

/// Uses bounded physical-owner proofs to classify register, shuffle, or shared
/// exchange. cta-staging-upper-bound-v1 is the nonzero fallback when the proof
/// budget or supported topology is exceeded. This is neither hardware cycles
/// nor the scratch allocation of a lowering plan.
FailureOr<CostEstimate> evaluateLayoutConversionCost(
    const LayoutVar &sourceVar, Attribute source, Attribute target,
    LayoutConversionEdge &edge, LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);

/// One cheap cloned operation's output-carrier byte work and static resources.
/// Eligibility, hard contracts and slice/leaf selection are the caller's job.
FailureOr<CostEstimate> evaluateLayoutCloneCost(
    const LayoutVar &result, Attribute encoding, Operation *insertionPoint, LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);
} // namespace mlir::frisk
#endif
