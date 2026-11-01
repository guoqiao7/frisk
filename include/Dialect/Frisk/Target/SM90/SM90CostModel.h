#ifndef FRISK_SM90_COST_MODEL_H
#define FRISK_SM90_COST_MODEL_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
#include "llvm/Support/Error.h"
#include <array>
namespace mlir::frisk {
StringRef getSM90LegacyCostModelID();
FailureOr<CostEstimate> evaluateSM90LayoutCost(const LayoutConstraintGraph &graph,
    const CandidateAssignment &assignment,
    LayoutCostObserver *observer = nullptr, LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);

/// Explicit calibrated, additive model. Coefficients are nonnegative integer
/// score ticks per feature unit, not CUDA cycles or a latency prediction.
/// The intercept used in fitting timings is deliberately outside this objective.
constexpr unsigned kSM90CostFeatureCount = 15;
struct SM90CostProfile {
  std::array<uint64_t, kSM90CostFeatureCount> coefficients{};
  std::string provenance;
};
ArrayRef<StringRef> getSM90CostFeatureNames();
llvm::Expected<SM90CostProfile> parseSM90CostProfile(StringRef json);
llvm::Expected<SM90CostProfile> loadSM90CostProfileFile(StringRef path);
std::string serializeSM90CostProfile(const SM90CostProfile &profile);
std::string getSM90CostProfileSHA256(const SM90CostProfile &profile);
std::string getSM90CalibratedCostModelID(const SM90CostProfile &profile,
    LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);
LogicalResult verifySM90CostProfile(const SM90CostProfile &profile);
FailureOr<CostEstimate> evaluateSM90CalibratedLayoutCost(
    const LayoutConstraintGraph &, const CandidateAssignment &,
    const SM90CostProfile &, LayoutCostObserver *observer = nullptr,
    LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);
FailureOr<CostEstimate> adjustSM90CalibratedConversionCost(
    const CostEstimate &, const LayoutConversionEdge &, const SM90CostProfile &);
FailureOr<CostEstimate> adjustSM90CalibratedCloneCost(
    const CostEstimate &, const LayoutVar &, Attribute, Operation *, Operation *,
    const SM90CostProfile &, LayoutBoundaryFrequencyMode);
} // namespace mlir::frisk
#endif
