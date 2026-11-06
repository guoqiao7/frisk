#ifndef FRISK_SM90_COST_MODEL_H
#define FRISK_SM90_COST_MODEL_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
#include "Dialect/Frisk/Analysis/ShuffleNetwork.h"
#include "llvm/Support/Error.h"
#include <array>
namespace mlir::frisk {
StringRef getSM90LegacyCostModelID();
StringRef getSM90DemandCostModelID();
struct ExperimentPolicy;
FailureOr<CostEstimate> evaluateSM90LayoutCost(const LayoutConstraintGraph &graph,
    const CandidateAssignment &assignment,
    LayoutCostObserver *observer = nullptr, LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);

/// Explicit calibrated, additive model. Coefficients are nonnegative integer
/// score ticks per feature unit, not CUDA cycles or a latency prediction.
/// The intercept used in fitting timings is deliberately outside this objective.
constexpr unsigned kSM90CostFeatureCount = 15;
constexpr unsigned kSM90DemandCostFeatureCount = 17;
constexpr unsigned kSM90OrderedDemandCostFeatureCount = 18;
constexpr unsigned kSM90SurvivingCostFeatureCount = 20;
struct SM90CostProfile {
  std::array<uint64_t, kSM90CostFeatureCount> coefficients{};
  std::string provenance;
  bool executionDemand = false;
  std::array<uint64_t,2> demandBankCoefficients{};
  bool executionOrder = false;
  uint64_t loadIssueGroupCoefficient = 0;
  bool survivingDemand = false;
  // Explicit migrated engineering proxy; never silently reinterpret v4/v6.
  bool networkProxy = false;
  shuffle_network::Objective networkObjective = shuffle_network::Objective::MinShuffle;
  // Shared requests, Shuffle instructions, communication address/mux ALU.
  std::array<uint64_t,3> communicationCoefficients{};
};
ArrayRef<StringRef> getSM90CostFeatureNames();
ArrayRef<StringRef> getSM90DemandCostFeatureNames();
ArrayRef<StringRef> getSM90OrderedDemandCostFeatureNames();
ArrayRef<StringRef> getSM90SurvivingCostFeatureNames();
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
FailureOr<CostEstimate> evaluateSM90DemandLayoutCost(const LayoutConstraintGraph &,
    const CandidateAssignment &, const ExperimentPolicy &, const SM90CostProfile *profile = nullptr,
    LayoutCostObserver *observer = nullptr,
    LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);
FailureOr<CostEstimate> adjustSM90CalibratedConversionCost(
    const CostEstimate &, const LayoutConversionEdge &, const SM90CostProfile &);
FailureOr<CostEstimate> adjustSM90CalibratedCloneCost(
    const CostEstimate &, const LayoutVar &, Attribute, Operation *, Operation *,
    const SM90CostProfile &, LayoutBoundaryFrequencyMode);
} // namespace mlir::frisk
#endif
