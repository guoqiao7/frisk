#ifndef FRISK_LAYOUT_EXPERIMENT_POLICY_H
#define FRISK_LAYOUT_EXPERIMENT_POLICY_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
#include "llvm/Support/JSON.h"
namespace mlir::frisk {
enum class ConversionPolicy { Independent, SharedOnly, Full };
enum class SelectionMode { Joint, ProducerFirst };
/// IDs belong to an already finalized, frozen graph; experimental entry
/// points reject unfinalized graphs instead of remapping scope identities.
struct SelectionScope {
  SmallVector<LayoutVarID> producers;
  SmallVector<LayoutConstraintID> producerInstructions;
  bool fullGraph = true;
  /// Include the declared producers' own operations, accesses and local
  /// storage auxiliaries. Necessary incoming repairs are included; future consumer boundaries are excluded.
  /// Stage two pins only producer definitions/geometry, not these auxiliaries.
  bool informedLocal = false;
};
struct SharedSearchBudget {
  uint64_t limit=65536, refinementLimit=256;
  uint64_t used=0, refinementUsed=0;
  uint64_t stageLimit=65536, stageRefinementLimit=256;
  uint64_t relationshipProbes=0, completeScores=0, refinementProbes=0;
  bool consume(bool refinement=false) {
    if(used>=limit || used>=stageLimit || (refinement &&
       (refinementUsed>=refinementLimit || refinementUsed>=stageRefinementLimit))) return false;
    ++used; if(refinement) ++refinementUsed; return true;
  }
};
struct ExperimentPolicy {
  ConversionPolicy conversion=ConversionPolicy::Full;
  SelectionScope scope;
};
/// Conservative support preflight for the whole frozen candidate domain.
/// Informed local O rejects potential incoming nonidentity repairs because a
/// shared/rematerialized charge may depend on future consumers. Empty = supported.
std::string experimentPolicyUnsupportedReason(
    const LayoutConstraintGraph &, const ExperimentPolicy &);
SmallVector<LayoutVarID> experimentInstructionDependencies(
    const LayoutConstraintGraph &, const LayoutConstraint &, const LayoutTarget &);
FailureOr<CostEstimate> scoreExperimentAssignment(const LayoutConstraintGraph &,
    const CandidateAssignment &, LayoutTarget &, const ExperimentPolicy &,
    LayoutCostObserver * = nullptr, llvm::json::Array *contributions = nullptr);
}
#endif
