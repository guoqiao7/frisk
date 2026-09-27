#ifndef FRISK_ANALYSIS_LAYOUTTARGET_H
#define FRISK_ANALYSIS_LAYOUTTARGET_H

#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::frisk {

struct CandidateAssignment {
  DenseMap<LayoutVarID, Attribute> values;
};

class LayoutTarget {
public:
  virtual ~LayoutTarget() = default;

  virtual void
  enumerateCandidates(const LayoutVar &var,
                      SmallVectorImpl<LayoutCandidate> &out) const = 0;
  virtual LogicalResult verifyCandidate(const LayoutVar &var,
                                        Attribute candidate,
                                        Location loc) const = 0;
  virtual FailureOr<CostVector>
  evaluate(const CandidateAssignment &assignment) const = 0;
  virtual FailureOr<CostEstimate>
  evaluate(const LayoutConstraintGraph &, const CandidateAssignment &assignment) const {
    auto cost = evaluate(assignment);
    if (failed(cost)) return failure();
    return CostEstimate{*cost, false};
  }
  // Costs must be nonnegative and separable by hard component. A partial
  // estimate must never exceed any complete extension. Component optima do
  // not imply an optimum of a saturated global sum (priority can collapse).
  virtual FailureOr<CostEstimate>
  lowerBound(const LayoutConstraintGraph &, const CandidateAssignment &) const {
    return CostEstimate{};
  }
  virtual StringRef costCoverage() const { return "target-defined"; }

  virtual LogicalResult
  prepareInstructionCandidates(LayoutConstraintGraph &graph) const {
    for (const auto &constraint : graph.getConstraints())
      if (constraint.kind == ConstraintKind::InstructionContract)
        return failure();
    return success();
  }
  virtual FailureOr<Attribute> buildInstructionContract(
      const LayoutConstraintGraph &, const LayoutConstraint &,
      ArrayRef<Attribute>) const { return failure(); }
  virtual LayoutProof verifyInstructionContract(
      const LayoutConstraintGraph &, const LayoutConstraint &,
      ArrayRef<Attribute>, Attribute) const {
    return {ProofStatus::Unknown, {}, "unsupported instruction contract"};
  }
  virtual FailureOr<Attribute> buildReductionContract(
      const LayoutConstraintGraph &, const LayoutConstraint &,
      Attribute, Attribute) const { return failure(); }
  virtual LayoutProof verifyReductionContract(
      const LayoutConstraintGraph &, const LayoutConstraint &,
      Attribute, Attribute, Attribute) const {
    return {ProofStatus::Unknown, {}, "unsupported reduction contract"};
  }
};

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTTARGET_H
