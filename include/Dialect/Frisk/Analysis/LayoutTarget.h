#ifndef FRISK_ANALYSIS_LAYOUTTARGET_H
#define FRISK_ANALYSIS_LAYOUTTARGET_H

#include "Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h"
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/Analysis/LayoutCostObserver.h"

#include "llvm/ADT/DenseMap.h"

namespace mlir::frisk {

struct CandidateAssignment {
  DenseMap<LayoutVarID, Attribute> values;
};

/// A checked construction result. A binding exists only for a Proven proof;
/// operand identifies the failing instruction role (0=source, 1=destination
/// for Copy). Reason codes describe rejection categories, not hardware limits.
struct InstructionContractBuildResult {
  Attribute binding;
  LayoutProof proof;
  std::string reasonCode;
  std::optional<unsigned> operand;
};

class LayoutTarget {
public:
  virtual ~LayoutTarget() = default;
  virtual LayoutBoundaryFrequencyMode boundaryFrequencyMode() const {
    return LayoutBoundaryFrequencyMode::Legacy;
  }

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
  /// Existing targets remain source-compatible; their actual aggregate is the
  /// fallback contribution. Targets with charge sites override this overload.
  virtual FailureOr<CostEstimate>
  evaluate(const LayoutConstraintGraph &graph,
           const CandidateAssignment &assignment,
           LayoutCostObserver *observer) const {
    auto cost = evaluate(graph, assignment);
    if (succeeded(cost) && observer)
      observer->recordCost("target-aggregate", 0, *cost);
    return cost;
  }
  // Costs must be nonnegative and separable by hard component. A partial
  // estimate must never exceed any complete extension. Component optima do
  // not imply an optimum of a saturated global sum (priority can collapse).
  virtual FailureOr<CostEstimate>
  lowerBound(const LayoutConstraintGraph &, const CandidateAssignment &) const {
    return CostEstimate{};
  }
  /// Re-score raw boundary proxies in the target's own objective before any
  /// conversion/rematerialization choice. Defaults preserve legacy targets.
  virtual FailureOr<CostEstimate> adjustConversionCost(
      const CostEstimate &proxy, const LayoutConversionEdge &) const {
    return proxy;
  }
  virtual FailureOr<CostEstimate> adjustCloneCost(
      const CostEstimate &proxy, const LayoutVar &, Attribute,
      Operation *clonedOperation, Operation *insertionPoint) const {
    return proxy;
  }
  virtual bool requiresRematerializationTrafficReduction() const { return true; }
  virtual StringRef costCoverage() const { return "target-defined"; }
  /// Complete model identity, independent of the JSON schema version.
  virtual StringRef costModelID() const { return "unversioned-target"; }

  // Proposals only: the engine retains every original hard requirement and
  // revalidates candidates. No IR or graph mutation is permitted here.
  virtual void enumerateRelaxedCandidates(const LayoutConstraintGraph &,
      const LayoutVar &, RelaxationKind,
      SmallVectorImpl<LayoutCandidate> &) const {}
  virtual StringRef relaxationCoverage(RelaxationKind) const {
    return "unsupported by target";
  }

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
  /// Geometry-conditioned fragment precheck. Unknown must not prune a role;
  /// the bounded complete builder remains authoritative for such targets.
  virtual LayoutProof precheckInstructionRole(const LayoutConstraintGraph &,
      const LayoutConstraint &, unsigned, Attribute, Attribute) const {
    return {ProofStatus::Unknown, {}, "target has no fragment precheck"};
  }
  virtual LayoutProof verifyInstructionContract(
      const LayoutConstraintGraph &, const LayoutConstraint &,
      ArrayRef<Attribute>, Attribute) const {
    return {ProofStatus::Unknown, {}, "unsupported instruction contract"};
  }
  /// Compatibility fallback for targets implementing the original interface.
  /// Detailed target builders return their existing proof without redoing it.
  virtual InstructionContractBuildResult buildInstructionContractDetailed(
      const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
      ArrayRef<Attribute> encodings) const {
    // An existing explicit plan has always used verification-only semantics:
    // legacy targets need not be able to generate a replacement for it.
    auto binding = constraint.instruction && constraint.instruction->binding
        ? FailureOr<Attribute>(constraint.instruction->binding)
        : buildInstructionContract(graph, constraint, encodings);
    if (failed(binding))
      return {{}, {ProofStatus::Unknown, {}, "bounded target decoder could not construct an instruction plan"},
              "instruction-decoder", {}};
    auto proof = verifyInstructionContract(graph, constraint, encodings, *binding);
    bool proven = proof.status == ProofStatus::Proven;
    return {proven ? *binding : Attribute{}, std::move(proof),
            proven ? "" : "instruction-contract", {}};
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
