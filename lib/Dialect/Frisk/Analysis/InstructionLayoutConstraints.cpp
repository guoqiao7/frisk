#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Diagnostics.h"
#include <functional>

namespace mlir::frisk {
const InstructionLayoutTuple *findInstructionSupport(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &assignment) {
  if (!constraint.instruction) return nullptr;
  for (const auto &tuple : constraint.instruction->tuples) {
    if (tuple.encodings.size() != constraint.vars.size() || !tuple.binding ||
        (constraint.instruction->binding &&
         tuple.binding != constraint.instruction->binding)) continue;
    DenseMap<LayoutVarID, Attribute> chosen;
    bool valid = true;
    for (auto [index, id] : llvm::enumerate(constraint.vars)) {
      Attribute value = tuple.encodings[index];
      auto previous = chosen.try_emplace(id, value);
      if ((!previous.second && previous.first->second != value) ||
          (assignment.count(id) && assignment.lookup(id) != value) ||
          llvm::none_of(graph.getVariable(id).candidates,
                        [&](const auto &c) { return c.value == value; })) {
        valid = false;
        break;
      }
    }
    if (valid) return &tuple;
  }
  return nullptr;
}

LogicalResult prepareInstructionTuples(LayoutConstraintGraph &graph,
                                       LayoutTarget &target, Location loc) {
  for (auto &constraint : graph.getConstraints()) {
    if (constraint.kind != ConstraintKind::InstructionContract) continue;
    if (!constraint.instruction || constraint.vars.size() != 4)
      return emitError(loc) << "unsupported instruction contract";
    auto &instruction = *constraint.instruction;
    instruction.tuples.clear();
    for (auto id : constraint.vars)
      if (graph.getVariable(id).candidates.size() > 4)
        return emitError(loc) << "instruction proof budget exceeded: candidate domain is too large";
    SmallVector<Attribute> encodings;
    DenseMap<LayoutVarID, Attribute> chosen;
    std::optional<LayoutProof> firstFailure;
    std::function<void(unsigned)> visit = [&](unsigned role) {
      if (role == constraint.vars.size()) {
        ++graph.getCandidatePreparationStatistics().instructionCombinations;
        FailureOr<Attribute> binding = instruction.binding
            ? FailureOr<Attribute>(instruction.binding)
            : target.buildInstructionContract(graph, constraint, encodings);
        if (failed(binding)) return;
        auto proof = target.verifyInstructionContract(graph, constraint, encodings, *binding);
        if (proof.status == ProofStatus::Proven)
          instruction.tuples.push_back({encodings, *binding});
        else if (!firstFailure)
          firstFailure = std::move(proof);
        return;
      }
      auto id = constraint.vars[role];
      if (auto prior = chosen.lookup(id)) {
        encodings.push_back(prior);
        visit(role + 1);
        encodings.pop_back();
        return;
      }
      for (const auto &candidate : graph.getVariable(id).candidates) {
        chosen[id] = candidate.value;
        encodings.push_back(candidate.value);
        visit(role + 1);
        encodings.pop_back();
        chosen.erase(id);
      }
    };
    // Rejected alternatives are not user errors. Report one joint failure if
    // there is no supported tuple, with pure-proof details for explicit plans.
    {
      ScopedDiagnosticHandler quiet(loc.getContext(),
                                     [](Diagnostic &) { return success(); });
      visit(0);
    }
    if (instruction.tuples.empty()) {
      auto diagnostic = emitError(instruction.source ? instruction.source->getLoc() : loc);
      diagnostic << "mma-joint-contract: no proven instruction tuple in frozen candidate domains";
      if (firstFailure) {
        diagnostic << "; " << (firstFailure->status == ProofStatus::Unknown
                                 ? "unknown proof: " : "counterexample: ") << firstFailure->reason;
        if (!firstFailure->counterexample.empty()) {
          std::string coordinates;
          llvm::raw_string_ostream stream(coordinates);
          llvm::interleaveComma(firstFailure->counterexample, stream);
          diagnostic << "; coordinate [" << coordinates << "]";
        }
      } else {
        diagnostic << "; unknown proof: no supported instruction plan could be "
                      "constructed in the bounded target decoder; this is not "
                      "a proof of hardware impossibility";
      }
      return failure();
    }
  }
  return success();
}
} // namespace mlir::frisk
