#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"

#include "Dialect/Frisk/IR/FriskAttributes.h"

#include "llvm/ADT/STLExtras.h"

#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {

namespace {
Location getVariableLoc(const LayoutVar &var) {
  return var.anchor ? var.anchor->getLoc()
                    : UnknownLoc::get(var.shapedType.getContext());
}

} // namespace

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target, Location loc) {
  if (failed(graph.verifyInvariants(loc))) return failure();
  for (const auto &entry : solution.reductionBindings)
    if (entry.first >= graph.getConstraints().size() ||
        graph.getConstraint(entry.first).kind != ConstraintKind::ReductionLayout)
      return emitError(loc) << "reduction binding does not identify an authorized graph constraint";
  for (const auto &entry : solution.instructionBindings)
    if (entry.first >= graph.getConstraints().size() ||
        graph.getConstraint(entry.first).kind != ConstraintKind::InstructionContract)
      return emitError(loc) << "instruction binding does not identify an authorized graph constraint";
  llvm::SmallDenseSet<LayoutConstraintID> converted;
  for (const LayoutConversionEdge &edge : solution.conversions) {
    if (edge.constraint >= graph.getConstraints().size())
      return emitError(loc) << "conversion does not identify an authorized graph edge";
    const auto &constraint = graph.getConstraint(edge.constraint);
    if (constraint.kind != ConstraintKind::Convertible || constraint.existingConversion ||
        !edge.use || constraint.use != edge.use ||
        edge.resolution != EdgeResolutionKind::Convert ||
        !converted.insert(edge.constraint).second ||
        edge.sourceEncoding != solution.assignments.lookup(constraint.vars[0]) ||
        edge.targetEncoding != solution.assignments.lookup(constraint.vars[1]) ||
        layoutEncodingsEqual(edge.sourceEncoding, edge.targetEncoding))
      return emitError(loc) << "invalid, duplicate, or identity conversion in layout solution";
    const auto &source = graph.getVariable(constraint.vars[0]);
    if (!source.value || edge.use->get() != source.value)
      return emitError(loc) << "conversion SSA use no longer matches its producer";
  }
  for (const LayoutVar &var : graph.getVariables()) {
    auto found = solution.assignments.find(var.id);
    if (found == solution.assignments.end())
      return emitError(loc)
             << "unresolved storage layout for layout variable "
             << var.stableName;
    if (llvm::none_of(var.candidates, [&](const LayoutCandidate &candidate) {
          return candidate.value == found->second;
        }))
      return emitError(loc) << "solution for " << var.stableName
                            << " is outside its candidate domain";
    if (failed(target.verifyCandidate(var, found->second,
                                      getVariableLoc(var))))
      return failure();
    if (failed(verifyLayoutStorageCapacity(graph, var, found->second)))
      return failure();
  }
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (constraint.kind == ConstraintKind::ReductionLayout) {
      Attribute binding = solution.reductionBindings.lookup(constraint.id);
      if (!constraint.reduction || !binding ||
          (constraint.reduction->binding && binding != constraint.reduction->binding))
        return emitError(loc) << "missing or conflicting reduction binding";
      auto proof = target.verifyReductionContract(graph, constraint,
          solution.assignments.lookup(constraint.vars[0]),
          solution.assignments.lookup(constraint.vars[1]), binding);
      if (proof.status != ProofStatus::Proven)
        return emitError(loc) << "reduce-contract: "
            << (proof.status == ProofStatus::Unknown ? "unknown proof: " : "") << proof.reason;
      continue;
    }
    if (constraint.kind == ConstraintKind::InstructionContract) {
      Attribute binding = solution.instructionBindings.lookup(constraint.id);
      if (!constraint.instruction || !binding ||
          (constraint.instruction->binding && binding != constraint.instruction->binding))
        return emitError(loc) << "missing or conflicting instruction binding";
      SmallVector<Attribute> encodings;
      for (auto id : constraint.vars) encodings.push_back(solution.assignments.lookup(id));
      auto proof = target.verifyInstructionContract(graph, constraint, encodings, binding);
      if (proof.status != ProofStatus::Proven)
        return emitError(loc) << "mma-joint-contract: "
            << (proof.status == ProofStatus::Unknown ? "unknown proof: " : "") << proof.reason;
      continue;
    }
    if (constraint.kind == ConstraintKind::Convertible && !constraint.existingConversion &&
        !layoutEncodingsEqual(solution.assignments.lookup(constraint.vars[0]),
                              solution.assignments.lookup(constraint.vars[1])) &&
        !converted.count(constraint.id))
      return emitError(loc) << "layout solution is missing a required consumer conversion";
    if (satisfiesLayoutConstraint(graph, constraint, solution.assignments,
                            /*requireComplete=*/true))
      continue;
    const LayoutProvenance &provenance =
        graph.getProvenances()[constraint.provenance];
    InFlightDiagnostic diagnostic = emitError(loc)
                                    << "solution violates hard layout constraint '"
                                    << provenance.rule << "': "
                                    << provenance.reason;
    std::string chain;
    llvm::raw_string_ostream stream(chain);
    if (succeeded(graph.printProvenanceChain(constraint.provenance, stream)))
      diagnostic.attachNote(loc) << chain;
    if (constraint.kind == ConstraintKind::AliasLayout) {
      auto proof = proveAliasLayoutRelation(graph, constraint.vars[0],
          solution.assignments.lookup(constraint.vars[0]), constraint.vars[1],
          solution.assignments.lookup(constraint.vars[1]));
      std::string coordinate;
      llvm::raw_string_ostream os(coordinate);
      llvm::interleaveComma(proof.counterexample, os);
      diagnostic.attachNote(loc) << proof.reason << "; coordinate [" << coordinate << "]";
    }
    return failure();
  }
  return success();
}

} // namespace mlir::frisk
