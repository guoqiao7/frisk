#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Diagnostics.h"
#include <functional>

namespace mlir::frisk {
const InstructionLayoutTuple *findInstructionSupport(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &assignment) {
  LayoutProfile::Scope profile(LayoutProfileKind::InstructionSupport);
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
                                       LayoutTarget &target, Location loc,
                                       LayoutTrace *trace) {
  for (auto &constraint : graph.getConstraints()) {
    if (constraint.kind != ConstraintKind::InstructionContract) continue;
    bool mma = constraint.instruction && isa_and_nonnull<MmaOp>(constraint.instruction->source);
    if (!constraint.instruction || constraint.vars.size() != (mma ? 5u : 4u))
      return emitError(loc) << "unsupported instruction contract";
    auto &instruction = *constraint.instruction;
    instruction.tuples.clear();
    constexpr unsigned completeProofBudget = 256;
    bool expandedEndpoint = false;
    SmallVector<Attribute> encodings(constraint.vars.size());
    SmallVector<SmallVector<Attribute>> baseDomains(constraint.vars.size());
    SmallVector<SmallVector<Attribute>> domains(constraint.vars.size());
    for (auto [role, id] : llvm::enumerate(constraint.vars)) {
      const auto &var = graph.getVariable(id);
      SmallVector<const LayoutConstraint *> requirements;
      if (var.candidates.size() > 4) {
        expandedEndpoint = true;
        for (const auto &requirement : graph.getConstraints())
          if (requirement.strength == ConstraintStrength::Hard &&
              requirement.vars.size() == 1 && requirement.vars.front() == id &&
              (requirement.kind == ConstraintKind::RequireEncoding ||
               requirement.kind == ConstraintKind::RootStorageContract))
            requirements.push_back(&requirement);
      }
      for (const auto &candidate : var.candidates) {
        bool incompatible = llvm::any_of(requirements, [&](const auto *requirement) {
          if (requirement->kind == ConstraintKind::RequireEncoding)
            return candidate.value != requirement->requiredEncoding;
          return proveUnaryLayoutConstraint(graph, *requirement, candidate.value).status ==
                 ProofStatus::Disproven;
        });
        if (!incompatible) baseDomains[role].push_back(candidate.value);
      }
    }
    DenseMap<LayoutVarID, Attribute> chosen;
    std::optional<InstructionContractBuildResult> firstFailure;
    bool sawUnknown = false;
    bool exhausted = false;
    unsigned completeAttempts = 0;
    std::function<void(unsigned)> visit = [&](unsigned role) {
      if (exhausted) return;
      if (role == (mma ? 4u : constraint.vars.size())) {
        if (completeAttempts == completeProofBudget) { exhausted = true; return; }
        ++completeAttempts;
        ++graph.getCandidatePreparationStatistics().instructionCombinations;
        auto result = [&] {
          LayoutProfile::Scope profile(LayoutProfileKind::InstructionContract);
          return target.buildInstructionContractDetailed(graph, constraint, encodings);
        }();
        // Construction already proves the returned binding. An explicit plan
        // must equal that checked plan; do not prove the same tuple twice.
        if (result.proof.status == ProofStatus::Proven && instruction.binding &&
            result.binding != instruction.binding) {
          result = {{}, {ProofStatus::Disproven, {},
              "explicit instruction binding differs from the checked descriptor, issuer or completion"},
              isa_and_nonnull<CopyOp>(instruction.source) ? "copy-binding" : "instruction-binding", {}};
        }
        bool proven = result.proof.status == ProofStatus::Proven && result.binding;
        if (trace) trace->recordEvent([&] {
          auto path = dyn_cast_or_null<StringAttr>(encodings.back());
          llvm::json::Array coordinate, endpoints;
          for (auto value : result.proof.counterexample) coordinate.push_back(value);
          for (auto id : constraint.vars) endpoints.push_back(id);
          llvm::json::Object event{{"event", proven ? "instruction-supported" : "instruction-rejected"},
              {"constraint", constraint.id}, {"endpoints", std::move(endpoints)},
              {"path", path ? path.getValue() : StringRef("mma")},
              {"status", proven ? "proven" : result.proof.status == ProofStatus::Unknown ? "unknown" : "disproven"},
              {"reason_code", result.reasonCode}, {"reason", result.proof.reason},
              {"counterexample", std::move(coordinate)}};
          if (result.operand) event["operand"] = *result.operand;
          if (mma) event["geometry"] = LayoutTrace::candidateID(encodings[4]);
          event["complete_proof_attempt"] = completeAttempts;
          return event;
        });
        if (proven)
          instruction.tuples.push_back({encodings, result.binding});
        else {
          sawUnknown |= result.proof.status == ProofStatus::Unknown;
          // Prefer unknown evidence if any alternative remains undecided;
          // never present mixed failed proofs as joint disproval.
          if (!firstFailure || (result.proof.status == ProofStatus::Unknown &&
                                firstFailure->proof.status != ProofStatus::Unknown))
            firstFailure = std::move(result);
        }
        return;
      }
      auto id = constraint.vars[role];
      if (auto prior = chosen.lookup(id)) {
        if (!llvm::is_contained(domains[role],prior)) return;
        encodings[role] = prior;
        visit(role + 1);
        return;
      }
      for (Attribute candidate : domains[role]) {
        chosen[id] = candidate;
        encodings[role] = candidate;
        visit(role + 1);
        chosen.erase(id);
        if (exhausted) break;
      }
    };
    // Only widened endpoint domains require a preflight product bound. Count
    // distinct variable assignments after intersecting repeated roles. The
    // geometry binding is already chosen for MMA and has cardinality one.
    auto visitBounded = [&] {
      if (!expandedEndpoint) {
        visit(0);
        return;
      }
      constexpr unsigned roleCount = 4;
      SmallVector<LayoutVarID> seen;
      SmallVector<size_t> counts;
      for (unsigned role = 0; role < roleCount; ++role) {
        LayoutVarID id = constraint.vars[role];
        if (llvm::is_contained(seen, id)) continue;
        seen.push_back(id);
        size_t count = 0;
        for (Attribute candidate : domains[role]) {
          if (Attribute prior = chosen.lookup(id); prior && prior != candidate)
            continue;
          bool matches = true;
          for (unsigned other = role + 1; other < roleCount; ++other)
            if (constraint.vars[other] == id && !llvm::is_contained(domains[other], candidate)) {
              matches = false;
              break;
            }
          count += matches;
        }
        if (!count) return; // No possible complete assignment.
        counts.push_back(count);
      }
      size_t product = 1, remaining = completeProofBudget - completeAttempts;
      for (size_t count : counts) {
        if (product > remaining / count) {
          exhausted = true;
          return;
        }
        product *= count;
      }
      visit(0);
    };
    // Rejected alternatives are not user errors. Report one joint failure if
    // there is no supported tuple, with pure-proof details for explicit plans.
    {
      ScopedDiagnosticHandler quiet(loc.getContext(),
                                     [](Diagnostic &) { return success(); });
      if (!mma) {
        for (unsigned role = 0; role < 4; ++role)
          domains[role] = baseDomains[role];
        visitBounded();
      } else for (Attribute geometry : baseDomains[4]) {
        encodings[4] = geometry;
        chosen[constraint.vars[4]] = geometry;
        for (unsigned role = 0; role < 4; ++role) {
          domains[role].clear();
          const auto &var = graph.getVariable(constraint.vars[role]);
          for (Attribute candidate : baseDomains[role]) {
            if (var.kind == LayoutKind::Distributed) {
              ++graph.getCandidatePreparationStatistics().instructionFragmentPrechecks;
              auto proof = target.precheckInstructionRole(graph,constraint,role,candidate,geometry);
              if (trace) trace->recordEvent([&] {
                return llvm::json::Object{{"event","instruction-fragment-precheck"},
                    {"constraint",constraint.id},{"operand",role},
                    {"geometry",LayoutTrace::candidateID(geometry)},
                    {"candidate",LayoutTrace::candidateID(candidate)},
                    {"status",proof.status == ProofStatus::Proven ? "proven" :
                              proof.status == ProofStatus::Disproven ? "disproven" : "unknown"},
                    {"reason",proof.reason}};
              });
              if (proof.status == ProofStatus::Disproven) {
                if (!firstFailure) firstFailure = InstructionContractBuildResult{
                    {},std::move(proof),"mma-fragment",role};
                continue;
              }
            }
            domains[role].push_back(candidate);
          }
        }
        visitBounded();
        chosen.erase(constraint.vars[4]);
        if (exhausted) break;
      }
    }
    if (exhausted) {
      instruction.tuples.clear();
      return emitError(loc)
          << (mma ? "mma-joint-contract" : "copy-joint-contract")
          << ": unknown proof: complete tuple proof budget exceeded ("
          << completeProofBudget << " attempts); partial table discarded";
    }
    if (instruction.tuples.empty()) {
      auto diagnostic = emitError(instruction.source ? instruction.source->getLoc() : loc);
      diagnostic << (isa_and_nonnull<CopyOp>(instruction.source) ? "copy-joint-contract" : "mma-joint-contract")
                 << ": no proven instruction tuple in frozen candidate domains";
      if (firstFailure) {
        diagnostic << "; " << (sawUnknown ? "unknown proof: " : "counterexample: ")
                   << firstFailure->reasonCode << ": " << firstFailure->proof.reason;
        if (firstFailure->operand) diagnostic << "; operand " << *firstFailure->operand;
        if (!firstFailure->proof.counterexample.empty()) {
          std::string coordinates;
          llvm::raw_string_ostream stream(coordinates);
          llvm::interleaveComma(firstFailure->proof.counterexample, stream);
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
