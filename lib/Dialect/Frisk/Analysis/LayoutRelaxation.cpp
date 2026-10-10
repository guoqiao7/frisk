#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "llvm/ADT/ScopeExit.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
namespace mlir::frisk {
namespace {
StringRef name(RelaxationKind kind) {
  switch (kind) {
  case RelaxationKind::Replication: return "replication";
  case RelaxationKind::StoragePaddingOrNarrowerVector: return "padding-or-narrower-vector";
  case RelaxationKind::GuardedRagged: return "guarded-ragged";
  case RelaxationKind::TMAFallback: return "tma-fallback";
  case RelaxationKind::WGMMAFallback: return "wgmma-fallback";
  }
  llvm_unreachable("unknown relaxation kind");
}
Location location(const LayoutConstraintGraph &graph) {
  const auto &var = graph.getVariables().front();
  return var.anchor ? var.anchor->getLoc() : UnknownLoc::get(var.shapedType.getContext());
}
// Propagation retains its existing diagnostics/statistics. If it fails, run the
// structured solver on the unpruned snapshot to distinguish proof/seed/budget
// failures without parsing diagnostic text or relaxing after an unknown proof.
FailureOr<LayoutSolution> runAttempt(LayoutConstraintGraph &graph,
    LayoutTarget &target, SolverOptions options, LayoutSolveFailure *why,
    LayoutAttemptStatistics *statistics) {
  if (statistics) prepareLayoutStatistics(graph, *statistics);
  auto before = graph;
  auto start = LayoutClock::now();
  auto propagated = propagateCommonToFixedPoint(graph);
  if (statistics) statistics->propagationTime = std::chrono::nanoseconds(layoutElapsedNS(start));
  if (failed(propagated)) {
    graph = std::move(before);
    ScopedDiagnosticHandler quiet(location(graph).getContext(),
                                 [](Diagnostic &) { return success(); });
    auto diagnosis = solveLayoutGraph(graph, target, options, why, statistics);
    if (succeeded(diagnosis)) {
      *why = {LayoutSolveFailureKind::Unsupported, UINT32_MAX, UINT32_MAX,
              "propagation rejected a solver-supported graph"};
    }
    if (statistics) finishLayoutStatistics(graph, *why, nullptr, *statistics);
    return failure();
  }
  auto result = solveLayoutGraph(graph, target, options, why, statistics);
  auto proofStart = LayoutClock::now();
  auto proof = succeeded(result) ? verifySolvedLayoutGraph(graph, *result, target, location(graph)) : success();
  if (statistics) statistics->verificationTime = std::chrono::nanoseconds(layoutElapsedNS(proofStart));
  if (failed(proof)) {
    *why = {LayoutSolveFailureKind::Unsupported, UINT32_MAX, UINT32_MAX,
            "selected layout proof failed"};
    if (statistics) finishLayoutStatistics(graph, *why, nullptr, *statistics);
    return failure();
  }
  return result;
}
} // namespace
FailureOr<LayoutSolution> attemptControlledRelaxation(
    LayoutConstraintGraph &graph, LayoutTarget &target, SolverOptions options,
    SmallVectorImpl<RelaxationRecord> &records, LayoutSolveFailure *why,
    LayoutStatisticsReport *statistics) {
  records.clear();
  LayoutSolveFailure local;
  if (!why) why = &local;
  // Failure preserves the caller graph, whose IDs may still be insertion-order
  // IDs. Keep the finalized identity owner alive until after report emission.
  std::optional<LayoutConstraintGraph> failureIdentity;
  auto firstAttempt = statistics ? statistics->attempts.size() : 0;
  auto finish = llvm::make_scope_exit([&] {
    if (!statistics) return;
    recordLayoutFailure(failureIdentity ? *failureIdentity : graph, *why, *statistics);
    statistics->status = why->kind == LayoutSolveFailureKind::None ? "solved" : "failed";
    for (size_t i = firstAttempt; i < statistics->attempts.size(); ++i)
      statistics->timings["verification_ns"] += statistics->attempts[i].verificationTime.count();
    for (const auto &record : records)
      statistics->relaxations.push_back(llvm::json::Object{
        {"kind", name(record.kind)}, {"generatedProposals", record.generatedCandidates},
        {"addedCandidates", record.addedCandidates},
        {"outcome", record.outcome == RelaxationOutcome::Selected ? "selected"
            : record.outcome == RelaxationOutcome::Rejected ? "rejected" : "no-new-candidates"},
        {"timings", llvm::json::Object{{"preparation_ns", record.preparationNS}}}});
  });
  // Empty/invalid graphs must go through the solver's input validation first.
  if (graph.getVariables().empty() || !graph.getVariables().front().shapedType) {
    if (statistics) statistics->attempts.emplace_back();
    return solveLayoutGraph(graph, target, options, why,
                            statistics ? &statistics->attempts.back() : nullptr);
  }
  auto original = graph;
  Location loc = location(original);
  if (failed(original.verifyInvariants(loc)) ||
      (!original.isFinalized() && failed(original.finalize(loc)))) {
    *why = {LayoutSolveFailureKind::InvalidInput, UINT32_MAX, UINT32_MAX,
            "invalid graph invariants"};
    return failure();
  }
  if (statistics) failureIdentity = original;
  auto trial = original;
  std::vector<Diagnostic> diagnostics;
  auto quietAttempt = [&](LayoutConstraintGraph &attempt, StringRef kind) {
    diagnostics.clear();
    ScopedDiagnosticHandler capture(loc.getContext(), [&](Diagnostic &d) {
      diagnostics.push_back(std::move(d));
      return success();
    });
    if (statistics) statistics->attempts.emplace_back();
    auto result = runAttempt(attempt, target, options, why,
                              statistics ? &statistics->attempts.back() : nullptr);
    if (statistics) statistics->attempts.back().kind = kind.str();
    return result;
  };
  auto result = quietAttempt(trial, "initial");
  if (succeeded(result)) {
    graph = std::move(trial);
    return result;
  }
  if (why->kind != LayoutSolveFailureKind::EmptyDomain) {
    for (auto &diagnostic : diagnostics)
      loc.getContext()->getDiagEngine().emit(std::move(diagnostic));
    return failure();
  }
  // Retain the original failure location and notes independently of trial
  // diagnostics. A later rejected proposal must not replace that evidence.
  auto originalDiagnostics = std::move(diagnostics);
  // An earlier empty component must not hide a mandatory conflict elsewhere.
  DenseMap<LayoutVarID, Attribute> mandatory;
  for (const auto &c : original.getConstraints()) {
    if (c.strength != ConstraintStrength::Hard ||
        c.kind != ConstraintKind::RequireEncoding) continue;
    auto id = c.vars.front();
    const auto &var = original.getVariable(id);
    auto [it, inserted] = mandatory.try_emplace(id, c.requiredEncoding);
    if ((!inserted && it->second != c.requiredEncoding) ||
        llvm::none_of(var.candidates, [&](const auto &candidate) {
          return candidate.value == c.requiredEncoding;
        })) {
      *why = {LayoutSolveFailureKind::HardConflict, id, c.id,
              "mandatory encoding conflicts with the original candidate domain"};
      emitError(loc) << why->reason;
      return failure();
    }
  }
  // Check relations among mandatory endpoints even when another endpoint in
  // the same connected component is empty and prevents complete enumeration.
  for (const auto &c : original.getConstraints()) {
    if (c.strength != ConstraintStrength::Hard ||
        !llvm::all_of(c.vars, [&](auto id) { return mandatory.count(id); }) ||
        satisfiesLayoutConstraint(original, c, mandatory, false)) continue;
    ProofStatus status = ProofStatus::Disproven;
    if (isSupportedUnaryLayoutConstraint(c.kind))
      status = proveUnaryLayoutConstraint(original, c, mandatory.lookup(c.vars.front())).status;
    else if (isSupportedLayoutRelation(c.kind))
      for (auto id : llvm::drop_begin(c.vars))
        if (proveLayoutRelation(original,c,c.vars.front(),mandatory.lookup(c.vars.front()),
                                id,mandatory.lookup(id)).status == ProofStatus::Unknown)
          status = ProofStatus::Unknown;
    *why = {status == ProofStatus::Unknown ? LayoutSolveFailureKind::Unsupported
                                          : LayoutSolveFailureKind::HardConflict,
            c.vars.empty() ? UINT32_MAX : c.vars.front(), c.id,
            "mandatory endpoint relation is not proven"};
    emitError(loc) << why->reason;
    return failure();
  }
  const LayoutSolveFailure trigger = *why;
  for (RelaxationKind kind : {RelaxationKind::Replication,
       RelaxationKind::StoragePaddingOrNarrowerVector, RelaxationKind::GuardedRagged,
       RelaxationKind::TMAFallback, RelaxationKind::WGMMAFallback}) {
    trial = original;
    RelaxationRecord record;
    record.kind = kind;
    record.variable = trigger.variable;
    record.trigger = trigger.constraint;
    record.reason = trigger.reason + "; " + target.relaxationCoverage(kind).str();
    auto preparationStart = LayoutClock::now();
    for (auto &var : trial.getVariables()) {
      bool fixed = llvm::any_of(original.getConstraints(), [&](const auto &c) {
        return c.strength == ConstraintStrength::Hard &&
               llvm::is_contained(c.vars, var.id) &&
               (c.kind == ConstraintKind::RequireEncoding ||
                c.kind == ConstraintKind::InstructionContract ||
                c.kind == ConstraintKind::ReductionLayout);
      });
      // Joint instruction/reduction alternatives belong to their target table,
      // not generic endpoint expansion. No stale support tables are accepted.
      if (fixed) continue;
      SmallVector<LayoutCandidate> candidates;
      target.enumerateRelaxedCandidates(original, original.getVariable(var.id),
                                        kind, candidates);
      record.generatedCandidates += candidates.size();
      for (auto candidate : candidates) {
        if (llvm::any_of(var.candidates, [&](const auto &known) {
              return known.value == candidate.value;
            })) continue;
        candidate.provenance = trial.addProvenance(std::nullopt, var.anchor,
            "controlled-relaxation", name(kind));
        var.candidates.push_back(candidate);
        ++record.addedCandidates;
      }
      var.state = var.candidates.empty() ? LayoutState::Conflict : LayoutState::CandidateSet;
    }
    record.preparationNS = layoutElapsedNS(preparationStart);
    if (!record.addedCandidates) {
      records.push_back(std::move(record));
      continue;
    }
    result = quietAttempt(trial, name(kind));
    record.outcome = succeeded(result) ? RelaxationOutcome::Selected : RelaxationOutcome::Rejected;
    if (succeeded(result)) record.selectedCost = result->cost;
    else record.reason += "; rejected: " + why->reason;
    records.push_back(std::move(record));
    if (succeeded(result)) {
      graph = std::move(trial);
      return result;
    }
    // A retry that hits a budget or unknown is inconclusive. Do not use it as
    // evidence that another more relaxed policy is necessary.
    if (why->kind != LayoutSolveFailureKind::EmptyDomain &&
        why->kind != LayoutSolveFailureKind::HardConflict) break;
  }
  for (auto &diagnostic : originalDiagnostics)
    loc.getContext()->getDiagEngine().emit(std::move(diagnostic));
  emitError(loc) << "controlled relaxation exhausted or unsupported: "
                 << why->reason << "; "
                 << (records.empty() ? StringRef() : StringRef(records.back().reason));
  return failure();
}
void printRelaxationRecords(ArrayRef<RelaxationRecord> records, raw_ostream &os) {
  for (const auto &r : records) {
    os << "relaxation " << name(r.kind) << " variable=" << r.variable
       << " constraint=" << r.trigger << " added=" << r.addedCandidates
       << " outcome=" << (r.outcome == RelaxationOutcome::Selected ? "selected"
           : r.outcome == RelaxationOutcome::Rejected ? "rejected" : "no-new-candidates")
       << " baseline=unavailable reason=" << r.reason;
    if (r.selectedCost) {
      const auto &c = r.selectedCost->cost;
      os << " selected-cost=[" << c.instructionPathAndWork << ',' << c.memoryTransactions
         << ',' << c.bankConflictDegree << ',' << c.conversionBytesAndSync << ','
         << c.spillRiskAndRegisters << ',' << c.sharedBytesAndOccupancy << ','
         << c.replication << ',' << c.codeSize << ',' << c.deterministicTieBreak << ']';
    }
    os << '\n';
  }
}
} // namespace mlir::frisk
