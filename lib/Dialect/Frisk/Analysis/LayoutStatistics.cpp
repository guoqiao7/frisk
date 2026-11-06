#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include <set>

namespace mlir::frisk {
void prepareLayoutStatistics(const LayoutConstraintGraph &graph,
                             LayoutAttemptStatistics &attempt) {
  attempt = {};
  attempt.prepared = true;
  attempt.statistics.variables = graph.getVariables().size();
  attempt.statistics.constraints = graph.getConstraints().size();
  for (const auto &var : graph.getVariables()) {
    llvm::DenseSet<Attribute> unique;
    for (const auto &candidate : var.candidates) unique.insert(candidate.value);
    attempt.statistics.preparedCandidates += unique.size();
  }
}
StringRef stringifyLayoutSolveFailureKind(LayoutSolveFailureKind kind) {
  switch (kind) {
  case LayoutSolveFailureKind::None: return "none";
  case LayoutSolveFailureKind::InvalidInput: return "invalid-input";
  case LayoutSolveFailureKind::EmptyDomain: return "empty-domain";
  case LayoutSolveFailureKind::HardConflict: return "hard-conflict";
  case LayoutSolveFailureKind::Unsupported: return "unsupported";
  case LayoutSolveFailureKind::SearchBudget: return "search-budget";
  case LayoutSolveFailureKind::SearchIncomplete: return "search-incomplete";
  case LayoutSolveFailureKind::CostFailure: return "cost-failure";
  }
  llvm_unreachable("unknown solve failure");
}
void finishLayoutStatistics(const LayoutConstraintGraph &graph,
                            const LayoutSolveFailure &failure,
                            const LayoutSolution *solution,
                            LayoutAttemptStatistics &attempt) {
  attempt.succeeded = solution != nullptr;
  attempt.failureKind = stringifyLayoutSolveFailureKind(failure.kind).str();
  attempt.failureReason = failure.reason;
  if (failure.variable < graph.getVariables().size())
    attempt.failureVariable = graph.getVariable(failure.variable).stableName;
  if (failure.constraint < graph.getConstraints().size()) {
    const auto &constraint = graph.getConstraint(failure.constraint);
    attempt.failureConstraint = std::to_string(constraint.id) + ":" +
        stringifyConstraintKind(constraint.kind).str() + ":" + constraint.stableUseKey;
  }
  if (!solution) return;
  auto &stats = attempt.statistics;
  std::set<LayoutConstraintID> conversions, rematerializations;
  for (const auto &edge : solution->conversions) {
    auto leader = edge.sharedLeader == UINT32_MAX ? edge.constraint : edge.sharedLeader;
    if (edge.resolution == EdgeResolutionKind::Convert) {
      ++stats.conversionUseEdges;
      if (conversions.insert(leader).second) {
        stats.conversionBytes += edge.bytes;
        auto &saturated = stats.weightedConversionBytesSaturated;
        uint64_t weighted = edge.bytes;
        if (edge.frequencyFactor && weighted > UINT64_MAX / edge.frequencyFactor) {
          weighted = UINT64_MAX; saturated = true;
        } else weighted *= edge.frequencyFactor;
        saturated |= edge.frequencySaturated && edge.bytes != 0;
        stats.weightedConversionBytesUsesProxy |= edge.frequencyUsesProxy;
        if (stats.weightedConversionBytes > UINT64_MAX - weighted) {
          stats.weightedConversionBytes = UINT64_MAX; saturated = true;
        } else stats.weightedConversionBytes += weighted;
      }
    } else if (edge.resolution == EdgeResolutionKind::Rematerialize) {
      ++stats.rematerializationUseEdges;
      rematerializations.insert(leader);
    }
  }
  stats.uniqueConversions = conversions.size();
  stats.uniqueRematerializations = rematerializations.size();
  stats.estimatedBankConflicts = solution->cost.cost.bankConflictDegree;
  stats.estimatedMemoryTransactions = solution->cost.cost.memoryTransactions;
}
static llvm::json::Object statsJSON(const LayoutStatistics &s, bool success) {
  llvm::json::Object result{
    {"variables", s.variables}, {"constraints", s.constraints}, {"components", s.components},
    {"preparedCandidates", s.preparedCandidates}, {"prunedCandidates", s.prunedCandidates},
    {"peakVariableDomain", s.peakVariableDomain},
    {"peakComponentCombinations", s.componentProductOverflow ? llvm::json::Value(nullptr) : llvm::json::Value(s.peakComponentCombinations)},
    {"componentProductOverflow", s.componentProductOverflow},
    {"peakFrontierStates", s.peakFrontierStates}, {"expandedStates", s.expandedStates}};
  if (success) {
    result["conversionUseEdges"] = s.conversionUseEdges;
    result["uniqueConversions"] = s.uniqueConversions;
    result["rematerializationUseEdges"] = s.rematerializationUseEdges;
    result["uniqueRematerializations"] = s.uniqueRematerializations;
    result["conversionBytes"] = s.conversionBytes;
    result["weightedConversionBytes"] = s.weightedConversionBytes;
    result["weightedConversionBytesUsesProxy"] = s.weightedConversionBytesUsesProxy;
    result["weightedConversionBytesSaturated"] = s.weightedConversionBytesSaturated;
    result["estimatedBankConflicts"] = s.estimatedBankConflicts;
    result["estimatedMemoryTransactions"] = s.estimatedMemoryTransactions;
  }
  return result;
}
void recordLayoutFailure(const LayoutConstraintGraph &graph, const LayoutSolveFailure &why,
                         LayoutStatisticsReport &report) {
  LayoutAttemptStatistics failure;
  finishLayoutStatistics(graph, why, nullptr, failure);
  report.failureKind = why.kind == LayoutSolveFailureKind::None ? "" : failure.failureKind;
  report.failureReason = failure.failureReason;
  report.failureVariable = failure.failureVariable;
  report.failureConstraint = failure.failureConstraint;
}
llvm::json::Object LayoutStatisticsReport::toJSON() const {
  llvm::json::Array records;
  for (const auto &attempt : attempts) {
    llvm::json::Object record{{"statistics", statsJSON(attempt.statistics, attempt.succeeded)},
      {"kind", attempt.kind},
      {"status", attempt.succeeded ? "solved" : "failed"},
      {"components", llvm::json::Array(attempt.components)}, {"mrvProbes", attempt.mrvProbes},
      {"expandedNodes", attempt.expandedNodes},
      {"refinementProbes", attempt.refinementProbes},
      {"refinementSweeps", attempt.refinementSweeps},
      {"refinementAccepted", attempt.refinementAccepted},
      {"timings", llvm::json::Object{{"solve_ns", attempt.statistics.solveTime.count()},
                                    {"propagation_ns", attempt.propagationTime.count()},
                                    {"verification_ns", attempt.verificationTime.count()}}}};
    if (!attempt.succeeded)
      record["failure"] = llvm::json::Object{{"kind", attempt.failureKind},
        {"variable", attempt.failureVariable}, {"constraint", attempt.failureConstraint},
        {"reason", attempt.failureReason}};
    records.push_back(std::move(record));
  }
  llvm::json::Object time;
  for (const auto &entry : timings) time[entry.first] = entry.second;
  llvm::json::Object result{{"schema_version", 1}, {"status", status},
    {"actual_ir_diagnostics", actualIRDiagnostics},
    {"cost_model_id", normalizeLayoutCostModelID(costModelID).str()},
    {"boundaryFrequency", staticTripCountFrequency ? "static-trip-count" : "legacy"},
    {"preparation", llvm::json::Object{{"rawGeneratedProposals", rawGeneratedProposals},
        {"instructionCombinations", instructionCombinations},
        {"instructionFragmentPrechecks", instructionFragmentPrechecks}}},
    {"attempts", std::move(records)}, {"relaxations", llvm::json::Array(relaxations)},
    {"timings", std::move(time)}};
  if (status == "failed") result["failure"] = llvm::json::Object{
      {"kind", failureKind}, {"reason", failureReason},
      {"variable", failureVariable}, {"constraint", failureConstraint}};
  return result;
}
LogicalResult writeLayoutStatistics(StringRef path, const LayoutStatisticsReport &report,
                                    std::string &error) {
  auto temp = llvm::sys::fs::TempFile::create(path + ".tmp-%%%%%%");
  if (!temp) { error = llvm::toString(temp.takeError()); return failure(); }
  llvm::raw_fd_ostream stream(temp->FD, false);
  stream << llvm::formatv("{0:2}\n", llvm::json::Value(report.toJSON()));
  stream.flush();
  if (stream.has_error()) {
    error = stream.error().message();
    stream.clear_error();
    llvm::consumeError(temp->discard());
    return failure();
  }
  if (auto e = temp->keep(path)) { error = llvm::toString(std::move(e)); return failure(); }
  return success();
}
} // namespace mlir::frisk
