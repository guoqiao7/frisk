#ifndef FRISK_ANALYSIS_LAYOUTRELAXATION_H
#define FRISK_ANALYSIS_LAYOUTRELAXATION_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"

namespace mlir::frisk {
enum class RelaxationOutcome { NoNewCandidates, Rejected, Selected };
struct RelaxationRecord {
  RelaxationKind kind;
  LayoutVarID variable = UINT32_MAX;
  LayoutConstraintID trigger = UINT32_MAX;
  uint64_t addedCandidates = 0;
  uint64_t generatedCandidates = 0, preparationNS = 0;
  RelaxationOutcome outcome = RelaxationOutcome::NoNewCandidates;
  std::string reason;
  std::optional<CostEstimate> baselineCost;
  std::optional<CostEstimate> selectedCost;
};
// Success replaces graph with the winning, validated snapshot. Failure leaves
// it byte-for-byte logically unchanged; records contain the attempt history.
FailureOr<LayoutSolution> attemptControlledRelaxation(
    LayoutConstraintGraph &graph, LayoutTarget &target, SolverOptions options,
    SmallVectorImpl<RelaxationRecord> &records, LayoutSolveFailure *why = nullptr,
    LayoutStatisticsReport *statistics = nullptr);
void printRelaxationRecords(ArrayRef<RelaxationRecord> records, raw_ostream &os);
} // namespace mlir::frisk
#endif
