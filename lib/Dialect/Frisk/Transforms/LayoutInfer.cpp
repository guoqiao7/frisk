#include "Dialect/Frisk/Transforms/Passes.h"
#include "LegacyFragmentNormalization.h"

#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90TargetInfo.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKINFERLAYOUTS
#include "Dialect/Frisk/Transforms/Passes.h.inc"

namespace {
LogicalResult verifySM90TargetBoundary(Operation *root) {
  WalkResult result = root->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    return failed(getSM90TargetInfo(operation)) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  FriskInferLayoutsPass() = default;
  FriskInferLayoutsPass(const FriskInferLayoutsPass &other)
      : impl::FriskInferLayoutsBase<FriskInferLayoutsPass>(other),
        authorizeInternalOptimization(other.authorizeInternalOptimization),
        statistics(other.statistics), scope(other.scope) {}
  bool authorizeInternalOptimization = false;
  LayoutStatisticsReport *statistics = nullptr;
  LayoutStatisticsScope scope = LayoutStatisticsScope::Standalone;
  explicit FriskInferLayoutsPass(const SolverOptions &options, bool authorize,
                                 LayoutStatisticsReport *statistics, LayoutStatisticsScope scope)
      : authorizeInternalOptimization(authorize), statistics(statistics), scope(scope) {
    exactCombinationLimit = options.exactCombinationLimit;
    beamWidth = options.beamWidth;
    maxExpandedStates = options.maxExpandedStates;
  }
  void runOnOperation() override {
    auto start = LayoutClock::now();
    LayoutStatisticsReport local;
    auto &report = statistics ? *statistics : local;
    if (scope == LayoutStatisticsScope::Standalone) report = {};
    ModuleOp original = getOperation();
    // A requested statistics write is part of the transaction, not an IR side effect.
    OwningOpRef<ModuleOp> staged;
    if (!statsFile.empty()) staged = cast<ModuleOp>(original->clone());
    ModuleOp module = staged ? *staged : original;
    auto result = infer(module, report);
    report.timings["inference_total_ns"] = layoutElapsedNS(start);
    if (!statsFile.empty()) {
      std::string error;
      if (failed(writeLayoutStatistics(statsFile, report, error))) {
        report.status = "failed";
        report.failureKind = "statistics-write";
        report.failureReason = error;
        original.emitError("cannot write layout statistics: ") << error;
        signalPassFailure();
        return;
      }
    }
    if (failed(result)) { signalPassFailure(); return; }
    if (staged && !analysisOnly) {
      original->setAttrs((*staged)->getAttrs());
      original.getBodyRegion().takeBody(staged->getBodyRegion());
    }
  }
  LogicalResult infer(ModuleOp module, LayoutStatisticsReport &report) {
    report.status = "failed";
    report.failureKind = "input-verification";
    if (failed(verifyNoLegacyLayoutIR(module)) || failed(verifySM90TargetBoundary(module)))
      return failure();
    report.failureKind.clear();
    std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
    FailureOr<LayoutConstraintGraph> graph =
        collectLayoutConstraints(module, *target,
                                 LayoutCollectionMode::InitializeForRelaxation, &report);
    if (failed(graph)) {
      report.failureKind = "collection";
      return failure();
    }
    SmallVector<RelaxationRecord> relaxations;
    LayoutSolveFailure why;
    auto solveStart = LayoutClock::now();
    FailureOr<LayoutSolution> solution =
        attemptControlledRelaxation(*graph, *target,
            {exactCombinationLimit, beamWidth, maxExpandedStates}, relaxations, &why, &report);
    report.timings["propagation_search_retries_ns"] = layoutElapsedNS(solveStart);
    if (dumpAnalysis) printRelaxationRecords(relaxations, llvm::errs());
    if (failed(solution)) {
      recordLayoutFailure(*graph, why, report);
      return failure();
    }
    auto proofStart = LayoutClock::now();
    auto proof = verifySolvedLayoutGraph(*graph, *solution, *target, module.getLoc());
    report.timings["verification_ns"] += layoutElapsedNS(proofStart);
    if (failed(proof)) {
      report.status = "failed";
      report.failureKind = "solved-graph-verification";
      return failure();
    }
    report.failureKind.clear(); report.failureReason.clear();
    report.failureVariable.clear(); report.failureConstraint.clear();
    if (dumpAnalysis) {
      printLayoutSolutionStatistics(*graph, *solution, *target, llvm::errs());
      graph->print(llvm::errs());
      for (const auto &var : graph->getVariables())
        if (var.kind == LayoutKind::Distributed)
          llvm::errs() << "distributed domain " << var.stableName << ": "
                       << var.candidates.size() << '\n';
      llvm::errs() << "conversions: " << solution->conversions.size() << '\n';
      for (const auto &edge : solution->conversions)
        llvm::errs() << (edge.resolution == EdgeResolutionKind::Rematerialize
                            ? "rematerialize " : "convert ")
                     << graph->getConstraint(edge.constraint).stableUseKey
                     << " bytes=" << edge.bytes << " sync=" << edge.synchronizationCost
                     << " estimate=" << edge.costProvenance
                     << " loop-depth=" << edge.loopDepth
                     << " leader=" << edge.sharedLeader
                     << " slice-ops=" << edge.rematerialization.size() << '\n';
    }
    report.status = "solved";
    if (analysisOnly) return success();
    auto materializeStart = LayoutClock::now();
    auto materialized = materializeLayouts(module, *graph, *solution,
                                           authorizeInternalOptimization);
    report.timings["materialization_ns"] = layoutElapsedNS(materializeStart);
    auto verificationStart = LayoutClock::now();
    auto verified = succeeded(materialized) ? verifyMaterializedLayouts(module, *target) : failure();
    report.timings["verification_ns"] += layoutElapsedNS(verificationStart);
    if (failed(materialized) || failed(verified)) {
      report.status = "failed";
      report.failureKind = "materialization-or-verification";
      return failure();
    }
    report.status = "materialized";
    return success();
  }
};
} // namespace

std::unique_ptr<Pass> createFriskInferLayoutsPass() {
  return std::make_unique<FriskInferLayoutsPass>();
}
std::unique_ptr<Pass> createFriskInferLayoutsPass(const SolverOptions &options,
                                              bool authorizeInternalOptimization,
                                              LayoutStatisticsReport *statistics,
                                              LayoutStatisticsScope scope) {
  return std::make_unique<FriskInferLayoutsPass>(options, authorizeInternalOptimization, statistics, scope);
}

} // namespace mlir::frisk
