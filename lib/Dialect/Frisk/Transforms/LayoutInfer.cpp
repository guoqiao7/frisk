#include "Dialect/Frisk/Transforms/Passes.h"
#include "LegacyFragmentNormalization.h"

#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
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
        statistics(other.statistics), scope(other.scope), trace(other.trace), profile(other.profile),
        calibratedProfile(other.calibratedProfile) {}
  bool authorizeInternalOptimization = false;
  LayoutStatisticsReport *statistics = nullptr;
  LayoutStatisticsScope scope = LayoutStatisticsScope::Standalone;
  LayoutTrace *trace = nullptr;
  LayoutProfile *profile = nullptr;
  std::optional<SM90CostProfile> calibratedProfile;
  explicit FriskInferLayoutsPass(const SolverOptions &options, bool authorize,
                                 LayoutStatisticsReport *statistics, LayoutStatisticsScope scope, LayoutTrace *trace,
                                 LayoutProfile *profile, LayoutBoundaryFrequencyMode mode,
                                 const SM90CostProfile *costProfile)
      : authorizeInternalOptimization(authorize), statistics(statistics), scope(scope), trace(trace), profile(profile) {
    if (costProfile) calibratedProfile = *costProfile;
    boundaryFrequency = mode == LayoutBoundaryFrequencyMode::Legacy ? "legacy" : "static-trip-count";
    exactCombinationLimit = options.exactCombinationLimit;
    beamWidth = options.beamWidth;
    maxExpandedStates = options.maxExpandedStates;
    refineBeamSolution = options.refineBeamSolution;
    maxRefinementProbes = options.maxRefinementProbes;
    memoizeCosts = options.memoizeCosts;
    seedBeamSolution = options.seedBeamSolution;
    conversionPolicy = options.policy.conversion == ConversionPolicy::Independent ? "independent" : options.policy.conversion == ConversionPolicy::SharedOnly ? "shared-only" : "full";
  }
  void runOnOperation() override {
    if (conversionPolicy != "full" && conversionPolicy != "shared-only" && conversionPolicy != "independent") { getOperation().emitError("unknown conversion-policy"); signalPassFailure(); return; }
    LayoutProfile ownedProfile(!profileFile.empty());
    auto *activeProfile = ownedProfile.enabled() ? &ownedProfile : profile;
    LayoutProfile::Invocation profileInvocation(activeProfile);
    auto start = LayoutClock::now();
    LayoutStatisticsReport local;
    auto &report = statistics ? *statistics : local;
    if (scope == LayoutStatisticsScope::Standalone) report = {};
    if (boundaryFrequency != "legacy" && boundaryFrequency != "static-trip-count") {
      getOperation().emitError("invalid boundary-frequency: expected legacy or static-trip-count");
      report.status = "failed"; report.failureKind = "boundary-frequency";
      report.costModelID = "unversioned-target";
      signalPassFailure(); return;
    }
    auto mode = boundaryFrequency == "legacy" ? LayoutBoundaryFrequencyMode::Legacy : LayoutBoundaryFrequencyMode::StaticTripCount;
    auto activeCalibration = calibratedProfile;
    if (!costProfileFile.empty()) {
      if (activeCalibration || costProfileFile == statsFile || costProfileFile == traceFile || costProfileFile == profileFile) {
        getOperation().emitError("cost-profile conflicts with supplied profile or output file");
        signalPassFailure(); return;
      }
      auto loaded = loadSM90CostProfileFile(costProfileFile);
      if (!loaded) {
        getOperation().emitError("invalid cost-profile: ") << llvm::toString(loaded.takeError());
        report.status = "failed"; report.failureKind = "cost-profile";
        signalPassFailure(); return;
      }
      activeCalibration = std::move(*loaded);
    }
    auto target = activeCalibration ? createSM90LayoutTarget(mode, *activeCalibration)
                                    : createSM90LayoutTarget(mode);
    report.costModelID = normalizeLayoutCostModelID(target->costModelID()).str();
    report.staticTripCountFrequency = mode == LayoutBoundaryFrequencyMode::StaticTripCount;
    auto initializeTrace = [&](LayoutTrace *recorder) {
      if (!recorder) return;
      SolverOptions traceSolver{exactCombinationLimit, beamWidth, maxExpandedStates,
                                refineBeamSolution, maxRefinementProbes};
      traceSolver.seedBeamSolution = seedBeamSolution;
      traceSolver.memoizeCosts = memoizeCosts;
      traceSolver.policy.conversion = conversionPolicy == "independent" ? ConversionPolicy::Independent : conversionPolicy == "shared-only" ? ConversionPolicy::SharedOnly : ConversionPolicy::Full;
      recorder->setSolverOptions(traceSolver);
      auto summary = recorder->getSummary();
      summary.target = "SM90";
      summary.costCoverage = target->costCoverage().str();
      summary.costModelID = report.costModelID;
      summary.costProfileSHA256 = activeCalibration ? getSM90CostProfileSHA256(*activeCalibration) : "";
      summary.staticTripCountFrequency = report.staticTripCountFrequency;
      recorder->setSummary(summary);
    };
    // A caller-owned recorder already exists: initialize its known metadata
    // even when validation rejects this invocation before creating owned output.
    initializeTrace(trace);
    ModuleOp original = getOperation();
    LayoutTraceOptions traceOptions{traceFile, traceMaxEvents, traceMaxBytes};
    std::string error;
    if (failed(LayoutProfile::validate(profileFile, statsFile, traceFile, error))) {
      original.emitError(error);
      report.status = "failed";
      report.failureKind = "profile-options";
      report.failureReason = error;
      signalPassFailure();
      return;
    }
    if (failed(traceOptions.validate(statsFile, error))) {
      original.emitError(error);
      report.status = "failed";
      report.failureKind = "trace-options";
      report.failureReason = error;
      signalPassFailure();
      return;
    }
    std::optional<LayoutTrace> ownedTrace;
    auto *activeTrace = trace;
    if (!traceFile.empty()) {
      ownedTrace.emplace(traceOptions);
      activeTrace = &*ownedTrace;
      initializeTrace(activeTrace);
    }
    // A requested statistics write is part of the transaction, not an IR side effect.
    OwningOpRef<ModuleOp> staged;
    if (!statsFile.empty() || ownedTrace || ownedProfile.enabled()) staged = cast<ModuleOp>(original->clone());
    ModuleOp module = staged ? *staged : original;
    auto result = infer(module, report, activeTrace, *target);
    report.timings["inference_total_ns"] = layoutElapsedNS(start);
    if (activeTrace) {
      auto summary = activeTrace->getSummary();
      summary.analysisStatus = failed(result) ? "failed" : "solved";
      summary.failureKind = report.failureKind;
      summary.failureReason = report.failureReason;
      if (failed(result)) summary.hasFinalCost = false;
      activeTrace->setSummary(summary);
    }
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
    if (ownedTrace && failed(ownedTrace->publish(error))) {
      report.status = "failed";
      report.failureKind = "trace-write";
      report.failureReason = error;
      original.emitError("cannot write layout trace '") << traceFile << "': " << error;
      signalPassFailure();
      return;
    }
    if (failed(ownedProfile.publish(profileFile, report.status, error))) {
      report.status = "failed";
      report.failureKind = "profile-write";
      report.failureReason = error;
      original.emitError("cannot write layout profile '") << profileFile << "': " << error;
      signalPassFailure();
      return;
    }
    if (failed(result)) { signalPassFailure(); return; }
    if (staged && !analysisOnly) {
      original->setAttrs((*staged)->getAttrs());
      original.getBodyRegion().takeBody(staged->getBodyRegion());
    }
  }
  LogicalResult infer(ModuleOp module, LayoutStatisticsReport &report,
                      LayoutTrace *activeTrace, LayoutTarget &target) {
    report.status = "failed";
    report.failureKind = "input-verification";
    if (failed(verifyNoLegacyLayoutIR(module)) || failed(verifySM90TargetBoundary(module)))
      return failure();
    report.failureKind.clear();
    FailureOr<LayoutConstraintGraph> graph =
        collectLayoutConstraints(module, target,
                                 LayoutCollectionMode::InitializeForRelaxation, &report, activeTrace);
    if (failed(graph)) {
      report.failureKind = "collection";
      return failure();
    }
    SmallVector<RelaxationRecord, 0> relaxations;
    LayoutSolveFailure why;
    auto solveStart = LayoutClock::now();
    SolverOptions solverOptions{exactCombinationLimit, beamWidth, maxExpandedStates,
                                refineBeamSolution, maxRefinementProbes};
    solverOptions.memoizeCosts = memoizeCosts;
    solverOptions.seedBeamSolution = seedBeamSolution;
    solverOptions.policy.conversion = conversionPolicy == "independent" ? ConversionPolicy::Independent : conversionPolicy == "shared-only" ? ConversionPolicy::SharedOnly : ConversionPolicy::Full;
    FailureOr<LayoutSolution> solution =
        attemptControlledRelaxation(*graph, target, solverOptions,
                                    relaxations, &why, &report, activeTrace);
    report.timings["propagation_search_retries_ns"] = layoutElapsedNS(solveStart);
    if (dumpAnalysis) printRelaxationRecords(relaxations, llvm::errs());
    if (failed(solution)) {
      recordLayoutFailure(*graph, why, report);
      return failure();
    }
    auto proofStart = LayoutClock::now();
    auto proof = verifySolvedLayoutGraph(*graph, *solution, target, module.getLoc(), solverOptions.policy.conversion);
    report.timings["verification_ns"] += layoutElapsedNS(proofStart);
    if (failed(proof)) {
      if (activeTrace) {
        auto summary = activeTrace->getSummary();
        summary.verificationStatus = "failed";
        activeTrace->setSummary(summary);
      }
      report.status = "failed";
      report.failureKind = "solved-graph-verification";
      return failure();
    }
    report.failureKind.clear(); report.failureReason.clear();
    report.failureVariable.clear(); report.failureConstraint.clear();
    if (dumpAnalysis) {
      printLayoutSolutionStatistics(*graph, *solution, target, llvm::errs());
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
    if (activeTrace) {
      auto summary = activeTrace->getSummary();
      summary.verificationStatus = "solved-graph-verified";
      activeTrace->setSummary(summary);
    }
    if (analysisOnly) return success();
    auto materializeStart = LayoutClock::now();
    auto materialized = materializeLayouts(module, *graph, *solution,
                                           authorizeInternalOptimization, &target, solverOptions.policy.conversion);
    report.timings["materialization_ns"] = layoutElapsedNS(materializeStart);
    auto verificationStart = LayoutClock::now();
    auto verified = succeeded(materialized) ? verifyMaterializedLayouts(module, target) : failure();
    report.timings["verification_ns"] += layoutElapsedNS(verificationStart);
    if (failed(materialized) || failed(verified)) {
      if (activeTrace && succeeded(materialized)) {
        auto summary = activeTrace->getSummary();
        summary.verificationStatus = "failed";
        activeTrace->setSummary(summary);
      }
      report.status = "failed";
      report.failureKind = "materialization-or-verification";
      return failure();
    }
    report.status = "materialized";
    if (activeTrace) {
      auto summary = activeTrace->getSummary();
      summary.verificationStatus = "materialized-verified";
      activeTrace->setSummary(summary);
    }
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
                                              LayoutStatisticsScope scope, LayoutTrace *trace, LayoutProfile *profile, LayoutBoundaryFrequencyMode mode,
                                 const SM90CostProfile *costProfile) {
  return std::make_unique<FriskInferLayoutsPass>(options, authorizeInternalOptimization, statistics, scope, trace, profile, mode, costProfile);
}

} // namespace mlir::frisk
