#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "PreserveStorageContracts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassInstrumentation.h"

namespace mlir::frisk {
namespace {
class LayoutTimingInstrumentation final : public PassInstrumentation {
public:
  explicit LayoutTimingInstrumentation(LayoutStatisticsReport &report) : report(report) {}
  void runBeforePass(Pass *, Operation *) override { start = LayoutClock::now(); }
  void runAfterPass(Pass *pass, Operation *) override {
    auto name = pass->getArgument();
    if (name == "frisk-normalize-layout-ir") report.timings["normalization_ns"] += layoutElapsedNS(start);
    if (name == "frisk-optimize-layout-conversions") report.timings["optimization_ns"] += layoutElapsedNS(start);
  }
  void runAfterPassFailed(Pass *pass, Operation *) override {
    auto name = pass->getArgument();
    if (name == "frisk-infer-layouts" && !report.failureKind.empty()) return;
    report.failureKind = name == "frisk-normalize-layout-ir" ? "normalization"
        : name == "frisk-optimize-layout-conversions" ? "conversion-optimization"
        : "storage-contract-preservation";
  }
private:
  LayoutStatisticsReport &report;
  LayoutClock::time_point start;
};
class PreserveRootContractsPass final
    : public PassWrapper<PreserveRootContractsPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreserveRootContractsPass)
  void runOnOperation() override {
    if (failed(preserveStorageContracts(getOperation()))) signalPassFailure();
  }
};

class LayoutPipelineTransactionPass final
    : public PassWrapper<LayoutPipelineTransactionPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutPipelineTransactionPass)
  LayoutPipelineTransactionPass() = default;
  explicit LayoutPipelineTransactionPass(SolverOptions options, StringRef statsFile = {},
                                         LayoutStatisticsReport *statistics = nullptr,
                                         LayoutTraceOptions traceOptions = {}, StringRef profileFile = {},
                                         bool irDiagnostics = false, LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy,
                                         StringRef costProfileFile = {})
      : options(options), statsFile(statsFile.str()), statistics(statistics), traceOptions(std::move(traceOptions)), profileFile(profileFile.str()), irDiagnostics(irDiagnostics), mode(mode), costProfileFile(costProfileFile.str()) {}
  LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy;
  SolverOptions options;
  std::string statsFile;
  LayoutStatisticsReport *statistics = nullptr;
  LayoutTraceOptions traceOptions;
  std::string profileFile;
  std::string costProfileFile;
  bool irDiagnostics = false;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FriskDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    LayoutProfile profile(!profileFile.empty());
    LayoutProfile::Invocation profileInvocation(&profile);
    auto start = LayoutClock::now();
    LayoutStatisticsReport local;
    auto &report = statistics ? *statistics : local;
    report = {};
    std::optional<SM90CostProfile> calibration;
    if (!costProfileFile.empty()) {
      if (costProfileFile == statsFile || costProfileFile == traceOptions.file || costProfileFile == profileFile) {
        getOperation().emitError("cost-profile conflicts with output file"); signalPassFailure(); return;
      }
      auto loaded = loadSM90CostProfileFile(costProfileFile);
      if (!loaded) {
        getOperation().emitError("invalid cost-profile: ") << llvm::toString(loaded.takeError());
        report.status = "failed"; report.failureKind = "cost-profile";
        signalPassFailure(); return;
      }
      calibration = std::move(*loaded);
    }
    // Load once: all pipeline stages use this same immutable profile value.
    auto target = calibration ? createSM90LayoutTarget(mode, *calibration) : createSM90LayoutTarget(mode);
    report.costModelID = normalizeLayoutCostModelID(target->costModelID()).str();
    report.staticTripCountFrequency = mode == LayoutBoundaryFrequencyMode::StaticTripCount;
    ModuleOp original = getOperation();
    std::string error;
    if (failed(LayoutProfile::validate(profileFile, statsFile, traceOptions.file, error))) {
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
    std::optional<LayoutTrace> trace;
    if (!traceOptions.file.empty()) trace.emplace(traceOptions);
    if (trace) {
      trace->setSolverOptions(options);
      auto summary = trace->getSummary();
      summary.target = "SM90";
      summary.costCoverage = target->costCoverage().str();
      summary.costModelID = report.costModelID;
      summary.costProfileSHA256 = calibration ? getSM90CostProfileSHA256(*calibration) : "";
      summary.staticTripCountFrequency = report.staticTripCountFrequency;
      trace->setSummary(summary);
    }
    OwningOpRef<ModuleOp> staged(cast<ModuleOp>(original->clone()));
    PassManager pipeline(&getContext());
    pipeline.addInstrumentation(std::make_unique<LayoutTimingInstrumentation>(report));
    pipeline.addPass(createNormalizeLayoutIRPass());
    pipeline.addPass(createFriskInferLayoutsPass(options, true, &report, LayoutStatisticsScope::Pipeline,
                                               trace ? &*trace : nullptr, &profile, mode, calibration ? &*calibration : nullptr));
    pipeline.addPass(std::make_unique<PreserveRootContractsPass>());
    if (options.policy.conversion == ConversionPolicy::Full)
      pipeline.addPass(createOptimizeLayoutConversionsPass(mode, calibration ? &*calibration : nullptr));
    bool failedPipeline = failed(pipeline.run(*staged));
    auto verificationStart = LayoutClock::now();
    bool failedFinalVerification = false;
    if (!failedPipeline && (failed(verify(*staged)) || failed(verifyMaterializedLayouts(*staged, *target)))) {
      failedPipeline = true;
      failedFinalVerification = true;
      report.failureKind = "pipeline-verification";
    }
    report.timings["verification_ns"] += layoutElapsedNS(verificationStart);
    report.timings["layout_pipeline_ns"] = layoutElapsedNS(start);
    if (failedPipeline) {
      report.status = "failed";
      if (report.failureKind.empty()) report.failureKind = "pipeline-verification";
    } else {
      report.status = "committed";
      if (irDiagnostics) {
        report.actualIRDiagnostics = llvm::json::Object{
            {"stage", options.policy.conversion == ConversionPolicy::Full ? "post-conversion-optimization" : "post-inference-selected-conversion-policy"},
            {"arithmetic", collectLayoutIRArithmetic(*staged)},
            {"resources", collectLayoutIRResources(*staged)}};
        if (trace) trace->recordEvent([&] {
          return llvm::json::Object{{"event", "actual-ir-diagnostics"},
              {"diagnostics", report.actualIRDiagnostics}};
        });
      }
    }
    if (trace) {
      auto summary = trace->getSummary();
      summary.analysisStatus = failedPipeline ? "failed" : "solved";
      if (failedFinalVerification) summary.verificationStatus = "failed";
      else if (!failedPipeline) summary.verificationStatus = "pipeline-verified";
      summary.failureKind = report.failureKind;
      summary.failureReason = report.failureReason;
      if (failedPipeline) summary.hasFinalCost = false;
      trace->setSummary(summary);
    }
    if (!statsFile.empty()) {
      std::string error;
      if (failed(writeLayoutStatistics(statsFile, report, error))) {
        original.emitError("cannot write layout statistics: ") << error;
        report.status = "failed";
        report.failureKind = "statistics-write";
        report.failureReason = error;
        report.actualIRDiagnostics = nullptr;
        signalPassFailure();
        return;
      }
    }
    if (trace && failed(trace->publish(error))) {
      original.emitError("cannot write layout trace '") << traceOptions.file << "': " << error;
      report.status = "failed";
      report.failureKind = "trace-write";
      report.failureReason = error;
      report.actualIRDiagnostics = nullptr;
      signalPassFailure();
      return;
    }
    if (failed(profile.publish(profileFile, report.status, error))) {
      original.emitError("cannot write layout profile '") << profileFile << "': " << error;
      report.status = "failed";
      report.failureKind = "profile-write";
      report.failureReason = error;
      report.actualIRDiagnostics = nullptr;
      signalPassFailure();
      return;
    }
    if (failedPipeline) {
      signalPassFailure();
      return;
    }
    original->setAttrs((*staged)->getAttrs());
    original.getBodyRegion().takeBody(staged->getBodyRegion());
  }
};
} // namespace

void buildFriskLayoutPipeline(OpPassManager &pm) {
  pm.addPass(std::make_unique<LayoutPipelineTransactionPass>());
}
void buildFriskLayoutPipeline(OpPassManager &pm, const SolverOptions &options,
                             StringRef statsFile, LayoutStatisticsReport *statistics,
                             LayoutTraceOptions traceOptions, StringRef profileFile,
                             bool irDiagnostics, LayoutBoundaryFrequencyMode mode, StringRef costProfileFile) {
  pm.addPass(std::make_unique<LayoutPipelineTransactionPass>(options, statsFile, statistics, std::move(traceOptions), profileFile, irDiagnostics, mode, costProfileFile));
}

void registerFriskLayoutPipelines() {
  struct Options : PassPipelineOptions<Options> {
    Option<std::string> costProfileFile{*this, "cost-profile", llvm::cl::desc("Explicit SM90 calibrated cost profile JSON"), llvm::cl::init("")};
    Option<std::string> boundaryFrequency{*this, "boundary-frequency", llvm::cl::desc("Boundary frequency: legacy or static-trip-count"), llvm::cl::init("legacy")};
    Option<bool> irDiagnostics{*this, "ir-diagnostics", llvm::cl::desc("Collect post-optimization static logical arithmetic and resource diagnostics"), llvm::cl::init(false)};
    Option<bool> memoizeCosts{*this, "memoize-costs", llvm::cl::desc("Reuse bounded graph-local pure cost queries"), llvm::cl::init(true)};
    Option<bool> seedBeamSolution{*this, "seed-beam-solution", llvm::cl::desc("Budgeted coherent-layout witnesses before beam search"), llvm::cl::init(false)};
    Option<std::string> conversionPolicy{*this, "conversion-policy", llvm::cl::desc("Conversion policy: full, shared-only or independent"), llvm::cl::init("full")};
    Option<std::string> profileFile{*this, "profile-file", llvm::cl::desc("Invocation-local inclusive layout profile JSON"), llvm::cl::init("")};
    Option<std::string> traceFile{*this, "trace-file", llvm::cl::desc("Layout decision trace JSON"), llvm::cl::init("")};
    Option<uint64_t> traceMaxEvents{*this, "trace-max-events", llvm::cl::desc("Maximum trace events"), llvm::cl::init(4096)};
    Option<uint64_t> traceMaxBytes{*this, "trace-max-bytes", llvm::cl::desc("Maximum trace bytes (at least 4096)"), llvm::cl::init(8388608)};
    Option<std::string> statsFile{*this, "stats-file",
        llvm::cl::desc("Atomically write committed layout statistics JSON"), llvm::cl::init("")};
    Option<uint64_t> exactCombinationLimit{*this, "exact-combination-limit",
        llvm::cl::desc("Maximum exact layout combinations"), llvm::cl::init(256)};
    Option<unsigned> beamWidth{*this, "beam-width",
        llvm::cl::desc("Layout beam width"), llvm::cl::init(32)};
    Option<uint64_t> maxExpandedStates{*this, "max-expanded-states",
        llvm::cl::desc("Per-component candidate expansion budget"), llvm::cl::init(65536)};
    Option<bool> refineBeamSolution{*this, "refine-beam-solution",
        llvm::cl::desc("Improve complete beam solutions"), llvm::cl::init(true)};
    Option<uint64_t> maxRefinementProbes{*this, "max-refinement-probes",
        llvm::cl::desc("Refinement probes within the total budget"), llvm::cl::init(256)};
  };
  static PassPipelineRegistration<Options> registration(
      "frisk-layout-pipeline", "Atomically normalize, infer and verify Frisk layouts",
      [](OpPassManager &pm, const Options &options) {
        if (options.boundaryFrequency != "legacy" && options.boundaryFrequency != "static-trip-count") {
          auto pass = createFriskInferLayoutsPass();
          (void)pass->initializeOptions("boundary-frequency=" + options.boundaryFrequency, [](const Twine &) { return failure(); });
          pm.addPass(std::move(pass));
          return;
        }
        SolverOptions solverOptions{options.exactCombinationLimit,
                                    options.beamWidth, options.maxExpandedStates,
                                    options.refineBeamSolution, options.maxRefinementProbes};
        solverOptions.memoizeCosts = options.memoizeCosts;
        solverOptions.seedBeamSolution = options.seedBeamSolution;
        if (options.conversionPolicy != "full" && options.conversionPolicy != "shared-only" && options.conversionPolicy != "independent") llvm::report_fatal_error("unknown conversion-policy");
        solverOptions.policy.conversion = options.conversionPolicy == "independent" ? ConversionPolicy::Independent : options.conversionPolicy == "shared-only" ? ConversionPolicy::SharedOnly : ConversionPolicy::Full;
        buildFriskLayoutPipeline(pm, solverOptions, options.statsFile, nullptr,
                                 {options.traceFile, options.traceMaxEvents, options.traceMaxBytes}, options.profileFile, options.irDiagnostics, options.boundaryFrequency == "legacy" ? LayoutBoundaryFrequencyMode::Legacy : LayoutBoundaryFrequencyMode::StaticTripCount, options.costProfileFile);
      });
  (void)registration;
}
} // namespace mlir::frisk
