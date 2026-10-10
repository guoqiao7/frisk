#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
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
                                         LayoutStatisticsReport *statistics = nullptr)
      : options(options), statsFile(statsFile.str()), statistics(statistics) {}
  SolverOptions options;
  std::string statsFile;
  LayoutStatisticsReport *statistics = nullptr;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FriskDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto start = LayoutClock::now();
    LayoutStatisticsReport local;
    auto &report = statistics ? *statistics : local;
    report = {};
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> staged(cast<ModuleOp>(original->clone()));
    PassManager pipeline(&getContext());
    pipeline.addInstrumentation(std::make_unique<LayoutTimingInstrumentation>(report));
    pipeline.addPass(createNormalizeLayoutIRPass());
    pipeline.addPass(createFriskInferLayoutsPass(options, true, &report, LayoutStatisticsScope::Pipeline));
    pipeline.addPass(std::make_unique<PreserveRootContractsPass>());
    pipeline.addPass(createOptimizeLayoutConversionsPass());
    auto target = createSM90LayoutTarget();
    bool failedPipeline = failed(pipeline.run(*staged));
    auto verificationStart = LayoutClock::now();
    if (!failedPipeline && (failed(verify(*staged)) || failed(verifyMaterializedLayouts(*staged, *target)))) {
      failedPipeline = true;
      report.failureKind = "pipeline-verification";
    }
    report.timings["verification_ns"] += layoutElapsedNS(verificationStart);
    report.timings["layout_pipeline_ns"] = layoutElapsedNS(start);
    if (failedPipeline) {
      report.status = "failed";
      if (report.failureKind.empty()) report.failureKind = "pipeline-verification";
    } else report.status = "committed";
    if (!statsFile.empty()) {
      std::string error;
      if (failed(writeLayoutStatistics(statsFile, report, error))) {
        original.emitError("cannot write layout statistics: ") << error;
        report.status = "failed";
        report.failureKind = "statistics-write";
        report.failureReason = error;
        signalPassFailure();
        return;
      }
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
                             StringRef statsFile, LayoutStatisticsReport *statistics) {
  pm.addPass(std::make_unique<LayoutPipelineTransactionPass>(options, statsFile, statistics));
}

void registerFriskLayoutPipelines() {
  struct Options : PassPipelineOptions<Options> {
    Option<std::string> statsFile{*this, "stats-file",
        llvm::cl::desc("Atomically write committed layout statistics JSON"), llvm::cl::init("")};
    Option<uint64_t> exactCombinationLimit{*this, "exact-combination-limit",
        llvm::cl::desc("Maximum exact layout combinations"), llvm::cl::init(256)};
    Option<unsigned> beamWidth{*this, "beam-width",
        llvm::cl::desc("Layout beam width"), llvm::cl::init(32)};
    Option<uint64_t> maxExpandedStates{*this, "max-expanded-states",
        llvm::cl::desc("Per-component candidate expansion budget"), llvm::cl::init(65536)};
  };
  static PassPipelineRegistration<Options> registration(
      "frisk-layout-pipeline", "Atomically normalize, infer and verify Frisk layouts",
      [](OpPassManager &pm, const Options &options) {
        buildFriskLayoutPipeline(pm, {options.exactCombinationLimit,
                                    options.beamWidth, options.maxExpandedStates}, options.statsFile);
      });
  (void)registration;
}
} // namespace mlir::frisk
