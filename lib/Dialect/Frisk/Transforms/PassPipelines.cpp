#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
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

namespace mlir::frisk {
namespace {
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
  explicit LayoutPipelineTransactionPass(SolverOptions options) : options(options) {}
  SolverOptions options;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<FriskDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> staged(cast<ModuleOp>(original->clone()));
    PassManager pipeline(&getContext());
    pipeline.addPass(createNormalizeLayoutIRPass());
    pipeline.addPass(createFriskInferLayoutsPass(options));
    pipeline.addPass(std::make_unique<PreserveRootContractsPass>());
    pipeline.addPass(createOptimizeLayoutConversionsPass());
    auto target = createSM90LayoutTarget();
    if (failed(pipeline.run(*staged)) || failed(verify(*staged)) ||
        failed(verifyMaterializedLayouts(*staged, *target))) {
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
void buildFriskLayoutPipeline(OpPassManager &pm, const SolverOptions &options) {
  pm.addPass(std::make_unique<LayoutPipelineTransactionPass>(options));
}

void registerFriskLayoutPipelines() {
  struct Options : PassPipelineOptions<Options> {
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
                                    options.beamWidth, options.maxExpandedStates});
      });
  (void)registration;
}
} // namespace mlir::frisk
