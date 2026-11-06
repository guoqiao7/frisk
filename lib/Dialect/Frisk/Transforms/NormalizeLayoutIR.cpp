#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "LegacyFragmentNormalization.h"
#include "PreserveStorageContracts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_NORMALIZELAYOUTIR
#include "Dialect/Frisk/Transforms/Passes.h.inc"
namespace {
class NormalizeLayoutIRPass final
    : public impl::NormalizeLayoutIRBase<NormalizeLayoutIRPass> {
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(original->clone()));
    if (failed(normalizeLegacyFragments(*clone)) ||
        failed(preserveStorageContracts(*clone)) || failed(verify(*clone))) {
      signalPassFailure();
      return;
    }
    original->setAttrs((*clone)->getAttrs());
    original.getBodyRegion().takeBody(clone->getBodyRegion());
  }
};
}
std::unique_ptr<Pass> createNormalizeLayoutIRPass() {
  return std::make_unique<NormalizeLayoutIRPass>();
}
}
