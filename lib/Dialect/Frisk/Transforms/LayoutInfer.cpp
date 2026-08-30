#include "Dialect/Frisk/Transforms/Passes.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::frisk {

#define GEN_PASS_DEF_FRISKINFERLAYOUTS
#include "Dialect/Frisk/Transforms/Passes.h.inc"

namespace {
class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  void runOnOperation() override {
    getOperation()->setAttr("frisk.layout_inference_ran",
                            UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<Pass> createFriskInferLayoutsPass() {
  return std::make_unique<FriskInferLayoutsPass>();
}

} // namespace mlir::frisk
