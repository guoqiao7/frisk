#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
// Follow only semantic identity conversions; a fake intermediate constant or
// disconnected kernel argument must never satisfy a dataflow assertion.
Value original(Value value) {
  while (auto convert = value.getDefiningOp<ConvertLayoutOp>())
    value = convert->getOperand(0);
  return value;
}

class AttentionLayoutIntegrationTest : public testing::Test {
protected:
  MLIRContext context;
  AttentionLayoutIntegrationTest() {
    context.loadDialect<FriskDialect, arith::ArithDialect,
        linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
        tensor::TensorDialect>();
  }
  OwningOpRef<ModuleOp> make() {
    return parseSourceFile<ModuleOp>(FRISK_ATTENTION_FIXTURE, &context);
  }
  std::string print(ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream(text) << module;
    return text;
  }
  LogicalResult infer(ModuleOp module, SolverOptions options = {}) {
    PassManager pm(&context);
    pm.enableVerifier(true);
    buildFriskLayoutPipeline(pm, options);
    return pm.run(module);
  }
  void checkTensorEncodings(ModuleOp module, bool encoded) {
    unsigned tensors = 0;
    auto check = [&](Value value) {
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        ++tensors;
        EXPECT_EQ(bool(type.getEncoding()), encoded);
        if (encoded)
          EXPECT_TRUE(isa<DistributedEncodingAttr>(type.getEncoding()));
      }
    };
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) check(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) check(arg);
    });
    EXPECT_GT(tensors, 15u);
  }
  void checkDataflow(ModuleOp module) {
    SmallVector<KernelOp> kernels;
    SmallVector<MmaOp> mmas;
    SmallVector<ReduceTensorOp> reductions;
    SmallVector<linalg::BroadcastOp> broadcasts;
    SmallVector<CopyOp> copies;
    SmallVector<TileStoreOp> stores;
    SmallVector<arith::MulFOp> scales;
    SmallVector<arith::SubFOp> centered;
    SmallVector<math::ExpOp> exponentials;
    SmallVector<arith::DivFOp> normalized;
    SmallVector<arith::TruncFOp> truncations;
    module.walk([&](KernelOp op) { kernels.push_back(op); });
    module.walk([&](MmaOp op) { mmas.push_back(op); });
    module.walk([&](ReduceTensorOp op) { reductions.push_back(op); });
    module.walk([&](linalg::BroadcastOp op) { broadcasts.push_back(op); });
    module.walk([&](CopyOp op) { copies.push_back(op); });
    module.walk([&](TileStoreOp op) { stores.push_back(op); });
    module.walk([&](arith::MulFOp op) { scales.push_back(op); });
    module.walk([&](arith::SubFOp op) { centered.push_back(op); });
    module.walk([&](math::ExpOp op) { exponentials.push_back(op); });
    module.walk([&](arith::DivFOp op) { normalized.push_back(op); });
    module.walk([&](arith::TruncFOp op) { truncations.push_back(op); });
    ASSERT_EQ(kernels.size(), 1u);
    ASSERT_EQ(mmas.size(), 2u);
    ASSERT_EQ(reductions.size(), 2u);
    ASSERT_EQ(broadcasts.size(), 2u);
    ASSERT_EQ(copies.size(), 3u);
    ASSERT_EQ(stores.size(), 1u);
    ASSERT_EQ(scales.size(), 1u);
    ASSERT_EQ(centered.size(), 1u);
    ASSERT_EQ(exponentials.size(), 1u);
    ASSERT_EQ(normalized.size(), 1u);
    ASSERT_EQ(truncations.size(), 1u);
    auto kernel = kernels.front();
    ASSERT_EQ(kernel.getNumArguments(), 4u);
    for (unsigned i = 0; i < 4; ++i) {
      auto type = cast<MemRefType>(kernel.getArgument(i).getType());
      EXPECT_EQ(type.getShape(), (ArrayRef<int64_t>{64, 16}));
      EXPECT_EQ(type.getMemorySpaceAsInt(), 1u);
      EXPECT_EQ(type.getElementType().isF32(), i == 3);
      EXPECT_EQ(type.getElementType().isF16(), i != 3);
    }
    for (unsigned i = 0; i < 3; ++i) {
      auto global = copies[i]->getOperand(0).getDefiningOp<LayoutViewOp>();
      auto shared = copies[i]->getOperand(1).getDefiningOp<LayoutViewOp>();
      ASSERT_TRUE(global);
      ASSERT_TRUE(shared);
      EXPECT_EQ(global.getSource(), kernel.getArgument(i));
      auto allocation = shared.getSource().getDefiningOp<memref::AllocOp>();
      ASSERT_TRUE(allocation);
      EXPECT_EQ(allocation.getType().getMemorySpaceAsInt(), 3u);
      EXPECT_EQ(allocation->getAttrOfType<IntegerAttr>("alignment").getInt(), 1024);
      Value mmaOperand = i == 0 ? mmas[0]->getOperand(0)
                        : i == 1 ? mmas[0]->getOperand(1) : mmas[1]->getOperand(1);
      EXPECT_EQ(mmaOperand, shared.getResult());
    }
    for (unsigned i = 0; i < 2; ++i) {
      EXPECT_EQ(mmas[i].getM(), 64);
      EXPECT_EQ(mmas[i].getN(), i == 0 ? 64 : 16);
      EXPECT_EQ(mmas[i].getK(), i == 0 ? 16 : 64);
      EXPECT_FALSE(mmas[i].getTransA());
      EXPECT_EQ(mmas[i].getTransB(), i == 0);
      auto zero = original(mmas[i]->getOperand(2)).getDefiningOp<arith::ConstantOp>();
      ASSERT_TRUE(zero);
      auto value = dyn_cast<DenseFPElementsAttr>(zero.getValue());
      ASSERT_TRUE(value && value.isSplat());
      EXPECT_TRUE(value.getSplatValue<APFloat>().isZero());
      EXPECT_EQ(reductions[i].getDim(), 1);
      EXPECT_EQ(reductions[i].getKind(), i == 0 ? "max" : "sum");
      EXPECT_EQ(cast<RankedTensorType>(reductions[i].getResult().getType()).getShape(),
                (ArrayRef<int64_t>{64}));
      EXPECT_EQ(broadcasts[i].getDimensions(), (ArrayRef<int64_t>{1}));
      EXPECT_EQ(original(broadcasts[i]->getOperand(0)), reductions[i].getResult());
    }
    EXPECT_EQ(original(scales[0]->getOperand(0)), mmas[0].getResult());
    auto scale = original(scales[0]->getOperand(1)).getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(scale);
    auto scaleValue = dyn_cast<DenseFPElementsAttr>(scale.getValue());
    ASSERT_TRUE(scaleValue && scaleValue.isSplat());
    EXPECT_EQ(scaleValue.getSplatValue<APFloat>().convertToDouble(), 0.25);
    EXPECT_EQ(original(reductions[0].getSource()), scales[0].getResult());
    EXPECT_EQ(original(centered[0]->getOperand(0)), scales[0].getResult());
    EXPECT_EQ(original(centered[0]->getOperand(1)), broadcasts[0]->getResult(0));
    EXPECT_EQ(original(exponentials[0]->getOperand(0)), centered[0].getResult());
    EXPECT_EQ(original(reductions[1].getSource()), exponentials[0].getResult());
    EXPECT_EQ(original(normalized[0]->getOperand(0)), exponentials[0].getResult());
    EXPECT_EQ(original(normalized[0]->getOperand(1)), broadcasts[1]->getResult(0));
    EXPECT_EQ(original(truncations[0]->getOperand(0)), normalized[0].getResult());
    EXPECT_TRUE(cast<RankedTensorType>(truncations[0]->getOperand(0).getType()).getElementType().isF32());
    EXPECT_TRUE(cast<RankedTensorType>(truncations[0].getType()).getElementType().isF16());
    EXPECT_EQ(original(mmas[1]->getOperand(0)), truncations[0].getResult());
    EXPECT_EQ(original(stores[0]->getOperand(0)), mmas[1].getResult());
    auto output = stores[0]->getOperand(1).getDefiningOp<LayoutViewOp>();
    ASSERT_TRUE(output);
    EXPECT_EQ(output.getSource(), kernel.getArgument(3));
  }
};

TEST_F(AttentionLayoutIntegrationTest, CompleteAttentionMaterializesAndReplays) {
  auto module = make();
  ASSERT_TRUE(module);
  checkTensorEncodings(*module, false);
  checkDataflow(*module);
  ASSERT_TRUE(succeeded(infer(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  checkTensorEncodings(*module, true);
  checkDataflow(*module);
  unsigned index = 0;
  module->walk([&](MmaOp op) {
    auto contract = op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
    ASSERT_TRUE(contract);
    EXPECT_EQ(contract.getPayload().getAs<StringAttr>("form").getValue(),
              index++ == 0 ? "ss" : "rs");
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](ReduceTensorOp op) {
    EXPECT_TRUE(op->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](CopyOp op) {
    EXPECT_TRUE(op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  auto target = createSM90LayoutTarget();
  ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  auto once = print(*module);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_EQ(once, print(*module));
  auto replay = parseSourceString<ModuleOp>(once, &context);
  ASSERT_TRUE(replay);
  ASSERT_TRUE(succeeded(verify(*replay)));
  ASSERT_TRUE(succeeded(infer(*replay)));
  EXPECT_EQ(once, print(*replay));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*replay, *target)));
  checkDataflow(*replay);
}

TEST_F(AttentionLayoutIntegrationTest, SearchBudgetOnePreservesUnencodedOriginal) {
  auto module = make();
  ASSERT_TRUE(module);
  auto originalText = print(*module);
  SolverOptions options;
  options.maxExpandedStates = 1;
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream(message) << diagnostic;
    return success();
  });
  EXPECT_TRUE(failed(infer(*module, options)));
  EXPECT_NE(message.find("search-budget-exceeded"), std::string::npos) << message;
  EXPECT_EQ(originalText, print(*module));
  checkTensorEncodings(*module, false);
  checkDataflow(*module);
}
} // namespace
