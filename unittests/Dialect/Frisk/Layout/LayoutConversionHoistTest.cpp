#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

namespace mlir::frisk::test {
namespace {
class LayoutConversionHoistTest : public testing::Test {
protected:
  MLIRContext context;
  OpBuilder b{&context};
  LayoutConversionHoistTest() {
    context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect,
                        scf::SCFDialect, gpu::GPUDialect>();
  }
  RankedTensorType type(bool swap, bool shuffle = false) {
    SmallVector<APInt> bits;
    for (unsigned i = 0; i < 7; ++i) {
      unsigned column = i;
      if (swap && (i == 0 || i == (shuffle ? 2u : 1u)))
        column = i == 0 ? (shuffle ? 2 : 1) : 0;
      for (unsigned j = 0; j < 7; ++j) bits.emplace_back(1, column == j);
    }
    auto map = BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("register"), b.getStringAttr("lane")}),
        b.getDenseI64ArrayAttr({2, 5}), b.getArrayAttr({b.getStringAttr("d0")}),
        b.getDenseI64ArrayAttr({7}), DenseIntElementsAttr::get(
            RankedTensorType::get({7, 7}, b.getI1Type()), bits));
    return RankedTensorType::get({128}, b.getF32Type(),
        DistributedEncodingAttr::get(&context, map,
            b.getDenseI64ArrayAttr({4, 32, 1, 1, 1}), b.getI64IntegerAttr(1)));
  }
  std::string print(ModuleOp module) {
    std::string s; llvm::raw_string_ostream os(s); module.print(os); return s;
  }
  LogicalResult run(ModuleOp module) {
    PassManager pm(&context); pm.addPass(createOptimizeLayoutConversionsPass());
    return pm.run(module);
  }
  OwningOpRef<ModuleOp> make(bool loop, bool shuffle = false, bool hard = false) {
    auto loc = b.getUnknownLoc();
    OwningOpRef<ModuleOp> m = ModuleOp::create(loc);
    auto a = type(false), z = type(true, shuffle);
    b.setInsertionPointToStart(m->getBody());
    auto f = b.create<func::FuncOp>(loc, "test", b.getFunctionType({a, z}, {z}));
    f.addEntryBlock(); b.setInsertionPointToStart(&f.front());
    Value result;
    if (!loop) {
      auto add = b.create<arith::AddFOp>(loc, f.getArgument(0), f.getArgument(0));
      if (hard) add->setAttr("frisk.layout", a.getEncoding());
      result = b.create<ConvertLayoutOp>(loc, z, add);
    } else {
      auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
      auto ten = b.create<arith::ConstantIndexOp>(loc, 10);
      auto one = b.create<arith::ConstantIndexOp>(loc, 1);
      auto forOp = b.create<scf::ForOp>(loc, zero, ten, one,
          ValueRange{f.getArgument(1)}, [&](OpBuilder &nested, Location l, Value, ValueRange args) {
            Value c = nested.create<ConvertLayoutOp>(l, z, f.getArgument(0));
            Value sum = nested.create<arith::AddFOp>(l, c, args[0]);
            nested.create<scf::YieldOp>(l, sum);
          });
      result = forOp.getResult(0);
    }
    b.create<func::ReturnOp>(loc, result);
    return m;
  }
};
TEST_F(LayoutConversionHoistTest, CommutesSingleUniqueInputWithoutGrowth) {
  auto m = make(false);
  auto f = *m->getOps<func::FuncOp>().begin();
  auto plain = RankedTensorType::get({128}, b.getF32Type());
  m->walk([&](ConvertLayoutOp c) { c.replaceAllUsesWith(c.getSource()); c.erase(); });
  f.getArgument(0).setType(plain);
  m->walk([&](arith::AddFOp add) { add.getResult().setType(plain); });
  f.setType(b.getFunctionType({plain, type(true)}, {plain}));
  ASSERT_TRUE(succeeded(verify(*m)));
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*m, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  for (auto &var : graph->getVariables())
    if (var.candidates.empty())
      var.candidates = {{type(false).getEncoding(), kInvalidProvenanceID, 0}};
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_TRUE(succeeded(materializeLayouts(*m, *graph, *solution,
                                         /*authorizeInternalOptimization=*/true)));
  f = m->lookupSymbol<func::FuncOp>("test");
  auto ret = cast<func::ReturnOp>(f.front().getTerminator());
  auto producer = ret.getOperand(0).getDefiningOp();
  ASSERT_EQ(producer->getAttr("frisk.inferred_result_layout"), type(false).getEncoding());
  b.setInsertionPoint(ret);
  auto boundary = b.create<ConvertLayoutOp>(ret.getLoc(), type(true), ret.getOperand(0));
  ret->setOperand(0, boundary);
  f.setType(b.getFunctionType(f.getArgumentTypes(), {type(true)}));
  ASSERT_TRUE(succeeded(verify(*m)));
  ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) { ++count; EXPECT_TRUE(isa<BlockArgument>(c.getSource())); });
  EXPECT_EQ(count, 1u);
  m->walk([&](Operation *op) { EXPECT_FALSE(op->hasAttr("frisk.inferred_result_layout")); });
  auto once = print(*m); ASSERT_TRUE(succeeded(run(*m))); EXPECT_EQ(once, print(*m));
}
TEST_F(LayoutConversionHoistTest, HoistsOnlyRegisterExchangeOutOfFor) {
  auto m = make(true); ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) { ++count; EXPECT_FALSE(c->getParentOfType<scf::ForOp>()); });
  EXPECT_EQ(count, 1u);
}
TEST_F(LayoutConversionHoistTest, CommutesTypeChangingExtension) {
  auto m = make(false);
  auto f = *m->getOps<func::FuncOp>().begin();
  auto input = RankedTensorType::get({128}, b.getF16Type(), type(false).getEncoding());
  f.getArgument(0).setType(input);
  f.setType(b.getFunctionType({input, type(true)}, {type(true)}));
  m->walk([&](arith::AddFOp add) {
    b.setInsertionPoint(add);
    auto ext = b.create<arith::ExtFOp>(add.getLoc(), type(false), f.getArgument(0));
    // Explicit internal authorization fixture; the first test exercises its
    // actual materializer origin and verifies that cleanup removes it.
    ext->setAttr("frisk.inferred_result_layout", type(false).getEncoding());
    add.replaceAllUsesWith(ext.getResult()); add.erase();
  });
  ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) {
    ++count;
    EXPECT_TRUE(isa<BlockArgument>(c.getSource()));
    EXPECT_EQ(c.getResult().getType().getElementType(), b.getF16Type());
    ASSERT_TRUE(c.getResult().hasOneUse());
    auto ext = dyn_cast<arith::ExtFOp>(*c.getResult().getUsers().begin());
    ASSERT_TRUE(ext);
    EXPECT_EQ(ext->getOperand(0), c.getResult());
  });
  EXPECT_EQ(count, 1u);
}
TEST_F(LayoutConversionHoistTest, DoesNotSpeculateShuffle) {
  auto m = make(true, true); ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) { ++count; EXPECT_TRUE(c->getParentOfType<scf::ForOp>()); });
  EXPECT_EQ(count, 1u);
}
TEST_F(LayoutConversionHoistTest, PreservesHardAnnotatedProducer) {
  auto m = make(false, false, true);
  m->walk([&](arith::AddFOp add) {
    add->setAttr("frisk.inferred_result_layout", type(false).getEncoding());
  });
  ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) { ++count; EXPECT_TRUE(c.getSource().getDefiningOp<arith::AddFOp>()); });
  EXPECT_EQ(count, 1u);
}
TEST_F(LayoutConversionHoistTest, DoesNotRetypeUnprovenEncodedProducer) {
  // A selected encoding and an original hard type cannot be distinguished by
  // absence of extra attributes. Without positive provenance it is immutable.
  auto m = make(false); ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) { ++count; EXPECT_TRUE(c.getSource().getDefiningOp<arith::AddFOp>()); });
  EXPECT_EQ(count, 1u);
}
TEST_F(LayoutConversionHoistTest, DoesNotGrowTwoIndependentConversions) {
  auto m = make(false);
  auto f = *m->getOps<func::FuncOp>().begin();
  f.front().addArgument(type(false), b.getUnknownLoc());
  f.setType(b.getFunctionType({type(false), type(true), type(false)}, {type(true)}));
  m->walk([&](arith::AddFOp add) { add->setOperand(1, f.getArgument(2)); });
  m->walk([&](arith::AddFOp add) {
    add->setAttr("frisk.inferred_result_layout", type(false).getEncoding());
  });
  ASSERT_TRUE(succeeded(run(*m)));
  unsigned count = 0;
  m->walk([&](ConvertLayoutOp c) {
    ++count; EXPECT_TRUE(c.getSource().getDefiningOp<arith::AddFOp>());
  });
  EXPECT_EQ(count, 1u);
}
TEST_F(LayoutConversionHoistTest, DoesNotCrossBarrierOrUnknownRegion) {
  for (bool region : {false, true}) {
    auto m = make(true);
    m->walk([&](scf::ForOp loop) {
      b.setInsertionPointToStart(loop.getBody());
      if (!region) b.create<gpu::BarrierOp>(b.getUnknownLoc());
      else {
        auto scope = b.create<scf::ExecuteRegionOp>(b.getUnknownLoc(), TypeRange{});
        auto *block = new Block(); scope.getRegion().push_back(block);
        OpBuilder::InsertionGuard guard(b); b.setInsertionPointToStart(block);
        b.create<scf::YieldOp>(b.getUnknownLoc());
      }
    });
    ASSERT_TRUE(succeeded(run(*m)));
    unsigned count = 0;
    m->walk([&](ConvertLayoutOp c) { ++count; EXPECT_TRUE(c->getParentOfType<scf::ForOp>()); });
    EXPECT_EQ(count, 1u);
  }
}
TEST_F(LayoutConversionHoistTest, RejectsBrokenDominanceWithoutMutation) {
  auto m = make(false);
  m->walk([&](ConvertLayoutOp c) { c->moveBefore(c.getSource().getDefiningOp()); });
  auto before = print(*m);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(run(*m))); EXPECT_EQ(before, print(*m));
}
TEST_F(LayoutConversionHoistTest, RejectsInvalidActualLayoutsTransactionally) {
  auto m = make(false);
  m->walk([&](arith::AddFOp add) { add.getResult().setType(type(true)); });
  auto before = print(*m);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(run(*m))); EXPECT_EQ(before, print(*m));
}
TEST_F(LayoutConversionHoistTest, RejectsValidMLIRWithUnresolvedActualLayouts) {
  auto m = make(false);
  auto f = *m->getOps<func::FuncOp>().begin();
  auto plain = RankedTensorType::get({128}, b.getF32Type());
  m->walk([&](ConvertLayoutOp c) { c.replaceAllUsesWith(c.getSource()); c.erase(); });
  f.getArgument(0).setType(plain);
  m->walk([&](arith::AddFOp add) { add.getResult().setType(plain); });
  f.setType(b.getFunctionType({plain, type(true)}, {plain}));
  ASSERT_TRUE(succeeded(verify(*m)));
  auto before = print(*m);
  std::string diagnostic;
  ScopedDiagnosticHandler quiet(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream os(diagnostic); d.print(os); return success();
  });
  EXPECT_TRUE(failed(run(*m))); EXPECT_EQ(before, print(*m));
  EXPECT_NE(diagnostic.find("unresolved distributed tensor encoding"), std::string::npos);
}
} // namespace
} // namespace mlir::frisk::test
