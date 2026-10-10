#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class LayoutConversionChoiceTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  LayoutConversionChoiceTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect,
                        scf::SCFDialect>();
  }
  DistributedEncodingAttr encoding(bool rotate = false) {
    SmallVector<APInt> bits;
    for (unsigned r = 0; r < 6; ++r)
      for (unsigned c = 0; c < 6; ++c)
        bits.emplace_back(1, c == (rotate ? (r + 1) % 6 : r));
    auto map = BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("register"), b.getStringAttr("lane")}),
        b.getDenseI64ArrayAttr({1,5}), b.getArrayAttr({b.getStringAttr("d")}),
        b.getDenseI64ArrayAttr({6}), DenseIntElementsAttr::get(
          RankedTensorType::get({6,6}, b.getI1Type()), bits));
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr({2,32,1,1,1}), b.getI64IntegerAttr(1));
  }
  std::string print(Operation *op) {
    std::string out; llvm::raw_string_ostream os(out); op->print(os); return out;
  }
  void fix(LayoutConstraintGraph &graph, function_ref<bool(const LayoutVar &)> source) {
    for (auto &v : graph.getVariables())
      v.candidates = {{encoding(!source(v)), kInvalidProvenanceID, 0}};
  }
};

TEST_F(LayoutConversionChoiceTest, SameBlockConsumersShareOneConversion) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @shared(%x: tensor<64xf32>) -> tensor<64xf32> {
      %sum = arith.addf %x, %x : tensor<64xf32>
      return %sum : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) { return v.value && isa<BlockArgument>(v.value); });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 2u);
  EXPECT_EQ(solution->cost.cost.conversionBytesAndSync, 512u);
  EXPECT_EQ(solution->conversions[0].sharedLeader, solution->conversions[1].sharedLeader);
  EXPECT_NE(solution->conversions[0].sharedLeader, UINT32_MAX);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  unsigned count = 0;
  module->walk([&](ConvertLayoutOp) { ++count; });
  EXPECT_EQ(count, 1u);
  module->walk([&](arith::AddFOp add) { EXPECT_EQ(add.getLhs(), add.getRhs()); });
}

TEST_F(LayoutConversionChoiceTest, CheapConstantRematerializesAtConsumer) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @constant() -> tensor<64xf32> {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      return %c : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) { return v.value && v.value.getDefiningOp<arith::ConstantOp>(); });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 1u);
  EXPECT_EQ(solution->conversions[0].resolution, EdgeResolutionKind::Rematerialize);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  unsigned conversions = 0, constants = 0;
  module->walk([&](ConvertLayoutOp) { ++conversions; });
  module->walk([&](arith::ConstantOp op) {
    ++constants;
    EXPECT_EQ(op.getValue().getType(), op.getType());
    EXPECT_DOUBLE_EQ(cast<DenseFPElementsAttr>(op.getValue()).getSplatValue<APFloat>().convertToDouble(), 3.0);
  });
  EXPECT_EQ(conversions, 0u);
  EXPECT_EQ(constants, 2u); // Original producer is deliberately retained.
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(LayoutConversionChoiceTest, SliceLimitCountsUniqueOperationsEightVersusNine) {
  auto base = encoding();
  auto replicated = DistributedEncodingAttr::get(&context, base.getMap(),
      b.getDenseI64ArrayAttr({2,32,8,1,1}), b.getI64IntegerAttr(8));
  for (unsigned size : {8u, 9u}) {
    auto module = ModuleOp::create(b.getUnknownLoc());
    OwningOpRef<ModuleOp> owner(module);
    OpBuilder builder(&context);
    builder.setInsertionPointToEnd(module.getBody());
    auto type = RankedTensorType::get({64}, b.getF32Type());
    auto func = builder.create<func::FuncOp>(b.getUnknownLoc(), "slice",
                                            b.getFunctionType({}, {type}));
    builder.setInsertionPointToStart(func.addEntryBlock());
    Value value = builder.create<arith::ConstantOp>(b.getUnknownLoc(),
        DenseElementsAttr::get(type, b.getF32FloatAttr(1.0)));
    for (unsigned i = 1; i < size; ++i)
      value = builder.create<arith::AddFOp>(b.getUnknownLoc(), value, value);
    builder.create<func::ReturnOp>(b.getUnknownLoc(), value);
    auto graph = collectLayoutConstraints(module, *target);
    ASSERT_TRUE(succeeded(graph));
    for (auto &var : graph->getVariables())
      var.candidates = {{var.functionResult ? encoding(true) : replicated,
                         kInvalidProvenanceID, 0}};
    auto solution = solveLayoutGraph(*graph, *target);
    ASSERT_TRUE(succeeded(solution));
    ASSERT_EQ(solution->conversions.size(), 1u);
    const auto &edge = solution->conversions.front();
    EXPECT_EQ(edge.resolution, size == 8 ? EdgeResolutionKind::Rematerialize
                                       : EdgeResolutionKind::Convert);
    EXPECT_EQ(edge.rematerialization.size(), size == 8 ? 8u : 0u);
    ASSERT_TRUE(succeeded(materializeLayouts(module, *graph, *solution)));
  }
}

TEST_F(LayoutConversionChoiceTest, ForgedPlacementOrSliceRollsBack) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @rollback() -> tensor<64xf32> {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      return %c : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) { return bool(v.value); });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 1u);
  auto before = print(*module);
  auto func = *module->getOps<func::FuncOp>().begin();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  auto bad = *solution;
  bad.conversions[0].insertionPoint = func;
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, bad)));
  EXPECT_EQ(before, print(*module));
  bad = *solution;
  bad.conversions[0].rematerialization.push_back(func);
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, bad)));
  EXPECT_EQ(before, print(*module));
  bad = *solution;
  ++bad.conversions[0].bytes;
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, bad)));
  EXPECT_EQ(before, print(*module));
}

TEST_F(LayoutConversionChoiceTest, MainPathCostCanOutweighConversionButTiesDoNot) {
  struct PathTarget : LayoutTarget {
    LayoutTarget &base;
    LayoutVarID result;
    Attribute preferred;
    bool equalPaths = false;
    PathTarget(LayoutTarget &base, LayoutVarID result, Attribute preferred)
        : base(base), result(result), preferred(preferred) {}
    void enumerateCandidates(const LayoutVar &v, SmallVectorImpl<LayoutCandidate> &out) const override {
      base.enumerateCandidates(v, out);
    }
    LogicalResult verifyCandidate(const LayoutVar &v, Attribute a, Location loc) const override {
      return base.verifyCandidate(v, a, loc);
    }
    FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
      CostVector cost;
      cost.instructionPathAndWork = equalPaths || a.values.lookup(result) == preferred ? 10 : 100;
      return cost;
    }
  };
  for (bool equal : {false, true}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @path(%x: tensor<64xf32>) -> tensor<64xf32> {
        return %x : tensor<64xf32>
      })mlir", &context);
    ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target);
    ASSERT_TRUE(succeeded(graph));
    LayoutVarID result = 0;
    for (auto &v : graph->getVariables()) {
      v.candidates = {{encoding(), kInvalidProvenanceID, 0}};
      if (v.functionResult) {
        result = v.id;
        v.candidates.push_back({encoding(true), kInvalidProvenanceID, 1});
      }
    }
    PathTarget ranked(*target, result, encoding(true));
    ranked.equalPaths = equal;
    auto solution = solveLayoutGraph(*graph, ranked);
    ASSERT_TRUE(succeeded(solution));
    EXPECT_EQ(solution->assignments.lookup(result), equal ? encoding() : encoding(true));
    EXPECT_EQ(solution->conversions.size(), equal ? 0u : 1u);
  }
}

TEST_F(LayoutConversionChoiceTest, ArithmeticCloneReusesTargetLayoutLeaves) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @arithmetic(%x: tensor<64xf32>) -> tensor<64xf32> {
      %s = arith.addf %x, %x : tensor<64xf32>
      return %s : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) {
    return (v.value && v.value.getDefiningOp<arith::AddFOp>()) ||
           (v.use && isa<arith::AddFOp>(v.use->getOwner()));
  });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  unsigned remat = 0;
  for (const auto &e : solution->conversions)
    remat += e.resolution == EdgeResolutionKind::Rematerialize;
  EXPECT_EQ(remat, 1u);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  auto func = *module->getOps<func::FuncOp>().begin();
  auto ret = cast<func::ReturnOp>(func.front().getTerminator());
  auto clone = ret.getOperand(0).getDefiningOp<arith::AddFOp>();
  ASSERT_TRUE(clone);
  EXPECT_EQ(clone.getLhs(), func.getArgument(0));
  EXPECT_EQ(clone.getRhs(), func.getArgument(0));
}

TEST_F(LayoutConversionChoiceTest, LeafConversionsMakeCloneMoreExpensive) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @costly(%x: tensor<64xf32>, %y: tensor<64xf32>) -> tensor<64xf32> {
      %s = arith.addf %x, %y : tensor<64xf32>
      return %s : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) { return !v.functionResult.has_value(); });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 1u);
  EXPECT_EQ(solution->conversions[0].resolution, EdgeResolutionKind::Convert);
  EXPECT_TRUE(solution->conversions[0].rematerialization.empty());
}

TEST_F(LayoutConversionChoiceTest, HardSeedPreventsCloningIntoDifferentEncoding) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @seed() -> tensor<64xf32> {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      return %c : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) { return bool(v.value); });
  for (const auto &v : graph->getVariables())
    if (v.value)
      graph->addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
          {v.id}, v.anchor, "hard-seed-test", "mandatory definition encoding", encoding());
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 1u);
  EXPECT_EQ(solution->conversions[0].resolution, EdgeResolutionKind::Convert);
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
}

TEST_F(LayoutConversionChoiceTest, CheapWhitelistRejectsUnknownRegionAndExpensiveOps) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @whitelist(%x: tensor<64xf32>) -> tensor<64xf32> {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      %a = arith.addf %c, %x : tensor<64xf32>
      %d = arith.divf %a, %x : tensor<64xf32>
      return %d : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<func::FuncOp>().begin();
  EXPECT_FALSE(isRematerializable(func));
  EXPECT_FALSE(isRematerializable(nullptr));
  for (Operation &op : func.front())
    EXPECT_EQ(isRematerializable(&op), (isa<arith::ConstantOp, arith::AddFOp>(op)));
}

TEST_F(LayoutConversionChoiceTest, SharedRematerializationIsChargedAndBuiltOnce) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @group() -> (tensor<64xf32>, tensor<64xf32>) {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      return %c, %c : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) { return bool(v.value); });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 2u);
  EXPECT_EQ(solution->cost.cost.conversionBytesAndSync, 256u);
  for (const auto &edge : solution->conversions)
    EXPECT_EQ(edge.resolution, EdgeResolutionKind::Rematerialize);
  auto leader = solution->conversions[0].sharedLeader;
  EXPECT_EQ(solution->conversions[1].sharedLeader, leader);
  // Duplicating candidate proposals cannot change edge plans or leaders.
  for (auto &v : graph->getVariables()) v.candidates.push_back(v.candidates.front());
  auto replay = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(replay));
  ASSERT_EQ(replay->conversions.size(), solution->conversions.size());
  for (auto [a, z] : llvm::zip(solution->conversions, replay->conversions))
    EXPECT_TRUE(sameLayoutConversionPlan(a, z));
  EXPECT_TRUE(succeeded(verifyLayoutConversionChoices(*graph, *solution, *target, module->getLoc())));
  auto before = print(*module);
  auto forged = *solution;
  forged.conversions[1].sharedLeader = UINT32_MAX;
  {
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, forged)));
  }
  EXPECT_EQ(before, print(*module));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  auto func = module->lookupSymbol<func::FuncOp>("group");
  auto ret = cast<func::ReturnOp>(func.front().getTerminator());
  EXPECT_EQ(ret.getOperand(0), ret.getOperand(1));
  EXPECT_TRUE(ret.getOperand(0).getDefiningOp<arith::ConstantOp>());
  unsigned constants = 0, conversions = 0;
  module->walk([&](arith::ConstantOp) { ++constants; });
  module->walk([&](ConvertLayoutOp) { ++conversions; });
  EXPECT_EQ(constants, 2u); // One clone, with original retained pre-cleanup.
  EXPECT_EQ(conversions, 0u);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(LayoutConversionChoiceTest, SharedRematerializationDominatesDistinctConsumers) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @consumers() -> (tensor<64xf32>, tensor<64xf32>) {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      %a = arith.addf %c, %c : tensor<64xf32>
      %m = arith.mulf %c, %c : tensor<64xf32>
      return %a, %m : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) {
    return v.value && v.value.getDefiningOp<arith::ConstantOp>();
  });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 4u);
  EXPECT_EQ(solution->cost.cost.conversionBytesAndSync, 256u);
  auto leader = solution->conversions.front().sharedLeader;
  auto placement = solution->conversions.front().insertionPoint;
  EXPECT_TRUE(isa<arith::AddFOp>(placement));
  for (const auto &edge : solution->conversions) {
    EXPECT_EQ(edge.resolution, EdgeResolutionKind::Rematerialize);
    EXPECT_EQ(edge.sharedLeader, leader);
    EXPECT_EQ(edge.insertionPoint, placement);
    ASSERT_EQ(edge.rematerialization.size(), 1u);
  }
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  Value shared;
  unsigned consumers = 0;
  module->walk([&](Operation *op) {
    if (!isa<arith::AddFOp, arith::MulFOp>(op)) return;
    ++consumers;
    if (!shared) shared = op->getOperand(0);
    EXPECT_EQ(op->getOperand(0), shared);
    EXPECT_EQ(op->getOperand(1), shared);
  });
  EXPECT_EQ(consumers, 2u);
  EXPECT_TRUE(shared.getDefiningOp<arith::ConstantOp>());
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(LayoutConversionChoiceTest, DifferentSourcesTargetsAndBlocksNeverShareLeader) {
  auto sources = parseSourceString<ModuleOp>(R"mlir(
    func.func @sources() -> (tensor<64xf32>, tensor<64xf32>) {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      %d = arith.constant dense<4.0> : tensor<64xf32>
      return %c, %d : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  auto targets = parseSourceString<ModuleOp>(R"mlir(
    func.func @targets() -> (tensor<64xf32>, tensor<64xf32>) {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      return %c, %c : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  auto blocks = parseSourceString<ModuleOp>(R"mlir(
    func.func @blocks(%cond: i1) -> tensor<64xf32> {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      %r = scf.if %cond -> tensor<64xf32> {
        scf.yield %c : tensor<64xf32>
      } else {
        scf.yield %c : tensor<64xf32>
      }
      return %r : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(sources); ASSERT_TRUE(targets); ASSERT_TRUE(blocks);
  for (auto module : {*sources, *targets, *blocks}) {
    auto graph = collectLayoutConstraints(module, *target);
    ASSERT_TRUE(succeeded(graph));
    fix(*graph, [](const LayoutVar &v) {
      return v.value && v.value.getDefiningOp<arith::ConstantOp>();
    });
    if (module == *targets) {
      auto normal = encoding();
      auto replicated = DistributedEncodingAttr::get(&context, normal.getMap(),
          b.getDenseI64ArrayAttr({2,32,8,1,1}), b.getI64IntegerAttr(8));
      for (auto &v : graph->getVariables())
        v.candidates = {{v.functionResult ? (*v.functionResult ? encoding(true) : normal)
                                          : replicated, kInvalidProvenanceID, 0}};
    }
    auto solution = solveLayoutGraph(*graph, *target);
    ASSERT_TRUE(succeeded(solution));
    ASSERT_EQ(solution->conversions.size(), 2u);
    EXPECT_NE(solution->conversions[0].sharedLeader, solution->conversions[1].sharedLeader);
    if (module == *blocks)
      for (const auto &edge : solution->conversions)
        EXPECT_EQ(edge.resolution, EdgeResolutionKind::Convert);
    ASSERT_TRUE(succeeded(materializeLayouts(module, *graph, *solution)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(module, *target)));
  }
}

TEST_F(LayoutConversionChoiceTest, EqualDynamicCostKeepsSharedConversionWithoutCloning) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @tie() -> (tensor<64xf32>, tensor<64xf32>) {
      %c = arith.constant dense<3.0> : tensor<64xf32>
      return %c, %c : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto layout = [&](bool swap) {
    SmallVector<APInt> bits;
    for (unsigned row = 0; row < 6; ++row)
      for (unsigned col = 0; col < 7; ++col)
        bits.emplace_back(1, col == (swap && row < 2 ? 1 - row : row));
    auto map = BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("register"), b.getStringAttr("lane")}),
        b.getDenseI64ArrayAttr({2,5}), b.getArrayAttr({b.getStringAttr("d")}),
        b.getDenseI64ArrayAttr({6}), DenseIntElementsAttr::get(
            RankedTensorType::get({6,7}, b.getI1Type()), bits));
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr({4,32,1,1,1}), b.getI64IntegerAttr(2));
  };
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  for (auto &var : graph->getVariables())
    var.candidates = {{layout(!var.value), kInvalidProvenanceID, 0}};
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 2u);
  for (const auto &edge : solution->conversions)
    EXPECT_EQ(edge.resolution, EdgeResolutionKind::Convert);
  EXPECT_EQ(solution->conversions[0].sharedLeader, solution->conversions[1].sharedLeader);
  EXPECT_EQ(solution->cost.cost.conversionBytesAndSync, 512u);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  unsigned constants = 0, conversions = 0;
  module->walk([&](arith::ConstantOp) { ++constants; });
  module->walk([&](ConvertLayoutOp) { ++conversions; });
  EXPECT_EQ(constants, 1u); EXPECT_EQ(conversions, 1u);
}

TEST_F(LayoutConversionChoiceTest, SharedSliceCannotBeForgedBeforeUnavailableLeaf) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @dependency(%x: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
      %leaf = arith.divf %x, %x : tensor<64xf32>
      %s = arith.addf %leaf, %leaf : tensor<64xf32>
      return %s, %s : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto normal = encoding();
  auto replicated = DistributedEncodingAttr::get(&context, normal.getMap(),
      b.getDenseI64ArrayAttr({2,32,8,1,1}), b.getI64IntegerAttr(8));
  Operation *leaf = nullptr;
  module->walk([&](arith::DivFOp op) { leaf = op; });
  ASSERT_TRUE(leaf);
  for (auto &v : graph->getVariables()) {
    bool sum = (v.value && v.value.getDefiningOp<arith::AddFOp>()) ||
               (v.use && isa<arith::AddFOp>(v.use->getOwner()));
    v.candidates = {{v.functionResult ? encoding(true) : sum ? replicated : normal,
                     kInvalidProvenanceID, 0}};
  }
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  auto forged = *solution;
  unsigned remat = 0;
  for (auto &edge : forged.conversions)
    if (edge.resolution == EdgeResolutionKind::Rematerialize) {
      ++remat;
      // Inserting immediately before the defining operation makes this leaf
      // unavailable, despite the otherwise valid shared source/target group.
      edge.insertionPoint = leaf;
    }
  ASSERT_EQ(remat, 2u);
  auto before = print(*module);
  {
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(materializeLayouts(*module, *graph, forged)));
  }
  EXPECT_EQ(before, print(*module));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(LayoutConversionChoiceTest, AcceptedCloneChargesAndMaterializesLeafConversionOnce) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @leaf(%x: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
      %s = arith.addf %x, %x : tensor<64xf32>
      return %s, %s : tensor<64xf32>, tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto normal = encoding();
  auto replicated = DistributedEncodingAttr::get(&context, normal.getMap(),
      b.getDenseI64ArrayAttr({2,32,8,1,1}), b.getI64IntegerAttr(8));
  for (auto &v : graph->getVariables()) {
    auto layout = v.functionResult ? encoding(true) :
        v.value && isa<BlockArgument>(v.value) ? normal : replicated;
    v.candidates = {{layout, kInvalidProvenanceID, 0}};
  }
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  unsigned remat = 0;
  for (const auto &edge : solution->conversions)
    if (edge.resolution == EdgeResolutionKind::Rematerialize) {
      ++remat;
      EXPECT_EQ(edge.bytes, 512u); // One leaf used twice, not two conversions.
    }
  EXPECT_EQ(remat, 2u);
  LayoutVar costVar;
  costVar.kind = LayoutKind::Distributed;
  costVar.shapedType = RankedTensorType::get({64}, b.getF32Type());
  LayoutConversionEdge originalEdge, leafEdge;
  auto originalCost = evaluateLayoutConversionCost(costVar, normal, replicated, originalEdge);
  auto leafCost = evaluateLayoutConversionCost(costVar, normal, encoding(true), leafEdge);
  auto cloneCost = evaluateLayoutCloneCost(costVar, encoding(true), nullptr);
  ASSERT_TRUE(succeeded(originalCost)); ASSERT_TRUE(succeeded(leafCost));
  ASSERT_TRUE(succeeded(cloneCost));
  EXPECT_EQ(solution->cost.cost.conversionBytesAndSync,
      originalCost->cost.conversionBytesAndSync + leafCost->cost.conversionBytesAndSync +
      cloneCost->cost.conversionBytesAndSync);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  auto func = *module->getOps<func::FuncOp>().begin();
  auto ret = cast<func::ReturnOp>(func.front().getTerminator());
  auto clone = ret.getOperand(0).getDefiningOp<arith::AddFOp>();
  ASSERT_TRUE(clone);
  EXPECT_EQ(ret.getOperand(0), ret.getOperand(1));
  EXPECT_EQ(clone.getLhs(), clone.getRhs());
  auto leaf = clone.getLhs().getDefiningOp<ConvertLayoutOp>();
  ASSERT_TRUE(leaf);
  EXPECT_EQ(leaf.getSource(), func.getArgument(0));
  unsigned count = 0;
  module->walk([&](ConvertLayoutOp) { ++count; });
  EXPECT_EQ(count, 2u); // Original computation input + cloned computation leaf.
  unsigned adds = 0;
  module->walk([&](arith::AddFOp) { ++adds; });
  EXPECT_EQ(adds, 2u); // Original + shared clone, not one clone per use.
}

TEST_F(LayoutConversionChoiceTest, LoopClonesStayInsideAndOuterProducerIsNotCloned) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @loop(%x: tensor<64xf32>, %n: index) -> tensor<64xf32> {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %outer = arith.constant dense<1.0> : tensor<64xf32>
      %r = scf.for %i = %zero to %n step %one iter_args(%acc = %x) -> tensor<64xf32> {
        %inner = arith.constant dense<2.0> : tensor<64xf32>
        %sum = arith.addf %inner, %outer : tensor<64xf32>
        scf.yield %sum : tensor<64xf32>
      }
      return %r : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  fix(*graph, [](const LayoutVar &v) {
    return v.value && v.value.getDefiningOp<arith::ConstantOp>();
  });
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->conversions.size(), 2u);
  unsigned remat = 0, converted = 0;
  for (const auto &edge : solution->conversions) {
    EXPECT_EQ(edge.loopDepth, 1u);
    EXPECT_TRUE(edge.insertionPoint->getParentOfType<scf::ForOp>());
    remat += edge.resolution == EdgeResolutionKind::Rematerialize;
    converted += edge.resolution == EdgeResolutionKind::Convert;
  }
  EXPECT_EQ(remat, 1u);
  EXPECT_EQ(converted, 1u);
  EXPECT_EQ(solution->cost.cost.conversionBytesAndSync, 2u * (256u + 512u));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  module->walk([&](scf::ForOp loop) {
    EXPECT_EQ(loop.getRegionIterArgs()[0].getType(), loop.getResult(0).getType());
    unsigned constants = 0, conversions = 0;
    loop.walk([&](arith::ConstantOp) { ++constants; });
    loop.walk([&](ConvertLayoutOp) { ++conversions; });
    EXPECT_EQ(constants, 2u); // inner original + inner clone; not outer clone.
    EXPECT_EQ(conversions, 1u);
  });
}
} // namespace
