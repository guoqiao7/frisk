#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "LayoutCostObserverTestSupport.h"

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
TEST_F(LayoutConversionChoiceTest, ObservedSharedConversionAndRematerializationChargeOnlyLeader) {
  for (bool constant : {false, true}) {
    auto module = parseSourceString<ModuleOp>(constant ? R"mlir(
      func.func @shared() -> tensor<64xf32> {
        %x = arith.constant dense<3.0> : tensor<64xf32>
        %sum = arith.addf %x, %x : tensor<64xf32>
        return %sum : tensor<64xf32>
      })mlir" : R"mlir(
      func.func @shared(%x: tensor<64xf32>) -> tensor<64xf32> {
        %sum = arith.addf %x, %x : tensor<64xf32>
        return %sum : tensor<64xf32>
      })mlir", &context);
    ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target);
    ASSERT_TRUE(succeeded(graph));
    fix(*graph, [](const LayoutVar &v) {
      return v.value && (isa<BlockArgument>(v.value) || v.value.getDefiningOp<arith::ConstantOp>());
    });
    CandidateAssignment assignment;
    for (const auto &var : graph->getVariables()) assignment.values[var.id] = var.candidates[0].value;
    auto plain = chooseLayoutConversions(*graph, assignment, *target);
    TestCostObserver observer;
    auto observed = chooseLayoutConversions(*graph, assignment, *target, &observer);
    ASSERT_TRUE(succeeded(plain));
    ASSERT_TRUE(succeeded(observed));
    EXPECT_EQ(plain->cost.cost, observed->cost.cost);
    EXPECT_EQ(plain->cost.saturated, observed->cost.saturated);
    ASSERT_EQ(observed->edges.size(), 2u);
    EXPECT_TRUE(llvm::equal(plain->edges, observed->edges, sameLayoutConversionPlan));
    ASSERT_EQ(observer.parts.size(), 1u);
    EXPECT_EQ(observer.parts[0].owner, observed->edges[0].sharedLeader);
    EXPECT_EQ(observer.parts[0].kind, constant ? "rematerialization" : "conversion");
    observer.expectSum(observed->cost);
  }
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
  // The optimizer deletes the unused original: diagnostics must not retain its
  // planned arithmetic count, nor multiply the surviving clone by its uses.
  PassManager optimize(&context);
  optimize.addPass(createOptimizeLayoutConversionsPass());
  ASSERT_TRUE(succeeded(optimize.run(*module)));
  unsigned actualAdds = 0;
  module->walk([&](arith::AddFOp op) {
    ++actualAdds;
    EXPECT_EQ(cast<RankedTensorType>(op.getType()).getShape(), ArrayRef<int64_t>({64}));
  });
  EXPECT_EQ(actualAdds, 1u);
  auto diagnostics = collectLayoutIRArithmetic(*module);
  auto *entries = diagnostics.getArray("entries");
  ASSERT_EQ(entries->size(), 1u);
  EXPECT_EQ((*entries)[0].getAsObject()->getInteger("op_count"), actualAdds);
  EXPECT_EQ((*entries)[0].getAsObject()->getInteger("element_ops"), 64);
}

TEST_F(LayoutConversionChoiceTest, ActualIRDiagnosticsCountsRetainedCloneAndSharedConversion) {
  for (bool cloneExpected : {false, true}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @actual(%x: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
        %s = arith.addf %x, %x : tensor<64xf32>
        return %s, %s : tensor<64xf32>, tensor<64xf32>
      })mlir", &context);
    ASSERT_TRUE(module);
    unsigned beforeAdds = 0;
    module->walk([&](arith::AddFOp) { ++beforeAdds; });
    ASSERT_EQ(beforeAdds, 1u);
    auto graph = collectLayoutConstraints(*module, *target);
    ASSERT_TRUE(succeeded(graph));
    if (cloneExpected) {
      auto normal = encoding();
      auto replicated = DistributedEncodingAttr::get(&context, normal.getMap(),
          b.getDenseI64ArrayAttr({2,32,8,1,1}), b.getI64IntegerAttr(8));
      for (auto &v : graph->getVariables()) {
        auto layout = v.functionResult && *v.functionResult == 1 ? encoding(true) :
            v.value && isa<BlockArgument>(v.value) ? normal : replicated;
        v.candidates = {{layout, kInvalidProvenanceID, 0}};
      }
    } else {
      fix(*graph, [](const LayoutVar &v) { return v.value && isa<BlockArgument>(v.value); });
    }
    auto solution = solveLayoutGraph(*graph, *target);
    ASSERT_TRUE(succeeded(solution));
    ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
    PassManager optimize(&context);
    optimize.addPass(createOptimizeLayoutConversionsPass());
    ASSERT_TRUE(succeeded(optimize.run(*module)));
    unsigned adds = 0, conversions = 0;
    uint64_t elements = 0;
    module->walk([&](arith::AddFOp op) {
      ++adds;
      auto type = cast<RankedTensorType>(op.getType());
      ASSERT_EQ(type.getShape(), ArrayRef<int64_t>({64}));
      elements += 64; // Independent logical shape count, not plan/carrier size.
    });
    bool shared = false;
    module->walk([&](ConvertLayoutOp op) { ++conversions; shared |= !op.getResult().hasOneUse(); });
    EXPECT_EQ(adds, cloneExpected ? 2u : 1u);
    auto function = *module->getOps<func::FuncOp>().begin();
    auto ret = cast<func::ReturnOp>(function.front().getTerminator());
    auto first = ret.getOperand(0).getDefiningOp<arith::AddFOp>();
    auto second = ret.getOperand(1).getDefiningOp<arith::AddFOp>();
    ASSERT_TRUE(first); ASSERT_TRUE(second);
    EXPECT_EQ(first != second, cloneExpected);
    EXPECT_EQ(conversions, cloneExpected ? 2u : 1u);
    EXPECT_TRUE(shared);
    auto diagnostics = collectLayoutIRArithmetic(*module);
    auto *entries = diagnostics.getArray("entries");
    ASSERT_EQ(entries->size(), 1u); // ConvertLayout is not arithmetic.
    EXPECT_EQ((*entries)[0].getAsObject()->getInteger("op_count"), adds);
    EXPECT_EQ((*entries)[0].getAsObject()->getInteger("element_ops"), elements);
    auto resources = collectLayoutIRResources(*module);
    EXPECT_EQ(resources.getString("status"), "available");
    auto *scopes = resources.getArray("scopes");
    ASSERT_TRUE(scopes);
    ASSERT_EQ(scopes->size(), 1u);
    auto *scope = scopes->front().getAsObject();
    auto *values = scope->getArray("tensor_values");
    ASSERT_TRUE(values);
    // Independent oracle: inspect the final block's real definitions and uses,
    // never the conversion plan, graph variables, or collector intervals.
    Block &body = function.front();
    DenseMap<Operation *, uint64_t> positions;
    uint64_t position = 0;
    for (Operation &op : body) positions[&op] = ++position;
    SmallVector<Value> actualValues;
    for (BlockArgument argument : body.getArguments())
      if (isa<RankedTensorType>(argument.getType()) && !argument.use_empty())
        actualValues.push_back(argument);
    for (Operation &op : body)
      for (Value result : op.getResults())
        if (isa<RankedTensorType>(result.getType())) actualValues.push_back(result);
    ASSERT_EQ(values->size(), actualValues.size());
    for (auto [id, value] : llvm::enumerate(actualValues)) {
      auto *entry = (*values)[id].getAsObject();
      auto type = cast<RankedTensorType>(value.getType());
      auto layout = cast<DistributedEncodingAttr>(type.getEncoding());
      uint64_t definition = isa<BlockArgument>(value) ? 0 : positions.lookup(value.getDefiningOp());
      uint64_t lastUse = definition;
      for (OpOperand &use : value.getUses())
        lastUse = std::max(lastUse, positions.lookup(use.getOwner()));
      EXPECT_EQ(entry->getInteger("id"), id);
      EXPECT_EQ(entry->getInteger("begin"), definition);
      EXPECT_EQ(entry->getInteger("end"), lastUse);
      EXPECT_EQ(entry->getBoolean("live_in"), isa<BlockArgument>(value));
      EXPECT_EQ(entry->getInteger("slots_per_thread"), layout.getTopology()[0]);
      EXPECT_EQ(entry->getInteger("bytes_per_thread"), layout.getTopology()[0] * 4);
    }
    uint64_t peakSlots = 0, peakBytes = 0, peakAt = 0;
    for (uint64_t boundary = 0; boundary <= position; ++boundary) {
      uint64_t slots = 0, bytes = 0;
      for (Value value : actualValues) {
        auto *definition = value.getDefiningOp();
        uint64_t definedAt = definition ? positions.lookup(definition) : 0;
        if (definedAt > boundary) continue;
        bool live = definition && definedAt == boundary;
        for (OpOperand &use : value.getUses())
          live |= positions.lookup(use.getOwner()) >= boundary;
        if (!live) continue;
        auto type = cast<RankedTensorType>(value.getType());
        uint64_t carriers = cast<DistributedEncodingAttr>(type.getEncoding()).getTopology()[0];
        slots += carriers;
        bytes += carriers * 4; // All actual SSA values in this fixture are f32.
      }
      if (slots > peakSlots) { peakSlots = slots; peakAt = boundary; }
      peakBytes = std::max(peakBytes, bytes);
    }
    EXPECT_EQ(scope->getInteger("logical_live_carrier_slots_per_thread"), peakSlots);
    EXPECT_EQ(scope->getInteger("logical_live_carrier_bytes_per_thread"), peakBytes);
    EXPECT_EQ(scope->getInteger("peak_slots_at"), peakAt);
    EXPECT_EQ(scope->getInteger("logical_live_shared_root_bytes"), 0);
    EXPECT_EQ(scope->getInteger("declared_shared_root_bytes"), 0);
    auto before = print(*module);
    PassManager replay(&context);
    replay.addPass(createOptimizeLayoutConversionsPass());
    ASSERT_TRUE(succeeded(replay.run(*module)));
    EXPECT_EQ(print(*module), before);
    EXPECT_EQ(llvm::json::Value(collectLayoutIRArithmetic(*module)), llvm::json::Value(std::move(diagnostics)));
    EXPECT_EQ(llvm::json::Value(collectLayoutIRResources(*module)), llvm::json::Value(std::move(resources)));
  }
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
TEST_F(LayoutConversionChoiceTest, BothModesFiniteOracleBoundsRefinementAndPlanIdentity) {
  for (auto mode : {LayoutBoundaryFrequencyMode::Legacy, LayoutBoundaryFrequencyMode::StaticTripCount}) {
    auto model = createSM90LayoutTarget(mode);
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @loop(%x: tensor<64xf32>) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %hundred = arith.constant 100 : index
        scf.for %i = %zero to %hundred step %one {
          %sum = arith.addf %x, %x : tensor<64xf32>
        }
        return
      }
    )mlir", &context);
    ASSERT_TRUE(module);
    arith::AddFOp use;
    module->walk([&](arith::AddFOp op) { use = op; });
    ASSERT_TRUE(use);
    auto a = encoding(), z = encoding(true);
    LayoutConstraintGraph graph;
    auto p = graph.addVariable(LayoutKind::Distributed, use.getType(), "producer");
    graph.getVariable(p).value = use.getLhs();
    graph.getVariable(p).candidates = {{a}, {z}};
    for (unsigned i = 0; i < 2; ++i) {
      auto u = graph.addVariable(LayoutKind::Distributed, use.getType(), "use" + std::to_string(i));
      graph.getVariable(u).use = &use->getOpOperand(i);
      graph.getVariable(u).candidates = {{z}};
      auto c = graph.addConstraint(ConstraintKind::Convertible, ConstraintStrength::Hard,
                                  {p,u}, use, "boundary", "finite oracle");
      graph.getConstraints()[c].use = &use->getOpOperand(i);
      graph.getConstraints()[c].stableUseKey = "use/" + std::to_string(i);
    }
    for (auto source : {a,z}) {
      CandidateAssignment full; full.values[p] = source;
      full.values[1] = z; full.values[2] = z;
      auto base = model->evaluate(graph, full);
      auto boundary = chooseLayoutConversions(graph, full, *model);
      ASSERT_TRUE(succeeded(base)); ASSERT_TRUE(succeeded(boundary));
      // Independent finite oracle: two same-block consumers share one shuffle.
      EXPECT_EQ(boundary->cost.cost.conversionBytesAndSync,
          source == z ? 0u : 512u * (mode == LayoutBoundaryFrequencyMode::Legacy ? 2u : 100u));
      EXPECT_EQ(boundary->cost.cost.codeSize, source == z ? 0u : 1u);
      auto complete = addLayoutCosts(*base, boundary->cost);
      for (unsigned mask = 0; mask < 8; ++mask) {
        CandidateAssignment partial;
        for (unsigned j = 0; j < 3; ++j) if (mask & (1u << j)) partial.values[j] = full.values.lookup(j);
        auto bound = model->lowerBound(graph, partial);
        ASSERT_TRUE(succeeded(bound)); EXPECT_FALSE(complete.cost < bound->cost);
      }
    }
    auto exact = solveLayoutGraph(graph, *model);
    auto plain = solveLayoutGraph(graph, *model, {1,1,65536,false,256});
    auto refined = solveLayoutGraph(graph, *model, {1,1,65536,true,256});
    ASSERT_TRUE(succeeded(exact)); ASSERT_TRUE(succeeded(plain)); ASSERT_TRUE(succeeded(refined));
    EXPECT_EQ(exact->cost.cost.conversionBytesAndSync, 0u);
    EXPECT_FALSE(refined->cost.cost < exact->cost.cost);
    EXPECT_FALSE(plain->cost.cost < refined->cost.cost);
    graph.getVariable(p).candidates = {{a}};
    auto forced = solveLayoutGraph(graph, *model);
    ASSERT_TRUE(succeeded(forced)); ASSERT_EQ(forced->conversions.size(), 2u);
    LayoutAttemptStatistics statistics;
    finishLayoutStatistics(graph, {}, &*forced, statistics);
    EXPECT_EQ(statistics.statistics.conversionBytes, 512u);
    EXPECT_EQ(statistics.statistics.weightedConversionBytes,
        mode == LayoutBoundaryFrequencyMode::Legacy ? 1024u : 51200u);
    EXPECT_EQ(statistics.statistics.uniqueConversions, 1u);
    auto other = createSM90LayoutTarget(mode == LayoutBoundaryFrequencyMode::Legacy ?
        LayoutBoundaryFrequencyMode::StaticTripCount : LayoutBoundaryFrequencyMode::Legacy);
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(verifyLayoutConversionChoices(graph, *forced, *other, b.getUnknownLoc())));
    auto forged = *forced; forged.conversions[0].frequencyUsesProxy = !forged.conversions[0].frequencyUsesProxy;
    EXPECT_TRUE(failed(verifyLayoutConversionChoices(graph, forged, *model, b.getUnknownLoc())));
  }
}
// Deliberately give cloning more legacy byte work but less target score. This
// catches mixed objectives, including a missed hook on a remat leaf conversion.
class BoundaryObjectiveTarget final : public LayoutTarget {
  LayoutTarget &base;
  bool trafficGuard;
public:
  BoundaryObjectiveTarget(LayoutTarget &base, bool guard) : base(base), trafficGuard(guard) {}
  void enumerateCandidates(const LayoutVar &var, SmallVectorImpl<LayoutCandidate> &out) const override {
    base.enumerateCandidates(var, out);
  }
  LogicalResult verifyCandidate(const LayoutVar &var, Attribute attr, Location loc) const override {
    return base.verifyCandidate(var, attr, loc);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &assignment) const override {
    return base.evaluate(assignment);
  }
  FailureOr<CostEstimate> adjustConversionCost(const CostEstimate &raw,
                                              const LayoutConversionEdge &edge) const override {
    auto result = raw;
    result.cost.instructionPathAndWork = edge.use ? 100 : 1;
    result.features.emplace_back(edge.use ? "output-conversion" : "leaf-conversion", 1);
    return result;
  }
  FailureOr<CostEstimate> adjustCloneCost(const CostEstimate &raw, const LayoutVar &,
      Attribute, Operation *, Operation *) const override {
    auto result = raw;
    result.cost.instructionPathAndWork = 1;
    result.features.emplace_back("cloned-op", 1);
    return result;
  }
  bool requiresRematerializationTrafficReduction() const override { return trafficGuard; }
};

TEST_F(LayoutConversionChoiceTest, CalibratedBoundaryObjectiveChargesSelectedCloneOnly) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @slice() -> tensor<64xf32> {
      %c = arith.constant dense<1.0> : tensor<64xf32>
      %a = arith.addf %c, %c : tensor<64xf32>
      %b = arith.addf %a, %a : tensor<64xf32>
      return %b : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  CandidateAssignment assignment;
  for (auto &var : graph->getVariables())
    assignment.values[var.id] = encoding(bool(var.functionResult));
  BoundaryObjectiveTarget oldGuard(*target, true), calibrated(*target, false);
  auto guarded = chooseLayoutConversions(*graph, assignment, oldGuard);
  ASSERT_TRUE(succeeded(guarded));
  ASSERT_EQ(guarded->edges.size(), 1u);
  EXPECT_EQ(guarded->edges[0].resolution, EdgeResolutionKind::Convert);
  TestCostObserver observer;
  auto chosen = chooseLayoutConversions(*graph, assignment, calibrated, &observer);
  ASSERT_TRUE(succeeded(chosen));
  ASSERT_EQ(chosen->edges.size(), 1u);
  EXPECT_EQ(chosen->edges[0].resolution, EdgeResolutionKind::Rematerialize);
  EXPECT_EQ(chosen->cost.cost.instructionPathAndWork, 3u);
  EXPECT_GT(chosen->cost.cost.conversionBytesAndSync, guarded->cost.cost.conversionBytesAndSync);
  ASSERT_EQ(chosen->cost.features.size(), 1u);
  EXPECT_EQ(chosen->cost.features[0].first, "cloned-op");
  EXPECT_EQ(chosen->cost.features[0].second, 3u);
  ASSERT_EQ(observer.parts.size(), 1u);
  EXPECT_EQ(observer.parts[0].kind, "rematerialization");
  observer.expectSum(chosen->cost);
}

TEST_F(LayoutConversionChoiceTest, CalibratedBoundaryObjectiveIncludesLeafConversion) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @leaf(%x: tensor<64xf32>) -> tensor<64xf32> {
      %p = arith.mulf %x, %x : tensor<64xf32>
      return %p : tensor<64xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  CandidateAssignment assignment;
  for (auto &var : graph->getVariables())
    assignment.values[var.id] = encoding(bool(var.functionResult));
  BoundaryObjectiveTarget calibrated(*target, false);
  auto chosen = chooseLayoutConversions(*graph, assignment, calibrated);
  ASSERT_TRUE(succeeded(chosen));
  ASSERT_EQ(chosen->edges.size(), 1u);
  EXPECT_EQ(chosen->edges[0].resolution, EdgeResolutionKind::Rematerialize);
  EXPECT_EQ(chosen->cost.cost.instructionPathAndWork, 2u);
  ASSERT_EQ(chosen->cost.features.size(), 2u);
  uint64_t leaves = 0, clones = 0;
  for (const auto &[name, value] : chosen->cost.features) {
    EXPECT_NE(name, "output-conversion");
    if (name == "leaf-conversion") leaves += value;
    if (name == "cloned-op") clones += value;
  }
  EXPECT_EQ(leaves, 1u);
  EXPECT_EQ(clones, 1u);
}

} // namespace

namespace {
using JointAblationPolicy = LayoutConversionChoiceTest;
TEST_F(JointAblationPolicy, IndependentDoesNotShareOrRematerialize) {
  for (bool constant : {false, true}) {
    auto module = parseSourceString<ModuleOp>(constant ? R"mlir(
      func.func @f() -> tensor<64xf32> {
        %x = arith.constant dense<3.0> : tensor<64xf32>
        %y = arith.mulf %x, %x : tensor<64xf32>
        return %y : tensor<64xf32>
      })mlir" : R"mlir(
      func.func @f(%x: tensor<64xf32>) -> tensor<64xf32> {
        %y = arith.mulf %x, %x : tensor<64xf32>
        return %y : tensor<64xf32>
      })mlir", &context);
    ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    fix(*graph, [](const LayoutVar &v) {return v.value && (isa<BlockArgument>(v.value) || v.value.getDefiningOp<arith::ConstantOp>());});
    SolverOptions options; options.policy.conversion = ConversionPolicy::Independent;
    auto result = solveLayoutGraph(*graph,*target,options); ASSERT_TRUE(succeeded(result));
    ASSERT_EQ(result->conversions.size(),2u);
    EXPECT_NE(result->conversions[0].sharedLeader,result->conversions[1].sharedLeader);
    for (const auto &e:result->conversions) EXPECT_EQ(e.resolution,EdgeResolutionKind::Convert);
    ASSERT_TRUE(succeeded(materializeLayouts(*module,*graph,*result,false,nullptr,ConversionPolicy::Independent)));
    unsigned count=0;module->walk([&](ConvertLayoutOp){++count;});EXPECT_EQ(count,2u);
  }
}
TEST_F(JointAblationPolicy, SharedOnlyDoesNotRematerialize) {
  auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f() -> tensor<64xf32> {
    %x=arith.constant dense<3.0> : tensor<64xf32>
    %y=arith.mulf %x,%x : tensor<64xf32>
    return %y : tensor<64xf32> })mlir",&context);
  ASSERT_TRUE(module);auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));
  fix(*graph,[](const LayoutVar &v){return v.value && v.value.getDefiningOp<arith::ConstantOp>();});
  SolverOptions options;options.policy.conversion=ConversionPolicy::SharedOnly;
  auto result=solveLayoutGraph(*graph,*target,options);ASSERT_TRUE(succeeded(result));ASSERT_EQ(result->conversions.size(),2u);
  EXPECT_EQ(result->conversions[0].sharedLeader,result->conversions[1].sharedLeader);
  for(const auto &e:result->conversions)EXPECT_EQ(e.resolution,EdgeResolutionKind::Convert);
}
TEST_F(JointAblationPolicy, CrossPolicyVerificationRejects) {
  auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%x:tensor<64xf32>) -> tensor<64xf32> {
    %y=arith.mulf %x,%x : tensor<64xf32>
    return %y : tensor<64xf32> })mlir",&context);
  ASSERT_TRUE(module);auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));
  fix(*graph,[](const LayoutVar &v){return v.value && isa<BlockArgument>(v.value);});
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
  for(auto policy:{ConversionPolicy::Independent,ConversionPolicy::Full}) {
    auto copy=*graph;SolverOptions options;options.policy.conversion=policy;
    auto result=solveLayoutGraph(copy,*target,options);ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(succeeded(verifyLayoutConversionChoices(copy,*result,*target,module->getLoc(),policy)));
    auto opposite=policy==ConversionPolicy::Independent?ConversionPolicy::Full:ConversionPolicy::Independent;
    EXPECT_TRUE(failed(materializeLayouts(*module,copy,*result,false,nullptr,opposite)));
    EXPECT_TRUE(failed(verifyLayoutConversionChoices(copy,*result,*target,module->getLoc(),opposite)));
    result->policy.conversion=opposite;
    EXPECT_TRUE(failed(verifyLayoutConversionChoices(copy,*result,*target,module->getLoc(),opposite)));
  }
}
TEST_F(JointAblationPolicy, IdentityDoesNotCreateWork) {
  auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%x:tensor<64xf32>) -> tensor<64xf32> {return %x : tensor<64xf32>})mlir",&context);
  auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));fix(*graph,[](const LayoutVar &){return true;});
  SolverOptions options;options.policy.conversion=ConversionPolicy::Independent;
  auto result=solveLayoutGraph(*graph,*target,options);ASSERT_TRUE(succeeded(result));EXPECT_TRUE(result->conversions.empty());
}
TEST_F(JointAblationPolicy, DefaultFullUnchanged) {
  auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f() -> tensor<64xf32> {
    %x=arith.constant dense<3.0> : tensor<64xf32>
    return %x : tensor<64xf32> })mlir",&context);
  auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));fix(*graph,[](const LayoutVar &v){return v.value && v.value.getDefiningOp<arith::ConstantOp>();});
  auto result=solveLayoutGraph(*graph,*target);ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->policy.conversion,ConversionPolicy::Full);ASSERT_EQ(result->conversions.size(),1u);
  EXPECT_EQ(result->conversions[0].resolution,EdgeResolutionKind::Rematerialize);
}
}
