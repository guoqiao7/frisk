#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "LayoutCostObserverTestSupport.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
TEST(SM90CostModelTest, CompleteModelIdentityIsIndependentOfCoverage) {
  auto target = createSM90LayoutTarget();
  EXPECT_EQ(getSM90LegacyCostModelID(), "sm90-layout-lex9-legacy-v1");
  EXPECT_EQ(target->costModelID(), getSM90LegacyCostModelID());
  EXPECT_NE(target->costCoverage(), target->costModelID());
  constexpr StringLiteral legacyCoverage =
      "bounded-owner-proof-v1 with cta-staging-upper-bound-v1 fallback; "
      "pure-slice-byte-work-v1; loop-depth proxy; register/replication slots; shared root spans; "
      "sm90-shape-work-v1; warp-issue-32B-sectors-v1 (unknown base phase upper bound); "
      "shared-128B-phase-32x4B-distinct-word-conflicts-v2; unmodeled=latency,cache,overlap";
  EXPECT_EQ(target->costCoverage(), legacyCoverage);
  auto staticTarget = createSM90LayoutTarget(LayoutBoundaryFrequencyMode::StaticTripCount);
  StringRef staticCoverage = staticTarget->costCoverage();
  EXPECT_TRUE(staticCoverage.contains("static signed trip counts"));
  EXPECT_TRUE(staticCoverage.contains("unknown-loop factor-2 proxy"));
  EXPECT_FALSE(staticCoverage.contains("loop-depth proxy"));
  staticTarget.reset();
  // Returned StringRef refers to static storage, independent of target lifetime.
  EXPECT_TRUE(staticCoverage.contains("static signed trip counts"));
  for (uint64_t maxBytes : {uint64_t{4096}, uint64_t{8388608}}) {
    LayoutConstraintGraph graph;
    LayoutTrace trace({"unused.json", 1, maxBytes});
    ASSERT_TRUE(succeeded(solveLayoutGraph(graph, *target, {}, nullptr, nullptr, &trace)));
    auto parsed = llvm::json::parse(trace.serialize());
    ASSERT_TRUE(bool(parsed));
    EXPECT_EQ(parsed->getAsObject()->getString("cost_model_id"), getSM90LegacyCostModelID());
    EXPECT_EQ(parsed->getAsObject()->getInteger("schema_version"), 1);
  }
}
// Small address fixtures exercise the actual graph scorer and alias footprint,
// without paying for Copy candidate enumeration. The binding states the already
// selected vector width; hardware-contract legality has separate tests.
class SM90AddressCostTest : public testing::Test {
protected:
  MLIRContext context;
  SM90AddressCostTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  FailureOr<CostEstimate> score(unsigned bits = 32, unsigned vectorBytes = 1,
      unsigned stride = 1, unsigned alignment = 32, bool global = false,
      bool broadcast = false, AccessKind access = AccessKind::Read,
      unsigned warps = 1, unsigned copies = 1) {
    Builder b(&context);
    unsigned registers = std::max(1u, vectorBytes * 8 / bits);
    unsigned logical = broadcast ? std::max(2u, registers) : 32 * registers;
    std::string type = "memref<" + std::to_string(logical) + "xi" +
        std::to_string(bits) + ",strided<[" + std::to_string(stride) +
        "]>," + (global ? "1" : "3") + ">";
    auto module = parseSourceString<ModuleOp>("func.func @f() { %a = memref.alloc() {alignment = " +
        std::to_string(alignment) + " : i64} : " + type + " return }", &context);
    if (!module) return failure();
    memref::AllocOp allocation;
    module->walk([&](memref::AllocOp op) { allocation = op; });
    auto alias = analyzeStorageAlias(allocation.getResult());
    if (failed(alias)) return failure();
    auto storage = buildRootLinearStorageCandidate(*alias);
    if (failed(storage)) return failure();
    unsigned regBits = llvm::Log2_64(registers);
    unsigned outBits = llvm::Log2_64(logical);
    SmallVector<APInt> matrix;
    for (unsigned r = 0; r < outBits; ++r)
      for (unsigned c = 0; c < regBits + 5; ++c)
        matrix.push_back(APInt(1, r == c));
    auto map = BitLinearLayoutMapAttr::get(&context,
        regBits ? b.getStrArrayAttr({"register", "lane"}) : b.getStrArrayAttr({"lane"}),
        regBits ? b.getDenseI64ArrayAttr({int64_t(regBits), 5}) : b.getDenseI64ArrayAttr({5}),
        b.getStrArrayAttr({"x"}), b.getDenseI64ArrayAttr({int64_t(outBits)}),
        DenseIntElementsAttr::get(RankedTensorType::get(
            {int64_t(outBits), int64_t(regBits + 5)}, b.getI1Type()), matrix));
    auto execution = DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr({registers, 32, warps, 1, 1}),
        b.getI64IntegerAttr(warps * 32 * registers / logical));
    LayoutConstraintGraph graph;
    auto s = graph.addVariable(LayoutKind::Storage, allocation.getType(), "s");
    graph.getVariable(s).storageAlias = *alias;
    auto e = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({logical}, b.getIntegerType(bits)), "e", allocation);
    graph.getVariable(e).operationExecution = OperationExecutionBinding{};
    auto p = graph.addVariable(LayoutKind::Instruction, allocation.getType(), "p");
    auto path = b.getStringAttr("vector");
    auto binding = b.getDictionaryAttr({b.getNamedAttr("path", path),
        b.getNamedAttr("vector_bytes", b.getI64IntegerAttr(vectorBytes))});
    auto instruction = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, {s, e, p}, allocation, "test", "selected width");
    graph.getConstraints()[instruction].instruction = InstructionLayoutContract{
        allocation, {}, {{{*storage, execution, path}, binding}}};
    for (unsigned i = 0; i < copies; ++i) {
      auto id = graph.addConstraint(ConstraintKind::StorageAccess,
          ConstraintStrength::Hard, {s, e}, allocation, "test", "address access");
      graph.getConstraints()[id].access = access;
    }
    CandidateAssignment assignment;
    assignment.values[s] = *storage;
    assignment.values[e] = execution;
    assignment.values[p] = path;
    auto plain = evaluateSM90LayoutCost(graph, assignment);
    TestCostObserver observer;
    auto observed = createSM90LayoutTarget()->evaluate(graph, assignment, &observer);
    EXPECT_EQ(succeeded(plain), succeeded(observed));
    if (succeeded(plain) && succeeded(observed)) {
      EXPECT_EQ(plain->cost, observed->cost);
      EXPECT_EQ(plain->saturated, observed->saturated);
      observer.expectSum(*observed);
    }
    return plain;
  }
};
TEST_F(SM90AddressCostTest, ContiguousWideVectorsHaveIndependentBankPhases) {
  for (unsigned width : {8u, 16u}) {
    auto cost = score(32, width);
    ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(cost->cost.bankConflictDegree, 0u) << width;
    EXPECT_EQ(cost->cost.instructionPathAndWork, 1u) << width;
  }
}
TEST_F(SM90AddressCostTest, NativeWideScalarHasIndependentBankPhases) {
  auto cost = score(64);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.bankConflictDegree, 0u);
  EXPECT_EQ(cost->cost.instructionPathAndWork, 1u);
}
TEST_F(SM90AddressCostTest, SameBankDistinctWordsConflictButReadBroadcastDoesNot) {
  auto conflict = score(32, 1, 32);
  auto broadcast = score(32, 1, 1, 32, false, true);
  ASSERT_TRUE(succeeded(conflict));
  ASSERT_TRUE(succeeded(broadcast));
  EXPECT_EQ(conflict->cost.bankConflictDegree, 31u);
  EXPECT_EQ(broadcast->cost.bankConflictDegree, 0u);
}
TEST_F(SM90AddressCostTest, WideConflictIsCountedWithinEachPhase) {
  auto cost = score(64, 1, 16);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.bankConflictDegree, 30u); // Two 16-lane, 16-way phases.
}
TEST_F(SM90AddressCostTest, GlobalSectorsUseExactAddressesAndWorstUnknownPhase) {
  auto contiguous = score(32, 1, 1, 32, true);
  auto strided = score(32, 1, 8, 32, true);
  auto unknown = score(32, 1, 1, 1, true);
  ASSERT_TRUE(succeeded(contiguous));
  ASSERT_TRUE(succeeded(strided));
  ASSERT_TRUE(succeeded(unknown));
  EXPECT_EQ(contiguous->cost.memoryTransactions, 4u);
  EXPECT_EQ(strided->cost.memoryTransactions, 32u);
  EXPECT_EQ(unknown->cost.memoryTransactions, 5u);
}
TEST_F(SM90AddressCostTest, WarpsAndAccessesAreNeverMerged) {
  auto cost = score(32, 1, 32, 32, false, false, AccessKind::Read, 2, 2);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.bankConflictDegree, 124u);
  EXPECT_EQ(cost->cost.instructionPathAndWork, 4u);
  EXPECT_FALSE(cost->saturated);
}
TEST_F(SM90AddressCostTest, PackedWritesCountDistinctWordsNotSubwordWriters) {
  auto cost = score(16, 1, 1, 32, false, false, AccessKind::Write);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.bankConflictDegree, 0u);
}
TEST_F(SM90AddressCostTest, OversizedIntegerRetainsConservativeNonzeroProxy) {
  // No native 256-byte scalar instruction is implied. Keep arbitrary legal IR
  // types scoreable with a conservative one-lane batch instead of dividing by
  // zero or rejecting an otherwise legal scalar fallback.
  auto cost = score(2048);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_GT(cost->cost.bankConflictDegree, 0u);
  EXPECT_FALSE(cost->saturated);
}
} // namespace
TEST(SM90CostModelTest, CopyHasNonzeroPathAndTransactionWork) {
  MLIRContext context;
  context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @copy(%s:memref<8x16xf32,1>, %d:memref<8x16xf32,3>) {
      %sv=frisk.layout_view %s : memref<8x16xf32,1> -> memref<8x16xf32,1>
      %dv=frisk.layout_view %d : memref<8x16xf32,3> -> memref<8x16xf32,3>
      "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:8,16>,dstExtents=array<i64:8,16>}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<8x16xf32,1>,memref<8x16xf32,3>)->()
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_GT(solution->cost.cost.instructionPathAndWork, 0u);
  EXPECT_GT(solution->cost.cost.memoryTransactions, 0u);
  SM90CostProfile calibration;
  calibration.coefficients.fill(1);
  calibration.provenance = "copy-publication-barrier-test-only";
  auto calibratedTarget = createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy, calibration);
  CandidateAssignment complete;
  complete.values = solution->assignments;
  TestCostObserver calibratedObserver;
  auto calibrated = calibratedTarget->evaluate(*graph, complete, &calibratedObserver);
  ASSERT_TRUE(succeeded(calibrated));
  calibratedObserver.expectSum(*calibrated);
  uint64_t barriers = 0;
  for (const auto &[name, value] : calibrated->features)
    if (name == "synchronization_barriers") barriers += value;
  EXPECT_GE(barriers, 1u);
  EXPECT_LE(barriers, 2u);
  // Same static operation, different physical ownership/address groupings.
  // Exercise the graph target scorer rather than a disconnected arithmetic API.
  uint64_t bestSectors = UINT64_MAX, worstSectors = 0;
  uint64_t bestBanks = UINT64_MAX, worstBanks = 0;
  for (const auto &c : graph->getConstraints()) {
    if (!c.instruction) continue;
    for (const auto &tuple : c.instruction->tuples) {
      auto binding = cast<DictionaryAttr>(tuple.binding);
      if (binding.getAs<StringAttr>("path").getValue() != "vector") continue;
      CandidateAssignment assignment;
      for (auto [id, value] : llvm::zip(c.vars, tuple.encodings)) assignment.values[id] = value;
      auto cost = target->evaluate(*graph, assignment);
      ASSERT_TRUE(succeeded(cost));
      TestCostObserver observer;
      auto observed = target->evaluate(*graph, assignment, &observer);
      ASSERT_TRUE(succeeded(observed));
      EXPECT_EQ(cost->cost, observed->cost);
      EXPECT_EQ(cost->saturated, observed->saturated);
      observer.expectSum(*observed);
      bestSectors = std::min(bestSectors, cost->cost.memoryTransactions);
      worstSectors = std::max(worstSectors, cost->cost.memoryTransactions);
      bestBanks = std::min(bestBanks, cost->cost.bankConflictDegree);
      worstBanks = std::max(worstBanks, cost->cost.bankConflictDegree);
      CandidateAssignment partial;
      // Assign both storage endpoints and execution, leaving only the Copy
      // instruction path unset. A partial bound must not assume a path width.
      for (auto [id, value] : llvm::zip(c.vars, tuple.encodings))
        if (graph->getVariable(id).kind != LayoutKind::Instruction)
          partial.values[id] = value;
      auto bound = target->lowerBound(*graph, partial);
      ASSERT_TRUE(succeeded(bound));
      EXPECT_FALSE(cost->cost < bound->cost);
    }
  }
  EXPECT_LT(bestSectors, worstSectors);
  EXPECT_LT(bestBanks, worstBanks);
}

namespace {
SM90CostProfile unitProfile() {
  SM90CostProfile profile;
  profile.coefficients.fill(1);
  profile.provenance = "unit-test-only-not-calibration";
  return profile;
}
TEST(SM90CalibratedCostTest, StrictProfileRoundtripAndParameterIdentity) {
  auto profile = unitProfile();
  auto text = serializeSM90CostProfile(profile);
  auto parsed = parseSM90CostProfile(text);
  ASSERT_TRUE(bool(parsed));
  EXPECT_EQ(serializeSM90CostProfile(*parsed), text);
  EXPECT_EQ(getSM90CostProfileSHA256(*parsed).size(), 64u);
  auto id = getSM90CalibratedCostModelID(profile);
  EXPECT_EQ(normalizeLayoutCostModelID(id), id);
  auto staticID = getSM90CalibratedCostModelID(profile, LayoutBoundaryFrequencyMode::StaticTripCount);
  EXPECT_NE(id, staticID);
  EXPECT_EQ(normalizeLayoutCostModelID(staticID), staticID);
  profile.coefficients[0]++;
  EXPECT_NE(getSM90CalibratedCostModelID(profile), id);
  for (auto bad : {"{}", "[]", "{\"schema_version\":1}", "[[[[[0]]]]]"}) {
    auto invalid = parseSM90CostProfile(bad);
    EXPECT_FALSE(bool(invalid));
    if (!invalid) llvm::consumeError(invalid.takeError());
  }
  auto duplicate = parseSM90CostProfile("{\"schema_version\":1," + text.substr(1));
  EXPECT_FALSE(bool(duplicate));
  if (!duplicate) llvm::consumeError(duplicate.takeError());
  auto escaped = parseSM90CostProfile("{\"schema_\\u0076ersion\":1," + text.substr(1));
  EXPECT_FALSE(bool(escaped));
  if (!escaped) llvm::consumeError(escaped.takeError());
  auto object = llvm::json::parse(text);
  ASSERT_TRUE(bool(object));
  object->getAsObject()->getObject("coefficients")->operator[]("alu_warp_issues") = -1;
  auto invalid = parseSM90CostProfile(llvm::formatv("{0}", *object).str());
  EXPECT_FALSE(bool(invalid));
  if (!invalid) llvm::consumeError(invalid.takeError());
  auto floatingText = text;
  auto keyPosition = floatingText.find("\"alu_warp_issues\"");
  ASSERT_NE(keyPosition, std::string::npos);
  auto numberPosition = floatingText.find_first_of("0123456789", keyPosition);
  ASSERT_NE(numberPosition, std::string::npos);
  auto numberEnd = floatingText.find_first_not_of("0123456789", numberPosition);
  floatingText.replace(numberPosition, numberEnd - numberPosition, "1.0");
  auto floating = parseSM90CostProfile(floatingText);
  EXPECT_FALSE(bool(floating));
  if (!floating) llvm::consumeError(floating.takeError());
}
TEST(SM90CalibratedCostTest, RouteAndBarrierCostsUseSameObjectiveAndFrequency) {
  auto profile = unitProfile();
  profile.coefficients[10] = 2;
  profile.coefficients[11] = 3;
  profile.coefficients[12] = 5;
  profile.coefficients[14] = 7;
  auto target = createSM90LayoutTarget(LayoutBoundaryFrequencyMode::StaticTripCount, profile);
  CostEstimate raw;
  raw.cost.conversionBytesAndSync = 1088;
  LayoutConversionEdge edge;
  edge.bytes = 128;
  edge.frequencyFactor = 3;
  for (auto [route, expected] : {
      std::pair{ConversionExchangeKind::Register, uint64_t{768}},
      std::pair{ConversionExchangeKind::Shuffle, uint64_t{1152}},
      std::pair{ConversionExchangeKind::Shared, uint64_t{1962}}}) {
    edge.exchange = route;
    edge.synchronizationCost = route == ConversionExchangeKind::Shared ? 2 : 0;
    auto cost = target->adjustConversionCost(raw, edge);
    ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(cost->cost.instructionPathAndWork, expected);
    EXPECT_EQ(cost->cost.conversionBytesAndSync, raw.cost.conversionBytesAndSync);
    EXPECT_FALSE(cost->features.empty());
  }
  EXPECT_FALSE(target->requiresRematerializationTrafficReduction());
  EXPECT_TRUE(createSM90LayoutTarget()->requiresRematerializationTrafficReduction());
}
TEST(SM90CalibratedCostTest, RawFeatureLedgerIsMergedAndSaturates) {
  CostEstimate a, b;
  a.features = {{"x", UINT64_MAX}, {"y", 1}};
  b.features = {{"x", 1}, {"z", 2}};
  auto sum = addLayoutCosts(a,b);
  EXPECT_TRUE(sum.saturated);
  EXPECT_EQ(sum.features.size(), 3u);
  EXPECT_EQ(sum.features[0].second, UINT64_MAX);
  EXPECT_EQ(sum.cost.instructionPathAndWork, 0u);
}
TEST(SM90CalibratedCostTest, ArithmeticDefinitionIsChargedOnceWithVariableOwner) {
  MLIRContext context;
  context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect, math::MathDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%x:tensor<32xf32>,%y:tensor<32xf32>) -> tensor<32xf32> {
      %a=arith.mulf %x,%y : tensor<32xf32>
      return %a : tensor<32xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  arith::MulFOp mul;
  module->walk([&](arith::MulFOp op){ mul=op; });
  ASSERT_TRUE(mul);
  Builder b(&context);
  SmallVector<APInt> matrix;
  for (unsigned row=0;row<5;++row)
    for (unsigned col=0;col<5;++col) matrix.push_back(APInt(1,row==col));
  auto map=BitLinearLayoutMapAttr::get(&context,b.getStrArrayAttr({"lane"}),
      b.getDenseI64ArrayAttr({5}),b.getStrArrayAttr({"x"}),b.getDenseI64ArrayAttr({5}),
      DenseIntElementsAttr::get(RankedTensorType::get({5,5},b.getI1Type()),matrix));
  auto encoding=DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({1,32,1,1,1}),b.getI64IntegerAttr(1));
  LayoutConstraintGraph graph;
  auto id=graph.addVariable(LayoutKind::Distributed,mul.getType(),"mul",mul);
  graph.getVariable(id).value=mul.getResult();
  CandidateAssignment assignment;
  assignment.values[id]=encoding;
  auto profile=unitProfile();profile.coefficients[6]=17;
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,profile);
  TestCostObserver observer;
  auto cost=target->evaluate(graph,assignment,&observer);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.instructionPathAndWork,17u);
  observer.expectSum(*cost);
  ASSERT_EQ(cost->features.size(),1u);
  EXPECT_EQ(cost->features[0].first,"alu_warp_issues");
  EXPECT_EQ(cost->features[0].second,1u);
  CandidateAssignment partial;
  auto bound=target->lowerBound(graph,partial);
  ASSERT_TRUE(succeeded(bound));
  EXPECT_FALSE(cost->cost < bound->cost);
  auto legacy=createSM90LayoutTarget()->evaluate(graph,assignment);
  ASSERT_TRUE(succeeded(legacy));
  EXPECT_EQ(legacy->cost.instructionPathAndWork,0u);
  auto unsupported = parseSourceString<ModuleOp>(R"mlir(
    func.func @unsupported(%x:tensor<32xf32>) -> tensor<32xf32> {
      %z=math.log %x : tensor<32xf32>
      return %z : tensor<32xf32>
    })mlir", &context);
  ASSERT_TRUE(unsupported);
  math::LogOp logarithm;
  unsupported->walk([&](math::LogOp op) { logarithm = op; });
  ASSERT_TRUE(logarithm);
  graph.getVariable(id).value = logarithm.getResult();
  EXPECT_TRUE(failed(target->evaluate(graph, assignment)));
}
} // namespace
