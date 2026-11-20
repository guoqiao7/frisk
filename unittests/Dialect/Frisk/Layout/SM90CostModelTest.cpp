#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
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
#include <array>
#include <map>
#include <set>
using namespace mlir;
using namespace mlir::frisk;
namespace {
TEST(SM90CostModelTest, CompleteModelIdentityIsIndependentOfCoverage) {
  auto target = createSM90LayoutTarget();
  EXPECT_EQ(getSM90LegacyCostModelID(), "sm90-layout-lex9-legacy-v2");
  EXPECT_EQ(target->costModelID(), getSM90LegacyCostModelID());
  EXPECT_NE(target->costCoverage(), target->costModelID());
  constexpr StringLiteral legacyCoverage =
      "symbolic-subspace-proof-v1 with cta-staging-upper-bound-v1 fallback; "
      "pure-slice-byte-work-v1; loop-depth proxy; register/replication slots; shared root spans; "
      "sm90-shape-work-v1; warp-issue-32B-sectors-v1 (unknown base phase upper bound); "
      "shared-128B-phase-32x4B-distinct-word-conflicts-v2; "
      "symbolic-access-cosets-v1 (logical<=65536; non-dyadic bounded fallback); "
      "unmodeled=latency,cache,overlap";
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
  EXPECT_TRUE(StringRef(id).starts_with("sm90-layout-calibrated-v3-"));
  EXPECT_TRUE(StringRef(serializeSM90CostProfile(profile)).contains("sm90-layout-calibrated-v2"));
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


namespace {
// Minimal actual Copy graph for scorer/cache tests. The selected bindings
// exercise scoring dependencies; instruction-contract legality has its own
// tests. Every graph passes finalize, including actual SSA role identities.
struct AccessCacheFixture {
  OwningOpRef<ModuleOp> module;
  LayoutConstraintGraph graph;
  CandidateAssignment assignment;
  Attribute sourceLayout, targetLayout, encoding;
  SmallVector<Attribute> bindings;
  LayoutVarID source = 0, destination = 0, execution = 0, path = 0;
  LayoutConstraintID access = 0;
  bool initialize(MLIRContext &context, unsigned bits = 32,
      unsigned alignment = 1, unsigned warps = 1, bool loop = false,
      unsigned rows = 8, unsigned columns = 16) {
    Builder b(&context);
    std::string shape = std::to_string(rows) + "x" + std::to_string(columns);
    std::string element = "i" + std::to_string(bits);
    std::string src = "memref<" + shape + "x" + element + ",1>";
    std::string dst = "memref<" + shape + "x" + element + ",3>";
    std::string copy = "\"frisk.copy\"(%s,%d) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:" +
        std::to_string(rows) + "," + std::to_string(columns) +
        ">,dstExtents=array<i64:" + std::to_string(rows) + "," + std::to_string(columns) +
        ">}> {operandSegmentSizes=array<i32:1,1,0,0>} : (" + src + "," + dst + ")->()\n";
    std::string body = "func.func @cache() { %s=memref.alloc() {alignment=128:i64}:" + src +
        " %d=memref.alloc() {alignment=128:i64}:" + dst + "\n";
    if (loop) body += "%c0=arith.constant 0:index %c1=arith.constant 1:index %c3=arith.constant 3:index "
                      "scf.for %i=%c0 to %c3 step %c1 {\n";
    body += copy;
    if (loop) body += "}\n";
    body += "return }";
    module = parseSourceString<ModuleOp>(body, &context);
    if (!module) return false;
    Operation *op = nullptr;
    module->walk([&](Operation *candidate) {
      if (candidate->getName().getStringRef() == "frisk.copy") op = candidate;
    });
    if (!op) return false;
    auto sourceAlias = analyzeStorageAlias(op->getOperand(0));
    auto targetAlias = analyzeStorageAlias(op->getOperand(1));
    if (failed(sourceAlias) || failed(targetAlias)) return false;
    sourceAlias->rootKey = "source";
    targetAlias->rootKey = "target";
    // Keep this a truthful, possibly weaker guarantee than the actual 128B
    // allocation. Upgrading the query to 32B later does not invent alignment.
    sourceAlias->rootAlignment = alignment;
    auto srcLayout = buildRootLinearStorageCandidate(*sourceAlias);
    auto dstLayout = buildRootLinearStorageCandidate(*targetAlias);
    if (failed(srcLayout) || failed(dstLayout)) return false;
    sourceLayout = *srcLayout;
    targetLayout = *dstLayout;
    unsigned registers = rows * columns / 32;
    unsigned registerBits = llvm::Log2_64(registers);
    unsigned rowBits = llvm::Log2_64(rows), columnBits = llvm::Log2_64(columns);
    unsigned warpBits = llvm::Log2_64(warps);
    SmallVector<APInt> matrix;
    for (unsigned row = 0; row < rowBits + columnBits; ++row)
      for (unsigned column = 0; column < registerBits + 5 + warpBits; ++column)
        matrix.emplace_back(1, row == column);
    SmallVector<StringRef> inputNames = {"register", "lane"};
    SmallVector<int64_t> inputBits = {int64_t(registerBits), 5};
    if (warpBits) { inputNames.push_back("warp"); inputBits.push_back(warpBits); }
    auto map = BitLinearLayoutMapAttr::get(&context, b.getStrArrayAttr(inputNames),
        b.getDenseI64ArrayAttr(inputBits), b.getStrArrayAttr({"dim0", "dim1"}),
        b.getDenseI64ArrayAttr({rowBits, columnBits}), DenseIntElementsAttr::get(
            RankedTensorType::get({rowBits + columnBits, registerBits + 5 + warpBits}, b.getI1Type()), matrix));
    encoding = DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr({registers, 32, warps, 1, 1}), b.getI64IntegerAttr(warps));
    auto tensor = RankedTensorType::get({rows, columns}, b.getIntegerType(bits));
    source = graph.addVariable(LayoutKind::Storage, op->getOperand(0).getType(), "source", op);
    destination = graph.addVariable(LayoutKind::Storage, op->getOperand(1).getType(), "target", op);
    execution = graph.addVariable(LayoutKind::Distributed, tensor, "execution", op);
    path = graph.addVariable(LayoutKind::Instruction, tensor, "path", op);
    graph.getVariable(source).value = op->getOperand(0);
    graph.getVariable(source).storageAlias = *sourceAlias;
    graph.getVariable(destination).value = op->getOperand(1);
    graph.getVariable(destination).storageAlias = *targetAlias;
    graph.getVariable(execution).operationExecution = OperationExecutionBinding{32, 1, "first_owner"};
    graph.getVariable(execution).requiredThreads = 32;
    graph.getVariable(path).requiredThreads = 32;
    graph.getVariable(path).instructionRole = true;
    auto instruction = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, {source, destination, execution, path}, op, "cache-test", "selected Copy");
    graph.getConstraints()[instruction].instruction = InstructionLayoutContract{op, {}, {}};
    for (auto [name, width] : {std::pair{StringRef("vector"), 1u},
                              std::pair{StringRef("vector"), 16u},
                              std::pair{StringRef("tma"), 16u}}) {
      auto binding = b.getDictionaryAttr({b.getNamedAttr("path", b.getStringAttr(name)),
          b.getNamedAttr("vector_bytes", b.getI64IntegerAttr(width))});
      bindings.push_back(binding);
      graph.getConstraints()[instruction].instruction->tuples.push_back(
          {{sourceLayout, targetLayout, encoding, binding}, binding});
      graph.getVariable(path).candidates.push_back({binding});
    }
    graph.getVariable(source).candidates.push_back({sourceLayout});
    graph.getVariable(destination).candidates.push_back({targetLayout});
    graph.getVariable(execution).candidates.push_back({encoding});
    access = graph.addConstraint(ConstraintKind::StorageAccess,
        ConstraintStrength::Hard, {source, execution}, op, "cache-test", "source access");
    if (failed(graph.finalize(op->getLoc()))) return false;
    rebind();
    select(0);
    return true;
  }
  void rebind() {
    source = *graph.lookupVariable("source");
    destination = *graph.lookupVariable("target");
    execution = *graph.lookupVariable("execution");
    path = *graph.lookupVariable("path");
    for (const auto &c : graph.getConstraints())
      if (c.kind == ConstraintKind::StorageAccess) access = c.id;
  }
  void select(unsigned which, bool complete = true) {
    assignment.values.clear();
    assignment.values[source] = sourceLayout;
    assignment.values[execution] = encoding;
    if (complete) {
      assignment.values[destination] = targetLayout;
      assignment.values[path] = bindings[which];
    }
  }
};
class SM90AccessCacheTest : public testing::Test {
protected:
  MLIRContext context;
  SM90AccessCacheTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect,
                        arith::ArithDialect, scf::SCFDialect>();
  }
  void sameEstimate(const CostEstimate &a, const CostEstimate &b) {
    EXPECT_EQ(a.cost, b.cost);
    EXPECT_EQ(a.saturated, b.saturated);
    EXPECT_EQ(a.features, b.features);
  }
  void sameParts(const TestCostObserver &a, const TestCostObserver &b) {
    ASSERT_EQ(a.parts.size(), b.parts.size());
    for (auto [x, y] : llvm::zip_equal(a.parts, b.parts)) {
      EXPECT_EQ(x.kind, y.kind); EXPECT_EQ(x.owner, y.owner);
      sameEstimate(x.cost, y.cost);
    }
  }
};
TEST_F(SM90AccessCacheTest, ColdWarmAndDisabledReplayIdenticalContributions) {
  AccessCacheFixture f;
  ASSERT_TRUE(f.initialize(context));
  auto target = createSM90LayoutTarget();
  LayoutProfile profile(true);
  LayoutProfile::Invocation invocation(&profile);
  TestCostObserver coldParts, warmParts, offParts;
  auto cold = target->evaluate(f.graph, f.assignment, &coldParts);
  auto warm = target->evaluate(f.graph, f.assignment, &warmParts);
  ASSERT_TRUE(succeeded(cold)); ASSERT_TRUE(succeeded(warm));
  sameEstimate(*cold, *warm); sameParts(coldParts, warmParts);
  warmParts.expectSum(*warm);
  EXPECT_EQ(f.graph.getAccessCostCache().size(), 1u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).misses, 1u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).hits, 1u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).entries, 1u);
  f.graph.setCostQueryCacheOptions(false);
  EXPECT_TRUE(f.graph.getAccessCostCache().empty());
  auto off = target->evaluate(f.graph, f.assignment, &offParts);
  ASSERT_TRUE(succeeded(off)); sameEstimate(*warm, *off); sameParts(warmParts, offParts);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).count, 2u);
}
TEST_F(SM90AccessCacheTest, CopyWidthBulkAndMissingBindingRemainDistinct) {
  AccessCacheFixture f;
  ASSERT_TRUE(f.initialize(context));
  auto target = createSM90LayoutTarget();
  LayoutProfile profile(true);
  LayoutProfile::Invocation invocation(&profile);
  SmallVector<CostEstimate> first;
  for (unsigned which = 0; which != 4; ++which) {
    f.select(which % 3, which != 3);
    auto cost = target->evaluate(f.graph, f.assignment);
    ASSERT_TRUE(succeeded(cost)); first.push_back(*cost);
  }
  EXPECT_EQ(f.graph.getAccessCostCache().size(), 4u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).misses, 4u);
  EXPECT_LT(first[1].cost.instructionPathAndWork, first[0].cost.instructionPathAndWork);
  for (unsigned which = 0; which != 4; ++which) {
    f.select(which % 3, which != 3);
    auto cost = target->evaluate(f.graph, f.assignment);
    ASSERT_TRUE(succeeded(cost)); sameEstimate(*cost, first[which]);
  }
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).hits, 4u);
  f.graph.setCostQueryCacheOptions(false);
  for (unsigned which = 0; which != 4; ++which) {
    f.select(which % 3, which != 3);
    auto cost = target->evaluate(f.graph, f.assignment);
    ASSERT_TRUE(succeeded(cost)); sameEstimate(*cost, first[which]);
  }
}
TEST_F(SM90AccessCacheTest, ReadWriteAndAlignmentFactsDoNotAlias) {
  AccessCacheFixture f;
  ASSERT_TRUE(f.initialize(context, 32, 1, 2));
  auto target = createSM90LayoutTarget();
  auto read = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(read));
  f.graph.getConstraints()[f.access].access = AccessKind::Write;
  auto write = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(write));
  EXPECT_LT(write->cost.instructionPathAndWork, read->cost.instructionPathAndWork);
  EXPECT_EQ(f.graph.getAccessCostCache().size(), 2u);
  // A defensive alignment key prevents the stronger valid guarantee from
  // reusing the unknown-phase upper bound. Addresses/layout stay unchanged.
  f.graph.getVariable(f.source).storageAlias->rootAlignment = 32;
  auto aligned = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(aligned));
  EXPECT_LT(aligned->cost.memoryTransactions, write->cost.memoryTransactions);
  EXPECT_EQ(f.graph.getAccessCostCache().size(), 3u);
  f.graph.setCostQueryCacheOptions(false);
  auto reference = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(reference)); sameEstimate(*aligned, *reference);
}
TEST_F(SM90AccessCacheTest, IndependentGraphsAndElementTypesHaveIndependentEntries) {
  AccessCacheFixture a, b;
  ASSERT_TRUE(a.initialize(context, 16)); ASSERT_TRUE(b.initialize(context, 32));
  auto target = createSM90LayoutTarget();
  auto first = target->evaluate(a.graph, a.assignment);
  auto second = target->evaluate(b.graph, b.assignment);
  ASSERT_TRUE(succeeded(first)); ASSERT_TRUE(succeeded(second));
  EXPECT_NE(first->cost.memoryTransactions, second->cost.memoryTransactions);
  EXPECT_EQ(a.graph.getAccessCostCache().size(), 1u);
  EXPECT_EQ(b.graph.getAccessCostCache().size(), 1u);
  auto again = target->evaluate(a.graph, a.assignment);
  ASSERT_TRUE(succeeded(again)); sameEstimate(*first, *again);
  b.graph.setCostQueryCacheOptions(false);
  auto off = target->evaluate(b.graph, b.assignment);
  ASSERT_TRUE(succeeded(off)); sameEstimate(*second, *off);
}
TEST_F(SM90AccessCacheTest, CapacityBypassAndStructuralInvalidationPreserveResults) {
  AccessCacheFixture f;
  ASSERT_TRUE(f.initialize(context));
  auto target = createSM90LayoutTarget();
  f.graph.setCostQueryCacheOptions(true, 1);
  LayoutProfile profile(true);
  LayoutProfile::Invocation invocation(&profile);
  auto first = target->evaluate(f.graph, f.assignment);
  f.select(1);
  auto second = target->evaluate(f.graph, f.assignment);
  auto repeated = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(first)); ASSERT_TRUE(succeeded(second)); ASSERT_TRUE(succeeded(repeated));
  sameEstimate(*second, *repeated);
  EXPECT_EQ(f.graph.getAccessCostCache().size(), 1u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).misses, 3u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).entries, 1u);
  f.graph.setCostQueryCacheOptions(true, 0);
  auto zero = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(zero)); sameEstimate(*second, *zero);
  EXPECT_TRUE(f.graph.getAccessCostCache().empty());
  f.graph.setCostQueryCacheOptions(true, UINT32_MAX);
  EXPECT_EQ(f.graph.getCostQueryCacheCapacity(), 4096u);
  ASSERT_TRUE(succeeded(target->evaluate(f.graph, f.assignment)));
  ASSERT_FALSE(f.graph.getAccessCostCache().empty());
  auto type = f.graph.getVariable(f.execution).shapedType;
  f.graph.addVariable(LayoutKind::Distributed, type, "added");
  EXPECT_FALSE(f.graph.isCostQueryCacheEnabled());
  EXPECT_TRUE(f.graph.getAccessCostCache().empty());
  ASSERT_TRUE(succeeded(f.graph.finalize(UnknownLoc::get(&context))));
  f.rebind(); f.select(1);
  ASSERT_TRUE(succeeded(target->evaluate(f.graph, f.assignment)));
  ASSERT_FALSE(f.graph.getAccessCostCache().empty());
  ASSERT_TRUE(succeeded(f.graph.finalize(UnknownLoc::get(&context))));
  EXPECT_TRUE(f.graph.getAccessCostCache().empty());
  f.rebind(); f.select(1);
  auto final = target->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(final)); sameEstimate(*second, *final);
  f.graph.getVariable(f.source).stableName = "execution";
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(f.graph.finalize(UnknownLoc::get(&context))));
  EXPECT_FALSE(f.graph.isFinalized());
  EXPECT_TRUE(f.graph.getAccessCostCache().empty());
}
TEST_F(SM90AccessCacheTest, UnknownFootprintIsNeverCachedAsAFreeCost) {
  AccessCacheFixture f;
  ASSERT_TRUE(f.initialize(context));
  // The type/layout are small and independently valid. This unsupported
  // symbolic alias transform reaches accessCost's footprint proof, rather
  // than failing the earlier immutable type/layout decoder on a large domain.
  f.graph.getVariable(f.source).storageAlias->viewToRoot = AffineMap::get(2, 1,
      {getAffineDimExpr(0, &context) + getAffineSymbolExpr(0, &context),
       getAffineDimExpr(1, &context)}, &context);
  ASSERT_TRUE(succeeded(f.graph.finalize(UnknownLoc::get(&context))));
  f.rebind(); f.select(0, false); // Avoid unrelated Shared-root/setup terms.
  LayoutProfile profile(true);
  LayoutProfile::Invocation invocation(&profile);
  auto target = createSM90LayoutTarget();
  EXPECT_TRUE(failed(target->evaluate(f.graph, f.assignment)));
  EXPECT_TRUE(failed(target->evaluate(f.graph, f.assignment)));
  EXPECT_EQ(getStorageAliasFootprint(f.graph, f.source,
      cast<StorageLayoutAttr>(f.sourceLayout)).proof.status, ProofStatus::Unknown);
  EXPECT_TRUE(f.graph.getAccessCostCache().empty());
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).misses, 2u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AccessCostCache).entries, 0u);
  f.graph.setCostQueryCacheOptions(false);
  EXPECT_TRUE(failed(target->evaluate(f.graph, f.assignment)));
}
TEST_F(SM90AccessCacheTest, CalibratedProfilesAndFrequencyAreAppliedAfterRawReplay) {
  AccessCacheFixture f;
  ASSERT_TRUE(f.initialize(context, 32, 1, 1, true));
  f.select(2);
  auto firstProfile = unitProfile();
  auto secondProfile = firstProfile;
  secondProfile.coefficients[0] = 7;
  secondProfile.coefficients[1] = 11;
  auto legacy = createSM90LayoutTarget();
  ASSERT_TRUE(succeeded(legacy->evaluate(f.graph, f.assignment)));
  auto first = createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy, firstProfile);
  auto second = createSM90LayoutTarget(LayoutBoundaryFrequencyMode::StaticTripCount, secondProfile);
  TestCostObserver warmParts, offParts;
  auto warm = second->evaluate(f.graph, f.assignment, &warmParts);
  auto different = first->evaluate(f.graph, f.assignment);
  ASSERT_TRUE(succeeded(warm)); ASSERT_TRUE(succeeded(different));
  EXPECT_NE(warm->cost.instructionPathAndWork, different->cost.instructionPathAndWork);
  EXPECT_EQ(f.graph.getAccessCostCache().size(), 1u);
  f.graph.setCostQueryCacheOptions(false);
  auto off = second->evaluate(f.graph, f.assignment, &offParts);
  ASSERT_TRUE(succeeded(off)); sameEstimate(*warm, *off); sameParts(warmParts, offParts);
  warmParts.expectSum(*warm);
}
} // namespace


namespace {
// Independent small-domain oracle: retain the original concrete hardware
// traversal and integer issue/phase partitioning. It deliberately does not use
// the symbolic implementation's bases, quotient classes or elected section.
struct SymbolicAccessFixture {
  OwningOpRef<ModuleOp> module;
  LayoutConstraintGraph graph;
  CandidateAssignment assignment;
  DistributedEncodingAttr encoding;
  StorageLayoutAttr storage;
  RankedTensorType type;
  LayoutVarID storageID = 0;
  unsigned bits = 32, width = 1, alignment = 32;
  bool global = true;
  StringRef path = "vector";
  AccessKind access = AccessKind::Read;

  bool initialize(MLIRContext &context, uint64_t logical,
      ArrayRef<int64_t> topology, ArrayRef<uint64_t> images,
      unsigned elementBits = 32, unsigned vectorBytes = 1,
      bool globalMemory = true, AccessKind kind = AccessKind::Read,
      unsigned stride = 1, unsigned rootAlignment = 32,
      StringRef copyPath = "vector", bool reverseInputs = false,
      bool omitUnused = false, uint64_t actual = 0) {
    Builder b(&context);
    bits = elementBits; width = vectorBytes; global = globalMemory;
    alignment = rootAlignment; access = kind; path = copyPath;
    if (!actual) actual = logical;
    std::string memref = "memref<" + std::to_string(actual) + "xi" +
        std::to_string(bits) + ",strided<[" + std::to_string(stride) +
        "]>," + (global ? "1" : "3") + ">";
    module = parseSourceString<ModuleOp>("func.func @symbolic() { %a=memref.alloc() "
        "{alignment=" + std::to_string(alignment) + ":i64}:" + memref +
        " return }", &context);
    if (!module) return false;
    memref::AllocOp allocation;
    module->walk([&](memref::AllocOp op) { allocation = op; });
    auto alias = analyzeStorageAlias(allocation.getResult());
    if (failed(alias)) return false;
    alias->rootKey = "symbolic-access-root";
    auto layout = buildRootLinearStorageCandidate(*alias);
    if (failed(layout)) return false;
    storage = *layout;
    constexpr const char *names[] = {"register", "lane", "warp", "warp_group", "cta"};
    std::array<unsigned, 5> offsets{}, widths{};
    unsigned hardwareBits = 0;
    for (unsigned i = 0; i < 5; ++i) {
      offsets[i] = hardwareBits;
      widths[i] = llvm::Log2_64(topology[i]);
      hardwareBits += widths[i];
    }
    if (images.size() != hardwareBits) return false;
    SmallVector<StringRef> inputNames;
    SmallVector<int64_t> inputWidths;
    SmallVector<unsigned> columns;
    for (unsigned n = 0; n < 5; ++n) {
      unsigned c = reverseInputs ? 4 - n : n;
      // Unit carriers are represented by omission, never a zero input width.
      if (!widths[c]) continue;
      bool used = false;
      for (unsigned j = 0; j < widths[c]; ++j) used |= images[offsets[c]+j] != 0;
      if (omitUnused && !used) continue;
      inputNames.push_back(names[c]); inputWidths.push_back(widths[c]);
      for (unsigned j = 0; j < widths[c]; ++j) columns.push_back(offsets[c]+j);
    }
    unsigned outputBits = llvm::Log2_64(logical);
    SmallVector<APInt> values;
    for (unsigned r = 0; r < outputBits; ++r)
      for (unsigned c : columns) values.emplace_back(1, (images[c] >> r) & 1);
    auto map = BitLinearLayoutMapAttr::get(&context, b.getStrArrayAttr(inputNames),
        b.getDenseI64ArrayAttr(inputWidths), b.getStrArrayAttr({"x"}),
        b.getDenseI64ArrayAttr({int64_t(outputBits)}), DenseIntElementsAttr::get(
            RankedTensorType::get({outputBits, int64_t(columns.size())}, b.getI1Type()), values));
    if (hardwareBits < outputBits || hardwareBits - outputBits >= 63) return false;
    encoding = DistributedEncodingAttr::get(&context, map, b.getDenseI64ArrayAttr(topology),
        b.getI64IntegerAttr(int64_t(1) << (hardwareBits-outputBits)));
    type = RankedTensorType::get({int64_t(logical)}, b.getIntegerType(bits));
    storageID = graph.addVariable(LayoutKind::Storage, allocation.getType(), "storage");
    graph.getVariable(storageID).storageAlias = *alias;
    auto execution = graph.addVariable(LayoutKind::Distributed, type, "execution", allocation);
    graph.getVariable(execution).operationExecution = OperationExecutionBinding{};
    auto plan = graph.addVariable(LayoutKind::Instruction, allocation.getType(), "plan");
    auto binding = b.getDictionaryAttr({b.getNamedAttr("path", b.getStringAttr(path)),
        b.getNamedAttr("vector_bytes", b.getI64IntegerAttr(width))});
    auto instruction = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, {storageID, execution, plan}, allocation,
        "symbolic-cost-test", "selected score-only width");
    // Like SM90AddressCostTest, test scoring dependencies independently of the
    // instruction legality decoder. This source is not a real Copy operation.
    graph.getConstraints()[instruction].instruction = InstructionLayoutContract{
        allocation, {}, {{{storage, encoding, binding}, binding}}};
    auto id = graph.addConstraint(ConstraintKind::StorageAccess,
        ConstraintStrength::Hard, {storageID, execution}, allocation,
        "symbolic-cost-test", "address cost");
    graph.getConstraints()[id].access = access;
    assignment.values[storageID] = storage;
    assignment.values[execution] = encoding;
    assignment.values[plan] = binding;
    return true;
  }
  FailureOr<CostEstimate> evaluate() {
    return evaluateSM90LayoutCost(graph, assignment);
  }
  FailureOr<CostVector> oracle() {
    ExecutionEnumeration points;
    if (enumerateExecutionLayout(encoding, type, points).status != ProofStatus::Proven)
      return failure();
    const auto &footprint = getStorageAliasFootprint(graph, storageID, storage);
    if (footprint.proof.status != ProofStatus::Proven) return failure();
    std::map<SmallVector<int64_t>, std::pair<uint64_t,uint64_t>> addresses;
    for (const auto &p : footprint.entries)
      addresses.emplace(p.view, std::make_pair(p.begin/8, llvm::divideCeil(p.end,uint64_t{8})));
    using Groups = std::map<SmallVector<int64_t>, std::vector<std::pair<uint64_t,uint64_t>>>;
    Groups issues, phases;
    unsigned chunk = std::max(1u, width*8/bits);
    unsigned phaseBytes = std::max(width, llvm::divideCeil(bits,8u));
    unsigned lanesPerPhase = std::max(1u,std::min(32u,128u/phaseBytes));
    for (uint64_t h = 0; h < points.logical.size(); ++h) {
      auto logical = points.logical[h];
      if (access == AccessKind::Write && points.first[logical] != h) continue;
      auto address = addresses.find(points.coordinate(logical));
      if (address == addresses.end()) continue;
      auto hw = points.hardware(h);
      SmallVector<int64_t> key = path == "tma" ? SmallVector<int64_t>{0} :
          SmallVector<int64_t>{hw[0],hw[1],hw[2],hw[4]/chunk};
      issues[key].push_back(address->second);
      key.push_back(hw[3]/lanesPerPhase);
      phases[key].push_back(address->second);
    }
    uint64_t worstSectors = 0, worstBanks = 0;
    for (uint64_t phase = 0; phase < 32; phase += std::min(32u,alignment)) {
      uint64_t sectors = 0, banks = 0;
      for (const auto &[key,ranges] : issues) {
        std::set<uint64_t> touched;
        for (auto [begin,end] : ranges)
          for (uint64_t s=(begin+phase)/32; s<=(end-1+phase)/32; ++s) touched.insert(s);
        sectors += touched.size();
      }
      for (const auto &[key,ranges] : phases) {
        std::map<unsigned,std::set<uint64_t>> words;
        for (auto [begin,end] : ranges)
          for (uint64_t w=(begin+phase)/4; w<=(end-1+phase)/4; ++w) words[w%32].insert(w);
        uint64_t degree = 1;
        for (const auto &[bank,touched] : words) degree = std::max<uint64_t>(degree,touched.size());
        banks += degree-1;
      }
      worstSectors = std::max(worstSectors,sectors);
      worstBanks = std::max(worstBanks,banks);
    }
    CostVector cost;
    if (global) cost.memoryTransactions = worstSectors;
    if (!global && path != "tma") cost.bankConflictDegree = worstBanks;
    if (path != "tma" && (path == "vector" || access == AccessKind::Read))
      cost.instructionPathAndWork = issues.size();
    return cost;
  }
};
class SM90SymbolicAccessTest : public testing::Test {
protected:
  MLIRContext context;
  SM90SymbolicAccessTest() {
    context.loadDialect<FriskDialect,func::FuncDialect,memref::MemRefDialect>();
  }
  void expectOracle(SymbolicAccessFixture &f) {
    auto expected = f.oracle(); auto actual = f.evaluate();
    ASSERT_TRUE(succeeded(expected)); ASSERT_TRUE(succeeded(actual));
    EXPECT_EQ(actual->cost.instructionPathAndWork,expected->instructionPathAndWork);
    EXPECT_EQ(actual->cost.memoryTransactions,expected->memoryTransactions);
    EXPECT_EQ(actual->cost.bankConflictDegree,expected->bankConflictDegree);
    EXPECT_FALSE(actual->saturated);
  }
};
TEST_F(SM90SymbolicAccessTest, MixedGF2MatricesMatchConcreteAddressOracle) {
  // Column swaps/XORs preserve full row rank but create actual mixed GF2 maps,
  // including lane/register/warp interactions rather than only permutations.
  uint32_t state = 0x91a2b3c4;
  auto next = [&] { state ^= state<<13; state ^= state>>17; state ^= state<<5; return state; };
  for (unsigned trial = 0; trial < 12; ++trial) {
    SmallVector<uint64_t> images(9,0);
    for (unsigned i = 0; i < 6; ++i) images[i] = uint64_t(1)<<i;
    for (unsigned i = 0; i < 30; ++i) {
      unsigned a=next()%9,b=next()%9;
      if (a != b) { if (next()&1) std::swap(images[a],images[b]); else images[a] ^= images[b]; }
    }
    for (auto access : {AccessKind::Read,AccessKind::Write})
      for (bool global : {false,true}) {
        SCOPED_TRACE(testing::Message()<<trial<<','<<int(access)<<','<<global);
        SymbolicAccessFixture f;
        ASSERT_TRUE(f.initialize(context,64,{4,32,2,2,1},images,
            trial%3==0 ? 64 : 32,trial%2 ? 16 : 1,global,access,
            trial%2 ? 7 : 1,trial%3 ? 32 : 1,"vector",trial%2));
        expectOracle(f);
      }
  }
}
TEST_F(SM90SymbolicAccessTest, FirstOwnerUsesCanonicalHardwareOrder) {
  // register bit0 and lane bit0 have identical output images: the first owner
  // must use the cheaper low hardware bit, independent of attribute input order.
  for (bool reverse : {false,true}) {
    SymbolicAccessFixture f;
    ASSERT_TRUE(f.initialize(context,4,{2,32,2,1,1},{1,1,2,0,0,0,0},
        32,1,true,AccessKind::Write,8,32,"vector",reverse));
    expectOracle(f);
    auto cost=f.evaluate(); ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(cost->cost.instructionPathAndWork,2u);
  }
}
TEST_F(SM90SymbolicAccessTest, BroadcastsRemainSeparateAcrossPhysicalWarpGroups) {
  SmallVector<uint64_t> images(8,0); images[0]=1;
  for (auto access : {AccessKind::Read,AccessKind::Write}) {
    SymbolicAccessFixture f;
    ASSERT_TRUE(f.initialize(context,2,{2,32,2,2,1},images,
        32,1,true,access,1,32,"vector",false,true));
    expectOracle(f);
    auto cost=f.evaluate(); ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(cost->cost.instructionPathAndWork,access==AccessKind::Read ? 8u : 2u);
  }
}
TEST_F(SM90SymbolicAccessTest, NonPowerOfTwoPartitionsKeepConcreteSemantics) {
  for (unsigned bits : {24u,40u,48u,80u}) {
    // i24 + 16B binding has floor chunk=5; i40/i48/i80 have non-power-of-two
    // native bank phase lane counts. These score-only bindings test the exact
    // legacy bounded fallback, not the separate Copy legality contract.
    SmallVector<uint64_t> images(8,0);
    for (unsigned i=0;i<6;++i) images[i]=uint64_t(1)<<i;
    for (auto access : {AccessKind::Read,AccessKind::Write}) {
      SymbolicAccessFixture f;
      ASSERT_TRUE(f.initialize(context,64,{4,32,2,1,1},images,bits,
          bits==24 ? 16 : 1,false,access,3,1));
      expectOracle(f);
    }
  }
}
TEST_F(SM90SymbolicAccessTest, GuardedTailsCountOnlyNonemptyActiveIssues) {
  for (auto access : {AccessKind::Read,AccessKind::Write}) {
    SymbolicAccessFixture f;
    ASSERT_TRUE(f.initialize(context,8,{2,32,2,1,1},{1,2,4,0,0,0,0},
        32,1,true,access,1,1,"vector",false,false,3));
    expectOracle(f);
  }
}
TEST_F(SM90SymbolicAccessTest, BulkAndAsyncBindingsMatchConcreteGrouping) {
  SmallVector<uint64_t> images(9,0);
  for (unsigned i=0;i<7;++i) images[i]=uint64_t(1)<<i;
  for (StringRef path : {StringRef("vector"),StringRef("cp.async"),StringRef("tma")})
    for (auto access : {AccessKind::Read,AccessKind::Write}) {
      SymbolicAccessFixture f;
      ASSERT_TRUE(f.initialize(context,128,{8,32,2,1,1},images,
          32,16,true,access,1,1,path));
      expectOracle(f);
    }
}
TEST_F(SM90SymbolicAccessTest, HardwareBudgetBoundaryAndLargeReplication) {
  for (unsigned registers : {512u,1024u}) {
    unsigned registerBits=llvm::Log2_64(registers);
    SmallVector<uint64_t> images(registerBits+7,0);
    for (unsigned i=0;i<registerBits;++i) images[i]=uint64_t(1)<<i;
    for (auto access : {AccessKind::Read,AccessKind::Write}) {
      SymbolicAccessFixture f;
      ASSERT_TRUE(f.initialize(context,registers,{registers,32,4,1,1},images,
          32,1,true,access));
      auto cost=f.evaluate(); ASSERT_TRUE(succeeded(cost));
      uint64_t issues=uint64_t(registers)*(access==AccessKind::Read ? 4 : 1);
      EXPECT_EQ(cost->cost.instructionPathAndWork,issues);
      EXPECT_EQ(cost->cost.memoryTransactions,issues);
      EXPECT_EQ(cost->cost.bankConflictDegree,0u);
      EXPECT_FALSE(cost->saturated);
      if (registers==512) expectOracle(f);
      else {
        ExecutionEnumeration points;
        EXPECT_EQ(enumerateExecutionLayout(f.encoding,f.type,points).status,ProofStatus::Unknown);
      }
    }
  }
}
TEST_F(SM90SymbolicAccessTest, UnitLogicalDomainSupportsUniqueAndReplicatedHardware) {
  for (unsigned lanes : {1u,32u})
    for (auto access : {AccessKind::Read,AccessKind::Write}) {
      SymbolicAccessFixture f;
      SmallVector<uint64_t> images(llvm::Log2_64(lanes),0);
      ASSERT_TRUE(f.initialize(context,1,{1,lanes,1,1,1},images,
          32,1,true,access,1,32,"vector",false,true));
      expectOracle(f);
    }
}
TEST_F(SM90SymbolicAccessTest, HugeOmittedCarrierSaturatesWithoutHardwareExpansion) {
  SymbolicAccessFixture f;
  SmallVector<uint64_t> images(62,0);
  ASSERT_TRUE(f.initialize(context,1,{1,32,1,int64_t(1)<<57,1},images,
      32768,1,true,AccessKind::Read,1,32,"vector",false,true));
  auto cost=f.evaluate(); ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.instructionPathAndWork,uint64_t(1)<<57);
  EXPECT_EQ(cost->cost.memoryTransactions,UINT64_MAX);
  EXPECT_EQ(cost->cost.bankConflictDegree,0u);
  EXPECT_TRUE(cost->saturated);
}

TEST_F(SM90SymbolicAccessTest, NonzeroDependentColumnsPreserveReadIssueMultiplicity) {
  // Every column is nonzero, but c0=c1 XOR c2. Removing only zero columns or
  // charging one copy of the image would undercount this valid replicated map.
  for (auto access : {AccessKind::Read,AccessKind::Write}) {
    SymbolicAccessFixture f;
    ASSERT_TRUE(f.initialize(context,4,{2,4,1,1,1},{3,1,2},
        32,1,true,access,8,32,"vector",true));
    expectOracle(f);
    auto cost=f.evaluate(); ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(cost->cost.instructionPathAndWork,2u);
  }
}
TEST_F(SM90SymbolicAccessTest, FullLogicalBoundaryMatchesConcreteOracle) {
  SmallVector<uint64_t> images(16,0);
  for (unsigned i=0;i<16;++i) images[i]=uint64_t(1)<<i;
  SymbolicAccessFixture f;
  ASSERT_TRUE(f.initialize(context,65536,{512,32,4,1,1},images,
      8,1,true,AccessKind::Read));
  expectOracle(f);
}
TEST_F(SM90SymbolicAccessTest, MissingCoverageAndWrongReplicationRemainFailures) {
  ScopedDiagnosticHandler silence(&context, [](Diagnostic &) { return success(); });
  SymbolicAccessFixture missing;
  ASSERT_TRUE(missing.initialize(context,4,{4,1,1,1,1},{1,0}));
  EXPECT_TRUE(failed(missing.evaluate()));
  SymbolicAccessFixture wrong;
  ASSERT_TRUE(wrong.initialize(context,4,{4,1,1,1,1},{1,2}));
  Builder b(&context);
  auto old=wrong.encoding;
  wrong.encoding=DistributedEncodingAttr::get(&context,old.getMap(),old.getTopology(),
      b.getI64IntegerAttr(2));
  for (auto &entry : wrong.assignment.values)
    if (entry.second==old) entry.second=wrong.encoding;
  for (auto &constraint : wrong.graph.getConstraints())
    if (constraint.instruction)
      for (auto &tuple : constraint.instruction->tuples)
        for (auto &encoding : tuple.encodings)
          if (encoding==old) encoding=wrong.encoding;
  EXPECT_TRUE(failed(wrong.evaluate()));
}
} // namespace
