#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
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
    return evaluateSM90LayoutCost(graph, assignment);
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
