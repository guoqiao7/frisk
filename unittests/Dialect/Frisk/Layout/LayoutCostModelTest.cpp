#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
constexpr uint64_t CostVector::*fields[] = {
    &CostVector::instructionPathAndWork, &CostVector::memoryTransactions,
    &CostVector::bankConflictDegree, &CostVector::conversionBytesAndSync,
    &CostVector::spillRiskAndRegisters, &CostVector::sharedBytesAndOccupancy,
    &CostVector::replication, &CostVector::codeSize,
    &CostVector::deterministicTieBreak};
TEST(LayoutCostModelTest, LexicographicDominanceOfEveryField) {
  for (unsigned i = 0; i < 9; ++i) {
    CostVector a, b; b.*fields[i] = 1;
    for (unsigned j = i + 1; j < 9; ++j) a.*fields[j] = UINT64_MAX;
    EXPECT_TRUE(a < b); EXPECT_FALSE(b < a); EXPECT_FALSE(a < a);
  }
}
TEST(LayoutCostModelTest, AdditionSaturatesEveryFieldAndPreservesFlag) {
  for (auto field : fields) {
    CostEstimate a, b; a.cost.*field = UINT64_MAX; b.cost.*field = 1;
    auto sum = addLayoutCosts(a, b);
    EXPECT_EQ(sum.cost.*field, UINT64_MAX); EXPECT_TRUE(sum.saturated);
    EXPECT_TRUE(addLayoutCosts(sum, {}).saturated);
  }
  EXPECT_FALSE(addLayoutCosts({}, {}).saturated);
}
class StaticLayoutCostTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  StaticLayoutCostTest() {
    context.allowUnregisteredDialects();
    context.loadDialect<FriskDialect, func::FuncDialect, scf::SCFDialect>();
  }
  DistributedEncodingAttr encoding(int64_t registers = 2, bool swap = false) {
    unsigned bits = llvm::Log2_64(registers) + 5;
    SmallVector<APInt> entries;
    for (unsigned row = 0; row < bits; ++row)
      for (unsigned col = 0; col < bits; ++col)
        entries.emplace_back(1, col == (swap ? (row + 1) % bits : row));
    auto map = BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("register"), b.getStringAttr("lane")}),
        b.getDenseI64ArrayAttr({int64_t(bits - 5), 5}),
        b.getArrayAttr({b.getStringAttr("d")}), b.getDenseI64ArrayAttr({bits}),
        DenseIntElementsAttr::get(RankedTensorType::get({bits, bits}, b.getI1Type()), entries));
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr({registers, 32, 1, 1, 1}), b.getI64IntegerAttr(1));
  }
  DistributedEncodingAttr permutedEncoding(int64_t registers,
                                            int64_t warps,
                                            unsigned lhs,
                                            unsigned rhs,
                                            int64_t lanes = 32) {
    unsigned registerBits = llvm::Log2_64(registers);
    unsigned laneBits = llvm::Log2_64(lanes);
    unsigned bits = registerBits + laneBits + llvm::Log2_64(warps);
    SmallVector<APInt> entries;
    for (unsigned row = 0; row < bits; ++row)
      for (unsigned col = 0; col < bits; ++col) {
        unsigned selected = row == lhs ? rhs : row == rhs ? lhs : row;
        entries.emplace_back(1, col == selected);
      }
    SmallVector<Attribute> names{b.getStringAttr("register"),
                                 b.getStringAttr("lane")};
    SmallVector<int64_t> widths{int64_t(registerBits), int64_t(laneBits)};
    if (warps > 1) {
      names.push_back(b.getStringAttr("warp"));
      widths.push_back(llvm::Log2_64(warps));
    }
    auto map = BitLinearLayoutMapAttr::get(
        &context, b.getArrayAttr(names), b.getDenseI64ArrayAttr(widths),
        b.getArrayAttr({b.getStringAttr("d")}),
        b.getDenseI64ArrayAttr({int64_t(bits)}),
        DenseIntElementsAttr::get(
            RankedTensorType::get({bits, bits}, b.getI1Type()), entries));
    return DistributedEncodingAttr::get(
        &context, map,
        b.getDenseI64ArrayAttr({registers, lanes, warps, 1, 1}),
        b.getI64IntegerAttr(1));
  }
};
TEST_F(StaticLayoutCostTest, ClassifiesIdentityAndRegisterExchange) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  var.shapedType = RankedTensorType::get({128}, b.getF32Type());
  auto a = permutedEncoding(4, 1, 0, 0);
  auto z = permutedEncoding(4, 1, 0, 1);
  LayoutConversionEdge edge; edge.bytes = 9; edge.synchronizationCost = 8;
  auto identity = evaluateLayoutConversionCost(var, a, a, edge);
  ASSERT_TRUE(succeeded(identity)); EXPECT_EQ(edge.bytes, 0u);
  EXPECT_EQ(edge.synchronizationCost, 0u); EXPECT_EQ(identity->cost.codeSize, 0u);
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Identity);
  auto exchange = evaluateLayoutConversionCost(var, a, z, edge);
  ASSERT_TRUE(succeeded(exchange)); EXPECT_EQ(edge.bytes, 4u * 128);
  EXPECT_EQ(edge.synchronizationCost, 0u);
  EXPECT_EQ(exchange->cost.conversionBytesAndSync, 512u);
  EXPECT_EQ(exchange->cost.codeSize, 1u);
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Register);
  EXPECT_EQ(edge.costProvenance, "bounded-owner-proof-v1/register");
  EXPECT_EQ(edge.sourceEncoding, a); EXPECT_EQ(edge.targetEncoding, z);
}
TEST_F(StaticLayoutCostTest, ClassifiesShuffleAndSharedAtMinimumScope) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  LayoutConversionEdge edge;
  var.shapedType = RankedTensorType::get({64}, b.getF32Type());
  auto identity = permutedEncoding(2, 1, 0, 0);
  auto laneSwap = permutedEncoding(2, 1, 1, 2);
  auto shuffle = evaluateLayoutConversionCost(var, identity, laneSwap, edge);
  ASSERT_TRUE(succeeded(shuffle));
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Shuffle);
  EXPECT_EQ(edge.bytes, 4u * (64 + 64));
  EXPECT_EQ(edge.synchronizationCost, 0u);
  EXPECT_EQ(shuffle->cost.conversionBytesAndSync, 512u);
  EXPECT_EQ(edge.costProvenance, "bounded-owner-proof-v1/shuffle");

  var.shapedType = RankedTensorType::get({128}, b.getF32Type());
  auto twoWarps = permutedEncoding(2, 2, 0, 0);
  auto warpSwap = permutedEncoding(2, 2, 1, 6);
  auto shared = evaluateLayoutConversionCost(var, twoWarps, warpSwap, edge);
  ASSERT_TRUE(succeeded(shared));
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Shared);
  EXPECT_EQ(edge.bytes, 4u * (128 + 128));
  EXPECT_EQ(edge.synchronizationCost, 2u);
  EXPECT_EQ(shared->cost.conversionBytesAndSync, 1024u + 4u * 64 * 2);
  EXPECT_EQ(edge.costProvenance, "bounded-owner-proof-v1/shared");
}
TEST_F(StaticLayoutCostTest, ConversionRejectsMissingOrDynamicInformation) {
  LayoutVar var; LayoutConversionEdge edge;
  EXPECT_TRUE(failed(evaluateLayoutConversionCost(var, {}, {}, edge)));
  var.shapedType = RankedTensorType::get({ShapedType::kDynamic}, b.getF32Type());
  EXPECT_TRUE(failed(evaluateLayoutConversionCost(var, encoding(), encoding(2,true), edge)));
}
TEST_F(StaticLayoutCostTest, ElementBitsRoundUpAndProofBudgetFallsBack) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  LayoutConversionEdge edge;
  for (auto [bits, bytes] : {std::pair<unsigned, uint64_t>{1, 1}, {16, 2}, {32, 4}}) {
    var.shapedType = RankedTensorType::get({64}, b.getIntegerType(bits));
    auto cost = evaluateLayoutConversionCost(var, encoding(), encoding(2, true), edge);
    ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(edge.exchange, ConversionExchangeKind::Shuffle);
    EXPECT_EQ(edge.bytes, 128u * bytes);
    EXPECT_EQ(cost->cost.conversionBytesAndSync, edge.bytes);
  }
  var.shapedType = RankedTensorType::get({131072}, b.getF16Type());
  auto cost = evaluateLayoutConversionCost(var, encoding(4096), encoding(4096, true), edge);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Conservative);
  EXPECT_EQ(edge.costProvenance, "cta-staging-upper-bound-v1");
  EXPECT_EQ(edge.bytes, 2u * 2u * 131072u);
  EXPECT_EQ(edge.synchronizationCost, 2u);
  EXPECT_GT(cost->cost.conversionBytesAndSync, edge.bytes);
}
TEST_F(StaticLayoutCostTest, DifferentThreadTopologyUsesConservativeUpperBound) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  var.shapedType = RankedTensorType::get({64}, b.getF32Type());
  auto src = encoding();
  auto extraWarp = DistributedEncodingAttr::get(&context, src.getMap(),
      b.getDenseI64ArrayAttr({2, 32, 2, 1, 1}), b.getI64IntegerAttr(2));
  auto multiCTA = DistributedEncodingAttr::get(&context, src.getMap(),
      b.getDenseI64ArrayAttr({2, 32, 1, 1, 2}), b.getI64IntegerAttr(2));
  LayoutConversionEdge edge;
  auto cost = evaluateLayoutConversionCost(var, src, extraWarp, edge);
  ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.conversionBytesAndSync, 4u*(64+128)+4u*64*2);
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Conservative);
  EXPECT_EQ(edge.costProvenance, "cta-staging-upper-bound-v1");
  EXPECT_TRUE(failed(evaluateLayoutConversionCost(var, src, multiCTA, edge)));
  var.shapedType = RankedTensorType::get({32}, b.getF32Type());
  auto halfWarp = permutedEncoding(2, 1, 0, 0, 16);
  auto halfWarpSwap = permutedEncoding(2, 1, 0, 1, 16);
  auto laneFallback =
      evaluateLayoutConversionCost(var, halfWarp, halfWarpSwap, edge);
  ASSERT_TRUE(succeeded(laneFallback));
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Conservative);
  EXPECT_EQ(edge.costProvenance, "cta-staging-upper-bound-v1");
}
TEST_F(StaticLayoutCostTest, LoopDepthWeightsOnlyDynamicBoundaryProxy) {
  OwningOpRef<ModuleOp> module(ModuleOp::create(b.getUnknownLoc()));
  OperationState forState(b.getUnknownLoc(), scf::ForOp::getOperationName());
  forState.addRegion();
  Operation *forOp = Operation::create(forState);
  module->getBody()->push_back(forOp);
  forOp->getRegion(0).push_back(new Block);
  OperationState whileState(b.getUnknownLoc(), scf::WhileOp::getOperationName());
  whileState.addRegion();
  whileState.addRegion();
  Operation *whileOp = Operation::create(whileState);
  forOp->getRegion(0).front().push_back(whileOp);
  whileOp->getRegion(0).push_back(new Block);
  OperationState useState(b.getUnknownLoc(), "test.use");
  Operation *use = Operation::create(useState);
  whileOp->getRegion(0).front().push_back(use);
  EXPECT_EQ(getLayoutLoopDepth(use), 2u);
  CostEstimate base;
  base.cost.conversionBytesAndSync = 7;
  base.cost.codeSize = 3;
  auto weighted = weightLayoutBoundaryCost(base, 2);
  EXPECT_EQ(weighted.cost.conversionBytesAndSync, 28u);
  EXPECT_EQ(weighted.cost.codeSize, 3u);
  CostEstimate zero;
  zero.cost.codeSize = 1;
  auto zeroWeighted = weightLayoutBoundaryCost(zero, UINT64_MAX);
  EXPECT_EQ(zeroWeighted.cost.conversionBytesAndSync, 0u);
  EXPECT_EQ(zeroWeighted.cost.codeSize, 1u);
  EXPECT_FALSE(zeroWeighted.saturated);
  CostEstimate maximumBoundary;
  maximumBoundary.cost.conversionBytesAndSync = UINT64_MAX;
  auto saturatedMaximum = weightLayoutBoundaryCost(maximumBoundary, 1);
  EXPECT_EQ(saturatedMaximum.cost.conversionBytesAndSync, UINT64_MAX);
  EXPECT_TRUE(saturatedMaximum.saturated);
}
TEST_F(StaticLayoutCostTest, ProducerIsChargedOnceNotPerUse) {
  auto type = RankedTensorType::get({64}, b.getF32Type());
  Block block; auto value = block.addArgument(type, b.getUnknownLoc());
  OperationState state(b.getUnknownLoc(), "test.use"); state.addOperands(value);
  auto *op = Operation::create(state); block.push_back(op);
  LayoutConstraintGraph graph;
  auto def = graph.addVariable(LayoutKind::Distributed, type, "def");
  graph.getVariable(def).value = value;
  auto use = graph.addVariable(LayoutKind::Distributed, type, "use");
  graph.getVariable(use).use = &op->getOpOperand(0);
  CandidateAssignment assignment; assignment.values[def] = encoding();
  assignment.values[use] = encoding(2, true);
  auto cost = evaluateStaticLayoutCost(graph, assignment);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.spillRiskAndRegisters, 2u);
  EXPECT_EQ(cost->cost.replication, 0u);
}
TEST_F(StaticLayoutCostTest, UnknownStaticEstimateFails) {
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed, {}, "unknown");
  CandidateAssignment assignment; assignment.values[id] = encoding();
  EXPECT_TRUE(failed(evaluateStaticLayoutCost(graph, assignment)));
}
TEST_F(StaticLayoutCostTest, ReplicationCountsExtraOwners) {
  auto encodingBase = encoding();
  auto replicated = DistributedEncodingAttr::get(&context, encodingBase.getMap(),
      b.getDenseI64ArrayAttr({2, 32, 2, 1, 1}), b.getI64IntegerAttr(2));
  LayoutConstraintGraph graph; CandidateAssignment assignment;
  auto id = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({64}, b.getF32Type()), "replicated");
  assignment.values[id] = replicated;
  auto cost = evaluateStaticLayoutCost(graph, assignment);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.replication, 1u);
  EXPECT_EQ(cost->cost.spillRiskAndRegisters, 2u);
}
TEST_F(StaticLayoutCostTest, ConversionMultiplicationSaturates) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  var.shapedType = RankedTensorType::get({int64_t{1} << 60}, b.getF64Type());
  LayoutConversionEdge edge;
  auto cost = evaluateLayoutConversionCost(var, encoding(int64_t{1} << 55),
      encoding(int64_t{1} << 55, true), edge);
  ASSERT_TRUE(succeeded(cost)); EXPECT_TRUE(cost->saturated);
  EXPECT_EQ(edge.bytes, UINT64_MAX);
  EXPECT_EQ(cost->cost.conversionBytesAndSync, UINT64_MAX);
}
TEST_F(StaticLayoutCostTest, SharedAliasRootCountsOnceAndDistinctRootsSum) {
  auto type = MemRefType::get({4}, b.getI32Type(), MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(3));
  auto layout = dyn_cast_or_null<StorageLayoutAttr>(parseAttribute(R"mlir(#frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[4],
      outputs=["byte_offset","bit_offset"],output_extents=[16,8],
      map=affine_map<(d)->(4*d,0)>>,
      memory_space=#frisk<memory_space Shared>,alignment=1,vector_granularity=1>
  )mlir", &context));
  ASSERT_TRUE(layout);
  OwningOpRef<func::FuncOp> function(func::FuncOp::create(
      b.getUnknownLoc(), "roots", b.getFunctionType({type,type}, {})));
  auto *block = function->addEntryBlock();
  auto root = block->getArgument(0), other = block->getArgument(1);
  LayoutConstraintGraph graph; CandidateAssignment assignment;
  for (auto [i, value] : llvm::enumerate(SmallVector<Value>{root, root, other})) {
    auto id = graph.addVariable(LayoutKind::Storage, type, "root" + std::to_string(i));
    auto alias = analyzeStorageAlias(value); ASSERT_TRUE(succeeded(alias));
    graph.getVariable(id).storageAlias = *alias;
    assignment.values[id] = layout;
  }
  auto cost = evaluateStaticLayoutCost(graph, assignment);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.sharedBytesAndOccupancy, 32u);
}
TEST_F(StaticLayoutCostTest, SyntheticSharedRootsAreNotConflated) {
  auto type = MemRefType::get({4}, b.getI32Type(), MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(3));
  auto layout = parseAttribute(R"mlir(#frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[4],
      outputs=["byte_offset","bit_offset"],output_extents=[16,8],
      map=affine_map<(d)->(4*d,0)>>,
      memory_space=#frisk<memory_space Shared>,alignment=1,vector_granularity=1>
  )mlir", &context);
  ASSERT_TRUE(layout);
  LayoutConstraintGraph graph; CandidateAssignment assignment;
  for (unsigned i = 0; i < 2; ++i) {
    auto id = graph.addVariable(LayoutKind::Storage, type, "synthetic" + std::to_string(i));
    assignment.values[id] = layout;
  }
  auto cost = evaluateStaticLayoutCost(graph, assignment);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.sharedBytesAndOccupancy, 32u);
}
TEST_F(StaticLayoutCostTest, OffsetAliasesChargeMaximumRootEndInEitherOrder) {
  context.getOrLoadDialect<memref::MemRefDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @offsets(%root: memref<8xi32, 3>) {
      %a = memref.subview %root[1] [2] [1] : memref<8xi32, 3>
          to memref<2xi32, strided<[1], offset: 1>, 3>
      %b = memref.subview %root[5] [2] [1] : memref<8xi32, 3>
          to memref<2xi32, strided<[1], offset: 5>, 3>
      return
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  SmallVector<Value> views;
  module->walk([&](memref::SubViewOp op) { views.push_back(op.getResult()); });
  ASSERT_EQ(views.size(), 2u);
  SmallVector<StorageLayoutAttr> layouts;
  for (int64_t offset : {4, 20}) {
    auto map = AffineLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("d")}), b.getDenseI64ArrayAttr({2}),
        b.getArrayAttr({b.getStringAttr("byte_offset"), b.getStringAttr("bit_offset")}),
        b.getDenseI64ArrayAttr({32, 8}),
        AffineMapAttr::get(AffineMap::get(1, 0,
            {getAffineDimExpr(0, &context) * 4 + offset,
             getAffineConstantExpr(0, &context)}, &context)));
    layouts.push_back(StorageLayoutAttr::get(&context, map,
        MemorySpaceAttr::get(&context, attr::MemorySpace::Shared),
        b.getI64IntegerAttr(1), b.getI64IntegerAttr(1)));
  }
  for (bool reverse : {false, true}) {
    LayoutConstraintGraph graph;
    CandidateAssignment assignment;
    for (unsigned i = 0; i < 2; ++i) {
      unsigned index = reverse ? 1 - i : i;
      auto alias = analyzeStorageAlias(views[index]);
      ASSERT_TRUE(succeeded(alias));
      ASSERT_EQ(verifyStorageAliasCandidate(*alias, layouts[index]).status,
                ProofStatus::Proven);
      auto id = graph.addVariable(LayoutKind::Storage, views[index].getType(),
                                  "offset" + std::to_string(i));
      graph.getVariable(id).storageAlias = *alias;
      assignment.values[id] = layouts[index];
    }
    auto cost = evaluateStaticLayoutCost(graph, assignment);
    ASSERT_TRUE(succeeded(cost));
    // Occupied intervals are [4,12) and [20,28). Count the root-origin span,
    // not their 8-byte lengths, their sum, or the entire 32-byte allocation.
    EXPECT_EQ(cost->cost.sharedBytesAndOccupancy, 28u);
  }
}
TEST_F(StaticLayoutCostTest, GlobalStorageHasNoSharedChargeInComponentAssignment) {
  auto type = MemRefType::get({4}, b.getI32Type(), MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(1));
  auto layout = parseAttribute(R"mlir(#frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[4],
      outputs=["byte_offset","bit_offset"],output_extents=[16,8],
      map=affine_map<(d)->(4*d,0)>>,
      memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>
  )mlir", &context);
  ASSERT_TRUE(layout);
  LayoutConstraintGraph graph; CandidateAssignment assignment;
  graph.addVariable(LayoutKind::Distributed, {}, "outside-component");
  auto id = graph.addVariable(LayoutKind::Storage, type, "global");
  assignment.values[id] = layout;
  auto cost = evaluateStaticLayoutCost(graph, assignment);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.sharedBytesAndOccupancy, 0u);
}
} // namespace
