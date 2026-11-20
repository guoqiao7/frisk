#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "LayoutCostObserverTestSupport.h"
#include <algorithm>
#include <tuple>

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
TEST(LayoutCostModelTest, LegacyTargetObserverDefaultsToActualAggregate) {
  struct LegacyTarget : LayoutTarget {
    void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
    LogicalResult verifyCandidate(const LayoutVar &, Attribute, Location) const override { return success(); }
    FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {
      CostVector cost; cost.codeSize = UINT64_MAX; return cost;
    }
  } target;
  LayoutConstraintGraph graph;
  CandidateAssignment assignment;
  TestCostObserver observer;
  auto result = static_cast<LayoutTarget &>(target).evaluate(graph, assignment, &observer);
  ASSERT_TRUE(succeeded(result));
  ASSERT_EQ(observer.parts.size(), 1u);
  EXPECT_EQ(observer.parts[0].kind, "target-aggregate");
  observer.expectSum(*result);
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
  DistributedEncodingAttr columnEncoding(ArrayRef<uint64_t> columns,
      ArrayRef<int64_t> topology, ArrayRef<int64_t> logicalWidths,
      bool reverseInputs = false) {
    SmallVector<unsigned> widths, offsets;
    unsigned inputBits = 0, outputBits = 0;
    uint64_t holders = 1;
    for (int64_t extent : topology) {
      offsets.push_back(inputBits);
      widths.push_back(llvm::Log2_64(extent));
      inputBits += widths.back(); holders *= extent;
    }
    for (int64_t width : logicalWidths) outputBits += width;
    EXPECT_EQ(columns.size(), inputBits);
    SmallVector<unsigned> order{0,1,2,3,4};
    if (reverseInputs) std::reverse(order.begin(), order.end());
    SmallVector<Attribute> names, outputs;
    SmallVector<int64_t> suppliedWidths;
    const char *carriers[] = {"register", "lane", "warp", "warp_group", "cta"};
    for (unsigned c : order) if (widths[c]) {
      names.push_back(b.getStringAttr(carriers[c]));
      suppliedWidths.push_back(widths[c]);
    }
    for (unsigned i = 0; i < logicalWidths.size(); ++i)
      outputs.push_back(b.getStringAttr("d" + std::to_string(i)));
    SmallVector<APInt> entries;
    for (unsigned row = 0; row < outputBits; ++row)
      for (unsigned c : order)
        for (unsigned bit = 0; bit < widths[c]; ++bit)
          entries.emplace_back(1, (columns[offsets[c]+bit] >> row) & 1);
    auto map = BitLinearLayoutMapAttr::get(&context, b.getArrayAttr(names),
        b.getDenseI64ArrayAttr(suppliedWidths), b.getArrayAttr(outputs),
        b.getDenseI64ArrayAttr(logicalWidths), DenseIntElementsAttr::get(
          RankedTensorType::get({outputBits, inputBits}, b.getI1Type()), entries));
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr(topology), b.getI64IntegerAttr(holders >> outputBits));
  }
  ConversionExchangeKind ownerOracle(DistributedEncodingAttr source,
      DistributedEncodingAttr target, RankedTensorType type) {
    ExecutionEnumeration s, t;
    EXPECT_EQ(enumerateExecutionLayout(source,type,s).status,ProofStatus::Proven);
    EXPECT_EQ(enumerateExecutionLayout(target,type,t).status,ProofStatus::Proven);
    auto result = ConversionExchangeKind::Register;
    for (unsigned d = 0; d < t.logical.size(); ++d) {
      auto dst = t.hardware(d); bool thread = false, warp = false;
      for (unsigned a = 0; a < s.logical.size(); ++a) {
        if (s.logical[a] != t.logical[d]) continue;
        auto src = s.hardware(a);
        bool sameWarp = src[0] == dst[0] && src[1] == dst[1] && src[2] == dst[2];
        warp |= sameWarp; thread |= sameWarp && src[3] == dst[3];
      }
      if (!warp) return ConversionExchangeKind::Shared;
      if (!thread) result = ConversionExchangeKind::Shuffle;
    }
    return result;
  }
};

TEST_F(StaticLayoutCostTest, ActualSm90TargetVersionsPreserveMmaGeometryDependencies) {
  context.loadDialect<arith::ArithDialect,memref::MemRefDialect>();
  auto module=parseSourceString<ModuleOp>(R"mlir(
    module attributes {frisk.target="sm_90a"} {
      func.func @mma_work() {
        %ar=memref.alloc() {alignment=1024:i64} : memref<128x16xf16,3>
        %a=frisk.layout_view %ar : memref<128x16xf16,3> -> memref<128x16xf16,3>
        %br=memref.alloc() {alignment=1024:i64} : memref<16x128xf16,3>
        %b=frisk.layout_view %br : memref<16x128xf16,3> -> memref<16x128xf16,3>
        %c=arith.constant dense<0.0> : tensor<128x128xf32>
        %r="frisk.mma"(%a,%b,%c) {m=128:i64,n=128:i64,k=16:i64,
          frisk.execution_threads=256:i64,policy=#frisk<gemm_warp_policy Auto>} :
          (memref<128x16xf16,3>,memref<16x128xf16,3>,tensor<128x128xf32>) -> tensor<128x128xf32>
        return
      }
    })mlir",&context);
  ASSERT_TRUE(module);
  auto legacy=createSM90LayoutTarget();
  auto graph=collectLayoutConstraints(*module,*legacy);
  ASSERT_TRUE(succeeded(graph));
  const LayoutConstraint *mma=nullptr;
  for(const auto &c:graph->getConstraints())
    if(c.instruction && c.instruction->source->getName().getStringRef()=="frisk.mma") mma=&c;
  ASSERT_TRUE(mma); ASSERT_EQ(mma->vars.size(),5u);
  auto geometry=mma->vars.back();
  EXPECT_EQ(graph->getVariable(geometry).kind,LayoutKind::Instruction);
  ASSERT_GT(graph->getVariable(geometry).candidates.size(),1u);
  SM90CostProfile profile; profile.coefficients.fill(1);
  profile.provenance="mma-dependency-contract-unit-test";
  ASSERT_TRUE(succeeded(verifySM90CostProfile(profile)));
  SmallVector<std::unique_ptr<LayoutTarget>> targets;
  targets.push_back(std::move(legacy));
  targets.push_back(createSM90LayoutTarget(LayoutBoundaryFrequencyMode::StaticTripCount));
  targets.push_back(createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,profile));
  targets.push_back(createSM90LayoutTarget(LayoutBoundaryFrequencyMode::StaticTripCount,profile));
  // Exercise factory-produced identities and a genuine five-role instruction
  // graph. No test reconstructs the recognized model-ID strings or predicate.
  for(const auto &target:targets) {
    SCOPED_TRACE(target->costModelID().str());
    EXPECT_EQ(experimentInstructionDependencies(*graph,*mma,*target),
              (SmallVector<LayoutVarID>{geometry}));
  }
  class UnknownWorkTarget final:public LayoutTarget {
  public:
    void enumerateCandidates(const LayoutVar &,SmallVectorImpl<LayoutCandidate> &)const override {}
    LogicalResult verifyCandidate(const LayoutVar &,Attribute,Location)const override {return success();}
    FailureOr<CostVector> evaluate(const CandidateAssignment &)const override {return CostVector{};}
    StringRef costModelID()const override {return "future-target-with-unknown-mma-work";}
  } unknown;
  EXPECT_EQ(experimentInstructionDependencies(*graph,*mma,unknown),
            SmallVector<LayoutVarID>(mma->vars));
}

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
  EXPECT_EQ(edge.costProvenance, "symbolic-subspace-proof-v1/register");
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
  EXPECT_EQ(edge.costProvenance, "symbolic-subspace-proof-v1/shuffle");

  var.shapedType = RankedTensorType::get({128}, b.getF32Type());
  auto twoWarps = permutedEncoding(2, 2, 0, 0);
  auto warpSwap = permutedEncoding(2, 2, 1, 6);
  auto shared = evaluateLayoutConversionCost(var, twoWarps, warpSwap, edge);
  ASSERT_TRUE(succeeded(shared));
  EXPECT_EQ(edge.exchange, ConversionExchangeKind::Shared);
  EXPECT_EQ(edge.bytes, 4u * (128 + 128));
  EXPECT_EQ(edge.synchronizationCost, 2u);
  EXPECT_EQ(shared->cost.conversionBytesAndSync, 1024u + 4u * 64 * 2);
  EXPECT_EQ(edge.costProvenance, "symbolic-subspace-proof-v1/shared");
}
TEST_F(StaticLayoutCostTest, ConversionRejectsMissingOrDynamicInformation) {
  LayoutVar var; LayoutConversionEdge edge;
  EXPECT_TRUE(failed(evaluateLayoutConversionCost(var, {}, {}, edge)));
  var.shapedType = RankedTensorType::get({ShapedType::kDynamic}, b.getF32Type());
  EXPECT_TRUE(failed(evaluateLayoutConversionCost(var, encoding(), encoding(2,true), edge)));
}
TEST_F(StaticLayoutCostTest, ElementBitsRoundUpAndUnsupportedRepresentationFallsBack) {
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

// The source and target matrices here include nonzero dependent columns, not
// just independent coordinate permutations or zero broadcast bases.
TEST_F(StaticLayoutCostTest, SymbolicScopesMatchIndependentBoundedOwnerOracle) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  auto type = RankedTensorType::get({64}, b.getF32Type()); var.shapedType = type;
  uint32_t state = 0x51c0ffee;
  auto next = [&]() { state = state * 1664525u + 1013904223u; return state; };
  for (unsigned trial = 0; trial < 32; ++trial) {
    SmallVector<uint64_t> a{1,2,4,8,16,32,0}, z = a;
    for (auto *columns : {&a, &z})
      for (unsigned step = 0; step < 24; ++step) {
        unsigned lhs = next()%7, rhs = next()%7;
        if (lhs != rhs) (*columns)[lhs] ^= (*columns)[rhs];
      }
    auto src = columnEncoding(a,{2,32,2,1,1},{6}, trial&1);
    auto dst = columnEncoding(z,{2,32,2,1,1},{6}, !(trial&1));
    LayoutConversionEdge edge;
    auto cost = evaluateLayoutConversionCost(var,src,dst,edge);
    ASSERT_TRUE(succeeded(cost));
    EXPECT_EQ(edge.exchange,ownerOracle(src,dst,type)) << "trial " << trial;
  }
  auto src = columnEncoding({1,1,2,4,8,16,32},{4,32,1,1,1},{6},true);
  auto dst = columnEncoding({1,2,1,4,8,16,32},{4,32,1,1,1},{6});
  LayoutConversionEdge edge;
  ASSERT_TRUE(succeeded(evaluateLayoutConversionCost(var,src,dst,edge)));
  EXPECT_EQ(edge.exchange,ConversionExchangeKind::Shuffle);
  EXPECT_EQ(edge.exchange,ownerOracle(src,dst,type));
}

TEST_F(StaticLayoutCostTest, SymbolicScopesHandleDifferentRegisterWidthsAndUnitTile) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  var.shapedType = RankedTensorType::get({32},b.getF32Type());
  auto src = columnEncoding({1,2,4,8,16},{1,32,1,1,1},{5});
  auto dst = columnEncoding({1,1,2,4,8,16},{2,32,1,1,1},{5},true);
  LayoutConversionEdge edge;
  ASSERT_TRUE(succeeded(evaluateLayoutConversionCost(var,src,dst,edge)));
  EXPECT_EQ(edge.exchange,ConversionExchangeKind::Shuffle);
  EXPECT_EQ(edge.bytes,4u*(32+64));
  ASSERT_TRUE(succeeded(evaluateLayoutConversionCost(var,dst,src,edge)));
  EXPECT_EQ(edge.exchange,ConversionExchangeKind::Register);
  EXPECT_EQ(edge.bytes,4u*32);
  var.shapedType = RankedTensorType::get({1},b.getF32Type());
  auto unitSrc = columnEncoding({0,0,0,0,0},{1,32,1,1,1},{0});
  auto unitDst = columnEncoding({0,0,0,0,0,0},{2,32,1,1,1},{0},true);
  ASSERT_TRUE(succeeded(evaluateLayoutConversionCost(var,unitSrc,unitDst,edge)));
  EXPECT_EQ(edge.exchange,ConversionExchangeKind::Register);
  EXPECT_EQ(edge.synchronizationCost,0u);
  EXPECT_EQ(edge.bytes,4u*64);
}

TEST_F(StaticLayoutCostTest, SymbolicScopesExceedBothOldEnumerationBounds) {
  LayoutVar var; var.kind = LayoutKind::Distributed;
  // Each coordinate axis fits the checked bit-linear representation, while
  // 131072 logical elements and 262144 holders/layout exceed the old bounds.
  var.shapedType = RankedTensorType::get({512,256},b.getF16Type());
  SmallVector<uint64_t> columns;
  for (unsigned i=0;i<17;++i) columns.push_back(uint64_t{1}<<i);
  columns.insert(columns.begin()+10,0); // dependent register bit, replication 2
  auto src=columnEncoding(columns,{2048,32,4,1,1},{9,8});
  for (auto [lhs,rhs,expected] : {
      std::tuple<unsigned,unsigned,ConversionExchangeKind>{0,1,ConversionExchangeKind::Register},
      {0,11,ConversionExchangeKind::Shuffle},
      {11,16,ConversionExchangeKind::Shared}}) {
    auto swapped=columns; std::swap(swapped[lhs],swapped[rhs]);
    auto dst=columnEncoding(swapped,{2048,32,4,1,1},{9,8},true);
    LayoutConversionEdge edge;
    auto cost=evaluateLayoutConversionCost(var,src,dst,edge);
    ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(edge.exchange,expected);
    EXPECT_EQ(edge.synchronizationCost,expected==ConversionExchangeKind::Shared?2u:0u);
    EXPECT_EQ(edge.bytes,expected==ConversionExchangeKind::Register?2u*262144:2u*524288);
  }
}

TEST_F(StaticLayoutCostTest, SymbolicScopesHandleMillionsOfReplicatedHolders) {
  LayoutVar var; var.kind=LayoutKind::Distributed;
  var.shapedType=RankedTensorType::get({64},b.getF32Type());
  SmallVector<uint64_t> columns(16,0);
  for(unsigned i=0;i<6;++i) columns.push_back(uint64_t{1}<<i);
  auto src=columnEncoding(columns,{65536,32,2,1,1},{6});
  auto swapped=columns; std::swap(swapped[16],swapped[21]);
  auto dst=columnEncoding(swapped,{65536,32,2,1,1},{6},true);
  LayoutConversionEdge edge;
  auto cost=evaluateLayoutConversionCost(var,src,dst,edge);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(edge.exchange,ConversionExchangeKind::Shared);
  EXPECT_EQ(edge.synchronizationCost,2u);
  EXPECT_EQ(edge.bytes,4u*(4194304u+4194304u));
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
  TestCostObserver observer;
  auto observed = evaluateStaticLayoutCost(graph, assignment, &observer);
  ASSERT_TRUE(succeeded(observed));
  EXPECT_EQ(observed->cost, cost->cost);
  EXPECT_EQ(observed->saturated, cost->saturated);
  ASSERT_EQ(observer.parts.size(), 1u);
  EXPECT_EQ(observer.parts[0].owner, def);
  observer.expectSum(*observed);
}
TEST_F(StaticLayoutCostTest, UnknownStaticEstimateFails) {
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed, {}, "unknown");
  CandidateAssignment assignment; assignment.values[id] = encoding();
  EXPECT_TRUE(failed(evaluateStaticLayoutCost(graph, assignment)));
}
TEST_F(StaticLayoutCostTest, ObservedChargesSaturateAndIgnoreAssignmentInsertionOrder) {
  LayoutConstraintGraph graph;
  auto type = RankedTensorType::get({int64_t{1} << 61}, b.getI1Type());
  auto layout = permutedEncoding(int64_t{1} << 60, 1, 0, 0, 2);
  for (unsigned i = 0; i < 17; ++i)
    graph.addVariable(LayoutKind::Distributed, type, "v" + std::to_string(i));
  for (bool reverse : {false, true}) {
    CandidateAssignment assignment;
    for (unsigned i = 0; i < 17; ++i)
      assignment.values[reverse ? 16 - i : i] = layout;
    TestCostObserver observer;
    auto plain = evaluateStaticLayoutCost(graph, assignment);
    auto observed = evaluateStaticLayoutCost(graph, assignment, &observer);
    ASSERT_TRUE(succeeded(plain));
    ASSERT_TRUE(succeeded(observed));
    EXPECT_EQ(plain->cost, observed->cost);
    EXPECT_EQ(plain->saturated, observed->saturated);
    EXPECT_TRUE(observed->saturated);
    EXPECT_EQ(observed->cost.spillRiskAndRegisters, UINT64_MAX);
    ASSERT_EQ(observer.parts.size(), 17u);
    for (unsigned i = 0; i < 17; ++i) EXPECT_EQ(observer.parts[i].owner, i);
    observer.expectSum(*observed);
  }
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
  TestCostObserver observer;
  auto observed = evaluateStaticLayoutCost(graph, assignment, &observer);
  ASSERT_TRUE(succeeded(observed));
  EXPECT_EQ(observed->cost, cost->cost);
  EXPECT_EQ(observed->saturated, cost->saturated);
  ASSERT_EQ(observer.parts.size(), 2u);
  EXPECT_EQ(observer.parts[0].kind, "shared-root");
  EXPECT_EQ(observer.parts[0].owner, 0u);
  EXPECT_EQ(observer.parts[1].owner, 2u);
  observer.expectSum(*observed);
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
    TestCostObserver observer;
    auto observed = evaluateStaticLayoutCost(graph, assignment, &observer);
    ASSERT_TRUE(succeeded(observed));
    EXPECT_EQ(observed->cost, cost->cost);
    EXPECT_EQ(observed->saturated, cost->saturated);
    ASSERT_EQ(observer.parts.size(), 1u);
    EXPECT_EQ(observer.parts[0].owner, 0u);
    observer.expectSum(*observed);
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
TEST_F(StaticLayoutCostTest, StaticFrequencyWeightsOnlyDynamicConversionAndClone) {
  context.getOrLoadDialect<arith::ArithDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @frequency() {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %hundred = arith.constant 100 : index
      scf.for %i = %zero to %hundred step %one {
        %marker = arith.constant 0 : i32
      }
      return
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  Operation *at = nullptr;
  module->walk([&](arith::ConstantOp op) { if (op.getType().isInteger(32)) at = op; });
  ASSERT_TRUE(at);
  LayoutVar var; var.kind = LayoutKind::Distributed;
  var.shapedType = RankedTensorType::get({128}, b.getF32Type());
  auto a = permutedEncoding(4, 1, 0, 0), z = permutedEncoding(4, 1, 0, 1);
  LayoutConversionEdge edge; edge.insertionPoint = at;
  auto legacy = evaluateLayoutConversionCost(var, a, z, edge);
  ASSERT_TRUE(succeeded(legacy)); EXPECT_EQ(legacy->cost.conversionBytesAndSync, 1024u);
  auto cost = evaluateLayoutConversionCost(var, a, z, edge, LayoutBoundaryFrequencyMode::StaticTripCount);
  ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.conversionBytesAndSync, 51200u);
  EXPECT_EQ(cost->cost.codeSize, 1u); EXPECT_EQ(edge.bytes, 512u);
  EXPECT_EQ(edge.frequencyFactor, 100u); EXPECT_FALSE(edge.frequencyUsesProxy);
  auto clone = evaluateLayoutCloneCost(var, a, at, LayoutBoundaryFrequencyMode::StaticTripCount);
  ASSERT_TRUE(succeeded(clone)); EXPECT_EQ(clone->cost.conversionBytesAndSync, 51200u);
  EXPECT_EQ(clone->cost.spillRiskAndRegisters, 4u); EXPECT_EQ(clone->cost.codeSize, 1u);
}
TEST_F(StaticLayoutCostTest, ZeroWorkAndEmptyLoopsPreserveSaturationSemantics) {
  context.getOrLoadDialect<arith::ArithDialect>();
  for (bool empty : {false,true}) {
    std::string text = R"mlir(func.func @huge() {
      %lb = arith.constant -9223372036854775808 : index
      %ub = arith.constant 9223372036854775807 : index
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.for %i = %lb to %ub step %one {
        scf.for %j = %lb to %ub step %one {
          scf.for %k = %zero to )mlir";
    text += empty ? "%zero" : "%one";
    text += R"mlir( step %one { %marker = arith.constant 0 : i32 }
        }
      }
      return
    })mlir";
    auto module = parseSourceString<ModuleOp>(text, &context); ASSERT_TRUE(module);
    Operation *at = nullptr;
    module->walk([&](arith::ConstantOp op) { if (op.getType().isInteger(32)) at = op; });
    LayoutVar var; var.kind = LayoutKind::Distributed;
    var.shapedType = RankedTensorType::get({64}, b.getF32Type());
    LayoutConversionEdge edge; edge.insertionPoint = at;
    auto identity = evaluateLayoutConversionCost(var, encoding(), encoding(), edge, LayoutBoundaryFrequencyMode::StaticTripCount);
    ASSERT_TRUE(succeeded(identity)); EXPECT_EQ(identity->cost.conversionBytesAndSync, 0u);
    EXPECT_FALSE(identity->saturated); EXPECT_EQ(edge.frequencySaturated, !empty);
    if (empty) {
      var.shapedType = RankedTensorType::get({int64_t{1} << 62}, b.getF64Type());
      auto cost = evaluateLayoutCloneCost(var, encoding(int64_t{1} << 57), at, LayoutBoundaryFrequencyMode::StaticTripCount);
      ASSERT_TRUE(succeeded(cost)); EXPECT_EQ(cost->cost.conversionBytesAndSync, 0u);
      EXPECT_TRUE(cost->saturated); EXPECT_EQ(cost->cost.codeSize, 1u);
    }
  }
}

TEST_F(StaticLayoutCostTest, PureValidationCacheReplaysEveryVariableCharge) {
  LayoutConstraintGraph graph;
  auto type = RankedTensorType::get({64}, b.getF32Type());
  graph.addVariable(LayoutKind::Distributed, type, "a");
  graph.addVariable(LayoutKind::Distributed, type, "b");
  ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
  CandidateAssignment assignment;
  assignment.values[0] = assignment.values[1] = encoding();
  LayoutProfile profile(true);
  LayoutProfile::Invocation invocation(&profile);
  TestCostObserver cold, warm, uncached;
  auto first = evaluateStaticLayoutCost(graph, assignment, &cold);
  auto second = evaluateStaticLayoutCost(graph, assignment, &warm);
  ASSERT_TRUE(succeeded(first)); ASSERT_TRUE(succeeded(second));
  EXPECT_EQ(first->cost, second->cost);
  EXPECT_EQ(first->saturated, second->saturated);
  EXPECT_EQ(first->cost.spillRiskAndRegisters, 4u);
  EXPECT_EQ(graph.getStaticCostValidationCache().size(), 1u);
  const auto &metric = profile.get(LayoutProfileKind::StaticCostValidationCache);
  EXPECT_EQ(metric.misses, 1u); EXPECT_EQ(metric.hits, 3u);
  EXPECT_EQ(profile.get(LayoutProfileKind::StaticCostValidation).count, 1u);
  graph.setCostQueryCacheOptions(false);
  auto third = evaluateStaticLayoutCost(graph, assignment, &uncached);
  ASSERT_TRUE(succeeded(third)); EXPECT_EQ(first->cost, third->cost);
  ASSERT_EQ(cold.parts.size(), 2u);
  ASSERT_EQ(warm.parts.size(), cold.parts.size());
  ASSERT_EQ(uncached.parts.size(), cold.parts.size());
  for (unsigned i = 0; i < cold.parts.size(); ++i) {
    EXPECT_EQ(cold.parts[i].kind, warm.parts[i].kind);
    EXPECT_EQ(cold.parts[i].owner, warm.parts[i].owner);
    EXPECT_EQ(cold.parts[i].cost.cost, warm.parts[i].cost.cost);
    EXPECT_EQ(cold.parts[i].kind, uncached.parts[i].kind);
    EXPECT_EQ(cold.parts[i].owner, uncached.parts[i].owner);
    EXPECT_EQ(cold.parts[i].cost.cost, uncached.parts[i].cost.cost);
  }
  cold.expectSum(*first); warm.expectSum(*second); uncached.expectSum(*third);
}
TEST_F(StaticLayoutCostTest, ValidationCacheIncludesTypeAndNeverCachesFailure) {
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({64}, b.getF32Type()), "a");
  ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
  CandidateAssignment assignment; assignment.values[id] = encoding();
  ASSERT_TRUE(succeeded(evaluateStaticLayoutCost(graph, assignment)));
  ASSERT_EQ(graph.getStaticCostValidationCache().size(), 1u);
  // Even an immutable-type change through the public mutable view cannot
  // reuse a validation performed for a different type.
  graph.getVariable(id).shapedType = RankedTensorType::get({128}, b.getF32Type());
  ScopedDiagnosticHandler capture(&context, [](Diagnostic &) { return success(); });
  LayoutProfile profile(true); LayoutProfile::Invocation invocation(&profile);
  EXPECT_TRUE(failed(evaluateStaticLayoutCost(graph, assignment)));
  EXPECT_TRUE(failed(evaluateStaticLayoutCost(graph, assignment)));
  EXPECT_EQ(graph.getStaticCostValidationCache().size(), 1u);
  EXPECT_EQ(profile.get(LayoutProfileKind::StaticCostValidationCache).misses, 2u);
  EXPECT_EQ(profile.get(LayoutProfileKind::StaticCostValidation).count, 2u);
}
TEST_F(StaticLayoutCostTest, ValidationCacheCapacityAndUnfinalizedFallback) {
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({64}, b.getF32Type()), "a");
  CandidateAssignment assignment; assignment.values[id] = encoding();
  ASSERT_TRUE(succeeded(evaluateStaticLayoutCost(graph, assignment)));
  EXPECT_TRUE(graph.getStaticCostValidationCache().empty());
  ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
  graph.setCostQueryCacheOptions(true, 1);
  auto a = evaluateStaticLayoutCost(graph, assignment); ASSERT_TRUE(succeeded(a));
  assignment.values[id] = encoding(2, true);
  auto z = evaluateStaticLayoutCost(graph, assignment); ASSERT_TRUE(succeeded(z));
  EXPECT_EQ(a->cost, z->cost);
  EXPECT_EQ(graph.getStaticCostValidationCache().size(), 1u);
  graph.setCostQueryCacheOptions(true, 0);
  EXPECT_TRUE(graph.getStaticCostValidationCache().empty());
  auto bypass = evaluateStaticLayoutCost(graph, assignment);
  ASSERT_TRUE(succeeded(bypass)); EXPECT_EQ(z->cost, bypass->cost);
  EXPECT_TRUE(graph.getStaticCostValidationCache().empty());
}
} // namespace
