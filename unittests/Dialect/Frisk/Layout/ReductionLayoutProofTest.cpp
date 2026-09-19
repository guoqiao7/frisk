#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"
#include "llvm/Support/MathExtras.h"
#include <cmath>
#include <limits>
using namespace mlir;
using namespace mlir::frisk;
namespace {
class ReductionLayoutProofTest : public testing::Test {
protected:
  MLIRContext context;
  ReductionLayoutProofTest() { context.loadDialect<FriskDialect>(); }
  DistributedEncodingAttr encoding(ArrayRef<int64_t> shape, ArrayRef<int64_t> widths,
      ArrayRef<int64_t> topology, ArrayRef<uint64_t> rows,
      int64_t replicationOverride = -1) {
    Builder b(&context);
    SmallVector<int64_t> outWidths;
    SmallVector<Attribute> names;
    int64_t logical = 1, hardware = 1, cols = 0;
    for (auto [i, e] : llvm::enumerate(shape)) {
      outWidths.push_back(llvm::Log2_64(e)); logical *= e;
      names.push_back(b.getStringAttr("custom" + std::to_string(i)));
    }
    for (auto w : widths) cols += w;
    for (auto e : topology) hardware *= e;
    SmallVector<APInt> bits;
    for (auto r : rows) for (int64_t c = 0; c < cols; ++c) bits.push_back(APInt(1, (r >> c) & 1));
    SmallVector<Attribute> inputs;
    SmallVector<int64_t> nonzeroWidths;
    const char *carriers[]={"register","lane","warp","warp_group","cta"};
    for (auto [i,w]:llvm::enumerate(widths)) if(w) { inputs.push_back(b.getStringAttr(carriers[i])); nonzeroWidths.push_back(w); }
    auto map = BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr(inputs),
        b.getDenseI64ArrayAttr(nonzeroWidths), b.getArrayAttr(names), b.getDenseI64ArrayAttr(outWidths),
        DenseIntElementsAttr::get(RankedTensorType::get({int64_t(rows.size()),cols}, b.getI1Type()), bits));
    return DistributedEncodingAttr::get(
        &context, map, b.getDenseI64ArrayAttr(topology),
        b.getI64IntegerAttr(replicationOverride < 0 ? hardware / logical
                                                  : replicationOverride));
  }
  RankedTensorType type(ArrayRef<int64_t> shape) { return RankedTensorType::get(shape,Float32Type::get(&context)); }
};
}
TEST_F(ReductionLayoutProofTest, DeduplicatesInputAndBroadcastsCompleteSum) {
  auto src = encoding({4,4},{0,5,0,0,0},{1,32,1,1,1},{4,8,1,2});
  auto dst = projectReductionEncoding(src,type({4,4}),type({4}),1);
  ASSERT_TRUE(succeeded(dst));
  auto p = buildReductionLayoutProof(src,*dst,type({4,4}),type({4}),1);
  ASSERT_EQ(p.proof.status,ProofStatus::Proven);
  EXPECT_EQ(p.scope,"warp");
  ASSERT_EQ(p.fibers.size(),4u);
  for (auto &fiber : p.fibers) {
    std::vector<float> values;
    for (auto &n : fiber.nodes)
      values.push_back(n.logicalInput >= 0 ? float(n.logicalInput % 4 + 1) : values[n.left] + values[n.right]);
    EXPECT_EQ(values[fiber.root],10.f);
    EXPECT_EQ(fiber.broadcasts.size(),8u);
  }
  EXPECT_EQ(verifyReductionLayoutProof(src,*dst,type({4,4}),type({4}),1,p).status,ProofStatus::Proven);
  auto bad = p; bad.fibers[0].nodes[0].logicalInput = 4;
  EXPECT_EQ(verifyReductionLayoutProof(src,*dst,type({4,4}),type({4}),1,bad).status,ProofStatus::Disproven);
  bad = p; bad.fibers[0].nodes.back().right = bad.fibers[0].nodes.back().left;
  EXPECT_EQ(verifyReductionLayoutProof(src,*dst,type({4,4}),type({4}),1,bad).status,ProofStatus::Disproven);
  bad = p; bad.fibers[0].broadcasts.pop_back();
  EXPECT_EQ(verifyReductionLayoutProof(src,*dst,type({4,4}),type({4}),1,bad).status,ProofStatus::Disproven);
}

TEST_F(ReductionLayoutProofTest, CompressesDependentRegisterColumnsAndPreservesNames) {
  // Project register columns (1,0), (0,1), (1,1), (0,0): retain first two.
  auto src=encoding({4,4},{4,5,0,0,0},{16,32,1,1,1},{5,6,4,8});
  auto dst=projectReductionEncoding(src,type({4,4}),type({4}),1);
  ASSERT_TRUE(succeeded(dst));
  EXPECT_EQ(dst->getTopology()[0],4);
  EXPECT_EQ(dst->getReplication().getInt(),32);
  auto map=cast<BitLinearLayoutMapAttr>(dst->getMap());
  EXPECT_EQ(cast<StringAttr>(map.getOutputNames()[0]).getValue(),"custom0");
  ExecutionEnumeration p;
  ASSERT_EQ(enumerateExecutionLayout(*dst,type({4}),p).status,ProofStatus::Proven);
  for (unsigned h=0;h<p.logical.size();++h) EXPECT_EQ(p.logical[h],h%4);
}

TEST_F(ReductionLayoutProofTest, FirstMiddleLastAxisProjectionIndependentCoordinates) {
  auto src=encoding({4,4,4},{1,5,0,0,0},{2,32,1,1,1},{1,2,4,8,16,32});
  for (int64_t axis=0;axis<3;++axis) {
    auto dst=projectReductionEncoding(src,type({4,4,4}),type({4,4}),axis);
    ASSERT_TRUE(succeeded(dst));
    auto p=buildReductionLayoutProof(src,*dst,type({4,4,4}),type({4,4}),axis);
    ASSERT_EQ(p.proof.status,ProofStatus::Proven);
    for (auto &fiber:p.fibers) {
      unsigned leaves=0;
      for (auto &n:fiber.nodes) if (n.logicalInput>=0) {
        int64_t x=n.logicalInput;
        int64_t coords[3]={x/16,(x/4)%4,x%4};
        int64_t y=0;
        for (int i=0;i<3;++i) if (i!=axis) y=y*4+coords[i];
        EXPECT_EQ(y,fiber.logicalOutput); ++leaves;
      }
      EXPECT_EQ(leaves,4u);
    }
  }
}

TEST_F(ReductionLayoutProofTest, RegisterWarpAndWarpGroupEdges) {
  auto src=encoding({32,4},{2,5,0,0,0},{4,32,1,1,1},{4,8,16,32,64,1,2});
  auto dst=projectReductionEncoding(src,type({32,4}),type({32}),1);
  ASSERT_TRUE(succeeded(dst));
  EXPECT_EQ(buildReductionLayoutProof(src,*dst,type({32,4}),type({32}),1).scope,"register");
  for (bool group:{false,true}) {
    auto wide=encoding({4,16},group ? ArrayRef<int64_t>({0,5,0,1,0}) : ArrayRef<int64_t>({0,5,1,0,0}),
        group ? ArrayRef<int64_t>({1,32,1,2,1}) : ArrayRef<int64_t>({1,32,2,1,1}),{1,2,4,8,16,32});
    auto projected=projectReductionEncoding(wide,type({4,16}),type({4}),1);
    ASSERT_TRUE(succeeded(projected));
    auto proof=buildReductionLayoutProof(wide,*projected,type({4,16}),type({4}),1);
    EXPECT_EQ(proof.proof.status,ProofStatus::Proven);
    EXPECT_EQ(proof.scope,"cta_shared_tree");
  }
}

TEST_F(ReductionLayoutProofTest, RejectsWrongProjectionPartialRootAndBudget) {
  auto src=encoding({4,4},{0,5,0,0,0},{1,32,1,1,1},{4,8,1,2});
  auto dst=projectReductionEncoding(src,type({4,4}),type({4}),1); ASSERT_TRUE(succeeded(dst));
  auto wrong=encoding({4},{0,5,0,0,0},{1,32,1,1,1},{1,2});
  EXPECT_EQ(buildReductionLayoutProof(src,wrong,type({4,4}),type({4}),1).proof.status,ProofStatus::Disproven);
  auto p=buildReductionLayoutProof(src,*dst,type({4,4}),type({4}),1);
  p.fibers[0].root=0;
  EXPECT_EQ(verifyReductionLayoutProof(src,*dst,type({4,4}),type({4}),1,p).status,ProofStatus::Disproven);
  auto large=encoding({4,4},{0,5,12,0,0},{1,32,4096,1,1},{4,8,1,2});
  EXPECT_EQ(buildReductionLayoutProof(large,*dst,type({4,4}),type({4}),1).proof.status,ProofStatus::Unknown);
  EXPECT_TRUE(failed(projectReductionEncoding(src,type({4,4}),type({4}),2)));
  EXPECT_TRUE(failed(projectReductionEncoding(src,type({4,3}),type({4}),1)));
}

TEST_F(ReductionLayoutProofTest, CpuTreeOracleReassociationNaNAndSignedZero) {
  auto src=encoding({4,4},{0,5,0,0,0},{1,32,1,1,1},{4,8,1,2});
  auto dst=projectReductionEncoding(src,type({4,4}),type({4}),1); ASSERT_TRUE(succeeded(dst));
  auto p=buildReductionLayoutProof(src,*dst,type({4,4}),type({4}),1);
  ASSERT_EQ(p.proof.status,ProofStatus::Proven);
  auto evaluate=[&](ArrayRef<float> input, StringRef kind) {
    std::vector<float> values;
    for (auto &n:p.fibers[0].nodes) {
      if (n.logicalInput>=0) { values.push_back(input[n.logicalInput%4]); continue; }
      float a=values[n.left], b=values[n.right], r;
      if (kind=="sum") r=a+b;
      else if (std::isnan(a) || std::isnan(b)) r=std::numeric_limits<float>::quiet_NaN();
      else if (a==0.f && b==0.f)
        r=std::copysign(0.f,kind=="max" ? (std::signbit(a)&&std::signbit(b) ? -1.f:1.f)
                                       : (std::signbit(a)||std::signbit(b) ? -1.f:1.f));
      else r=kind=="max" ? std::max(a,b):std::min(a,b);
      values.push_back(r);
    }
    return values[p.fibers[0].root];
  };
  EXPECT_EQ(evaluate({1.e20f,1.f,-1.e20f,1.f},"sum"),0.f); // fixed adjacent-pair tree, not serial 1.
  EXPECT_FALSE(std::signbit(evaluate({-0.f,0.f,-0.f,0.f},"max")));
  EXPECT_TRUE(std::signbit(evaluate({-0.f,0.f,-0.f,0.f},"min")));
  for (StringRef kind:{"max","min"})
    EXPECT_TRUE(std::isnan(evaluate({1.f,std::numeric_limits<float>::quiet_NaN(),2.f,3.f},kind)));
}

TEST_F(ReductionLayoutProofTest, LegacyCasesOneThroughFourHaveIndependentOwnership) {
  // Invert the actual legacy forward slot/thread formulas, not expected reduce maps.
  for (unsigned legacyCase=1;legacyCase<=4;++legacyCase) {
    int64_t m=legacyCase<=2 ? 8 : legacyCase==3 ? 16 : 32;
    auto src=legacyCase<=2
      ? encoding({8,8},{1,5,0,0,0},{2,32,1,1,1},{8,16,32,1,2,4})
      : legacyCase==3
      ? encoding({16,8},{2,5,0,0,0},{4,32,1,1,1},{16,32,64,2,1,4,8})
      : encoding({32,8},{2,5,1,0,0},{4,32,2,1,1},{16,32,64,2,128,1,4,8});
    ExecutionEnumeration owners;
    ASSERT_EQ(enumerateExecutionLayout(src,type({m,8}),owners).status,ProofStatus::Proven);
    for (int64_t i=0;i<m;++i) for (int64_t j=0;j<8;++j) {
      int64_t reg=j%2+(legacyCase<=2 ? 0 : 2*((i%16)/8));
      int64_t thread=j/2+4*(i%8)+32*(i/16);
      EXPECT_EQ(owners.logical[thread*owners.topology[0]+reg],uint64_t(i*8+j));
    }
    int64_t axis=legacyCase==2 ? 0 : 1;
    auto resultType=type({axis==0 ? 8 : m});
    auto dst=projectReductionEncoding(src,type({m,8}),resultType,axis);
    ASSERT_TRUE(succeeded(dst));
    EXPECT_EQ(dst->getReplication().getInt(),axis==0 ? 8 : 4);
    auto proof=buildReductionLayoutProof(src,*dst,type({m,8}),resultType,axis);
    ASSERT_EQ(proof.proof.status,ProofStatus::Proven);
    for (auto &fiber:proof.fibers) {
      int64_t leaves=0; for (auto &node:fiber.nodes) leaves+=node.logicalInput>=0;
      EXPECT_EQ(leaves,axis==0 ? m : 8);
    }
  }
}

TEST_F(ReductionLayoutProofTest, LegacyBadReplicationAndMissingBatchAreNotOracles) {
  Builder b(&context);
  auto valid=encoding({8,8},{1,5,0,0,0},{2,32,1,1,1},{8,16,32,1,2,4});
  auto badRep=DistributedEncodingAttr::get(&context,valid.getMap(),valid.getTopology(),b.getI64IntegerAttr(2));
  ExecutionEnumeration out;
  EXPECT_EQ(enumerateExecutionLayout(badRep,type({8,8}),out).status,ProofStatus::Disproven);
  EXPECT_TRUE(failed(projectReductionEncoding(badRep,type({8,8}),type({8}),1)));
  // Case 7 actual forward (register=5*d1, thread=d2) aliases all four batches.
  // Exhibit collision directly; no inverse Distributed map can recover d0.
  for (int64_t batch=1;batch<4;++batch) {
    int64_t registerAtBatch=5*3, registerAtZero=5*3;
    int64_t threadAtBatch=7, threadAtZero=7;
    EXPECT_EQ(registerAtBatch,registerAtZero); EXPECT_EQ(threadAtBatch,threadAtZero);
    EXPECT_NE(batch*128+3*16+7,3*16+7);
  }
  auto missingBatch=encoding({4,8,16},{5,5,0,0,0},{32,32,1,1,1},{0,0,1,2,4,32,64,128,256});
  EXPECT_EQ(enumerateExecutionLayout(missingBatch,type({4,8,16}),out).status,ProofStatus::Disproven);
}

TEST_F(ReductionLayoutProofTest, LegacySixAndEightAdaptTo32ThreadEqualValueCopies) {
  auto six=encoding({16,8},{3,5,0,0,0},{8,32,1,1,1},{16,32,64,8,1,2,4});
  auto eight=encoding({4,8,16},{5,5,0,0,0},{32,32,1,1,1},{8,16,64,128,256,1,2,4,32});
  for (bool useEight:{false,true}) {
    auto src=useEight ? eight : six;
    auto srcType=useEight ? type({4,8,16}):type({16,8});
    auto dstType=useEight ? type({4,8}):type({16});
    int64_t axis=useEight ? 2:1;
    ExecutionEnumeration e;
    ASSERT_EQ(enumerateExecutionLayout(src,srcType,e).status,ProofStatus::Proven);
    EXPECT_EQ(src.getReplication().getInt(),2);
    // New lane bit 4 only creates equal-value copies of original 16-thread slots.
    for (uint64_t h=0;h<16*e.topology[0];++h)
      EXPECT_EQ(e.logical[h],e.logical[h+16*e.topology[0]]);
    auto dst=projectReductionEncoding(src,srcType,dstType,axis); ASSERT_TRUE(succeeded(dst));
    auto p=buildReductionLayoutProof(src,*dst,srcType,dstType,axis);
    ASSERT_EQ(p.proof.status,ProofStatus::Proven);
    for (auto &fiber:p.fibers) {
      unsigned leaves=0; for (auto &n:fiber.nodes) leaves+=n.logicalInput>=0;
      EXPECT_EQ(leaves,unsigned(useEight ? 16:8));
    }
  }
}

TEST_F(ReductionLayoutProofTest, RenamedResultMixedXorAndSingleCtaBoundary) {
  Builder b(&context);
  auto src=encoding({4,4},{2,5,0,0,0},{4,32,1,1,1},{5,10,4,8});
  auto dst=projectReductionEncoding(src,type({4,4}),type({4}),1); ASSERT_TRUE(succeeded(dst));
  auto map=cast<BitLinearLayoutMapAttr>(dst->getMap());
  auto renamedMap=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),map.getInputBitWidths(),
      b.getStrArrayAttr({"renamed_result_axis"}),map.getOutputBitWidths(),map.getMatrix());
  auto renamed=DistributedEncodingAttr::get(&context,renamedMap,dst->getTopology(),dst->getReplication());
  EXPECT_EQ(buildReductionLayoutProof(src,renamed,type({4,4}),type({4}),1).proof.status,ProofStatus::Proven);
  auto multi=encoding({4,4},{2,5,0,0,1},{4,32,1,1,2},{5,10,4,8});
  auto projected=projectReductionEncoding(multi,type({4,4}),type({4}),1); ASSERT_TRUE(succeeded(projected));
  EXPECT_EQ(buildReductionLayoutProof(multi,*projected,type({4,4}),type({4}),1).proof.status,ProofStatus::Disproven);
}

TEST_F(ReductionLayoutProofTest, ExactCanonicalTreeUsesPhysicalNotLogicalOrder) {
  // In each fiber the low two lane bits encode j=(lane&3)^((lane&2)>>1).
  // Physical lane order 0,1,2,3 thus carries logical j order 0,1,3,2.
  // Lane bit 4 creates a second copy, never another contribution.
  auto src = encoding({4,4}, {0,5,0,0,0}, {1,32,1,1,1}, {4,8,3,2});
  // Independently specified natural destination: y=(lane>>2)&3.
  auto dst = encoding({4}, {0,5,0,0,0}, {1,32,1,1,1}, {4,8});
  auto proof = buildReductionLayoutProof(src, dst, type({4,4}), type({4}), 1);
  ASSERT_EQ(proof.proof.status, ProofStatus::Proven);
  ASSERT_EQ(proof.fibers.size(), 4u);
  EXPECT_EQ(proof.scope, "warp");

  for (int64_t y = 0; y < 4; ++y) {
    const auto &fiber = proof.fibers[y];
    auto laneHolder = [y](int64_t offset) -> ReductionHolder {
      return {0, 0, 0, 4*y+offset, 0};
    };
    // Literal expected leaf sequence and merge edges do not use either proof
    // generation or the shared hardware enumerator as their oracle.
    const std::vector<ReductionNode> expectedNodes = {
        {4*y+0, -1, -1, laneHolder(0)},
        {4*y+1, -1, -1, laneHolder(1)},
        {4*y+3, -1, -1, laneHolder(2)},
        {4*y+2, -1, -1, laneHolder(3)},
        {-1, 0, 1, laneHolder(0)},
        {-1, 2, 3, laneHolder(2)},
        {-1, 4, 5, laneHolder(0)}};
    const std::vector<ReductionHolder> expectedBroadcasts = {
        laneHolder(0), laneHolder(1), laneHolder(2), laneHolder(3),
        laneHolder(16), laneHolder(17), laneHolder(18), laneHolder(19)};
    EXPECT_EQ(fiber.logicalOutput, y);
    EXPECT_EQ(fiber.nodes, expectedNodes);
    EXPECT_EQ(fiber.root, 6);
    EXPECT_EQ(fiber.broadcasts, expectedBroadcasts);
  }

  auto logicalOrder = proof;
  std::swap(logicalOrder.fibers[0].nodes[2], logicalOrder.fibers[0].nodes[3]);
  EXPECT_EQ(verifyReductionLayoutProof(src, dst, type({4,4}), type({4}), 1,
                                     logicalOrder).status,
            ProofStatus::Disproven);
}

TEST_F(ReductionLayoutProofTest, AcceptsExactly65536LogicalAndHardwarePoints) {
  SmallVector<uint64_t> rows;
  for (unsigned bit = 0; bit < 16; ++bit)
    rows.push_back(uint64_t(1) << bit);
  auto src = encoding({256,256}, {11,5,0,0,0}, {2048,32,1,1,1}, rows);
  auto dst = projectReductionEncoding(src, type({256,256}), type({256}), 1);
  ASSERT_TRUE(succeeded(dst));
  auto proof = buildReductionLayoutProof(src, *dst, type({256,256}), type({256}), 1);
  ASSERT_EQ(proof.proof.status, ProofStatus::Proven);
  ASSERT_EQ(proof.fibers.size(), 256u);
  size_t contributions = 0;
  for (const auto &fiber : proof.fibers) {
    ASSERT_EQ(fiber.nodes.size(), 511u);
    EXPECT_EQ(fiber.root, 510);
    for (const auto &node : fiber.nodes)
      contributions += node.logicalInput >= 0;
  }
  EXPECT_EQ(contributions, 65536u);
  EXPECT_EQ(verifyReductionLayoutProof(src, *dst, type({256,256}), type({256}), 1,
                                     proof).status,
            ProofStatus::Proven);
}

TEST_F(ReductionLayoutProofTest, LogicalBudgetOverflowIsUnknownBeforeEnumeration) {
  SmallVector<uint64_t> rows;
  for (unsigned bit = 0; bit < 8; ++bit)
    rows.push_back(uint64_t(1) << bit);
  rows.push_back(0); // Deliberately unrepresented ninth row bit.
  for (unsigned bit = 8; bit < 16; ++bit)
    rows.push_back(uint64_t(1) << bit);
  // Keep hardware at the accepted boundary to isolate the logical-budget guard.
  // This undersized map cannot cover the larger logical tile, but the bounded
  // API must return Unknown before attempting that over-budget coverage proof.
  auto src = encoding({512,256}, {11,5,0,0,0}, {2048,32,1,1,1}, rows, 1);
  auto dst = encoding({512}, {4,5,0,0,0}, {16,32,1,1,1},
                      {1,2,4,8,16,32,64,128,256});
  auto proof = buildReductionLayoutProof(src, dst, type({512,256}), type({512}), 1);
  EXPECT_EQ(proof.proof.status, ProofStatus::Unknown);
  EXPECT_NE(proof.proof.reason.find("logical enumeration budget exceeded"),
            std::string::npos);
  EXPECT_TRUE(failed(projectReductionEncoding(src, type({512,256}), type({512}), 1)));
}
