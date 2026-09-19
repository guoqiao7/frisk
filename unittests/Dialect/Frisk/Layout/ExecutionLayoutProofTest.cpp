#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class ExecutionLayoutProofTest : public testing::Test {
protected:
  ExecutionLayoutProofTest() { context.loadDialect<FriskDialect>(); }
  DistributedEncodingAttr encoding(ArrayRef<StringRef> names,
      ArrayRef<int64_t> widths, ArrayRef<int64_t> topology,
      ArrayRef<unsigned> rows, unsigned outBits = 2, int64_t replication = 1) {
    Builder b(&context);
    unsigned columns = 0;
    for (auto width : widths) columns += width;
    SmallVector<APInt> bits;
    for (unsigned row : rows)
      for (unsigned c = 0; c < columns; ++c)
        bits.push_back(APInt(1, (row >> c) & 1));
    auto matrix = DenseIntElementsAttr::get(
        RankedTensorType::get({int64_t(outBits), int64_t(columns)}, b.getI1Type()), bits);
    auto map = BitLinearLayoutMapAttr::get(&context, b.getStrArrayAttr(names),
        b.getDenseI64ArrayAttr(widths), b.getStrArrayAttr({"x"}),
        b.getDenseI64ArrayAttr({int64_t(outBits)}), matrix);
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr(topology), b.getI64IntegerAttr(replication));
  }
  RankedTensorType type(unsigned bits = 32) {
    return RankedTensorType::get({4}, IntegerType::get(&context, bits));
  }
  std::pair<StorageAliasInfo, StorageAliasFootprint> storage(unsigned bits = 32,
      unsigned stride = 0, unsigned alignment = 16, unsigned offset = 0) {
    StorageAliasInfo info;
    info.rootType = info.viewType = MemRefType::get({4}, IntegerType::get(&context, bits));
    info.viewToRoot = AffineMap::getMultiDimIdentityMap(1, &context);
    info.rootAlignment = alignment;
    info.upperBit = 1024;
    StorageAliasFootprint footprint;
    footprint.rootType = info.rootType;
    footprint.proof = {ProofStatus::Proven, {}, "test footprint"};
    for (int64_t i = 0; i < 4; ++i) {
      uint64_t start = offset + i * (stride ? stride : bits);
      footprint.entries.push_back({{i}, {i}, start, start + bits});
    }
    return {info, footprint};
  }
  LayoutProof vector(DistributedEncodingAttr d, unsigned width,
      unsigned bits = 32, unsigned stride = 0, unsigned alignment = 16,
      AccessKind access = AccessKind::Read, StringRef policy = "all") {
    auto [info, footprint] = storage(bits, stride, alignment);
    return proveExecutionVectorAccess(d, info, footprint, width, access, policy);
  }
  MLIRContext context;
};

TEST_F(ExecutionLayoutProofTest, ReplicatedOwnersAndUnknownPolicy) {
  auto d = encoding({"lane"}, {5}, {1,32,1,1,1}, {1,2}, 2, 8);
  EXPECT_EQ(proveExecutionOwnership(d, type(), "all").status, ProofStatus::Disproven);
  EXPECT_EQ(proveExecutionOwnership(d, type(), "first_owner").status, ProofStatus::Proven);
  EXPECT_EQ(proveExecutionOwnership(d, type(), "invalid").status, ProofStatus::Disproven);
}
TEST_F(ExecutionLayoutProofTest, InjectiveAndMissingCoverage) {
  auto d = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2});
  EXPECT_EQ(proveExecutionOwnership(d, type(), "all").status, ProofStatus::Proven);
  auto missing = encoding({"register"}, {2}, {4,1,1,1,1}, {1,0});
  auto proof = proveExecutionOwnership(missing, type(), "first_owner");
  EXPECT_EQ(proof.status, ProofStatus::Disproven);
  EXPECT_FALSE(proof.counterexample.empty());
}
TEST_F(ExecutionLayoutProofTest, OmittedCarriersAndInputOrder) {
  auto a = encoding({"register", "lane"}, {2,1}, {4,2,1,1,1}, {1,2}, 2, 2);
  auto b = encoding({"lane", "register"}, {1,2}, {4,2,1,1,1}, {2,4}, 2, 2);
  auto c = encoding({"register"}, {2}, {4,2,1,1,1}, {1,2}, 2, 2);
  for (auto d : {a,b,c}) {
    EXPECT_EQ(vector(d, 16, 32, 0, 16, AccessKind::Write, "first_owner").status, ProofStatus::Proven);
    EXPECT_EQ(proveExecutionOwnership(d, type(), "all").status, ProofStatus::Disproven);
  }
}
TEST_F(ExecutionLayoutProofTest, MalformedTopologyAndBudget) {
  auto bad = encoding({"register"}, {2}, {8,1,1,1,1}, {1,2});
  EXPECT_EQ(proveExecutionOwnership(bad, type(), "all").status, ProofStatus::Disproven);
  auto large = encoding({"register"}, {2}, {4,32,1024,1,1}, {1,2}, 2, 32768);
  EXPECT_EQ(proveExecutionOwnership(large, type(), "first_owner").status, ProofStatus::Unknown);
}
TEST_F(ExecutionLayoutProofTest, RegisterVectorsAndIllegalWidth) {
  auto d = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2});
  EXPECT_EQ(vector(d, 16).status, ProofStatus::Proven);
  EXPECT_EQ(vector(d, 3).status, ProofStatus::Disproven);
  EXPECT_EQ(vector(d, 16, 32, 0, 4).status, ProofStatus::Disproven);
  EXPECT_EQ(vector(d, 16, 32, 64).status, ProofStatus::Disproven);
  auto [info, fp] = storage(32, 0, 16, 32);
  EXPECT_EQ(proveExecutionVectorAccess(d, info, fp, 16, AccessKind::Read, "all").status, ProofStatus::Disproven);
}
TEST_F(ExecutionLayoutProofTest, LanesCannotCombineAndElectedTailsFail) {
  auto lane = encoding({"lane"}, {2}, {1,4,1,1,1}, {1,2});
  EXPECT_EQ(vector(lane, 16).status, ProofStatus::Disproven);
  auto tails = encoding({"register", "lane"}, {2,1}, {4,2,1,1,1}, {1,4}, 2, 2);
  EXPECT_EQ(vector(tails, 16, 32, 0, 16, AccessKind::Write, "first_owner").status, ProofStatus::Disproven);
}
TEST_F(ExecutionLayoutProofTest, PackedBaselineAndUnprovenFootprint) {
  auto d = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2});
  EXPECT_EQ(vector(d, 1, 3).status, ProofStatus::Proven);
  EXPECT_EQ(vector(d, 2, 3).status, ProofStatus::Disproven);
  auto [info, fp] = storage();
  fp.proof.status = ProofStatus::Unknown;
  EXPECT_NE(proveExecutionVectorAccess(d, info, fp, 1, AccessKind::Read, "all").status, ProofStatus::Proven);
  fp.proof.status = ProofStatus::Proven;
  fp.entries.pop_back();
  EXPECT_EQ(proveExecutionVectorAccess(d, info, fp, 1, AccessKind::Read, "all").status, ProofStatus::Disproven);
}
TEST_F(ExecutionLayoutProofTest, RejectsFootprintFromDifferentViewOfSameRoot) {
  auto d = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2});
  auto [info, fp] = storage();
  info.rootType = MemRefType::get({8}, IntegerType::get(&context, 32));
  fp.rootType = info.rootType;
  EXPECT_EQ(proveExecutionVectorAccess(d, info, fp, 16, AccessKind::Read, "all").status,
            ProofStatus::Proven);
  info.viewToRoot = AffineMap::get(1, 0, getAffineDimExpr(0, &context) + 1);
  auto proof = proveExecutionVectorAccess(d, info, fp, 16, AccessKind::Read, "all");
  EXPECT_EQ(proof.status, ProofStatus::Disproven);
  EXPECT_FALSE(proof.counterexample.empty());
}
TEST_F(ExecutionLayoutProofTest, RejectsOverlappingScalarFootprintIntervals) {
  auto d = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2});
  auto [info, fp] = storage(3);
  fp.entries[1].begin = 2;
  fp.entries[1].end = 5;
  auto proof = proveExecutionVectorAccess(d, info, fp, 1, AccessKind::Read, "all");
  EXPECT_EQ(proof.status, ProofStatus::Disproven);
  EXPECT_FALSE(proof.counterexample.empty());
}
} // namespace
