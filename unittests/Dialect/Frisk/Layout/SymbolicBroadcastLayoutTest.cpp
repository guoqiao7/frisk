#include "Dialect/Frisk/Analysis/BroadcastLayoutConstraints.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class SymbolicBroadcastLayoutTest : public testing::Test {
protected:
  SymbolicBroadcastLayoutTest() { context.loadDialect<FriskDialect>(); }
  DistributedEncodingAttr encoding(ArrayRef<StringRef> names,
      ArrayRef<int64_t> widths, ArrayRef<int64_t> topology,
      ArrayRef<uint64_t> rows, ArrayRef<int64_t> outputWidths,
      int64_t replication = 1) {
    Builder b(&context);
    unsigned columns = 0;
    for (int64_t width : widths) columns += width;
    SmallVector<APInt> bits;
    for (uint64_t row : rows)
      for (unsigned c = 0; c < columns; ++c)
        bits.emplace_back(1, (row >> c) & 1);
    SmallVector<Attribute> outputs;
    for (unsigned i = 0; i < outputWidths.size(); ++i)
      outputs.push_back(b.getStringAttr("x" + std::to_string(i)));
    auto matrix = DenseIntElementsAttr::get(RankedTensorType::get(
        {int64_t(rows.size()), int64_t(columns)}, b.getI1Type()), bits);
    auto map = BitLinearLayoutMapAttr::get(&context, b.getStrArrayAttr(names),
        b.getDenseI64ArrayAttr(widths), b.getArrayAttr(outputs),
        b.getDenseI64ArrayAttr(outputWidths), matrix);
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr(topology), b.getI64IntegerAttr(replication));
  }
  RankedTensorType type(ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, IntegerType::get(&context, 32));
  }
  // Independent actual-coordinate oracle: enumerate destination holders, erase
  // logical axes, and search every source register in that very same thread.
  bool available(const ExecutionEnumeration &src, const ExecutionEnumeration &dst,
      uint64_t holder, ArrayRef<int64_t> axes) {
    auto coordinate = dst.coordinate(dst.logical[holder]);
    SmallVector<int64_t> wanted;
    for (unsigned axis = 0; axis < coordinate.size(); ++axis)
      if (!llvm::is_contained(axes, axis)) wanted.push_back(coordinate[axis]);
    uint64_t thread = holder / dst.topology[0];
    for (uint64_t reg = 0; reg < src.topology[0]; ++reg)
      if (src.coordinate(src.logical[thread * src.topology[0] + reg]) == wanted)
        return true;
    return false;
  }
  void check(DistributedEncodingAttr src, DistributedEncodingAttr dst,
      RankedTensorType input, RankedTensorType output, ArrayRef<int64_t> axes) {
    ExecutionEnumeration a, b;
    auto source = enumerateExecutionLayout(src, input, a);
    auto destination = enumerateExecutionLayout(dst, output, b);
    auto proof = proveBroadcastLayout(src, dst, input, output, axes);
    if (source.status != ProofStatus::Proven || destination.status != ProofStatus::Proven) {
      EXPECT_EQ(proof.status, ProofStatus::Disproven);
      return;
    }
    bool legal = true, actualWitness = false;
    for (uint64_t h = 0; h < b.logical.size(); ++h) {
      if (available(a, b, h, axes)) continue;
      legal = false;
      actualWitness |= b.coordinate(b.logical[h]) == proof.counterexample;
    }
    EXPECT_EQ(proof.status, legal ? ProofStatus::Proven : ProofStatus::Disproven);
    if (!legal) {
      EXPECT_TRUE(actualWitness) << proof.reason;
      EXPECT_EQ(proof.counterexample,
          proveBroadcastLayout(src, dst, input, output, axes).counterexample);
    }
  }
  MLIRContext context;
};

TEST_F(SymbolicBroadcastLayoutTest, BeyondEnumerationBudget) {
  SmallVector<uint64_t> rows;
  for (unsigned i = 0; i < 12; ++i) rows.push_back(uint64_t(1) << i);
  auto src = encoding({"register", "lane"}, {12,5}, {4096,32,1,1,1},
                      rows, {12}, 32);
  rows.push_back(uint64_t(1) << 12);
  auto dst = encoding({"register", "lane"}, {12,5}, {4096,32,1,1,1},
                      rows, {12,1}, 16);
  auto input = type({4096}), output = type({4096,2});
  ExecutionEnumeration points;
  EXPECT_EQ(enumerateExecutionLayout(src, input, points).status, ProofStatus::Unknown);
  EXPECT_EQ(enumerateExecutionLayout(dst, output, points).status, ProofStatus::Unknown);
  auto proof = proveBroadcastLayout(src, dst, input, output, {1});
  EXPECT_EQ(proof.status, ProofStatus::Proven) << proof.reason;
  // The entry coverage check is symbolic, but actual reduction projection still
  // needs its bounded table. Do not silently broaden that algorithm's domain.
  EXPECT_TRUE(failed(projectBroadcastInputEncoding(dst, input, output, {1})));
  EXPECT_TRUE(succeeded(projectBroadcastInputEncoding(dst, output, output, {})));
}

TEST_F(SymbolicBroadcastLayoutTest, BeyondBudgetSameThreadUnavailable) {
  SmallVector<uint64_t> rows{uint64_t(1) << 12};
  for (unsigned i = 0; i < 11; ++i) rows.push_back(uint64_t(1) << i);
  auto src = encoding({"register", "lane"}, {12,5}, {4096,32,1,1,1},
                      rows, {12}, 32);
  rows.clear();
  for (unsigned i = 0; i < 13; ++i) rows.push_back(uint64_t(1) << i);
  auto dst = encoding({"register", "lane"}, {12,5}, {4096,32,1,1,1},
                      rows, {12,1}, 16);
  auto proof = proveBroadcastLayout(src, dst, type({4096}), type({4096,2}), {1});
  EXPECT_EQ(proof.status, ProofStatus::Disproven) << proof.reason;
  ASSERT_EQ(proof.counterexample.size(), 2u);
  EXPECT_EQ(proof.counterexample, (SmallVector<int64_t>{1,0}));
  // Destination r=1,lane=0 actually holds (1,0). Every source register at that
  // lane has a zero low bit, so no register can supply the requested value.
  for (unsigned reg = 0; reg < 4096; ++reg)
    EXPECT_NE((reg & 2047) << 1, unsigned(proof.counterexample[0]));
}

TEST_F(SymbolicBroadcastLayoutTest, ExhaustiveSmallMatricesMatchActualRegisters) {
  // Every 2x3 source matrix and 3x3 destination matrix. The topology includes
  // both lane and warp bits, and the source has just one register bit, so this
  // includes real lane/warp incompatibilities as well as malformed ownership.
  for (uint64_t source = 0; source < 64; ++source) {
    auto src = encoding({"register", "lane", "warp"}, {1,1,1}, {2,2,2,1,1},
        {source & 7, source >> 3}, {2}, 2);
    for (uint64_t destination = 0; destination < 512; ++destination) {
      SCOPED_TRACE(testing::Message() << source << ',' << destination);
      auto dst = encoding({"register", "lane", "warp"}, {1,1,1}, {2,2,2,1,1},
          {destination & 7, (destination >> 3) & 7, destination >> 6}, {2,1});
      check(src, dst, type({4}), type({4,2}), {1});
    }
  }
}

TEST_F(SymbolicBroadcastLayoutTest, RegisterBasesReplicationAndReorderedCarriers) {
  // Source register columns contain a nonzero XOR-dependent column; the warp
  // carrier is omitted but still contributes replication. Destination register
  // columns are reordered/XORed, and the explicit carrier order is different.
  auto src = encoding({"lane", "register"}, {1,3}, {8,2,2,1,1}, {10,12}, {2}, 8);
  auto dst = encoding({"warp", "lane", "register"}, {1,1,2}, {4,2,2,1,1},
                      {12,4,2}, {2,1}, 2);
  check(src, dst, type({4}), type({4,2}), {1});
  EXPECT_EQ(proveBroadcastLayout(src, dst, type({4}), type({4,2}), {1}).status,
            ProofStatus::Proven);
  // A zero register column adds replication without changing availability.
  auto copied = encoding({"register"}, {3}, {8,2,2,1,1}, {1,2}, {2}, 8);
  check(copied, dst, type({4}), type({4,2}), {1});
  // Destination has no registers at all, so thread offsets differ in the other
  // direction too. Source registers can still supply every requested value.
  auto registerSrc = encoding({"register"}, {1}, {2,2,2,1,1}, {1}, {1}, 4);
  auto threadDst = encoding({"lane", "warp"}, {1,1}, {1,2,2,1,1}, {1,2}, {1,1});
  check(registerSrc, threadDst, type({2}), type({2,2}), {1});
  EXPECT_EQ(proveBroadcastLayout(registerSrc, threadDst, type({2}), type({2,2}), {1}).status,
            ProofStatus::Proven);
}

TEST_F(SymbolicBroadcastLayoutTest, ExactEnumerationBoundaryAndWideOmittedThreads) {
  SmallVector<uint64_t> rows;
  for (unsigned bit = 0; bit < 11; ++bit) rows.push_back(uint64_t(1) << bit);
  auto src = encoding({"register", "lane"}, {11,5}, {2048,32,1,1,1},
                      rows, {11}, 32);
  rows.push_back(uint64_t(1) << 11);
  auto dst = encoding({"register", "lane"}, {11,5}, {2048,32,1,1,1},
                      rows, {11,1}, 16);
  ExecutionEnumeration a, b;
  ASSERT_EQ(enumerateExecutionLayout(src, type({2048}), a).status, ProofStatus::Proven);
  ASSERT_EQ(enumerateExecutionLayout(dst, type({2048,2}), b).status, ProofStatus::Proven);
  EXPECT_EQ(a.logical.size(), 65536u);
  EXPECT_EQ(b.logical.size(), 65536u);
  EXPECT_EQ(proveBroadcastLayout(src, dst, type({2048}), type({2048,2}), {1}).status,
            ProofStatus::Proven);
  // A nontrivial omitted warp carrier contributes forty real hardware bits;
  // no hardware product, holder vector, or narrow shift is needed.
  const int64_t huge = int64_t(1) << 40;
  auto wideSrc = encoding({"register"}, {2}, {4,2,huge,1,1}, {1,2}, {2}, huge * 2);
  auto wideDst = encoding({"register", "lane"}, {2,1}, {4,2,huge,1,1},
                          {1,2,4}, {2,1}, huge);
  EXPECT_EQ(proveBroadcastLayout(wideSrc, wideDst, type({4}), type({4,2}), {1}).status,
            ProofStatus::Proven);
}

TEST_F(SymbolicBroadcastLayoutTest, AxisOrderAndThreadColumnFailuresHaveActualWitnesses) {
  auto src = encoding({"lane", "warp"}, {1,1}, {1,2,2,1,1}, {1,2}, {1,1});
  // Source bits live in the threads, while register bits supply inserted axes.
  auto dst = encoding({"register", "lane", "warp"}, {2,1,1}, {4,2,2,1,1},
                      {1,4,2,8}, {1,1,1,1});
  check(src, dst, type({2,2}), type({2,2,2,2}), {0,2});
  EXPECT_EQ(proveBroadcastLayout(src, dst, type({2,2}), type({2,2,2,2}), {0,2}).status,
            ProofStatus::Proven);
  check(src, dst, type({2,2}), type({2,2,2,2}), {1,3});
  EXPECT_EQ(proveBroadcastLayout(src, dst, type({2,2}), type({2,2,2,2}), {1,3}).status,
            ProofStatus::Disproven);
  // No source register bits: only a thread difference column can fail here.
  auto swapped = encoding({"register", "lane", "warp"}, {1,1,1}, {2,2,2,1,1},
                          {4,2,1}, {1,1,1});
  check(src, swapped, type({2,2}), type({2,2,2}), {2});
  auto proof = proveBroadcastLayout(src, swapped, type({2,2}), type({2,2,2}), {2});
  EXPECT_EQ(proof.status, ProofStatus::Disproven);
  EXPECT_EQ(proof.counterexample, (SmallVector<int64_t>{0,1,0}));
}

TEST_F(SymbolicBroadcastLayoutTest, OwnershipAndThreadGuardsRemainRequired) {
  auto input = type({2}), output = type({2,2});
  auto src = encoding({"register"}, {1}, {2,2,1,1,1}, {1}, {1}, 2);
  auto dst = encoding({"register", "lane"}, {1,1}, {2,2,1,1,1}, {1,2}, {1,1});
  for (auto broken : {
      encoding({"register"}, {1}, {2,2,1,1,1}, {0}, {1}, 2),
      encoding({"register"}, {1}, {2,2,1,1,1}, {1}, {1}, 1),
      encoding({"register"}, {1}, {2,4,1,1,1}, {1}, {1}, 4),
      encoding({"register"}, {1}, {2,2,2,1,1}, {1}, {1}, 4),
      encoding({"register"}, {1}, {2,2,1,2,1}, {1}, {1}, 4),
      encoding({"register"}, {1}, {2,2,1,1,2}, {1}, {1}, 4)})
    EXPECT_EQ(proveBroadcastLayout(broken, dst, input, output, {1}).status,
              ProofStatus::Disproven);
  auto uncovered = encoding({"register", "lane"}, {1,1}, {2,2,1,1,1}, {1,1}, {1,1});
  EXPECT_EQ(proveBroadcastLayout(src, uncovered, input, output, {1}).status,
            ProofStatus::Disproven);
  EXPECT_TRUE(failed(projectBroadcastInputEncoding(uncovered, input, output, {1})));
  // Matching CTA domains are still illegal when the CTA count is nonunit.
  auto ctaSrc = encoding({"register"}, {1}, {2,1,1,1,2}, {1}, {1}, 2);
  auto ctaDst = encoding({"register", "cta"}, {2,1}, {4,1,1,1,2}, {1,2}, {1,1}, 2);
  EXPECT_EQ(proveBroadcastLayout(ctaSrc, ctaDst, input, output, {1}).status,
            ProofStatus::Disproven);
}

TEST_F(SymbolicBroadcastLayoutTest, MalformedAndExplicitSymbolicBudget) {
  auto input = type({2}), output = type({2,2});
  auto src = encoding({"register"}, {1}, {2,1,1,1,1}, {1}, {1});
  auto dst = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2}, {1,1});
  EXPECT_EQ(proveBroadcastLayout({}, dst, input, output, {1}).status, ProofStatus::Disproven);
  EXPECT_EQ(proveBroadcastLayout(src, {}, input, output, {1}).status, ProofStatus::Disproven);
  EXPECT_TRUE(failed(projectBroadcastInputEncoding({}, input, output, {1})));
  // Valid attributes can still have proof-incompatible topology or logical
  // dimensions; these failures must reach the proof, including projection.
  auto badTopology = encoding({"register"}, {2}, {8,1,1,1,1}, {1,2}, {1,1});
  EXPECT_EQ(proveBroadcastLayout(src, badTopology, input, output, {1}).status,
            ProofStatus::Disproven);
  EXPECT_TRUE(failed(projectBroadcastInputEncoding(badTopology, input, output, {1})));
  auto badDimensions = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2}, {2});
  EXPECT_EQ(proveBroadcastLayout(src, badDimensions, input, output, {1}).status,
            ProofStatus::Disproven);
  EXPECT_TRUE(failed(projectBroadcastInputEncoding(badDimensions, input, output, {1})));

  // Structurally invalid attributes are rejected by the public parser/checked
  // builder. Unchecked get would assert before either Broadcast proof is called.
  unsigned diagnostics = 0;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &) {
    ++diagnostics;
    return success();
  });
  EXPECT_FALSE(parseAttribute(R"mlir(#frisk.bit_linear<inputs = ["register"],
      input_bits = [2], outputs = ["x", "y"], output_bits = [1,1]>)mlir", &context));
  Builder b(&context);
  auto map = cast<BitLinearLayoutMapAttr>(dst.getMap());
  EXPECT_FALSE(DistributedEncodingAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context, Attribute(map),
      b.getDenseI64ArrayAttr({4,1,1,1}), dst.getReplication()));
  auto wrongMatrix = DenseIntElementsAttr::get(
      RankedTensorType::get({1,2}, b.getI1Type()), ArrayRef<APInt>{APInt(1,1), APInt(1,0)});
  EXPECT_FALSE(BitLinearLayoutMapAttr::getChecked(
      [&] { return emitError(UnknownLoc::get(&context)); }, &context,
      map.getInputNames(), map.getInputBitWidths(), map.getOutputNames(),
      map.getOutputBitWidths(), wrongMatrix));
  EXPECT_GE(diagnostics, 3u);
  SmallVector<int64_t> shape(17, 65536), widths(17, 16);
  SmallVector<uint64_t> rows(272, 0);
  auto wideSrc = encoding({}, {}, {1,1,1,1,1}, rows, widths);
  auto wideShape = shape;
  wideShape.push_back(2);
  auto wideDst = encoding({}, {}, {1,1,1,1,1}, {}, {});
  auto proof = proveBroadcastLayout(wideSrc, wideDst, type(shape), type(wideShape), {17});
  EXPECT_EQ(proof.status, ProofStatus::Unknown);
  EXPECT_NE(proof.reason.find("256"), std::string::npos);
  // Exercise the output's bit budget independently of the source budget.
  wideShape = {2};
  llvm::append_range(wideShape, shape);
  widths.insert(widths.begin(), 1);
  rows.push_back(0);
  wideDst = encoding({}, {}, {1,1,1,1,1}, rows, widths);
  SmallVector<int64_t> axes;
  for (int64_t axis = 1; axis <= 17; ++axis) axes.push_back(axis);
  proof = proveBroadcastLayout(src, wideDst, input, type(wideShape), axes);
  EXPECT_EQ(proof.status, ProofStatus::Unknown);
  EXPECT_NE(proof.reason.find("256"), std::string::npos);
}
} // namespace

namespace {
TEST_F(SymbolicBroadcastLayoutTest, UnitAxesAndZeroLogicalBits) {
  EXPECT_TRUE(validBroadcastTypes(type({1}), type({1,8}), {1}));
  auto scalar = encoding({}, {}, {1,1,1,1,1}, {}, {0});
  auto vector = encoding({"register"}, {3}, {8,1,1,1,1}, {1,2,4}, {0,3});
  check(scalar, vector, type({1}), type({1,8}), {1});
  auto allUnit = encoding({}, {}, {1,1,1,1,1}, {}, {0,0});
  check(scalar, allUnit, type({1}), type({1,1}), {1});
  auto replicated = encoding({"lane"}, {1}, {1,2,1,1,1}, {}, {0}, 2);
  auto replicatedOut = encoding({"lane"}, {1}, {1,2,1,1,1}, {}, {0,0}, 2);
  check(replicated, replicatedOut, type({1}), type({1,1}), {0});
  EXPECT_EQ(proveExecutionOwnership(replicated,type({1}),"all").status, ProofStatus::Disproven);
  auto badReplication = encoding({"lane"}, {1}, {1,2,1,1,1}, {}, {0}, 1);
  EXPECT_EQ(proveBroadcastLayout(badReplication,replicatedOut,type({1}),type({1,1}),{0}).status,ProofStatus::Disproven);
  EXPECT_EQ(proveBroadcastLayout(scalar,replicatedOut,type({1}),type({1,1}),{0}).status,ProofStatus::Disproven);
  auto projected = projectBroadcastInputEncoding(vector,type({1}),type({1,8}),{1});
  ASSERT_TRUE(succeeded(projected));
  EXPECT_EQ(projected->getTopology()[0],1);
  EXPECT_EQ(proveExecutionOwnership(*projected,type({1}),"all").status,ProofStatus::Proven);
  // Two inserted unit axes surround a retained nonunit axis.
  auto src = encoding({"lane"},{3},{1,8,1,1,1},{1,2,4},{3});
  auto dst = encoding({"lane"},{3},{1,8,1,1,1},{1,2,4},{0,3,0});
  check(src,dst,type({8}),type({1,8,1}),{0,2});
  auto wrong = encoding({"lane"},{3},{1,8,1,1,1},{2,1,4},{0,3,0});
  check(src,wrong,type({8}),type({1,8,1}),{0,2});
  for (auto shape : {SmallVector<int64_t>{0}, {3}, {ShapedType::kDynamic}, {}})
    EXPECT_FALSE(validBroadcastTypes(type(shape),type({1,1}),{1}));
}
} // namespace
