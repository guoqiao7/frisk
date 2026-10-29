#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class SymbolicExecutionLayoutTest : public testing::Test {
protected:
  SymbolicExecutionLayoutTest() { context.loadDialect<FriskDialect>(); }
  DistributedEncodingAttr encoding(ArrayRef<StringRef> names,
      ArrayRef<int64_t> widths, ArrayRef<int64_t> topology,
      ArrayRef<uint64_t> rows, ArrayRef<int64_t> outputWidths,
      int64_t replication = 1, bool checked = false) {
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
    auto emit = [&] { return emitError(UnknownLoc::get(&context)); };
    auto map = checked ? BitLinearLayoutMapAttr::getChecked(emit, &context,
        b.getStrArrayAttr(names), b.getDenseI64ArrayAttr(widths),
        b.getArrayAttr(outputs), b.getDenseI64ArrayAttr(outputWidths), matrix)
        : BitLinearLayoutMapAttr::get(&context, b.getStrArrayAttr(names),
            b.getDenseI64ArrayAttr(widths), b.getArrayAttr(outputs),
            b.getDenseI64ArrayAttr(outputWidths), matrix);
    if (!map) return {};
    if (checked)
      return DistributedEncodingAttr::getChecked(emit, &context, Attribute(map),
          b.getDenseI64ArrayAttr(topology), b.getI64IntegerAttr(replication));
    return DistributedEncodingAttr::get(&context, map,
        b.getDenseI64ArrayAttr(topology), b.getI64IntegerAttr(replication));
  }
  RankedTensorType type(ArrayRef<int64_t> shape = {4}) {
    return RankedTensorType::get(shape, IntegerType::get(&context, 32));
  }
  MLIRContext context;
};

TEST_F(SymbolicExecutionLayoutTest, BoundedMatrixFamilyMatchesEnumeration) {
  // Exhaust all 2x3 binary matrices, including rank-deficient maps. The oracle
  // really enumerates eight hardware points; it does not call symbolic algebra.
  for (uint64_t a = 0; a < 8; ++a) {
    for (uint64_t b = 0; b < 8; ++b) {
      for (int64_t replication : {1, 2, 4}) {
        SCOPED_TRACE(testing::Message() << a << ',' << b << ',' << replication);
        auto d = encoding({"register", "lane"}, {2,1}, {4,2,1,1,1},
                          {a,b}, {2}, replication);
        ExecutionEnumeration points;
        auto oracle = enumerateExecutionLayout(d, type(), points);
        auto first = proveExecutionOwnership(d, type(), "first_owner");
        EXPECT_EQ(first.status, oracle.status);
        auto all = proveExecutionOwnership(d, type(), "all");
        EXPECT_EQ(all.status, ProofStatus::Disproven);
        if (first.status == ProofStatus::Disproven && !first.counterexample.empty()) {
          ASSERT_EQ(first.counterexample.size(), 1u);
          auto coordinate = uint64_t(first.counterexample[0]);
          EXPECT_LT(coordinate, 4u);
          EXPECT_EQ(llvm::count(points.logical, coordinate), 0);
        }
      }
    }
  }
  // All 2x2 matrices additionally exercise the genuinely unique-writer case.
  for (uint64_t a = 0; a < 4; ++a) {
    for (uint64_t b = 0; b < 4; ++b) {
      auto d = encoding({"register"}, {2}, {4,1,1,1,1}, {a,b}, {2});
      ExecutionEnumeration points;
      auto oracle = enumerateExecutionLayout(d, type(), points);
      EXPECT_EQ(proveExecutionOwnership(d, type(), "all").status, oracle.status);
    }
  }
}

TEST_F(SymbolicExecutionLayoutTest, ReorderedAndOmittedCarriersHaveActualWitnesses) {
  for (auto d : {
      encoding({"register", "lane"}, {2,1}, {4,2,1,1,1}, {1,2}, {2}, 2),
      encoding({"lane", "register"}, {1,2}, {4,2,1,1,1}, {2,4}, {2}, 2),
      encoding({"register"}, {2}, {4,2,1,1,1}, {1,2}, {2}, 2)}) {
    EXPECT_EQ(proveExecutionOwnership(d, type(), "first_owner").status, ProofStatus::Proven);
    auto proof = proveExecutionOwnership(d, type(), "all");
    ASSERT_EQ(proof.status, ProofStatus::Disproven);
    ASSERT_EQ(proof.counterexample.size(), 5u);
    EXPECT_EQ(proof.counterexample,
              proveExecutionOwnership(d, type(), "all").counterexample);
    ExecutionEnumeration points;
    ASSERT_EQ(enumerateExecutionLayout(d, type(), points).status, ProofStatus::Proven);
    uint64_t id = 0;
    for (unsigned c = 0; c < 5; ++c) {
      int64_t coordinate = proof.counterexample[c];
      ASSERT_GE(coordinate, 0);
      ASSERT_LT(uint64_t(coordinate), points.topology[4-c]);
      id = id * points.topology[4-c] + coordinate;
    }
    ASSERT_NE(id, 0u);
    EXPECT_EQ(points.logical[id], points.logical[0]);
  }
}

TEST_F(SymbolicExecutionLayoutTest, ZeroBitDomains) {
  for (auto shape : {SmallVector<int64_t>{1}, SmallVector<int64_t>{}}) {
    SmallVector<int64_t> widths(shape.size(), 0);
    auto unique = encoding({}, {}, {1,1,1,1,1}, {}, widths);
    EXPECT_EQ(proveExecutionOwnership(unique, type(shape), "all").status, ProofStatus::Proven);
    auto replicated = encoding({}, {}, {1,2,1,1,1}, {}, widths, 2);
    EXPECT_EQ(proveExecutionOwnership(replicated, type(shape), "first_owner").status, ProofStatus::Proven);
    auto duplicate = proveExecutionOwnership(replicated, type(shape), "all");
    EXPECT_EQ(duplicate.status, ProofStatus::Disproven);
    EXPECT_EQ(duplicate.counterexample, (SmallVector<int64_t>{0,0,0,1,0}));
  }
  // A positive logical domain cannot be covered by an empty hardware domain.
  auto missing = encoding({}, {}, {1,1,1,1,1}, {0,0}, {2});
  auto proof = proveExecutionOwnership(missing, type(), "first_owner");
  EXPECT_EQ(proof.status, ProofStatus::Disproven);
  ASSERT_EQ(proof.counterexample.size(), 1u);
  EXPECT_GT(proof.counterexample[0], 0);
}

TEST_F(SymbolicExecutionLayoutTest, WideOmittedDomainsDoNotOverflowReplication) {
  const int64_t huge = int64_t(1) << 62;
  auto d = encoding({"register"}, {2}, {4,1,huge,huge,1}, {1,2}, {2}, 1);
  EXPECT_EQ(proveExecutionOwnership(d, type(), "first_owner").status, ProofStatus::Disproven);
  auto covered = encoding({"register"}, {2}, {4,1,huge,1,1}, {1,2}, {2}, huge);
  EXPECT_EQ(proveExecutionOwnership(covered, type(), "first_owner").status, ProofStatus::Proven);
  auto tooWide = encoding({}, {}, {huge,1,huge,huge,huge}, {}, {0}, 1);
  // 248 hardware bits remain inside the explicit 256-bit algebra bound.
  EXPECT_EQ(proveExecutionOwnership(tooWide, type({1}), "first_owner").status, ProofStatus::Disproven);
  // The lane limit makes the maximum legal five-carrier domain 253 bits.
  auto over = encoding({}, {}, {huge,32,huge,huge,huge}, {}, {0}, 1);
  EXPECT_EQ(proveExecutionOwnership(over, type({1}), "first_owner").status, ProofStatus::Disproven);
}

TEST_F(SymbolicExecutionLayoutTest, CanonicalRepresentationAndCarrierBlocks) {
  auto d = encoding({"lane", "register"}, {1,2}, {4,2,2,1,1}, {3,4}, {2}, 4);
  LinearExecutionLayout map;
  ASSERT_EQ(analyzeLinearExecutionLayout(d, type(), map).status, ProofStatus::Proven);
  EXPECT_EQ(map.hardwareBits, 4u);
  EXPECT_EQ(map.logicalBits, 2u);
  EXPECT_EQ(map.hardwareOffsets, (std::array<unsigned,5>{0,2,3,4,4}));
  ASSERT_TRUE(succeeded(map.getMatrix()));
  auto reg = map.getCarrierMatrix(0), lane = map.getCarrierMatrix(1), warp = map.getCarrierMatrix(2);
  ASSERT_TRUE(succeeded(reg));
  ASSERT_TRUE(succeeded(lane));
  ASSERT_TRUE(succeeded(warp));
  EXPECT_EQ(reg->apply(APInt(2, 3)).getZExtValue(), 3u);
  EXPECT_EQ(lane->apply(APInt(1, 1)).getZExtValue(), 1u);
  EXPECT_EQ(warp->rank(), 0u);
  EXPECT_TRUE(failed(map.getCarrierMatrix(3)));
  EXPECT_TRUE(failed(map.getCarrierMatrix(5)));
  EXPECT_EQ(map.apply(APInt(4, 5)).getZExtValue(), 0u);
  EXPECT_EQ(map.hardwareCoordinate(APInt(4, 13)), (SmallVector<int64_t>{0,0,1,1,1}));
  EXPECT_EQ(map.logicalCoordinate(APInt(2, 3)), (SmallVector<int64_t>{3}));

  auto empty = encoding({}, {}, {1,1,1,1,1}, {}, {0});
  ASSERT_EQ(analyzeLinearExecutionLayout(empty, type({1}), map).status, ProofStatus::Proven);
  EXPECT_TRUE(failed(map.getMatrix()));
  EXPECT_EQ(map.apply(APInt(1, 0)), APInt(1, 0));

  // A structurally valid representation is deliberately not a coverage proof.
  auto uncovered = encoding({"register"}, {2}, {4,1,1,1,1}, {0,0}, {2});
  EXPECT_EQ(analyzeLinearExecutionLayout(uncovered, type(), map).status, ProofStatus::Proven);
  EXPECT_EQ(proveExecutionOwnership(uncovered, type(), "all").status, ProofStatus::Disproven);
}

TEST_F(SymbolicExecutionLayoutTest, LogicalBitBudgetIsExplicit) {
  SmallVector<int64_t> shape(17, 65536), widths(17, 16);
  SmallVector<uint64_t> rows(272, 0);
  auto d = encoding({}, {}, {1,1,1,1,1}, rows, widths);
  auto proof = proveExecutionOwnership(d, type(shape), "first_owner");
  EXPECT_EQ(proof.status, ProofStatus::Unknown);
  EXPECT_NE(proof.reason.find("256"), std::string::npos);
}

TEST_F(SymbolicExecutionLayoutTest, MalformedAttributesAreRejected) {
  auto valid = encoding({"register"}, {2}, {4,1,1,1,1}, {1,2}, {2});
  EXPECT_EQ(proveExecutionOwnership({}, type(), "all").status, ProofStatus::Disproven);
  EXPECT_EQ(proveExecutionOwnership(valid, {}, "all").status, ProofStatus::Disproven);
  EXPECT_EQ(proveExecutionOwnership(valid, type({ShapedType::kDynamic}), "all").status, ProofStatus::Disproven);
  EXPECT_EQ(proveExecutionOwnership(valid, type({3}), "all").status, ProofStatus::Disproven);
  // These attrs satisfy construction invariants but fail proof requirements.
  for (auto d : {
      encoding({"register"}, {2}, {4,3,1,1,1}, {1,2}, {2}),
      encoding({"register"}, {2}, {8,1,1,1,1}, {1,2}, {2}),
      encoding({"register"}, {2}, {4,1,1,1,1}, {1,2}, {2}, 2)}) {
    EXPECT_EQ(proveExecutionOwnership(d, type(), "first_owner").status, ProofStatus::Disproven);
  }

  // Structurally invalid attributes must be rejected at construction, not
  // manufactured with unchecked get (which asserts in assertion-enabled MLIR).
  unsigned diagnostics = 0;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &) {
    ++diagnostics;
    return success();
  });
  for (auto d : {
      encoding({"register"}, {2}, {4,1,1,1}, {1,2}, {2}, 1, true),
      encoding({"register"}, {2}, {4,64,1,1,1}, {1,2}, {2}, 1, true),
      encoding({"register"}, {2}, {4,0,1,1,1}, {1,2}, {2}, 1, true),
      encoding({"alien"}, {2}, {4,1,1,1,1}, {1,2}, {2}, 1, true),
      encoding({"register", "register"}, {1,1}, {2,1,1,1,1}, {1,2}, {2}, 1, true),
      encoding({"register"}, {2}, {4,1,1,1,1}, {1}, {2}, 1, true),
      encoding({"register"}, {2}, {4,1,1,1,1}, {1,2}, {2}, 0, true)})
    EXPECT_FALSE(d);

  // The parsers reject missing mandatory fields before attribute verification,
  // and reject an overflowing i64 replication before extracting its value.
  for (StringRef text : {
      R"mlir(#frisk.bit_linear<input_bits = [2], outputs = ["x"], output_bits = [2], matrix = dense<0> : tensor<2x2xi1>>)mlir",
      R"mlir(#frisk.bit_linear<inputs = ["register"], input_bits = [2], outputs = ["x"], output_bits = [2]>)mlir",
      R"mlir(#frisk.distributed<map = #frisk.bit_linear<inputs = ["register"], input_bits = [2], outputs = ["x"], output_bits = [2], matrix = dense<[[1,0],[0,1]]> : tensor<2x2xi1>>, topology = [4,1,1,1,1], replication = 1267650600228229401496703205376>)mlir"})
    EXPECT_FALSE(parseAttribute(text, &context));

  Builder b(&context);
  auto map = cast<BitLinearLayoutMapAttr>(valid.getMap());
  for (auto replication : {IntegerAttr(), b.getI64IntegerAttr(-1)}) {
    auto d = DistributedEncodingAttr::getChecked(
        [&] { return emitError(UnknownLoc::get(&context)); },
        &context, Attribute(map), valid.getTopology(), replication);
    EXPECT_FALSE(d);
  }
  EXPECT_GE(diagnostics, 12u);
}
} // namespace
