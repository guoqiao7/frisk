#include "Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <limits>
#include <string>
#include <vector>

namespace mlir::frisk::test {
namespace {
class LayoutBoundaryFrequencyTest : public testing::Test {
protected:
  MLIRContext context;
  static constexpr auto Static = LayoutBoundaryFrequencyMode::StaticTripCount;
  static constexpr auto Legacy = LayoutBoundaryFrequencyMode::Legacy;
  LayoutBoundaryFrequencyTest() {
    context.loadDialect<arith::ArithDialect, func::FuncDialect, scf::SCFDialect>();
  }
  struct Loop {
    std::string lower, upper, step = "1", type = "index";
  };
  OwningOpRef<ModuleOp> make(const std::vector<Loop> &loops) {
    std::string text = "module { func.func @test(%dynamic: index) {\n";
    unsigned i = 0;
    for (const auto &loop : loops) {
      auto n = std::to_string(i++);
      text += "%lb" + n + " = arith.constant " + loop.lower + " : " + loop.type + "\n";
      if (loop.upper != "dynamic")
        text += "%ub" + n + " = arith.constant " + loop.upper + " : " + loop.type + "\n";
      text += "%step" + n + " = arith.constant " + loop.step + " : " + loop.type + "\n";
      text += "scf.for %iv" + n + " = %lb" + n + " to %" +
          (loop.upper == "dynamic" ? "dynamic" : "ub" + n) + " step %step" + n;
      if (loop.type != "index") text += " : " + loop.type;
      text += " {\n";
    }
    text += "%marker = arith.constant 0 : i32\n";
    for (unsigned j = 0; j < loops.size(); ++j) text += "}\n";
    text += "return } }";
    return parseSourceString<ModuleOp>(text, &context);
  }
  Operation *marker(ModuleOp module) {
    Operation *result = nullptr;
    module.walk([&](arith::ConstantOp op) {
      if (op.getType().isInteger(32)) result = op;
    });
    return result;
  }
  void expect(const std::vector<Loop> &loops, uint64_t factor,
              bool proxy = false, bool saturated = false,
              LayoutBoundaryFrequencyMode mode = Static) {
    auto module = make(loops);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    auto point = marker(*module);
    ASSERT_NE(point, nullptr);
    auto frequency = getLayoutBoundaryFrequency(point, mode);
    EXPECT_EQ(frequency.factor, factor);
    EXPECT_EQ(frequency.usesProxy, proxy);
    EXPECT_EQ(frequency.saturated, saturated);
  }
};
TEST_F(LayoutBoundaryFrequencyTest, ConstantZeroOneAndHundred) {
  expect({{"0", "0"}}, 0);
  expect({{"0", "1"}}, 1);
  expect({{"0", "100"}}, 100);
}
TEST_F(LayoutBoundaryFrequencyTest, NegativeLowerAndRoundedStep) {
  expect({{"-10", "10", "3"}}, 7);
  expect({{"10", "-10", "3"}}, 0);
}
TEST_F(LayoutBoundaryFrequencyTest, NestedExactAndDynamic) {
  expect({{"0", "3"}, {"0", "5"}}, 15);
  expect({{"0", "3"}, {"0", "dynamic"}}, 6, true);
}
TEST_F(LayoutBoundaryFrequencyTest, ZeroDominatesUnknownAndOverflow) {
  expect({{"0", "0"}, {"0", "dynamic"}}, 0);
  expect({{"0", "dynamic"}, {"0", "0"}}, 0);
  Loop huge{"-9223372036854775808", "9223372036854775807"};
  expect({{"0", "0"}, huge, huge}, 0);
  expect({huge, huge, {"0", "0"}}, 0);
}
TEST_F(LayoutBoundaryFrequencyTest, ExactUint64MaxAndOverflow) {
  Loop huge{"-9223372036854775808", "9223372036854775807"};
  expect({huge}, std::numeric_limits<uint64_t>::max());
  expect({huge, {"0", "2"}}, std::numeric_limits<uint64_t>::max(), false, true);
  expect({huge, huge, huge}, std::numeric_limits<uint64_t>::max(), false, true);
  expect({huge, {"0", "dynamic"}}, std::numeric_limits<uint64_t>::max(), true, true);
}
TEST_F(LayoutBoundaryFrequencyTest, ExtremeSignedBoundsAndStep) {
  expect({{"-9223372036854775808", "9223372036854775807", "9223372036854775807"}}, 3);
  expect({{"-128", "127", "3", "i8"}}, 85);
  expect({{"-10", "10", "3", "i64"}}, 7);
}
TEST_F(LayoutBoundaryFrequencyTest, WideConstantsNeverTruncate) {
  expect({{"0", "18446744073709551616", "1", "i128"}},
         std::numeric_limits<uint64_t>::max(), false, true);
  expect({{"18446744073709551616", "18446744073709551619", "1", "i128"}}, 3);
}
TEST_F(LayoutBoundaryFrequencyTest, DynamicBounds) {
  expect({{"0", "dynamic"}}, 2, true);
}
TEST_F(LayoutBoundaryFrequencyTest, WhileUsesProxy) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { func.func @test() {
      scf.while : () -> () {
        %marker = arith.constant 0 : i32
        %false = arith.constant false
        scf.condition(%false)
      } do { scf.yield }
      return
    } }
  )mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto result = getLayoutBoundaryFrequency(marker(*module), Static);
  EXPECT_EQ(result.factor, 2u);
  EXPECT_TRUE(result.usesProxy);
  EXPECT_FALSE(result.saturated);
}
TEST_F(LayoutBoundaryFrequencyTest, UnsignedMarkerIsConservative) {
  auto module = make({{"0", "100"}});
  ASSERT_TRUE(module);
  module->walk([&](scf::ForOp op) { op->setAttr("unsignedCmp", UnitAttr::get(&context)); });
  ASSERT_TRUE(succeeded(verify(*module)));
  auto result = getLayoutBoundaryFrequency(marker(*module), Static);
  EXPECT_EQ(result.factor, 2u);
  EXPECT_TRUE(result.usesProxy);
  EXPECT_FALSE(result.saturated);
}
TEST_F(LayoutBoundaryFrequencyTest, NullAndNoAncestors) {
  for (auto mode : {Static, Legacy}) {
    auto result = getLayoutBoundaryFrequency(nullptr, mode);
    EXPECT_EQ(result.factor, 1u);
    EXPECT_FALSE(result.usesProxy);
    EXPECT_FALSE(result.saturated);
    expect({}, 1, false, false, mode);
  }
}
TEST_F(LayoutBoundaryFrequencyTest, InsertionPointItselfIsExcluded) {
  auto module = make({{"0", "3"}, {"0", "5"}});
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto inner = marker(*module)->getParentOp();
  auto result = getLayoutBoundaryFrequency(inner, Static);
  EXPECT_EQ(result.factor, 3u);
  EXPECT_FALSE(result.usesProxy);
  EXPECT_FALSE(result.saturated);
  auto outer = getLayoutBoundaryFrequency(inner->getParentOp(), Static);
  EXPECT_EQ(outer.factor, 1u);
  EXPECT_FALSE(outer.usesProxy);
}
TEST_F(LayoutBoundaryFrequencyTest, LegacyIgnoresKnownTripCounts) {
  expect({{"0", "0"}}, 2, true, false, Legacy);
  expect({{"0", "100"}, {"0", "5"}}, 4, true, false, Legacy);
  std::vector<Loop> deep(64, {"0", "0"});
  expect(deep, std::numeric_limits<uint64_t>::max(), true, true, Legacy);
}
} // namespace
TEST_F(LayoutBoundaryFrequencyTest, ParallelIsProxyOnlyInStaticMode) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @parallel() {
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      scf.parallel (%i) = (%zero) to (%one) step (%one) {
        %marker = arith.constant 0 : i32
        scf.reduce
      }
      return
    }
  )mlir", &context);
  ASSERT_TRUE(module);
  auto legacy = getLayoutBoundaryFrequency(marker(*module), Legacy);
  EXPECT_EQ(legacy.factor, 1u); EXPECT_FALSE(legacy.usesProxy);
  auto frequency = getLayoutBoundaryFrequency(marker(*module), Static);
  EXPECT_EQ(frequency.factor, 2u); EXPECT_TRUE(frequency.usesProxy);
}
} // namespace mlir::frisk::test
