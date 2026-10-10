#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"
#include <limits>

using namespace mlir;
using namespace mlir::frisk;
namespace {
class LayoutAffineEvaluationTest : public testing::Test {
protected:
  MLIRContext context;
  Builder builder{&context};
  LayoutAffineEvaluationTest() { context.loadDialect<FriskDialect>(); }
  AffineLayoutMapAttr map(AffineExpr expression, int64_t extent = 16) {
    return AffineLayoutMapAttr::get(&context, builder.getStrArrayAttr({"x"}),
        builder.getDenseI64ArrayAttr({extent}), builder.getStrArrayAttr({"y"}),
        builder.getDenseI64ArrayAttr({std::numeric_limits<int64_t>::max()}),
        AffineMapAttr::get(AffineMap::get(1, 0, expression)));
  }
};
TEST_F(LayoutAffineEvaluationTest, FiniteCoordinatesMatchIndependentIntegerArithmetic) {
  auto dim = getAffineDimExpr(0, &context);
  for (int64_t scale : {1, 2, 4, 7})
    for (int64_t bias : {-12, 0, 9}) {
      auto affine = map(dim * scale + bias);
      for (int64_t x = 0; x < 16; ++x) {
        auto result = evaluateLayoutMap(affine, {x});
        ASSERT_TRUE(succeeded(result));
        EXPECT_EQ((*result)[0], x * scale + bias);
      }
    }
}
TEST_F(LayoutAffineEvaluationTest, DivisionAndRemainderMatchIndependentFiniteOracle) {
  auto dim = getAffineDimExpr(0, &context);
  for (int64_t divisor : {2, 3, 5})
    for (int64_t x = -16; x < 16; ++x) {
      int64_t numerator = x * 3 - 17;
      // Enumerate the mathematical floor quotient independently of LLVM's
      // signed division helpers; these tiny bounded inputs cannot overflow.
      int64_t floor = -128;
      while ((floor + 1) * divisor <= numerator) ++floor;
      int64_t ceil = floor + (floor * divisor != numerator);
      for (auto pair : {std::make_pair((dim * 3 - 17).floorDiv(divisor), floor),
                        std::make_pair((dim * 3 - 17).ceilDiv(divisor), ceil),
                        std::make_pair((dim * 3 - 17) % divisor, numerator - floor * divisor)}) {
        auto result = evaluateLayoutMap(map(pair.first), {x});
        ASSERT_TRUE(succeeded(result));
        EXPECT_EQ((*result)[0], pair.second);
      }
    }
}
TEST_F(LayoutAffineEvaluationTest, UnrepresentableIntermediateFailsIncludingBelowDivision) {
  auto dim = getAffineDimExpr(0, &context);
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  for (AffineExpr expression : {dim * 3, dim + maximum, (dim * 3).floorDiv(2),
                               (dim * 3) % 5}) {
    auto affine = map(expression, maximum);
    EXPECT_TRUE(failed(evaluateLayoutMap(affine, {maximum - 1})));
  }
}
TEST_F(LayoutAffineEvaluationTest, SafeInt64BoundaryRemainsExact) {
  auto dim = getAffineDimExpr(0, &context);
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  auto result = evaluateLayoutMap(map(dim * 2, maximum), {maximum / 2});
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ((*result)[0], maximum - 1);
}
TEST_F(LayoutAffineEvaluationTest, InvalidDivisorsAndSignedDivisionOverflowFail) {
  auto dim = getAffineDimExpr(0, &context);
  auto zero = getAffineConstantExpr(0, &context);
  auto negative = getAffineConstantExpr(-1, &context);
  for (AffineExpr expression : {dim.floorDiv(zero), dim.ceilDiv(zero), dim % zero, dim % negative})
    EXPECT_TRUE(failed(evaluateLayoutMap(map(expression), {3})));
  for (AffineExpr expression : {dim.floorDiv(negative), dim.ceilDiv(negative)})
    EXPECT_TRUE(failed(evaluateLayoutMap(map(expression), {std::numeric_limits<int64_t>::min()})));
}
TEST_F(LayoutAffineEvaluationTest, SymbolsRemainUnsupported) {
  auto symbol = getAffineSymbolExpr(0, &context);
  auto symbolic = AffineLayoutMapAttr::get(&context, builder.getStrArrayAttr({"x"}),
      builder.getDenseI64ArrayAttr({ShapedType::kDynamic}), builder.getStrArrayAttr({"y"}),
      builder.getDenseI64ArrayAttr({16}), AffineMapAttr::get(AffineMap::get(1, 1, symbol)));
  EXPECT_TRUE(failed(evaluateLayoutMap(symbolic, {0})));
}
TEST_F(LayoutAffineEvaluationTest, EmptyResultMapRemainsUnsupported) {
  auto empty = AffineLayoutMapAttr::get(&context, builder.getStrArrayAttr({"x"}),
      builder.getDenseI64ArrayAttr({16}), builder.getArrayAttr({}),
      builder.getDenseI64ArrayAttr({}),
      AffineMapAttr::get(AffineMap::get(1, 0, ArrayRef<AffineExpr>{}, &context)));
  EXPECT_TRUE(failed(evaluateLayoutMap(empty, {0})));
}
} // namespace
