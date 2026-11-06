#ifndef FRISK_TEST_LAYOUTCOSTOBSERVER_H
#define FRISK_TEST_LAYOUTCOSTOBSERVER_H
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "gtest/gtest.h"
namespace mlir::frisk {
inline bool operator==(const CostVector &a, const CostVector &b) {
  return !(a < b) && !(b < a);
}
struct TestCostObserver : LayoutCostObserver {
  struct Part { std::string kind; uint64_t owner; CostEstimate cost; };
  SmallVector<Part> parts;
  void recordCost(StringRef kind, uint64_t owner,
                  const CostEstimate &cost) override {
    parts.push_back({kind.str(), owner, cost});
  }
  void expectSum(const CostEstimate &expected) const {
    CostEstimate total;
    for (const auto &part : parts) total = addLayoutCosts(total, part.cost);
    EXPECT_EQ(total.cost, expected.cost);
    EXPECT_EQ(total.saturated, expected.saturated);
  }
};
}
#endif
