#ifndef FRISK_ANALYSIS_LAYOUTCOSTOBSERVER_H
#define FRISK_ANALYSIS_LAYOUTCOSTOBSERVER_H

#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

namespace mlir::frisk {
/// Observes already computed charges, never proposals or a second cost model.
/// Owners are graph variable IDs for registers/shared-root, constraint IDs for
/// access/copy-setup/mma/reduction/conversion/rematerialization, and 0 for a
/// legacy target-aggregate. Shared roots use the first assigned graph variable;
/// shared boundary plans use their selected leader constraint. Callbacks must
/// not mutate the graph, assignment, target or IR. On evaluation failure, any
/// preceding contributions belong to the failed evaluation, not a final cost.
class LayoutCostObserver {
public:
  virtual ~LayoutCostObserver() = default;
  virtual void recordCost(StringRef kind, uint64_t stableOwner,
                          const CostEstimate &cost) = 0;
};
} // namespace mlir::frisk
#endif
