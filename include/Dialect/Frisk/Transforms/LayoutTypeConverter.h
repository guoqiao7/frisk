#ifndef FRISK_TRANSFORMS_LAYOUTTYPECONVERTER_H
#define FRISK_TRANSFORMS_LAYOUTTYPECONVERTER_H

#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::frisk {
/// Layout inference is SSA-value dependent, not a conversion of Type alone.
/// The inherited Type-only path deliberately rejects all tensor types.
class LayoutTypeConverter : public TypeConverter {
public:
  LayoutTypeConverter(const LayoutConstraintGraph &graph,
                      const LayoutSolution &solution);
  FailureOr<RankedTensorType> convertLayoutBearingTensor(Value value) const;

private:
  const LayoutConstraintGraph &graph;
  const LayoutSolution &solution;
};

/// target is borrowed for this call; null preserves the legacy SM90 default.
LogicalResult materializeDistributedLayouts(
    Operation *root, const LayoutConstraintGraph &graph,
    const LayoutSolution &solution, ArrayRef<LayoutConversionEdge> conversions,
    bool authorizeInternalOptimization = false,
    LayoutTarget *target = nullptr,
    ConversionPolicy expectedPolicy = ConversionPolicy::Full);
} // namespace mlir::frisk
#endif
