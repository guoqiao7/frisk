#ifndef FRISK_TARGET_SM90_SM90LAYOUTTARGET_H
#define FRISK_TARGET_SM90_SM90LAYOUTTARGET_H

#include <memory>

#include "Dialect/Frisk/Analysis/LayoutTarget.h"

namespace mlir::frisk {
struct SM90CostProfile;

std::unique_ptr<LayoutTarget> createSM90LayoutTarget(LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy);

std::unique_ptr<LayoutTarget> createSM90LayoutTarget(
    LayoutBoundaryFrequencyMode mode, const SM90CostProfile &profile);

} // namespace mlir::frisk

#endif // FRISK_TARGET_SM90_SM90LAYOUTTARGET_H
