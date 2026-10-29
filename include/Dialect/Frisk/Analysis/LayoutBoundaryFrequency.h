#ifndef FRISK_ANALYSIS_LAYOUTBOUNDARYFREQUENCY_H
#define FRISK_ANALYSIS_LAYOUTBOUNDARYFREQUENCY_H

#include <cstdint>

namespace mlir {
class Operation;
namespace frisk {
enum class LayoutBoundaryFrequencyMode { Legacy, StaticTripCount };
struct LayoutBoundaryFrequency {
  uint64_t factor = 1;
  bool usesProxy = false;
  bool saturated = false;
};
/// Estimate execution frequency from ancestor loops, excluding insertionPoint.
/// Unknown loops use a factor of two; this proxy is not an upper bound.
LayoutBoundaryFrequency getLayoutBoundaryFrequency(
    Operation *insertionPoint, LayoutBoundaryFrequencyMode mode);
} // namespace frisk
} // namespace mlir
#endif
