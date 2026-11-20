#ifndef FRISK_ANALYSIS_LAYOUTDEMANDFOOTPRINT_H
#define FRISK_ANALYSIS_LAYOUTDEMANDFOOTPRINT_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>

namespace mlir::frisk::detail {
// Internal arithmetic-only primitive. The caller remains responsible for the
// full storage/ownership/affine-emission proofs; this cache has no IR identity.
struct DemandAccessFootprintKey {
  bool global;
  uint64_t alignment;
  std::array<uint64_t,2> active{};
  std::array<uint64_t,128> offsets{};
  bool operator<(const DemandAccessFootprintKey &other) const {
    return std::tie(global,alignment,active,offsets)<
           std::tie(other.global,other.alignment,other.active,other.offsets);
  }
};
struct DemandAccessFootprint {
  uint64_t warpIssues=0,sectors=0,bankExcess=0;
};
class DemandAccessFootprintCache {
public:
  // Only successful selected-space footprints are memoized, capped at 256
  // entries. A zero capacity bypasses lookup/insertion. Original addresses are
  // checked for phase-addition overflow even on a hit.
  std::optional<DemandAccessFootprint> evaluate(
      const std::array<uint64_t,128> &offsets,
      const std::array<bool,128> &active,bool global,uint64_t alignment,
      size_t capacity);
  size_t size() const { return entries.size(); }
private:
  std::map<DemandAccessFootprintKey,DemandAccessFootprint> entries;
};
} // namespace mlir::frisk::detail
#endif
