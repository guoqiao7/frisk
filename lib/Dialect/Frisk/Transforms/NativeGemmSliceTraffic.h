// Static traffic for the native two-warpgroup SS GEMM executor.
#pragma once
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace mlir::frisk {
struct NativeSliceTraffic {
  int64_t extent, width, stages;
  int64_t repeatedA = 0, repeatedB = 0, retainedSteps = 0, savedB = 0;
  int64_t sharedBytes = 0;

  int64_t sectors() const { return (repeatedA + repeatedB - savedB) / 32; }
  DictionaryAttr receipt(Builder &b) const {
    NamedAttrList out;
    auto add = [&](StringRef k, int64_t v) { out.set(k, b.getI64IntegerAttr(v)); };
    add("k_extent", extent);
    add("resident_atoms", width);
    add("staged_k_tiles", stages);
    add("repeated_a_bytes", repeatedA);
    add("repeated_b_bytes", repeatedB);
    add("retained_b_k16", retainedSteps);
    add("saved_b_bytes", savedB);
    add("shared_bytes", sharedBytes);
    add("extra_global_sectors", sectors());
    return out.getDictionary(b.getContext());
  }
};

// Count distinct copy ranges in each group in the accumulator's certified
// register order. A panel shared by both warpgroups is issued only once.
inline std::optional<NativeSliceTraffic>
priceNativeSlices(const SM90MmaGeometry &g, int64_t extent, int64_t width,
                  int64_t stages, bool columnSum) {
  int64_t atoms = g.mRepeat * g.nRepeat;
  if (g.threads != 256 || g.k != 16 || width < 1 || width > atoms ||
      extent <= 0 || extent % 16 || extent > INT32_MAX / std::max(g.m, g.n) ||
      g.m > INT32_MAX / g.n || (stages != 1 && stages != 2))
    return std::nullopt;
  NativeSliceTraffic result{extent, width, stages};
  int64_t slices = llvm::divideCeil(atoms, width);
  bool fullB = true;
  for (int64_t begin = 0; begin < atoms; begin += width) {
    llvm::SmallBitVector rows(g.m / 64), cols(g.n / g.atomN);
    for (int64_t i = begin; i < std::min(begin + width, atoms); ++i) {
      int64_t mr = i / g.nRepeat, nr = i % g.nRepeat;
      for (int wg = 0; wg < 2; ++wg) {
        int64_t gr = wg / g.gN;
        rows.set(g.interleavedRows ? mr * g.gM + gr : gr * g.mRepeat + mr);
        cols.set((wg % g.gN) * g.nRepeat + nr);
      }
    }
    result.repeatedA += int64_t(rows.count()) * 64 * extent * 2;
    result.repeatedB += int64_t(cols.count()) * g.atomN * extent * 2;
    fullB &= cols.all();
  }
  result.repeatedA -= g.m * extent * 2;
  result.repeatedB -= g.n * extent * 2;

  // Same memory contract as _b_staging. The executor independently
  // checks this receipt against its selected copy ranges before compiling.
  int64_t scratch = std::max(64 * (g.m + g.n), 32 * g.atomN);
  int64_t partials = columnSum ? g.m / 64 * g.n * 4 : 0;
  result.sharedBytes = std::max<int64_t>(65536, scratch + partials);
  constexpr int64_t smBytes = 228 * 1024, reserved = 1024;
  if (result.sharedBytes > smBytes - reserved) return std::nullopt;
  int64_t live = width * (g.atomN / 2) + 4;
  if (!columnSum || slices == 1 || !fullB || live > 255) return result;
  int64_t registers = llvm::alignTo(live, 8);
  int64_t ctas = std::min(65536 / (256 * registers), smBytes / (result.sharedBytes + reserved));
  int64_t base = std::max(stages * g.m * 32, 32 * g.atomN);
  int64_t slab = g.n * 32, steps = extent / 16;
  int64_t available = (smBytes / ctas - reserved - base - partials) / slab;
  int64_t retained = available >= steps ? steps : (available - stages) / stages * stages;
  if (retained <= 0) return result;
  result.retainedSteps = retained;
  result.savedB = (slices - 1) * retained * slab;
  int64_t used = retained == steps ? steps : retained + stages;
  result.sharedBytes = std::max(result.sharedBytes, base + used * slab + partials);
  return result;
}
} // namespace mlir::frisk
