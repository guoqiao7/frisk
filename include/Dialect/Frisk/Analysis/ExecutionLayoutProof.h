#ifndef FRISK_ANALYSIS_EXECUTIONLAYOUTPROOF_H
#define FRISK_ANALYSIS_EXECUTIONLAYOUTPROOF_H
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "llvm/ADT/StringRef.h"
#include <array>
namespace mlir::frisk {
class DistributedEncodingAttr;
enum class AccessKind;
/// Validated, bounded index shared by execution and reduction proofs.
struct ExecutionEnumeration {
  SmallVector<uint64_t> logical, first;
  SmallVector<int64_t> shape;
  std::array<uint64_t, 5> topology;
  SmallVector<int64_t> hardware(uint64_t id) const {
    SmallVector<int64_t> point(5);
    for (unsigned c = 0; c < 5; ++c) {
      point[4-c] = id % topology[c]; id /= topology[c];
    }
    return point;
  }
  SmallVector<int64_t> coordinate(uint64_t id) const {
    SmallVector<int64_t> point(shape.size());
    for (size_t i = shape.size(); i-- > 0;) {
      point[i] = id % shape[i]; id /= shape[i];
    }
    return point;
  }
};
LayoutProof enumerateExecutionLayout(DistributedEncodingAttr execution,
    RankedTensorType logicalType, ExecutionEnumeration &out);
/// Pure bounded proof (65536 hardware/logical points). Ownership failures name
/// logical coordinates; duplicate/vector hardware counterexamples use canonical
/// (cta, warp_group, warp, lane, register) order, independent of map input order.
LayoutProof proveExecutionOwnership(DistributedEncodingAttr execution,
                                    RankedTensorType logicalType,
                                    StringRef writerPolicy);
/// Compose the execution map with an already Proven, matching storage footprint.
/// Recheck descriptor-to-root coordinates and nonoverlapping bit intervals before
/// consuming the footprint, including for the packed scalar baseline.
/// Reads check all holders; writes check the policy's elected owners. Width one
/// is the packed scalar baseline; wider accesses require aligned register chunks.
LayoutProof proveExecutionVectorAccess(DistributedEncodingAttr execution,
    const StorageAliasInfo &storageInfo, const StorageAliasFootprint &storage,
    unsigned vectorBytes, AccessKind access, StringRef writerPolicy);
} // namespace mlir::frisk
#endif
