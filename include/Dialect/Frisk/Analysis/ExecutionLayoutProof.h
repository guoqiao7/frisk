#ifndef FRISK_ANALYSIS_EXECUTIONLAYOUTPROOF_H
#define FRISK_ANALYSIS_EXECUTIONLAYOUTPROOF_H
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "llvm/ADT/StringRef.h"
namespace mlir::frisk {
class DistributedEncodingAttr;
enum class AccessKind;
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
