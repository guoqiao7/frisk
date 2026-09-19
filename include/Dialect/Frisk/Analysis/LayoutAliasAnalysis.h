#ifndef FRISK_ANALYSIS_LAYOUTALIASANALYSIS_H
#define FRISK_ANALYSIS_LAYOUTALIASANALYSIS_H

#include "Dialect/Frisk/Analysis/LayoutCommon.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::frisk {
class StorageLayoutAttr;

/// Storage maps use the common root's aligned pointer as their byte/bit origin.
/// The descriptor offset is included in [lowerBit, upperBit), never added to a
/// bound map a second time. rootAlignment is a guarantee, not a child demand.
struct StorageAliasInfo {
  Value root;
  MemRefType rootType;
  MemRefType viewType;
  AffineMap viewToRoot;
  std::string rootKey;
  uint64_t lowerBit = 0, upperBit = 0;
  uint64_t rootAlignment = 1;
  std::string alignmentEvidence = "baseline byte alignment; no stronger root contract";
};

struct StorageAliasPointAddress {
  SmallVector<int64_t> view, root;
  uint64_t begin = 0, end = 0;
};
/// Immutable graph-local proof input, sorted by physical bit start. Rebuild
/// after IR changes; a failed proof must never be used as a partial footprint.
struct StorageAliasFootprint {
  Value root;
  MemRefType rootType;
  LayoutProof proof;
  SmallVector<StorageAliasPointAddress> entries;
};
StorageAliasFootprint buildStorageAliasFootprint(const StorageAliasInfo &info,
                                                StorageLayoutAttr candidate);
LayoutProof proveStorageAliasFootprints(const StorageAliasFootprint &a,
                                        const StorageAliasFootprint &b);

FailureOr<StorageAliasInfo> analyzeStorageAlias(Value endpoint);
/// Checked coordinate evaluation shared by alias and execution proofs.
FailureOr<SmallVector<int64_t>> evaluateStorageViewCoordinates(
    const StorageAliasInfo &info, ArrayRef<int64_t> point);
LayoutProof verifyStorageAliasCandidate(const StorageAliasInfo &info,
                                        StorageLayoutAttr candidate);
LayoutProof proveStorageAliasCompatible(const StorageAliasInfo &a,
                                        StorageLayoutAttr aCandidate,
                                        const StorageAliasInfo &b,
                                        StorageLayoutAttr bCandidate);
/// Only restrict known source coordinates; never extend a partial view.
FailureOr<StorageLayoutAttr> projectStorageAliasCandidate(
    const StorageAliasInfo &source, StorageLayoutAttr candidate,
    const StorageAliasInfo &destination);
FailureOr<StorageLayoutAttr>
buildRootLinearStorageCandidate(const StorageAliasInfo &rootInfo);
/// Finite Copy suggestion, rebased to the destination descriptor origin.
/// This is not an alias proof or a hard equality between independent roots.
FailureOr<StorageLayoutAttr> rebaseStorageCopyCandidate(
    const StorageAliasInfo &source, StorageLayoutAttr candidate,
    const StorageAliasInfo &destination);
} // namespace mlir::frisk
#endif
