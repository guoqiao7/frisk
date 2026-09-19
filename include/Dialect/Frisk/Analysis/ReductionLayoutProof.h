#ifndef FRISK_ANALYSIS_REDUCTIONLAYOUTPROOF_H
#define FRISK_ANALYSIS_REDUCTIONLAYOUTPROOF_H
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include <array>
#include <vector>
namespace mlir::frisk {
using ReductionHolder = std::array<int64_t, 5>; // cta, warp_group, warp, lane, register
struct ReductionNode {
  int64_t logicalInput = -1; // leaf iff nonnegative; flattened source coordinate
  int64_t left = -1, right = -1; // earlier node indices for a merge
  ReductionHolder holder{};
  bool operator==(const ReductionNode &o) const {
    return logicalInput == o.logicalInput && left == o.left && right == o.right && holder == o.holder;
  }
};
struct ReductionFiber {
  int64_t logicalOutput = -1;
  std::vector<ReductionNode> nodes;
  int64_t root = -1;
  std::vector<ReductionHolder> broadcasts;
  bool operator==(const ReductionFiber &o) const {
    return logicalOutput == o.logicalOutput && nodes == o.nodes && root == o.root && broadcasts == o.broadcasts;
  }
};
struct ReductionLayoutProof {
  LayoutProof proof;
  std::vector<ReductionFiber> fibers;
  std::string scope;
};
/// Delete one logical axis and take the ordered register-column quotient.
/// Preserves non-register topology and logical names; recomputes replication.
/// Fails for malformed/unsupported shapes, uncovered maps or exceeded budgets.
/// These layout-only APIs require matching element types but do not enforce the
/// operation's f16/bf16/f32 whitelist or combiner semantics. Callers must first
/// verify ReduceTensorOp (or establish the same mathematical prerequisites).
FailureOr<DistributedEncodingAttr> projectReductionEncoding(
    DistributedEncodingAttr source, RankedTensorType sourceType,
    RankedTensorType resultType, int64_t axis);
/// Pure bounded dependency proof, not GPU lowering or shared resource validation.
/// Elects one physical first_owner for each distinct source coordinate, forms
/// canonical_fiber_tree_v1 and broadcasts completed roots to every result holder.
/// Each source/result hardware and logical enumeration is capped at 65536.
/// Thread-count/target binding is checked by Target, not this topology analysis.
ReductionLayoutProof buildReductionLayoutProof(DistributedEncodingAttr source,
    DistributedEncodingAttr result, RankedTensorType sourceType,
    RankedTensorType resultType, int64_t axis);
/// Reconstruct and compare the unique canonical dependency proof from actual
/// encodings. No candidate generation, candidate-pair enumeration or search.
LayoutProof verifyReductionLayoutProof(DistributedEncodingAttr source,
    DistributedEncodingAttr result, RankedTensorType sourceType,
    RankedTensorType resultType, int64_t axis, const ReductionLayoutProof &proof);
} // namespace mlir::frisk
#endif
