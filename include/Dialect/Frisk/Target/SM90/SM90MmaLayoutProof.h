#ifndef FRISK_SM90_MMA_LAYOUT_PROOF_H
#define FRISK_SM90_MMA_LAYOUT_PROOF_H
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include <array>
namespace mlir::frisk {
struct SM90MmaGeometry {
  int64_t m, n, k, threads, gM, gN, atomN, mRepeat, nRepeat, kRepeat;
};
FailureOr<SM90MmaGeometry> getSM90MmaGeometry(MmaOp op, int64_t threads);
/// Logical (row,column) for register/lane/warp/warp_group carriers. A slots
/// denote successive low/high halfwords of four packed 32-bit registers.
std::array<int64_t, 2> getSM90MmaFragmentCoordinate(
    const SM90MmaGeometry &g, bool operandA, bool transA,
    int64_t reg, int64_t lane, int64_t warp, int64_t group);
LayoutProof verifySM90MmaFragment(const SM90MmaGeometry &g,
    bool operandA, bool transA, RankedTensorType type, DistributedEncodingAttr encoding);
/// PTX 9.4 section 9.7.17.5.1.2 canonical address equation; offsets in bytes.
uint64_t getSM90DescriptorAddress(StringRef major, unsigned swizzle,
    uint64_t leading, uint64_t stride, uint64_t start, uint64_t phase,
    uint64_t mn, uint64_t k);
LayoutProof verifySM90MmaDescriptor(const StorageAliasInfo &info,
    StorageLayoutAttr layout, MmaDescriptorPlanAttr plan, bool kIsSecond,
    int64_t mnExtent, int64_t kExtent, int64_t atomMN);
LayoutProof verifySM90MmaContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings, Attribute binding);
} // namespace mlir::frisk
#endif
