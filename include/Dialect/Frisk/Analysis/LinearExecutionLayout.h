#ifndef FRISK_ANALYSIS_LINEAREXECUTIONLAYOUT_H
#define FRISK_ANALYSIS_LINEAREXECUTIONLAYOUT_H

#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include <array>

namespace mlir::frisk {
class DistributedEncodingAttr;

/// Checked bit-linear representation, not an ownership/coverage certificate.
/// Carrier order is register,lane,warp,warp_group,cta, least-significant first.
/// Omitted carriers have explicit zero columns. Both axes are bounded to 256
/// bits; no product of hardware or logical extents is required.
struct LinearExecutionLayout {
  std::array<uint64_t, 5> topology{};
  std::array<unsigned, 5> hardwareWidths{}, hardwareOffsets{};
  SmallVector<unsigned> logicalWidths;
  unsigned hardwareBits = 0, logicalBits = 0;
  /// One row per logical bit, APInt width max(1, hardwareBits).
  SmallVector<llvm::APInt> rows;

  /// GF2Matrix requires nonempty axes: failure for either zero-width axis.
  FailureOr<GF2Matrix> getMatrix() const;
  /// Same row space, restricted to one carrier; failure for empty axes or an
  /// invalid carrier index. A zero-bit carrier is the unique zero-column map.
  FailureOr<GF2Matrix> getCarrierMatrix(unsigned carrier) const;
  /// Input/output APInt widths are max(1, hardwareBits/logicalBits).
  llvm::APInt apply(const llvm::APInt &hardware) const;
  /// Hardware counterexamples use cta,warp_group,warp,lane,register order.
  SmallVector<int64_t> hardwareCoordinate(const llvm::APInt &hardware) const;
  SmallVector<int64_t> logicalCoordinate(const llvm::APInt &logical) const;
};

/// Proven means only that the representation and positive replication are
/// structurally validated. It does NOT imply coverage or correct replication.
/// Unknown means the explicit 256-bit algebra limit was exceeded.
LayoutProof analyzeLinearExecutionLayout(DistributedEncodingAttr execution,
    RankedTensorType logicalType, LinearExecutionLayout &out);
} // namespace mlir::frisk
#endif
