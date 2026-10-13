#ifndef FRISK_ANALYSIS_BROADCASTLAYOUTCONSTRAINTS_H
#define FRISK_ANALYSIS_BROADCASTLAYOUTCONSTRAINTS_H

#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"

namespace mlir::frisk {
bool validBroadcastTypes(RankedTensorType input, RankedTensorType output,
                         ArrayRef<int64_t> axes);
/// Finite reverse projection. There is deliberately no input->output inverse.
FailureOr<DistributedEncodingAttr> projectBroadcastInputEncoding(
    DistributedEncodingAttr output, RankedTensorType inputType,
    RankedTensorType outputType, ArrayRef<int64_t> axes);
/// Bounded actual-only coordinate/ownership proof, independent of projection.
LayoutProof proveBroadcastLayout(DistributedEncodingAttr input,
    DistributedEncodingAttr output, RankedTensorType inputType,
    RankedTensorType outputType, ArrayRef<int64_t> axes);
} // namespace mlir::frisk
#endif
