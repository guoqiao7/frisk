#include "Dialect/Frisk/Analysis/BroadcastLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::frisk;

bool mlir::frisk::validBroadcastTypes(RankedTensorType input,
    RankedTensorType output, ArrayRef<int64_t> axes) {
  if (!input || !output || input.getRank() == 0 ||
      input.getRank() + axes.size() != unsigned(output.getRank()) ||
      input.getElementType() != output.getElementType()) return false;
  for (auto type : {input, output})
    if (!type.hasStaticShape() || llvm::any_of(type.getShape(), [](int64_t e) {
          return e <= 1 || !llvm::isPowerOf2_64(e);
        })) return false;
  int64_t previous = -1;
  for (int64_t axis : axes) {
    if (axis <= previous || axis >= output.getRank()) return false;
    previous = axis;
  }
  unsigned sourceAxis = 0;
  for (int64_t axis = 0; axis < output.getRank(); ++axis)
    if (!llvm::is_contained(axes, axis) &&
        output.getDimSize(axis) != input.getDimSize(sourceAxis++)) return false;
  return true;
}

FailureOr<DistributedEncodingAttr> mlir::frisk::projectBroadcastInputEncoding(
    DistributedEncodingAttr output, RankedTensorType inputType,
    RankedTensorType outputType, ArrayRef<int64_t> axes) {
  if (!validBroadcastTypes(inputType, outputType, axes)) return failure();
  ExecutionEnumeration checked;
  if (enumerateExecutionLayout(output, outputType, checked).status != ProofStatus::Proven)
    return failure();
  // This reuses only the coordinate projection algebra, not reduction's
  // contributor/tree semantics. Descending deletion preserves axis positions.
  // Each step keeps an ordered basis of the surviving register subspace and
  // retains all lane/warp/CTA columns, including replication columns.
  auto current = output;
  auto type = outputType;
  for (int64_t axis : llvm::reverse(axes)) {
    SmallVector<int64_t> shape(type.getShape());
    shape.erase(shape.begin() + axis);
    auto nextType = RankedTensorType::get(shape, type.getElementType());
    auto next = projectReductionEncoding(current, type, nextType, axis);
    if (failed(next)) return failure();
    current = *next;
    type = nextType;
  }
  return current;
}

LayoutProof mlir::frisk::proveBroadcastLayout(DistributedEncodingAttr input,
    DistributedEncodingAttr output, RankedTensorType inputType,
    RankedTensorType outputType, ArrayRef<int64_t> axes) {
  if (!validBroadcastTypes(inputType, outputType, axes))
    return {ProofStatus::Disproven, {}, "invalid broadcast shapes or ordered axes"};
  ExecutionEnumeration source, result;
  auto proof = enumerateExecutionLayout(input, inputType, source);
  if (proof.status != ProofStatus::Proven) return proof;
  proof = enumerateExecutionLayout(output, outputType, result);
  if (proof.status != ProofStatus::Proven) return proof;
  if (source.topology[4] != 1 || result.topology[4] != 1 ||
      !std::equal(source.topology.begin() + 1, source.topology.end(),
                  result.topology.begin() + 1))
    return {ProofStatus::Disproven, {}, "broadcast requires matching threads in one CTA"};
  // Canonical enumeration is register-minor. A set over (thread, logical
  // source coordinate) handles arbitrary register bases and source replication.
  // Both enumerations and this index obey the existing 65536-point proof bound.
  llvm::DenseSet<std::pair<uint64_t, uint64_t>> available;
  for (uint64_t h = 0; h < source.logical.size(); ++h)
    available.insert({h / source.topology[0], source.logical[h]});
  for (uint64_t h = 0; h < result.logical.size(); ++h) {
    auto coordinate = result.coordinate(result.logical[h]);
    uint64_t wanted = 0;
    for (unsigned axis = 0; axis < coordinate.size(); ++axis)
      if (!llvm::is_contained(axes, axis))
        wanted = wanted * outputType.getDimSize(axis) + coordinate[axis];
    if (!available.count({h / result.topology[0], wanted}))
      return {ProofStatus::Disproven, coordinate,
              "broadcast source coordinate unavailable in destination thread"};
  }
  return {ProofStatus::Proven, {}, "complete same-thread delete-axis broadcast coverage"};
}
