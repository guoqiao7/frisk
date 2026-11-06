#include "Dialect/Frisk/Analysis/BroadcastLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::frisk;

bool mlir::frisk::validBroadcastTypes(RankedTensorType input,
    RankedTensorType output, ArrayRef<int64_t> axes) {
  if (!input || !output || input.getRank() == 0 ||
      input.getRank() + axes.size() != unsigned(output.getRank()) ||
      input.getElementType() != output.getElementType()) return false;
  for (auto type : {input, output})
    if (!type.hasStaticShape() || llvm::any_of(type.getShape(), [](int64_t e) {
          return e <= 0 || !llvm::isPowerOf2_64(e);
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
  if (proveExecutionOwnership(output, outputType, "first_owner").status != ProofStatus::Proven)
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
  // Structural validity is not coverage: certify both ownership domains before
  // using their checked matrices in the same-thread subspace calculation.
  auto proof = proveExecutionOwnership(input, inputType, "first_owner");
  if (proof.status != ProofStatus::Proven) return proof;
  proof = proveExecutionOwnership(output, outputType, "first_owner");
  if (proof.status != ProofStatus::Proven) return proof;
  LinearExecutionLayout source, result;
  proof = analyzeLinearExecutionLayout(input, inputType, source);
  if (proof.status != ProofStatus::Proven) return proof;
  proof = analyzeLinearExecutionLayout(output, outputType, result);
  if (proof.status != ProofStatus::Proven) return proof;
  if (source.topology[4] != 1 || result.topology[4] != 1 ||
      !std::equal(source.topology.begin() + 1, source.topology.end(),
                  result.topology.begin() + 1))
    return {ProofStatus::Disproven, {}, "broadcast requires matching threads in one CTA"};
  // With no retained logical bits every validated source holder owns the sole
  // coordinate. Matching thread domains above then guarantee availability,
  // including a hardware domain with zero bits. Avoid empty GF2 matrices.
  if (!source.logicalBits)
    return {ProofStatus::Proven, {}, "unit-coordinate same-thread broadcast coverage"};
  // Delete output axes in actual logical-bit order. D contains destination
  // register columns followed by (destination XOR source) thread columns;
  // arbitrary source registers can compensate exactly image(R).
  unsigned sourceRegisters = source.hardwareWidths[0];
  unsigned destinationRegisters = result.hardwareWidths[0];
  unsigned threadBits = source.hardwareBits - sourceRegisters;
  SmallVector<APInt> registers, differences, combined;
  unsigned sourceRow = 0, destinationRow = 0;
  for (unsigned axis = 0; axis < result.logicalWidths.size(); ++axis) {
    for (unsigned bit = 0; bit < result.logicalWidths[axis]; ++bit, ++destinationRow) {
      if (llvm::is_contained(axes, axis)) continue;
      APInt reg(std::max(1u, sourceRegisters), 0);
      if (sourceRegisters)
        reg = source.rows[sourceRow].extractBits(sourceRegisters, 0);
      APInt difference = result.rows[destinationRow];
      if (threadBits)
        difference ^= source.rows[sourceRow].extractBits(threadBits, sourceRegisters)
            .zextOrTrunc(result.hardwareBits).shl(destinationRegisters);
      APInt joined = reg.zextOrTrunc(sourceRegisters + result.hardwareBits);
      joined |= difference.zext(sourceRegisters + result.hardwareBits).shl(sourceRegisters);
      registers.push_back(reg);
      differences.push_back(difference);
      combined.push_back(joined);
      ++sourceRow;
    }
  }
  // The unit-coordinate case above guarantees positive source logical bits;
  // ownership guarantees a nonempty destination hardware domain. Pad only the
  // possibly empty source register block with a zero column for GF2Matrix.
  auto registerMap = GF2Matrix::get(source.logicalBits,
      std::max(1u, sourceRegisters), registers);
  auto combinedMap = GF2Matrix::get(source.logicalBits,
      sourceRegisters + result.hardwareBits, combined);
  if (combinedMap->rank() == registerMap->rank())
    return {ProofStatus::Proven, {}, "symbolic same-thread delete-axis broadcast coverage"};

  // A left-kernel separator of R detects a D column outside image(R). Setting
  // only that destination hardware bit yields a real unavailable coordinate:
  // projected B*h XOR A_thread*t cannot be produced by any source register.
  auto separators = registerMap->transpose().kernelBasis();
  for (unsigned column = 0; column < result.hardwareBits; ++column) {
    APInt value(source.logicalBits, 0);
    for (unsigned row = 0; row < source.logicalBits; ++row)
      if (differences[row][column]) value.setBit(row);
    if (llvm::any_of(separators, [&](const APInt &separator) {
          return (separator & value).popcount() & 1;
        })) {
      APInt hardware(result.hardwareBits, 0);
      hardware.setBit(column);
      return {ProofStatus::Disproven,
              result.logicalCoordinate(result.apply(hardware)),
              "symbolic broadcast source coordinate unavailable in destination thread"};
    }
  }
  llvm_unreachable("rank increased without a column outside the register image");
}
