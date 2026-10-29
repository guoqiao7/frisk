#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::frisk;

FailureOr<GF2Matrix> LinearExecutionLayout::getMatrix() const {
  // GF2Matrix requires both dimensions to be nonempty. Unit-only logical
  // domains and hardware domains without bits are handled by their consumers.
  if (!logicalBits || !hardwareBits) return failure();
  return GF2Matrix::get(logicalBits, hardwareBits, rows);
}

FailureOr<GF2Matrix>
LinearExecutionLayout::getCarrierMatrix(unsigned carrier) const {
  if (carrier >= 5 || !logicalBits || !hardwareWidths[carrier]) return failure();
  SmallVector<APInt> block;
  for (const auto &row : rows)
    block.push_back(row.extractBits(hardwareWidths[carrier], hardwareOffsets[carrier]));
  return GF2Matrix::get(logicalBits, hardwareWidths[carrier], block);
}

APInt LinearExecutionLayout::apply(const APInt &hardware) const {
  assert(hardware.getBitWidth() == std::max(1u, hardwareBits));
  APInt logical(std::max(1u, logicalBits), 0);
  for (unsigned r = 0; r < logicalBits; ++r)
    if ((rows[r] & hardware).popcount() & 1) logical.setBit(r);
  return logical;
}

SmallVector<int64_t>
LinearExecutionLayout::hardwareCoordinate(const APInt &hardware) const {
  SmallVector<int64_t> point(5, 0);
  for (unsigned c = 0; c < 5; ++c)
    if (hardwareWidths[c])
      point[4-c] = hardware.extractBitsAsZExtValue(hardwareWidths[c], hardwareOffsets[c]);
  return point;
}

SmallVector<int64_t>
LinearExecutionLayout::logicalCoordinate(const APInt &logical) const {
  SmallVector<int64_t> point(logicalWidths.size(), 0);
  unsigned offset = 0;
  for (unsigned axis = 0; axis < logicalWidths.size(); ++axis) {
    if (logicalWidths[axis])
      point[axis] = logical.extractBitsAsZExtValue(logicalWidths[axis], offset);
    offset += logicalWidths[axis];
  }
  return point;
}

LayoutProof mlir::frisk::analyzeLinearExecutionLayout(
    DistributedEncodingAttr execution, RankedTensorType type,
    LinearExecutionLayout &out) {
  out = LinearExecutionLayout{};
  auto reject = [](StringRef reason) -> LayoutProof {
    return {ProofStatus::Disproven, {}, reason.str()};
  };
  auto limit = []() -> LayoutProof {
    return {ProofStatus::Unknown, {}, "symbolic execution bit budget exceeded (256)"};
  };
  if (!execution || !type || !type.hasStaticShape())
    return reject("execution requires a static logical tensor");
  auto topology = execution.getTopology();
  auto map = dyn_cast_or_null<BitLinearLayoutMapAttr>(execution.getMap());
  if (!topology || topology.size() != 5 || !map)
    return reject("malformed hardware topology or non-bit-linear execution map");
  for (unsigned c = 0; c < 5; ++c) {
    int64_t extent = topology[c];
    if (extent <= 0 || !llvm::isPowerOf2_64(extent) || (c == 1 && extent > 32))
      return reject("invalid hardware topology extent");
    out.topology[c] = extent;
    out.hardwareWidths[c] = llvm::Log2_64(extent);
    out.hardwareOffsets[c] = out.hardwareBits;
    if (out.hardwareWidths[c] > 256 - out.hardwareBits) return limit();
    out.hardwareBits += out.hardwareWidths[c];
  }
  auto names = map.getInputNames(), outputNames = map.getOutputNames();
  auto widths = map.getInputBitWidths(), outputWidths = map.getOutputBitWidths();
  if (!names || !widths || !outputNames || !outputWidths ||
      names.size() != widths.size() || outputNames.size() != outputWidths.size() ||
      outputWidths.size() != unsigned(type.getRank()))
    return reject("malformed execution map dimensions");
  constexpr std::array<StringLiteral, 5> carriers = {
      "register", "lane", "warp", "warp_group", "cta"};
  llvm::StringSet<> seen;
  SmallVector<unsigned> canonicalColumns;
  for (unsigned i = 0; i < names.size(); ++i) {
    auto name = dyn_cast<StringAttr>(names[i]);
    if (!name || !seen.insert(name.getValue()).second)
      return reject("invalid or repeated execution carrier");
    unsigned c = 0;
    while (c < 5 && carriers[c] != name.getValue()) ++c;
    if (c == 5 || widths[i] < 0 || widths[i] > 16 ||
        unsigned(widths[i]) != out.hardwareWidths[c])
      return reject("carrier width does not match hardware topology");
    for (unsigned bit = 0; bit < unsigned(widths[i]); ++bit)
      canonicalColumns.push_back(out.hardwareOffsets[c] + bit);
  }
  seen.clear();
  for (unsigned i = 0; i < outputWidths.size(); ++i) {
    int64_t extent = type.getDimSize(i), width = outputWidths[i];
    auto name = dyn_cast<StringAttr>(outputNames[i]);
    if (!name || !seen.insert(name.getValue()).second || extent <= 0 ||
        !llvm::isPowerOf2_64(extent) || width < 0 || width > 16 ||
        unsigned(width) != llvm::Log2_64(extent))
      return reject("logical output widths do not match tensor shape");
    if (unsigned(width) > 256 - out.logicalBits) return limit();
    out.logicalWidths.push_back(width);
    out.logicalBits += width;
  }
  auto matrix = map.getMatrix();
  auto matrixType = matrix ? dyn_cast<RankedTensorType>(matrix.getType()) : RankedTensorType();
  if (!matrixType || matrixType.getRank() != 2 ||
      !matrixType.getElementType().isInteger(1) ||
      matrixType.getDimSize(0) != out.logicalBits ||
      matrixType.getDimSize(1) != int64_t(canonicalColumns.size()))
    return reject("malformed execution matrix");
  auto replication = execution.getReplication();
  if (!replication || !replication.getValue().isSignedIntN(64) ||
      replication.getValue().getSExtValue() <= 0)
    return reject("replication must be a positive signed 64-bit integer");
  out.rows.assign(out.logicalBits, APInt(std::max(1u, out.hardwareBits), 0));
  // Use the existing checked GF2 parser only after validating its nonempty-axis
  // preconditions; no second dense-attribute matrix parser is maintained here.
  if (out.logicalBits && !canonicalColumns.empty()) {
    auto supplied = map.getMatrixValue();
    if (failed(supplied)) return reject("malformed execution matrix");
    auto transposed = supplied->transpose();
    for (unsigned r = 0; r < out.logicalBits; ++r) {
      APInt unit(out.logicalBits, 0);
      unit.setBit(r);
      APInt row = transposed.apply(unit);
      for (unsigned c = 0; c < canonicalColumns.size(); ++c)
        if (row[c]) out.rows[r].setBit(canonicalColumns[c]);
    }
  }
  return {ProofStatus::Proven, {}, "validated symbolic execution representation (not ownership)"};
}
