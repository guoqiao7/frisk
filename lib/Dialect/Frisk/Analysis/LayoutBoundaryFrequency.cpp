#include "Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/APInt.h"
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::frisk;

namespace {
std::optional<APInt> getSignedConstant(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr)))
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(attr);
  if (!integer || !(integer.getType().isIndex() ||
                   integer.getType().isSignlessInteger()))
    return std::nullopt;
  return integer.getValue();
}

LayoutBoundaryFrequency getLoopFrequency(Operation *op) {
  auto loop = dyn_cast<scf::ForOp>(op);
  // Older SCF versions do not declare this attribute. Its presence still
  // prevents us from assuming signed comparison semantics.
  if (!loop || loop->hasAttr("unsignedCmp"))
    return {2, true, false};
  auto lower = getSignedConstant(loop.getLowerBound());
  auto upper = getSignedConstant(loop.getUpperBound());
  auto step = getSignedConstant(loop.getStep());
  if (!lower || !upper || !step || step->isNegative() || step->isZero())
    return {2, true, false};

  // Widen before subtraction, preserving the sign of arbitrary-width integer
  // constants. Two extra bits also leave room for rounding the quotient up.
  unsigned width = std::max({lower->getBitWidth(), upper->getBitWidth(),
                             step->getBitWidth()}) + 2;
  APInt lb = lower->sext(width), ub = upper->sext(width);
  if (ub.sle(lb))
    return {0, false, false};
  APInt distance = ub - lb, stride = step->sext(width);
  APInt count = distance.udiv(stride);
  if (!distance.urem(stride).isZero())
    ++count;
  if (count.getActiveBits() > 64)
    return {std::numeric_limits<uint64_t>::max(), false, true};
  return {count.getZExtValue(), false, false};
}
} // namespace

LayoutBoundaryFrequency mlir::frisk::getLayoutBoundaryFrequency(
    Operation *insertionPoint, LayoutBoundaryFrequencyMode mode) {
  LayoutBoundaryFrequency result;
  for (Operation *op = insertionPoint ? insertionPoint->getParentOp() : nullptr;
       op; op = op->getParentOp()) {
    if (mode == LayoutBoundaryFrequencyMode::Legacy
            ? !isa<scf::ForOp, scf::WhileOp>(op)
            : !isa<LoopLikeOpInterface>(op))
      continue;
    auto loop = mode == LayoutBoundaryFrequencyMode::Legacy
                    ? LayoutBoundaryFrequency{2, true, false}
                    : getLoopFrequency(op);
    // Keep traversing even after saturation: a provably empty ancestor wins
    // over both the proxy flag and all intermediate arithmetic overflow.
    if (loop.factor == 0)
      return {0, false, false};
    result.usesProxy |= loop.usesProxy;
    result.saturated |= loop.saturated;
    if (result.factor > std::numeric_limits<uint64_t>::max() / loop.factor) {
      result.factor = std::numeric_limits<uint64_t>::max();
      result.saturated = true;
    } else {
      result.factor *= loop.factor;
    }
  }
  return result;
}
