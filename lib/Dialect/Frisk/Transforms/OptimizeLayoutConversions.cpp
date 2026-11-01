#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_OPTIMIZELAYOUTCONVERSIONS
#include "Dialect/Frisk/Transforms/Passes.h.inc"
namespace {
constexpr StringLiteral inferredLayout = "frisk.inferred_result_layout";

bool hasHardAttributes(Operation *op) {
  return llvm::any_of(op->getAttrs(), [](NamedAttribute attr) {
    return attr.getName().strref().starts_with("frisk.") &&
           attr.getName().strref() != inferredLayout;
  });
}

bool supportedElementwise(Operation *op) {
  if (!op || hasHardAttributes(op) || op->getNumRegions() ||
      op->getNumResults() != 1 || !op->hasTrait<OpTrait::Elementwise>())
    return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects || !effects.hasNoEffect()) return false;
  // Extend changes element type, but not shape or the physical ownership map.
  // There is no modeled tensor broadcast in this IR: do not invent one here.
  StringRef name = op->getName().getStringRef();
  return isRematerializable(op) || name == "arith.extf" ||
         name == "arith.extsi" || name == "arith.extui";
}

FailureOr<CostEstimate> conversionCost(Value source, Attribute destination,
                                       Operation *at, LayoutConversionEdge &edge, LayoutTarget &target) {
  auto type = dyn_cast<RankedTensorType>(source.getType());
  if (!type) return failure();
  LayoutVar var;
  var.kind = LayoutKind::Distributed;
  var.shapedType = type;
  var.value = source;
  edge.insertionPoint = at;
  auto proxy = evaluateLayoutConversionCost(var, type.getEncoding(), destination, edge, target.boundaryFrequencyMode());
  return failed(proxy) ? FailureOr<CostEstimate>(failure()) : target.adjustConversionCost(*proxy, edge);
}

bool verifiedCandidate(ModuleOp module, LayoutTarget &target) {
  // An unsupported destination is a rejected optimization, not a pass failure.
  ScopedDiagnosticHandler quiet(module.getContext(), [](Diagnostic &) {
    return success();
  });
  return succeeded(verifyMaterializedLayouts(module, target));
}

bool commute(ConvertLayoutOp convert, ModuleOp module, LayoutTarget &target) {
  Operation *producer = convert.getSource().getDefiningOp();
  if (hasHardAttributes(convert) || !supportedElementwise(producer) ||
      !producer->getResult(0).hasOneUse() ||
      producer->getBlock() != convert->getBlock() ||
      producer->getNextNode() != convert.getOperation()) return false;
  // Encoded types alone are hard seeds. Only the materializer can authorize
  // retyping a result that was originally unencoded, recording its exact
  // selected encoding. Never infer permission from absent hard attributes.
  if (producer->getAttr(inferredLayout) !=
      cast<RankedTensorType>(producer->getResult(0).getType()).getEncoding())
    return false;
  Value input;
  for (Value operand : producer->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType())) continue;
    if (input && input != operand) return false; // No conversion/code growth.
    input = operand;
  }
  if (!input || !DominanceInfo(module).properlyDominates(input, producer))
    return false;
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resultType = convert.getResult().getType();
  if (inputType.getShape() != resultType.getShape()) return false;
  LayoutConversionEdge beforeEdge, afterEdge;
  auto before = conversionCost(convert.getSource(), resultType.getEncoding(),
                               convert, beforeEdge, target);
  auto after = conversionCost(input, resultType.getEncoding(), producer, afterEdge, target);
  if (failed(before) || failed(after) || before->saturated || after->saturated ||
      before->cost < after->cost) return false;

  OpBuilder builder(producer);
  auto movedType = RankedTensorType::get(inputType.getShape(),
      inputType.getElementType(), resultType.getEncoding());
  auto moved = builder.create<ConvertLayoutOp>(convert.getLoc(), movedType, input);
  SmallVector<Value> operands(producer->getOperands());
  Type oldType = producer->getResult(0).getType();
  for (OpOperand &operand : producer->getOpOperands())
    if (operand.get() == input) operand.set(moved.getResult());
  producer->getResult(0).setType(resultType);
  // Keep the now-identity original conversion until verification succeeds so
  // rejection restores the exact original SSA graph, including all uses.
  if (!verifiedCandidate(module, target)) {
    producer->setOperands(operands);
    producer->getResult(0).setType(oldType);
    moved.erase();
    return false;
  }
  convert.replaceAllUsesWith(producer->getResult(0));
  convert.erase();
  producer->setAttr(inferredLayout, resultType.getEncoding());
  return true;
}

bool hoistLoop(ConvertLayoutOp convert, ModuleOp module, LayoutTarget &target) {
  auto loop = dyn_cast_or_null<scf::ForOp>(convert->getParentOp());
  if (!loop || hasHardAttributes(convert) || hasHardAttributes(loop) ||
      !DominanceInfo(module).properlyDominates(convert.getSource(), loop))
    return false;
  for (Operation *parent = loop->getParentOp(); parent; parent = parent->getParentOp())
    if (!isa<ModuleOp, func::FuncOp, scf::ForOp>(parent)) return false;
  LayoutConversionEdge oldEdge, newEdge;
  auto before = conversionCost(convert.getSource(), convert.getResult().getType().getEncoding(),
                               convert, oldEdge, target);
  auto after = conversionCost(convert.getSource(), convert.getResult().getType().getEncoding(),
                              loop, newEdge, target);
  // Pure is not permission to speculate a shuffle or a collective barrier.
  if (failed(before) || failed(after) || oldEdge.exchange != ConversionExchangeKind::Register ||
      before->saturated || after->saturated || before->cost < after->cost) return false;
  if (target.boundaryFrequencyMode() == LayoutBoundaryFrequencyMode::StaticTripCount &&
      !(after->cost.conversionBytesAndSync < before->cost.conversionBytesAndSync))
    return false;
  for (Operation &op : *loop.getBody()) {
    if (&op == convert.getOperation() || isa<scf::YieldOp>(op)) continue;
    if (op.getNumRegions() || hasHardAttributes(&op)) return false;
    if (op.getName().getStringRef() == "arith.constant" && isMemoryEffectFree(&op)) continue;
    if (!supportedElementwise(&op)) return false;
  }
  Operation *next = convert->getNextNode();
  convert->moveBefore(loop);
  if (!verifiedCandidate(module, target)) {
    convert->moveBefore(next);
    return false;
  }
  return true;
}

class OptimizeLayoutConversionsPass final
    : public impl::OptimizeLayoutConversionsBase<OptimizeLayoutConversionsPass> {
public:
  OptimizeLayoutConversionsPass() = default;
  OptimizeLayoutConversionsPass(const OptimizeLayoutConversionsPass &other)
      : impl::OptimizeLayoutConversionsBase<OptimizeLayoutConversionsPass>(other),
        calibratedProfile(other.calibratedProfile) {}
  std::optional<SM90CostProfile> calibratedProfile;
  explicit OptimizeLayoutConversionsPass(LayoutBoundaryFrequencyMode mode,
                                         const SM90CostProfile *profile) {
    if (profile) calibratedProfile = *profile;
    boundaryFrequency = mode == LayoutBoundaryFrequencyMode::Legacy ? "legacy" : "static-trip-count";
  }
  void runOnOperation() override {
    ModuleOp original = getOperation();
    if (boundaryFrequency != "legacy" && boundaryFrequency != "static-trip-count") {
      original.emitError("invalid boundary-frequency: expected legacy or static-trip-count");
      signalPassFailure(); return;
    }
    auto activeCalibration = calibratedProfile;
    if (!costProfileFile.empty()) {
      if (activeCalibration) { original.emitError("duplicate supplied cost-profile"); signalPassFailure(); return; }
      auto loaded = loadSM90CostProfileFile(costProfileFile);
      if (!loaded) { original.emitError("invalid cost-profile: ") << llvm::toString(loaded.takeError()); signalPassFailure(); return; }
      activeCalibration = std::move(*loaded);
    }
    auto mode = boundaryFrequency == "legacy" ? LayoutBoundaryFrequencyMode::Legacy : LayoutBoundaryFrequencyMode::StaticTripCount;
    auto target = activeCalibration ? createSM90LayoutTarget(mode, *activeCalibration) : createSM90LayoutTarget(mode);
    if (failed(verifyMaterializedLayouts(original, *target))) {
      signalPassFailure(); return;
    }
    OwningOpRef<ModuleOp> staged(cast<ModuleOp>(original->clone()));
    bool changed;
    do {
      RewritePatternSet patterns(&getContext());
      ConvertLayoutOp::getCanonicalizationPatterns(patterns, &getContext());
      if (failed(applyPatternsGreedily(*staged, std::move(patterns))) ||
          failed(verifyMaterializedLayouts(*staged, *target))) {
        signalPassFailure(); return;
      }
      changed = false;
      SmallVector<ConvertLayoutOp> conversions;
      staged->walk([&](ConvertLayoutOp op) { conversions.push_back(op); });
      // Existing commute/hoist comparisons omit the changed producer work.
      // Until they have a complete calibrated objective check, v2 retains
      // semantic canonicalization only; do not silently optimize with v1.
      for (ConvertLayoutOp convert : conversions) {
        if (activeCalibration) break;
        if (commute(convert, *staged, *target) || hoistLoop(convert, *staged, *target)) {
          changed = true;
          break; // Rebuild handles and dominance after every accepted edit.
        }
      }
      // Each move crosses a preceding producer or one enclosing loop; nothing
      // moves downstream or grows the graph. This gives a finite structural
      // bound and a deterministic fixed point, including canonical cleanup.
    } while (changed);
    staged->walk([](Operation *op) { op->removeAttr(inferredLayout); });
    if (failed(verify(*staged)) || failed(verifyMaterializedLayouts(*staged, *target))) {
      signalPassFailure(); return;
    }
    original.getBodyRegion().takeBody(staged->getBodyRegion());
  }
};
}
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass(LayoutBoundaryFrequencyMode mode,
                                                             const SM90CostProfile *profile) {
  return std::make_unique<OptimizeLayoutConversionsPass>(mode, profile);
}
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass() {
  return std::make_unique<OptimizeLayoutConversionsPass>();
}
}
