#include "Dialect/Frisk/Analysis/OperationLayoutConstraints.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::frisk {
namespace {
bool isWholeTileStaticCopy(CopyOp copy) {
  auto src = copy.getSrcMemRefType(), dst = copy.getDstMemRefType();
  return src.hasStaticShape() && dst.hasStaticShape() &&
         src.getShape() == copy.getSrcExtents() &&
         dst.getShape() == copy.getDstExtents() && src.getShape() == dst.getShape() &&
         src.getElementType() == dst.getElementType() &&
         copy.getSrcIndices().empty() && copy.getDstIndices().empty() &&
         copy.getSrcMap().getNumInputs() == 0 && copy.getDstMap().getNumInputs() == 0 &&
         copy.getSrcMap().getNumResults() == 0 && copy.getDstMap().getNumResults() == 0;
}
LogicalResult checkExecutionShape(Operation *op, MemRefType type) {
  if (!type.hasStaticShape() || type.getRank() == 0 ||
      !type.getElementType().isIntOrFloat() ||
      llvm::any_of(type.getShape(), [](int64_t size) {
        return size <= 1 || !llvm::isPowerOf2_64(size);
      }))
    return op->emitError("operation execution layout requires static power-of-two tile extents greater than one");
  return success();
}
} // namespace

LogicalResult collectOperationLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder) {
  auto result = root->walk([&](Operation *op) -> WalkResult {
    if (auto parallel = dyn_cast<ParallelOp>(op)) {
      if (failed(parallel.verify())) return WalkResult::interrupt();
    }
    auto copy = dyn_cast<CopyOp>(op);
    auto fill = dyn_cast<FillOp>(op);
    if (!copy && !fill) return WalkResult::advance();
    if (failed(verifyOperationExecutionAttributes(op))) return WalkResult::interrupt();
    if (copy && !isWholeTileStaticCopy(copy)) {
      copy.emitOpError("unsupported storage layout inference for non-whole-tile or dynamic copy");
      return WalkResult::interrupt();
    }
    Value dst = copy ? copy.getDst() : fill.getMemref();
    Value src = copy ? copy.getSrc() : Value();
    if (!dst.getDefiningOp<LayoutViewOp>() || (copy && !src.getDefiningOp<LayoutViewOp>())) {
      op->emitError(copy ? "M2 storage layout inference requires whole-tile copy operands to be layout_view results"
                         : "fill layout inference requires a layout_view target");
      return WalkResult::interrupt();
    }
    auto type = cast<MemRefType>(dst.getType());
    if (failed(checkExecutionShape(op, type))) return WalkResult::interrupt();
    OperationExecutionBinding binding;
    if (auto threads = op->getAttrOfType<IntegerAttr>("frisk.execution_threads"))
      binding.threads = threads.getInt();
    if (auto parallel = op->getParentOfType<ParallelOp>()) {
      if (op->hasAttr("frisk.execution_threads") && binding.threads != parallel.getThreads()) {
        op->emitError("execution_threads conflicts with enclosing parallel thread count");
        return WalkResult::interrupt();
      }
      binding.threads = parallel.getThreads();
    }
    if (auto width = op->getAttrOfType<IntegerAttr>("frisk.vector_bytes"))
      binding.vectorBytes = width.getInt();
    if (auto policy = op->getAttrOfType<StringAttr>("frisk.writer_policy"))
      binding.writerPolicy = policy.getValue().str();
    auto execution = builder.createOperationExecutionVar(op,
        RankedTensorType::get(type.getShape(), type.getElementType()), binding);
    if (auto declared = op->getAttr("frisk.execution_layout"))
      if (failed(builder.require(execution, declared, op, "operation-execution-binding")))
        return WalkResult::interrupt();
    auto target = builder.getOrCreateStorageVar(dst);
    (void)builder.storageAccess(execution, target, AccessKind::Write, op,
                                copy ? "copy-write" : "fill-write");
    graph.addConstraint(ConstraintKind::Ownership, ConstraintStrength::Hard,
        {execution}, op, "unique-operation-writer",
        "each live logical point has exactly one elected writer");
    if (copy) {
      auto source = builder.getOrCreateStorageVar(src);
      (void)builder.storageAccess(execution, source, AccessKind::Read, op, "copy-read");
      graph.addConstraint(ConstraintKind::CopyAccess, ConstraintStrength::Hard,
          {source, target}, op, "whole-tile-copy",
          "logical pointwise copy; known-root overlap must be disjoint or identity; distinct arguments require nonoverlap precondition");
    }
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

LogicalResult collectParallelResourceConstraints(LayoutConstraintGraph &graph) {
  for (auto &var : graph.getVariables()) {
    if (var.kind != LayoutKind::Distributed) continue;
    auto parallel = var.anchor ? var.anchor->getParentOfType<ParallelOp>() : ParallelOp();
    int64_t threads = var.requiredThreads ? var.requiredThreads
                      : var.operationExecution ? var.operationExecution->threads
                      : parallel ? parallel.getThreads() : 0;
    if (!threads) continue;
    if (!isSupportedExecutionThreadCount(threads))
      return var.anchor->emitError("unsupported execution thread count");
    var.requiredThreads = threads;
    auto id = graph.addConstraint(ConstraintKind::ResourceLimit, ConstraintStrength::Hard,
        {var.id}, var.anchor, "execution-thread-topology",
        "lane * warp * warp_group must match " + std::to_string(threads) + " threads in one CTA");
    graph.getConstraints()[id].requiredThreads = threads;
  }
  return success();
}
} // namespace mlir::frisk
