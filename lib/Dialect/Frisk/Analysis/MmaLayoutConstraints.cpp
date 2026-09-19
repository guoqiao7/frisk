#include "Dialect/Frisk/Analysis/MmaLayoutConstraints.h"
#include "Dialect/Frisk/IR/FriskOps.h"

namespace mlir::frisk {
LogicalResult collectMmaLayoutConstraints(Operation *root,
    LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder) {
  auto result = root->walk([&](MmaOp mma) -> WalkResult {
    if (failed(mma.verify())) return WalkResult::interrupt();
    int64_t threads = 128;
    if (auto attr = mma->getAttr("frisk.execution_threads")) {
      auto count = dyn_cast<IntegerAttr>(attr);
      if (!count || !count.getType().isSignlessInteger(64)) {
        mma.emitError("sm90-mma-thread-group: execution_threads must be signless i64");
        return WalkResult::interrupt();
      }
      threads = count.getInt();
    }
    if (auto parallel = mma->getParentOfType<ParallelOp>()) {
      if (mma->hasAttr("frisk.execution_threads") && threads != parallel.getThreads()) {
        mma.emitError("sm90-mma-thread-group: execution_threads conflicts with enclosing parallel");
        return WalkResult::interrupt();
      }
      threads = parallel.getThreads();
    }
    if (threads != 128 && threads != 256 && threads != 512 && threads != 1024) {
      mma.emitError("sm90-mma-thread-group: expected 128, 256, 512 or 1024 threads");
      return WalkResult::interrupt();
    }
    SmallVector<LayoutVarID> roles;
    for (unsigned index = 0; index < 3; ++index) {
      auto &use = mma->getOpOperand(index);
      LayoutVarID id;
      if (isa<RankedTensorType>(use.get().getType())) {
        id = builder.getOrCreateDistributedUse(use);
      } else {
        auto view = use.get().getDefiningOp<LayoutViewOp>();
        auto memory = getFriskMemorySpace(cast<MemRefType>(use.get().getType()));
        if (!view || !memory || *memory != attr::MemorySpace::Shared) {
          mma.emitError("sm90-mma-descriptor: shared operands must be direct layout_view results; Local MemRef is not a fragment");
          return WalkResult::interrupt();
        }
        id = builder.getOrCreateStorageVar(use.get());
      }
      roles.push_back(id);
    }
    roles.push_back(builder.getOrCreateDistributedVar(mma.getResult()));
    for (auto id : roles) {
      auto &var = graph.getVariable(id);
      var.instructionRole = true;
      if (var.kind == LayoutKind::Distributed) var.requiredThreads = threads;
    }
    (void)builder.same(roles[2], roles[3], mma, "mma-accumulator-slot");
    auto id = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, roles, mma, "mma-joint-contract",
        "A/B/init/result must satisfy the same instruction contract");
    graph.getConstraints()[id].instruction = InstructionLayoutContract{
        mma, mma->getAttr("frisk.mma_contract"), {}};
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}
} // namespace mlir::frisk
