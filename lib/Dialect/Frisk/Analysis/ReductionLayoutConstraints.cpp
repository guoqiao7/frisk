#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::frisk {
namespace {
FailureOr<int64_t> inferSourceThreads(const LayoutConstraintGraph &graph,
                                      LayoutVarID start, Operation *op) {
  SmallVector<LayoutVarID> pending{start};
  llvm::SmallSet<LayoutVarID, 16> visited;
  llvm::SmallSet<int64_t, 4> counts;
  while (!pending.empty()) {
    auto id = pending.pop_back_val();
    if (!visited.insert(id).second) continue;
    const auto &var = graph.getVariable(id);
    if (var.requiredThreads) counts.insert(var.requiredThreads);
    auto type = dyn_cast<RankedTensorType>(var.shapedType);
    auto enc = type ? dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding())
                    : DistributedEncodingAttr();
    if (enc) {
      auto t = enc.getTopology();
      if (t.size() != 5 || t[1] != 32 || t[4] != 1 || t[2] <= 0 ||
          t[3] <= 0 || t[2] > 32 || t[3] > 32 || t[2] * t[3] > 32) {
        op->emitError("reduce-thread-scope: unsupported source topology");
        return failure();
      }
      counts.insert(t[1] * t[2] * t[3]);
    }
    // A known endpoint defines this value's execution environment. Its
    // producer may legitimately use another environment across a conversion.
    // SameLayout peers remain authoritative independent sources and must all
    // be checked, including when this endpoint is already known.
    for (const auto &c : graph.getConstraints()) {
      if (c.kind == ConstraintKind::SameLayout && llvm::is_contained(c.vars, id)) {
        llvm::append_range(pending, c.vars);
      } else if (!var.requiredThreads && !enc &&
                 (c.kind == ConstraintKind::Convertible ||
                  c.kind == ConstraintKind::TransformLayout ||
                  c.kind == ConstraintKind::ReductionLayout) &&
                 c.vars.size() == 2 && c.vars[1] == id) {
        pending.push_back(c.vars[0]);
      }
    }
  }
  if (counts.size() > 1) {
    op->emitError("reduce-thread-scope: conflicting producer thread requirements");
    return failure();
  }
  return counts.empty() ? 32 : *counts.begin();
}
} // namespace

LogicalResult collectReductionLayoutConstraints(Operation *root,
    LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder) {
  auto result = root->walk([&](ReduceTensorOp reduce) -> WalkResult {
    if (failed(reduce.verify())) return WalkResult::interrupt();
    auto source = builder.getOrCreateDistributedVar(reduce.getSource());
    int64_t threads = 0;
    if (auto explicitCount = reduce->getAttrOfType<IntegerAttr>("frisk.execution_threads"))
      threads = explicitCount.getInt();
    if (auto parallel = reduce->getParentOfType<ParallelOp>()) {
      if (threads && threads != parallel.getThreads()) {
        reduce.emitError("reduce-thread-scope: execution_threads conflicts with enclosing parallel");
        return WalkResult::interrupt();
      }
      threads = parallel.getThreads();
    }
    if (!threads) {
      auto inferred = inferSourceThreads(graph, source, reduce);
      if (failed(inferred)) return WalkResult::interrupt();
      threads = *inferred;
    }
    if (threads < 32 || threads > 1024 || (threads & (threads - 1))) {
      reduce.emitError("reduce-thread-scope: expected 32/64/128/256/512/1024 threads");
      return WalkResult::interrupt();
    }
    // A private, unencoded producer may adopt its sole consumer's environment.
    // Otherwise its independent default-32 family and the requested family can
    // unnecessarily overflow the finite domain before any conversion is chosen.
    // Shared or explicitly encoded producers keep their own layout requirements.
    auto &producer = graph.getVariable(source);
    auto producerType = cast<RankedTensorType>(producer.shapedType);
    auto producerParallel = producer.anchor
        ? producer.anchor->getParentOfType<ParallelOp>() : ParallelOp();
    if (!producer.requiredThreads && !producerType.getEncoding() &&
        reduce.getSource().hasOneUse() &&
        (!producerParallel || producerParallel.getThreads() == threads))
      producer.requiredThreads = threads;
    auto src = builder.getOrCreateDistributedUse(reduce->getOpOperand(0));
    auto dst = builder.getOrCreateDistributedVar(reduce.getResult());
    graph.getVariable(src).requiredThreads = threads;
    graph.getVariable(dst).requiredThreads = threads;
    graph.getVariable(dst).reductionResult = true;
    auto id = graph.addConstraint(ConstraintKind::ReductionLayout,
        ConstraintStrength::Hard, {src, dst}, reduce, "reduce-projection-contributors",
        "natural coordinate projection and exactly-once complete reduction");
    graph.getConstraints()[id].reduction = ReductionLayoutContract{
        reduce, reduce.getDimAttr().getInt(), reduce->getAttr("frisk.reduction_contract"), {}};
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

const ReductionLayoutPair *findReductionSupport(
    const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &assignment) {
  if (!constraint.reduction || constraint.vars.size() != 2) return nullptr;
  auto src = constraint.vars[0], dst = constraint.vars[1];
  for (const auto &pair : constraint.reduction->pairs) {
    if (!pair.binding ||
        (constraint.reduction->binding && pair.binding != constraint.reduction->binding) ||
        (assignment.count(src) && assignment.lookup(src) != pair.sourceEncoding) ||
        (assignment.count(dst) && assignment.lookup(dst) != pair.resultEncoding)) continue;
    auto inDomain = [&](LayoutVarID id, Attribute encoding) {
      return llvm::any_of(graph.getVariable(id).candidates,
                         [&](const auto &c) { return c.value == encoding; });
    };
    if (inDomain(src, pair.sourceEncoding) && inDomain(dst, pair.resultEncoding))
      return &pair;
  }
  return nullptr;
}

LogicalResult prepareReductionPairs(LayoutConstraintGraph &graph,
                                    LayoutTarget &target, Location loc) {
  for (auto &constraint : graph.getConstraints()) {
    if (constraint.kind != ConstraintKind::ReductionLayout) continue;
    if (!constraint.reduction || constraint.vars.size() != 2)
      return emitError(loc) << "invalid reduction relation";
    auto &reduction = *constraint.reduction;
    reduction.pairs.clear();
    const auto &src = graph.getVariable(constraint.vars[0]);
    const auto &dst = graph.getVariable(constraint.vars[1]);
    if (src.candidates.size() > 4 || dst.candidates.size() > 4)
      return emitError(loc) << "bootstrap layout solver limit exceeded: reduction candidate domain is too large";
    std::optional<LayoutProof> firstFailure;
    {
      ScopedDiagnosticHandler quiet(loc.getContext(), [](Diagnostic &) { return success(); });
      for (const auto &a : src.candidates) {
        for (const auto &b : dst.candidates) {
          ++graph.getCandidatePreparationStatistics().reductionCombinations;
          auto binding = reduction.binding ? FailureOr<Attribute>(reduction.binding)
              : target.buildReductionContract(graph, constraint, a.value, b.value);
          if (failed(binding)) continue;
          auto proof = target.verifyReductionContract(graph, constraint, a.value, b.value, *binding);
          if (proof.status == ProofStatus::Proven)
            reduction.pairs.push_back({a.value, b.value, *binding});
          else if (!firstFailure) firstFailure = std::move(proof);
        }
      }
    }
    if (reduction.pairs.empty()) {
      auto error = reduction.source->emitError("reduce-contract: no proven pair in frozen candidate domains");
      if (firstFailure) error << "; " << firstFailure->reason;
      else error << "; unknown proof: no bounded supported plan";
      return failure();
    }
  }
  return success();
}
} // namespace mlir::frisk
