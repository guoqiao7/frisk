#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::frisk {
bool isRematerializable(Operation *op) {
  if (!op || op->getNumRegions() || op->getNumResults() != 1 ||
      !isa<RankedTensorType>(op->getResult(0).getType())) return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effects || !effects.hasNoEffect()) return false;
  StringRef name = op->getName().getStringRef();
  if (name == "arith.constant")
    return isa_and_nonnull<DenseElementsAttr>(op->getAttr("value"));
  return op->hasTrait<OpTrait::Elementwise>() &&
      llvm::StringSwitch<bool>(name)
          .Cases("arith.addf", "arith.subf", "arith.mulf", true)
          .Cases("arith.addi", "arith.subi", "arith.muli", true)
          .Cases("arith.andi", "arith.ori", "arith.xori", true)
          .Case("arith.select", true).Default(false);
}

namespace {
bool dominates(Value value, Operation *op) {
  if (auto arg = dyn_cast<BlockArgument>(value))
    if (arg.getOwner() == op->getBlock()) return true;
  if (auto *def = value.getDefiningOp())
    if (def->getBlock() == op->getBlock())
      return def != op && def->isBeforeInBlock(op);
  if (!op->getParentRegion()) return false;
  return DominanceInfo().properlyDominates(value, op);
}

const LayoutVar *definition(const LayoutConstraintGraph &graph, Value value) {
  for (const auto &var : graph.getVariables())
    if (var.kind == LayoutKind::Distributed && var.value == value) return &var;
  return nullptr;
}

// A slice has its own values, not a reassignment of the original definitions.
// Recheck hard annotations and execution requirements on every cloned result
// and operand role, without tying the clone to original consumers' encodings.
bool legalClone(const LayoutConstraintGraph &graph, Operation *op,
                Attribute encoding, LayoutTarget &target) {
  for (const auto &var : graph.getVariables()) {
    bool belongs = (var.value && var.value.getDefiningOp() == op) ||
                   (var.use && var.use->getOwner() == op);
    if (!belongs) continue;
    if (var.kind != LayoutKind::Distributed || var.operationExecution ||
        var.instructionRole || var.reductionResult) return false;
    if (failed(target.verifyCandidate(var, encoding, op->getLoc()))) return false;
    for (const auto &c : graph.getConstraints()) {
      if (c.strength != ConstraintStrength::Hard ||
          !llvm::is_contained(c.vars, var.id)) continue;
      if (c.kind == ConstraintKind::RequireEncoding &&
          c.requiredEncoding != encoding) return false;
      if (isSupportedUnaryLayoutConstraint(c.kind) &&
          proveUnaryLayoutConstraint(graph, c, encoding).status != ProofStatus::Proven)
        return false;
      if (c.kind != ConstraintKind::RequireEncoding &&
          !isSupportedUnaryLayoutConstraint(c.kind) &&
          c.kind != ConstraintKind::SameLayout && c.kind != ConstraintKind::Convertible)
        return false;
    }
  }
  return true;
}

FailureOr<CostEstimate> rematerialize(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment,
    LayoutTarget &target, LayoutConversionEdge &edge) {
  Value source = edge.use->get();
  auto type = dyn_cast<RankedTensorType>(source.getType());
  auto encoding = dyn_cast<DistributedEncodingAttr>(edge.targetEncoding);
  Operation *at = edge.use->getOwner();
  if (!type || !encoding || !isRematerializable(source.getDefiningOp())) return failure();
  llvm::SmallPtrSet<Operation *, 8> seen;
  SmallVector<Value> leaves;
  SmallVector<Operation *> slice;
  std::function<LogicalResult(Value)> visit = [&](Value value) -> LogicalResult {
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor) return success(dominates(value, at));
    if (tensor.getShape() != type.getShape()) return failure();
    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != at->getBlock() || !isRematerializable(op)) {
      if (!dominates(value, at)) return failure();
      if (!llvm::is_contained(leaves, value)) leaves.push_back(value);
      return success();
    }
    if (!seen.insert(op).second) return success();
    if (seen.size() > 8 || !dominates(value, at)) return failure();
    // Unsupported proposals are not diagnostics about the original program.
    ScopedDiagnosticHandler quiet(op->getContext(), [](Diagnostic &) { return success(); });
    if (!legalClone(graph, op, encoding, target)) return failure();
    for (Value input : op->getOperands())
      if (failed(visit(input))) return failure();
    slice.push_back(op);
    return success();
  };
  if (failed(visit(source)) || slice.empty() || slice.back() != source.getDefiningOp())
    return failure();

  CostEstimate total;
  // All tensor leaf conversions are explicit and charged once per clone slice.
  edge.bytes = edge.synchronizationCost = 0;
  for (Value leaf : leaves) {
    const LayoutVar *var = definition(graph, leaf);
    if (!var) return failure();
    Attribute from = assignment.values.lookup(var->id);
    if (!from) return failure();
    LayoutConversionEdge boundary;
    boundary.insertionPoint = at;
    auto cost = evaluateLayoutConversionCost(*var, from, encoding, boundary, target.boundaryFrequencyMode());
    if (failed(cost)) return failure();
    total = addLayoutCosts(total, *cost);
    // Saturating addition for diagnostic traffic metadata as well as scores.
    edge.bytes = edge.bytes > UINT64_MAX - boundary.bytes ? UINT64_MAX
                                                       : edge.bytes + boundary.bytes;
    edge.synchronizationCost = edge.synchronizationCost > UINT64_MAX - boundary.synchronizationCost
        ? UINT64_MAX : edge.synchronizationCost + boundary.synchronizationCost;
  }
  for (Operation *op : slice) {
    LayoutVar synthetic;
    synthetic.kind = LayoutKind::Distributed;
    synthetic.shapedType = op->getResult(0).getType();
    auto clone = evaluateLayoutCloneCost(synthetic, encoding, at, target.boundaryFrequencyMode());
    if (failed(clone)) return failure();
    total = addLayoutCosts(total, *clone);
  }
  edge.resolution = EdgeResolutionKind::Rematerialize;
  edge.exchange = ConversionExchangeKind::Identity;
  edge.insertionPoint = at;
  edge.sharedLeader = edge.constraint;
  edge.loopDepth = getLayoutLoopDepth(at);
  auto frequency = getLayoutBoundaryFrequency(at, target.boundaryFrequencyMode());
  edge.frequencyFactor = frequency.factor;
  edge.frequencyUsesProxy = frequency.usesProxy;
  edge.frequencySaturated = frequency.saturated;
  edge.staticTripCountFrequency = target.boundaryFrequencyMode() == LayoutBoundaryFrequencyMode::StaticTripCount;
  edge.costProvenance = "pure-slice-byte-work-v1";
  edge.rematerialization = std::move(slice);
  return total;
}
} // namespace

FailureOr<LayoutConversionChoices> chooseLayoutConversions(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment,
    LayoutTarget &target, LayoutCostObserver *observer) {
  LayoutProfile::Scope profile(LayoutProfileKind::ConversionChoice);
  LayoutConversionChoices result;
  struct Group { SmallVector<LayoutConversionEdge> edges; SmallVector<CostEstimate> costs; };
  SmallVector<Group, 0> groups;
  for (const auto &c : graph.getConstraints()) {
    if (c.kind != ConstraintKind::Convertible) continue;
    Attribute from = assignment.values.lookup(c.vars[0]);
    Attribute to = assignment.values.lookup(c.vars[1]);
    if (!from && !to) continue; // Outside this complete component assignment.
    if (!from || !to) return failure();
    if (layoutEncodingsEqual(from, to)) continue;
    const auto &var = graph.getVariable(c.vars[0]);
    if (!c.use || !var.value || c.use->get() != var.value ||
        !dominates(var.value, c.use->getOwner())) return failure();
    LayoutConversionEdge edge;
    edge.use = c.use;
    edge.constraint = edge.sharedLeader = c.id;
    edge.insertionPoint = c.use->getOwner();
    edge.resolution = EdgeResolutionKind::Convert;
    auto cost = evaluateLayoutConversionCost(var, from, to, edge, target.boundaryFrequencyMode());
    if (failed(cost)) return failure();
    if (c.existingConversion) {
      result.cost = addLayoutCosts(result.cost, *cost);
      if (observer) observer->recordCost("conversion", c.id, *cost);
      continue;
    }
    auto found = llvm::find_if(groups, [&](const Group &g) {
      const auto &first = g.edges.front();
      return first.use->get() == var.value && first.targetEncoding == to &&
             first.insertionPoint->getBlock() == edge.insertionPoint->getBlock();
    });
    if (found == groups.end()) { groups.emplace_back(); found = std::prev(groups.end()); }
    found->edges.push_back(edge);
    found->costs.push_back(*cost);
  }
  for (auto &group : groups) {
    unsigned leader = 0;
    for (unsigned i = 1; i < group.edges.size(); ++i) {
      Operation *a = group.edges[i].insertionPoint, *b = group.edges[leader].insertionPoint;
      if ((a != b && a->isBeforeInBlock(b)) ||
          (a == b && graph.getConstraint(group.edges[i].constraint).stableUseKey <
                     graph.getConstraint(group.edges[leader].constraint).stableUseKey))
        leader = i;
    }
    // Compare three bounded group plans: a shared conversion, independent
    // per-use repairs, and one shared slice at the earliest consumer. The
    // group has identical source/target/block, so a slice legal at this first
    // use has the same dependencies and dominates every remaining consumer.
    // rematerialize proves all leaves and hard constraints at that position;
    // no cross-block/global CSE or unselected cloning is introduced here.
    SmallVector<LayoutConversionEdge> independent;
    SmallVector<CostEstimate> independentCharges;
    CostEstimate independentCost;
    for (unsigned i = 0; i < group.edges.size(); ++i) {
      auto edge = group.edges[i];
      auto cost = group.costs[i];
      auto clone = edge;
      auto cloneCost = rematerialize(graph, assignment, target, clone);
      if (succeeded(cloneCost) &&
          cloneCost->cost.conversionBytesAndSync < cost.cost.conversionBytesAndSync &&
          cloneCost->cost < cost.cost) {
        edge = std::move(clone);
        cost = *cloneCost;
      }
      independent.push_back(std::move(edge));
      if (observer) independentCharges.push_back(cost);
      independentCost = addLayoutCosts(independentCost, cost);
    }
    auto shared = group.edges[leader];
    auto sharedCost = group.costs[leader];
    auto sharedClone = shared;
    auto sharedCloneCost = rematerialize(graph, assignment, target, sharedClone);
    if (succeeded(sharedCloneCost) &&
        sharedCloneCost->cost.conversionBytesAndSync <
            sharedCost.cost.conversionBytesAndSync &&
        sharedCloneCost->cost < sharedCost.cost) {
      shared = std::move(sharedClone);
      sharedCost = *sharedCloneCost;
    }
    // Strict comparison retains the no-clone conversion on a clone/copy tie.
    // For a shared slice all clones and leaf conversions are charged once.
    if (independentCost.cost < sharedCost.cost) {
      result.cost = addLayoutCosts(result.cost, independentCost);
      if (observer)
        for (auto [edge, cost] : llvm::zip_equal(independent, independentCharges))
          observer->recordCost(edge.resolution == EdgeResolutionKind::Rematerialize
                                   ? "rematerialization" : "conversion",
                               edge.sharedLeader, cost);
      llvm::append_range(result.edges, independent);
    } else {
      result.cost = addLayoutCosts(result.cost, sharedCost);
      if (observer)
        observer->recordCost(shared.resolution == EdgeResolutionKind::Rematerialize
                                 ? "rematerialization" : "conversion",
                             shared.sharedLeader, sharedCost);
      for (const auto &original : group.edges) {
        auto edge = shared;
        edge.use = original.use;
        edge.constraint = original.constraint;
        result.edges.push_back(std::move(edge));
      }
    }
  }
  llvm::sort(result.edges, [&](const auto &a, const auto &b) {
    return graph.getConstraint(a.constraint).stableUseKey <
           graph.getConstraint(b.constraint).stableUseKey;
  });
  return result;
}

bool sameLayoutConversionPlan(const LayoutConversionEdge &a,
                              const LayoutConversionEdge &b) {
  return a.use == b.use && a.constraint == b.constraint &&
      a.sourceEncoding == b.sourceEncoding && a.targetEncoding == b.targetEncoding &&
      a.resolution == b.resolution && a.bytes == b.bytes &&
      a.synchronizationCost == b.synchronizationCost && a.exchange == b.exchange &&
      a.frequencyFactor == b.frequencyFactor &&
      a.frequencyUsesProxy == b.frequencyUsesProxy &&
      a.frequencySaturated == b.frequencySaturated &&
      a.staticTripCountFrequency == b.staticTripCountFrequency &&
      a.loopDepth == b.loopDepth && a.costProvenance == b.costProvenance &&
      a.insertionPoint == b.insertionPoint && a.sharedLeader == b.sharedLeader &&
      a.rematerialization == b.rematerialization;
}

LogicalResult verifyLayoutConversionChoices(const LayoutConstraintGraph &graph,
    const LayoutSolution &solution, LayoutTarget &target, Location loc) {
  CandidateAssignment assignment;
  assignment.values = solution.assignments;
  auto expected = chooseLayoutConversions(graph, assignment, target);
  if (failed(expected) || expected->edges.size() != solution.conversions.size() ||
      !llvm::equal(expected->edges, solution.conversions, sameLayoutConversionPlan))
    return emitError(loc) << "invalid layout boundary placement, sharing, cost or rematerialization plan";
  return success();
}
} // namespace mlir::frisk
