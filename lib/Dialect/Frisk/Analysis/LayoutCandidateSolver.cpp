#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"
#include "mlir/IR/Diagnostics.h"
#include <map>

namespace mlir::frisk {
namespace {
Location getVariableLoc(const LayoutVar &var) {
  return var.anchor ? var.anchor->getLoc()
                    : UnknownLoc::get(var.shapedType.getContext());
}

SmallVector<SmallVector<LayoutVarID>>
getHardConstraintComponents(const LayoutConstraintGraph &graph) {
  SmallVector<SmallVector<LayoutVarID>> adjacency(graph.getVariables().size());
  for (const LayoutConstraint &constraint : graph.getConstraints()) {
    if (constraint.strength != ConstraintStrength::Hard ||
        constraint.vars.size() < 2)
      continue;
    for (size_t index = 1; index < constraint.vars.size(); ++index) {
      adjacency[constraint.vars.front()].push_back(constraint.vars[index]);
      adjacency[constraint.vars[index]].push_back(constraint.vars.front());
    }
  }

  SmallVector<SmallVector<LayoutVarID>> components;
  SmallVector<bool> visited(graph.getVariables().size());
  for (LayoutVarID start = 0; start < graph.getVariables().size(); ++start) {
    if (visited[start])
      continue;
    SmallVector<LayoutVarID> component;
    SmallVector<LayoutVarID> worklist = {start};
    visited[start] = true;
    while (!worklist.empty()) {
      LayoutVarID id = worklist.pop_back_val();
      component.push_back(id);
      llvm::sort(adjacency[id]);
      for (LayoutVarID next : adjacency[id]) {
        if (!visited[next]) {
          visited[next] = true;
          worklist.push_back(next);
        }
      }
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }
  return components;
}

} // namespace

FailureOr<LayoutSolution>
solveLayoutGraph(LayoutConstraintGraph &graph, LayoutTarget &target,
                 SolverOptions options) {
  const bool validOptions = options.beamWidth && options.maxExpandedStates &&
                            options.exactCombinationLimit;
  if (graph.getVariables().empty()) {
    if (!validOptions || !graph.getConstraints().empty() ||
        !graph.getRegionEdges().empty()) {
      llvm::errs() << "invalid layout solver options or empty graph invariants\n";
      return failure();
    }
    LayoutSolution empty;
    return empty;
  }
  if (!graph.getVariables().front().shapedType &&
      !graph.getVariables().front().anchor) {
    llvm::errs() << "invalid layout graph: variable has no shaped type\n";
    return failure();
  }
  Location loc = getVariableLoc(graph.getVariables().front());
  if (failed(graph.verifyInvariants(loc))) return failure();
  if (!validOptions) {
    emitError(loc) << "invalid layout solver options: budgets must be positive";
    return failure();
  }
  // Public callers may supply a valid but not yet canonical graph. Avoid
  // re-finalizing production graphs: that would erase propagation statistics.
  if (!graph.isFinalized() && failed(graph.finalize(loc))) return failure();
  for (const auto &c : graph.getConstraints())
    if (c.strength == ConstraintStrength::Hard &&
        (!isSupportedHardLayoutConstraint(c.kind) ||
         (c.kind == ConstraintKind::InstructionContract && !c.instruction) ||
         (c.kind == ConstraintKind::ReductionLayout && !c.reduction))) {
      emitError(loc) << "layout solver does not support hard constraint '"
                     << stringifyConstraintKind(c.kind) << "'";
      return failure();
    }

  LayoutSolution solution;
  auto product = [](uint64_t value, uint64_t factor, bool &overflow) {
    if (factor && value > UINT64_MAX / factor) {
      overflow = true;
      return UINT64_MAX;
    }
    return value * factor;
  };
  for (const auto &component : getHardConstraintComponents(graph)) {
    LayoutSolution::ComponentStatistics stats;
    stats.root = component.front();
    stats.beamWidth = options.beamWidth;
    SmallVector<LayoutConstraintID> constraints;
    SmallVector<unsigned> degree(graph.getVariables().size(), 0);
    for (const auto &c : graph.getConstraints())
      if (c.strength == ConstraintStrength::Hard && !c.vars.empty() &&
          llvm::is_contained(component, c.vars.front())) {
        constraints.push_back(c.id);
        for (auto id : c.vars) ++degree[id];
      }
    // Domains are frozen before search; support tables use Attributes, not
    // mutable candidate indices. Hash iteration order never selects a winner.
    for (auto id : component) {
      auto &var = graph.getVariable(id);
      stats.rawCombinations = product(stats.rawCombinations,
                                      var.candidates.size(), stats.rawProductOverflow);
      llvm::sort(var.candidates, [](const auto &a, const auto &b) {
        return layoutCandidateKey(a.value) < layoutCandidateKey(b.value);
      });
      SmallVector<LayoutCandidate> kept;
      for (const auto &candidate : var.candidates) {
        if (llvm::any_of(kept, [&](const auto &c) {
              return c.value == candidate.value;
            })) {
          ++stats.deduplicated;
          continue;
        }
        std::string reason;
        bool valid;
        {
          ScopedDiagnosticHandler capture(var.shapedType.getContext(),
              [&](Diagnostic &d) {
                llvm::raw_string_ostream stream(reason);
                d.print(stream);
                return success();
              });
          valid = succeeded(target.verifyCandidate(var, candidate.value,
                                                    getVariableLoc(var))) &&
                  succeeded(verifyLayoutStorageCapacity(graph, var, candidate.value));
        }
        LayoutConstraintID rejected = UINT32_MAX;
        DenseMap<LayoutVarID, Attribute> single;
        single[id] = candidate.value;
        if (valid)
          for (auto cid : constraints) {
            const auto &c = graph.getConstraint(cid);
            if (!llvm::is_contained(c.vars, id)) continue;
            if (!satisfiesLayoutConstraint(graph, c, single, false)) {
              valid = false;
              rejected = cid;
              reason = "hard constraint not proven: ";
              reason += stringifyConstraintKind(c.kind).str();
              if (isSupportedUnaryLayoutConstraint(c.kind)) {
                auto proof = proveUnaryLayoutConstraint(graph, c, candidate.value);
                reason += ": " + proof.reason;
              }
              break;
            }
          }
        if (valid) kept.push_back(candidate);
        else solution.rejections.push_back(
            {id, layoutCandidateKey(candidate.value), reason, rejected});
      }
      var.candidates = std::move(kept);
      if (var.candidates.empty()) {
        emitError(getVariableLoc(var)) << "no proven layout candidate for "
            << var.stableName << " (hard constraints not proven): "
            << (solution.rejections.empty() ? "empty input domain"
                                           : solution.rejections.back().reason);
        return failure();
      }
      stats.domains.push_back(var.candidates.size());
      stats.combinations = product(stats.combinations, var.candidates.size(),
                                    stats.productOverflow);
    }
    stats.usedBeam = stats.combinations > options.exactCombinationLimit ||
                     stats.productOverflow;
    struct State {
      CandidateAssignment assignment;
      SmallVector<unsigned> key;
      CostEstimate bound;
    };
    // Cache complete relation checks by immutable canonical-domain indices.
    // Partial tuple/pair support checks remain attribute keyed.
    std::map<std::vector<unsigned>, bool> relationCache;
    auto viable = [&](const State &s) {
      for (auto cid : constraints) {
        const auto &c = graph.getConstraint(cid);
        std::vector<unsigned> key{cid};
        for (auto id : c.vars) {
          auto it = llvm::find(component, id);
          key.push_back(s.key[it - component.begin()]);
        }
        auto found = relationCache.find(key);
        bool ok;
        if (found != relationCache.end()) ok = found->second;
        else {
          ok = satisfiesLayoutConstraint(graph, c, s.assignment.values, false);
          relationCache.emplace(std::move(key), ok);
        }
        if (!ok) return false;
      }
      return true;
    };
    auto evaluate = [&](State &s, bool complete,
                        SmallVectorImpl<LayoutConversionEdge> *edges = nullptr) {
      auto cost = complete ? target.evaluate(graph, s.assignment)
                           : target.lowerBound(graph, s.assignment);
      if (failed(cost)) return failure();
      for (auto cid : constraints) {
        const auto &c = graph.getConstraint(cid);
        if (c.kind != ConstraintKind::Convertible) continue;
        Attribute from = s.assignment.values.lookup(c.vars[0]);
        Attribute to = s.assignment.values.lookup(c.vars[1]);
        if (!from || !to || layoutEncodingsEqual(from, to)) continue;
        LayoutConversionEdge edge;
        edge.use = c.use;
        edge.sourceEncoding = from;
        edge.targetEncoding = to;
        edge.resolution = EdgeResolutionKind::Convert;
        edge.constraint = cid;
        auto conversion = evaluateLayoutConversionCost(
            graph.getVariable(c.vars[0]), from, to, edge);
        if (failed(conversion)) return failure();
        *cost = addLayoutCosts(*cost, *conversion);
        if (edges && !c.existingConversion) edges->push_back(edge);
      }
      s.bound = *cost;
      return success();
    };
    auto less = [](const State &a, const State &b) {
      if (a.bound.cost < b.bound.cost) return true;
      if (b.bound.cost < a.bound.cost) return false;
      return std::lexicographical_compare(a.key.begin(), a.key.end(),
                                          b.key.begin(), b.key.end());
    };
    State initial;
    initial.key.assign(component.size(), UINT32_MAX);
    SmallVector<State> frontier;
    frontier.push_back(std::move(initial));
    for (size_t depth = 0; depth < component.size(); ++depth) {
      SmallVector<State> next;
      for (const auto &state : frontier) {
        size_t chosen = component.size();
        SmallVector<State> children;
        // MRV counts only viable extensions; every attempted extension,
        // including MRV probes and hard rejections, consumes the budget.
        for (size_t index = 0; index < component.size(); ++index) {
          if (state.key[index] != UINT32_MAX) continue;
          auto id = component[index];
          SmallVector<State> probes;
          const auto &domain = graph.getVariable(id).candidates;
          for (unsigned ci = 0; ci < domain.size(); ++ci) {
            if (stats.expanded == options.maxExpandedStates) {
              emitError(getVariableLoc(graph.getVariable(stats.root)))
                  << "search-budget-exceeded: expanded=" << stats.expanded
                  << ", limit=" << options.maxExpandedStates;
              return failure();
            }
            ++stats.expanded;
            State child = state;
            child.key[index] = ci;
            child.assignment.values[id] = domain[ci].value;
            if (viable(child)) probes.push_back(std::move(child));
            else ++stats.hardRejected;
          }
          if (chosen == component.size() || probes.size() < children.size() ||
              (probes.size() == children.size() &&
               degree[id] > degree[component[chosen]])) {
            chosen = index;
            children = std::move(probes);
          }
          if (children.empty()) break;
        }
        for (auto &child : children) {
          if (failed(evaluate(child, depth + 1 == component.size()))) {
            emitError(loc) << "layout cost evaluation failed; not a free solution";
            return failure();
          }
          next.push_back(std::move(child));
        }
      }
      llvm::sort(next, less);
      if (stats.usedBeam && next.size() > options.beamWidth) {
        stats.beamDiscarded += next.size() - options.beamWidth;
        next.resize(options.beamWidth);
      }
      frontier = std::move(next);
      if (frontier.empty()) break;
    }
    stats.exhaustive = stats.beamDiscarded == 0;
    if (frontier.empty()) {
      emitError(getVariableLoc(graph.getVariable(stats.root)))
          << (stats.exhaustive ? "no feasible proven layout assignment"
                              : "search-incomplete: beam discarded alternatives");
      return failure();
    }
    auto &best = frontier.front();
    if (failed(evaluate(best, true, &solution.conversions))) return failure();
    stats.cost = best.bound;
    solution.cost = addLayoutCosts(solution.cost, best.bound);
    for (auto id : component)
      solution.assignments[id] = best.assignment.values.lookup(id);
    solution.components.push_back(std::move(stats));
  }
  llvm::sort(solution.conversions, [&](const auto &a, const auto &b) {
    return graph.getConstraint(a.constraint).stableUseKey <
           graph.getConstraint(b.constraint).stableUseKey;
  });
  for (const auto &c : graph.getConstraints()) {
    if (c.kind == ConstraintKind::ReductionLayout) {
      const auto *pair = findReductionSupport(graph, c, solution.assignments);
      if (!pair) return failure();
      solution.reductionBindings[c.id] = pair->binding;
    }
    if (c.kind == ConstraintKind::InstructionContract) {
      const auto *tuple = findInstructionSupport(graph, c, solution.assignments);
      if (!tuple) return failure();
      solution.instructionBindings[c.id] = tuple->binding;
    }
  }
  return solution;
}

void printLayoutSolutionStatistics(const LayoutConstraintGraph &graph,
                                   const LayoutSolution &solution,
                                   LayoutTarget &target, llvm::raw_ostream &os) {
  os << "cost coverage: " << target.costCoverage() << '\n';
  for (const auto &s : solution.components) {
    os << "solver component " << graph.getVariable(s.root).stableName
       << ": mode=" << (s.usedBeam ? "beam" : "exact")
       << " beam-width=" << s.beamWidth << " variables=" << s.domains.size()
       << " raw=" << s.rawCombinations << " post-prune=" << s.combinations
       << " raw-overflow=" << s.rawProductOverflow
       << " product-overflow=" << s.productOverflow << " domains=[";
    llvm::interleaveComma(s.domains, os);
    const auto &c = s.cost.cost;
    os << "] expanded=" << s.expanded << " hard-rejected=" << s.hardRejected
       << " beam-discarded=" << s.beamDiscarded
       << " deduplicated=" << s.deduplicated
       << " exhaustive=" << s.exhaustive << " cost=["
       << c.instructionPathAndWork << ',' << c.memoryTransactions << ','
       << c.bankConflictDegree << ',' << c.conversionBytesAndSync << ','
       << c.spillRiskAndRegisters << ',' << c.sharedBytesAndOccupancy << ','
       << c.replication << ',' << c.codeSize << ',' << c.deterministicTieBreak
       << "] saturated=" << s.cost.saturated << '\n';
  }
  for (const auto &r : solution.rejections)
    os << "candidate-rejected " << graph.getVariable(r.variable).stableName
       << " key=" << r.candidate << " constraint=" << r.constraint
       << " reason=" << r.reason << '\n';
  const auto &c = solution.cost.cost;
  os << "solver total cost=[" << c.instructionPathAndWork << ','
     << c.memoryTransactions << ',' << c.bankConflictDegree << ','
     << c.conversionBytesAndSync << ',' << c.spillRiskAndRegisters << ','
     << c.sharedBytesAndOccupancy << ',' << c.replication << ',' << c.codeSize
     << ',' << c.deterministicTieBreak << "] saturated=" << solution.cost.saturated
     << '\n';
}

} // namespace mlir::frisk
