#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/DenseSet.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
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
                 SolverOptions options, LayoutSolveFailure *why,
                 LayoutAttemptStatistics *statistics) {
  LayoutSolveFailure localFailure;
  if (!why) why = &localFailure;
  LayoutSolution solution;
  bool solved = false;
  auto start = LayoutClock::now();
  if (statistics && !statistics->prepared) prepareLayoutStatistics(graph, *statistics);
  auto finish = llvm::make_scope_exit([&] {
    if (!statistics) return;
    statistics->statistics.solveTime += std::chrono::nanoseconds(layoutElapsedNS(start));
    uint64_t remaining = 0;
    for (const auto &var : graph.getVariables()) {
      llvm::DenseSet<Attribute> unique;
      for (const auto &candidate : var.candidates) unique.insert(candidate.value);
      remaining += unique.size();
    }
    statistics->statistics.prunedCandidates = statistics->statistics.preparedCandidates - remaining;
    if (!solved) finishLayoutStatistics(graph, *why, nullptr, *statistics);
  });
  *why = {LayoutSolveFailureKind::InvalidInput, UINT32_MAX, UINT32_MAX,
          "invalid graph or solver options"};
  const bool validOptions = options.beamWidth && options.maxExpandedStates &&
                            options.exactCombinationLimit;
  if (graph.getVariables().empty()) {
    if (!validOptions || !graph.getConstraints().empty() ||
        !graph.getRegionEdges().empty()) {
      llvm::errs() << "invalid layout solver options or empty graph invariants\n";
      return failure();
    }
    *why = {};
    if (statistics) finishLayoutStatistics(graph, *why, &solution, *statistics);
    solved = true;
    return solution;
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
      *why = {LayoutSolveFailureKind::Unsupported, UINT32_MAX, c.id,
              "unsupported hard constraint"};
      return failure();
    }

  LayoutSolveFailure emptyFailure;
  auto product = [](uint64_t value, uint64_t factor, bool &overflow) {
    if (factor && value > UINT64_MAX / factor) {
      overflow = true;
      return UINT64_MAX;
    }
    return value * factor;
  };
  for (auto component : getHardConstraintComponents(graph)) {
    bool emptyComponent = false;
    LayoutSolution::ComponentStatistics stats;
    stats.root = component.front();
    stats.beamWidth = options.beamWidth;
    auto finishComponent = llvm::make_scope_exit([&] {
      if (!statistics) return;
      auto &aggregate = statistics->statistics;
      ++aggregate.components;
      aggregate.expandedStates += stats.expanded;
      aggregate.peakFrontierStates = std::max(aggregate.peakFrontierStates, stats.peakFrontier);
      aggregate.peakComponentCombinations = std::max(aggregate.peakComponentCombinations, stats.combinations);
      aggregate.componentProductOverflow |= stats.productOverflow;
      for (auto domain : stats.domains)
        aggregate.peakVariableDomain = std::max<uint64_t>(aggregate.peakVariableDomain, domain);
      statistics->mrvProbes += stats.mrvProbes;
      statistics->expandedNodes += stats.expandedNodes;
      llvm::json::Array domains;
      for (auto domain : stats.domains) domains.push_back(domain);
      statistics->components.push_back(llvm::json::Object{
        {"root", graph.getVariable(stats.root).stableName},
        {"combinations", stats.productOverflow ? llvm::json::Value(nullptr) : llvm::json::Value(stats.combinations)},
        {"productOverflow", stats.productOverflow}, {"domains", std::move(domains)},
        {"mode", stats.usedBeam ? "beam" : "exact"}, {"beamWidth", stats.beamWidth},
        {"expanded", stats.expanded}, {"peakFrontier", stats.peakFrontier},
        {"mrvProbes", stats.mrvProbes}, {"expandedNodes", stats.expandedNodes},
        {"exhaustive", stats.exhaustive}});
    });
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
      LayoutSolveFailure rejection{LayoutSolveFailureKind::EmptyDomain, id,
                                  UINT32_MAX, "empty input domain"};
      if (var.candidatePreparationUnsupported)
        rejection = {LayoutSolveFailureKind::Unsupported, id, UINT32_MAX,
                     "candidate preparation rejected an unsupported target encoding"};
      // A missing mandatory seed is not permission to invent another encoding.
      for (auto cid : constraints) {
        const auto &c = graph.getConstraint(cid);
        if (c.kind == ConstraintKind::RequireEncoding &&
            llvm::is_contained(c.vars, id))
          rejection = {LayoutSolveFailureKind::HardConflict, id, cid,
                       "mandatory encoding has no legal candidate"};
      }
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
        // A failed target/capacity decoder has no three-valued proof here.
        // Treat it as unsupported, never as proven emptiness for relaxation.
        LayoutSolveFailureKind rejectedKind = valid
            ? LayoutSolveFailureKind::EmptyDomain
            : LayoutSolveFailureKind::Unsupported;
        DenseMap<LayoutVarID, Attribute> single;
        single[id] = candidate.value;
        if (valid)
          for (auto cid : constraints) {
            const auto &c = graph.getConstraint(cid);
            if (!llvm::is_contained(c.vars, id)) continue;
            if (!satisfiesLayoutConstraint(graph, c, single, false)) {
              valid = false;
              rejected = cid;
              rejectedKind = LayoutSolveFailureKind::HardConflict;
              reason = "hard constraint not proven: ";
              reason += stringifyConstraintKind(c.kind).str();
              if (isSupportedUnaryLayoutConstraint(c.kind)) {
                auto proof = proveUnaryLayoutConstraint(graph, c, candidate.value);
                reason += ": " + proof.reason;
                rejectedKind = proof.status == ProofStatus::Unknown
                    ? LayoutSolveFailureKind::Unsupported
                    : LayoutSolveFailureKind::EmptyDomain;
              }
              break;
            }
          }
        if (valid) kept.push_back(candidate);
        else {
          solution.rejections.push_back(
              {id, layoutCandidateKey(candidate.value), reason, rejected});
          if (rejection.kind == LayoutSolveFailureKind::EmptyDomain ||
              rejectedKind == LayoutSolveFailureKind::Unsupported)
            rejection = {rejectedKind, id, rejected, reason};
        }
      }
      var.candidates = std::move(kept);
      if (var.candidates.empty()) {
        if (rejection.kind != LayoutSolveFailureKind::EmptyDomain) {
          *why = rejection;
          emitError(getVariableLoc(var)) << "no proven layout candidate for "
              << var.stableName << " (hard constraints not proven): " << rejection.reason;
          return failure();
        }
        if (emptyFailure.kind == LayoutSolveFailureKind::None) emptyFailure = rejection;
        emptyComponent = true;
      }
      stats.domains.push_back(var.candidates.size());
      stats.combinations = product(stats.combinations, var.candidates.size(),
                                    stats.productOverflow);
    }
    // Still inspect every other component: an empty domain must not hide an
    // unsupported decoder, hard conflict, incomplete search or exhausted budget.
    if (emptyComponent) {
      // Audit the remaining binary subdomains in this component as well. An
      // empty neighbour cannot hide an unsupported address/transform proof.
      // The same finite search budget bounds this residual proof work.
      for (auto cid : constraints) {
        const auto &c = graph.getConstraint(cid);
        if (!isSupportedLayoutRelation(c.kind) || c.vars.size() < 2) continue;
        auto lhs = c.vars.front();
        for (auto rhs : llvm::drop_begin(c.vars)) {
          const auto &a = graph.getVariable(lhs).candidates;
          const auto &b = graph.getVariable(rhs).candidates;
          if (a.empty() || b.empty()) continue;
          bool supported = false, unknown = false;
          for (const auto &x : a) {
            for (const auto &y : b) {
              if (stats.expanded == options.maxExpandedStates) {
                *why = {LayoutSolveFailureKind::SearchBudget, stats.root, cid,
                        "search-budget-exceeded in residual domain audit"};
                emitError(loc) << why->reason;
                return failure();
              }
              ++stats.expanded;
              auto proof = proveLayoutRelation(graph,c,lhs,x.value,rhs,y.value);
              supported |= proof.status == ProofStatus::Proven;
              unknown |= proof.status == ProofStatus::Unknown;
              if (supported) break;
            }
            if (supported) break;
          }
          if (!supported) {
            *why = {unknown ? LayoutSolveFailureKind::Unsupported
                            : LayoutSolveFailureKind::HardConflict,
                    lhs,cid,"nonempty residual relation has no proven support"};
            emitError(loc) << why->reason;
            return failure();
          }
        }
      }
      // Pairwise support is not joint feasibility (for example an inconsistent
      // equality cycle). Run the same bounded search on the nonempty residual
      // component, retaining complete constraints and the budget already spent.
      // Partially missing n-ary constraints cannot be projected generically.
      for (auto cid : constraints) {
        const auto &c = graph.getConstraint(cid);
        auto present = llvm::count_if(c.vars, [&](auto id) {
          return !graph.getVariable(id).candidates.empty();
        });
        if (present > 1 && present < c.vars.size()) {
          *why = {LayoutSolveFailureKind::Unsupported, stats.root, cid,
                  "cannot prove partial n-ary residual constraint"};
          emitError(loc) << why->reason;
          return failure();
        }
      }
      llvm::erase_if(constraints, [&](auto cid) {
        return llvm::any_of(graph.getConstraint(cid).vars, [&](auto id) {
          return graph.getVariable(id).candidates.empty();
        });
      });
      llvm::erase_if(component, [&](auto id) {
        return graph.getVariable(id).candidates.empty();
      });
      if (component.empty()) continue;
      stats.combinations = 1;
      stats.productOverflow = false;
      for (auto id : component)
        stats.combinations = product(stats.combinations,
            graph.getVariable(id).candidates.size(), stats.productOverflow);
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
    bool unknownRelation = false;
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
          if (!ok && c.kind != ConstraintKind::InstructionContract &&
              c.kind != ConstraintKind::ReductionLayout &&
              llvm::all_of(c.vars, [&](auto id) { return s.assignment.values.count(id); })) {
            if (isSupportedUnaryLayoutConstraint(c.kind))
              unknownRelation |= proveUnaryLayoutConstraint(graph, c,
                  s.assignment.values.lookup(c.vars.front())).status == ProofStatus::Unknown;
            else if (isSupportedLayoutRelation(c.kind))
              for (auto id : llvm::drop_begin(c.vars))
                unknownRelation |= proveLayoutRelation(graph, c,
                    c.vars.front(), s.assignment.values.lookup(c.vars.front()),
                    id, s.assignment.values.lookup(id)).status == ProofStatus::Unknown;
          }
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
      // Boundary choices may later share a conversion or replace it with a
      // cheap clone. Zero is a safe partial bound; per-use charges are not.
      if (complete) {
        auto choices = chooseLayoutConversions(graph, s.assignment, target);
        if (failed(choices)) return failure();
        *cost = addLayoutCosts(*cost, choices->cost);
        if (edges) llvm::append_range(*edges, choices->edges);
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
    stats.peakFrontier = 1;
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
              *why = {LayoutSolveFailureKind::SearchBudget, stats.root,
                      UINT32_MAX, "search-budget-exceeded"};
              emitError(getVariableLoc(graph.getVariable(stats.root)))
                  << "search-budget-exceeded: expanded=" << stats.expanded
                  << ", limit=" << options.maxExpandedStates;
              return failure();
            }
            ++stats.expanded;
            ++stats.mrvProbes;
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
          ++stats.expandedNodes;
          if (failed(evaluate(child, !emptyComponent &&
                                     depth + 1 == component.size()))) {
            *why = {LayoutSolveFailureKind::CostFailure, stats.root,
                    UINT32_MAX, "layout cost evaluation failed"};
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
      stats.peakFrontier = std::max<uint64_t>(stats.peakFrontier, frontier.size());
      if (frontier.empty()) break;
    }
    stats.exhaustive = stats.beamDiscarded == 0;
    if (frontier.empty()) {
      *why = {unknownRelation ? LayoutSolveFailureKind::Unsupported
                : stats.exhaustive ? LayoutSolveFailureKind::HardConflict
                              : LayoutSolveFailureKind::SearchIncomplete,
              stats.root, UINT32_MAX,
              unknownRelation ? "unknown binary layout relation proof"
                : stats.exhaustive ? "no feasible proven assignment"
                               : "beam discarded alternatives"};
      emitError(getVariableLoc(graph.getVariable(stats.root)))
          << (unknownRelation ? "unsupported: unknown binary layout relation proof"
                : stats.exhaustive ? "no feasible proven layout assignment"
                              : "search-incomplete: beam discarded alternatives");
      return failure();
    }
    // A residual assignment proves only joint feasibility, not a complete
    // component solution. Its missing conversion/instruction endpoints cannot
    // be priced by the complete-assignment API; only partial lower bounds above
    // are meaningful. The deferred EmptyDomain still rejects the original graph.
    if (emptyComponent) continue;
    auto &best = frontier.front();
    if (failed(evaluate(best, true, &solution.conversions))) {
      *why = {LayoutSolveFailureKind::CostFailure, stats.root, UINT32_MAX,
              "selected boundary cost evaluation failed"};
      return failure();
    }
    stats.cost = best.bound;
    solution.cost = addLayoutCosts(solution.cost, best.bound);
    for (auto id : component)
      solution.assignments[id] = best.assignment.values.lookup(id);
    solution.components.push_back(stats);
  }
  if (emptyFailure.kind != LayoutSolveFailureKind::None) {
    *why = emptyFailure;
    const auto &var = graph.getVariable(why->variable);
    emitError(getVariableLoc(var)) << "no proven layout candidate for "
        << var.stableName << " (hard constraints not proven): " << why->reason;
    return failure();
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
  *why = {};
  if (statistics) finishLayoutStatistics(graph, *why, &solution, *statistics);
  solved = true;
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
