#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
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
                 LayoutAttemptStatistics *statistics, LayoutTrace *trace,
                 bool diagnostic) {
  if (trace && !trace->enabled()) trace = nullptr;
  if (trace) {
    trace->setSolverOptions(options);
    auto summary = trace->getSummary();
    summary.costModelID = normalizeLayoutCostModelID(target.costModelID()).str();
    summary.staticTripCountFrequency = target.boundaryFrequencyMode() == LayoutBoundaryFrequencyMode::StaticTripCount;
    trace->setSummary(summary);
  }
  LayoutSolveFailure localFailure;
  if (!why) why = &localFailure;
  LayoutSolution solution;
  solution.policy = options.policy;
  if (trace) trace->recordEvent([&] { return llvm::json::Object{
      {"event", "solver-options"}, {"diagnostic", diagnostic},
      {"exactCombinationLimit", options.exactCombinationLimit},
      {"beamWidth", options.beamWidth}, {"maxExpandedStates", options.maxExpandedStates},
      {"refineBeamSolution", options.refineBeamSolution},
      {"maxRefinementProbes", options.maxRefinementProbes},
      {"costCoverage", target.costCoverage()}}; });
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
  const bool experimental = options.sharedBudget || !options.policy.scope.fullGraph ||
      options.policy.conversion != ConversionPolicy::Full ||
      !options.policy.scope.producers.empty() || !options.policy.scope.producerInstructions.empty();
  if (experimental && !graph.isFinalized()) {
    emitError(loc) << "experimental layout scope requires a finalized frozen graph";
    return failure();
  }
  for(auto id:options.policy.scope.producers)if(id>=graph.getVariables().size())return failure();
  for(auto id:options.policy.scope.producerInstructions)if(id>=graph.getConstraints().size())return failure();
  auto unsupportedPolicy=experimentPolicyUnsupportedReason(graph,options.policy);
  if(!unsupportedPolicy.empty()) {
    *why={LayoutSolveFailureKind::Unsupported,UINT32_MAX,UINT32_MAX,unsupportedPolicy};
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
  bool refinementEvaluation = false;
  bool sharedBudgetExhausted = false;
  auto debit = [&](bool refinement, bool completeScore) {
    if (!options.sharedBudget) return true;
    auto &budget = *options.sharedBudget;
    if (!budget.consume(refinement)) { sharedBudgetExhausted=true; return false; }
    if (refinement) ++budget.refinementProbes;
    else if (completeScore) ++budget.completeScores;
    else ++budget.relationshipProbes;
    return true;
  };
  auto evaluateComplete = [&](const CandidateAssignment &assignment,
      SmallVectorImpl<LayoutConversionEdge> *edges = nullptr) -> FailureOr<CostEstimate> {
    // Winner certificate rebuilding is diagnostic, not a new search probe.
    if (!edges && !refinementEvaluation && !debit(false,true)) return failure();
    auto *observer = edges && !diagnostic ? trace : nullptr;
    auto cost = scoreExperimentAssignment(graph, assignment, target, options.policy, observer);
    if (failed(cost)) return failure();
    if (edges) {
      auto choices=chooseLayoutConversions(graph,assignment,target,nullptr,options.policy.conversion);
      if(failed(choices))return failure();
      llvm::append_range(*edges,choices->edges);
    }
    return *cost;
  };
  struct PendingComponent {
    CandidateAssignment original, refined;
    CostEstimate originalCost, refinedCost;
    size_t solutionIndex, statisticsIndex;
  };
  SmallVector<PendingComponent, 1> pending;
  for (auto component : getHardConstraintComponents(graph)) {
    bool emptyComponent = false;
    bool selectedComponent = false;
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
      statistics->refinementProbes += stats.refinementProbes;
      statistics->refinementSweeps += stats.refinementSweeps;
      statistics->refinementAccepted += stats.refinementAccepted;
      llvm::json::Array domains;
      for (auto domain : stats.domains) domains.push_back(domain);
      statistics->components.push_back(llvm::json::Object{
        {"root", graph.getVariable(stats.root).stableName},
        {"combinations", stats.productOverflow ? llvm::json::Value(nullptr) : llvm::json::Value(stats.combinations)},
        {"productOverflow", stats.productOverflow}, {"domains", std::move(domains)},
        {"mode", stats.usedBeam ? "beam" : "exact"}, {"beamWidth", stats.beamWidth},
        {"expanded", stats.expanded}, {"peakFrontier", stats.peakFrontier},
        {"mrvProbes", stats.mrvProbes}, {"expandedNodes", stats.expandedNodes},
        {"refinementProbes", stats.refinementProbes},
        {"refinementSweeps", stats.refinementSweeps},
        {"refinementAccepted", stats.refinementAccepted},
        {"refinementStartCost", stats.refinementStopReason == "no-incumbent" ? llvm::json::Value(nullptr) : llvm::json::Value(layoutCostJSON(stats.refinementStartCost))},
        {"refinementEndCost", stats.refinementStopReason == "no-incumbent" ? llvm::json::Value(nullptr) : llvm::json::Value(layoutCostJSON(stats.refinementEndCost))},
        {"refinementStopReason", stats.refinementStopReason},
        {"refinementGlobalFallback", stats.refinementGlobalFallback},
        {"selectedCost", selectedComponent ? llvm::json::Value(layoutCostJSON(stats.cost)) : llvm::json::Value(nullptr)},
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
        StringRef rejectionStatus = "Unknown";
        SmallVector<int64_t> counterexample;
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
              rejectionStatus = c.kind == ConstraintKind::RequireEncoding ? "Disproven" : "NoSupport";
              if (isSupportedUnaryLayoutConstraint(c.kind)) {
                auto proof = proveUnaryLayoutConstraint(graph, c, candidate.value);
                reason += ": " + proof.reason;
                rejectionStatus = proof.status == ProofStatus::Unknown ? "Unknown" : "Disproven";
                if (trace) counterexample = proof.counterexample;
                rejectedKind = proof.status == ProofStatus::Unknown
                    ? LayoutSolveFailureKind::Unsupported
                    : LayoutSolveFailureKind::EmptyDomain;
              }
              break;
            }
          }
        if (valid) kept.push_back(candidate);
        else {
          if (trace) trace->recordRejection(id, candidate.value, rejected,
              diagnostic ? "diagnostic-filter" : "search-filter",
              rejectionStatus, reason, counterexample);
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
              if (stats.expanded == options.maxExpandedStates || !debit(false,false)) {
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
    struct ConstraintCheck { bool legal; StringRef status; };
    std::map<std::vector<unsigned>, ConstraintCheck> relationCache;
    bool unknownRelation = false;
    auto viable = [&](const State &s, StringRef *status = nullptr,
                      LayoutConstraintID *rejected = nullptr) {
      for (auto cid : constraints) {
        const auto &c = graph.getConstraint(cid);
        std::vector<unsigned> key{cid};
        for (auto id : c.vars) {
          auto it = llvm::find(component, id);
          key.push_back(s.key[it - component.begin()]);
        }
        LayoutProfile::Scope profile(LayoutProfileKind::RelationCache);
        auto found = relationCache.find(key);
        LayoutProfile::cacheLookup(LayoutProfileKind::RelationCache, found != relationCache.end());
        ConstraintCheck check{true, "Proven"};
        if (found != relationCache.end()) check = found->second;
        else {
          check.legal = satisfiesLayoutConstraint(graph, c, s.assignment.values, false);
          if (!check.legal) check.status = "NoSupport";
          if (!check.legal && c.kind != ConstraintKind::InstructionContract &&
              c.kind != ConstraintKind::ReductionLayout &&
              llvm::all_of(c.vars, [&](auto id) { return s.assignment.values.count(id); })) {
            bool unknown = false;
            if (isSupportedUnaryLayoutConstraint(c.kind))
              unknown = proveUnaryLayoutConstraint(graph, c,
                  s.assignment.values.lookup(c.vars.front())).status == ProofStatus::Unknown;
            else if (isSupportedLayoutRelation(c.kind))
              for (auto id : llvm::drop_begin(c.vars))
                unknown |= proveLayoutRelation(graph, c,
                    c.vars.front(), s.assignment.values.lookup(c.vars.front()),
                    id, s.assignment.values.lookup(id)).status == ProofStatus::Unknown;
            unknownRelation |= unknown;
            check.status = unknown ? "Unknown" : "Disproven";
          }
          if (relationCache.emplace(std::move(key), check).second)
            LayoutProfile::cacheInsertion(LayoutProfileKind::RelationCache);
        }
        if (!check.legal) {
          if (status) *status = check.status;
          if (rejected) *rejected = cid;
          return false;
        }
      }
      return true;
    };
    auto evaluate = [&](State &s, bool complete,
                        SmallVectorImpl<LayoutConversionEdge> *edges = nullptr) {
      auto cost = complete ? evaluateComplete(s.assignment, edges)
                           : options.policy.scope.fullGraph ? target.lowerBound(graph, s.assignment)
                               : FailureOr<CostEstimate>(CostEstimate{});
      if (failed(cost)) return failure();
      // Boundary choices may later share a conversion or replace it with a
      // cheap clone. Zero is a safe partial bound; per-use charges are not.
      s.bound = *cost;
      return success();
    };
    auto less = [&](const State &a, const State &b) {
      if (a.bound.cost < b.bound.cost) return true;
      if (b.bound.cost < a.bound.cost) return false;
      if (!options.policy.scope.fullGraph) {
        SmallVector<LayoutVarID> producers(options.policy.scope.producers);
        llvm::sort(producers, [&](auto x,auto y) {
          return graph.getVariable(x).stableName < graph.getVariable(y).stableName;
        });
        for (auto id:producers) {
          if (!llvm::is_contained(component,id)) continue;
          auto ordinal = [&](const State &state) {
            auto value=state.assignment.values.lookup(id);
            for (auto &c:graph.getVariable(id).candidates)
              if(c.value==value)return c.stableOrdinal;
            return UINT64_MAX;
          };
          auto x=ordinal(a),y=ordinal(b);if(x!=y)return x<y;
        }
        return false; // Consumer completion never ranks a producer tuple.
      }
      return std::lexicographical_compare(a.key.begin(), a.key.end(),
                                          b.key.begin(), b.key.end());
    };
    State initial;
    initial.key.assign(component.size(), UINT32_MAX);
    SmallVector<State> frontier;
    frontier.push_back(std::move(initial));
    stats.peakFrontier = 1;
    for (size_t depth = 0; depth < component.size(); ++depth) {
      const auto probesBefore = stats.mrvProbes, expandedBefore = stats.expandedNodes;
      const auto rejectedBefore = stats.hardRejected;
      // Counts, not a second search tree. Allocation is trace-only.
      std::map<LayoutVarID, uint64_t> selections;
      SmallVector<State> next;
      bool completedDepth = false;
      auto finishDepth = llvm::make_scope_exit([&] {
        if (!trace || completedDepth) return;
        trace->recordEvent([&] {
          llvm::json::Array mrv;
          for (auto [id, count] : selections)
            mrv.push_back(llvm::json::Object{{"variable", id}, {"count", count}});
          return llvm::json::Object{{"event", "search-depth"}, {"diagnostic", diagnostic},
              {"component", stats.root}, {"depth", depth}, {"completed", false},
              {"costKind", !emptyComponent && depth + 1 == component.size() ? "complete" : "partial-lower-bound"},
              {"mrvSelections", std::move(mrv)}, {"probes", stats.mrvProbes - probesBefore},
              {"expanded", stats.expandedNodes - expandedBefore}, {"hardRejected", stats.hardRejected - rejectedBefore},
              {"retained", 0}, {"beamDiscarded", 0}, {"pending", next.size()},
              {"retainedExamples", llvm::json::Array{}}, {"discardedExamples", llvm::json::Array{}},
              {"stopReason", why->reason}};
        });
      });
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
            if (stats.expanded == options.maxExpandedStates || !debit(false,false)) {
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
        if (trace && trace->acceptingEvents() && chosen != component.size())
          ++selections[component[chosen]];
        for (auto &child : children) {
          ++stats.expandedNodes;
          if (failed(evaluate(child, !emptyComponent &&
                                     depth + 1 == component.size()))) {
            *why = {sharedBudgetExhausted ? LayoutSolveFailureKind::SearchBudget : LayoutSolveFailureKind::CostFailure, stats.root,
                    UINT32_MAX, sharedBudgetExhausted ? "search-budget-exceeded in complete score" : "layout cost evaluation failed"};
            emitError(loc) << "layout cost evaluation failed; not a free solution";
            return failure();
          }
          next.push_back(std::move(child));
        }
      }
      llvm::sort(next, less);
      if (trace) trace->recordEvent([&] {
        const size_t retained = stats.usedBeam ? std::min<size_t>(next.size(), options.beamWidth) : next.size();
        auto examples = [&](size_t begin, size_t end) {
          llvm::json::Array states;
          for (size_t i = begin; i < std::min(end, begin + 4); ++i) {
            llvm::json::Array key;
            for (size_t j = 0; j < component.size(); ++j)
              key.push_back(next[i].key[j] == UINT32_MAX ? llvm::json::Value(nullptr) :
                  llvm::json::Value(LayoutTrace::candidateID(graph.getVariable(component[j]).candidates[next[i].key[j]].value)));
            states.push_back(llvm::json::Object{{"key", std::move(key)}, {"cost", layoutCostJSON(next[i].bound)}});
          }
          return states;
        };
        llvm::json::Array mrv, variables;
        for (auto id : component) variables.push_back(id);
        for (auto [id, count] : selections) mrv.push_back(llvm::json::Object{{"variable", id}, {"count", count}});
        return llvm::json::Object{{"event", "search-depth"}, {"diagnostic", diagnostic},
            {"completed", true},
            {"component", stats.root}, {"variables", std::move(variables)}, {"depth", depth},
            {"costKind", !emptyComponent && depth + 1 == component.size() ? "complete" : "partial-lower-bound"},
            {"mrvSelections", std::move(mrv)}, {"probes", stats.mrvProbes - probesBefore},
            {"expanded", stats.expandedNodes - expandedBefore}, {"hardRejected", stats.hardRejected - rejectedBefore},
            {"retained", retained}, {"beamDiscarded", next.size() - retained},
            {"discardReason", "beam-ranking-not-proof-rejection"},
            {"retainedExamples", examples(0, retained)}, {"discardedExamples", examples(retained, next.size())}};
      });
      completedDepth = true;
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
    bool mayRefine = stats.usedBeam && options.refineBeamSolution && options.maxRefinementProbes;
    CandidateAssignment original;
    if (mayRefine) original = best.assignment;
    stats.refinementStartCost = stats.refinementEndCost = best.bound;
    stats.refinementStopReason = !stats.usedBeam ? "exact-mode" : "disabled";
    if (mayRefine) {
      stats.refinementStopReason.clear();
      auto finishRefinement = llvm::make_scope_exit([&] {
        stats.refinementEndCost = best.bound;
        if (trace) trace->recordEvent([&] { return llvm::json::Object{
            {"event", "refinement-stop"}, {"diagnostic", diagnostic},
            {"component", stats.root}, {"reason", stats.refinementStopReason},
            {"probes", stats.refinementProbes}, {"sweeps", stats.refinementSweeps},
            {"accepted", stats.refinementAccepted},
            {"startCost", layoutCostJSON(stats.refinementStartCost)},
            {"endCost", layoutCostJSON(stats.refinementEndCost)}}; });
      });
      if (trace) trace->recordEvent([&] { return llvm::json::Object{
          {"event", "refinement-start"}, {"diagnostic", diagnostic},
          {"component", stats.root}, {"cost", layoutCostJSON(best.bound)}}; });
      auto budgetExhausted = [&] {
        if (stats.expanded == options.maxExpandedStates ||
            (options.sharedBudget && (options.sharedBudget->used >= options.sharedBudget->stageLimit ||
                                     options.sharedBudget->used >= options.sharedBudget->limit)))
          stats.refinementStopReason = "total-budget";
        else if (options.sharedBudget && (options.sharedBudget->refinementUsed >= options.sharedBudget->stageRefinementLimit ||
                                         options.sharedBudget->refinementUsed >= options.sharedBudget->refinementLimit))
          stats.refinementStopReason = "refinement-budget";
        else if (stats.refinementProbes == options.maxRefinementProbes)
          stats.refinementStopReason = "refinement-budget";
        return !stats.refinementStopReason.empty();
      };
      while (!budgetExhausted()) {
        ++stats.refinementSweeps;
        State improving = best;
        size_t changed = component.size();
        for (size_t index = 0; index < component.size(); ++index) {
          auto id = component[index];
          const auto &domain = graph.getVariable(id).candidates;
          for (unsigned ci = 0; ci < domain.size(); ++ci) {
            if (ci == best.key[index]) continue;
            if (budgetExhausted()) break;
            if (!debit(true,false)) {stats.refinementStopReason="shared-budget";break;}
            ++stats.expanded;
            ++stats.refinementProbes;
            State neighbor = best;
            neighbor.key[index] = ci;
            neighbor.assignment.values[id] = domain[ci].value;
            StringRef status = "Proven";
            LayoutConstraintID rejected = UINT32_MAX;
            bool legal = viable(neighbor, &status, &rejected);
            refinementEvaluation=true;
            bool priced = legal && succeeded(evaluate(neighbor, true));
            refinementEvaluation=false;
            bool improves = priced && (options.policy.scope.fullGraph
                ? neighbor.bound.cost < best.bound.cost : less(neighbor,best));
            if (trace) trace->recordEvent([&] { return llvm::json::Object{
                {"event", "refinement-probe"}, {"diagnostic", diagnostic},
                {"component", stats.root}, {"sweep", stats.refinementSweeps},
                {"probe", stats.refinementProbes}, {"variable", id},
                {"oldCandidate", LayoutTrace::candidateID(best.assignment.values.lookup(id))},
                {"newCandidate", LayoutTrace::candidateID(domain[ci].value)},
                {"status", status}, {"constraint", rejected}, {"legal", legal},
                {"result", !legal ? "constraint-not-proven" : !priced ? "cost-failure" :
                    improves ? "strict-improvement" : "not-improving"},
                {"cost", priced ? llvm::json::Value(layoutCostJSON(neighbor.bound)) : llvm::json::Value(nullptr)}}; });
            if (legal && !priced) {
              stats.refinementStopReason = "cost-failure";
              *why = {LayoutSolveFailureKind::CostFailure, id, UINT32_MAX,
                      "refinement cost evaluation failed"};
              emitError(loc) << "refinement cost evaluation failed; not a free solution";
              return failure();
            }
            if (improves && less(neighbor, improving)) {
              improving = std::move(neighbor);
              changed = index;
            }
          }
          if (!stats.refinementStopReason.empty()) break;
        }
        // Even an interrupted sweep may retain its best verified improvement.
        if (changed != component.size()) {
          ++stats.refinementAccepted;
          auto id = component[changed];
          if (trace) trace->recordEvent([&] { return llvm::json::Object{
              {"event", "refinement-accepted"}, {"diagnostic", diagnostic},
              {"component", stats.root}, {"sweep", stats.refinementSweeps},
              {"variable", id}, {"accepted", true},
              {"oldCandidate", LayoutTrace::candidateID(best.assignment.values.lookup(id))},
              {"newCandidate", LayoutTrace::candidateID(improving.assignment.values.lookup(id))},
              {"cost", layoutCostJSON(improving.bound)}}; });
          best = std::move(improving);
        } else if (!budgetExhausted()) {
          stats.refinementStopReason = "no-improvement";
        }
        if (!stats.refinementStopReason.empty()) break;
      }
    }
    if (mayRefine) {
      // Saturation can erase an earlier improvement while preserving a worse
      // later dimension. Defer only mutable beam selections until the whole
      // saturated totals can be compared; never undo observed contributions.
      pending.push_back({std::move(original), best.assignment,
          stats.refinementStartCost, best.bound, solution.components.size(),
          statistics ? statistics->components.size() : 0});
    } else {
      if (failed(evaluate(best, true, &solution.conversions))) {
        *why = {LayoutSolveFailureKind::CostFailure, stats.root, UINT32_MAX,
                "selected boundary cost evaluation failed"};
        return failure();
      }
      selectedComponent = true;
      solution.cost = addLayoutCosts(solution.cost, best.bound);
      for (auto id : component)
        solution.assignments[id] = best.assignment.values.lookup(id);
    }
    stats.cost = best.bound;
    solution.components.push_back(stats);
  }
  if (emptyFailure.kind != LayoutSolveFailureKind::None) {
    *why = emptyFailure;
    const auto &var = graph.getVariable(why->variable);
    emitError(getVariableLoc(var)) << "no proven layout candidate for "
        << var.stableName << " (hard constraints not proven): " << why->reason;
    return failure();
  }
  if (!pending.empty()) {
    CostEstimate originalCost = solution.cost, refinedCost = solution.cost;
    for (const auto &p : pending) {
      originalCost = addLayoutCosts(originalCost, p.originalCost);
      refinedCost = addLayoutCosts(refinedCost, p.refinedCost);
    }
    bool fallback = originalCost.cost < refinedCost.cost;
    if (!options.policy.scope.fullGraph && !(refinedCost.cost < originalCost.cost) &&
        !(originalCost.cost < refinedCost.cost)) {
      CandidateAssignment original, refined;
      original.values=refined.values=solution.assignments;
      for(const auto &p:pending) {
        for(auto [id,value]:p.original.values)original.values[id]=value;
        for(auto [id,value]:p.refined.values)refined.values[id]=value;
      }
      auto producers=options.policy.scope.producers;
      llvm::sort(producers,[&](auto x,auto y){return graph.getVariable(x).stableName<graph.getVariable(y).stableName;});
      for(auto id:producers) {
        auto ordinal=[&](Attribute value){for(const auto &c:graph.getVariable(id).candidates)if(c.value==value)return c.stableOrdinal;return UINT64_MAX;};
        auto a=ordinal(original.values.lookup(id)),b=ordinal(refined.values.lookup(id));
        if(a!=b){fallback=a<b;break;}
      }
    }
    if (trace) trace->recordEvent([&] { return llvm::json::Object{
        {"event", "refinement-global-selection"}, {"diagnostic", diagnostic},
        {"fallback", fallback}, {"originalCost", layoutCostJSON(originalCost)},
        {"refinedCost", layoutCostJSON(refinedCost)},
        {"selectedCost", layoutCostJSON(fallback ? originalCost : refinedCost)}}; });
    for (const auto &p : pending) {
      auto &stats = solution.components[p.solutionIndex];
      const auto &assignment = fallback ? p.original : p.refined;
      stats.refinementGlobalFallback = fallback;
      if (statistics)
        (*statistics->components[p.statisticsIndex].getAsObject())["refinementGlobalFallback"] = fallback;
      auto cost = evaluateComplete(assignment, &solution.conversions);
      if (failed(cost)) {
        *why = {LayoutSolveFailureKind::CostFailure, stats.root, UINT32_MAX,
                "selected boundary cost evaluation failed"};
        return failure();
      }
      stats.cost = *cost;
      if (statistics)
        (*statistics->components[p.statisticsIndex].getAsObject())["selectedCost"] = layoutCostJSON(*cost);
      solution.cost = addLayoutCosts(solution.cost, *cost);
      for (auto [id, value] : assignment.values)
        solution.assignments[id] = value;
    }
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
  if (trace && !diagnostic) trace->recordSolution(graph, solution);
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
