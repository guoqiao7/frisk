#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <random>

using namespace mlir;
using namespace mlir::frisk;
namespace {
class RankedTarget : public LayoutTarget {
public:
  bool reverse = false;
  bool failCost = false;
  void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &, Attribute a, Location) const override {
    return success(isa<StringAttr>(a));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
    if (failCost) return failure();
    CostVector result;
    for (auto entry : a.values) {
      auto name = cast<StringAttr>(entry.second).getValue();
      result.instructionPathAndWork += (name == "a") == reverse ? 0 : 1;
    }
    return result;
  }
};
class LayoutCandidateSolverTest : public testing::Test {
protected:
  MLIRContext context;
  Builder builder{&context};
  Location loc = builder.getUnknownLoc();
  RankedTarget target;
  LayoutConstraintGraph graph;
  OwningOpRef<ModuleOp> source{ModuleOp::create(loc)};
  LayoutVarID variable(StringRef name, unsigned count = 2) {
    auto id = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({4}, builder.getF32Type()), name);
    for (unsigned i = 0; i < count; ++i)
      graph.getVariable(id).candidates.push_back(
          {builder.getStringAttr(std::string(1, char('a' + i))), kInvalidProvenanceID, 99-i});
    graph.getVariable(id).state = LayoutState::CandidateSet;
    return id;
  }
  void same(LayoutVarID a, LayoutVarID b) {
    graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                        {a,b}, nullptr, "same", "same test layout");
  }
  // A finite, independently specified binary relation represented by a
  // four-role instruction table (the last three roles share one endpoint).
  void pair(LayoutVarID a, LayoutVarID b, unsigned mask, bool balanced = false) {
    auto id = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, {a,b,balanced ? a : b,b}, *source, "finite", "finite relation");
    auto &c = graph.getConstraints()[id];
    c.instruction = InstructionLayoutContract{*source, {}, {}};
    for (unsigned x = 0; x < 2; ++x)
      for (unsigned y = 0; y < 2; ++y)
        if (mask & (1u << (2*x+y))) {
          auto lhs = builder.getStringAttr(x ? "b" : "a");
          auto rhs = builder.getStringAttr(y ? "b" : "a");
          c.instruction->tuples.push_back({{lhs,rhs,balanced ? lhs : rhs,rhs}, builder.getStringAttr("plan")});
        }
  }
};

// A zero partial bound deliberately hides the first variable's complete cost.
// Beam width one fixes it to a; the last variable is still priced completely.
class NeighborhoodTarget : public RankedTarget {
public:
  CostVector costs[4] = {{10}, {12}, {5}, {0}};
  int failingAssignment = -1;
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
    unsigned index = 2 * (cast<StringAttr>(a.values.lookup(0)).getValue() == "b") +
                     (cast<StringAttr>(a.values.lookup(1)).getValue() == "b");
    if (int(index) == failingAssignment) return failure();
    return costs[index];
  }
};

TEST_F(LayoutCandidateSolverTest, RefinementDefaultRecoversMissedNeighborInMultipleSweeps) {
  variable("x"); variable("y"); pair(0, 1, 15, true);
  NeighborhoodTarget costs;
  SolverOptions options{1, 1, 65536};
  EXPECT_TRUE(options.refineBeamSolution);
  EXPECT_EQ(options.maxRefinementProbes, 256u);
  LayoutAttemptStatistics statistics;
  LayoutTrace trace({"unused.json"});
  auto result = solveLayoutGraph(graph, costs, options, nullptr, &statistics, &trace);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->cost.cost.instructionPathAndWork, 0u);
  auto &s = result->components.front();
  EXPECT_EQ(s.refinementAccepted, 2u);
  EXPECT_EQ(s.refinementSweeps, 3u);
  EXPECT_EQ(s.refinementProbes, 6u);
  EXPECT_EQ(s.refinementStopReason, "no-improvement");
  EXPECT_EQ(s.refinementStartCost.cost.instructionPathAndWork, 10u);
  EXPECT_FALSE(s.refinementEndCost.cost < result->cost.cost);
  EXPECT_FALSE(result->cost.cost < s.refinementEndCost.cost);
  EXPECT_EQ(s.expanded, s.mrvProbes + s.refinementProbes);
  EXPECT_EQ(statistics.refinementProbes, 6u);
  EXPECT_FALSE(s.exhaustive);
  auto json = llvm::json::parse(trace.serialize()); ASSERT_TRUE(bool(json));
  unsigned probes = 0, accepted = 0, stopped = 0, contributions = 0;
  for (auto &value : *json->getAsObject()->getArray("events")) {
    auto *event = value.getAsObject(); auto kind = event->getString("event");
    probes += kind == "refinement-probe";
    accepted += kind == "refinement-accepted";
    stopped += kind == "refinement-stop";
    if (kind == "cost-contribution") {
      ++contributions;
      EXPECT_EQ((*event->getObject("estimate")->getArray("cost"))[0].getAsInteger(), 0);
    }
  }
  EXPECT_EQ(probes, 6u); EXPECT_EQ(accepted, 2u); EXPECT_EQ(stopped, 1u);
  EXPECT_EQ(contributions, 1u); // Only the refined winner, never proposals.
  auto copy = graph;
  LayoutTrace capped({"unused.json", 1, 4096});
  auto cappedResult = solveLayoutGraph(copy, costs, options, nullptr, nullptr, &capped);
  ASSERT_TRUE(succeeded(cappedResult));
  EXPECT_EQ(cappedResult->assignments, result->assignments);
  EXPECT_EQ(cappedResult->components.front().refinementProbes, 6u);
  EXPECT_EQ(capped.omittedCost().cost.instructionPathAndWork, 0u);
}

TEST_F(LayoutCandidateSolverTest, RefinementUsesEveryLexicographicDimensionAndCanonicalImprovementTie) {
  variable("x"); variable("y"); pair(0, 1, 15, true);
  for (unsigned dimension = 0; dimension < 9; ++dimension) {
    NeighborhoodTarget costs;
    uint64_t v[9] = {};
    v[dimension] = 10;
    costs.costs[0] = costs.costs[1] = {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]};
    v[dimension] = 5;
    if (dimension + 1 < 9) v[dimension + 1] = 1000;
    costs.costs[2] = costs.costs[3] = {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]};
    auto copy = graph;
    auto result = solveLayoutGraph(copy, costs, {1,1,65536});
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->assignments.lookup(0), builder.getStringAttr("b"));
    EXPECT_EQ(result->components.front().refinementAccepted, 1u);
  }
  // With three candidates, two unseen improving neighbors tie. Canonical b
  // wins even if the input candidate insertion order is reversed.
  class TieTarget : public RankedTarget {
  public:
    bool tied = true;
    FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
      auto name = cast<StringAttr>(a.values.lookup(0)).getValue();
      return CostVector{name == "a" ? 10u : name == "b" && !tied ? 5u : 0u};
    }
  } tie;
  graph = LayoutConstraintGraph(); variable("x", 3); variable("y", 3);
  auto cid = graph.addConstraint(ConstraintKind::InstructionContract,
      ConstraintStrength::Hard, {0,1,0,1}, *source, "tie", "full support");
  auto &c = graph.getConstraints()[cid]; c.instruction = InstructionLayoutContract{*source,{}, {}};
  for (auto &x : graph.getVariable(0).candidates) for (auto &y : graph.getVariable(1).candidates)
    c.instruction->tuples.push_back({{x.value,y.value,x.value,y.value},builder.getStringAttr("plan")});
  for (bool tied : {false,true}) for (bool reverse : {false,true}) {
    tie.tied = tied;
    auto copy = graph;
    if (reverse) for (auto &v : copy.getVariables()) std::reverse(v.candidates.begin(),v.candidates.end());
    auto result = solveLayoutGraph(copy, tie, {1,1,65536});
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->assignments.lookup(0), builder.getStringAttr(tied ? "b" : "c"));
    EXPECT_EQ(result->components.front().refinementAccepted, 1u);
  }
}

TEST_F(LayoutCandidateSolverTest, RefinementDisabledExactAndBudgetBoundaries) {
  variable("x"); variable("y"); pair(0, 1, 15, true);
  NeighborhoodTarget costs;
  for (unsigned mode = 0; mode < 6; ++mode) {
    auto copy = graph;
    SolverOptions options{1, 1, 65536};
    if (mode == 0) options.refineBeamSolution = false;
    if (mode == 1) options.maxRefinementProbes = 0;
    if (mode == 2) options.exactCombinationLimit = 256;
    if (mode == 3) options.maxExpandedStates = 6;
    if (mode == 4) options.maxExpandedStates = 7;
    if (mode == 5) options.maxRefinementProbes = 1;
    auto result = solveLayoutGraph(copy, costs, options);
    ASSERT_TRUE(succeeded(result)) << mode;
    auto &s = result->components.front();
    EXPECT_EQ(result->cost.cost.instructionPathAndWork, mode == 2 ? 0u : mode >= 4 ? 5u : 10u);
    EXPECT_EQ(s.refinementProbes, mode >= 4 ? 1u : 0u);
    EXPECT_LE(s.expanded, options.maxExpandedStates);
    EXPECT_EQ(s.refinementAccepted, mode >= 4 ? 1u : 0u);
    EXPECT_EQ(s.refinementStopReason, mode == 2 ? "exact-mode" : mode < 2 ? "disabled" :
              mode < 5 ? "total-budget" : "refinement-budget");
  }
}

TEST_F(LayoutCandidateSolverTest, RefinementStrictDescentLegalityAndCostFailure) {
  NeighborhoodTarget costs;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (unsigned mode = 0; mode < 4; ++mode) {
    graph = LayoutConstraintGraph();
    variable("x"); variable("y");
    pair(0, 1, mode == 1 ? 9 : mode == 2 ? 11 : 15, true);
    costs.costs[0] = CostVector{10}; costs.costs[1] = CostVector{12};
    costs.costs[2] = CostVector{mode == 0 ? 10u : 5u};
    costs.costs[3] = CostVector{0};
    costs.failingAssignment = mode == 3 ? 2 : -1;
    LayoutSolveFailure why;
    LayoutAttemptStatistics statistics;
    auto result = solveLayoutGraph(graph, costs, {1, 1, 65536}, &why, &statistics);
    if (mode == 3) {
      EXPECT_TRUE(failed(result));
      EXPECT_EQ(why.kind, LayoutSolveFailureKind::CostFailure);
      EXPECT_EQ(statistics.refinementProbes, 1u);
    } else {
      ASSERT_TRUE(succeeded(result));
      EXPECT_EQ(result->cost.cost.instructionPathAndWork, 10u);
      EXPECT_EQ(result->components.front().refinementAccepted, 0u);
      EXPECT_EQ(result->assignments.lookup(0), builder.getStringAttr("a"));
    }
  }
}

TEST_F(LayoutCandidateSolverTest, RefinementCostFailureDoesNotRetryOrPublishPartialGraph) {
  variable("x"); variable("y"); pair(0,1,15,true);
  // Preserve even noncanonical caller candidate order on coordinator failure.
  std::reverse(graph.getVariable(0).candidates.begin(),graph.getVariable(0).candidates.end());
  auto before = graph.getVariable(0).candidates;
  NeighborhoodTarget costs; costs.failingAssignment = 2;
  LayoutStatisticsReport report; LayoutSolveFailure why;
  SmallVector<RelaxationRecord> records;
  LayoutTrace trace({"unused.json"});
  std::string original; llvm::raw_string_ostream(original) << *source;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  auto result = attemptControlledRelaxation(graph,costs,{1,1,65536},records,&why,&report,&trace);
  EXPECT_TRUE(failed(result)); EXPECT_EQ(why.kind,LayoutSolveFailureKind::CostFailure);
  EXPECT_TRUE(records.empty()); ASSERT_EQ(report.attempts.size(),1u);
  EXPECT_EQ(report.attempts.front().refinementProbes,1u);
  EXPECT_EQ(report.attempts.front().components.front().getAsObject()->getString("refinementStopReason"),"cost-failure");
  EXPECT_FALSE(graph.isFinalized());
  for (unsigned i=0;i<before.size();++i)
    EXPECT_EQ(graph.getVariable(0).candidates[i].value,before[i].value);
  std::string after; llvm::raw_string_ostream(after) << *source;
  EXPECT_EQ(after,original);
  auto json = llvm::json::parse(trace.serialize()); ASSERT_TRUE(bool(json));
  EXPECT_EQ(json->getAsObject()->getString("failure_kind"),"cost-failure");
  EXPECT_EQ(json->getAsObject()->getInteger("contributionsSeen"),0);
  EXPECT_TRUE(json->getAsObject()->get("finalCost")->getAsNull());
}

TEST_F(LayoutCandidateSolverTest, SaturatedGlobalRefinementNeverWorsensMixedComponents) {
  class SaturatingTarget : public RankedTarget {
  public:
    bool failImproved = false;
    FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &g,
                                   const CandidateAssignment &a) const override {
      CostEstimate result;
      bool x = false, y = false, hasBeam = false;
      for (auto [id, attr] : a.values) {
        auto name = g.getVariable(id).stableName;
        if (name == "fixed") result = addLayoutCosts(result, {{UINT64_MAX},false});
        if (name == "x") { x = cast<StringAttr>(attr).getValue() == "b"; hasBeam = true; }
        if (name == "y") y = cast<StringAttr>(attr).getValue() == "b";
      }
      if (hasBeam && x && failImproved) return failure();
      if (hasBeam) result = addLayoutCosts(result, {{x ? (y ? 6u : 5u) : (y ? 12u : 10u), x ? 100u : 0u},false});
      return result;
    }
    LayoutProof verifyInstructionContract(const LayoutConstraintGraph &,
        const LayoutConstraint &c, ArrayRef<Attribute> values, Attribute binding) const override {
      for (auto &t : c.instruction->tuples)
        if (ArrayRef<Attribute>(t.encodings) == values && t.binding == binding)
          return {ProofStatus::Proven,{},"finite member"};
      return {ProofStatus::Disproven,{},"not a finite member"};
    }
  } costs;
  variable("fixed",1); variable("x"); variable("y"); pair(1,2,15,true);
  // Distinct selected instruction bindings expose stale refined-plan leakage.
  for (auto &tuple : graph.getConstraints().front().instruction->tuples)
    tuple.binding = tuple.encodings[0];
  CostVector reference;
  for (unsigned mode=0;mode<6;++mode) {
    auto copy = graph;
    LayoutTrace trace(mode == 1 ? LayoutTraceOptions{} :
                      LayoutTraceOptions{"unused.json",mode == 3 ? 1u : 4096u,mode == 3 ? 4096u : 8388608u});
    SolverOptions options{1,1,65536,mode != 0,256};
    if (mode == 4) options.maxRefinementProbes = 1;
    if (mode == 5) options.maxExpandedStates = 7;
    LayoutAttemptStatistics statistics;
    auto result = solveLayoutGraph(copy,costs,options,nullptr,&statistics,&trace);
    ASSERT_TRUE(succeeded(result));
    if (!mode) reference = result->cost.cost;
    EXPECT_FALSE(reference < result->cost.cost);
    EXPECT_EQ(result->cost.cost.memoryTransactions,0u);
    EXPECT_TRUE(result->cost.saturated);
    EXPECT_EQ(result->assignments.lookup(1),builder.getStringAttr("a"));
    EXPECT_EQ(result->instructionBindings.begin()->second,builder.getStringAttr("a"));
    EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(copy,*result,costs,loc)));
    EXPECT_EQ(result->components.size(),2u);
    EXPECT_FALSE(result->components.front().usedBeam);
    EXPECT_TRUE(result->components.back().usedBeam);
    EXPECT_EQ(result->components.back().cost.cost.memoryTransactions,0u);
    EXPECT_EQ(result->components.back().refinementGlobalFallback,mode != 0);
    EXPECT_EQ(result->components.back().refinementEndCost.cost.memoryTransactions,mode ? 100u : 0u);
    auto *componentJSON = statistics.components.back().getAsObject();
    EXPECT_EQ(componentJSON->getBoolean("refinementGlobalFallback"),mode != 0);
    EXPECT_EQ((*componentJSON->getObject("selectedCost")->getArray("cost"))[1].getAsInteger(),0);
    if (mode >= 4) {
      EXPECT_EQ(result->components.back().refinementProbes,1u);
      EXPECT_EQ(result->components.back().refinementStopReason,
                mode == 4 ? "refinement-budget" : "total-budget");
    }
    EXPECT_EQ(result->components.back().expanded,
              result->components.back().mrvProbes + result->components.back().refinementProbes);
    if (mode) EXPECT_EQ(result->components.back().refinementAccepted,1u);
    if (!trace.enabled()) continue;
    auto json = llvm::json::parse(trace.serialize()); ASSERT_TRUE(bool(json));
    CostEstimate explained = trace.omittedCost();
    for (auto &event : *json->getAsObject()->getArray("events")) {
      auto *e = event.getAsObject();
      if (e->getString("event") == "refinement-global-selection") {
        EXPECT_EQ(e->getBoolean("fallback"),true);
        EXPECT_EQ((*e->getObject("selectedCost")->getArray("cost"))[1].getAsInteger(),0);
      }
      if (e->getString("event") != "cost-contribution") continue;
      auto *v = e->getObject("estimate")->getArray("cost");
      CostVector contribution{*(*v)[0].getAsUINT64(),*(*v)[1].getAsUINT64()};
      explained = addLayoutCosts(explained,{contribution,*e->getObject("estimate")->getBoolean("saturated")});
    }
    EXPECT_FALSE(result->cost.cost < explained.cost);
    EXPECT_FALSE(explained.cost < result->cost.cost);
    EXPECT_EQ(explained.saturated,result->cost.saturated);
  }
  costs.failImproved = true;
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){ return success(); });
  LayoutSolveFailure why; auto copy = graph;
  auto failedCost = solveLayoutGraph(copy,costs,{1,1,65536},&why);
  EXPECT_TRUE(failed(failedCost)); EXPECT_EQ(why.kind,LayoutSolveFailureKind::CostFailure);
}

TEST_F(LayoutCandidateSolverTest, NineVariablesChooseCostNotOrdinal) {
  for (unsigned i = 0; i < 9; ++i) {
    auto id = variable("v" + std::to_string(i));
    if (i) same(id - 1, id);
  }
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto solution = solveLayoutGraph(graph, target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->assignments.size(), 9u);
  for (auto assignment : solution->assignments)
    EXPECT_EQ(assignment.second, builder.getStringAttr("b"));
  EXPECT_EQ(solution->cost.cost.instructionPathAndWork, 0u);
  ASSERT_EQ(solution->components.size(), 1u);
  EXPECT_TRUE(solution->components[0].usedBeam);
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph, *solution, target, loc)));
}

TEST_F(LayoutCandidateSolverTest, MoreThanFourCandidatesAndCanonicalTie) {
  variable("only", 6);
  graph.getVariable(0).candidates.push_back(graph.getVariable(0).candidates.back());
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto solution = solveLayoutGraph(graph, target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_EQ(solution->assignments.lookup(0), builder.getStringAttr("b"));
  EXPECT_EQ(solution->components[0].deduplicated, 1u);
  EXPECT_TRUE(solution->components[0].exhaustive);
}

TEST_F(LayoutCandidateSolverTest, BudgetCountsRejectedAndAcceptedAttempts) {
  variable("one"); variable("two"); same(0,1);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  SolverOptions options;
  options.maxExpandedStates = 1;
  std::string errors;
  ScopedDiagnosticHandler quiet(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream(errors) << d; return success();
  });
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target, options)));
  EXPECT_NE(errors.find("search-budget-exceeded"), std::string::npos);
}

TEST_F(LayoutCandidateSolverTest, InvalidOptionsAndCostAreNotFreeSolutions) {
  variable("one"); ASSERT_TRUE(succeeded(graph.finalize(loc)));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  SolverOptions options; options.beamWidth = 0;
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target, options)));
  target.failCost = true;
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target)));
}

TEST_F(LayoutCandidateSolverTest, EmptyGraphStillValidatesOptionsAndConstraints) {
  SolverOptions options; options.beamWidth=0;
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target,options)));
  EXPECT_TRUE(succeeded(solveLayoutGraph(graph,target)));
  graph.addConstraint(ConstraintKind::SameLayout,ConstraintStrength::Hard,
                      {},nullptr,"invalid","no endpoints");
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target)));
}

TEST_F(LayoutCandidateSolverTest, UnfinalizedInsertionOrderHasCanonicalTie) {
  for (bool reverse : {false,true}) {
    graph=LayoutConstraintGraph();
    variable(reverse?"z":"a"); variable(reverse?"a":"z");
    pair(0,1,0b0110);
    auto s=solveLayoutGraph(graph,target);
    ASSERT_TRUE(succeeded(s));
    EXPECT_EQ(graph.getVariable(0).stableName,"a");
    EXPECT_EQ(s->assignments.lookup(0),builder.getStringAttr("a"));
    EXPECT_EQ(s->assignments.lookup(1),builder.getStringAttr("b"));
  }
}

TEST_F(LayoutCandidateSolverTest, Exact256BoundaryAndAbove) {
  for (unsigned n : {7u,8u,9u}) {
    graph = LayoutConstraintGraph();
    for (unsigned i = 0; i < n; ++i) {
      variable("v" + std::to_string(i));
      if (i) same(i-1,i);
    }
    ASSERT_TRUE(succeeded(graph.finalize(loc)));
    auto s = solveLayoutGraph(graph,target);
    ASSERT_TRUE(succeeded(s));
    EXPECT_EQ(s->components[0].combinations, 1u << n);
    EXPECT_EQ(s->components[0].usedBeam, n > 8);
    EXPECT_TRUE(s->components[0].exhaustive);
  }
}

TEST_F(LayoutCandidateSolverTest, BeamLosingOnlySolutionReportsIncomplete) {
  variable("x"); variable("y"); variable("z");
  pair(0,1,0b1001); pair(1,2,0b1001); pair(0,2,0b1110);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  SolverOptions options; options.exactCombinationLimit=1; options.beamWidth=1;
  std::string errors;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream(errors) << d; return success();
  });
  SmallVector<RelaxationRecord> records;
  LayoutSolveFailure why;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,options,records,&why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::SearchIncomplete);
  EXPECT_TRUE(records.empty());
  EXPECT_NE(errors.find("search-incomplete"),std::string::npos) << errors;
  EXPECT_EQ(errors.find("no feasible"),std::string::npos);
  options.beamWidth=8;
  auto s = solveLayoutGraph(graph,target,options);
  ASSERT_TRUE(succeeded(s));
  EXPECT_EQ(s->instructionBindings.size(),3u);
  for (auto entry : s->assignments) EXPECT_EQ(entry.second,builder.getStringAttr("b"));
  EXPECT_TRUE(s->components[0].exhaustive);
}

TEST_F(LayoutCandidateSolverTest, ExhaustedInconsistentRelationsAreNotBudgetFailure) {
  variable("x"); variable("y"); variable("z");
  pair(0,1,0b1001); pair(1,2,0b1001); pair(0,2,0b0110);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  std::string errors;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream(errors) << d; return success();
  });
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target)));
  EXPECT_NE(errors.find("no feasible proven"),std::string::npos);
  EXPECT_EQ(errors.find("search-incomplete"),std::string::npos);
}

TEST_F(LayoutCandidateSolverTest, DomainProductOverflowDoesNotBecomeExact) {
  for (unsigned i=0;i<65;++i) {
    variable("v"+std::to_string(i)); if (i) same(i-1,i);
  }
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_TRUE(s->components[0].productOverflow);
  EXPECT_EQ(s->components[0].combinations,UINT64_MAX);
  EXPECT_TRUE(s->components[0].usedBeam);
  EXPECT_EQ(s->assignments.size(),65u);
}

TEST_F(LayoutCandidateSolverTest, RawOverflowCanPruneToExactSingleton) {
  for (unsigned i=0;i<65;++i) {
    variable("v"+std::to_string(i)); if (i) same(i-1,i);
    graph.addConstraint(ConstraintKind::RequireEncoding,ConstraintStrength::Hard,
        {i},nullptr,"require","hard seed",builder.getStringAttr("b"));
  }
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_TRUE(s->components[0].rawProductOverflow);
  EXPECT_FALSE(s->components[0].productOverflow);
  EXPECT_EQ(s->components[0].combinations,1u);
  EXPECT_FALSE(s->components[0].usedBeam);
  EXPECT_EQ(s->rejections.size(),65u);
}

TEST_F(LayoutCandidateSolverTest, FixedSeedIndependentExhaustiveOracleAndPermutation) {
  std::mt19937 random(230923);
  // Six binary variables = 64 complete assignments, independently enumerated.
  for (unsigned seed=0;seed<24;++seed) {
    graph=LayoutConstraintGraph();
    for (unsigned i=0;i<6;++i) variable("v"+std::to_string(i));
    SmallVector<unsigned> masks;
    for (unsigned i=0;i<6;++i) {
      unsigned mask=(random()%15)+1;
      masks.push_back(mask); pair(i,(i+1)%6,mask);
    }
    bool found=false; unsigned bestCost=99, best=0;
    for (unsigned bits=0;bits<64;++bits) {
      bool valid=true; unsigned cost=0;
      for (unsigned i=0;i<6;++i) {
        unsigned x=(bits>>(5-i))&1, y=(bits>>(5-(i+1)%6))&1;
        valid &= (masks[i] & (1u<<(2*x+y))) != 0;
        cost += !x;
      }
      if (valid && (!found || cost<bestCost)) {found=true;bestCost=cost;best=bits;}
    }
    ASSERT_TRUE(succeeded(graph.finalize(loc)));
    ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
    for (bool beam : {false,true}) {
      auto copy=graph;
      for (auto &v : copy.getVariables()) std::shuffle(v.candidates.begin(),v.candidates.end(),random);
      SolverOptions options; options.exactCombinationLimit=beam?1:4096;
      options.beamWidth=4096;
      auto s=solveLayoutGraph(copy,target,options);
      ASSERT_EQ(succeeded(s),found) << seed;
      if (!found) continue;
      EXPECT_EQ(s->cost.cost.instructionPathAndWork,bestCost) << seed;
      for (unsigned i=0;i<6;++i)
        EXPECT_EQ(s->assignments.lookup(i),builder.getStringAttr((best>>(5-i))&1?"b":"a")) << seed;
      EXPECT_TRUE(s->components[0].exhaustive);
    }
  }
}

class PhysicalSolverTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  RankedTensorType type = RankedTensorType::get({64}, b.getF32Type());
  class Target : public LayoutTarget {
  public:
    Attribute preferred;
    bool latePreference = false, permissive = false, saturatingFixed = false;
    void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
    LogicalResult verifyCandidate(const LayoutVar &v, Attribute a, Location loc) const override {
      if (permissive) return success();
      auto e = dyn_cast<DistributedEncodingAttr>(a);
      return e ? e.verifyForType(cast<ShapedType>(v.shapedType),loc) : failure();
    }
    FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
      CostVector cost;
      if (preferred && a.values.lookup(0) != preferred) {
        if (latePreference) cost.spillRiskAndRegisters = 100;
        else cost.instructionPathAndWork = 100;
      }
      return cost;
    }
    FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &g,
                                   const CandidateAssignment &a) const override {
      if (!saturatingFixed) return LayoutTarget::evaluate(g,a);
      CostEstimate cost;
      for (auto [id,value] : a.values) {
        auto name = g.getVariable(id).stableName;
        if (name == "fixed") cost = addLayoutCosts(cost,{{UINT64_MAX},false});
        if (name == "producer" && value != preferred)
          cost = addLayoutCosts(cost,{{100},false});
      }
      return cost;
    }
    LayoutProof verifyInstructionContract(const LayoutConstraintGraph &,
        const LayoutConstraint &c, ArrayRef<Attribute> values, Attribute binding) const override {
      for (auto &t : c.instruction->tuples)
        if (ArrayRef<Attribute>(t.encodings) == values && t.binding == binding)
          return {ProofStatus::Proven,{},"finite member"};
      return {ProofStatus::Disproven,{},"not a finite member"};
    }
  } target;
  PhysicalSolverTest() {
    context.loadDialect<FriskDialect>(); context.allowUnregisteredDialects();
  }
  DistributedEncodingAttr encoding(bool rotate=false, bool replicate=false) {
    SmallVector<APInt> matrix;
    for (unsigned i=0;i<6;++i)
      for (unsigned j=0;j<6;++j) matrix.emplace_back(1,j==(rotate?(i+1)%6:i));
    auto map=BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("register"),b.getStringAttr("lane")}),
        b.getDenseI64ArrayAttr({1,5}),b.getArrayAttr({b.getStringAttr("d")}),
        b.getDenseI64ArrayAttr({6}),
        DenseIntElementsAttr::get(RankedTensorType::get({6,6},b.getI1Type()),matrix));
    return DistributedEncodingAttr::get(&context,map,
        b.getDenseI64ArrayAttr({2,32,replicate?2:1,1,1}),b.getI64IntegerAttr(replicate?2:1));
  }
  LayoutConstraintGraph make(Block &block, ArrayRef<Attribute> sources,
                             Attribute destination, bool existing=false) {
    auto value=block.addArgument(type,b.getUnknownLoc());
    OperationState state(b.getUnknownLoc(),"test.use"); state.addOperands(value);
    auto *op=Operation::create(state); block.push_back(op);
    LayoutConstraintGraph graph;
    auto p=graph.addVariable(LayoutKind::Distributed,type,"producer");
    auto u=graph.addVariable(LayoutKind::Distributed,type,"use");
    graph.getVariable(p).value=value;
    graph.getVariable(u).use=&op->getOpOperand(0);
    for (auto a:sources) graph.getVariable(p).candidates.push_back({a,kInvalidProvenanceID,99});
    graph.getVariable(u).candidates.push_back({destination,kInvalidProvenanceID,0});
    auto id=graph.addConstraint(ConstraintKind::Convertible,ConstraintStrength::Hard,
        {p,u},op,"convert","actual use");
    graph.getConstraints()[id].use=&op->getOpOperand(0);
    graph.getConstraints()[id].stableUseKey="use/0";
    graph.getConstraints()[id].existingConversion=existing;
    return graph;
  }
};

TEST_F(PhysicalSolverTest, RefinementPricesFullConversionAndRebuildsSelectedPlan) {
  Attribute a = encoding(), z = encoding(true);
  if (layoutCandidateKey(z) < layoutCandidateKey(a)) std::swap(a,z);
  for (bool late : {false,true}) {
    Block block;
    auto graph = make(block,{a,z},a);
    auto dummy = graph.addVariable(LayoutKind::Distributed,type,"z");
    graph.getVariable(dummy).candidates = {{a},{z}};
    auto host = OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
    auto cid = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard,{0,dummy,0,dummy},*host,"connect","full support");
    auto &c = graph.getConstraints()[cid]; c.instruction = InstructionLayoutContract{*host,{}, {}};
    for (auto x : {a,z}) for (auto y : {a,z})
      c.instruction->tuples.push_back({{x,y,x,y}, b.getStringAttr("plan")});
    target.preferred = z; target.latePreference = late;
    auto result = solveLayoutGraph(graph,target,{1,1,65536});
    ASSERT_TRUE(succeeded(result));
    EXPECT_EQ(result->assignments.lookup(0),late ? a : z);
    EXPECT_EQ(result->components.front().refinementAccepted,late ? 0u : 1u);
    EXPECT_EQ(result->cost.cost.conversionBytesAndSync,late ? 0u : 512u);
    EXPECT_EQ(result->conversions.size(),late ? 0u : 1u);
    if (!late) EXPECT_EQ(result->conversions.front().sourceEncoding,z);
  }
}

TEST_F(PhysicalSolverTest, SaturatedFallbackRebuildsOriginalConversionAndBindingPlan) {
  Attribute a = encoding(), z = encoding(true);
  if (layoutCandidateKey(z) < layoutCandidateKey(a)) std::swap(a,z);
  target.preferred = z; target.saturatingFixed = true;
  for (unsigned mode=0;mode<4;++mode) {
    Block block;
    auto graph = make(block,{a,z},a);
    auto dummy = graph.addVariable(LayoutKind::Distributed,type,"z");
    graph.getVariable(dummy).candidates = {{a},{z}};
    auto fixed = graph.addVariable(LayoutKind::Distributed,type,"fixed");
    graph.getVariable(fixed).candidates = {{a}};
    auto host = OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
    auto cid = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard,{0,dummy,0,dummy},*host,"connect","full support");
    auto &c = graph.getConstraints()[cid]; c.instruction = InstructionLayoutContract{*host,{}, {}};
    for (auto x : {a,z}) for (auto y : {a,z})
      c.instruction->tuples.push_back({{x,y,x,y},x});
    LayoutTrace trace(mode == 1 ? LayoutTraceOptions{} :
                      LayoutTraceOptions{"unused.json",mode == 3 ? 1u : 4096u,mode == 3 ? 4096u : 8388608u});
    auto result = solveLayoutGraph(graph,target,{1,1,65536,mode != 0,256},nullptr,nullptr,&trace);
    ASSERT_TRUE(succeeded(result));
    EXPECT_TRUE(result->cost.saturated);
    EXPECT_EQ(result->cost.cost.instructionPathAndWork,UINT64_MAX);
    EXPECT_EQ(result->cost.cost.conversionBytesAndSync,0u);
    EXPECT_TRUE(result->conversions.empty());
    EXPECT_EQ(result->instructionBindings.begin()->second,a);
    auto &component = result->components.back();
    EXPECT_EQ(component.refinementAccepted,mode ? 1u : 0u);
    EXPECT_EQ(component.refinementGlobalFallback,mode != 0);
    EXPECT_EQ(component.refinementEndCost.cost.conversionBytesAndSync,mode ? 512u : 0u);
    EXPECT_EQ(component.cost.cost.conversionBytesAndSync,0u);
    for (auto &v : graph.getVariables())
      if (v.stableName == "producer") EXPECT_EQ(result->assignments.lookup(v.id),a);
    EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,*result,target,b.getUnknownLoc())));
    if (!trace.enabled()) continue;
    auto json = llvm::json::parse(trace.serialize()); ASSERT_TRUE(bool(json));
    EXPECT_EQ(trace.omittedCost().cost.conversionBytesAndSync,0u);
    for (auto &value : *json->getAsObject()->getArray("events")) {
      auto *event = value.getAsObject();
      if (event->getString("event") == "cost-contribution")
        EXPECT_NE(event->getString("kind"),"conversion");
      if (event->getString("event") == "solution")
        EXPECT_TRUE(event->getArray("conversions")->empty());
    }
  }
}

TEST_F(PhysicalSolverTest, RefinementUnknownNeighborIsNotLegalAndCachedTraceRetainsStatus) {
  LayoutConstraintGraph graph;
  auto a = encoding(); auto opaque = b.getStringAttr("opaque");
  auto x = graph.addVariable(LayoutKind::Distributed,type,"x");
  auto y = graph.addVariable(LayoutKind::Distributed,type,"y");
  graph.getVariable(x).candidates = {{a},{opaque}};
  graph.getVariable(y).candidates = {{a}};
  auto cid = graph.addConstraint(ConstraintKind::TransformLayout,
      ConstraintStrength::Hard,{x,y},nullptr,"identity","unknown source decoder");
  graph.getConstraints()[cid].coordinateTransform = b.getDenseI64ArrayAttr({0});
  target.permissive = true; target.preferred = opaque;
  LayoutTrace trace({"unused.json"});
  auto result = solveLayoutGraph(graph,target,{1,1,65536},nullptr,nullptr,&trace);
  ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->assignments.lookup(x),a);
  EXPECT_EQ(result->components.front().refinementProbes,1u);
  EXPECT_EQ(result->components.front().refinementAccepted,0u);
  auto json = llvm::json::parse(trace.serialize()); ASSERT_TRUE(bool(json));
  unsigned unknown = 0;
  for (auto &value : *json->getAsObject()->getArray("events")) {
    auto *event = value.getAsObject();
    if (event->getString("event") != "refinement-probe") continue;
    EXPECT_EQ(event->getString("status"),"Unknown");
    EXPECT_EQ(event->getBoolean("legal"),false); ++unknown;
  }
  EXPECT_EQ(unknown,1u);
}

TEST_F(PhysicalSolverTest, BetterMainComputationBeatsZeroConversions) {
  Block block;
  auto a=encoding(), z=encoding(true);
  auto graph=make(block,{a,z},a);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  target.preferred=z;
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_EQ(s->assignments.lookup(0),z);
  ASSERT_EQ(s->conversions.size(),1u);
  EXPECT_EQ(s->cost.cost.instructionPathAndWork,0u);
  EXPECT_EQ(s->cost.cost.conversionBytesAndSync,512u); // Proven warp shuffle, no CTA barrier.
  EXPECT_EQ(s->conversions[0].bytes,512u);
  EXPECT_EQ(s->conversions[0].synchronizationCost,0u);
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,*s,target,b.getUnknownLoc())));
}

TEST_F(PhysicalSolverTest, EqualMainCostChoosesLowerConversionBytes) {
  Block block;
  auto a=encoding(), z=encoding(true), replicated=encoding(true,true);
  auto graph=make(block,{replicated,z},a);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_EQ(s->assignments.lookup(0),z);
  EXPECT_EQ(s->cost.cost.conversionBytesAndSync,512u);
}

TEST_F(PhysicalSolverTest, ExistingConversionChargedOnceWithoutReinsertion) {
  Block block;
  auto a=encoding(), z=encoding(true);
  auto graph=make(block,{z},a,true);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_TRUE(s->conversions.empty());
  EXPECT_EQ(s->cost.cost.conversionBytesAndSync,512u);
  EXPECT_EQ(s->cost.cost.codeSize,1u);
}
} // namespace
