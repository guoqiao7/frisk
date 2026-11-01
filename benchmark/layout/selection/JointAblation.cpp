#include "JointAblation.h"
#include <chrono>
#include "SelectionOracle.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
namespace mlir::frisk::selection {
namespace {
void solveFailure(llvm::json::Object &record,const LayoutSolveFailure &why) {
 StringRef kind="fatal-integrity";
 switch(why.kind){case LayoutSolveFailureKind::EmptyDomain:case LayoutSolveFailureKind::HardConflict:kind="no-feasible";break;case LayoutSolveFailureKind::Unsupported:kind="unsupported";break;case LayoutSolveFailureKind::SearchBudget:kind="search-budget";break;case LayoutSolveFailureKind::SearchIncomplete:kind="unknown";break;default:break;}
 record["failure_kind"]=kind;record["fatal"]=kind=="fatal-integrity";record["failure"]=why.reason;
}
bool complete(const LayoutSolution &s) {
  return llvm::all_of(s.components,[](const auto &c) {
    return c.exhaustive && c.refinementStopReason!="total-budget" &&
      c.refinementStopReason!="refinement-budget" && c.refinementStopReason!="shared-budget";
  });
}
llvm::json::Object counters(const SharedSearchBudget &b) {
  return llvm::json::Object{{"probes",b.used},{"refinement",b.refinementUsed},
    {"relationship",b.relationshipProbes},{"complete_scores",b.completeScores},{"refinement_probes",b.refinementProbes}};
}
}
FailureOr<LayoutSolution> solveAblation(LayoutConstraintGraph &graph,LayoutTarget &target,
    SelectionMode mode,SolverOptions options,llvm::json::Object &record) {
  using Clock=std::chrono::steady_clock;
  auto solveStart=Clock::now();
  auto elapsed=[](auto start){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();};
  int64_t stage1Ns=0,stage2Ns=0;
  auto timing=[&](){record["solve_timing_ns"]=llvm::json::Object{{"stage1",stage1Ns},{"stage2",stage2Ns},{"total",elapsed(solveStart)}};};
  SharedSearchBudget owned;auto &budget=options.sharedBudget?*options.sharedBudget:owned;
  options.sharedBudget=&budget;
  record["legal_return"]=false;record["search_complete"]=false;record["stage2_improved"]=false;
  if(!graph.isFinalized() || budget.limit>65536 || budget.refinementLimit>256) {
    record["failure"]="experiment requires finalized frozen graph and budgets <=65536/256";
    return failure();
  }
  record["conversion_policy"]=options.policy.conversion==ConversionPolicy::Independent?"independent":options.policy.conversion==ConversionPolicy::SharedOnly?"shared-only":"full";
  // All benchmark graphs are nonempty; use the first variable's context.
  if(graph.getVariables().empty())return failure();
  Location location=UnknownLoc::get(graph.getVariables().front().shapedType.getContext());
  LayoutSolveFailure why;
  if(mode==SelectionMode::Joint) {
    options.policy.scope.fullGraph=true;budget.stageLimit=budget.limit;budget.stageRefinementLimit=budget.refinementLimit;
    auto result=solveLayoutGraph(graph,target,options,&why);
    timing();
    record["total"]=counters(budget);
    if(failed(result)){solveFailure(record,why);return failure();}
    if(failed(verifySolvedLayoutGraph(graph,*result,target,location,options.policy.conversion)))return failure();
    record["legal_return"]=true;record["search_complete"]=complete(*result);return result;
  }
  record["local_scope_restriction"]="Distributed definitions and instruction geometry only; Storage producers unsupported";
  record["cost_model_id"]=target.costModelID();
  record["selection_mode"]=options.policy.scope.informedLocal?"informed-producer-first":"producer-first";
  record["local_objective"]=options.policy.scope.informedLocal?"own-operations-input-repairs-and-resources-v2":"restricted-producer-v1";
  record["local_auxiliary_policy"]=options.policy.scope.informedLocal?"optimize own instruction storage/use auxiliaries in stage one; stage two pins only declared producer definitions and geometry":"not included";
  auto common=graph;
  options.policy.scope.fullGraph=false;
  auto unsupported=experimentPolicyUnsupportedReason(graph,options.policy);
  if(!unsupported.empty()) {
    record["failure_kind"]="unsupported";record["fatal"]=false;record["failure"]=unsupported;
    record["local_support_policy"]="whole-domain incoming-repair preflight; no assignment filtering";
    record["total"]=counters(budget);return failure();
  }
  record["local_support_policy"]=options.policy.scope.informedLocal?
      "C includes own incoming repairs; O requires identity incoming repairs throughout frozen domain":"legacy restricted scope";
  budget.stageLimit=budget.limit/2;budget.stageRefinementLimit=budget.refinementLimit/2;
  auto stage1Start=Clock::now();
  auto first=solveLayoutGraph(graph,target,options,&why);
  stage1Ns=elapsed(stage1Start);
  record["stage1"]=counters(budget);
  if(failed(first)){solveFailure(record,why);record["total"]=counters(budget);return failure();}
  if(failed(verifySolvedLayoutGraph(graph,*first,target,location,options.policy.conversion)))return failure();
  llvm::json::Array contributions;
  auto witnessed=scoreExperimentAssignment(graph,CandidateAssignment{first->assignments},target,options.policy,nullptr,&contributions);
  if(failed(witnessed))return failure();
  auto stage1Local=*witnessed;
  record["scope_dependency_table"]=std::move(contributions);
  record["stage1_cost"]=layoutCostJSON(first->cost);
  record["stage1_assignment"]=assignmentJSON(graph,first->assignments);bool firstComplete=complete(*first);
  auto stageStart=budget;
  auto secondGraph=common;
  for(auto id:options.policy.scope.producers) {
    if(id>=secondGraph.getVariables().size())return failure();
    auto &v=secondGraph.getVariable(id);auto selected=first->assignments.lookup(id);
    llvm::erase_if(v.candidates,[&](const auto &c){return c.value!=selected;});
  }
  for(auto id:options.policy.scope.producerInstructions) {
    auto &c=secondGraph.getConstraints()[id];
    if(c.instruction) {
      llvm::erase_if(c.instruction->tuples,[&](const auto &t){
        for(auto [var,encoding]:llvm::zip_equal(c.vars,t.encodings))
          if((llvm::is_contained(options.policy.scope.producers,var) || secondGraph.getVariable(var).kind==LayoutKind::Instruction) && encoding!=first->assignments.lookup(var))return true;
        return false;
      });
      for(auto var:c.vars)if(secondGraph.getVariable(var).kind==LayoutKind::Instruction)
        llvm::erase_if(secondGraph.getVariable(var).candidates,[&](const auto &candidate){return candidate.value!=first->assignments.lookup(var);});
    }
  }
  options.policy.scope.fullGraph=true;budget.stageLimit=budget.limit;budget.stageRefinementLimit=budget.refinementLimit;
  auto full=scoreAssignment(common,CandidateAssignment{first->assignments},target,options.policy);
  if(failed(full))return failure();
  first->cost=full->cost;first->conversions=std::move(full->conversions);first->policy=options.policy;
  auto stage2Start=Clock::now();
  auto second=solveLayoutGraph(secondGraph,target,options,&why);
  stage2Ns=elapsed(stage2Start);timing();
  auto delta=budget;delta.used-=stageStart.used;delta.refinementUsed-=stageStart.refinementUsed;
  delta.relationshipProbes-=stageStart.relationshipProbes;delta.completeScores-=stageStart.completeScores;delta.refinementProbes-=stageStart.refinementProbes;
  record["stage2"]=counters(delta);record["total"]=counters(budget);
  if(failed(second) && why.kind!=LayoutSolveFailureKind::SearchBudget){solveFailure(record,why);return failure();}
  bool secondComplete=succeeded(second)&&complete(*second);
  bool improved=succeeded(second)&&second->cost.cost<first->cost.cost;
  auto result=improved?*second:*first;
  if(failed(verifySolvedLayoutGraph(common,result,target,location,options.policy.conversion)))return failure();
  if(options.policy.scope.informedLocal) {
    auto localPolicy=options.policy;localPolicy.scope.fullGraph=false;llvm::json::Array localRows;
    auto local=scoreExperimentAssignment(common,CandidateAssignment{result.assignments},target,localPolicy,nullptr,&localRows);
    if(failed(local))return failure();
    record["stage2_local_cost"]=layoutCostJSON(*local);
    record["stage2_local_contributions"]=std::move(localRows);
    record["local_cost_changed_during_completion"]=local->cost<stage1Local.cost ||
        stage1Local.cost<local->cost || local->saturated!=stage1Local.saturated;
  }
  record["stage2_improved"]=improved;record["legal_return"]=true;
  record["search_complete"]=firstComplete&&secondComplete;
  record["stage2_status"]=failed(second)?"budget-exhausted-witness-retained":secondComplete?"complete":"incomplete";
  return result;
}
}
