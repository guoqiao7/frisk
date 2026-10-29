#include "SelectionExperiment.h"
#include "SelectionOracle.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::frisk::selection {
namespace {
std::string jsonText(llvm::json::Object object) {
  return llvm::formatv("{0:2}\n",llvm::json::Value(std::move(object))).str();
}
bool sameCost(const CostEstimate &a,const CostEstimate &b) {
  return !(a.cost<b.cost) && !(b.cost<a.cost) && a.saturated==b.saturated;
}
}

void runControlledExperiment(ModuleOp module,const ExperimentOptions &o,
                             ExperimentResult &r,std::string &diagnostics) {
  auto fail=[&](StringRef kind,StringRef why) {
    auto f=classifyFailure(kind,why,diagnostics);
    r.status="failed";r.failureKind=f.kind;r.error=f.reason;r.fatal=f.fatal;
  };
  auto path=[&](StringRef file){return o.outputDir+"/"+file.str();};
  auto save=[&](StringRef file,StringRef text) {
    std::error_code ec;llvm::raw_fd_ostream out(path(file),ec);
    if(ec) {fail("artifact-write",ec.message());return false;}
    out<<text;out.close();
    if(out.has_error()) {auto error=out.error().message();out.clear_error();fail("artifact-write",error);return false;}
    return true;
  };
  bool fixed=o.spec.consumer==Consumer::Fixed0 || o.spec.consumer==Consumer::Fixed1;
  bool local=o.phase=="local";
  if(o.spec.policy!="Auto" || (local!=(o.spec.consumer==Consumer::Local))) {
    fail("invalid-options","controlled/local requires Auto and matching consumer scope");return;
  }
  // A fixed ConvertLayout consumer deliberately has no source encoding yet.
  // Normalize verifies its input and therefore must not precede collection.
  if(!fixed) {
    PassManager normalization(module.getContext());normalization.addPass(createNormalizeLayoutIRPass());
    if(failed(normalization.run(module))) {fail("normalization",diagnostics);return;}
  }
  if(!save("preparation-input.mlir",printModule(module))) return;
  r.record["preparation"]=fixed ? "raw-pre-layout-analysis-fixture" : "production-normalize-layout-ir";
  r.record["native_comparability"]=llvm::json::Object{{"status",fixed || local ? "not-comparable" : "pending-native-comparison"},
    {"reason",fixed ? "raw fixed consumer is not a native pipeline input" :
      "same normalization and initial preparation; native final attempt identity must be matched separately (no relaxed-domain assumption)"}};
  auto target=createSM90LayoutTarget();LayoutStatisticsReport stats;
  stats.costModelID=normalizeLayoutCostModelID(target->costModelID()).str();
  LayoutTraceOptions traceOptions;traceOptions.file=path("trace.json");LayoutTrace trace(traceOptions);trace.setSolverOptions({});
  auto initialSummary=trace.getSummary();initialSummary.target="sm90";
  initialSummary.costCoverage=target->costCoverage().str();initialSummary.costModelID=stats.costModelID;
  trace.setSummary(initialSummary);
  auto preserveFailure=llvm::make_scope_exit([&] {
    if(r.status!="failed") return;
    stats.status="failed";stats.failureKind=r.failureKind;stats.failureReason=r.error;
    r.record["statistics"]=stats.toJSON();
    auto summary=trace.getSummary();summary.analysisStatus="failed";
    summary.failureKind=r.failureKind;summary.failureReason=r.error;summary.hasFinalCost=false;
    trace.setSummary(summary);std::string error;
    if(failed(writeLayoutStatistics(path("statistics.json"),stats,error)) || failed(trace.publish(error)))
      fail("artifact-write",error);
  });
  auto prepared=collectLayoutConstraints(module,*target,LayoutCollectionMode::InitializeCandidates,&stats,&trace);
  if(failed(prepared)) {fail("collection",diagnostics);return;}
  LayoutTraceOptions snapshotOptions;snapshotOptions.file="in-memory-snapshot";
  LayoutTrace snapshot(snapshotOptions);snapshot.beginAttempt("initial");
  snapshot.recordGraph(*prepared);snapshot.recordDomains(*prepared,"prepared");
  auto snapshotJSON=llvm::json::parse(snapshot.serialize());
  if(!snapshotJSON) {fail("native-snapshot-projection",llvm::toString(snapshotJSON.takeError()));return;}
  auto projection=preparedTraceProjection(*snapshotJSON->getAsObject(),1);
  if(failed(projection)) {fail("native-snapshot-projection","prepared snapshot incomplete");return;}
  auto projectionText=jsonText(std::move(*projection));
  if(!save("native-snapshot-projection.json",projectionText)) return;
  r.record["native_snapshot_projection"]=llvm::json::Object{{"sha256",sha256(projectionText)},
    {"artifact","native-snapshot-projection.json"},{"scope","unrestricted-controlled-initial-preparation"}};
  auto proofs=freezeRelationProofs(*prepared);
  auto frozenText=jsonText(domainJSON(*prepared,proofs));
  auto frozenHash=sha256(frozenText);
  if(!save("domains.json",frozenText)) return;
  r.record["frozen_domain"]=llvm::json::Object{{"sha256",frozenHash},{"artifact","domains.json"},
    {"stage","prepared-before-propagation"}};
  auto branch=restrictGeometry(*prepared,o.restrictGrid,module.getLoc());
  if(failed(branch)) {fail("invalid-restriction","geometry is not in the prepared Auto domain or graph invariant failed");return;}
  auto branchText=jsonText(domainJSON(*branch,proofs));
  if(!save("branch-domains.json",branchText)) return;
  llvm::json::Array geometries;
  for(const auto &v:prepared->getVariables()) if(v.kind==LayoutKind::Instruction)
    for(const auto &c:v.candidates) if(auto grid=geometryGrid(c.value))
      geometries.push_back(llvm::json::Object{{"grid",llvm::json::Array{grid->first,grid->second}},
        {"attribute",[&]{std::string s;llvm::raw_string_ostream(s)<<c.value;return s;}()}});
  r.record["frozen_domain"]=llvm::json::Object{{"sha256",frozenHash},{"restricted_sha256",sha256(branchText)},
    {"artifact","domains.json"},{"branch_artifact","branch-domains.json"},
    {"geometry_candidates",std::move(geometries)},{"stage","prepared-before-propagation"}};
  r.record["branch"]=o.restrictGrid ? std::to_string(o.restrictGrid->first)+","+std::to_string(o.restrictGrid->second) : "joint";
  r.record["solver_options"]=llvm::json::Object{{"exact_combination_limit",256},{"beam_width",32},
    {"max_expanded_states",65536},{"max_refinement_probes",256}};

  // Run the oracle on the unpropagated restricted snapshot, not the default
  // solver's pruned domains. Both graphs borrow the still-unmodified module.
  std::optional<OracleResult> oracle;
  if(o.oracle) {
    oracle=enumerateOracle(*branch,proofs,*target);
    auto record=oracle->toJSON();record["frozen_domain_sha256"]=frozenHash;
    record["restricted_domain_sha256"]=sha256(branchText);
    if(oracle->best) {
      record["assignment"]=assignmentJSON(*branch,oracle->best->assignments);
      record["conversions"]=conversionsJSON(*branch,oracle->best->conversions);
      record["existing_conversions"]=existingConversionsJSON(*branch,CandidateAssignment{oracle->best->assignments});
      if(failed(verifySolvedLayoutGraph(*branch,*oracle->best,*target,module.getLoc()))) {
        fail("oracle-solved-graph-verification",diagnostics);return;
      }
    }
    r.record["oracle"]=llvm::json::Object(record);
    if(!save("oracle.json",jsonText(std::move(record)))) return;
    if(oracle->fatal) {fail("oracle-correctness",oracle->reason);return;}
  }

  auto graph=*branch;LayoutSolveFailure why;
  stats.attempts.emplace_back();auto &attempt=stats.attempts.back();prepareLayoutStatistics(graph,attempt);
  trace.beginAttempt("controlled-frozen-initial");trace.recordGraph(graph);trace.recordDomains(graph,"prepared");
  auto start=LayoutClock::now();
  auto propagated=propagateCommonToFixedPoint(graph,&trace);
  attempt.propagationTime=std::chrono::nanoseconds(layoutElapsedNS(start));
  if(failed(propagated)) {
    graph=*branch;
    auto diagnosis=solveLayoutGraph(graph,*target,{},&why,&attempt,&trace,true);
    if(succeeded(diagnosis)) {fail("unsupported","propagation rejected a solver-supported graph");return;}
    recordLayoutFailure(graph,why,stats);
    fail(stringifyLayoutSolveFailureKind(why.kind),why.reason);return;
  }
  auto solution=solveLayoutGraph(graph,*target,{},&why,&attempt,&trace);
  trace.finishAttempt(why,succeeded(solution) ? &*solution : nullptr);
  if(failed(solution)) {
    recordLayoutFailure(graph,why,stats);
    r.record["statistics"]=stats.toJSON();fail(stringifyLayoutSolveFailureKind(why.kind),why.reason);return;
  }
  if(failed(verifySolvedLayoutGraph(graph,*solution,*target,module.getLoc()))) {
    fail("solved-graph-verification",diagnostics);return;
  }
  CandidateAssignment selected{solution->assignments};
  auto scored=scoreAssignment(*branch,selected,*target);
  if(failed(scored) || !sameCost(scored->cost,solution->cost)) {
    fail("cost-failure","default total differs from exactly-once target plus conversion contributions");return;
  }
  r.record["cost_reconciled"]=true;r.record["cost"]=layoutCostJSON(scored->cost);
  r.record["cost_contributions"]=std::move(scored->contributions);
  r.record["assignment"]=assignmentJSON(*branch,solution->assignments);
  r.record["conversions"]=conversionsJSON(*branch,solution->conversions);
  r.record["existing_conversions"]=existingConversionsJSON(*branch,selected);
  llvm::json::Array selectedGrids;for(auto &g:assignmentGeometries(*branch,selected)) selectedGrids.push_back(g);
  r.record["selected_geometries"]=std::move(selectedGrids);
  r.record["oracle_comparison"]="not-requested";
  if(oracle) {
    if(oracle->status=="infeasible" || (oracle->status=="complete" &&
        (!oracle->best || solution->cost.cost<oracle->best->cost.cost))) {
      fail("default-below-complete-oracle","default contradicts exhaustive same-domain oracle");return;
    }
    r.record["oracle_comparison"]=oracle->status=="complete"
      ? (oracle->best->cost.cost<solution->cost.cost ? "search-gap" : "equal-optimum") : "incomplete-no-optimality-claim";
  }
  stats.status="solved";r.record["statistics"]=stats.toJSON();std::string error;
  if(failed(writeLayoutStatistics(path("statistics.json"),stats,error))) {fail("statistics-write",error);return;}
  auto summary=trace.getSummary();summary.target="sm90";summary.costCoverage=target->costCoverage().str();
  summary.analysisStatus="solved";summary.verificationStatus="solved-graph-verified";
  summary.hasFinalCost=true;summary.finalCost=solution->cost;trace.setSummary(summary);
  if(failed(trace.publish(error))) {fail("trace-write",error);return;}

  // The graph is never reused after mutation. Any alternate incumbent is
  // rebuilt against its own clone and domain identity is checked before use.
  auto materializeAndReplay=[&](ModuleOp m,const LayoutConstraintGraph &g,
      const LayoutSolution &s,StringRef prefix,bool main) {
    auto file=[&](StringRef suffix){return prefix.str()+suffix.str();};
    if(failed(materializeLayouts(m,g,s)) || failed(verify(m)) ||
        failed(verifyMaterializedLayouts(m,*target))) {
      fail("actual-verification",diagnostics);return false;
    }
    if(local) return save(file("output.mlir"),printModule(m));
    if(failed(verifyOutputABI(m,o.spec,error))) {fail("actual-verification",error);return false;}
    if(!save(file("materialized-pipeline-input.mlir"),printModule(m))) return false;
    PassManager pipeline(m.getContext());LayoutStatisticsReport pipelineStats;
    buildFriskLayoutPipeline(pipeline,{},path(file("pipeline-statistics.json")),&pipelineStats);
    if(failed(pipeline.run(m)) || failed(verify(m)) || failed(verifyMaterializedLayouts(m,*target)) ||
        failed(verifyOutputABI(m,o.spec,error))) {fail("actual-verification",error.empty() ? diagnostics : error);return false;}
    auto output=printModule(m);if(!save(file("output.mlir"),output)) return false;
    auto replay=parseSourceString<ModuleOp>(output,m.getContext());
    if(!replay) {fail("replay-parse",diagnostics);return false;}
    PassManager pipeline2(m.getContext());LayoutStatisticsReport replayStats;
    buildFriskLayoutPipeline(pipeline2,{},path(file("replay-statistics.json")),&replayStats);
    if(failed(pipeline2.run(*replay)) || failed(verify(*replay)) || failed(verifyMaterializedLayouts(*replay,*target)) ||
        failed(verifyOutputABI(*replay,o.spec,error))) {fail("replay-verification",error.empty() ? diagnostics : error);return false;}
    auto replayText=printModule(*replay);if(!save(file("replay.mlir"),replayText)) return false;
    if(output!=replayText) {fail("replay-instability","full pipeline changed printed/parsed output");return false;}
    if(main) {r.record["replay_stable"]=true;r.record["output_abi_verified"]=true;}
    return true;
  };
  bool reuseOracleWitness=oracle && oracle->best &&
      oracle->best->assignments==solution->assignments &&
      oracle->best->conversions.size()==solution->conversions.size() &&
      llvm::equal(oracle->best->conversions,solution->conversions,sameLayoutConversionPlan);
  if(oracle && oracle->best && !reuseOracleWitness) {
    OwningOpRef<ModuleOp> clone(cast<ModuleOp>(module->clone()));
    auto cloneGraph=collectLayoutConstraints(*clone,*target);
    if(failed(cloneGraph) || jsonText(domainJSON(*cloneGraph,freezeRelationProofs(*cloneGraph)))!=frozenText) {
      fail("clone-domain-identity","oracle materialization clone changed frozen domain identity");return;
    }
    auto cloneBranch=restrictGeometry(*cloneGraph,o.restrictGrid,clone->getLoc());
    if(failed(cloneBranch)) {fail("clone-domain-identity","restriction clone failed");return;}
    // Attribute assignments and IDs are stable; all SSA uses/conversion edges
    // must be rebuilt from the clone, never copied from the original graph.
    auto cloneScore=scoreAssignment(*cloneBranch,CandidateAssignment{oracle->best->assignments},*target);
    if(failed(cloneScore) || !sameCost(cloneScore->cost,oracle->best->cost)) {
      fail("clone-cost-identity","oracle clone cost changed");return;
    }
    auto cloneSolution=*oracle->best;cloneSolution.conversions=std::move(cloneScore->conversions);
    if(failed(verifySolvedLayoutGraph(*cloneBranch,cloneSolution,*target,clone->getLoc()))) {
      fail("oracle-clone-solved-graph-verification",diagnostics);return;
    }
    if(!materializeAndReplay(*clone,*cloneBranch,cloneSolution,"oracle-",false)) return;
  }
  if(!materializeAndReplay(module,graph,*solution,"",true)) return;
  if(oracle && oracle->best) {
    r.record["oracle_incumbent_actual_verified"]=true;
    auto *record=r.record.getObject("oracle");
    (*record)["incumbent_actual_verified"]=true;
    (*record)["incumbent_validation"]=reuseOracleWitness
      ? "reused-default-identical-assignment-and-conversion-plan" : "separate-clone-recollected-domain-identity-checked";
    (*record)["incumbent_artifact_prefix"]=reuseOracleWitness ? "" : "oracle-";
    (*record)["incumbent_scope"]=local ? "local-analysis-only" : "actual-IR-ABI-pipeline-replay";
    if(!save("oracle.json",jsonText(llvm::json::Object(*record)))) return;
  }
  // No graph access below this point: materialization invalidates IR pointers.
  llvm::json::Array bindings;unsigned converts=0;
  module.walk([&](ConvertLayoutOp){++converts;});
  module.walk([&](MmaOp op){
    auto contract=op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
    if(!contract) return;
    std::string text;llvm::raw_string_ostream(text)<<contract;
    llvm::json::Array grid;for(auto n:contract.getPayload().getAs<DenseI64ArrayAttr>("grid").asArrayRef()) grid.push_back(n);
    bindings.push_back(llvm::json::Object{{"contract",text},{"grid",std::move(grid)}});
  });
  if(bindings.size()!=1) {fail("missing-live-mma","expected one bound MMA");return;}
  r.record["bindings"]=std::move(bindings);r.record["actual_convert_layout_count"]=converts;
  r.status=local ? "local-analysis-only" : "success";
}
} // namespace mlir::frisk::selection
