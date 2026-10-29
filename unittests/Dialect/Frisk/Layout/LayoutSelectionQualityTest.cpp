#include "SelectionExperiment.h"
#include "SelectionOracle.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/IR/Builders.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"
#include <set>

using namespace mlir;
using namespace mlir::frisk;
using namespace mlir::frisk::selection;
namespace {
struct SelectionQualityTest : testing::Test {
  MLIRContext context;
  std::string error;
  llvm::SmallString<128> temp;
  SelectionQualityTest() {
    context.disableMultithreading();
    EXPECT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-selection-test", temp)));
  }
  ~SelectionQualityTest() override { EXPECT_FALSE(bool(llvm::sys::fs::remove_directories(temp))); }
  std::string output(llvm::StringRef name) { return (llvm::Twine(temp)+"/"+name).str(); }
  void expectModelIdentity(const ExperimentResult &result, const ExperimentOptions &options) {
    auto expected=createSM90LayoutTarget()->costModelID().str();
    auto *statistics=result.record.getObject("statistics");
    ASSERT_TRUE(statistics);
    EXPECT_EQ(statistics->getString("cost_model_id"),expected);
    for (StringRef file : {"statistics.json","trace.json","result.json"}) {
      SCOPED_TRACE(file.str());
      auto buffer=llvm::MemoryBuffer::getFile(options.outputDir+"/"+file.str());
      ASSERT_TRUE(bool(buffer));
      auto json=llvm::json::parse((*buffer)->getBuffer());
      ASSERT_TRUE(bool(json));
      auto *object=json->getAsObject();
      ASSERT_TRUE(object);
      if (file=="result.json") object=object->getObject("statistics");
      ASSERT_TRUE(object);
      EXPECT_EQ(object->getString("cost_model_id"),expected);
    }
  }
};

TEST_F(SelectionQualityTest, ManifestIsExactlyEightShapesTwoModesFourPolicies) {
  auto cases = nativeCases(); ASSERT_EQ(cases.size(),64u);
  const int64_t dims[][4] = {{64,64,16,128},{128,128,16,256},{128,512,16,256},
    {256,128,16,256},{128,128,32,256},{256,256,16,512},{512,128,16,1024},{128,128,16,512}};
  std::set<std::string> identities;
  for (auto &c : cases) {
    ASSERT_EQ(c.id.size(),2u); unsigned i = c.id[1]-'0'; ASSERT_LT(i,8u);
    EXPECT_EQ(c.m,dims[i][0]); EXPECT_EQ(c.n,dims[i][1]);
    EXPECT_EQ(c.k,dims[i][2]); EXPECT_EQ(c.threads,dims[i][3]);
    EXPECT_TRUE(c.policy=="Auto" || c.policy=="Square" || c.policy=="FullRow" || c.policy=="FullCol");
    EXPECT_EQ(c.consumer,Consumer::Store);
    EXPECT_TRUE(identities.insert(c.id+modeName(c.mode).str()+c.policy).second);
    auto m = buildFixture(context,c,error); ASSERT_TRUE(m) << error;
    EXPECT_TRUE(succeeded(verify(*m))); EXPECT_TRUE(succeeded(verifyOutputABI(*m,c,error))) << error;
    unsigned mma=0,copy=0; m->walk([&](MmaOp op) { ++mma; EXPECT_EQ(op.getK(),c.k); });
    m->walk([&](CopyOp) { ++copy; }); EXPECT_EQ(mma,1u); EXPECT_EQ(copy,0u);
  }
}

TEST_F(SelectionQualityTest, CliRejectsInvalidAndIncompatibleOptions) {
  ExperimentOptions options;
  EXPECT_TRUE(succeeded(parseOptions({"--case","C1","--mode","ss","--policy","Auto",
      "--consumer","store","--phase","native","--output-dir",output("valid")},options,error))) << error;
  EXPECT_EQ(options.spec.m,128); EXPECT_EQ(options.spec.threads,256);
  for (std::vector<std::string> args : {
      std::vector<std::string>{"--case","C8"}, {"--mode","SS"}, {"--policy","auto"},
      {"--unknown","1"}, {"--consumer","fixed0"}, {"--consumer-grid","1,2"},
      {"--consumer","fixed0","--consumer-grid","0,2"},
      {"--consumer","fixed0","--consumer-grid","1,2,3"},
      {"--case","C1","--case","C2"}, {"--phase","controlled"}, {"--output-dir"}}) {
    error.clear(); EXPECT_TRUE(failed(parseOptions(args,options,error)));
    EXPECT_FALSE(error.empty());
  }
}

TEST_F(SelectionQualityTest, NativeStoreAndBothAxesFreeActuallyReplay) {
  for (auto mode : {MmaMode::SS,MmaMode::RS})
    for (auto consumer : {Consumer::Store,Consumer::Reduce0,Consumer::Reduce1}) {
      ExperimentOptions o; o.spec.mode=mode; o.spec.consumer=consumer;
      if (consumer==Consumer::Fixed0 || consumer==Consumer::Fixed1) o.spec.consumerGrid={{1,2}};
      o.outputDir=output(modeName(mode).str()+consumerName(consumer).str());
      o.command={"test","--case","C1"};
      auto result=runExperiment(context,o);
      ASSERT_EQ(result.status,"success") << result.error;
      expectModelIdentity(result,o);
      EXPECT_FALSE(result.fatal); EXPECT_EQ(result.record.getBoolean("replay_stable"),true);
      EXPECT_EQ(result.record.getBoolean("output_abi_verified"),true);
      for (llvm::StringRef file : {"input.mlir","output.mlir","replay.mlir","statistics.json","trace.json","result.json","source-manifest.txt"})
        EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/"+file.str())) << file.str();
      auto text=llvm::MemoryBuffer::getFile(o.outputDir+"/output.mlir"); ASSERT_TRUE(bool(text));
      auto module=parseSourceString<ModuleOp>((*text)->getBuffer(),&context); ASSERT_TRUE(module);
      unsigned mma=0; module->walk([&](MmaOp op) { ++mma; EXPECT_TRUE(op->hasAttr("frisk.mma_contract")); });
      EXPECT_EQ(mma,1u); EXPECT_TRUE(succeeded(verifyOutputABI(*module,o.spec,error))) << error;
    }
}

TEST_F(SelectionQualityTest, FixedAnalysisFixtureIsEmittedButNotMislabeledNativeSuccess) {
  for (auto mode : {MmaMode::SS,MmaMode::RS}) for (auto consumer : {Consumer::Fixed0,Consumer::Fixed1}) {
    ExperimentOptions o; o.spec.mode=mode;o.spec.consumer=consumer;o.spec.consumerGrid={{1,2}};
    o.outputDir=output(modeName(mode).str()+consumerName(consumer).str());
    auto r=runExperiment(context,o);
    EXPECT_EQ(r.status,"failed");EXPECT_EQ(r.failureKind,"unsupported-input-form") << r.error;
    EXPECT_FALSE(r.fatal);EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/input.mlir"));
    EXPECT_FALSE(llvm::sys::fs::exists(o.outputDir+"/output.mlir"));
    EXPECT_EQ(r.record.getString("input_kind"),"pre-layout-analysis-fixture");
    o.phase="emit-input";o.outputDir+="-emit";
    auto emitted=runExperiment(context,o);EXPECT_EQ(emitted.status,"input-emitted") << emitted.error;
    EXPECT_EQ(emitted.record.getString("input_verification"),"native-unverified");
  }
}

TEST_F(SelectionQualityTest, EmitOnlyAndLocalCannotPretendPipelineSuccessOrOverwrite) {
  ExperimentOptions o; o.phase="emit-input"; o.outputDir=output("emit");
  auto r=runExperiment(context,o); EXPECT_EQ(r.status,"input-emitted") << r.error;
  EXPECT_FALSE(llvm::sys::fs::exists(o.outputDir+"/output.mlir"));
  auto overwrite=runExperiment(context,o); EXPECT_TRUE(overwrite.fatal); EXPECT_EQ(overwrite.failureKind,"output-directory");
  o.phase="native"; o.spec.consumer=Consumer::Local; o.outputDir=output("local");
  auto local=runExperiment(context,o); EXPECT_EQ(local.status,"local-analysis-only") << local.error;
  EXPECT_NE(local.record.getBoolean("replay_stable"),true);
}

TEST_F(SelectionQualityTest, RejectsIllegalConsumerGeometryWithoutPinningMma) {
  CaseSpec c; c.consumer=Consumer::Fixed0; c.consumerGrid={{4,4}};
  EXPECT_FALSE(buildFixture(context,c,error)); EXPECT_FALSE(error.empty());
  c.consumerGrid={{1,2}}; auto module=buildFixture(context,c,error); ASSERT_TRUE(module) << error;
  module->walk([&](MmaOp op) { EXPECT_FALSE(op->hasAttr("frisk.mma_contract"));
    EXPECT_FALSE(cast<RankedTensorType>(op.getResult().getType()).getEncoding()); });
  unsigned converts=0; module->walk([&](ConvertLayoutOp) { ++converts; }); EXPECT_EQ(converts,1u);
}

TEST_F(SelectionQualityTest, ProofAndCostBugsAreFatalNotUnsupportedSamples) {
  for (llvm::StringRef kind : {"cost-failure","solved-graph-verification",
      "materialization-or-verification","actual-verification","replay-verification",
      "new-unclassified-internal-error"}) EXPECT_TRUE(isFatalFailure(kind)) << kind.str();
  for (llvm::StringRef kind : {"empty-domain","hard-conflict","unsupported",
      "unknown-proof","search-budget","search-incomplete"}) EXPECT_FALSE(isFatalFailure(kind)) << kind.str();
}

TEST_F(SelectionQualityTest, SharedNativeLocalClassifierKeepsInternalFailuresFatal) {
  for (llvm::StringRef kind : {"materialization-or-verification","solved-graph-verification",
      "pipeline-verification","cost-failure","statistics-write"}) {
    auto failure=classifyFailure(kind,"internal failure","unknown proof: test witness");
    EXPECT_EQ(failure.kind,kind); EXPECT_TRUE(failure.fatal);
  }
  for (llvm::StringRef kind : {"collection","local-collection"}) {
    auto unknown=classifyFailure(kind,"","unknown proof: hardware point budget");
    EXPECT_EQ(unknown.kind,"unknown-proof");EXPECT_FALSE(unknown.fatal);
    auto internal=classifyFailure(kind,"","graph finalization invariant failed");
    EXPECT_EQ(internal.kind,kind);EXPECT_TRUE(internal.fatal);
  }
  for (llvm::StringRef reason : {"selected layout proof failed",
      "propagation rejected a solver-supported graph"}) {
    // These are the exact Unsupported reasons from runAttempt(), shared by
    // production native inference and local low-level analysis.
    for (llvm::StringRef diagnostics : {"","unknown proof: diagnostic detail"}) {
      auto failure=classifyFailure("unsupported",reason,diagnostics);
      EXPECT_EQ(failure.kind,"unsupported");EXPECT_EQ(failure.reason,reason);
      EXPECT_TRUE(failure.fatal);
    }
  }
  EXPECT_FALSE(classifyFailure("unsupported","mandatory endpoint relation is not proven","").fatal);
}

TEST_F(SelectionQualityTest, RealPublicationFailureRefreshesReturnedJsonAndExitClassification) {
  ExperimentResult result;result.status="success";
  std::string target=output("result.json");
  ASSERT_FALSE(bool(llvm::sys::fs::create_directory(target)));
  EXPECT_TRUE(failed(publishResult(result,target)));
  EXPECT_EQ(result.status,"failed");EXPECT_EQ(result.failureKind,"result-write");EXPECT_TRUE(result.fatal);
  EXPECT_EQ(result.record.getString("status"),"failed");EXPECT_EQ(result.record.getBoolean("fatal"),true);
  auto *failure=result.record.getObject("failure");ASSERT_TRUE(failure);
  EXPECT_EQ(failure->getString("kind"),"result-write");EXPECT_FALSE(failure->getString("reason")->empty());
  // Same condition used by the actual CLI, with no injected callback/hook.
  EXPECT_EQ(result.status=="failed" ? (result.fatal ? 2 : 1) : 0,2);
}

TEST_F(SelectionQualityTest, ControlledCliAndRealFrozenOracleReplay) {
  ExperimentOptions o;
  ASSERT_TRUE(succeeded(parseOptions({"--phase","controlled","--oracle",
      "--restrict-grid","1,2","--output-dir",output("controlled")},o,error))) << error;
  auto r=runExperiment(context,o);
  ASSERT_EQ(r.status,"success") << r.error;
  expectModelIdentity(r,o);
  EXPECT_EQ(r.record.getBoolean("replay_stable"),true);
  EXPECT_EQ(r.record.getBoolean("output_abi_verified"),true);
  auto *domain=r.record.getObject("frozen_domain");ASSERT_TRUE(domain);
  EXPECT_TRUE(domain->getString("sha256"));
  auto *oracle=r.record.getObject("oracle");ASSERT_TRUE(oracle);
  EXPECT_EQ(oracle->getString("status"),"complete");
  EXPECT_TRUE(oracle->getBoolean("exhausted").value_or(false));
  EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/domains.json"));
  EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/materialized-pipeline-input.mlir"));
}

// A real finite zero-cost target isolates traversal boundaries from SM90 proof
// runtimes. SM90 integration tests below exercise actual scoring/materializing.
struct ZeroTarget : LayoutTarget {
  mutable unsigned evaluations=0;
  void enumerateCandidates(const LayoutVar &,SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &,Attribute,Location) const override {return success();}
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {++evaluations;return CostVector{};}
};
TEST_F(SelectionQualityTest, OracleExactCapEdgesZeroLimitsAndUnknownProofs) {
  Builder b(&context);LayoutConstraintGraph graph;ZeroTarget target;
  auto id=graph.addVariable(LayoutKind::Distributed,RankedTensorType::get({1},b.getF32Type()),"x");
  graph.getVariable(id).candidates={{b.getI64IntegerAttr(1)},{b.getI64IntegerAttr(2)}};
  auto proofs=freezeRelationProofs(graph);
  auto complete=enumerateOracle(graph,proofs,target,{3,2,120000});
  EXPECT_EQ(complete.status,"complete");EXPECT_TRUE(complete.exhausted);
  EXPECT_EQ(complete.visitedNodes,3u);EXPECT_EQ(complete.evaluatedAssignments,2u);
  EXPECT_EQ(target.evaluations,2u);
  auto nodes=enumerateOracle(graph,proofs,target,{2,2,120000});
  EXPECT_EQ(nodes.status,"oracle-incomplete");EXPECT_FALSE(nodes.exhausted);EXPECT_TRUE(nodes.best);
  auto evaluations=enumerateOracle(graph,proofs,target,{3,1,120000});
  EXPECT_EQ(evaluations.status,"oracle-incomplete");EXPECT_FALSE(evaluations.exhausted);EXPECT_TRUE(evaluations.best);
  for(auto limits:{OracleLimits{0,2,120000},OracleLimits{3,0,120000},OracleLimits{3,2,0}}) {
    auto zero=enumerateOracle(graph,proofs,target,limits);
    EXPECT_EQ(zero.status,"oracle-incomplete");EXPECT_FALSE(zero.exhausted);EXPECT_FALSE(zero.best);
  }
  auto tooLarge=enumerateOracle(graph,proofs,target,{65537,4096,120000});EXPECT_TRUE(tooLarge.fatal);
  graph.addConstraint(ConstraintKind::Ownership,ConstraintStrength::Hard,{id},nullptr,"unknown","missing metadata");
  auto unknown=enumerateOracle(graph,freezeRelationProofs(graph),target);
  EXPECT_EQ(unknown.status,"proof-incomplete");EXPECT_TRUE(unknown.exhausted);
  EXPECT_EQ(unknown.evaluatedAssignments,0u);EXPECT_EQ(unknown.unknownAssignments,2u);EXPECT_FALSE(unknown.best);
  graph.getVariable(id).candidates.clear();
  auto empty=enumerateOracle(graph,freezeRelationProofs(graph),target);
  EXPECT_EQ(empty.status,"infeasible");EXPECT_TRUE(empty.exhausted);EXPECT_EQ(empty.evaluatedAssignments,0u);
}

TEST_F(SelectionQualityTest, OracleRetainsEveryNineFieldTiedGeometry) {
  Builder b(&context);LayoutConstraintGraph graph;ZeroTarget target;
  auto id=graph.addVariable(LayoutKind::Instruction,RankedTensorType::get({1},b.getF32Type()),"geometry");
  for(auto grid:{std::pair<int64_t,int64_t>{1,2},{2,1}})
    graph.getVariable(id).candidates.push_back({b.getDictionaryAttr({
      b.getNamedAttr("kind",b.getStringAttr("sm90-mma-geometry")),
      b.getNamedAttr("grid",b.getDenseI64ArrayAttr({grid.first,grid.second}))})});
  auto r=enumerateOracle(graph,freezeRelationProofs(graph),target);
  EXPECT_EQ(r.status,"complete");EXPECT_EQ(r.optimalGeometries,(std::set<std::string>{"1,2","2,1"}));
}

TEST_F(SelectionQualityTest, OraclePairUnknownAndFatalCostFailureRemainDistinct) {
  Builder b(&context);LayoutConstraintGraph graph;ZeroTarget target;
  auto type=RankedTensorType::get({1},b.getF32Type());
  auto x=graph.addVariable(LayoutKind::Distributed,type,"x");
  auto y=graph.addVariable(LayoutKind::Distributed,type,"y");
  graph.getVariable(x).candidates={{b.getI64IntegerAttr(1)}};
  graph.getVariable(y).candidates={{b.getI64IntegerAttr(2)}};
  graph.addConstraint(ConstraintKind::TransformLayout,ConstraintStrength::Hard,{x,y},nullptr,"unknown-pair","missing transform decoder");
  auto proofs=freezeRelationProofs(graph);ASSERT_EQ(proofs.entries.size(),1u);
  EXPECT_EQ(proofs.entries.front().proof.status,ProofStatus::Unknown);
  auto unknown=enumerateOracle(graph,proofs,target);
  EXPECT_EQ(unknown.status,"proof-incomplete");EXPECT_TRUE(unknown.exhausted);EXPECT_FALSE(unknown.best);
  graph.addConstraint(ConstraintKind::RequireEncoding,ConstraintStrength::Hard,{x},nullptr,"contradiction","hard counterexample",b.getI64IntegerAttr(3));
  auto impossible=enumerateOracle(graph,freezeRelationProofs(graph),target);
  EXPECT_EQ(impossible.status,"infeasible");EXPECT_TRUE(impossible.exhausted);
  struct FailingCostTarget : ZeroTarget {
    FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {
      if(++evaluations==2) return failure();return CostVector{};
    }
  } failing;
  LayoutConstraintGraph costGraph;auto z=costGraph.addVariable(LayoutKind::Distributed,type,"z");
  costGraph.getVariable(z).candidates={{b.getI64IntegerAttr(1)},{b.getI64IntegerAttr(2)}};
  auto failedCost=enumerateOracle(costGraph,freezeRelationProofs(costGraph),failing);
  EXPECT_EQ(failedCost.status,"failed");EXPECT_TRUE(failedCost.fatal);EXPECT_FALSE(failedCost.exhausted);EXPECT_TRUE(failedCost.best);
}

TEST_F(SelectionQualityTest, NativeProjectionNeverMixesAttemptsOrAcceptsTruncation) {
  auto trace=llvm::json::parse(R"({"truncated":false,"events":[
    {"event":"graph","attempt":1,"tag":"first"},
    {"event":"domains","stage":"prepared","attempt":1},
    {"event":"graph","attempt":2,"tag":"second"},
    {"event":"domains","stage":"prepared","attempt":2}]})");
  ASSERT_TRUE(bool(trace));auto *object=trace->getAsObject();ASSERT_TRUE(object);
  auto second=preparedTraceProjection(*object,2);ASSERT_TRUE(succeeded(second));
  EXPECT_EQ(second->getObject("graph")->getString("tag"),"second");
  EXPECT_FALSE(second->getObject("graph")->get("attempt"));
  EXPECT_TRUE(failed(preparedTraceProjection(*object,3)));
  object->getArray("events")->pop_back();EXPECT_TRUE(failed(preparedTraceProjection(*object,2)));
  (*object)["truncated"]=true;EXPECT_TRUE(failed(preparedTraceProjection(*object,1)));
}

TEST_F(SelectionQualityTest, ControlledRestrictionCopiesOnlyGeometryAndRelatedTuples) {
  CaseSpec s;auto m=buildFixture(context,s,error);ASSERT_TRUE(m);
  auto target=createSM90LayoutTarget();auto graph=collectLayoutConstraints(*m,*target);ASSERT_TRUE(succeeded(graph));
  auto frozen=domainJSON(*graph,freezeRelationProofs(*graph));
  auto *variables=frozen.getArray("variables");ASSERT_TRUE(variables);
  unsigned aliasMetadata=0;
  for(const auto &value:*variables) if(auto *alias=value.getAsObject()->getObject("storage_alias")) {
    ++aliasMetadata;EXPECT_TRUE(alias->getString("view_to_root"));
    EXPECT_TRUE(alias->getString("root_type"));EXPECT_TRUE(alias->getInteger("upper_bit"));
  }
  EXPECT_GT(aliasMetadata,0u);
  auto branch=restrictGeometry(*graph,std::pair<int64_t,int64_t>{1,2},m->getLoc());ASSERT_TRUE(succeeded(branch));
  unsigned geometryCount=0;
  for(const auto &v:graph->getVariables()) {
    const auto &w=branch->getVariable(v.id);
    if(v.kind==LayoutKind::Instruction) {EXPECT_EQ(v.candidates.size(),2u);EXPECT_EQ(w.candidates.size(),1u);++geometryCount;}
    else {ASSERT_EQ(v.candidates.size(),w.candidates.size());for(unsigned i=0;i<v.candidates.size();++i) EXPECT_EQ(v.candidates[i].value,w.candidates[i].value);}
    EXPECT_EQ(v.anchor,w.anchor);EXPECT_EQ(v.use,w.use);EXPECT_EQ(v.requiredThreads,w.requiredThreads);
  }
  EXPECT_EQ(geometryCount,1u);
  EXPECT_EQ(llvm::json::Value(std::move(frozen)),llvm::json::Value(domainJSON(*graph,freezeRelationProofs(*graph))));
  EXPECT_TRUE(failed(restrictGeometry(*graph,std::pair<int64_t,int64_t>{4,4},m->getLoc())));
}

TEST_F(SelectionQualityTest, ControlledCoreCompleteOraclesAndRealFixedConsumerFees) {
  for(auto mode:{MmaMode::SS,MmaMode::RS})
    for(auto consumer:{Consumer::Store,Consumer::Reduce0,Consumer::Reduce1,Consumer::Fixed0,Consumer::Fixed1}) {
      if(mode==MmaMode::RS && consumer!=Consumer::Store) continue;
      std::vector<std::optional<std::pair<int64_t,int64_t>>> grids={std::nullopt};
      if(consumer==Consumer::Fixed0 || consumer==Consumer::Fixed1) grids={std::pair<int64_t,int64_t>{1,2},std::pair<int64_t,int64_t>{2,1}};
      for(auto grid:grids) {
        ExperimentOptions o;o.phase="controlled";o.oracle=true;o.spec.mode=mode;o.spec.consumer=consumer;o.spec.consumerGrid=grid;
        o.outputDir=output(modeName(mode).str()+consumerName(consumer).str()+(grid ? std::to_string(grid->first) : ""));
        auto r=runExperiment(context,o);ASSERT_EQ(r.status,"success")<<r.error;
        auto *oracle=r.record.getObject("oracle");ASSERT_TRUE(oracle);EXPECT_EQ(oracle->getString("status"),"complete");
        EXPECT_EQ(r.record.getBoolean("cost_reconciled"),true);EXPECT_EQ(r.record.getBoolean("replay_stable"),true);
        if(grid) {EXPECT_EQ(r.record.getString("input_kind"),"pre-layout-analysis-fixture");
          EXPECT_TRUE(r.record.getArray("cost_contributions"));}
      }
    }
}

TEST_F(SelectionQualityTest, ControlledLocalIsAnalysisOnlyAndCliRejectsDomainMixing) {
  ExperimentOptions o;
  ASSERT_TRUE(succeeded(parseOptions({"--phase","local","--oracle","--output-dir",output("local-oracle")},o,error)));
  auto r=runExperiment(context,o);ASSERT_EQ(r.status,"local-analysis-only")<<r.error;
  auto *oracle=r.record.getObject("oracle");ASSERT_TRUE(oracle);EXPECT_EQ(oracle->getString("status"),"complete");
  EXPECT_NE(r.record.getBoolean("replay_stable"),true);
  for(std::vector<std::string> args: {std::vector<std::string>{"--phase","controlled","--policy","Square"},
      {"--phase","native","--oracle"},{"--phase","local","--consumer","store"},
      {"--phase","controlled","--restrict-grid","1,2,3"}}) {
    args.insert(args.end(),{"--output-dir",output("invalid")});EXPECT_TRUE(failed(parseOptions(args,o,error)));
  }
}

TEST_F(SelectionQualityTest, ControlledAndNativePreparedSnapshotIdentityActuallyMatches) {
  ExperimentOptions o;o.outputDir=output("native-snapshot");
  auto native=runExperiment(context,o);ASSERT_EQ(native.status,"success")<<native.error;
  auto *nativeProjection=native.record.getObject("native_snapshot_projection");ASSERT_TRUE(nativeProjection);
  auto nativeHash=nativeProjection->getString("sha256");ASSERT_TRUE(nativeHash);
  o.phase="controlled";o.outputDir=output("controlled-snapshot");
  auto controlled=runExperiment(context,o);ASSERT_EQ(controlled.status,"success")<<controlled.error;
  auto *controlledProjection=controlled.record.getObject("native_snapshot_projection");ASSERT_TRUE(controlledProjection);
  EXPECT_EQ(controlledProjection->getString("sha256"),nativeHash);
  EXPECT_EQ(controlled.record.getString("input_sha256"),native.record.getString("input_sha256"));
}

TEST_F(SelectionQualityTest, ControlledExplicitConsumerBoundaryIsChargedExactlyOnce) {
  ExperimentOptions o;o.phase="controlled";o.spec.consumer=Consumer::Fixed0;
  o.spec.consumerGrid={{1,2}};o.restrictGrid={{2,1}};o.outputDir=output("explicit-cost");
  auto r=runExperiment(context,o);ASSERT_EQ(r.status,"success")<<r.error;
  auto *existing=r.record.getArray("existing_conversions");ASSERT_TRUE(existing);
  ASSERT_EQ(existing->size(),1u);auto *edge=existing->front().getAsObject();ASSERT_TRUE(edge);
  EXPECT_EQ(edge->getBoolean("identity"),false);
  auto *contributions=r.record.getArray("cost_contributions");ASSERT_TRUE(contributions);
  unsigned charges=0;
  for(const auto &value:*contributions) {auto *c=value.getAsObject();ASSERT_TRUE(c);
    if(c->getString("kind")=="conversion" && c->getInteger("owner")==edge->getInteger("constraint")) {
      ++charges;auto *cost=c->getObject("cost");ASSERT_TRUE(cost);
      auto *fields=cost->getArray("cost");ASSERT_TRUE(fields);EXPECT_GT((*fields)[3].getAsInteger().value_or(0),0);
    }
  }
  EXPECT_EQ(charges,1u);
}

TEST_F(SelectionQualityTest, ControlledInvalidRestrictionPreservesFrozenEvidence) {
  ExperimentOptions o;o.phase="controlled";o.restrictGrid={{4,4}};o.outputDir=output("invalid-grid");
  auto r=runExperiment(context,o);EXPECT_EQ(r.status,"failed");EXPECT_TRUE(r.fatal);
  EXPECT_EQ(r.failureKind,"invalid-restriction");
  expectModelIdentity(r,o);
  EXPECT_TRUE(r.record.getObject("frozen_domain"));
  EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/statistics.json"));
  EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/trace.json"));
  EXPECT_TRUE(llvm::sys::fs::exists(o.outputDir+"/domains.json"));
}
} // namespace
