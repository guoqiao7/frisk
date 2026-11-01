#include "JointAblation.h"
#include "SelectionOracle.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
using namespace mlir::frisk::selection;
namespace {
class JointAblationStages:public testing::Test {
protected:
 MLIRContext context;Builder b{&context};
 std::unique_ptr<LayoutTarget> target=createSM90LayoutTarget();
 JointAblationStages(){context.loadDialect<FriskDialect,func::FuncDialect,arith::ArithDialect>();}
 DistributedEncodingAttr encoding(bool rotate=false){
  SmallVector<APInt> bits;
  for(unsigned row=0;row<6;++row)for(unsigned col=0;col<6;++col)bits.emplace_back(1,col==(rotate?(row+1)%6:row));
  auto map=BitLinearLayoutMapAttr::get(&context,b.getArrayAttr({b.getStringAttr("register"),b.getStringAttr("lane")}),b.getDenseI64ArrayAttr({1,5}),b.getArrayAttr({b.getStringAttr("d")}),b.getDenseI64ArrayAttr({6}),DenseIntElementsAttr::get(RankedTensorType::get({6,6},b.getI1Type()),bits));
  return DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({2,32,1,1,1}),b.getI64IntegerAttr(1));
 }
};
TEST_F(JointAblationStages, ProducerAndOracleSharePolicyAndScope){
 auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%x:tensor<64xf32>)->tensor<64xf32>{
  %y=arith.mulf %x,%x : tensor<64xf32>
  return %y : tensor<64xf32> })mlir",&context);
 auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));
 SolverOptions options;options.policy.conversion=ConversionPolicy::Independent;
 for(auto &v:graph->getVariables()){
  v.candidates={{encoding(),kInvalidProvenanceID,0},{encoding(true),kInvalidProvenanceID,1}};
  if(v.value && isa<BlockArgument>(v.value))options.policy.scope.producers.push_back(v.id);
 }
 SharedSearchBudget budget;options.sharedBudget=&budget;llvm::json::Object record;
 auto result=solveAblation(*graph,*target,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));
 EXPECT_EQ(record.getBoolean("legal_return"),true);EXPECT_EQ(record.getBoolean("search_complete"),true);
 EXPECT_NE(record.getArray("scope_dependency_table"),nullptr);
 EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph,*result,*target,module->getLoc(),ConversionPolicy::Independent)));
 auto score=scoreAssignment(*graph,CandidateAssignment{result->assignments},*target,result->policy);ASSERT_TRUE(succeeded(score));EXPECT_FALSE(score->cost.cost<result->cost.cost);EXPECT_FALSE(result->cost.cost<score->cost.cost);
 EXPECT_LE(budget.used,budget.limit);EXPECT_EQ(budget.used,budget.relationshipProbes+budget.completeScores+budget.refinementProbes);
 auto oracle=enumerateOracle(*graph,freezeRelationProofs(*graph),*target,{},result->policy);EXPECT_EQ(oracle.status,"complete");ASSERT_TRUE(oracle.best);EXPECT_FALSE(result->cost.cost<oracle.best->cost.cost);
}
TEST_F(JointAblationStages, NoWitnessReportsFailure){
 auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%x:tensor<64xf32>)->tensor<64xf32>{return %x : tensor<64xf32>})mlir",&context);
 auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));
 for(auto &v:graph->getVariables())v.candidates={{encoding(),kInvalidProvenanceID,0}};
 SharedSearchBudget budget;budget.limit=1;SolverOptions options;options.sharedBudget=&budget;
 llvm::json::Object record;ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
 auto result=solveAblation(*graph,*target,SelectionMode::ProducerFirst,options,record);EXPECT_TRUE(failed(result));EXPECT_EQ(record.getBoolean("legal_return"),false);EXPECT_EQ(record.getString("failure_kind"),"search-budget");EXPECT_EQ(record.getBoolean("fatal"),false);
}
}

namespace {
class FiniteTarget : public LayoutTarget {
public:
 void enumerateCandidates(const LayoutVar &,SmallVectorImpl<LayoutCandidate> &)const override{}
 LogicalResult verifyCandidate(const LayoutVar &,Attribute a,Location)const override{return success(isa<StringAttr>(a));}
 FailureOr<CostVector> evaluate(const CandidateAssignment &)const override{return CostVector{};}
 FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &g,const CandidateAssignment &a,LayoutCostObserver *o)const override {
  CostEstimate total;
  for(auto [id,value]:a.values){CostEstimate c;if(g.getVariable(id).stableName=="consumer" && cast<StringAttr>(value).getValue()=="a")c.cost.instructionPathAndWork=10;total=addLayoutCosts(total,c);if(o)o->recordCost("registers",id,c);}
  return total;
 }
 FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &g,const CandidateAssignment &a)const override {
  auto consumer=llvm::find_if(g.getVariables(),[](const auto &v){return v.stableName=="consumer";});CostEstimate c;auto value=consumer==g.getVariables().end()?Attribute{}:a.values.lookup(consumer->id);if(value && cast<StringAttr>(value).getValue()=="a")c.cost.instructionPathAndWork=10;return c;
 }
 LayoutProof verifyInstructionContract(const LayoutConstraintGraph &,const LayoutConstraint &,ArrayRef<Attribute>,Attribute)const override{return {ProofStatus::Proven,{},"finite table"};}
};
class JointAblationFinite:public testing::Test {
protected:
 MLIRContext context;Builder b{&context};LayoutConstraintGraph graph;FiniteTarget target;
 OwningOpRef<ModuleOp> module{ModuleOp::create(b.getUnknownLoc())};
 LayoutVarID variable(StringRef name,unsigned n){auto id=graph.addVariable(LayoutKind::Distributed,RankedTensorType::get({4},b.getF32Type()),name);for(unsigned i=0;i<n;++i)graph.getVariable(id).candidates.push_back({b.getStringAttr(std::string(1,'a'+i)),kInvalidProvenanceID,n-i});return id;}
 void relation(){auto id=graph.addConstraint(ConstraintKind::InstructionContract,ConstraintStrength::Hard,{0,1,2,1},*module,"finite","finite table");auto &c=graph.getConstraints()[id];c.instruction=InstructionLayoutContract{*module,{}, {}};
  for(auto &p:graph.getVariable(0).candidates)for(auto &x:graph.getVariable(1).candidates)for(auto &y:graph.getVariable(2).candidates)
   if(cast<StringAttr>(x.value).getValue()!="a" || cast<StringAttr>(y.value).getValue()=="a")c.instruction->tuples.push_back({{p.value,x.value,y.value,x.value},p.value});
 }
};
TEST_F(JointAblationFinite, StageTwoExhaustionRetainsCompleteWitness){
 variable("producer",1);variable("consumer",2);variable("tail",3);relation();ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 SharedSearchBudget budget;budget.limit=30;SolverOptions options;options.sharedBudget=&budget;options.exactCombinationLimit=1;options.beamWidth=1;options.refineBeamSolution=false;options.policy.scope.producers={1};
 llvm::json::Object record;ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
 auto result=solveAblation(graph,target,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));
 EXPECT_EQ(record.getBoolean("legal_return"),true);EXPECT_EQ(record.getBoolean("search_complete"),false);EXPECT_EQ(record.getString("stage2_status"),"budget-exhausted-witness-retained");
 EXPECT_EQ(result->assignments.size(),3u);EXPECT_EQ(result->instructionBindings.size(),1u);
 EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,*result,target,module->getLoc())));
 EXPECT_EQ(budget.used,30u);EXPECT_EQ(budget.used,budget.relationshipProbes+budget.completeScores+budget.refinementProbes);
 EXPECT_LE(*record.getObject("stage1")->getInteger("probes"),15);
 EXPECT_EQ(*record.getObject("stage1")->getInteger("probes")+*record.getObject("stage2")->getInteger("probes"),30);
}
TEST_F(JointAblationFinite, PinsInstructionGeometry){
 variable("geometry",2);variable("consumer",2);variable("tail",2);relation();graph.getVariable(0).kind=LayoutKind::Instruction;ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 SolverOptions options;options.policy.scope.producers={1};options.policy.scope.producerInstructions={0};SharedSearchBudget budget;options.sharedBudget=&budget;
 llvm::json::Object record;auto result=solveAblation(graph,target,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));
 EXPECT_EQ(result->assignments.lookup(1),b.getStringAttr("b"));EXPECT_EQ(result->instructionBindings.lookup(0),b.getStringAttr("b"));
}
TEST_F(JointAblationFinite, OracleProducerTieUsesStableOrdinal){
 variable("producer",2);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));ExperimentPolicy policy;policy.scope.fullGraph=false;policy.scope.producers={0};
 auto oracle=enumerateOracle(graph,freezeRelationProofs(graph),target,{},policy);ASSERT_TRUE(oracle.best);EXPECT_EQ(oracle.best->assignments.lookup(0),b.getStringAttr("b"));
}
}

namespace {
TEST_F(JointAblationFinite, OracleRejectsTargetInvalidCheapCandidate){
 variable("producer",2);
 class Checked:public FiniteTarget {public:LogicalResult verifyCandidate(const LayoutVar &,Attribute a,Location)const override{return success(cast<StringAttr>(a).getValue()=="a");}} checked;
 auto oracle=enumerateOracle(graph,freezeRelationProofs(graph),checked);ASSERT_TRUE(oracle.best);EXPECT_EQ(oracle.best->assignments.lookup(0),b.getStringAttr("a"));EXPECT_EQ(oracle.evaluatedAssignments,1u);
}
}

namespace {
TEST_F(JointAblationFinite, RejectsUnfinalizedGraphBeforeScopeOrBudgetUse) {
 variable("zproducer",2);variable("aconsumer",2);variable("tail",2);relation();
 SolverOptions options;options.policy.scope.producers={0};options.policy.scope.producerInstructions={0};SharedSearchBudget budget;options.sharedBudget=&budget;llvm::json::Object record;
 EXPECT_TRUE(failed(solveAblation(graph,target,SelectionMode::ProducerFirst,options,record)));EXPECT_FALSE(graph.isFinalized());EXPECT_EQ(budget.used,0u);
 auto policy=options.policy;policy.scope.fullGraph=false;auto oracle=enumerateOracle(graph,freezeRelationProofs(graph),target,{},policy);EXPECT_EQ(oracle.status,"invalid-domain");EXPECT_EQ(oracle.evaluatedAssignments,0u);
}
TEST_F(JointAblationFinite, RejectsRaisedExperimentBudget) {
 variable("p",1);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 for(bool refinement:{false,true}){SharedSearchBudget budget;if(refinement)budget.refinementLimit=257;else budget.limit=65537;SolverOptions options;options.sharedBudget=&budget;llvm::json::Object record;EXPECT_TRUE(failed(solveAblation(graph,target,SelectionMode::Joint,options,record)));EXPECT_EQ(budget.used,0u);}
}
TEST_F(JointAblationStages, ActualScopeTableMatchesObservedCosts) {
 auto m=parseSourceString<ModuleOp>(R"mlir(func.func @f(%x:tensor<64xf32>)->tensor<64xf32>{%y=arith.mulf %x,%x : tensor<64xf32> return %y : tensor<64xf32>})mlir",&context);
 auto g=collectLayoutConstraints(*m,*target);ASSERT_TRUE(succeeded(g));SolverOptions options;
 for(auto &v:g->getVariables()){v.candidates={{encoding(),kInvalidProvenanceID,0}};if(v.value && isa<BlockArgument>(v.value))options.policy.scope.producers.push_back(v.id);}
 llvm::json::Object record;auto result=solveAblation(*g,*target,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));auto *rows=record.getArray("scope_dependency_table");ASSERT_TRUE(rows);
 bool included=false;uint64_t totalRegisters=0;
 for(auto &value:*rows){auto *row=value.getAsObject();ASSERT_TRUE(row);EXPECT_TRUE(row->getObject("cost"));EXPECT_TRUE(row->getString("owner_kind"));auto *deps=row->getArray("dependencies");ASSERT_TRUE(deps);for(auto &d:*deps){auto *dep=d.getAsObject();ASSERT_TRUE(dep);EXPECT_TRUE(dep->getString("stable_name"));EXPECT_TRUE(dep->getInteger("id"));}
 if(row->getBoolean("included")==true){included=true;totalRegisters+=uint64_t(*(*row->getObject("cost")->getArray("cost"))[4].getAsInteger());EXPECT_EQ(row->getString("kind"),"registers");EXPECT_EQ(deps->size(),1u);}}
 EXPECT_TRUE(included);EXPECT_EQ(totalRegisters,uint64_t(*(*record.getObject("stage1_cost")->getArray("cost"))[4].getAsInteger()));
}
}

namespace {
TEST_F(JointAblationFinite, FrozenProofsRejectDifferentDomain) {
 variable("p",2);auto proofs=freezeRelationProofs(graph);
 graph.getVariable(0).stableName="other";
 auto r=enumerateOracle(graph,proofs,target);EXPECT_TRUE(r.fatal);EXPECT_EQ(r.evaluatedAssignments,0u);
}
TEST_F(JointAblationFinite, FrozenProofsRejectDeletedCoverage) {
 variable("p",2);graph.addConstraint(ConstraintKind::RequireEncoding,ConstraintStrength::Hard,{0},*module,"test","coverage");
 graph.getConstraints().back().requiredEncoding=b.getStringAttr("a");
 auto proofs=freezeRelationProofs(graph);ASSERT_FALSE(proofs.entries.empty());proofs.entries.clear();
 auto r=enumerateOracle(graph,proofs,target);EXPECT_TRUE(r.fatal);EXPECT_EQ(r.evaluatedAssignments,0u);
}
}

namespace {
TEST_F(JointAblationFinite, FrozenCandidateCoverageIsRequired) {
 variable("p",2);auto proofs=freezeRelationProofs(graph,&target);ASSERT_EQ(proofs.candidates.size(),2u);proofs.candidates.pop_back();
 auto r=enumerateOracle(graph,proofs,target);EXPECT_TRUE(r.fatal);EXPECT_EQ(r.evaluatedAssignments,0u);
}
TEST_F(JointAblationFinite, FrozenProofProjectionPreservesIdentity) {
 variable("p",2);auto proofs=freezeRelationProofs(graph,&target);auto restricted=graph;restricted.getVariable(0).candidates.pop_back();
 auto projected=projectFrozenProofs(graph,restricted,proofs);ASSERT_TRUE(succeeded(projected));EXPECT_EQ(projected->commonDigest,proofs.commonDigest);EXPECT_NE(projected->domainDigest,proofs.domainDigest);
 auto r=enumerateOracle(restricted,*projected,target);EXPECT_FALSE(r.fatal);EXPECT_EQ(r.evaluatedAssignments,1u);
 restricted.getVariable(0).stableName="wrong";EXPECT_TRUE(failed(projectFrozenProofs(graph,restricted,proofs)));
}
}

namespace {
TEST_F(JointAblationFinite, StageTwoDoesNotPinNonproducerInstructionRoles) {
 variable("geometry",1);variable("consumer",2);variable("tail",1);relation();graph.getVariable(0).kind=LayoutKind::Instruction;
 auto &contract=*graph.getConstraints()[0].instruction;for(auto &t:contract.tuples)t.binding=t.encodings[1];
 ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));SolverOptions options;options.policy.scope.producers={*graph.lookupVariable("geometry")};options.policy.scope.producerInstructions={0};
 llvm::json::Object record;auto result=solveAblation(graph,target,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));
 EXPECT_EQ(result->assignments.lookup(*graph.lookupVariable("consumer")),b.getStringAttr("b"));EXPECT_EQ(result->instructionBindings.lookup(0),b.getStringAttr("b"));EXPECT_EQ(result->cost.cost.instructionPathAndWork,0u);
}
}

namespace {
TEST_F(JointAblationFinite, ProofProjectionCannotChangeDeclaredInstructionBinding) {
 variable("p",1);variable("x",1);variable("y",1);relation();auto proofs=freezeRelationProofs(graph);auto restricted=graph;
 restricted.getConstraints()[0].instruction->binding=b.getStringAttr("other");EXPECT_TRUE(failed(projectFrozenProofs(graph,restricted,proofs)));
 graph.getConstraints()[0].instruction->binding=b.getStringAttr("a");proofs=freezeRelationProofs(graph);restricted=graph;restricted.getConstraints()[0].instruction->binding={};EXPECT_TRUE(failed(projectFrozenProofs(graph,restricted,proofs)));
}
}

namespace {
TEST_F(JointAblationFinite, FrozenCandidateProofsRejectTargetValidityDrift) {
 variable("p",2);auto proofs=freezeRelationProofs(graph,&target);
 class Narrow:public FiniteTarget {public:LogicalResult verifyCandidate(const LayoutVar &,Attribute a,Location)const override{return success(cast<StringAttr>(a).getValue()=="a");}} narrow;
 auto result=enumerateOracle(graph,proofs,narrow);EXPECT_TRUE(result.fatal);EXPECT_EQ(result.evaluatedAssignments,0u);
}
}

namespace {
TEST_F(JointAblationFinite, LocalOracleKeepsCostTiedGeometrySetSeparateFromStableWinner) {
 auto id=graph.addVariable(LayoutKind::Instruction,RankedTensorType::get({4},b.getF32Type()),"geometry");
 for(int64_t n:{1,2}){auto a=b.getDictionaryAttr({b.getNamedAttr("kind",b.getStringAttr("sm90-mma-geometry")),b.getNamedAttr("grid",b.getDenseI64ArrayAttr({n,1}))});graph.getVariable(id).candidates.push_back({a,kInvalidProvenanceID,uint64_t(3-n)});}
 ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 class Any:public FiniteTarget {public:LogicalResult verifyCandidate(const LayoutVar &,Attribute,Location)const override{return success();}} any;
 ExperimentPolicy policy;policy.scope.fullGraph=false;policy.scope.producers={id};auto result=enumerateOracle(graph,freezeRelationProofs(graph),any,{},policy);ASSERT_TRUE(result.best);EXPECT_EQ(result.optimalGeometries.size(),2u);EXPECT_EQ(geometryGrid(result.best->assignments.lookup(id))->first,2);
}
}

namespace {
TEST_F(JointAblationFinite, NoFeasibleOutcomeIsTypedAndNonfatal) {
 variable("p",0);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));SolverOptions options;llvm::json::Object record;ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
 auto result=solveAblation(graph,target,SelectionMode::Joint,options,record);EXPECT_TRUE(failed(result));EXPECT_EQ(record.getString("failure_kind"),"no-feasible");EXPECT_EQ(record.getBoolean("fatal"),false);
}
}


namespace {
class OwnOperationCostTarget : public FiniteTarget {
public:
  SmallVector<std::pair<std::string,uint64_t>> rows;
  FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &,
      const CandidateAssignment &,LayoutCostObserver *observer)const override {
    CostEstimate total;
    for(auto [index,row]:llvm::enumerate(rows)) {
      CostEstimate charge;charge.cost.memoryTransactions=index+1;
      total=addLayoutCosts(total,charge);
      if(observer)observer->recordCost(row.first,row.second,charge);
    }
    return total;
  }
};
TEST_F(JointAblationFinite, InformedIncludesOwnAccessAndLocalStorageAuxiliary) {
  auto p=variable("producer",1),aux=variable("storage",1),future=variable("future",1);
  graph.getVariable(aux).kind=LayoutKind::Storage;
  OwningOpRef<ModuleOp> consumer{ModuleOp::create(b.getUnknownLoc())};
  auto instruction=graph.addConstraint(ConstraintKind::InstructionContract,
      ConstraintStrength::Hard,{p,aux,p,aux},*module,"own","producer instruction");
  graph.getConstraints()[instruction].instruction=InstructionLayoutContract{*module,{}, {}};
  auto access=graph.addConstraint(ConstraintKind::StorageAccess,ConstraintStrength::Hard,
      {p,aux},*module,"own","producer access");
  auto downstream=graph.addConstraint(ConstraintKind::StorageAccess,ConstraintStrength::Hard,
      {future,aux},*consumer,"future","consumer access");
  ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  p=*graph.lookupVariable("producer");aux=*graph.lookupVariable("storage");
  for(const auto &c:graph.getConstraints()) {
    if(c.instruction)instruction=c.id;
    if(c.kind==ConstraintKind::StorageAccess && graph.getProvenances()[c.provenance].source==*module)access=c.id;
    if(c.kind==ConstraintKind::StorageAccess && graph.getProvenances()[c.provenance].source==*consumer)downstream=c.id;
  }
  OwnOperationCostTarget own;own.rows={{"access",access},{"access",downstream}};
  CandidateAssignment assignment;for(const auto &v:graph.getVariables())assignment.values[v.id]=v.candidates.front().value;
  ExperimentPolicy restricted;restricted.scope.fullGraph=false;restricted.scope.producers={p};restricted.scope.producerInstructions={instruction};
  auto old=scoreExperimentAssignment(graph,assignment,own,restricted);ASSERT_TRUE(succeeded(old));EXPECT_EQ(old->cost.memoryTransactions,0u);
  auto informed=restricted;informed.scope.informedLocal=true;llvm::json::Array table;
  auto score=scoreExperimentAssignment(graph,assignment,own,informed,nullptr,&table);ASSERT_TRUE(succeeded(score));EXPECT_EQ(score->cost.memoryTransactions,1u);
  ASSERT_EQ(table.size(),2u);auto *row=table[0].getAsObject();ASSERT_TRUE(row);EXPECT_EQ(row->getBoolean("included"),true);
  auto *deps=row->getArray("dependencies");ASSERT_TRUE(deps);EXPECT_EQ(deps->size(),2u);
  EXPECT_EQ(table[1].getAsObject()->getBoolean("included"),false);
}
TEST_F(JointAblationFinite, InformedIncludesOwnRootAndRejectsNonlocalRootMaximum) {
  auto p=variable("producer",1),storage=variable("storage",1);
  graph.getVariable(storage).kind=LayoutKind::Storage;
  auto instruction=graph.addConstraint(ConstraintKind::InstructionContract,
      ConstraintStrength::Hard,{p,storage,p,storage},*module,"own","producer instruction");
  graph.getConstraints()[instruction].instruction=InstructionLayoutContract{*module,{}, {}};
  graph.addConstraint(ConstraintKind::StorageAccess,ConstraintStrength::Hard,
      {p,storage},*module,"own","producer access");
  ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));p=*graph.lookupVariable("producer");storage=*graph.lookupVariable("storage");for(const auto &c:graph.getConstraints())if(c.instruction)instruction=c.id;
  OwnOperationCostTarget own;own.rows={{"shared-root",storage}};
  CandidateAssignment assignment;for(const auto &v:graph.getVariables())assignment.values[v.id]=v.candidates.front().value;
  ExperimentPolicy policy;policy.scope.fullGraph=false;policy.scope.informedLocal=true;policy.scope.producers={p};policy.scope.producerInstructions={instruction};
  auto score=scoreExperimentAssignment(graph,assignment,own,policy);ASSERT_TRUE(succeeded(score));EXPECT_EQ(score->cost.memoryTransactions,1u);
  OwningOpRef<ModuleOp> future{ModuleOp::create(b.getUnknownLoc())};
  graph.addConstraint(ConstraintKind::StorageAccess,ConstraintStrength::Hard,
      {p,storage},*future,"future","consumer access");
  ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  EXPECT_TRUE(failed(scoreExperimentAssignment(graph,assignment,own,policy)));
}
TEST_F(JointAblationFinite, InformedSelectionAndIndependentOracleIncludeOwnArithmetic) {
  variable("producer",2);variable("consumer",2);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  class ArithmeticTarget:public FiniteTarget {
  public:
    FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &g,
        const CandidateAssignment &a,LayoutCostObserver *observer)const override {
      CostEstimate total;
      for(const auto &v:g.getVariables()) {
        auto selected=a.values.lookup(v.id);if(!selected)continue;
        CostEstimate c;c.cost.instructionPathAndWork=
            cast<StringAttr>(a.values.lookup(v.id)).getValue()=="a"?9:1;
        total=addLayoutCosts(total,c);if(observer)observer->recordCost("arithmetic",v.id,c);
      }
      return total;
    }
    FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &,const CandidateAssignment &)const override{return CostEstimate{};}
  } arithmetic;
  SolverOptions options;options.policy.scope.producers={*graph.lookupVariable("producer")};options.policy.scope.informedLocal=true;
  SharedSearchBudget budget;options.sharedBudget=&budget;llvm::json::Object record;
  auto result=solveAblation(graph,arithmetic,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->assignments.lookup(options.policy.scope.producers.front()),b.getStringAttr("b"));
  EXPECT_EQ(record.getString("local_objective"),"own-operations-input-repairs-and-resources-v2");
  auto local=options.policy;local.scope.fullGraph=false;auto oracle=enumerateOracle(graph,freezeRelationProofs(graph),arithmetic,{},local);ASSERT_TRUE(oracle.best);
  EXPECT_EQ(oracle.status,"complete");EXPECT_TRUE(oracle.exhausted);EXPECT_EQ(oracle.best->assignments.lookup(options.policy.scope.producers.front()),result->assignments.lookup(options.policy.scope.producers.front()));
  EXPECT_LE(budget.used,budget.limit);EXPECT_EQ(budget.used,budget.relationshipProbes+budget.completeScores+budget.refinementProbes);
}
TEST_F(JointAblationFinite, InformedRejectsUnknownContributionRatherThanOmittingIt) {
  variable("producer",1);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  OwnOperationCostTarget unknown;unknown.rows={{"unregistered-cost",0}};
  CandidateAssignment assignment;assignment.values[0]=graph.getVariable(0).candidates.front().value;
  ExperimentPolicy policy;policy.scope.fullGraph=false;policy.scope.informedLocal=true;policy.scope.producers={0};
  EXPECT_TRUE(failed(scoreExperimentAssignment(graph,assignment,unknown,policy)));
}

TEST_F(JointAblationFinite, InformedObjectiveStillExcludesFutureConsumerPreference) {
  variable("producer",2);variable("consumer",2);variable("tail",1);relation();
  ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  class DistinctTarget:public FiniteTarget {
  public:
    FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &g,
        const CandidateAssignment &a,LayoutCostObserver *observer)const override {
      CostEstimate total;
      for(const auto &v:g.getVariables()) {
        auto selected=a.values.lookup(v.id);if(!selected)continue;
        CostEstimate charge;
        if(v.stableName=="producer")charge.cost.instructionPathAndWork=
            cast<StringAttr>(a.values.lookup(v.id)).getValue()=="a"?0:1;
        if(v.stableName=="consumer")charge.cost.instructionPathAndWork=
            cast<StringAttr>(a.values.lookup(v.id)).getValue()=="a"?10:0;
        total=addLayoutCosts(total,charge);if(observer)observer->recordCost("arithmetic",v.id,charge);
      }
      return total;
    }
    FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &,const CandidateAssignment &)const override{return CostEstimate{};}
  } target;
  // Replace the fixture relation by p==consumer: the producer's own best
  // choice is a, whereas paying one own unit enables a ten-unit future win.
  auto p=*graph.lookupVariable("producer"),c=*graph.lookupVariable("consumer"),t=*graph.lookupVariable("tail");
  auto &contract=graph.getConstraints()[0];contract.instruction->tuples.clear();
  for(const auto &choice:graph.getVariable(p).candidates) {
    SmallVector<Attribute> enc;
    for(auto id:contract.vars)enc.push_back(id==t?graph.getVariable(t).candidates.front().value:choice.value);
    contract.instruction->tuples.push_back({enc,choice.value});
  }
  ASSERT_TRUE(succeeded(graph.verifyInvariants(module->getLoc())));
  SolverOptions options;options.policy.scope.informedLocal=true;options.policy.scope.producers={p};
  auto localPolicy=options.policy;localPolicy.scope.fullGraph=false;
  auto local=enumerateOracle(graph,freezeRelationProofs(graph),target,{},localPolicy);
  auto global=enumerateOracle(graph,freezeRelationProofs(graph),target);
  ASSERT_TRUE(local.best);ASSERT_TRUE(global.best);EXPECT_EQ(local.status,"complete");EXPECT_EQ(global.status,"complete");
  EXPECT_EQ(local.best->assignments.lookup(p),b.getStringAttr("a"));
  EXPECT_EQ(global.best->assignments.lookup(p),b.getStringAttr("b"));
  llvm::json::Object record;auto result=solveAblation(graph,target,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(result));
  EXPECT_EQ(result->assignments.lookup(p),b.getStringAttr("a"));EXPECT_EQ(result->cost.cost.instructionPathAndWork,10u);
}

TEST_F(JointAblationFinite, InformedRetainedContributionsOwnBorrowedKind) {
  variable("producer",1);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  class TemporaryKindTarget:public FiniteTarget {
  public:
    FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &,
        const CandidateAssignment &,LayoutCostObserver *observer)const override {
      CostEstimate charge;charge.cost.instructionPathAndWork=1;
      std::string temporary="arithmetic";
      if(observer)observer->recordCost(temporary,0,charge);
      // The observer contract borrows the name only for recordCost. Overwrite
      // it before return to make any retained non-owning JSON deterministic.
      std::fill(temporary.begin(),temporary.end(),'x');
      return charge;
    }
  } ephemeral;
  CandidateAssignment a;a.values[0]=graph.getVariable(0).candidates.front().value;
  auto full=scoreAssignment(graph,a,ephemeral);ASSERT_TRUE(succeeded(full));ASSERT_EQ(full->contributions.size(),1u);
  EXPECT_EQ(full->contributions[0].getAsObject()->getString("kind"),"arithmetic");
  ExperimentPolicy policy;policy.scope.fullGraph=false;policy.scope.informedLocal=true;policy.scope.producers={0};llvm::json::Array rows;
  auto local=scoreExperimentAssignment(graph,a,ephemeral,policy,nullptr,&rows);ASSERT_TRUE(succeeded(local));ASSERT_EQ(rows.size(),1u);
  EXPECT_EQ(rows[0].getAsObject()->getString("kind"),"arithmetic");
}
} // namespace


namespace {
TEST_F(JointAblationStages, InformedIncomingRepairsReverseOwnChoiceInBothModels) {
  auto module=parseSourceString<ModuleOp>(R"mlir(func.func @incoming(%q:tensor<64xf32>)->tensor<64xf32>{
    %p=arith.mulf %q,%q : tensor<64xf32>
    %f=arith.mulf %p,%p : tensor<64xf32>
    return %f : tensor<64xf32> })mlir",&context);
  ASSERT_TRUE(module);SmallVector<Operation *> muls;module->walk([&](arith::MulFOp op){muls.push_back(op);});ASSERT_EQ(muls.size(),2u);
  Value q=muls[0]->getOperand(0),p=muls[0]->getResult(0),f=muls[1]->getResult(0);
  LayoutConstraintGraph graph;LayoutConstraintBuilder builder(graph);
  builder.getOrCreateDistributedVar(q);auto pid=builder.getOrCreateDistributedVar(p),fid=builder.getOrCreateDistributedVar(f);
  for(auto *op:muls)for(auto &use:op->getOpOperands()) {
    auto useID=builder.getOrCreateDistributedUse(use);
    ASSERT_TRUE(succeeded(builder.same(op==muls[0]?pid:fid,useID,op,"arithmetic own layout")));
  }
  auto a=encoding(),b=encoding(true);
  for(auto &v:graph.getVariables()) {
    bool own=(v.value==p && !v.use)||(v.use && v.use->getOwner()==muls[0]);
    v.candidates=own?SmallVector<LayoutCandidate>{{a,kInvalidProvenanceID,0},{b,kInvalidProvenanceID,1}}:
                     SmallVector<LayoutCandidate>{{b,kInvalidProvenanceID,1}};
  }
  ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));pid=*graph.lookupVariable(p);
  SM90CostProfile profile;profile.coefficients.fill(1);profile.provenance="unit own incoming repair";
  auto calibrated=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,profile);
  for(auto *model:{target.get(),calibrated.get()}) {
    SolverOptions options;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={pid};
    auto restrictedGraph=graph;llvm::json::Object oldRecord;
    auto restricted=solveAblation(restrictedGraph,*model,SelectionMode::ProducerFirst,options,oldRecord);ASSERT_TRUE(succeeded(restricted));EXPECT_EQ(restricted->assignments.lookup(pid),a);
    options.policy.scope.informedLocal=true;auto informedGraph=graph;llvm::json::Object record;
    auto informed=solveAblation(informedGraph,*model,SelectionMode::ProducerFirst,options,record);ASSERT_TRUE(succeeded(informed));EXPECT_EQ(informed->assignments.lookup(pid),b);
    EXPECT_TRUE(record.getObject("stage2_local_cost"));EXPECT_EQ(record.getBoolean("local_cost_changed_during_completion"),false);
    auto local=options.policy;local.scope.fullGraph=false;auto proofs=freezeRelationProofs(graph,model);
    auto oracle=enumerateOracle(graph,proofs,*model,{},local);ASSERT_TRUE(oracle.best);EXPECT_EQ(oracle.status,"complete");EXPECT_EQ(oracle.best->assignments.lookup(pid),b);
    auto global=options.policy;global.scope.fullGraph=true;auto joint=enumerateOracle(graph,proofs,*model,{},global);ASSERT_TRUE(joint.best);EXPECT_EQ(joint.best->assignments.lookup(pid),b);
    CandidateAssignment forced;
    for(const auto &v:graph.getVariables())forced.values[v.id]=(v.value==p && !v.use)||(v.use && v.use->getOwner()==muls[0])?a:b;
    llvm::json::Array rows;auto score=scoreExperimentAssignment(graph,forced,*model,local,nullptr,&rows);ASSERT_TRUE(succeeded(score));unsigned own=0,future=0;
    for(const auto &row:rows)if(row.getAsObject()->getString("kind")=="conversion") {
      auto *entry=row.getAsObject();EXPECT_EQ(entry->getArray("dependencies")->size(),2u);
      if(entry->getBoolean("included")==true)++own;else ++future;
    }
    EXPECT_EQ(own,2u);EXPECT_EQ(future,2u);
    // Shared/rematerialized O is rejected for the entire prepared domain,
    // even if this particular B assignment would have no actual repair.
    local.conversion=ConversionPolicy::Full;EXPECT_FALSE(experimentPolicyUnsupportedReason(graph,local).empty());
    EXPECT_TRUE(failed(scoreExperimentAssignment(graph,CandidateAssignment{informed->assignments},*model,local)));
    auto unsupportedOracle=enumerateOracle(graph,proofs,*model,{},local);EXPECT_EQ(unsupportedOracle.status,"unsupported");EXPECT_FALSE(unsupportedOracle.exhausted);EXPECT_EQ(unsupportedOracle.evaluatedAssignments,0u);
    auto oGraph=graph;auto oOptions=options;oOptions.policy=local;SharedSearchBudget budget;oOptions.sharedBudget=&budget;LayoutSolveFailure why;
    auto unsupported=solveLayoutGraph(oGraph,*model,oOptions,&why);EXPECT_TRUE(failed(unsupported));EXPECT_EQ(why.kind,LayoutSolveFailureKind::Unsupported);EXPECT_EQ(budget.used,0u);
    oGraph=graph;llvm::json::Object unsupportedRecord;unsupported=solveAblation(oGraph,*model,SelectionMode::ProducerFirst,oOptions,unsupportedRecord);EXPECT_TRUE(failed(unsupported));EXPECT_EQ(unsupportedRecord.getString("failure_kind"),"unsupported");EXPECT_EQ(budget.used,0u);
  }
}
} // namespace
