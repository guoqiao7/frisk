#include "JointAblation.h"
#include "ArchiveSourceGrammar.h"
#include "SelectionOracle.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
TEST_F(JointAblationStages, DemandLineageDistinguishesExistingTransportResultFromRewrittenRepairOperand) {
 context.allowUnregisteredDialects();
 auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%a:tensor<4xf32>)->tensor<4xf32>{
  %source=arith.addf %a,%a : tensor<4xf32>
  %existing="test.existing_transport"(%source) : (tensor<4xf32>)->tensor<4xf32>
  %consumer=arith.addf %existing,%existing : tensor<4xf32>
  return %consumer : tensor<4xf32> })mlir",&context);ASSERT_TRUE(module);
 Operation *source=nullptr,*existing=nullptr,*consumer=nullptr;
 module->walk([&](Operation *o){if(o->getName().getStringRef()=="test.existing_transport")existing=o;else if(isa<arith::AddFOp>(o)){if(!source)source=o;else consumer=o;}});ASSERT_TRUE(source && existing && consumer);
 LayoutConstraintGraph graph;auto type=cast<RankedTensorType>(source->getResult(0).getType());
 auto src=graph.addVariable(LayoutKind::Distributed,type,"source");graph.getVariable(src).value=source->getResult(0);
 auto dst=graph.addVariable(LayoutKind::Distributed,type,"existing");graph.getVariable(dst).value=existing->getResult(0);
 auto use=graph.addVariable(LayoutKind::Distributed,type,"consumer/use0");graph.getVariable(use).use=&consumer->getOpOperand(0);
 auto old=graph.addConstraint(ConstraintKind::Convertible,ConstraintStrength::Hard,{src,dst},existing,"existing","actual existing transport");
 graph.getConstraints()[old].use=&existing->getOpOperand(0);graph.getConstraints()[old].existingConversion=true;graph.getConstraints()[old].stableUseKey="existing/use0";
 auto repair=graph.addConstraint(ConstraintKind::Convertible,ConstraintStrength::Hard,{dst,use},consumer,"repair","prospective consumer repair");
 graph.getConstraints()[repair].use=&consumer->getOpOperand(0);graph.getConstraints()[repair].stableUseKey="consumer/use0";
 ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));for(const auto &c:graph.getConstraints())if(c.kind==ConstraintKind::Convertible){if(c.existingConversion)old=c.id;else repair=c.id;}markDemandLineage(graph);
 OpBuilder builder(consumer);OperationState state(consumer->getLoc(),"test.materialized_transport");state.addOperands(existing->getResult(0));state.addTypes(type);auto *inserted=builder.create(state);consumer->setOperand(0,inserted->getResult(0));
 ASSERT_TRUE(succeeded(resolveDemandLineage(*module)));
 auto ids=[&](Operation *o){SmallVector<int64_t> values;if(auto a=o->getAttrOfType<ArrayAttr>("frisk.execution_audit_boundaries"))llvm::append_range(values,cast<DenseI64ArrayAttr>(a[0]).asArrayRef());return values;};
 EXPECT_TRUE(llvm::is_contained(ids(existing),old));EXPECT_FALSE(llvm::is_contained(ids(source),old));
 EXPECT_TRUE(llvm::is_contained(ids(inserted),repair));EXPECT_FALSE(llvm::is_contained(ids(existing),repair));
 EXPECT_FALSE(consumer->hasAttr("frisk.execution_audit_operand_boundaries"));
}
TEST_F(JointAblationStages, DemandLineageRejectsMalformedOperandArity) {
 auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%a:tensor<4xf32>)->tensor<4xf32>{
  %sum=arith.addf %a,%a : tensor<4xf32> return %sum : tensor<4xf32> })mlir",&context);ASSERT_TRUE(module);
 module->walk([&](arith::AddFOp op){op->setAttr("frisk.execution_audit_operand_boundaries",b.getArrayAttr({b.getDenseI64ArrayAttr({0})}));});
 EXPECT_TRUE(failed(resolveDemandLineage(*module)));
}
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
 EXPECT_EQ(record.getBoolean("incumbent_full_score_budget_charged"),true);
 EXPECT_NE(record.getObject("incumbent_full_score"),nullptr);
 ASSERT_NE(record.getObject("stage2"),nullptr);
 EXPECT_GE(record.getObject("stage2")->getInteger("complete_scores").value_or(0),1);
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

namespace {
TEST_F(JointAblationFinite, SharedArchivePinsLocalAndJointWithinOneBudget) {
 variable("producer",2);variable("consumer",2);variable("tail",2);relation();ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};options.seedBeamSolution=true;
 SmallVector<ArchivedLayoutCandidate,0> candidates;llvm::json::Object record;
 auto result=solveCandidateArchive(graph,target,SelectionMode::Joint,options,4,candidates,record);ASSERT_TRUE(succeeded(result));
 EXPECT_LE(candidates.size(),4u);EXPECT_TRUE(llvm::any_of(candidates,[](const auto &x){return x.localIncumbent;}));EXPECT_TRUE(llvm::any_of(candidates,[](const auto &x){return x.jointIncumbent;}));
 EXPECT_LE(budget.used,512u);EXPECT_LE(budget.refinementUsed,64u);EXPECT_EQ(budget.limit,512u);EXPECT_EQ(budget.used,budget.relationshipProbes+budget.completeScores+budget.refinementProbes);
 const auto *receipt=record.getObject("candidate_archive_search");ASSERT_TRUE(receipt);EXPECT_EQ(receipt->getInteger("local_terminal_probes"),128);EXPECT_LE(*receipt->getObject("local_end")->getInteger("probes"),128);
 for(const auto &x:candidates){EXPECT_EQ(x.solution.assignments.size(),3u);EXPECT_EQ(x.solution.instructionBindings.size(),1u);EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,x.solution,target,module->getLoc(),ConversionPolicy::Independent)));EXPECT_FALSE(x.solution.cost.cost<result->cost.cost);}
 for(size_t i=0;i<candidates.size();++i)for(size_t j=i+1;j<candidates.size();++j){bool same=true;for(auto [id,value]:candidates[i].solution.assignments)same&=candidates[j].solution.assignments.lookup(id)==value;EXPECT_FALSE(same);}
}
TEST_F(JointAblationFinite, LocalArchiveUsesFullSameBudgetWithoutJointPhase) {
 variable("producer",2);variable("consumer",2);variable("tail",2);relation();ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};
 SmallVector<ArchivedLayoutCandidate,0> candidates;llvm::json::Object record;auto result=solveCandidateArchive(graph,target,SelectionMode::ProducerFirst,options,4,candidates,record);ASSERT_TRUE(succeeded(result));
 const auto *receipt=record.getObject("candidate_archive_search");ASSERT_TRUE(receipt);EXPECT_EQ(receipt->getInteger("local_terminal_probes"),512);EXPECT_EQ(receipt->getString("joint_status"),"not-requested-local-control");
 EXPECT_TRUE(llvm::any_of(candidates,[](const auto &x){return x.localIncumbent;}));EXPECT_FALSE(llvm::any_of(candidates,[](const auto &x){return x.jointIncumbent;}));EXPECT_LE(budget.used,512u);
 auto producer=candidates.front().solution.assignments.lookup(0);for(const auto &x:candidates)EXPECT_EQ(x.solution.assignments.lookup(0),producer);
}
TEST_F(JointAblationFinite, ArchiveRejectsInvalidCapacityAndUsedBudgetBeforeSearch) {
 variable("producer",2);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 for(unsigned k:{0u,1u,7u}){SharedSearchBudget budget;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;EXPECT_TRUE(failed(solveCandidateArchive(graph,target,SelectionMode::Joint,options,k,out,record)));EXPECT_EQ(budget.used,0u);}
 SharedSearchBudget budget;budget.used=1;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;EXPECT_TRUE(failed(solveCandidateArchive(graph,target,SelectionMode::Joint,options,4,out,record)));EXPECT_EQ(budget.used,1u);
}
}

namespace {
TEST_F(JointAblationFinite, ArchiveCostFailureDoesNotBecomeBudgetFallback) {
 variable("producer",2);variable("consumer",2);variable("tail",2);relation();ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 class Broken:public FiniteTarget {public:FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &,const CandidateAssignment &,LayoutCostObserver *)const override{return failure();}} broken;
 SharedSearchBudget budget;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
 ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});EXPECT_TRUE(failed(solveCandidateArchive(graph,broken,SelectionMode::Joint,options,2,out,record)));EXPECT_EQ(record.getString("failure_kind"),"fatal-integrity");EXPECT_TRUE(out.empty());EXPECT_FALSE(record.getObject("candidate_archive_search"));
}
TEST_F(JointAblationFinite, NetworkK4PreservesPinsAndReturnsUnusedVariantBudgetOnTypedUnavailableProof) {
 variable("producer",2);variable("consumer",2);variable("tail",2);relation();ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 for(auto mode:{SelectionMode::Joint,SelectionMode::ProducerFirst}) {
  SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};
  SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
  auto result=solveCandidateArchive(graph,target,mode,options,4,out,record,&target);ASSERT_TRUE(succeeded(result));
  EXPECT_LE(out.size(),4u);EXPECT_EQ(budget.limit,512u);EXPECT_LE(budget.used,511u);
  EXPECT_EQ(budget.used,budget.relationshipProbes+budget.completeScores+budget.refinementProbes);
  auto *search=record.getObject("candidate_archive_search");ASSERT_TRUE(search);EXPECT_EQ(search->getString("version"),"shared-network-candidate-archive-v2");
  EXPECT_EQ(search->getInteger("local_terminal_probes"),mode==SelectionMode::Joint?128:511);
  auto *variant=search->getObject("network_variant");ASSERT_TRUE(variant);EXPECT_EQ(variant->getInteger("layout_search_limit"),511);
  EXPECT_EQ(variant->getInteger("variant_scores"),0);EXPECT_EQ(variant->getInteger("unused_reserved_score_probes"),1);
  EXPECT_EQ(variant->getString("status"),"unavailable-plan-proof-keep-auto");EXPECT_EQ(variant->getBoolean("plan_comparison_proven"),false);
  auto *pins=variant->getArray("pins");ASSERT_TRUE(pins);EXPECT_FALSE(pins->empty());EXPECT_LE(pins->size(),3u);
  bool keptLocal=false;for(const auto &entry:out){keptLocal|=entry.localIncumbent;EXPECT_FALSE(entry.networkVariant);EXPECT_EQ(entry.networkObjective,"min-shfl");}
  EXPECT_TRUE(keptLocal);for(const auto &pin:*pins){auto *o=pin.getAsObject();ASSERT_TRUE(o);auto slot=o->getInteger("retained_slot");ASSERT_TRUE(slot);ASSERT_LT(uint64_t(*slot),out.size());EXPECT_TRUE(out[*slot].localIncumbent || out[*slot].jointIncumbent || o->getBoolean("static_first").value_or(false));}
 }
}
TEST_F(JointAblationFinite, ArchiveSmallBudgetRetainsPaidWitnessOrTypedNoWitness) {
 variable("producer",2);variable("consumer",2);variable("tail",2);relation();ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 SharedSearchBudget budget;budget.limit=16;budget.refinementLimit=4;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
 ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});auto result=solveCandidateArchive(graph,target,SelectionMode::Joint,options,2,out,record);EXPECT_LE(budget.used,16u);EXPECT_LE(budget.refinementUsed,4u);ASSERT_TRUE(record.getObject("candidate_archive_search"));
 if(succeeded(result)){EXPECT_FALSE(out.empty());for(const auto &entry:out)EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,entry.solution,target,module->getLoc(),ConversionPolicy::Independent)));}
 else {EXPECT_TRUE(out.empty());auto kind=record.getString("failure_kind");ASSERT_TRUE(kind);EXPECT_TRUE(*kind=="search-budget" || *kind=="unknown" || *kind=="no-feasible" || *kind=="unsupported");}
}
}


namespace {
// Full target scores deliberately disagree with an independent paid memory axis.
// The archive must preserve that tradeoff, without evaluating another model or
// seeing an independently generated local archive.
class FeatureTradeoffTarget:public FiniteTarget {
 public:
  bool identical=false;
  FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &,
      const CandidateAssignment &a,LayoutCostObserver *observer)const override {
    CostEstimate result;
    for(auto [id,value]:a.values) {
      auto text=cast<StringAttr>(value).getValue();unsigned i=text.front()-'a';
      result.cost.instructionPathAndWork+=identical?10:10+10*i;
      result.features.emplace_back("global_sectors32",identical?100:(i==5?1:100));
      result.features.emplace_back("alu_warp_issues",identical?10:10+10*i);
      result.cost.spillRiskAndRegisters+=identical?1:1+i;
    }
    if(observer)observer->recordCost("registers",0,result);
    return result;
  }
  FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &,
      const CandidateAssignment &)const override{return CostEstimate{};}
};
TEST_F(JointAblationFinite, FeatureRetentionKeepsPaidAxisExtremeBeyondLegacyCostBand) {
  variable("producer",6);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));FeatureTradeoffTarget model;
  SmallVector<Attribute> oldAssignments,newAssignments;
  uint64_t baselineUsed=0,baselineScores=0;
  for(auto retention:{ArchiveRetentionPolicy::LegacyHammingV2,ArchiveRetentionPolicy::PaidFeatureFrontierV1}) {
    SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;
    SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};
    auto common=graph;SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
    auto solved=solveCandidateArchive(common,model,SelectionMode::Joint,options,4,out,record,nullptr,retention);ASSERT_TRUE(succeeded(solved));
    auto &assignments=retention==ArchiveRetentionPolicy::LegacyHammingV2?oldAssignments:newAssignments;
    for(const auto &candidate:out){assignments.push_back(candidate.solution.assignments.lookup(0));EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,candidate.solution,model,module->getLoc(),ConversionPolicy::Independent)));}
    EXPECT_TRUE(llvm::any_of(out,[](const auto &x){return x.localIncumbent;}));EXPECT_TRUE(llvm::any_of(out,[](const auto &x){return x.jointIncumbent;}));
    EXPECT_LE(out.size(),4u);EXPECT_LE(budget.used,512u);EXPECT_EQ(budget.used,budget.relationshipProbes+budget.completeScores+budget.refinementProbes);
    auto *search=record.getObject("candidate_archive_search");ASSERT_TRUE(search);auto *receipt=search->getObject("retention");ASSERT_TRUE(receipt);
    EXPECT_EQ(receipt->getInteger("extra_candidate_scores"),0);
    if(retention==ArchiveRetentionPolicy::LegacyHammingV2){baselineUsed=budget.used;baselineScores=budget.completeScores;}
    else {
      EXPECT_EQ(search->getString("version"),"shared-complete-candidate-archive-v2");EXPECT_EQ(search->getInteger("frontier_rejections"),0);
      EXPECT_EQ(receipt->getInteger("reservoir_capacity"),32);EXPECT_EQ(receipt->getString("policy"),"paid-feature-frontier-v1");
      EXPECT_EQ(budget.used,baselineUsed);EXPECT_EQ(budget.completeScores,baselineScores);
      EXPECT_TRUE(llvm::any_of(out,[](const auto &x){return x.retentionReason=="paid-feature-frontier-diversity";}));
    }
  }
  EXPECT_FALSE(llvm::is_contained(oldAssignments,b.getStringAttr("f")));EXPECT_TRUE(llvm::is_contained(newAssignments,b.getStringAttr("f")));
}
TEST_F(JointAblationFinite, FeatureRetentionNeverDeduplicatesEqualFeaturesAndIsDeterministic) {
  variable("producer",6);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));FeatureTradeoffTarget model;model.identical=true;
  SmallVector<Attribute> first;
  for(unsigned repeat=0;repeat<2;++repeat) {
    SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};
    auto common=graph;SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
    ASSERT_TRUE(succeeded(solveCandidateArchive(common,model,SelectionMode::Joint,options,4,out,record,&model,ArchiveRetentionPolicy::PaidFeatureFrontierV1,repeat!=0)));
    ASSERT_EQ(out.size(),4u);SmallVector<Attribute> current;
    for(const auto &x:out){EXPECT_FALSE(llvm::is_contained(current,x.solution.assignments.lookup(0)));current.push_back(x.solution.assignments.lookup(0));}
    if(!repeat)first=current;else EXPECT_EQ(first,current);
    auto *search=record.getObject("candidate_archive_search");ASSERT_TRUE(search);EXPECT_EQ(search->getString("version"),"shared-network-candidate-archive-v3");
    EXPECT_EQ(search->getObject("network_variant")->getInteger("variant_scores"),0);EXPECT_LE(budget.used,511u);
    if(repeat){auto *audit=search->getObject("paid_observation_audit");ASSERT_TRUE(audit);EXPECT_EQ(audit->getArray("entries")->size(),search->getInteger("paid_complete_observations").value_or(0));}
    else EXPECT_FALSE(search->getObject("paid_observation_audit"));
  }
}
TEST_F(JointAblationFinite, FeatureReservoirRemainsBoundedAndPreservesExtremeAcrossEvictions) {
  auto id=graph.addVariable(LayoutKind::Distributed,RankedTensorType::get({4},b.getF32Type()),"producer");
  for(unsigned i=0;i<40;++i)graph.getVariable(id).candidates.push_back({b.getStringAttr("v"+std::to_string(i)),kInvalidProvenanceID,i});
  ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  class RangeTarget:public FiniteTarget {
   public:
    FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &,
        const CandidateAssignment &a,LayoutCostObserver *observer)const override {
      auto value=a.values.lookup(0);if(!value)return CostEstimate{};
      unsigned index=0;bool invalid=cast<StringAttr>(value).getValue().drop_front().getAsInteger(10,index);
      if(invalid)return failure();CostEstimate c;c.cost.instructionPathAndWork=index+1;
      // Exercise exact ratio arithmetic near uint64 limits as well as eviction.
      c.features.emplace_back("global_sectors32",index==39?0:UINT64_MAX-index);
      c.features.emplace_back("alu_warp_issues",index+1);
      if(observer)observer->recordCost("registers",0,c);return c;
    }
    FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &,
        const CandidateAssignment &)const override{return CostEstimate{};}
  } model;
  SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};
  SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
  ASSERT_TRUE(succeeded(solveCandidateArchive(graph,model,SelectionMode::Joint,options,4,out,record,nullptr,ArchiveRetentionPolicy::PaidFeatureFrontierV1)));
  EXPECT_LE(out.size(),4u);EXPECT_TRUE(llvm::any_of(out,[&](const auto &x){return x.solution.assignments.lookup(0)==b.getStringAttr("v39");}));
  auto *retention=record.getObject("candidate_archive_search")->getObject("retention");ASSERT_TRUE(retention);
  EXPECT_EQ(retention->getInteger("reservoir_peak"),32);EXPECT_EQ(retention->getInteger("observer_working_set_peak"),33);
  auto *reasons=retention->getObject("discard_reason_counts");ASSERT_TRUE(reasons);EXPECT_GT(reasons->getInteger("bounded-reservoir-diversity").value_or(0),0);
  EXPECT_LE(retention->getArray("discard_examples")->size(),16u);EXPECT_LE(budget.used,512u);
}
TEST_F(JointAblationFinite, FeatureRetentionRejectsNonK4BeforeSearch) {
  variable("producer",2);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
  for(unsigned k:{2u,3u,5u,6u}) {
    SharedSearchBudget budget;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;
    SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
    EXPECT_TRUE(failed(solveCandidateArchive(graph,target,SelectionMode::Joint,options,k,out,record,nullptr,ArchiveRetentionPolicy::PaidFeatureFrontierV1)));
    EXPECT_EQ(budget.used,0u);EXPECT_TRUE(out.empty());EXPECT_EQ(record.getString("failure_kind"),"fatal-integrity");
  }
}
} // namespace


namespace {
DistributedEncodingAttr structureEncoding(MLIRContext &context,ArrayRef<uint64_t> registers,ArrayRef<uint64_t> lanes) {
 Builder b(&context);SmallVector<APInt> matrix;
 SmallVector<uint64_t> columns(registers);llvm::append_range(columns,lanes);
 for(unsigned row=0;row<6;++row)for(uint64_t column:columns)matrix.emplace_back(1,(column>>row)&1);
 auto map=BitLinearLayoutMapAttr::get(&context,b.getArrayAttr({b.getStringAttr("register"),b.getStringAttr("lane")}),b.getDenseI64ArrayAttr({int64_t(registers.size()),int64_t(lanes.size())}),b.getArrayAttr({b.getStringAttr("d")}),b.getDenseI64ArrayAttr({6}),DenseIntElementsAttr::get(RankedTensorType::get({6,int64_t(columns.size())},b.getI1Type()),matrix));
 return DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({int64_t(1ULL<<registers.size()),32,1,1,1}),b.getI64IntegerAttr(1ULL<<(columns.size()-6)));
}
TEST_F(JointAblationStages, GF2OwnershipDistanceIsBasisInvariantAndIgnoresUnusedNonvolatileLoads) {
 auto a=structureEncoding(context,{1,2},{4,8,16,32,0}),sameImage=structureEncoding(context,{1,3},{4,8,16,32,0}),other=structureEncoding(context,{1,4},{2,8,16,32,0});
 LayoutConstraintGraph graph;auto id=graph.addVariable(LayoutKind::Distributed,RankedTensorType::get({64},b.getF32Type()),"load");
 CandidateAssignment aa{{{id,a}}},bb{{{id,sameImage}}},cc{{{id,other}}};
 auto invariant=archiveStructureDistance(graph,aa,bb),different=archiveStructureDistance(graph,aa,cc);ASSERT_TRUE(succeeded(invariant));ASSERT_TRUE(succeeded(different));EXPECT_EQ(*invariant,0u);EXPECT_EQ(*different,2u);EXPECT_NE(a,sameImage);
 auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%a:memref<64xf32,1>){%unused=frisk.tile_load %a:memref<64xf32,1>->tensor<64xf32> return})mlir",&context);ASSERT_TRUE(module);
 module->walk([&](TileLoadOp load){graph.getVariable(id).value=load.getResult();});
 auto dead=archiveStructureDistance(graph,aa,cc);ASSERT_TRUE(succeeded(dead));EXPECT_EQ(*dead,0u);
 auto description=describeArchiveStructure(graph,aa);EXPECT_EQ(description.getBoolean("proven"),true);EXPECT_EQ((*description.getArray("ownership"))[0].getAsObject()->getBoolean("active"),false);
}
TEST_F(JointAblationStages, GF2CutKeepsStaticallyDominatedAlternativeAndLimitsRedundantSameCut) {
 auto a=structureEncoding(context,{1},{2,4,8,16,32});
 SmallVector<Attribute> encodings{a,structureEncoding(context,{2},{1,4,8,16,32}),structureEncoding(context,{2},{1,8,4,16,32}),structureEncoding(context,{2},{1,16,8,4,32}),structureEncoding(context,{2},{1,32,8,16,4})};
 auto module=parseSourceString<ModuleOp>(R"mlir(func.func @f(%x:tensor<64xf32>)->tensor<64xf32>{%y=arith.mulf %x,%x:tensor<64xf32> return %y:tensor<64xf32>})mlir",&context);ASSERT_TRUE(module);arith::MulFOp mul;module->walk([&](arith::MulFOp op){mul=op;});ASSERT_TRUE(mul);
 LayoutConstraintGraph graph;auto type=RankedTensorType::get({64},b.getF32Type());auto source=graph.addVariable(LayoutKind::Distributed,type,"source");graph.getVariable(source).value=mul.getLhs();graph.getVariable(source).candidates={{a,kInvalidProvenanceID,0}};
 auto use=graph.addVariable(LayoutKind::Distributed,type,"use");graph.getVariable(use).use=&mul->getOpOperand(0);
 for(unsigned i=0;i<encodings.size();++i)graph.getVariable(use).candidates.push_back({encodings[i],kInvalidProvenanceID,i});
 auto cid=graph.addConstraint(ConstraintKind::Convertible,ConstraintStrength::Hard,{source,use},mul,"edge","finite symbolic cut");graph.getConstraints()[cid].use=&mul->getOpOperand(0);graph.getConstraints()[cid].stableUseKey="mul/use0";ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 class Model:public FiniteTarget {
  public:SmallVector<Attribute> encodings;
   LogicalResult verifyCandidate(const LayoutVar &,Attribute value,Location)const override{return success(isa<DistributedEncodingAttr>(value));}
   bool needsCompleteAssignmentCost()const override{return true;}
   FailureOr<CostEstimate> evaluateExecutionCost(const LayoutConstraintGraph &,const CandidateAssignment &assignment,const ExperimentPolicy &policy,LayoutCostObserver *observer)const override {
    auto found=llvm::find(encodings,assignment.values.lookup(1));if(found==encodings.end())return failure();unsigned index=found-encodings.begin();CostEstimate cost;
    cost.cost.instructionPathAndWork=policy.scope.fullGraph?(index?10000*index:10):10;cost.features.emplace_back("global_sectors32",cost.cost.instructionPathAndWork);
    if(observer)observer->recordCost("registers",0,cost);return cost;
   }
   FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &,const CandidateAssignment &)const override{return CostEstimate{};}
 } model;model.encodings=encodings;
 for(auto retention:{ArchiveRetentionPolicy::LegacyHammingV2,ArchiveRetentionPolicy::PaidFeatureFrontierV1}) {
  SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={source};SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;auto common=graph;
  auto result=solveCandidateArchive(common,model,SelectionMode::Joint,options,4,out,record,&model,retention);ASSERT_TRUE(succeeded(result));EXPECT_LE(out.size(),4u);EXPECT_TRUE(llvm::any_of(out,[](const auto &x){return x.localIncumbent && x.jointIncumbent;}));EXPECT_LE(budget.used,511u);
  if(retention==ArchiveRetentionPolicy::LegacyHammingV2)EXPECT_EQ(out.size(),1u);
  else {EXPECT_EQ(out.size(),4u);EXPECT_TRUE(llvm::any_of(out,[&](const auto &x){return x.solution.assignments.lookup(use)==encodings[1];}));EXPECT_FALSE(llvm::any_of(out,[&](const auto &x){return x.solution.assignments.lookup(use)==encodings[4];}));auto *receipt=record.getObject("candidate_archive_search");EXPECT_EQ(receipt->getObject("network_variant")->getInteger("variant_scores"),0);for(const auto &x:out)EXPECT_EQ(x.retentionStructure.getBoolean("proven"),true);}
 }
}
} // namespace


namespace {
TEST_F(JointAblationFinite, UnknownSourceGrammarUsesExactLegacyTrajectoryWithOneBudget) {
 variable("producer",2);variable("consumer",2);variable("tail",2);relation();
 ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 for(auto mode:{SelectionMode::Joint,SelectionMode::ProducerFirst}) {
  SmallVector<ArchivedLayoutCandidate,0> baseline;uint64_t probes=0,scores=0,refinement=0,relations=0;
  for(auto retention:{ArchiveRetentionPolicy::LegacyHammingV2,ArchiveRetentionPolicy::CalibratedSourceGrammarV1}) {
   SharedSearchBudget budget;budget.limit=512;budget.refinementLimit=64;SolverOptions options;
   options.sharedBudget=&budget;options.policy.conversion=ConversionPolicy::Independent;options.policy.scope.producers={0};
   SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;auto common=graph;
   auto result=solveCandidateArchive(common,target,mode,options,4,out,record,&target,retention,false,{});
   ASSERT_TRUE(succeeded(result));ASSERT_LE(out.size(),4u);
   auto *search=record.getObject("candidate_archive_search");ASSERT_TRUE(search);
   EXPECT_EQ(search->getString("version"),"shared-network-candidate-archive-v2");
   EXPECT_EQ(search->getInteger("local_terminal_probes"),mode==SelectionMode::Joint?128:511);
   EXPECT_EQ(search->getObject("network_variant")->getInteger("unused_reserved_score_probes"),1);
   if(retention==ArchiveRetentionPolicy::LegacyHammingV2) {
    baseline=out;probes=budget.used;scores=budget.completeScores;refinement=budget.refinementUsed;relations=budget.relationshipProbes;
    EXPECT_FALSE(search->getObject("retention_resolution"));continue;
   }
   EXPECT_EQ(budget.used,probes);EXPECT_EQ(budget.completeScores,scores);
   EXPECT_EQ(budget.refinementUsed,refinement);EXPECT_EQ(budget.relationshipProbes,relations);
   ASSERT_EQ(out.size(),baseline.size());
   for(size_t i=0;i<out.size();++i) {
    EXPECT_EQ(out[i].solution.assignments,baseline[i].solution.assignments);
    EXPECT_EQ(out[i].origins,baseline[i].origins);EXPECT_EQ(out[i].localIncumbent,baseline[i].localIncumbent);
    EXPECT_EQ(out[i].jointIncumbent,baseline[i].jointIncumbent);EXPECT_EQ(out[i].networkObjective,baseline[i].networkObjective);
    EXPECT_EQ(out[i].probesAtObservation,baseline[i].probesAtObservation);
   }
   auto *resolution=search->getObject("retention_resolution");ASSERT_TRUE(resolution);
   EXPECT_EQ(resolution->getString("requested"),"calibrated-source-grammar-v1");
   EXPECT_EQ(resolution->getString("effective"),"legacy-hamming-v2");
   EXPECT_EQ(resolution->getInteger("resolver_calls"),1);EXPECT_EQ(resolution->getInteger("additional_searches"),0);
   EXPECT_EQ(resolution->getBoolean("native_output_guaranteed"),false);
   EXPECT_EQ(resolution->getObject("gate")->getString("status"),"unknown");
  }
 }
}
TEST_F(JointAblationFinite, SourceGrammarRecipeRejectsNonK4BeforeAnySearch) {
 variable("producer",2);ASSERT_TRUE(succeeded(graph.finalize(module->getLoc())));
 for(unsigned k:{2u,3u,5u,6u}) {
  SharedSearchBudget budget;SolverOptions options;options.sharedBudget=&budget;
  options.policy.conversion=ConversionPolicy::Independent;SmallVector<ArchivedLayoutCandidate,0> out;llvm::json::Object record;
  EXPECT_TRUE(failed(solveCandidateArchive(graph,target,SelectionMode::Joint,options,k,out,record,nullptr,ArchiveRetentionPolicy::CalibratedSourceGrammarV1)));
  EXPECT_EQ(budget.used,0u);EXPECT_TRUE(out.empty());
 }
}
} // namespace

namespace {
const char *sourceGrammarFixture() { return R"mlir(module attributes {frisk.execution_domain = "sm90-scalar-f32-dag-v1", frisk.target = "sm_90a"} {
  frisk.kernel @grammar_example(memref<16x16xf32, 1>, memref<16x16xf32, 1>, memref<16x16xf32, 1>, memref<16x16xf32, 1>) {
  ^bb0(%arg0: memref<16x16xf32, 1>, %arg1: memref<16x16xf32, 1>, %arg2: memref<16x16xf32, 1>, %arg3: memref<16x16xf32, 1>):
    frisk.storage_contract %arg0 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
    frisk.storage_contract %arg1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
    frisk.storage_contract %arg2 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
    frisk.storage_contract %arg3 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
    frisk.parallel (%arg4) = (1), threads = 128 {
      frisk.storage_contract %arg0 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
      %0 = frisk.layout_view %arg0 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1> -> memref<16x16xf32, 1>
      frisk.storage_contract %arg3 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
      %1 = frisk.layout_view %arg3 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1> -> memref<16x16xf32, 1>
      frisk.storage_contract %arg1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
      %2 = frisk.layout_view %arg1 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1> -> memref<16x16xf32, 1>
      frisk.storage_contract %arg2 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1>
      %3 = frisk.layout_view %arg2 {layout = #frisk.storage<map = #frisk.affine_layout<inputs = ["dim0", "dim1"], input_extents = [16, 16], outputs = ["byte_offset", "bit_offset"], output_extents = [1024, 8], map = affine_map<(d0, d1) -> ((d0 * 16 + d1) * 4, 0)>>, memory_space = #frisk<memory_space Global>, alignment = 1, vector_granularity = 1>} : memref<16x16xf32, 1> -> memref<16x16xf32, 1>
      %4 = frisk.tile_load %0 : memref<16x16xf32, 1> -> tensor<16x16xf32>
      %5 = frisk.tile_load %2 : memref<16x16xf32, 1> -> tensor<16x16xf32>
      %6 = frisk.tile_load %3 : memref<16x16xf32, 1> -> tensor<16x16xf32>
      %7 = arith.mulf %4, %5 : tensor<16x16xf32>
      %8 = math.exp %7 : tensor<16x16xf32>
      frisk.tile_store %8, %1 : tensor<16x16xf32>, memref<16x16xf32, 1>
      frisk.end
    }
    frisk.end
  }
})mlir"; }
TEST_F(JointAblationStages, SourceGrammarChecksCompleteIRAndAllFrozenCandidates) {
 context.loadDialect<math::MathDialect,memref::MemRefDialect>();context.allowUnregisteredDialects();
 // An independent small shape proves source grammar eligibility, not a case or
 // shape lookup, and cannot promise a native compiler's eventual output.
 for(StringRef mutation:{"none","math-log-no-div","unknown-op-no-div","region-no-div","call-no-div",
                         "half","dynamic","unknown-candidate","bad-storage-alias","extra-root","threads","grid","fastmath"}) {
  auto module=parseSourceString<ModuleOp>(sourceGrammarFixture(),&context);ASSERT_TRUE(module);
  auto graph=collectLayoutConstraints(*module,*target,LayoutCollectionMode::InitializeCandidates);ASSERT_TRUE(succeeded(graph));
  auto positive=checkArchiveSourceGrammar(*module,*graph);ASSERT_TRUE(positive.eligible)<<positive.evidence.getString("reason").value_or("").str();
  Operation *exp=nullptr,*parallel=nullptr,*kernel=nullptr;
  module->walk([&](Operation *op){if(isa<math::ExpOp>(op))exp=op;if(isa<ParallelOp>(op))parallel=op;if(isa<KernelOp>(op))kernel=op;});
  ASSERT_TRUE(exp && parallel && kernel);
  if(mutation=="math-log-no-div" || mutation=="unknown-op-no-div" || mutation=="region-no-div" || mutation=="call-no-div") {
   OpBuilder builder(exp);StringRef name=mutation=="math-log-no-div"?"math.log":mutation=="call-no-div"?"func.call":"test.unknown";
   OperationState state(exp->getLoc(),name);
   if(mutation=="math-log-no-div"){state.addOperands(exp->getOperands());state.addTypes(exp->getResultTypes());}
   if(mutation=="call-no-div")state.addAttribute("callee",FlatSymbolRefAttr::get(&context,"unknown"));
   if(mutation=="region-no-div"){auto *region=state.addRegion();region->push_back(new Block);}
   builder.create(state);
  } else if(mutation=="half")exp->getResult(0).setType(RankedTensorType::get({16,16},b.getF16Type()));
  else if(mutation=="dynamic")exp->getResult(0).setType(RankedTensorType::get({ShapedType::kDynamic,16},b.getF32Type()));
  else if(mutation=="unknown-candidate")graph->getVariable(0).candidates.push_back({b.getStringAttr("opaque-layout"),kInvalidProvenanceID,999});
  else if(mutation=="bad-storage-alias"){bool changed=false;for(auto &v:graph->getVariables())if(v.storageAlias){++v.storageAlias->lowerBit;changed=true;break;}ASSERT_TRUE(changed);}
  else if(mutation=="extra-root"){OpBuilder builder(&context);builder.setInsertionPointToEnd(module->getBody());OperationState state(module->getLoc(),"test.extra_root");builder.create(state);}
  else if(mutation=="threads")parallel->setAttr("threads",b.getI64IntegerAttr(64));
  else if(mutation=="grid")parallel->setAttr("ranges",b.getDenseI64ArrayAttr({2}));
  else if(mutation=="fastmath")exp->setAttr("unrecognized_lowering_contract",b.getUnitAttr());
  auto gate=checkArchiveSourceGrammar(*module,*graph);
  EXPECT_EQ(gate.eligible,mutation=="none")<<mutation.str()<<" "<<gate.evidence.getString("reason").value_or("").str();
  EXPECT_EQ(gate.evidence.getBoolean("native_output_guaranteed"),false);
  EXPECT_EQ(gate.evidence.getBoolean("source_eligibility_only"),true);
  if(mutation!="none")EXPECT_FALSE(gate.evidence.getString("reason").value_or("").empty());
 }
}
} // namespace
