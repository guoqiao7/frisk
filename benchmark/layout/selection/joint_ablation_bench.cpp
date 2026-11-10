#include "JointAblation.h"
#include "../../../lib/Dialect/Frisk/Transforms/PreserveStorageContracts.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "mlir/AsmParser/AsmParser.h"
#include "SelectionExperiment.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/MemoryBuffer.h"
#include "SelectionOracle.h"
#include "Dialect/Frisk/Analysis/LayoutDemand.h"

#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "llvm/ADT/ScopeExit.h"
#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include <filesystem>
#include "llvm/Support/FormatVariadic.h"
using namespace mlir;using namespace mlir::frisk;using namespace mlir::frisk::selection;
int matrixMain(int argc,char **argv);
int main(int argc,char **argv){
 if(argc>1 && StringRef(argv[1]).starts_with("--"))return matrixMain(argc,argv);
 if(argc!=6){llvm::errs()<<"usage: input.mlir case A|B independent|full outputdir\n";return 2;}
 StringRef caseID=argv[2],forceID=argv[3],policyID=argv[4];
 if(!llvm::is_contained(ArrayRef<StringRef>{"no-conflict","conflict","diamond","remat"},caseID) ||
    (forceID!="A" && forceID!="B") || (policyID!="independent" && policyID!="full")) {
  llvm::errs()<<"invalid case, forced producer or conversion policy\n";return 2;
 }
 std::string out=argv[5];std::error_code directoryError;
 bool exists=std::filesystem::exists(out,directoryError);
 if(directoryError || (exists && (!std::filesystem::is_directory(out,directoryError) || directoryError || !std::filesystem::is_empty(out,directoryError) || directoryError))) {
  llvm::errs()<<"output directory must be fresh or empty\n";return 2;
 }
 MLIRContext context;context.disableMultithreading();context.loadDialect<FriskDialect,arith::ArithDialect,memref::MemRefDialect>();
 auto m=parseSourceFile<ModuleOp>(argv[1],&context);if(!m)return 2;
 auto target=createSM90LayoutTarget();auto graph=collectLayoutConstraints(*m,*target,LayoutCollectionMode::InitializeCandidates);if(failed(graph))return 2;
 // Controlled candidate injection is declared as module attributes;
 // it does not add contradictory hard RequireEncoding annotations.
 auto a=m->getOperation()->getAttr("joint.A"),b=m->getOperation()->getAttr("joint.B");if(!a||!b)return 2;
 SmallVector<Operation*> muls;Value load; m->walk([&](Operation *op){if(isa<arith::MulFOp>(op))muls.push_back(op);if(!load && isa<TileLoadOp>(op))load=op->getResult(0);});
 std::string id=argv[2];
 if(!load || muls.size()<(id=="no-conflict"?1u:id=="diamond"?3u:4u)) {llvm::errs()<<"controlled graph lacks required load/arithmetic structure\n";return 2;}
 Value producer=id=="diamond"?load:muls.front()->getResult(0);
 Operation *branchB=id=="no-conflict"?nullptr:muls[id=="diamond"?1:2];
 auto producerID=graph->lookupVariable(producer);if(!producerID)return 2;
 for(auto &v:graph->getVariables())if(v.kind==LayoutKind::Distributed){
  Attribute encoding=(v.use && v.use->getOwner()==branchB)||(v.value && v.value.getDefiningOp()==branchB)?b:a;
  if(id=="remat" && v.value==load)encoding=b;
  v.candidates={{encoding,kInvalidProvenanceID,encoding==a?0u:1u}};
  if(v.id==*producerID)v.candidates={{a,kInvalidProvenanceID,0},{b,kInvalidProvenanceID,1}};
 }
 auto proofs=freezeRelationProofs(*graph);auto domain=domainJSON(*graph,proofs);
 if(auto ec=llvm::sys::fs::create_directories(out)){llvm::errs()<<"cannot create output directory: "<<ec.message()<<"\n";return 2;}
 auto save=[&](StringRef name,StringRef text){std::error_code ec;llvm::raw_fd_ostream os(out+"/"+name.str(),ec);if(ec){llvm::errs()<<ec.message()<<"\n";return false;}os<<text;os.flush();os.close();if(os.has_error()){llvm::errs()<<"artifact write failed: "<<os.error().message()<<"\n";os.clear_error();return false;}return true;};
 if(!save("domains.json",llvm::formatv("{0:2}",llvm::json::Value(std::move(domain))).str()))return 2;
 auto &pv=graph->getVariable(*producerID);Attribute forced=StringRef(argv[3])=="A"?a:b;llvm::erase_if(pv.candidates,[&](auto &c){return c.value!=forced;});
 SolverOptions options;options.policy.conversion=StringRef(argv[4])=="independent"?ConversionPolicy::Independent:ConversionPolicy::Full;
 options.policy.scope.producers={*producerID};SharedSearchBudget budget;options.sharedBudget=&budget;
 llvm::json::Object record;auto solution=solveAblation(*graph,*target,SelectionMode::Joint,options,record);if(failed(solution))return 1;
 record["assignment"]=assignmentJSON(*graph,solution->assignments);record["conversions"]=conversionsJSON(*graph,solution->conversions);record["producer_id"]=*producerID;record["producer_stable_name"]=graph->getVariable(*producerID).stableName;record["candidate_injection"]="all Distributed definitions/use endpoints singleton A or branchB B; producer A/B; remat q0 B";
 record["domain_combinations"]=2;auto scored=scoreAssignment(*graph,CandidateAssignment{solution->assignments},*target,options.policy);if(failed(scored))return 1;record["cost_contributions"]=std::move(scored->contributions);
 llvm::json::Array repairs;for(auto &c:graph->getConstraints())if(c.kind==ConstraintKind::Convertible)repairs.push_back(llvm::json::Object{{"constraint",c.id},{"existing_conversion",c.existingConversion},{"use",c.stableUseKey}});record["repair_edges"]=std::move(repairs);
 if(failed(materializeLayouts(*m,*graph,*solution,false,target.get(),options.policy.conversion))||failed(verify(*m))||failed(verifyMaterializedLayouts(*m,*target)))return 1;
 std::string text;llvm::raw_string_ostream os(text);os<<*m;if(!save("output.mlir",text))return 2;
 PassManager pm(&context);auto pass=createExportExecutionPass();if(failed(pass->initializeOptions("output="+out+"/execution.json",[](const Twine &){return failure();})))return 2;pm.addPass(std::move(pass));if(failed(pm.run(*m)))return 1;
 record["status"]="success";if(!save("record.json",llvm::formatv("{0:2}",llvm::json::Value(std::move(record))).str()))return 2;return 0;
}

namespace {
std::string failureDirectory;
// Failure sidecars preserve the live shared ledger without changing the
// published five-field failure.json envelope consumed by older runners.
const llvm::json::Object *failureSearchRecord = nullptr;
const SharedSearchBudget *failureSearchBudget = nullptr;
bool failureSearchPerformed = false;
[[noreturn]] void matrixStop(const std::string &kind,const std::string &message,bool fatal){
 llvm::errs()<<message<<"\n";
 if(!failureDirectory.empty()){std::error_code ec;llvm::raw_fd_ostream os(failureDirectory+"/failure.json",ec);if(!ec){os<<llvm::formatv("{0:2}",llvm::json::Value(llvm::json::Object{{"status","failed"},{"failure_kind",kind},{"fatal",fatal},{"reason",message},{"limits",llvm::json::Object{{"probes",65536},{"refinement",256}}}}));os.close();if(os.has_error())os.clear_error();}}
 if(!failureDirectory.empty()) {
   SharedSearchBudget noSearch;
   const auto &b=failureSearchBudget?*failureSearchBudget:noSearch;
   llvm::json::Object totals{{"probes",b.used},{"refinement",b.refinementUsed},
       {"relationship",b.relationshipProbes},{"complete_scores",b.completeScores},
       {"refinement_probes",b.refinementProbes}};
   llvm::json::Object contract{{"kind","shared-total-search-probes-v1"},
       {"search_performed",failureSearchPerformed},{"limit",b.limit},
       {"refinement_limit",b.refinementLimit},{"used",b.used},
       {"refinement_used",b.refinementUsed},
       {"includes","constraint extensions, complete objective scores and refinement probes; no second seed budget"},
       {"certificate_rescores","selected-certificate verification only, no candidate selection"}};
   llvm::json::Object budget{{"schema_version",1},
       {"budget_context_available",failureSearchBudget!=nullptr},
       {"total",llvm::json::Object(totals)},{"budget_contract",llvm::json::Object(contract)}};
   llvm::json::Object snapshot=failureSearchRecord?llvm::json::Object(*failureSearchRecord):llvm::json::Object{};
   snapshot["status"]="failed";snapshot["failure_kind"]=kind;snapshot["fatal"]=fatal;
   snapshot["reason"]=message;snapshot["total"]=std::move(totals);
   snapshot["budget_contract"]=std::move(contract);
   snapshot["budget_context_available"]=failureSearchBudget!=nullptr;
   auto sidecar=[&](StringRef name,llvm::json::Object object) {
     std::error_code ec;llvm::raw_fd_ostream os(failureDirectory+"/"+name.str(),ec);
     if(!ec){os<<llvm::formatv("{0:2}",llvm::json::Value(std::move(object)));os.close();if(os.has_error())os.clear_error();}
   };
   sidecar("failure-budget.json",std::move(budget));
   sidecar("failure-search-record.json",std::move(snapshot));
 }
 std::exit(fatal?2:3);
}
[[noreturn]] void matrixError(const std::string &message){matrixStop("fatal-integrity",message,true);}
[[noreturn]] void documentedFailure(const LayoutSolveFailure &why){
 switch(why.kind){case LayoutSolveFailureKind::EmptyDomain:case LayoutSolveFailureKind::HardConflict:matrixStop("no-feasible",why.reason,false);case LayoutSolveFailureKind::Unsupported:matrixStop("unsupported",why.reason,false);case LayoutSolveFailureKind::SearchBudget:matrixStop("search-budget",why.reason,false);case LayoutSolveFailureKind::SearchIncomplete:matrixStop("unknown",why.reason,false);default:matrixError(why.reason.empty()?"solver failed without typed reason":why.reason);}
}

std::string jsonText(llvm::json::Value v){return llvm::formatv("{0:2}",v).str();}
std::string moduleText(ModuleOp m){std::string text;llvm::raw_string_ostream(text)<<m;return text;}
llvm::json::Value readJSON(StringRef path){auto b=llvm::MemoryBuffer::getFile(path);if(!b)matrixError("cannot read "+path.str());auto j=llvm::json::parse((*b)->getBuffer());if(!j)matrixError("malformed JSON");return std::move(*j);}
void writeFile(const std::string &dir,StringRef name,StringRef text){std::error_code ec;llvm::raw_fd_ostream os(dir+"/"+name.str(),ec);if(ec)matrixError(ec.message());os<<text;os.close();if(os.has_error()){os.clear_error();matrixError("artifact write failed");}}
void fresh(const std::string &out){if(std::filesystem::exists(out) && !std::filesystem::is_empty(out))matrixError("output directory must be fresh or empty");std::filesystem::create_directories(out);failureDirectory=out;}
Attribute parseJSONAttribute(MLIRContext &ctx,const llvm::json::Object &o,StringRef key){auto t=o.getString(key);if(!t)matrixError("missing attribute");if(t->empty())return {};auto a=parseAttribute(*t,&ctx);if(!a)matrixError("malformed attribute");return a;}
Attribute candidate(MLIRContext &ctx,const llvm::json::Object &o){auto a=parseJSONAttribute(ctx,o,"attribute");if(o.getString("id")!=sha256(layoutCandidateKey(a)))matrixError("candidate ID mismatch");return a;}
const llvm::json::Object &object(const llvm::json::Value &v){auto *o=v.getAsObject();if(!o)matrixError("JSON schema: expected object");return *o;}
const llvm::json::Array &array(const llvm::json::Object &o,StringRef key){auto *a=o.getArray(key);if(!a)matrixError("JSON schema: expected array "+key.str());return *a;}
StringRef string(const llvm::json::Object &o,StringRef key){auto v=o.getString(key);if(!v)matrixError("JSON schema: expected string "+key.str());return *v;}
int64_t integer(const llvm::json::Object &o,StringRef key,int64_t lo=0,int64_t hi=INT64_MAX){auto v=o.getInteger(key);if(!v || *v<lo || *v>hi)matrixError("JSON schema: invalid integer "+key.str());return *v;}
bool boolean(const llvm::json::Object &o,StringRef key){auto v=o.getBoolean(key);if(!v)matrixError("JSON schema: expected boolean "+key.str());return *v;}
void digestField(const llvm::json::Object &o,StringRef key){auto value=string(o,key);if(value.size()!=64 || !llvm::all_of(value,[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');}))matrixError("JSON schema: invalid digest");}
void candidateSchema(const llvm::json::Object &o){string(o,"attribute");digestField(o,"id");}
void ids(const llvm::json::Array &values,int64_t limit,bool unique=true){std::set<int64_t> seen;for(const auto &v:values){auto n=v.getAsInteger();if(!n || *n<0 || *n>=limit || (unique && !seen.insert(*n).second))matrixError("JSON schema: invalid/duplicate ID");}}
void bindingSchema(const llvm::json::Array &values,int64_t limit){std::set<int64_t> seen;for(const auto &v:values){const auto &o=object(v);auto id=integer(o,"constraint",0,limit-1);if(!seen.insert(id).second)matrixError("JSON schema: duplicate binding");string(o,"binding");}}
void preparedSchema(const llvm::json::Object &d,const llvm::json::Object &meta){
 integer(d,"schema_version",1,1);string(d,"stage");string(d,"support_table_policy");string(d,"support_absence_meaning");
 const auto &vars=array(d,"variables"),&constraints=array(d,"constraints"),&provenances=array(d,"provenances");if(vars.empty())matrixError("JSON schema: empty domain");
 std::set<int64_t> seen;std::set<std::string> names;
 for(const auto &entry:vars){const auto &v=object(entry);auto id=integer(v,"id",0,vars.size()-1);if(!seen.insert(id).second || !names.insert(string(v,"name").str()).second)matrixError("JSON schema: duplicate variable");integer(v,"kind",0,2);integer(v,"state",0,3);integer(v,"required_threads");string(v,"type");boolean(v,"instruction_role");boolean(v,"reduction_result");boolean(v,"candidate_preparation_unsupported");std::set<std::string> candidates;
  for(const auto &x:array(v,"candidates")){const auto &c=object(x);candidateSchema(c);integer(c,"ordinal");integer(c,"provenance",0,UINT32_MAX);if(!candidates.insert(string(c,"id").str()).second)matrixError("JSON schema: duplicate candidate");}
 }
 seen.clear();for(const auto &entry:constraints){const auto &c=object(entry);auto id=integer(c,"id",0,constraints.size()-1);if(!seen.insert(id).second)matrixError("JSON schema: duplicate constraint");string(c,"kind");string(c,"strength");ids(array(c,"variables"),vars.size(),false);string(c,"required_encoding");string(c,"coordinate_transform");integer(c,"access",0,1);boolean(c,"existing_conversion");string(c,"stable_use_key");integer(c,"required_threads");integer(c,"provenance",0,provenances.size()-1);
  if(auto *item=c.get("instruction")){const auto &instruction=object(*item);string(instruction,"binding");boolean(instruction,"support_complete");for(const auto &v:array(instruction,"tuples")){const auto &t=object(v);string(t,"binding");string(t,"proof");const auto &enc=array(t,"encodings");if(enc.size()!=array(c,"variables").size())matrixError("JSON schema: instruction tuple arity");for(const auto &a:enc)candidateSchema(object(a));}}
  if(auto *item=c.get("reduction")){const auto &reduction=object(*item);integer(reduction,"axis");string(reduction,"binding");boolean(reduction,"support_complete");for(const auto &v:array(reduction,"pairs")){const auto &pair=object(v);string(pair,"binding");string(pair,"proof");for(auto key:{"source","result"}){auto *a=pair.get(key);if(!a)matrixError("JSON schema: missing reduction candidate");candidateSchema(object(*a));}}}
 }
 seen.clear();for(const auto &entry:provenances){const auto &v=object(entry);auto id=integer(v,"id",0,provenances.size()-1);if(!seen.insert(id).second)matrixError("JSON schema: duplicate provenance");string(v,"rule");string(v,"reason");auto *parent=v.get("parent");if(!parent || (!parent->getAsNull() && (!parent->getAsInteger() || *parent->getAsInteger()<0 || uint64_t(*parent->getAsInteger())>=provenances.size())))matrixError("JSON schema: invalid parent");}
 for(const auto &entry:array(d,"region_edges")){const auto &v=object(entry);integer(v,"source",0,vars.size()-1);integer(v,"target",0,vars.size()-1);integer(v,"constraint",0,constraints.size()-1);string(v,"kind");string(v,"key");integer(v,"slot");}
 for(const auto &entry:array(d,"relation_proofs")){const auto &v=object(entry);integer(v,"constraint",0,constraints.size()-1);for(auto key:{"lhs","rhs"}){auto id=integer(v,key,0,UINT32_MAX);if(id!=UINT32_MAX && uint64_t(id)>=vars.size())matrixError("JSON schema: proof endpoint");}for(auto key:{"a","b"}){auto *c=v.get(key);if(!c)matrixError("JSON schema: missing proof candidate");candidateSchema(object(*c));}auto status=string(v,"status");if(status!="proven"&&status!="disproven"&&status!="unknown")matrixError("JSON schema: proof status");string(v,"reason");for(const auto &x:array(v,"counterexample"))if(!x.getAsInteger())matrixError("JSON schema: proof counterexample");}
 for(auto key:{"common_digest","domain_digest","coverage_digest","package_digest","input_sha256","domain_file_sha256"})digestField(meta,key);
 for(auto key:{"case","cost_model","frequency_mode","candidate_preparation"})string(meta,key);ids(array(meta,"producers"),vars.size());ids(array(meta,"producer_instructions"),constraints.size());std::set<std::pair<int64_t,std::string>> coverage;
 for(const auto &entry:array(meta,"candidate_proofs")){const auto &v=object(entry);auto id=integer(v,"variable",0,vars.size()-1);auto attribute=string(v,"attribute");boolean(v,"valid");boolean(v,"capacity");if(!coverage.insert({id,attribute.str()}).second)matrixError("JSON schema: duplicate candidate proof");}
}
void assignmentSchema(const llvm::json::Object &o,const LayoutConstraintGraph &g){
 digestField(o,"prepared_digest");digestField(o,"complete_assignment_sha256");auto policy=string(o,"conversion_policy");if(policy!="independent"&&policy!="shared-only"&&policy!="full")matrixError("JSON schema: conversion policy");
 const auto &values=array(o,"assignment");if(values.size()!=g.getVariables().size())matrixError("JSON schema: incomplete assignment");std::set<int64_t> seen;
 for(const auto &entry:values){const auto &v=object(entry);auto id=integer(v,"variable",0,g.getVariables().size()-1);if(!seen.insert(id).second || string(v,"name")!=g.getVariable(id).stableName)matrixError("JSON schema: assignment variable identity");auto *c=v.get("candidate");if(!c)matrixError("JSON schema: missing candidate");candidateSchema(object(*c));}
 bindingSchema(array(o,"instruction_bindings"),g.getConstraints().size());bindingSchema(array(o,"reduction_bindings"),g.getConstraints().size());
}
void restore(LayoutConstraintGraph &g,const llvm::json::Object &d,MLIRContext &ctx){
 auto *vars=d.getArray("variables"),*constraints=d.getArray("constraints"),*provenances=d.getArray("provenances");
 if(!vars || vars->size()!=g.getVariables().size() || !constraints || constraints->size()!=g.getConstraints().size() || !provenances)matrixError("domain shape mismatch");
 for(auto &j:*vars){auto &v=*j.getAsObject();auto id=*v.getInteger("id");if(id<0 || unsigned(id)>=g.getVariables().size())matrixError("variable ID mismatch");auto &dst=g.getVariable(id);if(v.getString("name")!=dst.stableName)matrixError("stable variable mismatch");dst.candidates.clear();dst.state=LayoutState(*v.getInteger("state"));dst.candidatePreparationUnsupported=*v.getBoolean("candidate_preparation_unsupported");
  for(auto &x:*v.getArray("candidates")){auto &c=*x.getAsObject();dst.candidates.push_back({candidate(ctx,c),ProvenanceID(*c.getInteger("provenance")),uint64_t(*c.getInteger("ordinal"))});}
 }
 for(auto &j:*constraints){auto &src=*j.getAsObject();auto id=*src.getInteger("id");if(id<0 || unsigned(id)>=g.getConstraints().size())matrixError("constraint ID mismatch");auto &dst=g.getConstraints()[id];
  if(auto *instruction=src.getObject("instruction")){if(!dst.instruction)matrixError("instruction mismatch");dst.instruction->binding=parseJSONAttribute(ctx,*instruction,"binding");dst.instruction->tuples.clear();for(auto &x:*instruction->getArray("tuples")){auto &t=*x.getAsObject();SmallVector<Attribute> enc;for(auto &e:*t.getArray("encodings"))enc.push_back(candidate(ctx,*e.getAsObject()));dst.instruction->tuples.push_back({enc,parseJSONAttribute(ctx,t,"binding")});}}
  if(auto *reduction=src.getObject("reduction")){if(!dst.reduction)matrixError("reduction mismatch");dst.reduction->binding=parseJSONAttribute(ctx,*reduction,"binding");dst.reduction->pairs.clear();for(auto &x:*reduction->getArray("pairs")){auto &v=*x.getAsObject();dst.reduction->pairs.push_back({candidate(ctx,*v.getObject("source")),candidate(ctx,*v.getObject("result")),parseJSONAttribute(ctx,v,"binding")});}}
 }
 for(unsigned i=g.getProvenances().size();i<provenances->size();++i){auto &p=*(*provenances)[i].getAsObject();std::optional<ProvenanceID> parent;if(auto n=p.getInteger("parent"))parent=*n;g.addProvenance(parent,nullptr,*p.getString("rule"),*p.getString("reason"));}
}


FrozenProofs readProofs(const llvm::json::Object &d,const llvm::json::Object &meta,MLIRContext &ctx){
 FrozenProofs p;p.commonDigest=meta.getString("common_digest")->str();p.domainDigest=meta.getString("domain_digest")->str();p.coverageDigest=meta.getString("coverage_digest")->str();
 for(auto &j:*d.getArray("relation_proofs")){auto &v=*j.getAsObject();SmallVector<int64_t> example;for(auto &x:*v.getArray("counterexample"))example.push_back(*x.getAsInteger());auto status=*v.getString("status");p.entries.push_back({LayoutConstraintID(*v.getInteger("constraint")),LayoutVarID(*v.getInteger("lhs")),LayoutVarID(*v.getInteger("rhs")),parseJSONAttribute(ctx,*v.getObject("a"),"attribute"),parseJSONAttribute(ctx,*v.getObject("b"),"attribute"),{status=="proven"?ProofStatus::Proven:status=="disproven"?ProofStatus::Disproven:ProofStatus::Unknown,example,v.getString("reason")->str()}});}
 for(auto &j:*meta.getArray("candidate_proofs")){auto &v=*j.getAsObject();p.candidates.push_back({LayoutVarID(*v.getInteger("variable")),parseJSONAttribute(ctx,v,"attribute"),*v.getBoolean("valid"),*v.getBoolean("capacity")});}return p;
}
llvm::json::Array bindingJSON(const DenseMap<LayoutConstraintID,Attribute> &bindings){SmallVector<LayoutConstraintID> ids;for(auto &b:bindings)ids.push_back(b.first);llvm::sort(ids);llvm::json::Array result;for(auto id:ids){std::string text;llvm::raw_string_ostream(text)<<bindings.lookup(id);result.push_back(llvm::json::Object{{"constraint",id},{"binding",text}});}return result;}
}
namespace {
llvm::json::Value costProfileJSON(const SM90CostProfile &profile) {
  auto value=llvm::json::parse(serializeSM90CostProfile(profile));
  if(!value)matrixError(llvm::toString(value.takeError()));
  return std::move(*value);
}
}
llvm::json::Object networkStaticContractJSON(const SM90CostProfile &profile,StringRef frequency) {
 if(!profile.networkProxy)matrixError("common network contract requires explicit engineering proxy");
 auto value=costProfileJSON(profile);auto coefficients=llvm::json::Object(*value.getAsObject()->getObject("coefficients"));
 const std::string contract="sm90-network-static-common-v1",grammar="ordered-surviving-network-source-issues-v1";
 const std::string provenance="migrated-engineering-proxy-not-refit",units="nonnegative-score-ticks-per-feature-unit-not-cycles";
 std::string encoding=contract+"\n"+grammar+"\n"+frequency.str()+"\n"+provenance+"\n"+units+"\n";
 llvm::json::Array order;
 for(StringRef name:getSM90SurvivingCostFeatureNames()) {
  auto *entry=coefficients.get(name);if(!entry || !entry->getAsUINT64())matrixError("missing network common coefficient");
  order.push_back(name.str());encoding+=name.str()+"="+std::to_string(*entry->getAsUINT64())+"\n";
 }
 return llvm::json::Object{{"contract",contract},{"grammar",grammar},{"frequency_mode",frequency.str()},
   {"weight_provenance",provenance},{"units",units},{"feature_order",std::move(order)},
   {"coefficients",std::move(coefficients)},{"sha256",sha256(encoding)}};
}
int matrixMain(int argc,char **argv){
 MLIRContext context;context.disableMultithreading();context.loadDialect<FriskDialect,arith::ArithDialect,memref::MemRefDialect,func::FuncDialect,math::MathDialect,tensor::TensorDialect,linalg::LinalgDialect>();
 std::string profilePath;for(int i=1;i<argc;++i)if(StringRef(argv[i]).starts_with("--cost-profile=")) {
   if(!profilePath.empty())matrixError("duplicate cost profile");profilePath=StringRef(argv[i]).drop_front(15).str();if(profilePath.empty())matrixError("empty cost profile");
  }
  std::optional<SM90CostProfile> profile;std::unique_ptr<LayoutTarget> target;
  if(!profilePath.empty()) {
   auto buffer=llvm::MemoryBuffer::getFile(profilePath);if(!buffer)matrixError("cannot read cost profile");
   auto parsed=parseSM90CostProfile((*buffer)->getBuffer());if(!parsed)matrixError(llvm::toString(parsed.takeError()));
   profile=std::move(*parsed);target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,*profile);
  } else target=createSM90LayoutTarget();std::string frequency=target->boundaryFrequencyMode()==LayoutBoundaryFrequencyMode::Legacy?"legacy":"static-trip-count";
 if(StringRef(argv[1])=="--prepare"){
  if(argc<5 || argc>7)matrixError("--prepare input case outputdir [--natural] [--cost-profile=path]");
  bool natural=false;for(int i=5;i<argc;++i){StringRef a=argv[i];if(a=="--natural"){if(natural)matrixError("duplicate natural preparation option");natural=true;}else if(!a.starts_with("--cost-profile="))matrixError("unknown prepare option");}
  StringRef caseID=argv[3];if(!llvm::is_contained(ArrayRef<StringRef>{"no-conflict","conflict","diamond","remat","attention"},caseID))matrixError("unknown preparation case");
  std::string out=argv[4];fresh(out);auto m=parseSourceFile<ModuleOp>(argv[2],&context);if(!m)matrixError("parse failed");
  bool controlled=caseID!="attention" && !natural;
  if(!controlled){PassManager pm(&context);pm.addPass(createNormalizeLayoutIRPass());if(failed(pm.run(*m)))matrixError("normalization failed");}
  auto graph=collectLayoutConstraints(*m,*target,LayoutCollectionMode::InitializeCandidates);if(failed(graph))matrixError("collection failed");SmallVector<LayoutVarID> producers,instructions;
  if(caseID!="attention"){
   // Keep the local baseline's declared producer scope identical for the
   // controlled mechanism experiment and the natural candidate-domain anchor.
   // Natural preparation never overwrites any collector-generated candidates.
   SmallVector<Operation*> muls;Value load;m->walk([&](Operation *op){if(isa<arith::MulFOp>(op))muls.push_back(op);if(!load && isa<TileLoadOp>(op))load=op->getResult(0);});
   unsigned required=caseID=="no-conflict"?1u:caseID=="diamond"?3u:4u;if(!load || muls.size()<required)matrixError("graph lacks the declared producer/branch structure");
   Value producer=caseID=="diamond"?load:muls.front()->getResult(0);auto producerID=graph->lookupVariable(producer);if(!producerID)matrixError("missing producer");producers.push_back(*producerID);
   if(controlled){auto a=m->getOperation()->getAttr("joint.A"),b=m->getOperation()->getAttr("joint.B");if(!a||!b)matrixError("missing experimental A/B declaration");Operation *branchB=caseID=="no-conflict"?nullptr:muls[caseID=="diamond"?1:2];
    for(auto &v:graph->getVariables())if(v.kind==LayoutKind::Distributed){Attribute enc=(v.use && v.use->getOwner()==branchB)||(v.value && v.value.getDefiningOp()==branchB)?b:a;if(caseID=="remat" && v.value==load)enc=b;v.candidates={{enc,kInvalidProvenanceID,enc==a?0u:1u}};if(v.id==*producerID)v.candidates={{a,kInvalidProvenanceID,0},{b,kInvalidProvenanceID,1}};}
   }
  }else{for(const auto &c:graph->getConstraints())if(c.instruction){instructions.push_back(c.id);for(auto id:c.vars)if((graph->getVariable(id).kind==LayoutKind::Distributed || graph->getVariable(id).kind==LayoutKind::Instruction) && !graph->getVariable(id).use)producers.push_back(id);}llvm::sort(producers);producers.erase(std::unique(producers.begin(),producers.end()),producers.end());}
  auto proofs=freezeRelationProofs(*graph,target.get());auto d=domainJSON(*graph,proofs);llvm::json::Array candidateProofs,ps,is;for(auto id:producers)ps.push_back(id);for(auto id:instructions)is.push_back(id);for(auto &v:proofs.candidates){std::string text;llvm::raw_string_ostream(text)<<v.candidate;candidateProofs.push_back(llvm::json::Object{{"variable",v.variable},{"attribute",text},{"valid",v.valid},{"capacity",v.capacity}});}
  llvm::json::Object meta{{"common_digest",proofs.commonDigest},{"domain_digest",proofs.domainDigest},{"coverage_digest",proofs.coverageDigest},{"candidate_proofs",std::move(candidateProofs)},{"producers",std::move(ps)},{"producer_instructions",std::move(is)},{"case",argv[3]},{"cost_model",target->costModelID()},{"frequency_mode",frequency},{"candidate_preparation",controlled?"experimental-A/B-injection":"natural"}};
  if(auto executionDomain=m->getOperation()->getAttrOfType<StringAttr>("frisk.execution_domain"))meta["execution_domain"]=executionDomain.getValue();
   if(profile){meta["cost_profile_sha256"]=getSM90CostProfileSHA256(*profile);meta["cost_profile"]=costProfileJSON(*profile);}
   auto inputText=moduleText(*m),domainText=jsonText(llvm::json::Object(d));meta["input_sha256"]=sha256(inputText);meta["domain_file_sha256"]=sha256(domainText);meta["package_digest"]=sha256(inputText+domainText+jsonText(llvm::json::Object(meta)));writeFile(out,"input.mlir",inputText);writeFile(out,"domains.json",domainText);writeFile(out,"prepared.json",jsonText(std::move(meta)));return 0;
 }
 std::string prepared,out,selection="J",conversion="C",assignment,replay,forcedProducer,networkObjectiveOverride,archiveRetention="legacy-hamming-v2";bool invocationProfileEnabled=false,disableCostQueryCache=false;bool archiveRetentionSet=false,auditPaidObservations=false;bool networkArchiveV2=false,oracle=false,seedBeam=false,demandLedger=false,minifyExecutionJSON=false;unsigned archiveCapacity=0;for(int i=1;i<argc;++i){StringRef a=argv[i];auto split=a.split('=');if(split.first=="--prepared")prepared=split.second.str();else if(split.first=="--output-dir")out=split.second.str();else if(split.first=="--selection")selection=split.second.str();else if(split.first=="--conversion")conversion=split.second.str();else if(split.first=="--assignment")assignment=split.second.str();else if(split.first=="--replay")replay=split.second.str();else if(split.first=="--force-producer"){if(!forcedProducer.empty() || (split.second!="A" && split.second!="B"))matrixError("invalid or duplicate forced producer");forcedProducer=split.second.str();}else if(split.first=="--seed-beam-solution"){if(split.second!="true" && split.second!="false")matrixError("seed-beam-solution requires true or false");seedBeam=split.second=="true";}else if(split.first=="--candidate-archive"){if(split.second.getAsInteger(10,archiveCapacity) || archiveCapacity<2 || archiveCapacity>6)matrixError("candidate-archive requires capacity 2..6");}else if(split.first=="--archive-audit-paid-observations"){if(split.second!="true" && split.second!="false")matrixError("archive-audit-paid-observations requires true or false");auditPaidObservations=split.second=="true";}else if(split.first=="--archive-retention"){if(archiveRetentionSet || (split.second!="legacy-hamming-v2" && split.second!="paid-feature-frontier-v1" && split.second!="paid-static-feature-frontier-v1" && split.second!="calibrated-source-grammar-v1"))matrixError("invalid or duplicate archive-retention policy");archiveRetentionSet=true;archiveRetention=split.second.str();}else if(split.first=="--network-archive-v2"){if(split.second!="true" && split.second!="false")matrixError("network-archive-v2 requires true or false");networkArchiveV2=split.second=="true";}else if(split.first=="--network-objective"){if(split.second!="min-shfl" && split.second!="min-work")matrixError("unknown network objective");networkObjectiveOverride=split.second.str();}else if(split.first=="--minify-execution-json"){if(a=="--minify-execution-json")minifyExecutionJSON=true;else if(split.second=="true" || split.second=="false")minifyExecutionJSON=split.second=="true";else matrixError("minify-execution-json requires true or false");}else if(a=="--layout-profile")invocationProfileEnabled=true;else if(a=="--disable-cost-query-cache")disableCostQueryCache=true;else if(a=="--demand-ledger")demandLedger=true;else if(a=="--oracle")oracle=true;else if(split.first=="--cost-profile"){if(split.second.empty())matrixError("empty cost profile");}else if(split.first!="--case")matrixError("unknown option");}
 if(prepared.empty()||out.empty()||(selection!="L" && selection!="LI" && selection!="J")||(conversion!="C" && conversion!="O"))matrixError("invalid matrix options");fresh(out);
 LayoutProfile invocationProfile(invocationProfileEnabled);LayoutProfile::Invocation invocationProfileBinding(&invocationProfile);
 auto publishInvocationProfile=llvm::make_scope_exit([&]{if(invocationProfileEnabled)writeFile(out,"layout-profile.json",invocationProfile.serialize("success"));});
 auto domain=readJSON(prepared+"/domains.json"),metadata=readJSON(prepared+"/prepared.json");if(!domain.getAsObject() || !metadata.getAsObject())matrixError("malformed domain JSON");auto &d=*domain.getAsObject(),&meta=*metadata.getAsObject();preparedSchema(d,meta);if(meta.getString("cost_model")!=target->costModelID() || meta.getString("frequency_mode")!=frequency)matrixError("cost model or frequency identity mismatch");if(profile && meta.getString("cost_profile_sha256")!=getSM90CostProfileSHA256(*profile))matrixError("full cost profile identity mismatch");auto m=parseSourceFile<ModuleOp>(prepared+"/input.mlir",&context);if(!m)matrixError("input parse failed");if(auto executionDomain=m->getOperation()->getAttrOfType<StringAttr>("frisk.execution_domain")){if(meta.getString("execution_domain")!=executionDomain.getValue())matrixError("execution domain identity mismatch");}else if(meta.getString("execution_domain"))matrixError("unexpected execution domain metadata");auto claimedPackage=meta.getString("package_digest");if(!claimedPackage)matrixError("missing prepared package digest");std::string package=claimedPackage->str();auto packageMetadata=meta;packageMetadata.erase("package_digest");auto inputBuffer=llvm::MemoryBuffer::getFile(prepared+"/input.mlir");if(!inputBuffer || sha256((*inputBuffer)->getBuffer().str()+jsonText(domain)+jsonText(std::move(packageMetadata)))!=package)matrixError("prepared package integrity mismatch");auto graph=collectLayoutConstraints(*m,*target,LayoutCollectionMode::RelationsOnly);if(failed(graph))matrixError("relation reconstruction failed");restore(*graph,d,context);if(disableCostQueryCache)graph->setCostQueryCacheOptions(false,0);auto proofs=readProofs(d,meta,context);if(jsonText(domainJSON(*graph,proofs))!=jsonText(domain))matrixError("serialized graph/domain mismatch");
 const std::string preparedCostModel=target->costModelID().str();
 const std::string preparedProfileSHA=profile?getSM90CostProfileSHA256(*profile):"";
 if(!networkObjectiveOverride.empty()) {
   if(!profile || !profile->networkProxy || assignment.empty() || archiveCapacity)matrixError("network objective override requires a fixed-assignment replay of the explicit network proxy");
   profile->networkObjective=networkObjectiveOverride=="min-work"?shuffle_network::Objective::MinWork:shuffle_network::Objective::MinShuffle;
   target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,*profile);
 }
 std::unique_ptr<LayoutTarget> minWorkVariantTarget;
 if(networkArchiveV2) {
   if(archiveCapacity!=4 || !profile || !profile->networkProxy || profile->networkObjective!=shuffle_network::Objective::MinShuffle || !assignment.empty())matrixError("network archive v2 requires K4 auto search with the explicit network proxy");
   auto alternate=*profile;alternate.networkObjective=shuffle_network::Objective::MinWork;
   minWorkVariantTarget=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,alternate);
 }
 const auto retention=archiveRetention=="calibrated-source-grammar-v1"?ArchiveRetentionPolicy::CalibratedSourceGrammarV1:archiveRetention=="paid-feature-frontier-v1"?ArchiveRetentionPolicy::PaidFeatureFrontierV1:archiveRetention=="paid-static-feature-frontier-v1"?ArchiveRetentionPolicy::PaidStaticFeaturesV1:ArchiveRetentionPolicy::LegacyHammingV2;
 if(auditPaidObservations && !archiveCapacity)matrixError("paid observation audit requires an archive search");
 if(archiveRetentionSet && (!archiveCapacity || (retention!=ArchiveRetentionPolicy::LegacyHammingV2 && archiveCapacity!=4)))matrixError("archive-retention requires archive search; paid feature retention requires K4");
 const std::string effectiveNetworkObjective=profile && profile->networkObjective==shuffle_network::Objective::MinWork?"min-work":"min-shfl";
 SolverOptions options;options.memoizeCosts=!disableCostQueryCache;options.seedBeamSolution=seedBeam;options.policy.scope.informedLocal=selection=="LI";options.policy.conversion=conversion=="C"?ConversionPolicy::Independent:ConversionPolicy::Full;for(auto &id:*meta.getArray("producers"))options.policy.scope.producers.push_back(*id.getAsInteger());for(auto &id:*meta.getArray("producer_instructions"))options.policy.scope.producerInstructions.push_back(*id.getAsInteger());
 if(!replay.empty()){if(replay!="independent" && replay!="shared" && replay!="remat" && replay!="post")matrixError("invalid replay");options.policy.conversion=replay=="independent"?ConversionPolicy::Independent:replay=="shared"?ConversionPolicy::SharedOnly:ConversionPolicy::Full;}
 if(archiveCapacity && ((selection!="J" && selection!="LI") || !assignment.empty() || !replay.empty() || !forcedProducer.empty() || oracle || conversion!="C"))matrixError("candidate archive requires independent LI/J without replay, forced assignment or oracle");
  if(selection=="LI" && assignment.empty()) {
    auto localPolicy=options.policy;localPolicy.scope.fullGraph=false;
    auto unsupported=experimentPolicyUnsupportedReason(*graph,localPolicy);
    if(!unsupported.empty())matrixStop("unsupported",unsupported,false);
  }
  auto preSolveText=moduleText(*m);auto common=*graph;if(enumerateOracle(common,proofs,*target,{0,0,0},options.policy).fatal)matrixError("frozen proof integrity failed");SharedSearchBudget budget;options.sharedBudget=&budget;llvm::json::Object record;
  failureSearchBudget=&budget;failureSearchRecord=&record;failureSearchPerformed=assignment.empty();
  record["prepared_digest"]=package;record["common_domain_digest"]=proofs.commonDigest;
  if(profile && profile->networkProxy){
    auto baseAuto=*profile;baseAuto.networkObjective=shuffle_network::Objective::MinShuffle;
    record["prepared_cost_model"]=preparedCostModel;record["prepared_cost_profile_sha256"]=preparedProfileSHA;
    record["network_objective"]=effectiveNetworkObjective;record["cost_semantics"]=target->costCoverage().str();
    record["static_base_model_id"]=getSM90CalibratedCostModelID(baseAuto,LayoutBoundaryFrequencyMode::Legacy);
    record["static_base_profile_sha256"]=getSM90CostProfileSHA256(baseAuto);
    record["static_effective_model_id"]=target->costModelID().str();record["static_effective_profile_sha256"]=getSM90CostProfileSHA256(*profile);
    record["common_static_contract"]=networkStaticContractJSON(*profile,frequency);
  }
  record["cost_model"]=target->costModelID();record["selection"]=selection;
  record["seed_beam_solution"]=seedBeam;
  record["conversion_policy"]=options.policy.conversion==ConversionPolicy::Independent?"independent":options.policy.conversion==ConversionPolicy::SharedOnly?"shared-only":"full";
  if(profile)record["cost_profile_sha256"]=getSM90CostProfileSHA256(*profile);
  FailureOr<LayoutSolution> solution=failure();
 if(!forcedProducer.empty()) {
   if(!assignment.empty() || (forcedProducer!="A" && forcedProducer!="B") ||
      meta.getString("candidate_preparation")!="experimental-A/B-injection" || options.policy.scope.producers.size()!=1)
     matrixError("forced producer requires one controlled A/B producer and no assignment replay");
   auto id=options.policy.scope.producers.front();auto &v=graph->getVariable(id);
   uint64_t ordinal=forcedProducer=="A"?0:1;unsigned matches=0;
   for(const auto &candidate:v.candidates)matches+=candidate.stableOrdinal==ordinal;
   if(matches!=1)matrixError("forced producer ordinal missing or ambiguous");
   llvm::erase_if(v.candidates,[&](const auto &candidate){return candidate.stableOrdinal!=ordinal;});
   auto restrictedProofs=projectFrozenProofs(common,*graph,proofs);if(failed(restrictedProofs))matrixError("forced producer proof projection failed");
   record["producer_restriction"]=llvm::json::Object{{"variable",id},{"stable_name",v.stableName},{"ordinal",ordinal},{"label",forcedProducer},{"restricted_domain_digest",restrictedProofs->domainDigest},{"original_package_preserved",true}};
  }
  record["seed_beam_solution"]=seedBeam;
record["selection_restricted"]=!forcedProducer.empty();
  SmallVector<ArchivedLayoutCandidate,0> archivedCandidates;
  if(archiveCapacity){if(failed(propagateCommonToFixedPoint(*graph))){auto diagnosis=common;LayoutSolveFailure why;auto witness=solveLayoutGraph(diagnosis,*target,options,&why,nullptr,nullptr,true);if(succeeded(witness))matrixError("propagation contradicts a legal archive witness");documentedFailure(why);}solution=solveCandidateArchive(*graph,*target,selection=="J"?SelectionMode::Joint:SelectionMode::ProducerFirst,options,archiveCapacity,archivedCandidates,record,minWorkVariantTarget.get(),retention,auditPaidObservations,*m);}
  else if(assignment.empty()){if(failed(propagateCommonToFixedPoint(*graph))){auto diagnosis=common;LayoutSolveFailure why;auto witness=solveLayoutGraph(diagnosis,*target,options,&why,nullptr,nullptr,true);if(succeeded(witness))matrixError("propagation contradicts a legal full witness");documentedFailure(why);}solution=solveAblation(*graph,*target,selection!="J"?SelectionMode::ProducerFirst:SelectionMode::Joint,options,record);}else{auto input=readJSON(assignment);if(!input.getAsObject())matrixError("malformed assignment JSON");auto &o=*input.getAsObject();if(auto objective=o.getString("network_objective");objective && *objective!=effectiveNetworkObjective)matrixError("assignment network objective differs from explicit replay target");assignmentSchema(o,common);if(!o.getArray("assignment") || !o.getArray("instruction_bindings") || !o.getArray("reduction_bindings"))matrixError("missing complete assignment bindings");if(replay.empty() && o.getString("conversion_policy")!=(conversion=="C"?"independent":"full"))matrixError("cross-policy assignment rejected");llvm::json::Object originalComplete{{"assignment",llvm::json::Array(*o.getArray("assignment"))},{"instruction_bindings",llvm::json::Array(*o.getArray("instruction_bindings"))},{"reduction_bindings",llvm::json::Array(*o.getArray("reduction_bindings"))}};if(o.getString("complete_assignment_sha256")!=sha256(jsonText(std::move(originalComplete))))matrixError("assignment identity mismatch");if(o.getString("prepared_digest")!=package)matrixError("assignment domain mismatch");if(o.getString("network_objective")) {if(o.getString("cost_model")!=target->costModelID())matrixError("assignment effective cost model mismatch");llvm::json::Object programIdentity{{"complete_assignment_sha256",o.getString("complete_assignment_sha256")->str()},{"network_objective",effectiveNetworkObjective},{"cost_model",target->costModelID().str()}};if(o.getString("complete_program_sha256")!=sha256(jsonText(std::move(programIdentity))))matrixError("assignment program identity mismatch");}LayoutSolution s;s.policy=options.policy;for(auto &j:*o.getArray("assignment")){auto &v=*j.getAsObject();s.assignments[*v.getInteger("variable")]=candidate(context,*v.getObject("candidate"));}auto support=target->verifyExecutionSupport(common,CandidateAssignment{s.assignments});
  if(support.status!=ProofStatus::Proven)matrixStop(support.status==ProofStatus::Unknown?"unsupported":"no-feasible","assignment replay outside declared execution domain: "+support.reason,false);
  auto scored=scoreAssignment(common,CandidateAssignment{s.assignments},*target,s.policy);if(failed(scored))matrixError("assignment scoring failed");s.cost=scored->cost;s.conversions=std::move(scored->conversions);for(auto &j:*o.getArray("instruction_bindings")){auto &v=*j.getAsObject();s.instructionBindings[*v.getInteger("constraint")]=parseJSONAttribute(context,v,"binding");}for(auto &j:*o.getArray("reduction_bindings")){auto &v=*j.getAsObject();s.reductionBindings[*v.getInteger("constraint")]=parseJSONAttribute(context,v,"binding");}solution=s;}
 if(failed(solution)){auto kind=record.getString("failure_kind");if(kind && *kind!="fatal-integrity")matrixStop(kind->str(),record.getString("failure").value_or("documented search outcome").str(),false);matrixError(record.getString("failure").value_or("unclassified solver failure").str());}if(failed(verifySolvedLayoutGraph(common,*solution,*target,m->getLoc(),options.policy.conversion)))matrixError("solution verification failed");
 if(archiveCapacity){
    llvm::json::Array entries;unsigned index=0;
    for(const auto &candidate:archivedCandidates){
      const auto &x=candidate.solution;
      llvm::json::Object entry{{"index",index++},{"assignment",assignmentJSON(common,x.assignments)},
        {"instruction_bindings",bindingJSON(x.instructionBindings)},{"reduction_bindings",bindingJSON(x.reductionBindings)},
        {"cost",layoutCostJSON(x.cost)},{"prepared_digest",package},{"conversion_policy","independent"},
        {"local_incumbent",candidate.localIncumbent},{"joint_incumbent",candidate.jointIncumbent},
        {"probes_at_observation",candidate.probesAtObservation},{"refinement_at_observation",candidate.refinementAtObservation},
        {"certificate_verified",true}};
       if(record.getObject("candidate_archive_search")->getString("version")=="shared-network-candidate-archive-v3" || record.getObject("candidate_archive_search")->getString("version")=="shared-complete-candidate-archive-v2"){entry["retention_reason"]=candidate.retentionReason;entry["retention_structure"]=llvm::json::Object(candidate.retentionStructure);}
      if(networkArchiveV2) {
        entry["network_objective"]=candidate.networkObjective;entry["network_variant"]=candidate.networkVariant;
        entry["cost_model"]=candidate.networkVariant?minWorkVariantTarget->costModelID().str():target->costModelID().str();
        auto effective=*profile;effective.networkObjective=candidate.networkVariant?shuffle_network::Objective::MinWork:shuffle_network::Objective::MinShuffle;
        auto baseAuto=*profile;baseAuto.networkObjective=shuffle_network::Objective::MinShuffle;
        entry["cost_profile"]=costProfileJSON(effective);entry["cost_profile_sha256"]=getSM90CostProfileSHA256(effective);
        entry["static_base_model_id"]=getSM90CalibratedCostModelID(baseAuto,LayoutBoundaryFrequencyMode::Legacy);
        entry["static_base_profile_sha256"]=getSM90CostProfileSHA256(baseAuto);
        entry["static_effective_model_id"]=entry.getString("cost_model")->str();
        entry["static_effective_profile_sha256"]=getSM90CostProfileSHA256(effective);
        entry["common_static_contract"]=networkStaticContractJSON(effective,frequency);
      }
      llvm::json::Array origins;for(const auto &origin:candidate.origins)origins.push_back(origin);entry["origins"]=std::move(origins);
      llvm::json::Object identity{{"assignment",llvm::json::Array(*entry.getArray("assignment"))},{"instruction_bindings",llvm::json::Array(*entry.getArray("instruction_bindings"))},{"reduction_bindings",llvm::json::Array(*entry.getArray("reduction_bindings"))}};
      entry["complete_assignment_sha256"]=sha256(jsonText(std::move(identity)));
      if(networkArchiveV2)entry["complete_program_sha256"]=sha256(jsonText(llvm::json::Object{{"complete_assignment_sha256",entry.getString("complete_assignment_sha256")->str()},{"network_objective",candidate.networkObjective},{"cost_model",entry.getString("cost_model")->str()}}));
      entries.push_back(std::move(entry));
    }
    llvm::json::Object archive{{"schema_version",1},{"version",record.getObject("candidate_archive_search")->getString("version")->str()},{"prepared_digest",package},{"cost_model",target->costModelID()},
      {"capacity",archiveCapacity},{"entries",std::move(entries)},{"search",llvm::json::Object(*record.getObject("candidate_archive_search"))},
      {"export_contract","each entry is a complete --assignment replay; replay only reconstructs/verifies/materializes and performs no solve"}};
    writeFile(out,"candidate-archive.json",jsonText(std::move(archive)));
    record["candidate_archive_file"]="candidate-archive.json";
  }
  if(demandLedger){auto ledger=analyzeLayoutDemand(common,CandidateAssignment{solution->assignments},{},65536,true,profile?profile->networkObjective:shuffle_network::Objective::MinShuffle);if(options.policy.conversion!=ConversionPolicy::Independent){ledger.exact=false;ledger.reason="independent conversion required";}record["demand_coverage"]=layoutDemandJSON(ledger);if(profile && profile->survivingDemand){auto surviving=analyzeLayoutSurvivingDemand(common,CandidateAssignment{solution->assignments},{},65536,profile->networkObjective);if(options.policy.conversion!=ConversionPolicy::Independent){surviving.proven=false;surviving.reason="independent conversion required";}record["surviving_coverage"]=layoutSurvivingDemandJSON(surviving);}}
  if(profile){record["cost_profile_sha256"]=getSM90CostProfileSHA256(*profile);record["cost_profile"]=costProfileJSON(*profile);}
    record["budget_contract"]=llvm::json::Object{
      {"kind","shared-total-search-probes-v1"},
      {"search_performed",assignment.empty()},
      {"limit",budget.limit},{"refinement_limit",budget.refinementLimit},
      {"used",budget.used},{"refinement_used",budget.refinementUsed},
      {"includes","constraint extensions, complete objective scores and refinement probes; no second seed budget"},
      {"producer_first_full_transition_charged",selection=="J" || !assignment.empty() ? llvm::json::Value(nullptr) : llvm::json::Value(record.getBoolean("incumbent_full_score_budget_charged").value_or(false))},
      {"certificate_rescores","selected-certificate verification only, no candidate selection"}};
  llvm::json::Array searchComponents;
  for(const auto &component:solution->components)searchComponents.push_back(llvm::json::Object{
      {"root",component.root},{"used_beam",component.usedBeam},
      {"expanded",component.expanded},{"mrv_probes",component.mrvProbes},
      {"refinement_probes",component.refinementProbes},
      {"seed_probes",component.seedProbes},{"seed_scores",component.seedScores},
      {"seed_witnesses",component.seedWitnesses},
      {"has_seed_witness",component.seedWitnesses!=0},
      {"seed_start_cost",component.seedWitnesses ? llvm::json::Value(layoutCostJSON(component.seedStartCost)) : llvm::json::Value(nullptr)},
      {"seed_selected_before_refinement",component.seedSelected},
      {"seed_stop_reason",component.seedStopReason},
      {"beam_budget_fallback",component.beamBudgetFallback},
      {"selected_component_cost",layoutCostJSON(component.cost)},
      {"seed_comparison_scope",selection=="J" ? "full-objective-component" : "returned-stage-component"},
      {"exhaustive",component.exhaustive}});
  record["search_components"]=std::move(searchComponents);
  record["prepared_digest"]=package;record["common_domain_digest"]=proofs.commonDigest;record["assignment"]=assignmentJSON(common,solution->assignments);record["instruction_bindings"]=bindingJSON(solution->instructionBindings);record["reduction_bindings"]=bindingJSON(solution->reductionBindings);llvm::json::Object complete{{"assignment",llvm::json::Array(*record.getArray("assignment"))},{"instruction_bindings",llvm::json::Array(*record.getArray("instruction_bindings"))},{"reduction_bindings",llvm::json::Array(*record.getArray("reduction_bindings"))}};record["complete_assignment_sha256"]=sha256(jsonText(std::move(complete)));if(profile && profile->networkProxy)record["complete_program_sha256"]=sha256(jsonText(llvm::json::Object{{"complete_assignment_sha256",record.getString("complete_assignment_sha256")->str()},{"network_objective",effectiveNetworkObjective},{"cost_model",target->costModelID().str()}}));record["conversion_policy"]=options.policy.conversion==ConversionPolicy::Independent?"independent":options.policy.conversion==ConversionPolicy::SharedOnly?"shared-only":"full";record["selection"]=selection;record["selection_bypassed"]=!assignment.empty();record["cost"]=layoutCostJSON(solution->cost);record["conversions"]=conversionsJSON(common,solution->conversions);auto scored=scoreAssignment(common,CandidateAssignment{solution->assignments},*target,options.policy);if(failed(scored))matrixError("score failed");record["cost_contributions"]=std::move(scored->contributions);
 if(oracle){
 auto result=enumerateOracle(common,proofs,*target,{},options.policy);record["oracle"]=result.toJSON();if(result.fatal)matrixError(result.reason);
 auto comparison=[&](const CostEstimate &a,const CostEstimate &b){auto x=layoutCostJSON(a),y=layoutCostJSON(b);auto *xs=x.getArray("cost"),*ys=y.getArray("cost");llvm::json::Object answer{{"status","equal"}};for(unsigned i=0;i<xs->size();++i)if((*xs)[i]!=(*ys)[i]){answer["status"]="different";answer["first_component"]=i;answer["selected"]=llvm::json::Value((*xs)[i]);answer["oracle"]=llvm::json::Value((*ys)[i]);break;}return answer;};
 if(selection!="J") {
  auto local=options.policy;local.scope.fullGraph=false;auto first=enumerateOracle(common,proofs,*target,{},local);record["stage1_oracle"]=first.toJSON();if(first.fatal)matrixError(first.reason);
  bool sameTuple=bool(first.best);if(first.best)for(auto id:local.scope.producers)sameTuple &= first.best->assignments.lookup(id)==solution->assignments.lookup(id);if(first.best)record["stage1_oracle_assignment"]=assignmentJSON(common,first.best->assignments);if(first.status=="complete" && first.exhausted)record["stage1_producer_optimum_tuple"]=sameTuple;else record["stage1_producer_optimum_tuple"]=nullptr;
  auto pinned=common;for(auto id:local.scope.producers)llvm::erase_if(pinned.getVariable(id).candidates,[&](const auto &c){return c.value!=solution->assignments.lookup(id);});for(auto id:local.scope.producerInstructions){auto &c=pinned.getConstraints()[id];if(c.instruction){llvm::erase_if(c.instruction->tuples,[&](const auto &t){for(auto [var,encoding]:llvm::zip_equal(c.vars,t.encodings))if((llvm::is_contained(local.scope.producers,var)||pinned.getVariable(var).kind==LayoutKind::Instruction) && encoding!=solution->assignments.lookup(var))return true;return false;});for(auto v:c.vars)if(pinned.getVariable(v).kind==LayoutKind::Instruction)llvm::erase_if(pinned.getVariable(v).candidates,[&](const auto &x){return x.value!=solution->assignments.lookup(v);});}}
  auto projected=projectFrozenProofs(common,pinned,proofs);if(failed(projected))matrixError("L2 proof projection failed");auto second=enumerateOracle(pinned,*projected,*target,{},options.policy);record["stage2_oracle"]=second.toJSON();record["stage2_restriction_digest"]=projected->domainDigest;if(second.fatal)matrixError(second.reason);if(second.best){auto c=comparison(solution->cost,second.best->cost);c["reference_kind"]="fixed-producer-L2";c["reference_status"]=second.status=="complete" && second.exhausted?"complete-optimum":"best-so-far-incumbent";record["full_cost_oracle_comparison"]=std::move(c);}
 }else if(result.best){auto c=comparison(solution->cost,result.best->cost);c["reference_kind"]="global";c["reference_status"]=result.status=="complete" && result.exhausted?"complete-optimum":"best-so-far-incumbent";record["full_cost_oracle_comparison"]=std::move(c);}
}
 bool post=replay.empty()?conversion=="O":replay=="post";record["post_opt"]=post;record["cost_model"]=target->costModelID();record["frequency_mode"]=frequency;record["scope"]=selection=="LI"?"informed-producer-first-then-full":selection=="L"?"producer-first-then-full":"full";record["failure_kind"]="";
 if(moduleText(*m)!=preSolveText)matrixError("selection mutated input IR");record["selection_preserved_input_ir"]=true;SmallVector<std::string> expectedContracts;for(const auto &entry:solution->instructionBindings){std::string text;llvm::raw_string_ostream(text)<<entry.second;expectedContracts.push_back(text);}llvm::sort(expectedContracts);writeFile(out,"record.json",jsonText(llvm::json::Object(record)));writeFile(out,"prepared.mlir",moduleText(*m));if(demandLedger)markDemandLineage(common);
 if(failed(materializeLayouts(*m,common,*solution,true,target.get(),options.policy.conversion))||failed(preserveStorageContracts(*m))||failed(verify(*m))||failed(verifyMaterializedLayouts(*m,*target)))matrixError("materialization verification failed");SmallVector<std::string> actualContracts;m->walk([&](Operation *op){if(auto a=op->getAttr("frisk.mma_contract")){std::string t;llvm::raw_string_ostream(t)<<a;actualContracts.push_back(t);}if(auto a=op->getAttr("frisk.copy_contract")){std::string t;llvm::raw_string_ostream(t)<<a;actualContracts.push_back(t);}});llvm::sort(actualContracts);if(actualContracts!=expectedContracts)matrixError("materialized contracts differ from pinned assignment");record["instruction_contracts_verified"]=true;writeFile(out,"materialized.mlir",moduleText(*m));record["materialized_diagnostics"]=llvm::json::Object{{"arithmetic",collectLayoutIRArithmetic(*m)},{"resources",collectLayoutIRResources(*m)}};if(post){PassManager pm(&context);pm.addPass(createOptimizeLayoutConversionsPass(LayoutBoundaryFrequencyMode::Legacy,profile?&*profile:nullptr));if(failed(pm.run(*m)))matrixError("post optimization failed");}
 if(demandLedger && failed(resolveDemandLineage(*m)))matrixError("execution demand operand lineage arity changed");
  if(failed(verify(*m))||failed(verifyMaterializedLayouts(*m,*target)))matrixError("post verification failed");record["output_diagnostics"]=llvm::json::Object{{"arithmetic",collectLayoutIRArithmetic(*m)},{"resources",collectLayoutIRResources(*m)}};auto text=moduleText(*m);writeFile(out,"output.mlir",text);auto reparsed=parseSourceString<ModuleOp>(text,&context);if(!reparsed || failed(verify(*reparsed))||failed(verifyMaterializedLayouts(*reparsed,*target)))matrixError("reparse verification failed");writeFile(out,"reparsed.mlir",moduleText(*reparsed));
 PassManager pm(&context);auto pass=createExportExecutionPass();std::string exportOptions="output="+out+"/execution.json";if(minifyExecutionJSON)exportOptions+=" minify=true";exportOptions+=" network-objective="+effectiveNetworkObjective;if(failed(pass->initializeOptions(exportOptions,[](const Twine &){return failure();})))matrixError("export options");pm.addPass(std::move(pass));if(failed(pm.run(*m)))matrixStop("backend-unsupported","backend unsupported",false);record["status"]="success";writeFile(out,"record.json",jsonText(std::move(record)));return 0;
 }
