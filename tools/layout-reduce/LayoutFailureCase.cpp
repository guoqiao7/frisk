#include "LayoutFailureCase.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Builders.h"
#include <map>
#include <set>
#include <tuple>

using namespace mlir;
using namespace mlir::frisk;
using namespace mlir::frisk::testing;
namespace {
// This target is explicitly a finite fixture oracle, not a hardware target.
class FiniteTarget final : public LayoutTarget {
public:
  void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &, Attribute a, Location) const override {
    return success(isa<StringAttr>(a));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {
    return CostVector{};
  }
};
bool keys(const llvm::json::Object &o, std::initializer_list<StringRef> allowed) {
  if (o.size() != allowed.size()) return false;
  for (auto k : allowed) if (!o.get(k)) return false;
  return true;
}
bool strings(const llvm::json::Array *a, std::vector<std::string> &out) {
  if (!a) return false;
  for (const auto &v : *a) {
    auto s = v.getAsString(); if (!s || s->empty()) return false;
    out.push_back(s->str());
  }
  return true;
}
llvm::json::Array array(const std::vector<std::string> &v) {
  llvm::json::Array result;
  for (auto &s : v) result.push_back(s);
  return result;
}
StringRef kindName(LayoutSolveFailureKind k) {
  switch (k) {
#define KIND(K) case LayoutSolveFailureKind::K: return #K
    KIND(None); KIND(InvalidInput); KIND(EmptyDomain); KIND(HardConflict);
    KIND(Unsupported); KIND(SearchBudget); KIND(SearchIncomplete); KIND(CostFailure);
#undef KIND
  }
  llvm_unreachable("failure kind");
}
bool valid(const LayoutFailureCase &c) {
  if (!c.options.beamWidth || !c.options.maxExpandedStates ||
      !c.options.exactCombinationLimit || c.variables.size() > 8 ||
      c.relations.size() > 16) return false;
  std::set<std::string> ids, relations;
  for (const auto &v : c.variables) {
    if (v.id.empty() || !ids.insert(v.id).second || v.candidates.size() > 4) return false;
    std::set<std::string> candidates;
    for (auto &a : v.candidates)
      if (a.empty() || !candidates.insert(a).second) return false;
  }
  for (const auto &r : c.relations) {
    if (r.id.empty() || !relations.insert(r.id).second || r.rule.empty()) return false;
    if (r.kind != "same" && r.kind != "require") return false;
    if (r.vars.size() != (r.kind == "same" ? 2u : 1u)) return false;
    if ((r.kind == "require") != !r.required.empty()) return false;
    for (auto &v : r.vars) if (!ids.count(v)) return false;
  }
  return true;
}
std::vector<LayoutFailureCase> deletions(const LayoutFailureCase &c) {
  std::vector<LayoutFailureCase> out;
  for (size_t i=0;i<c.relations.size();++i) {
    auto t=c; t.relations.erase(t.relations.begin()+i); out.push_back(std::move(t));
  }
  for (size_t i=0;i<c.variables.size();++i) {
    bool referenced=false;
    for (auto &r:c.relations)
      for (auto &v:r.vars) referenced |= v==c.variables[i].id;
    if (!referenced) {
      auto t=c; t.variables.erase(t.variables.begin()+i); out.push_back(std::move(t));
    }
  }
  for (size_t i=0;i<c.variables.size();++i)
    for (size_t j=0;j<c.variables[i].candidates.size();++j) {
      auto t=c; auto &v=t.variables[i].candidates;
      v.erase(v.begin()+j); out.push_back(std::move(t));
    }
  return out;
}
void canonicalize(LayoutFailureCase &c) {
  llvm::sort(c.variables, [](auto &a,auto &b){return a.id<b.id;});
  llvm::sort(c.relations, [](auto &a,auto &b){return a.id<b.id;});
  for (auto &v:c.variables) llvm::sort(v.candidates);
}
}
bool FailureSignature::operator==(const FailureSignature &b) const {
  return std::tie(kind,variable,relation,rule,reason)==
         std::tie(b.kind,b.variable,b.relation,b.rule,b.reason);
}
std::string mlir::frisk::testing::serializeFailureCase(const LayoutFailureCase &input) {
  auto c=input; canonicalize(c);
  llvm::json::Array vars, rels;
  for (auto &v:c.variables)
    vars.push_back(llvm::json::Object{{"id",v.id},{"candidates",array(v.candidates)}});
  for (auto &r:c.relations)
    rels.push_back(llvm::json::Object{{"id",r.id},{"kind",r.kind},
      {"vars",array(r.vars)},{"required",r.required},{"rule",r.rule},{"reason",r.reason}});
  llvm::json::Object root{{"schema",1},{"domain","finite-same-require-v1"},
    {"seed",c.seed},{"host",c.host},{"variables",std::move(vars)},{"relations",std::move(rels)},
    {"options",llvm::json::Object{{"exact",c.options.exactCombinationLimit},
      {"beam",c.options.beamWidth},{"expanded",c.options.maxExpandedStates}}},
    {"failure",llvm::json::Object{{"kind",c.signature.kind},{"variable",c.signature.variable},
      {"relation",c.signature.relation},{"rule",c.signature.rule},{"reason",c.signature.reason}}}};
  std::string text; llvm::raw_string_ostream(text) << llvm::formatv("{0:2}",llvm::json::Value(std::move(root))) << '\n';
  return text;
}
FailureOr<LayoutFailureCase> mlir::frisk::testing::parseFailureCase(
    StringRef text, MLIRContext &context, std::string &error) {
  error="unsupported or malformed finite-same-require-v1 failure case";
  auto parsed=llvm::json::parse(text);
  if (!parsed) { llvm::consumeError(parsed.takeError()); return failure(); }
  auto *o=parsed->getAsObject();
  if (!o || !keys(*o,{"schema","domain","seed","host","variables","relations","options","failure"}) ||
      o->getInteger("schema")!=1 || o->getString("domain")!="finite-same-require-v1") return failure();
  auto seed=o->getInteger("seed"); auto host=o->getString("host");
  auto *opts=o->getObject("options"); auto *sig=o->getObject("failure");
  auto *vars=o->getArray("variables"); auto *rels=o->getArray("relations");
  if (!seed || *seed<0 || !host || !opts || !sig || !vars || !rels ||
      !keys(*opts,{"exact","beam","expanded"}) ||
      !keys(*sig,{"kind","variable","relation","rule","reason"})) return failure();
  auto exact=opts->getInteger("exact"), beam=opts->getInteger("beam"), expanded=opts->getInteger("expanded");
  if (!exact || *exact<=0 || !beam || *beam<=0 || *beam>UINT32_MAX || !expanded || *expanded<=0) return failure();
  LayoutFailureCase c; c.seed=*seed; c.host=host->str(); c.options={uint64_t(*exact),unsigned(*beam),uint64_t(*expanded)};
  auto read=[&](const llvm::json::Object &obj,StringRef key,std::string &dst){
    auto s=obj.getString(key); if (!s) return false; dst=s->str(); return true;
  };
  if (!read(*sig,"kind",c.signature.kind) || !read(*sig,"variable",c.signature.variable) ||
      !read(*sig,"relation",c.signature.relation) || !read(*sig,"rule",c.signature.rule) ||
      !read(*sig,"reason",c.signature.reason)) return failure();
  bool known=false;
  for (auto k:{LayoutSolveFailureKind::EmptyDomain,LayoutSolveFailureKind::HardConflict,
      LayoutSolveFailureKind::Unsupported,LayoutSolveFailureKind::SearchBudget,
      LayoutSolveFailureKind::SearchIncomplete,LayoutSolveFailureKind::CostFailure})
    known |= c.signature.kind==kindName(k);
  if (!known) return failure();
  for (auto &v:*vars) {
    auto *obj=v.getAsObject(); FailureVariable var;
    if (!obj || !keys(*obj,{"id","candidates"}) || !read(*obj,"id",var.id) ||
        !strings(obj->getArray("candidates"),var.candidates)) return failure();
    c.variables.push_back(std::move(var));
  }
  for (auto &v:*rels) {
    auto *obj=v.getAsObject(); FailureRelation r;
    if (!obj || !keys(*obj,{"id","kind","vars","required","rule","reason"}) ||
        !read(*obj,"id",r.id) || !read(*obj,"kind",r.kind) || !read(*obj,"required",r.required) ||
        !read(*obj,"rule",r.rule) || !read(*obj,"reason",r.reason) ||
        !strings(obj->getArray("vars"),r.vars)) return failure();
    c.relations.push_back(std::move(r));
  }
  context.getOrLoadDialect<FriskDialect>();
  if (!valid(c) || !parseSourceString<ModuleOp>(c.host,&context) ||
      failed(rebuildFailureGraph(c,context))) return failure();
  if (!c.signature.variable.empty() && !llvm::any_of(c.variables,[&](auto &v){return v.id==c.signature.variable;})) return failure();
  if (!c.signature.relation.empty() && !llvm::any_of(c.relations,[&](auto &r){return r.id==c.signature.relation;})) return failure();
  canonicalize(c); error.clear(); return c;
}
FailureOr<LayoutConstraintGraph> mlir::frisk::testing::rebuildFailureGraph(
    const LayoutFailureCase &input, MLIRContext &context) {
  if (!valid(input)) return failure();
  auto c=input; canonicalize(c); Builder b(&context); LayoutConstraintGraph graph;
  for (auto &v:c.variables) {
    auto id=graph.addVariable(LayoutKind::Distributed,RankedTensorType::get({4},b.getF32Type()),v.id);
    auto &var=graph.getVariable(id); var.state=LayoutState::CandidateSet;
    for (auto &a:v.candidates) var.candidates.push_back({b.getStringAttr(a),kInvalidProvenanceID,var.candidates.size()});
  }
  for (auto &r:c.relations) {
    SmallVector<LayoutVarID> ids;
    for (auto &v:r.vars) ids.push_back(*graph.lookupVariable(v));
    auto id=graph.addConstraint(r.kind=="same"?ConstraintKind::SameLayout:ConstraintKind::RequireEncoding,
      ConstraintStrength::Hard,ids,nullptr,r.rule,r.reason,
      r.kind=="require"?Attribute(b.getStringAttr(r.required)):Attribute());
    graph.getConstraints()[id].stableUseKey=r.id;
  }
  if (failed(graph.finalize(b.getUnknownLoc()))) return failure();
  return graph;
}
FailureOr<FailureSignature> mlir::frisk::testing::replayFailureCase(
    const LayoutFailureCase &c, MLIRContext &context) {
  if (!parseSourceString<ModuleOp>(c.host,&context)) return failure();
  auto graph=rebuildFailureGraph(c,context); if (failed(graph)) return failure();
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
  FiniteTarget target; LayoutSolveFailure why;
  if (succeeded(solveLayoutGraph(*graph,target,c.options,&why))) return failure();
  FailureSignature s; s.kind=kindName(why.kind).str(); s.reason=why.reason;
  if (why.variable<graph->getVariables().size()) s.variable=graph->getVariable(why.variable).stableName;
  if (why.constraint<graph->getConstraints().size()) {
    auto &r=graph->getConstraint(why.constraint); s.relation=r.stableUseKey;
    auto &p=graph->getProvenances()[r.provenance]; s.rule=p.rule;
    s.reason += " | "+p.reason;
  }
  return s;
}
FailureOr<LayoutFailureCase> mlir::frisk::testing::reduceFailureCase(
    const LayoutFailureCase &input, MLIRContext &context) {
  auto s=replayFailureCase(input,context);
  if (failed(s) || !(*s==input.signature)) return failure();
  auto c=input; canonicalize(c);
  for (;;) {
    bool changed=false;
    for (auto &t:deletions(c)) {
      auto signature=replayFailureCase(t,context);
      if (succeeded(signature) && *signature==c.signature) {
        c=std::move(t); changed=true; break;
      }
    }
    if (!changed) return c;
  }
}
FailureOr<LayoutConstraintGraph> mlir::frisk::testing::reduceFailingLayoutGraph(
    const LayoutConstraintGraph &graph,
    llvm::function_ref<bool(const LayoutConstraintGraph &)> fails) {
  // Raw entry is deliberately narrower than the owning file format. Reject
  // every unsupported field before copying; never drop live IR references.
  LayoutFailureCase c; MLIRContext *ctx=nullptr;
  // The file fixture uses candidate strings as identities. The raw API also
  // accepts caller-supplied ordinals: retain them separately across rebuilding
  // and deletion instead of replacing surviving candidates' identities.
  std::map<std::pair<std::string,std::string>,uint64_t> ordinals;
  if (!graph.getRegionEdges().empty() ||
      graph.getProvenances().size()!=graph.getConstraints().size()) return failure();
  for (auto &v:graph.getVariables()) {
    if (v.anchor || v.value || v.use || v.functionResult || v.storageAlias || v.operationExecution ||
        v.requiredThreads || v.instructionRole || v.reductionResult || v.candidatePreparationUnsupported ||
        v.kind!=LayoutKind::Distributed || v.state!=LayoutState::CandidateSet ||
        !isa<RankedTensorType>(v.shapedType)) return failure();
    auto type=cast<RankedTensorType>(v.shapedType);
    if (type.getShape()!=ArrayRef<int64_t>{4} || !type.getElementType().isF32() || type.getEncoding()) return failure();
    ctx=v.shapedType.getContext(); FailureVariable var{v.stableName,{}};
    for (auto &a:v.candidates) {
      auto str=dyn_cast<StringAttr>(a.value); if (!str || a.provenance!=kInvalidProvenanceID) return failure();
      var.candidates.push_back(str.getValue().str());
      ordinals.emplace(std::make_pair(v.stableName,str.getValue().str()),a.stableOrdinal);
    }
    c.variables.push_back(std::move(var));
  }
  for (auto &r:graph.getConstraints()) {
    if ((r.kind!=ConstraintKind::SameLayout && r.kind!=ConstraintKind::RequireEncoding) ||
        r.strength!=ConstraintStrength::Hard || r.use || r.coordinateTransform ||
        r.existingConversion || r.access!=AccessKind::Read || r.requiredThreads || r.instruction || r.reduction || r.rootStorage ||
        r.provenance>=graph.getProvenances().size()) return failure();
    auto &p=graph.getProvenances()[r.provenance];
    if (p.source || p.parent) return failure();
    FailureRelation rel{r.stableUseKey,r.kind==ConstraintKind::SameLayout?"same":"require",p.rule,p.reason,{},{}};
    if (rel.id.empty()) return failure();
    for (auto id:r.vars) {if(id>=graph.getVariables().size()) return failure(); rel.vars.push_back(graph.getVariable(id).stableName);}
    if (r.requiredEncoding) {
      auto a=dyn_cast<StringAttr>(r.requiredEncoding); if(!a) return failure(); rel.required=a.getValue().str();
    }
    c.relations.push_back(std::move(rel));
  }
  if (!ctx || failed(graph.verifyInvariants(UnknownLoc::get(ctx))) || !fails(graph)) return failure();
  canonicalize(c);
  auto rebuild=[&](const LayoutFailureCase &input)->FailureOr<LayoutConstraintGraph> {
    auto result=rebuildFailureGraph(input,*ctx);
    if(failed(result)) return failure();
    for(auto &v:result->getVariables()) for(auto &a:v.candidates) {
      auto key=std::make_pair(v.stableName,cast<StringAttr>(a.value).getValue().str());
      auto found=ordinals.find(key);
      if(found==ordinals.end()) return failure();
      a.stableOrdinal=found->second;
    }
    if(failed(result->verifyInvariants(UnknownLoc::get(ctx)))) return failure();
    return result;
  };
  // Normalization may alter insertion-order-dependent predicates even when
  // all supported semantic fields are preserved. Such inputs fail closed.
  auto initial=rebuild(c);
  if(failed(initial) || !fails(*initial)) return failure();
  for (;;) {
    bool changed=false;
    for (auto &t:deletions(c)) {
      auto g=rebuild(t);
      if (succeeded(g) && fails(*g)) {c=std::move(t); changed=true; break;}
    }
    if(!changed) {
      auto result=rebuild(c);
      if(failed(result) || !fails(*result)) return failure();
      return result;
    }
  }
}
