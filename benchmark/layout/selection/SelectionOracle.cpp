#include "SelectionOracle.h"
#include "SelectionExperiment.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "llvm/ADT/STLExtras.h"
#include <chrono>
#include <functional>
#include "llvm/Support/FormatVariadic.h"

namespace mlir::frisk::selection {
namespace {
std::string attributeText(Attribute a) {return a ? layoutCandidateKey(a) : "";}
std::string typeText(Type t) {std::string s;llvm::raw_string_ostream(s)<<t;return s;}
StringRef proofName(ProofStatus s) {
  return s==ProofStatus::Proven ? "proven" : s==ProofStatus::Disproven ? "disproven" : "unknown";
}
llvm::json::Object candidateJSON(Attribute a) {
  auto text=attributeText(a);return llvm::json::Object{{"id",sha256(text)},{"attribute",text}};
}
llvm::json::Object aliasJSON(const StorageAliasInfo &a) {
  std::string map;llvm::raw_string_ostream(map)<<a.viewToRoot;
  return llvm::json::Object{{"root_key",a.rootKey},{"root_type",typeText(a.rootType)},
    {"view_type",typeText(a.viewType)},{"view_to_root",map},
    {"lower_bit",a.lowerBit},{"upper_bit",a.upperBit},{"root_alignment",a.rootAlignment},
    {"alignment_evidence",a.alignmentEvidence}};
}
class Contributions : public LayoutCostObserver {
public:
  CostEstimate sum;
  llvm::json::Array entries;
  void recordCost(StringRef kind,uint64_t owner,const CostEstimate &cost) override {
    sum=addLayoutCosts(sum,cost);
    entries.push_back(llvm::json::Object{{"kind",kind.str()},{"owner",owner},{"cost",layoutCostJSON(cost)}});
  }
};
bool sameCost(const CostEstimate &a,const CostEstimate &b) {
  return !(a.cost<b.cost) && !(b.cost<a.cost) && a.saturated==b.saturated;
}
}

std::optional<std::pair<int64_t,int64_t>> geometryGrid(Attribute a) {
  auto d=dyn_cast_or_null<DictionaryAttr>(a);
  if(!d || d.getAs<StringAttr>("kind")==StringAttr{} || d.getAs<StringAttr>("kind").getValue()!="sm90-mma-geometry") return {};
  auto grid=d.getAs<DenseI64ArrayAttr>("grid");if(!grid || grid.size()!=2) return {};
  return std::pair<int64_t,int64_t>{grid[0],grid[1]};
}
std::set<std::string> assignmentGeometries(const LayoutConstraintGraph &g,const CandidateAssignment &a) {
  std::set<std::string> result;
  for(const auto &v:g.getVariables()) if(v.kind==LayoutKind::Instruction)
    if(auto grid=geometryGrid(a.values.lookup(v.id))) result.insert(std::to_string(grid->first)+","+std::to_string(grid->second));
  return result;
}
FailureOr<LayoutConstraintGraph> restrictGeometry(const LayoutConstraintGraph &g,
    std::optional<std::pair<int64_t,int64_t>> grid,Location loc) {
  auto branch=g;
  if(grid) {
    bool found=false;
    for(auto &v:branch.getVariables()) if(v.kind==LayoutKind::Instruction &&
        llvm::any_of(v.candidates,[](const auto &c){return bool(geometryGrid(c.value));})) {
      llvm::erase_if(v.candidates,[&](const auto &c){return geometryGrid(c.value)!=grid;});
      if(v.candidates.empty()) return failure();
      found=true;
    }
    if(!found) return failure();
    for(auto &c:branch.getConstraints()) if(c.instruction) {
      llvm::erase_if(c.instruction->tuples,[&](const auto &tuple) {
        for(auto [id,a]:llvm::zip_equal(c.vars,tuple.encodings))
          if(branch.getVariable(id).kind==LayoutKind::Instruction && geometryGrid(a) && geometryGrid(a)!=grid) return true;
        return false;
      });
    }
  }
  if(failed(branch.verifyInvariants(loc))) return failure();
  return branch;
}

namespace {
std::string domainDigest(const LayoutConstraintGraph &g) {
  return sha256(llvm::formatv("{0}",llvm::json::Value(domainJSON(g,FrozenProofs{}))).str());
}
std::string coverageDigest(const FrozenProofs &proofs) {
  llvm::json::Array candidates;
  for(const auto &e:proofs.candidates)candidates.push_back(llvm::json::Object{{"variable",e.variable},{"candidate",attributeText(e.candidate)},{"valid",e.valid},{"capacity",e.capacity}});
  return sha256(llvm::formatv("{0}",llvm::json::Value(llvm::json::Object{{"relations",proofs.toJSON()},{"candidates",std::move(candidates)}})).str());
}
}
FrozenProofs freezeRelationProofs(const LayoutConstraintGraph &g, LayoutTarget *target) {
  FrozenProofs result;
  for(const auto &c:g.getConstraints()) {
    if(c.strength!=ConstraintStrength::Hard || c.instruction || c.reduction) continue;
    if(c.vars.empty()) {result.entries.push_back({c.id,UINT32_MAX,UINT32_MAX,{}, {},
      {ProofStatus::Unknown,{},"hard relation has no endpoints"}});continue;}
    auto lhs=c.vars.front();
    if(c.kind==ConstraintKind::RequireEncoding || isSupportedUnaryLayoutConstraint(c.kind)) {
      for(const auto &a:g.getVariable(lhs).candidates) {
        auto proof=c.kind==ConstraintKind::RequireEncoding
          ? LayoutProof{a.value==c.requiredEncoding ? ProofStatus::Proven : ProofStatus::Disproven,{},"exact required encoding"}
          : proveUnaryLayoutConstraint(g,c,a.value);
        result.entries.push_back({c.id,lhs,lhs,a.value,{},std::move(proof)});
      }
    } else if(isSupportedLayoutRelation(c.kind) && c.vars.size()>=2) {
      for(auto rhs:llvm::drop_begin(c.vars)) for(const auto &a:g.getVariable(lhs).candidates)
        for(const auto &b:g.getVariable(rhs).candidates)
          result.entries.push_back({c.id,lhs,rhs,a.value,b.value,
            proveLayoutRelation(g,c,lhs,a.value,rhs,b.value)});
    } else for(const auto &a:g.getVariable(lhs).candidates)
      result.entries.push_back({c.id,lhs,lhs,a.value,{}, {ProofStatus::Unknown,{},"unsupported hard relation"}});
  }
  if(target)for(const auto &v:g.getVariables())for(const auto &c:v.candidates) {
    auto loc=v.anchor?v.anchor->getLoc():UnknownLoc::get(v.shapedType.getContext());
    ScopedDiagnosticHandler quiet(loc.getContext(),[](Diagnostic &){return success();});
    result.candidates.push_back({v.id,c.value,succeeded(target->verifyCandidate(v,c.value,loc)),succeeded(verifyLayoutStorageCapacity(g,v,c.value))});
  }
  result.commonDigest=result.domainDigest=domainDigest(g);result.coverageDigest=coverageDigest(result);
  return result;
}
FailureOr<FrozenProofs> projectFrozenProofs(const LayoutConstraintGraph &common,const LayoutConstraintGraph &restricted,const FrozenProofs &proofs) {
 if(domainDigest(common)!=proofs.domainDigest || coverageDigest(proofs)!=proofs.coverageDigest || common.getVariables().size()!=restricted.getVariables().size() || common.getConstraints().size()!=restricted.getConstraints().size())return failure();
 auto restored=restricted;
 for(auto &v:restored.getVariables()) {
  const auto &original=common.getVariable(v.id);
  for(const auto &c:v.candidates)if(!llvm::any_of(original.candidates,[&](const auto &x){return x.value==c.value && x.stableOrdinal==c.stableOrdinal && x.provenance==c.provenance;}))return failure();
  v.candidates=original.candidates;
 }
 for(auto &c:restored.getConstraints())if(c.instruction) {
  const auto &original=common.getConstraint(c.id);
  if(!original.instruction || c.instruction->binding!=original.instruction->binding)return failure();
  for(const auto &t:c.instruction->tuples)if(!llvm::any_of(original.instruction->tuples,[&](const auto &x){return x.encodings==t.encodings && x.binding==t.binding;}))return failure();
  c.instruction=original.instruction;
 }
 if(domainDigest(restored)!=proofs.domainDigest)return failure();
 auto result=proofs;result.domainDigest=domainDigest(restricted);return result;
}
llvm::json::Array FrozenProofs::toJSON() const {
  llvm::json::Array result;
  for(const auto &e:entries) {
    llvm::json::Array witness;for(auto n:e.proof.counterexample) witness.push_back(n);
    result.push_back(llvm::json::Object{{"constraint",e.constraint},{"lhs",e.lhs},{"rhs",e.rhs},
      {"a",candidateJSON(e.a)},{"b",candidateJSON(e.b)},{"status",proofName(e.proof.status)},
      {"reason",e.proof.reason},{"counterexample",std::move(witness)}});
  }
  return result;
}
llvm::json::Object domainJSON(const LayoutConstraintGraph &g,const FrozenProofs &proofs) {
  llvm::json::Array variables,constraints,provenances,regions;
  for(const auto &v:g.getVariables()) {
    llvm::json::Array candidates;
    for(const auto &c:v.candidates) {
      auto candidate=candidateJSON(c.value);candidate["ordinal"]=c.stableOrdinal;
      candidate["provenance"]=c.provenance;candidates.push_back(std::move(candidate));
    }
    llvm::json::Object var{{"id",v.id},{"name",v.stableName},{"kind",int(v.kind)},
      {"type",typeText(v.shapedType)},{"state",int(v.state)},{"required_threads",v.requiredThreads},
      {"instruction_role",v.instructionRole},{"reduction_result",v.reductionResult},
      {"candidate_preparation_unsupported",v.candidatePreparationUnsupported},{"candidates",std::move(candidates)}};
    if(v.operationExecution) var["operation_execution"]=llvm::json::Object{
      {"threads",v.operationExecution->threads},{"vector_bytes",v.operationExecution->vectorBytes},
      {"writer_policy",v.operationExecution->writerPolicy}};
    if(v.storageAlias) var["storage_alias"]=aliasJSON(*v.storageAlias);
    if(v.functionResult) var["function_result"]=*v.functionResult;
    if(v.anchor) var["anchor_operation"]=v.anchor->getName().getStringRef();
    if(v.use) var["use_operand"]=v.use->getOperandNumber();
    variables.push_back(std::move(var));
  }
  for(const auto &c:g.getConstraints()) {
    llvm::json::Array ids;for(auto id:c.vars) ids.push_back(id);
    llvm::json::Object entry{{"id",c.id},{"kind",stringifyConstraintKind(c.kind)},
      {"strength",stringifyConstraintStrength(c.strength)},{"variables",std::move(ids)},
      {"required_encoding",attributeText(c.requiredEncoding)},{"coordinate_transform",attributeText(c.coordinateTransform)},
      {"access",int(c.access)},{"existing_conversion",c.existingConversion},{"stable_use_key",c.stableUseKey},
      {"required_threads",c.requiredThreads},{"provenance",c.provenance}};
    if(c.instruction) {
      llvm::json::Array tuples;
      for(const auto &t:c.instruction->tuples) {
        llvm::json::Array encodings;for(auto a:t.encodings) encodings.push_back(candidateJSON(a));
        tuples.push_back(llvm::json::Object{{"encodings",std::move(encodings)},{"binding",attributeText(t.binding)},{"proof","proven"}});
      }
      entry["instruction"]=llvm::json::Object{{"binding",attributeText(c.instruction->binding)},
        {"support_complete",true},{"tuples",std::move(tuples)}};
    }
    if(c.reduction) {
      llvm::json::Array pairs;
      for(const auto &p:c.reduction->pairs) pairs.push_back(llvm::json::Object{
        {"source",candidateJSON(p.sourceEncoding)},{"result",candidateJSON(p.resultEncoding)},
        {"binding",attributeText(p.binding)},{"proof","proven"}});
      entry["reduction"]=llvm::json::Object{{"axis",c.reduction->axis},{"binding",attributeText(c.reduction->binding)},
        {"support_complete",true},{"pairs",std::move(pairs)}};
    }
    if(c.broadcast) entry["broadcast_axes"]=attributeText(c.broadcast->axes);
    if(c.rootStorage) entry["root_storage_contract"]=llvm::json::Object{
      {"layout",attributeText(c.rootStorage->layout)},{"root",aliasJSON(c.rootStorage->root)}};
    constraints.push_back(std::move(entry));
  }
  for(const auto &p:g.getProvenances()) {
    llvm::json::Object entry{{"id",p.id},{"rule",p.rule},{"reason",p.reason}};
    if(p.parent) entry["parent"]=*p.parent;else entry["parent"]=nullptr;
    provenances.push_back(std::move(entry));
  }
  for(const auto &e:g.getRegionEdges()) regions.push_back(llvm::json::Object{
    {"source",e.source},{"target",e.target},{"constraint",e.constraint},
    {"kind",stringifyRegionLayoutEdgeKind(e.kind)},{"key",e.stableKey},{"slot",e.slot}});
  return llvm::json::Object{{"schema_version",1},{"stage","prepared-before-propagation"},
    {"support_table_policy","collector-completed-instruction-and-reduction-proofs"},
    {"support_absence_meaning","outside finite proven support domain, not hardware impossibility"},
    {"variables",std::move(variables)},{"constraints",std::move(constraints)},
    {"provenances",std::move(provenances)},{"region_edges",std::move(regions)},
    {"relation_proofs",proofs.toJSON()}};
}

FailureOr<ScoredAssignment> scoreAssignment(const LayoutConstraintGraph &g,
    const CandidateAssignment &a,LayoutTarget &target,const ExperimentPolicy &policy) {
  Contributions observer;
  // Shared objective evaluation; the subsequent plan-only regeneration
  // obtains its certificate without adding observed charges or search probes.
  auto cost=scoreExperimentAssignment(g,a,target,policy,&observer);if(failed(cost))return failure();
  auto choices=chooseLayoutConversions(g,a,target,nullptr,policy.conversion);if(failed(choices))return failure();
  if(!sameCost(*cost,observer.sum))return failure();
  return ScoredAssignment{*cost,std::move(choices->edges),std::move(observer.entries)};
}

OracleResult enumerateOracle(const LayoutConstraintGraph &g,const FrozenProofs &proofs,
                             LayoutTarget &target,OracleLimits limits,const ExperimentPolicy &policy) {
  OracleResult r;r.limits=limits;
  if(domainDigest(g)!=proofs.domainDigest || coverageDigest(proofs)!=proofs.coverageDigest) {
    r.status="invalid-domain";r.reason="frozen domain identity or proof coverage mismatch";r.fatal=true;return r;
  }
  for(const auto &v:g.getVariables())for(const auto &candidate:v.candidates)if(!proofs.candidates.empty()) {
    auto frozen=llvm::find_if(proofs.candidates,[&](const auto &e){return e.variable==v.id && e.candidate==candidate.value;});
    auto loc=v.anchor?v.anchor->getLoc():UnknownLoc::get(v.shapedType.getContext());ScopedDiagnosticHandler quiet(loc.getContext(),[](Diagnostic &){return success();});
    bool valid=succeeded(target.verifyCandidate(v,candidate.value,loc)),capacity=succeeded(verifyLayoutStorageCapacity(g,v,candidate.value));
    if(frozen==proofs.candidates.end() || frozen->valid!=valid || frozen->capacity!=capacity){r.status="invalid-domain";r.reason="frozen candidate validity/capacity drift";r.fatal=true;return r;}
  }
  if(!policy.scope.fullGraph && !g.isFinalized()) {
    r.status="invalid-domain";r.reason="scoped oracle requires finalized frozen graph";r.fatal=true;return r;
  }
  if(limits.nodes>65536 || limits.evaluations>4096 || limits.milliseconds>120000) {
    r.status="invalid-limits";r.reason="oracle limits may only be reduced";r.fatal=true;return r;
  }
  auto unsupportedPolicy=experimentPolicyUnsupportedReason(g,policy);
  if(!unsupportedPolicy.empty()) {
    r.status="unsupported";r.reason=unsupportedPolicy;return r;
  }
  auto start=std::chrono::steady_clock::now();CandidateAssignment current;
  bool stopped=false, unknownExecution=false;
  auto stop=[&](StringRef reason){stopped=true;r.reason=reason.str();};
  auto timedOut=[&] {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count()>=int64_t(limits.milliseconds);
  };
  // Stable graph ID / prepared candidate order, with only hard feasibility
  // pruning. No production component decomposition, MRV, cost bounds or beam.
  std::function<void(unsigned)> visit=[&](unsigned depth) {
    if(stopped) return;
    if(timedOut()) {stop("timeout");return;}
    if(r.visitedNodes==limits.nodes) {stop("node-limit");return;}
    ++r.visitedNodes;
    bool unknown=false;
    for(const auto &c:g.getConstraints()) if(c.strength==ConstraintStrength::Hard) {
      if(c.instruction && !findInstructionSupport(g,c,current.values)) {++r.rejected;return;}
      if(c.reduction && !findReductionSupport(g,c,current.values)) {++r.rejected;return;}
    }
    for(const auto &p:proofs.entries) {
      if(p.lhs!=UINT32_MAX && (current.values.lookup(p.lhs)!=p.a ||
          (p.b && current.values.lookup(p.rhs)!=p.b))) continue;
      if(p.proof.status==ProofStatus::Disproven) {++r.rejected;return;}
      unknown|=p.proof.status==ProofStatus::Unknown;
    }
    if(depth==g.getVariables().size()) {
      if(unknown) {++r.unknownAssignments;return;}
      auto execution=target.verifyExecutionSupport(g,current);
      if(execution.status==ProofStatus::Disproven) {++r.rejected;return;}
      if(execution.status==ProofStatus::Unknown) {
        ++r.unknownAssignments;unknownExecution=true;
        r.reason="unknown execution-domain proof: "+execution.reason;
        return;
      }
      if(r.evaluatedAssignments==limits.evaluations) {stop("evaluation-limit");return;}
      ++r.evaluatedAssignments;
      auto scored=scoreAssignment(g,current,target,policy);
      if(failed(scored)) {r.fatal=true;stop("cost-failure-or-contribution-mismatch");return;}
      bool better=!r.best || scored->cost.cost<r.best->cost.cost;
      bool strictCostImprovement=better;
      if(r.best && !policy.scope.fullGraph && !(r.best->cost.cost<scored->cost.cost) && !(scored->cost.cost<r.best->cost.cost)) {
        SmallVector<LayoutVarID> ids(policy.scope.producers);
        llvm::sort(ids,[&](auto x,auto y){return g.getVariable(x).stableName<g.getVariable(y).stableName;});
        for(auto id:ids) {
          auto ordinal=[&](Attribute attr) {for(const auto &c:g.getVariable(id).candidates)if(c.value==attr)return c.stableOrdinal;return UINT64_MAX;};
          auto x=ordinal(current.values.lookup(id)),y=ordinal(r.best->assignments.lookup(id));
          if(x!=y){better=x<y;break;}
        }
      }
      if(better) {
        LayoutSolution best;best.policy=policy;best.assignments=current.values;best.cost=scored->cost;best.conversions=scored->conversions;
        for(const auto &c:g.getConstraints()) {
          if(c.instruction) best.instructionBindings[c.id]=findInstructionSupport(g,c,current.values)->binding;
          if(c.reduction) best.reductionBindings[c.id]=findReductionSupport(g,c,current.values)->binding;
        }
        r.best=std::move(best);r.contributions=std::move(scored->contributions);if(strictCostImprovement)r.optimalGeometries.clear();
      }
      if(!(r.best->cost.cost<scored->cost.cost)) {
        auto geometries=assignmentGeometries(g,current);r.optimalGeometries.insert(geometries.begin(),geometries.end());
      }
      // A single noninterruptible proof/cost call may cross the safety timer.
      // Its incumbent is retained, but cannot establish completed optimality.
      if(timedOut()) stop("timeout");
      return;
    }
    const auto &v=g.getVariable(depth);
    for(const auto &candidate:v.candidates) {
      auto frozen=llvm::find_if(proofs.candidates,[&](const auto &e){return e.variable==v.id && e.candidate==candidate.value;});
      if(!proofs.candidates.empty()) {
        if(frozen==proofs.candidates.end()){r.fatal=true;stop("missing-candidate-proof");return;}
        if(!frozen->valid || !frozen->capacity){++r.rejected;continue;}
      }
      auto loc=v.anchor?v.anchor->getLoc():UnknownLoc::get(v.shapedType.getContext());
      bool valid;
      if(!proofs.candidates.empty())valid=true;
      else {ScopedDiagnosticHandler quiet(loc.getContext(),[](Diagnostic &){return success();});
       valid=succeeded(target.verifyCandidate(v,candidate.value,loc)) && succeeded(verifyLayoutStorageCapacity(g,v,candidate.value));}
      if(!valid){++r.rejected;continue;}
      current.values[v.id]=candidate.value;visit(depth+1);if(stopped) break;
    }
    current.values.erase(v.id);
  };
  visit(0);r.exhausted=!stopped;
  r.status=r.fatal ? "failed" : stopped ? "oracle-incomplete" : unknownExecution ? "unsupported" : r.unknownAssignments ? "proof-incomplete" : r.best ? "complete" : "infeasible";
  return r;
}
llvm::json::Array assignmentJSON(const LayoutConstraintGraph &g,const DenseMap<LayoutVarID,Attribute> &a) {
  llvm::json::Array result;
  for(const auto &v:g.getVariables()) if(auto encoding=a.lookup(v.id))
    result.push_back(llvm::json::Object{{"variable",v.id},{"name",v.stableName},{"candidate",candidateJSON(encoding)}});
  return result;
}
llvm::json::Array conversionsJSON(const LayoutConstraintGraph &g,ArrayRef<LayoutConversionEdge> edges) {
  llvm::json::Array result;
  for(const auto &e:edges) {
    llvm::json::Array remat;for(auto *op:e.rematerialization) remat.push_back(op->getName().getStringRef());
    result.push_back(llvm::json::Object{{"constraint",e.constraint},{"use",g.getConstraint(e.constraint).stableUseKey},
      {"source",candidateJSON(e.sourceEncoding)},{"target",candidateJSON(e.targetEncoding)},
      {"resolution",int(e.resolution)},{"exchange",int(e.exchange)},{"bytes",e.bytes},
      {"synchronization_cost",e.synchronizationCost},{"loop_depth",e.loopDepth},{"shared_leader",e.sharedLeader},
      {"cost_provenance",e.costProvenance},{"rematerialization",std::move(remat)}});
  }
  return result;
}
llvm::json::Array existingConversionsJSON(const LayoutConstraintGraph &g,const CandidateAssignment &a) {
  llvm::json::Array result;
  for(const auto &c:g.getConstraints()) if(c.kind==ConstraintKind::Convertible && c.existingConversion) {
    auto source=a.values.lookup(c.vars[0]),target=a.values.lookup(c.vars[1]);
    result.push_back(llvm::json::Object{{"constraint",c.id},{"use",c.stableUseKey},
      {"source",candidateJSON(source)},{"target",candidateJSON(target)},
      {"identity",layoutEncodingsEqual(source,target)},
      {"cost_owner",c.id},{"cost_kind","conversion"},
      {"note","explicit IR boundary; charged once by chooseLayoutConversions unless identity"}});
  }
  return result;
}
llvm::json::Object OracleResult::toJSON() const {
  llvm::json::Array geometries;for(const auto &g:optimalGeometries) geometries.push_back(g);
  llvm::json::Object result{{"status",status},{"reason",reason},{"fatal",fatal},{"exhausted",exhausted},
    {"visited_nodes",visitedNodes},{"evaluated_assignments",evaluatedAssignments},{"hard_rejected",rejected},
    {"unknown_assignments",unknownAssignments},{"limits",llvm::json::Object{{"nodes",limits.nodes},
      {"evaluations",limits.evaluations},{"milliseconds",limits.milliseconds}}},
    {"tie_definition","all nine CostVector fields lexicographically equal; saturation separately recorded"},
    {"geometry_set_kind",status=="complete" ? "all-minimum-cost-geometries-independent-of-stable-producer-winner" : "incumbent-cost-only"},
    {"optimal_geometries",std::move(geometries)},{"cost_contributions",llvm::json::Array(contributions)}};
  if(best) result["best_cost"]=layoutCostJSON(best->cost);else result["best_cost"]=nullptr;
  return result;
}
} // namespace mlir::frisk::selection
