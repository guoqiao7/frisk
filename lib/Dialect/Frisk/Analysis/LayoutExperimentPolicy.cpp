#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
namespace mlir::frisk {
SmallVector<LayoutVarID> experimentInstructionDependencies(
    const LayoutConstraintGraph &graph,const LayoutConstraint &c,const LayoutTarget &target) {
  // Versioned SM90 MMA work is gM*gN*mRepeat*nRepeat*kRepeat. Shapes and
  // execution threads are fixed; the selected geometry alone determines it.
  // Storage descriptors affect legality, not this work charge.
  if((target.costModelID()=="sm90-layout-lex9-legacy-v1" || target.costModelID().starts_with("sm90-layout-calibrated-v2")) && c.instruction &&
     c.instruction->source->getName().getStringRef()=="frisk.mma" && c.vars.size()==5)
    return {c.vars[4]};
  return SmallVector<LayoutVarID>(c.vars);
}
namespace {
// Local cost ownership is operation-based. Its input/storage/use choices may
// remain local auxiliaries; only declared output/geometry decisions are pinned
// for stage two. This does not grant visibility into future consumer costs.
bool producerOperation(const LayoutConstraintGraph &g,const SelectionScope &scope,
                       Operation *op) {
  if(!op)return false;
  for(auto id:scope.producerInstructions)
    if(id<g.getConstraints().size()) {
      const auto &c=g.getConstraint(id);
      if(c.instruction && c.instruction->source==op)return true;
    }
  for(auto id:scope.producers)if(id<g.getVariables().size()) {
    const auto &v=g.getVariable(id);
    if(v.value && v.value.getDefiningOp()==op)return true;
    if(v.operationExecution && v.anchor==op)return true;
  }
  return false;
}
Operation *contributionOperation(const LayoutConstraintGraph &g,
                                 const LayoutConstraint &c) {
  if(c.instruction)return c.instruction->source;
  if(c.reduction)return c.reduction->source;
  if(c.use)return c.use->getOwner();
  return c.provenance<g.getProvenances().size()?g.getProvenances()[c.provenance].source:nullptr;
}
void canonicalizeDependencies(SmallVector<LayoutVarID> &deps) {
  llvm::sort(deps);deps.erase(std::unique(deps.begin(),deps.end()),deps.end());
}
Value storageRoot(const LayoutVar &v) {
  if(v.storageAlias)return v.storageAlias->root;
  if(v.value) {
    auto alias=analyzeStorageAlias(v.value);
    if(succeeded(alias))return alias->root;
  }
  return v.value;
}
struct ContributionRule {
  bool known=false,included=false,variableOwner=false;
  SmallVector<LayoutVarID> dependencies;
  StringRef reason="excluded future consumer, storage, or boundary contribution";
};
ContributionRule contributionRule(const LayoutConstraintGraph &g,const LayoutTarget &target,
    const SelectionScope &scope,StringRef kind,uint64_t owner) {
  ContributionRule rule;
  rule.variableOwner=kind=="registers" || kind=="shared-root" || kind=="arithmetic";
  rule.known=llvm::is_contained(ArrayRef<StringRef>{"registers","shared-root","arithmetic","access","copy-setup","mma","reduction","conversion","rematerialization","target-aggregate"},kind);
  if((rule.variableOwner && owner>=g.getVariables().size()) ||
     (!rule.variableOwner && kind!="target-aggregate" && owner>=g.getConstraints().size()))rule.known=false;
  if(kind=="shared-root" || kind=="conversion" || kind=="rematerialization" || kind=="access" || kind=="target-aggregate") {
    for(const auto &v:g.getVariables())rule.dependencies.push_back(v.id);
  } else if(rule.variableOwner && owner<g.getVariables().size())rule.dependencies.push_back(owner);
  else if(kind!="target-aggregate" && owner<g.getConstraints().size()) {
    const auto &c=g.getConstraint(owner);
    rule.dependencies=(kind=="mma" || kind=="copy-setup" || kind=="reduction")
      ? experimentInstructionDependencies(g,c,target):SmallVector<LayoutVarID>(c.vars);
  }
  if(scope.fullGraph){rule.included=true;rule.reason="full graph objective";return rule;}
  if(kind=="target-aggregate")rule.known=false;
  if(!scope.informedLocal) {
    // Preserve the original restricted baseline byte-for-byte in intent.
    if(kind=="registers" || kind=="arithmetic")rule.included=llvm::is_contained(scope.producers,owner);
    if(kind=="mma" || kind=="copy-setup" || kind=="reduction")
      rule.included=llvm::is_contained(scope.producerInstructions,owner) &&
        llvm::all_of(rule.dependencies,[&](auto id){return llvm::is_contained(scope.producers,id);});
    if(rule.included)rule.reason="all registered contribution dependencies producer-bound";
  } else if(rule.known) {
    if(kind=="registers" || kind=="arithmetic") {
      const auto &v=g.getVariable(owner);
      rule.included=llvm::is_contained(scope.producers,owner) ||
          (!v.use && v.value && producerOperation(g,scope,v.value.getDefiningOp()));
    } else if(kind=="access" || kind=="copy-setup" || kind=="mma" || kind=="reduction") {
      const auto &c=g.getConstraint(owner);auto *op=contributionOperation(g,c);
      rule.dependencies=SmallVector<LayoutVarID>(c.vars);
      // Copy's selected path is selected by its complete tuple, even for an
      // individual access row. Track those local auxiliary dependencies too.
      if(kind=="access")for(const auto &instruction:g.getConstraints())
        if(instruction.instruction && instruction.instruction->source==op)
          llvm::append_range(rule.dependencies,instruction.vars);
      canonicalizeDependencies(rule.dependencies);
      rule.included=producerOperation(g,scope,op);
    } else if(kind=="conversion") {
      const auto &c=g.getConstraint(owner);
      rule.known=c.kind==ConstraintKind::Convertible && c.use && c.vars.size()==2;
      rule.dependencies=SmallVector<LayoutVarID>(c.vars);
      rule.included=rule.known && producerOperation(g,scope,c.use->getOwner());
    } else if(kind=="shared-root") {
      const auto &v=g.getVariable(owner);auto root=storageRoot(v);
      rule.dependencies.clear();
      for(const auto &x:g.getVariables())if(x.kind==LayoutKind::Storage && !x.use &&
          ((root && storageRoot(x)==root) || (!root && x.id==owner)))
        rule.dependencies.push_back(x.id);
      bool local=false,future=false;
      for(const auto &c:g.getConstraints())if(c.kind==ConstraintKind::StorageAccess &&
          llvm::any_of(c.vars,[&](auto id){const auto &x=g.getVariable(id);return x.kind==LayoutKind::Storage &&
            ((root && storageRoot(x)==root) || (!root && id==owner));})) {
        if(producerOperation(g,scope,contributionOperation(g,c)))local=true;else future=true;
      }
      rule.included=local && !future;
      if(local && future) {
        // A global alias maximum depending on a future consumer cannot be
        // attributed to this producer. Fail closed rather than omit its own
        // storage pressure while calling the objective fully informed.
        rule.known=false;rule.reason="nonlocal shared-root maximum spans producer and future consumer";
      }
    }
    if(rule.included)rule.reason="own producer operation cost, including local input/storage auxiliaries";
  }
  if(!rule.known && rule.reason!="nonlocal shared-root maximum spans producer and future consumer")
    rule.reason="unsupported contribution kind or owner";
  return rule;
}
class ScopedObserver final : public LayoutCostObserver {
public:
  const LayoutConstraintGraph &graph;
  const SelectionScope &scope;
  const LayoutTarget &target;
  LayoutCostObserver *downstream;
  llvm::json::Array *contributions;
  CostEstimate sum,observed;
  bool unsupported=false;
  ScopedObserver(const LayoutConstraintGraph &g,const SelectionScope &s,const LayoutTarget &t,
      LayoutCostObserver *o,llvm::json::Array *rows):graph(g),scope(s),target(t),downstream(o),contributions(rows){}
  void recordCost(StringRef kind,uint64_t owner,const CostEstimate &cost) override {
    observed=addLayoutCosts(observed,cost);
    auto rule=contributionRule(graph,target,scope,kind,owner);
    unsupported |= !scope.fullGraph && !rule.known;
    if(rule.included){sum=addLayoutCosts(sum,cost);if(downstream)downstream->recordCost(kind,owner,cost);}
    if(contributions) {
      llvm::json::Array dependencies;
      for(auto id:rule.dependencies)dependencies.push_back(llvm::json::Object{{"id",id},{"stable_name",graph.getVariable(id).stableName}});
      std::string stableOwner=std::to_string(owner);
      if(rule.variableOwner && owner<graph.getVariables().size())stableOwner=graph.getVariable(owner).stableName;
      else if(owner<graph.getConstraints().size()) {
        const auto &c=graph.getConstraint(owner);const auto &p=graph.getProvenances()[c.provenance];
        stableOwner=p.rule+":"+p.reason+":"+c.stableUseKey;
      }
      contributions->push_back(llvm::json::Object{{"kind",kind.str()},{"owner",owner},
        {"owner_kind",rule.variableOwner?"variable":"constraint"},{"stable_owner",stableOwner},
        {"dependencies",std::move(dependencies)},{"included",rule.included},{"known",rule.known},
        {"local_objective",scope.informedLocal?"own-operations-input-repairs-and-resources-v2":"restricted-producer-v1"},
        {"reason",rule.reason},{"cost",layoutCostJSON(cost)}});
    }
  }
};}
std::string experimentPolicyUnsupportedReason(const LayoutConstraintGraph &graph,
    const ExperimentPolicy &policy) {
  if(policy.scope.fullGraph || !policy.scope.informedLocal ||
     policy.conversion==ConversionPolicy::Independent)return {};
  for(auto id:policy.scope.producers)if(id>=graph.getVariables().size())
    return "informed local support preflight: invalid producer index";
  for(auto id:policy.scope.producerInstructions)if(id>=graph.getConstraints().size())
    return "informed local support preflight: invalid producer instruction index";
  for(const auto &c:graph.getConstraints()) {
    if(c.kind==ConstraintKind::Convertible && (c.vars.size()!=2 ||
       llvm::any_of(c.vars,[&](auto id){return id>=graph.getVariables().size();})))
      return "informed local support preflight: invalid incoming constraint indices";
    if(c.kind!=ConstraintKind::Convertible || !c.use ||
       !producerOperation(graph,policy.scope,c.use->getOwner()))continue;
    for(const auto &source:graph.getVariable(c.vars[0]).candidates)
      for(const auto &target:graph.getVariable(c.vars[1]).candidates)
        if(!layoutEncodingsEqual(source.value,target.value))
          return "informed local O unsupported: frozen domain has potential nonidentity producer incoming repair at "+
              c.stableUseKey+"; shared/rematerialized groups cannot be attributed locally";
  }
  return {};
}
FailureOr<CostEstimate> scoreExperimentAssignment(const LayoutConstraintGraph &graph,
    const CandidateAssignment &assignment,LayoutTarget &target,const ExperimentPolicy &policy,LayoutCostObserver *observer,llvm::json::Array *contributions) {
  if(!policy.scope.fullGraph) {
    if(!graph.isFinalized())return failure();
    for(auto id:policy.scope.producers)
      if(id>=graph.getVariables().size() ||
         (graph.getVariable(id).kind!=LayoutKind::Distributed && graph.getVariable(id).kind!=LayoutKind::Instruction) ||
         graph.getVariable(id).use) return failure();
    for(auto id:policy.scope.producerInstructions) {
      if(id>=graph.getConstraints().size())return failure();
      const auto &c=graph.getConstraint(id);
      auto deps=experimentInstructionDependencies(graph,c,target);
      if(deps.size()!=c.vars.size() &&
         (!llvm::is_contained(policy.scope.producers,c.vars[3]) ||
          !llvm::is_contained(policy.scope.producers,c.vars[4])))return failure();
    }
  }
  if(!experimentPolicyUnsupportedReason(graph,policy).empty())return failure();
  ScopedObserver scoped(graph,policy.scope,target,observer,contributions);
  auto *sink=policy.scope.fullGraph && !contributions?observer:&scoped;
  auto cost=target.evaluate(graph,assignment,sink);
  if(failed(cost))return failure();
  // Full legality and the selected policy's complete plan are checked even
  // when their boundary charges are excluded from the producer objective.
  auto choices=chooseLayoutConversions(graph,assignment,target,sink,policy.conversion);
  if(failed(choices) || scoped.unsupported)return failure();
  if(!policy.scope.fullGraph) {
    auto expected=addLayoutCosts(*cost,choices->cost);
    if(expected.cost<scoped.observed.cost || scoped.observed.cost<expected.cost || expected.saturated!=scoped.observed.saturated)return failure();
  }
  return policy.scope.fullGraph?addLayoutCosts(*cost,choices->cost):scoped.sum;
}
}
