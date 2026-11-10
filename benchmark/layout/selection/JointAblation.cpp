#include "JointAblation.h"
#include "ArchiveSourceGrammar.h"
#include "mlir/IR/Builders.h"
#include <chrono>
#include "SelectionOracle.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutDemand.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"
#include <optional>
#include <array>
#include <tuple>
#include <limits>
namespace mlir::frisk::selection {
StringRef archiveRetentionPolicyName(ArchiveRetentionPolicy policy) {
  if(policy==ArchiveRetentionPolicy::CalibratedSourceGrammarV1)return "calibrated-source-grammar-v1";
  if(policy==ArchiveRetentionPolicy::PaidStaticFeaturesV1)return "paid-static-feature-frontier-v1";
  return policy==ArchiveRetentionPolicy::PaidFeatureFrontierV1?
      "paid-feature-frontier-v1":"legacy-hamming-v2";
}
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
namespace {
// Optional diagnostic lineage. Mark source results and consumer operands before
// materialization, then resolve operand markers through the rewritten IR. No
// raw graph Value/OpOperand pointer is dereferenced after materialization.
void addResultLineage(Value value,StringRef name,int64_t identity) {
 auto result=dyn_cast<OpResult>(value);if(!result)return;
 auto *op=result.getOwner();Builder b(op->getContext());SmallVector<Attribute> arrays;
 if(auto old=op->getAttrOfType<ArrayAttr>(name))llvm::append_range(arrays,old);
 else arrays.assign(op->getNumResults(),b.getDenseI64ArrayAttr({}));
 auto ids=cast<DenseI64ArrayAttr>(arrays[result.getResultNumber()]).asArrayRef();
 SmallVector<int64_t> merged(ids);merged.push_back(identity);llvm::sort(merged);merged.erase(std::unique(merged.begin(),merged.end()),merged.end());
 arrays[result.getResultNumber()]=b.getDenseI64ArrayAttr(merged);op->setAttr(name,b.getArrayAttr(arrays));
}
void addOperandLineage(OpOperand *operand,StringRef name,int64_t identity) {
 if(!operand)return;auto *op=operand->getOwner();Builder b(op->getContext());SmallVector<Attribute> arrays;
 if(auto old=op->getAttrOfType<ArrayAttr>(name))llvm::append_range(arrays,old);
 else arrays.assign(op->getNumOperands(),b.getDenseI64ArrayAttr({}));
 auto ids=cast<DenseI64ArrayAttr>(arrays[operand->getOperandNumber()]).asArrayRef();SmallVector<int64_t> merged(ids);merged.push_back(identity);
 llvm::sort(merged);merged.erase(std::unique(merged.begin(),merged.end()),merged.end());arrays[operand->getOperandNumber()]=b.getDenseI64ArrayAttr(merged);op->setAttr(name,b.getArrayAttr(arrays));
}
}
void markDemandLineage(const LayoutConstraintGraph &graph) {
 for(const auto &v:graph.getVariables())if(v.kind==LayoutKind::Distributed) {
  if(v.use)addOperandLineage(v.use,"frisk.execution_audit_operand_variables",v.id);
  else if(v.value)addResultLineage(v.value,"frisk.execution_audit_variables",v.id);
 }
 for(const auto &c:graph.getConstraints())if(c.kind==ConstraintKind::Convertible) {
  // Existing transport is defined by its output. A prospective use repair is
  // instead tracked through the operand that materialization will rewrite.
  if(c.existingConversion && c.vars.size()==2)
   addResultLineage(graph.getVariable(c.vars[1]).value,"frisk.execution_audit_boundaries",c.id);
  else addOperandLineage(c.use,"frisk.execution_audit_operand_boundaries",c.id);
 }
}
LogicalResult resolveDemandLineage(ModuleOp module) {
 bool invalid=false;module.walk([&](Operation *op){
  for(auto pair:{std::make_pair("frisk.execution_audit_operand_variables","frisk.execution_audit_variables"),std::make_pair("frisk.execution_audit_operand_boundaries","frisk.execution_audit_boundaries")}) {
   if(auto arrays=op->getAttrOfType<ArrayAttr>(pair.first)) {
    if(arrays.size()!=op->getNumOperands()){invalid=true;continue;}
    for(unsigned i=0;i<arrays.size();++i)for(int64_t id:cast<DenseI64ArrayAttr>(arrays[i]).asArrayRef())addResultLineage(op->getOperand(i),pair.second,id);
    op->removeAttr(pair.first);
   }
  }
 });
 return failure(invalid);
}

FailureOr<LayoutSolution> solveAblation(LayoutConstraintGraph &graph,LayoutTarget &target,
    SelectionMode mode,SolverOptions options,llvm::json::Object &record) {
  using Clock=std::chrono::steady_clock;
  auto solveStart=Clock::now();
  auto elapsed=[](auto start){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();};
  int64_t stage1Ns=0,stage2Ns=0,incumbentScoreNs=0;
  auto timing=[&](){record["solve_timing_ns"]=llvm::json::Object{{"stage1",stage1Ns},{"stage2",stage2Ns},{"incumbent_full_score",incumbentScoreNs},{"total",elapsed(solveStart)}};};
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
  // Stage one used a different objective. Its first full-graph score can
  // change the final incumbent choice and therefore is a real search charge.
  if(!budget.consume(false)) {
    why={LayoutSolveFailureKind::SearchBudget,UINT32_MAX,UINT32_MAX,"producer-first full incumbent score budget exhausted"};
    solveFailure(record,why);record["total"]=counters(budget);timing();return failure();
  }
  ++budget.completeScores;record["incumbent_full_score_budget_charged"]=true;
  auto incumbentScoreStart=Clock::now();
  auto full=scoreAssignment(common,CandidateAssignment{first->assignments},target,options.policy);
  incumbentScoreNs=elapsed(incumbentScoreStart);
  if(failed(full)) {
    why={LayoutSolveFailureKind::CostFailure,UINT32_MAX,UINT32_MAX,"producer-first full incumbent cost evaluation failed"};
    solveFailure(record,why);record["total"]=counters(budget);timing();return failure();
  }
  record["incumbent_full_score"]=layoutCostJSON(full->cost);
  first->cost=full->cost;first->conversions=std::move(full->conversions);first->policy=options.policy;
  if(options.completeCandidateObserver && !full->cost.saturated)
    options.completeCandidateObserver->recordComplete(common,
        CandidateAssignment{first->assignments},full->cost,options.policy,budget);
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
namespace {
bool sameAssignment(const DenseMap<LayoutVarID,Attribute> &a,
                    const DenseMap<LayoutVarID,Attribute> &b) {
  if(a.size()!=b.size())return false;
  for(auto [id,value]:a)if(b.lookup(id)!=value)return false;
  return true;
}
uint64_t assignmentDistance(const DenseMap<LayoutVarID,Attribute> &a,
                            const DenseMap<LayoutVarID,Attribute> &b) {
  uint64_t distance=0;for(auto [id,value]:a)distance+=b.lookup(id)!=value;return distance;
}
// These are already-paid static counts, never a second target invocation or a
// physical register prediction. Sparse zero/unmodeled fields are not proof of
// free native work. Keeping their independent axes avoids inheriting the one
// engineering coefficient vector's ordering into candidate coverage.
constexpr std::array<llvm::StringLiteral,23> retentionAxes = {
  "access_warp_issues", "global_sectors32", "shared_bank_excess",
  "copy_setup_units", "mma_instructions", "reduction_element_ops",
  "alu_warp_issues", "div_warp_issues", "exp_warp_issues", "cast_warp_issues",
  "conversion_register_bytes", "conversion_shuffle_bytes",
  "conversion_shared_bytes", "conversion_conservative_bytes",
  "synchronization_barriers", "shared_read_bank_excess",
  "shared_write_bank_excess", "shared_warp_issues", "shuffle_warp_issues",
  "communication_alu_warp_issues", "logical_register_proxy",
  "logical_shared_byte_proxy", "logical_replication_proxy"};
using RetentionFeatures=std::array<uint64_t,retentionAxes.size()>;
RetentionFeatures retentionFeatures(const CostEstimate &cost) {
  RetentionFeatures f{};
  for(const auto &[name,value]:cost.features)
    for(unsigned i=0;i<20;++i)if(StringRef(name)==retentionAxes[i])f[i]=value;
  f[20]=cost.cost.spillRiskAndRegisters;
  f[21]=cost.cost.sharedBytesAndOccupancy;f[22]=cost.cost.replication;
  return f;
}
// Canonical column spaces use highest-bit pivots, reduced against every pivot.
// Their equality/distance is invariant under basis choice and never enumerates
// hardware holders. The current64-bit structural bound is explicit; larger maps
// retain an Unknown structure and fall back to the paid feature view.
SmallVector<uint64_t> canonicalSpace(ArrayRef<uint64_t> columns) {
  std::array<uint64_t,64> pivots{};
  for(uint64_t value:columns)for(int bit=63;value && bit>=0;--bit)if((value>>bit)&1) {
    if(pivots[bit])value^=pivots[bit];
    else {for(unsigned j=0;j<64;++j)if((pivots[j]>>bit)&1)pivots[j]^=value;pivots[bit]=value;break;}
  }
  // Insertion order can leave low pivot bits in an earlier row. Reduce again.
  for(unsigned bit=0;bit<64;++bit)if(pivots[bit])
    for(unsigned j=bit+1;j<64;++j)if((pivots[j]>>bit)&1)pivots[j]^=pivots[bit];
  SmallVector<uint64_t> result;for(auto value:pivots)if(value)result.push_back(value);return result;
}
uint64_t spanDefect(ArrayRef<uint64_t> source,ArrayRef<uint64_t> other) {
  SmallVector<uint64_t> both(source);llvm::append_range(both,other);
  return canonicalSpace(both).size()-canonicalSpace(source).size();
}
struct OwnershipView {
  LayoutVarID variable=0;Attribute encoding;bool proven=false,active=true;
  unsigned bits=0,nullity=0;
  std::array<SmallVector<uint64_t>,5> columns;
  SmallVector<uint64_t> registers,warp;
};
struct StructureView {
  bool proven=true;std::string reason;
  SmallVector<OwnershipView,0> ownership;
  struct Boundary {LayoutConstraintID id;LayoutVarID source,target;uint64_t registerDefect=0,warpDefect=0,scope=3;};
  SmallVector<Boundary> boundaries;
  SmallVector<uint64_t> signature;
  uint64_t crossThread=0;
  llvm::json::Object json()const {
    llvm::json::Array owners,edges,signatureJSON;for(auto value:signature)signatureJSON.push_back(value);
    auto array=[](ArrayRef<uint64_t> values){llvm::json::Array result;for(auto v:values)result.push_back(v);return result;};
    for(const auto &o:ownership) {
      llvm::json::Array columns;for(const auto &carrier:o.columns)columns.push_back(array(carrier));
      owners.push_back(llvm::json::Object{{"variable",o.variable},{"proven",o.proven},{"active",o.active},
        {"logical_bits",o.bits},{"register_nullity",o.nullity},{"carrier_columns",std::move(columns)},
        {"register_basis",array(o.registers)},{"warp_basis",array(o.warp)}});
    }
    for(const auto &b:boundaries)edges.push_back(llvm::json::Object{{"constraint",b.id},{"source",b.source},{"target",b.target},
      {"register_defect",b.registerDefect},{"warp_defect",b.warpDefect},{"scope",b.scope}});
    return llvm::json::Object{{"contract","gf2-ownership-cut-v1"},{"proven",proven},{"reason",reason},
      {"communication_signature",std::move(signatureJSON)},{"cross_thread_boundaries",crossThread},
      {"ownership",std::move(owners)},{"boundaries",std::move(edges)}};
  }
};
class StructureCache {
  DenseMap<Attribute,OwnershipView> layouts;
  DenseMap<std::pair<Attribute,Attribute>,uint64_t> distances;
 public:
  uint64_t layoutHits=0,layoutMisses=0,distanceHits=0,distanceMisses=0;
  OwnershipView ownership(const LayoutVar &var,Attribute encoding) {
    auto it=layouts.find(encoding);OwnershipView view;
    if(it!=layouts.end()){++layoutHits;view=it->second;}
    else {
      ++layoutMisses;view.encoding=encoding;
      auto distributed=dyn_cast_or_null<DistributedEncodingAttr>(encoding);
      auto type=dyn_cast<RankedTensorType>(var.shapedType);LinearExecutionLayout linear;
      if(distributed && type && analyzeLinearExecutionLayout(distributed,type,linear).status==ProofStatus::Proven && linear.logicalBits<=64) {
        view.proven=true;view.bits=linear.logicalBits;
        for(unsigned carrier=0;carrier<5;++carrier)for(unsigned bit=0;bit<linear.hardwareWidths[carrier];++bit) {
          uint64_t column=0;for(unsigned row=0;row<linear.logicalBits;++row)
            if(linear.rows[row][linear.hardwareOffsets[carrier]+bit])column|=uint64_t(1)<<row;
          view.columns[carrier].push_back(column);
        }
        view.registers=canonicalSpace(view.columns[0]);SmallVector<uint64_t> warp(view.columns[0]);llvm::append_range(warp,view.columns[1]);view.warp=canonicalSpace(warp);
        view.nullity=view.columns[0].size()-view.registers.size();
      }
      layouts.try_emplace(encoding,view);
    }
    view.variable=var.id;
    // Definitions absent from the emitted root closure cannot justify a native
    // slot. The scalar backend has nonvolatile tile loads; an unused load result
    // is discarded just like unused pure arithmetic. Live/read-used carriers stay.
    if(var.value && var.value.use_empty())if(auto *op=var.value.getDefiningOp()) {
      auto dialect=op->getName().getDialectNamespace();view.active=dialect!="arith" && dialect!="math" && op->getName().getStringRef()!="frisk.tile_load";
    }
    return view;
  }
  StructureView analyze(const LayoutConstraintGraph &graph,const DenseMap<LayoutVarID,Attribute> &values) {
    StructureView result;DenseMap<LayoutVarID,unsigned> indices;
    for(const auto &var:graph.getVariables())if(var.kind==LayoutKind::Distributed) {
      auto view=ownership(var,values.lookup(var.id));if(!view.proven){result.proven=false;result.reason="unavailable bit-linear distributed ownership or >64 logical bits";}
      indices[var.id]=result.ownership.size();result.ownership.push_back(std::move(view));
    }
    for(const auto &c:graph.getConstraints())if(c.kind==ConstraintKind::Convertible && c.vars.size()==2) {
      StructureView::Boundary boundary{c.id,c.vars[0],c.vars[1]};
      if(indices.count(c.vars[0]) && indices.count(c.vars[1])) {
        const auto &source=result.ownership[indices[c.vars[0]]],&target=result.ownership[indices[c.vars[1]]];
        if(source.proven && target.proven && source.bits==target.bits) {
          SmallVector<uint64_t> registerNeed(target.columns[0]),warpNeed(target.columns[0]);llvm::append_range(warpNeed,target.columns[1]);
          for(unsigned carrier=1;carrier<5;++carrier)for(unsigned i=0;i<std::max(source.columns[carrier].size(),target.columns[carrier].size());++i) {
            uint64_t difference=(i<source.columns[carrier].size()?source.columns[carrier][i]:0)^(i<target.columns[carrier].size()?target.columns[carrier][i]:0);
            registerNeed.push_back(difference);if(carrier>1)warpNeed.push_back(difference);
          }
          boundary.registerDefect=spanDefect(source.registers,registerNeed);boundary.warpDefect=spanDefect(source.warp,warpNeed);
          boundary.scope=boundary.registerDefect==0?0:boundary.warpDefect==0?1:2;
        }
      }
      if(boundary.scope==3){result.proven=false;result.reason="unavailable symbolic communication endpoint";}
      result.signature.push_back(boundary.scope);result.crossThread+=boundary.scope==1 || boundary.scope==2;result.boundaries.push_back(boundary);
    }
    return result;
  }
  uint64_t distance(const StructureView &a,const StructureView &b) {
    if(!a.proven || !b.proven || a.ownership.size()!=b.ownership.size())return 0;
    uint64_t result=0;
    for(auto [x,y]:llvm::zip_equal(a.ownership,b.ownership))if(x.active && y.active) {
      auto key=std::make_pair(x.encoding,y.encoding);auto found=distances.find(key);
      if(found!=distances.end()){++distanceHits;result+=found->second;continue;}
      ++distanceMisses;uint64_t d=0;
      for(auto spaces:{std::make_pair(ArrayRef<uint64_t>(x.registers),ArrayRef<uint64_t>(y.registers)),std::make_pair(ArrayRef<uint64_t>(x.warp),ArrayRef<uint64_t>(y.warp))}) {
        SmallVector<uint64_t> combined(spaces.first);llvm::append_range(combined,spaces.second);
        d+=2*canonicalSpace(combined).size()-spaces.first.size()-spaces.second.size();
      }
      d+=x.nullity>y.nullity?x.nullity-y.nullity:y.nullity-x.nullity;distances.try_emplace(key,d);result+=d;
    }
    return result;
  }
};
bool sameCut(const StructureView &a,const StructureView &b) {return a.proven && b.proven && a.signature==b.signature;}
uint64_t cutDistance(const StructureView &a,const StructureView &b) {
  if(!a.proven || !b.proven || a.signature.size()!=b.signature.size())return 0;
  uint64_t result=0;for(auto [x,y]:llvm::zip_equal(a.signature,b.signature))result+=x>y?x-y:y-x;return result;
}
struct PaidSample {
  DenseMap<LayoutVarID,Attribute> values;
  CostEstimate cost;
  RetentionFeatures features{};
  StructureView structure;
  SmallVector<uint64_t> key;
  std::string origin;
  uint64_t probes=0, refinement=0;
};
bool sampleLess(const PaidSample &a,const PaidSample &b) {
  if(a.cost.cost<b.cost.cost)return true;if(b.cost.cost<a.cost.cost)return false;
  return std::lexicographical_compare(a.key.begin(),a.key.end(),b.key.begin(),b.key.end());
}
bool featureDominates(const RetentionFeatures &a,const RetentionFeatures &b) {
  bool strict=false;
  for(unsigned i=0;i<a.size();++i){if(a[i]>b[i])return false;strict|=a[i]<b[i];}
  return strict;
}
struct FeatureRange {
  RetentionFeatures low,high;
  FeatureRange(){low.fill(UINT64_MAX);high.fill(0);}
  void include(const RetentionFeatures &f) {
    for(unsigned i=0;i<f.size();++i){low[i]=std::min(low[i],f[i]);high[i]=std::max(high[i],f[i]);}
  }
  uint64_t normalized(uint64_t delta,unsigned i)const {
    uint64_t span=high[i]-low[i];
    // Exact integer quantization, with no floating-point/order/platform drift.
    return span?uint64_t((static_cast<unsigned __int128>(delta)*1048576)/span):0;
  }
  uint64_t distance(const RetentionFeatures &a,const RetentionFeatures &b)const {
    uint64_t result=0;
    for(unsigned i=0;i<a.size();++i)result+=normalized(a[i]>b[i]?a[i]-b[i]:b[i]-a[i],i);
    return result;
  }
};
class PaidArchive final:public LayoutCompleteCandidateObserver {
 public:
  PaidArchive(unsigned capacity,ArchiveRetentionPolicy policy,bool auditPaidObservations)
      :capacity(capacity),policy(policy),reservoirCapacity(featurePolicy()?32:capacity),auditPaidObservations(auditPaidObservations){}
  unsigned capacity;ArchiveRetentionPolicy policy;unsigned reservoirCapacity;
  uint64_t observed=0,duplicates=0,frontierRejected=0,capacityDiscarded=0;
  uint64_t peak=0,dominatedDiscarded=0,nondominatedDiscarded=0;
  uint64_t finalDuplicates=0,finalUnselected=0,certificateRebuilds=0;
  uint64_t featureObservations=0,structureProven=0,structureUnknown=0;
  StructureCache structures;
  std::string origin="joint-search";
  SmallVector<PaidSample,0> samples;
  llvm::json::Array discardExamples,paidObservationAudit;
  bool auditPaidObservations;
  bool featurePolicy()const {return policy!=ArchiveRetentionPolicy::LegacyHammingV2;}
  bool structurePolicy()const {return policy==ArchiveRetentionPolicy::PaidFeatureFrontierV1;}
  void noteDiscard(const PaidSample &sample,StringRef reason) {
    if(discardExamples.size()>=16)return;
    llvm::json::Array key,features;for(auto n:sample.key)key.push_back(n);
    for(auto n:sample.features)features.push_back(n);
    discardExamples.push_back(llvm::json::Object{{"reason",reason},{"assignment_stable_ordinals",std::move(key)},
      {"feature_values",std::move(features)},{"origin",sample.origin},
      {"main_score",sample.cost.cost.instructionPathAndWork},{"probes_at_observation",sample.probes}});
  }
  void retainStaticFeatureReservoir() {
    if(samples.size()<=reservoirCapacity)return;
    FeatureRange range;for(const auto &x:samples)range.include(x.features);SmallVector<unsigned> chosen;
    auto add=[&](unsigned i){if(!llvm::is_contained(chosen,i))chosen.push_back(i);};
    unsigned best=0;for(unsigned i=1;i<samples.size();++i)if(sampleLess(samples[i],samples[best]))best=i;add(best);
    for(unsigned axis=0;axis<retentionAxes.size();++axis) {
      if(range.low[axis]==range.high[axis])continue;
      best=0;for(unsigned i=1;i<samples.size();++i)if(samples[i].features[axis]<samples[best].features[axis] || (samples[i].features[axis]==samples[best].features[axis] && sampleLess(samples[i],samples[best])))best=i;add(best);
    }
    while(chosen.size()<reservoirCapacity) {
      bool found=false;unsigned pick=0;bool pickFrontier=false;uint64_t pickDistance=0,pickHamming=0;
      for(unsigned i=0;i<samples.size();++i)if(!llvm::is_contained(chosen,i)) {
        bool frontier=llvm::none_of(samples,[&](const auto &x){return featureDominates(x.features,samples[i].features);});uint64_t distance=UINT64_MAX,hamming=UINT64_MAX;
        for(unsigned j:chosen){distance=std::min(distance,range.distance(samples[i].features,samples[j].features));hamming=std::min(hamming,assignmentDistance(samples[i].values,samples[j].values));}
        auto score=std::make_tuple(frontier,distance,hamming),prior=std::make_tuple(pickFrontier,pickDistance,pickHamming);
        if(!found || score>prior || (score==prior && sampleLess(samples[i],samples[pick]))) {found=true;pick=i;pickFrontier=frontier;pickDistance=distance;pickHamming=hamming;}
      }
      if(!found)break;add(pick);
    }
    for(unsigned i=0;i<samples.size();++i)if(!llvm::is_contained(chosen,i)) {
      bool dominated=llvm::any_of(samples,[&](const auto &x){return featureDominates(x.features,samples[i].features);});dominated?++dominatedDiscarded:++nondominatedDiscarded;noteDiscard(samples[i],dominated?"bounded-reservoir-dominated":"bounded-reservoir-diversity");
    }
    SmallVector<PaidSample,0> retained;for(unsigned i:chosen)retained.push_back(std::move(samples[i]));capacityDiscarded+=samples.size()-retained.size();samples=std::move(retained);
  }
  unsigned selectStaticFeatureCandidate(ArrayRef<ArchivedLayoutCandidate> retained)const {
    FeatureRange range;for(const auto &x:samples)range.include(x.features);SmallVector<RetentionFeatures> prior;RetentionFeatures bestPrior;bestPrior.fill(UINT64_MAX);
    for(const auto &x:retained){auto f=retentionFeatures(x.solution.cost);range.include(f);prior.push_back(f);for(unsigned i=0;i<f.size();++i)bestPrior[i]=std::min(bestPrior[i],f[i]);}
    unsigned best=0;bool found=false,pickFrontier=false;uint64_t pickGain=0,pickDistance=0,pickHamming=0;
    for(unsigned i=0;i<samples.size();++i) {
      const auto &sample=samples[i];if(llvm::any_of(retained,[&](const auto &x){return sameAssignment(sample.values,x.solution.assignments);}))continue;
      bool frontier=llvm::none_of(samples,[&](const auto &x){return featureDominates(x.features,sample.features);}) && llvm::none_of(prior,[&](const auto &f){return featureDominates(f,sample.features);});
      uint64_t gain=0,distance=prior.empty()?0:UINT64_MAX,hamming=retained.empty()?0:UINT64_MAX;
      for(unsigned axis=0;axis<sample.features.size();++axis)if(!prior.empty() && bestPrior[axis]>sample.features[axis])gain+=range.normalized(bestPrior[axis]-sample.features[axis],axis);
      for(const auto &f:prior)distance=std::min(distance,range.distance(sample.features,f));for(const auto &x:retained)hamming=std::min(hamming,assignmentDistance(sample.values,x.solution.assignments));
      auto score=std::make_tuple(frontier,gain,distance,hamming),old=std::make_tuple(pickFrontier,pickGain,pickDistance,pickHamming);
      if(!found || score>old || (score==old && sampleLess(sample,samples[best]))) {found=true;best=i;pickFrontier=frontier;pickGain=gain;pickDistance=distance;pickHamming=hamming;}
    }
    return found?best:samples.size();
  }
  bool comparableFeatures(const PaidSample &a,const PaidSample &b)const {
    return (!a.structure.proven && !b.structure.proven) || sameCut(a.structure,b.structure);
  }
  bool cutParetoDominates(const PaidSample &a,const PaidSample &b)const {
    return a.structure.proven && b.structure.proven &&
        a.cost.cost.instructionPathAndWork<=b.cost.cost.instructionPathAndWork &&
        a.structure.crossThread<=b.structure.crossThread &&
        (a.cost.cost.instructionPathAndWork<b.cost.cost.instructionPathAndWork || a.structure.crossThread<b.structure.crossThread);
  }
  bool keyLess(const PaidSample &a,const PaidSample &b)const {
    return std::lexicographical_compare(a.key.begin(),a.key.end(),b.key.begin(),b.key.end());
  }
  void retainFeatureReservoir() {
    if(!structurePolicy()){retainStaticFeatureReservoir();return;}
    if(samples.size()<=reservoirCapacity)return;
    FeatureRange range;for(const auto &x:samples)range.include(x.features);
    SmallVector<unsigned> chosen;
    auto add=[&](unsigned i){if(chosen.size()<reservoirCapacity && !llvm::is_contained(chosen,i))chosen.push_back(i);};
    unsigned anchor=0;for(unsigned i=1;i<samples.size();++i)if(sampleLess(samples[i],samples[anchor]))anchor=i;
    add(anchor);
    for(unsigned axis=0;axis<retentionAxes.size();++axis) {
      if(range.low[axis]==range.high[axis])continue;
      unsigned best=0;for(unsigned i=1;i<samples.size();++i)
        if(samples[i].features[axis]<samples[best].features[axis] ||
            (samples[i].features[axis]==samples[best].features[axis] && sampleLess(samples[i],samples[best])))best=i;
      add(best);
    }
    // Model totals may be biased across different communication placements.
    // Protect a representative of each non-anchor cut on the (main score,
    // cross-thread boundary count) front. Equal primary scores use ownership
    // novelty, not the legacy code-size tie that can favor folded-away work.
    SmallVector<unsigned> representatives;
    for(unsigned i=0;i<samples.size();++i)if(samples[i].structure.proven && !sameCut(samples[i].structure,samples[anchor].structure)) {
      auto found=llvm::find_if(representatives,[&](unsigned j){return sameCut(samples[i].structure,samples[j].structure);});
      if(found==representatives.end())representatives.push_back(i);
      else {
        unsigned j=*found;auto x=samples[i].cost.cost.instructionPathAndWork,y=samples[j].cost.cost.instructionPathAndWork;
        auto dx=structures.distance(samples[i].structure,samples[anchor].structure),dy=structures.distance(samples[j].structure,samples[anchor].structure);
        if(x<y || (x==y && (dx>dy || (dx==dy && keyLess(samples[i],samples[j])))))*found=i;
      }
    }
    SmallVector<unsigned> frontier;
    for(unsigned i:representatives)if(llvm::none_of(representatives,[&](unsigned j){return cutParetoDominates(samples[j],samples[i]);}))frontier.push_back(i);
    while(!frontier.empty() && chosen.size()<reservoirCapacity) {
      unsigned pick=0;uint64_t bestCut=0,bestOwnership=0;
      for(unsigned n=0;n<frontier.size();++n) {
        auto i=frontier[n];uint64_t cd=cutDistance(samples[i].structure,samples[anchor].structure),od=structures.distance(samples[i].structure,samples[anchor].structure);
        if(!n || cd>bestCut || (cd==bestCut && (od>bestOwnership || (od==bestOwnership && keyLess(samples[i],samples[frontier[pick]]))))) {pick=n;bestCut=cd;bestOwnership=od;}
      }
      add(frontier[pick]);frontier.erase(frontier.begin()+pick);
    }
    while(chosen.size()<reservoirCapacity) {
      bool found=false;unsigned pick=0;bool pickNewCut=false,pickFrontier=false;uint64_t pickCut=0,pickOwner=0,pickDistance=0,pickHamming=0;
      for(unsigned i=0;i<samples.size();++i)if(!llvm::is_contained(chosen,i)) {
        bool newCut=samples[i].structure.proven && llvm::none_of(chosen,[&](unsigned j){return sameCut(samples[i].structure,samples[j].structure);});
        bool frontier=llvm::none_of(samples,[&](const auto &x){return comparableFeatures(x,samples[i]) && featureDominates(x.features,samples[i].features);});
        uint64_t cd=UINT64_MAX,od=UINT64_MAX,distance=UINT64_MAX,hamming=UINT64_MAX;
        for(unsigned j:chosen){cd=std::min(cd,cutDistance(samples[i].structure,samples[j].structure));od=std::min(od,structures.distance(samples[i].structure,samples[j].structure));distance=std::min(distance,range.distance(samples[i].features,samples[j].features));hamming=std::min(hamming,assignmentDistance(samples[i].values,samples[j].values));}
        auto score=std::make_tuple(newCut,frontier,cd,od,distance,hamming),bestScore=std::make_tuple(pickNewCut,pickFrontier,pickCut,pickOwner,pickDistance,pickHamming);
        if(!found || score>bestScore || (score==bestScore && sampleLess(samples[i],samples[pick]))) {found=true;pick=i;pickNewCut=newCut;pickFrontier=frontier;pickCut=cd;pickOwner=od;pickDistance=distance;pickHamming=hamming;}
      }
      if(!found)break;add(pick);
    }
    for(unsigned i=0;i<samples.size();++i)if(!llvm::is_contained(chosen,i)) {
      bool dominated=llvm::any_of(samples,[&](const auto &x){return comparableFeatures(x,samples[i]) && featureDominates(x.features,samples[i].features);});
      dominated?++dominatedDiscarded:++nondominatedDiscarded;
      noteDiscard(samples[i],dominated?"bounded-reservoir-dominated":"bounded-reservoir-diversity");
    }
    SmallVector<PaidSample,0> retained;for(unsigned i:chosen)retained.push_back(std::move(samples[i]));
    capacityDiscarded+=samples.size()-retained.size();samples=std::move(retained);
  }
  unsigned selectFeatureCandidate(const LayoutConstraintGraph &graph,ArrayRef<ArchivedLayoutCandidate> retained) {
    if(!structurePolicy())return selectStaticFeatureCandidate(retained);
    FeatureRange range;for(const auto &x:samples)range.include(x.features);
    SmallVector<RetentionFeatures> prior;SmallVector<StructureView,0> priorStructures,pinStructures;RetentionFeatures bestPrior;bestPrior.fill(UINT64_MAX);
    for(const auto &x:retained) {
      auto f=retentionFeatures(x.solution.cost);range.include(f);prior.push_back(f);for(unsigned i=0;i<f.size();++i)bestPrior[i]=std::min(bestPrior[i],f[i]);
      auto structure=structures.analyze(graph,x.solution.assignments);priorStructures.push_back(structure);if(x.localIncumbent || x.jointIncumbent)pinStructures.push_back(structure);
    }
    auto duplicate=[&](const PaidSample &sample){return llvm::any_of(retained,[&](const auto &x){return sameAssignment(sample.values,x.solution.assignments);});};
    auto newCut=[&](const PaidSample &sample){return sample.structure.proven && llvm::none_of(priorStructures,[&](const auto &x){return sameCut(sample.structure,x);});};
    bool hasNewCut=llvm::any_of(samples,[&](const auto &sample){return !duplicate(sample) && newCut(sample);});
    unsigned best=0;bool found=false,pickFrontier=false;uint64_t pickCut=0,pickOwner=0,pickGain=0,pickDistance=0,pickHamming=0;
    for(unsigned i=0;i<samples.size();++i) {
      const auto &sample=samples[i];if(duplicate(sample) || (hasNewCut && !newCut(sample)))continue;
      bool frontier;
      if(hasNewCut)frontier=llvm::none_of(samples,[&](const auto &x){return !duplicate(x) && newCut(x) && cutParetoDominates(x,sample);});
      else frontier=llvm::none_of(samples,[&](const auto &x){return comparableFeatures(x,sample) && featureDominates(x.features,sample.features);});
      uint64_t cd=priorStructures.empty()?0:UINT64_MAX,od=pinStructures.empty()?0:UINT64_MAX;
      for(const auto &x:priorStructures)cd=std::min(cd,cutDistance(sample.structure,x));
      // Compare ownership to the fixed witnesses. Changing one cut must not
      // force a different representative of an unrelated cut simply because an
      // optional candidate was appended first.
      for(const auto &x:pinStructures)od=std::min(od,structures.distance(sample.structure,x));
      uint64_t gain=0,distance=prior.empty()?0:UINT64_MAX,hamming=retained.empty()?0:UINT64_MAX;
      for(unsigned axis=0;axis<sample.features.size();++axis)
        if(!prior.empty() && bestPrior[axis]>sample.features[axis])gain+=range.normalized(bestPrior[axis]-sample.features[axis],axis);
      for(const auto &f:prior)distance=std::min(distance,range.distance(sample.features,f));
      for(const auto &x:retained)hamming=std::min(hamming,assignmentDistance(sample.values,x.solution.assignments));
      auto score=std::make_tuple(frontier,cd,od,gain,distance,hamming),bestScore=std::make_tuple(pickFrontier,pickCut,pickOwner,pickGain,pickDistance,pickHamming);
      bool lowerPrimary=sample.cost.cost.instructionPathAndWork<samples[best].cost.cost.instructionPathAndWork;
      bool equalPrimary=sample.cost.cost.instructionPathAndWork==samples[best].cost.cost.instructionPathAndWork;
      // Within a proven structural front, primary score precedes feature/Hamming
      // ties; ownership/cut novelty is intentionally outside the biased totals.
      auto structuralScore=std::make_tuple(frontier,cd,od),bestStructural=std::make_tuple(pickFrontier,pickCut,pickOwner);
      bool better=hasNewCut?(structuralScore>bestStructural || (structuralScore==bestStructural && (lowerPrimary || (equalPrimary && keyLess(sample,samples[best]))))):
          (score>bestScore || (score==bestScore && sampleLess(sample,samples[best])));
      if(!found || better) {found=true;best=i;pickFrontier=frontier;pickCut=cd;pickOwner=od;pickGain=gain;pickDistance=distance;pickHamming=hamming;}
    }
    return found?best:samples.size();
  }
  llvm::json::Object receipt() {
    llvm::json::Array axes;for(auto name:retentionAxes)axes.push_back(name.str());
    return llvm::json::Object{{"contract",featurePolicy()?(structurePolicy()?"paid-gf2-structure-reservoir-K4-v1":"paid-static-feature-reservoir-K4-v1"):"paid-fullcost-hamming-v2"},
      {"policy",archiveRetentionPolicyName(policy)},{"reservoir_capacity",reservoirCapacity},{"reservoir_peak",std::min<uint64_t>(peak,reservoirCapacity)},{"observer_working_set_peak",peak},{"pending_incoming_capacity",1},
      {"structure_contract","gf2-ownership-cut-v1"},{"structure_proven_observations",structureProven},{"structure_unknown_observations",structureUnknown},
      {"structure_layout_cache_hits",structures.layoutHits},{"structure_layout_cache_misses",structures.layoutMisses},{"structure_distance_cache_hits",structures.distanceHits},{"structure_distance_cache_misses",structures.distanceMisses},
      {"structural_frontier",structurePolicy()?"uncovered communication cuts: main score versus cross-thread boundary count; ownership Grassmann/nullity distance to fixed pins":"disabled; static-only feature ablation"},
      {"feature_axes",std::move(axes)},{"observations_with_recognized_features",featureObservations},
      {"feature_values_are_physical_register_predictions",false},{"feature_zero_contract","sparse model counts or unmodeled proxies; not proof of free native work"},
      {"identity_duplicate_scope","exact assignments in live reservoir; equal features never deduplicate programs"},
      {"extra_candidate_scores",0},{"certificate_rebuilds",certificateRebuilds},
      {"discard_reason_counts",llvm::json::Object{{"legacy-main-cost-frontier",frontierRejected},
        {"bounded-reservoir-dominated",dominatedDiscarded},{"bounded-reservoir-diversity",nondominatedDiscarded},
        {"legacy-hamming-capacity",featurePolicy()?0:capacityDiscarded},
        {"final-pin-identity-duplicate",finalDuplicates},{"final-native-capacity",finalUnselected}}},
      {"discard_examples",std::move(discardExamples)}};
  }
  void recordComplete(const LayoutConstraintGraph &g,const CandidateAssignment &a,
      const CostEstimate &cost,const ExperimentPolicy &policy,const SharedSearchBudget &budget)override {
    if(!policy.scope.fullGraph || cost.saturated || a.values.size()!=g.getVariables().size())return;
    ++observed;
    auto audit=[&](ArrayRef<uint64_t> ordinals) {
      if(!auditPaidObservations)return;
      llvm::json::Array key,features;for(auto n:ordinals)key.push_back(n);
      for(auto n:retentionFeatures(cost))features.push_back(n);
      paidObservationAudit.push_back(llvm::json::Object{{"assignment_stable_ordinals",std::move(key)},
        {"feature_values",std::move(features)},{"cost",layoutCostJSON(cost)},{"origin",origin},
        {"probes_at_observation",budget.used},{"refinement_at_observation",budget.refinementUsed}});
    };
    for(const auto &x:samples)if(sameAssignment(x.values,a.values)){audit(x.key);++duplicates;return;}
    PaidSample incoming;incoming.values=a.values;incoming.cost=cost;incoming.origin=origin;
    incoming.features=retentionFeatures(cost);
    if(featurePolicy()){incoming.structure=structures.analyze(g,a.values);incoming.structure.proven?++structureProven:++structureUnknown;}
    featureObservations+=llvm::any_of(cost.features,[](const auto &f){return llvm::any_of(ArrayRef<llvm::StringLiteral>(retentionAxes.data(),20),[&](auto name){return StringRef(f.first)==name;});});
    incoming.probes=budget.used;incoming.refinement=budget.refinementUsed;
    for(const auto &v:g.getVariables()) {
      auto candidate=llvm::find_if(v.candidates,[&](const auto &c){return c.value==a.values.lookup(v.id);});
      if(candidate==v.candidates.end())return;
      incoming.key.push_back(candidate->stableOrdinal);
    }
    audit(incoming.key);samples.push_back(std::move(incoming));peak=std::max<uint64_t>(peak,samples.size());
    if(featurePolicy()){retainFeatureReservoir();return;}
    const auto &frontierBest=*llvm::min_element(samples,sampleLess);
    const auto bestCost=frontierBest.cost.cost;
    auto outsideFrontier=[&](const PaidSample &x){
      auto best=bestCost.instructionPathAndWork, value=x.cost.cost.instructionPathAndWork;
      if(!best)return bestCost<x.cost.cost || x.cost.cost<bestCost;
      return best<=UINT64_MAX/2 && value>2*best;
    };
    frontierRejected+=llvm::count_if(samples,outsideFrontier);llvm::erase_if(samples,outsideFrontier);
    // Reproduction path: unchanged full-cost best plus exact-assignment Hamming.
    SmallVector<PaidSample,0> selected;
    while(!samples.empty() && selected.size()<capacity) {
      size_t best=0;uint64_t bestDistance=0;
      for(size_t i=0;i<samples.size();++i) {
        uint64_t distance=std::numeric_limits<uint64_t>::max();
        for(const auto &prior:selected)distance=std::min(distance,assignmentDistance(samples[i].values,prior.values));
        if(i==0 || distance>bestDistance || (distance==bestDistance && sampleLess(samples[i],samples[best]))) {best=i;bestDistance=distance;}
      }
      selected.push_back(std::move(samples[best]));samples.erase(samples.begin()+best);
    }
    capacityDiscarded+=samples.size();samples=std::move(selected);
  }
};
FailureOr<LayoutSolution> rebuildArchiveCertificate(const LayoutConstraintGraph &g,
    LayoutTarget &target,const PaidSample &sample,const ExperimentPolicy &policy) {
  if(sample.values.size()!=g.getVariables().size())return failure();
  auto scored=scoreAssignment(g,CandidateAssignment{sample.values},target,policy);
  if(failed(scored) || scored->cost.saturated || scored->cost.cost<sample.cost.cost || sample.cost.cost<scored->cost.cost)return failure();
  LayoutSolution s;s.policy=policy;s.assignments=sample.values;s.cost=scored->cost;s.conversions=std::move(scored->conversions);
  for(const auto &c:g.getConstraints()) {
    if(c.instruction){auto *tuple=findInstructionSupport(g,c,s.assignments);if(!tuple)return failure();s.instructionBindings[c.id]=tuple->binding;}
    if(c.reduction){auto *pair=findReductionSupport(g,c,s.assignments);if(!pair)return failure();s.reductionBindings[c.id]=pair->binding;}
  }
  auto loc=UnknownLoc::get(g.getVariables().front().shapedType.getContext());
  if(failed(verifySolvedLayoutGraph(g,s,target,loc,policy.conversion)))return failure();
  return s;
}
}
llvm::json::Object describeArchiveStructure(const LayoutConstraintGraph &graph,const CandidateAssignment &assignment) {
  StructureCache cache;return cache.analyze(graph,assignment.values).json();
}
FailureOr<uint64_t> archiveStructureDistance(const LayoutConstraintGraph &graph,const CandidateAssignment &a,const CandidateAssignment &b) {
  StructureCache cache;auto lhs=cache.analyze(graph,a.values),rhs=cache.analyze(graph,b.values);
  if(!lhs.proven || !rhs.proven)return failure();return cache.distance(lhs,rhs);
}
FailureOr<LayoutSolution> solveCandidateArchive(LayoutConstraintGraph &graph,LayoutTarget &target,
    SelectionMode mode,SolverOptions options,unsigned capacity,SmallVectorImpl<ArchivedLayoutCandidate> &out,
    llvm::json::Object &record,LayoutTarget *minWorkVariantTarget,ArchiveRetentionPolicy retention,bool auditPaidObservations,ModuleOp source) {
  using Clock=std::chrono::steady_clock;auto start=Clock::now();
  auto ns=[](auto t){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();};
  // Resolve once on the complete source and frozen domain, before attaching an
  // observer or partitioning the one shared budget. Unknown means the complete
  // legacy archive algorithm; this never discovers or unions a second pool.
  const bool sourceGrammarRecipe=retention==ArchiveRetentionPolicy::CalibratedSourceGrammarV1;
  std::optional<llvm::json::Object> resolution;
  if(retention==ArchiveRetentionPolicy::CalibratedSourceGrammarV1) {
    auto gate=checkArchiveSourceGrammar(source,graph);
    retention=gate.eligible?ArchiveRetentionPolicy::PaidFeatureFrontierV1:ArchiveRetentionPolicy::LegacyHammingV2;
    resolution=llvm::json::Object{{"contract","calibrated-source-grammar-retention-routing-v1"},
      {"requested","calibrated-source-grammar-v1"},{"effective",archiveRetentionPolicyName(retention)},
      {"resolved_before_search",true},{"resolver_calls",1},{"additional_searches",0},
      {"native_output_guaranteed",false},{"gate",std::move(gate.evidence)}};
  }
  SharedSearchBudget owned;auto &budget=options.sharedBudget?*options.sharedBudget:owned;options.sharedBudget=&budget;
  if(capacity<2 || capacity>6 || !out.empty() || !graph.isFinalized() || graph.getVariables().empty() ||
      budget.used || budget.refinementUsed || budget.limit>65536 || budget.refinementLimit>256 ||
      budget.limit<16 || budget.refinementLimit<4 || options.policy.conversion!=ConversionPolicy::Independent) {
    record["failure_kind"]="fatal-integrity";record["failure"]="invalid candidate archive options or nonfresh shared budget";return failure();
  }
  if((sourceGrammarRecipe || retention!=ArchiveRetentionPolicy::LegacyHammingV2) && capacity!=4) {record["failure_kind"]="fatal-integrity";record["failure"]="paid feature retention requires native capacity K4";return failure();}
  const auto globalLimit=budget.limit,globalRefinement=budget.refinementLimit;
  const bool networkArchive=bool(minWorkVariantTarget);
  if(networkArchive && capacity!=4){record["failure_kind"]="fatal-integrity";record["failure"]="network archive requires native capacity K4";return failure();}
  const uint64_t reservedVariantProbes=networkArchive?1:0;
  const uint64_t searchLimit=globalLimit-reservedVariantProbes;
  const uint64_t localLimit=mode==SelectionMode::Joint?globalLimit/4:searchLimit;
  // Producer-local completion needs a complete informed witness. It may use
  // the shared refinement cap; joint search receives only the actual remainder.
  // Probe work still has the separate one-quarter terminal cap.
  const uint64_t localRefinement=globalRefinement;
  PaidArchive archive(capacity,retention,auditPaidObservations);options.completeCandidateObserver=&archive;
  auto localGraph=graph;auto localOptions=options;localOptions.seedBeamSolution=false;localOptions.policy.scope.informedLocal=true;
  budget.limit=budget.stageLimit=localLimit;budget.refinementLimit=budget.stageRefinementLimit=localRefinement;
  llvm::json::Object localRecord;archive.origin="local-completion";auto localStart=Clock::now();
  auto local=solveAblation(localGraph,target,SelectionMode::ProducerFirst,localOptions,localRecord);auto localNs=ns(localStart);
  auto localEnd=counters(budget);
  budget.limit=budget.stageLimit=searchLimit;budget.refinementLimit=budget.stageRefinementLimit=globalRefinement;
  auto recoverable=[](const llvm::json::Object &r){auto k=r.getString("failure_kind");return k && (*k=="search-budget" || *k=="unknown" || *k=="unsupported" || *k=="no-feasible");};
  if(failed(local) && !recoverable(localRecord)) {
    record["failure_kind"]="fatal-integrity";record["failure"]="local archive phase failed integrity";record["local"]=std::move(localRecord);record["total"]=counters(budget);return failure();
  }
  auto jointGraph=graph;auto jointOptions=options;jointOptions.policy.scope.fullGraph=true;jointOptions.policy.scope.informedLocal=false;
  llvm::json::Object jointRecord;archive.origin="joint-search";auto jointStart=Clock::now();
  FailureOr<LayoutSolution> joint=failure();
  if(mode==SelectionMode::Joint)joint=solveAblation(jointGraph,target,SelectionMode::Joint,jointOptions,jointRecord);
  else jointRecord["status"]="not-requested-local-control";
  auto jointNs=mode==SelectionMode::Joint?ns(jointStart):0;
  if(mode==SelectionMode::Joint && failed(joint) && !recoverable(jointRecord)) {
    record["failure_kind"]="fatal-integrity";record["failure"]="joint archive phase failed integrity";record["joint"]=std::move(jointRecord);record["total"]=counters(budget);return failure();
  }
  auto policy=options.policy;policy.scope.fullGraph=true;policy.scope.informedLocal=false;
  auto append=[&](const LayoutSolution &s,StringRef origin,bool isLocal,bool isJoint,uint64_t probes,uint64_t refinement) {
    for(auto &prior:out)if(sameAssignment(prior.solution.assignments,s.assignments)) {
      prior.origins.push_back(origin.str());prior.localIncumbent|=isLocal;prior.jointIncumbent|=isJoint;return;
    }
    ArchivedLayoutCandidate value;value.solution=s;value.solution.policy=policy;value.origins.push_back(origin.str());
    if(archive.featurePolicy())value.retentionStructure=archive.structures.analyze(graph,s.assignments).json();
    value.localIncumbent=isLocal;value.jointIncumbent=isJoint;value.retentionReason=isLocal?"local-incumbent-pin":isJoint?"joint-incumbent-pin":archive.featurePolicy()?(archive.structurePolicy() && value.retentionStructure.getBoolean("proven").value_or(false)?"paid-gf2-cut-frontier-diversity":"paid-feature-frontier-diversity"):"legacy-hamming-diversity";value.probesAtObservation=probes;value.refinementAtObservation=refinement;out.push_back(std::move(value));
  };
  if(succeeded(local))append(*local,"local-incumbent",true,false,*localEnd.getInteger("probes"),*localEnd.getInteger("refinement"));
  if(succeeded(joint))append(*joint,"joint-incumbent",false,true,budget.used,budget.refinementUsed);
  // Pins are reserved before any optional diversity candidate can fill a slot.
  auto certificateStart=Clock::now();
  while(out.size()<capacity && !archive.samples.empty()) {
    size_t best=0;uint64_t bestDistance=0;
    for(size_t i=0;i<archive.samples.size();++i) {
      uint64_t distance=std::numeric_limits<uint64_t>::max();
      for(const auto &prior:out)distance=std::min(distance,assignmentDistance(archive.samples[i].values,prior.solution.assignments));
      if(i==0 || distance>bestDistance || (distance==bestDistance && sampleLess(archive.samples[i],archive.samples[best]))) {best=i;bestDistance=distance;}
    }
    if(archive.featurePolicy()) {best=archive.selectFeatureCandidate(graph,out);if(best==archive.samples.size())break;}
    auto sample=std::move(archive.samples[best]);archive.samples.erase(archive.samples.begin()+best);
    if(!archive.featurePolicy() && !out.empty()) {
      const auto &incumbent=*llvm::min_element(out,[](const auto &a,const auto &b){return a.solution.cost.cost<b.solution.cost.cost;});
      auto bound=incumbent.solution.cost.cost.instructionPathAndWork;
      auto value=sample.cost.cost.instructionPathAndWork;
      bool outside=!bound?(incumbent.solution.cost.cost<sample.cost.cost || sample.cost.cost<incumbent.solution.cost.cost):bound<=UINT64_MAX/2 && value>2*bound;
      if(outside){++archive.frontierRejected;continue;}
    }
    if(llvm::any_of(out,[&](const auto &prior){return sameAssignment(prior.solution.assignments,sample.values);})) {++archive.finalDuplicates;continue;}
    ++archive.certificateRebuilds;auto rebuilt=rebuildArchiveCertificate(graph,target,sample,policy);
    if(failed(rebuilt)){record["failure_kind"]="fatal-integrity";record["failure"]="paid archive candidate certificate mismatch";return failure();}
    append(*rebuilt,sample.origin,false,false,sample.probes,sample.refinement);
  }
  for(const auto &sample:archive.samples) {
    if(llvm::any_of(out,[&](const auto &x){return sameAssignment(sample.values,x.solution.assignments);}))++archive.finalDuplicates;
    else {++archive.finalUnselected;archive.noteDiscard(sample,"final-native-capacity");}
  }
  // Revalidate pins under the common graph as well; no score is used to expand
  // the archive here, and none of these certificate-only checks is a search.
  for(auto &entry:out) {
    PaidSample sample;sample.values=entry.solution.assignments;sample.cost=entry.solution.cost;
    ++archive.certificateRebuilds;auto rebuilt=rebuildArchiveCertificate(graph,target,sample,policy);
    if(failed(rebuilt)){record["failure_kind"]="fatal-integrity";record["failure"]="archive incumbent certificate mismatch";return failure();}
    auto components=std::move(entry.solution.components);entry.solution=*rebuilt;entry.solution.components=std::move(components);
  }
  auto certificateNs=ns(certificateStart);
  llvm::json::Object variantRecord;
  if(networkArchive) {
    // The native K4 budget covers (assignment, objective), not four layouts
    // plus an uncharged fifth compile. Every new numerical score consumes the
    // reserved complete-score probe; a structural plan check is separately timed.
    budget.limit=budget.stageLimit=globalLimit;
    variantRecord=llvm::json::Object{{"contract","fixed-K4-network-template-v1"},{"reserved_score_probes",1},{"layout_search_limit",searchLimit},{"variant_scores",0},{"variant_plan_checks",0},{"native_capacity",capacity},{"objective","min-work"}};
    if(!out.empty()) {
      unsigned staticFirst=0;
      for(unsigned i=1;i<out.size();++i)if(out[i].solution.cost.cost<out[staticFirst].solution.cost.cost)staticFirst=i;
      SmallVector<unsigned> order;
      for(unsigned i=0;i<out.size();++i)if(out[i].localIncumbent || out[i].jointIncumbent)order.push_back(i);
      if(!llvm::is_contained(order,staticFirst))order.push_back(staticFirst);
      for(unsigned i=0;i<out.size();++i)if(!llvm::is_contained(order,i))order.push_back(i);
      SmallVector<ArchivedLayoutCandidate,0> ordered;
      for(unsigned i:order)ordered.push_back(std::move(out[i]));out=std::move(ordered);
      unsigned first=llvm::find(order,staticFirst)-order.begin();
      llvm::json::Array originalOrder;for(unsigned i:order)originalOrder.push_back(i);
      variantRecord["auto_original_indices"]=std::move(originalOrder);variantRecord["static_first_auto_index"]=first;
      llvm::json::Array pins;
      for(unsigned i=0;i<out.size();++i)if(out[i].localIncumbent || out[i].jointIncumbent || i==first)
        pins.push_back(llvm::json::Object{{"original_index",order[i]},{"retained_slot",i},{"local_incumbent",out[i].localIncumbent},{"joint_incumbent",out[i].jointIncumbent},{"static_first",i==first},{"assignment",assignmentJSON(graph,out[i].solution.assignments)}});
      variantRecord["pins"]=std::move(pins);
      auto checkStart=Clock::now();auto comparison=compareLayoutNetworkObjectives(graph,CandidateAssignment{out[first].solution.assignments});
      variantRecord["variant_plan_checks"]=1;variantRecord["plan_check_wall_ns"]=ns(checkStart);
      variantRecord["plan_comparison_proven"]=comparison.proven;variantRecord["plan_changed"]=comparison.changed;
      variantRecord["compared_shuffle_boundaries"]=comparison.comparedShuffleBoundaries;variantRecord["plan_comparison_reason"]=comparison.reason;
      variantRecord["boundary_plans"]=std::move(comparison.boundaryPlans);
      if(comparison.proven && comparison.changed) {
        if(!budget.consume(false)){record["failure_kind"]="fatal-integrity";record["failure"]="reserved network variant probe unavailable";return failure();}
        ++budget.completeScores;variantRecord["variant_scores"]=1;
        auto scoreStart=Clock::now();auto scored=scoreAssignment(graph,CandidateAssignment{out[first].solution.assignments},*minWorkVariantTarget,policy);
        variantRecord["score_wall_ns"]=ns(scoreStart);
        if(failed(scored) || scored->cost.saturated){record["failure_kind"]="fatal-integrity";record["failure"]="network variant complete score unavailable";return failure();}
        auto alternative=out[first];alternative.solution.cost=scored->cost;alternative.solution.conversions=std::move(scored->conversions);
        alternative.networkObjective="min-work";alternative.retentionReason="fixed-min-work-template";alternative.networkVariant=true;alternative.localIncumbent=false;alternative.jointIncumbent=false;alternative.origins={"fixed-min-work-template"};
        alternative.probesAtObservation=budget.used;alternative.refinementAtObservation=budget.refinementUsed;
        if(failed(verifySolvedLayoutGraph(graph,alternative.solution,*minWorkVariantTarget,UnknownLoc::get(graph.getVariables().front().shapedType.getContext()),policy.conversion))){record["failure_kind"]="fatal-integrity";record["failure"]="network variant certificate invalid";return failure();}
        if(out.size()==capacity) {
          if(first==capacity-1 || out.back().localIncumbent || out.back().jointIncumbent){record["failure_kind"]="fatal-integrity";record["failure"]="network template would discard a required auto witness";return failure();}
          variantRecord["replaced_auto_original_index"]=order.back();
          variantRecord["replaced_auto_assignment"]=assignmentJSON(graph,out.back().solution.assignments);
          out.resize(capacity-1);
        }
        variantRecord["variant_slot"]=out.size();
        out.push_back(std::move(alternative));variantRecord["status"]=variantRecord.getInteger("replaced_auto_original_index")?"min-work-replaces-fourth-auto":"min-work-added-within-K4";
      } else variantRecord["status"]=comparison.proven?"identical-plans-keep-auto":"unavailable-plan-proof-keep-auto";
    } else variantRecord["status"]="no-auto-witness";
    variantRecord["unused_reserved_score_probes"]=1-variantRecord.getInteger("variant_scores").value_or(0);
    variantRecord["budget_limit_restored"]=globalLimit;variantRecord["budget_refinement_unchanged"]=globalRefinement;
    variantRecord["resource_proxy_is_physical_register_prediction"]=false;
  }
  llvm::json::Object summary{{"version",networkArchive?(archive.featurePolicy()?"shared-network-candidate-archive-v3":"shared-network-candidate-archive-v2"):(archive.featurePolicy()?"shared-complete-candidate-archive-v2":"shared-complete-candidate-archive-v1")},{"capacity",capacity},{"retained",out.size()},
    {"limit",globalLimit},{"refinement_limit",globalRefinement},{"local_terminal_probes",localLimit},{"local_terminal_refinement",localRefinement},
    {"local_end",std::move(localEnd)},{"total",counters(budget)},{"local_status",succeeded(local)?"complete-witness":"no-local-witness"},
    {"joint_status",mode==SelectionMode::Joint?(succeeded(joint)?"complete-witness":"no-joint-witness"):"not-requested-local-control"},{"paid_complete_observations",archive.observed},{"observed_duplicates",archive.duplicates},{"frontier_rejections",archive.frontierRejected},{"capacity_discards",archive.capacityDiscarded},
    {"extra_candidate_frontier","main-score <=2x best observed/incumbent; primary-zero requires equal full tuple; pins exempt"},
    {"local_wall_ns",localNs},{"joint_wall_ns",jointNs},{"certificate_wall_ns",certificateNs},{"total_wall_ns",ns(start)},
    {"deduplication","exact full assignment attributes, never cost/features"},{"diversity","streaming full-cost best plus exact-assignment farthest-first Hamming"},
    {"pins","retain local and joint final incumbents before optional explored candidates"},{"certificate_rescores","retained candidates only; no new candidate or search"}};
  summary["retention"]=archive.receipt();
  if(resolution)summary["retention_resolution"]=std::move(*resolution);
  if(auditPaidObservations)summary["paid_observation_audit"]=llvm::json::Object{
    {"contract","paid-full-assignment-observer-ledger-v1"},{"selection_influence","none; append-only diagnostics; no scores"},
    {"bound",globalLimit},{"observed",archive.observed},{"entries",std::move(archive.paidObservationAudit)}};
  if(archive.featurePolicy()) {
    summary["extra_candidate_frontier"]="bounded multiobjective paid feature reservoir; no main-score cutoff; pins exempt";
    summary["diversity"]=archive.structurePolicy()?"GF2 communication-cut Pareto representatives and live-operation ownership-subspace novelty; static feature dominance only within a cut; exact-assignment identity":"static-only paid feature extrema/Pareto/gain/distance ablation; structure descriptors are diagnostic and do not influence selection";
  }
  auto components=[](const LayoutSolution &solution){
    llvm::json::Array result;for(const auto &c:solution.components)result.push_back(llvm::json::Object{
      {"root",c.root},{"used_beam",c.usedBeam},{"expanded",c.expanded},{"mrv_probes",c.mrvProbes},
      {"refinement_probes",c.refinementProbes},{"seed_probes",c.seedProbes},{"seed_scores",c.seedScores},
      {"seed_witnesses",c.seedWitnesses},{"has_seed_witness",c.seedWitnesses!=0},
      {"seed_start_cost",c.seedWitnesses?llvm::json::Value(layoutCostJSON(c.seedStartCost)):llvm::json::Value(nullptr)},
      {"seed_stop_reason",c.seedStopReason},{"beam_budget_fallback",c.beamBudgetFallback},
      {"selected_component_cost",layoutCostJSON(c.cost)},{"exhaustive",c.exhaustive}});return result;
  };
  if(networkArchive) {
    summary["network_variant"]=std::move(variantRecord);
    summary["deduplication"]="exact full assignment attributes plus verified network objective; no feature-vector dedup";
    summary["pins"]="local/joint auto incumbents plus static-first auto retained before replacing optional fourth slot";
    summary["certificate_rescores"]="auto verification only; one genuinely new variant score charged in total probes";
  }
  summary["local_components"]=succeeded(local)?components(*local):llvm::json::Array{};
  summary["joint_components"]=succeeded(joint)?components(*joint):llvm::json::Array{};
  summary["local_component_cost_scope"]="returned internal LI stage; candidate entry cost is complete full-objective";
  summary["local"]=std::move(localRecord);summary["joint"]=std::move(jointRecord);record["candidate_archive_search"]=std::move(summary);record["total"]=counters(budget);
  if(out.empty()){
    auto &cause=mode==SelectionMode::Joint?*record.getObject("candidate_archive_search")->getObject("joint"):*record.getObject("candidate_archive_search")->getObject("local");
    record["failure_kind"]=cause.getString("failure_kind").value_or("unknown");
    record["failure"]="no complete verified candidate in shared archive";return failure();
  }
  auto best=llvm::min_element(out,[](const auto &a,const auto &b){
    if(a.networkVariant!=b.networkVariant)return !a.networkVariant;
    return a.solution.cost.cost<b.solution.cost.cost;
  });
  record["legal_return"]=true;record["search_complete"]=false;record["archive_complete_domain_exhausted"]=false;
  record["solve_timing_ns"]=llvm::json::Object{{"total",ns(start)}};
  return best->solution;
}

}
