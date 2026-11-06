#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"
#include <array>
#include <numeric>
#include <random>
using namespace mlir;
using namespace mlir::frisk;
namespace {
using Nine=std::array<uint64_t,9>;
CostVector costVector(const Nine &v) {
  return {v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]};
}
struct Relation {unsigned a,b; uint16_t mask; bool same;};
struct Fixture {
  unsigned n=0, domain=0;
  std::vector<std::array<Nine,4>> costs;
  std::vector<Relation> relations;
};
class FiniteCostTarget : public LayoutTarget {
public:
  const Fixture &f; bool failCost=false;
  unsigned mutable retries=0;
  explicit FiniteCostTarget(const Fixture &f):f(f) {}
  void enumerateCandidates(const LayoutVar &,SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &,Attribute a,Location) const override {return success(isa<StringAttr>(a));}
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {return failure();}
  FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &g,const CandidateAssignment &a) const override {
    if(failCost) return failure();
    Nine sum{};
    for(auto [id,attr]:a.values) {
      unsigned v=std::stoul(g.getVariable(id).stableName.substr(1));
      unsigned candidate=cast<StringAttr>(attr).getValue()[0]-'a';
      for(unsigned i=0;i<9;++i) sum[i]+=f.costs[v][candidate][i];
    }
    return CostEstimate{costVector(sum),false};
  }
  void enumerateRelaxedCandidates(const LayoutConstraintGraph &,const LayoutVar &,RelaxationKind,
                                  SmallVectorImpl<LayoutCandidate> &) const override {++retries;}
  LayoutProof verifyInstructionContract(const LayoutConstraintGraph &,
      const LayoutConstraint &c,ArrayRef<Attribute> values,Attribute binding) const override {
    for(auto &tuple:c.instruction->tuples)
      if(ArrayRef<Attribute>(tuple.encodings)==values && tuple.binding==binding)
        return {ProofStatus::Proven,{},"finite table member"};
    return {ProofStatus::Disproven,{},"not a finite table member"};
  }
};
bool legal(const Fixture &f,const std::vector<unsigned> &values) {
  for(auto &r:f.relations) {
    unsigned a=values[r.a],b=values[r.b];
    if(r.same ? a!=b : !(r.mask & (1u<<(4*a+b)))) return false;
  }
  return true;
}
std::optional<Nine> oracle(const Fixture &f) {
  uint64_t count=1; for(unsigned i=0;i<f.n;++i) count*=f.domain;
  std::optional<Nine> best;
  for(uint64_t number=0;number<count;++number) {
    auto remaining=number; std::vector<unsigned> assignment(f.n); Nine cost{};
    for(unsigned i=0;i<f.n;++i) {
      assignment[i]=remaining%f.domain; remaining/=f.domain;
      for(unsigned j=0;j<9;++j) cost[j]+=f.costs[i][assignment[i]][j];
    }
    if(legal(f,assignment) && (!best || cost<*best)) best=cost;
  }
  return best;
}
LayoutConstraintGraph graphFor(const Fixture &f,Builder &b,ModuleOp host,std::mt19937_64 &rng) {
  LayoutConstraintGraph graph; std::vector<unsigned> order(f.n); std::iota(order.begin(),order.end(),0);
  std::shuffle(order.begin(),order.end(),rng); std::vector<LayoutVarID> ids(f.n);
  for(unsigned v:order) {
    auto id=graph.addVariable(LayoutKind::Distributed,RankedTensorType::get({4},b.getF32Type()),"v"+std::to_string(v));
    ids[v]=id; auto &var=graph.getVariable(id); var.state=LayoutState::CandidateSet;
    for(unsigned a=0;a<f.domain;++a) var.candidates.push_back({b.getStringAttr(std::string(1,'a'+a)),kInvalidProvenanceID,a});
    std::shuffle(var.candidates.begin(),var.candidates.end(),rng);
  }
  order.resize(f.relations.size());std::iota(order.begin(),order.end(),0);std::shuffle(order.begin(),order.end(),rng);
  for(unsigned ri:order) {
    auto &r=f.relations[ri]; auto a=ids[r.a],d=ids[r.b];
    auto id=graph.addConstraint(r.same?ConstraintKind::SameLayout:ConstraintKind::InstructionContract,
      ConstraintStrength::Hard,r.same?SmallVector<LayoutVarID>{a,d}:SmallVector<LayoutVarID>{a,d,d,d},
      r.same?nullptr:host.getOperation(),"r"+std::to_string(ri),"finite relation");
    if(!r.same) {
      auto &c=graph.getConstraints()[id];c.instruction=InstructionLayoutContract{host,{}, {}};
      for(unsigned x=0;x<f.domain;++x) for(unsigned y=0;y<f.domain;++y)
        if(r.mask & (1u<<(4*x+y))) {
          auto lhs=b.getStringAttr(std::string(1,'a'+x)),rhs=b.getStringAttr(std::string(1,'a'+y));
          c.instruction->tuples.push_back({{lhs,rhs,rhs,rhs},b.getStringAttr("binding-r"+std::to_string(ri))});
        }
    }
  }
  return graph;
}
std::string canonical(const LayoutConstraintGraph &g,const LayoutSolution &s) {
  // This finite fixture domain has no boundary/reduction plans. Do not let a
  // future extension silently compare dumps that omit actual plan differences.
  EXPECT_TRUE(s.conversions.empty());
  EXPECT_TRUE(s.reductionBindings.empty());
  std::vector<std::string> entries;
  for(auto &v:g.getVariables()) {
    std::string line=v.stableName+"="; llvm::raw_string_ostream(line)<<s.assignments.lookup(v.id);entries.push_back(line);
  }
  for(auto [id,a]:s.instructionBindings) {
    std::string line=g.getProvenances()[g.getConstraint(id).provenance].rule+"=";
    llvm::raw_string_ostream(line)<<a;entries.push_back(line);
  }
  llvm::sort(entries); std::string result;
  for(auto &e:entries) result+=e+"\n";
  return result;
}
TEST(LayoutSolverPropertyTest, FixedSeedsIndependentNineDimensionalOracleAndTwentyShuffles) {
  MLIRContext ctx; Builder b(&ctx); auto host=OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
  ScopedDiagnosticHandler quiet(&ctx,[](Diagnostic &){return success();});
  for(uint64_t seed:{1u,7u,42u,20260816u}) {
    std::mt19937_64 rng(seed);
    for(unsigned ci=0;ci<16;++ci) {
      SCOPED_TRACE("seed="+std::to_string(seed)+" case="+std::to_string(ci));
      Fixture f; f.n=ci==15?8:2+rng()%7;f.domain=1+ci%4;f.costs.resize(f.n);
      for(auto &v:f.costs) for(auto &c:v) for(auto &dimension:c) dimension=rng()%4;
      // Mixed equality and genuinely non-binary tables; preserve a known
      // witness in most cases, leave others possibly infeasible.
      for(unsigned i=1;i<f.n;++i) {
        auto mask=uint16_t(rng()); if(ci%3) mask|=1;
        f.relations.push_back({i-1,i,mask,i%3==0});
      }
      for(unsigned i=0;i<ci%8;++i) {
        unsigned a=rng()%f.n,d=(a+1+rng()%(f.n-1))%f.n;
        auto mask=uint16_t(rng());if(ci%3)mask|=1;
        f.relations.push_back({a,d,mask,false});
      }
      auto best=oracle(f);std::string first;
      for(unsigned shuffle=0;shuffle<20;++shuffle) {
        SCOPED_TRACE("shuffle="+std::to_string(shuffle));
        auto graph=graphFor(f,b,*host,rng);FiniteCostTarget target(f);LayoutSolveFailure why;
        SolverOptions options;options.exactCombinationLimit=65536;options.maxExpandedStates=10000000;
        auto s=solveLayoutGraph(graph,target,options,&why);
        ASSERT_EQ(succeeded(s),bool(best)) << why.reason;
        if(!best) {EXPECT_EQ(why.kind,LayoutSolveFailureKind::HardConflict);continue;}
        EXPECT_FALSE(s->cost.cost<costVector(*best));EXPECT_FALSE(costVector(*best)<s->cost.cost);
        std::vector<unsigned> assignment(f.n);
        for(auto &v:graph.getVariables()) assignment[std::stoul(v.stableName.substr(1))]=cast<StringAttr>(s->assignments.lookup(v.id)).getValue()[0]-'a';
        EXPECT_TRUE(legal(f,assignment));
        EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,*s,target,b.getUnknownLoc())));
        for(auto &component:s->components) EXPECT_TRUE(component.exhaustive);
        auto dump=canonical(graph,*s); if(!shuffle)first=dump;else EXPECT_EQ(first,dump);
      }
    }
  }
}
TEST(LayoutSolverPropertyTest, DefaultBudgetsAndNonRetryableFailures) {
  SolverOptions options;EXPECT_EQ(options.exactCombinationLimit,256u);EXPECT_EQ(options.beamWidth,32u);EXPECT_EQ(options.maxExpandedStates,65536u);
  MLIRContext ctx; Builder b(&ctx);auto host=OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
  ScopedDiagnosticHandler quiet(&ctx,[](Diagnostic &){return success();});std::mt19937_64 rng(42);
  for(unsigned n:{4u,5u}) {
    Fixture f;f.n=n;f.domain=4;f.costs.resize(n);
    for(unsigned i=1;i<n;++i)f.relations.push_back({i-1,i,0xffff,false});
    auto g=graphFor(f,b,*host,rng);FiniteCostTarget target(f);auto s=solveLayoutGraph(g,target);
    ASSERT_TRUE(succeeded(s));ASSERT_EQ(s->components.size(),1u);
    EXPECT_EQ(s->components[0].usedBeam,n==5);EXPECT_LE(s->components[0].expanded,65536u);
    if(n==5){EXPECT_EQ(s->components[0].beamWidth,32u);EXPECT_GT(s->components[0].beamDiscarded,0u);}
    for(bool costFailure:{false,true}) {
      g=graphFor(f,b,*host,rng);target.failCost=costFailure;
      auto opts=options;if(!costFailure)opts.maxExpandedStates=1;
      SmallVector<RelaxationRecord, 0> records;LayoutSolveFailure why;
      EXPECT_TRUE(failed(attemptControlledRelaxation(g,target,opts,records,&why)));
      EXPECT_EQ(why.kind,costFailure?LayoutSolveFailureKind::CostFailure:LayoutSolveFailureKind::SearchBudget);
      EXPECT_TRUE(records.empty());EXPECT_EQ(target.retries,0u);
    }
  }
}
TEST(LayoutSolverPropertyTest, RefinementNeverWorsensBeamAndRespectsIndependentOracle) {
  MLIRContext ctx; Builder b(&ctx);
  auto host = OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
  ScopedDiagnosticHandler quiet(&ctx, [](Diagnostic &) { return success(); });
  for (uint64_t seed : {1u, 7u, 42u, 20260816u}) {
    std::mt19937_64 rng(seed);
    for (unsigned ci = 0; ci < 12; ++ci) {
      SCOPED_TRACE("seed=" + std::to_string(seed) + " case=" + std::to_string(ci));
      Fixture f; f.n = 3 + ci % 4; f.domain = 2 + ci % 2; f.costs.resize(f.n);
      for (auto &v : f.costs) for (auto &c : v) for (auto &d : c) d = rng() % 7;
      for (unsigned i = 1; i < f.n; ++i)
        f.relations.push_back({i-1, i, uint16_t(rng() | 1u), false});
      auto optimum = oracle(f); ASSERT_TRUE(optimum);
      std::string first;
      for (unsigned shuffle = 0; shuffle < 20; ++shuffle) {
        auto offGraph = graphFor(f, b, *host, rng), onGraph = offGraph;
        FiniteCostTarget target(f);
        SolverOptions options{1, 1, 65536, false, 256};
        auto off = solveLayoutGraph(offGraph, target, options);
        options.refineBeamSolution = true;
        auto on = solveLayoutGraph(onGraph, target, options);
        ASSERT_EQ(succeeded(on), succeeded(off));
        if (failed(on)) continue; // Refinement cannot rescue an absent incumbent.
        EXPECT_FALSE(off->cost.cost < on->cost.cost);
        EXPECT_FALSE(on->cost.cost < costVector(*optimum));
        EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(onGraph, *on, target, b.getUnknownLoc())));
        auto dump = canonical(onGraph, *on);
        if (!shuffle) first = dump; else EXPECT_EQ(first, dump);
        for (auto &c : on->components) {
          EXPECT_EQ(c.expanded, c.mrvProbes + c.refinementProbes);
          EXPECT_LE(c.refinementProbes, 256u);
          EXPECT_LE(c.expanded, 65536u);
        }
      }
    }
  }
}
TEST(LayoutSolverPropertyTest, EachOfNineCostDimensionsCanDecideTheOptimum) {
  MLIRContext ctx;Builder b(&ctx);auto host=OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
  std::mt19937_64 rng(42);
  for(unsigned dimension=0;dimension<9;++dimension) {
    SCOPED_TRACE("cost dimension="+std::to_string(dimension));
    Fixture f;f.n=1;f.domain=3;f.costs.resize(1);
    for(unsigned candidate=0;candidate<3;++candidate) {
      f.costs[0][candidate].fill(9);
      f.costs[0][candidate][dimension]=candidate==2?0:1;
    }
    auto expected=oracle(f);ASSERT_TRUE(bool(expected));
    auto graph=graphFor(f,b,*host,rng);FiniteCostTarget target(f);
    auto s=solveLayoutGraph(graph,target);ASSERT_TRUE(succeeded(s));
    EXPECT_EQ(s->assignments.lookup(0),b.getStringAttr("c"));
    EXPECT_FALSE(s->cost.cost<costVector(*expected));EXPECT_FALSE(costVector(*expected)<s->cost.cost);
  }
}
TEST(LayoutSolverPropertyTest, EmptyUnknownIncompleteAndDefaultExpansionCeiling) {
  MLIRContext ctx;Builder b(&ctx);auto host=OwningOpRef<ModuleOp>(ModuleOp::create(b.getUnknownLoc()));
  ScopedDiagnosticHandler quiet(&ctx,[](Diagnostic &){return success();});std::mt19937_64 rng(7);
  Fixture f;f.n=3;f.domain=2;f.costs.resize(3);
  // a=a,b=b equality for first two tables. Third forbids (a,a).
  f.relations={{0,1,uint16_t((1<<0)|(1<<5)),false},
               {1,2,uint16_t((1<<0)|(1<<5)),false},
               {0,2,uint16_t((1<<1)|(1<<4)|(1<<5)),false}};
  FiniteCostTarget target(f);SmallVector<RelaxationRecord, 0> records;LayoutSolveFailure why;
  auto g=graphFor(f,b,*host,rng);SolverOptions opts;opts.exactCombinationLimit=1;opts.beamWidth=1;
  EXPECT_TRUE(failed(attemptControlledRelaxation(g,target,opts,records,&why)));
  EXPECT_EQ(why.kind,LayoutSolveFailureKind::SearchIncomplete);EXPECT_TRUE(records.empty());EXPECT_EQ(target.retries,0u);
  f.relations.clear();g=graphFor(f,b,*host,rng);g.getVariable(0).candidates.clear();
  EXPECT_TRUE(failed(solveLayoutGraph(g,target,{},&why)));EXPECT_EQ(why.kind,LayoutSolveFailureKind::EmptyDomain);
  EXPECT_LT(why.variable,g.getVariables().size());EXPECT_EQ(why.reason,"empty input domain");
  g=graphFor(f,b,*host,rng);g.getVariable(0).candidates.clear();g.getVariable(0).candidatePreparationUnsupported=true;
  EXPECT_TRUE(failed(attemptControlledRelaxation(g,target,{},records,&why)));
  EXPECT_EQ(why.kind,LayoutSolveFailureKind::Unsupported);EXPECT_TRUE(records.empty());EXPECT_EQ(target.retries,0u);
  f.n=8;f.domain=4;f.costs.resize(8);
  for(unsigned i=1;i<8;++i)f.relations.push_back({i-1,i,0xffff,false});
  g=graphFor(f,b,*host,rng);opts={};opts.exactCombinationLimit=65536;
  EXPECT_TRUE(failed(attemptControlledRelaxation(g,target,opts,records,&why)));
  EXPECT_EQ(why.kind,LayoutSolveFailureKind::SearchBudget);EXPECT_TRUE(records.empty());EXPECT_EQ(target.retries,0u);
}
} // namespace
