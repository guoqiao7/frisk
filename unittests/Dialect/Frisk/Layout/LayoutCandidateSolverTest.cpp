#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <random>

using namespace mlir;
using namespace mlir::frisk;
namespace {
class RankedTarget : public LayoutTarget {
public:
  bool reverse = false;
  bool failCost = false;
  void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &, Attribute a, Location) const override {
    return success(isa<StringAttr>(a));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
    if (failCost) return failure();
    CostVector result;
    for (auto entry : a.values) {
      auto name = cast<StringAttr>(entry.second).getValue();
      result.instructionPathAndWork += (name == "a") == reverse ? 0 : 1;
    }
    return result;
  }
};
class LayoutCandidateSolverTest : public testing::Test {
protected:
  MLIRContext context;
  Builder builder{&context};
  Location loc = builder.getUnknownLoc();
  RankedTarget target;
  LayoutConstraintGraph graph;
  OwningOpRef<ModuleOp> source{ModuleOp::create(loc)};
  LayoutVarID variable(StringRef name, unsigned count = 2) {
    auto id = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({4}, builder.getF32Type()), name);
    for (unsigned i = 0; i < count; ++i)
      graph.getVariable(id).candidates.push_back(
          {builder.getStringAttr(std::string(1, char('a' + i))), kInvalidProvenanceID, 99-i});
    graph.getVariable(id).state = LayoutState::CandidateSet;
    return id;
  }
  void same(LayoutVarID a, LayoutVarID b) {
    graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
                        {a,b}, nullptr, "same", "same test layout");
  }
  // A finite, independently specified binary relation represented by a
  // four-role instruction table (the last three roles share one endpoint).
  void pair(LayoutVarID a, LayoutVarID b, unsigned mask) {
    auto id = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, {a,b,b,b}, *source, "finite", "finite relation");
    auto &c = graph.getConstraints()[id];
    c.instruction = InstructionLayoutContract{*source, {}, {}};
    for (unsigned x = 0; x < 2; ++x)
      for (unsigned y = 0; y < 2; ++y)
        if (mask & (1u << (2*x+y))) {
          auto lhs = builder.getStringAttr(x ? "b" : "a");
          auto rhs = builder.getStringAttr(y ? "b" : "a");
          c.instruction->tuples.push_back({{lhs,rhs,rhs,rhs}, builder.getStringAttr("plan")});
        }
  }
};

TEST_F(LayoutCandidateSolverTest, NineVariablesChooseCostNotOrdinal) {
  for (unsigned i = 0; i < 9; ++i) {
    auto id = variable("v" + std::to_string(i));
    if (i) same(id - 1, id);
  }
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto solution = solveLayoutGraph(graph, target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(solution->assignments.size(), 9u);
  for (auto assignment : solution->assignments)
    EXPECT_EQ(assignment.second, builder.getStringAttr("b"));
  EXPECT_EQ(solution->cost.cost.instructionPathAndWork, 0u);
  ASSERT_EQ(solution->components.size(), 1u);
  EXPECT_TRUE(solution->components[0].usedBeam);
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph, *solution, target, loc)));
}

TEST_F(LayoutCandidateSolverTest, MoreThanFourCandidatesAndCanonicalTie) {
  variable("only", 6);
  graph.getVariable(0).candidates.push_back(graph.getVariable(0).candidates.back());
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto solution = solveLayoutGraph(graph, target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_EQ(solution->assignments.lookup(0), builder.getStringAttr("b"));
  EXPECT_EQ(solution->components[0].deduplicated, 1u);
  EXPECT_TRUE(solution->components[0].exhaustive);
}

TEST_F(LayoutCandidateSolverTest, BudgetCountsRejectedAndAcceptedAttempts) {
  variable("one"); variable("two"); same(0,1);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  SolverOptions options;
  options.maxExpandedStates = 1;
  std::string errors;
  ScopedDiagnosticHandler quiet(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream(errors) << d; return success();
  });
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target, options)));
  EXPECT_NE(errors.find("search-budget-exceeded"), std::string::npos);
}

TEST_F(LayoutCandidateSolverTest, InvalidOptionsAndCostAreNotFreeSolutions) {
  variable("one"); ASSERT_TRUE(succeeded(graph.finalize(loc)));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  SolverOptions options; options.beamWidth = 0;
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target, options)));
  target.failCost = true;
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target)));
}

TEST_F(LayoutCandidateSolverTest, EmptyGraphStillValidatesOptionsAndConstraints) {
  SolverOptions options; options.beamWidth=0;
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target,options)));
  EXPECT_TRUE(succeeded(solveLayoutGraph(graph,target)));
  graph.addConstraint(ConstraintKind::SameLayout,ConstraintStrength::Hard,
                      {},nullptr,"invalid","no endpoints");
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target)));
}

TEST_F(LayoutCandidateSolverTest, UnfinalizedInsertionOrderHasCanonicalTie) {
  for (bool reverse : {false,true}) {
    graph=LayoutConstraintGraph();
    variable(reverse?"z":"a"); variable(reverse?"a":"z");
    pair(0,1,0b0110);
    auto s=solveLayoutGraph(graph,target);
    ASSERT_TRUE(succeeded(s));
    EXPECT_EQ(graph.getVariable(0).stableName,"a");
    EXPECT_EQ(s->assignments.lookup(0),builder.getStringAttr("a"));
    EXPECT_EQ(s->assignments.lookup(1),builder.getStringAttr("b"));
  }
}

TEST_F(LayoutCandidateSolverTest, Exact256BoundaryAndAbove) {
  for (unsigned n : {7u,8u,9u}) {
    graph = LayoutConstraintGraph();
    for (unsigned i = 0; i < n; ++i) {
      variable("v" + std::to_string(i));
      if (i) same(i-1,i);
    }
    ASSERT_TRUE(succeeded(graph.finalize(loc)));
    auto s = solveLayoutGraph(graph,target);
    ASSERT_TRUE(succeeded(s));
    EXPECT_EQ(s->components[0].combinations, 1u << n);
    EXPECT_EQ(s->components[0].usedBeam, n > 8);
    EXPECT_TRUE(s->components[0].exhaustive);
  }
}

TEST_F(LayoutCandidateSolverTest, BeamLosingOnlySolutionReportsIncomplete) {
  variable("x"); variable("y"); variable("z");
  pair(0,1,0b1001); pair(1,2,0b1001); pair(0,2,0b1110);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  SolverOptions options; options.exactCombinationLimit=1; options.beamWidth=1;
  std::string errors;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream(errors) << d; return success();
  });
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target,options)));
  EXPECT_NE(errors.find("search-incomplete"),std::string::npos) << errors;
  EXPECT_EQ(errors.find("no feasible"),std::string::npos);
  options.beamWidth=8;
  auto s = solveLayoutGraph(graph,target,options);
  ASSERT_TRUE(succeeded(s));
  EXPECT_EQ(s->instructionBindings.size(),3u);
  for (auto entry : s->assignments) EXPECT_EQ(entry.second,builder.getStringAttr("b"));
  EXPECT_TRUE(s->components[0].exhaustive);
}

TEST_F(LayoutCandidateSolverTest, ExhaustedInconsistentRelationsAreNotBudgetFailure) {
  variable("x"); variable("y"); variable("z");
  pair(0,1,0b1001); pair(1,2,0b1001); pair(0,2,0b0110);
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  std::string errors;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream(errors) << d; return success();
  });
  EXPECT_TRUE(failed(solveLayoutGraph(graph,target)));
  EXPECT_NE(errors.find("no feasible proven"),std::string::npos);
  EXPECT_EQ(errors.find("search-incomplete"),std::string::npos);
}

TEST_F(LayoutCandidateSolverTest, DomainProductOverflowDoesNotBecomeExact) {
  for (unsigned i=0;i<65;++i) {
    variable("v"+std::to_string(i)); if (i) same(i-1,i);
  }
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_TRUE(s->components[0].productOverflow);
  EXPECT_EQ(s->components[0].combinations,UINT64_MAX);
  EXPECT_TRUE(s->components[0].usedBeam);
  EXPECT_EQ(s->assignments.size(),65u);
}

TEST_F(LayoutCandidateSolverTest, RawOverflowCanPruneToExactSingleton) {
  for (unsigned i=0;i<65;++i) {
    variable("v"+std::to_string(i)); if (i) same(i-1,i);
    graph.addConstraint(ConstraintKind::RequireEncoding,ConstraintStrength::Hard,
        {i},nullptr,"require","hard seed",builder.getStringAttr("b"));
  }
  ASSERT_TRUE(succeeded(graph.finalize(loc)));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_TRUE(s->components[0].rawProductOverflow);
  EXPECT_FALSE(s->components[0].productOverflow);
  EXPECT_EQ(s->components[0].combinations,1u);
  EXPECT_FALSE(s->components[0].usedBeam);
  EXPECT_EQ(s->rejections.size(),65u);
}

TEST_F(LayoutCandidateSolverTest, FixedSeedIndependentExhaustiveOracleAndPermutation) {
  std::mt19937 random(230923);
  // Six binary variables = 64 complete assignments, independently enumerated.
  for (unsigned seed=0;seed<24;++seed) {
    graph=LayoutConstraintGraph();
    for (unsigned i=0;i<6;++i) variable("v"+std::to_string(i));
    SmallVector<unsigned> masks;
    for (unsigned i=0;i<6;++i) {
      unsigned mask=(random()%15)+1;
      masks.push_back(mask); pair(i,(i+1)%6,mask);
    }
    bool found=false; unsigned bestCost=99, best=0;
    for (unsigned bits=0;bits<64;++bits) {
      bool valid=true; unsigned cost=0;
      for (unsigned i=0;i<6;++i) {
        unsigned x=(bits>>(5-i))&1, y=(bits>>(5-(i+1)%6))&1;
        valid &= (masks[i] & (1u<<(2*x+y))) != 0;
        cost += !x;
      }
      if (valid && (!found || cost<bestCost)) {found=true;bestCost=cost;best=bits;}
    }
    ASSERT_TRUE(succeeded(graph.finalize(loc)));
    ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
    for (bool beam : {false,true}) {
      auto copy=graph;
      for (auto &v : copy.getVariables()) std::shuffle(v.candidates.begin(),v.candidates.end(),random);
      SolverOptions options; options.exactCombinationLimit=beam?1:4096;
      options.beamWidth=4096;
      auto s=solveLayoutGraph(copy,target,options);
      ASSERT_EQ(succeeded(s),found) << seed;
      if (!found) continue;
      EXPECT_EQ(s->cost.cost.instructionPathAndWork,bestCost) << seed;
      for (unsigned i=0;i<6;++i)
        EXPECT_EQ(s->assignments.lookup(i),builder.getStringAttr((best>>(5-i))&1?"b":"a")) << seed;
      EXPECT_TRUE(s->components[0].exhaustive);
    }
  }
}

class PhysicalSolverTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  RankedTensorType type = RankedTensorType::get({64}, b.getF32Type());
  class Target : public LayoutTarget {
  public:
    Attribute preferred;
    void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
    LogicalResult verifyCandidate(const LayoutVar &v, Attribute a, Location loc) const override {
      auto e = dyn_cast<DistributedEncodingAttr>(a);
      return e ? e.verifyForType(cast<ShapedType>(v.shapedType),loc) : failure();
    }
    FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
      CostVector cost;
      if (preferred && a.values.lookup(0) != preferred) cost.instructionPathAndWork=100;
      return cost;
    }
  } target;
  PhysicalSolverTest() {
    context.loadDialect<FriskDialect>(); context.allowUnregisteredDialects();
  }
  DistributedEncodingAttr encoding(bool rotate=false, bool replicate=false) {
    SmallVector<APInt> matrix;
    for (unsigned i=0;i<6;++i)
      for (unsigned j=0;j<6;++j) matrix.emplace_back(1,j==(rotate?(i+1)%6:i));
    auto map=BitLinearLayoutMapAttr::get(&context,
        b.getArrayAttr({b.getStringAttr("register"),b.getStringAttr("lane")}),
        b.getDenseI64ArrayAttr({1,5}),b.getArrayAttr({b.getStringAttr("d")}),
        b.getDenseI64ArrayAttr({6}),
        DenseIntElementsAttr::get(RankedTensorType::get({6,6},b.getI1Type()),matrix));
    return DistributedEncodingAttr::get(&context,map,
        b.getDenseI64ArrayAttr({2,32,replicate?2:1,1,1}),b.getI64IntegerAttr(replicate?2:1));
  }
  LayoutConstraintGraph make(Block &block, ArrayRef<Attribute> sources,
                             Attribute destination, bool existing=false) {
    auto value=block.addArgument(type,b.getUnknownLoc());
    OperationState state(b.getUnknownLoc(),"test.use"); state.addOperands(value);
    auto *op=Operation::create(state); block.push_back(op);
    LayoutConstraintGraph graph;
    auto p=graph.addVariable(LayoutKind::Distributed,type,"producer");
    auto u=graph.addVariable(LayoutKind::Distributed,type,"use");
    graph.getVariable(p).value=value;
    graph.getVariable(u).use=&op->getOpOperand(0);
    for (auto a:sources) graph.getVariable(p).candidates.push_back({a,kInvalidProvenanceID,99});
    graph.getVariable(u).candidates.push_back({destination,kInvalidProvenanceID,0});
    auto id=graph.addConstraint(ConstraintKind::Convertible,ConstraintStrength::Hard,
        {p,u},op,"convert","actual use");
    graph.getConstraints()[id].use=&op->getOpOperand(0);
    graph.getConstraints()[id].stableUseKey="use/0";
    graph.getConstraints()[id].existingConversion=existing;
    return graph;
  }
};

TEST_F(PhysicalSolverTest, BetterMainComputationBeatsZeroConversions) {
  Block block;
  auto a=encoding(), z=encoding(true);
  auto graph=make(block,{a,z},a);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  target.preferred=z;
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_EQ(s->assignments.lookup(0),z);
  ASSERT_EQ(s->conversions.size(),1u);
  EXPECT_EQ(s->cost.cost.instructionPathAndWork,0u);
  EXPECT_EQ(s->cost.cost.conversionBytesAndSync,768u);
  EXPECT_EQ(s->conversions[0].bytes,512u);
  EXPECT_EQ(s->conversions[0].synchronizationCost,2u);
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph,*s,target,b.getUnknownLoc())));
}

TEST_F(PhysicalSolverTest, EqualMainCostChoosesLowerConversionBytes) {
  Block block;
  auto a=encoding(), z=encoding(true), replicated=encoding(true,true);
  auto graph=make(block,{replicated,z},a);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_EQ(s->assignments.lookup(0),z);
  EXPECT_EQ(s->cost.cost.conversionBytesAndSync,768u);
}

TEST_F(PhysicalSolverTest, ExistingConversionChargedOnceWithoutReinsertion) {
  Block block;
  auto a=encoding(), z=encoding(true);
  auto graph=make(block,{z},a,true);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  auto s=solveLayoutGraph(graph,target);
  ASSERT_TRUE(succeeded(s));
  EXPECT_TRUE(s->conversions.empty());
  EXPECT_EQ(s->cost.cost.conversionBytesAndSync,768u);
  EXPECT_EQ(s->cost.cost.codeSize,1u);
}
} // namespace
