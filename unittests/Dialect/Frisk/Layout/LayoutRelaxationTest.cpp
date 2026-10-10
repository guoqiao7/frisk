#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/IR/Builders.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
// Explicit finite-domain target: exercises the real retry engine and solver.
class FiniteRelaxationTarget : public LayoutTarget {
public:
  mutable SmallVector<RelaxationKind> calls;
  bool failCost = false;
  bool delayed = false;
  bool wrongFirst = false;
  bool expectEmptyProposals = true;
  bool rejectGenerated = false;
  Attribute proposedEncoding;
  void enumerateCandidates(const LayoutVar &var, SmallVectorImpl<LayoutCandidate> &out) const override {
    if (rejectGenerated) out.push_back({StringAttr::get(var.shapedType.getContext(), "generated")});
  }
  void enumerateRelaxedCandidates(const LayoutConstraintGraph &graph,
      const LayoutVar &var, RelaxationKind kind,
      SmallVectorImpl<LayoutCandidate> &out) const override {
    calls.push_back(kind);
    if (expectEmptyProposals) EXPECT_TRUE(graph.getVariable(var.id).candidates.empty());
    if (kind == RelaxationKind::Replication && !delayed)
      out.push_back({proposedEncoding ? proposedEncoding :
          Attribute(StringAttr::get(var.shapedType.getContext(), "a"))});
    if ((delayed || wrongFirst) && kind == RelaxationKind::StoragePaddingOrNarrowerVector)
      out.push_back({StringAttr::get(var.shapedType.getContext(), "b")});
  }
  LogicalResult verifyCandidate(const LayoutVar &, Attribute value, Location) const override {
    return success(!rejectGenerated &&
                   (isa<StringAttr>(value) || value == proposedEncoding));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {
    if (failCost) return failure();
    CostVector result;
    result.codeSize = 3;
    return result;
  }
};
// A bounded-domain adapter around real SM90 instruction proofs: withhold only
// ordinary Tensor defaults to exercise a raw-empty component during collection.
class EmptyTensorDefaults : public LayoutTarget {
  std::unique_ptr<LayoutTarget> sm90 = createSM90LayoutTarget();
public:
  void enumerateCandidates(const LayoutVar &v, SmallVectorImpl<LayoutCandidate> &out) const override {
    if (v.kind != LayoutKind::Distributed || v.operationExecution)
      sm90->enumerateCandidates(v,out);
  }
  LogicalResult verifyCandidate(const LayoutVar &v,Attribute a,Location l) const override {
    return sm90->verifyCandidate(v,a,l);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
    return sm90->evaluate(a);
  }
  LogicalResult prepareInstructionCandidates(LayoutConstraintGraph &g) const override {
    return sm90->prepareInstructionCandidates(g);
  }
  FailureOr<Attribute> buildInstructionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c,ArrayRef<Attribute> a) const override {
    return sm90->buildInstructionContract(g,c,a);
  }
  LayoutProof verifyInstructionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c,ArrayRef<Attribute> a,Attribute b) const override {
    return sm90->verifyInstructionContract(g,c,a,b);
  }
};
class LayoutRelaxationTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  LayoutConstraintGraph graph;
  FiniteRelaxationTarget target;
  SmallVector<RelaxationRecord> records;
  LayoutSolveFailure why;
  LayoutVarID add(StringRef name, bool seeded = false) {
    auto id = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({32}, b.getF32Type()), name);
    if (seeded) graph.getVariable(id).candidates.push_back({b.getStringAttr("a")});
    return id;
  }
};
TEST_F(LayoutRelaxationTest, EmptyNeighbourCannotHideJointResidualConflict) {
  auto empty = add("empty"), x = add("x"), y = add("y"), z = add("z");
  for (auto value : {"p", "q"}) graph.getVariable(x).candidates.push_back({b.getStringAttr(value)});
  for (auto value : {"q", "r"}) graph.getVariable(y).candidates.push_back({b.getStringAttr(value)});
  for (auto value : {"p", "r"}) graph.getVariable(z).candidates.push_back({b.getStringAttr(value)});
  for (auto pair : {std::make_pair(empty, x), std::make_pair(x, y),
                    std::make_pair(y, z), std::make_pair(z, x)})
    graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
        {pair.first, pair.second}, nullptr, "same", "same");
  target.expectEmptyProposals = false;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::HardConflict);
  EXPECT_TRUE(records.empty());
  EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, OrdinarySuccessNeverRetries) {
  add("v", true);
  ASSERT_TRUE(succeeded(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_TRUE(records.empty());
  EXPECT_TRUE(target.calls.empty());
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::None);
}
TEST_F(LayoutRelaxationTest, ResidualProofDoesNotPriceAnIncompleteConversion) {
  context.allowUnregisteredDialects();
  context.loadDialect<FriskDialect>();
  auto empty = add("empty"), source = add("source", true);
  auto sm90 = createSM90LayoutTarget();
  SmallVector<LayoutCandidate> physical;
  sm90->enumerateCandidates(graph.getVariable(source), physical);
  ASSERT_FALSE(physical.empty());
  target.proposedEncoding = physical.front().value;
  graph.getVariable(source).candidates = {physical.front()};
  Block block;
  auto value = block.addArgument(graph.getVariable(source).shapedType, b.getUnknownLoc());
  OperationState state(b.getUnknownLoc(), "test.use");
  state.addOperands(value);
  auto *use = Operation::create(state);
  block.push_back(use);
  graph.getVariable(source).value = value;
  auto cid = graph.addConstraint(ConstraintKind::Convertible, ConstraintStrength::Hard,
      {source, empty}, use, "convert", "missing destination until retry");
  graph.getConstraints()[cid].use = &use->getOpOperand(0);
  graph.getConstraints()[cid].stableUseKey = "use/0";
  target.expectEmptyProposals = false;
  auto original = graph;
  EXPECT_TRUE(failed(solveLayoutGraph(original, target, {}, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::EmptyDomain);
  ASSERT_TRUE(succeeded(attemptControlledRelaxation(graph, target, {}, records, &why)));
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records.front().outcome, RelaxationOutcome::Selected);
}
TEST_F(LayoutRelaxationTest, EmptyDomainExpandsAndRecordsNoInventedBaseline) {
  add("v");
  auto solution = attemptControlledRelaxation(graph, target, {}, records, &why);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(records.size(), 1u);
  EXPECT_EQ(records[0].kind, RelaxationKind::Replication);
  EXPECT_EQ(records[0].addedCandidates, 1u);
  EXPECT_EQ(records[0].outcome, RelaxationOutcome::Selected);
  ASSERT_TRUE(records[0].selectedCost);
  EXPECT_EQ(records[0].selectedCost->cost.codeSize, 3u);
  EXPECT_FALSE(records[0].baselineCost);
  EXPECT_EQ(solution->assignments.lookup(0), b.getStringAttr("a"));
}
TEST_F(LayoutRelaxationTest, OrderedIndependentSnapshots) {
  add("v"); target.delayed = true;
  ASSERT_TRUE(succeeded(attemptControlledRelaxation(graph, target, {}, records, &why)));
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].outcome, RelaxationOutcome::NoNewCandidates);
  EXPECT_EQ(records[1].kind, RelaxationKind::StoragePaddingOrNarrowerVector);
  EXPECT_EQ(graph.getVariable(0).candidates.front().value, b.getStringAttr("b"));
}
TEST_F(LayoutRelaxationTest, RejectedCandidateNeverLeaksIntoNextStage) {
  auto fixed = add("fixed", true);
  graph.getVariable(fixed).candidates.front().value = b.getStringAttr("b");
  auto empty = add("empty");
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {fixed}, nullptr, "fixed", "hard", b.getStringAttr("b"));
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {fixed, empty}, nullptr, "same", "same");
  target.wrongFirst = true;
  auto solution = attemptControlledRelaxation(graph, target, {}, records, &why);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(records.size(), 2u);
  EXPECT_EQ(records[0].outcome, RelaxationOutcome::Rejected);
  EXPECT_EQ(records[1].outcome, RelaxationOutcome::Selected);
  for (const auto &var : graph.getVariables()) {
    ASSERT_EQ(var.candidates.size(), 1u);
    EXPECT_EQ(var.candidates.front().value, b.getStringAttr("b"));
  }
}
TEST_F(LayoutRelaxationTest, HardSeedConflictDoesNotRetryOrMutate) {
  add("v", true);
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {0}, nullptr, "explicit", "mandatory encoding", b.getStringAttr("b"));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::HardConflict);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
  ASSERT_EQ(graph.getVariable(0).candidates.size(), 1u);
  EXPECT_EQ(graph.getVariable(0).candidates.front().value, b.getStringAttr("a"));
}
TEST_F(LayoutRelaxationTest, EmptyEarlierComponentDoesNotHideHardSeedConflict) {
  add("a"); auto fixed = add("z", true);
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {fixed}, nullptr, "explicit", "mandatory encoding", b.getStringAttr("b"));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::HardConflict);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, EmptyComponentCannotHideBudgetOrCostFailure) {
  add("a_empty"); auto x = add("x", true); auto y = add("y", true);
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {x,y}, nullptr, "same", "same");
  target.expectEmptyProposals = false;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  SolverOptions options; options.maxExpandedStates = 1;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,options,records,&why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::SearchBudget);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
  target.calls.clear(); target.failCost = true;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,{},records,&why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::CostFailure);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, FixedRelationConflictInAnEmptyComponentNeverRetries) {
  auto a=add("a"); auto x=add("x",true); auto y=add("y",true);
  graph.getVariable(y).candidates.front().value=b.getStringAttr("b");
  for (auto id : {x,y})
    graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
        {id}, nullptr,"fixed","fixed",graph.getVariable(id).candidates.front().value);
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {a,x}, nullptr,"same","same");
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {x,y}, nullptr,"same","same");
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,{},records,&why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::HardConflict);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, UnsupportedBinaryDecoderIsNotHardConflict) {
  auto x=add("x",true); auto y=add("y",true);
  auto id=graph.addConstraint(ConstraintKind::TransformLayout, ConstraintStrength::Hard,
      {x,y}, nullptr,"transform","unsupported encoding decoder");
  graph.getConstraints()[id].coordinateTransform = AffineMapAttr::get(
      AffineMap::getPermutationMap(ArrayRef<unsigned>{0}, &context));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,{},records,&why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::Unsupported);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
  auto empty=add("a_empty"); target.expectEmptyProposals = false;
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {empty,x}, nullptr,"same","connect empty and unknown subdomains");
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,{},records,&why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::Unsupported);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, BudgetAndCostFailuresDoNotRetry) {
  add("a", true); add("b", true);
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {0, 1}, nullptr, "same", "same");
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  SolverOptions options; options.maxExpandedStates = 1;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, options, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::SearchBudget);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
  target.failCost = true;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::CostFailure);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, SM90BoundedReplicationIsARealVerifiedCandidate) {
  context.loadDialect<FriskDialect>();
  add("v");
  graph.getVariable(0).requiredThreads = 32;
  auto sm90 = createSM90LayoutTarget();
  auto solution = attemptControlledRelaxation(graph, *sm90, {}, records, &why);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_EQ(records.size(), 1u);
  auto encoding = cast<DistributedEncodingAttr>(solution->assignments.lookup(0));
  EXPECT_EQ(encoding.getTopology()[1], 32);
  EXPECT_GT(encoding.getReplication().getInt(), 1);
  EXPECT_LE(encoding.getReplication().getInt(), 4);
}
TEST_F(LayoutRelaxationTest, UnknownOwnershipProofDoesNotRetry) {
  context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%p: memref<32xf32,3>) {
      %v = frisk.layout_view %p : memref<32xf32,3> -> memref<32xf32,3>
      frisk.fill %v {value=0.0:f32} : memref<32xf32,3>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  add("v", true);
  module->walk([&](FillOp op) { graph.getVariable(0).anchor = op; });
  graph.getVariable(0).operationExecution = OperationExecutionBinding{};
  graph.getVariable(0).requiredThreads = 32;
  graph.addConstraint(ConstraintKind::Ownership, ConstraintStrength::Hard,
      {0}, nullptr, "ownership", "unknown encoding decoder");
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::Unsupported);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
  add("a_empty"); target.expectEmptyProposals = false;
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::Unsupported);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, CollectorRetainsUnknownCandidateRejectionEvidence) {
  context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect>();
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @f() -> tensor<32xf32> {
      %c = arith.constant dense<1.0> : tensor<32xf32>
      return %c : tensor<32xf32>
    })mlir", &context);
  ASSERT_TRUE(module); target.rejectGenerated = true;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  auto collected = collectLayoutConstraints(*module, target, LayoutCollectionMode::InitializeForRelaxation);
  ASSERT_TRUE(succeeded(collected));
  EXPECT_TRUE(failed(attemptControlledRelaxation(*collected, target, {}, records, &why)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::Unsupported);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
TEST_F(LayoutRelaxationTest, RawEmptyDoesNotSkipUnrelatedCopyTuples) {
  context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect, memref::MemRefDialect>();
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @f(%s:memref<8x16xf32,1>, %d:memref<8x16xf32,3>) -> tensor<32xf32> {
      %sv=frisk.layout_view %s : memref<8x16xf32,1> -> memref<8x16xf32,1>
      %dv=frisk.layout_view %d : memref<8x16xf32,3> -> memref<8x16xf32,3>
      "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:8,16>,dstExtents=array<i64:8,16>}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<8x16xf32,1>,memref<8x16xf32,3>)->()
      %c = arith.constant dense<1.0> : tensor<32xf32>
      return %c : tensor<32xf32>
    })mlir", &context);
  ASSERT_TRUE(module); EmptyTensorDefaults bounded;
  auto collected=collectLayoutConstraints(*module,bounded,LayoutCollectionMode::InitializeForRelaxation);
  ASSERT_TRUE(succeeded(collected));
  unsigned contracts=0, empty=0;
  for (const auto &c:collected->getConstraints()) if(c.instruction) {
    ++contracts; EXPECT_FALSE(c.instruction->tuples.empty());
  }
  for (const auto &v:collected->getVariables()) empty += v.candidates.empty();
  EXPECT_EQ(contracts,1u); EXPECT_GT(empty,0u);
}
TEST_F(LayoutRelaxationTest, MutatedFinalizedGraphIsRejectedBeforePropagation) {
  auto x=add("x",true); auto y=add("y",true);
  graph.addConstraint(ConstraintKind::SameLayout,ConstraintStrength::Hard,
      {x,y},nullptr,"same","same");
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  graph.getConstraints()[0].vars[0]=99;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph,target,{},records,&why)));
  EXPECT_EQ(why.kind,LayoutSolveFailureKind::InvalidInput);
  EXPECT_TRUE(records.empty()); EXPECT_TRUE(target.calls.empty());
}
} // namespace
