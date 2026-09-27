#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "gtest/gtest.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
class FiniteInstructionTarget : public LayoutTarget {
public:
  mutable unsigned builds = 0, proofs = 0;
  void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &, Attribute candidate, Location) const override {
    return success(isa<StringAttr>(candidate));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override { return CostVector{}; }
  FailureOr<Attribute> buildInstructionContract(const LayoutConstraintGraph &,
      const LayoutConstraint &c, ArrayRef<Attribute>) const override {
    ++builds;
    return Attribute(StringAttr::get(c.instruction->source->getContext(), "plan"));
  }
  LayoutProof verifyInstructionContract(const LayoutConstraintGraph &,
      const LayoutConstraint &, ArrayRef<Attribute> encodings, Attribute binding) const override {
    ++proofs;
    bool valid = encodings.size() == 4 && encodings[2] == encodings[3] &&
        binding == StringAttr::get(binding.getContext(), "plan");
    return {valid ? ProofStatus::Proven : ProofStatus::Disproven, {}, "finite test instruction"};
  }
};

class InstructionConstraintTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  OwningOpRef<ModuleOp> source{ModuleOp::create(b.getUnknownLoc())};
  Attribute x = b.getStringAttr("x"), y = b.getStringAttr("y");
  Attribute plan = b.getStringAttr("plan");
  LayoutConstraintGraph graph;
  SmallVector<LayoutVarID> ids;

  void makeGraph() {
    for (StringRef name : {"z-A", "y-B", "x-init", "w-result"}) {
      auto id = graph.addVariable(LayoutKind::Distributed,
          RankedTensorType::get({8, 8}, b.getF32Type()), name);
      graph.getVariable(id).candidates = {{x, kInvalidProvenanceID, 0},
                                        {y, kInvalidProvenanceID, 1}};
      ids.push_back(id);
    }
  }
  LayoutConstraint &contract(ArrayRef<LayoutVarID> endpoints) {
    auto id = graph.addConstraint(ConstraintKind::InstructionContract,
        ConstraintStrength::Hard, endpoints, *source, "test-instruction", "joint tuple");
    auto &c = graph.getConstraints()[id];
    c.instruction = InstructionLayoutContract{*source, {}, {}};
    return c;
  }
};

TEST_F(InstructionConstraintTest, FailedGenerationIsUnknownNotCounterexample) {
  class NoPlanTarget : public FiniteInstructionTarget {
    FailureOr<Attribute> buildInstructionContract(const LayoutConstraintGraph &,
        const LayoutConstraint &, ArrayRef<Attribute>) const override {
      return failure();
    }
  } target;
  makeGraph();
  contract(ids);
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(message);
    diagnostic.print(stream);
    return success();
  });
  EXPECT_TRUE(failed(prepareInstructionTuples(graph, target, b.getUnknownLoc())));
  EXPECT_NE(message.find("unknown"), std::string::npos) << message;
  EXPECT_EQ(message.find("counterexample"), std::string::npos) << message;
}

TEST_F(InstructionConstraintTest, StrictFiltersWithoutSingletonAndRemapsRoles) {
  makeGraph();
  contract(ids).instruction->tuples = {{{x, x, x, x}, plan}, {{x, y, y, y}, plan}};
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  auto &c = graph.getConstraints().front();
  EXPECT_EQ(graph.getVariable(c.vars[0]).stableName, "z-A");
  ASSERT_TRUE(succeeded(propagateStrict(graph)));
  const auto &a = graph.getVariable(c.vars[0]);
  ASSERT_EQ(a.candidates.size(), 1u);
  EXPECT_EQ(a.candidates[0].value, x);
  EXPECT_EQ(graph.getVariable(c.vars[1]).candidates.size(), 2u);
  const auto &stats = graph.getPropagationStatistics().strict;
  EXPECT_EQ(stats.deletedCandidates, 1u);
  EXPECT_TRUE(stats.hasValidBounds());
}

TEST_F(InstructionConstraintTest, PartialAssignmentMustExtendSameTuple) {
  makeGraph();
  auto &c = contract(ids);
  // Odd parity: every pair permits x/x, but x/x/x has no complete support.
  c.instruction->tuples = {{{x, x, y, y}, plan}, {{x, y, x, x}, plan},
                          {{y, x, x, x}, plan}, {{y, y, y, y}, plan}};
  DenseMap<LayoutVarID, Attribute> trial;
  trial[ids[0]] = x;
  trial[ids[1]] = x;
  EXPECT_NE(findInstructionSupport(graph, c, trial), nullptr);
  trial[ids[2]] = x;
  EXPECT_EQ(findInstructionSupport(graph, c, trial), nullptr);
  trial[ids[2]] = y;
  EXPECT_NE(findInstructionSupport(graph, c, trial), nullptr);
}

TEST_F(InstructionConstraintTest, RepeatedEndpointAndDuplicateTupleAreRejected) {
  makeGraph();
  SmallVector<LayoutVarID> repeated{ids[0], ids[1], ids[2], ids[2]};
  auto &c = contract(repeated);
  c.instruction->tuples = {{{x, x, x, y}, plan}};
  ScopedDiagnosticHandler silence(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(graph.verifyInvariants(b.getUnknownLoc())));
  DenseMap<LayoutVarID, Attribute> emptyAssignment;
  EXPECT_EQ(findInstructionSupport(graph, c, emptyAssignment), nullptr);
  c.instruction->tuples = {{{x, x, x, x}, plan}};
  EXPECT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  auto &sorted = graph.getConstraints().front();
  sorted.instruction->tuples.push_back(sorted.instruction->tuples.front());
  EXPECT_TRUE(failed(graph.verifyInvariants(b.getUnknownLoc())));
}

TEST_F(InstructionConstraintTest, EnumerationAndSolverChooseOneWholeBinding) {
  makeGraph();
  contract(ids);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  FiniteInstructionTarget target;
  ASSERT_TRUE(succeeded(prepareInstructionTuples(graph, target, b.getUnknownLoc())));
  EXPECT_EQ(target.builds, 16u);
  EXPECT_EQ(graph.getConstraints().front().instruction->tuples.size(), 8u);
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(graph)));
  auto solution = solveLayoutGraph(graph, target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_EQ(solution->instructionBindings.lookup(0), plan);
  ASSERT_TRUE(succeeded(verifySolvedLayoutGraph(graph, *solution, target, b.getUnknownLoc())));
  EXPECT_EQ(target.builds, 16u); // Verification must not search for a replacement plan.
  graph.getConstraints().front().instruction->tuples.clear();
  EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(graph, *solution, target, b.getUnknownLoc())));
  ScopedDiagnosticHandler silence(&context, [](Diagnostic &) { return success(); });
  solution->instructionBindings[0] = b.getStringAttr("tampered");
  EXPECT_TRUE(failed(verifySolvedLayoutGraph(graph, *solution, target, b.getUnknownLoc())));
}

TEST_F(InstructionConstraintTest, ExplicitBindingOnlyVerifiesAndRepeatedRolesEnumerateOnce) {
  makeGraph();
  auto &c = contract({ids[0], ids[1], ids[2], ids[2]});
  c.instruction->binding = plan;
  FiniteInstructionTarget target;
  ASSERT_TRUE(succeeded(prepareInstructionTuples(graph, target, b.getUnknownLoc())));
  EXPECT_EQ(target.builds, 0u);
  EXPECT_EQ(target.proofs, 8u);
  EXPECT_EQ(graph.getCandidatePreparationStatistics().instructionCombinations, 8u);
  ASSERT_TRUE(succeeded(graph.finalize(b.getUnknownLoc())));
  EXPECT_TRUE(succeeded(propagateStrict(graph)));
}

TEST_F(InstructionConstraintTest, FourCandidateDomainsHaveExactly256Combinations) {
  makeGraph();
  for (auto &var : graph.getVariables())
    for (StringRef extra : {"z", "w"})
      var.candidates.push_back({b.getStringAttr(extra), kInvalidProvenanceID, var.candidates.size()});
  contract(ids);
  FiniteInstructionTarget target;
  ASSERT_TRUE(succeeded(prepareInstructionTuples(graph, target, b.getUnknownLoc())));
  EXPECT_EQ(target.builds, 256u);
  EXPECT_EQ(graph.getConstraints().front().instruction->tuples.size(), 64u);
  graph.getVariable(ids[0]).candidates.push_back({b.getStringAttr("fifth"), kInvalidProvenanceID, 4});
  ScopedDiagnosticHandler silence(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(prepareInstructionTuples(graph, target, b.getUnknownLoc())));
  EXPECT_EQ(target.builds, 256u); // Reject before enumerating an oversized domain.
}

TEST_F(InstructionConstraintTest, ExplicitUnprovableBindingRetainsUnknownDiagnostic) {
  class UnknownInstructionTarget final : public FiniteInstructionTarget {
    LayoutProof verifyInstructionContract(const LayoutConstraintGraph &,
        const LayoutConstraint &, ArrayRef<Attribute>, Attribute) const override {
      return {ProofStatus::Unknown, {7, 9}, "test descriptor proof budget"};
    }
  } target;
  makeGraph();
  contract(ids).instruction->binding = plan;
  std::string diagnostic;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &d) {
    llvm::raw_string_ostream stream(diagnostic);
    d.print(stream);
    return success();
  });
  EXPECT_TRUE(failed(prepareInstructionTuples(graph, target, b.getUnknownLoc())));
  EXPECT_NE(diagnostic.find("unknown proof"), std::string::npos);
  EXPECT_NE(diagnostic.find("test descriptor proof budget"), std::string::npos);
  EXPECT_NE(diagnostic.find("7, 9"), std::string::npos);
  EXPECT_EQ(target.builds, 0u);
}

TEST_F(InstructionConstraintTest, UnprovenInstructionCannotSurvivePropagation) {
  MLIRContext context;
  Builder b(&context);
  LayoutConstraintGraph graph;
  SmallVector<LayoutVarID> ids;
  for (StringRef name : {"a", "b", "init", "result"}) {
    auto id = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({8, 8}, b.getF32Type()), name);
    graph.getVariable(id).candidates = {
        {b.getStringAttr("x"), kInvalidProvenanceID, 0},
        {b.getStringAttr("y"), kInvalidProvenanceID, 1}};
    ids.push_back(id);
  }
  graph.addConstraint(ConstraintKind::InstructionContract,
      ConstraintStrength::Hard, ids, nullptr, "unknown-instruction", "no proof");
  ScopedDiagnosticHandler silence(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(propagateStrict(graph)));
}
} // namespace
