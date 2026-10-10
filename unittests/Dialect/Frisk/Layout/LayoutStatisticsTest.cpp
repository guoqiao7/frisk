#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class StatisticsTarget : public LayoutTarget {
public:
  void enumerateCandidates(const LayoutVar &, SmallVectorImpl<LayoutCandidate> &) const override {}
  LogicalResult verifyCandidate(const LayoutVar &, Attribute a, Location) const override {
    return success(isa<StringAttr>(a));
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &) const override { return CostVector{}; }
};
TEST(LayoutStatisticsTest, DeduplicatedDomainsAndActualFrontier) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "stable-var");
  for (auto name : {"a", "b", "b"})
    graph.getVariable(id).candidates.push_back({builder.getStringAttr(name), kInvalidProvenanceID, 0});
  StatisticsTarget target;
  LayoutAttemptStatistics attempt;
  auto solution = solveLayoutGraph(graph, target, {}, nullptr, &attempt);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_EQ(attempt.statistics.variables, 1u);
  EXPECT_EQ(attempt.statistics.preparedCandidates, 2u);
  EXPECT_EQ(attempt.statistics.prunedCandidates, 0u);
  EXPECT_EQ(attempt.statistics.peakVariableDomain, 2u);
  EXPECT_EQ(attempt.statistics.peakComponentCombinations, 2u);
  EXPECT_EQ(attempt.statistics.peakFrontierStates, 2u);
  EXPECT_EQ(attempt.mrvProbes, 2u);
  EXPECT_EQ(attempt.expandedNodes, 2u);
}
TEST(LayoutStatisticsTest, FailureHasStableIdentityAndNoSuccessCost) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "missing-domain");
  StatisticsTarget target;
  LayoutAttemptStatistics attempt;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(solveLayoutGraph(graph, target, {}, nullptr, &attempt)));
  EXPECT_EQ(attempt.failureKind, "empty-domain");
  EXPECT_EQ(attempt.failureVariable, "missing-domain");
  EXPECT_FALSE(attempt.succeeded);
}
TEST(LayoutStatisticsTest, StandaloneCollectorIsPerInvocationAndSuccessClearsFailure) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  LayoutStatisticsReport report;
  for (unsigned i = 0; i < 2; ++i) {
    auto module = parseSourceString<ModuleOp>("module {}", &context);
    PassManager pm(&context);
    pm.addPass(createFriskInferLayoutsPass({}, false, &report));
    ASSERT_TRUE(succeeded(pm.run(*module)));
    EXPECT_EQ(report.attempts.size(), 1u);
    EXPECT_TRUE(report.failureKind.empty());
  }
}
TEST(LayoutStatisticsTest, StandaloneWriteFailureUpdatesExternalCollector) {
  MLIRContext context;
  context.loadDialect<FriskDialect>();
  LayoutStatisticsReport report;
  auto module = parseSourceString<ModuleOp>("module {}", &context);
  PassManager pm(&context);
  auto pass = createFriskInferLayoutsPass({}, false, &report);
  ASSERT_TRUE(succeeded(pass->initializeOptions("stats-file=/nonexistent-frisk-statistics-directory/stats.json",
      [](const Twine &) { return failure(); })));
  pm.addPass(std::move(pass));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(pm.run(*module)));
  EXPECT_EQ(report.status, "failed");
  EXPECT_EQ(report.failureKind, "statistics-write");
}
TEST(LayoutStatisticsTest, SharedUseEdgesChargeOnePhysicalConversion) {
  LayoutConstraintGraph graph;
  LayoutSolution solution;
  LayoutConversionEdge a;
  a.resolution = EdgeResolutionKind::Convert;
  a.constraint = 2; a.sharedLeader = 2; a.bytes = 128;
  solution.conversions.push_back(a);
  a.constraint = 3;
  solution.conversions.push_back(a);
  a.resolution = EdgeResolutionKind::Rematerialize;
  a.constraint = 4; a.sharedLeader = 4;
  solution.conversions.push_back(a);
  a.constraint = 5;
  solution.conversions.push_back(a);
  LayoutAttemptStatistics attempt;
  finishLayoutStatistics(graph, {}, &solution, attempt);
  EXPECT_EQ(attempt.statistics.conversionUseEdges, 2u);
  EXPECT_EQ(attempt.statistics.uniqueConversions, 1u);
  EXPECT_EQ(attempt.statistics.conversionBytes, 128u);
  EXPECT_EQ(attempt.statistics.rematerializationUseEdges, 2u);
  EXPECT_EQ(attempt.statistics.uniqueRematerializations, 1u);
}
TEST(LayoutStatisticsTest, FailedStatisticsWriteLeavesPipelineIRUnmodified) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  auto module = parseSourceString<ModuleOp>(R"(
    func.func @f(%x:tensor<4xf32>) -> tensor<4xf32> {
      %y = arith.addf %x, %x : tensor<4xf32>
      return %y : tensor<4xf32>
    })", &context);
  ASSERT_TRUE(module);
  auto print = [&] { std::string out; llvm::raw_string_ostream os(out); module->print(os); return out; };
  auto before = print();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  PassManager pm(&context);
  LayoutStatisticsReport report;
  buildFriskLayoutPipeline(pm, {}, "/nonexistent-frisk-statistics-directory/stats.json", &report);
  EXPECT_TRUE(failed(pm.run(*module)));
  EXPECT_EQ(before, print());
  EXPECT_EQ(report.status, "failed");
  EXPECT_EQ(report.failureKind, "statistics-write");
  ASSERT_FALSE(report.attempts.empty());
  EXPECT_TRUE(report.attempts.back().succeeded);
  EXPECT_GT(report.timings["materialization_ns"], 0u);
  EXPECT_GT(report.timings["optimization_ns"], 0u);
}
TEST(LayoutStatisticsTest, StableSerializationAndFailureDoesNotInventCost) {
  LayoutStatisticsReport report;
  report.attempts.emplace_back();
  report.attempts.back().failureKind = "empty-domain";
  report.attempts.back().failureVariable = "v0";
  auto first = llvm::formatv("{0:2}", llvm::json::Value(report.toJSON())).str();
  auto second = llvm::formatv("{0:2}", llvm::json::Value(report.toJSON())).str();
  EXPECT_EQ(first, second);
  EXPECT_EQ(first.find("conversionBytes"), std::string::npos);
  EXPECT_NE(first.find("empty-domain"), std::string::npos);
}
TEST(LayoutStatisticsTest, ActualCoordinatorConflictHasStableFinalIdentity) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  auto variable = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "mandatory-variable");
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {variable}, nullptr, "mandatory", "required", builder.getStringAttr("a"));
  LayoutStatisticsReport report;
  StatisticsTarget target;
  SmallVector<RelaxationRecord> records;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, nullptr, &report)));
  ASSERT_EQ(report.attempts.size(), 1u);
  // Mandatory seed conflicts are already discovered by the hardened solver.
  EXPECT_EQ(report.attempts.back().failureKind, "hard-conflict");
  EXPECT_EQ(report.failureKind, "hard-conflict");
  EXPECT_EQ(report.failureVariable, "mandatory-variable");
  EXPECT_FALSE(report.failureConstraint.empty());
  auto json = report.toJSON();
  EXPECT_EQ(*json.getObject("failure")->getString("kind"), "hard-conflict");
}
TEST(LayoutStatisticsTest, FinalFailureSerializationDoesNotOverwriteAttemptHistory) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  auto variable = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "mandatory-variable");
  auto constraint = graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {variable}, nullptr, "mandatory", "required", builder.getStringAttr("a"));
  LayoutStatisticsReport report;
  report.attempts.emplace_back();
  report.attempts.back().failureKind = "empty-domain";
  recordLayoutFailure(graph, {LayoutSolveFailureKind::HardConflict, variable, constraint,
                             "mandatory endpoint conflict"}, report);
  EXPECT_EQ(report.attempts.back().failureKind, "empty-domain");
  EXPECT_EQ(report.failureKind, "hard-conflict");
  EXPECT_EQ(report.failureVariable, "mandatory-variable");
  EXPECT_FALSE(report.failureConstraint.empty());
}
TEST(LayoutStatisticsTest, FinalFailureUsesFinalizedIdentitiesWithoutMutatingCaller) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  auto z = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "z-variable");
  auto a = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "a-variable");
  graph.getVariable(z).candidates.push_back({builder.getStringAttr("a")});
  graph.getVariable(a).candidates.push_back({builder.getStringAttr("a")});
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {z}, nullptr, "z-required", "z", builder.getStringAttr("a"));
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {a}, nullptr, "a-required", "a", builder.getStringAttr("b"));
  graph.getConstraints()[0].stableUseKey = "z-use";
  graph.getConstraints()[1].stableUseKey = "a-use";
  StatisticsTarget target;
  LayoutStatisticsReport report;
  SmallVector<RelaxationRecord> records;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {}, records, nullptr, &report)));
  ASSERT_EQ(report.attempts.size(), 1u);
  EXPECT_EQ(report.failureVariable, "a-variable");
  EXPECT_EQ(report.failureVariable, report.attempts.back().failureVariable);
  EXPECT_EQ(report.failureConstraint, report.attempts.back().failureConstraint);
  EXPECT_NE(report.failureConstraint.find("a-use"), std::string::npos);
  EXPECT_EQ(graph.getVariable(0).stableName, "z-variable");
  EXPECT_FALSE(graph.isFinalized());
}
TEST(LayoutStatisticsTest, IndependentNonemptyRunsHaveIdenticalNonTimingJSON) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  std::string first;
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    auto module = parseSourceString<ModuleOp>(R"(
      func.func @f(%x:tensor<4xf32>) -> tensor<4xf32> {
        %y = arith.addf %x, %x : tensor<4xf32>
        return %y : tensor<4xf32>
      })", &context);
    LayoutStatisticsReport report;
    PassManager pm(&context);
    buildFriskLayoutPipeline(pm, {}, {}, &report);
    ASSERT_TRUE(succeeded(pm.run(*module)));
    ASSERT_FALSE(report.attempts.empty());
    EXPECT_GT(report.attempts.back().statistics.variables, 0u);
    report.timings.clear();
    for (auto &attempt : report.attempts) {
      attempt.statistics.solveTime = std::chrono::nanoseconds(0);
      attempt.propagationTime = std::chrono::nanoseconds(0);
      attempt.verificationTime = std::chrono::nanoseconds(0);
    }
    auto text = llvm::formatv("{0:2}", llvm::json::Value(report.toJSON())).str();
    if (!iteration) first = text;
    else EXPECT_EQ(first, text);
  }
}
TEST(LayoutStatisticsTest, RetryProposalsAndAttemptsRemainSeparate) {
  class RetryingTarget : public StatisticsTarget {
    void enumerateRelaxedCandidates(const LayoutConstraintGraph &, const LayoutVar &var,
        RelaxationKind kind, SmallVectorImpl<LayoutCandidate> &out) const override {
      if (kind == RelaxationKind::Replication) {
        auto value = StringAttr::get(var.shapedType.getContext(), "a");
        out.push_back({value}); out.push_back({value});
      }
    }
  } target;
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  graph.addVariable(LayoutKind::Distributed, RankedTensorType::get({4}, builder.getF32Type()), "retry");
  LayoutStatisticsReport report;
  SmallVector<RelaxationRecord> records;
  ASSERT_TRUE(succeeded(attemptControlledRelaxation(graph, target, {}, records, nullptr, &report)));
  ASSERT_EQ(report.attempts.size(), 2u);
  EXPECT_EQ(report.attempts[0].kind, "initial");
  EXPECT_EQ(report.attempts[1].kind, "replication");
  EXPECT_EQ(report.attempts[0].statistics.preparedCandidates, 0u);
  EXPECT_EQ(report.attempts[1].statistics.preparedCandidates, 1u);
  ASSERT_EQ(report.relaxations.size(), 1u);
  EXPECT_EQ(*report.relaxations[0].getAsObject()->getInteger("generatedProposals"), 2);
  EXPECT_EQ(*report.relaxations[0].getAsObject()->getInteger("addedCandidates"), 1);
}
}
