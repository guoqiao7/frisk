#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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
TEST(LayoutStatisticsTest, JSONOwnsModelIdentityBeforeReportMutation) {
  LayoutStatisticsReport report;
  const std::string expected(64, 'A');
  report.costModelID = expected;
  auto json = report.toJSON();
  // Mutate the same live allocation: a borrowed StringRef fails deterministically
  // without relying on allocator reuse after the report has been destroyed.
  for (char &c : report.costModelID) c = 'B';
  EXPECT_EQ(json.getString("cost_model_id"), expected);
}
TEST(LayoutStatisticsTest, DetachedJSONSurvivesReportDestruction) {
  auto json = [] {
    LayoutStatisticsReport report;
    report.costModelID = std::string(64, 'Z');
    return report.toJSON();
  }();
  auto text = llvm::formatv("{0}", llvm::json::Value(std::move(json))).str();
  auto parsed = llvm::json::parse(text);
  ASSERT_TRUE(bool(parsed));
  EXPECT_EQ(parsed->getAsObject()->getString("cost_model_id"), std::string(64, 'Z'));
}
TEST(LayoutStatisticsTest, UnsolvedReportsNormalizeCompleteIdentity) {
  LayoutStatisticsReport report;
  EXPECT_EQ(report.toJSON().getString("cost_model_id"), "unversioned-target");
  for (const std::string id : {std::string(), std::string(65, 'a'),
         std::string("model with space"), std::string("模型"), std::string("id\0bad", 6)}) {
    report.costModelID = id;
    EXPECT_EQ(report.toJSON().getString("cost_model_id"), "unversioned-target");
  }
  report.costModelID = std::string(64, 'Z');
  EXPECT_EQ(report.toJSON().getString("cost_model_id"), report.costModelID);
}
TEST(LayoutStatisticsTest, CustomTargetIdentitySurvivesPublicSolverBoundaries) {
  class NamedTarget : public StatisticsTarget {
  public:
    explicit NamedTarget(std::string id) : id(std::move(id)) {}
    StringRef costModelID() const override { return id; }
    LayoutBoundaryFrequencyMode boundaryFrequencyMode() const override { return LayoutBoundaryFrequencyMode::StaticTripCount; }
    FailureOr<CostVector> evaluate(const CandidateAssignment &) const override {
      return CostVector{1, 2, 3, 4, 5, 6, 7, 8, 9};
    }
    std::string id;
  };
  MLIRContext context;
  Builder builder(&context);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (std::string id : {std::string("custom.A_2-v1"), std::string(65, 'a'), std::string("bad id")}) {
    NamedTarget target(id);
    std::string expected = id == "custom.A_2-v1" ? id : "unversioned-target";
    for (bool coordinator : {false, true}) {
      for (bool invalid : {false, true}) {
        for (auto limits : {LayoutTraceOptions{"unused.json"}, LayoutTraceOptions{"unused.json", 1, 4096}}) {
          LayoutConstraintGraph graph;
          auto var = graph.addVariable(LayoutKind::Distributed,
              RankedTensorType::get({4}, builder.getF32Type()), "v");
          graph.getVariable(var).candidates.push_back({builder.getStringAttr("a")});
          if (invalid) graph.getVariable(var).id = 42;
          LayoutTrace trace(limits);
          LayoutStatisticsReport report;
          report.costModelID = "stale-model";
          SmallVector<RelaxationRecord, 0> records;
          auto result = coordinator
              ? attemptControlledRelaxation(graph, target, {}, records, nullptr, &report, &trace)
              : solveLayoutGraph(graph, target, {}, nullptr, nullptr, &trace);
          EXPECT_EQ(succeeded(result), !invalid);
          auto parsed = llvm::json::parse(trace.serialize());
          ASSERT_TRUE(bool(parsed));
          auto *object = parsed->getAsObject();
          EXPECT_EQ(object->getString("cost_model_id"), expected);
          EXPECT_EQ(object->getString("boundaryFrequency"), "static-trip-count");
          EXPECT_EQ(trace.getSummary().costModelID, expected);
          if (coordinator) {
            EXPECT_EQ(report.costModelID, expected);
            EXPECT_EQ(report.toJSON().getString("boundaryFrequency"), "static-trip-count");
            EXPECT_EQ(report.toJSON().getString("cost_model_id"), expected);
          }
          if (!invalid && coordinator) {
            auto *final = object->getObject("finalCost");
            ASSERT_TRUE(final);
            auto *omitted = object->getObject("omittedCost")->getArray("cost");
            for (unsigned field = 0; field < 9; ++field) {
              uint64_t sum = *(*omitted)[field].getAsUINT64();
              for (auto &entry : *object->getArray("events")) {
                auto *event = entry.getAsObject();
                if (event->getString("event") == "cost-contribution")
                  sum += *(*event->getObject("estimate")->getArray("cost"))[field].getAsUINT64();
              }
              EXPECT_EQ(sum, *(*final->getArray("cost"))[field].getAsUINT64());
            }
          }
        }
      }
    }
  }
  StatisticsTarget legacy;
  EXPECT_EQ(legacy.costModelID(), "unversioned-target");
  EXPECT_EQ(legacy.boundaryFrequencyMode(), LayoutBoundaryFrequencyMode::Legacy);
}
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
    EXPECT_EQ(report.toJSON().getString("cost_model_id"), "sm90-layout-lex9-legacy-v2");
    EXPECT_EQ(report.toJSON().getInteger("schema_version"), 1);
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
TEST(LayoutStatisticsTest, CallerOwnedTraceMetadataPrecedesOptionValidation) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-trace-options-external", directory)));
  std::string output = (directory + "/must-not-be-written.json").str();
  SolverOptions settings{17, 5, 23, false, 29};
  const std::pair<std::string, std::string> scenarios[] = {
      {"trace-max-events=0", "trace-options"},
      {"trace-max-bytes=4095 trace-file=" + output, "trace-options"},
      {"profile-file=" + output + " stats-file=" + output, "profile-options"},
      {"profile-file=" + output + " trace-file=" + output, "profile-options"}};
  for (const auto &[arguments, failureKind] : scenarios) {
    for (bool stale : {false, true}) {
      SCOPED_TRACE(arguments + (stale ? " stale" : " fresh"));
      LayoutTrace trace({output, 1, 4096});
      if (stale) {
        LayoutTraceSummary summary;
        summary.target = "other-target";
        summary.costCoverage = "other-coverage";
        summary.costModelID = "other-model-v1";
        trace.setSummary(summary);
      }
      auto module = parseSourceString<ModuleOp>(R"(
        func.func @f(%x: tensor<4xf32>) -> tensor<4xf32> {
          %y = arith.addf %x, %x : tensor<4xf32>
          return %y : tensor<4xf32>
        })", &context);
      ASSERT_TRUE(module);
      auto print = [&] { std::string text; llvm::raw_string_ostream(text) << *module; return text; };
      auto before = print();
      LayoutStatisticsReport report;
      PassManager pm(&context);
      auto pass = createFriskInferLayoutsPass(settings, false, &report,
          LayoutStatisticsScope::Standalone, &trace);
      ASSERT_TRUE(succeeded(pass->initializeOptions(arguments,
          [](const Twine &) { return failure(); })));
      pm.addPass(std::move(pass));
      EXPECT_TRUE(failed(pm.run(*module)));
      EXPECT_EQ(print(), before);
      EXPECT_EQ(report.status, "failed");
      EXPECT_EQ(report.failureKind, failureKind);
      EXPECT_FALSE(report.failureReason.empty());
      EXPECT_TRUE(report.attempts.empty());
      EXPECT_EQ(report.toJSON().getString("cost_model_id"), "sm90-layout-lex9-legacy-v2");
      auto json = llvm::json::parse(trace.serialize());
      ASSERT_TRUE(bool(json));
      auto *object = json->getAsObject();
      EXPECT_EQ(object->getString("cost_model_id"), report.costModelID);
      EXPECT_EQ(object->getString("target"), "SM90");
      EXPECT_NE(object->getString("cost_coverage"), "other-coverage");
      auto *options = object->getObject("solver_options");
      ASSERT_TRUE(options);
      EXPECT_EQ(options->getInteger("exactCombinationLimit"), 17);
      EXPECT_EQ(options->getInteger("beamWidth"), 5);
      EXPECT_EQ(options->getInteger("maxExpandedStates"), 23);
      EXPECT_EQ(options->getBoolean("refineBeamSolution"), false);
      EXPECT_EQ(options->getInteger("maxRefinementProbes"), 29);
      EXPECT_FALSE(llvm::sys::fs::exists(output));
      std::error_code directoryError;
      llvm::sys::fs::directory_iterator files(directory, directoryError);
      EXPECT_FALSE(bool(directoryError));
      EXPECT_TRUE(files == llvm::sys::fs::directory_iterator());
    }
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
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
  SmallVector<RelaxationRecord, 0> records;
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
  SmallVector<RelaxationRecord, 0> records;
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
    FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override {
      CostVector cost;
      cost.codeSize = a.values.size() * 7;
      return cost;
    }
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
  auto seed = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "a-successful-component");
  graph.getVariable(seed).candidates.push_back({builder.getStringAttr("a")});
  graph.addVariable(LayoutKind::Distributed, RankedTensorType::get({4}, builder.getF32Type()), "z-retry");
  LayoutStatisticsReport report;
  LayoutTrace trace({"unused.json"});
  SmallVector<RelaxationRecord, 0> records;
  auto retryGraph = graph;
  auto solution = attemptControlledRelaxation(graph, target, {}, records, nullptr, &report, &trace);
  ASSERT_TRUE(succeeded(solution));
  auto parsed = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(parsed));
  EXPECT_EQ(parsed->getAsObject()->getInteger("costScope"), 2);
  EXPECT_EQ(parsed->getAsObject()->getInteger("attempts"), 2);
  EXPECT_EQ(parsed->getAsObject()->getString("analysis_status"), "solved");
  EXPECT_EQ(parsed->getAsObject()->getString("verification_status"), "solved-graph-verified");
  EXPECT_TRUE(parsed->getAsObject()->getObject("finalCost"));
  EXPECT_EQ(solution->cost.cost.codeSize, 14u);
  uint64_t historical = 0, selected = 0;
  unsigned attemptGraphs = 0;
  for (auto &e : *parsed->getAsObject()->getArray("events")) {
    auto *event = e.getAsObject();
    if (event->getString("event") == "graph") ++attemptGraphs;
    if (event->getString("event") == "attempt-end" && event->getInteger("attempt") == 1)
      EXPECT_EQ(event->getString("failureKind"), "empty-domain");
    if (event->getString("event") != "cost-contribution") continue;
    auto cost = (*event->getObject("estimate")->getArray("cost"))[7].getAsUINT64();
    ASSERT_TRUE(cost);
    if (event->getInteger("costScope") == 1) historical += *cost;
    if (event->getInteger("costScope") == 2) selected += *cost;
  }
  EXPECT_EQ(historical, 7u);
  EXPECT_EQ(selected, 14u);
  EXPECT_EQ(attemptGraphs, 2u);
  LayoutTrace capped({"unused.json", 1, 4096});
  ASSERT_TRUE(succeeded(attemptControlledRelaxation(retryGraph, target, {}, records, nullptr, nullptr, &capped)));
  auto cappedJSON = llvm::json::parse(capped.serialize());
  ASSERT_TRUE(bool(cappedJSON));
  EXPECT_EQ(cappedJSON->getAsObject()->getInteger("costScope"), 2);
  EXPECT_EQ((*cappedJSON->getAsObject()->getObject("omittedCost")->getArray("cost"))[7].getAsUINT64(), 14u);
  ASSERT_EQ(report.attempts.size(), 2u);
  EXPECT_EQ(report.attempts[0].kind, "initial");
  EXPECT_EQ(report.attempts[1].kind, "replication");
  EXPECT_EQ(report.attempts[0].statistics.preparedCandidates, 1u);
  EXPECT_EQ(report.attempts[1].statistics.preparedCandidates, 2u);
  ASSERT_EQ(report.relaxations.size(), 1u);
  EXPECT_EQ(*report.relaxations[0].getAsObject()->getInteger("generatedProposals"), 4);
  EXPECT_EQ(*report.relaxations[0].getAsObject()->getInteger("addedCandidates"), 1);
}
TEST(LayoutTraceIntegrationTest, BudgetFailureRecordsInterruptedSearchDepth) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "v");
  graph.getVariable(id).candidates = {{builder.getStringAttr("a")}, {builder.getStringAttr("b")}};
  LayoutTrace trace({"unused.json"});
  StatisticsTarget target;
  SmallVector<RelaxationRecord, 0> records;
  LayoutSolveFailure why;
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(attemptControlledRelaxation(graph, target, {256, 32, 1}, records, &why, nullptr, &trace)));
  EXPECT_EQ(why.kind, LayoutSolveFailureKind::SearchBudget);
  auto json = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(json));
  unsigned depths = 0;
  for (auto &e : *json->getAsObject()->getArray("events")) {
    auto *event = e.getAsObject();
    if (event->getString("event") != "search-depth") continue;
    ++depths;
    EXPECT_EQ(event->getInteger("probes"), 1);
    EXPECT_EQ(event->getBoolean("completed"), false);
    EXPECT_EQ(event->getInteger("retained"), 0);
  }
  EXPECT_EQ(depths, 1u);
}
TEST(LayoutTraceIntegrationTest, ExistingRequiredEncodingCheckIsDisprovenNotMissingSupport) {
  MLIRContext context;
  Builder builder(&context);
  LayoutConstraintGraph graph;
  auto id = graph.addVariable(LayoutKind::Distributed,
      RankedTensorType::get({4}, builder.getF32Type()), "v");
  graph.getVariable(id).candidates = {{builder.getStringAttr("a")}, {builder.getStringAttr("b")}};
  graph.addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {id}, nullptr, "required", "test seed", builder.getStringAttr("a"));
  LayoutTrace trace({"unused.json"});
  StatisticsTarget target;
  ASSERT_TRUE(succeeded(solveLayoutGraph(graph, target, {}, nullptr, nullptr, &trace)));
  auto json = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(json));
  unsigned rejected = 0;
  for (auto &e : *json->getAsObject()->getArray("events")) {
    auto *event = e.getAsObject();
    if (event->getString("event") != "candidate-rejected") continue;
    ++rejected;
    EXPECT_EQ(event->getString("status"), "Disproven");
  }
  EXPECT_EQ(rejected, 1u);
}
TEST(LayoutTraceIntegrationTest, PropagationPreservesThreeValuedRejectionEvidence) {
  MLIRContext context;
  Builder builder(&context);
  for (auto kind : {ConstraintKind::RequireEncoding, ConstraintKind::Ownership,
                    ConstraintKind::InstructionContract}) {
    LayoutConstraintGraph graph;
    auto id = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({4}, builder.getF32Type()), "v");
    graph.getVariable(id).candidates = {{builder.getStringAttr("b")}};
    graph.addConstraint(kind, ConstraintStrength::Hard, {id}, nullptr,
        "evidence", "test", kind == ConstraintKind::RequireEncoding ? builder.getStringAttr("a") : Attribute{});
    if (kind == ConstraintKind::InstructionContract) graph.getConstraints().front().instruction.emplace();
    LayoutTrace trace({"unused.json"});
    EXPECT_TRUE(failed(propagateStrict(graph, &trace)));
    auto json = llvm::json::parse(trace.serialize());
    ASSERT_TRUE(bool(json));
    unsigned rejections = 0;
    for (auto &e : *json->getAsObject()->getArray("events")) {
      auto *event = e.getAsObject();
      if (event->getString("event") != "candidate-rejected") continue;
      ++rejections;
      EXPECT_EQ(event->getString("status"), kind == ConstraintKind::RequireEncoding ? "Disproven"
          : kind == ConstraintKind::Ownership ? "Unknown" : "NoSupport");
      EXPECT_EQ(event->getString("stage"), "strict");
    }
    EXPECT_EQ(rejections, 1u);
  }
}
TEST(LayoutTraceIntegrationTest, MandatorySolverOptionsSurviveTruncationAndEarlyFailure) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-trace-options", directory)));
  std::string path = (directory + "/trace.json").str();
  SolverOptions options{17, 5, 23, false, 29};
  auto check = [&](StringRef text, StringRef expectedModelID) {
    auto json = llvm::json::parse(text);
    ASSERT_TRUE(bool(json));
    EXPECT_LE(text.size(), 4096u);
    EXPECT_TRUE(json->getAsObject()->getArray("events")->empty());
    EXPECT_EQ(json->getAsObject()->getString("cost_model_id"), expectedModelID);
    auto *solver = json->getAsObject()->getObject("solver_options");
    ASSERT_TRUE(solver);
    EXPECT_EQ(solver->getInteger("exactCombinationLimit"), 17);
    EXPECT_EQ(solver->getInteger("beamWidth"), 5);
    EXPECT_EQ(solver->getInteger("maxExpandedStates"), 23);
    EXPECT_EQ(solver->getBoolean("refineBeamSolution"), false);
    EXPECT_EQ(solver->getInteger("maxRefinementProbes"), 29);
  };
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  // Unknown operation with tensor operands fails collection/normalization
  // before solveLayoutGraph, so transaction-boundary initialization is needed.
  context.allowUnregisteredDialects();
  for (bool pipeline : {false, true}) {
    auto module = parseSourceString<ModuleOp>(R"(func.func @f(%x:tensor<4xf32>) {
      "unknown.layout_op"(%x) : (tensor<4xf32>) -> ()
      return
    })", &context);
    ASSERT_TRUE(module);
    LayoutStatisticsReport report;
    PassManager pm(&context);
    if (pipeline) buildFriskLayoutPipeline(pm, options, {}, &report, {path, 1, 4096});
    else {
      auto pass = createFriskInferLayoutsPass(options, false, &report);
      ASSERT_TRUE(succeeded(pass->initializeOptions("trace-file=" + path + " trace-max-bytes=4096",
          [](const Twine &) { return failure(); })));
      pm.addPass(std::move(pass));
    }
    EXPECT_TRUE(failed(pm.run(*module)));
    EXPECT_TRUE(report.attempts.empty());
    auto file = llvm::MemoryBuffer::getFile(path);
    ASSERT_TRUE(bool(file));
    EXPECT_EQ(report.costModelID, "sm90-layout-lex9-legacy-v2");
    EXPECT_EQ(report.toJSON().getString("cost_model_id"), report.costModelID);
    check((*file)->getBuffer(), "sm90-layout-lex9-legacy-v2");
  }
  StatisticsTarget target;
  for (bool coordinator : {false, true}) {
    LayoutConstraintGraph graph;
    // The coordinator rejects these invariants without entering the solver,
    // independently exercising both public API initialization boundaries.
    Builder builder(&context);
    auto id = graph.addVariable(LayoutKind::Distributed,
        RankedTensorType::get({4}, builder.getF32Type()), "invalid-id");
    graph.getVariable(id).id = 42;
    LayoutTrace trace({path, 1, 4096});
    SmallVector<RelaxationRecord, 0> records;
    auto result = coordinator ? attemptControlledRelaxation(graph, target, options, records, nullptr, nullptr, &trace)
                              : solveLayoutGraph(graph, target, options, nullptr, nullptr, &trace);
    EXPECT_TRUE(failed(result));
    check(trace.serialize(), "unversioned-target");
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(path)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
TEST(LayoutTraceIntegrationTest, TransactionRollbackForInvalidOptionsAndWriteFailures) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  const char *input = R"(func.func @f(%x:tensor<4xf32>) -> tensor<4xf32> {
    %y = arith.addf %x, %x : tensor<4xf32>
    return %y : tensor<4xf32>
  })";
  for (bool pipeline : {false, true}) {
    for (unsigned scenario = 0; scenario < 4; ++scenario) {
      auto module = parseSourceString<ModuleOp>(input, &context);
      auto print = [&] { std::string s; llvm::raw_string_ostream(s) << *module; return s; };
      auto before = print();
      LayoutTraceOptions options{"/nonexistent-frisk-trace-directory/trace.json"};
      std::string stats;
      if (scenario == 1) options.maxEvents = 0;
      if (scenario == 2) options.maxBytes = 4095;
      if (scenario == 3) stats = options.file;
      LayoutStatisticsReport report;
      PassManager pm(&context);
      if (pipeline) buildFriskLayoutPipeline(pm, {}, stats, &report, options);
      else {
        auto pass = createFriskInferLayoutsPass({}, false, &report);
        std::string text = "trace-file=" + options.file + " trace-max-events=" + std::to_string(options.maxEvents)
            + " trace-max-bytes=" + std::to_string(options.maxBytes);
        if (!stats.empty()) text += " stats-file=" + stats;
        ASSERT_TRUE(succeeded(pass->initializeOptions(text, [](const Twine &) { return failure(); })));
        pm.addPass(std::move(pass));
      }
      EXPECT_TRUE(failed(pm.run(*module)));
      EXPECT_EQ(print(), before);
      EXPECT_EQ(report.costModelID, "sm90-layout-lex9-legacy-v2");
      EXPECT_EQ(report.failureKind, scenario ? "trace-options" : "trace-write");
      if (scenario) EXPECT_TRUE(report.attempts.empty());
      else {
        ASSERT_FALSE(report.attempts.empty());
        EXPECT_TRUE(report.attempts.back().succeeded);
      }
    }
  }
}
TEST(LayoutTraceIntegrationTest, TraceOnOffAndRepeatedSemanticJSONAreIdentical) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-trace-integration", directory)));
  std::string path = (directory + "/trace.json").str();
  std::string referenceIR, referenceStats, referenceTrace;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    auto module = parseSourceString<ModuleOp>(R"(func.func @f(%x:tensor<4xf32>) -> tensor<4xf32> {
      %y = arith.addf %x, %x : tensor<4xf32>
      return %y : tensor<4xf32>
    })", &context);
    LayoutStatisticsReport report;
    PassManager pm(&context);
    buildFriskLayoutPipeline(pm, {}, {}, &report, {iteration ? path : ""});
    ASSERT_TRUE(succeeded(pm.run(*module)));
    EXPECT_EQ(report.costModelID, "sm90-layout-lex9-legacy-v2");
    std::string ir;
    llvm::raw_string_ostream(ir) << *module;
    report.timings.clear();
    for (auto &attempt : report.attempts) {
      attempt.statistics.solveTime = attempt.propagationTime = attempt.verificationTime = std::chrono::nanoseconds(0);
    }
    auto stats = llvm::formatv("{0}", llvm::json::Value(report.toJSON())).str();
    if (!iteration) { referenceIR = ir; referenceStats = stats; }
    else {
      EXPECT_EQ(ir, referenceIR);
      EXPECT_EQ(stats, referenceStats);
      auto file = llvm::MemoryBuffer::getFile(path);
      ASSERT_TRUE(bool(file));
      auto json = llvm::json::parse((*file)->getBuffer());
      ASSERT_TRUE(bool(json));
      EXPECT_EQ(json->getAsObject()->getString("cost_model_id"), report.costModelID);
      EXPECT_EQ(json->getAsObject()->getBoolean("costExplanationComplete"), true);
      if (iteration == 1) referenceTrace = (*file)->getBuffer().str();
      else EXPECT_EQ((*file)->getBuffer(), referenceTrace);
    }
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(path)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
TEST(LayoutStatisticsTest, WeightedTrafficDeduplicatesConvertsAndSaturatesWithoutClones) {
  LayoutConstraintGraph graph;
  LayoutSolution solution;
  LayoutConversionEdge edge;
  edge.resolution = EdgeResolutionKind::Convert;
  edge.constraint = 0; edge.sharedLeader = 0;
  edge.bytes = UINT64_MAX; edge.frequencyFactor = 2; edge.frequencyUsesProxy = true;
  solution.conversions.push_back(edge);
  edge.constraint = 1; solution.conversions.push_back(edge);
  edge.constraint = edge.sharedLeader = 2; edge.resolution = EdgeResolutionKind::Rematerialize;
  solution.conversions.push_back(edge);
  LayoutAttemptStatistics attempt;
  finishLayoutStatistics(graph, {}, &solution, attempt);
  EXPECT_EQ(attempt.statistics.conversionBytes, UINT64_MAX);
  EXPECT_EQ(attempt.statistics.weightedConversionBytes, UINT64_MAX);
  EXPECT_TRUE(attempt.statistics.weightedConversionBytesSaturated);
  EXPECT_TRUE(attempt.statistics.weightedConversionBytesUsesProxy);
  EXPECT_EQ(attempt.statistics.uniqueConversions, 1u);
  EXPECT_EQ(attempt.statistics.uniqueRematerializations, 1u);
}
TEST(LayoutTraceIntegrationTest, StaticEarlyFailurePreservesModelAndOriginalIR) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  context.allowUnregisteredDialects();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (bool pipeline : {false,true}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @bad(%x: tensor<4xf32>) {
        "unknown.layout_op"(%x) : (tensor<4xf32>) -> ()
        return
      }
    )mlir", &context);
    ASSERT_TRUE(module);
    auto print = [&] { std::string text; llvm::raw_string_ostream(text) << *module; return text; };
    auto before = print();
    LayoutStatisticsReport report;
    LayoutTrace trace({"unused.json"});
    PassManager pm(&context);
    if (pipeline) buildFriskLayoutPipeline(pm, {}, {}, &report, {}, {}, false, LayoutBoundaryFrequencyMode::StaticTripCount);
    else pm.addPass(createFriskInferLayoutsPass({}, false, &report, LayoutStatisticsScope::Standalone,
                                               &trace, nullptr, LayoutBoundaryFrequencyMode::StaticTripCount));
    EXPECT_TRUE(failed(pm.run(*module)));
    EXPECT_EQ(print(), before);
    EXPECT_EQ(report.costModelID, "sm90-layout-lex9-static-trip-count-v2");
    EXPECT_TRUE(report.staticTripCountFrequency);
    if (!pipeline) {
      EXPECT_EQ(trace.getSummary().costModelID, report.costModelID);
      EXPECT_TRUE(trace.getSummary().staticTripCountFrequency);
      EXPECT_FALSE(trace.getSummary().hasFinalCost);
    }
  }
}
TEST(LayoutTraceIntegrationTest, CalibratedProfileNormalEntryAndFullDigest) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-cost-profile", directory)));
  std::string path = (directory + "/profile.json").str();
  std::string tracePath = (directory + "/trace.json").str();
  SM90CostProfile calibration;
  calibration.coefficients.fill(1);
  calibration.provenance = "normal-pass-integration-test-only";
  {
    std::error_code ec;
    llvm::raw_fd_ostream file(path, ec);
    ASSERT_FALSE(bool(ec));
    file << serializeSM90CostProfile(calibration);
  }
  for (bool pipeline : {false, true}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @f(%x:tensor<32xf32>, %y:tensor<32xf32>) -> tensor<32xf32> {
        %z=arith.mulf %x,%y : tensor<32xf32>
        return %z : tensor<32xf32>
      })mlir", &context);
    ASSERT_TRUE(module);
    LayoutStatisticsReport report;
    PassManager pm(&context);
    if (pipeline)
      buildFriskLayoutPipeline(pm, {}, {}, &report, {tracePath}, {}, false,
                               LayoutBoundaryFrequencyMode::Legacy, path);
    else {
      auto pass = createFriskInferLayoutsPass({}, false, &report);
      ASSERT_TRUE(succeeded(pass->initializeOptions("cost-profile=" + path + " trace-file=" + tracePath, [](const Twine &) { return failure(); })));
      pm.addPass(std::move(pass));
    }
    ASSERT_TRUE(succeeded(pm.run(*module)));
    EXPECT_EQ(report.costModelID, getSM90CalibratedCostModelID(calibration));
    auto file = llvm::MemoryBuffer::getFile(tracePath);
    ASSERT_TRUE(bool(file));
    auto json = llvm::json::parse((*file)->getBuffer());
    ASSERT_TRUE(bool(json));
    EXPECT_EQ(json->getAsObject()->getString("cost_profile_sha256"), getSM90CostProfileSHA256(calibration));
    auto *cost = json->getAsObject()->getObject("finalCost");
    ASSERT_TRUE(cost);
    EXPECT_FALSE(cost->getObject("features"));
    bool sawArithmetic = false;
    for (const auto &entry : *json->getAsObject()->getArray("events")) {
      auto *event = entry.getAsObject();
      if (event->getString("event") != "cost-contribution") continue;
      auto *features = event->getObject("estimate")->getObject("features");
      if (features && features->getInteger("alu_warp_issues").value_or(0) > 0)
        sawArithmetic = true;
    }
    EXPECT_TRUE(sawArithmetic);
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(tracePath)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(path)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
TEST(LayoutTraceIntegrationTest, InvalidCalibrationIsAtomicallyRejected) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-invalid-cost-profile", directory)));
  std::string path = (directory + "/profile.json").str();
  {
    std::error_code ec;
    llvm::raw_fd_ostream file(path, ec);
    ASSERT_FALSE(bool(ec)); file << "{}";
  }
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (unsigned entry : {0u,1u,2u}) {
    auto module = parseSourceString<ModuleOp>("func.func @f() { return }", &context);
    ASSERT_TRUE(module);
    auto print = [&] { std::string out; llvm::raw_string_ostream(out) << *module; return out; };
    auto before = print();
    PassManager pm(&context);
    if (entry == 2) buildFriskLayoutPipeline(pm, {}, {}, nullptr, {}, {}, false,
                                            LayoutBoundaryFrequencyMode::Legacy, path);
    else {
      auto pass = entry == 0 ? createFriskInferLayoutsPass() : createOptimizeLayoutConversionsPass();
      ASSERT_TRUE(succeeded(pass->initializeOptions("cost-profile=" + path, [](const Twine &) { return failure(); })));
      pm.addPass(std::move(pass));
    }
    EXPECT_TRUE(failed(pm.run(*module)));
    EXPECT_EQ(print(), before);
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(path)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}

}
