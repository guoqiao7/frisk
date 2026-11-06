#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <thread>

using namespace mlir;
using namespace mlir::frisk;
namespace {
TEST(LayoutProfileTest, DisabledNestedScopesAndIndependentInvocations) {
  LayoutProfile disabled, outer(true), inner(true);
  EXPECT_FALSE(disabled.enabled());
  EXPECT_EQ(LayoutProfile::current(), nullptr);
  {
    LayoutProfile::Invocation invocation(&outer);
    { LayoutProfile::Scope event(LayoutProfileKind::AccessCost); }
    {
      LayoutProfile::Invocation nested(&inner);
      LayoutProfile::Scope event(LayoutProfileKind::AccessCost);
    }
    EXPECT_EQ(LayoutProfile::current(), &outer);
    {
      LayoutProfile::Invocation off(&disabled);
      LayoutProfile::Scope event(LayoutProfileKind::AccessCost);
      EXPECT_EQ(LayoutProfile::current(), nullptr);
    }
    { LayoutProfile::Scope event(LayoutProfileKind::AccessCost); }
  }
  EXPECT_EQ(LayoutProfile::current(), nullptr);
  EXPECT_EQ(outer.get(LayoutProfileKind::AccessCost).count, 2u);
  EXPECT_EQ(inner.get(LayoutProfileKind::AccessCost).count, 1u);
  EXPECT_EQ(disabled.get(LayoutProfileKind::AccessCost).count, 0u);
  EXPECT_TRUE(disabled.serialize("solved").empty());
}
TEST(LayoutProfileTest, ThreadIsolationAndCacheInsertionSchema) {
  LayoutProfile first(true), second(true);
  {
    LayoutProfile::Invocation invocation(&first);
    std::thread worker([&] {
      EXPECT_EQ(LayoutProfile::current(), nullptr);
      LayoutProfile::Invocation invocation(&second);
      LayoutProfile::Scope event(LayoutProfileKind::FootprintCache);
      LayoutProfile::cacheLookup(LayoutProfileKind::FootprintCache, false);
      LayoutProfile::cacheInsertion(LayoutProfileKind::FootprintCache);
    });
    worker.join();
    LayoutProfile::Scope event(LayoutProfileKind::FootprintCache);
    LayoutProfile::cacheLookup(LayoutProfileKind::FootprintCache, true);
  }
  EXPECT_EQ(first.get(LayoutProfileKind::FootprintCache).hits, 1u);
  EXPECT_EQ(first.get(LayoutProfileKind::FootprintCache).misses, 0u);
  EXPECT_EQ(second.get(LayoutProfileKind::FootprintCache).entries, 1u);
  auto json = llvm::json::parse(second.serialize("failed"));
  ASSERT_TRUE(bool(json));
  auto *object = json->getAsObject();
  EXPECT_EQ(object->getInteger("version"), 1);
  EXPECT_EQ(object->getString("status"), "failed");
  EXPECT_EQ(object->getString("timing_kind"), "inclusive");
  EXPECT_EQ(object->getString("entries_kind"), "insertions");
  auto *metric = object->getObject("metrics")->getObject("footprint_cache");
  EXPECT_EQ(metric->getInteger("count"), 1);
  EXPECT_GE(*metric->getInteger("inclusive_ns"), 0);
  EXPECT_EQ(metric->getInteger("hits"), 0);
  EXPECT_EQ(metric->getInteger("misses"), 1);
  EXPECT_EQ(metric->getInteger("entries"), 1);
}
TEST(LayoutProfileTest, RealFootprintAndAliasCachesCountInsertionsNotLiveEntries) {
  MLIRContext context;
  context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
  auto module = parseSourceString<ModuleOp>(
      "func.func @f(%r: memref<4xi32, 3>) { return }", &context);
  ASSERT_TRUE(module);
  auto root = module->lookupSymbol<func::FuncOp>("f").getArgument(0);
  auto info = analyzeStorageAlias(root);
  ASSERT_TRUE(succeeded(info));
  auto encoding = buildRootLinearStorageCandidate(*info);
  ASSERT_TRUE(succeeded(encoding));
  LayoutConstraintGraph graph;
  for (StringRef name : {"first", "second"}) {
    auto id = graph.addVariable(LayoutKind::Storage, root.getType(), name);
    graph.getVariable(id).storageAlias = *info;
  }
  LayoutProfile profile(true);
  {
    LayoutProfile::Invocation invocation(&profile);
    EXPECT_EQ(getStorageAliasFootprint(graph, 0, *encoding).proof.status, ProofStatus::Proven);
    (void)getStorageAliasFootprint(graph, 0, *encoding);
    EXPECT_EQ(proveAliasLayoutRelation(graph, 0, *encoding, 1, *encoding).status, ProofStatus::Proven);
    (void)proveAliasLayoutRelation(graph, 1, *encoding, 0, *encoding);
    EXPECT_EQ(profile.get(LayoutProfileKind::AddressEnumeration).count, 2u);
    EXPECT_EQ(profile.get(LayoutProfileKind::FootprintCache).count, 6u);
    EXPECT_EQ(profile.get(LayoutProfileKind::FootprintCache).hits, 4u);
    EXPECT_EQ(profile.get(LayoutProfileKind::FootprintCache).misses, 2u);
    EXPECT_EQ(profile.get(LayoutProfileKind::FootprintCache).entries, 2u);
    EXPECT_EQ(profile.get(LayoutProfileKind::AliasPairCache).count, 2u);
    EXPECT_EQ(profile.get(LayoutProfileKind::AliasPairCache).hits, 1u);
    EXPECT_EQ(profile.get(LayoutProfileKind::AliasPairCache).misses, 1u);
    EXPECT_EQ(profile.get(LayoutProfileKind::AliasPairCache).entries, 1u);
    graph.getAliasFootprintCache().clear();
    (void)getStorageAliasFootprint(graph, 0, *encoding);
  }
  EXPECT_EQ(graph.getAliasFootprintCache().size(), 1u);
  EXPECT_EQ(profile.get(LayoutProfileKind::FootprintCache).entries, 3u);
  EXPECT_EQ(profile.get(LayoutProfileKind::AddressEnumeration).count, 3u);
  EXPECT_EQ(graph.getCandidatePreparationStatistics().footprintEvaluations, 3u);
  EXPECT_EQ(graph.getCandidatePreparationStatistics().pairProofEvaluations, 1u);
}
TEST(LayoutProfileTest, OutputPathAliasesAndAtomicRenameFailure) {
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-profile-paths", directory)));
  auto path = (directory + "/report.json").str();
  auto link = (directory + "/link.json").str();
  auto hard = (directory + "/hard.json").str();
  auto destination = (directory + "/destination").str();
  std::string error;
  LayoutProfile profile(true);
  ASSERT_TRUE(succeeded(profile.publish(path, "solved", error)));
  ASSERT_FALSE(bool(llvm::sys::fs::create_link(path, link)));
  ASSERT_FALSE(bool(llvm::sys::fs::create_hard_link(path, hard)));
  for (auto alias : {path, link, hard, (directory + "/./report.json").str()}) {
    EXPECT_TRUE(failed(LayoutProfile::validate(path, alias, "", error)));
    EXPECT_TRUE(failed(LayoutProfile::validate(path, "", alias, error)));
  }
  ASSERT_FALSE(bool(llvm::sys::fs::create_directory(destination)));
  EXPECT_TRUE(failed(profile.publish(destination, "failed", error)));
  EXPECT_TRUE(llvm::sys::fs::is_directory(destination));
  unsigned files = 0;
  std::error_code ec;
  for (llvm::sys::fs::directory_iterator it(directory, ec), end; it != end && !ec; it.increment(ec)) ++files;
  EXPECT_FALSE(bool(ec));
  EXPECT_EQ(files, 4u); // no leaked temporary sibling after failed rename
  EXPECT_FALSE(bool(llvm::sys::fs::remove(path)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(link)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(hard)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(destination)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
std::string printModule(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream(text) << module;
  return text;
}
TEST(LayoutProfileIntegrationTest, ProfileOptionsAndWriteFailuresRollback) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (bool pipeline : {false, true}) {
    for (unsigned scenario = 0; scenario < 3; ++scenario) {
      auto module = parseSourceString<ModuleOp>(R"(func.func @f(%x:tensor<4xf32>) -> tensor<4xf32> {
        %y = arith.addf %x, %x : tensor<4xf32>
        return %y : tensor<4xf32>
      })", &context);
      auto before = printModule(*module);
      std::string path = "/nonexistent-frisk-profile-directory/profile.json";
      std::string stats = scenario == 1 ? path : "";
      LayoutTraceOptions trace{scenario == 2 ? path : ""};
      LayoutStatisticsReport report;
      PassManager pm(&context);
      if (pipeline) buildFriskLayoutPipeline(pm, {}, stats, &report, trace, path);
      else {
        auto pass = createFriskInferLayoutsPass({}, false, &report);
        std::string options = "profile-file=" + path;
        if (!stats.empty()) options += " stats-file=" + stats;
        if (!trace.file.empty()) options += " trace-file=" + trace.file;
        ASSERT_TRUE(succeeded(pass->initializeOptions(options, [](const Twine &) { return failure(); })));
        pm.addPass(std::move(pass));
      }
      LayoutProfile ambient(true);
      {
        LayoutProfile::Invocation unrelated(&ambient);
        EXPECT_TRUE(failed(pm.run(*module)));
        EXPECT_EQ(LayoutProfile::current(), &ambient);
      }
      EXPECT_EQ(ambient.get(LayoutProfileKind::RelationQuery).count, 0u);
      EXPECT_EQ(printModule(*module), before);
      EXPECT_EQ(report.failureKind, scenario ? "profile-options" : "profile-write");
      EXPECT_EQ(LayoutProfile::current(), nullptr);
    }
  }
}
TEST(LayoutProfileIntegrationTest, ProfilePreservesIRDomainsSolutionCostsProbesAndTrace) {
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-profile", directory)));
  auto profilePath = (directory + "/profile.json").str();
  auto tracePath = (directory + "/trace.json").str();
  std::string referenceIR, referenceStats, referenceTrace, referenceCounts;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    auto module = parseSourceString<ModuleOp>(R"(func.func @f(%x:tensor<4xf32>) -> tensor<4xf32> {
      %y = arith.addf %x, %x : tensor<4xf32>
      return %y : tensor<4xf32>
    })", &context);
    LayoutStatisticsReport report;
    PassManager pm(&context);
    buildFriskLayoutPipeline(pm, {}, {}, &report, {tracePath}, iteration ? profilePath : "");
    LayoutProfile ambient(true);
    {
      LayoutProfile::Invocation unrelated(&ambient);
      ASSERT_TRUE(succeeded(pm.run(*module)));
      EXPECT_EQ(LayoutProfile::current(), &ambient);
    }
    EXPECT_EQ(ambient.get(LayoutProfileKind::RelationQuery).count, 0u);
    EXPECT_EQ(LayoutProfile::current(), nullptr);
    report.timings.clear();
    for (auto &attempt : report.attempts)
      attempt.statistics.solveTime = attempt.propagationTime = attempt.verificationTime = std::chrono::nanoseconds(0);
    auto stats = llvm::formatv("{0}", llvm::json::Value(report.toJSON())).str();
    auto trace = llvm::MemoryBuffer::getFile(tracePath);
    ASSERT_TRUE(bool(trace));
    if (!iteration) {
      referenceIR = printModule(*module); referenceStats = stats;
      referenceTrace = (*trace)->getBuffer().str();
    } else {
      EXPECT_EQ(printModule(*module), referenceIR);
      EXPECT_EQ(stats, referenceStats);
      // Trace includes prepared domains, the selected solution and all nine cost fields.
      EXPECT_EQ((*trace)->getBuffer(), referenceTrace);
      auto file = llvm::MemoryBuffer::getFile(profilePath);
      ASSERT_TRUE(bool(file));
      auto json = llvm::json::parse((*file)->getBuffer());
      ASSERT_TRUE(bool(json));
      auto *metrics = json->getAsObject()->getObject("metrics");
      ASSERT_TRUE(metrics);
      EXPECT_GT(*metrics->getObject("relation_query")->getInteger("count"), 0);
      EXPECT_GT(*metrics->getObject("relation_cache")->getInteger("misses"), 0);
      for (auto &metric : *metrics) (*metric.second.getAsObject())["inclusive_ns"] = 0;
      auto counts = llvm::formatv("{0}", *json).str();
      if (iteration == 1) referenceCounts = counts;
      else EXPECT_EQ(counts, referenceCounts);
    }
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(profilePath)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(tracePath)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
} // namespace
