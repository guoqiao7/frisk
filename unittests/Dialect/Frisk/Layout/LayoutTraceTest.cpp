#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include "LayoutCostObserverTestSupport.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
LayoutTraceOptions enabled(uint64_t events = 4096, uint64_t bytes = 8388608) {
  return {"unused.json", events, bytes};
}
TEST(LayoutTraceTest, UnsolvedTraceHasExplicitUnversionedModelIdentity) {
  LayoutTrace trace(enabled());
  auto json = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(json));
  EXPECT_EQ(json->getAsObject()->getString("cost_model_id"), "unversioned-target");
  EXPECT_EQ(json->getAsObject()->getInteger("schema_version"), 1);
}
TEST(LayoutTraceTest, ModelIdentitySurvivesTruncationFailureAndRepeatedSerialization) {
  for (auto options : {enabled(), enabled(1), enabled(100, 4096)}) {
    LayoutTrace trace(options);
    LayoutTraceSummary summary;
    summary.costModelID = "sm90-layout-lex9-legacy-v1";
    trace.setSummary(summary);
    trace.recordEvent([] { return llvm::json::Object{{"event", "first"}}; });
    trace.recordEvent([] { return llvm::json::Object{{"event", "second"}}; });
    LayoutSolveFailure why{LayoutSolveFailureKind::InvalidInput, UINT32_MAX,
                           UINT32_MAX, "early failure"};
    trace.finishAttempt(why, nullptr);
    auto text = trace.serialize();
    EXPECT_EQ(text, trace.serialize());
    EXPECT_LE(text.size(), options.maxBytes);
    auto json = llvm::json::parse(text);
    ASSERT_TRUE(bool(json));
    EXPECT_EQ(json->getAsObject()->getString("cost_model_id"), "sm90-layout-lex9-legacy-v1");
    EXPECT_EQ(json->getAsObject()->getString("analysis_status"), "failed");
  }
}
TEST(LayoutTraceTest, InvalidModelIdentityIsRejectedNotTruncated) {
  for (const std::string id : {std::string(), std::string(65, 'a'),
         std::string(64, 'a') + "!", std::string("model with space"),
         std::string("model\n"), std::string("模型"), std::string("id\0bad", 6)}) {
    LayoutTrace trace(enabled());
    LayoutTraceSummary summary;
    summary.costModelID = id;
    trace.setSummary(summary);
    auto json = llvm::json::parse(trace.serialize());
    ASSERT_TRUE(bool(json));
    EXPECT_EQ(json->getAsObject()->getString("cost_model_id"), "unversioned-target");
    EXPECT_EQ(trace.getSummary().costModelID, "unversioned-target");
  }
  LayoutTrace trace(enabled());
  LayoutTraceSummary summary;
  summary.costModelID = "custom.A_2-" + std::string(53, 'Z');
  ASSERT_EQ(summary.costModelID.size(), 64u);
  trace.setSummary(summary);
  EXPECT_EQ(trace.getSummary().costModelID, summary.costModelID);
}
TEST(LayoutTraceTest, DefaultsValidationAndOutputPathConflict) {
  LayoutTraceOptions options;
  EXPECT_TRUE(options.file.empty());
  EXPECT_EQ(options.maxEvents, 4096u);
  EXPECT_EQ(options.maxBytes, 8388608u);
  std::string error;
  EXPECT_TRUE(succeeded(options.validate("", error)));
  options = enabled(0);
  EXPECT_TRUE(failed(options.validate("", error)));
  options = enabled(1, 4095);
  EXPECT_TRUE(failed(options.validate("", error)));
  options = enabled(1, 4096);
  EXPECT_TRUE(succeeded(options.validate("", error)));
  EXPECT_TRUE(failed(options.validate("./unused.json", error)));
}
TEST(LayoutTraceTest, OutputPathsResolveSymlinkBeforeParentTraversal) {
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-trace-path", directory)));
  std::string base = (directory + "/base").str();
  std::string other = (directory + "/other").str();
  ASSERT_FALSE(bool(llvm::sys::fs::create_directory(base)));
  ASSERT_FALSE(bool(llvm::sys::fs::create_directory(other)));
  ASSERT_FALSE(bool(llvm::sys::fs::create_directory(other + "/child")));
  ASSERT_FALSE(bool(llvm::sys::fs::create_link(other + "/child", base + "/link")));
  auto options = enabled();
  options.file = base + "/link/../report.json";
  ASSERT_FALSE(llvm::sys::fs::exists(other + "/report.json"));
  ASSERT_FALSE(llvm::sys::fs::exists(base + "/report.json"));
  std::string error;
  EXPECT_TRUE(failed(options.validate(other + "/report.json", error)));
  EXPECT_FALSE(error.empty());
  EXPECT_TRUE(succeeded(options.validate(base + "/report.json", error)));
  EXPECT_TRUE(error.empty());
  EXPECT_FALSE(bool(llvm::sys::fs::remove(base + "/link")));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(other + "/child")));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(base)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(other)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
TEST(LayoutTraceTest, DisabledAndTruncatedRecordingNeverBuildsLazyPayloads) {
  unsigned calls = 0;
  auto payload = [&] { ++calls; return llvm::json::Object{{"attribute", "x"}}; };
  LayoutTrace off;
  off.recordEvent(payload);
  EXPECT_EQ(calls, 0u);
  EXPECT_TRUE(off.serialize().empty());
  LayoutTrace trace(enabled(1));
  EXPECT_TRUE(trace.recordEvent(payload));
  EXPECT_FALSE(trace.recordEvent(payload));
  EXPECT_FALSE(trace.recordEvent(payload));
  EXPECT_EQ(calls, 1u);
  auto json = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(json));
  EXPECT_EQ(json->getAsObject()->getInteger("eventsSeen"), 3);
  EXPECT_EQ(json->getAsObject()->getInteger("eventsOmitted"), 2);
  EXPECT_EQ(json->getAsObject()->getArray("events")->size(), 1u);
  EXPECT_EQ(json->getAsObject()->getBoolean("truncated"), true);
}
TEST(LayoutTraceTest, ByteCapCountsEnvelopeEscapingAndWholeLargeEvents) {
  LayoutTrace trace(enabled(100, 5000));
  EXPECT_TRUE(trace.recordEvent([] { return llvm::json::Object{{"text", "quote\"\n\\"}}; }));
  EXPECT_FALSE(trace.recordEvent([] { return llvm::json::Object{{"attribute", std::string(100000, '\1')}}; }));
  unsigned calls = 0;
  trace.recordEvent([&] { ++calls; return llvm::json::Object{}; });
  EXPECT_EQ(calls, 0u);
  LayoutTraceSummary summary;
  summary.target = std::string(10000, '\1');
  summary.costCoverage = summary.target;
  summary.costModelID = std::string(64, 'Z');
  summary.analysisStatus = summary.target;
  summary.verificationStatus = summary.target;
  summary.failureKind = summary.target;
  summary.failureReason = summary.target;
  summary.attempts = UINT64_MAX;
  summary.exactCombinationLimit = UINT64_MAX;
  summary.beamWidth = UINT_MAX;
  summary.maxExpandedStates = UINT64_MAX;
  summary.refineBeamSolution = false;
  summary.maxRefinementProbes = UINT64_MAX;
  summary.finalCost.cost = {UINT64_MAX, UINT64_MAX, UINT64_MAX,
                           UINT64_MAX, UINT64_MAX, UINT64_MAX,
                           UINT64_MAX, UINT64_MAX, UINT64_MAX};
  summary.finalCost.saturated = true;
  summary.costProfileSHA256 = std::string(64, 'a');
  for (StringRef name : getSM90CostFeatureNames())
    summary.finalCost.features.emplace_back(name.str(), UINT64_MAX);
  summary.finalCost.features.emplace_back(std::string(1000000, 'x'), UINT64_MAX);
  summary.hasFinalCost = true;
  trace.setSummary(summary);
  EXPECT_TRUE(trace.getSummary().finalCost.features.empty());
  auto serialized = trace.serialize();
  EXPECT_LE(serialized.size(), 5000u);
  EXPECT_EQ(serialized, trace.serialize());
  auto json = llvm::json::parse(serialized);
  ASSERT_TRUE(bool(json));
  auto *events = json->getAsObject()->getArray("events");
  ASSERT_EQ(events->size(), 1u);
  EXPECT_EQ((*events)[0].getAsObject()->getString("text"), "quote\"\n\\");
  EXPECT_EQ(json->getAsObject()->getBoolean("summaryTextTruncated"), true);
  LayoutTrace minimum(enabled(UINT64_MAX, 4096));
  minimum.beginCostScope(UINT64_MAX);
  minimum.setSummary(summary);
  minimum.recordEvent([] { return llvm::json::Object{{"x", "y"}}; });
  minimum.recordCost("conversion", 0, summary.finalCost);
  EXPECT_LE(minimum.serialize().size(), 4096u);
  // Four counters are currently single-digit; reserve their remaining digits.
  EXPECT_LE(minimum.serialize().size() + 4 * 19, 4096u);
  EXPECT_TRUE(bool(llvm::json::parse(minimum.serialize())));
  EXPECT_EQ(minimum.omittedCost().cost, summary.finalCost.cost);
  EXPECT_TRUE(minimum.omittedCost().features.empty());
  auto minJSON = llvm::json::parse(minimum.serialize());
  ASSERT_TRUE(bool(minJSON));
  auto *solver = minJSON->getAsObject()->getObject("solver_options");
  EXPECT_EQ(minJSON->getAsObject()->getString("cost_model_id"), summary.costModelID);
  EXPECT_EQ(minJSON->getAsObject()->getString("cost_profile_sha256"), summary.costProfileSHA256);
  EXPECT_FALSE(minJSON->getAsObject()->getObject("finalCost")->getObject("features"));
  EXPECT_FALSE(minJSON->getAsObject()->getObject("omittedCost")->getObject("features"));
  ASSERT_TRUE(solver);
  EXPECT_EQ(solver->get("exactCombinationLimit")->getAsUINT64(), UINT64_MAX);
  EXPECT_EQ(solver->get("beamWidth")->getAsUINT64(), UINT_MAX);
  EXPECT_EQ(solver->get("maxExpandedStates")->getAsUINT64(), UINT64_MAX);
  EXPECT_EQ(solver->getBoolean("refineBeamSolution"), false);
  ASSERT_TRUE(solver->get("maxRefinementProbes"));
  EXPECT_EQ(solver->get("maxRefinementProbes")->getAsUINT64(), UINT64_MAX);
}
TEST(LayoutTraceTest, OmittedContributionsUseSaturatedActualCosts) {
  LayoutTrace trace(enabled(1));
  CostEstimate a, b, c;
  a.cost.codeSize = 7;
  b.cost.codeSize = UINT64_MAX;
  c.cost.codeSize = 1;
  trace.recordCost("registers", 0, a);
  trace.recordCost("conversion", 1, b);
  trace.recordCost("rematerialization", 2, c);
  EXPECT_EQ(trace.omittedCost().cost.codeSize, UINT64_MAX);
  EXPECT_TRUE(trace.omittedCost().saturated);
  auto sum = addLayoutCosts(a, trace.omittedCost());
  EXPECT_EQ(sum.cost, addLayoutCosts(addLayoutCosts(a, b), c).cost);
  EXPECT_TRUE(sum.saturated);
  auto json = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(json));
  EXPECT_EQ(json->getAsObject()->getInteger("contributionsSeen"), 3);
  EXPECT_EQ(json->getAsObject()->getBoolean("costExplanationComplete"), false);
}
TEST(LayoutTraceTest, BoundedSummaryPreservesUTF8AndCostCompleteness) {
  LayoutTrace trace(enabled());
  LayoutTraceSummary summary;
  summary.target = std::string(63, 'x') + "中";
  summary.costCoverage = std::string(127, 'x') + "文";
  summary.analysisStatus = std::string(31, 'x') + "✓";
  summary.verificationStatus = summary.analysisStatus;
  summary.hasFinalCost = true;
  summary.finalCost.cost.codeSize = 1;
  trace.recordCost("conversion", 1, summary.finalCost);
  trace.setSummary(summary);
  auto serialized = trace.serialize();
  auto json = llvm::json::parse(serialized);
  ASSERT_TRUE(bool(json));
  EXPECT_EQ(json->getAsObject()->getString("target"), std::string(63, 'x'));
  EXPECT_EQ(json->getAsObject()->getBoolean("costExplanationComplete"), true);
}
TEST(LayoutTraceTest, AttemptCostScopesNeverMixHistoricalOrOmittedCosts) {
  LayoutTrace trace(enabled(1));
  CostEstimate previous, selected;
  previous.cost.codeSize = 17;
  selected.cost.memoryTransactions = 3;
  trace.beginCostScope(1);
  trace.recordCost("conversion", 1, previous);
  trace.recordCost("conversion", 2, previous);
  EXPECT_EQ(trace.omittedCost().cost.codeSize, 17u);
  trace.beginCostScope(2);
  trace.recordCost("access", 3, selected);
  LayoutTraceSummary summary;
  summary.finalCost = selected;
  summary.hasFinalCost = true;
  trace.setSummary(summary);
  EXPECT_EQ(trace.omittedCost().cost, selected.cost);
  auto json = llvm::json::parse(trace.serialize());
  ASSERT_TRUE(bool(json));
  auto *object = json->getAsObject();
  EXPECT_EQ(object->getInteger("costScope"), 2);
  EXPECT_EQ(object->getInteger("contributionsSeen"), 1);
  EXPECT_EQ(object->getInteger("contributionsOmitted"), 1);
  EXPECT_EQ(object->getInteger("eventsSeen"), 3);
  auto *events = object->getArray("events");
  ASSERT_EQ(events->size(), 1u);
  EXPECT_EQ((*events)[0].getAsObject()->getInteger("costScope"), 1);
}
TEST(LayoutTraceTest, AtomicPublicationAndFailurePreserveExistingDestination) {
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-trace", directory)));
  auto options = enabled();
  options.file = (directory + "/trace.json").str();
  LayoutTrace trace(options);
  trace.recordEvent([] { return llvm::json::Object{{"value", 42}}; });
  std::string error;
  ASSERT_TRUE(succeeded(trace.publish(error))) << error;
  auto contents = llvm::MemoryBuffer::getFile(options.file);
  ASSERT_TRUE(bool(contents));
  EXPECT_EQ((*contents)->getBuffer(), trace.serialize());
  // Renaming a file over a nonempty directory must fail and retain the report.
  auto failedOptions = options;
  failedOptions.file = directory.str().str();
  LayoutTrace failedTrace(failedOptions);
  EXPECT_TRUE(failed(failedTrace.publish(error)));
  EXPECT_FALSE(error.empty());
  auto after = llvm::MemoryBuffer::getFile(options.file);
  ASSERT_TRUE(bool(after));
  EXPECT_EQ((*after)->getBuffer(), (*contents)->getBuffer());
  failedOptions.file = (directory + "/missing/trace.json").str();
  EXPECT_TRUE(failed(LayoutTrace(failedOptions).publish(error)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(options.file)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
} // namespace
