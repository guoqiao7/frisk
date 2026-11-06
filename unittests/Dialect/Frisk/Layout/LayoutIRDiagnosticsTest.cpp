#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FormatVariadic.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
class LayoutIRDiagnosticsTest : public testing::Test {
protected:
  MLIRContext context;
  LayoutIRDiagnosticsTest() {
    context.loadDialect<arith::ArithDialect, math::MathDialect,
                        func::FuncDialect, scf::SCFDialect>();
  }
  std::string print(Operation *op) {
    std::string text;
    llvm::raw_string_ostream(text) << *op;
    return text;
  }
};
constexpr StringLiteral pipelineArithmetic = R"mlir(
func.func @arithmetic(%x: tensor<64xf32>) -> tensor<64xf32> {
  %a = arith.mulf %x, %x : tensor<64xf32>
  return %a : tensor<64xf32>
})mlir";

TEST(LayoutIRDiagnosticsLifetimeTest, DetachedJSONOwnsOpcodesBeyondIRContext) {
  llvm::json::Object result;
  {
    MLIRContext context;
    context.loadDialect<arith::ArithDialect, func::FuncDialect>();
    auto module = parseSourceString<ModuleOp>(R"mlir(
func.func @lifetime(%dynamic: tensor<?xf32>, %scalar: f32) {
  %d = arith.mulf %dynamic, %dynamic : tensor<?xf32>
  %s = arith.addf %scalar, %scalar : f32
  return
})mlir", &context);
    ASSERT_TRUE(module);
    StringRef contextOpcode;
    module->walk([&](arith::MulFOp op) { contextOpcode = op->getName().getStringRef(); });
    result = collectLayoutIRArithmetic(*module);
    auto *unavailable = result.getArray("unavailable_operations");
    ASSERT_TRUE(unavailable);
    ASSERT_EQ(unavailable->size(), 1u);
    auto opcode = (*unavailable)[0].getAsObject()->getString("opcode");
    ASSERT_TRUE(opcode);
    ASSERT_EQ(*opcode, contextOpcode);
    // Context storage is immutable, so prove non-aliasing while it is alive.
    // Stop before destruction on failure to avoid a dangling read in the RED run.
    ASSERT_NE(opcode->data(), contextOpcode.data());
  }
  auto text = llvm::formatv("{0}", llvm::json::Value(std::move(result))).str();
  auto parsed = llvm::json::parse(text);
  ASSERT_TRUE(bool(parsed));
  auto *object = parsed->getAsObject();
  ASSERT_TRUE(object);
  EXPECT_EQ(object->getString("status"), "partial");
  auto *unavailable = object->getArray("unavailable_operations");
  ASSERT_TRUE(unavailable);
  ASSERT_EQ(unavailable->size(), 1u);
  EXPECT_EQ((*unavailable)[0].getAsObject()->getString("opcode"), "arith.mulf");
  EXPECT_EQ((*unavailable)[0].getAsObject()->getString("reason"), "dynamic-shape");
  auto *entries = object->getArray("entries");
  ASSERT_TRUE(entries);
  ASSERT_EQ(entries->size(), 1u);
  EXPECT_EQ((*entries)[0].getAsObject()->getString("opcode"), "arith.addf");
  EXPECT_EQ((*entries)[0].getAsObject()->getString("dtype"), "f32");
}

TEST_F(LayoutIRDiagnosticsTest, PipelineIsOptInPureAndReplayStableWithTraceCostReconciliation) {
  context.loadDialect<FriskDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-ir-diagnostics", directory)));
  std::string tracePath = (directory + "/trace.json").str();
  std::string expectedIR, expectedStats, expectedCost;
  for (bool enabled : {false, true}) {
    for (bool truncated : {false, true}) {
      auto module = parseSourceString<ModuleOp>(pipelineArithmetic, &context);
      ASSERT_TRUE(module);
      LayoutStatisticsReport report;
      PassManager pm(&context);
      buildFriskLayoutPipeline(pm, {}, {}, &report,
          {tracePath, truncated ? 1u : 4096u, truncated ? 4096u : 8388608u}, {}, enabled);
      ASSERT_TRUE(succeeded(pm.run(*module)));
      auto json = report.toJSON();
      ASSERT_TRUE(json.get("actual_ir_diagnostics"));
      EXPECT_EQ(report.actualIRDiagnostics.getAsObject() != nullptr, enabled);
      if (enabled) {
        auto *diagnostics = report.actualIRDiagnostics.getAsObject();
        ASSERT_TRUE(diagnostics);
        EXPECT_EQ(diagnostics->getString("stage"), "post-conversion-optimization");
        EXPECT_EQ(llvm::formatv("{0}", *diagnostics->get("arithmetic")).str(),
            llvm::formatv("{0}", llvm::json::Value(collectLayoutIRArithmetic(*module))).str());
        auto *resources = diagnostics->getObject("resources");
        ASSERT_TRUE(resources);
        EXPECT_EQ(resources->getString("status"), "available");
        EXPECT_EQ(*diagnostics->get("resources"),
                  llvm::json::Value(collectLayoutIRResources(*module)));
      }
      auto traceBuffer = llvm::MemoryBuffer::getFile(tracePath);
      ASSERT_TRUE(bool(traceBuffer));
      auto trace = llvm::json::parse((*traceBuffer)->getBuffer());
      ASSERT_TRUE(bool(trace));
      auto *object = trace->getAsObject();
      EXPECT_EQ(object->getString("cost_model_id"), "sm90-layout-lex9-legacy-v2");
      auto *final = object->getObject("finalCost");
      ASSERT_TRUE(final);
      for (unsigned field = 0; field < 9; ++field) {
        uint64_t sum = *(*object->getObject("omittedCost")->getArray("cost"))[field].getAsUINT64();
        for (auto &entry : *object->getArray("events")) {
          auto *event = entry.getAsObject();
          if (event->getString("event") == "cost-contribution")
            sum += *(*event->getObject("estimate")->getArray("cost"))[field].getAsUINT64();
        }
        EXPECT_EQ(sum, *(*final->getArray("cost"))[field].getAsUINT64());
      }
      unsigned diagnosticsEvents = 0;
      for (auto &entry : *object->getArray("events")) {
        auto *event = entry.getAsObject();
        if (event->getString("event") == "actual-ir-diagnostics") {
          ++diagnosticsEvents;
          EXPECT_EQ(*event->get("diagnostics"), report.actualIRDiagnostics);
        }
      }
      EXPECT_EQ(diagnosticsEvents, enabled && !truncated ? 1u : 0u);
      if (truncated) EXPECT_EQ(object->getBoolean("truncated"), true);
      json.erase("actual_ir_diagnostics");
      json.erase("timings");
      for (auto &attempt : *json.getArray("attempts")) attempt.getAsObject()->erase("timings");
      auto stats = llvm::formatv("{0}", llvm::json::Value(std::move(json))).str();
      auto cost = llvm::formatv("{0}", *object->get("finalCost")).str();
      if (expectedIR.empty()) {
        expectedIR = print(*module); expectedStats = stats; expectedCost = cost;
      } else {
        EXPECT_EQ(print(*module), expectedIR);
        EXPECT_EQ(stats, expectedStats);
        EXPECT_EQ(cost, expectedCost);
      }
      PassManager replay(&context);
      buildFriskLayoutPipeline(replay, {}, {}, &report, {}, {}, enabled);
      ASSERT_TRUE(succeeded(replay.run(*module)));
      EXPECT_EQ(print(*module), expectedIR);
      if (enabled)
        EXPECT_EQ(*report.actualIRDiagnostics.getAsObject()->get("resources"),
                  llvm::json::Value(collectLayoutIRResources(*module)));
    }
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(tracePath)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}

TEST_F(LayoutIRDiagnosticsTest, UnsupportedResourceControlFlowDoesNotRejectValidPipeline) {
  context.loadDialect<FriskDialect>();
  constexpr StringLiteral control = R"mlir(
func.func @control(%x: f32, %condition: i1) -> f32 {
  %lo = arith.constant 0 : index
  %hi = arith.constant 3 : index
  %step = arith.constant 1 : index
  %result = scf.for %i = %lo to %hi step %step iter_args(%v = %x) -> (f32) {
    %a = arith.mulf %v, %x : f32
    %next = scf.if %condition -> f32 {
      %b = math.exp %a : f32
      scf.yield %b : f32
    } else {
      scf.yield %a : f32
    }
    scf.yield %next : f32
  }
  return %result : f32
})mlir";
  std::string expectedIR;
  for (bool enabled : {false, true}) {
    auto module = parseSourceString<ModuleOp>(control, &context);
    ASSERT_TRUE(module);
    LayoutStatisticsReport report;
    PassManager pm(&context);
    buildFriskLayoutPipeline(pm, {}, {}, &report, {}, {}, enabled);
    ASSERT_TRUE(succeeded(pm.run(*module)));
    EXPECT_EQ(report.status, "committed");
    if (!enabled) {
      EXPECT_EQ(report.actualIRDiagnostics, llvm::json::Value(nullptr));
      expectedIR = print(*module);
      continue;
    }
    EXPECT_EQ(print(*module), expectedIR);
    auto *diagnostics = report.actualIRDiagnostics.getAsObject();
    ASSERT_TRUE(diagnostics);
    EXPECT_EQ(diagnostics->getObject("arithmetic")->getString("status"), "available");
    auto *resources = diagnostics->getObject("resources");
    ASSERT_TRUE(resources);
    EXPECT_EQ(resources->getString("status"), "unavailable");
    auto *scopes = resources->getArray("scopes");
    ASSERT_TRUE(scopes);
    ASSERT_EQ(scopes->size(), 1u);
    auto *scope = scopes->front().getAsObject();
    EXPECT_EQ(scope->getString("status"), "unavailable");
    EXPECT_EQ(*scope->get("logical_live_carrier_slots_per_thread"), llvm::json::Value(nullptr));
    auto *unavailable = scope->getArray("unavailable_operations");
    ASSERT_TRUE(unavailable);
    ASSERT_FALSE(unavailable->empty());
    EXPECT_EQ(unavailable->front().getAsObject()->getString("opcode"), "scf.for");
    EXPECT_EQ(unavailable->front().getAsObject()->getString("reason"), "unsupported-control-flow");
  }
}

TEST_F(LayoutIRDiagnosticsTest, PipelineFailuresClearStaleDiagnosticsAndRollbackIR) {
  context.loadDialect<FriskDialect>();
  llvm::SmallString<128> directory;
  ASSERT_FALSE(bool(llvm::sys::fs::createUniqueDirectory("frisk-ir-diagnostics-failure", directory)));
  auto badPath = (directory + "/missing/out.json").str();
  auto goodStats = (directory + "/stats.json").str();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (StringRef scenario : {"inference", "stats", "trace", "profile", "trace-options", "profile-options"}) {
    SCOPED_TRACE(scenario.str());
    auto successfulModule = parseSourceString<ModuleOp>(pipelineArithmetic, &context);
    ASSERT_TRUE(successfulModule);
    LayoutStatisticsReport report;
    PassManager successful(&context);
    buildFriskLayoutPipeline(successful, {}, {}, &report, {}, {}, true);
    ASSERT_TRUE(succeeded(successful.run(*successfulModule)));
    ASSERT_TRUE(report.actualIRDiagnostics.getAsObject());
    // Seed the reused report separately: failure must roll back fresh IR, not
    // an already materialized fixed point where an accidental commit is hidden.
    auto module = parseSourceString<ModuleOp>(pipelineArithmetic, &context);
    ASSERT_TRUE(module);
    ASSERT_NE(print(*module), print(*successfulModule));
    if (scenario == "inference")
      (*module->getOps<func::FuncOp>().begin())->setAttr("frisk.target", StringAttr::get(&context, "sm_80"));
    auto before = print(*module);
    PassManager failing(&context);
    LayoutTraceOptions trace;
    std::string stats, profile;
    if (scenario == "stats") stats = badPath;
    if (scenario == "trace") { stats = goodStats; trace.file = badPath; }
    if (scenario == "profile") { stats = goodStats; profile = badPath; }
    if (scenario == "trace-options") { trace.file = badPath; trace.maxEvents = 0; }
    if (scenario == "profile-options") { stats = goodStats; profile = goodStats; }
    buildFriskLayoutPipeline(failing, {}, stats, &report, trace, profile, true);
    EXPECT_TRUE(failed(failing.run(*module)));
    EXPECT_EQ(print(*module), before);
    EXPECT_EQ(report.status, "failed");
    EXPECT_EQ(report.actualIRDiagnostics, llvm::json::Value(nullptr));
    EXPECT_EQ(*report.toJSON().get("actual_ir_diagnostics"), llvm::json::Value(nullptr));
    // Existing publication is per-file, not a multi-file transaction: an
    // earlier statistics file may describe the successfully verified stage.
    if (scenario == "trace" || scenario == "profile") {
      auto buffer = llvm::MemoryBuffer::getFile(goodStats);
      ASSERT_TRUE(bool(buffer));
      auto json = llvm::json::parse((*buffer)->getBuffer());
      ASSERT_TRUE(bool(json));
      EXPECT_TRUE(json->getAsObject()->getObject("actual_ir_diagnostics"));
    }
  }
  EXPECT_FALSE(bool(llvm::sys::fs::remove(goodStats)));
  EXPECT_FALSE(bool(llvm::sys::fs::remove(directory)));
}
TEST_F(LayoutIRDiagnosticsTest, CountsActualOperationsNotUses) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
func.func @arithmetic(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %m = arith.mulf %a, %b : tensor<8x8xf32>
  %e = math.exp %m : tensor<8x8xf32>
  %d = arith.divf %e, %m : tensor<8x8xf32>
  return %d, %m : tensor<8x8xf32>, tensor<8x8xf32>
})mlir", &context);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRArithmetic(*module);
  auto *entries = result.getArray("entries");
  ASSERT_TRUE(entries);
  ASSERT_EQ(entries->size(), 3u);
  for (auto &entry : *entries) {
    ASSERT_TRUE(entry.getAsObject());
    EXPECT_EQ(entry.getAsObject()->getInteger("op_count"), 1);
    EXPECT_EQ(entry.getAsObject()->getInteger("element_ops"), 64);
  }
  arith::MulFOp mul;
  module->walk([&](arith::MulFOp op) { mul = op; });
  OpBuilder builder(mul);
  builder.clone(*mul);
  result = collectLayoutIRArithmetic(*module);
  entries = result.getArray("entries");
  ASSERT_EQ(entries->size(), 3u);
  EXPECT_EQ((*entries)[1].getAsObject()->getString("opcode"), "arith.mulf");
  EXPECT_EQ((*entries)[1].getAsObject()->getInteger("op_count"), 2);
  EXPECT_EQ((*entries)[1].getAsObject()->getInteger("element_ops"), 128);
}

TEST_F(LayoutIRDiagnosticsTest, GroupsResultDtypesAndCountsScalarsAndCasts) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
func.func @typed(%a: tensor<8xf32>, %b: tensor<8xf16>, %i: i32, %x: index, %s: f32) {
  %m = arith.mulf %a, %a : tensor<8xf32>
  %n = arith.mulf %b, %b : tensor<8xf16>
  %c = arith.truncf %a : tensor<8xf32> to tensor<8xf16>
  %cmp = arith.cmpf olt, %a, %a : tensor<8xf32>
  %ii = arith.addi %i, %i : i32
  %xx = arith.addi %x, %x : index
  %ss = math.exp %s : f32
  return
})mlir", &context);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRArithmetic(*module);
  EXPECT_EQ(result.getString("status"), "available");
  auto *entries = result.getArray("entries");
  ASSERT_TRUE(entries);
  ASSERT_EQ(entries->size(), 7u);
  const char *dtypes[] = {"i32", "index", "i1", "f16", "f32", "f16", "f32"};
  const int64_t elements[] = {1, 1, 8, 8, 8, 8, 1};
  for (unsigned i = 0; i < entries->size(); ++i) {
    auto *entry = (*entries)[i].getAsObject();
    EXPECT_EQ(entry->getString("dtype"), dtypes[i]);
    EXPECT_EQ(entry->getInteger("element_ops"), elements[i]);
    EXPECT_EQ(entry->getBoolean("saturated"), false);
  }
}

TEST_F(LayoutIRDiagnosticsTest, UnsupportedComputationsAreOrderedPartialNotZero) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
func.func @partial(%d: tensor<?xf32>, %u: tensor<*xf32>, %v: vector<4xf32>, %i: i32) {
  %a = arith.mulf %d, %d : tensor<?xf32>
  %b = arith.mulf %u, %u : tensor<*xf32>
  %c = math.exp %v : vector<4xf32>
  %sum, %overflow = arith.addui_extended %i, %i : i32, i1
  return
})mlir", &context);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRArithmetic(*module);
  EXPECT_EQ(result.getString("status"), "partial");
  ASSERT_TRUE(result.getArray("entries"));
  EXPECT_TRUE(result.getArray("entries")->empty());
  auto *unavailable = result.getArray("unavailable_operations");
  ASSERT_TRUE(unavailable);
  ASSERT_EQ(unavailable->size(), 4u);
  const char *opcodes[] = {"arith.mulf", "arith.mulf", "math.exp", "arith.addui_extended"};
  const char *reasons[] = {"dynamic-shape", "unsupported-result-type", "unsupported-result-type", "multiple-results"};
  for (unsigned i = 0; i < 4; ++i) {
    EXPECT_EQ((*unavailable)[i].getAsObject()->getString("opcode"), opcodes[i]);
    EXPECT_EQ((*unavailable)[i].getAsObject()->getString("reason"), reasons[i]);
  }
}

TEST_F(LayoutIRDiagnosticsTest, ZeroFirstAndSaturatingProductsAndSums) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
func.func @huge(%z: tensor<9223372036854775807x3x0xf32>,
                %h: tensor<9223372036854775807x3xf32>,
                %s: tensor<9223372036854775807x2xf16>) {
  %zero = arith.addf %z, %z : tensor<9223372036854775807x3x0xf32>
  %huge = arith.mulf %h, %h : tensor<9223372036854775807x3xf32>
  %sum1 = arith.mulf %s, %s : tensor<9223372036854775807x2xf16>
  %sum2 = arith.mulf %s, %s : tensor<9223372036854775807x2xf16>
  return
})mlir", &context);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRArithmetic(*module);
  auto *entries = result.getArray("entries");
  ASSERT_TRUE(entries);
  ASSERT_EQ(entries->size(), 3u);
  EXPECT_EQ((*entries)[0].getAsObject()->getInteger("element_ops"), 0);
  EXPECT_EQ((*entries)[0].getAsObject()->getBoolean("saturated"), false);
  for (unsigned i = 1; i < 3; ++i) {
    EXPECT_EQ(*(*entries)[i].getAsObject()->get("element_ops")->getAsUINT64(), UINT64_MAX);
    EXPECT_EQ((*entries)[i].getAsObject()->getBoolean("saturated"), true);
  }
}

TEST_F(LayoutIRDiagnosticsTest, NestedControlFlowIsStaticAndAnalysisIsPureStable) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
func.func @control(%x: f32, %condition: i1) {
  %lo = arith.constant 0 : index
  %hi = arith.constant 100 : index
  %step = arith.constant 1 : index
  scf.for %i = %lo to %hi step %step {
    %a = arith.mulf %x, %x : f32
    scf.if %condition {
      %b = math.exp %a : f32
    } else {
      %c = math.exp %a : f32
    }
  }
  return
})mlir", &context);
  ASSERT_TRUE(module);
  auto before = print(*module);
  auto result = collectLayoutIRArithmetic(*module);
  EXPECT_EQ(result.getString("counting"), "static-logical-result-elements-once-per-ir-op");
  EXPECT_EQ(result.getString("execution_frequency"), "not-applied");
  EXPECT_EQ(result.getInteger("schema_version"), 1);
  ASSERT_TRUE(result.getArray("entries"));
  ASSERT_EQ(result.getArray("entries")->size(), 2u);
  EXPECT_EQ((*result.getArray("entries"))[1].getAsObject()->getInteger("element_ops"), 2);
  EXPECT_EQ(llvm::formatv("{0}", llvm::json::Value(collectLayoutIRArithmetic(*module))).str(),
            llvm::formatv("{0}", llvm::json::Value(std::move(result))).str());
  EXPECT_EQ(print(*module), before);
}

TEST_F(LayoutIRDiagnosticsTest, NoArithmeticHasAvailableEmptyTable) {
  auto module = parseSourceString<ModuleOp>("func.func @empty() { %c = arith.constant 1 : i32 return }", &context);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRArithmetic(*module);
  EXPECT_EQ(result.getString("status"), "available");
  ASSERT_TRUE(result.getArray("entries"));
  EXPECT_TRUE(result.getArray("entries")->empty());
  ASSERT_TRUE(result.getArray("unavailable_operations"));
  EXPECT_TRUE(result.getArray("unavailable_operations")->empty());
}
} // namespace
