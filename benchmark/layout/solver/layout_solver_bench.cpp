#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace mlir::frisk;
static std::string digest(StringRef input) {
  llvm::SHA256 sha;
  sha.update(input);
  return llvm::toHex(sha.final(), true);
}
static std::string corpus(unsigned operations) {
  // Ten actual dependent tensor operations per component, never isolated constants.
  std::string ir = "module {\n";
  for (unsigned begin = 0; begin < operations; begin += 10) {
    ir += "func.func @chain" + std::to_string(begin) + "(%x:tensor<4xf32>) -> tensor<4xf32> {\n";
    for (unsigned i = 0; i < 10; ++i) {
      auto previous = i ? "%v" + std::to_string(i - 1) : "%x";
      ir += "%v" + std::to_string(i) + " = arith.addf " + previous + ", " + previous + " : tensor<4xf32>\n";
    }
    ir += "return %v9 : tensor<4xf32>\n}\n";
  }
  return ir + "}\n";
}
static llvm::json::Object summary(const llvm::json::Array &samples) {
  std::vector<int64_t> sorted;
  for (const auto &sample : samples) sorted.push_back(*sample.getAsInteger());
  llvm::sort(sorted);
  return llvm::json::Object{{"median_ns", (sorted[14] + sorted[15]) / 2.0}, {"p95_ns", sorted[28]}};
}
int main(int argc, char **argv) {
  MLIRContext context;
  context.disableMultithreading();
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect, tensor::TensorDialect>();
  auto target = createSM90LayoutTarget();
  llvm::json::Array cases;
  auto executable = llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main));
  auto binary = llvm::MemoryBuffer::getFile(executable);
  if (!binary) { llvm::errs() << "cannot hash measured executable\n"; return 1; }
  auto binaryHash = digest((*binary)->getBuffer());
  for (unsigned operations : {10, 50, 100, 250}) {
    llvm::errs() << "layout benchmark connected_" << operations << " starting (5 warmup + 30 samples)\n";
    auto input = corpus(operations);
    auto original = parseSourceString<ModuleOp>(input, &context);
    if (!original) return 1;
    unsigned counted = 0;
    original->walk([&](arith::AddFOp) { ++counted; });
    if (counted != operations) return 1;
    std::string stableHash;
    llvm::json::Array collection, preparation, solving, pipelineSamples;
    llvm::json::Object finalStatistics;
    llvm::json::Array finalComponents;
    for (unsigned iteration = 0; iteration < 35; ++iteration) {
      if (iteration % 5 == 0) llvm::errs() << "  iteration " << iteration << "/35\n";
      LayoutStatisticsReport report;
      auto graph = collectLayoutConstraints(*original, *target,
          LayoutCollectionMode::InitializeForRelaxation, &report);
      if (failed(graph)) return 1;
      auto solveStart = LayoutClock::now();
      SmallVector<RelaxationRecord, 0> relaxations;
      auto solution = attemptControlledRelaxation(*graph, *target, {}, relaxations, nullptr, &report);
      uint64_t solveNS = layoutElapsedNS(solveStart);
      if (failed(solution) || failed(verifySolvedLayoutGraph(*graph, *solution, *target, original->getLoc()))) return 1;
      OwningOpRef<ModuleOp> staged(cast<ModuleOp>(original->clone()));
      LayoutStatisticsReport pipelineReport;
      PassManager pm(&context);
      buildFriskLayoutPipeline(pm, {}, {}, &pipelineReport);
      auto pipelineStart = LayoutClock::now();
      if (failed(pm.run(*staged))) return 1;
      uint64_t pipelineNS = layoutElapsedNS(pipelineStart);
      // Rendering and hashing are intentionally outside every timed interval.
      std::string output;
      llvm::raw_string_ostream stream(output);
      staged->print(stream);
      auto hash = digest(output);
      if (stableHash.empty()) stableHash = hash;
      else if (stableHash != hash) { llvm::errs() << "nondeterministic solution\n"; return 1; }
      if (iteration < 5) continue;
      collection.push_back(report.timings.at("collection_ns"));
      preparation.push_back(report.timings.at("preparation_ns"));
      solving.push_back(solveNS);
      pipelineSamples.push_back(pipelineNS);
      auto json = report.toJSON();
      auto &attempt = *json.getArray("attempts")->back().getAsObject();
      finalStatistics = *attempt.getObject("statistics");
      finalComponents = *attempt.getArray("components");
    }
    llvm::json::Object summaries{{"collection", summary(collection)}, {"preparation", summary(preparation)},
      {"solve", summary(solving)}, {"layout_pipeline", summary(pipelineSamples)}};
    cases.push_back(llvm::json::Object{{"id", "connected_" + std::to_string(operations)},
      {"operations", operations}, {"status", "success"}, {"warmup", 5}, {"iterations", 30},
      {"input_sha256", digest(input)}, {"solution_hash", stableHash}, {"deterministic", true},
      {"statistics", std::move(finalStatistics)}, {"components", std::move(finalComponents)},
      {"summary", std::move(summaries)},
      {"samples", llvm::json::Object{{"collection_ns", std::move(collection)},
        {"preparation_ns", std::move(preparation)}, {"solve_ns", std::move(solving)},
        {"layout_pipeline_ns", std::move(pipelineSamples)}}}});
  }
  llvm::json::Object output{{"schema_version", 1}, {"kind", "solver"}, {"status", "success"},
    {"environment", llvm::json::Object{{"cpu", llvm::sys::getHostCPUName()},
      {"build_type", FRISK_BENCH_BUILD_TYPE}, {"compiler", __VERSION__},
      {"source_digest", FRISK_BENCH_SOURCE_DIGEST}, {"config_digest", FRISK_BENCH_CONFIG_DIGEST},
      {"binary_sha256", binaryHash}}},
    {"options", llvm::json::Object{{"exact_combination_limit", 256}, {"beam_width", 32},
                                  {"max_expanded_states", 65536}}}, {"cases", std::move(cases)}};
  llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(output)));
}
