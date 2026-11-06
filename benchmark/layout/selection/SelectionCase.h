#ifndef FRISK_BENCHMARK_SELECTION_CASE_H
#define FRISK_BENCHMARK_SELECTION_CASE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::frisk::selection {
enum class MmaMode { SS, RS };
enum class Consumer { Store, Reduce0, Reduce1, Fixed0, Fixed1, Local };
struct CaseSpec {
  std::string id = "C1";
  int64_t m = 128, n = 128, k = 16, threads = 256;
  MmaMode mode = MmaMode::SS;
  std::string policy = "Auto";
  Consumer consumer = Consumer::Store;
  std::optional<std::pair<int64_t, int64_t>> consumerGrid;
};
std::vector<CaseSpec> nativeCases();
llvm::StringRef modeName(MmaMode);
llvm::StringRef consumerName(Consumer);
llvm::json::Object caseJSON(const CaseSpec &);
LogicalResult validateCase(const CaseSpec &, std::string &error);
OwningOpRef<ModuleOp> buildFixture(MLIRContext &, const CaseSpec &,
                                 std::string &error);
LogicalResult verifyOutputABI(ModuleOp, const CaseSpec &, std::string &error);
std::string printModule(ModuleOp);
} // namespace mlir::frisk::selection
#endif
