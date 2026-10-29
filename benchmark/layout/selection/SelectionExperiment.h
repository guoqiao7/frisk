#ifndef FRISK_BENCHMARK_SELECTION_EXPERIMENT_H
#define FRISK_BENCHMARK_SELECTION_EXPERIMENT_H
#include "SelectionCase.h"

namespace mlir::frisk::selection {
struct ExperimentOptions {
  CaseSpec spec;
  std::string phase = "native", outputDir, binaryPath;
  std::vector<std::string> command;
  bool listCases = false;
  bool oracle = false;
  std::optional<std::pair<int64_t, int64_t>> restrictGrid;
};
struct ExperimentResult {
  std::string status, failureKind, error;
  bool fatal = false;
  llvm::json::Object record;
};
LogicalResult parseOptions(llvm::ArrayRef<std::string> args,
                           ExperimentOptions &, std::string &error);
ExperimentResult runExperiment(MLIRContext &, const ExperimentOptions &);
std::string sha256(llvm::StringRef);
/// Only documented support/search failures are nonfatal experimental outcomes.
bool isFatalFailure(llvm::StringRef kind);
struct ExperimentFailure {
  std::string kind, reason;
  bool fatal = false;
};
/// Shared native/local boundary for upstream typed failures and diagnostics.
ExperimentFailure classifyFailure(llvm::StringRef kind, llvm::StringRef reason,
                                 llvm::StringRef diagnostics);
/// Publish real JSON. On I/O failure both result fields and returned JSON are
/// refreshed to the same fatal result-write outcome, even if no file is writable.
LogicalResult publishResult(ExperimentResult &, llvm::StringRef path);
/// Project one exact attempt's full graph/supports and prepared domains from a
/// nontruncated trace. Only attempt numbers are removed, never support data.
FailureOr<llvm::json::Object> preparedTraceProjection(
    const llvm::json::Object &trace, uint64_t attempt);
/// The caller owns fixture, diagnostics and artifact publication. This path
/// never enters the native normalization boundary for raw fixed consumers.
void runControlledExperiment(ModuleOp, const ExperimentOptions &,
                             ExperimentResult &, std::string &diagnostics);
} // namespace mlir::frisk::selection
#endif
