#ifndef FRISK_ANALYSIS_LAYOUTSTATISTICS_H
#define FRISK_ANALYSIS_LAYOUTSTATISTICS_H

#include "llvm/Support/JSON.h"
#include "mlir/Support/LogicalResult.h"
#include <chrono>
#include <map>

namespace mlir::frisk {
class LayoutConstraintGraph;
struct LayoutSolution;
struct LayoutSolveFailure;
enum class LayoutSolveFailureKind;
llvm::StringRef stringifyLayoutSolveFailureKind(LayoutSolveFailureKind kind);
struct LayoutStatistics {
  uint64_t variables = 0, constraints = 0, components = 0;
  uint64_t preparedCandidates = 0, prunedCandidates = 0;
  uint64_t peakVariableDomain = 0, peakComponentCombinations = 0;
  bool componentProductOverflow = false;
  uint64_t peakFrontierStates = 0, expandedStates = 0;
  uint64_t conversionUseEdges = 0, uniqueConversions = 0;
  uint64_t rematerializationUseEdges = 0, uniqueRematerializations = 0;
  uint64_t conversionBytes = 0;
  uint64_t weightedConversionBytes = 0;
  bool weightedConversionBytesUsesProxy = false, weightedConversionBytesSaturated = false;
  uint64_t estimatedBankConflicts = 0, estimatedMemoryTransactions = 0;
  std::chrono::nanoseconds solveTime{0};
};
struct LayoutAttemptStatistics {
  LayoutStatistics statistics;
  bool prepared = false, succeeded = false;
  uint64_t mrvProbes = 0, expandedNodes = 0;
  uint64_t refinementProbes = 0, refinementSweeps = 0, refinementAccepted = 0;
  std::string failureKind, failureVariable, failureConstraint, failureReason;
  llvm::json::Array components;
  std::chrono::nanoseconds propagationTime{0};
  std::chrono::nanoseconds verificationTime{0};
  std::string kind = "initial";
};
struct LayoutStatisticsReport {
  /// Opt-in, successfully verified post-optimization IR only; null otherwise.
  /// An output-file failure clears the caller's report, but does not retract an
  /// earlier independently atomically published statistics file.
  llvm::json::Value actualIRDiagnostics = nullptr;
  std::string costModelID = "unversioned-target";
  bool staticTripCountFrequency = false;
  std::string status = "failed", failureKind, failureReason, failureVariable, failureConstraint;
  std::vector<LayoutAttemptStatistics> attempts;
  llvm::json::Array relaxations;
  std::map<std::string, uint64_t> timings;
  uint64_t rawGeneratedProposals = 0;
  uint64_t instructionCombinations = 0, instructionFragmentPrechecks = 0;
  llvm::json::Object toJSON() const;
};
/// Snapshot the deduplicated domain before any attempt-local propagation.
void prepareLayoutStatistics(const LayoutConstraintGraph &, LayoutAttemptStatistics &);
void finishLayoutStatistics(const LayoutConstraintGraph &, const LayoutSolveFailure &,
                            const LayoutSolution *, LayoutAttemptStatistics &);
void recordLayoutFailure(const LayoutConstraintGraph &, const LayoutSolveFailure &,
                         LayoutStatisticsReport &);
/// Create a temporary sibling and atomically replace only after a complete write.
LogicalResult writeLayoutStatistics(llvm::StringRef path,
                                    const LayoutStatisticsReport &, std::string &error);
using LayoutClock = std::chrono::steady_clock;
inline uint64_t layoutElapsedNS(LayoutClock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(LayoutClock::now() - start).count();
}
} // namespace mlir::frisk
#endif
