#ifndef FRISK_ANALYSIS_LAYOUTTRACE_H
#define FRISK_ANALYSIS_LAYOUTTRACE_H

#include "Dialect/Frisk/Analysis/LayoutCostObserver.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/JSON.h"

namespace mlir::frisk {
struct LayoutSolution;
struct LayoutSolveFailure;
struct SolverOptions;
struct LayoutTraceOptions {
  std::string file;
  uint64_t maxEvents = 4096;
  uint64_t maxBytes = 8388608;
  /// Call before any IR modification. Rejects invalid limits and normalized
  /// identical paths (including existing hard links/symlinks).
  LogicalResult validate(StringRef statsFile, std::string &error) const;
};

/// Fixed-size mandatory summary. Text is bounded on setSummary (target 64,
/// coverage 128, statuses 32, failure kind 64/reason 128 bytes); complete text
/// belongs in a lazy event. UTF-8 codepoints are never split at these bounds.
/// Model identity is validated as a complete 1..64-byte ASCII token, never
/// truncated; invalid identities are explicitly marked unversioned.
/// States describe analysis/verification, never promise that IR was committed.
struct LayoutTraceSummary {
  std::string target, costCoverage;
  std::string costModelID = "unversioned-target";
  std::string costProfileSHA256;
  // Mandatory summary costs retain only nine score fields and saturation.
  // Raw feature ledgers are retained exclusively in byte-bounded events.
  bool staticTripCountFrequency = false;
  std::string analysisStatus = "not-started";
  std::string verificationStatus = "not-started";
  std::string failureKind, failureReason;
  uint64_t attempts = 0;
  // Mandatory fixed-width settings survive all optional-event truncation.
  uint64_t exactCombinationLimit = 256;
  unsigned beamWidth = 32;
  uint64_t maxExpandedStates = 65536;
  bool refineBeamSolution = true;
  uint64_t maxRefinementProbes = 256;
  bool hasFinalCost = false;
  CostEstimate finalCost;
};

/// Optional schema-v1 recorder; neither owns IR nor computes costs/proofs.
/// Events are JSON objects supplied in deterministic order with stable IDs.
/// A 4096-byte envelope reserve bounds the mandatory summary, counters, costs,
/// JSON escaping and newline even after all optional events are dropped.
/// The first overflow permanently closes optional recording; later lazy
/// builders are not invoked, but counts and omitted costs still accumulate.
class LayoutTrace : public LayoutCostObserver {
public:
  explicit LayoutTrace(LayoutTraceOptions options = {});
  bool enabled() const;
  bool acceptingEvents() const;
  bool recordEvent(llvm::function_ref<llvm::json::Object()> build);
  /// Start each attempt with a unique monotonically increasing ID (default
  /// scope is 0). Historical cost events remain tagged with their old scope.
  /// omittedCost/contributionsSeen/contributionsOmitted cover ONLY this scope;
  /// eventsSeen/eventsOmitted remain global. Optional truncation never resets.
  /// Observe only existing final component evaluations, not search proposals.
  void beginCostScope(uint64_t attemptID);
  void recordCost(StringRef kind, uint64_t stableOwner,
                  const CostEstimate &cost) override;
  void setSummary(const LayoutTraceSummary &summary);
  /// Initialize before transaction work or public solve/coordinator validation.
  void setSolverOptions(const SolverOptions &options);
  const LayoutTraceSummary &getSummary() const { return summary; }
  void beginAttempt(StringRef kind);
  void finishAttempt(const LayoutSolveFailure &failure,
                     const LayoutSolution *solution);
  void recordGraph(const LayoutConstraintGraph &graph);
  void recordDomains(const LayoutConstraintGraph &graph, StringRef stage);
  void recordSolution(const LayoutConstraintGraph &graph, const LayoutSolution &solution);
  void recordRejection(LayoutVarID var, Attribute candidate,
                       LayoutConstraintID constraint, StringRef stage,
                       StringRef status, StringRef reason,
                       ArrayRef<int64_t> counterexample = {});
  /// Content-addressed attribute identity, independent of mutable domain order.
  /// Only call inside a lazy event builder.
  static std::string candidateID(Attribute attribute);
  const CostEstimate &omittedCost() const { return omitted; }
  /// Stable compact JSON including newline; empty when disabled.
  std::string serialize() const;
  /// Temporary sibling + atomic rename. Does not alter existing file on error.
  LogicalResult publish(std::string &error) const;

private:
  LayoutTraceOptions options;
  LayoutTraceSummary summary;
  SmallVector<std::string> events;
  uint64_t eventBytes = 0, eventsSeen = 0, eventsOmitted = 0;
  uint64_t contributionsSeen = 0, contributionsOmitted = 0;
  uint64_t costScope = 0;
  bool truncated = false, summaryTextTruncated = false;
  CostEstimate omitted;
};
llvm::json::Object layoutCostJSON(const CostEstimate &cost);
} // namespace mlir::frisk
#endif
