#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "llvm/Support/SHA256.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
constexpr uint64_t envelopeBytes = 4096;
// The maximum-envelope test covers all current solver settings and reserves
// the remaining decimal growth of the four event/contribution counters.
void increment(uint64_t &value) { if (value != UINT64_MAX) ++value; }
// Serialization itself cannot allocate the size of an oversized attribute.
// The caller owns the transient JSON payload; retained storage is cap-bounded.
class BoundedStream : public llvm::raw_ostream {
public:
  explicit BoundedStream(uint64_t limit) : limit(limit) { SetUnbuffered(); }
  std::string text;
  bool overflow = false;
private:
  uint64_t limit;
  void write_impl(const char *ptr, size_t size) override {
    if (overflow) return;
    if (size > limit - text.size()) { overflow = true; return; }
    text.append(ptr, size);
  }
  uint64_t current_pos() const override { return text.size(); }
};
std::string normalizedPath(StringRef path) {
  llvm::SmallString<256> absolute(path);
  if (llvm::sys::fs::make_absolute(absolute)) return path.str();
  // Resolve the original parent before collapsing '..': it must traverse
  // symlinks using filesystem semantics, even for a not-yet-created output.
  llvm::SmallString<256> parent;
  if (!llvm::sys::fs::real_path(llvm::sys::path::parent_path(absolute), parent)) {
    llvm::sys::path::append(parent, llvm::sys::path::filename(absolute));
    return parent.str().str();
  }
  llvm::sys::path::remove_dots(absolute, true);
  return absolute.str().str();
}
} // namespace

LogicalResult LayoutTraceOptions::validate(StringRef statsFile,
                                           std::string &error) const {
  error.clear();
  if (maxEvents == 0 || maxBytes < envelopeBytes) {
    error = "trace-max-events must be positive and trace-max-bytes must be at least 4096";
    return failure();
  }
  if (!file.empty() && !statsFile.empty() &&
      (normalizedPath(file) == normalizedPath(statsFile) ||
       llvm::sys::fs::equivalent(file, statsFile))) {
    error = "trace-file and stats-file must name different output paths";
    return failure();
  }
  return success();
}

LayoutTrace::LayoutTrace(LayoutTraceOptions options) : options(std::move(options)) {}
bool LayoutTrace::enabled() const { return !options.file.empty(); }
bool LayoutTrace::acceptingEvents() const {
  return enabled() && !truncated && events.size() < options.maxEvents &&
         options.maxBytes > envelopeBytes + eventBytes;
}
bool LayoutTrace::recordEvent(llvm::function_ref<llvm::json::Object()> build) {
  if (!enabled()) return false;
  increment(eventsSeen);
  if (acceptingEvents()) {
    uint64_t remaining = options.maxBytes - envelopeBytes - eventBytes;
    BoundedStream stream(remaining - 1); // Each event also reserves a comma.
    auto event = build();
    event["attempt"] = costScope;
    stream << llvm::json::Value(std::move(event));
    if (!stream.overflow) {
      eventBytes += stream.text.size() + 1;
      events.push_back(std::move(stream.text));
      return true;
    }
  }
  truncated = true;
  increment(eventsOmitted);
  return false;
}
llvm::json::Object mlir::frisk::layoutCostJSON(const CostEstimate &cost) {
  const auto &c = cost.cost;
  llvm::json::Object result{{"cost", llvm::json::Array{c.instructionPathAndWork,
      c.memoryTransactions, c.bankConflictDegree, c.conversionBytesAndSync,
      c.spillRiskAndRegisters, c.sharedBytesAndOccupancy, c.replication,
      c.codeSize, c.deterministicTieBreak}}, {"saturated", cost.saturated}};
  if(!cost.features.empty()) {
    llvm::json::Object features;for(const auto &[name,count]:cost.features)features[name]=count;
    result["features"]=std::move(features);
  }
  return result;
}
void LayoutTrace::recordCost(StringRef kind, uint64_t stableOwner,
                             const CostEstimate &cost) {
  if (!enabled()) return;
  increment(contributionsSeen);
  if (!recordEvent([&] {
        return llvm::json::Object{{"event", "cost-contribution"},
            {"kind", kind}, {"owner", stableOwner}, {"costScope", costScope},
            {"estimate", layoutCostJSON(cost)}};
      })) {
    increment(contributionsOmitted);
    // The mandatory envelope has a fixed reserve. Raw feature ledgers belong
    // only to bounded events, never to an unbounded omission accumulator.
    omitted = addLayoutCosts(omitted, CostEstimate{cost.cost, cost.saturated});
  }
}
void LayoutTrace::beginCostScope(uint64_t attemptID) {
  if (!enabled()) return;
  costScope = attemptID;
  contributionsSeen = contributionsOmitted = 0;
  omitted = {};
  summary.hasFinalCost = false;
}
std::string LayoutTrace::candidateID(Attribute attribute) {
  auto text = layoutCandidateKey(attribute);
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)), true);
}
void LayoutTrace::beginAttempt(StringRef kind) {
  if (!enabled()) return;
  beginCostScope(++summary.attempts);
  summary.analysisStatus = "running";
  summary.verificationStatus = "not-started";
  summary.failureKind.clear();
  summary.failureReason.clear();
  recordEvent([&] { return llvm::json::Object{{"event", "attempt-start"}, {"kind", kind}}; });
}
void LayoutTrace::finishAttempt(const LayoutSolveFailure &failure,
                                const LayoutSolution *solution) {
  if (!enabled()) return;
  auto completed = summary;
  completed.hasFinalCost = solution != nullptr;
  completed.analysisStatus = solution ? "solved" : "failed";
  completed.failureKind = solution ? "" : stringifyLayoutSolveFailureKind(failure.kind).str();
  completed.failureReason = solution ? "" : failure.reason;
  if (solution) completed.finalCost = {solution->cost.cost, solution->cost.saturated};
  setSummary(completed);
  recordEvent([&] { return llvm::json::Object{{"event", "attempt-end"},
      {"success", solution != nullptr}, {"failureKind", stringifyLayoutSolveFailureKind(failure.kind)},
      {"reason", failure.reason}}; });
}
void LayoutTrace::recordDomains(const LayoutConstraintGraph &graph, StringRef stage) {
  recordEvent([&] {
    llvm::json::Array domains;
    for (const auto &var : graph.getVariables()) {
      llvm::json::Array candidates;
      for (const auto &candidate : var.candidates)
        candidates.push_back(llvm::json::Object{{"id", candidateID(candidate.value)},
            {"attribute", layoutCandidateKey(candidate.value)}, {"provenance", candidate.provenance}});
      domains.push_back(llvm::json::Object{{"variable", var.id}, {"candidates", std::move(candidates)}});
    }
    return llvm::json::Object{{"event", "domains"}, {"stage", stage}, {"domains", std::move(domains)}};
  });
}
void LayoutTrace::recordGraph(const LayoutConstraintGraph &graph) {
  recordEvent([&] {
    llvm::json::Array variables, constraints, provenance, regions;
    for (const auto &v : graph.getVariables())
      variables.push_back(llvm::json::Object{{"id", v.id}, {"name", v.stableName},
          {"kind", v.kind == LayoutKind::Storage ? "storage" : "distributed"}});
    for (const auto &p : graph.getProvenances()) {
      std::string location;
      if (p.source) { llvm::raw_string_ostream os(location); p.source->getLoc().print(os); }
      provenance.push_back(llvm::json::Object{{"id", p.id}, {"rule", p.rule},
          {"reason", p.reason}, {"location", location},
          {"parent", p.parent ? llvm::json::Value(*p.parent) : llvm::json::Value(nullptr)}});
    }
    for (const auto &c : graph.getConstraints()) {
      llvm::json::Array vars, supports;
      for (auto id : c.vars) vars.push_back(id);
      auto support = [&](ArrayRef<Attribute> attrs, Attribute binding) {
        llvm::json::Array ids;
        for (auto attr : attrs) ids.push_back(candidateID(attr));
        supports.push_back(llvm::json::Object{{"candidates", std::move(ids)},
            {"binding", layoutCandidateKey(binding)}, {"status", "Proven"}});
      };
      if (c.instruction) for (const auto &t : c.instruction->tuples) support(t.encodings, t.binding);
      if (c.reduction) for (const auto &p : c.reduction->pairs)
        support({p.sourceEncoding, p.resultEncoding}, p.binding);
      constraints.push_back(llvm::json::Object{{"id", c.id}, {"kind", stringifyConstraintKind(c.kind)},
          {"strength", stringifyConstraintStrength(c.strength)}, {"variables", std::move(vars)},
          {"provenance", c.provenance}, {"use", c.stableUseKey}, {"supports", std::move(supports)}});
    }
    for (const auto &e : graph.getRegionEdges())
      regions.push_back(llvm::json::Object{{"source", e.source}, {"target", e.target},
          {"constraint", e.constraint}, {"kind", stringifyRegionLayoutEdgeKind(e.kind)}, {"key", e.stableKey}});
    return llvm::json::Object{{"event", "graph"}, {"variables", std::move(variables)},
        {"constraints", std::move(constraints)}, {"provenance", std::move(provenance)}, {"regions", std::move(regions)}};
  });
}
void LayoutTrace::recordRejection(LayoutVarID var, Attribute candidate,
    LayoutConstraintID constraint, StringRef stage, StringRef status,
    StringRef reason, ArrayRef<int64_t> counterexample) {
  recordEvent([&] {
    llvm::json::Array coordinate;
    for (auto i : counterexample) coordinate.push_back(i);
    return llvm::json::Object{{"event", "candidate-rejected"}, {"variable", var},
        {"candidate", candidateID(candidate)}, {"constraint", constraint},
        {"stage", stage}, {"status", status}, {"reason", reason}, {"counterexample", std::move(coordinate)}};
  });
}
void LayoutTrace::recordSolution(const LayoutConstraintGraph &graph, const LayoutSolution &solution) {
  recordEvent([&] {
    llvm::json::Array assignments, contracts, conversions;
    for (const auto &v : graph.getVariables())
      if (auto a = solution.assignments.lookup(v.id))
        assignments.push_back(llvm::json::Object{{"variable", v.id}, {"candidate", candidateID(a)}});
    for (const auto &c : graph.getConstraints()) {
      auto a = solution.instructionBindings.lookup(c.id);
      if (!a) a = solution.reductionBindings.lookup(c.id);
      if (a) contracts.push_back(llvm::json::Object{{"constraint", c.id}, {"binding", layoutCandidateKey(a)}});
    }
    for (const auto &e : solution.conversions) {
      llvm::json::Array operations;
      for (auto *op : e.rematerialization) operations.push_back(op->getName().getStringRef());
      conversions.push_back(llvm::json::Object{{"constraint", e.constraint}, {"leader", e.sharedLeader},
          {"resolution", static_cast<int>(e.resolution)}, {"exchange", static_cast<int>(e.exchange)},
          {"source", candidateID(e.sourceEncoding)}, {"target", candidateID(e.targetEncoding)},
          {"bytes", e.bytes}, {"synchronization", e.synchronizationCost}, {"loopDepth", e.loopDepth},
          {"frequencyFactor", e.frequencyFactor}, {"frequencyUsesProxy", e.frequencyUsesProxy},
          {"frequencySaturated", e.frequencySaturated},
          {"boundaryFrequency", e.staticTripCountFrequency ? "static-trip-count" : "legacy"},
          {"provenance", e.costProvenance}, {"rematerialization", std::move(operations)}});
    }
    return llvm::json::Object{{"event", "solution"}, {"assignments", std::move(assignments)},
        {"contracts", std::move(contracts)}, {"conversions", std::move(conversions)}, {"cost", layoutCostJSON(solution.cost)}};
  });
}
void LayoutTrace::setSummary(const LayoutTraceSummary &value) {
  if (!enabled()) return;
  // Do not copy oversized input strings before bounding them.
  auto bounded = [&](const std::string &text, size_t limit) {
    if (text.size() > limit) {
      summaryTextTruncated = true;
      // Back up to the beginning of a partial UTF-8 codepoint.
      while (limit && (static_cast<unsigned char>(text[limit]) & 0xc0) == 0x80)
        --limit;
    }
    return text.substr(0, limit);
  };
  summary.target = bounded(value.target, 64);
  summary.costCoverage = bounded(value.costCoverage, 128);
  summary.costModelID = normalizeLayoutCostModelID(value.costModelID).str();
  summary.costProfileSHA256 = value.costProfileSHA256.size() == 64 &&
      llvm::all_of(value.costProfileSHA256, [](unsigned char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
      }) ? value.costProfileSHA256 : "";
  summary.staticTripCountFrequency = value.staticTripCountFrequency;
  summary.analysisStatus = bounded(value.analysisStatus, 32);
  summary.verificationStatus = bounded(value.verificationStatus, 32);
  summary.failureKind = bounded(value.failureKind, 64);
  summary.failureReason = bounded(value.failureReason, 128);
  summary.attempts = value.attempts;
  summary.exactCombinationLimit = value.exactCombinationLimit;
  summary.beamWidth = value.beamWidth;
  summary.maxExpandedStates = value.maxExpandedStates;
  summary.refineBeamSolution = value.refineBeamSolution;
  summary.maxRefinementProbes = value.maxRefinementProbes;
  summary.hasFinalCost = value.hasFinalCost;
  summary.finalCost = {value.finalCost.cost, value.finalCost.saturated};
}
void LayoutTrace::setSolverOptions(const SolverOptions &value) {
  if (!enabled()) return;
  summary.exactCombinationLimit = value.exactCombinationLimit;
  summary.beamWidth = value.beamWidth;
  summary.maxExpandedStates = value.maxExpandedStates;
  summary.refineBeamSolution = value.refineBeamSolution;
  summary.maxRefinementProbes = value.maxRefinementProbes;
}
std::string LayoutTrace::serialize() const {
  if (!enabled()) return {};
  llvm::json::Object envelope{
      {"schema_version", 1}, {"target", summary.target},
      {"cost_model_id", summary.costModelID},
      {"boundaryFrequency", summary.staticTripCountFrequency ? "static-trip-count" : "legacy"},
      {"cost_coverage", summary.costCoverage},
      {"analysis_status", summary.analysisStatus},
      {"verification_status", summary.verificationStatus},
      {"failure_kind", summary.failureKind}, {"failure_reason", summary.failureReason},
      {"attempts", summary.attempts},
      {"solver_options", llvm::json::Object{
          {"exactCombinationLimit", summary.exactCombinationLimit},
          {"beamWidth", summary.beamWidth},
          {"maxExpandedStates", summary.maxExpandedStates},
          {"refineBeamSolution", summary.refineBeamSolution},
          {"maxRefinementProbes", summary.maxRefinementProbes}}},
      {"trace_max_events", options.maxEvents}, {"trace_max_bytes", options.maxBytes},
      {"truncated", truncated}, {"summaryTextTruncated", summaryTextTruncated},
      {"eventsSeen", eventsSeen}, {"eventsOmitted", eventsOmitted},
      {"contributionsSeen", contributionsSeen}, {"contributionsOmitted", contributionsOmitted},
      {"costScope", costScope},
      {"costExplanationComplete", summary.hasFinalCost && contributionsOmitted == 0},
      {"finalCost", summary.hasFinalCost ? llvm::json::Value(layoutCostJSON(summary.finalCost))
                                         : llvm::json::Value(nullptr)},
      {"omittedCost", layoutCostJSON(omitted)}};
  if (!summary.costProfileSHA256.empty()) envelope["cost_profile_sha256"] = summary.costProfileSHA256;
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::json::Value(std::move(envelope));
  // Append already serialized whole events; avoid a second JSON object tree.
  result.pop_back();
  stream << ",\"events\":[";
  for (auto [index, event] : llvm::enumerate(events)) {
    if (index) stream << ',';
    stream << event;
  }
  stream << "]}\n";
  assert(result.size() <= options.maxBytes && "trace envelope reserve too small");
  return result;
}
LogicalResult LayoutTrace::publish(std::string &error) const {
  error.clear();
  if (!enabled()) return success();
  if (failed(options.validate("", error))) return failure();
  auto temp = llvm::sys::fs::TempFile::create(options.file + ".tmp-%%%%%%");
  if (!temp) { error = llvm::toString(temp.takeError()); return failure(); }
  llvm::raw_fd_ostream stream(temp->FD, false);
  stream << serialize();
  stream.flush();
  if (stream.has_error()) {
    error = stream.error().message();
    stream.clear_error();
    llvm::consumeError(temp->discard());
    return failure();
  }
  // Close before publishing, and never use keep(destination): LLVM may copy
  // on rename failure, which would destroy the atomic-publication guarantee.
  std::string temporaryPath = temp->TmpName;
  if (auto e = temp->keep()) {
    error = llvm::toString(std::move(e));
    llvm::sys::fs::remove(temporaryPath);
    return failure();
  }
  if (auto e = llvm::sys::fs::rename(temporaryPath, options.file)) {
    error = e.message();
    llvm::sys::fs::remove(temporaryPath);
    return failure();
  }
  return success();
}
