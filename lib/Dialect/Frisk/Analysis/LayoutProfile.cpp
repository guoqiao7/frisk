#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>

using namespace mlir;
using namespace mlir::frisk;
namespace {
thread_local LayoutProfile *activeProfile = nullptr;
constexpr const char *names[] = {
  "address_enumeration", "execution_enumeration", "access_cost",
  "conversion_classification", "conversion_choice", "instruction_support",
  "instruction_contract", "relation_query", "relation_cache", "alias_pair_cache",
  "footprint_cache", "access_cost_cache", "static_cost_validation_cache",
  "static_cost_validation", "demand_footprint_cache"};
static_assert(std::size(names) == static_cast<size_t>(LayoutProfileKind::Count));
void increment(uint64_t &value) { if (value != UINT64_MAX) ++value; }
} // namespace

LayoutProfile *LayoutProfile::current() { return activeProfile; }
LayoutProfile::Invocation::Invocation(LayoutProfile *profile)
    : previous(activeProfile) {
  activeProfile = profile && profile->enabled() ? profile : nullptr;
}
LayoutProfile::Invocation::~Invocation() { activeProfile = previous; }
LayoutProfile::Scope::Scope(LayoutProfileKind kind) {
  if (!activeProfile) return;
  metric = &activeProfile->metrics[static_cast<size_t>(kind)];
  increment(metric->count);
  start = std::chrono::steady_clock::now();
}
LayoutProfile::Scope::~Scope() {
  if (!metric) return;
  auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start).count();
  uint64_t ns = elapsed > 0 ? uint64_t(elapsed) : 0;
  metric->inclusiveNS += std::min(ns, UINT64_MAX - metric->inclusiveNS);
}
void LayoutProfile::cacheLookup(LayoutProfileKind kind, bool hit) {
  if (!activeProfile) return;
  auto &metric = activeProfile->metrics[static_cast<size_t>(kind)];
  increment(hit ? metric.hits : metric.misses);
}
void LayoutProfile::cacheInsertion(LayoutProfileKind kind) {
  if (activeProfile)
    increment(activeProfile->metrics[static_cast<size_t>(kind)].entries);
}
void LayoutProfile::cacheSize(LayoutProfileKind kind, uint64_t size) {
  if (activeProfile) {
    auto &metric = activeProfile->metrics[static_cast<size_t>(kind)];
    metric.peakEntries = std::max(metric.peakEntries, size);
  }
}
const LayoutProfileMetric &LayoutProfile::get(LayoutProfileKind kind) const {
  return metrics[static_cast<size_t>(kind)];
}
llvm::json::Object LayoutProfile::toJSON(StringRef status) const {
  llvm::json::Object values;
  for (size_t i = 0; i < metrics.size(); ++i) {
    const auto &m = metrics[i];
    values[names[i]] = llvm::json::Object{{"count", m.count},
        {"inclusive_ns", m.inclusiveNS}, {"hits", m.hits},
        {"misses", m.misses}, {"entries", m.entries},
        {"peak_entries", m.peakEntries}};
  }
  return llvm::json::Object{{"version", 1}, {"status", status}, {"timing_kind", "inclusive"},
          {"entries_kind", "insertions"},
          {"timing_note", "Nested inclusive times overlap and are not additive: access_cost includes execution_enumeration/footprint_cache; relation_query and relation_cache include instruction_support/alias_pair_cache; footprint_cache includes address_enumeration. Counts measure actual entries, including cache hits; misses measure cache computation, entries successful insertions. access_cost_cache stores only successful unweighted terms and replays observer contributions; static_cost_validation_cache stores only successful immutable type/attribute validations. Both are graph-local, capped at 4096 entries each (lower limits/disabled operation may bypass storage); their insertions are not live/peak size. static_cost_validation counts actual decoder calls after misses or cache bypass. demand_footprint_cache memoizes the selected memory-space footprint after exact per-warp integer-translation normalization; it is builder-local and capped at min(256, graph cache bound), with graph cache disabling honored. Its timing covers normalization and actual misses, including disabled operation. peak_entries records the maximum explicitly observed live size; currently only demand_footprint_cache supplies this observation, so zero on other metrics is not a measured zero."},
          {"metrics", std::move(values)}};
}
std::string LayoutProfile::serialize(StringRef status) const {
  if (!enabled()) return {};
  return llvm::formatv("{0:2}\n", llvm::json::Value(toJSON(status))).str();
}
LogicalResult LayoutProfile::validate(StringRef path, StringRef statsFile,
                                     StringRef traceFile, std::string &error) {
  error.clear();
  if (path.empty()) return success();
  // Reuse trace's filesystem-aware normalized/identity alias semantics rather
  // than maintaining a second, subtly different path canonicalizer.
  LayoutTraceOptions paths{path.str()};
  if (failed(paths.validate(statsFile, error))) {
    error = "profile-file and stats-file must name different output paths";
    return failure();
  }
  if (failed(paths.validate(traceFile, error))) {
    error = "profile-file and trace-file must name different output paths";
    return failure();
  }
  return success();
}
LogicalResult LayoutProfile::publish(StringRef path, StringRef status,
                                    std::string &error) const {
  if (!enabled() || path.empty()) return success();
  auto temp = llvm::sys::fs::TempFile::create(path + ".tmp-%%%%%%");
  if (!temp) { error = llvm::toString(temp.takeError()); return failure(); }
  llvm::raw_fd_ostream stream(temp->FD, false);
  stream << serialize(status);
  stream.flush();
  if (stream.has_error()) {
    error = stream.error().message();
    stream.clear_error();
    llvm::consumeError(temp->discard());
    return failure();
  }
  if (auto e = temp->keep(path)) { error = llvm::toString(std::move(e)); return failure(); }
  return success();
}
