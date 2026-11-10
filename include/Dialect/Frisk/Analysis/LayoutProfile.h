#ifndef FRISK_ANALYSIS_LAYOUTPROFILE_H
#define FRISK_ANALYSIS_LAYOUTPROFILE_H

#include "llvm/Support/JSON.h"
#include "mlir/Support/LogicalResult.h"
#include <array>
#include <chrono>

namespace mlir::frisk {
enum class LayoutProfileKind {
  AddressEnumeration, ExecutionEnumeration, AccessCost,
  ConversionClassification, ConversionChoice, InstructionSupport,
  InstructionContract, RelationQuery, RelationCache, AliasPairCache,
  FootprintCache, AccessCostCache, StaticCostValidationCache,
  StaticCostValidation, DemandFootprintCache, Count
};
struct LayoutProfileMetric {
  uint64_t count = 0, inclusiveNS = 0, hits = 0, misses = 0, entries = 0, peakEntries = 0;
};
/// Invocation-local observation only: no IR references, query data, or caches.
/// Times are inclusive and overlap (e.g. access cost includes enumeration and
/// footprint queries, relation queries include instruction/alias support).
/// entries counts successful insertions, NOT live or peak cache size. Caches
/// may be cleared or recreated during a single invocation.
class LayoutProfile {
public:
  explicit LayoutProfile(bool enabled = false) : isEnabled(enabled) {}
  bool enabled() const { return isEnabled; }
  const LayoutProfileMetric &get(LayoutProfileKind kind) const;
  llvm::json::Object toJSON(llvm::StringRef status) const;
  std::string serialize(llvm::StringRef status) const;
  LogicalResult publish(llvm::StringRef path, llvm::StringRef status,
                        std::string &error) const;
  static LogicalResult validate(llvm::StringRef path, llvm::StringRef statsFile,
                                llvm::StringRef traceFile, std::string &error);
  static LayoutProfile *current();
  /// Always binds, including nullptr/disabled: independent invocations do not
  /// inherit an outer observer. Explicitly pass the pipeline observer to its
  /// nested inference. Destruction restores the previous thread-local binding.
  class Invocation {
  public:
    explicit Invocation(LayoutProfile *profile);
    ~Invocation();
    Invocation(const Invocation &) = delete;
    Invocation &operator=(const Invocation &) = delete;
  private:
    LayoutProfile *previous;
  };
  /// Function-entry counter and optional timer; disabled scopes never read a
  /// clock or build event payloads. Nested scopes retain their entry observer.
  class Scope {
  public:
    explicit Scope(LayoutProfileKind kind);
    ~Scope();
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
  private:
    LayoutProfileMetric *metric = nullptr;
    std::chrono::steady_clock::time_point start;
  };
  static void cacheLookup(LayoutProfileKind kind, bool hit);
  static void cacheInsertion(LayoutProfileKind kind);
  static void cacheSize(LayoutProfileKind kind, uint64_t size);
private:
  bool isEnabled;
  std::array<LayoutProfileMetric, static_cast<size_t>(LayoutProfileKind::Count)> metrics{};
};
} // namespace mlir::frisk
#endif
