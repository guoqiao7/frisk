#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/Support/MathExtras.h"
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::frisk;

StringRef mlir::frisk::getSM90LegacyCostModelID() {
  return "sm90-layout-lex9-legacy-v1";
}

namespace {
// Warp issues are grouped by physical warp and register vector, never across
// different instructions. Scores are static work proxies, not hardware cycles.
using Issues = std::map<SmallVector<int64_t>, std::vector<std::pair<uint64_t, uint64_t>>>;
DictionaryAttr selectedCopy(const LayoutConstraintGraph &graph, Operation *op,
                            const CandidateAssignment &assignment) {
  for (const auto &c : graph.getConstraints()) {
    if (!c.instruction || c.instruction->source != op) continue;
    for (const auto &tuple : c.instruction->tuples) {
      bool matches = true;
      for (auto [id, encoding] : llvm::zip(c.vars, tuple.encodings))
        matches &= assignment.values.lookup(id) == encoding;
      if (matches) return dyn_cast<DictionaryAttr>(tuple.binding);
    }
  }
  return {};
}

FailureOr<CostEstimate> accessCost(const LayoutConstraintGraph &graph,
    const LayoutConstraint &c, const CandidateAssignment &assignment) {
  LayoutProfile::Scope profile(LayoutProfileKind::AccessCost);
  const auto &a = graph.getVariable(c.vars[0]), &b = graph.getVariable(c.vars[1]);
  const auto &execution = a.kind == LayoutKind::Distributed ? a : b;
  const auto &storage = a.kind == LayoutKind::Storage ? a : b;
  auto encoding = dyn_cast_or_null<DistributedEncodingAttr>(assignment.values.lookup(execution.id));
  auto layout = dyn_cast_or_null<StorageLayoutAttr>(assignment.values.lookup(storage.id));
  if (!encoding || !layout) return CostEstimate{};
  if (!storage.storageAlias) return failure();
  const auto &footprint = getStorageAliasFootprint(graph, storage.id, layout);
  if (footprint.proof.status != ProofStatus::Proven) return failure();
  auto copy = execution.operationExecution ? selectedCopy(graph, execution.anchor, assignment) : DictionaryAttr{};
  bool bulk = copy && copy.getAs<StringAttr>("path").getValue() == "tma";
  unsigned width = copy ? copy.getAs<IntegerAttr>("vector_bytes").getInt() : 1;
  ExecutionEnumeration enumeration;
  if (enumerateExecutionLayout(encoding, cast<RankedTensorType>(execution.shapedType), enumeration).status != ProofStatus::Proven)
    return failure();
  std::map<SmallVector<int64_t>, std::pair<uint64_t, uint64_t>> addresses;
  for (const auto &p : footprint.entries) addresses.emplace(p.view, std::make_pair(p.begin / 8, llvm::divideCeil(p.end, uint64_t{8})));
  unsigned bits = cast<ShapedType>(storage.shapedType).getElementTypeBitWidth();
  unsigned chunk = std::max(1u, width * 8 / bits);
  // A wide shared instruction has independent 128-byte phases: contiguous
  // groups of 16 lanes for 8B/lane and 8 lanes for 16B/lane. The scalar path's
  // width=1 sentinel must not hide a native 8-byte element. Keep warp issues
  // separate from these bank batches so instruction/global costs are unchanged.
  // Native values wider than 16B are a conservative distinct-word proxy, not
  // a claim about their eventual instruction lowering. Clamp to one lane for
  // >128B elements so arbitrary legal integer types remain scoreable.
  unsigned phaseBytes = std::max(width, llvm::divideCeil(bits, 8u));
  unsigned lanesPerPhase = std::max(1u, std::min(32u, 128u / phaseBytes));
  Issues issues, bankPhases;
  for (uint64_t h = 0; h < enumeration.logical.size(); ++h) {
    auto logical = enumeration.logical[h];
    if (c.access == AccessKind::Write && enumeration.first[logical] != h) continue;
    auto point = enumeration.coordinate(logical);
    auto address = addresses.find(point);
    // Guarded Copy uses a padded execution domain. No inactive point generates
    // an address or a memory transaction.
    if (address == addresses.end()) continue;
    auto hw = enumeration.hardware(h);
    SmallVector<int64_t> issue = bulk ? SmallVector<int64_t>{0} :
        SmallVector<int64_t>{hw[0], hw[1], hw[2], hw[4] / chunk};
    issues[issue].push_back(address->second);
    issue.push_back(hw[3] / lanesPerPhase);
    bankPhases[issue].push_back(address->second);
  }
  CostEstimate result;
  auto space = layout.getMemorySpace().getValue();
  // Unknown base phase is an upper bound over all phases consistent with the
  // root guarantee, not an invented aligned base address.
  uint64_t step = std::min<uint64_t>(32, storage.storageAlias->rootAlignment);
  uint64_t worstSectors = 0, worstBanks = 0;
  for (uint64_t phase = 0; phase < 32; phase += step) {
    uint64_t sectors = 0, banks = 0;
    for (const auto &[key, ranges] : issues) {
      std::set<uint64_t> touchedSectors;
      for (auto [begin, end] : ranges) {
        for (uint64_t sector = (begin + phase) / 32; sector <= (end - 1 + phase) / 32; ++sector)
          touchedSectors.insert(sector);
      }
      sectors += touchedSectors.size();
    }
    for (const auto &[key, ranges] : bankPhases) {
      std::map<unsigned, std::set<uint64_t>> words;
      for (auto [begin, end] : ranges)
        for (uint64_t word = (begin + phase) / 4; word <= (end - 1 + phase) / 4; ++word)
          words[word % 32].insert(word);
      uint64_t degree = 1;
      for (const auto &[bank, distinct] : words) degree = std::max<uint64_t>(degree, distinct.size());
      // A read broadcast counts once. Packed writes also count distinct words:
      // this is a bandwidth proxy, not a subword-write/race correctness proof.
      banks += degree - 1;
    }
    worstSectors = std::max(worstSectors, sectors);
    worstBanks = std::max(worstBanks, banks);
  }
  if (space == attr::MemorySpace::Global) result.cost.memoryTransactions = worstSectors;
  if (space == attr::MemorySpace::Shared && !bulk) result.cost.bankConflictDegree = worstBanks;
  // Copy vector/scalar uses one load and one store issue; cp.async combines
  // that pair, so charge its source once. TMA has a separate bulk work proxy.
  if (!bulk && (!copy || copy.getAs<StringAttr>("path").getValue() == "vector" ||
                c.access == AccessKind::Read))
    result.cost.instructionPathAndWork = issues.size();
  return result;
}
} // namespace

FailureOr<CostEstimate> mlir::frisk::evaluateSM90LayoutCost(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment,
    LayoutCostObserver *observer, LayoutBoundaryFrequencyMode mode) {
  (void)mode; // Ordinary arithmetic and resource terms remain static.
  auto total = evaluateStaticLayoutCost(graph, assignment, observer);
  if (failed(total)) return failure();
  for (const auto &c : graph.getConstraints()) {
    bool complete = llvm::all_of(c.vars, [&](auto id) { return assignment.values.count(id); });
    if (!complete) continue;
    CostEstimate extra;
    StringRef kind;
    if (c.kind == ConstraintKind::StorageAccess) {
      kind = "access";
      auto cost = accessCost(graph, c, assignment);
      if (failed(cost)) return failure();
      extra = *cost;
    } else if (c.instruction && isa_and_nonnull<CopyOp>(c.instruction->source)) {
      kind = "copy-setup";
      auto binding = selectedCopy(graph, c.instruction->source, assignment);
      if (!binding) return failure();
      auto type = cast<ShapedType>(graph.getVariable(c.vars[0]).shapedType);
      uint64_t bytes = llvm::divideCeil(uint64_t(type.getNumElements()) * type.getElementTypeBitWidth(), uint64_t{8});
      auto path = binding.getAs<StringAttr>("path").getValue();
      // Versioned setup/work model: TMA amortizes a four-unit setup over 1KiB
      // chunks; cp.async has commit/wait setup plus one warp issue per vector;
      // synchronous Copy needs load and store. Not a latency prediction.
      extra.cost.instructionPathAndWork = path == "tma" ? 4 + llvm::divideCeil(bytes, uint64_t{1024}) :
          (path == "cp.async" ? 2 : 0);
    } else if (c.instruction && isa_and_nonnull<MmaOp>(c.instruction->source)) {
      kind = "mma";
      auto op = cast<MmaOp>(c.instruction->source);
      auto tuple = findInstructionSupport(graph,c,assignment.values);
      if (!tuple) return failure();
      auto geometry = getSM90MmaContractGeometry(op, graph.getVariable(c.vars[3]).requiredThreads,tuple->binding);
      if (failed(geometry)) return failure();
      extra.cost.instructionPathAndWork = geometry->gM * geometry->gN *
          geometry->mRepeat * geometry->nRepeat * geometry->kRepeat;
    } else if (c.kind == ConstraintKind::ReductionLayout) {
      kind = "reduction";
      auto source = cast<ShapedType>(graph.getVariable(c.vars[0]).shapedType);
      auto result = cast<ShapedType>(graph.getVariable(c.vars[1]).shapedType);
      extra.cost.instructionPathAndWork = source.getNumElements() - result.getNumElements();
    }
    // Access-local counters are bounded: enumeration has <=65536 carriers,
    // each range covers one element of unsigned bit width, hence fewer than
    // 2^48 word/sector visits in total. Proven signed bit addresses also leave
    // ample room for the <=31-byte base phase. Cross-constraint accumulation
    // uses the shared saturating helper, including its saturation flag.
    *total = addLayoutCosts(*total, extra);
    if (observer && !kind.empty()) observer->recordCost(kind, c.id, extra);
  }
  return *total;
}
