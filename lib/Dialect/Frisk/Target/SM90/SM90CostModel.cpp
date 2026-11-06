#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <map>
#include <set>

using namespace mlir;
using namespace mlir::frisk;

StringRef mlir::frisk::getSM90LegacyCostModelID() {
  return "sm90-layout-lex9-legacy-v2";
}

namespace {
// Warp issues are grouped by physical warp and register vector, never across
// different instructions. Scores are static work proxies, not hardware cycles.
using Issues = std::map<SmallVector<int64_t>, std::vector<std::pair<uint64_t, uint64_t>>>;
// An image basis in logical-bit space. Columns are inserted in canonical
// hardware-bit order (register first). The recorded preimage therefore sets
// every non-basis, higher hardware bit to zero: it is the numerically smallest
// owner, including when nonzero columns are linearly dependent (not just zero
// broadcast columns). Reducing a logical value gives its canonical coset.
struct AccessImageBasis {
  SmallVector<uint64_t> pivots;
  SmallVector<APInt> preimages;
  unsigned rank = 0, ownerWidth;
  AccessImageBasis(unsigned logicalBits, unsigned hardwareBits)
      : pivots(logicalBits, 0),
        preimages(logicalBits, APInt(std::max(1u, hardwareBits), 0)),
        ownerWidth(std::max(1u, hardwareBits)) {}
  void insert(uint64_t value, APInt owner) {
    for (unsigned bit = pivots.size(); bit-- > 0;) {
      if (!(value & (uint64_t{1} << bit))) continue;
      if (pivots[bit]) {
        value ^= pivots[bit];
        owner ^= preimages[bit];
      } else {
        pivots[bit] = value;
        preimages[bit] = owner;
        ++rank;
        return;
      }
    }
  }
  uint64_t coset(uint64_t value) const {
    for (unsigned bit = pivots.size(); bit-- > 0;)
      if (value & (uint64_t{1} << bit)) value ^= pivots[bit];
    return value;
  }
  APInt firstOwner(uint64_t value) const {
    APInt owner(ownerWidth, 0);
    for (unsigned bit = pivots.size(); bit-- > 0;)
      if (value & (uint64_t{1} << bit)) {
        assert(pivots[bit] && "ownership must be proven before solving");
        value ^= pivots[bit];
        owner ^= preimages[bit];
      }
    return owner;
  }
};

uint64_t weightAccessCount(uint64_t value, unsigned multiplicityBits,
                           bool &saturated) {
  if (!value) return 0;
  constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
  if (multiplicityBits >= 64 || value > (maximum >> multiplicityBits)) {
    saturated = true;
    return maximum;
  }
  return value << multiplicityBits;
}

// Exact push-forward of physical issue groups, bounded by logical addresses,
// not by the number of hardware holders. For reads, let U be the image of the
// bits free within one issue (lane and low register-vector bits). Its address
// set is one coset of U. Every coset occurs equally often, with multiplicity
// 2^(number of issue-key bits - (logicalBits - rank(U))). Distinct hardware
// issues are still charged separately even if they read the same addresses.
// Writes instead visit each logical address once at its canonical first owner.
LogicalResult buildSymbolicAccessIssues(DistributedEncodingAttr encoding,
    RankedTensorType type, const StorageAliasFootprint &footprint,
    AccessKind access, bool bulk, unsigned chunk, unsigned lanesPerPhase,
    Issues &issues, Issues &bankPhases, unsigned &issueMultiplicityBits,
    unsigned &bankMultiplicityBits) {
  LinearExecutionLayout layout;
  if (analyzeLinearExecutionLayout(encoding, type, layout).status != ProofStatus::Proven ||
      proveExecutionOwnership(encoding, type, "first_owner").status != ProofStatus::Proven)
    return failure();
  // Storage footprints and guarded padded domains retain their existing 65536
  // logical-point bound. There is no hardware-product or replication bound.
  if (layout.logicalBits > 16) return failure();
  AccessImageBasis owners(layout.logicalBits, layout.hardwareBits);
  AccessImageBasis issueImage(layout.logicalBits, layout.hardwareBits);
  AccessImageBasis bankImage(layout.logicalBits, layout.hardwareBits);
  unsigned regBits = std::min(layout.hardwareWidths[0], unsigned(llvm::Log2_64(chunk)));
  unsigned laneBits = layout.hardwareWidths[1];
  unsigned bankLaneBits = std::min(laneBits, unsigned(llvm::Log2_64(lanesPerPhase)));
  for (unsigned bit = 0; bit < layout.hardwareBits; ++bit) {
    APInt unit(std::max(1u, layout.hardwareBits), 0);
    unit.setBit(bit);
    uint64_t column = layout.apply(unit).getZExtValue();
    owners.insert(column, unit);
    bool regFree = bit < regBits;
    bool laneFree = bit >= layout.hardwareOffsets[1] &&
                    bit < layout.hardwareOffsets[1] + laneBits;
    bool bankLaneFree = bit >= layout.hardwareOffsets[1] &&
                        bit < layout.hardwareOffsets[1] + bankLaneBits;
    if (regFree || laneFree) issueImage.insert(column, unit);
    if (regFree || bankLaneFree) bankImage.insert(column, unit);
  }
  if (!bulk && access == AccessKind::Read) {
    issueMultiplicityBits = layout.hardwareBits - regBits - laneBits -
                            (layout.logicalBits - issueImage.rank);
    bankMultiplicityBits = layout.hardwareBits - regBits - bankLaneBits -
                           (layout.logicalBits - bankImage.rank);
  }
  for (const auto &entry : footprint.entries) {
    // As in the enumerated decoder, footprint points outside the execution
    // domain do not belong to an issue. Guarded tail points have no entry.
    if (entry.view.size() != unsigned(type.getRank())) continue;
    bool inDomain = true;
    for (unsigned axis = 0; axis < entry.view.size(); ++axis)
      inDomain &= entry.view[axis] >= 0 && entry.view[axis] < type.getDimSize(axis);
    if (!inDomain) continue;
    uint64_t logical = 0;
    unsigned offset = 0;
    for (auto [coordinate, width] : llvm::zip_equal(entry.view, layout.logicalWidths)) {
      logical |= uint64_t(coordinate) << offset;
      offset += width;
    }
    auto address = std::make_pair(entry.begin / 8, llvm::divideCeil(entry.end, uint64_t{8}));
    if (bulk) {
      issues[{0}].push_back(address);
    } else if (access == AccessKind::Read) {
      issues[{int64_t(issueImage.coset(logical))}].push_back(address);
      bankPhases[{int64_t(bankImage.coset(logical))}].push_back(address);
    } else {
      auto hw = layout.hardwareCoordinate(owners.firstOwner(logical));
      SmallVector<int64_t> issue{hw[0], hw[1], hw[2], hw[4] / chunk};
      issues[issue].push_back(address);
      issue.push_back(hw[3] / lanesPerPhase);
      bankPhases[issue].push_back(address);
    }
  }
  return success();
}

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

FailureOr<CostEstimate> computeAccessCost(const LayoutConstraintGraph &graph,
    const LayoutConstraint &c, const LayoutVar &execution, const LayoutVar &storage,
    DistributedEncodingAttr encoding, StorageLayoutAttr layout,
    DictionaryAttr copy) {
  const auto &footprint = getStorageAliasFootprint(graph, storage.id, layout);
  if (footprint.proof.status != ProofStatus::Proven) return failure();
  bool bulk = copy && copy.getAs<StringAttr>("path").getValue() == "tma";
  unsigned width = copy ? copy.getAs<IntegerAttr>("vector_bytes").getInt() : 1;
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
  unsigned issueMultiplicityBits = 0, bankMultiplicityBits = 0;
  if (bulk || (llvm::isPowerOf2_32(chunk) && llvm::isPowerOf2_32(lanesPerPhase))) {
    if (failed(buildSymbolicAccessIssues(encoding,
          cast<RankedTensorType>(execution.shapedType), footprint, c.access,
          bulk, chunk, lanesPerPhase, issues, bankPhases,
          issueMultiplicityBits, bankMultiplicityBits))) return failure();
  } else {
    // Unusual element widths can make register chunks or 128B lane phases
    // non-dyadic (e.g. i40 has 25 lanes/phase). Retain the exact existing
    // bounded decoder for these groups; taking log2 would silently mis-score.
    ExecutionEnumeration enumeration;
    if (enumerateExecutionLayout(encoding,
          cast<RankedTensorType>(execution.shapedType), enumeration).status != ProofStatus::Proven)
      return failure();
    std::map<SmallVector<int64_t>, std::pair<uint64_t, uint64_t>> addresses;
    for (const auto &p : footprint.entries)
      addresses.emplace(p.view, std::make_pair(p.begin / 8, llvm::divideCeil(p.end, uint64_t{8})));
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
  if (space == attr::MemorySpace::Global)
    result.cost.memoryTransactions = weightAccessCount(worstSectors, issueMultiplicityBits, result.saturated);
  if (space == attr::MemorySpace::Shared && !bulk)
    result.cost.bankConflictDegree = weightAccessCount(worstBanks, bankMultiplicityBits, result.saturated);
  // Copy vector/scalar uses one load and one store issue; cp.async combines
  // that pair, so charge its source once. TMA has a separate bulk work proxy.
  if (!bulk && (!copy || copy.getAs<StringAttr>("path").getValue() == "vector" ||
                c.access == AccessKind::Read))
    result.cost.instructionPathAndWork = weightAccessCount(issues.size(), issueMultiplicityBits, result.saturated);
  return result;
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
  auto copy = execution.operationExecution
      ? selectedCopy(graph, execution.anchor, assignment) : DictionaryAttr{};
  if (!graph.isCostQueryCacheEnabled() || !graph.getCostQueryCacheCapacity())
    return computeAccessCost(graph, c, execution, storage, encoding, layout, copy);

  // The selected Copy contract matters even for otherwise identical endpoint
  // layouts: vector width and TMA change physical issue/bank grouping. A null
  // binding is the scalar fallback, never a wide/bulk cache hit. The graph owns
  // immutable root/alias facts; do not reuse this cache after changing its IR.
  LayoutConstraintGraph::AccessCostCacheKey key{c.id, execution.shapedType,
      encoding, storage.shapedType, layout, copy,
      storage.storageAlias->rootAlignment, unsigned(c.access)};
  LayoutProfile::Scope lookup(LayoutProfileKind::AccessCostCache);
  auto &cache = graph.getAccessCostCache();
  auto found = cache.find(key);
  LayoutProfile::cacheLookup(LayoutProfileKind::AccessCostCache, found != cache.end());
  if (found != cache.end()) return found->second;
  auto result = computeAccessCost(graph, c, execution, storage, encoding, layout, copy);
  // Unknown/failing queries remain failures and are not stored. At capacity,
  // bypass insertion only; the same scoring and proof budgets still apply.
  if (succeeded(result) && cache.size() < graph.getCostQueryCacheCapacity() &&
      cache.try_emplace(std::move(key), *result).second)
    LayoutProfile::cacheInsertion(LayoutProfileKind::AccessCostCache);
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
    // Each representative table has <=65536 logical addresses. Its local
    // word/sector visits fit uint64_t; potentially enormous physical issue
    // multiplicities are applied with saturation before cross-constraint sums.
    // Proven signed bit addresses leave room for the <=31-byte base phase.
    *total = addLayoutCosts(*total, extra);
    if (observer && !kind.empty()) observer->recordCost(kind, c.id, extra);
  }
  return *total;
}
