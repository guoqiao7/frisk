#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"


#include <algorithm>
#include <limits>

using namespace mlir;
using namespace mlir::frisk;

namespace {
constexpr uint64_t CostVector::*fields[] = {
    &CostVector::instructionPathAndWork, &CostVector::memoryTransactions,
    &CostVector::bankConflictDegree, &CostVector::conversionBytesAndSync,
    &CostVector::spillRiskAndRegisters, &CostVector::sharedBytesAndOccupancy,
    &CostVector::replication, &CostVector::codeSize,
    &CostVector::deterministicTieBreak};
constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
uint64_t add(uint64_t a, uint64_t b, bool &saturated) {
  if (a > maximum - b) { saturated = true; return maximum; }
  return a + b;
}
uint64_t multiply(uint64_t a, uint64_t b, bool &saturated) {
  if (b && a > maximum / b) { saturated = true; return maximum; }
  return a * b;
}

LogicalResult validateDistributed(DistributedEncodingAttr encoding,
                                  ShapedType type) {
  if (!encoding || !type || !type.hasStaticShape() ||
      !type.getElementType().isIntOrFloat() || encoding.getTopology().size() != 5)
    return failure();
  for (auto extent : encoding.getTopology().asArrayRef())
    if (extent <= 0) return failure();
  return encoding.verifyForType(type, UnknownLoc::get(type.getContext()));
}
// Only immutable (type, encoding) validation is reused. Resource aggregation,
// alias-root ownership and observer order still execute for every assignment.
// Failure diagnostics are never cached or suppressed.
LogicalResult validateStaticCostCandidate(const LayoutConstraintGraph &graph,
                                          Type type, Attribute candidate) {
  std::optional<LayoutProfile::Scope> lookupProfile;
  auto key = std::make_pair(type, candidate);
  auto &cache = graph.getStaticCostValidationCache();
  const bool enabled = graph.isCostQueryCacheEnabled();
  if (enabled) {
    lookupProfile.emplace(LayoutProfileKind::StaticCostValidationCache);
    bool hit = cache.contains(key);
    LayoutProfile::cacheLookup(LayoutProfileKind::StaticCostValidationCache, hit);
    if (hit) return success();
  }
  LayoutProfile::Scope validationProfile(LayoutProfileKind::StaticCostValidation);
  LogicalResult result = failure();
  if (auto encoding = dyn_cast_or_null<DistributedEncodingAttr>(candidate)) {
    result = validateDistributed(encoding, dyn_cast_or_null<ShapedType>(type));
  } else if (auto layout = dyn_cast_or_null<StorageLayoutAttr>(candidate)) {
    auto memref = dyn_cast_or_null<MemRefType>(type);
    if (memref && memref.hasStaticShape())
      result = layout.verifyForType(memref, UnknownLoc::get(type.getContext()));
  }
  if (succeeded(result) && enabled && cache.size() < graph.getCostQueryCacheCapacity()) {
    if (cache.insert(key).second)
      LayoutProfile::cacheInsertion(LayoutProfileKind::StaticCostValidationCache);
  }
  return result;
}

FailureOr<uint64_t> threads(DistributedEncodingAttr encoding) {
  auto topology = encoding.getTopology();
  if (topology[4] != 1) return failure();
  uint64_t count = 1;
  bool saturated = false;
  for (unsigned i = 1; i < 4; ++i)
    count = multiply(count, topology[i], saturated);
  if (saturated) return failure();
  return count;
}

// A conversion can stay in a scope iff every target logical value has a
// source owner in the same physical scope. Let U be the image of source
// registers (thread scope), or source registers and lanes (warp scope).
// Target varying columns, and target/source differences for every fixed
// carrier column, must all belong to U. This tests the complete affine family
// by GF(2) image membership, including nonzero dependent/broadcast columns;
// no logical coordinates or hardware holders are enumerated.
FailureOr<bool> sourceScopeContains(const LinearExecutionLayout &source,
                                    const LinearExecutionLayout &target,
                                    unsigned freeCarriers) {
  if (!source.logicalBits) return true;
  unsigned freeBits = 0;
  for (unsigned c = 0; c < freeCarriers; ++c)
    freeBits += source.hardwareWidths[c];
  // A zero-dimensional source image uses one zero column because GF2Matrix
  // deliberately rejects empty axes. Otherwise use precisely the free columns.
  SmallVector<APInt> rows;
  for (const auto &row : source.rows)
    rows.push_back(freeBits ? row.extractBits(freeBits, 0) : APInt(1, 0));
  auto image = GF2Matrix::get(source.logicalBits, std::max(1u, freeBits), rows);
  if (failed(image)) return failure();
  // im(U) = ker(U^T)^perp. Eliminate once for the annihilator, then each
  // membership query is only parity dot products. An empty annihilator proves
  // that the source free image covers the whole logical bit space.
  auto annihilator = image->transpose().kernelBasis();
  if (annihilator.empty()) return true;
  for (unsigned c = 0; c < 5; ++c)
    for (unsigned bit = 0; bit < target.hardwareWidths[c]; ++bit) {
      APInt column(source.logicalBits, 0);
      for (unsigned r = 0; r < source.logicalBits; ++r) {
        bool value = target.rows[r][target.hardwareOffsets[c] + bit];
        if (c >= freeCarriers)
          value ^= source.rows[r][source.hardwareOffsets[c] + bit];
        column.setBitVal(r, value);
      }
      for (const auto &normal : annihilator)
        if ((normal & column).popcount() & 1) return false;
    }
  return true;
}

FailureOr<ConversionExchangeKind>
classifyExchange(RankedTensorType type, DistributedEncodingAttr source,
                 DistributedEncodingAttr target) {
  LayoutProfile::Scope profile(LayoutProfileKind::ConversionClassification);
  auto sourceTopology = source.getTopology().asArrayRef();
  auto targetTopology = target.getTopology().asArrayRef();
  if (sourceTopology.drop_front() != targetTopology.drop_front() ||
      sourceTopology[1] != 32)
    return ConversionExchangeKind::Conservative;

  auto sourceMap = dyn_cast<BitLinearLayoutMapAttr>(source.getMap());
  auto targetMap = dyn_cast<BitLinearLayoutMapAttr>(target.getMap());
  if (!sourceMap || !targetMap ||
      sourceMap.getOutputNames() != targetMap.getOutputNames() ||
      sourceMap.getOutputBitWidths() != targetMap.getOutputBitWidths())
    return ConversionExchangeKind::Conservative;
  LinearExecutionLayout sourceLinear, targetLinear;
  if (analyzeLinearExecutionLayout(source, type, sourceLinear).status != ProofStatus::Proven ||
      analyzeLinearExecutionLayout(target, type, targetLinear).status != ProofStatus::Proven ||
      proveExecutionOwnership(source, type, "first_owner").status != ProofStatus::Proven ||
      proveExecutionOwnership(target, type, "first_owner").status != ProofStatus::Proven)
    return ConversionExchangeKind::Conservative;
  auto thread = sourceScopeContains(sourceLinear, targetLinear, 1);
  if (failed(thread)) return ConversionExchangeKind::Conservative;
  if (*thread) return ConversionExchangeKind::Register;
  auto warp = sourceScopeContains(sourceLinear, targetLinear, 2);
  if (failed(warp)) return ConversionExchangeKind::Conservative;
  if (*warp) return ConversionExchangeKind::Shuffle;
  // Both layouts cover the logical domain and use the same single CTA;
  // a source exists globally even when neither narrower scope suffices.
  return ConversionExchangeKind::Shared;
}

StringRef provenance(ConversionExchangeKind kind) {
  switch (kind) {
  case ConversionExchangeKind::Identity:
    return "bounded-owner-proof-v1/identity";
  case ConversionExchangeKind::Register:
    return "symbolic-subspace-proof-v1/register";
  case ConversionExchangeKind::Shuffle:
    return "symbolic-subspace-proof-v1/shuffle";
  case ConversionExchangeKind::Shared:
    return "symbolic-subspace-proof-v1/shared";
  case ConversionExchangeKind::Conservative:
    return "cta-staging-upper-bound-v1";
  }
  llvm_unreachable("unknown conversion exchange kind");
}
} // namespace

StringRef mlir::frisk::normalizeLayoutCostModelID(StringRef id) {
  if (id.empty() || id.size() > 64 || !llvm::all_of(id, [](unsigned char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
      }))
    return "unversioned-target";
  return id;
}

bool mlir::frisk::operator<(const CostVector &lhs, const CostVector &rhs) {
  for (auto field : fields)
    if (lhs.*field != rhs.*field) return lhs.*field < rhs.*field;
  return false;
}

CostEstimate mlir::frisk::addLayoutCosts(const CostEstimate &lhs,
                                        const CostEstimate &rhs) {
  CostEstimate result;
  result.saturated = lhs.saturated || rhs.saturated;
  for (auto field : fields)
    result.cost.*field = add(lhs.cost.*field, rhs.cost.*field, result.saturated);
  for (const auto *estimate : {&lhs, &rhs})
    for (const auto &[name, value] : estimate->features) {
      auto existing = llvm::find_if(result.features, [&](const auto &feature) {
        return feature.first == name;
      });
      if (existing == result.features.end()) result.features.emplace_back(name, value);
      else existing->second = add(existing->second, value, result.saturated);
    }
  return result;
}

uint64_t mlir::frisk::getLayoutLoopDepth(Operation *insertionPoint) {
  uint64_t depth = 0;
  for (Operation *owner = insertionPoint ? insertionPoint->getParentOp() : nullptr;
       owner; owner = owner->getParentOp())
    if (isa<scf::ForOp, scf::WhileOp>(owner)) ++depth;
  return depth;
}

CostEstimate mlir::frisk::weightLayoutBoundaryCost(CostEstimate cost,
                                                   uint64_t loopDepth) {
  if (cost.cost.conversionBytesAndSync == 0) return cost;
  if (loopDepth == 0) return cost;
  if (cost.cost.conversionBytesAndSync == maximum) {
    cost.saturated = true;
    return cost;
  }
  if (loopDepth >= 64) {
    cost.cost.conversionBytesAndSync = maximum;
    cost.saturated = true;
    return cost;
  }
  for (uint64_t i = 0;
       i < loopDepth && cost.cost.conversionBytesAndSync != maximum; ++i)
    cost.cost.conversionBytesAndSync =
        multiply(cost.cost.conversionBytesAndSync, 2, cost.saturated);
  return cost;
}

FailureOr<CostEstimate> mlir::frisk::evaluateLayoutConversionCost(
    const LayoutVar &sourceVar, Attribute source, Attribute target,
    LayoutConversionEdge &edge, LayoutBoundaryFrequencyMode mode) {
  auto type = dyn_cast_or_null<RankedTensorType>(sourceVar.shapedType);
  if (!type) return failure();
  auto src = dyn_cast_or_null<DistributedEncodingAttr>(source);
  auto dst = dyn_cast_or_null<DistributedEncodingAttr>(target);
  if (failed(validateDistributed(src, type)) ||
      failed(validateDistributed(dst, type))) return failure();
  auto sourceThreads = threads(src), targetThreads = threads(dst);
  if (failed(sourceThreads) || failed(targetThreads)) return failure();
  // Layouts may describe different active thread subsets in the same CTA.
  // Preserve the existing legal 32 -> 128 conversions across execution scopes.
  const uint64_t ctaThreads = std::max(*sourceThreads, *targetThreads);
  CostEstimate result;
  uint64_t bytes = 0, sync = 0;
  ConversionExchangeKind exchange = ConversionExchangeKind::Identity;
  if (source != target) {
    uint64_t sourceCarriers = 1, targetCarriers = 1;
    for (int64_t extent : src.getTopology().asArrayRef())
      sourceCarriers = multiply(sourceCarriers, extent, result.saturated);
    for (int64_t extent : dst.getTopology().asArrayRef())
      targetCarriers = multiply(targetCarriers, extent, result.saturated);
    uint64_t bits = type.getElementTypeBitWidth();
    uint64_t elementBytes = bits / 8 + (bits % 8 != 0);
    auto classified = classifyExchange(type, src, dst);
    exchange = succeeded(classified) ? *classified
                                     : ConversionExchangeKind::Conservative;
    uint64_t carriers = exchange == ConversionExchangeKind::Register
                            ? targetCarriers
                            : add(sourceCarriers, targetCarriers,
                                  result.saturated);
    bytes = multiply(elementBytes, carriers, result.saturated);
    sync = exchange == ConversionExchangeKind::Shared ||
                   exchange == ConversionExchangeKind::Conservative
               ? 2
               : 0;
    uint64_t synchronization = multiply(4, ctaThreads, result.saturated);
    synchronization = multiply(synchronization, sync, result.saturated);
    result.cost.conversionBytesAndSync = add(bytes, synchronization, result.saturated);
    result.cost.codeSize = 1;
  }
  Operation *insertionPoint = edge.insertionPoint;
  if (!insertionPoint && edge.use) insertionPoint = edge.use->getOwner();
  edge.loopDepth = getLayoutLoopDepth(insertionPoint);
  auto frequency = getLayoutBoundaryFrequency(insertionPoint, mode);
  edge.frequencyFactor = frequency.factor;
  edge.frequencyUsesProxy = frequency.usesProxy;
  edge.frequencySaturated = frequency.saturated;
  edge.staticTripCountFrequency = mode == LayoutBoundaryFrequencyMode::StaticTripCount;
  if (mode == LayoutBoundaryFrequencyMode::Legacy)
    result = weightLayoutBoundaryCost(result, edge.loopDepth);
  else {
    result.cost.conversionBytesAndSync = multiply(result.cost.conversionBytesAndSync,
                                                frequency.factor, result.saturated);
    result.saturated |= frequency.saturated && result.cost.conversionBytesAndSync != 0;
  }
  edge.sourceEncoding = source;
  edge.targetEncoding = target;
  edge.bytes = bytes;
  edge.synchronizationCost = sync;
  edge.exchange = exchange;
  edge.costProvenance = provenance(exchange).str();
  return result;
}

FailureOr<CostEstimate> mlir::frisk::evaluateLayoutCloneCost(
    const LayoutVar &result, Attribute candidate, Operation *insertionPoint, LayoutBoundaryFrequencyMode mode) {
  auto type = dyn_cast_or_null<RankedTensorType>(result.shapedType);
  auto encoding = dyn_cast_or_null<DistributedEncodingAttr>(candidate);
  if (failed(validateDistributed(encoding, type)) || failed(threads(encoding)))
    return failure();
  CostEstimate cost;
  uint64_t bits = type.getElementTypeBitWidth();
  uint64_t work = bits / 8 + (bits % 8 != 0);
  for (int64_t extent : encoding.getTopology().asArrayRef())
    work = multiply(work, extent, cost.saturated);
  cost.cost.conversionBytesAndSync = work;
  cost.cost.spillRiskAndRegisters = encoding.getTopology()[0];
  cost.cost.replication = encoding.getReplication().getValue().getLimitedValue() - 1;
  cost.cost.codeSize = 1;
  if (mode == LayoutBoundaryFrequencyMode::Legacy)
    return weightLayoutBoundaryCost(cost, getLayoutLoopDepth(insertionPoint));
  auto frequency = getLayoutBoundaryFrequency(insertionPoint, mode);
  cost.cost.conversionBytesAndSync = multiply(cost.cost.conversionBytesAndSync,
                                            frequency.factor, cost.saturated);
  cost.saturated |= frequency.saturated && cost.cost.conversionBytesAndSync != 0;
  return cost;
}

FailureOr<CostEstimate> mlir::frisk::evaluateStaticLayoutCost(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment,
    LayoutCostObserver *observer) {
  CostEstimate result;
  DenseMap<Value, unsigned> sharedRoots;
  struct RootCharge { LayoutVarID owner; uint64_t bytes; };
  SmallVector<RootCharge> roots;
  auto chargeVariable = [&](LayoutVarID id, Attribute attr) -> LogicalResult {
    if (id >= graph.getVariables().size() || !attr) return failure();
    const auto &var = graph.getVariable(id);
    if (var.use || var.kind == LayoutKind::Instruction) return success();
    if (var.kind == LayoutKind::Distributed) {
      auto type = dyn_cast_or_null<ShapedType>(var.shapedType);
      auto encoding = dyn_cast<DistributedEncodingAttr>(attr);
      if (failed(validateStaticCostCandidate(graph, type, encoding))) return failure();
      CostEstimate charge;
      charge.cost.spillRiskAndRegisters = encoding.getTopology()[0];
      charge.cost.replication = encoding.getReplication().getValue().getLimitedValue() - 1;
      result = addLayoutCosts(result, charge);
      if (observer) observer->recordCost("registers", id, charge);
      return success();
    }
    auto type = dyn_cast_or_null<MemRefType>(var.shapedType);
    auto layout = dyn_cast<StorageLayoutAttr>(attr);
    if (!type || !layout || !type.hasStaticShape() ||
        failed(validateStaticCostCandidate(graph, type, layout)))
      return failure();
    if (layout.getMemorySpace().getValue() != attr::MemorySpace::Shared) return success();
    std::optional<StorageAliasInfo> alias = var.storageAlias;
    if (!alias && var.value) {
      auto analyzed = analyzeStorageAlias(var.value);
      if (failed(analyzed)) return failure();
      alias = *analyzed;
    }
    uint64_t bytes = 0;
    if (alias) {
      StorageAliasFootprint uncached;
      if (!var.storageAlias) uncached = buildStorageAliasFootprint(*alias, layout);
      const auto &footprint = var.storageAlias
          ? getStorageAliasFootprint(graph, id, layout) : uncached;
      if (footprint.proof.status != ProofStatus::Proven) return failure();
      for (const auto &entry : footprint.entries)
        bytes = std::max(bytes, entry.end / 8 + (entry.end % 8 != 0));
      auto inserted = sharedRoots.try_emplace(alias->root, roots.size());
      if (inserted.second) roots.push_back({id, bytes});
      else roots[inserted.first->second].bytes =
          std::max(roots[inserted.first->second].bytes, bytes);
    } else {
      auto footprint = getStorageFootprintBytes(layout, type);
      if (failed(footprint)) return failure();
      roots.push_back({id, *footprint});
    }
    return success();
  };
  if (!observer) {
    // Partial lower bounds are a hot path: visit only assigned variables,
    // with no sorting, trace snapshot, or work on unrelated graph components.
    for (const auto &[id, attr] : assignment.values)
      if (failed(chargeVariable(id, attr))) return failure();
  } else {
    // IDs follow graph order. Sort only assigned IDs for stable event order
    // and Shared-root ownership, without scanning the rest of the graph.
    SmallVector<LayoutVarID> assignedIDs;
    assignedIDs.reserve(assignment.values.size());
    for (const auto &[id, attr] : assignment.values) {
      if (id >= graph.getVariables().size() || !attr) return failure();
      assignedIDs.push_back(id);
    }
    llvm::sort(assignedIDs);
    for (auto id : assignedIDs)
      if (failed(chargeVariable(id, assignment.values.lookup(id)))) return failure();
  }
  for (const auto &root : roots) {
    CostEstimate charge;
    charge.cost.sharedBytesAndOccupancy = root.bytes;
    result = addLayoutCosts(result, charge);
    if (observer) observer->recordCost("shared-root", root.owner, charge);
  }
  return result;
}
