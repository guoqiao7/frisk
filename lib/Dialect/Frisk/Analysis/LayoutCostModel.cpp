#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/DenseSet.h"

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
constexpr uint64_t maximumLogicalElements = 65536;
constexpr uint64_t maximumCarriers = 262144;
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

struct Owner {
  uint64_t reg = 0, lane = 0, warp = 0, warpGroup = 0;
};

FailureOr<APInt> encodeOwner(BitLinearLayoutMapAttr map, const Owner &owner,
                             unsigned bits) {
  APInt encoded(bits, 0);
  unsigned offset = 0;
  for (auto [nameAttr, width] :
       llvm::zip_equal(map.getInputNames(),
                       map.getInputBitWidths().asArrayRef())) {
    StringRef name = cast<StringAttr>(nameAttr).getValue();
    uint64_t value;
    if (name == "register") value = owner.reg;
    else if (name == "lane") value = owner.lane;
    else if (name == "warp") value = owner.warp;
    else if (name == "warp_group") value = owner.warpGroup;
    else if (name == "cta") value = 0;
    else return failure();
    encoded |= APInt(bits, value).shl(offset);
    offset += width;
  }
  return encoded;
}

FailureOr<uint64_t> logicalSlot(BitLinearLayoutMapAttr map,
                                const GF2Matrix &matrix, const Owner &owner,
                                ArrayRef<int64_t> shape) {
  auto encoded = encodeOwner(map, owner, matrix.getNumColumns());
  if (failed(encoded)) return failure();
  APInt output = matrix.apply(*encoded);
  uint64_t slot = 0;
  unsigned offset = 0;
  for (auto [extent, width] :
       llvm::zip_equal(shape, map.getOutputBitWidths().asArrayRef())) {
    uint64_t coordinate = output.extractBitsAsZExtValue(width, offset);
    if (coordinate >= uint64_t(extent)) return failure();
    slot = slot * uint64_t(extent) + coordinate;
    offset += width;
  }
  return slot;
}

uint64_t physicalThread(const Owner &owner, ArrayRef<int64_t> topology) {
  return (owner.warpGroup * uint64_t(topology[2]) + owner.warp) *
             uint64_t(topology[1]) +
         owner.lane;
}

uint64_t physicalWarp(const Owner &owner, ArrayRef<int64_t> topology) {
  return owner.warpGroup * uint64_t(topology[2]) + owner.warp;
}

template <typename Callback>
LogicalResult enumerateOwners(ArrayRef<int64_t> topology,
                              Callback &&callback) {
  for (uint64_t warpGroup = 0; warpGroup < uint64_t(topology[3]); ++warpGroup)
    for (uint64_t warp = 0; warp < uint64_t(topology[2]); ++warp)
      for (uint64_t lane = 0; lane < uint64_t(topology[1]); ++lane)
        for (uint64_t reg = 0; reg < uint64_t(topology[0]); ++reg)
          if (failed(callback(Owner{reg, lane, warp, warpGroup})))
            return failure();
  return success();
}

FailureOr<ConversionExchangeKind>
classifyExchange(RankedTensorType type, DistributedEncodingAttr source,
                 DistributedEncodingAttr target) {
  auto sourceTopology = source.getTopology().asArrayRef();
  auto targetTopology = target.getTopology().asArrayRef();
  if (sourceTopology.drop_front() != targetTopology.drop_front() ||
      sourceTopology[1] != 32)
    return ConversionExchangeKind::Conservative;

  uint64_t volume = 1;
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || uint64_t(extent) > maximumLogicalElements / volume)
      return ConversionExchangeKind::Conservative;
    volume *= uint64_t(extent);
  }
  uint64_t sourceCarriers = 1, targetCarriers = 1;
  bool saturated = false;
  for (int64_t extent : sourceTopology)
    sourceCarriers = multiply(sourceCarriers, extent, saturated);
  for (int64_t extent : targetTopology)
    targetCarriers = multiply(targetCarriers, extent, saturated);
  if (saturated || sourceCarriers > maximumCarriers ||
      targetCarriers > maximumCarriers - sourceCarriers)
    return ConversionExchangeKind::Conservative;

  auto sourceCanonical = source.getCanonicalMap(type);
  auto targetCanonical = target.getCanonicalMap(type);
  if (failed(sourceCanonical) || failed(targetCanonical))
    return ConversionExchangeKind::Conservative;
  auto sourceMap = dyn_cast<BitLinearLayoutMapAttr>(*sourceCanonical);
  auto targetMap = dyn_cast<BitLinearLayoutMapAttr>(*targetCanonical);
  if (!sourceMap || !targetMap ||
      sourceMap.getOutputNames() != targetMap.getOutputNames() ||
      sourceMap.getOutputBitWidths() != targetMap.getOutputBitWidths())
    return ConversionExchangeKind::Conservative;
  auto sourceMatrix = sourceMap.getMatrixValue();
  auto targetMatrix = targetMap.getMatrixValue();
  if (failed(sourceMatrix) || failed(targetMatrix))
    return ConversionExchangeKind::Conservative;

  DenseSet<std::pair<uint64_t, uint64_t>> threadOwners;
  DenseSet<std::pair<uint64_t, uint64_t>> warpOwners;
  DenseSet<uint64_t> globalOwners;
  if (failed(enumerateOwners(sourceTopology, [&](const Owner &owner) {
        auto slot = logicalSlot(sourceMap, *sourceMatrix, owner,
                                type.getShape());
        if (failed(slot)) return failure();
        threadOwners.insert({*slot, physicalThread(owner, sourceTopology)});
        warpOwners.insert({*slot, physicalWarp(owner, sourceTopology)});
        globalOwners.insert(*slot);
        return success();
      })))
    return ConversionExchangeKind::Conservative;

  ConversionExchangeKind result = ConversionExchangeKind::Register;
  if (failed(enumerateOwners(targetTopology, [&](const Owner &owner) {
        auto slot = logicalSlot(targetMap, *targetMatrix, owner,
                                type.getShape());
        if (failed(slot) || !globalOwners.contains(*slot)) return failure();
        if (threadOwners.contains(
                {*slot, physicalThread(owner, targetTopology)}))
          return success();
        if (warpOwners.contains({*slot, physicalWarp(owner, targetTopology)})) {
          if (result == ConversionExchangeKind::Register)
            result = ConversionExchangeKind::Shuffle;
          return success();
        }
        result = ConversionExchangeKind::Shared;
        return success();
      })))
    return ConversionExchangeKind::Conservative;
  return result;
}

StringRef provenance(ConversionExchangeKind kind) {
  switch (kind) {
  case ConversionExchangeKind::Identity:
    return "bounded-owner-proof-v1/identity";
  case ConversionExchangeKind::Register:
    return "bounded-owner-proof-v1/register";
  case ConversionExchangeKind::Shuffle:
    return "bounded-owner-proof-v1/shuffle";
  case ConversionExchangeKind::Shared:
    return "bounded-owner-proof-v1/shared";
  case ConversionExchangeKind::Conservative:
    return "cta-staging-upper-bound-v1";
  }
  llvm_unreachable("unknown conversion exchange kind");
}
} // namespace

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
    LayoutConversionEdge &edge) {
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
  result = weightLayoutBoundaryCost(result, edge.loopDepth);
  edge.sourceEncoding = source;
  edge.targetEncoding = target;
  edge.bytes = bytes;
  edge.synchronizationCost = sync;
  edge.exchange = exchange;
  edge.costProvenance = provenance(exchange).str();
  return result;
}

FailureOr<CostEstimate> mlir::frisk::evaluateLayoutCloneCost(
    const LayoutVar &result, Attribute candidate, Operation *insertionPoint) {
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
  return weightLayoutBoundaryCost(cost, getLayoutLoopDepth(insertionPoint));
}

FailureOr<CostEstimate> mlir::frisk::evaluateStaticLayoutCost(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment) {
  CostEstimate result;
  DenseMap<Value, uint64_t> sharedRoots;
  DenseMap<LayoutVarID, uint64_t> syntheticRoots;
  for (const auto &[id, attr] : assignment.values) {
    if (id >= graph.getVariables().size() || !attr) return failure();
    const auto &var = graph.getVariable(id);
    if (var.use || var.kind == LayoutKind::Instruction) continue;
    if (var.kind == LayoutKind::Distributed) {
      auto type = dyn_cast_or_null<ShapedType>(var.shapedType);
      auto encoding = dyn_cast<DistributedEncodingAttr>(attr);
      if (failed(validateDistributed(encoding, type))) return failure();
      result.cost.spillRiskAndRegisters = add(result.cost.spillRiskAndRegisters,
          encoding.getTopology()[0], result.saturated);
      result.cost.replication = add(result.cost.replication,
          encoding.getReplication().getValue().getLimitedValue() - 1,
          result.saturated);
      continue;
    }
    auto type = dyn_cast_or_null<MemRefType>(var.shapedType);
    auto layout = dyn_cast<StorageLayoutAttr>(attr);
    if (!type || !layout || !type.hasStaticShape() ||
        failed(layout.verifyForType(type, UnknownLoc::get(type.getContext()))))
      return failure();
    if (layout.getMemorySpace().getValue() != attr::MemorySpace::Shared) continue;
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
      sharedRoots[alias->root] = std::max(sharedRoots[alias->root], bytes);
    } else {
      auto footprint = getStorageFootprintBytes(layout, type);
      if (failed(footprint)) return failure();
      syntheticRoots[id] = *footprint;
    }
  }
  for (const auto &[root, bytes] : sharedRoots)
    result.cost.sharedBytesAndOccupancy =
        add(result.cost.sharedBytesAndOccupancy, bytes, result.saturated);
  for (const auto &[id, bytes] : syntheticRoots)
    result.cost.sharedBytesAndOccupancy =
        add(result.cost.sharedBytesAndOccupancy, bytes, result.saturated);
  return result;
}
