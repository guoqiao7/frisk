#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"

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
  if (source != target) {
    uint64_t sourceCarriers = 1, targetCarriers = 1;
    for (int64_t extent : src.getTopology().asArrayRef())
      sourceCarriers = multiply(sourceCarriers, extent, result.saturated);
    for (int64_t extent : dst.getTopology().asArrayRef())
      targetCarriers = multiply(targetCarriers, extent, result.saturated);
    uint64_t bits = type.getElementTypeBitWidth();
    uint64_t elementBytes = bits / 8 + (bits % 8 != 0);
    bytes = multiply(elementBytes,
        add(sourceCarriers, targetCarriers, result.saturated), result.saturated);
    sync = 2;
    uint64_t synchronization = multiply(4, ctaThreads, result.saturated);
    synchronization = multiply(synchronization, sync, result.saturated);
    result.cost.conversionBytesAndSync = add(bytes, synchronization, result.saturated);
    result.cost.codeSize = 1;
  }
  edge.sourceEncoding = source;
  edge.targetEncoding = target;
  edge.bytes = bytes;
  edge.synchronizationCost = sync;
  return result;
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
