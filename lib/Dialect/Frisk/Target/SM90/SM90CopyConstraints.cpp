#include "Dialect/Frisk/Target/SM90/SM90CopyConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90TargetInfo.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"
#include <map>

using namespace mlir;
using namespace mlir::frisk;

bool mlir::frisk::isSM90CopyPath(Attribute candidate) {
  auto path = dyn_cast_or_null<StringAttr>(candidate);
  return path && (path.getValue() == "vector" || path.getValue() == "cp.async" ||
                  path.getValue() == "tma");
}

LogicalResult mlir::frisk::prepareSM90CopyCandidates(LayoutConstraintGraph &graph) {
  for (auto &v : graph.getVariables()) {
    if (v.kind != LayoutKind::Instruction || !isa_and_nonnull<CopyOp>(v.anchor)) continue;
    Builder b(v.anchor->getContext());
    for (StringRef path : {"vector", "cp.async", "tma"}) {
      ++graph.getCandidatePreparationStatistics().generatedProposals;
      v.candidates.push_back({b.getStringAttr(path), kInvalidProvenanceID, v.candidates.size()});
    }
    if (auto binding = v.anchor->getAttrOfType<DictionaryAttr>("frisk.copy_contract")) {
      auto path = binding.getAs<StringAttr>("path");
      if (!isSM90CopyPath(path)) return v.anchor->emitError("invalid Copy instruction path");
      v.candidates = {{path, kInvalidProvenanceID, 0}};
    }
  }
  return success();
}

namespace {
using Addresses = std::map<SmallVector<int64_t>, uint64_t>;
FailureOr<Addresses> addresses(const StorageAliasFootprint &footprint) {
  if (footprint.proof.status != ProofStatus::Proven) return failure();
  Addresses result;
  for (const auto &p : footprint.entries) {
    if (p.begin % 8 || p.end % 8) return failure();
    result.emplace(p.view, p.begin / 8);
  }
  return result;
}

// Reconstruct both descriptor strides and destination swizzle from actual root
// addresses, rather than inferring legality from a layout's descriptive name.
FailureOr<DictionaryAttr> tmaDescriptor(const StorageAliasInfo &source,
    const StorageAliasInfo &destination, const StorageAliasFootprint &src,
    const StorageAliasFootprint &dst) {
  auto type = source.viewType;
  auto element = type.getElementType();
  unsigned bits = type.getElementTypeBitWidth();
  if ((!element.isF16() && !element.isBF16() && !element.isF32()) ||
      type.getRank() < 1 || type.getRank() > 5 || source.rootAlignment < 16 ||
      destination.rootAlignment < 128) return failure();
  auto shape = type.getShape();
  for (auto size : shape) if (size <= 0 || size > 256) return failure();
  uint64_t bytes = bits / 8;
  uint64_t inner = shape.back() * bytes;
  if (inner % 16) return failure();
  auto s = addresses(src), d = addresses(dst);
  if (failed(s) || failed(d)) return failure();
  SmallVector<int64_t> zero(shape.size(), 0), strides(shape.size(), bytes);
  uint64_t start = s->at(zero), dstStart = d->at(zero);
  if (start % 16 || dstStart % 128) return failure();
  for (int64_t axis = shape.size() - 2; axis >= 0; --axis) {
    if (shape[axis] == 1) strides[axis] = strides[axis + 1] * shape[axis + 1];
    else {
      auto point = zero; point[axis] = 1;
      uint64_t address = s->at(point);
      if (address < start || address - start >= (uint64_t{1} << 40)) return failure();
      strides[axis] = address - start;
    }
    if (strides[axis] % 16 || strides[axis] < strides[axis + 1] * shape[axis + 1])
      return failure();
  }
  for (const auto &[point, address] : *s) {
    uint64_t expected = start;
    for (auto [axis, coordinate] : llvm::enumerate(point)) expected += coordinate * strides[axis];
    if (address != expected) return failure();
  }
  for (unsigned swizzle : {0u, 32u, 64u, 128u}) {
    if (swizzle && (inner > swizzle || destination.rootAlignment < swizzle * 8 ||
                    dstStart % (swizzle * 8))) continue;
    bool matches = true;
    for (const auto &[point, address] : *d) {
      uint64_t flat = 0;
      for (auto [axis, coordinate] : llvm::enumerate(point)) flat = flat * shape[axis] + coordinate;
      uint64_t expected = dstStart + flat * bytes;
      if (swizzle) expected ^= ((expected >> 7) & (swizzle / 16 - 1)) << 4;
      if (address != expected) { matches = false; break; }
    }
    if (!matches) continue;
    Builder b(type.getContext());
    SmallVector<int64_t> box(shape.rbegin(), shape.rend());
    SmallVector<int64_t> globalStrides;
    for (int64_t i = strides.size() - 2; i >= 0; --i) globalStrides.push_back(strides[i]);
    return b.getDictionaryAttr({
        b.getNamedAttr("box", b.getDenseI64ArrayAttr(box)),
        b.getNamedAttr("global_strides", b.getDenseI64ArrayAttr(globalStrides)),
        b.getNamedAttr("source_offset", b.getI64IntegerAttr(start)),
        b.getNamedAttr("destination_offset", b.getI64IntegerAttr(dstStart)),
        b.getNamedAttr("swizzle_bytes", b.getI64IntegerAttr(swizzle)),
        b.getNamedAttr("interleave", b.getStringAttr("none"))});
  }
  return failure();
}
} // namespace

FailureOr<Attribute> mlir::frisk::buildSM90CopyContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &c, ArrayRef<Attribute> encodings) {
  if (!c.instruction || c.vars.size() != 4 || encodings.size() != 4) return failure();
  auto op = dyn_cast_or_null<CopyOp>(c.instruction->source);
  auto src = dyn_cast<StorageLayoutAttr>(encodings[0]);
  auto dst = dyn_cast<StorageLayoutAttr>(encodings[1]);
  auto execution = dyn_cast<DistributedEncodingAttr>(encodings[2]);
  auto path = dyn_cast<StringAttr>(encodings[3]);
  if (!op || !src || !dst || !execution || !isSM90CopyPath(path)) return failure();
  auto target = getSM90TargetInfo(op);
  if (failed(target)) return failure();
  const auto &source = graph.getVariable(c.vars[0]);
  const auto &destination = graph.getVariable(c.vars[1]);
  const auto &executing = graph.getVariable(c.vars[2]);
  if (!source.storageAlias || !destination.storageAlias || !executing.operationExecution)
    return failure();
  const auto &s = *source.storageAlias, &d = *destination.storageAlias;
  const auto &binding = *executing.operationExecution;
  const auto &sf = getStorageAliasFootprint(graph, source.id, src);
  const auto &df = getStorageAliasFootprint(graph, destination.id, dst);
  auto executionType = cast<RankedTensorType>(executing.shapedType);
  bool guarded = executionType.getShape() != s.viewType.getShape();
  if (guarded && (path.getValue() != "vector" || binding.vectorBytes != 1)) return failure();
  auto validVector = [&](unsigned width) {
    if (guarded)
      return proveGuardedExecutionVectorAccess(execution, executionType, s, sf, width,
                 AccessKind::Read, binding.writerPolicy).status == ProofStatus::Proven &&
          proveGuardedExecutionVectorAccess(execution, executionType, d, df, width,
                 AccessKind::Write, binding.writerPolicy).status == ProofStatus::Proven;
    return proveExecutionVectorAccess(execution, s, sf, width, AccessKind::Read,
               binding.writerPolicy).status == ProofStatus::Proven &&
        proveExecutionVectorAccess(execution, d, df, width, AccessKind::Write,
               binding.writerPolicy).status == ProofStatus::Proven;
  };
  if (!validVector(binding.vectorBytes)) return failure();
  bool globalShared = src.getMemorySpace().getValue() == attr::MemorySpace::Global &&
                      dst.getMemorySpace().getValue() == attr::MemorySpace::Shared;
  unsigned width = binding.vectorBytes;
  // The selected path can prove a wider implementation without weakening the
  // user's minimum vector requirement. Its exact width is stored in the plan.
  if (path.getValue() != "tma" && !guarded)
    for (unsigned candidate : {16u, 8u, 4u, 2u, 1u})
      if (candidate >= binding.vectorBytes && validVector(candidate)) { width = candidate; break; }
  if (path.getValue() == "cp.async" && (!globalShared || width < 4 || !target->supportsCpAsync)) return failure();
  Builder b(op.getContext());
  NamedAttrList attrs;
  attrs.set("version", b.getI64IntegerAttr(1));
  attrs.set("path", path);
  attrs.set("target", b.getStringAttr(target->targetKey));
  attrs.set("vector_bytes", b.getI64IntegerAttr(width));
  attrs.set("valid_shape", b.getDenseI64ArrayAttr(s.viewType.getShape()));
  attrs.set("guarded", b.getBoolAttr(guarded));
  attrs.set("issuer", b.getStringAttr(path.getValue() == "tma" ? "elected-cta-thread" : "execution-map"));
  attrs.set("completion", b.getStringAttr(path.getValue() == "tma" ? "mbarrier-and-async-proxy" :
      path.getValue() == "cp.async" ? "commit-wait-and-cta-visibility" : "synchronous"));
  if (path.getValue() == "tma") {
    if (!globalShared || !target->supportsTMA) return failure();
    auto descriptor = tmaDescriptor(s, d, sf, df);
    if (failed(descriptor)) return failure();
    attrs.set("descriptor", *descriptor);
  }
  return Attribute(attrs.getDictionary(op.getContext()));
}

LayoutProof mlir::frisk::verifySM90CopyContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &c, ArrayRef<Attribute> encodings, Attribute binding) {
  auto expected = buildSM90CopyContract(graph, c, encodings);
  if (failed(expected) || *expected != binding)
    return {ProofStatus::Disproven, {}, "sm90-copy-contract: descriptor, path, guard or completion differs from proven actual operands"};
  return {ProofStatus::Proven, {}, "sm90-copy-contract: actual addresses and execution proven"};
}
