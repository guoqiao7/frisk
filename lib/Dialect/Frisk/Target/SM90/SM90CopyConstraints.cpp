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
InstructionContractBuildResult reject(StringRef code, StringRef reason,
    std::optional<unsigned> operand = {}, ArrayRef<int64_t> point = {},
    ProofStatus status = ProofStatus::Disproven) {
  return {{}, {status, SmallVector<int64_t>(point), reason.str()}, code.str(), operand};
}

// Categorize an existing lower-level proof, never manufacture its status or
// rerun it. Keep the complete reason and coordinate as the authoritative detail.
InstructionContractBuildResult rejectProof(LayoutProof proof, unsigned operand) {
  StringRef reason(proof.reason);
  StringRef code = "copy-address-contiguity";
  if (reason.contains("budget") || reason.contains("65536") || reason.contains("overflow"))
    code = "copy-budget";
  else if (reason.contains("alignment") || reason.contains("aligned"))
    code = "copy-alignment";
  else if (reason.contains("writer") || reason.contains("owner"))
    code = "copy-writer";
  else if (reason.contains("shape") || reason.contains("rank") || reason.contains("domain must exactly"))
    code = "copy-shape";
  else if (reason.contains("memory space"))
    code = "copy-direction";
  else if (proof.status == ProofStatus::Unknown || reason.contains("malformed") ||
           reason.contains("unsupported"))
    code = "copy-decoder";
  return {{}, std::move(proof), code.str(), operand};
}

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
InstructionContractBuildResult tmaDescriptor(const StorageAliasInfo &source,
    const StorageAliasInfo &destination, const StorageAliasFootprint &src,
    const StorageAliasFootprint &dst) {
  auto type = source.viewType;
  auto element = type.getElementType();
  unsigned bits = type.getElementTypeBitWidth();
  if ((!element.isF16() && !element.isBF16() && !element.isF32()) ||
      type.getRank() < 1 || type.getRank() > 5)
    return reject("copy-descriptor", "TMA descriptor requires rank 1..5 and f16/bf16/f32", 0);
  if (source.rootAlignment < 16)
    return reject("copy-alignment", "TMA source root alignment is less than 16 bytes", 0);
  if (destination.rootAlignment < 128)
    return reject("copy-alignment", "TMA destination root alignment is less than 128 bytes", 1);
  auto shape = type.getShape();
  for (auto size : shape) if (size <= 0 || size > 256)
    return reject("copy-descriptor", "TMA box extent is outside 1..256", 0);
  uint64_t bytes = bits / 8;
  uint64_t inner = shape.back() * bytes;
  if (inner % 16)
    return reject("copy-descriptor", "TMA innermost box byte width is not a multiple of 16", 0);
  auto s = addresses(src), d = addresses(dst);
  if (failed(s) || failed(d))
    return reject("copy-decoder", "TMA descriptor requires byte-addressable actual footprints",
                  failed(s) ? 0 : 1);
  SmallVector<int64_t> zero(shape.size(), 0), strides(shape.size(), bytes);
  uint64_t start = s->at(zero), dstStart = d->at(zero);
  if (start % 16)
    return reject("copy-alignment", "TMA actual source start is not 16-byte aligned", 0, zero);
  if (dstStart % 128)
    return reject("copy-alignment", "TMA actual destination start is not 128-byte aligned", 1, zero);
  for (int64_t axis = shape.size() - 2; axis >= 0; --axis) {
    auto point = zero; point[axis] = 1;
    if (shape[axis] == 1) strides[axis] = strides[axis + 1] * shape[axis + 1];
    else {
      uint64_t address = s->at(point);
      if (address < start || address - start >= (uint64_t{1} << 40))
        return reject("copy-descriptor", "TMA outer stride is outside the positive 40-bit descriptor range", 0, point);
      strides[axis] = address - start;
    }
    if (strides[axis] % 16 || strides[axis] < strides[axis + 1] * shape[axis + 1])
      return reject("copy-descriptor", "TMA outer stride is not a non-overlapping multiple of 16 bytes", 0, point);
  }
  for (const auto &[point, address] : *s) {
    uint64_t expected = start;
    for (auto [axis, coordinate] : llvm::enumerate(point)) expected += coordinate * strides[axis];
    if (address != expected)
      return reject("copy-descriptor", "TMA source addresses do not match descriptor strides", 0, point);
  }
  SmallVector<int64_t> representativeMismatch;
  for (unsigned swizzle : {0u, 32u, 64u, 128u}) {
    if (swizzle && (inner > swizzle || destination.rootAlignment < swizzle * 8 ||
                    dstStart % (swizzle * 8))) continue;
    bool matches = true;
    for (const auto &[point, address] : *d) {
      uint64_t flat = 0;
      for (auto [axis, coordinate] : llvm::enumerate(point)) flat = flat * shape[axis] + coordinate;
      uint64_t expected = dstStart + flat * bytes;
      if (swizzle) expected ^= ((expected >> 7) & (swizzle / 16 - 1)) << 4;
      if (address != expected) {
        // The unswizzled alternative is always tested first. Keep its first
        // mismatch, without claiming that point witnesses every swizzle.
        if (swizzle == 0) representativeMismatch = point;
        matches = false; break;
      }
    }
    if (!matches) continue;
    Builder b(type.getContext());
    SmallVector<int64_t> box(shape.rbegin(), shape.rend());
    SmallVector<int64_t> globalStrides;
    for (int64_t i = strides.size() - 2; i >= 0; --i) globalStrides.push_back(strides[i]);
    auto descriptor = b.getDictionaryAttr({
        b.getNamedAttr("box", b.getDenseI64ArrayAttr(box)),
        b.getNamedAttr("global_strides", b.getDenseI64ArrayAttr(globalStrides)),
        b.getNamedAttr("source_offset", b.getI64IntegerAttr(start)),
        b.getNamedAttr("destination_offset", b.getI64IntegerAttr(dstStart)),
        b.getNamedAttr("swizzle_bytes", b.getI64IntegerAttr(swizzle)),
        b.getNamedAttr("interleave", b.getStringAttr("none"))});
    return {descriptor, {ProofStatus::Proven, {}, "actual TMA descriptor addresses proven"}, {}, {}};
  }
  return reject("copy-descriptor", "TMA destination addresses match no admitted linear/swizzled descriptor; representative mismatch for swizzle_bytes=0",
                1, representativeMismatch);
}
} // namespace

InstructionContractBuildResult mlir::frisk::buildSM90CopyContractDetailed(const LayoutConstraintGraph &graph,
    const LayoutConstraint &c, ArrayRef<Attribute> encodings) {
  if (!c.instruction || c.vars.size() != 4 || encodings.size() != 4)
    return reject("copy-shape", "Copy contract requires four roles");
  auto op = dyn_cast_or_null<CopyOp>(c.instruction->source);
  auto src = dyn_cast_or_null<StorageLayoutAttr>(encodings[0]);
  auto dst = dyn_cast_or_null<StorageLayoutAttr>(encodings[1]);
  auto execution = dyn_cast_or_null<DistributedEncodingAttr>(encodings[2]);
  auto path = dyn_cast_or_null<StringAttr>(encodings[3]);
  if (!op || !src || !dst || !execution || !isSM90CopyPath(path))
    return reject("copy-decoder", "unsupported Copy role type or instruction path", {}, {}, ProofStatus::Unknown);
  for (auto id : c.vars)
    if (id >= graph.getVariables().size())
      return reject("copy-shape", "Copy role references an invalid graph variable");
  auto target = getSM90TargetInfo(op);
  if (failed(target))
    return reject("copy-target", "Copy target is outside the SM90 support table", {}, {}, ProofStatus::Unknown);
  const auto &source = graph.getVariable(c.vars[0]);
  const auto &destination = graph.getVariable(c.vars[1]);
  const auto &executing = graph.getVariable(c.vars[2]);
  if (!source.storageAlias || !destination.storageAlias || !executing.operationExecution)
    return reject("copy-decoder", "missing storage alias or execution facts", {}, {}, ProofStatus::Unknown);
  const auto &s = *source.storageAlias, &d = *destination.storageAlias;
  const auto &binding = *executing.operationExecution;
  auto executionType = dyn_cast<RankedTensorType>(executing.shapedType);
  if (!s.viewType || !d.viewType || !executionType ||
      s.viewType.getShape() != d.viewType.getShape() ||
      s.viewType.getElementType() != d.viewType.getElementType() ||
      executionType.getElementType() != s.viewType.getElementType())
    return reject("copy-shape", "Copy source, destination and execution element/shape facts disagree");
  bool guarded = executionType.getShape() != s.viewType.getShape();
  if (guarded && (path.getValue() != "vector" || binding.vectorBytes != 1))
    return reject("copy-shape", "guarded Copy only admits the scalar vector path");
  const auto &sf = getStorageAliasFootprint(graph, source.id, src);
  if (sf.proof.status != ProofStatus::Proven) return rejectProof(sf.proof, 0);
  const auto &df = getStorageAliasFootprint(graph, destination.id, dst);
  if (df.proof.status != ProofStatus::Proven) return rejectProof(df.proof, 1);
  // The mandatory width and wider alternatives share proof results. In
  // particular, a successful build is already checked by this construction.
  std::map<unsigned, InstructionContractBuildResult> widths;
  auto validVector = [&](unsigned width) -> const InstructionContractBuildResult & {
    auto [it, inserted] = widths.try_emplace(width);
    if (!inserted) return it->second;
    for (unsigned operand : {0u, 1u}) {
      const auto &info = operand ? d : s;
      const auto &footprint = operand ? df : sf;
      auto access = operand ? AccessKind::Write : AccessKind::Read;
      auto proof = guarded
          ? proveGuardedExecutionVectorAccess(execution, executionType, info, footprint,
                width, access, binding.writerPolicy)
          : proveExecutionVectorAccess(execution, info, footprint, width, access,
                binding.writerPolicy);
      if (proof.status != ProofStatus::Proven) {
        it->second = rejectProof(std::move(proof), operand);
        return it->second;
      }
    }
    it->second.proof = {ProofStatus::Proven, {}, "source/destination vector access proven"};
    return it->second;
  };
  const auto &mandatory = validVector(binding.vectorBytes);
  if (mandatory.proof.status != ProofStatus::Proven) return mandatory;
  bool globalShared = src.getMemorySpace().getValue() == attr::MemorySpace::Global &&
                      dst.getMemorySpace().getValue() == attr::MemorySpace::Shared;
  unsigned width = binding.vectorBytes;
  // The selected path can prove a wider implementation without weakening the
  // user's minimum vector requirement. Its exact width is stored in the plan.
  if (path.getValue() != "tma" && !guarded)
    for (unsigned candidate : {16u, 8u, 4u, 2u, 1u})
      if (candidate >= binding.vectorBytes &&
          validVector(candidate).proof.status == ProofStatus::Proven) { width = candidate; break; }
  if (path.getValue() != "vector" && !globalShared)
    return reject("copy-direction", "async Copy requires Global source and Shared destination");
  if (path.getValue() == "cp.async") {
    if (!target->supportsCpAsync)
      return reject("copy-target", "target does not support cp.async");
    if (width < 4) return validVector(4);
  }
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
    if (!target->supportsTMA)
      return reject("copy-target", "target does not support TMA");
    auto descriptor = tmaDescriptor(s, d, sf, df);
    if (descriptor.proof.status != ProofStatus::Proven) return descriptor;
    attrs.set("descriptor", descriptor.binding);
  }
  return {attrs.getDictionary(op.getContext()),
          {ProofStatus::Proven, {}, "sm90-copy-contract: actual addresses and execution proven"}, {}, {}};
}

FailureOr<Attribute> mlir::frisk::buildSM90CopyContract(
    const LayoutConstraintGraph &graph, const LayoutConstraint &c,
    ArrayRef<Attribute> encodings) {
  auto result = buildSM90CopyContractDetailed(graph, c, encodings);
  if (result.proof.status != ProofStatus::Proven || !result.binding) return failure();
  return result.binding;
}

LayoutProof mlir::frisk::verifySM90CopyContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &c, ArrayRef<Attribute> encodings, Attribute binding) {
  auto expected = buildSM90CopyContractDetailed(graph, c, encodings);
  if (expected.proof.status != ProofStatus::Proven) return expected.proof;
  if (expected.binding != binding)
    return {ProofStatus::Disproven, {}, "sm90-copy-contract: descriptor, path, guard or completion differs from proven actual operands"};
  return {ProofStatus::Proven, {}, "sm90-copy-contract: actual addresses and execution proven"};
}
