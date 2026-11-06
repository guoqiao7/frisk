#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include <array>

using namespace mlir;
using namespace mlir::frisk;
namespace {
constexpr uint64_t budget = 65536;
constexpr std::array<StringLiteral, 5> carriers = {
    "register", "lane", "warp", "warp_group", "cta"};
LayoutProof reject(StringRef reason, ArrayRef<int64_t> point = {}) {
  return {ProofStatus::Disproven, SmallVector<int64_t>(point), reason.str()};
}
LayoutProof proven() { return {ProofStatus::Proven, {}, "bounded execution proof"}; }
using Enumeration = ExecutionEnumeration;
LayoutProof enumerate(DistributedEncodingAttr execution, RankedTensorType type,
                      StringRef policy, Enumeration &out) {
  LayoutProfile::Scope profile(LayoutProfileKind::ExecutionEnumeration);
  if (policy != "all" && policy != "first_owner")
    return reject("unknown writer policy");
  if (!execution || !type || !type.hasStaticShape())
    return reject("execution requires a static logical tensor");
  auto topology = execution.getTopology();
  auto map = dyn_cast_or_null<BitLinearLayoutMapAttr>(execution.getMap());
  if (!topology || topology.size() != 5 || !map)
    return reject("malformed hardware topology or non-bit-linear execution map");
  uint64_t hardwareCount = 1, logicalCount = 1;
  for (unsigned c = 0; c < 5; ++c) {
    int64_t extent = topology[c];
    if (extent <= 0 || !llvm::isPowerOf2_64(extent) || (c == 1 && extent > 32))
      return reject("invalid hardware topology extent", {int64_t(c), extent});
    if (uint64_t(extent) > budget / hardwareCount)
      return {ProofStatus::Unknown, {}, "hardware enumeration budget exceeded (65536)"};
    hardwareCount *= extent;
    out.topology[c] = extent;
  }
  for (int64_t extent : type.getShape()) {
    if (extent <= 0 || !llvm::isPowerOf2_64(extent))
      return reject("logical tile must have positive power-of-two extents");
    if (uint64_t(extent) > budget / logicalCount)
      return {ProofStatus::Unknown, {}, "logical enumeration budget exceeded (65536)"};
    logicalCount *= extent;
  }
  auto names = map.getInputNames(), outputNames = map.getOutputNames();
  auto widths = map.getInputBitWidths(), outputWidths = map.getOutputBitWidths();
  if (!names || !widths || !outputNames || !outputWidths ||
      names.size() != widths.size() || outputNames.size() != outputWidths.size() ||
      outputWidths.size() != unsigned(type.getRank()))
    return reject("malformed execution map dimensions");
  SmallVector<unsigned> indices;
  llvm::StringSet<> seen;
  unsigned inputBits = 0, outputBits = 0;
  for (unsigned i = 0; i < names.size(); ++i) {
    auto name = dyn_cast<StringAttr>(names[i]);
    if (!name || !seen.insert(name.getValue()).second)
      return reject("invalid or repeated execution carrier");
    unsigned c = 0;
    while (c < 5 && carriers[c] != name.getValue()) ++c;
    if (c == 5 || widths[i] < 0 || widths[i] > 16 ||
        (uint64_t(1) << widths[i]) != out.topology[c])
      return reject("carrier width does not match hardware topology");
    indices.push_back(c);
    inputBits += widths[i];
  }
  seen.clear();
  for (unsigned i = 0; i < outputWidths.size(); ++i) {
    auto name = dyn_cast<StringAttr>(outputNames[i]);
    if (!name || !seen.insert(name.getValue()).second || outputWidths[i] < 0 ||
        outputWidths[i] > 16 || (int64_t(1) << outputWidths[i]) != type.getDimSize(i))
      return reject("logical output widths do not match tensor shape");
    outputBits += outputWidths[i];
  }
  auto matrix = map.getMatrix();
  auto matrixType = matrix ? dyn_cast<RankedTensorType>(matrix.getType()) : RankedTensorType();
  if (!matrixType || matrixType.getRank() != 2 ||
      !matrixType.getElementType().isInteger(1) ||
      matrixType.getDimSize(0) != outputBits || matrixType.getDimSize(1) != inputBits)
    return reject("malformed execution matrix");
  SmallVector<uint64_t> rows(outputBits, 0);
  unsigned bitIndex = 0;
  for (APInt value : matrix.getValues<APInt>()) {
    if (!value.isZero()) rows[bitIndex / inputBits] |= uint64_t(1) << (bitIndex % inputBits);
    ++bitIndex;
  }
  out.shape.assign(type.getShape().begin(), type.getShape().end());
  out.first.assign(logicalCount, hardwareCount);
  out.logical.reserve(hardwareCount);
  for (uint64_t h = 0; h < hardwareCount; ++h) {
    auto point = out.hardware(h);
    uint64_t input = 0;
    unsigned shift = 0;
    for (unsigned i = 0; i < indices.size(); ++i) {
      input |= uint64_t(point[4-indices[i]]) << shift;
      shift += widths[i];
    }
    uint64_t output = 0;
    for (unsigned r = 0; r < rows.size(); ++r)
      output |= uint64_t(llvm::popcount(rows[r] & input) & 1) << r;
    uint64_t flat = 0;
    shift = 0;
    for (unsigned i = 0; i < outputWidths.size(); ++i) {
      uint64_t coordinate = (output >> shift) & ((uint64_t(1) << outputWidths[i])-1);
      flat = flat * type.getDimSize(i) + coordinate;
      shift += outputWidths[i];
    }
    out.logical.push_back(flat);
    if (out.first[flat] == hardwareCount) out.first[flat] = h;
  }
  for (uint64_t i = 0; i < logicalCount; ++i)
    if (out.first[i] == hardwareCount)
      return reject("logical coordinate has no hardware owner", out.coordinate(i));
  auto replication = execution.getReplication();
  if (!replication || replication.getInt() <= 0 ||
      uint64_t(replication.getInt()) != hardwareCount / logicalCount)
    return reject("replication does not match actual hardware ownership");
  if (policy == "all")
    for (uint64_t h = 0; h < hardwareCount; ++h)
      if (out.first[out.logical[h]] != h)
        return reject("multiple writers for logical coordinate; hardware counterexample in cta,warp_group,warp,lane,register order", out.hardware(h));
  return proven();
}
} // namespace

LayoutProof mlir::frisk::enumerateExecutionLayout(DistributedEncodingAttr execution,
    RankedTensorType logicalType, ExecutionEnumeration &out) {
  out = ExecutionEnumeration{};
  return enumerate(execution, logicalType, "first_owner", out);
}

LayoutProof mlir::frisk::proveExecutionOwnership(DistributedEncodingAttr execution,
    RankedTensorType logicalType, StringRef writerPolicy) {
  if (writerPolicy != "all" && writerPolicy != "first_owner")
    return reject("unknown writer policy");
  LinearExecutionLayout layout;
  auto proof = analyzeLinearExecutionLayout(execution, logicalType, layout);
  if (proof.status != ProofStatus::Proven) return proof;
  // A zero-row or zero-column map has rank zero. Handle that domain before
  // requesting nonempty GF2 algebra, then retain the same coverage,
  // replication and writer-policy checks below.
  FailureOr<GF2Matrix> matrix = failure();
  unsigned rank = 0;
  if (layout.logicalBits && layout.hardwareBits) {
    matrix = layout.getMatrix();
    if (succeeded(matrix)) rank = matrix->rank();
  }
  if (rank != layout.logicalBits) {
    // A nonzero left-kernel vector separates the image from at least one
    // coordinate unit vector. That unit is a deterministic uncovered value.
    APInt missing(std::max(1u, layout.logicalBits), 0);
    unsigned bit = 0;
    if (succeeded(matrix))
      bit = matrix->transpose().kernelBasis().front().countTrailingZeros();
    missing.setBit(bit);
    return reject("symbolic ownership: logical coordinate has no hardware owner",
                  layout.logicalCoordinate(missing));
  }
  unsigned replicationBits = layout.hardwareBits - layout.logicalBits;
  // The attribute is a positive signed 64-bit value. Never form 2^k when it
  // cannot be represented, and never multiply the complete hardware domain.
  if (replicationBits >= 63 ||
      execution.getReplication().getValue().getZExtValue() != (uint64_t(1) << replicationBits))
    return reject("symbolic ownership: replication does not match actual hardware ownership");
  if (writerPolicy == "all" && rank != layout.hardwareBits) {
    APInt duplicate(std::max(1u, layout.hardwareBits), 0);
    if (succeeded(matrix)) duplicate = matrix->kernelBasis().front();
    else duplicate.setBit(0);
    return reject("symbolic ownership: multiple writers; hardware counterexample in cta,warp_group,warp,lane,register order",
                  layout.hardwareCoordinate(duplicate));
  }
  return {ProofStatus::Proven, {}, "symbolic execution ownership proof"};
}

static LayoutProof proveAccess(DistributedEncodingAttr execution,
    RankedTensorType executionType,
    const StorageAliasInfo &info, const StorageAliasFootprint &storage,
    unsigned vectorBytes, AccessKind access, StringRef writerPolicy) {
  if (vectorBytes != 1 && vectorBytes != 2 && vectorBytes != 4 && vectorBytes != 8 && vectorBytes != 16)
    return reject("vector byte width must be one of 1,2,4,8,16");
  if (storage.proof.status != ProofStatus::Proven)
    return {storage.proof.status, storage.proof.counterexample, "storage footprint is not proven: " + storage.proof.reason};
  if (!info.viewType || !info.rootType || storage.root != info.root || storage.rootType != info.rootType)
    return reject("storage footprint root/type mismatch");
  auto element = info.viewType.getElementType();
  if (!element.isIntOrFloat()) return reject("unsupported storage element type");
  uint64_t bits = element.getIntOrFloatBitWidth();
  if (!bits) return reject("zero-width storage element");
  Enumeration points;
  auto proof = enumerate(execution, executionType, writerPolicy, points);
  if (proof.status != ProofStatus::Proven) return proof;
  auto shape = info.viewType.getShape();
  uint64_t actualCount = 1;
  for (int64_t extent : shape) {
    if (extent <= 0 || uint64_t(extent) > budget / actualCount)
      return reject("storage logical domain exceeds bounded positive tile");
    actualCount *= extent;
  }
  if (storage.entries.size() != actualCount)
    return reject("storage footprint does not cover logical tile");
  SmallVector<const StorageAliasPointAddress *> addresses(actualCount, nullptr);
  for (const auto &entry : storage.entries) {
    auto root = evaluateStorageViewCoordinates(info, entry.view);
    if (failed(root) || *root != entry.root)
      return reject("storage footprint root coordinate does not match the view descriptor", entry.view);
    if (entry.view.size() != shape.size()) return reject("footprint coordinate rank mismatch", entry.view);
    uint64_t flat = 0;
    for (unsigned i = 0; i < entry.view.size(); ++i) {
      if (entry.view[i] < 0 || entry.view[i] >= shape[i])
        return reject("footprint coordinate outside logical tile", entry.view);
      flat = flat * shape[i] + entry.view[i];
    }
    if (addresses[flat] || entry.begin < info.lowerBit || entry.end > info.upperBit ||
        entry.end < entry.begin || entry.end - entry.begin != bits)
      return reject("inconsistent or out-of-bounds storage footprint", entry.view);
    addresses[flat] = &entry;
  }
  // A Proven marker is not permission to consume inconsistent cached data.
  // Check physical intervals independently of the input entry order, including
  // the packed scalar baseline, without disturbing logical address indexing.
  auto intervals = addresses;
  llvm::sort(intervals, [](const auto *a, const auto *b) {
    return a->begin < b->begin;
  });
  for (unsigned i = 1; i < intervals.size(); ++i)
    if (intervals[i]->begin < intervals[i-1]->end)
      return reject("storage footprint physical bit intervals overlap", intervals[i]->view);
  // Scalar guarded execution consumes only the actual-domain table above.
  // Enumeration proves every padded point has an elected owner, hence every
  // active point does too. No address is looked up or synthesized for a tail.
  if (vectorBytes == 1) return proven(); // Packed scalar baseline.
  if (bits % 8) return reject("packed elements cannot use wide vectors");
  uint64_t bytes = bits / 8;
  if (!info.rootAlignment || info.rootAlignment % vectorBytes)
    return reject("root alignment evidence is insufficient for vector access");
  uint64_t chunk = bytes >= vectorBytes ? 1 : vectorBytes / bytes;
  if ((bytes >= vectorBytes && bytes % vectorBytes) ||
      (bytes < vectorBytes && vectorBytes % bytes))
    return reject("vector width does not contain complete scalar subdivisions");
  auto selected = [&](uint64_t h) {
    return access == AccessKind::Read || points.first[points.logical[h]] == h;
  };
  uint64_t registers = points.topology[0];
  for (uint64_t thread = 0; thread < points.logical.size(); thread += registers) {
    for (uint64_t reg = 0; reg < registers; reg += chunk) {
      uint64_t end = std::min(registers, reg + chunk);
      bool any = false;
      for (uint64_t r = reg; r < end; ++r) any |= selected(thread+r);
      if (!any) continue;
      if (end-reg != chunk)
        return reject("incomplete per-thread register vector tail", points.hardware(thread+reg));
      uint64_t begin = addresses[points.logical[thread+reg]]->begin;
      if (begin % (vectorBytes*8))
        return reject("root-relative vector address is misaligned", points.hardware(thread+reg));
      for (uint64_t r = reg; r < end; ++r) {
        uint64_t h = thread+r;
        if (!selected(h)) return reject("elected writers leave a register vector hole", points.hardware(h));
        // All addresses are bounded by upperBit; subtract to avoid overflow.
        uint64_t address = addresses[points.logical[h]]->begin;
        if (address < begin || address-begin != (r-reg)*bits)
          return reject("per-thread register vector addresses are not contiguous", points.hardware(h));
      }
    }
  }
  return proven();
}

LayoutProof mlir::frisk::proveExecutionVectorAccess(DistributedEncodingAttr execution,
    const StorageAliasInfo &info, const StorageAliasFootprint &storage,
    unsigned vectorBytes, AccessKind access, StringRef writerPolicy) {
  if (!info.viewType) return reject("missing storage view type");
  return proveAccess(execution,
      RankedTensorType::get(info.viewType.getShape(), info.viewType.getElementType()),
      info, storage, vectorBytes, access, writerPolicy);
}

LayoutProof mlir::frisk::proveGuardedExecutionVectorAccess(
    DistributedEncodingAttr execution, RankedTensorType padded,
    const StorageAliasInfo &info, const StorageAliasFootprint &storage,
    unsigned vectorBytes, AccessKind access, StringRef writerPolicy) {
  if (vectorBytes != 1) return reject("guarded Copy requires scalar vector_bytes=1");
  if (!info.viewType || !info.viewType.hasStaticShape() || !padded ||
      !padded.hasStaticShape() || padded.getRank() == 0 ||
      padded.getRank() != info.viewType.getRank() ||
      padded.getElementType() != info.viewType.getElementType())
    return reject("guarded execution requires matching static storage and execution types");
  uint64_t actualCount = 1, paddedCount = 1;
  for (auto [actual, carrier] : llvm::zip_equal(info.viewType.getShape(), padded.getShape())) {
    if (actual <= 0 || uint64_t(actual) > budget / actualCount)
      return reject("guarded logical enumeration budget exceeded (65536)");
    actualCount *= actual;
    uint64_t expected = llvm::PowerOf2Ceil(uint64_t(actual));
    if (carrier <= 0 || uint64_t(carrier) != expected)
      return reject("guarded execution shape is not the exact padded storage shape");
    if (expected > budget / paddedCount)
      return reject("guarded padded enumeration budget exceeded (65536)");
    paddedCount *= expected;
  }
  return proveAccess(execution, padded, info, storage, vectorBytes, access, writerPolicy);
}
