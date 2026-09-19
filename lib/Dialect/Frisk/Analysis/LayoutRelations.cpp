#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::frisk {
const StorageAliasFootprint &getStorageAliasFootprint(
    const LayoutConstraintGraph &graph, LayoutVarID id, Attribute candidate) {
  auto &cache = graph.getAliasFootprintCache();
  auto key = LayoutConstraintGraph::AliasCandidateKey{id, candidate};
  auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  StorageAliasFootprint footprint;
  auto encoding = dyn_cast_or_null<StorageLayoutAttr>(candidate);
  const auto &info = graph.getVariable(id).storageAlias;
  if (info && encoding)
    footprint = buildStorageAliasFootprint(*info, encoding);
  else
    footprint.proof = {ProofStatus::Unknown, {}, "missing coordinate alias metadata or storage encoding"};
  ++graph.getCandidatePreparationStatistics().footprintEvaluations;
  return cache.try_emplace(key, std::move(footprint)).first->second;
}

LayoutProof proveAliasLayoutRelation(const LayoutConstraintGraph &graph,
                                    LayoutVarID lhs, Attribute a,
                                    LayoutVarID rhs, Attribute b) {
  if (rhs < lhs) { std::swap(lhs, rhs); std::swap(a, b); }
  auto key = LayoutConstraintGraph::AliasPairKey{{lhs, a}, {rhs, b}};
  auto &cache = graph.getAliasPairCache();
  auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  // Populate both before taking references: DenseMap insertion may rehash.
  (void)getStorageAliasFootprint(graph, lhs, a);
  (void)getStorageAliasFootprint(graph, rhs, b);
  auto proof = proveStorageAliasFootprints(
      getStorageAliasFootprint(graph, lhs, a), getStorageAliasFootprint(graph, rhs, b));
  ++graph.getCandidatePreparationStatistics().pairProofEvaluations;
  cache.try_emplace(key, proof);
  return proof;
}

std::string layoutCandidateKey(Attribute value) {
  std::string key;
  llvm::raw_string_ostream(key) << value;
  return key;
}

bool isSupportedLayoutRelation(ConstraintKind kind) {
  return kind == ConstraintKind::SameLayout ||
         kind == ConstraintKind::AliasLayout ||
         kind == ConstraintKind::StorageAccess ||
         kind == ConstraintKind::CopyAccess ||
         kind == ConstraintKind::TransformLayout ||
         kind == ConstraintKind::Convertible;
}

bool isSupportedUnaryLayoutConstraint(ConstraintKind kind) {
  return kind == ConstraintKind::Ownership || kind == ConstraintKind::ResourceLimit;
}

bool matchesLayoutThreadCount(Attribute candidate, int64_t threads) {
  auto encoding = dyn_cast_or_null<DistributedEncodingAttr>(candidate);
  if (!encoding || threads <= 0) return false;
  auto topology = encoding.getTopology();
  if (topology.size() != 5 || topology[1] != 32 || topology[4] != 1) return false;
  uint64_t count = 1;
  for (unsigned i = 1; i <= 3; ++i) {
    if (topology[i] <= 0 || uint64_t(topology[i]) > uint64_t(threads) / count)
      return false;
    count *= topology[i];
  }
  return count == uint64_t(threads);
}

LayoutProof proveUnaryLayoutConstraint(const LayoutConstraintGraph &graph,
                                      const LayoutConstraint &constraint,
                                      Attribute candidate) {
  if (constraint.vars.size() != 1)
    return {ProofStatus::Unknown, {}, "unary execution constraint requires one endpoint"};
  const auto &var = graph.getVariable(constraint.vars.front());
  if (constraint.kind == ConstraintKind::ResourceLimit)
    return {matchesLayoutThreadCount(candidate, constraint.requiredThreads)
                ? ProofStatus::Proven : ProofStatus::Disproven,
            {}, "execution topology must match " + std::to_string(constraint.requiredThreads) + " threads in one CTA"};
  auto layout = dyn_cast_or_null<DistributedEncodingAttr>(candidate);
  auto type = dyn_cast<RankedTensorType>(var.shapedType);
  if (constraint.kind != ConstraintKind::Ownership || !var.operationExecution || !layout || !type)
    return {ProofStatus::Unknown, {}, "missing operation execution ownership contract"};
  return proveExecutionOwnership(layout, type, var.operationExecution->writerPolicy);
}

bool layoutEncodingsEqual(Attribute lhs, Attribute rhs) {
  // SSA tensor type equality is exact, including named canonical-map metadata.
  // Do not equate differently named maps and then produce unequal operand types.
  return lhs == rhs;
}

static bool sameLogicalType(Type lhs, Type rhs) {
  auto a = dyn_cast<ShapedType>(lhs), b = dyn_cast<ShapedType>(rhs);
  return a && b && a.hasStaticShape() && b.hasStaticShape() &&
         a.getShape() == b.getShape() && a.getElementType() == b.getElementType();
}

static FailureOr<Attribute> permuteEncoding(Attribute candidate,
                                           Attribute transform,
                                           ShapedType target, bool inverse,
                                           ArrayAttr outputNames = {}) {
  auto encoding = dyn_cast<DistributedEncodingAttr>(candidate);
  auto permutation = dyn_cast_or_null<DenseI64ArrayAttr>(transform);
  if (!encoding || !permutation)
    return failure();
  auto map = dyn_cast<BitLinearLayoutMapAttr>(encoding.getMap());
  if (!map || permutation.size() != map.getOutputBitWidths().size())
    return failure();
  unsigned rank = permutation.size();
  // Logical output labels describe the destination's positional axes. Prefer
  // an actual destination candidate's names for compatibility; generation may
  // use declared endpoint names as a proposal, never as an extra hard binding.
  if (!outputNames)
    if (auto tensor = dyn_cast<RankedTensorType>(target))
      if (auto declared = dyn_cast_or_null<DistributedEncodingAttr>(tensor.getEncoding()))
        if (auto declaredMap = dyn_cast<BitLinearLayoutMapAttr>(declared.getMap()))
          outputNames = declaredMap.getOutputNames();
  if (!outputNames)
    outputNames = map.getOutputNames();
  if (outputNames.size() != rank)
    return failure();
  SmallVector<int64_t> order(permutation.asArrayRef());
  SmallVector<bool> seen(rank, false);
  for (int64_t dim : order) {
    if (dim < 0 || dim >= rank || seen[dim])
      return failure();
    seen[dim] = true;
  }
  if (inverse) {
    auto copy = order;
    for (unsigned i = 0; i < rank; ++i)
      order[copy[i]] = i;
  }
  SmallVector<unsigned> starts;
  unsigned rows = 0;
  for (int64_t width : map.getOutputBitWidths().asArrayRef()) {
    starts.push_back(rows);
    rows += width;
  }
  SmallVector<APInt> matrixValues;
  auto matrix = map.getMatrix().getValues<APInt>();
  unsigned columns = map.getMatrix().getType().getShape()[1];
  SmallVector<int64_t> widths;
  Builder builder(candidate.getContext());
  for (int64_t source : order) {
    int64_t width = map.getOutputBitWidths()[source];
    widths.push_back(width);
    for (unsigned row = starts[source]; row < starts[source] + width; ++row)
      for (unsigned col = 0; col < columns; ++col)
        matrixValues.push_back(matrix[row * columns + col]);
  }
  auto transformed = BitLinearLayoutMapAttr::get(
      candidate.getContext(), map.getInputNames(), map.getInputBitWidths(),
      outputNames, builder.getDenseI64ArrayAttr(widths),
      DenseIntElementsAttr::get(map.getMatrix().getType(), matrixValues));
  auto result = DistributedEncodingAttr::get(candidate.getContext(), transformed,
                                             encoding.getTopology(),
                                             encoding.getReplication());
  // Shape/permutation mismatch is a missing support, not a speculative diagnostic.
  if (target.getRank() != rank)
    return failure();
  for (unsigned i = 0; i < rank; ++i)
    if (widths[i] >= 63 || target.getDimSize(i) != (int64_t{1} << widths[i]))
      return failure();
  return Attribute(result);
}

FailureOr<Attribute> projectLayoutCandidate(
    const LayoutConstraintGraph &graph, const LayoutConstraint &relation,
    LayoutVarID source, Attribute candidate, LayoutVarID target) {
  const LayoutVar &dst = graph.getVariable(target);
  if (dst.requiredThreads && !matchesLayoutThreadCount(candidate, dst.requiredThreads) &&
      dst.kind == LayoutKind::Distributed)
    return failure();
  if (relation.kind == ConstraintKind::CopyAccess) {
    auto encoding = dyn_cast<StorageLayoutAttr>(candidate);
    const auto &src = graph.getVariable(source).storageAlias;
    if (!encoding || !src || !dst.storageAlias) return failure();
    auto projected = rebaseStorageCopyCandidate(*src, encoding, *dst.storageAlias);
    if (failed(projected)) return failure();
    return Attribute(*projected);
  }
  if (relation.kind == ConstraintKind::AliasLayout) {
    auto encoding = dyn_cast<StorageLayoutAttr>(candidate);
    const auto &src = graph.getVariable(source).storageAlias;
    if (!encoding || !src || !dst.storageAlias) return failure();
    auto projected = projectStorageAliasCandidate(*src, encoding, *dst.storageAlias);
    if (failed(projected)) return failure();
    return Attribute(*projected);
  }
  if (relation.kind == ConstraintKind::SameLayout ||
      relation.kind == ConstraintKind::Convertible)
    return candidate;
  if (relation.kind == ConstraintKind::TransformLayout)
    return permuteEncoding(candidate, relation.coordinateTransform,
                           cast<ShapedType>(dst.shapedType),
                           source != relation.vars.front());
  if (relation.kind != ConstraintKind::StorageAccess)
    return failure();
  auto storage = dyn_cast<StorageLayoutAttr>(candidate);
  auto type = dyn_cast<MemRefType>(dst.shapedType);
  if (!storage || !type)
    return failure(); // StorageAccess is not a preferred-order hard equality.
  auto space = getFriskMemorySpace(type);
  if (!space || *space == attr::MemorySpace::Local)
    return failure();
  return Attribute(StorageLayoutAttr::get(
      type.getContext(), storage.getMap(),
      MemorySpaceAttr::get(type.getContext(), *space), storage.getAlignment(),
      storage.getVectorGranularity()));
}

bool layoutRelationCompatible(const LayoutConstraintGraph &graph,
                              const LayoutConstraint &relation,
                              LayoutVarID lhsID, Attribute lhs,
                              LayoutVarID rhsID, Attribute rhs) {
  const LayoutVar &a = graph.getVariable(lhsID), &b = graph.getVariable(rhsID);
  if (relation.kind == ConstraintKind::CopyAccess) {
    if (!sameLogicalType(a.shapedType, b.shapedType) || !a.storageAlias || !b.storageAlias)
      return false;
    (void)getStorageAliasFootprint(graph, lhsID, lhs);
    (void)getStorageAliasFootprint(graph, rhsID, rhs);
    const auto &x = getStorageAliasFootprint(graph, lhsID, lhs);
    const auto &y = getStorageAliasFootprint(graph, rhsID, rhs);
    if (x.proof.status != ProofStatus::Proven || y.proof.status != ProofStatus::Proven)
      return false;
    // Distinct identities are NOT a NoAlias proof: nonoverlap is the Copy
    // caller precondition. A proven common root permits a stronger check.
    if (x.root != y.root) return true;
    bool identity = x.entries.size() == y.entries.size() &&
        llvm::equal(x.entries, y.entries, [](const auto &p, const auto &q) {
          return p.view == q.view && p.begin == q.begin && p.end == q.end;
        });
    if (identity) return true;
    unsigned i = 0, j = 0;
    while (i < x.entries.size() && j < y.entries.size()) {
      const auto &p = x.entries[i], &q = y.entries[j];
      if (p.begin < q.end && q.begin < p.end) return false;
      if (p.end <= q.end) ++i; else ++j;
    }
    return true;
  }
  if (relation.kind == ConstraintKind::AliasLayout)
    return proveAliasLayoutRelation(graph, lhsID, lhs, rhsID, rhs).status ==
           ProofStatus::Proven;
  if (relation.kind == ConstraintKind::TransformLayout) {
    auto destination = dyn_cast<DistributedEncodingAttr>(rhs);
    if (!destination)
      return false;
    auto destinationMap = dyn_cast<BitLinearLayoutMapAttr>(destination.getMap());
    if (!destinationMap)
      return false;
    auto projected = permuteEncoding(
        lhs, relation.coordinateTransform, cast<ShapedType>(b.shapedType),
        lhsID != relation.vars.front(), destinationMap.getOutputNames());
    return succeeded(projected) && layoutEncodingsEqual(*projected, rhs);
  }
  if (relation.kind == ConstraintKind::Convertible) {
    auto d0 = dyn_cast<DistributedEncodingAttr>(lhs);
    auto d1 = dyn_cast<DistributedEncodingAttr>(rhs);
    return d0 && d1 && sameLogicalType(a.shapedType, b.shapedType) &&
           d0.getTopology()[4] == 1 && d1.getTopology()[4] == 1;
  }
  if (relation.kind == ConstraintKind::StorageAccess) {
    auto s0 = dyn_cast<StorageLayoutAttr>(lhs);
    auto s1 = dyn_cast<StorageLayoutAttr>(rhs);
    if (s0 && s1)
      return s0.getMap() == s1.getMap(); // Existing whole-tile M2 copy.
    auto d0 = dyn_cast<DistributedEncodingAttr>(lhs);
    auto d1 = dyn_cast<DistributedEncodingAttr>(rhs);
    if ((d0 && s1) || (s0 && d1)) {
      const auto &execution = d0 ? a : b;
      const auto &storage = d0 ? b : a;
      if (execution.operationExecution) {
        if (!sameLogicalType(a.shapedType, b.shapedType) || !storage.storageAlias)
          return false;
        const auto &contract = *execution.operationExecution;
        const auto &footprint = getStorageAliasFootprint(graph, storage.id, d0 ? rhs : lhs);
        return proveExecutionVectorAccess(d0 ? d0 : d1, *storage.storageAlias,
            footprint, contract.vectorBytes, relation.access, contract.writerPolicy).status == ProofStatus::Proven;
      }
      // Verified D covers the logical tile; verified S addresses that exact
      // domain. Therefore S(D(h)) is valid, independent of coalescing. Store
      // lowering elects a deterministic owner for replicated logical elements.
      return sameLogicalType(a.shapedType, b.shapedType);
    }
  }
  return layoutEncodingsEqual(lhs, rhs);
}
} // namespace mlir::frisk
