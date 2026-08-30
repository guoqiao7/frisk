#include "Dialect/Frisk/Utils/LayoutUtils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

namespace mlir::frisk {
namespace {

struct SplitInfo {
  int64_t lowerFactor = 1;
  int64_t extent = 1;
};

static bool operator==(const SplitInfo &lhs, const SplitInfo &rhs) {
  return lhs.lowerFactor == rhs.lowerFactor && lhs.extent == rhs.extent;
}

static bool exprUsesDim(AffineExpr expr, unsigned dim) {
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
    return dimExpr.getPosition() == dim;
  if (isa<AffineSymbolExpr>(expr) || isa<AffineConstantExpr>(expr))
    return false;
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(expr))
    return exprUsesDim(bin.getLHS(), dim) || exprUsesDim(bin.getRHS(), dim);
  return false;
}

static std::optional<int64_t> computeSpan(AffineExpr expr, unsigned dim,
                                          int64_t baseSpan) {
  if (dyn_cast<AffineConstantExpr>(expr))
    return int64_t(1);
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
    return dimExpr.getPosition() == dim ? baseSpan : int64_t(1);
  if (dyn_cast<AffineSymbolExpr>(expr))
    return int64_t(1);

  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;

  auto maybeL = computeSpan(bin.getLHS(), dim, baseSpan);
  auto maybeR = computeSpan(bin.getRHS(), dim, baseSpan);
  if (!maybeL || !maybeR)
    return std::nullopt;

  int64_t lhs = *maybeL;
  int64_t rhs = *maybeR;
  int64_t span = 1;

  switch (expr.getKind()) {
  case AffineExprKind::Add:
    if (llvm::AddOverflow(lhs, rhs, span))
      return std::nullopt;
    return std::max<int64_t>(int64_t(1), span - 1);
  case AffineExprKind::Mul: {
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
      int64_t factor = rhsConst.getValue();
      if (factor == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      if (llvm::MulOverflow(lhs, std::abs(factor), span))
        return std::nullopt;
      return span;
    }
    if (auto lhsConst = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
      int64_t factor = lhsConst.getValue();
      if (factor == std::numeric_limits<int64_t>::min())
        return std::nullopt;
      if (llvm::MulOverflow(rhs, std::abs(factor), span))
        return std::nullopt;
      return span;
    }
    return std::nullopt;
  }
  case AffineExprKind::FloorDiv: {
    auto rhsConst = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!rhsConst || rhsConst.getValue() <= 0)
      return std::nullopt;
    return llvm::divideCeil(lhs, rhsConst.getValue());
  }
  case AffineExprKind::Mod: {
    auto rhsConst = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!rhsConst || rhsConst.getValue() <= 0)
      return std::nullopt;
    return std::min<int64_t>(lhs, rhsConst.getValue());
  }
  default:
    return std::nullopt;
  }
}

static LogicalResult collectSplits(AffineExpr expr, unsigned dim,
                                   int64_t baseExtent, int64_t lowerFactor,
                                   SmallVectorImpl<SplitInfo> &splits) {
  if (!exprUsesDim(expr, dim))
    return success();

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() == dim)
      splits.push_back({lowerFactor, baseExtent});
    return success();
  }

  if (isa<AffineConstantExpr>(expr) || isa<AffineSymbolExpr>(expr))
    return success();

  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return failure();

  switch (expr.getKind()) {
  case AffineExprKind::Add:
    if (failed(collectSplits(bin.getLHS(), dim, baseExtent, lowerFactor, splits)))
      return failure();
    return collectSplits(bin.getRHS(), dim, baseExtent, lowerFactor, splits);
  case AffineExprKind::Mul:
    if (isa<AffineConstantExpr>(bin.getLHS()))
      return collectSplits(bin.getRHS(), dim, baseExtent, lowerFactor, splits);
    if (isa<AffineConstantExpr>(bin.getRHS()))
      return collectSplits(bin.getLHS(), dim, baseExtent, lowerFactor, splits);
    return failure();
  case AffineExprKind::FloorDiv: {
    auto divisor = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!divisor || divisor.getValue() <= 0)
      return failure();
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return failure();
    int64_t newLower = 0;
    if (llvm::MulOverflow(lowerFactor, divisor.getValue(), newLower))
      return failure();
    return collectSplits(bin.getLHS(), dim,
                         llvm::divideCeil(*span, divisor.getValue()), newLower,
                         splits);
  }
  case AffineExprKind::Mod: {
    auto modulus = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!modulus || modulus.getValue() <= 0)
      return failure();
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return failure();
    return collectSplits(bin.getLHS(), dim,
                         std::min<int64_t>(*span, modulus.getValue()),
                         lowerFactor, splits);
  }
  default:
    return failure();
  }
}

static SmallVector<SplitInfo, 4>
sortAndUniqueSplits(ArrayRef<SplitInfo> splits) {
  SmallVector<SplitInfo, 4> unique(splits.begin(), splits.end());
  std::sort(unique.begin(), unique.end(),
            [](const SplitInfo &lhs, const SplitInfo &rhs) {
              if (lhs.lowerFactor != rhs.lowerFactor)
                return lhs.lowerFactor < rhs.lowerFactor;
              return lhs.extent < rhs.extent;
            });
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  return unique;
}

static std::optional<SmallVector<SplitInfo, 4>>
collectUniqueSplitsForDim(AffineMap map, unsigned dim, int64_t extent) {
  SmallVector<SplitInfo, 4> splits;
  for (AffineExpr result : map.getResults()) {
    if (!exprUsesDim(result, dim))
      continue;
    if (failed(collectSplits(result, dim, extent, /*lowerFactor=*/1, splits)))
      return std::nullopt;
  }
  return sortAndUniqueSplits(splits);
}

static std::optional<int64_t>
computeSplitProduct(ArrayRef<SplitInfo> splits) {
  int64_t product = 1;
  for (const SplitInfo &split : splits) {
    if (split.extent <= 0)
      return std::nullopt;
    if (llvm::MulOverflow(product, split.extent, product))
      return std::nullopt;
  }
  return product;
}

static std::optional<SmallVector<SplitInfo, 4>>
computeUnusedSplits(int64_t fullExtent, ArrayRef<SplitInfo> usedSplits) {
  if (fullExtent <= 0)
    return std::nullopt;

  SmallVector<SplitInfo, 4> unused;
  int64_t expectedLower = 1;
  for (const SplitInfo &split : usedSplits) {
    if (split.lowerFactor <= 0 || split.extent <= 0)
      return std::nullopt;
    if (split.lowerFactor < expectedLower)
      return std::nullopt;
    if (split.lowerFactor > expectedLower) {
      if (split.lowerFactor % expectedLower != 0)
        return std::nullopt;
      unused.push_back(
          {expectedLower, split.lowerFactor / expectedLower});
      expectedLower = split.lowerFactor;
    }
    int64_t nextLower = 0;
    if (llvm::MulOverflow(split.lowerFactor, split.extent, nextLower))
      return std::nullopt;
    expectedLower = nextLower;
  }

  if (expectedLower > fullExtent)
    return std::nullopt;
  if (expectedLower < fullExtent) {
    if (fullExtent % expectedLower != 0)
      return std::nullopt;
    unused.push_back({expectedLower, fullExtent / expectedLower});
  }
  return unused;
}

static std::optional<int64_t>
lookupNewLower(ArrayRef<std::pair<SplitInfo, int64_t>> remap,
               SplitInfo split) {
  for (const auto &entry : remap) {
    if (entry.first == split)
      return entry.second;
  }
  return std::nullopt;
}

static std::optional<AffineExpr>
buildCompressedSplitExpr(OpBuilder &builder,
                         ArrayRef<std::pair<SplitInfo, int64_t>> remap,
                         AffineExpr replicateExpr, SplitInfo split,
                         int64_t compressedExtent) {
  if (split.extent <= 1)
    return builder.getAffineConstantExpr(0);

  auto maybeLower = lookupNewLower(remap, split);
  if (!maybeLower)
    return std::nullopt;

  int64_t newLower = *maybeLower;
  if (newLower == 1 && split.extent == compressedExtent)
    return replicateExpr;

  AffineExpr result = replicateExpr;
  if (newLower != 1)
    result = result.floorDiv(builder.getAffineConstantExpr(newLower));
  result = result % builder.getAffineConstantExpr(split.extent);
  return result;
}

static std::optional<AffineExpr>
rewriteCompressedExpr(OpBuilder &builder, AffineExpr expr, unsigned dim,
                      int64_t baseExtent, int64_t lowerFactor,
                      ArrayRef<std::pair<SplitInfo, int64_t>> remap,
                      AffineExpr replicateExpr, int64_t compressedExtent) {
  if (!exprUsesDim(expr, dim))
    return expr;

  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    if (dimExpr.getPosition() != dim)
      return expr;
    return buildCompressedSplitExpr(builder, remap, replicateExpr,
                                    {lowerFactor, baseExtent},
                                    compressedExtent);
  }

  if (isa<AffineConstantExpr>(expr) || isa<AffineSymbolExpr>(expr))
    return expr;

  auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!bin)
    return std::nullopt;

  switch (expr.getKind()) {
  case AffineExprKind::Add: {
    auto lhs = rewriteCompressedExpr(builder, bin.getLHS(), dim, baseExtent,
                                     lowerFactor, remap, replicateExpr,
                                     compressedExtent);
    auto rhs = rewriteCompressedExpr(builder, bin.getRHS(), dim, baseExtent,
                                     lowerFactor, remap, replicateExpr,
                                     compressedExtent);
    if (!lhs || !rhs)
      return std::nullopt;
    return *lhs + *rhs;
  }
  case AffineExprKind::Mul: {
    if (auto lhsConst = dyn_cast<AffineConstantExpr>(bin.getLHS())) {
      auto rhs = rewriteCompressedExpr(builder, bin.getRHS(), dim, baseExtent,
                                       lowerFactor, remap, replicateExpr,
                                       compressedExtent);
      if (!rhs)
        return std::nullopt;
      return builder.getAffineConstantExpr(lhsConst.getValue()) * *rhs;
    }
    if (auto rhsConst = dyn_cast<AffineConstantExpr>(bin.getRHS())) {
      auto lhs = rewriteCompressedExpr(builder, bin.getLHS(), dim, baseExtent,
                                       lowerFactor, remap, replicateExpr,
                                       compressedExtent);
      if (!lhs)
        return std::nullopt;
      return *lhs * builder.getAffineConstantExpr(rhsConst.getValue());
    }
    return std::nullopt;
  }
  case AffineExprKind::FloorDiv: {
    auto divisor = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!divisor || divisor.getValue() <= 0)
      return std::nullopt;
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return std::nullopt;
    int64_t newLower = 0;
    if (llvm::MulOverflow(lowerFactor, divisor.getValue(), newLower))
      return std::nullopt;
    return rewriteCompressedExpr(builder, bin.getLHS(), dim,
                                 llvm::divideCeil(*span, divisor.getValue()),
                                 newLower, remap, replicateExpr,
                                 compressedExtent);
  }
  case AffineExprKind::Mod: {
    auto modulus = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!modulus || modulus.getValue() <= 0)
      return std::nullopt;
    auto span = computeSpan(bin.getLHS(), dim, baseExtent);
    if (!span)
      return std::nullopt;
    return rewriteCompressedExpr(builder, bin.getLHS(), dim,
                                 std::min<int64_t>(*span, modulus.getValue()),
                                 lowerFactor, remap, replicateExpr,
                                 compressedExtent);
  }
  default:
    return std::nullopt;
  }
}

} // namespace

std::optional<int64_t>
computeUsedExtentForDim(AffineMapAttr mapAttr, unsigned placeholderPos,
                        int64_t placeholderExtent) {
  if (!mapAttr)
    return int64_t(1);
  if (placeholderExtent <= 0)
    return std::nullopt;

  AffineMap map = mapAttr.getValue();
  if (placeholderPos >= map.getNumDims())
    return std::nullopt;

  auto splits = collectUniqueSplitsForDim(map, placeholderPos, placeholderExtent);
  if (!splits)
    return std::nullopt;
  return computeSplitProduct(*splits);
}

std::optional<CompressedReplicateInfo>
compressReplicateDimInMap(OpBuilder &builder, AffineMapAttr mapAttr,
                          unsigned placeholderPos, int64_t placeholderExtent) {
  if (!mapAttr)
    return CompressedReplicateInfo{AffineMapAttr(), 1};
  if (placeholderExtent <= 0)
    return std::nullopt;

  AffineMap map = mapAttr.getValue();
  if (placeholderPos >= map.getNumDims() || placeholderPos + 1 != map.getNumDims())
    return std::nullopt;

  auto splits = collectUniqueSplitsForDim(map, placeholderPos, placeholderExtent);
  if (!splits)
    return std::nullopt;

  auto maybeRepExtent = computeSplitProduct(*splits);
  if (!maybeRepExtent)
    return std::nullopt;
  int64_t compressedExtent = *maybeRepExtent;

  SmallVector<std::pair<SplitInfo, int64_t>, 4> remap;
  int64_t nextLower = 1;
  for (const SplitInfo &split : *splits) {
    remap.push_back({split, nextLower});
    if (llvm::MulOverflow(nextLower, split.extent, nextLower))
      return std::nullopt;
  }

  unsigned originalSymCount = map.getNumSymbols();
  AffineExpr compressedReplicate = builder.getAffineSymbolExpr(originalSymCount);
  SmallVector<AffineExpr> rewrittenResults;
  rewrittenResults.reserve(map.getNumResults());

  bool needsReplicateSymbol = false;
  for (AffineExpr result : map.getResults()) {
    AffineExpr rewritten = result;
    if (exprUsesDim(result, placeholderPos)) {
      auto maybeRewritten =
          rewriteCompressedExpr(builder, result, placeholderPos,
                                placeholderExtent, /*lowerFactor=*/1, remap,
                                compressedReplicate, compressedExtent);
      if (!maybeRewritten)
        return std::nullopt;
      rewritten = *maybeRewritten;
      needsReplicateSymbol |=
          (compressedExtent > 1 && rewritten.isFunctionOfSymbol(originalSymCount));
    }
    rewrittenResults.push_back(rewritten);
  }

  unsigned newDimCount = map.getNumDims() - 1;
  unsigned newSymCount = originalSymCount + (needsReplicateSymbol ? 1 : 0);

  AffineMap newMap =
      AffineMap::get(newDimCount, newSymCount, rewrittenResults,
                     builder.getContext());
  return CompressedReplicateInfo{AffineMapAttr::get(newMap), compressedExtent};
}

std::optional<AffineMapAttr>
inferFragmentIndexFromThreadMap(OpBuilder &builder, AffineMapAttr threadMapAttr,
                                ArrayRef<int64_t> inputShape) {
  if (!threadMapAttr)
    return std::nullopt;

  AffineMap threadMap = threadMapAttr.getValue();
  if (threadMap.getNumDims() != inputShape.size())
    return std::nullopt;
  for (int64_t extent : inputShape)
    if (extent <= 0)
      return std::nullopt;

  AffineExpr indexExpr = builder.getAffineConstantExpr(0);
  int64_t scale = 1;
  for (int64_t dim = static_cast<int64_t>(inputShape.size()) - 1; dim >= 0;
       --dim) {
    auto usedSplits =
        collectUniqueSplitsForDim(threadMap, static_cast<unsigned>(dim),
                                  inputShape[dim]);
    if (!usedSplits)
      return std::nullopt;
    auto unusedSplits = computeUnusedSplits(inputShape[dim], *usedSplits);
    if (!unusedSplits)
      return std::nullopt;

    for (const SplitInfo &split : *unusedSplits) {
      AffineExpr term = builder.getAffineDimExpr(dim);
      if (split.lowerFactor != 1)
        term = term.floorDiv(builder.getAffineConstantExpr(split.lowerFactor));
      if (split.extent != 1)
        term = term % builder.getAffineConstantExpr(split.extent);
      if (scale != 1)
        term = term * builder.getAffineConstantExpr(scale);
      indexExpr = indexExpr + term;
      if (llvm::MulOverflow(scale, split.extent, scale))
        return std::nullopt;
    }
  }

  indexExpr = simplifyAffineExpr(indexExpr, threadMap.getNumDims(), 0);
  return AffineMapAttr::get(
      AffineMap::get(threadMap.getNumDims(), /*numSymbols=*/0, indexExpr,
                     builder.getContext()));
}

} // namespace mlir::frisk
