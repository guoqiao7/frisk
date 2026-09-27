#include "LegacyLayoutOracle.h"
#include "LegacyLayoutOracleUtils.h"
#include <algorithm>
#include <atomic>
#include <array>
#include <cassert>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Visitors.h"

namespace mlir::frisk::test {
static std::atomic<uint64_t> legacyParallelCalls{0};
uint64_t getLegacyParallelOracleCallCount() { return legacyParallelCalls.load(); }

namespace {

struct TargetInfo {
  bool isCuda = false;
  bool isCDNA = false;
  unsigned smVersion = 0;
  unsigned warpSize = 32;
  StringRef rawName;
};

enum class GemmInst { MMA, WGMMA };

static bool isHopper(const TargetInfo &info) {
  return info.isCuda && info.smVersion >= 90;
}

static bool isAmpere(const TargetInfo &info) {
  return info.isCuda && info.smVersion >= 80 && info.smVersion < 90;
}

static std::optional<TargetInfo> detectTargetInfo(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto attr = cur->getAttrOfType<StringAttr>("frisk.target")) {
      TargetInfo info;
      info.rawName = attr.getValue();
      StringRef value = info.rawName;
      if (value.consume_front("sm_") || value.consume_front("sm")) {
        info.isCuda = true;
        unsigned parsed = 0;
        if (!value.getAsInteger(/*Radix=*/10, parsed)) {
          info.smVersion = parsed;
          info.warpSize = 32;
        }
        return info;
      }
      if (value.consume_front("gfx")) {
        info.isCDNA = value.consume_front("9");
        info.warpSize = info.isCDNA ? 64 : 32;
        return info;
      }
      // Fallthrough—treat unknown string as CUDA-like for now.
      info.isCuda = true;
      info.warpSize = 32;
      return info;
    }
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

static std::optional<int64_t> inferThreadBlockSize(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto attr = cur->getAttrOfType<IntegerAttr>("frisk.threads"))
      return attr.getInt();
    if (auto parallel = dyn_cast<ParallelOp>(cur))
      return parallel.getThreads();
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

static AffineExpr getDimExpr(unsigned idx, MLIRContext *ctx) {
  return mlir::getAffineDimExpr(idx, ctx);
}

static AffineExpr getConstExpr(int64_t value, MLIRContext *ctx) {
  return mlir::getAffineConstantExpr(value, ctx);
}

static AffineExpr floorDivConst(AffineExpr expr, int64_t divisor,
                                MLIRContext *ctx) {
  return expr.floorDiv(divisor);
}

static AffineExpr modConst(AffineExpr expr, int64_t divisor, MLIRContext *ctx) {
  return expr % getConstExpr(divisor, ctx);
}

static AffineExpr xor2x2(AffineExpr a, AffineExpr b, MLIRContext *ctx) {
  AffineExpr sum = a + b;
  AffineExpr two = getConstExpr(2, ctx);
  return sum - two * floorDivConst(sum, 2, ctx);
}

static AffineExpr xor4x4(AffineExpr a, AffineExpr b, MLIRContext *ctx) {
  AffineExpr i0 = modConst(a, 2, ctx);
  AffineExpr j0 = modConst(b, 2, ctx);
  AffineExpr i1 = floorDivConst(a, 2, ctx);
  AffineExpr j1 = floorDivConst(b, 2, ctx);
  return getConstExpr(2, ctx) * xor2x2(i1, j1, ctx) + xor2x2(i0, j0, ctx);
}

static AffineExpr xor8x8(AffineExpr a, AffineExpr b, MLIRContext *ctx) {
  AffineExpr i0 = modConst(a, 2, ctx);
  AffineExpr j0 = modConst(b, 2, ctx);
  AffineExpr i1 = floorDivConst(a, 2, ctx);
  AffineExpr j1 = floorDivConst(b, 2, ctx);
  return getConstExpr(2, ctx) * xor4x4(i1, j1, ctx) + xor2x2(i0, j0, ctx);
}

static LayoutAttr makeLayoutAttr(OpBuilder &builder,
                                 ArrayRef<int64_t> shape,
                                 ArrayRef<AffineExpr> results) {
  MLIRContext *ctx = builder.getContext();
  auto shapeAttr = builder.getDenseI64ArrayAttr(shape);
  auto map = AffineMap::get(shape.size(), 0, results, ctx);
  return LayoutAttr::get(ctx, shapeAttr, AffineMapAttr::get(map),
                         AffineMapAttr(), IntegerAttr());
}

struct FragmentExpr {
  MLIRContext *ctx;
  SmallVector<int64_t, 2> shape;
  AffineExpr indexExpr;
  int64_t indexExtent = 1;
  AffineExpr threadExpr;
  int64_t threadExtent = 1;
  int64_t replicateSize = 1;

  FragmentExpr repeat(ArrayRef<int64_t> repeats, bool repeatOnThread,
                      bool lowerDimFirst) const {
    assert(repeats.size() == shape.size());
    FragmentExpr next = *this;
    SmallVector<int64_t, 2> oldShape = shape;
    for (size_t i = 0; i < repeats.size(); ++i)
      next.shape[i] *= repeats[i];

    auto substituteDims = [&](AffineExpr expr) {
      for (size_t i = 0; i < oldShape.size(); ++i) {
        if (oldShape[i] <= 0)
          continue;
        AffineExpr dim = getDimExpr(i, ctx);
        expr = expr.replace(dim, modConst(dim, oldShape[i], ctx));
      }
      return expr;
    };

    AffineExpr localIndex = substituteDims(indexExpr);
    AffineExpr localThread = substituteDims(threadExpr);

    AffineExpr repeatsIndex = getConstExpr(0, ctx);
    AffineExpr repeatStride = getConstExpr(1, ctx);

    auto addContribution = [&](int64_t dimIdx) {
      AffineExpr dim = getDimExpr(dimIdx, ctx);
      if (oldShape[dimIdx] <= 0)
        return;
      AffineExpr quotient = floorDivConst(dim, oldShape[dimIdx], ctx);
      repeatsIndex = repeatsIndex + quotient * repeatStride;
      repeatStride =
          repeatStride * getConstExpr(repeats[dimIdx], ctx);
    };

    if (lowerDimFirst) {
      for (int64_t i = static_cast<int64_t>(oldShape.size()) - 1; i >= 0; --i)
        addContribution(i);
    } else {
      for (size_t i = 0; i < oldShape.size(); ++i)
        addContribution(i);
    }

    int64_t repeatProduct = 1;
    for (int64_t value : repeats)
      repeatProduct *= value;

    if (repeatOnThread) {
      next.threadExpr = localThread +
                        getConstExpr(threadExtent, ctx) * repeatsIndex;
      next.threadExtent *= repeatProduct;
      next.indexExpr = localIndex;
    } else {
      next.threadExpr = localThread;
      next.indexExpr =
          localIndex + getConstExpr(indexExtent, ctx) * repeatsIndex;
      next.indexExtent *= repeatProduct;
    }
    next.replicateSize = replicateSize;
    return next;
  }

  FragmentExpr replicate(int64_t repeats) const {
    assert(repeats >= 1 && "replicate factor must be positive");
    FragmentExpr next = *this;
    next.replicateSize *= repeats;
    return next;
  }
};

static FragmentExpr makeFragment8x4(MLIRContext *ctx) {
  FragmentExpr frag;
  frag.ctx = ctx;
  frag.shape = {8, 4};
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  frag.indexExpr = modConst(j, 1, ctx);
  frag.indexExtent = 1;
  frag.threadExpr = floorDivConst(j, 1, ctx) + getConstExpr(4, ctx) * i;
  frag.threadExtent = 32;
  frag.replicateSize = 1;
  return frag;
}

static FragmentExpr makeFragment8x8(MLIRContext *ctx) {
  FragmentExpr frag;
  frag.ctx = ctx;
  frag.shape = {8, 8};
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  frag.indexExpr = modConst(j, 2, ctx);
  frag.indexExtent = 2;
  frag.threadExpr = floorDivConst(j, 2, ctx) + getConstExpr(4, ctx) * i;
  frag.threadExtent = 32;
  frag.replicateSize = 1;
  return frag;
}

static FragmentExpr makeFragment8x16(MLIRContext *ctx) {
  FragmentExpr frag;
  frag.ctx = ctx;
  frag.shape = {8, 16};
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  frag.indexExpr = modConst(j, 4, ctx);
  frag.indexExtent = 4;
  frag.threadExpr = floorDivConst(j, 4, ctx) + getConstExpr(4, ctx) * i;
  frag.threadExtent = 32;
  frag.replicateSize = 1;
  return frag;
}

static FragmentExpr makeFragment8x8Transposed(MLIRContext *ctx) {
  FragmentExpr frag;
  frag.ctx = ctx;
  frag.shape = {8, 8};
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  frag.indexExpr = modConst(i, 2, ctx);
  frag.indexExtent = 2;
  frag.threadExpr = floorDivConst(i, 2, ctx) + getConstExpr(4, ctx) * j;
  frag.threadExtent = 32;
  frag.replicateSize = 1;
  return frag;
}

static LayoutAttr makeFragmentLayout(OpBuilder &builder,
                                     const FragmentExpr &frag) {
  MLIRContext *ctx = builder.getContext();
  auto shapeAttr = builder.getDenseI64ArrayAttr(frag.shape);
  SmallVector<AffineExpr, 1> indexResults = {frag.indexExpr};
  SmallVector<AffineExpr, 1> threadResults = {frag.threadExpr};
  auto indexMap =
      AffineMap::get(frag.shape.size(), 0, indexResults, ctx);
  auto threadMap =
      AffineMap::get(frag.shape.size(), 0, threadResults, ctx);
  auto replicateAttr = builder.getI64IntegerAttr(frag.replicateSize);
  return LayoutAttr::get(ctx, shapeAttr, AffineMapAttr::get(indexMap),
                         AffineMapAttr::get(threadMap), replicateAttr);
}

static std::optional<FragmentExpr>
buildAmpereFragmentC(MLIRContext *ctx, int64_t blockM, int64_t blockN,
                     int64_t warpTileM, int64_t warpTileN) {
  if (warpTileM % 16 != 0 || warpTileN % 8 != 0)
    return std::nullopt;
  if (blockM % warpTileM != 0 || blockN % warpTileN != 0)
    return std::nullopt;

  FragmentExpr base = makeFragment8x8(ctx).repeat({2, 1}, /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/true);
  int64_t warpRepeatM = std::max<int64_t>(1, blockM / warpTileM);
  int64_t warpRepeatN = std::max<int64_t>(1, blockN / warpTileN);
  FragmentExpr warpLayout =
      base.repeat({warpRepeatM, warpRepeatN}, /*repeatOnThread=*/true,
                  /*lowerDimFirst=*/false);
  int64_t innerRepeatM = std::max<int64_t>(1, warpTileM / 16);
  int64_t innerRepeatN = std::max<int64_t>(1, warpTileN / 8);
  FragmentExpr blockLayout =
      warpLayout.repeat({innerRepeatM, innerRepeatN},
                        /*repeatOnThread=*/false, /*lowerDimFirst=*/false);
  return blockLayout;
}

static std::optional<FragmentExpr>
buildHopperFragmentC(MLIRContext *ctx, int64_t blockM, int64_t blockN,
                     int64_t warpTileM, int64_t warpTileN) {
  if (warpTileM % 16 != 0 || warpTileN % 8 != 0)
    return std::nullopt;
  if (blockM % warpTileM != 0 || blockN % warpTileN != 0)
    return std::nullopt;

  int64_t warpRepeatN = std::max<int64_t>(1, warpTileN / 8);
  FragmentExpr warpLayout =
      makeFragment8x8(ctx).repeat({2, warpRepeatN}, /*repeatOnThread=*/false,
                                  /*lowerDimFirst=*/false);
  FragmentExpr blockLayout =
      warpLayout.repeat({std::max<int64_t>(1, blockM / warpTileM),
                         std::max<int64_t>(1, blockN / warpTileN)},
                        /*repeatOnThread=*/true, /*lowerDimFirst=*/false);
  FragmentExpr finalLayout =
      blockLayout.repeat({std::max<int64_t>(1, warpTileM / 16), 1},
                         /*repeatOnThread=*/false, /*lowerDimFirst=*/false);
  return finalLayout;
}

static std::optional<FragmentExpr>
buildGemmFragmentA(MLIRContext *ctx, int64_t blockM, int64_t blockN,
                   int64_t blockK, int64_t warpTileM, int64_t warpTileN,
                   int64_t elementBits, bool transposed) {
  if (blockM <= 0 || blockN <= 0 || blockK <= 0 || warpTileM <= 0 ||
      warpTileN <= 0)
    return std::nullopt;
  if (blockM % warpTileM != 0 || blockN % warpTileN != 0)
    return std::nullopt;
  if (warpTileM % 16 != 0 || blockK % 16 != 0)
    return std::nullopt;
  if (elementBits != 8 && elementBits != 16 && elementBits != 32)
    return std::nullopt;

  int64_t warpRepeatM = blockM / warpTileM;
  int64_t warpRepeatN = blockN / warpTileN;

  if (transposed) {
    FragmentExpr base = makeFragment8x8Transposed(ctx).repeat({2, 2}, /*repeatOnThread=*/false,
                                                              /*lowerDimFirst=*/true);
    FragmentExpr warpLayout = base.repeat({1, warpRepeatM}, /*repeatOnThread=*/true,
                                          /*lowerDimFirst=*/false)
                                  .replicate(warpRepeatN);
    FragmentExpr blockLayout = warpLayout.repeat({blockK / 16, warpTileM / 16},
                                                  /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/true);
    return blockLayout;
  }

  if (elementBits == 8) {
    if (blockK % 32 != 0)
      return std::nullopt;
    FragmentExpr base = makeFragment8x16(ctx).repeat({2, 2}, /*repeatOnThread=*/false,
                                                     /*lowerDimFirst=*/false);
    FragmentExpr warpLayout = base.repeat({warpRepeatM, 1}, /*repeatOnThread=*/true,
                                          /*lowerDimFirst=*/true)
                                  .replicate(warpRepeatN);
    FragmentExpr blockLayout = warpLayout.repeat({warpTileM / 16, blockK / 32},
                                                  /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/false);
    return blockLayout;
  }
  if (elementBits == 16) {
    FragmentExpr base = makeFragment8x8(ctx).repeat({2, 2}, /*repeatOnThread=*/false,
                                    /*lowerDimFirst=*/false);
    FragmentExpr warpLayout = base.repeat({warpRepeatM, 1}, /*repeatOnThread=*/true,
                                          /*lowerDimFirst=*/true)
                                  .replicate(warpRepeatN);
    FragmentExpr blockLayout = warpLayout.repeat({warpTileM / 16, blockK / 16},
                                                  /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/false);
    return blockLayout;
  }
  if (blockK % 8 != 0)
    return std::nullopt;
  FragmentExpr base = makeFragment8x4(ctx).repeat({2, 2}, /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/false);
  FragmentExpr warpLayout = base.repeat({warpRepeatM, 1}, /*repeatOnThread=*/true,
                                        /*lowerDimFirst=*/true)
                                .replicate(warpRepeatN);
  FragmentExpr blockLayout = warpLayout.repeat({warpTileM / 16, blockK / 8},
                                              /*repeatOnThread=*/false,
                                              /*lowerDimFirst=*/false);
  return blockLayout;
}

static std::optional<FragmentExpr>
buildGemmFragmentB(MLIRContext *ctx, int64_t blockM, int64_t blockN,
                   int64_t blockK, int64_t warpTileM, int64_t warpTileN,
                   bool transposed) {
  if (blockM <= 0 || blockN <= 0 || blockK <= 0 || warpTileM <= 0 ||
      warpTileN <= 0)
    return std::nullopt;
  if (blockM % warpTileM != 0 || blockN % warpTileN != 0)
    return std::nullopt;
  if (warpTileN % 8 != 0 || blockK % 16 != 0)
    return std::nullopt;

  int64_t warpRepeatM = blockM / warpTileM;
  int64_t warpRepeatN = blockN / warpTileN;

  if (transposed) {
    FragmentExpr base = makeFragment8x8(ctx).repeat({1, 2}, /*repeatOnThread=*/false,
                                                    /*lowerDimFirst=*/false);
    FragmentExpr warpLayout = base.replicate(warpRepeatM)
                                  .repeat({warpRepeatN, 1}, /*repeatOnThread=*/true,
                                          /*lowerDimFirst=*/false);
    FragmentExpr blockLayout = warpLayout.repeat({warpTileN / 8, blockK / 16},
                                                  /*repeatOnThread=*/false,
                                                  /*lowerDimFirst=*/false);
    return blockLayout;
  }

  FragmentExpr base = makeFragment8x8Transposed(ctx).repeat({2, 1}, /*repeatOnThread=*/false,
                                                            /*lowerDimFirst=*/false);
  FragmentExpr warpLayout = base.replicate(warpRepeatM)
                                .repeat({1, warpRepeatN}, /*repeatOnThread=*/true,
                                        /*lowerDimFirst=*/true);
  FragmentExpr blockLayout = warpLayout.repeat({blockK / 16, warpTileN / 8},
                                                /*repeatOnThread=*/false,
                                                /*lowerDimFirst=*/true);
  return blockLayout;
}

static LayoutAttr makeLinearLayout(OpBuilder &builder, int64_t rows,
                                   int64_t cols, int64_t linearStride = -1) {
  MLIRContext *ctx = builder.getContext();
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  int64_t stride = (linearStride < 0) ? cols : linearStride;
  AffineExpr expr = i * getConstExpr(stride, ctx) + j;
  SmallVector<AffineExpr, 1> results = {expr};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makePaddedLayout(OpBuilder &builder, int64_t rows,
                                   int64_t cols, int64_t elementBits) {
  int64_t padded = cols;
  if (elementBits > 0 && (elementBits * cols) % 256 == 0)
    padded += 128 / elementBits;
  return makeLinearLayout(builder, rows, cols, padded);
}

static LayoutAttr makeLayoutF64_Kinner(OpBuilder &builder, int64_t rows, int64_t cols) {
  MLIRContext *ctx = builder.getContext();
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr tc = floorDivConst(j, 16, ctx);
  AffineExpr ts = floorDivConst(i, 4, ctx);
  AffineExpr c = modConst(j, 16, ctx);
  AffineExpr s = modConst(i, 4, ctx);
  AffineExpr cSwizzle = floorDivConst(c, 4, ctx) * 4 + xor4x4(modConst(c, 4, ctx), s, ctx);
  AffineExpr index = cSwizzle + s * getConstExpr(16, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makeLayoutF64_Kouter(OpBuilder &builder, int64_t rows, int64_t cols) {
  MLIRContext *ctx = builder.getContext();
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr tc = floorDivConst(j, 16, ctx);
  AffineExpr ts = floorDivConst(i, 4, ctx);
  AffineExpr c = modConst(j, 16, ctx);
  AffineExpr s = modConst(i, 4, ctx);
  AffineExpr cSwizzle = floorDivConst(c, 4, ctx) + xor4x4(modConst(c, 4, ctx), s, ctx) * 4;
  AffineExpr index = cSwizzle + s * getConstExpr(16, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makeQuarterBankSwizzle(OpBuilder &builder, int64_t rows,
                                         int64_t cols, int64_t elementBits) {
  MLIRContext *ctx = builder.getContext();
  int64_t vectorSize = 128 / elementBits;
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr ts = floorDivConst(i, 8, ctx);
  AffineExpr s = modConst(i, 8, ctx);
  AffineExpr block = floorDivConst(j, vectorSize, ctx);
  AffineExpr tc = floorDivConst(block, 2, ctx);
  AffineExpr c = modConst(block, 2, ctx);
  AffineExpr vec = modConst(j, vectorSize, ctx);
  AffineExpr cSwizzle = xor2x2(c, floorDivConst(s, 4, ctx), ctx);
  AffineExpr index = vec +
                     (cSwizzle + s * getConstExpr(2, ctx)) *
                         getConstExpr(vectorSize, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makeHalfBankSwizzle(OpBuilder &builder, int64_t rows,
                                      int64_t cols, int64_t elementBits) {
  MLIRContext *ctx = builder.getContext();
  int64_t vectorSize = 128 / elementBits;
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr ts = floorDivConst(i, 8, ctx);
  AffineExpr s = modConst(i, 8, ctx);
  AffineExpr block = floorDivConst(j, vectorSize, ctx);
  AffineExpr tc = floorDivConst(block, 4, ctx);
  AffineExpr c = modConst(block, 4, ctx);
  AffineExpr vec = modConst(j, vectorSize, ctx);
  AffineExpr cSwizzle = xor4x4(c, floorDivConst(s, 2, ctx), ctx);
  AffineExpr index = vec +
                     (cSwizzle + s * getConstExpr(4, ctx)) *
                         getConstExpr(vectorSize, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr makeFullBankSwizzle(OpBuilder &builder, int64_t rows,
                                      int64_t cols, int64_t elementBits) {
  MLIRContext *ctx = builder.getContext();
  int64_t vectorSize = 128 / elementBits;
  AffineExpr i = getDimExpr(0, ctx);
  AffineExpr j = getDimExpr(1, ctx);
  AffineExpr ts = floorDivConst(i, 8, ctx);
  AffineExpr s = modConst(i, 8, ctx);
  AffineExpr block = floorDivConst(j, vectorSize, ctx);
  AffineExpr tc = floorDivConst(block, 8, ctx);
  AffineExpr c = modConst(block, 8, ctx);
  AffineExpr vec = modConst(j, vectorSize, ctx);
  AffineExpr cSwizzle = xor8x8(c, s, ctx);
  AffineExpr index = vec +
                     (cSwizzle + s * getConstExpr(8, ctx)) *
                         getConstExpr(vectorSize, ctx);
  SmallVector<AffineExpr, 3> results = {tc, ts, index};
  return makeLayoutAttr(builder, {rows, cols}, results);
}

static LayoutAttr buildAmpereSharedLayout(OpBuilder &builder, MemRefType type,
                                          bool kInner) {
  if (type.getRank() < 2)
    return LayoutAttr();
  ArrayRef<int64_t> shape = type.getShape();
  int64_t stride = shape[shape.size() - 2];
  int64_t cont = shape.back();
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0)
    return LayoutAttr();

  if (elementBits == 64){
    if (!kInner && cont % 16 == 0) //float64 KxN
      return makeLayoutF64_Kouter(builder, stride, cont);
    if (kInner && cont % 16 == 0) //float64 NxK
      return makeLayoutF64_Kinner(builder, stride, cont);
    return makePaddedLayout(builder, stride, cont, elementBits);
  }

  int64_t vectorSize = 128 / elementBits;
  if (!kInner && elementBits == 8)
    return makePaddedLayout(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 8) == 0)
    return makeFullBankSwizzle(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 4) == 0)
    return makeHalfBankSwizzle(builder, stride, cont, elementBits);
  return makePaddedLayout(builder, stride, cont, elementBits);
}

static LayoutAttr buildHopperSharedLayout(OpBuilder &builder, MemRefType type,
                                          bool kInner) {
  if (type.getRank() < 2)
    return LayoutAttr();
  ArrayRef<int64_t> shape = type.getShape();
  int64_t stride = shape[shape.size() - 2];
  int64_t cont = shape.back();
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0)
    return LayoutAttr();

  if (elementBits == 64){
    if (!kInner && cont % 16 == 0) //float64 KxN
      return makeLayoutF64_Kouter(builder, stride, cont);
    if (kInner && cont % 16 == 0) //float64 NxK
      return makeLayoutF64_Kinner(builder, stride, cont);
    return makeQuarterBankSwizzle(builder, stride, cont, elementBits);
  }

  int64_t vectorSize = 128 / elementBits;
  if (cont % (vectorSize * 8) == 0)
    return makeFullBankSwizzle(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 4) == 0)
    return makeHalfBankSwizzle(builder, stride, cont, elementBits);
  if (cont % (vectorSize * 2) == 0)
    return makeQuarterBankSwizzle(builder, stride, cont, elementBits);
  if (cont % vectorSize == 0)
    return makeLinearLayout(builder, stride, cont);
  return LayoutAttr();
}

static GemmInst inferGemmInst(const TargetInfo &info, int64_t blockSize,
                              int64_t M, int64_t N) {
  if (isHopper(info) && blockSize % 128 == 0 && M >= 64 && N >= 64)
    return GemmInst::WGMMA;
  return GemmInst::MMA;
}

static FailureOr<std::array<int64_t, 2>> extractMatrixShape(MemRefType type) {
  if (type.getRank() < 2)
    return failure();
  int64_t rows = type.getDimSize(type.getRank() - 2);
  int64_t cols = type.getDimSize(type.getRank() - 1);
  if (rows < 0 || cols < 0)
    return failure();
  return std::array<int64_t, 2>{rows, cols};
}

static DenseI64ArrayAttr buildShapeAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> dims) {
  return builder.getDenseI64ArrayAttr(dims);
}

static LayoutAttr buildLinearLayoutAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> dims, int64_t stride,
                                        bool transpose) {
  assert(dims.size() == 2 && "expect 2-D layout");
  int64_t rows = dims[0];
  int64_t cols = dims[1];
  if (transpose)
    std::swap(rows, cols);
  DenseI64ArrayAttr shapeAttr = buildShapeAttr(builder, {rows, cols});

  MLIRContext *ctx = builder.getContext();
  AffineExpr row = builder.getAffineDimExpr(0);
  AffineExpr col = builder.getAffineDimExpr(1);
  AffineExpr expr = row * builder.getAffineConstantExpr(stride) + col;
  auto indexMap = AffineMapAttr::get(AffineMap::get(2, 0, expr, ctx));

  return LayoutAttr::get(ctx, shapeAttr, indexMap, AffineMapAttr(),
                         IntegerAttr());
}

static LayoutAttr buildSharedLayoutForType(OpBuilder &builder, MemRefType type,
                                           bool transpose) {
  auto maybeShape = extractMatrixShape(type);
  if (failed(maybeShape))
    return LayoutAttr();
  auto dims = *maybeShape;
  int64_t elementBits = type.getElementTypeBitWidth();
  int64_t cols = transpose ? dims[0] : dims[1];
  int64_t padded = cols;
  if (elementBits > 0 && (elementBits * cols) % 256 == 0)
    padded += 128 / elementBits;
  return buildLinearLayoutAttr(builder, {dims[0], dims[1]}, padded, transpose);
}

static LayoutAttr buildFragmentLayoutAttr(OpBuilder &builder,
                                          MemRefType type,
                                          int64_t tileRows, int64_t tileCols,
                                          unsigned warpSize) {
  auto maybeShape = extractMatrixShape(type);
  if (failed(maybeShape))
    return LayoutAttr();
  auto dims = *maybeShape;
  DenseI64ArrayAttr shapeAttr = buildShapeAttr(builder, {dims[0], dims[1]});
  MLIRContext *ctx = builder.getContext();

  AffineExpr row = builder.getAffineDimExpr(0);
  AffineExpr col = builder.getAffineDimExpr(1);
  AffineExpr index =
      row * builder.getAffineConstantExpr(dims[1]) + col;
  auto indexMap = AffineMapAttr::get(AffineMap::get(2, 0, index, ctx));

  int64_t laneM = std::max<int64_t>(1, std::min<int64_t>(tileRows, 16));
  int64_t laneN = std::max<int64_t>(1, std::min<int64_t>(tileCols, 8));
  // Clamp to warp size to avoid producing values outside of lane bounds.
  if (laneM * laneN > static_cast<int64_t>(warpSize)) {
    laneN = std::max<int64_t>(1, warpSize / laneM);
  }
  AffineExpr lane =
      (row % laneM) * builder.getAffineConstantExpr(laneN) + (col % laneN);
  auto threadMap = AffineMapAttr::get(AffineMap::get(2, 0, lane, ctx));
  auto replicate = builder.getI64IntegerAttr(1);
  return LayoutAttr::get(ctx, shapeAttr, indexMap, threadMap, replicate);
}

static std::optional<int64_t> tryComputeStaticElementCount(MemRefType type) {
  int64_t total = 1;
  for (int64_t dim : type.getShape()) {
    if (dim < 0)
      return std::nullopt;
    total *= dim;
  }
  return total;
}

static LayoutAttr buildDefaultLinearLayout(OpBuilder &builder, MemRefType type,
                                           bool attachThreadMap) {
  ArrayRef<int64_t> shape = type.getShape();
  for (int64_t dim : shape) {
    if (dim < 0)
      return LayoutAttr();
  }
  SmallVector<int64_t> dims(shape.begin(), shape.end());
  if (dims.empty())
    dims.push_back(1);
  auto shapeAttr = builder.getDenseI64ArrayAttr(dims);
  MLIRContext *ctx = builder.getContext();
  AffineExpr expr = builder.getAffineConstantExpr(0);
  if (!dims.empty()) {
    int64_t stride = 1;
    for (int64_t dim = static_cast<int64_t>(dims.size()) - 1; dim >= 0; --dim) {
      AffineExpr dimExpr = builder.getAffineDimExpr(dim);
      expr = expr + dimExpr * builder.getAffineConstantExpr(stride);
      int64_t size = std::max<int64_t>(1, dims[dim]);
      stride *= size;
    }
  }
  auto indexMap =
      AffineMapAttr::get(AffineMap::get(dims.size(), 0, expr, ctx));
  AffineMapAttr threadMap;
  if (attachThreadMap) {
    AffineExpr lane = builder.getAffineConstantExpr(0);
    threadMap =
        AffineMapAttr::get(AffineMap::get(dims.size(), 0, lane, ctx));
  }
  return LayoutAttr::get(ctx, shapeAttr, indexMap, threadMap, IntegerAttr());
}

static LayoutAttr attachReplicate(LayoutAttr layout, OpBuilder &builder,
                                  int64_t replicate) {
  if (!layout || replicate <= 1)
    return layout;
  return LayoutAttr::get(layout.getContext(), layout.getInputShape(),
                         layout.getForwardIndex(), layout.getForwardThread(),
                         builder.getI64IntegerAttr(replicate));
}

static LayoutAttr inferSharedLayoutForType(OpBuilder &builder,
                                           MemRefType type) {
  LayoutAttr layout = buildSharedLayoutForType(builder, type,
                                               /*transpose=*/false);
  if (!layout)
    layout = buildDefaultLinearLayout(builder, type,
                                      /*attachThreadMap=*/false);
  return layout;
}

static LayoutAttr inferFragmentLayoutForType(OpBuilder &builder,
                                             MemRefType type,
                                             const TargetInfo *targetInfo) {
  unsigned warpSize = targetInfo ? targetInfo->warpSize : 32;
  LayoutAttr layout;
  auto maybeShape = extractMatrixShape(type);
  if (succeeded(maybeShape)) {
    int64_t tileRows = (*maybeShape)[0];
    int64_t tileCols = (*maybeShape)[1];
    layout =
        buildFragmentLayoutAttr(builder, type, tileRows, tileCols, warpSize);
  }
  if (!layout)
    layout = buildDefaultLinearLayout(builder, type,
                                      /*attachThreadMap=*/true);
  return layout;
}

static bool valueHandledBySpecializedInference(Value buffer) {
  return llvm::any_of(buffer.getUsers(), [](Operation *user) {
    return isa<GemmOp>(user);
  });
}

static LayoutAttr inferLayoutForAllocBuffer(OpBuilder &builder,
                                            AllocBufferOp alloc,
                                            const TargetInfo *targetInfo,
                                            int64_t threadCount) {
  auto memType = dyn_cast<MemRefType>(alloc.getResult().getType());
  if (!memType)
    return LayoutAttr();

  LayoutAttr layout;
  switch (alloc.getMemorySpace()) {
  case attr::MemorySpace::Shared:
    layout = inferSharedLayoutForType(builder, memType);
    break;
  case attr::MemorySpace::Local:
    layout = inferFragmentLayoutForType(builder, memType, targetInfo);
    break;
  case attr::MemorySpace::Global:
  default:
    layout = buildDefaultLinearLayout(builder, memType,
                                      /*attachThreadMap=*/false);
    break;
  }
  if (!layout)
    return LayoutAttr();

  if (auto elements = tryComputeStaticElementCount(memType)) {
    if (*elements == 1)
      layout = attachReplicate(
          layout, builder, std::max<int64_t>(int64_t(1), threadCount));
  }
  return layout;
}

static std::pair<int64_t, int64_t>
computeWarpPartition(attr::GemmWarpPolicy policy, int64_t M, int64_t N,
                     int64_t blockSize, const TargetInfo &target) {
  int64_t warpSize = target.warpSize;
  int64_t numWarps = warpSize ? blockSize / warpSize : 1;
  int64_t kMPerWarp = 16;
  int64_t kNPerWarp = target.isCuda && target.smVersion >= 70 ? 8 : 16;
  if (M == 0 || N == 0 || numWarps == 0)
    return {1, 1};

  int64_t mWarp = 1;
  int64_t nWarp = numWarps;
  switch (policy) {
  case attr::GemmWarpPolicy::FullRow: {
    mWarp = numWarps;
    if (M % (mWarp * kMPerWarp) != 0) {
      int64_t maxMWarp = std::max<int64_t>(1, M / kMPerWarp);
      mWarp = std::max<int64_t>(1, std::min<int64_t>(numWarps, maxMWarp));
      nWarp = std::max<int64_t>(1, numWarps / mWarp);
    } else {
      nWarp = std::max<int64_t>(1, numWarps / mWarp);
    }
    break;
  }
  case attr::GemmWarpPolicy::FullCol: {
    nWarp = numWarps;
    if (N % (nWarp * kNPerWarp) != 0) {
      int64_t maxNWarp = std::max<int64_t>(1, N / kNPerWarp);
      nWarp = std::max<int64_t>(1, std::min<int64_t>(numWarps, maxNWarp));
      mWarp = std::max<int64_t>(1, numWarps / nWarp);
    } else {
      mWarp = std::max<int64_t>(1, numWarps / nWarp);
    }
    break;
  }
  case attr::GemmWarpPolicy::Square:
  default: {
    float idealRatio = N > 0 ? static_cast<float>(M) / static_cast<float>(N)
                             : 1.0f;
    float bestScore = std::numeric_limits<float>::max();
    for (int64_t m = 1; m <= numWarps; ++m) {
      if (numWarps % m != 0)
        continue;
      int64_t n = numWarps / m;
      float mPerWarp = static_cast<float>(M) / (m * kMPerWarp);
      float nPerWarp = static_cast<float>(N) / (n * kNPerWarp);
      if (mPerWarp < 1 || nPerWarp < 1)
        continue;
      float score = std::abs(mPerWarp / nPerWarp - idealRatio);
      if (score < bestScore) {
        bestScore = score;
        mWarp = m;
        nWarp = n;
      }
    }
    break;
  }
  }
  if (mWarp <= 0)
    mWarp = 1;
  if (nWarp <= 0)
    nWarp = 1;
  return {mWarp, nWarp};
}

} // namespace

namespace {

static std::optional<int64_t> inferReduceThreadBlockSize(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto attr = cur->getAttrOfType<IntegerAttr>("frisk.threads"))
      return attr.getInt();
    if (auto parallel = dyn_cast<ParallelOp>(cur))
      return parallel.getThreads();
    cur = cur->getParentOp();
  }
  return std::nullopt;
}

static AffineMapAttr remapReduceDimension(OpBuilder &builder,
                                          AffineMapAttr mapAttr,
                                          unsigned removeDim,
                                          int64_t reduceExtent,
                                          bool useFloorDiv) {
  if (!mapAttr)
    return AffineMapAttr();

  AffineMap map = mapAttr.getValue();
  unsigned srcDims = map.getNumDims();
  if (removeDim >= srcDims)
    return AffineMapAttr();

  unsigned dstDims = srcDims - 1;
  unsigned placeholderDim = dstDims;
  unsigned newDimCount = dstDims + 1;

  MLIRContext *ctx = builder.getContext();
  AffineExpr placeholder = builder.getAffineDimExpr(placeholderDim);

  SmallVector<AffineExpr> dimSubs;
  dimSubs.reserve(srcDims);
  for (unsigned i = 0; i < srcDims; ++i) {
    if (i < removeDim) {
      dimSubs.push_back(builder.getAffineDimExpr(i));
      continue;
    }
    if (i == removeDim) {
      if (ShapedType::isDynamic(reduceExtent)) {
        dimSubs.push_back(placeholder);
      } else if (reduceExtent == 1) {
        dimSubs.push_back(builder.getAffineConstantExpr(0));
      } else {
        AffineExpr constant = builder.getAffineConstantExpr(reduceExtent);
        dimSubs.push_back(useFloorDiv ? placeholder.floorDiv(constant)
                                      : placeholder % constant);
      }
      continue;
    }
    dimSubs.push_back(builder.getAffineDimExpr(i - 1));
  }

  SmallVector<AffineExpr> symSubs;
  symSubs.reserve(map.getNumSymbols());
  for (unsigned i = 0; i < map.getNumSymbols(); ++i)
    symSubs.push_back(builder.getAffineSymbolExpr(i));

  SmallVector<AffineExpr> newResults;
  newResults.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults())
    newResults.push_back(expr.replaceDimsAndSymbols(dimSubs, symSubs));

  AffineMap newMap =
      AffineMap::get(newDimCount, map.getNumSymbols(), newResults, ctx);
  return AffineMapAttr::get(newMap);
}

static DenseI64ArrayAttr buildReduceShapeAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> shape) {
  SmallVector<int64_t> values(shape.begin(), shape.end());
  if (values.empty())
    values.push_back(1);
  return builder.getDenseI64ArrayAttr(values);
}

static bool mapsAgreeOnDomain(MLIRContext *ctx, AffineMap lhs, AffineMap rhs,
                              ArrayRef<int64_t> dimExtents,
                              ArrayRef<int64_t> lhsSymbolExtents,
                              ArrayRef<int64_t> rhsSymbolExtents) {
  if (lhs.getNumDims() != rhs.getNumDims() ||
      lhs.getNumResults() != rhs.getNumResults())
    return false;
  if (lhs.getNumDims() != dimExtents.size() ||
      lhs.getNumSymbols() != lhsSymbolExtents.size() ||
      rhs.getNumSymbols() != rhsSymbolExtents.size())
    return false;

  Builder builder(ctx);
  SmallVector<int64_t> dimValues(dimExtents.size(), 0);
  SmallVector<int64_t> lhsSymValues(lhsSymbolExtents.size(), 0);
  SmallVector<int64_t> rhsSymValues(rhsSymbolExtents.size(), 0);

  auto foldMap = [&](AffineMap map, ArrayRef<int64_t> dims,
                     ArrayRef<int64_t> symbols,
                     SmallVectorImpl<Attribute> &results) -> LogicalResult {
    SmallVector<Attribute> operands;
    operands.reserve(dims.size() + symbols.size());
    for (int64_t value : dims)
      operands.push_back(builder.getI64IntegerAttr(value));
    for (int64_t value : symbols)
      operands.push_back(builder.getI64IntegerAttr(value));
    return map.constantFold(operands, results);
  };

  std::function<bool(unsigned)> enumerateDims = [&](unsigned index) -> bool {
    if (index == dimExtents.size()) {
      std::function<bool(unsigned)> enumerateLhsSymbols =
          [&](unsigned lhsIndex) -> bool {
        if (lhsIndex == lhsSymbolExtents.size()) {
          std::function<bool(unsigned)> enumerateRhsSymbols =
              [&](unsigned rhsIndex) -> bool {
            if (rhsIndex == rhsSymbolExtents.size()) {
              SmallVector<Attribute> lhsResults;
              SmallVector<Attribute> rhsResults;
              if (failed(foldMap(lhs, dimValues, lhsSymValues, lhsResults)) ||
                  failed(foldMap(rhs, dimValues, rhsSymValues, rhsResults)))
                return false;
              return lhsResults == rhsResults;
            }
            for (int64_t value = 0; value < rhsSymbolExtents[rhsIndex];
                 ++value) {
              rhsSymValues[rhsIndex] = value;
              if (!enumerateRhsSymbols(rhsIndex + 1))
                return false;
            }
            return true;
          };
          return enumerateRhsSymbols(0);
        }
        for (int64_t value = 0; value < lhsSymbolExtents[lhsIndex]; ++value) {
          lhsSymValues[lhsIndex] = value;
          if (!enumerateLhsSymbols(lhsIndex + 1))
            return false;
        }
        return true;
      };
      return enumerateLhsSymbols(0);
    }

    for (int64_t value = 0; value < dimExtents[index]; ++value) {
      dimValues[index] = value;
      if (!enumerateDims(index + 1))
        return false;
    }
    return true;
  };

  return enumerateDims(0);
}

static LogicalResult checkLayoutCompatibility(ReduceOp op,
                                              LayoutAttr computed,
                                              LayoutAttr existing) {
  auto emitConflict = [&](StringRef detail) -> LogicalResult {
    op.emitOpError("destination layout conflicts with reduce inference: ")
        << detail << "\nexpected=" << layoutDebugString(computed)
        << "\nactual=" << layoutDebugString(existing);
    return failure();
  };

  if (!existing)
    return success();

  auto dimShape = computed.getInputShape();
  if (dimShape != existing.getInputShape())
    return emitConflict("input shapes disagree");

  auto lhsRep = computed.getReplicateSize();
  auto rhsRep = existing.getReplicateSize();
  int64_t computedRep = lhsRep ? lhsRep.getInt() : 1;
  int64_t existingRep = rhsRep ? rhsRep.getInt() : 1;
  if (computedRep <= 0 || existingRep <= 0)
    return emitConflict("replicate extent must be positive");
  if (computedRep > existingRep)
    return emitConflict("existing layout replicates fewer lanes than inferred");

  SmallVector<int64_t> dimExtents(dimShape.asArrayRef().begin(),
                                  dimShape.asArrayRef().end());
  if (!mapsAgreeOnDomain(op.getContext(), computed.getForwardIndex().getValue(),
                         existing.getForwardIndex().getValue(), dimExtents,
                         /*lhsSymbolExtents=*/{}, /*rhsSymbolExtents=*/{}))
    return emitConflict("forward index maps differ");

  AffineMap computedThread = computed.getForwardThread().getValue();
  AffineMap existingThread = existing.getForwardThread().getValue();
  SmallVector<int64_t> computedSymbols(computedThread.getNumSymbols(),
                                       computedRep);
  SmallVector<int64_t> existingSymbols(existingThread.getNumSymbols(),
                                       computedRep);
  if (!mapsAgreeOnDomain(op.getContext(), computedThread, existingThread,
                         dimExtents, computedSymbols, existingSymbols))
    return emitConflict("thread maps differ");
  return success();
}

} // namespace

LogicalResult inferLegacyParallelLayout(ParallelOp parallel, OpBuilder &builder,
                                    DenseMap<Value, Attribute> &layoutMap) {
  ++legacyParallelCalls;
  bool updated = false;
  auto targetInfo = detectTargetInfo(parallel.getOperation());
  int64_t threadCount = parallel.getThreadNum();

  // Seed layouts for buffers that do not have a dedicated inference routine.
  parallel.getRegion().walk([&](AllocBufferOp alloc) {
    if (auto parent = alloc->getParentOfType<ParallelOp>())
      if (parent != parallel)
        return;
    Value buffer = alloc.getResult();
    if (layoutMap.count(buffer))
      return;
    if (valueHandledBySpecializedInference(buffer))
      return;
    LayoutAttr layout =
        inferLayoutForAllocBuffer(builder, alloc,
                                  targetInfo ? &*targetInfo : nullptr,
                                  threadCount);
    if (!layout)
      return;
    if (layoutMap.try_emplace(buffer, layout).second)
      updated = true;
  });

  WalkResult walkResult =
      parallel.getRegion().walk([&](Operation *nested) -> WalkResult {
        if (auto innerParallel = dyn_cast<ParallelOp>(nested)) {
          if (innerParallel == parallel)
            return WalkResult::advance();
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(innerParallel);
          auto result = inferLegacyParallelLayout(innerParallel, builder, layoutMap);
          if (failed(result))
            return WalkResult::interrupt();
          updated |= succeeded(result);
          return WalkResult::skip();
        }
        if (auto gemm = dyn_cast<GemmOp>(nested)) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(gemm);
          auto result = inferLegacyGemmLayout(gemm, builder, layoutMap);
          if (failed(result))
            return WalkResult::interrupt();
          updated |= succeeded(result);
        }
        return WalkResult::advance();
      });
  if (walkResult.wasInterrupted())
    return failure();
  return success(updated);
}

LogicalResult inferLegacyGemmLayout(GemmOp gemm, OpBuilder &builder,
                                    DenseMap<Value, Attribute> &layoutMap) {
  Operation *op = gemm.getOperation();
  auto targetInfo = detectTargetInfo(op);
  if (!targetInfo)
    return gemm.emitOpError()
           << "layout inference requires a 'frisk.target' string attribute";
  auto blockSize = inferThreadBlockSize(op);
  if (!blockSize)
    return gemm.emitOpError()
           << "layout inference requires an enclosing 'frisk.parallel' "
              "operation to provide a thread count";

  auto memA = dyn_cast<MemRefType>(gemm.getA().getType());
  auto memB = dyn_cast<MemRefType>(gemm.getB().getType());
  auto memC = dyn_cast<MemRefType>(gemm.getC().getType());
  if (!memA || !memB || !memC)
    return gemm.emitOpError("all operands must be memref values for layout inference");

  auto parseMemorySpace = [&](MemRefType type,
                              StringRef label) -> std::optional<attr::MemorySpace> {
    if (std::optional<attr::MemorySpace> memorySpace =
            getFriskMemorySpace(type))
      return memorySpace;
    gemm.emitOpError() << "operand " << label
                  << " resides in an unsupported memory space";
    return std::nullopt;
  };

  auto aSpace = parseMemorySpace(memA, "A");
  auto bSpace = parseMemorySpace(memB, "B");
  auto cSpace = parseMemorySpace(memC, "C");
  if (!aSpace || !bSpace || !cSpace)
    return failure();

  if (*cSpace != attr::MemorySpace::Local)
    return gemm.emitOpError("operand C must reside in local memory space");
  auto isSupportedOperandSpace = [&](attr::MemorySpace space,
                                     StringRef label) -> LogicalResult {
    if (space != attr::MemorySpace::Shared && space != attr::MemorySpace::Local)
      return gemm.emitOpError()
             << "operand " << label
             << " must reside in shared or local memory space for layout inference";
    return success();
  };
  if (failed(isSupportedOperandSpace(*aSpace, "A")) ||
      failed(isSupportedOperandSpace(*bSpace, "B")))
    return failure();

  bool hopper = isHopper(*targetInfo);
  bool ampere = isAmpere(*targetInfo);
  if (!hopper && !ampere)
    return gemm.emitOpError(
        "layout inference currently supports only Ampere or Hopper targets");

  int64_t blockM = gemm.getM();
  int64_t blockN = gemm.getN();
  int64_t blockK = gemm.getK();

  auto warpPartition =
      computeWarpPartition(gemm.getPolicyAttr().getValue(), blockM, blockN,
                           *blockSize, *targetInfo);
  int64_t warpCountM = std::max<int64_t>(warpPartition.first, 1);
  int64_t warpCountN = std::max<int64_t>(warpPartition.second, 1);
  if (warpCountM == 0 || warpCountN == 0)
    return gemm.emitOpError("invalid warp partition derived from warp policy");
  if (blockM % warpCountM != 0 || blockN % warpCountN != 0)
    return gemm.emitOpError("block shape must be divisible by warp partition");
  int64_t warpTileM = blockM / warpCountM;
  int64_t warpTileN = blockN / warpCountN;
  GemmInst inst = inferGemmInst(*targetInfo, *blockSize, blockM, blockN);

  LayoutAttr layoutA;
  if (*aSpace == attr::MemorySpace::Shared) {
    layoutA = hopper ? buildHopperSharedLayout(builder, memA, !gemm.getTransA())
                     : buildAmpereSharedLayout(builder, memA, !gemm.getTransA());
    if (!layoutA)
      return gemm.emitOpError("unable to materialize shared-memory layout for operand A");
  } else {
    auto fragmentA = buildGemmFragmentA(builder.getContext(), blockM, blockN,
                                        blockK, warpTileM, warpTileN,
                                        memA.getElementTypeBitWidth(),
                                        gemm.getTransA());
    if (!fragmentA)
      return gemm.emitOpError("unable to build fragment layout for operand A; "
                         "check tile shape and element type");
    layoutA = makeFragmentLayout(builder, *fragmentA);
  }

  LayoutAttr layoutB;
  if (*bSpace == attr::MemorySpace::Shared) {
    layoutB = hopper ? buildHopperSharedLayout(builder, memB, gemm.getTransB())
                     : buildAmpereSharedLayout(builder, memB, gemm.getTransB());
    if (!layoutB)
      return gemm.emitOpError("unable to materialize shared-memory layout for operand B");
  } else {
    auto fragmentB = buildGemmFragmentB(builder.getContext(), blockM, blockN,
                                        blockK, warpTileM, warpTileN,
                                        gemm.getTransB());
    if (!fragmentB)
      return gemm.emitOpError("unable to build fragment layout for operand B; "
                         "check warp policy and tile shape");
    layoutB = makeFragmentLayout(builder, *fragmentB);
  }

  std::optional<FragmentExpr> fragment;
  if (hopper && inst == GemmInst::WGMMA)
    fragment = buildHopperFragmentC(builder.getContext(), gemm.getM(), gemm.getN(),
                                    warpTileM, warpTileN);
  else
    fragment = buildAmpereFragmentC(builder.getContext(), gemm.getM(), gemm.getN(),
                                    warpTileM, warpTileN);
  if (!fragment)
    return gemm.emitOpError("unable to build accumulator fragment for current warp "
                       "shape; check warp policy and target");
  LayoutAttr layoutC = makeFragmentLayout(builder, *fragment);

  bool updated = false;
  updated |= layoutMap.try_emplace(gemm.getA(), layoutA).second;
  updated |= layoutMap.try_emplace(gemm.getB(), layoutB).second;
  updated |= layoutMap.try_emplace(gemm.getC(), layoutC).second;
  return success(updated);
}
LogicalResult inferLegacyReduceLayout(ReduceOp reduce, OpBuilder &builder,
                                    DenseMap<Value, Attribute> &layoutMap) {
  auto srcType = dyn_cast<MemRefType>(reduce.getSrc().getType());
  auto dstType = dyn_cast<MemRefType>(reduce.getDst().getType());
  if (!srcType || !dstType)
    return reduce.emitOpError("layout inference requires memref operands");

  auto parseMemorySpace = [&](MemRefType type,
                              StringRef label) -> std::optional<attr::MemorySpace> {
    if (std::optional<attr::MemorySpace> memorySpace =
            getFriskMemorySpace(type))
      return memorySpace;
    reduce.emitOpError() << "operand " << label
                  << " resides in an unsupported memory space";
    return std::nullopt;
  };

  auto srcSpace = parseMemorySpace(srcType, "src");
  auto dstSpace = parseMemorySpace(dstType, "dst");
  if (!srcSpace || !dstSpace)
    return failure();

  if (*srcSpace != attr::MemorySpace::Local ||
      *dstSpace != attr::MemorySpace::Local)
    return success();

  auto srcIt = layoutMap.find(reduce.getSrc());
  if (srcIt == layoutMap.end())
    return success();

  auto srcLayout = dyn_cast<LayoutAttr>(srcIt->second);
  if (!srcLayout)
    return reduce.emitOpError("source layout entry must be a frisk.layout attribute");

  auto srcThreadMap = srcLayout.getForwardThread();
  if (!srcThreadMap)
    return success();

  DenseI64ArrayAttr layoutShape = srcLayout.getInputShape();
  if (!layoutShape)
    return reduce.emitOpError("source layout missing input shape metadata");

  ArrayRef<int64_t> layoutDims = layoutShape.asArrayRef();
  if (layoutDims.size() != static_cast<size_t>(srcType.getRank()))
    return reduce.emitOpError("source layout rank does not match source memref rank");

  int64_t dim = reduce.getDim();
  if (dim < 0 || dim >= srcType.getRank())
    return reduce.emitOpError("invalid reduce dimension ") << dim;

  int64_t reduceExtent = srcType.getShape()[dim];
  if (ShapedType::isDynamic(reduceExtent))
    return reduce.emitOpError(
        "layout inference requires static extent along the reduce dimension");
  if (reduceExtent <= 0)
    return reduce.emitOpError("reduce extent must be positive for layout inference");

  auto remappedThread = remapReduceDimension(
      builder, srcThreadMap, static_cast<unsigned>(dim), reduceExtent,
      /*useFloorDiv=*/false);

  std::optional<int64_t> replicateValue;
  int64_t baseReplicate = 1;
  if (auto replicateAttr = srcLayout.getReplicateSize()) {
    baseReplicate = replicateAttr.getInt();
    if (baseReplicate <= 0)
      return reduce.emitOpError("source layout replicate extent must be positive");
    replicateValue = baseReplicate;
  }

  int64_t uncondensedReplicate = std::max<int64_t>(int64_t(1), baseReplicate);
  if (llvm::MulOverflow(uncondensedReplicate, reduceExtent, uncondensedReplicate))
    return reduce.emitOpError("replicate extent overflow while inferring layout");

  auto threads = inferReduceThreadBlockSize(reduce.getOperation());
  if (threads && *threads > 0 && uncondensedReplicate > 0 &&
      (*threads % uncondensedReplicate != 0) &&
      (uncondensedReplicate % *threads != 0)) {
    return reduce.emitOpError()
           << "reduce layout inference requires thread count divisible by "
              "replicate extent before condense (threads="
           << *threads << ", replicate=" << uncondensedReplicate << ")";
  }

  if (!remappedThread)
    return reduce.emitOpError("failed to remap source thread layout for reduce op");
  AffineMap remappedThreadMap = remappedThread.getValue();
  if (remappedThreadMap.getNumDims() == 0)
    return reduce.emitOpError("thread layout missing placeholder dimension");

  unsigned placeholder = remappedThreadMap.getNumDims() - 1;
  auto compressedThread =
      compressReplicateDimInMap(builder, remappedThread, placeholder,
                                reduceExtent);
  if (!compressedThread || !compressedThread->mapAttr)
    return reduce.emitOpError("unable to condense reduce replicate dimension in "
                       "thread map");

  auto dstThreadMap = compressedThread->mapAttr;
  auto dstIndexMap =
      inferFragmentIndexFromThreadMap(builder, dstThreadMap, dstType.getShape());
  if (!dstIndexMap)
    return reduce.emitOpError("failed to infer destination fragment index from "
                       "thread map");

  int64_t condensedReplicate = std::max<int64_t>(
      int64_t(1), compressedThread->replicateExtent);
  int64_t finalReplicate = std::max<int64_t>(int64_t(1), baseReplicate);
  if (llvm::MulOverflow(finalReplicate, condensedReplicate, finalReplicate))
    return reduce.emitOpError("replicate extent overflow while inferring layout");
  replicateValue = finalReplicate;

  IntegerAttr replicateAttr = builder.getI64IntegerAttr(finalReplicate);

  auto dstShapeAttr = buildReduceShapeAttr(builder, dstType.getShape());
  LayoutAttr dstLayout = LayoutAttr::get(
      builder.getContext(), dstShapeAttr, *dstIndexMap, dstThreadMap,
      replicateAttr);

  auto dstIt = layoutMap.find(reduce.getDst());
  if (dstIt != layoutMap.end()) {
    auto existingLayout = dyn_cast<LayoutAttr>(dstIt->second);
    if (!existingLayout)
      dstIt->second = dstLayout;
    if (auto layout = dyn_cast<LayoutAttr>(dstIt->second)) {
      if (failed(checkLayoutCompatibility(reduce, dstLayout, layout)))
        return failure();
      return success();
    }
    dstIt->second = dstLayout;
    return success(true);
  }

  layoutMap.try_emplace(reduce.getDst(), dstLayout);
  return success(true);
}



} // namespace mlir::frisk::test
