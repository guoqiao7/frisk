#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>
#include <string>

using namespace mlir;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void printHeader(StringRef title) {
  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  llvm::outs() << "-- " << title << " --\n";
  llvm::outs() << std::string(60, '-') << "\n";
}

MemRefType makeLocalType(OpBuilder &builder, ArrayRef<int64_t> shape,
                         Type elemTy) {
  return MemRefType::get(shape, elemTy, AffineMapAttr(),
                         builder.getI64IntegerAttr(static_cast<int64_t>(
                             mlir::frisk::attr::MemorySpace::Local)));
}

MemRefType makeSharedType(OpBuilder &builder, ArrayRef<int64_t> shape,
                          Type elemTy) {
  return MemRefType::get(shape, elemTy, AffineMapAttr(),
                         builder.getI64IntegerAttr(static_cast<int64_t>(
                             mlir::frisk::attr::MemorySpace::Shared)));
}

mlir::frisk::LayoutAttr buildLayoutAttr(OpBuilder &builder,
                                        ArrayRef<int64_t> shape,
                                        AffineMap indexMap,
                                        AffineMap threadMap,
                                        int64_t replicate) {
  auto shapeAttr = builder.getDenseI64ArrayAttr(shape);
  auto replicateAttr = builder.getI64IntegerAttr(replicate);
  return mlir::frisk::LayoutAttr::get(builder.getContext(), shapeAttr,
                                      AffineMapAttr::get(indexMap),
                                      AffineMapAttr::get(threadMap),
                                      replicateAttr);
}

/// Brute-force check that two affine maps produce the same outputs on all
/// points in the given domain.  Both maps must have the same number of dims
/// and symbols; symbols are enumerated in lockstep (paired comparison).
bool mapsEqualOnDomain(AffineMap lhs, AffineMap rhs,
                       ArrayRef<int64_t> dimExtents,
                       ArrayRef<int64_t> symbolExtents) {
  if (lhs.getNumDims() != rhs.getNumDims() ||
      lhs.getNumSymbols() != rhs.getNumSymbols() ||
      lhs.getNumResults() != rhs.getNumResults())
    return false;
  if (lhs.getNumDims() != dimExtents.size() ||
      lhs.getNumSymbols() != symbolExtents.size())
    return false;

  Builder builder(lhs.getContext());
  unsigned totalSlots = dimExtents.size() + symbolExtents.size();
  SmallVector<int64_t> values(totalSlots, 0);

  std::function<bool(unsigned)> enumerate = [&](unsigned index) -> bool {
    if (index == totalSlots) {
      SmallVector<Attribute> attrs;
      attrs.reserve(totalSlots);
      for (int64_t v : values)
        attrs.push_back(builder.getI64IntegerAttr(v));
      SmallVector<Attribute> lr, rr;
      if (failed(lhs.constantFold(attrs, lr)) ||
          failed(rhs.constantFold(attrs, rr)))
        return false;
      return lr == rr;
    }
    int64_t extent = index < dimExtents.size()
                         ? dimExtents[index]
                         : symbolExtents[index - dimExtents.size()];
    for (int64_t v = 0; v < extent; ++v) {
      values[index] = v;
      if (!enumerate(index + 1))
        return false;
    }
    return true;
  };

  return enumerate(0);
}

// ---------------------------------------------------------------------------
// Parameterised reduce test case
// ---------------------------------------------------------------------------

struct ReduceCase {
  const char *name;

  // Source / destination geometry.
  SmallVector<int64_t, 3> srcShape;
  SmallVector<int64_t, 3> dstShape;
  int64_t reduceDim;
  StringRef reduceKind;

  // Source layout.
  AffineMap srcIndexMap;
  AffineMap srcThreadMap;
  int64_t srcReplicate;

  // Expected destination layout.
  AffineMap expectedIndexMap;
  AffineMap expectedThreadMap;
  int64_t expectedReplicate;

  // Thread block size.
  int64_t threads;
};

bool runReduceCase(MLIRContext &context, const ReduceCase &tc) {
  printHeader(tc.name);

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto f16 = builder.getF16Type();
  auto srcType = makeLocalType(builder, tc.srcShape, f16);
  auto dstType = makeLocalType(builder, tc.dstShape, f16);
  auto funcType = builder.getFunctionType({srcType, dstType}, {});

  auto kernel =
      builder.create<mlir::frisk::KernelOp>(loc, tc.name, funcType);
  Block *entry = kernel.addEntryBlock();

  OpBuilder bodyBuilder(&context);
  bodyBuilder.setInsertionPointToStart(entry);
  auto reduce = bodyBuilder.create<mlir::frisk::ReduceOp>(
      loc, entry->getArgument(0), entry->getArgument(1), tc.reduceKind,
      tc.reduceDim, /*clear=*/false);
  reduce->setAttr("frisk.threads",
                  bodyBuilder.getI64IntegerAttr(tc.threads));
  bodyBuilder.create<mlir::frisk::EndOp>(loc);

  // Seed the source layout.
  DenseMap<Value, Attribute> layoutMap;
  OpBuilder inferBuilder(&context);
  inferBuilder.setInsertionPoint(reduce);
  layoutMap.try_emplace(
      reduce.getSrc(),
      buildLayoutAttr(inferBuilder, tc.srcShape, tc.srcIndexMap,
                      tc.srcThreadMap, tc.srcReplicate));

  if (failed(reduce.inferLayout(inferBuilder, layoutMap))) {
    llvm::errs() << "inferLayout failed for '" << tc.name << "'\n";
    return false;
  }

  auto it = layoutMap.find(reduce.getDst());
  if (it == layoutMap.end()) {
    llvm::errs() << "missing destination layout for '" << tc.name << "'\n";
    return false;
  }

  auto dstLayout = dyn_cast<mlir::frisk::LayoutAttr>(it->second);
  if (!dstLayout) {
    llvm::errs() << "destination layout wrong type for '" << tc.name << "'\n";
    return false;
  }

  // --- Check replicate ---
  std::optional<int64_t> replicateValue;
  if (auto attr = dstLayout.getReplicateSize())
    replicateValue = attr.getInt();
  if (!replicateValue || *replicateValue != tc.expectedReplicate) {
    llvm::errs() << "replicate mismatch for '" << tc.name << "': want "
                 << tc.expectedReplicate << " got ";
    if (replicateValue)
      llvm::errs() << *replicateValue;
    else
      llvm::errs() << "<none>";
    llvm::errs() << "\n";
    return false;
  }

  // --- Check input shape ---
  auto outputShape = dstLayout.getInputShape();
  SmallVector<int64_t> expectedShape(tc.dstShape.begin(), tc.dstShape.end());
  if (outputShape.asArrayRef() != ArrayRef<int64_t>(expectedShape)) {
    llvm::errs() << "shape mismatch for '" << tc.name << "'\n";
    return false;
  }

  // --- Check forward index ---
  auto actualIndex = dstLayout.getForwardIndex();
  SmallVector<int64_t> dstDimExtents(tc.dstShape.begin(), tc.dstShape.end());

  if (!actualIndex ||
      !mapsEqualOnDomain(actualIndex.getValue(), tc.expectedIndexMap,
                         dstDimExtents, /*symbolExtents=*/{})) {
    llvm::errs() << "index map mismatch for '" << tc.name << "'\nwant: "
                 << AffineMapAttr::get(tc.expectedIndexMap)
                 << "\ngot:  " << actualIndex << "\n";
    return false;
  }

  // --- Check forward thread ---
  auto actualThread = dstLayout.getForwardThread();
  SmallVector<int64_t> symExtents(
      tc.expectedThreadMap.getNumSymbols(), tc.expectedReplicate);

  if (!actualThread ||
      !mapsEqualOnDomain(actualThread.getValue(), tc.expectedThreadMap,
                         dstDimExtents, symExtents)) {
    llvm::errs() << "thread map mismatch for '" << tc.name << "'\nwant: "
                 << AffineMapAttr::get(tc.expectedThreadMap)
                 << "\ngot:  " << actualThread << "\n";
    return false;
  }

  llvm::outs() << "  " << mlir::frisk::layoutDebugString(dstLayout) << "\n";
  llvm::outs() << "  PASS\n";
  return true;
}

// ---------------------------------------------------------------------------
// Integration test: GEMM C fragment -> reduce
// ---------------------------------------------------------------------------

bool runGemmReduceIntegration(MLIRContext &context, StringRef target,
                              int64_t blockM, int64_t blockN, int64_t blockK,
                              int64_t threads, int64_t reduceDim) {
  std::string label;
  llvm::raw_string_ostream labelOS(label);
  labelOS << "Integration: " << target << " " << blockM << "x" << blockN
          << " reduce dim=" << reduceDim << " threads=" << threads;
  labelOS.flush();
  printHeader(label);

  OpBuilder moduleBuilder(&context);
  auto loc = moduleBuilder.getUnknownLoc();
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
  module->getOperation()->setAttr("frisk.target",
                                  moduleBuilder.getStringAttr(target));

  auto f16 = moduleBuilder.getF16Type();
  auto makeSpaceAttr = [&](mlir::frisk::attr::MemorySpace space) {
    return moduleBuilder.getI64IntegerAttr(static_cast<int64_t>(space));
  };

  auto memAType = MemRefType::get({blockM, blockK}, f16, AffineMapAttr(),
                                  makeSpaceAttr(mlir::frisk::attr::MemorySpace::Shared));
  auto memBType = MemRefType::get({blockK, blockN}, f16, AffineMapAttr(),
                                  makeSpaceAttr(mlir::frisk::attr::MemorySpace::Shared));
  auto memCType = MemRefType::get({blockM, blockN}, f16, AffineMapAttr(),
                                  makeSpaceAttr(mlir::frisk::attr::MemorySpace::Local));

  // Destination shape after reducing one dimension of C.
  SmallVector<int64_t> dstShape;
  for (int64_t d = 0; d < 2; ++d) {
    if (d != reduceDim)
      dstShape.push_back(memCType.getShape()[d]);
  }
  auto memDType = MemRefType::get(dstShape, f16, AffineMapAttr(),
                                  makeSpaceAttr(mlir::frisk::attr::MemorySpace::Local));

  auto funcType = moduleBuilder.getFunctionType(
      {memAType, memBType, memCType, memDType}, {});
  Block &moduleBlock = module->getBodyRegion().front();
  moduleBuilder.setInsertionPointToStart(&moduleBlock);
  auto kernel = moduleBuilder.create<mlir::frisk::KernelOp>(
      loc, "gemm_reduce_test", funcType);
  Block *entry = kernel.addEntryBlock();

  OpBuilder bodyBuilder(&context);
  bodyBuilder.setInsertionPoint(entry->getTerminator());

  auto gemm = bodyBuilder.create<mlir::frisk::GemmOp>(
      loc, entry->getArgument(0), entry->getArgument(1),
      entry->getArgument(2), false, false,
      static_cast<uint64_t>(blockM), static_cast<uint64_t>(blockN),
      static_cast<uint64_t>(blockK),
      mlir::frisk::attr::GemmWarpPolicy::Square, false);
  gemm->setAttr("frisk.threads", bodyBuilder.getI64IntegerAttr(threads));

  auto reduce = bodyBuilder.create<mlir::frisk::ReduceOp>(
      loc, entry->getArgument(2), entry->getArgument(3), "add",
      reduceDim, /*clear=*/true);
  reduce->setAttr("frisk.threads", bodyBuilder.getI64IntegerAttr(threads));

  // Step 1: infer GEMM layout (populates C's layout).
  OpBuilder gemmInferBuilder(&context);
  gemmInferBuilder.setInsertionPoint(gemm);
  DenseMap<Value, Attribute> layoutMap;
  if (failed(gemm.inferLayout(gemmInferBuilder, layoutMap))) {
    llvm::errs() << "GEMM inferLayout failed for '" << label << "'\n";
    return false;
  }

  auto cIt = layoutMap.find(gemm.getC());
  if (cIt == layoutMap.end()) {
    llvm::errs() << "GEMM did not produce C layout for '" << label << "'\n";
    return false;
  }
  auto cLayout = dyn_cast<mlir::frisk::LayoutAttr>(cIt->second);
  if (!cLayout || !cLayout.getForwardThread()) {
    llvm::errs() << "GEMM C layout incomplete for '" << label << "'\n";
    return false;
  }
  llvm::outs() << "  C layout: " << mlir::frisk::layoutDebugString(cLayout)
               << "\n";

  // Step 2: infer reduce layout.
  OpBuilder reduceInferBuilder(&context);
  reduceInferBuilder.setInsertionPoint(reduce);
  // Seed C's layout into the reduce's source.
  layoutMap.try_emplace(reduce.getSrc(), cLayout);

  if (failed(reduce.inferLayout(reduceInferBuilder, layoutMap))) {
    llvm::errs() << "Reduce inferLayout failed for '" << label << "'\n";
    return false;
  }

  auto dIt = layoutMap.find(reduce.getDst());
  if (dIt == layoutMap.end()) {
    llvm::errs() << "Reduce did not produce dst layout for '" << label
                 << "'\n";
    return false;
  }
  auto dLayout = dyn_cast<mlir::frisk::LayoutAttr>(dIt->second);
  if (!dLayout) {
    llvm::errs() << "Reduce dst layout wrong type for '" << label << "'\n";
    return false;
  }
  llvm::outs() << "  D layout: " << mlir::frisk::layoutDebugString(dLayout)
               << "\n";

  // Structural checks.
  bool ok = true;

  // Shape must match destination memref.
  if (dLayout.getInputShape().asArrayRef() != ArrayRef<int64_t>(dstShape)) {
    llvm::errs() << "  FAIL: shape mismatch\n";
    ok = false;
  }

  // Must have thread map and index map.
  if (!dLayout.getForwardThread() || !dLayout.getForwardIndex()) {
    llvm::errs() << "  FAIL: missing thread or index map\n";
    ok = false;
  }

  // Replicate must be positive.
  if (auto rep = dLayout.getReplicateSize()) {
    if (rep.getInt() <= 0) {
      llvm::errs() << "  FAIL: non-positive replicate\n";
      ok = false;
    }
    // Replicate must be compatible with thread count.
    if (threads % rep.getInt() != 0 && rep.getInt() % threads != 0) {
      llvm::errs() << "  FAIL: replicate " << rep.getInt()
                   << " incompatible with threads " << threads << "\n";
      ok = false;
    }
  }

  if (ok)
    llvm::outs() << "  PASS\n";
  return ok;
}

} // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  DialectRegistry registry;
  registry.insert<mlir::frisk::FriskDialect>();
  MLIRContext context(registry);
  context.loadDialect<mlir::frisk::FriskDialect>();

  OpBuilder builder(&context);
  MLIRContext *ctx = &context;

  // Commonly used affine expressions.
  auto d = [&](unsigned i) { return builder.getAffineDimExpr(i); };
  auto c = [&](int64_t v) { return builder.getAffineConstantExpr(v); };
  auto s = [&](unsigned i) { return builder.getAffineSymbolExpr(i); };
  auto mkMap = [&](unsigned dims, unsigned syms, AffineExpr expr) {
    return AffineMap::get(dims, syms, expr, ctx);
  };

  // -----------------------------------------------------------------------
  // Case 1: Tensor-core 8x8 fragment, reduce dim=1 (row-wise, e.g. softmax)
  //
  // This is the base building block of an Ampere GEMM C output.
  // makeFragment8x8: shape={8,8}, index = j%2, thread = j/2 + 4*i
  // Each of 32 threads holds 2 elements.
  // Reducing dim=1 collapses columns: j/2 contributes 4 thread values from j.
  //
  // Expected: dst={8}, index=0, thread=s0+4*d0, replicate=4
  // -----------------------------------------------------------------------
  ReduceCase case1 = {
      "TC 8x8 base fragment, reduce dim=1 (row-wise)",
      /*srcShape=*/{8, 8},
      /*dstShape=*/{8},
      /*reduceDim=*/1,
      /*reduceKind=*/"max",
      /*srcIndexMap=*/mkMap(2, 0, d(1) % c(2)),
      /*srcThreadMap=*/mkMap(2, 0, d(1).floorDiv(c(2)) + c(4) * d(0)),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(1, 0, c(0)),
      /*expectedThreadMap=*/mkMap(1, 1, s(0) + c(4) * d(0)),
      /*expectedReplicate=*/4,
      /*threads=*/32,
  };

  // -----------------------------------------------------------------------
  // Case 2: TC 8x8 base fragment, reduce dim=0 (column reduction)
  //
  // Same source as case 1, but reducing rows instead of columns.
  // i contributes 4*i to thread (8 values from 8 rows).
  //
  // Expected: dst={8}, index=d0%2, thread=d0/2+4*s0, replicate=8
  // -----------------------------------------------------------------------
  ReduceCase case2 = {
      "TC 8x8 base fragment, reduce dim=0 (column-wise)",
      /*srcShape=*/{8, 8},
      /*dstShape=*/{8},
      /*reduceDim=*/0,
      /*reduceKind=*/"add",
      /*srcIndexMap=*/mkMap(2, 0, d(1) % c(2)),
      /*srcThreadMap=*/mkMap(2, 0, d(1).floorDiv(c(2)) + c(4) * d(0)),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(1, 0, d(0) % c(2)),
      /*expectedThreadMap=*/mkMap(1, 1, d(0).floorDiv(c(2)) + c(4) * s(0)),
      /*expectedReplicate=*/8,
      /*threads=*/32,
  };

  // -----------------------------------------------------------------------
  // Case 3: Ampere GEMM C 16x8 fragment, reduce dim=1 (softmax max)
  //
  // makeFragment8x8.repeat({2,1}, repeatOnThread=false):
  //   shape={16,8}, index = j%2 + 2*(i/8), thread = j/2 + 4*(i%8)
  // This is the single-warp Ampere MMA C accumulator shape.
  // Each thread holds 4 elements (2 per 8-row group × 2 from j%2).
  //
  // Expected: dst={16}, index=(d0/8)%2, thread=s0+4*(d0%8), replicate=4
  // -----------------------------------------------------------------------
  ReduceCase case3 = {
      "Ampere 16x8 fragment (1 warp), reduce dim=1",
      /*srcShape=*/{16, 8},
      /*dstShape=*/{16},
      /*reduceDim=*/1,
      /*reduceKind=*/"max",
      /*srcIndexMap=*/mkMap(2, 0, d(1) % c(2) + c(2) * d(0).floorDiv(c(8))),
      /*srcThreadMap=*/mkMap(2, 0, d(1).floorDiv(c(2)) + c(4) * (d(0) % c(8))),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(1, 0, (d(0).floorDiv(c(8))) % c(2)),
      /*expectedThreadMap=*/mkMap(1, 1, s(0) + c(4) * (d(0) % c(8))),
      /*expectedReplicate=*/4,
      /*threads=*/32,
  };

  // -----------------------------------------------------------------------
  // Case 4: 2-warp Ampere C fragment 32x8, reduce dim=1
  //
  // base 16x8 (case 3) + repeat({2,1}, repeatOnThread=true, lowerDimFirst=false)
  //   shape={32,8},
  //   index = j%2 + 2*((i%16)/8)
  //   thread = j/2 + 4*(i%8) + 32*(i/16)
  //
  // 64 threads (2 warps). j contributes j/2 → 4 thread values.
  // Expected: dst={32}, replicate=4, index=(d0/8)%2
  //            thread=s0+4*(d0%8)+32*(d0/16)
  // -----------------------------------------------------------------------
  AffineExpr case4Thread =
      d(1).floorDiv(c(2)) + c(4) * (d(0) % c(8)) +
      c(32) * d(0).floorDiv(c(16));
  AffineExpr case4Index = d(1) % c(2) + c(2) * ((d(0) % c(16)).floorDiv(c(8)));

  ReduceCase case4 = {
      "2-warp 32x8 fragment, reduce dim=1",
      /*srcShape=*/{32, 8},
      /*dstShape=*/{32},
      /*reduceDim=*/1,
      /*reduceKind=*/"add",
      /*srcIndexMap=*/mkMap(2, 0, case4Index),
      /*srcThreadMap=*/mkMap(2, 0, case4Thread),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(1, 0, (d(0).floorDiv(c(8))) % c(2)),
      /*expectedThreadMap=*/mkMap(1, 1, s(0) + c(4) * (d(0) % c(8)) +
                                            c(32) * d(0).floorDiv(c(16))),
      /*expectedReplicate=*/4,
      /*threads=*/64,
  };

  // -----------------------------------------------------------------------
  // Case 5: Source with replicate > 1 (fragment already replicated)
  //
  // Same 8x8 tensor-core base as case 1, but source replicate = 2.
  // This models a buffer that was already replicated across warps
  // (e.g., B fragment with .replicate(warpRepeatM)).
  //
  // finalReplicate = srcReplicate(2) * condensedReplicate(4) = 8
  // 此case的srcMap计算存在问题，单看对replicate的压缩效果
  // -----------------------------------------------------------------------
  ReduceCase case5 = {
      "TC 8x8 with srcReplicate=2, reduce dim=1",
      /*srcShape=*/{8, 8},
      /*dstShape=*/{8},
      /*reduceDim=*/1,
      /*reduceKind=*/"max",
      /*srcIndexMap=*/mkMap(2, 0, d(1) % c(2)),
      /*srcThreadMap=*/mkMap(2, 0, d(1).floorDiv(c(2)) + c(4) * d(0)),
      /*srcReplicate=*/2,
      /*expectedIndexMap=*/mkMap(1, 0, c(0)),
      /*expectedThreadMap=*/mkMap(1, 1, s(0) + c(4) * d(0)),
      /*expectedReplicate=*/8,
      /*threads=*/32,
  };

  // -----------------------------------------------------------------------
  // Case 6: Thread map does not use the reduced dimension
  //
  // src={16, 8}, thread = (i%8)*2 + i/8 (only uses dim 0), index = i*8+j
  // Reducing dim=1 collapses a dimension the thread map doesn't reference.
  // Thread map fully covers dim 0 (splits {1,8} and {8,2} = extent 16).
  // No cross-thread reduction needed, replicate stays 1.
  //
  // Expected: index=0 (thread fully covers d0), replicate = 1
  // -----------------------------------------------------------------------
  ReduceCase case6 = {
      "Reduce dim unused by thread map (no cross-thread reduce)",
      /*srcShape=*/{16, 8},
      /*dstShape=*/{16},
      /*reduceDim=*/1,
      /*reduceKind=*/"add",
      // /*srcIndexMap=*/mkMap(2, 0, d(0) * c(8) + d(1)),
      /*srcIndexMap=*/mkMap(2, 0, d(1)),
      /*srcThreadMap=*/mkMap(2, 0, (d(0) % c(8)) * c(2) + d(0).floorDiv(c(8))),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(1, 0, c(0)),
      /*expectedThreadMap=*/mkMap(1, 0, (d(0) % c(8)) * c(2) +
                                            d(0).floorDiv(c(8))),
      /*expectedReplicate=*/1,
      /*threads=*/16,
  };

  // -----------------------------------------------------------------------
  // Case 7: 3D batch + reduce middle dim (e.g., flash-attn split-K)
  //
  // src={4, 8, 16}, thread = d2 (thread-per-column), index = d0*128+d1*16+d2
  // Reducing dim=1 removes the middle dimension.
  // Thread doesn't use d0 or d1, so d0 becomes fully unused in output.
  //
  // Expected: dst={4, 16}, replicate=1, index=d0 mod 4
  //   (d1 in dst = d2 from src, fully used by thread → no index contribution)
  // -----------------------------------------------------------------------
  ReduceCase case7 = {
      "3D batch reduce middle dim (split-K pattern)",
      /*srcShape=*/{4, 8, 16},
      /*dstShape=*/{4, 16},
      /*reduceDim=*/1,
      /*reduceKind=*/"max",
      // /*srcIndexMap=*/mkMap(3, 0, d(0) * c(128) + d(1) * c(16) + d(2)),
      /*srcIndexMap=*/mkMap(3, 0, d(1) * c(4) + d(1)),
      /*srcThreadMap=*/mkMap(3, 0, d(2)),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(2, 0, d(0) % c(4)),
      /*expectedThreadMap=*/mkMap(2, 0, d(1)),
      /*expectedReplicate=*/1,
      /*threads=*/16,
  };

  // -----------------------------------------------------------------------
  // Case 8: 3D batch, reduce last dim, thread uses reduced dim
  //
  // src={4, 8, 16}, thread = d1*2 + d2/8 (uses both d1 and d2)
  // reduce dim=2, dst={4, 8}
  // d2 contributes d2/8 = 2 thread values (out of 16 positions in dim 2)
  // Thread doesn't use d0 → d0 becomes unused index contribution.
  //
  // Expected: replicate = 2, index = d0 mod 4
  // -----------------------------------------------------------------------
  ReduceCase case8 = {
      "3D reduce last dim, thread partially uses reduced dim",
      /*srcShape=*/{4, 8, 16},
      /*dstShape=*/{4, 8},
      /*reduceDim=*/2,
      /*reduceKind=*/"add",
      // /*srcIndexMap=*/mkMap(3, 0, d(0) * c(128) + d(1) * c(16) + d(2)),
      /*srcIndexMap=*/mkMap(3, 0, d(0) * c(8) + d(2) % 8),
      /*srcThreadMap=*/mkMap(3, 0, d(1) * c(2) + d(2).floorDiv(c(8))),
      /*srcReplicate=*/1,
      /*expectedIndexMap=*/mkMap(2, 0, d(0) % c(4)),
      /*expectedThreadMap=*/mkMap(2, 1, d(1) * c(2) + s(0)),
      /*expectedReplicate=*/2,
      /*threads=*/16,
  };

  // -----------------------------------------------------------------------
  // Run all parameterised cases
  // -----------------------------------------------------------------------

  ReduceCase cases[] = {case1, case2, case3, case4,
                        case5, case6, case7, case8};

  bool success = true;
  for (const ReduceCase &tc : cases)
    success &= runReduceCase(context, tc);

  // -----------------------------------------------------------------------
  // Integration tests: GEMM C → reduce (end-to-end)
  // -----------------------------------------------------------------------

  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  llvm::outs() << "-- Integration Tests: GEMM C -> Reduce --\n";
  llvm::outs() << std::string(60, '-') << "\n";

  // Ampere 128x128, reduce columns (softmax pattern), 128 threads.
  success &= runGemmReduceIntegration(context, "sm_80",
                                      /*blockM=*/128, /*blockN=*/128,
                                      /*blockK=*/32, /*threads=*/128,
                                      /*reduceDim=*/1);

  // Ampere 64x64, reduce rows, 128 threads.
  success &= runGemmReduceIntegration(context, "sm_80",
                                      /*blockM=*/64, /*blockN=*/64,
                                      /*blockK=*/32, /*threads=*/128,
                                      /*reduceDim=*/0);

  // Hopper 128x128, reduce columns, 128 threads.
  success &= runGemmReduceIntegration(context, "sm_90",
                                      /*blockM=*/128, /*blockN=*/128,
                                      /*blockK=*/64, /*threads=*/128,
                                      /*reduceDim=*/1);

  // Ampere 128x64, reduce columns, 256 threads.
  success &= runGemmReduceIntegration(context, "sm_80",
                                      /*blockM=*/128, /*blockN=*/64,
                                      /*blockK=*/32, /*threads=*/256,
                                      /*reduceDim=*/1);

  // -----------------------------------------------------------------------
  // Summary
  // -----------------------------------------------------------------------

  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  if (success) {
    llvm::outs() << "All reduce layout inference tests passed.\n";
    return 0;
  }

  llvm::errs() << "Some reduce layout inference tests FAILED.\n";
  return 1;
}
