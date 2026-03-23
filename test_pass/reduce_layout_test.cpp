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

void printHeader(StringRef title) {
  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  llvm::outs() << "-- " << title << " --\n";
  llvm::outs() << std::string(60, '-') << "\n";
}

MemRefType makeLocalType(OpBuilder &builder, ArrayRef<int64_t> shape, Type elemTy) {
  return MemRefType::get(shape, elemTy, AffineMapAttr(),
                         builder.getI64IntegerAttr(static_cast<int64_t>(
                             mlir::frisk::attr::MemorySpace::Local)));
}

mlir::frisk::LayoutAttr buildLayoutAttr(OpBuilder &builder, MemRefType type,
                                        AffineMap indexMap, AffineMap threadMap,
                                        int64_t replicate) {
  auto shapeAttr = builder.getDenseI64ArrayAttr(type.getShape());
  auto replicateAttr = builder.getI64IntegerAttr(replicate);
  return mlir::frisk::LayoutAttr::get(builder.getContext(), shapeAttr,
                                      AffineMapAttr::get(indexMap),
                                      AffineMapAttr::get(threadMap),
                                      replicateAttr);
}

bool mapsEqualOnDomain(AffineMap lhs, AffineMap rhs, ArrayRef<int64_t> dimExtents,
                       ArrayRef<int64_t> symbolExtents) {
  if (lhs.getNumDims() != rhs.getNumDims() ||
      lhs.getNumSymbols() != rhs.getNumSymbols() ||
      lhs.getNumResults() != rhs.getNumResults())
    return false;
  if (lhs.getNumDims() != dimExtents.size() ||
      lhs.getNumSymbols() != symbolExtents.size())
    return false;

  Builder builder(lhs.getContext());
  SmallVector<int64_t> values(lhs.getNumDims() + lhs.getNumSymbols(), 0);
  SmallVector<Attribute> attrs(values.size());
  SmallVector<Attribute> lhsResults;
  SmallVector<Attribute> rhsResults;

  std::function<bool(unsigned)> enumerate = [&](unsigned index) -> bool {
    if (index == values.size()) {
      for (size_t i = 0; i < values.size(); ++i)
        attrs[i] = builder.getI64IntegerAttr(values[i]);
      lhsResults.clear();
      rhsResults.clear();
      if (failed(lhs.constantFold(attrs, lhsResults)) ||
          failed(rhs.constantFold(attrs, rhsResults)))
        return false;
      return lhsResults == rhsResults;
    }

    int64_t extent =
        index < dimExtents.size() ? dimExtents[index]
                                  : symbolExtents[index - dimExtents.size()];
    for (int64_t value = 0; value < extent; ++value) {
      values[index] = value;
      if (!enumerate(index + 1))
        return false;
    }
    return true;
  };

  return enumerate(0);
}

struct ReduceCase {
  const char *name;
  AffineMap srcIndexMap;
  AffineMap srcThreadMap;
  AffineMap expectedIndexMap;
  AffineMap expectedThreadMap;
  int64_t expectedReplicate;
  int64_t threads;
};

bool runReduceCase(MLIRContext &context, const ReduceCase &testCase) {
  printHeader(testCase.name);

  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  auto f16 = builder.getF16Type();
  auto srcType = makeLocalType(builder, {4, 8, 16}, f16);
  auto dstType = makeLocalType(builder, {4, 16}, f16);
  auto funcType = builder.getFunctionType({srcType, dstType}, {});

  auto kernel =
      builder.create<mlir::frisk::KernelOp>(loc, testCase.name, funcType);
  Block *entry = kernel.addEntryBlock();

  OpBuilder bodyBuilder(&context);
  bodyBuilder.setInsertionPointToStart(entry);
  auto reduce = bodyBuilder.create<mlir::frisk::ReduceOp>(
      loc, entry->getArgument(0), entry->getArgument(1), "add",
      /*dim=*/static_cast<int64_t>(1), /*clear=*/false);
  reduce->setAttr("frisk.threads",
                  bodyBuilder.getI64IntegerAttr(testCase.threads));
  bodyBuilder.create<mlir::frisk::EndOp>(loc);

  DenseMap<Value, Attribute> layoutMap;
  OpBuilder inferBuilder(&context);
  inferBuilder.setInsertionPoint(reduce);
  layoutMap.try_emplace(
      reduce.getSrc(),
      buildLayoutAttr(inferBuilder, srcType, testCase.srcIndexMap,
                      testCase.srcThreadMap, /*replicate=*/1));

  if (failed(reduce.inferLayout(inferBuilder, layoutMap))) {
    llvm::errs() << "inferLayout failed for case '" << testCase.name << "'\n";
    return false;
  }

  auto it = layoutMap.find(reduce.getDst());
  if (it == layoutMap.end()) {
    llvm::errs() << "missing inferred destination layout for case '"
                 << testCase.name << "'\n";
    return false;
  }

  auto dstLayout = dyn_cast<mlir::frisk::LayoutAttr>(it->second);
  if (!dstLayout) {
    llvm::errs() << "destination layout has wrong type for case '"
                 << testCase.name << "'\n";
    return false;
  }

  std::optional<int64_t> replicateValue;
  if (auto attr = dstLayout.getReplicateSize())
    replicateValue = attr.getInt();
  if (!replicateValue || *replicateValue != testCase.expectedReplicate) {
    llvm::errs() << "unexpected replicate extent for case '" << testCase.name
                 << "', want " << testCase.expectedReplicate << " got ";
    if (replicateValue)
      llvm::errs() << *replicateValue;
    else
      llvm::errs() << "<none>";
    llvm::errs() << "\n";
    return false;
  }

  auto actualIndex = dstLayout.getForwardIndex();
  auto actualThread = dstLayout.getForwardThread();
  if (!actualIndex ||
      !mapsEqualOnDomain(actualIndex.getValue(), testCase.expectedIndexMap,
                         /*dimExtents=*/{4, 16}, /*symbolExtents=*/{})) {
    llvm::errs() << "unexpected forward index for case '" << testCase.name
                 << "'\nwant: " << AffineMapAttr::get(testCase.expectedIndexMap)
                 << "\ngot:  " << actualIndex << "\n";
    return false;
  }
  if (!actualThread ||
      !mapsEqualOnDomain(actualThread.getValue(), testCase.expectedThreadMap,
                         /*dimExtents=*/{4, 16},
                         actualThread.getValue().getNumSymbols()
                             ? ArrayRef<int64_t>{testCase.expectedReplicate}
                             : ArrayRef<int64_t>{})) {
    llvm::errs() << "unexpected forward thread for case '" << testCase.name
                 << "'\nwant: " << AffineMapAttr::get(testCase.expectedThreadMap)
                 << "\ngot:  " << actualThread << "\n";
    return false;
  }

  llvm::outs() << mlir::frisk::layoutDebugString(dstLayout) << "\n";
  return true;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<mlir::frisk::FriskDialect>();
  MLIRContext context(registry);
  context.loadDialect<mlir::frisk::FriskDialect>();

  OpBuilder builder(&context);
  MLIRContext *ctx = &context;

  AffineExpr d0 = builder.getAffineDimExpr(0);
  AffineExpr d1 = builder.getAffineDimExpr(1);
  AffineExpr d2 = builder.getAffineDimExpr(2);
  AffineExpr s0 = builder.getAffineSymbolExpr(0);

  AffineMap srcIndexMap =
      AffineMap::get(3, 0, d0 * builder.getAffineConstantExpr(128) +
                               d1 * builder.getAffineConstantExpr(16) + d2,
                     ctx);
  AffineMap expectedIndexMap = AffineMap::get(
      2, 0,
      (d0 % builder.getAffineConstantExpr(4)) *
              builder.getAffineConstantExpr(16) +
          (d1 % builder.getAffineConstantExpr(16)),
      ctx);

  ReduceCase cases[] = {
      {
          "Reduce Layout: reduced dim unused by thread map",
          srcIndexMap,
          AffineMap::get(3, 0, d2, ctx),
          AffineMap::get(2, 0, d0 % builder.getAffineConstantExpr(4), ctx),
          AffineMap::get(2, 0, d1, ctx),
          /*expectedReplicate=*/1,
          /*threads=*/64,
      },
      {
          "Reduce Layout: reduced dim directly drives thread map",
          srcIndexMap,
          AffineMap::get(3, 0, d1, ctx),
          expectedIndexMap,
          AffineMap::get(2, 1, s0, ctx),
          /*expectedReplicate=*/8,
          /*threads=*/64,
      },
      {
          "Reduce Layout: reduced dim split by floordiv in thread map",
          srcIndexMap,
          AffineMap::get(3, 0, d1.floorDiv(builder.getAffineConstantExpr(2)),
                         ctx),
          expectedIndexMap,
          AffineMap::get(2, 1, s0, ctx),
          /*expectedReplicate=*/4,
          /*threads=*/64,
      },
  };

  bool success = true;
  for (const ReduceCase &testCase : cases)
    success &= runReduceCase(context, testCase);

  llvm::outs() << "\n" << std::string(60, '=') << "\n";
  if (success) {
    llvm::outs() << "Reduce layout inference checks passed.\n";
    return 0;
  }

  llvm::errs() << "Reduce layout inference checks failed.\n";
  return 1;
}
