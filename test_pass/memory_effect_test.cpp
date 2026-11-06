#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {

template <typename EffectTy>
bool recursivelyHasEffect(Operation *op) {
  auto effects = getEffectsRecursively(op);
  return effects && llvm::any_of(*effects, [](const auto &effect) {
           return isa<EffectTy>(effect.getEffect());
         });
}

bool check(bool condition, StringRef message) {
  if (condition)
    return true;
  llvm::errs() << message << "\n";
  return false;
}

} // namespace

int main() {
  MLIRContext context;
  context.getOrLoadDialect<FriskDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  ModuleOp module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());

  Type f16 = builder.getF16Type();
  Attribute localSpace = builder.getI64IntegerAttr(
      static_cast<int64_t>(attr::MemorySpace::Local));
  auto matrixType =
      MemRefType::get({16, 16}, f16, AffineMap(), localSpace);
  auto vectorType = MemRefType::get({16}, f16, AffineMap(), localSpace);
  auto functionType = builder.getFunctionType(
      {matrixType, matrixType, matrixType, vectorType}, {});

  auto kernel = builder.create<KernelOp>(loc, "memory_effects", functionType);
  Block *entry = kernel.addEntryBlock();
  builder.setInsertionPoint(entry->getTerminator());

  Value a = entry->getArgument(0);
  Value b = entry->getArgument(1);
  Value c = entry->getArgument(2);
  Value reduced = entry->getArgument(3);

  auto alloc = builder.create<AllocBufferOp>(
      loc, matrixType, ArrayRef<int64_t>{16, 16}, f16, 0,
      attr::MemorySpace::Local);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices = {zero, zero};
  auto copy = builder.create<CopyOp>(loc, a, c, indices, indices);
  auto fill = builder.create<FillOp>(loc, c, builder.getF16FloatAttr(0.0));
  auto gemm = builder.create<GemmOp>(
      loc, a, b, c, false, false, static_cast<uint64_t>(16),
      static_cast<uint64_t>(16), static_cast<uint64_t>(16),
      attr::GemmWarpPolicy::Square, false);
  auto reduce = builder.create<ReduceOp>(loc, c, reduced, "add",
                                          static_cast<int64_t>(1), false);

  auto parallel =
      builder.create<ParallelOp>(loc, ArrayRef<int64_t>{1}, 32);
  Block *parallelEntry = parallel.addEntryBlock();
  OpBuilder nestedBuilder(&context);
  nestedBuilder.setInsertionPoint(parallelEntry->getTerminator());

  BlockOp block;
  ForOp forOp;
  block = nestedBuilder.create<BlockOp>(
      loc, ArrayRef<int64_t>{1}, [&](ValueRange) {
        forOp = nestedBuilder.create<ForOp>(
            loc, 0, 1, 1, [&](Value) {
              nestedBuilder.create<FillOp>(loc, c,
                                            nestedBuilder.getF16FloatAttr(1.0));
            });
      });

  bool ok = true;
  ok &= check(!isMemoryEffectFree(copy), "copy must not be memory-effect free");
  ok &= check(hasEffect<MemoryEffects::Read>(copy, a),
              "copy source must have a read effect");
  ok &= check(hasEffect<MemoryEffects::Write>(copy, c),
              "copy destination must have a write effect");

  ok &= check(!isMemoryEffectFree(fill), "fill must not be memory-effect free");
  ok &= check(hasEffect<MemoryEffects::Write>(fill, c),
              "fill destination must have a write effect");

  ok &= check(!isMemoryEffectFree(gemm), "gemm must not be memory-effect free");
  ok &= check(hasEffect<MemoryEffects::Read>(gemm, a),
              "gemm A must have a read effect");
  ok &= check(hasEffect<MemoryEffects::Read>(gemm, b),
              "gemm B must have a read effect");
  ok &= check(hasEffect<MemoryEffects::Read>(gemm, c),
              "gemm C must have a read effect");
  ok &= check(hasEffect<MemoryEffects::Write>(gemm, c),
              "gemm C must have a write effect");

  ok &= check(!isMemoryEffectFree(reduce),
              "reduce must not be memory-effect free");
  ok &= check(hasEffect<MemoryEffects::Read>(reduce, c),
              "reduce source must have a read effect");
  ok &= check(hasEffect<MemoryEffects::Write>(reduce, reduced),
              "reduce destination must have a write effect");

  ok &= check(hasEffect<MemoryEffects::Allocate>(alloc, alloc.getResult()),
              "alloc result must have an allocate effect");

  ok &= check(recursivelyHasEffect<MemoryEffects::Read>(kernel),
              "kernel must recursively expose reads");
  ok &= check(recursivelyHasEffect<MemoryEffects::Write>(kernel),
              "kernel must recursively expose writes");
  ok &= check(recursivelyHasEffect<MemoryEffects::Allocate>(kernel),
              "kernel must recursively expose allocations");
  ok &= check(recursivelyHasEffect<MemoryEffects::Write>(parallel),
              "parallel must recursively expose nested writes");
  ok &= check(recursivelyHasEffect<MemoryEffects::Write>(block),
              "block must recursively expose nested writes");
  ok &= check(recursivelyHasEffect<MemoryEffects::Write>(forOp),
              "for must recursively expose nested writes");

  return ok ? 0 : 1;
}
