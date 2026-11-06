#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/IR/LegacyImportSemantics.h"

#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {

class LegacyImportSemanticsTest : public testing::Test {
protected:
  MLIRContext context;
  OpBuilder builder{&context};

  LegacyImportSemanticsTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect>();
  }

  MemRefType matrix(int64_t rows, int64_t cols, Type elementType) {
    return MemRefType::get({rows, cols}, elementType);
  }

  std::optional<std::string> validate(int64_t ar, int64_t ac, int64_t br,
                                      int64_t bc, int64_t cr, int64_t cc,
                                      bool transA, bool transB, int64_t m,
                                      int64_t n, int64_t k,
                                      Type input = Type(), Type accum = Type()) {
    if (!input)
      input = builder.getF16Type();
    if (!accum)
      accum = input;
    return getLegacyGemmValidationError(
        matrix(ar, ac, input), matrix(br, bc, input), matrix(cr, cc, accum),
        transA, transB, m, n, k);
  }
};

TEST_F(LegacyImportSemanticsTest, AcceptsAllEffectiveTransposeShapes) {
  EXPECT_FALSE(validate(6, 4, 4, 5, 6, 5, false, false, 6, 5, 4));
  EXPECT_FALSE(validate(4, 6, 4, 5, 6, 5, true, false, 6, 5, 4));
  EXPECT_FALSE(validate(6, 4, 5, 4, 6, 5, false, true, 6, 5, 4));
  EXPECT_FALSE(validate(4, 6, 5, 4, 6, 5, true, true, 6, 5, 4));
}

TEST_F(LegacyImportSemanticsTest, AcceptsLowPrecisionInputsAndF32Accumulator) {
  EXPECT_FALSE(validate(6, 4, 4, 5, 6, 5, false, false, 6, 5, 4,
                        builder.getF16Type(), builder.getF32Type()));
  EXPECT_FALSE(validate(6, 4, 4, 5, 6, 5, false, false, 6, 5, 4,
                        builder.getBF16Type(), builder.getF32Type()));
}

TEST_F(LegacyImportSemanticsTest, RejectsRankDynamicShapeTypeAndMNKMismatch) {
  Type f16 = builder.getF16Type();
  EXPECT_TRUE(getLegacyGemmValidationError(
      MemRefType::get({6, 4, 1}, f16), matrix(4, 5, f16), matrix(6, 5, f16),
      false, false, 6, 5, 4));
  EXPECT_TRUE(getLegacyGemmValidationError(
      matrix(ShapedType::kDynamic, 4, f16), matrix(4, 5, f16),
      matrix(6, 5, f16), false, false, 6, 5, 4));
  EXPECT_TRUE(validate(6, 3, 4, 5, 6, 5, false, false, 6, 5, 4));
  EXPECT_TRUE(validate(6, 4, 4, 5, 7, 5, false, false, 6, 5, 4));
  EXPECT_TRUE(validate(6, 4, 4, 5, 6, 5, false, false, 7, 5, 4));
  EXPECT_TRUE(getLegacyGemmValidationError(
      matrix(6, 4, f16), matrix(4, 5, builder.getBF16Type()),
      matrix(6, 5, f16), false, false, 6, 5, 4));
  EXPECT_TRUE(validate(6, 4, 4, 5, 6, 5, false, false, 6, 5, 4,
                       f16, builder.getBF16Type()));
}

TEST_F(LegacyImportSemanticsTest, MarkerUsesNearestDeclaration) {
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  module->setAttr("frisk.legacy_semantics", builder.getStringAttr("tensor_v1"));
  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<func::FuncOp>(loc, "f", builder.getFunctionType({}, {}));
  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto value = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  EXPECT_TRUE(succeeded(verifyLegacyTensorV1Semantics(value)));

  ScopedDiagnosticHandler silence(&context,
      [](Diagnostic &) { return success(); });
  fn->setAttr("frisk.legacy_semantics", builder.getStringAttr("unknown"));
  EXPECT_TRUE(failed(verifyLegacyTensorV1Semantics(value)));
  fn->setAttr("frisk.legacy_semantics", builder.getI64IntegerAttr(1));
  EXPECT_TRUE(failed(verifyLegacyTensorV1Semantics(value)));
  fn->removeAttr("frisk.legacy_semantics");
  value->setAttr("frisk.legacy_semantics", builder.getStringAttr("bad"));
  EXPECT_TRUE(failed(verifyLegacyTensorV1Semantics(value)));
}

TEST_F(LegacyImportSemanticsTest, MarkerMissingHasStableDiagnostic) {
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<func::FuncOp>(loc, "f", builder.getFunctionType({}, {}));
  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto value = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    llvm::raw_string_ostream os(diagnostic);
    diag.print(os);
    return success();
  });
  EXPECT_TRUE(failed(verifyLegacyTensorV1Semantics(value)));
  EXPECT_NE(diagnostic.find(
                "legacy-math-contract: explicit tensor_v1 semantics required"),
            std::string::npos);
}

static unsigned countEffect(Operation *op, Value value,
                            const SideEffects::Effect *wanted) {
  auto interface = cast<MemoryEffectOpInterface>(op);
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::count_if(effects, [&](const auto &effect) {
    return effect.getValue() == value && effect.getEffect() == wanted;
  });
}

TEST_F(LegacyImportSemanticsTest, DestinationReadEffectsFollowClearFlags) {
  Location loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module.getBody());
  Type f16 = builder.getF16Type();
  auto aTy = matrix(6, 4, f16), bTy = matrix(4, 5, f16), cTy = matrix(6, 5, f16);
  auto fn = builder.create<func::FuncOp>(
      loc, "f", builder.getFunctionType({aTy, bTy, cTy}, {}));
  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto gemm = builder.create<GemmOp>(loc, entry->getArgument(0),
      entry->getArgument(1), entry->getArgument(2), false, false, 6, 5, 4,
      attr::GemmWarpPolicy::Square, false);
  EXPECT_TRUE(succeeded(gemm.verify()))
      << "legacy IR remains valid without an import marker";
  EXPECT_EQ(countEffect(gemm, gemm.getA(), MemoryEffects::Read::get()), 1u);
  EXPECT_EQ(countEffect(gemm, gemm.getB(), MemoryEffects::Read::get()), 1u);
  EXPECT_EQ(countEffect(gemm, gemm.getC(), MemoryEffects::Write::get()), 1u);
  EXPECT_EQ(countEffect(gemm, gemm.getC(), MemoryEffects::Read::get()), 1u);
  gemm.setClearAccum(true);
  EXPECT_EQ(countEffect(gemm, gemm.getC(), MemoryEffects::Read::get()), 0u);

  auto reduce = builder.create<ReduceOp>(loc, entry->getArgument(0),
                                         entry->getArgument(2), "add",
                                         int64_t{1}, false);
  EXPECT_EQ(countEffect(reduce, reduce.getSrc(), MemoryEffects::Read::get()), 1u);
  EXPECT_EQ(countEffect(reduce, reduce.getDst(), MemoryEffects::Write::get()), 1u);
  EXPECT_EQ(countEffect(reduce, reduce.getDst(), MemoryEffects::Read::get()), 1u);
  reduce.setClear(true);
  EXPECT_EQ(countEffect(reduce, reduce.getDst(), MemoryEffects::Read::get()), 0u);
}

} // namespace
