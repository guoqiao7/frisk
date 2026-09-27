#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class LegacyNormalizationTest : public testing::Test {
protected:
  MLIRContext context;
  LegacyNormalizationTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect,
                        arith::ArithDialect, scf::SCFDialect>();
    registerFriskPasses();
  }
  OwningOpRef<ModuleOp> parse(StringRef text) {
    return parseSourceString<ModuleOp>(text, &context);
  }
  LogicalResult normalize(ModuleOp module) {
    PassManager pm(&context);
    if (failed(parsePassPipeline("frisk-normalize-layout-ir", pm)))
      return failure();
    return pm.run(module);
  }
  std::string print(ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os);
    return text;
  }
  template <typename Op> SmallVector<Op> ops(ModuleOp module) {
    SmallVector<Op> result;
    module.walk([&](Op op) { result.push_back(op); });
    return result;
  }
  void rejects(ModuleOp module, StringRef expected) {
    std::string before = print(module), diagnostic;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
      llvm::raw_string_ostream os(diagnostic); diag.print(os); return success();
    });
    EXPECT_TRUE(failed(normalize(module)));
    EXPECT_EQ(before, print(module));
    EXPECT_NE(diagnostic.find(expected.str()), std::string::npos) << diagnostic;
  }
  CopyOp copy(OpBuilder &builder, Value src, Value dst) {
    return builder.create<CopyOp>(src.getLoc(), src, dst,
      builder.getEmptyAffineMap(), builder.getEmptyAffineMap(), ValueRange{}, ValueRange{});
  }
  OwningOpRef<ModuleOp> gemm(bool clear, bool initialize = false, bool rs = false) {
    auto module = parse(R"mlir(module attributes {frisk.target="sm_90a", frisk.legacy_semantics="tensor_v1"} {
      func.func @f(%a:memref<64x16xf16,3>, %b:memref<16x64xf16,3>) {
        %c = memref.alloc() : memref<64x64xf32>
        return
      }
    })mlir");
    if (!module) return {};
    auto fn = *module->getOps<func::FuncOp>().begin();
    OpBuilder builder(fn.getBody().front().getTerminator());
    Value a = fn.getArgument(0), b = fn.getArgument(1);
    auto c = ops<memref::AllocOp>(*module)[0];
    if (initialize) builder.create<FillOp>(fn.getLoc(), c, builder.getF32FloatAttr(-2.0));
    if (rs) {
      auto local = builder.create<memref::AllocOp>(fn.getLoc(), MemRefType::get({64,16}, builder.getF16Type()));
      copy(builder, a, local);
      a = local;
    }
    builder.create<GemmOp>(fn.getLoc(), a, b, c, false, false, 64, 64, 16,
                          attr::GemmWarpPolicy::Square, clear);
    return module;
  }
};

TEST_F(LegacyNormalizationTest, FillPreservesNegativeZeroAndIsIdempotent) {
  auto module = parse(R"mlir(module { func.func @f() {
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value = -0.0 : f32} : memref<4xf32>
    memref.dealloc %x : memref<4xf32>
    return
  } })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  EXPECT_TRUE(ops<memref::AllocOp>(*module).empty());
  EXPECT_TRUE(ops<memref::DeallocOp>(*module).empty());
  auto constants = ops<arith::ConstantOp>(*module);
  ASSERT_EQ(constants.size(), 1u);
  auto values = cast<DenseFPElementsAttr>(constants[0].getValue());
  EXPECT_TRUE(values.getSplatValue<APFloat>().isNegZero());
  auto once = print(*module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  EXPECT_EQ(once, print(*module));
}

TEST_F(LegacyNormalizationTest, IfAppendsTensorAndRequiresBothPathsInitialized) {
  auto module = parse(R"mlir(func.func @f(%c:i1) {
    %x = memref.alloc() : memref<4xf32>
    scf.if %c {
      frisk.fill %x {value=2.0:f32} : memref<4xf32>
    } else {
      frisk.fill %x {value=3.0:f32} : memref<4xf32>
    }
    return
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  auto branches = ops<scf::IfOp>(*module);
  ASSERT_EQ(branches.size(), 1u);
  EXPECT_EQ(branches[0].getNumResults(), 1u);
  EXPECT_EQ(branches[0].thenYield().getNumOperands(), 1u);
  EXPECT_EQ(branches[0].elseYield().getNumOperands(), 1u);
  auto missing = parse(R"mlir(func.func @f(%c:i1) {
    %x = memref.alloc() : memref<4xf32>
    scf.if %c { frisk.fill %x {value=2.0:f32} : memref<4xf32> }
    return
  })mlir");
  ASSERT_TRUE(missing);
  rejects(*missing, "legacy-fragment-uninitialized");
}

TEST_F(LegacyNormalizationTest, ForAppendsCarriedStateAfterExistingTensor) {
  auto module = parse(R"mlir(func.func @f(%n:index) {
    %c0 = arith.constant 0:index
    %c1 = arith.constant 1:index
    %t = arith.constant dense<1.0> : tensor<4xf32>
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value=2.0:f32} : memref<4xf32>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%v = %t) -> (tensor<4xf32>) {
      frisk.fill %x {value=3.0:f32} : memref<4xf32>
      scf.yield %v : tensor<4xf32>
    }
    return
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  auto loops = ops<scf::ForOp>(*module);
  ASSERT_EQ(loops.size(), 1u);
  EXPECT_EQ(loops[0].getNumResults(), 2u);
  EXPECT_EQ(loops[0].getRegionIterArgs().size(), 2u);
}

TEST_F(LegacyNormalizationTest, WhilePreservesUnequalTuplesAndBeforeWrites) {
  auto module = parse(R"mlir(func.func @f() {
    %c = arith.constant false
    %c0 = arith.constant 0:index
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value=2.0:f32} : memref<4xf32>
    %r:2 = scf.while (%i = %c0) : (index) -> (index, i1) {
      frisk.fill %x {value=3.0:f32} : memref<4xf32>
      scf.condition(%c) %i, %c : index, i1
    } do {
    ^bb0(%j:index, %b:i1):
      frisk.fill %x {value=4.0:f32} : memref<4xf32>
      scf.yield %j : index
    }
    return
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  auto loops = ops<scf::WhileOp>(*module);
  ASSERT_EQ(loops.size(), 1u);
  EXPECT_EQ(loops[0].getInits().size(), 2u);
  EXPECT_EQ(loops[0].getNumResults(), 3u);
  EXPECT_EQ(loops[0].getBeforeArguments().size(), 2u);
  EXPECT_EQ(loops[0].getAfterArguments().size(), 3u);
  auto beforeValue = loops[0].getConditionOp().getArgs().back();
  EXPECT_EQ(cast<DenseFPElementsAttr>(beforeValue.getDefiningOp<arith::ConstantOp>().getValue())
                .getSplatValue<APFloat>().convertToFloat(), 3.0f);
}

TEST_F(LegacyNormalizationTest, UnknownUseRejectedTransactionally) {
  auto module = parse(R"mlir(func.func @f() {
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value=2.0:f32} : memref<4xf32>
    %i = arith.constant 0:index
    %v = memref.load %x[%i] : memref<4xf32>
    return
  })mlir");
  ASSERT_TRUE(module);
  rejects(*module, "legacy local fragment escapes supported normalization boundary");
}

TEST_F(LegacyNormalizationTest, WholeCopyAliasFanoutPreservesOrderAndSnapshots) {
  auto module = parse(R"mlir(func.func @f(%src:memref<4xf32,3>, %dst:memref<4xf32,1>) {
    %x = memref.alloc() : memref<4xf32>
    %y = memref.alloca() : memref<4xf32>
    %alias = memref.cast %x : memref<4xf32> to memref<4xf32>
    return
  })mlir");
  ASSERT_TRUE(module);
  auto fn = *module->getOps<func::FuncOp>().begin();
  auto x = ops<memref::AllocOp>(*module)[0];
  auto y = ops<memref::AllocaOp>(*module)[0];
  auto alias = ops<memref::CastOp>(*module)[0];
  OpBuilder builder(fn.getBody().front().getTerminator());
  copy(builder, fn.getArgument(0), alias);
  copy(builder, x, y);
  builder.create<FillOp>(fn.getLoc(), alias, builder.getF32FloatAttr(7.0));
  copy(builder, y, fn.getArgument(1));
  copy(builder, x, fn.getArgument(1));
  builder.create<memref::DeallocOp>(fn.getLoc(), x);
  ASSERT_TRUE(succeeded(normalize(*module)));
  ASSERT_TRUE(ops<CopyOp>(*module).empty());
  ASSERT_TRUE(ops<memref::CastOp>(*module).empty());
  auto loads = ops<TileLoadOp>(*module);
  auto stores = ops<TileStoreOp>(*module);
  ASSERT_EQ(loads.size(), 1u);
  ASSERT_EQ(stores.size(), 2u);
  EXPECT_EQ(stores[0].getValue(), loads[0].getResult());
  EXPECT_TRUE(stores[1].getValue().getDefiningOp<arith::ConstantOp>());
  EXPECT_TRUE(loads[0]->isBeforeInBlock(stores[0]));
}

TEST_F(LegacyNormalizationTest, GemmClearPositiveZeroAndAccumulationDependence) {
  for (bool rs : {false, true}) for (bool clear : {false, true}) {
    auto module = gemm(clear, true, rs);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(normalize(*module)));
    auto mma = ops<MmaOp>(*module);
    ASSERT_EQ(mma.size(), 1u);
    auto init = mma[0].getInit().getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(init);
    APFloat value = cast<DenseFPElementsAttr>(init.getValue()).getSplatValue<APFloat>();
    if (clear) EXPECT_TRUE(value.isZero() && !value.isNegative());
    else EXPECT_EQ(value.convertToFloat(), -2.0f);
    EXPECT_EQ(isa<RankedTensorType>(mma[0].getA().getType()), rs);
    EXPECT_TRUE(mma[0].getB().getDefiningOp<LayoutViewOp>());
  }
  auto uninitialized = gemm(false);
  rejects(*uninitialized, "legacy-fragment-uninitialized");
  auto clearUninitialized = gemm(true);
  EXPECT_TRUE(succeeded(normalize(*clearUninitialized)));
}

TEST_F(LegacyNormalizationTest, MathMarkerTargetDtypeAndThreadContracts) {
  auto missing = gemm(true);
  missing->getOperation()->removeAttr("frisk.legacy_semantics");
  rejects(*missing, "legacy-math-contract: explicit tensor_v1 semantics required");
  auto badNearest = gemm(true);
  OpBuilder builder(&context);
  ops<GemmOp>(*badNearest)[0]->setAttr("frisk.legacy_semantics", builder.getI64IntegerAttr(1));
  rejects(*badNearest, "legacy-math-contract");
  auto target = gemm(true);
  target->getOperation()->removeAttr("frisk.target");
  rejects(*target, "explicit sm_90a target required");
  auto dtype = gemm(true);
  ops<memref::AllocOp>(*dtype)[0].getResult().setType(MemRefType::get({64,64}, builder.getF16Type()));
  rejects(*dtype, "legacy-gemm-accumulator: f32 required; implicit promotion is forbidden");
  auto threads = gemm(true);
  auto old = ops<GemmOp>(*threads)[0];
  old->setAttr("frisk.threads", builder.getI64IntegerAttr(128));
  old->setAttr("frisk.execution_threads", builder.getI64IntegerAttr(256));
  rejects(*threads, "legacy-thread-contract");
  auto alias = gemm(true);
  ops<GemmOp>(*alias)[0]->setAttr("frisk.threads", builder.getI64IntegerAttr(128));
  ASSERT_TRUE(succeeded(normalize(*alias)));
  auto mma = ops<MmaOp>(*alias)[0];
  EXPECT_FALSE(mma->hasAttr("frisk.threads"));
  EXPECT_EQ(mma->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
}

TEST_F(LegacyNormalizationTest, ReduceMergeKindAndOldDestinationOnLeft) {
  for (StringRef kind : {"add", "max", "min"}) for (bool clear : {false, true}) {
    auto module = parse(R"mlir(module attributes {frisk.legacy_semantics="tensor_v1"} {
      func.func @f() {
        %src = memref.alloc() : memref<4x8xf32>
        %dst = memref.alloc() : memref<4xf32>
        frisk.fill %src {value=2.0:f32} : memref<4x8xf32>
        frisk.fill %dst {value=-0.0:f32} : memref<4xf32>
        return
      }
    })mlir");
    ASSERT_TRUE(module);
    auto fn = *module->getOps<func::FuncOp>().begin();
    auto allocations = ops<memref::AllocOp>(*module);
    OpBuilder builder(fn.getBody().front().getTerminator());
    builder.create<ReduceOp>(fn.getLoc(), allocations[0], allocations[1], kind, int64_t{1}, clear);
    ASSERT_TRUE(succeeded(normalize(*module)));
    auto reduced = ops<ReduceTensorOp>(*module)[0];
    EXPECT_EQ(reduced.getKind(), kind == "add" ? "sum" : kind);
    if (clear) EXPECT_TRUE(reduced.getResult().use_empty());
    else {
      ASSERT_TRUE(reduced.getResult().hasOneUse());
      auto merge = *reduced.getResult().getUsers().begin();
      EXPECT_EQ(merge->getName().getStringRef(), kind == "add" ? "arith.addf" : kind == "max" ? "arith.maximumf" : "arith.minimumf");
      EXPECT_EQ(merge->getOperand(1), reduced.getResult());
      auto prior = merge->getOperand(0).getDefiningOp<arith::ConstantOp>();
      ASSERT_TRUE(prior);
      EXPECT_TRUE(cast<DenseFPElementsAttr>(prior.getValue()).getSplatValue<APFloat>().isNegZero());
    }
  }
}

TEST_F(LegacyNormalizationTest, RejectsLifetimeAndAttributeBoundaries) {
  for (StringRef body : {
    R"mlir(%x = memref.alloc() : memref<4xf32>
           memref.dealloc %x : memref<4xf32>
           frisk.fill %x {value=1.0:f32} : memref<4xf32>)mlir",
    R"mlir(%x = memref.alloc() : memref<4xf32>
           %s = memref.subview %x[0] [2] [1] : memref<4xf32> to memref<2xf32>)mlir",
    R"mlir(%x = memref.alloc() : memref<4xf32>
           %t = frisk.tile_load %x : memref<4xf32> -> tensor<4xf32>)mlir",
    R"mlir(%x = memref.alloc() : memref<3xf32>)mlir"}) {
    auto module = parse(("func.func @f() {" + body + " return }").str());
    ASSERT_TRUE(module);
    rejects(*module, "legacy local fragment escapes supported normalization boundary");
  }
  auto annotated = parse(R"mlir(func.func @f() {
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value=1.0:f32, frisk.vector_bytes=4:i64} : memref<4xf32>
    return
  })mlir");
  ASSERT_TRUE(annotated);
  rejects(*annotated, "legacy-attribute-contract");
  auto parameter = parse(R"mlir(func.func @f(%x:memref<4xf32>) {
    frisk.fill %x {value=1.0:f32} : memref<4xf32>
    return
  })mlir");
  ASSERT_TRUE(parameter);
  rejects(*parameter, "legacy local fragment escapes supported normalization boundary");
}

TEST_F(LegacyNormalizationTest, LoopEntryMustBeInitializedEvenIfBodyOverwrites) {
  auto module = parse(R"mlir(func.func @f() {
    %c0 = arith.constant 0:index
    %c1 = arith.constant 1:index
    %x = memref.alloc() : memref<4xf32>
    scf.for %i = %c0 to %c1 step %c1 {
      frisk.fill %x {value=3.0:f32} : memref<4xf32>
    }
    return
  })mlir");
  ASSERT_TRUE(module);
  rejects(*module, "legacy-fragment-uninitialized");
}

TEST_F(LegacyNormalizationTest, IfWithoutElseForwardsInitializedEntry) {
  auto module = parse(R"mlir(func.func @f(%c:i1) {
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value=1.0:f32} : memref<4xf32>
    scf.if %c { frisk.fill %x {value=2.0:f32} : memref<4xf32> }
    return
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  auto branch = ops<scf::IfOp>(*module)[0];
  ASSERT_EQ(branch.getNumResults(), 1u);
  auto prior = branch.elseYield().getOperand(0).getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(prior);
  EXPECT_EQ(cast<DenseFPElementsAttr>(prior.getValue()).getSplatValue<APFloat>().convertToFloat(), 1.0f);
}

TEST_F(LegacyNormalizationTest, LegacyForAndIterationLocalLifetimes) {
  auto module = parse(R"mlir(func.func @f() {
    %x = memref.alloc() : memref<4xf32>
    frisk.fill %x {value=1.0:f32} : memref<4xf32>
    "frisk.for"() <{lower=0:i64, upper=4:i64, step=1:i64}> ({
    ^bb0(%iv:index):
      %y = memref.alloc() : memref<4xf32>
      frisk.fill %y {value=2.0:f32} : memref<4xf32>
      frisk.fill %x {value=3.0:f32} : memref<4xf32>
      "frisk.end"() : () -> ()
    }) : () -> ()
    return
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  ASSERT_TRUE(ops<ForOp>(*module).empty());
  auto loops = ops<scf::ForOp>(*module);
  ASSERT_EQ(loops.size(), 1u);
  EXPECT_EQ(loops[0].getNumResults(), 1u);
  EXPECT_EQ(loops[0].getBody()->getNumArguments(), 2u);
  EXPECT_EQ(ops<arith::ConstantOp>(*module).size(), 6u);
}

TEST_F(LegacyNormalizationTest, ParallelLocalAcceptedAndExternalCaptureRejected) {
  auto good = parse(R"mlir("frisk.kernel"() <{sym_name="k", function_type=()->()}> ({
    "frisk.parallel"() <{ranges=array<i64:4>, threads=128:i64}> ({
    ^bb0(%i:index):
      %x = memref.alloc() : memref<4xf32>
      frisk.fill %x {value=2.0:f32} : memref<4xf32>
      "frisk.end"() : () -> ()
    }) : () -> ()
    "frisk.end"() : () -> ()
  }) : () -> ())mlir");
  ASSERT_TRUE(good);
  ASSERT_TRUE(succeeded(normalize(*good)));
  EXPECT_TRUE(ops<memref::AllocOp>(*good).empty());
  EXPECT_EQ(ops<arith::ConstantOp>(*good).size(), 1u);
  auto bad = parse(R"mlir("frisk.kernel"() <{sym_name="k", function_type=()->()}> ({
    %x = memref.alloc() : memref<4xf32>
    "frisk.parallel"() <{ranges=array<i64:4>, threads=128:i64}> ({
    ^bb0(%i:index):
      frisk.fill %x {value=2.0:f32} : memref<4xf32>
      "frisk.end"() : () -> ()
    }) : () -> ()
    "frisk.end"() : () -> ()
  }) : () -> ())mlir");
  ASSERT_TRUE(bad);
  rejects(*bad, "Local root crosses a Parallel boundary");
}

TEST_F(LegacyNormalizationTest, RejectsNestedLegacyLayoutsAndUnknownSemanticAttributes) {
  auto legacy = parse(R"mlir(module attributes {
    test.debug_contract = [{nested = #frisk.layout<[4], affine_map<(d)->(d)>>}]
  } {})mlir");
  ASSERT_TRUE(legacy);
  rejects(*legacy, "legacy-layout-attribute");
  auto unknown = parse(R"mlir(module attributes {frisk.unrecognized_contract = 1:i64} {})mlir");
  ASSERT_TRUE(unknown);
  rejects(*unknown, "legacy-attribute-contract");
}

TEST_F(LegacyNormalizationTest, CompleteRejectedRootAndEscapeBoundaries) {
  for (StringRef body : {
    R"mlir(func.func @f(%n:index) { %x=memref.alloc(%n):memref<?xf32> return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<1x4xf32> return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<131072xf32> return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<4xi32> return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<4xf32, strided<[2]>> return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<4xf32>
      %alias=memref.cast %x:memref<4xf32> to memref<?xf32> return })mlir",
    R"mlir(func.func @f() -> memref<4xf32> { %x=memref.alloc():memref<4xf32>
      return %x:memref<4xf32> })mlir",
    R"mlir(func.func private @sink(memref<4xf32>)
      func.func @f() { %x=memref.alloc():memref<4xf32>
      func.call @sink(%x):(memref<4xf32>)->() return })mlir",
    R"mlir(func.func @f(%c:i1) { %x=memref.alloc():memref<4xf32>
      scf.if %c { memref.dealloc %x:memref<4xf32> } return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<4xf32>
      %c0=arith.constant 0:index
      %c1=arith.constant 1:index
      %r=scf.for %i=%c0 to %c1 step %c1 iter_args(%p=%x)->(memref<4xf32>) {
        scf.yield %p:memref<4xf32>
      } return })mlir",
    R"mlir(memref.global "private" @g : memref<4xf32> = uninitialized
      func.func @f() { %x=memref.get_global @g:memref<4xf32>
      frisk.fill %x {value=1.0:f32}:memref<4xf32> return })mlir",
    R"mlir(func.func @f() { %x=memref.alloc():memref<4xf32>
      %z=arith.constant 0:index
      %v=arith.constant 1.0:f32
      memref.store %v,%x[%z]:memref<4xf32> return })mlir"
  }) {
    SCOPED_TRACE(body.str());
    auto module = parse(body);
    ASSERT_TRUE(module);
    rejects(*module, "legacy local fragment escapes supported normalization boundary");
  }
}

TEST_F(LegacyNormalizationTest, SelfCopyMustReadInitializedValueAndRejectExplicitContract) {
  for (bool initialized : {false, true}) {
    auto module = parse(R"mlir(func.func @f() {
      %x=memref.alloc():memref<4xf32>
      return
    })mlir");
    auto fn = *module->getOps<func::FuncOp>().begin();
    auto x = ops<memref::AllocOp>(*module)[0];
    OpBuilder builder(fn.getBody().front().getTerminator());
    if (initialized) builder.create<FillOp>(fn.getLoc(), x, builder.getF32FloatAttr(1.0));
    copy(builder, x, x);
    if (initialized) EXPECT_TRUE(succeeded(normalize(*module)));
    else rejects(*module, "legacy-fragment-uninitialized");
  }
  auto module = parse(R"mlir(func.func @f(%src:memref<4xf32,3>) {
    %x=memref.alloc():memref<4xf32>
    return
  })mlir");
  auto fn = *module->getOps<func::FuncOp>().begin();
  OpBuilder builder(fn.getBody().front().getTerminator());
  auto op = copy(builder, fn.getArgument(0), ops<memref::AllocOp>(*module)[0]);
  op->setAttr("frisk.writer_policy", builder.getStringAttr("first_owner"));
  rejects(*module, "legacy-attribute-contract");
}

TEST_F(LegacyNormalizationTest, ReduceRejectsMulAndMissingInitAndDeadMissingMarker) {
  for (unsigned test = 0; test < 3; ++test) {
    auto module = parse(R"mlir(module attributes {frisk.legacy_semantics="tensor_v1"} {
      func.func @f() {
        %src=memref.alloc():memref<4x8xf32>
        %dst=memref.alloc():memref<4xf32>
        frisk.fill %src {value=2.0:f32}:memref<4x8xf32>
        return
      }
    })mlir");
    auto fn = *module->getOps<func::FuncOp>().begin();
    auto allocations = ops<memref::AllocOp>(*module);
    OpBuilder builder(fn.getBody().front().getTerminator());
    builder.create<ReduceOp>(fn.getLoc(), allocations[0], allocations[1],
                            test == 0 ? "mul" : "add", int64_t{1}, test != 1);
    if (test == 2) module->getOperation()->removeAttr("frisk.legacy_semantics");
    rejects(*module, test == 0 ? "legacy-reduce-kind" : test == 1
        ? "legacy-fragment-uninitialized" : "legacy-math-contract");
  }
}

TEST_F(LegacyNormalizationTest, BothInputDtypesAndEffectiveTransposeShapes) {
  for (StringRef dtype : {"f16", "bf16"})
    for (bool ta : {false, true}) for (bool tb : {false, true}) {
      std::string a = (ta ? "16x64x" : "64x16x") + dtype.str();
      std::string b = (tb ? "64x16x" : "16x64x") + dtype.str();
      std::string text;
      llvm::raw_string_ostream os(text);
      os << "module attributes {frisk.target=\"sm_90a\",frisk.legacy_semantics=\"tensor_v1\"} {"
         << "func.func @f(%a:memref<" << a << ",3>,%b:memref<" << b << ",3>) {"
         << "%c=memref.alloc():memref<64x64xf32> "
         << "frisk.gemm (%a,%b,%c) {M=64:i64,N=64:i64,K=16:i64,clear_accum=true,"
         << "transA=" << (ta ? "true" : "false") << ",transB=" << (tb ? "true" : "false")
         << ",policy=#frisk<gemm_warp_policy FullCol>} : memref<" << a << ",3>,memref<" << b
         << ",3>,memref<64x64xf32> return } }";
      auto module = parse(text);
      ASSERT_TRUE(module);
      ASSERT_TRUE(succeeded(normalize(*module)));
      auto mma = ops<MmaOp>(*module)[0];
      EXPECT_EQ(mma.getTransA(), ta);
      EXPECT_EQ(mma.getTransB(), tb);
      EXPECT_EQ(mma.getPolicy(), attr::GemmWarpPolicy::FullCol);
    }
}

TEST_F(LegacyNormalizationTest, LoopReadOnlyCaptureAndAccumulatorRecurrence) {
  auto module = parse(R"mlir(module attributes {frisk.legacy_semantics="tensor_v1"} {
    func.func @f(%n:index, %out:memref<4xf32,1>) {
      %src=memref.alloc():memref<4x8xf32>
      %dst=memref.alloc():memref<4xf32>
      frisk.fill %src {value=2.0:f32}:memref<4x8xf32>
      frisk.fill %dst {value=1.0:f32}:memref<4xf32>
      %c0=arith.constant 0:index
      %c1=arith.constant 1:index
      scf.for %i=%c0 to %n step %c1 {
        frisk.reduce %src,%dst {kind="add",dim=1:i64,clear=false}:memref<4x8xf32>,memref<4xf32>
      }
      return
    }
  })mlir");
  ASSERT_TRUE(module);
  auto fn = *module->getOps<func::FuncOp>().begin();
  OpBuilder builder(fn.getBody().front().getTerminator());
  copy(builder, ops<memref::AllocOp>(*module)[1], fn.getArgument(1));
  ASSERT_TRUE(succeeded(normalize(*module)));
  auto loop = ops<scf::ForOp>(*module)[0];
  ASSERT_EQ(loop.getNumResults(), 1u); // Read-only source must not get a slot.
  auto merge = ops<arith::AddFOp>(*module)[0];
  EXPECT_EQ(merge.getLhs(), loop.getRegionIterArgs()[0]);
  EXPECT_EQ(cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(0), merge.getResult());
  EXPECT_EQ(ops<TileStoreOp>(*module)[0].getValue(), loop.getResult(0));
  EXPECT_TRUE(loop.getInitArgs()[0].getDefiningOp<arith::ConstantOp>());
}

TEST_F(LegacyNormalizationTest, IfPreservesOriginalSlotsAndStableRootOrder) {
  auto module = parse(R"mlir(func.func @f(%c:i1) -> i32 {
    %x=memref.alloc():memref<4xf32>
    %y=memref.alloc():memref<4xf32>
    frisk.fill %x {value=1.0:f32}:memref<4xf32>
    frisk.fill %y {value=2.0:f32}:memref<4xf32>
    %a=arith.constant 11:i32
    %b=arith.constant 12:i32
    %r=scf.if %c -> (i32) {
      frisk.fill %y {value=4.0:f32}:memref<4xf32>
      frisk.fill %x {value=3.0:f32}:memref<4xf32>
      scf.yield %a:i32
    } else {
      scf.yield %b:i32
    }
    return %r:i32
  })mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(normalize(*module)));
  auto branch = ops<scf::IfOp>(*module)[0];
  ASSERT_EQ(branch.getNumResults(), 3u);
  EXPECT_TRUE(branch.getResult(0).getType().isInteger(32));
  auto splat = [](Value value) {
    return cast<DenseFPElementsAttr>(value.getDefiningOp<arith::ConstantOp>().getValue())
        .getSplatValue<APFloat>().convertToFloat();
  };
  EXPECT_EQ(splat(branch.thenYield().getOperand(1)), 3.0f);
  EXPECT_EQ(splat(branch.thenYield().getOperand(2)), 4.0f);
  EXPECT_EQ(splat(branch.elseYield().getOperand(1)), 1.0f);
  EXPECT_EQ(splat(branch.elseYield().getOperand(2)), 2.0f);
  EXPECT_EQ(ops<func::ReturnOp>(*module)[0].getOperand(0), branch.getResult(0));
}
} // namespace
