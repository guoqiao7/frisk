#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class LayoutPipelineTest : public testing::Test {
protected:
  MLIRContext context;
  LayoutPipelineTest() {
    context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect, scf::SCFDialect>();
  }
  std::string print(Operation *op) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    op->print(stream);
    return text;
  }
  LogicalResult run(ModuleOp module, SolverOptions options = {}) {
    PassManager pm(&context);
    buildFriskLayoutPipeline(pm, options);
    return pm.run(module);
  }
  void expectLegacyMmaGraph(LayoutConstraintGraph &graph, LayoutTarget &target,
                           unsigned storage, unsigned distributed,
                           uint64_t dataCombinationBound, DictionaryAttr &geometry) {
    unsigned storageCount = 0, distributedCount = 0, instructionCount = 0;
    for (const auto &var : graph.getVariables()) {
      storageCount += var.kind == LayoutKind::Storage;
      distributedCount += var.kind == LayoutKind::Distributed;
      instructionCount += var.kind == LayoutKind::Instruction;
    }
    EXPECT_EQ(storageCount, storage); EXPECT_EQ(distributedCount, distributed);
    ASSERT_EQ(instructionCount, 1u);
    EXPECT_EQ(graph.getVariables().size(), storage + distributed + 1u);
    const LayoutConstraint *contract = nullptr;
    for (const auto &c : graph.getConstraints()) if (c.instruction) {
      ASSERT_FALSE(contract); contract = &c;
    }
    ASSERT_TRUE(contract); ASSERT_EQ(contract->vars.size(), 5u);
    auto mma = dyn_cast_or_null<MmaOp>(contract->instruction->source); ASSERT_TRUE(mma);
    EXPECT_EQ(mma.getPolicy(), attr::GemmWarpPolicy::Square);
    auto &role = graph.getVariable(contract->vars[4]);
    EXPECT_EQ(role.kind, LayoutKind::Instruction); EXPECT_EQ(role.anchor, mma.getOperation());
    EXPECT_TRUE(role.instructionRole); EXPECT_FALSE(role.value); EXPECT_FALSE(role.use);
    EXPECT_FALSE(role.functionResult); EXPECT_EQ(role.requiredThreads, 128);
    EXPECT_EQ(role.shapedType, mma.getResult().getType());
    auto raw = getSM90MmaGeometries(mma, 128); ASSERT_TRUE(succeeded(raw));
    ASSERT_EQ(raw->size(), 1u); EXPECT_EQ(raw->front().gM, 1); EXPECT_EQ(raw->front().gN, 1);
    geometry = encodeSM90MmaGeometry(&context, raw->front());
    auto binding = mma->getAttr("frisk.mma_contract");
    EXPECT_EQ(contract->instruction->binding, binding);
    // RelationsOnly leaves geometry unseeded. Prepare proposals, not support
    // tuples, to establish its actual singleton without rerunning tuple proofs.
    ASSERT_TRUE(contract->instruction->tuples.empty());
    ASSERT_TRUE(succeeded(target.prepareInstructionCandidates(graph)));
    ASSERT_EQ(role.candidates.size(), 1u); EXPECT_EQ(role.candidates.front().value, geometry);
    EXPECT_TRUE(contract->instruction->tuples.empty());
    EXPECT_EQ(mma->getAttr("frisk.mma_contract"), binding);
    uint64_t maximumCombinations = role.candidates.size();
    for (const auto &var : graph.getVariables()) if (var.kind != LayoutKind::Instruction) {
      EXPECT_LE(var.candidates.size(), 4u) << var.stableName;
      maximumCombinations *= 4;
    }
    // The Instruction singleton contributes factor one: retain each fixture's
    // original worst-case four-candidate DATA-domain product, not a larger cap.
    EXPECT_EQ(maximumCombinations, dataCombinationBound);
    if (binding) {
      auto selected = getSM90MmaContractGeometry(mma,128,binding); ASSERT_TRUE(succeeded(selected));
      EXPECT_EQ(encodeSM90MmaGeometry(&context,*selected),geometry);
    }
  }
  OwningOpRef<ModuleOp> legacyMma(bool rs, bool reduction) {
    std::string outputType = reduction ? "memref<64xf32,1>" : "memref<64x64xf32,1>";
    auto module = parseSourceString<ModuleOp>(
        "module attributes {frisk.target=\"sm_90a\",frisk.legacy_semantics=\"tensor_v1\"} { "
        "func.func @f(%out:" + outputType + ") { return } }", &context);
    if (!module) return {};
    auto fn = *module->getOps<func::FuncOp>().begin();
    OpBuilder b(fn.getBody().front().getTerminator());
    auto loc = fn.getLoc();
    auto a = b.create<memref::AllocOp>(loc,
        MemRefType::get({64,16}, b.getF16Type(), {}, rs ? 0 : 3));
    if (!rs) a->setAttr("alignment", b.getI64IntegerAttr(1024));
    else b.create<FillOp>(loc, a, b.getFloatAttr(b.getF16Type(), 1.0));
    auto operandB = b.create<memref::AllocOp>(loc,
        MemRefType::get({16,64}, b.getF16Type(), {}, 3));
    operandB->setAttr("alignment", b.getI64IntegerAttr(1024));
    auto c = b.create<memref::AllocOp>(loc, MemRefType::get({64,64}, b.getF32Type()));
    b.create<GemmOp>(loc, a, operandB, c, false, false, 64, 64, 16,
                     attr::GemmWarpPolicy::Square, true);
    Value result = c;
    if (reduction) {
      auto reduced = b.create<memref::AllocOp>(loc, MemRefType::get({64}, b.getF32Type()));
      b.create<ReduceOp>(loc, c, reduced, "add", int64_t{1}, true);
      result = reduced;
    }
    b.create<CopyOp>(loc, result, fn.getArgument(0), ValueRange{}, ValueRange{});
    return module;
  }
};
constexpr StringLiteral durable = R"mlir(
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
module {
func.func @f(%r:memref<4xi32,3>, %x:i32) {
  %z = arith.constant 0:index
  %rv = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  memref.store %x, %v[%z] : memref<2xi32,strided<[1],offset:1>,3>
  return
}
})mlir";

TEST_F(LayoutPipelineTest, DurableContractSurvivesOptimizationAndReplay) {
  auto module = parseSourceString<ModuleOp>(durable, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(run(*module)));
  auto first = print(*module);
  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_EQ(first, print(*module));
  auto reparsed = parseSourceString<ModuleOp>(first, &context);
  ASSERT_TRUE(reparsed);
  PassManager canonicalize(&context);
  canonicalize.addPass(createCanonicalizerPass());
  ASSERT_TRUE(succeeded(canonicalize.run(*reparsed)));
  auto target = createSM90LayoutTarget();
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*reparsed, *target)));
  unsigned count = 0;
  reparsed->walk([&](StorageContractOp) { ++count; });
  EXPECT_EQ(count, 1u);
}

TEST_F(LayoutPipelineTest, LaterInferenceFailureRollsBackPreservation) {
  auto module = parseSourceString<ModuleOp>(durable, &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<func::FuncOp>().begin();
  function->setAttr("frisk.target", StringAttr::get(&context, "sm_80"));
  auto original = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(run(*module)));
  EXPECT_EQ(original, print(*module));
}

TEST_F(LayoutPipelineTest, PreservesIdentityCoordinatesAcrossDifferentDescriptors) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[4],
outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
func.func @f(%r:memref<4xi32,3>) {
  %s = memref.subview %r[0] [4] [1] : memref<4xi32,3> to memref<4xi32,strided<[1]>,3>
  %v = frisk.layout_view %s {layout=#root} : memref<4xi32,strided<[1]>,3> -> memref<4xi32,strided<[1]>,3>
  return
})mlir", &context);
  ASSERT_TRUE(module);
  PassManager normalize(&context);
  normalize.addPass(createNormalizeLayoutIRPass());
  ASSERT_TRUE(succeeded(normalize.run(*module)));
  unsigned count = 0;
  module->walk([&](StorageContractOp) { ++count; });
  EXPECT_EQ(count, 1u);
}

TEST_F(LayoutPipelineTest, PreservesSupportedStridedEntryRoot) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[4],
outputs=["byte_offset","bit_offset"],output_extents=[32,8],map=affine_map<(d)->(8*d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
func.func @f(%r:memref<4xi32,strided<[2]>,3>) {
  %v = frisk.layout_view %r {layout=#root} : memref<4xi32,strided<[2]>,3> -> memref<4xi32,strided<[2]>,3>
  return
})mlir", &context);
  ASSERT_TRUE(module);
  PassManager normalize(&context);
  normalize.addPass(createNormalizeLayoutIRPass());
  ASSERT_TRUE(succeeded(normalize.run(*module)));
  unsigned count = 0;
  module->walk([&](StorageContractOp) { ++count; });
  EXPECT_EQ(count, 1u);
}

TEST_F(LayoutPipelineTest, SeparateNormalizeCommitIsNotRolledBackByLaterPass) {
  auto module = parseSourceString<ModuleOp>(durable, &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<func::FuncOp>().begin();
  function->setAttr("frisk.target", StringAttr::get(&context, "sm_80"));
  PassManager normalize(&context);
  normalize.addPass(createNormalizeLayoutIRPass());
  ASSERT_TRUE(succeeded(normalize.run(*module)));
  auto normalized = print(*module);
  EXPECT_NE(normalized.find("frisk.storage_contract"), std::string::npos);
  PassManager infer(&context);
  infer.addPass(createFriskInferLayoutsPass());
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(infer.run(*module)));
  EXPECT_EQ(normalized, print(*module));
}

TEST_F(LayoutPipelineTest, LegacySSAndRSMmaReachExternalStoreWithinBudget) {
  for (bool rs : {false, true}) {
    SCOPED_TRACE(rs ? "rs" : "ss");
    auto module = legacyMma(rs, false);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(run(*module)));
    unsigned mma = 0, stores = 0;
    module->walk([&](MmaOp) { ++mma; });
    module->walk([&](TileStoreOp) { ++stores; });
    EXPECT_EQ(mma, 1u);
    EXPECT_EQ(stores, 1u);
    auto target = createSM90LayoutTarget();
    auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
    ASSERT_TRUE(succeeded(graph));
    DictionaryAttr geometry;
    expectLegacyMmaGraph(*graph,*target,rs ? 2u : 3u,rs ? 6u : 4u,
                         rs ? 65536u : 16384u,geometry);
    EXPECT_LE(graph->getVariables().size() - 1u, 8u); // Data-variable budget is unchanged.
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  }
}

TEST_F(LayoutPipelineTest, NineVariableFullChainMaterializesAndReplays) {
  auto module = legacyMma(false, true);
  ASSERT_TRUE(module);
  OwningOpRef<ModuleOp> normalized(cast<ModuleOp>((*module)->clone()));
  PassManager normalize(&context);
  normalize.addPass(createNormalizeLayoutIRPass());
  ASSERT_TRUE(succeeded(normalize.run(*normalized)));
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*normalized, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  DictionaryAttr geometry;
  expectLegacyMmaGraph(*graph,*target,3u,6u,262144u,geometry);
  ASSERT_TRUE(geometry); // Nine data roles, plus one singleton geometry role.
  ASSERT_TRUE(succeeded(run(*module)));
  unsigned mma = 0, reductions = 0, stores = 0;
  module->walk([&](MmaOp op) {
    ++mma;
    auto binding = op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
    ASSERT_TRUE(binding);
    for (StringRef field : {"grid","atom","repeats"})
      EXPECT_EQ(binding.getPayload().get(field),geometry.get(field));
  });
  module->walk([&](ReduceTensorOp op) {
    ++reductions;
    EXPECT_TRUE(op->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](TileStoreOp op) {
    ++stores;
    auto view = op->getOperand(1).getDefiningOp<LayoutViewOp>();
    ASSERT_TRUE(view);
    EXPECT_EQ(view.getSource(),
              (*module->getOps<func::FuncOp>().begin()).getArgument(0));
  });
  EXPECT_EQ(mma, 1u);
  EXPECT_EQ(reductions, 1u);
  EXPECT_EQ(stores, 1u);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  auto once = print(*module);
  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_EQ(once, print(*module));
  auto reparsed = parseSourceString<ModuleOp>(once, &context);
  ASSERT_TRUE(reparsed);
  ASSERT_TRUE(succeeded(run(*reparsed)));
  EXPECT_EQ(once, print(*reparsed));
}

TEST_F(LayoutPipelineTest, ExplicitSearchBudgetRollsBackFullChainNormalization) {
  auto module = legacyMma(false, true);
  ASSERT_TRUE(module);
  auto original = print(*module);
  SolverOptions options;
  options.maxExpandedStates = 1;
  std::string diagnostic;
  ScopedDiagnosticHandler handler(&context, [&](Diagnostic &diag) {
    llvm::raw_string_ostream os(diagnostic); diag.print(os); return success();
  });
  EXPECT_TRUE(failed(run(*module, options)));
  EXPECT_NE(diagnostic.find("search-budget-exceeded"), std::string::npos) << diagnostic;
  EXPECT_EQ(original, print(*module));
}

TEST_F(LayoutPipelineTest, GlobalCopyParallelMmaReduceStoreMaterializesAndReplays) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
module attributes {frisk.target="sm_90a"} {
  "frisk.kernel"() <{sym_name="full_chain",
      function_type=(memref<64x16xf16,1>,memref<16x64xf16,1>,memref<64xf32,1>)->()}> ({
  ^bb0(%globalA:memref<64x16xf16,1>, %globalB:memref<16x64xf16,1>, %out:memref<64xf32,1>):
    %a = memref.alloc() {alignment=1024:i64} : memref<64x16xf16,3>
    %b = memref.alloc() {alignment=1024:i64} : memref<16x64xf16,3>
    "frisk.parallel"() <{ranges=array<i64:1>,threads=128:i64}> ({
    ^bb0(%i:index):
      %ga = frisk.layout_view %globalA : memref<64x16xf16,1> -> memref<64x16xf16,1>
      %gb = frisk.layout_view %globalB : memref<16x64xf16,1> -> memref<16x64xf16,1>
      %av = frisk.layout_view %a : memref<64x16xf16,3> -> memref<64x16xf16,3>
      %bv = frisk.layout_view %b : memref<16x64xf16,3> -> memref<16x64xf16,3>
      frisk.copy %ga[], %av[] {srcExtents=array<i64:64,16>,dstExtents=array<i64:64,16>}
        : memref<64x16xf16,1>, memref<64x16xf16,3>
      frisk.copy %gb[], %bv[] {srcExtents=array<i64:16,64>,dstExtents=array<i64:16,64>}
        : memref<16x64xf16,1>, memref<16x64xf16,3>
      %zero = arith.constant dense<0.0> : tensor<64x64xf32>
      %c = "frisk.mma"(%av,%bv,%zero) {m=64:i64,n=64:i64,k=16:i64}
        : (memref<64x16xf16,3>,memref<16x64xf16,3>,tensor<64x64xf32>) -> tensor<64x64xf32>
      %r = frisk.reduce_tensor %c {kind="sum",dim=1:i64} : tensor<64x64xf32> -> tensor<64xf32>
      %ov = frisk.layout_view %out : memref<64xf32,1> -> memref<64xf32,1>
      frisk.tile_store %r, %ov : tensor<64xf32>, memref<64xf32,1>
      frisk.end
    }) : () -> ()
    frisk.end
  }) : () -> ()
})mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(run(*module)));
  unsigned copies = 0, parallel = 0, mma = 0, reductions = 0, stores = 0;
  module->walk([&](CopyOp op) {
    ++copies;
    EXPECT_TRUE(op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](ParallelOp op) {
    ++parallel;
    EXPECT_EQ(op.getThreads(), 128);
  });
  module->walk([&](MmaOp op) {
    ++mma;
    EXPECT_TRUE(op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract"));
  });
  module->walk([&](ReduceTensorOp op) {
    ++reductions;
    EXPECT_TRUE(op->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](TileStoreOp op) {
    ++stores;
    auto view = op->getOperand(1).getDefiningOp<LayoutViewOp>();
    ASSERT_TRUE(view);
    auto output = dyn_cast<BlockArgument>(view.getSource());
    ASSERT_TRUE(output);
    EXPECT_EQ(output.getArgNumber(), 2u);
    EXPECT_TRUE(isa<KernelOp>(output.getOwner()->getParentOp()));
  });
  EXPECT_EQ(copies, 2u);
  EXPECT_EQ(parallel, 1u);
  EXPECT_EQ(mma, 1u);
  EXPECT_EQ(reductions, 1u);
  EXPECT_EQ(stores, 1u);
  auto target = createSM90LayoutTarget();
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  auto once = print(*module);
  ASSERT_TRUE(succeeded(run(*module)));
  EXPECT_EQ(once, print(*module));
  auto reparsed = parseSourceString<ModuleOp>(once, &context);
  ASSERT_TRUE(reparsed);
  ASSERT_TRUE(succeeded(run(*reparsed)));
  EXPECT_EQ(once, print(*reparsed));
}

TEST_F(LayoutPipelineTest, LegacyFillReduceStoreHasObservableOutput) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
module attributes {frisk.legacy_semantics="tensor_v1"} {
func.func @f(%out:memref<4xf32,1>) {
  %src = memref.alloc() : memref<4x4xf32>
  %dst = memref.alloc() : memref<4xf32>
  frisk.fill %src {value=2.0:f32} : memref<4x4xf32>
  frisk.reduce %src, %dst {kind="add",dim=1:i64,clear=true} : memref<4x4xf32>,memref<4xf32>
  frisk.copy %dst[], %out[] {srcExtents=array<i64:4>,dstExtents=array<i64:4>} : memref<4xf32>, memref<4xf32,1>
  return
}})mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(run(*module)));
  unsigned reductions = 0, stores = 0;
  module->walk([&](ReduceTensorOp) { ++reductions; });
  module->walk([&](TileStoreOp) { ++stores; });
  EXPECT_EQ(reductions, 1u);
  EXPECT_EQ(stores, 1u);
}
} // namespace
