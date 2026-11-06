#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
// Follow only semantic identity conversions; a fake intermediate constant or
// disconnected kernel argument must never satisfy a dataflow assertion.
Value original(Value value) {
  while (auto convert = value.getDefiningOp<ConvertLayoutOp>())
    value = convert->getOperand(0);
  return value;
}

class AttentionLayoutIntegrationTest : public testing::Test {
protected:
  MLIRContext context;
  AttentionLayoutIntegrationTest() {
    context.loadDialect<FriskDialect, arith::ArithDialect,
        linalg::LinalgDialect, math::MathDialect, memref::MemRefDialect,
        tensor::TensorDialect>();
  }
  OwningOpRef<ModuleOp> make() {
    return parseSourceFile<ModuleOp>(FRISK_ATTENTION_FIXTURE, &context);
  }
  StorageLayoutAttr rowMajor(MemRefType type, int64_t alignment,
                            bool reverseColumns = false) {
    Builder builder(&context);
    auto i = getAffineDimExpr(0, &context);
    auto j = getAffineDimExpr(1, &context);
    auto column = reverseColumns ? type.getDimSize(1) - 1 - j : j;
    int64_t bytes = type.getElementTypeBitWidth() / 8;
    auto map = AffineLayoutMapAttr::get(&context,
        builder.getStrArrayAttr({"dim0", "dim1"}),
        builder.getDenseI64ArrayAttr(type.getShape()),
        builder.getStrArrayAttr({"byte_offset", "bit_offset"}),
        builder.getDenseI64ArrayAttr({type.getNumElements() * bytes, 8}),
        AffineMapAttr::get(AffineMap::get(2, 0,
            {bytes * (type.getDimSize(1) * i + column),
             getAffineConstantExpr(0, &context)}, &context)));
    return StorageLayoutAttr::get(&context, map,
        MemorySpaceAttr::get(&context, attr::MemorySpace::Global),
        builder.getI64IntegerAttr(alignment), builder.getI64IntegerAttr(1));
  }
  OwningOpRef<ModuleOp> makeFixed(int64_t alignment, int64_t rootRows = 64,
      int64_t rootColumns = 16, int64_t rowOffset = 0,
      int64_t columnOffset = 0) {
    auto module = parseSourceFile<ModuleOp>(FRISK_ATTENTION_FIXED_ABI_FIXTURE,
                                           &context);
    if (!module) return module;
    KernelOp kernel;
    module->walk([&](KernelOp op) { kernel = op; });
    SmallVector<Type> inputs;
    for (BlockArgument arg : kernel->getRegion(0).front().getArguments()) {
      auto old = cast<MemRefType>(arg.getType());
      auto type = MemRefType::get({rootRows, rootColumns}, old.getElementType(),
          MemRefLayoutAttrInterface{}, old.getMemorySpace());
      arg.setType(type);
      inputs.push_back(type);
    }
    kernel->setAttr("function_type", TypeAttr::get(
        FunctionType::get(&context, inputs, {})));
    module->walk([&](StorageContractOp op) {
      op->setAttr("layout", rowMajor(cast<MemRefType>(op.getRoot().getType()),
                                    alignment));
    });
    if (rootRows != 64 || rootColumns != 16 || rowOffset || columnOffset) {
      module->walk([&](LayoutViewOp view) {
        if (!isa<BlockArgument>(view.getSource()) ||
            cast<MemRefType>(view.getSource().getType()).getMemorySpaceAsInt() != 1)
          return;
        OpBuilder builder(view);
        auto slice = builder.create<memref::SubViewOp>(view.getLoc(),
            view.getSource(), ArrayRef<int64_t>{rowOffset, columnOffset},
            ArrayRef<int64_t>{64, 16}, ArrayRef<int64_t>{1, 1});
        view->setOperand(0, slice.getResult());
        view.getResult().setType(slice.getType());
      });
    }
    return module;
  }
  std::string print(ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream(text) << module;
    return text;
  }
  LogicalResult infer(ModuleOp module, SolverOptions options = {}) {
    PassManager pm(&context);
    pm.enableVerifier(true);
    buildFriskLayoutPipeline(pm, options);
    return pm.run(module);
  }
  void checkTensorEncodings(ModuleOp module, bool encoded) {
    unsigned tensors = 0;
    auto check = [&](Value value) {
      if (auto type = dyn_cast<RankedTensorType>(value.getType())) {
        ++tensors;
        EXPECT_EQ(bool(type.getEncoding()), encoded);
        if (encoded)
          EXPECT_TRUE(isa<DistributedEncodingAttr>(type.getEncoding()));
      }
    };
    module.walk([&](Operation *op) {
      for (Value result : op->getResults()) check(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument arg : block.getArguments()) check(arg);
    });
    EXPECT_GT(tensors, 15u);
  }
  void checkDataflow(ModuleOp module, int64_t rootRows = 64,
      int64_t rootColumns = 16, int64_t rowOffset = 0,
      int64_t columnOffset = 0) {
    SmallVector<KernelOp> kernels;
    SmallVector<MmaOp> mmas;
    SmallVector<ReduceTensorOp> reductions;
    SmallVector<linalg::BroadcastOp> broadcasts;
    SmallVector<CopyOp> copies;
    SmallVector<TileStoreOp> stores;
    SmallVector<arith::MulFOp> scales;
    SmallVector<arith::SubFOp> centered;
    SmallVector<math::ExpOp> exponentials;
    SmallVector<arith::DivFOp> normalized;
    SmallVector<arith::TruncFOp> truncations;
    module.walk([&](KernelOp op) { kernels.push_back(op); });
    module.walk([&](MmaOp op) { mmas.push_back(op); });
    module.walk([&](ReduceTensorOp op) { reductions.push_back(op); });
    module.walk([&](linalg::BroadcastOp op) { broadcasts.push_back(op); });
    module.walk([&](CopyOp op) { copies.push_back(op); });
    module.walk([&](TileStoreOp op) { stores.push_back(op); });
    module.walk([&](arith::MulFOp op) { scales.push_back(op); });
    module.walk([&](arith::SubFOp op) { centered.push_back(op); });
    module.walk([&](math::ExpOp op) { exponentials.push_back(op); });
    module.walk([&](arith::DivFOp op) { normalized.push_back(op); });
    module.walk([&](arith::TruncFOp op) { truncations.push_back(op); });
    ASSERT_EQ(kernels.size(), 1u);
    ASSERT_EQ(mmas.size(), 2u);
    ASSERT_EQ(reductions.size(), 2u);
    ASSERT_EQ(broadcasts.size(), 2u);
    ASSERT_EQ(copies.size(), 3u);
    ASSERT_EQ(stores.size(), 1u);
    ASSERT_EQ(scales.size(), 1u);
    ASSERT_EQ(centered.size(), 1u);
    ASSERT_EQ(exponentials.size(), 1u);
    ASSERT_EQ(normalized.size(), 1u);
    ASSERT_EQ(truncations.size(), 1u);
    auto kernel = kernels.front();
    ASSERT_EQ(kernel.getNumArguments(), 4u);
    for (unsigned i = 0; i < 4; ++i) {
      auto type = cast<MemRefType>(kernel.getArgument(i).getType());
      EXPECT_EQ(type.getShape(), (ArrayRef<int64_t>{rootRows, rootColumns}));
      EXPECT_EQ(type.getMemorySpaceAsInt(), 1u);
      EXPECT_EQ(type.getElementType().isF32(), i == 3);
      EXPECT_EQ(type.getElementType().isF16(), i != 3);
    }
    for (unsigned i = 0; i < 3; ++i) {
      auto global = copies[i]->getOperand(0).getDefiningOp<LayoutViewOp>();
      auto shared = copies[i]->getOperand(1).getDefiningOp<LayoutViewOp>();
      ASSERT_TRUE(global);
      ASSERT_TRUE(shared);
      auto alias = analyzeStorageAlias(global.getSource());
      ASSERT_TRUE(succeeded(alias));
      EXPECT_EQ(alias->root, kernel.getArgument(i));
      for (auto point : {SmallVector<int64_t>{0, 0},
                         SmallVector<int64_t>{63, 15}}) {
        auto coordinate = evaluateStorageViewCoordinates(*alias, point);
        ASSERT_TRUE(succeeded(coordinate));
        EXPECT_EQ(*coordinate, (SmallVector<int64_t>{point[0] + rowOffset,
                                                   point[1] + columnOffset}));
      }
      auto allocation = shared.getSource().getDefiningOp<memref::AllocOp>();
      ASSERT_TRUE(allocation);
      EXPECT_EQ(allocation.getType().getMemorySpaceAsInt(), 3u);
      EXPECT_EQ(allocation->getAttrOfType<IntegerAttr>("alignment").getInt(), 1024);
      Value mmaOperand = i == 0 ? mmas[0]->getOperand(0)
                        : i == 1 ? mmas[0]->getOperand(1) : mmas[1]->getOperand(1);
      EXPECT_EQ(mmaOperand, shared.getResult());
    }
    for (unsigned i = 0; i < 2; ++i) {
      EXPECT_EQ(mmas[i].getM(), 64);
      EXPECT_EQ(mmas[i].getN(), i == 0 ? 64 : 16);
      EXPECT_EQ(mmas[i].getK(), i == 0 ? 16 : 64);
      EXPECT_FALSE(mmas[i].getTransA());
      EXPECT_EQ(mmas[i].getTransB(), i == 0);
      auto zero = original(mmas[i]->getOperand(2)).getDefiningOp<arith::ConstantOp>();
      ASSERT_TRUE(zero);
      auto value = dyn_cast<DenseFPElementsAttr>(zero.getValue());
      ASSERT_TRUE(value && value.isSplat());
      EXPECT_TRUE(value.getSplatValue<APFloat>().isZero());
      EXPECT_EQ(reductions[i].getDim(), 1);
      EXPECT_EQ(reductions[i].getKind(), i == 0 ? "max" : "sum");
      EXPECT_EQ(cast<RankedTensorType>(reductions[i].getResult().getType()).getShape(),
                (ArrayRef<int64_t>{64}));
      EXPECT_EQ(broadcasts[i].getDimensions(), (ArrayRef<int64_t>{1}));
      EXPECT_EQ(original(broadcasts[i]->getOperand(0)), reductions[i].getResult());
    }
    EXPECT_EQ(original(scales[0]->getOperand(0)), mmas[0].getResult());
    auto scale = original(scales[0]->getOperand(1)).getDefiningOp<arith::ConstantOp>();
    ASSERT_TRUE(scale);
    auto scaleValue = dyn_cast<DenseFPElementsAttr>(scale.getValue());
    ASSERT_TRUE(scaleValue && scaleValue.isSplat());
    EXPECT_EQ(scaleValue.getSplatValue<APFloat>().convertToDouble(), 0.25);
    EXPECT_EQ(original(reductions[0].getSource()), scales[0].getResult());
    EXPECT_EQ(original(centered[0]->getOperand(0)), scales[0].getResult());
    EXPECT_EQ(original(centered[0]->getOperand(1)), broadcasts[0]->getResult(0));
    EXPECT_EQ(original(exponentials[0]->getOperand(0)), centered[0].getResult());
    EXPECT_EQ(original(reductions[1].getSource()), exponentials[0].getResult());
    EXPECT_EQ(original(normalized[0]->getOperand(0)), exponentials[0].getResult());
    EXPECT_EQ(original(normalized[0]->getOperand(1)), broadcasts[1]->getResult(0));
    EXPECT_EQ(original(truncations[0]->getOperand(0)), normalized[0].getResult());
    EXPECT_TRUE(cast<RankedTensorType>(truncations[0]->getOperand(0).getType()).getElementType().isF32());
    EXPECT_TRUE(cast<RankedTensorType>(truncations[0].getType()).getElementType().isF16());
    EXPECT_EQ(original(mmas[1]->getOperand(0)), truncations[0].getResult());
    EXPECT_EQ(original(stores[0]->getOperand(0)), mmas[1].getResult());
    auto output = stores[0]->getOperand(1).getDefiningOp<LayoutViewOp>();
    ASSERT_TRUE(output);
    auto outputAlias = analyzeStorageAlias(output.getSource());
    ASSERT_TRUE(succeeded(outputAlias));
    EXPECT_EQ(outputAlias->root, kernel.getArgument(3));
    auto outputOrigin = evaluateStorageViewCoordinates(*outputAlias, {0, 0});
    ASSERT_TRUE(succeeded(outputOrigin));
    EXPECT_EQ(*outputOrigin, (SmallVector<int64_t>{rowOffset, columnOffset}));
  }
  void checkFixedAddresses(ModuleOp module, int64_t alignment,
      int64_t rootColumns, int64_t rowOffset, int64_t columnOffset) {
    auto target = createSM90LayoutTarget();
    auto contracts = collectStorageRootContracts(module);
    ASSERT_TRUE(succeeded(contracts));
    SmallVector<Value> externalRoots;
    for (const auto &contract : *contracts) {
      if (contract.root.rootType.getMemorySpaceAsInt() != 1) continue;
      if (!llvm::is_contained(externalRoots, contract.root.root))
        externalRoots.push_back(contract.root.root);
      auto layout = cast<StorageLayoutAttr>(contract.layout);
      EXPECT_EQ(layout.getAlignment().getInt(), alignment);
      EXPECT_EQ(layout.getVectorGranularity().getInt(), 1);
      auto footprint = buildStorageAliasFootprint(contract.root, layout);
      ASSERT_EQ(footprint.proof.status, ProofStatus::Proven);
      int64_t bytes = contract.root.rootType.getElementTypeBitWidth() / 8;
      ASSERT_EQ(footprint.entries.size(),
                unsigned(contract.root.rootType.getNumElements()));
      for (const auto &entry : footprint.entries) {
        EXPECT_EQ(entry.begin, 8 * bytes * (entry.view[0] * rootColumns + entry.view[1]));
        EXPECT_EQ(entry.end - entry.begin, 8 * bytes);
      }
    }
    EXPECT_EQ(externalRoots.size(), 4u);
    auto graph = collectLayoutConstraints(module, *target,
                                          LayoutCollectionMode::RelationsOnly);
    ASSERT_TRUE(succeeded(graph));
    unsigned globals = 0;
    for (const auto &var : graph->getVariables()) {
      if (!var.storageAlias || var.storageAlias->rootType.getMemorySpaceAsInt() != 1)
        continue;
      ++globals;
      auto layout = var.anchor->getAttrOfType<StorageLayoutAttr>("layout");
      ASSERT_TRUE(layout);
      EXPECT_EQ(var.storageAlias->rootAlignment, unsigned(alignment));
      EXPECT_LE(layout.getAlignment().getInt(), alignment);
      const auto &footprint = getStorageAliasFootprint(*graph, var.id, layout);
      ASSERT_EQ(footprint.proof.status, ProofStatus::Proven);
      ASSERT_EQ(footprint.entries.size(), 64u * 16u);
      int64_t bytes = var.storageAlias->viewType.getElementTypeBitWidth() / 8;
      for (const auto &entry : footprint.entries) {
        auto i = entry.view[0], j = entry.view[1];
        EXPECT_EQ(entry.root, (SmallVector<int64_t>{i + rowOffset, j + columnOffset}));
        EXPECT_EQ(entry.begin, 8 * bytes * ((i + rowOffset) * rootColumns + j + columnOffset));
        EXPECT_EQ(entry.end - entry.begin, 8 * bytes);
      }
    }
    EXPECT_EQ(globals, 4u);
  }
  void checkFixedReplay(int64_t alignment, int64_t rootRows = 64,
      int64_t rootColumns = 16, int64_t rowOffset = 0,
      int64_t columnOffset = 0) {
    auto module = makeFixed(alignment, rootRows, rootColumns, rowOffset, columnOffset);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    checkTensorEncodings(*module, false);
    checkDataflow(*module, rootRows, rootColumns, rowOffset, columnOffset);
    SmallVector<Attribute> originalContracts;
    module->walk([&](StorageContractOp op) { originalContracts.push_back(op.getLayout()); });
    module->walk([&](LayoutViewOp op) { EXPECT_FALSE(op->hasAttr("layout")); });
    ASSERT_TRUE(succeeded(infer(*module)));
    ASSERT_TRUE(succeeded(verify(*module)));
    auto target = createSM90LayoutTarget();
    ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
    checkTensorEncodings(*module, true);
    checkDataflow(*module, rootRows, rootColumns, rowOffset, columnOffset);
    checkFixedAddresses(*module, alignment, rootColumns, rowOffset, columnOffset);
    unsigned index = 0;
    module->walk([&](StorageContractOp op) {
      if (!isa<KernelOp>(op->getParentOp())) return;
      ASSERT_LT(index, originalContracts.size());
      EXPECT_EQ(op.getLayout(), originalContracts[index++]);
    });
    EXPECT_EQ(index, 4u);
    module->walk([&](CopyOp op) {
      auto contract = op->getAttrOfType<DictionaryAttr>("frisk.copy_contract");
      ASSERT_TRUE(contract);
      if (alignment == 1 || columnOffset == 1) {
        EXPECT_EQ(contract.getAs<StringAttr>("path").getValue(), "vector");
        EXPECT_EQ(contract.getAs<IntegerAttr>("vector_bytes").getInt(),
                  alignment == 1 ? 1 : 2);
      }
      if (columnOffset == 1) {
        auto alias = analyzeStorageAlias(op->getOperand(0));
        ASSERT_TRUE(succeeded(alias));
        alias->rootAlignment = alignment; // Explicit precondition checked above.
        auto view = op->getOperand(0).getDefiningOp<LayoutViewOp>();
        auto layout = view->getAttrOfType<StorageLayoutAttr>("layout");
        auto footprint = buildStorageAliasFootprint(*alias, layout);
        auto execution = op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout");
        ASSERT_EQ(footprint.proof.status, ProofStatus::Proven);
        EXPECT_EQ(proveExecutionVectorAccess(execution, *alias, footprint,
            2, AccessKind::Read, "all").status, ProofStatus::Proven);
        for (unsigned width : {4u, 8u, 16u}) {
          auto proof = proveExecutionVectorAccess(execution, *alias, footprint,
              width, AccessKind::Read, "all");
          EXPECT_EQ(proof.status, ProofStatus::Disproven);
          EXPECT_NE(proof.reason.find("misaligned"), std::string::npos) << proof.reason;
        }
      }
    });
    auto once = print(*module);
    ASSERT_TRUE(succeeded(infer(*module)));
    EXPECT_EQ(once, print(*module));
    auto replay = parseSourceString<ModuleOp>(once, &context);
    ASSERT_TRUE(replay);
    ASSERT_TRUE(succeeded(verify(*replay)));
    ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*replay, *target)));
    checkFixedAddresses(*replay, alignment, rootColumns, rowOffset, columnOffset);
    checkDataflow(*replay, rootRows, rootColumns, rowOffset, columnOffset);
    ASSERT_TRUE(succeeded(infer(*replay)));
    EXPECT_EQ(once, print(*replay));
  }
};

TEST_F(AttentionLayoutIntegrationTest, FixedABIByteAlignmentMaterializesAndReplays) {
  checkFixedReplay(1);
}

TEST_F(AttentionLayoutIntegrationTest, FixedABIExplicitAlignmentMaterializesAndReplays) {
  checkFixedReplay(16);
}

TEST_F(AttentionLayoutIntegrationTest, FixedABIStaticRowOffsetMaterializesAndReplays) {
  checkFixedReplay(16, 128, 16, 16, 0);
}

TEST_F(AttentionLayoutIntegrationTest, FixedABIMisalignedSubviewUsesScalarElements) {
  checkFixedReplay(16, 128, 32, 16, 1);
}

TEST_F(AttentionLayoutIntegrationTest, FixedABIIncompatibleHardViewPreservesOriginal) {
  auto module = makeFixed(1);
  ASSERT_TRUE(module);
  LayoutViewOp firstGlobal;
  module->walk([&](LayoutViewOp op) {
    if (!firstGlobal && cast<MemRefType>(op.getType()).getMemorySpaceAsInt() == 1)
      firstGlobal = op;
  });
  ASSERT_TRUE(firstGlobal);
  firstGlobal->setAttr("layout", rowMajor(cast<MemRefType>(firstGlobal.getType()),
                                        1, true));
  ASSERT_TRUE(succeeded(verify(*module)));
  auto contracts = collectStorageRootContracts(*module);
  ASSERT_TRUE(succeeded(contracts));
  ASSERT_EQ(contracts->size(), 4u);
  auto alias = analyzeStorageAlias(firstGlobal.getResult());
  ASSERT_TRUE(succeeded(alias));
  EXPECT_EQ(proveRootStorageContract(contracts->front(), *alias,
      firstGlobal->getAttr("layout")).status, ProofStatus::Disproven);
  auto before = print(*module);
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream(message) << diagnostic;
    return success();
  });
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_FALSE(message.empty());
  EXPECT_EQ(before, print(*module));
  checkTensorEncodings(*module, false);
  checkDataflow(*module);
}

TEST_F(AttentionLayoutIntegrationTest, CompleteAttentionMaterializesAndReplays) {
  auto module = make();
  ASSERT_TRUE(module);
  checkTensorEncodings(*module, false);
  checkDataflow(*module);
  ASSERT_TRUE(succeeded(infer(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  checkTensorEncodings(*module, true);
  checkDataflow(*module);
  unsigned index = 0;
  module->walk([&](MmaOp op) {
    auto contract = op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
    ASSERT_TRUE(contract);
    EXPECT_EQ(contract.getPayload().getAs<StringAttr>("form").getValue(),
              index++ == 0 ? "ss" : "rs");
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](ReduceTensorOp op) {
    EXPECT_TRUE(op->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  module->walk([&](CopyOp op) {
    EXPECT_TRUE(op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout"));
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
  });
  auto target = createSM90LayoutTarget();
  ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  auto once = print(*module);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_EQ(once, print(*module));
  auto replay = parseSourceString<ModuleOp>(once, &context);
  ASSERT_TRUE(replay);
  ASSERT_TRUE(succeeded(verify(*replay)));
  ASSERT_TRUE(succeeded(infer(*replay)));
  EXPECT_EQ(once, print(*replay));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*replay, *target)));
  checkDataflow(*replay);
}

TEST_F(AttentionLayoutIntegrationTest, FreeLayoutRefinementImprovesFullCostAndVerifiesGraphAndIR) {
  CostVector baseline;
  const CostVector known{8293,576,64,0,428,6144,6,0,0};
  for (bool refine : {false, true}) {
    auto module = make(); ASSERT_TRUE(module);
    PassManager normalize(&context);
    normalize.addPass(createNormalizeLayoutIRPass());
    ASSERT_TRUE(succeeded(normalize.run(*module)));
    auto target = createSM90LayoutTarget();
    auto graph = collectLayoutConstraints(*module, *target,
        LayoutCollectionMode::InitializeForRelaxation);
    ASSERT_TRUE(succeeded(graph));
    SolverOptions options; options.refineBeamSolution = refine;
    SmallVector<RelaxationRecord, 0> records;
    LayoutSolveFailure why;
    auto solution = attemptControlledRelaxation(*graph, *target, options, records, &why);
    ASSERT_TRUE(succeeded(solution)) << why.reason;
    EXPECT_TRUE(succeeded(verifySolvedLayoutGraph(*graph, *solution, *target, module->getLoc())));
    if (!refine) {
      baseline = solution->cost.cost;
      EXPECT_EQ(baseline.memoryTransactions, 608u);
    } else {
      EXPECT_FALSE(known < solution->cost.cost);
      EXPECT_TRUE(solution->cost.cost < baseline);
      EXPECT_GT(solution->components.front().refinementAccepted, 0u);
    }
    auto materialized = make(); ASSERT_TRUE(materialized);
    LayoutStatisticsReport report;
    PassManager pipeline(&context);
    buildFriskLayoutPipeline(pipeline, options, {}, &report);
    ASSERT_TRUE(succeeded(pipeline.run(*materialized)));
    EXPECT_TRUE(succeeded(verify(*materialized)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*materialized, *target)));
    checkDataflow(*materialized);
    checkTensorEncodings(*materialized, true);
    ASSERT_FALSE(report.attempts.empty());
    EXPECT_EQ(report.attempts.back().statistics.estimatedMemoryTransactions,
              solution->cost.cost.memoryTransactions);
  }
}

TEST_F(AttentionLayoutIntegrationTest, SearchBudgetOnePreservesUnencodedOriginal) {
  auto module = make();
  ASSERT_TRUE(module);
  auto originalText = print(*module);
  SolverOptions options;
  options.maxExpandedStates = 1;
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream(message) << diagnostic;
    return success();
  });
  EXPECT_TRUE(failed(infer(*module, options)));
  EXPECT_NE(message.find("search-budget-exceeded"), std::string::npos) << message;
  EXPECT_EQ(originalText, print(*module));
  checkTensorEncodings(*module, false);
  checkDataflow(*module);
}
} // namespace
