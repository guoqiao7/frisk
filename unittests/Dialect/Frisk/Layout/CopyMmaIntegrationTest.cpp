#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90CopyConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class CopyMmaIntegrationTest : public testing::Test {
protected:
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  CopyMmaIntegrationTest() {
    context.loadDialect<FriskDialect, arith::ArithDialect, memref::MemRefDialect>();
  }
  OwningOpRef<ModuleOp> make(int64_t alignment = 128) {
    auto module = parseSourceFile<ModuleOp>(FRISK_COPY_MMA_FIXTURE, &context);
    if (module && alignment != 128) module->walk([&](StorageContractOp op) {
      auto layout = op.getLayout();
      op.setLayoutAttr(StorageLayoutAttr::get(&context, layout.getMap(),
          layout.getMemorySpace(), Builder(&context).getI64IntegerAttr(alignment),
          layout.getVectorGranularity()));
    });
    return module;
  }
  LogicalResult infer(ModuleOp module) {
    PassManager pm(&context);
    pm.enableVerifier(true);
    buildFriskLayoutPipeline(pm);
    return pm.run(module);
  }
  std::string print(ModuleOp module) {
    std::string result;
    llvm::raw_string_ostream(result) << module;
    return result;
  }
  // Pin complete, checked plans from a compatible joint solution. A fabricated
  // {path=...} dictionary does not stand in for an instruction contract.
  LogicalResult pin(ModuleOp module, StringRef path) {
    auto graph = collectLayoutConstraints(module, *target);
    if (failed(graph)) return failure();
    for (auto &var : graph->getVariables())
      if (var.kind == LayoutKind::Instruction && isa_and_nonnull<CopyOp>(var.anchor))
        llvm::erase_if(var.candidates, [&](const LayoutCandidate &candidate) {
          return cast<StringAttr>(candidate.value).getValue() != path;
        });
    if (failed(propagateCommonToFixedPoint(*graph))) return failure();
    auto solution = solveLayoutGraph(*graph, *target);
    if (failed(solution)) return failure();
    for (const auto &c : graph->getConstraints()) {
      if (!c.instruction || !isa_and_nonnull<CopyOp>(c.instruction->source)) continue;
      SmallVector<Attribute> encodings;
      for (auto id : c.vars) encodings.push_back(solution->assignments.lookup(id));
      auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
      if (result.proof.status != ProofStatus::Proven || !result.binding) return failure();
      c.instruction->source->setAttr("frisk.copy_contract", result.binding);
    }
    return success();
  }
  void check(ModuleOp module, StringRef path = {}, int64_t alignment = 128) {
    ASSERT_TRUE(succeeded(verify(module)));
    ASSERT_TRUE(succeeded(verifyMaterializedLayouts(module, *target)));
    SmallVector<CopyOp> copies;
    MmaOp mma;
    TileStoreOp store;
    module.walk([&](CopyOp op) {
      copies.push_back(op);
      auto binding = op->getAttrOfType<DictionaryAttr>("frisk.copy_contract");
      ASSERT_TRUE(binding);
      if (!path.empty()) EXPECT_EQ(binding.getAs<StringAttr>("path").getValue(), path);
      EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
    });
    module.walk([&](MmaOp op) { mma = op; });
    module.walk([&](TileStoreOp op) { store = op; });
    ASSERT_EQ(copies.size(), 2u);
    ASSERT_TRUE(mma); ASSERT_TRUE(store);
    EXPECT_EQ(copies[0]->getOperand(1), mma->getOperand(0));
    EXPECT_EQ(copies[1]->getOperand(1), mma->getOperand(1));
    auto result = store->getOperand(0);
    while (auto convert = result.getDefiningOp<ConvertLayoutOp>()) result = convert->getOperand(0);
    EXPECT_EQ(result, mma->getResult(0));
    auto contract = mma->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
    ASSERT_TRUE(contract);
    EXPECT_EQ(contract.getPayload().getAs<StringAttr>("form").getValue(), "ss");
    EXPECT_TRUE(mma.getTransB());
    auto roots = collectStorageRootContracts(module);
    ASSERT_TRUE(succeeded(roots));
    SmallVector<Value> externalRoots;
    for (const auto &root : *roots) {
      if (root.root.rootType.getMemorySpaceAsInt() != 1) continue;
      if (!llvm::is_contained(externalRoots, root.root.root)) externalRoots.push_back(root.root.root);
      auto layout = cast<StorageLayoutAttr>(root.layout);
      EXPECT_EQ(layout.getAlignment().getInt(), alignment);
      auto footprint = buildStorageAliasFootprint(root.root, layout);
      ASSERT_EQ(footprint.proof.status, ProofStatus::Proven);
      auto type = root.root.rootType;
      ASSERT_EQ(footprint.entries.size(), unsigned(type.getNumElements()));
      int64_t bytes = type.getElementTypeBitWidth() / 8;
      for (const auto &entry : footprint.entries) {
        EXPECT_EQ(entry.begin, 8 * bytes * (entry.view[0] * type.getDimSize(1) + entry.view[1]));
        EXPECT_EQ(entry.end - entry.begin, 8 * bytes);
      }
    }
    EXPECT_EQ(externalRoots.size(), 3u);
    // Check actual materialized external views, not only the declaration maps.
    auto graph = collectLayoutConstraints(module, *target, LayoutCollectionMode::RelationsOnly);
    ASSERT_TRUE(succeeded(graph));
    unsigned globals = 0;
    for (const auto &var : graph->getVariables()) {
      if (!var.storageAlias || var.storageAlias->rootType.getMemorySpaceAsInt() != 1) continue;
      ++globals;
      auto layout = var.anchor->getAttrOfType<StorageLayoutAttr>("layout");
      ASSERT_TRUE(layout);
      const auto &footprint = getStorageAliasFootprint(*graph, var.id, layout);
      ASSERT_EQ(footprint.proof.status, ProofStatus::Proven);
      auto type = var.storageAlias->viewType;
      ASSERT_EQ(footprint.entries.size(), unsigned(type.getNumElements()));
      for (const auto &entry : footprint.entries)
        EXPECT_EQ(entry.begin, type.getElementTypeBitWidth() *
            (entry.view[0] * type.getDimSize(1) + entry.view[1]));
    }
    EXPECT_EQ(globals, 3u);
  }
};

TEST_F(CopyMmaIntegrationTest, FixedABICompletePathsAndFreeSelectionReplay) {
  for (StringRef path : {"vector", "cp.async", "tma", ""}) {
    SCOPED_TRACE(path.str());
    auto module = make(); ASSERT_TRUE(module);
    if (!path.empty()) ASSERT_TRUE(succeeded(pin(*module, path)));
    ASSERT_TRUE(succeeded(infer(*module)));
    check(*module, path);
    auto once = print(*module);
    auto replay = parseSourceString<ModuleOp>(once, &context); ASSERT_TRUE(replay);
    ASSERT_TRUE(succeeded(infer(*replay)));
    EXPECT_EQ(once, print(*replay));
    check(*replay, path);
  }
}

TEST_F(CopyMmaIntegrationTest, SingleFieldPlanTamperingFailsAtomically) {
  for (StringRef field : {"issuer", "completion", "descriptor"}) {
    SCOPED_TRACE(field.str());
    auto module = make(); ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(pin(*module, "tma")));
    bool changed = false;
    module->walk([&](CopyOp op) {
      if (changed) return;
      auto binding = op->getAttrOfType<DictionaryAttr>("frisk.copy_contract");
      NamedAttrList altered(binding);
      if (field == "descriptor") {
        NamedAttrList descriptor(binding.getAs<DictionaryAttr>(field));
        descriptor.set("source_offset", Builder(&context).getI64IntegerAttr(2));
        altered.set(field, descriptor.getDictionary(&context));
      } else altered.set(field, Builder(&context).getStringAttr("forged"));
      op->setAttr("frisk.copy_contract", altered.getDictionary(&context));
      changed = true;
    });
    auto before = print(*module);
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(infer(*module)));
    EXPECT_EQ(before, print(*module));
  }
}

TEST_F(CopyMmaIntegrationTest, ByteAlignmentFallsBackWithoutChangingExternalABI) {
  auto module = make(1); ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  check(*module, "vector", 1);
}

TEST_F(CopyMmaIntegrationTest, ByteAlignmentWithHardWideCopyFailsAtomically) {
  auto module = make(1); ASSERT_TRUE(module);
  bool changed = false;
  module->walk([&](CopyOp op) {
    if (changed) return;
    op->setAttr("frisk.vector_bytes", Builder(&context).getI64IntegerAttr(16));
    changed = true;
  });
  auto before = print(*module);
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream(message) << diagnostic; return success();
  });
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_EQ(before, print(*module));
  // Projected instruction candidates include actual-address proofs exceeding
  // the enumeration budget. Mixed failure evidence must remain Unknown.
  EXPECT_NE(message.find("unknown proof: copy-budget"), std::string::npos) << message;
}
} // namespace
