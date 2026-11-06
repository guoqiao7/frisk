#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class NoEnumerationTarget : public LayoutTarget {
public:
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  mutable unsigned enumerations = 0;
  void enumerateCandidates(const LayoutVar &var,
                           SmallVectorImpl<LayoutCandidate> &out) const override {
    ++enumerations;
    target->enumerateCandidates(var, out);
  }
  LogicalResult verifyCandidate(const LayoutVar &var, Attribute candidate,
                                Location loc) const override {
    return target->verifyCandidate(var, candidate, loc);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &assignment) const override {
    return target->evaluate(assignment);
  }
};
constexpr StringLiteral layouts = R"mlir(
#root = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(4*d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
#reverse = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0"],input_extents=[4],
outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(d)->(12-4*d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=4>
)mlir";
class StorageRootContractTest : public testing::Test {
protected:
  MLIRContext context;
  StorageRootContractTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect,
                        arith::ArithDialect, scf::SCFDialect>();
  }
  OwningOpRef<ModuleOp> parse(StringRef body, bool verifyAfterParse = true) {
    return parseSourceString<ModuleOp>((layouts + body).str(),
                                      ParserConfig(&context, verifyAfterParse));
  }
};

TEST_F(StorageRootContractTest, UnaryContractDoesNotAddVariablesAndRechecksIR) {
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>) {
  frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  EXPECT_EQ(graph->getVariables().size(), 1u);
  unsigned unary = 0;
  for (const auto &c : graph->getConstraints())
    unary += c.kind == ConstraintKind::RootStorageContract;
  EXPECT_EQ(unary, 1u);
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  NoEnumerationTarget actual;
  ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, actual)));
  EXPECT_EQ(actual.enumerations, 0u);
  module->walk([](StorageContractOp op) { op.erase(); });
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, actual)));
  EXPECT_EQ(actual.enumerations, 0u);
}

TEST_F(StorageRootContractTest, ConflictingUnusedDeclarationsFail) {
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>) {
  frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
  frisk.storage_contract %r {layout=#reverse} : memref<4xi32,3>
  return
})mlir");
  ASSERT_TRUE(module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectStorageRootContracts(*module)));
}

TEST_F(StorageRootContractTest, BranchContractCannotSupplyOutsideAlignment) {
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>, %c:i1) {
  scf.if %c {
    frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
  }
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_EQ(graph->getVariables().size(), 1u);
  EXPECT_EQ(graph->getVariables()[0].storageAlias->rootAlignment, 1u);
  for (const auto &c : graph->getConstraints())
    EXPECT_NE(c.kind, ConstraintKind::RootStorageContract);
}

TEST_F(StorageRootContractTest, BranchWholeViewCannotSupplyOutsideAlignment) {
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>, %c:i1) {
  scf.if %c {
    %rv = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
  }
  %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
  %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_EQ(graph->getVariables().size(), 2u);
  for (const auto &var : graph->getVariables())
    if (isa<func::FuncOp>(var.anchor->getParentOp()))
      EXPECT_EQ(var.storageAlias->rootAlignment, 1u);
}

TEST_F(StorageRootContractTest, SliceCannotMasqueradeAsContractRoot) {
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>) {
  %v = frisk.layout_view %r : memref<4xi32,3> -> memref<4xi32,3>
  frisk.storage_contract %v {layout=#root} : memref<4xi32,3>
  return
})mlir");
  EXPECT_FALSE(module);
}

TEST_F(StorageRootContractTest, UnknownRegionCannotSupplyWholeViewEvidence) {
  context.allowUnregisteredDialects();
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>) {
  "test.region"() ({
    %s = memref.subview %r[1] [2] [1] : memref<4xi32,3> to memref<2xi32,strided<[1],offset:1>,3>
    %v = frisk.layout_view %s : memref<2xi32,strided<[1],offset:1>,3> -> memref<2xi32,strided<[1],offset:1>,3>
    %rv = frisk.layout_view %r {layout=#root} : memref<4xi32,3> -> memref<4xi32,3>
    "test.end"() : () -> ()
  }) : () -> ()
  return
})mlir");
  ASSERT_TRUE(module);
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(*module, *target, LayoutCollectionMode::RelationsOnly);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_EQ(graph->getVariables().size(), 2u);
  for (const auto &var : graph->getVariables())
    EXPECT_EQ(var.storageAlias->rootAlignment, 1u);
}

TEST_F(StorageRootContractTest, UnknownRegionDeclarationFailsEvenUnused) {
  context.allowUnregisteredDialects();
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>) {
  "test.region"() ({
    frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
    "test.end"() : () -> ()
  }) : () -> ()
  return
})mlir");
  ASSERT_TRUE(module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  NoEnumerationTarget target;
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
  EXPECT_EQ(target.enumerations, 0u);
}

TEST_F(StorageRootContractTest, UnusedOverBudgetDeclarationIsNotAssumedValid) {
  auto module = parse(R"mlir(
#large = #frisk.storage<map=#frisk.affine_layout<inputs=["d"],input_extents=[131072],
outputs=["byte_offset","bit_offset"],output_extents=[131072,8],map=affine_map<(d)->(d,0)>>,
memory_space=#frisk<memory_space Shared>,alignment=1,vector_granularity=1>
func.func @f(%r:memref<131072xi8,3>) {
  frisk.storage_contract %r {layout=#large} : memref<131072xi8,3>
  return
})mlir", /*verifyAfterParse=*/false);
  ASSERT_TRUE(module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  NoEnumerationTarget target;
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
  EXPECT_EQ(target.enumerations, 0u);
}

TEST_F(StorageRootContractTest, ActualMapTamperingFailsWithoutEnumeration) {
  auto module = parse(R"mlir(
func.func @f(%r:memref<4xi32,3>) {
  frisk.storage_contract %r {layout=#root} : memref<4xi32,3>
  %v = frisk.layout_view %r {layout=#reverse} : memref<4xi32,3> -> memref<4xi32,3>
  return
})mlir");
  ASSERT_TRUE(module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  NoEnumerationTarget target;
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
  EXPECT_EQ(target.enumerations, 0u);
}
} // namespace
