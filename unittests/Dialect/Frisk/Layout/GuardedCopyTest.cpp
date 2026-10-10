#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class GuardedCopyTest : public testing::Test {
protected:
  GuardedCopyTest() { context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>(); }
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  OwningOpRef<ModuleOp> copy(int64_t m, int64_t n) {
    std::string shape = std::to_string(m) + "x" + std::to_string(n) + "xf32";
    return parseSourceString<ModuleOp>(
      "func.func @copy(%s:memref<" + shape + ",1>,%d:memref<" + shape + ",3>) {\n"
      "%sv=frisk.layout_view %s : memref<" + shape + ",1> -> memref<" + shape + ",1>\n"
      "%dv=frisk.layout_view %d : memref<" + shape + ",3> -> memref<" + shape + ",3>\n"
      "\"frisk.copy\"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,srcExtents=array<i64:" +
      std::to_string(m) + "," + std::to_string(n) + ">,dstExtents=array<i64:" +
      std::to_string(m) + "," + std::to_string(n) + ">}> {operandSegmentSizes=array<i32:1,1,0,0>} : (memref<" +
      shape + ",1>,memref<" + shape + ",3>)->()\nreturn\n}", &context);
  }
  LogicalResult infer(ModuleOp module) {
    auto graph = collectLayoutConstraints(module, *target);
    if (failed(graph) || failed(propagateCommonToFixedPoint(*graph))) return failure();
    auto solution = solveLayoutGraph(*graph, *target);
    if (failed(solution)) return failure();
    return materializeLayouts(module, *graph, *solution);
  }
};
TEST_F(GuardedCopyTest, UnitAxesUseZeroOutputBitsWithoutPadding) {
  for (auto shape : {SmallVector<int64_t>{1,8}, SmallVector<int64_t>{1,1}}) {
    LayoutVar var;
    var.kind=LayoutKind::Distributed;
    var.shapedType=RankedTensorType::get(shape,Float32Type::get(&context));
    var.requiredThreads=32;
    SmallVector<LayoutCandidate> candidates;
    target->enumerateCandidates(var,candidates);
    ASSERT_FALSE(candidates.empty());
    for (const auto &candidate:candidates) {
      auto encoding=cast<DistributedEncodingAttr>(candidate.value);
      auto map=cast<BitLinearLayoutMapAttr>(encoding.getMap());
      EXPECT_EQ(map.getOutputBitWidths()[0],0);
      EXPECT_TRUE(succeeded(map.verifyMap(UnknownLoc::get(&context))));
      EXPECT_TRUE(succeeded(map.canonicalizeMap()));
      EXPECT_TRUE(succeeded(encoding.verifyForType(cast<RankedTensorType>(var.shapedType),UnknownLoc::get(&context))));
    }
  }
}
TEST_F(GuardedCopyTest, RaggedAndUnitAxesHaveBoundedPaddedExecution) {
  for (auto [m,n] : {std::pair<int64_t,int64_t>{3,5},{7,18},{1,8},{1,1}}) {
    SCOPED_TRACE(std::to_string(m) + "x" + std::to_string(n));
    auto module = copy(m,n);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(infer(*module)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,*target)));
    module->walk([&](CopyOp op) {
      EXPECT_EQ(op.getSrcMemRefType().getDimSize(0),m);
      EXPECT_EQ(op.getSrcMemRefType().getDimSize(1),n);
      EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.vector_bytes").getInt(),1);
    });
  }
}
TEST_F(GuardedCopyTest, RejectsRaggedWideVectorsAndOversizeDomains) {
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  auto module = copy(3,5);
  ASSERT_TRUE(module);
  module->walk([&](CopyOp op) { op->setAttr("frisk.vector_bytes",Builder(&context).getI64IntegerAttr(4)); });
  EXPECT_TRUE(failed(infer(*module)));
  for (auto [m,n] : {std::pair<int64_t,int64_t>{257,129},{65537,1}}) {
    auto oversized = copy(m,n);
    ASSERT_TRUE(oversized);
    EXPECT_TRUE(failed(collectLayoutConstraints(*oversized,*target)));
  }
}
TEST_F(GuardedCopyTest, RaggedFillRemainsUnsupported) {
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @fill(%p:memref<3x5xf32,3>) {
      %v=frisk.layout_view %p : memref<3x5xf32,3> -> memref<3x5xf32,3>
      frisk.fill %v {value=0.0:f32} : memref<3x5xf32,3>
      return
    })mlir",&context);
  ASSERT_TRUE(module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectLayoutConstraints(*module,*target)));
}
TEST_F(GuardedCopyTest, NonWholeTileAndDynamicCopyRemainUnsupported) {
  auto module=copy(3,5);
  ASSERT_TRUE(module);
  module->walk([&](CopyOp op) {
    op.setSrcExtentsAttr(Builder(&context).getDenseI64ArrayAttr({2,5}));
    op.setDstExtentsAttr(Builder(&context).getDenseI64ArrayAttr({2,5}));
  });
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectLayoutConstraints(*module,*target)));
  auto dynamic=parseSourceString<ModuleOp>(R"mlir(
    func.func @dynamic(%s:memref<?xf32,1>,%d:memref<?xf32,3>) {
      %sv=frisk.layout_view %s : memref<?xf32,1> -> memref<?xf32,1>
      %dv=frisk.layout_view %d : memref<?xf32,3> -> memref<?xf32,3>
      "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,
        srcExtents=array<i64:3>,dstExtents=array<i64:3>}> {operandSegmentSizes=array<i32:1,1,0,0>}
        : (memref<?xf32,1>,memref<?xf32,3>)->()
      return
    })mlir",&context);
  ASSERT_TRUE(dynamic);
  EXPECT_TRUE(failed(collectLayoutConstraints(*dynamic,*target)));
}
TEST_F(GuardedCopyTest, GuardedProofRejectsForgedDomainsAndFootprints) {
  Builder b(&context);
  auto padded=RankedTensorType::get({4,8},b.getF32Type());
  LayoutVar var;
  var.kind=LayoutKind::Distributed;
  var.shapedType=padded;
  var.requiredThreads=32;
  SmallVector<LayoutCandidate> candidates;
  target->enumerateCandidates(var,candidates);
  ASSERT_FALSE(candidates.empty());
  auto encoding=cast<DistributedEncodingAttr>(candidates.front().value);
  StorageAliasInfo info;
  info.viewType=info.rootType=MemRefType::get({3,5},b.getF32Type());
  info.viewToRoot=AffineMap::getMultiDimIdentityMap(2,&context);
  info.upperBit=15*32;
  StorageAliasFootprint footprint;
  footprint.rootType=info.rootType;
  footprint.proof={ProofStatus::Proven,{},"actual domain"};
  for(int64_t i=0;i<3;++i) for(int64_t j=0;j<5;++j) {
    uint64_t begin=(i*5+j)*32;
    footprint.entries.push_back({{i,j},{i,j},begin,begin+32});
  }
  auto proof=[&](RankedTensorType type,unsigned width=1,StringRef policy="first_owner") {
    return proveGuardedExecutionVectorAccess(encoding,type,info,footprint,width,AccessKind::Write,policy).status;
  };
  EXPECT_EQ(proof(padded),ProofStatus::Proven);
  EXPECT_EQ(proof(RankedTensorType::get({8,8},b.getF32Type())),ProofStatus::Disproven);
  EXPECT_EQ(proof(RankedTensorType::get({4,8},b.getI32Type())),ProofStatus::Disproven);
  EXPECT_EQ(proof(padded,4),ProofStatus::Disproven);
  EXPECT_EQ(proof(padded,1,"invalid"),ProofStatus::Disproven);
  auto original=footprint;
  footprint.entries.back().view={3,0};
  EXPECT_EQ(proof(padded),ProofStatus::Disproven);
  footprint=original;
  footprint.entries[1].begin=0;
  footprint.entries[1].end=32;
  EXPECT_EQ(proof(padded),ProofStatus::Disproven);
  footprint=original;
  footprint.entries.pop_back();
  EXPECT_EQ(proof(padded),ProofStatus::Disproven);
  footprint=original;
  footprint.proof.status=ProofStatus::Unknown;
  EXPECT_NE(proof(padded),ProofStatus::Proven);
}
} // namespace
