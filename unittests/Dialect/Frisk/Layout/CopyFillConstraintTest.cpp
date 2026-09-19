#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class CountingExecutionTarget : public LayoutTarget {
public:
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  mutable unsigned enumerations = 0;
  void enumerateCandidates(const LayoutVar &var, SmallVectorImpl<LayoutCandidate> &out) const override {
    ++enumerations;
    target->enumerateCandidates(var,out);
  }
  LogicalResult verifyCandidate(const LayoutVar &var, Attribute candidate, Location loc) const override {
    return target->verifyCandidate(var,candidate,loc);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override { return target->evaluate(a); }
};
class CopyFillConstraintTest : public testing::Test {
protected:
  CopyFillConstraintTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect,
                        arith::ArithDialect>();
  }
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  OwningOpRef<ModuleOp> fill(StringRef extra = "") {
    return parseSourceString<ModuleOp>((Twine(R"mlir(
      func.func @fill(%p: memref<4xf32,3>) {
        %v = frisk.layout_view %p : memref<4xf32,3> -> memref<4xf32,3>
        frisk.fill %v {value = 0.0 : f32)mlir") + extra + R"mlir(} : memref<4xf32,3>
        return
      })mlir").str(), &context);
  }
  LogicalResult infer(ModuleOp module) {
    auto graph = collectLayoutConstraints(module, *target);
    if (failed(graph) || failed(propagateCommonToFixedPoint(*graph))) return failure();
    auto solution = solveBootstrapLayoutGraph(*graph, *target);
    if (failed(solution)) return failure();
    return materializeLayouts(module, *graph, *solution);
  }
  OwningOpRef<ModuleOp> parallelFill(int64_t threads) {
    return parseSourceString<ModuleOp>((Twine(R"mlir(
      func.func @parallel_fill(%p: memref<4xf32,3>) {
        "frisk.kernel"() <{sym_name="k",function_type=()->()}> ({
          "frisk.parallel"() <{ranges=array<i64:4>,threads=)mlir") + Twine(threads) + R"mlir(:i64}> ({
          ^bb0(%i:index):
            %v = frisk.layout_view %p : memref<4xf32,3> -> memref<4xf32,3>
            frisk.fill %v {value=0.0:f32} : memref<4xf32,3>
            "frisk.end"() : ()->()
          }) : ()->()
          "frisk.end"() : ()->()
        }) : ()->()
        return
      })mlir").str(), &context);
  }
  std::string print(Operation *op) {
    std::string result;
    llvm::raw_string_ostream os(result);
    op->print(os);
    return result;
  }
};

TEST_F(CopyFillConstraintTest, FillMaterializesExecutionAndReplaysExactly) {
  auto module = fill();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  FillOp op;
  module->walk([&](FillOp f) { op = f; });
  auto layout = op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout");
  ASSERT_TRUE(layout);
  EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 32);
  EXPECT_EQ(op->getAttrOfType<StringAttr>("frisk.writer_policy").getValue(), "first_owner");
  EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.vector_bytes").getInt(), 1);
  auto once = print(*module);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_EQ(print(*module), once);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(CopyFillConstraintTest, ReplicatedAllWriterFailsWithoutChangingIR) {
  auto module = fill(", frisk.writer_policy = \"all\"");
  ASSERT_TRUE(module);
  auto before = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_EQ(print(*module), before);
}

TEST_F(CopyFillConstraintTest, MissingActualExecutionBindingIsRejected) {
  auto module = fill();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  module->walk([&](FillOp f) { f->removeAttr("frisk.execution_layout"); });
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(CopyFillConstraintTest, ParallelSeedsExactThreadEnvironment) {
  for (int64_t threads : {32,64,128,256,512,1024}) {
    auto module = parallelFill(threads);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(infer(*module)));
    FillOp op;
    module->walk([&](FillOp f) { op=f; });
    auto layout = op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout");
    ASSERT_TRUE(layout);
    EXPECT_EQ(layout.getTopology()[1]*layout.getTopology()[2]*layout.getTopology()[3], threads);
    EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), threads);
  }
}

TEST_F(CopyFillConstraintTest, ParallelRejectsInvalidResourceAndExplicitOverride) {
  auto module = parallelFill(128);
  ASSERT_TRUE(module);
  FillOp op;
  ParallelOp parallel;
  module->walk([&](FillOp f) { op=f; });
  module->walk([&](ParallelOp p) { parallel=p; });
  Builder b(&context);
  op->setAttr("frisk.execution_threads", b.getI64IntegerAttr(32));
  auto before = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_EQ(print(*module), before);
  op->removeAttr("frisk.execution_threads");
  parallel.setThreads(48);
  EXPECT_TRUE(failed(verify(*module)));
}

TEST_F(CopyFillConstraintTest, ExecutionAttributeFormatIsVerified) {
  auto module = fill();
  ASSERT_TRUE(module);
  FillOp op;
  module->walk([&](FillOp f) { op=f; });
  Builder b(&context);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  for (auto [name, value] : {
      std::pair<StringRef,Attribute>{"frisk.vector_bytes", b.getI64IntegerAttr(3)},
      {"frisk.writer_policy", b.getStringAttr("race")},
      {"frisk.execution_layout", b.getStringAttr("not-a-layout")},
      {"frisk.execution_threads", b.getI32IntegerAttr(32)}}) {
    op->setAttr(name, value);
    EXPECT_TRUE(failed(verify(*module))) << name.str();
    op->removeAttr(name);
  }
}

TEST_F(CopyFillConstraintTest, ExplicitEncodingCannotOverrideParallelTopology) {
  auto scalar = fill();
  ASSERT_TRUE(scalar);
  ASSERT_TRUE(succeeded(infer(*scalar)));
  Attribute encoding;
  scalar->walk([&](FillOp f) { encoding = f->getAttr("frisk.execution_layout"); });
  auto module = parallelFill(128);
  ASSERT_TRUE(module);
  module->walk([&](FillOp f) { f->setAttr("frisk.execution_layout", encoding); });
  auto before = print(*module);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_EQ(print(*module), before);
}

TEST_F(CopyFillConstraintTest, CopyAllowsDifferentPhysicalStorageLayouts) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    #a = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[2,2],
      outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(i,j)->(8*i+4*j,0)>>,
      memory_space=#frisk<memory_space Global>,alignment=4,vector_granularity=1>
    #b = #frisk.storage<map=#frisk.affine_layout<inputs=["dim0","dim1"],input_extents=[2,2],
      outputs=["byte_offset","bit_offset"],output_extents=[16,8],map=affine_map<(i,j)->(4*i+8*j,0)>>,
      memory_space=#frisk<memory_space Shared>,alignment=4,vector_granularity=1>
    func.func @copy(%s:memref<2x2xf32,1>, %d:memref<2x2xf32,3>) {
      %sv = frisk.layout_view %s {layout=#a} : memref<2x2xf32,1> -> memref<2x2xf32,1>
      %dv = frisk.layout_view %d {layout=#b} : memref<2x2xf32,3> -> memref<2x2xf32,3>
      "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,
        srcExtents=array<i64:2,2>,dstExtents=array<i64:2,2>}> {operandSegmentSizes=array<i32:1,1,0,0>}
        : (memref<2x2xf32,1>,memref<2x2xf32,3>)->()
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(CopyFillConstraintTest, KernelEntryMemrefIsAStaticStorageRoot) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    "frisk.kernel"() <{sym_name="entry",function_type=(memref<4xf32,3>)->()}> ({
    ^bb0(%p:memref<4xf32,3>):
      "frisk.parallel"() <{ranges=array<i64:4>,threads=128:i64}> ({
      ^bb0(%i:index):
        %v = frisk.layout_view %p : memref<4xf32,3> -> memref<4xf32,3>
        frisk.fill %v {value=0.0:f32} : memref<4xf32,3>
        "frisk.end"() : ()->()
      }) : ()->()
      "frisk.end"() : ()->()
    }) : ()->()
  )mlir", &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  // Actual printer/parser roundtrip must preserve operation contracts.
  auto roundtrip = parseSourceString<ModuleOp>(print(*module), &context);
  ASSERT_TRUE(roundtrip);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*roundtrip, *target)));
}

TEST_F(CopyFillConstraintTest, AllActualContractFieldsAreRequiredAndChecked) {
  for (StringRef name : {"frisk.execution_layout", "frisk.execution_threads",
                         "frisk.writer_policy", "frisk.vector_bytes"}) {
    auto module = fill();
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(infer(*module)));
    FillOp op;
    module->walk([&](FillOp f) { op=f; });
    op->removeAttr(name);
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target))) << name.str();
  }
  for (auto [name, value] : {
      std::pair<StringRef,Attribute>{"frisk.execution_layout", StringAttr::get(&context,"corrupt")},
      std::pair<StringRef,Attribute>{"frisk.execution_threads", IntegerAttr::get(IntegerType::get(&context,64),128)},
      {"frisk.writer_policy", StringAttr::get(&context,"all")},
      {"frisk.vector_bytes", IntegerAttr::get(IntegerType::get(&context,64),16)}}) {
    auto module = fill();
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(infer(*module)));
    module->walk([&](FillOp f) { f->setAttr(name,value); });
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target))) << name.str();
  }
}

TEST_F(CopyFillConstraintTest, NewPassDoesNotCallLegacyParallelInference) {
  auto module = parallelFill(128);
  ASSERT_TRUE(module);
  ParallelOp op;
  module->walk([&](ParallelOp p) { op=p; });
  auto count = getLegacyParallelInferenceCallCount();
  OpBuilder builder(&context);
  DenseMap<Value, Attribute> legacy;
  (void)op.inferLayout(builder, legacy);
  EXPECT_EQ(getLegacyParallelInferenceCallCount(), count+1);
  count = getLegacyParallelInferenceCallCount();
  PassManager pm(&context);
  pm.addPass(createFriskInferLayoutsPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  EXPECT_EQ(getLegacyParallelInferenceCallCount(), count);
}

TEST_F(CopyFillConstraintTest, CapturedTensorConvertsAtParallelUse) {
  // Pin a real pre-existing outer encoding to 32 threads before collecting the
  // inner 128-thread use. This must not retag the captured producer.
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @capture(%x:tensor<4xf32>) {
      "frisk.kernel"() <{sym_name="k",function_type=()->()}> ({
        "frisk.parallel"() <{ranges=array<i64:4>,threads=128:i64}> ({
        ^bb0(%i:index):
          %sum = arith.addf %x, %x : tensor<4xf32>
          "frisk.end"() : ()->()
        }) : ()->()
        "frisk.end"() : ()->()
      }) : ()->()
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<func::FuncOp>().begin();
  LayoutVar seed;
  seed.kind = LayoutKind::Distributed;
  seed.shapedType = function.getArgument(0).getType();
  seed.requiredThreads = 32;
  SmallVector<LayoutCandidate> candidates;
  target->enumerateCandidates(seed,candidates);
  ASSERT_FALSE(candidates.empty());
  auto type = RankedTensorType::get({4}, Float32Type::get(&context), candidates.front().value);
  function.getArgument(0).setType(type);
  function.setType(FunctionType::get(&context,{type},{}));
  ASSERT_TRUE(succeeded(infer(*module)));
  function = *module->getOps<func::FuncOp>().begin();
  EXPECT_EQ(function.getArgument(0).getType(), type);
  unsigned conversions = 0;
  module->walk([&](ConvertLayoutOp c) {
    ++conversions;
    auto layout = cast<DistributedEncodingAttr>(cast<RankedTensorType>(c.getType()).getEncoding());
    EXPECT_EQ(layout.getTopology()[1]*layout.getTopology()[2]*layout.getTopology()[3],128);
  });
  EXPECT_EQ(conversions,2u);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(CopyFillConstraintTest, ActualVerifierNeverEnumeratesExecutionCandidates) {
  auto module=fill();
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  CountingExecutionTarget counted;
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,counted)));
  EXPECT_EQ(counted.enumerations,0u);
  module->walk([&](FillOp op) { op->removeAttr("frisk.writer_policy"); });
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module,counted)));
  EXPECT_EQ(counted.enumerations,0u);
}

TEST_F(CopyFillConstraintTest, CopySeedsTransferInBothDirectionsWithRebasedOrigins) {
  for (bool sourceSeed : {true,false}) {
    auto module = parseSourceString<ModuleOp>(R"mlir(
      func.func @rebased(%s:memref<4xf32,strided<[1],offset:5>,1>,
                         %d:memref<4xf32,strided<[1],offset:3>,3>) {
        %sv=frisk.layout_view %s : memref<4xf32,strided<[1],offset:5>,1> -> memref<4xf32,strided<[1],offset:5>,1>
        %dv=frisk.layout_view %d : memref<4xf32,strided<[1],offset:3>,3> -> memref<4xf32,strided<[1],offset:3>,3>
        "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,
          srcExtents=array<i64:4>,dstExtents=array<i64:4>}> {operandSegmentSizes=array<i32:1,1,0,0>}
          : (memref<4xf32,strided<[1],offset:5>,1>,memref<4xf32,strided<[1],offset:3>,3>)->()
        return
      })mlir", &context);
    ASSERT_TRUE(module);
    SmallVector<LayoutViewOp> views;
    module->walk([&](LayoutViewOp v) { views.push_back(v); });
    auto selected=sourceSeed?views[0]:views[1];
    auto info=analyzeStorageAlias(selected.getResult());
    ASSERT_TRUE(succeeded(info));
    auto seed=buildRootLinearStorageCandidate(*info);
    ASSERT_TRUE(succeeded(seed));
    selected->setAttr("layout",*seed);
    auto graph=collectLayoutConstraints(*module,*target);
    ASSERT_TRUE(succeeded(graph));
    auto destination=graph->lookupVariable((sourceSeed?views[1]:views[0]).getResult());
    ASSERT_TRUE(destination);
    bool projected=false;
    for (const auto &candidate : graph->getVariable(*destination).candidates) {
      if (candidate.provenance < graph->getProvenances().size())
        projected |= graph->getProvenances()[candidate.provenance].rule=="storage-copy-projection";
      auto proof=getStorageAliasFootprint(*graph,*destination,candidate.value);
      EXPECT_EQ(proof.proof.status,ProofStatus::Proven);
      EXPECT_EQ(proof.entries.front().begin,(sourceSeed?3u:5u)*32);
    }
    EXPECT_TRUE(projected);
    ASSERT_TRUE(succeeded(infer(*module)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,*target)));
  }
}

TEST_F(CopyFillConstraintTest, SameRootCopyRejectsShiftedOverlapButAllowsIdentityAndDisjoint) {
  for (int offset : {0,2,4}) {
    auto code=(Twine(R"mlir(
      func.func @overlap(%r:memref<8xf32,3>) {
        %a=memref.subview %r[0] [4] [1] : memref<8xf32,3> to memref<4xf32,strided<[1]>,3>
        %b=memref.subview %r[)mlir") + Twine(offset) + R"mlir(] [4] [1] : memref<8xf32,3> to memref<4xf32,strided<[1],offset:)mlir" + Twine(offset) + R"mlir(>,3>
        %av=frisk.layout_view %a : memref<4xf32,strided<[1]>,3> -> memref<4xf32,strided<[1]>,3>
        %bv=frisk.layout_view %b : memref<4xf32,strided<[1],offset:)mlir" + Twine(offset) + R"mlir(>,3> -> memref<4xf32,strided<[1],offset:)mlir" + Twine(offset) + R"mlir(>,3>
        "frisk.copy"(%av,%bv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,
          srcExtents=array<i64:4>,dstExtents=array<i64:4>}> {operandSegmentSizes=array<i32:1,1,0,0>}
          : (memref<4xf32,strided<[1]>,3>,memref<4xf32,strided<[1],offset:)mlir" + Twine(offset) + R"mlir(>,3>)->()
        return
      })mlir").str();
    auto module=parseSourceString<ModuleOp>(code,&context);
    ASSERT_TRUE(module);
    auto before=print(*module);
    ScopedDiagnosticHandler quiet(&context,[](Diagnostic &){return success();});
    auto result=infer(*module);
    EXPECT_EQ(succeeded(result),offset!=2);
    if (offset==2) EXPECT_EQ(print(*module),before);
  }
}

TEST_F(CopyFillConstraintTest, VectorWidthFiltersCandidateDomainsAndPreservesHardBinding) {
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @vector() {
      %r=memref.alloc() {alignment=16:i64} : memref<128xf32,3>
      %v=frisk.layout_view %r : memref<128xf32,3> -> memref<128xf32,3>
      frisk.fill %v {value=0.0:f32,frisk.vector_bytes=16:i64,frisk.writer_policy="all"} : memref<128xf32,3>
      return
    })mlir",&context);
  ASSERT_TRUE(module);
  auto graph=collectLayoutConstraints(*module,*target);
  ASSERT_TRUE(succeeded(graph));
  LayoutVarID execution=0;
  size_t before=0;
  for (const auto &v:graph->getVariables()) if(v.operationExecution) {execution=v.id;before=v.candidates.size();}
  EXPECT_GT(before,1u);
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  EXPECT_LT(graph->getVariable(execution).candidates.size(),before);
  EXPECT_TRUE(graph->getPropagationStatistics().strict.hasValidBounds());
  EXPECT_TRUE(graph->getPropagationStatistics().common.hasValidBounds());
  ASSERT_TRUE(succeeded(infer(*module)));
  FillOp fillOp;
  module->walk([&](FillOp f){fillOp=f;});
  EXPECT_EQ(fillOp->getAttrOfType<IntegerAttr>("frisk.vector_bytes").getInt(),16);
  EXPECT_EQ(fillOp->getAttrOfType<StringAttr>("frisk.writer_policy").getValue(),"all");
  // Remove both allocation evidence and the explicit whole-root binding
  // precondition (Task 18 intentionally preserves the latter across replay).
  // The operation's vector requirement is not itself alignment evidence.
  unsigned allocations = 0;
  module->walk([&](memref::AllocOp a) {
    ++allocations;
    a.removeAlignmentAttr();
    EXPECT_FALSE(a.getAlignmentAttr());
  });
  ASSERT_EQ(allocations, 1u);
  auto info = analyzeStorageAlias(fillOp.getMemref());
  ASSERT_TRUE(succeeded(info));
  ASSERT_EQ(info->rootAlignment, 1u);
  CountingExecutionTarget counted;
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, counted)));
  EXPECT_EQ(counted.enumerations, 0u);
  module->walk([&](LayoutViewOp view) {
    auto layout = view.getLayoutAttr();
    auto one = IntegerAttr::get(IntegerType::get(&context, 64), 1);
    view.setLayoutAttr(StorageLayoutAttr::get(&context, layout.getMap(),
        layout.getMemorySpace(), one, one));
  });
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, counted))) << print(*module);
  EXPECT_EQ(counted.enumerations, 0u);
}
} // namespace
