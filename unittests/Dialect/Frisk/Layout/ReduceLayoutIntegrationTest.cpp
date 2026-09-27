#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class ReductionObservedTarget : public LayoutTarget {
public:
  std::unique_ptr<LayoutTarget> impl = createSM90LayoutTarget();
  mutable unsigned generations = 0, proofs = 0;
  void enumerateCandidates(const LayoutVar &v, SmallVectorImpl<LayoutCandidate> &out) const override {
    ++generations; impl->enumerateCandidates(v, out);
  }
  LogicalResult verifyCandidate(const LayoutVar &v, Attribute a, Location l) const override {
    return impl->verifyCandidate(v, a, l);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override { return impl->evaluate(a); }
  LogicalResult prepareInstructionCandidates(LayoutConstraintGraph &g) const override {
    ++generations; return impl->prepareInstructionCandidates(g);
  }
  FailureOr<Attribute> buildInstructionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, ArrayRef<Attribute> e) const override {
    ++generations; return impl->buildInstructionContract(g,c,e);
  }
  LayoutProof verifyInstructionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, ArrayRef<Attribute> e, Attribute b) const override {
    return impl->verifyInstructionContract(g,c,e,b);
  }
  FailureOr<Attribute> buildReductionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, Attribute a, Attribute b) const override {
    ++generations; return impl->buildReductionContract(g,c,a,b);
  }
  LayoutProof verifyReductionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, Attribute a, Attribute b, Attribute binding) const override {
    ++proofs; return impl->verifyReductionContract(g,c,a,b,binding);
  }
};
class ReduceLayoutIntegrationTest : public testing::Test {
protected:
  MLIRContext context;
  ReductionObservedTarget target;
  ReduceLayoutIntegrationTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect,
        memref::MemRefDialect, scf::SCFDialect, linalg::LinalgDialect, tensor::TensorDialect>();
  }
  OwningOpRef<ModuleOp> simple(StringRef extra = "") {
    return parseSourceString<ModuleOp>(
        "module { func.func @rows() { %a = arith.constant dense<1.0> : tensor<64x64xf32> "
        "%r = \"frisk.reduce_tensor\"(%a) {kind = \"sum\", dim = 1 : i64 " + extra.str() +
        "} : (tensor<64x64xf32>) -> tensor<64xf32> return } }", &context);
  }
  LogicalResult infer(ModuleOp module) {
    auto g = collectLayoutConstraints(module,target);
    if (failed(g) || failed(propagateStrict(*g)) || failed(propagateCommonToFixedPoint(*g))) return failure();
    auto s = solveLayoutGraph(*g,target);
    if (failed(s) || failed(verifySolvedLayoutGraph(*g,*s,target,module.getLoc()))) return failure();
    return materializeLayouts(module,*g,*s);
  }
  ReduceTensorOp reduce(ModuleOp module) {
    ReduceTensorOp found; module.walk([&](ReduceTensorOp r) { if (!found) found=r; }); return found;
  }
  std::string print(ModuleOp module) {
    std::string s; llvm::raw_string_ostream(s) << module; return s;
  }
};

TEST_F(ReduceLayoutIntegrationTest, RowsBindingReplayAndActualOnly) {
  auto m=simple(); ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
  auto op=reduce(*m);
  EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(),32);
  EXPECT_TRUE(op->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract"));
  target.generations=target.proofs=0;
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m,target)));
  EXPECT_EQ(target.generations,0u); EXPECT_GT(target.proofs,0u);
  auto text=print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(text,print(*m));
  auto parsed=parseSourceString<ModuleOp>(text,&context); ASSERT_TRUE(parsed);
  ASSERT_TRUE(succeeded(infer(*parsed))); EXPECT_EQ(text,print(*parsed));
}

TEST_F(ReduceLayoutIntegrationTest, FrozenPairsReverseFilteringAndGraphInvariants) {
  auto m=simple(); ASSERT_TRUE(m);
  auto graph=collectLayoutConstraints(*m,target); ASSERT_TRUE(succeeded(graph));
  LayoutConstraint *relation=nullptr;
  for (auto &c:graph->getConstraints()) if(c.reduction) relation=&c;
  ASSERT_NE(relation,nullptr); ASSERT_FALSE(relation->reduction->pairs.empty());
  auto &src=graph->getVariable(relation->vars[0]);
  auto &dst=graph->getVariable(relation->vars[1]);
  auto output=relation->reduction->pairs.front().resultEncoding;
  SmallVector<Attribute> expected;
  for (const auto &p:relation->reduction->pairs)
    if(p.resultEncoding==output && !llvm::is_contained(expected,p.sourceEncoding)) expected.push_back(p.sourceEncoding);
  auto originals=src.candidates;
  dst.candidates={{output,kInvalidProvenanceID,0}};
  ASSERT_TRUE(succeeded(propagateStrict(*graph)));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  EXPECT_EQ(src.candidates.size(),expected.size());
  for(auto c:src.candidates) EXPECT_TRUE(llvm::is_contained(expected,c.value));
  EXPECT_TRUE(graph->getPropagationStatistics().strict.hasValidBounds());
  EXPECT_TRUE(graph->getPropagationStatistics().common.hasValidBounds());
  EXPECT_LE(graph->getCandidatePreparationStatistics().reductionCombinations,16u);
  auto axis=relation->reduction->axis;
  relation->reduction->axis=0;
  { ScopedDiagnosticHandler quiet(&context,[](Diagnostic&){return success();});
    EXPECT_TRUE(failed(graph->verifyInvariants(m->getLoc()))); }
  relation->reduction->axis=axis;
  ASSERT_TRUE(succeeded(graph->finalize(m->getLoc())));
  EXPECT_TRUE(succeeded(graph->verifyInvariants(m->getLoc())));
}

TEST_F(ReduceLayoutIntegrationTest, ActualBindingTamperingFailsClosed) {
  for(StringRef field:{"missing","threads","scope","target","kind","axis"}) {
    SCOPED_TRACE(field.str()); auto m=simple(); ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
    auto op=reduce(*m); Builder b(&context);
    if(field=="missing") op->removeAttr("frisk.reduction_contract");
    else if(field=="threads") op->setAttr("frisk.execution_threads",b.getI64IntegerAttr(64));
    else {
      auto contract=op->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract");
      NamedAttrList attrs(contract.getPayload());
      if(field=="axis") attrs.set(field,b.getI64IntegerAttr(0));
      else if(field=="kind") attrs.set(field,b.getStringAttr("max"));
      else if(field=="target") attrs.set(field,b.getStringAttr("sm_90a"));
      else {
        auto old=cast<StringAttr>(attrs.get(field)).getValue();
        attrs.set(field,b.getStringAttr(old=="register"?"warp":"register"));
      }
      op->setAttr("frisk.reduction_contract",ReductionContractAttr::get(&context,attrs.getDictionary(&context)));
    }
    target.generations=0;
    ScopedDiagnosticHandler quiet(&context,[](Diagnostic&){return success();});
    EXPECT_TRUE(failed(verifyMaterializedLayouts(*m,target))); EXPECT_EQ(target.generations,0u);
  }
}

TEST_F(ReduceLayoutIntegrationTest, FailedMaterializationPreservesOriginal) {
  auto m=simple(); ASSERT_TRUE(m); auto before=print(*m);
  auto g=collectLayoutConstraints(*m,target); ASSERT_TRUE(succeeded(g));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*g)));
  auto solution=solveLayoutGraph(*g,target); ASSERT_TRUE(succeeded(solution));
  ASSERT_FALSE(solution->reductionBindings.empty());
  auto &entry=*solution->reductionBindings.begin();
  auto contract=cast<ReductionContractAttr>(entry.second); NamedAttrList attrs(contract.getPayload());
  attrs.set("kind",StringAttr::get(&context,"max"));
  entry.second=ReductionContractAttr::get(&context,attrs.getDictionary(&context));
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic&){return success();});
  EXPECT_TRUE(failed(materializeLayouts(*m,*g,*solution))); EXPECT_EQ(before,print(*m));
}

TEST_F(ReduceLayoutIntegrationTest, ExplicitThreadsAndUnsupportedTarget) {
  for(int64_t count:{32,64,128,256,512,1024}) {
    auto m=simple(", frisk.execution_threads = "+std::to_string(count)+" : i64"); ASSERT_TRUE(m);
    ASSERT_TRUE(succeeded(infer(*m)));
    EXPECT_EQ(reduce(*m)->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(),count);
  }
  auto m=simple(); ASSERT_TRUE(m); m->getOperation()->setAttr("frisk.target",StringAttr::get(&context,"sm_80"));
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic&){return success();});
  EXPECT_TRUE(failed(infer(*m)));
}

TEST_F(ReduceLayoutIntegrationTest, ConsecutiveReductions) {
  auto m=parseSourceString<ModuleOp>(R"mlir(module { func.func @chain() {
    %a = arith.constant dense<1.0> : tensor<4x8x16xbf16>
    %r = "frisk.reduce_tensor"(%a) {kind="max", dim=1:i64} : (tensor<4x8x16xbf16>) -> tensor<4x16xbf16>
    %s = "frisk.reduce_tensor"(%r) {kind="min", dim=0:i64} : (tensor<4x16xbf16>) -> tensor<16xbf16>
    return } })mlir",&context);
  ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m))); auto once=print(*m);
  ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once,print(*m));
}

TEST_F(ReduceLayoutIntegrationTest, TransposeThenReduceAndForCarriedSource) {
  for (StringRef body : {
      R"mlir(%a = arith.constant dense<1.0> : tensor<4x8x16xf32>
        %empty = tensor.empty() : tensor<8x4x16xf32>
        %t = linalg.transpose ins(%a : tensor<4x8x16xf32>) outs(%empty : tensor<8x4x16xf32>) permutation = [1, 0, 2]
        %r = "frisk.reduce_tensor"(%t) {kind="sum",dim=1:i64} : (tensor<8x4x16xf32>) -> tensor<8x16xf32>)mlir",
      R"mlir(%a = arith.constant dense<1.0> : tensor<64x64xf32>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %out = scf.for %i = %c0 to %c2 step %c1 iter_args(%carry = %a) -> tensor<64x64xf32> {
          %r = "frisk.reduce_tensor"(%carry) {kind="sum",dim=1:i64} : (tensor<64x64xf32>) -> tensor<64xf32>
          scf.yield %carry : tensor<64x64xf32>
        })mlir"}) {
    auto m=parseSourceString<ModuleOp>("module { func.func @test() { "+body.str()+" return } }",&context);
    ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
    auto once=print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once,print(*m));
  }
}

TEST_F(ReduceLayoutIntegrationTest, MmaResultInherits128Threads) {
  auto m=parseSourceString<ModuleOp>(R"mlir(module attributes {frisk.target="sm_90a"} {
    func.func @chain() {
      %a = memref.alloc() {alignment=1024:i64} : memref<128x64xbf16,3>
      %b = memref.alloc() {alignment=1024:i64} : memref<64x128xbf16,3>
      %av = frisk.layout_view %a : memref<128x64xbf16,3> -> memref<128x64xbf16,3>
      %bv = frisk.layout_view %b : memref<64x128xbf16,3> -> memref<64x128xbf16,3>
      %z = arith.constant dense<0.0> : tensor<128x128xf32>
      %c = "frisk.mma"(%av,%bv,%z) {m=128:i64,n=128:i64,k=64:i64}
        : (memref<128x64xbf16,3>,memref<64x128xbf16,3>,tensor<128x128xf32>) -> tensor<128x128xf32>
      %r = "frisk.reduce_tensor"(%c) {kind="sum",dim=1:i64} : (tensor<128x128xf32>) -> tensor<128xf32>
      return
    } })mlir",&context);
  ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
  EXPECT_EQ(reduce(*m)->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(),128);
  auto once=print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once,print(*m));
  // Adding real store endpoints produces nine connected variables. Re-solving
  // must preserve the MMA/reduction bindings and materialize the entire chain.
  auto function=*m->getOps<func::FuncOp>().begin();
  OpBuilder b(function.getBody().front().getTerminator());
  auto memory=MemRefType::get({128},b.getF32Type(),MemRefLayoutAttrInterface{},b.getI64IntegerAttr(1));
  auto allocation=b.create<memref::AllocOp>(m->getLoc(),memory);
  auto view=b.create<LayoutViewOp>(m->getLoc(),memory,allocation,StorageLayoutAttr());
  b.create<TileStoreOp>(m->getLoc(),reduce(*m).getResult(),view);
  auto graph=collectLayoutConstraints(*m,target); ASSERT_TRUE(succeeded(graph));
  EXPECT_EQ(graph->getVariables().size(),9u);
  ASSERT_TRUE(succeeded(infer(*m)));
  target.generations=target.proofs=0;
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m,target)));
  EXPECT_EQ(target.generations,0u);
  EXPECT_GT(target.proofs,0u);
  EXPECT_TRUE(reduce(*m)->getAttrOfType<ReductionContractAttr>("frisk.reduction_contract"));
  once=print(*m);
  ASSERT_TRUE(succeeded(infer(*m)));
  EXPECT_EQ(once,print(*m));
  auto reparsed=parseSourceString<ModuleOp>(once,&context);
  ASSERT_TRUE(reparsed);
  ASSERT_TRUE(succeeded(infer(*reparsed)));
  EXPECT_EQ(once,print(*reparsed));
}

TEST_F(ReduceLayoutIntegrationTest, ReduceToStoreUsesCompletedOutputOwner) {
  auto m=parseSourceString<ModuleOp>(R"mlir(module { func.func @store() {
    %a = arith.constant dense<1.0> : tensor<64x64xf32>
    %r = "frisk.reduce_tensor"(%a) {kind="sum",dim=1:i64} : (tensor<64x64xf32>) -> tensor<64xf32>
    %buffer = memref.alloc() : memref<64xf32,1>
    %view = frisk.layout_view %buffer : memref<64xf32,1> -> memref<64xf32,1>
    frisk.tile_store %r, %view : tensor<64xf32>, memref<64xf32,1>
    return } })mlir",&context);
  ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m,target)));
}

TEST_F(ReduceLayoutIntegrationTest, ExternalHardSourceConvertsAtConsumer) {
  auto type=RankedTensorType::get({64,64},Float32Type::get(&context));
  LayoutVar v; v.kind=LayoutKind::Distributed; v.shapedType=type; v.requiredThreads=32;
  SmallVector<LayoutCandidate> candidates; target.enumerateCandidates(v,candidates);
  ASSERT_FALSE(candidates.empty()); auto encoding=candidates.front().value;
  std::string typed; llvm::raw_string_ostream(typed) << RankedTensorType::get({64,64},type.getElementType(),encoding);
  auto m=parseSourceString<ModuleOp>("module { func.func @external(%a: "+typed+") { "
      "%r = frisk.reduce_tensor %a {kind=\"sum\", dim=1:i64, frisk.execution_threads=128:i64} : "+typed+
      " -> tensor<64xf32> return } }",&context);
  ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
  auto function=*m->getOps<func::FuncOp>().begin();
  EXPECT_EQ(cast<RankedTensorType>(function.getArgument(0).getType()).getEncoding(),encoding);
  EXPECT_TRUE(reduce(*m).getSource().getDefiningOp<ConvertLayoutOp>());
  auto once=print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once,print(*m));
}

TEST_F(ReduceLayoutIntegrationTest, HardResultNamesAndIncompatibleProjection) {
  auto type=RankedTensorType::get({64,64},Float32Type::get(&context));
  LayoutVar v; v.kind=LayoutKind::Distributed; v.shapedType=type; v.requiredThreads=32;
  SmallVector<LayoutCandidate> candidates; target.enumerateCandidates(v,candidates);
  ASSERT_FALSE(candidates.empty()); auto src=cast<DistributedEncodingAttr>(candidates.front().value);
  auto projected=projectReductionEncoding(src,type,RankedTensorType::get({64},type.getElementType()),1);
  ASSERT_TRUE(succeeded(projected)); auto map=cast<BitLinearLayoutMapAttr>(projected->getMap()); Builder b(&context);
  auto renamed=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),map.getInputBitWidths(),
      b.getArrayAttr({b.getStringAttr("row")}),map.getOutputBitWidths(),map.getMatrix());
  auto dst=DistributedEncodingAttr::get(&context,renamed,projected->getTopology(),projected->getReplication());
  std::string srcText,dstText;
  llvm::raw_string_ostream(srcText)<<RankedTensorType::get({64,64},type.getElementType(),src);
  llvm::raw_string_ostream(dstText)<<RankedTensorType::get({64},type.getElementType(),dst);
  auto m=parseSourceString<ModuleOp>("module { func.func @names(%a: "+srcText+") { "
      "%r=frisk.reduce_tensor %a {kind=\"sum\",dim=1:i64} : "+srcText+" -> "+dstText+" return } }",&context);
  ASSERT_TRUE(m); ASSERT_TRUE(succeeded(infer(*m)));
  EXPECT_EQ(cast<RankedTensorType>(reduce(*m).getResult().getType()).getEncoding(),dst);
  // A complete, legal encoding for the other projection is not a valid hard
  // result for this reduction, even though it has the same shape.
  auto wrong=projectReductionEncoding(src,type,RankedTensorType::get({64},type.getElementType()),0);
  ASSERT_TRUE(succeeded(wrong));
  auto result=reduce(*m).getResult(); result.setType(RankedTensorType::get({64},type.getElementType(),*wrong));
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic&){return success();});
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*m,target)));
}

TEST_F(ReduceLayoutIntegrationTest, DownstreamConversionsDoNotPolluteNaturalResult) {
  auto type = RankedTensorType::get({64, 64}, Float32Type::get(&context));
  LayoutVar v;
  v.kind = LayoutKind::Distributed;
  v.shapedType = type;
  v.requiredThreads = 32;
  SmallVector<LayoutCandidate> candidates;
  target.enumerateCandidates(v, candidates);
  ASSERT_FALSE(candidates.empty());
  auto src = cast<DistributedEncodingAttr>(candidates.front().value);
  auto outputType = RankedTensorType::get({64}, type.getElementType());
  auto natural = projectReductionEncoding(src, type, outputType, 1);
  ASSERT_TRUE(succeeded(natural));
  auto map = cast<BitLinearLayoutMapAttr>(natural->getMap());
  auto typeText = [](Type t) {
    std::string text;
    llvm::raw_string_ostream(text) << t;
    return text;
  };
  std::string input = typeText(RankedTensorType::get(type.getShape(), type.getElementType(), src));
  std::string output = typeText(RankedTensorType::get({64}, type.getElementType(), *natural));
  std::string body = "module { func.func @fanout(%a: " + input + ") { "
      "%r=frisk.reduce_tensor %a {kind=\"sum\",dim=1:i64} : " + input +
      " -> " + output + " ";
  Builder b(&context);
  for (unsigned i = 0; i < 4; ++i) {
    auto renamed = BitLinearLayoutMapAttr::get(&context, map.getInputNames(),
        map.getInputBitWidths(), b.getArrayAttr({b.getStringAttr("row" + std::to_string(i))}),
        map.getOutputBitWidths(), map.getMatrix());
    auto dst = DistributedEncodingAttr::get(&context, renamed,
        natural->getTopology(), natural->getReplication());
    body += "%c" + std::to_string(i) + "=frisk.convert_layout %r : " + output + " -> " +
        typeText(RankedTensorType::get({64}, type.getElementType(), dst)) + " ";
  }
  auto m = parseSourceString<ModuleOp>(body + "return } }", &context);
  ASSERT_TRUE(m);
  auto graph = collectLayoutConstraints(*m, target);
  ASSERT_TRUE(succeeded(graph));
  EXPECT_EQ(graph->getVariables().size(), 7u);
  for (const auto &var : graph->getVariables())
    EXPECT_EQ(var.candidates.size(), 1u) << var.stableName;
  ASSERT_TRUE(succeeded(infer(*m)));
  auto once = print(*m);
  ASSERT_TRUE(succeeded(infer(*m)));
  EXPECT_EQ(once, print(*m));
}

TEST_F(ReduceLayoutIntegrationTest, InheritedThreadsStopAtExecutionBoundary) {
  for (bool explicitConvert : {false, true}) {
    SCOPED_TRACE(explicitConvert);
    auto type = RankedTensorType::get({4, 8, 16}, Float32Type::get(&context));
    auto encoded = [&](int64_t threads) {
      LayoutVar v;
      v.kind = LayoutKind::Distributed;
      v.shapedType = type;
      v.requiredThreads = threads;
      SmallVector<LayoutCandidate> candidates;
      target.enumerateCandidates(v, candidates);
      std::string text;
      llvm::raw_string_ostream(text) << RankedTensorType::get(type.getShape(),
          type.getElementType(), candidates.front().value);
      return text;
    };
    auto src = encoded(32), converted = encoded(128);
    std::string body = "module { func.func @boundary(%a: " + src + ") { ";
    if (explicitConvert)
      body += "%c=frisk.convert_layout %a : " + src + " -> " + converted + " ";
    body += "%r=frisk.reduce_tensor " + std::string(explicitConvert ? "%c" : "%a") +
        " {kind=\"sum\",dim=1:i64" +
        std::string(explicitConvert ? "" : ",frisk.execution_threads=128:i64") +
        "} : " + (explicitConvert ? converted : src) + " -> tensor<4x16xf32> ";
    if (!explicitConvert)
      body += "%s=frisk.reduce_tensor %r {kind=\"sum\",dim=0:i64} : tensor<4x16xf32> -> tensor<16xf32> ";
    auto m = parseSourceString<ModuleOp>(body + "return } }", &context);
    ASSERT_TRUE(m);
    if (failed(infer(*m))) {
      ADD_FAILURE() << "thread inheritance crossed an execution boundary";
      continue;
    }
    m->walk([&](ReduceTensorOp op) {
      EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), 128);
    });
    auto once = print(*m);
    ASSERT_TRUE(succeeded(infer(*m)));
    EXPECT_EQ(once, print(*m));
  }
}

TEST_F(ReduceLayoutIntegrationTest, IndependentSameLayoutThreadsStillConflict) {
  auto m = simple();
  ASSERT_TRUE(m);
  LayoutConstraintGraph graph;
  LayoutConstraintBuilder builder(graph);
  auto source = builder.getOrCreateDistributedVar(reduce(*m).getSource());
  auto peer = graph.addVariable(LayoutKind::Distributed,
      reduce(*m).getSource().getType(), "independent", reduce(*m));
  graph.getVariable(source).requiredThreads = 32;
  graph.getVariable(peer).requiredThreads = 128;
  graph.addConstraint(ConstraintKind::SameLayout, ConstraintStrength::Hard,
      {source, peer}, reduce(*m), "independent-sources", "equal layouts");
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectReductionLayoutConstraints(*m, graph, builder)));
}

TEST_F(ReduceLayoutIntegrationTest, ParallelScopeAgreementAndMismatch) {
  for (bool mismatch:{false,true}) {
    auto m=parseSourceString<ModuleOp>("module { \"frisk.kernel\"() <{sym_name=\"parallel\", function_type=()->()}> ({ "
      "\"frisk.parallel\"() <{ranges=array<i64:4>,threads=128:i64}> ({ ^bb0(%i:index): "
      "%a=arith.constant dense<1.0>:tensor<64x64xf32> "
      "%r=frisk.reduce_tensor %a {kind=\"sum\",dim=1:i64"+
      std::string(mismatch?",frisk.execution_threads=32:i64":"")+
      "} : tensor<64x64xf32> -> tensor<64xf32> frisk.end }) : () -> () frisk.end }) : () -> () }",&context);
    ASSERT_TRUE(m);
    ScopedDiagnosticHandler quiet(&context,[](Diagnostic&){return success();});
    EXPECT_EQ(succeeded(infer(*m)),!mismatch);
  }
}
} // namespace
