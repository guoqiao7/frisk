#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/BroadcastLayoutConstraints.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Builders.h"
#include "LayoutFailureCase.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class BroadcastObservedTarget : public LayoutTarget {
public:
  std::unique_ptr<LayoutTarget> impl = createSM90LayoutTarget();
  mutable unsigned generations = 0;
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
  FailureOr<Attribute> buildReductionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, Attribute a, Attribute b) const override {
    ++generations; return impl->buildReductionContract(g, c, a, b);
  }
  LayoutProof verifyReductionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, Attribute a, Attribute b, Attribute binding) const override {
    return impl->verifyReductionContract(g, c, a, b, binding);
  }
};
class BroadcastLayoutTest : public ::testing::Test {
protected:
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  BroadcastLayoutTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect,
        linalg::LinalgDialect, math::MathDialect, tensor::TensorDialect>();
  }
  OwningOpRef<ModuleOp> simple(StringRef input, StringRef output, StringRef axes) {
    return parseSourceString<ModuleOp>("module { func.func @broadcast(%a: " + input.str() +
      ") { %e=tensor.empty():" + output.str() +
      " %b=linalg.broadcast ins(%a:" + input.str() + ") outs(%e:" + output.str() +
      ") dimensions=[" + axes.str() + "] return } }", &context);
  }
  LogicalResult infer(ModuleOp module) {
    auto graph = collectLayoutConstraints(module, *target);
    if (failed(graph) || failed(propagateCommonToFixedPoint(*graph))) return failure();
    auto solution = solveLayoutGraph(*graph, *target);
    if (failed(solution) || failed(verifySolvedLayoutGraph(*graph, *solution,
          *target, module.getLoc()))) return failure();
    return materializeLayouts(module, *graph, *solution);
  }
  std::string print(ModuleOp m) {
    std::string s; llvm::raw_string_ostream(s) << m; return s;
  }
  // Independent semantic oracle: for every destination holder, delete the
  // broadcast axes and search ALL source registers in that exact thread/CTA.
  bool coordinates(linalg::BroadcastOp op) {
    auto src = cast<RankedTensorType>(op->getOperand(0).getType());
    auto dst = cast<RankedTensorType>(op->getResult(0).getType());
    ExecutionEnumeration a, b;
    if (enumerateExecutionLayout(cast<DistributedEncodingAttr>(src.getEncoding()), src, a).status != ProofStatus::Proven ||
        enumerateExecutionLayout(cast<DistributedEncodingAttr>(dst.getEncoding()), dst, b).status != ProofStatus::Proven) return false;
    for (uint64_t h = 0; h < b.logical.size(); ++h) {
      auto out = b.coordinate(b.logical[h]);
      SmallVector<int64_t> wanted;
      for (unsigned axis = 0; axis < out.size(); ++axis)
        if (!llvm::is_contained(op.getDimensions(), axis)) wanted.push_back(out[axis]);
      auto holder = b.hardware(h); holder.pop_back();
      bool found = false;
      for (uint64_t s = 0; s < a.logical.size(); ++s) {
        auto owner = a.hardware(s); owner.pop_back();
        if (owner == holder && a.coordinate(a.logical[s]) == wanted) { found = true; break; }
      }
      if (!found) return false;
    }
    return true;
  }
};

TEST_F(BroadcastLayoutTest, AxisZeroAxisOneAndMultidimensionalReplay) {
  for (auto spec : {std::array<StringRef, 3>{"tensor<8xf32>", "tensor<4x8xf32>", "0"},
                    {"tensor<4xf32>", "tensor<4x8xf32>", "1"},
                    {"tensor<4x8xf32>", "tensor<2x4x4x8xf32>", "0, 2"}}) {
    auto m = simple(spec[0], spec[1], spec[2]); ASSERT_TRUE(m);
    ASSERT_TRUE(succeeded(infer(*m)));
    m->walk([&](linalg::BroadcastOp op) { EXPECT_TRUE(coordinates(op)); });
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m, *target)));
    auto once = print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once, print(*m));
  }
}

TEST_F(BroadcastLayoutTest, CoordinateProofNamesReplicationAndWrongAxis) {
  auto type = RankedTensorType::get({4, 4}, Float32Type::get(&context));
  auto input = RankedTensorType::get({4}, type.getElementType());
  LayoutVar v; v.kind = LayoutKind::Distributed; v.shapedType = type;
  SmallVector<LayoutCandidate> candidates; target->enumerateCandidates(v, candidates);
  ASSERT_FALSE(candidates.empty());
  auto out = cast<DistributedEncodingAttr>(candidates.front().value);
  auto good = projectBroadcastInputEncoding(out, input, type, {1});
  auto bad = projectBroadcastInputEncoding(out, input, type, {0});
  ASSERT_TRUE(succeeded(good)); ASSERT_TRUE(succeeded(bad));
  EXPECT_EQ(proveBroadcastLayout(*good, out, input, type, {1}).status, ProofStatus::Proven);
  EXPECT_EQ(proveBroadcastLayout(*bad, out, input, type, {1}).status, ProofStatus::Disproven);
  Builder b(&context);
  auto map = cast<BitLinearLayoutMapAttr>(good->getMap());
  auto renamed = BitLinearLayoutMapAttr::get(&context, map.getInputNames(), map.getInputBitWidths(),
      b.getArrayAttr({b.getStringAttr("unrelated_axis_name")}), map.getOutputBitWidths(), map.getMatrix());
  auto equivalent = DistributedEncodingAttr::get(&context, renamed, good->getTopology(), good->getReplication());
  EXPECT_EQ(proveBroadcastLayout(equivalent, out, input, type, {1}).status, ProofStatus::Proven);

  // Extra source register replication is accessible, not an exact map-equality
  // requirement. Add a zero register column ahead of the original carriers.
  SmallVector<Attribute> names{b.getStringAttr("register")};
  llvm::append_range(names, map.getInputNames().getValue());
  SmallVector<int64_t> widths{1}; llvm::append_range(widths, map.getInputBitWidths().asArrayRef());
  auto shape = map.getMatrix().getType().getShape();
  SmallVector<APInt> bits;
  auto old = map.getMatrix().getValues<APInt>();
  for (int64_t row = 0; row < shape[0]; ++row) {
    bits.push_back(APInt(1, 0));
    for (int64_t col = 0; col < shape[1]; ++col) bits.push_back(old[row * shape[1] + col]);
  }
  auto replicatedMap = BitLinearLayoutMapAttr::get(&context, b.getArrayAttr(names), b.getDenseI64ArrayAttr(widths),
      map.getOutputNames(), map.getOutputBitWidths(), DenseIntElementsAttr::get(
        RankedTensorType::get({shape[0], shape[1] + 1}, b.getI1Type()), bits));
  SmallVector<int64_t> topology(good->getTopology().asArrayRef());
  ASSERT_EQ(topology[0], 1); topology[0] = 2;
  auto replicated = DistributedEncodingAttr::get(&context, replicatedMap, b.getDenseI64ArrayAttr(topology),
      b.getI64IntegerAttr(good->getReplication().getInt() * 2));
  EXPECT_EQ(proveBroadcastLayout(replicated, out, input, type, {1}).status, ProofStatus::Proven);
}

TEST_F(BroadcastLayoutTest, ShapeAndAxisValidation) {
  auto f = Float32Type::get(&context);
  auto src = RankedTensorType::get({4}, f), dst = RankedTensorType::get({4, 8}, f);
  EXPECT_TRUE(validBroadcastTypes(src, dst, {1}));
  EXPECT_FALSE(validBroadcastTypes(src, dst, {0}));
  EXPECT_FALSE(validBroadcastTypes(src, dst, {-1}));
  EXPECT_FALSE(validBroadcastTypes(src, dst, {2}));
  EXPECT_FALSE(validBroadcastTypes(src, dst, {1, 1}));
  EXPECT_FALSE(validBroadcastTypes(src, RankedTensorType::get({2, 4, 8}, f), {2, 0}));
  EXPECT_FALSE(validBroadcastTypes(RankedTensorType::get({}, f), dst, {0, 1}));
  EXPECT_TRUE(validBroadcastTypes(src, RankedTensorType::get({4, 1}, f), {1}));
  EXPECT_FALSE(validBroadcastTypes(src, RankedTensorType::get({4, 3}, f), {1}));
  EXPECT_FALSE(validBroadcastTypes(src, RankedTensorType::get({4, ShapedType::kDynamic}, f), {1}));
  EXPECT_FALSE(validBroadcastTypes(src, RankedTensorType::get({4, 8}, Float16Type::get(&context)), {1}));
}

TEST_F(BroadcastLayoutTest, NonzeroDependentRegistersAndReindexing) {
  auto out = cast<DistributedEncodingAttr>(parseAttribute(R"mlir(#frisk.distributed<map = #frisk.bit_linear<
      inputs = ["register", "lane"], input_bits = [3, 5],
      outputs = ["row", "column"], output_bits = [2, 2],
      matrix = dense<[[1,0,1,0,0,0,0,0], [0,1,1,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0]]> : tensor<4x8xi1>>,
      topology = [8,32,1,1,1], replication = 16>)mlir", &context));
  ASSERT_TRUE(out);
  auto input = RankedTensorType::get({4}, Float32Type::get(&context));
  auto output = RankedTensorType::get({4, 4}, input.getElementType());
  auto projected = projectBroadcastInputEncoding(out, input, output, {1});
  ASSERT_TRUE(succeeded(projected));
  EXPECT_EQ(projected->getTopology()[0], 4);
  auto map = cast<BitLinearLayoutMapAttr>(projected->getMap());
  auto shape = map.getMatrix().getType().getShape();
  auto values = map.getMatrix().getValues<APInt>();
  SmallVector<APInt> bits;
  for (int64_t row = 0; row < shape[0]; ++row)
    for (int64_t col = 0; col < shape[1]; ++col)
      bits.push_back(values[row * shape[1] + (col < 2 ? 1 - col : col)]);
  auto reindexedMap = BitLinearLayoutMapAttr::get(&context, map.getInputNames(), map.getInputBitWidths(),
      map.getOutputNames(), map.getOutputBitWidths(), DenseIntElementsAttr::get(map.getMatrix().getType(), bits));
  auto reindexed = DistributedEncodingAttr::get(&context, reindexedMap, projected->getTopology(), projected->getReplication());
  EXPECT_NE(reindexed, *projected);
  EXPECT_EQ(proveBroadcastLayout(reindexed, out, input, output, {1}).status, ProofStatus::Proven);
  auto m = simple("tensor<4xf32>", "tensor<4x4xf32>", "1"); ASSERT_TRUE(m);
  auto function = m->lookupSymbol<func::FuncOp>("broadcast");
  auto encoded = RankedTensorType::get({4}, input.getElementType(), reindexed);
  function.getArgument(0).setType(encoded);
  function.setType(Builder(&context).getFunctionType({encoded}, {}));
  m->walk([&](linalg::BroadcastOp op) {
    auto dst = RankedTensorType::get({4, 4}, input.getElementType(), out);
    op->getOperand(1).setType(dst); op->getResult(0).setType(dst);
    EXPECT_TRUE(coordinates(op));
  });
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m, *target)));
}

TEST_F(BroadcastLayoutTest, FailureReducerRejectsBroadcastPayload) {
  mlir::frisk::testing::LayoutFailureCase fixture;
  fixture.variables = {{"v", {}}};
  fixture.relations = {{"same", "same", "rule", "reason", {"v", "v"}, ""}};
  auto graph = mlir::frisk::testing::rebuildFailureGraph(fixture, context);
  ASSERT_TRUE(succeeded(graph));
  // Even an otherwise supported relation must not silently lose a foreign
  // payload in this deliberately narrower replay serialization format.
  graph->getConstraints().front().broadcast = BroadcastLayoutContract{};
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(mlir::frisk::testing::reduceFailingLayoutGraph(*graph,
      [](const LayoutConstraintGraph &) { return true; })));
}

TEST_F(BroadcastLayoutTest, ConflictingUseBindingAndGraphMetadataFailClosed) {
  auto m = simple("tensor<4xf32>", "tensor<4x4xf32>", "1"); ASSERT_TRUE(m);
  auto g = collectLayoutConstraints(*m, *target); ASSERT_TRUE(succeeded(g));
  LayoutConstraint *relation = nullptr;
  for (auto &c : g->getConstraints()) if (c.kind == ConstraintKind::BroadcastLayout) relation = &c;
  ASSERT_NE(relation, nullptr);
  auto src = relation->vars[0], dst = relation->vars[1];
  auto out = g->getVariable(dst).candidates.front().value;
  auto bad = projectBroadcastInputEncoding(cast<DistributedEncodingAttr>(out),
      cast<RankedTensorType>(g->getVariable(src).shapedType),
      cast<RankedTensorType>(g->getVariable(dst).shapedType), {0});
  ASSERT_TRUE(succeeded(bad));
  auto axes = relation->broadcast->axes;
  relation->broadcast->axes = Builder(&context).getDenseI64ArrayAttr({0});
  {
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(g->verifyInvariants(m->getLoc())));
  }
  relation->broadcast->axes = axes;
  g->getVariable(src).candidates = {{*bad, kInvalidProvenanceID, 0}};
  g->getVariable(dst).candidates = {{out, kInvalidProvenanceID, 0}};
  g->addConstraint(ConstraintKind::RequireEncoding, ConstraintStrength::Hard,
      {src}, relation->broadcast->source, "hard-broadcast-use", "conflicting ownership", *bad);
  ASSERT_TRUE(succeeded(g->finalize(m->getLoc())));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(propagateCommonToFixedPoint(*g)));
}

TEST_F(BroadcastLayoutTest, SourceUseConversionActualTamperingAndRollback) {
  auto m = simple("tensor<4xf32>", "tensor<4x4xf32>", "1"); ASSERT_TRUE(m);
  auto function = m->lookupSymbol<func::FuncOp>("broadcast");
  auto input = cast<RankedTensorType>(function.getArgument(0).getType());
  LayoutVar v; v.kind = LayoutKind::Distributed; v.shapedType = RankedTensorType::get({4, 4}, input.getElementType());
  SmallVector<LayoutCandidate> candidates; target->enumerateCandidates(v, candidates);
  auto out = cast<DistributedEncodingAttr>(candidates.front().value);
  auto bad = projectBroadcastInputEncoding(out, input, cast<RankedTensorType>(v.shapedType), {0});
  ASSERT_TRUE(succeeded(bad));
  auto encoded = RankedTensorType::get({4}, input.getElementType(), *bad);
  function.getArgument(0).setType(encoded);
  function.setType(Builder(&context).getFunctionType({encoded}, {}));
  m->walk([&](linalg::BroadcastOp op) {
    auto output = RankedTensorType::get({4, 4}, input.getElementType(), out);
    op->getResult(0).setType(output); op->getOperand(1).setType(output);
  });
  ASSERT_TRUE(succeeded(infer(*m)));
  linalg::BroadcastOp broadcast; m->walk([&](linalg::BroadcastOp op) { broadcast = op; });
  ASSERT_TRUE(broadcast->getOperand(0).getDefiningOp<ConvertLayoutOp>());
  EXPECT_TRUE(coordinates(broadcast));
  BroadcastObservedTarget observed;
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m, observed)));
  EXPECT_EQ(observed.generations, 0u);
  auto before = print(*m);
  auto g = collectLayoutConstraints(*m, *target); ASSERT_TRUE(succeeded(g));
  auto s = solveLayoutGraph(*g, *target); ASSERT_TRUE(succeeded(s));
  for (const auto &c : g->getConstraints())
    if (c.kind == ConstraintKind::BroadcastLayout) s->assignments[c.vars[0]] = *bad;
  {
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(materializeLayouts(*m, *g, *s)));
  }
  EXPECT_EQ(before, print(*m));
  broadcast->getOperand(0).setType(encoded);
  EXPECT_FALSE(coordinates(broadcast));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*m, observed)));
  EXPECT_EQ(observed.generations, 0u);
}

TEST_F(BroadcastLayoutTest, StableSoftmaxRealLoadStoreAndThreadInheritance) {
  for (auto shape : {std::array<int64_t, 3>{4, 8, 32}, {8, 16, 128}}) {
    auto row = std::to_string(shape[0]);
    auto tile = row + "x" + std::to_string(shape[1]);
    auto tensor = "tensor<" + tile + "xf32>";
    auto vector = "tensor<" + row + "xf32>";
    auto memory = "memref<" + tile + "xf32,1>";
    auto m = parseSourceString<ModuleOp>(
        "module { func.func @softmax(%src:" + memory + ", %dst:" + memory + ") { "
        "%sv=frisk.layout_view %src : " + memory + " -> " + memory +
        " %dv=frisk.layout_view %dst : " + memory + " -> " + memory +
        " %x=frisk.tile_load %sv : " + memory + " -> " + tensor +
        " %max=frisk.reduce_tensor %x {kind=\"max\",dim=1:i64,frisk.execution_threads=" +
        std::to_string(shape[2]) + ":i64} : " + tensor + " -> " + vector +
        " %e0=tensor.empty():" + tensor +
        " %bm=linalg.broadcast ins(%max:" + vector + ") outs(%e0:" + tensor + ") dimensions=[1] "
        "%center=arith.subf %x,%bm:" + tensor + " %exp=math.exp %center:" + tensor +
        " %sum=frisk.reduce_tensor %exp {kind=\"sum\",dim=1:i64} : " + tensor + " -> " + vector +
        " %e1=tensor.empty():" + tensor +
        " %bs=linalg.broadcast ins(%sum:" + vector + ") outs(%e1:" + tensor + ") dimensions=[1] "
        "%y=arith.divf %exp,%bs:" + tensor +
        " frisk.tile_store %y,%dv:" + tensor + "," + memory + " return } }", &context);
    ASSERT_TRUE(m);
    ASSERT_TRUE(succeeded(infer(*m)));
    m->walk([&](linalg::BroadcastOp op) { EXPECT_TRUE(coordinates(op)); });
    m->walk([&](ReduceTensorOp op) {
      EXPECT_EQ(op->getAttrOfType<IntegerAttr>("frisk.execution_threads").getInt(), shape[2]);
    });
    BroadcastObservedTarget observed;
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m, observed)));
    EXPECT_EQ(observed.generations, 0u);
    auto once = print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once, print(*m));
    auto reparsed = parseSourceString<ModuleOp>(once, &context); ASSERT_TRUE(reparsed);
    ASSERT_TRUE(succeeded(infer(*reparsed))); EXPECT_EQ(once, print(*reparsed));
  }
}
} // namespace

namespace {
TEST_F(BroadcastLayoutTest, UnitAxesInferenceReplayAndActualOnly) {
  for (auto spec : {std::array<StringRef,3>{"tensor<1xf32>","tensor<1x8xf32>","1"},
      {"tensor<8xf32>","tensor<8x1xf32>","1"},
      {"tensor<1xf32>","tensor<1x1xf32>","0"},
      {"tensor<8xf32>","tensor<1x8x1xf32>","0,2"}}) {
    auto m=simple(spec[0],spec[1],spec[2]); ASSERT_TRUE(m);
    ASSERT_TRUE(succeeded(infer(*m)));
    m->walk([&](linalg::BroadcastOp op){EXPECT_TRUE(coordinates(op));});
    BroadcastObservedTarget observed;
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*m,observed)));
    EXPECT_EQ(observed.generations,0u);
    auto once=print(*m); ASSERT_TRUE(succeeded(infer(*m))); EXPECT_EQ(once,print(*m));
    auto parsed=parseSourceString<ModuleOp>(once,&context); ASSERT_TRUE(parsed);
    ASSERT_TRUE(succeeded(infer(*parsed))); EXPECT_EQ(once,print(*parsed));
  }
}
} // namespace
