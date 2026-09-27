#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class ObservedTarget : public LayoutTarget {
public:
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  mutable unsigned enumerations = 0, preparations = 0, builds = 0, proofs = 0;
  void enumerateCandidates(const LayoutVar &var, SmallVectorImpl<LayoutCandidate> &out) const override {
    ++enumerations; target->enumerateCandidates(var, out);
  }
  LogicalResult verifyCandidate(const LayoutVar &var, Attribute attr, Location loc) const override {
    return target->verifyCandidate(var, attr, loc);
  }
  FailureOr<CostVector> evaluate(const CandidateAssignment &a) const override { return target->evaluate(a); }
  LogicalResult prepareInstructionCandidates(LayoutConstraintGraph &g) const override {
    ++preparations; return target->prepareInstructionCandidates(g);
  }
  FailureOr<Attribute> buildInstructionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, ArrayRef<Attribute> e) const override {
    ++builds; return target->buildInstructionContract(g, c, e);
  }
  LayoutProof verifyInstructionContract(const LayoutConstraintGraph &g,
      const LayoutConstraint &c, ArrayRef<Attribute> e, Attribute binding) const override {
    ++proofs; return target->verifyInstructionContract(g, c, e, binding);
  }
};
class GemmLayoutIntegrationTest : public testing::Test {
protected:
  MLIRContext context;
  ObservedTarget target;
  GemmLayoutIntegrationTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect,
                        memref::MemRefDialect>();
  }
  OwningOpRef<ModuleOp> make(bool rs, StringRef dtype = "f16", int m = 64,
                           int n = 64, int k = 16, bool ta = false, bool tb = false) {
    std::string text;
    llvm::raw_string_ostream s(text);
    auto shape = [&](int x, int y, StringRef element) {
      return std::to_string(x) + "x" + std::to_string(y) + "x" + element.str();
    };
    auto aShape = shape(ta ? k : m, ta ? m : k, dtype);
    auto bShape = shape(tb ? n : k, tb ? k : n, dtype);
    auto cShape = shape(m, n, "f32");
    std::string aType = rs ? "tensor<" + aShape + ">" : "memref<" + aShape + ", 3>";
    std::string bType = "memref<" + bShape + ", 3>", cType = "tensor<" + cShape + ">";
    s << "module attributes {frisk.target = \"sm_90a\"} { func.func @test() {\n";
    if (rs) s << "%a = arith.constant dense<1.0> : " << aType << "\n";
    else {
      s << "%ar = memref.alloc() {alignment = 1024 : i64} : " << aType << "\n";
      s << "%a = frisk.layout_view %ar : " << aType << " -> " << aType << "\n";
    }
    s << "%br = memref.alloc() {alignment = 1024 : i64} : " << bType << "\n";
    s << "%b = frisk.layout_view %br : " << bType << " -> " << bType << "\n";
    s << "%init = arith.constant dense<0.0> : " << cType << "\n";
    s << "%r = \"frisk.mma\"(%a, %b, %init) {m = " << m << " : i64, n = " << n
      << " : i64, k = " << k << " : i64, trans_a = " << (ta ? "true" : "false")
      << ", trans_b = " << (tb ? "true" : "false") << "} : (" << aType << ", "
      << bType << ", " << cType << ") -> " << cType << "\n return } }";
    return parseSourceString<ModuleOp>(text, &context);
  }
  LogicalResult infer(ModuleOp module) {
    auto graph = collectLayoutConstraints(module, target);
    if (failed(graph) || failed(propagateCommonToFixedPoint(*graph))) return failure();
    auto solution = solveLayoutGraph(*graph, target);
    if (failed(solution)) return failure();
    return materializeLayouts(module, *graph, *solution);
  }
  MmaOp mma(ModuleOp module) {
    MmaOp result;
    module.walk([&](MmaOp op) { result = op; });
    return result;
  }
  std::string print(ModuleOp module) {
    std::string text; llvm::raw_string_ostream s(text); module.print(s); return text;
  }
};

TEST_F(GemmLayoutIntegrationTest, SSAndRSBothDtypesAndTransposeCombinations) {
  for (bool rs : {false, true})
    for (StringRef dtype : {"f16", "bf16"})
      for (bool ta : {false, true})
        for (bool tb : {false, true}) {
          SCOPED_TRACE(::testing::Message() << "rs=" << rs << " dtype=" << dtype.str()
                       << " ta=" << ta << " tb=" << tb);
          auto module = make(rs, dtype, 64, 64, 16, ta, tb);
          ASSERT_TRUE(module);
          ASSERT_TRUE(succeeded(infer(*module)));
          EXPECT_TRUE(mma(*module)->hasAttr("frisk.mma_contract"));
          EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, target)));
        }
}

TEST_F(GemmLayoutIntegrationTest, LargeTileReplayAndActualOnlyNoEnumeration) {
  for (bool rs : {false, true}) for (StringRef dtype : {"f16", "bf16"}) {
    auto module = make(rs, dtype, 128, 128, 64);
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(infer(*module)));
    auto once = print(*module);
    ObservedTarget actualOnly;
    ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, actualOnly)));
    EXPECT_EQ(actualOnly.enumerations, 0u);
    EXPECT_EQ(actualOnly.preparations, 0u);
    EXPECT_EQ(actualOnly.builds, 0u);
    EXPECT_GT(actualOnly.proofs, 0u);
    ASSERT_TRUE(succeeded(infer(*module)));
    EXPECT_EQ(once, print(*module));
    auto replay = parseSourceString<ModuleOp>(once, &context);
    ASSERT_TRUE(replay);
    ASSERT_TRUE(succeeded(infer(*replay)));
    EXPECT_EQ(once, print(*replay));
  }
}

TEST_F(GemmLayoutIntegrationTest, MissingAndTamperedContractsAreRejected) {
  auto module = make(false);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  auto op = mma(*module);
  Attribute contract = op->removeAttr("frisk.mma_contract");
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
  op->setAttr("frisk.mma_contract", contract);
  Attribute threads = op->removeAttr("frisk.execution_threads");
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
  op->setAttr("frisk.execution_threads", threads);
  Builder b(&context);
  op->setAttr("frisk.execution_threads", b.getI64IntegerAttr(256));
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
  op->setAttr("frisk.execution_threads", threads);
  module->getOperation()->setAttr("frisk.target", b.getStringAttr("sm_90"));
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, target)));
}

TEST_F(GemmLayoutIntegrationTest, InvalidSolutionLeavesOriginalIRUnchanged) {
  auto module = make(true);
  ASSERT_TRUE(module);
  auto before = print(*module);
  auto graph = collectLayoutConstraints(*module, target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveLayoutGraph(*graph, target);
  ASSERT_TRUE(succeeded(solution));
  solution->instructionBindings.clear();
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(materializeLayouts(*module, *graph, *solution)));
  EXPECT_EQ(before, print(*module));
}

TEST_F(GemmLayoutIntegrationTest, ExplicitProducersRemainUnchangedWithConsumerConversions) {
  auto module = make(true);
  ASSERT_TRUE(module);
  SmallVector<Attribute> originals;
  module->walk([&](arith::ConstantOp constant) {
    auto type = cast<RankedTensorType>(constant.getType());
    LayoutVar var;
    var.kind = LayoutKind::Distributed;
    var.shapedType = type;
    var.requiredThreads = 128;
    SmallVector<LayoutCandidate> candidates;
    target.enumerateCandidates(var, candidates);
    ASSERT_FALSE(candidates.empty());
    auto encoded = RankedTensorType::get(type.getShape(), type.getElementType(), candidates[0].value);
    originals.push_back(candidates[0].value);
    constant.getResult().setType(encoded);
    constant->setAttr("value", cast<DenseElementsAttr>(constant.getValue()).reshape(encoded));
  });
  ASSERT_EQ(originals.size(), 2u);
  ASSERT_TRUE(succeeded(infer(*module)));
  unsigned index = 0, conversions = 0;
  module->walk([&](arith::ConstantOp constant) {
    EXPECT_EQ(cast<RankedTensorType>(constant.getType()).getEncoding(), originals[index++]);
  });
  module->walk([&](ConvertLayoutOp) { ++conversions; });
  EXPECT_EQ(conversions, 2u);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, target)));
  auto once = print(*module);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_EQ(once, print(*module));
}

TEST_F(GemmLayoutIntegrationTest, TwoMmasShareFiniteStorageDomains) {
  auto module = make(false);
  ASSERT_TRUE(module);
  auto op = mma(*module);
  OpBuilder builder(&context);
  builder.setInsertionPointAfter(op);
  builder.clone(*op);
  auto graph = collectLayoutConstraints(*module, target);
  ASSERT_TRUE(succeeded(graph));
  for (const auto &var : graph->getVariables()) EXPECT_LE(var.candidates.size(), 4u);
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveLayoutGraph(*graph, target);
  ASSERT_TRUE(succeeded(solution));
  EXPECT_EQ(solution->instructionBindings.size(), 2u);
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, target)));
}

TEST_F(GemmLayoutIntegrationTest, RejectsMissingTargetAndInsufficientRootAlignment) {
  auto module = make(false);
  ASSERT_TRUE(module);
  module->getOperation()->removeAttr("frisk.target");
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(infer(*module)));
  module = make(false);
  ASSERT_TRUE(module);
  module->walk([&](memref::AllocOp op) { op->removeAttr("alignment"); });
  auto before = print(*module);
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_EQ(before, print(*module));
}

TEST_F(GemmLayoutIntegrationTest, RejectsUnsupportedTargetDtypesAndOperandForms) {
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  Builder b(&context);
  for (StringRef name : {"sm_90", "sm_80"}) {
    auto module = make(false);
    ASSERT_TRUE(module);
    module->getOperation()->setAttr("frisk.target", b.getStringAttr(name));
    EXPECT_TRUE(failed(infer(*module)));
  }
  auto module = make(false, "f32");
  ASSERT_TRUE(module);
  EXPECT_TRUE(failed(infer(*module)));
  module = make(false);
  ASSERT_TRUE(module);
  std::string halfAccumulator = print(*module);
  size_t position = 0;
  while ((position = halfAccumulator.find("xf32", position)) != std::string::npos)
    halfAccumulator.replace(position, 4, "xf16");
  auto half = parseSourceString<ModuleOp>(halfAccumulator, &context);
  ASSERT_TRUE(half);
  EXPECT_TRUE(failed(infer(*half)));

  // Tensor B is mathematically expressible, but there is no WGMMA register B.
  module = make(true);
  ASSERT_TRUE(module);
  OpBuilder builder(&context);
  auto op = mma(*module);
  builder.setInsertionPoint(op);
  auto bt = RankedTensorType::get({16, 64}, b.getF16Type());
  auto value = DenseElementsAttr::get(bt, ArrayRef<Attribute>{b.getF16FloatAttr(1.0)});
  auto tensorB = builder.create<arith::ConstantOp>(op.getLoc(), bt, value);
  op->setOperand(1, tensorB);
  ASSERT_TRUE(succeeded(verify(*module)));
  EXPECT_TRUE(failed(infer(*module)));

  // Neither A nor B may reinterpret a Local MemRef as a Tensor fragment.
  for (unsigned role : {0u, 1u}) {
    module = make(false);
    ASSERT_TRUE(module);
    op = mma(*module);
    builder.setInsertionPoint(op);
    auto original = cast<MemRefType>(op->getOperand(role).getType());
    auto localType = MemRefType::get(original.getShape(), original.getElementType(), {},
                                    unsigned(attr::MemorySpace::Local));
    auto allocation = builder.create<memref::AllocOp>(op.getLoc(), localType);
    auto view = builder.create<LayoutViewOp>(op.getLoc(), localType, allocation, StorageLayoutAttr());
    op->setOperand(role, view);
    ASSERT_TRUE(succeeded(verify(*module)));
    EXPECT_TRUE(failed(infer(*module)));
  }
}

TEST_F(GemmLayoutIntegrationTest, ParallelThreadScopeIsPreservedAndConflictsRejected) {
  for (bool conflict : {false, true}) {
    auto module = make(false);
    ASSERT_TRUE(module);
    auto op = mma(*module);
    OpBuilder builder(&context);
    builder.setInsertionPoint(op);
    auto kernel = builder.create<KernelOp>(op.getLoc(), "scope",
                                          builder.getFunctionType({}, {}));
    Block *entry = kernel.addEntryBlock();
    builder.setInsertionPoint(entry->getTerminator());
    auto parallel = builder.create<ParallelOp>(op.getLoc(), ArrayRef<int64_t>{64, 64}, 128);
    // Parallel's custom builder creates the region, not its block or terminator.
    builder.createBlock(&parallel.getRegion(), {},
        {builder.getIndexType(), builder.getIndexType()}, {op.getLoc(), op.getLoc()});
    auto end = builder.create<EndOp>(op.getLoc());
    op->moveBefore(end);
    op->setAttr("frisk.execution_threads", builder.getI64IntegerAttr(conflict ? 256 : 128));
    ASSERT_TRUE(succeeded(verify(*module)));
    if (conflict) {
      auto before = print(*module);
      ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
      EXPECT_TRUE(failed(infer(*module)));
      EXPECT_EQ(before, print(*module));
    } else {
      ASSERT_TRUE(succeeded(infer(*module)));
      EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, target)));
    }
  }
}

TEST_F(GemmLayoutIntegrationTest, ActualOnlyRejectsDescriptorPackingGridAndFragmentTampering) {
  auto module = make(true);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(infer(*module)));
  auto op = mma(*module);
  auto original = op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
  ASSERT_TRUE(original);
  Builder b(&context);
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  ObservedTarget actualOnly;
  for (unsigned kind = 0; kind < 3; ++kind) {
    NamedAttrList fields(original.getPayload());
    if (kind == 0) fields.set("packing", b.getStringAttr("none"));
    if (kind == 1) fields.set("grid", b.getDenseI64ArrayAttr({2, 1}));
    if (kind == 2) {
      auto descriptor = original.getPayload().getAs<MmaDescriptorPlanAttr>("b_descriptor");
      NamedAttrList plan(descriptor.getPayload());
      plan.set("stride", b.getI64IntegerAttr(descriptor.getPayload().getAs<IntegerAttr>("stride").getInt() + 16));
      fields.set("b_descriptor", MmaDescriptorPlanAttr::get(&context, plan.getDictionary(&context)));
    }
    auto payload = fields.getDictionary(&context);
    if (kind == 0) {
      // Invalid RS packing is rejected by the typed schema itself. Use the
      // checked constructor (get asserts on malformed attributes), then test
      // that raw dictionary injection cannot bypass that schema in actual IR.
      EXPECT_FALSE(MmaInstructionContractAttr::getChecked(
          [&] { return emitError(op.getLoc()); }, &context, payload));
      op->setAttr("frisk.mma_contract", payload);
    } else {
      op->setAttr("frisk.mma_contract", MmaInstructionContractAttr::get(&context, payload));
    }
    EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, actualOnly))) << "tamper kind=" << kind;
    op->setAttr("frisk.mma_contract", original);
  }
  // A bijective row-bit permutation remains a valid distributed map, but no
  // longer has the instruction's exact per-register / packed-half ordering.
  auto constant = op.getA().getDefiningOp<arith::ConstantOp>();
  ASSERT_TRUE(constant);
  auto type = cast<RankedTensorType>(constant.getType());
  auto encoding = cast<DistributedEncodingAttr>(type.getEncoding());
  auto map = cast<BitLinearLayoutMapAttr>(encoding.getMap());
  SmallVector<APInt> matrix(map.getMatrix().getValues<APInt>());
  auto matrixType = cast<RankedTensorType>(map.getMatrix().getType());
  for (int64_t col = 0; col < matrixType.getDimSize(1); ++col)
    std::swap(matrix[col], matrix[col + matrixType.getDimSize(1)]);
  auto wrongMap = BitLinearLayoutMapAttr::get(&context, map.getInputNames(), map.getInputBitWidths(),
      map.getOutputNames(), map.getOutputBitWidths(), DenseIntElementsAttr::get(matrixType, matrix));
  auto wrong = DistributedEncodingAttr::get(&context, wrongMap, encoding.getTopology(), encoding.getReplication());
  auto wrongType = RankedTensorType::get(type.getShape(), type.getElementType(), wrong);
  ASSERT_TRUE(succeeded(wrong.verifyForType(wrongType, op.getLoc())));
  constant.getResult().setType(wrongType);
  constant->setAttr("value", cast<DenseElementsAttr>(constant.getValue()).reshape(wrongType));
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, actualOnly)));
  EXPECT_EQ(actualOnly.enumerations, 0u);
  EXPECT_EQ(actualOnly.preparations, 0u);
  EXPECT_EQ(actualOnly.builds, 0u);
}

TEST_F(GemmLayoutIntegrationTest, ProofBudgetFailureIsUnknown) {
  auto module = make(true, "f16", 1024, 128, 16);
  ASSERT_TRUE(module);
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(message);
    diagnostic.print(stream);
    return success();
  });
  EXPECT_TRUE(failed(infer(*module)));
  EXPECT_NE(message.find("unknown"), std::string::npos) << message;
  EXPECT_NE(message.find("budget"), std::string::npos) << message;
}
} // namespace
