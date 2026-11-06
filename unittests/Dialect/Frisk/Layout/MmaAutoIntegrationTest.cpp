#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class MmaAutoIntegrationTest : public testing::Test {
protected:
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  MmaAutoIntegrationTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, arith::ArithDialect, memref::MemRefDialect>();
  }
  OwningOpRef<ModuleOp> make(bool rs = false, int m = 128, int n = 128,
                           int threads = 256, StringRef policy = "Auto") {
    std::string text;
    llvm::raw_string_ostream s(text);
    std::string a = (rs ? "tensor<" : "memref<") + std::to_string(m) + "x16xf16" + (rs ? ">" : ",3>");
    std::string b = "memref<16x" + std::to_string(n) + "xf16,3>";
    std::string c = "tensor<" + std::to_string(m) + "x" + std::to_string(n) + "xf32>";
    s << "module attributes {frisk.target=\"sm_90a\"} { func.func @test() {\n";
    if (rs) s << "%a=arith.constant dense<1.0> : " << a << "\n";
    else s << "%ar=memref.alloc() {alignment=1024:i64} : " << a << "\n%a=frisk.layout_view %ar : " << a << " -> " << a << "\n";
    s << "%br=memref.alloc() {alignment=1024:i64} : " << b << "\n%b=frisk.layout_view %br : " << b << " -> " << b
      << "\n%c=arith.constant dense<0.0> : " << c << "\n%r=\"frisk.mma\"(%a,%b,%c) {m=" << m
      << ":i64,n=" << n << ":i64,k=16:i64,frisk.execution_threads=" << threads
      << ":i64,policy=#frisk<gemm_warp_policy " << policy << ">} : (" << a << "," << b << "," << c << ") -> " << c << "\nreturn } }";
    return parseSourceString<ModuleOp>(text, &context);
  }
  MmaOp mma(ModuleOp m) { MmaOp result; m.walk([&](MmaOp op) { result = op; }); return result; }
  std::string print(ModuleOp m) { std::string s; llvm::raw_string_ostream(s) << m; return s; }
  LogicalResult infer(ModuleOp m) {
    auto graph = collectLayoutConstraints(m,*target);
    if (failed(graph) || failed(propagateCommonToFixedPoint(*graph))) return failure();
    auto solution = solveLayoutGraph(*graph,*target);
    if (failed(solution)) return failure();
    return materializeLayouts(m,*graph,*solution);
  }
  LayoutConstraint &instruction(LayoutConstraintGraph &graph) {
    return *llvm::find_if(graph.getConstraints(),[](const auto &c) { return bool(c.instruction); });
  }
  DistributedEncodingAttr reordered(DistributedEncodingAttr e) {
    auto map = cast<BitLinearLayoutMapAttr>(e.getMap());
    SmallVector<Attribute> names;
    SmallVector<int64_t> widths, columns;
    int64_t offset = map.getMatrix().getType().getShape()[1];
    for (int i = map.getInputNames().size()-1; i >= 0; --i) {
      names.push_back(map.getInputNames()[i]); widths.push_back(map.getInputBitWidths()[i]);
      offset -= widths.back();
      for (int64_t j = 0; j < widths.back(); ++j) columns.push_back(offset+j);
    }
    SmallVector<APInt> old(map.getMatrix().getValues<APInt>()), bits;
    for (int64_t row = 0; row < map.getMatrix().getType().getShape()[0]; ++row)
      for (auto col : columns) bits.push_back(old[row*columns.size()+col]);
    Builder b(&context);
    auto changed = BitLinearLayoutMapAttr::get(&context,b.getArrayAttr(names),b.getDenseI64ArrayAttr(widths),
        map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(map.getMatrix().getType(),bits));
    return DistributedEncodingAttr::get(&context,changed,e.getTopology(),e.getReplication());
  }
};

TEST_F(MmaAutoIntegrationTest, JointFiveRolesSelectCheaperGeometryAndReplay) {
  for (bool rs : {false, true}) {
    auto module = make(rs); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    const LayoutConstraint *instruction = nullptr;
    for (const auto &c : graph->getConstraints()) if (c.instruction) instruction = &c;
    ASSERT_TRUE(instruction); ASSERT_EQ(instruction->vars.size(), 5u);
    auto &geometry = graph->getVariable(instruction->vars[4]);
    EXPECT_EQ(geometry.kind, LayoutKind::Instruction);
    EXPECT_EQ(geometry.candidates.size(), 2u);
    ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
    auto solution = solveLayoutGraph(*graph, *target); ASSERT_TRUE(succeeded(solution));
    auto selected = cast<MmaInstructionContractAttr>(solution->instructionBindings.lookup(instruction->id));
    EXPECT_EQ(selected.getPayload().getAs<DenseI64ArrayAttr>("grid").asArrayRef(), (ArrayRef<int64_t>{2,1}));
    ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
    ASSERT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
    auto once = print(*module);
    auto replay = parseSourceString<ModuleOp>(once, &context); ASSERT_TRUE(replay);
    ASSERT_TRUE(succeeded(infer(*replay))); EXPECT_EQ(once, print(*replay));
  }
}

TEST_F(MmaAutoIntegrationTest, SameDomainExactOracleAndLegacyCosts) {
  for (StringRef policy : {"Auto","Square","FullRow","FullCol"}) {
    auto module = make(false,128,128,256,policy); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
    auto &c = instruction(*graph);
    EXPECT_EQ(graph->getVariable(c.vars[4]).candidates.size(),policy == "Auto" ? 2u : 1u);
    ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
    auto solution = solveLayoutGraph(*graph,*target); ASSERT_TRUE(succeeded(solution));
    // Independent exhaustive traversal of the same frozen domains, using hard
    // relations rather than production search ordering/pruning.
    CandidateAssignment current;
    std::optional<CostEstimate> best;
    std::function<void(unsigned)> visit = [&](unsigned index) {
      if (index == graph->getVariables().size()) {
        for (auto &relation : graph->getConstraints())
          if (relation.strength == ConstraintStrength::Hard &&
              !satisfiesLayoutConstraint(*graph,relation,current.values,true)) return;
        auto cost = target->evaluate(*graph,current); ASSERT_TRUE(succeeded(cost));
        auto choices = chooseLayoutConversions(*graph,current,*target); ASSERT_TRUE(succeeded(choices));
        *cost = addLayoutCosts(*cost,choices->cost);
        if (!best || cost->cost < best->cost) best = *cost;
        return;
      }
      auto &var = graph->getVariable(index);
      for (auto &candidate : var.candidates) {
        current.values[var.id] = candidate.value; visit(index+1);
      }
      current.values.erase(var.id);
    };
    visit(0); ASSERT_TRUE(best);
    EXPECT_FALSE(solution->cost.cost < best->cost);
    EXPECT_FALSE(best->cost < solution->cost.cost);
    for (const auto &tuple : c.instruction->tuples) {
      CandidateAssignment local;
      for (auto [id,encoding] : llvm::zip_equal(c.vars,tuple.encodings)) local.values[id] = encoding;
      auto cost = target->evaluate(*graph,local); ASSERT_TRUE(succeeded(cost));
      auto binding = cast<MmaInstructionContractAttr>(tuple.binding);
      auto grid = binding.getPayload().getAs<DenseI64ArrayAttr>("grid");
      EXPECT_EQ(cost->cost.instructionPathAndWork, grid[0] == 1 ? 4u : 2u);
    }
  }
}

TEST_F(MmaAutoIntegrationTest, ExplicitNonDefaultGeometryAndTamperRollback) {
  auto module = make(true); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
  auto &c = instruction(*graph);
  Attribute fixed;
  for (auto &tuple : c.instruction->tuples) {
    auto binding = cast<MmaInstructionContractAttr>(tuple.binding);
    if (binding.getPayload().getAs<DenseI64ArrayAttr>("grid")[0] == 2) { fixed = binding; break; }
  }
  ASSERT_TRUE(fixed); mma(*module)->setAttr("frisk.mma_contract",fixed);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_EQ(mma(*module)->getAttr("frisk.mma_contract"),fixed);
  auto once = print(*module); ASSERT_TRUE(succeeded(infer(*module))); EXPECT_EQ(once,print(*module));
  for (StringRef field : {"grid","repeats","packing","descriptor","policy","threads"}) {
    SCOPED_TRACE(field.str());
    auto copy = parseSourceString<ModuleOp>(once,&context); ASSERT_TRUE(copy);
    auto op = mma(*copy); Builder b(&context);
    NamedAttrList fields(cast<MmaInstructionContractAttr>(fixed).getPayload());
    if (field == "policy") op.setPolicy(attr::GemmWarpPolicy::FullCol);
    else if (field == "threads") op->setAttr("frisk.execution_threads",b.getI64IntegerAttr(128));
    else {
      if (field == "grid") fields.set("grid",b.getDenseI64ArrayAttr({2,2}));
      if (field == "repeats") fields.set("repeats",b.getDenseI64ArrayAttr({2,1,1}));
      if (field == "packing") fields.set("packing",b.getStringAttr("none"));
      if (field == "descriptor") {
        auto descriptor = cast<MmaInstructionContractAttr>(fixed).getPayload().getAs<MmaDescriptorPlanAttr>("b_descriptor");
        NamedAttrList p(descriptor.getPayload());
        p.set("stride",b.getI64IntegerAttr(descriptor.getPayload().getAs<IntegerAttr>("stride").getInt()+16));
        fields.set("b_descriptor",MmaDescriptorPlanAttr::get(&context,p.getDictionary(&context)));
      }
      ScopedDiagnosticHandler quiet(&context,[](Diagnostic &) { return success(); });
      auto attr = MmaInstructionContractAttr::getChecked([&] { return op.emitError(); },&context,fields.getDictionary(&context));
      op->setAttr("frisk.mma_contract",attr ? Attribute(attr) : Attribute(fields.getDictionary(&context)));
    }
    auto before = print(*copy);
    ScopedDiagnosticHandler quiet(&context,[](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(verifyMaterializedLayouts(*copy,*target)));
    EXPECT_TRUE(failed(infer(*copy))); EXPECT_EQ(before,print(*copy));
  }
}

TEST_F(MmaAutoIntegrationTest, ExplicitMoreExpensiveGeometryCannotBeReplaced) {
  auto module = make(); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
  auto &c = instruction(*graph);
  Attribute fixed;
  for (auto &tuple : c.instruction->tuples)
    if (cast<MmaInstructionContractAttr>(tuple.binding).getPayload().getAs<DenseI64ArrayAttr>("grid")[0] == 1) {
      fixed = tuple.binding; break;
    }
  ASSERT_TRUE(fixed); mma(*module)->setAttr("frisk.mma_contract",fixed);
  ASSERT_TRUE(succeeded(infer(*module)));
  EXPECT_EQ(mma(*module)->getAttr("frisk.mma_contract"),fixed);
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,*target)));
}

TEST_F(MmaAutoIntegrationTest, ExplicitReplicatedRsHardwareBudgetIsUnknown) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module attributes {frisk.target="sm_90a"} {
      func.func @test(%a:tensor<128x512xf16>, %c:tensor<128x128xf32>) {
        %root=memref.alloc() {alignment=1024:i64} : memref<512x128xf16,3>
        %b=frisk.layout_view %root : memref<512x128xf16,3> -> memref<512x128xf16,3>
        %r="frisk.mma"(%a,%b,%c) {m=128:i64,n=128:i64,k=512:i64,
          frisk.execution_threads=256:i64,policy=#frisk<gemm_warp_policy Auto>} :
          (tensor<128x512xf16>,memref<512x128xf16,3>,tensor<128x128xf32>)->tensor<128x128xf32>
        return
      }
    })mlir", &context);
  ASSERT_TRUE(module);
  auto op = mma(*module); Builder b(&context);
  SM90MmaGeometry g{128,128,512,256,1,2,64,2,1,32};
  // Build the linear hardware map from its basis, without asking the bounded
  // production fragment builder to enumerate 131072 hardware points.
  SmallVector<APInt> matrix(16*17, APInt(1,0));
  const int widths[] = {9,5,2,1};
  unsigned column = 0;
  for (unsigned axis = 0; axis < 4; ++axis)
    for (int bit = 0; bit < widths[axis]; ++bit, ++column) {
      int64_t point[4] = {}; point[axis] = int64_t{1} << bit;
      auto coordinate = getSM90MmaFragmentCoordinate(g,true,false,point[0],point[1],point[2],point[3]);
      uint64_t packed = coordinate[0] | (coordinate[1] << 7);
      for (unsigned row = 0; row < 16; ++row) matrix[row*17+column] = APInt(1,(packed>>row)&1);
    }
  auto map = BitLinearLayoutMapAttr::get(&context,
      b.getStrArrayAttr({"register","lane","warp","warp_group"}),b.getDenseI64ArrayAttr({9,5,2,1}),
      b.getStrArrayAttr({"dim0","dim1"}),b.getDenseI64ArrayAttr({7,9}),
      DenseIntElementsAttr::get(RankedTensorType::get({16,17},b.getI1Type()),matrix));
  auto a = DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({512,32,4,2,1}),b.getI64IntegerAttr(2));
  auto accumulator = buildSM90MmaFragment(&context,g,false,false); ASSERT_TRUE(succeeded(accumulator));
  op.getA().setType(RankedTensorType::get({128,512},b.getF16Type(),a));
  op.getInit().setType(RankedTensorType::get({128,128},b.getF32Type(),*accumulator));
  op.getResult().setType(op.getInit().getType());
  op->getParentOfType<func::FuncOp>().setType(b.getFunctionType({op.getA().getType(),op.getInit().getType()},{}));
  auto k = b.getAffineDimExpr(0), mn = b.getAffineDimExpr(1);
  auto address = (mn%8)*2+mn.floorDiv(8)*128+(k%8)*16+k.floorDiv(8)*2048;
  auto storageMap = AffineLayoutMapAttr::get(&context,b.getStrArrayAttr({"dim0","dim1"}),
      b.getDenseI64ArrayAttr({512,128}),b.getStrArrayAttr({"byte_offset","bit_offset"}),
      b.getDenseI64ArrayAttr({131072,8}),AffineMapAttr::get(AffineMap::get(2,0,{address,b.getAffineConstantExpr(0)},&context)));
  auto storage = StorageLayoutAttr::get(&context,storageMap,MemorySpaceAttr::get(&context,attr::MemorySpace::Shared),
      b.getI64IntegerAttr(1024),b.getI64IntegerAttr(1));
  cast<LayoutViewOp>(op.getB().getDefiningOp()).setLayoutAttr(storage);
  SmallVector<int64_t> entries;
  for (int64_t n : {0,64}) for (int64_t k = 0; k < 512; k += 16) entries.append({n,k,16*n+256*k,0});
  auto descriptor = MmaDescriptorPlanAttr::get(&context,b.getDictionaryAttr({
      b.getNamedAttr("major",b.getStringAttr("mn")),b.getNamedAttr("swizzle",b.getI64IntegerAttr(0)),
      b.getNamedAttr("leading",b.getI64IntegerAttr(2048)),b.getNamedAttr("stride",b.getI64IntegerAttr(128)),
      b.getNamedAttr("entries",b.getDenseI64ArrayAttr(entries))}));
  auto fields = b.getDictionaryAttr({b.getNamedAttr("version",b.getI64IntegerAttr(1)),
      b.getNamedAttr("target",b.getStringAttr("sm_90a")),b.getNamedAttr("form",b.getStringAttr("rs")),
      b.getNamedAttr("input_type",TypeAttr::get(b.getF16Type())),b.getNamedAttr("accumulator_type",TypeAttr::get(b.getF32Type())),
      b.getNamedAttr("atom",b.getDenseI64ArrayAttr({64,64,16})),b.getNamedAttr("grid",b.getDenseI64ArrayAttr({1,2})),
      b.getNamedAttr("repeats",b.getDenseI64ArrayAttr({2,1,32})),b.getNamedAttr("packing",b.getStringAttr("f16x2-low-high")),
      b.getNamedAttr("b_descriptor",descriptor)});
  auto binding = MmaInstructionContractAttr::getChecked([&] { return op.emitError(); },&context,fields);
  ASSERT_TRUE(binding); op->setAttr("frisk.mma_contract",binding);
  auto graph = collectLayoutConstraints(*module,*target,LayoutCollectionMode::RelationsOnly); ASSERT_TRUE(succeeded(graph));
  auto &c = instruction(*graph);
  SmallVector<Attribute> encodings{a,storage,*accumulator,*accumulator,encodeSM90MmaGeometry(&context,g)};
  ASSERT_TRUE(succeeded(a.verifyForType(cast<RankedTensorType>(op.getA().getType()),op.getLoc())));
  auto descriptorProof = verifySM90MmaDescriptor(*graph->getVariable(c.vars[1]).storageAlias,storage,descriptor,false,128,512,64);
  ASSERT_EQ(descriptorProof.status,ProofStatus::Proven) << descriptorProof.reason;
  for (auto policy : {attr::GemmWarpPolicy::Auto,attr::GemmWarpPolicy::Square}) {
    op.setPolicy(policy);
    auto proof = target->verifyInstructionContract(*graph,c,encodings,binding);
    EXPECT_EQ(proof.status,ProofStatus::Unknown) << proof.reason;
    EXPECT_NE(proof.reason.find("hardware point budget"),std::string::npos) << proof.reason;
  }
  op.setPolicy(attr::GemmWarpPolicy::Auto);
  auto domain = getSM90MmaGeometries(op,256); ASSERT_TRUE(succeeded(domain));
  ASSERT_EQ(domain->size(),1u); EXPECT_EQ(domain->front().gM,2);
  // The supported alternative remains provable, but cannot replace this hard binding.
  auto alternative = domain->front();
  auto alternativeA = buildSM90MmaFragment(&context,alternative,true,false); ASSERT_TRUE(succeeded(alternativeA));
  EXPECT_EQ(verifySM90MmaFragment(alternative,true,false,cast<RankedTensorType>(op.getA().getType()),*alternativeA).status,ProofStatus::Proven);
  auto alternativeAccumulator = buildSM90MmaFragment(&context,alternative,false,false);
  ASSERT_TRUE(succeeded(alternativeAccumulator));
  NamedAttrList alternativeDescriptor(descriptor.getPayload());
  alternativeDescriptor.set("entries",b.getDenseI64ArrayAttr(ArrayRef<int64_t>(entries).take_front(entries.size()/2)));
  NamedAttrList alternativeFields(fields);
  auto alternativeGeometry = encodeSM90MmaGeometry(&context,alternative);
  for (StringRef field : {"grid","atom","repeats"}) alternativeFields.set(field,alternativeGeometry.get(field));
  alternativeFields.set("b_descriptor",MmaDescriptorPlanAttr::get(&context,alternativeDescriptor.getDictionary(&context)));
  auto alternativeBinding = MmaInstructionContractAttr::getChecked([&] { return op.emitError(); },&context,alternativeFields.getDictionary(&context));
  ASSERT_TRUE(alternativeBinding);
  SmallVector<Attribute> alternativeEncodings{*alternativeA,storage,*alternativeAccumulator,*alternativeAccumulator,alternativeGeometry};
  auto supportedProof = target->verifyInstructionContract(*graph,c,alternativeEncodings,alternativeBinding);
  EXPECT_EQ(supportedProof.status,ProofStatus::Proven) << supportedProof.reason;
  auto before = print(*module);
  std::string diagnostics;
  ScopedDiagnosticHandler capture(&context,[&](Diagnostic &d) { llvm::raw_string_ostream(diagnostics) << d; return success(); });
  EXPECT_TRUE(failed(target->prepareInstructionCandidates(*graph)));
  EXPECT_NE(diagnostics.find("unknown proof: hardware point budget"),std::string::npos) << diagnostics;
  EXPECT_EQ(print(*module),before);
  diagnostics.clear();
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module,*target)));
  EXPECT_NE(diagnostics.find("unknown proof:"),std::string::npos) << diagnostics;
  EXPECT_NE(diagnostics.find("hardware point budget"),std::string::npos) << diagnostics;
  PassManager pipeline(&context); buildFriskLayoutPipeline(pipeline);
  EXPECT_TRUE(failed(pipeline.run(*module))); EXPECT_EQ(print(*module),before);
  EXPECT_EQ(op->getAttr("frisk.mma_contract"),binding);
  NamedAttrList invalid(fields); invalid.set("grid",b.getDenseI64ArrayAttr({2,2}));
  auto bad = MmaInstructionContractAttr::getChecked([&] { return op.emitError(); },&context,invalid.getDictionary(&context));
  ASSERT_TRUE(bad);
  EXPECT_EQ(target->verifyInstructionContract(*graph,c,encodings,bad).status,ProofStatus::Disproven);
  op.setPolicy(attr::GemmWarpPolicy::FullRow);
  EXPECT_EQ(target->verifyInstructionContract(*graph,c,encodings,binding).status,ProofStatus::Disproven);
  op.setPolicy(attr::GemmWarpPolicy::Auto); op.setM(64);
  EXPECT_EQ(target->verifyInstructionContract(*graph,c,encodings,binding).status,ProofStatus::Disproven);
}

TEST_F(MmaAutoIntegrationTest, ConflictingEqualityConnectedHardSeedsAreNotDropped) {
  auto module = make(); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module,*target,LayoutCollectionMode::RelationsOnly); ASSERT_TRUE(succeeded(graph));
  auto &c = instruction(*graph); auto init = c.vars[2], result = c.vars[3];
  auto geometries = getSM90MmaGeometries(mma(*module),256); ASSERT_TRUE(succeeded(geometries));
  auto first = buildSM90MmaFragment(&context,(*geometries)[0],false,false);
  auto second = buildSM90MmaFragment(&context,(*geometries)[1],false,false);
  ASSERT_TRUE(succeeded(first)); ASSERT_TRUE(succeeded(second));
  LayoutConstraintBuilder builder(*graph);
  ASSERT_TRUE(succeeded(builder.require(init,*first,mma(*module),"test-init-seed")));
  ASSERT_TRUE(succeeded(builder.require(result,*second,mma(*module),"test-result-seed")));
  ASSERT_TRUE(succeeded(target->prepareInstructionCandidates(*graph)));
  EXPECT_EQ(graph->getVariable(init).candidates.size(),1u);
  EXPECT_EQ(graph->getVariable(result).candidates.size(),1u);
  EXPECT_EQ(graph->getVariable(init).candidates.front().value,*first);
  EXPECT_EQ(graph->getVariable(result).candidates.front().value,*second);
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(prepareInstructionTuples(*graph,*target,module->getLoc())));
}

TEST_F(MmaAutoIntegrationTest, ExplicitSsLogicalBudgetRetainsUnknownDiagnostic) {
  auto module = make(false,256,512,128); ASSERT_TRUE(module);
  auto op = mma(*module); Builder b(&context);
  SmallVector<Attribute> storage;
  SmallVector<MmaDescriptorPlanAttr> descriptors;
  for (unsigned role : {0u,1u}) {
    bool a = role == 0;
    int64_t extent = a ? 256 : 512;
    auto mn = b.getAffineDimExpr(a ? 0 : 1), k = b.getAffineDimExpr(a ? 1 : 0);
    auto address = a ? (mn%8)*16+mn.floorDiv(8)*128+(k%8)*2+k.floorDiv(8)*(extent*16)
                     : (mn%8)*2+mn.floorDiv(8)*128+(k%8)*16+k.floorDiv(8)*(extent*16);
    auto type = cast<MemRefType>(op->getOperand(role).getType());
    auto map = AffineLayoutMapAttr::get(&context,b.getStrArrayAttr({"dim0","dim1"}),
        b.getDenseI64ArrayAttr(type.getShape()),b.getStrArrayAttr({"byte_offset","bit_offset"}),
        b.getDenseI64ArrayAttr({extent*32,8}),AffineMapAttr::get(AffineMap::get(2,0,{address,b.getAffineConstantExpr(0)},&context)));
    auto layout = StorageLayoutAttr::get(&context,map,MemorySpaceAttr::get(&context,attr::MemorySpace::Shared),
        b.getI64IntegerAttr(1024),b.getI64IntegerAttr(1));
    cast<LayoutViewOp>(op->getOperand(role).getDefiningOp()).setLayoutAttr(layout);
    SmallVector<int64_t> entries;
    for (int64_t mn = 0; mn < extent; mn += a ? 64 : 256) entries.append({mn,0,16*mn,0});
    auto descriptor = MmaDescriptorPlanAttr::get(&context,b.getDictionaryAttr({
        b.getNamedAttr("major",b.getStringAttr(a ? "k" : "mn")),b.getNamedAttr("swizzle",b.getI64IntegerAttr(0)),
        b.getNamedAttr("leading",b.getI64IntegerAttr(extent*16)),b.getNamedAttr("stride",b.getI64IntegerAttr(128)),
        b.getNamedAttr("entries",b.getDenseI64ArrayAttr(entries))}));
    auto alias = analyzeStorageAlias(op->getOperand(role)); ASSERT_TRUE(succeeded(alias));
    auto proof = verifySM90MmaDescriptor(*alias,layout,descriptor,a,extent,16,a ? 64 : 256);
    ASSERT_EQ(proof.status,ProofStatus::Proven) << proof.reason;
    storage.push_back(layout); descriptors.push_back(descriptor);
  }
  auto binding = MmaInstructionContractAttr::getChecked([&] { return op.emitError(); },&context,b.getDictionaryAttr({
      b.getNamedAttr("version",b.getI64IntegerAttr(1)),b.getNamedAttr("target",b.getStringAttr("sm_90a")),
      b.getNamedAttr("form",b.getStringAttr("ss")),b.getNamedAttr("input_type",TypeAttr::get(b.getF16Type())),
      b.getNamedAttr("accumulator_type",TypeAttr::get(b.getF32Type())),b.getNamedAttr("atom",b.getDenseI64ArrayAttr({64,256,16})),
      b.getNamedAttr("grid",b.getDenseI64ArrayAttr({1,1})),b.getNamedAttr("repeats",b.getDenseI64ArrayAttr({4,2,1})),
      b.getNamedAttr("packing",b.getStringAttr("none")),b.getNamedAttr("a_descriptor",descriptors[0]),
      b.getNamedAttr("b_descriptor",descriptors[1])}));
  ASSERT_TRUE(binding); op->setAttr("frisk.mma_contract",binding);
  SM90MmaGeometry geometry{256,512,16,128,1,1,256,4,2,1};
  SmallVector<Attribute> encodings{storage[0],storage[1],{},{},encodeSM90MmaGeometry(&context,geometry)};
  std::string diagnostics;
  ScopedDiagnosticHandler capture(&context,[&](Diagnostic &d) { llvm::raw_string_ostream(diagnostics) << d; return success(); });
  for (auto policy : {attr::GemmWarpPolicy::Auto,attr::GemmWarpPolicy::Square}) {
    op.setPolicy(policy);
    auto graph = collectLayoutConstraints(*module,*target,LayoutCollectionMode::RelationsOnly); ASSERT_TRUE(succeeded(graph));
    auto &c = instruction(*graph);
    // The result has 131072 logical points: no fragment enumeration is needed
    // to classify this complete schema-valid, descriptor-compatible binding.
    auto proof = target->verifyInstructionContract(*graph,c,encodings,binding);
    EXPECT_EQ(proof.status,ProofStatus::Unknown) << proof.reason;
    EXPECT_NE(proof.reason.find("logical point budget exceeds 65536"),std::string::npos);
    auto before = print(*module); diagnostics.clear();
    EXPECT_TRUE(failed(target->prepareInstructionCandidates(*graph)));
    EXPECT_NE(diagnostics.find("unknown proof: logical point budget exceeds 65536"),std::string::npos) << diagnostics;
    EXPECT_EQ(diagnostics.find("invalid geometry"),std::string::npos) << diagnostics;
    EXPECT_EQ(print(*module),before);
    PassManager pipeline(&context); buildFriskLayoutPipeline(pipeline);
    EXPECT_TRUE(failed(pipeline.run(*module))); EXPECT_EQ(print(*module),before);
    EXPECT_EQ(op->getAttr("frisk.mma_contract"),binding);
  }
}

TEST_F(MmaAutoIntegrationTest, ExplicitSupportedDomainInvalidGeometryIsNotUnknown) {
  auto module = make(); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
  auto &c = instruction(*graph); ASSERT_FALSE(c.instruction->tuples.empty());
  auto tuple = c.instruction->tuples.front();
  Builder b(&context); auto op = mma(*module);
  NamedAttrList fields(cast<MmaInstructionContractAttr>(tuple.binding).getPayload());
  fields.set("grid",b.getDenseI64ArrayAttr({2,2}));
  auto binding = MmaInstructionContractAttr::getChecked([&] { return op.emitError(); },&context,fields.getDictionary(&context));
  ASSERT_TRUE(binding);
  EXPECT_EQ(target->verifyInstructionContract(*graph,c,tuple.encodings,binding).status,ProofStatus::Disproven);
  op->setAttr("frisk.mma_contract",binding);
  auto explicitGraph = collectLayoutConstraints(*module,*target,LayoutCollectionMode::RelationsOnly); ASSERT_TRUE(succeeded(explicitGraph));
  auto before = print(*module); std::string diagnostics;
  ScopedDiagnosticHandler capture(&context,[&](Diagnostic &d) { llvm::raw_string_ostream(diagnostics) << d; return success(); });
  EXPECT_TRUE(failed(target->prepareInstructionCandidates(*explicitGraph)));
  EXPECT_NE(diagnostics.find("explicit binding has invalid geometry"),std::string::npos) << diagnostics;
  EXPECT_EQ(diagnostics.find("unknown proof"),std::string::npos) << diagnostics;
  EXPECT_EQ(print(*module),before);
}

TEST_F(MmaAutoIntegrationTest, FourGridReorderedHardSeedIsPreserved) {
  for (bool rs : {false,true}) {
    auto module = make(rs,512,128,1024); ASSERT_TRUE(module);
    auto op = mma(*module);
    auto geometries = getSM90MmaGeometries(op,1024); ASSERT_TRUE(succeeded(geometries));
    ASSERT_EQ(geometries->size(),4u);
    auto fragment = buildSM90MmaFragment(&context,(*geometries)[3],false,false); ASSERT_TRUE(succeeded(fragment));
    auto seed = reordered(*fragment); EXPECT_NE(seed,*fragment);
    auto type = cast<RankedTensorType>(op.getResult().getType());
    EXPECT_EQ(verifySM90MmaFragment((*geometries)[3],false,false,type,seed).status,ProofStatus::Proven);
    op.getResult().setType(RankedTensorType::get(type.getShape(),type.getElementType(),seed));
    auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
    auto &c = instruction(*graph);
    EXPECT_EQ(graph->getVariable(c.vars[3]).candidates.size(),1u);
    EXPECT_EQ(graph->getVariable(c.vars[3]).candidates[0].value,seed);
    for (auto &var : graph->getVariables()) EXPECT_LE(var.candidates.size(),4u) << var.stableName;
    ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
    auto solution = solveLayoutGraph(*graph,*target); ASSERT_TRUE(succeeded(solution));
    EXPECT_EQ(solution->assignments.lookup(c.vars[3]),seed);
    ASSERT_TRUE(succeeded(materializeLayouts(*module,*graph,*solution)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,*target)));
  }
}

TEST_F(MmaAutoIntegrationTest, FifthRoleInvariantsAndMaterializationRollback) {
  auto module = make(); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
  auto &c = instruction(*graph); auto id = c.vars[4];
  auto &var = graph->getVariable(id);
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic &) { return success(); });
  auto saved = var;
  var.kind = LayoutKind::Distributed; EXPECT_TRUE(failed(graph->verifyInvariants(module->getLoc()))); var = saved;
  var.anchor = module->getOperation(); EXPECT_TRUE(failed(graph->verifyInvariants(module->getLoc()))); var = saved;
  var.requiredThreads = 128; EXPECT_TRUE(failed(graph->verifyInvariants(module->getLoc()))); var = saved;
  var.use = &mma(*module)->getOpOperand(2); EXPECT_TRUE(failed(graph->verifyInvariants(module->getLoc()))); var = saved;
  c.vars.pop_back(); EXPECT_TRUE(failed(graph->verifyInvariants(module->getLoc()))); c.vars.push_back(id);
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveLayoutGraph(*graph,*target); ASSERT_TRUE(succeeded(solution));
  auto before = print(*module);
  auto selected = solution->assignments.lookup(id);
  for (auto &candidate : var.candidates) if (candidate.value != selected) { solution->assignments[id] = candidate.value; break; }
  EXPECT_TRUE(failed(materializeLayouts(*module,*graph,*solution)));
  EXPECT_EQ(before,print(*module));
}

TEST_F(MmaAutoIntegrationTest, BoundedAttemptsCountFailuresAndDiscardPartialTable) {
  // Deliberately undecoded role attributes isolate the target-independent
  // enumeration budget from SM90's usually much smaller compatible domains.
  class CountingTarget : public LayoutTarget {
  public:
    mutable unsigned attempts = 0;
    bool reject = false;
    void enumerateCandidates(const LayoutVar &,SmallVectorImpl<LayoutCandidate> &) const override {}
    LogicalResult verifyCandidate(const LayoutVar &,Attribute,Location) const override { return success(); }
    FailureOr<CostVector> evaluate(const CandidateAssignment &) const override { return CostVector{}; }
    InstructionContractBuildResult buildInstructionContractDetailed(const LayoutConstraintGraph &,
        const LayoutConstraint &,ArrayRef<Attribute> e) const override {
      ++attempts;
      return {reject ? Attribute{} : e[0],{reject ? ProofStatus::Disproven : ProofStatus::Proven,{},"budget test"},"test",{}};
    }
  } counting;
  for (bool reject : {false,true}) {
    auto module = make(); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module,*target,LayoutCollectionMode::RelationsOnly); ASSERT_TRUE(succeeded(graph));
    auto &c = instruction(*graph);
    for (auto id : c.vars) for (int i = 0; i < 4; ++i)
      graph->getVariable(id).candidates.push_back({Builder(&context).getI64IntegerAttr(i),kInvalidProvenanceID,uint64_t(i)});
    counting.attempts = 0; counting.reject = reject;
    std::string message;
    ScopedDiagnosticHandler quiet(&context,[&](Diagnostic &d) { llvm::raw_string_ostream(message) << d; return success(); });
    EXPECT_TRUE(failed(prepareInstructionTuples(*graph,counting,module->getLoc())));
    EXPECT_EQ(counting.attempts,256u);
    EXPECT_EQ(graph->getCandidatePreparationStatistics().instructionCombinations,256u);
    EXPECT_TRUE(c.instruction->tuples.empty());
    EXPECT_NE(message.find("unknown proof"),std::string::npos);
    EXPECT_NE(message.find("partial table discarded"),std::string::npos);
    // Candidate overflow fails before any tuple build, preserving the hard seed.
    auto &domain = graph->getVariable(c.vars[4]).candidates;
    domain.push_back({Builder(&context).getI64IntegerAttr(4),kInvalidProvenanceID,4});
    counting.attempts = 0;
    EXPECT_TRUE(failed(prepareInstructionTuples(*graph,counting,module->getLoc())));
    EXPECT_EQ(counting.attempts,0u); EXPECT_EQ(domain.size(),5u);
  }
}

TEST_F(MmaAutoIntegrationTest, FragmentPrechecksAndProfileAreSeparateFromCompleteAttempts) {
  auto module = make(true); ASSERT_TRUE(module);
  LayoutProfile profile(true); LayoutProfile::Invocation invocation(&profile);
  LayoutTraceOptions options; options.file = "/tmp/unused-mma-trace.json";
  LayoutTrace trace(options);
  auto graph = collectLayoutConstraints(*module,*target,LayoutCollectionMode::InitializeCandidates,nullptr,&trace);
  ASSERT_TRUE(succeeded(graph));
  auto &stats = graph->getCandidatePreparationStatistics();
  EXPECT_EQ(stats.instructionFragmentPrechecks,12u); // 2 geometries x 3 roles x 2 candidates.
  EXPECT_GT(stats.instructionCombinations,0u); EXPECT_LE(stats.instructionCombinations,256u);
  EXPECT_EQ(profile.get(LayoutProfileKind::InstructionContract).count,stats.instructionCombinations);
  EXPECT_NE(trace.serialize().find("instruction-fragment-precheck"),std::string::npos);
  EXPECT_NE(trace.serialize().find("geometry"),std::string::npos);
}

TEST_F(MmaAutoIntegrationTest, LegalThreadSubdomainsForSSAndRS) {
  for (bool rs : {false,true}) for (int threads : {128,256,512,1024}) {
    SCOPED_TRACE(::testing::Message() << "rs=" << rs << " threads=" << threads);
    auto module = make(rs,128,128,threads); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
    auto &c = instruction(*graph);
    for (auto &tuple : c.instruction->tuples)
      EXPECT_EQ(verifySM90MmaContract(*graph,c,tuple.encodings,tuple.binding).status,ProofStatus::Proven);
    ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
    auto solution = solveLayoutGraph(*graph,*target); ASSERT_TRUE(succeeded(solution));
    ASSERT_TRUE(succeeded(materializeLayouts(*module,*graph,*solution)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,*target)));
  }
}

TEST_F(MmaAutoIntegrationTest, ReduceBothAxesFixedConsumerConversionAndExactOracle) {
  for (unsigned axis : {0u,1u}) for (bool fixedConsumer : {false,true}) {
    SCOPED_TRACE(::testing::Message() << "axis=" << axis << " fixed=" << fixedConsumer);
    auto module = make(); ASSERT_TRUE(module); auto op = mma(*module);
    OpBuilder b(&context); b.setInsertionPointAfter(op);
    Value source = op.getResult();
    if (fixedConsumer) {
      auto geometries = getSM90MmaGeometries(op,256); ASSERT_TRUE(succeeded(geometries));
      auto e = buildSM90MmaFragment(&context,geometries->front(),false,false); ASSERT_TRUE(succeeded(e));
      auto t = RankedTensorType::get({128,128},b.getF32Type(),*e);
      source = b.create<ConvertLayoutOp>(op.getLoc(),t,source);
    }
    auto reduce = b.create<ReduceTensorOp>(op.getLoc(),RankedTensorType::get({128},b.getF32Type()),source,"sum",axis);
    reduce->setAttr("frisk.execution_threads",b.getI64IntegerAttr(256));
    auto storageType = MemRefType::get({128},b.getF32Type(),{},1);
    auto root = b.create<memref::AllocOp>(op.getLoc(),storageType);
    root->setAttr("alignment",b.getI64IntegerAttr(128));
    auto storage = AffineLayoutMapAttr::get(&context,b.getArrayAttr({b.getStringAttr("dim0")}),
        b.getDenseI64ArrayAttr({128}),b.getArrayAttr({b.getStringAttr("byte_offset"),b.getStringAttr("bit_offset")}),
        b.getDenseI64ArrayAttr({512,8}),AffineMapAttr::get(AffineMap::get(1,0,{b.getAffineDimExpr(0)*4,b.getAffineConstantExpr(0)},&context)));
    auto layout = StorageLayoutAttr::get(&context,storage,MemorySpaceAttr::get(&context,attr::MemorySpace::Global),
        b.getI64IntegerAttr(128),b.getI64IntegerAttr(1));
    auto view = b.create<LayoutViewOp>(op.getLoc(),storageType,root,layout);
    b.create<TileStoreOp>(op.getLoc(),reduce.getResult(),view.getResult());
    auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
    ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
    auto solution = solveLayoutGraph(*graph,*target); ASSERT_TRUE(succeeded(solution));
    CandidateAssignment current;
    std::optional<CostEstimate> best;
    unsigned feasible = 0;
    std::function<void(unsigned)> visit = [&](unsigned index) {
      for (auto &c : graph->getConstraints())
        if (c.strength == ConstraintStrength::Hard &&
            !satisfiesLayoutConstraint(*graph,c,current.values,false)) return;
      if (index == graph->getVariables().size()) {
        auto cost = target->evaluate(*graph,current); ASSERT_TRUE(succeeded(cost));
        auto choices = chooseLayoutConversions(*graph,current,*target); ASSERT_TRUE(succeeded(choices));
        *cost = addLayoutCosts(*cost,choices->cost);
        ++feasible;
        if (!best || cost->cost < best->cost) best = *cost;
        return;
      }
      auto &v = graph->getVariable(index);
      for (auto &candidate : v.candidates) { current.values[v.id] = candidate.value; visit(index+1); }
      current.values.erase(v.id);
    };
    visit(0); ASSERT_TRUE(best); EXPECT_GT(feasible,0u);
    std::string witness;
    llvm::raw_string_ostream(witness) << llvm::json::Value(layoutCostJSON(solution->cost));
    std::string key = "consumer_axis" + std::to_string(axis) + "_fixed" + std::to_string(fixedConsumer);
    RecordProperty(key+"_cost",witness);
    RecordProperty(key+"_feasible",int(feasible));
    RecordProperty(key+"_fragment_prechecks",int(graph->getCandidatePreparationStatistics().instructionFragmentPrechecks));
    RecordProperty(key+"_complete_attempts",int(graph->getCandidatePreparationStatistics().instructionCombinations));
    EXPECT_FALSE(best->cost < solution->cost.cost); EXPECT_FALSE(solution->cost.cost < best->cost);
    auto &c = instruction(*graph);
    auto contract = cast<MmaInstructionContractAttr>(solution->instructionBindings.lookup(c.id));
    // This witness is joint, but does NOT claim a consumer-driven reversal.
    EXPECT_EQ(contract.getPayload().getAs<DenseI64ArrayAttr>("grid")[0],2);
    ASSERT_TRUE(succeeded(materializeLayouts(*module,*graph,*solution)));
    if (fixedConsumer) {
      unsigned conversions = 0;
      module->walk([&](ConvertLayoutOp convert) {
        if (convert.getSource().getType() != convert.getResult().getType()) ++conversions;
      });
      EXPECT_GT(conversions,0u); // Explicit conversion is not a newly inserted edge.
    }
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module,*target)));
    // Production pipeline stabilizes the live graph, including fixed ABI;
    // compare two full pipeline invocations (not low-level vs canonicalizer).
    PassManager pm(&context); buildFriskLayoutPipeline(pm);
    ASSERT_TRUE(succeeded(pm.run(*module)));
    auto once = print(*module);
    auto replay = parseSourceString<ModuleOp>(once,&context); ASSERT_TRUE(replay);
    PassManager again(&context); buildFriskLayoutPipeline(again);
    ASSERT_TRUE(succeeded(again.run(*replay))); EXPECT_EQ(once,print(*replay));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*replay,*target)));
    unsigned stores = 0;
    replay->walk([&](TileStoreOp store) {
      ++stores; auto actual = store.getTarget().getDefiningOp<LayoutViewOp>();
      ASSERT_TRUE(actual); EXPECT_EQ(actual.getLayoutAttr(),layout);
    });
    EXPECT_EQ(stores,1u);
  }
}
} // namespace

TEST_F(MmaAutoIntegrationTest, ProducerScopeGeometryWorkHasNoStorageDependency) {
 auto module=make(false,64,64,128);ASSERT_TRUE(module);
 auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
 auto solved=solveLayoutGraph(*graph,*target);ASSERT_TRUE(succeeded(solved));auto &c=instruction(*graph);
 ASSERT_EQ(graph->getVariable(c.vars[0]).kind,LayoutKind::Storage);
 ExperimentPolicy policy;policy.scope.fullGraph=false;policy.scope.producers={c.vars[3],c.vars[4]};policy.scope.producerInstructions={c.id};
 CandidateAssignment a{solved->assignments};auto local=scoreExperimentAssignment(*graph,a,*target,policy);ASSERT_TRUE(succeeded(local));EXPECT_GT(local->cost.instructionPathAndWork,0u);
 auto missing=policy;missing.scope.producers={c.vars[3]};EXPECT_TRUE(failed(scoreExperimentAssignment(*graph,a,*target,missing)));
 auto storage=policy;storage.scope.producers.push_back(c.vars[0]);EXPECT_TRUE(failed(scoreExperimentAssignment(*graph,a,*target,storage)));
 auto future=policy;future.scope.producers.push_back(c.vars[2]);EXPECT_TRUE(failed(scoreExperimentAssignment(*graph,a,*target,future)));
}
