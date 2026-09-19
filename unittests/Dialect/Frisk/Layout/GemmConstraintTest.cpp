#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"

using namespace mlir;
using namespace mlir::frisk;
TEST(GemmMath, AcceptsRectangularTransposeAndReadOnlyEffects) {
  MLIRContext context;
  context.loadDialect<FriskDialect, func::FuncDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a: memref<16x64xf16,3>, %b: memref<32x16xf16,3>,
                    %c: tensor<64x32xf32>) -> tensor<64x32xf32> {
      %r = "frisk.mma"(%a,%b,%c) {m=64:i64,n=32:i64,k=16:i64,
        trans_a=true,trans_b=true} :
        (memref<16x64xf16,3>,memref<32x16xf16,3>,tensor<64x32xf32>) -> tensor<64x32xf32>
      return %r : tensor<64x32xf32>
    })mlir", &context);
  ASSERT_TRUE(module);
  Operation *mma = &module->getBody()->front().getRegion(0).front().front();
  auto effects = dyn_cast<MemoryEffectOpInterface>(mma);
  ASSERT_TRUE(effects);
  SmallVector<MemoryEffects::EffectInstance> list;
  effects.getEffects(list);
  ASSERT_EQ(list.size(), 2u);
  for (auto effect : list) EXPECT_TRUE(isa<MemoryEffects::Read>(effect.getEffect()));
}

namespace {
class GemmMathTest : public testing::Test {
protected:
  MLIRContext context;
  GemmMathTest() { context.loadDialect<FriskDialect,func::FuncDialect>(); }
  OwningOpRef<ModuleOp> make() {
    return parseSourceString<ModuleOp>(R"mlir(
      func.func @test(%a:tensor<128x64xbf16>,%b:memref<64x128xbf16,3>,%c:tensor<128x128xf32>) {
        %r="frisk.mma"(%a,%b,%c) {m=128:i64,n=128:i64,k=64:i64} :
          (tensor<128x64xbf16>,memref<64x128xbf16,3>,tensor<128x128xf32>) -> tensor<128x128xf32>
        return
      })mlir",&context);
  }
  MmaOp op(ModuleOp m) { MmaOp result;m.walk([&](MmaOp o){result=o;});return result; }
};
}

TEST_F(GemmMathTest, RejectsNonpositiveDimensionsAndWrongRectangularTranspose) {
  for (StringRef name:{"m","n","k"}) for(int64_t value:{int64_t(0),int64_t(-1)}) {
    auto m=make();ASSERT_TRUE(m);
    op(*m)->setAttr(name,IntegerAttr::get(IntegerType::get(&context,64),value));
    ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
    EXPECT_TRUE(failed(verify(*m)));
  }
  auto m=make();ASSERT_TRUE(m);
  op(*m)->setAttr("trans_a",BoolAttr::get(&context,true));
  ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
  EXPECT_TRUE(failed(verify(*m)));
}

TEST_F(GemmMathTest, RejectsRankDynamicAndMismatchedAccumulatorTypes) {
  for (int mode=0;mode<4;++mode) {
    auto m=make();ASSERT_TRUE(m);auto mma=op(*m);
    Type wrong;
    if (mode==0) wrong=RankedTensorType::get({128,128,1},Float32Type::get(&context));
    if (mode==1) wrong=RankedTensorType::get({128,ShapedType::kDynamic},Float32Type::get(&context));
    if (mode==2) wrong=RankedTensorType::get({128,128},Float16Type::get(&context));
    if (mode==3) wrong=RankedTensorType::get({128,128},IntegerType::get(&context,32));
    mma->getResult(0).setType(wrong);
    ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
    EXPECT_TRUE(failed(mma.verify()));
  }
}

TEST_F(GemmMathTest, RejectsUnknownTensorEncoding) {
  auto m=make();ASSERT_TRUE(m);auto mma=op(*m);
  mma.getResult().setType(RankedTensorType::get({128,128},Float32Type::get(&context),StringAttr::get(&context,"bad")));
  ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
  EXPECT_TRUE(failed(mma.verify()));
}

TEST_F(GemmMathTest, PolicyAndAllSupportedThreadCounts) {
  auto m=make();ASSERT_TRUE(m);auto mma=op(*m);
  for (auto policy:{attr::GemmWarpPolicy::Square,attr::GemmWarpPolicy::FullRow,attr::GemmWarpPolicy::FullCol})
    for (int64_t threads:{128,256,512,1024}) {
      mma.setPolicy(policy);
      auto g=getSM90MmaGeometry(mma,threads);ASSERT_TRUE(succeeded(g));
      EXPECT_EQ(g->gM*g->gN,threads/128);
      if (threads==256 && policy==attr::GemmWarpPolicy::FullRow) { EXPECT_EQ(g->gM,2);EXPECT_EQ(g->gN,1); }
      if (threads==256 && policy==attr::GemmWarpPolicy::FullCol) { EXPECT_EQ(g->gM,1);EXPECT_EQ(g->gN,2); }
      if (threads==256 && policy==attr::GemmWarpPolicy::Square) { EXPECT_EQ(g->gM,1);EXPECT_EQ(g->gN,2); }
    }
  for(int64_t threads:{0,32,64,192,2048}) EXPECT_TRUE(failed(getSM90MmaGeometry(mma,threads)));
  mma.setM(64);mma.setN(8);
  EXPECT_TRUE(failed(getSM90MmaGeometry(mma,1024))); // No legal grid, not truncated threads.
}

TEST_F(GemmMathTest, StrictTypedSchemasRejectPartialAndUnknownFields) {
  auto emit=[&]{return emitError(UnknownLoc::get(&context));};
  Builder b(&context);
  ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
  EXPECT_TRUE(failed(MmaInstructionContractAttr::verify(emit,b.getDictionaryAttr({}))));
  EXPECT_TRUE(failed(MmaDescriptorPlanAttr::verify(emit,b.getDictionaryAttr({}))));
  auto desc=b.getDictionaryAttr({b.getNamedAttr("major",b.getStringAttr("k")),
    b.getNamedAttr("swizzle",b.getI64IntegerAttr(0)),b.getNamedAttr("leading",b.getI64IntegerAttr(16)),
    b.getNamedAttr("stride",b.getI64IntegerAttr(128)),b.getNamedAttr("entries",b.getDenseI64ArrayAttr({0,0,0,0}))});
  EXPECT_TRUE(succeeded(MmaDescriptorPlanAttr::verify(emit,desc)));
  NamedAttrList bad(desc);bad.set("pointer",b.getI64IntegerAttr(0));
  EXPECT_TRUE(failed(MmaDescriptorPlanAttr::verify(emit,bad.getDictionary(&context))));
}

TEST_F(GemmMathTest, JointProofNamesFailingInstructionRole) {
  auto module=make();ASSERT_TRUE(module);auto mma=op(*module);
  Builder b(&context);module->getOperation()->setAttr("frisk.target",b.getStringAttr("sm_90a"));
  LayoutConstraintGraph graph;
  SmallVector<LayoutVarID> roles;
  for(auto [index,type]:llvm::enumerate(SmallVector<Type>{mma.getA().getType(),mma.getB().getType(),mma.getInit().getType(),mma.getResult().getType()})) {
    roles.push_back(graph.addVariable(index==1?LayoutKind::Storage:LayoutKind::Distributed,type,(Twine("role")+Twine(index)).str(),mma));
    graph.getVariable(roles.back()).requiredThreads=128;
  }
  LayoutConstraint c;c.vars=roles;c.instruction=InstructionLayoutContract{};c.instruction->source=mma;
  auto descriptor=MmaDescriptorPlanAttr::get(&context,b.getDictionaryAttr({
    b.getNamedAttr("major",b.getStringAttr("mn")),b.getNamedAttr("swizzle",b.getI64IntegerAttr(0)),
    b.getNamedAttr("leading",b.getI64IntegerAttr(2048)),b.getNamedAttr("stride",b.getI64IntegerAttr(128)),
    b.getNamedAttr("entries",b.getDenseI64ArrayAttr({0,0,0,0}))}));
  auto binding=MmaInstructionContractAttr::get(&context,b.getDictionaryAttr({
    b.getNamedAttr("version",b.getI64IntegerAttr(1)),b.getNamedAttr("target",b.getStringAttr("sm_90a")),
    b.getNamedAttr("form",b.getStringAttr("rs")),b.getNamedAttr("input_type",TypeAttr::get(b.getBF16Type())),
    b.getNamedAttr("accumulator_type",TypeAttr::get(b.getF32Type())),b.getNamedAttr("atom",b.getDenseI64ArrayAttr({64,128,16})),
    b.getNamedAttr("grid",b.getDenseI64ArrayAttr({1,1})),b.getNamedAttr("repeats",b.getDenseI64ArrayAttr({2,1,4})),
    b.getNamedAttr("packing",b.getStringAttr("f16x2-low-high")),b.getNamedAttr("b_descriptor",descriptor)}));
  SmallVector<Attribute> encodings(4);
  EXPECT_NE(verifySM90MmaContract(graph,c,encodings,binding).reason.find("init"),std::string::npos);
  auto g=getSM90MmaGeometry(mma,128);ASSERT_TRUE(succeeded(g));
  auto accum=buildSM90MmaFragment(&context,*g,false,false);ASSERT_TRUE(succeeded(accum));
  encodings[2]=*accum;encodings[3]=*accum;
  EXPECT_NE(verifySM90MmaContract(graph,c,encodings,binding).reason.find("A"),std::string::npos);
}
