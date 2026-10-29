#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;

TEST(SM90MmaGeometry, AutoParsesAndSquareRemainsDefault) {
  MLIRContext context;
  context.loadDialect<FriskDialect, func::FuncDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a:memref<128x16xf16,3>, %b:memref<16x128xf16,3>, %c:tensor<128x128xf32>) {
      %r = "frisk.mma"(%a,%b,%c) {m=128:i64,n=128:i64,k=16:i64,policy=#frisk<gemm_warp_policy Auto>} :
        (memref<128x16xf16,3>,memref<16x128xf16,3>,tensor<128x128xf32>) -> tensor<128x128xf32>
      return
    })mlir", &context);
  ASSERT_TRUE(module);
  module->walk([&](MmaOp op) {
    auto all = getSM90MmaGeometries(op, 256);
    ASSERT_TRUE(succeeded(all)); ASSERT_EQ(all->size(), 2u);
    EXPECT_TRUE(failed(getSM90MmaGeometry(op,256)));
    EXPECT_EQ((*all)[0].gM,1); EXPECT_EQ((*all)[0].gN,2);
    EXPECT_EQ((*all)[0].atomN,64); EXPECT_EQ((*all)[0].mRepeat,2);
    EXPECT_EQ((*all)[1].gM,2); EXPECT_EQ((*all)[1].gN,1);
    EXPECT_EQ((*all)[1].atomN,128); EXPECT_EQ((*all)[1].mRepeat,1);
    Builder b(&context);
    for (auto &g : *all) {
      auto encoded = encodeSM90MmaGeometry(&context,g);
      EXPECT_TRUE(succeeded(decodeSM90MmaGeometry(op,256,encoded)));
      for (StringRef field : {"kind","version","shape","threads","grid","atom","repeats","extra"}) {
        NamedAttrList bad(encoded);
        bad.set(field,b.getStringAttr("invalid"));
        EXPECT_TRUE(failed(decodeSM90MmaGeometry(op,256,bad.getDictionary(&context)))) << field.str();
      }
      for (auto [field,value] : SmallVector<std::pair<StringRef,Attribute>>{
               {"shape",b.getDenseI64ArrayAttr({64,128,16})},
               {"threads",b.getI64IntegerAttr(512)},
               {"grid",b.getDenseI64ArrayAttr({2,2})},
               {"atom",b.getDenseI64ArrayAttr({64,32,16})},
               {"repeats",b.getDenseI64ArrayAttr({3,1,1})}}) {
        NamedAttrList bad(encoded); bad.set(field,value);
        EXPECT_TRUE(failed(decodeSM90MmaGeometry(op,256,bad.getDictionary(&context)))) << field.str();
      }
      EXPECT_TRUE(failed(decodeSM90MmaGeometry(op,128,encoded)));
    }
    for (int64_t threads : {128,256,512,1024}) {
      auto domain = getSM90MmaGeometries(op,threads); ASSERT_TRUE(succeeded(domain));
      EXPECT_LE(domain->size(),4u);
      for (auto &g : *domain) {
        EXPECT_EQ(g.gM*g.gN,threads/128);
        EXPECT_EQ(g.mRepeat*64*g.gM,128);
        EXPECT_EQ(g.nRepeat*g.atomN*g.gN,128);
      }
    }
    for (int64_t threads : {0,32,64,192,2048}) EXPECT_TRUE(failed(getSM90MmaGeometries(op,threads)));
    op.setPolicy(attr::GemmWarpPolicy::Square);
    auto g = getSM90MmaGeometry(op, 256);
    ASSERT_TRUE(succeeded(g));
    EXPECT_EQ(g->gM, 1); EXPECT_EQ(g->gN, 2);
    auto legacy = getSM90MmaGeometries(op,256);
    ASSERT_TRUE(succeeded(legacy)); EXPECT_EQ(legacy->size(),1u);
    EXPECT_TRUE(failed(decodeSM90MmaGeometry(op,256,encodeSM90MmaGeometry(&context,(*all)[1]))));
  });
}

TEST(SM90MmaGeometry, LegacyGemmRejectsAuto) {
  MLIRContext context;
  context.loadDialect<FriskDialect,func::FuncDialect>();
  ScopedDiagnosticHandler quiet(&context,[](Diagnostic &) { return success(); });
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @test(%a:memref<64x16xf16,3>,%b:memref<16x64xf16,3>,%c:memref<64x64xf32>) {
      frisk.gemm (%a,%b,%c) {transA=false,transB=false,M=64:i64,N=64:i64,K=16:i64,policy=#frisk<gemm_warp_policy Auto>,clear_accum=true} : memref<64x16xf16,3>,memref<16x64xf16,3>,memref<64x64xf32>
      return
    })mlir",&context);
  EXPECT_FALSE(module);
}
