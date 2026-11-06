#include "gtest/gtest.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
using namespace mlir;
using namespace mlir::frisk;
TEST(SM90MmaProof, IndependentPackedAAndAccumulatorOracle) {
  SM90MmaGeometry g{64,64,16,128,1,1,64,1,1,1};
  // PTX 9.4 wgmma-64N16-A.png and wgmma-64N16-D.png.
  struct Row {int64_t r,l,w; std::array<int64_t,2> a,d;};
  const Row rows[] = {
    {0,0,0,{0,0},{0,0}}, {1,0,0,{0,1},{0,1}},
    {2,0,0,{8,0},{8,0}}, {3,0,0,{8,1},{8,1}},
    {4,0,0,{0,8},{0,8}}, {7,31,3,{63,15},{63,15}},
    {5,17,2,{36,11},{36,11}}
  };
  for (const Row &r : rows) {
    EXPECT_EQ(getSM90MmaFragmentCoordinate(g,true,false,r.r,r.l,r.w,0),r.a);
    EXPECT_EQ(getSM90MmaFragmentCoordinate(g,false,false,r.r,r.l,r.w,0),r.d);
  }
  EXPECT_EQ(getSM90MmaFragmentCoordinate(g,false,false,31,31,3,0),
            (std::array<int64_t,2>{63,63}));
}
TEST(SM90MmaProof, RepeatsAndReadOnlyReplicationOracle) {
  SM90MmaGeometry g{128,128,64,256,1,2,64,2,1,4};
  EXPECT_EQ(getSM90MmaFragmentCoordinate(g,true,false,63,31,3,1),
            (std::array<int64_t,2>{127,63}));
  EXPECT_EQ(getSM90MmaFragmentCoordinate(g,true,true,63,31,3,1),
            (std::array<int64_t,2>{63,127}));
  EXPECT_EQ(getSM90MmaFragmentCoordinate(g,false,false,63,31,3,1),
            (std::array<int64_t,2>{127,127}));
}

TEST(SM90MmaProof, FullFragmentCoveragePackingTopologyAndBudget) {
  MLIRContext ctx;ctx.loadDialect<FriskDialect>();Builder b(&ctx);
  SM90MmaGeometry g{128,128,64,256,1,2,64,2,1,4};
  for(bool a:{true,false}) for(bool trans:{true,false}) {
    if(!a && trans) continue;
    auto encoding=buildSM90MmaFragment(&ctx,g,a,trans);ASSERT_TRUE(succeeded(encoding));
    SmallVector<int64_t> shape{128,a?64:128};if(a && trans)std::swap(shape[0],shape[1]);
    auto type=RankedTensorType::get(shape,a?Type(b.getBF16Type()):Type(b.getF32Type()));
    EXPECT_EQ(verifySM90MmaFragment(g,a,trans,type,*encoding).status,ProofStatus::Proven);
    auto wrongTopology=DistributedEncodingAttr::get(&ctx,encoding->getMap(),
      b.getDenseI64ArrayAttr({64,32,8,1,1}),encoding->getReplication());
    EXPECT_NE(verifySM90MmaFragment(g,a,trans,type,wrongTopology).status,ProofStatus::Proven);
    auto map=cast<BitLinearLayoutMapAttr>(encoding->getMap());
    SmallVector<APInt> bits(map.getMatrix().getValues<APInt>());
    unsigned columns=map.getMatrix().getType().getShape()[1];
    // Swap register bits 0/1: preserves element coverage but breaks packed
    // low/high halves and hardware register order.
    for(unsigned row=0;row<bits.size()/columns;++row) std::swap(bits[row*columns],bits[row*columns+1]);
    auto wrongMap=BitLinearLayoutMapAttr::get(&ctx,map.getInputNames(),map.getInputBitWidths(),
      map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(map.getMatrix().getType(),bits));
    auto wrong=DistributedEncodingAttr::get(&ctx,wrongMap,encoding->getTopology(),encoding->getReplication());
    EXPECT_NE(verifySM90MmaFragment(g,a,trans,type,wrong).status,ProofStatus::Proven);
  }
  SM90MmaGeometry over{128,512,512,1024,1,8,64,2,1,32};
  auto small=buildSM90MmaFragment(&ctx,g,true,false);ASSERT_TRUE(succeeded(small));
  EXPECT_EQ(verifySM90MmaFragment(over,true,false,RankedTensorType::get({128,512},b.getBF16Type()),*small).status,ProofStatus::Unknown);
}

TEST(SM90MmaProof, EverySupportedAtomicNHasExactLastSlot) {
  MLIRContext ctx;ctx.loadDialect<FriskDialect>();Builder b(&ctx);
  for(int64_t n:{8,16,32,64,128,256}) {
    SM90MmaGeometry g{64,n,16,128,1,1,n,1,1,1};
    EXPECT_EQ(getSM90MmaFragmentCoordinate(g,false,false,n/2-1,31,3,0),
              (std::array<int64_t,2>{63,n-1}));
    auto encoding=buildSM90MmaFragment(&ctx,g,false,false);ASSERT_TRUE(succeeded(encoding));
    EXPECT_EQ(verifySM90MmaFragment(g,false,false,RankedTensorType::get({64,n},b.getF32Type()),*encoding).status,
              ProofStatus::Proven);
  }
}
TEST(SM90MmaProof, DescriptorIndependentCanonicalTable) {
  // PTX canonical layouts normalized to 16-bit elements, offsets in bytes.
  EXPECT_EQ(getSM90DescriptorAddress("k",0,256,128,0,0,9,9),402u);
  EXPECT_EQ(getSM90DescriptorAddress("mn",0,256,128,0,0,9,9),402u);
  EXPECT_EQ(getSM90DescriptorAddress("k",32,16,256,0,0,9,9),306u);
  EXPECT_EQ(getSM90DescriptorAddress("mn",32,256,512,0,0,17,9),802u);
}

namespace {
class DescriptorProofTest : public testing::Test {
protected:
  MLIRContext ctx;
  Builder b{&ctx};
  DescriptorProofTest() { ctx.loadDialect<FriskDialect,func::FuncDialect,memref::MemRefDialect>(); }
  // Independent canonical storage oracle: PTX canonical table expanded with
  // the following literal element strides (MN-low,MN-high,K-low,K-high).
  AffineExpr oracle(bool majorK, int sw, int64_t mnExtent, AffineExpr mn, AffineExpr k) {
    AffineExpr raw;
    if (majorK) {
      switch (sw) {
      case 0: raw=(mn%8)*16+mn.floorDiv(8)*128+(k%8)*2+k.floorDiv(8)*(mnExtent*16); break;
      case 32: raw=mn*32+(k%16)*2+k.floorDiv(16)*(mnExtent*32); break;
      case 64: raw=mn*64+(k%32)*2+k.floorDiv(32)*(mnExtent*64); break;
      default: raw=mn*128+(k%64)*2+k.floorDiv(64)*(mnExtent*128); break;
      }
    } else {
      switch (sw) {
      case 0: raw=(mn%8)*2+mn.floorDiv(8)*128+(k%8)*16+k.floorDiv(8)*(mnExtent*16); break;
      case 32: raw=(mn%16)*2+mn.floorDiv(16)*256+(k%8)*32+k.floorDiv(8)*(mnExtent*16); break;
      case 64: raw=(mn%32)*2+mn.floorDiv(32)*512+(k%8)*64+k.floorDiv(8)*(mnExtent*16); break;
      default: raw=(mn%64)*2+mn.floorDiv(64)*1024+(k%8)*128+k.floorDiv(8)*(mnExtent*16); break;
      }
    }
    AffineExpr result=raw;
    for (int dst : {16,32,64}) {
      if (sw<=dst) break;
      auto original=raw.floorDiv(dst)%2;
      result=result+(((original+raw.floorDiv(dst*8)%2)%2)-original)*dst;
    }
    return result;
  }
  StorageLayoutAttr layout(AffineExpr address, ArrayRef<int64_t> shape, int64_t bytes) {
    auto map=AffineLayoutMapAttr::get(&ctx,b.getArrayAttr({b.getStringAttr("dim0"),b.getStringAttr("dim1")}),
      b.getDenseI64ArrayAttr(shape),b.getArrayAttr({b.getStringAttr("byte_offset"),b.getStringAttr("bit_offset")}),
      b.getDenseI64ArrayAttr({bytes,8}),AffineMapAttr::get(AffineMap::get(2,0,{address,b.getAffineConstantExpr(0)},&ctx)));
    return StorageLayoutAttr::get(&ctx,map,MemorySpaceAttr::get(&ctx,attr::MemorySpace::Shared),
                                b.getI64IntegerAttr(1024),b.getI64IntegerAttr(1));
  }
  MmaDescriptorPlanAttr plan(bool majorK,int sw,int64_t leading,int64_t stride,
                            AffineExpr address, int64_t mnExtent,int64_t kExtent) {
    auto map=AffineMap::get(2,0,address);
    SmallVector<int64_t> entries;
    for (int64_t mn=0;mn<mnExtent;mn+=64)
      for (int64_t k=0;k<kExtent;k+=16) {
        SmallVector<Attribute> folded;
        EXPECT_TRUE(succeeded(map.constantFold({b.getIndexAttr(mn),b.getIndexAttr(k)},folded)));
        int64_t start=cast<IntegerAttr>(folded[0]).getInt();
        entries.append({mn,k,start,sw ? (start>>7)&7 : 0});
      }
    return MmaDescriptorPlanAttr::get(&ctx,b.getDictionaryAttr({
      b.getNamedAttr("major",b.getStringAttr(majorK?"k":"mn")),
      b.getNamedAttr("swizzle",b.getI64IntegerAttr(sw)),
      b.getNamedAttr("leading",b.getI64IntegerAttr(leading)),
      b.getNamedAttr("stride",b.getI64IntegerAttr(stride)),
      b.getNamedAttr("entries",b.getDenseI64ArrayAttr(entries))}));
  }
};
}

TEST_F(DescriptorProofTest, AllEightTemplatesAndTamperedOffsets) {
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @test() {
      %root=memref.alloc() {alignment=1024:i64} : memref<64x64xf16,3>
      %v=frisk.layout_view %root : memref<64x64xf16,3> -> memref<64x64xf16,3>
      return
    })mlir",&ctx);
  ASSERT_TRUE(module);
  LayoutViewOp view; module->walk([&](LayoutViewOp v){view=v;});
  auto info=analyzeStorageAlias(view.getResult()); ASSERT_TRUE(succeeded(info));
  for(bool majorK:{true,false}) for(int sw:{0,32,64,128}) {
    SCOPED_TRACE(::testing::Message()<<"majorK="<<majorK<<" sw="<<sw);
    auto address=oracle(majorK,sw,64,b.getAffineDimExpr(0),b.getAffineDimExpr(1));
    auto storage=layout(address,{64,64},8192);
    int64_t leading=majorK?(sw?16:1024):(sw?8*sw:1024);
    int64_t stride=majorK?(sw?8*sw:128):(sw?1024:128);
    auto descriptor=plan(majorK,sw,leading,stride,address,64,64);
    auto proof=verifySM90MmaDescriptor(*info,storage,descriptor,true,64,64,64);
    EXPECT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
    NamedAttrList bad(descriptor.getPayload());
    bad.set("stride",b.getI64IntegerAttr(stride+16));
    EXPECT_NE(verifySM90MmaDescriptor(*info,storage,
      MmaDescriptorPlanAttr::get(&ctx,bad.getDictionary(&ctx)),true,64,64,64).status,ProofStatus::Proven);
    auto weak=*info; weak.rootAlignment=8;
    EXPECT_NE(verifySM90MmaDescriptor(weak,storage,descriptor,true,64,64,64).status,ProofStatus::Proven);
  }
}

TEST_F(DescriptorProofTest, ParentPanelStrideNonzeroStartAndSwizzlePhase) {
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @test() {
      %root=memref.alloc() {alignment=1024:i64} : memref<128x64xf16,3>
      %s=memref.subview %root[1,16] [64,16] [1,1] : memref<128x64xf16,3> to memref<64x16xf16,strided<[64,1],offset:80>,3>
      %v=frisk.layout_view %s : memref<64x16xf16,strided<[64,1],offset:80>,3> -> memref<64x16xf16,strided<[64,1],offset:80>,3>
      return
    })mlir",&ctx);
  ASSERT_TRUE(module);
  LayoutViewOp view;module->walk([&](LayoutViewOp v){view=v;});
  auto info=analyzeStorageAlias(view.getResult());ASSERT_TRUE(succeeded(info));
  auto address=oracle(true,128,128,b.getAffineDimExpr(0)+1,b.getAffineDimExpr(1)+16);
  auto storage=layout(address,{64,16},16384);
  auto descriptor=plan(true,128,16,1024,address,64,16);
  auto proof=verifySM90MmaDescriptor(*info,storage,descriptor,true,64,16,64);
  ASSERT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
  auto entries=descriptor.getPayload().getAs<DenseI64ArrayAttr>("entries");
  EXPECT_EQ(entries[2],176); EXPECT_EQ(entries[3],1);
  SmallVector<int64_t> wrong(entries.asArrayRef());wrong[3]=0;
  NamedAttrList fields(descriptor.getPayload());fields.set("entries",b.getDenseI64ArrayAttr(wrong));
  EXPECT_NE(verifySM90MmaDescriptor(*info,storage,MmaDescriptorPlanAttr::get(&ctx,fields.getDictionary(&ctx)),true,64,16,64).status,ProofStatus::Proven);
  auto bounded=*info;bounded.upperBit=uint64_t(entries[2])*8;
  EXPECT_NE(verifySM90MmaDescriptor(bounded,storage,descriptor,true,64,16,64).status,ProofStatus::Proven);
}

TEST_F(DescriptorProofTest, ParentKPanelStartsAreNotGuessedFromSliceExtent) {
  auto module=parseSourceString<ModuleOp>(R"mlir(
    func.func @test() {
      %root=memref.alloc() {alignment=1024:i64} : memref<128x128xbf16,3>
      %s=memref.subview %root[0,16] [64,32] [1,1] : memref<128x128xbf16,3> to memref<64x32xbf16,strided<[128,1],offset:16>,3>
      %v=frisk.layout_view %s : memref<64x32xbf16,strided<[128,1],offset:16>,3> -> memref<64x32xbf16,strided<[128,1],offset:16>,3>
      return
    })mlir",&ctx);
  ASSERT_TRUE(module);
  LayoutViewOp view;module->walk([&](LayoutViewOp v){view=v;});
  auto info=analyzeStorageAlias(view.getResult());ASSERT_TRUE(succeeded(info));
  auto address=oracle(true,32,128,b.getAffineDimExpr(0),b.getAffineDimExpr(1)+16);
  auto storage=layout(address,{64,32},32768);
  auto descriptor=plan(true,32,16,256,address,64,32);
  auto proof=verifySM90MmaDescriptor(*info,storage,descriptor,true,64,32,64);
  ASSERT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
  auto entries=descriptor.getPayload().getAs<DenseI64ArrayAttr>("entries");
  EXPECT_EQ(entries[2],4096);EXPECT_EQ(entries[6],8192);
  SmallVector<int64_t> wrong(entries.asArrayRef());wrong[6]=wrong[2]+64*32;
  NamedAttrList fields(descriptor.getPayload());fields.set("entries",b.getDenseI64ArrayAttr(wrong));
  EXPECT_NE(verifySM90MmaDescriptor(*info,storage,MmaDescriptorPlanAttr::get(&ctx,fields.getDictionary(&ctx)),true,64,32,64).status,ProofStatus::Proven);
}
