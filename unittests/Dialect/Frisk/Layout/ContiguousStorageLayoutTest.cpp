#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/IR/ContiguousStorageLayout.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
class ContiguousStorageLayoutTest : public testing::Test {
protected:
  MLIRContext context;
  ContiguousStorageLayoutTest() { context.loadDialect<FriskDialect, func::FuncDialect>(); }
  MemRefType type(int64_t m = 4, int64_t n = 8, Type element = {},
                  attr::MemorySpace space = attr::MemorySpace::Global,
                  MemRefLayoutAttrInterface descriptor = {}) {
    Builder b(&context);
    return MemRefType::get({m,n}, element ? element : b.getF32Type(), descriptor,
                          MemorySpaceAttr::get(&context,space));
  }
  StorageLayoutAttr layout(int64_t m = 4, int64_t n = 8, AffineExpr bytes = {},
                           AffineExpr bits = {}, int64_t outputBytes = -1,
                           attr::MemorySpace space = attr::MemorySpace::Global,
                           int64_t alignment = 1, int64_t vector = 1) {
    Builder b(&context);
    if (!bytes) bytes = (b.getAffineDimExpr(0)*n+b.getAffineDimExpr(1))*4;
    if (!bits) bits = b.getAffineConstantExpr(0);
    auto map = AffineLayoutMapAttr::get(&context,b.getStrArrayAttr({"row","column"}),
        b.getDenseI64ArrayAttr({m,n}),b.getStrArrayAttr({"byte_offset","bit_offset"}),
        b.getDenseI64ArrayAttr({outputBytes < 0 ? 4*m*n : outputBytes,8}),
        AffineMapAttr::get(AffineMap::get(2,0,{bytes,bits},&context)));
    return StorageLayoutAttr::get(&context,map,MemorySpaceAttr::get(&context,space),
        b.getI64IntegerAttr(alignment),b.getI64IntegerAttr(vector));
  }
};
TEST_F(ContiguousStorageLayoutTest, ExactAlgebraAndLargestFullFootprintAgree) {
  auto t=type(512,128);auto l=layout(512,128);
  auto descriptor=matchContiguousIdentityStorage(l,t);ASSERT_TRUE(descriptor);
  EXPECT_EQ(descriptor->elements,65536u);EXPECT_EQ(descriptor->bytes,262144u);
  EXPECT_TRUE(succeeded(l.verifyForType(t,UnknownLoc::get(&context))));
  auto bytes=getStorageFootprintBytes(l,t);ASSERT_TRUE(succeeded(bytes));EXPECT_EQ(*bytes,262144u);
  auto module=parseSourceString<ModuleOp>("func.func @f(%r: memref<512x128xf32,1>) { return }",&context);
  ASSERT_TRUE(module);auto root=analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));
  ASSERT_TRUE(succeeded(root));ASSERT_TRUE(matchContiguousIdentityStorageAlias(*root,l));
  EXPECT_EQ(verifyStorageAliasCandidate(*root,l).status,ProofStatus::Proven);
  // Data-consuming callers must still receive the complete dense footprint.
  auto footprint=buildStorageAliasFootprint(*root,l);ASSERT_EQ(footprint.proof.status,ProofStatus::Proven);
  ASSERT_EQ(footprint.entries.size(),65536u);
  for (uint64_t i=0;i<65536;++i) {
    const auto &entry=footprint.entries[i];
    EXPECT_EQ(entry.view,SmallVector<int64_t>({int64_t(i/128),int64_t(i%128)}));
    EXPECT_EQ(entry.root,entry.view);EXPECT_EQ(entry.begin,32*i);EXPECT_EQ(entry.end,32*(i+1));
  }
}
TEST_F(ContiguousStorageLayoutTest, EquivalentUnsupportedExpressionsUseDenseProof) {
  Builder b(&context);auto x=b.getAffineDimExpr(0),y=b.getAffineDimExpr(1);
  // floor(x/4) is zero on this domain, but is not an admitted AST node.
  auto l=layout(4,8,((x*8+y)+x.floorDiv(4))*4);
  EXPECT_FALSE(matchContiguousIdentityStorage(l,type()));
  EXPECT_TRUE(succeeded(l.verifyForType(type(),UnknownLoc::get(&context))));
  auto bytes=getStorageFootprintBytes(l,type());ASSERT_TRUE(succeeded(bytes));EXPECT_EQ(*bytes,128u);
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,(x*8+y)*4-x.floorDiv(4)),type()));
}
TEST_F(ContiguousStorageLayoutTest, RefusesDifferentMappingAndUnsafeExpression) {
  Builder b(&context);auto x=b.getAffineDimExpr(0),y=b.getAffineDimExpr(1);
  for (AffineExpr expr : {4*(x+4*y),4*(8*x+y)+4,x*int64_t(9223372036854775807)+4*y})
    EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,expr),type()));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,{},b.getAffineConstantExpr(1)),type()));
}
TEST_F(ContiguousStorageLayoutTest, RefusesUnsupportedTypeDomainAndDescriptor) {
  Builder b(&context);
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(4,8,b.getF16Type())));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(4,8,b.getI32Type())));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(4,8,{},attr::MemorySpace::Shared)));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,{}, {},-1,attr::MemorySpace::Shared),type()));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(4,8,{},attr::MemorySpace::Global,
      StridedLayoutAttr::get(&context,1,{8,1}))));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(4,8,{},attr::MemorySpace::Global,
      StridedLayoutAttr::get(&context,0,{9,1}))));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(1024,128),type(1024,128)));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(ShapedType::kDynamic,8)));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),type(0,8)));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,{}, {},-1,attr::MemorySpace::Global,2),type()));
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,{}, {},-1,attr::MemorySpace::Global,2,2),type()));
  auto rank1=MemRefType::get({32},b.getF32Type(),{},1);
  EXPECT_FALSE(matchContiguousIdentityStorage(layout(),rank1));
}
TEST_F(ContiguousStorageLayoutTest, RefusesMetadataDriftEvenWithSameFormula) {
  Builder b(&context);auto good=layout();auto map=cast<AffineLayoutMapAttr>(good.getMap());
  for (auto inputs : {b.getStrArrayAttr({"row","row"}),b.getStrArrayAttr({"","column"})}) {
    auto changed=AffineLayoutMapAttr::get(&context,inputs,map.getInputExtents(),map.getOutputNames(),map.getOutputExtents(),map.getAffineMap());
    auto l=StorageLayoutAttr::get(&context,changed,good.getMemorySpace(),good.getAlignment(),good.getVectorGranularity());
    EXPECT_FALSE(matchContiguousIdentityStorage(l,type()));
  }
  for (int64_t extent : {127,129}) EXPECT_FALSE(matchContiguousIdentityStorage(layout(4,8,{}, {},extent),type()));
  auto changed=AffineLayoutMapAttr::get(&context,map.getInputNames(),map.getInputExtents(),
      b.getStrArrayAttr({"bit_offset","byte_offset"}),map.getOutputExtents(),map.getAffineMap());
  EXPECT_FALSE(matchContiguousIdentityStorage(StorageLayoutAttr::get(&context,changed,
      good.getMemorySpace(),good.getAlignment(),good.getVectorGranularity()),type()));
  auto symbol=AffineLayoutMapAttr::get(&context,map.getInputNames(),map.getInputExtents(),
      map.getOutputNames(),map.getOutputExtents(),AffineMapAttr::get(AffineMap::get(2,1,
          {b.getAffineDimExpr(0)*32+b.getAffineDimExpr(1)*4+b.getAffineSymbolExpr(0),b.getAffineConstantExpr(0)},&context)));
  EXPECT_FALSE(matchContiguousIdentityStorage(StorageLayoutAttr::get(&context,symbol,
      good.getMemorySpace(),good.getAlignment(),good.getVectorGranularity()),type()));
}
TEST_F(ContiguousStorageLayoutTest, NonIdentifierLabelsRetainDenseProof) {
  Builder b(&context);auto good=layout();auto map=cast<AffineLayoutMapAttr>(good.getMap());
  for (StringRef name : {"row]", "row-tag", "row\\tag", "2row", "row label"}) {
    auto changed=AffineLayoutMapAttr::get(&context,b.getStrArrayAttr({name,"column"}),
        map.getInputExtents(),map.getOutputNames(),map.getOutputExtents(),map.getAffineMap());
    auto l=StorageLayoutAttr::get(&context,changed,good.getMemorySpace(),
        good.getAlignment(),good.getVectorGranularity());
    EXPECT_FALSE(matchContiguousIdentityStorage(l,type()));
    // Refusing compression must not remove support for a valid original map.
    EXPECT_TRUE(succeeded(l.verifyForType(type(),UnknownLoc::get(&context))));
    auto bytes=getStorageFootprintBytes(l,type());ASSERT_TRUE(succeeded(bytes));
    EXPECT_EQ(*bytes,128u);
  }
}
TEST_F(ContiguousStorageLayoutTest, SameShapeIsNotRootMappingOrAliasProof) {
  auto module=parseSourceString<ModuleOp>("func.func @f(%r: memref<4x8xf32,1>) { return }",&context);
  ASSERT_TRUE(module);auto root=analyzeStorageAlias(module->lookupSymbol<func::FuncOp>("f").getArgument(0));ASSERT_TRUE(succeeded(root));
  auto row=layout();Builder b(&context);auto column=layout(4,8,(b.getAffineDimExpr(0)+4*b.getAffineDimExpr(1))*4);
  EXPECT_EQ(verifyStorageAliasCandidate(*root,column).status,ProofStatus::Proven);
  EXPECT_FALSE(matchContiguousIdentityStorageAlias(*root,column));
  EXPECT_EQ(proveStorageAliasCompatible(*root,row,*root,column).status,ProofStatus::Disproven);
  auto bad=*root;bad.upperBit-=32;EXPECT_FALSE(matchContiguousIdentityStorageAlias(bad,row));
  EXPECT_NE(verifyStorageAliasCandidate(bad,row).status,ProofStatus::Proven);
  bad=*root;bad.viewToRoot=AffineMap::get(2,0,{b.getAffineDimExpr(0),7-b.getAffineDimExpr(1)},&context);
  EXPECT_FALSE(matchContiguousIdentityStorageAlias(bad,row));
  EXPECT_EQ(proveStorageAliasCompatible(*root,row,bad,row).status,ProofStatus::Disproven);
  bad=*root;bad.viewToRoot=AffineMap::getMultiDimIdentityMap(1,&context);
  EXPECT_FALSE(matchContiguousIdentityStorageAlias(bad,row));
}
} // namespace
