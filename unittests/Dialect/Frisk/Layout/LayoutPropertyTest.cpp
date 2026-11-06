#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "gtest/gtest.h"
#include <random>
#include <set>
using namespace mlir;
using namespace mlir::frisk;
namespace {
// Independent bit parity, not the production apply/compose implementation.
uint64_t evaluate(ArrayRef<llvm::APInt> rows,uint64_t x) {
  uint64_t y=0;
  for (unsigned r=0;r<rows.size();++r) {
    unsigned parity=0;
    for(unsigned b=0;b<rows[r].getBitWidth();++b) parity^=rows[r][b] && ((x>>b)&1);
    y|=uint64_t(parity)<<r;
  }
  return y;
}
TEST(LayoutPropertyTest, FixedSeedsExhaustiveMatrixCoordinates) {
  for(uint64_t seed:{1u,7u,42u,20260816u}) {
    std::mt19937_64 rng(seed);
    for(unsigned c=0;c<32;++c) {
      unsigned n=1+rng()%8, m=1+rng()%8;
      SCOPED_TRACE("seed="+std::to_string(seed)+" case="+std::to_string(c));
      SmallVector<llvm::APInt> rows, rhsRows, identityRows;
      for(unsigned i=0;i<m;++i) rows.emplace_back(n,rng()&((1u<<n)-1));
      for(unsigned i=0;i<n;++i) rhsRows.emplace_back(n,rng()&((1u<<n)-1));
      for(unsigned i=0;i<m;++i) identityRows.emplace_back(m,uint64_t(1)<<i);
      auto a=GF2Matrix::get(m,n,rows), b=GF2Matrix::get(n,n,rhsRows);
      auto identity=GF2Matrix::get(m,m,identityRows);
      ASSERT_TRUE(succeeded(a)); ASSERT_TRUE(succeeded(b)); ASSERT_TRUE(succeeded(identity));
      auto composed=a->compose(*b), same=identity->compose(*a);
      ASSERT_TRUE(succeeded(composed)); ASSERT_TRUE(succeeded(same)); EXPECT_EQ(*same,*a);
      std::set<uint64_t> image;
      for(uint64_t x=0;x<(1u<<n);++x) {
        SCOPED_TRACE("first coordinate="+std::to_string(x));
        auto y=evaluate(rows,x); image.insert(y);
        ASSERT_EQ(a->apply(llvm::APInt(n,x)).getZExtValue(),y);
        ASSERT_EQ(composed->apply(llvm::APInt(n,x)).getZExtValue(),evaluate(rows,evaluate(rhsRows,x)));
      }
      auto inverse=a->inverse(), right=a->rightInverse();
      bool surjective=image.size()==(1u<<m), invertible=n==m && surjective;
      ASSERT_EQ(succeeded(inverse),invertible); ASSERT_EQ(succeeded(right),surjective);
      if(invertible) for(uint64_t x=0;x<(1u<<n);++x)
        ASSERT_EQ(inverse->apply(llvm::APInt(m,evaluate(rows,x))).getZExtValue(),x);
      if(surjective) for(uint64_t y=0;y<(1u<<m);++y)
        ASSERT_EQ(evaluate(rows,right->apply(llvm::APInt(m,y)).getZExtValue()),y);
      SmallVector<llvm::APInt> zero(m,llvm::APInt(n,0));
      auto singular=GF2Matrix::get(m,n,zero); ASSERT_TRUE(succeeded(singular));
      EXPECT_TRUE(failed(singular->inverse())); EXPECT_TRUE(failed(singular->rightInverse()));
      SmallVector<llvm::APInt> wrong(n+1,llvm::APInt(1,0));
      EXPECT_TRUE(failed(a->compose(*GF2Matrix::get(n+1,1,wrong))));
    }
  }
}
TEST(LayoutPropertyTest, CanonicalizationAndProductStagedCoordinates) {
  MLIRContext ctx; ctx.getOrLoadDialect<FriskDialect>();
  for(uint64_t seed:{1u,7u,42u,20260816u}) {
    std::mt19937_64 rng(seed);
    for(unsigned c=0;c<32;++c) {
      SCOPED_TRACE("seed="+std::to_string(seed)+" case="+std::to_string(c));
      unsigned bits=1+rng()%4, split=1u<<bits, outer=1+rng()%4;
      std::string matrix="[";
      for(unsigned r=0;r<bits;++r) {
        if(r)matrix+=",";matrix+="[";
        for(unsigned b=0;b<bits;++b){if(b)matrix+=",";matrix+=(r==b?"1":"0");}
        matrix+="]";
      }
      matrix+="]";
      std::string text="#frisk.product<outer = #frisk.affine_layout<inputs = [\"o\"], input_extents = ["+
        std::to_string(outer)+"], outputs = [\"x\"], output_extents = ["+std::to_string(outer*split)+
        "], map = affine_map<(d0) -> (d0 * "+std::to_string(split)+")>>, inner = #frisk.bit_linear<inputs = [\"i\"], input_bits = ["+
        std::to_string(bits)+"], outputs = [\"x\"], output_bits = ["+std::to_string(bits)+
        "], matrix = dense<"+matrix+"> : tensor<"+std::to_string(bits)+"x"+std::to_string(bits)+"xi1>>, split_extents = ["+std::to_string(split)+"]>";
      auto p=dyn_cast_or_null<ProductLayoutMapAttr>(parseAttribute(text,&ctx)); ASSERT_TRUE(p);
      auto canonical=p.canonicalizeMap(); ASSERT_TRUE(succeeded(canonical));
      auto twice=cast<LayoutMapAttrInterface>(*canonical).canonicalizeMap();
      ASSERT_TRUE(succeeded(twice)); EXPECT_EQ(*twice,*canonical);
      EXPECT_EQ(checkCoverage(p,{int64_t(split*outer)}).status,ProofStatus::Proven);
      for(unsigned o=0;o<outer;++o)for(unsigned i=0;i<split;++i) {
        SCOPED_TRACE("coordinate="+std::to_string(o)+","+std::to_string(i));
        auto value=evaluateLayoutMap(p,{int64_t(o),int64_t(i)});
        ASSERT_TRUE(succeeded(value)); ASSERT_EQ(value->size(),1u);
        ASSERT_EQ((*value)[0],int64_t(o*split+i));
      }
    }
  }
}
} // namespace
