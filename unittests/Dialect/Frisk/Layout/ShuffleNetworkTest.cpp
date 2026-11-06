#include "Dialect/Frisk/Analysis/ShuffleNetwork.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <numeric>
#include <random>
using namespace mlir::frisk::shuffle_network;
namespace {
uint64_t swapBits(uint64_t x,unsigned a,unsigned b) {
 if(((x>>a)^(x>>b))&1)x^=(uint64_t{1}<<a)|(uint64_t{1}<<b);return x;
}
std::vector<uint64_t> transpose(unsigned r,unsigned exchanged) {
 unsigned rb=0;for(unsigned x=r;x>1;x>>=1)++rb;
 std::vector<uint64_t> owners(128*r);
 for(unsigned i=0;i<owners.size();++i) {
  uint64_t x=i;for(unsigned bit=0;bit<exchanged;++bit)x=swapBits(x,bit,rb+bit);owners[i]=x;
 }
 return owners;
}
TEST(ShuffleNetworkTest, FiveBitTransposeUsesFusedCommunicationWithExactWork) {
 auto owners=transpose(512,5);auto p=plan(512,512,owners);ASSERT_TRUE(p);
 EXPECT_EQ(p->kind,"bit-permutation-fused");EXPECT_EQ(p->counts.shfl,496u);
 EXPECT_EQ(p->counts.selects,5120u);EXPECT_EQ(p->counts.mapInstructions,10u);
 EXPECT_EQ(p->counts.totalWork,5626u);EXPECT_EQ(p->counts.shuffleWarpIssues,1984u);
 EXPECT_EQ(p->counts.addressAluWarpIssues,20520u);EXPECT_EQ(p->legacyShfl,16384u);
 EXPECT_EQ(p->legacyWork,59904u);EXPECT_EQ(p->verifiedHolders,65536u);
 for(unsigned i=0;i<owners.size();++i)ASSERT_EQ(sourceHolder(*p,i),owners[i]);
}
TEST(ShuffleNetworkTest, ExplicitMinWorkRetainsVerifiedCommunicationResourceTradeoff) {
 auto owners=transpose(512,5);auto communication=plan(512,512,owners);
 auto work=plan(512,512,owners,Objective::MinWork);ASSERT_TRUE(communication);ASSERT_TRUE(work);
 EXPECT_EQ(communication->kind,"bit-permutation-fused");EXPECT_EQ(work->kind,"bit-permutation-ship");
 EXPECT_EQ(work->counts.shfl,1280u);EXPECT_EQ(work->counts.selects,3840u);
 EXPECT_EQ(work->counts.mapInstructions,10u);EXPECT_EQ(work->counts.totalWork,5130u);
 EXPECT_FALSE(sameExecutionPlan(*communication,*work));
 for(auto *selected:{&*communication,&*work}) {
  EXPECT_LE(selected->counts.shfl,selected->legacyShfl);EXPECT_LE(selected->counts.totalWork,selected->legacyWork);
  for(unsigned i=0;i<owners.size();++i)ASSERT_EQ(sourceHolder(*selected,i),owners[i]);
 }
 auto cr=structuralResources(*communication),wr=structuralResources(*work);
 EXPECT_EQ(cr.logicalCarrierWords,512u);EXPECT_EQ(wr.logicalCarrierWords,512u);
 EXPECT_EQ(cr.b32TemporaryDefinitions,5621u);EXPECT_EQ(wr.b32TemporaryDefinitions,5125u);
 EXPECT_EQ(cr.predicateDefinitions,5u);EXPECT_EQ(wr.predicateDefinitions,5u);
 EXPECT_EQ(cr.communicationStages,1u);EXPECT_EQ(wr.communicationStages,5u);
 auto alternatives=frontier(512,512,owners);EXPECT_EQ(alternatives.size(),2u);
 EXPECT_TRUE(sameExecutionPlan(*communication,*communication));
 auto relabeled=*communication;relabeled.kind="diagnostic-label";EXPECT_TRUE(sameExecutionPlan(*communication,relabeled));
 relabeled.counts.shfl++;EXPECT_FALSE(sameExecutionPlan(*communication,relabeled));
}
TEST(ShuffleNetworkTest, ObjectiveLabelsAloneDoNotCreateExecutableVariants) {
 auto owners=transpose(8,1);auto a=plan(8,8,owners),b=plan(8,8,owners,Objective::MinWork);
 ASSERT_TRUE(a);ASSERT_TRUE(b);EXPECT_TRUE(sameExecutionPlan(*a,*b));
 EXPECT_FALSE(plan(8,8,owners,static_cast<Objective>(99)));
}
TEST(ShuffleNetworkTest, SingleMixedBitChoosesShipAtEqualCommunication) {
 auto owners=transpose(8,1);auto p=plan(8,8,owners);ASSERT_TRUE(p);
 EXPECT_EQ(p->kind,"bit-permutation-ship");EXPECT_EQ(p->counts.shfl,4u);
 EXPECT_EQ(p->counts.selects,12u);EXPECT_EQ(p->counts.mapInstructions,2u);
 for(unsigned i=0;i<owners.size();++i)ASSERT_EQ(sourceHolder(*p,i),owners[i]);
}
TEST(ShuffleNetworkTest, AffineWarpOffsetsAndConstantPreserveEveryOwner) {
 auto owners=transpose(32,3);
 for(unsigned i=0;i<owners.size();++i) {
  unsigned warp=i/(32*32);owners[i]^=7u^((warp&1)?(1u<<2):0)^((warp&2)?(1u<<8):0);
 }
 auto p=plan(32,32,owners);ASSERT_TRUE(p);EXPECT_EQ(p->affineBase,7u);
 for(unsigned i=0;i<owners.size();++i)ASSERT_EQ(sourceHolder(*p,i),owners[i]);
}
TEST(ShuffleNetworkTest, DenseInvertibleMapsUseVerifiedShearsOrBoundedFallback) {
 std::mt19937 rng(74136);unsigned accepted=0;
 for(unsigned trial=0;trial<100;++trial) {
  constexpr unsigned r=32,dimension=10;
  std::vector<uint64_t> rows(dimension);for(unsigned i=0;i<dimension;++i)rows[i]=1u<<i;
  for(unsigned operation=0;operation<40;++operation) {
   unsigned a=rng()%dimension,b=rng()%dimension;if(a==b)continue;
   if(rng()%5==0)std::swap(rows[a],rows[b]);else rows[a]^=rows[b];
  }
  std::vector<uint64_t> owners(128*r);
  for(unsigned i=0;i<owners.size();++i) {
   unsigned warp=i/(32*r);uint64_t low=i%(32*r),value=0;
   for(unsigned b=0;b<dimension;++b)if(__builtin_parityll(rows[b]&low))value|=1u<<b;
   owners[i]=warp*(32*r)+value;
  }
  auto p=plan(r,r,owners);if(!p)continue;++accepted;
  EXPECT_EQ(p->kind,"linear-shear");EXPECT_LE(p->counts.totalWork,p->legacyWork);
  EXPECT_LE(p->counts.shfl,p->legacyShfl);
  for(unsigned i=0;i<owners.size();++i)ASSERT_EQ(sourceHolder(*p,i),owners[i]);
 }
 EXPECT_GT(accepted,0u);
}
TEST(ShuffleNetworkTest, LowerTotalWorkCannotPermitExtraShuffleCommunication) {
 // This static affine counterexample has a 392-work/92-SHFL shear network,
 // versus 456-work/64-SHFL exact support. Transport must not regress merely
 // because the aggregate work proxy is smaller. A future genuinely better
 // network may be accepted, provided both independent bounds still hold.
 constexpr unsigned registers=8,dimension=8;
 const std::vector<uint64_t> columns={1,38,226,7,21,249,120,225};
 std::vector<uint64_t> owners(128*registers);
 for(unsigned index=0;index<owners.size();++index) {
  uint64_t low=index%(32*registers),source=0;
  for(unsigned bit=0;bit<dimension;++bit)if(low&(1u<<bit))source^=columns[bit];
  owners[index]=(index/(32*registers))*(32*registers)+source;
 }
 uint64_t legacyShfl=0;
 for(unsigned reg=0;reg<registers;++reg) {
  unsigned support=0;
  for(unsigned thread=0;thread<128;++thread)
   support|=1u<<(owners[thread*registers+reg]%registers);
  legacyShfl+=__builtin_popcount(support);
 }
 ASSERT_EQ(legacyShfl,64u);
 if(auto selected=plan(registers,registers,owners)) {
  EXPECT_LE(selected->counts.shfl,legacyShfl);
  EXPECT_LE(selected->counts.totalWork,456u);
  for(unsigned index=0;index<owners.size();++index)
   ASSERT_EQ(sourceHolder(*selected,index),owners[index]);
 }
}
TEST(ShuffleNetworkTest, NonlinearBijectiveOwnerMutationCannotClaimLinearNetwork) {
 auto owners=transpose(8,2);std::swap(owners[0],owners[1]);EXPECT_FALSE(plan(8,8,owners));
}
TEST(ShuffleNetworkTest, ReplicatedAndCrossWarpOwnersKeepExactSupportFallback) {
 auto owners=transpose(8,2);
 for(auto &owner:owners)owner&=~uint64_t{1};EXPECT_FALSE(plan(8,8,owners));
 owners=transpose(8,2);owners[0]^=32*8;EXPECT_FALSE(plan(8,8,owners));
}
TEST(ShuffleNetworkTest, MismatchedOrMalformedCarrierDomainIsNotZeroCost) {
 auto owners=transpose(8,2);EXPECT_FALSE(plan(16,8,owners));owners.pop_back();EXPECT_FALSE(plan(8,8,owners));
 EXPECT_FALSE(plan(0,0,{}));EXPECT_FALSE(plan(3,3,std::vector<uint64_t>(128*3)));
}
} // namespace
