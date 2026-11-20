#include "Dialect/Frisk/Analysis/LayoutDemand.h"
#include "Dialect/Frisk/Analysis/LayoutDemandFootprint.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include <set>
#include <random>
#include <limits>
#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
using namespace mlir;using namespace mlir::frisk;
namespace {
// Deliberately retain the independent pre-optimization set/map algorithm as
// the oracle. Test data includes misaligned f32 addresses, unlike most fixtures.
mlir::frisk::detail::DemandAccessFootprint referenceFootprint(
    const std::array<uint64_t,128> &offsets,const std::array<bool,128> &active,
    bool global,uint64_t alignment) {
  mlir::frisk::detail::DemandAccessFootprint result;
  for(unsigned phase=0;phase<32;phase+=alignment) {
    uint64_t sectors=0,excess=0;
    for(unsigned warp=0;warp<4;++warp) {
      std::set<uint64_t> touched;
      std::map<unsigned,std::set<uint64_t>> words;
      bool any=false;
      for(unsigned lane=0;lane<32;++lane) {
        unsigned t=warp*32+lane;if(!active[t])continue;any=true;
        for(uint64_t n=(offsets[t]+phase)/32;n<=(offsets[t]+3+phase)/32;++n)
          touched.insert(n);
        for(uint64_t n=(offsets[t]+phase)/4;n<=(offsets[t]+3+phase)/4;++n)
          words[n%32].insert(n);
      }
      if(phase==0 && any)++result.warpIssues;
      sectors+=touched.size();uint64_t degree=1;
      for(auto &[bank,values]:words)degree=std::max<uint64_t>(degree,values.size());
      excess+=degree-1;
    }
    if(global)result.sectors=std::max(result.sectors,sectors);
    else result.bankExcess=std::max(result.bankExcess,excess);
  }
  return result;
}
void expectSameFootprint(const mlir::frisk::detail::DemandAccessFootprint &a,
                         const mlir::frisk::detail::DemandAccessFootprint &b) {
  EXPECT_EQ(a.warpIssues,b.warpIssues);EXPECT_EQ(a.sectors,b.sectors);
  EXPECT_EQ(a.bankExcess,b.bankExcess);
}
TEST(LayoutDemandFootprintTest, IntegerTranslationMemoMatchesSetOracle) {
  std::mt19937_64 rng(0x6f776e6572);
  mlir::frisk::detail::DemandAccessFootprintCache cache;
  LayoutProfile profile(true);LayoutProfile::Invocation invocation(&profile);
  for(unsigned sample=0;sample<80;++sample) {
    std::array<uint64_t,128> offsets{};std::array<bool,128> active{};
    for(unsigned t=0;t<128;++t) {
      // All-active read, first-owner-like write masks, broadcasts, duplicates,
      // sparse/empty warps, and offsets on both sides of word/sector edges.
      offsets[t]=sample%3==0?(t%8)*4:sample%3==1?t*4+31:rng()%8192;
      active[t]=sample%5==0?true:sample%5==1?t%32==0:
                sample%5==2?t<32:sample%5==3?bool(rng()&1):false;
    }
    for(bool global:{true,false})for(uint64_t alignment:{1,2,3,4,8,16,31,32}) {
      SCOPED_TRACE(sample);
      SCOPED_TRACE(global);
      SCOPED_TRACE(alignment);
      auto expected=referenceFootprint(offsets,active,global,alignment);
      auto noMemo=cache.evaluate(offsets,active,global,alignment,0);
      auto memo=cache.evaluate(offsets,active,global,alignment,256);
      ASSERT_TRUE(noMemo);ASSERT_TRUE(memo);
      expectSameFootprint(*noMemo,expected);expectSameFootprint(*memo,expected);
      auto shifted=offsets;uint64_t unit=global?32:4;
      for(unsigned t=0;t<128;++t)
        shifted[t]=active[t]?offsets[t]+unit*(17+t/32):UINT64_MAX;
      auto translated=cache.evaluate(shifted,active,global,alignment,256);
      ASSERT_TRUE(translated);expectSameFootprint(*translated,expected);
      EXPECT_LE(cache.size(),256u);
    }
  }
  EXPECT_EQ(cache.size(),256u);
  EXPECT_GT(profile.get(LayoutProfileKind::DemandFootprintCache).hits,0u);
  EXPECT_GT(profile.get(LayoutProfileKind::DemandFootprintCache).misses,256u);
  EXPECT_EQ(profile.get(LayoutProfileKind::DemandFootprintCache).entries,256u);
  EXPECT_EQ(profile.get(LayoutProfileKind::DemandFootprintCache).peakEntries,256u);
}
TEST(LayoutDemandFootprintTest, OriginalOverflowCannotBeHiddenByNormalizedHit) {
  mlir::frisk::detail::DemandAccessFootprintCache cache;
  std::array<uint64_t,128> offsets{};std::array<bool,128> active{};
  active[0]=active[33]=true;offsets[0]=offsets[33]=4;
  ASSERT_TRUE(cache.evaluate(offsets,active,true,1,256));
  for(bool global:{true,false})for(uint64_t alignment:{1,3,4,16,32}) {
    uint64_t tail=(31/alignment)*alignment+3;
    offsets[0]=offsets[33]=UINT64_MAX-tail;
    auto valid=cache.evaluate(offsets,active,global,alignment,256);
    ASSERT_TRUE(valid);expectSameFootprint(*valid,referenceFootprint(offsets,active,global,alignment));
    size_t previous=cache.size();offsets[0]++;
    EXPECT_FALSE(cache.evaluate(offsets,active,global,alignment,256));
    EXPECT_EQ(cache.size(),previous);
    // Inactive addresses cannot contribute to the footprint or overflow.
    active[0]=false;offsets[0]=UINT64_MAX;
    EXPECT_TRUE(cache.evaluate(offsets,active,global,alignment,256));active[0]=true;
  }
  EXPECT_FALSE(cache.evaluate(offsets,active,true,0,256));
  EXPECT_FALSE(cache.evaluate(offsets,active,true,33,256));
}
class LayoutDemandTest:public testing::Test {
protected:
 MLIRContext context;
 LayoutDemandTest(){context.loadDialect<FriskDialect,arith::ArithDialect,math::MathDialect,scf::SCFDialect,func::FuncDialect,memref::MemRefDialect>();}
 template<class F,class M>void withGraph(F f,M mutate,StringRef filename=FRISK_DEMAND_MIXED_FIXTURE) {
  auto module=parseSourceFile<ModuleOp>(filename,&context);ASSERT_TRUE(module);
  mutate(*module);
  auto target=createSM90LayoutTarget();auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));
  SolverOptions options;options.policy.conversion=ConversionPolicy::Independent;
  auto solution=solveLayoutGraph(*graph,*target,options);ASSERT_TRUE(succeeded(solution));
  f(*graph,CandidateAssignment{solution->assignments});
 }
 template<class F>void withGraph(F f) {withGraph(f,[](ModuleOp){});}
 template<class F>void withFinalStoreRepair(F f) {
  auto module=parseSourceFile<ModuleOp>(FRISK_DEMAND_FINAL_STORE_FIXTURE,&context);ASSERT_TRUE(module);
  auto target=createSM90LayoutTarget();auto graph=collectLayoutConstraints(*module,*target);ASSERT_TRUE(succeeded(graph));
  auto bindings=(*module)->getAttrOfType<DictionaryAttr>("frisk.test_assignments");ASSERT_TRUE(bindings);
  CandidateAssignment assignment;
  for(const auto &v:graph->getVariables()){auto value=bindings.get(v.stableName);ASSERT_TRUE(value)<<v.stableName;assignment.values[v.id]=value;}
  ASSERT_EQ(assignment.values.size(),bindings.size());f(*module,*graph,assignment);
 }
 // Exercise the pairwise execution-domain precheck on a finalized graph.
 // A third, unassigned endpoint deliberately defers the whole-kernel ABI proof.
 template<class F>void withBackendConversion(unsigned logicalBits,bool crossWarp,F f,int swapColumnOverride=-1) {
  Builder b(&context);unsigned registerBits=logicalBits-7;
  int64_t elements=int64_t{1}<<logicalBits,registers=int64_t{1}<<registerBits;
  auto type=RankedTensorType::get({elements},b.getF32Type());
  auto makeEncoding=[&](bool permute) {
   unsigned swapColumn=swapColumnOverride>=0?unsigned(swapColumnOverride):crossWarp?registerBits+5:1;
   SmallVector<APInt> bits;
   for(unsigned row=0;row<logicalBits;++row)for(unsigned col=0;col<logicalBits;++col) {
    unsigned selected=permute && row==0?swapColumn:permute && row==swapColumn?0:row;
    bits.emplace_back(1,col==selected);
   }
   auto map=BitLinearLayoutMapAttr::get(&context,
    b.getArrayAttr({b.getStringAttr("register"),b.getStringAttr("lane"),b.getStringAttr("warp")}),
    b.getDenseI64ArrayAttr({int64_t(registerBits),5,2}),
    b.getArrayAttr({b.getStringAttr("dim0")}),b.getDenseI64ArrayAttr({int64_t(logicalBits)}),
    DenseIntElementsAttr::get(RankedTensorType::get({logicalBits,logicalBits},b.getI1Type()),bits));
   return DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({registers,32,4,1,1}),b.getI64IntegerAttr(1));
  };
  auto source=makeEncoding(false),destination=makeEncoding(true);
  // Both maps are bijections with valid ownership, including the 64 KiB
  // case. Backend rejection must not be interpreted as hardware illegality.
  for(auto encoding:{source,destination}) {
   LinearExecutionLayout linear;
   ASSERT_EQ(analyzeLinearExecutionLayout(encoding,type,linear).status,ProofStatus::Proven);
   ASSERT_EQ(proveExecutionOwnership(encoding,type,"first_owner").status,ProofStatus::Proven);
  }
  std::string tensor="tensor<"+std::to_string(elements)+"xf32>";
  auto module=parseSourceString<ModuleOp>("func.func @conversion(%x: "+tensor+", %y: "+tensor+") -> "+tensor+" { %sum = arith.addf %x, %y : "+tensor+" return %sum : "+tensor+" }",&context);
  ASSERT_TRUE(module);arith::AddFOp use;module->walk([&](arith::AddFOp op){use=op;});ASSERT_TRUE(use);
  LayoutConstraintGraph graph;
  auto src=graph.addVariable(LayoutKind::Distributed,type,"source");graph.getVariable(src).value=use.getLhs();
  auto dst=graph.addVariable(LayoutKind::Distributed,type,"target");graph.getVariable(dst).use=&use->getOpOperand(0);
  graph.addVariable(LayoutKind::Distributed,type,"unassigned");
  auto c=graph.addConstraint(ConstraintKind::Convertible,ConstraintStrength::Hard,{src,dst},use,"static-backend-boundary","pairwise presearch support");
  graph.getConstraints()[c].use=&use->getOpOperand(0);graph.getConstraints()[c].stableUseKey="conversion/operand0";
  ASSERT_TRUE(succeeded(graph.finalize(UnknownLoc::get(&context))));
  CandidateAssignment assignment;assignment.values[*graph.lookupVariable("source")]=source;assignment.values[*graph.lookupVariable("target")]=destination;
  f(graph,assignment,uint64_t(elements)*4);
 }
};
TEST_F(LayoutDemandTest, StaticSharedBackendLimitIsSeparateFromValidHardwareOwnership) {
 for(unsigned logicalBits:{13u,14u}) {
  SCOPED_TRACE(logicalBits);
  withBackendConversion(logicalBits,true,[&](const auto &g,const auto &a,uint64_t bytes) {
   auto proof=verifyLayoutDemandBackendSupport(g,a);
   if(bytes==32768)EXPECT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
   else {
    ASSERT_EQ(bytes,65536u);EXPECT_EQ(proof.status,ProofStatus::Disproven);
    EXPECT_NE(proof.reason.find("backend unsupported"),std::string::npos)<<proof.reason;
    EXPECT_NE(proof.reason.find("static backend limit49152 bytes"),std::string::npos)<<proof.reason;
   }
  });
 }
}
TEST_F(LayoutDemandTest, StaticSharedLimitDoesNotRejectSameShapeRegisterConversion) {
 withBackendConversion(14,false,[&](const auto &g,const auto &a,uint64_t bytes) {
  ASSERT_EQ(bytes,65536u);auto proof=verifyLayoutDemandBackendSupport(g,a);
  EXPECT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
 });
}
TEST_F(LayoutDemandTest, NetworkShuffleChargesActualTransportIncludingWarpDependence) {
 for(bool warpDependent:{false,true}) {
  Operation *added=nullptr;
  withGraph([&](const auto &g,const auto &a){
   auto ledger=analyzeLayoutDemand(g,a);ASSERT_TRUE(ledger.exact)<<ledger.reason;
   unsigned matched=0;
   for(const auto &[id,b]:ledger.boundaries) {
    auto use=g.getConstraint(id).use;if(!use || use->getOwner()!=added)continue;
    ++matched;EXPECT_EQ(b.exchange,ConversionExchangeKind::Shuffle);
    EXPECT_EQ(b.sourceVectors,8u);EXPECT_EQ(b.targetVectors,8u);
    // One mixed bit uses four static Ship shuffles. A warp-controlled
    // register offset adds eight selects and a separate two-instruction
    // predicate, while reusing the same verified transport network.
    EXPECT_EQ(b.shuffleWarpIssues,16u);
    EXPECT_EQ(b.bytes,16u*32*4);
    EXPECT_EQ(b.addressAluWarpIssues,warpDependent?96u:56u);
   }
   EXPECT_EQ(matched,1u);
  },[&](ModuleOp module){
   Operation *store=nullptr,*load=nullptr;
   module.walk([&](Operation *o){if(o->getName().getStringRef()=="frisk.tile_store")store=o;if(!load && o->getName().getStringRef()=="frisk.tile_load")load=o;});
   ASSERT_TRUE(store && load);auto type=cast<RankedTensorType>(load->getResult(0).getType());
   auto enc=cast<DistributedEncodingAttr>(type.getEncoding());ASSERT_EQ(enc.getTopology()[0],8);
   auto map=cast<BitLinearLayoutMapAttr>(enc.getMap());auto mt=cast<RankedTensorType>(map.getMatrix().getType());
   SmallVector<APInt> bits(map.getMatrix().getValues<APInt>());
   for(int64_t row=0;row<mt.getDimSize(0);++row) {
    std::swap(bits[row*mt.getDimSize(1)],bits[row*mt.getDimSize(1)+3]);
    if(warpDependent)bits[row*mt.getDimSize(1)+8]^=bits[row*mt.getDimSize(1)+1];
   }
   auto changed=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),map.getInputBitWidths(),map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(mt,bits));
   auto output=RankedTensorType::get(type.getShape(),type.getElementType(),DistributedEncodingAttr::get(&context,changed,enc.getTopology(),enc.getReplication()));
   OpBuilder builder(store);OperationState conversion(store->getLoc(),"frisk.convert_layout");conversion.addOperands(load->getResult(0));conversion.addTypes(output);added=builder.create(conversion);
   builder.create<math::ExpOp>(store->getLoc(),added->getResult(0));
  },FRISK_DEMAND_ORDER_EAGER_FIXTURE);
 }
}
TEST_F(LayoutDemandTest, ExplicitBackendDomainAccepts512ShuffleAt65536LogicalElements) {
 withBackendConversion(16,false,[&](const auto &g,const auto &a,uint64_t bytes) {
  ASSERT_EQ(bytes,262144u);auto proof=verifyLayoutDemandBackendSupport(g,a);
  EXPECT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
 },9); // register0 <-> lane0; same warp, no Shared allocation.
}
TEST_F(LayoutDemandTest, Bounded512ShufflePlanCopiesEveryGF2OwnerWithinRealWorkCap) {
 Builder b(&context);auto loc=UnknownLoc::get(&context);
 for(unsigned rank:{1u,7u}) {
  auto make=[&](bool target,Type element) {
   SmallVector<APInt> bits;
   for(unsigned row=0;row<16;++row)for(unsigned col=0;col<16;++col)
    bits.emplace_back(1,row==col || (target && row<rank && col==9+row) || (target && row==9 && col==8));
   auto map=BitLinearLayoutMapAttr::get(&context,b.getArrayAttr({b.getStringAttr("register"),b.getStringAttr("lane"),b.getStringAttr("warp")}),b.getDenseI64ArrayAttr({9,5,2}),b.getArrayAttr({b.getStringAttr("dim0")}),b.getDenseI64ArrayAttr({16}),DenseIntElementsAttr::get(RankedTensorType::get({16,16},b.getI1Type()),bits));
   auto enc=DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({512,32,4,1,1}),b.getI64IntegerAttr(1));
   return RankedTensorType::get({65536},element,enc);
  };
  auto source=make(false,b.getF32Type()),destination=make(true,b.getF32Type());
  auto plan=test::planRedistribution(source,destination,loc);ASSERT_TRUE(succeeded(plan));
  ASSERT_EQ(plan->mode,test::ExchangeMode::Shuffle);ASSERT_EQ(plan->destinations.size(),65536u);ASSERT_EQ(plan->sourceSlots.size(),65536u);
  LinearExecutionLayout linear;ASSERT_EQ(analyzeLinearExecutionLayout(cast<DistributedEncodingAttr>(destination.getEncoding()),destination,linear).status,ProofStatus::Proven);
  for(unsigned t=0;t<128;++t)for(unsigned r=0;r<512;++r) {
   const auto &entry=plan->destinations[t*512+r];uint64_t expected=linear.apply(APInt(16,uint64_t(t)*512+r)).getZExtValue();
   EXPECT_EQ(entry.slot,expected);EXPECT_EQ((entry.source.warp*32+entry.source.lane)*512+entry.source.reg,expected);EXPECT_EQ(entry.source.warp,t/32);
  }
  if(rank==7) {
   // The same support with two payload words exceeds the unchanged65536 cap.
   EXPECT_TRUE(failed(test::planRedistribution(make(false,b.getF64Type()),make(true,b.getF64Type()),loc)));
  }
 }
}
TEST_F(LayoutDemandTest, BoundedFootprintMemoPreservesFullReadWriteAndLocalLedgers) {
 withGraph([&](auto &g,const auto &a){
  auto text=[](const auto &l){return llvm::formatv("{0}",llvm::json::Value(layoutDemandJSON(l))).str();};
  g.setCostQueryCacheOptions(false,0);auto reference=analyzeLayoutDemand(g,a);
  ASSERT_TRUE(reference.exact)<<reference.reason;ASSERT_FALSE(reference.accesses.empty());
  LayoutVarID large=0;for(auto &v:g.getVariables())
   if(v.value && !v.use && v.kind==LayoutKind::Distributed &&
      cast<DistributedEncodingAttr>(a.values.lookup(v.id)).getTopology()[0]==1024)large=v.id;
  auto localReference=analyzeLayoutDemand(g,a,{large});ASSERT_TRUE(localReference.exact);
  for(size_t capacity:{1u,16u,256u,4096u}) {
   SCOPED_TRACE(capacity);g.setCostQueryCacheOptions(true,capacity);
   for(unsigned warm=0;warm<2;++warm) {
    auto actual=analyzeLayoutDemand(g,a);ASSERT_TRUE(actual.exact)<<actual.reason;
    EXPECT_EQ(text(actual),text(reference));
    auto local=analyzeLayoutDemand(g,a,{large});ASSERT_TRUE(local.exact)<<local.reason;
    EXPECT_EQ(text(local),text(localReference));
   }
  }
 });
}
TEST_F(LayoutDemandTest, StoreRepairLedgerIsIdenticalWithFootprintMemoDisabled) {
 withFinalStoreRepair([&](ModuleOp,auto &g,const auto &a){
  auto text=[](const auto &l){return llvm::formatv("{0}",llvm::json::Value(layoutDemandJSON(l))).str();};
  g.setCostQueryCacheOptions(false,0);auto reference=analyzeLayoutDemand(g,a);
  ASSERT_TRUE(reference.exact)<<reference.reason;
  for(size_t capacity:{1u,256u,4096u}) {
   g.setCostQueryCacheOptions(true,capacity);
   for(unsigned warm=0;warm<2;++warm) {
    auto actual=analyzeLayoutDemand(g,a);ASSERT_TRUE(actual.exact)<<actual.reason;
    EXPECT_EQ(text(actual),text(reference));
   }
  }
 });
}
TEST_F(LayoutDemandTest, AffineConsumerQueriesMatchHybridBackendExecution) {
 withGraph([&](const auto &g,const auto &a){auto l=analyzeLayoutDemand(g,a);ASSERT_TRUE(l.exact)<<l.reason;EXPECT_TRUE(l.demandLowering);
  EXPECT_EQ(l.loadVectors,24u);EXPECT_EQ(l.materializedQueryVectors,104u);EXPECT_LT(l.materializedQueryVectors,l.declaredRegisterVectors);
  uint64_t access=0,shared=0;for(auto &[id,c]:l.accesses)access+=c.warpIssues;
  for(auto &[id,b]:l.boundaries)if(b.exchange==ConversionExchangeKind::Shared){++shared;EXPECT_EQ(b.sourceVectors,8u);EXPECT_EQ(b.targetVectors,8u);EXPECT_EQ(b.barriers,2u);}
  EXPECT_EQ(access,128u);EXPECT_EQ(shared,4u);
  for(auto &[id,v]:l.values)if(g.getVariable(id).value && !g.getVariable(id).use && cast<DistributedEncodingAttr>(a.values.lookup(id)).getTopology()[0]==1024)EXPECT_EQ(v.queries.size(),8u);
 });
}
TEST_F(LayoutDemandTest, UnknownOrBudgetExceededNeverBecomesZeroCostCertificate) {
 withGraph([&](const auto &g,const auto &a){auto limited=analyzeLayoutDemand(g,a,{},1);EXPECT_FALSE(limited.exact);EXPECT_NE(limited.reason.find("budget"),std::string::npos);auto incomplete=a;incomplete.values.erase(incomplete.values.begin());auto missing=analyzeLayoutDemand(g,incomplete);EXPECT_FALSE(missing.exact);EXPECT_EQ(missing.reason,"incomplete demand assignment");});
}
TEST_F(LayoutDemandTest, ProducerLocalRootDoesNotUseFutureConsumerDemand) {
 withGraph([&](const auto &g,const auto &a){LayoutVarID large=0;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && cast<DistributedEncodingAttr>(a.values.lookup(v.id)).getTopology()[0]==1024)large=v.id;
  auto local=analyzeLayoutDemand(g,a,{large});ASSERT_TRUE(local.exact)<<local.reason;EXPECT_TRUE(local.localScope);EXPECT_EQ(local.loadVectors,1024u);uint64_t issues=0;for(auto &[id,c]:local.accesses)issues+=c.warpIssues;EXPECT_EQ(issues,4096u);EXPECT_TRUE(local.boundaries.empty());
 });
}
TEST_F(LayoutDemandTest, LocalExecutorModeCannotObserveUnreachableFutureLargeValue) {
 withGraph([&](const auto &g,const auto &a){LayoutVarID small=0;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && v.value.getDefiningOp()->getName().getStringRef()=="frisk.tile_load" && cast<DistributedEncodingAttr>(a.values.lookup(v.id)).getTopology()[0]==8){small=v.id;break;}
  auto local=analyzeLayoutDemand(g,a,{small});ASSERT_TRUE(local.exact)<<local.reason;EXPECT_FALSE(local.demandLowering);EXPECT_EQ(local.loadVectors,8u);
 });
}
TEST_F(LayoutDemandTest, DemandScoreUsesActualAccessCoverageAndExplicitModelFormat) {
 withGraph([&](const auto &g,const auto &a){SM90CostProfile p;p.executionDemand=true;p.coefficients.fill(0);p.coefficients[0]=1;p.provenance="query-coverage-unit";
  auto text=serializeSM90CostProfile(p);EXPECT_NE(text.find("sm90-layout-demand-calibrated-v1"),std::string::npos);auto parsed=parseSM90CostProfile(text);ASSERT_TRUE(bool(parsed));EXPECT_TRUE(parsed->executionDemand);EXPECT_EQ(serializeSM90CostProfile(*parsed),text);EXPECT_LE(getSM90CalibratedCostModelID(p).size(),64u);
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;auto cost=evaluateSM90DemandLayoutCost(g,a,policy,&p);ASSERT_TRUE(succeeded(cost));EXPECT_EQ(cost->cost.instructionPathAndWork,128u);
  p.executionDemand=false;EXPECT_EQ(serializeSM90CostProfile(p).find("demand-calibrated"),std::string::npos);
 });
}
TEST_F(LayoutDemandTest, CommonExecutionDomainProofRetainsNoKnownPartialConflict) {
 withGraph([&](const auto &g,const auto &a){EXPECT_EQ(verifyLayoutDemandBackendSupport(g,a).status,ProofStatus::Proven);auto partial=a;partial.values.clear();// A partial assignment with no known counterexample remains searchable;
  // final complete-certificate validation is still mandatory.
  EXPECT_EQ(verifyLayoutDemandBackendSupport(g,partial).status,ProofStatus::Proven);});
}
TEST_F(LayoutDemandTest, DeclaredDeadSharedScratchDoesNotLeakIntoIndependentLocalRoot) {
 withGraph([&](const auto &g,const auto &a){
  auto full=analyzeLayoutDemand(g,a);ASSERT_TRUE(full.exact)<<full.reason;
  EXPECT_TRUE(full.demandLowering);EXPECT_TRUE(full.boundaries.empty());
  EXPECT_EQ(full.scratchMaxBytes,4096u);
  LayoutVarID large=0;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && cast<DistributedEncodingAttr>(a.values.lookup(v.id)).getTopology()[0]==1024)large=v.id;
  auto local=analyzeLayoutDemand(g,a,{large});ASSERT_TRUE(local.exact)<<local.reason;
  EXPECT_EQ(local.scratchMaxBytes,0u);EXPECT_TRUE(local.boundaries.empty());
  SM90CostProfile p;p.executionDemand=true;p.coefficients[0]=1;p.provenance="declared-scratch-unit";
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.sharedBytesAndOccupancy,4096u);
 },[&](ModuleOp module){
  Operation *store=nullptr;Value large;
  module.walk([&](Operation *op){if(op->getName().getStringRef()=="frisk.tile_store")store=op;
   if(op->getName().getStringRef()=="frisk.tile_load") {auto t=cast<RankedTensorType>(op->getResult(0).getType());if(cast<DistributedEncodingAttr>(t.getEncoding()).getTopology()[0]==1024)large=op->getResult(0);}});
  ASSERT_TRUE(store && large);store->setOperand(0,large);
 });
}
TEST_F(LayoutDemandTest, MaterializedReceiverMuxIsIncludedInChargedBoundaryAndALUTotal) {
 withGraph([&](const auto &g,const auto &a){
  auto l=analyzeLayoutDemand(g,a);ASSERT_TRUE(l.exact)<<l.reason;
  uint64_t mux=0,address=0;unsigned variableBoundaries=0;
  for(auto &[id,x]:l.accesses)address+=x.addressAluWarpIssues;
  for(auto &[id,b]:l.boundaries){address+=b.addressAluWarpIssues;mux+=b.receiverMuxAluWarpIssues;
   if(b.receiverMuxAluWarpIssues){++variableBoundaries;EXPECT_EQ(b.exchange,ConversionExchangeKind::Shared);EXPECT_EQ(b.receiverMuxAluWarpIssues,640u);EXPECT_GE(b.addressAluWarpIssues,b.receiverMuxAluWarpIssues);}}
  EXPECT_EQ(variableBoundaries,1u);EXPECT_EQ(mux,640u);EXPECT_EQ(address,l.addressAluWarpIssues);
  SM90CostProfile p;p.executionDemand=true;p.coefficients[6]=1;p.provenance="receiver-mux-unit";
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.instructionPathAndWork,l.addressAluWarpIssues+l.arithmeticWarpIssues);
 },[&](ModuleOp module){
  Operation *store=nullptr,*shared=nullptr;
  module.walk([&](Operation *op){if(op->getName().getStringRef()=="frisk.tile_store")store=op;
   if(!shared && op->getName().getStringRef()=="frisk.convert_layout")shared=op;});
  ASSERT_TRUE(store && shared);auto type=cast<RankedTensorType>(shared->getResult(0).getType());
  auto e=cast<DistributedEncodingAttr>(type.getEncoding());auto map=cast<BitLinearLayoutMapAttr>(e.getMap());
  auto mt=cast<RankedTensorType>(map.getMatrix().getType());SmallVector<APInt> bits(map.getMatrix().getValues<APInt>());
  // Elementary input-column operation: destination lane[0] also toggles
  // register[0]. The route stays Register, with source q(r,t)=r XOR(t&1).
  for(int64_t row=0;row<mt.getDimSize(0);++row)bits[row*mt.getDimSize(1)+3]^=bits[row*mt.getDimSize(1)];
  auto twist=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),map.getInputBitWidths(),map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(mt,bits));
  auto encoding=DistributedEncodingAttr::get(&context,twist,e.getTopology(),e.getReplication());
  auto output=RankedTensorType::get(type.getShape(),type.getElementType(),encoding);
  OpBuilder builder(store);OperationState state(store->getLoc(),"frisk.convert_layout");state.addOperands(shared->getResult(0));state.addTypes(output);
  auto *conversion=builder.create(state);store->setOperand(0,conversion->getResult(0));
 });
}
TEST_F(LayoutDemandTest, ExplicitConvertDAGUsesDemandTargetAndBankOnlyProfileIsValid) {
 withGraph([&](const auto &g,const auto &a){
  SM90CostProfile p;p.executionDemand=true;p.coefficients[0]=1;p.provenance="target-pure-route-unit";
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));EXPECT_EQ(cost->cost.instructionPathAndWork,128u);
  p.coefficients.fill(0);p.demandBankCoefficients={2,3};EXPECT_TRUE(succeeded(verifySM90CostProfile(p)));
  auto parsed=parseSM90CostProfile(serializeSM90CostProfile(p));ASSERT_TRUE(bool(parsed));
  target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto banks=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(banks));
  auto ledger=analyzeLayoutDemand(g,a);uint64_t expected=0;for(auto &[id,b]:ledger.boundaries)expected+=2*b.sharedReadBankExcess+3*b.sharedWriteBankExcess;
  EXPECT_EQ(banks->cost.instructionPathAndWork,expected);
  policy.conversion=ConversionPolicy::Full;EXPECT_TRUE(failed(scoreExperimentAssignment(g,a,*target,policy)));
 });
}
TEST_F(LayoutDemandTest, ExecutionDomainRejectsNonWeakRootDeclaration) {
 withGraph([&](const auto &g,const auto &a){
  auto proof=verifyLayoutDemandBackendSupport(g,a);
  EXPECT_EQ(proof.status,ProofStatus::Disproven);EXPECT_NE(proof.reason.find("weak-alignment1"),std::string::npos)<<proof.reason;
 },[&](ModuleOp module){
  Value chosen;module.walk([&](StorageContractOp declaration){if(!chosen)chosen=declaration.getRoot();});ASSERT_TRUE(chosen);
  module.walk([&](StorageContractOp declaration){if(declaration.getRoot()!=chosen)return;
   auto old=declaration.getLayout();Builder b(&context);
   declaration->setAttr("layout",StorageLayoutAttr::get(&context,old.getMap(),old.getMemorySpace(),b.getI64IntegerAttr(2),old.getVectorGranularity()));});
 });
}
TEST_F(LayoutDemandTest, ExecutionDomainIncludesUnusedDeclaredRoots) {
 for(bool shared:{false,true})withGraph([&](const auto &g,const auto &a){
  auto proof=verifyLayoutDemandBackendSupport(g,a);EXPECT_EQ(proof.status,ProofStatus::Disproven);
  EXPECT_NE(proof.reason.find(shared?"including unused roots":"exactly four"),std::string::npos)<<proof.reason;
 },[&](ModuleOp module){
  Operation *contract=nullptr;module.walk([&](Operation *op){if(!contract && op->getName().getStringRef()=="frisk.storage_contract")contract=op;});ASSERT_TRUE(contract);
  auto old=cast<StorageLayoutAttr>(contract->getAttr("layout"));auto oldType=cast<MemRefType>(contract->getOperand(0).getType());OpBuilder builder(contract);
  auto type=MemRefType::get(oldType.getShape(),oldType.getElementType(),MemRefLayoutAttrInterface{},builder.getI64IntegerAttr(shared?3:1));
  auto unused=builder.create<memref::AllocOp>(contract->getLoc(),type);
  auto layout=StorageLayoutAttr::get(&context,old.getMap(),MemorySpaceAttr::get(&context,shared?attr::MemorySpace::Shared:attr::MemorySpace::Global),builder.getI64IntegerAttr(1),builder.getI64IntegerAttr(1));
  OperationState declaration(contract->getLoc(),"frisk.storage_contract");declaration.addOperands(unused.getResult());declaration.addAttribute("layout",layout);builder.create(declaration);
 });
}
TEST_F(LayoutDemandTest, NestedControlIsWholeGraphCarrierFallbackAndNotExactDemandDomain) {
 withGraph([&](const auto &g,const auto &a){
  auto ledger=analyzeLayoutDemand(g,a);EXPECT_FALSE(ledger.exact);
  auto proof=verifyLayoutDemandBackendSupport(g,a);EXPECT_EQ(proof.status,ProofStatus::Disproven);
  SM90CostProfile p;p.executionDemand=true;p.coefficients[0]=1;p.provenance="control-fallback-unit";
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto fallback=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(fallback));
  p.executionDemand=false;auto carrier=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto expected=scoreExperimentAssignment(g,a,*carrier,policy);ASSERT_TRUE(succeeded(expected));
  EXPECT_EQ(fallback->cost.instructionPathAndWork,expected->cost.instructionPathAndWork);
 },[&](ModuleOp module){
  Operation *parallel=nullptr;module.walk([&](Operation *op){if(!parallel && op->getName().getStringRef()=="frisk.parallel")parallel=op;});ASSERT_TRUE(parallel);
  OpBuilder builder(parallel);auto zero=builder.create<arith::ConstantIndexOp>(parallel->getLoc(),0),one=builder.create<arith::ConstantIndexOp>(parallel->getLoc(),1);
  auto loop=builder.create<scf::ForOp>(parallel->getLoc(),zero,one,one);parallel->moveBefore(loop.getBody()->getTerminator());
 });
}
TEST_F(LayoutDemandTest, UnknownEnclosingExecutionContainerIsNotAnExactCertificate) {
 withGraph([&](const auto &g,const auto &a){
  Operation *parallel=nullptr;for(auto &v:g.getVariables())if(v.kind==LayoutKind::Distributed && !v.use && v.value) {parallel=v.value.getDefiningOp()->getParentOp();break;}
  ASSERT_TRUE(parallel && isa<ParallelOp>(parallel));context.allowUnregisteredDialects();
  OpBuilder builder(parallel);OperationState state(parallel->getLoc(),"test.unknown_execution_container");state.addRegion();auto *wrapper=builder.create(state);
  auto *body=new Block();wrapper->getRegion(0).push_back(body);parallel->moveBefore(body,body->end());
  auto ledger=analyzeLayoutDemand(g,a);EXPECT_FALSE(ledger.exact);EXPECT_NE(ledger.reason.find("unsupported enclosing execution container"),std::string::npos);
  auto proof=verifyLayoutDemandBackendSupport(g,a);EXPECT_EQ(proof.status,ProofStatus::Disproven);
 });
}
TEST_F(LayoutDemandTest, NonSplatTensorConstantIsOutsideTheScalarDemandDomain) {
 withGraph([&](const auto &g,const auto &a){
  auto ledger=analyzeLayoutDemand(g,a);EXPECT_FALSE(ledger.exact);EXPECT_NE(ledger.reason.find("scalar splat"),std::string::npos);
  auto proof=verifyLayoutDemandBackendSupport(g,a);EXPECT_EQ(proof.status,ProofStatus::Disproven);
 },[&](ModuleOp module){
  Operation *store=nullptr;RankedTensorType type;module.walk([&](Operation *op){if(op->getName().getStringRef()=="frisk.tile_store")store=op;
   if(!type && op->getName().getStringRef()=="frisk.tile_load")type=cast<RankedTensorType>(op->getResult(0).getType());});ASSERT_TRUE(store && type);
  SmallVector<float> values(type.getNumElements(),1.0f);values.back()=2.0f;OpBuilder builder(store);
  builder.create<arith::ConstantOp>(store->getLoc(),type,DenseElementsAttr::get(type,ArrayRef<float>(values)));
 });
}
TEST_F(LayoutDemandTest, CompactSourceRegisterLimitIsCheckedBeforeDemandPruning) {
 withGraph([&](const auto &g,const auto &a){
  LayoutVarID large=0;bool found=false;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && cast<DistributedEncodingAttr>(a.values.lookup(v.id)).getTopology()[0]==1024){large=v.id;found=true;break;}ASSERT_TRUE(found);
  auto original=cast<DistributedEncodingAttr>(a.values.lookup(large));auto map=cast<BitLinearLayoutMapAttr>(original.getMap());auto mt=cast<RankedTensorType>(map.getMatrix().getType());
  SmallVector<APInt> old(map.getMatrix().template getValues<APInt>());Builder b(&context);
  auto extend=[&](unsigned regBits){
   unsigned extra=regBits-10,columns=mt.getDimSize(1)+extra;SmallVector<APInt> matrix(mt.getDimSize(0)*columns,APInt(1,0));
   for(unsigned row=0;row<mt.getDimSize(0);++row)for(unsigned col=0;col<mt.getDimSize(1);++col)matrix[row*columns+col+(col>=10?extra:0)]=old[row*mt.getDimSize(1)+col];
   SmallVector<int64_t> widths(map.getInputBitWidths().asArrayRef());widths[0]=regBits;
   auto extended=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),b.getDenseI64ArrayAttr(widths),map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(RankedTensorType::get({mt.getDimSize(0),columns},b.getI1Type()),matrix));
   int64_t nr=int64_t{1}<<regBits;return DistributedEncodingAttr::get(&context,extended,b.getDenseI64ArrayAttr({nr,32,4,1,1}),b.getI64IntegerAttr(nr/8));
  };
  auto boundary=a;boundary.values[large]=extend(16);auto allowed=verifyLayoutDemandBackendSupport(g,boundary);EXPECT_EQ(allowed.status,ProofStatus::Proven)<<allowed.reason;
  auto beyond=a;beyond.values[large]=extend(17);auto rejected=verifyLayoutDemandBackendSupport(g,beyond);EXPECT_EQ(rejected.status,ProofStatus::Disproven);EXPECT_NE(rejected.reason.find("at most65536 declared registers"),std::string::npos)<<rejected.reason;
 });
}
TEST_F(LayoutDemandTest, EachDemandedScalarSplatChargesOneALUMovePerWarp) {
 withGraph([&](const auto &g,const auto &a){
  auto ledger=analyzeLayoutDemand(g,a);ASSERT_TRUE(ledger.exact)<<ledger.reason;
  EXPECT_EQ(ledger.arithmeticWarpIssues,32u);EXPECT_EQ(ledger.loadVectors,0u);
  SM90CostProfile p;p.executionDemand=true;p.coefficients[6]=1;p.provenance="constant-mov-grammar-unit";
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));
  EXPECT_EQ(cost->cost.instructionPathAndWork,ledger.addressAluWarpIssues+32u);
 },[&](ModuleOp module){
  Operation *store=nullptr;RankedTensorType type;module.walk([&](Operation *op){if(op->getName().getStringRef()=="frisk.tile_store")store=op;
   if(!type && op->getName().getStringRef()=="frisk.tile_load")type=cast<RankedTensorType>(op->getResult(0).getType());});ASSERT_TRUE(store && type);
  float value=0.25f;OpBuilder builder(store);auto constant=builder.create<arith::ConstantOp>(store->getLoc(),type,DenseElementsAttr::get(type,ArrayRef<float>(value)));store->setOperand(0,constant.getResult());
 });
}
TEST_F(LayoutDemandTest, KnownEqualWorkMixedImagesHaveSixteenVersusOneLoadIssueGroups) {
 uint64_t dfsGroups=0,eagerGroups=0;std::map<std::string,uint64_t> dfsFeatures,eagerFeatures;
 for(bool dfs:{true,false})withGraph([&](const auto &g,const auto &a){
  auto ledger=analyzeLayoutDemand(g,a);ASSERT_TRUE(ledger.exact)<<ledger.reason;
  EXPECT_EQ(ledger.globalLoadIssueGroups,dfs?16u:1u);(dfs?dfsGroups:eagerGroups)=ledger.globalLoadIssueGroups;
  SM90CostProfile legacy;legacy.executionDemand=true;legacy.coefficients.fill(1);legacy.demandBankCoefficients={1,1};legacy.provenance="frozen-v4-cost-regression";
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto old=evaluateSM90DemandLayoutCost(g,a,policy,&legacy);ASSERT_TRUE(succeeded(old));
  auto &features=dfs?dfsFeatures:eagerFeatures;for(auto &f:old->features)features[f.first]=f.second;
  EXPECT_EQ(features["access_warp_issues"],128u);EXPECT_EQ(features["global_sectors32"],640u);EXPECT_EQ(features["alu_warp_issues"],5952u);EXPECT_EQ(features["div_warp_issues"],32u);EXPECT_EQ(features["exp_warp_issues"],32u);
  EXPECT_EQ(old->cost.instructionPathAndWork,6784u);EXPECT_FALSE(features.count("global_load_issue_groups"));
  auto frozenText=serializeSM90CostProfile(legacy);auto frozenID=getSM90CalibratedCostModelID(legacy);
  auto frozen=parseSM90CostProfile(frozenText);ASSERT_TRUE(bool(frozen));EXPECT_EQ(serializeSM90CostProfile(*frozen),frozenText);EXPECT_EQ(getSM90CalibratedCostModelID(*frozen),frozenID);EXPECT_NE(frozenID.find("v4d"),std::string::npos);
  auto ordered=legacy;ordered.executionOrder=true;ordered.coefficients.fill(0);ordered.demandBankCoefficients={0,0};ordered.loadIssueGroupCoefficient=3;
  auto text=serializeSM90CostProfile(ordered);EXPECT_NE(text.find("sm90-layout-demand-calibrated-v2"),std::string::npos);
  auto parsed=parseSM90CostProfile(text);ASSERT_TRUE(bool(parsed));EXPECT_TRUE(parsed->executionOrder);EXPECT_EQ(serializeSM90CostProfile(*parsed),text);EXPECT_NE(getSM90CalibratedCostModelID(*parsed).find("v5d"),std::string::npos);EXPECT_LE(getSM90CalibratedCostModelID(*parsed).size(),64u);
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,*parsed);auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));EXPECT_EQ(cost->cost.instructionPathAndWork,3*ledger.globalLoadIssueGroups);
 },[](ModuleOp){},dfs?FRISK_DEMAND_ORDER_DFS_FIXTURE:FRISK_DEMAND_ORDER_EAGER_FIXTURE);
 EXPECT_EQ(dfsGroups,16u);EXPECT_EQ(eagerGroups,1u);EXPECT_EQ(dfsFeatures,eagerFeatures);
}
TEST_F(LayoutDemandTest, EagerLoadGroupingUsesIROrderRatherThanFinalizedVariableIDs) {
 withGraph([&](const auto &g,const auto &a){
  auto initial=analyzeLayoutDemand(g,a);ASSERT_TRUE(initial.exact)<<initial.reason;EXPECT_EQ(initial.globalLoadIssueGroups,1u);
  Operation *lastLoad=nullptr,*lastMultiply=nullptr;
  for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed) {
   auto *op=v.value.getDefiningOp();auto name=op->getName().getStringRef();
   if(name=="frisk.tile_load" && (!lastLoad || lastLoad->isBeforeInBlock(op)))lastLoad=op;
   if(name=="arith.mulf" && (!lastMultiply || lastMultiply->isBeforeInBlock(op)))lastMultiply=op;
  }
  ASSERT_TRUE(lastLoad && lastMultiply);lastLoad->moveAfter(lastMultiply);
  auto reordered=analyzeLayoutDemand(g,a);ASSERT_TRUE(reordered.exact)<<reordered.reason;
  EXPECT_EQ(reordered.globalLoadIssueGroups,2u);EXPECT_EQ(reordered.loadVectors,initial.loadVectors);EXPECT_EQ(reordered.arithmeticWarpIssues,initial.arithmeticWarpIssues);EXPECT_EQ(reordered.addressAluWarpIssues,initial.addressAluWarpIssues);
  LayoutVarID root=0;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && v.value.getDefiningOp()->getName().getStringRef()=="arith.subf")root=v.id;
  auto local=analyzeLayoutDemand(g,a,{root});ASSERT_TRUE(local.exact)<<local.reason;EXPECT_FALSE(local.demandLowering);EXPECT_EQ(local.globalLoadIssueGroups,2u);
 },[](ModuleOp){},FRISK_DEMAND_ORDER_EAGER_FIXTURE);
}
TEST_F(LayoutDemandTest, LocalEagerIssueGroupsIgnoreUnreachableFutureLargeLayout) {
 withGraph([&](const auto &g,const auto &a){
  LayoutVarID producer=0;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && v.value.getDefiningOp()->getName().getStringRef()=="arith.mulf"){producer=v.id;break;}
  auto local=analyzeLayoutDemand(g,a,{producer});ASSERT_TRUE(local.exact)<<local.reason;EXPECT_FALSE(local.demandLowering);EXPECT_EQ(local.loadVectors,16u);EXPECT_EQ(local.globalLoadIssueGroups,1u);
 });
}
TEST_F(LayoutDemandTest, OrderedLocalRootSetIsInvariantToCallerPermutation) {
 withGraph([&](const auto &g,const auto &a){
  LayoutVarID large=0,producer=0;bool haveLarge=false,haveProducer=false;
  for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed) {
   if(cast<DistributedEncodingAttr>(a.values.lookup(v.id)).getTopology()[0]==1024){large=v.id;haveLarge=true;}
   if(!haveProducer && v.value.getDefiningOp()->getName().getStringRef()=="arith.mulf"){producer=v.id;haveProducer=true;}
  }
  ASSERT_TRUE(haveLarge && haveProducer);auto first=analyzeLayoutDemand(g,a,{large,producer}),second=analyzeLayoutDemand(g,a,{producer,large});
  ASSERT_TRUE(first.exact)<<first.reason;ASSERT_TRUE(second.exact)<<second.reason;
  EXPECT_TRUE(first.demandLowering);EXPECT_EQ(first.globalLoadIssueGroups,8u);EXPECT_EQ(first.globalLoadIssueGroups,second.globalLoadIssueGroups);EXPECT_EQ(first.loadVectors,second.loadVectors);EXPECT_EQ(first.materializedQueryVectors,second.materializedQueryVectors);
  SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.loadIssueGroupCoefficient=1;p.provenance="root-set-order-unit";auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;policy.scope.fullGraph=false;policy.scope.informedLocal=true;policy.scope.producers={large,producer};
  auto aCost=scoreExperimentAssignment(g,a,*target,policy);policy.scope.producers={producer,large};auto bCost=scoreExperimentAssignment(g,a,*target,policy);
  ASSERT_TRUE(succeeded(aCost));ASSERT_TRUE(succeeded(bCost));EXPECT_EQ(aCost->cost.instructionPathAndWork,bCost->cost.instructionPathAndWork);
 });
}
TEST_F(LayoutDemandTest, EagerFinalStoreUseRepairHasQueriesCommunicationAndPrice) {
 withFinalStoreRepair([&](ModuleOp module,const auto &g,const auto &a){
  const LayoutConstraint *repair=nullptr;for(const auto &c:g.getConstraints())if(c.kind==ConstraintKind::Convertible && c.use && c.use->getOwner()->getName().getStringRef()=="frisk.tile_store")repair=&c;
  ASSERT_TRUE(repair);ASSERT_TRUE(g.getVariable(repair->vars[1]).use);EXPECT_FALSE(repair->existingConversion);
  auto ledger=analyzeLayoutDemand(g,a);ASSERT_TRUE(ledger.exact)<<ledger.reason;EXPECT_FALSE(ledger.demandLowering);
  EXPECT_EQ(ledger.materializedQueryVectors,384u);EXPECT_EQ(ledger.declaredRegisterVectors,384u);EXPECT_EQ(ledger.loadVectors,96u);EXPECT_EQ(ledger.globalLoadIssueGroups,1u);EXPECT_EQ(ledger.scratchMaxBytes,16384u);
  ASSERT_TRUE(ledger.boundaries.count(repair->id));const auto &b=ledger.boundaries.at(repair->id);
  EXPECT_EQ(b.exchange,ConversionExchangeKind::Shared);EXPECT_EQ(b.queries.size(),32u);EXPECT_EQ(b.sourceVectors,32u);EXPECT_EQ(b.targetVectors,32u);EXPECT_EQ(b.bytes,32768u);EXPECT_EQ(b.barriers,2u);EXPECT_EQ(b.addressAluWarpIssues,15488u);EXPECT_EQ(b.sharedBankExcess,384u);EXPECT_EQ(b.sharedWriteBankExcess,384u);EXPECT_EQ(b.sharedReadBankExcess,0u);
  EXPECT_EQ(verifyLayoutDemandBackendSupport(g,a).status,ProofStatus::Proven);
  auto capped=analyzeLayoutDemand(g,a,{},383);EXPECT_FALSE(capped.exact);EXPECT_NE(capped.reason.find("budget"),std::string::npos);
  SM90CostProfile profile;profile.executionDemand=true;profile.executionOrder=true;profile.coefficients[12]=1;profile.coefficients[14]=1;profile.provenance="store-use-repair-unit";
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,profile);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));EXPECT_EQ(cost->cost.instructionPathAndWork,131080u);
  uint64_t bytes=0,barriers=0;for(auto &f:cost->features){if(f.first=="conversion_shared_bytes")bytes=f.second;if(f.first=="synchronization_barriers")barriers=f.second;}
  EXPECT_EQ(bytes,131072u);EXPECT_EQ(barriers,8u);
 });
}
TEST_F(LayoutDemandTest, CommonEagerQueryBudgetIncludesFinalStoreRepair) {
 for(unsigned deadConstants:{2036u,2037u})withFinalStoreRepair([&](ModuleOp module,const auto &g,const auto &a){
  llvm::DenseMap<Value,Attribute> values;llvm::DenseMap<OpOperand*,Attribute> uses;Operation *store=nullptr;RankedTensorType constantType;
  for(const auto &v:g.getVariables()) {
   if(v.use)uses[v.use]=a.values.lookup(v.id);else if(v.value)values[v.value]=a.values.lookup(v.id);
   if(v.kind==LayoutKind::Distributed && !v.use && v.value && !constantType) {auto t=cast<RankedTensorType>(v.shapedType);constantType=RankedTensorType::get(t.getShape(),t.getElementType(),a.values.lookup(v.id));}
  }
  module.walk([&](Operation *op){if(op->getName().getStringRef()=="frisk.tile_store")store=op;});ASSERT_TRUE(store && constantType);
  OpBuilder builder(store);float scalar=0.25f;auto splat=DenseElementsAttr::get(constantType,ArrayRef<float>(scalar));
  for(unsigned i=0;i<deadConstants;++i)builder.create<arith::ConstantOp>(store->getLoc(),constantType,splat);
  auto target=createSM90LayoutTarget();auto expanded=collectLayoutConstraints(module,*target);ASSERT_TRUE(succeeded(expanded));CandidateAssignment full;
  for(const auto &v:expanded->getVariables()) {
   Attribute encoding=v.use?uses.lookup(v.use):values.lookup(v.value);
   if(!encoding && v.kind==LayoutKind::Distributed)encoding=cast<RankedTensorType>(v.shapedType).getEncoding();
   ASSERT_TRUE(encoding)<<v.stableName;full.values[v.id]=encoding;
  }
  auto proof=verifyLayoutDemandBackendSupport(*expanded,full);
  if(deadConstants==2036)EXPECT_EQ(proof.status,ProofStatus::Proven)<<proof.reason; //65536 actual queries.
  else {EXPECT_EQ(proof.status,ProofStatus::Disproven);EXPECT_NE(proof.reason.find("query budget"),std::string::npos)<<proof.reason;} //65568, including store repair.
 });
}
TEST_F(LayoutDemandTest, SurvivingWholeSSADeletesDeadReadWithoutClaimingNativeCSE) {
 withGraph([&](const auto &g,const auto &a){
  auto x=analyzeLayoutSurvivingDemand(g,a);ASSERT_TRUE(x.proven)<<x.reason;
  EXPECT_EQ(x.emitted.loadVectors,24u);EXPECT_EQ(x.surviving.loadVectors,16u);EXPECT_EQ(x.removedQueryVectors,32u);
  EXPECT_EQ(x.emitted.materializedQueryVectors,64u);EXPECT_EQ(x.surviving.materializedQueryVectors,32u);EXPECT_EQ(x.surviving.arithmeticWarpIssues,64u);EXPECT_TRUE(x.retainedCommunication.empty());
  auto json=layoutSurvivingDemandJSON(x);EXPECT_FALSE(json.getBoolean("native_exact").value_or(true));ASSERT_TRUE(json.getObject("surviving"));EXPECT_FALSE(json.getObject("surviving")->getInteger("global_load_issue_groups").has_value());
  SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.survivingDemand=true;p.coefficients[0]=1;p.coefficients[14]=1;p.communicationCoefficients={1,1,1};p.provenance="survival-unit";
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));EXPECT_EQ(cost->cost.instructionPathAndWork,96u);
  p.survivingDemand=false;p.communicationCoefficients.fill(0);target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);auto old=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(old));EXPECT_EQ(old->cost.instructionPathAndWork,128u);
  EXPECT_FALSE(analyzeLayoutSurvivingDemand(g,a,{},1).proven);
 },[&](ModuleOp module){Operation *store=nullptr,*exp=nullptr;module.walk([&](Operation *o){if(o->getName().getStringRef()=="frisk.tile_store")store=o;if(o->getName().getStringRef()=="math.exp")exp=o;});ASSERT_TRUE(store && exp);store->setOperand(0,exp->getResult(0));},FRISK_DEMAND_ORDER_EAGER_FIXTURE);
}
TEST_F(LayoutDemandTest, ExplicitNetworkObjectiveChangesActualBoundaryAndTwentyFeatureCost) {
 Operation *added=nullptr;
 withGraph([&](const auto &g,const auto &a){
  auto automatic=analyzeLayoutDemand(g,a,{},65536,true,shuffle_network::Objective::MinShuffle);
  auto minWork=analyzeLayoutDemand(g,a,{},65536,true,shuffle_network::Objective::MinWork);
  ASSERT_TRUE(automatic.exact)<<automatic.reason;ASSERT_TRUE(minWork.exact)<<minWork.reason;
  unsigned matched=0;
  for(const auto &[id,b]:automatic.boundaries) {
   auto use=g.getConstraint(id).use;if(!use || use->getOwner()!=added)continue;++matched;
   ASSERT_TRUE(b.shuffleNetwork);ASSERT_TRUE(minWork.boundaries.at(id).shuffleNetwork);
   EXPECT_EQ(b.shuffleNetwork->kind,"bit-permutation-fused");EXPECT_EQ(minWork.boundaries.at(id).shuffleNetwork->kind,"bit-permutation-ship");
   EXPECT_EQ(b.shuffleWarpIssues,24u);EXPECT_EQ(b.addressAluWarpIssues,144u);
   EXPECT_EQ(minWork.boundaries.at(id).shuffleWarpIssues,32u);EXPECT_EQ(minWork.boundaries.at(id).addressAluWarpIssues,112u);
  }
  EXPECT_EQ(matched,1u);auto comparison=compareLayoutNetworkObjectives(g,a);
  EXPECT_TRUE(comparison.proven)<<comparison.reason;EXPECT_TRUE(comparison.changed);EXPECT_GE(comparison.comparedShuffleBoundaries,1u);
  EXPECT_FALSE(analyzeLayoutDemand(g,a,{},65536,false,shuffle_network::Objective::MinWork).exact);
  SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.survivingDemand=true;p.networkProxy=true;
  p.coefficients[14]=1;p.communicationCoefficients={1,7,2};p.provenance="explicit-network-proxy-unit";
  ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto x=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(x));
  p.networkObjective=shuffle_network::Objective::MinWork;target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto y=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(y));
  EXPECT_EQ(x->cost.instructionPathAndWork,y->cost.instructionPathAndWork+8u);
 },[&](ModuleOp module){
  Operation *store=nullptr,*load=nullptr;module.walk([&](Operation *o){if(o->getName().getStringRef()=="frisk.tile_store")store=o;if(!load && o->getName().getStringRef()=="frisk.tile_load")load=o;});ASSERT_TRUE(store && load);
  auto type=cast<RankedTensorType>(load->getResult(0).getType());auto e=cast<DistributedEncodingAttr>(type.getEncoding());ASSERT_EQ(e.getTopology()[0],8);
  auto map=cast<BitLinearLayoutMapAttr>(e.getMap());auto mt=cast<RankedTensorType>(map.getMatrix().getType());SmallVector<APInt> bits(map.getMatrix().getValues<APInt>());
  for(int64_t row=0;row<mt.getDimSize(0);++row){std::swap(bits[row*mt.getDimSize(1)],bits[row*mt.getDimSize(1)+3]);std::swap(bits[row*mt.getDimSize(1)+1],bits[row*mt.getDimSize(1)+4]);}
  auto changed=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),map.getInputBitWidths(),map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(mt,bits));
  auto output=RankedTensorType::get(type.getShape(),type.getElementType(),DistributedEncodingAttr::get(&context,changed,e.getTopology(),e.getReplication()));OpBuilder b(store);OperationState state(store->getLoc(),"frisk.convert_layout");state.addOperands(load->getResult(0));state.addTypes(output);added=b.create(state);b.create<math::ExpOp>(store->getLoc(),added->getResult(0));
 },FRISK_DEMAND_ORDER_EAGER_FIXTURE);
}
TEST(SM90NetworkProxyProfileTest, ExplicitObjectiveAndNonRefitContractCannotAliasHistoricalProfiles) {
 SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.survivingDemand=true;p.networkProxy=true;p.coefficients[14]=1;p.communicationCoefficients={1,2,3};p.provenance="migrated-proxy-unit";
 auto text=serializeSM90CostProfile(p);EXPECT_NE(text.find("sm90-layout-network-surviving-proxy-v1"),std::string::npos);
 auto parsed=parseSM90CostProfile(text);ASSERT_TRUE(bool(parsed));EXPECT_TRUE(parsed->networkProxy);EXPECT_EQ(serializeSM90CostProfile(*parsed),text);
 EXPECT_NE(text.find("migrated-engineering-proxy-not-refit"),std::string::npos);
 auto autoID=getSM90CalibratedCostModelID(p);EXPECT_LE(autoID.size(),64u);EXPECT_NE(autoID.find("sm90-layout-network-v1"),std::string::npos);
 auto work=p;work.networkObjective=shuffle_network::Objective::MinWork;EXPECT_NE(getSM90CostProfileSHA256(p),getSM90CostProfileSHA256(work));EXPECT_NE(autoID,getSM90CalibratedCostModelID(work));
 auto historical=p;historical.networkProxy=false;EXPECT_NE(autoID,getSM90CalibratedCostModelID(historical));EXPECT_TRUE(succeeded(verifySM90CostProfile(historical)));
 historical.networkObjective=shuffle_network::Objective::MinWork;EXPECT_TRUE(failed(verifySM90CostProfile(historical)));
 auto tamper=[&](StringRef key,StringRef value){auto j=llvm::json::parse(text);ASSERT_TRUE(bool(j));auto *o=j->getAsObject();(*o)[key]=value.str();auto result=parseSM90CostProfile(llvm::formatv("{0}",*j).str());EXPECT_FALSE(bool(result));if(!result)llvm::consumeError(result.takeError());};
 tamper("network_objective","unknown");tamper("weight_provenance","calibrated-from-gpu");
 auto j=llvm::json::parse(text);ASSERT_TRUE(bool(j));j->getAsObject()->erase("network_objective");auto result=parseSM90CostProfile(llvm::formatv("{0}",*j).str());EXPECT_FALSE(bool(result));if(!result)llvm::consumeError(result.takeError());
}
TEST_F(LayoutDemandTest, SurvivingLocalClosureExcludesFutureCommunication) {
 withGraph([&](const auto &g,const auto &a){LayoutVarID root=0;for(auto &v:g.getVariables())if(v.value && !v.use && v.kind==LayoutKind::Distributed && v.value.getDefiningOp()->getName().getStringRef()=="arith.mulf"){root=v.id;break;}auto x=analyzeLayoutSurvivingDemand(g,a,{root});ASSERT_TRUE(x.proven)<<x.reason;EXPECT_FALSE(x.surviving.demandLowering);EXPECT_EQ(x.surviving.loadVectors,16u);EXPECT_TRUE(x.retainedCommunication.empty());EXPECT_EQ(x.surviving.scratchMaxBytes,0u);});
}
TEST_F(LayoutDemandTest, SurvivingDeadSharedScaffoldStillHasPositiveMarginalPrice) {
 withGraph([&](const auto &g,const auto &a){
  auto x=analyzeLayoutSurvivingDemand(g,a);ASSERT_TRUE(x.proven)<<x.reason;ASSERT_EQ(x.retainedCommunication.size(),1u);auto id=x.retainedCommunication.front();ASSERT_TRUE(x.surviving.boundaries.count(id));auto &b=x.surviving.boundaries.at(id);
  EXPECT_EQ(b.sharedReadWarpIssues,32u);EXPECT_EQ(b.sharedWriteWarpIssues,32u);EXPECT_EQ(b.addressAluWarpIssues,3872u);EXPECT_EQ(b.bytes,8192u);EXPECT_EQ(b.barriers,2u);EXPECT_GT(x.removedQueryVectors,0u);EXPECT_EQ(x.surviving.scratchMaxBytes,4096u);
  SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.survivingDemand=true;p.coefficients[14]=1;p.communicationCoefficients={2,3,4};p.provenance="dead-communication-unit";auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;auto cost=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(cost));EXPECT_EQ(cost->cost.instructionPathAndWork,15618u);
 },[&](ModuleOp module){
  Operation *store=nullptr,*load=nullptr;module.walk([&](Operation *o){if(o->getName().getStringRef()=="frisk.tile_store")store=o;if(!load && o->getName().getStringRef()=="frisk.tile_load")load=o;});ASSERT_TRUE(store && load);
  auto type=cast<RankedTensorType>(load->getResult(0).getType());auto e=cast<DistributedEncodingAttr>(type.getEncoding());auto map=cast<BitLinearLayoutMapAttr>(e.getMap());auto mt=cast<RankedTensorType>(map.getMatrix().getType());SmallVector<APInt> bits(map.getMatrix().getValues<APInt>());for(int64_t row=0;row<mt.getDimSize(0);++row)std::swap(bits[row*mt.getDimSize(1)],bits[row*mt.getDimSize(1)+8]);
  auto changed=BitLinearLayoutMapAttr::get(&context,map.getInputNames(),map.getInputBitWidths(),map.getOutputNames(),map.getOutputBitWidths(),DenseIntElementsAttr::get(mt,bits));auto output=RankedTensorType::get(type.getShape(),type.getElementType(),DistributedEncodingAttr::get(&context,changed,e.getTopology(),e.getReplication()));OpBuilder builder(store);OperationState state(store->getLoc(),"frisk.convert_layout");state.addOperands(load->getResult(0));state.addTypes(output);auto *conversion=builder.create(state);builder.create<math::ExpOp>(store->getLoc(),conversion->getResult(0));
 },FRISK_DEMAND_ORDER_EAGER_FIXTURE);
}
TEST_F(LayoutDemandTest, SurvivingCommunicationOnlyProfileCannotPriceUnsupportedCarrierGraphs) {
 auto profile=[] {SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.survivingDemand=true;p.coefficients[14]=1;p.communicationCoefficients={1,1,1};p.provenance="surviving-carrier-boundary-unit";return p;};
 withGraph([&](const auto &g,const auto &a){
  auto p=profile();ASSERT_TRUE(succeeded(verifySM90CostProfile(p)));auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  EXPECT_TRUE(succeeded(scoreExperimentAssignment(g,a,*target,policy)));
  policy.conversion=ConversionPolicy::Full;EXPECT_TRUE(failed(scoreExperimentAssignment(g,a,*target,policy)));
  p.coefficients[6]=1;target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);
  auto fallback=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(fallback));EXPECT_GT(fallback->cost.instructionPathAndWork,0u);
 });
 withGraph([&](const auto &g,const auto &a){
  auto p=profile();auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;
  EXPECT_TRUE(failed(scoreExperimentAssignment(g,a,*target,policy)));
  p.coefficients[6]=1;target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);auto fallback=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(fallback));EXPECT_GT(fallback->cost.instructionPathAndWork,0u);
 },[&](ModuleOp module){
  Operation *parallel=nullptr;module.walk([&](Operation *op){if(!parallel && isa<ParallelOp>(op))parallel=op;});ASSERT_TRUE(parallel);OpBuilder b(parallel);auto zero=b.create<arith::ConstantIndexOp>(parallel->getLoc(),0),one=b.create<arith::ConstantIndexOp>(parallel->getLoc(),1);auto loop=b.create<scf::ForOp>(parallel->getLoc(),zero,one,one);parallel->moveBefore(loop.getBody()->getTerminator());
 });
 withGraph([&](const auto &g,const auto &a){
  bool hasMMA=false;for(const auto &v:g.getVariables())hasMMA|=v.anchor && v.anchor->getName().getStringRef()=="frisk.mma";ASSERT_TRUE(hasMMA);
  auto p=profile();auto target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);ExperimentPolicy policy;policy.conversion=ConversionPolicy::Independent;EXPECT_TRUE(failed(scoreExperimentAssignment(g,a,*target,policy)));
  p.coefficients[4]=1;target=createSM90LayoutTarget(LayoutBoundaryFrequencyMode::Legacy,p);auto fallback=scoreExperimentAssignment(g,a,*target,policy);ASSERT_TRUE(succeeded(fallback));EXPECT_GT(fallback->cost.instructionPathAndWork,0u);
 },[](ModuleOp){},FRISK_DEMAND_MMA_FALLBACK_FIXTURE);
}
TEST(SM90SurvivingProfileTest, TwentyKeysRequirePositiveCommunicationAndFixedZeroBytes) {
 EXPECT_EQ(getSM90SurvivingCostFeatureNames().size(),20u);SM90CostProfile p;p.executionDemand=true;p.executionOrder=true;p.survivingDemand=true;p.coefficients[14]=2;p.communicationCoefficients={3,4,1};p.provenance="surviving-profile-unit";
 auto text=serializeSM90CostProfile(p);EXPECT_NE(text.find("sm90-layout-surviving-calibrated-v1"),std::string::npos);EXPECT_EQ(text.find("global_load_issue_groups"),std::string::npos);auto parsed=parseSM90CostProfile(text);ASSERT_TRUE(bool(parsed));EXPECT_TRUE(parsed->survivingDemand);EXPECT_EQ(serializeSM90CostProfile(*parsed),text);EXPECT_NE(getSM90CalibratedCostModelID(*parsed).find("v6d"),std::string::npos);EXPECT_LE(getSM90CalibratedCostModelID(*parsed).size(),64u);
 for(unsigned i=0;i<3;++i){auto bad=p;bad.communicationCoefficients[i]=0;EXPECT_TRUE(failed(verifySM90CostProfile(bad)));}auto bad=p;bad.coefficients[14]=0;EXPECT_TRUE(failed(verifySM90CostProfile(bad)));bad=p;bad.coefficients[12]=1;EXPECT_TRUE(failed(verifySM90CostProfile(bad)));bad=p;bad.loadIssueGroupCoefficient=1;EXPECT_TRUE(failed(verifySM90CostProfile(bad)));bad=p;bad.coefficients[2]=1;EXPECT_TRUE(failed(verifySM90CostProfile(bad)));
}
TEST(SM90OrderedDemandProfileTest, Existing15And17NamesRemainStableAndEighteenthIsExplicit) {
 EXPECT_EQ(getSM90CostFeatureNames().size(),15u);EXPECT_EQ(getSM90DemandCostFeatureNames().size(),17u);EXPECT_EQ(getSM90OrderedDemandCostFeatureNames().size(),18u);
 EXPECT_EQ(getSM90OrderedDemandCostFeatureNames().back(),"global_load_issue_groups");
 SM90CostProfile profile;profile.executionDemand=true;profile.executionOrder=true;profile.loadIssueGroupCoefficient=1;profile.provenance="ordered-only-profile-unit";
 EXPECT_TRUE(succeeded(verifySM90CostProfile(profile)));auto parsed=parseSM90CostProfile(serializeSM90CostProfile(profile));ASSERT_TRUE(bool(parsed));EXPECT_EQ(parsed->loadIssueGroupCoefficient,1u);
 profile.executionOrder=false;EXPECT_TRUE(failed(verifySM90CostProfile(profile)));
}
TEST(LayoutRegisterQueryTest, AffineMapRetainsDistinctInstructionMultiplicity) {
 LayoutRegisterQuery first,second;first.terms={3,1,2,4,8,16,32,64};second=first;second.terms[0]^=1;
 for(unsigned t=0;t<128;++t){EXPECT_EQ(first.apply(t),uint64_t(t^3));EXPECT_EQ(second.apply(t),uint64_t(t^2));}EXPECT_FALSE(first==second);
}
}
