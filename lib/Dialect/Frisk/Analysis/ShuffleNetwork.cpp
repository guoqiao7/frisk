#include "Dialect/Frisk/Analysis/ShuffleNetwork.h"
#include <algorithm>
#include <set>
#include <tuple>

// Independently implemented from the linear-algebraic ideas in Linear Layouts
// section 5.4 (arXiv:2505.23819v5) and Triton's Swap/Ship decomposition:
// https://github.com/triton-lang/triton/blob/1887644bdea84fde07825813158782fa526871fd/lib/Conversion/TritonGPUToLLVM/ConvertLayoutOpToLLVM.cpp
// We additionally synthesize general invertible GF(2) holder maps by elementary
// shears. No Triton source is included or linked by this implementation.
namespace mlir::frisk::shuffle_network {
namespace {
unsigned pop(uint64_t x) { return __builtin_popcountll(x); }
unsigned log2(unsigned x) { return 31u - __builtin_clz(x); }
unsigned parity(uint64_t x) { return __builtin_parityll(x); }
uint64_t applyLane(const std::array<uint64_t,5> &columns, uint64_t x) {
  uint64_t y=0; for(unsigned b=0;b<5;++b)if(x&(1u<<b))y^=columns[b]; return y;
}
std::array<uint64_t,5> identityLane() { return {1,2,4,8,16}; }
bool laneIdentity(const Step &s) { return s.columns==identityLane(); }
uint64_t swapBits(uint64_t x,unsigned a,unsigned b) {
  if(((x>>a)^(x>>b))&1)x^=(uint64_t{1}<<a)|(uint64_t{1}<<b);return x;
}
Step permutation(const std::vector<uint64_t> &p) {Step s;s.op="reg_permute";s.permutation=p;return s;}
Step lane(unsigned r,const std::array<uint64_t,5> &columns=identityLane()) {
 Step s;s.op="lane";s.columns=columns;s.registerMasks.assign(r,0);return s;
}
void append(std::vector<Step> &steps,Step s,unsigned r) {
 if(s.op=="reg_select" && !s.threadMask) {
  if(!s.constant)return;
  std::vector<uint64_t> p(r);for(unsigned i=0;i<r;++i)p[i]=i^(1u<<s.registerBit);
  s=permutation(p);
 }
 if(s.op=="reg_permute") {bool nonidentity=false;for(unsigned i=0;i<r;++i)nonidentity|=s.permutation[i]!=i;if(!nonidentity)return;}
 if(s.op=="lane" && laneIdentity(s) &&
    std::all_of(s.offsets.begin(),s.offsets.end(),[](uint64_t x){return x==0;}) &&
    std::all_of(s.registerMasks.begin(),s.registerMasks.end(),[](uint64_t x){return x==0;}))return;
 steps.push_back(std::move(s));
}
Step regSelect(unsigned bit,unsigned mask,unsigned constant=0) {Step s;s.op="reg_select";s.registerBit=bit;s.threadMask=mask;s.constant=constant;return s;}
Step mixed(unsigned reg,unsigned laneBit) {Step s;s.op="mixed_swap";s.registerBit=reg;s.laneBit=laneBit;return s;}
Counts count(const std::vector<Step> &steps,unsigned r) {
 Counts out;std::set<std::pair<unsigned,unsigned>> predicates;
 auto predicate=[&](unsigned mask,unsigned constant) {
  if(mask && predicates.emplace(mask,constant).second)out.mapInstructions+=pop(mask)==1?2:4;
 };
 for(const Step &s:steps) {
  if(s.op=="reg_permute")continue;
  if(s.op=="reg_select") {predicate(s.threadMask,s.constant);out.selects+=r;continue;}
  if(s.op=="mixed_swap") {predicate(1u<<s.laneBit,0);out.selects+=3*(r/2);out.shfl+=r/2;continue;}
  bool immediate=laneIdentity(s) && std::all_of(s.offsets.begin(),s.offsets.end(),[&](uint64_t x){return x==s.offsets[0];});
  if(immediate) {
   for(unsigned i=0;i<r;++i)out.shfl+=bool(s.offsets[0]^s.registerMasks[i]);
  } else {
   // A nonidentity linear lane map, or a warp-varying affine offset,
   // cannot be the identity on any whole register carrier.
   out.shfl+=r;
   for(auto mask:s.registerMasks)if(mask)++out.mapInstructions;
   out.mapInstructions+=1;
   for(auto column:s.columns)if(column)out.mapInstructions+=4;
   if(s.offsets[1]^s.offsets[0])out.mapInstructions+=4;
   if(s.offsets[2]^s.offsets[0])out.mapInstructions+=4;
  }
 }
 out.totalWork=out.shfl+out.selects+out.mapInstructions;
 out.shuffleWarpIssues=4*out.shfl;out.addressAluWarpIssues=4*(out.selects+out.mapInstructions);return out;
}
std::vector<Step> prefix(unsigned r,unsigned rb,const std::array<uint64_t,4> &offsets) {
 std::vector<Step> out;
 for(unsigned b=0;b<rb;++b) {
  unsigned constant=(offsets[0]>>b)&1;
  unsigned mask=(((offsets[1]>>b)&1)^constant)*32+(((offsets[2]>>b)&1)^constant)*64;
  append(out,regSelect(b,mask,constant),r);
 }
 Step s=lane(r);for(unsigned w=0;w<4;++w)s.offsets[w]=offsets[w]>>rb;append(out,std::move(s),r);return out;
}
void appendSwap(std::vector<Step> &out,unsigned a,unsigned b,unsigned rb,unsigned r) {
 if(a<rb && b<rb) {std::vector<uint64_t> p(r);for(unsigned i=0;i<r;++i)p[i]=swapBits(i,a,b);append(out,permutation(p),r);}
 else if(a>=rb && b>=rb) {Step s=lane(r);for(auto &column:s.columns)column=swapBits(column,a-rb,b-rb);append(out,std::move(s),r);}
 else append(out,mixed(std::min(a,b),std::max(a,b)-rb),r);
}
void appendShear(std::vector<Step> &out,unsigned target,unsigned control,unsigned rb,unsigned r) {
 if(target<rb && control<rb) {std::vector<uint64_t> p(r);for(unsigned i=0;i<r;++i)p[i]=i^(((i>>control)&1)<<target);append(out,permutation(p),r);}
 else if(target<rb)append(out,regSelect(target,1u<<(control-rb)),r);
 else if(control>=rb) {Step s=lane(r);s.columns[control-rb]^=1u<<(target-rb);append(out,std::move(s),r);}
 else {Step s=lane(r);for(unsigned i=0;i<r;++i)s.registerMasks[i]=((i>>control)&1)<<(target-rb);append(out,std::move(s),r);}
}
} // namespace
uint64_t sourceHolder(const Plan &p,uint64_t index) {
 unsigned r=index%p.registers,l=(index/p.registers)%32,w=index/(32*p.registers);
 for(auto it=p.steps.rbegin();it!=p.steps.rend();++it) {const Step &s=*it;
  if(s.op=="reg_permute")r=s.permutation[r];
  else if(s.op=="reg_select") {if(parity((w*32+l)&s.threadMask)^s.constant)r^=1u<<s.registerBit;}
  else if(s.op=="mixed_swap") {if(((r>>s.registerBit)^(l>>s.laneBit))&1){r^=1u<<s.registerBit;l^=1u<<s.laneBit;}}
  else l=applyLane(s.columns,l)^s.offsets[w]^s.registerMasks[r];
 }
 return (w*32+l)*p.registers+r;
}
std::vector<Plan> candidates(unsigned nr,unsigned dr,const std::vector<uint64_t> &owners) {
 if(nr!=dr || !nr || nr>512 || (nr&(nr-1)) || owners.size()!=128*nr)return {};
 unsigned rb=log2(nr),d=rb+5;uint64_t mask=(uint64_t{1}<<d)-1,base=owners[0]&mask;
 std::vector<uint64_t> columns(rb+7);
 for(unsigned b=0;b<columns.size();++b)columns[b]=(owners[uint64_t{1}<<b]&mask)^base;
 for(unsigned i=0;i<owners.size();++i) {
  if(owners[i]>=owners.size() || owners[i]/(32*nr)!=i/(32*nr))return {};
  uint64_t value=base;for(unsigned b=0;b<columns.size();++b)if(i&(1u<<b))value^=columns[b];
  if(value!=(owners[i]&mask))return {};
 }
 std::vector<uint64_t> rows(d,0);
 for(unsigned b=0;b<d;++b)for(unsigned j=0;j<d;++j)if(columns[j]&(1u<<b))rows[b]|=uint64_t{1}<<j;
 auto eliminated=rows;unsigned rank=0;
 for(unsigned b=0;b<d;++b) {unsigned p=rank;while(p<d && !(eliminated[p]&(1u<<b)))++p;if(p==d)continue;std::swap(eliminated[p],eliminated[rank]);for(unsigned j=0;j<d;++j)if(j!=rank && (eliminated[j]&(1u<<b)))eliminated[j]^=eliminated[rank];++rank;}
 if(rank!=d)return {};
 std::array<uint64_t,4> offsets{};for(unsigned w=0;w<4;++w)offsets[w]=base^((w&1)?columns[d]:0)^((w&2)?columns[d+1]:0);
 uint64_t legacyWork=0,legacyShfl=0;
 for(unsigned reg=0;reg<nr;++reg) {
  std::set<uint64_t> support;uint64_t regs[128],lanes[128];
  for(unsigned t=0;t<128;++t){regs[t]=owners[t*nr+reg]%nr;lanes[t]=(owners[t*nr+reg]/nr)%32;support.insert(regs[t]);}
  auto mapWork=[](const uint64_t *a){uint64_t n=1;for(unsigned b=0;b<7;++b)if(a[1u<<b]^a[0])n+=4;return n;};
  uint64_t n=support.size();legacyShfl+=n;legacyWork+=mapWork(regs)+mapWork(lanes)+n+1+2*(n-1);
 }
 std::vector<Plan> candidates;
 auto add=[&](std::string kind,std::vector<Step> steps) {
  Plan p;p.registers=nr;p.kind=std::move(kind);p.steps=std::move(steps);p.counts=count(p.steps,nr);p.legacyWork=legacyWork;p.legacyShfl=legacyShfl;p.verifiedHolders=owners.size();p.affineBase=base;p.affineColumns=columns;
  // A lower total-work count cannot compensate for extra communication.
  // Keep both guarantees before comparing the eligible network alternatives.
  if(p.counts.totalWork>legacyWork || p.counts.shfl>legacyShfl)return;
  for(unsigned i=0;i<owners.size();++i)if(sourceHolder(p,i)!=owners[i])return;
  candidates.push_back(std::move(p));
 };
 bool monomial=std::all_of(rows.begin(),rows.end(),[](uint64_t row){return pop(row)==1;});
 if(monomial) {
  std::vector<unsigned> regRows,laneRows;
  for(unsigned i=0;i<rb;++i)if(rows[i]>>rb)regRows.push_back(i);
  for(unsigned i=rb;i<d;++i)if(rows[i]&((1u<<rb)-1))laneRows.push_back(i);
  if(regRows.size()!=laneRows.size())return {};
  auto block=rows;
  for(unsigned i=0;i<regRows.size();++i)std::swap(block[regRows[i]],block[laneRows[i]]);
  std::vector<uint64_t> regPermutation(nr);
  for(unsigned reg=0;reg<nr;++reg)for(unsigned i=0;i<rb;++i)if(parity(block[i]&reg))regPermutation[reg]|=1u<<i;
  std::array<uint64_t,5> laneColumns{};
  for(unsigned c=0;c<5;++c)for(unsigned row=0;row<5;++row)if(block[rb+row]&(1u<<(rb+c)))laneColumns[c]|=1u<<row;
  auto ship=prefix(nr,rb,offsets);
  for(unsigned i=0;i<regRows.size();++i)append(ship,mixed(regRows[i],laneRows[i]-rb),nr);
  append(ship,permutation(regPermutation),nr);append(ship,lane(nr,laneColumns),nr);add("bit-permutation-ship",std::move(ship));
  auto fused=prefix(nr,rb,offsets);
  for(unsigned i=0;i<regRows.size();++i)append(fused,regSelect(regRows[i],1u<<(laneRows[i]-rb)),nr);
  Step middle=lane(nr,laneColumns);
  for(unsigned reg=0;reg<nr;++reg)for(unsigned i=0;i<regRows.size();++i)if(reg&(1u<<regRows[i]))middle.registerMasks[reg]^=1u<<(laneRows[i]-rb);
  append(fused,std::move(middle),nr);
  for(unsigned i=0;i<regRows.size();++i) {
   uint64_t row=block[laneRows[i]]>>rb;
   append(fused,regSelect(regRows[i],unsigned(row)),nr);
  }
  append(fused,permutation(regPermutation),nr);add("bit-permutation-fused",std::move(fused));
 } else {
  auto steps=prefix(nr,rb,offsets);auto matrix=rows;
  for(unsigned i=0;i<d;++i) {
   unsigned p=i;while(p<d && !(matrix[p]&(1u<<i)))++p;if(p==d)return {};
   if(p!=i){std::swap(matrix[p],matrix[i]);appendSwap(steps,i,p,rb,nr);}
   for(unsigned j=0;j<d;++j)if(j!=i && (matrix[j]&(1u<<i))){matrix[j]^=matrix[i];appendShear(steps,j,i,rb,nr);}
  }
  add("linear-shear",std::move(steps));
 }
 return candidates;
}
std::optional<Plan> plan(unsigned nr,unsigned dr,const std::vector<uint64_t> &owners,Objective objective) {
 if(objective!=Objective::MinShuffle && objective!=Objective::MinWork)return std::nullopt;
 auto options=candidates(nr,dr,owners);if(options.empty())return std::nullopt;
 auto chosen=std::min_element(options.begin(),options.end(),[&](const Plan &a,const Plan &b){
  if(objective==Objective::MinWork)return std::tie(a.counts.totalWork,a.counts.shfl,a.kind)<std::tie(b.counts.totalWork,b.counts.shfl,b.kind);
  return std::tie(a.counts.shfl,a.counts.totalWork,a.kind)<std::tie(b.counts.shfl,b.counts.totalWork,b.kind);
 });return std::move(*chosen);
}
bool sameExecutionPlan(const Plan &a,const Plan &b) {
 if(a.registers!=b.registers || a.affineBase!=b.affineBase || a.affineColumns!=b.affineColumns || a.steps.size()!=b.steps.size())return false;
 const auto &x=a.counts,&y=b.counts;
 if(std::tie(x.shfl,x.selects,x.mapInstructions,x.totalWork,x.shuffleWarpIssues,x.addressAluWarpIssues)!=std::tie(y.shfl,y.selects,y.mapInstructions,y.totalWork,y.shuffleWarpIssues,y.addressAluWarpIssues))return false;
 for(unsigned i=0;i<a.steps.size();++i) {
  const auto &x=a.steps[i],&y=b.steps[i];
  if(std::tie(x.op,x.permutation,x.registerBit,x.threadMask,x.constant,x.laneBit,x.columns,x.offsets,x.registerMasks)!=std::tie(y.op,y.permutation,y.registerBit,y.threadMask,y.constant,y.laneBit,y.columns,y.offsets,y.registerMasks))return false;
 }return true;
}
StructuralResources structuralResources(const Plan &p) {
 StructuralResources result;result.logicalCarrierWords=p.registers;std::set<std::pair<unsigned,unsigned>> predicates;
 for(const auto &s:p.steps) {
  if(s.op=="reg_permute")continue;
  if(s.op=="reg_select" || s.op=="mixed_swap") {
   unsigned mask=s.op=="reg_select"?s.threadMask:1u<<s.laneBit,constant=s.op=="reg_select"?s.constant:0;
   if(predicates.emplace(mask,constant).second){++result.b32TemporaryDefinitions;++result.predicateDefinitions;}
   result.b32TemporaryDefinitions+=s.op=="reg_select"?p.registers:2*p.registers;
   if(s.op=="mixed_swap")++result.communicationStages;
  } else {
   bool immediate=laneIdentity(s) && std::all_of(s.offsets.begin(),s.offsets.end(),[&](uint64_t x){return x==s.offsets[0];});
   unsigned active=0;for(auto mask:s.registerMasks)active+=!immediate || bool(s.offsets[0]^mask);
   if(active){++result.communicationStages;result.b32TemporaryDefinitions+=active;}
   if(!immediate) {
    ++result.b32TemporaryDefinitions;
    for(auto column:s.columns)result.b32TemporaryDefinitions+=bool(column);
    result.b32TemporaryDefinitions+=bool(s.offsets[1]^s.offsets[0])+bool(s.offsets[2]^s.offsets[0]);
    for(auto mask:s.registerMasks)result.b32TemporaryDefinitions+=bool(mask);
   }
  }
 }return result;
}
std::vector<Plan> frontier(unsigned nr,unsigned dr,const std::vector<uint64_t> &owners) {
 auto all=candidates(nr,dr,owners);std::vector<Plan> result;
 for(unsigned i=0;i<all.size();++i) {
  auto resource=structuralResources(all[i]);bool dominated=false;
  for(unsigned j=0;j<all.size();++j)if(i!=j) {
   auto other=structuralResources(all[j]);
   bool le=all[j].counts.shfl<=all[i].counts.shfl && all[j].counts.totalWork<=all[i].counts.totalWork && other.b32TemporaryDefinitions<=resource.b32TemporaryDefinitions;
   bool lt=all[j].counts.shfl<all[i].counts.shfl || all[j].counts.totalWork<all[i].counts.totalWork || other.b32TemporaryDefinitions<resource.b32TemporaryDefinitions;
   if(le && lt){dominated=true;break;}
  }
  if(!dominated)result.push_back(all[i]);
 }return result;
}

} // namespace mlir::frisk::shuffle_network
