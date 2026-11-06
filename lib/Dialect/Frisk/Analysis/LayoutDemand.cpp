#include "Dialect/Frisk/Analysis/LayoutDemand.h"
#include "Dialect/Frisk/Analysis/LayoutDemandFootprint.h"
#include "Dialect/Frisk/Analysis/ShuffleNetwork.h"
#include "Dialect/Frisk/Analysis/ShuffleNetworkJSON.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutProfile.h"
#include "mlir/IR/Builders.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include <algorithm>
#include <tuple>
#include <set>
#include <optional>
#include <limits>
using namespace mlir;
using namespace mlir::frisk;
uint64_t LayoutRegisterQuery::apply(unsigned t) const {
  uint64_t r=terms[0];for(unsigned b=0;b<7;++b)if(t&(1u<<b))r^=terms[b+1];return r;
}
std::optional<mlir::frisk::detail::DemandAccessFootprint> mlir::frisk::detail::DemandAccessFootprintCache::evaluate(const std::array<uint64_t,128> &offsets,
                                      const std::array<bool,128> &active,
                                      bool global,uint64_t alignment,size_t capacity) {
    LayoutProfile::Scope profile(LayoutProfileKind::DemandFootprintCache);
    if(!alignment || alignment>32)return std::nullopt;
    // Validate original addresses before normalization. Production callers also
    // retain the stricter whole-footprint bounded-u32 proof in addressTable().
    uint64_t tail=(31/alignment)*alignment+3;
    for(unsigned t=0;t<128;++t)
      if(active[t] && offsets[t]>UINT64_MAX-tail)return std::nullopt;
    bool memo=capacity!=0;
    DemandAccessFootprintKey key{global,alignment};
    if(memo) {
      // Each warp is scored separately. A multiple-of-32 byte translation keeps
      // its global sector cardinality for every root phase. In Shared memory a
      // multiple-of-4 byte translation only rotates the 32 bank labels, keeping
      // broadcasts and maximum bank degree. Integer subtraction (not XOR) is
      // essential: the phase additions may carry into the next sector/word.
      const uint64_t unit=global?32:4;
      for(unsigned warp=0;warp<4;++warp) {
        uint64_t base=UINT64_MAX;
        for(unsigned lane=0;lane<32;++lane) {
          unsigned t=warp*32+lane;
          if(active[t])base=std::min(base,offsets[t]);
        }
        if(base==UINT64_MAX)continue;
        base=(base/unit)*unit;
        for(unsigned lane=0;lane<32;++lane) {
          unsigned t=warp*32+lane;if(!active[t])continue;
          key.active[t/64]|=uint64_t{1}<<(t%64);
          key.offsets[t]=offsets[t]-base;
        }
      }
      auto found=entries.find(key);
      LayoutProfile::cacheLookup(LayoutProfileKind::DemandFootprintCache,
                                found!=entries.end());
      if(found!=entries.end())return found->second;
    }
    DemandAccessFootprint cost;
    for(unsigned warp=0;warp<4;++warp)
      if(llvm::any_of(ArrayRef<bool>(active).slice(warp*32,32),[](bool a){return a;}))
        ++cost.warpIssues;
    for(unsigned phase=0;phase<32;phase+=alignment) {
      uint64_t total=0;
      for(unsigned warp=0;warp<4;++warp) {
        // A 4-byte f32 access touches at most two sectors/words per lane.
        // Sorting this fixed stack buffer avoids one heap allocation per
        // distinct address and does not compute an unused memory-space metric.
        std::array<uint64_t,64> touched;unsigned count=0;
        uint64_t unit=global?32:4;
        for(unsigned lane=0;lane<32;++lane) {
          unsigned t=warp*32+lane;if(!active[t])continue;
          uint64_t first=(offsets[t]+phase)/unit,last=(offsets[t]+3+phase)/unit;
          touched[count++]=first;if(last!=first)touched[count++]=last;
        }
        std::sort(touched.begin(),touched.begin()+count);
        auto end=std::unique(touched.begin(),touched.begin()+count);
        if(global)total+=std::distance(touched.begin(),end);
        else {
          std::array<unsigned,32> banks{};unsigned degree=1;
          for(auto it=touched.begin();it!=end;++it)
            degree=std::max(degree,++banks[*it%32]);
          total+=degree-1;
        }
      }
      if(global)cost.sectors=std::max(cost.sectors,total);
      else cost.bankExcess=std::max(cost.bankExcess,total);
    }
    // Builder-local storage is deliberately small and does not survive an
    // assignment. Disabling the graph cost cache disables this derived memo too.
    if(memo && entries.size()<std::min<size_t>(256,capacity)) {
      entries.emplace(std::move(key),cost);
      LayoutProfile::cacheInsertion(LayoutProfileKind::DemandFootprintCache);
      LayoutProfile::cacheSize(LayoutProfileKind::DemandFootprintCache,entries.size());
    }
    return cost;
  }

namespace {
struct Basis {
  std::array<uint64_t,16> values{},owners{};
  void insert(uint64_t v,uint64_t owner) {
    for(unsigned b=16;b-->0;)if(v&(uint64_t{1}<<b)) {
      if(values[b]){v^=values[b];owner^=owners[b];}
      else{values[b]=v;owners[b]=owner;return;}
    }
  }
  std::optional<uint64_t> solve(uint64_t v) const {
    uint64_t owner=0;for(unsigned b=16;b-->0;)if(v&(uint64_t{1}<<b)) {
      if(!values[b])return std::nullopt;v^=values[b];owner^=owners[b];
    }return owner;
  }
};
struct Encoding {
  DistributedEncodingAttr attr;LinearExecutionLayout linear;Basis registers,warp,all;
  uint64_t logical(uint64_t r,unsigned t) const {
    return linear.apply(APInt(std::max(1u,linear.hardwareBits),uint64_t(t)*linear.topology[0]+r)).getZExtValue();
  }
  uint64_t slot(uint64_t bits) const {
    uint64_t index=0;
    auto coordinates=linear.logicalCoordinate(APInt(std::max(1u,linear.logicalBits),bits));
    for(unsigned i=0;i<coordinates.size();++i)index=(index<<linear.logicalWidths[i])+coordinates[i];
    return index;
  }
};
struct DemandBuilder {
  const LayoutConstraintGraph &g;const CandidateAssignment &assignment;LayoutDemandLedger result;
  std::map<LayoutVarID,Encoding> enc;
  std::map<OpOperand*,LayoutVarID> uses;
  std::map<LayoutVarID,const LayoutConstraint*> conversions,readAccess;
  std::map<const LayoutConstraint*,std::map<uint64_t,std::pair<uint64_t,uint64_t>>> addresses;
  std::map<std::pair<LayoutVarID,LayoutRegisterQuery>,uint64_t> cache;
  mlir::frisk::detail::DemandAccessFootprintCache accessFootprints;
  std::set<std::pair<LayoutVarID,LayoutRegisterQuery>> pending;
  std::set<LayoutVarID> materialized;
  std::map<LayoutVarID,uint64_t> boundaryDepth;
  shuffle_network::Objective networkObjective;
  uint64_t budget;bool details=true,orderedEmission=true,failed=false,globalLoadGroupOpen=false;Block *producerBlock=nullptr;LayoutVarID storeValue=0;const LayoutConstraint *storeAccess=nullptr;
  DemandBuilder(const LayoutConstraintGraph &g,const CandidateAssignment &a,uint64_t b,bool d=true,bool ordered=true,shuffle_network::Objective objective=shuffle_network::Objective::MinShuffle):g(g),assignment(a),networkObjective(objective),budget(std::min<uint64_t>(b,65536)),details(d),orderedEmission(ordered){}
  void reject(StringRef reason){if(!failed)result.reason=reason.str();failed=true;}
  void finishGlobalLoadGroup(){globalLoadGroupOpen=false;}
  void issueGlobalLoad(){if(!globalLoadGroupOpen)++result.globalLoadIssueGroups;globalLoadGroupOpen=true;}
  LayoutRegisterQuery uniform(uint64_t r){LayoutRegisterQuery q;q.terms[0]=r;return q;}
  bool prepare() {
    if(!g.isFinalized()){reject("demand graph is not finalized");return false;}
    for(const auto &v:g.getVariables()) {
      if(!assignment.values.count(v.id)){reject("incomplete demand assignment");return false;}
      if(v.kind!=LayoutKind::Distributed)continue;
      auto type=dyn_cast<RankedTensorType>(v.shapedType);auto attr=dyn_cast_or_null<DistributedEncodingAttr>(assignment.values.lookup(v.id));
      Encoding e;e.attr=attr;
      if(!type || !type.getElementType().isF32() || !attr ||
         analyzeLinearExecutionLayout(attr,type,e.linear).status!=ProofStatus::Proven ||
         e.linear.topology[1]!=32 || e.linear.topology[2]!=4 || e.linear.topology[3]!=1 || e.linear.topology[4]!=1 ||
         e.linear.logicalBits>16 || e.linear.hardwareBits>63 ||
         proveExecutionOwnership(attr,type,"first_owner").status!=ProofStatus::Proven) {
        reject("demand scope requires proven dyadic f32 128-thread layouts with <=65536 logical elements");return false;
      }
      for(unsigned bit=0;bit<e.linear.hardwareBits;++bit) {
        uint64_t column=e.linear.apply(APInt(std::max(1u,e.linear.hardwareBits),uint64_t{1}<<bit)).getZExtValue();
        e.all.insert(column,uint64_t{1}<<bit);
        if(bit<e.linear.hardwareWidths[0])e.registers.insert(column,uint64_t{1}<<bit);
        if(bit<e.linear.hardwareOffsets[2])e.warp.insert(column,uint64_t{1}<<bit);
      }
      result.demandLowering |= e.linear.topology[0]>256;
      if(!v.use)result.declaredRegisterVectors+=e.linear.topology[0];
      enc.emplace(v.id,std::move(e));if(v.use)uses[v.use]=v.id;
      if(!v.use && v.value) {
        auto *op=v.value.getDefiningOp();if(!op || op->getNumResults()!=1 || op->getNumRegions()!=0) {reject("demand needs pure single-result SSA producers");return false;}
        StringRef name=op->getName().getStringRef();
        if(name!="frisk.tile_load" && name!="frisk.convert_layout" && name!="arith.constant" && name!="arith.mulf" && name!="arith.subf" && name!="arith.divf" && name!="math.exp") {reject("operation outside exact demand backend scope");return false;}
        if(name=="arith.constant") {
          auto value=op->getAttrOfType<DenseElementsAttr>("value");
          if(!value || !value.isSplat()){reject("demand constant requires a dense f32 scalar splat");return false;}
        }
      }
    }
    if(enc.empty()){reject("no distributed demand values");return false;}
    unsigned stores=0;
    for(const auto &c:g.getConstraints()) {
      if(c.kind==ConstraintKind::Convertible) {
        if(c.vars.size()!=2 || !enc.count(c.vars[0]) || !enc.count(c.vars[1])){reject("invalid demand conversion");return false;}
        LayoutVarID endpoint=c.vars[1];
        if(!c.existingConversion && !g.getVariable(endpoint).use) {
          endpoint=g.getVariables().size()+c.id;enc.emplace(endpoint,enc.at(c.vars[1]));
          if(enc.at(c.vars[0]).attr!=enc.at(endpoint).attr)result.declaredRegisterVectors+=enc.at(endpoint).linear.topology[0];
        } else if(orderedEmission && !c.existingConversion && g.getVariable(endpoint).use && enc.at(c.vars[0]).attr!=enc.at(endpoint).attr) {
          // A nonidentity store/use repair becomes an actual SSA tensor too.
          result.declaredRegisterVectors+=enc.at(endpoint).linear.topology[0];
        }
        if(!conversions.emplace(endpoint,&c).second){reject("multiple demand conversion definitions");return false;}
        if(!c.existingConversion && c.use)uses[c.use]=endpoint;
      }
      if(c.kind==ConstraintKind::StorageAccess) {
        if(c.vars.size()!=2){reject("invalid demand access");return false;}
        LayoutVarID distributed=enc.count(c.vars[0])?c.vars[0]:c.vars[1];
        if(!enc.count(distributed)){reject("no demand execution endpoint");return false;}
        if(c.access==AccessKind::Read)readAccess[distributed]=&c;
        else {++stores;storeValue=distributed;storeAccess=&c;}
      }
      if(c.instruction || c.reduction || c.broadcast || c.kind==ConstraintKind::TransformLayout){reject("instruction, reduction or transform outside exact demand scope");return false;}
    }
    if(stores!=1){reject("demand requires one final store");return false;}
    // A single straight-line producer DAG is required; loops/regions would
    // need separate demand roots/frequency transfer semantics.
    Block *block=storeAccess->use?storeAccess->use->getOwner()->getBlock():nullptr;
    if(!block && g.getVariable(storeValue).use)block=g.getVariable(storeValue).use->getOwner()->getBlock();
    producerBlock=block;Operation *container=block?block->getParentOp():nullptr;
    bool bodyContainer=container && (container->getName().getStringRef()=="func.func" || container->getName().getStringRef()=="frisk.kernel");
    if(auto parallel=dyn_cast_or_null<ParallelOp>(container))
      bodyContainer=parallel.getThreads()==128 && llvm::all_of(parallel.getRanges(),[](int64_t n){return n==1;});
    if(!bodyContainer || container->getNumRegions()!=1 || !container->getRegion(0).hasOneBlock()) {
      reject("demand requires one function/kernel body or one single-CTA128 parallel body");return false;
    }
    unsigned parallelAncestors=0;
    for(Operation *parent=container;parent;parent=parent->getParentOp()) {
      StringRef name=parent->getName().getStringRef();
      if(name!="builtin.module" && name!="func.func" && name!="frisk.kernel" && name!="frisk.parallel") {
        reject("unsupported enclosing execution container outside exact demand scope");return false;
      }
      if(parent->getName().getDialectNamespace()=="scf"){reject("control-flow frequency outside exact demand scope");return false;}
      if(isa<ParallelOp>(parent) && ++parallelAncestors>1){reject("nested parallel execution outside exact demand scope");return false;}
    }
    for(const auto &[id,e]:enc) {
      if(id>=g.getVariables().size())continue;auto &v=g.getVariable(id);if(v.value && !v.use && v.value.getDefiningOp()->getBlock()!=block){reject("demand producer crosses a control-flow block");return false;}
      auto *store=g.getVariable(storeValue).use?g.getVariable(storeValue).use->getOwner():nullptr;
      if(store && v.value && !v.use && !v.value.getDefiningOp()->isBeforeInBlock(store)){reject("demand requires final store after all producers");return false;}
    }
    return true;
  }
  bool sameScope(const Encoding &s,const Encoding &d,bool warp) {
    const Basis &basis=warp?s.warp:s.registers;unsigned free=warp?s.linear.hardwareOffsets[2]:s.linear.hardwareWidths[0];
    for(unsigned bit=0;bit<d.linear.hardwareBits;++bit) {
      auto value=d.linear.apply(APInt(std::max(1u,d.linear.hardwareBits),uint64_t{1}<<bit)).getZExtValue();
      if(bit>=d.linear.hardwareWidths[0]) {
        unsigned relative=bit-d.linear.hardwareWidths[0];unsigned sb=s.linear.hardwareWidths[0]+relative;
        if(sb>=free)value^=s.linear.apply(APInt(std::max(1u,s.linear.hardwareBits),uint64_t{1}<<sb)).getZExtValue();
      }
      if(!basis.solve(value))return false;
    }
    return true;
  }
  ConversionExchangeKind route(const Encoding &s,const Encoding &d) {
    if(s.attr==d.attr)return ConversionExchangeKind::Identity;
    if(sameScope(s,d,false))return ConversionExchangeKind::Register;
    if(sameScope(s,d,true))return ConversionExchangeKind::Shuffle;
    return ConversionExchangeKind::Shared;
  }
  std::optional<LayoutRegisterQuery> registerSource(const Encoding &s,const Encoding &d,const LayoutRegisterQuery &q) {
    LayoutRegisterQuery source;
    for(unsigned i=0;i<8;++i) {
      uint64_t input=q.terms[i];if(i)input|=uint64_t{1}<<(d.linear.hardwareWidths[0]+i-1);
      uint64_t logical=d.linear.apply(APInt(std::max(1u,d.linear.hardwareBits),input)).getZExtValue();
      if(i)logical^=s.linear.apply(APInt(std::max(1u,s.linear.hardwareBits),uint64_t{1}<<(s.linear.hardwareWidths[0]+i-1))).getZExtValue();
      auto selected=s.registers.solve(logical);if(!selected)return std::nullopt;source.terms[i]=*selected;
    }return source;
  }
  std::optional<LayoutVarID> operand(OpOperand &operand) {
    auto i=uses.find(&operand);if(i!=uses.end())return i->second;
    return g.lookupVariable(operand.get());
  }
  uint64_t mapInstructions(const std::array<uint64_t,128> &table) {
    std::array<uint64_t,7> basis{};uint64_t n=1;
    for(unsigned b=0;b<7;++b){basis[b]=table[1u<<b]^table[0];if(basis[b])n+=4;}
    for(unsigned t=0;t<128;++t) {
      uint64_t x=table[0];for(unsigned b=0;b<7;++b)if(t&(1u<<b))x^=basis[b];
      if(x!=table[t]){reject("demand address/query is outside checked affine emission domain");return 0;}
    }return n;
  }
  const std::map<uint64_t,std::pair<uint64_t,uint64_t>> *addressTable(const LayoutConstraint &c,LayoutVarID v) {
    auto found=addresses.find(&c);if(found!=addresses.end())return &found->second;
    LayoutVarID storage=c.vars[0]==v?c.vars[1]:c.vars[0];
    const auto &sv=g.getVariable(storage);auto attr=dyn_cast_or_null<StorageLayoutAttr>(assignment.values.lookup(storage));
    if(!attr || !sv.storageAlias){reject("demand missing storage alias/layout");return nullptr;}
    const auto &footprint=getStorageAliasFootprint(g,storage,attr);
    if(footprint.proof.status!=ProofStatus::Proven){reject("demand storage footprint unproven");return nullptr;}
    auto type=cast<RankedTensorType>(g.getVariable(v).shapedType);std::map<uint64_t,std::pair<uint64_t,uint64_t>> table;
    for(const auto &entry:footprint.entries) {
      if(entry.view.size()!=unsigned(type.getRank()))continue;uint64_t slot=0;bool valid=true;
      for(unsigned a=0;a<entry.view.size();++a){valid&=entry.view[a]>=0 && entry.view[a]<type.getDimSize(a);slot=slot*type.getDimSize(a)+entry.view[a];}
      if(!valid)continue;
      if(entry.begin%8 || entry.end-entry.begin!=32 || entry.end/8>uint64_t(std::numeric_limits<uint32_t>::max())){reject("demand requires bounded u32 f32 root byte addresses");return nullptr;}
      table[slot]={entry.begin/8,entry.end/8};
    }
    if(table.size()!=uint64_t(type.getNumElements())){reject("demand access footprint incomplete");return nullptr;}
    return &addresses.emplace(&c,std::move(table)).first->second;
  }
  void access(const LayoutConstraint &c,LayoutVarID id,const LayoutRegisterQuery &q) {
    if(!details)return;
    const auto &e=enc.at(id);LayoutVarID sid=c.vars[0]==id?c.vars[1]:c.vars[0];const auto &sv=g.getVariable(sid);
    auto storage=cast<StorageLayoutAttr>(assignment.values.lookup(sid));
    mlir::Builder attrs(storage.getContext());SmallVector<int64_t> terms;for(auto t:q.terms)terms.push_back(int64_t(t));
    auto signature=attrs.getDictionaryAttr({attrs.getNamedAttr("kind",attrs.getStringAttr("gf2-demand-query-v1")),attrs.getNamedAttr("terms",attrs.getDenseI64ArrayAttr(terms))});
    LayoutConstraintGraph::AccessCostCacheKey cacheKey{c.id,g.getVariable(id).shapedType,e.attr,sv.shapedType,storage,signature,sv.storageAlias?sv.storageAlias->rootAlignment:0,unsigned(c.access)};
    bool memo=g.isCostQueryCacheEnabled() && g.getCostQueryCacheCapacity();
    std::optional<LayoutProfile::Scope> profile;
    if(memo) {
      profile.emplace(LayoutProfileKind::AccessCostCache);auto found=g.getAccessCostCache().find(cacheKey);
      LayoutProfile::cacheLookup(LayoutProfileKind::AccessCostCache,found!=g.getAccessCostCache().end());
      if(found!=g.getAccessCostCache().end()) {
        const auto &cost=found->second;auto &out=result.accesses[c.id];out.constraint=c.id;++out.queryVectors;
        out.warpIssues+=cost.cost.instructionPathAndWork;out.globalSectors32+=cost.cost.memoryTransactions;out.sharedBankExcess+=cost.cost.bankConflictDegree;
        uint64_t work=cost.features.front().second;out.addressAluWarpIssues+=work;result.addressAluWarpIssues+=work;return;
      }
    }
    const auto *table=addressTable(c,id);if(!table)return;
    std::array<uint64_t,128> offsets{},owners{},current{};std::array<bool,128> active{};
    for(unsigned t=0;t<128;++t) {
      uint64_t logical=e.logical(q.apply(t),t),slot=e.slot(logical);auto it=table->find(slot);
      if(it==table->end()){reject("demand address absent");return;}offsets[t]=it->second.first;
      auto owner=e.all.solve(logical);if(!owner){reject("demand owner absent");return;}
      owners[t]=*owner;current[t]=uint64_t(t)*e.linear.topology[0]+q.apply(t);
      active[t]=c.access==AccessKind::Read || owners[t]==current[t];
    }
    uint64_t mapWork=mapInstructions(offsets)+2;
    if(c.access==AccessKind::Write)mapWork+=mapInstructions(owners)+mapInstructions(current)+1;
    if(failed)return;
    auto &out=result.accesses[c.id];out.constraint=c.id;++out.queryVectors;out.addressAluWarpIssues+=4*mapWork;result.addressAluWarpIssues+=4*mapWork;
    uint64_t alignment=std::min<uint64_t>(32,sv.storageAlias->rootAlignment);
    if(!alignment){reject("zero storage alignment");return;}
    bool global=storage.getMemorySpace().getValue()==attr::MemorySpace::Global;
    auto footprint=accessFootprints.evaluate(offsets,active,global,alignment,
                                           memo?g.getCostQueryCacheCapacity():0);
    if(!footprint){reject("demand footprint address arithmetic overflows");return;}
    out.warpIssues+=footprint->warpIssues;
    out.globalSectors32+=footprint->sectors;out.sharedBankExcess+=footprint->bankExcess;
    if(memo && g.getAccessCostCache().size()<g.getCostQueryCacheCapacity()) {
      CostEstimate cost;cost.cost.instructionPathAndWork=footprint->warpIssues;
      cost.cost.memoryTransactions=footprint->sectors;cost.cost.bankConflictDegree=footprint->bankExcess;
      cost.features.emplace_back("demand-address-alu-warp-issues-v1",4*mapWork);
      if(g.getAccessCostCache().try_emplace(std::move(cacheKey),std::move(cost)).second)LayoutProfile::cacheInsertion(LayoutProfileKind::AccessCostCache);
    }
  }
  uint64_t fetch(LayoutVarID id,const LayoutRegisterQuery &q) {
    if(failed)return 0;auto ei=enc.find(id);if(ei==enc.end()){reject("demand operand has no distributed layout");return 0;}
    const auto &e=ei->second;for(auto term:q.terms)if(term>=e.linear.topology[0]){reject("demand query exceeds register dimension");return 0;}
    auto alias=conversions.find(id);
    if((id>=g.getVariables().size() || g.getVariable(id).use) && alias!=conversions.end() && enc.at(alias->second->vars[0]).attr==e.attr)
      return fetch(alias->second->vars[0],q);
    auto key=std::make_pair(id,q);auto cached=cache.find(key);if(cached!=cache.end())return cached->second;
    if(cache.size()+pending.size()>=budget){reject("demand affine-query budget exceeded");return 0;}
    if(!pending.insert(key).second){reject("cyclic demand SSA");return 0;}uint64_t depth=0;
    auto cv=conversions.find(id);
    if(cv!=conversions.end()) {
      const auto &c=*cv->second;LayoutVarID source=c.vars[0];const auto &s=enc.at(source);auto kind=route(s,e);
      if(kind==ConversionExchangeKind::Identity)depth=fetch(source,q);
      else if(result.demandLowering && kind==ConversionExchangeKind::Register) {
        auto sq=registerSource(s,e,q);if(!sq){reject("demand Register composition failed");return 0;}depth=fetch(source,*sq);
        auto &b=result.boundaries[c.id];b.constraint=c.id;b.source=source;b.target=c.vars[1];b.exchange=kind;++b.targetVectors;
      } else {
        if(std::max(s.linear.topology[0],e.linear.topology[0])>512){reject("large cross-thread demand boundary unsupported");return 0;}
        if(materialized.insert(id).second) {
          for(uint64_t r=0;r<s.linear.topology[0];++r)depth=std::max(depth,fetch(source,uniform(r)));
          if(failed)return 0;
          if(kind==ConversionExchangeKind::Shared || kind==ConversionExchangeKind::Shuffle)finishGlobalLoadGroup();
          auto &b=result.boundaries[c.id];b.constraint=c.id;b.source=source;b.target=c.vars[1];b.exchange=kind;b.sourceVectors=s.linear.topology[0];b.targetVectors=e.linear.topology[0];
          if(!details) { b.barriers=kind==ConversionExchangeKind::Shared?2:0; }
          else if(kind==ConversionExchangeKind::Shared) {
            uint64_t writers=0;
            for(uint64_t r=0;r<b.sourceVectors;++r) {
              std::array<uint64_t,128> offsets{},owners{},current{};std::array<bool,128> active{};
              for(unsigned t=0;t<128;++t){uint64_t logical=s.logical(r,t);offsets[t]=s.slot(logical)*4;owners[t]=*s.all.solve(logical);current[t]=uint64_t(t)*b.sourceVectors+r;active[t]=owners[t]==current[t];writers+=active[t];}
              b.addressAluWarpIssues+=4*(mapInstructions(offsets)+mapInstructions(owners)+mapInstructions(current)+3);
              for(unsigned w=0;w<4;++w){std::map<unsigned,std::set<uint64_t>> banks;for(unsigned l=0;l<32;++l){unsigned t=w*32+l;if(active[t])banks[(offsets[t]/4)%32].insert(offsets[t]/4);}uint64_t degree=1;for(auto &[bank,words]:banks)degree=std::max<uint64_t>(degree,words.size());b.sharedBankExcess+=degree-1;b.sharedWriteBankExcess+=degree-1;if(!banks.empty())++b.sharedWriteWarpIssues;}
            }
            for(uint64_t r=0;r<b.targetVectors;++r){std::array<uint64_t,128> offsets{};for(unsigned t=0;t<128;++t)offsets[t]=e.slot(e.logical(r,t))*4;b.addressAluWarpIssues+=4*(mapInstructions(offsets)+2);for(unsigned w=0;w<4;++w){std::map<unsigned,std::set<uint64_t>> banks;for(unsigned l=0;l<32;++l){unsigned t=w*32+l;banks[(offsets[t]/4)%32].insert(offsets[t]/4);}uint64_t degree=1;for(auto &[bank,words]:banks)degree=std::max<uint64_t>(degree,words.size());b.sharedBankExcess+=degree-1;b.sharedReadBankExcess+=degree-1;++b.sharedReadWarpIssues;}}
            b.bytes=4*(writers+128*b.targetVectors);b.barriers=2;
            result.scratchMaxBytes=std::max(result.scratchMaxBytes,uint64_t{4}<<e.linear.logicalBits);
          } else {
            std::vector<uint64_t> shuffleOwners;
            if (kind==ConversionExchangeKind::Shuffle && orderedEmission)
              shuffleOwners.resize(128*b.targetVectors);
            for(uint64_t r=0;r<b.targetVectors;++r){std::array<uint64_t,128> registers{},lanes{};llvm::SmallBitVector support(b.sourceVectors);
              for(unsigned t=0;t<128;++t){uint64_t logical=e.logical(r,t);std::optional<uint64_t> owner;
                if(kind==ConversionExchangeKind::Register)owner=s.registers.solve(logical^s.logical(0,t));
                else {
                  // The exported plan prefers a same-thread holder before the
                  // canonical warp holder, including replicated GF(2) layouts.
                  auto local=s.registers.solve(logical^s.logical(0,t));
                  owner=local?std::optional<uint64_t>(*local+(t%32)*b.sourceVectors):s.warp.solve(logical^s.logical(0,(t/32)*32));
                }
                if(!owner){reject("bounded conversion owner unavailable");return 0;}registers[t]=*owner%b.sourceVectors;lanes[t]=(*owner/b.sourceVectors)%32;
                if(kind==ConversionExchangeKind::Shuffle) {
                  support.set(registers[t]);
                  if(!shuffleOwners.empty())shuffleOwners[t*b.targetVectors+r]=
                      uint64_t((t/32)*32+lanes[t])*b.sourceVectors+registers[t];
                }
              }
              uint64_t count=kind==ConversionExchangeKind::Shuffle?support.count():b.sourceVectors;
              b.addressAluWarpIssues+=4*(mapInstructions(registers)+1+2*(count-1));
              if(kind==ConversionExchangeKind::Shuffle){b.addressAluWarpIssues+=4*mapInstructions(lanes);b.shuffleWarpIssues+=4*count;}
            }
            if(kind==ConversionExchangeKind::Shuffle) {
              if(!shuffleOwners.empty())
                if(auto network=shuffle_network::plan(b.sourceVectors,b.targetVectors,shuffleOwners,networkObjective)) {
                  b.shuffleNetwork=*network;
                  b.addressAluWarpIssues=network->counts.addressAluWarpIssues;
                  b.shuffleWarpIssues=network->counts.shuffleWarpIssues;
                }
              b.bytes=b.shuffleWarpIssues*32*4;
            }
          }
          result.addressAluWarpIssues+=b.addressAluWarpIssues;
          boundaryDepth[id]=depth+1+b.barriers;
        }
        depth=boundaryDepth[id];
        // Variable register selection is bounded and algebra checked.
        bool variable=false;for(unsigned i=1;i<8;++i)variable|=q.terms[i]!=0;
        if(variable && details) {
          std::array<uint64_t,128> table{};for(unsigned t=0;t<128;++t)table[t]=q.apply(t);
          uint64_t muxWork=4*(mapInstructions(table)+1+2*(e.linear.topology[0]-1));
          // This receiver selection is emitted after the materialized exchange.
          // Keep it in the charged boundary, not just the aggregate diagnostic.
          auto &b=result.boundaries[c.id];b.addressAluWarpIssues+=muxWork;
          b.receiverMuxAluWarpIssues+=muxWork;result.addressAluWarpIssues+=muxWork;
        }
      }
    } else {
      const auto &v=g.getVariable(id);Operation *op=v.value?v.value.getDefiningOp():nullptr;
      if(!op){reject("demand SSA producer missing");return 0;}
      StringRef name=op->getName().getStringRef();
      if(name=="frisk.tile_load") {
        auto a=readAccess.find(id);if(a==readAccess.end()){reject("load has no demand access constraint");return 0;}
        access(*a->second,id,q);++result.loadVectors;depth=1;
        const auto &accessConstraint=*a->second;
        auto storageID=accessConstraint.vars[0]==id?accessConstraint.vars[1]:accessConstraint.vars[0];
        auto storage=dyn_cast_or_null<StorageLayoutAttr>(assignment.values.lookup(storageID));
        if(storage && storage.getMemorySpace().getValue()==attr::MemorySpace::Global)issueGlobalLoad();
        else finishGlobalLoadGroup();
      } else if(name=="arith.constant"){depth=0;result.arithmeticWarpIssues+=4;}
      else {
        for(auto &operand:op->getOpOperands()) {
          auto a=this->operand(operand);if(!a){reject("demand arithmetic operand missing");return 0;}
          const auto &other=enc.at(*a);
          // These are affine maps over F2^7: zero and the seven unit vectors
          // prove equality on the entire thread domain, rather than sampling it.
          for(unsigned i=0;i<8;++i){unsigned t=i?1u<<(i-1):0;if(other.logical(q.apply(t),t)!=e.logical(q.apply(t),t)){reject("demand arithmetic layout mismatch");return 0;}}
          depth=std::max(depth,fetch(*a,q));
        }finishGlobalLoadGroup();++depth;result.arithmeticWarpIssues+=4;
      }
    }
    pending.erase(key);if(failed)return 0;cache.emplace(key,depth);
    if(id<g.getVariables().size() && !g.getVariable(id).use) {
      auto &v=result.values[id];v.variable=id;v.queries.push_back(q);v.dependencyDepth=std::max(v.dependencyDepth,depth);
    }
    if(cv!=conversions.end() && enc.at(cv->second->vars[0]).attr!=e.attr)result.boundaries[cv->second->id].queries.push_back(q);
    result.dependencySpan=std::max(result.dependencySpan,depth);return depth;
  }
};
}
static void runDemandBuilder(DemandBuilder &b,ArrayRef<LayoutVarID> localRoots) {
  const auto &graph=b.g;
  if(!b.prepare())return;
  b.result.localScope=!localRoots.empty();
  std::set<LayoutVarID> reachable;
  SmallVector<LayoutVarID> orderedRoots(localRoots);
  if(b.orderedEmission)llvm::sort(orderedRoots,[&](LayoutVarID x,LayoutVarID y) {
    if(x<graph.getVariables().size() && y<graph.getVariables().size()) {
      const auto &vx=graph.getVariable(x),&vy=graph.getVariable(y);
      auto *a=vx.value?vx.value.getDefiningOp():nullptr,*c=vy.value?vy.value.getDefiningOp():nullptr;
      if(a && c && a!=c && a->getBlock()==c->getBlock())return a->isBeforeInBlock(c);
      if(vx.stableName!=vy.stableName)return vx.stableName<vy.stableName;
    }
    return x<y;
  });
  if(!localRoots.empty()) {
    // The local executor mode must depend only on its producer cone, never a
    // future consumer's large-register candidate elsewhere in the graph.
    b.result.demandLowering=false;SmallVector<LayoutVarID> queue(localRoots);
    while(!queue.empty()) {
      auto id=queue.pop_back_val();if(!reachable.insert(id).second)continue;
      auto e=b.enc.find(id);if(e==b.enc.end()){b.reject("local demand root missing");break;}
      b.result.demandLowering|=e->second.linear.topology[0]>256;
      auto cv=b.conversions.find(id);if(cv!=b.conversions.end()){queue.push_back(cv->second->vars[0]);continue;}
      if(id>=graph.getVariables().size())continue;
      auto &v=graph.getVariable(id);auto *op=v.value?v.value.getDefiningOp():nullptr;
      if(!op || op->getName().getStringRef()=="frisk.tile_load" || op->hasTrait<OpTrait::ConstantLike>())continue;
      for(auto &operand:op->getOpOperands()){auto input=b.operand(operand);if(input)queue.push_back(*input);}
    }
  }
  // The adapter allocates scratch before traversing demand. Every declared
  // Shared conversion reserves the same serial segment even if its output is
  // dead in the large-demand traversal. An independent local producer kernel
  // declares only its backward cone; future consumers cannot reserve scratch.
  if(b.details)for(const auto &[endpoint,c]:b.conversions) {
    if(!localRoots.empty() && !reachable.count(endpoint))continue;
    const auto &source=b.enc.at(c->vars[0]),&destination=b.enc.at(endpoint);
    if(b.route(source,destination)==ConversionExchangeKind::Shared)
      b.result.scratchMaxBytes=std::max(b.result.scratchMaxBytes,uint64_t{4}<<destination.linear.logicalBits);
  }
  if(!b.orderedEmission) {
    // Frozen v4d traversal. Its local objective predates the ordered-emission
    // contract; retain those scores when loading an existing17-feature profile.
    if(!localRoots.empty()) {
      for(auto id:localRoots) {
        auto e=b.enc.find(id);if(e==b.enc.end()){b.reject("local demand root not distributed");break;}
        if(e->second.linear.topology[0]>b.budget){b.reject("local demand root exceeds query budget");break;}
        for(uint64_t r=0;r<e->second.linear.topology[0] && !b.failed;++r)b.fetch(id,b.uniform(r));
      }
    } else if(b.result.demandLowering) {
      uint64_t nr=b.enc.at(b.storeValue).linear.topology[0];
      if(nr>b.budget)b.reject("final store exceeds demand query budget");
      for(uint64_t r=0;r<nr && !b.failed;++r){auto q=b.uniform(r);b.fetch(b.storeValue,q);b.access(*b.storeAccess,b.storeValue,q);}
    } else {
      // Small eager backend materializes every tensor definition, not merely
      // consumers reachable from the store. Preserve its actual execution scope.
      for(const auto &[id,e]:b.enc)if(id<graph.getVariables().size() && !graph.getVariable(id).use)
        for(uint64_t r=0;r<e.linear.topology[0] && !b.failed;++r)b.fetch(id,b.uniform(r));
      uint64_t nr=b.enc.at(b.storeValue).linear.topology[0];
      for(uint64_t r=0;r<nr && !b.failed;++r){auto q=b.uniform(r);b.fetch(b.storeValue,q);b.access(*b.storeAccess,b.storeValue,q);}
    }
  } else {
    if(!b.result.demandLowering) {
      // The eager adapter walks original SSA operations, not finalized graph
      // IDs. A local kernel includes only the producer cone in that IR order.
      for(Operation &op:*b.producerBlock)for(Value value:op.getResults()) {
        auto id=graph.lookupVariable(value);
        if(!id || !b.enc.count(*id) || (!localRoots.empty() && !reachable.count(*id)))continue;
        const auto &e=b.enc.at(*id);
        for(uint64_t r=0;r<e.linear.topology[0] && !b.failed;++r)b.fetch(*id,b.uniform(r));
      }
    } else if(!localRoots.empty()) {
      // SelectionScope roots form a set. The independent local output order
      // is the producer IR order, never caller vector order or graph IDs.
      for(auto id:orderedRoots) {
        auto e=b.enc.find(id);if(e==b.enc.end()){b.reject("local demand root not distributed");break;}
        if(e->second.linear.topology[0]>b.budget){b.reject("local demand root exceeds query budget");break;}
        for(uint64_t r=0;r<e->second.linear.topology[0] && !b.failed;++r)b.fetch(id,b.uniform(r));
      }
    } else {
      uint64_t nr=b.enc.at(b.storeValue).linear.topology[0];
      if(nr>b.budget)b.reject("final store exceeds demand query budget");
      for(uint64_t r=0;r<nr && !b.failed;++r)b.fetch(b.storeValue,b.uniform(r));
    }
    if(localRoots.empty()) {
      // Both actual adapters issue the one final store only after all reads
      // and computations. Interleaved footprint accounting is not emission.
      uint64_t nr=b.enc.at(b.storeValue).linear.topology[0];
      // A store execution endpoint is a use variable, not an original SSA
      // result visited above. Materialize its required input repair before
      // writes, including in count-only backend-budget proofs. Cached source
      // queries do not execute again; the actual conversion still does.
      for(uint64_t r=0;r<nr && !b.failed;++r)b.fetch(b.storeValue,b.uniform(r));
      b.finishGlobalLoadGroup();
      for(uint64_t r=0;r<nr && !b.failed;++r)b.access(*b.storeAccess,b.storeValue,b.uniform(r));
    }
  }
  b.result.materializedQueryVectors=b.cache.size();b.result.exact=!b.failed;
  if(!b.failed)b.result.reason=b.orderedEmission?"gf2-affine-consumer-demand-v2-ordered":"gf2-affine-consumer-demand-v1";
}
LayoutDemandLedger mlir::frisk::analyzeLayoutDemand(const LayoutConstraintGraph &graph,
    const CandidateAssignment &assignment,ArrayRef<LayoutVarID> localRoots,uint64_t queryBudget,bool orderedEmission,shuffle_network::Objective networkObjective) {
  if(!orderedEmission && networkObjective!=shuffle_network::Objective::MinShuffle) {LayoutDemandLedger out;out.reason="historical v4 grammar cannot select a network variant";return out;}
  DemandBuilder b(graph,assignment,queryBudget,true,orderedEmission,networkObjective);runDemandBuilder(b,localRoots);return std::move(b.result);
}

llvm::json::Object mlir::frisk::layoutDemandJSON(const LayoutDemandLedger &l) {
  llvm::json::Array values,accesses,boundaries;
  for(const auto &[id,v]:l.values){llvm::json::Array queries;for(const auto &q:v.queries){llvm::json::Array terms;for(auto t:q.terms)terms.push_back(t);queries.push_back(llvm::json::Object{{"terms",std::move(terms)}});}values.push_back(llvm::json::Object{{"variable",id},{"queries",std::move(queries)},{"dependency_depth",v.dependencyDepth}});}
  for(const auto &[id,a]:l.accesses)accesses.push_back(llvm::json::Object{{"constraint",id},{"query_vectors",a.queryVectors},{"warp_issues",a.warpIssues},{"global_sectors32",a.globalSectors32},{"shared_bank_excess",a.sharedBankExcess},{"address_alu_warp_issues",a.addressAluWarpIssues}});
  for(const auto &[id,b]:l.boundaries){llvm::json::Array queries;for(const auto &q:b.queries){llvm::json::Array terms;for(auto t:q.terms)terms.push_back(t);queries.push_back(llvm::json::Object{{"terms",std::move(terms)}});}boundaries.push_back(llvm::json::Object{{"constraint",id},{"source",b.source},{"target",b.target},{"route",unsigned(b.exchange)},{"source_vectors",b.sourceVectors},{"target_vectors",b.targetVectors},{"bytes",b.bytes},{"barriers",b.barriers},{"shared_bank_excess",b.sharedBankExcess},{"shared_read_bank_excess",b.sharedReadBankExcess},{"shared_write_bank_excess",b.sharedWriteBankExcess},{"shared_read_warp_issues",b.sharedReadWarpIssues},{"shared_write_warp_issues",b.sharedWriteWarpIssues},{"address_alu_warp_issues",b.addressAluWarpIssues},{"receiver_mux_alu_warp_issues",b.receiverMuxAluWarpIssues},{"shuffle_warp_issues",b.shuffleWarpIssues},{"queries",std::move(queries)}});}
  return llvm::json::Object{{"schema_version",1},{"exact",l.exact},{"reason",l.reason},{"demand_lowering",l.demandLowering},{"local_scope",l.localScope},{"declared_register_vectors",l.declaredRegisterVectors},{"materialized_query_vectors",l.materializedQueryVectors},{"load_vectors",l.loadVectors},{"global_load_issue_groups",l.globalLoadIssueGroups},{"arithmetic_warp_issues",l.arithmeticWarpIssues},{"address_alu_warp_issues",l.addressAluWarpIssues},{"dependency_span",l.dependencySpan},{"scratch_max_bytes",l.scratchMaxBytes},{"values",std::move(values)},{"accesses",std::move(accesses)},{"boundaries",std::move(boundaries)}};
}

LayoutSurvivingDemandAnalysis mlir::frisk::analyzeLayoutSurvivingDemand(
    const LayoutConstraintGraph &graph,const CandidateAssignment &assignment,
    ArrayRef<LayoutVarID> localRoots,uint64_t queryBudget,shuffle_network::Objective networkObjective) {
  DemandBuilder builder(graph,assignment,queryBudget,true,true,networkObjective);runDemandBuilder(builder,localRoots);
  LayoutSurvivingDemandAnalysis out;out.emitted=std::move(builder.result);
  if(!out.emitted.exact){out.reason="emission analysis unavailable: "+out.emitted.reason;return out;}
  std::map<LayoutConstraintID,LayoutVarID> endpoint;
  for(const auto &[id,c]:builder.conversions)endpoint.emplace(c->id,id);
  std::set<LayoutVarID> retained;SmallVector<LayoutVarID> queue(localRoots);
  if(localRoots.empty())queue.push_back(builder.storeValue);
  // These are emitted communication roots, not every declared conversion.
  // Removing their scaffolds would assume inter-thread/barrier DCE in ptxas.
  for(const auto &[id,b]:out.emitted.boundaries)
    if(b.exchange==ConversionExchangeKind::Shared || b.exchange==ConversionExchangeKind::Shuffle) {
      auto found=endpoint.find(id);
      if(found==endpoint.end()){out.reason="communication endpoint missing";return out;}
      queue.push_back(found->second);out.retainedCommunication.push_back(id);
    }
  while(!queue.empty()) {
    auto id=queue.pop_back_val();if(!retained.insert(id).second)continue;
    if(!builder.enc.count(id)){out.reason="survival root has no distributed encoding";return out;}
    auto conversion=builder.conversions.find(id);
    if(conversion!=builder.conversions.end()) {queue.push_back(conversion->second->vars[0]);continue;}
    if(id>=graph.getVariables().size()){out.reason="survival endpoint definition missing";return out;}
    const auto &v=graph.getVariable(id);auto *op=v.value?v.value.getDefiningOp():nullptr;
    if(!op){out.reason="survival SSA producer missing";return out;}
    if(op->getName().getStringRef()=="frisk.tile_load" || op->hasTrait<OpTrait::ConstantLike>())continue;
    for(auto &operand:op->getOpOperands()) {
      auto input=builder.operand(operand);
      if(!input){out.reason="survival arithmetic operand missing";return out;}
      queue.push_back(*input);
    }
  }
  out.surviving=out.emitted;
  for(const auto &v:graph.getVariables())if(v.kind==LayoutKind::Distributed)
    (retained.count(v.id)?out.retainedVariables:out.removedVariables).push_back(v.id);
  for(auto it=out.surviving.values.begin();it!=out.surviving.values.end();)
    if(!retained.count(it->first))it=out.surviving.values.erase(it);else ++it;
  for(auto it=out.surviving.accesses.begin();it!=out.surviving.accesses.end();) {
    const auto &c=graph.getConstraint(it->first);LayoutVarID value=builder.enc.count(c.vars[0])?c.vars[0]:c.vars[1];
    if(c.access==AccessKind::Read && !retained.count(value))it=out.surviving.accesses.erase(it);else ++it;
  }
  for(auto it=out.surviving.boundaries.begin();it!=out.surviving.boundaries.end();) {
    auto found=endpoint.find(it->first);
    if(found==endpoint.end()){out.reason="surviving boundary endpoint missing";return out;}
    if(!retained.count(found->second))it=out.surviving.boundaries.erase(it);else ++it;
  }
  out.surviving.materializedQueryVectors=0;out.surviving.dependencySpan=0;
  for(const auto &[key,depth]:builder.cache)if(retained.count(key.first)) {
    ++out.surviving.materializedQueryVectors;
    out.surviving.dependencySpan=std::max(out.surviving.dependencySpan,depth);
  }
  out.removedQueryVectors=out.emitted.materializedQueryVectors-out.surviving.materializedQueryVectors;
  out.surviving.loadVectors=0;out.surviving.arithmeticWarpIssues=0;out.surviving.addressAluWarpIssues=0;
  for(const auto &[id,a]:out.surviving.accesses) {
    if(graph.getConstraint(id).access==AccessKind::Read)out.surviving.loadVectors+=a.queryVectors;
    out.surviving.addressAluWarpIssues+=a.addressAluWarpIssues;
  }
  for(const auto &[id,value]:out.surviving.values) {
    auto *op=graph.getVariable(id).value.getDefiningOp();auto dialect=op->getName().getDialectNamespace();
    if(dialect=="arith" || dialect=="math")out.surviving.arithmeticWarpIssues+=4*value.queries.size();
  }
  for(const auto &[id,b]:out.surviving.boundaries)out.surviving.addressAluWarpIssues+=b.addressAluWarpIssues;
  // There is no claimed native schedule for a filtered ledger. v6 never
  // prices this old emitter-order feature; JSON exports it as unavailable.
  out.surviving.globalLoadIssueGroups=0;
  out.surviving.reason="whole-ssa-survival-with-emitted-communication-roots-v1";
  out.proven=true;out.reason=out.surviving.reason;return out;
}

llvm::json::Object mlir::frisk::layoutSurvivingDemandJSON(const LayoutSurvivingDemandAnalysis &a) {
  llvm::json::Array retained,removed,communication;
  for(auto id:a.retainedVariables)retained.push_back(id);
  for(auto id:a.removedVariables)removed.push_back(id);
  for(auto id:a.retainedCommunication)communication.push_back(id);
  auto surviving=layoutDemandJSON(a.surviving);
  surviving["global_load_issue_groups"]=nullptr;
  surviving["order_semantics"]="not rescheduled; native scheduling is separate compiled feedback";
  return llvm::json::Object{{"schema_version",1},{"proof_status",a.proven?"Proven":"Unavailable"},
    {"work_basis","whole-ssa-survival-with-emitted-communication-roots-v1"},{"native_exact",false},
    {"reason",a.reason},{"retained_variables",std::move(retained)},{"removed_variables",std::move(removed)},
    {"retained_communication",std::move(communication)},{"removed_query_vectors",a.removedQueryVectors},
    {"assumptions","bounds-valid nonvolatile pure f32 DAG; no lane pruning, cross-SSA/address CSE or register-allocation prediction"},
    {"emitted",layoutDemandJSON(a.emitted)},{"surviving",std::move(surviving)}};
}

LayoutNetworkVariantComparison mlir::frisk::compareLayoutNetworkObjectives(
    const LayoutConstraintGraph &graph,const CandidateAssignment &assignment) {
  auto a=analyzeLayoutDemand(graph,assignment,{},65536,true,shuffle_network::Objective::MinShuffle);
  auto b=analyzeLayoutDemand(graph,assignment,{},65536,true,shuffle_network::Objective::MinWork);
  LayoutNetworkVariantComparison result;
  if(!a.exact || !b.exact){result.reason="actual ordered execution ledger unavailable: "+(!a.exact?a.reason:b.reason);return result;}
  if(a.boundaries.size()!=b.boundaries.size()){result.reason="objective changed execution boundary set";return result;}
  for(const auto &[id,x]:a.boundaries) {
    auto found=b.boundaries.find(id);if(found==b.boundaries.end()){result.reason="objective changed execution boundary identity";return result;}
    const auto &y=found->second;
    if(x.exchange!=y.exchange || x.source!=y.source || x.target!=y.target || x.sourceVectors!=y.sourceVectors || x.targetVectors!=y.targetVectors){result.reason="objective changed transport endpoint or mapping domain";return result;}
    if(x.exchange!=ConversionExchangeKind::Shuffle)continue;++result.comparedShuffleBoundaries;
    llvm::json::Object proof{{"constraint",id},{"source",x.source},{"target",x.target},{"source_registers",x.sourceVectors},{"target_registers",x.targetVectors}};
    proof["min-shfl"]=x.shuffleNetwork?llvm::json::Value(shuffle_network::planJSON(*x.shuffleNetwork)):llvm::json::Value(nullptr);
    proof["min-work"]=y.shuffleNetwork?llvm::json::Value(shuffle_network::planJSON(*y.shuffleNetwork)):llvm::json::Value(nullptr);
    result.boundaryPlans.push_back(std::move(proof));
    if(bool(x.shuffleNetwork)!=bool(y.shuffleNetwork)){result.changed=true;continue;}
    if(x.shuffleNetwork && !shuffle_network::sameExecutionPlan(*x.shuffleNetwork,*y.shuffleNetwork))result.changed=true;
  }
  result.proven=true;result.reason=result.changed?"verified actual transport steps/counts differ":"all actual transport steps/counts and owner maps identical";return result;
}

LayoutProof mlir::frisk::verifyLayoutDemandBackendSupport(const LayoutConstraintGraph &g,const CandidateAssignment &a) {
  DemandBuilder b(g,a,65536);bool partial=llvm::any_of(g.getVariables(),[&](const auto &v){return !a.values.count(v.id);});
  for(const auto &v:g.getVariables())if(v.kind==LayoutKind::Distributed) {
    auto found=a.values.find(v.id);if(found==a.values.end()){partial=true;continue;}
    auto type=dyn_cast<RankedTensorType>(v.shapedType);Encoding e;e.attr=dyn_cast_or_null<DistributedEncodingAttr>(found->second);
    if(e.attr && !e.attr.getTopology().empty() && e.attr.getTopology()[0]>65536)
      return {ProofStatus::Disproven,{},"scalar execution domain v1 requires at most65536 declared registers per tensor"};
    if(!type || !type.getElementType().isF32() || !e.attr || analyzeLinearExecutionLayout(e.attr,type,e.linear).status!=ProofStatus::Proven || e.linear.logicalBits>16 || e.linear.hardwareBits>=63 || e.linear.topology[1]!=32 || e.linear.topology[2]!=4 || e.linear.topology[3]!=1 || e.linear.topology[4]!=1 || proveExecutionOwnership(e.attr,type,"first_owner").status!=ProofStatus::Proven)
      return {ProofStatus::Disproven,{},"scalar execution domain requires proven f32 128-thread dyadic layout"};
    for(unsigned bit=0;bit<e.linear.hardwareBits;++bit){uint64_t col=e.linear.apply(APInt(std::max(1u,e.linear.hardwareBits),uint64_t{1}<<bit)).getZExtValue();e.all.insert(col,uint64_t{1}<<bit);if(bit<e.linear.hardwareWidths[0])e.registers.insert(col,uint64_t{1}<<bit);if(bit<e.linear.hardwareOffsets[2])e.warp.insert(col,uint64_t{1}<<bit);}
    b.enc.emplace(v.id,std::move(e));
  }
  for(const auto &c:g.getConstraints())if(c.kind==ConstraintKind::Convertible && c.vars.size()==2 && b.enc.count(c.vars[0]) && b.enc.count(c.vars[1])) {
    const auto &s=b.enc.at(c.vars[0]),&d=b.enc.at(c.vars[1]);if(s.attr==d.attr)continue;
    uint64_t sr=s.linear.topology[0],dr=d.linear.topology[0];auto route=b.route(s,d);
    // This backend emits static Shared allocations. Their ptxas limit is
    // 48 KiB; SM90's larger opt-in dynamic capacity is a different backend
    // contract, not a reason to reject the hardware layout itself.
    if(route==ConversionExchangeKind::Shared && (uint64_t{4}<<d.linear.logicalBits)>49152)return {ProofStatus::Disproven,{},"scalar execution domain v1: backend unsupported: serial Shared scratch exceeds current static backend limit49152 bytes"};
    if(sr>512) {
      if(route!=ConversionExchangeKind::Register || d.linear.hardwareBits>16)return {ProofStatus::Disproven,{},"scalar execution domain: large source requires same-thread Register and bounded destination"};
    } else if(dr>512)return {ProofStatus::Disproven,{},"scalar execution domain: ordinary redistribution destination exceeds512 registers"};
  }
  if(!partial) {
    // v1 is the current four-pointer weak ABI, including declared roots
    // unused by demand. Shared roots would still be physically allocated by
    // the backend; they cannot disappear from this feasibility certificate.
    std::set<const void*> roots,rootDeclarations;
    for(const auto &v:g.getVariables())if(v.kind==LayoutKind::Storage) {
      auto storage=dyn_cast_or_null<StorageLayoutAttr>(a.values.lookup(v.id));
      if(!storage || storage.getMemorySpace().getValue()!=attr::MemorySpace::Global || !v.storageAlias || !v.storageAlias->root)
        return {ProofStatus::Disproven,{},"scalar execution domain v1 requires every declared storage root/view, including unused roots, to be Global"};
      roots.insert(v.storageAlias->root.getAsOpaquePointer());
    }
    // Root declarations deliberately do not create graph variables. Inspect
    // their immutable IR identities so an unused declaration is still counted.
    // Their full storage proofs were checked while finalizing this graph; no
    // logical-footprint expansion is repeated for each candidate assignment.
    Operation *scope=nullptr;
    if(!g.getVariables().empty()) {
      const auto &v=g.getVariables().front();scope=v.anchor;
      Value anchorValue=v.value;if(!scope && anchorValue)scope=anchorValue.getParentBlock()->getParentOp();
      while(scope && scope->getParentOp())scope=scope->getParentOp();
    }
    bool nonGlobalRoot=false,nonWeakRoot=false;
    if(scope)scope->walk([&](StorageContractOp declaration) {
      auto layout=declaration.getLayout();
      rootDeclarations.insert(declaration.getRoot().getAsOpaquePointer());
      nonGlobalRoot|=layout.getMemorySpace().getValue()!=attr::MemorySpace::Global;
      nonWeakRoot|=layout.getAlignment().getInt()!=1;
    });
    if(nonGlobalRoot)
      return {ProofStatus::Disproven,{},"scalar execution domain v1 requires every declared storage root/view, including unused roots, to be Global"};
    if(nonWeakRoot)
      return {ProofStatus::Disproven,{},"scalar execution domain v1 requires four weak-alignment1 Global root declarations"};
    if(rootDeclarations.size()!=4 || !llvm::all_of(roots,[&](const void *root){return rootDeclarations.count(root);}))
      return {ProofStatus::Disproven,{},"scalar execution domain v1 requires exactly four declared Global roots"};
    DemandBuilder count(g,a,65536,false);runDemandBuilder(count,{});
    if(!count.result.exact)return {ProofStatus::Disproven,{},"scalar compiler scope/query budget v1: "+count.result.reason};
    // The v1 execution domain declares ordinary row-major Global f32 views.
    // Verify it for every access, not just for the layouts selected by v4.
    for(const auto &c:g.getConstraints())if(c.kind==ConstraintKind::StorageAccess && c.vars.size()==2) {
      LayoutVarID execution=g.getVariable(c.vars[0]).kind==LayoutKind::Distributed?c.vars[0]:c.vars[1];
      LayoutVarID storage=execution==c.vars[0]?c.vars[1]:c.vars[0];
      auto enc=dyn_cast_or_null<StorageLayoutAttr>(a.values.lookup(storage));
      if(!enc || enc.getMemorySpace().getValue()!=attr::MemorySpace::Global)return {ProofStatus::Disproven,{},"scalar execution domain v1 requires Global row-major f32 storage"};
      auto table=count.addressTable(c,execution);
      if(!table || table->empty())return {ProofStatus::Disproven,{},"scalar execution domain v1 storage footprint unavailable"};
      uint64_t base=table->begin()->second.first;
      if(base!=0)return {ProofStatus::Disproven,{},"scalar execution domain v1 requires zero root-relative view offset"};
      for(const auto &[slot,range]:*table)if(range.second-range.first!=4 || range.first!=base+4*slot)
        return {ProofStatus::Disproven,{},"scalar execution domain v1 requires a contiguous row-major f32 view"};
    }
  }
  return {ProofStatus::Proven,{},partial?"no assigned execution endpoint pair violates the declared domain":"scalar execution domain and query budget proven"};
}
