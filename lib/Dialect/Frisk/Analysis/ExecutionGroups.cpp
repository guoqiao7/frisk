#include "Dialect/Frisk/Analysis/ExecutionGroups.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::frisk;

namespace {
// The FA3 storage proof is root-relative. A SW128 descriptor has a 1024-byte
// period. Checking every ring origin lets that proof cover each physical slot.
int64_t contractInt(DictionaryAttr dict, StringRef name) {
  auto attr = dict ? dict.getAs<IntegerAttr>(name) : IntegerAttr();
  return attr && attr.getType().isSignlessInteger(64) ? attr.getInt() : -1;
}

LogicalResult verifyOnlineResources(ModuleOp module, DictionaryAttr plan,
                                    ArrayRef<int64_t> widths) {
  auto kind = plan.getAs<StringAttr>("kind");
  auto buffers = plan.getAs<DictionaryAttr>("buffers");
  auto roles = plan.getAs<DictionaryAttr>("roles");
  int64_t batch = contractInt(plan, "batch");
  if (!kind || !buffers || contractInt(plan,"version") != 1 ||
      contractInt(plan,"wait_pending") != 0)
    return module.emitError("invalid execution buffer or wait protocol");
  bool grouped = kind.getValue() == "fa3_qk_next" || kind.getValue() == "fa3_same_tile";
  if (!grouped) {
    auto publication = plan.getAs<StringAttr>("publication");
    if (kind.getValue() != "single" || widths.size()!=1 || widths.front()!=256 ||
        (batch!=1 && batch!=4) || !publication || publication.getValue()!="cta_barrier")
      return module.emitError("invalid single-group publication or wait protocol");
    for (StringRef key : {"q","k","v"})
      if (contractInt(buffers.getAs<DictionaryAttr>(key),"versions")!=1)
        return module.emitError("single-group publication requires one buffer version");
  } else {
    int64_t c = contractInt(roles,"qk"), p = contractInt(roles,"k");
    if (widths.size()!=2 || c<0 || c>1 || p!=1-c || widths[c]!=256 || widths[p]!=128 ||
        contractInt(roles,"pv")!=c || contractInt(roles,"q")!=c ||
        contractInt(roles,"reduce")!=c || contractInt(roles,"v")!=p || batch!=8)
      return module.emitError("incompatible FA3 execution domains or intrinsic wait batch");
    const int64_t arena = contractInt(plan,"arena");
    if (arena<=0 || arena>227328) return module.emitError("invalid FA3 shared arena");
    struct LiveBuffer { StringRef name; int64_t begin, end; };
    SmallVector<LiveBuffer> live;
    auto edges = plan.getAs<DictionaryAttr>("edges");
    auto count = contractInt(plan,"barrier_slots");
    if (!edges || edges.size()!=5 || count<=0 || count>32)
      return module.emitError("invalid ready/release barrier set");
    uint64_t occupied = 0;
    for (StringRef key : {"q","k","v","o"}) {
      auto buffer = buffers.getAs<DictionaryAttr>(key);
      int64_t n = contractInt(buffer,"versions"), bytes = contractInt(buffer,"tile_bytes");
      int64_t base = contractInt(buffer,"offset");
      if (bytes!=32768 || n<1 || n>8 || base<0 || base%1024 || base>arena-n*bytes ||
          ((key=="q" || key=="o") && n!=1))
        return module.emitError("invalid descriptor origin, tile size or ring extent");
      for (auto prior : live)
        if (base<prior.end && prior.begin<base+n*bytes &&
            !(key=="o" && prior.name=="q" && base==prior.begin))
          return module.emitError("execution contract overlaps live shared versions");
      live.push_back({key,base,base+n*bytes});
      if (key=="o") continue;
      for (bool release : {false,true}) {
        if (key=="q" && release) continue;
        auto edge=edges.getAs<DictionaryAttr>((key+(release?"_free":"_ready")).str());
        int64_t first=contractInt(edge,"first");
        if (contractInt(edge,"slots")!=n || first<0 || first>count-n ||
            contractInt(edge,"distance")!=(release?n:0) ||
            contractInt(edge,"arrivals")!=(release?widths[c]:1))
          return module.emitError("incompatible ready/release phase or participant count");
        uint64_t mask=((uint64_t(1)<<n)-1)<<first;
        if (occupied & mask) return module.emitError("overlapping barrier slots");
        occupied |= mask;
      }
    }
    if (occupied!=(uint64_t(1)<<count)-1) return module.emitError("unused barrier slots");
  }
  auto walked=module.walk([&](MmaOp mma) -> WalkResult {
    if (mma.getKAttr().getInt() % (16*batch)) {
      mma.emitError("MMA depth cannot retire complete execution-contract wait batches");
      return WalkResult::interrupt();
    }
    if (grouped && (mma.getMAttr().getInt()!=128 || mma.getNAttr().getInt()!=128 ||
                    mma.getKAttr().getInt()!=128)) {
      mma.emitError("MMA tile disagrees with FA3 buffer geometry");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

} // namespace

// Bind a logical holder's thread domain before constructing the layout graph.
LogicalResult mlir::frisk::bindExecutionGroups(ModuleOp module) {
  auto raw = module->getAttr("frisk.execution_contract");
  if (!raw) return success();
  auto plan = dyn_cast<DictionaryAttr>(raw);
  auto groups = plan ? plan.getAs<ArrayAttr>("groups") : ArrayAttr();
  auto cta = plan ? plan.getAs<IntegerAttr>("cta") : IntegerAttr();
  auto roles = plan ? plan.getAs<DictionaryAttr>("roles") : DictionaryAttr();
  if (!groups || groups.empty() || !cta || !roles)
    return module.emitError("malformed execution contract");
  SmallVector<int64_t> widths;
  int64_t next = 0;
  for (Attribute item : groups) {
    auto group = dyn_cast<DictionaryAttr>(item);
    auto base = group ? group.getAs<IntegerAttr>("base") : IntegerAttr();
    auto count = group ? group.getAs<IntegerAttr>("threads") : IntegerAttr();
    if (!base || !count || base.getInt() != next || count.getInt() <= 0 ||
        count.getInt() % 32 || count.getInt() > 1024 - next)
      return module.emitError("invalid CTA thread partition");
    widths.push_back(count.getInt());
    next += widths.back();
  }
  if (next != cta.getInt()) return module.emitError("CTA thread count mismatch");
  if (failed(verifyOnlineResources(module, plan, widths))) return failure();
  auto walked = module.walk([&](Operation *op) -> WalkResult {
    if (!isa<MmaOp, CopyOp, ReduceTensorOp>(op)) return WalkResult::advance();
    int64_t index = 0;
    if (auto role = op->getAttrOfType<StringAttr>("frisk.execution_role")) {
      auto group = roles.getAs<IntegerAttr>(role.getValue());
      if (!group || group.getInt() < 0 || (size_t)group.getInt() >= widths.size()) {
        op->emitError("undeclared execution role"); return WalkResult::interrupt();
      }
      index = group.getInt();
    } else if (widths.size() != 1) {
      op->emitError("multiple execution groups require a role"); return WalkResult::interrupt();
    }
    auto width = widths[index];
    if (auto old = op->getAttrOfType<IntegerAttr>("frisk.execution_threads"))
      if (old.getInt() != width) {
        op->emitError("thread count disagrees with execution group"); return WalkResult::interrupt();
      }
    op->setAttr("frisk.execution_threads", Builder(op->getContext()).getI64IntegerAttr(width));
    return WalkResult::advance();
  });
  return failure(walked.wasInterrupted());
}

