// Consumer-specific SM90 stores, selected only inside the local MMA optimum.
#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/Analysis/ExecutionGroups.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h"
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "MultiAnchorSolver.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "PreserveStorageContracts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"
#include <array>
#include <optional>
#include <set>

using namespace mlir;
using namespace mlir::frisk;

namespace {
using Cost = std::array<int64_t,9>;
using Policy = attr::GemmWarpPolicy;
MmaOp onlyMma(ModuleOp mod) {
  SmallVector<MmaOp> ops;
  mod.walk([&](MmaOp mma){ ops.push_back(mma); });
  return ops.size()==1 ? ops[0] : MmaOp();
}

// Retain storage contracts and exact constant/operand encodings. Computed
// tensor edges are typed function arguments at the instruction-local boundary.
FailureOr<OwningOpRef<ModuleOp>> projectAnchor(ModuleOp mod, MmaOp mma) {
  OpBuilder b(mod.getContext());
  OwningOpRef<ModuleOp> local(ModuleOp::create(mod.getLoc()));
  if (auto target=mod->getAttr("frisk.target")) (*local)->setAttr("frisk.target",target);
  if (auto plan = mod->getAttr("frisk.execution_contract"))
    (*local)->setAttr("frisk.execution_contract", plan);
  auto fn=func::FuncOp::create(mod.getLoc(),"anchor",b.getFunctionType({},{}));
  local->push_back(fn);
  b.setInsertionPointToStart(fn.addEntryBlock());
  IRMapping values;
  auto boundary = [&](Value v) {
    auto &entry = fn.getBody().front();
    values.map(v, entry.addArgument(v.getType(), mma.getLoc()));
    fn.setType(b.getFunctionType(entry.getArgumentTypes(), {}));
    return success();
  };
  std::function<LogicalResult(Value)> copy=[&](Value v)->LogicalResult {
    if (values.contains(v)) return success();
    if (auto arg=dyn_cast<BlockArgument>(v)) {
      auto loop=dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      if (!loop || arg.getArgNumber()==0 || v!=mma->getOperand(2)) return boundary(v);
      unsigned slot=arg.getArgNumber()-1;
      if (loop.getBody()->getTerminator()->getOperand(slot)!=mma.getResult()) return boundary(v);
      auto init=loop.getInitArgs()[slot];
      if (failed(copy(init))) return failure();
      values.map(arg,values.lookup(init));
      return success();
    }
    auto *def=v.getDefiningOp();
    // Tensor computation outside the instruction is a typed incoming edge.
    // Preserve any hard encoding; its selected use is rechecked on the full graph.
    if (!isa<LayoutViewOp,memref::AllocOp,arith::ConstantOp>(def))
      return isa<RankedTensorType>(v.getType()) ? boundary(v) : failure();
    for (auto operand:def->getOperands()) if (failed(copy(operand))) return failure();
    b.clone(*def,values);
    return success();
  };
  for (auto v:mma->getOperands()) if (failed(copy(v))) return failure();
  mod.walk([&](StorageContractOp contract) {
    if (values.contains(contract->getOperand(0))) b.clone(*contract,values);
  });
  b.clone(*mma,values);
  b.create<func::ReturnOp>(mma.getLoc());
  return std::move(local);
}

// The local certificate requires exact, unsaturated Frisk component solutions.
FailureOr<Cost> infer(ModuleOp mod, bool exact) {
  LayoutStatisticsReport stats;
  PassManager pm(mod.getContext());
  SolverOptions options;
  if (exact) {
    // Harvest the instruction before cleanup folds the unused probe result.
    pm.addPass(createNormalizeLayoutIRPass());
    pm.addPass(createFriskInferLayoutsPass(options,true,&stats));
  } else {
    buildFriskLayoutPipeline(pm,options,{},&stats);
  }
  if (failed(pm.run(mod)) || stats.attempts.empty()) return failure();
  Cost total{};
  for (const auto &entry:stats.attempts.back().components) {
    auto *component=entry.getAsObject();
    if (!component || (exact && !component->getBoolean("exhaustive").value_or(false))) return failure();
    auto *selected=component->getObject("selectedCost");
    if (!selected || selected->getBoolean("saturated").value_or(true)) return failure();
    auto *cost=selected->getArray("cost");
    if (!cost || cost->size()!=total.size()) return failure();
    for (unsigned i=0;i<total.size();++i) {
      auto value=(*cost)[i].getAsInteger();
      if (!value || *value<0 || *value>INT64_MAX-total[i]) return failure();
      total[i]+=*value;
    }
  }
  return total;
}

SmallVector<Attribute> binding(MmaOp op) {
  SmallVector<Attribute> attrs{op->getAttr("frisk.mma_contract")};
  for (Value value:op->getOperands()) {
    if (auto view=value.getDefiningOp<LayoutViewOp>()) attrs.push_back(view->getAttr("layout"));
    else if (auto tensor = dyn_cast<RankedTensorType>(value.getType()))
      attrs.push_back(tensor.getEncoding());
    else
      attrs.push_back(TypeAttr::get(value.getType()));
  }
  attrs.push_back(cast<RankedTensorType>(op.getResult().getType()).getEncoding());
  return attrs;
}

bool supportedStore(TileStoreOp op) {
  auto view=op.getTarget().getDefiningOp<LayoutViewOp>();
  auto layout=view ? view->getAttrOfType<StorageLayoutAttr>("layout") : StorageLayoutAttr();
  auto t=cast<MemRefType>(op.getTarget().getType());
  if (!layout || layout.getMemorySpace().getValue()!=attr::MemorySpace::Global ||
      !t.hasStaticShape() || !t.getLayout().isIdentity() || !t.getElementType().isF32() ||
      t.getRank()<1 || t.getRank()>2 || t.getNumElements()>65536 ||
      layout.getAlignment().getInt()<32 || layout.getVectorGranularity().getInt()!=1) return false;
  auto map=dyn_cast<AffineLayoutMapAttr>(layout.getMap());
  if (!map) return false;
  AffineMap address=map.getAffineMap().getValue();
  if (address.getNumSymbols() || address.getNumDims()!=unsigned(t.getRank()) ||
      address.getNumResults()!=2) return false;
  SmallVector<int64_t> point(t.getRank());
  for (int64_t linear=0;linear<t.getNumElements();++linear) {
    int64_t rest=linear;
    for (int i=t.getRank()-1;i>=0;--i) {
      point[i]=rest%t.getDimSize(i);
      rest/=t.getDimSize(i);
    }
    auto bits=address.compose(point);
    if (bits[0]!=linear*4 || bits[1]!=0) return false;
  }
  return true;
}

// GF(2) basis for scalar, consecutive addresses across participating threads.
// It is constructed from shape and topology, independently of other compilers.
DistributedEncodingAttr striped(RankedTensorType t, DistributedEncodingAttr src) {
  auto map=dyn_cast<BitLinearLayoutMapAttr>(src.getMap());
  if (!map) return {};
  auto topo=src.getTopology().asArrayRef();
  int64_t threads=topo[1]*topo[2]*topo[3];
  int64_t slots=t.getNumElements()/threads;
  if (topo[1]!=32 || topo[4]!=1 || !slots || t.getNumElements()%threads) return {};
  unsigned regs=llvm::Log2_64(slots),bits=llvm::Log2_64(t.getNumElements());
  if (!bits || !llvm::isPowerOf2_64(slots)) return {};
  Builder b(t.getContext());
  SmallVector<APInt> matrix(bits*bits,APInt(1,0));
  unsigned row=0,suffix=bits;
  for (auto width:map.getOutputBitWidths().asArrayRef()) {
    suffix-=width;
    for (int64_t k=0;k<width;++k,++row) {
      unsigned flat=suffix+k,threadBits=bits-regs;
      unsigned col=flat<threadBits ? regs+flat : flat-threadBits;
      matrix[row*bits+col]=APInt(1,1);
    }
  }
  SmallVector<Attribute> names;
  SmallVector<int64_t> widths;
  SmallVector<int64_t> topology(topo.begin(),topo.end()); topology[0]=slots;
  const char *axes[]={"register","lane","warp","warp_group","cta"};
  for (unsigned i=0;i<5;++i) if (topology[i]>1) {
    names.push_back(b.getStringAttr(axes[i])); widths.push_back(llvm::Log2_64(topology[i]));
  }
  auto dest=BitLinearLayoutMapAttr::get(t.getContext(),b.getArrayAttr(names),
    b.getDenseI64ArrayAttr(widths),map.getOutputNames(),map.getOutputBitWidths(),
    DenseIntElementsAttr::get(RankedTensorType::get({bits,bits},b.getI1Type()),matrix));
  return DistributedEncodingAttr::get(t.getContext(),dest,b.getDenseI64ArrayAttr(topology),b.getI64IntegerAttr(1));
}

// Register allocation is CTA-uniform. A predicated physical slot counts once
// in a stripe even when only a subset of lanes can reach it.
constexpr int64_t stripeElements = 16384;
constexpr int64_t sm90RegisterLimit = 255;
struct Traffic {
  int64_t stripeIssues = 0, sectors = 0, tailSectors = 0, groupStripes = 0;
  SmallVector<int64_t> touchedSlots;
};
FailureOr<Traffic> traffic(RankedTensorType t, DistributedEncodingAttr enc,
                           int64_t tailElements = 0) {
  LinearExecutionLayout m;
  if (analyzeLinearExecutionLayout(enc,t,m).status!=ProofStatus::Proven ||
      m.topology[1]!=32 || m.topology[4]!=1 || m.hardwareBits>18) return failure();
  unsigned slots=m.topology[0], threads=m.topology[1]*m.topology[2]*m.topology[3];
  SmallVector<int> first(t.getNumElements(),-1),address(threads*slots);
  for (unsigned i=0;i<address.size();++i) {
    int index=0;
    auto coord=m.logicalCoordinate(m.apply(APInt(m.hardwareBits,i)));
    for (auto [dim,v]:llvm::enumerate(coord)) index=index*t.getDimSize(dim)+v;
    if (index<0 || index>=t.getNumElements()) return failure();
    address[i]=index;
    if (first[index]==-1) first[index]=i;
  }
  if (llvm::is_contained(first,-1)) return failure();
  Traffic out;
  SmallVector<llvm::SmallBitVector> live(
      llvm::divideCeil(t.getNumElements(), stripeElements),
      llvm::SmallBitVector(slots));

  // Scoped to the native ownership map. A warpgroup that publishes any
  // first-owned element in a router stripe must run that publish phase.
  SmallVector<llvm::SmallBitVector> publishers(
      live.size(), llvm::SmallBitVector(m.topology[3]));
  for (unsigned w=0;w<threads;w+=32) for (unsigned r=0;r<slots;++r) {
    std::set<int> sectors, tailSectors, stripes;
    for (unsigned l=0;l<32;++l) {
      unsigned id=(w+l)*slots+r;
      if (first[address[id]]==int(id)) {
        sectors.insert(address[id]/8);
        if (address[id] < tailElements)
          tailSectors.insert(address[id]/8);
        unsigned phase = address[id] / stripeElements;
        stripes.insert(phase);
        live[phase].set(r);
        publishers[phase].set(w / (m.topology[1] * m.topology[2]));
      }
    }
    out.stripeIssues+=stripes.size();
    out.sectors+=sectors.size();
    out.tailSectors+=tailSectors.size();
  }
  for (const auto &groups : publishers)
    out.groupStripes += groups.count();
  for (const auto &phase : live)
    out.touchedSlots.push_back(phase.count());
  return out;
}

// This is a working-set proxy for the scalar shared router, not a count of
// allocated registers. The MMA accumulator sets a common pressure floor.
// A stripe that needs many predicated input slots alongside its output slots
// can add spill/reload pairs. Never use ptxas results or timings in the pass.
struct SpillEstimate {
  int64_t direct = 0, routed = 0, peakStripeSlots = 0;
};
SpillEstimate estimateSpill(const Traffic &source, const Traffic &writer,
                           int64_t sourceSlots, int64_t warps) {
  auto excess = [](int64_t slots) {
    return std::max<int64_t>(0, slots - sm90RegisterLimit);
  };
  SpillEstimate estimate;
  estimate.direct = 2 * warps * excess(sourceSlots);
  estimate.routed = estimate.direct;
  for (auto [inputs, outputs] : llvm::zip_equal(source.touchedSlots, writer.touchedSlots)) {
    int64_t workingSet = inputs + outputs;
    estimate.peakStripeSlots = std::max(estimate.peakStripeSlots, workingSet);
    estimate.routed += 2 * warps * excess(workingSet);
  }
  return estimate;
}

// Units are static service requests. Charge one per global sector, one per
// shared warp instruction, and one per participating warp per CTA barrier.
// The common lowering publishes and gathers in 64 KiB stripes.
// The launch covers independent row-major matrices, with one CTA per row tile.
// Only global stores are predicated on the last tile; the scalar shared route
// still publishes and gathers its entire tile. All rates are H100 PCIe Q10.
struct StoreSchedule {
  int64_t tiles, tailElements, globalRate, routeRate;
  // Relative service of the two final machine-code loops, per CTA, in Q10.
  DistributedEncodingAttr pairedSource;
  int64_t mainloopDeltaQ10 = 0;
  int64_t segmentDeltaQ10 = 0;
  bool hasSegments = false;
  bool writerPair = false;
  bool drainPair = false, guardedWave = false;
  int64_t gridCTAs = 0, smCount = 0;
  int64_t drainAddress = 0, drainInterstore = 0;
  int64_t tailMissSectors = 0;
  bool nativePair = false;
};

FailureOr<StoreSchedule> readStoreSchedule(Attribute attr, RankedTensorType t) {
  auto dict = dyn_cast<DictionaryAttr>(attr);
  if (!dict || t.getRank() != 2 || t.getDimSize(0) <= 0 || t.getDimSize(1) <= 0 ||
      t.getDimSize(1) % 8 != 0 ||
      dict.size() != (dict.get("paired_mainloop") ? 7u : 6u))
    return failure();
  auto integer = [&](StringRef key) -> int64_t {
    auto v = dict.getAs<IntegerAttr>(key);
    if (!v || !v.getType().isInteger(64) || v.getInt() < 0 || v.getInt() > (1LL << 30))
      return -1;
    return v.getInt();
  };
  int64_t rows = integer("rows"), instances = integer("instances");
  int64_t sms = integer("sm_count"), issues = integer("mma_issues");
  int64_t waits = integer("mma_waits");
  auto overlap = dict.getAs<BoolAttr>("copy_overlap");
  if (rows < 1 || instances < 1 || sms < 1 || issues < 0 ||
      waits < 0 || waits > issues || !overlap)
    return failure();

  int64_t height = t.getDimSize(0);
  int64_t tiles = llvm::divideCeil(rows, height);
  int64_t busy = std::min(sms, tiles * instances);
  // Static estimates of global-sector and exposed shared-router service.
  // Coefficients fit on paired schedule/store interventions, before holdout.
  int64_t globalRate = 867 + (266 * busy + sms / 2) / sms;
  int64_t waitRate = issues ? (3 * waits + issues / 2) / issues : 0;
  int64_t routeRate = 1024 + 555 * int64_t(overlap.getValue()) + waitRate;
  StoreSchedule schedule{tiles, rows % height * t.getDimSize(1), globalRate, routeRate, {}};
  schedule.gridCTAs = tiles * instances;
  schedule.smCount = sms;
  if (Attribute pair = dict.get("paired_mainloop")) {
    auto fields = dyn_cast<DictionaryAttr>(pair);
    if (!fields) return failure();
    auto backend = fields.getAs<StringAttr>("backend");
    schedule.nativePair = backend && backend.getValue() == "native";
    bool memoryPair = schedule.nativePair && fields.get("warp_load_wait_delta");
    schedule.hasSegments = memoryPair && fields.get("cache_miss_sectors_delta");
    schedule.writerPair = schedule.hasSegments && fields.get("tail_cache_miss_sectors_delta");
    schedule.drainPair = schedule.writerPair && fields.get("drain_address_issues_delta");
    if (Attribute wave = fields.get("guarded_wave")) {
      auto enabled = dyn_cast<BoolAttr>(wave);
      if (!schedule.drainPair || !enabled || !enabled.getValue() || !schedule.tailElements)
        return failure();
      schedule.guardedWave = true;
    }
    if (fields.size() != (schedule.nativePair ? 4u : 3u) + (memoryPair ? 2u : 0u) +
                             (schedule.hasSegments ? 2u : 0u) + (schedule.writerPair ? 1u : 0u) + (schedule.drainPair ? 2u : 0u) + (schedule.guardedWave ? 1u : 0u))
      return failure();
    schedule.pairedSource = fields.getAs<DistributedEncodingAttr>("source");
    auto work = fields.getAs<IntegerAttr>("warp_issues_delta");
    auto madd = fields.getAs<IntegerAttr>("warp_imads_delta");
    auto valid = [](IntegerAttr v) {
      return v && v.getType().isInteger(64) &&
             v.getInt() >= -(1LL << 30) && v.getInt() <= (1LL << 30);
    };
    if (!schedule.pairedSource || !valid(work) || !valid(madd)) return failure();
    // Only the portion of the IMAD change that can belong to net added
    // (or removed) issues earns the incremental rate. Opcode substitutions
    // within the same issue budget have no established signed service cost.
    int64_t issuesDelta = work.getInt();
    int64_t additionalImads = std::clamp(madd.getInt(), std::min<int64_t>(0, issuesDelta),
                                         std::max<int64_t>(0, issuesDelta));
    schedule.mainloopDeltaQ10 = 43 * issuesDelta + 1254 * additionalImads;
    if (memoryPair) {
      auto wait = fields.getAs<IntegerAttr>("warp_load_wait_delta");
      auto local = fields.getAs<IntegerAttr>("warp_local_word_delta");
      if (!valid(wait) || !valid(local)) return failure();
      // Exposed RAW slots share the ordinary issue rate. A local load or
      // store adds service for each 32-bit word beyond its issue slot.
      schedule.mainloopDeltaQ10 += 43 * wait.getInt() + 1024 * local.getInt();
      if (schedule.hasSegments) {
        auto misses = fields.getAs<IntegerAttr>("cache_miss_sectors_delta");
        auto address = fields.getAs<IntegerAttr>("tail_address_issues_delta");
        if (!valid(misses) || !valid(address)) return failure();
        // One downstream sector request or additional scalar issue per unit.
        if (schedule.writerPair) {
          auto tail = fields.getAs<IntegerAttr>("tail_cache_miss_sectors_delta");
          if (!valid(tail)) return failure();
          schedule.tailMissSectors = tail.getInt();
          if (schedule.drainPair) {
            auto addr = fields.getAs<IntegerAttr>("drain_address_issues_delta");
            auto gap = fields.getAs<IntegerAttr>("drain_interstore_issues_delta");
            if (!valid(addr) || !valid(gap)) return failure();
            schedule.drainAddress = addr.getInt();
            schedule.drainInterstore = gap.getInt();
          }
        }
        schedule.segmentDeltaQ10 = 43 * (misses.getInt() + address.getInt());
      }
    }
  }
  return schedule;
}

int64_t expectedSectorsQ10(const StoreSchedule &s, const Traffic &traffic) {
  int64_t full = s.tiles - (s.tailElements != 0);
  return ((full * traffic.sectors + traffic.tailSectors) * 1024 + s.tiles / 2) / s.tiles;
}

// Keep the per-use decision independent from the deduplicated physical bill.
// Register, Shuffle, Shared costs are warp service estimates, not SASS counts.
struct StoreRoute {
  StringRef kind = "Identity";
  int64_t reg = 0, shuffle = 0, shared = 0, barriers = 0, scratch = 0;
  int64_t publishExtra = 0, gatherExtra = 0;
  bool feasible = true;
  int64_t service() const { return reg + shuffle + shared + barriers; }
};

// One aligned f32 scratch slot is one bank word. Warp loads of the same
// word broadcast. A nonempty request already pays for its first wavefront.
int64_t extraBankWaves(SmallVectorImpl<uint64_t> &slots) {
  llvm::sort(slots);
  const auto end = std::unique(slots.begin(), slots.end());
  std::array<unsigned, 32> counts{};
  unsigned waves = 0;
  for (auto it = slots.begin(); it != end; ++it)
    waves = std::max(waves, ++counts[*it % counts.size()]);
  return waves ? waves - 1 : 0;
}

void countWriterBanks(const test::RedistributionPlan &plan, StoreRoute &route) {
  for (unsigned warp = 0; warp < plan.threads / 32; ++warp) {
    for (unsigned reg = 0; reg < plan.sourceRegisters; ++reg) {
      SmallVector<uint64_t, 32> slots;
      for (unsigned lane = 0; lane < 32; ++lane) {
        unsigned thread = warp * 32 + lane;
        uint64_t slot = plan.sourceSlots[thread * plan.sourceRegisters + reg];
        if (plan.globalWriters[slot] == plan.owner(thread, reg))
          slots.push_back(slot);
      }
      route.publishExtra += extraBankWaves(slots);
    }
    for (unsigned reg = 0; reg < plan.destinationRegisters; ++reg) {
      SmallVector<uint64_t, 32> slots;
      for (unsigned lane = 0; lane < 32; ++lane)
        slots.push_back(plan.destinations[(warp * 32 + lane) * plan.destinationRegisters + reg].slot);
      route.gatherExtra += extraBankWaves(slots);
    }
  }
}

StoreRoute planStoreRoute(RankedTensorType src, RankedTensorType dst,
                          const Traffic &input, const Traffic &writer,
                          Location loc, bool stripedReplay, bool countBanks, bool allowStripes = false) {
  StoreRoute route;
  if (src == dst) return route;
  auto topology = cast<DistributedEncodingAttr>(src.getEncoding()).getTopology();
  int64_t warps = topology[2] * topology[3];
  if (stripedReplay) {
    // The scalar replay kernel explicitly stages every non-identity mapping
    // through 64 KiB stripes, including maps the native emitter keeps local.
    route.kind = "Shared";
    route.shared = input.stripeIssues + writer.stripeIssues;
    route.barriers = 2 * warps * llvm::divideCeil(src.getNumElements(), stripeElements);
    route.scratch = 4 * std::min(src.getNumElements(), stripeElements);
    return route;
  }
  // An unsupported optional exchange never invalidates a legal in-place store.
  ScopedDiagnosticHandler optionalCandidate(loc.getContext(),
                                            [](Diagnostic &) { return success(); });
  auto plan = test::planExecutionRedistribution(src, dst, loc, shuffle_network::Objective::MinShuffle, allowStripes);
  if (failed(plan)) {
    route.kind = "Unavailable";
    route.feasible = false;
    return route;
  }
  route.scratch = plan->scratchBytes;
  switch (plan->mode) {
  case test::ExchangeMode::Shared: {
    route.kind = "Shared";
    // Native scalar lowering publishes elected writers, gathers every
    // destination carrier and brackets scratch reuse with two CTA barriers.
    if (plan->stripeElements) {
      // Count only warp/register issues whose elected addresses intersect this
      // stripe. An issue spanning two stripes is issued twice by the emitter.
      uint64_t phases = llvm::divideCeil(plan->volume, plan->stripeElements);
      size_t publishWidth = warps * plan->sourceRegisters;
      size_t gatherWidth = warps * plan->destinationRegisters;
      llvm::SmallBitVector publish(phases * publishWidth), gather(phases * gatherWidth);
      for (auto [slot, owner] : llvm::enumerate(plan->globalWriters)) {
        auto warp = plan->thread(owner) / 32;
        publish.set((slot / plan->stripeElements) * publishWidth +
                    warp * plan->sourceRegisters + owner.reg);
      }
      for (auto [index, dest] : llvm::enumerate(plan->destinations)) {
        auto warp = index / plan->destinationRegisters / 32;
        gather.set((dest.slot / plan->stripeElements) * gatherWidth +
                   warp * plan->destinationRegisters + index % plan->destinationRegisters);
      }
      route.shared = publish.count() + gather.count();
      route.barriers = 2 * warps * phases;
      return route;
    }
    llvm::SmallBitVector writes(warps * plan->sourceRegisters);
    for (auto owner : plan->globalWriters)
      writes.set((plan->thread(owner) / 32) * plan->sourceRegisters + owner.reg);
    route.shared = writes.count() + warps * plan->destinationRegisters;
    route.barriers = 2 * warps;
    if (countBanks) countWriterBanks(*plan, route);
    // Per-CTA ceiling of the SM90a executor. Remaining shared allocations
    // and their lifetimes are verified by the final allocation pass.
    route.feasible = route.scratch <= 227328;
    return route;
  }
  case test::ExchangeMode::Shuffle:
    route.kind = "Shuffle";
    if (plan->shuffleNetwork) {
      route.shuffle = plan->shuffleNetwork->counts.shuffleWarpIssues;
      route.reg = plan->shuffleNetwork->counts.addressAluWarpIssues;
      return route;
    }
    break;
  case test::ExchangeMode::Register:
    route.kind = "Register";
    break;
  }
  // A fixed register rename is free. A register that varies by thread needs
  // a mux. The fallback Shuffle emitter fetches every supported source word.
  for (unsigned r = 0; r < plan->destinationRegisters; ++r) {
    llvm::SmallBitVector support(plan->sourceRegisters);
    for (unsigned t = 0; t < plan->threads; ++t)
      support.set(plan->destinations[t * plan->destinationRegisters + r].source.reg);
    route.reg += warps * (support.count() - 1);
    if (plan->mode == test::ExchangeMode::Shuffle)
      route.shuffle += warps * support.count() * plan->payload.words.size();
  }
  return route;
}

constexpr StringLiteral routeMarker = "frisk.protected_store_route";
constexpr StringLiteral useMarker = "frisk.protected_store_use";

// Old open-mode files have no ownership marker on the conversion. Recognize
// the previous writer's positive receipt only when its two types match the
// exact transformation it generated. Never peel an arbitrary explicit cast.
enum class OldRoute { External, Generated, Ambiguous };

OldRoute classifyLegacyRoute(TileStoreOp store, ConvertLayoutOp route, ModuleOp mod) {
  if (store->hasAttr("frisk.store_route_kind")) return OldRoute::External;
  auto receipt = store->getAttrOfType<DenseI64ArrayAttr>("frisk.store_score");
  if (!receipt || receipt.size() != 4) return OldRoute::External;
  if (route->hasAttr(routeMarker)) return OldRoute::Generated;
  if (!mod->hasAttr("frisk.protected_candidates")) return OldRoute::External;
  auto type = cast<RankedTensorType>(route.getSource().getType());
  auto src = dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding());
  auto dst = cast<RankedTensorType>(route.getResult().getType()).getEncoding();
  if (!src || !dst || src == dst || dst != striped(type, src)) return OldRoute::External;
  if (receipt[3] == 1) return OldRoute::Generated;
  // The old fixed pass could overwrite the last positive receipt on an
  // untagged open route. The remaining IR is also consistent with an
  // explicitly supplied conversion. Neither interpretation is provable.
  if (mod->hasAttr("frisk.fixed_store_cost") && receipt[3] == 0 &&
      receipt[2] == 0 && receipt[0] == receipt[1])
    return OldRoute::Ambiguous;
  return OldRoute::External;
}

LogicalResult restoreStoreInputs(ModuleOp mod) {
  llvm::SetVector<Operation *> bypassed;
  auto result = mod.walk([&](TileStoreOp store) -> WalkResult {
    auto route = store->getOperand(0).getDefiningOp<ConvertLayoutOp>();
    auto legacy = route ? classifyLegacyRoute(store, route, mod) : OldRoute::External;
    if (legacy == OldRoute::Ambiguous) {
      store.emitOpError("legacy store origin is ambiguous after an untagged route was rescored; "
                        "rescore the materialized input or the original open-pass output");
      return WalkResult::interrupt();
    }
    if (route && ((store->hasAttr(useMarker) && route->hasAttr(routeMarker)) ||
                  legacy == OldRoute::Generated)) {
      store->setOperand(0, route.getSource());
      bypassed.insert(route);
    }
    // Receipts describe the old decision even if this consumer has since
    // become unsupported. Remove them before collecting the new receipts.
    for (StringRef attr : {"frisk.store_score", "frisk.scheduled_store_score",
                           "frisk.mainloop_delta_q10", "frisk.segment_delta_q10", "frisk.writer_delta_q10", "frisk.drain_delta_q10", "frisk.wave_delta_q10", "frisk.wave_sectors",
                           "frisk.writer_bank_extra", "frisk.spill_pressure", "frisk.store_route_kind",
                           "frisk.store_route_service", "frisk.store_route_feasible",
                           "frisk.store_route_backend", "frisk.store_route_charge_q10"})
      store->removeAttr(attr);
    store->removeAttr(useMarker);
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  for (Operation *op : bypassed)
    if (op->use_empty()) op->erase();
  return success();
}

struct StoreTieBreak {
  int64_t peakSlots = 0, exposedPeakSlots = 0, publisherGroupStripes = 0;
};

// A group can reuse only an existing SSA definition dominating its use. No
// exchange is speculated into a branch or hoisted out of its loop.
FailureOr<int64_t> chooseStores(ModuleOp mod, bool scorePressure,
                                StoreTieBreak &tie, bool stripedReplay, bool fixedAnchors = false,
                                StringRef materialize = "auto") {
  if (failed(restoreStoreInputs(mod))) return failure();
  unsigned stores = 0;
  mod.walk([&](TileStoreOp) { ++stores; });
  if (materialize != "auto" && stores != 1) {
    mod.emitError("materialize-store requires exactly one terminal store");
    return failure();
  }
  DominanceInfo dominance(mod);
  DenseMap<std::pair<Value, Type>, SmallVector<Value>> routes;
  DenseSet<Value> billed;
  mod.walk([&](ConvertLayoutOp op) {
    if (op->hasAttr(routeMarker))
      routes[{op.getSource(), op.getResult().getType()}].push_back(op.getResult());
  });

  // Weight the store and its exchange at their actual execution sites. Keep
  // Q10 until the graph total is rounded; two fractional edges may add up.
  APInt total(128, 0), ceiling = APInt(128, INT64_MAX).shl(10);
  auto charge = [&](int64_t costQ10, Operation *site) {
    auto freq = getLayoutBoundaryFrequency(site, LayoutBoundaryFrequencyMode::StaticTripCount);
    total = llvm::APIntOps::umin(ceiling, total + APInt(128, costQ10) * APInt(128, freq.factor));
  };
  auto result = mod.walk<WalkOrder::PreOrder>([&](TileStoreOp op) -> WalkResult {
    Attribute context = op->getAttr("frisk.h100_store_schedule");
    if (!supportedStore(op)) {
      if (context || materialize != "auto") {
        op.emitOpError("scheduled scoring requires a supported scalar global f32 store");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    auto value = op->getOperand(0);
    auto t = cast<RankedTensorType>(value.getType());
    auto src = dyn_cast_or_null<DistributedEncodingAttr>(t.getEncoding());
    if (!src) return WalkResult::interrupt();
    std::optional<StoreSchedule> schedule;
    if (context) {
      auto parsed = readStoreSchedule(context, t);
      if (failed(parsed)) {
        op.emitOpError("invalid frisk.h100_store_schedule or unsupported row geometry");
        return WalkResult::interrupt();
      }
      schedule = *parsed;
    }
    auto dst = striped(t, src);
    if (!dst) {
      if (schedule || materialize != "auto") {
        op.emitOpError("scheduled scoring requires a supported distributed store layout");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    int64_t tail = schedule ? schedule->tailElements : 0;
    auto a = traffic(t, src, tail), b = traffic(t, dst, tail);
    if (failed(a) || failed(b)) return WalkResult::interrupt();
    auto type = RankedTensorType::get(t.getShape(), t.getElementType(), dst);
    auto route = planStoreRoute(t, type, *a, *b, op.getLoc(), stripedReplay, schedule && schedule->writerPair);
    auto topo = src.getTopology().asArrayRef();
    int64_t warps = topo[2] * topo[3];
    auto pressure = estimateSpill(*a, *b, topo[0], warps);
    if (!stripedReplay || src == dst) {
      // The calibrated stripe overlap surcharge applies specifically to the
      // replay implementation. Native routes retain the accumulator floor.
      pressure.routed = pressure.direct;
      pressure.peakStripeSlots = std::max(topo[0], dst.getTopology()[0]);
    }
    int64_t directSectors = schedule ? expectedSectorsQ10(*schedule, *a) : 1024 * a->sectors;
    int64_t writerSectors = schedule ? expectedSectorsQ10(*schedule, *b) : 1024 * b->sectors;
    int64_t globalRate = schedule ? schedule->globalRate : 1024;
    int64_t routeRate = schedule && route.kind == "Shared" ? schedule->routeRate : 1024;
    auto storeCost = [&](int64_t sectors) {
      return (globalRate * sectors + 512) / 1024 + (scorePressure ? 1024 * pressure.direct : 0);
    };
    int64_t direct = storeCost(directSectors), writer = storeCost(writerSectors);
    int64_t exchange = routeRate * route.service();
    if (scorePressure)
      exchange += 1024 * (3 * (pressure.routed - pressure.direct) / 4);
    // Only incremental serialization is added to the native paired score.
    // Keep physical base traffic and the fixed gate identical to prior policies.
    // H100 paired bank permutation and global-sector permutation: marginal
    // shared serialization costs 29/16 of one global sector on the calibration.
    int64_t extraWaves = route.publishExtra + route.gatherExtra;
    int64_t bankCost = (29 * globalRate * extraWaves + 8) / 16;
    int64_t writerDelta = schedule && schedule->writerPair
        ? bankCost + 43 * schedule->tailMissSectors
        : 0;
    // Masked scalar epilogue on H100. Price net address work over the whole
    // writer; moving an instruction across the first store costs nothing here.
    // For exposed store service require both excess sectors and excess issue
    // slots between stores. Each caps the other. Subtract existing base rates.
    constexpr int64_t guardedSectorRate = 3794;
    int64_t interactionDelta = 0;
    int64_t drainDelta = 0;
    if (schedule && schedule->drainPair) {
      int64_t sectors = writerSectors - directSectors;
      int64_t gap = schedule->drainInterstore;
      drainDelta = (1408 - 43) * schedule->drainAddress;
      if ((gap > 0 && sectors > 0) || (gap < 0 && sectors < 0)) {
        int64_t marginal = std::max<int64_t>(0, guardedSectorRate - globalRate);
        int64_t exposed = std::min<int64_t>(514 * std::abs(gap),
                                           (marginal * std::abs(sectors) + 512) / 1024);
        interactionDelta = gap > 0 ? exposed : -exposed;
        drainDelta += interactionDelta;
      }
    }
    // A sparse grid can finish at the pace of its most expensive CTA. The
    // throughput estimate already accounts for grid-average sectors and the
    // signed store interaction. Add only the uncovered differential service.
    int64_t waveDelta = 0;
    if (schedule && schedule->guardedWave) {
      int64_t edge = b->tailSectors - a->tailSectors;
      int64_t full = schedule->tiles > 1 ? b->sectors - a->sectors : 0;
      if (full == 0 || edge == 0 || (full > 0) == (edge > 0)) {
        int64_t peak = std::max(std::abs(full), std::abs(edge));
        int64_t grid = schedule->gridCTAs, busy = std::min(grid, schedule->smCount);
        int64_t critical = (guardedSectorRate * peak * busy + grid / 2) / grid;
        int64_t existing = storeCost(writerSectors) - storeCost(directSectors) + interactionDelta;
        int64_t increase = std::max<int64_t>(0, critical - std::abs(existing));
        waveDelta = (full < 0 || edge < 0) ? -increase : increase;
      }
    }
    int64_t loopDelta = schedule ? schedule->mainloopDeltaQ10 + schedule->segmentDeltaQ10 + writerDelta + drainDelta + waveDelta : 0;
    if (schedule && schedule->pairedSource) {
      // A pair measures one whole kernel. It cannot be reused for another
      // anchor or counted once per consumer of a different multi-store graph.
      if (!fixedAnchors || schedule->nativePair == stripedReplay || stores != 1 ||
          (schedule->nativePair && !mod->hasAttr("frisk.online_launch")) || src != schedule->pairedSource ||
          route.kind != "Shared" || !route.feasible) {
        op.emitOpError("paired main-loop scoring requires the matching fixed source and one shared store from the same lowering");
        return WalkResult::interrupt();
      }
      // Subtract the cheaper body as a common constant. Both remaining bills
      // are nonnegative, and the physical exchange accounting stays separate.
      direct += std::max<int64_t>(0, -loopDelta);
      writer += std::max<int64_t>(0, loopDelta);
    }
    bool change = route.feasible && src != dst && writer + exchange < direct;
    if (materialize != "auto") {
      if (stores != 1 || !fixedAnchors || stripedReplay || !route.feasible ||
          src == dst || (schedule && schedule->pairedSource)) {
        op.emitOpError("forced native candidates require a single feasible frozen store without feedback");
        return WalkResult::interrupt();
      }
      change = materialize == "striped";
    }
    OpBuilder builder(op);
    if (schedule && schedule->pairedSource)
      op->setAttr("frisk.mainloop_delta_q10", builder.getI64IntegerAttr(schedule->mainloopDeltaQ10));
    if (schedule && schedule->writerPair) {
      op->setAttr("frisk.writer_bank_extra", builder.getDenseI64ArrayAttr({route.publishExtra, route.gatherExtra}));
      op->setAttr("frisk.writer_delta_q10", builder.getI64IntegerAttr(writerDelta));
    }
    if (schedule && schedule->guardedWave) {
      op->setAttr("frisk.wave_delta_q10", builder.getI64IntegerAttr(waveDelta));
      op->setAttr("frisk.wave_sectors", builder.getDenseI64ArrayAttr(
          {a->sectors, b->sectors, a->tailSectors, b->tailSectors, schedule->gridCTAs, schedule->smCount}));
    }
    if (schedule && schedule->drainPair)
      op->setAttr("frisk.drain_delta_q10", builder.getI64IntegerAttr(drainDelta));
    if (schedule && schedule->hasSegments)
      op->setAttr("frisk.segment_delta_q10", builder.getI64IntegerAttr(schedule->segmentDeltaQ10));
    if (schedule)
      op->setAttr("frisk.scheduled_store_score", builder.getDenseI64ArrayAttr(
          {directSectors, writerSectors, globalRate, routeRate, direct, writer + exchange}));
    else
      op->removeAttr("frisk.scheduled_store_score");
    op->setAttr("frisk.store_score", builder.getDenseI64ArrayAttr(
        {a->sectors, b->sectors, route.service(), change ? 1 : 0}));
    op->setAttr("frisk.spill_pressure", builder.getDenseI64ArrayAttr(
        {topo[0], pressure.peakStripeSlots, pressure.direct, pressure.routed}));
    op->setAttr("frisk.store_route_backend", builder.getStringAttr(
        stripedReplay ? "scalar-striped-replay" : "native"));
    op->setAttr("frisk.store_route_kind", builder.getStringAttr(route.kind));
    op->setAttr("frisk.store_route_service", builder.getDenseI64ArrayAttr(
        {route.reg, route.shuffle, route.shared, route.barriers, route.scratch}));
    op->setAttr("frisk.store_route_feasible", builder.getBoolAttr(route.feasible));

    int64_t paid = 0;
    if (change) {
      auto &group = routes[{value, type}];
      Value converted;
      for (Value existing : group)
        if (dominance.dominates(existing, op)) {
          converted = existing;
          break;
        }
      if (!converted) {
        converted = builder.create<ConvertLayoutOp>(op.getLoc(), type, value);
        converted.getDefiningOp()->setAttr(routeMarker, builder.getUnitAttr());
        group.push_back(converted);
      }
      op->setOperand(0, converted);
      op->setAttr(useMarker, builder.getUnitAttr());
      if (billed.insert(converted).second) {
        paid = exchange;
        if (scorePressure && stripedReplay)
          tie.publisherGroupStripes += a->groupStripes;
        charge(exchange, converted.getDefiningOp());
      }
    }
    op->setAttr("frisk.store_route_charge_q10", builder.getI64IntegerAttr(paid));
    charge(change ? writer : direct, op);
    if (scorePressure) {
      int64_t peak = change ? pressure.peakStripeSlots : topo[0];
      tie.peakSlots = std::max(tie.peakSlots, peak);
      if (change && pressure.direct > 0)
        tie.exposedPeakSlots = std::max(tie.exposedPeakSlots, peak);
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  return static_cast<int64_t>((total + 512).lshr(10).getZExtValue());
}

struct Candidate {
  SM90MmaGeometry grid;
  Policy pinned;
  Cost local;
  SmallVector<Attribute> anchor;
};

SmallVector<MmaOp> anchorsIn(Operation *scope) {
  SmallVector<MmaOp> anchors;
  scope->walk([&](MmaOp op) { anchors.push_back(op); });
  return anchors;
}

StringAttr partition(Builder &b, const Candidate &c) {
  return b.getStringAttr(c.grid.interleavedRows ? "interleaved64" : "contiguous");
}

FailureOr<SmallVector<Candidate, 2>> certify(ModuleOp module, MmaOp mma) {
  Builder b(module.getContext());
  auto threads = mma->getAttrOfType<IntegerAttr>("frisk.execution_threads");
  if (!threads)
    if (auto parallel = mma->getParentOfType<ParallelOp>())
      threads = b.getI64IntegerAttr(parallel.getThreads());
  if (!threads || (threads.getInt() != 128 && threads.getInt() != 256))
    return mma.emitOpError("protection requires one or two explicit warpgroups");
  auto grids = getSM90MmaGeometries(mma, threads.getInt(), true);
  if (failed(grids) || grids->empty())
    return mma.emitOpError(diagnoseSM90MmaGeometryFailure(mma,threads.getInt()).reason);
  auto isolated = projectAnchor(module, mma);
  if (failed(isolated))
    return mma.emitOpError("cannot project the MMA operand boundary");
  SmallVector<Candidate, 2> candidates;
  for (auto g : *grids) {
    auto policy = mma.getPolicy() == Policy::Auto ?
        (g.gM == 1 ? Policy::FullCol : Policy::FullRow) : mma.getPolicy();
    OwningOpRef<ModuleOp> local(cast<ModuleOp>((*isolated)->clone()));
    auto anchor = onlyMma(*local);
    anchor->setAttr("frisk.execution_threads", threads);
    anchor->setAttr("frisk.row_partition", b.getStringAttr(
        g.interleavedRows ? "interleaved64" : "contiguous"));
    anchor.setPolicyAttr(GemmWarpPolicyAttr::get(b.getContext(), policy));
    auto cost = infer(*local, true);
    if (failed(cost)) return mma.emitOpError("could not certify an exhaustive local anchor optimum");
    anchor = onlyMma(*local);
    if (!anchor) return failure();
    auto tuple = binding(anchor);
    auto observed = getSM90MmaContractGeometry(anchor, threads.getInt(), tuple.front());
    if (failed(observed) || observed->gM != g.gM || observed->gN != g.gN ||
        observed->interleavedRows != g.interleavedRows)
      return mma.emitOpError("pinned geometry disagrees with the local proof");
    candidates.push_back({g, policy, *cost, std::move(tuple)});
  }
  return candidates;
}

bool hasReduction(Operation *scope) {
  return scope->walk([](ReduceTensorOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}


bool hasPointwiseWork(Operation *scope) {
  return scope->walk([](Operation *op) {
    if (isa<TileLoadOp, TileStoreOp>(op))
      return WalkResult::interrupt();
    // A standalone tensor function may have no memory access to seed.
    // The distributed collector already models same-shape arith/math ops.
    StringRef dialect = op->getName().getDialectNamespace();
    bool elementwise = (dialect == "arith" || dialect == "math") &&
        op->hasTrait<OpTrait::Elementwise>() && op->getNumResults() &&
        isa<RankedTensorType>(op->getResult(0).getType());
    return elementwise ? WalkResult::interrupt() : WalkResult::advance();
  }).wasInterrupted();
}

// A blocked, nonreplicated FP32 parent with a unique contiguous register
// axis. Non-separable, replicated and warp-local conversions use their
// existing route price. The native executor has a different scratch map.
struct BlockedScratchTile {
  unsigned minor, vectorWords;
  std::array<unsigned, 2> extent;
  SmallVector<int> warpOwner;
};

std::optional<BlockedScratchTile> blockedScratchTile(RankedTensorType type) {
  auto enc = dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding());
  LinearExecutionLayout l;
  if (!enc || type.getRank() != 2 || !type.getElementType().isF32() ||
      analyzeLinearExecutionLayout(enc, type, l).status != ProofStatus::Proven ||
      l.logicalBits != l.hardwareBits || l.hardwareBits > 16 ||
      enc.getReplication().getInt() != 1 || l.topology[1] != 32 ||
      l.topology[3] != 1 || l.topology[4] != 1)
    return std::nullopt;
  BlockedScratchTile tile{0, 0, {1, 1}, {}};
  llvm::SmallBitVector used(l.hardwareBits);
  unsigned start = 0;
  for (unsigned axis = 0; axis < 2; ++axis) {
    unsigned inner = 0, span = 0;
    for (unsigned b = 0; b < l.logicalWidths[axis]; ++b) {
      const auto &row = l.rows[start+b];
      if (!row.isPowerOf2()) return std::nullopt;
      unsigned col = row.countr_zero();
      if (used.test(col)) return std::nullopt;
      used.set(col);
      if (b == inner && col == inner && col < l.hardwareWidths[0]) ++inner;
      if (col >= l.hardwareOffsets[1]) span = b+1;
      tile.warpOwner.push_back(col >= l.hardwareOffsets[2] ?
          int(col-l.hardwareOffsets[2]) : -1);
    }
    if (!span) return std::nullopt;
    tile.extent[axis] = 1u << span;
    if (inner) {
      tile.minor = axis;
      tile.vectorWords = std::min(4u, 1u << inner);
    }
    start += l.logicalWidths[axis];
  }
  if (!tile.vectorWords) return std::nullopt;
  return tile;
}

std::optional<int64_t> gluonScratchService(RankedTensorType from,
                                           RankedTensorType to, bool bankFloor) {
  auto in = blockedScratchTile(from), out = blockedScratchTile(to);
  if (!in || !out || in->minor == out->minor || in->warpOwner == out->warpOwner)
    return std::nullopt;
  auto enc = cast<DistributedEncodingAttr>(from.getEncoding());
  if (enc.getTopology() != cast<DistributedEncodingAttr>(to.getEncoding()).getTopology())
    return std::nullopt;
  // Gluon 3.4 uses destination order, one scalar per publishing lane
  // for a change of minor axis, and pads the scratch pitch by the larger
  // shared vector width. A destination with minor dim0 uses scalar gathers.
  const unsigned minor = out->minor;
  auto span = [&](unsigned d) { return std::max(in->extent[d], out->extent[d]); };
  const unsigned fast = span(minor), slow = span(1-minor);
  const unsigned pitch = fast + (minor == 1 ? out->vectorWords : 1);
  if (uint64_t(pitch) * slow * sizeof(float) > 227328) return std::nullopt;
  ExecutionEnumeration owners;
  if (enumerateExecutionLayout(enc, from, owners).status != ProofStatus::Proven)
    return std::nullopt;
  const unsigned regs = owners.topology[0], warps = owners.topology[2];
  const int64_t issues = int64_t(regs) * warps;
  int64_t publish = 0;
  if (bankFloor) {
    const unsigned cols = from.getDimSize(1);
    for (unsigned w = 0; w < warps; ++w)
      for (unsigned reg = 0; reg < regs; ++reg) {
        std::array<unsigned, 32> banks{};
        for (unsigned lane = 0; lane < 32; ++lane) {
          uint64_t flat = owners.logical[(w*32+lane)*regs+reg];
          std::array<uint64_t, 2> coord{flat/cols, flat%cols};
          uint64_t word = (coord[1-minor] % slow)*pitch + (coord[minor] % fast);
          ++banks[word % 32];
        }
        publish += *std::max_element(banks.begin(), banks.end());
      }
  }
  // Keep the existing issue estimate as a floor. The store side alone
  // needs at least 'publish' bank wavefronts. Charging the maximum avoids
  // adding both the original publish estimate and the same work again.
  return std::max(2*issues, publish) + 2*warps;
}

// A single integer problem spans all producers and uses in one function.
LogicalResult selectIntegerAnchors(ModuleOp module, unsigned freeBits,
                                   unsigned timeout, unsigned oracleLimit, bool scoreReduction, bool allowStripes,
                                   bool replayScratch, bool bankFloor, StringRef globalAccess, bool subgroupCandidates) {
  PassManager prepare(module.getContext());
  prepare.addPass(createNormalizeLayoutIRPass());
  if (failed(prepare.run(module)))
    return failure();
  auto anchors = anchorsIn(module);
  if (anchors.empty() && !hasReduction(module) && !hasPointwiseWork(module))
    return module.emitError("multi-anchor solver requires an MMA, reduction, tile access or tensor elementwise operation");
  Builder b(module.getContext());
  SmallVector<SmallVector<Candidate, 2>, 2> domains;
  SmallVector<Attribute> proofs;
  CertifiedInstructionDomains certificates;
  StringRef marker = "__frisk_integer_anchor_id";
  SmallVector<Attribute> originalIDs;
  for (auto [id, op] : llvm::enumerate(anchors)) {
    auto choices = certify(module, op);
    if (failed(choices))
      return failure();
    auto &domain = domains.emplace_back(std::move(*choices));
    auto minimum =
        llvm::min_element(domain, [](const Candidate &a, const Candidate &b) {
          return a.local < b.local;
        })->local;
    SmallVector<Attribute> rows;
    for (const auto &c : domain) {
      NamedAttrList row;
      row.set("eligible", b.getBoolAttr(c.local == minimum));
      row.set("grid", b.getDenseI64ArrayAttr({c.grid.gM, c.grid.gN}));
      row.set("partition", partition(b, c));
      row.set("local_cost", b.getDenseI64ArrayAttr(c.local));
      rows.push_back(row.getDictionary(b.getContext()));
      if (c.local != minimum)
        continue;
      InstructionLayoutTuple t;
      t.binding = c.anchor.front();
      llvm::append_range(t.encodings,
                         ArrayRef<Attribute>(c.anchor).drop_front());
      t.encodings.push_back(encodeSM90MmaGeometry(module.getContext(), c.grid));
      certificates[op].push_back(std::move(t));
    }
    proofs.push_back(b.getArrayAttr(rows));
    originalIDs.push_back(op->getAttr(marker));
    op->setAttr(marker, b.getI64IntegerAttr(id));
  }
  // Replay scratch geometry is a property of the selected backend. It can
  // stage a large logical conversion in smaller repetitions.
  auto replayPrice = [&](RankedTensorType from, RankedTensorType to) -> std::optional<int64_t> {
    if (!replayScratch) return std::nullopt;
    return gluonScratchService(from, to, bankFloor);
  };
  auto price = [&](RankedTensorType from,
                   RankedTensorType to) -> FailureOr<int64_t> {
    if (auto work = replayPrice(from, to)) {
      auto plan = module->getAttrOfType<DictionaryAttr>("frisk.execution_contract");
      if (plan && plan.getAs<ArrayAttr>("groups").size() > 1) return failure();
      return *work;
    }
    auto route = planStoreRoute(from, to, Traffic{}, Traffic{}, module.getLoc(),
                                false, false, allowStripes);
    if (!route.feasible)
      return failure();
    if (route.scratch) {
      auto plan =
          module->getAttrOfType<DictionaryAttr>("frisk.execution_contract");
      // The specialized producer group never participates in a CTA-wide
      // redistribution barrier. That lowering has no cross-group scratch ABI.
      if (plan && plan.getAs<ArrayAttr>("groups").size() > 1)
        return failure();
    }
    return route.service();
  };
  auto solved = solveCertifiedMultiAnchor(module, certificates, price, striped,
                                          freeBits, timeout, oracleLimit, scoreReduction,
                                          !replayScratch && (globalAccess == "native" ||
                                            (globalAccess == "pointwise" && anchors.empty() && !hasReduction(module))), subgroupCandidates);
  if (failed(solved))
    return failure();
  SmallVector<int64_t> selected(domains.size(), -1);
  for (auto op : anchorsIn(module)) {
    auto id = op->getAttrOfType<IntegerAttr>(marker);
    if (!id || id.getInt() < 0 || size_t(id.getInt()) >= domains.size())
      return op.emitError("lost certified anchor identity");
    size_t idx = id.getInt();
    auto tuple = binding(op);
    auto &d = domains[idx];
    auto minimum =
        llvm::min_element(d, [](const Candidate &a, const Candidate &b) {
          return a.local < b.local;
        })->local;
    auto it = llvm::find_if(d, [&](const Candidate &c) {
      return c.local == minimum && c.anchor == tuple;
    });
    if (it == d.end() || selected[idx] != -1)
      return op.emitError(
          "integer materialization changed the certified binding");
    selected[idx] = it - d.begin();
    if (originalIDs[idx])
      op->setAttr(marker, originalIDs[idx]);
    else
      op->removeAttr(marker);
  }
  if (llvm::is_contained(selected, -1))
    return module.emitError("missing certified anchor");
  // Emit the precise lowering that was priced, including across an existing
  // explicit conversion. An unmarked route retains its legacy packet ABI.
  auto routeCheck = module.walk([&](ConvertLayoutOp op) {
    auto src = cast<RankedTensorType>(op->getOperand(0).getType());
    auto dst = cast<RankedTensorType>(op->getResult(0).getType());
    bool marked = op->hasAttr("frisk.native_shared_stripes");
    if (!marked && replayPrice(src, dst)) return WalkResult::advance();
    if (!allowStripes && !marked) return WalkResult::advance();
    auto p = test::planExecutionRedistribution(src, dst, op.getLoc(),
                 shuffle_network::Objective::MinShuffle, allowStripes);
    if (failed(p) || (marked && !p->stripeElements)) {
      op.emitError("materialized native route differs from integer stripe policy");
      return WalkResult::interrupt();
    }
    if (p->stripeElements) op->setAttr("frisk.native_shared_stripes", b.getUnitAttr());
    return WalkResult::advance();
  });
  if (routeCheck.wasInterrupted()) return failure();
  if (failed(preserveStorageContracts(module)))
    return failure();
  NamedAttrList proof(*solved);
  if (replayScratch) {
    proof.set("scratch_backend", b.getStringAttr("gluon34"));
    proof.set("scratch_bank_floor", b.getBoolAttr(bankFloor));
  }
  proof.set("anchors", b.getArrayAttr(proofs));
  proof.set("selected", b.getDenseI64ArrayAttr(selected));
  module->setAttr("frisk.protected_graph", proof.getDictionary(b.getContext()));
  return success();
}

// The complete Cartesian product of the *certified* minima is considered.
// A budget rejection publishes no partial assignment or best-so-far proof.
LogicalResult selectAnchors(ModuleOp module, bool pressure, bool replay, unsigned limit) {
  auto anchors = anchorsIn(module);
  if (anchors.empty()) return module.emitError("protected selection requires an MMA");
  Builder b(module.getContext());
  SmallVector<SmallVector<Candidate, 2>, 2> domains;
  SmallVector<SmallVector<unsigned>> eligible;
  SmallVector<SmallVector<NamedAttrList>> receipts;
  size_t count = 1;
  for (MmaOp op : anchors) {
    auto choices = certify(module, op);
    if (failed(choices)) return failure();
    auto &domain = domains.emplace_back(std::move(*choices));
    auto &indices = eligible.emplace_back();
    auto &rows = receipts.emplace_back();
    const Cost minimum = llvm::min_element(domain, [](auto &x, auto &y) {
      return x.local < y.local;
    })->local;
    for (auto [i, c] : llvm::enumerate(domain)) {
      auto &row = rows.emplace_back();
      row.set("grid", b.getDenseI64ArrayAttr({c.grid.gM,c.grid.gN}));
      row.set("local_cost", b.getDenseI64ArrayAttr(c.local));
      if (c.grid.interleavedRows) row.set("row_partition", partition(b,c));
      row.set("eligible", b.getBoolAttr(c.local == minimum));
      if (c.local == minimum) indices.push_back(i);
    }
    if (count > limit / indices.size())
      return op.emitOpError("local-optimum combinations exceed max-combinations");
    count *= indices.size();
  }
  using Key = std::tuple<int64_t, Cost, int64_t, int64_t>;
  Key bestKey;
  OwningOpRef<ModuleOp> best;
  SmallVector<SmallVector<Attribute>> incumbent;
  bool attemptedIncumbent = false;
  SmallVector<Attribute> combinations;
  SmallVector<int64_t> winner;
  for (size_t index = 0; index < count; ++index) {
    size_t cursor = index;
    SmallVector<int64_t> picked(anchors.size());
    for (size_t i = anchors.size(); i-- > 0;) {
      picked[i] = eligible[i][cursor % eligible[i].size()];
      cursor /= eligible[i].size();
    }
    OwningOpRef<ModuleOp> trial(cast<ModuleOp>(module->clone()));
    auto trialOps = anchorsIn(*trial);
    // Candidate verification follows operation identity even if the
    // ordinary pipeline reorders independent MMAs during materialization.
    StringRef identity = "__frisk_protected_trial_anchor";
    for (auto [i, op] : llvm::enumerate(trialOps)) {
      op->setAttr(identity,b.getI64IntegerAttr(i));
      const auto &c = domains[i][picked[i]];
      op.setPolicyAttr(GemmWarpPolicyAttr::get(b.getContext(),c.pinned));
      op->setAttr("frisk.row_partition", partition(b,c));
      op->setAttr("frisk.mma_contract", c.anchor.front());
    }
    auto graph = infer(*trial, false);
    if (failed(graph)) return module.emitError("could not score a protected candidate graph");
    trialOps.assign(anchors.size(), MmaOp{});
    for (MmaOp op : anchorsIn(*trial)) {
      auto id = op->getAttrOfType<IntegerAttr>(identity);
      if (!id || id.getInt()<0 || size_t(id.getInt())>=anchors.size() || trialOps[id.getInt()])
        return module.emitError("materialization lost a protected anchor identity");
      trialOps[id.getInt()] = op;
      if (auto originalID = anchors[id.getInt()]->getAttr(identity)) op->setAttr(identity,originalID);
      else op->removeAttr(identity);
    }
    if (llvm::any_of(trialOps, [](MmaOp op){ return !op; }))
      return module.emitError("materialization removed a protected anchor");
    bool matched = true;
    for (auto [i, op] : llvm::enumerate(trialOps)) {
      if (binding(op) != domains[i][picked[i]].anchor) matched = false;
      op.setPolicyAttr(anchors[i].getPolicyAttr());
      if (!anchors[i]->hasAttr("frisk.row_partition")) op->removeAttr("frisk.row_partition");
    }
    NamedAttrList combo;
    combo.set("candidate_indices", b.getDenseI64ArrayAttr(picked));
    combo.set("bindings_verified", b.getBoolAttr(matched));
    if (matched) {
      int64_t internal = 0;
      trial->walk([&](ConvertLayoutOp) { ++internal; });
      StoreTieBreak tie;
      auto edge = chooseStores(*trial, pressure, tie, replay);
      auto target = createSM90LayoutTarget();
      if (failed(edge) || failed(verify(*trial)) || failed(verifyMaterializedLayouts(*trial,*target)))
        return module.emitError("could not verify the complete protected graph");
      NamedAttrList &row = anchors.size()==1 ? receipts[0][picked[0]] : combo;
      row.set("store_cost",b.getI64IntegerAttr(*edge));
      row.set("writer_peak_slots",b.getI64IntegerAttr(tie.peakSlots));
      row.set("writer_exposed_peak_slots",b.getI64IntegerAttr(tie.exposedPeakSlots));
      row.set("writer_publisher_group_stripes",b.getI64IntegerAttr(tie.publisherGroupStripes));
      row.set("graph_cost",b.getDenseI64ArrayAttr(*graph));
      combo.set("internal_conversions",b.getI64IntegerAttr(internal));
      Key key{*edge,*graph,tie.exposedPeakSlots,tie.publisherGroupStripes};
      bool prefer = !best || key < bestKey;
      if (best && key == bestKey) {
        if (!attemptedIncumbent) {
          attemptedIncumbent = true;
          ScopedDiagnosticHandler optionalBaseline(module.getContext(), [](Diagnostic &) {return success();});
          OwningOpRef<ModuleOp> base(cast<ModuleOp>(module->clone()));
          if (succeeded(infer(*base,false)))
            for (auto op : anchorsIn(*base)) incumbent.push_back(binding(op));
        }
        prefer = incumbent.size() == anchors.size();
        if (prefer)
          for (auto [i, j] : llvm::enumerate(picked))
            prefer &= domains[i][j].anchor == incumbent[i];
      }
      if (prefer) { best = std::move(trial); bestKey = key; winner = picked; }
    }
    combinations.push_back(combo.getDictionary(b.getContext()));
  }
  if (!best) return module.emitError("no compatible graph preserved all certified MMA bindings");
  module.getBodyRegion().takeBody(best->getBodyRegion());
  SmallVector<Attribute> certificates;
  for (auto &rows : receipts) {
    SmallVector<Attribute> values;
    for (auto &row : rows) values.push_back(row.getDictionary(b.getContext()));
    certificates.push_back(b.getArrayAttr(values));
  }
  if (anchors.size() == 1) module->setAttr("frisk.protected_candidates",certificates[0]);
  else {
    NamedAttrList proof;
    if (attemptedIncumbent) proof.set("incumbent_available", b.getBoolAttr(incumbent.size() == anchors.size()));
    proof.set("anchors",b.getArrayAttr(certificates));
    proof.set("combinations",b.getArrayAttr(combinations));
    proof.set("selected",b.getDenseI64ArrayAttr(winner));
    module->setAttr("frisk.protected_graph",proof.getDictionary(b.getContext()));
  }
  return success();
}

class ProtectedStoreLayoutPass : public PassWrapper<ProtectedStoreLayoutPass,OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProtectedStoreLayoutPass)
  ProtectedStoreLayoutPass() = default;
  ProtectedStoreLayoutPass(const ProtectedStoreLayoutPass &other) : PassWrapper(other) {}
  explicit ProtectedStoreLayoutPass(bool integer) {
    if (integer)
      anchorSolver = "integer";
  }
  Option<std::string> anchorSolver{*this, "anchor-solver",
                                   llvm::cl::desc("enumerate or integer"),
                                   llvm::cl::init("enumerate")};
  Option<bool> subgroupCandidates{*this, "integer-subgroup-candidates",
      llvm::cl::desc("Propose warp- and warpgroup-compatible consumer bases"), llvm::cl::init(false)};
  Option<unsigned> completionBits{
      *this, "integer-free-bits",
      llvm::cl::desc("Expand full-rank intermediate F2 completions up to this "
                     "number of differing coefficients (0..8)"),
      llvm::cl::init(4)};
  Option<unsigned> oracleLimit{
      *this, "integer-oracle-limit",
      llvm::cl::desc("Verify optimum by independent exhaustive enumeration up "
                     "to N visited nodes (0 disables)"),
      llvm::cl::init(0)};
  Option<bool> allowStripes{*this,"integer-native-stripes",llvm::cl::desc("Offer bounded 64 KiB two-warpgroup shared transfers"),llvm::cl::init(true)};
  Option<bool> scoreReduction{*this,"integer-reduction-communication",llvm::cl::desc("Charge warp and warpgroup communication in reduction fibers"),llvm::cl::init(true)};
  Option<std::string> scratchBackend{*this, "integer-scratch-backend",
      llvm::cl::desc("Shared scratch lowering used for pricing (native or gluon34)"),
      llvm::cl::init("native")};
  Option<std::string> globalAccess{*this, "integer-global-access",
      llvm::cl::desc("Price aligned native vector loads/stores: scalar, pointwise, or native"),
      llvm::cl::init("scalar")};
  Option<bool> bankFloor{*this, "integer-scratch-bank-floor",
      llvm::cl::desc("Charge the blocked replay publisher's bank serialization floor"),
      llvm::cl::init(true)};
  Option<unsigned> integerTimeout{
      *this, "integer-timeout-ms",
      llvm::cl::desc("Integer optimization timeout; incomplete solutions fail"),
      llvm::cl::init(60000)};
  Option<bool> scorePressure{*this, "score-spill-risk",
      llvm::cl::desc("Charge SM90 register pressure of the chosen accumulator/store pair"),
      llvm::cl::init(true)};
  Option<bool> stripedReplay{*this, "striped-shared-replay",
      llvm::cl::desc("Price the forced scalar 64 KiB striped replay router instead of native redistribution"),
      llvm::cl::init(false)};
  Option<unsigned> maxCombinations{*this,"max-combinations",
      llvm::cl::desc("Maximum complete anchor assignments per function"),
      llvm::cl::init(256)};
  Option<std::string> materialize{*this, "materialize-store",
      llvm::cl::desc("Materialize one fixed native store candidate (auto, direct or striped)"),
      llvm::cl::init("auto")};
  Option<bool> fixedAnchors{*this, "fixed-anchors",
      llvm::cl::desc("Score stores of an already materialized graph while retaining all instruction bindings"),
      llvm::cl::init(false)};
  StringRef getArgument() const final { return "frisk-protected-store-layout"; }
  StringRef getDescription() const final {
    return "Select compatible verified MMA local optima with per-consumer scalar global-store layouts";
  }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<FriskDialect,func::FuncDialect,memref::MemRefDialect,arith::ArithDialect,scf::SCFDialect>();
  }
  void runOnOperation() override {
    ModuleOp original=getOperation();
    auto error=[&](StringRef msg){original.emitError(msg);signalPassFailure();};
    if (oracleLimit == std::numeric_limits<unsigned>::max())
      return error("integer oracle limit too large");
    if (anchorSolver == "integer" && completionBits > 8)
      return error("integer-free-bits must be between 0 and 8");
    if (anchorSolver != "enumerate" && anchorSolver != "integer")
      return error("unknown anchor solver");
    if (anchorSolver == "integer" &&
        (fixedAnchors || stripedReplay || materialize != "auto"))
      return error("integer anchor solving uses native SSA delivery");
    if (globalAccess != "scalar" && globalAccess != "pointwise" && globalAccess != "native")
      return error("unknown integer Global access pricing mode");
    if (scratchBackend != "native" && scratchBackend != "gluon34")
      return error("unknown integer scratch backend");
    if (scratchBackend == "gluon34" && (anchorSolver != "integer" || !anchorsIn(original).empty()))
      return error("gluon34 scratch pricing requires an integer reduction graph without MMA");
    if (original->hasAttr("frisk.compiled_sum_spill") &&
        (scratchBackend != "gluon34" || anchorSolver != "integer"))
      return error("compiled Sum spill evidence requires the Gluon integer scoring path");
    if (subgroupCandidates && anchorSolver != "integer")
      return error("subgroup candidates require integer anchor solving");
    auto select = [&](ModuleOp mod) -> LogicalResult {
      if (anchorSolver != "integer")
        return selectAnchors(mod, scorePressure, stripedReplay, maxCombinations);
      auto solve = [&](ModuleOp graph, bool expand) {
        return selectIntegerAnchors(graph, completionBits, integerTimeout,
                                    oracleLimit, scoreReduction, allowStripes,
                                    scratchBackend == "gluon34", bankFloor,
                                    globalAccess, expand);
      };
      if (!subgroupCandidates)
        return solve(mod, false);

      // An enlarged Z3 domain can change an existing witness even when
      // none of the added bases is selected. Preserve the incumbent IR on
      // objective ties, rather than penalizing only newly selected bases.
      OwningOpRef<ModuleOp> challenger(cast<ModuleOp>(mod->clone()));
      bool feasible;
      {
        ScopedDiagnosticHandler optional(mod.getContext(), [](Diagnostic &) { return success(); });
        feasible = succeeded(solve(mod, false));
      }
      bool expanded;
      if (feasible) {
        ScopedDiagnosticHandler optional(mod.getContext(), [](Diagnostic &) { return success(); });
        expanded = succeeded(solve(*challenger, true));
      } else {
        expanded = succeeded(solve(*challenger, true));
        if (!expanded) return failure();
      }
      auto receipt = [](ModuleOp graph) {
        return graph->getAttrOfType<DictionaryAttr>("frisk.protected_graph");
      };
      auto objective = [](DictionaryAttr proof) {
        // billWitness has checked the full sum for signed overflow.
        int64_t service = 0;
        for (StringRef k : {"main_loop_units", "reduction_units", "access_units",
                            "conversion_units", "compiled_spill_units"})
          if (auto v = proof.getAs<IntegerAttr>(k)) service += v.getInt();
        return std::make_pair(service, proof.getAs<IntegerAttr>("distinct_conversions").getInt());
      };
      auto candidate = expanded ? receipt(*challenger) : DictionaryAttr{};
      bool promote = expanded && (!feasible || objective(candidate) < objective(receipt(mod)));
      if (promote) {
        mod->setAttrs((*challenger)->getAttrs());
        mod.getBodyRegion().takeBody(challenger->getBodyRegion());
      }
      Builder b(mod.getContext());
      NamedAttrList report(receipt(mod));
      report.set("subgroup_challenger_certified", b.getBoolAttr(expanded));
      report.set("subgroup_promoted", b.getBoolAttr(promote));
      if (expanded)
        report.set("subgroup_challenger_candidates", candidate.get("subgroup_candidates"));
      mod->setAttr("frisk.protected_graph", report.getDictionary(mod.getContext()));
      return success();
    };
    if (materialize != "auto" && (!fixedAnchors || stripedReplay ||
        (materialize != "direct" && materialize != "striped")))
      return error("materialize-store requires fixed native anchors and a known candidate");
    if (fixedAnchors) {
      OwningOpRef<ModuleOp> trial(cast<ModuleOp>(original->clone()));
      auto target = createSM90LayoutTarget();
      auto chip = (*trial)->getAttrOfType<StringAttr>("frisk.target");
      if (!chip || chip.getValue() != "sm_90a" ||
          failed(verifyMaterializedLayouts(*trial, *target)))
        return error("fixed anchors require a verified materialized SM90a graph");
      SmallVector<SmallVector<Attribute>> prior;
      trial->walk([&](MmaOp mma) { prior.push_back(binding(mma)); });
      StoreTieBreak tie;
      auto cost = chooseStores(*trial, scorePressure, tie, stripedReplay, /*fixedAnchors=*/true, materialize.getValue());
      if (failed(cost)) return error("fixed-anchor store address coverage failed");
      SmallVector<SmallVector<Attribute>> current;
      trial->walk([&](MmaOp mma) { current.push_back(binding(mma)); });
      if (prior != current || failed(verify(*trial)) ||
          failed(verifyMaterializedLayouts(*trial, *target)))
        return error("fixed-anchor store selection invalidated the materialized graph");
      Builder b(&getContext());
      auto score = b.getDenseI64ArrayAttr({*cost, tie.exposedPeakSlots,
                                           tie.publisherGroupStripes});
      original.getBodyRegion().takeBody(trial->getBodyRegion());
      original->setAttr("frisk.fixed_store_cost", score);
      return;
    }
    OwningOpRef<ModuleOp> staged(cast<ModuleOp>(original->clone()));
    SmallVector<Operation *> scopes;
    for (Operation &op : staged->getBody()->getOperations())
      if (!anchorsIn(&op).empty() || (anchorSolver == "integer" && (hasReduction(&op) || hasPointwiseWork(&op))))
        scopes.push_back(&op);
    if (scopes.size() > 1 && original->hasAttr("frisk.compiled_sum_spill"))
      return error("compiled Sum spill evidence must describe one kernel");
    if (scopes.size() <= 1) {
      if (failed(select(*staged))) {
        signalPassFailure(); return;
      }
    } else {
      // Independent functions have independent local domains and budgets.
      // Keep all replacements on the clone until every function verifies.
      NamedAttrList byFunction;
      for (Operation *scope : scopes) {
        auto name = scope->getAttrOfType<StringAttr>("sym_name");
        if (!name) return error("expected a named top-level anchor scope");
        OwningOpRef<ModuleOp> part(ModuleOp::create(original.getLoc()));
        (*part)->setAttrs((*staged)->getAttrs());
        part->push_back(scope->clone());
        if (failed(select(*part))) {
          signalPassFailure(); return;
        }
        auto proof=(*part)->getAttr("frisk.protected_graph");
        if (!proof) proof=(*part)->getAttr("frisk.protected_candidates");
        byFunction.set(name.getValue(),proof);
        part->getBody()->front().moveBefore(scope);
        scope->erase();
      }
      (*staged)->setAttr("frisk.protected_functions",byFunction.getDictionary(&getContext()));
    }
    if (failed(verify(*staged))) return error("invalid protected module");
    (*staged)->removeAttr("frisk.compiled_sum_spill");
    original->setAttrs((*staged)->getAttrs());
    original.getBodyRegion().takeBody(staged->getBodyRegion());
  }
};
// A spill is a property of a complete compiled SSA placement. Price that
// placement once, after lowering. Never attach the same traffic independently
// to its reductions, broadcasts, or operands.
// Charge the actual scalar/64-bit/128-bit instruction mix against the
// materialized, root-relative addresses. For mixed widths the histogram is
// sufficient only if the sector count does not depend on the register group.
static FailureOr<int64_t> globalSectorCount(Value storage, RankedTensorType type,
    ArrayRef<int64_t> counts, int64_t &warpIssues) {
  auto view = storage.getDefiningOp<LayoutViewOp>();
  auto layout = view ? view->getAttrOfType<StorageLayoutAttr>("layout") : StorageLayoutAttr{};
  auto enc = dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding());
  if (!layout || !enc || layout.getMemorySpace().getValue() != attr::MemorySpace::Global ||
      type.getRank() != 2 || !type.getElementType().isF32() || !type.hasStaticShape())
    return failure();
  auto root = analyzeStorageAlias(storage);
  if (failed(root)) return failure();
  DominanceInfo dominance(view->getParentOfType<ModuleOp>());
  for (Operation *user : root->root.getUsers()) {
    auto contract = dyn_cast<StorageContractOp>(user);
    if (contract && contract.getRoot() == root->root &&
        storageContractDominates(contract, view, dominance))
      root->rootAlignment = std::max<uint64_t>(root->rootAlignment,
          contract.getLayout().getAlignment().getInt());
  }
  if (root->rootAlignment < 32) return failure();
  auto footprint = buildStorageAliasFootprint(*root, layout);
  ExecutionEnumeration e;
  if (footprint.proof.status != ProofStatus::Proven ||
      enumerateExecutionLayout(enc, type, e).status != ProofStatus::Proven ||
      e.topology[1] != 32 || e.topology[2] != 4 || e.topology[3] != 1 ||
      e.topology[4] != 1 || e.logical.size() != footprint.entries.size() ||
      e.logical.size() != size_t(type.getNumElements()))
    return failure();
  const uint64_t regs = e.topology[0];
  uint64_t covered = 0;
  for (auto [i,count] : llvm::enumerate(counts)) {
    if (count < 0 || uint64_t(count) > regs) return failure();
    covered += uint64_t(count) << i;
  }
  if (covered != regs) return failure();
  SmallVector<uint64_t> addresses(e.logical.size(), UINT64_MAX);
  for (const auto &entry : footprint.entries) {
    if (entry.view.size() != 2 || entry.end - entry.begin != 32 || entry.begin % 32)
      return failure();
    int64_t i = entry.view[0], j = entry.view[1];
    if (i < 0 || i >= type.getDimSize(0) || j < 0 || j >= type.getDimSize(1))
      return failure();
    auto &address = addresses[i * type.getDimSize(1) + j];
    if (address != UINT64_MAX) return failure();
    address = entry.begin / 8;
  }
  auto address = [&](uint64_t lane, uint64_t reg) {
    return addresses[e.logical[lane * regs + reg]];
  };
  int64_t sectors = 0;
  for (auto [i,count] : llvm::enumerate(counts)) {
    if (!count) continue;
    uint64_t width = 1u << i;
    int64_t perIssue = -1, allGroups = 0;
    for (uint64_t reg = 0; reg < regs; reg += width) {
      int64_t group = 0;
      for (unsigned warp = 0; warp < 4; ++warp) {
        llvm::SmallDenseSet<uint64_t,32> touched;
        for (unsigned t = warp * 32; t < (warp + 1) * 32; ++t) {
          auto start = address(t, reg);
          if (start % (4 * width)) return failure();
          for (unsigned k = 1; k < width; ++k)
            if (address(t, reg + k) != start + 4 * k) return failure();
          touched.insert(start / 32);
        }
        group += touched.size();
      }
      if (perIssue < 0) perIssue = group;
      if (uint64_t(count) != regs / width && group != perIssue) return failure();
      allGroups += group;
    }
    sectors += uint64_t(count) == regs / width ? allGroups : count * perIssue;
    warpIssues += count * 4;
  }
  return sectors;
}

struct PriceNormalizationSpill : PassWrapper<PriceNormalizationSpill, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PriceNormalizationSpill)
  PriceNormalizationSpill() = default;
  PriceNormalizationSpill(const PriceNormalizationSpill &other) : PassWrapper(other) {}
  StringRef getArgument() const final { return "frisk-price-normalization-spill"; }
  StringRef getDescription() const final { return "Bind SM90 Gluon local traffic to one certified whole-graph normalization layout"; }
  Option<std::string> graph{*this,"graph",llvm::cl::desc("Sealed materialized graph SHA-256"), llvm::cl::init("")};
  Option<std::string> cubin{*this,"cubin",llvm::cl::desc("Disassembled cubin SHA-256"), llvm::cl::init("")};
  Option<int64_t> loads{*this,"load-words",llvm::cl::init(-1)};
  Option<int64_t> stores{*this,"store-words",llvm::cl::init(-1)};
  Option<int64_t> exposed{*this,"exposed-reloads",llvm::cl::desc("Certified immediate arithmetic uses of local reloads per warp"),llvm::cl::init(-1)};
  Option<std::string> ldg{*this,"ldg",llvm::cl::desc("Unpredicated 32/64/128-bit load counts"),llvm::cl::init("")};
  Option<std::string> stg{*this,"stg",llvm::cl::desc("Unpredicated 32/64/128-bit store counts"),llvm::cl::init("")};
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    auto fail = [&](StringRef msg) { mod.emitError(msg); signalPassFailure(); };
    auto proof = mod->getAttrOfType<DictionaryAttr>("frisk.protected_graph");
    if (!proof || !proof.get("normalization_reduction_choices") || proof.get("compiled_spill_units"))
      return fail("expected an unpriced certified normalization alternative");
    unsigned blocks = 0;
    bool legal = true;
    mod.walk([&](ParallelOp op) { ++blocks; legal &= op.getThreads() == 128; });
    if (blocks != 1 || !legal || !anchorsIn(mod).empty())
      return fail("expected a single four-warp normalization scope");
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    // The seal is metadata about the graph, not part of its identity.
    OwningOpRef<ModuleOp> canonical(cast<ModuleOp>(mod->clone()));
    auto seal = (*canonical)->removeAttr("frisk.normalization_graph_digest");
    canonical->print(stream, OpPrintingFlags().useLocalScope());
    stream.flush();
    std::string digest = llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(printed)), true);
    Builder b(mod.getContext());
    if (seal && seal != b.getStringAttr(digest))
      return fail("stale normalization graph seal");
    const bool physical = !ldg.getValue().empty() || !stg.getValue().empty();
    if (exposed < -1 || (exposed >= 0 && (!physical || exposed > loads)))
      return fail("invalid exposed local reload evidence");
    if (!physical && graph.getValue().empty() && cubin.getValue().empty() && loads == -1 && stores == -1) {
      mod->setAttr("frisk.normalization_graph_digest", b.getStringAttr(digest));
      return;
    }
    if (graph.getValue() != digest || cubin.getValue().size() != 64 ||
        !llvm::all_of(cubin.getValue(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); }) ||
        loads < 0 || stores < 0)
      return fail("mismatched normalization graph, cubin or local traffic evidence");
    int64_t words, sectors, total;
    if (llvm::AddOverflow(loads.getValue(), stores.getValue(), words) ||
        llvm::MulOverflow(words, int64_t{16}, sectors))
      return fail("normalization spill sector overflow");
    total = sectors;
    for (StringRef key : {"access_units","conversion_units","main_loop_units","reduction_units"}) {
      auto v = proof.getAs<IntegerAttr>(key);
      if (!v || v.getInt() < 0 || llvm::AddOverflow(total,v.getInt(),total))
        return fail("invalid normalization service receipt");
    }
    NamedAttrList result(proof);
    if (physical) {
      // Replace the pre-lowering global term; it includes both scalar issue
      // work and scalar coalescing, so adding the new term would double count.
      SmallVector<TileLoadOp> reads;
      SmallVector<TileStoreOp> writes;
      mod.walk([&](TileLoadOp op) { reads.push_back(op); });
      mod.walk([&](TileStoreOp op) { writes.push_back(op); });
      if (reads.size() != 1 || writes.size() != 1)
        return fail("global access evidence requires one load and one store");
      auto counts = [](StringRef text) -> FailureOr<SmallVector<int64_t>> {
        SmallVector<StringRef> fields;
        text.split(fields, ',');
        SmallVector<int64_t> values;
        for (auto field : fields) {
          int64_t n;
          if (field.getAsInteger(10,n) || n < 0) return failure();
          values.push_back(n);
        }
        if (values.size() != 3) return failure();
        return values;
      };
      auto rd = counts(ldg.getValue()), wr = counts(stg.getValue());
      if (failed(rd) || failed(wr)) return fail("invalid global instruction histogram");
      int64_t readIssues = 0, writeIssues = 0;
      auto r = globalSectorCount(reads[0].getSource(),
          cast<RankedTensorType>(reads[0].getResult().getType()), *rd, readIssues);
      auto w = globalSectorCount(writes[0].getTarget(),
          cast<RankedTensorType>(writes[0]->getOperand(0).getType()), *wr, writeIssues);
      if (failed(r) || failed(w))
        return fail("global instruction footprint is not certified");
      constexpr int64_t issueWeight = 8;
      int64_t globalUnits = *r + *w + issueWeight * (readIssues + writeIssues);
      total -= proof.getAs<IntegerAttr>("access_units").getInt();
      if (llvm::AddOverflow(total, globalUnits, total)) return fail("global service overflow");
      if (exposed >= 0) {
        // A wait with no intervening instruction exposes a separate issue
        // opportunity for each of the four warps. Reuse the global issue unit.
        int64_t units, dependent;
        if (llvm::MulOverflow(exposed.getValue(), 4 * issueWeight, units) ||
            llvm::AddOverflow(total, units, dependent))
          return fail("compiled local dependency cost overflow");
        result.set("compiled_exposed_reloads", b.getI64IntegerAttr(exposed));
        result.set("compiled_exposed_reload_units", b.getI64IntegerAttr(units));
        result.set("compiled_dependency_service_units", b.getI64IntegerAttr(dependent));
      }
      result.set("compiled_global_read_sectors", b.getI64IntegerAttr(*r));
      result.set("compiled_global_write_sectors", b.getI64IntegerAttr(*w));
      result.set("compiled_global_read_issues", b.getI64IntegerAttr(readIssues));
      result.set("compiled_global_write_issues", b.getI64IntegerAttr(writeIssues));
      result.set("compiled_global_issue_weight", b.getI64IntegerAttr(issueWeight));
      result.set("compiled_ldg_counts", b.getDenseI64ArrayAttr(*rd));
      result.set("compiled_stg_counts", b.getDenseI64ArrayAttr(*wr));
    }
    result.set("normalization_graph_digest", b.getStringAttr(digest));
    result.set("compiled_spill_units", b.getI64IntegerAttr(sectors));
    result.set("compiled_spill_cubin", b.getStringAttr(cubin.getValue()));
    result.set("compiled_service_units", b.getI64IntegerAttr(total));
    mod->setAttr("frisk.protected_graph",result.getDictionary(mod.getContext()));
  }
};
} // namespace

namespace mlir::frisk {
void registerProtectedStoreLayoutPass() {
  static PassRegistration<ProtectedStoreLayoutPass> registration;
  (void)registration;
  static PassRegistration<PriceNormalizationSpill> normalization;
  (void)normalization;
  // A top-level opt-in pipeline on raw Frisk IR. The incumbent and each
  // pinned candidate continue to use the ordinary layout pipeline internally.
  static PassPipelineRegistration<> pipeline("frisk-protected-layout-pipeline",
      "Materialize an SM90 MMA graph while preserving instruction-local optima",
      [](OpPassManager &pm) { pm.addPass(std::make_unique<ProtectedStoreLayoutPass>()); });
  (void)pipeline;
  static PassPipelineRegistration<> multi(
      "frisk-multi-anchor-layout-pipeline",
      "Joint integer layout selection over certified local MMA optima",
      [](OpPassManager &pm) {
        pm.addPass(std::make_unique<ProtectedStoreLayoutPass>(true));
      });
  (void)multi;
}
} // namespace mlir::frisk
