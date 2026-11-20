#include "MultiAnchorSolver.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutConversionChoice.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutConstraints.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/MathExtras.h"
#include <map>
#ifdef FRISK_HAS_Z3
#include <z3++.h>
#endif

using namespace mlir;
using namespace mlir::frisk;
#ifdef FRISK_HAS_Z3
namespace {
// Expand a common ownership matrix only at ordinary unencoded endpoints.
// The local MMA tuple table remains authoritative at instruction roles.
struct Completions {
  DenseMap<LayoutVarID, SmallVector<Attribute>> matrices, writers;
  unsigned skippedWide = 0;
};
Completions addCompletions(LayoutConstraintGraph &graph, LayoutTarget &target,
                           ModuleOp mod, unsigned limit, StoreMap storeMap,
                           bool offerConsecutiveWriters) {
  Completions out;
  for (auto &var : graph.getVariables()) {
    auto type = dyn_cast<RankedTensorType>(var.shapedType);
    if (!type || var.kind != LayoutKind::Distributed || var.instructionRole ||
        var.reductionResult)
      continue;
    if (llvm::any_of(graph.getConstraints(), [&](const auto &c) {
          return c.kind == ConstraintKind::RequireEncoding &&
                 llvm::is_contained(c.vars, var.id);
        }))
      continue;
    auto add = [&](Attribute a, StringRef reason) {
      if (!a || llvm::any_of(var.candidates,
                             [&](const auto &c) { return c.value == a; }))
        return;
      ScopedDiagnosticHandler optional(mod.getContext(),
                                       [](Diagnostic &) { return success(); });
      if (failed(target.verifyCandidate(var, a, mod.getLoc())))
        return;
      auto origin = graph.addProvenance(std::nullopt, var.anchor,
                                        "protected-completion", reason);
      var.candidates.push_back({a, origin, var.id});
      (reason == "existing consecutive writer" ? out.writers
                                               : out.matrices)[var.id]
          .push_back(a);
    };
    auto seeds = var.candidates;
    // Offer the existing native consecutive-writer policy at each store use.
    if (offerConsecutiveWriters && var.use && isa<TileStoreOp>(var.use->getOwner()) &&
        type.getRank() == 2 && type.getElementType().isF32())
      for (auto &seed : seeds) {
        auto enc = dyn_cast<DistributedEncodingAttr>(seed.value);
        if (enc)
          add(storeMap(type, enc), "existing consecutive writer");
      }
    if (limit == 0 || seeds.size() < 2)
      continue;
    SmallVector<Attribute> seen;
    for (auto &seed : seeds) {
      auto base = dyn_cast<DistributedEncodingAttr>(seed.value);
      if (!base || base.getReplication().getInt() != 1 ||
          llvm::is_contained(seen, base))
        continue;
      auto ref = dyn_cast<BitLinearLayoutMapAttr>(base.getMap());
      if (!ref || !llvm::all_of(type.getShape(), [](int64_t n) {
            return n > 0 && llvm::isPowerOf2_64(n);
          }))
        continue;
      auto dims = ref.getMatrix().getType().getShape();
      if (dims[0] != dims[1])
        continue;
      SmallVector<APInt> entries(ref.getMatrix().getValues<APInt>().begin(),
                                 ref.getMatrix().getValues<APInt>().end());
      llvm::SmallBitVector differs(entries.size());
      for (auto &other : seeds) {
        auto e = dyn_cast<DistributedEncodingAttr>(other.value);
        auto m = e ? dyn_cast<BitLinearLayoutMapAttr>(e.getMap())
                   : BitLinearLayoutMapAttr();
        if (!m || e.getTopology() != base.getTopology() ||
            e.getReplication() != base.getReplication() ||
            m.getInputNames() != ref.getInputNames() ||
            m.getOutputNames() != ref.getOutputNames() ||
            m.getInputBitWidths() != ref.getInputBitWidths() ||
            m.getOutputBitWidths() != ref.getOutputBitWidths())
          continue;
        seen.push_back(e);
        unsigned i = 0;
        for (APInt x : m.getMatrix().getValues<APInt>()) {
          if (x != entries[i])
            differs.set(i);
          ++i;
        }
      }
      if (!differs.any())
        continue;
      if (differs.count() > limit) {
        ++out.skippedWide;
        continue;
      }
      for (unsigned choice = 0; choice < (1u << differs.count()); ++choice) {
        unsigned bit = 0;
        for (int i : differs.set_bits())
          entries[i] = APInt(1, (choice >> bit++) & 1);
        auto matrix =
            DenseIntElementsAttr::get(ref.getMatrix().getType(), entries);
        auto m = BitLinearLayoutMapAttr::get(
            mod.getContext(), ref.getInputNames(), ref.getInputBitWidths(),
            ref.getOutputNames(), ref.getOutputBitWidths(), matrix);
        auto gf = m.getMatrixValue();
        if (failed(gf) || gf->rank() != unsigned(dims[0]))
          continue;
        add(DistributedEncodingAttr::get(
                mod.getContext(), m, base.getTopology(), base.getReplication()),
            "full-rank completion of free GF(2) coefficients");
      }
    }
  }
  return out;
}

// Use the same bounded loop-frequency proxy for memory service and transfers.
int64_t frequency(Operation *op) {
  return int64_t{1} << std::min<uint64_t>(getLayoutLoopDepth(op), 16);
}
FailureOr<int64_t> accessUnits(const LayoutConstraintGraph &graph,
                               const LayoutConstraint &c,
                               const CandidateAssignment &assignment, bool nativeVector) {
  auto cost = evaluateSM90AccessCost(graph, c, assignment, nativeVector);
  if (failed(cost) || cost->saturated)
    return failure();
  auto &v = cost->cost;
  auto weight = frequency(graph.getProvenances()[c.provenance].source);
  auto maxTerm = uint64_t(INT64_MAX / weight / 3);
  if (v.memoryTransactions > maxTerm || v.bankConflictDegree > maxTerm ||
      v.instructionPathAndWork > maxTerm)
    return failure();
  return int64_t(v.memoryTransactions + v.bankConflictDegree +
                 v.instructionPathAndWork) *
         weight;
}

// A fully owned vector tile. Take up to 128 contiguous bits per thread,
// then distribute lanes along the requested minor axis and warps across it.
DistributedEncodingAttr memoryOrientation(RankedTensorType t, int threads,
                                          unsigned minor, bool vectorize) {
  if (!t || t.getRank() != 2 || !t.hasStaticShape() || threads < 32 ||
      threads > 1024 || !llvm::isPowerOf2_64(threads) ||
      !t.getElementType().isIntOrFloat())
    return {};
  unsigned bitsPerElement = t.getElementTypeBitWidth();
  if (!bitsPerElement || bitsPerElement > 128) return {};
  auto shape = t.getShape();
  if (!llvm::isPowerOf2_64(shape[0]) || !llvm::isPowerOf2_64(shape[1]))
    return {};
  unsigned w[2] = {llvm::Log2_64(shape[0]), llvm::Log2_64(shape[1])};
  unsigned bits = w[0] + w[1], threadBits = llvm::Log2_64(threads);
  // The SM90 access-footprint proof supports up to 65536 logical values.
  if (bits < threadBits || bits > 16) return {};
  unsigned regs = bits - threadBits, used[2] = {};
  SmallVector<unsigned> reg, lane, warp;
  auto take = [&](SmallVectorImpl<unsigned> &carrier, unsigned dim, unsigned n) {
    unsigned count = std::min(n, w[dim] - used[dim]);
    for (unsigned i = 0; i < count; ++i)
      carrier.push_back((dim ? w[0] : 0) + used[dim]++);
    return n - count;
  };
  if (vectorize) take(reg, minor, std::min(regs, unsigned(llvm::Log2_64(128 / bitsPerElement))));
  unsigned major = 1 - minor;
  unsigned remainingLanes = take(lane, minor, 5);
  take(lane, major, remainingLanes);
  unsigned remainingWarps = take(warp, major, threadBits - 5);
  take(warp, minor, remainingWarps);
  take(reg, minor, w[minor]);
  take(reg, major, w[major]);
  if (reg.size() != regs || lane.size() != 5 || warp.size() != threadBits-5)
    return {};
  Builder b(t.getContext());
  SmallVector<APInt> matrix(bits * bits, APInt(1, 0));
  SmallVector<Attribute> names;
  SmallVector<int64_t> widths;
  unsigned col = 0;
  auto append = [&](StringRef name, ArrayRef<unsigned> rows) {
    if (rows.empty()) return;
    names.push_back(b.getStringAttr(name)); widths.push_back(rows.size());
    for (unsigned row : rows) matrix[row * bits + col++] = APInt(1, 1);
  };
  append("register", reg); append("lane", lane); append("warp", warp);
  auto map = BitLinearLayoutMapAttr::get(t.getContext(), b.getArrayAttr(names),
      b.getDenseI64ArrayAttr(widths), b.getStrArrayAttr({"dim0","dim1"}),
      b.getDenseI64ArrayAttr({w[0],w[1]}), DenseIntElementsAttr::get(
      RankedTensorType::get({bits,bits}, b.getI1Type()), matrix));
  return DistributedEncodingAttr::get(t.getContext(), map,
      b.getDenseI64ArrayAttr({int64_t{1} << regs,32,threads/32,1,1}),
      b.getI64IntegerAttr(1));
}

// Offer preferred orientations across a pointwise SSA component. Bound roles
// retain their original domain. Non-pointwise relations use target proposals.
bool seedPointwiseAccesses(LayoutConstraintGraph &g, LayoutTarget &target,
                           ModuleOp mod, bool nativeVector) {
  auto constraints = g.getConstraints();
  if (llvm::any_of(constraints, [](const auto &c) {
        return c.instruction || c.reduction || c.kind == ConstraintKind::TransformLayout ||
               c.kind == ConstraintKind::BroadcastLayout;
      })) return false;
  using SeedSet = SmallVector<Attribute, 3>;
  SmallVector<SeedSet> seeds(g.getVariables().size());
  SmallVector<bool> bound(seeds.size(), false);
  bool proposed = false;
  auto insert = [&](LayoutVarID id, Attribute a) {
    if (llvm::is_contained(seeds[id], a)) return false;
    seeds[id].push_back(a);
    return true;
  };
  for (const auto &c : constraints) {
    if (c.kind == ConstraintKind::RequireEncoding) {
      auto id = c.vars.front();
      bound[id] = true;
      if (g.getVariable(id).kind == LayoutKind::Distributed)
        insert(id, c.requiredEncoding);
    }
    if (c.kind != ConstraintKind::StorageAccess || c.vars.size() != 2) continue;
    auto id = c.vars[0], storage = c.vars[1];
    auto &v = g.getVariable(id);
    auto &stores = g.getVariable(storage).candidates;
    if (stores.size() != 1) continue;
    auto t = dyn_cast<RankedTensorType>(v.shapedType);
    // Scalar and vector memory instructions admit different ownership maps.
    // Offer local minima within each width without changing the service model.
    for (bool vectorize : {false, true}) {
      Attribute candidates[2]; int64_t units[2];
      bool valid = true;
      for (unsigned axis = 0; axis < 2; ++axis) {
        candidates[axis] = memoryOrientation(t, v.requiredThreads, axis, vectorize);
        if (!candidates[axis] || failed(target.verifyCandidate(v,candidates[axis],mod.getLoc()))) {
          valid = false; break;
        }
        CandidateAssignment a;
        a.values[id] = candidates[axis]; a.values[storage] = stores.front().value;
        auto cost = accessUnits(g,c,a,nativeVector);
        if (failed(cost)) { valid = false; break; }
        units[axis] = *cost;
      }
      if (valid && units[0] != units[1]) {
        insert(id,candidates[units[1] < units[0]]);
        proposed = true;
      }
    }
  }
  if (!proposed) return false;
  bool changed;
  do {
    changed = false;
    for (const auto &c : constraints) {
      if (c.kind != ConstraintKind::SameLayout && c.kind != ConstraintKind::Convertible)
        continue;
      for (auto from : c.vars) for (auto to : c.vars) {
        if (from == to) continue;
        for (Attribute a : seeds[from]) changed |= insert(to,a);
      }
    }
  } while (changed);
  for (auto &v : g.getVariables()) {
    if (v.kind != LayoutKind::Distributed || bound[v.id] || seeds[v.id].empty()) continue;
    SmallVector<LayoutCandidate> domain;
    for (Attribute a : seeds[v.id]) {
      if (failed(target.verifyCandidate(v,a,mod.getLoc()))) continue;
      auto origin = g.addProvenance(std::nullopt, v.anchor ? v.anchor : mod.getOperation(),
                                    "memory-orientation", "pointwise access or compatible bound role");
      domain.push_back({a,origin,v.id});
    }
    if (!domain.empty()) v.candidates = std::move(domain);
  }
  return true;
}

// For a hardware prefix H, rank(M_H) - rank(dropAxis(M_H)) is the
// number of independent reduction directions available without leaving H.
// This counts contributions, including for mixed GF(2) columns, not replicas.
struct FiberRanks {
  unsigned fiber, outputs;
};
FiberRanks reductionRanks(const LinearExecutionLayout &layout, unsigned axisStart,
                          unsigned axisWidth, unsigned hardwareBits) {
  if (!hardwareBits)
    return {0, 0};
  const auto mask = llvm::APInt::getLowBitsSet(layout.hardwareBits, hardwareBits);
  SmallVector<llvm::APInt> all, kept;
  for (auto [i, row] : llvm::enumerate(layout.rows)) {
    all.push_back(row & mask);
    if (i < axisStart || i >= axisStart + axisWidth)
      kept.push_back(all.back());
  }
  auto rank = [&](ArrayRef<llvm::APInt> rows) -> unsigned {
    if (rows.empty()) return 0;
    return GF2Matrix::get(rows.size(), layout.hardwareBits, rows)->rank();
  };
  unsigned outputs = rank(kept);
  return {rank(all) - outputs, outputs};
}

FailureOr<int64_t> reductionService(const LayoutConstraintGraph &graph,
                                    const LayoutConstraint &c, Attribute source) {
  auto type = cast<RankedTensorType>(graph.getVariable(c.vars.front()).shapedType);
  auto encoding = cast<DistributedEncodingAttr>(source);
  LinearExecutionLayout layout;
  if (analyzeLinearExecutionLayout(encoding, type, layout).status != ProofStatus::Proven)
    return failure();
  unsigned axis = c.reduction->axis;
  unsigned first = 0;
  for (unsigned i = 0; i < axis; ++i) first += layout.logicalWidths[i];
  unsigned width = layout.logicalWidths[axis];
  if (!width) return int64_t{0};
  auto group = reductionRanks(layout, first, width, layout.hardwareOffsets[3]);
  if (group.fiber > width) return failure();
  unsigned levels = width - group.fiber;
  if (layout.topology[3] == 1) {
    auto thread = reductionRanks(layout, first, width, layout.hardwareOffsets[1]);
    auto warp = reductionRanks(layout, first, width, layout.hardwareOffsets[2]);
    if (levels || warp.fiber < thread.fiber || group.fiber < warp.fiber ||
        thread.outputs >= 63)
      return failure();
    const unsigned laneLevels = warp.fiber - thread.fiber;
    const unsigned sharedLevels = group.fiber - warp.fiber;
    // One warp shuffle per projected register per lane stage. A cross-warp
    // stage publishes and reloads partials and synchronizes their lifetime.
    // These are warp-service proxies; lowering can combine tree levels.
    const int64_t registers = int64_t{1} << thread.outputs;
    int64_t work;
    if (llvm::MulOverflow(registers, int64_t(laneLevels + 2 * sharedLevels), work) ||
        llvm::AddOverflow(work, int64_t(2 * sharedLevels), work) ||
        llvm::MulOverflow(work, int64_t(layout.topology[2]), work) ||
        llvm::MulOverflow(work, frequency(c.reduction->source), work))
      return failure();
    return work;
  }
  if (!levels) return int64_t{0};
  if (levels >= 31 || (uint64_t{1} << levels) > layout.topology[3]) return failure();
  // One publish and one remote read per partial, two CTA barrier issues per
  // participating warp at each tree level. Units are warp service estimates.
  int64_t outputs = type.getNumElements() / type.getDimSize(axis);
  int64_t warps = layout.topology[2] * layout.topology[3];
  int64_t packets = llvm::divideCeil(outputs, int64_t{32});
  int64_t merges = (int64_t{1} << levels) - 1;
  int64_t weight = frequency(c.reduction->source);
  if (packets > (INT64_MAX / weight - 2 * warps * levels) / (2 * merges)) return failure();
  return weight * (2 * packets * merges + 2 * warps * levels);
}

struct Domain {
  SmallVector<Attribute> values;
  std::vector<z3::expr> selected;
};
struct Model {
  ModuleOp module;
  LayoutConstraintGraph &graph;
  LayoutTarget &target;
  bool scoreReduction = true;
  RoutePrice price;
  z3::context ctx;
  z3::optimize solver{ctx};
  z3::expr access{ctx.int_val(0)}, movement{ctx.int_val(0)}, reduction{ctx.int_val(0)},
      spill{ctx.int_val(0)}, routes{ctx.int_val(0)};
  std::vector<Domain> vars;
  int64_t fixedBits = 0, freeBits = 0, bitDecisions = 0, next = 0;
  SmallVector<Attribute> masks;
  DenseMap<std::pair<Type, Type>, int64_t> prices;
  bool nativeVector = false;
  Model(ModuleOp m, LayoutConstraintGraph &g, LayoutTarget &t, RoutePrice p)
      : module(m), graph(g), target(t), price(p) {}
  z3::expr bit() {
    return ctx.bool_const(("b" + std::to_string(next++)).c_str());
  }
  z3::expr number(int64_t v) { return ctx.int_val(std::to_string(v).c_str()); }
  z3::expr charge(const z3::expr &flag, int64_t amount) {
    return z3::ite(flag, number(amount), ctx.int_val(0));
  }
  z3::expr pick(LayoutVarID id, Attribute a) {
    auto &v = vars[id];
    auto it = llvm::find(v.values, a);
    return it == v.values.end() ? ctx.bool_val(false)
                                : v.selected[it - v.values.begin()];
  }
  // F2 bits describe ownership only with the same carrier names, widths,
  // replication, and coordinate axes. Different metadata needs a table tag.
  bool matrixDomain(const LayoutVar &var, Domain &d) {
    auto type = dyn_cast<RankedTensorType>(var.shapedType);
    if (!type || var.kind != LayoutKind::Distributed)
      return false;
    SmallVector<LinearExecutionLayout, 2> maps;
    BitLinearLayoutMapAttr first;
    DistributedEncodingAttr base;
    for (auto a : d.values) {
      auto e = dyn_cast<DistributedEncodingAttr>(a);
      auto map = e ? dyn_cast<BitLinearLayoutMapAttr>(e.getMap())
                   : BitLinearLayoutMapAttr();
      if (!map)
        return false;
      if (!first) {
        first = map;
        base = e;
      }
      if (e.getTopology() != base.getTopology() ||
          e.getReplication() != base.getReplication() ||
          map.getInputNames() != first.getInputNames() ||
          map.getOutputNames() != first.getOutputNames() ||
          map.getInputBitWidths() != first.getInputBitWidths() ||
          map.getOutputBitWidths() != first.getOutputBitWidths())
        return false;
      LinearExecutionLayout decoded;
      if (analyzeLinearExecutionLayout(e, type, decoded).status !=
          ProofStatus::Proven)
        return false;
      maps.push_back(std::move(decoded));
    }
    auto &m = maps.front();
    std::map<std::string, z3::expr> unique;
    std::vector<z3::expr> selected(maps.size(), ctx.bool_val(true));
    SmallVector<int64_t> ones, free;
    for (unsigned r = 0; r < m.logicalBits; ++r)
      for (unsigned c = 0; c < m.hardwareBits; ++c) {
        std::string pattern;
        for (auto &candidate : maps)
          pattern.push_back(candidate.rows[r][c] ? '1' : '0');
        int64_t index = r * m.hardwareBits + c;
        if (pattern.find_first_not_of(pattern.front()) == std::string::npos) {
          ++fixedBits;
          if (pattern.front() == '1')
            ones.push_back(index);
          continue;
        }
        ++freeBits;
        free.push_back(index);
        if (unique.count(pattern))
          continue;
        auto v = bit();
        unique.emplace(pattern, v);
        ++bitDecisions;
        for (unsigned j = 0; j < maps.size(); ++j)
          selected[j] = selected[j] && (pattern[j] == '1' ? v : !v);
      }
    if (var.instructionRole) {
      Builder b(module.getContext());
      NamedAttrList mask;
      mask.set("variable", b.getStringAttr(var.stableName));
      mask.set("shape",
               b.getDenseI64ArrayAttr({m.logicalBits, m.hardwareBits}));
      mask.set("one", b.getDenseI64ArrayAttr(ones));
      mask.set("free", b.getDenseI64ArrayAttr(free));
      masks.push_back(mask.getDictionary(b.getContext()));
    }
    d.selected = std::move(selected);
    return true;
  }
  LogicalResult buildDomains() {
    for (auto &v : graph.getVariables()) {
      auto &d = vars.emplace_back();
      // A tensor type and a fixed access can independently require the same
      // encoding. The exactly-one indicators range over distinct layouts.
      for (auto &candidate : v.candidates)
        if (!llvm::is_contained(d.values, candidate.value))
          d.values.push_back(candidate.value);
      if (d.values.empty())
        return module.emitError("empty protected integer domain: ")
               << v.stableName;
      if (!matrixDomain(v, d)) {
        if (d.values.size() == 1)
          d.selected.push_back(ctx.bool_val(true));
        else
          for (size_t i = 0; i < d.values.size(); ++i)
            d.selected.push_back(bit());
      }
      z3::expr_vector indicators(ctx);
      for (auto &x : d.selected)
        indicators.push_back(x);
      if (indicators.size() > 1) {
        solver.add(z3::atleast(indicators, 1));
        solver.add(z3::atmost(indicators, 1));
      }
    }
    return success();
  }
  z3::expr row(ArrayRef<LayoutVarID> ids, ArrayRef<Attribute> encodings) {
    z3::expr match = ctx.bool_val(true);
    for (auto [id, encoding] : llvm::zip_equal(ids, encodings))
      match = match && pick(id, encoding);
    return match;
  }
  void constrain(const LayoutConstraint &c) {
    if (c.strength != ConstraintStrength::Hard)
      return;
    if (c.instruction || c.reduction) {
      z3::expr_vector allowed(ctx);
      if (c.instruction) {
        for (auto &t : c.instruction->tuples)
          allowed.push_back(row(c.vars, t.encodings));
      } else {
        for (auto &p : c.reduction->pairs)
          allowed.push_back(row(c.vars, {p.sourceEncoding, p.resultEncoding}));
      }
      solver.add(z3::mk_or(allowed));
      return;
    }
    auto src = c.vars.front();
    if (c.vars.size() == 1) {
      for (auto a : vars[src].values) {
        DenseMap<LayoutVarID, Attribute> assignment{{src, a}};
        if (!satisfiesLayoutConstraint(graph, c, assignment, true))
          solver.add(!pick(src, a));
      }
      return;
    }
    for (auto dst : ArrayRef<LayoutVarID>(c.vars).drop_front()) {
      for (auto a : vars[src].values) {
        z3::expr_vector support(ctx);
        for (auto b : vars[dst].values)
          if (layoutRelationCompatible(graph, c, src, a, dst, b))
            support.push_back(pick(dst, b));
        solver.add(z3::implies(pick(src, a), z3::mk_or(support)));
      }
    }
  }

  // Whole-kernel local traffic can be assigned to a layout only when that
  // layout determines the complete placement (a pinned, single-use Sum).
  struct CompiledSpill {
    int64_t sectors;
    StringAttr cubin;
  };
  std::optional<LayoutVarID> spillInput;
  DenseMap<Attribute, CompiledSpill> compiledArms;

  LogicalResult bindCompiledSpill() {
    auto attr = module->getAttr("frisk.compiled_sum_spill");
    if (!attr) return success();
    auto profile = dyn_cast<DictionaryAttr>(attr);
    auto error = [&](StringRef message) -> LogicalResult {
      return module.emitError() << "compiled Sum spill profile: " << message;
    };
    if (!profile) return error("expected a dictionary");
    Builder b(module.getContext());
    auto matchesI64 = [&](StringRef key, int64_t n) {
      return profile.get(key) == b.getI64IntegerAttr(n);
    };
    if (profile.get("backend") != b.getStringAttr("gluon34") ||
        !matchesI64("stages", 1) || !matchesI64("threads", 128))
      return error("expected the single-stage, four-warp Gluon backend");
    for (const auto &c : graph.getConstraints()) {
      if (!c.reduction) continue;
      if (spillInput) return error("requires exactly one reduction");
      auto op = dyn_cast<ReduceTensorOp>(c.reduction->source);
      if (!op || op.getKind() != "sum" || !op.getSource().hasOneUse() ||
          !op.getResult().hasOneUse() || !matchesI64("axis", c.reduction->axis))
        return error("reduction or axis does not match");
      auto load = op.getSource().getDefiningOp<TileLoadOp>();
      auto store = dyn_cast<TileStoreOp>(*op.getResult().getUsers().begin());
      if (!load || !store || load->getBlock() != op->getBlock() ||
          store->getBlock() != op->getBlock() ||
          !load->hasAttr("frisk.fixed_access_layout") ||
          !store->hasAttr("frisk.fixed_access_layout") ||
          load->getAttr("frisk.fixed_access_layout") != profile.get("read") ||
          store->getAttr("frisk.fixed_access_layout") != profile.get("write"))
        return error("expected matching fixed read and write layouts");
      auto readView = load.getSource().getDefiningOp<LayoutViewOp>();
      auto writeView = store.getTarget().getDefiningOp<LayoutViewOp>();
      if (!readView || !writeView || !profile.get("read_storage") ||
          !profile.get("write_storage") ||
          readView->getAttr("layout") != profile.get("read_storage") ||
          writeView->getAttr("layout") != profile.get("write_storage"))
        return error("storage mappings do not match");
      auto type = cast<RankedTensorType>(op.getSource().getType());
      if (!type.getElementType().isF32() || type.getRank() != 2 ||
          graph.getVariable(c.vars[0]).requiredThreads != 128 ||
          frequency(op) != 1)
        return error("requires one fixed rank-two FP32 Sum per CTA");
      spillInput = c.vars[0];
    }
    if (!spillInput) return error("missing Sum");
    unsigned loads = 0, stores = 0;
    module.walk([&](TileLoadOp) { ++loads; });
    module.walk([&](TileStoreOp) { ++stores; });
    if (loads != 1 || stores != 1) return error("requires one load and one store");
    auto arms = profile.getAs<ArrayAttr>("arms");
    auto &domain = vars[*spillInput].values;
    if (!arms || arms.size() != domain.size())
      return error("profile must cover the complete candidate domain");
    for (Attribute entry : arms) {
      auto record = dyn_cast<DictionaryAttr>(entry);
      if (!record) return error("invalid candidate record");
      auto encoding = record.get("layout");
      auto digest = record.getAs<StringAttr>("cubin");
      auto loadWords = record.getAs<IntegerAttr>("load_words");
      auto storeWords = record.getAs<IntegerAttr>("store_words");
      auto validWords = [](IntegerAttr n) {
        return n && n.getType().isInteger(64) && n.getInt() >= 0;
      };
      if (!llvm::is_contained(domain, encoding) || compiledArms.count(encoding))
        return error("duplicate or unrecognized candidate layout");
      if (!digest || digest.getValue().size() != 64 ||
          !llvm::all_of(digest.getValue(), [](char v) {
            return (v >= '0' && v <= '9') || (v >= 'a' && v <= 'f');
          }) || !validWords(loadWords) || !validWords(storeWords))
        return error("invalid cubin digest or local-memory word count");
      int64_t words, sectors;
      // Local memory interleaves successive lanes at 32-bit word granularity.
      // Four warps, four 32-byte sectors per full warp-word. Count each
      // lowered access once at kernel scope, independently of conversion CSE.
      if (llvm::AddOverflow(loadWords.getInt(), storeWords.getInt(), words) ||
          llvm::MulOverflow(words, int64_t{128 * 4 / 32}, sectors))
        return error("local-memory traffic overflow");
      compiledArms.try_emplace(encoding, CompiledSpill{sectors, digest});
      spill = spill + charge(pick(*spillInput, encoding), sectors);
    }
    return success();
  }

  LogicalResult reductionFactor(const LayoutConstraint &c) {
    for (auto source : vars[c.vars[0]].values) {
      auto units = reductionService(graph, c, source);
      if (failed(units)) {
        // Other certified layouts for this tensor can still be executable.
        solver.add(!pick(c.vars[0], source));
        continue;
      }
      if (*units) reduction = reduction + charge(pick(c.vars[0], source), *units);
    }
    return success();
  }
  LogicalResult accessFactor(const LayoutConstraint &c) {
    const LayoutConstraint *copy = nullptr;
    for (auto id : c.vars) {
      const auto &v = graph.getVariable(id);
      if (!v.operationExecution || !isa_and_nonnull<CopyOp>(v.anchor))
        continue;
      for (const auto &other : graph.getConstraints())
        if (other.instruction && other.instruction->source == v.anchor) {
          copy = &other;
          break;
        }
    }
    auto add = [&](CandidateAssignment &assignment,
                   const z3::expr &flag) -> LogicalResult {
      auto units = accessUnits(graph, c, assignment, nativeVector);
      if (failed(units))
        return module.emitError("unsupported multi-anchor access cost");
      if (*units)
        access = access + charge(flag, *units);
      return success();
    };
    if (copy) {
      for (auto &t : copy->instruction->tuples) {
        CandidateAssignment assignment;
        for (auto [id, a] : llvm::zip_equal(copy->vars, t.encodings))
          assignment.values[id] = a;
        for (auto id : c.vars)
          if (!assignment.values.count(id))
            return module.emitError("missing copy access role");
        if (failed(add(assignment, row(copy->vars, t.encodings))))
          return failure();
      }
    } else {
      auto x = c.vars[0], y = c.vars[1];
      for (auto a : vars[x].values)
        for (auto b : vars[y].values) {
          CandidateAssignment assignment;
          assignment.values[x] = a;
          assignment.values[y] = b;
          if (failed(add(assignment, pick(x, a) && pick(y, b))))
            return failure();
        }
    }
    return success();
  }
  int64_t routeUnits(RankedTensorType type, Attribute a, Attribute b) {
    auto src = RankedTensorType::get(type.getShape(), type.getElementType(), a);
    auto dst = RankedTensorType::get(type.getShape(), type.getElementType(), b);
    auto key = std::make_pair(Type(src), Type(dst));
    auto cached = prices.find(key);
    if (cached == prices.end()) {
      auto work = price(src, dst);
      cached = prices.try_emplace(key, succeeded(work) ? *work : -1).first;
    }
    return cached->second;
  }
  LogicalResult boundaryFactors() {
    // Exactly the native SharedOnly grouping: SSA source, target mapping and
    // insertion block. Existing explicit converts each keep their own charge.
    using Key = std::tuple<LayoutVarID, Attribute, Attribute, Block *, int64_t>;
    struct Less {
      bool operator()(const Key &l, const Key &r) const {
        auto raw = [](const Key &k) {
          return std::make_tuple(std::get<0>(k),
                                 std::get<1>(k).getAsOpaquePointer(),
                                 std::get<2>(k).getAsOpaquePointer(),
                                 std::get<3>(k), std::get<4>(k));
        };
        return raw(l) < raw(r);
      }
    };
    struct Group {
      z3::expr active;
      int64_t units;
    };
    std::map<Key, size_t, Less> indices;
    std::vector<Group> groups;
    for (const auto &c : graph.getConstraints()) {
      if (c.kind != ConstraintKind::Convertible)
        continue;
      auto x = c.vars[0], y = c.vars[1];
      auto type = cast<RankedTensorType>(graph.getVariable(x).shapedType);
      for (auto a : vars[x].values)
        for (auto b : vars[y].values) {
          if (layoutEncodingsEqual(a, b))
            continue;
          auto unitsPerUse = routeUnits(type, a, b);
          auto flag = pick(x, a) && pick(y, b);
          if (unitsPerUse < 0) {
            solver.add(!flag);
            continue;
          }
          auto *at = c.use->getOwner();
          int64_t weight = frequency(at);
          if (unitsPerUse > INT64_MAX / weight)
            return module.emitError("multi-anchor route cost overflow");
          int64_t units = unitsPerUse * weight;
          Key k{x, a, b, at->getBlock(),
                c.existingConversion ? int64_t(c.id) : -1};
          auto insertion = indices.emplace(k, groups.size());
          if (insertion.second)
            groups.push_back({flag, units});
          else {
            auto &group = groups[insertion.first->second];
            group.active = group.active || flag;
          }
        }
    }
    for (auto &g : groups) {
      movement = movement + charge(g.active, g.units);
      routes = routes + charge(g.active, 1);
    }
    return success();
  }
};
// Reprice a concrete assignment through the native materializer's list of
// sharing leaders. This traversal is independent of the Z3 factor tables.
struct Bill {
  int64_t access = 0, movement = 0, reduction = 0, spill = 0, count = 0;
  auto objective() const {
    return std::pair<int64_t, int64_t>(access + movement + reduction + spill, count);
  }
};

std::optional<Bill> billWitness(Model &m, const CandidateAssignment &assignment,
                                const LayoutConversionChoices &choices) {
  Bill bill;
  if (m.spillInput) {
    auto arm = m.compiledArms.find(assignment.values.lookup(*m.spillInput));
    if (arm == m.compiledArms.end()) return std::nullopt;
    bill.spill = arm->second.sectors;
  }
  auto chargeRoute = [&](const LayoutConstraint &c) {
    auto a = assignment.values.lookup(c.vars[0]);
    auto b = assignment.values.lookup(c.vars[1]);
    auto type =
        cast<RankedTensorType>(m.graph.getVariable(c.vars[0]).shapedType);
    auto work = m.routeUnits(type, a, b);
    if (work < 0)
      return false;
    int64_t weight = frequency(c.use->getOwner());
    if (work > (INT64_MAX - bill.movement) / weight)
      return false;
    bill.movement += work * weight;
    ++bill.count;
    return true;
  };
  for (const auto &c : m.graph.getConstraints()) {
    if (m.scoreReduction && c.reduction) {
      auto work = reductionService(m.graph, c, assignment.values.lookup(c.vars[0]));
      if (failed(work) || *work > INT64_MAX - bill.reduction) return std::nullopt;
      bill.reduction += *work;
    }
    if (c.kind == ConstraintKind::StorageAccess) {
      auto amount = accessUnits(m.graph, c, assignment, m.nativeVector);
      if (failed(amount) || *amount > INT64_MAX - bill.access)
        return std::nullopt;
      bill.access += *amount;
    }
    if (c.kind == ConstraintKind::Convertible && c.existingConversion &&
        !layoutEncodingsEqual(assignment.values.lookup(c.vars[0]),
                              assignment.values.lookup(c.vars[1])) &&
        !chargeRoute(c))
      return std::nullopt;
  }
  for (const auto &e : choices.edges)
    if (e.constraint == e.sharedLeader &&
        !chargeRoute(m.graph.getConstraint(e.constraint)))
      return std::nullopt;
  if (bill.access > INT64_MAX - bill.movement ||
      bill.access + bill.movement > INT64_MAX - bill.reduction ||
      bill.access + bill.movement + bill.reduction > INT64_MAX - bill.spill)
    return std::nullopt;
  return bill;
}

// Small diagnostic oracle. Enumerate actual encodings, pruning only with the
// existing hard relation verifier. Never consult Z3 indicators or objectives.
LogicalResult verifyByEnumeration(Model &m, unsigned limit,
                                  const Bill &selected, unsigned &nodes,
                                  unsigned &witnesses,
                                  std::optional<int64_t> &noConversion) {
  if (!limit)
    return success();
  CandidateAssignment a;
  SmallVector<LayoutVarID> order;
  for (const auto &v : m.graph.getVariables()) {
    if (v.candidates.size() == 1)
      a.values[v.id] = v.candidates.front().value;
    else
      order.push_back(v.id);
  }
  llvm::stable_sort(order, [&](auto x, auto y) {
    return m.graph.getVariable(x).candidates.size() <
           m.graph.getVariable(y).candidates.size();
  });
  std::optional<std::pair<int64_t, int64_t>> best;
  auto visit = [&](auto &&self, size_t level) -> void {
    if (nodes >= limit) {
      nodes = limit + 1;
      return;
    }
    ++nodes;
    for (const auto &c : m.graph.getConstraints())
      if (!satisfiesLayoutConstraint(m.graph, c, a.values,
                                     level == order.size()))
        return;
    if (level < order.size()) {
      auto id = order[level];
      for (const auto &v : m.graph.getVariable(id).candidates) {
        a.values[id] = v.value;
        self(self, level + 1);
        if (nodes > limit)
          break;
      }
      a.values.erase(id);
      return;
    }
    if (m.target.verifyExecutionSupport(m.graph, a).status !=
        ProofStatus::Proven)
      return;
    auto edges = chooseLayoutConversions(m.graph, a, m.target, nullptr,
                                         ConversionPolicy::SharedOnly);
    if (failed(edges))
      return;
    auto bill = billWitness(m, a, *edges);
    if (!bill)
      return;
    ++witnesses;
    if (bill->count == 0 && (!noConversion || bill->objective().first < *noConversion))
      noConversion = bill->objective().first;
    if (!best || bill->objective() < *best)
      best = bill->objective();
  };
  visit(visit, 0);
  if (nodes > limit)
    return m.module.emitError(
        "multi-anchor enumeration audit reached its node limit");
  if (!best || *best != selected.objective())
    return m.module.emitError(
        "independent enumeration disagrees with the integer optimum");
  return success();
}
} // namespace
#endif

FailureOr<DictionaryAttr> mlir::frisk::solveCertifiedMultiAnchor(
    ModuleOp module, const CertifiedInstructionDomains &domains,
    RoutePrice price, StoreMap storeMap, unsigned freeBits, unsigned timeoutMs,
    unsigned oracleLimit, bool scoreReduction, bool nativeVector) {
#ifndef FRISK_HAS_Z3
  return module.emitError(
      "multi-anchor integer solver requires Z3; configure FRISK_Z3_ROOT");
#else
  auto target = createSM90LayoutTarget();
  auto graph = collectLayoutConstraints(
      module, *target, LayoutCollectionMode::InitializeCandidates, nullptr,
      nullptr, &domains);
  if (failed(graph))
    return failure();
  bool memorySeeds = domains.empty() && seedPointwiseAccesses(*graph, *target, module, nativeVector);
  // Pointwise stores already propose layouts through their storage access.
  // Honor the ordinary F2 expansion budget independently of that choice.
  auto completions = addCompletions(*graph, *target, module, freeBits, storeMap,
                                    /*offerConsecutiveWriters=*/!memorySeeds);
  if (failed(propagateCommonToFixedPoint(*graph)))
    return failure();
  Model m(module, *graph, *target, price);
  m.scoreReduction = scoreReduction;
  m.nativeVector = nativeVector;
  if (failed(m.buildDomains()) || failed(m.bindCompiledSpill()))
    return failure();
  for (const auto &c : graph->getConstraints()) {
    m.constrain(c);
    if (scoreReduction && c.reduction && failed(m.reductionFactor(c))) return failure();
    if (c.kind == ConstraintKind::StorageAccess && failed(m.accessFactor(c)))
      return failure();
  }
  if (failed(m.boundaryFactors()))
    return failure();
  z3::params p(m.ctx);
  p.set("timeout", timeoutMs);
  m.solver.set(p);
  auto service = m.solver.minimize(m.access + m.movement + m.reduction + m.spill);
  auto unique = m.solver.minimize(m.routes);
  auto status = m.solver.check();
  if (status == z3::unsat)
    return module.emitError(
        "no feasible assignment in the certified domains and native routes");
  if (status == z3::unknown)
    return module.emitError("multi-anchor integer solve incomplete: ")
           << Z3_optimize_get_reason_unknown(m.ctx, m.solver);
  auto model = m.solver.get_model();
  auto optimal = [&](auto h, z3::expr objective) {
    return z3::eq(m.solver.lower(h), m.solver.upper(h)) &&
           z3::eq(model.eval(objective, true).simplify(),
                  m.solver.lower(h).simplify());
  };
  if (!optimal(service, m.access + m.movement + m.reduction + m.spill) || !optimal(unique, m.routes))
    return module.emitError(
        "multi-anchor integer objective has no optimality certificate");

  LayoutSolution solution;
  solution.policy.conversion = ConversionPolicy::SharedOnly;
  for (auto [id, d] : llvm::enumerate(m.vars))
    for (auto [j, sel] : llvm::enumerate(d.selected))
      if (model.eval(sel, true).is_true())
        solution.assignments[id] = d.values[j];
  CandidateAssignment assignment;
  assignment.values = solution.assignments;
  for (const auto &c : graph->getConstraints()) {
    if (!satisfiesLayoutConstraint(*graph, c, solution.assignments, true))
      return module.emitError("invalid integer witness");
    if (c.instruction) {
      auto tuple = findInstructionSupport(*graph, c, solution.assignments);
      if (!tuple)
        return failure();
      solution.instructionBindings[c.id] = tuple->binding;
    }
    if (c.reduction) {
      auto pair = findReductionSupport(*graph, c, solution.assignments);
      if (!pair)
        return failure();
      solution.reductionBindings[c.id] = pair->binding;
    }
  }
  if (target->verifyExecutionSupport(*graph, assignment).status !=
      ProofStatus::Proven)
    return module.emitError("integer witness outside target execution domain");
  auto conversion = chooseLayoutConversions(
      *graph, assignment, *target, nullptr, solution.policy.conversion);
  auto cost =
      scoreExperimentAssignment(*graph, assignment, *target, solution.policy);
  if (failed(conversion) || failed(cost) || cost->saturated)
    return failure();
  auto bill = billWitness(m, assignment, *conversion);
  if (!bill || bill->access != model.eval(m.access, true).get_numeral_int64() ||
      bill->movement != model.eval(m.movement, true).get_numeral_int64() ||
      bill->reduction != model.eval(m.reduction,true).get_numeral_int64() ||
      bill->spill != model.eval(m.spill,true).get_numeral_int64() ||
      bill->count != model.eval(m.routes, true).get_numeral_int64())
    return module.emitError(
        "physical conversion or access bill disagrees with integer model");
  unsigned nodes = 0, witnesses = 0;
  std::optional<int64_t> noConversion;
  if (failed(verifyByEnumeration(m, oracleLimit, *bill, nodes, witnesses,
                                 noConversion)))
    return failure();
  solution.conversions = std::move(conversion->edges);
  solution.cost = *cost;
  auto count = [&](const auto &added) {
    int64_t offered = 0, used = 0;
    for (auto &entry : added) {
      for (auto a : entry.second)
        offered += llvm::is_contained(m.vars[entry.first].values, a);
      used += llvm::is_contained(entry.second,
                                 solution.assignments.lookup(entry.first));
    }
    return std::pair<int64_t, int64_t>{offered, used};
  };
  auto matrices = count(completions.matrices),
       writers = count(completions.writers);
  if (failed(materializeLayouts(module, *graph, solution, false, target.get(),
                                solution.policy.conversion)) ||
      failed(verifyMaterializedLayouts(module, *target)))
    return failure();
  Builder b(module.getContext());
  NamedAttrList receipt;
  receipt.set("solver", b.getStringAttr("certified-f2-integer-v1"));
  receipt.set("reduction_units",b.getI64IntegerAttr(bill->reduction));
  if (m.spillInput) {
    const auto &arm = m.compiledArms.find(assignment.values.lookup(*m.spillInput))->second;
    receipt.set("compiled_spill_units", b.getI64IntegerAttr(bill->spill));
    receipt.set("compiled_spill_cubin", arm.cubin);
  }
  if (memorySeeds) {
    SmallVector<Attribute> roles;
    for (const auto &v : graph->getVariables()) {
      if (v.kind != LayoutKind::Distributed) continue;
      NamedAttrList info;
      info.set("role", b.getStringAttr(v.stableName));
      info.set("layouts", b.getArrayAttr(m.vars[v.id].values));
      roles.push_back(info.getDictionary(b.getContext()));
    }
    receipt.set("memory_domains", b.getArrayAttr(roles));
  }
  receipt.set("optimal", b.getBoolAttr(true));
  receipt.set(
      "access_units",
      b.getI64IntegerAttr(model.eval(m.access, true).get_numeral_int64()));
  receipt.set(
      "conversion_units",
      b.getI64IntegerAttr(model.eval(m.movement, true).get_numeral_int64()));
  receipt.set(
      "distinct_conversions",
      b.getI64IntegerAttr(model.eval(m.routes, true).get_numeral_int64()));
  receipt.set("fixed_coefficients", b.getI64IntegerAttr(m.fixedBits));
  receipt.set("free_coefficients", b.getI64IntegerAttr(m.freeBits));
  receipt.set("coefficient_variables", b.getI64IntegerAttr(m.bitDecisions));
  receipt.set("completion_bits_limit", b.getI64IntegerAttr(freeBits));
  receipt.set("f2_completions", b.getI64IntegerAttr(matrices.first));
  receipt.set("f2_completions_selected", b.getI64IntegerAttr(matrices.second));
  receipt.set("writer_candidates", b.getI64IntegerAttr(writers.first));
  receipt.set("writer_candidates_selected",
              b.getI64IntegerAttr(writers.second));
  if (oracleLimit) {
    if (noConversion)
      receipt.set("best_without_conversions",
                  b.getI64IntegerAttr(*noConversion));
    receipt.set("enumeration_nodes", b.getI64IntegerAttr(nodes));
    receipt.set("enumeration_feasible", b.getI64IntegerAttr(witnesses));
  }
  receipt.set("wide_completion_domains",
              b.getI64IntegerAttr(completions.skippedWide));
  receipt.set("anchor_masks", b.getArrayAttr(m.masks));
  return receipt.getDictionary(b.getContext());
#endif
}
