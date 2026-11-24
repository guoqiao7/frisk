#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "Dialect/Frisk/Target/SM90/SM90InstructionTable.h"
#include "Dialect/Frisk/Target/SM90/SM90TargetInfo.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <map>

using namespace mlir;
using namespace mlir::frisk;
namespace {
LayoutProof reject(StringRef reason, ArrayRef<int64_t> p = {}) {
  return {ProofStatus::Disproven, SmallVector<int64_t>(p), reason.str()};
}
LayoutProof unknown(StringRef reason) { return {ProofStatus::Unknown, {}, reason.str()}; }
bool bounded(int64_t a, int64_t b) {
  return a > 1 && b > 1 && llvm::isPowerOf2_64(a) && llvm::isPowerOf2_64(b) &&
         a <= 65536 && b <= 65536 / a;
}
}

FailureOr<SM90MmaGeometry> mlir::frisk::getSM90MmaGeometry(MmaOp op, int64_t threads) {
  if (op.getPolicy() == attr::GemmWarpPolicy::Auto) return failure();
  int64_t m = op.getMAttr().getInt(), n = op.getNAttr().getInt(), k = op.getKAttr().getInt();
  if (!isSM90WGMMAThreadCount(threads) ||
      !bounded(m,k) || !bounded(n,k) || m % 64 || k % 16 || n < 8)
    return failure();
  int64_t bestM = 0, bestN = 0, score = std::numeric_limits<int64_t>::max();
  for (int64_t gm = 1; gm <= threads / 128; gm *= 2) {
    int64_t gn = threads / 128 / gm;
    if (m % (64*gm) || n % (8*gn)) continue;
    int64_t s = std::abs(m/gm - n/gn);
    if (op.getPolicy() == attr::GemmWarpPolicy::FullRow) s = -gm;
    if (op.getPolicy() == attr::GemmWarpPolicy::FullCol) s = -gn;
    if (s < score) { bestM = gm; bestN = gn; score = s; }
  }
  if (!bestM) return failure();
  int64_t atomN = 256;
  while ((n / bestN) % atomN) atomN /= 2;
  bool rs = isa<RankedTensorType>(op.getA().getType());
  if (!lookupSM90WGMMA(cast<ShapedType>(op.getA().getType()).getElementType(),
                      cast<ShapedType>(op.getResult().getType()).getElementType(),
                      rs ? "rs" : "ss", 64, atomN, 16, threads, "k", "k"))
    return failure();
  return SM90MmaGeometry{m,n,k,threads,bestM,bestN,atomN,
                        m/(64*bestM),n/(atomN*bestN),k/16};
}

// Semantic instruction geometries, independent of the generated proof domain.
// Explicit bindings must still reach the fragment proof when replication puts
// an otherwise valid geometry beyond its hardware enumeration budget.
static FailureOr<SmallVector<SM90MmaGeometry>> getSM90MmaBaseGeometries(
    MmaOp op, int64_t threads) {
  if (op.getPolicy() != attr::GemmWarpPolicy::Auto) {
    auto legacy = getSM90MmaGeometry(op, threads);
    if (failed(legacy)) return failure();
    return SmallVector<SM90MmaGeometry>{*legacy};
  }
  int64_t m = op.getM(), n = op.getN(), k = op.getK();
  if (!isSM90WGMMAThreadCount(threads) || !bounded(m,k) || !bounded(n,k) || m % 64 || k % 16 || n < 8)
    return failure();
  SmallVector<SM90MmaGeometry> result;
  bool rs = isa<RankedTensorType>(op.getA().getType());
  for (int64_t gm = 1; gm <= threads/128; gm *= 2) {
    int64_t gn = threads/128/gm;
    if (m % (64*gm) || n % (8*gn)) continue;
    int64_t atomN = 256;
    while ((n/gn) % atomN) atomN /= 2;
    if (!lookupSM90WGMMA(cast<ShapedType>(op.getA().getType()).getElementType(),
        cast<ShapedType>(op.getResult().getType()).getElementType(),
        rs ? "rs" : "ss",64,atomN,16,threads,"k","k")) continue;
    result.push_back({m,n,k,threads,gm,gn,atomN,m/(64*gm),n/(atomN*gn),k/16});
  }
  if (result.empty() || result.size() > 4) return failure();
  return result;
}

static FailureOr<SmallVector<SM90MmaGeometry>> getSM90MmaSemanticGeometries(
    MmaOp op, int64_t threads) {
  auto base = getSM90MmaBaseGeometries(op, threads);
  if (failed(base)) return failure();
  SmallVector<SM90MmaGeometry> result;
  auto pin = op->getAttrOfType<StringAttr>("frisk.row_partition");
  if (op->hasAttr("frisk.row_partition") &&
      (!pin || (pin.getValue() != "contiguous" && pin.getValue() != "interleaved64")))
    return failure();
  for (auto g : *base) {
    if (!pin || pin.getValue() == "contiguous") result.push_back(g);
    if (g.gM == 2 && g.gN == 1 && g.mRepeat > 1 &&
        (!pin || pin.getValue() == "interleaved64")) {
      g.interleavedRows = true;
      result.push_back(g);
    }
  }
  if (result.empty()) return failure();
  return result;
}

FailureOr<SmallVector<SM90MmaGeometry>> mlir::frisk::getSM90MmaGeometries(
    MmaOp op, int64_t threads, bool exploreRowPartitions) {
  auto result = getSM90MmaSemanticGeometries(op,threads);
  if (failed(result)) return failure();
  auto binding = op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
  if (exploreRowPartitions && binding) {
    auto fixed=getSM90MmaContractGeometry(op,threads,binding);
    if (failed(fixed)) return failure();
    llvm::erase_if(*result,[&](const auto &g) {
      return g.gM!=fixed->gM || g.gN!=fixed->gN || g.interleavedRows!=fixed->interleavedRows;
    });
  } else if (!exploreRowPartitions && !op->hasAttr("frisk.row_partition") &&
             !(binding && binding.getPayload().get("row_partition"))) {
    // Keep ordinary generation independent of a fixed grid. The binding proof
    // separately diagnoses legal RS geometries beyond the enumeration budget.
    llvm::erase_if(*result, [](const auto &g) { return g.interleavedRows; });
  }
  if (op.getPolicy() == attr::GemmWarpPolicy::Auto && isa<RankedTensorType>(op.getA().getType()))
    llvm::erase_if(*result,[](const SM90MmaGeometry &g) {
      // Keep the bounded generated domain unchanged, including RS replication.
      return g.m*g.k > 65536/g.gN;
    });
  if (result->empty()) return failure();
  return result;
}

LayoutProof mlir::frisk::diagnoseSM90MmaGeometryFailure(MmaOp op, int64_t threads) {
  int64_t m = op.getM(), n = op.getN(), k = op.getK();
  if (isSM90WGMMAThreadCount(threads) && m > 0 && n > 0 && k > 0 &&
      llvm::isPowerOf2_64(m) && llvm::isPowerOf2_64(n) && llvm::isPowerOf2_64(k) &&
      m % 64 == 0 && n >= 8 && k % 16 == 0) {
    for (bool a : {true, false}) {
      int64_t extent = a ? m : n;
      if (extent > 65536/k)
        return unknown((Twine("sm90-mma-") +
            (a && isa<RankedTensorType>(op.getA().getType()) ? "fragment" : "descriptor") +
            ": unknown proof: " + (a ? "A " : "B ") + Twine(extent) + "x" + Twine(k) +
            " logical point budget exceeds 65536").str());
    }
  }
  return reject((Twine("sm90-mma-thread-group: unsupported binding geometry, tile, threads or policy grouping: ") +
       Twine(m)+"x"+Twine(n)+"x"+Twine(k)+", consumer threads="+Twine(threads)).str());
}

DictionaryAttr mlir::frisk::encodeSM90MmaGeometry(MLIRContext *ctx,
                                                const SM90MmaGeometry &g) {
  Builder b(ctx);
  NamedAttrList geometry({
      b.getNamedAttr("kind", b.getStringAttr("sm90-mma-geometry")),
      b.getNamedAttr("version", b.getI64IntegerAttr(1)),
      b.getNamedAttr("shape", b.getDenseI64ArrayAttr({g.m,g.n,g.k})),
      b.getNamedAttr("threads", b.getI64IntegerAttr(g.threads)),
      b.getNamedAttr("grid", b.getDenseI64ArrayAttr({g.gM,g.gN})),
      b.getNamedAttr("atom", b.getDenseI64ArrayAttr({64,g.atomN,16})),
      b.getNamedAttr("repeats", b.getDenseI64ArrayAttr({g.mRepeat,g.nRepeat,g.kRepeat}))});
  if (g.interleavedRows) geometry.set("row_partition", b.getStringAttr("interleaved64"));
  return geometry.getDictionary(ctx);
}

FailureOr<SM90MmaGeometry> mlir::frisk::decodeSM90MmaGeometry(
    MmaOp op, int64_t threads, Attribute candidate) {
  auto dictionary = dyn_cast_or_null<DictionaryAttr>(candidate);
  if (!dictionary || (dictionary.size() != 7 && dictionary.size() != 8)) return failure();
  auto legal = getSM90MmaSemanticGeometries(op,threads);
  if (failed(legal)) return failure();
  // Exact checked serialization validates every field, type and semantic
  // relationship without arithmetic on attacker-supplied grid/repeat values.
  for (const auto &g : *legal)
    if (dictionary == encodeSM90MmaGeometry(op.getContext(),g)) return g;
  return failure();
}

FailureOr<SM90MmaGeometry> mlir::frisk::getSM90MmaContractGeometry(
    MmaOp op, int64_t threads, Attribute binding) {
  auto contract = dyn_cast_or_null<MmaInstructionContractAttr>(binding);
  if (!contract) return failure();
  auto p = contract.getPayload();
  ScopedDiagnosticHandler quiet(op.getContext(), [](Diagnostic &) { return success(); });
  if (failed(MmaInstructionContractAttr::verify([&] { return op.emitOpError(); },p)))
    return failure();
  auto legal = getSM90MmaSemanticGeometries(op,threads);
  if (failed(legal)) return failure();
  for (const auto &g : *legal) {
    auto encoded = encodeSM90MmaGeometry(op.getContext(),g);
    if (p.get("grid") == encoded.get("grid") && p.get("atom") == encoded.get("atom") &&
        p.get("repeats") == encoded.get("repeats") &&
        p.get("row_partition") == encoded.get("row_partition")) return g;
  }
  return failure();
}

std::array<int64_t,2> mlir::frisk::getSM90MmaFragmentCoordinate(
    const SM90MmaGeometry &g, bool a, bool transA,
    int64_t reg, int64_t lane, int64_t warp, int64_t group) {
  // PTX 9.4 §9.7.17.5.1.1.1, wgmma-64N16-{A,D}.png.
  int64_t slots = a ? 8 : g.atomN/2;
  int64_t inner = reg % slots, outer = reg / slots;
  int64_t repeat = a ? g.kRepeat : g.nRepeat;
  int64_t atomRow = g.interleavedRows ? (outer/repeat)*g.gM + group/g.gN
                                      : (group/g.gN)*g.mRepeat + outer/repeat;
  int64_t row = atomRow*64 +
                warp*16 + lane/4 + (inner/2%2)*8;
  int64_t col = (a ? 0 : group%g.gN*(g.n/g.gN)) +
                outer%repeat*(a ? 16 : g.atomN) +
                lane%4*2 + inner%2 + inner/4*8;
  if (a && transA) return {col,row};
  return {row,col};
}

FailureOr<int64_t> mlir::frisk::getSM90MmaFragmentSlots(const SM90MmaGeometry &g,
                                                      bool operandA) {
  auto extent = [](int64_t v) { return v > 0 && v <= 65536 && llvm::isPowerOf2_64(v); };
  if (!isSM90WGMMAThreadCount(g.threads) || !extent(g.m) || !extent(g.n) ||
      !extent(g.k) || g.gM < 1 || g.gM > g.threads/128 ||
      !llvm::isPowerOf2_64(g.gM) || g.gN != g.threads/128/g.gM ||
      g.atomN < 8 || g.atomN > 256 || !llvm::isPowerOf2_64(g.atomN) ||
      g.m % (64*g.gM) || g.n % (g.atomN*g.gN) || g.k % 16 ||
      g.mRepeat != g.m/(64*g.gM) || g.nRepeat != g.n/(g.atomN*g.gN) ||
      g.kRepeat != g.k/16 ||
      (g.interleavedRows && (g.gM != 2 || g.gN != 1 || g.mRepeat <= 1)))
    return failure();
  return operandA ? g.mRepeat*g.kRepeat*8 : g.mRepeat*g.nRepeat*(g.atomN/2);
}

LayoutProof mlir::frisk::verifySM90MmaFragment(const SM90MmaGeometry &g,
    bool a, bool transA, RankedTensorType type, DistributedEncodingAttr encoding) {
  if (!type || !encoding) return reject("sm90-mma-fragment: missing Distributed encoding");
  // These checks also establish the power-of-two factors needed by the basis
  // proof. Validate before multiplying or dividing caller-supplied geometry.
  auto count = getSM90MmaFragmentSlots(g, a);
  if (failed(count)) return reject("sm90-mma-fragment: inconsistent geometry or repeats");
  int64_t regs = *count;
  if (a && regs > 65536/g.threads)
    return unknown("sm90-mma-fragment: A hardware point budget exceeds 65536");
  SmallVector<int64_t, 2> shape{g.m, a ? g.k : g.n};
  if (a && transA) std::swap(shape[0], shape[1]);
  if (type.getShape() != ArrayRef<int64_t>(shape))
    return reject("sm90-mma-fragment: logical shape disagrees with geometry");
  if (encoding.getTopology().asArrayRef() != ArrayRef<int64_t>({regs,32,4,g.threads/128,1}) ||
      encoding.getReplication().getInt() != (a ? g.gN : 1))
    return reject("sm90-mma-fragment: noncanonical warpgroup topology/replication");
  auto ownership = proveExecutionOwnership(encoding,type,"first_owner");
  if (ownership.status != ProofStatus::Proven) return ownership;
  LinearExecutionLayout layout;
  auto proof = analyzeLinearExecutionLayout(encoding, type, layout);
  if (proof.status != ProofStatus::Proven) return proof;
  // Within a validated PTX atom the lane, warp and register fields occupy
  // disjoint output bits. Repeats and group IDs append disjoint high bits
  // (interleaving exchanges the group and row-repeat fields). The origin is
  // zero, so the hardware equation is linear over F2. Equality on every
  // carrier basis vector proves equality everywhere, even for replicated A.
  for (unsigned bit = 0; bit < layout.hardwareBits; ++bit) {
    APInt unit = APInt::getOneBitSet(layout.hardwareBits, bit);
    auto hw = layout.hardwareCoordinate(unit); // CTA, group, warp, lane, reg.
    auto expected = getSM90MmaFragmentCoordinate(g,a,transA,hw[4],hw[3],hw[2],hw[1]);
    auto actual = layout.logicalCoordinate(layout.apply(unit));
    if (actual[0] != expected[0] || actual[1] != expected[1])
      return reject("sm90-mma-fragment: register/packed-half coordinate mismatch",
                    {hw[1],hw[2],hw[3],hw[4],expected[0],expected[1]});
  }
  return {ProofStatus::Proven,{},"sm90-mma-fragment: exact hardware coordinates by GF(2) basis equality"};
}

uint64_t mlir::frisk::getSM90DescriptorAddress(StringRef major, unsigned swizzle,
    uint64_t leading, uint64_t stride, uint64_t start, uint64_t phase,
    uint64_t mn, uint64_t k) {
  // PTX 9.4 §9.7.17.5.1.2.1.3: T=8 for 16-bit inputs.
  uint64_t s = swizzle ? swizzle/16 : 1, relative;
  if (major == "k")
    relative = mn%8*(16*s) + mn/8*stride + k%8*2 + k/8*(swizzle ? 16 : leading);
  else if (!swizzle)
    relative = mn%8*2 + mn/8*stride + k%8*16 + k/8*leading;
  else
    relative = mn%(8*s)*2 + mn/(8*s)*leading + k%8*(16*s) + k/8*stride;
  uint64_t mask = swizzle ? swizzle/16-1 : 0;
  uint64_t unswizzled = (start ^ ((phase & mask)<<4)) + relative;
  return unswizzled ^ (((unswizzled>>7)&mask)<<4);
}

LayoutProof mlir::frisk::verifySM90MmaDescriptor(const StorageAliasInfo &info,
    StorageLayoutAttr layout, MmaDescriptorPlanAttr plan, bool kIsSecond,
    int64_t mnExtent, int64_t kExtent, int64_t atomMN) {
  if (!layout || !plan || layout.getMemorySpace().getValue() != attr::MemorySpace::Shared)
    return reject("sm90-mma-descriptor: requires Shared storage and typed plan");
  if (!bounded(mnExtent,kExtent)) return unknown("sm90-mma-descriptor: logical point budget exceeds 65536");
  auto footprint = buildStorageAliasFootprint(info,layout);
  if (footprint.proof.status != ProofStatus::Proven) return footprint.proof;
  auto p = plan.getPayload();
  // Revalidate unchecked programmatic attrs as well as parser-checked attrs.
  ScopedDiagnosticHandler silence(info.viewType.getContext(),[](Diagnostic &) { return success(); });
  if (failed(MmaDescriptorPlanAttr::verify([&] { return emitError(UnknownLoc::get(info.viewType.getContext())); },p)))
    return reject("sm90-mma-descriptor: malformed plan schema");
  auto sw = p.getAs<IntegerAttr>("swizzle").getInt();
  auto leading = p.getAs<IntegerAttr>("leading").getInt();
  auto stride = p.getAs<IntegerAttr>("stride").getInt();
  auto major = p.getAs<StringAttr>("major").getValue();
  if (info.rootAlignment < uint64_t(sw ? sw*8 : 16))
    return reject("sm90-mma-descriptor: insufficient independently established root alignment");
  if (major == "k" && sw && leading != 16)
    return reject("sm90-mma-descriptor: swizzled K-major leading field must encode one");
  auto entries = p.getAs<DenseI64ArrayAttr>("entries").asArrayRef();
  if (int64_t(entries.size()) != (mnExtent/atomMN)*(kExtent/16)*4)
    return reject("sm90-mma-descriptor: missing or extra atom starts");
  std::map<std::pair<int64_t,int64_t>,uint64_t> addresses;
  for (auto &point : footprint.entries) {
    if (point.view.size()!=2 || point.begin%16 || point.end-point.begin!=16)
      return reject("sm90-mma-descriptor: requires 16-bit aligned matrix elements");
    addresses[{point.view[kIsSecond?0:1],point.view[kIsSecond?1:0]}] = point.begin/8;
  }
  size_t entry = 0;
  for (int64_t mn = 0; mn < mnExtent; mn += atomMN)
    for (int64_t k = 0; k < kExtent; k += 16, entry += 4) {
      int64_t start = entries[entry+2], phase = entries[entry+3];
      if (entries[entry]!=mn || entries[entry+1]!=k ||
          phase != (sw ? (start>>7)&7 : 0))
        return reject("sm90-mma-descriptor: wrong atom key/swizzle phase",{mn,k});
      for (int64_t i = 0; i < atomMN; ++i)
        for (int64_t j = 0; j < 16; ++j) {
          auto actual = addresses.find({mn+i,k+j});
          uint64_t expected = getSM90DescriptorAddress(major,sw,leading,stride,start,phase,i,j);
          if (actual == addresses.end() || actual->second != expected ||
              expected > (std::numeric_limits<uint64_t>::max()-16)/8 ||
              expected*8 < info.lowerBit || expected*8+16 > info.upperBit)
            return reject("sm90-mma-descriptor: reconstructed address differs from actual root-relative storage",{mn+i,k+j});
        }
    }
  return {ProofStatus::Proven,{},"sm90-mma-descriptor: exact bounded root-relative addresses"};
}

LayoutProof mlir::frisk::verifySM90MmaContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings, Attribute binding) {
  if (!constraint.instruction || constraint.vars.size()!=5 || encodings.size()!=5)
    return reject("mma-joint-contract: expected five roles and an MMA source");
  auto op = dyn_cast_or_null<MmaOp>(constraint.instruction->source);
  auto contract = dyn_cast_or_null<MmaInstructionContractAttr>(binding);
  if (!op || !contract) return reject("mma-joint-contract: unknown instruction or missing typed binding");
  auto p = contract.getPayload();
  ScopedDiagnosticHandler silence(op.getContext(),[](Diagnostic &) { return success(); });
  if (failed(MmaInstructionContractAttr::verify([&] {return op.emitOpError();},p)) ||
      failed(op.verify()))
    return reject("mma-joint-contract: malformed binding or mathematical shape");
  auto target = getSM90TargetInfo(op);
  if (failed(target)) return reject("sm90-mma-target: malformed nearest target");
  if (!target->supportsWGMMA || p.getAs<StringAttr>("target").getValue()!=target->targetKey)
    return reject("sm90-mma-target: explicit sm_90a target required");
  auto input = cast<ShapedType>(op.getA().getType()).getElementType();
  auto result = cast<RankedTensorType>(op.getResult().getType());
  if (!lookupSM90WGMMA(input,result.getElementType(),"ss",64,8,16,128,"k","k") ||
      p.getAs<TypeAttr>("input_type").getValue()!=input ||
      p.getAs<TypeAttr>("accumulator_type").getValue()!=result.getElementType())
    return reject("sm90-mma-fragment: requires f16/bf16 inputs and f32 accumulator");
  int64_t threads = graph.getVariable(constraint.vars[3]).requiredThreads;
  auto g = getSM90MmaContractGeometry(op,threads,binding);
  if (failed(g)) return diagnoseSM90MmaGeometryFailure(op,threads);
  if (auto attr = op->getAttrOfType<IntegerAttr>("frisk.execution_threads"))
    if (attr.getInt()!=threads) return reject("sm90-mma-thread-group: explicit threads conflict with execution scope");
  if (failed(decodeSM90MmaGeometry(op,threads,encodings[4])) ||
      encodings[4] != encodeSM90MmaGeometry(op.getContext(),*g))
    return reject("sm90-mma-thread-group: selected geometry disagrees with binding");
  bool rs = isa<RankedTensorType>(op.getA().getType());
  if (p.getAs<StringAttr>("form").getValue()!=(rs ? "rs" : "ss"))
    return reject("mma-joint-contract: instruction form disagrees with operand A");
  StringRef majorA = rs ? StringRef("k") :
      p.getAs<MmaDescriptorPlanAttr>("a_descriptor").getPayload().getAs<StringAttr>("major").getValue();
  StringRef majorB = p.getAs<MmaDescriptorPlanAttr>("b_descriptor").getPayload().getAs<StringAttr>("major").getValue();
  auto rule = lookupSM90WGMMA(input,result.getElementType(),rs ? "rs" : "ss",
                            64,g->atomN,16,threads,majorA,majorB);
  if (!rule || p.getAs<StringAttr>("packing").getValue()!=rule->packing)
    return reject("sm90-mma-fragment: unsupported instruction major or packing");
  for (unsigned role = 2; role < 4; ++role) {
    auto proof = verifySM90MmaFragment(*g,false,false,result,
                                      dyn_cast_or_null<DistributedEncodingAttr>(encodings[role]));
    if (proof.status != ProofStatus::Proven) {
      proof.reason = std::string(role == 2 ? "init: " : "result: ") + proof.reason;
      return proof;
    }
  }
  if (encodings[2]!=encodings[3]) return reject("mma-joint-contract: init-use/result must share encoding");
  if (rs) {
    auto proof = verifySM90MmaFragment(*g,true,op.getTransA(),
        cast<RankedTensorType>(op.getA().getType()),dyn_cast_or_null<DistributedEncodingAttr>(encodings[0]));
    if (proof.status != ProofStatus::Proven) {
      proof.reason = "A: " + proof.reason;
      return proof;
    }
  }
  for (unsigned role = rs ? 1 : 0; role < 2; ++role) {
    auto &var = graph.getVariable(constraint.vars[role]);
    Value operand = op->getOperand(role);
    if (!isa<MemRefType>(operand.getType()) || !operand.getDefiningOp<LayoutViewOp>() || !var.storageAlias)
      return reject(role ? "B: sm90-mma-descriptor: requires direct layout_view Shared operand" :
                           "A: sm90-mma-descriptor: requires direct layout_view Shared operand");
    auto plan = p.getAs<MmaDescriptorPlanAttr>(role ? "b_descriptor" : "a_descriptor");
    auto proof = verifySM90MmaDescriptor(*var.storageAlias,
        dyn_cast_or_null<StorageLayoutAttr>(encodings[role]),plan,
        role ? op.getTransB() : !op.getTransA(),role ? g->n : g->m,g->k,role ? g->atomN : 64);
    if (proof.status != ProofStatus::Proven) {
      proof.reason = std::string(role ? "B: " : "A: ") + proof.reason;
      return proof;
    }
  }
  if (!rs) {
    auto &a = *graph.getVariable(constraint.vars[0]).storageAlias;
    auto &b = *graph.getVariable(constraint.vars[1]).storageAlias;
    if (a.root==b.root) {
      auto proof = proveStorageAliasCompatible(a,cast<StorageLayoutAttr>(encodings[0]),
                                               b,cast<StorageLayoutAttr>(encodings[1]));
      if (proof.status!=ProofStatus::Proven) return proof;
    }
  }
  return {ProofStatus::Proven,{},"mma-joint-contract: all roles share one exact instruction contract"};
}
