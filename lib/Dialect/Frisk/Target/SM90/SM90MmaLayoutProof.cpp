#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutAlgebra.h"
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
  int64_t m = op.getMAttr().getInt(), n = op.getNAttr().getInt(), k = op.getKAttr().getInt();
  if ((threads != 128 && threads != 256 && threads != 512 && threads != 1024) ||
      !bounded(m,n) || !bounded(m,k) || !bounded(n,k) || m % 64 || k % 16 || n < 8)
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
  return SM90MmaGeometry{m,n,k,threads,bestM,bestN,atomN,
                        m/(64*bestM),n/(atomN*bestN),k/16};
}

std::array<int64_t,2> mlir::frisk::getSM90MmaFragmentCoordinate(
    const SM90MmaGeometry &g, bool a, bool transA,
    int64_t reg, int64_t lane, int64_t warp, int64_t group) {
  // PTX 9.4 §9.7.17.5.1.1.1, wgmma-64N16-{A,D}.png.
  int64_t slots = a ? 8 : g.atomN/2;
  int64_t inner = reg % slots, outer = reg / slots;
  int64_t repeat = a ? g.kRepeat : g.nRepeat;
  int64_t row = group/g.gN*(g.m/g.gM) + outer/repeat*64 +
                warp*16 + lane/4 + (inner/2%2)*8;
  int64_t col = (a ? 0 : group%g.gN*(g.n/g.gN)) +
                outer%repeat*(a ? 16 : g.atomN) +
                lane%4*2 + inner%2 + inner/4*8;
  if (a && transA) return {col,row};
  return {row,col};
}

LayoutProof mlir::frisk::verifySM90MmaFragment(const SM90MmaGeometry &g,
    bool a, bool transA, RankedTensorType type, DistributedEncodingAttr encoding) {
  if (!type || !encoding) return reject("sm90-mma-fragment: missing Distributed encoding");
  int64_t regs = a ? g.mRepeat*g.kRepeat*8 : g.mRepeat*g.nRepeat*g.atomN/2;
  if (regs > 65536/g.threads) return unknown("sm90-mma-fragment: hardware point budget exceeds 65536");
  if (encoding.getTopology().asArrayRef() != ArrayRef<int64_t>({regs,32,4,g.threads/128,1}) ||
      encoding.getReplication().getInt() != (a ? g.gN : 1))
    return reject("sm90-mma-fragment: noncanonical warpgroup topology/replication");
  auto ownership = proveExecutionOwnership(encoding,type,"first_owner");
  if (ownership.status != ProofStatus::Proven) return ownership;
  auto map = dyn_cast<BitLinearLayoutMapAttr>(encoding.getMap());
  if (!map) return unknown("sm90-mma-fragment: unsupported map representation");
  auto matrix = map.getMatrixValue();
  if (failed(matrix)) return reject("sm90-mma-fragment: malformed matrix");
  const StringRef carrierNames[] = {"register","lane","warp","warp_group","cta"};
  for (int64_t group = 0; group < g.threads/128; ++group)
    for (int64_t warp = 0; warp < 4; ++warp)
      for (int64_t lane = 0; lane < 32; ++lane)
        for (int64_t reg = 0; reg < regs; ++reg) {
          int64_t coordinates[] = {reg,lane,warp,group,0};
          APInt input(matrix->getNumColumns(),0);
          unsigned offset = 0;
          for (unsigned d = 0; d < map.getInputNames().size(); ++d) {
            auto name = cast<StringAttr>(map.getInputNames()[d]).getValue();
            auto found = llvm::find(ArrayRef<StringRef>(carrierNames),name);
            if (found == std::end(carrierNames)) return reject("sm90-mma-fragment: unknown carrier");
            input |= APInt(input.getBitWidth(),coordinates[found-std::begin(carrierNames)]) << offset;
            offset += map.getInputBitWidths()[d];
          }
          APInt out = matrix->apply(input);
          auto expected = getSM90MmaFragmentCoordinate(g,a,transA,reg,lane,warp,group);
          offset = 0;
          for (unsigned d = 0; d < 2; ++d) {
            unsigned width = map.getOutputBitWidths()[d];
            if (out.extractBitsAsZExtValue(width,offset) != uint64_t(expected[d]))
              return reject("sm90-mma-fragment: register/packed-half coordinate mismatch",
                            {group,warp,lane,reg,expected[0],expected[1]});
            offset += width;
          }
        }
  return {ProofStatus::Proven,{},"sm90-mma-fragment: exact hardware coordinates"};
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
  if (!constraint.instruction || constraint.vars.size()!=4 || encodings.size()!=4)
    return reject("mma-joint-contract: expected four roles and an MMA source");
  auto op = dyn_cast_or_null<MmaOp>(constraint.instruction->source);
  auto contract = dyn_cast_or_null<MmaInstructionContractAttr>(binding);
  if (!op || !contract) return reject("mma-joint-contract: unknown instruction or missing typed binding");
  auto p = contract.getPayload();
  ScopedDiagnosticHandler silence(op.getContext(),[](Diagnostic &) { return success(); });
  if (failed(MmaInstructionContractAttr::verify([&] {return op.emitOpError();},p)) ||
      failed(op.verify()))
    return reject("mma-joint-contract: malformed binding or mathematical shape");
  StringAttr target;
  for (Operation *scope = op; scope && !target; scope = scope->getParentOp())
    if (Attribute attr = scope->getAttr("frisk.target")) {
      target = dyn_cast<StringAttr>(attr);
      if (!target) return reject("sm90-mma-target: malformed nearest target");
    }
  if (!target || target.getValue()!="sm_90a" || p.getAs<StringAttr>("target")!=target)
    return reject("sm90-mma-target: explicit sm_90a target required");
  auto input = cast<ShapedType>(op.getA().getType()).getElementType();
  auto result = cast<RankedTensorType>(op.getResult().getType());
  if ((!input.isF16() && !input.isBF16()) || !result.getElementType().isF32() ||
      p.getAs<TypeAttr>("input_type").getValue()!=input ||
      p.getAs<TypeAttr>("accumulator_type").getValue()!=result.getElementType())
    return reject("sm90-mma-fragment: requires f16/bf16 inputs and f32 accumulator");
  int64_t threads = graph.getVariable(constraint.vars[3]).requiredThreads;
  auto g = getSM90MmaGeometry(op,threads);
  if (failed(g)) {
    auto exceeds = [](ShapedType t) {
      return t.getRank()==2 && t.getDimSize(0)>0 && t.getDimSize(1)>0 &&
             (t.getDimSize(0)>65536 || t.getDimSize(1)>65536/t.getDimSize(0));
    };
    if (exceeds(result) || exceeds(cast<ShapedType>(op.getA().getType())) ||
        exceeds(cast<ShapedType>(op.getB().getType())))
      return unknown("sm90-mma-fragment: logical point budget exceeds 65536");
    return reject("sm90-mma-thread-group: unsupported tile, threads or policy grouping");
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>("frisk.execution_threads"))
    if (attr.getInt()!=threads) return reject("sm90-mma-thread-group: explicit threads conflict with execution scope");
  if (p.getAs<DenseI64ArrayAttr>("grid").asArrayRef()!=ArrayRef<int64_t>({g->gM,g->gN}) ||
      p.getAs<DenseI64ArrayAttr>("atom").asArrayRef()!=ArrayRef<int64_t>({64,g->atomN,16}) ||
      p.getAs<DenseI64ArrayAttr>("repeats").asArrayRef()!=ArrayRef<int64_t>({g->mRepeat,g->nRepeat,g->kRepeat}))
    return reject("sm90-mma-thread-group: binding disagrees with deterministic policy geometry");
  bool rs = isa<RankedTensorType>(op.getA().getType());
  if (p.getAs<StringAttr>("form").getValue()!=(rs ? "rs" : "ss"))
    return reject("mma-joint-contract: instruction form disagrees with operand A");
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
