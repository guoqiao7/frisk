#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90InstructionTable.h"
#include "Dialect/Frisk/Target/SM90/SM90TargetInfo.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"
#include <map>

using namespace mlir;
using namespace mlir::frisk;
namespace {
BitLinearLayoutMapAttr makeLinearMap(MLIRContext *ctx,
    ArrayRef<StringRef> names, ArrayRef<int64_t> extents,
    ArrayRef<int64_t> shape,
    function_ref<SmallVector<int64_t>(ArrayRef<int64_t>)> evaluate) {
  Builder b(ctx);
  SmallVector<Attribute> inputs, outputs;
  SmallVector<int64_t> widths, outputWidths;
  unsigned columns = 0, rows = 0;
  for (auto [name,extent] : llvm::zip_equal(names,extents)) {
    if (extent==1) continue;
    inputs.push_back(b.getStringAttr(name));
    widths.push_back(llvm::Log2_64(extent));
    columns += widths.back();
  }
  for (auto [dim,extent] : llvm::enumerate(shape)) {
    outputs.push_back(b.getStringAttr("dim"+Twine(dim)));
    outputWidths.push_back(llvm::Log2_64(extent));
    rows += outputWidths.back();
  }
  SmallVector<APInt> matrix(rows*columns,APInt(1,0));
  unsigned column = 0;
  for (unsigned axis = 0; axis < extents.size(); ++axis)
    for (unsigned bit = 0; bit < llvm::Log2_64(extents[axis]); ++bit,++column) {
      SmallVector<int64_t> point(extents.size(),0);
      point[axis] = int64_t{1}<<bit;
      auto out = evaluate(point);
      unsigned row = 0;
      for (unsigned d = 0; d < shape.size(); ++d)
        for (int64_t r = 0; r < outputWidths[d]; ++r,++row)
          matrix[row*columns+column] = APInt(1,(out[d]>>r)&1);
    }
  return BitLinearLayoutMapAttr::get(ctx,b.getArrayAttr(inputs),b.getDenseI64ArrayAttr(widths),
      b.getArrayAttr(outputs),b.getDenseI64ArrayAttr(outputWidths),
      DenseIntElementsAttr::get(RankedTensorType::get({rows,columns},b.getI1Type()),matrix));
}

FailureOr<MmaDescriptorPlanAttr> decodeDescriptor(const StorageAliasInfo &info,
    StorageLayoutAttr layout, bool kSecond, int64_t mnExtent, int64_t kExtent, int64_t atomMN) {
  auto footprint = buildStorageAliasFootprint(info,layout);
  if (footprint.proof.status!=ProofStatus::Proven) return failure();
  std::map<std::pair<int64_t,int64_t>,uint64_t> addresses;
  for (auto &point : footprint.entries) {
    if (point.view.size()!=2 || point.begin%16 || point.end-point.begin!=16) return failure();
    addresses[{point.view[kSecond?0:1],point.view[kSecond?1:0]}] = point.begin/8;
  }
  Builder b(info.viewType.getContext());
  // Fixed key order: major k,mn; swizzle 0,32,64,128.
  for (StringRef major : {StringRef("k"),StringRef("mn")})
    for (int64_t sw : {0,32,64,128}) {
      if (info.rootAlignment<uint64_t(sw ? sw*8 : 16)) continue;
      uint64_t mask = sw ? sw/16-1 : 0;
      auto raw = [&](int64_t mn,int64_t k) {
        uint64_t value = addresses.at({mn,k});
        return value ^ (((value>>7)&mask)<<4);
      };
      uint64_t origin = raw(0,0);
      int64_t s = sw ? sw/16 : 1;
      int64_t leading = 16, stride = 128*s;
      bool valid = true;
      auto delta = [&](int64_t mn,int64_t k,int64_t &field) {
        if (mn>=mnExtent || k>=kExtent) return;
        uint64_t value = raw(mn,k);
        if (value<origin || value-origin>=262144 || (value-origin)%16) valid=false;
        else field = value-origin;
      };
      if (major=="k") {
        if (!sw) delta(0,8,leading);
        delta(8,0,stride);
      } else if (!sw) {
        delta(0,8,leading);
        delta(8,0,stride);
      } else {
        leading = 128*s;
        delta(8*s,0,leading);
        delta(0,8,stride);
      }
      if (!valid) continue;
      SmallVector<int64_t> entries;
      for (int64_t mn = 0; mn < mnExtent; mn += atomMN)
        for (int64_t k = 0; k < kExtent; k += 16) {
          uint64_t start = addresses.at({mn,k});
          entries.append({mn,k,int64_t(start),sw ? int64_t((start>>7)&7) : 0});
        }
      auto p = b.getDictionaryAttr({b.getNamedAttr("major",b.getStringAttr(major)),
          b.getNamedAttr("swizzle",b.getI64IntegerAttr(sw)),
          b.getNamedAttr("leading",b.getI64IntegerAttr(leading)),
          b.getNamedAttr("stride",b.getI64IntegerAttr(stride)),
          b.getNamedAttr("entries",b.getDenseI64ArrayAttr(entries))});
      auto plan = MmaDescriptorPlanAttr::get(info.viewType.getContext(),p);
      if (verifySM90MmaDescriptor(info,layout,plan,kSecond,mnExtent,kExtent,atomMN).status==ProofStatus::Proven)
        return plan;
    }
  return failure();
}

FailureOr<StorageLayoutAttr> storageProposal(const StorageAliasInfo &info,
                                            bool kSecond, unsigned sw) {
  auto type = info.rootType;
  if (type.getRank()!=2 || !type.hasStaticShape() || type.getElementTypeBitWidth()!=16 ||
      info.rootAlignment < (sw ? sw*8 : 16) || info.lowerBit%128)
    return failure();
  int64_t mnExtent = type.getDimSize(kSecond ? 0 : 1), kExtent = type.getDimSize(kSecond ? 1 : 0);
  if (mnExtent<=1 || kExtent<=1 || mnExtent>65536 || kExtent>65536/mnExtent ||
      !llvm::isPowerOf2_64(mnExtent) || !llvm::isPowerOf2_64(kExtent)) return failure();
  // The default major follows the original final dimension: K or MN.
  bool majorK = kSecond;
  if ((majorK && kExtent<int64_t(sw/2)) || (!majorK && mnExtent<int64_t(sw/2)))
    return failure();
  MLIRContext *ctx = type.getContext();
  Builder b(ctx);
  auto mn = getAffineDimExpr(kSecond?0:1,ctx), k = getAffineDimExpr(kSecond?1:0,ctx);
  int64_t s = sw ? sw/16 : 1;
  AffineExpr address;
  if (majorK)
    address = (mn%8)*(16*s) + mn.floorDiv(8)*(128*s) +
              (k%(8*s))*2 + k.floorDiv(8*s)*(mnExtent*16*s);
  else
    address = (mn%(8*s))*2 + mn.floorDiv(8*s)*(128*s) +
              (k%8)*(16*s) + k.floorDiv(8)*(mnExtent*16);
  // Root-relative descriptor origin participates in the same swizzle, once.
  address = address + int64_t(info.lowerBit/8);
  AffineExpr swizzled = address;
  for (unsigned bit = 0; bit < (sw ? llvm::Log2_64(sw/16) : 0); ++bit) {
    int64_t dst = int64_t{1}<<(4+bit), src = int64_t{1}<<(7+bit);
    auto old = address.floorDiv(dst)%2;
    auto parity = (old + address.floorDiv(src)%2)%2;
    swizzled = swizzled + (parity-old)*dst;
  }
  auto map = AffineLayoutMapAttr::get(ctx,b.getArrayAttr({b.getStringAttr("dim0"),b.getStringAttr("dim1")}),
      b.getDenseI64ArrayAttr(type.getShape()),b.getArrayAttr({b.getStringAttr("byte_offset"),b.getStringAttr("bit_offset")}),
      b.getDenseI64ArrayAttr({int64_t(info.upperBit/8),8}),
      AffineMapAttr::get(AffineMap::get(2,0,{swizzled,getAffineConstantExpr(0,ctx)},ctx)));
  auto layout = StorageLayoutAttr::get(ctx,map,MemorySpaceAttr::get(ctx,attr::MemorySpace::Shared),
      b.getI64IntegerAttr(info.rootAlignment),b.getI64IntegerAttr(1));
  StorageAliasInfo root = info;
  root.viewType = root.rootType;
  root.viewToRoot = AffineMap::getMultiDimIdentityMap(2,ctx);
  if (verifyStorageAliasCandidate(root,layout).status!=ProofStatus::Proven) return failure();
  return projectStorageAliasCandidate(root,layout,info);
}
}

FailureOr<DistributedEncodingAttr> mlir::frisk::buildSM90MmaFragment(MLIRContext *ctx,
    const SM90MmaGeometry &g, bool a, bool transA) {
  int64_t regs = a ? g.mRepeat*g.kRepeat*8 : g.mRepeat*g.nRepeat*g.atomN/2;
  if (regs<=0 || g.threads<=0 || regs>65536/g.threads) return failure();
  SmallVector<int64_t> shape{g.m,a ? g.k : g.n};
  if (a && transA) std::swap(shape[0],shape[1]);
  SmallVector<int64_t> topology{regs,32,4,g.threads/128,1};
  auto map = makeLinearMap(ctx,{"register","lane","warp","warp_group","cta"},topology,shape,
      [&](ArrayRef<int64_t> p) {
        auto v = getSM90MmaFragmentCoordinate(g,a,transA,p[0],p[1],p[2],p[3]);
        return SmallVector<int64_t>{v[0],v[1]};
      });
  Builder b(ctx);
  return DistributedEncodingAttr::get(ctx,map,b.getDenseI64ArrayAttr(topology),b.getI64IntegerAttr(a ? g.gN : 1));
}

LogicalResult mlir::frisk::prepareSM90MmaCandidates(LayoutConstraintGraph &graph) {
  for (auto &constraint : graph.getConstraints()) {
    if (!constraint.instruction) continue;
    auto op = dyn_cast_or_null<MmaOp>(constraint.instruction->source);
    if (!op || constraint.vars.size()!=5) continue;
    int64_t threads = graph.getVariable(constraint.vars[3]).requiredThreads;
    auto geometries = getSM90MmaGeometries(op,threads);
    if (failed(geometries)) {
      auto exceeds = [](ShapedType type) {
        return type.getRank() == 2 && type.getDimSize(0) > 0 &&
               type.getDimSize(1) > 0 &&
               (type.getDimSize(0) > 65536 ||
                type.getDimSize(1) > 65536 / type.getDimSize(0));
      };
      if (exceeds(cast<ShapedType>(op.getA().getType())) ||
          exceeds(cast<ShapedType>(op.getB().getType())) ||
          exceeds(cast<ShapedType>(op.getResult().getType())))
        return op.emitOpError("sm90-mma-fragment: unknown proof: logical point budget exceeds 65536");
    }
    // Preserve logical-budget evidence before the bounded semantic decoder can
    // reject a valid explicit plan whose logical domain is too large to prove.
    FailureOr<SM90MmaGeometry> fixed = failure();
    if (constraint.instruction->binding) {
      fixed = getSM90MmaContractGeometry(op,threads,constraint.instruction->binding);
      if (failed(fixed))
        return op.emitOpError("sm90-mma-thread-group: explicit binding has invalid geometry");
      // The semantic decoder admits this hard binding, but Auto may omit it
      // from the generated domain solely for the RS hardware proof budget.
      if (failed(geometries) || llvm::none_of(*geometries,[&](const SM90MmaGeometry &g) {
            return g.gM == fixed->gM && g.gN == fixed->gN;
          }))
        return op.emitOpError("sm90-mma-fragment: unknown proof: hardware point budget exceeds 65536");
    }
    if (failed(geometries))
      return op.emitOpError("sm90-mma-thread-group: unsupported tile/policy/threads");
    // SameLayout makes a hard-bound accumulator authoritative for both roles.
    // Gather only exact equality requirements, never Convertible preferences.
    SmallVector<SmallVector<Attribute>> hardRequirements(5);
    for (unsigned role = 0; role < 5; ++role) {
      SmallVector<LayoutVarID> peers{constraint.vars[role]};
      for (size_t i = 0; i < peers.size(); ++i)
        for (const auto &c : graph.getConstraints()) {
          if (c.strength != ConstraintStrength::Hard || !llvm::is_contained(c.vars,peers[i])) continue;
          if (c.kind == ConstraintKind::RequireEncoding) hardRequirements[role].push_back(c.requiredEncoding);
          if (c.kind == ConstraintKind::SameLayout)
            for (auto peer : c.vars) if (!llvm::is_contained(peers,peer)) peers.push_back(peer);
        }
    }
    auto add = [&](unsigned role, Attribute attr) {
      ++graph.getCandidatePreparationStatistics().generatedProposals;
      auto &var = graph.getVariable(constraint.vars[role]);
      // Hard seeds are retained, including semantically equivalent reordered
      // carriers. An impossible generated alternative cannot consume the cap.
      if (llvm::any_of(hardRequirements[role],[&](Attribute required) { return required != attr; })) return;
      if (llvm::none_of(var.candidates,[&](const LayoutCandidate &c) { return c.value==attr; })) {
        auto provenance = graph.addProvenance(std::nullopt,op,"mma-target-candidate",
            (Twine("finite canonical proposal for MMA role ")+Twine(role)).str());
        var.candidates.push_back({attr,provenance,var.candidates.size()});
      }
    };
    bool rs = isa<RankedTensorType>(op.getA().getType());
    for (const auto &g : *geometries) {
      if (constraint.instruction->binding) {
        if (fixed->gM != g.gM || fixed->gN != g.gN) continue;
      }
      add(4,encodeSM90MmaGeometry(op.getContext(),g));
      auto accum = buildSM90MmaFragment(op.getContext(),g,false,false);
      if (failed(accum)) return op.emitOpError("sm90-mma-fragment: unknown proof: init/result hardware enumeration budget exceeded");
      add(2,*accum); add(3,*accum);
      if (rs) {
        auto a = buildSM90MmaFragment(op.getContext(),g,true,op.getTransA());
        if (failed(a)) return op.emitOpError("sm90-mma-fragment: unknown proof: A hardware enumeration budget exceeded");
        add(0,*a);
      }
    }
    for (unsigned role = rs?1:0; role<2; ++role) {
      auto &var = graph.getVariable(constraint.vars[role]);
      if (!var.storageAlias) continue;
      // Explicit role seeds are retained exactly and need no alternate proposal.
      bool hard = llvm::any_of(graph.getConstraints(),[&](const LayoutConstraint &c) {
        return c.kind==ConstraintKind::RequireEncoding && llvm::is_contained(c.vars,var.id);
      });
      if (hard) continue;
      for (unsigned sw : {0u,32u,64u,128u}) {
        auto candidate = storageProposal(*var.storageAlias,role ? op.getTransB() : !op.getTransA(),sw);
        if (succeeded(candidate)) add(role,*candidate);
      }
    }
  }
  return success();
}

InstructionContractBuildResult mlir::frisk::buildSM90MmaContractDetailed(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings) {
  auto unsupported = [](StringRef reason) -> InstructionContractBuildResult {
    return {{},{ProofStatus::Unknown,{},reason.str()},"mma-decoder",{}};
  };
  auto checked = [&](Attribute binding) -> InstructionContractBuildResult {
    auto proof = verifySM90MmaContract(graph,constraint,encodings,binding);
    bool proven = proof.status == ProofStatus::Proven;
    return {proven ? binding : Attribute{},std::move(proof),proven ? "" : "mma-contract",{}};
  };
  if (!constraint.instruction || constraint.vars.size()!=5 || encodings.size()!=5)
    return unsupported("mma-joint-contract: expected five roles");
  auto op = dyn_cast_or_null<MmaOp>(constraint.instruction->source);
  if (!op) return unsupported("mma-joint-contract: missing MMA source");
  if (Attribute explicitBinding = constraint.instruction->binding) return checked(explicitBinding);
  auto g = decodeSM90MmaGeometry(op,graph.getVariable(constraint.vars[3]).requiredThreads,encodings[4]);
  if (failed(g)) return unsupported("sm90-mma-thread-group: invalid geometry candidate");
  auto target = getSM90TargetInfo(op);
  if (failed(target) || !target->supportsWGMMA) return unsupported("sm90-mma-target: explicit sm_90a required");
  bool rs = isa<RankedTensorType>(op.getA().getType());
  auto rule = lookupSM90WGMMA(cast<ShapedType>(op.getA().getType()).getElementType(),
      cast<ShapedType>(op.getResult().getType()).getElementType(),rs ? "rs" : "ss",
      64,g->atomN,16,g->threads,"k","k");
  if (!rule) return unsupported("sm90-mma-fragment: unsupported instruction form");
  Builder b(op.getContext());
  SmallVector<NamedAttribute> fields{
    b.getNamedAttr("version",b.getI64IntegerAttr(1)),
    b.getNamedAttr("target",b.getStringAttr(target->targetKey)),
    b.getNamedAttr("form",b.getStringAttr(rs ? "rs" : "ss")),
    b.getNamedAttr("input_type",TypeAttr::get(cast<ShapedType>(op.getA().getType()).getElementType())),
    b.getNamedAttr("accumulator_type",TypeAttr::get(cast<ShapedType>(op.getResult().getType()).getElementType())),
    b.getNamedAttr("atom",b.getDenseI64ArrayAttr({64,g->atomN,16})),
    b.getNamedAttr("grid",b.getDenseI64ArrayAttr({g->gM,g->gN})),
    b.getNamedAttr("repeats",b.getDenseI64ArrayAttr({g->mRepeat,g->nRepeat,g->kRepeat})),
    b.getNamedAttr("packing",b.getStringAttr(rule->packing))};
  for (unsigned role = rs?1:0; role<2; ++role) {
    auto &var = graph.getVariable(constraint.vars[role]);
    auto storage = dyn_cast_or_null<StorageLayoutAttr>(encodings[role]);
    if (!var.storageAlias || !storage) return unsupported("sm90-mma-descriptor: missing storage alias/layout");
    auto plan = decodeDescriptor(*var.storageAlias,storage,role ? op.getTransB() : !op.getTransA(),
                                role ? g->n : g->m,g->k,role ? g->atomN : 64);
    if (failed(plan)) return unsupported("sm90-mma-descriptor: bounded descriptor decoder could not construct a plan");
    fields.push_back(b.getNamedAttr(role ? "b_descriptor" : "a_descriptor",*plan));
  }
  Attribute binding = MmaInstructionContractAttr::get(op.getContext(),b.getDictionaryAttr(fields));
  return checked(binding);
}

FailureOr<Attribute> mlir::frisk::buildSM90MmaContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, ArrayRef<Attribute> encodings) {
  auto result = buildSM90MmaContractDetailed(graph,constraint,encodings);
  if (result.proof.status != ProofStatus::Proven || !result.binding) return failure();
  return result.binding;
}
