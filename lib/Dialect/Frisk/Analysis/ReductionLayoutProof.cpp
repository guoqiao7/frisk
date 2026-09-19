#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MathExtras.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
LayoutProof reject(StringRef s) { return {ProofStatus::Disproven, {}, s.str()}; }
bool validTypes(RankedTensorType src, RankedTensorType dst, int64_t axis) {
  if (!src || !dst || src.getRank() < 2 || dst.getRank()+1 != src.getRank() ||
      axis < 0 || axis >= src.getRank() || src.getElementType() != dst.getElementType()) return false;
  for (auto t : {src,dst})
    if (!t.hasStaticShape() || llvm::any_of(t.getShape(), [](int64_t e) {
          return e <= 1 || !llvm::isPowerOf2_64(e); })) return false;
  SmallVector<int64_t> shape(src.getShape()); shape.erase(shape.begin()+axis);
  return dst.getShape() == ArrayRef<int64_t>(shape);
}
ReductionHolder holder(const ExecutionEnumeration &e, uint64_t id) {
  auto p = e.hardware(id); ReductionHolder h;
  std::copy(p.begin(),p.end(),h.begin()); return h;
}
unsigned edgeScope(const ReductionHolder &a, const ReductionHolder &b) {
  if (a[0] != b[0]) return 3;
  if (a[1] != b[1] || a[2] != b[2]) return 2;
  if (a[3] != b[3]) return 1;
  return 0;
}
} // namespace

FailureOr<DistributedEncodingAttr> mlir::frisk::projectReductionEncoding(
    DistributedEncodingAttr src, RankedTensorType srcType, RankedTensorType dstType, int64_t axis) {
  if (!validTypes(srcType,dstType,axis)) return failure();
  ExecutionEnumeration points;
  if (enumerateExecutionLayout(src,srcType,points).status != ProofStatus::Proven) return failure();
  auto map = cast<BitLinearLayoutMapAttr>(src.getMap());
  unsigned begin = 0;
  for (int64_t i=0; i<axis; ++i) begin += map.getOutputBitWidths()[i];
  unsigned end = begin+map.getOutputBitWidths()[axis];
  unsigned cols=0, regBegin=0, regWidth=0;
  int regIndex=-1;
  for (auto [i,w] : llvm::enumerate(map.getInputBitWidths().asArrayRef())) {
    if (cast<StringAttr>(map.getInputNames()[i]).getValue() == "register") {
      regIndex=i; regBegin=cols; regWidth=w;
    }
    cols += w;
  }
  SmallVector<uint64_t> columns(cols,0);
  unsigned oldRow=0,newRow=0,c=0;
  for (auto bit : map.getMatrix().getValues<APInt>()) {
    if ((oldRow < begin || oldRow >= end) && !bit.isZero()) columns[c] |= uint64_t(1)<<newRow;
    if (++c == cols) { c=0; if (oldRow < begin || oldRow >= end) ++newRow; ++oldRow; }
  }
  // Choose an ordered basis of the projected register subspace, including
  // nonzero dependent-column elimination (not merely dropping zero columns).
  SmallVector<unsigned> keep;
  std::array<uint64_t,64> basis{};
  for (unsigned i=0; i<cols; ++i) {
    if (i < regBegin || i >= regBegin+regWidth) { keep.push_back(i); continue; }
    uint64_t v=columns[i];
    for (unsigned b=0;b<64 && v;++b) if (v & (uint64_t(1)<<b)) {
      if (basis[b]) v ^= basis[b]; else { basis[b]=v; keep.push_back(i); break; }
    }
  }
  Builder b(src.getContext());
  SmallVector<int64_t> widths(map.getInputBitWidths().asArrayRef());
  SmallVector<Attribute> inputNames(map.getInputNames().getValue());
  unsigned keptRegs=regWidth-(cols-keep.size());
  if (regIndex >= 0) widths[regIndex]=keptRegs;
  if (regIndex >= 0 && !keptRegs) {
    widths.erase(widths.begin()+regIndex); inputNames.erase(inputNames.begin()+regIndex);
  }
  SmallVector<int64_t> topology(src.getTopology().asArrayRef()); topology[0]=int64_t(1)<<keptRegs;
  SmallVector<Attribute> names;
  SmallVector<int64_t> outWidths;
  for (unsigned i=0;i<map.getOutputNames().size();++i) if (i != unsigned(axis)) {
    names.push_back(map.getOutputNames()[i]); outWidths.push_back(map.getOutputBitWidths()[i]);
  }
  SmallVector<APInt> bits;
  for (unsigned r=0;r<newRow;++r) for (unsigned col:keep) bits.push_back(APInt(1,(columns[col]>>r)&1));
  auto projected=BitLinearLayoutMapAttr::get(src.getContext(),b.getArrayAttr(inputNames),b.getDenseI64ArrayAttr(widths),
      b.getArrayAttr(names),b.getDenseI64ArrayAttr(outWidths),DenseIntElementsAttr::get(
          RankedTensorType::get({int64_t(newRow),int64_t(keep.size())},b.getI1Type()),bits));
  int64_t hardware=1; for (auto e:topology) hardware*=e;
  auto result=DistributedEncodingAttr::get(src.getContext(),projected,b.getDenseI64ArrayAttr(topology),
      b.getI64IntegerAttr(hardware/dstType.getNumElements()));
  ExecutionEnumeration output;
  if (enumerateExecutionLayout(result,dstType,output).status != ProofStatus::Proven) return failure();
  return result;
}

ReductionLayoutProof mlir::frisk::buildReductionLayoutProof(DistributedEncodingAttr src,
    DistributedEncodingAttr dst, RankedTensorType srcType, RankedTensorType dstType,int64_t axis) {
  ReductionLayoutProof out;
  out.proof=reject("invalid reduction shapes");
  if (!validTypes(srcType,dstType,axis)) return out;
  ExecutionEnumeration source,result;
  out.proof=enumerateExecutionLayout(src,srcType,source);
  if (out.proof.status != ProofStatus::Proven) return out;
  out.proof=enumerateExecutionLayout(dst,dstType,result);
  if (out.proof.status != ProofStatus::Proven) return out;
  if (source.topology[4] != 1 || result.topology[4] != 1) {
    out.proof=reject("reduction requires a single CTA"); return out;
  }
  auto natural=projectReductionEncoding(src,srcType,dstType,axis);
  ExecutionEnumeration projected;
  if (failed(natural) || enumerateExecutionLayout(*natural,dstType,projected).status != ProofStatus::Proven ||
      projected.topology != result.topology || projected.logical != result.logical) {
    out.proof=reject("result encoding differs from natural reduction projection"); return out;
  }
  out.fibers.resize(result.first.size());
  for (size_t y=0;y<out.fibers.size();++y) out.fibers[y].logicalOutput=y;
  // Iterate hardware once: first owners are automatically sorted by physical key.
  for (uint64_t h=0;h<source.logical.size();++h) {
    uint64_t x=source.logical[h]; if (source.first[x] != h) continue;
    auto coord=source.coordinate(x); uint64_t y=0;
    for (unsigned i=0;i<coord.size();++i) if (i != unsigned(axis)) y=y*srcType.getDimSize(i)+coord[i];
    out.fibers[y].nodes.push_back({int64_t(x),-1,-1,holder(source,h)});
  }
  unsigned scope=0;
  for (auto &fiber:out.fibers) {
    SmallVector<int64_t> layer;
    for (size_t i=0;i<fiber.nodes.size();++i) layer.push_back(i);
    while (layer.size()>1) {
      SmallVector<int64_t> next;
      for (size_t i=0;i<layer.size();i+=2) {
        if (i+1==layer.size()) { next.push_back(layer[i]); continue; }
        auto left=layer[i],right=layer[i+1];
        auto owner=fiber.nodes[left].holder;
        scope=std::max(scope,edgeScope(owner,fiber.nodes[right].holder));
        next.push_back(fiber.nodes.size()); fiber.nodes.push_back({-1,left,right,owner});
      }
      layer=std::move(next);
    }
    fiber.root=layer.front();
  }
  for (uint64_t h=0;h<result.logical.size();++h) {
    auto &fiber=out.fibers[result.logical[h]];
    auto owner=holder(result,h); fiber.broadcasts.push_back(owner);
    scope=std::max(scope,edgeScope(fiber.nodes[fiber.root].holder,owner));
  }
  out.scope=scope==0 ? "register" : scope==1 ? "warp" : "cta_shared_tree";
  out.proof={ProofStatus::Proven,{},"canonical_fiber_tree_v1: disjoint distinct-input tree and complete broadcasts"};
  return out;
}

LayoutProof mlir::frisk::verifyReductionLayoutProof(DistributedEncodingAttr src,
    DistributedEncodingAttr dst,RankedTensorType srcType,RankedTensorType dstType,
    int64_t axis,const ReductionLayoutProof &proof) {
  // Pure reconstruction, never candidate generation/search. Comparing every leaf,
  // edge and destination enforces the unique specified tree, and thus induction
  // over disjoint contribution sets without quadratic materialized sets.
  auto expected=buildReductionLayoutProof(src,dst,srcType,dstType,axis);
  if (expected.proof.status != ProofStatus::Proven) return expected.proof;
  if (proof.proof.status != ProofStatus::Proven || proof.scope != expected.scope || proof.fibers != expected.fibers)
    return reject("reduction contribution tree, first_owner selection or complete broadcast coverage mismatch");
  return expected.proof;
}
