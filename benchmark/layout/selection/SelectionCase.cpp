#include "SelectionCase.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;
using namespace mlir::frisk;
namespace mlir::frisk::selection {
namespace {
constexpr int64_t shapes[][4] = {{64,64,16,128},{128,128,16,256},
  {128,512,16,256},{256,128,16,256},{128,128,32,256},{256,256,16,512},
  {512,128,16,1024},{128,128,16,512}};
bool fixed(Consumer c) { return c==Consumer::Fixed0 || c==Consumer::Fixed1; }
bool reduction(Consumer c) { return c!=Consumer::Store && c!=Consumer::Local; }
unsigned axis(Consumer c) { return c==Consumer::Reduce1 || c==Consumer::Fixed1 ? 1 : 0; }
SmallVector<int64_t> outputShape(const CaseSpec &s) {
  if (reduction(s.consumer)) return {axis(s.consumer)==0 ? s.n : s.m};
  return {s.m,s.n};
}
} // namespace

std::vector<CaseSpec> nativeCases() {
  std::vector<CaseSpec> result;
  for (unsigned i=0;i<8;++i) for (auto mode : {MmaMode::SS,MmaMode::RS})
    for (const char *policy : {"Auto","Square","FullRow","FullCol"}) {
      CaseSpec c; c.id="C"+std::to_string(i); c.m=shapes[i][0]; c.n=shapes[i][1];
      c.k=shapes[i][2]; c.threads=shapes[i][3]; c.mode=mode; c.policy=policy;
      result.push_back(c);
    }
  return result;
}
llvm::StringRef modeName(MmaMode m) { return m==MmaMode::SS ? "ss" : "rs"; }
llvm::StringRef consumerName(Consumer c) {
  switch(c) {
  case Consumer::Store: return "store";
  case Consumer::Reduce0: return "reduce0";
  case Consumer::Reduce1: return "reduce1";
  case Consumer::Fixed0: return "fixed0";
  case Consumer::Fixed1: return "fixed1";
  case Consumer::Local: return "local";
  }
  llvm_unreachable("invalid consumer");
}
llvm::json::Object caseJSON(const CaseSpec &s) {
  llvm::json::Object result{{"id",s.id},{"m",s.m},{"n",s.n},{"k",s.k},
    {"threads",s.threads},{"mode",modeName(s.mode)},{"policy",s.policy},
    {"consumer",consumerName(s.consumer)}, {"input_type","f16"},
    {"accumulator_type","f32"},{"trans_a",false},{"trans_b",false},
    {"producer_a",s.mode==MmaMode::SS ? "shared-alloc-layout-view" : "tensor-dense-one"},
    {"producer_b","shared-alloc-layout-view"},{"init","tensor-dense-zero"},
    {"shared_alignment",1024},{"global_alignment",128},
    {"input_scope","shared/tensor-already-prepared-no-global-copy"}};
  if (s.consumerGrid) result["consumer_grid"]=llvm::json::Array{s.consumerGrid->first,s.consumerGrid->second};
  else result["consumer_grid"]=nullptr;
  return result;
}
LogicalResult validateCase(const CaseSpec &s,std::string &error) {
  error.clear();
  if (s.id.size()!=2 || s.id[0]!='C' || s.id[1]<'0' || s.id[1]>'7') error="unknown case id";
  else {
    const auto &dims=shapes[s.id[1]-'0'];
    if (s.m!=dims[0] || s.n!=dims[1] || s.k!=dims[2] || s.threads!=dims[3])
      error="case dimensions/threads do not match manifest";
  }
  if (error.empty() && s.policy!="Auto" && s.policy!="Square" && s.policy!="FullRow" && s.policy!="FullCol") error="unknown policy";
  if (error.empty() && fixed(s.consumer)!=bool(s.consumerGrid)) error="fixed consumers require --consumer-grid; other consumers forbid it";
  if (error.empty() && s.consumerGrid && (s.consumerGrid->first<=0 || s.consumerGrid->second<=0)) error="consumer grid must be positive";
  return success(error.empty());
}
std::string printModule(ModuleOp m) { std::string s; llvm::raw_string_ostream(s)<<m; return s; }

OwningOpRef<ModuleOp> buildFixture(MLIRContext &context,const CaseSpec &s,std::string &error) {
  if (failed(validateCase(s,error))) return {};
  context.loadDialect<FriskDialect,func::FuncDialect,arith::ArithDialect,memref::MemRefDialect>();
  std::string text; llvm::raw_string_ostream os(text);
  std::string a=(s.mode==MmaMode::RS ? "tensor<" : "memref<")+std::to_string(s.m)+"x"+std::to_string(s.k)+"xf16"+(s.mode==MmaMode::RS ? ">" : ",3>");
  std::string b="memref<"+std::to_string(s.k)+"x"+std::to_string(s.n)+"xf16,3>";
  std::string c="tensor<"+std::to_string(s.m)+"x"+std::to_string(s.n)+"xf32>";
  os<<"module attributes {frisk.target=\"sm_90a\"} { func.func @selection() {\n";
  if(s.mode==MmaMode::RS) os<<"%a=arith.constant dense<1.0> : "<<a<<"\n";
  else os<<"%ar=memref.alloc() {alignment=1024:i64} : "<<a<<"\n%a=frisk.layout_view %ar : "<<a<<" -> "<<a<<"\n";
  os<<"%br=memref.alloc() {alignment=1024:i64} : "<<b<<"\n%b=frisk.layout_view %br : "<<b<<" -> "<<b
    <<"\n%c=arith.constant dense<0.0> : "<<c<<"\n%r=\"frisk.mma\"(%a,%b,%c) {m="<<s.m
    <<":i64,n="<<s.n<<":i64,k="<<s.k<<":i64,frisk.execution_threads="<<s.threads
    <<":i64,policy=#frisk<gemm_warp_policy "<<s.policy<<">} : ("<<a<<","<<b<<","<<c<<") -> "<<c<<"\nreturn } }";
  auto module=parseSourceString<ModuleOp>(text,&context);
  if (!module) { error="fixture parse failed"; return {}; }
  if(s.consumer==Consumer::Local) return module;
  MmaOp mma; module->walk([&](MmaOp op){mma=op;});
  OpBuilder builder(&context); builder.setInsertionPointAfter(mma);
  Value source=mma.getResult();
  if(fixed(s.consumer)) {
    // Consumer identities come from the Auto legal geometry domain regardless
    // of the native policy being measured. This does not seed the MMA result.
    auto policy=mma.getPolicy(); mma.setPolicy(attr::GemmWarpPolicy::Auto);
    auto geometries=getSM90MmaGeometries(mma,s.threads); mma.setPolicy(policy);
    if(failed(geometries)) { error="cannot enumerate legal consumer geometries"; return {}; }
    auto found=llvm::find_if(*geometries,[&](const auto &g){return g.gM==s.consumerGrid->first && g.gN==s.consumerGrid->second;});
    if(found==geometries->end()) { error="consumer grid is not a legal geometry for this case"; return {}; }
    auto encoding=buildSM90MmaFragment(&context,*found,false,false);
    if(failed(encoding)) { error="canonical consumer fragment exceeds proof domain"; return {}; }
    auto type=RankedTensorType::get({s.m,s.n},builder.getF32Type(),*encoding);
    source=builder.create<ConvertLayoutOp>(mma.getLoc(),type,source);
  }
  auto shape=outputShape(s);
  if(reduction(s.consumer)) {
    auto reduce=builder.create<ReduceTensorOp>(mma.getLoc(),RankedTensorType::get(shape,builder.getF32Type()),source,"sum",axis(s.consumer));
    reduce->setAttr("frisk.execution_threads",builder.getI64IntegerAttr(s.threads));
    source=reduce.getResult();
  }
  auto storageType=MemRefType::get(shape,builder.getF32Type(),{},1);
  auto root=builder.create<memref::AllocOp>(mma.getLoc(),storageType);
  root->setAttr("alignment",builder.getI64IntegerAttr(128));
  int64_t elements=1; for(auto extent:shape) elements*=extent;
  auto address=builder.getAffineDimExpr(0)*(shape.size()==2 ? shape[1]*4 : 4);
  SmallVector<Attribute> names{builder.getStringAttr("dim0")};
  if(shape.size()==2) { address=address+builder.getAffineDimExpr(1)*4; names.push_back(builder.getStringAttr("dim1")); }
  auto map=AffineLayoutMapAttr::get(&context,builder.getArrayAttr(names),builder.getDenseI64ArrayAttr(shape),
    builder.getStrArrayAttr({"byte_offset","bit_offset"}),builder.getDenseI64ArrayAttr({elements*4,8}),
    AffineMapAttr::get(AffineMap::get(shape.size(),0,{address,builder.getAffineConstantExpr(0)},&context)));
  auto layout=StorageLayoutAttr::get(&context,map,MemorySpaceAttr::get(&context,attr::MemorySpace::Global),builder.getI64IntegerAttr(128),builder.getI64IntegerAttr(1));
  auto view=builder.create<LayoutViewOp>(mma.getLoc(),storageType,root,layout);
  builder.create<TileStoreOp>(mma.getLoc(),source,view.getResult());
  // A fixed target with an unencoded source is deliberately a pre-layout
  // analysis fixture. Production ConvertLayoutOp verification requires both
  // encodings; controlled experiments must materialize before full pipeline.
  if(!fixed(s.consumer) && failed(verify(*module))) { error="constructed fixture is invalid"; return {}; }
  return module;
}

LogicalResult verifyOutputABI(ModuleOp module,const CaseSpec &s,std::string &error) {
  error.clear();
  if(s.consumer==Consumer::Local) { error="local analysis has no output ABI"; return failure(); }
  SmallVector<TileStoreOp> stores; module.walk([&](TileStoreOp op){stores.push_back(op);});
  if(stores.size()!=1) { error="expected exactly one observable Global Store"; return failure(); }
  auto dest=stores.front()->getOperand(1);
  auto view=dest.getDefiningOp<LayoutViewOp>();
  auto type=dyn_cast<MemRefType>(dest.getType()); auto shape=outputShape(s);
  auto layout=view ? dyn_cast_or_null<StorageLayoutAttr>(view.getLayoutAttr()) : StorageLayoutAttr{};
  auto root=view ? view->getOperand(0).getDefiningOp<memref::AllocOp>() : memref::AllocOp{};
  if(!type || type.getShape()!=ArrayRef<int64_t>(shape) || !type.getElementType().isF32() ||
      !layout || layout.getMemorySpace().getValue()!=attr::MemorySpace::Global ||
      layout.getAlignment().getInt()!=128 || !root ||
      !root->getAttrOfType<IntegerAttr>("alignment") || root->getAttrOfType<IntegerAttr>("alignment").getInt()!=128) {
    error="output shape/type/Global storage/alignment does not match external ABI"; return failure();
  }
  int64_t elements=1; for(auto n:shape) elements*=n;
  for(int64_t i=0;i<elements;++i) {
    SmallVector<int64_t> coordinate=shape.size()==2 ? SmallVector<int64_t>{i/shape[1],i%shape[1]} : SmallVector<int64_t>{i};
    auto actual=evaluateLayoutMap(layout.getMap(),coordinate);
    if(failed(actual) || actual->size()!=2 || (*actual)[0]!=i*4 || (*actual)[1]!=0) {
      error="output byte address differs from row-major f32 ABI at linear coordinate "+std::to_string(i); return failure();
    }
  }
  return success();
}
} // namespace mlir::frisk::selection
