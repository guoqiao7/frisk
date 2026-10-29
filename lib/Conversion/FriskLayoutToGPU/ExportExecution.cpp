#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
namespace mlir::frisk {
#define GEN_PASS_DEF_EXPORTEXECUTION
#include "Dialect/Frisk/Transforms/Passes.h.inc"
namespace {
template<class Range> llvm::json::Array numbers(const Range &r) {
 llvm::json::Array a; for (auto x:r) a.push_back(int64_t(x)); return a;
}
std::string text(Attribute a) { std::string s; llvm::raw_string_ostream os(s); os<<a; return s; }
std::string text(Type a) { std::string s; llvm::raw_string_ostream os(s); os<<a; return s; }
llvm::json::Value attributeJSON(Attribute a) {
 if(auto x=dyn_cast<IntegerAttr>(a)) return x.getInt();
 if(auto x=dyn_cast<StringAttr>(a)) return x.getValue().str();
 if(auto x=dyn_cast<DenseI64ArrayAttr>(a)) return numbers(x.asArrayRef());
 if(auto x=dyn_cast<MmaInstructionContractAttr>(a))return attributeJSON(x.getPayload());
 if(auto x=dyn_cast<MmaDescriptorPlanAttr>(a))return attributeJSON(x.getPayload());
 if(auto x=dyn_cast<ReductionContractAttr>(a))return attributeJSON(x.getPayload());
 if(auto x=dyn_cast<DictionaryAttr>(a)){llvm::json::Object o;for(auto n:x)o[n.getName().strref()]=attributeJSON(n.getValue());return o;}
 return text(a);
}
struct ExportExecutionPass: impl::ExportExecutionBase<ExportExecutionPass> {
 void runOnOperation() override {
  ModuleOp module=getOperation();
  if(failed(verify(module))) {signalPassFailure();return;}
  bool invalid=false;
  auto reject=[&](Operation *op,StringRef why) {op->emitError(why);invalid=true;};
  // Preflight every op before file creation, including unused/unknown operations.
  module.walk([&](Operation *op){
   StringRef n=op->getName().getStringRef();
   if(!llvm::is_contained(ArrayRef<StringRef>{"builtin.module","frisk.kernel","frisk.parallel","frisk.end","frisk.storage_contract","frisk.layout_view","memref.alloc","frisk.copy","frisk.mma","frisk.reduce_tensor","frisk.tile_load","frisk.tile_store","frisk.convert_layout","arith.constant","arith.mulf","arith.subf","arith.divf","arith.truncf","math.exp","tensor.empty","linalg.broadcast","linalg.yield"},n))
    reject(op,"unsupported execution operation");
  });
  int kernels=0,parallels=0;
  module.walk([&](KernelOp op){++kernels;if(!llvm::hasSingleElement(op.getRegion()))reject(op,"requires a single-block kernel");});
  module.walk([&](ParallelOp op){++parallels;if(op.getThreads()!=128 || op.getRanges().size()!=1 || op.getRanges()[0]!=1 || !isa<KernelOp>(op->getParentOp()))reject(op,"requires one 128-thread single-CTA parallel");});
  auto target=module->getAttrOfType<StringAttr>("frisk.target");
  if(kernels!=1||parallels!=1||!target||target.getValue()!="sm_90a")reject(module,"requires one sm_90a kernel/parallel");
  if(!invalid){auto target=createSM90LayoutTarget();if(failed(verifyMaterializedLayouts(module,*target)))invalid=true;}
  if(invalid){signalPassFailure();return;}
  llvm::DenseMap<Value,int64_t> ids;
  SmallVector<Value> values;
  auto id=[&](Value v)->int64_t {auto [i,newValue]=ids.try_emplace(v,ids.size());if(newValue)values.push_back(v);return i->second;};
  llvm::json::Array operations;
  module.walk<WalkOrder::PreOrder>([&](Operation *op){
   for(Region &r:op->getRegions()) for(Block &b:r)for(BlockArgument a:b.getArguments())id(a);
   llvm::json::Array operands,results;
   for(Value v:op->getOperands())operands.push_back(id(v));
   for(Value v:op->getResults())results.push_back(id(v));
   llvm::json::Object attrs;
   for(NamedAttribute a:op->getAttrs())attrs[a.getName().strref()]=attributeJSON(a.getValue());
   llvm::json::Object o{{"name",op->getName().getStringRef().str()},{"operands",std::move(operands)},{"results",std::move(results)},{"attrs",std::move(attrs)}};
   if(auto copy=dyn_cast<CopyOp>(op)) {
    auto enc=op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout");
    auto ty=cast<MemRefType>(copy.getSrc().getType());
    ExecutionEnumeration e;
    if(!enc || enumerateExecutionLayout(enc,RankedTensorType::get(ty.getShape(),ty.getElementType(),enc),e).status!=ProofStatus::Proven){reject(op,"copy execution map is not Proven");return;}
    o["execution"]=llvm::json::Object{{"topology",numbers(e.topology)},{"shape",numbers(e.shape)},{"logical",numbers(e.logical)},{"first",numbers(e.first)}};
   }
   if(auto reduce=dyn_cast<ReduceTensorOp>(op)) {
    auto src=cast<RankedTensorType>(op->getOperand(0).getType()),dst=cast<RankedTensorType>(op->getResult(0).getType());
    auto se=dyn_cast_or_null<DistributedEncodingAttr>(src.getEncoding()),de=dyn_cast_or_null<DistributedEncodingAttr>(dst.getEncoding());
    if(!se||!de){reject(op,"reduction requires selected encodings");return;}
    auto proof=buildReductionLayoutProof(se,de,src,dst,reduce.getDim());
    if(proof.proof.status != ProofStatus::Proven){reject(op,"reduction dependency proof is not Proven");return;}
    llvm::json::Array fibers;
    for(auto &f:proof.fibers){
     llvm::json::Array nodes,broadcasts;
     for(auto &n:f.nodes)nodes.push_back(llvm::json::Object{{"logicalInput",n.logicalInput},{"left",n.left},{"right",n.right},{"holder",numbers(n.holder)}});
     for(auto &h:f.broadcasts)broadcasts.push_back(numbers(h));
     fibers.push_back(llvm::json::Object{{"logicalOutput",f.logicalOutput},{"root",f.root},{"nodes",std::move(nodes)},{"broadcasts",std::move(broadcasts)}});
    }
    o["fibers"]=std::move(fibers);o["scope"]=proof.scope;
   }
   operations.push_back(std::move(o));
  });
  llvm::json::Array serialized;
  for(Value v:values){
   llvm::json::Object o{{"id",ids.lookup(v)},{"type",text(v.getType())}};
   if(auto type=dyn_cast<RankedTensorType>(v.getType())){
    auto enc=dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding());
    ExecutionEnumeration e;
    if(!enc||enumerateExecutionLayout(enc,type,e).status != ProofStatus::Proven){reject(module,"tensor execution enumeration is not Proven");continue;}
    o["distributed"]=llvm::json::Object{{"topology",numbers(e.topology)},{"shape",numbers(e.shape)},{"logical",numbers(e.logical)},{"first",numbers(e.first)},{"replication",enc.getReplication().getInt()},{"encoding",text(enc)}};
   }
   if(isa<MemRefType>(v.getType())){
    StorageLayoutAttr enc;
    if(auto view=v.getDefiningOp<LayoutViewOp>())enc=view.getLayout().value_or(StorageLayoutAttr());
    if(!enc)for(Operation *user:v.getUsers())if(auto contract=dyn_cast<StorageContractOp>(user))enc=contract.getLayout();
    if(enc){
     auto alias=analyzeStorageAlias(v);
     if(failed(alias)){reject(module,"storage alias is not supported");continue;}
     auto footprint=buildStorageAliasFootprint(*alias,enc);
     if(footprint.proof.status != ProofStatus::Proven){reject(module,"storage footprint is not Proven");continue;}
     llvm::json::Array addresses;
     for(auto &a:footprint.entries)addresses.push_back(llvm::json::Object{{"view",numbers(a.view)},{"root",numbers(a.root)},{"begin_bit",int64_t(a.begin)},{"end_bit",int64_t(a.end)}});
     o["storage"]=llvm::json::Object{{"layout",text(enc)},{"alignment",enc.getAlignment().getInt()},{"root",id(alias->root)},{"addresses",std::move(addresses)}};
    }
   }
   serialized.push_back(std::move(o));
  }
  if(invalid){signalPassFailure();return;}
  if(output.empty()){module.emitError("requires output JSON path");signalPassFailure();return;}
  llvm::json::Object result{{"schema_version",1},{"values",std::move(serialized)},{"operations",std::move(operations)}};
  llvm::SmallString<256> temporary;int fd;
  if(auto ec=llvm::sys::fs::createUniqueFile(output+".tmp-%%%%%%",fd,temporary)) {module.emitError(ec.message());signalPassFailure();return;}
  bool writeFailed=false;
  {llvm::raw_fd_ostream stream(fd,true);stream<<llvm::formatv("{0:2}",llvm::json::Value(std::move(result)));stream.flush();writeFailed=stream.has_error();stream.clear_error();}
  if(writeFailed||llvm::sys::fs::rename(temporary,output)){llvm::sys::fs::remove(temporary);module.emitError("atomic execution export failed");signalPassFailure();}
 }
};
}
std::unique_ptr<Pass> createExportExecutionPass(){return std::make_unique<ExportExecutionPass>();}
}
