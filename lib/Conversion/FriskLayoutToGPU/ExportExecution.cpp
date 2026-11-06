#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/ShuffleNetworkJSON.h"
#include "Dialect/Frisk/Analysis/LinearExecutionLayout.h"
#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Dominance.h"
#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
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
StringRef proofName(ProofStatus status) {
 switch(status) {
 case ProofStatus::Proven:return "Proven";
 case ProofStatus::Disproven:return "Disproven";
 case ProofStatus::Unknown:return "Unknown";
 }
 llvm_unreachable("invalid layout proof status");
}
struct ExportExecutionPass: impl::ExportExecutionBase<ExportExecutionPass> {
 void runOnOperation() override {
  ModuleOp module=getOperation();
  const std::string objectiveName=networkObjective.getValue();
  if(objectiveName!="min-shfl" && objectiveName!="min-work"){module.emitError("unknown Shuffle network objective");signalPassFailure();return;}
  const auto objective=objectiveName=="min-work"?shuffle_network::Objective::MinWork:shuffle_network::Objective::MinShuffle;
  bool invalid=false;
  // Keep rejection stage, category, and proof state machine-readable without
  // publishing an invalid execution package or replacing an existing output.
  auto reject=[&](Operation *op,StringRef why,
                  StringRef code="unsupported-operation",
                  StringRef stage="export-preflight",
                  StringRef proof="not-applicable") {
   op->emitError()<<"[frisk-execution-export stage="<<stage<<" code="<<code
                  <<" proof="<<proof<<"] "<<why;
   invalid=true;
  };
  auto rejectProof=[&](Operation *op,StringRef why,StringRef code,
                       const LayoutProof &proof) {
   reject(op,why.str()+": "+proof.reason,code,"export-proof",proofName(proof.status));
  };
  if(failed(verify(module))) {
   reject(module,"module verification failed","invalid-ir","ir-verification","not-Proven");
   signalPassFailure();return;
  }
  // Preflight every op before file creation, including unused/unknown operations.
  module.walk([&](Operation *op){
   StringRef n=op->getName().getStringRef();
   if (n == "arith.addf" || n == "math.rsqrt") {
    unsigned arity = n == "arith.addf" ? 2 : 1;
    auto result = op->getNumResults() == 1
        ? dyn_cast<RankedTensorType>(op->getResult(0).getType()) : RankedTensorType();
    if (op->getNumOperands() != arity || !result || !result.hasStaticShape() ||
        !result.getElementType().isF32() ||
        llvm::any_of(op->getOperandTypes(), [&](Type type) { return type != result; }))
     reject(op, "requires matching static f32 tensor operands and result",
            "elementwise-type");
    return;
   }
   if(!llvm::is_contained(ArrayRef<StringRef>{"builtin.module","frisk.kernel","frisk.parallel","frisk.end","frisk.storage_contract","frisk.layout_view","memref.alloc","frisk.copy","frisk.mma","frisk.reduce_tensor","frisk.tile_load","frisk.tile_store","frisk.convert_layout","arith.constant","arith.mulf","arith.subf","arith.divf","arith.truncf","math.exp","tensor.empty","linalg.broadcast","linalg.yield"},n))
    reject(op,"unsupported execution operation");
  });
  int kernels=0,parallels=0;
  module.walk([&](KernelOp op){++kernels;if(!llvm::hasSingleElement(op.getRegion()))reject(op,"requires a single-block kernel","kernel-region");});
  module.walk([&](ParallelOp op){++parallels;if(op.getThreads()!=128 || op.getRanges().size()!=1 || op.getRanges()[0]!=1 || !isa<KernelOp>(op->getParentOp()))reject(op,"requires one 128-thread single-CTA parallel","execution-topology");});
  auto target=module->getAttrOfType<StringAttr>("frisk.target");
  if(kernels!=1||parallels!=1||!target||target.getValue()!="sm_90a")reject(module,"requires one sm_90a kernel/parallel","target-envelope");
  if(!invalid){auto target=createSM90LayoutTarget();if(failed(verifyMaterializedLayouts(module,*target)))reject(module,"materialized layout verification failed","materialized-layout","layout-verification","not-Proven");}
  if(invalid){signalPassFailure();return;}
  llvm::DenseMap<Value,int64_t> ids;
  SmallVector<Value> values;
  auto id=[&](Value v)->int64_t {auto [i,newValue]=ids.try_emplace(v,ids.size());if(newValue)values.push_back(v);return i->second;};
  llvm::json::Array operations;
  bool compactUsed=false;
  auto executionJSON=[&](DistributedEncodingAttr encoding,RankedTensorType type,
                         Operation *op,StringRef code,bool tensor)
      ->std::optional<llvm::json::Object> {
   if(!encoding) {
    reject(op,"execution requires a selected layout",code,"export-proof","not-Proven");
    return std::nullopt;
   }
   ExecutionEnumeration enumeration;
   auto proof=enumerateExecutionLayout(encoding,type,enumeration);
   llvm::json::Object object;
   if(proof.status==ProofStatus::Proven) {
    object=llvm::json::Object{{"topology",numbers(enumeration.topology)},
      {"shape",numbers(enumeration.shape)},{"logical",numbers(enumeration.logical)},
      {"first",numbers(enumeration.first)}};
   } else {
    // Budget exhaustion is not a counterexample. Prove the complete map with
    // algebra, then serialize its canonical rows for on-demand holder queries.
    LinearExecutionLayout linear;
    auto representation=analyzeLinearExecutionLayout(encoding,type,linear);
    if(representation.status!=ProofStatus::Proven) {
     rejectProof(op,"execution representation is not Proven",code,representation);
     return std::nullopt;
    }
    auto ownership=proveExecutionOwnership(encoding,type,"first_owner");
    if(ownership.status!=ProofStatus::Proven) {
     rejectProof(op,"execution ownership is not Proven",code,ownership);
     return std::nullopt;
    }
    if(proof.status!=ProofStatus::Unknown || linear.hardwareBits<=16 || linear.logicalBits>16) {
     rejectProof(op,"execution table is not Proven",code,proof);
     return std::nullopt;
    }
    llvm::json::Array rows;
    for(const auto &row:linear.rows) {
     llvm::SmallString<80> digits;row.toString(digits,16,false);
     rows.push_back(digits.str().str());
    }
    llvm::json::Object symbolic{{"version","gf2-canonical-v1"},
      {"hardware_bits",linear.hardwareBits},{"logical_bits",linear.logicalBits},
      {"hardware_widths",numbers(linear.hardwareWidths)},
      {"logical_widths",numbers(linear.logicalWidths)},{"rows_hex",std::move(rows)}};
    object=llvm::json::Object{{"topology",numbers(linear.topology)},
      {"shape",numbers(type.getShape())},{"symbolic",std::move(symbolic)},
      {"replication",encoding.getReplication().getInt()}};
    compactUsed=true;
   }
   if(tensor) {
    object["replication"]=encoding.getReplication().getInt();
    object["encoding"]=text(encoding);
   }
   return object;
  };

  module.walk<WalkOrder::PreOrder>([&](Operation *op){
   for(Region &r:op->getRegions()) for(Block &b:r)for(BlockArgument a:b.getArguments())id(a);
   llvm::json::Array operands,results;
   for(Value v:op->getOperands())operands.push_back(id(v));
   for(Value v:op->getResults())results.push_back(id(v));
   llvm::json::Object attrs;
   for(NamedAttribute a:op->getAttrs())attrs[a.getName().strref()]=attributeJSON(a.getValue());
   llvm::json::Object o{{"name",op->getName().getStringRef().str()},{"operands",std::move(operands)},{"results",std::move(results)},{"attrs",std::move(attrs)}};
   if(auto conversion=dyn_cast<ConvertLayoutOp>(op)) {
    auto source=cast<RankedTensorType>(op->getOperand(0).getType());
    auto destination=cast<RankedTensorType>(op->getResult(0).getType());
    auto se=cast<DistributedEncodingAttr>(source.getEncoding());
    auto p=se.getTopology()[0]>512
      ? test::planSymbolicRegisterRedistribution(source,destination,op->getLoc())
      : test::planRedistribution(source,destination,op->getLoc(),objective);
    if(failed(p)){reject(op,"redistribution plan failed","redistribution-plan","export-proof","not-Proven");return;}
    auto owner=[](test::Owner x){return llvm::json::Object{{"reg",x.reg},{"lane",x.lane},{"warp",x.warp},{"warp_group",x.warpGroup}};};
    llvm::json::Array destinations,writers,words;
    for(auto d:p->destinations)destinations.push_back(llvm::json::Object{{"source",owner(d.source)},{"slot",int64_t(d.slot)}});
    for(auto w:p->globalWriters)writers.push_back(owner(w));
    for(auto w:p->payload.words)words.push_back(llvm::json::Object{{"bit_offset",w.bitOffset},{"live_bits",w.liveBits}});
    o["redistribution"]=llvm::json::Object{{"mode",p->mode==test::ExchangeMode::Register?"Register":p->mode==test::ExchangeMode::Shuffle?"Shuffle":"Shared"},
      {"threads",p->threads},{"warps",p->warps},{"source_registers",p->sourceRegisters},{"destination_registers",p->destinationRegisters},
      {"volume",int64_t(p->volume)},{"scratch_bytes",int64_t(p->scratchBytes)},
      {"payload",llvm::json::Object{{"bits",p->payload.bits},{"words",std::move(words)}}},
      {"destinations",std::move(destinations)},{"source_slots",numbers(p->sourceSlots)},{"global_writers",std::move(writers)}};
    if(p->mode==test::ExchangeMode::Shuffle)
      (*o.getObject("redistribution"))["shuffle_network_objective"]=objectiveName;
    if(p->shuffleNetwork) {
      auto *redistribution=o.getObject("redistribution");
      (*redistribution)["shuffle_network"]=shuffle_network::planJSON(*p->shuffleNetwork);
      auto resources=shuffle_network::structuralResources(*p->shuffleNetwork);
      (*redistribution)["shuffle_network_resource_proxy"]=llvm::json::Object{
        {"contract","emitted-network-definition-proxy-v1"},{"logical_carrier_words",resources.logicalCarrierWords},
        {"b32_temporary_definitions",resources.b32TemporaryDefinitions},{"predicate_definitions",resources.predicateDefinitions},
        {"communication_stages",resources.communicationStages},{"physical_register_prediction",false}};
    }
   }
   if(auto plan=o.getObject("redistribution");plan && plan->getArray("source_slots")->empty()) {
    // The dedicated large-source API leaves this table absent. Consumers must
    // validate its bounded destination owners against the symbolic source map.
    plan->erase("source_slots");
    (*plan)["source_slots_kind"]="source-distributed-symbolic-v1";
    compactUsed=true;
   }
   if(auto copy=dyn_cast<CopyOp>(op)) {
    auto enc=op->getAttrOfType<DistributedEncodingAttr>("frisk.execution_layout");
    auto ty=cast<MemRefType>(copy.getSrc().getType());
    auto execution=executionJSON(enc,RankedTensorType::get(ty.getShape(),ty.getElementType()),op,"copy-execution",false);
    if(!execution)return;
    o["execution"]=std::move(*execution);
   }
   if(auto reduce=dyn_cast<ReduceTensorOp>(op)) {
    auto src=cast<RankedTensorType>(op->getOperand(0).getType()),dst=cast<RankedTensorType>(op->getResult(0).getType());
    auto se=dyn_cast_or_null<DistributedEncodingAttr>(src.getEncoding()),de=dyn_cast_or_null<DistributedEncodingAttr>(dst.getEncoding());
    if(!se||!de){reject(op,"reduction requires selected encodings","reduction-layout","export-proof","not-Proven");return;}
    auto proof=buildReductionLayoutProof(se,de,src,dst,reduce.getDim());
    if(proof.proof.status != ProofStatus::Proven){rejectProof(op,"reduction dependency proof is not Proven","reduction-dependency",proof.proof);return;}
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
  auto rootContracts=collectStorageRootContracts(module);
  if(failed(rootContracts)){reject(module,"storage root contracts are not Proven","storage-root-contract","export-proof","not-Proven");signalPassFailure();return;}
  DominanceInfo dominance(module);
  llvm::json::Array serialized;
  for(Value v:values){
   llvm::json::Object o{{"id",ids.lookup(v)},{"type",text(v.getType())}};
   if(auto value=dyn_cast<OpResult>(v)) {
    for(auto pair:{std::make_pair("frisk.execution_audit_variables","layout_variable_ids"),std::make_pair("frisk.execution_audit_boundaries","layout_boundary_ids")})
     if(auto arrays=value.getOwner()->getAttrOfType<ArrayAttr>(pair.first)) {
      if(arrays.size()!=value.getOwner()->getNumResults()){reject(module,"execution demand lineage arity mismatch","demand-lineage","export-proof","invalid");continue;}
      auto entries=dyn_cast<DenseI64ArrayAttr>(arrays[value.getResultNumber()]);
      if(!entries){reject(module,"execution demand lineage requires integer identities","demand-lineage","export-proof","invalid");continue;}
      if(!entries.empty()){llvm::json::Array identities;for(int64_t identity:entries.asArrayRef())identities.push_back(identity);o[pair.second]=std::move(identities);}
     }
   }
   if(auto type=dyn_cast<RankedTensorType>(v.getType())){
    auto enc=dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding());
    auto execution=executionJSON(enc,type,module,"tensor-execution",true);
    if(!execution)continue;
    o["distributed"]=std::move(*execution);
   }
   if(isa<MemRefType>(v.getType())){
    StorageLayoutAttr enc;
    if(auto view=v.getDefiningOp<LayoutViewOp>())enc=view.getLayout().value_or(StorageLayoutAttr());
    if(!enc)for(Operation *user:v.getUsers())if(auto contract=dyn_cast<StorageContractOp>(user))enc=contract.getLayout();
    if(enc){
     auto alias=analyzeStorageAlias(v);
     if(failed(alias)){reject(module,"storage alias is not supported","storage-alias","export-proof","not-Proven");continue;}
     auto compact=matchContiguousIdentityStorageAlias(*alias,enc);
     bool compactRoot=false;
     // Only explicit validated root declarations may strengthen the ABI.
     // A view must be dominated by that declaration; root serialization uses
     // its own complete declaration as the external launch precondition.
     for(const auto &contract:*rootContracts) {
      if(contract.root.root!=alias->root)continue;
      Operation *anchor=v.getDefiningOp();
      if((v==alias->root && enc==contract.layout) ||
         (anchor && storageContractDominates(contract.declaration,anchor,dominance))) {
       alias->rootAlignment=std::max(alias->rootAlignment,contract.root.rootAlignment);
       // Compare the actual validated root map, not merely shape/root identity.
       if(compact && matchContiguousIdentityStorageAlias(contract.root,
             cast<StorageLayoutAttr>(contract.layout)))compactRoot=true;
      }
     }
     llvm::json::Object storage{{"layout",text(enc)},{"alignment",enc.getAlignment().getInt()},{"root",id(alias->root)}};
     if(compact && compactRoot) {
      storage["addresses_compact"]=llvm::json::Object{
        {"contract","contiguous-identity-addresses-v1"},
        {"shape",llvm::json::Array{compact->rows,compact->columns}},
        {"element_bits",32},{"base_bit",0},
        {"strides_bits",llvm::json::Array{compact->columns*32,int64_t(32)}}};
      compactUsed=true;
     } else {
      auto footprint=buildStorageAliasFootprint(*alias,enc);
      if(footprint.proof.status != ProofStatus::Proven){rejectProof(module,"storage footprint is not Proven","storage-footprint",footprint.proof);continue;}
      llvm::json::Array addresses;
      for(auto &a:footprint.entries)addresses.push_back(llvm::json::Object{{"view",numbers(a.view)},{"root",numbers(a.root)},{"begin_bit",int64_t(a.begin)},{"end_bit",int64_t(a.end)}});
      storage["addresses"]=std::move(addresses);
     }
     o["storage"]=std::move(storage);
    }
   }
   serialized.push_back(std::move(o));
  }
  if(invalid){signalPassFailure();return;}
  if(output.empty()){reject(module,"requires output JSON path","output-path","export-publication");signalPassFailure();return;}
  llvm::json::Object result{{"schema_version",compactUsed?2:1},{"values",std::move(serialized)},
    {"operations",std::move(operations)}};
  llvm::SmallString<256> temporary;int fd;
  if(auto ec=llvm::sys::fs::createUniqueFile(output+".tmp-%%%%%%",fd,temporary)) {reject(module,ec.message(),"output-create","export-publication");signalPassFailure();return;}
  bool writeFailed=false;
  {llvm::raw_fd_ostream stream(fd,true);stream<<llvm::formatv(minify?"{0}":"{0:2}",llvm::json::Value(std::move(result)));stream.flush();writeFailed=stream.has_error();stream.clear_error();}
  if(writeFailed||llvm::sys::fs::rename(temporary,output)){llvm::sys::fs::remove(temporary);reject(module,"atomic execution export failed","output-publish","export-publication");signalPassFailure();}
 }
};
}
std::unique_ptr<Pass> createExportExecutionPass(){return std::make_unique<ExportExecutionPass>();}
}
