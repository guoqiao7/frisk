#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "llvm/Support/MathExtras.h"
#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/JSON.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
class ExecutionExportTest: public testing::Test {
protected:
 MLIRContext context;
 llvm::SmallString<256> path;
 ExecutionExportTest(){context.loadDialect<FriskDialect,arith::ArithDialect,linalg::LinalgDialect,math::MathDialect,memref::MemRefDialect,tensor::TensorDialect>();}
 void SetUp() override {ASSERT_FALSE(bool(llvm::sys::fs::createTemporaryFile("frisk-execution", "json",path)));}
 void TearDown() override {llvm::sys::fs::remove(path);}
 LogicalResult run(ModuleOp m){PassManager pm(&context);auto p=createExportExecutionPass();if(failed(p->initializeOptions("output="+path.str().str(),[](const Twine &){return failure();})))return failure();pm.addPass(std::move(p));return pm.run(m);}
 std::string print(ModuleOp m){std::string s;llvm::raw_string_ostream os(s);os<<m;return s;}
 OwningOpRef<ModuleOp> selected(){auto m=parseSourceFile<ModuleOp>(FRISK_ATTENTION_FIXED_ABI_FIXTURE,&context);if(!m)return m;PassManager pm(&context);pm.addPass(createFriskInferLayoutsPass());if(failed(pm.run(*m)))return {};return m;}
};
TEST_F(ExecutionExportTest, ActualAttentionExportsCanonicalTreeWithoutMutation){
 auto m=selected();ASSERT_TRUE(m);auto before=print(*m);ASSERT_TRUE(succeeded(run(*m)));EXPECT_EQ(before,print(*m));auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));auto *object=parsed->getAsObject();ASSERT_NE(object,nullptr);auto *operations=object->getArray("operations");ASSERT_NE(operations,nullptr);unsigned reductions=0;
 for(auto &entry:*operations){auto *o=entry.getAsObject();if(o->getString("name")!="frisk.reduce_tensor")continue;++reductions;auto *fibers=o->getArray("fibers");ASSERT_NE(fibers,nullptr);EXPECT_EQ(fibers->size(),64u);for(auto &f:*fibers){auto *fo=f.getAsObject();EXPECT_EQ(fo->getArray("nodes")->size(),127u);EXPECT_EQ(fo->getArray("broadcasts")->size(),4u);}}
 EXPECT_EQ(reductions,2u);
}
TEST_F(ExecutionExportTest, RejectsForgedTargetWithoutPublishingOrMutating){
 auto m=selected();ASSERT_TRUE(m);m->getOperation()->setAttr("frisk.target",StringAttr::get(&context,"sm_80"));auto before=print(*m);{std::error_code ec;llvm::raw_fd_ostream os(path,ec);ASSERT_FALSE(bool(ec));os<<"preserve";}
 ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});EXPECT_TRUE(failed(run(*m)));EXPECT_EQ(before,print(*m));auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));EXPECT_EQ((*bytes)->getBuffer(),"preserve");
}
TEST_F(ExecutionExportTest, StrongAsyncABIExportsActualRootAlignment){
 for(auto suffix:{"cp-async","tma"}) {
  std::string fixture=FRISK_ATTENTION_FIXED_ABI_FIXTURE;
  fixture.replace(fixture.find("fixed"),5,suffix);
  auto m=parseSourceFile<ModuleOp>(fixture,&context);ASSERT_TRUE(m);
  PassManager pm(&context);pm.addPass(createFriskInferLayoutsPass());ASSERT_TRUE(succeeded(pm.run(*m)));
  ASSERT_TRUE(succeeded(run(*m)));
  auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));
  auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));
  unsigned strong=0;
  for(auto &v:*parsed->getAsObject()->getArray("values")) {
   auto *o=v.getAsObject();auto *storage=o->getObject("storage");
   if(storage && storage->getInteger("alignment")==128)++strong;
  }
  EXPECT_GE(strong,6u);
 }
}

TEST_F(ExecutionExportTest, EveryConversionExportsCheckedPlanFields) {
 auto m=selected();ASSERT_TRUE(m);
 Value first; m->walk([&](Operation *op){for(Value value:op->getResults())if(!first && isa<RankedTensorType>(value.getType()))first=value;});ASSERT_TRUE(first);
 OpBuilder builder(first.getDefiningOp());builder.setInsertionPointAfter(first.getDefiningOp());
 OperationState state(first.getLoc(),"frisk.convert_layout");state.addOperands(first);state.addTypes(first.getType());
 auto *conversion=builder.create(state);first.replaceAllUsesExcept(conversion->getResult(0),conversion);
 ASSERT_TRUE(succeeded(run(*m)));
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));
 unsigned count=0;
 for(auto &entry:*parsed->getAsObject()->getArray("operations")) {
  auto *o=entry.getAsObject();if(o->getString("name")!="frisk.convert_layout")continue;++count;
  auto *p=o->getObject("redistribution");ASSERT_NE(p,nullptr);
  auto expected=test::planRedistribution(cast<RankedTensorType>(first.getType()),cast<RankedTensorType>(conversion->getResult(0).getType()),first.getLoc());ASSERT_TRUE(succeeded(expected));
  EXPECT_EQ(p->getString("mode"),"Register");EXPECT_EQ(p->getInteger("threads"),expected->threads);EXPECT_EQ(p->getInteger("warps"),expected->warps);
  EXPECT_EQ(p->getInteger("source_registers"),expected->sourceRegisters);EXPECT_EQ(p->getInteger("destination_registers"),expected->destinationRegisters);EXPECT_EQ(p->getInteger("volume"),expected->volume);EXPECT_EQ(p->getInteger("scratch_bytes"),expected->scratchBytes);
  auto *payload=p->getObject("payload");ASSERT_NE(payload,nullptr);EXPECT_EQ(payload->getInteger("bits"),expected->payload.bits);ASSERT_EQ(payload->getArray("words")->size(),expected->payload.words.size());
  for(auto [i,w]:llvm::enumerate(expected->payload.words)){auto *word=(*payload->getArray("words"))[i].getAsObject();EXPECT_EQ(word->getInteger("bit_offset"),w.bitOffset);EXPECT_EQ(word->getInteger("live_bits"),w.liveBits);}
  auto checkOwner=[](llvm::json::Object *o,test::Owner w){ASSERT_NE(o,nullptr);EXPECT_EQ(o->getInteger("reg"),w.reg);EXPECT_EQ(o->getInteger("lane"),w.lane);EXPECT_EQ(o->getInteger("warp"),w.warp);EXPECT_EQ(o->getInteger("warp_group"),w.warpGroup);};
  ASSERT_EQ(p->getArray("destinations")->size(),expected->destinations.size());
  for(auto [i,d]:llvm::enumerate(expected->destinations)){auto *entry=(*p->getArray("destinations"))[i].getAsObject();EXPECT_EQ(entry->getInteger("slot"),d.slot);checkOwner(entry->getObject("source"),d.source);}
  ASSERT_EQ(p->getArray("source_slots")->size(),expected->sourceSlots.size());for(auto [i,slot]:llvm::enumerate(expected->sourceSlots))EXPECT_EQ((*p->getArray("source_slots"))[i].getAsInteger(),slot);
  ASSERT_EQ(p->getArray("global_writers")->size(),expected->globalWriters.size());for(auto [i,w]:llvm::enumerate(expected->globalWriters))checkOwner((*p->getArray("global_writers"))[i].getAsObject(),w);
 }
 EXPECT_GT(count,0u);
}

// Build actual mixed bit-linear encodings. Columns refer to canonical hardware
// order; the optional input reversal checks representation-order independence.
RankedTensorType executionType(MLIRContext &context,ArrayRef<int64_t> shape,
                              unsigned registers,ArrayRef<uint64_t> columns,
                              bool reverse=false) {
 Builder b(&context);unsigned rb=llvm::Log2_64(registers),logicalBits=0;
 SmallVector<int64_t> widths;SmallVector<StringRef> outputs;
 for(auto [index,extent]:llvm::enumerate(shape)) {
  widths.push_back(llvm::Log2_64(extent));logicalBits+=widths.back();
 }
 SmallVector<std::string> outputStorage;
 for(unsigned i=0;i<shape.size();++i)outputStorage.push_back("dim"+std::to_string(i));
 for(auto &name:outputStorage)outputs.push_back(name);
 SmallVector<StringRef> names={"register","lane","warp"};
 SmallVector<int64_t> inputWidths={int64_t(rb),5,2};
 SmallVector<unsigned> canonical;
 for(unsigned i=0;i<columns.size();++i)canonical.push_back(i);
 if(reverse) {
  names={"warp","lane","register"};inputWidths={2,5,int64_t(rb)};canonical.clear();
  for(unsigned i=rb+5;i<rb+7;++i)canonical.push_back(i);
  for(unsigned i=rb;i<rb+5;++i)canonical.push_back(i);
  for(unsigned i=0;i<rb;++i)canonical.push_back(i);
 }
 SmallVector<APInt> values;
 for(unsigned row=0;row<logicalBits;++row)
  for(unsigned column:canonical)values.emplace_back(1,(columns[column]>>row)&1);
 auto map=BitLinearLayoutMapAttr::get(&context,b.getStrArrayAttr(names),b.getDenseI64ArrayAttr(inputWidths),
     b.getStrArrayAttr(outputs),b.getDenseI64ArrayAttr(widths),DenseIntElementsAttr::get(
       RankedTensorType::get({int64_t(logicalBits),int64_t(columns.size())},b.getI1Type()),values));
 auto encoding=DistributedEncodingAttr::get(&context,map,b.getDenseI64ArrayAttr({registers,32,4,1,1}),
     b.getI64IntegerAttr(uint64_t(registers)*128/(uint64_t{1}<<logicalBits)));
 return RankedTensorType::get(shape,b.getF32Type(),encoding);
}

TEST_F(ExecutionExportTest, SymbolicLargeRegisterPlanUsesExactThreadLocalOwners) {
 for(bool reverse:{false,true}) {
  SmallVector<uint64_t> source(17,0),destination(10,0);
  for(unsigned i=0;i<10;++i){source[i]=uint64_t{1}<<i;destination[i]=uint64_t{1}<<i;}
  // Non-permutation target columns are an invertible change of basis.
  destination[1]^=destination[0];destination[7]^=destination[3];
  auto src=executionType(context,{32,32},1024,source,reverse);
  auto dst=executionType(context,{32,32},8,destination,!reverse);
  auto plan=test::planSymbolicRegisterRedistribution(src,dst,UnknownLoc::get(&context));ASSERT_TRUE(succeeded(plan));
  EXPECT_TRUE(plan->symbolicSource);EXPECT_TRUE(plan->sourceSlots.empty());
  EXPECT_EQ(plan->mode,test::ExchangeMode::Register);EXPECT_EQ(plan->scratchBytes,0u);
  ASSERT_EQ(plan->destinations.size(),1024u);ASSERT_EQ(plan->globalWriters.size(),1024u);
  for(unsigned hardware=0;hardware<1024;++hardware) {
   uint64_t packed=0;for(unsigned bit=0;bit<destination.size();++bit)if(hardware>>bit&1)packed^=destination[bit];
   auto d=plan->destinations[hardware];unsigned thread=hardware/8;
   EXPECT_EQ(d.source.reg,packed);EXPECT_EQ(plan->thread(d.source),thread);
   EXPECT_EQ(d.slot,(packed&31)*32+(packed>>5));
  }
  for(unsigned slot=0;slot<1024;++slot) {
   auto writer=plan->globalWriters[slot];EXPECT_EQ(plan->thread(writer),0u);
   EXPECT_EQ(writer.reg,slot/32+(slot%32)*32);
  }
 }
}

TEST_F(ExecutionExportTest, SymbolicRegisterPlanAgreesWithConcreteSmallOracle) {
 // Source columns [3,1,2] create nonzero dependencies between register and
 // lane bits; enumerate each source thread independently as the oracle.
 SmallVector<uint64_t> source(9,0),destination(9,0);
 source[0]=3;source[1]=1;source[2]=2;destination=source;
 auto src=executionType(context,{4},4,source,true),dst=executionType(context,{4},4,destination);
 auto plan=test::planSymbolicRegisterRedistribution(src,dst,UnknownLoc::get(&context));ASSERT_TRUE(succeeded(plan));
 auto logical=[&](unsigned hardware){uint64_t slot=0;for(unsigned bit=0;bit<source.size();++bit)if(hardware>>bit&1)slot^=source[bit];return slot;};
 for(unsigned hardware=0;hardware<512;++hardware) {
  unsigned thread=hardware/4;auto point=plan->destinations[hardware];unsigned expected=0;
  while(expected<4 && logical(thread*4+expected)!=logical(hardware))++expected;
  ASSERT_LT(expected,4u);EXPECT_EQ(point.source.reg,expected);EXPECT_EQ(plan->thread(point.source),thread);
 }
}

TEST_F(ExecutionExportTest, SymbolicRegisterPlanRejectsCrossThreadCommunication) {
 SmallVector<uint64_t> source(17,0),destination(10,0);
 // Source: first 9 logical bits in registers, high bit in lane. Despite 1024
 // register slots, a given thread cannot supply target values with both bits.
 for(unsigned i=0;i<9;++i)source[i]=uint64_t{1}<<i;
 source[10]=uint64_t{1}<<9;
 for(unsigned i=0;i<10;++i)destination[i]=uint64_t{1}<<i;
 auto src=executionType(context,{32,32},1024,source),dst=executionType(context,{32,32},8,destination);
 ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
 EXPECT_TRUE(failed(test::planSymbolicRegisterRedistribution(src,dst,UnknownLoc::get(&context))));
}

TEST_F(ExecutionExportTest, LargeProvenMapExportsCompactWithoutHolderTable) {
 SmallVector<uint64_t> columns(17,0);for(unsigned i=0;i<10;++i)columns[i]=uint64_t{1}<<i;
 auto type=executionType(context,{32,32},1024,columns,true);
 std::string typeText;llvm::raw_string_ostream(typeText)<<type;
 auto module=parseSourceString<ModuleOp>("module attributes {frisk.target=\"sm_90a\"} { frisk.kernel @large() { frisk.parallel (%i)=(1), threads=128 { %x=tensor.empty() : "+typeText+" frisk.end } frisk.end } }",&context);
 ASSERT_TRUE(module);auto original=print(*module);ASSERT_TRUE(succeeded(run(*module)));EXPECT_EQ(print(*module),original);
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));
 auto *object=parsed->getAsObject();EXPECT_EQ(object->getInteger("schema_version"),2);
 unsigned compact=0;
 for(auto &value:*object->getArray("values"))if(auto *distribution=value.getAsObject()->getObject("distributed")) {
  ++compact;EXPECT_EQ(distribution->getArray("logical"),nullptr);EXPECT_EQ(distribution->getArray("first"),nullptr);
  auto *symbolic=distribution->getObject("symbolic");ASSERT_NE(symbolic,nullptr);
  EXPECT_EQ(symbolic->getString("version"),"gf2-canonical-v1");EXPECT_EQ(symbolic->getInteger("hardware_bits"),17);EXPECT_EQ(symbolic->getInteger("logical_bits"),10);
  EXPECT_EQ(symbolic->getArray("rows_hex")->size(),10u);
  for(unsigned bit=0;bit<10;++bit) {
   auto row=(*symbolic->getArray("rows_hex"))[bit].getAsString();ASSERT_TRUE(row);
   uint64_t mask=0;ASSERT_FALSE(row->getAsInteger(16,mask));EXPECT_EQ(mask,uint64_t{1}<<bit);
  }
 }
 EXPECT_EQ(compact,1u);EXPECT_LT((*bytes)->getBufferSize(),12000u);
}

TEST_F(ExecutionExportTest, UnsupportedDeadOperationHasTypedAtomicRejection) {
 auto m=selected();ASSERT_TRUE(m);Value first;
 m->walk([&](Operation *op){for(Value value:op->getResults())if(!first && isa<RankedTensorType>(value.getType()))first=value;});ASSERT_TRUE(first);
 OpBuilder b(first.getDefiningOp());b.setInsertionPointAfter(first.getDefiningOp());b.create<arith::AddFOp>(first.getLoc(),first,first);
 {std::error_code ec;llvm::raw_fd_ostream stream(path,ec);ASSERT_FALSE(bool(ec));stream<<"preserve";}
 std::string diagnostics;
 ScopedDiagnosticHandler capture(&context,[&](Diagnostic &diagnostic){llvm::raw_string_ostream stream(diagnostics);diagnostic.print(stream);return success();});
 EXPECT_TRUE(failed(run(*m)));
 EXPECT_NE(diagnostics.find("stage=export-preflight code=unsupported-operation"),std::string::npos);
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));EXPECT_EQ((*bytes)->getBuffer(),"preserve");
}

TEST_F(ExecutionExportTest, WarpShuffleExportsRecomputableNetworkAndExactCounts) {
 SmallVector<uint64_t> sourceColumns(10),destinationColumns;
 for(unsigned i=0;i<10;++i)sourceColumns[i]=uint64_t{1}<<i;
 destinationColumns=sourceColumns;std::swap(destinationColumns[0],destinationColumns[3]);
 auto source=executionType(context,{1024},8,sourceColumns);
 auto destination=executionType(context,{1024},8,destinationColumns,true);
 std::string sourceText;llvm::raw_string_ostream(sourceText)<<source;
 auto module=parseSourceString<ModuleOp>("module attributes {frisk.target=\"sm_90a\"} { frisk.kernel @shuffle() { frisk.parallel (%i)=(1), threads=128 { %x=tensor.empty() : "+sourceText+" frisk.end } frisk.end } }",&context);
 ASSERT_TRUE(module);Value first;module->walk([&](tensor::EmptyOp op){first=op.getResult();});ASSERT_TRUE(first);
 OpBuilder builder(first.getDefiningOp());builder.setInsertionPointAfter(first.getDefiningOp());
 OperationState state(first.getLoc(),"frisk.convert_layout");state.addOperands(first);state.addTypes(destination);builder.create(state);
 ASSERT_TRUE(succeeded(run(*module)));
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));
 unsigned found=0;
 for(auto &entry:*parsed->getAsObject()->getArray("operations")) {
  auto *object=entry.getAsObject();if(object->getString("name")!="frisk.convert_layout")continue;++found;
  auto *redistribution=object->getObject("redistribution");ASSERT_NE(redistribution,nullptr);
  auto *network=redistribution->getObject("shuffle_network");ASSERT_NE(network,nullptr);
  EXPECT_EQ(network->getString("contract"),"gf2-shuffle-network-v1");
  EXPECT_EQ(network->getString("kind"),"bit-permutation-ship");
  EXPECT_EQ(network->getInteger("verified_holders"),1024u);
  auto *counts=network->getObject("counts");ASSERT_NE(counts,nullptr);
  EXPECT_EQ(counts->getInteger("shfl"),4u);EXPECT_EQ(counts->getInteger("selects"),12u);
  EXPECT_EQ(counts->getInteger("map_instructions"),2u);
  EXPECT_EQ(counts->getInteger("shuffle_warp_issues"),16u);
  EXPECT_EQ(counts->getInteger("address_alu_warp_issues"),56u);
  ASSERT_NE(network->getArray("steps"),nullptr);EXPECT_FALSE(network->getArray("steps")->empty());
 }
 EXPECT_EQ(found,1u);
}


// No tensor/symbolic payload: compact storage itself must upgrade the wire.
OwningOpRef<ModuleOp> storageOnlyModule(MLIRContext &context, StringRef byteExpr) {
 std::string text=(Twine("#s=#frisk.storage<map=#frisk.affine_layout<inputs=[\"row\",\"column\"],input_extents=[2,4],outputs=[\"byte_offset\",\"bit_offset\"],output_extents=[32,8],map=affine_map<(i,j)->(")+byteExpr+",0)>>,memory_space=#frisk<memory_space Global>,alignment=1,vector_granularity=1>\n"
   "module attributes {frisk.target=\"sm_90a\"} { \"frisk.kernel\"() <{sym_name=\"storage_only\",function_type=(memref<2x4xf32,1>)->()}> ({ ^bb0(%r:memref<2x4xf32,1>): "
   "frisk.storage_contract %r {layout=#s} : memref<2x4xf32,1> "
   "frisk.parallel (%i)=(1),threads=128 { %v=frisk.layout_view %r {layout=#s} : memref<2x4xf32,1> -> memref<2x4xf32,1> frisk.end } frisk.end }) : ()->() }").str();
 return parseSourceString<ModuleOp>(text,&context);
}
TEST_F(ExecutionExportTest, CompactStorageAloneUsesSchemaTwoForRootAndIdentityView) {
 auto module=storageOnlyModule(context,"4*(4*i+j)");ASSERT_TRUE(module);
 auto before=print(*module);ASSERT_TRUE(succeeded(run(*module)));EXPECT_EQ(before,print(*module));
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));
 auto *object=parsed->getAsObject();EXPECT_EQ(object->getInteger("schema_version"),2);
 unsigned count=0;int64_t root=-1;
 for(auto &value:*object->getArray("values"))if(auto *storage=value.getAsObject()->getObject("storage")) {
  ++count;auto *compact=storage->getObject("addresses_compact");ASSERT_NE(compact,nullptr);
  EXPECT_EQ(storage->getArray("addresses"),nullptr);EXPECT_EQ(compact->size(),5u);
  EXPECT_EQ(compact->getString("contract"),"contiguous-identity-addresses-v1");
  EXPECT_EQ(compact->getInteger("element_bits"),32);EXPECT_EQ(compact->getInteger("base_bit"),0);
  auto *shape=compact->getArray("shape"),*strides=compact->getArray("strides_bits");ASSERT_NE(shape,nullptr);ASSERT_NE(strides,nullptr);ASSERT_EQ(shape->size(),2u);ASSERT_EQ(strides->size(),2u);
  EXPECT_EQ((*shape)[0].getAsInteger(),2);EXPECT_EQ((*shape)[1].getAsInteger(),4);
  EXPECT_EQ((*strides)[0].getAsInteger(),128);EXPECT_EQ((*strides)[1].getAsInteger(),32);
  ASSERT_TRUE(storage->getInteger("root"));if(root<0)root=*storage->getInteger("root");else EXPECT_EQ(storage->getInteger("root"),root);
 }
 EXPECT_EQ(count,2u);
}
TEST_F(ExecutionExportTest, SameShapeColumnMajorStorageRetainsDenseWire) {
 auto module=storageOnlyModule(context,"4*(i+2*j)");ASSERT_TRUE(module);ASSERT_TRUE(succeeded(run(*module)));
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));
 EXPECT_EQ(parsed->getAsObject()->getInteger("schema_version"),1);unsigned count=0;
 for(auto &value:*parsed->getAsObject()->getArray("values"))if(auto *storage=value.getAsObject()->getObject("storage")) {
  ++count;EXPECT_EQ(storage->getObject("addresses_compact"),nullptr);auto *addresses=storage->getArray("addresses");ASSERT_NE(addresses,nullptr);ASSERT_EQ(addresses->size(),8u);
  // Existing ordering is by bit address; compare every coordinate and interval.
  for(unsigned k=0;k<8;++k) {auto *entry=(*addresses)[k].getAsObject();ASSERT_NE(entry,nullptr);
   EXPECT_EQ(entry->getInteger("begin_bit"),k*32);EXPECT_EQ(entry->getInteger("end_bit"),(k+1)*32);
   auto *coord=entry->getArray("view");ASSERT_NE(coord,nullptr);EXPECT_EQ((*coord)[0].getAsInteger(),k%2);EXPECT_EQ((*coord)[1].getAsInteger(),k/2);
  }
 }
 EXPECT_EQ(count,2u);
}
TEST_F(ExecutionExportTest, AttentionKeepsHalfAndSharedDenseWithCompactFloatRoots) {
 auto module=selected();ASSERT_TRUE(module);ASSERT_TRUE(succeeded(run(*module)));
 auto bytes=llvm::MemoryBuffer::getFile(path);ASSERT_TRUE(bool(bytes));auto parsed=llvm::json::parse((*bytes)->getBuffer());ASSERT_TRUE(bool(parsed));unsigned compact=0,dense=0;
 for(auto &value:*parsed->getAsObject()->getArray("values"))if(auto *storage=value.getAsObject()->getObject("storage")) {
  if(storage->getObject("addresses_compact")){++compact;EXPECT_EQ(storage->getArray("addresses"),nullptr);}
  else {++dense;EXPECT_NE(storage->getArray("addresses"),nullptr);}
 }
 EXPECT_GE(compact,2u);EXPECT_GE(dense,6u);
}

}
