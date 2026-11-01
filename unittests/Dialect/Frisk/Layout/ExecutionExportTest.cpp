#include "Dialect/Frisk/IR/FriskDialect.h"
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

}
