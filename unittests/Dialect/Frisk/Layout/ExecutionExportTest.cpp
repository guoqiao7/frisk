#include "Dialect/Frisk/IR/FriskDialect.h"
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

}
