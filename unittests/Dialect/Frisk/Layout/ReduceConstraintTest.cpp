#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
TEST(ReduceConstraintTest, RegistersPureMathematicalReduction) {
  MLIRContext context;
  context.loadDialect<FriskDialect, func::FuncDialect>();
  auto module = parseSourceString<ModuleOp>(R"(
    func.func @reduce(%x: tensor<4x4xf32>) -> tensor<4xf32> {
      %r = "frisk.reduce_tensor"(%x) {kind="sum",dim=1:i64} : (tensor<4x4xf32>) -> tensor<4xf32>
      return %r : tensor<4xf32>
    })", &context);
  ASSERT_TRUE(module);
  Operation &op = module->getBody()->front().getRegion(0).front().front();
  EXPECT_TRUE(isMemoryEffectFree(&op));
}

TEST(ReduceConstraintTest, VerifiesKindsDtypesShapesAndThreads) {
  MLIRContext context;
  context.loadDialect<FriskDialect,func::FuncDialect>();
  ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
  for (StringRef kind:{"sum","max","min","add","mul"})
    for (StringRef dtype:{"f16","bf16","f32","f64","i32"}) {
      std::string text="func.func @f(%x:tensor<4x4x"+dtype.str()+">) { %r=frisk.reduce_tensor %x {kind=\""+kind.str()+"\",dim=1:i64} : tensor<4x4x"+dtype.str()+"> -> tensor<4x"+dtype.str()+"> return }";
      auto module=parseSourceString<ModuleOp>(text,&context);
      EXPECT_EQ(bool(module),kind!="add" && kind!="mul" && dtype!="f64" && dtype!="i32");
    }
  for (StringRef source:{"4x4","4x3","4x1","4x?","4"})
    for (int64_t axis:{-1,0,1,2}) {
      std::string text="func.func @f(%x:tensor<"+source.str()+"xf32>) { %r=frisk.reduce_tensor %x {kind=\"sum\",dim="+std::to_string(axis)+":i64} : tensor<"+source.str()+"xf32> -> tensor<4xf32> return }";
      auto module=parseSourceString<ModuleOp>(text,&context);
      EXPECT_EQ(bool(module),source=="4x4" && (axis==0 || axis==1));
    }
}

TEST(ReduceConstraintTest, StrictCompleteSchemaRejectsEveryMissingAndUnknownField) {
  MLIRContext context;
  context.loadDialect<FriskDialect>(); Builder b(&context);
  NamedAttrList fields;
  fields.set("version",b.getI64IntegerAttr(1)); fields.set("axis",b.getI64IntegerAttr(1));
  fields.set("threads",b.getI64IntegerAttr(32));
  fields.set("source_shape",b.getDenseI64ArrayAttr({4,4}));
  fields.set("result_shape",b.getDenseI64ArrayAttr({4}));
  for (auto pair : {std::pair<const char *,const char *>{"target","sm_90"},
       {"kind","sum"},{"dtype","f32"},{"algorithm","canonical_fiber_tree_v1"},
       {"input_policy","first_owner"},{"output_policy","broadcast_complete"},{"scope","warp"}})
    fields.set(pair.first,b.getStringAttr(pair.second));
  ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
  auto check=[&](NamedAttrList f) {
    return succeeded(ReductionContractAttr::verify([&]{return emitError(b.getUnknownLoc());},f.getDictionary(&context)));
  };
  EXPECT_TRUE(check(fields));
  for (auto attr : fields) {
    auto missing=fields; missing.erase(attr.getName()); EXPECT_FALSE(check(missing));
    auto wrong=fields; wrong.set(attr.getName(),b.getUnitAttr()); EXPECT_FALSE(check(wrong));
  }
  auto extra=fields; extra.set("extra",b.getUnitAttr()); EXPECT_FALSE(check(extra));
  auto algorithm=fields; algorithm.set("algorithm",b.getStringAttr("other")); EXPECT_FALSE(check(algorithm));
  auto threads=fields; threads.set("threads",b.getI64IntegerAttr(16)); EXPECT_FALSE(check(threads));
}

TEST(ReduceConstraintTest, Legacy16ThreadExecutionIsExplicitlyUnsupported) {
  MLIRContext context; context.loadDialect<FriskDialect,func::FuncDialect>();
  ScopedDiagnosticHandler silence(&context,[](Diagnostic &){return success();});
  for (StringRef shape:{"16x8","4x8x16"}) {
    std::string result=shape=="16x8" ? "16":"4x8";
    std::string axis=shape=="16x8" ? "1":"2";
    std::string text="func.func @f(%x:tensor<"+shape.str()+"xf32>) { %r=frisk.reduce_tensor %x {kind=\"sum\",dim="+axis+":i64,frisk.execution_threads=16:i64} : tensor<"+shape.str()+"xf32> -> tensor<"+result+"xf32> return }";
    EXPECT_FALSE(parseSourceString<ModuleOp>(text,&context));
  }
}
