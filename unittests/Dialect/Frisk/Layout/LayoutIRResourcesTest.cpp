#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
class LayoutIRResourcesTest : public testing::Test {
protected:
  MLIRContext context;
  LayoutIRResourcesTest() {
    context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect, scf::SCFDialect, linalg::LinalgDialect>();
  }
  std::string encoding(unsigned bits = 5) {
    std::string matrix;
    for (unsigned row = 0; row < bits; ++row) {
      if (row) matrix += ",";
      matrix += "[";
      for (unsigned col = 0; col < bits; ++col) {
        if (col) matrix += ",";
        matrix += row == col ? "1" : "0";
      }
      matrix += "]";
    }
    return "#e = #frisk.distributed<map=#frisk.bit_linear<inputs=[\"register\"],input_bits=[" +
        std::to_string(bits) + "],outputs=[\"d0\"],output_bits=[" + std::to_string(bits) +
        "],matrix=dense<[" + matrix + "]>:tensor<" + std::to_string(bits) + "x" +
        std::to_string(bits) + "xi1>>,topology=[" + std::to_string(uint64_t(1) << bits) +
        ",1,1,1,1],replication=1>\n";
  }
  OwningOpRef<ModuleOp> parse(StringRef text, unsigned bits = 5) {
    return parseSourceString<ModuleOp>(encoding(bits) + text.str(), &context);
  }
  static llvm::json::Object &scope(llvm::json::Object &result, unsigned i = 0) {
    return *(*result.getArray("scopes"))[i].getAsObject();
  }
};

TEST_F(LayoutIRResourcesTest, CanonicalBroadcastCountsOnlyOuterSSAAtInclusiveBoundary) {
  auto module = parse(R"mlir(
func.func private @use(tensor<32xf32,#e>)
func.func @broadcast(%init:tensor<32xf32,#e>) -> tensor<32xf32,#e> {
  %input=arith.constant dense<1.0>:tensor<32xf32,#e>
  %output=linalg.broadcast ins(%input:tensor<32xf32,#e>) outs(%init:tensor<32xf32,#e>) dimensions=[]
  func.call @use(%output):(tensor<32xf32,#e>)->()
  func.call @use(%output):(tensor<32xf32,#e>)->()
  return %output:tensor<32xf32,#e>
})mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  std::string before;
  llvm::raw_string_ostream(before) << *module;
  auto result = collectLayoutIRResources(*module);
  ASSERT_EQ(result.getString("status"), "available");
  auto &s = scope(result);
  EXPECT_EQ(s.getInteger("logical_live_carrier_slots_per_thread"), 96);
  EXPECT_EQ(s.getInteger("logical_live_carrier_bytes_per_thread"), 384);
  EXPECT_EQ(s.getInteger("peak_slots_at"), 2);
  auto &values = *s.getArray("tensor_values");
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[0].getAsObject()->getInteger("end"), 2);
  EXPECT_EQ(values[1].getAsObject()->getInteger("begin"), 1);
  EXPECT_EQ(values[1].getAsObject()->getInteger("end"), 2);
  EXPECT_EQ(values[2].getAsObject()->getInteger("begin"), 2);
  EXPECT_EQ(values[2].getAsObject()->getInteger("end"), 5);
  EXPECT_EQ(llvm::formatv("{0}", llvm::json::Value(collectLayoutIRResources(*module))).str(),
            llvm::formatv("{0}", llvm::json::Value(std::move(result))).str());
  std::string after;
  llvm::raw_string_ostream(after) << *module;
  EXPECT_EQ(before, after);
}

TEST_F(LayoutIRResourcesTest, CanonicalRankExpandingBroadcastIsAtomic) {
  auto expandedEncoding = encoding();
  auto outputs = expandedEncoding.find("outputs=[\"d0\"],output_bits=[5]");
  ASSERT_NE(outputs, std::string::npos);
  expandedEncoding.replace(outputs, std::string("outputs=[\"d0\"],output_bits=[5]").size(),
                           "outputs=[\"d0\",\"d1\"],output_bits=[5,0]");
  expandedEncoding.replace(0, 2, "#b");
  auto module = parse(expandedEncoding + R"mlir(
func.func @expand(%a:tensor<32xf32,#e>,%init:tensor<32x1xf32,#b>) -> tensor<32x1xf32,#b> {
  %out=linalg.broadcast ins(%a:tensor<32xf32,#e>) outs(%init:tensor<32x1xf32,#b>) dimensions=[1]
  return %out:tensor<32x1xf32,#b>
})mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto result = collectLayoutIRResources(*module);
  ASSERT_EQ(result.getString("status"), "available");
  EXPECT_EQ(scope(result).getArray("tensor_values")->size(), 3u);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 96);
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 1);
}

TEST_F(LayoutIRResourcesTest, CanonicalBroadcastCapturesInSingleCTAParallel) {
  auto module = parse(R"mlir(
"frisk.kernel"() <{sym_name="broadcast",function_type=(tensor<32xf32,#e>,tensor<32xf32,#e>)->()}> ({
^bb0(%a:tensor<32xf32,#e>,%init:tensor<32xf32,#e>):
  "frisk.parallel"() <{ranges=array<i64:1>,threads=32:i64}> ({
  ^bb0(%i:index):
    %out=linalg.broadcast ins(%a:tensor<32xf32,#e>) outs(%init:tensor<32xf32,#e>) dimensions=[]
    frisk.end
  }) : ()->()
  frisk.end
}) : ()->()
)mlir");
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  auto result = collectLayoutIRResources(*module);
  ASSERT_EQ(result.getString("status"), "available");
  EXPECT_EQ(scope(result).getString("scope"), "kernel@broadcast/parallel");
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 96);
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 1);
  ASSERT_EQ(scope(result).getArray("tensor_values")->size(), 3u);
  EXPECT_TRUE((*scope(result).getArray("tensor_values"))[0].getAsObject()->getBoolean("live_in").value());
}

TEST_F(LayoutIRResourcesTest, MutatedBroadcastRegionsRemainUnavailableQuietAndPure) {
  for (unsigned variant = 0; variant < 8; ++variant) {
    auto module = parse(R"mlir(
func.func @broadcast(%a:tensor<32xf32,#e>,%init:tensor<32xf32,#e>) {
  %out=linalg.broadcast ins(%a:tensor<32xf32,#e>) outs(%init:tensor<32xf32,#e>) dimensions=[]
  return
})mlir");
    ASSERT_TRUE(module);
    ASSERT_TRUE(succeeded(verify(*module)));
    linalg::BroadcastOp broadcast;
    module->walk([&](linalg::BroadcastOp op) { broadcast = op; });
    if (variant == 7) {
      OperationState state(broadcast.getLoc(), linalg::BroadcastOp::getOperationName());
      state.addOperands(broadcast->getOperands());
      state.addTypes(broadcast->getResultTypes());
      state.addAttributes(broadcast->getAttrs());
      auto *malformed = Operation::create(state);
      OpBuilder insertBuilder(broadcast);
      insertBuilder.insert(malformed);
      broadcast->replaceAllUsesWith(malformed);
      broadcast.erase();
    } else {
      auto &region = broadcast->getRegion(0);
      auto &block = region.front();
      OpBuilder builder(&context);
      if (variant == 0) block.back().setOperand(0, block.getArgument(1));
      if (variant == 1) {
        builder.setInsertionPoint(&block.back());
        builder.create<arith::AddFOp>(broadcast.getLoc(), block.getArgument(0), block.getArgument(0));
      }
      if (variant == 2) region.push_back(new Block());
      if (variant == 3) block.addArgument(builder.getF32Type(), broadcast.getLoc());
      if (variant == 4) block.back().erase();
      if (variant == 5) region.getBlocks().clear();
      if (variant == 6) broadcast->setAttr("dimensions", builder.getDenseI64ArrayAttr({0}));
    }
    std::string before;
    llvm::raw_string_ostream beforeStream(before);
    module->print(beforeStream, OpPrintingFlags().printGenericOpForm());
    unsigned emitted = 0;
    ScopedDiagnosticHandler capture(&context, [&](Diagnostic &) { ++emitted; return success(); });
    auto result = collectLayoutIRResources(*module);
    ASSERT_EQ(result.getString("status"), "unavailable") << variant;
    EXPECT_EQ(emitted, 0u);
    EXPECT_EQ(*scope(result).get("logical_live_carrier_slots_per_thread"), llvm::json::Value(nullptr));
    EXPECT_EQ(scope(result).getArray("unavailable_operations")->front().getAsObject()->getString("reason"),
              "unsupported-control-flow");
    std::string after;
    llvm::raw_string_ostream afterStream(after);
    module->print(afterStream, OpPrintingFlags().printGenericOpForm());
    EXPECT_EQ(before, after);
  }
}

TEST_F(LayoutIRResourcesTest, DisjointAndOverlappingActualSSAHave32And64Slots) {
  for (bool overlap : {false, true}) {
    auto module = parse("func.func private @use(tensor<32xf32,#e>)\n"
      "func.func @sample() {\n"
      "%a = arith.constant dense<0.0> : tensor<32xf32,#e>\n" +
      std::string(overlap ? "" : "func.call @use(%a) : (tensor<32xf32,#e>)->()\n") +
      "%b = arith.constant dense<0.0> : tensor<32xf32,#e>\n" +
      std::string(overlap ? "func.call @use(%a) : (tensor<32xf32,#e>)->()\n" : "") +
      "func.call @use(%b) : (tensor<32xf32,#e>)->()\nreturn\n}");
    ASSERT_TRUE(module);
    auto result = collectLayoutIRResources(*module);
    EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), overlap ? 64 : 32);
    EXPECT_EQ(scope(result).getInteger("logical_live_carrier_bytes_per_thread"), overlap ? 256 : 128);
  }
}

TEST_F(LayoutIRResourcesTest, ConversionSharesOneSSAAndReturnDoesNotDuplicate) {
  auto module = parse(R"mlir(
func.func private @use(tensor<32xf32,#e>)
func.func @sample(%a: tensor<32xf32,#e>) -> tensor<32xf32,#e> {
  %b = frisk.convert_layout %a : tensor<32xf32,#e> -> tensor<32xf32,#e>
  func.call @use(%b) : (tensor<32xf32,#e>)->()
  func.call @use(%b) : (tensor<32xf32,#e>)->()
  return %b : tensor<32xf32,#e>
}
func.func @identity(%a: tensor<32xf32,#e>) -> tensor<32xf32,#e> {
  return %a : tensor<32xf32,#e>
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 64);
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 1);
  EXPECT_EQ(scope(result).getArray("tensor_values")->size(), 2u);
  EXPECT_EQ(scope(result, 1).getInteger("logical_live_carrier_slots_per_thread"), 32);
  EXPECT_EQ(scope(result, 1).getInteger("peak_slots_at"), 0);
  auto &values = *scope(result).getArray("tensor_values");
  EXPECT_EQ(values[0].getAsObject()->getInteger("begin"), 0);
  EXPECT_EQ(values[0].getAsObject()->getBoolean("live_in"), true);
  EXPECT_EQ(values[1].getAsObject()->getInteger("end"), 4);
}

TEST_F(LayoutIRResourcesTest, MixedDtypesAndUnusedResultsCountAtDefinition) {
  auto module = parse(R"mlir(
func.func @sample(%a: tensor<32xf32,#e>, %unused: tensor<32xf32,#e>) {
  %b = arith.truncf %a : tensor<32xf32,#e> to tensor<32xf16,#e>
  %c = arith.constant dense<true> : tensor<32xi1,#e>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 64);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_bytes_per_thread"), 192);
  auto &values = *scope(result).getArray("tensor_values");
  ASSERT_EQ(values.size(), 3u);
  EXPECT_EQ(values[1].getAsObject()->getInteger("bytes_per_thread"), 64);
  EXPECT_EQ(values[2].getAsObject()->getInteger("bytes_per_thread"), 32);
  EXPECT_EQ(values[2].getAsObject()->getInteger("begin"), 2);
  EXPECT_EQ(values[2].getAsObject()->getInteger("end"), 2);
}

TEST_F(LayoutIRResourcesTest, SharedAliasDedupAndUnusedDeclaration) {
  auto module = parse(R"mlir(
func.func @shared() {
  %a = memref.alloc() : memref<1024xf32,3>
  %unused = memref.alloc() : memref<1024xf32,3>
  %v = frisk.layout_view %a : memref<1024xf32,3> -> memref<1024xf32,3>
  %w = memref.cast %v : memref<1024xf32,3> to memref<?xf32,3>
  %x = memref.cast %w : memref<?xf32,3> to memref<1024xf32,3>
  %c = arith.constant 0 : index
  %one = memref.load %v[%c] : memref<1024xf32,3>
  %two = memref.load %x[%c] : memref<1024xf32,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "available");
  EXPECT_EQ(scope(result).getInteger("declared_shared_root_bytes"), 8192);
  EXPECT_EQ(scope(result).getInteger("logical_live_shared_root_bytes"), 4096);
  auto &roots = *scope(result).getArray("shared_roots");
  ASSERT_EQ(roots.size(), 2u);
  EXPECT_EQ(roots[0].getAsObject()->getInteger("begin"), 7);
  EXPECT_EQ(roots[0].getAsObject()->getInteger("end"), 8);
  EXPECT_EQ(*roots[1].getAsObject()->get("begin"), llvm::json::Value(nullptr));
}

TEST_F(LayoutIRResourcesTest, DisjointSharedRootsKeepDeclaredSum) {
  auto module = parse(R"mlir(
func.func @shared() {
  %a = memref.alloc() : memref<1024xf32,3>
  %b = memref.alloc() : memref<1024xf32,3>
  %c = arith.constant 0 : index
  %x = memref.load %a[%c] : memref<1024xf32,3>
  %y = memref.load %b[%c] : memref<1024xf32,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(scope(result).getInteger("declared_shared_root_bytes"), 8192);
  EXPECT_EQ(scope(result).getInteger("logical_live_shared_root_bytes"), 4096);
  EXPECT_EQ(scope(result).getInteger("peak_shared_at"), 4);
}

TEST_F(LayoutIRResourcesTest, ExternalRootOffsetStrideAndDroppedAxisUseWholeSpan) {
  auto module = parse(R"mlir(
func.func @shared(%r: memref<3x1x8xi32,strided<[8,8,1],offset:3>,3>) {
  %s = memref.subview %r[1,0,1] [1,1,4] [1,1,2] :
    memref<3x1x8xi32,strided<[8,8,1],offset:3>,3> to
    memref<1x4xi32,strided<[8,2],offset:12>,3>
  %v = frisk.layout_view %s : memref<1x4xi32,strided<[8,2],offset:12>,3>
    -> memref<1x4xi32,strided<[8,2],offset:12>,3>
  %c = arith.constant 0 : index
  %x = memref.load %v[%c,%c] : memref<1x4xi32,strided<[8,2],offset:12>,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "available");
  EXPECT_EQ(scope(result).getInteger("declared_shared_root_bytes"), 108);
  EXPECT_EQ(scope(result).getInteger("logical_live_shared_root_bytes"), 108);
  auto &root = *scope(result).getArray("shared_roots")->front().getAsObject();
  EXPECT_EQ(root.getString("origin"), "external");
}

TEST_F(LayoutIRResourcesTest, ExternalMetadataOnlyDeclaresCapacityWithoutPayload) {
  auto module = parse(R"mlir(
func.func @shared(%r:memref<4xi3,strided<[1],offset:1>,3>) {
  %v = frisk.layout_view %r : memref<4xi3,strided<[1],offset:1>,3>
    -> memref<4xi3,strided<[1],offset:1>,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(scope(result).getInteger("declared_shared_root_bytes"), 2);
  EXPECT_EQ(scope(result).getInteger("logical_live_shared_root_bytes"), 0);
  ASSERT_EQ(scope(result).getArray("shared_roots")->size(), 1u);
  EXPECT_EQ(*scope(result).getArray("shared_roots")->front().getAsObject()->get("begin"), llvm::json::Value(nullptr));
}

TEST_F(LayoutIRResourcesTest, UnknownExternalDeclarationCapacityIsUnavailableNotZero) {
  auto module = parse(R"mlir(
func.func @shared(%r:memref<?xf32,3>) {
  %c=arith.constant 0:index
  %d=memref.dim %r,%c:memref<?xf32,3>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "unavailable");
}

TEST_F(LayoutIRResourcesTest, ParallelUnsupportedRangesMultiplicityAndOutsideWorkAreExplicit) {
  for (unsigned variant = 0; variant < 3; ++variant) {
    std::string parallel = "\"frisk.parallel\"() <{ranges=array<i64:" +
      std::string(variant == 0 ? "2" : "1") + ">,threads=32:i64}> ({^bb0(%i:index): frisk.end}) : ()->()\n";
    auto module = parse("\"frisk.kernel\"() <{sym_name=\"bad\",function_type=(tensor<32xf32,#e>)->()}> ({\n"
      "^bb0(%a:tensor<32xf32,#e>):\n" + parallel + (variant == 1 ? parallel : "") +
      (variant == 2 ? "%b=arith.addf %a,%a:tensor<32xf32,#e>\n" : "") + "frisk.end}) : ()->()");
    ASSERT_TRUE(module);
    auto result = collectLayoutIRResources(*module);
    EXPECT_EQ(result.getString("status"), "unavailable");
    auto &reason = *scope(result).getArray("unavailable_operations")->front().getAsObject();
    EXPECT_EQ(reason.getString("reason"), variant == 2 ? "tensor-work-outside-parallel" : "unsupported-parallel-domain");
  }
}

TEST_F(LayoutIRResourcesTest, SlotAndBytePeaksCanOccurAtDifferentBoundaries) {
  auto module = parse(R"mlir(
func.func private @two(tensor<32xi1,#e>,tensor<32xi1,#e>)
func.func @different() {
  %a=arith.constant dense<true>:tensor<32xi1,#e>
  %b=arith.constant dense<true>:tensor<32xi1,#e>
  func.call @two(%a,%b):(tensor<32xi1,#e>,tensor<32xi1,#e>)->()
  %c=arith.constant dense<0.0>:tensor<32xf64,#e>
  return
})mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 2);
  EXPECT_EQ(scope(result).getInteger("peak_bytes_at"), 4);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 64);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_bytes_per_thread"), 256);
}

TEST_F(LayoutIRResourcesTest, ExplicitAsyncTokenEscapeIsUnavailableAndDiagnosticsStayLocal) {
  context.allowUnregisteredDialects();
  auto module = parse(R"mlir(
func.func @escape(%r:memref<4xf32,3>) -> !async.token {
  %t="test.start"(%r):(memref<4xf32,3>)->!async.token
  return %t:!async.token
})mlir");
  ASSERT_TRUE(module);
  unsigned emitted = 0;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &) { ++emitted; return success(); });
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(emitted, 0u);
  EXPECT_EQ(result.getString("status"), "unavailable");
}

TEST_F(LayoutIRResourcesTest, RegisteredGPUAsyncMemcpyTokenEscapeIsUnavailableAndQuiet) {
  context.loadDialect<gpu::GPUDialect>();
  auto module = parse(R"mlir(
func.func @escape(%src:memref<4xf32,3>, %dst:memref<4xf32,3>) -> !gpu.async.token {
  %token = gpu.memcpy async %dst, %src : memref<4xf32,3>, memref<4xf32,3>
  return %token : !gpu.async.token
})mlir");
  ASSERT_TRUE(module);
  unsigned emitted = 0;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &) { ++emitted; return success(); });
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(emitted, 0u);
  EXPECT_EQ(result.getString("status"), "unavailable");
  auto &s = scope(result);
  for (StringRef field : {"logical_live_carrier_slots_per_thread", "logical_live_carrier_bytes_per_thread",
       "logical_live_shared_root_bytes", "declared_shared_root_bytes", "peak_slots_at", "peak_bytes_at", "peak_shared_at"})
    EXPECT_EQ(*s.get(field), llvm::json::Value(nullptr));
  auto &failures = *s.getArray("unavailable_operations");
  ASSERT_FALSE(failures.empty());
  EXPECT_EQ(failures.front().getAsObject()->getString("reason"), "unsupported-shared-lifetime");
  EXPECT_EQ(failures.front().getAsObject()->getString("opcode"), "gpu.memcpy");
}

TEST_F(LayoutIRResourcesTest, RegisteredGPUSharedDeallocationIsUnavailableAndQuiet) {
  context.loadDialect<gpu::GPUDialect>();
  auto module = parse(R"mlir(
func.func @free(%r:memref<4xf32,3>) {
  gpu.dealloc %r : memref<4xf32,3>
  return
})mlir");
  ASSERT_TRUE(module);
  unsigned emitted = 0;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &) { ++emitted; return success(); });
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(emitted, 0u);
  EXPECT_EQ(result.getString("status"), "unavailable");
  auto &s = scope(result);
  for (StringRef field : {"logical_live_carrier_slots_per_thread", "logical_live_carrier_bytes_per_thread",
       "logical_live_shared_root_bytes", "declared_shared_root_bytes", "peak_slots_at", "peak_bytes_at", "peak_shared_at"})
    EXPECT_EQ(*s.get(field), llvm::json::Value(nullptr));
  auto &failures = *s.getArray("unavailable_operations");
  ASSERT_FALSE(failures.empty());
  EXPECT_EQ(failures.front().getAsObject()->getString("reason"), "unsupported-shared-lifetime");
  EXPECT_EQ(failures.front().getAsObject()->getString("opcode"), "gpu.dealloc");
}

TEST_F(LayoutIRResourcesTest, CopyFillExecutionContractsDoNotCreateTensorSSA) {
  auto executionEncoding = encoding();
  auto topology = executionEncoding.find("topology=[32,1,1,1,1],replication=1");
  ASSERT_NE(topology, std::string::npos);
  executionEncoding.replace(topology, std::string("topology=[32,1,1,1,1],replication=1").size(),
                            "topology=[32,32,1,1,1],replication=32");
  auto module = parseSourceString<ModuleOp>(executionEncoding + R"mlir(
func.func @metadata(%src:memref<32xf32,1>, %dst:memref<32xf32,3>) {
  frisk.fill %dst {value=0.0:f32,frisk.execution_layout=#e,
    frisk.execution_threads=32:i64,frisk.writer_policy="first_owner",frisk.vector_bytes=1:i64} : memref<32xf32,3>
  frisk.copy %src[], %dst[] {srcExtents=array<i64:32>,dstExtents=array<i64:32>,
    frisk.execution_layout=#e,frisk.execution_threads=32:i64,
    frisk.writer_policy="first_owner",frisk.vector_bytes=1:i64} : memref<32xf32,1>, memref<32xf32,3>
  return
})mlir", &context);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "available");
  auto &s = scope(result);
  EXPECT_EQ(s.getInteger("logical_live_carrier_slots_per_thread"), 0);
  EXPECT_EQ(s.getInteger("logical_live_carrier_bytes_per_thread"), 0);
  EXPECT_TRUE(s.getArray("tensor_values")->empty());
  EXPECT_EQ(s.getInteger("declared_shared_root_bytes"), 128);
  EXPECT_EQ(s.getInteger("logical_live_shared_root_bytes"), 128);
}

TEST_F(LayoutIRResourcesTest, AsyncAndUnknownUsersHoldSharedRootsToEnd) {
  for (bool unknown : {false, true}) {
    context.allowUnregisteredDialects();
    auto module = parse("func.func @shared(%a: memref<1024xf32,3>, %b: memref<1024xf32,3>) {\n" +
      std::string(unknown ? "\"test.read\"(%a) : (memref<1024xf32,3>)->()\n" :
        "memref.copy %a, %b {frisk.copy_contract={path=\"cp.async\"}} : memref<1024xf32,3> to memref<1024xf32,3>\n") +
      "%c = arith.constant 0 : index\n%x = memref.load %b[%c] : memref<1024xf32,3>\nreturn\n}");
    ASSERT_TRUE(module);
    auto result = collectLayoutIRResources(*module);
    EXPECT_EQ(result.getString("status"), "available");
    EXPECT_EQ(scope(result).getInteger("logical_live_shared_root_bytes"), 8192);
    auto &root = *scope(result).getArray("shared_roots")->front().getAsObject();
    EXPECT_EQ(root.getBoolean("held_to_scope_end"), true);
    EXPECT_EQ(root.getInteger("end"), 4);
  }
}

TEST_F(LayoutIRResourcesTest, ParallelCaptureIsLiveInAndIncludesOuterAllocations) {
  auto module = parse(R"mlir(
"frisk.kernel"() <{sym_name="capture",function_type=(tensor<32xf32,#e>)->()}> ({
^bb0(%a:tensor<32xf32,#e>):
  %r = memref.alloc() : memref<32xf32,3>
  %unused = memref.alloc() : memref<1024xf32,3>
  "frisk.parallel"() <{ranges=array<i64:1>,threads=32:i64}> ({
  ^bb0(%i:index):
    frisk.tile_store %a, %r : tensor<32xf32,#e>, memref<32xf32,3>
    frisk.end
  }) : ()->()
  frisk.end
}) : ()->()
)mlir");
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "available");
  EXPECT_EQ(scope(result).getString("scope"), "kernel@capture/parallel");
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 32);
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 0);
  EXPECT_EQ(scope(result).getInteger("declared_shared_root_bytes"), 4224);
  EXPECT_EQ(scope(result).getInteger("logical_live_shared_root_bytes"), 128);
  EXPECT_EQ(scope(result).getArray("tensor_values")->front().getAsObject()->getBoolean("live_in"), true);
}

TEST_F(LayoutIRResourcesTest, UnsupportedDomainsAreUnavailableNotZeroOrSkipped) {
  for (StringRef body : {
      "func.func @bad(%x:tensor<32xf32>) { %a=arith.addf %x,%x:tensor<32xf32> return }",
      "func.func @bad(%c:i1) { scf.if %c {} return }",
      "func.func @bad() { %l=arith.constant 0:index %h=arith.constant 2:index %s=arith.constant 1:index scf.for %i=%l to %h step %s {} return }",
      "func.func @bad(%c:i1) { scf.while : ()->() { scf.condition(%c) } do { scf.yield } return }",
      "func.func @bad(%c:i1) { cf.cond_br %c,^a,^b ^a: return ^b: return }",
      "func.func @bad(%r:memref<4xf32,3>) { memref.dealloc %r:memref<4xf32,3> return }",
      "func.func @bad(%r:memref<?xf32,3>) { %c=arith.constant 0:index %x=memref.load %r[%c]:memref<?xf32,3> return }",
      "func.func @bad(%r:memref<4xf32,3>) -> memref<4xf32,3> { return %r:memref<4xf32,3> }"}) {
    // cf is intentionally built as generic registered-on-demand by parser.
    context.allowUnregisteredDialects();
    auto module = parseSourceString<ModuleOp>(encoding() + body.str(), ParserConfig(&context, false));
    ASSERT_TRUE(module) << body.str();
    auto result = collectLayoutIRResources(*module);
    EXPECT_EQ(result.getString("status"), "unavailable") << body.str();
    auto &s = scope(result);
    EXPECT_EQ(s.getString("status"), "unavailable");
    for (StringRef field : {"logical_live_carrier_slots_per_thread", "logical_live_carrier_bytes_per_thread",
         "logical_live_shared_root_bytes", "declared_shared_root_bytes", "peak_slots_at", "peak_bytes_at", "peak_shared_at"})
      EXPECT_EQ(*s.get(field), llvm::json::Value(nullptr));
    ASSERT_FALSE(s.getArray("unavailable_operations")->empty());
  }
}

TEST_F(LayoutIRResourcesTest, SaturationIsCheckedPerValueAndPerBoundaryWithoutEnumeration) {
  auto module = parse(R"mlir(
func.func private @use(tensor<4611686018427387904xi1,#e>,tensor<4611686018427387904xi1,#e>,tensor<4611686018427387904xi1,#e>,tensor<4611686018427387904xi1,#e>)
func.func @huge(%a:tensor<4611686018427387904xi1,#e>,%b:tensor<4611686018427387904xi1,#e>,%c:tensor<4611686018427387904xi1,#e>,%d:tensor<4611686018427387904xi1,#e>) {
  func.call @use(%a,%b,%c,%d) : (tensor<4611686018427387904xi1,#e>,tensor<4611686018427387904xi1,#e>,tensor<4611686018427387904xi1,#e>,tensor<4611686018427387904xi1,#e>)->()
  return
}
func.func @wide(%a:tensor<4611686018427387904xf64,#e>) -> tensor<4611686018427387904xf64,#e> {
  return %a:tensor<4611686018427387904xf64,#e>
})mlir", 62);
  ASSERT_TRUE(module);
  auto result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "available");
  EXPECT_EQ(*scope(result).get("logical_live_carrier_slots_per_thread")->getAsUINT64(), UINT64_MAX);
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 0);
  EXPECT_EQ(scope(result).getBoolean("saturated"), true);
  EXPECT_EQ(*scope(result, 1).get("logical_live_carrier_bytes_per_thread")->getAsUINT64(), UINT64_MAX);
  EXPECT_EQ(scope(result, 1).getBoolean("saturated"), true);
}

TEST_F(LayoutIRResourcesTest, PureStableEmptyAndPartialScopes) {
  auto empty = parse("module {}");
  ASSERT_TRUE(empty);
  auto result = collectLayoutIRResources(*empty);
  EXPECT_EQ(result.getString("status"), "available");
  EXPECT_TRUE(result.getArray("scopes")->empty());
  auto module = parse("func.func @empty() { return }\nfunc.func @bad(%x:tensor<32xf32>) { %y=arith.addf %x,%x:tensor<32xf32> return }");
  ASSERT_TRUE(module);
  std::string before;
  llvm::raw_string_ostream(before) << *module;
  result = collectLayoutIRResources(*module);
  EXPECT_EQ(result.getString("status"), "partial");
  EXPECT_EQ(scope(result).getInteger("peak_slots_at"), 0);
  EXPECT_EQ(scope(result).getInteger("logical_live_carrier_slots_per_thread"), 0);
  EXPECT_EQ(llvm::formatv("{0}", llvm::json::Value(collectLayoutIRResources(*module))).str(),
            llvm::formatv("{0}", llvm::json::Value(std::move(result))).str());
  std::string after;
  llvm::raw_string_ostream(after) << *module;
  EXPECT_EQ(before, after);
}

TEST(LayoutIRDiagnosticsLifetimeTest, ResourceJSONOwnsStringsAfterContextDestruction) {
  llvm::json::Object result;
  {
    MLIRContext context;
    context.loadDialect<func::FuncDialect, arith::ArithDialect>();
    auto module = parseSourceString<ModuleOp>("func.func @detached(%a:tensor<32xf32>) { %b=arith.addf %a,%a:tensor<32xf32> return }", &context);
    ASSERT_TRUE(module);
    result = collectLayoutIRResources(*module);
    auto &s = *result.getArray("scopes")->front().getAsObject();
    auto &entry = *s.getArray("unavailable_operations")->front().getAsObject();
    EXPECT_EQ(entry.getString("opcode"), "arith.addf");
    StringRef opcode;
    module->walk([&](arith::AddFOp op) { opcode = op->getName().getStringRef(); });
    ASSERT_NE(entry.getString("opcode")->data(), opcode.data());
  }
  auto parsed = llvm::json::parse(llvm::formatv("{0}", llvm::json::Value(std::move(result))).str());
  ASSERT_TRUE(bool(parsed));
  EXPECT_EQ(parsed->getAsObject()->getString("status"), "unavailable");
}
} // namespace
