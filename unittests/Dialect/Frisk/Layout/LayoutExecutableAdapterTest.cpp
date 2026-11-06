#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
class LayoutExecutableAdapterTest : public testing::Test {
protected:
  MLIRContext context;
  std::string diagnostics;
  std::unique_ptr<ScopedDiagnosticHandler> handler;
  LayoutExecutableAdapterTest() {
    context.loadDialect<FriskDialect, gpu::GPUDialect, arith::ArithDialect>();
    handler = std::make_unique<ScopedDiagnosticHandler>(&context, [&](Diagnostic &d) {
      llvm::raw_string_ostream os(diagnostics); d.print(os); return success();
    });
  }
  OwningOpRef<ModuleOp> module(StringRef memref = "memref<32xf32,1>", StringRef extra = "") {
    std::string ir = R"mlir(
#A = #frisk.distributed<map=#frisk.bit_linear<inputs=["lane"],input_bits=[5],
outputs=["m"],output_bits=[5],matrix=dense<[[1,0,0,0,0],[0,1,0,0,0],
[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]> : tensor<5x5xi1>>,
topology=[1,32,1,1,1],replication=1>
gpu.module @kernels {
gpu.func @copy(%input:MEMREF,%output:MEMREF) kernel attributes {known_block_size=array<i32:32,1,1>} {
%x = frisk.tile_load %input : MEMREF -> tensor<32xf32,#A>
frisk.tile_store %x,%output : tensor<32xf32,#A>,MEMREF
EXTRA
gpu.return
}}
)mlir";
    for (size_t pos; (pos = ir.find("MEMREF")) != std::string::npos;)
      ir.replace(pos, 6, memref.str());
    ir.replace(ir.find("EXTRA"), 5, extra.str());
    return parseSourceString<ModuleOp>(ir, &context);
  }
  LogicalResult run(ModuleOp module) {
    PassManager pm(&context);
    auto pass = createTestLowerLayoutConversionsPass();
    if (failed(pass->initializeOptions("executable", [](const Twine &) { return failure(); })))
      return failure();
    pm.addPass(std::move(pass));
    return pm.run(module);
  }
  std::string print(ModuleOp module) {
    std::string text; llvm::raw_string_ostream os(text); module.print(os); return text;
  }
};
TEST_F(LayoutExecutableAdapterTest, GenericAndIntegerGlobalHaveNoBridges) {
  for (auto type : {"memref<32xf32>", "memref<32xf32,1>"}) {
    auto input = module(type); ASSERT_TRUE(input) << diagnostics;
    ASSERT_TRUE(succeeded(run(*input))) << diagnostics;
    auto text = print(*input);
    EXPECT_EQ(text.find("unrealized_conversion_cast"), std::string::npos);
    EXPECT_EQ(text.find("tensor<"), std::string::npos);
    EXPECT_EQ(text.find("frisk.tile"), std::string::npos);
    EXPECT_NE(text.find("memref.load"), std::string::npos);
    EXPECT_NE(text.find("memref.store"), std::string::npos);
  }
}
TEST_F(LayoutExecutableAdapterTest, RejectsEveryNonGlobalMemorySpaceWithoutMutation) {
  for (auto type : {"memref<32xf32,3>", "memref<32xf32,#gpu.address_space<workgroup>>",
                    "memref<32xf32,#gpu.address_space<private>>", "memref<32xf32,\"unknown\">"}) {
    SCOPED_TRACE(type);
    auto input = module(type); ASSERT_TRUE(input) << diagnostics;
    auto before = print(*input);
    EXPECT_TRUE(failed(run(*input)));
    EXPECT_EQ(before, print(*input));
  }
}
TEST_F(LayoutExecutableAdapterTest, RejectsStridedStorageAndUnsupportedOperation) {
  auto strided = module("memref<32xf32,strided<[2]>,1>"); ASSERT_TRUE(strided) << diagnostics;
  EXPECT_TRUE(failed(run(*strided)));
  auto unsupported = module("memref<32xf32,1>", "%unused = arith.constant 0 : i32");
  ASSERT_TRUE(unsupported) << diagnostics;
  auto before = print(*unsupported);
  EXPECT_TRUE(failed(run(*unsupported)));
  EXPECT_EQ(before, print(*unsupported));
}
TEST_F(LayoutExecutableAdapterTest, RejectsTileLargerThanStorage) {
  // The operation verifier must reject this before the executable pass runs.
  EXPECT_FALSE(module("memref<16xf32,1>"));
  EXPECT_FALSE(diagnostics.empty());
}
} // namespace
