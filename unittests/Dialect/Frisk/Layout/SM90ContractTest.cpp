#include "Dialect/Frisk/Target/SM90/SM90TargetInfo.h"
#include "Dialect/Frisk/Target/SM90/SM90InstructionTable.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/AsmParser/AsmParser.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
class SM90ContractTest : public testing::Test {
protected:
  MLIRContext ctx;
  Builder b{&ctx};
  SM90ContractTest() { ctx.loadDialect<NVVM::NVVMDialect, gpu::GPUDialect, DLTIDialect>(); }
};
TEST_F(SM90ContractTest, DefaultDoesNotEnableWGMMA) {
  auto module = ModuleOp::create(b.getUnknownLoc());
  auto target = getSM90TargetInfo(module);
  ASSERT_TRUE(succeeded(target));
  EXPECT_EQ(target->targetKey, "sm_90");
  EXPECT_FALSE(target->supportsWGMMA);
  EXPECT_TRUE(target->supportsTMA);
  EXPECT_EQ(target->warpSize, 32);
  module.erase();
}
TEST_F(SM90ContractTest, NvvmAndMigrationMustAgree) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {
    nvvm.target = #nvvm.target<chip = "sm_90a", features = "+ptx80">,
    frisk.target = "sm_90a"} {})", &ctx);
  ASSERT_TRUE(module);
  auto target = getSM90TargetInfo(*module);
  ASSERT_TRUE(succeeded(target));
  EXPECT_TRUE(target->supportsWGMMA);
  (*module)->setAttr("frisk.target", b.getStringAttr("sm_90"));
  EXPECT_TRUE(failed(getSM90TargetInfo(*module)));
}
TEST_F(SM90ContractTest, RejectMalformedAndUnsupportedTargets) {
  auto module = ModuleOp::create(b.getUnknownLoc());
  for (Attribute value : {Attribute(b.getI32IntegerAttr(90)), Attribute(b.getStringAttr("sm_80"))}) {
    module->setAttr("frisk.target", value);
    EXPECT_TRUE(failed(getSM90TargetInfo(module)));
  }
  module.erase();
}
TEST_F(SM90ContractTest, GpuModuleTargetArray) {
  auto module = parseSourceString<ModuleOp>(R"(module {
    gpu.module @kernel [#nvvm.target<chip = "sm_90a", features = "+ptx80">] {}
  })", &ctx);
  ASSERT_TRUE(module);
  auto gpuModule = *module->getOps<gpu::GPUModuleOp>().begin();
  auto target = getSM90TargetInfo(gpuModule);
  ASSERT_TRUE(succeeded(target));
  EXPECT_TRUE(target->supportsWGMMA);
}
TEST_F(SM90ContractTest, InstructionTableBoundaries) {
  for (Type input : {Type(b.getF16Type()), Type(b.getBF16Type())})
    for (int64_t n : {8, 16, 32, 64, 128, 256})
      for (StringRef form : {StringRef("ss"), StringRef("rs")})
        for (int64_t threads : {128, 256, 512, 1024})
          EXPECT_NE(lookupSM90WGMMA(input, b.getF32Type(), form, 64, n, 16,
                                   threads, "k", "mn"), nullptr);
  EXPECT_EQ(lookupSM90WGMMA(b.getF32Type(), b.getF32Type(), "ss", 64, 8, 16, 128, "k", "k"), nullptr);
  EXPECT_EQ(lookupSM90WGMMA(b.getF16Type(), b.getF32Type(), "ss", 64, 24, 16, 128, "k", "k"), nullptr);
  EXPECT_EQ(lookupSM90WGMMA(b.getF16Type(), b.getF32Type(), "ss", 64, 8, 16, 64, "k", "k"), nullptr);
  EXPECT_EQ(lookupSM90WGMMA(b.getF16Type(), b.getF32Type(), "rs", 64, 8, 16, 128, "mn", "k"), nullptr);
  EXPECT_EQ(lookupSM90WGMMA(b.getF16Type(), b.getF32Type(), "ss", 64, 8, 16, 128, "bad", "k"), nullptr);
}
TEST_F(SM90ContractTest, DltiFactsAndFeaturesAreValidated) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {
    dlti.target_system_spec = #dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"chip" = "sm_90a", "warp_size" = 32 : i64>>
  } {})", &ctx);
  ASSERT_TRUE(module);
  auto target = getSM90TargetInfo(*module);
  ASSERT_TRUE(succeeded(target));
  EXPECT_TRUE(target->supportsWGMMA);
  (*module)->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<chip = "sm_90a", features = "+ptx80,-sm_90a">)", &ctx));
  EXPECT_TRUE(failed(getSM90TargetInfo(*module)));
}
TEST_F(SM90ContractTest, RejectDltiHardwareMismatch) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {
    dlti.target_system_spec = #dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"chip" = "sm_90a", "warp_size" = 64 : i64>>
  } {})", &ctx);
  ASSERT_TRUE(module);
  EXPECT_TRUE(failed(getSM90TargetInfo(*module)));
}
TEST_F(SM90ContractTest, NearestScopeOverridesOuterChip) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {frisk.target = "sm_90"} {
    module @inner attributes {frisk.target = "sm_90a"} {}
  })", &ctx);
  ASSERT_TRUE(module);
  auto target = getSM90TargetInfo(*module->getOps<ModuleOp>().begin());
  ASSERT_TRUE(succeeded(target));
  EXPECT_TRUE(target->supportsWGMMA);
}
TEST_F(SM90ContractTest, DisabledPtxCannotInventCapability) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {
    nvvm.target = #nvvm.target<chip = "sm_90a", features = "-ptx80">
  } {})", &ctx);
  ASSERT_TRUE(module);
  EXPECT_TRUE(failed(getSM90TargetInfo(*module)));
  (*module)->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<chip = "sm_90a", features = "+ptx999">)", &ctx));
  EXPECT_TRUE(failed(getSM90TargetInfo(*module)));
}
TEST_F(SM90ContractTest, PtxAndChipBothLimitCapabilities) {
  auto module = ModuleOp::create(b.getUnknownLoc());
  module->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<chip = "sm_90", features = "+ptx80">)", &ctx));
  auto ordinary = getSM90TargetInfo(module);
  ASSERT_TRUE(succeeded(ordinary));
  EXPECT_FALSE(ordinary->supportsWGMMA);
  EXPECT_TRUE(ordinary->supportsTMA);
  module->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<chip = "sm_90a", features = "+ptx70">)", &ctx));
  auto oldIsa = getSM90TargetInfo(module);
  ASSERT_TRUE(succeeded(oldIsa));
  EXPECT_FALSE(oldIsa->supportsWGMMA);
  EXPECT_FALSE(oldIsa->supportsTMA);
  EXPECT_TRUE(oldIsa->supportsCpAsync);
  module->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<chip = "sm_90a", features = "+ptx60">)", &ctx));
  auto tooOld = getSM90TargetInfo(module);
  ASSERT_TRUE(succeeded(tooOld));
  EXPECT_FALSE(tooOld->supportsCpAsync);
  module->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<chip = "sm_90", features = "+ptx80,+sm_90a">)", &ctx));
  EXPECT_TRUE(failed(getSM90TargetInfo(module)));
  module.erase();
}
TEST_F(SM90ContractTest, EnclosingDltiIsIndependentOfNearestGpuTarget) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {
    dlti.target_system_spec = #dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"warp_size" = 32 : i64, "features" = "+ptx80", "chip" = "sm_90a">>
  } {
    gpu.module @kernel [#nvvm.target<chip = "sm_90a", features = "+ptx80">] {}
  })", &ctx);
  ASSERT_TRUE(module);
  auto gpuModule = *module->getOps<gpu::GPUModuleOp>().begin();
  auto compatible = getSM90TargetInfo(gpuModule);
  ASSERT_TRUE(succeeded(compatible));
  EXPECT_TRUE(compatible->supportsWGMMA);
  for (StringRef device : {
      R"(#dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"warp_size" = 64 : i64>>)",
      R"(#dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"features" = "+ptx70">>)",
      R"(#dlti.target_system_spec<"GPU" = #dlti.target_device_spec<"chip" = "sm_90">>)"}) {
    (*module)->setAttr("dlti.target_system_spec", parseAttribute(device, &ctx));
    EXPECT_TRUE(failed(getSM90TargetInfo(gpuModule)));
  }
}
TEST_F(SM90ContractTest, NvvmTripleMustBeARealSupportedArchitecture) {
  auto module = parseSourceString<ModuleOp>(R"(module attributes {
    nvvm.target = #nvvm.target<triple = "nvptx64-nvidia-cuda", chip = "sm_90a", features = "+ptx80">
  } {})", &ctx);
  ASSERT_TRUE(module);
  EXPECT_TRUE(succeeded(getSM90TargetInfo(*module)));
  (*module)->setAttr("nvvm.target", parseAttribute(R"(#nvvm.target<triple = "nvptxgarbage", chip = "sm_90a", features = "+ptx80">)", &ctx));
  EXPECT_TRUE(failed(getSM90TargetInfo(*module)));
}
} // namespace
