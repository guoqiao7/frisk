#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/JSON.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
llvm::cl::opt<std::string> chip("target", llvm::cl::init("sm_90a"));
llvm::cl::opt<int64_t> m("block-m", llvm::cl::init(128)),
    n("block-n", llvm::cl::init(128)), k("block-k", llvm::cl::init(64)),
    threads("threads", llvm::cl::init(128));
llvm::cl::opt<std::string> dtype("dtype", llvm::cl::init("fp16")),
    aSpace("a-space", llvm::cl::init("shared")),
    bSpace("b-space", llvm::cl::init("shared")),
    cSpace("c-space", llvm::cl::init("local"));
llvm::cl::opt<bool> transA("trans-a", llvm::cl::init(false)),
    transB("trans-b", llvm::cl::init(false)),
    ldmatrix("ldmatrix", llvm::cl::desc("Deprecated reporting flag, not an instruction guarantee"));
llvm::cl::opt<unsigned> laneVector("lane-vector", llvm::cl::init(1));
llvm::cl::list<int64_t> grid("grid", llvm::cl::CommaSeparated);
std::string print(Attribute attr) {
  std::string text;
  llvm::raw_string_ostream os(text);
  attr.print(os);
  return text;
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Frisk canonical SM90 layout inference\n");
  if ((dtype != "fp16" && dtype != "bf16") ||
      (aSpace != "shared" && aSpace != "local") || bSpace != "shared" || cSpace != "local" ||
      m <= 0 || n <= 0 || k <= 0 || llvm::any_of(grid, [](int64_t extent) { return extent != 1; })) {
    llvm::errs() << "supported: positive static tiles; fp16/bf16; A shared/local, B shared, C local; single-CTA grid only\n";
    return 1;
  }
  if (ldmatrix || laneVector != 1)
    llvm::errs() << "legacy bank-analysis flags are not instruction or vector guarantees; reporting canonical contracts\n";
  MLIRContext context;
  context.loadDialect<FriskDialect, arith::ArithDialect, func::FuncDialect, memref::MemRefDialect>();
  OpBuilder b(&context);
  auto loc = b.getUnknownLoc();
  OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
  (*module)->setAttr("frisk.target", b.getStringAttr(chip));
  b.setInsertionPointToEnd(module->getBody());
  auto resultType = RankedTensorType::get({m, n}, b.getF32Type());
  auto function = b.create<func::FuncOp>(loc, "layout_example", b.getFunctionType({}, {resultType}));
  b.setInsertionPointToStart(function.addEntryBlock());
  Type input = dtype == "fp16" ? Type(b.getF16Type()) : Type(b.getBF16Type());
  auto operand = [&](ArrayRef<int64_t> shape, bool shared) -> Value {
    if (!shared) {
      auto type = RankedTensorType::get(shape, input);
      return b.create<arith::ConstantOp>(loc, type, DenseElementsAttr::get(type, b.getFloatAttr(input, 1.0)));
    }
    auto type = MemRefType::get(shape, input, MemRefLayoutAttrInterface{}, b.getI64IntegerAttr(3));
    auto allocation = b.create<memref::AllocOp>(loc, type);
    allocation->setAttr("alignment", b.getI64IntegerAttr(1024));
    return b.create<LayoutViewOp>(loc, type, allocation, StorageLayoutAttr());
  };
  SmallVector<int64_t> aShape = transA ? SmallVector<int64_t>{k, m} : SmallVector<int64_t>{m, k};
  SmallVector<int64_t> bShape = transB ? SmallVector<int64_t>{n, k} : SmallVector<int64_t>{k, n};
  Value a = operand(aShape, aSpace == "shared"), bb = operand(bShape, true);
  auto zero = b.create<arith::ConstantOp>(loc, resultType, DenseElementsAttr::get(resultType, b.getF32FloatAttr(0)));
  OperationState state(loc, MmaOp::getOperationName());
  state.addOperands({a, bb, zero});
  state.addTypes(resultType);
  state.addAttribute("m", b.getI64IntegerAttr(m));
  state.addAttribute("n", b.getI64IntegerAttr(n));
  state.addAttribute("k", b.getI64IntegerAttr(k));
  state.addAttribute("trans_a", b.getBoolAttr(transA));
  state.addAttribute("trans_b", b.getBoolAttr(transB));
  state.addAttribute("frisk.execution_threads", b.getI64IntegerAttr(threads));
  auto mma = b.create(state);
  // Keep the mathematical result observable through canonicalization.
  b.create<func::ReturnOp>(loc, mma->getResults());
  if (failed(verify(*module))) return 1;
  PassManager pipeline(&context);
  buildFriskLayoutPipeline(pipeline);
  if (failed(pipeline.run(*module))) return 1;
  llvm::json::Object output;
  output["schema"] = "frisk-canonical-layout-v1";
  output["target"] = std::string(chip);
  output["threads"] = int64_t(threads);
  output["performance_claim"] = "static contracts only; no GPU benchmark";
  llvm::json::Array gridJson;
  for (int64_t extent : grid) gridJson.push_back(extent);
  output["requested_grid"] = std::move(gridJson);
  llvm::json::Array layouts;
  module->walk([&](LayoutViewOp view) {
    layouts.push_back(print(view.getLayoutAttr()));
  });
  module->walk([&](MmaOp mma) {
    output["mma_contract"] = print(mma->getAttr("frisk.mma_contract"));
    output["accumulator_encoding"] = print(cast<RankedTensorType>(mma.getResult().getType()).getEncoding());
    if (auto aType = dyn_cast<RankedTensorType>(mma.getA().getType()))
      output["a_encoding"] = print(aType.getEncoding());
  });
  output["storage_layouts"] = std::move(layouts);
  std::string ir;
  llvm::raw_string_ostream os(ir);
  module->print(os);
  output["ir"] = ir;
  llvm::json::OStream json(llvm::outs(), 2);
  json.value(llvm::json::Value(std::move(output)));
  llvm::outs() << "\n";
  return 0;
}
