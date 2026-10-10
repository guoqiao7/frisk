// Fixed coordinate definitions, not planner-produced ownership tables.
// The emitted Frisk IR is compiled through the real layout pipeline + adapter.
#include <iostream>
#include <string>
#include <set>
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"

static int promote(const char *input, const char *output) {
  using namespace mlir;
  MLIRContext context;
  context.loadDialect<frisk::FriskDialect, func::FuncDialect, gpu::GPUDialect>();
  auto module = parseSourceFile<ModuleOp>(input, &context);
  if (!module || failed(verify(*module))) return 1;
  auto functions = module->getOps<func::FuncOp>();
  if (!llvm::hasSingleElement(functions)) { llvm::errs() << "promotion requires exactly one function\n"; return 1; }
  auto function = *functions.begin();
  unsigned threads = 0;
  bool invalidTopology = false;
  function.walk([&](frisk::TileLoadOp load) {
    auto type = cast<RankedTensorType>(load.getResult().getType());
    auto encoding = dyn_cast_or_null<frisk::DistributedEncodingAttr>(type.getEncoding());
    if (!encoding) { invalidTopology = true; return; }
    auto topology = encoding.getTopology().asArrayRef();
    if (topology.size() != 5 || topology[0] != 1 || topology[1] != 32 ||
        (topology[2] != 1 && topology[2] != 2) || topology[3] != 1 || topology[4] != 1) {
      invalidTopology = true; return;
    }
    unsigned current = 32 * topology[2];
    if (threads && threads != current) invalidTopology = true;
    threads = current;
  });
  if (invalidTopology || !threads ||
      !llvm::hasSingleElement(function.getBody()) || function.getNumArguments() != 2 ||
      function.getNumResults() || !isa<func::ReturnOp>(function.getBody().front().getTerminator())) {
    llvm::errs() << "unsupported conversion kernel ABI or topology\n"; return 1;
  }
  for (Type type : function.getArgumentTypes()) {
    auto memref = dyn_cast<MemRefType>(type);
    auto space = memref ? dyn_cast_or_null<IntegerAttr>(memref.getMemorySpace()) : IntegerAttr{};
    if (!memref || memref.getRank() != 1 || !memref.hasStaticShape() || !memref.getLayout().isIdentity() ||
        !space || space.getInt() != 1) { llvm::errs() << "unsupported conversion kernel storage ABI\n"; return 1; }
  }
  if (function.getArgumentTypes()[0] != function.getArgumentTypes()[1]) return 1;
  // gpu.func arguments are deliberately not production storage roots. The
  // verified contract survives in explicit whole-root layout views consumed by
  // the executable adapter; never widen production alias/root recognition.
  SmallVector<frisk::StorageContractOp> contracts;
  function.walk([&](frisk::StorageContractOp contract) { contracts.push_back(contract); });
  for (auto contract : contracts) {
    bool represented = false;
    for (Operation *user : contract.getRoot().getUsers()) {
      if (user == contract.getOperation()) continue;
      auto view = dyn_cast<frisk::LayoutViewOp>(user);
      if (!view || view.getLayoutAttr() != contract.getLayout()) {
        llvm::errs() << "root contract cannot be represented by explicit ABI views\n";
        return 1;
      }
      represented = true;
    }
    if (!represented) { llvm::errs() << "unrepresented storage root contract\n"; return 1; }
  }
  for (auto contract : contracts) contract.erase();
  OpBuilder builder(module->getBodyRegion());
  auto gpuModule = builder.create<gpu::GPUModuleOp>(function.getLoc(), "kernels");
  builder.setInsertionPointToStart(gpuModule.getBody());
  auto kernel = builder.create<gpu::GPUFuncOp>(function.getLoc(), function.getName(), function.getFunctionType());
  for (auto attribute : function->getAttrs()) kernel->setAttr(attribute.getName(), attribute.getValue());
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(), builder.getUnitAttr());
  kernel->setAttr("known_block_size", builder.getDenseI32ArrayAttr({int32_t(threads), 1, 1}));
  kernel.getBody().takeBody(function.getBody());
  auto *terminator = kernel.getBody().front().getTerminator();
  builder.setInsertionPoint(terminator);
  builder.create<gpu::ReturnOp>(terminator->getLoc());
  terminator->erase();
  function.erase();
  (*module)->setAttr(gpu::GPUDialect::getContainerModuleAttrName(), builder.getUnitAttr());
  if (failed(verify(*module))) return 1;
  std::error_code error;
  llvm::ToolOutputFile file(output, error, llvm::sys::fs::OF_None);
  if (error) { llvm::errs() << error.message() << '\n'; return 1; }
  module->print(file.os());
  file.keep();
  return 0;
}

static std::string map(unsigned bits, unsigned swapA, unsigned swapB, bool permute) {
  std::string out = "#frisk.bit_linear<inputs=[\"lane\"";
  if (bits == 6) out += ",\"warp\"";
  out += "],input_bits=[5";
  if (bits == 6) out += ",1";
  out += "],outputs=[\"m\"],output_bits=[" + std::to_string(bits) + "],matrix=dense<[";
  for (unsigned row = 0; row < bits; ++row) {
    if (row) out += ',';
    out += '[';
    unsigned column = row;
    if (permute && row == swapA) column = swapB;
    if (permute && row == swapB) column = swapA;
    for (unsigned c = 0; c < bits; ++c) {
      if (c) out += ',';
      out += c == column ? '1' : '0';
    }
    out += ']';
  }
  return out + "]> : tensor<" + std::to_string(bits) + "x" + std::to_string(bits) + "xi1>>";
}
int main(int argc, char **argv) {
  if (argc == 4 && std::string(argv[1]) == "--promote") return promote(argv[2], argv[3]);
  if (argc != 2) { std::cerr << "usage: generate_conversion_cases CASE_ID\n"; return 1; }
  std::string id = argv[1];
  auto split = id.rfind('_');
  if (split == std::string::npos) return 1;
  std::string kind = id.substr(0, split), dtype = id.substr(split + 1);
  if (!std::set<std::string>{"identity", "shuffle32", "shared64", "replicated64"}.count(kind) ||
      !std::set<std::string>{"i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"}.count(dtype)) return 1;
  unsigned bits = kind == "shared64" ? 6 : 5;
  unsigned threads = kind == "shared64" || kind == "replicated64" ? 64 : 32;
  unsigned volume = 1u << bits;
  unsigned bytes = dtype == "i8" ? 1 : (dtype == "i16" || dtype == "f16" || dtype == "bf16") ? 2 : (dtype == "i64" || dtype == "f64") ? 8 : 4;
  std::string memref = "memref<" + std::to_string(volume) + "x" + dtype + ",1>";
  auto tensor = [&](char layout) { return "tensor<" + std::to_string(volume) + "x" + dtype + ",#" + layout + ">"; };
  std::cout << "#a=" << map(bits, 0, bits == 6 ? 5 : 1, false) << '\n'
    << "#b=" << map(bits, 0, bits == 6 ? 5 : 1, kind != "identity") << '\n'
    << "#A=#frisk.distributed<map=#a,topology=[1,32," << threads / 32 << ",1,1],replication=" << threads / volume << ">\n"
    << "#B=#frisk.distributed<map=#b,topology=[1,32," << threads / 32 << ",1,1],replication=" << threads / volume << ">\n"
    << "#S=#frisk.storage<map=#frisk.affine_layout<inputs=[\"m\"],input_extents=[" << volume
    << "],outputs=[\"byte_offset\",\"bit_offset\"],output_extents=[" << volume * bytes
    << ",8],map=affine_map<(d)->(" << bytes << "*d,0)>>,memory_space=#frisk<memory_space Global>,alignment=" << bytes << ",vector_granularity=" << bytes << ">\n"
    << "module attributes {frisk.target=\"sm_90\"} {\n"
    << "func.func @inferred_" << id << "(%input:" << memref << ",%output:" << memref << ") {\n"
    << "%src=frisk.layout_view %input {layout=#S} : " << memref << " -> " << memref << '\n'
    << "%dst=frisk.layout_view %output {layout=#S} : " << memref << " -> " << memref << '\n'
    << "%x=frisk.tile_load %src : " << memref << " -> " << tensor('A') << '\n'
    << "%y=frisk.convert_layout %x : " << tensor('A') << " -> " << tensor('B') << '\n'
    << "frisk.tile_store %y,%dst : " << tensor('B') << ',' << memref << "\nreturn\n}\n}\n";
}
