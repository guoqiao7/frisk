#include "Dialect/Frisk/Target/SM90/SM90InstructionTable.h"
using namespace mlir;
using namespace mlir::frisk;
namespace {
// PTX ISA 9.7.17.5: f16/bf16 m64nNk16 SS/RS, 128 threads per warpgroup.
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
#define ATOM(T, F, N, P) {T, F, 64, N, 16, 128, P}
#define SHAPES(T, F, P) ATOM(T,F,8,P), ATOM(T,F,16,P), ATOM(T,F,32,P), \
                       ATOM(T,F,64,P), ATOM(T,F,128,P), ATOM(T,F,256,P)
const SM90WGMMARule rules[] = {
  SHAPES("f16", "ss", "none"), SHAPES("bf16", "ss", "none"),
  SHAPES("f16", "rs", "f16x2-low-high"),
  SHAPES("bf16", "rs", "f16x2-low-high")};
#undef SHAPES
#undef ATOM
}
llvm::ArrayRef<SM90WGMMARule> mlir::frisk::getSM90WGMMARules() { return rules; }
bool mlir::frisk::isSM90WGMMAThreadCount(int64_t threads) {
  return threads == 128 || threads == 256 || threads == 512 || threads == 1024;
}
const SM90WGMMARule *mlir::frisk::lookupSM90WGMMA(
    Type input, Type accumulator, llvm::StringRef form, int64_t m, int64_t n,
    int64_t k, int64_t threads, llvm::StringRef majorA, llvm::StringRef majorB) {
  if (!input || !accumulator || !accumulator.isF32() ||
      !isSM90WGMMAThreadCount(threads) ||
      (majorA != "k" && majorA != "mn") ||
      (majorB != "k" && majorB != "mn") || (form == "rs" && majorA != "k"))
    return nullptr;
  llvm::StringRef type = input.isF16() ? "f16" : input.isBF16() ? "bf16" : "";
  for (const auto &rule : rules)
    if (rule.inputType == type && rule.form == form && rule.m == m &&
        rule.n == n && rule.k == k)
      return &rule;
  return nullptr;
}
