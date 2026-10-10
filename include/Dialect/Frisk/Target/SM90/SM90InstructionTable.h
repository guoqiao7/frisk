#ifndef FRISK_SM90_INSTRUCTION_TABLE_H
#define FRISK_SM90_INSTRUCTION_TABLE_H
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
namespace mlir::frisk {
struct SM90WGMMARule {
  llvm::StringRef inputType;
  llvm::StringRef form;
  int64_t m, n, k;
  int64_t warpgroupThreads;
  llvm::StringRef packing;
};
/// Deliberately bounded to the power-of-two N atoms supported by our canonical
/// maps; this is not a claim that other PTX WGMMA atoms are hardware-illegal.
llvm::ArrayRef<SM90WGMMARule> getSM90WGMMARules();
bool isSM90WGMMAThreadCount(int64_t threads);
const SM90WGMMARule *lookupSM90WGMMA(
    Type input, Type accumulator, llvm::StringRef form, int64_t m, int64_t n,
    int64_t k, int64_t threads, llvm::StringRef majorA, llvm::StringRef majorB);
} // namespace mlir::frisk
#endif
