#ifndef FRISK_TRANSFORMS_PASSPIPELINES_H
#define FRISK_TRANSFORMS_PASSPIPELINES_H
#include "llvm/ADT/StringRef.h"
namespace mlir {
class OpPassManager;
namespace frisk {
struct SolverOptions;
struct LayoutStatisticsReport;
/// Append a module-atomic normalization/inference/optimization transaction.
void buildFriskLayoutPipeline(OpPassManager &pm);
void buildFriskLayoutPipeline(OpPassManager &pm, const SolverOptions &options,
                             llvm::StringRef statsFile = {},
                             LayoutStatisticsReport *statistics = nullptr);
/// Explicit host registration; no static initialization dependency.
void registerFriskLayoutPipelines();
} // namespace frisk
} // namespace mlir
#endif
