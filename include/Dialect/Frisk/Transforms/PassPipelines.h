#ifndef FRISK_TRANSFORMS_PASSPIPELINES_H
#define FRISK_TRANSFORMS_PASSPIPELINES_H
#include "Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "llvm/ADT/StringRef.h"
namespace mlir {
class OpPassManager;
namespace frisk {
struct SolverOptions;
struct LayoutStatisticsReport;
/// Append a module-atomic normalization/inference/optimization transaction.
void buildFriskLayoutPipeline(OpPassManager &pm);
/// irDiagnostics observes only the final verified staged IR, without scoring.
void buildFriskLayoutPipeline(OpPassManager &pm, const SolverOptions &options,
                             llvm::StringRef statsFile = {},
                             LayoutStatisticsReport *statistics = nullptr,
                             LayoutTraceOptions traceOptions = {},
                             llvm::StringRef profileFile = {},
                             bool irDiagnostics = false,
                             LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy,
                              llvm::StringRef costProfileFile = {});
/// Explicit host registration; no static initialization dependency.
void registerFriskLayoutPipelines();
} // namespace frisk
} // namespace mlir
#endif
