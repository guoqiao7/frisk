#ifndef FRISK_TRANSFORMS_PASSES_H
#define FRISK_TRANSFORMS_PASSES_H
#include "Dialect/Frisk/Analysis/LayoutBoundaryFrequency.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::frisk {
struct SolverOptions;
struct SM90CostProfile;
struct LayoutStatisticsReport;
class LayoutTrace;
class LayoutProfile;
enum class LayoutStatisticsScope { Standalone, Pipeline };

#define GEN_PASS_DECL
#include "Dialect/Frisk/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createFriskInferLayoutsPass();
std::unique_ptr<Pass> createFriskInferLayoutsPass(const SolverOptions &options,
                                              bool authorizeInternalOptimization = false,
                                              LayoutStatisticsReport *statistics = nullptr,
                                              LayoutStatisticsScope scope = LayoutStatisticsScope::Standalone,
                                              LayoutTrace *trace = nullptr,
                                              LayoutProfile *profile = nullptr,
                                              LayoutBoundaryFrequencyMode mode = LayoutBoundaryFrequencyMode::Legacy,
                                              const SM90CostProfile *calibratedProfile = nullptr);
std::unique_ptr<Pass> createNormalizeLayoutIRPass();
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass();
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass(LayoutBoundaryFrequencyMode mode,
    const SM90CostProfile *calibratedProfile = nullptr);
std::unique_ptr<Pass> createTestLowerLayoutConversionsPass();
std::unique_ptr<Pass> createExportExecutionPass();

#define GEN_PASS_REGISTRATION
#include "Dialect/Frisk/Transforms/Passes.h.inc"

} // namespace mlir::frisk

#endif // FRISK_TRANSFORMS_PASSES_H
