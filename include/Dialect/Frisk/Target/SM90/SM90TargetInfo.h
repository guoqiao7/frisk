#ifndef FRISK_SM90_TARGET_INFO_H
#define FRISK_SM90_TARGET_INFO_H
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include <string>
namespace mlir::frisk {
struct SM90TargetInfo {
  std::string chip = "sm_90";
  std::string targetKey = "sm_90";
  bool supportsWGMMA = false;
  bool supportsTMA = true;
  bool supportsCpAsync = true;
  unsigned warpSize = 32;
  unsigned sectorBytes = 32;
  unsigned bankCount = 32;
  unsigned bankWidthBytes = 4;
};
/// Reconcile declarations in the nearest explicit target scope. Migration
/// frisk.target and NVVM chip declarations must agree. NVVM triples are bounded
/// to nvptx64-nvidia-cuda and nvptx-nvidia-cuda. The GPU device of the nearest
/// builtin ModuleOp's DLTI target_system_spec is reconciled independently of
/// nearer chip declarations and can supply chip/features and the
/// hardware facts warp_size/sector_bytes/bank_count/bank_width_bytes. Facts must
/// agree with SM90. A missing target uses ordinary SM90, never WGMMA capability.
FailureOr<SM90TargetInfo> getSM90TargetInfo(Operation *op);
} // namespace mlir::frisk
#endif
