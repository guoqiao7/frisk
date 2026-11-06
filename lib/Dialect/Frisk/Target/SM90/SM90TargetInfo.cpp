#include "Dialect/Frisk/Target/SM90/SM90TargetInfo.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include <optional>
using namespace mlir;
using namespace mlir::frisk;
namespace {
struct ScopeTarget {
  std::optional<std::string> chip;
  std::optional<unsigned> ptx;
  llvm::StringMap<bool> features;
  LogicalResult setChip(StringRef value) {
    if (value != "sm_90" && value != "sm_90a") return failure();
    if (chip && *chip != value) return failure();
    chip = value.str();
    return success();
  }
  LogicalResult setFeatures(StringRef value) {
    if (value.empty()) return success();
    SmallVector<StringRef> tokens;
    value.split(tokens, ',');
    for (StringRef token : tokens) {
      token = token.trim();
      if (token.size() < 2 || (token.front() != '+' && token.front() != '-')) return failure();
      bool enabled = token.front() == '+';
      StringRef name = token.drop_front();
      auto [it, inserted] = features.try_emplace(name, enabled);
      if (!inserted && it->second != enabled) return failure();
      if (name.starts_with("ptx")) {
        unsigned version;
        // PTX is a version selector, not a set of independent boolean
        // capabilities. Do not interpret a disabled selector as no declaration.
        // This is the supported feature list of the pinned NVPTX toolchain.
        static constexpr unsigned versions[] = {32, 40, 41, 42, 43, 50,
            60, 61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 76, 77, 78,
            80, 81, 82, 83, 84, 85, 86, 87};
        if (!enabled || name.drop_front(3).getAsInteger(10, version) ||
            !llvm::is_contained(versions, version)) return failure();
        if (ptx && *ptx != version) return failure();
        ptx = version;
      } else if (!name.starts_with("sm_")) {
        // The pinned NVPTX target exposes only sm_* and ptx* features.
        return failure();
      }
    }
    return success();
  }
  LogicalResult validateFeatures() const {
    for (const auto &feature : features) {
      StringRef name = feature.getKey();
      if (!name.starts_with("sm_")) continue;
      if (!chip || (name != "sm_90" && name != "sm_90a") ||
          (feature.second && name != *chip) || (!feature.second && name == *chip))
        return failure();
    }
    return success();
  }
};
LogicalResult readDevice(TargetDeviceSpecAttr device, ScopeTarget &state) {
  for (auto entry : device.getEntries()) {
    auto key = dyn_cast<StringAttr>(entry.getKey());
    if (!key) return failure();
    StringRef name = key.getValue();
    Attribute value = entry.getValue();
    if (name == "chip" || name == "features") {
      auto text = dyn_cast<StringAttr>(value);
      if (!text || failed(name == "chip" ? state.setChip(text.getValue()) : state.setFeatures(text.getValue())))
        return failure();
      continue;
    }
    unsigned expected = name == "warp_size" || name == "sector_bytes" || name == "bank_count" ? 32 :
                        name == "bank_width_bytes" ? 4 : 0;
    if (!expected) continue;
    auto number = dyn_cast<IntegerAttr>(value);
    if (!number || number.getValue() != expected) return failure();
  }
  return success();
}
LogicalResult readScope(Operation *scope, ScopeTarget &state) {
  if (auto attr = scope->getAttr("frisk.target")) {
    auto text = dyn_cast<StringAttr>(attr);
    if (!text || failed(state.setChip(text.getValue()))) return failure();
  }
  auto readNvvm = [&](Attribute attr) -> LogicalResult {
    auto target = dyn_cast<NVVM::NVVMTargetAttr>(attr);
    // Bound this adapter to the two NVPTX CUDA triples; a prefix match accepts
    // malformed architecture names that NVVMTargetAttr itself does not check.
    if (!target || (target.getTriple() != "nvptx64-nvidia-cuda" &&
                    target.getTriple() != "nvptx-nvidia-cuda") ||
        failed(state.setChip(target.getChip())) || failed(state.setFeatures(target.getFeatures())))
      return failure();
    return success();
  };
  if (auto attr = scope->getAttr("nvvm.target"))
    if (failed(readNvvm(attr))) return failure();
  if (isa<gpu::GPUModuleOp>(scope))
    if (auto attr = scope->getAttr("targets")) {
      auto targets = dyn_cast<ArrayAttr>(attr);
      if (!targets || targets.empty()) return failure();
      for (Attribute target : targets)
        if (failed(readNvvm(target))) return failure();
    }
  // The pinned DLTI dialect only permits device specs nested in a system spec.
  if (scope->hasAttr("dlti.target_device_spec")) return failure();
  return success();
}
LogicalResult readDltiScope(Operation *scope, ScopeTarget &state) {
  if (scope->hasAttr("dlti.target_device_spec")) return failure();
  if (auto attr = scope->getAttr("dlti.target_system_spec")) {
    auto system = dyn_cast<TargetSystemSpecAttr>(attr);
    if (!system) return failure();
    for (auto entry : system.getEntries()) {
      auto key = dyn_cast<StringAttr>(entry.getKey());
      if (!key) return failure();
      if (key.getValue() != "GPU") continue;
      auto device = dyn_cast<TargetDeviceSpecAttr>(entry.getValue());
      if (!device || failed(readDevice(device, state))) return failure();
    }
  }
  return success();
}
}
FailureOr<SM90TargetInfo> mlir::frisk::getSM90TargetInfo(Operation *op) {
  if (!op) return failure();
  ScopeTarget state;
  // Match the pinned DataLayoutInterfaces.cpp getTargetSystemSpec: target
  // device information comes from the nearest builtin module, independently
  // of a nearer gpu.module's architecture declaration.
  ModuleOp module = dyn_cast<ModuleOp>(op);
  if (!module) module = op->getParentOfType<ModuleOp>();
  bool readApplicableDlti = false;
  for (Operation *scope = op; scope; scope = scope->getParentOp()) {
    if (failed(readScope(scope, state))) {
      op->emitError("sm90-target: malformed, unsupported or conflicting target declarations");
      return failure();
    }
    if (scope == module.getOperation()) {
      readApplicableDlti = true;
      if (failed(readDltiScope(scope, state))) {
        op->emitError("sm90-target: malformed or conflicting applicable DLTI device specification");
        return failure();
      }
    }
    if (state.chip) break;
  }
  if (module && !readApplicableDlti && failed(readDltiScope(module, state))) {
    op->emitError("sm90-target: malformed or conflicting applicable DLTI device specification");
    return failure();
  }
  if (failed(state.validateFeatures())) {
    op->emitError("sm90-target: architecture features conflict with target chip");
    return failure();
  }
  SM90TargetInfo result;
  if (state.chip) result.chip = result.targetKey = *state.chip;
  result.supportsTMA = !state.ptx || *state.ptx >= 80;
  result.supportsCpAsync = !state.ptx || *state.ptx >= 70;
  result.supportsWGMMA = state.chip && *state.chip == "sm_90a" && result.supportsTMA;
  return result;
}
