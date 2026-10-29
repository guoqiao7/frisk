#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include <limits>

namespace mlir::frisk {
namespace {
uint64_t add(uint64_t a, uint64_t b, bool &saturated) {
  if (b > UINT64_MAX - a) { saturated = true; return UINT64_MAX; }
  return a + b;
}
uint64_t multiply(uint64_t a, uint64_t b, bool &saturated) {
  if (a && b > UINT64_MAX / a) { saturated = true; return UINT64_MAX; }
  return a * b;
}
bool isShared(Value value) {
  auto type = dyn_cast<BaseMemRefType>(value.getType());
  if (!type) return false;
  auto space = type.getMemorySpace();
  if (auto integer = dyn_cast_or_null<IntegerAttr>(space)) return integer.getInt() == 3;
  if (auto memory = dyn_cast_or_null<MemorySpaceAttr>(space))
    return memory.getValue() == attr::MemorySpace::Shared;
  return false;
}
bool isMetadata(Operation *op) {
  return isa<memref::AllocOp, memref::AllocaOp, memref::GetGlobalOp,
             memref::CastOp, memref::SubViewOp, LayoutViewOp, StorageContractOp>(op) ||
         op->getName().getStringRef() == "memref.dim";
}
bool isSynchronousAccess(Operation *op) {
  return isa<memref::LoadOp, memref::StoreOp, memref::CopyOp, CopyOp,
             FillOp, TileLoadOp, TileStoreOp, MmaOp>(op);
}
bool hasTensor(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<TensorType>(t); }) ||
         llvm::any_of(op->getResultTypes(), [](Type t) { return isa<TensorType>(t); });
}
// A named broadcast is atomic only when its implicit scalar region is exactly
// the canonical input yield. Outer operands/results already describe every
// Tensor SSA value; the scalar block arguments add no carrier lifetimes.
bool isCanonicalTensorBroadcast(Operation *op) {
  auto broadcast = dyn_cast<linalg::BroadcastOp>(op);
  if (!broadcast || op->getNumOperands() != 2 || op->getNumResults() != 1 ||
      op->getNumRegions() != 1)
    return false;
  auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto init = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!input || !init || op->getResult(0).getType() != init ||
      input.getElementType() != init.getElementType())
    return false;
  // Check dimensions before invoking the Linalg verifier/indexing-map builder,
  // including malformed in-memory mutations that a parser would reject.
  auto dimensions = op->getAttrOfType<DenseI64ArrayAttr>("dimensions");
  if (!dimensions || input.getRank() + dimensions.size() != init.getRank())
    return false;
  for (int64_t dimension : dimensions.asArrayRef())
    if (dimension < 0 || dimension >= init.getRank() ||
        llvm::count(dimensions.asArrayRef(), dimension) != 1)
      return false;
  unsigned inputDim = 0;
  for (int64_t dim = 0; dim < init.getRank(); ++dim)
    if (!llvm::is_contained(dimensions.asArrayRef(), dim) &&
        input.getDimSize(inputDim++) != init.getDimSize(dim))
      return false;
  auto &region = op->getRegion(0);
  if (!llvm::hasSingleElement(region)) return false;
  auto &block = region.front();
  if (block.getNumArguments() != 2 || !llvm::hasSingleElement(block) ||
      block.getArgument(0).getType() != input.getElementType() ||
      block.getArgument(1).getType() != init.getElementType())
    return false;
  auto yield = dyn_cast<linalg::YieldOp>(block.front());
  return yield && yield->getNumOperands() == 1 &&
         yield->getOperand(0) == block.getArgument(0) && succeeded(verify(op));
}
struct TensorInterval {
  Value value;
  uint64_t begin, end, slots = 0, bytes = 0;
  bool liveIn;
};
struct SharedInterval {
  Value root;
  uint64_t bytes;
  std::optional<uint64_t> begin, end;
  bool external, held = false;
};
struct ScopeAnalysis {
  Operation *owner;
  Block *body = nullptr;
  std::string name;
  llvm::json::Array unavailable;
  SmallVector<TensorInterval> tensors;
  SmallVector<SharedInterval> roots;
  llvm::DenseMap<Value, unsigned> tensorIDs, rootIDs;
  llvm::DenseMap<Operation *, uint64_t> operationOrder;
  SmallVector<std::pair<Operation *, std::string>> failures;
  bool saturated = false;

  ScopeAnalysis(Operation *owner, std::string name) : owner(owner), name(std::move(name)) {
    uint64_t index = 0;
    owner->walk<WalkOrder::PreOrder>([&](Operation *op) { operationOrder[op] = index++; });
  }

  uint64_t position(Operation *op) const {
    if (!body || op->getBlock() != body) return 0;
    uint64_t index = 1;
    for (auto &item : *body) {
      if (&item == op) return index;
      ++index;
    }
    return 0;
  }
  void reject(Operation *op, StringRef reason) {
    for (auto &failure : failures)
      if (failure.first == op && failure.second == reason) return;
    failures.emplace_back(op, reason.str());
  }
  bool selectBody() {
    auto &region = owner->getRegion(0);
    if (!llvm::hasSingleElement(region)) { reject(owner, "multiple-blocks"); return false; }
    body = &region.front();
    SmallVector<ParallelOp> parallels;
    for (Operation &op : *body)
      if (auto parallel = dyn_cast<ParallelOp>(op)) parallels.push_back(parallel);
    if (!parallels.empty()) {
      if (!isa<KernelOp>(owner) || parallels.size() != 1 ||
          parallels[0].getRanges() != ArrayRef<int64_t>{1}) {
        reject(parallels[0], "unsupported-parallel-domain"); return false;
      }
      auto parallel = parallels[0];
      if (!llvm::hasSingleElement(parallel.getRegion())) {
        reject(parallel, "multiple-blocks"); return false;
      }
      for (Operation &op : *body) {
        if (&op == parallel.getOperation()) continue;
        if (op.getNumRegions()) reject(&op, "unsupported-control-flow");
        if (hasTensor(&op)) reject(&op, "tensor-work-outside-parallel");
        if (!isMetadata(&op) && llvm::any_of(op.getOperands(), isShared))
          reject(&op, "unsupported-shared-lifetime");
      }
      name += "/parallel";
      body = &parallel.getRegion().front();
    }
    for (Operation &op : *body)
      if ((isa<linalg::BroadcastOp>(op) || op.getNumRegions()) &&
          !isCanonicalTensorBroadcast(&op))
        reject(&op, "unsupported-control-flow");
    return failures.empty();
  }
  void addTensor(Value value, uint64_t begin, bool liveIn, Operation *at) {
    if (!isa<TensorType>(value.getType()) || tensorIDs.count(value)) return;
    tensorIDs[value] = tensors.size();
    tensors.push_back({value, begin, begin, 0, 0, liveIn});
    auto &interval = tensors.back();
    auto type = dyn_cast<RankedTensorType>(value.getType());
    auto encoding = type ? dyn_cast_or_null<DistributedEncodingAttr>(type.getEncoding()) : DistributedEncodingAttr();
    if (!type || !type.hasStaticShape() || !type.getElementType().isIntOrFloat() ||
        llvm::any_of(type.getShape(), [](int64_t extent) { return extent <= 0; }) ||
        !encoding || encoding.getTopology().size() != 5 ||
        encoding.getTopology()[4] != 1 || failed(encoding.verifyForType(type, at->getLoc()))) {
      reject(at, "unsupported-tensor-layout"); return;
    }
    interval.slots = encoding.getTopology()[0];
    uint64_t bytes = (uint64_t(type.getElementTypeBitWidth()) + 7) / 8;
    interval.bytes = multiply(interval.slots, bytes, saturated);
  }
  std::optional<unsigned> addRoot(Value endpoint, Operation *at) {
    auto info = analyzeStorageAlias(endpoint);
    if (failed(info)) { reject(at, "unsupported-storage-alias"); return std::nullopt; }
    auto found = rootIDs.find(info->root);
    if (found != rootIDs.end()) return found->second;
    auto *definition = info->root.getDefiningOp();
    bool internal = definition && isa<memref::AllocOp, memref::AllocaOp>(definition) &&
                    owner->isAncestor(definition);
    if (internal && definition->getBlock() != body && definition->getBlock() != &owner->getRegion(0).front()) {
      reject(at, "unsupported-shared-lifetime"); return std::nullopt;
    }
    unsigned id = roots.size();
    rootIDs[info->root] = id;
    roots.push_back({info->root, info->upperBit / 8 + (info->upperBit % 8 != 0),
                     std::nullopt, std::nullopt, !internal});
    return id;
  }
  void intervals() {
    // Block arguments, then first-appearance captures, then actual definitions.
    for (BlockArgument argument : body->getArguments())
      for (Operation &op : *body)
        if (llvm::is_contained(op.getOperands(), argument)) {
          addTensor(argument, 0, true, &op);
          break;
        }
    for (Operation &op : *body)
      for (Value value : op.getOperands())
        if (!value.getDefiningOp() || value.getDefiningOp()->getBlock() != body)
          addTensor(value, 0, true, &op);
    uint64_t index = 0;
    for (Operation &op : *body) {
      ++index;
      for (Value value : op.getResults()) addTensor(value, index, false, &op);
    }
    // Unused owned allocations still contribute declaration capacity.
    owner->walk([&](Operation *op) {
      if (isa<memref::AllocOp, memref::AllocaOp>(op))
        for (Value result : op->getResults())
          if (isShared(result)) addRoot(result, op);
    });
    index = 0;
    uint64_t end = body->getOperations().size();
    for (Operation &op : *body) {
      ++index;
      for (Value value : op.getOperands()) {
        auto found = tensorIDs.find(value);
        if (found != tensorIDs.end()) tensors[found->second].end = index;
      }
      if (isMetadata(&op)) {
        // A dynamic cast intermediary is not itself an endpoint proof demand:
        // a later static endpoint may recover the proven source descriptor.
        bool hasStaticEndpoint = false;
        Value dynamicEndpoint;
        for (Value value : llvm::concat<Value>(op.getOperands(), op.getResults()))
          if (isShared(value)) {
            dynamicEndpoint = value;
            if (auto type = dyn_cast<MemRefType>(value.getType()); type && type.hasStaticShape())
              { hasStaticEndpoint = true; addRoot(value, &op); }
          }
        if (dynamicEndpoint && !hasStaticEndpoint) addRoot(dynamicEndpoint, &op);
        continue;
      }
      bool shared = llvm::any_of(op.getOperands(), isShared) || llvm::any_of(op.getResults(), isShared);
      if (!shared) continue;
      bool async = false;
      if (auto contract = op.getAttrOfType<DictionaryAttr>("frisk.copy_contract"))
        if (auto path = contract.getAs<StringAttr>("path"))
          async = path.getValue() == "cp.async" || path.getValue() == "tma";
      bool token = llvm::any_of(op.getResultTypes(), [](Type type) {
        if (isa<gpu::AsyncTokenType>(type)) return true;
        if (auto opaque = dyn_cast<OpaqueType>(type))
          return opaque.getDialectNamespace() == "async";
        return type.getDialect().getNamespace() == "async";
      });
      bool freesShared = false;
      if (auto interface = dyn_cast<MemoryEffectOpInterface>(&op)) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        interface.getEffects(effects);
        freesShared = llvm::any_of(effects, [](const auto &effect) {
          return isa<MemoryEffects::Free>(effect.getEffect()) &&
                 (!effect.getValue() || isShared(effect.getValue()));
        });
      }
      if (freesShared || op.hasTrait<OpTrait::ReturnLike>() || token)
        reject(&op, "unsupported-shared-lifetime");
      for (Value value : op.getOperands()) {
        if (!isShared(value)) continue;
        auto id = addRoot(value, &op);
        if (!id) continue;
        auto &root = roots[*id];
        if (!root.begin) root.begin = index;
        root.held |= async || !isSynchronousAccess(&op);
        root.end = root.held ? end : index;
      }
      for (Value value : op.getResults())
        if (isShared(value)) addRoot(value, &op);
    }
  }
  llvm::json::Object run() {
    if (selectBody()) intervals();
    llvm::stable_sort(failures, [&](const auto &a, const auto &b) {
      return operationOrder.lookup(a.first) < operationOrder.lookup(b.first);
    });
    for (auto &[op, reason] : failures)
      unavailable.push_back(llvm::json::Object{
          {"scope", name}, {"opcode", op->getName().getStringRef().str()},
          {"operation", position(op)}, {"reason", reason}});
    bool available = failures.empty();
    uint64_t slots = 0, bytes = 0, shared = 0, declared = 0;
    uint64_t slotsAt = 0, bytesAt = 0, sharedAt = 0;
    llvm::json::Array tensorJSON, rootJSON;
    if (available) {
      for (auto [id, value] : llvm::enumerate(tensors)) {
        std::string dtype;
        llvm::raw_string_ostream(dtype) << cast<RankedTensorType>(value.value.getType()).getElementType();
        tensorJSON.push_back(llvm::json::Object{{"id", id}, {"dtype", dtype},
            {"slots_per_thread", value.slots}, {"bytes_per_thread", value.bytes},
            {"begin", value.begin}, {"end", value.end}, {"live_in", value.liveIn}});
      }
      for (auto [id, root] : llvm::enumerate(roots)) {
        declared = add(declared, root.bytes, saturated);
        rootJSON.push_back(llvm::json::Object{{"id", id},
            {"origin", std::string(root.external ? "external" : "internal")}, {"bytes", root.bytes},
            {"begin", root.begin ? llvm::json::Value(*root.begin) : llvm::json::Value(nullptr)},
            {"end", root.end ? llvm::json::Value(*root.end) : llvm::json::Value(nullptr)},
            {"held_to_scope_end", root.held}});
      }
      // Re-sum inclusive boundaries: never subtract from a saturated total.
      for (uint64_t point = 0; point <= body->getOperations().size(); ++point) {
        uint64_t liveSlots = 0, liveBytes = 0, liveShared = 0;
        for (auto &value : tensors)
          if (value.begin <= point && point <= value.end) {
            liveSlots = add(liveSlots, value.slots, saturated);
            liveBytes = add(liveBytes, value.bytes, saturated);
          }
        for (auto &root : roots)
          if (root.begin && *root.begin <= point && point <= *root.end)
            liveShared = add(liveShared, root.bytes, saturated);
        if (liveSlots > slots) { slots = liveSlots; slotsAt = point; }
        if (liveBytes > bytes) { bytes = liveBytes; bytesAt = point; }
        if (liveShared > shared) { shared = liveShared; sharedAt = point; }
      }
    }
    auto number = [&](uint64_t n) { return available ? llvm::json::Value(n) : llvm::json::Value(nullptr); };
    return llvm::json::Object{{"scope", name}, {"status", std::string(available ? "available" : "unavailable")},
        {"logical_live_carrier_slots_per_thread", number(slots)},
        {"logical_live_carrier_bytes_per_thread", number(bytes)},
        {"logical_live_shared_root_bytes", number(shared)}, {"declared_shared_root_bytes", number(declared)},
        {"peak_slots_at", number(slotsAt)}, {"peak_bytes_at", number(bytesAt)}, {"peak_shared_at", number(sharedAt)},
        {"saturated", saturated}, {"tensor_values", std::move(tensorJSON)},
        {"shared_roots", std::move(rootJSON)}, {"unavailable_operations", std::move(unavailable)}};
  }
};
} // namespace

llvm::json::Object collectLayoutIRResources(Operation *operation) {
  ScopedDiagnosticHandler quiet(operation->getContext(), [](Diagnostic &) { return success(); });
  llvm::json::Array scopes;
  unsigned available = 0, unavailable = 0;
  operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<func::FuncOp, KernelOp>(op)) return;
    if (op->getRegion(0).empty()) return;
    auto symbol = op->getAttrOfType<StringAttr>("sym_name");
    std::string name = isa<KernelOp>(op) ? "kernel@" : "func@";
    name += symbol ? symbol.getValue().str() : "anonymous";
    auto scope = ScopeAnalysis(op, name).run();
    if (scope.getString("status") == "available") ++available;
    else ++unavailable;
    scopes.push_back(std::move(scope));
  });
  return llvm::json::Object{{"schema_version", 1},
      {"status", std::string(!unavailable ? "available" : available ? "partial" : "unavailable")},
      {"counting", "conservative-static-ssa-and-shared-root-liveness"},
      {"physical_registers", "not-modeled"}, {"execution_temporaries", "not-modeled"},
      {"backend_allocation_reuse", "not-modeled"}, {"scopes", std::move(scopes)}};
}
} // namespace mlir::frisk
