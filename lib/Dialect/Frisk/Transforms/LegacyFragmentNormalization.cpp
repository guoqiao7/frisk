#include "LegacyFragmentNormalization.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/IR/LegacyImportSemantics.h"
#include "Dialect/Frisk/Target/SM90/SM90MmaLayoutProof.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::frisk;
namespace {
constexpr StringLiteral escapeMessage =
    "legacy local fragment escapes supported normalization boundary";
using State = DenseMap<Value, Value>;

bool isLocal(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && getFriskMemorySpace(type) == attr::MemorySpace::Local;
}
RankedTensorType tensorType(Value value) {
  auto type = cast<MemRefType>(value.getType());
  return RankedTensorType::get(type.getShape(), type.getElementType());
}
bool isSupportedRegion(Operation *op) {
  return isa<ModuleOp, func::FuncOp, KernelOp, ParallelOp, scf::IfOp,
             scf::ForOp, scf::WhileOp, ForOp>(op);
}
Operation *functionScope(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (isa<func::FuncOp, KernelOp>(parent)) return parent;
  return nullptr;
}
LogicalResult reject(Operation *op, Twine reason = {}) {
  return op->emitError(escapeMessage) << (reason.isTriviallyEmpty() ? "" : ": ") << reason;
}

bool containsLegacyLayout(Type type);
bool containsLegacyLayout(Attribute attribute) {
  if (isa<LayoutAttr>(attribute)) return true; // Also represents legacy fragments.
  bool found = false;
  attribute.walkImmediateSubElements(
      [&](Attribute nested) { found |= containsLegacyLayout(nested); },
      [&](Type nested) { found |= containsLegacyLayout(nested); });
  return found;
}
bool containsLegacyLayout(Type type) {
  bool found = false;
  type.walkImmediateSubElements(
      [&](Attribute nested) { found |= containsLegacyLayout(nested); },
      [&](Type nested) { found |= containsLegacyLayout(nested); });
  return found;
}

LogicalResult verifyImportAttributes(Operation *scope, bool allowLegacyThreads) {
  auto result = scope->walk([&](Operation *op) -> WalkResult {
    for (NamedAttribute named : op->getAttrs()) {
      if (containsLegacyLayout(named.getValue())) {
        op->emitError("legacy-layout-attribute: obsolete layout/fragment contract is not importable");
        return WalkResult::interrupt();
      }
      StringRef name = named.getName().getValue();
      if (!name.starts_with("frisk.")) continue;
      if (name == "frisk.target" || name == "frisk.legacy_semantics" ||
          name == "frisk.execution_threads" || name == "frisk.execution_layout" ||
          name == "frisk.writer_policy" || name == "frisk.vector_bytes" ||
          name == "frisk.mma_contract" || name == "frisk.reduction_contract" ||
          (allowLegacyThreads && name == "frisk.threads" && isa<GemmOp>(op)))
        continue;
      op->emitError("legacy-attribute-contract: unsupported semantic attribute ") << name;
      return WalkResult::interrupt();
    }
    for (Type type : op->getResultTypes())
      if (containsLegacyLayout(type)) {
        op->emitError("legacy-layout-attribute: obsolete tensor/type layout contract");
        return WalkResult::interrupt();
      }
    // Region arguments are independent type-bearing definitions, including
    // unused arguments whose types cannot be discovered through operand uses.
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (containsLegacyLayout(argument.getType())) {
            op->emitError("legacy-layout-attribute: obsolete block argument type layout contract");
            return WalkResult::interrupt();
          }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

class Normalizer {
public:
  explicit Normalizer(ModuleOp module) : module(module), builder(module.getContext()) {}
  LogicalResult run();

private:
  ModuleOp module;
  OpBuilder builder;
  SmallVector<Value> roots;
  DenseMap<Value, Value> aliases;
  SmallVector<Operation *> eraseLifetimes;

  Value root(Value value) const { return aliases.lookup(value); }
  LogicalResult preflight();
  LogicalResult collectAliases(Value value, Value allocation);
  LogicalResult checkAttributes(Operation *op, ArrayRef<StringRef> allowed = {});
  LogicalResult checkLifetime(Value allocation);
  LogicalResult process(Block &block, State &state);
  LogicalResult processIndependentStorage(Operation *scope);
  LogicalResult processIf(scf::IfOp op, State &state);
  LogicalResult processFor(scf::ForOp op, State &state);
  LogicalResult processWhile(scf::WhileOp op, State &state);
  LogicalResult processLegacyFor(ForOp op, State &state);
  LogicalResult processMath(Operation *op, State &state);
  LogicalResult processCopy(CopyOp op, State &state);
  FailureOr<Value> read(Value value, Operation *at, State &state);
  FailureOr<Value> storageView(Value value, Operation *at);
  SmallVector<Value> modifiedExternal(Operation *scope, const State &state);
  LogicalResult requireInitialized(ArrayRef<Value> values, Operation *at, State &state);
  FailureOr<int64_t> threads(Operation *op, bool gemm);
  void copyMetadata(Operation *from, Operation *to);
};

LogicalResult Normalizer::checkAttributes(Operation *op, ArrayRef<StringRef> allowed) {
  for (NamedAttribute attr : op->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if ((name.starts_with("frisk.") || name == "layout") &&
        !llvm::is_contained(allowed, name))
      return op->emitError("legacy-attribute-contract: unsupported semantic attribute ") << name;
  }
  return success();
}

void Normalizer::copyMetadata(Operation *from, Operation *to) {
  for (NamedAttribute attr : from->getDiscardableAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name != "frisk.threads" && name != "frisk.legacy_semantics")
      to->setAttr(attr.getName(), attr.getValue());
  }
}

LogicalResult Normalizer::collectAliases(Value value, Value allocation) {
  if (aliases.count(value)) return success();
  aliases[value] = allocation;
  for (Operation *user : value.getUsers()) {
    auto cast = dyn_cast<memref::CastOp>(user);
    if (!cast) continue;
    if (cast.getType() != value.getType() || failed(checkAttributes(cast)))
      return reject(cast, "Local cast must preserve the complete static identity descriptor");
    if (failed(collectAliases(cast.getResult(), allocation))) return failure();
  }
  return success();
}

LogicalResult Normalizer::checkLifetime(Value allocation) {
  Operation *alloc = allocation.getDefiningOp();
  auto type = cast<MemRefType>(allocation.getType());
  uint64_t points = 1;
  if (!type.hasStaticShape() || !type.getLayout().isIdentity() || type.getRank() == 0 ||
      (!type.getElementType().isF16() && !type.getElementType().isBF16() &&
       !type.getElementType().isF32()))
    return reject(alloc, "Local root requires static identity f16/bf16/f32 storage");
  for (int64_t extent : type.getShape()) {
    if (extent <= 1 || !llvm::isPowerOf2_64(extent) || uint64_t(extent) > 65536 / points)
      return reject(alloc, "Local root requires power-of-two extents >1 and at most 65536 elements");
    points *= extent;
  }
  if (failed(checkAttributes(alloc))) return failure();
  for (Operation *parent = alloc->getParentOp(); parent; parent = parent->getParentOp()) {
    if (!isSupportedRegion(parent)) return reject(alloc, "unsupported enclosing region");
    for (Region &region : parent->getRegions())
      if (!region.empty() && !llvm::hasSingleElement(region))
        return reject(parent, "general CFG is not a sequential fragment lifetime");
  }
  memref::DeallocOp dealloc;
  SmallVector<Operation *> users;
  for (const auto &entry : aliases) {
    if (entry.second != allocation) continue;
    for (Operation *user : entry.first.getUsers()) {
      users.push_back(user);
      if (!isa<FillOp, CopyOp, GemmOp, ReduceOp, memref::CastOp, memref::DeallocOp>(user))
        return reject(user);
      if (user->getParentOfType<ParallelOp>() != alloc->getParentOfType<ParallelOp>())
        return reject(user, "Local root crosses a Parallel boundary");
      if (functionScope(user) != functionScope(alloc))
        return reject(user, "Local root crosses a function or Kernel boundary");
      for (Operation *parent = user->getParentOp(); parent && parent != alloc->getParentOp();
           parent = parent->getParentOp())
        if (!isSupportedRegion(parent)) return reject(user, "unsupported enclosing region");
      if (auto release = dyn_cast<memref::DeallocOp>(user)) {
        if (dealloc || release->getBlock() != alloc->getBlock())
          return reject(release, "dealloc must be unique and in the allocation block");
        dealloc = release;
      }
    }
  }
  if (dealloc) {
    DominanceInfo dominance(module);
    if (!dominance.dominates(alloc, dealloc)) return reject(dealloc, "dealloc does not follow allocation");
    for (Operation *user : users) {
      if (user == dealloc) continue;
      Operation *ancestor = alloc->getBlock()->findAncestorOpInBlock(*user);
      if (!ancestor || !ancestor->isBeforeInBlock(dealloc))
        return reject(user, "use after dealloc");
    }
  }
  return success();
}

LogicalResult Normalizer::preflight() {
  // Validate math before rewriting even dead computations or allocations.
  auto math = module.walk([&](Operation *op) -> WalkResult {
    if (isa<GemmOp, ReduceOp>(op) && failed(verifyLegacyTensorV1Semantics(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (math.wasInterrupted()) return failure();
  if (failed(verifyImportAttributes(module, /*allowLegacyThreads=*/true)))
    return failure();
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<AllocBufferOp, memref::AllocOp, memref::AllocaOp>(op) && isLocal(op->getResult(0)))
      roots.push_back(op->getResult(0));
  });
  for (Value value : roots)
    if (failed(collectAliases(value, value))) return failure();
  for (Value value : roots)
    if (failed(checkLifetime(value))) return failure();
  auto uses = module.walk([&](Operation *op) -> WalkResult {
    if (isa<CopyOp, FillOp, GemmOp, ReduceOp>(op))
      for (Value operand : op->getOperands())
        if (isLocal(operand) && !root(operand)) {
          (void)reject(op, "Local operand is not a promotable allocation");
          return WalkResult::interrupt();
        }
    if (isa<GemmOp, ReduceOp>(op) &&
        failed(checkAttributes(op, {"frisk.legacy_semantics", "frisk.target",
                                   "frisk.execution_threads", "frisk.threads"})))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return failure(uses.wasInterrupted());
}

FailureOr<Value> Normalizer::read(Value value, Operation *at, State &state) {
  Value allocation = root(value);
  if (!allocation) {
    (void)reject(at, "expected promotable Local fragment");
    return failure();
  }
  Value current = state.lookup(allocation);
  if (!current) {
    at->emitError("legacy-fragment-uninitialized");
    return failure();
  }
  return current;
}

FailureOr<Value> Normalizer::storageView(Value value, Operation *at) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || isLocal(value) || failed(analyzeStorageAlias(value))) {
    at->emitError("legacy-storage-boundary: unsupported storage alias or descriptor");
    return failure();
  }
  if (value.getDefiningOp<LayoutViewOp>()) return value;
  DominanceInfo dominance(module);
  for (Operation *user : value.getUsers())
    if (auto view = dyn_cast<LayoutViewOp>(user))
      if (dominance.dominates(view.getOperation(), at)) return view.getResult();
  builder.setInsertionPoint(at);
  return builder.create<LayoutViewOp>(at->getLoc(), type, value, StorageLayoutAttr()).getResult();
}

SmallVector<Value> Normalizer::modifiedExternal(Operation *scope, const State &state) {
  DenseSet<Value> modified;
  scope->walk([&](Operation *op) {
    Value destination;
    if (auto fill = dyn_cast<FillOp>(op)) destination = fill.getMemref();
    if (auto copy = dyn_cast<CopyOp>(op)) destination = copy.getDst();
    if (auto gemm = dyn_cast<GemmOp>(op)) destination = gemm.getC();
    if (auto reduce = dyn_cast<ReduceOp>(op)) destination = reduce.getDst();
    if (destination && root(destination) && state.count(root(destination)))
      modified.insert(root(destination));
  });
  SmallVector<Value> ordered;
  for (Value allocation : roots)
    if (modified.count(allocation)) ordered.push_back(allocation);
  return ordered;
}

LogicalResult Normalizer::requireInitialized(ArrayRef<Value> values, Operation *at, State &state) {
  for (Value value : values)
    if (failed(read(value, at, state))) return failure();
  return success();
}

LogicalResult Normalizer::processIf(scf::IfOp op, State &state) {
  auto modified = modifiedExternal(op, state);
  State thenState = state, elseState = state;
  if (failed(process(op.getThenRegion().front(), thenState))) return failure();
  if (!op.getElseRegion().empty() && failed(process(op.getElseRegion().front(), elseState)))
    return failure();
  if (modified.empty()) return success();
  if (failed(requireInitialized(modified, op, thenState)) ||
      failed(requireInitialized(modified, op, elseState))) return failure();
  SmallVector<Type> types(op.getResultTypes());
  for (Value allocation : modified) types.push_back(tensorType(allocation));
  builder.setInsertionPoint(op);
  auto replacement = builder.create<scf::IfOp>(op.getLoc(), types, op.getCondition(), true);
  replacement->setAttrs(op->getAttrs());
  replacement.getThenRegion().takeBody(op.getThenRegion());
  if (!op.getElseRegion().empty()) {
    replacement.getElseRegion().takeBody(op.getElseRegion());
  } else {
    // Result-bearing IfOp builders deliberately leave the blocks unterminated.
    builder.setInsertionPointToEnd(&replacement.getElseRegion().front());
    builder.create<scf::YieldOp>(op.getLoc());
  }
  for (auto [region, branchState] : {std::make_pair(&replacement.getThenRegion(), &thenState),
                                   std::make_pair(&replacement.getElseRegion(), &elseState)}) {
    auto yield = cast<scf::YieldOp>(region->front().getTerminator());
    SmallVector<Value> operands(yield.getOperands());
    for (Value allocation : modified) operands.push_back(branchState->lookup(allocation));
    yield->setOperands(operands);
  }
  for (auto [oldResult, newResult] : llvm::zip(op.getResults(), replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  for (auto [index, allocation] : llvm::enumerate(modified))
    state[allocation] = replacement.getResult(op.getNumResults() + index);
  op.erase();
  return success();
}

LogicalResult Normalizer::processFor(scf::ForOp op, State &state) {
  auto modified = modifiedExternal(op, state);
  if (failed(requireInitialized(modified, op, state))) return failure();
  State bodyState = state;
  if (modified.empty()) return process(*op.getBody(), bodyState);
  SmallVector<Value> inits(op.getInitArgs());
  for (Value allocation : modified) inits.push_back(state.lookup(allocation));
  builder.setInsertionPoint(op);
  auto replacement = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
      op.getUpperBound(), op.getStep(), inits);
  replacement->setAttrs(op->getAttrs());
  replacement.getRegion().takeBody(op.getRegion());
  for (Value allocation : modified)
    bodyState[allocation] = replacement.getBody()->addArgument(tensorType(allocation), op.getLoc());
  if (failed(process(*replacement.getBody(), bodyState))) return failure();
  auto yield = cast<scf::YieldOp>(replacement.getBody()->getTerminator());
  SmallVector<Value> operands(yield.getOperands());
  for (Value allocation : modified) operands.push_back(bodyState.lookup(allocation));
  yield->setOperands(operands);
  for (auto [oldResult, newResult] : llvm::zip(op.getResults(), replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  for (auto [index, allocation] : llvm::enumerate(modified))
    state[allocation] = replacement.getResult(op.getNumResults() + index);
  op.erase();
  return success();
}

LogicalResult Normalizer::processWhile(scf::WhileOp op, State &state) {
  auto modified = modifiedExternal(op, state);
  if (failed(requireInitialized(modified, op, state))) return failure();
  State beforeState = state, afterState = state;
  if (modified.empty()) {
    if (failed(process(*op.getBeforeBody(), beforeState))) return failure();
    return process(*op.getAfterBody(), afterState);
  }
  SmallVector<Value> inits(op.getInits());
  SmallVector<Type> types(op.getResultTypes());
  for (Value allocation : modified) {
    inits.push_back(state.lookup(allocation));
    types.push_back(tensorType(allocation));
  }
  builder.setInsertionPoint(op);
  auto replacement = builder.create<scf::WhileOp>(op.getLoc(), types, inits);
  replacement->setAttrs(op->getAttrs());
  replacement.getBefore().takeBody(op.getBefore());
  replacement.getAfter().takeBody(op.getAfter());
  for (Value allocation : modified) {
    beforeState[allocation] = replacement.getBeforeBody()->addArgument(tensorType(allocation), op.getLoc());
    afterState[allocation] = replacement.getAfterBody()->addArgument(tensorType(allocation), op.getLoc());
  }
  if (failed(process(*replacement.getBeforeBody(), beforeState)) ||
      failed(process(*replacement.getAfterBody(), afterState))) return failure();
  auto condition = replacement.getConditionOp();
  SmallVector<Value> conditionOperands(condition.getOperands());
  auto yield = replacement.getYieldOp();
  SmallVector<Value> yieldOperands(yield.getOperands());
  for (Value allocation : modified) {
    conditionOperands.push_back(beforeState.lookup(allocation));
    yieldOperands.push_back(afterState.lookup(allocation));
  }
  condition->setOperands(conditionOperands);
  yield->setOperands(yieldOperands);
  for (auto [oldResult, newResult] : llvm::zip(op.getResults(), replacement.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  for (auto [index, allocation] : llvm::enumerate(modified))
    state[allocation] = replacement.getResult(op.getNumResults() + index);
  op.erase();
  return success();
}

LogicalResult Normalizer::processLegacyFor(ForOp op, State &state) {
  bool touches = false;
  op->walk([&](Operation *nested) {
    for (Value value : nested->getOperands()) touches |= bool(root(value));
    for (Value value : nested->getResults()) touches |= bool(root(value));
  });
  if (!touches) return process(*op.getBody(), state);
  if (op.getStep() <= 0 || op.getLower() > op.getUpper() || failed(checkAttributes(op)))
    return reject(op, "legacy for requires a static positive-step half-open range");
  builder.setInsertionPoint(op);
  auto lb = builder.create<arith::ConstantIndexOp>(op.getLoc(), op.getLower());
  auto ub = builder.create<arith::ConstantIndexOp>(op.getLoc(), op.getUpper());
  auto step = builder.create<arith::ConstantIndexOp>(op.getLoc(), op.getStep());
  auto replacement = builder.create<scf::ForOp>(op.getLoc(), lb, ub, step);
  copyMetadata(op, replacement);
  replacement.getRegion().takeBody(op.getRegion());
  Operation *end = replacement.getBody()->getTerminator();
  builder.setInsertionPoint(end);
  builder.create<scf::YieldOp>(end->getLoc());
  end->erase();
  op.erase();
  return processFor(replacement, state);
}

FailureOr<int64_t> Normalizer::threads(Operation *op, bool gemm) {
  std::optional<int64_t> explicitThreads;
  for (StringRef name : {StringRef("frisk.execution_threads"), StringRef("frisk.threads")}) {
    Attribute value = op->getAttr(name);
    if (!value) continue;
    auto integer = dyn_cast<IntegerAttr>(value);
    if ((!gemm && name == "frisk.threads") || !integer ||
        !integer.getType().isSignlessInteger(64) || integer.getInt() <= 0 ||
        (explicitThreads && *explicitThreads != integer.getInt())) {
      op->emitError("legacy-thread-contract: malformed or conflicting execution threads");
      return failure();
    }
    explicitThreads = integer.getInt();
  }
  if (auto parallel = op->getParentOfType<ParallelOp>()) {
    if (explicitThreads && *explicitThreads != parallel.getThreads()) {
      op->emitError("legacy-thread-contract: execution threads conflict with Parallel");
      return failure();
    }
    explicitThreads = parallel.getThreads();
  }
  return explicitThreads.value_or(128);
}

LogicalResult Normalizer::processMath(Operation *operation, State &state) {
  builder.setInsertionPoint(operation);
  auto executionThreads = threads(operation, isa<GemmOp>(operation));
  if (failed(executionThreads)) return failure();
  if (auto gemm = dyn_cast<GemmOp>(operation)) {
    auto aType = cast<MemRefType>(gemm.getA().getType());
    auto bType = cast<MemRefType>(gemm.getB().getType());
    auto cType = cast<MemRefType>(gemm.getC().getType());
    if (!cType.getElementType().isF32())
      return gemm.emitError("legacy-gemm-accumulator: f32 required; implicit promotion is forbidden");
    if ((!aType.getElementType().isF16() && !aType.getElementType().isBF16()) ||
        aType.getElementType() != bType.getElementType())
      return gemm.emitError("legacy-gemm-input: matching f16/bf16 A/B required");
    if (auto error = getLegacyGemmValidationError(aType, bType, cType,
          gemm.getTransA(), gemm.getTransB(), gemm.getM(), gemm.getN(), gemm.getK()))
      return gemm.emitError(*error);
    StringAttr target;
    for (Operation *scope = operation; scope; scope = scope->getParentOp())
      if (Attribute attr = scope->getAttr("frisk.target")) {
        target = dyn_cast<StringAttr>(attr);
        break;
      }
    if (!target || target.getValue() != "sm_90a")
      return gemm.emitError("sm90-mma-target: explicit sm_90a target required");
    if (!root(gemm.getC()) || getFriskMemorySpace(bType) != attr::MemorySpace::Shared ||
        (!root(gemm.getA()) && getFriskMemorySpace(aType) != attr::MemorySpace::Shared))
      return reject(gemm, "Gemm requires Local C, Shared B and Local/Shared A");
    auto a = root(gemm.getA()) ? read(gemm.getA(), operation, state) : storageView(gemm.getA(), operation);
    auto b = storageView(gemm.getB(), operation);
    if (failed(a) || failed(b)) return failure();
    builder.setInsertionPoint(operation);
    Value init;
    if (gemm.getClearAccum()) {
      auto type = tensorType(gemm.getC());
      init = builder.create<arith::ConstantOp>(gemm.getLoc(), type,
          DenseElementsAttr::get(type, builder.getF32FloatAttr(0.0)));
    } else {
      auto current = read(gemm.getC(), operation, state);
      if (failed(current)) return failure();
      init = *current;
    }
    auto mma = builder.create<MmaOp>(gemm.getLoc(), tensorType(gemm.getC()), *a, *b,
        init, gemm.getM(), gemm.getN(), gemm.getK(), gemm.getTransA(), gemm.getTransB(), gemm.getPolicy());
    copyMetadata(gemm, mma);
    mma->setAttr("frisk.execution_threads", builder.getI64IntegerAttr(*executionThreads));
    if (failed(getSM90MmaGeometry(mma, *executionThreads)))
      return gemm.emitError("sm90-mma-thread-group: unsupported tile, threads or policy grouping");
    state[root(gemm.getC())] = mma.getResult();
    gemm.erase();
    return success();
  }
  auto reduce = cast<ReduceOp>(operation);
  if (reduce.getKind() != "add" && reduce.getKind() != "max" && reduce.getKind() != "min")
    return reduce.emitError("legacy-reduce-kind: only add/max/min are supported");
  if (!root(reduce.getDst())) return reject(reduce, "Reduce requires a Local destination");
  FailureOr<Value> source = failure();
  if (root(reduce.getSrc())) source = read(reduce.getSrc(), operation, state);
  else {
    auto view = storageView(reduce.getSrc(), operation);
    if (failed(view)) return failure();
    builder.setInsertionPoint(operation);
    source = builder.create<TileLoadOp>(reduce.getLoc(), tensorType(reduce.getSrc()), *view).getResult();
  }
  if (failed(source)) return failure();
  Value previous;
  if (!reduce.getClear()) {
    auto current = read(reduce.getDst(), operation, state);
    if (failed(current)) return failure();
    previous = *current;
  }
  builder.setInsertionPoint(operation);
  auto reduced = builder.create<ReduceTensorOp>(reduce.getLoc(), tensorType(reduce.getDst()),
      *source, reduce.getKind() == "add" ? "sum" : reduce.getKind(), reduce.getDim());
  copyMetadata(reduce, reduced);
  reduced->setAttr("frisk.execution_threads", builder.getI64IntegerAttr(*executionThreads));
  Value result = reduced.getResult();
  if (previous) {
    if (reduce.getKind() == "add") result = builder.create<arith::AddFOp>(reduce.getLoc(), previous, result);
    else if (reduce.getKind() == "max") result = builder.create<arith::MaximumFOp>(reduce.getLoc(), previous, result);
    else result = builder.create<arith::MinimumFOp>(reduce.getLoc(), previous, result);
  }
  state[root(reduce.getDst())] = result;
  reduce.erase();
  return success();
}

LogicalResult Normalizer::processCopy(CopyOp copy, State &state) {
  Value src = copy.getSrc(), dst = copy.getDst();
  auto srcType = cast<MemRefType>(src.getType()), dstType = cast<MemRefType>(dst.getType());
  if (!srcType.hasStaticShape() || !dstType.hasStaticShape() ||
      srcType.getShape() != dstType.getShape() || srcType.getElementType() != dstType.getElementType() ||
      srcType.getShape() != copy.getSrcExtents() || dstType.getShape() != copy.getDstExtents() ||
      !copy.getSrcIndices().empty() || !copy.getDstIndices().empty() ||
      copy.getSrcMap().getNumInputs() || copy.getDstMap().getNumInputs() ||
      copy.getSrcMap().getNumResults() || copy.getDstMap().getNumResults())
    return copy.emitError("legacy-copy-boundary: requires static whole-tile copy with empty maps/indices");
  bool srcLocal = bool(root(src)), dstLocal = bool(root(dst));
  if ((srcLocal || dstLocal) && failed(checkAttributes(copy))) return failure();
  if (!srcLocal && !dstLocal) {
    auto source = storageView(src, copy), destination = storageView(dst, copy);
    if (failed(source) || failed(destination)) return failure();
    copy.getSrcMutable().assign(*source);
    copy.getDstMutable().assign(*destination);
    return success();
  }
  FailureOr<Value> value = failure();
  if (srcLocal) value = read(src, copy, state);
  else {
    auto source = storageView(src, copy);
    if (failed(source)) return failure();
    builder.setInsertionPoint(copy);
    auto load = builder.create<TileLoadOp>(copy.getLoc(), tensorType(src), *source);
    copyMetadata(copy, load);
    value = load.getResult();
  }
  if (failed(value)) return failure();
  if (dstLocal) state[root(dst)] = *value;
  else {
    auto destination = storageView(dst, copy);
    if (failed(destination)) return failure();
    builder.setInsertionPoint(copy);
    auto store = builder.create<TileStoreOp>(copy.getLoc(), *value, *destination);
    copyMetadata(copy, store);
  }
  copy.erase();
  return success();
}

LogicalResult Normalizer::processIndependentStorage(Operation *scope) {
  // Preflight already excluded any promotable Local definition or capture in
  // this region/CFG. Binding storage endpoints at their original operations
  // needs no assumptions about region scheduling or inter-block execution.
  auto result = scope->walk([&](Operation *op) -> WalkResult {
    if (isa<GemmOp, ReduceOp>(op)) {
      (void)reject(op, "math import requires a supported sequential region");
      return WalkResult::interrupt();
    }
    if (auto fill = dyn_cast<FillOp>(op)) {
      auto view = storageView(fill.getMemref(), fill);
      if (failed(view)) return WalkResult::interrupt();
      fill.getMemrefMutable().assign(*view);
    }
    if (auto copy = dyn_cast<CopyOp>(op)) {
      if (root(copy.getSrc()) || root(copy.getDst())) {
        (void)reject(copy);
        return WalkResult::interrupt();
      }
      State noLocalState;
      if (failed(processCopy(copy, noLocalState))) return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult Normalizer::process(Block &block, State &state) {
  for (Operation &operation : llvm::make_early_inc_range(block)) {
    Operation *op = &operation;
    if (isa<AllocBufferOp, memref::AllocOp, memref::AllocaOp>(op) && root(op->getResult(0))) {
      state[op->getResult(0)] = Value();
      eraseLifetimes.push_back(op);
    } else if (auto cast = dyn_cast<memref::CastOp>(op); cast && root(cast.getResult())) {
      eraseLifetimes.push_back(op);
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(op); dealloc && root(dealloc.getMemref())) {
      eraseLifetimes.push_back(op);
    } else if (auto fill = dyn_cast<FillOp>(op)) {
      if (Value allocation = root(fill.getMemref())) {
        if (failed(checkAttributes(fill))) return failure();
        auto value = dyn_cast<FloatAttr>(fill.getValue());
        auto type = tensorType(allocation);
        if (!value || value.getType() != type.getElementType())
          return fill.emitError("legacy-fill-constant: exact same-dtype floating constant required");
        builder.setInsertionPoint(fill);
        auto constant = builder.create<arith::ConstantOp>(fill.getLoc(), type, DenseElementsAttr::get(type, value));
        copyMetadata(fill, constant);
        state[allocation] = constant;
        fill.erase();
      } else {
        auto view = storageView(fill.getMemref(), fill);
        if (failed(view)) return failure();
        fill.getMemrefMutable().assign(*view);
      }
    } else if (auto copy = dyn_cast<CopyOp>(op)) {
      if (failed(processCopy(copy, state))) return failure();
    } else if (isa<GemmOp, ReduceOp>(op)) {
      if (failed(processMath(op, state))) return failure();
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (failed(processIf(branch, state))) return failure();
    } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (failed(processFor(loop, state))) return failure();
    } else if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      if (failed(processWhile(loop, state))) return failure();
    } else if (auto loop = dyn_cast<ForOp>(op)) {
      if (failed(processLegacyFor(loop, state))) return failure();
    } else if (op->getNumRegions()) {
      if (!isSupportedRegion(op)) {
        if (failed(processIndependentStorage(op))) return failure();
        continue;
      }
      State nestedState; // Function/Kernel/Parallel are independent execution scopes.
      for (Region &region : op->getRegions()) {
        if (region.empty()) continue;
        if (!llvm::hasSingleElement(region)) {
          if (failed(processIndependentStorage(op))) return failure();
          continue;
        }
        if (failed(process(region.front(), nestedState))) return failure();
      }
    }
  }
  return success();
}

LogicalResult Normalizer::run() {
  if (failed(preflight())) return failure();
  // Native storage roots must exist before Task 18 alias analysis is queried.
  SmallVector<AllocBufferOp> native;
  module.walk([&](AllocBufferOp op) { if (!isLocal(op.getResult())) native.push_back(op); });
  for (AllocBufferOp op : native) {
    auto type = op.getMemRefType();
    auto space = getFriskMemorySpace(type);
    if (!type.hasStaticShape() || !type.getLayout().isIdentity() || !space ||
        (*space != attr::MemorySpace::Shared && *space != attr::MemorySpace::Global) ||
        op.getAlignment() < 0 || failed(checkAttributes(op)))
      return op.emitError("legacy-storage-root: unsupported native allocation");
    builder.setInsertionPoint(op);
    auto allocation = builder.create<memref::AllocOp>(op.getLoc(), type);
    if (op.getAlignment() > 0) allocation.setAlignmentAttr(builder.getI64IntegerAttr(op.getAlignment()));
    copyMetadata(op, allocation);
    op.getResult().replaceAllUsesWith(allocation);
    op.erase();
  }
  State state;
  if (failed(process(*module.getBody(), state))) return failure();
  // Reverse lexical visitation erases aliases/releases before their roots.
  for (Operation *op : llvm::reverse(eraseLifetimes)) {
    if (!op->use_empty()) return reject(op, "residual Local uses after promotion");
    op->erase();
  }
  return verifyNoLegacyLayoutIR(module);
}
} // namespace

LogicalResult mlir::frisk::normalizeLegacyFragments(ModuleOp module) {
  return Normalizer(module).run();
}

LogicalResult mlir::frisk::verifyNoLegacyLayoutIR(Operation *scope) {
  if (failed(verifyImportAttributes(scope, /*allowLegacyThreads=*/false))) return failure();
  auto result = scope->walk([&](Operation *op) -> WalkResult {
    bool residual = isa<AllocBufferOp, GemmOp, ReduceOp>(op);
    if (isa<CopyOp, FillOp>(op))
      for (Value value : op->getOperands()) residual |= isLocal(value);
    if (!residual) return WalkResult::advance();
    op->emitError("legacy layout IR requires frisk-normalize-layout-ir or frisk-layout-pipeline");
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}
