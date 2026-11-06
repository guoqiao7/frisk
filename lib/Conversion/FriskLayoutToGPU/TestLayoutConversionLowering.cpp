#include "Conversion/FriskLayoutToGPU/LayoutConversionPlan.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_TESTLOWERLAYOUTCONVERSIONS
#include "Dialect/Frisk/Transforms/Passes.h.inc"
namespace {
using namespace mlir::frisk::test;

// The finite lookup tables are the exact planner output, not a second layout
// interpretation. This is deliberately a test adapter, not optimized codegen.
Value lookup(OpBuilder &b, Location loc, Value thread, ArrayRef<uint64_t> values) {
  if (llvm::all_equal(values))
    return b.create<arith::ConstantIntOp>(loc, values.front(), 32);
  // Recover a GF(2) affine expression only after exhaustively checking the
  // finite planner table. This removes dynamic constant-vector stack lookups.
  bool linear = llvm::isPowerOf2_64(values.size());
  for (unsigned t = 0; linear && t < values.size(); ++t) {
    uint64_t expected = values.front();
    for (unsigned bit = 0; (1u << bit) < values.size(); ++bit)
      if (t & (1u << bit)) expected ^= values[1u << bit] ^ values.front();
    linear &= values[t] == expected;
  }
  if (linear) {
    Value tid = b.create<arith::IndexCastOp>(loc, b.getI32Type(), thread);
    if (llvm::all_of(llvm::enumerate(values), [](auto entry) { return entry.index() == entry.value(); })) return tid;
    Value result = b.create<arith::ConstantIntOp>(loc, values.front(), 32);
    Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
    for (unsigned bit = 0; (1u << bit) < values.size(); ++bit) {
      auto mask = values[1u << bit] ^ values.front();
      if (!mask) continue;
      Value shift = b.create<arith::ConstantIntOp>(loc, bit, 32);
      Value selected = b.create<arith::ShRUIOp>(loc, tid, shift);
      selected = b.create<arith::AndIOp>(loc, selected, one);
      Value basis = b.create<arith::ConstantIntOp>(loc, mask, 32);
      selected = b.create<arith::MulIOp>(loc, selected, basis);
      result = b.create<arith::XOrIOp>(loc, result, selected);
    }
    return result;
  }
  auto ty = VectorType::get({int64_t(values.size())}, b.getI32Type());
  SmallVector<APInt> bits;
  for (uint64_t value : values) bits.emplace_back(32, value);
  Value table = b.create<arith::ConstantOp>(loc, ty, DenseIntElementsAttr::get(ty, bits));
  return b.create<vector::ExtractElementOp>(loc, table, thread);
}
Value asIndex(OpBuilder &b, Location loc, Value value) {
  return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value);
}

SmallVector<Value> encodeWords(OpBuilder &b, Location loc, Value value,
                               const PayloadTransportPlan &p) {
  auto bitsType = b.getIntegerType(p.bits);
  if (value.getType() != bitsType)
    value = b.create<arith::BitcastOp>(loc, bitsType, value);
  SmallVector<Value> words;
  for (auto word : p.words) {
    Value part = value;
    if (word.bitOffset) {
      Value shift = b.create<arith::ConstantIntOp>(loc, word.bitOffset, p.bits);
      part = b.create<arith::ShRUIOp>(loc, part, shift);
    }
    if (p.bits < 32) part = b.create<arith::ExtUIOp>(loc, b.getI32Type(), part);
    else if (p.bits > 32) part = b.create<arith::TruncIOp>(loc, b.getI32Type(), part);
    words.push_back(part);
  }
  return words;
}
Value decodeWords(OpBuilder &b, Location loc, ArrayRef<Value> words,
                    Type type, const PayloadTransportPlan &p) {
  auto bitsType = b.getIntegerType(p.bits);
  Value result;
  for (auto [word, input] : llvm::zip_equal(p.words, words)) {
    Value value = input;
    if (p.bits < 32) value = b.create<arith::TruncIOp>(loc, bitsType, value);
    else if (p.bits > 32) value = b.create<arith::ExtUIOp>(loc, bitsType, value);
    if (word.bitOffset) {
      Value shift = b.create<arith::ConstantIntOp>(loc, word.bitOffset, p.bits);
      value = b.create<arith::ShLIOp>(loc, value, shift);
    }
    result = result ? b.create<arith::OrIOp>(loc, result, value) : value;
  }
  if (result.getType() != type) result = b.create<arith::BitcastOp>(loc, type, result);
  return result;
}

Value emitPlan(ConvertLayoutOp op, gpu::GPUFuncOp function,
              const RedistributionPlan &p, Value input = {}) {
  OpBuilder b(op);
  Location loc = op.getLoc();
  auto sourceType = cast<RankedTensorType>(op.getSource().getType());
  Type element = sourceType.getElementType();
  auto sourceVector = VectorType::get({p.sourceRegisters}, element);
  auto resultVector = VectorType::get({p.destinationRegisters}, element);
  // Legacy structural fixtures retain an explicitly non-executable bridge.
  // Executable mode supplies a vector emitted by the real tile-load ABI.
  Value source = input ? input : b.create<UnrealizedConversionCastOp>(loc, sourceVector, op.getSource()).getResult(0);
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  SmallVector<Value> registers;
  for (unsigned r = 0; r < p.sourceRegisters; ++r) {
    Value index = b.create<arith::ConstantIndexOp>(loc, r);
    registers.push_back(b.create<vector::ExtractElementOp>(loc, source, index));
  }
  Value result = b.create<arith::ConstantOp>(loc, resultVector, b.getZeroAttr(resultVector));
  Value scratch;
  if (p.mode == ExchangeMode::Shared) {
    auto space = gpu::AddressSpaceAttr::get(b.getContext(), gpu::AddressSpace::Workgroup);
    auto ty = MemRefType::get({int64_t(p.volume)}, element, MemRefLayoutAttrInterface{}, space);
    unsigned attribution = function.getNumWorkgroupAttributions();
    scratch = function.addWorkgroupAttribution(ty, loc);
    // GPUCommon lowering forwards llvm.align to the workgroup LLVM global.
    function.setworkgroupAttributionAttrs(attribution,
        b.getDictionaryAttr({b.getNamedAttr("llvm.align", b.getI64IntegerAttr(p.payload.bits / 8))}));
    for (unsigned r = 0; r < p.sourceRegisters; ++r) {
      SmallVector<uint64_t> slots, elected;
      for (unsigned t = 0; t < p.threads; ++t) {
        auto slot = p.sourceSlots[t * p.sourceRegisters + r];
        slots.push_back(slot);
        elected.push_back(p.owner(t, r) == p.globalWriters[slot]);
      }
      Value slot = asIndex(b, loc, lookup(b, loc, thread, slots));
      Value writer = lookup(b, loc, thread, elected);
      Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
      Value predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, writer, zero);
      b.create<scf::IfOp>(loc, predicate, [&](OpBuilder &nested, Location l) {
        nested.create<memref::StoreOp>(l, registers[r], scratch, ValueRange{slot});
        nested.create<scf::YieldOp>(l);
      });
    }
    b.create<gpu::BarrierOp>(loc);
  }
  SmallVector<SmallVector<Value>> payloads;
  if (p.mode == ExchangeMode::Shuffle)
    for (Value reg : registers) payloads.push_back(encodeWords(b, loc, reg, p.payload));
  for (unsigned r = 0; r < p.destinationRegisters; ++r) {
    SmallVector<uint64_t> slots, sourceRegs, lanes;
    llvm::SmallSet<unsigned, 8> possibleRegs;
    for (unsigned t = 0; t < p.threads; ++t) {
      auto d = p.destinations[t * p.destinationRegisters + r];
      slots.push_back(d.slot); sourceRegs.push_back(d.source.reg); lanes.push_back(d.source.lane);
      possibleRegs.insert(d.source.reg);
    }
    Value value;
    if (scratch) {
      Value slot = asIndex(b, loc, lookup(b, loc, thread, slots));
      value = b.create<memref::LoadOp>(loc, scratch, ValueRange{slot});
    } else {
      Value sourceReg = lookup(b, loc, thread, sourceRegs);
      if (p.mode == ExchangeMode::Register) {
        value = b.create<vector::ExtractElementOp>(loc, source, asIndex(b, loc, sourceReg));
      } else {
        Value lane = lookup(b, loc, thread, lanes);
        Value width = b.create<arith::ConstantIntOp>(loc, 32, 32);
        SmallVector<Value> selected(p.payload.words.size());
        // All lanes execute every candidate register shuffle. Selection occurs
        // only afterwards and is based on the destination lane's request.
        for (unsigned k : possibleRegs) {
          SmallVector<Value> shuffled;
          for (Value word : payloads[k])
            shuffled.push_back(b.create<gpu::ShuffleOp>(loc, word, lane, width,
                gpu::ShuffleMode::IDX).getShuffleResult());
          Value constant = b.create<arith::ConstantIntOp>(loc, k, 32);
          Value choose = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, sourceReg, constant);
          for (unsigned w = 0; w < selected.size(); ++w)
            selected[w] = selected[w] ? b.create<arith::SelectOp>(loc, choose, shuffled[w], selected[w]) : shuffled[w];
        }
        value = decodeWords(b, loc, selected, element, p.payload);
      }
    }
    Value index = b.create<arith::ConstantIndexOp>(loc, r);
    result = b.create<vector::InsertElementOp>(loc, value, result, index);
  }
  if (!input) {
    Value tensor = b.create<UnrealizedConversionCastOp>(loc, op.getResult().getType(), result).getResult(0);
    op.getResult().replaceAllUsesWith(tensor);
    op.erase();
  }
  return result;
}

// The executable ABI is deliberately narrow: static rank-one global tiles.
// Row-major storage is proven structurally, never inferred from a pointer type.
bool supportedStorageEncoding(StorageLayoutAttr layout, MemRefType type) {
  auto affine = dyn_cast<AffineLayoutMapAttr>(layout.getMap());
  if (!affine) return false;
  auto context = type.getContext();
  auto expected = AffineMap::get(1, 0,
      {getAffineDimExpr(0, context) * (type.getElementTypeBitWidth() / 8),
       getAffineConstantExpr(0, context)}, context);
  return affine.getAffineMap().getValue() == expected;
}
bool supportedStorage(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
      !type.getLayout().isIdentity() || !type.getElementType().isIntOrFloat()) return false;
  auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
  if (type.getMemorySpace() && (!space || space.getInt() != 1)) return false;
  if (auto view = value.getDefiningOp<LayoutViewOp>()) {
    auto layout = view.getLayoutAttr();
    if (!layout) return false;
    return supportedStorageEncoding(layout, type) && supportedStorage(view.getSource());
  }
  return isa<BlockArgument>(value);
}

Value emitTileLoad(TileLoadOp op, const RedistributionPlan &plan) {
  OpBuilder b(op);
  auto loc = op.getLoc();
  auto type = cast<RankedTensorType>(op.getResult().getType());
  auto vectorType = VectorType::get({plan.sourceRegisters}, type.getElementType());
  Value result = b.create<arith::ConstantOp>(loc, vectorType, b.getZeroAttr(vectorType));
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  for (unsigned r = 0; r < plan.sourceRegisters; ++r) {
    SmallVector<uint64_t> slots;
    for (unsigned t = 0; t < plan.threads; ++t) slots.push_back(plan.sourceSlots[t * plan.sourceRegisters + r]);
    Value slot = asIndex(b, loc, lookup(b, loc, thread, slots));
    Value value = b.create<memref::LoadOp>(loc, op.getSource(), ValueRange{slot});
    Value index = b.create<arith::ConstantIndexOp>(loc, r);
    result = b.create<vector::InsertElementOp>(loc, value, result, index);
  }
  return result;
}
void emitTileStore(TileStoreOp op, const RedistributionPlan &plan, Value input) {
  OpBuilder b(op);
  auto loc = op.getLoc();
  Value thread = b.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  for (unsigned r = 0; r < plan.sourceRegisters; ++r) {
    SmallVector<uint64_t> slots, elected;
    for (unsigned t = 0; t < plan.threads; ++t) {
      auto slot = plan.sourceSlots[t * plan.sourceRegisters + r];
      slots.push_back(slot);
      elected.push_back(plan.owner(t, r) == plan.globalWriters[slot]);
    }
    Value slot = asIndex(b, loc, lookup(b, loc, thread, slots));
    Value index = b.create<arith::ConstantIndexOp>(loc, r);
    Value value = b.create<vector::ExtractElementOp>(loc, input, index);
    if (llvm::all_of(elected, [](auto v) { return v == 1; }))
      b.create<memref::StoreOp>(loc, value, op.getTarget(), ValueRange{slot});
    else {
      Value writer = lookup(b, loc, thread, elected);
      Value zero = b.create<arith::ConstantIntOp>(loc, 0, 32);
      Value predicate = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, writer, zero);
      b.create<scf::IfOp>(loc, predicate, [&](OpBuilder &nested, Location l) {
        nested.create<memref::StoreOp>(l, value, op.getTarget(), ValueRange{slot});
        nested.create<scf::YieldOp>(l);
      });
    }
  }
}

class TestLowerLayoutConversionsPass final
    : public impl::TestLowerLayoutConversionsBase<TestLowerLayoutConversionsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }
  void runOnOperation() override {
    struct Pending { ConvertLayoutOp op; gpu::GPUFuncOp function; RedistributionPlan plan; };
    SmallVector<Pending, 0> pending;
    DenseMap<Operation *, uint64_t> usedScratch;
    WalkResult walked = getOperation().walk([&](ConvertLayoutOp op) {
      auto fail = [&](StringRef reason) {
        op.emitError("test lowering requires a static single-CTA redistribution: ") << reason;
        return WalkResult::interrupt();
      };
      auto function = op->getParentOfType<gpu::GPUFuncOp>();
      if (!function || !function.isKernel() || !llvm::hasSingleElement(function.getBody()) ||
          op->getBlock() != &function.getBody().front())
        return fail("requires uniform entry-block execution in a single-block gpu.func kernel");
      auto plan = planRedistribution(cast<RankedTensorType>(op.getSource().getType()),
                                     cast<RankedTensorType>(op.getResult().getType()), op.getLoc());
      if (failed(plan)) return WalkResult::interrupt();
      auto block = function.getKnownBlockSize();
      if (!block || block->size() != 3 || (*block)[0] != int32_t(plan->threads) ||
          (*block)[1] != 1 || (*block)[2] != 1)
        return fail("requires known_block_size matching topology");
      if (plan->mode == ExchangeMode::Shared) {
        auto [it, inserted] = usedScratch.try_emplace(function, 0);
        if (inserted) {
          auto workgroupType = [](Type type) {
            auto memref = dyn_cast<BaseMemRefType>(type);
            if (!memref) return false;
            Attribute space = memref.getMemorySpace();
            if (auto gpuSpace = dyn_cast_or_null<gpu::AddressSpaceAttr>(space))
              return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
            auto integer = dyn_cast_or_null<IntegerAttr>(space);
            return integer && integer.getInt() == 3;
          };
          // Dynamic/shared allocations and incoming aliases are not bounded by
          // the attribution accounting below. Refuse rather than undercount.
          bool unaccounted = llvm::any_of(function.getArgumentTypes(), workgroupType);
          function.walk([&](Operation *nested) {
            unaccounted |= llvm::any_of(nested->getResultTypes(), workgroupType);
          });
          if (unaccounted) return fail("cannot account for non-attributed workgroup storage");
          for (auto [i, arg] : llvm::enumerate(function.getWorkgroupAttributions())) {
            auto ty = dyn_cast<MemRefType>(arg.getType());
            if (!ty || !ty.hasStaticShape() || !ty.getLayout().isIdentity())
              return fail("cannot account for existing workgroup attribution footprint");
            auto payload = planPayload(ty.getElementType(), op.getLoc());
            if (failed(payload)) return WalkResult::interrupt();
            uint64_t bytes = payload->bits / 8;
            for (int64_t extent : ty.getShape()) {
              if (extent < 0 || (extent && bytes > scratchBudget / uint64_t(extent)))
                return fail("existing workgroup attribution exceeds adapter scratch budget");
              bytes *= extent;
            }
            uint64_t alignment = payload->bits / 8;
            if (auto attr = function.getWorkgroupAttributionAttr(i, bString("llvm.align"))) {
              auto integer = dyn_cast<IntegerAttr>(attr);
              if (!integer || integer.getInt() <= 0) return fail("invalid existing workgroup alignment");
              alignment = std::max(alignment, uint64_t(integer.getInt()));
            }
            auto total = reserveScratch(it->second, bytes, alignment, scratchBudget, op.getLoc());
            if (failed(total)) return WalkResult::interrupt();
            it->second = *total;
          }
        }
        auto total = reserveScratch(it->second, plan->scratchBytes, plan->payload.bits / 8,
                                     scratchBudget, op.getLoc());
        if (failed(total)) return WalkResult::interrupt();
        it->second = *total;
      }
      pending.push_back({op, function, std::move(*plan)});
      return WalkResult::advance();
    });
    if (walked.wasInterrupted()) { signalPassFailure(); return; }
    if (executable) {
      DenseMap<Operation *, RedistributionPlan> plans;
      for (auto &item : pending) plans[item.op] = item.plan;
      SmallVector<Operation *> sequence, declarations;
      bool invalid = false;
      getOperation().walk([&](gpu::GPUFuncOp function) {
        auto reject = [&](Operation *op, StringRef reason) { op->emitError(reason); invalid = true; };
        if (!function.isKernel() || !llvm::hasSingleElement(function.getBody())) {
          reject(function, "executable adapter requires a single-block kernel"); return;
        }
        for (BlockArgument arg : function.getArguments())
          if (!supportedStorage(arg)) reject(function, "executable ABI requires static global rank-one memrefs");
        for (Operation &op : function.getBody().front()) {
          if (isa<gpu::ReturnOp>(op)) continue;
          if (auto view = dyn_cast<LayoutViewOp>(op)) {
            if (!supportedStorage(view.getResult())) reject(&op, "executable adapter requires proven row-major storage");
            declarations.push_back(&op); continue;
          }
          if (auto contract = dyn_cast<StorageContractOp>(op)) {
            if (!supportedStorage(contract.getRoot()) ||
                !supportedStorageEncoding(contract.getLayout(), cast<MemRefType>(contract.getRoot().getType())))
              reject(&op, "executable adapter requires a proven row-major root contract");
            declarations.push_back(&op); continue;
          }
          RankedTensorType type;
          if (auto load = dyn_cast<TileLoadOp>(op)) {
            if (!supportedStorage(load.getSource())) reject(&op, "unsupported tile-load storage");
            type = cast<RankedTensorType>(load.getResult().getType());
          } else if (auto store = dyn_cast<TileStoreOp>(op)) {
            if (!supportedStorage(store.getTarget())) reject(&op, "unsupported tile-store storage");
            type = cast<RankedTensorType>(store.getValue().getType());
          } else if (!isa<ConvertLayoutOp>(op)) {
            reject(&op, "unsupported operation in executable conversion adapter"); continue;
          }
          if (type) {
            auto plan = planRedistribution(type, type, op.getLoc());
            if (failed(plan)) { invalid = true; continue; }
            auto block = function.getKnownBlockSize();
            if (!block || block->size() != 3 || (*block)[0] != int32_t(plan->threads) ||
                (*block)[1] != 1 || (*block)[2] != 1) {
              reject(&op, "tile ABI requires known_block_size matching topology"); continue;
            }
            plans[&op] = std::move(*plan);
          }
          for (Value operand : op.getOperands())
            if (isa<RankedTensorType>(operand.getType()) &&
                !isa_and_nonnull<TileLoadOp, ConvertLayoutOp>(operand.getDefiningOp()))
              reject(&op, "tile vector must originate from a supported load or conversion");
          for (Value result : op.getResults())
            for (Operation *user : result.getUsers())
              if (!isa<TileStoreOp, ConvertLayoutOp>(user))
                reject(user, "unsupported executable tile consumer");
          sequence.push_back(&op);
        }
      });
      if (invalid) { signalPassFailure(); return; }
      DenseMap<Value, Value> vectors;
      for (Operation *op : sequence) {
        if (auto load = dyn_cast<TileLoadOp>(op)) vectors[load.getResult()] = emitTileLoad(load, plans[op]);
        else if (auto convert = dyn_cast<ConvertLayoutOp>(op))
          vectors[convert.getResult()] = emitPlan(convert, convert->getParentOfType<gpu::GPUFuncOp>(),
                                                  plans[op], vectors.lookup(convert.getSource()));
        else if (auto store = dyn_cast<TileStoreOp>(op)) emitTileStore(store, plans[op], vectors.lookup(store.getValue()));
      }
      for (auto *op : llvm::reverse(sequence)) op->erase();
      for (auto *op : llvm::reverse(declarations)) {
        if (isa<StorageContractOp>(op)) { op->erase(); continue; }
        auto view = cast<LayoutViewOp>(op);
        view.getResult().replaceAllUsesWith(view.getSource());
        view.erase();
      }
      return;
    }
    // No mutations until every conversion and aggregate scratch budget passes.
    for (auto &item : pending) emitPlan(item.op, item.function, item.plan);
  }
  StringAttr bString(StringRef name) { return StringAttr::get(&getContext(), name); }
};
} // namespace
std::unique_ptr<Pass> createTestLowerLayoutConversionsPass() {
  return std::make_unique<TestLowerLayoutConversionsPass>();
}
} // namespace mlir::frisk
