#ifndef FRISK_SELECTION_ARCHIVE_SOURCE_GRAMMAR_H
#define FRISK_SELECTION_ARCHIVE_SOURCE_GRAMMAR_H
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <string>
namespace mlir::frisk::selection {
struct ArchiveSourceGrammarResult {
  bool eligible = false;
  llvm::json::Object evidence;
};
namespace archive_source_grammar_detail {
inline constexpr const char *grammar =
    "static-rank2-f32-singleCTA128-Global-rowmajor-pure-load-mul-sub-exp-splat-view-convert-finalstore-v1";
inline bool powerOfTwo(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }
inline unsigned bits(int64_t n) {
  unsigned result = 0;
  while (n > 1) { n >>= 1; ++result; }
  return result;
}
inline bool shape(Type type, SmallVectorImpl<int64_t> *result = nullptr) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasRank() || shaped.getRank() != 2 ||
      !shaped.hasStaticShape() || !shaped.getElementType().isF32())
    return false;
  auto dims = shaped.getShape();
  if (!powerOfTwo(dims[0]) || !powerOfTwo(dims[1]) ||
      dims[0] > 65536 || dims[1] > 65536 || dims[0] > 65536 / dims[1])
    return false;
  if (result) result->assign(dims.begin(), dims.end());
  return true;
}
inline bool names(ArrayAttr value, ArrayRef<StringRef> expected) {
  if (value.size() != expected.size()) return false;
  for (unsigned i = 0; i < value.size(); ++i) {
    auto name = dyn_cast<StringAttr>(value[i]);
    if (!name || name.getValue() != expected[i]) return false;
  }
  return true;
}
inline bool memoryType(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  auto space = memref ? getFriskMemorySpace(memref) : std::nullopt;
  return memref && shape(type) && memref.getLayout().isIdentity() &&
         space && *space == attr::MemorySpace::Global;
}
inline bool storage(Attribute value, Type type) {
  auto layout = dyn_cast<StorageLayoutAttr>(value);
  SmallVector<int64_t> dims;
  if (!layout || !memoryType(type) || !shape(type, &dims) ||
      layout.getMemorySpace().getValue() != attr::MemorySpace::Global ||
      layout.getAlignment().getInt() != 1 ||
      layout.getVectorGranularity().getInt() != 1)
    return false;
  auto map = dyn_cast<AffineLayoutMapAttr>(layout.getMap());
  if (!map || !names(map.getInputNames(), {"dim0", "dim1"}) ||
      !names(map.getOutputNames(), {"byte_offset", "bit_offset"}) ||
      map.getInputExtents().asArrayRef() != ArrayRef<int64_t>(dims) ||
      map.getOutputExtents().asArrayRef() !=
          ArrayRef<int64_t>({dims[0] * dims[1] * 4, 8}))
    return false;
  auto *ctx = type.getContext();
  auto d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
  auto expected = AffineMap::get(2, 0, {(d0 * dims[1] + d1) * 4,
                                       getAffineConstantExpr(0, ctx)}, ctx);
  return map.getAffineMap().getValue() == expected;
}
inline bool distributed(Attribute value, Type type) {
  auto layout = dyn_cast<DistributedEncodingAttr>(value);
  SmallVector<int64_t> dims;
  if (!layout || !isa<RankedTensorType>(type) || !shape(type, &dims))
    return false;
  auto topology = layout.getTopology().asArrayRef();
  if (topology.size() != 5 || !powerOfTwo(topology[0]) ||
      topology[0] > 65536 || topology[1] != 32 || topology[2] != 4 ||
      topology[3] != 1 || topology[4] != 1)
    return false;
  // No native register-allocation prediction: large replicated carriers are
  // deliberately allowed, including the existing checked demand lowering.
  auto map = dyn_cast<BitLinearLayoutMapAttr>(layout.getMap());
  if (!map || !names(map.getOutputNames(), {"dim0", "dim1"}) ||
      map.getOutputBitWidths().asArrayRef() !=
          ArrayRef<int64_t>({int64_t(bits(dims[0])), int64_t(bits(dims[1]))}))
    return false;
  auto inputNames = map.getInputNames();
  auto widths = map.getInputBitWidths().asArrayRef();
  if (inputNames.size() != widths.size()) return false;
  bool seen[5] = {false, false, false, false, false};
  const StringRef axes[] = {"register", "lane", "warp", "warp_group", "cta"};
  const int64_t expected[] = {int64_t(bits(topology[0])), 5, 2, 0, 0};
  for (unsigned i = 0; i < inputNames.size(); ++i) {
    auto name = dyn_cast<StringAttr>(inputNames[i]);
    if (!name) return false;
    unsigned axis = 0;
    while (axis < 5 && name.getValue() != axes[axis]) ++axis;
    if (axis == 5 || seen[axis] || widths[i] != expected[axis]) return false;
    seen[axis] = true;
  }
  for (unsigned i = 0; i < 5; ++i)
    if (expected[i] && !seen[i]) return false;
  return succeeded(layout.verifyForType(cast<ShapedType>(type), UnknownLoc::get(type.getContext())));
}
inline bool tensorType(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && shape(type) &&
         (!tensor.getEncoding() || distributed(tensor.getEncoding(), type));
}
inline bool onlyAttrs(Operation *op, ArrayRef<StringRef> allowed) {
  for (auto named : op->getAttrs()) {
    auto name = named.getName().getValue();
    bool found = false;
    for (auto expected : allowed) found |= name == expected;
    if (!found) return false;
    if (name == "fastmath") {
      auto flags = dyn_cast<arith::FastMathFlagsAttr>(named.getValue());
      if (!flags || flags.getValue() != arith::FastMathFlags::none) return false;
    }
  }
  return true;
}
inline ModuleOp owner(Operation *op) {
  if (!op) return {};
  if (auto module = dyn_cast<ModuleOp>(op)) return module;
  return op->getParentOfType<ModuleOp>();
}
} // namespace archive_source_grammar_detail

/// Pure, fail-closed source eligibility. This does not prove native compiler
/// CFG/opcode coverage, resource allocation, calibrated ranking, or GPU time.
/// It consumes no search budget and never changes the module or candidate graph.
inline ArchiveSourceGrammarResult
checkArchiveSourceGrammar(ModuleOp module, const LayoutConstraintGraph &graph) {
  using namespace archive_source_grammar_detail;
  uint64_t checkedOps = 0, checkedVars = 0, checkedCandidates = 0;
  uint64_t kernels = 0, parallels = 0;
  int64_t threads = 0;
  std::string target, domain;
  llvm::json::Array grid;
  auto finish = [&](StringRef reason, bool eligible = false) {
    llvm::json::Array copiedGrid;
    for (auto &value : grid) copiedGrid.push_back(value.getAsInteger().value_or(0));
    return ArchiveSourceGrammarResult{eligible, llvm::json::Object{
        {"contract", "scalar-source-grammar-archive-eligibility-v1"},
        {"status", eligible ? "eligible" : "unknown"}, {"reason", reason.str()},
        {"source_eligibility_only", true}, {"native_output_guaranteed", false},
        {"checked_operations", checkedOps}, {"checked_variables", checkedVars},
        {"checked_candidates", checkedCandidates}, {"kernel_count", kernels},
        {"parallel_count", parallels}, {"threads", threads}, {"grid", std::move(copiedGrid)},
        {"source_target", target}, {"execution_domain", domain}, {"grammar", grammar}}};
  };
  if (!module) return finish("missing-source-module");
  if (auto value = module->getAttrOfType<StringAttr>("frisk.target")) target = value.getValue().str();
  if (auto value = module->getAttrOfType<StringAttr>("frisk.execution_domain")) domain = value.getValue().str();
  SmallVector<Operation *> operations;
  module.walk([&](Operation *op) {
    operations.push_back(op);
    kernels += op->getName().getStringRef() == "frisk.kernel";
    parallels += op->getName().getStringRef() == "frisk.parallel";
  });
  if (target != "sm_90a" || domain != "sm90-scalar-f32-dag-v1")
    return finish("source-target-or-execution-domain-outside-grammar");
  if (!module.getRegion().hasOneBlock() ||
      !onlyAttrs(module, {"frisk.target", "frisk.execution_domain"}) ||
      kernels != 1 || parallels != 1 || module.getBody()->getOperations().size() != 1)
    return finish("whole-module-structure-outside-grammar");
  Operation *kernel = &module.getBody()->front();
  if (kernel->getName().getStringRef() != "frisk.kernel" ||
      kernel->getNumRegions() != 1 || !kernel->getRegion(0).hasOneBlock() ||
      kernel->getNumOperands() || kernel->getNumResults() ||
      !onlyAttrs(kernel, {"sym_name", "function_type"}))
    return finish("kernel-structure-outside-grammar");
  auto symbol = kernel->getAttrOfType<StringAttr>("sym_name");
  if (!symbol || symbol.getValue().empty()) return finish("kernel-symbol-required");
  auto function = kernel->getAttrOfType<TypeAttr>("function_type");
  auto functionType = function ? dyn_cast<FunctionType>(function.getValue()) : FunctionType();
  Block &body = kernel->getRegion(0).front();
  if (!functionType || functionType.getNumInputs() != 4 || functionType.getNumResults() ||
      body.getNumArguments() != 4)
    return finish("four-root-kernel-ABI-required");
  SmallVector<int64_t> commonShape;
  if (!shape(body.getArgument(0).getType(), &commonShape))
    return finish("kernel-type-outside-static-rank2-f32-domain");
  llvm::DenseMap<Value, unsigned> roots;
  for (unsigned i = 0; i < 4; ++i) {
    auto type = body.getArgument(i).getType();
    if (!memoryType(type) || type != functionType.getInput(i) ||
        cast<MemRefType>(type).getShape() != ArrayRef<int64_t>(commonShape))
      return finish("kernel-root-type-or-shape-outside-grammar");
    roots[body.getArgument(i)] = i;
  }
  Operation *parallel = nullptr;
  bool contracted[4] = {false, false, false, false};
  for (auto &op : body) {
    auto name = op.getName().getStringRef();
    if (name == "frisk.storage_contract" && !parallel) {
      if (op.getNumRegions() || op.getNumOperands() != 1 || op.getNumResults() ||
          !onlyAttrs(&op, {"layout"}) || !roots.count(op.getOperand(0)) ||
          !storage(op.getAttr("layout"), op.getOperand(0).getType()))
        return finish("kernel-storage-contract-outside-grammar");
      contracted[roots.lookup(op.getOperand(0))] = true;
    } else if (name == "frisk.parallel" && !parallel) {
      parallel = &op;
    } else if (name != "frisk.end" || &op != &body.back() ||
               op.getNumOperands() || op.getNumResults() || op.getNumRegions() ||
               !onlyAttrs(&op, {})) {
      return finish("kernel-body-outside-grammar");
    }
  }
  if (!parallel || !contracted[0] || !contracted[1] || !contracted[2] || !contracted[3])
    return finish("complete-dominating-four-root-contracts-required");
  auto threadAttr = parallel->getAttrOfType<IntegerAttr>("threads");
  auto ranges = parallel->getAttrOfType<DenseI64ArrayAttr>("ranges");
  if (threadAttr) threads = threadAttr.getInt();
  if (ranges) for (auto value : ranges.asArrayRef()) grid.push_back(value);
  if (!onlyAttrs(parallel, {"threads", "ranges"}) || threads != 128 || !ranges ||
      ranges.size() != 1 || ranges.asArrayRef()[0] != 1 ||
      parallel->getNumOperands() || parallel->getNumResults() ||
      parallel->getNumRegions() != 1 || !parallel->getRegion(0).hasOneBlock())
    return finish("singleCTA128-parallel-required");
  Block &block = parallel->getRegion(0).front();
  if (block.getNumArguments() != 1 || !block.getArgument(0).getType().isIndex() ||
      !block.getArgument(0).use_empty())
    return finish("dynamic-CTA-index-use-outside-grammar");
  unsigned stores = 0, loads = 0;
  bool seenStore = false;
  for (auto &op : block) {
    auto name = op.getName().getStringRef();
    if (seenStore && name != "frisk.end") return finish("store-must-be-final");
    if (op.getNumRegions()) return finish("nested-region-outside-grammar");
    if (name == "frisk.end") {
      if (&op != &block.back() || op.getNumOperands() || op.getNumResults() ||
          !onlyAttrs(&op, {})) return finish("invalid-parallel-terminator");
    } else if (name == "frisk.storage_contract") {
      if (op.getNumOperands() != 1 || op.getNumResults() ||
          !onlyAttrs(&op, {"layout"}) || !roots.count(op.getOperand(0)) ||
          !storage(op.getAttr("layout"), op.getOperand(0).getType()))
        return finish("body-storage-contract-outside-grammar");
    } else if (name == "frisk.layout_view") {
      if (op.getNumOperands() != 1 || op.getNumResults() != 1 ||
          !onlyAttrs(&op, {"layout"}) || !roots.count(op.getOperand(0)) ||
          op.getOperand(0).getType() != op.getResult(0).getType() ||
          !storage(op.getAttr("layout"), op.getResult(0).getType()))
        return finish("nonidentity-or-unknown-storage-view");
      roots[op.getResult(0)] = roots.lookup(op.getOperand(0));
    } else if (name == "frisk.tile_load") {
      if (op.getNumOperands() != 1 || op.getNumResults() != 1 ||
          !onlyAttrs(&op, {}) || !roots.count(op.getOperand(0)) ||
          roots.lookup(op.getOperand(0)) == 3 || !tensorType(op.getResult(0).getType()) ||
          cast<RankedTensorType>(op.getResult(0).getType()).getShape() != ArrayRef<int64_t>(commonShape))
        return finish("load-outside-pure-input-grammar");
      ++loads;
    } else if (name == "frisk.tile_store") {
      if (op.getNumOperands() != 2 || op.getNumResults() || !onlyAttrs(&op, {}) ||
          !tensorType(op.getOperand(0).getType()) || !roots.count(op.getOperand(1)) ||
          roots.lookup(op.getOperand(1)) != 3 ||
          cast<RankedTensorType>(op.getOperand(0).getType()).getShape() != ArrayRef<int64_t>(commonShape))
        return finish("store-outside-pure-output-grammar");
      ++stores; seenStore = true;
    } else if (name == "arith.constant") {
      auto value = op.getAttrOfType<DenseElementsAttr>("value");
      if (op.getNumOperands() || op.getNumResults() != 1 ||
          !onlyAttrs(&op, {"value"}) || !tensorType(op.getResult(0).getType()) ||
          !value || !value.isSplat() || value.getType() != op.getResult(0).getType())
        return finish("constant-outside-f32-splat-grammar");
    } else if (name == "arith.mulf" || name == "arith.subf" || name == "math.exp" ||
               name == "frisk.convert_layout") {
      unsigned arity = (name == "arith.mulf" || name == "arith.subf") ? 2 : 1;
      if (op.getNumOperands() != arity || op.getNumResults() != 1 ||
          !onlyAttrs(&op, name == "frisk.convert_layout" ? ArrayRef<StringRef>{} : ArrayRef<StringRef>{"fastmath"}) ||
          !tensorType(op.getResult(0).getType()))
        return finish("pure-scalar-operation-schema-outside-grammar");
      auto result = cast<RankedTensorType>(op.getResult(0).getType());
      if (result.getShape() != ArrayRef<int64_t>(commonShape))
        return finish("scalar-operation-shape-outside-grammar");
      for (auto operand : op.getOperands()) {
        auto type = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorType(operand.getType()) || type.getShape() != result.getShape() ||
            (name != "frisk.convert_layout" && type != result))
          return finish("scalar-operation-type-outside-grammar");
      }
    } else {
      return finish(std::string("operation-outside-grammar:") + name.str());
    }
  }
  if (stores != 1 || loads == 0 || block.empty() ||
      block.back().getName().getStringRef() != "frisk.end" ||
      body.empty() || body.back().getName().getStringRef() != "frisk.end")
    return finish("pure-input-DAG-and-one-final-store-required");
  // Full walk: no extra operation can hide in an additional scope or module.
  for (auto *op : operations) {
    ++checkedOps;
    if (op == module.getOperation() || op == kernel || op == parallel) continue;
    if (op->getParentOp() != kernel && op->getParentOp() != parallel)
      return finish("operation-outside-checked-kernel-and-parallel");
  }
  if (graph.getVariables().empty()) return finish("empty-candidate-domain");
  for (const auto &variable : graph.getVariables()) {
    ++checkedVars;
    if (variable.candidates.empty() || variable.state == LayoutState::Uninitialized ||
        variable.state == LayoutState::Conflict || variable.candidatePreparationUnsupported ||
        variable.instructionRole || variable.reductionResult || variable.operationExecution ||
        (variable.requiredThreads && variable.requiredThreads != 128))
      return finish("unknown-or-nonscalar-candidate-variable");
    SmallVector<int64_t> variableShape;
    if (!shape(variable.shapedType, &variableShape) || variableShape != commonShape ||
        (!variable.anchor && !variable.value && !variable.use))
      return finish("candidate-type-or-source-binding-outside-grammar");
    Value sourceValue = variable.value;
    if ((variable.anchor && owner(variable.anchor) != module) ||
        (sourceValue && (!sourceValue.getParentRegion() ||
            owner(sourceValue.getParentRegion()->getParentOp()) != module)) ||
        (variable.use && owner(variable.use->getOwner()) != module))
      return finish("candidate-domain-not-bound-to-source-module");
    if (variable.kind == LayoutKind::Storage) {
      Value endpoint = variable.value ? variable.value :
                       (variable.use ? variable.use->get() : Value());
      if (!endpoint || !variable.storageAlias)
        return finish("missing-storage-alias-source-proof");
      auto fresh = analyzeStorageAlias(endpoint);
      const auto &alias = *variable.storageAlias;
      auto root = alias.root ? dyn_cast<BlockArgument>(alias.root) : BlockArgument();
      // The checked module has one named kernel and its first/only block.
      // Match the collector's stable root label, without using the symbol name
      // to choose an archive policy. The diagnostic alignment prose is not an
      // address/alignment proof and is extended by collection.
      std::string rootKey;
      if (root) rootKey = std::to_string(symbol.getValue().size()) + ":" +
          symbol.getValue().str() + "/b0/arg" + std::to_string(root.getArgNumber()) + "/storage";
      if (failed(fresh) || !root || root.getOwner() != &body ||
          alias.root != fresh->root || alias.rootType != fresh->rootType ||
          alias.viewType != fresh->viewType || alias.viewToRoot != fresh->viewToRoot ||
          alias.rootKey != rootKey || alias.lowerBit != fresh->lowerBit ||
          alias.upperBit != fresh->upperBit || alias.rootAlignment != fresh->rootAlignment ||
          alias.alignmentEvidence.empty() ||
          alias.rootType != root.getType() || alias.viewType != variable.shapedType ||
          !alias.viewToRoot || !alias.viewToRoot.isIdentity() ||
          alias.lowerBit != 0 || alias.upperBit != uint64_t(commonShape[0] * commonShape[1] * 32) ||
          alias.rootAlignment != 1 || !roots.count(endpoint))
        return finish("nonidentity-or-unbound-storage-alias");
    } else if (variable.storageAlias) {
      return finish("storage-alias-on-nonstorage-variable");
    }
    for (const auto &candidate : variable.candidates) {
      ++checkedCandidates;
      if (variable.kind == LayoutKind::Storage) {
        if (!storage(candidate.value, variable.shapedType) ||
            !matchContiguousIdentityStorageAlias(
                *variable.storageAlias, cast<StorageLayoutAttr>(candidate.value)))
          return finish("storage-candidate-outside-rowmajor-Global-domain");
      } else if (variable.kind == LayoutKind::Distributed) {
        if (!distributed(candidate.value, variable.shapedType))
          return finish("distributed-candidate-outside-checked-GF2-domain");
      } else return finish("instruction-or-unknown-layout-candidate");
    }
  }
  return finish("", true);
}
} // namespace mlir::frisk::selection
#endif
