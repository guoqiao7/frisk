#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
#include <set>

using namespace mlir;
using namespace mlir::frisk;
namespace {
class SM90CopyContractTest : public testing::Test {
protected:
  MLIRContext context;
  std::unique_ptr<LayoutTarget> target = createSM90LayoutTarget();
  SM90CopyContractTest() {
    context.loadDialect<FriskDialect, func::FuncDialect, memref::MemRefDialect>();
  }
  OwningOpRef<ModuleOp> make(int64_t rows = 32, int64_t columns = 64,
                            int64_t sourceAlignment = 128, int64_t destinationAlignment = 1024) {
    std::string text = R"mlir(
      module attributes {frisk.target = "sm_90"} {
        func.func @copy() {
          %s = memref.alloc() {alignment = 128 : i64} : memref<32x64xf32,1>
          %d = memref.alloc() {alignment = 1024 : i64} : memref<32x64xf32,3>
          %sv = frisk.layout_view %s : memref<32x64xf32,1> -> memref<32x64xf32,1>
          %dv = frisk.layout_view %d : memref<32x64xf32,3> -> memref<32x64xf32,3>
          "frisk.copy"(%sv,%dv) <{srcMap=affine_map<()->()>,dstMap=affine_map<()->()>,
            srcExtents=array<i64:32,64>,dstExtents=array<i64:32,64>}> {operandSegmentSizes=array<i32:1,1,0,0>}
            : (memref<32x64xf32,1>,memref<32x64xf32,3>)->()
          return
        }
      })mlir";
    auto replace = [&](StringRef from, std::string to) {
      size_t position = 0;
      while ((position = text.find(from.str(), position)) != std::string::npos) {
        text.replace(position, from.size(), to); position += to.size();
      }
    };
    replace("32x64", std::to_string(rows) + "x" + std::to_string(columns));
    replace("32,64", std::to_string(rows) + "," + std::to_string(columns));
    replace("alignment = 128", "alignment = " + std::to_string(sourceAlignment));
    replace("alignment = 1024", "alignment = " + std::to_string(destinationAlignment));
    return parseSourceString<ModuleOp>(text, &context);
  }
  OwningOpRef<ModuleOp> makeND(ArrayRef<int64_t> shape, Type element,
      unsigned sourceSpace = 1, unsigned destinationSpace = 3,
      int64_t sourceAlignment = 16, int64_t destinationAlignment = 1024,
      bool badStride = false) {
    OpBuilder b(&context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    b.setInsertionPointToEnd(module->getBody());
    auto function = b.create<func::FuncOp>(loc, "copy", b.getFunctionType({}, {}));
    b.setInsertionPointToStart(function.addEntryBlock());
    auto view = [&](unsigned space, int64_t alignment, bool strided) -> Value {
      MemRefLayoutAttrInterface map;
      if (strided) map = StridedLayoutAttr::get(&context, 0, {10, 1});
      auto type = MemRefType::get(shape, element, map, b.getI64IntegerAttr(space));
      auto allocation = b.create<memref::AllocOp>(loc, type);
      allocation->setAttr("alignment", b.getI64IntegerAttr(alignment));
      auto endpoint = b.create<LayoutViewOp>(loc, type, allocation, StorageLayoutAttr());
      auto info = analyzeStorageAlias(endpoint);
      if (failed(info)) return {};
      auto linear = buildRootLinearStorageCandidate(*info);
      if (failed(linear)) return {};
      endpoint.setLayoutAttr(*linear);
      return endpoint;
    };
    auto src = view(sourceSpace, sourceAlignment, badStride);
    auto dst = view(destinationSpace, destinationAlignment, false);
    if (!src || !dst) return {};
    b.create<CopyOp>(loc, src, dst, ValueRange{}, ValueRange{});
    b.create<func::ReturnOp>(loc);
    return module;
  }
  std::set<std::string> paths(const LayoutConstraintGraph &graph) {
    std::set<std::string> result;
    for (const auto &c : graph.getConstraints()) if (c.instruction)
      for (const auto &tuple : c.instruction->tuples) {
        DenseMap<LayoutVarID, Attribute> assignment;
        for (auto [id, value] : llvm::zip(c.vars, tuple.encodings)) assignment[id] = value;
        // Collection freezes proposals before hard propagation. Report only
        // tuples compatible with the explicitly pinned storage requirements.
        bool legal = llvm::all_of(graph.getConstraints(), [&](const auto &hard) {
          return hard.kind != ConstraintKind::RequireEncoding ||
              satisfiesLayoutConstraint(graph, hard, assignment, true);
        });
        if (legal) result.insert(cast<DictionaryAttr>(tuple.binding).getAs<StringAttr>("path").getValue().str());
      }
    return result;
  }
};

TEST_F(SM90CopyContractTest, CopyHasAnExplicitInstructionPathVariable) {
  auto module = make(); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  unsigned paths = 0;
  for (const auto &v : graph->getVariables()) paths += v.kind == LayoutKind::Instruction;
  EXPECT_EQ(paths, 1u);
}

TEST_F(SM90CopyContractTest, SelectedContractIsMaterializedAndReverified) {
  auto module = make(); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph)));
  auto solution = solveLayoutGraph(*graph, *target);
  ASSERT_TRUE(succeeded(solution));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  CopyOp copy; module->walk([&](CopyOp op) { copy = op; });
  auto contract = copy->getAttrOfType<DictionaryAttr>("frisk.copy_contract");
  ASSERT_TRUE(contract);
  EXPECT_EQ(contract.getAs<StringAttr>("path").getValue(), "tma");
  EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
  NamedAttrList tampered(contract);
  tampered.set("completion", Builder(&context).getStringAttr("none"));
  copy->setAttr("frisk.copy_contract", tampered.getDictionary(&context));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(SM90CopyContractTest, SmallCopyPrefersVectorButRetainsAsyncAlternatives) {
  auto module = make(8, 16); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  std::set<std::string> paths;
  for (const auto &c : graph->getConstraints()) if (c.instruction)
    for (const auto &tuple : c.instruction->tuples)
      paths.insert(cast<DictionaryAttr>(tuple.binding).getAs<StringAttr>("path").getValue().str());
  EXPECT_EQ(paths, (std::set<std::string>{"vector", "cp.async", "tma"}));
  auto solution = solveLayoutGraph(*graph, *target); ASSERT_TRUE(succeeded(solution));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  module->walk([&](CopyOp copy) {
    EXPECT_EQ(copy->getAttrOfType<DictionaryAttr>("frisk.copy_contract").getAs<StringAttr>("path").getValue(), "vector");
  });
}

TEST_F(SM90CopyContractTest, UnprovedAlignmentKeepsCorrectFallback) {
  auto module = make(8, 16, 1, 1); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction)
    for (const auto &tuple : c.instruction->tuples)
      EXPECT_EQ(cast<DictionaryAttr>(tuple.binding).getAs<StringAttr>("path").getValue(), "vector");
  auto solution = solveLayoutGraph(*graph, *target); ASSERT_TRUE(succeeded(solution));
  EXPECT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
}

TEST_F(SM90CopyContractTest, SharedCandidatesContainCanonicalTMASwizzles) {
  for (int64_t width : {32, 64, 128}) {
    auto type = MemRefType::get({8, width / 4}, Float32Type::get(&context), MemRefLayoutAttrInterface{},
                                IntegerAttr::get(IntegerType::get(&context, 64), 3));
    LayoutVar var; var.kind = LayoutKind::Storage; var.shapedType = type;
    SmallVector<LayoutCandidate> candidates;
    target->enumerateCandidates(var, candidates);
    bool found = false;
    for (const auto &candidate : candidates) {
      auto storage = cast<StorageLayoutAttr>(candidate.value);
      auto map = dyn_cast<BitLinearLayoutMapAttr>(storage.getMap());
      if (!map) continue;
      auto matrix = map.getMatrixValue();
      ASSERT_TRUE(succeeded(matrix));
      bool matches = true;
      for (int64_t row = 0; row < 8; ++row)
        for (int64_t col = 0; col < width / 4; ++col) {
          auto address = matrix->apply(APInt(matrix->getNumColumns(), row | (col << 3))).getZExtValue();
          uint64_t linear = row * width + col * 4;
          uint64_t expected = linear ^ (((linear >> 7) & (width / 16 - 1)) << 4);
          matches &= address == expected;
        }
      found |= matches;
    }
    EXPECT_TRUE(found) << "missing canonical TMA swizzle " << width;
  }
}

TEST_F(SM90CopyContractTest, DescriptorTuplesProveAllThreeSwizzles) {
  for (int64_t swizzle : {32, 64, 128}) {
    auto module = make(8, swizzle / 4); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    bool found = false;
    for (const auto &c : graph->getConstraints()) if (c.instruction)
      for (const auto &tuple : c.instruction->tuples) {
        auto binding = cast<DictionaryAttr>(tuple.binding);
        if (binding.getAs<StringAttr>("path").getValue() != "tma") continue;
        auto descriptor = binding.getAs<DictionaryAttr>("descriptor");
        found |= descriptor.getAs<IntegerAttr>("swizzle_bytes").getInt() == swizzle;
      }
    EXPECT_TRUE(found) << swizzle;
  }
}

TEST_F(SM90CopyContractTest, GuardedDomainCannotBeForged) {
  auto module = make(3, 5); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  auto solution = solveLayoutGraph(*graph, *target); ASSERT_TRUE(succeeded(solution));
  ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
  CopyOp copy; module->walk([&](CopyOp op) { copy = op; });
  auto contract = copy->getAttrOfType<DictionaryAttr>("frisk.copy_contract");
  EXPECT_TRUE(contract.getAs<BoolAttr>("guarded").getValue());
  NamedAttrList tampered(contract);
  tampered.set("valid_shape", Builder(&context).getDenseI64ArrayAttr({4, 8}));
  copy->setAttr("frisk.copy_contract", tampered.getDictionary(&context));
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verifyMaterializedLayouts(*module, *target)));
}

TEST_F(SM90CopyContractTest, TmaRanksAndDtypesUseProvenLinearDescriptors) {
  for (unsigned rank = 1; rank <= 5; ++rank) {
    SmallVector<int64_t> shape(rank, 2); shape.back() = 8;
    for (Type element : {Type(Float16Type::get(&context)), Type(BFloat16Type::get(&context)), Type(Float32Type::get(&context))}) {
      auto module = makeND(shape, element); ASSERT_TRUE(module);
      auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
      EXPECT_EQ(paths(*graph).count("tma"), 1u) << rank;
    }
  }
}

TEST_F(SM90CopyContractTest, UnsupportedTmaCasesRetainFallback) {
  for (auto shape : {SmallVector<int64_t>{512, 4}, {2, 2, 2, 2, 2, 4}, {8, 2}}) {
    auto module = makeND(shape, Float32Type::get(&context)); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    auto candidates = paths(*graph);
    EXPECT_EQ(candidates.count("tma"), 0u);
    EXPECT_EQ(candidates.count("vector"), 1u);
  }
  for (auto spaces : {std::pair<unsigned,unsigned>{3, 1}, {3, 3}, {1, 1}}) {
    auto module = makeND({8,8}, Float32Type::get(&context), spaces.first, spaces.second); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    EXPECT_EQ(paths(*graph), (std::set<std::string>{"vector"}));
  }
  auto module = makeND({8,8}, Float32Type::get(&context), 1, 3, 16, 1024, true); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  EXPECT_EQ(paths(*graph).count("tma"), 0u); // 40-byte global row stride.
  EXPECT_EQ(paths(*graph).count("vector"), 1u);
  auto wide = makeND({8,8}, Float64Type::get(&context)); ASSERT_TRUE(wide);
  auto wideGraph = collectLayoutConstraints(*wide, *target); ASSERT_TRUE(succeeded(wideGraph));
  EXPECT_EQ(paths(*wideGraph).count("tma"), 0u); // Not in the admitted dtype table.
}

TEST_F(SM90CopyContractTest, AsyncWidthsRespectActualAlignmentAndHardMinimum) {
  for (int64_t width : {4, 8, 16}) {
    auto module = makeND({8,16}, Float32Type::get(&context), 1, 3, width, width); ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    bool exact = false;
    for (const auto &c : graph->getConstraints()) if (c.instruction)
      for (const auto &tuple : c.instruction->tuples) {
        auto binding = cast<DictionaryAttr>(tuple.binding);
        if (binding.getAs<StringAttr>("path").getValue() != "cp.async") continue;
        int64_t selected = binding.getAs<IntegerAttr>("vector_bytes").getInt();
        EXPECT_GE(selected, 4); EXPECT_LE(selected, width); exact |= selected == width;
      }
    EXPECT_TRUE(exact) << width;
  }
  auto module = makeND({8,16}, Float32Type::get(&context), 1, 3, 4, 4); ASSERT_TRUE(module);
  module->walk([&](CopyOp op) { op->setAttr("frisk.vector_bytes", Builder(&context).getI64IntegerAttr(16)); });
  ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(collectLayoutConstraints(*module, *target)));
}

TEST_F(SM90CopyContractTest, DescriptorOffsetTamperingIsRejected) {
  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  bool checked = false;
  for (const auto &c : graph->getConstraints()) if (c.instruction)
    for (const auto &tuple : c.instruction->tuples) {
      auto binding = cast<DictionaryAttr>(tuple.binding);
      if (binding.getAs<StringAttr>("path").getValue() != "tma") continue;
      NamedAttrList descriptor(binding.getAs<DictionaryAttr>("descriptor"));
      descriptor.set("source_offset", Builder(&context).getI64IntegerAttr(16));
      NamedAttrList altered(binding); altered.set("descriptor", descriptor.getDictionary(&context));
      EXPECT_EQ(target->verifyInstructionContract(*graph, c, tuple.encodings, altered.getDictionary(&context)).status,
                ProofStatus::Disproven);
      checked = true;
    }
  EXPECT_TRUE(checked);
}
} // namespace
