#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Target/SM90/SM90CopyConstraints.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutTrace.h"
#include "Dialect/Frisk/Analysis/LayoutAliasAnalysis.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
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
      bool badStride = false, int64_t sourceOffset = 0,
      int64_t sourceColumnStride = 1) {
    OpBuilder b(&context);
    auto loc = b.getUnknownLoc();
    OwningOpRef<ModuleOp> module(ModuleOp::create(loc));
    b.setInsertionPointToEnd(module->getBody());
    auto function = b.create<func::FuncOp>(loc, "copy", b.getFunctionType({}, {}));
    b.setInsertionPointToStart(function.addEntryBlock());
    auto view = [&](unsigned space, int64_t alignment, bool strided,
                    int64_t offset = 0, int64_t columnStride = 1) -> Value {
      MemRefLayoutAttrInterface map;
      if (strided || offset || columnStride != 1)
        map = StridedLayoutAttr::get(&context, offset,
            {strided ? 10 : shape.back() * columnStride, columnStride});
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
    auto src = view(sourceSpace, sourceAlignment, badStride, sourceOffset, sourceColumnStride);
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
  // Select an actual-address tuple, not an alternative inconsistent with the
  // fixture's hard storage layouts.
  SmallVector<Attribute> actualTuple(const LayoutConstraintGraph &graph,
                                    const LayoutConstraint &c) {
    for (const auto &tuple : c.instruction->tuples) {
      DenseMap<LayoutVarID, Attribute> assignment;
      for (auto [id, value] : llvm::zip(c.vars, tuple.encodings)) assignment[id] = value;
      if (llvm::all_of(graph.getConstraints(), [&](const auto &hard) {
        return hard.kind != ConstraintKind::RequireEncoding ||
            satisfiesLayoutConstraint(graph, hard, assignment, true);
      })) return tuple.encodings;
    }
    return {};
  }
  void expectAtomicFailure(ModuleOp module) {
    std::string before;
    llvm::raw_string_ostream(before) << module;
    PassManager pipeline(&context);
    buildFriskLayoutPipeline(pipeline);
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    EXPECT_TRUE(failed(pipeline.run(module)));
    std::string after;
    llvm::raw_string_ostream(after) << module;
    EXPECT_EQ(before, after);
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

TEST_F(SM90CopyContractTest, RejectedHardAlignmentReportsActualReason) {
  auto module = makeND({8,16}, Float32Type::get(&context), 1, 3, 1, 1024);
  ASSERT_TRUE(module);
  module->walk([&](CopyOp op) {
    op->setAttr("frisk.vector_bytes", Builder(&context).getI64IntegerAttr(16));
  });
  std::string message;
  ScopedDiagnosticHandler capture(&context, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream(message) << diagnostic;
    return success();
  });
  EXPECT_TRUE(failed(collectLayoutConstraints(*module, *target)));
  EXPECT_NE(message.find("copy-alignment"), std::string::npos) << message;
  EXPECT_NE(message.find("operand 0"), std::string::npos) << message;
  EXPECT_NE(message.find("root alignment"), std::string::npos) << message;
}

TEST_F(SM90CopyContractTest, DetailedAlignmentPreservesSourceAndDestinationProofs) {
  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    ASSERT_FALSE(c.instruction->tuples.empty());
    auto encodings = c.instruction->tuples.front().encodings;
    encodings[3] = Builder(&context).getStringAttr("vector");
    graph->getVariable(c.vars[2]).operationExecution->vectorBytes = 16;
    for (unsigned operand : {0u, 1u}) {
      auto &info = *graph->getVariable(c.vars[operand]).storageAlias;
      auto alignment = info.rootAlignment;
      info.rootAlignment = 1;
      auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
      EXPECT_EQ(result.proof.status, ProofStatus::Disproven);
      EXPECT_EQ(result.reasonCode, "copy-alignment");
      EXPECT_EQ(result.operand, operand);
      EXPECT_FALSE(result.binding);
      EXPECT_NE(result.proof.reason.find("root alignment"), std::string::npos);
      info.rootAlignment = alignment;
    }
  }
}

TEST_F(SM90CopyContractTest, DetailedFootprintFailurePreservesUnknownAndCoordinate) {
  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = c.instruction->tuples.front().encodings;
    auto key = LayoutConstraintGraph::AliasCandidateKey{c.vars[0], encodings[0]};
    auto &footprint = graph->getAliasFootprintCache()[key];
    auto original = footprint;
    footprint.proof = {ProofStatus::Unknown, {3,7}, "storage alias proof domain exceeds 65536 live points or is dynamic/empty"};
    auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
    EXPECT_EQ(result.proof.status, ProofStatus::Unknown);
    EXPECT_EQ(result.reasonCode, "copy-budget");
    EXPECT_EQ(result.operand, 0u);
    EXPECT_EQ(result.proof.counterexample, (SmallVector<int64_t>{3,7}));
    EXPECT_EQ(result.proof.reason, footprint.proof.reason);
    EXPECT_FALSE(result.binding);
    footprint = original;
  }
}

TEST_F(SM90CopyContractTest, ActualEnumerationBudgetIsUnknownNotDisproven) {
  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
    auto execution = cast<DistributedEncodingAttr>(encodings[2]);
    auto topology = llvm::to_vector(execution.getTopology().asArrayRef());
    // Omitted carriers are legal replication. This exceeds the actual address
    // enumeration budget even though symbolic ownership may be decidable.
    topology[4] = 65536;
    encodings[2] = DistributedEncodingAttr::get(&context, execution.getMap(),
        Builder(&context).getDenseI64ArrayAttr(topology),
        Builder(&context).getI64IntegerAttr(execution.getReplication().getInt() * 65536));
    ASSERT_TRUE(succeeded(cast<DistributedEncodingAttr>(encodings[2]).verifyForType(
        cast<ShapedType>(graph->getVariable(c.vars[2]).shapedType), module->getLoc())));
    auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
    EXPECT_EQ(result.proof.status, ProofStatus::Unknown);
    EXPECT_EQ(result.reasonCode, "copy-budget");
    EXPECT_EQ(result.operand, 0u);
    EXPECT_NE(result.proof.reason.find("hardware enumeration budget"), std::string::npos);
    EXPECT_FALSE(result.binding);
  }
}

TEST_F(SM90CopyContractTest, DetailedDuplicateWriterKeepsHardwareCoordinate) {
  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
    auto execution = cast<DistributedEncodingAttr>(encodings[2]);
    auto topology = llvm::to_vector(execution.getTopology().asArrayRef());
    topology[4] = 2;
    encodings[2] = DistributedEncodingAttr::get(&context, execution.getMap(),
        Builder(&context).getDenseI64ArrayAttr(topology),
        Builder(&context).getI64IntegerAttr(execution.getReplication().getInt() * 2));
    ASSERT_TRUE(succeeded(cast<DistributedEncodingAttr>(encodings[2]).verifyForType(
        cast<ShapedType>(graph->getVariable(c.vars[2]).shapedType), module->getLoc())));
    graph->getVariable(c.vars[2]).operationExecution->writerPolicy = "all";
    auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
    EXPECT_EQ(result.proof.status, ProofStatus::Disproven);
    EXPECT_EQ(result.reasonCode, "copy-writer");
    EXPECT_EQ(result.proof.counterexample.size(), 5u);
    EXPECT_FALSE(result.binding);
  }
}

TEST_F(SM90CopyContractTest, DetailedDescriptorStrideAndDirectionAreDistinct) {
  auto module = makeND({8,8}, Float32Type::get(&context), 1, 3, 16, 1024, true);
  ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
    encodings[3] = Builder(&context).getStringAttr("tma");
    auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
    EXPECT_EQ(result.proof.status, ProofStatus::Disproven);
    EXPECT_EQ(result.reasonCode, "copy-descriptor");
    EXPECT_EQ(result.operand, 0u);
    EXPECT_NE(result.proof.reason.find("outer stride"), std::string::npos);
  }
  auto reverse = makeND({8,8}, Float32Type::get(&context), 3, 1, 1024, 128);
  ASSERT_TRUE(reverse);
  auto reverseGraph = collectLayoutConstraints(*reverse, *target); ASSERT_TRUE(succeeded(reverseGraph));
  for (const auto &c : reverseGraph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*reverseGraph, c); ASSERT_EQ(encodings.size(), 4u);
    encodings[3] = Builder(&context).getStringAttr("cp.async");
    auto result = buildSM90CopyContractDetailed(*reverseGraph, c, encodings);
    EXPECT_EQ(result.proof.status, ProofStatus::Disproven);
    EXPECT_EQ(result.reasonCode, "copy-direction");
    EXPECT_FALSE(result.binding);
  }
}

TEST_F(SM90CopyContractTest, TmaDestinationMismatchKeepsRepresentativeCoordinate) {
  auto module = makeND({8,8}, Float32Type::get(&context)); ASSERT_TRUE(module);
  Builder b(&context);
  auto address = b.getAffineDimExpr(0)*4+b.getAffineDimExpr(1)*32;
  auto map = AffineLayoutMapAttr::get(&context,b.getStrArrayAttr({"dim0","dim1"}),
      b.getDenseI64ArrayAttr({8,8}),b.getStrArrayAttr({"byte_offset","bit_offset"}),
      b.getDenseI64ArrayAttr({256,8}),AffineMapAttr::get(AffineMap::get(2,0,{address,b.getAffineConstantExpr(0)},&context)));
  auto destination = StorageLayoutAttr::get(&context,map,MemorySpaceAttr::get(&context,attr::MemorySpace::Shared),
      b.getI64IntegerAttr(1024),b.getI64IntegerAttr(1));
  module->walk([&](CopyOp op) { cast<LayoutViewOp>(op.getDst().getDefiningOp()).setLayoutAttr(destination); });
  auto graph = collectLayoutConstraints(*module,*target); ASSERT_TRUE(succeeded(graph));
  EXPECT_EQ(paths(*graph).count("tma"),0u);
  EXPECT_EQ(paths(*graph).count("vector"),1u);
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph,c); ASSERT_EQ(encodings.size(),4u);
    encodings[3] = b.getStringAttr("tma");
    auto result = buildSM90CopyContractDetailed(*graph,c,encodings);
    EXPECT_EQ(result.proof.status,ProofStatus::Disproven);
    EXPECT_EQ(result.reasonCode,"copy-descriptor"); EXPECT_EQ(result.operand,1u);
    EXPECT_EQ(result.proof.counterexample,(SmallVector<int64_t>{0,1}));
    EXPECT_NE(result.proof.reason.find("representative mismatch for swizzle_bytes=0"),std::string::npos) << result.proof.reason;
    EXPECT_FALSE(result.binding);
  }
}

TEST_F(SM90CopyContractTest, DetailedTargetAndDecoderAreNotHardwareDisproofs) {
  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
    encodings[3] = Builder(&context).getStringAttr("invented-path");
    auto decoder = buildSM90CopyContractDetailed(*graph, c, encodings);
    EXPECT_EQ(decoder.proof.status, ProofStatus::Unknown);
    EXPECT_EQ(decoder.reasonCode, "copy-decoder");
    encodings[3] = Builder(&context).getStringAttr("vector");
    (*module)->setAttr("frisk.target", Builder(&context).getStringAttr("sm_80"));
    ScopedDiagnosticHandler quiet(&context, [](Diagnostic &) { return success(); });
    auto unsupported = buildSM90CopyContractDetailed(*graph, c, encodings);
    EXPECT_EQ(unsupported.proof.status, ProofStatus::Unknown);
    EXPECT_EQ(unsupported.reasonCode, "copy-target");
    EXPECT_FALSE(unsupported.binding);
  }
}

TEST_F(SM90CopyContractTest, SingleF16OffsetAndNoncontiguousElementsKeepScalarFallback) {
  for (bool offset : {true, false}) {
    auto module = makeND({8,16}, Float16Type::get(&context), 1, 3,
        128, 1024, false, offset ? 1 : 0, offset ? 1 : 2);
    ASSERT_TRUE(module);
    auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
    EXPECT_EQ(paths(*graph), (std::set<std::string>{"vector"}));
    for (const auto &c : graph->getConstraints()) if (c.instruction) {
      auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
      encodings[3] = Builder(&context).getStringAttr("cp.async");
      auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
      EXPECT_EQ(result.proof.status, ProofStatus::Disproven);
      EXPECT_EQ(result.reasonCode, offset ? "copy-alignment" : "copy-address-contiguity");
      EXPECT_EQ(result.operand, 0u);
      EXPECT_FALSE(result.proof.counterexample.empty());
      EXPECT_FALSE(result.binding);
    }
    auto solution = solveLayoutGraph(*graph, *target); ASSERT_TRUE(succeeded(solution));
    ASSERT_TRUE(succeeded(materializeLayouts(*module, *graph, *solution)));
    EXPECT_TRUE(succeeded(verifyMaterializedLayouts(*module, *target)));
    module->walk([&](CopyOp op) {
      auto binding = op->getAttrOfType<DictionaryAttr>("frisk.copy_contract");
      EXPECT_EQ(binding.getAs<IntegerAttr>("vector_bytes").getInt(), 2);
    });
  }
}

TEST_F(SM90CopyContractTest, OffsetAndNoncontiguousHardVectorsFailAtomically) {
  for (bool offset : {true, false}) {
    auto module = makeND({8,16}, Float16Type::get(&context), 1, 3,
        128, 1024, false, offset ? 1 : 0, offset ? 1 : 2);
    ASSERT_TRUE(module);
    module->walk([&](CopyOp op) {
      op->setAttr("frisk.vector_bytes", Builder(&context).getI64IntegerAttr(4));
    });
    expectAtomicFailure(*module);
  }
}

TEST_F(SM90CopyContractTest, UnsupportedTargetAndDuplicateWritersFailAtomically) {
  auto unsupported = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(unsupported);
  (*unsupported)->setAttr("frisk.target", Builder(&context).getStringAttr("sm_80"));
  expectAtomicFailure(*unsupported);

  auto module = makeND({8,16}, Float32Type::get(&context)); ASSERT_TRUE(module);
  module->walk([&](CopyOp op) {
    op->setAttr("frisk.writer_policy", Builder(&context).getStringAttr("all"));
  });
  auto graph = collectLayoutConstraints(*module, *target); ASSERT_TRUE(succeeded(graph));
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
    auto execution = cast<DistributedEncodingAttr>(encodings[2]);
    auto topology = llvm::to_vector(execution.getTopology().asArrayRef()); topology[4] = 2;
    auto duplicated = DistributedEncodingAttr::get(&context, execution.getMap(),
        Builder(&context).getDenseI64ArrayAttr(topology),
        Builder(&context).getI64IntegerAttr(execution.getReplication().getInt() * 2));
    c.instruction->source->setAttr("frisk.execution_layout", duplicated);
  }
  expectAtomicFailure(*module);
}

TEST_F(SM90CopyContractTest, CheckedAsyncPlanRejectsChangedDirectionAndStrideAtomically) {
  auto baseline = makeND({8,8}, Float32Type::get(&context)); ASSERT_TRUE(baseline);
  auto graph = collectLayoutConstraints(*baseline, *target); ASSERT_TRUE(succeeded(graph));
  Attribute checked;
  for (const auto &c : graph->getConstraints()) if (c.instruction) {
    auto encodings = actualTuple(*graph, c); ASSERT_EQ(encodings.size(), 4u);
    encodings[3] = Builder(&context).getStringAttr("tma");
    auto result = buildSM90CopyContractDetailed(*graph, c, encodings);
    ASSERT_EQ(result.proof.status, ProofStatus::Proven); checked = result.binding;
  }
  ASSERT_TRUE(checked);
  auto stride = makeND({8,8}, Float32Type::get(&context), 1, 3, 16, 1024, true);
  ASSERT_TRUE(stride);
  stride->walk([&](CopyOp op) { op->setAttr("frisk.copy_contract", checked); });
  expectAtomicFailure(*stride);
  baseline->walk([&](CopyOp op) {
    op->setAttr("frisk.copy_contract", checked);
    Value source = op->getOperand(0), destination = op->getOperand(1);
    op->setOperand(0, destination); op->setOperand(1, source);
  });
  expectAtomicFailure(*baseline);
}

TEST_F(SM90CopyContractTest, TraceSeparatesSupportedRejectedAndSelectedTuples) {
  auto module = makeND({8,16}, Float32Type::get(&context), 1, 3, 1, 1024);
  ASSERT_TRUE(module);
  LayoutTrace trace({"/tmp/frisk-copy-trace-unused.json", 4096, 8388608});
  auto graph = collectLayoutConstraints(*module, *target,
      LayoutCollectionMode::InitializeCandidates, nullptr, &trace);
  ASSERT_TRUE(succeeded(graph));
  ASSERT_TRUE(succeeded(propagateCommonToFixedPoint(*graph, &trace)));
  auto solution = solveLayoutGraph(*graph, *target, {}, nullptr, nullptr, &trace);
  ASSERT_TRUE(succeeded(solution));
  auto text = trace.serialize();
  for (StringRef required : {"instruction-supported", "instruction-rejected", "copy-alignment",
                             "endpoints", "operand", "counterexample", "solution", "cp.async", "tma"})
    EXPECT_NE(text.find(required.str()), std::string::npos) << required.str();
  auto full = llvm::json::parse(text); ASSERT_TRUE(bool(full));
  auto *events = full->getAsObject()->getArray("events"); ASSERT_TRUE(events);
  bool sawSupported = false, sawAlignment = false, sawSelected = false;
  for (const auto &value : *events) {
    auto *event = value.getAsObject(); ASSERT_TRUE(event);
    auto kind = event->getString("event");
    if (kind == "instruction-supported") {
      sawSupported = true;
      EXPECT_EQ(event->getString("status"), "proven");
      EXPECT_EQ(event->getString("path"), "vector");
    } else if (kind == "instruction-rejected") {
      EXPECT_TRUE(event->getString("path"));
      EXPECT_TRUE(event->getString("reason_code"));
      auto *endpoints = event->getArray("endpoints"); ASSERT_TRUE(endpoints);
      EXPECT_EQ(endpoints->size(), 4u);
      EXPECT_TRUE(event->getArray("counterexample"));
      if (event->getString("reason_code") == "copy-alignment") {
        sawAlignment = true;
        EXPECT_EQ(event->getString("status"), "disproven");
        EXPECT_EQ(event->getInteger("operand"), 0);
      }
    } else if (kind == "solution") {
      sawSelected = true;
      auto *contracts = event->getArray("contracts"); ASSERT_TRUE(contracts);
      ASSERT_EQ(contracts->size(), 1u);
      auto binding = contracts->front().getAsObject()->getString("binding");
      ASSERT_TRUE(binding);
      EXPECT_TRUE(binding->contains("\"vector\""));
    }
  }
  EXPECT_TRUE(sawSupported); EXPECT_TRUE(sawAlignment); EXPECT_TRUE(sawSelected);
  LayoutTrace bounded({"/tmp/frisk-copy-trace-unused.json", 1, 4096});
  ASSERT_TRUE(succeeded(prepareInstructionTuples(*graph, *target, module->getLoc(), &bounded)));
  auto json = llvm::json::parse(bounded.serialize()); ASSERT_TRUE(bool(json));
  EXPECT_LE(bounded.serialize().size(), 4096u);
  EXPECT_TRUE(json->getAsObject()->getBoolean("truncated").value_or(false));
  EXPECT_LE(json->getAsObject()->getArray("events")->size(), 1u);
  EXPECT_GT(json->getAsObject()->getInteger("eventsOmitted").value_or(0), 0);
}
} // namespace
