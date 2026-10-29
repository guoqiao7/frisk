add_unittest(FriskUnitTests FriskLayoutUnitTests PARTIAL_SOURCES_INTENDED
  LayoutIRDiagnosticsTest.cpp
  LayoutBoundaryFrequencyTest.cpp
  LayoutIRResourcesTest.cpp
  LayoutProfileTest.cpp
  LayoutTraceTest.cpp
  LayoutStatisticsTest.cpp
  LayoutExecutableAdapterTest.cpp
  LayoutAffineEvaluationTest.cpp
  TileLangDifferentialTest.cpp
  TileLangCorpus.cpp
  LayoutFailureReductionTest.cpp
  LayoutPropertyTest.cpp
  LayoutSolverPropertyTest.cpp
  LayoutRelaxationTest.cpp
  LayoutConversionHoistTest.cpp
  SM90CopyContractTest.cpp
  CopyMmaIntegrationTest.cpp
  GuardedCopyTest.cpp
  SM90ContractTest.cpp
  SM90CostModelTest.cpp
  LayoutConversionChoiceTest.cpp
  LayoutCandidateSolverTest.cpp
  LayoutCostModelTest.cpp
  ReduceConstraintTest.cpp
  ReductionLayoutProofTest.cpp
  GemmConstraintTest.cpp
  SM90MmaLayoutProofTest.cpp
  SM90MmaGeometryTest.cpp
  MmaAutoIntegrationTest.cpp
  InstructionConstraintTest.cpp
  GemmLayoutIntegrationTest.cpp
  AttentionLayoutIntegrationTest.cpp
  ReduceLayoutIntegrationTest.cpp
  CopyFillConstraintTest.cpp
  AliasRegionTest.cpp
  LayoutAliasAnalysisTest.cpp
  ExecutionLayoutProofTest.cpp
  SymbolicExecutionLayoutTest.cpp
  StorageAliasIntegrationTest.cpp
  StorageRootContractTest.cpp
  LayoutPipelineTest.cpp
  LayoutConversionPlanTest.cpp
  BitLinearLayoutTest.cpp
  GF2MatrixTest.cpp
  LayoutEncodingTest.cpp
  LayoutInterfaceTest.cpp
  LayoutConstraintTest.cpp
  LayoutPropagationTest.cpp
  DistributedPropagationTest.cpp
  BroadcastLayoutTest.cpp
  SymbolicBroadcastLayoutTest.cpp
  LayoutVerifierTest.cpp
  LayoutMaterializationTest.cpp
  LegacyLayoutAdapterTest.cpp
  LegacyImportSemanticsTest.cpp
  LegacyNormalizationTest.cpp
  LegacyNormalizationNumericTest.cpp
  ProductLayoutTest.cpp
)

add_unittest(FriskUnitTests FriskLayoutSelectionTests PARTIAL_SOURCES_INTENDED LayoutSelectionQualityTest.cpp)
target_link_libraries(FriskLayoutSelectionTests PRIVATE FriskLayoutSelectionSupport)

set_target_properties(FriskLayoutUnitTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_compile_definitions(FriskLayoutUnitTests PRIVATE
  FRISK_COPY_MMA_FIXTURE="${PROJECT_SOURCE_DIR}/test/Transforms/copy-mma-contract.mlir"
  FRISK_ATTENTION_FIXTURE="${PROJECT_SOURCE_DIR}/test/Transforms/attention-layout.mlir"
  FRISK_ATTENTION_FIXED_ABI_FIXTURE="${PROJECT_SOURCE_DIR}/test/Transforms/attention-layout-fixed-abi.mlir"
  FRISK_TILELANG_CORPUS="${PROJECT_SOURCE_DIR}/test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json")

target_link_libraries(FriskLayoutUnitTests
  PRIVATE
  FriskLayoutFailureSupport
  FriskLegacyLayoutOracle
  FriskLayoutToGPU
  FriskTransforms
  FriskAnalysis
  FriskIR
  FriskLayoutAnalysis
  FriskSM90LayoutTarget
  FriskLegacyLayoutAdapter
  FriskTransforms
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRParser
)
