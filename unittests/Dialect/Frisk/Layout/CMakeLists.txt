add_unittest(FriskUnitTests FriskLayoutUnitTests
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
  InstructionConstraintTest.cpp
  GemmLayoutIntegrationTest.cpp
  ReduceLayoutIntegrationTest.cpp
  CopyFillConstraintTest.cpp
  AliasRegionTest.cpp
  LayoutAliasAnalysisTest.cpp
  ExecutionLayoutProofTest.cpp
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
  LayoutVerifierTest.cpp
  LayoutMaterializationTest.cpp
  LegacyLayoutAdapterTest.cpp
  LegacyImportSemanticsTest.cpp
  LegacyNormalizationTest.cpp
  LegacyNormalizationNumericTest.cpp
  ProductLayoutTest.cpp
)

set_target_properties(FriskLayoutUnitTests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_compile_definitions(FriskLayoutUnitTests PRIVATE
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
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRIR
  MLIRParser
)
