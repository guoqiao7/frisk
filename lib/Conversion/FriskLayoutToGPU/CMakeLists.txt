add_frisk_library(FriskLayoutToGPU
  ExportExecution.cpp
  LayoutConversionPlan.cpp
  TestLayoutConversionLowering.cpp
  DEPENDS FriskTransformsIncGen
  LINK_LIBS PUBLIC FriskIR FriskLayoutAnalysis FriskSM90LayoutTarget MLIRIR MLIRPass
    MLIRArithDialect MLIRGPUDialect MLIRMemRefDialect MLIRSCFDialect MLIRVectorDialect
)
