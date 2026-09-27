add_frisk_library(FriskAnalysis
  LayoutAlgebra.cpp
  LayoutConstraint.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSCFDialect
)

# The migration adapter depends on both the standalone algebra and concrete
# Frisk attrs, so keep it out of FriskAnalysis to avoid an FriskIR cycle.
add_frisk_library(FriskLegacyLayoutAdapter
  LegacyLayoutAdapter.cpp

  LINK_LIBS PUBLIC
  FriskAnalysis
  FriskIR
  MLIRIR
)

add_frisk_library(FriskLayoutAnalysis
  LayoutCostModel.cpp
  LayoutCandidateSolver.cpp
  ReductionLayoutProof.cpp
  ReductionLayoutConstraints.cpp
  InstructionLayoutConstraints.cpp
  MmaLayoutConstraints.cpp
  ExecutionLayoutProof.cpp
  LayoutAliasAnalysis.cpp
  StorageAliasCandidates.cpp
  StorageRootContracts.cpp
  LayoutPropagation.cpp
  LayoutVerifier.cpp
  LayoutRelations.cpp
  DistributedLayoutConstraints.cpp
  OperationLayoutConstraints.cpp

  LINK_LIBS PUBLIC
  FriskAnalysis
  FriskIR
  MLIRIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRTensorDialect
  MLIRMemRefDialect
)
