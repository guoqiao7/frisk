add_frisk_library(FriskSM90LayoutTarget
  SM90LayoutTarget.cpp
  SM90DistributedCandidates.cpp
  SM90MmaLayoutProof.cpp
  SM90GemmConstraints.cpp
  SM90ReduceConstraints.cpp

  LINK_LIBS PUBLIC
  FriskAnalysis
  FriskLayoutAnalysis
  FriskIR
  MLIRIR
)
