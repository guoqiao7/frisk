add_frisk_library(FriskTransforms
    NormalizeLayoutIR.cpp
    LegacyFragmentNormalization.cpp
    LayoutInfer.cpp
    PreserveStorageContracts.cpp
    PassPipelines.cpp
    MultiAnchorSolver.cpp
    ProtectedStoreLayout.cpp
    MaterializeLayouts.cpp
    LayoutTypeConverter.cpp
    OptimizeLayoutConversions.cpp

    DEPENDS FriskTransformsIncGen

    LINK_LIBS PUBLIC
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    MLIRTransforms
    FriskLayoutToGPU
    FriskLayoutAnalysis
    FriskIR
    FriskSM90LayoutTarget
)

# Optional in-process integer optimizer for the certified multi-anchor pass.
set(FRISK_Z3_ROOT "" CACHE PATH "Z3 installation prefix")
find_path(FRISK_Z3_INCLUDE_DIR z3++.h HINTS "${FRISK_Z3_ROOT}/include")
find_library(FRISK_Z3_LIBRARY NAMES z3 HINTS "${FRISK_Z3_ROOT}/lib")
if(FRISK_Z3_INCLUDE_DIR AND FRISK_Z3_LIBRARY)
  target_include_directories(FriskTransforms SYSTEM PRIVATE "${FRISK_Z3_INCLUDE_DIR}")
  target_link_libraries(FriskTransforms PRIVATE "${FRISK_Z3_LIBRARY}")
  target_compile_definitions(FriskTransforms PRIVATE FRISK_HAS_Z3=1)
endif()
