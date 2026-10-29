add_library(FriskLayoutSelectionSupport STATIC SelectionCase.cpp SelectionExperiment.cpp
  SelectionControlled.cpp SelectionOracle.cpp)
target_include_directories(FriskLayoutSelectionSupport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(FriskLayoutSelectionSupport PUBLIC
  FriskTransforms FriskLayoutAnalysis FriskSM90LayoutTarget FriskIR
  MLIRParser MLIRArithDialect MLIRFuncDialect MLIRMemRefDialect MLIRPass)
add_llvm_executable(frisk_layout_selection_bench PARTIAL_SOURCES_INTENDED layout_selection_bench.cpp)
target_link_libraries(frisk_layout_selection_bench PRIVATE FriskLayoutSelectionSupport)

# Identity covers this support library, production implementation and all build
# configuration, rather than identifying an already-dirty tree only by HEAD.
file(GLOB_RECURSE FRISK_SELECTION_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.td"
  "${PROJECT_SOURCE_DIR}/lib/*.cpp" "${PROJECT_SOURCE_DIR}/lib/*.h"
  "${PROJECT_SOURCE_DIR}/lib/*CMakeLists.txt"
  "${PROJECT_SOURCE_DIR}/benchmark/layout/selection/*.cpp"
  "${PROJECT_SOURCE_DIR}/benchmark/layout/selection/*.h"
  "${PROJECT_SOURCE_DIR}/cmake/*.cmake")
list(APPEND FRISK_SELECTION_SOURCES
  "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
  "${PROJECT_SOURCE_DIR}/benchmark/CMakeLists.txt"
  "${PROJECT_SOURCE_DIR}/benchmark/layout/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt"
  "${PROJECT_SOURCE_DIR}/unittests/Dialect/Frisk/Layout/CMakeLists.txt"
  "${PROJECT_SOURCE_DIR}/unittests/Dialect/Frisk/Layout/LayoutSelectionQualityTest.cpp")
list(SORT FRISK_SELECTION_SOURCES)
set(FRISK_SELECTION_MANIFEST "")
foreach(source IN LISTS FRISK_SELECTION_SOURCES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(SHA256 "${source}" digest)
  file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${source}")
  string(APPEND FRISK_SELECTION_MANIFEST "${relative}:${digest}\n")
endforeach()
string(SHA256 FRISK_SELECTION_SOURCE_DIGEST "${FRISK_SELECTION_MANIFEST}")
string(SHA256 FRISK_SELECTION_CONFIG_DIGEST
  "${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_RELEASE};${MLIR_DIR};${LLVM_DIR}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.txt" CONTENT "${FRISK_SELECTION_MANIFEST}")
target_compile_definitions(FriskLayoutSelectionSupport PRIVATE
  FRISK_SELECTION_SOURCE_DIGEST="${FRISK_SELECTION_SOURCE_DIGEST}"
  FRISK_SELECTION_CONFIG_DIGEST="${FRISK_SELECTION_CONFIG_DIGEST}"
  FRISK_SELECTION_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  FRISK_SELECTION_SOURCE_MANIFEST="${CMAKE_CURRENT_BINARY_DIR}/source-manifest.txt")
