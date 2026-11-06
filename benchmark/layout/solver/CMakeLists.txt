add_llvm_executable(frisk_layout_solver_bench layout_solver_bench.cpp)
file(GLOB_RECURSE FRISK_BENCH_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.td"
  "${PROJECT_SOURCE_DIR}/lib/*.cpp" "${PROJECT_SOURCE_DIR}/lib/*.h"
  "${PROJECT_SOURCE_DIR}/lib/CMakeLists.txt" "${PROJECT_SOURCE_DIR}/benchmark/layout/solver/*.cpp")
list(APPEND FRISK_BENCH_SOURCES "${PROJECT_SOURCE_DIR}/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
list(SORT FRISK_BENCH_SOURCES)
set(FRISK_BENCH_MANIFEST "")
foreach(source IN LISTS FRISK_BENCH_SOURCES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(SHA256 "${source}" digest)
  file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${source}")
  string(APPEND FRISK_BENCH_MANIFEST "${relative}:${digest}\n")
endforeach()
string(SHA256 FRISK_BENCH_SOURCE_DIGEST "${FRISK_BENCH_MANIFEST}")
string(SHA256 FRISK_BENCH_CONFIG_DIGEST
  "${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_RELEASE};${MLIR_DIR};${LLVM_DIR}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.txt" CONTENT "${FRISK_BENCH_MANIFEST}")
target_compile_definitions(frisk_layout_solver_bench PRIVATE
  FRISK_BENCH_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
  FRISK_BENCH_SOURCE_DIGEST="${FRISK_BENCH_SOURCE_DIGEST}"
  FRISK_BENCH_CONFIG_DIGEST="${FRISK_BENCH_CONFIG_DIGEST}"
  FRISK_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
target_link_libraries(frisk_layout_solver_bench PRIVATE
  FriskTransforms FriskLayoutAnalysis FriskSM90LayoutTarget FriskIR
  MLIRParser MLIRArithDialect MLIRFuncDialect MLIRTensorDialect MLIRPass)
