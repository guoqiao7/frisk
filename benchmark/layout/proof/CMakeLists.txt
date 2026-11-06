add_llvm_executable(frisk_layout_proof_bench layout_proof_bench.cpp)
target_link_libraries(frisk_layout_proof_bench PRIVATE FriskLayoutAnalysis FriskIR LLVMTargetParser)

# Track implementation and build configuration identity, including dirty sources.
file(GLOB_RECURSE FRISK_PROOF_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.td"
  "${PROJECT_SOURCE_DIR}/lib/*.cpp" "${PROJECT_SOURCE_DIR}/lib/*.h"
  "${PROJECT_SOURCE_DIR}/lib/CMakeLists.txt" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
list(APPEND FRISK_PROOF_SOURCES "${PROJECT_SOURCE_DIR}/CMakeLists.txt"
  "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${PROJECT_SOURCE_DIR}/benchmark/layout/CMakeLists.txt")
list(SORT FRISK_PROOF_SOURCES)
set(FRISK_PROOF_MANIFEST "")
foreach(source IN LISTS FRISK_PROOF_SOURCES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(SHA256 "${source}" digest)
  file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${source}")
  string(APPEND FRISK_PROOF_MANIFEST "${relative}:${digest}\n")
endforeach()
string(SHA256 FRISK_PROOF_SOURCE_DIGEST "${FRISK_PROOF_MANIFEST}")
string(SHA256 FRISK_PROOF_CONFIG_DIGEST
  "${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_RELEASE};${MLIR_DIR};${LLVM_DIR}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.txt" CONTENT "${FRISK_PROOF_MANIFEST}")
target_compile_definitions(frisk_layout_proof_bench PRIVATE
  FRISK_PROOF_SOURCE_DIGEST="${FRISK_PROOF_SOURCE_DIGEST}"
  FRISK_PROOF_CONFIG_DIGEST="${FRISK_PROOF_CONFIG_DIGEST}"
  FRISK_PROOF_BUILD_TYPE="$<IF:$<BOOL:$<CONFIG>>,$<CONFIG>,unspecified>")

add_test(NAME FriskLayoutProofBenchmarkTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/python/layout_proof_benchmark_test.py
    --benchmark $<TARGET_FILE:frisk_layout_proof_bench> -v)
