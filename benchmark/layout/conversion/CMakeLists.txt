find_package(CUDAToolkit REQUIRED)
file(GLOB_RECURSE FRISK_CONVERSION_SOURCES CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.td"
  "${PROJECT_SOURCE_DIR}/lib/*.cpp" "${PROJECT_SOURCE_DIR}/lib/*.h"
  "${PROJECT_SOURCE_DIR}/tools/*.cpp" "${PROJECT_SOURCE_DIR}/benchmark/layout/conversion/*.cpp"
  "${PROJECT_SOURCE_DIR}/benchmark/layout/conversion/*.cu" "${PROJECT_SOURCE_DIR}/benchmark/layout/conversion/*.py"
  "${PROJECT_SOURCE_DIR}/benchmark/layout/conversion/*.in" "${PROJECT_SOURCE_DIR}/benchmark/layout/conversion/CMakeLists.txt")
list(APPEND FRISK_CONVERSION_SOURCES "${PROJECT_SOURCE_DIR}/CMakeLists.txt")
list(SORT FRISK_CONVERSION_SOURCES)
set(FRISK_CONVERSION_MANIFEST "")
foreach(source IN LISTS FRISK_CONVERSION_SOURCES)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source}")
  file(SHA256 "${source}" digest)
  file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}" "${source}")
  string(APPEND FRISK_CONVERSION_MANIFEST "${relative}:${digest}\n")
endforeach()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/source-manifest.txt" CONTENT "${FRISK_CONVERSION_MANIFEST}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build-configuration.txt" CONTENT
  "${CMAKE_CXX_COMPILER};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_RELEASE};${CMAKE_CUDA_COMPILER};${CMAKE_CUDA_COMPILER_VERSION};${CMAKE_CUDA_FLAGS};${CMAKE_CUDA_FLAGS_RELEASE};${MLIR_DIR};${LLVM_DIR}\n")
add_llvm_executable(frisk_generate_conversion_cases generate_conversion_cases.cpp)
target_link_libraries(frisk_generate_conversion_cases PRIVATE
  FriskIR MLIRParser MLIRIR MLIRFuncDialect MLIRGPUDialect)

add_executable(frisk_layout_conversion_runtime layout_conversion_bench.cu)
set_target_properties(frisk_layout_conversion_runtime PROPERTIES CUDA_ARCHITECTURES 90)
target_link_libraries(frisk_layout_conversion_runtime PRIVATE CUDA::cuda_driver)
add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/reference.cubin"
  COMMAND ${CMAKE_CUDA_COMPILER} --cubin -DFRISK_REFERENCE_ONLY -std=c++17 -arch=sm_90 -O3 -lineinfo
    "${CMAKE_CURRENT_SOURCE_DIR}/layout_conversion_bench.cu" -o "${CMAKE_CURRENT_BINARY_DIR}/reference.cubin"
  DEPENDS layout_conversion_bench.cu VERBATIM)
add_custom_target(frisk_layout_reference_cubin DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/reference.cubin")

configure_file(run_conversion_bench.py.in frisk_layout_conversion_bench @ONLY)
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/frisk_layout_conversion_bench"
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
add_custom_target(frisk_layout_conversion_bench
  DEPENDS frisk_layout_conversion_runtime frisk_generate_conversion_cases frisk-opt frisk_layout_reference_cubin)
add_test(NAME FriskLayoutConversionGeneratorTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/python/layout_conversion_generator_test.py
    $<TARGET_FILE:frisk_generate_conversion_cases> $<TARGET_FILE:frisk-opt>)
