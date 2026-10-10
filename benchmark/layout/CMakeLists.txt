add_subdirectory(solver)
if(FRISK_ENABLE_CUDA_BENCHMARKS)
  add_subdirectory(conversion)
endif()
