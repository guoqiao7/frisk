# Deliberately not add_frisk_library: DEEPPIPE_LIBS is linked into frisk_ffi.
add_library(FriskLegacyLayoutOracle STATIC
  LegacyLayoutOracle.cpp
  LegacyLayoutOracleUtils.cpp
)
llvm_update_compile_flags(FriskLegacyLayoutOracle)
target_include_directories(FriskLegacyLayoutOracle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(FriskLegacyLayoutOracle PUBLIC FriskIR MLIRIR)
