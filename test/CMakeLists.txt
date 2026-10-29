set(FRISK_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(FRISK_BINARY_DIR ${PROJECT_BINARY_DIR})
add_subdirectory(Support)

add_test(NAME FriskTileLangExporterTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/python/tilelang_exporter_test.py)
add_test(NAME FriskLayoutBenchmarkRunnerTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/python/layout_benchmark_runner_test.py)
add_test(NAME FriskLayoutConversionCompilerTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/python/layout_conversion_compiler_test.py)
add_test(NAME FriskLayoutSelectionQualityRunnerTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test/python/layout_selection_quality_test.py)
set_tests_properties(FriskLayoutSelectionQualityRunnerTest PROPERTIES
  ENVIRONMENT "FRISK_SELECTION_BENCH=${PROJECT_BINARY_DIR}/benchmark/layout/selection/frisk_layout_selection_bench"
  TIMEOUT 240 LABELS "layout;selection;python;integration")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-frisk "Running Frisk regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS frisk-opt layout-reduce
)

set_target_properties(check-frisk PROPERTIES FOLDER "Tests")
