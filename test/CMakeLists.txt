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

add_test(NAME FriskDemandCostSearchRepairTest
  COMMAND ${Python3_EXECUTABLE} -m unittest
    test_demand_repair_fixture test_demand_coverage_audit
    test_demand_repair test_demand_search_audit test_splat_constant_bits)
set_tests_properties(FriskDemandCostSearchRepairTest PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark/layout/execution
  TIMEOUT 180 LABELS "layout;selection;demand;python")

add_test(NAME FriskNativeCostRepairTest
  COMMAND ${Python3_EXECUTABLE} -m unittest
    test_native_cost_analysis test_native_cost_calibration
    test_native_cost_fixture test_native_cost_qualification
    test_native_surviving_audit test_surviving_cost_calibration)
set_tests_properties(FriskNativeCostRepairTest PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark/layout/execution
  TIMEOUT 180 LABELS "layout;selection;native-cost;python")

add_test(NAME FriskPartialCostEfficiencyTest
  COMMAND ${Python3_EXECUTABLE} -m unittest
    test_partial_cost_selection test_native_local_calibration
    test_native_compile_cache test_native_compiler_input_cache test_partial_cost_fixture
    test_partial_cost_qualification test_archive_efficiency_bench
    test_native_static_support)
set_tests_properties(FriskPartialCostEfficiencyTest PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark/layout/execution
  TIMEOUT 180 LABELS "layout;selection;partial-cost;python")

add_test(NAME FriskGF2DecisionRefinementTest
  COMMAND ${Python3_EXECUTABLE} -m unittest
    test_decision_priority test_gf2_decision_fixture
    test_sparse_shuffle test_offset_memo test_compatible_native_profiles
    test_minified_execution_json)
set_tests_properties(FriskGF2DecisionRefinementTest PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark/layout/execution
  TIMEOUT 300 LABELS "layout;selection;gf2-decision;python")

add_test(NAME FriskShuffleNetworkTest
  COMMAND ${Python3_EXECUTABLE} -m unittest
    test_shuffle_network test_shuffle_network_codegen test_sparse_shuffle)
set_tests_properties(FriskShuffleNetworkTest PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark/layout/execution
  TIMEOUT 300 LABELS "layout;shuffle-network;python")

add_test(NAME FriskNetworkSelectionExecutionTest
  COMMAND ${Python3_EXECUTABLE} -m unittest
    test_network_archive test_network_selection_fixture
    test_resource_decision test_common_static_contract
    test_network_selection_analysis test_pipeline_observer
    test_predicate_cost_recalibration test_tiled_execution
    test_emit_reuse test_redistribution test_storage_addresses
    test_parallel_compile_batch test_network_compilation_adapter)
set_tests_properties(FriskNetworkSelectionExecutionTest PROPERTIES
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/benchmark/layout/execution
  TIMEOUT 300 LABELS "layout;network-selection;execution;python")

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
