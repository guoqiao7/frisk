add_executable(frisk_attr_test
  attr_test.cpp
)

llvm_update_compile_flags(frisk_attr_test)

target_link_libraries(frisk_attr_test
  PRIVATE
  FriskIR
  MLIRFuncDialect
  MLIRParser
)

add_executable(frisk_reduce_layout_test
  reduce_layout_test.cpp
)

llvm_update_compile_flags(frisk_reduce_layout_test)

target_link_libraries(frisk_reduce_layout_test
  PRIVATE
  FriskIR
  MLIRFuncDialect
  MLIRParser
)

add_test(NAME FriskAttrTest COMMAND frisk_attr_test)
add_test(NAME FriskReduceLayoutTest COMMAND frisk_reduce_layout_test)
