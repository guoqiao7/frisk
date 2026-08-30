#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def FriskInferLayouts : InterfacePass<"frisk-infer-layouts", "FunctionOpInterface"> {
  let summary = "Infer and materialize Frisk layouts";
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
}

#endif // FRISK_PASSES
