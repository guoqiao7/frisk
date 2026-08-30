#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect>();

  mlir::registerAllPasses();
  mlir::frisk::registerFriskPasses();

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Frisk optimizer\n", registry));
}
