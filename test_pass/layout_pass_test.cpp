#include "Dialect/Frisk/Transforms/Passes.h"

int main() {
  std::unique_ptr<mlir::Pass> pass =
      mlir::frisk::createFriskInferLayoutsPass();
  if (!pass || pass->getArgument() != "frisk-infer-layouts")
    return 1;
  return 0;
}
