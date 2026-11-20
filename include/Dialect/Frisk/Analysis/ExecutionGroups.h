#pragma once
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::frisk {
LogicalResult bindExecutionGroups(ModuleOp module);

// A parallel region supplies the default logical domain. A planned operation
// may execute in a different, explicitly assigned subset of the physical CTA.
inline int64_t executionGroupWidth(Operation *op, ParallelOp parallel) {
  auto module = op->getParentOfType<ModuleOp>();
  auto role = op->getAttrOfType<StringAttr>("frisk.execution_role");
  if (!role) return parallel.getThreads();
  auto plan = module ? module->getAttrOfType<DictionaryAttr>("frisk.execution_contract") : DictionaryAttr();
  auto roles = plan ? plan.getAs<DictionaryAttr>("roles") : DictionaryAttr();
  auto groups = plan ? plan.getAs<ArrayAttr>("groups") : ArrayAttr();
  auto index = roles ? roles.getAs<IntegerAttr>(role.getValue()) : IntegerAttr();
  if (!groups || !index || index.getInt()<0 || uint64_t(index.getInt())>=groups.size()) return -1;
  auto group = dyn_cast<DictionaryAttr>(groups[index.getInt()]);
  auto width = group ? group.getAs<IntegerAttr>("threads") : IntegerAttr();
  return width ? width.getInt() : -1;
}
} // namespace mlir::frisk
