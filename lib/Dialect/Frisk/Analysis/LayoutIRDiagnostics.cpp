#include "Dialect/Frisk/Analysis/LayoutIRDiagnostics.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include <limits>
#include <map>

namespace mlir::frisk {
namespace {
struct Counts {
  uint64_t operations = 0, elements = 0;
  bool saturated = false;
};
uint64_t addSaturated(uint64_t lhs, uint64_t rhs, bool &saturated) {
  if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
    saturated = true;
    return std::numeric_limits<uint64_t>::max();
  }
  return lhs + rhs;
}
} // namespace

llvm::json::Object collectLayoutIRArithmetic(Operation *operation) {
  std::map<std::pair<std::string, std::string>, Counts> groups;
  llvm::json::Array unavailable;
  operation->walk([&](Operation *op) {
    auto dialect = op->getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math") ||
        op->hasTrait<OpTrait::ConstantLike>() ||
        !op->hasTrait<OpTrait::Elementwise>())
      return;
    StringRef opcode = op->getName().getStringRef();
    auto reject = [&](StringRef reason) {
      unavailable.push_back(llvm::json::Object{{"opcode", opcode.str()}, {"reason", reason}});
    };
    if (op->getNumResults() != 1) {
      reject("multiple-results");
      return;
    }
    Type dtype = op->getResult(0).getType();
    uint64_t elements = 1;
    bool saturated = false;
    if (auto tensor = dyn_cast<RankedTensorType>(dtype)) {
      if (!tensor.hasStaticShape()) {
        reject("dynamic-shape");
        return;
      }
      dtype = tensor.getElementType();
      // Zero must win even if an earlier prefix would overflow uint64_t.
      if (llvm::is_contained(tensor.getShape(), 0)) {
        elements = 0;
      } else {
        for (int64_t extent : tensor.getShape()) {
          if (uint64_t(extent) > std::numeric_limits<uint64_t>::max() / elements) {
            elements = std::numeric_limits<uint64_t>::max();
            saturated = true;
            break;
          }
          elements *= uint64_t(extent);
        }
      }
    }
    if (!isa<IntegerType, FloatType, IndexType>(dtype)) {
      reject("unsupported-result-type");
      return;
    }
    std::string name;
    llvm::raw_string_ostream(name) << dtype;
    auto &counts = groups[{opcode.str(), name}];
    counts.saturated |= saturated;
    counts.operations = addSaturated(counts.operations, 1, counts.saturated);
    counts.elements = addSaturated(counts.elements, elements, counts.saturated);
  });
  llvm::json::Array entries;
  for (const auto &[key, counts] : groups)
    entries.push_back(llvm::json::Object{{"opcode", key.first}, {"dtype", key.second},
        {"op_count", counts.operations}, {"element_ops", counts.elements},
        {"saturated", counts.saturated}});
  bool partial = !unavailable.empty();
  return llvm::json::Object{{"schema_version", 1}, {"status", partial ? "partial" : "available"},
      {"counting", "static-logical-result-elements-once-per-ir-op"},
      {"execution_frequency", "not-applied"}, {"entries", std::move(entries)},
      {"unavailable_operations", std::move(unavailable)}};
}
} // namespace mlir::frisk
