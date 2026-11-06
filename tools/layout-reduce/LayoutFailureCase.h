#ifndef FRISK_TOOLS_LAYOUT_FAILURE_CASE_H
#define FRISK_TOOLS_LAYOUT_FAILURE_CASE_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"

namespace mlir::frisk::testing {
struct LayoutPropertyOptions {
  uint64_t seed = 1;
  unsigned cases = 32, maxBits = 8, maxVars = 8;
};
// Deliberately finite fixture format: no SSA, alias, region or conversion
// payload is accepted. Host IR is retained and parsed, never pointer encoded.
struct FailureVariable {
  std::string id;
  std::vector<std::string> candidates;
};
struct FailureRelation {
  std::string id, kind, rule, reason;
  std::vector<std::string> vars;
  std::string required;
};
struct FailureSignature {
  std::string kind, variable, relation, rule, reason;
  bool operator==(const FailureSignature &b) const;
};
struct LayoutFailureCase {
  uint64_t seed = 1;
  SolverOptions options;
  std::string host = "module {}";
  std::vector<FailureVariable> variables;
  std::vector<FailureRelation> relations;
  FailureSignature signature;
};
FailureOr<LayoutFailureCase> parseFailureCase(StringRef text,
                                             MLIRContext &context,
                                             std::string &error);
std::string serializeFailureCase(const LayoutFailureCase &input);
FailureOr<LayoutConstraintGraph> rebuildFailureGraph(
    const LayoutFailureCase &input, MLIRContext &context);
FailureOr<FailureSignature> replayFailureCase(const LayoutFailureCase &input,
                                             MLIRContext &context);
FailureOr<LayoutFailureCase> reduceFailureCase(const LayoutFailureCase &input,
                                              MLIRContext &context);
FailureOr<LayoutConstraintGraph> reduceFailingLayoutGraph(
    const LayoutConstraintGraph &graph,
    llvm::function_ref<bool(const LayoutConstraintGraph &)> fails);
} // namespace mlir::frisk::testing
#endif
