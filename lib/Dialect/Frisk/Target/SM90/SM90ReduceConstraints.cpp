#include "Dialect/Frisk/Target/SM90/SM90ReduceConstraints.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/IR/FriskOps.h"

namespace mlir::frisk {
namespace {
FailureOr<StringRef> reductionTarget(Operation *op) {
  for (auto *scope = op; scope; scope = scope->getParentOp()) {
    if (auto attr = scope->getAttr("frisk.target")) {
      auto name = dyn_cast<StringAttr>(attr);
      if (!name || (name.getValue() != "sm_90" && name.getValue() != "sm_90a"))
        return failure();
      return name.getValue();
    }
  }
  return StringRef("sm_90");
}

/// Decode the one versioned algorithm, never enumerate candidates or plans.
FailureOr<DictionaryAttr> decodeCanonicalContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, Attribute source, Attribute result,
    LayoutProof &proof) {
  proof = {ProofStatus::Disproven, {}, "invalid reduction endpoint or metadata"};
  if (!constraint.reduction || constraint.vars.size() != 2) return failure();
  auto op = dyn_cast_or_null<ReduceTensorOp>(constraint.reduction->source);
  auto src = dyn_cast_or_null<DistributedEncodingAttr>(source);
  auto dst = dyn_cast_or_null<DistributedEncodingAttr>(result);
  if (!op || !src || !dst) return failure();
  auto target = reductionTarget(op);
  if (failed(target)) {
    proof.reason = "reduce-target: only sm_90/sm_90a are supported";
    return failure();
  }
  int64_t threads = graph.getVariable(constraint.vars[0]).requiredThreads;
  if (threads != graph.getVariable(constraint.vars[1]).requiredThreads ||
      !matchesLayoutThreadCount(src, threads) || !matchesLayoutThreadCount(dst, threads)) {
    proof.reason = "reduce-thread-scope: input/result do not match execution_threads";
    return failure();
  }
  auto srcType = cast<RankedTensorType>(op.getSource().getType());
  auto dstType = cast<RankedTensorType>(op.getResult().getType());
  auto analysis = buildReductionLayoutProof(src, dst, srcType, dstType, op.getDim());
  proof = analysis.proof;
  if (proof.status != ProofStatus::Proven) return failure();
  Builder b(op->getContext());
  std::string dtype;
  llvm::raw_string_ostream(dtype) << srcType.getElementType();
  NamedAttrList fields;
  fields.set("version", b.getI64IntegerAttr(1));
  fields.set("target", b.getStringAttr(*target));
  fields.set("kind", b.getStringAttr(op.getKind()));
  fields.set("axis", b.getI64IntegerAttr(op.getDim()));
  fields.set("dtype", b.getStringAttr(dtype));
  fields.set("source_shape", b.getDenseI64ArrayAttr(srcType.getShape()));
  fields.set("result_shape", b.getDenseI64ArrayAttr(dstType.getShape()));
  fields.set("threads", b.getI64IntegerAttr(threads));
  fields.set("algorithm", b.getStringAttr("canonical_fiber_tree_v1"));
  fields.set("input_policy", b.getStringAttr("first_owner"));
  fields.set("output_policy", b.getStringAttr("broadcast_complete"));
  fields.set("scope", b.getStringAttr(analysis.scope));
  return fields.getDictionary(op->getContext());
}
} // namespace

FailureOr<Attribute> buildSM90ReductionContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, Attribute source, Attribute result) {
  LayoutProof proof;
  auto fields = decodeCanonicalContract(graph, constraint, source, result, proof);
  if (failed(fields)) return failure();
  return Attribute(ReductionContractAttr::get(fields->getContext(), *fields));
}

LayoutProof verifySM90ReductionContract(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint, Attribute source, Attribute result,
    Attribute binding) {
  auto contract = dyn_cast_or_null<ReductionContractAttr>(binding);
  if (!contract)
    return {ProofStatus::Disproven, {}, "reduce-contract: expected complete typed binding"};
  LayoutProof proof;
  auto fields = decodeCanonicalContract(graph, constraint, source, result, proof);
  if (failed(fields)) return proof;
  if (contract.getPayload() != *fields)
    return {ProofStatus::Disproven, {}, "reduce-contract: binding differs from actual types, target, threads or communication scope"};
  return proof;
}
} // namespace mlir::frisk
