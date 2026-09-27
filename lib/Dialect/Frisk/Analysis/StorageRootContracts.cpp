#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"

namespace mlir::frisk {
bool hasStorageContractScope(Operation *op, DominanceInfo &dominance) {
  for (Region *region = op->getParentRegion(); region;
       region = region->getParentRegion()) {
    Operation *parent = region->getParentOp();
    // Module is a symbol container, not an execution region.
    if (isa<ModuleOp>(parent)) return true;
    if (!isa<func::FuncOp, KernelOp, ParallelOp, scf::IfOp, scf::ForOp,
             scf::WhileOp>(parent) || !dominance.hasSSADominance(region))
      return false;
  }
  return false;
}

bool storageContractDominates(Operation *declaration, Operation *use,
                              DominanceInfo &dominance) {
  return hasStorageContractScope(declaration, dominance) &&
         hasStorageContractScope(use, dominance) &&
         dominance.dominates(declaration, use);
}

FailureOr<SmallVector<RootStorageContract>>
collectStorageRootContracts(Operation *scope) {
  SmallVector<RootStorageContract> contracts;
  DominanceInfo dominance(scope);
  bool invalid = false;
  scope->walk([&](StorageContractOp op) {
    if (failed(op.verify())) { invalid = true; return; }
    if (!hasStorageContractScope(op, dominance)) {
      op.emitOpError("storage-root-contract: unsupported region scope");
      invalid = true;
      return;
    }
    auto info = analyzeStorageAlias(op.getRoot());
    if (failed(info)) { invalid = true; return; }
    // Alignment is an explicit precondition, not inferred allocation metadata.
    info->rootAlignment = std::max<uint64_t>(info->rootAlignment,
                                           op.getLayout().getAlignment().getInt());
    info->alignmentEvidence += "; explicit storage_contract precondition";
    auto proof = verifyStorageAliasCandidate(*info, op.getLayout());
    if (proof.status != ProofStatus::Proven) {
      op.emitOpError("storage-root-contract: ") << proof.reason;
      invalid = true;
      return;
    }
    contracts.push_back({op, *info, op.getLayout()});
  });
  if (invalid) return failure();
  for (unsigned i = 0; i < contracts.size(); ++i)
    for (unsigned j = i + 1; j < contracts.size(); ++j) {
      auto &a = contracts[i];
      auto &b = contracts[j];
      if (a.root.root != b.root.root) continue;
      if (!storageContractDominates(a.declaration, b.declaration, dominance) &&
          !storageContractDominates(b.declaration, a.declaration, dominance)) continue;
      auto proof = proveStorageAliasCompatible(a.root,
          cast<StorageLayoutAttr>(a.layout), b.root, cast<StorageLayoutAttr>(b.layout));
      if (proof.status != ProofStatus::Proven) {
        b.declaration->emitError("storage-root-contract: incompatible declarations: ")
            << proof.reason;
        return failure();
      }
    }
  return contracts;
}

LayoutProof proveRootStorageContract(const RootStorageContract &contract,
                                    const StorageAliasInfo &view,
                                    Attribute candidate) {
  auto storage = dyn_cast_or_null<StorageLayoutAttr>(candidate);
  auto rootLayout = dyn_cast_or_null<StorageLayoutAttr>(contract.layout);
  if (!storage || !rootLayout || view.root != contract.root.root)
    return {ProofStatus::Unknown, {}, "invalid root storage contract endpoint"};
  return proveStorageAliasCompatible(contract.root, rootLayout, view, storage);
}

LogicalResult attachStorageRootContracts(Operation *scope,
                                        LayoutConstraintGraph &graph) {
  auto contracts = collectStorageRootContracts(scope);
  if (failed(contracts)) return failure();
  DominanceInfo dominance(scope);
  for (auto &var : graph.getVariables()) {
    if (!var.storageAlias) continue;
    for (const auto &contract : *contracts) {
      if (contract.root.root != var.storageAlias->root ||
          !storageContractDominates(contract.declaration, var.anchor, dominance)) continue;
      var.storageAlias->rootAlignment = std::max(var.storageAlias->rootAlignment,
                                                contract.root.rootAlignment);
      var.storageAlias->alignmentEvidence += "; dominating storage_contract";
      auto id = graph.addConstraint(ConstraintKind::RootStorageContract,
          ConstraintStrength::Hard, {var.id}, contract.declaration,
          "root-storage-contract", "dominating complete root address precondition");
      graph.getConstraints()[id].rootStorage = contract;
    }
  }
  return success();
}
} // namespace mlir::frisk
