#include "PreserveStorageContracts.h"
#include "Dialect/Frisk/Analysis/StorageRootContracts.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "mlir/IR/Dominance.h"

namespace mlir::frisk {
LogicalResult preserveStorageContracts(Operation *scope) {
  auto declarations = collectStorageRootContracts(scope);
  if (failed(declarations)) return failure();
  DominanceInfo dominance(scope);
  SmallVector<LayoutViewOp> views;
  scope->walk([&](LayoutViewOp view) { views.push_back(view); });
  for (auto view : views) {
    auto layout = view.getLayoutAttr();
    if (!layout) continue;
    auto info = analyzeStorageAlias(view.getResult());
    if (failed(info)) return failure();
    if (info->viewType.getShape() != info->rootType.getShape() ||
        info->viewType.getElementType() != info->rootType.getElementType() ||
        getFriskMemorySpace(info->viewType) != getFriskMemorySpace(info->rootType) ||
        !info->viewToRoot.isIdentity()) continue;
    if (!hasStorageContractScope(view, dominance))
      return view.emitOpError("storage-root-contract: unsupported region scope");
    bool exists = llvm::any_of(*declarations, [&](const auto &decl) {
      return decl.root.root == info->root && decl.layout == layout &&
             decl.declaration->getBlock() == view->getBlock() &&
             storageContractDominates(decl.declaration, view, dominance);
    });
    if (exists) continue;
    info->rootAlignment = std::max<uint64_t>(info->rootAlignment, layout.getAlignment().getInt());
    auto proof = verifyStorageAliasCandidate(*info, layout);
    if (proof.status != ProofStatus::Proven)
      return view.emitOpError("storage-root-contract: cannot preserve invalid whole-root binding: ")
          << proof.reason;
    OpBuilder builder(view);
    auto declaration = builder.create<StorageContractOp>(view.getLoc(), info->root, layout);
    declarations->push_back({declaration, *info, layout});
  }
  // Validate newly co-effective declarations, not just each isolated map.
  return success(succeeded(collectStorageRootContracts(scope)));
}
} // namespace mlir::frisk
