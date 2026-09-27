#ifndef FRISK_ANALYSIS_STORAGEROOTCONTRACTS_H
#define FRISK_ANALYSIS_STORAGEROOTCONTRACTS_H

#include "Dialect/Frisk/Analysis/LayoutConstraint.h"

namespace mlir { class DominanceInfo; }
namespace mlir::frisk {
/// Storage evidence requires known execution regions with SSA dominance;
/// generic graph-region dominance does not establish execution order.
bool hasStorageContractScope(Operation *op, DominanceInfo &dominance);
bool storageContractDominates(Operation *declaration, Operation *use,
                              DominanceInfo &dominance);
/// Parses every declaration afresh, including otherwise unused contracts.
/// Performs bounded proofs only; never enumerates candidates.
FailureOr<SmallVector<RootStorageContract>>
collectStorageRootContracts(Operation *scope);
LogicalResult attachStorageRootContracts(Operation *scope,
                                        LayoutConstraintGraph &graph);
LayoutProof proveRootStorageContract(const RootStorageContract &contract,
                                    const StorageAliasInfo &view,
                                    Attribute candidate);
} // namespace mlir::frisk
#endif
