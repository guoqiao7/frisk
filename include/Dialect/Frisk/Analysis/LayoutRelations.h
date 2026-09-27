#ifndef FRISK_ANALYSIS_LAYOUTRELATIONS_H
#define FRISK_ANALYSIS_LAYOUTRELATIONS_H
#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
namespace mlir::frisk {
bool isSupportedHardLayoutConstraint(ConstraintKind kind);
bool satisfiesLayoutConstraint(const LayoutConstraintGraph &graph,
    const LayoutConstraint &constraint,
    const DenseMap<LayoutVarID, Attribute> &assignment, bool requireComplete);
LogicalResult verifyLayoutStorageCapacity(const LayoutConstraintGraph &graph,
    const LayoutVar &var, Attribute candidate);
bool isSupportedLayoutRelation(ConstraintKind kind);
bool isSupportedUnaryLayoutConstraint(ConstraintKind kind);
LayoutProof proveUnaryLayoutConstraint(const LayoutConstraintGraph &graph,
                                      const LayoutConstraint &constraint,
                                      Attribute candidate);
bool matchesLayoutThreadCount(Attribute encoding, int64_t threads);
bool layoutEncodingsEqual(Attribute lhs, Attribute rhs);
bool layoutRelationCompatible(const LayoutConstraintGraph &graph,
                              const LayoutConstraint &relation,
                              LayoutVarID lhsID, Attribute lhs,
                              LayoutVarID rhsID, Attribute rhs);
FailureOr<Attribute> projectLayoutCandidate(
    const LayoutConstraintGraph &graph, const LayoutConstraint &relation,
    LayoutVarID source, Attribute candidate, LayoutVarID target);
std::string layoutCandidateKey(Attribute value);
const StorageAliasFootprint &getStorageAliasFootprint(
    const LayoutConstraintGraph &graph, LayoutVarID id, Attribute candidate);
LayoutProof proveAliasLayoutRelation(const LayoutConstraintGraph &graph,
                                    LayoutVarID lhs, Attribute a,
                                    LayoutVarID rhs, Attribute b);
} // namespace mlir::frisk
#endif
