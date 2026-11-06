#ifndef FRISK_BENCHMARK_SELECTION_ORACLE_H
#define FRISK_BENCHMARK_SELECTION_ORACLE_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "llvm/Support/JSON.h"
#include <optional>
#include <set>

namespace mlir::frisk::selection {
/// These limits may only be reduced by tests; the CLI uses these exact values.
struct OracleLimits {
  uint64_t nodes = 65536, evaluations = 4096, milliseconds = 120000;
};
struct RelationProofEntry {
  LayoutConstraintID constraint;
  LayoutVarID lhs, rhs;
  Attribute a, b;
  LayoutProof proof;
};
struct CandidateProofEntry { LayoutVarID variable; Attribute candidate; bool valid, capacity; };
struct FrozenProofs {
  std::string commonDigest, domainDigest, coverageDigest;
  std::vector<CandidateProofEntry> candidates;
  std::vector<RelationProofEntry> entries;
  llvm::json::Array toJSON() const;
};
FrozenProofs freezeRelationProofs(const LayoutConstraintGraph &, LayoutTarget * = nullptr);
FailureOr<FrozenProofs> projectFrozenProofs(const LayoutConstraintGraph &common, const LayoutConstraintGraph &restricted, const FrozenProofs &);
llvm::json::Object domainJSON(const LayoutConstraintGraph &, const FrozenProofs &);
std::optional<std::pair<int64_t,int64_t>> geometryGrid(Attribute);
std::set<std::string> assignmentGeometries(const LayoutConstraintGraph &,
                                         const CandidateAssignment &);
FailureOr<LayoutConstraintGraph> restrictGeometry(const LayoutConstraintGraph &,
    std::optional<std::pair<int64_t,int64_t>>, Location);

struct ScoredAssignment {
  CostEstimate cost;
  SmallVector<LayoutConversionEdge> conversions;
  llvm::json::Array contributions;
};
FailureOr<ScoredAssignment> scoreAssignment(const LayoutConstraintGraph &,
    const CandidateAssignment &, LayoutTarget &, const ExperimentPolicy & = {});
struct OracleResult {
  std::string status, reason;
  bool exhausted = false, fatal = false;
  uint64_t visitedNodes = 0, evaluatedAssignments = 0, rejected = 0,
           unknownAssignments = 0;
  std::optional<LayoutSolution> best;
  std::set<std::string> optimalGeometries;
  llvm::json::Array contributions;
  OracleLimits limits;
  llvm::json::Object toJSON() const;
};
OracleResult enumerateOracle(const LayoutConstraintGraph &, const FrozenProofs &,
                             LayoutTarget &, OracleLimits = {}, const ExperimentPolicy & = {});
llvm::json::Array assignmentJSON(const LayoutConstraintGraph &,
                                const DenseMap<LayoutVarID,Attribute> &);
llvm::json::Array conversionsJSON(const LayoutConstraintGraph &,
                                 ArrayRef<LayoutConversionEdge>);
llvm::json::Array existingConversionsJSON(const LayoutConstraintGraph &,
                                        const CandidateAssignment &);
} // namespace mlir::frisk::selection
#endif
