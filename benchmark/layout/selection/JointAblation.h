#ifndef FRISK_JOINT_ABLATION_H
#define FRISK_JOINT_ABLATION_H
#include "Dialect/Frisk/Analysis/LayoutSolver.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/JSON.h"
namespace mlir::frisk::selection {
/// Bind frozen graph IDs to actual SSA results across layout materialization.
void markDemandLineage(const LayoutConstraintGraph &);
LogicalResult resolveDemandLineage(ModuleOp);
FailureOr<LayoutSolution> solveAblation(LayoutConstraintGraph &,LayoutTarget &,
    SelectionMode,SolverOptions,llvm::json::Object &record);
/// Symbolic diagnostics: no target score, candidate enumeration, or hardware slot expansion.
llvm::json::Object describeArchiveStructure(const LayoutConstraintGraph &,const CandidateAssignment &);
FailureOr<uint64_t> archiveStructureDistance(const LayoutConstraintGraph &,const CandidateAssignment &,const CandidateAssignment &);
enum class ArchiveRetentionPolicy { LegacyHammingV2, PaidFeatureFrontierV1, PaidStaticFeaturesV1, CalibratedSourceGrammarV1 };
StringRef archiveRetentionPolicyName(ArchiveRetentionPolicy);
struct ArchivedLayoutCandidate {
  LayoutSolution solution;
  SmallVector<std::string> origins;
  uint64_t probesAtObservation = 0, refinementAtObservation = 0;
  bool localIncumbent = false, jointIncumbent = false;
  std::string networkObjective = "min-shfl";
  bool networkVariant = false;
  std::string retentionReason;
  llvm::json::Object retentionStructure;
};
/// One coordinated budget: informed-local (two internal stages) followed by
/// joint search, plus a bounded archive of paid complete full-objective states.
FailureOr<LayoutSolution> solveCandidateArchive(LayoutConstraintGraph &,LayoutTarget &,
    SelectionMode,SolverOptions,unsigned capacity,SmallVectorImpl<ArchivedLayoutCandidate> &,
    llvm::json::Object &record,LayoutTarget *minWorkVariantTarget=nullptr,
    ArchiveRetentionPolicy retention=ArchiveRetentionPolicy::LegacyHammingV2,
    bool auditPaidObservations=false,ModuleOp source={});
}
#endif
