#ifndef FRISK_ANALYSIS_LAYOUTDEMAND_H
#define FRISK_ANALYSIS_LAYOUTDEMAND_H
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
#include "Dialect/Frisk/Analysis/ShuffleNetwork.h"
#include <array>
#include "llvm/Support/JSON.h"
#include <map>
namespace mlir::frisk {
/// An affine register query q:F2^7 -> F2^registerBits. Terms[0] is
/// the constant; terms[1..7] are thread-bit columns. It is a whole hardware
/// instruction vector, not a set of logical values: distinct queries retain
/// real execution multiplicity even when their logical images overlap.
struct LayoutRegisterQuery {
  std::array<uint64_t,8> terms{};
  uint64_t apply(unsigned thread) const;
  bool operator<(const LayoutRegisterQuery &other) const { return terms<other.terms; }
  bool operator==(const LayoutRegisterQuery &other) const { return terms==other.terms; }
};
struct LayoutDemandAccess {
  LayoutConstraintID constraint=0;
  uint64_t queryVectors=0,warpIssues=0,globalSectors32=0,sharedBankExcess=0;
  uint64_t addressAluWarpIssues=0;
};
struct LayoutDemandBoundary {
  LayoutConstraintID constraint=0;
  LayoutVarID source=0,target=0;
  ConversionExchangeKind exchange=ConversionExchangeKind::Identity;
  uint64_t sourceVectors=0,targetVectors=0,bytes=0,barriers=0;
  uint64_t addressAluWarpIssues=0,shuffleWarpIssues=0,sharedBankExcess=0;
  uint64_t sharedReadBankExcess=0,sharedWriteBankExcess=0;
  uint64_t sharedReadWarpIssues=0,sharedWriteWarpIssues=0;
  uint64_t receiverMuxAluWarpIssues=0;
  std::optional<shuffle_network::Plan> shuffleNetwork;
  SmallVector<LayoutRegisterQuery> queries;
};
struct LayoutDemandValue {
  LayoutVarID variable=0;
  SmallVector<LayoutRegisterQuery> queries;
  uint64_t dependencyDepth=0;
};
struct LayoutDemandLedger {
  bool exact=false,demandLowering=false,localScope=false;
  std::string reason;
  std::map<LayoutVarID,LayoutDemandValue> values;
  std::map<LayoutConstraintID,LayoutDemandAccess> accesses;
  std::map<LayoutConstraintID,LayoutDemandBoundary> boundaries;
  uint64_t declaredRegisterVectors=0,materializedQueryVectors=0;
  uint64_t loadVectors=0,arithmeticWarpIssues=0,addressAluWarpIssues=0;
  // Consecutive emitted global f32 requests, split by dependent arithmetic
  // or communication; a structural issue proxy, not measured concurrency.
  uint64_t globalLoadIssueGroups=0;
  // Declared serial scratch allocation; it can exceed live-demand scratch.
  uint64_t dependencySpan=0,scratchMaxBytes=0;
};
/// A static, conservative survival proof over the emitted query ledger.
/// Whole SSA values outside the root/communication dependency closure may be
/// discarded. Kept queries retain all128threads; no lane predication, cross-SSA
/// CSE, address CSE or physical register allocation is assumed. Actual emitted
/// Shared/Shuffle scaffolds are conservative roots, while declared scratch
/// remains allocated even when a runtime boundary was never emitted.
struct LayoutSurvivingDemandAnalysis {
  bool proven=false;
  std::string reason;
  LayoutDemandLedger emitted,surviving;
  SmallVector<LayoutVarID> retainedVariables,removedVariables;
  SmallVector<LayoutConstraintID> retainedCommunication;
  uint64_t removedQueryVectors=0;
};
LayoutSurvivingDemandAnalysis analyzeLayoutSurvivingDemand(
    const LayoutConstraintGraph &,const CandidateAssignment &,
    ArrayRef<LayoutVarID> localRoots={},uint64_t queryBudget=65536,
    shuffle_network::Objective networkObjective=shuffle_network::Objective::MinShuffle);
llvm::json::Object layoutSurvivingDemandJSON(const LayoutSurvivingDemandAnalysis &);

/// Backend-aligned affine query propagation for f32 pure single-block DAGs,
/// one final store, SM90 128 threads, C conversion semantics. Whole graph roots
/// use the final store; a nonempty localRoots list denotes independent producer
/// outputs (all their carriers), so no future-consumer demand enters LI.
/// Unsupported/incomplete/over-budget analysis returns exact=false plus reason;
/// it never supplies a zero substitute for unavailable work. No full replicated
/// hardware holder table is constructed. Access tables remain bounded logical
/// footprints; each demanded instruction visits only the fixed 128 threads.
llvm::json::Object layoutDemandJSON(const LayoutDemandLedger &);
LayoutProof verifyLayoutDemandBackendSupport(const LayoutConstraintGraph &,
    const CandidateAssignment &);
LayoutDemandLedger analyzeLayoutDemand(const LayoutConstraintGraph &,
    const CandidateAssignment &,ArrayRef<LayoutVarID> localRoots={},
    uint64_t queryBudget=65536,
    // false is reserved for frozen v4d historical model compatibility.
    // Diagnostics, backend-domain proofs and v5d use actual ordered emission.
    bool orderedEmission=true,
    shuffle_network::Objective networkObjective=shuffle_network::Objective::MinShuffle);
struct LayoutNetworkVariantComparison {
  bool proven=false,changed=false;
  std::string reason;
  uint64_t comparedShuffleBoundaries=0;
  llvm::json::Array boundaryPlans;
};
LayoutNetworkVariantComparison compareLayoutNetworkObjectives(
    const LayoutConstraintGraph &,const CandidateAssignment &);
} // namespace mlir::frisk
#endif
