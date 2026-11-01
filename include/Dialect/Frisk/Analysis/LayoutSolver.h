#ifndef FRISK_ANALYSIS_LAYOUTSOLVER_H
#define FRISK_ANALYSIS_LAYOUTSOLVER_H

#include "Dialect/Frisk/Analysis/LayoutConstraint.h"
#include "Dialect/Frisk/Analysis/LayoutTarget.h"
#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"

#include "llvm/ADT/DenseMap.h"

#include "mlir/IR/Value.h"

namespace mlir::frisk {
struct LayoutAttemptStatistics;
struct LayoutStatisticsReport;
class LayoutTrace;

struct LayoutSolution {
  ExperimentPolicy policy;
  DenseMap<LayoutVarID, Attribute> assignments;
  SmallVector<LayoutConversionEdge> conversions;
  DenseMap<LayoutConstraintID, Attribute> instructionBindings;
  DenseMap<LayoutConstraintID, Attribute> reductionBindings;
  CostEstimate cost;
  struct ComponentStatistics {
    LayoutVarID root = 0;
    uint64_t rawCombinations = 1, combinations = 1;
    uint64_t expanded = 0, hardRejected = 0, beamDiscarded = 0;
    uint64_t deduplicated = 0;
    uint64_t peakFrontier = 0, mrvProbes = 0, expandedNodes = 0;
    uint64_t refinementProbes = 0, refinementSweeps = 0, refinementAccepted = 0;
    CostEstimate refinementStartCost, refinementEndCost;
    std::string refinementStopReason = "no-incumbent";
    // Local refinement history is retained even when global saturation makes
    // the original incumbent preferable. `cost` describes the selected plan.
    bool refinementGlobalFallback = false;
    bool rawProductOverflow = false, productOverflow = false;
    bool usedBeam = false, exhaustive = false;
    unsigned beamWidth = 0;
    SmallVector<unsigned> domains;
    CostEstimate cost;
  };
  struct CandidateRejection {
    LayoutVarID variable;
    std::string candidate, reason;
    LayoutConstraintID constraint;
  };
  SmallVector<ComponentStatistics, 1> components;
  SmallVector<CandidateRejection> rejections;
};

struct SolverOptions {
  uint64_t exactCombinationLimit = 256;
  unsigned beamWidth = 32;
  uint64_t maxExpandedStates = 65536;
  bool refineBeamSolution = true;
  uint64_t maxRefinementProbes = 256;
  ExperimentPolicy policy;
  SharedSearchBudget *sharedBudget = nullptr;
};

enum class LayoutSolveFailureKind {
  None, InvalidInput, EmptyDomain, HardConflict, Unsupported,
  SearchBudget, SearchIncomplete, CostFailure
};
struct LayoutSolveFailure {
  LayoutSolveFailureKind kind = LayoutSolveFailureKind::None;
  LayoutVarID variable = UINT32_MAX;
  LayoutConstraintID constraint = UINT32_MAX;
  std::string reason;
};

FailureOr<LayoutSolution>
solveLayoutGraph(LayoutConstraintGraph &graph, LayoutTarget &target,
                 SolverOptions options = {}, LayoutSolveFailure *why = nullptr,
                 LayoutAttemptStatistics *statistics = nullptr,
                 LayoutTrace *trace = nullptr, bool diagnostic = false);
void printLayoutSolutionStatistics(const LayoutConstraintGraph &graph,
                                   const LayoutSolution &solution,
                                   LayoutTarget &target, llvm::raw_ostream &os);


class LayoutConstraintBuilder {
public:
  explicit LayoutConstraintBuilder(LayoutConstraintGraph &graph)
      : graph(graph) {}

  LayoutVarID getOrCreateStorageVar(Value anchor);
  LayoutVarID getOrCreateDistributedVar(Value value);
  LogicalResult require(LayoutVarID var, Attribute encoding,
                        Operation *source, StringRef rule);
  LogicalResult same(LayoutVarID lhs, LayoutVarID rhs, Operation *source,
                     StringRef rule);
  LayoutVarID getOrCreateDistributedUse(OpOperand &use);
  LayoutVarID createOperationExecutionVar(Operation *op, RankedTensorType type,
                                          OperationExecutionBinding binding);
  LogicalResult transform(LayoutVarID src, LayoutVarID dst,
                          Attribute coordinateTransform, Operation *source,
                          StringRef rule);
  LogicalResult storageAccess(LayoutVarID distributed, LayoutVarID storage,
                              AccessKind access, Operation *source,
                              StringRef rule);
  LogicalResult convertible(LayoutVarID src, LayoutVarID dst, OpOperand &use,
                            bool existing = false);

  std::optional<LayoutVarID>
  lookup(Value value, LayoutKind kind = LayoutKind::Storage) const;

private:
  LayoutVarID getOrCreate(Value value, LayoutKind kind);

  LayoutConstraintGraph &graph;
  DenseMap<Value, LayoutVarID> storageVariablesByValue;
  DenseMap<Value, LayoutVarID> distributedVariablesByValue;
  DenseMap<OpOperand *, LayoutVarID> distributedVariablesByUse;
  uint64_t nextStableOrdinal = 0;
};

enum class LayoutCollectionMode {
  InitializeCandidates, RelationsOnly,
  // Keep empty ordinary domains for the structured relaxation coordinator.
  // Unsupported instruction/alias proofs still fail collection normally.
  InitializeForRelaxation
};

FailureOr<LayoutConstraintGraph> collectLayoutConstraints(
    Operation *root, LayoutTarget &target,
    LayoutCollectionMode mode = LayoutCollectionMode::InitializeCandidates,
    LayoutStatisticsReport *statistics = nullptr, LayoutTrace *trace = nullptr);

/// Extends the graph before finalization, including structured Tensor joins.
LogicalResult collectDistributedLayoutConstraints(
    Operation *root, LayoutConstraintGraph &graph, LayoutConstraintBuilder &builder);

/// Finite origins, visited once per actual endpoint; never a synthetic root var.
LogicalResult initializeStorageAliasCandidates(
    Operation *root, LayoutConstraintGraph &graph, LayoutTarget &target);

LogicalResult propagateStrict(LayoutConstraintGraph &graph, LayoutTrace *trace = nullptr);
LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph, LayoutTrace *trace = nullptr);

} // namespace mlir::frisk

#endif // FRISK_ANALYSIS_LAYOUTSOLVER_H
