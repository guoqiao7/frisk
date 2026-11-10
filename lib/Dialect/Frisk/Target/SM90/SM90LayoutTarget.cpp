#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"
#include "Dialect/Frisk/Analysis/LayoutDemand.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90ReduceConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90CopyConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/IR/FriskOps.h"

#include "Dialect/Frisk/IR/FriskAttributes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::frisk {

void enumerateSM90DistributedCandidates(
    const LayoutVar &var, SmallVectorImpl<LayoutCandidate> &out);

namespace {

ArrayAttr getDimensionNames(Builder &builder, unsigned rank) {
  SmallVector<Attribute> names;
  for (unsigned index = 0; index < rank; ++index)
    names.push_back(builder.getStringAttr("dim" + Twine(index)));
  return builder.getArrayAttr(names);
}

std::optional<attr::MemorySpace> getMemorySpace(MemRefType type) {
  std::optional<attr::MemorySpace> space = getFriskMemorySpace(type);
  if (!space || *space == attr::MemorySpace::Local)
    return std::nullopt;
  return space;
}

StorageLayoutAttr buildAffineStorage(MemRefType type, AffineExpr linear,
                                     int64_t storageElements) {
  MLIRContext *context = type.getContext();
  Builder builder(context);
  int64_t elementBits = type.getElementTypeBitWidth();
  AffineExpr bitAddress = linear * elementBits;
  AffineMap map = AffineMap::get(
      type.getRank(), 0, {bitAddress.floorDiv(8), bitAddress % 8}, context);
  int64_t storageBytes = llvm::divideCeil(
      static_cast<uint64_t>(storageElements) * elementBits, uint64_t{8});
  auto affine = AffineLayoutMapAttr::get(
      context, getDimensionNames(builder, type.getRank()),
      builder.getDenseI64ArrayAttr(type.getShape()),
      builder.getArrayAttr({builder.getStringAttr("byte_offset"),
                            builder.getStringAttr("bit_offset")}),
      builder.getDenseI64ArrayAttr({storageBytes, int64_t{8}}),
      AffineMapAttr::get(map));
  std::optional<attr::MemorySpace> space = getMemorySpace(type);
  if (!space)
    return {};
  int64_t elementBytes = std::max<int64_t>(1, elementBits / 8);
  int64_t alignment = llvm::isPowerOf2_64(elementBytes) ? elementBytes : 1;
  return StorageLayoutAttr::get(
      context, affine, MemorySpaceAttr::get(context, *space),
      builder.getI64IntegerAttr(alignment),
      builder.getI64IntegerAttr(alignment));
}

StorageLayoutAttr buildLinear(MemRefType type) {
  MLIRContext *context = type.getContext();
  AffineExpr linear = getAffineConstantExpr(0, context);
  int64_t stride = 1;
  for (int64_t dim = type.getRank() - 1; dim >= 0; --dim) {
    linear = linear + getAffineDimExpr(dim, context) * stride;
    stride *= type.getDimSize(dim);
  }
  return buildAffineStorage(type, linear, stride);
}

StorageLayoutAttr buildTranspose(MemRefType type) {
  if (type.getRank() != 2)
    return {};
  MLIRContext *context = type.getContext();
  AffineExpr linear = getAffineDimExpr(1, context) * type.getDimSize(0) +
                      getAffineDimExpr(0, context);
  return buildAffineStorage(type, linear, type.getNumElements());
}

StorageLayoutAttr buildPadded(MemRefType type) {
  if (type.getRank() != 2)
    return {};
  int64_t elementBits = type.getElementTypeBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return {};
  int64_t elementsPer32Bytes = std::max<int64_t>(1, 256 / elementBits);
  int64_t columns = type.getDimSize(1);
  int64_t paddedColumns = llvm::alignTo(columns, elementsPer32Bytes);
  if (paddedColumns == columns)
    paddedColumns += elementsPer32Bytes;
  MLIRContext *context = type.getContext();
  AffineExpr linear = getAffineDimExpr(0, context) * paddedColumns +
                      getAffineDimExpr(1, context);
  return buildAffineStorage(type, linear,
                            type.getDimSize(0) * paddedColumns);
}

DenseIntElementsAttr getDenseMatrix(MLIRContext *context, unsigned rows,
                                    unsigned columns,
                                    ArrayRef<uint64_t> basisOutputs) {
  SmallVector<APInt> values;
  values.reserve(rows * columns);
  for (unsigned row = 0; row < rows; ++row)
    for (unsigned column = 0; column < columns; ++column)
      values.emplace_back(1, (basisOutputs[column] >> row) & 1);
  auto matrixType = RankedTensorType::get(
      {static_cast<int64_t>(rows), static_cast<int64_t>(columns)},
      IntegerType::get(context, 1));
  return DenseIntElementsAttr::get(matrixType, values);
}

StorageLayoutAttr buildXor(MemRefType type, unsigned swizzleBytes) {
  if (type.getRank() != 2 || !llvm::isPowerOf2_64(type.getDimSize(0)) ||
      !llvm::isPowerOf2_64(type.getDimSize(1)))
    return {};
  unsigned elementBits = type.getElementTypeBitWidth();
  if (elementBits < 8 || elementBits % 8 != 0 ||
      !llvm::isPowerOf2_64(elementBits / 8))
    return {};

  unsigned rowBits = llvm::Log2_64(type.getDimSize(0));
  unsigned columnBits = llvm::Log2_64(type.getDimSize(1));
  unsigned elementByteBits = llvm::Log2_64(elementBits / 8);
  unsigned byteBits = rowBits + columnBits + elementByteBits;
  if (rowBits == 0 || byteBits >= 63)
    return {};

  SmallVector<uint64_t> basisOutputs(rowBits + columnBits);
  for (unsigned bit = 0; bit < rowBits; ++bit)
    basisOutputs[bit] = uint64_t{1}
                        << (columnBits + elementByteBits + bit);
  for (unsigned bit = 0; bit < columnBits; ++bit)
    basisOutputs[rowBits + bit] = uint64_t{1}
                                  << (elementByteBits + bit);
  // SM90 TMA non-interleaved canonical swizzle: byte[4:4+B] XOR
  // byte[7:7+B], B=1/2/3. Derive every basis from the byte equation.
  for (auto &basis : basisOutputs)
    basis ^= ((basis >> 7) & (swizzleBytes / 16 - 1)) << 4;

  MLIRContext *context = type.getContext();
  Builder builder(context);
  auto bitLinear = BitLinearLayoutMapAttr::get(
      context, getDimensionNames(builder, type.getRank()),
      builder.getDenseI64ArrayAttr(
          {static_cast<int64_t>(rowBits), static_cast<int64_t>(columnBits)}),
      builder.getArrayAttr({builder.getStringAttr("byte_offset"),
                            builder.getStringAttr("bit_offset")}),
      builder.getDenseI64ArrayAttr(
          {static_cast<int64_t>(byteBits), int64_t{3}}),
      getDenseMatrix(context, byteBits + 3, rowBits + columnBits,
                     basisOutputs));
  std::optional<attr::MemorySpace> space = getMemorySpace(type);
  if (!space)
    return {};
  int64_t alignment = std::max<unsigned>(1, elementBits / 8);
  return StorageLayoutAttr::get(
      context, bitLinear, MemorySpaceAttr::get(context, *space),
      builder.getI64IntegerAttr(alignment),
      builder.getI64IntegerAttr(alignment));
}

class SM90LayoutTarget final : public LayoutTarget {
  LayoutBoundaryFrequencyMode mode;
  std::optional<SM90CostProfile> profile;
  std::string calibratedID;
public:
  explicit SM90LayoutTarget(LayoutBoundaryFrequencyMode mode) : mode(mode) {}
  SM90LayoutTarget(LayoutBoundaryFrequencyMode mode, const SM90CostProfile &p)
      : mode(mode), profile(p), calibratedID(getSM90CalibratedCostModelID(p, mode)) {}
  LayoutBoundaryFrequencyMode boundaryFrequencyMode() const override { return mode; }
  void enumerateRelaxedCandidates(const LayoutConstraintGraph &,
      const LayoutVar &var, RelaxationKind kind,
      SmallVectorImpl<LayoutCandidate> &out) const override;
  StringRef relaxationCoverage(RelaxationKind kind) const override {
    switch (kind) {
    case RelaxationKind::Replication:
      return "bounded intermediate lane replication, at most 65536 carrier slots";
    case RelaxationKind::StoragePaddingOrNarrowerVector:
      return "ordinary storage padding and legal vector widths already enumerated; hard minima unchanged";
    case RelaxationKind::GuardedRagged:
      return "static whole-tile guarded Copy already in ordinary candidates; other ragged operations unsupported";
    case RelaxationKind::TMAFallback:
      return "cp.async/vector already compete in ordinary Copy candidates";
    case RelaxationKind::WGMMAFallback:
      return "unsupported: no implemented warp MMA/SIMT replacement for WGMMA";
    }
    llvm_unreachable("unknown relaxation kind");
  }
  FailureOr<Attribute> buildReductionContract(const LayoutConstraintGraph &graph,
      const LayoutConstraint &constraint, Attribute source, Attribute result) const override {
    return buildSM90ReductionContract(graph, constraint, source, result);
  }
  LayoutProof verifyReductionContract(const LayoutConstraintGraph &graph,
      const LayoutConstraint &constraint, Attribute source, Attribute result,
      Attribute binding) const override {
    return verifySM90ReductionContract(graph, constraint, source, result, binding);
  }
  LogicalResult prepareInstructionCandidates(LayoutConstraintGraph &graph) const override {
    if (failed(prepareSM90CopyCandidates(graph))) return failure();
    return prepareSM90MmaCandidates(graph);
  }
  FailureOr<Attribute> buildInstructionContract(const LayoutConstraintGraph &graph,
      const LayoutConstraint &constraint, ArrayRef<Attribute> encodings) const override {
    if (constraint.instruction && isa_and_nonnull<CopyOp>(constraint.instruction->source))
      return buildSM90CopyContract(graph, constraint, encodings);
    return buildSM90MmaContract(graph, constraint, encodings);
  }
  LayoutProof verifyInstructionContract(const LayoutConstraintGraph &graph,
      const LayoutConstraint &constraint, ArrayRef<Attribute> encodings,
      Attribute binding) const override {
    if (constraint.instruction && isa_and_nonnull<CopyOp>(constraint.instruction->source))
      return verifySM90CopyContract(graph, constraint, encodings, binding);
    return verifySM90MmaContract(graph, constraint, encodings, binding);
  }
  InstructionContractBuildResult buildInstructionContractDetailed(
      const LayoutConstraintGraph &graph, const LayoutConstraint &constraint,
      ArrayRef<Attribute> encodings) const override {
    if (constraint.instruction && isa_and_nonnull<CopyOp>(constraint.instruction->source))
      return buildSM90CopyContractDetailed(graph, constraint, encodings);
    return buildSM90MmaContractDetailed(graph, constraint, encodings);
  }
  LayoutProof precheckInstructionRole(const LayoutConstraintGraph &graph,
      const LayoutConstraint &constraint, unsigned role, Attribute candidate,
      Attribute geometry) const override {
    auto op = constraint.instruction ? dyn_cast_or_null<MmaOp>(constraint.instruction->source) : MmaOp{};
    if (!op || constraint.vars.size() != 5 || role >= 4)
      return {ProofStatus::Unknown, {}, "unsupported geometry-conditioned role"};
    auto g = decodeSM90MmaGeometry(op,graph.getVariable(constraint.vars[4]).requiredThreads,geometry);
    if (failed(g)) return {ProofStatus::Disproven, {}, "sm90-mma-thread-group: invalid geometry"};
    auto type = dyn_cast<RankedTensorType>(graph.getVariable(constraint.vars[role]).shapedType);
    if (!type) return {ProofStatus::Unknown, {}, "not a fragment role"};
    return verifySM90MmaFragment(*g,role == 0,role == 0 && op.getTransA(),type,
        dyn_cast_or_null<DistributedEncodingAttr>(candidate));
  }
  void enumerateCandidates(
      const LayoutVar &var,
      SmallVectorImpl<LayoutCandidate> &out) const override {
    if (var.kind == LayoutKind::Distributed) {
      enumerateSM90DistributedCandidates(var, out);
      return;
    }
    auto type = dyn_cast<MemRefType>(var.shapedType);
    if (var.kind != LayoutKind::Storage || !type || !type.hasStaticShape() ||
        !type.getElementType().isIntOrFloat())
      return;
    auto append = [&](Attribute candidate, uint64_t ordinal) {
      if (candidate)
        out.push_back({candidate, kInvalidProvenanceID, ordinal});
    };
    append(buildLinear(type), 0);
    append(buildTranspose(type), 1);
    append(buildPadded(type), 2);
    if (type.getRank() == 2 && type.getElementTypeBitWidth() % 8 == 0) {
      uint64_t rowBytes = type.getDimSize(1) *
                          (type.getElementTypeBitWidth() / 8);
      unsigned swizzleBytes = rowBytes <= 32 ? 32 : rowBytes <= 64 ? 64 : 128;
      uint64_t ordinal = swizzleBytes == 32 ? 3 : swizzleBytes == 64 ? 4 : 5;
      append(buildXor(type, swizzleBytes), ordinal);
    }
  }

  LogicalResult verifyCandidate(const LayoutVar &var, Attribute candidate,
                                Location loc) const override {
    if (var.kind == LayoutKind::Instruction) {
      if (auto mma = dyn_cast_or_null<MmaOp>(var.anchor))
        return success(succeeded(decodeSM90MmaGeometry(mma,var.requiredThreads,candidate)));
      return success(isa_and_nonnull<CopyOp>(var.anchor) && isSM90CopyPath(candidate));
    }
    if (var.kind == LayoutKind::Distributed) {
      auto tensor = dyn_cast<RankedTensorType>(var.shapedType);
      auto encoding = dyn_cast<DistributedEncodingAttr>(candidate);
      if (!tensor || !encoding)
        return emitError(loc) << "SM90 distributed candidate has incompatible kind";
      if (encoding.getTopology()[4] != 1)
        return emitError(loc) << "SM90 bootstrap distributed candidate requires a single CTA";
      return encoding.verifyForType(tensor, loc);
    }
    auto type = dyn_cast<MemRefType>(var.shapedType);
    auto storage = dyn_cast<StorageLayoutAttr>(candidate);
    if (var.kind != LayoutKind::Storage || !type || !storage)
      return emitError(loc) << "SM90 storage candidate has incompatible kind";
    return storage.verifyForType(type, loc);
  }

  bool needsCompleteAssignmentValidation(const LayoutConstraintGraph &graph) const override {
    if(graph.getVariables().empty())return false;
    const auto &v=graph.getVariables().front();Operation *root=v.anchor;Value value=v.value;
    if(!root && value)root=value.getParentBlock()->getParentOp();
    while(root && root->getParentOp())root=root->getParentOp();
    return root && root->hasAttr("frisk.execution_domain");
  }
  LayoutProof verifyExecutionSupport(const LayoutConstraintGraph &graph,
      const CandidateAssignment &assignment) const override {
    if (graph.getVariables().empty()) return {ProofStatus::Proven, {}, "empty graph"};
    const auto &v=graph.getVariables().front();
    Operation *root=v.anchor;
    Value anchorValue=v.value;
    if(!root && anchorValue)root=anchorValue.getParentBlock()->getParentOp();
    while(root && root->getParentOp())root=root->getParentOp();
    Attribute declared = root ? root->getAttr("frisk.execution_domain") : Attribute{};
    if(!declared)return {ProofStatus::Proven, {}, "no additional execution domain"};
    auto policy = dyn_cast<StringAttr>(declared);
    if(!policy)return {ProofStatus::Unknown, {}, "execution domain must be a recognized string contract"};
    if(policy.getValue()!="sm90-scalar-f32-dag-v1")
      return {ProofStatus::Unknown, {}, "unrecognized execution domain"};
    return verifyLayoutDemandBackendSupport(graph,assignment);
  }
  bool needsCompleteAssignmentCost() const override {
    return profile && profile->executionDemand;
  }
  FailureOr<CostEstimate> evaluateExecutionCost(const LayoutConstraintGraph &graph,
      const CandidateAssignment &assignment,const ExperimentPolicy &policy,
      LayoutCostObserver *observer) const override {
    if (!needsCompleteAssignmentCost()) return failure();
    bool pure = policy.conversion == ConversionPolicy::Independent &&
                policy.scope.producerInstructions.empty();
    Block *producerBlock = nullptr;
    for (const auto &v : graph.getVariables()) {
      if (v.kind != LayoutKind::Distributed || v.use) continue;
      auto ty = dyn_cast<RankedTensorType>(v.shapedType);
      Operation *op = v.value ? v.value.getDefiningOp() : nullptr;
      if (!ty || !ty.getElementType().isF32() || !op) { pure = false; break; }
      // A DAG belongs to one function or kernel body block. In particular, tensor
      // arithmetic nested in scf regions is a whole-graph carrier fallback,
      // rather than an assignment-dependent demand-analysis failure.
      Operation *parent = op->getParentOp();
      bool bodyContainer = parent &&
          (parent->getName().getStringRef() == "func.func" ||
           parent->getName().getStringRef() == "frisk.kernel");
      if (auto parallel = dyn_cast_or_null<ParallelOp>(parent))
        bodyContainer = parallel.getThreads() == 128 &&
            llvm::all_of(parallel.getRanges(), [](int64_t n) { return n == 1; });
      if (!bodyContainer || parent->getNumRegions() != 1 ||
          !parent->getRegion(0).hasOneBlock() ||
          (producerBlock && producerBlock != op->getBlock())) {
        pure = false; break;
      }
      unsigned parallelDepth = 0;
      for (Operation *ancestor = parent; ancestor; ancestor = ancestor->getParentOp()) {
        StringRef container = ancestor->getName().getStringRef();
        if (container == "frisk.parallel") ++parallelDepth;
        if ((container != "builtin.module" && container != "func.func" &&
             container != "frisk.kernel" && container != "frisk.parallel") ||
            parallelDepth > 1) { pure = false; break; }
      }
      if (!pure) break;
      producerBlock = op->getBlock();
      StringRef name = op->getName().getStringRef();
      if (name == "arith.constant") {
        auto value = op->getAttrOfType<DenseElementsAttr>("value");
        if (!value || !value.isSplat()) { pure = false; break; }
      }
      if (!llvm::is_contained(ArrayRef<StringRef>{"frisk.tile_load", "frisk.convert_layout",
            "arith.constant", "arith.mulf", "arith.subf", "arith.divf", "math.exp"}, name)) {
        pure = false; break;
      }
    }
    if (pure)
      return evaluateSM90DemandLayoutCost(graph, assignment, policy, &*profile, observer, mode);
    // Graph/policy-wide fallback, never an assignment-dependent free price.
    // MMA/reduction/control-flow and O remain the published carrier objective.
    // A bank-only demand profile is valid on pure DAGs, but supplies no
    // carrier objective. Fail the profile/scope combination explicitly; zero
    // carrier coefficients must never turn unsupported graphs into free work.
    // v6 requires a positive BAR fee on pure DAGs. That mandatory fee is
    // not an arithmetic/access carrier objective for unsupported graphs.
    bool hasCarrierObjective = false;
    for (unsigned i = 0; i < profile->coefficients.size(); ++i)
      if ((!profile->survivingDemand || i != 14) && profile->coefficients[i])
        hasCarrierObjective = true;
    if (!hasCarrierObjective) {
      for (const auto &v : graph.getVariables())
        if (v.anchor) {
          v.anchor->emitError("execution-demand-profile-scope: carrier fallback requires a nonzero carrier coefficient");
          break;
        }
      return failure();
    }
    auto carrier = *profile;
    carrier.executionDemand = false;
    carrier.executionOrder = false;
    carrier.survivingDemand = false;
    carrier.networkProxy = false;
    carrier.networkObjective = shuffle_network::Objective::MinShuffle;
    carrier.communicationCoefficients.fill(0);
    carrier.loadIssueGroupCoefficient = 0;
    carrier.demandBankCoefficients.fill(0);
    SM90LayoutTarget fallback(mode, carrier);
    return scoreExperimentAssignment(graph, assignment, fallback, policy, observer);
  }
  FailureOr<CostVector>
  evaluate(const CandidateAssignment &) const override {
    return needsCompleteAssignmentCost() ? FailureOr<CostVector>(failure()) : FailureOr<CostVector>(CostVector{});
  }
  FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &graph,
      const CandidateAssignment &assignment) const override {
    if (needsCompleteAssignmentCost()) { ExperimentPolicy p; p.conversion=ConversionPolicy::Independent; return evaluateExecutionCost(graph,assignment,p,nullptr); }
    return profile ? evaluateSM90CalibratedLayoutCost(graph, assignment, *profile, nullptr, mode)
                   : evaluateSM90LayoutCost(graph, assignment, nullptr, mode);
  }
  FailureOr<CostEstimate> evaluate(const LayoutConstraintGraph &graph,
      const CandidateAssignment &assignment, LayoutCostObserver *observer) const override {
    if (needsCompleteAssignmentCost()) { ExperimentPolicy p; p.conversion=ConversionPolicy::Independent; return evaluateExecutionCost(graph,assignment,p,observer); }
    return profile ? evaluateSM90CalibratedLayoutCost(graph, assignment, *profile, observer, mode)
                   : evaluateSM90LayoutCost(graph, assignment, observer, mode);
  }
  FailureOr<CostEstimate> lowerBound(const LayoutConstraintGraph &graph,
      const CandidateAssignment &assignment) const override {
    // Zero is a safe bound for nonnegative calibrated feature scores. Peak
    // SSA liveness and physical occupancy are deliberately not additive costs.
    return profile ? FailureOr<CostEstimate>(CostEstimate{})
                   : evaluateStaticLayoutCost(graph, assignment);
  }
  FailureOr<CostEstimate> adjustConversionCost(const CostEstimate &proxy,
      const LayoutConversionEdge &edge) const override {
    return profile ? adjustSM90CalibratedConversionCost(proxy, edge, *profile) : proxy;
  }
  FailureOr<CostEstimate> adjustCloneCost(const CostEstimate &proxy,
      const LayoutVar &result, Attribute encoding, Operation *op,
      Operation *at) const override {
    return profile ? adjustSM90CalibratedCloneCost(proxy, result, encoding, op, at, *profile, mode) : proxy;
  }
  bool requiresRematerializationTrafficReduction() const override { return !profile; }
  StringRef costModelID() const override { if (profile) return calibratedID; return mode == LayoutBoundaryFrequencyMode::Legacy ? getSM90LegacyCostModelID() : "sm90-layout-lex9-static-trip-count-v2"; }
  StringRef costCoverage() const override {
    if(profile && profile->networkProxy)return "sm90-network-surviving-proxy-v1; ordered-actual-network-grammar; explicit-min-shfl-or-min-work; migrated20weights-not-refit; structural-temporary-definitions-not-physical-registers; physical-resources-and-schedule=separate-compiler-driver-feedback";
    if (needsCompleteAssignmentCost() && profile->survivingDemand)
      return "sm90-surviving-demand-v1; emitted-and-surviving-query-ledgers; dead-pure-SSA-closure; actual-communication-roots-retained; positive-communication-issue-fees; no-emitter-load-group-price; graph-policy-fallback=carrier-v3-MMA-control-O; native-schedule=separate-bounded-compiler-feedback";
    if (needsCompleteAssignmentCost() && profile->executionOrder)
      return "sm90-affine-demand-v2; exact=f32-pure-DAG-splat-C128; producer-rooted-LI; full-assignment-dependencies; ordered-global-load-issue-groups; address-and-mux-ALU; graph-policy-fallback=carrier-v3-MMA-control-O; physical-registers=compiler-feedback; schedule-groups-are-not-measured-hardware-concurrency";
    if (needsCompleteAssignmentCost()) return "sm90-affine-demand-v1; exact=f32-pure-DAG-splat-C128; producer-rooted-LI; full-assignment-dependencies; address-and-mux-ALU; graph-policy-fallback=carrier-v3-MMA-control-O; physical-registers=compiler-feedback";
    if (profile)
      return "sm90-calibrated-features-v3; nonnegative-integer-score-ticks-not-cycles; symbolic-subspace-proof-v1; "
             "tensor-arithmetic-warp-issues-f16-f32-i1-i32; route-specific-conversion-bytes; "
             "CTA-barriers=conversion-and-global-to-shared-Copy-only; explicit-copy-setup-MMA-and-reduction-proxies; "
             "alignment-aware-32B-sectors-and-shared-bank-excess; frequency-weighted; "
             "physical-registers-occupancy-and-SSA-peaks=post-materialization-diagnostics; "
             "symbolic-access-cosets-v1 (logical<=65536; non-dyadic bounded fallback); unmodeled=cache,overlap,dependency-latency,MMA-shared-read-bank-behavior,scalar-control";
    if (mode == LayoutBoundaryFrequencyMode::StaticTripCount)
      return "symbolic-subspace-proof-v1 with cta-staging-upper-bound-v1 fallback; "
             "pure-slice-byte-work-v1; static signed trip counts with unknown-loop factor-2 proxy; "
             "register/replication slots; shared root spans; "
             "sm90-shape-work-v1; warp-issue-32B-sectors-v1 (unknown base phase upper bound); "
             "shared-128B-phase-32x4B-distinct-word-conflicts-v2; symbolic-access-cosets-v1 (logical<=65536; non-dyadic bounded fallback); unmodeled=latency,cache,overlap";
    return "symbolic-subspace-proof-v1 with cta-staging-upper-bound-v1 fallback; "
           "pure-slice-byte-work-v1; loop-depth proxy; register/replication slots; shared root spans; "
           "sm90-shape-work-v1; warp-issue-32B-sectors-v1 (unknown base phase upper bound); "
           "shared-128B-phase-32x4B-distinct-word-conflicts-v2; symbolic-access-cosets-v1 (logical<=65536; non-dyadic bounded fallback); unmodeled=latency,cache,overlap";
  }
};

} // namespace

void enumerateSM90ReplicatedCandidates(const LayoutVar &,
                                      SmallVectorImpl<LayoutCandidate> &);
void SM90LayoutTarget::enumerateRelaxedCandidates(const LayoutConstraintGraph &,
    const LayoutVar &var, RelaxationKind kind,
    SmallVectorImpl<LayoutCandidate> &out) const {
  if (kind == RelaxationKind::Replication && var.kind == LayoutKind::Distributed)
    enumerateSM90ReplicatedCandidates(var, out);
}

std::unique_ptr<LayoutTarget> createSM90LayoutTarget(LayoutBoundaryFrequencyMode mode) {
  return std::make_unique<SM90LayoutTarget>(mode);
}

std::unique_ptr<LayoutTarget> createSM90LayoutTarget(
    LayoutBoundaryFrequencyMode mode, const SM90CostProfile &profile) {
  return std::make_unique<SM90LayoutTarget>(mode, profile);
}

} // namespace mlir::frisk
