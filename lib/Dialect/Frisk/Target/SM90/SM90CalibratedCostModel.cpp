#include "Dialect/Frisk/Target/SM90/SM90CostModel.h"
#include "Dialect/Frisk/Analysis/LayoutCostModel.h"
#include "Dialect/Frisk/Analysis/LayoutDemand.h"
#include "Dialect/Frisk/Analysis/LayoutExperimentPolicy.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Analysis/InstructionLayoutConstraints.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <limits>
#include <set>

using namespace mlir;
using namespace mlir::frisk;
namespace {
// The frozen coefficient file format is independent of cost semantics. Keep
// existing v2 profiles and hashes stable while versioning symbolic conversion
// routes, which can improve a previously conservative, nonzero estimate.
constexpr StringRef profileModel = "sm90-layout-calibrated-v2";
constexpr StringRef model = "sm90-layout-calibrated-v3";
const StringRef names[] = {
    "access_warp_issues", "global_sectors32", "shared_bank_excess",
    "copy_setup_units", "mma_instructions", "reduction_element_ops",
    "alu_warp_issues", "div_warp_issues", "exp_warp_issues", "cast_warp_issues",
    "conversion_register_bytes", "conversion_shuffle_bytes",
    "conversion_shared_bytes", "conversion_conservative_bytes",
    "synchronization_barriers"};
const StringRef demandBankNames[] = {"shared_read_bank_excess","shared_write_bank_excess"};
const StringRef loadIssueGroupName = "global_load_issue_groups";
const StringRef communicationNames[] = {"shared_warp_issues","shuffle_warp_issues","communication_alu_warp_issues"};
// Keep the historical group slot17; v6 skips it and uses18..20. Public v6
// profile keys are the old17 followed by the three communication channels.
constexpr unsigned featureStorageCount=21;
using Features = std::array<uint64_t, featureStorageCount>;
StringRef featureName(unsigned i) {return i<kSM90CostFeatureCount?names[i]:(i<kSM90DemandCostFeatureCount?demandBankNames[i-kSM90CostFeatureCount]:(i==17?loadIssueGroupName:communicationNames[i-18]));}
constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
uint64_t add(uint64_t a, uint64_t b, bool &saturated) {
  if (a > maximum - b) { saturated = true; return maximum; }
  return a + b;
}
uint64_t mul(uint64_t a, uint64_t b, bool &saturated) {
  if (b && a > maximum / b) { saturated = true; return maximum; }
  return a * b;
}
void weight(Features &features, LayoutBoundaryFrequency f, bool &saturated) {
  bool nonzero = false;
  for (auto &n : features) { n = mul(n, f.factor, saturated); nonzero |= n != 0; }
  saturated |= f.saturated && nonzero;
}
CostEstimate score(CostEstimate proxy, const Features &features,
                   const SM90CostProfile &profile, bool saturated = false) {
  proxy.features.clear();
  proxy.cost.instructionPathAndWork = 0;
  proxy.saturated |= saturated;
  unsigned count=profile.survivingDemand?featureStorageCount:(profile.executionDemand?(profile.executionOrder?kSM90OrderedDemandCostFeatureCount:kSM90DemandCostFeatureCount):kSM90CostFeatureCount);
  for (unsigned i = 0; i < count; ++i) {
    if(profile.survivingDemand && i==17)continue;
    StringRef name=featureName(i);
    uint64_t coefficient=i<kSM90CostFeatureCount?profile.coefficients[i]:(i<kSM90DemandCostFeatureCount?profile.demandBankCoefficients[i-kSM90CostFeatureCount]:(i==17?profile.loadIssueGroupCoefficient:profile.communicationCoefficients[i-18]));
    if (features[i]) proxy.features.emplace_back(name.str(), features[i]);
    uint64_t contribution = mul(features[i], coefficient, proxy.saturated);
    proxy.cost.instructionPathAndWork = add(proxy.cost.instructionPathAndWork,
                                          contribution, proxy.saturated);
  }
  return proxy;
}
llvm::Error invalid(StringRef reason) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s", reason.str().c_str());
}

/// Per-warp scalar issue work of the actual scalarizing adapter. This is not
/// logical element work and includes replicated carrier execution. No claim
/// about folding, instruction scheduling or packed native half arithmetic.
FailureOr<Features> arithmetic(Operation *op, Attribute candidate,
                               LayoutBoundaryFrequencyMode mode,
                               Operation *at, bool &saturated) {
  Features features{};
  if (!op) return failure();
  StringRef dialect = op->getName().getDialectNamespace();
  if (dialect != "arith" && dialect != "math") return features;
  if (op->hasTrait<OpTrait::ConstantLike>()) return features;
  if (!op->hasTrait<OpTrait::Elementwise>() || op->getNumResults() != 1)
    return failure();
  auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  auto encoding = dyn_cast_or_null<DistributedEncodingAttr>(candidate);
  if (!type || !type.hasStaticShape() || !encoding ||
      failed(encoding.verifyForType(type, op->getLoc()))) return failure();
  auto topology = encoding.getTopology().asArrayRef();
  if (topology.size() != 5 || topology[1] != 32 || topology[4] != 1)
    return failure();
  for (Value value : op->getResults()) {
    auto t = cast<RankedTensorType>(value.getType());
    unsigned bits = t.getElementTypeBitWidth();
    if ((!t.getElementType().isIntOrFloat()) || bits == 0 || bits > 32)
      return failure();
  }
  for (Value value : op->getOperands())
    if (auto t = dyn_cast<ShapedType>(value.getType()))
      if (!t.getElementType().isIntOrFloat() || t.getElementTypeBitWidth() > 32)
        return failure();
  StringRef name = op->getName().getStringRef();
  int category = llvm::StringSwitch<int>(name)
      .Cases("arith.addf", "arith.subf", "arith.mulf", 6)
      .Cases("arith.addi", "arith.subi", "arith.muli", 6)
      .Cases("arith.andi", "arith.ori", "arith.xori", 6)
      .Cases("arith.maximumf", "arith.minimumf", "arith.maxnumf", 6)
      .Cases("arith.minnumf", "arith.cmpf", "arith.cmpi", 6)
      .Cases("arith.select", "arith.negf", "arith.shli", 6)
      .Cases("arith.shrsi", "arith.shrui", 6)
      .Case("arith.divf", 7)
      .Cases("math.exp", "math.exp2", 8)
      .Cases("arith.extf", "arith.truncf", "arith.extsi", 9)
      .Cases("arith.extui", "arith.trunci", "arith.sitofp", 9)
      .Cases("arith.uitofp", "arith.fptosi", "arith.fptoui", 9)
      .Case("arith.bitcast", 9).Default(-1);
  if (category < 0) return failure(); // Unsupported operations are not free.
  uint64_t issues = 1;
  for (unsigned dimension : {0u, 2u, 3u}) {
    if (topology[dimension] <= 0) return failure();
    issues = mul(issues, topology[dimension], saturated);
  }
  features[category] = issues;
  weight(features, getLayoutBoundaryFrequency(at ? at : op, mode), saturated);
  return features;
}
struct Charge { std::string kind; uint64_t owner; CostEstimate estimate; };
class Charges final : public LayoutCostObserver {
public:
  SmallVector<Charge> charges;
  void recordCost(StringRef kind, uint64_t owner, const CostEstimate &cost) override {
    charges.push_back({kind.str(), owner, cost});
  }
};
} // namespace

ArrayRef<StringRef> mlir::frisk::getSM90CostFeatureNames() { return names; }
ArrayRef<StringRef> mlir::frisk::getSM90DemandCostFeatureNames() {
 static const SmallVector<StringRef> all=[] {SmallVector<StringRef> result;llvm::append_range(result,names);llvm::append_range(result,demandBankNames);return result;}();return all;
}
ArrayRef<StringRef> mlir::frisk::getSM90OrderedDemandCostFeatureNames() {
 static const SmallVector<StringRef> all=[] {SmallVector<StringRef> result;llvm::append_range(result,getSM90DemandCostFeatureNames());result.push_back(loadIssueGroupName);return result;}();return all;
}
ArrayRef<StringRef> mlir::frisk::getSM90SurvivingCostFeatureNames() {
 static const SmallVector<StringRef> all=[] {SmallVector<StringRef> result;llvm::append_range(result,getSM90DemandCostFeatureNames());llvm::append_range(result,communicationNames);return result;}();return all;
}
LogicalResult mlir::frisk::verifySM90CostProfile(const SM90CostProfile &p) {
  if((p.networkProxy && !p.survivingDemand) ||
     (!p.networkProxy && p.networkObjective!=shuffle_network::Objective::MinShuffle))return failure();
  if(p.networkObjective!=shuffle_network::Objective::MinShuffle && p.networkObjective!=shuffle_network::Objective::MinWork)return failure();
  if (p.provenance.empty() || p.provenance.size() > 16384 ||
      p.provenance.find('\0') != std::string::npos ||
      (!llvm::any_of(p.coefficients, [](uint64_t n) { return n != 0; }) &&
       !(p.executionDemand && llvm::any_of(p.demandBankCoefficients,[](uint64_t n){return n!=0;})) &&
       !(p.executionDemand && p.executionOrder && p.loadIssueGroupCoefficient)))
    return failure();
  auto bounded=[](uint64_t n){return n<=uint64_t(std::numeric_limits<int64_t>::max());};
  if(p.survivingDemand && (!p.executionDemand || !p.executionOrder || p.loadIssueGroupCoefficient ||
      p.coefficients[2] || !p.coefficients[14] ||
      llvm::any_of(ArrayRef<uint64_t>(p.coefficients).slice(10,4),[](uint64_t n){return n!=0;}) ||
      !llvm::all_of(p.communicationCoefficients,[&](uint64_t n){return n>0 && bounded(n);})))return failure();
  return success((!p.executionOrder || p.executionDemand) && llvm::all_of(p.coefficients,bounded) && (!p.executionDemand || llvm::all_of(p.demandBankCoefficients,bounded)) && (!p.executionOrder || bounded(p.loadIssueGroupCoefficient)));
}
std::string mlir::frisk::serializeSM90CostProfile(const SM90CostProfile &p) {
  llvm::json::Object coefficients;
  for (unsigned i = 0; i < kSM90CostFeatureCount; ++i)
    coefficients[names[i]] = p.coefficients[i];
  if(p.executionDemand)for(unsigned i=0;i<2;++i)coefficients[demandBankNames[i]]=p.demandBankCoefficients[i];
  if(p.executionDemand && p.executionOrder && !p.survivingDemand)coefficients[loadIssueGroupName]=p.loadIssueGroupCoefficient;
  if(p.survivingDemand)for(unsigned i=0;i<3;++i)coefficients[communicationNames[i]]=p.communicationCoefficients[i];
  llvm::json::Object object{{"schema_version", 1}, {"model", p.networkProxy?StringRef("sm90-layout-network-surviving-proxy-v1"):(p.survivingDemand?StringRef("sm90-layout-surviving-calibrated-v1"):(p.executionDemand ? (p.executionOrder?StringRef("sm90-layout-demand-calibrated-v2"):StringRef("sm90-layout-demand-calibrated-v1")) : profileModel))},
      {"coefficients", std::move(coefficients)}, {"provenance", p.provenance}};
  if(p.networkProxy) {
    object["network_objective"]=p.networkObjective==shuffle_network::Objective::MinWork?"min-work":"min-shfl";
    object["weight_provenance"]="migrated-engineering-proxy-not-refit";
  }
  return llvm::formatv("{0}", llvm::json::Value(std::move(object))).str();
}
llvm::Expected<SM90CostProfile> mlir::frisk::parseSM90CostProfile(StringRef text) {
  if (text.size() > 65536) return invalid("calibrated profile exceeds bounded input size");
  // The schema has two object levels. Bound nesting before LLVM's recursive
  // parser sees malformed or adversarial configuration text.
  unsigned depth = 0;
  bool inString = false, escaped = false;
  for (char c : text) {
    if (inString) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') inString = false;
    } else if (c == '"') inString = true;
    else if (c == '{' || c == '[') {
      if (++depth > 4) return invalid("calibrated profile exceeds bounded nesting");
    } else if (c == '}' || c == ']') {
      if (!depth) return invalid("invalid calibrated profile nesting");
      --depth;
    }
  }
  auto value = llvm::json::parse(text);
  if (!value) return value.takeError();
  if (text.size() > 65536) return invalid("calibrated profile exceeds bounded input size");
  // LLVM JSON objects overwrite duplicate keys. Reject them before schema
  // validation, including spellings made equal by JSON escape decoding.
  SmallVector<std::set<std::string>> objectKeys;
  for (size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '{') objectKeys.emplace_back();
    else if (text[i] == '}') objectKeys.pop_back();
    else if (text[i] == '"') {
      size_t begin = i++;
      while (i < text.size()) {
        if (text[i] == '\\') { i += 2; continue; }
        if (text[i] == '"') break;
        ++i;
      }
      size_t after = i + 1;
      while (after < text.size() && llvm::isSpace(text[after])) ++after;
      if (after < text.size() && text[after] == ':') {
        auto key = llvm::json::parse(text.slice(begin, i + 1));
        if (!key) return key.takeError();
        if (objectKeys.empty() || !key->getAsString() ||
            !objectKeys.back().insert(key->getAsString()->str()).second)
          return invalid("duplicate calibrated profile key");
      }
    }
  }
  auto *object = value->getAsObject();
  const auto *version = object ? object->get("schema_version") : nullptr;
  bool networkModel=object && object->getString("model")=="sm90-layout-network-surviving-proxy-v1";
  if (!object || object->size() != (networkModel?6:4) || !version || version->getAsUINT64() != uint64_t{1} ||
      (object->getString("model") != profileModel && object->getString("model") != "sm90-layout-demand-calibrated-v1" && object->getString("model") != "sm90-layout-demand-calibrated-v2" && object->getString("model") != "sm90-layout-surviving-calibrated-v1" && !networkModel))
    return invalid("invalid or unknown calibrated profile schema/model/fields");
  for (const auto &[key, ignored] : *object)
    if (key != "schema_version" && key != "model" && key != "coefficients" && key != "provenance" && !(networkModel && (key=="network_objective" || key=="weight_provenance")))
      return invalid("unknown calibrated profile field");
  auto *coefficients = object->getObject("coefficients");
  auto provenance = object->getString("provenance");
  if (!coefficients || coefficients->size() != ((networkModel || object->getString("model")=="sm90-layout-surviving-calibrated-v1")?kSM90SurvivingCostFeatureCount:(object->getString("model")=="sm90-layout-demand-calibrated-v2"?kSM90OrderedDemandCostFeatureCount:(object->getString("model")=="sm90-layout-demand-calibrated-v1"?kSM90DemandCostFeatureCount:kSM90CostFeatureCount))) || !provenance)
    return invalid("missing calibrated feature coefficients or provenance");
  SM90CostProfile profile;
  profile.networkProxy=networkModel;
  if(networkModel) {
    auto objective=object->getString("network_objective");
    if((objective!="min-shfl" && objective!="min-work") || object->getString("weight_provenance")!="migrated-engineering-proxy-not-refit")return invalid("network proxy requires an explicit objective and non-refit provenance");
    profile.networkObjective=objective=="min-work"?shuffle_network::Objective::MinWork:shuffle_network::Objective::MinShuffle;
  }
  profile.survivingDemand=networkModel || object->getString("model")=="sm90-layout-surviving-calibrated-v1";
  profile.executionOrder = profile.survivingDemand || object->getString("model") == "sm90-layout-demand-calibrated-v2";
  profile.executionDemand = profile.executionOrder || object->getString("model") == "sm90-layout-demand-calibrated-v1";
  profile.provenance = provenance->str();
  for (unsigned i = 0; i < kSM90CostFeatureCount; ++i) {
    auto *value = coefficients->get(names[i]);
    auto n = value ? value->getAsUINT64() : std::nullopt;
    if (!n || *n > uint64_t(std::numeric_limits<int64_t>::max()))
      return invalid("feature coefficient must be a nonnegative signed-64 integer");
    profile.coefficients[i] = *n;
  }
  if(profile.executionDemand)for(unsigned i=0;i<2;++i){auto *value=coefficients->get(demandBankNames[i]);auto n=value?value->getAsUINT64():std::nullopt;if(!n || *n>uint64_t(std::numeric_limits<int64_t>::max()))return invalid("demand bank coefficient must be a nonnegative signed-64 integer");profile.demandBankCoefficients[i]=*n;}
  if(profile.executionOrder && !profile.survivingDemand) {
    auto *value=coefficients->get(loadIssueGroupName);auto n=value?value->getAsUINT64():std::nullopt;
    if(!n || *n>uint64_t(std::numeric_limits<int64_t>::max()))return invalid("load issue group coefficient must be a nonnegative signed-64 integer");
    profile.loadIssueGroupCoefficient=*n;
  }
  if(profile.survivingDemand)for(unsigned i=0;i<3;++i) {
    auto *value=coefficients->get(communicationNames[i]);auto n=value?value->getAsUINT64():std::nullopt;
    if(!n || !*n || *n>uint64_t(std::numeric_limits<int64_t>::max()))return invalid("surviving communication coefficients must be positive signed-64 integers");
    profile.communicationCoefficients[i]=*n;
  }
  if (failed(verifySM90CostProfile(profile)))
    return invalid("profile needs bounded provenance/coefficients; v6 requires positive communication/barrier floors and fixed-zero byte/aggregate-bank channels");
  return profile;
}
llvm::Expected<SM90CostProfile> mlir::frisk::loadSM90CostProfileFile(StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) return llvm::errorCodeToError(buffer.getError());
  return parseSM90CostProfile((*buffer)->getBuffer());
}
std::string mlir::frisk::getSM90CostProfileSHA256(const SM90CostProfile &p) {
  std::string canonical = serializeSM90CostProfile(p);
  auto digest = llvm::SHA256::hash(ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(canonical.data()), canonical.size()));
  return llvm::toHex(digest, true);
}
std::string mlir::frisk::getSM90CalibratedCostModelID(const SM90CostProfile &p,
    LayoutBoundaryFrequencyMode mode) {
  return ((p.networkProxy?StringRef("sm90-layout-network-v1"):(p.survivingDemand?StringRef("sm90-layout-calibrated-v6d"):(p.executionDemand ? (p.executionOrder?StringRef("sm90-layout-calibrated-v5d"):StringRef("sm90-layout-calibrated-v4d")) : model))) + "-" + getSM90CostProfileSHA256(p).substr(0, 32) +
          (mode == LayoutBoundaryFrequencyMode::Legacy ? "-leg" : "-stc")).str();
}

FailureOr<CostEstimate> mlir::frisk::evaluateSM90CalibratedLayoutCost(
    const LayoutConstraintGraph &graph, const CandidateAssignment &assignment,
    const SM90CostProfile &profile, LayoutCostObserver *observer,
    LayoutBoundaryFrequencyMode mode) {
  if (failed(verifySM90CostProfile(profile))) return failure();
  Charges raw;
  if (failed(evaluateSM90LayoutCost(graph, assignment, &raw, mode))) return failure();
  CostEstimate total;
  auto append = [&](StringRef kind, uint64_t owner, CostEstimate cost) {
    total = addLayoutCosts(total, cost);
    if (observer) observer->recordCost(kind, owner, cost);
  };
  for (const auto &charge : raw.charges) {
    Features features{};
    auto &c = charge.estimate.cost;
    if (charge.kind == "access") {
      features[0] = c.instructionPathAndWork;
      features[1] = c.memoryTransactions;
      features[2] = c.bankConflictDegree;
    } else if (charge.kind == "copy-setup") {
      features[3] = c.instructionPathAndWork;
      const auto &constraint = graph.getConstraint(charge.owner);
      auto *op = constraint.instruction ? constraint.instruction->source : nullptr;
      auto tuple = findInstructionSupport(graph, constraint, assignment.values);
      auto binding = tuple ? dyn_cast<DictionaryAttr>(tuple->binding) : DictionaryAttr{};
      auto path = binding ? binding.getAs<StringAttr>("path") : StringAttr{};
      if (!op || op->getNumOperands() < 2 || !path) return failure();
      auto src = dyn_cast<MemRefType>(op->getOperand(0).getType());
      auto dst = dyn_cast<MemRefType>(op->getOperand(1).getType());
      if (!src || !dst) return failure();
      // Matches the current single-stage scalar/vector, cp.async and TMA
      // adapters: one publication barrier, plus TMA's initialization barrier.
      // mbarrier polling, proxy fences and MMA waits are NOT CTA barriers.
      if (getFriskMemorySpace(src) == attr::MemorySpace::Global &&
          getFriskMemorySpace(dst) == attr::MemorySpace::Shared)
        features[14] = path.getValue() == "tma" ? 2 : 1;
    }
    else if (charge.kind == "mma") features[4] = c.instructionPathAndWork;
    else if (charge.kind == "reduction") features[5] = c.instructionPathAndWork;
    else if (charge.kind != "registers" && charge.kind != "shared-root") return failure();
    bool saturated = false;
    if (charge.kind != "registers" && charge.kind != "shared-root") {
      if (charge.owner >= graph.getConstraints().size()) return failure();
      auto &constraint = graph.getConstraint(charge.owner);
      Operation *source = constraint.instruction ? constraint.instruction->source :
          constraint.reduction ? constraint.reduction->source : nullptr;
      if (!source && constraint.provenance < graph.getProvenances().size())
        source = graph.getProvenances()[constraint.provenance].source;
      weight(features, getLayoutBoundaryFrequency(source, mode), saturated);
    }
    append(charge.kind, charge.owner, score(charge.estimate, features, profile, saturated));
  }
  for (const auto &var : graph.getVariables()) {
    if (var.kind != LayoutKind::Distributed || var.use || !var.value ||
        !assignment.values.count(var.id)) continue;
    Operation *op = var.value.getDefiningOp();
    if (!op) continue;
    auto dialect = op->getName().getDialectNamespace();
    if (dialect != "arith" && dialect != "math") continue;
    bool saturated = false;
    auto features = arithmetic(op, assignment.values.lookup(var.id), mode, op, saturated);
    if (failed(features)) return failure();
    append("arithmetic", var.id, score({}, *features, profile, saturated));
  }
  return total;
}
FailureOr<CostEstimate> mlir::frisk::adjustSM90CalibratedConversionCost(
    const CostEstimate &proxy, const LayoutConversionEdge &edge,
    const SM90CostProfile &profile) {
  if (failed(verifySM90CostProfile(profile))) return failure();
  Features features{};
  switch (edge.exchange) {
  case ConversionExchangeKind::Identity: break;
  case ConversionExchangeKind::Register: features[10] = edge.bytes; break;
  case ConversionExchangeKind::Shuffle: features[11] = edge.bytes; break;
  case ConversionExchangeKind::Shared: features[12] = edge.bytes; break;
  case ConversionExchangeKind::Conservative: features[13] = edge.bytes; break;
  }
  features[14] = edge.synchronizationCost;
  bool saturated = false;
  weight(features, {edge.frequencyFactor, edge.frequencyUsesProxy, edge.frequencySaturated}, saturated);
  return score(proxy, features, profile, saturated);
}
FailureOr<CostEstimate> mlir::frisk::adjustSM90CalibratedCloneCost(
    const CostEstimate &proxy, const LayoutVar &result, Attribute encoding,
    Operation *operation, Operation *insertionPoint, const SM90CostProfile &profile,
    LayoutBoundaryFrequencyMode mode) {
  if (failed(verifySM90CostProfile(profile)) || !operation ||
      operation->getNumResults() != 1 || operation->getResult(0).getType() != result.shapedType)
    return failure();
  bool saturated = false;
  auto features = arithmetic(operation, encoding, mode, insertionPoint, saturated);
  if (failed(features)) return failure();
  return score(proxy, *features, profile, saturated);
}

StringRef mlir::frisk::getSM90DemandCostModelID() { return "sm90-layout-demand-v1"; }
FailureOr<CostEstimate> mlir::frisk::evaluateSM90DemandLayoutCost(
    const LayoutConstraintGraph &graph,const CandidateAssignment &assignment,
    const ExperimentPolicy &policy,const SM90CostProfile *profile,
    LayoutCostObserver *observer,LayoutBoundaryFrequencyMode mode) {
  if(policy.conversion!=ConversionPolicy::Independent ||
     (!policy.scope.fullGraph && !policy.scope.informedLocal) ||
     !policy.scope.producerInstructions.empty())return failure();
  if(profile && (!profile->executionDemand || failed(verifySM90CostProfile(*profile))))return failure();
  auto roots=policy.scope.fullGraph?ArrayRef<LayoutVarID>{}:ArrayRef<LayoutVarID>(policy.scope.producers);
  bool surviving=profile && profile->survivingDemand;LayoutDemandLedger ledger;
  if(surviving) {
    auto analysis=analyzeLayoutSurvivingDemand(graph,assignment,roots,65536,profile->networkObjective);
    if(!analysis.proven)return failure();ledger=std::move(analysis.surviving);
  } else ledger=analyzeLayoutDemand(graph,assignment,roots,65536,!profile || profile->executionOrder,profile?profile->networkObjective:shuffle_network::Objective::MinShuffle);
  if(!ledger.exact)return failure();
  CostEstimate total;
  auto append=[&](StringRef kind,uint64_t owner,CostEstimate raw,Features f) {
    if(profile)raw=score(raw,f,*profile);
    else for(unsigned i=0;i<kSM90OrderedDemandCostFeatureCount;++i)if(f[i])raw.features.emplace_back(featureName(i).str(),f[i]);
    total=addLayoutCosts(total,raw);if(observer)observer->recordCost(kind,owner,raw);
  };
  for(const auto &[id,access]:ledger.accesses) {
    Features f{};f[0]=access.warpIssues;f[1]=access.globalSectors32;f[2]=access.sharedBankExcess;f[6]=access.addressAluWarpIssues;
    CostEstimate raw;raw.cost.instructionPathAndWork=f[0]+f[6];raw.cost.memoryTransactions=f[1];raw.cost.bankConflictDegree=f[2];
    append("access",id,raw,f);
  }
  for(const auto &[id,value]:ledger.values) {
    const auto &var=graph.getVariable(id);if(var.use || !var.value)continue;auto *op=var.value.getDefiningOp();
    CostEstimate resource;resource.cost.spillRiskAndRegisters=value.queries.size();append("registers",id,resource,{});
    if(!op)continue;auto dialect=op->getName().getDialectNamespace();if(dialect!="arith" && dialect!="math")continue;
    bool saturated=false;auto f=arithmetic(op,assignment.values.lookup(id),mode,op,saturated);if(failed(f))return failure();
    auto topology=cast<DistributedEncodingAttr>(assignment.values.lookup(id)).getTopology();
    uint64_t original=uint64_t(topology[0])*topology[2]*topology[3];
    // Demand lowering emits one mov.b32 for each queried scalar splat.
    // Keep the legacy arithmetic helper's constant semantics unchanged.
    if(op->hasTrait<OpTrait::ConstantLike>())(*f)[6]=uint64_t(value.queries.size())*4;
    for(auto &count:*f)if(count)count=uint64_t(value.queries.size())*4;
    CostEstimate raw;for(unsigned category=6;category<10;++category)raw.cost.instructionPathAndWork+=(*f)[category];
    raw.saturated=saturated;append("arithmetic",id,raw,*f);(void)original;
  }
  for(const auto &[id,b]:ledger.boundaries) {
    Features f{};
    if(surviving){f[18]=b.sharedReadWarpIssues+b.sharedWriteWarpIssues;f[19]=b.shuffleWarpIssues;f[20]=b.addressAluWarpIssues;}
    else f[6]=b.addressAluWarpIssues+b.shuffleWarpIssues;
    if(b.exchange==ConversionExchangeKind::Shuffle)f[11]=b.bytes;
    if(b.exchange==ConversionExchangeKind::Shared)f[12]=b.bytes;
    f[14]=b.barriers;f[2]=b.sharedBankExcess;f[15]=b.sharedReadBankExcess;f[16]=b.sharedWriteBankExcess;CostEstimate raw;raw.cost.instructionPathAndWork=f[6]+f[18]+f[19]+f[20];
    raw.cost.bankConflictDegree=b.sharedBankExcess;raw.cost.conversionBytesAndSync=b.bytes+512*b.barriers;raw.cost.codeSize=1;
    append("conversion",id,raw,f);
  }
  if(!surviving && (!profile || profile->executionOrder) && ledger.globalLoadIssueGroups) {
    Features f{};f[17]=ledger.globalLoadIssueGroups;CostEstimate raw;
    raw.cost.instructionPathAndWork=f[17];append("global-load-issue-groups",0,raw,f);
  }
  if(ledger.scratchMaxBytes) {
    CostEstimate scratch;scratch.cost.sharedBytesAndOccupancy=ledger.scratchMaxBytes;
    append("demand-shared-scratch",ledger.boundaries.empty()?0:ledger.boundaries.begin()->first,scratch,{});
  }
  return total;
}
