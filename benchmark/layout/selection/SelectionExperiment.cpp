#include "SelectionExperiment.h"
#include "Dialect/Frisk/Analysis/LayoutRelaxation.h"
#include "Dialect/Frisk/Analysis/LayoutStatistics.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "Dialect/Frisk/Transforms/PassPipelines.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include <set>

using namespace mlir;
using namespace mlir::frisk;
namespace mlir::frisk::selection {
std::string sha256(llvm::StringRef text) {
  return llvm::toHex(llvm::SHA256::hash(llvm::arrayRefFromStringRef(text)),true);
}
LogicalResult parseOptions(llvm::ArrayRef<std::string> args,ExperimentOptions &result,std::string &error) {
  ExperimentOptions o; error.clear(); std::set<std::string> seen;
  auto reject=[&](llvm::Twine why){error=why.str();return failure();};
  std::string id="C1";
  for(size_t i=0;i<args.size();++i) {
    const auto &flag=args[i];
    if(!seen.insert(flag).second) return reject("duplicate option: "+flag);
    if(flag=="--list-cases") { o.listCases=true; continue; }
    if(flag=="--oracle") { o.oracle=true; continue; }
    if(flag!="--case" && flag!="--mode" && flag!="--policy" && flag!="--consumer" && flag!="--phase" && flag!="--output-dir" && flag!="--consumer-grid" && flag!="--restrict-grid")
      return reject("unknown option: "+flag);
    if(++i==args.size() || llvm::StringRef(args[i]).starts_with("--")) return reject("missing value for "+flag);
    llvm::StringRef value=args[i];
    if(flag=="--case") id=value.str();
    else if(flag=="--mode") {
      if(value!="ss" && value!="rs") return reject("mode must be ss or rs");
      o.spec.mode=value=="ss" ? MmaMode::SS : MmaMode::RS;
    } else if(flag=="--policy") o.spec.policy=value.str();
    else if(flag=="--consumer") {
      bool found=false;
      for(auto c:{Consumer::Store,Consumer::Reduce0,Consumer::Reduce1,Consumer::Fixed0,Consumer::Fixed1,Consumer::Local})
        if(value==consumerName(c)) {o.spec.consumer=c;found=true;break;}
      if(!found) return reject("unknown consumer");
    } else if(flag=="--phase") o.phase=value.str();
    else if(flag=="--output-dir") o.outputDir=value.str();
    else {
      auto pair=value.split(','); int64_t m=0,n=0;
      if(pair.first.getAsInteger(10,m) || pair.second.getAsInteger(10,n) || m<=0 || n<=0)
        return reject("consumer grid must be two positive integers M,N");
      if(flag=="--restrict-grid") o.restrictGrid={{m,n}};
      else o.spec.consumerGrid={{m,n}};
    }
  }
  if(o.listCases) {
    if(seen.size()!=1) return reject("--list-cases cannot be combined with experiment options");
    result=std::move(o); return success();
  }
  bool found=false;
  for(auto c:nativeCases()) if(c.id==id) {
    o.spec.id=c.id; o.spec.m=c.m; o.spec.n=c.n; o.spec.k=c.k; o.spec.threads=c.threads; found=true; break;
  }
  if(!found) return reject("unknown case id: "+id);
  if(o.phase!="native" && o.phase!="emit-input" && o.phase!="controlled" && o.phase!="local") return reject("unknown phase");
  bool controlled=o.phase=="controlled" || o.phase=="local";
  if(controlled && o.spec.policy!="Auto") return reject("controlled/local phases require Auto");
  if(!controlled && (o.oracle || o.restrictGrid)) return reject("oracle/restrict-grid require controlled/local phase");
  if(o.phase=="local") {
    if(seen.count("--consumer") && o.spec.consumer!=Consumer::Local) return reject("local phase requires local consumer");
    o.spec.consumer=Consumer::Local;
  }
  if(o.phase=="controlled" && o.spec.consumer==Consumer::Local) return reject("use local phase for local analysis");
  if(failed(validateCase(o.spec,error))) return failure();
  if(o.outputDir.empty()) return reject("--output-dir is required");
  result=std::move(o); return success();
}
namespace {
LogicalResult writeFile(llvm::StringRef path,llvm::StringRef data,std::string &error) {
  std::error_code ec; llvm::raw_fd_ostream out(path,ec);
  if(ec) {error=ec.message();return failure();}
  out<<data; out.close();
  if(out.has_error()) {error=out.error().message();out.clear_error();return failure();}
  return success();
}
std::optional<std::string> readFile(llvm::StringRef path) {
  auto file=llvm::MemoryBuffer::getFile(path);
  if(!file) return std::nullopt;
  return (*file)->getBuffer().str();
}
} // namespace
bool isFatalFailure(llvm::StringRef kind) {
  return kind!="empty-domain" && kind!="hard-conflict" && kind!="unsupported" &&
    kind!="unknown-proof" && kind!="search-budget" && kind!="search-incomplete" &&
    kind!="unsupported-input-form";
}

ExperimentFailure classifyFailure(llvm::StringRef kind, llvm::StringRef reason,
                                 llvm::StringRef diagnostics) {
  ExperimentFailure result;
  result.kind=kind.empty() ? "pipeline-failure-unclassified" : kind.str();
  result.reason=(reason.empty() ? diagnostics : reason).str();
  // Upstream uses Unsupported for these two internal consistency failures.
  // Preserve the actual typed kind, but never treat them as support samples.
  auto internalProofFailure=[](llvm::StringRef text) {
    return text.contains("selected layout proof failed") ||
      text.contains("propagation rejected a solver-supported graph");
  };
  bool internal=internalProofFailure(reason) || internalProofFailure(diagnostics);
  // Diagnostic classification is allowed only at candidate collection, whose
  // public API does not expose a typed Unknown status. It must not override
  // later solved-graph, materialization, verification, cost or I/O failures.
  if(!internal && (kind=="collection" || kind=="local-collection") &&
     (reason.contains("unknown proof") || diagnostics.contains("unknown proof")))
    result.kind="unknown-proof";
  result.fatal=internal || isFatalFailure(result.kind);
  return result;
}

LogicalResult publishResult(ExperimentResult &result,llvm::StringRef path) {
  auto refreshRecord=[&]() {
    result.record["status"]=result.status;result.record["fatal"]=result.fatal;
    result.record["failure"]=llvm::json::Object{{"kind",result.failureKind},{"reason",result.error}};
  };
  refreshRecord();
  auto json=llvm::formatv("{0:2}\n",llvm::json::Value(llvm::json::Object(result.record))).str();
  std::string error;
  if(failed(writeFile(path,json,error))) {
    result.status="failed";result.failureKind="result-write";result.error=error;result.fatal=true;
    refreshRecord();
    return failure();
  }
  return success();
}

FailureOr<llvm::json::Object> preparedTraceProjection(const llvm::json::Object &trace,uint64_t attempt) {
  if(trace.getBoolean("truncated")!=false) return failure();
  auto *events=trace.getArray("events");if(!events) return failure();
  llvm::json::Object result;
  for(const auto &event:*events) {
    auto *e=event.getAsObject();if(!e || e->getInteger("attempt")!=int64_t(attempt)) continue;
    auto name=e->getString("event");
    if(name!="graph" && !(name=="domains" && e->getString("stage")=="prepared")) continue;
    if(result.get(*name)) return failure();
    auto copy=*e;copy.erase("attempt");result[*name]=std::move(copy);
  }
  if(!result.get("graph") || !result.get("domains")) return failure();
  return result;
}

ExperimentResult runExperiment(MLIRContext &context,const ExperimentOptions &o) {
  ExperimentResult r;
  auto fail=[&](llvm::StringRef kind,llvm::StringRef why,bool fatal) {
    r.status="failed";r.failureKind=kind.str();r.error=why.str();r.fatal=fatal;
  };
  if(o.outputDir.empty()) {fail("output-directory","output directory is required",true);return r;}
  // IgnoreExisting=false makes this operation an exclusive directory claim.
  if(auto ec=llvm::sys::fs::create_directory(o.outputDir,false)) {
    fail("output-directory",ec.message(),true);return r;
  }
  auto path=[&](llvm::StringRef name){return o.outputDir+"/"+name.str();};
  llvm::json::Array command;for(const auto &arg:o.command) command.push_back(arg);
  r.record=llvm::json::Object{{"schema_version",1},{"kind","layout-selection"},
    {"spec",caseJSON(o.spec)},{"phase",o.phase},{"command",std::move(command)},
    {"source_digest",FRISK_SELECTION_SOURCE_DIGEST},{"config_digest",FRISK_SELECTION_CONFIG_DIGEST},
    {"build_type",FRISK_SELECTION_BUILD_TYPE},{"compiler",__VERSION__},
    {"output_abi_verified",false},{"replay_stable",false},
    {"frozen_domain",nullptr},{"oracle",nullptr}};
  auto executable=o.binaryPath.empty() ? llvm::sys::fs::getMainExecutable(nullptr,reinterpret_cast<void *>(&runExperiment)) : o.binaryPath;
  r.record["binary_path"]=executable;
  auto binary=readFile(executable);
  if(binary) r.record["binary_sha256"]=sha256(*binary);
  else fail("binary-identity","cannot read executable for SHA256",true);
  std::string diagnostics,error;
  ScopedDiagnosticHandler capture(&context,[&](Diagnostic &d){llvm::raw_string_ostream(diagnostics)<<d<<'\n';return success();});
  auto finish=[&]() -> ExperimentResult {
    if(failed(writeFile(path("diagnostics.txt"),diagnostics,error))) fail("artifact-write",error,true);
    llvm::json::Object hashes;
    for(llvm::StringRef name:{"input.mlir","output.mlir","replay.mlir","statistics.json","trace.json","replay-statistics.json","replay-trace.json","diagnostics.txt","source-manifest.txt","domains.json","branch-domains.json","preparation-input.mlir","materialized-pipeline-input.mlir","oracle.json","pipeline-statistics.json","native-snapshot-projection.json","oracle-output.mlir","oracle-materialized-pipeline-input.mlir","oracle-pipeline-statistics.json","oracle-replay-statistics.json","oracle-replay.mlir"})
      if(auto bytes=readFile(path(name))) hashes[name]=sha256(*bytes);
    r.record["artifacts_sha256"]=std::move(hashes);
    r.record["input_sha256"]=nullptr; r.record["output_sha256"]=nullptr;
    if(auto bytes=readFile(path("input.mlir"))) r.record["input_sha256"]=sha256(*bytes);
    if(auto bytes=readFile(path("output.mlir"))) r.record["output_sha256"]=sha256(*bytes);
    (void)publishResult(r,path("result.json"));
    return std::move(r);
  };
  auto manifest=readFile(FRISK_SELECTION_SOURCE_MANIFEST);
  if(!manifest || failed(writeFile(path("source-manifest.txt"),manifest ? *manifest : "",error))) {
    fail("source-identity","cannot preserve source manifest: "+error,true);return finish();
  }
  if(!r.failureKind.empty()) return finish();
  if(o.phase!="native" && o.phase!="emit-input" && o.phase!="controlled" && o.phase!="local") {
    fail("invalid-options","unsupported phase",true);return finish();
  }
  auto module=buildFixture(context,o.spec,error);
  if(!module) {fail("invalid-fixture",error,true);return finish();}
  bool fixedConsumer=o.spec.consumer==Consumer::Fixed0 || o.spec.consumer==Consumer::Fixed1;
  r.record["input_kind"]=fixedConsumer ? "pre-layout-analysis-fixture" : "native-pipeline-fixture";
  r.record["input_verification"]=fixedConsumer ? "native-unverified" : "structurally-verified";
  if(failed(writeFile(path("input.mlir"),printModule(*module),error))) {fail("input-write",error,true);return finish();}
  if(o.phase=="emit-input") {r.status="input-emitted";return finish();}
  if(o.phase=="controlled" || o.phase=="local") {
    runControlledExperiment(*module,o,r,diagnostics);return finish();
  }
  if(fixedConsumer) {
    diagnostics="Explicit ConvertLayout requires an encoded source before native normalization. "
      "This fixture intentionally leaves the MMA result unbound; use the future controlled "
      "collect/solve/materialize path. This is unsupported input form, not proven infeasibility.\n";
    fail("unsupported-input-form",diagnostics,false);return finish();
  }
  LayoutStatisticsReport stats;
  LayoutTraceOptions traceOptions;traceOptions.file=path("trace.json");
  auto target=createSM90LayoutTarget();
  if(o.spec.consumer==Consumer::Local) {
    LayoutTrace trace(traceOptions);trace.setSolverOptions({});
    auto graph=collectLayoutConstraints(*module,*target,LayoutCollectionMode::InitializeForRelaxation,&stats,&trace);
    if(failed(graph)) {
      auto classified=classifyFailure("local-collection","",diagnostics);
      fail(classified.kind,classified.reason,classified.fatal);return finish();
    }
    SmallVector<RelaxationRecord, 0> relaxations;
    auto solution=attemptControlledRelaxation(*graph,*target,{},relaxations,nullptr,&stats,&trace);
    if(failed(solution)) {
      auto classified=classifyFailure(stats.failureKind,stats.failureReason,diagnostics);
      fail(classified.kind,classified.reason,classified.fatal);return finish();
    }
    if(failed(materializeLayouts(*module,*graph,*solution)) || failed(verifyMaterializedLayouts(*module,*target))) {
      fail("actual-verification",diagnostics,true);return finish();
    }
    if(failed(writeLayoutStatistics(path("statistics.json"),stats,error)) || failed(trace.publish(error)) ||
       failed(writeFile(path("output.mlir"),printModule(*module),error))) {
      fail("artifact-write",error,true);return finish();
    }
    r.record["local_cost"]=layoutCostJSON(solution->cost);
    r.status="local-analysis-only";return finish();
  }
  PassManager pipeline(&context);
  buildFriskLayoutPipeline(pipeline,{},path("statistics.json"),&stats,traceOptions);
  if(failed(pipeline.run(*module))) {
    auto classified=classifyFailure(stats.failureKind,stats.failureReason,diagnostics);
    fail(classified.kind,classified.reason,classified.fatal);
    return finish();
  }
  if(failed(verify(*module)) || failed(verifyMaterializedLayouts(*module,*target)) || failed(verifyOutputABI(*module,o.spec,error))) {
    fail("actual-verification",error.empty() ? diagnostics : error,true);return finish();
  }
  r.record["output_abi_verified"]=true;
  auto output=printModule(*module);
  if(failed(writeFile(path("output.mlir"),output,error))) {fail("output-write",error,true);return finish();}
  auto replay=parseSourceString<ModuleOp>(output,&context);
  if(!replay) {fail("replay-parse",diagnostics,true);return finish();}
  PassManager replayPipeline(&context); LayoutStatisticsReport replayStats;
  LayoutTraceOptions replayTrace;replayTrace.file=path("replay-trace.json");
  buildFriskLayoutPipeline(replayPipeline,{},path("replay-statistics.json"),&replayStats,replayTrace);
  if(failed(replayPipeline.run(*replay)) || failed(verifyMaterializedLayouts(*replay,*target)) || failed(verifyOutputABI(*replay,o.spec,error))) {
    fail("replay-verification",error.empty() ? diagnostics : error,true);return finish();
  }
  auto replayText=printModule(*replay);
  if(failed(writeFile(path("replay.mlir"),replayText,error))) {fail("replay-write",error,true);return finish();}
  if(output!=replayText) {fail("replay-instability","full pipeline changed printed/parsed output",true);return finish();}
  r.record["replay_stable"]=true;
  auto traceText=readFile(path("trace.json"));
  auto trace=traceText ? llvm::json::parse(*traceText) : llvm::Expected<llvm::json::Value>(llvm::createStringError("missing trace"));
  if(!trace) {error=llvm::toString(trace.takeError());fail("trace-json",error,true);return finish();}
  auto *traceObject=trace->getAsObject();
  if(!traceObject || traceObject->getBoolean("truncated")!=false || traceObject->getBoolean("costExplanationComplete")!=true) {
    fail("trace-incomplete","cannot publish success without full trace/cost contributions",true);return finish();
  }
  r.record["cost"]=*traceObject->get("finalCost");
  llvm::json::Array solutions,contributions;
  auto costScope=traceObject->getInteger("costScope");
  auto projection=costScope ? preparedTraceProjection(*traceObject,*costScope) : FailureOr<llvm::json::Object>(failure());
  if(failed(projection)) {fail("native-snapshot-projection","missing complete final-attempt prepared snapshot",true);return finish();}
  auto projectionText=llvm::formatv("{0:2}\n",llvm::json::Value(std::move(*projection))).str();
  if(failed(writeFile(path("native-snapshot-projection.json"),projectionText,error))) {fail("artifact-write",error,true);return finish();}
  r.record["native_snapshot_projection"]=llvm::json::Object{{"sha256",sha256(projectionText)},
    {"artifact","native-snapshot-projection.json"},{"attempt",*costScope},{"scope","actual-native-final-attempt"}};
  if(auto *events=traceObject->getArray("events")) for(const auto &event:*events) {
    auto *e=event.getAsObject();if(!e) continue;
    if(e->getString("event")=="solution") solutions.push_back(event);
    if(e->getString("event")=="cost-contribution" && e->getInteger("costScope")==costScope) contributions.push_back(event);
  }
  r.record["solutions"]=std::move(solutions);r.record["cost_contributions"]=std::move(contributions);
  llvm::json::Array bindings;unsigned conversions=0;
  module->walk([&](ConvertLayoutOp){++conversions;});
  module->walk([&](MmaOp op){
    auto contract=op->getAttrOfType<MmaInstructionContractAttr>("frisk.mma_contract");
    if(!contract) return;
    std::string binding;llvm::raw_string_ostream(binding)<<contract;
    llvm::json::Array grid;for(auto value:contract.getPayload().getAs<DenseI64ArrayAttr>("grid").asArrayRef()) grid.push_back(value);
    bindings.push_back(llvm::json::Object{{"contract",binding},{"grid",std::move(grid)}});
  });
  if(bindings.size()!=1) {fail("missing-live-mma","successful pipeline must preserve exactly one bound MMA",true);return finish();}
  r.record["bindings"]=std::move(bindings);r.record["actual_convert_layout_count"]=conversions;
  r.record["statistics"]=stats.toJSON();r.status="success";
  return finish();
}
} // namespace mlir::frisk::selection
