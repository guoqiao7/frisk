#ifndef FRISK_PASSES
#define FRISK_PASSES

include "mlir/Pass/PassBase.td"

def NormalizeLayoutIR : Pass<"frisk-normalize-layout-ir", "::mlir::ModuleOp"> {
  let summary = "Transactionally import legacy fragment lifetimes into tensor SSA";
  let constructor = "mlir::frisk::createNormalizeLayoutIRPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect", "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect", "::mlir::scf::SCFDialect", "::mlir::func::FuncDialect"];
}

def FriskInferLayouts : Pass<"frisk-infer-layouts", "::mlir::ModuleOp"> {
  let summary = "Infer and materialize Frisk layouts";
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
  let options = [
    Option<"costProfileFile", "cost-profile", "std::string", "\"\"", "Read an explicit SM90 calibrated cost profile JSON">,
    Option<"boundaryFrequency", "boundary-frequency", "std::string", "\"legacy\"", "Boundary frequency: legacy or static-trip-count">,
    Option<"memoizeCosts", "memoize-costs", "bool", "true", "Reuse bounded graph-local pure cost queries">,
    Option<"seedBeamSolution", "seed-beam-solution", "bool", "false", "Generate budgeted coherent-layout witnesses before beam search">,
    Option<"conversionPolicy", "conversion-policy", "std::string", "\"full\"", "Conversion policy: full, shared-only or independent">,
    Option<"profileFile", "profile-file", "std::string", "\"\"", "Atomically write invocation-local inclusive layout profile JSON">,
    Option<"traceFile", "trace-file", "std::string", "\"\"", "Atomically write layout decision trace JSON">,
    Option<"traceMaxEvents", "trace-max-events", "uint64_t", "4096", "Maximum trace events">,
    Option<"traceMaxBytes", "trace-max-bytes", "uint64_t", "8388608", "Maximum trace bytes (at least 4096)">,
    Option<"statsFile", "stats-file", "std::string", "\"\"",
           "Atomically write attempt-local layout statistics JSON">,
    Option<"analysisOnly", "analysis-only", "bool", "false",
           "Collect, solve and verify layouts without modifying IR">,
    Option<"dumpAnalysis", "dump-analysis", "bool", "false",
           "Print the deterministic graph, domain sizes and conversion decisions">,
    Option<"exactCombinationLimit", "exact-combination-limit", "uint64_t", "256",
           "Maximum domain product for exact layout search">,
    Option<"beamWidth", "beam-width", "unsigned", "32", "Layout search beam width">,
    Option<"maxExpandedStates", "max-expanded-states", "uint64_t", "65536",
           "Per-component attempted candidate expansion budget">,
    Option<"refineBeamSolution", "refine-beam-solution", "bool", "true",
           "Improve complete beam solutions by single-variable replacements">,
    Option<"maxRefinementProbes", "max-refinement-probes", "uint64_t", "256",
           "Per-component refinement budget within the total expansion budget">
  ];
}

def OptimizeLayoutConversions : Pass<"frisk-optimize-layout-conversions", "::mlir::ModuleOp"> {
  let summary = "Fold and compose type-safe distributed layout conversions";
  let options = [Option<"costProfileFile", "cost-profile", "std::string", "\"\"", "Read an explicit SM90 calibrated cost profile JSON">,
    Option<"boundaryFrequency", "boundary-frequency", "std::string", "\"legacy\"", "Boundary frequency: legacy or static-trip-count">];
  let constructor = "mlir::frisk::createOptimizeLayoutConversionsPass()";
}

def TestLowerLayoutConversions : Pass<"frisk-test-lower-layout-conversions", "::mlir::ModuleOp"> {
  let summary = "Test-only static single-CTA layout redistribution adapter";
  let constructor = "mlir::frisk::createTestLowerLayoutConversionsPass()";
  let options = [Option<"scratchBudget", "scratch-budget", "uint64_t", "49152",
    "Conservative adapter workgroup byte budget (not the SM90 maximum)">,
    Option<"executable", "executable", "bool", "false",
    "Lower verified rank-one global tile load/conversion/store kernels without tensor bridges">];
}

def ExportExecution : Pass<"frisk-export-execution", "::mlir::ModuleOp"> {
 let summary = "Export bounded checked execution maps and reduction proofs";
 let constructor = "mlir::frisk::createExportExecutionPass()";
 let options = [Option<"output", "output", "std::string", "\"\"", "Atomic JSON output path">,
   Option<"minify", "minify", "bool", "false", "Write execution JSON without indentation">,
   Option<"networkObjective", "network-objective", "std::string", "\"min-shfl\"", "Verified Shuffle objective: min-shfl or min-work">];
}

#endif // FRISK_PASSES
