#include "SelectionExperiment.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
using namespace mlir::frisk::selection;
int main(int argc,char **argv) {
  std::vector<std::string> args(argv+1,argv+argc);
  ExperimentOptions options; std::string error;
  if(mlir::failed(parseOptions(args,options,error))) {llvm::errs()<<error<<"\n";return 2;}
  if(options.listCases) {
    llvm::json::Array cases;for(const auto &spec:nativeCases()) cases.push_back(caseJSON(spec));
    llvm::outs()<<llvm::formatv("{0:2}\n",llvm::json::Value(llvm::json::Object{{"schema_version",1},{"cases",std::move(cases)}}));return 0;
  }
  options.command=std::vector<std::string>(argv,argv+argc);
  options.binaryPath=llvm::sys::fs::getMainExecutable(argv[0],reinterpret_cast<void *>(&main));
  mlir::MLIRContext context;context.disableMultithreading();
  auto result=runExperiment(context,options);
  llvm::outs()<<llvm::formatv("{0:2}\n",llvm::json::Value(std::move(result.record)));
  if(!result.error.empty()) llvm::errs()<<result.failureKind<<": "<<result.error<<"\n";
  return result.status=="failed" ? (result.fatal ? 2 : 1) : 0;
}
