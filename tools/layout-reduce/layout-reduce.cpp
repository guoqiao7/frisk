#include "LayoutFailureCase.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
using namespace mlir;
using namespace mlir::frisk::testing;
int main(int argc, char **argv) {
  if (argc!=3 || (StringRef(argv[1])!="--round-trip" && StringRef(argv[1])!="--reduce" && StringRef(argv[1])!="--replay")) {
    llvm::errs()<<"usage: layout-reduce --round-trip|--replay|--reduce FILE\n"
      "Strict finite-same-require-v1 fixtures only; SSA/use/alias/region/conversion payloads unsupported.\n";
    return 2;
  }
  auto buffer=llvm::MemoryBuffer::getFileOrSTDIN(argv[2]);
  if(!buffer){llvm::errs()<<buffer.getError().message()<<'\n';return 2;}
  MLIRContext context; std::string error;
  auto c=parseFailureCase((*buffer)->getBuffer(),context,error);
  if(failed(c)){llvm::errs()<<error<<'\n';return 2;}
  if(StringRef(argv[1])=="--round-trip") {llvm::outs()<<serializeFailureCase(*c);return 0;}
  if(StringRef(argv[1])=="--replay") {
    auto s=replayFailureCase(*c,context);
    if(failed(s) || !(*s==c->signature)){llvm::errs()<<"failure signature did not reproduce\n";return 1;}
    llvm::outs()<<"reproduced "<<s->kind<<'\n';return 0;
  }
  auto reduced=reduceFailureCase(*c,context);
  if(failed(reduced)){llvm::errs()<<"failure signature did not reproduce\n";return 1;}
  llvm::outs()<<serializeFailureCase(*reduced);return 0;
}
