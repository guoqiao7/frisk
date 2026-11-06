#ifndef FRISK_TEST_TILELANG_CORPUS_H
#define FRISK_TEST_TILELANG_CORPUS_H
#include "llvm/Support/Error.h"
#include "llvm/ADT/StringRef.h"
#include <array>
#include <map>
#include <string>
#include <vector>
namespace mlir::frisk::test {
// Carrier order matches canonical Frisk map inputs, not election priority.
using CorpusCarrier = std::array<int64_t, 5>; // register, lane, warp, warp_group, cta
struct CorpusPoint {
  CorpusCarrier carrier;
  std::vector<int64_t> logical;
  int64_t copyGroup, addend;
  bool valid;
};
struct CorpusStoragePoint {
  std::vector<int64_t> logical;
  int64_t byteOffset, bitOffset;
};
struct TileLangCase {
  std::string name, operation, dtype, valueSemantics, writerPolicy;
  std::string expectation, diagnostic, reason;
  std::vector<int64_t> shape, validShape;
  CorpusCarrier extents;
  int64_t storageBytes;
  std::map<std::string, int64_t> parameters;
  std::vector<CorpusPoint> distributed;
  std::vector<CorpusStoragePoint> storage;
  std::vector<CorpusCarrier> writers;
};
struct TileLangCorpus { std::vector<TileLangCase> cases; };
/// Strict bounded data reader. Never executes paths, commands, or attribute text.
llvm::Expected<TileLangCorpus> readTileLangCorpus(llvm::StringRef text);
} // namespace mlir::frisk::test
#endif
