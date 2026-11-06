#include "Dialect/Frisk/Analysis/BroadcastLayoutConstraints.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/TargetParser/Host.h"
#include <chrono>
#include <cstdlib>
#include <unordered_set>

using namespace mlir;
using namespace mlir::frisk;
namespace {
[[noreturn]] static void fail(StringRef message) {
  llvm::errs() << message << '\n';
  std::exit(1);
}

struct Options {
  unsigned samples = 3, iterations = 1;
};
struct Case {
  std::string id;
  bool broadcast;
  uint64_t hardwarePoints;
  DistributedEncodingAttr source, destination;
  RankedTensorType input, output;
  SmallVector<int64_t> axes;
  ProofStatus symbolicExpected, enumerationExpected;
};

static StringRef status(ProofStatus value) {
  switch (value) {
  case ProofStatus::Proven: return "Proven";
  case ProofStatus::Disproven: return "Disproven";
  case ProofStatus::Unknown: return "Unknown";
  }
  llvm_unreachable("invalid proof status");
}

static DistributedEncodingAttr encoding(MLIRContext &context, unsigned regBits,
    ArrayRef<uint64_t> rows, ArrayRef<int64_t> outputWidths, int64_t replication) {
  Builder b(&context);
  unsigned columns = regBits + 5;
  SmallVector<APInt> bits;
  for (uint64_t row : rows)
    for (unsigned column = 0; column < columns; ++column)
      bits.emplace_back(1, (row >> column) & 1);
  SmallVector<Attribute> names;
  for (unsigned i = 0; i < outputWidths.size(); ++i)
    names.push_back(b.getStringAttr("x" + std::to_string(i)));
  auto matrix = DenseIntElementsAttr::get(RankedTensorType::get(
      {int64_t(rows.size()), int64_t(columns)}, b.getI1Type()), bits);
  auto map = BitLinearLayoutMapAttr::get(&context,
      b.getStrArrayAttr({"register", "lane"}),
      b.getDenseI64ArrayAttr({int64_t(regBits), 5}), b.getArrayAttr(names),
      b.getDenseI64ArrayAttr(outputWidths), matrix);
  return DistributedEncodingAttr::get(&context, map,
      b.getDenseI64ArrayAttr({int64_t(1) << regBits, 32, 1, 1, 1}),
      b.getI64IntegerAttr(replication));
}

// All attributes, matrices, types, and corpus construction are outside timing.
static SmallVector<Case> corpus(MLIRContext &context) {
  auto type = [&](ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, IntegerType::get(&context, 32));
  };
  SmallVector<Case> result;
  for (bool broadcast : {false, true}) {
    for (auto size : {std::make_pair("small", 2u),
                      std::make_pair("boundary", 11u),
                      std::make_pair("overbudget", 12u)}) {
      unsigned regBits = size.second;
      int64_t registers = int64_t(1) << regBits;
      SmallVector<uint64_t> rows;
      for (unsigned bit = 0; bit < regBits; ++bit)
        rows.push_back(uint64_t(1) << bit);
      auto source = encoding(context, regBits, rows, {int64_t(regBits)}, 32);
      rows.push_back(uint64_t(1) << regBits);
      auto destination = encoding(context, regBits, rows, {int64_t(regBits), 1}, 16);
      result.push_back({std::string(broadcast ? "broadcast_" : "ownership_") + size.first,
          broadcast, uint64_t(registers * 32), source, destination,
          type({registers}), type({registers, 2}), {1}, ProofStatus::Proven,
          regBits == 12 ? ProofStatus::Unknown : ProofStatus::Proven});
    }
  }
  // At lane zero the source's low logical bit is always zero; destination
  // register one needs (1,0). Both layouts nevertheless have complete coverage.
  result.push_back({"broadcast_same_thread_unavailable", true, 128,
      encoding(context, 2, {4, 1}, {2}, 32),
      encoding(context, 2, {1, 2, 4}, {2, 1}, 16),
      type({4}), type({4, 2}), {1}, ProofStatus::Disproven, ProofStatus::Disproven});
  return result;
}

static LayoutProof symbolic(const Case &c) {
  return c.broadcast
      ? proveBroadcastLayout(c.source, c.destination, c.input, c.output, c.axes)
      : proveExecutionOwnership(c.source, c.input, "first_owner");
}

// Independent of GF(2) and the production Broadcast proof: retain the original
// bounded coordinate enumeration, then use actual source-register sets per
// thread. Set construction avoids a quadratic register scan without changing
// the same-thread availability predicate. Tables are rebuilt on EVERY call.
static LayoutProof enumeration(const Case &c) {
  ExecutionEnumeration source, destination;
  auto proof = enumerateExecutionLayout(c.source, c.input, source);
  if (!c.broadcast || proof.status != ProofStatus::Proven) return proof;
  proof = enumerateExecutionLayout(c.destination, c.output, destination);
  if (proof.status != ProofStatus::Proven) return proof;
  if (!llvm::equal(ArrayRef(source.topology).drop_front(),
                   ArrayRef(destination.topology).drop_front()))
    return {ProofStatus::Disproven, {}, "enumeration oracle: thread topology mismatch"};
  std::unordered_set<uint64_t> available;
  available.reserve(source.topology[0]);
  uint64_t threads = destination.logical.size() / destination.topology[0];
  for (uint64_t thread = 0; thread < threads; ++thread) {
    available.clear();
    for (uint64_t reg = 0; reg < source.topology[0]; ++reg)
      available.insert(source.logical[thread * source.topology[0] + reg]);
    for (uint64_t reg = 0; reg < destination.topology[0]; ++reg) {
      auto coordinate = destination.coordinate(
          destination.logical[thread * destination.topology[0] + reg]);
      uint64_t wanted = 0;
      unsigned inputAxis = 0;
      for (unsigned axis = 0; axis < coordinate.size(); ++axis)
        if (!llvm::is_contained(c.axes, axis)) {
          wanted = wanted * c.input.getDimSize(inputAxis++) + coordinate[axis];
        }
      if (!available.count(wanted))
        return {ProofStatus::Disproven, coordinate,
                "enumeration oracle: source value unavailable in destination thread"};
    }
  }
  return {ProofStatus::Proven, {}, "bounded coordinate/register-set Broadcast oracle"};
}

static uint64_t fingerprint(const LayoutProof &proof) {
  uint64_t value = proof.status == ProofStatus::Proven ? 1 :
                   proof.status == ProofStatus::Disproven ? 2 : 3;
  for (int64_t coordinate : proof.counterexample)
    value = value * 17 + uint64_t(coordinate) + 1;
  return value;
}

using ProofFunction = LayoutProof (*)(const Case &);
static llvm::json::Object measure(const Case &c, ProofFunction function,
    const LayoutProof &warmup, ProofStatus expected, const Options &options) {
  volatile uint64_t sink = 0;
  uint64_t expectedFingerprint = fingerprint(warmup);
  llvm::json::Array samples, perIteration, counterexample;
  for (unsigned sample = 0; sample < options.samples; ++sample) {
    auto begin = std::chrono::steady_clock::now();
    for (unsigned iteration = 0; iteration < options.iterations; ++iteration)
      sink += fingerprint(function(c));
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - begin).count();
    samples.push_back(int64_t(elapsed));
    perIteration.push_back(double(elapsed) / options.iterations);
  }
  uint64_t iterations = uint64_t(options.samples) * options.iterations;
  uint64_t expectedSink = expectedFingerprint * iterations;
  if (sink != expectedSink) fail(c.id + ": proof sink changed");
  for (int64_t coordinate : warmup.counterexample) counterexample.push_back(coordinate);
  return llvm::json::Object{{"status", status(warmup.status)},
      {"expected_status", status(expected)}, {"reason", warmup.reason},
      {"counterexample", std::move(counterexample)}, {"time_unit", "ns"},
      {"sample_count", options.samples}, {"iterations_per_sample", options.iterations},
      {"measured_iterations", iterations}, {"samples_ns", std::move(samples)},
      {"samples_ns_per_iteration", std::move(perIteration)},
      {"sink", uint64_t(sink)}, {"expected_sink", expectedSink}, {"sink_verified", true}};
}

static bool parseOptions(int argc, char **argv, Options &options) {
  bool seenSamples = false, seenIterations = false;
  for (int i = 1; i < argc; ++i) {
    StringRef argument(argv[i]);
    bool isSamples = argument == "--samples";
    if ((!isSamples && argument != "--iterations") || i + 1 == argc) return false;
    bool &seen = isSamples ? seenSamples : seenIterations;
    unsigned value;
    StringRef number(argv[++i]);
    if (seen || number.empty() ||
        !llvm::all_of(number, [](char c) { return c >= '0' && c <= '9'; }) ||
        number.getAsInteger(10, value) || !value || value > (isSamples ? 100u : 1000u))
      return false;
    seen = true;
    (isSamples ? options.samples : options.iterations) = value;
  }
  return true;
}
} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parseOptions(argc, argv, options)) {
    llvm::errs() << "usage: frisk_layout_proof_bench [--samples 1..100] [--iterations 1..1000]\n";
    return 2;
  }
  {
    auto executable = llvm::sys::fs::getMainExecutable(argv[0], reinterpret_cast<void *>(&main));
    auto binary = llvm::MemoryBuffer::getFile(executable);
    if (!binary) fail("cannot hash measured executable");
    llvm::SHA256 sha;
    sha.update((*binary)->getBuffer());
    std::string binaryHash = llvm::toHex(sha.final(), true);
    MLIRContext context;
    context.disableMultithreading();
    context.loadDialect<FriskDialect>();
    auto inputs = corpus(context);
    llvm::json::Array cases;
    for (const Case &c : inputs) {
      // Both methods validate expected outcomes before either timed region.
      auto symbolicWarmup = symbolic(c), enumerationWarmup = enumeration(c);
      if (symbolicWarmup.status != c.symbolicExpected ||
          enumerationWarmup.status != c.enumerationExpected)
        fail(c.id + ": unexpected preflight proof status");
      cases.push_back(llvm::json::Object{{"id", c.id},
          {"proof_kind", c.broadcast ? "broadcast" : "ownership"},
          {"hardware_points", c.hardwarePoints},
          {"equivalent_work", c.symbolicExpected == c.enumerationExpected},
          {"symbolic", measure(c, symbolic, symbolicWarmup, c.symbolicExpected, options)},
          {"enumeration", measure(c, enumeration, enumerationWarmup, c.enumerationExpected, options)}});
    }
    llvm::json::Object output{{"schema_version", 1}, {"kind", "layout-proof"},
        {"status", "success"},
        {"measurement_scope", "CPU proof calls plus result consumption; corpus construction excluded; enumeration tables rebuilt per call; no global cache; symbolic then enumeration"},
        {"options", llvm::json::Object{{"samples", options.samples},
            {"iterations_per_sample", options.iterations}, {"warmup_iterations", 1},
            {"enumeration_point_limit", 65536}}},
        {"environment", llvm::json::Object{{"binary_path", executable},
            {"binary_sha256", binaryHash}, {"compiler", __VERSION__},
            {"build_type", FRISK_PROOF_BUILD_TYPE}, {"cpu", llvm::sys::getHostCPUName()},
            {"source_digest", FRISK_PROOF_SOURCE_DIGEST},
            {"config_digest", FRISK_PROOF_CONFIG_DIGEST}}}, {"cases", std::move(cases)}};
    llvm::outs() << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(output)));
  }
  return 0;
}
