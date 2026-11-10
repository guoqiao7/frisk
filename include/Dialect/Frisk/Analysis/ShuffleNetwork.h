#ifndef FRISK_SHUFFLE_NETWORK_H
#define FRISK_SHUFFLE_NETWORK_H

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::frisk::shuffle_network {
enum class Objective { MinShuffle, MinWork };
// A pullback step maps each destination holder to the previous SSA carrier.
// This small standalone planner is independent of MLIR attrs and the backend.
struct Step {
  std::string op;
  std::vector<uint64_t> permutation;
  unsigned registerBit = 0, threadMask = 0, constant = 0, laneBit = 0;
  std::array<uint64_t, 5> columns{};
  std::array<uint64_t, 4> offsets{};
  std::vector<uint64_t> registerMasks;
};
struct Counts {
  uint64_t shfl = 0, selects = 0, mapInstructions = 0, totalWork = 0;
  uint64_t shuffleWarpIssues = 0, addressAluWarpIssues = 0;
};
struct Plan {
  unsigned registers = 0;
  std::string kind;
  std::vector<Step> steps;
  Counts counts;
  uint64_t legacyWork = 0, legacyShfl = 0, verifiedHolders = 0;
  uint64_t affineBase = 0;
  std::vector<uint64_t> affineColumns;
};
// The full 128-thread elected owner table is thread-major, then register-major.
// Source holders are encoded as sourceThread*sourceRegisters+sourceRegister.
// Rejection is a fallback decision, not a claim that a layout is invalid.
std::vector<Plan> candidates(unsigned sourceRegisters, unsigned destinationRegisters,
                             const std::vector<uint64_t> &sourceHolders);
std::optional<Plan> plan(unsigned sourceRegisters, unsigned destinationRegisters,
                         const std::vector<uint64_t> &sourceHolders,
                         Objective objective = Objective::MinShuffle);
bool sameExecutionPlan(const Plan &, const Plan &);
struct StructuralResources {
  uint64_t logicalCarrierWords = 0, b32TemporaryDefinitions = 0;
  uint64_t predicateDefinitions = 0, communicationStages = 0;
};
// Counts declarations in the specified emitted schedule, not physical registers,
// compiler liveness, spill traffic, occupancy or a latency prediction.
StructuralResources structuralResources(const Plan &);
std::vector<Plan> frontier(unsigned sourceRegisters, unsigned destinationRegisters,
                           const std::vector<uint64_t> &sourceHolders);
uint64_t sourceHolder(const Plan &plan, uint64_t destinationHolder);
} // namespace mlir::frisk::shuffle_network
#endif
