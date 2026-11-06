#ifndef FRISK_ANALYSIS_SHUFFLENETWORKJSON_H
#define FRISK_ANALYSIS_SHUFFLENETWORKJSON_H
#include "Dialect/Frisk/Analysis/ShuffleNetwork.h"
#include "llvm/Support/JSON.h"
namespace mlir::frisk::shuffle_network {
template<class Range> inline llvm::json::Array planNumbers(const Range &r) {
 llvm::json::Array a;for(auto x:r)a.push_back(int64_t(x));return a;
}
inline llvm::json::Object planJSON(const Plan &p) {
 llvm::json::Array steps;
 for(const auto &step:p.steps) {
  llvm::json::Object object{{"op",step.op}};
  if(step.op=="reg_permute")object["permutation"]=planNumbers(step.permutation);
  else if(step.op=="reg_select") {
   object["register_bit"]=step.registerBit;object["thread_mask"]=step.threadMask;
   object["constant"]=step.constant;
  } else if(step.op=="mixed_swap") {
   object["register_bit"]=step.registerBit;object["lane_bit"]=step.laneBit;
  } else {
   object["columns"]=planNumbers(step.columns);object["offsets"]=planNumbers(step.offsets);
   object["register_masks"]=planNumbers(step.registerMasks);
  }
  steps.push_back(std::move(object));
 }
 const auto &c=p.counts;
 llvm::json::Object counts{{"shfl",c.shfl},{"selects",c.selects},
   {"map_instructions",c.mapInstructions},{"total_work",c.totalWork},
   {"shuffle_warp_issues",c.shuffleWarpIssues},
   {"address_alu_warp_issues",c.addressAluWarpIssues}};
 return llvm::json::Object{{"contract","gf2-shuffle-network-v1"},
   {"registers",p.registers},{"kind",p.kind},{"steps",std::move(steps)},
   {"counts",std::move(counts)},{"legacy_work",p.legacyWork},
   {"legacy_shfl",p.legacyShfl},{"verified_holders",p.verifiedHolders},
   {"affine_base",p.affineBase},{"affine_columns",planNumbers(p.affineColumns)}};
}
} // namespace mlir::frisk::shuffle_network
#endif
