#ifndef FRISK_TEST_LEGACY_LAYOUT_ORACLE_H
#define FRISK_TEST_LEGACY_LAYOUT_ORACLE_H
#include "Dialect/Frisk/IR/FriskOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"
namespace mlir::frisk::test {
/// Historical formula oracle only. Never link into production compiler/FFI.
LogicalResult inferLegacyParallelLayout(ParallelOp op, OpBuilder &builder,
                                       DenseMap<Value, Attribute> &layouts);
LogicalResult inferLegacyGemmLayout(GemmOp op, OpBuilder &builder,
                                   DenseMap<Value, Attribute> &layouts);
LogicalResult inferLegacyReduceLayout(ReduceOp op, OpBuilder &builder,
                                     DenseMap<Value, Attribute> &layouts);
uint64_t getLegacyParallelOracleCallCount();
} // namespace mlir::frisk::test
#endif
