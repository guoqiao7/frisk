#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Visitors.h"

#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskAttributes.h"
#include "Dialect/Frisk/IR/FriskEnums.h"
#include "Dialect/Frisk/IR/LegacyImportSemantics.h"

// #include "Dialect/Frisk/IR/FriskEnums.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/Frisk/IR/FriskOps.cpp.inc"

// move dialect def in this file to make compiler happy
#include "Dialect/Frisk/IR/FriskDialect.cpp.inc"
namespace mlir {
namespace frisk {

} // namespace frisk
} // namespace mlir

namespace mlir {
namespace frisk {

class GemmOp;

bool isSupportedExecutionThreadCount(int64_t threads) {
  return threads == 32 || threads == 64 || threads == 128 ||
         threads == 256 || threads == 512 || threads == 1024;
}

LogicalResult verifyOperationExecutionAttributes(Operation *op) {
  if (auto raw = op->getAttr("frisk.execution_layout"))
    if (!isa<DistributedEncodingAttr>(raw))
      return op->emitOpError("frisk.execution_layout must be a distributed encoding");
  if (auto raw = op->getAttr("frisk.writer_policy")) {
    auto policy = dyn_cast<StringAttr>(raw);
    if (!policy || (policy.getValue() != "all" && policy.getValue() != "first_owner"))
      return op->emitOpError("frisk.writer_policy must be all or first_owner");
  }
  for (StringRef name : {"frisk.execution_threads", "frisk.vector_bytes"}) {
    auto raw = op->getAttr(name);
    if (!raw) continue;
    auto value = dyn_cast<IntegerAttr>(raw);
    if (!value || !value.getType().isSignlessInteger(64))
      return op->emitOpError() << name << " must be an i64 attribute";
    int64_t n = value.getInt();
    if (name == "frisk.execution_threads" ? !isSupportedExecutionThreadCount(n)
        : !(n == 1 || n == 2 || n == 4 || n == 8 || n == 16))
      return op->emitOpError() << "unsupported " << name << " value " << n;
  }
  return success();
}



//===----------------------------------------------------------------------===//
// -- KernelOp --
//===----------------------------------------------------------------------===//
void KernelOp::build(OpBuilder &builder, OperationState &state, StringRef name, FunctionType type) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addRegion();
}

LogicalResult KernelOp::verify() {
  auto typeAttr = getFunctionTypeAttr();
  if (!typeAttr)
    return emitOpError("requires a 'function_type' attribute");

  auto functionType = llvm::dyn_cast<FunctionType>(typeAttr.getValue());
  if (!functionType)
    return emitOpError("requires a function type");

  // 验证区域参数
  if (getBody(0)->getNumArguments() != functionType.getNumInputs())
    return emitOpError("region argument count does not match function type");

  for (unsigned i = 0; i < getBody(0)->getNumArguments(); ++i) {
    if (getBody(0)->getArgument(i).getType() != functionType.getInput(i))
      return emitOpError("region argument type mismatch");
  }

  return success();
}

ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析符号名称
  StringAttr symName;
  if (parser.parseSymbolName(symName, "sym_name", result.attributes))
    return failure();
  // 解析参数列表
  SmallVector<Type> argTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen()) &&
      (parser.parseTypeList(argTypes) || parser.parseRParen()))
    return failure();
  // 解析结果类型 - 可选，如果没有结果就是空
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen()) &&
        (parser.parseTypeList(resultTypes) || parser.parseRParen()))
      return failure();
  }
  // 创建函数类型属性
  auto functionType = parser.getBuilder().getFunctionType(argTypes, resultTypes);
  result.addAttribute("function_type", TypeAttr::get(functionType));
  // 解析区域
  Region *body = result.addRegion();
  // Signature carries types; the printed entry block carries actual SSA names.
  if (parser.parseRegion(*body) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void KernelOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printSymbolName(getSymName());
  // 打印参数类型
  auto funcType = getFunctionType();
  auto funcTy = dyn_cast<FunctionType>(funcType);
  p << "(";
  llvm::interleaveComma(funcTy.getInputs(), p);
  p << ")";
  // 只有当有结果时才打印结果类型
  if (!funcTy.getResults().empty()) {
    p << " -> (";
    llvm::interleaveComma(funcTy.getResults(), p);
    p << ")";
  }
  // 打印区域
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/true, /*printBlockTerminators=*/true);
  // 打印属性
  p.printOptionalAttrDict((*this)->getAttrs(), {"sym_name", "function_type"});
}

//===----------------------------------------------------------------------===//
// -- ParallelOp --
//===----------------------------------------------------------------------===//
void ParallelOp::build(OpBuilder &builder, OperationState &state,
                      llvm::ArrayRef<int64_t> ranges, int64_t thread_num) {
  state.addAttribute("threads", builder.getI64IntegerAttr(thread_num));
  state.addAttribute("ranges", builder.getDenseI64ArrayAttr(ranges));
  state.addRegion();
  // Region *region = state.regions[0].get();
  // Block *entry = new Block();
  // region->push_back(entry);
  // for (unsigned i=0; i<ranges.size(); ++i) {
  //   entry->addArgument(builder.getIndexType(), state.location);
  // }
}

ParseResult ParallelOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析迭代变量列表: (%arg1, %arg2)
  SmallVector<OpAsmParser::Argument, 4> inductionVars;
  if (parser.parseArgumentList(inductionVars, AsmParser::Delimiter::Paren))
    return failure();
  for (auto &arg : inductionVars) arg.type = parser.getBuilder().getIndexType();
  // 解析等号和范围: = (8, 8)
  if (parser.parseEqual() || parser.parseLParen())
    return failure();
  SmallVector<int64_t> ranges;
  if (parser.parseCommaSeparatedList([&]() {
        int64_t range;
        if (parser.parseInteger(range))
          return failure();
        ranges.push_back(range);
        return success();
      }) || parser.parseRParen())
    return failure();
  // 解析线程数量: , threads=128
  int64_t threads = 0;
  if (parser.parseComma() || parser.parseKeyword("threads") || 
      parser.parseEqual() || parser.parseInteger(threads))
    return failure();
  // 添加属性
  result.addAttribute("ranges", parser.getBuilder().getDenseI64ArrayAttr(ranges));
  result.addAttribute("threads", parser.getBuilder().getI64IntegerAttr(threads));
  // 解析区域
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, inductionVars))
    return failure();
  return success();
}

void ParallelOp::print(OpAsmPrinter &p) {
  // 打印迭代变量: (%arg1, %arg2)
  p << " (";
  llvm::interleaveComma(getBody(0)->getArguments(), p, [&](BlockArgument arg) {
    p << arg;
  });
  p << ")";
  // 打印范围和线程配置: = (8, 8), threads=128
  p << " = (";
  auto grid = getGrid();
  llvm::interleaveComma(grid, p);
  p << "), threads = " << getThreadNum();
  // 打印区域（不打印终止符）
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/true);
}

LogicalResult ParallelOp::verify() {
  if (!isSupportedExecutionThreadCount(getThreads()))
    return emitOpError("threads must be one of 32,64,128,256,512,1024");
  if (getRanges().empty() || llvm::any_of(getRanges(), [](int64_t n) { return n <= 0; }))
    return emitOpError("ranges must contain positive static extents");
  if (getRegion().empty() || getRegion().front().getNumArguments() != getRanges().size())
    return emitOpError("range count must match the region induction arguments");
  for (auto arg : getRegion().front().getArguments())
    if (!arg.getType().isIndex()) return emitOpError("induction arguments must have index type");
  return success();
}

//===----------------------------------------------------------------------===//
// -- BlockOp --
//===----------------------------------------------------------------------===//
void BlockOp::build(OpBuilder &builder, OperationState &state, ArrayRef<int64_t> ranges, BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);
  
  state.addAttribute("ranges", builder.getDenseI64ArrayAttr(ranges));

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  llvm::SmallVector<Value> inductionVars;
  for (int64_t range : ranges) {
    inductionVars.push_back(bodyBlock->addArgument(builder.getIndexType(), state.location));
  }
  if (!bodyBuilder) {
    ensureTerminator(*bodyRegion, builder, state.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(inductionVars);
    builder.create<EndOp>(state.location);
  }
}

ParseResult BlockOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析参数列表: (%arg3, %arg4)
  SmallVector<OpAsmParser::Argument, 4> blockArgs;
  if (parser.parseArgumentList(blockArgs, OpAsmParser::Delimiter::Paren))  // 解析 "to" 关键字
    return failure();
  if (parser.parseKeyword("to"))  // 解析边界列表: (128, 128)
    return failure();
  SmallVector<int64_t, 4> ranges;
  if (parser.parseCommaSeparatedList(
        OpAsmParser::Delimiter::Paren,
        [&]() -> ParseResult {
          int64_t range;
          if (parser.parseInteger(range))
            return failure();
          ranges.push_back(range);
          return success();
        }))
    return failure();
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs))  // 解析区域 { ... }
    return failure();
  if (parser.parseOptionalAttrDict(result.attributes))  // 解析可选的属性字典
    return failure();
  if (!ranges.empty()) {  // 将边界添加到属性中
    result.addAttribute("ranges", parser.getBuilder().getDenseI64ArrayAttr(ranges));
  }
  return success();
}

void BlockOp::print(OpAsmPrinter &p) {
  // 打印迭代变量: (%arg1, %arg2)
  p << " (";
  llvm::interleaveComma(getBody(0)->getArguments(), p, [&](BlockArgument arg) {
    p << arg;
  });
  p << ")";
  // 打印范围和线程配置: = (8, 8), threads=128
  p << " to (";
  auto ranges = getBlockRanges();
  llvm::interleaveComma(ranges, p);
  p << ")";
  // 打印区域（不打印终止符）
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

LogicalResult BlockOp::verify() {   // 暂时关闭block op的验证方法
  // auto &blockBody = getRegion();
  // if (blockBody.empty()) {
  //   return emitOpError("block must have a body");
  // }
  // auto &block = blockBody.front();
  // bool hasAffineLoadOp = false;
  // bool hasAffineStoreOp = false;
  
  // for (auto &op : block.getOperations()) {
  //   if (isa<affine::AffineLoadOp>(op)) {    // 检查是否是affine.load操作
  //     hasAffineLoadOp = true;
  //   }
  //   else if (isa<affine::AffineStoreOp>(op)) {    // 检查是否是affine.store操作  
  //     hasAffineStoreOp = true;
  //   }
  //   // 如果已经找到两种操作，可以提前退出
  //   if (hasAffineLoadOp && hasAffineStoreOp) {
  //     break;
  //   }
  // }
  // // 验证结果
  // if (!hasAffineLoadOp) {
  //   return emitOpError("block must contain at least one affine.load operation");
  // }
  // if (!hasAffineStoreOp) {
  //   return emitOpError("block must contain at least one affine.store operation");
  // }
  return success();
}

//===----------------------------------------------------------------------===//
// -- ForOp --
//===----------------------------------------------------------------------===//
void ForOp::build(OpBuilder &builder, OperationState &state, 
                  int64_t lowerBound, int64_t upperBound, int64_t step, 
                  BodyBuilderFn bodyBuilder) {
  OpBuilder::InsertionGuard guard(builder);

  state.addAttribute("lower", builder.getI64IntegerAttr(lowerBound));
  state.addAttribute("upper", builder.getI64IntegerAttr(upperBound));
  state.addAttribute("step", builder.getI64IntegerAttr(step));

  Region *bodyRegion = state.addRegion();
  Block *bodyBlock = builder.createBlock(bodyRegion);
  auto iv = bodyBlock->addArgument(builder.getIndexType(), state.location);
  if (!bodyBuilder) {
    ensureTerminator(*bodyRegion, builder, state.location);
  } else {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(iv);
    builder.create<EndOp>(state.location);
  }
}

ParseResult ForOp::parse(OpAsmParser &parser, OperationState &result) {
  // 解析格式：frisk.for %arg1 = 0 to 1024 step = 32 { ... }
  OpAsmParser::UnresolvedOperand inductionVar;
  IntegerAttr lowerAttr, upperAttr, stepAttr;

  if (parser.parseOperand(inductionVar) || parser.parseEqual())  // 解析循环变量 "%arg1 ="
    return failure();
  auto builder = parser.getBuilder();
  if (parser.parseAttribute(lowerAttr, builder.getIntegerType(64), "lower", result.attributes))  // 解析下界
    return failure();
  if (parser.parseKeyword("to"))  // 解析 "to"
    return failure();
  if (parser.parseAttribute(upperAttr, builder.getIntegerType(64), "upper", result.attributes))  // 解析上界
    return failure();
  if (parser.parseKeyword("step") || parser.parseEqual())  // 解析 "step ="
    return failure();
  if (parser.parseAttribute(stepAttr, builder.getIntegerType(64), "step", result.attributes))  // 解析步长
    return failure();

  std::unique_ptr<Region> region = std::make_unique<Region>();  // 解析区域
  llvm::SmallVector<OpAsmParser::Argument, 4> regionArgs;  // 解析区域参数类型（循环变量）
  llvm::SmallVector<Type, 4> regionArgTypes;

  regionArgTypes.push_back(builder.getIndexType());  // 循环变量类型为index
  if (parser.parseArgumentList(regionArgs, OpAsmParser::Delimiter::Paren, true) || parser.parseRegion(*region, regionArgs))
    return failure();
  result.addRegion(std::move(region));
  return success();
}

void ForOp::print(OpAsmPrinter &p) {
  p << " ";
  p << getInductionVar() << " = " << getLower();  // 打印循环变量 //打印下界
  p << " to " << getUpper(); // 打印上界
  if (getStep() > 1) {
    p << " step = " << getStep();  // 打印步长
  }
  p << " ";
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false, /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// -- GemmOp --
//===----------------------------------------------------------------------===//
LogicalResult GemmOp::verify() {
  if (auto error = getLegacyGemmValidationError(
          getA().getType(), getB().getType(), getC().getType(), getTransA(),
          getTransB(), getM(), getN(), getK()))
    return emitOpError(*error);
  return success();
}

void GemmOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto add = [&](OpOperand &operand, MemoryEffects::Effect *effect) {
    effects.emplace_back(effect, &operand, SideEffects::DefaultResource::get());
  };
  add(getOperation()->getOpOperand(0), MemoryEffects::Read::get());
  add(getOperation()->getOpOperand(1), MemoryEffects::Read::get());
  add(getOperation()->getOpOperand(2), MemoryEffects::Write::get());
  if (!getClearAccum())
    add(getOperation()->getOpOperand(2), MemoryEffects::Read::get());
}

//===----------------------------------------------------------------------===//
// -- AllocBufferOp --
//===----------------------------------------------------------------------===//
// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType) {
//   build(builder, state, shape, elementType, /*alignment=*/0, /*memorySpace=*/0);
// }

// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType, 
//                           int64_t alignment) {
//   build(builder, state, shape, elementType, alignment, /*memorySpace=*/0);
// }

// void AllocBufferOp::build(OpBuilder &builder, OperationState &state,
//                           ArrayRef<int64_t> shape, Type elementType,
//                           int64_t alignment, int64_t memorySpace) {
//   // 创建 memref 类型
//   auto memrefType = MemRefType::get(shape, elementType, /*layout=*/{}, memorySpace);
//   // 添加属性
//   state.addAttribute("shape", builder.getDenseI64ArrayAttr(shape));
//   state.addAttribute("elementType", TypeAttr::get(elementType));
//   state.addAttribute("alignment", builder.getI64IntegerAttr(alignment));
//   state.addAttribute("memorySpace", builder.getI64IntegerAttr(memorySpace));
//   // 添加结果类型
//   state.addTypes(memrefType);
// }

LogicalResult AllocBufferOp::verify() {
  auto resultType = getResult().getType();
  
  // 检查结果类型是否是 memref
  if (!isa<MemRefType>(resultType)) {
    return emitOpError("result must be a memref type");
  }
  
  auto memrefType = cast<MemRefType>(resultType);
  // 检查属性与结果类型是否一致
  auto attrShape = getShape();
  auto attrElementType = getElementType();
  auto attrMemorySpace = getMemorySpace();
  if (memrefType.getShape() != attrShape) {
    return emitOpError("shape attribute must match result memref shape");
  }
  if (memrefType.getElementType() != attrElementType) {
    return emitOpError("elementType attribute must match result memref element type");
  }

  std::optional<attr::MemorySpace> resultMemorySpace =
      getFriskMemorySpace(memrefType);
  if (!resultMemorySpace || *resultMemorySpace != attrMemorySpace) {
    return emitOpError("memorySpace attribute must match result memref memory space");
  }
  // 检查对齐值是否有效
  int64_t alignment = getAlignment();
  if (alignment < 0) {
    return emitOpError("alignment must be non-negative");
  }
  // 检查对齐值是否是 2 的幂（可选，但推荐）
  if (alignment > 0 && (alignment & (alignment - 1)) != 0) {
    return emitOpError("alignment must be a power of 2");
  }
  // 检查 memorySpace 是否有效
  switch (attrMemorySpace) {
    case ::mlir::frisk::attr::MemorySpace::Local:
      break;
    case ::mlir::frisk::attr::MemorySpace::Global:
      break;
    case ::mlir::frisk::attr::MemorySpace::Shared:
      break;
    default:
      return emitOpError("memorySpace must be Local, Global, or Shared");
  }
  return success();
}

ParseResult AllocBufferOp::parse(OpAsmParser &parser, OperationState &result) {
  // 直接解析属性字典
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // 解析结果类型: -> memref<32x128xf32>
  if (parser.parseArrow())
    return failure();
  Type resultType;
  if (parser.parseType(resultType))
    return failure();
  // 检查结果类型是否是 memref
  if (!isa<MemRefType>(resultType)) {
    return parser.emitError(parser.getCurrentLocation(), "expected memref type for result");
  }
  auto memrefType = cast<MemRefType>(resultType);
  // 从结果类型提取 shape 和 elementType
  auto shape = memrefType.getShape();
  auto elementType = memrefType.getElementType();
  auto memorySpace = getFriskMemorySpace(memrefType);
  if (!memorySpace)
    return parser.emitError(parser.getCurrentLocation(), "unsupported alloc_buffer memory space");
  // 添加结果类型
  result.addTypes(resultType);
  // 添加必要的属性（如果属性字典中没有）
  auto &builder = parser.getBuilder();
  if (!result.attributes.get("shape")) {
    result.addAttribute("shape", builder.getDenseI64ArrayAttr(shape));
  }
  if (!result.attributes.get("elementType")) {
    result.addAttribute("elementType", TypeAttr::get(elementType));
  }
  if (!result.attributes.get("memorySpace")) {
    result.addAttribute("memorySpace", MemorySpaceAttr::get(builder.getContext(), *memorySpace));
  }
  return success();
}

// 自定义汇编格式打印
void AllocBufferOp::print(OpAsmPrinter &p) {
  // Preserve semantic annotations so normalization can reject unsupported ones.
  // Inherent type metadata is recoverable from the printed result type.
  p.printOptionalAttrDict((*this)->getAttrs(), {"shape", "elementType", "memorySpace"});
  p << " -> " << getResult().getType();
}


//===----------------------------------------------------------------------===//
// -- CopyOp --
//===----------------------------------------------------------------------===//
void CopyOp::build(OpBuilder &builder, OperationState &state, 
                  Value srcMemref, Value dstMemref, ValueRange srcIndices, ValueRange dstIndices) {
  auto srcMemrefType = llvm::cast<MemRefType>(srcMemref.getType());
  auto dstMemrefType = llvm::cast<MemRefType>(dstMemref.getType());
  int64_t srcRank = srcMemrefType.getRank();
  int64_t dstRank = dstMemrefType.getRank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  // Empty indices denote a whole tile, not an indexed identity projection.
  auto srcMap = srcIndices.empty() ? builder.getEmptyAffineMap()
                                  : builder.getMultiDimIdentityMap(srcRank);
  auto dstMap = dstIndices.empty() ? builder.getEmptyAffineMap()
                                  : builder.getMultiDimIdentityMap(dstRank);
  build(builder, state, srcMemref, dstMemref, srcMap, dstMap, srcIndices, dstIndices);
}

void CopyOp::build(OpBuilder &builder, OperationState &state, 
                    Value srcMemref, Value dstMemref, 
                    AffineMap srcMap, AffineMap dstMap,
                    ValueRange srcIndices, ValueRange dstIndices) {
  assert(srcMap.getNumInputs() == srcIndices.size() && 
    "source map inputs must match source indices count");
  assert(dstMap.getNumInputs() == dstIndices.size() && 
    "destination map inputs must match destination indices count");

  auto srcMemrefType = llvm::cast<MemRefType>(srcMemref.getType());
  auto dstMemrefType = llvm::cast<MemRefType>(dstMemref.getType());
  int64_t srcRank = srcMemrefType.getRank();
  int64_t dstRank = dstMemrefType.getRank();
  std::vector<int64_t> srcExtents;
  auto dstExtents = dstMemrefType.getShape(); 

  assert(srcRank >= dstRank && "src rank msut >= dst rank");
  for (unsigned i=0; i<srcRank; i++) {
    if (i < srcRank - dstRank) {
      srcExtents.push_back(1);
    } else {
      srcExtents.push_back(dstExtents[i-(srcRank-dstRank)]);
    }
  }

  build(builder, state, srcMemref, dstMemref, srcMap, dstMap, srcIndices, dstIndices, 
    builder.getDenseI64ArrayAttr(srcExtents), builder.getDenseI64ArrayAttr(dstExtents));
}

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();
  // 解析源操作数和索引
  OpAsmParser::UnresolvedOperand srcMemrefInfo;
  AffineMapAttr srcMapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> srcMapOperands;
  if (parser.parseOperand(srcMemrefInfo) ||
      parser.parseAffineMapOfSSAIds(srcMapOperands, srcMapAttr, "srcMap", result.attributes) ||
      parser.parseComma())
    return failure();
  // 解析目标操作数和索引
  OpAsmParser::UnresolvedOperand dstMemrefInfo;
  AffineMapAttr dstMapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dstMapOperands;
  if (parser.parseOperand(dstMemrefInfo) ||
      parser.parseAffineMapOfSSAIds(dstMapOperands, dstMapAttr, "dstMap", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  // 解析类型信息
  SmallVector<Type, 2> memrefTypes;
  if (parser.parseColonTypeList(memrefTypes))
    return failure();
  if (memrefTypes.size() != 2)
    return parser.emitError(parser.getNameLoc(), "expected two memref types");
  auto srcType = dyn_cast<MemRefType>(memrefTypes[0]);
  auto dstType = dyn_cast<MemRefType>(memrefTypes[1]);
  if (!srcType || !dstType)
    return parser.emitError(parser.getNameLoc(), "expected memref types");
  // 解析操作数
  if (parser.resolveOperand(srcMemrefInfo, srcType, result.operands) ||
      parser.resolveOperand(dstMemrefInfo, dstType, result.operands) ||
      parser.resolveOperands(srcMapOperands, indexTy, result.operands) ||
      parser.resolveOperands(dstMapOperands, indexTy, result.operands))
    return failure();
  // 设置 operandSegmentSizes 属性（隐藏的）
  SmallVector<int32_t> segmentSizes = {
      1, 1, 
      static_cast<int32_t>(srcMapOperands.size()),  // srcIndices
      static_cast<int32_t>(dstMapOperands.size())   // dstIndices
  };
  result.addAttribute(CopyOp::getOperandSegmentSizesAttrName(result.name), 
    builder.getDenseI32ArrayAttr(segmentSizes));
  return success();
}

void CopyOp::print(OpAsmPrinter &p) {
  p << " " << getSrcMemRef() << "[";
  p.printAffineMapOfSSAIds(getSrcMapAttr(), getSrcIndices());
  p << "], " << getDstMemRef() << "[";
  p.printAffineMapOfSSAIds(getDstMapAttr(), getDstIndices());
  p << "]";
  // ODS properties are not necessarily in getAttrs(). Preserve both extents
  // explicitly, using the same dictionary spelling consumed by the parser.
  NamedAttrList attrs((*this)->getAttrs());
  attrs.set("srcExtents", getSrcExtentsAttr());
  attrs.set("dstExtents", getDstExtentsAttr());
  SmallVector<StringRef> elidedAttrs = { "srcMap", "dstMap",
      CopyOp::getOperandSegmentSizesAttrName((*this)->getName()).getValue()
  };
  p.printOptionalAttrDict(attrs.getAttrs(), elidedAttrs);
  p << " : " << getSrcMemRef().getType() << ", " << getDstMemRef().getType();
}

LogicalResult CopyOp::verify() {
  if (failed(verifyOperationExecutionAttributes(*this))) return failure();
  AffineMap srcMap = getSrcMap();
  AffineMap dstMap = getDstMap();
  // 正确：srcIndices 长度必须等于 srcMap 输入维度
  if (getSrcIndices().size() != srcMap.getNumInputs()) {
    return emitOpError("expected ") << srcMap.getNumInputs()
           << " source indices, but got " << getSrcIndices().size();
  }
  // 正确：dstIndices 长度必须等于 dstMap 输入维度
  if (getDstIndices().size() != dstMap.getNumInputs()) {
    return emitOpError("expected ") << dstMap.getNumInputs()
           << " destination indices, but got " << getDstIndices().size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// -- FillOp --
//===----------------------------------------------------------------------===//
LogicalResult FillOp::verify() {
  if (failed(verifyOperationExecutionAttributes(*this))) return failure();
  auto memrefType = dyn_cast<MemRefType>(getMemref().getType());
  auto elemType = memrefType.getElementType();
  auto valueAttr = getValueAttr();
  if (auto floatAttr = dyn_cast<FloatAttr>(valueAttr)) {
    Type valueType = floatAttr.getType();
    if (elemType != valueType) {
      return emitOpError("value type ") << valueType << " does not match memref element type " << elemType;
    }
  } else {
    return emitOpError("fill value must be a float attribute");
  }
  return success();
}

// ParseResult FillOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand dstOperand;
//   Attribute valueAttr;
//   Type dstType;
//   if (parser.parseOperand(dstOperand) || parser.parseComma() ||
//       parser.parseAttribute(valueAttr, "value", result.attributes))// 解析目标操作数
//     return failure();
//   if (parser.parseColonType(dstType))// 解析类型
//     return failure();
//   if (parser.resolveOperand(dstOperand, dstType, result.operands))// 解析操作数
//     return failure();
//   return success();
// }

// void FillOp::print(OpAsmPrinter &p) {
//   p << " " << getMemref() << ", value = " << getValueAttr();
//   p << " : " << getMemref().getType();
// }

//===----------------------------------------------------------------------===//
// -- ReduceOp --
//===----------------------------------------------------------------------===//
void ReduceOp::build(OpBuilder &builder, OperationState &state, 
                    Value src, Value dst, StringRef kind, int64_t dim, bool clear) {
  state.addOperands({src, dst});
  state.addAttribute("kind", builder.getStringAttr(kind));
  state.addAttribute("dim", builder.getI64IntegerAttr(dim));
  state.addAttribute("clear", builder.getBoolAttr(clear));
}

LogicalResult ReduceOp::verify() {
  auto srcType = getSrcType();
  auto dstType = getDstType();
  auto kind = getKind();
  auto dim = getDim();
  if (!isa<MemRefType>(srcType) || !isa<MemRefType>(dstType)) {  // 检查源和目标是否是 memref 类型
    return emitOpError("source and destination must be memref types");
  }
  if (dim < 0 || dim >= srcType.getRank()) {  // 检查维度有效性
    return emitOpError("dimension ") << dim << " is out of range for source of rank " << srcType.getRank();
  }
  if (kind != "add" && kind != "mul" && kind != "min" && kind != "max") {  // 检查 reduce 类型有效性
    return emitOpError("unsupported reduce kind: ") << kind << ". Supported: add, mul, min, max";
  }
  // 检查源和目标形状兼容性
  auto srcShape = srcType.getShape();
  auto dstShape = dstType.getShape();
  if (srcType.getRank() != dstType.getRank() + 1) {  // 目标 rank 应该比源 rank 少 1
    return emitOpError("destination rank must be one less than source rank. Got: ")
           << dstType.getRank() << " vs " << srcType.getRank();
  }
  for (int64_t i = 0, j = 0; i < srcType.getRank(); ++i) {  // 检查非归约维度是否匹配
    if (i == dim) continue; // 跳过归约维度
    if (srcShape[i] != dstShape[j]) {
      return emitOpError("non-reduced dimension mismatch: source[")
             << i << "] = " << srcShape[i] << " vs destination["
             << j << "] = " << dstShape[j];
    }
    j++;
  }
  if (srcType.getElementType() != dstType.getElementType()) {  // 检查元素类型兼容性
    return emitOpError("source and destination must have the same element type");
  }
  return success();
}

void ReduceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto add = [&](OpOperand &operand, MemoryEffects::Effect *effect) {
    effects.emplace_back(effect, &operand, SideEffects::DefaultResource::get());
  };
  add(getOperation()->getOpOperand(0), MemoryEffects::Read::get());
  add(getOperation()->getOpOperand(1), MemoryEffects::Write::get());
  if (!getClear())
    add(getOperation()->getOpOperand(1), MemoryEffects::Read::get());
}

// // 自定义汇编格式解析
// ParseResult ReduceOp::parse(OpAsmParser &parser, OperationState &result) {
//   OpAsmParser::UnresolvedOperand srcOperand, dstOperand;
//   Type srcType, dstType;
//   // 解析操作数: %src, %dst
//   if (parser.parseOperand(srcOperand) || parser.parseComma() || parser.parseOperand(dstOperand))
//     return failure();
//   if (parser.parseOptionalAttrDict(result.attributes))// 解析属性字典: {dim=0, clear=1, kind="max"}
//     return failure();
//   // 解析类型: : memref<1xf16>, memref<1024xf16>
//   if (parser.parseColon() || parser.parseType(srcType) || parser.parseComma() || parser.parseType(dstType))
//     return failure();
//   // 解析操作数
//   if (parser.resolveOperand(srcOperand, srcType, result.operands) || 
//       parser.resolveOperand(dstOperand, dstType, result.operands))
//     return failure();
//   return success();
// }

// // 自定义汇编格式打印
// void ReduceOp::print(OpAsmPrinter &p) {
//   p << " " << getSrc() << ", " << getDst();
//   // 打印属性字典
//   p << " {";
//   p << "dim = " << getDim();
//   p << ", clear = " << getClear();
//   p << ", kind = \"" << getKind() << "\"";
//   p << "}";
//   p << " : " << getSrc().getType() << ", " << getDst().getType();
// }



} // namespace frisk
} // namespace mlir
