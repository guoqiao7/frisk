#include "TileLangCorpus.h"
#include "gtest/gtest.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "Dialect/Frisk/Analysis/ExecutionLayoutProof.h"
#include "Dialect/Frisk/Analysis/ReductionLayoutProof.h"
#include "Dialect/Frisk/Analysis/LayoutRelations.h"
#include "Dialect/Frisk/Analysis/LayoutVerifier.h"
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/IR/FriskOps.h"
#include "Dialect/Frisk/Target/SM90/SM90GemmConstraints.h"
#include "Dialect/Frisk/Target/SM90/SM90LayoutTarget.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>
#include <set>

using namespace mlir;
using namespace mlir::frisk;
using namespace mlir::frisk::test;
namespace {
std::string minimalCorpus() {
  return R"json({"schema_version":2,
    "tilelang_commit":"5e149e31674658f94779c7d0c6039549a1853123",
    "dependencies":{"3rdparty/tvm":"907a88c8791ccf33b9874821bc875e7abf624367","3rdparty/cutlass":"b2dd65dc864e09688245b316ac46c4a6cd07e15c"},
    "exporter_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "source_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "cases":[{"name":"replicated","operation":"owner_compatibility","shape":[1],"valid_shape":[1],"dtype":"f32",
    "carrier_extents":[1,2,1,1,1],"storage_bytes":0,"value_semantics":"complete","writer_policy":"compatible_first_owner",
    "parameters":{},"frisk_expectation":{"kind":"equivalent","diagnostic":"","reason":""},
    "distributed_points":[{"carrier":[0,0,0,0,0],"logical":[0],"copy_group":0,"addend":-1,"valid":true},
      {"carrier":[0,1,0,0,0],"logical":[0],"copy_group":0,"addend":-1,"valid":true}],
    "storage_points":[],"writer_owners":[[0,0,0,0,0]]}]})json";
}
llvm::json::Value fixture() {
  auto parsed=llvm::json::parse(minimalCorpus());
  if(!parsed)llvm::report_fatal_error(llvm::toString(parsed.takeError()).c_str());
  return std::move(*parsed);
}
std::string serialize(const llvm::json::Value &value) {
  std::string text; llvm::raw_string_ostream os(text); os << value; return text;
}
llvm::json::Object &first(llvm::json::Value &v) {
  return *v.getAsObject()->getArray("cases")->front().getAsObject();
}
void rejected(llvm::json::Value value, llvm::StringRef category) {
  auto result = readTileLangCorpus(serialize(value));
  ASSERT_FALSE(result);
  EXPECT_NE(llvm::toString(result.takeError()).find(category.str()), std::string::npos);
}
}

TEST(TileLangCorpusReaderTest, AcceptsExplicitSameValueCopies) {
  auto result = readTileLangCorpus(minimalCorpus());
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->cases.size(), 1u);
}
TEST(TileLangCorpusReaderTest, RejectsSchemaRevisionAndDependencyMismatch) {
  auto v=fixture(); (*v.getAsObject())["schema_version"]=1; rejected(std::move(v),"schema");
  v=fixture(); (*v.getAsObject())["tilelang_commit"]="other"; rejected(std::move(v),"revision");
  v=fixture(); (*v.getAsObject()->getObject("dependencies"))["3rdparty/tvm"]="other"; rejected(std::move(v),"dependency");
}
TEST(TileLangCorpusReaderTest, RejectsDuplicateNamesCarriersAndConflictingCarriers) {
  auto v=fixture(); auto *cases=v.getAsObject()->getArray("cases"); cases->push_back(cases->front()); rejected(std::move(v),"duplicate case");
  v=fixture(); auto *points=first(v).getArray("distributed_points"); points->push_back(points->front()); rejected(std::move(v),"carrier");
  v=fixture(); points=first(v).getArray("distributed_points"); (*points)[1].getAsObject()->operator[]("carrier")=llvm::json::Array{0,0,0,0,0}; rejected(std::move(v),"carrier");
}
TEST(TileLangCorpusReaderTest, RejectsMissingCoverageInvalidWriterAndUnexplainedCopies) {
  auto v=fixture(); first(v)["shape"]=llvm::json::Array{2}; first(v)["valid_shape"]=llvm::json::Array{2}; rejected(std::move(v),"coverage");
  v=fixture(); first(v)["writer_owners"]=llvm::json::Array{llvm::json::Array{0,1,0,0,0}}; rejected(std::move(v),"writer");
  v=fixture(); (*first(v).getArray("distributed_points"))[1].getAsObject()->operator[]("copy_group")=1; rejected(std::move(v),"copy group");
}
TEST(TileLangCorpusReaderTest, RejectsOutOfRangeStorageBitsAndUnknownFields) {
  auto v=fixture(); first(v)["storage_bytes"]=4;
  first(v)["storage_points"]=llvm::json::Array{llvm::json::Object{{"logical",llvm::json::Array{0}},{"byte_offset",0},{"bit_offset",8}}};
  rejected(std::move(v),"bit offset");
  v=fixture(); first(v)["shell_command"]="never execute"; rejected(std::move(v),"unknown field");
}

TEST(TileLangCorpusReaderTest, DistinguishesPartialAddendsFromCompletedCopies) {
  auto v=fixture(); auto &c=first(v);
  c["operation"]="partial_fragment";
  c["value_semantics"]="partial"; c["writer_policy"]="none";
  c["writer_owners"]=llvm::json::Array{};
  c["parameters"]=llvm::json::Object{{"combine",2}};
  c["frisk_expectation"]=llvm::json::Object{{"kind","reject"},{"diagnostic","partial-contribution"},{"reason","unsupported partial values"}};
  auto &p=*c.getArray("distributed_points");
  (*p[0].getAsObject())["addend"]=0; (*p[1].getAsObject())["addend"]=1;
  auto valid=readTileLangCorpus(serialize(v)); ASSERT_TRUE(bool(valid))<<llvm::toString(valid.takeError());
  auto bad=v; first(bad)["value_semantics"]="complete"; rejected(std::move(bad),"addend semantics");
  bad=v; (*first(bad).getArray("distributed_points"))[1].getAsObject()->operator[]("addend")=0; rejected(std::move(bad),"partial contribution");
  bad=v; (*first(bad).getArray("distributed_points"))[1].getAsObject()->operator[]("addend")=2; rejected(std::move(bad),"partial contribution");
}

TEST(TileLangCorpusReaderTest, RejectsInvalidDomainAndResourceBounds) {
  auto v=fixture(); (*first(v).getArray("distributed_points"))[0].getAsObject()->operator[]("valid")=false; rejected(std::move(v),"valid domain");
  v=fixture(); first(v)["carrier_extents"]=llvm::json::Array{65536,65536,1,1,1}; rejected(std::move(v),"budget");
  v=fixture(); first(v)["carrier_extents"]=llvm::json::Array{0,1,1,1,1}; rejected(std::move(v),"extent");
}

TEST(TileLangCorpusReaderTest, RejectsMissingOperationParameters) {
  auto v=fixture(); first(v)["operation"]="copy"; rejected(std::move(v),"operation parameters");
  v=fixture(); first(v)["operation"]="execute_path"; rejected(std::move(v),"operation");
}

namespace {
class TileLangDifferentialTest : public testing::Test {
protected:
  MLIRContext context;
  Builder b{&context};
  TileLangDifferentialTest() { context.loadDialect<FriskDialect,func::FuncDialect,memref::MemRefDialect>(); }
  Type dtype(const TileLangCase &c) {
    if(c.dtype=="f16")return b.getF16Type();
    if(c.dtype=="bf16")return b.getBF16Type();
    if(c.dtype=="i8")return b.getI8Type();
    return b.getF32Type();
  }
  FailureOr<DistributedEncodingAttr> constructEncoding(const TileLangCase &c);
  FailureOr<StorageLayoutAttr> constructStorage(const TileLangCase &c);
  void compare(const TileLangCase &c);
  void reject(const TileLangCase &c);
};
}

TEST_F(TileLangDifferentialTest, AuditedPinnedCorpus) {
  auto file=llvm::MemoryBuffer::getFile(FRISK_TILELANG_CORPUS);
  ASSERT_TRUE(bool(file)) << "missing actual pinned TileLang export: " << FRISK_TILELANG_CORPUS;
  auto corpus=readTileLangCorpus((*file)->getBuffer());
  ASSERT_TRUE(bool(corpus)) << llvm::toString(corpus.takeError());
  size_t equivalent=0,rejected=0;
  std::set<std::string> categories;
  for(const auto &c:corpus->cases) {
    SCOPED_TRACE(c.name);
    categories.insert(c.operation);
    if(c.expectation=="equivalent") { compare(c); ++equivalent; }
    else { reject(c); ++rejected; }
  }
  EXPECT_EQ(categories,(std::set<std::string>{"gemm","shared","copy","reduce","owner_compatibility","broadcast","guarded_copy","dtype_view","partial_fragment"}));
  EXPECT_EQ(equivalent,20u);
  EXPECT_EQ(rejected,3u);
  RecordProperty("equivalent_cases",int(equivalent));
  RecordProperty("expected_rejection_cases",int(rejected));
  llvm::outs()<<"TileLang corpus: "<<equivalent<<" equivalent; "<<rejected<<" expected rejections (not supported equivalences)\n";
}

namespace {
constexpr const char *carrierNames[]={"register","lane","warp","warp_group","cta"};
std::string coordinates(ArrayRef<int64_t> point) {
  std::string text; llvm::raw_string_ostream os(text); os<<"[";
  llvm::interleaveComma(point,os);os<<"]";return text;
}

// Build a real GF(2) map from an independently specified register packing
// contract. Only shape/operation parameters are read here, NEVER corpus points.
DistributedEncodingAttr packing(MLIRContext *ctx, ArrayRef<int64_t> shape,
    int64_t registers, function_ref<int64_t(int64_t,int64_t)> flatCoordinate) {
  Builder b(ctx);
  SmallVector<Attribute> inputs,outputs;
  SmallVector<int64_t> widths,outWidths;
  if(registers>1) { inputs.push_back(b.getStringAttr("register")); widths.push_back(llvm::Log2_64(registers)); }
  inputs.push_back(b.getStringAttr("lane")); widths.push_back(5);
  int64_t rows=0, columns=5+llvm::Log2_64(registers), logicalCount=1;
  for(auto [i,e]:llvm::enumerate(shape)) {
    outputs.push_back(b.getStringAttr("dim"+Twine(i)));
    outWidths.push_back(llvm::Log2_64(e));rows+=outWidths.back();logicalCount*=e;
  }
  SmallVector<APInt> bits(rows*columns, APInt(1,0));
  for(int64_t col=0;col<columns;++col) {
    auto flat=flatCoordinate(col<int64_t(llvm::Log2_64(registers)) ? int64_t(1)<<col : 0,
        col>=int64_t(llvm::Log2_64(registers)) ? int64_t(1)<<(col-llvm::Log2_64(registers)) : 0);
    SmallVector<int64_t> logical(shape.size());
    for(size_t i=shape.size();i-->0;) {logical[i]=flat%shape[i];flat/=shape[i];}
    int64_t row=0;
    for(size_t i=0;i<shape.size();++i)for(int64_t bit=0;bit<outWidths[i];++bit)
      bits[(row++)*columns+col]=APInt(1,(logical[i]>>bit)&1);
  }
  auto map=BitLinearLayoutMapAttr::get(ctx,b.getArrayAttr(inputs),b.getDenseI64ArrayAttr(widths),
      b.getArrayAttr(outputs),b.getDenseI64ArrayAttr(outWidths),
      DenseIntElementsAttr::get(RankedTensorType::get({rows,columns},b.getI1Type()),bits));
  return DistributedEncodingAttr::get(ctx,map,b.getDenseI64ArrayAttr({registers,32,1,1,1}),b.getI64IntegerAttr(registers*32/logicalCount));
}

FailureOr<DistributedEncodingAttr> TileLangDifferentialTest::constructEncoding(const TileLangCase &c) {
  if(c.operation=="gemm") {
    if(c.parameters.at("m")!=64 || c.parameters.at("n")!=64 || c.parameters.at("k")!=16)return failure();
    SM90MmaGeometry g{64,64,16,128,1,1,64,1,1,1};
    return buildSM90MmaFragment(&context,g,c.parameters.at("operand_a"),false);
  }
  if(c.operation=="reduce") {
    TileLangCase source=c;source.operation="copy";source.shape={8,16};source.parameters={{"vector_bytes",4}};
    auto encoding=constructEncoding(source);if(failed(encoding))return failure();
    if(!c.parameters.at("result"))return *encoding;
    return projectReductionEncoding(*encoding,RankedTensorType::get({8,16},dtype(c)),
        RankedTensorType::get(c.shape,dtype(c)),c.parameters.at("axis"));
  }
  if(c.operation=="copy") {
    int64_t width=c.parameters.at("vector_bytes");
    if(c.shape!=std::vector<int64_t>({8,16}) || (width!=4 && width!=8 && width!=16))return failure();
    int64_t elements=width/4;
    return packing(&context,c.shape,4,[=](int64_t reg,int64_t lane) {
      return (reg/elements)*32*elements+lane*elements+reg%elements;
    });
  }
  if(c.operation=="guarded_copy") {
    if(c.shape!=std::vector<int64_t>({4,8}) || c.validShape!=std::vector<int64_t>({3,5}))return failure();
    return packing(&context,c.shape,1,[](int64_t,int64_t lane){return lane;});
  }
  if(c.operation=="owner_compatibility") {
    if(c.shape!=std::vector<int64_t>({4}))return failure();
    return packing(&context,c.shape,4,[](int64_t reg,int64_t){return reg;});
  }
  return failure();
}

FailureOr<StorageLayoutAttr> TileLangDifferentialTest::constructStorage(const TileLangCase &c) {
  if(c.operation!="shared" && c.operation!="copy" && c.operation!="guarded_copy")return failure();
  int64_t bytes=dtype(c).getIntOrFloatBitWidth()/8;
  AffineExpr address=b.getAffineConstantExpr(0);
  int64_t capacity=bytes;
  SmallVector<Attribute> names;
  for(auto [i,extent]:llvm::enumerate(c.validShape)) {
    address=address*extent+b.getAffineDimExpr(i);
    capacity*=extent;
    names.push_back(b.getStringAttr("dim"+Twine(i)));
  }
  address=address*bytes;
  if(c.operation=="shared") {
    if(c.shape!=std::vector<int64_t>({8,64}) || c.dtype!="f16")return failure();
    int64_t sw=c.parameters.at("swizzle");
    auto row=b.getAffineDimExpr(0),col=b.getAffineDimExpr(1);
    if(sw==-1)address=(col*c.shape[0]+row)*bytes;
    else if(sw) {
      if(sw!=32 && sw!=64 && sw!=128)return failure();
      // PTX canonical K-major panels; XOR byte bits 4..6 with 7..9.
      auto raw=row*sw+(col%(sw/bytes))*bytes+col.floorDiv(sw/bytes)*(c.shape[0]*sw);
      address=raw;
      for(int64_t bit=16;bit<sw;bit*=2) {
        auto original=raw.floorDiv(bit)%2;
        address=address+(((original+raw.floorDiv(bit*8)%2)%2)-original)*bit;
      }
    }
  }
  auto map=AffineLayoutMapAttr::get(&context,b.getArrayAttr(names),b.getDenseI64ArrayAttr(c.validShape),
      b.getArrayAttr({b.getStringAttr("byte_offset"),b.getStringAttr("bit_offset")}),
      b.getDenseI64ArrayAttr({capacity,8}),
      AffineMapAttr::get(AffineMap::get(c.validShape.size(),0,{address,b.getAffineConstantExpr(0)},&context)));
  return StorageLayoutAttr::get(&context,map,MemorySpaceAttr::get(&context,attr::MemorySpace::Shared),b.getI64IntegerAttr(16),b.getI64IntegerAttr(1));
}

void TileLangDifferentialTest::compare(const TileLangCase &c) {
  DistributedEncodingAttr encoding;
  auto type=RankedTensorType::get(c.shape,dtype(c));
  if(!c.distributed.empty()) {
    auto built=constructEncoding(c);ASSERT_TRUE(succeeded(built))<<"unsupported independent construction";
    encoding=*built;
    ASSERT_TRUE(succeeded(encoding.verifyForType(type,b.getUnknownLoc())));
    auto map=cast<BitLinearLayoutMapAttr>(encoding.getMap());
    ASSERT_EQ(encoding.getTopology().asArrayRef(),ArrayRef<int64_t>(c.extents));
    std::map<std::vector<int64_t>,std::set<CorpusCarrier>> actualOwners,referenceOwners;
    // Full physical domain, not just the JSON's listed points.
    for(int64_t reg=0;reg<c.extents[0];++reg)for(int64_t lane=0;lane<c.extents[1];++lane)
      for(int64_t warp=0;warp<c.extents[2];++warp)for(int64_t group=0;group<c.extents[3];++group)
        for(int64_t cta=0;cta<c.extents[4];++cta) {
          CorpusCarrier carrier{reg,lane,warp,group,cta}; SmallVector<int64_t> inputs;
          for(Attribute name:map.getInputNames()) {
            auto text=cast<StringAttr>(name).getValue();
            unsigned index=0;while(index<5 && text!=carrierNames[index])++index;
            ASSERT_LT(index,5u);inputs.push_back(carrier[index]);
          }
          auto value=evaluateLayoutMap(map,inputs);ASSERT_TRUE(succeeded(value));
          actualOwners[std::vector<int64_t>(value->begin(),value->end())].insert(carrier);
        }
    for(auto &point:c.distributed)referenceOwners[point.logical].insert(point.carrier);
    ASSERT_EQ(actualOwners.size(),referenceOwners.size())<<"logical coverage";
    for(auto &entry:referenceOwners) {
      ASSERT_EQ(actualOwners[entry.first],entry.second)<<"first owner counterexample logical="<<coordinates(entry.first);
      ASSERT_EQ(int64_t(entry.second.size()),encoding.getReplication().getInt())<<"copy-group cardinality logical="<<coordinates(entry.first);
    }
    auto ownerProof=proveExecutionOwnership(encoding,type,"first_owner");
    ASSERT_EQ(ownerProof.status,ProofStatus::Proven)<<ownerProof.reason;
    ExecutionEnumeration enumeration;
    ASSERT_EQ(enumerateExecutionLayout(encoding,type,enumeration).status,ProofStatus::Proven);
    std::set<CorpusCarrier> actualWriters;
    for(uint64_t logical=0;logical<enumeration.first.size();++logical) {
      auto coordinate=enumeration.coordinate(logical);bool valid=true;
      for(size_t i=0;i<coordinate.size();++i) valid&=coordinate[i]<c.validShape[i];
      if(!valid)continue;
      auto physical=enumeration.hardware(enumeration.first[logical]);
      actualWriters.insert({physical[4],physical[3],physical[2],physical[1],physical[0]});
    }
    ASSERT_EQ(actualWriters,std::set<CorpusCarrier>(c.writers.begin(),c.writers.end()))<<"writer policy compatibility";
    if(c.operation=="gemm") {
      SM90MmaGeometry g{64,64,16,128,1,1,64,1,1,1};
      auto proof=verifySM90MmaFragment(g,c.parameters.at("operand_a"),false,type,encoding);
      ASSERT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
    }
    if(c.operation=="reduce" && c.parameters.at("result")) {
      TileLangCase input=c;input.parameters["result"]=0;input.shape={8,16};
      auto source=constructEncoding(input);ASSERT_TRUE(succeeded(source));
      auto proof=buildReductionLayoutProof(*source,encoding,RankedTensorType::get({8,16},dtype(c)),type,c.parameters.at("axis"));
      ASSERT_EQ(proof.proof.status,ProofStatus::Proven)<<proof.proof.reason;
      ASSERT_EQ(verifyReductionLayoutProof(*source,encoding,RankedTensorType::get({8,16},dtype(c)),type,c.parameters.at("axis"),proof).status,ProofStatus::Proven);
    }
  }
  if(!c.storage.empty()) {
    auto built=constructStorage(c);ASSERT_TRUE(succeeded(built));
    auto storage=*built;
    auto memref=MemRefType::get(c.validShape,dtype(c),MemRefLayoutAttrInterface{},b.getI64IntegerAttr(3));
    ASSERT_TRUE(succeeded(storage.verifyForType(memref,b.getUnknownLoc())));
    auto map=cast<AffineLayoutMapAttr>(storage.getMap());
    ASSERT_EQ(map.getOutputExtents()[0],c.storageBytes)<<"storage capacity";
    for(auto &point:c.storage) {
      auto actual=evaluateLayoutMap(map,point.logical);ASSERT_TRUE(succeeded(actual));
      ASSERT_EQ((*actual)[0],point.byteOffset)<<"byte offset counterexample logical="<<coordinates(point.logical);
      ASSERT_EQ((*actual)[1],point.bitOffset)<<"bit offset counterexample logical="<<coordinates(point.logical);
    }
    if(encoding) {
      std::string spelling="memref<";
      for(auto e:c.validShape)spelling+=std::to_string(e)+"x";
      spelling+=c.dtype+",3>";
      auto module=parseSourceString<ModuleOp>("func.func @storage() { %root=memref.alloc() {alignment=16:i64} : "+spelling+
          " %v=frisk.layout_view %root : "+spelling+" -> "+spelling+" return }",&context);
      ASSERT_TRUE(module);
      LayoutViewOp view;module->walk([&](LayoutViewOp op){view=op;});
      ASSERT_TRUE(view);
      auto analyzed=analyzeStorageAlias(view.getResult());ASSERT_TRUE(succeeded(analyzed));
      const StorageAliasInfo &info=*analyzed;
      auto footprint=buildStorageAliasFootprint(info,storage);
      ASSERT_EQ(footprint.proof.status,ProofStatus::Proven)<<footprint.proof.reason;
      for(auto access:{AccessKind::Read,AccessKind::Write}) {
        auto proof=c.operation=="guarded_copy"
          ? proveGuardedExecutionVectorAccess(encoding,type,info,footprint,c.parameters.at("vector_bytes"),access,"first_owner")
          : proveExecutionVectorAccess(encoding,info,footprint,c.parameters.at("vector_bytes"),access,"first_owner");
        ASSERT_EQ(proof.status,ProofStatus::Proven)<<proof.reason;
      }
    }
  }
}

void TileLangDifferentialTest::reject(const TileLangCase &c) {
  std::string diagnostics;
  ScopedDiagnosticHandler handler(&context,[&](Diagnostic &d) {
    llvm::raw_string_ostream os(diagnostics);d.print(os);return success();
  });
  if(c.operation=="broadcast") {
    ASSERT_EQ(c.shape,std::vector<int64_t>({3}));
    ASSERT_EQ(c.diagnostic,"non-power-of-two");
    auto module=parseSourceString<ModuleOp>("func.func @broadcast_input(%x:tensor<3xf32>)->tensor<3xf32> { return %x:tensor<3xf32> }",&context);
    ASSERT_TRUE(module);
    auto target=createSM90LayoutTarget();
    EXPECT_TRUE(failed(collectLayoutConstraints(*module,*target)));
    EXPECT_NE(diagnostics.find("power-of-two"),std::string::npos)<<diagnostics;
  } else if(c.operation=="dtype_view") {
    ASSERT_EQ(c.shape,std::vector<int64_t>({8}));
    ASSERT_EQ(c.diagnostic,"dtype-changing-view");
    auto module=parseSourceString<ModuleOp>("func.func @view(%x:memref<4xf32,3>) { %v=frisk.layout_view %x : memref<4xf32,3> -> memref<8xf16,3> return }",&context);
    EXPECT_FALSE(module);
    EXPECT_NE(diagnostics.find("identical memref types"),std::string::npos)<<diagnostics;
  } else if(c.operation=="partial_fragment") {
    ASSERT_EQ(c.diagnostic,"partial-contribution");ASSERT_EQ(c.valueSemantics,"partial");
    NamedAttrList fields;
    fields.set("version",b.getI64IntegerAttr(1));fields.set("axis",b.getI64IntegerAttr(1));fields.set("threads",b.getI64IntegerAttr(32));
    fields.set("source_shape",b.getDenseI64ArrayAttr({4,c.parameters.at("combine")}));fields.set("result_shape",b.getDenseI64ArrayAttr(c.shape));
    for(auto [key,value]:{std::pair<const char *,const char *>{"target","sm_90"},{"kind","sum"},{"dtype","f32"},
        {"algorithm","canonical_fiber_tree_v1"},{"input_policy","first_owner"},{"output_policy","partial_addends"},{"scope","warp"}})
      fields.set(key,b.getStringAttr(value));
    EXPECT_TRUE(failed(ReductionContractAttr::verify([&]{return emitError(b.getUnknownLoc());},fields.getDictionary(&context))));
    EXPECT_NE(diagnostics.find("reduction contract requires complete v1 typed schema"),std::string::npos)<<diagnostics;
    fields.set("output_policy",b.getStringAttr("broadcast_complete"));
    EXPECT_TRUE(succeeded(ReductionContractAttr::verify([&]{return emitError(b.getUnknownLoc());},fields.getDictionary(&context))))<<"control contract must be valid without partial policy";
  } else FAIL()<<"unknown expected rejection category: "<<c.operation;
}
} // namespace
