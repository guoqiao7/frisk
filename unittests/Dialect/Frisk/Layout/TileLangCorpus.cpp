#include "TileLangCorpus.h"
#include "llvm/Support/JSON.h"
#include <algorithm>
#include <set>

using namespace llvm;
namespace mlir::frisk::test {
namespace {
struct Reader {
  std::string error;
  bool check(bool ok, StringRef message) {
    if (!ok && error.empty()) error=message.str();
    return ok;
  }
  bool fields(const json::Object &o, std::initializer_list<StringRef> names) {
    std::set<std::string> allowed;
    for (auto name:names) allowed.insert(name.str());
    for (auto &entry:o)
      if (!check(allowed.count(entry.first.str()), "unknown field")) return false;
    return check(o.size()==allowed.size(), "missing field");
  }
  std::string str(const json::Object &o, StringRef key) {
    auto value=o.getString(key); check(bool(value), "expected string");
    return value ? value->str() : "";
  }
  int64_t integer(const json::Object &o, StringRef key) {
    auto value=o.getInteger(key); check(bool(value), "expected integer");
    return value.value_or(-1);
  }
  std::vector<int64_t> ints(const json::Array *array) {
    std::vector<int64_t> result;
    if (!check(array!=nullptr, "expected integer array")) return result;
    for (auto &value:*array) {
      auto n=value.getAsInteger();
      if (!check(bool(n), "expected integer")) return {};
      result.push_back(*n);
    }
    return result;
  }
  CorpusCarrier carrier(const json::Array *array) {
    auto values=ints(array); CorpusCarrier result{};
    if (check(values.size()==5,"carrier rank")) std::copy(values.begin(),values.end(),result.begin());
    return result;
  }
  bool hex(StringRef value, size_t size) {
    return value.size()==size && llvm::all_of(value, [](char c) {
      return (c>='0' && c<='9') || (c>='a' && c<='f');
    });
  }
  int64_t volume(ArrayRef<int64_t> extents) {
    int64_t n=1;
    for (auto e:extents) {
      if (!check(e>0 && e<=65536 && n<=65536/e, "extent/proof budget")) return 0;
      n*=e;
    }
    return n;
  }
  bool inDomain(ArrayRef<int64_t> p, ArrayRef<int64_t> shape) {
    if (p.size()!=shape.size()) return false;
    for (size_t i=0;i<p.size();++i) if(p[i]<0 || p[i]>=shape[i]) return false;
    return true;
  }
  // Physical first_owner order is CTA, warp group, warp, lane, register.
  static bool before(const CorpusCarrier &a,const CorpusCarrier &b) {
    return std::lexicographical_compare(a.rbegin(),a.rend(),b.rbegin(),b.rend());
  }
  bool readCase(const json::Object &o, TileLangCase &c) {
    if (!fields(o,{"name","operation","shape","valid_shape","dtype","carrier_extents","storage_bytes",
                   "value_semantics","writer_policy","parameters","frisk_expectation","distributed_points","storage_points","writer_owners"})) return false;
    c.name=str(o,"name"); c.operation=str(o,"operation"); c.dtype=str(o,"dtype");
    c.valueSemantics=str(o,"value_semantics"); c.writerPolicy=str(o,"writer_policy");
    c.shape=ints(o.getArray("shape")); c.validShape=ints(o.getArray("valid_shape"));
    c.extents=carrier(o.getArray("carrier_extents")); c.storageBytes=integer(o,"storage_bytes");
    auto *expect=o.getObject("frisk_expectation");
    if(!check(expect!=nullptr,"expectation") || !fields(*expect,{"kind","diagnostic","reason"})) return false;
    c.expectation=str(*expect,"kind");c.diagnostic=str(*expect,"diagnostic");c.reason=str(*expect,"reason");
    if(!check((c.expectation=="equivalent" && c.diagnostic.empty() && c.reason.empty()) ||
              (c.expectation=="reject" && !c.diagnostic.empty() && !c.reason.empty()),"expectation")) return false;
    if(!check(!c.name.empty() && !c.operation.empty(),"name/operation")) return false;
    if(!check(c.valueSemantics=="complete" || (c.valueSemantics=="partial" && c.expectation=="reject"),"value semantics")) return false;
    if(!check(c.writerPolicy=="compatible_first_owner" || c.writerPolicy=="none","writer policy")) return false;
    int bits=c.dtype=="f16" || c.dtype=="bf16" ? 16 : c.dtype=="f32" ? 32 : c.dtype=="i8" ? 8 : 0;
    if(!check(bits>0,"dtype") || !check(!c.shape.empty() && c.shape.size()<=4 && c.shape.size()==c.validShape.size(),"shape rank")) return false;
    int64_t logicalCount=volume(c.shape), validCount=volume(c.validShape), physicalCount=volume(c.extents);
    if(!logicalCount || !validCount || !physicalCount) return false;
    for(size_t i=0;i<c.shape.size();++i) if(!check(c.validShape[i]<=c.shape[i],"valid domain"))return false;
    if(!check(c.storageBytes>=0 && c.storageBytes<=65536*8,"storage bound")) return false;
    auto *params=o.getObject("parameters"); if(!check(params!=nullptr,"parameters"))return false;
    for(auto &p:*params) {auto n=p.second.getAsInteger();if(!check(bool(n),"parameter integer"))return false;c.parameters[p.first.str()]=*n;}
    std::set<std::string> parameterKeys;
    if(c.operation=="gemm")parameterKeys={"m","n","k","operand_a","input_bits"};
    else if(c.operation=="shared")parameterKeys={"swizzle"};
    else if(c.operation=="copy" || c.operation=="guarded_copy")parameterKeys={"vector_bytes"};
    else if(c.operation=="reduce")parameterKeys={"axis","result"};
    else if(c.operation=="dtype_view")parameterKeys={"source_bits"};
    else if(c.operation=="partial_fragment")parameterKeys={"combine"};
    else if(c.operation!="owner_compatibility" && c.operation!="broadcast")return check(false,"unknown operation");
    std::set<std::string> actualKeys;
    for(auto &p:c.parameters)actualKeys.insert(p.first);
    if(!check(parameterKeys==actualKeys,"operation parameters"))return false;
    int64_t combine=1;
    if(c.valueSemantics=="partial") {
      if(!check(c.operation=="partial_fragment","partial contribution operation"))return false;
      combine=c.parameters.at("combine");
      if(!check(combine>0 && combine<=65536,"partial contribution combine bound"))return false;
    }
    auto *distributed=o.getArray("distributed_points"), *storage=o.getArray("storage_points"), *writers=o.getArray("writer_owners");
    if(!check(distributed && storage && writers,"point arrays"))return false;
    if(!check(!distributed->empty() || !storage->empty(),"empty case coverage"))return false;
    if(!check(distributed->size()<=65536 && storage->size()<=65536 && writers->size()<=65536,"point budget"))return false;
    std::set<CorpusCarrier> carriers;
    std::map<std::vector<int64_t>,std::vector<CorpusPoint>> holders;
    for(auto &v:*distributed) {
      auto *p=v.getAsObject(); if(!check(p!=nullptr,"point object") || !fields(*p,{"carrier","logical","copy_group","addend","valid"}))return false;
      CorpusPoint point{carrier(p->getArray("carrier")),ints(p->getArray("logical")),integer(*p,"copy_group"),integer(*p,"addend"),false};
      auto valid=p->getBoolean("valid"); if(!check(bool(valid),"point valid"))return false;point.valid=*valid;
      if(!check(inDomain(point.carrier,c.extents),"carrier bounds") || !check(carriers.insert(point.carrier).second,"duplicate/conflicting carrier"))return false;
      if(!check(inDomain(point.logical,c.shape),"logical bounds") || !check(point.valid==inDomain(point.logical,c.validShape),"valid domain"))return false;
      if(!check(point.copyGroup>=0 && ((c.valueSemantics=="complete" && point.addend==-1) || (c.valueSemantics=="partial" && point.addend>=0)),"copy group/addend semantics"))return false;
      holders[point.logical].push_back(point);c.distributed.push_back(point);
    }
    if(!distributed->empty()) {
      if(!check(int64_t(carriers.size())==physicalCount,"carrier coverage") || !check(int64_t(holders.size())==logicalCount,"logical coverage"))return false;
      for(auto &entry:holders) {
        if(c.valueSemantics=="complete") {
          for(auto &p:entry.second) if(!check(p.copyGroup==entry.second.front().copyGroup,"unexplained copy group"))return false;
        } else {
          std::set<std::pair<int64_t,int64_t>> partials;
          int64_t maxGroup=0;
          for(auto &p:entry.second) {
            if(!check(p.addend<combine && p.copyGroup<65536 && partials.insert({p.copyGroup,p.addend}).second,"invalid/duplicate partial contribution"))return false;
            maxGroup=std::max(maxGroup,p.copyGroup);
          }
          if(!check(int64_t(partials.size())==(maxGroup+1)*combine,"incomplete partial contribution groups"))return false;
        }
      }
    }
    std::set<std::vector<int64_t>> stored;
    std::set<int64_t> occupiedBits;
    for(auto &v:*storage) {
      auto *p=v.getAsObject();if(!check(p!=nullptr,"storage object") || !fields(*p,{"logical","byte_offset","bit_offset"}))return false;
      CorpusStoragePoint point{ints(p->getArray("logical")),integer(*p,"byte_offset"),integer(*p,"bit_offset")};
      if(!check(point.bitOffset>=0 && point.bitOffset<8,"bit offset bounds"))return false;
      if(!check(point.byteOffset>=0 && point.byteOffset<c.storageBytes && point.byteOffset*8+point.bitOffset+bits<=c.storageBytes*8,"storage bit range"))return false;
      if(!check(inDomain(point.logical,c.validShape) && stored.insert(point.logical).second,"storage logical coverage"))return false;
      for(int64_t bit=point.byteOffset*8+point.bitOffset;bit<point.byteOffset*8+point.bitOffset+bits;++bit)
        if(!check(occupiedBits.insert(bit).second,"overlapping storage bits"))return false;
      c.storage.push_back(point);
    }
    if(!storage->empty() && !check(int64_t(stored.size())==validCount,"storage coverage"))return false;
    if(storage->empty() && !check(c.storageBytes==0,"empty storage capacity"))return false;
    std::set<CorpusCarrier> expectedWriters, actualWriters;
    if(c.writerPolicy=="compatible_first_owner") {
      if(!check(c.valueSemantics=="complete" && !holders.empty(),"writer partial/empty"))return false;
      for(auto &entry:holders) if(entry.second.front().valid) {
        CorpusCarrier elected=entry.second.front().carrier;
        for(auto &p:entry.second) if(before(p.carrier,elected))elected=p.carrier;
        expectedWriters.insert(elected);
      }
    }
    for(auto &v:*writers) { auto p=carrier(v.getAsArray()); if(!check(actualWriters.insert(p).second,"duplicate writer"))return false;c.writers.push_back(p); }
    return check(expectedWriters==actualWriters,"illegal or missing writer") && error.empty();
  }
};
}
Expected<TileLangCorpus> readTileLangCorpus(StringRef text) {
  if(text.size()>64*1024*1024) return createStringError(inconvertibleErrorCode(),"corpus size budget");
  auto parsed=json::parse(text);if(!parsed)return parsed.takeError();
  Reader r; auto *root=parsed->getAsObject(); TileLangCorpus result;
  auto parse=[&]() {
    if(!r.check(root!=nullptr,"root object") || !r.fields(*root,{"schema_version","tilelang_commit","dependencies","exporter_sha256","source_sha256","cases"}))return false;
    if(!r.check(r.integer(*root,"schema_version")==2,"schema version"))return false;
    if(!r.check(r.str(*root,"tilelang_commit")=="5e149e31674658f94779c7d0c6039549a1853123","revision mismatch"))return false;
    auto *deps=root->getObject("dependencies");if(!r.check(deps!=nullptr,"dependency revisions"))return false;
    if(!r.check(deps->getString("3rdparty/tvm")=="907a88c8791ccf33b9874821bc875e7abf624367" &&
                deps->getString("3rdparty/cutlass")=="b2dd65dc864e09688245b316ac46c4a6cd07e15c","dependency revision mismatch"))return false;
    for(auto &entry:*deps) if(!r.check(entry.second.getAsString() && r.hex(*entry.second.getAsString(),40),"dependency hash"))return false;
    for(StringRef key:{"exporter_sha256","source_sha256"}) if(!r.check(r.hex(r.str(*root,key),64),"source digest"))return false;
    auto *cases=root->getArray("cases");if(!r.check(cases && !cases->empty() && cases->size()<=256,"cases budget"))return false;
    std::set<std::string> names;
    for(auto &v:*cases) {
      auto *o=v.getAsObject(); TileLangCase c;
      if(!r.check(o!=nullptr,"case object") || !r.readCase(*o,c))return false;
      if(!r.check(names.insert(c.name).second,"duplicate case"))return false;
      result.cases.push_back(std::move(c));
    }
    return true;
  };
  if(!parse())return createStringError(inconvertibleErrorCode(),"TileLang corpus: %s",r.error.c_str());
  return result;
}
} // namespace mlir::frisk::test
