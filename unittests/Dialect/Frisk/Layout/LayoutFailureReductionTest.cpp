#include "LayoutFailureCase.h"
#include "gtest/gtest.h"
using namespace mlir;
using namespace mlir::frisk;
using namespace mlir::frisk::testing;
namespace {
LayoutFailureCase example() {
  LayoutFailureCase c;
  c.variables = {{"empty", {}}, {"unrelated", {"a", "b"}}};
  c.signature = {"EmptyDomain", "empty", "", "", ""};
  return c;
}
TEST(LayoutFailureReductionTest, RoundTripPreservesStableIdentityAndHost) {
  MLIRContext ctx; std::string error;
  auto input = example();
  auto decoded = parseFailureCase(serializeFailureCase(input), ctx, error);
  ASSERT_TRUE(succeeded(decoded)) << error;
  EXPECT_EQ(serializeFailureCase(*decoded), serializeFailureCase(input));
}
TEST(LayoutFailureReductionTest, RejectsSchemaUnknownPayloadAndReferences) {
  MLIRContext ctx; std::string error;
  for (auto text : {"{}", "{\"schema\":99}", "{\"schema\":1,\"alias\":{}}"})
    EXPECT_TRUE(failed(parseFailureCase(text,ctx,error)));
  auto c = example();
  c.relations = {{"bad", "same", "r", "why", {"missing", "empty"}, ""}};
  EXPECT_TRUE(failed(parseFailureCase(serializeFailureCase(c),ctx,error)));
  c = example(); c.host = "not valid mlir";
  ScopedDiagnosticHandler quiet(&ctx, [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(parseFailureCase(serializeFailureCase(c),ctx,error)));
}
TEST(LayoutFailureReductionTest, ReductionPreservesSignatureAndIsOneMinimal) {
  MLIRContext ctx;
  auto c = example();
  auto signature = replayFailureCase(c,ctx);
  ASSERT_TRUE(succeeded(signature)); c.signature = *signature;
  auto reduced = reduceFailureCase(c,ctx);
  ASSERT_TRUE(succeeded(reduced));
  ASSERT_EQ(reduced->variables.size(),1u);
  EXPECT_EQ(reduced->variables.front().id,"empty");
  EXPECT_TRUE(reduced->signature == c.signature);
  auto twice = reduceFailureCase(*reduced,ctx);
  ASSERT_TRUE(succeeded(twice));
  EXPECT_EQ(serializeFailureCase(*twice),serializeFailureCase(*reduced));
  reduced->variables.clear();
  EXPECT_TRUE(failed(replayFailureCase(*reduced,ctx)));
}
TEST(LayoutFailureReductionTest, DifferentFailureCannotReplaceOriginal) {
  MLIRContext ctx; auto c = example();
  c.signature.kind = "HardConflict";
  EXPECT_TRUE(failed(reduceFailureCase(c,ctx)));
}
TEST(LayoutFailureReductionTest, HardConflictPreservesProvenanceAndRejectsEveryFurtherDeletion) {
  MLIRContext ctx; std::string error;
  LayoutFailureCase c;
  c.variables={{"v",{"a","b","c"}},{"unused",{"a"}}};
  c.relations={{"must-a","require","required-layout","original declaration",{"v"},"a"},
               {"must-b","require","required-layout","conflicting declaration",{"v"},"b"}};
  auto s=replayFailureCase(c,ctx);ASSERT_TRUE(succeeded(s));c.signature=*s;
  EXPECT_EQ(s->kind,"HardConflict"); EXPECT_FALSE(s->relation.empty());
  EXPECT_EQ(s->rule,"required-layout");
  auto reduced=reduceFailureCase(c,ctx);ASSERT_TRUE(succeeded(reduced));
  auto parsed=parseFailureCase(serializeFailureCase(*reduced),ctx,error);
  ASSERT_TRUE(succeeded(parsed))<<error;
  auto twice=reduceFailureCase(*parsed,ctx);ASSERT_TRUE(succeeded(twice));
  EXPECT_EQ(serializeFailureCase(*twice),serializeFailureCase(*reduced));
  for(size_t i=0;i<reduced->relations.size();++i) {
    auto t=*reduced;t.relations.erase(t.relations.begin()+i);
    auto why=replayFailureCase(t,ctx);EXPECT_TRUE(failed(why)||!(*why==*s));
  }
  for(size_t i=0;i<reduced->variables.size();++i) {
    auto t=*reduced;t.variables.erase(t.variables.begin()+i);
    auto why=replayFailureCase(t,ctx);EXPECT_TRUE(failed(why)||!(*why==*s));
    for(size_t j=0;j<reduced->variables[i].candidates.size();++j) {
      t=*reduced;auto &v=t.variables[i].candidates;v.erase(v.begin()+j);
      why=replayFailureCase(t,ctx);EXPECT_TRUE(failed(why)||!(*why==*s));
    }
  }
}
TEST(LayoutFailureReductionTest, RejectsEveryUnsupportedNestedFieldAndDuplicateIdentity) {
  MLIRContext ctx;std::string error;auto c=example();
  for(StringRef key:{"ssa","use","alias","region","anchor","operationExecution"}) {
    auto value=llvm::json::parse(serializeFailureCase(c));ASSERT_TRUE(bool(value));
    auto *vars=value->getAsObject()->getArray("variables");
    (*vars)[0].getAsObject()->try_emplace(key,nullptr);
    std::string text;llvm::raw_string_ostream(text)<<*value;
    EXPECT_TRUE(failed(parseFailureCase(text,ctx,error)))<<key.str();
  }
  c.variables.push_back(c.variables.front());
  EXPECT_TRUE(failed(parseFailureCase(serializeFailureCase(c),ctx,error)));
  c=example();c.relations={{"convert","convertible","r","why",{"empty","unrelated"},""}};
  EXPECT_TRUE(failed(parseFailureCase(serializeFailureCase(c),ctx,error)));
}
TEST(LayoutFailureReductionTest, RawEntryRebuildsAndFailsClosedOnBorrowedIdentity) {
  MLIRContext ctx;auto c=example();auto graph=rebuildFailureGraph(c,ctx);ASSERT_TRUE(succeeded(graph));
  auto predicate=[](const LayoutConstraintGraph &g){
    auto id=g.lookupVariable("empty");return id && g.getVariable(*id).candidates.empty();
  };
  auto reduced=reduceFailingLayoutGraph(*graph,predicate);ASSERT_TRUE(succeeded(reduced));
  EXPECT_EQ(reduced->getVariables().size(),1u);
  auto host=OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&ctx)));
  graph->getVariable(0).anchor=*host;
  EXPECT_TRUE(failed(reduceFailingLayoutGraph(*graph,predicate)));
}
TEST(LayoutFailureReductionTest, RawEntryPreservesOrdinalSensitiveFailure) {
  MLIRContext ctx; LayoutFailureCase c; c.variables={{"v",{"a"}}};
  auto graph=rebuildFailureGraph(c,ctx); ASSERT_TRUE(succeeded(graph));
  graph->getVariable(0).candidates.front().stableOrdinal=99;
  auto predicate=[](const LayoutConstraintGraph &g) {
    auto id=g.lookupVariable("v");
    return id && g.getVariable(*id).candidates.size()==1 &&
      g.getVariable(*id).candidates.front().stableOrdinal==99;
  };
  ASSERT_TRUE(predicate(*graph));
  auto reduced=reduceFailingLayoutGraph(*graph,predicate);
  ASSERT_TRUE(succeeded(reduced));
  EXPECT_TRUE(predicate(*reduced));
}
TEST(LayoutFailureReductionTest, RawCandidateDeletionPreservesSurvivorOrdinal) {
  MLIRContext ctx; LayoutFailureCase c; c.variables={{"v",{"a","b"}}};
  auto graph=rebuildFailureGraph(c,ctx); ASSERT_TRUE(succeeded(graph));
  auto predicate=[](const LayoutConstraintGraph &g) {
    auto id=g.lookupVariable("v"); if(!id) return false;
    for(auto &a:g.getVariable(*id).candidates)
      if(cast<StringAttr>(a.value).getValue()=="b" && a.stableOrdinal==1) return true;
    return false;
  };
  auto reduced=reduceFailingLayoutGraph(*graph,predicate);
  ASSERT_TRUE(succeeded(reduced)); EXPECT_TRUE(predicate(*reduced));
  ASSERT_EQ(reduced->getVariables().size(),1u);
  ASSERT_EQ(reduced->getVariable(0).candidates.size(),1u);
  EXPECT_EQ(reduced->getVariable(0).candidates.front().stableOrdinal,1u);
}
TEST(LayoutFailureReductionTest, RawEntryRejectsPredicateLostByInitialCanonicalization) {
  MLIRContext ctx;LayoutFailureCase c;c.variables={{"a",{"a"}},{"z",{"a"}}};
  auto graph=rebuildFailureGraph(c,ctx);ASSERT_TRUE(succeeded(graph));
  // Legal stable identities, but a predicate that is sensitive to insertion
  // order cannot be represented by the raw entry's canonical fixture form.
  graph->getVariable(0).stableName="z";graph->getVariable(1).stableName="a";
  auto predicate=[](const LayoutConstraintGraph &g) {
    return g.getVariables().size()==2 && g.getVariable(0).stableName=="z";
  };
  ASSERT_TRUE(succeeded(graph->verifyInvariants(UnknownLoc::get(&ctx))));
  ASSERT_TRUE(predicate(*graph));
  EXPECT_TRUE(failed(reduceFailingLayoutGraph(*graph,predicate)));
}
} // namespace
