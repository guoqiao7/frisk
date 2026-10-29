import json
import sys

with open(sys.argv[1]) as source:
    profile = json.load(source)
assert profile["version"] == 1
assert profile["status"] == sys.argv[2]
assert profile["timing_kind"] == "inclusive"
assert profile["entries_kind"] == "insertions"
assert "not additive" in profile["timing_note"]
expected = {"address_enumeration", "execution_enumeration", "access_cost",
            "conversion_classification", "conversion_choice", "instruction_support",
            "instruction_contract", "relation_query", "relation_cache",
            "alias_pair_cache", "footprint_cache"}
assert set(profile["metrics"]) == expected
for metric in profile["metrics"].values():
    assert set(metric) == {"count", "inclusive_ns", "hits", "misses", "entries"}
    assert all(isinstance(value, int) and value >= 0 for value in metric.values())
    assert metric["entries"] <= metric["misses"]
for name in sys.argv[3:]:
    assert profile["metrics"][name]["count"] > 0, name
for name in ("relation_cache", "alias_pair_cache", "footprint_cache"):
    metric = profile["metrics"][name]
    assert metric["count"] == metric["hits"] + metric["misses"]
