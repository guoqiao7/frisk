"""Semantic trace checks shared by lit and the actual Attention acceptance run."""
import json
import pathlib
import sys


def check(path, mode):
    raw = pathlib.Path(path).read_bytes()
    doc = json.loads(raw)
    assert doc["schema_version"] == 1
    settings = doc["solver_options"]
    assert set(settings) == {"exactCombinationLimit", "beamWidth", "maxExpandedStates",
                             "refineBeamSolution", "maxRefinementProbes"}
    assert type(settings["refineBeamSolution"]) is bool
    assert all(type(value) is int for key, value in settings.items() if key != "refineBeamSolution")
    assert len(raw) <= doc["trace_max_bytes"]
    assert len(doc["events"]) <= doc["trace_max_events"]
    assert doc["eventsSeen"] == len(doc["events"]) + doc["eventsOmitted"]
    assert "committed" not in doc["analysis_status"]
    assert "committed" not in doc["verification_status"]
    if mode == "failed":
        assert doc["finalCost"] is None
        assert doc["failure_kind"]
        assert doc["analysis_status"] == "failed"
        if doc["failure_kind"] == "search-budget":
            assert doc["verification_status"] == "not-started"
        return doc
    assert doc["finalCost"] is not None
    total = list(doc["omittedCost"]["cost"])
    saturated = doc["omittedCost"]["saturated"]
    limit = 2**64 - 1
    for event in doc["events"]:
        if event["event"] == "cost-contribution" and event["costScope"] == doc["costScope"]:
            values = [a + b for a, b in zip(total, event["estimate"]["cost"])]
            saturated |= event["estimate"]["saturated"] or any(v > limit for v in values)
            total = [min(v, limit) for v in values]
        if event["event"] == "search-depth":
            assert len(event["retainedExamples"]) <= 4
            assert len(event["discardedExamples"]) <= 4
            assert event["costKind"] in ("complete", "partial-lower-bound")
        if event["event"] == "refinement-probe":
            assert event["status"] in ("Proven", "Disproven", "Unknown", "NoSupport")
            assert event["legal"] == (event["status"] == "Proven")
            assert len(event["oldCandidate"]) == len(event["newCandidate"]) == 64
            assert "estimate" not in event and "costScope" not in event
            if event["result"] not in ("cost-failure", "constraint-not-proven"):
                assert len(event["cost"]["cost"]) == 9
        if event["event"] == "refinement-global-selection":
            original = event["originalCost"]
            refined = event["refinedCost"]
            selected = event["selectedCost"]
            assert selected["cost"] <= original["cost"]
            assert event["fallback"] == (original["cost"] < refined["cost"])
            assert selected == (original if event["fallback"] else refined)
            assert "estimate" not in event and "costScope" not in event
    assert {"cost": total, "saturated": saturated} == doc["finalCost"]
    if mode == "capped":
        assert doc["truncated"] and doc["eventsOmitted"]
        return doc
    assert not doc["truncated"]
    assert doc["costExplanationComplete"]
    graphs = [e for e in doc["events"] if e["event"] == "graph"]
    assert graphs
    graph = graphs[0]
    ids = {v["id"] for v in graph["variables"]}
    if ids:
        stages = {e["stage"]: e for e in doc["events"] if e["event"] == "domains" and e["attempt"] == doc["costScope"]}
        assert {"prepared", "strict", "common"} <= stages.keys()
        for event in stages.values():
            assert {d["variable"] for d in event["domains"]} == ids
            for domain in event["domains"]:
                for candidate in domain["candidates"]:
                    assert len(candidate["id"]) == 64 and candidate["attribute"]
        solutions = [e for e in doc["events"] if e["event"] == "solution"]
        assert {a["variable"] for a in solutions[-1]["assignments"]} == ids
    if mode == "attention":
        assert len(ids) > 15
        assert any(c["supports"] for c in graph["constraints"])
        kinds = {e["kind"] for e in doc["events"] if e["event"] == "cost-contribution"}
        assert {"registers", "shared-root", "mma", "reduction"} <= kinds
    if mode == "refinement-disabled":
        assert not settings["refineBeamSolution"] or settings["maxRefinementProbes"] == 0
        assert not any(e["event"].startswith("refinement-") for e in doc["events"])
        assert doc["finalCost"]["cost"] == [8293, 608, 64, 0, 428, 6144, 6, 0, 0]
    if mode == "refinement-enabled":
        assert settings["refineBeamSolution"] and settings["maxRefinementProbes"] == 256
        assert doc["finalCost"]["cost"] <= [8293, 576, 64, 0, 428, 6144, 6, 0, 0]
        stops = [e for e in doc["events"] if e["event"] == "refinement-stop"]
        assert stops and all(e["reason"] == "no-improvement" for e in stops)
        assert sum(e["accepted"] for e in stops) > 0
        assert sum(e["probes"] for e in stops) == sum(e["event"] == "refinement-probe" for e in doc["events"])
    return doc


if __name__ == "__main__":
    if sys.argv[2] == "compare-stats":
        def strip_timings(value):
            if isinstance(value, dict):
                return {k: strip_timings(v) for k, v in value.items() if k != "timings"}
            if isinstance(value, list):
                return [strip_timings(v) for v in value]
            return value
        assert strip_timings(json.loads(pathlib.Path(sys.argv[1]).read_text())) == strip_timings(json.loads(pathlib.Path(sys.argv[3]).read_text()))
    elif sys.argv[2] == "compare-trace":
        # Trace currently has no nondeterministic fields: compare everything.
        assert check(sys.argv[1], "full") == check(sys.argv[3], "full")
    else:
        check(sys.argv[1], sys.argv[2])
