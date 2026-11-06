"""Check real CLI diagnostics, determinism and unchanged selection/costs."""
import json
import pathlib
import sys

prefix = sys.argv[1]


def read(suffix):
    return json.loads(pathlib.Path(prefix + suffix).read_text())


off, on, capped, replay = [read(f".{mode}.json") for mode in ("off", "on", "cap", "replay")]
assert off["actual_ir_diagnostics"] is None
arithmetic = {
    "schema_version": 1,
    "status": "available",
    "counting": "static-logical-result-elements-once-per-ir-op",
    "execution_frequency": "not-applied",
    "entries": [
        {"opcode": opcode, "dtype": "f32", "op_count": 1,
         "element_ops": 64, "saturated": False}
        for opcode in ("arith.divf", "arith.mulf", "math.exp")
    ],
    "unavailable_operations": [],
}
resources = {
    "schema_version": 1,
    "status": "available",
    "counting": "conservative-static-ssa-and-shared-root-liveness",
    "physical_registers": "not-modeled",
    "execution_temporaries": "not-modeled",
    "backend_allocation_reuse": "not-modeled",
    "scopes": [{
        "scope": "func@arithmetic", "status": "available",
        "logical_live_carrier_slots_per_thread": 3,
        "logical_live_carrier_bytes_per_thread": 12,
        "logical_live_shared_root_bytes": 0, "declared_shared_root_bytes": 0,
        "peak_slots_at": 1, "peak_bytes_at": 1, "peak_shared_at": 0,
        "saturated": False,
        # Actual final SSA: a,b live through mul; m through return; e through
        # div; d through return. Each selected layout has one f32 carrier.
        "tensor_values": [
            {"id": i, "dtype": "f32", "slots_per_thread": 1,
             "bytes_per_thread": 4, "begin": begin, "end": end,
             "live_in": i < 2}
            for i, (begin, end) in enumerate(((0, 1), (0, 1), (1, 4), (2, 3), (3, 4)))
        ],
        "shared_roots": [], "unavailable_operations": [],
    }],
}
expected = {"stage": "post-conversion-optimization", "arithmetic": arithmetic,
            "resources": resources}
for stats in (on, capped, replay):
    assert stats["actual_ir_diagnostics"] == expected
for stats in (off, on, capped):
    assert stats["cost_model_id"] == "sm90-layout-lex9-legacy-v2"
    assert stats["status"] == "committed"
    stats.pop("timings")
    stats.pop("actual_ir_diagnostics")
    for attempt in stats["attempts"]:
        attempt.pop("timings")
assert off == on == capped

traces = [read(f".{mode}.trace.json") for mode in ("off", "on", "cap")]
assert traces[0]["finalCost"] == traces[1]["finalCost"] == traces[2]["finalCost"]
for index, trace in enumerate(traces):
    assert trace["cost_model_id"] == "sm90-layout-lex9-legacy-v2"
    total = list(trace["omittedCost"]["cost"])
    saturated = trace["omittedCost"]["saturated"]
    limit = 2**64 - 1
    for event in trace["events"]:
        if event["event"] == "cost-contribution" and event["costScope"] == trace["costScope"]:
            values = [a + b for a, b in zip(total, event["estimate"]["cost"])]
            saturated |= event["estimate"]["saturated"] or any(v > limit for v in values)
            total = [min(v, limit) for v in values]
    assert {"cost": total, "saturated": saturated} == trace["finalCost"]
    diagnostics = [e for e in trace["events"] if e["event"] == "actual-ir-diagnostics"]
    assert len(diagnostics) == (1 if index == 1 else 0)
    if diagnostics:
        assert diagnostics[0]["diagnostics"] == expected
assert traces[2]["truncated"] and traces[2]["eventsOmitted"] > 0
assert pathlib.Path(prefix + ".cap.trace.json").stat().st_size <= 4096
