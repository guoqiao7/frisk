"""Evidence-only classification and Markdown for selection-quality experiments."""
import json
import statistics

# The exact order in LayoutCost::operator<; no weighted scalar exists.
COST_FIELDS = ("instructionPathAndWork", "memoryTransactions", "bankConflictDegree",
               "conversionBytesAndSync", "spillRiskAndRegisters", "sharedBytesAndOccupancy",
               "replication", "codeSize", "deterministicTieBreak")
FAILED_STATUSES = {"infrastructure-error", "fatal-correctness", "process-timeout", "process-launch-failure",
                   "timing-instability", "preflight-blocked"}


def cost_difference(left, right):
    if len(left) != 9 or len(right) != 9:
        raise ValueError("expected nine lexicographic fields")
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "field": COST_FIELDS[index], "left": a, "right": b}
    return None


def complete(record):
    o = record.get("oracle") or {}
    return (record.get("status", "success") in ("success", "local-analysis-only")
            and o.get("status") == "complete" and o.get("exhausted") is True
            and not o.get("fatal") and o.get("geometry_set_kind") in ("all-optimal", "all-minimum-cost-geometries-independent-of-stable-producer-winner")
            and bool(o.get("optimal_geometries")) and o.get("best_cost") is not None)


def classify(local, joint, branches):
    """Compare geometry sets across scopes, but costs only in the full graph."""
    result = {"classification": "heuristic-only", "scope": "finite-proven-support-domain"}
    if not complete(local) or not complete(joint):
        result["reason"] = "incomplete local/full oracle; incumbent is not an optimum"
        return result
    gl = set(local["oracle"]["optimal_geometries"])
    gf = set(joint["oracle"]["optimal_geometries"])
    result.update(local_optimal_geometries=sorted(gl), full_optimal_geometries=sorted(gf))
    if not gl.issubset(branches):
        result.update(classification="not-comparable", reason="local optimum absent from full graph domain/branches")
        return result
    if any(not complete(branches[g]) for g in gl):
        result["reason"] = "incomplete restricted full oracle"
        return result
    if any(r.get("oracle_comparison") == "search-gap" for r in [local, joint] + [branches[g] for g in gl]):
        result["reason"] = "default search gap; raw solver and oracle retained separately"
        return result
    full = joint["oracle"]["best_cost"]["cost"]
    restricted = {g: branches[g]["oracle"]["best_cost"]["cost"] for g in sorted(gl)}
    result.update(joint_full_cost=full, local_geometry_full_costs=restricted,
                  first_differences={g: cost_difference(full, c) for g, c in restricted.items()})
    if gl.isdisjoint(gf) and all(full < c for c in restricted.values()):
        result["classification"] = "strict-geometry-reversal"
    elif len(gl) > 1 and gf < gl:
        result["classification"] = "local-tie-resolved"
    elif (len(joint.get("selected_geometries", [])) == 1
          and joint["selected_geometries"][0] in gl & gf
          and branches[joint["selected_geometries"][0]].get("selected_geometries") == joint["selected_geometries"]
          and any(joint.get(key) != branches[joint["selected_geometries"][0]].get(key)
                  for key in ("assignment", "conversions", "existing_conversions"))):
        result["classification"] = "same-geometry-layout-conversion-change"
        result["same_selected_geometry"] = joint["selected_geometries"][0]
        result["interpretation"] = "tied binding/plan change, not consumer cost benefit" if full == restricted[joint["selected_geometries"][0]] else "same-geometry binding/plan change"
    else:
        result["classification"] = "no-observed-improvement"
    return result


def native_comparison(native, controlled):
    result = {"status": "not-comparable"}
    if native.get("status") != "success" or controlled.get("status") != "success":
        return dict(result, reason="native/controlled did not both succeed")
    if not isinstance(controlled.get("spec"), dict) or controlled["spec"].get("consumer") != "store" or controlled.get("phase") != "controlled":
        return dict(result, reason="different native entry/scope")
    keys = ["source_digest", "config_digest", "binary_sha256", "input_sha256", "spec", "solver_options"]
    mismatches = [key for key in keys if not native.get(key) or native.get(key) != controlled.get(key)]
    a, b = native.get("native_snapshot_projection", {}), controlled.get("native_snapshot_projection", {})
    if not a.get("attempt") or not a.get("sha256") or a.get("sha256") != b.get("sha256"):
        mismatches.append("actual-final-attempt/prepared-projection")
    return {"status": "not-comparable" if mismatches else "same-domain", "mismatches": mismatches}


def is_formal(data):
    return data.get("mode") in ("correctness", "core") or (
        data.get("mode") == "merged" and any(m in ("correctness", "core") for m in data.get("component_modes", [])))


def summarize(rows):
    """Preserve manifest order and missing/fatal rows; never resize a matrix."""
    by_id = {r["manifest"]["id"]: r for r in rows}
    comparisons, classifications = [], []
    for row in rows:
        m, record = row["manifest"], row.get("result", {})
        if m.get("phase") != "controlled" or m.get("restrict_grid"):
            continue
        if m.get("consumer") == "store":
            native = by_id.get(f"{m['case']}-{m['mode']}-native-Auto", {}).get("result", {})
            comparisons.append({"id": m["id"], **native_comparison(native, record)})
        else:
            local = by_id.get(f"{m['case']}-{m['mode']}-local", {}).get("result", {})
            branches = {r["manifest"]["restrict_grid"]: r.get("result", {}) for r in rows
                        if r["manifest"].get("parent") == m["id"]}
            classifications.append({"id": m["id"], **classify(local, record, branches)})
    counts = {}
    for row in rows:
        key = row["status"]
        counts[key] = counts.get(key, 0) + 1
    native, consumers, witness, timing = {}, {}, {}, {}
    for r in rows:
        m, record = r["manifest"], r.get("result", {})
        if m.get("phase") == "native" and m.get("consumer") == "store" and "warmup" not in m:
            failure = record.get("failure", {}).get("kind") or r["status"]
            native[failure] = native.get(failure, 0) + 1
        if m.get("phase") == "controlled" and m.get("consumer") != "store" and not m.get("parent"):
            consumers[r["status"]] = consumers.get(r["status"], 0) + 1
        mode = (record.get("oracle") or {}).get("incumbent_validation")
        if mode:
            witness[mode] = witness.get(mode, 0) + 1
        if "warmup" in m:
            key = m["case"] + "-" + m["policy"]
            timing.setdefault(key, []).append(r)
    timing_summary = []
    for key, samples in timing.items():
        measured = [s for s in samples if not s["manifest"]["warmup"] and s["status"] == "success"]
        walls = [s["process"]["wall_seconds"] for s in measured]
        signatures = {(s.get("output_sha256"), s.get("non_timing_statistics_sha256")) for s in samples}
        timing_summary.append({"id": key, "warmups": sum(s["manifest"]["warmup"] for s in samples),
                               "measured_successes": len(measured), "stable": len(signatures) == 1 and all(s["status"] == "success" for s in samples),
                               "wall_seconds": walls, "median_wall_seconds": statistics.median(walls) if walls else None,
                               "peak_rss_kib": [s["process"].get("peak_rss_kib") for s in measured]})
    return {"status_counts": counts, "native_comparisons": comparisons, "q2_classifications": classifications,
            "native_q1_status_counts": native, "q2_graph_status_counts": consumers,
            "oracle_incumbent_validation_modes": witness, "timing": timing_summary}


def render(data):
    summary = data.get("summary", summarize(data.get("rows", [])))
    gate = data.get("core_gate", {})
    accepted = gate.get("passed") is True and not data.get("integrity_errors") and not any(r["status"] in FAILED_STATUSES for r in data.get("rows", []))
    acceptance = ("已验收 / accepted" if accepted else "未验收 / not accepted") if is_formal(data) else "诊断 / diagnostic only; not stage acceptance"
    lines = ["# Selection-quality evidence", "", f"Stage status: {acceptance}.",
             f"Missing/incomplete core rows: `{json.dumps(gate.get('missing_or_incomplete', []))}`.",
             f"Integrity errors: `{json.dumps(data.get('integrity_errors', []), sort_keys=True)}`.", "",
             "Scope: finite current Proven support domain; not hardware impossibility or GPU performance.",
             "Inputs are Shared/Tensor-already-prepared MMA subgraphs; Global→Shared input movement is excluded.",
             "Correctness wall times are diagnostic, not performance rankings. Costs are nine-field lexicographic vectors.", "",
             f"Rows: {len(data.get('rows', []))}; statuses: `{json.dumps(summary['status_counts'], sort_keys=True)}`.", "",
             "| Manifest row | Status / failure | Default cost | Oracle status / cost / geometries |", "| --- | --- | --- | --- |"]
    for row in data.get("rows", []):
        r = row.get("result", {})
        o = r.get("oracle") or {}
        cost = (r.get("cost") or {}).get("cost")
        failure = r.get("failure", {}).get("kind", row.get("error", ""))
        lines.append(f"| {row['manifest']['id']} | {row['status']} {failure} | `{cost}` | {o.get('status', 'not-requested')} / `{(o.get('best_cost') or {}).get('cost')}` / `{o.get('optimal_geometries', [])}` |")
    lines.extend(["", "## Same-domain checks", ""])
    lines.extend(f"- {c['id']}: {c['status']}; {c.get('mismatches', c.get('reason', ''))}" for c in summary["native_comparisons"])
    lines.extend(["", "## Consumer classifications", ""])
    for c in summary["q2_classifications"]:
        lines.append(f"- {c['id']}: {c['classification']}; `{json.dumps(c, sort_keys=True)}`")
    lines.extend(["", "## Native Q1 coverage", "",
                  f"Store/native only (entry rejection probes and timing repeats excluded): `{json.dumps(summary['native_q1_status_counts'], sort_keys=True)}`.",
                  f"Full Q2 joint graph coverage: `{json.dumps(summary['q2_graph_status_counts'], sort_keys=True)}`.",
                  "Different native policies may prepare different domains; policy cost comparisons do not isolate geometry-only effects.",
                  "", "## Oracle witness validation", "",
                  f"`{json.dumps(summary['oracle_incumbent_validation_modes'], sort_keys=True)}`.",
                  "Every published full-graph oracle incumbent must have actual IR/ABI/full-pipeline/parse-replay verification; local-only witnesses do not count as pipeline success.",
                  "", "## Decision example", ""])
    example = next((r for r in data.get("rows", []) if r.get("result", {}).get("phase") == "controlled" and complete(r["result"])), None)
    if example:
        r = example["result"]
        lines.extend([f"{example['manifest']['id']}: default geometry `{r.get('selected_geometries')}`, full default cost `{r['cost']['cost']}`; oracle `{r['oracle']['best_cost']['cost']}`, all tied geometries `{r['oracle']['optimal_geometries']}`.", "",
                      "| Contribution kind | Owner | Nine-field cost |", "| --- | --- | --- |"])
        for contribution in r["cost_contributions"]:
            estimate = contribution.get("cost", contribution.get("estimate"))
            lines.append(f"| {contribution['kind']} | {contribution['owner']} | `{estimate['cost']}` |")
        lines.extend(["", f"Selected conversion/rematerialization plan: `{json.dumps(r.get('conversions', []), sort_keys=True)}`.",
                      f"Existing explicit conversion boundaries: `{json.dumps(r.get('existing_conversions', []), sort_keys=True)}`."])
    else:
        lines.append("No complete controlled decision example in this dataset; no optimality claim.")
    lines.extend(["", "## Whole-CLI timing", "",
                  "Serial Release frisk-opt; parse + pipeline + IR output + required statistics. No oracle, trace or profile. One warmup and five measured calls per configuration.",
                  "Parent wall observation uses a 5 ms wait4 poll; very small differences are not significant. Per-child Linux peak RSS is KiB. No CPU isolation/randomization claim.", "",
                  "| Configuration | Measured successes | Stable IR/plan/stats | Median wall (s) | Peak RSS KiB (five raw values) |",
                  "| --- | --- | --- | --- | --- |"])
    for t in summary["timing"]:
        lines.append(f"| {t['id']} | {t['measured_successes']} | {t['stable']} | {t['median_wall_seconds']} | `{t['peak_rss_kib']}` |")
    lines.extend(["", "Complete oracle means exhaustive within the finite prepared Proven support tables only. Infeasible does not mean hardware-impossible.",
                  "Unknown/proof/search/time limits cannot establish optimality. Default solver results are never replaced by oracle results.", ""])
    return "\n".join(lines)
