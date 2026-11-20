"""Read-only known-development coverage audit; write only to a new output root."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import execution_cost_summary as summary


def audit(cases_dir, output_dir, *, workers=4):
    cases, output = Path(cases_dir).resolve(strict=True), Path(output_dir).resolve()
    if output == cases or cases in output.parents or "network_selection_execution_artifacts" in output.parts or "baopeihua-frisk-network-20261105" in output.parts:
        raise ValueError("output must be outside the sealed input tree")
    if output.exists() and any(output.iterdir()): raise ValueError("audit output already exists and is nonempty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); paths = sorted(p for p in cases.iterdir() if (p / "kernel.cubin").is_file())
    if not paths: raise ValueError("no cases")
    def one(path):
        result = summary.summarize(path)
        text = json.dumps(result, sort_keys=True, indent=2) + "\n"
        target = output / (path.name + ".json"); target.write_text(text)
        old_path = path / "native-analysis.json"; old = json.loads(old_path.read_text())
        return dict(case=path.name, summary_path=str(target), summary_file_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    native14_status=old.get("status"), native14_analysis_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                    status=result["status"], cost_family=result["cost_family"], reasons=result.get("reasons", []),
                    native_instructions=(result.get("features") or {}).get("native_instructions"),
                    execution_upper_unknown=result.get("execution_bounds", {}).get("upper") is None if result.get("execution_bounds") else True,
                    summary_sha256=result.get("summary_sha256"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(one, paths))
    old_unsupported = [r for r in rows if r["native14_status"] != "available"]
    result = dict(contract="frisk-execution-cost-summary-known-development-coverage-v1", input_cases=str(cases), role="known-development-regression-not-a-fresh-holdout",
                  cases=len(rows), old_native14=dict(Counter(r["native14_status"] for r in rows)),
                  summary_status=dict(Counter(r["status"] for r in rows)), cost_families=dict(Counter(r["cost_family"] or "unknown" for r in rows)),
                  old_unsupported=len(old_unsupported), old_unsupported_with_complete_static_summary=sum(r["status"] == "complete" for r in old_unsupported),
                  remaining_summary_unknown=sum(r["status"] != "complete" for r in old_unsupported),
                  unbounded_dynamic_execution=sum(r["execution_upper_unknown"] for r in rows),
                  rows=rows, analyzer_sha256=summary.SOURCE_SHA256,
                  semantics="complete means static reachable closure, not exact dynamic work or latency; old native14 statuses remain unchanged; no GPU, native compilation, timing-file read, or sealed-input write", elapsed_seconds=time.monotonic()-started)
    (output / "coverage.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases_dir"); parser.add_argument("output_dir"); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = audit(args.cases_dir, args.output_dir, workers=args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, sort_keys=True, indent=2))
