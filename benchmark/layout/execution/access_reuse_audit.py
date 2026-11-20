"""Known-development audit for checked GF2 address-window locality evidence."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import access_reuse as reuse


def audit(cases_dir, output_dir, *, workers=4):
    cases, output = Path(cases_dir).resolve(strict=True), Path(output_dir).resolve()
    if output == cases or cases in output.parents or "network_selection_execution_artifacts" in output.parts or "baopeihua-frisk-network-20261105" in output.parts: raise ValueError("output must be outside sealed inputs")
    if output.exists() and any(output.iterdir()): raise ValueError("output exists and is nonempty")
    output.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    paths=sorted(p for p in cases.iterdir() if (p/"kernel.cubin").is_file())
    def one(path):
        result=reuse.summarize(path); text=json.dumps(result,sort_keys=True,indent=2)+"\n"; dest=output/(path.name+".json");dest.write_text(text)
        methods=Counter(); projected=0
        for access in result.get("accesses",[]):
            for values in access["counts"].values():methods.update(values["methods"]);projected+=values["projected_tags_enumerated"]
        return dict(case=path.name,summary_path=str(dest),summary_file_sha256=hashlib.sha256(text.encode()).hexdigest(),summary_sha256=result["summary_sha256"],status=result["status"],reasons=result.get("reasons",[]),cost_family=result["cost_family"],methods=dict(methods),projected_tags_enumerated=projected)
    with ThreadPoolExecutor(max_workers=workers) as executor:rows=list(executor.map(one,paths))
    counts=Counter(row["status"] for row in rows);methods=Counter()
    for row in rows:methods.update(row["methods"])
    result=dict(contract="frisk-gf2-access-reuse-known-development-coverage-v1",role="known-development-regression-not-fresh-holdout",cases=len(rows),status=dict(counts),methods=dict(methods),projected_tags_enumerated=sum(r["projected_tags_enumerated"] for r in rows),analyzer_sha256=reuse.SOURCE_SHA256,feature_names=list(reuse.FEATURE_NAMES),footprint_features=list(reuse.FOOTPRINT_FEATURES),rows=rows,elapsed_seconds=time.monotonic()-start,semantics="relative-base-zero source/emitter query-window locality potential; full TTGIR+cfg exact match and surviving query coverage; not native order or hardware cache/TLB misses; unsupported is never a zero feature vector")
    (output/"coverage.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("cases_dir");parser.add_argument("output_dir");parser.add_argument("--workers",type=int,default=4);args=parser.parse_args()
    result=audit(args.cases_dir,args.output_dir,workers=args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows","feature_names","footprint_features"}},sort_keys=True,indent=2))
