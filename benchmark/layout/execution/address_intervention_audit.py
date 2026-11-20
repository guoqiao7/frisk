"""Timing-free complete-source/native intervention groups on checked scalar rows."""
from __future__ import annotations
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import address_intervention as proof


def _one(task):
    case_root,locality_row,output=task
    locpath=Path(locality_row['summary_path']);raw=locpath.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=locality_row['summary_file_sha256']:raise ValueError('locality index/file mismatch')
    locality=json.loads(raw)
    if locality['summary_sha256']!=locality_row['summary_sha256']:raise ValueError('locality index/payload mismatch')
    root=Path(case_root)/locality_row['case'];result=proof.summarize(root,locality)
    text=json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n';target=Path(output)/(root.name+'.json');target.write_text(text)
    return dict(case=root.name,path=str(target),file_sha256=hashlib.sha256(text.encode()).hexdigest(),summary_sha256=result['summary_sha256'],status=result['status'],reasons=result.get('reasons',[]),controlled_group_sha256=result.get('controlled_group_sha256'),cubin_sha256=result['provenance'].get('dependencies',{}).get('kernel.cubin'),source_nonaddress_sha256=result.get('source',{}).get('nonaddress_sha256'),native_skeleton_sha256=result.get('native',{}).get('skeleton_sha256'))


def audit(case_root,locality_index,output,*,workers=4):
    root=Path(case_root).resolve(strict=True);out=Path(output).resolve()
    if 'network_selection_execution_artifacts' in out.parts or 'baopeihua-frisk-network-20261105' in out.parts or root in out.parents:raise ValueError('new output outside sealed inputs required')
    if out.exists():raise FileExistsError('fresh intervention output required')
    out.mkdir(parents=True);index=json.loads(Path(locality_index).read_text());selected=[r for r in index['rows'] if r['status']=='complete']
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:rows=list(executor.map(_one,[(str(root),row,str(out)) for row in selected]))
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='available':groups[row['controlled_group_sha256']].append(row)
    grouped=[]
    for key,members in sorted(groups.items()):
        cubins={r['cubin_sha256'] for r in members}
        grouped.append(dict(controlled_group_sha256=key,members=[r['case'] for r in members],candidate_occurrences=len(members),distinct_cubins=len(cubins),nontrivial_address_intervention=len(cubins)>1))
    result=dict(contract='frisk-controlled-source-address-cohort-v1',role='known-development-not-fresh-no-timing-labels-read',analyzer_sha256=proof.SOURCE_SHA256,locality_index=dict(path=str(Path(locality_index).resolve()),sha256=hashlib.sha256(Path(locality_index).read_bytes()).hexdigest()),cases=len(rows),status=dict(Counter(r['status'] for r in rows)),group_count=len(grouped),nontrivial_groups=sum(r['nontrivial_address_intervention'] for r in grouped),nontrivial_occurrences=sum(r['candidate_occurrences'] for r in grouped if r['nontrivial_address_intervention']),groups=grouped,rows=rows,elapsed_seconds=time.monotonic()-started,limitations=['controlled source address DU and compiled static equivalence, not native DU','per-thread input values, dynamic masks and slowpath counts need not coincide','no cache/TLB miss, physical page size or hardware latency constant is inferred'])
    (out/'groups.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');return result
