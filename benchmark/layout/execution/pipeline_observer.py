"""Process-tree wall/RSS evidence for complete server-side pipeline phases.

This observer never loads CUDA or compiler libraries. Child commands are argv,
not shell text. Every failed child remains a completed retained phase. Peak tree
RSS is a sampled sum and can double-count shared pages; it is not unique memory.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ENVIRONMENT=('CUDA_VISIBLE_DEVICES','CUDA_MODULE_LOADING','TRITON_CACHE_DIR',
             'TRITON_ALWAYS_COMPILE','LD_LIBRARY_PATH','PYTHONDONTWRITEBYTECODE')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def observe(directory,argv,*,affinity=None,cwd=None,interval=.25,descendant_grace=60.):
    d=Path(directory).resolve();d.mkdir(parents=True,exist_ok=False)
    if not argv or any(not isinstance(x,str) for x in argv):raise ValueError('nonempty explicit argv required')
    cores=list(affinity) if affinity is not None else sorted(os.sched_getaffinity(0))
    if not cores or any(type(c) is not int or c<0 for c in cores):raise ValueError('CPU affinity must be explicit nonnegative cores')
    before=dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),argv=list(argv),cwd=str(Path(cwd or os.getcwd()).resolve()),
        affinity=cores,environment={n:os.environ.get(n) for n in ENVIRONMENT},interval_seconds=interval,
        descendant_grace_seconds=descendant_grace)
    _write(d/'command.json',before)
    peak=0;peak_individual=0;samples=0;violations=[];error=None;pid=None;parent_code=None;remaining=[];tracked={}
    def bind():os.sched_setaffinity(0,set(cores))
    start=time.perf_counter_ns();tail_start=None;process=None;all_reaped=False;adopted_exits=[]
    libc=ctypes.CDLL(None,use_errno=True);prior_subreaper=ctypes.c_int();enabled=False
    if psutil.Process().children():error='observer requires an isolated process without existing children'
    elif libc.prctl(37,ctypes.byref(prior_subreaper),0,0,0)!=0:error='PR_GET_CHILD_SUBREAPER failed'
    elif libc.prctl(36,1,0,0,0)!=0:error='PR_SET_CHILD_SUBREAPER failed'
    else:enabled=True
    with (d/'stdout.log').open('w') as out,(d/'stderr.log').open('w') as err,(d/'resources.jsonl').open('w') as raw:
        try:
            if error is None:process=subprocess.Popen(argv,cwd=before['cwd'],stdout=out,stderr=err,env=dict(os.environ),preexec_fn=bind,start_new_session=True)
        except (OSError,subprocess.SubprocessError) as exc:error=type(exc).__name__+': '+str(exc)
        if process is not None:
            pid=process.pid
            while True:
                stamp=time.monotonic_ns();family={};rows=[]
                # A separate owned process group survives parent reaping, so an
                # immediately orphaned compiler child does not escape accounting.
                for proc in psutil.process_iter(['pid','create_time']):
                    try:
                        if os.getpgid(proc.pid)==pid:family[proc.pid]=proc
                    except (ProcessLookupError,PermissionError,psutil.NoSuchProcess):pass
                for seen_pid,created in list(tracked.items()):
                    try:
                        proc=psutil.Process(seen_pid)
                        if proc.create_time()==created:family[seen_pid]=proc
                    except psutil.NoSuchProcess:pass
                for proc in list(family.values()):
                    try:
                        for child in proc.children(recursive=True):family[child.pid]=child
                    except psutil.NoSuchProcess:pass
                for proc in family.values():
                    try:
                        state=proc.status()
                        if state in (psutil.STATUS_ZOMBIE,psutil.STATUS_DEAD):continue
                        info=dict(pid=proc.pid,created=proc.create_time(),rss=proc.memory_info().rss,
                            cpu_affinity=proc.cpu_affinity(),status=state)
                        tracked[proc.pid]=info['created']
                        if not set(info['cpu_affinity'])<=set(cores):violations.append(dict(sample=samples,**info))
                        rows.append(info)
                    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                rss=sum(x['rss'] for x in rows);peak=max(peak,rss)
                peak_individual=max([peak_individual]+[x['rss'] for x in rows])
                parent_code=process.poll()
                remaining=[r for r in rows if r['pid']!=pid]
                raw.write(json.dumps(dict(sample=samples,monotonic_ns=stamp,rss_tree_sum=rss,
                    parent_exit_code=parent_code,processes=rows))+'\n');raw.flush();samples+=1
                if parent_code is not None:
                    # Kernel adoption closes the snapshot/fork/parent-exit race:
                    # ECHILD proves no unreaped descendants remain in this
                    # isolated observer, including double-fork/setsid children.
                    while True:
                        try:adopted,status=os.waitpid(-1,os.WNOHANG)
                        except ChildProcessError:all_reaped=True;break
                        if adopted==0:break
                        adopted_exits.append(dict(pid=adopted,exit_code=os.waitstatus_to_exitcode(status)))
                    if all_reaped:remaining=[];break
                    if tail_start is None:tail_start=time.monotonic()
                    if time.monotonic()-tail_start>descendant_grace:
                        error='owned descendants remain after parent exit; phase incomplete';break
                time.sleep(interval)
            parent_code=process.wait()
    end=time.perf_counter_ns();scope_complete=process is not None and all_reaped
    if enabled and (all_reaped or process is None):libc.prctl(36,prior_subreaper.value,0,0,0)
    if error is None and any(x['exit_code']!=0 for x in adopted_exits):error='adopted descendant exited unsuccessfully'
    code=127 if process is None else parent_code if error is None else 94
    result=dict(status='complete' if code==0 and scope_complete else 'failed',exit_code=code,parent_exit_code=parent_code,
        spawn_error=error if process is None else None,error=error,pid=pid,owned_process_group=pid,
        scope_complete=scope_complete,remaining_descendants=remaining,adopted_descendant_exits=adopted_exits,
        descendant_accounting='Linux child-subreaper; parent reaped before waitpid ECHILD completion',
        wall_ns=end-start,started_unix_ns=before['wall_ns'],completed_unix_ns=time.time_ns(),
        samples=samples,sampled_peak_tree_rss_bytes=peak,sampled_peak_individual_rss_bytes=peak_individual,
        affinity_violations=violations,child_directly_reaped=process is not None,
        memory_semantics='sampled RSS sum of owned process group plus observed descendants; shared pages may be double-counted; processes escaping both ancestry and group between samples may be missed',
        files={n:dict(bytes=(d/n).stat().st_size,sha256=_sha(d/n)) for n in ('command.json','stdout.log','stderr.log','resources.jsonl')})
    _write(d/'completion.json',result)
    return result


def summarize(rows):
    import numpy as np
    if not rows:raise ValueError('observed phases required')
    times=np.asarray([r['wall_ns']/1e9 for r in rows])
    return dict(phases=len(rows),failures=sum(r['status']!='complete' for r in rows),
        sum_phase_seconds=float(times.sum()),p50_phase_seconds=float(np.quantile(times,.5)),
        p95_phase_seconds=float(np.quantile(times,.95)),
        max_sampled_tree_rss_bytes=max(r['sampled_peak_tree_rss_bytes'] for r in rows),
        interpretation='descriptive distribution of these heterogeneous phases, not repeated-run latency CI')


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--affinity',default='8,9,10,11');p.add_argument('--cwd',type=Path)
    p.add_argument('argv',nargs=argparse.REMAINDER);a=p.parse_args();argv=a.argv[1:] if a.argv[:1]==['--'] else a.argv
    result=observe(a.directory,argv,affinity=[int(x) for x in a.affinity.split(',')],cwd=a.cwd)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','affinity_violations')}),flush=True)
    raise SystemExit(result['exit_code'])

if __name__=='__main__':main()
