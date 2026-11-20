"""Explicit optional four-process native batch; default serial API is untouched.

Standard-library/psutil supervisor; no CUDA/torch import in the parent. Every
request still uses CompilerSession.compile. This wrapper performs no GPU timing.
"""
from __future__ import annotations
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback
import psutil

CONTRACT='fixed-ttgir-four-worker-batch-v1'
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
INPUTS=('execution.json','emitted.ttgir','emitted-config.json')
CONTEXT_KEYS={'source','tools','target','options','environment'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def safe(path,*,directory=False):
    p=Path(path).absolute()
    if any(q.is_symlink() for q in (p,*p.parents)):raise ValueError('symlink batch path rejected')
    if directory and not p.is_dir():raise ValueError('existing request directory required')
    return p


def request(directory,index):
    p=safe(directory,directory=True)
    files={n:dict(bytes=(p/n).stat().st_size,sha256=sha(safe(p/n))) for n in INPUTS}
    return dict(index=index,path=str(p),files=files,ttgir_sha256=files['emitted.ttgir']['sha256'])


def check_plan(plan):
    if plan.get('contract')!=CONTRACT or plan.get('workers')!=4:raise ValueError('fixed four-worker contract required')
    cpus=plan.get('cpus');rows=plan.get('requests')
    if not isinstance(cpus,list) or len(cpus)!=4 or len(set(cpus))!=4 or any(type(c) is not int or c<0 for c in cpus):raise ValueError('four distinct CPU cores required')
    if not isinstance(rows,list) or not rows or len(rows)!=plan.get('request_cap'):raise ValueError('request cap/coverage differs')
    if [r.get('index') for r in rows]!=list(range(len(rows))) or len({r.get('path') for r in rows})!=len(rows):raise ValueError('duplicate or reordered request identity')
    groups={}
    for row in rows:
        h=row.get('ttgir_sha256')
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('invalid exact TTGIR identity')
        if set(row.get('files',{}))!=set(INPUTS) or row['files']['emitted.ttgir']['sha256']!=h:raise ValueError('request emission identity differs')
        groups.setdefault(h,[]).append(row['index'])
    expected=[dict(ttgir_sha256=h,worker=int(h,16)%4,request_indices=groups[h]) for h in sorted(groups)]
    if plan.get('groups')!=expected:raise ValueError('group duplicated, split, or routed inconsistently')
    if type(plan.get('compiler_input_reuse')) is not bool or set(plan.get('context',{}))-CONTEXT_KEYS:raise ValueError('compiler context/options differ')
    return expected


def make_plan(directories,cache_root,output,cpus,execution_modules,*,context=None,compiler_input_reuse=False):
    rows=[request(p,i) for i,p in enumerate(directories)]
    groups={}
    for row in rows:groups.setdefault(row['ttgir_sha256'],[]).append(row['index'])
    modules=safe(execution_modules,directory=True)
    bindings={str(p):sha(p) for p in (Path(__file__).resolve(),modules/'native_compile_cache.py',modules/'pipeline_observer.py')}
    plan=dict(contract=CONTRACT,workers=4,cpus=list(cpus),request_cap=len(rows),requests=rows,
        groups=[dict(ttgir_sha256=h,worker=int(h,16)%4,request_indices=groups[h]) for h in sorted(groups)],
        cache_root=str(safe(cache_root)),output=str(safe(output)),execution_modules=str(modules),
        compiler_input_reuse=compiler_input_reuse,context=context or {},bindings=bindings,
        thread_environment={n:'1' for n in THREAD_ENV},routing='int(exact generated TTGIR SHA256,16) modulo 4; stable across warm runs')
    check_plan(plan);return plan


def check_inputs(row):
    for name,item in row['files'].items():
        p=safe(Path(row['path'])/name)
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('frozen generation input changed: '+str(p))


def check_bindings(plan):
    for path,h in plan['bindings'].items():
        if sha(safe(path))!=h:raise ValueError('batch implementation changed: '+path)


def worker(plan_path,expected_sha,index):
    if sha(plan_path)!=expected_sha:raise ValueError('worker plan digest differs')
    plan=json.loads(Path(plan_path).read_text());groups=check_plan(plan);check_bindings(plan)
    os.sched_setaffinity(0,{plan['cpus'][index]})
    if any(os.environ.get(n)!='1' for n in THREAD_ENV):raise ValueError('worker thread environment differs')
    own={i for g in groups if g['worker']==index for i in g['request_indices']}
    rows=[r for r in plan['requests'] if r['index'] in own]
    out=Path(plan['output'])/('worker-'+str(index));result=dict(contract=CONTRACT,worker=index,
        plan_sha256=expected_sha,status='running',planned_requests=len(rows),attempts=[],session=None,
        affinity=sorted(os.sched_getaffinity(0)),thread_environment={n:os.environ[n] for n in THREAD_ENV})
    started=time.perf_counter_ns();session=None;error=None
    prior_signals={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    def stop_worker(signum,frame):raise InterruptedError('worker interrupted by signal '+str(signum))
    for sig in prior_signals:signal.signal(sig,stop_worker)
    try:
        if rows:
            # CUDA/compiler libraries are first imported in this fresh child.
            import torch
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            result['torch_threads']=dict(intraop=torch.get_num_threads(),interop=torch.get_num_interop_threads())
            sys.path.insert(0,plan['execution_modules'])
            from native_compile_cache import CompilerSession
            kwargs=dict(plan['context'])
            if plan['compiler_input_reuse']:
                if 'compiler_input_reuse' not in inspect.signature(CompilerSession).parameters:raise ValueError('requested compiler-input reuse is not supported by this source')
                kwargs['compiler_input_reuse']=True
            session=CompilerSession(Path(plan['cache_root'])/('worker-'+str(index)),**kwargs)
            for row in rows:
                begin=time.perf_counter_ns();attempt=dict(index=row['index'],path=row['path'],ttgir_sha256=row['ttgir_sha256'])
                try:
                    check_bindings(plan);check_inputs(row)
                    compiled=session.compile(row['path']);check_inputs(row)
                    attempt.update(status='complete',compile_result=compiled)
                except Exception as exc:
                    attempt.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
                attempt['wall_ns']=time.perf_counter_ns()-begin;result['attempts'].append(attempt)
                result['session']=session.summary();write(out/'worker-result.json',result)
        result['status']='complete' if all(a['status']=='complete' for a in result['attempts']) else 'failed'
    except Exception as exc:error=type(exc).__name__+': '+str(exc);result.update(status='failed',error=error,traceback=traceback.format_exc())
    finally:
        if session is not None:
            try:session.close()
            except Exception as exc:result.update(status='failed',close_error=type(exc).__name__+': '+str(exc))
            result['session']=session.summary()
        result.update(wall_ns=time.perf_counter_ns()-started,attempted_requests=len(result['attempts']),
            not_attempted_requests=len(rows)-len(result['attempts']))
        write(out/'worker-result.json',result)
        for sig,handler in prior_signals.items():signal.signal(sig,handler)
    return 0 if result['status']=='complete' else 1


def run_plan(plan,*,worker_commands=None,interval=.025,timeout=None,cleanup_seconds=10.):
    """Run a predeclared plan. Alternate argv are explicit injected CPU tests."""
    check_plan(plan);check_bindings(plan)
    for row in plan['requests']:check_inputs(row)
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=False)
    write(output/'plan.json',plan);plan_sha=sha(output/'plan.json')
    available=os.sched_getaffinity(0)
    if not set(plan['cpus'])<=available:raise ValueError('declared worker CPU unavailable')
    if psutil.Process().children():raise ValueError('batch supervisor must be isolated without prior children')
    original_affinity=set(available)
    libc=ctypes.CDLL(None,use_errno=True);prior=ctypes.c_int()
    if libc.prctl(37,ctypes.byref(prior),0,0,0) or libc.prctl(36,1,0,0,0):raise OSError('Linux child-subreaper required')
    os.sched_setaffinity(0,set(plan['cpus']))
    started=time.perf_counter_ns();usage=resource.getrusage(resource.RUSAGE_CHILDREN);own_usage=resource.getrusage(resource.RUSAGE_SELF);children=[];handles=[];samples=0;peak=0;peak_threads=0;error=None;adopted=[];complete=False;all_reaped=False
    old_handlers={}
    def interrupted(signum,frame):raise InterruptedError('batch interrupted by signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM):old_handlers[sig]=signal.signal(sig,interrupted)
    def family():
        result=[]
        for p in psutil.Process().children(recursive=True):
            try:
                if p.status() not in (psutil.STATUS_DEAD,psutil.STATUS_ZOMBIE):result.append(p)
            except psutil.NoSuchProcess:pass
        return result
    def terminate_owned():
        # Scope is actual descendants of this isolated subreaper, never names.
        end=time.monotonic()+cleanup_seconds
        for sig in (signal.SIGTERM,signal.SIGKILL):
            for proc in reversed(family()):
                try:proc.send_signal(sig)
                except psutil.NoSuchProcess:pass
            until=end if sig==signal.SIGTERM else time.monotonic()+cleanup_seconds
            while time.monotonic()<until and family():time.sleep(.01)
    try:
        env=dict(os.environ,**plan['thread_environment'],PYTHONDONTWRITEBYTECODE='1')
        observer=Path(plan['execution_modules'])/'pipeline_observer.py'
        for index in range(4):
            wd=output/('worker-'+str(index));wd.mkdir()
            argv=([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(output/'plan.json'),plan_sha,str(index)]
                  if worker_commands is None else list(worker_commands[index]))
            command=[sys.executable,'-B',str(observer),'--affinity',str(plan['cpus'][index]),str(wd/'observation'),'--',*argv]
            log=(wd/'observer-console.log').open('w');handles.append(log)
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            children.append(process)
        with (output/'resources.jsonl').open('w') as raw:
            while True:
                rows=[]
                for proc in [psutil.Process(),*family()]:
                    try:rows.append(dict(pid=proc.pid,rss=proc.memory_info().rss,threads=proc.num_threads(),affinity=proc.cpu_affinity()))
                    except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
                rss=sum(r['rss'] for r in rows);peak=max(peak,rss);peak_threads=max(peak_threads,sum(r['threads'] for r in rows));samples+=1
                raw.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),rss_tree_sum=rss,processes=rows))+'\n');raw.flush()
                codes=[p.poll() for p in children]
                if any(c not in (None,0) for c in codes):raise RuntimeError('worker observer failed: '+str(codes))
                if all(c is not None for c in codes):break
                if timeout is not None and (time.perf_counter_ns()-started)/1e9>timeout:raise TimeoutError('batch timeout')
                time.sleep(interval)
        for p in children:p.wait()
        complete=True
    except BaseException as exc:error=type(exc).__name__+': '+str(exc)
    finally:
        if error is not None:terminate_owned()
        for p in children:
            try:p.wait(timeout=cleanup_seconds)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        deadline=time.monotonic()+cleanup_seconds
        while True:
            try:pid,status=os.waitpid(-1,os.WNOHANG)
            except ChildProcessError:all_reaped=True;break
            if pid:adopted.append(dict(pid=pid,exit_code=os.waitstatus_to_exitcode(status)))
            elif time.monotonic()>deadline:
                terminate_owned();error=(error+'; ' if error else '')+'descendant cleanup deadline exceeded';complete=False;break
            else:time.sleep(.01)
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
        libc.prctl(36,prior.value,0,0,0)
        os.sched_setaffinity(0,original_affinity)
        for h in handles:h.close()
    ended=time.perf_counter_ns();end_usage=resource.getrusage(resource.RUSAGE_CHILDREN);end_own_usage=resource.getrusage(resource.RUSAGE_SELF)
    observations=[];workers=[]
    for index in range(4):
        wd=output/('worker-'+str(index));cp=wd/'observation/completion.json';wp=wd/'worker-result.json'
        observations.append(json.loads(cp.read_text()) if cp.exists() else dict(status='missing',scope_complete=False))
        workers.append(json.loads(wp.read_text()) if wp.exists() else dict(worker=index,status='missing',attempts=[],session=None))
    try:check_bindings(plan)
    except Exception as exc:error=(error+'; ' if error else '')+str(exc)
    if not all_reaped:error=(error+'; ' if error else '')+'descendant scope did not reach ECHILD'
    if any(x['exit_code']!=0 for x in adopted):error=(error+'; ' if error else '')+'adopted descendant failed'
    attempts=[a for w in workers for a in w['attempts']];ids=[a['index'] for a in attempts]
    if len(set(ids))!=len(ids) or any(i not in range(plan['request_cap']) for i in ids):error=(error+'; ' if error else '')+'duplicate/unowned request attempt'
    contexts={w['session']['context_sha256'] for w in workers if w.get('session')}
    if len(contexts)>1:error=(error+'; ' if error else '')+'worker compiler contexts differ'
    tests=worker_commands is not None
    ok=complete and error is None and all(o.get('scope_complete') and o.get('status')=='complete' and not o.get('affinity_violations') for o in observations)
    if not tests:ok=ok and len(attempts)==plan['request_cap'] and all(w['status']=='complete' for w in workers)
    summary=dict(contract=CONTRACT,status='complete' if ok else 'failed',provider_mode='injected-cpu-test' if tests else 'production',
        plan_sha256=plan_sha,bindings_before=plan['bindings'],
        bindings_after={p:sha(p) for p in plan['bindings']},group_count=len(plan['groups']),request_cap=plan['request_cap'],
        requests=len(attempts),attempted_requests=len(attempts),succeeded_requests=sum(a.get('status')=='complete' for a in attempts),
        failed_requests=sum(a.get('status')!='complete' for a in attempts),not_attempted_requests=plan['request_cap']-len(attempts),
        compiler_api_calls=sum(w['session']['compiler_api_calls'] for w in workers if w.get('session')),
        compiler_api_wall_ns=sum(w['session'].get('compiler_api_wall_ns',0) for w in workers if w.get('session')),
        sum_worker_session_initialization_ns=sum(w['session'].get('session_initialization_wall_ns',0) for w in workers if w.get('session')),
        exact_cache_hits=sum(w['session']['exact_cache_hits'] for w in workers if w.get('session')),
        compiler_input_hits=sum(w['session'].get('compiler_input_hits',0) for w in workers if w.get('session')),
        affinity=plan['cpus'],worker_cache_roots=[str(Path(plan['cache_root'])/('worker-'+str(i))) for i in range(4)],
        actual_wall_ns=ended-started,supervisor_user_cpu_seconds=end_own_usage.ru_utime-own_usage.ru_utime,
        supervisor_system_cpu_seconds=end_own_usage.ru_stime-own_usage.ru_stime,children_user_cpu_seconds=end_usage.ru_utime-usage.ru_utime,
        children_system_cpu_seconds=end_usage.ru_stime-usage.ru_stime,
        sampled_peak_tree_rss_bytes=peak,samples=samples,sampled_peak_tree_threads=peak_threads,
        memory_semantics='sampled sum includes supervisor/observers/workers/descendants; shared pages may be double counted; not unique RSS',
        error=error,descendants_reaped=all_reaped,adopted_exits=adopted,worker_exit_codes=[p.returncode for p in children],
        group_routing='each exact generated TTGIR group has one fixed worker; every original request is still compiled/checked',
        observations=observations,workers=workers)
    values=sorted(a['wall_ns'] for a in attempts if type(a.get('wall_ns')) is int)
    def quantile(q):
        if not values:return None
        x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1)
        return values[i]+(values[j]-values[i])*(x-i)
    summary['attempt_wall_distribution']=dict(count=len(values),p50_ns=quantile(.5),p95_ns=quantile(.95),
        maximum_ns=max(values) if values else None,scope='all individually recorded attempted requests, including failures; heterogeneous descriptive distribution, not repeated-run CI')
    summary['total_cpu_seconds']=summary['supervisor_user_cpu_seconds']+summary['supervisor_system_cpu_seconds']+summary['children_user_cpu_seconds']+summary['children_system_cpu_seconds']
    write(output/'summary.json',summary);return summary



def compile_many_parallel(directories,cache_dir,*,output,execution_modules=None,
                          cpus=(8,9,10,11),compiler_input_reuse=False,timeout=None,**context):
    """Explicit opt-in companion; serial native_compile_cache.compile_many stays unchanged."""
    plan=make_plan(directories,cache_dir,output,cpus,execution_modules or Path(__file__).parent,
                   context=context,compiler_input_reuse=compiler_input_reuse)
    return run_plan(plan,timeout=timeout)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--worker':return worker(Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]))
    parser=argparse.ArgumentParser(description='Explicit optional parallel compilation; existing serial pipeline unchanged')
    parser.add_argument('--directories',type=Path,required=True,help='JSON list of original request directories, no extra candidates')
    parser.add_argument('--cache-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execution-modules',type=Path,required=True);parser.add_argument('--cpus',default='8,9,10,11')
    parser.add_argument('--context',type=Path);parser.add_argument('--compiler-input-reuse',action='store_true');parser.add_argument('--timeout',type=float)
    a=parser.parse_args();context=json.loads(a.context.read_text()) if a.context else None
    plan=make_plan(json.loads(a.directories.read_text()),a.cache_root,a.output,[int(c) for c in a.cpus.split(',')],a.execution_modules,context=context,compiler_input_reuse=a.compiler_input_reuse)
    result=run_plan(plan,timeout=a.timeout);return 0 if result['status']=='complete' else 1

if __name__=='__main__':raise SystemExit(main())
