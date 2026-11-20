import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import psutil
import parallel_compile_batch as batch

ROOT=Path(__file__).resolve().parent
if not (ROOT/'native_compile_cache.py').is_file():
 ROOT=next(p/'benchmark/layout/execution' for p in Path(__file__).resolve().parents if (p/'benchmark/layout/execution/native_compile_cache.py').is_file())
FAKE=r"""import hashlib,json,os,sys,time
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text());idx=int(sys.argv[2]);mode=sys.argv[3]
out=Path(plan['output'])/('worker-'+str(idx))
if mode=='escape':os.sched_setaffinity(0,set(plan['cpus']));time.sleep(2)
if mode in ('orphan','wait','fail'):
 pid=os.fork()
 if pid==0:
  os.setsid();(out/'orphan.pid').write_text(str(os.getpid()))
  time.sleep(.20 if mode=='orphan' else 10)
  (out/'orphan-finished').write_text('done');os._exit(0)
 if mode=='wait':time.sleep(10)
 if mode=='fail':raise SystemExit(7)
indices={i for g in plan['groups'] if g['worker']==idx for i in g['request_indices']}
rows=[r for r in plan['requests'] if r['index'] in indices]
groups={r['ttgir_sha256'] for r in rows}
result=dict(status='complete',worker=idx,attempts=[dict(index=r['index'],path=r['path'],status='complete',wall_ns=1000+r['index']) for r in rows],
 session=dict(context_sha256='a'*64,compiler_api_calls=len(groups),exact_cache_hits=0,compiler_input_hits=len(rows)-len(groups)))
(out/'worker-result.json').write_text(json.dumps(result))
"""
class ParallelBatchTest(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.cpus=sorted(os.sched_getaffinity(0))[:4]
  if len(self.cpus)!=4:self.skipTest('four cores required')
  self.requests=[]
  for i,text in enumerate(('a','a','b','c','d','e','a')):
   d=self.root/('r'+str(i));d.mkdir();(d/'execution.json').write_text(json.dumps({'request':i}))
   (d/'emitted.ttgir').write_text(text);(d/'emitted-config.json').write_text(json.dumps({'request':i}))
   self.requests.append(d)
 def tearDown(self):self.temp.cleanup()
 def plan(self,name='out'):
  return batch.make_plan(self.requests,self.root/'cache',self.root/name,self.cpus,ROOT,compiler_input_reuse=True)
 def commands(self,p,mode='normal'):
  path=self.root/'input-plan.json';path.write_text(json.dumps(p));script=self.root/'fake-worker.py';script.write_text(FAKE)
  return [[sys.executable,'-B',str(script),str(path),str(i),mode if i==0 else 'normal'] for i in range(4)]
 def assert_closed(self,result):
  self.assertTrue(result['descendants_reaped']);self.assertFalse(psutil.Process().children())
 def test_fixed_group_owner_route_and_cap(self):
  p=self.plan();batch.check_plan(p)
  self.assertEqual(len(p['groups']),5)
  owner={i:g['worker'] for g in p['groups'] for i in g['request_indices']}
  self.assertEqual(owner[0],owner[1]);self.assertEqual(owner[1],owner[6])
  warm=self.plan('warm');self.assertEqual(p['groups'],warm['groups'])
  for attack in ('split','duplicate','cap','boolcpu'):
   q=copy.deepcopy(p)
   if attack=='split':q['groups'][0]['worker']=(q['groups'][0]['worker']+1)%4
   if attack=='duplicate':q['requests'][1]['path']=q['requests'][0]['path']
   if attack=='cap':q['request_cap']+=1
   if attack=='boolcpu':q['cpus'][0]=True
   with self.subTest(attack=attack),self.assertRaises(ValueError):batch.check_plan(q)
 def test_generation_mutation_fails_before_launch(self):
  p=self.plan();self.requests[1].joinpath('emitted-config.json').write_text('{}')
  with self.assertRaises(ValueError):batch.run_plan(p,worker_commands=self.commands(p))
  self.assertFalse(Path(p['output']).exists())
 def test_budget_conservation_context_and_no_duplicate_group_compile(self):
  p=self.plan();r=batch.run_plan(p,worker_commands=self.commands(p),interval=.01)
  self.assertEqual(r['status'],'complete');self.assertEqual(r['provider_mode'],'injected-cpu-test')
  self.assertEqual(r['attempted_requests'],7);self.assertEqual(r['succeeded_requests'],7);self.assertEqual(r['not_attempted_requests'],0)
  self.assertEqual(r['compiler_api_calls'],5);self.assertEqual(r['compiler_input_hits'],2)
  self.assertEqual(sum(w['session']['compiler_api_calls'] for w in r['workers']),len(p['groups']))
  self.assertGreater(r['actual_wall_ns'],0);self.assertGreater(r['sampled_peak_tree_rss_bytes'],0)
  self.assertEqual(r['attempt_wall_distribution']['count'],7);self.assert_closed(r)
 def test_orphan_setsid_child_is_waited_and_reaped(self):
  p=self.plan();r=batch.run_plan(p,worker_commands=self.commands(p,'orphan'),interval=.01)
  self.assertEqual(r['status'],'complete');self.assertGreaterEqual(r['actual_wall_ns'],200000000)
  self.assertTrue(Path(p['output']).joinpath('worker-0/orphan-finished').exists());self.assert_closed(r)
 def test_failure_cost_and_cleanup_are_retained(self):
  p=self.plan();r=batch.run_plan(p,worker_commands=self.commands(p,'fail'),interval=.01,timeout=.7,cleanup_seconds=.2)
  self.assertEqual(r['status'],'failed');self.assertGreater(r['actual_wall_ns'],0)
  self.assertEqual(r['attempted_requests']+r['not_attempted_requests'],7);self.assert_closed(r)
  pidfile=Path(p['output'])/'worker-0/orphan.pid'
  if pidfile.exists():self.assertFalse(psutil.pid_exists(int(pidfile.read_text())))
 def test_compiler_cannot_escape_one_core_affinity(self):
  p=self.plan();r=batch.run_plan(p,worker_commands=self.commands(p,'escape'),interval=.01)
  self.assertEqual(r['status'],'failed');self.assertTrue(r['observations'][0]['affinity_violations']);self.assert_closed(r)
 def test_timeout_reclaims_async_descendants(self):
  p=self.plan();r=batch.run_plan(p,worker_commands=self.commands(p,'wait'),interval=.01,timeout=.25,cleanup_seconds=.2)
  self.assertEqual(r['status'],'failed');self.assertIn('TimeoutError',r['error']);self.assert_closed(r)
  pidfile=Path(p['output'])/'worker-0/orphan.pid'
  if pidfile.exists():self.assertFalse(psutil.pid_exists(int(pidfile.read_text())))
 def test_spawn_failure_is_real_observed_failure(self):
  p=self.plan();commands=self.commands(p);commands[0]=['/nonexistent-compiler-executable']
  r=batch.run_plan(p,worker_commands=commands,interval=.01,timeout=2,cleanup_seconds=.2)
  self.assertEqual(r['status'],'failed');self.assertEqual(r['observations'][0]['exit_code'],127);self.assert_closed(r)
 def test_signal_interrupt_closes_owned_workers(self):
  p=self.plan();commands=self.commands(p,'wait');pp=self.root/'external-plan.json';pp.write_text(json.dumps(p));cc=self.root/'commands.json';cc.write_text(json.dumps(commands))
  driver=self.root/'driver.py'
  driver.write_text('\n'.join(['import json,sys','from pathlib import Path',
    'sys.path.insert(0,'+repr(str(Path(batch.__file__).parent))+')',
    'import parallel_compile_batch as b',
    'p=json.loads(Path(sys.argv[1]).read_text());c=json.loads(Path(sys.argv[2]).read_text())',
    'r=b.run_plan(p,worker_commands=c,interval=.01,cleanup_seconds=.2)',
    'raise SystemExit(0 if r["status"]=="complete" else 1)'])+'\n')
  proc=subprocess.Popen([sys.executable,'-B',str(driver),str(pp),str(cc)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  deadline=time.monotonic()+4
  marker=Path(p['output'])/'worker-0/orphan.pid'
  while not marker.exists() and proc.poll() is None and time.monotonic()<deadline:time.sleep(.01)
  self.assertTrue(marker.exists());proc.terminate();out,err=proc.communicate(timeout=4)
  self.assertEqual(proc.returncode,1,(out,err));result=json.loads((Path(p['output'])/'summary.json').read_text())
  self.assertEqual(result['status'],'failed');self.assertTrue(result['descendants_reaped'])
  self.assertFalse(psutil.pid_exists(int(marker.read_text())))


 def test_real_worker_entry_calls_every_request_and_closes_session(self):
  import types
  from unittest.mock import patch
  p=self.plan();out=Path(p['output']);out.mkdir();pp=out/'plan.json';pp.write_text(json.dumps(p));h=batch.sha(pp)
  called=[];closed=[];settings=[]
  class Session:
   def __init__(self,cache_dir,*,compiler_input_reuse=False):
    self.cache=Path(cache_dir);self.n=0;self.reuse=compiler_input_reuse
    self_outer.assertTrue(compiler_input_reuse)
   def compile(self,directory):
    row=json.loads((Path(directory)/'execution.json').read_text())
    cfg=json.loads((Path(directory)/'emitted-config.json').read_text())
    self_outer.assertEqual(row['request'],cfg['request']);called.append(row['request']);self.n+=1
    return dict(checked_request=row['request'])
   def close(self):closed.append(str(self.cache))
   def summary(self):return dict(context_sha256='a'*64,compiler_api_calls=self.n,exact_cache_hits=0,compiler_input_hits=0)
  self_outer=self
  torch=types.SimpleNamespace(set_num_threads=lambda n:settings.append(('intra',n)),set_num_interop_threads=lambda n:settings.append(('inter',n)),get_num_threads=lambda:1,get_num_interop_threads=lambda:1)
  native=types.SimpleNamespace(CompilerSession=Session);affinity=os.sched_getaffinity(0)
  try:
   with patch.dict(sys.modules,{'torch':torch,'native_compile_cache':native}),patch.dict(os.environ,p['thread_environment']):
    for i in range(4):
     (out/('worker-'+str(i))).mkdir();self.assertEqual(batch.worker(pp,h,i),0)
  finally:os.sched_setaffinity(0,affinity)
  self.assertEqual(sorted(called),list(range(7)));self.assertEqual(len(called),len(set(called)))
  self.assertEqual(len(set(closed)),len(closed));self.assertTrue(all(n==1 for _,n in settings))
 def test_real_worker_failed_request_is_retained_and_session_closed(self):
  import types
  from unittest.mock import patch
  p=self.plan();out=Path(p['output']);out.mkdir();pp=out/'plan.json';pp.write_text(json.dumps(p));h=batch.sha(pp)
  closed=[];called=[]
  class Session:
   def __init__(self,cache_dir,*,compiler_input_reuse=False):self.n=0
   def compile(self,directory):
    self.n+=1;called.append(str(directory));raise RuntimeError('intentional injected compile failure')
   def close(self):closed.append(True)
   def summary(self):return dict(context_sha256='a'*64,compiler_api_calls=self.n,exact_cache_hits=0)
  torch=types.SimpleNamespace(set_num_threads=lambda n:None,set_num_interop_threads=lambda n:None,get_num_threads=lambda:1,get_num_interop_threads=lambda:1)
  affinity=os.sched_getaffinity(0)
  try:
   with patch.dict(sys.modules,{'torch':torch,'native_compile_cache':types.SimpleNamespace(CompilerSession=Session)}),patch.dict(os.environ,p['thread_environment']):
    for i in range(4):
     wd=out/('worker-'+str(i));wd.mkdir();owned=any(g['worker']==i for g in p['groups'])
     self.assertEqual(batch.worker(pp,h,i),1 if owned else 0)
     result=json.loads((wd/'worker-result.json').read_text())
     self.assertEqual(result['not_attempted_requests'],0)
     self.assertTrue(all(a['status']=='failed' for a in result['attempts']))
  finally:os.sched_setaffinity(0,affinity)
  self.assertEqual(len(called),7);self.assertTrue(closed)

if __name__=='__main__':unittest.main()
