import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import pipeline_observer as o

class PipelineObserverTest(unittest.TestCase):
    def test_real_process_tree_and_failure_cost_are_retained(self):
        with tempfile.TemporaryDirectory(prefix='frisk-observer-') as tmp:
            cores=[min(os.sched_getaffinity(0))]
            good=o.observe(Path(tmp)/'ok',[sys.executable,'-c',"import time; a=bytearray(8000000); print('ready',flush=True); time.sleep(.12)"],affinity=cores,interval=.02)
            bad=o.observe(Path(tmp)/'bad',[sys.executable,'-c',"import sys; print('retained'); sys.exit(7)"],affinity=cores,interval=.02)
            self.assertEqual((good['exit_code'],bad['exit_code']),(0,7))
            self.assertGreater(good['sampled_peak_tree_rss_bytes'],8000000)
            self.assertFalse(good['affinity_violations']);self.assertTrue(good['child_directly_reaped'])
            self.assertIn('retained',(Path(tmp)/'bad/stdout.log').read_text())
            for directory,result in [('ok',good),('bad',bad)]:
                for name,row in result['files'].items():self.assertEqual(o._sha(Path(tmp)/directory/name),row['sha256'])
            s=o.summarize([good,bad]);self.assertEqual((s['phases'],s['failures']),(2,1))
            self.assertGreaterEqual(s['p95_phase_seconds'],s['p50_phase_seconds'])

    def test_spawn_failure_has_completion_and_cost(self):
        with tempfile.TemporaryDirectory(prefix='frisk-observer-spawn-') as tmp:
            d=Path(tmp)/'failed'
            result=o.observe(d,['/definitely/not/a/frisk/executable'],affinity=[min(os.sched_getaffinity(0))],interval=.01)
            self.assertEqual(result['exit_code'],127);self.assertEqual(result['status'],'failed')
            self.assertTrue((d/'completion.json').is_file());self.assertGreater(result['wall_ns'],0)
    def test_orphan_descendant_is_included_in_wall_time(self):
        with tempfile.TemporaryDirectory(prefix='frisk-observer-child-') as tmp:
            source="import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(.3)'])"
            result=o.observe(Path(tmp)/'children',[sys.executable,'-c',source],affinity=[min(os.sched_getaffinity(0))],interval=.01)
            self.assertEqual(result['status'],'complete');self.assertTrue(result['scope_complete'])
            self.assertFalse(result['remaining_descendants']);self.assertGreater(result['wall_ns']/1e9,.25)

if __name__=='__main__':unittest.main()
