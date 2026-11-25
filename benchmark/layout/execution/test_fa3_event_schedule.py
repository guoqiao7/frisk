import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from attention_pipeline_selection import AttentionPipeline, attention_pipelines, compile_attention_pipeline
from fa3_event_schedule import FA3EventSchedule, SCHEDULES, resolve_events


class FA3EventTest(unittest.TestCase):
    def test_lookahead_dependencies(self):
        accepted = set()
        for order in itertools.permutations(("qk", "release_k", "pv", "update")):
            try:
                accepted.add(FA3EventSchedule("lookahead", order))
            except ValueError:
                pass
        self.assertEqual(accepted, {v for v in SCHEDULES.values() if v.frame == "lookahead"})
        with self.assertRaises(ValueError):
            FA3EventSchedule("same_tile", SCHEDULES["late_k"].steady)
        with self.assertRaises(ValueError):
            FA3EventSchedule("lookahead", ("qk", "pv", "pv", "update"))
        with self.assertRaises(ValueError):
            resolve_events("single", "late_k")

    def test_fuse_only_adjacent_return(self):
        self.assertEqual(resolve_events("fa3_same_tile"), SCHEDULES["same_tile"])
        self.assertEqual(resolve_events("fa3_qk_next"), SCHEDULES["qk_next"])
        self.assertEqual(SCHEDULES["qk_next"].statements()[0], "scores(t+1);")
        self.assertEqual(SCHEDULES["late_k"].statements()[:3],
                         ["scores_defer_k(t+1);", "accumulate(t);", "bars[K_FREE+kr.slot(t+1)].arrive();"])

    def test_register_variant_keeps_event_schedule(self):
        with TemporaryDirectory() as scratch:
            path=Path(scratch)/"imported.json"
            path.write_text("{}")
            with patch("attention_pipeline_candidates.from_overlap", return_value={"kind":"fa3_qk_next"}):
                plans=attention_pipelines(16,[path],include_late_k=True)
            late=[p for p in plans if p.event_schedule=="late_k"]
            self.assertEqual([(p.name,p.writeback) for p in late],
                             [("imported-late-k","tma"),("imported-late-k-register","register")])

    def test_reuse_requires_same_ir_and_physical_contract(self):
        def fixture(qtile, dim, rows, heads, kvheads, causal, execution_contract, **kwargs):
            return json.dumps([rows, execution_contract])

        def solve(cmd, check):
            folder = Path(cmd[1]).parent
            (folder/"execution.json").write_text((folder/"input.mlir").read_text())
            (folder/"solved.mlir").write_text("checked")

        def lower(data, event_schedule, **options):
            return None, {"execution_contract": dict(data[1], kind=event_schedule.contract_kind)}

        with TemporaryDirectory() as scratch, patch("attention_pipeline_selection.online_fixture", fixture), \
                patch("attention_pipeline_selection.subprocess.run", side_effect=solve) as optimizer, \
                patch("attention_pipeline_selection.compile_execution", side_effect=lower):
            cache = {}
            for name,kind,events,arena,rows in (
                ("first","fa3_same_tile",None,163840,1024),
                ("next","fa3_qk_next",None,163840,1024),
                ("late","fa3_qk_next","late_k",163840,1024),
                ("new_arena","fa3_qk_next",None,196608,1024),
                ("new_rows","fa3_qk_next",None,163840,2048),
            ):
                p=AttentionPipeline(name,128,dict(kind=kind,arena=arena),event_schedule=events)
                _,info=compile_attention_pipeline(p,"frisk-opt",Path(scratch)/name,
                    rows=rows,heads=16,dim=128,causal=False,_layout_cache=cache)
                if name in ("next","late"):
                    self.assertEqual(Path(info["source_packet"]).parent.name,"first")
            self.assertEqual(optimizer.call_count,3)
