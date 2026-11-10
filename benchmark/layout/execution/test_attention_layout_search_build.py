"""Regression for shared native images across paid Attention queues."""

import json
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase, mock

import build_attention_layout_search as build


class AttentionSearchBuildTests(TestCase):
    def test_each_alias_rebinds_to_its_own_compile_future(self):
        def preview(layout, schedule, digest):
            return SimpleNamespace(layout=layout, schedule=schedule, tuning={}, features={},
                                   proof={'ttgir_sha256': digest, 'execution_kv_width': 64})

        a, b, c, d = (preview('coalesced', f'schedule-{key}', key) for key in 'abcd')
        different_owner = preview('warp-interleaved', a.schedule, 'e')
        modes = dict(J=[a, b, c, different_owner], LI=[a, b, c, d], legacy=[d, c, b, a])
        audit = dict(local_layout='coalesced', common_preview_attempts=12)

        def compile_one(_triton, _out, _case, _spec, item):
            return dict(status='compiled', ttgir_sha256=item.proof['ttgir_sha256'],
                        physical_compile_invoked=True, native_resources={})

        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, CUDA_VISIBLE_DEVICES=''),
            mock.patch.object(build, '_prepare', return_value=({}, {'paths': {'source': 'source-hash'}})),
            mock.patch.object(build, '_toolchain', return_value=(None, {})),
            mock.patch.object(build, '_sources', return_value={}),
            mock.patch.object(build, 'select_attention', return_value=(modes, audit)),
            mock.patch.object(build, 'compiled_score', return_value={'local_or_stack': False, 'cost': 1.0}),
            mock.patch.object(build, '_compile', side_effect=compile_one) as native,
        ):
            result = build.build_search(Path(tmp)/'archive', sys.executable, [(1, 1, 64)], mode='all')
            document = json.loads(result.read_text())

        self.assertEqual(native.call_count, 5)
        self.assertEqual(document['native_compiler_invocations'], 5)
        self.assertEqual(document['charged_slots'], 12)
        rows = {(row['engine'], row['slot']): row for row in document['cases']}
        for engine, candidates in modes.items():
            for slot, item in enumerate(candidates):
                row = rows[engine, slot]
                self.assertEqual(row['charged_slot'], slot)
                self.assertEqual(row['ttgir_sha256'], item.proof['ttgir_sha256'])
                if alias := row.get('compile_alias_of'):
                    origin = rows[alias['engine'], alias['slot']]
                    self.assertTrue(origin['native_compiler_invoked'])
                    self.assertEqual(origin['ttgir_sha256'], row['ttgir_sha256'])
