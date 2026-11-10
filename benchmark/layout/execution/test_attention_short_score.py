"""CPU regressions for checked KV128 score reuse over an unpadded key prefix."""

import re
from unittest import TestCase, mock

from attention_fragment_layout import ROW_LANE_REGISTER
from attention_layout_search import (
    CONTRACT_FIELDS, K4, K_STAGE_CONTROL, K_STAGE_REUSE, PREVIEW_CAP,
    SHORT_SCORE_BULK, SHORT_SCORE_STAGED, select_attention,
)
from attention_tile_mma import ROOT_BYTES, THREADS
from tiled_attention import _output_distribution, checked_geometry, emit_tiled_attention
from valid_domain import dense_copy_execution


SHAPES = ((1, 8, 96), (1, 16, 97), (2, 20, 121), (1, 48, 127))
MMA = re.compile(r'^wgmma\.mma_async\.sync\.aligned\.m64n(\d+)k16\.')
WAIT = 'wgmma.wait_group.sync.aligned 0;'
PUBLISH = 'fence.proxy.async.shared::cta;'


def spec_for(batch, heads, length):
    return dict(B=batch, H=heads, N=length, D=64, q_block=64, kv_block=64)


def copy_packet():
    return {'operations': [{'name': 'frisk.copy', 'execution': dense_copy_execution((64, 64))}]}


def checked_tile(packet):
    return packet['operations'][0]['execution'], _output_distribution()


class ShortScoreAttentionTests(TestCase):
    def test_checked_tail_carrier_keeps_contract_and_drains_before_v(self):
        packet = copy_packet()
        for shape in (*SHAPES, (1, 1, 65)):
            spec = spec_for(*shape)
            length = spec['N']
            with self.subTest(length=length), mock.patch('tiled_attention._checked_tile_layouts', checked_tile):
                _, baseline = emit_tiled_attention(spec, tile_packet=packet, retain_tail_scores=True,
                    factored_copy=True, k_stage_policy='resident-k')
                for bulk in (False, True):
                    with self.subTest(bulk=bulk):
                        text, proof = emit_tiled_attention(spec, tile_packet=packet, kv_width=128,
                            retain_all_scores=True, tail_score_carrier=True, factored_copy=True,
                            bulk_value_publish=bulk)
                        self.assertEqual({key: proof[key] for key in CONTRACT_FIELDS},
                                         {key: baseline[key] for key in CONTRACT_FIELDS})
                        self.assertEqual(proof['head_bytes'], [128 * length] * 3 + [256 * length])
                        self.assertTrue(proof['tail_score_carrier'])
                        self.assertEqual((proof['execution_kv_width'], proof['score_evaluations']), (128, 1))
                        self.assertEqual((proof['score_valid_prefix'], proof['score_tail_prefix']),
                                         (length, length - 64))

                        # Independently enumerate the key coordinate at each
                        # holder, including the halves of the final packed pair.
                        counts = [sum(2 * (tid & 3) + (slot & 1) + 8 * (slot >> 2) < length
                                      for tid in range(THREADS)) for slot in range(64)]
                        self.assertEqual(list(proof['score_holder_counts']), counts)
                        self.assertEqual(sum(counts), length * 64)
                        slots = sum(count > 0 for count in counts)
                        self.assertEqual(proof['score_active_slots'], slots)

                        lines = text.split(r'\0A')
                        waits = [i for i, line in enumerate(lines) if line == WAIT]
                        fences = [i for i, line in enumerate(lines) if line == PUBLISH]
                        stores = [i for i, line in enumerate(lines) if 'st.shared.v4.b32' in line]
                        self.assertEqual([int(m[1]) for line in lines if (m := MMA.match(line))],
                                         [128] * 4 + [64] * 8)
                        self.assertEqual(len(waits), 2 if bulk else 3)
                        self.assertEqual(len(fences), len(waits))
                        self.assertEqual(len(stores), 20)
                        self.assertEqual(sum('ex2.approx.f32 ' in line for line in lines), slots)
                        self.assertEqual(sum('div.rn.f32 ' in line for line in lines), slots)
                        self.assertEqual(sum('ld.global.v4.b32 ' in line for line in lines),
                                         12 + 2 * ((length - 64 + 15) // 16))

                        # Q/K must be complete before the single QK group.
                        # Every V publication occurs after that QK drain or
                        # after the previous PV drain when reusing one root.
                        start = -1
                        for group, (fence, wait) in enumerate(zip(fences, waits, strict=True)):
                            window = lines[start + 1:wait]
                            self.assertLess(start, fence)
                            self.assertLess(fence, wait)
                            self.assertEqual(sum(MMA.match(line) is not None for line in window),
                                             8 if group and bulk else 4)
                            self.assertTrue(all(i > fence for i in range(start + 1, wait)
                                                if MMA.match(lines[i])))
                            self.assertEqual(sum(start < i < fence for i in stores),
                                             12 if group == 0 else (8 if bulk else 4))
                            start = wait
                        stage = proof['score_tail_schedule']
                        self.assertEqual(stage['wgmma_wait0'], len(waits))
                        self.assertEqual(stage['proxy_fences'], len(fences))
                        self.assertEqual(stage['barriers'], lines.count('bar.sync 0;'))
                        self.assertEqual(stage['kv_global_bytes'], 256 * length)
                        self.assertEqual(stage['kv_shared_store_bytes'], 4 * ROOT_BYTES)
                        self.assertEqual(proof['static_shared_bytes'], 3 * ROOT_BYTES)
                        self.assertEqual(tuple(stage['v_offsets']), tuple(stage['k_offsets']) if bulk
                                         else (stage['k_offsets'][-1],) * 2)

    def test_short_admission_fails_closed_outside_the_checked_policy(self):
        packet = copy_packet()
        base = dict(kv_width=128, retain_all_scores=True, tail_score_carrier=True)
        for length in (64, 128, 129):
            with self.subTest(length=length), self.assertRaises(ValueError):
                emit_tiled_attention(spec_for(1, 1, length), tile_packet=packet, **base)
        spec = spec_for(1, 1, 97)
        for changes in (dict(kv_width=64), dict(retain_all_scores=False),
                        dict(tail_score_carrier=False), dict(shared_swizzle=64),
                        dict(fragment_layout=ROW_LANE_REGISTER), dict(retain_tail_scores=True),
                        dict(narrow_tail=True), dict(streaming_bulk_values=True)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                emit_tiled_attention(spec, tile_packet=packet, **dict(base, **changes))
        with self.assertRaises(ValueError):
            checked_geometry(spec, kv_width=128)

    def test_both_modes_pay_for_two_short_carriers_and_the_matched_k_pair(self):
        with mock.patch('tiled_attention._checked_tile_layouts', checked_tile):
            modes, audit = select_attention(spec_for(1, 16, 97), copy_packet(), problem_sha256='0' * 64)
        expected = {SHORT_SCORE_STAGED, SHORT_SCORE_BULK, K_STAGE_CONTROL, K_STAGE_REUSE}
        self.assertLessEqual(audit['common_preview_attempts'], PREVIEW_CAP)
        self.assertEqual(audit['paid_slots_per_mode'], K4)
        for mode in ('J', 'LI'):
            with self.subTest(mode=mode):
                queue = modes[mode]
                self.assertEqual({item.schedule for item in queue}, expected)
                self.assertEqual(len({item.proof['ttgir_sha256'] for item in queue}), K4)
                paid = {item.schedule: item for item in queue}
                self.assertEqual(len({item.layout for item in queue}), 1)
                self.assertEqual(paid[SHORT_SCORE_STAGED].proof['pv_issue_groups'], 2)
                self.assertEqual(paid[SHORT_SCORE_BULK].proof['pv_issue_groups'], 1)
                for name in (SHORT_SCORE_STAGED, SHORT_SCORE_BULK):
                    self.assertEqual(paid[name].features['copy_tail']['storage_rows'], 64)
        self.assertEqual(len(modes['legacy']), K4)
        self.assertTrue(all(item.schedule not in expected for item in modes['legacy']))
