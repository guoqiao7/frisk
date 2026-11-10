"""Check the shape-guarded K-stage queue before native compilation."""

from unittest import TestCase

from attention_layout_search import K4, K_STAGE_CONTROL, K_STAGE_REUSE, schedules
from tiled_attention import checked_geometry


class AttentionKStageSearchTests(TestCase):
    def test_checked_reuse_and_matched_control_share_flags(self):
        policies = {96: ('late-v-control', 'resident-k'), 127: ('late-v-control', 'resident-k'),
                    128: ('boundary-control', 'boundary-reuse'),
                    704: ('boundary-control', 'boundary-reuse'),
                    1024: ('boundary-control', 'boundary-reuse')}
        for length, pair in policies.items():
            with self.subTest(length=length):
                legacy, common = schedules(length)
                self.assertEqual(len(legacy), K4)
                self.assertNotIn(K_STAGE_REUSE, legacy)
                control = dict(common[K_STAGE_CONTROL])
                reuse = dict(common[K_STAGE_REUSE])
                self.assertEqual((control.pop('k_stage_policy'), reuse.pop('k_stage_policy')), pair)
                self.assertEqual(control, reuse)
                self.assertTrue(reuse['factored_copy'])
                self.assertEqual(reuse['kv_width'], 64)
                self.assertEqual(reuse['retain_tail_scores'], length < 128)

    def test_short_legacy_menu_charges_four_legal_pre_reuse_slots(self):
        for length in (65, 96, 127):
            spec = dict(B=1, H=1, N=length, D=64, q_block=64, kv_block=64)
            legacy, _ = schedules(length)
            self.assertEqual(next(iter(legacy)), 'kv64')
            self.assertEqual(len(legacy), K4)
            for candidate in legacy.values():
                checked_geometry(spec, kv_width=candidate['kv_width'])
                self.assertNotIn('k_stage_policy', candidate)

    def test_long_tail_keeps_previous_queue(self):
        legacy, common = schedules(705)
        self.assertEqual(len(legacy), K4)
        self.assertNotIn(K_STAGE_CONTROL, common)
        self.assertNotIn(K_STAGE_REUSE, common)
