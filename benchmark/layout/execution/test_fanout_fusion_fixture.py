import unittest

from fanout_fusion_fixture import CUTS, REDUCTION_POLICIES, carriers, select, source


class FanoutFusionTests(unittest.TestCase):
    def test_cut_is_shared_by_both_reductions_and_broadcast_consumers(self):
        text = source(64, 4, 'A', 'B')
        self.assertEqual(text.count('frisk.convert_layout'), 1)
        self.assertEqual(text.count('frisk.reduce_tensor %transported'), 2)
        self.assertEqual(text.count('arith.divf %transported'), 2)
        self.assertNotIn('frisk.convert_layout', source(64, 4, 'B', 'B'))
        a, b = carriers(64, 4).values()
        self.assertEqual(a.warps, b.warps)
        self.assertEqual(len(set(a.registers) ^ set(b.registers)), 2)
        self.assertEqual(len(set(a.lanes) ^ set(b.lanes)), 2)

    def test_strict_local_sees_both_completions_and_spends_four_slots(self):
        full = {'AA':237, 'AB':225, 'BA':257, 'BB':209}
        inventory = {(cut, policy):dict(local_score=(136,2) if cut[0]=='A' else (144,4),
                                        full_score=(full[cut]+100*(policy=='eager'),))
                     for cut in CUTS for policy in REDUCTION_POLICIES}
        queues, fixed, local_scores, selected = select(inventory)
        self.assertEqual(selected, 'A')
        self.assertLess(local_scores['A'], local_scores['B'])
        self.assertEqual(set(queues['LI']), {('AA','auto'),('AA','eager'),('AB','auto'),('AB','eager')})
        self.assertEqual(queues['LI'][fixed['LI']], ('AB','auto'))
        self.assertEqual(queues['J'][fixed['J']], ('BB','auto'))
        inventory['AB','auto']['local_score'] = (144,4)
        with self.assertRaisesRegex(ValueError, 'changed the own-producer score'):
            select(inventory)
