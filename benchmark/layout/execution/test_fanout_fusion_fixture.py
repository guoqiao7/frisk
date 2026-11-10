import unittest

from fanout_fusion_fixture import (CUTS, REDUCTION_POLICIES, carriers, checked_pair_score,
                                  select, source)


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

    def test_issued_work_penalizes_repeated_reduction_combines_and_divisions(self):
        features = dict(reduction_shuffles=4, global_sector_issues32_upper=256,
                        global_load_sites=2, global_store_sites=2)
        arithmetic = {'add.rn.f32': 8, 'shfl.sync.bfly.b32': 4, 'div.rn.f32': 2,
                      'ex2.approx.f32': 2, 'rsqrt.approx.f32': 2, 'mov.b32': 8}

        def trace(combines):
            return [dict(operation='frisk.reduce_tensor', reduction_axis=2,
                         reduction_combines=combines, shuffles=4)]

        baseline, proof = checked_pair_score(arithmetic, features, trace(4))
        replicated, _ = checked_pair_score(arithmetic, features, trace(8))
        shifted = dict(arithmetic, **{'div.rn.f32': 4, 'mov.b32': 6})
        expensive, _ = checked_pair_score(shifted, features, trace(4))
        self.assertEqual(proof['by_reduction'], [dict(axis=2, combines=4, shuffles=4)])
        self.assertEqual(replicated[0] - baseline[0], 4)
        self.assertEqual(expensive[0] - baseline[0], 24)
        with self.assertRaisesRegex(ValueError, 'does not match the checked emission'):
            checked_pair_score(arithmetic, features, trace(9))

    def test_tied_issue_cost_prefers_fewer_exact_divisions(self):
        trace = [dict(operation='frisk.reduce_tensor', reduction_axis=1,
                      reduction_combines=4, shuffles=2)]
        common = {'add.rn.f32': 4, 'shfl.sync.bfly.b32': 2}
        slower = dict(common, **{'div.rn.f32': 2})
        faster = dict(common, **{'div.rn.f32': 1, 'mov.b32': 13})
        features = dict(reduction_shuffles=2, global_sector_issues32_upper=128,
                        global_load_sites=2, global_store_sites=2)
        higher_sector = dict(features, global_sector_issues32_upper=136)
        more_divisions, _ = checked_pair_score(slower, features, trace)
        fewer_divisions, _ = checked_pair_score(faster, higher_sector, trace)
        self.assertEqual(more_divisions[0], fewer_divisions[0])
        self.assertLess(fewer_divisions, more_divisions)
