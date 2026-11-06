"""Budget, conservative fixed selection, and meaningful baseline separation."""

from unittest import TestCase
from unittest.mock import patch

import execution_plan_search as search
from execution_plan_gate import promotion_verdict


class SelectionTests(TestCase):
    def run_pool(self, native=4, budget=6, tradeoff=True):
        rows=[search.LayoutReplay(name,dict(name=name,operations=[{'name':'frisk.tile_store'}]),'a'*64)
              for name in ('local','global')]

        def emit(policy, packet):
            value=packet['name']
            cost={'direct':10,'shared':2}[policy.store]
            config=dict(copy_path='vector',global_alignments=[16]*4,root_order=[0,1,2,3],tma_plans=[],
                        static_shared_bytes=0,cost=cost-(value=='global'),strategy=policy.store)
            return value+':'+policy.store,config

        def comparison(a,b):
            relation='incomparable' if tradeoff else ('dominates' if a['cost']<b['cost'] else 'equal')
            return dict(structural_relation=relation)

        with patch.object(search.Policy,'emit',emit), patch.object(search,'extract_features',side_effect=lambda _,cfg,**k:cfg),              patch.object(search,'static_key',side_effect=lambda f,**k:f['cost']),              patch.object(search,'compare_features',side_effect=comparison):
            return search.select_pairs(rows,local_incumbent='local',baseline_layout='local',
                                       legacy_layouts=['local'],grid=(1,1,1),preview_budget=budget,native_budget=native)

    def test_tradeoffs_remain_exploration_and_fixed_is_pinned(self):
        queues,audit=self.run_pool()
        self.assertEqual(queues['J'][0].layout,'local')
        self.assertEqual(queues['J'][0].policy,search.BASELINE)
        self.assertEqual({p.layout for p in queues['LI']},{'local'})
        self.assertEqual(len(queues['legacy']),1)
        self.assertEqual(len(audit['preview_attempts']),6)
        self.assertEqual(audit['distinct_programs'],4)
        self.assertTrue(any(p.policy.store=='shared' for p in queues['J'][1:]))

    def test_dominance_can_change_fixed_and_single_slot_keeps_fallback(self):
        queues,_=self.run_pool(tradeoff=False)
        self.assertEqual(queues['J'][0].layout,'global')
        self.assertEqual(queues['J'][0].policy.store,'shared')
        self.assertEqual(queues['J'][1].policy,search.BASELINE)
        self.assertEqual(self.run_pool(native=1,tradeoff=False)[0]['J'][0].policy,search.BASELINE)

    def test_exhaustion_and_problem_binding(self):
        self.assertEqual(len(self.run_pool(budget=2)[1]['preview_attempts']),2)
        rows=[search.LayoutReplay(x,{},x*64) for x in 'ab']
        with self.assertRaisesRegex(ValueError,'checked prepared problem'):
            search.select_pairs(rows,local_incumbent='a',baseline_layout='a',legacy_layouts=['a'],grid=(1,1,1))

    def test_joint_k4_reserves_a_tied_grouped_mma_swizzle(self):
        layouts = []
        for name, swizzles in (('local', (32, 0)), ('two-changes', (0, 32)), ('one-change', (32, 32))):
            contract = {role: {'swizzle': width} for role, width in
                        zip(('a_descriptor', 'b_descriptor'), swizzles)}
            op = dict(name='frisk.mma', attrs={'frisk.mma_contract': contract})
            layouts.append(search.LayoutReplay(name, dict(name=name, operations=[op]), 'a'*64))

        def emit(policy, packet):
            cost = 0 if policy.mma_group == 16 else 1 if policy == search.BASELINE else 2
            config = dict(copy_path='vector', global_alignments=[16]*4, root_order=[0,1,2,3],
                          tma_plans=[], static_shared_bytes=0, scope='mma', static={'cost': cost})
            return f"{packet['name']}:{policy}", config

        def compare(a, b):
            relation = 'dominates' if a['static']['cost'] < b['static']['cost'] else 'equal'
            return dict(structural_relation=relation)

        with patch.object(search.Policy, 'emit', emit), \
             patch.object(search, 'extract_features', side_effect=lambda _, cfg, **kwargs: cfg), \
             patch.object(search, 'static_key', side_effect=lambda features, **kwargs: features['static']['cost']), \
             patch.object(search, 'compare_features', side_effect=compare):
            queues, audit = search.select_pairs(layouts, local_incumbent='local', baseline_layout='local',
                                                 legacy_layouts=['local'], grid=(1,1,1))
            capped, cap_audit = search.select_pairs(layouts, local_incumbent='local', baseline_layout='local',
                                                    legacy_layouts=['local'], grid=(1,1,1), native_budget=3)

        self.assertEqual(len(queues['J']), 4)
        self.assertEqual(queues['J'][:3], queues['LI'][:3])
        self.assertEqual(queues['J'][3].layout, 'one-change')
        self.assertEqual(queues['J'][3].policy, queues['J'][0].policy)
        self.assertEqual(audit['joint_swizzle_probe']['slot'], 3)
        self.assertEqual(queues['legacy'][0].policy, search.BASELINE)
        self.assertEqual(capped['J'][:2], queues['J'][:2])
        self.assertIsNone(cap_audit['joint_swizzle_probe'])

    def test_schedule_ablation_matches_three_slots_and_spends_one_on_joint_layout(self):
        ops=[dict(name=name) for name,count in search.ATTENTION_SCOPE.items() for _ in range(count)]
        layouts=[search.LayoutReplay(name,dict(name=name,operations=ops),'a'*64)
                 for name in ('local','joint','unused')]
        def emit(policy, packet):
            shared=0 if packet['name']=='unused' else 128
            config=dict(copy_path='vector',global_alignments=[16]*4,root_order=[0,1,2,3],
                        tma_plans=[],static_shared_bytes=0,scope='one-problem',
                        static={'shared_allocated_peak_bytes':shared})
            return f"{packet['name']}:{policy}",config
        with patch.object(search.Policy,'emit',emit), \
             patch.object(search,'extract_features',side_effect=lambda _,cfg,**kwargs:cfg), \
             patch.object(search,'static_key',side_effect=lambda f,**kwargs:f['static']['shared_allocated_peak_bytes']):
            queues,audit=search.select_pairs(layouts,local_incumbent='local',baseline_layout='local',
                                            legacy_layouts=['local'],grid=(1,1,1),
                                            attention_schedule_ablation=True,joint_layouts=('local','joint'))
        self.assertEqual(queues['J'][:3],queues['LI'][:3])
        self.assertEqual([p.policy.shared for p in queues['LI']],['declared','late','reuse-k','declared'])
        self.assertEqual((queues['J'][3].layout,queues['J'][3].policy.shared),('joint','reuse-k'))
        self.assertEqual(audit['joint_schedule_probe']['slot'],3)
        self.assertEqual(len(audit['preview_attempts']),15)
        self.assertEqual(queues['legacy'][0].policy,search.BASELINE)

    def test_promotion_requires_regret_and_complete_scope(self):
        item=dict(fixed_ratios={'frisk:J/frisk:legacy':{'bonferroni_ci95':[.9,.92]}},
                  descriptive_finite_pool_regret={'frisk:J':{'ci95':[1,1.01]}})
        self.assertEqual(promotion_verdict({'a':item},expected_cases=['a'],complete_scope=True)['status'],'passed')
        self.assertEqual(promotion_verdict({'a':item},expected_cases=['a'],complete_scope=False)['status'],'held')
        item['descriptive_finite_pool_regret']['frisk:J']['ci95'][1]=1.2
        self.assertEqual(promotion_verdict({'a':item},expected_cases=['a'],complete_scope=True)['status'],'held')
