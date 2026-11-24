import unittest
from unittest.mock import Mock, patch

from fa3_static_schedule import _after_prefix, rank_fa3_events
from native_fa3_execution import compile_budgeted_fa3, select_budgeted_fa3


class BudgetedFA3Test(unittest.TestCase):
    plan = dict(kind="fa3_qk_next", buffers={"k":{"versions":2},"v":{"versions":2}},
                batch=8, wait_pending=0)

    def test_count_refills_for_causal_and_rectangular_workloads(self):
        for q in range(1,10):
            for k in range(1,9):
                for causal in (True,False):
                    for prefix in (0,1,3):
                        active=[min(i+1,k) if causal else k for i in range(q)]
                        self.assertEqual(_after_prefix(q,k,causal,prefix),
                                         sum(max(0,n-prefix) for n in active))

    def test_unused_slots_and_exposed_refills(self):
        short=rank_fa3_events(self.plan,384,384,False)
        deep=rank_fa3_events(self.plan,2048,2048,True,2)
        self.assertEqual(short["possible_k_refills_per_head"],0)
        self.assertEqual(short["selected"],["same_tile"])
        self.assertIn("late_k",deep["selected"])
        self.assertGreater(deep["possible_k_refills_per_head"],0)
        with self.assertRaises(ValueError): rank_fa3_events(self.plan,2048,2048,True,3)
        with self.assertRaises(ValueError): rank_fa3_events(self.plan,129,2048,False,1)

    def test_phase_accounting_at_fill_and_drain(self):
        for q,k,h,c in [(256,129,4,True),(256,384,16,False),(2048,513,32,True)]:
            report=rank_fa3_events(self.plan,q,k,c,2,heads=h)
            self.assertAlmostEqual(report["effective_transitions"]-report["effective_same_tile_middle"],
                                   report["separate_last_fraction"])
            for cost in report["costs"].values():
                self.assertAlmostEqual(cost["total_us"],sum(cost[key] for key in
                    ("startup_us","steady_us","drain_us","boundary_correction_us","wave_adjustment_us")))
        for c in (False,True):
            self.assertEqual(rank_fa3_events(self.plan,384,384,c)["selected"],["same_tile"])
        for n in (1024,8192):
            self.assertEqual(rank_fa3_events(self.plan,n,n,False)["selected"],["qk_next"])

    def test_initial_ring_and_unreachable_causal_fringe(self):
        for keys, expected in [(129, 1), (384, 2), (1024, 2)]:
            score = rank_fa3_events(self.plan, 256, keys, False)
            self.assertEqual(score["effective_initial_ring_transitions"], expected)
        # Half the weight goes to the longest CTA when the grid fills two waves.
        score = rank_fa3_events(self.plan, 8 * 128, 385, True, heads=4, sm_count=16)
        counts = [min(2, min(block + 1, 4) - 1) for block in range(8)]
        self.assertAlmostEqual(score["effective_initial_ring_transitions"],
                               (sum(counts) / len(counts) + max(counts)) / 2)
        full = rank_fa3_events(self.plan, 256, 1024, True)
        fringe = rank_fa3_events(self.plan, 256, 1023, True)
        self.assertEqual(fringe["costs"], full["costs"])

    def test_single_wave_lookahead_refills(self):
        sparse = rank_fa3_events(self.plan, 1024, 1024, False, heads=6)
        repeated = rank_fa3_events(self.plan, 1024, 1024, False, heads=16)
        self.assertEqual(sparse["selected"], ["same_tile"])
        self.assertEqual(repeated["selected"], ["qk_next"])
        self.assertGreater(sparse["lookahead_grid_us"], 0)
        self.assertEqual(repeated["lookahead_grid_us"], 0)
        for keys, causal in [(384, False), (895, False), (896, True)]:
            self.assertEqual(rank_fa3_events(self.plan, 896, keys, causal,
                                             heads=6)["lookahead_grid_us"], 0)

    def test_causal_tail_is_ranked_with_its_admitting_cta(self):
        result = rank_fa3_events(self.plan, 896, 896, True, 2, heads=18)
        self.assertEqual(result["selected"][0], "same_tile")
        # A grid that fits on the SMs uses the existing single-wave score.
        one_wave = rank_fa3_events(self.plan, 896, 896, True, heads=16)
        self.assertTrue(all(cost["wave_adjustment_us"] == 0
                            for cost in one_wave["costs"].values()))

    def test_screen_compensates_for_launch_position_and_missing_orders(self):
        images = {"A": None, "B": None}
        def observations(speedup):
            rows = []
            for cycle in range(24):
                for sequence in [("A", "B"), ("B", "A")]:
                    us = {"A": 10, "B": 10 * speedup}
                    us[sequence[0]] *= 1.06
                    rows.append(dict(cycle=cycle, order=sequence, us=us))
            return rows
        self.assertEqual(select_budgeted_fa3(images, lambda _: observations(1)), "A")
        self.assertEqual(select_budgeted_fa3(images, lambda _: observations(.98)), "B")
        self.assertEqual(select_budgeted_fa3(images, lambda _: observations(.98)[::2]), "A")
        with self.assertRaises(ValueError):
            select_budgeted_fa3(images, lambda _: [observations(1)[0]] * 2)

    def test_default_launches_one_native_compile(self):
        with patch("native_fa3_execution._geometry",return_value=(1024,16,False)), \
                patch("native_fa3_execution._kv_rows",return_value=1024), \
                patch("native_fa3_execution._prepare_transport") as prepare, \
                patch("native_fa3_execution.rank_transport_events",return_value={"selected":["qk_next"]}), \
                patch("backend.compile_execution",return_value=(object(),{})) as compile_one:
            prepare.return_value.facts={"moves":{"q":{"box":[64,128]}}}
            images,receipt=compile_budgeted_fa3({})
            self.assertEqual(compile_one.call_args.kwargs["cta_dispatch"],"forward")
            self.assertEqual(receipt["selected"],["qk_next"])
            prepare.assert_called_once_with({}, "tma")
            compile_one.assert_called_once()
            self.assertIs(compile_one.call_args.kwargs["fa3_transport"],prepare.return_value)
            self.assertEqual(compile_one.call_args.kwargs["fa3_writeback"],"tma")
            never=Mock(side_effect=AssertionError)
            self.assertEqual(select_budgeted_fa3(images,never),"qk_next")
            never.assert_not_called()

    def test_dual_selection_uses_paired_ratios_and_retains_primary_on_tie(self):
        images={"first":None,"second":None}
        self.assertEqual(select_budgeted_fa3(images,lambda _:[{"first":10,"second":9}]),"first")
        self.assertEqual(select_budgeted_fa3(images,lambda _:[{"first":10,"second":9}]*12),"second")
        self.assertEqual(select_budgeted_fa3(images,lambda _:[{"first":2,"second":2}]),"first")
        for rows in ([],[{"first":2}],[{"first":2,"second":float("nan")}]):
            with self.assertRaises(ValueError): select_budgeted_fa3(images,lambda _:rows)
        with self.assertRaises(ValueError): select_budgeted_fa3(images)
