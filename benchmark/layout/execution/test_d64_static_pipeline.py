import unittest
from d64_static_pipeline import features, rank_d64_pipelines

class StaticD64Test(unittest.TestCase):
    def test_short_launch_only_has_legal_plans(self):
        result = rank_d64_pipelines(129, 3, False)
        self.assertTrue(result['estimated_us'])
        self.assertTrue(all(name.startswith('cp-') for name in result['estimated_us']))

    def test_causal_counts_actual_tile_visits(self):
        # Three Q64 CTAs per head at N=150. Only the third visits both KV128 tiles.
        regular = features(150, 2, False, 64, 128, 3, 114)
        masked = features(150, 2, True, 64, 128, 3, 114)
        self.assertAlmostEqual(regular[4], 12/114)
        self.assertAlmostEqual(masked[4], 8/114)

    def test_long_launch_preserves_causal_transport_contract(self):
        regular = rank_d64_pipelines(4099, 10, False)
        causal = rank_d64_pipelines(4099, 10, True, causal_v_tma=False)
        self.assertTrue(any(name.endswith('-v-tma') for name in rank_d64_pipelines(4099,10,True)['estimated_us']))
        self.assertTrue(any(name.endswith('-v-tma') for name in regular['estimated_us']))
        self.assertFalse(any(name.endswith('-v-tma') for name in causal['estimated_us']))
        self.assertIn(regular['selected'], regular['estimated_us'])

    def test_long_causal_transport_compares_both_tma_widths(self):
        result = rank_d64_pipelines(8192, 8, True, causal_v_tma=False)
        self.assertEqual(result['selected'], 'q64-kv128-k-tma')

    def test_grouped_static_obeys_ring_depth_and_head_mapping(self):
        from d64_static_pipeline import rank_grouped_attention
        out=rank_grouped_attention(192,16,4,128,True)
        self.assertEqual(list(out['estimated_ratio']),["cp-q64-producer-wait4-late-v"])
        self.assertNotIn("q64-kv128-k-tma-v-tma",rank_grouped_attention(384,16,4,128,False)['estimated_ratio'])
        with self.assertRaises(ValueError):
            rank_grouped_attention(512,16,3,128,True)

    def test_grouped_width_compares_one_fixed_transport_family(self):
        for rows, heads, kv_heads, causal, width in ((1729,20,4,True,128),(3520,8,1,False,64)):
            result=rank_d64_pipelines(rows,heads,causal,kv_heads=kv_heads)
            comparison=result["grouped_width"]
            expected_at_q64 = comparison[f"kv{width}"]
            self.assertEqual(min((comparison["kv64"], comparison["kv128"]),
                                 key=result["estimated_us"].get), expected_at_q64)
            if causal:
                self.assertEqual(result["selected"], expected_at_q64)
            self.assertEqual(comparison["kv64"].replace("kv64","kv128"),comparison["kv128"])
            self.assertNotIn("grouped_width",rank_d64_pipelines(rows,heads,causal))
        self.assertNotIn("grouped_width",rank_d64_pipelines(384,16,True,kv_heads=4))

    def test_joint_geometry_can_reverse_width_at_the_second_query_tile(self):
        import math
        rank = rank_d64_pipelines(3520,8,False,kv_heads=1)
        joint = rank["grouped_tiles"]
        scores = joint["estimated_us"]
        self.assertEqual(len(scores), 4)
        narrow, wide = ("q64-kv64-k-tma-post-qk-v-tma", "q64-kv128-k-tma-post-qk-v-tma")
        self.assertLess(scores[narrow], scores[wide])
        narrow, wide = (n.replace("q64-", "q128-", 1) for n in (narrow, wide))
        self.assertLess(scores[wide], scores[narrow])
        self.assertEqual(rank["selected"], wide)
        self.assertAlmostEqual(math.log(scores[narrow] / scores[wide]),
                               joint["q128_kv64_over_kv128_log_ratio"])

    def test_joint_geometry_admission_and_comparison_with_other_families(self):
        from d64_static_pipeline import _score_joint_grouped_tiles
        for rank in (rank_d64_pipelines(320,8,False,kv_heads=2),
                     rank_d64_pipelines(1024,8,False),
                     rank_d64_pipelines(1024,8,True,kv_heads=2),
                     rank_d64_pipelines(1024,8,False,kv_heads=2,output_bits=16)):
            self.assertNotIn("grouped_tiles", rank)
        costs = {"cp-q64-wait4": 1., "q64-kv64-k-tma-post-qk-v-tma": 12.,
                 "q64-kv128-k-tma-post-qk-v-tma": 10.}
        pair, joint = _score_joint_grouped_tiles(costs, 640, 8, 114)
        self.assertIsNone(pair)
        self.assertEqual(len(joint["estimated_us"]), 4)
        self.assertEqual(min(costs, key=costs.get), "cp-q64-wait4")

    def test_underfilled_joint_tiles_include_finite_loop_cost(self):
        for rows, heads, kv_heads, width in ((560, 20, 10, 64), (704, 12, 4, 128)):
            result = rank_d64_pipelines(rows, heads, False, kv_heads=kv_heads)
            winner = f"q64-kv{width}-k-tma-post-qk-v-tma"
            joint = result["grouped_tiles"]
            v = joint["v_transport"]
            self.assertEqual(min(v["sw32_estimated_us"], key=v["sw32_estimated_us"].get), winner)
            self.assertEqual(result["selected"], "q64-kv128-k-tma-post-qk-v-tma")
            self.assertEqual(joint["light_grid"]["weight"], 1)
            self.assertEqual(min(joint["light_grid"]["service_us"],
                                 key=joint["light_grid"]["service_us"].get), winner)
            self.assertEqual(joint["estimated_us"][winner], result["estimated_us"][winner])
        full_grid = rank_d64_pipelines(2336, 18, False, kv_heads=6)
        self.assertNotIn("light_grid", full_grid["grouped_tiles"])

    def test_shared_kv_reads_change_query_choice_at_fixed_grid(self):
        import math
        sparse = rank_d64_pipelines(552, 22, False, kv_heads=11)
        shared = rank_d64_pipelines(552, 22, False, kv_heads=2)
        self.assertTrue(sparse["selected"].startswith("q64-"))
        prior = shared["grouped_tiles"]["v_transport"]["sw32_estimated_us"]
        self.assertTrue(min(prior, key=prior.get).startswith("q128-"))
        self.assertEqual(shared["selected"], "q64-kv128-k-tma-post-qk-v-tma")
        a, b = (r["grouped_tiles"] for r in (sparse, shared))
        self.assertGreater(b["light_grid"]["q64_over_q128_log_adjustment"],
                           a["light_grid"]["q64_over_q128_log_adjustment"])
        for q in (64, 128):
            narrow = f"q{q}-kv64-k-tma-post-qk-v-tma"
            wide = narrow.replace("kv64", "kv128")
            prior_scores = [r["v_transport"]["sw32_estimated_us"] for r in (a, b)]
            ratios = [math.log(v[narrow] / v[wide]) for v in prior_scores]
            self.assertAlmostEqual(*ratios)
        self.assertNotIn("grouped_tiles", rank_d64_pipelines(552, 22, True, kv_heads=2))

    def test_grouped_query_geometry_preserves_width_and_public_contract(self):
        from d64_static_pipeline import rank_grouped_attention
        receipt = rank_grouped_attention(1984,14,2,64,False)
        pair = receipt["grouped_query_tile"]
        self.assertEqual(pair["q128"], receipt["selected"])
        self.assertEqual(pair["q64"].replace("q64-", "q128-", 1), pair["q128"])
        self.assertEqual(pair["critical_ctas"], [4,2])
        self.assertEqual(pair["resident_waves"], [1,1])
        for result in (rank_d64_pipelines(1984,14,False),
                       rank_grouped_attention(1984,14,2,64,True),
                       rank_grouped_attention(1984,14,2,64,False,output_bits=16)):
            self.assertNotIn("grouped_query_tile", result)

    def test_grouped_kv128_partial_layer_service_is_monotonic(self):
        from d64_static_pipeline import _grouped_kv128_service
        for q, capacity in ((64,3), (128,2)):
            times = [_grouped_kv128_service(q, blocks, 114, capacity)
                     for blocks in range(1, 114 * 12)]
            self.assertTrue(all(b + 1e-10 >= a for a,b in zip(times, times[1:])))

    def test_grouped_kv128_query_tile_charges_partial_sm_layer(self):
        for rows, kv_heads, old_query, counts in ((2336,6,128,[666,342]), (4160,3,64,[1170,594])):
            ranked = rank_d64_pipelines(rows,18,False,kv_heads=kv_heads)
            pair = ranked["grouped_query_tile"]
            self.assertEqual(pair["cta_counts"], counts)
            prior = ranked["grouped_tiles"]["v_transport"]["sw32_estimated_us"]
            self.assertEqual(min(prior, key=prior.get), pair[f"q{old_query}"])
            self.assertEqual(ranked["selected"], pair["q64"])
            self.assertIn("kv128-k-tma-post-qk-v-tma", ranked["selected"])

    def test_row_major_v_cost_is_geometry_specific(self):
        for rows, heads, kv_heads in ((696, 15, 1), (552, 22, 2)):
            rank = rank_d64_pipelines(rows, heads, False, kv_heads=kv_heads)
            joint = rank["grouped_tiles"]
            receipt = joint["v_transport"]
            self.assertEqual((receipt["swizzle_bytes"], receipt["tensor_map_rank"], receipt["kv_slots"]),
                             (128, 3, 2))
            self.assertEqual(rank["selected"], "q64-kv128-k-tma-post-qk-v-tma")
            narrow = "q128-kv64-k-tma-post-qk-v-tma"
            self.assertLess(receipt["time_ratio_to_sw32"][rank["selected"]],
                            receipt["time_ratio_to_sw32"][narrow])
            self.assertEqual(set(receipt["without_transfer_us"]), set(joint["estimated_us"]))

    def test_resident_grid_crossing_changes_query_geometry(self):
        for rows, heads, kv_heads in ((760, 30, 2), (725, 29, 1)):
            rank = rank_d64_pipelines(rows, heads, False, kv_heads=kv_heads)
            v = rank["grouped_tiles"]["v_transport"]["resident"]
            self.assertEqual(rank["selected"], "q128-kv128-k-tma-post-qk-v-tma")
            self.assertEqual(min(v["before_us"], key=v["before_us"].get), "q64-kv128-k-tma-post-qk-v-tma")
            self.assertEqual(v["width_weight"], 0.)

    def test_extra_padded_work_at_unequal_residency(self):
        rank = rank_d64_pipelines(1053, 33, False, kv_heads=1)
        r = rank["grouped_tiles"]["v_transport"]["resident"]
        self.assertTrue(r["extra_half_box"])
        self.assertGreater(r["extra_waves"], 0)
        self.assertGreater(r["width_weight"], 0.)
        self.assertEqual(rank["selected"], "q64-kv64-k-tma-post-qk-v-tma")

    def test_equal_wave_width_tie_retains_incumbent(self):
        rank = rank_d64_pipelines(678, 25, False, kv_heads=1)
        receipt = rank["grouped_tiles"]["v_transport"]["resident"]
        self.assertEqual(receipt["extra_waves"], 0)
        self.assertTrue(receipt["retained_width_in_tie"])
        self.assertEqual(rank["selected"], "q64-kv128-k-tma-post-qk-v-tma")

    def test_singleton_drain_uses_the_fraction_of_affected_sms(self):
        from d64_static_pipeline import _singleton_drain
        self.assertEqual(_singleton_drain(3*114,114,3),0)
        self.assertEqual(_singleton_drain(4*114,114,3),1)
        self.assertEqual(_singleton_drain(3*114+57,114,3),0.5)
        self.assertEqual(_singleton_drain(4*114,114,5),0)

    def test_uniform_loop_uses_the_busiest_sm_for_a_width_override(self):
        for rows,heads,group in ((385,12,4),(959,14,2),(2193,14,2)):
            pair=rank_d64_pipelines(rows,heads,False,kv_heads=group)["grouped_width"]
            self.assertEqual(pair["log_ratio"],pair["prior_log_ratio"])
    def test_invalid_profile(self):
        with self.assertRaises(ValueError):
            rank_d64_pipelines(1024, 8, False, sm_count=42)
        with self.assertRaises(ValueError):
            rank_d64_pipelines(0, 8, False)

    def test_shared_issue_time_for_equal_ctas(self):
        from d64_static_pipeline import _resident_service
        # Six equal CTAs, two SMs and three resident CTAs per SM.
        wall, resident = _resident_service(128, 3, False, 64, 3, 2, 1.0)
        self.assertEqual((wall, resident), (9.0, 3.0))
        wall, resident = _resident_service(128, 3, False, 64, 3, 2, 0.5)
        self.assertEqual((wall, resident), (6.0, 3.0))

    def test_fixed_path_pair_choices(self):
        for rows, heads, causal, expected in (
            (1348, 6, False, 'q64-kv128-k-tma-post-qk-v-tma'),
            (3649, 10, True, 'q64-kv128-k-tma'),
            (3072, 12, True, 'q64-kv64-k-tma'),
        ):
            with self.subTest(rows=rows, causal=causal):
                self.assertEqual(rank_d64_pipelines(rows, heads, causal, causal_v_tma=False)['selected'], expected)

    def test_pair_update_leaves_unpaired_candidates_alone(self):
        from d64_static_pipeline import _score_kv_pairs
        costs = {'q64-kv64-k-tma': 2.5, 'cp-q64-wait4': 3.5}
        expected = dict(costs)
        _score_kv_pairs(costs, 129, 3, False, 114)
        self.assertEqual(costs, expected)

    def test_width_comparison_preserves_incumbent_family_score(self):
        from d64_static_pipeline import _score_kv_pairs
        small, large = ('q64-kv64-k-tma', 'q64-kv128-k-tma')
        other = 'cp-q64-producer-wait4-late-v'
        costs = {small: 100.0, large: 104.0, other: 101.0}
        _score_kv_pairs(costs, 3649, 10, True, 114)
        self.assertEqual(costs[large], 100.0)
        self.assertEqual(costs[other], 101.0)
        self.assertGreater(costs[small], costs[large])
