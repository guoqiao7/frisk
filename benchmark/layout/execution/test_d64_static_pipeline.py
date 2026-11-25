import unittest
from d64_static_pipeline import features, rank_d64_pipelines

def family_costs(receipt):
    return {family: receipt["estimated_us"][name]
            for family, name in receipt["physical_by_family"].items()}


class StaticD64Test(unittest.TestCase):
    def test_short_launch_only_has_legal_plans(self):
        result = rank_d64_pipelines(129, 3, False)
        self.assertTrue(result['estimated_us'])
        self.assertTrue(all(name.startswith('cp-') for name in result['physical_by_family']))

    def test_short_mha_transport_prices_the_tail_against_the_same_cp_plan(self):
        post = "q64-kv64-k-tma-post-qk-v-tma"
        late = "cp-q64-producer-wait4-late-v"
        aligned = rank_d64_pipelines(256, 12, False)
        tail = rank_d64_pipelines(255, 12, False)
        self.assertEqual(aligned["selected"], post)
        self.assertEqual(aligned["short_transport"]["v_tail_visit_fraction"], 0)
        self.assertEqual(tail["short_transport"]["v_tail_visit_fraction"], 1/4)
        self.assertGreater(tail["short_transport"]["post_over_late"],
                           aligned["short_transport"]["post_over_late"])
        scores = family_costs(aligned)
        self.assertAlmostEqual(scores[post]/scores[late], aligned["short_transport"]["post_over_late"])

    def test_short_mha_transport_requires_the_profiled_launch_contract(self):
        for n,h,causal,opts in [(128,12,False,{}), (512,12,False,{}), (256,256,False,{}),
                                (256,12,True,{}), (256,12,False,{"kv_heads":3}),
                                (256,12,False,{"output_bits":16})]:
            with self.subTest(rows=n,heads=h,causal=causal,opts=opts):
                self.assertNotIn("short_transport",rank_d64_pipelines(n,h,causal,**opts))

    def test_short_causal_width_charges_extra_fragments_across_the_grid(self):
        low = rank_d64_pipelines(256, 2, True)
        mid = rank_d64_pipelines(256, 12, True)
        high = rank_d64_pipelines(256, 32, True)
        self.assertEqual(low["selected"], "q64-kv128-k-tma-v-tma")
        self.assertEqual(mid["selected"], low["selected"])
        self.assertEqual(high["selected"], "q64-kv64-k-tma-v-tma")
        work = low["short_causal_width"]
        self.assertEqual(work["kv128_visits"], [1, 1, 2, 2])
        self.assertEqual(work["critical_visits_saved"], 2)
        self.assertAlmostEqual(work["extra_kv64_tiles_per_sm"], 4 / 114)
        one = rank_d64_pipelines(96, 4, True)
        self.assertEqual(one["short_causal_width"]["kv128_visits"], [1, 1])
        self.assertEqual(one["selected"], low["selected"]+"-sw128")

    def test_exposed_v_cost_is_charged_once_per_admitted_width(self):
        wide = "q64-kv128-k-tma-v-tma"
        narrow = "q64-kv64-k-tma-v-tma"
        def model(n, h):
            return rank_d64_pipelines(n, h, True)["short_causal_width"]
        few, more, both = model(160, 14), model(184, 30), model(224, 19)
        self.assertEqual(few["v_ready_saving_us"], more["v_ready_saving_us"])
        self.assertEqual(set(few["v_ready_saving_us"]), {wide})
        self.assertGreater(both["v_ready_saving_us"][wide], both["v_ready_saving_us"][narrow])
        self.assertAlmostEqual(model(96, 4)["v_ready_saving_us"][wide],
                               0.75 * few["v_ready_saving_us"][wide])
        self.assertEqual(few["charged_extra_kv64_tiles_per_sm"], 0)
        self.assertGreater(model(160, 48)["charged_extra_kv64_tiles_per_sm"], 0)
        aligned = model(256, 32)
        self.assertEqual(aligned["v_ready_saving_us"], {})
        self.assertEqual(aligned["charged_extra_kv64_tiles_per_sm"], aligned["extra_kv64_tiles_per_sm"])

    def test_partial_resident_wave_compares_only_the_admitted_sw128_widths(self):
        from d64_static_pipeline import _score_resident_causal_width
        small, wide = "q64-kv64-k-tma-v-tma", "q64-kv128-k-tma-v-tma"
        for n, h, expected in ((206, 31, small), (218, 33, small),
                               (236, 48, wide), (245, 56, wide)):
            costs = {small: 7.0, wide: 6.8, "cp.async": 8.0}
            stats = _score_resident_causal_width(costs, wide, small, n, h, 114)
            self.assertEqual(min(costs, key=costs.get), expected)
            self.assertEqual(costs[expected], 6.8)
            self.assertEqual(costs["cp.async"], 8.0)
            self.assertAlmostEqual(stats["wide_over_narrow"], costs[wide] / costs[small])
        for n, h in ((160, 48), (207, 28), (256, 32)):
            with self.subTest(n=n, h=h):
                self.assertNotIn("resident_width", rank_d64_pipelines(n, h, True)["short_causal_width"])

    def test_short_causal_prices_single_slot_against_prefetch(self):
        from online_execution_contract import packed_copy_supports_lookahead
        wide, copy = "q64-kv128-k-tma-v-tma", "cp-q64-producer-wait4-late-v"
        self.assertFalse(packed_copy_supports_lookahead(3))
        self.assertTrue(packed_copy_supports_lookahead(4))
        for n, h in ((174, 69), (157, 63)):
            rank = rank_d64_pipelines(n, h, True)
            model = rank["short_causal_width"]["single_slot_transport"]
            self.assertEqual((model["packed_versions"], model["tma_versions"]), (1, 2))
            self.assertAlmostEqual(model["wide_over_copy"],
                                   family_costs(rank)[wide] / family_costs(rank)[copy])
            self.assertLess(family_costs(rank)[wide], family_costs(rank)[copy])
        for n, h in ((160, 38), (120, 82), (192, 69), (206, 31), (174, 115)):
            rank = rank_d64_pipelines(n, h, True)
            self.assertNotIn("single_slot_transport", rank.get("short_causal_width", {}))

    def test_three_query_block_narrow_ring_uses_an_available_reference(self):
        narrow, wide = "q64-kv64-k-tma-v-tma", "q64-kv128-k-tma-v-tma"
        copy = "cp-q64-producer-wait4-late-v"
        for n, h, reference in ((129, 2, wide), (174, 69, wide), (157, 63, wide),
                                (192, 16, copy), (150, 115, copy)):
            rank = rank_d64_pipelines(n, h, True)
            price = rank["three_block_kv64"]
            self.assertEqual(price["reference"], reference)
            self.assertAlmostEqual(family_costs(rank)[narrow], price["narrow_over_reference"] *
                                   family_costs(rank)[reference])
        for n,h,c,kw in ((128,12,True,{}),(193,12,True,{}),(177,12,False,{}),
                         (177,12,True,{"kv_heads":3}),(177,12,True,{"output_bits":16}),
                         (177,12,True,{"causal_v_tma":False})):
            self.assertNotIn("three_block_kv64", rank_d64_pipelines(n,h,c,**kw))

    def test_two_query_block_width_prices_available_transport_paths(self):
        narrow, wide = "q64-kv64-k-tma-v-tma", "q64-kv128-k-tma-v-tma"
        copy = "cp-q64-producer-wait4-late-v"
        for n, h, refs in ((116, 54, {wide,copy}), (128, 12, {copy}), (88, 180, {copy})):
            rank = rank_d64_pipelines(n, h, True)
            ratios = rank["two_block_kv64"]["narrow_over_reference"]
            self.assertEqual(set(ratios), refs)
            costs = family_costs(rank)
            self.assertAlmostEqual(costs[narrow], min(costs[a] for a in refs) * max(ratios.values()))
        self.assertEqual(rank_d64_pipelines(116, 54, True)["selected"], narrow+"-sw128")
        self.assertEqual(rank_d64_pipelines(120, 12, True)["selected"], wide+"-sw128")
        for n,c,kw in ((64,True,{}),(129,True,{}),(116,False,{}),(116,True,{"kv_heads":3}),
                       (116,True,{"output_bits":16}),(116,True,{"causal_v_tma":False})):
            self.assertNotIn("two_block_kv64", rank_d64_pipelines(n,12,c,**kw))

    def test_short_causal_width_requires_the_native_transport_contract(self):
        for n,h,c,opts in [(7,4,True,{}), (257,12,True,{}), (256,120,True,{}),
                           (256,12,False,{}), (256,12,True,{"kv_heads":3}),
                           (256,12,True,{"output_bits":16}), (256,12,True,{"causal_v_tma":False})]:
            with self.subTest(rows=n, heads=h, causal=c, opts=opts):
                self.assertNotIn("short_causal_width", rank_d64_pipelines(n,h,c,**opts))

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
        self.assertEqual(list(out['estimated_ratio']),["q64-producer-wait4-late-v"])
        self.assertNotIn("q64-kv128-k-tma-v-tma",rank_grouped_attention(384,16,4,128,False)['estimated_ratio'])
        with self.assertRaises(ValueError):
            rank_grouped_attention(512,16,3,128,True)

    def test_grouped_width_compares_one_fixed_transport_family(self):
        for rows, heads, kv_heads, causal, width in ((1729,20,4,True,128),(3520,8,1,False,64)):
            result=rank_d64_pipelines(rows,heads,causal,kv_heads=kv_heads)
            comparison=result["grouped_width"]
            expected_at_q64 = comparison[f"kv{width}"]
            self.assertEqual(min((comparison["kv64"], comparison["kv128"]),
                                 key=family_costs(result).get), expected_at_q64)
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
        self.assertEqual(rank["selected"], wide+"-sw128")
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
            self.assertEqual(result["selected"], "q64-kv128-k-tma-post-qk-v-tma-sw128")
            self.assertEqual(joint["light_grid"]["weight"], 1)
            self.assertEqual(min(joint["light_grid"]["service_us"],
                                 key=joint["light_grid"]["service_us"].get), winner)
            self.assertEqual(joint["estimated_us"][winner], family_costs(result)[winner])
        full_grid = rank_d64_pipelines(2336, 18, False, kv_heads=6)
        self.assertNotIn("light_grid", full_grid["grouped_tiles"])

    def test_shared_kv_reads_change_query_choice_at_fixed_grid(self):
        import math
        sparse = rank_d64_pipelines(552, 22, False, kv_heads=11)
        shared = rank_d64_pipelines(552, 22, False, kv_heads=2)
        self.assertTrue(sparse["selected"].startswith("q64-"))
        prior = shared["grouped_tiles"]["v_transport"]["sw32_estimated_us"]
        self.assertTrue(min(prior, key=prior.get).startswith("q128-"))
        self.assertEqual(shared["selected"], "q64-kv128-k-tma-post-qk-v-tma-sw128")
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
        self.assertEqual(receipt["physical_by_family"][pair["q128"]], receipt["selected"])
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
            self.assertEqual(ranked["selected"], ranked["physical_by_family"][pair["q64"]])
            self.assertIn("kv128-k-tma-post-qk-v-tma", ranked["selected"])

    def test_row_major_v_cost_is_geometry_specific(self):
        for rows, heads, kv_heads in ((696, 15, 1), (552, 22, 2)):
            rank = rank_d64_pipelines(rows, heads, False, kv_heads=kv_heads)
            joint = rank["grouped_tiles"]
            receipt = joint["v_transport"]
            self.assertEqual((receipt["swizzle_bytes"], receipt["tensor_map_rank"], receipt["kv_slots"]),
                             (128, 3, 2))
            self.assertEqual(rank["selected"], "q64-kv128-k-tma-post-qk-v-tma-sw128")
            narrow = "q128-kv64-k-tma-post-qk-v-tma"
            self.assertLess(receipt["time_ratio_to_sw32"][rank["ranked_family"]],
                            receipt["time_ratio_to_sw32"][narrow])
            self.assertEqual(set(receipt["without_transfer_us"]), set(joint["estimated_us"]))

    def test_resident_grid_crossing_changes_query_geometry(self):
        for rows, heads, kv_heads in ((760, 30, 2), (725, 29, 1)):
            rank = rank_d64_pipelines(rows, heads, False, kv_heads=kv_heads)
            v = rank["grouped_tiles"]["v_transport"]["resident"]
            self.assertEqual(rank["selected"], "q128-kv128-k-tma-post-qk-v-tma-sw128")
            self.assertEqual(min(v["before_us"], key=v["before_us"].get), "q64-kv128-k-tma-post-qk-v-tma")
            self.assertEqual(v["width_weight"], 0.)

    def test_extra_padded_work_at_unequal_residency(self):
        rank = rank_d64_pipelines(1053, 33, False, kv_heads=1)
        r = rank["grouped_tiles"]["v_transport"]["resident"]
        self.assertTrue(r["extra_half_box"])
        self.assertGreater(r["extra_waves"], 0)
        self.assertGreater(r["width_weight"], 0.)
        self.assertEqual(rank["selected"], "q64-kv64-k-tma-post-qk-v-tma-sw128")

    def test_equal_wave_width_tie_retains_incumbent(self):
        rank = rank_d64_pipelines(678, 25, False, kv_heads=1)
        receipt = rank["grouped_tiles"]["v_transport"]["resident"]
        self.assertEqual(receipt["extra_waves"], 0)
        self.assertTrue(receipt["retained_width_in_tie"])
        self.assertEqual(rank["selected"], "q64-kv128-k-tma-post-qk-v-tma-sw128")

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
