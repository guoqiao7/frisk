import unittest
from unittest.mock import patch

from attention_pipeline_selection import attention_pipelines, select_attention_pipeline, tune_attention_pipelines
from native_fa3_execution import _geometry


class AttentionPipelineSelectionTest(unittest.TestCase):
    def test_half_output_works_without_an_import_and_deferral_requires_alignment(self):
        for aligned in (False,True):
            plans = attention_pipelines(16,native_delivery=True,aligned_inputs=aligned)
            self.assertTrue(all(p.half_output and p.writeback=="producer" for p in plans))
            self.assertEqual([p.defer_v_wait for p in plans], [False,True] if aligned else [False])

    def test_base2_softmax_and_row_reciprocal_are_explicit_in_ssa(self):
        from native_online_fixture import online_fixture
        from online_execution_contract import single_group
        options = dict(execution_contract=single_group(4,128),packed_copy=True)
        plain = online_fixture(64,64,193,2,2,True,**options)
        fast = online_fixture(64,64,193,2,2,True,base2_logits=True,row_reciprocal=True,**options)
        self.assertEqual(plain.count("math.exp "),2)
        self.assertNotIn("math.exp2",plain)
        self.assertEqual(fast.count("math.exp2 "),2)
        self.assertIn("%inv = arith.divf %ones, %lfinal",fast)
        self.assertIn("%out = arith.mulf",fast)

    def test_tune_skips_illegal_plans_and_returns_the_measured_image(self):
        def compile_one(plan, *args, **kwargs):
            if plan.query_tile == 256:
                raise ValueError('not admitted')
            return plan.name, {'wait': plan.contract['batch']}
        def measure(images):
            return [{name: 1 / meta['wait'] for name, (_, meta) in images.items()}]
        with patch('attention_pipeline_selection.compile_attention_pipeline', side_effect=compile_one):
            image, info, receipt = tune_attention_pipelines(
                'opt', '.', rows=256, heads=8, dim=64, causal=True,
                output_bits=32, measure=measure)
        self.assertEqual((image, info['wait']), ('q128-wait4', 4))
        self.assertEqual(len(receipt['rejected']), 2)

    def test_extended_causal_tuner_compares_cp_k_and_kv_tma(self):
        paths = {}
        def compile_one(plan, *args, **options):
            self.assertEqual((options["kv_heads"],options["kv_rows"],options["value_dim"]), (4,768,64))
            paths[plan.name] = (options["k_transport"],plan.v_transport)
            return plan.name, {}
        with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=compile_one):
            tune_attention_pipelines("opt",".",rows=768,heads=16,kv_heads=4,kv_rows=768,dim=64,
                value_dim=64,causal=True,aligned_inputs=True,output_bits=32,
                measure=lambda images: [{name: 1.0 for name in images}])
        self.assertEqual(set(paths.values()),{("cp.async","cp.async"),("tma","cp.async"),("tma","tma")})

    def test_short_causal_tuner_emits_the_two_slot_wide_plan(self):
        for causal, heads, kv_heads, rows, expected in (
                (True, 12, 12, 128, True), (True, 12, 12, 256, True),
                (True, 12, 12, 257, False), (True, 12, 3, 256, False), (False, 12, 12, 256, False)):
            plans = {}
            def capture(plan, *args, **options):
                plans[plan.name] = (options["k_transport"], plan.v_transport)
                return plan.name, {}
            with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=capture):
                tune_attention_pipelines("opt", ".", rows=rows, heads=heads, kv_heads=kv_heads, dim=64,
                    causal=causal, aligned_inputs=True, output_bits=32,
                    measure=lambda images: [{name: 1.0 for name in images}])
            self.assertEqual(plans.get("q64-kv128-k-tma-v-tma"), ("tma", "tma") if expected else None)

    def test_short_causal_partial_query_adds_a_row_major_candidate(self):
        for n,causal,kv_heads,expected in [(207, True, 12, True), (241, True, 12, True),
                                           (224, True, 12, True), (192, True, 12, False),
                                           (256, True, 12, False), (265, True, 12, False),
                                           (207, False, 12, False), (207, True, 3, False)]:
            captured = {}
            def record(plan, *args, **options):
                captured[plan.name] = (plan.v_row_major_tma, options["k_transport"], plan.v_transport)
                return plan.name, {}
            with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=record):
                tune_attention_pipelines("opt", ".", rows=n, heads=12, kv_heads=kv_heads, dim=64,
                    causal=causal, aligned_inputs=True, output_bits=32,
                    measure=lambda images: [{name: 1.0 for name in images}])
            name = "q64-kv128-k-tma-v-tma"
            self.assertEqual(captured.get(name+"-sw128"), (True,"tma","tma") if expected else None)
            if expected:
                self.assertEqual(captured[name], (False,"tma","tma"))

    def test_tuner_admits_short_narrow_sw128_ring_only_in_its_domain(self):
        for n, causal, kv_heads, expected in ((64,True,8,False),(65,True,8,True),
                                              (128,True,8,True),(129,True,8,True),
                                              (192,True,8,True),(149,False,8,False),
                                              (149,True,2,False)):
            plans = {}
            def capture(plan, *args, **opts):
                plans[plan.name] = (plan.kv_tile, plan.v_row_major_tma, opts["k_transport"], plan.v_transport)
                return plan.name, {}
            with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=capture):
                tune_attention_pipelines("opt", ".", rows=n, heads=8, kv_heads=kv_heads, dim=64,
                    causal=causal, aligned_inputs=True, output_bits=32,
                    measure=lambda images: [{name: 1.0 for name in images}])
            self.assertEqual(plans.get("q64-kv64-k-tma-v-tma-sw128"),
                             (64, True, "tma", "tma") if expected else None)

    def test_regular_grouped_d64_tuner_pairs_q_rows_on_joint_transfer(self):
        captured = {}
        def capture(plan, *args, **options):
            captured[plan.name] = (plan.query_tile, plan.kv_tile, plan.contract["cta"],
                                   options["k_transport"], plan.v_transport, plan.prefetch_after_qk)
            return plan.name, {}
        with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=capture):
            tune_attention_pipelines("opt", ".", rows=641, heads=12, kv_heads=4, dim=64,
                causal=False, aligned_inputs=True, output_bits=32,
                measure=lambda images: [{name: 1.0 for name in images}])
        self.assertEqual({v for n,v in captured.items() if n.startswith("q128-kv")},
                         {(128, width, 256, "tma", "tma", late)
                          for width in (64,128) for late in (False,True)})

    def test_default_regular_d64_tuner_includes_both_copy_paths_and_edges(self):
        for bits in (16, 32):
            arms = {}
            def compile_one(plan, *args, **options):
                self.assertEqual(plan.half_output, bits == 16)
                arms[plan.name] = (options["k_transport"], plan.v_transport, plan.prefetch_after_qk)
                return plan.name, {}
            with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=compile_one):
                tune_attention_pipelines("opt", ".", rows=641, heads=8, dim=64,
                    causal=False, aligned_inputs=True, output_bits=bits, fa3_query_tile="auto",
                    measure=lambda images: [{name: 1.0 for name in images}])
            self.assertIn(("cp.async", "cp.async", False), arms.values())
            for width in (64, 128):
                for suffix, edge in (("", False), ("-post-qk", True)):
                    name = f"q64-kv{width}-k-tma{suffix}-v-tma"
                    self.assertEqual(arms[name], ("tma", "tma", edge))

    def test_static_winner_is_one_of_the_tunable_physical_kernels(self):
        from attention_pipeline_selection import compile_static_d64_attention, compile_static_gqa_attention
        for n, h, g, d, causal, bits, physical in (
                (920, 15, 1, 64, False, 32, "q64-kv128-k-tma-post-qk-v-tma-sw128"),
                (760, 30, 2, 64, False, 32, "q128-kv128-k-tma-post-qk-v-tma-sw128"),
                (116, 54, 54, 64, True, 32, "q64-kv64-k-tma-v-tma-sw128"),
                (224, 19, 19, 64, True, 32, "q64-kv128-k-tma-v-tma-sw128"),
                (256, 8, 8, 64, False, 32, "q64-kv64-k-tma-post-qk-v-tma"),
                (64, 5, 5, 64, False, 16, "q64-producer-wait4-late-v"),
                (193, 8, 1, 128, True, 32, "q64-kv64-k-tma-v-tma")):
            with self.subTest(shape=(n, h, g, d, causal, bits)):
                args = dict(rows=n, heads=h, causal=causal, output_bits=bits, aligned_inputs=True,
                            row_reciprocal=True, softmax_math="base2-ftz")
                def capture(plan, *args, **options):
                    return (plan, options["k_transport"], options["k_map_abi"]), {"execution_audit": {}}
                with patch("attention_pipeline_selection.compile_attention_pipeline", side_effect=capture) as compile_one:
                    if h == g:
                        image, _, receipt = compile_static_d64_attention("opt", ".", **args)
                    else:
                        image, _, receipt = compile_static_gqa_attention("opt", ".", dim=d, kv_heads=g, **args)
                    compile_one.assert_called_once()
                    self.assertEqual(receipt["selected"], physical)
                    self.assertEqual(receipt["physical_by_family"][receipt["ranked_family"]], physical)
                    from d64_static_pipeline import rank_d64_pipelines, rank_grouped_attention
                    ranked = (rank_d64_pipelines(n, h, causal, output_bits=bits) if h == g else
                              rank_grouped_attention(n, h, g, d, causal, output_bits=bits))
                    self.assertEqual(ranked["selected"], physical)
                    def measure(images):
                        self.assertEqual(image, images[physical][0])
                        costs = receipt.get("estimated_us", receipt.get("estimated_ratio"))
                        self.assertIn(physical, costs)
                        self.assertTrue(costs.keys() <= images.keys())
                        return [{name: 1. if name == physical else 2. for name in images}]
                    tuned, _, winner = tune_attention_pipelines("opt", ".", dim=d, kv_heads=g,
                                                                 measure=measure, **args)
                    self.assertEqual((tuned, winner["selected"]), (image, physical))

    def test_grouped_swizzles_are_distinct_and_obey_explicit_restrictions(self):
        from attention_pipeline_candidates import pipeline_candidates
        args = dict(rows=920, heads=15, kv_heads=1, dim=64, causal=False,
                    output_bits=32, aligned_inputs=True)
        base = "q64-kv128-k-tma-post-qk-v-tma"
        candidates = {c.plan.name: c for c in pipeline_candidates(**args)}
        self.assertFalse(candidates[base].plan.v_row_major_tma)
        self.assertIsNone(candidates[base].ranked_family)
        self.assertTrue(candidates[base+"-sw128"].plan.v_row_major_tma)
        self.assertEqual(candidates[base+"-sw128"].ranked_family, base)
        for opts in (dict(include_v_tma=False), dict(k_transport="cp.async"),
                     dict(k_transport="tma", k_map_abi="global")):
            pool = pipeline_candidates(**args, **opts)
            self.assertFalse(any(c.plan.v_transport == "tma" for c in pool))

    def test_compared_plans_remain_bound_to_their_contract(self):
        plans = attention_pipelines(32)[:2]
        a, b = (p.name for p in plans)
        selected, _ = select_attention_pipeline(plans, [{a: 5, b: 4}, {a: 3, b: 4}, {a: 7, b: 4}])
        self.assertIs(selected, plans[1])
        self.assertEqual(selected.contract['batch'], 4)
        for rows in ([{a: 2}], [{a: 2, b: float('nan')}], []):
            with self.assertRaises(ValueError):
                select_attention_pipeline(plans, rows)

    def test_q64_uses_one_warpgroup_for_copies_and_mma(self):
        from native_online_fixture import online_fixture
        from online_execution_contract import validate
        plans = [p for p in attention_pipelines(32) if p.query_tile == 64]
        self.assertEqual([p.contract["batch"] for p in plans], [1, 4])
        for plan in plans:
            validate(plan.contract)
            self.assertEqual(plan.contract["cta"], 128)
            ir = online_fixture(64,64,129,4,4,True,execution_contract=plan.contract,
                                packed_copy=True)
            self.assertIn('threads=128:i64', ir)
            self.assertNotIn('threads=256:i64', ir)

    def test_causal_elision_declines_an_unrecognized_carry(self):
        from native_online_memory import identity_suffix
        self.assertFalse(identity_suffix([], {"body_arguments":[1,2]}, None, None, None))

    def test_fringe_admission_uses_residency_and_arithmetic_policy(self):
        from attention_pipeline_selection import AttentionPipeline, fringe_pipelines
        p=AttentionPipeline('fa3',128,{'kind':'fa3','arena':163840},'register')
        def policies(h,q,k,ca,allowed=True):
            return [v.fringe_policy for v in fringe_pipelines(p,q,k,h,128,ca,114,allowed)]
        self.assertEqual(policies(12,1153,1153,False),['short','split'])
        self.assertEqual(policies(11,1153,1153,False),['short'])
        self.assertEqual(policies(12,1153,1153,True,False),['reverse'])
        self.assertEqual(policies(12,1152,1153,True),[])
        self.assertEqual(policies(12,1153,1152,True),['reverse'])
        self.assertEqual(policies(12,1152,1152,False),[])

    def test_full_tile_geometry_and_causal_sign_extension(self):
        attrs = dict(rows=640, heads=8, kv_heads=8, causal=-1)
        data = {'operations':[{'name':'builtin.module','attrs':{'frisk.online_launch':attrs}}]}
        self.assertEqual(_geometry(data), (640, 8, True))
        attrs["rows"] = 641
        self.assertEqual(_geometry(data), (641,8,True))
        attrs["kv_heads"]=2
        self.assertEqual(_geometry(data), (641,8,True))
        for key, value in [('rows',128), ('kv_heads',3), ('causal',2), ('heads',0)]:
            old = attrs[key]
            attrs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _geometry(data)
            attrs[key] = old


if __name__ == '__main__':
    unittest.main()
