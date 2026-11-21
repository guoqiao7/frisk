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
        for key, value in [('rows',128), ('kv_heads',2), ('causal',2), ('heads',0)]:
            old = attrs[key]
            attrs[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                _geometry(data)
            attrs[key] = old


if __name__ == '__main__':
    unittest.main()
