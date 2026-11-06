import struct
import unittest

import numpy as np

from qualify_attention_tail import _check_ptx, compare_neutral
from tail_attention_reference import EXACT, check_output, geometry, inputs, ordered_sum, references


def specification(n):
    return dict(B=2,H=2,N=n,D=64,q_block=64,kv_block=64)


def rn32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def balanced(values):
    if len(values)==1:
        return values[0]
    midpoint = len(values)//2
    return rn32(balanced(values[:midpoint])+balanced(values[midpoint:]))


class TailAttentionReferenceTests(unittest.TestCase):
    def test_compact_non_power_of_two_query_grid(self):
        self.assertEqual(geometry(specification(512)), ((2,2,512,64),[32,1,1]))
        self.assertEqual(geometry(specification(513)), ((2,2,513,64),[36,1,1]))
        with self.assertRaises(ValueError):
            geometry(specification(514))

    def test_sum_matches_independent_recursive_fp32_tree(self):
        rng = np.random.default_rng(4)
        for n in (512,513):
            with self.subTest(n=n):
                data = np.exp2(rng.integers(-24,1,n).astype(np.float32))
                lanes = []
                for lane in range(4):
                    values = [float(v) for i,v in enumerate(data) if i//2%4==lane]
                    width = 1 << (len(values)-1).bit_length()
                    lanes.append(balanced(values+[0.]*(width-len(values))))
                expected = rn32(rn32(lanes[0]+lanes[1])+rn32(lanes[2]+lanes[3]))
                self.assertEqual(ordered_sum(data)[0], expected)

    def test_analytic_fixtures_observe_the_last_key_and_query(self):
        for n in (512,513):
            for name in EXACT:
                with self.subTest(n=n,name=name):
                    values = inputs(specification(n),name)
                    expected = references(values,name)
                    facts = check_output(expected['strict'],expected,name)
                    self.assertTrue(facts['analytical_bitwise'])
                    if name=='last-key-dominant':
                        self.assertFalse(np.array_equal(expected['strict'][...,0,:],
                                                        expected['strict'][...,-1,:]))

    def test_neutral_extension_isolation(self):
        cases = [dict(case=str(n),spec=specification(n)) for n in (512,513)]
        outputs = {row['case']:references(inputs(row['spec'],'neutral-extension'),
                                          'neutral-extension')['strict'] for row in cases}
        self.assertEqual(compare_neutral(cases,outputs)[0]['status'],'bitwise')
        outputs['513'][0,0,0,0] += 1
        with self.assertRaisesRegex(ValueError,'zero-weight KV extension'):
            compare_neutral(cases,outputs)

    def test_unreachable_scratch_required_by_manifest(self):
        ptx = ('.version 8.7\n.target sm_90a\n'
               '.visible .entry attention(.param .u64 q, .param .u64 k, .param .u64 v,'
               ' .param .u64 o, .param .u64 scratch)\n.reqntid 128,1,1\n{ ret; }\n')
        self.assertEqual(_check_ptx(ptx,'attention'),['q','k','v','o','scratch'])
        with self.assertRaisesRegex(ValueError,'scratch ABI slot'):
            _check_ptx(ptx.replace('ret;', 'ld.param.u64 %rd1, [scratch]; ret;'),'attention')
        with self.assertRaisesRegex(ValueError,'no approximate FP32 division'):
            _check_ptx(ptx.replace('ret;', 'div.approx.f32 a,b,c; ret;'),'attention')


if __name__=='__main__':
    unittest.main()
