from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from qualify_attention_tail import _check_ptx, compare_neutral, load_manifest
from tail_attention_reference import EXACT, check_output, geometry, inputs, ordered_sum, references


_PTX = ('.version 8.7\n.target sm_90a\n'
        '.visible .entry attention(.param .u64 q, .param .u64 k, .param .u64 v,'
        ' .param .u64 o, .param .u64 scratch)\n.reqntid 128,1,1\n{ ret; }\n')


def specification(n, *, heads=2):
    return dict(B=2,H=heads,N=n,D=64,q_block=64,kv_block=64)


def inventory(directory, lengths, *, heads=2):
    image = dict(block=[128,1,1], alignment_bytes=16, entry='attention', dynamic_shared_bytes=0)
    for kind, data in (('cubin', b'\x7fELF'), ('ptx', _PTX.encode())):
        path = directory/f'attention.{kind}'
        path.write_bytes(data)
        image.update({kind:str(path), kind+'_sha256':sha256(data).hexdigest()})
    cases = []
    for n in lengths:
        spec = specification(n, heads=heads)
        cases.append(dict(case=f'n{n}', spec=spec, image=dict(image, grid=geometry(spec)[1])))
    manifest = directory/'programs.json'
    manifest.write_text(json.dumps(dict(cases=cases)))
    return manifest


def rn32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def balanced(values):
    if len(values)==1:
        return values[0]
    midpoint = len(values)//2
    return rn32(balanced(values[:midpoint])+balanced(values[midpoint:]))


class TailAttentionReferenceTests(unittest.TestCase):
    def test_compact_non_power_of_two_query_grid(self):
        for n, ctas in ((64,4), (65,8), (128,8), (129,12), (191,12), (192,12),
                        (193,16), (512,32), (513,36), (1025,68)):
            self.assertEqual(geometry(specification(n)), ((2,2,n,64),[ctas,1,1]))
        self.assertEqual(geometry(specification(513, heads=32)), ((2,32,513,64),[576,1,1]))
        for n in (63, 1026):
            with self.assertRaises(ValueError):
                geometry(specification(n))

    def test_ordinary_manifest_admits_singletons_and_expanded_sweeps(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for lengths in ((129,), (512,), (513,), (128,129,191,192,193,512,513)):
                manifest = inventory(directory, lengths)
                self.assertEqual([row['spec']['N'] for row in load_manifest(manifest)], list(lengths))

    def test_legacy_only_manifest_keeps_multihead_pair_requirement(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = inventory(directory, (512,513))
            self.assertEqual(len(load_manifest(manifest)), 2)
            manifest = inventory(directory, (512,513), heads=1)
            with self.assertRaisesRegex(ValueError, 'paired N512/N513 inventory'):
                load_manifest(manifest)
            self.assertEqual(len(load_manifest(manifest, allow_partial=True)), 2)
            self.assertEqual(len(load_manifest(manifest, 'n513')), 1)

    def test_sum_matches_independent_recursive_fp32_tree(self):
        rng = np.random.default_rng(4)
        for n in (64,65,128,129,191,192,193,512,513,1025):
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
        for n in (64,65,128,129,191,192,193,512,513,1025):
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
        lengths = (128,129,191,192,193,257,512,513)
        cases = [dict(case=str(n),spec=specification(n)) for n in lengths]
        outputs = {row['case']:references(inputs(row['spec'],'neutral-extension'),
                                          'neutral-extension')['strict'] for row in cases}
        checks = compare_neutral(cases,outputs)
        self.assertEqual([(x['base'],x['tail']) for x in checks],
                         [('128','129'), ('128','191'), ('192','193'), ('512','513')])
        self.assertTrue(all(x['status']=='bitwise' and x['bitwise'] for x in checks))
        self.assertTrue(checks[-1]['duplicate_query_512_matches_0'])
        original = outputs['513'][0,0,0,0]
        outputs['513'][0,0,0,0] += 1
        with self.assertRaisesRegex(ValueError,'zero-weight KV extension'):
            compare_neutral(cases,outputs)
        outputs['513'][0,0,0,0] = original
        outputs['191'][0,0,-2,0] += 1
        with self.assertRaisesRegex(ValueError,'duplicated query 0'):
            compare_neutral(cases,outputs)

    def test_unreachable_scratch_required_by_manifest(self):
        ptx = _PTX
        self.assertEqual(_check_ptx(ptx,'attention'),['q','k','v','o','scratch'])
        with self.assertRaisesRegex(ValueError,'scratch ABI slot'):
            _check_ptx(ptx.replace('ret;', 'ld.param.u64 %rd1, [scratch]; ret;'),'attention')
        with self.assertRaisesRegex(ValueError,'no approximate FP32 division'):
            _check_ptx(ptx.replace('ret;', 'div.approx.f32 a,b,c; ret;'),'attention')


if __name__=='__main__':
    unittest.main()
