"""Current static-PTX allocation domain, distinct from device capacity."""
import copy
import unittest
from native_static_support import (STATIC_SHARED_LIMIT_BYTES, StaticSharedUnsupported,
                                   _SM90_TTGIR_HEADER, check_sm90_residency_binary,
                                   check_static_shared, target_sm90_residency)


class NativeStaticSupportTest(unittest.TestCase):
    def test_supported_zero_normal_and_exact_boundary_are_nonmutating(self):
        for size in (0, 32768, 49152):
            info={'static_shared_bytes':size,'shared_limit_bytes':227328,
                  'diagnostics':{'logical':list(range(4))}}
            original=copy.deepcopy(info)
            proof=check_static_shared(info)
            self.assertEqual(info,original)
            self.assertEqual(proof['static_shared_bytes'],size)
            self.assertEqual(proof['static_shared_limit_bytes'],49152)
            self.assertTrue(proof['supported'])
            self.assertEqual(proof['allocation_kind'],'static-ptx')

    def test_total_shared_capacity_does_not_bypass_static_backend_bound(self):
        info={'static_shared_bytes':65536,'shared_limit_bytes':227328}
        with self.assertRaises(StaticSharedUnsupported) as caught:check_static_shared(info)
        self.assertEqual(caught.exception.static_shared_bytes,65536)
        self.assertEqual(caught.exception.static_shared_limit_bytes,STATIC_SHARED_LIMIT_BYTES)
        self.assertIn('current static PTX backend unsupported',str(caught.exception))
        self.assertEqual(info['shared_limit_bytes'],227328)
        with self.assertRaises(StaticSharedUnsupported):check_static_shared({'static_shared_bytes':49153})

    def test_invalid_or_missing_allocation_never_defaults_to_zero(self):
        for info in (None, [], {}, {'physical_shared_bytes':0},
                     {'static_shared_bytes':True}, {'static_shared_bytes':False},
                     {'static_shared_bytes':-1}, {'static_shared_bytes':0.0},
                     {'static_shared_bytes':'0'}, {'static_shared_bytes':None}):
            with self.subTest(info=info):
                with self.assertRaises(ValueError) as caught:check_static_shared(info)
                self.assertNotIsInstance(caught.exception,StaticSharedUnsupported)


class Sm90ResidencyTargetTest(unittest.TestCase):
    def setUp(self):
        self.ttgir = _SM90_TTGIR_HEADER + '} {\n  checked body\n}\n'
        self.info = dict(static_shared_bytes=36864, physical_shared_bytes=36864)

    def test_three_ctas_round_to_168_registers_without_modifying_the_body(self):
        text, target = target_sm90_residency(self.ttgir, self.info, 3)
        self.assertEqual(target['max_registers_per_thread'], 168)
        self.assertEqual(target['threads_per_block'], 128)
        self.assertEqual(text.replace(', "ttg.maxnreg" = 168 : i32', ''), self.ttgir)
        self.assertEqual(self.info, dict(static_shared_bytes=36864, physical_shared_bytes=36864))
        for ttgir, blocks in ((text, 3), (self.ttgir.replace('cuda:90', 'cuda:80'), 3),
                             (self.ttgir, 1), (self.ttgir, True)):
            with self.assertRaises(ValueError):
                target_sm90_residency(ttgir, self.info, blocks)

    def test_ignores_no_failed_native_resource_contract(self):
        _, target = target_sm90_residency(self.ttgir, self.info, 3)
        ptx = '.reqntid 128\n.maxnreg 168\n'
        resources = dict(registers_per_thread=168, stack_bytes_per_thread=0, local_bytes=0)
        ops = {'HMMA.16816.F32': 16}
        self.assertEqual(check_sm90_residency_binary(target, ptx, resources, ops)
                         ['native_spill_instructions'], 0)
        for kwargs in (
            dict(ptx='.reqntid 128\n.maxnreg 176\n'),
            dict(resources=dict(resources, registers_per_thread=169)),
            dict(resources=dict(resources, stack_bytes_per_thread=4)),
            dict(resources=dict(resources, local_bytes=4)),
            dict(mnemonics={'LDL.64': 1, **ops}),
        ):
            with self.assertRaises(ValueError):
                check_sm90_residency_binary(target, **(dict(ptx=ptx, resources=resources, mnemonics=ops) | kwargs))


if __name__=='__main__':unittest.main()
