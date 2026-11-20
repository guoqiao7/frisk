"""Current static-PTX allocation domain, distinct from device capacity."""
import copy
import unittest
from native_static_support import (STATIC_SHARED_LIMIT_BYTES, StaticSharedUnsupported,
                                   check_static_shared)


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


if __name__=='__main__':unittest.main()
