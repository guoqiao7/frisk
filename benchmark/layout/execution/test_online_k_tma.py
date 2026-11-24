import unittest
from types import SimpleNamespace

from online_k_tma import configure, initialize, issue


class TensorMapContractTest(unittest.TestCase):
    def setUp(self):
        self.source = [2*i for i in range(4096)]
        self.target = [2*i ^ ((i//64 & 7) << 4) for i in range(4096)]
        self.emitter = SimpleNamespace(
            k_map_abi="global", kv_tile=64,
            qtile=64, dim=64, value_dim=64, kv_rows=209, heads=3, group=1,
            ring_stride=16384, roots=range(4), pitches=[209]*4,
            kcopy={'operands': ['K', 'shared']}, storage=lambda value: {'root': value},
            offsets=lambda value: self.source if value=='K' else self.target,
        )
        self.allocation = dict(native_kv_versions={'shared':[8192,24576]},
                               physical_shared_bytes=40960, shared_limit_bytes=65536)

    def test_row_major_v_checks_every_swizzled_offset_and_uses_three_coordinates(self):
        from online_k_tma import configure_v, issue_v
        cols, tile, rows = 64, 128, 919
        offsets = [2*i ^ (((i // cols) & 7) << 4) for i in range(tile*cols)]
        source = [2*i for i in range(tile*cols)]
        e = SimpleNamespace(k_tma={'abi': 'value'}, kv_rows=rows, kv_tile=tile,
                            pitches=[rows]*4, vcopy={'operands':['src','dst']},
                            storage=lambda v: {'root':v}, offsets=lambda v: source if v=='src' else offsets,
                            pv_op={'attrs':{'frisk.mma_contract':{'b_descriptor':{'major':'mn','swizzle':128}}}},
                            value_dim=cols, heads=30, group=15, roots=[0,1,2,3])
        alloc = {'native_kv_versions':{'dst':[24576,57344]}}
        e.v_tma = configure_v(e,alloc)
        self.assertEqual(e.v_tma['dimensions'], [64,919,2])
        self.assertEqual(e.v_tma['strides'], [128,919*128])
        self.assertNotIn('tail_rows',e.v_tma)
        code = []
        e.tmp, e.emit = lambda: 'dst', code.append
        e.k_issuer, e.krow, e.kv_head, e.k_barriers = 'issuer','row','head',['b0','b1']
        issue_v(e,1)
        self.assertIn('tensor.3d',code[-1])
        self.assertIn('{0, row, head}',code[-1])
        self.assertIn('[b1]',code[-1])
        offsets[-1] ^= 32
        with self.assertRaisesRegex(ValueError,'solved row-major'):
            configure_v(e,alloc)

    def test_grouped_d128_requires_two_certified_sw128_panels(self):
        import itertools
        e=self.emitter
        e.dim=e.value_dim=128
        e.group,e.heads,e.k_map_abi=4,12,"value"
        self.source=[2*i for i in range(64*128)]
        self.target=[(col//64*8192 + row*128 + 2*(col%64)) ^ ((row&7)*16)
                     for row in range(64) for col in range(128)]
        configure(e,self.allocation)
        self.assertEqual(e.k_tma["dimensions"],[128,209,3])
        self.assertEqual(e.k_tma["transaction_bytes"],16384)
        code=[];counts=itertools.count()
        e.tmp=lambda *args: "r"+str(next(counts))
        e.emit=code.append;e.krow,e.kv_head="row","head"
        initialize(e);issue(e,0)
        tx=[line for line in code if "cp.async.bulk.tensor.3d" in line]
        self.assertEqual(len(tx),2)
        self.assertIn("{0, row, head}",tx[0])
        self.assertIn("{64, row, head}",tx[1])
        self.target[64]^=16
        with self.assertRaisesRegex(ValueError,"SW128"):
            configure(e,self.allocation)

    def test_boxes_and_barriers(self):
        plan = configure(self.emitter, self.allocation)
        self.assertEqual(plan['physical_shared_bytes'], 40976)
        self.assertEqual(self.emitter.k_tma['dimensions'], [64,209,3])
        self.assertEqual(self.emitter.k_tma['barriers'], [40960,40968])

    def test_double_height_tile_charges_full_transaction(self):
        self.emitter.kv_tile = 128
        self.source = [2*i for i in range(8192)]
        self.target = [2*i ^ ((i//64 & 7)<<4) for i in range(8192)]
        self.allocation['native_kv_versions']['shared'] = [8192, 40960]
        self.allocation['physical_shared_bytes'] = 73728
        self.allocation['shared_limit_bytes'] = 98304
        configure(self.emitter,self.allocation)
        self.assertEqual(self.emitter.k_tma['box'], [64,128,1])
        self.assertEqual(self.emitter.k_tma['transaction_bytes'], 16384)

    def test_mismatched_element(self):
        self.target[-1] ^= 16
        with self.assertRaisesRegex(ValueError, 'SW128'):
            configure(self.emitter, self.allocation)

    def test_shifted_alternate_slot(self):
        self.allocation['native_kv_versions']['shared'][1] += 128
        with self.assertRaisesRegex(ValueError, 'SW128'):
            configure(self.emitter, self.allocation)

    def test_barriers_need_capacity(self):
        self.allocation['shared_limit_bytes'] = 40968
        with self.assertRaisesRegex(ValueError, 'insufficient'):
            configure(self.emitter, self.allocation)

    def test_unsupported_ring(self):
        self.emitter.ring_stride = 0
        with self.assertRaisesRegex(ValueError, 'two-slot'):
            configure(self.emitter, self.allocation)


    def test_value_map_has_completion_barriers_without_a_proxy_acquire(self):
        self.emitter.k_map_abi = 'value'
        configure(self.emitter, self.allocation)
        code = []
        regs = iter(['issuer','b0','b1','dst'])
        self.emitter.tmp, self.emitter.emit = lambda *args: next(regs), code.append
        self.emitter.krow, self.emitter.kv_head = 'row', 'head'
        initialize(self.emitter)
        issue(self.emitter, 0)
        self.assertFalse(any('tensormap::generic.acquire' in s for s in code))
        self.assertEqual(sum('mbarrier.init' in s for s in code), 2)
        self.assertTrue(any('mbarrier.arrive.expect_tx' in s for s in code))
        self.assertTrue(any('cp.async.bulk.tensor.3d' in s for s in code))

    def test_host_map_acquired_before_first_issue(self):
        configure(self.emitter, self.allocation)
        code = []
        registers = iter(["issuer", "bar0", "bar1", "destination"])
        self.emitter.tmp = lambda *args: next(registers)
        self.emitter.emit = code.append
        self.emitter.krow, self.emitter.kv_head = "row", "head"
        initialize(self.emitter)
        issue(self.emitter, 0)
        acquire, = [i for i, text in enumerate(code) if "tensormap::generic.acquire.sys" in text]
        transfer, = [i for i, text in enumerate(code) if "cp.async.bulk.tensor.3d" in text]
        self.assertLess(acquire, transfer)
        self.assertTrue(all(code[i].startswith("@issuer ") for i in (acquire, transfer)))


class ValueTensorMapContractTest(unittest.TestCase):
    def setUp(self):
        self.source = list(range(0, 8192, 2))
        self.target = []
        for row in range(64):
            for col in range(64):
                linear = (row//8 * 1024 + col//16 * 256 + row%8 * 32 + col%16 * 2)
                self.target.append(linear ^ ((linear >> 7 & 1) << 4))
        self.e = SimpleNamespace(k_tma={'abi':'value'}, kv_tile=64, kv_rows=72, pitches=[72]*4,
                                 pv_op={'attrs':{'frisk.mma_contract':{'b_descriptor':{'swizzle':32,'major':'mn'}}}},
                                 heads=4, group=1, roots=range(4), value_dim=64, vcopy={'operands':['V','S']},
                                 storage=lambda v: {'root':v},
                                 offsets=lambda v:self.source if v=='V' else self.target)
        self.allocation = {'native_kv_versions':{'S':[16384,32768]}}

    def test_partial_box_preserves_full_eight_row_groups(self):
        from online_k_tma import configure_v
        plan = configure_v(self.e, self.allocation)
        self.assertEqual(plan['dimensions'], [16,8,4,9,4])
        self.assertEqual(plan['strides'], [128,32,1024,9216])
        self.assertEqual(plan['box'], [16,8,4,8,1])
        self.assertEqual(plan['descriptor_arg'], 5)

    def test_floor_box_and_exact_head_pitch(self):
        from online_k_tma import configure_v
        for remainder in range(1,8):
            self.e.kv_rows = 64 + remainder
            self.e.pitches = [self.e.kv_rows]*4
            plan = configure_v(self.e, self.allocation)
            self.assertEqual(plan['tail_rows'], remainder)
            self.assertEqual(plan['dimensions'][3], 8)
            self.assertEqual(plan['strides'][-1], self.e.kv_rows*128)

    def test_rejects_wrong_swizzle(self):
        from online_k_tma import configure_v
        self.target[1025] ^= 16
        with self.assertRaisesRegex(ValueError, 'SW32'):
            configure_v(self.e, self.allocation)
