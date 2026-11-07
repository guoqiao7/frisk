import re
import unittest

from attention_streaming_bulk import _StreamingBulkAttention
from attention_tile_mma import ROOT_BYTES, SHARED_ALIGNMENT, build_d64_wgmma_tiles
from tiled_attention import _row_fibers


class StreamingBulkTests(unittest.TestCase):
    def probe(self):
        kernel = object.__new__(_StreamingBulkAttention)
        kernel.length = 256
        kernel.kv_width = 128
        kernel.mma_tiles = build_d64_wgmma_tiles(kv_tile=128)
        kernel.decl = [f'.shared .align {SHARED_ALIGNMENT} .b8 frisk_shared[{kernel.mma_tiles.shared_bytes}];']
        kernel.code, kernel.counter = [], 0
        kernel._configure_streaming_storage()
        kernel.smem_base = 'smem'
        kernel.roots = ['q', 'k', 'v', 'out']
        kernel.zero = 'zero'
        kernel.row_index = _row_fibers()[1]
        kernel.rows = _row_fibers()[0]
        kernel.tail_prefix = 0
        kernel.residual_row = None
        copies = []
        kernel._copy = lambda root, row, shared, guard: copies.append((root, row, shared, guard))
        kernel._full_loop = lambda label, body: body(0, 0)
        return kernel, copies

    def test_disjoint_full_kv_publication_precedes_qk_and_complete_pv_group(self):
        kernel, copies = self.probe()
        kernel._value_pass(['max0', 'max1'], ['sum0', 'sum1'])
        self.assertEqual(kernel.mma_tiles.shared_bytes, 5 * ROOT_BYTES)
        self.assertEqual(kernel.mma_tiles.v_offset, 3 * ROOT_BYTES)
        self.assertEqual([copy[0] for copy in copies], ['k', 'v', 'k', 'v'])
        self.assertEqual([copy[2] for copy in copies], [ROOT_BYTES, 3 * ROOT_BYTES, 2 * ROOT_BYTES, 4 * ROOT_BYTES])
        code = kernel.code
        qk = [i for i, line in enumerate(code) if line.startswith('wgmma.mma_async.sync.aligned.m64n128')]
        pv = [i for i, line in enumerate(code) if line.startswith('wgmma.mma_async.sync.aligned.m64n64')]
        self.assertEqual(len(qk), 4)
        self.assertEqual(len(pv), 8)
        self.assertEqual(code.count('bar.sync 0;'), 2)
        self.assertEqual(code.count('wgmma.wait_group.sync.aligned 0;'), 2)
        packs = [i for i, line in enumerate(code) if line.startswith('cvt.rn.f16x2.f32')]
        self.assertEqual(len(packs), 32)
        self.assertGreater(min(packs), max(qk))
        self.assertLess(max(packs), min(pv))
        self.assertTrue(all(code[i].startswith('wgmma.mma_async.') for i in range(min(pv), max(pv) + 1)))
        groups = [re.search(r'}, ({.*}), t[0-9]+,', code[i]).group(1) for i in pv]
        sources = [reg for group in groups for reg in group.strip('{}').split(', ')]
        self.assertEqual(len(sources), 32)
        self.assertEqual(len(set(sources)), 32)

    def test_streaming_descriptors_cover_both_value_roots(self):
        kernel, _ = self.probe()
        offsets = [desc.offset for descriptors in kernel.bulk_value_descriptors for desc in descriptors]
        self.assertEqual(offsets, [3 * ROOT_BYTES + atom * 2048 for atom in range(4)]
                         + [4 * ROOT_BYTES + atom * 2048 for atom in range(4)])


if __name__ == '__main__':
    unittest.main()

