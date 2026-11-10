import re
import unittest

from attention_single_score_bulk import _SingleScoreBulkAttention
from attention_tile_mma import ROOT_BYTES, build_d64_wgmma_tiles
from tiled_attention import _row_fibers


class BulkSingleScoreTests(unittest.TestCase):
    def probe(self, width):
        kernel = object.__new__(_SingleScoreBulkAttention)
        kernel.length = width
        kernel.mma_tiles = build_d64_wgmma_tiles(kv_tile=width)
        kernel.smem_base = 'smem'
        kernel.roots = ['q', 'k', 'v', 'out']
        kernel.row_index = _row_fibers()[1]
        kernel.code, kernel.decl, kernel.counter = [], [], 0
        copies = []
        kernel._copy = lambda root, row, shared, guard: copies.append((root, row, shared, guard))
        kernel._publish = lambda: kernel.emit('PUBLISHED')
        kernel._consume_values([f'exp{i}' for i in range(width // 2)], ['sum0', 'sum1'])
        return kernel, copies

    def test_rs_inputs_are_complete_before_one_ascending_pv_group(self):
        for width in (64, 128, 256):
            kernel, copies = self.probe(width)
            expected_roots = ((kernel.mma_tiles.v_offset,) if width == 64 else
                              tuple(kernel.mma_tiles.k_offset + i * ROOT_BYTES for i in range(width // 64)))
            self.assertEqual(copies, [('v', i * 64, root, False) for i, root in enumerate(expected_roots)])
            code = kernel.code
            self.assertEqual(code[0], 'bar.sync 0;')
            self.assertEqual(code.count('PUBLISHED'), 1)
            self.assertEqual(code.count('wgmma.commit_group.sync.aligned;'), 1)
            self.assertEqual(code.count('wgmma.wait_group.sync.aligned 0;'), 1)
            native = [i for i, line in enumerate(code) if line.startswith('wgmma.mma_async.')]
            self.assertEqual(len(native), width // 16)
            packs = [i for i, line in enumerate(code) if line.startswith('cvt.rn.f16x2.f32')]
            self.assertEqual(len(packs), width // 4)
            self.assertLess(max(packs), min(native))
            self.assertTrue(all(code[i].startswith('wgmma.mma_async.') for i in range(min(native), max(native) + 1)))
            groups = [re.search(r'}, ({.*}), t[0-9]+,', code[i]).group(1) for i in native]
            inputs = [name for group in groups for name in group.strip('{}').split(', ')]
            self.assertEqual(len(inputs), width // 4)
            self.assertEqual(len(set(inputs)), width // 4)

    def test_root_cover_is_exact_and_rejects_q_overlap(self):
        kernel, _ = self.probe(256)
        roots = kernel._value_roots()
        self.assertEqual(roots, (8192, 16384, 24576, 32768))
        self.assertEqual(roots[-1] + ROOT_BYTES, kernel.mma_tiles.shared_bytes)
        from dataclasses import replace
        kernel.mma_tiles = replace(kernel.mma_tiles, q_offset=roots[0])
        with self.assertRaises(ValueError):
            kernel._value_roots()


if __name__ == '__main__':
    unittest.main()

