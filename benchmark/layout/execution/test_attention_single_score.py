import struct
import unittest

import numpy as np

from attention_single_score import emit_single_score_attention, single_score_rows
from attention_tile_mma import THREADS, build_d64_wgmma_tiles, fragment_coordinate


def rounded(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def independent_tree(values):
    if len(values) == 1:
        return values[0]
    middle = len(values) // 2
    return rounded(independent_tree(values[:middle]) + independent_tree(values[middle:]))


class SingleScoreAttentionTests(unittest.TestCase):
    def test_full_native_carrier_projects_to_ordered_pv_rows(self):
        for width in (64, 128, 256):
            plan = build_d64_wgmma_tiles(kv_tile=width)
            rows = single_score_rows(width, tuple(plan.score_fragment_columns))
            self.assertEqual(sorted(rows[0] + rows[1]), list(range(width // 2)))
            for thread in range(THREADS):
                lane = thread % 4
                expected = [column for column in range(width) if column // 2 % 4 == lane]
                for slots in rows:
                    actual = [fragment_coordinate(thread, slot, kv_tile=width)[1] for slot in slots]
                    self.assertEqual(actual, expected)

    def test_sum_matches_independent_whole_row_association(self):
        rng = np.random.default_rng(48)
        for width in (64, 128, 256):
            plan = build_d64_wgmma_tiles(kv_tile=width)
            rows = single_score_rows(width, tuple(plan.score_fragment_columns))
            values = np.exp2(rng.integers(-24, 1, width).astype(np.float32))
            lane_sums = []
            for lane in range(4):
                columns = [fragment_coordinate(lane, slot, kv_tile=width)[1] for slot in rows[0]]
                lane_sums.append(independent_tree([float(values[column]) for column in columns]))
            got = independent_tree(lane_sums)
            expected_lanes = [independent_tree([float(values[column]) for column in range(width)
                                                if column // 2 % 4 == lane]) for lane in range(4)]
            self.assertEqual(got, independent_tree(expected_lanes))

    def test_wrong_native_map_and_unaligned_shapes_fail_closed(self):
        plan = build_d64_wgmma_tiles()
        changed = list(plan.score_fragment_columns)
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaises(ValueError):
            single_score_rows(64, tuple(changed))
        for width in (True, 32, 65, 127, 129, 257, 512):
            spec = dict(B=1, H=1, N=width, D=64, q_block=64, kv_block=64)
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    emit_single_score_attention(spec, tile_packet=None)


if __name__ == '__main__':
    unittest.main()

