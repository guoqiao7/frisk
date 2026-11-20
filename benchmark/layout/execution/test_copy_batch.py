"""Check Shared publication boundaries for optional synchronous Copy16 batching."""

import unittest

from execution_schedule import synchronous_copy_batches


def copy16(source, target):
    return {
        'name': 'frisk.copy',
        'operands': [source, target],
        'attrs': {
            'frisk.copy_contract': {
                'path': 'vector', 'completion': 'synchronous', 'vector_bytes': 16,
                'guarded': 0, 'issuer': 'execution-map',
            },
            'frisk.writer_policy': 'first_owner',
        },
    }


def storage(root, begin=0, end=128):
    return {'storage': {'root': root, 'addresses': [{'begin_bit': begin, 'end_bit': end}]}}


class CopyBatchTest(unittest.TestCase):
    def setUp(self):
        self.values = {i: storage(i) for i in range(6)}
        self.allocation = {
            'global_roots': [0, 1, 2],
            'allocations': [
                {'root_id': 3, 'offset': 0, 'bytes': 16},
                {'root_id': 4, 'offset': 16, 'bytes': 16},
                {'root_id': 5, 'offset': 32, 'bytes': 16},
            ],
        }
        self.copies = [copy16(0, 3), copy16(1, 4), copy16(2, 5)]

    def batches(self, ops=None):
        return synchronous_copy_batches(ops if ops is not None else self.copies,
                                        self.values, self.allocation)

    def test_adjacent_disjoint_endpoints_share_publication(self):
        self.assertEqual(self.batches(), [{
            'operations': [0, 1, 2], 'shared_bit_ranges': [[0, 128], [128, 256], [256, 384]],
        }])

    def test_existing_publish_boundary_and_read_are_not_crossed(self):
        self.allocation['reuse_barriers'] = [1]
        self.assertEqual(self.batches(), [{
            'operations': [1, 2], 'shared_bit_ranges': [[128, 256], [256, 384]],
        }])
        del self.allocation['reuse_barriers']
        read = {'name': 'frisk.mma', 'operands': [3]}
        self.assertEqual(self.batches(self.copies[:1] + [read] + self.copies[1:2]), [])

    def test_distinct_roots_with_overlapping_allocations_are_not_batched(self):
        self.allocation['allocations'][1]['offset'] = 0
        self.assertEqual(self.batches(self.copies[:2]), [])

    def test_unproven_width_and_guard_end_the_batch(self):
        for field, value in [('vector_bytes', 1), ('guarded', 1), ('path', 'cp.async')]:
            with self.subTest(field=field):
                other = copy16(1, 4)
                other['attrs']['frisk.copy_contract'][field] = value
                self.assertEqual(self.batches([self.copies[0], other]), [])

    def test_out_of_allocation_rejected(self):
        self.values[3] = storage(3, 0, 256)
        with self.assertRaisesRegex(ValueError, 'destination exceeds'):
            self.batches()
