import itertools
import unittest

from valid_domain import (PrefixDomain, dense_copy_execution,
                          plan_guarded_copy_owners, plan_guarded_store)


_SHAPE = (64, 64)
_SHARED_OFFSETS = tuple((row * 128 + col * 2) ^ ((row & 7) << 4)
                        for row in range(64) for col in range(64))


def copy_plan(valid_rows, *, policy='gf2', **overrides):
    options = dict(valid_shape=(valid_rows, 64), source_row_stride_bytes=128,
                   source_span_bytes=valid_rows * 128, destination_offsets=_SHARED_OFFSETS,
                   destination_span_bytes=8192, owner_policy=policy)
    options.update(overrides)
    return plan_guarded_copy_owners(dense_copy_execution(_SHAPE), **options)


def native_accumulator():
    logical = []
    for thread in range(128):
        for reg in range(32):
            row = (thread // 32) * 16 + (thread % 32) // 4 + (reg % 4 // 2) * 8
            col = (thread % 4) * 2 + reg % 2 + (reg // 4) * 8
            logical.append(row * 64 + col)
    first = [0] * len(logical)
    for owner, slot in enumerate(logical):
        first[slot] = owner
    return dict(shape=[64, 64], topology=[32, 32, 4, 1, 1],
                replication=1, logical=logical, first=first)


class PrefixDomainTests(unittest.TestCase):
    def test_dyadic_pullback_covers_exact_mask_for_an_affine_replicated_map(self):
        # Mix logical axes, affine offset, and a null hardware direction.
        columns, base = (5, 9, 17, 0, 3), 18
        table = [base]
        for column in columns:
            table += [point ^ column for point in table]
        for rows, cols in itertools.product(range(5), range(9)):
            with self.subTest(rows=rows, cols=cols):
                domain = PrefixDomain((4, 8), (rows, cols))
                expected = {i for i, slot in enumerate(table)
                            if slot // 8 < rows and slot % 8 < cols}
                blocks = domain.affine_pullback(base, columns)
                self.assertEqual(sum(block.size for block in blocks), len(expected))
                self.assertEqual(domain.activity_mask(table), sum(1 << i for i in expected))

    def test_invalid_shapes_and_non_affine_tables_fail_closed(self):
        for carrier, valid in (((3, 8), (3, 5)), ((4, 8), (4, 9)), ((4, 8), (True, 4))):
            with self.subTest(carrier=carrier, valid=valid):
                with self.assertRaises(ValueError):
                    PrefixDomain(carrier, valid)
        with self.assertRaisesRegex(ValueError, 'GF.2'):
            PrefixDomain((4, 8), (3, 5)).activity_mask([0, 1, 2, 5])


class GuardedCopyTests(unittest.TestCase):
    def test_full_one_row_and_other_prefixes_keep_exact_pairs_and_zero_coverage(self):
        expected_dst = {_SHARED_OFFSETS[slot] for slot in range(0, 4096, 8)}
        for policy, rows in itertools.product(('execution', 'gf2'), (0, 1, 13, 63, 64)):
            with self.subTest(policy=policy, rows=rows):
                plan = copy_plan(rows, policy=policy)
                read, initialized = set(), set()
                self.assertEqual(len(plan.groups), 4)
                for group in plan.groups:
                    for lane, (row, col, src, dst) in enumerate(zip(
                            group.rows, group.columns, group.source_offsets,
                            group.destination_offsets)):
                        self.assertEqual(src, 128 * row + col * 2)
                        self.assertEqual(dst, _SHARED_OFFSETS[64 * row + col])
                        self.assertNotIn(dst, initialized)
                        initialized.add(dst)
                        active = bool(group.active_mask >> lane & 1)
                        self.assertEqual(active, row < rows)
                        if active:
                            read.add(src)
                self.assertEqual(initialized, expected_dst)
                self.assertEqual(read, set(range(0, rows * 128, 16)))

    def test_column_aligned_tail_is_supported_and_partial_vector_rejected(self):
        plan = copy_plan(3, valid_shape=(3, 24), source_row_stride_bytes=128,
                         source_span_bytes=2 * 128 + 48)
        active = {offset for group in plan.groups for lane, offset in enumerate(group.source_offsets)
                  if group.active_mask >> lane & 1}
        self.assertEqual(active, {row * 128 + column for row in range(3) for column in (0, 16, 32)})
        with self.assertRaisesRegex(ValueError, 'complete unique vector'):
            copy_plan(3, valid_shape=(3, 23))

    def test_corrupt_endpoint_group_and_span_fail_closed(self):
        bad_shared = list(_SHARED_OFFSETS)
        bad_shared[:8] = bad_shared[8:16]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            copy_plan(1, destination_offsets=bad_shared)
        with self.assertRaisesRegex(ValueError, 'Global byte span'):
            copy_plan(1, source_span_bytes=127)
        with self.assertRaisesRegex(ValueError, 'Shared'):
            copy_plan(1, destination_span_bytes=8191)
        with self.assertRaisesRegex(ValueError, 'GF.2'):
            copy_plan(3, source_row_stride_bytes=144, source_span_bytes=512)


class GuardedStoreTests(unittest.TestCase):
    def test_native_fragment_stores_only_the_valid_prefix(self):
        for rows, cols in ((1, 64), (13, 63), (63, 64), (64, 64)):
            with self.subTest(rows=rows, cols=cols):
                plan = plan_guarded_store(native_accumulator(), valid_shape=(rows, cols),
                                          destination_row_stride_bytes=256,
                                          destination_span_bytes=(rows - 1) * 256 + cols * 4)
                output = []
                for group in plan.groups:
                    self.assertEqual(group.owner_mask.bit_count(), 128)
                    for lane, (row, col, offset) in enumerate(zip(group.rows, group.columns,
                                                                   group.destination_offsets)):
                        if group.active_mask >> lane & 1:
                            self.assertEqual(offset, row * 256 + col * 4)
                            output.append(offset)
                expected = {row * 256 + col * 4 for row in range(rows) for col in range(cols)}
                self.assertEqual(len(output), len(expected))
                self.assertEqual(set(output), expected)

    def test_replicated_carrier_uses_first_owners(self):
        execution = native_accumulator()
        execution['topology'][0] *= 2
        # A new low register bit replicates each logical slot.
        execution['logical'] = [slot for slot in execution['logical'] for _ in range(2)]
        execution['first'] = [owner * 2 for owner in execution['first']]
        execution['replication'] = 2
        plan = plan_guarded_store(execution, valid_shape=(1, 64),
                                  destination_row_stride_bytes=256, destination_span_bytes=256)
        self.assertEqual(sum(group.active_mask.bit_count() for group in plan.groups), 64)
        self.assertTrue(all(group.owner_mask == 0 for group in plan.groups[1::2]))
        execution['first'][0] += 1
        with self.assertRaisesRegex(ValueError, 'first-owner'):
            plan_guarded_store(execution, valid_shape=(1, 64),
                               destination_row_stride_bytes=256, destination_span_bytes=256)


if __name__ == '__main__':
    unittest.main()
