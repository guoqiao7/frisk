"""Full-holder proofs and fail-closed constraints for a GF(2) store carrier."""
from copy import deepcopy
from unittest import TestCase

from access_plan import plan_access
from redistribution import validate
from shuffle_network import plan_shuffle_network, simulate_network
from warp_store_layout import plan_warp_store


def carrier(logical):
    first = [0] * len(logical)
    for holder, slot in enumerate(logical):
        first[slot] = holder
    return dict(type='tensor<64x16xf32>', distributed=dict(shape=[64, 16],
                topology=[8, 32, 4, 1, 1], replication=1, logical=logical, first=first))


def mma_source():
    logical = []
    for thread in range(128):
        lane = thread % 32
        for register in range(8):
            row = thread // 32 * 16 + lane // 4 + (register >> 1 & 1) * 8
            column = (lane % 4) * 2 + (register & 1) + (register >> 2) * 8
            logical.append(row * 16 + column)
    return carrier(logical)


def row_major_offsets():
    return {slot: 4 * slot for slot in range(1024)}


class WarpStoreLayoutTests(TestCase):
    def test_full_domain_native_mma_carrier(self):
        src = mma_source()
        offsets = row_major_offsets()
        dst, operation, swap = plan_warp_store(src, offsets, 16)
        self.assertEqual(swap, (2, 0))
        plan = validate(operation, {0: src, 1: dst})
        network = plan_shuffle_network(plan)
        self.assertEqual(network['verified_holders'], 1024)
        self.assertEqual(network['counts']['shfl'], 4)
        logical = dst['distributed']['logical']
        self.assertEqual(simulate_network(network, src['distributed']['logical']), logical)
        tables = [[offsets[logical[thread * 8 + reg]] for thread in range(128)]
                  for reg in range(8)]
        access = plan_access(tables, element_bytes=4, root_alignment=16,
                             root_bytes=4096, owner_masks=[(1 << 128) - 1] * 8)
        self.assertEqual([group.width_bytes for group in access.groups], [16, 16])

    def test_tampered_source_owner_is_rejected(self):
        src = mma_source()
        src['distributed']['first'][0] = 1
        with self.assertRaisesRegex(ValueError, 'canonical source owners'):
            plan_warp_store(src, row_major_offsets(), 16)

    def test_tampered_conversion_is_rejected(self):
        src = mma_source()
        dst, operation, _ = plan_warp_store(src, row_major_offsets(), 16)
        plan = operation['redistribution']
        plan['destinations'][0]['source']['lane'] ^= 1
        with self.assertRaisesRegex(ValueError, 'destination coordinate'):
            validate(operation, {0: src, 1: dst})

    def test_non_affine_bijection_is_rejected(self):
        logical = mma_source()['distributed']['logical']
        logical[1000], logical[1001] = logical[1001], logical[1000]
        with self.assertRaisesRegex(ValueError, 'affine source'):
            plan_warp_store(carrier(logical), row_major_offsets(), 16)

    def test_address_and_replication_contracts(self):
        src = mma_source()
        offsets = row_major_offsets()
        for bad_offsets, alignment in ((offsets, 8), ({slot: v + 4 for slot, v in offsets.items()}, 16)):
            with self.subTest(alignment=alignment), self.assertRaisesRegex(ValueError, 'aligned dense FP32'):
                plan_warp_store(src, bad_offsets, alignment)
        changed = deepcopy(src)
        changed['distributed']['replication'] = 2
        with self.assertRaisesRegex(ValueError, 'bijective holder'):
            plan_warp_store(changed, offsets, 16)

    def test_two_lane_or_warp_directions_are_rejected(self):
        offsets = row_major_offsets()
        # Direct row-major is already vector-local. A 7-bit rotate places both
        # low address bits outside the register file. Neither needs/is solved by
        # the one-lane swap. Moving address bit1 to a warp bit is also rejected.
        for logical in (list(range(1024)),
                        [((holder << 7) | (holder >> 3)) & 1023 for holder in range(1024)],
                        [(holder ^ 258) if bool(holder & 2) != bool(holder & 256) else holder
                         for holder in range(1024)]):
            with self.assertRaisesRegex(ValueError, 'one movable lane direction'):
                plan_warp_store(carrier(logical), offsets, 16)
