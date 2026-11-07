"""Regression checks for native Copy16 address factorization."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

from attention_copy_schedule import emit_factored_copy, prove_copy_schedule
from attention_tile_mma import score_width_for_prefix, shared_offsets_64x64
from valid_domain import dense_copy_execution, plan_guarded_copy_owners


def plan(prefix=64, policy='gf2'):
    return plan_guarded_copy_owners(
        dense_copy_execution((64,64)), valid_shape=(prefix,64),
        source_row_stride_bytes=128, source_span_bytes=prefix*128,
        destination_offsets=shared_offsets_64x64(), destination_span_bytes=8192,
        owner_policy=policy)


class FakeEmitter:
    def __init__(self, prefix=64, policy='gf2', simplify=False):
        self.copy_plans = {prefix:plan(prefix,policy)}
        self.simplify_tail_copy = simplify
        self.length = 513
        self.smem_base = '%shared'
        self.mma_tiles = SimpleNamespace(shared_bytes=24576)
        self.code, self.maps = [], []
        self.counter = 0

    def tmp(self, kind='b32'):
        self.counter += 1
        return f'%r{self.counter}'

    def emit(self, text):
        self.code.append(text)

    def _value(self, opcode, *args, kind='b32'):
        register = self.tmp(kind)
        self.emit(f"{opcode} {register}, {', '.join(map(str,args))};")
        return register

    def map(self, values):
        self.maps.append(tuple(values))
        return self.tmp()

    def _tile_pointer(self, root, tile_row, row_bytes):
        return self._value('test.tile', root, tile_row, row_bytes, kind='b64')


class CopyScheduleTests(unittest.TestCase):
    def test_all_prefixes_preserve_addresses_and_shifted_row_predicates(self):
        for prefix in range(1,65):
            proof = prove_copy_schedule(plan(prefix))
            self.assertIsNotNone(proof)
            self.assertEqual([(x.source_delta,x.destination_delta,x.row_delta)
                              for x in proof.translates],
                             [(0,0,0),(2048,2048,16),(4096,4096,32),(6144,6144,48)])
            for native, translation in zip(plan(prefix).groups, proof.translates, strict=True):
                for lane in range(128):
                    self.assertEqual(proof.source_base[lane]+translation.source_delta,
                                     native.source_offsets[lane])
                    self.assertEqual(proof.destination_base[lane]+translation.destination_delta,
                                     native.destination_offsets[lane])
                    self.assertEqual(proof.rows_base[lane]+translation.row_delta,native.rows[lane])
                    # This also covers the one real query in a runtime Q64 tail.
                    for tile_row,length in ((512,512+prefix),(0,513),(512,513)):
                        original = tile_row+native.rows[lane] < length
                        factored = tile_row+proof.rows_base[lane] < length-translation.row_delta
                        self.assertEqual(original,factored)
                    width = score_width_for_prefix(prefix)
                    if min(native.rows)<width:
                        self.assertEqual(native.rows[lane]<width,
                                         proof.rows_base[lane]<width-translation.row_delta)

    def test_copy_emission_factors_full_tile_and_keeps_exact_tail_guards(self):
        full = FakeEmitter()
        self.assertTrue(emit_factored_copy(full,'%root','%tile',8192,guard=False))
        self.assertEqual(len(full.maps),2)
        self.assertEqual(sum('ld.global.v4.b32' in x for x in full.code),4)
        self.assertTrue(any('+6144]' in x for x in full.code))

        tail = FakeEmitter(63,simplify=True)
        self.assertTrue(emit_factored_copy(tail,'%root',512,8192,guard=True,prefix=63))
        self.assertEqual(len(tail.maps),3)
        self.assertEqual(sum(x.startswith('mov.b32 ') for x in tail.code),4)
        self.assertEqual(sum('@' in x and 'ld.global' in x for x in tail.code),1)

        tiny = FakeEmitter(1,simplify=True)
        self.assertTrue(emit_factored_copy(tiny,'%root',512,8192,guard=True,prefix=1,storage_rows=8))
        self.assertEqual(sum('ld.global' in x for x in tiny.code),1)
        self.assertEqual(sum('st.shared' in x for x in tiny.code),1)
        self.assertEqual(sum('@' in x and 'st.shared' in x for x in tiny.code),1)

    def test_unsupported_or_corrupt_factorizations_fall_back_without_emitting(self):
        unsupported = FakeEmitter(policy='execution')
        self.assertFalse(emit_factored_copy(unsupported,'%root','%tile',8192,guard=False))
        self.assertFalse(unsupported.code)
        self.assertFalse(unsupported.maps)

        original = plan()
        group = replace(original.groups[1], destination_offsets=original.groups[0].destination_offsets)
        corrupt = replace(original,groups=(original.groups[0],group,*original.groups[2:]))
        self.assertIsNone(prove_copy_schedule(corrupt))
        wrong_mask = replace(original.groups[0],active_mask=1)
        self.assertIsNone(prove_copy_schedule(replace(original,groups=(wrong_mask,*original.groups[1:]))))


if __name__=='__main__':
    unittest.main()

