"""Regression checks for the tied Attention Copy16 layout axis."""

from copy import deepcopy
from unittest import TestCase, mock

from attention_copy_layout import (attention_copy_layout_evidence, derive_attention_copy_layouts,
                                   native_copy_plan)
from attention_tile_mma import shared_offsets_64x64
from valid_domain import dense_copy_execution


def copy_packet():
    return dict(values=[], operations=[
        *(dict(name='frisk.copy', operands=[i, i + 3], execution=dense_copy_execution((64, 64)))
          for i in range(3)),
        dict(name='frisk.mma', operands=[3, 4], attrs={'strict': True}),
    ])


def checked_execution(packet):
    # This fixture isolates ownership derivation from the unrelated WGMMA
    # graph certificate; native_copy_plan below independently proves ownership.
    return packet['operations'][0]['execution'], None


class AttentionCopyLayoutTests(TestCase):
    def test_derived_owners_are_tied_and_leave_the_source_untouched(self):
        source = copy_packet()
        original = deepcopy(source)
        with mock.patch('tiled_attention._checked_tile_layouts', side_effect=checked_execution):
            variants = derive_attention_copy_layouts(source)
        self.assertEqual(source, original)
        self.assertEqual(tuple(variants), ('coalesced', 'warp-interleaved'))
        for variant in variants.values():
            self.assertEqual(variant['operations'][-1], source['operations'][-1])
            maps = [op['execution'] for op in variant['operations'][:-1]]
            self.assertTrue(all(mapping == maps[0] for mapping in maps))
            self.assertIsNot(maps[0], maps[1])
        baseline = native_copy_plan(checked_execution(source)[0], 64, owner_policy='gf2')
        coalesced = native_copy_plan(checked_execution(variants['coalesced'])[0], 64, owner_policy='execution')
        self.assertEqual(coalesced.groups, baseline.groups)
        self.assertNotEqual(checked_execution(variants['coalesced'])[0],
                            checked_execution(variants['warp-interleaved'])[0])

    def test_every_valid_prefix_preserves_the_same_copy16_byte_pairs(self):
        shared = shared_offsets_64x64()
        with mock.patch('tiled_attention._checked_tile_layouts', side_effect=checked_execution):
            variants = derive_attention_copy_layouts(copy_packet())
        for name, packet in variants.items():
            for rows in range(1, 65):
                with self.subTest(layout=name, rows=rows):
                    plan = native_copy_plan(checked_execution(packet)[0], rows, owner_policy='execution')
                    active = {(src, dst) for group in plan.groups for tid, (src, dst) in
                              enumerate(zip(group.source_offsets, group.destination_offsets, strict=True))
                              if (group.active_mask >> tid) & 1}
                    self.assertEqual(active, {(slot * 2, shared[slot]) for slot in range(0, rows * 64, 8)})
                    self.assertEqual(sum(group.active_mask.bit_count() for group in plan.groups), rows * 8)

    def test_full_cost_ties_but_compact_publication_layout_changes(self):
        with mock.patch('tiled_attention._checked_tile_layouts', side_effect=checked_execution):
            variants = derive_attention_copy_layouts(copy_packet())
            reports = [attention_copy_layout_evidence(packet) for packet in variants.values()]
            tails = [attention_copy_layout_evidence(packet, valid_rows=16, storage_rows=16)
                     for packet in variants.values()]
        for report in reports:
            self.assertEqual(report['global_sectors32'], [256, 272])
            self.assertEqual(report['shared_excess_wavefronts'], 0)
            self.assertTrue(report['carry_free_translates'])
        self.assertNotEqual(reports[0]['gf2_columns'], reports[1]['gf2_columns'])
        self.assertEqual([[group['active_vectors'] for group in tail['groups']] for tail in tails],
                         [[128, 0, 0, 0], [64, 64, 0, 0]])

    def test_noncanonical_source_and_corrupt_owner_map_fail_closed(self):
        source = copy_packet()
        with mock.patch('tiled_attention._checked_tile_layouts', side_effect=checked_execution):
            derived = derive_attention_copy_layouts(source)['warp-interleaved']
            with self.assertRaisesRegex(ValueError, 'original dense C\\+\\+'):
                derive_attention_copy_layouts(derived)
            derived['operations'][0]['execution']['first'][0] = 2
            with self.assertRaisesRegex(ValueError, 'first-owner'):
                attention_copy_layout_evidence(derived)

